"""S3 robust — 2-particle both-gifts circuit on 3 backends, mitigated, with bootstrap error bars.
Extends the coexistence into the entangled (many-body) regime. Reuses the 2-particle circuit/metrics
(ibm_eqtopo_2particle) + the mitigation/multi-device/bootstrap pattern (ibm_eqtopo_robust)."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ibm_eqtopo_2particle import (build2, pairdist_counts, roll_pairs, tvd_mat, equiv_c3_pairs,
                                  START_ORBIT, PHI, T2, SHOTS, ROOTRUN)

LABELS = ([f"sym{k}" for k in range(3)] + [f"brk{k}" for k in range(3)]
          + ["chir+", "chir-", "triv+", "triv-"])


def make_circuits():
    return ([build2(START_ORBIT[k]) for k in range(3)] + [build2(START_ORBIT[k], detune=0) for k in range(3)]
            + [build2(START_ORBIT[0], phi=+PHI), build2(START_ORBIT[0], phi=-PHI),
               build2(START_ORBIT[0], phi=+PHI, t2=0.0), build2(START_ORBIT[0], phi=-PHI, t2=0.0)])


def metrics(counts_by_label):
    G = {lab: pairdist_counts(c) for lab, c in counts_by_label.items()}
    e3 = equiv_c3_pairs([G[f"sym{k}"][0] for k in range(3)])
    e3b = equiv_c3_pairs([G[f"brk{k}"][0] for k in range(3)])
    chir = tvd_mat(G["chir+"][0], G["chir-"][0])
    chir_t = tvd_mat(G["triv+"][0], G["triv-"][0])
    cons = float(np.mean([G[k][1] for k in G]))
    return dict(E_C3=e3, E_C3_broken=e3b, chirality=chir, chirality_triv=chir_t, conservation=cons)


def bootstrap(counts_by_label, B=200, seed=0):
    rng = np.random.default_rng(seed); keys = ["E_C3", "E_C3_broken", "chirality", "chirality_triv"]
    prep = {lab: (np.array(list(c.keys())), np.array(list(c.values()), float)) for lab, c in counts_by_label.items()}
    samp = {k: [] for k in keys}
    for _ in range(B):
        rs = {}
        for lab, (bs, ct) in prep.items():
            draw = rng.multinomial(int(ct.sum()), ct / ct.sum())
            rs[lab] = {bs[i]: int(draw[i]) for i in range(len(bs)) if draw[i] > 0}
        m = metrics(rs)
        for k in keys:
            samp[k].append(m[k])
    return {k: (float(np.mean(samp[k])), float(np.std(samp[k]))) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["ibm_marrakesh", "ibm_kingston", "ibm_fez"])
    a = ap.parse_args()
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(); qcs = make_circuits()
    jobs = []
    for bn in a.backends:
        be = svc.backend(bn)
        ref = transpile(qcs[0], be, optimization_level=3, seed_transpiler=11)
        init = ref.layout.initial_index_layout(filter_ancillas=True)
        tq = transpile(qcs, be, optimization_level=3, initial_layout=init, seed_transpiler=11)
        n2 = max(sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in tq)
        s = SamplerV2(mode=be); s.options.default_shots = SHOTS
        s.options.dynamical_decoupling.enable = True; s.options.dynamical_decoupling.sequence_type = "XY4"
        s.options.twirling.enable_gates = True; s.options.twirling.enable_measure = True
        s.options.twirling.num_randomizations = 32
        job = s.run(tq); jobs.append((bn, job.job_id(), job, n2))
        print(f"[submit] {bn}: {job.job_id()} | 2q={n2} | mitig=ON", flush=True)
    print(f"\n[waiting] {len(jobs)} 2-particle jobs ...", flush=True)
    per = {}
    for bn, jid, job, n2 in jobs:
        res = job.result(); cbl = {}
        for lab, r in zip(LABELS, res):
            d = r.data; cbl[lab] = (d.c.get_counts() if hasattr(d, "c") else d[list(d.keys())[0]].get_counts())
        pt = metrics(cbl); eb = bootstrap(cbl)
        per[bn] = {"job_id": jid, "point": pt, "err": eb}
        print(f"[done] {bn}: E_C3={pt['E_C3']:.3f}±{eb['E_C3'][1]:.3f} brk={pt['E_C3_broken']:.3f} "
              f"chir={pt['chirality']:.3f}±{eb['chirality'][1]:.3f} triv={pt['chirality_triv']:.3f} cons={pt['conservation']:.3f}", flush=True)
    def agg(k): v = [per[b]["point"][k] for b in per]; return float(np.mean(v)), float(np.std(v))
    A = {k: agg(k) for k in ["E_C3", "E_C3_broken", "chirality", "chirality_triv", "conservation"]}
    def zsep(x, y): mx, sx = agg(x); my, sy = agg(y); return abs(mx - my) / ((sx**2 + sy**2) ** 0.5 + 1e-9)
    print(f"\n=== 2-PARTICLE CROSS-BACKEND ({len(per)} devices), mitigation ON ===")
    print(f"  SYMMETRY (2-body): E_C3 = {A['E_C3'][0]:.3f}±{A['E_C3'][1]:.3f} vs broken {A['E_C3_broken'][0]:.3f}±{A['E_C3_broken'][1]:.3f}  ({A['E_C3_broken'][0]/max(A['E_C3'][0],1e-3):.1f}x, {zsep('E_C3_broken','E_C3'):.0f}sigma)")
    print(f"  TOPOLOGY (2-body): chirality = {A['chirality'][0]:.3f}±{A['chirality'][1]:.3f} vs trivial {A['chirality_triv'][0]:.3f}±{A['chirality_triv'][1]:.3f}  ({zsep('chirality','chirality_triv'):.0f}sigma)")
    print(f"  conservation = {A['conservation'][0]:.3f}±{A['conservation'][1]:.3f}")
    json.dump({"per_backend": per, "aggregate": A, "sym_sigma": zsep('E_C3_broken', 'E_C3'),
               "topo_sigma": zsep('chirality', 'chirality_triv')},
              open(os.path.join(ROOTRUN, "ibm_eqtopo_2p_robust.json"), "w"), indent=2)
    print("[saved] runs/ibm_eqtopo_2p_robust.json")


if __name__ == "__main__":
    main()
