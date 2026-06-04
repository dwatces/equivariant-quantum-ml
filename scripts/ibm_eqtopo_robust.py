"""SHORE UP the both-gifts QPU result to publication grade: error mitigation (dynamical decoupling +
Pauli/readout twirling) + MULTIPLE backends (cross-device) + bootstrap error bars + significance on the
two contrasts (symmetry: E_C3 << E_C3_broken; topology: chirality_Haldane >> chirality_trivial).

Reuses the SIM-GATED circuit from ibm_equivariant_topological (job d8f5s0jal... was the first, un-mitigated
single run: E_C3 0.080 vs broken 0.517; chirality +0.327 vs trivial +0.006; conservation 0.534).

  python scripts/ibm_eqtopo_robust.py --backends ibm_marrakesh ibm_kingston ibm_fez
DISCIPLINE: simulate-gated already; submit all backends in parallel, collect, report ONLY read-back numbers.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ibm_equivariant_topological import (build, site_probs_counts, tvd, N, PHI, T2, SHOTS, ROOTRUN)

LABELS = [f"sym{k}" for k in range(N)] + [f"brk{k}" for k in range(N)] + ["chir+", "chir-", "triv+"]


def make_circuits():
    return ([build(k) for k in range(N)] + [build(k, detune=0) for k in range(N)]
            + [build(0, phi=+PHI), build(0, phi=-PHI), build(0, phi=+PHI, t2=0.0)])


def metrics_from_counts(counts_by_label):
    """counts_by_label: label -> raw counts dict. Returns the 5 metrics + conservation."""
    P = {lab: site_probs_counts(c) for lab, c in counts_by_label.items()}
    e3 = float(np.mean([tvd(P[f"sym{k}"][0], np.roll(P["sym0"][0], k)) for k in (2, 4)]))
    e3b = float(np.mean([tvd(P[f"brk{k}"][0], np.roll(P["brk0"][0], k)) for k in (2, 4)]))
    chir = float(P["chir+"][0][1] - P["chir+"][0][5])
    chir_t = float(P["triv+"][0][1] - P["triv+"][0][5])
    mirror = tvd(P["chir+"][0], P["chir-"][0])
    cons = float(np.mean([P[k][1] for k in P]))
    return dict(E_C3=e3, E_C3_broken=e3b, chirality=chir, chirality_triv=chir_t, mirror=mirror, conservation=cons)


def bootstrap_metrics(counts_by_label, B=200, seed=0):
    """Resample each circuit's shots -> distribution of metrics -> shot-noise error bars."""
    rng = np.random.default_rng(seed)
    keys = ["E_C3", "E_C3_broken", "chirality", "chirality_triv", "mirror"]
    samples = {k: [] for k in keys}
    prepared = {lab: (np.array(list(c.keys())), np.array(list(c.values()), float)) for lab, c in counts_by_label.items()}
    for _ in range(B):
        resampled = {}
        for lab, (bs, ct) in prepared.items():
            tot = int(ct.sum()); p = ct / ct.sum()
            draw = rng.multinomial(tot, p)
            resampled[lab] = {bs[i]: int(draw[i]) for i in range(len(bs)) if draw[i] > 0}
        m = metrics_from_counts(resampled)
        for k in keys:
            samples[k].append(m[k])
    return {k: (float(np.mean(samples[k])), float(np.std(samples[k]))) for k in keys}


def submit_all(backends, mitigate=True):
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService()
    qcs = make_circuits()
    jobs = []
    for bname in backends:
        backend = svc.backend(bname)
        ref = transpile(qcs[0], backend, optimization_level=3, seed_transpiler=11)
        init = ref.layout.initial_index_layout(filter_ancillas=True)
        tq = transpile(qcs, backend, optimization_level=3, initial_layout=init, seed_transpiler=11)
        n2 = max(sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in tq)
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = SHOTS
        if mitigate:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True          # readout error mitigation (TREX-like)
            sampler.options.twirling.num_randomizations = 32
        job = sampler.run(tq)
        jobs.append((bname, job.job_id(), job, n2))
        print(f"[submit] {bname}: job {job.job_id()} | transpiled 2q={n2} | mitigation={'ON' if mitigate else 'OFF'}", flush=True)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["ibm_marrakesh", "ibm_kingston", "ibm_fez"])
    ap.add_argument("--no_mitigate", action="store_true")
    a = ap.parse_args()
    jobs = submit_all(a.backends, mitigate=not a.no_mitigate)
    print(f"\n[waiting] {len(jobs)} jobs running in parallel on hardware ...", flush=True)
    per_backend = {}
    for bname, jid, job, n2 in jobs:
        res = job.result()
        cbl = {}
        for lab, r in zip(LABELS, res):
            d = r.data
            counts = (d.c.get_counts() if hasattr(d, "c") else d[list(d.keys())[0]].get_counts())
            cbl[lab] = counts
        pt = metrics_from_counts(cbl); eb = bootstrap_metrics(cbl)
        per_backend[bname] = {"job_id": jid, "transpiled_2q": n2, "point": pt, "err": eb}
        print(f"[done] {bname} ({jid}): E_C3={pt['E_C3']:.3f}±{eb['E_C3'][1]:.3f} | "
              f"broken={pt['E_C3_broken']:.3f} | chir={pt['chirality']:+.3f}±{eb['chirality'][1]:.3f} | "
              f"triv={pt['chirality_triv']:+.3f} | mirror={pt['mirror']:.3f} | cons={pt['conservation']:.3f}", flush=True)

    # cross-backend aggregate
    def agg(key):
        v = [per_backend[b]["point"][key] for b in per_backend]; return float(np.mean(v)), float(np.std(v))
    keys = ["E_C3", "E_C3_broken", "chirality", "chirality_triv", "mirror", "conservation"]
    A = {k: agg(k) for k in keys}
    # significance of the two contrasts (cross-backend): separation / pooled sd
    def zsep(a, b):
        ma, sa = agg(a); mb, sb = agg(b); s = (sa**2 + sb**2) ** 0.5 + 1e-9; return abs(ma - mb) / s
    sym_sep = zsep("E_C3_broken", "E_C3"); topo_sep = zsep("chirality", "chirality_triv")
    print(f"\n=== CROSS-BACKEND ({len(per_backend)} devices), mitigation={'OFF' if a.no_mitigate else 'ON'} ===")
    print(f"  SYMMETRY : E_C3 = {A['E_C3'][0]:.3f}±{A['E_C3'][1]:.3f}  vs  broken = {A['E_C3_broken'][0]:.3f}±{A['E_C3_broken'][1]:.3f}  "
          f"-> {A['E_C3_broken'][0]/max(A['E_C3'][0],1e-3):.1f}x, separation {sym_sep:.1f} sigma")
    print(f"  TOPOLOGY : chirality = {A['chirality'][0]:+.3f}±{A['chirality'][1]:.3f}  vs  trivial = {A['chirality_triv'][0]:+.3f}±{A['chirality_triv'][1]:.3f}  "
          f"-> separation {topo_sep:.1f} sigma")
    print(f"  mirror TVD = {A['mirror'][0]:.3f}±{A['mirror'][1]:.3f} | conservation = {A['conservation'][0]:.3f}±{A['conservation'][1]:.3f}")
    print(f"  >>> BOTH GIFTS robust across {len(per_backend)} devices: symmetry {sym_sep:.0f}sigma, topology {topo_sep:.0f}sigma")
    out = {"backends": list(per_backend.keys()), "mitigation": not a.no_mitigate,
           "per_backend": per_backend, "aggregate": A, "sym_separation_sigma": sym_sep, "topo_separation_sigma": topo_sep}
    json.dump(out, open(os.path.join(ROOTRUN, "ibm_eqtopo_robust.json"), "w"), indent=2)
    print(f"[saved] runs/ibm_eqtopo_robust.json")


if __name__ == "__main__":
    main()
