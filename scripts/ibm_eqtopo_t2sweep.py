"""Hardware t2 sweep — turns the causal-attribution result (S1) from simulation-only into a REAL-HARDWARE
one: does the measured chirality scale monotonically with the topological strength t2 on real qubits,
vanishing at t2=0? Reuses the validated 6-ring circuit. Mitigated, multi-device.

  python scripts/ibm_eqtopo_t2sweep.py --t2 0 0.1 0.2 0.3 0.4 --backends ibm_marrakesh ibm_fez
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from ibm_equivariant_topological import build, site_probs_counts, tvd, PHI, SHOTS, ROOTRUN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--backends", nargs="+", default=["ibm_marrakesh", "ibm_fez", "ibm_kingston"])
    ap.add_argument("--rand", type=int, default=16)
    a = ap.parse_args()
    labels = [f"{s}{t}" for t in a.t2 for s in ("p", "m")]    # chir+ , chir- at each t2
    qcs = [build(0, phi=(PHI if s == "p" else -PHI), t2=t) for t in a.t2 for s in ("p", "m")]
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(); jobs = []
    for bn in a.backends:
        be = svc.backend(bn)
        ref = transpile(qcs[0], be, optimization_level=3, seed_transpiler=11)
        init = ref.layout.initial_index_layout(filter_ancillas=True)
        tq = transpile(qcs, be, optimization_level=3, initial_layout=init, seed_transpiler=11)
        s = SamplerV2(mode=be); s.options.default_shots = SHOTS
        s.options.dynamical_decoupling.enable = True; s.options.dynamical_decoupling.sequence_type = "XY4"
        s.options.twirling.enable_gates = True; s.options.twirling.enable_measure = True
        s.options.twirling.num_randomizations = a.rand
        job = s.run(tq); jobs.append((bn, job.job_id(), job))
        print(f"[submit] {bn}: {job.job_id()}", flush=True)
    print(f"[waiting] t2 sweep on {len(jobs)} devices ...", flush=True)
    per = {}
    for bn, jid, job in jobs:
        res = job.result(); P = {}
        for lab, r in zip(labels, res):
            d = r.data; c = (d.c.get_counts() if hasattr(d, "c") else d[list(d.keys())[0]].get_counts())
            P[lab] = site_probs_counts(c)[0]
        per[bn] = {t: {"chirality": float(P[f"p{t}"][1] - P[f"p{t}"][5]),
                       "mirror": tvd(P[f"p{t}"], P[f"m{t}"])} for t in a.t2}
    print(f"\n=== HARDWARE t2 SWEEP (mean over {len(per)} devices) ===")
    print(f"{'t2':>6}{'chirality':>12}{'mirrorTVD':>12}")
    agg = {}
    for t in a.t2:
        ch = np.mean([per[b][t]["chirality"] for b in per]); ms = np.mean([per[b][t]["mirror"] for b in per])
        agg[t] = {"chirality": float(ch), "mirror": float(ms)}
        print(f"{t:>6.2f}{ch:>+12.4f}{ms:>12.4f}")
    mono = all(agg[a.t2[i]]["chirality"] <= agg[a.t2[i + 1]]["chirality"] + 0.02 for i in range(len(a.t2) - 1))
    print(f">>> chirality {'MONOTONIC in t2 (causal on hardware), ~0 at t2=0' if mono and abs(agg[a.t2[0]]['chirality'])<0.05 else 'not cleanly monotonic — report honestly'}")
    json.dump({"t2": a.t2, "per_backend": per, "aggregate": agg},
              open(os.path.join(ROOTRUN, "ibm_eqtopo_t2sweep.json"), "w"), indent=2)
    print("[saved] runs/ibm_eqtopo_t2sweep.json")


if __name__ == "__main__":
    main()
