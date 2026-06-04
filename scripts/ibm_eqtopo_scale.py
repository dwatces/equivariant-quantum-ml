"""S4 — RING-SIZE SCALING of the both-gifts circuit (N=6 -> N=12). Shows the coexistence is not a
six-qubit accident. C3 requires N divisible by 6 (even for 2-colouring + divisible by 3 for the 120deg
rotation), so the clean scaling point is N=12 (12 qubits). Honest: depth ~doubles (48 logical 2q ->
~188 transpiled) so single-excitation conservation drops (~0.2-0.3); the contrasts (E_C3 vs broken,
chirality vs trivial) are the trustworthy reads. Mitigated, multi-device, bootstrap error bars.

  python scripts/ibm_eqtopo_scale.py --N 12 --backends ibm_marrakesh ibm_kingston ibm_fez
"""
import os, sys, math, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector
J, T2, PHI, DT, STEPS, MU, SHOTS = 1.0, 0.30, math.pi / 2, 0.6, 2, 3.0, 8192
ROOTRUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def build(N, start, phi=PHI, t2=T2, detune=None, measure=True):
    qc = QuantumCircuit(N, N); qc.x(start)
    LA = [(i, i + 1) for i in range(0, N, 2)]; LB = [(i, (i + 1) % N) for i in range(1, N, 2)]
    NE = [(i, (i + 2) % N) for i in range(0, N, 2)]; NO = [(i, (i + 2) % N) for i in range(1, N, 2)]
    for _ in range(STEPS):
        for (i, j) in LA: qc.append(XXPlusYYGate(2 * J * DT), [i, j])
        for (i, j) in LB: qc.append(XXPlusYYGate(2 * J * DT), [i, j])
        if t2 != 0.0:
            for (i, j) in NE: qc.append(XXPlusYYGate(2 * t2 * DT, phi), [i, j])
            for (i, j) in NO: qc.append(XXPlusYYGate(2 * t2 * DT, phi), [i, j])
        if detune is not None: qc.rz(2 * MU * DT, detune)
    if measure: qc.measure(range(N), range(N))
    return qc


def probs_counts(counts, N):
    s = sum(counts.values()); p = np.zeros(N); cons = 0
    for b, c in counts.items():
        o = [N - 1 - k for k, ch in enumerate(b.replace(" ", "")) if ch == "1"]
        if len(o) == 1: p[o[0]] += c; cons += c
    return p / max(1, cons), cons / s


def tvd(a, b): return 0.5 * float(np.abs(a - b).sum())


def metrics(cbl, N):
    orb = [0, N // 3, 2 * N // 3]
    P = {lab: probs_counts(c, N) for lab, c in cbl.items()}
    e3 = float(np.mean([tvd(P[f"sym{k}"][0], np.roll(P["sym0"][0], orb[k])) for k in (1, 2)]))
    e3b = float(np.mean([tvd(P[f"brk{k}"][0], np.roll(P["brk0"][0], orb[k])) for k in (1, 2)]))
    chir = float(P["chir+"][0][1] - P["chir+"][0][N - 1])
    chir_t = float(P["triv+"][0][1] - P["triv+"][0][N - 1])
    mirror = tvd(P["chir+"][0], P["chir-"][0])
    cons = float(np.mean([P[k][1] for k in P]))
    return dict(E_C3=e3, E_C3_broken=e3b, chirality=chir, chirality_triv=chir_t, mirror=mirror, conservation=cons)


def bootstrap(cbl, N, B=200, seed=0):
    rng = np.random.default_rng(seed); keys = ["E_C3", "E_C3_broken", "chirality", "chirality_triv", "mirror"]
    prep = {lab: (np.array(list(c.keys())), np.array(list(c.values()), float)) for lab, c in cbl.items()}
    samp = {k: [] for k in keys}
    for _ in range(B):
        rs = {lab: {bs[i]: int(d[i]) for i in range(len(bs)) if d[i] > 0}
              for lab, (bs, ct) in prep.items() for d in [rng.multinomial(int(ct.sum()), ct / ct.sum())]}
        m = metrics(rs, N)
        for k in keys: samp[k].append(m[k])
    return {k: (float(np.mean(samp[k])), float(np.std(samp[k]))) for k in keys}


def make(N):
    orb = [0, N // 3, 2 * N // 3]
    labels = [f"sym{k}" for k in range(3)] + [f"brk{k}" for k in range(3)] + ["chir+", "chir-", "triv+"]
    qcs = ([build(N, orb[k]) for k in range(3)] + [build(N, orb[k], detune=0) for k in range(3)]
           + [build(N, 0, phi=+PHI), build(N, 0, phi=-PHI), build(N, 0, phi=+PHI, t2=0.0)])
    return labels, qcs


def sim_gate(N):
    labels, qcs = make(N)
    def Psv(qc):
        sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False)); p = np.zeros(N); c = 0
        for b, pr in sv.probabilities_dict().items():
            o = [N - 1 - k for k, ch in enumerate(b) if ch == "1"]
            if len(o) == 1: p[o[0]] += pr; c += pr
        return p / max(c, 1e-12)
    cbl = {lab: ({"x" * N: 8192}) for lab in labels}  # placeholder, recompute from sv below
    Pd = {lab: Psv(qc) for lab, qc in zip(labels, qcs)}
    orb = [0, N // 3, 2 * N // 3]
    e3 = np.mean([tvd(Pd[f"sym{k}"], np.roll(Pd["sym0"], orb[k])) for k in (1, 2)])
    e3b = np.mean([tvd(Pd[f"brk{k}"], np.roll(Pd["brk0"], orb[k])) for k in (1, 2)])
    chir = Pd["chir+"][1] - Pd["chir+"][N - 1]; chir_t = Pd["triv+"][1] - Pd["triv+"][N - 1]
    n2q = sum(1 for inst in qcs[0].data if inst.operation.num_qubits == 2)
    print(f"[sim-gate N={N}] E_C3={e3:.4f} broken={e3b:.3f} chir={chir:+.3f} triv={chir_t:+.3f} | logical 2q={n2q}")
    return (e3 < 0.06) and (e3b > 5 * e3) and (chir > 0.05) and (abs(chir_t) < 0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--backends", nargs="+", default=["ibm_marrakesh", "ibm_kingston", "ibm_fez"])
    ap.add_argument("--randomizations", type=int, default=16)
    a = ap.parse_args()
    if not sim_gate(a.N):
        print("[abort] sim-gate failed."); return
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(); labels, qcs = make(a.N)
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
        s.options.twirling.num_randomizations = a.randomizations
        job = s.run(tq); jobs.append((bn, job.job_id(), job, n2))
        print(f"[submit] {bn}: {job.job_id()} | transpiled 2q={n2} | mitig=ON({a.randomizations})", flush=True)
    print(f"\n[waiting] N={a.N} on {len(jobs)} devices ...", flush=True)
    per = {}
    for bn, jid, job, n2 in jobs:
        res = job.result(); cbl = {}
        for lab, r in zip(labels, res):
            d = r.data; cbl[lab] = (d.c.get_counts() if hasattr(d, "c") else d[list(d.keys())[0]].get_counts())
        pt = metrics(cbl, a.N); eb = bootstrap(cbl, a.N)
        per[bn] = {"job_id": jid, "transpiled_2q": n2, "point": pt, "err": eb}
        print(f"[done] {bn}: E_C3={pt['E_C3']:.3f}±{eb['E_C3'][1]:.3f} brk={pt['E_C3_broken']:.3f} "
              f"chir={pt['chirality']:+.3f}±{eb['chirality'][1]:.3f} triv={pt['chirality_triv']:+.3f} cons={pt['conservation']:.3f}", flush=True)
    def agg(k): v = [per[b]["point"][k] for b in per]; return float(np.mean(v)), float(np.std(v))
    A = {k: agg(k) for k in ["E_C3", "E_C3_broken", "chirality", "chirality_triv", "mirror", "conservation"]}
    print(f"\n=== N={a.N} CROSS-BACKEND ({len(per)} devices) ===")
    print(f"  SYMMETRY: E_C3 = {A['E_C3'][0]:.3f}±{A['E_C3'][1]:.3f} vs broken {A['E_C3_broken'][0]:.3f}±{A['E_C3_broken'][1]:.3f} ({A['E_C3_broken'][0]/max(A['E_C3'][0],1e-3):.1f}x)")
    print(f"  TOPOLOGY: chirality = {A['chirality'][0]:+.3f}±{A['chirality'][1]:.3f} vs trivial {A['chirality_triv'][0]:+.3f}±{A['chirality_triv'][1]:.3f}")
    print(f"  conservation = {A['conservation'][0]:.3f}±{A['conservation'][1]:.3f}")
    json.dump({"N": a.N, "per_backend": per, "aggregate": A},
              open(os.path.join(ROOTRUN, f"ibm_eqtopo_scale_N{a.N}.json"), "w"), indent=2)
    print(f"[saved] runs/ibm_eqtopo_scale_N{a.N}.json")


if __name__ == "__main__":
    main()
