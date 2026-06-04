"""BOTH GIFTS ON ONE QPU CIRCUIT — C3-equivariance (symmetry) + Haldane chirality (topology) together.

The honest QPU test of "the previous theory": classical sim showed topology gives no single-particle
computational advantage — but topology LIVES on real hardware (this project's 97% chirality mirror +
2-particle steering). The realizable test is whether the honeycomb substrate's TWO gifts COEXIST and
survive hardware noise in ONE circuit:
  * SYMMETRY  -> 6-site ring, NN hop via 2-coloured Trotter (commutes with the 120deg rotation S^2).
    Start the walker at 0 vs 2 vs 4 -> outputs must be C3-rotations: E_C3 small.
  * TOPOLOGY  -> Haldane complex NEXT-nearest hopping (chiral phase phi) on triangles (0,2,4)&(1,3,5):
    breaks time-reversal AND the ring's mirror symmetry -> the walker CIRCULATES (P[1]!=P[5]; flips
    under phi->-phi). Trivial ring (t2=0): mirror-symmetric, no chirality.

HONEST physics note (found via sim): genuine topology (Haldane NNN) is what makes chirality visible in
POPULATIONS (a uniform flux keeps the ring's mirror symmetry -> chirality hidden), but the NNN triangle
is an odd cycle, so its Trotter step is NOT exactly C3-invariant -> a SMALL E_C3 artifact (~0.024 in the
ideal circuit, ~30x below the explicitly-broken control). So this is APPROXIMATE C3 + STRONG chirality,
not exact-symmetry. Scope: SIGNATURE COEXISTENCE + hardware robustness, NOT computational advantage.

DISCIPLINE: SIMULATE-FIRST. Only --submit after you confirm; never report an unread hardware number.
"""
import os, sys, math, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector

N = 6
LAYER_A = [(0, 1), (2, 3), (4, 5)]
LAYER_B = [(1, 2), (3, 4), (5, 0)]
TRI_A = [(0, 2), (2, 4), (4, 0)]
TRI_B = [(1, 3), (3, 5), (5, 1)]
J, T2, PHI, DT, STEPS, SHOTS = 1.0, 0.30, math.pi / 2, 0.6, 2, 8192
MU = 3.0
C3_STARTS = [0, 2, 4]


def build(start, phi=PHI, t2=T2, detune=None, measure=True):
    qc = QuantumCircuit(N, N); qc.x(start)
    for _ in range(STEPS):
        for (i, j) in LAYER_A:
            qc.append(XXPlusYYGate(2 * J * DT), [i, j])
        for (i, j) in LAYER_B:
            qc.append(XXPlusYYGate(2 * J * DT), [i, j])
        if t2 != 0.0:
            for (i, j) in TRI_A:
                qc.append(XXPlusYYGate(2 * t2 * DT, phi), [i, j])
            for (i, j) in TRI_B:
                qc.append(XXPlusYYGate(2 * t2 * DT, phi), [i, j])
        if detune is not None:
            qc.rz(2 * MU * DT, detune)
    if measure:
        qc.measure(range(N), range(N))
    return qc


def site_probs_sv(qc):
    sv = Statevector.from_instruction(qc); p = np.zeros(N); cons = 0.0
    for b, pr in sv.probabilities_dict().items():
        ones = [N - 1 - k for k, ch in enumerate(b) if ch == "1"]
        if len(ones) == 1:
            p[ones[0]] += pr; cons += pr
    return p / max(cons, 1e-12), cons


def site_probs_counts(counts):
    s = sum(counts.values()); p = np.zeros(N); cons = 0
    for b, c in counts.items():
        ones = [N - 1 - k for k, ch in enumerate(b.replace(" ", "")) if ch == "1"]
        if len(ones) == 1:
            p[ones[0]] += c; cons += c
    return p / max(1, cons), cons / s


def tvd(a, b): return 0.5 * float(np.abs(a - b).sum())
def equiv_c3(P): return float(np.mean([tvd(P[k], np.roll(P[0], k)) for k in (2, 4)]))


def simulate():
    sym = [site_probs_sv(build(k, measure=False))[0] for k in range(N)]
    brk = [site_probs_sv(build(k, detune=0, measure=False))[0] for k in range(N)]
    triv = [site_probs_sv(build(k, t2=0.0, measure=False))[0] for k in range(N)]
    Pp = site_probs_sv(build(0, phi=+PHI, measure=False))[0]
    Pm = site_probs_sv(build(0, phi=-PHI, measure=False))[0]
    Pp0 = site_probs_sv(build(0, phi=+PHI, t2=0.0, measure=False))[0]
    E3, E3b, E3t = equiv_c3(sym), equiv_c3(brk), equiv_c3(triv)
    n2q = sum(1 for inst in build(0).data if inst.operation.num_qubits == 2)
    spread = int((Pp > 0.01).sum())
    chir, chir_triv = float(Pp[1] - Pp[5]), float(Pp0[1] - Pp0[5])
    mirror, mirror_triv = tvd(Pp, Pm), tvd(Pp0, Pp0)
    print(f"[sim] Haldane ring N={N} | J={J} t2={T2} phi={PHI:.3f} dt={DT} steps={STEPS} | "
          f"logical 2q gates/circuit = {n2q}")
    print(f"[sim] P_0 (start 0, phi+) = {np.round(Pp,3)}  (spread {spread} sites)")
    print(f"\n  --- SYMMETRY gift ---")
    print(f"[sim] E_C3 (Haldane ring, orbit 0,2,4)               = {E3:.6f}  (small = approx preserved)")
    print(f"[sim] E_C3 (trivial t2=0 ring, control)              = {E3t:.6f}  (exact)")
    print(f"[sim] E_C3_broken (detuned site0)                    = {E3b:.6f}  (should be >> E_C3)")
    print(f"  --- TOPOLOGY (chirality) gift ---")
    print(f"[sim] chirality P[1]-P[5] (Haldane)                  = {chir:+.4f}  (!=0 = chiral)")
    print(f"[sim] chirality P[1]-P[5] (trivial t2=0)             = {chir_triv:+.4f}  (~0)")
    print(f"[sim] mirror TVD(phi+,phi-) (Haldane)                = {mirror:.4f}  (>0 = topology)")
    sym_ok = (E3 < 0.05) and (E3b > 8 * max(E3, 1e-3)) and (E3t < 1e-4)
    topo_ok = (abs(chir) > 0.05) and (abs(chir_triv) < 0.02) and (mirror > 0.05)
    print(f"\n[gate] SYMMETRY (E_C3 small + exact-for-trivial + broken>>): {sym_ok}")
    print(f"[gate] TOPOLOGY (chirality present + vanishes for trivial):   {topo_ok}")
    print(f"[gate] depth: {n2q} logical 2q -> ~{n2q*2}-{n2q*3} transpiled "
          f"({'feasible for 6q (cf. proven 12->42 ring)' if n2q <= 28 else 'may be too deep'})")
    clean = sym_ok and topo_ok and n2q <= 28
    print(f"[gate] {'>>> BOTH GIFTS clean in sim -> submittable (confirm before spending slot)' if clean else '>>> NOT clean -> do NOT submit'}")
    return clean


def submit(backend_name):
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    labels = ([f"sym{k}" for k in range(N)] + [f"brk{k}" for k in range(N)]
              + ["chir+", "chir-", "triv+"])
    qcs = ([build(k) for k in range(N)] + [build(k, detune=0) for k in range(N)]
           + [build(0, phi=+PHI), build(0, phi=-PHI), build(0, phi=+PHI, t2=0.0)])
    svc = QiskitRuntimeService()
    backend = svc.backend(backend_name) if backend_name else svc.least_busy(operational=True, simulator=False)
    print(f"[backend] {backend.name} ({backend.num_qubits}q), queue={backend.status().pending_jobs}", flush=True)
    ref = transpile(qcs[0], backend, optimization_level=3, seed_transpiler=11)
    init = ref.layout.initial_index_layout(filter_ancillas=True)
    tq = transpile(qcs, backend, optimization_level=3, initial_layout=init, seed_transpiler=11)
    n2 = max(sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in tq)
    print(f"[transpile] pinned layout {init[:N]} | max transpiled 2q = {n2}", flush=True)
    sampler = SamplerV2(mode=backend); sampler.options.default_shots = SHOTS
    job = sampler.run(tq)
    print(f"[job] {job.job_id()} submitted to {backend.name}", flush=True)
    open(os.path.join(ROOTRUN, "ibm_eqtopo_jobid.txt"), "w").write(job.job_id())
    res = job.result()
    P = {}
    for lab, r in zip(labels, res):
        c = r.data.c.get_counts() if hasattr(r.data, "c") else r.data[list(r.data.keys())[0]].get_counts()
        P[lab] = site_probs_counts(c)
    out = {"backend": backend.name, "job_id": job.job_id(),
           "P": {k: [v[0].tolist(), v[1]] for k, v in P.items()}}
    E3 = float(np.mean([tvd(P[f"sym{k}"][0], np.roll(P["sym0"][0], k)) for k in (2, 4)]))
    E3b = float(np.mean([tvd(P[f"brk{k}"][0], np.roll(P["brk0"][0], k)) for k in (2, 4)]))
    out["E_C3_hw"], out["E_C3_broken_hw"] = E3, E3b
    out["chirality_hw"] = float(P["chir+"][0][1] - P["chir+"][0][5])
    out["chirality_triv_hw"] = float(P["triv+"][0][1] - P["triv+"][0][5])
    out["mirror_hw"] = tvd(P["chir+"][0], P["chir-"][0])
    out["conservation"] = float(np.mean([P[k][1] for k in P]))
    json.dump(out, open(os.path.join(ROOTRUN, "ibm_eqtopo_result.json"), "w"), indent=2)
    print(f"\n[HARDWARE] E_C3={E3:.3f} (broken {E3b:.3f}) | chirality={out['chirality_hw']:+.3f} "
          f"(trivial {out['chirality_triv_hw']:+.3f}) | mirror={out['mirror_hw']:.3f} | "
          f"conservation={out['conservation']:.3f}")
    print(f"[saved] runs/ibm_eqtopo_result.json")


ROOTRUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    clean = simulate()
    if not a.submit:
        print("\n(sim-only; pass --submit after confirming, to run on hardware)")
        return
    if not clean:
        print("\n[abort] gate not clean — refusing to submit (hard rule)."); return
    submit(a.backend)


if __name__ == "__main__":
    main()
