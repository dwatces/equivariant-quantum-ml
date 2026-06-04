"""S3 — TWO-PARTICLE (many-body) version of the both-gifts circuit: pushes the coexistence from single-
particle into the genuinely-quantum ENTANGLED regime (the paper's main caveat). Two excitations on the
6-ring with a hard-core constraint (a free interaction) + the same NN 2-coloured Trotter (C3) + Haldane
chiral NNN (topology). Measures whether the 2-BODY correlation is BOTH C3-equivariant AND chiral.

  * SYMMETRY: start the 2 excitations on a C3 orbit of pairs ({0,3}->{2,5}->{4,1} under S^2). The
    2-particle pair-distribution G(i,j) of rotated starts must be the C3-rotation of G(start0): E_C3.
  * TOPOLOGY: Haldane phase steers the 2-body correlation -> mirror TVD(phi+,phi-) > 0; vanishes for t2=0.
Reuses the validated ring/Trotter/Haldane build. SIM-GATE first; submit only on confirm.
"""
import os, sys, math, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector
from ibm_equivariant_topological import LAYER_A, LAYER_B, TRI_A, TRI_B, J, T2, PHI, DT, STEPS, MU, N, SHOTS, ROOTRUN

# C3 orbit of the 2-excitation start under S^2 (k->k+2): {0,3} -> {2,5} -> {4,1}
START_ORBIT = [(0, 3), (2, 5), (4, 1)]


def build2(starts, phi=PHI, t2=T2, detune=None, measure=True):
    qc = QuantumCircuit(N, N)
    for s in starts:
        qc.x(s)
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


def pairdist_sv(qc):
    """2-excitation-sector pair distribution G[i,j] (i<j), normalized; + conservation."""
    sv = Statevector.from_instruction(qc)
    G = np.zeros((N, N)); cons = 0.0
    for b, pr in sv.probabilities_dict().items():
        ones = [N - 1 - k for k, ch in enumerate(b) if ch == "1"]
        if len(ones) == 2:
            i, j = sorted(ones); G[i, j] += pr; cons += pr
    return G / max(cons, 1e-12), cons


def pairdist_counts(counts):
    G = np.zeros((N, N)); s = sum(counts.values()); cons = 0
    for b, c in counts.items():
        ones = [N - 1 - k for k, ch in enumerate(b.replace(" ", "")) if ch == "1"]
        if len(ones) == 2:
            i, j = sorted(ones); G[i, j] += c; cons += c
    return G / max(1, cons), cons / s


def roll_pairs(G, s):
    """C3 rotation by s*120deg (S^2 = k->k+2 per step) acting on the pair matrix."""
    R = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ni, nj = sorted(((i + 2 * s) % N, (j + 2 * s) % N))
            R[ni, nj] += G[i, j]
    return R


def tvd_mat(A, B): return 0.5 * float(np.abs(A - B).sum())


def equiv_c3_pairs(Glist):
    """Glist[k] = pair-dist from START_ORBIT[k]. E_C3 = mean TVD(G_k, C3^k . G_0)."""
    return float(np.mean([tvd_mat(Glist[k], roll_pairs(Glist[0], k)) for k in (1, 2)]))


def simulate():
    sym = [pairdist_sv(build2(START_ORBIT[k], measure=False))[0] for k in range(3)]
    brk = [pairdist_sv(build2(START_ORBIT[k], detune=0, measure=False))[0] for k in range(3)]
    triv = [pairdist_sv(build2(START_ORBIT[k], t2=0.0, measure=False))[0] for k in range(3)]
    Gp = pairdist_sv(build2(START_ORBIT[0], phi=+PHI, measure=False))[0]
    Gm = pairdist_sv(build2(START_ORBIT[0], phi=-PHI, measure=False))[0]
    Gp0 = pairdist_sv(build2(START_ORBIT[0], phi=+PHI, t2=0.0, measure=False))[0]
    Gm0 = pairdist_sv(build2(START_ORBIT[0], phi=-PHI, t2=0.0, measure=False))[0]
    E3, E3b, E3t = equiv_c3_pairs(sym), equiv_c3_pairs(brk), equiv_c3_pairs(triv)
    chir = tvd_mat(Gp, Gm); chir_triv = tvd_mat(Gp0, Gm0)
    n2q = sum(1 for inst in build2(START_ORBIT[0]).data if inst.operation.num_qubits == 2)
    print(f"[sim] 2-particle Haldane ring | starts {START_ORBIT} | t2={T2} phi={PHI:.3f} | 2q gates={n2q}")
    print(f"\n  --- SYMMETRY (2-body) ---")
    print(f"[sim] E_C3 (pair-dist, Haldane)         = {E3:.6f}  (small = equivariant 2-body correlation)")
    print(f"[sim] E_C3 (trivial t2=0)               = {E3t:.6f}")
    print(f"[sim] E_C3_broken (detuned)             = {E3b:.6f}  (>> E_C3)")
    print(f"  --- TOPOLOGY (2-body chirality) ---")
    print(f"[sim] mirror TVD(phi+,phi-) (Haldane)   = {chir:.4f}  (>0 = topology steers 2-body corr)")
    print(f"[sim] mirror TVD(phi+,phi-) (trivial)   = {chir_triv:.4f}  (~0)")
    sym_ok = (E3 < 0.10) and (E3b > 4 * max(E3, 1e-3)) and (E3t < 1e-4)
    topo_ok = (chir > 0.05) and (chir_triv < 0.02)
    print(f"\n[gate] SYMMETRY (2-body E_C3 small, broken>>, trivial exact): {sym_ok}")
    print(f"[gate] TOPOLOGY (2-body chirality present + vanishes trivial): {topo_ok}")
    print(f"[gate] depth {n2q} logical 2q -> ~{n2q*2}-{n2q*3} transpiled; 2-excitation conservation will be "
          f"LOWER than single-particle (more leakage) — expect ~0.3-0.5 (cf. project 2-particle runs).")
    clean = sym_ok and topo_ok
    print(f"[gate] {'>>> 2-PARTICLE both gifts clean in sim -> submittable (confirm)' if clean else '>>> NOT clean -> do NOT submit'}")
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--backend", default="ibm_marrakesh")
    a = ap.parse_args()
    clean = simulate()
    if not a.submit:
        print("\n(sim-only; pass --submit after confirming)")
        return
    if not clean:
        print("[abort] gate not clean."); return
    print("[submit] (build hardware submit on confirm — reuses the robust runner's mitigation path)")


if __name__ == "__main__":
    main()
