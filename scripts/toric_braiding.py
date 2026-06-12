"""ANYON BRAIDING — the -1 mutual statistics, the genuine TQC primitive (classical ground truth).

The deepest on-thesis test: the honeycomb Kitaev's anisotropic limit IS the toric code, whose e/m
anyons have -1 MUTUAL STATISTICS. Braiding an anyon around the torus cycle <=> the logical X-loop
and Z-loop (which cross once) ANTICOMMUTE: dragging the anyon flips the conjugate logical's sign.

8-qubit toric code (2x2 torus, qubits on edges). Stars A_v (X-type), plaquettes B_p (Z-type).
Ground state |0_L> from |0..0> (already +1 of all B_p) projected onto A_v=+1.
  * <Zbar> = +1 on |0_L> (vacuum).
  * apply logical Xbar (anticommutes Zbar = an anyon dragged around the crossing cycle) -> <Zbar> = -1
    == the -1 BRAIDING phase.
  * apply a CONTRACTIBLE X-loop (a star = stabilizer, no anyon dragged around) -> <Zbar> stays +1.
The (+1 control) vs (-1 braid) contrast IS the mutual statistics. Logical operators are found
NUMERICALLY (commute with all stabilizers, anticommute with each other) so the geometry can't be
mis-set. No number recorded before its run completes.
"""
import os, sys, json, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kitaev_order import op

N = 8
# edges: 0=h00 1=h01 2=h10 3=h11 4=v00 5=v01 6=v10 7=v11
STARS = [[0, 1, 4, 6], [0, 1, 5, 7], [2, 3, 4, 6], [2, 3, 5, 7]]      # X-type A_v
PLAQS = [[0, 2, 4, 5], [1, 3, 4, 5], [0, 2, 6, 7], [1, 3, 6, 7]]      # Z-type B_p


def pauli_str(sites, p):
    return op({s: p for s in sites}, N)


def commutes(A, B):
    return np.linalg.norm(A @ B - B @ A) < 1e-9


def main():
    os.makedirs("runs", exist_ok=True)
    logf = open("runs/toric_braiding.log", "a")
    def log(m): print(m); logf.write(m + "\n"); logf.flush()
    log(f"\n=== TORIC-CODE ANYON BRAIDING (-1 mutual statistics) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")

    stab = [pauli_str(s, "X") for s in STARS] + [pauli_str(s, "Z") for s in PLAQS]
    # verify stabilizers mutually commute
    okcomm = all(commutes(stab[i], stab[j]) for i in range(len(stab)) for j in range(i + 1, len(stab)))
    log(f"stabilizers: {len(STARS)} stars (X) + {len(PLAQS)} plaquettes (Z) | all commute: {okcomm}")

    # ground-state projector and a ground state
    I = np.eye(2**N)
    P = I.copy()
    for S in stab:
        P = P @ ((I + S) / 2)
    rng = np.random.default_rng(0)
    G = P @ (rng.standard_normal(2**N) + 1j * rng.standard_normal(2**N))
    G /= np.linalg.norm(G)
    deg = int(round(np.real(np.trace(P))))      # ground-space dimension = 2^(#logical qubits)
    log(f"ground-space dimension = {deg} (=> {int(np.log2(deg))} logical qubits)")

    # find logical operators numerically: X-string and Z-string commuting with ALL stabilizers,
    # anticommuting with each other (=> a conjugate logical pair Xbar/Zbar).
    def find_logical(ptype):
        cands = []
        for w in (2, 3, 4):
            for sub in itertools.combinations(range(N), w):
                Lp = pauli_str(sub, ptype)
                if all(commutes(Lp, S) for S in stab):
                    # exclude stabilizers themselves (act as identity on ground space)
                    if np.linalg.norm(Lp @ G - G) > 1e-6 and np.linalg.norm(Lp @ G + G) > 1e-6:
                        cands.append((sub, Lp))
            if cands:
                break
        return cands

    Xcands = find_logical("X"); Zcands = find_logical("Z")
    Xbar = Zbar = None
    for sx, LX in Xcands:
        for sz, LZ in Zcands:
            if not commutes(LX, LZ):                # anticommute => conjugate logical pair
                Xbar, Zbar, sxw, szw = LX, LZ, sx, sz; break
        if Xbar is not None:
            break
    log(f"logical Xbar = X{list(sxw)} | logical Zbar = Z{list(szw)} | anticommute: {not commutes(Xbar, Zbar)}")

    # |0_L>: ground state with <Zbar>=+1 (|0..0> already +1 of plaquettes; project onto Zbar=+1 too)
    psi = ((I + Zbar) / 2) @ G
    psi /= np.linalg.norm(psi)
    z0 = float(np.real(psi.conj() @ Zbar @ psi))
    log(f"\nvacuum |0_L>: <Zbar> = {z0:+.3f} (want +1)")

    # BRAID: apply logical Xbar (drags an anyon around the crossing cycle)
    psib = Xbar @ psi
    zb = float(np.real(psib.conj() @ Zbar @ psib))
    log(f"after BRAID (apply Xbar, anticommutes Zbar): <Zbar> = {zb:+.3f}  == the -1 mutual-statistics phase")

    # CONTROL: apply a contractible X-loop (a star = stabilizer; no anyon dragged around the cycle)
    star = pauli_str(STARS[0], "X")
    psic = star @ psi
    zc = float(np.real(psic.conj() @ Zbar @ psic))
    log(f"CONTROL (apply a star = contractible loop): <Zbar> = {zc:+.3f}  (stays +1 => no braiding)")

    log(f"\nVERDICT: braid -> {zb:+.2f} vs control -> {zc:+.2f}.  The -1 vs +1 contrast = anyon "
        f"mutual statistics (Xbar/Zbar cross once -> anticommute). Toric code = honeycomb Kitaev "
        f"anisotropic limit; square has no such anyons.")
    json.dump(dict(stab_commute=okcomm, ground_dim=deg, xbar=list(sxw), zbar=list(szw),
                   vacuum=z0, braid=zb, control=zc), open("runs/toric_braiding.json", "w"),
              indent=2, default=float)
    log("=== done ===")


if __name__ == "__main__":
    main()
