"""
MAGIC (non-stabilizerness) across the Kitaev honeycomb phase diagram. 2026-06-10.
The quantum/classical separator measured on the project's own model: stabilizer Renyi
entropy M2 (Leone-Oliviero-Hamma, PRL 128.050402) of the exact ground state vs the
uniform field h that opens the NON-ABELIAN phase.

PREDICTIONS (logged before the run, per the hard rule):
  (a) toric corner (Jz dominant, h=0) = lowest magic (toric-code limit is stabilizer);
  (b) magic GROWS with h (the non-Abelian-phase knob);
  (c) isotropic point > toric at fixed h.

Method: exact ED (LinearOperator matvec + Lanczos) on honeycomb tori n=8 (2x2 cells)
and n=12 (2x3 cells). SRE via Walsh-Hadamard trick: for each X-mask a, the vector
v_s = conj(psi_{s XOR a}) psi_s; WHT over s gives <X^a Z^b> for ALL b at once.
M2 = -log2( sum_P |<P>|^4 / 2^n ).  Output: runs/kitaev_magic.{log,json}.
"""
import json
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

T0 = time.time()
OUT = {"predictions": ["toric h=0 lowest", "M2 grows with h", "iso > toric at fixed h"]}


def log(m):
    print(m, flush=True)


def build_terms(L1, L2):
    """Bond lists (i,j,type) for the honeycomb torus; site = 2*(x*L2+y)+s."""
    idx = lambda x, y, s: 2 * ((x % L1) * L2 + (y % L2)) + s
    bonds = []
    for x in range(L1):
        for y in range(L2):
            a = idx(x, y, 0)
            bonds.append((a, idx(x, y, 1), "z"))
            bonds.append((a, idx(x - 1, y, 1), "x"))
            bonds.append((a, idx(x, y - 1, 1), "y"))
    return bonds


def make_H(L1, L2, Jx, Jy, Jz, h):
    n = 2 * L1 * L2
    dim = 1 << n
    bonds = build_terms(L1, L2)
    J = {"x": Jx, "y": Jy, "z": Jz}
    s = np.arange(dim, dtype=np.int64)
    # precompute per-term index/phase arrays
    terms = []
    for (i, j, t) in bonds:
        bi, bj = 1 << i, 1 << j
        if t == "z":
            diag = ((s >> i) & 1) ^ ((s >> j) & 1)
            terms.append(("diag", -J[t] * (1.0 - 2.0 * diag)))  # ZZ = +1 same, -1 diff
        else:
            flip = s ^ bi ^ bj
            if t == "x":
                terms.append(("off", flip, -J[t] * np.ones(dim)))  # XX
            else:  # YY = (iY)(iY)*(-1) ... Y|b> = i(-1)^b|1-b>
                zi = (s >> i) & 1
                zj = (s >> j) & 1
                ph = (1.0 - 2.0 * zi) * (1.0 - 2.0 * zj) * (-1.0)
                terms.append(("off", flip, -J[t] * ph))
    if h != 0.0:
        c = h / np.sqrt(3.0)
        for q in range(n):
            bq = 1 << q
            flip = s ^ bq
            terms.append(("off", flip, -c * np.ones(dim)))                      # X
            zq = (s >> q) & 1
            terms.append(("off", flip, -c * 1j * (1.0 - 2.0 * zq) * (-1.0)))    # Y: i(-1)^b... see note
            terms.append(("diag", -c * (1.0 - 2.0 * zq)))                        # Z
    diag_tot = np.zeros(dim)
    offs = []
    for t in terms:
        if t[0] == "diag":
            diag_tot = diag_tot + t[1]
        else:
            offs.append((t[1], t[2]))

    def mv(psi):
        out = diag_tot * psi
        for flip, amp in offs:
            out = out + amp * psi[flip]
        return out

    return LinearOperator((dim, dim), matvec=mv, dtype=complex), n


def sre_m2(psi, n):
    """Stabilizer Renyi entropy M2 = -log2( sum_P <P>^4 / 2^n ), |<P>| via WHT."""
    dim = 1 << n
    psi = psi / np.linalg.norm(psi)
    tot = 0.0
    sarr = np.arange(dim, dtype=np.int64)
    for a in range(dim):
        v = np.conj(psi[sarr ^ a]) * psi
        # Walsh-Hadamard transform of v -> <X^a Z^b> (up to Hermitian phase, |.| safe)
        w = v.copy()
        step = 1
        while step < dim:
            w = w.reshape(-1, 2 * step)
            left = w[:, :step].copy()
            right = w[:, step:].copy()
            w[:, :step] = left + right
            w[:, step:] = left - right
            w = w.reshape(-1)
            step *= 2
        tot += float(np.sum(np.abs(w) ** 4))
    return -np.log2(tot / dim)


def flux_op_exp(psi, L1, L2):
    """<W_p> for plaquette ops (sanity: flux conservation degrades with h).
    W_p over hexagon: product sigma^x/y/z pattern; use the 8-site standard: skip
    exact op bookkeeping for n=12 and report only for n=8 2x2 (plaquette = XYZXYZ
    on sites of one hexagon h(0,0): A(0,0)z..., use the known 6-site cycle)."""
    return None  # kept minimal; phase integrity tracked via energy continuity instead


CASES = [("iso", 1.0, 1.0, 1.0), ("toric", 1.0, 1.0, 2.5)]
HS = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6]
RES = {}
for (L1, L2) in [(2, 2), (2, 3)]:
    n = 2 * L1 * L2
    for name, Jx, Jy, Jz in CASES:
        key = f"n{n}_{name}"
        RES[key] = []
        for h in HS:
            H, _ = make_H(L1, L2, Jx, Jy, Jz, h)
            w, v = eigsh(H, k=1, which="SA", maxiter=5000, tol=1e-10)
            m2 = sre_m2(v[:, 0], n)
            RES[key].append({"h": h, "E0": float(w[0]), "M2": float(m2),
                             "M2_per_qubit": float(m2 / n)})
            log(f"{key} h={h:.2f}: E0={w[0]:.4f}  M2={m2:.4f}  ({m2/n:.4f}/qubit)")
OUT["results"] = RES
OUT["runtime_s"] = time.time() - T0
with open("runs/kitaev_magic.json", "w") as f:
    json.dump(OUT, f, indent=1)
log(f"DONE in {OUT['runtime_s']:.0f}s -> runs/kitaev_magic.json")
