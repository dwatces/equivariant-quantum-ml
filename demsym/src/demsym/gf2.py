"""Dense GF(2) linear algebra, sized for detector-error-model problems."""
from __future__ import annotations

import numpy as np


def solve_gf2(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Solve A x = b over GF(2). Returns one solution, or None if inconsistent.

    A: (m, n) 0/1 matrix, b: (m,) 0/1 vector. Free variables are set to 0.
    """
    A = (np.asarray(A).copy() % 2).astype(np.int8)
    b = (np.asarray(b).copy() % 2).astype(np.int8)
    m, n = A.shape
    piv, r = [], 0
    for col in range(n):
        if r >= m:
            break
        rows = np.nonzero(A[r:, col])[0]
        if len(rows) == 0:
            continue
        A[[r, r + rows[0]]] = A[[r + rows[0], r]]
        b[[r, r + rows[0]]] = b[[r + rows[0], r]]
        mask = A[:, col].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        b[mask] ^= b[r]
        piv.append(col)
        r += 1
    if np.any(b[r:]):
        return None
    x = np.zeros(n, dtype=np.int64)
    for k, col in enumerate(piv):
        x[col] = b[k]
    return x


def gl2_matrices(k: int) -> list[np.ndarray]:
    """All invertible k x k matrices over GF(2). Sizes: 1, 6, 168 for k=1,2,3."""
    if k > 3:
        raise ValueError(
            f"enumerating GL({k},F2) ({_gl_order(k)} matrices) is not done "
            "automatically; pass A_candidates explicitly (e.g. permutation "
            "matrices, or the subgroup your lattice action can produce)")
    out = []
    for bits in range(1 << (k * k)):
        M = np.array([[(bits >> (i * k + j)) & 1 for j in range(k)]
                      for i in range(k)], dtype=np.int64)
        if _gf2_det(M):
            out.append(M)
    # identity first: it is the most common answer and exits the search early
    out.sort(key=lambda M: 0 if (M == np.eye(k, dtype=np.int64)).all() else 1)
    return out


def _gf2_det(M: np.ndarray) -> int:
    M = M.copy() % 2
    n = M.shape[0]
    for col in range(n):
        rows = np.nonzero(M[col:, col])[0]
        if len(rows) == 0:
            return 0
        r = col + rows[0]
        M[[col, r]] = M[[r, col]]
        mask = M[:, col].astype(bool)
        mask[col] = False
        M[mask] ^= M[col]
    return 1


def _gl_order(k: int) -> int:
    n = 1
    for i in range(k):
        n *= (1 << k) - (1 << i)
    return n
