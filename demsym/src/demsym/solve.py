"""Solve the twisted label action (A, X) for a candidate detector permutation.

Method (per candidate A): for every detector-set group v, the image group
perm(v) must carry the same multiset of (observable, probability) pairs
after o -> A o XOR b, where b = X·indicator(perm(v)) is a single unknown
GF(2) bit-vector per group. Each group constrains b to a consistent set;
bits forced to a single value become linear equations on the rows of X,
solved by GF(2) elimination; the candidate is accepted only after a final
gauge-free multiset verification of the full DEM.
"""
from __future__ import annotations

import numpy as np

from .dem import Dem
from .element import Element, apply_A
from .gf2 import gl2_matrices, solve_gf2


def solve_action(dem: Dem, perm, name: str = "g",
                 A_candidates: list[np.ndarray] | None = None
                 ) -> Element | None:
    """Find (A, X) making `perm` a twisted symmetry of `dem`.

    perm: array of length num_detectors, detector d -> perm[d]. Must be a
    bijection (raises otherwise).
    A_candidates: optional list of k x k GF(2) matrices to try, in order.
    Defaults to all of GL(k, F2) for k <= 3; for larger k pass candidates
    explicitly (e.g. the 2^k observable permutation matrices).

    Returns a verified Element, or None if no candidate A admits a
    consistent X. None is a real answer: the permutation is not a twisted
    symmetry of this DEM under any tried A.
    """
    perm = np.asarray(perm, dtype=np.int64)
    nd = dem.num_detectors
    if perm.shape != (nd,) or len(set(perm.tolist())) != nd:
        raise ValueError(f"perm must be a bijection on {nd} detectors")
    k = dem.num_observables
    if A_candidates is None:
        A_candidates = gl2_matrices(k)

    groups = dem.groups
    nb = 1 << k
    for Amat in A_candidates:
        Amat = np.asarray(Amat, dtype=np.int64) % 2
        eq_rows = [[] for _ in range(k)]
        eq_t = [[] for _ in range(k)]
        ok = True
        for v, group in groups.items():
            v2 = frozenset(int(perm[d]) for d in v)
            tgt = groups.get(v2)
            if tgt is None:
                ok = False
                break
            src = sorted((apply_A(Amat, o), p) for o, p in group)
            cons = [b for b in range(nb)
                    if tuple(sorted((o ^ b, p) for o, p in src)) == tgt]
            if not cons:
                ok = False
                break
            row = None
            for bit in range(k):
                vals = {(b >> bit) & 1 for b in cons}
                if len(vals) == 1:
                    if row is None:
                        row = np.zeros(nd, dtype=np.int8)
                        for d in v2:
                            row[d] = 1
                    eq_rows[bit].append(row)
                    eq_t[bit].append(vals.pop())
        if not ok:
            continue
        X, good = [], True
        for bit in range(k):
            if eq_rows[bit]:
                sol = solve_gf2(np.array(eq_rows[bit]),
                                np.array(eq_t[bit], dtype=np.int8))
                if sol is None:
                    good = False
                    break
                X.append(sol)
            else:
                X.append(np.zeros(nd, dtype=np.int64))
        if not good:
            continue
        cand = Element(name, perm, Amat, np.stack(X))
        if cand.verify(dem):
            return cand
    return None


def close(generators: list[Element], dem: Dem,
          max_size: int = 1024) -> list[Element]:
    """Group closure of verified elements, with the identity included.

    Every product is re-verified against the DEM (cheap, and a convention
    self-check). Elements are deduplicated by their detector permutation.
    Raises if the closure exceeds max_size or a product fails to verify.
    """
    from .element import identity
    if not generators:
        return [identity(dem)]
    els = {identity(dem).perm.tobytes(): identity(dem)}
    frontier = []
    for g in generators:
        key = g.perm.tobytes()
        if key not in els:
            if not g.verify(dem):
                raise ValueError(f"generator {g.name} does not verify")
            els[key] = g
            frontier.append(g)
    while frontier:
        new = []
        for g in frontier:
            for h in list(els.values()):
                for prod in (g @ h, h @ g):
                    key = prod.perm.tobytes()
                    if key in els:
                        continue
                    if not prod.verify(dem):
                        raise AssertionError(
                            f"product {prod.name} of verified elements fails "
                            "verification — composition convention bug")
                    els[key] = prod
                    new.append(prod)
                    if len(els) > max_size:
                        raise ValueError(f"closure exceeds {max_size}")
        frontier = new
    return list(els.values())
