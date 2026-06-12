"""
BRIDGE: our DEM-based (A, X) solver vs The END's analytic M_g(sigma)
(Egorov-Bondesan-Welling, ICLR 2023, arXiv:2304.07362) on THEIR setting —
toric code, code capacity (i.i.d. depolarizing, perfect syndromes).

The END derive the optimal-decoder symmetry action analytically from lattice
paths: labels gamma=(X1,X2,Z1,Z2) transform by a permutation P_g composed with
syndrome-dependent flips (path sums Delta_g(sigma)). Our pipeline instead SOLVES
the action (A, X) from the detector error model per det-set group and verifies it
mechanism-by-mechanism (the route that still exists at circuit level, where
analytic paths don't). If both agree on the toric code, "independent
reconstruction" becomes a verified statement.

PREDICTIONS LOGGED BEFORE THE RUN (from The END Fig. 3; hard rule — no results
recorded before completion):
  P1 translations T10/T01: A = I (labels fixed), chi != 0 on the moved labels.
  P2 rotation 90 deg:      A = perm swapping X1<->X2 and Z1<->Z2 (homology mix).
  P3 mirror j -> -j:       A = I, chi != 0 only on the X2 row (their Fig 3).
  P4 all elements verify against the FULL mechanism set, and the A found by
     blind search over all 24 label permutations is UNIQUE.

Setting: L x L torus, qubits on edges h(i,j) [(i,j)-(i,j+1)] and v(i,j)
[(i,j)-(i+1,j)]; A(i,j)=X on {h(i,j),h(i,j-1),v(i,j),v(i-1,j)} (dets 0..L^2-1),
B(i,j)=Z on {h(i,j),h(i+1,j),v(i,j),v(i,j+1)} (dets L^2..2L^2-1).
Logicals: X1=X{h(i,0)}, X2=X{v(0,j)}, Z1=Z{h(0,j)}, Z2=Z{v(i,0)}.
Mechanisms: per edge X_e, Z_e, Y_e at p/3 (code capacity depolarizing).
Output: runs/toric_end_bridge.{log,json}
"""
import itertools
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import solve_gf2, log  # noqa: E402

L = 6
P = 0.1
T0 = time.time()


def edges():
    for i in range(L):
        for j in range(L):
            yield ("h", i, j)
            yield ("v", i, j)


EID = {e: k for k, e in enumerate(edges())}


def vdet(i, j):
    return (i % L) * L + (j % L)


def pdet(i, j):
    return L * L + (i % L) * L + (j % L)


def edge_dets_X(e):  # X error -> plaquettes containing e (Z-stabilizers)
    t, i, j = e
    if t == "h":  # h(i,j) in B(i,j) and B(i-1,j)
        return {pdet(i, j), pdet(i - 1, j)}
    return {pdet(i, j), pdet(i, j - 1)}  # v(i,j) in B(i,j), B(i,j-1)


def edge_dets_Z(e):  # Z error -> vertices containing e (X-stabilizers)
    t, i, j = e
    if t == "h":  # h(i,j) in A(i,j) and A(i,j+1)
        return {vdet(i, j), vdet(i, j + 1)}
    return {vdet(i, j), vdet(i + 1, j)}  # v(i,j) in A(i,j), A(i+1,j)


XBAR1 = {("h", i, 0) for i in range(L)}
XBAR2 = {("v", 0, j) for j in range(L)}
ZBAR1 = {("h", 0, j) for j in range(L)}
ZBAR2 = {("v", i, 0) for i in range(L)}


def obs_of(e, pauli):
    """gamma bits (X1,X2,Z1,Z2) flipped by a single-qubit error."""
    o = 0
    if pauli in ("Z", "Y"):  # Z component anticommutes with X-logicals through e
        o |= (e in XBAR1) << 0
        o |= (e in XBAR2) << 1
    if pauli in ("X", "Y"):
        o |= (e in ZBAR1) << 2
        o |= (e in ZBAR2) << 3
    return o


def sanity():
    # stabilizer/logical commutation checks
    for i in range(L):
        for j in range(L):
            Av = {("h", i, j % L), ("h", i, (j - 1) % L),
                  ("v", i, j % L), ("v", (i - 1) % L, j)}
            Av = {(t, a % L, b % L) for t, a, b in Av}
            Bp = {("h", i % L, j), ("h", (i + 1) % L, j),
                  ("v", i, j % L), ("v", i, (j + 1) % L)}
            Bp = {(t, a % L, b % L) for t, a, b in Bp}
            for Lset in (XBAR1, XBAR2):
                assert len(Lset & Bp) % 2 == 0  # X-logical vs Z-plaquette
            for Lset in (ZBAR1, ZBAR2):
                assert len(Lset & Av) % 2 == 0
    assert len(XBAR1 & ZBAR1) % 2 == 1 and len(XBAR2 & ZBAR2) % 2 == 1
    assert len(XBAR1 & ZBAR2) % 2 == 0 and len(XBAR2 & ZBAR1) % 2 == 0
    log("sanity: stabilizer/logical commutation table correct")


def mechanisms():
    by_v = {}
    for e in EID:
        for pauli in "XZY":
            dets = set()
            if pauli in ("X", "Y"):
                dets ^= edge_dets_X(e)
            if pauli in ("Z", "Y"):
                dets ^= edge_dets_Z(e)
            by_v.setdefault(frozenset(dets), []).append(
                (obs_of(e, pauli), round(P / 3, 9)))
    return by_v


# ---- lattice maps on edges and detectors -------------------------------------
def T10(e):
    t, i, j = e
    return (t, (i + 1) % L, j)


def T01(e):
    t, i, j = e
    return (t, i, (j + 1) % L)


def ROT(e):  # vertex (i,j) -> (j, -i)
    t, i, j = e
    if t == "h":
        return ("v", j % L, (-i) % L)
    return ("h", j % L, (-i - 1) % L)


def MIR(e):  # vertex (i,j) -> (i, -j)
    t, i, j = e
    if t == "h":
        return ("h", i, (-j - 1) % L)
    return ("v", i, (-j) % L)


def det_perm(emap, vmapf, pmapf):
    perm = np.zeros(2 * L * L, dtype=np.int64)
    for i in range(L):
        for j in range(L):
            perm[vdet(i, j)] = vmapf(i, j)
            perm[pdet(i, j)] = pmapf(i, j)
    assert len(set(perm.tolist())) == 2 * L * L
    return perm


# vertex (i,j) maps with the vertex map; plaquette B(i,j) is the cell whose
# corner vertex is (i,j) — derive its image by mapping its edge set
def perm_from_edgemap(emap):
    """Detector permutation induced by an edge map, derived from incidence."""
    # vertex image: A(i,j) edge set mapped -> must equal A(i',j') edge set
    vsets = {}
    psets = {}
    for i in range(L):
        for j in range(L):
            vs = frozenset({("h", i, j), ("h", i, (j - 1) % L),
                            ("v", i, j), ("v", (i - 1) % L, j)})
            ps = frozenset({("h", i, j), ("h", (i + 1) % L, j),
                            ("v", i, j), ("v", i, (j + 1) % L)})
            vsets[vs] = vdet(i, j)
            psets[ps] = pdet(i, j)
    perm = np.zeros(2 * L * L, dtype=np.int64)
    for sets, look_self, look_other in ((vsets, vsets, psets),
                                        (psets, psets, vsets)):
        for s, d in sets.items():
            img = frozenset(emap(e) for e in s)
            if img in look_self:
                perm[d] = look_self[img]
            elif img in look_other:
                perm[d] = look_other[img]
            else:
                raise AssertionError("edge map is not a lattice automorphism")
    assert len(set(perm.tolist())) == 2 * L * L
    return perm


def apply_A4(Amat, ob):
    o = np.array([(ob >> k) & 1 for k in range(4)], dtype=np.int64)
    o2 = Amat.dot(o) % 2
    return int(sum((int(o2[k]) & 1) << k for k in range(4)))


def solve_element4(perm, by_v, nd, candidates):
    def ms(g):
        return tuple(sorted(g))

    for Amat in candidates:
        rows = [[] for _ in range(4)]
        tgts = [[] for _ in range(4)]
        ok = True
        for v, group in by_v.items():
            v2 = frozenset(int(perm[d]) for d in v)
            tgt = by_v.get(v2)
            if tgt is None:
                ok = False
                break
            tgt_ms = ms(tgt)
            cons = [b for b in range(16)
                    if ms([(apply_A4(Amat, o) ^ b, p) for o, p in group]) == tgt_ms]
            if not cons:
                ok = False
                break
            row = np.zeros(nd, dtype=np.int8)
            for d in v2:
                row[d] = 1
            for bit in range(4):
                vals = {(b >> bit) & 1 for b in cons}
                if len(vals) == 1:
                    rows[bit].append(row)
                    tgts[bit].append(vals.pop())
        if not ok:
            continue
        X, good = [], True
        for bit in range(4):
            if rows[bit]:
                sol = solve_gf2(np.array(rows[bit]),
                                np.array(tgts[bit], dtype=np.int8))
                if sol is None:
                    good = False
                    break
                X.append(sol)
            else:
                X.append(np.zeros(nd, dtype=np.int64))
        if not good:
            continue
        X = np.stack(X)
        full = all(
            ms(by_v[frozenset(int(perm[d]) for d in v)]) ==
            ms([(apply_A4(Amat, o) ^ int(sum(
                (int(X[k][[int(perm[d]) for d in v]].sum() % 2) & 1) << k
                for k in range(4))), p) for o, p in g])
            for v, g in by_v.items())
        if full:
            return Amat, X
    return None


def main():
    sanity()
    by_v = mechanisms()
    nd = 2 * L * L
    nmech = sum(len(g) for g in by_v.values())
    log(f"toric L={L} code capacity: {nd} detectors, {nmech} mechanisms, "
        f"{len(by_v)} det-set groups")

    perms24 = [np.eye(4, dtype=np.int64)[list(pp)]
               for pp in itertools.permutations(range(4))]
    expected = {
        "T10": np.eye(4, dtype=np.int64),
        "T01": np.eye(4, dtype=np.int64),
        "ROT90": np.eye(4, dtype=np.int64)[[1, 0, 3, 2]],  # X1<->X2, Z1<->Z2
        "MIR": np.eye(4, dtype=np.int64),
    }
    out = {"L": L, "p": P, "n_mech": nmech, "elements": {}}
    for name, emap in (("T10", T10), ("T01", T01), ("ROT90", ROT), ("MIR", MIR)):
        perm = perm_from_edgemap(emap)
        sols = []
        for Amat in perms24:
            s = solve_element4(perm, by_v, nd, [Amat])
            if s is not None:
                sols.append(s)
        assert sols, f"{name}: no label permutation verifies"
        uniq = len(sols)
        Amat, X = sols[0]
        match = bool((Amat == expected[name]).all())
        chiw = [int(X[k].sum()) for k in range(4)]
        log(f"  {name}: A found by blind search (unique among 24: "
            f"{uniq == 1}), matches The END Fig.3: {match}, |chi| rows {chiw}")
        out["elements"][name] = {"A": Amat.tolist(), "unique": uniq == 1,
                                 "matches_END": match, "chi_weights": chiw}
    out["runtime_s"] = time.time() - T0
    with open("runs/toric_end_bridge.json", "w") as f:
        json.dump(out, f, indent=1)
    log(f"DONE {out['runtime_s']:.1f}s -> runs/toric_end_bridge.json")


if __name__ == "__main__":
    main()
