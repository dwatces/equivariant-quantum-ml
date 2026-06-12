"""Shared toric-code code-capacity construction (The END's setting).

L x L torus, qubits on edges; A(i,j) X-stabilizers are detectors 0..L^2-1,
B(i,j) Z-stabilizers are detectors L^2..2L^2-1. Four logical observables
gamma = (X1, X2, Z1, Z2) as bits 0..3. Mechanisms: X, Z, Y on each edge at
p/3 (i.i.d. depolarizing, perfect syndromes).
"""
from __future__ import annotations

import numpy as np

from demsym import Dem


def build(L: int = 6, p: float = 0.1):
    def vdet(i, j):
        return (i % L) * L + (j % L)

    def pdet(i, j):
        return L * L + (i % L) * L + (j % L)

    def edge_dets_X(e):
        t, i, j = e
        if t == "h":
            return {pdet(i, j), pdet(i - 1, j)}
        return {pdet(i, j), pdet(i, j - 1)}

    def edge_dets_Z(e):
        t, i, j = e
        if t == "h":
            return {vdet(i, j), vdet(i, j + 1)}
        return {vdet(i, j), vdet(i + 1, j)}

    edges = [(t, i, j) for i in range(L) for j in range(L) for t in ("h", "v")]
    XBAR1 = {("h", i, 0) for i in range(L)}
    XBAR2 = {("v", 0, j) for j in range(L)}
    ZBAR1 = {("h", 0, j) for j in range(L)}
    ZBAR2 = {("v", i, 0) for i in range(L)}

    def obs_of(e, pauli):
        o = 0
        if pauli in ("Z", "Y"):
            o |= (e in XBAR1) << 0
            o |= (e in XBAR2) << 1
        if pauli in ("X", "Y"):
            o |= (e in ZBAR1) << 2
            o |= (e in ZBAR2) << 3
        return o

    mechs = []
    for e in edges:
        for pauli in "XZY":
            dets = set()
            if pauli in ("X", "Y"):
                dets ^= edge_dets_X(e)
            if pauli in ("Z", "Y"):
                dets ^= edge_dets_Z(e)
            mechs.append((dets, obs_of(e, pauli), p / 3))
    dem = Dem.from_mechanisms(mechs, 2 * L * L, 4)

    def perm_from_edgemap(emap):
        vsets, psets = {}, {}
        for i in range(L):
            for j in range(L):
                vs = frozenset({("h", i, j), ("h", i, (j - 1) % L),
                                ("v", i, j), ("v", (i - 1) % L, j)})
                ps = frozenset({("h", i, j), ("h", (i + 1) % L, j),
                                ("v", i, j), ("v", i, (j + 1) % L)})
                vsets[vs] = vdet(i, j)
                psets[ps] = pdet(i, j)
        perm = np.zeros(2 * L * L, dtype=np.int64)
        for sets in (vsets, psets):
            for s, d in sets.items():
                img = frozenset(emap(e) for e in s)
                tgt = vsets.get(img, psets.get(img))
                assert tgt is not None, "edge map is not a lattice automorphism"
                perm[d] = tgt
        assert len(set(perm.tolist())) == 2 * L * L
        return perm

    def T10(e):
        t, i, j = e
        return (t, (i + 1) % L, j)

    def T01(e):
        t, i, j = e
        return (t, i, (j + 1) % L)

    def ROT(e):
        t, i, j = e
        if t == "h":
            return ("v", j % L, (-i) % L)
        return ("h", j % L, (-i - 1) % L)

    def MIR(e):
        t, i, j = e
        if t == "h":
            return ("h", i, (-j - 1) % L)
        return ("v", i, (-j) % L)

    perms = {name: perm_from_edgemap(f)
             for name, f in (("T10", T10), ("T01", T01),
                             ("ROT90", ROT), ("MIR", MIR))}
    return dem, perms
