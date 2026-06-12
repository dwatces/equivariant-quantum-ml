"""Twisted symmetry elements of a detector error model.

An element acts on (syndrome, label) as

    x  ->  x_g            with  x_g[perm[d]] = x[d]
    c  ->  A c  XOR  X x_g

where c is the vector of logical-observable bits, A is invertible over
GF(2), and X (one row per observable) is a syndrome-linear correction.
The element is a symmetry when this map sends the DEM's mechanism multiset
to itself exactly (see `Element.verify`).

Gauge note: X is determined only modulo directions invisible to every
mechanism detector set, so two correct X's may differ as vectors while
acting identically on realizable syndromes. All checks here are therefore
multiset checks against the DEM, never X-equality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dem import Dem


def apply_A(Amat: np.ndarray, ob: int) -> int:
    """Apply a GF(2) matrix to a bit-packed observable mask."""
    k = Amat.shape[0]
    o = np.array([(ob >> i) & 1 for i in range(k)], dtype=np.int64)
    o2 = Amat.dot(o) % 2
    return int(sum((int(o2[i]) & 1) << i for i in range(k)))


@dataclass
class Element:
    """A verified-or-candidate twisted symmetry element."""
    name: str
    perm: np.ndarray   # (nd,) int64, detector d -> perm[d]
    A: np.ndarray      # (k, k) int64 over GF(2)
    X: np.ndarray      # (k, nd) int64 over GF(2)

    def __post_init__(self):
        self.perm = np.asarray(self.perm, dtype=np.int64)
        self.A = np.asarray(self.A, dtype=np.int64) % 2
        self.X = np.asarray(self.X, dtype=np.int64) % 2

    @property
    def num_detectors(self) -> int:
        return len(self.perm)

    @property
    def num_observables(self) -> int:
        return self.A.shape[0]

    @property
    def is_identity_perm(self) -> bool:
        return bool((self.perm == np.arange(len(self.perm))).all())

    def chi_weights(self) -> list[int]:
        return [int(r.sum()) for r in self.X]

    def __matmul__(self, first: "Element") -> "Element":
        """self @ first = apply `first`, then `self` (function composition)."""
        if len(first.perm) != len(self.perm):
            raise ValueError("element sizes differ")
        perm = self.perm[first.perm]
        A = (self.A @ first.A) % 2
        # X of `first` acts on the intermediate syndrome x_1; rewrite it on
        # the final syndrome x_12 (x_1[e] = x_12[self.perm[e]]), then mix
        # its rows through self.A and add self's own correction.
        Y = np.zeros_like(first.X)
        Y[:, self.perm] = first.X
        X = ((self.A @ Y) + self.X) % 2
        return Element(f"{self.name}.{first.name}", perm, A, X)

    def verify(self, dem: Dem) -> bool:
        """Exact multiset check: does this element map the DEM to itself?"""
        perm = self.perm
        for v, group in dem.groups.items():
            v2 = frozenset(int(perm[d]) for d in v)
            tgt = dem.groups.get(v2)
            if tgt is None:
                return False
            idx = [int(perm[d]) for d in v]
            b = 0
            for i in range(self.num_observables):
                b |= int(self.X[i][idx].sum() % 2) << i
            img = sorted((apply_A(self.A, o) ^ b, p) for o, p in group)
            if tuple(img) != tgt:
                return False
        return True

    def __repr__(self):
        a = "I" if (self.A == np.eye(self.num_observables,
                                     dtype=np.int64)).all() \
            else self.A.tolist()
        return (f"Element({self.name!r}, nd={self.num_detectors}, "
                f"A={a}, |chi|={self.chi_weights()})")


def identity(dem: Dem, name: str = "I") -> Element:
    """The identity element for a given DEM."""
    k = dem.num_observables
    return Element(name, np.arange(dem.num_detectors),
                   np.eye(k, dtype=np.int64),
                   np.zeros((k, dem.num_detectors), dtype=np.int64))
