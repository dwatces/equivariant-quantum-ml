"""Validation against The END's analytic symmetry action (arXiv:2304.07362).

On their setting — toric code, code capacity, i.i.d. depolarizing — the
blind DEM solve must recover their Fig. 3 action UNIQUELY among all 24
observable permutation matrices, for all four lattice elements. This is the
ground-truth bridge: no analytic input, the action comes out of the DEM.
"""
import itertools

import numpy as np
import pytest

from demsym import solve_action

from toric_cc import build

PERMS24 = [np.eye(4, dtype=np.int64)[list(p)]
           for p in itertools.permutations(range(4))]
EXPECTED_A = {
    "T10": np.eye(4, dtype=np.int64),
    "T01": np.eye(4, dtype=np.int64),
    "ROT90": np.eye(4, dtype=np.int64)[[1, 0, 3, 2]],  # X1<->X2, Z1<->Z2
    "MIR": np.eye(4, dtype=np.int64),
}


@pytest.fixture(scope="module")
def toric():
    return build(L=6, p=0.1)


@pytest.mark.parametrize("name", ["T10", "T01", "ROT90", "MIR"])
def test_blind_solve_unique_and_matches_end(toric, name):
    dem, perms = toric
    sols = [s for A in PERMS24
            if (s := solve_action(dem, perms[name], name=name,
                                  A_candidates=[A])) is not None]
    assert len(sols) == 1, f"{name}: A not unique among 24 label perms"
    assert (sols[0].A == EXPECTED_A[name]).all()
    assert sols[0].verify(dem)


def test_translations_carry_nonzero_twist(toric):
    dem, perms = toric
    e = solve_action(dem, perms["T10"], A_candidates=[np.eye(4, dtype=np.int64)])
    assert e is not None and sum(e.chi_weights()) > 0
