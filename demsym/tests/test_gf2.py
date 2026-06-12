import numpy as np
import pytest

from demsym import Dem, gl2_matrices, solve_gf2


def test_gl_sizes():
    assert len(gl2_matrices(1)) == 1
    assert len(gl2_matrices(2)) == 6
    assert len(gl2_matrices(3)) == 168


def test_gl_identity_first_and_invertible():
    for k in (1, 2, 3):
        mats = gl2_matrices(k)
        assert (mats[0] == np.eye(k, dtype=np.int64)).all()


def test_gl_k4_requires_explicit_candidates():
    with pytest.raises(ValueError):
        gl2_matrices(4)


def test_solve_gf2_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m, n = rng.integers(3, 12, size=2)
        A = rng.integers(0, 2, size=(m, n))
        x = rng.integers(0, 2, size=n)
        b = A.dot(x) % 2
        sol = solve_gf2(A, b)
        assert sol is not None
        assert (A.dot(sol) % 2 == b).all()


def test_solve_gf2_inconsistent():
    A = np.array([[1, 0], [1, 0]])
    b = np.array([0, 1])
    assert solve_gf2(A, b) is None


def test_dem_groups_merge_by_detector_set():
    dem = Dem.from_mechanisms(
        [({0, 1}, 0b01, 0.1), ({1, 0}, 0b10, 0.2), ({2}, 0, 0.3)],
        num_detectors=3, num_observables=2)
    assert dem.num_mechanisms == 3
    assert len(dem.groups) == 2
    assert dem.groups[frozenset({0, 1})] == ((1, 0.1), (2, 0.2))
