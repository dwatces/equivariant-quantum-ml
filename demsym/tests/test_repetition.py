"""The 20-detector miniature of the whole story, on stim generated circuits:

- phenomenological noise: the mirror is an exact twisted symmetry, and the
  twist is REAL (A = I with X = 0 does not verify — the moved logical needs
  the syndrome-linear correction);
- circuit-level noise on the same code: the same mirror has NO solution —
  extraction-schedule errors de-symmetrize the DEM (the obstruction).
"""
import numpy as np
import pytest
import stim

from demsym import Dem, Element, identity, perm_from_coords, solve_action

D = 5


def gen(**noise):
    return stim.Circuit.generated("repetition_code:memory", distance=D,
                                  rounds=4, **noise)


def mirror(c):
    return (2.0 * (D - 1) - c[0],) + tuple(c[1:])


@pytest.fixture(scope="module")
def phenom():
    c = gen(before_round_data_depolarization=0.04,
            before_measure_flip_probability=0.01)
    return c, Dem.from_circuit(c), perm_from_coords(c, mirror)


def test_mirror_verifies_with_twist(phenom):
    c, dem, perm = phenom
    e = solve_action(dem, perm, name="mirror")
    assert e is not None and e.verify(dem)
    assert (e.A == np.eye(1, dtype=np.int64)).all()
    assert sum(e.chi_weights()) > 0
    # the twist is forced: the untwisted candidate is NOT a symmetry
    bare = Element("mirror_untwisted", perm, e.A, np.zeros_like(e.X))
    assert not bare.verify(dem)


def test_identity_always_solves(phenom):
    _, dem, _ = phenom
    e = solve_action(dem, np.arange(dem.num_detectors))
    assert e is not None and (e.A == np.eye(1, dtype=np.int64)).all()
    assert identity(dem).verify(dem)


def test_circuit_level_noise_obstructs_the_mirror():
    c = gen(after_clifford_depolarization=0.01,
            before_measure_flip_probability=0.01)
    dem = Dem.from_circuit(c)
    perm = perm_from_coords(c, mirror)
    assert solve_action(dem, perm, name="mirror") is None


def test_perm_from_coords_rejects_non_layout_maps(phenom):
    c, _, _ = phenom
    with pytest.raises(ValueError):
        perm_from_coords(c, lambda co: (co[0] + 2.0,) + tuple(co[1:]))
