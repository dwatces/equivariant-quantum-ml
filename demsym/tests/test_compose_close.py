"""Composition law and verified group closure on the toric setting."""
import numpy as np
import pytest

from demsym import close, identity, solve_action

from toric_cc import build


@pytest.fixture(scope="module")
def toric4():
    return build(L=4, p=0.1)


def solved(dem, perms, name, cands):
    e = solve_action(dem, perms[name], name=name, A_candidates=cands)
    assert e is not None
    return e


def test_translation_closure_is_the_full_torus_group(toric4):
    dem, perms = toric4
    I4 = [np.eye(4, dtype=np.int64)]
    t10 = solved(dem, perms, "T10", I4)
    t01 = solved(dem, perms, "T01", I4)
    grp = close([t10, t01], dem)
    assert len(grp) == 16  # Z_4 x Z_4, identity included
    assert all((e.A == np.eye(4, dtype=np.int64)).all() for e in grp)


def test_rotation_powers_close_to_order_4(toric4):
    dem, perms = toric4
    rot = solved(dem, perms, "ROT90",
                 [np.eye(4, dtype=np.int64)[[1, 0, 3, 2]]])
    grp = close([rot], dem)
    assert len(grp) == 4
    r2 = rot @ rot
    assert r2.verify(dem)
    # C2 inverts both axes: A back to identity
    assert (r2.A == np.eye(4, dtype=np.int64)).all()
    r4 = r2 @ r2
    assert r4.is_identity_perm
    assert (r4.A == np.eye(4, dtype=np.int64)).all()
    # X of the full cycle may be pure gauge; it must still verify as identity
    assert r4.verify(dem)


def test_composition_is_associative_on_perms_and_A(toric4):
    dem, perms = toric4
    I4 = [np.eye(4, dtype=np.int64)]
    t10 = solved(dem, perms, "T10", I4)
    t01 = solved(dem, perms, "T01", I4)
    rot = solved(dem, perms, "ROT90",
                 [np.eye(4, dtype=np.int64)[[1, 0, 3, 2]]])
    a = (rot @ t10) @ t01
    b = rot @ (t10 @ t01)
    assert (a.perm == b.perm).all() and (a.A == b.A).all()
    assert a.verify(dem) and b.verify(dem)


def test_identity_neutral(toric4):
    dem, perms = toric4
    I4 = [np.eye(4, dtype=np.int64)]
    t10 = solved(dem, perms, "T10", I4)
    e = identity(dem)
    assert ((e @ t10).perm == t10.perm).all()
    assert ((t10 @ e).perm == t10.perm).all()
    assert (t10 @ e).verify(dem)
