"""Exact equivariance of the torch wrapper on realizable syndromes."""
import numpy as np
import pytest
import stim

from demsym import Dem, close, perm_from_coords, solve_action

torch = pytest.importorskip("torch")

from demsym.nn import TwistedEquivariant, equivariance_probe, mlp_backbone  # noqa: E402

D = 5


@pytest.fixture(scope="module")
def setting():
    c = stim.Circuit.generated("repetition_code:memory", distance=D, rounds=4,
                               before_round_data_depolarization=0.04,
                               before_measure_flip_probability=0.01)
    dem = Dem.from_circuit(c)
    perm = perm_from_coords(c, lambda co: (2.0 * (D - 1) - co[0],)
                            + tuple(co[1:]))
    e = solve_action(dem, perm, name="mirror")
    grp = close([e], dem)
    assert len(grp) == 2
    shots = c.compile_detector_sampler(seed=7).sample(128).astype(np.float32)
    return dem, grp, shots


def test_probe_is_exact_on_realizable_syndromes(setting):
    dem, grp, shots = setting
    torch.manual_seed(0)
    model = TwistedEquivariant(mlp_backbone(dem.num_detectors, 1, hidden=16),
                               grp)
    assert equivariance_probe(model, grp, shots) < 1e-5


def test_forward_shape_and_gradients(setting):
    dem, grp, shots = setting
    torch.manual_seed(0)
    model = TwistedEquivariant(mlp_backbone(dem.num_detectors, 1, hidden=16),
                               grp)
    x = torch.tensor(shots[:32])
    out = model(x)
    assert out.shape == (32, 2)
    loss = torch.nn.functional.cross_entropy(
        out, torch.zeros(32, dtype=torch.long))
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_requires_identity_element(setting):
    dem, grp, _ = setting
    non_id = [e for e in grp if not e.is_identity_perm]
    with pytest.raises(ValueError):
        TwistedEquivariant(mlp_backbone(dem.num_detectors, 1, hidden=16),
                           non_id)
