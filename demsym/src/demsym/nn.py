"""PyTorch twisted-equivariant wrapper (requires the `torch` extra).

The model averages a backbone over the group with the label twist applied:

    f(x)[c] = (1/|G|) sum_g backbone(x_g)[ A_g c  XOR  X_g x_g ]

where x_g[perm_g[d]] = x[d]. With elements verified against the DEM, f is
exactly equivariant on realizable syndromes (X is gauge-fixed only on
those — probe with sampled shots, not random bit vectors).
"""
from __future__ import annotations

import numpy as np

from .element import Element, apply_A


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("demsym.nn requires torch: pip install "
                          "'demsym[torch]'") from e


def TwistedEquivariant(backbone, elements: list[Element]):
    """Wrap a logits backbone into a twisted-equivariant model.

    backbone: torch.nn.Module mapping (batch, num_detectors) floats to
    (batch, 2**k) logits, where k = num_observables. Class c is the
    bit-packed joint observable value (bit i = observable i).
    elements: the FULL group including the identity — pass the output of
    demsym.close(...).
    """
    torch = _torch()
    nn = torch.nn

    if not elements:
        raise ValueError("need at least the identity element")
    k = elements[0].num_observables
    nclass = 1 << k
    if not any(e.is_identity_perm for e in elements):
        raise ValueError("element list must include the identity "
                         "(use demsym.close, which always adds it)")

    class _TwistedEquivariant(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.num_observables = k
            invps, chis, cmaps = [], [], []
            for e in elements:
                invp = np.zeros(len(e.perm), dtype=np.int64)
                invp[e.perm] = np.arange(len(e.perm))
                invps.append(torch.tensor(invp, dtype=torch.long))
                chis.append(torch.tensor(e.X.astype(np.float32)))
                cmaps.append(torch.tensor(
                    [apply_A(e.A, c) for c in range(nclass)],
                    dtype=torch.long))
            self.register_buffer("IP", torch.stack(invps))   # (|G|, nd)
            self.register_buffer("CH", torch.stack(chis))    # (|G|, k, nd)
            self.register_buffer("CM", torch.stack(cmaps))   # (|G|, 2^k)

        def forward(self, x):
            outs = []
            for g in range(self.IP.shape[0]):
                xg = x[:, self.IP[g]]
                logit = self.backbone(xg)
                b = torch.remainder(xg @ self.CH[g].T, 2.0).long()
                bidx = torch.zeros_like(b[:, 0])
                for i in range(self.num_observables):
                    bidx = bidx | (b[:, i] << i)
                idx = self.CM[g][None, :].expand(x.shape[0], nclass) \
                    ^ bidx[:, None]
                outs.append(torch.gather(logit, 1, idx))
            return torch.stack(outs).mean(dim=0)

    return _TwistedEquivariant()


def mlp_backbone(num_detectors: int, num_observables: int, hidden: int = 128):
    """The matched-parameter 2-layer MLP used in the demsym experiments."""
    torch = _torch()
    nn = torch.nn
    return nn.Sequential(nn.Linear(num_detectors, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1 << num_observables))


def equivariance_probe(model, elements: list[Element], x) -> float:
    """Worst-case |f(x) - twisted f(x_g)| over the group, on given syndromes.

    x must be REALIZABLE syndromes (sampled shots): X carries a gauge
    freedom invisible to mechanisms, so random bit-vectors would report
    false violations. Expect ~1e-6 (float32 averaging noise) when the
    construction is correct.
    """
    torch = _torch()
    nclass = 1 << elements[0].num_observables
    x = torch.as_tensor(x, dtype=torch.float32,
                        device=next(model.parameters()).device)
    worst = 0.0
    model.eval()
    with torch.no_grad():
        fx = model(x)
        for e in elements:
            if e.is_identity_perm:
                continue
            invp = np.zeros(len(e.perm), dtype=np.int64)
            invp[e.perm] = np.arange(len(e.perm))
            xg = x[:, torch.tensor(invp, dtype=torch.long, device=x.device)]
            fg = model(xg)
            b = torch.remainder(
                xg @ torch.tensor(e.X.astype(np.float32),
                                  device=x.device).T, 2.0).long()
            bidx = torch.zeros_like(b[:, 0])
            for i in range(e.num_observables):
                bidx = bidx | (b[:, i] << i)
            cm = torch.tensor([apply_A(e.A, c) for c in range(nclass)],
                              device=x.device)
            idx = cm[None, :].expand(x.shape[0], nclass) ^ bidx[:, None]
            worst = max(worst,
                        float((fx - torch.gather(fg, 1, idx)).abs().max()))
    return worst
