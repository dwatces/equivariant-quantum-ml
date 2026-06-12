"""
HONEYCOMB DECODER, CNN x p3_9 (2026-06-12): compose translation weight-tying
(3D CNN over the (time, face_i, face_j) detector grid, CIRCULAR padding on the
torus dims) with the exact 4-class twisted group averaging from
honeycomb_decoder_core. Attacks the 20 pp MLP capacity wall found by the
completed MLP sweep (runs/honeycomb_decoder.json: p3_9 0.7443@64k vs MWPM 0.9471).

PREDICTIONS LOGGED BEFORE THE RUN (hard rule):
  P1 plain CNN (translation prior only, no group averaging) learns at least
     partially where the plain MLP was majority-dead — uncertain how far.
  P2 cnn_p3_9 >= cnn_c3tw >= cnn at scarce n; cnn_p3_9 closes part of the
     20 pp wall at 64k.
  P3 (open question, either answer informative): does the conv translation
     prior make the T22/T44 averaging redundant (cnn_c3tw ~ cnn_p3_9)?

Embedding (verified collision-free at build): channel 0 = face window
comparisons at t = window index k; channel 1 = s=0 init-edge detectors at t=0
placed at the A-endpoint cell of each color-0 edge (one per cell, bijective);
channel 2 = color-0 face k=0 init windows at t=0; channel 3 = final
window-vs-data comparisons at t = T-1. Torus translations act as cyclic shifts
of (i, j) -> conv with circular padding is natively equivariant; C3 + all
chi/A twists are handled exactly by the group-average wrapper (probe-gated).
"""
import time

import numpy as np

from honeycomb_decoder_core import (apply_A, build_all, build_lattice, log,
                                    matching_from, window_phase)


def det_embedding(det_meta, L, M, edges):
    """Map each detector to a (channel, t, i, j) cell; return (flat_idx, dims)."""
    ts = []
    for meta in det_meta:
        kind, key, cf, s = meta
        if kind == "face":
            ts.append((s - window_phase(cf) - 1) // 3)
    tmax_face = max(ts)
    T = tmax_face + 2  # 0=init, 1..tmax_face bulk(+virtual), T-1=final
    C = 4
    cells = []
    for meta in det_meta:
        kind, key, cf, s = meta
        if kind == "face":
            k = (s - window_phase(cf) - 1) // 3
            c, t, (i, j) = 0, k, key
        elif kind == "edge":
            va, vb, _ = edges[key]
            v = va if va[0] == "A" else vb
            c, t, (i, j) = 1, 0, (v[1], v[2])
        elif kind == "face0init":
            c, t, (i, j) = 2, 0, key
        elif kind == "facefin":
            c, t, (i, j) = 3, T - 1, key
        else:
            raise AssertionError(kind)
        cells.append((c, t, i, j))
    assert len(set(cells)) == len(cells), "embedding collision"
    flat = np.array([((c * T + t) * L + i) * M + j for c, t, i, j in cells],
                    dtype=np.int64)
    return flat, (C, T, L, M)


def make_cnn_models(flat, dims):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    C, T, L, M = dims
    ncell = C * T * L * M

    class HexCNN4(nn.Module):
        """3D CNN over (T, L, M); zero pad in t, CIRCULAR pad in (i, j)."""
        def __init__(self, nd, ch=24):
            super().__init__()
            self.register_buffer("flat", torch.tensor(flat))
            self.c1 = nn.Conv3d(C, ch, 3)
            self.c2 = nn.Conv3d(ch, ch, 3)
            self.c3 = nn.Conv3d(ch, ch, 3)
            self.proj = nn.Conv3d(ch, 8, 1)
            self.head = nn.Linear(8 * T * L * M, 4)

        def _pad(self, x):
            x = F.pad(x, (1, 1, 1, 1, 0, 0), mode="circular")  # (j, i) circular
            return F.pad(x, (0, 0, 0, 0, 1, 1))                # t zeros

        def forward(self, x):
            B = x.shape[0]
            g = torch.zeros(B, ncell, device=x.device, dtype=x.dtype)
            g.scatter_(1, self.flat[None, :].expand(B, -1), x)
            g = g.view(B, C, T, L, M)
            g = torch.relu(self.c1(self._pad(g)))
            g = torch.relu(self.c2(self._pad(g)))
            g = torch.relu(self.c3(self._pad(g)))
            g = self.proj(g)
            return self.head(g.flatten(1))

    class GroupInvCNN4(nn.Module):
        """Same exact twisted 4-class average as GroupInv4, CNN backbone."""
        def __init__(self, nd, elements, ch=24):
            super().__init__()
            self.net = HexCNN4(nd, ch)
            invps, chis, cmaps = [], [], []
            for perm, Amat, X in elements:
                invp = np.zeros(len(perm), dtype=np.int64)
                invp[perm] = np.arange(len(perm))
                invps.append(torch.tensor(invp, dtype=torch.long))
                chis.append(torch.tensor(np.asarray(X, dtype=np.float32)))
                cmaps.append(torch.tensor([apply_A(np.asarray(Amat), cc)
                                           for cc in range(4)],
                                          dtype=torch.long))
            self.register_buffer("IP", torch.stack(invps))
            self.register_buffer("CH", torch.stack(chis))
            self.register_buffer("CM", torch.stack(cmaps))

        def forward(self, x):
            outs = []
            for k in range(self.IP.shape[0]):
                xg = x[:, self.IP[k]]
                logit = self.net(xg)
                b = torch.remainder(xg @ self.CH[k].T, 2.0).long()
                bidx = b[:, 0] | (b[:, 1] << 1)
                idx = self.CM[k][None, :].expand(x.shape[0], 4) ^ bidx[:, None]
                outs.append(torch.gather(logit, 1, idx))
            return torch.stack(outs).mean(dim=0)

    return HexCNN4, GroupInvCNN4


def train_eval(model, Xtr, ytr, Xte, yte, dev, steps=4000, bs=256, lr=1e-3):
    import torch
    import torch.nn as nn
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev, dtype=torch.long)
    n = len(ytr)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=dev)
        opt.zero_grad()
        loss = lossf(model(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
    model.eval()
    hits = []
    with torch.no_grad():
        for i in range(0, len(yte), 20000):
            xb = torch.tensor(Xte[i:i + 20000], device=dev)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            hits.append(pred == yte[i:i + 20000])
    return float(np.mean(np.concatenate(hits)))


def run_experiment_cnn(L=6, M=6, R=9, p=0.008, mode="cx", s_extra=1,
                       ntrains=(250, 1000, 4000, 16000, 64000),
                       seeds=(0, 1, 2, 3, 4), ntest=200_000, steps=4000,
                       ch=24, smoke=False):
    import torch
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if smoke:
        ntrains, seeds, steps, ntest = (250,), (0,), 200, 20000
    c, det_meta, elements, res = build_all(L, M, R, p, mode, s_extra=s_extra)
    nd = c.num_detectors
    _, _, edges, _, _ = build_lattice(L, M)
    flat, dims = det_embedding(det_meta, L, M, edges)
    log(f"embedding: dims {dims}, {len(flat)} detectors placed, no collisions")
    HexCNN4, GroupInvCNN4 = make_cnn_models(flat, dims)

    ident = (np.arange(nd), np.eye(2, dtype=np.int64), np.zeros((2, nd), np.int64))
    by = {n_: (pm, A, X) for n_, pm, A, X in elements}
    need = ["C3", "C3sq", "T22", "T44", "C3.T22", "C3.T44", "C3sq.T22",
            "C3sq.T44"]
    assert all(n_ in by for n_ in need)
    c3_elems = [ident, by["C3"], by["C3sq"]]
    p3_elems = [ident] + [by[n_] for n_ in need]

    def sample(n, seed):
        s = c.compile_detector_sampler(seed=seed)
        ds, os_ = s.sample(n, separate_observables=True)
        y = (os_[:, 0].astype(np.int64) | (os_[:, 1].astype(np.int64) << 1))
        return ds.astype(np.float32), y

    # invariance probe on realizable syndromes (same gate as the MLP run)
    Xp, _ = sample(64, seed=777)
    probe = GroupInvCNN4(nd, p3_elems, ch=ch).to(dev)
    xb = torch.tensor(Xp, device=dev)
    worst = 0.0
    with torch.no_grad():
        fx = probe(xb)
        for pm, A, X in [by[n_] for n_ in need]:
            invp = np.zeros(nd, dtype=np.int64)
            invp[pm] = np.arange(nd)
            xg = xb[:, torch.tensor(invp, device=dev)]
            fg = probe(xg)
            bb = torch.remainder(
                xg @ torch.tensor(np.asarray(X, dtype=np.float32),
                                  device=dev).T, 2.0).long()
            bidx = bb[:, 0] | (bb[:, 1] << 1)
            cm = torch.tensor([apply_A(np.asarray(A), cc) for cc in range(4)],
                              device=dev)
            idx = cm[None, :].expand(xb.shape[0], 4) ^ bidx[:, None]
            worst = max(worst, float((fx - torch.gather(fg, 1, idx)).abs().max()))
    log(f"CNN twisted-average invariance probe: {worst:.2e}")
    res["invariance_probe_cnn"] = worst
    assert worst < 1e-4, "CNN equivariant construction violated"

    Xte, yte = sample(ntest, seed=999)
    m, _ = matching_from(c)
    pred = m.decode_batch(Xte.astype(bool))
    mw = float(np.mean((pred[:, 0].astype(np.int64) |
                        (pred[:, 1].astype(np.int64) << 1)) == yte))
    maj = float(np.max(np.bincount(yte, minlength=4)) / len(yte))
    nparams = sum(pp.numel() for pp in HexCNN4(nd, ch).parameters())
    log(f"test: MWPM joint {mw:.4f}, majority {maj:.4f}, params/branch "
        f"{nparams}, nd={nd}, dev={dev}")
    res.update(mwpm=mw, majority=maj, params=nparams, device=dev, sweep={})

    for n in ntrains:
        row = {}
        for arm, mkm in (("cnn", lambda: HexCNN4(nd, ch)),
                         ("cnn_c3tw", lambda: GroupInvCNN4(nd, c3_elems, ch)),
                         ("cnn_p3_9", lambda: GroupInvCNN4(nd, p3_elems, ch))):
            accs = []
            for seed in seeds:
                torch.manual_seed(seed)
                Xtr, ytr = sample(n, seed=17 * seed + 5)
                accs.append(train_eval(mkm(), Xtr, ytr, Xte, yte, dev,
                                       steps=steps))
            row[arm] = (float(np.mean(accs)), float(np.std(accs)))
        res["sweep"][n] = row
        log(f"  n={n:6d}: " + "  ".join(
            f"{a} {row[a][0]:.4f}±{row[a][1]:.4f}" for a in row))
    res["runtime_s"] = time.time() - t0
    return res
