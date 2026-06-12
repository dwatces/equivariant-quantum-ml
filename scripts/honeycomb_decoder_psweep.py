"""NOISE-RATE SWEEP — closes the referee gap "is the learnability switch a
one-noise-rate artifact?" (the standard QEC-paper question our single
p=0.002 point invites).

Arms: honeycomb L=6 cx-naive MY termination, p in {0.001, 0.002, 0.004}
(the learnable band: MWPM spans ~0.99 -> ~0.62 across it), plain vs p3_9,
n in {4,000, 64,000}, 3 seeds, 4,000 steps (published protocol). MWPM +
majority recomputed at every p on fresh 200k test sets.

PREDICTIONS LOGGED BEFORE THE RUN (hard rule):
  P-A the switch persists at every p: plain stays at/near the majority
      baseline at both n while p3_9 learns (clear separation), at all
      three noise rates.
  P-B ordering p3_9 > plain holds at every (p, n); MWPM remains unbeaten
      everywhere.
  P-C (uncertain) the absolute p3_9 level falls as p rises (task mixes
      toward uniform); reported either way.
Output: runs/honeycomb_decoder_psweep.{log,json}
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import (build_all, make_models, matching_from,
                                    log)  # noqa: E402

import torch  # noqa: E402

T0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
MLP4, GroupInv4 = make_models()
NEED = ["C3", "C3sq", "T22", "T44", "C3.T22", "C3.T44", "C3sq.T22",
        "C3sq.T44"]
out = {"p_values": [0.001, 0.002, 0.004], "sweep": {}}

for p in out["p_values"]:
    c, det_meta, elements, res = build_all(L=6, M=6, R=9, p=p, mode="cx",
                                           auto_p=False, s_extra=1)
    nd = c.num_detectors
    ident = (np.arange(nd), np.eye(2, dtype=np.int64),
             np.zeros((2, nd), np.int64))
    by = {n_: (pm, A, X) for n_, pm, A, X in elements}
    assert all(n_ in by for n_ in NEED), f"p={p}: element solve incomplete"
    p3_elems = [ident] + [by[n_] for n_ in NEED]

    def sample(n, seed):
        s = c.compile_detector_sampler(seed=seed)
        ds, os_ = s.sample(n, separate_observables=True)
        y = (os_[:, 0].astype(np.int64) | (os_[:, 1].astype(np.int64) << 1))
        return ds.astype(np.float32), y

    Xte, yte = sample(200_000, seed=999)
    m, _ = matching_from(c)
    pred = m.decode_batch(Xte.astype(bool))
    mw = float(np.mean((pred[:, 0].astype(np.int64) |
                        (pred[:, 1].astype(np.int64) << 1)) == yte))
    maj = float(np.max(np.bincount(yte, minlength=4)) / len(yte))
    log(f"p={p}: MWPM {mw:.4f} majority {maj:.4f}")
    row = {"mwpm": mw, "majority": maj, "arms": {}}

    def train_eval(model, Xtr, ytr, steps=4000, bs=256, lr=1e-3):
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = torch.nn.CrossEntropyLoss()
        Xt = torch.tensor(Xtr, device=dev)
        yt = torch.tensor(ytr, device=dev, dtype=torch.long)
        n = len(ytr)
        for _ in range(steps):
            idx = torch.randint(0, n, (min(bs, n),), device=dev)
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        hits = []
        with torch.no_grad():
            for i in range(0, len(yte), 50000):
                xb = torch.tensor(Xte[i:i + 50000], device=dev)
                hits.append(model(xb).argmax(dim=1).cpu().numpy()
                            == yte[i:i + 50000])
        return float(np.mean(np.concatenate(hits)))

    for n in (4000, 64000):
        for arm, mk in (("plain", lambda: MLP4(nd)),
                        ("p3_9", lambda: GroupInv4(nd, p3_elems))):
            accs = []
            for seed in (0, 1, 2):
                torch.manual_seed(seed)
                Xtr, ytr = sample(n, seed=17 * seed + 5)
                accs.append(train_eval(mk(), Xtr, ytr))
            row["arms"][f"{arm}_n{n}"] = (float(np.mean(accs)),
                                          float(np.std(accs)))
            mu, sd = row["arms"][f"{arm}_n{n}"]
            log(f"  p={p} {arm} n={n}: {mu:.4f}±{sd:.4f}")
    out["sweep"][str(p)] = row

out["runtime_s"] = time.time() - T0
with open("runs/honeycomb_decoder_psweep.json", "w") as f:
    json.dump(out, f, indent=1)
log(f"DONE {out['runtime_s']:.0f}s -> runs/honeycomb_decoder_psweep.json")
