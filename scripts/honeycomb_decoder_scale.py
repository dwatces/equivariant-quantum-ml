"""SCALE ABLATION — answers the referee question on the learnability-switch
claim: is plain-MLP-never-learns an artifact of small capacity (63k params)?

Arms (honeycomb L=6 circuit-level task, p=0.002, same test set):
  plain_h512  @64k   (~0.5M params, 20k steps)  - does width alone wake it?
  plain_h1024 @64k   (~1.6M params, 20k steps)  - more width?
  plain_h1024 @256k  (best case for plain: width + data + steps)
  p3_9_h512   @64k   (does symmetry + scale push past 0.75?)
  p3_9_h512   @256k  (gap to MWPM 0.9471 with everything we have)

PREDICTIONS LOGGED BEFORE THE RUN (hard rule):
  P-A if wide plain LEARNS at 64k: the "switch" is capacity-conditional ->
      reframe the paper claim as capacity-and-data efficiency, honestly.
  P-B if wide plain stays near majority at 64k: the switch claim is robust
      to >=25x parameter scale and 5x step budget.
  P-C p3_9_h512 @256k improves on 0.806 (h=128) - by how much decides whether
      "architecture scale" closes or merely narrows the MWPM gap.
Output: runs/honeycomb_decoder_scale.{log,json}
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import (build_all, make_models, train_eval,
                                    matching_from, log)  # noqa: E402

import torch  # noqa: E402

T0 = time.time()
dev = "cuda" if torch.cuda.is_available() else "cpu"
c, det_meta, elements, res = build_all(L=6, M=6, R=9, p=0.008, mode="cx",
                                       s_extra=1)
nd = c.num_detectors
MLP4, GroupInv4 = make_models()

ident = (np.arange(nd), np.eye(2, dtype=np.int64), np.zeros((2, nd), np.int64))
by = {n_: (pm, A, X) for n_, pm, A, X in elements}
need = ["C3", "C3sq", "T22", "T44", "C3.T22", "C3.T44", "C3sq.T22", "C3sq.T44"]
p3_elems = [ident] + [by[n_] for n_ in need]


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
for h in (512, 1024):
    log(f"params plain h={h}: "
        f"{sum(p.numel() for p in MLP4(nd, h=h).parameters())}")
log(f"MWPM {mw:.4f} majority {maj:.4f} dev={dev}")
res.update(mwpm=mw, majority=maj, sweep={})

ARMS = [
    ("plain_h512_n64k", lambda: MLP4(nd, h=512), 64000),
    ("plain_h1024_n64k", lambda: MLP4(nd, h=1024), 64000),
    ("plain_h1024_n256k", lambda: MLP4(nd, h=1024), 256000),
    ("p3_9_h512_n64k", lambda: GroupInv4(nd, p3_elems, h=512), 64000),
    ("p3_9_h512_n256k", lambda: GroupInv4(nd, p3_elems, h=512), 256000),
]
SEEDS = (0, 1)
for name, mk, n in ARMS:
    accs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        Xtr, ytr = sample(n, seed=17 * seed + 5)
        accs.append(train_eval(mk(), Xtr, ytr, Xte, yte, dev, steps=20000))
    res["sweep"][name] = (float(np.mean(accs)), float(np.std(accs)))
    log(f"  {name}: {res['sweep'][name][0]:.4f}±{res['sweep'][name][1]:.4f}")

res["runtime_s"] = time.time() - T0
with open("runs/honeycomb_decoder_scale.json", "w") as f:
    json.dump(res, f, indent=1, default=str)
log(f"DONE {res['runtime_s']:.0f}s -> runs/honeycomb_decoder_scale.json")
