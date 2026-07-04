"""Artifact-#6 candidate probe (2026-07-03): is the square substrate's
tight-mode (spacing ~2a) population in the crossover sweep a REAL second
length scale, or threshold fragmentation of kernel-scale blobs over faint
lattice-scale ripple?

Discriminators, per exemplar (re-simulated exactly: same substrate, R,
seed, (mu, sigma), 400 steps, n=256):
  D1 threshold-sensitivity: n_fragments at 0.25/0.3/0.4/0.5/0.6 x max —
     a real dot array is stable in count; ripple fragmentation swings.
  D2 modulation depth: on the occupied region (A > 0.05 max), quantiles of
     A/max. A true 2a dot array is bimodal with deep zeros between dots
     (q10 ~ 0); shallow corrugation on a plateau keeps q10 high.
  D3 looking at it: rendered field + central zoom (figures/).
Verdict rule (pre-stated): if fragment count swings >3x across thresholds
AND q10/max > 0.3 on tight exemplars -> artifact (fragmentation);
if counts are threshold-stable AND inter-dot A ~ 0 -> real second phase.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lenia import Lenia, render, soup  # noqa: E402

ROOT = Path(__file__).parent.parent
SCRATCH = Path("/tmp/claude-1000/-home-daniel-Hex/"
               "f06d748f-6dcd-43f8-8b2b-31471b55950a/scratchpad")
STEPS, N = 400, 256


def fragment_count(arr: np.ndarray, substrate: str, frac: float) -> int:
    from scipy import ndimage
    if substrate == "hex":
        structure = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, nl = ndimage.label(arr > frac * arr.max(), structure=structure)
    return int(nl)


def main():
    torch.set_num_threads(2)
    picks = json.load(open(SCRATCH / "probe_picks.json"))
    dev = torch.device("cpu")
    out = []
    fields = []
    for p in picks:
        sim = Lenia(p["substrate"], n=N, R=p["R"], dev=dev)
        A = soup(256, N, seed=p["seed"], dev=dev)[p["b"]:p["b"] + 1]
        mu = torch.tensor([p["mu"]])
        sg = torch.tensor([p["sigma"]])
        A, _, _ = sim.run(A, mu, sg, STEPS, record_every=0)
        arr = A[0].cpu().numpy()
        mx = float(arr.max())
        occ = arr[arr > 0.05 * mx] / mx
        counts = {f: fragment_count(arr, p["substrate"], f)
                  for f in (0.25, 0.3, 0.4, 0.5, 0.6)}
        rec = {"tag": p["tag"], "substrate": p["substrate"], "R": p["R"],
               "seed": p["seed"], "b": p["b"], "mu": p["mu"],
               "sigma": p["sigma"], "max": mx,
               "sweep_n_spots": p["n_spots"],
               "sweep_spacing": p["spacing"],
               "frag_counts_by_thresh": counts,
               "occ_quantiles_over_max": {
                   q: float(np.quantile(occ, q / 100))
                   for q in (10, 25, 50, 75, 90)},
               "occupied_frac": float((arr > 0.05 * mx).mean())}
        out.append(rec)
        fields.append((p["tag"], render(A[0], sim, px=512).numpy(),
                       render(A[0, 64:128, 64:128].contiguous(),
                              Lenia(p["substrate"], n=64, R=p["R"], dev=dev),
                              px=256).numpy()))
        print(json.dumps(rec, indent=1), flush=True)
    (ROOT / "runs" / "hexlife_frag_probe.json").write_text(
        json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(fields), figsize=(4 * len(fields), 8))
    for k, (tag, full, zoom) in enumerate(fields):
        axes[0, k].imshow(full, cmap="magma", origin="lower")
        axes[0, k].set_title(tag, fontsize=9)
        axes[1, k].imshow(zoom, cmap="magma", origin="lower")
        axes[1, k].set_title("central 64-cell zoom", fontsize=8)
        for ax in (axes[0, k], axes[1, k]):
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "hexlife_frag_probe.png", dpi=110)
    print("figure -> figures/hexlife_frag_probe.png")


if __name__ == "__main__":
    main()
