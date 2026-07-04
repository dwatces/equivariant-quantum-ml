"""The missing panel of the two-regime picture: matched-symmetry
orientation locking vs spot spacing (in lattice constants), assembled from
completed runs only:
  * commensurate sweep (R 2-4), combined 10-seed summary
    -> runs/hexlife_commensurate_summary.json
  * crossover sweep (R 4.5-10), mode-split (wide = true kernel-scale
    arrays; tight = the lattice-scale morphology channel)
    -> runs/hexlife_crossover_modes.json / _summary.json
Output: figures/hexlife_crossover_locking.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent.parent
BLUE, AQUA = "#2a78d6", "#1baf7a"          # validated pair (CVD dE 73.6)
INK, MUT = "#1a1a19", "#6b6a63"

comm = json.loads((ROOT / "runs" / "hexlife_commensurate_summary.json")
                  .read_text())
modes = json.loads((ROOT / "runs" / "hexlife_crossover_modes.json")
                   .read_text())
xsum = json.loads((ROOT / "runs" / "hexlife_crossover_summary.json")
                  .read_text())

R_COMM = (2.0, 2.5, 3.0, 3.5, 4.0)
R_X = (5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0)


def series(sub, chan):
    """(spacing, lock, ci) triples: commensurate R<=4 + R=4.5 bridge +
    crossover wide mode; and the tight-mode cluster separately."""
    arr = []
    for Rv in R_COMM:
        d = comm[f"{sub}_R{Rv}"]
        arr.append((d["spacing_over_a"], d[chan][0], d[chan][1]))
    d = xsum[f"{sub}_R4.5"]
    arr.append((d["spacing_over_a"], d[chan][0], d[chan][1]))
    for Rv in R_X:
        d = modes[f"{sub}_R{Rv}_wide"]
        arr.append((d["med_spacing_a"], d[chan][0], d[chan][1]))
    arr.sort(key=lambda t: t[0])
    tight = []
    for Rv in R_X:
        d = modes[f"{sub}_R{Rv}_tight"]
        tight.append((d["med_spacing_a"], d[chan][0], d[chan][1]))
    return arr, tight


fig, (ax, axb) = plt.subplots(
    1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.9, 1.0]})
for a in (ax, axb):
    a.grid(True, alpha=0.22, linewidth=0.7)
    a.spines[["top", "right"]].set_visible(False)

ax.axhline(0, color=MUT, linewidth=1.0, zorder=1)
for sub, chan, col, mk, lab in (
        ("square", "lock4", BLUE, "s", "square · ⟨cos 4Φ₄⟩"),
        ("hex", "lock6", AQUA, "h", "hex · ⟨cos 6Φ₆⟩")):
    arr, tight = series(sub, chan)
    xs, ys, es = zip(*arr)
    ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=7 if mk == "h"
                else 6, lw=2, capsize=2.5, zorder=3, label=lab)
    tx, ty, te = zip(*tight)
    ax.errorbar(tx, ty, yerr=te, color=col, marker=mk,
                ms=7 if mk == "h" else 6, lw=0, elinewidth=1.2,
                capsize=2, markerfacecolor="white", zorder=2,
                label=f"{sub} · morphology channel (rings/texture)")
ax.set_xlabel("spot spacing (lattice constants a)")
ax.set_ylabel("matched-symmetry orientation lock")
ax.set_ylim(-1.12, 1.12)
ax.set_title("orientation locking vs spacing — arrays unlock, square by "
             "~6.5a; the ~2–3a channel stays pinned", fontsize=10,
             color=INK, loc="left")
ax.annotate("square arrays: dead ≥6.5a", xy=(7.9, -0.30),
            fontsize=8.5, color=BLUE, ha="center", va="top")
ax.annotate("hex arrays: +0.2 tail\n(box-invariant n=192/256/384)",
            xy=(8.55, 0.38), fontsize=8.5, color=AQUA, ha="center",
            va="bottom")
ax.annotate("rings/texture\npinned at 2–3a", xy=(3.2, -0.86),
            fontsize=8.5, color=MUT, ha="left", va="center")
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

for sub, col, mk in (("square", BLUE, "s"), ("hex", AQUA, "h")):
    xs = list(R_X)
    ys = [modes[f"tightprev_R{r}"][f"{sub}_mean"] for r in R_X]
    axb.plot(xs, ys, color=col, marker=mk, ms=7 if mk == "h" else 6, lw=2)
    axb.annotate(sub, xy=(xs[-1], ys[-1]), xytext=(4, 0),
                 textcoords="offset points", fontsize=9, color=col,
                 va="center")
axb.set_xlabel("kernel radius R (cells)")
axb.set_ylabel("fraction of patterns in the channel")
axb.set_ylim(0, 1)
axb.set_xlim(4.6, 11.0)
axb.set_title("lattice-scale morphology channel:\nsquare carries 2–3× "
              "more (5/5 seeds at every R)", fontsize=10, color=INK,
              loc="left")

fig.suptitle("hexlife crossover — the 3–8a panel (n=256, seeds 11–15, "
             "wide/tight split per probe)", fontsize=11, color=INK, x=0.02,
             ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "figures" / "hexlife_crossover_locking.png"
fig.savefig(out, dpi=130)
print(f"-> {out.relative_to(ROOT)}")
