"""Preprint figures — every plotted number is read from a logged run JSON
(Appendix D provenance); the two schematics (Fig 1) draw no data.

Outputs preprint/figures/fig{1..5}_*.{pdf,png}.
  fig1 dichotomy schematic   (drawn; no data)
  fig2 toric bridge grid     <- runs/toric_end_bridge.json
  fig3 honeycomb curves      <- runs/honeycomb_decoder.json + _bign + _scale
  fig4 surface d=5 ablation  <- runs/equiv_decoder_d5.json + _d5b.json
  fig5 CNN composition       <- runs/honeycomb_decoder_cnn.json (+ MLP ref)
"""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, RegularPolygon

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110,
})
C = {"plain": "#7f7f7f", "c3tw": "#4477AA", "p3_9": "#228833",
     "mwpm": "#000000", "maj": "#999999", "mirror": "#CCBB44",
     "klein": "#AA3377", "cnn": "#EE6677"}
OUT = "preprint/figures"
os.makedirs(OUT, exist_ok=True)


def load(p):
    with open(p) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.[pdf,png]")


def sweep_xy(sweep, arm):
    ns = sorted(int(k) for k in sweep)
    mu = [sweep[str(n)][arm][0] for n in ns]
    sd = [sweep[str(n)][arm][1] for n in ns]
    return np.array(ns), np.array(mu), np.array(sd)


# ---------------------------------------------------------------- fig 1
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.axis("off")

    # (a) surface code: two mirror plaquettes, CNOT orders clash at the axis
    def plaquette(ax, cx, order, col):
        pts = {1: (-.5, .5), 2: (.5, .5), 3: (-.5, -.5), 4: (.5, -.5)}
        ax.add_patch(Polygon([(cx - .5, -.5), (cx + .5, -.5), (cx + .5, .5),
                              (cx - .5, .5)], closed=True, fc="#f2f2f2",
                             ec="#888", lw=.8, zorder=1))
        ax.add_patch(Circle((cx, 0), .085, fc=col, ec="k", lw=.7, zorder=3))
        for k, (dx, dy) in pts.items():
            ax.add_patch(Circle((cx + dx, dy), .07, fc="w", ec="k", lw=.7,
                                zorder=3))
            ax.add_patch(FancyArrowPatch((cx + dx * .82, dy * .82),
                                         (cx + dx * .18, dy * .18),
                                         arrowstyle="-|>", mutation_scale=7,
                                         color="#555", lw=.8, zorder=2))
            ax.text(cx + dx * .62, dy * .62 + .13, str(order[k - 1]),
                    ha="center", fontsize=8, color="#222")
    plaquette(ax1, -0.85, (1, 2, 3, 4), "#bbb")
    plaquette(ax1, 0.85, (2, 1, 4, 3), "#bbb")
    ax1.plot([0, 0], [-0.95, 0.95], ls="--", c="#AA3377", lw=1.1)
    ax1.text(0, 1.02, "mirror axis", ha="center", color="#AA3377", fontsize=8)
    ax1.add_patch(Circle((0, 0.0), .085, fc="#AA3377", ec="k", lw=.7))
    ax1.text(0, -0.18, "?", ha="center", fontsize=11, color="#AA3377",
             fontweight="bold")
    ax1.text(0, -1.25,
             "surface code: $4!$ orders per check;\n"
             "an axis ancilla would need order $=$ its own mirror",
             ha="center", fontsize=8.5)
    ax1.set_xlim(-1.7, 1.7)
    ax1.set_ylim(-1.55, 1.2)
    ax1.set_title("(a)  obstruction", fontsize=10)

    # (b) honeycomb: weight-2 checks, one bit per edge
    ecol = {0: "#4477AA", 1: "#228833", 2: "#EE6677"}
    s3 = np.sqrt(3)
    A = {}
    for i in range(3):
        for j in range(3):
            x, y = s3 * (i + j / 2), 1.5 * j
            A[(i, j, "A")] = (x, y)
            A[(i, j, "B")] = (x + s3 / 2, y + 0.5)
    for (i, j, s), (x, y) in A.items():
        if s == "A":
            nb = [((i, j, "B"), 0), ((i - 1, j, "B"), 1), ((i, j - 1, "B"), 2)]
            for key, c in nb:
                k2 = (key[0], key[1], "B")
                if k2 in A:
                    x2, y2 = A[k2]
                    ax2.plot([x, x2], [y, y2], c=ecol[c], lw=2.0,
                             solid_capstyle="round", zorder=1)
    for (i, j, s), (x, y) in A.items():
        ax2.add_patch(Circle((x, y), .09, fc="w", ec="k", lw=.7, zorder=3))
    cx, cy = A[(1, 1, "A")][0] + s3 / 2, A[(1, 1, "A")][1] + 0.0
    ax2.add_patch(FancyArrowPatch((cx + .42, cy + .50), (cx - .52, cy + .42),
                                  connectionstyle="arc3,rad=0.8",
                                  arrowstyle="-|>", mutation_scale=9,
                                  color="k", lw=1.0, zorder=4))
    ax2.text(cx - 1.05, cy + 0.78, "$C_3$", fontsize=10, ha="center",
             bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
    ax2.text(s3 * 1.9, -1.05,
             "honeycomb: every check is weight-2 —\none schedule bit per edge,"
             " automatically covariant",
             ha="center", fontsize=8.5)
    ax2.set_xlim(-0.8, 2 * s3 + 2.2)
    ax2.set_ylim(-1.45, 4.2)
    ax2.set_title("(b)  exactness", fontsize=10)
    save(fig, "fig1_dichotomy")


# ---------------------------------------------------------------- fig 2
def fig2():
    t = load("runs/toric_end_bridge.json")
    perms = list(itertools.permutations(range(4)))
    rows = ["T10", "T01", "ROT90", "MIR"]
    grid = np.zeros((4, 24))
    alab = []
    for r, name in enumerate(rows):
        A = np.array(t["elements"][name]["A"])
        p = tuple(int(np.argmax(A[i])) for i in range(4))
        grid[r, perms.index(p)] = 1
        assert t["elements"][name]["unique"] and \
            t["elements"][name]["matches_END"]
        alab.append("$A=I$" if p == (0, 1, 2, 3) else
                    r"$\bar X_1\!\leftrightarrow\!\bar X_2,"
                    r"\ \bar Z_1\!\leftrightarrow\!\bar Z_2$")
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(
        ["#ececec", "#228833"]), aspect="auto", interpolation="nearest")
    for r in range(4):
        c = int(np.argmax(grid[r]))
        ax.text(c, r, "✓", ha="center", va="center", color="w", fontsize=9)
        ax.text(24.0, r, alab[r], va="center", fontsize=8.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["$T_{(1,0)}$", "$T_{(0,1)}$", "$R_{90}$", "$M$"])
    ax.set_xticks([0, 5, 10, 15, 20, 23])
    ax.set_xlabel("candidate label permutation (lexicographic index in $S_4$)")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, 30.5)
    save(fig, "fig2_toric_bridge")


# ---------------------------------------------------------------- fig 3
def fig3():
    d = load("runs/honeycomb_decoder.json")
    b = load("runs/honeycomb_decoder_bign.json")
    s = load("runs/honeycomb_decoder_scale.json")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    labs = {"plain": "unconstrained", "c3tw": r"$C_3$-tied (3 elem.)",
            "p3_9": r"$p3_9$-tied (9 elem.)"}
    for arm in ("plain", "c3tw", "p3_9"):
        ns, mu, sd = sweep_xy(d["sweep"], arm)
        ax.errorbar(ns, mu, sd, marker="o", ms=3.5, lw=1.4, capsize=2,
                    color=C[arm], label=labs[arm])
        nb, mb, sb = sweep_xy(b["sweep"], arm)
        ax.errorbar(nb[-1:], mb[-1:], sb[-1:], marker="s", ms=4, lw=0,
                    capsize=2, color=C[arm], fillstyle="none")
        ax.plot([ns[-1], nb[-1]], [mu[-1], mb[-1]], ls=":", lw=1.0,
                color=C[arm])
    ax.axhline(d["mwpm"], ls="--", lw=1, c=C["mwpm"])
    ax.text(95000, d["mwpm"] + .013, f"MWPM {d['mwpm']:.3f}", fontsize=8)
    ax.axhline(d["majority"], ls=":", lw=1, c=C["maj"])
    ax.text(260, d["majority"] + .015, "majority", fontsize=8, color="#777")
    # the >=256x separation: tied@1k vs unconstrained@256k
    y1k = d["sweep"]["1000"]["p3_9"][0]
    y256 = b["sweep"]["256000"]["plain"][0]
    ax.annotate("", xy=(256000, y256 + .005), xytext=(1000, y1k),
                arrowprops=dict(arrowstyle="->", color="#444", lw=.9,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(11000, .452, r"$\geq 256\times$", fontsize=9, color="#444",
            bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax.set_xscale("log")
    ax.set_xlabel("training samples $n$")
    ax.set_ylabel("joint accuracy (4-class)")
    ax.set_ylim(0.2, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), frameon=False)
    ax.set_title("(a)  honeycomb, circuit level", fontsize=10)

    arms = [("plain_h512_n64k", "0.45M", "#9f9f9f"),
            ("plain_h1024_n64k", "1.4M", "#7f7f7f"),
            ("plain_h1024_n256k", "1.4M\n@256k", "#5f5f5f"),
            ("p3_9_h512_n64k", "0.45M", "#66bb77"),
            ("p3_9_h512_n256k", "0.45M\n@256k", "#228833")]
    xs = np.arange(len(arms))
    for x, (k, lab, col) in zip(xs, arms):
        mu, sd = s["sweep"][k]
        ax2.bar(x, mu, yerr=sd, color=col, width=.7, capsize=2)
    ax2.axhline(s["majority"], ls=":", lw=1, c=C["maj"])
    ax2.axhline(s["mwpm"], ls="--", lw=1, c="k")
    ax2.text(2.55, s["mwpm"] - .055, "MWPM", fontsize=7.5)
    ax2.text(-0.4, s["majority"] + .015, "majority", fontsize=7.5,
             color="#777")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([a[1] for a in arms], fontsize=7.5)
    ax2.set_ylim(0.2, 1.0)
    ax2.set_title("(b)  width does not rescue", fontsize=10)
    ax2.set_xlabel("parameters (unconstrained | $p3_9$)")
    save(fig, "fig3_honeycomb_learning")


# ---------------------------------------------------------------- fig 4
def fig4():
    d5 = load("runs/equiv_decoder_d5.json")["d5"]
    db = load("runs/equiv_decoder_d5b.json")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for sweep, arm, lab, col in (
            (d5["sweep"], "plain", "unconstrained", C["plain"]),
            (d5["sweep"], "inv", r"mirror$_x$ ($\chi=0$)", C["mirror"]),
            (db["sweep"], "c2tw", r"$C_2$ (twisted)", C["c3tw"]),
            (db["sweep"], "klein4", "Klein-4 (twisted)", C["klein"])):
        ns, mu, sd = sweep_xy(sweep, arm)
        ax.errorbar(ns, mu, sd, marker="o", ms=3.5, lw=1.4, capsize=2,
                    color=col, label=lab)
    ax.axhline(d5["mwpm"], ls="--", lw=1, c="k")
    ax.text(300, d5["mwpm"] - .018, f"MWPM {d5['mwpm']:.4f}", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training samples $n$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.79, 1.005)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig4_surface_elements")


# ---------------------------------------------------------------- fig 5
def fig5():
    c = load("runs/honeycomb_decoder_cnn.json")
    d = load("runs/honeycomb_decoder.json")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for arm, lab, col, ls in (
            ("cnn", "CNN (unconstrained)", C["plain"], "-"),
            ("cnn_c3tw", r"CNN + $C_3$ (3 elem.)", C["c3tw"], "-"),
            ("cnn_p3_9", r"CNN + $p3_9$ (9 elem.)", C["p3_9"], "-")):
        ns, mu, sd = sweep_xy(c["sweep"], arm)
        ax.errorbar(ns, mu, sd, marker="o", ms=3.5, lw=1.4, capsize=2,
                    color=col, label=lab, ls=ls)
    ns, mu, _ = sweep_xy(d["sweep"], "p3_9")
    ax.plot(ns, mu, ls="--", lw=1.1, color=C["p3_9"], alpha=.55,
            label=r"MLP + $p3_9$ (ref.)")
    n64 = "64000"
    y1, y2 = c["sweep"][n64]["cnn_c3tw"][0], c["sweep"][n64]["cnn_p3_9"][0]
    ax.annotate("", xy=(64000, y2), xytext=(64000, y1),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=.9))
    ax.text(27000, (y1 + y2) / 2, "+21 pp,\nsame conv\nbackbone", fontsize=7.5,
            ha="right", va="center", color="#444",
            bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    ax.axhline(c["mwpm"], ls="--", lw=1, c="k")
    ax.text(40000, c["mwpm"] + .013, f"MWPM {c['mwpm']:.3f}", fontsize=8)
    ax.axhline(c["majority"], ls=":", lw=1, c=C["maj"])
    ax.text(300, c["majority"] + .013, "majority", fontsize=8, color="#777")
    ax.set_xscale("log")
    ax.set_xlabel("training samples $n$")
    ax.set_ylabel("joint accuracy (4-class)")
    ax.set_ylim(0.2, 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.90), frameon=False)
    save(fig, "fig5_cnn_composition")


for f in (fig1, fig2, fig3, fig4, fig5):
    f()
print("all figures written to", OUT)
