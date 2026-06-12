"""Portfolio-grade figures for the Kitaev topological-order QPU program (2026-06-07).
Reads runs/ibm_shore_up.json (braiding + flux, 3 devices, error bars) and
runs/ibm_kitaev_localize_result.json (localized anyon). Writes figures/kitaev_*.png.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs"); FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(p):
    fp = os.path.join(RUNS, p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def main():
    shore = load("ibm_shore_up.json")
    loc = load("ibm_kitaev_localize_result.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # ---- Panel A: anyon braiding (-1 mutual statistics), 3 devices ----
    ax = axes[0]
    if shore and "braiding" in shore:
        b = shore["braiding"]
        labs = ["vacuum", "braid", "control"]
        means = [b[l]["mean"] for l in labs]
        stds = [b[l]["std"] for l in labs]
        cols = ["#4477aa", "#cc3311", "#228833"]
        ax.bar(labs, means, yerr=stds, capsize=6, color=cols, alpha=0.85)
        nd = b["braid"]["n_dev"]
        ax.set_title(f"Anyon braiding: -1 mutual statistics\n(toric code, {nd} devices, DD)")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel(r"$\langle \bar Z \rangle$")
    ax.set_ylim(-1.1, 1.1); ax.axhline(1, ls=":", c="gray", lw=0.8); ax.axhline(-1, ls=":", c="gray", lw=0.8)
    ax.annotate("braid flips sign\n= -1 phase", (1, -0.5), ha="center", fontsize=9, color="#cc3311")

    # ---- Panel B: Kitaev Z2 flux conservation, 3 devices ----
    ax = axes[1]
    if shore and "flux" in shore:
        f = shore["flux"]
        labs = ["t0", "kitaev_s2", "field_s2"]
        disp = ["t=0", "Kitaev\n(2 steps)", "Kitaev+field\n(control)"]
        means = [f[l]["mean"] for l in labs]
        stds = [f[l]["std"] for l in labs]
        ax.bar(disp, means, yerr=stds, capsize=6,
               color=["#4477aa", "#4477aa", "#cc3311"], alpha=0.85)
        nd = f["kitaev_s2"]["n_dev"]
        ax.set_title(f"Z2 plaquette flux: conserved by Kitaev,\nbroken by field ({nd} devices, DD)")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel(r"$\langle W_p \rangle$")
    ax.set_ylim(-0.1, 1.1); ax.axhline(1, ls=":", c="gray", lw=0.8)

    # ---- Panel C: localized anyon (3 devices if available, else single) ----
    ax = axes[2]
    locm = load("ibm_shore_localize.json")
    if locm and "vis_Wp1" in locm:
        groups = ["vacuum", "vison (after Y0)"]
        wp1 = [locm["vac_Wp1"]["mean"], locm["vis_Wp1"]["mean"]]
        wp2 = [locm["vac_Wp2"]["mean"], locm["vis_Wp2"]["mean"]]
        e1 = [locm["vac_Wp1"]["std"], locm["vis_Wp1"]["std"]]
        e2 = [locm["vac_Wp2"]["std"], locm["vis_Wp2"]["std"]]
        x = np.arange(2); w = 0.35
        ax.bar(x - w / 2, wp1, w, yerr=e1, capsize=5, label=r"$\langle W_{p1}\rangle$ (anyon site)", color="#cc3311", alpha=0.85)
        ax.bar(x + w / 2, wp2, w, yerr=e2, capsize=5, label=r"$\langle W_{p2}\rangle$ (neighbour)", color="#228833", alpha=0.85)
        nd = locm["vis_Wp1"]["n_dev"]
        ax.set_xticks(x); ax.set_xticklabels(groups); ax.legend(fontsize=8)
        ax.set_title(f"Localized anyon: local op flips p1,\nneighbour p2 untouched ({nd} devices)")
    elif loc:
        groups = ["vacuum", "vison (after Y0)"]
        wp1 = [loc["vacuum"]["Wp1"], loc["vison"]["Wp1"]]
        wp2 = [loc["vacuum"]["Wp2"], loc["vison"]["Wp2"]]
        x = np.arange(2); w = 0.35
        ax.bar(x - w / 2, wp1, w, label=r"$\langle W_{p1}\rangle$ (anyon site)", color="#cc3311", alpha=0.85)
        ax.bar(x + w / 2, wp2, w, label=r"$\langle W_{p2}\rangle$ (neighbour)", color="#228833", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(groups); ax.legend(fontsize=8)
        ax.set_title("Localized anyon: local op flips p1,\nneighbour p2 untouched (hardware)")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("flux"); ax.set_ylim(-1.1, 1.1)
    ax.axhline(1, ls=":", c="gray", lw=0.8); ax.axhline(-1, ls=":", c="gray", lw=0.8)

    fig.suptitle("Honeycomb Kitaev topological order on IBM quantum hardware (2026-06-07)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "kitaev_topological_order.png")
    fig.savefig(out, dpi=140)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
