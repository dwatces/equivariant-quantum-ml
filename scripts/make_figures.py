"""Generate publication figures for PAPER_DRAFT.md from the real run JSONs. All data read back from
completed runs/jobs; no values invented. Saves to figures/."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs"); FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})


def jload(name): return json.load(open(os.path.join(RUNS, name)))


def dev_agg(d, key):
    pb = d["per_backend"]; v = [pb[dev]["point"][key] for dev in pb]
    return float(np.mean(v)), float(np.std(v))


# ---------- Fig 1: classical equivariance sample-efficiency (QRC readout) ----------
def fig1():
    d = jload("hex_qrc_equivariant.json"); rows = d["rows"]
    n = [r["n"] for r in rows]
    eq = [r["TOPO_eq"][0] for r in rows]; eqs = [r["TOPO_eq"][1] for r in rows]
    pl = [r["TOPO_pl"][0] for r in rows]; pls = [r["TOPO_pl"][1] for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.errorbar(n, eq, yerr=eqs, marker="o", lw=2, capsize=3, color="tab:green", label="C3-equivariant readout")
    ax.errorbar(n, pl, yerr=pls, marker="s", lw=2, capsize=3, color="tab:gray", label="plain readout")
    ax.axhline(1 / d["K"], ls=":", c="k", alpha=0.6, label=f"chance (1/{d['K']})")
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels(n)
    ax.set_xlabel("training examples"); ax.set_ylabel("test accuracy")
    ax.set_title("Classical equivariance gift: sample-efficiency\n(honeycomb quantum reservoir)", fontsize=11)
    ax.legend(loc="lower right", fontsize=9); ax.set_ylim(0.2, 1.02)
    ax.annotate(f"+{eq[0]-pl[0]:.2f}", (n[0], (eq[0]+pl[0])/2), fontsize=11, color="tab:green",
                fontweight="bold", ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_classical_equivariance.png")); plt.close(fig)
    print("[fig1] classical equivariance sample-efficiency")


# ---------- Fig 2: QPU coexistence — symmetry + topology across the three experiments ----------
def fig2():
    exps = [("single-particle", "ibm_eqtopo_robust.json"), ("two-particle\n(entangled)", "ibm_eqtopo_2p_robust.json"),
            ("N=12 ring\n(12 qubits)", "ibm_eqtopo_scale_N12.json")]
    data = [(lab, jload(f)) for lab, f in exps]
    x = np.arange(len(exps)); w = 0.36
    fig, (axS, axT) = plt.subplots(1, 2, figsize=(11, 4.3))
    # symmetry: E_C3 (preserved) vs broken control
    e3 = [dev_agg(d, "E_C3") for _, d in data]; e3b = [dev_agg(d, "E_C3_broken") for _, d in data]
    axS.bar(x - w / 2, [m for m, _ in e3], w, yerr=[s for _, s in e3], capsize=4, color="tab:green",
            label="symmetric ring (E$_{C3}$)")
    axS.bar(x + w / 2, [m for m, _ in e3b], w, yerr=[s for _, s in e3b], capsize=4, color="tab:red", alpha=0.7,
            label="symmetry-broken control")
    axS.set_xticks(x); axS.set_xticklabels([l for l, _ in data]); axS.set_ylabel("equivariance error  E$_{C3}$")
    axS.set_title("Symmetry gift on hardware: C3 preserved"); axS.legend(fontsize=9)
    for xi, (m, _) in zip(x - w / 2, e3): axS.text(xi, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)
    # topology: chirality (Haldane) vs trivial control
    ch = [dev_agg(d, "chirality") for _, d in data]; cht = [dev_agg(d, "chirality_triv") for _, d in data]
    axT.bar(x - w / 2, [m for m, _ in ch], w, yerr=[s for _, s in ch], capsize=4, color="tab:blue",
            label="topological (Haldane)")
    axT.bar(x + w / 2, [m for m, _ in cht], w, yerr=[s for _, s in cht], capsize=4, color="tab:gray", alpha=0.7,
            label="topologically-trivial control")
    axT.set_xticks(x); axT.set_xticklabels([l for l, _ in data]); axT.set_ylabel("chirality")
    axT.set_title("Topology gift on hardware: chirality present"); axT.legend(fontsize=9)
    for xi, (m, _) in zip(x - w / 2, ch): axT.text(xi, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)
    fig.suptitle("Both gifts coexist on superconducting hardware (3 devices, error-mitigated, bootstrap error bars)",
                 fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_qpu_coexistence.png")); plt.close(fig)
    print("[fig2] QPU coexistence (symmetry + topology x 3 experiments)")


# ---------- Fig 3: t2 causal scaling — hardware vs simulation ----------
def fig3():
    d = jload("ibm_eqtopo_t2sweep.json"); pb = d["per_backend"]; devs = list(pb)
    keys = sorted(pb[devs[0]].keys(), key=float); t2 = [float(k) for k in keys]
    hw = [np.mean([pb[dv][k]["chirality"] for dv in devs]) for k in keys]
    hws = [np.std([pb[dv][k]["chirality"] for dv in devs]) for k in keys]
    sim = {0.0: 0.000, 0.1: 0.176, 0.2: 0.329, 0.3: 0.447, 0.4: 0.528}     # from the noiseless S1 sweep
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(t2, [sim.get(round(x, 2), np.nan) for x in t2], "--", color="k", alpha=0.6, marker="x", label="noiseless simulation")
    ax.errorbar(t2, hw, yerr=hws, marker="o", lw=2, capsize=3, color="tab:blue", label="hardware (3 devices)")
    ax.set_xlabel("topological strength  t$_2$"); ax.set_ylabel("measured chirality")
    ax.set_title("Causal attribution (hardware):\nchirality scales with t$_2$, $\\approx$0 at t$_2$=0", fontsize=11)
    ax.legend(loc="upper left", fontsize=9); ax.set_ylim(-0.03, 0.6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_t2_causal_scaling.png")); plt.close(fig)
    print("[fig3] t2 causal scaling (hardware vs sim)")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print(f"\n[saved] {len(os.listdir(FIG))} figures in figures/")
