"""Mechanism test for the lattice-scale morphology channel (2026-07-04).

FINDING to explain: at R=5-10 the square substrate sustains 2-3x more
lattice-scale (~2-3a) morphology (rings / micro-texture) than hex
(prevalence sq 86->34% vs hex 30->7%, 5/5 seeds every R).

HYPOTHESIS (from E1): the lattice SAMPLING of the continuous ring
kernel leaves discretization residue at short wavelengths — a
secondary band in the kernel's transfer function K^(k) near the
lattice scale. Lenia's linear response around a uniform state grows
modes in proportion to G'(U0) * K^(k), so a stronger short-wavelength
band = a stronger available channel for lattice-scale texture. The
square Cartesian comb should alias the ring more strongly than the
(more isotropic) hex comb.

TEST (analytical, no dynamics): build the exact kernels used by the
sweeps (lenia.Lenia), FFT to K^(k) on the true reciprocal lattice,
and measure the SECONDARY BAND strength = max |K^| over wavelengths
1.5a-3.5a, normalized by the primary band max |K^| at wavelengths
> 3.5a. Compare square vs hex across R = 4.5-10, and CORRELATE with
the measured tight-mode prevalence from the crossover sweep.

PRE-REGISTERED: (i) secondary/primary ratio square > hex at every
R >= 5; (ii) the ratio's R-dependence tracks the measured prevalence
(both decline with R). FALSIFIER: ratio ~ equal across substrates, or
anti-correlated with prevalence -> hypothesis dead, channel needs a
different mechanism (e.g. nonlinear saturation, not linear response).

Honest bound: linear-response logic around uniform U0; the (mu,sigma)
dependence enters through G' and is NOT modeled — this supports or
kills the AVAILABILITY of the channel, it does not predict which
params enter it.

Run: python3 hexlife/kernel_spectrum.py
Out: runs/hexlife_kernel_spectrum.json + figures/hexlife_kernel_spectrum.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lenia import BASES, HEX_A, Lenia  # noqa: E402

ROOT = Path(__file__).parent.parent
N = 256
R_LIST = (4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0)
A_LAT = {"square": 1.0, "hex": HEX_A}

# measured tight-mode prevalence (crossover sweep, 5 seeds, n=256)
PREVALENCE = {
    "square": {5.0: 0.86, 5.5: 0.86, 6.0: 0.78, 7.0: 0.70, 8.0: 0.58,
               9.0: 0.47, 10.0: 0.34},
    "hex": {5.0: 0.30, 5.5: 0.24, 6.0: 0.34, 7.0: 0.36, 8.0: 0.23,
            9.0: 0.17, 10.0: 0.07},
}


def k_magnitudes(name: str, n: int) -> np.ndarray:
    """|k| for every FFT mode on the substrate's TRUE reciprocal
    lattice: mode (mi, mj) -> k = (mi/n) g1 + (mj/n) g2, minimum-image
    over neighbouring reciprocal cells (the FFT indexes fractional
    reciprocal coords; the physical |k| needs the reciprocal metric)."""
    B = BASES[name].numpy().astype(np.float64)   # rows b1, b2
    # position of cell (i,j) = i*b1 + j*b2; plane wave e^{i k.r} has
    # FFT index m when k = (m1/n) g1 + (m2/n) g2 with gi.bj = 2pi dij
    # -> G rows: g1, g2 with G = 2pi (B^{-1})^T
    G = 2 * math.pi * np.linalg.inv(B).T
    assert np.allclose(G @ B.T, 2 * math.pi * np.eye(2), atol=1e-9)
    idx = np.fft.fftfreq(n) * n      # integer mode numbers
    MI, MJ = np.meshgrid(idx, idx, indexing="xy")
    best = None
    for wi in (-n, 0, n):
        for wj in (-n, 0, n):
            kx = (MI + wi) / n * G[0, 0] + (MJ + wj) / n * G[1, 0]
            ky = (MI + wi) / n * G[0, 1] + (MJ + wj) / n * G[1, 1]
            k = np.sqrt(kx ** 2 + ky ** 2)
            best = k if best is None else np.minimum(best, k)
    return best


def band_ratio(name: str, R: float, n: int = N):
    sim = Lenia(name, n=n, R=R, dev=torch.device("cpu"))
    K = sim.kernel.cpu().numpy()
    Kf = np.abs(np.fft.fft2(K))
    kmag = k_magnitudes(name, n)
    a = A_LAT[name]
    lam = np.full_like(kmag, np.inf)
    nz = kmag > 1e-12
    lam[nz] = 2 * math.pi / kmag[nz]
    primary = float(Kf[(lam > 3.5 * a) & (lam < 100 * a)].max())
    secondary = float(Kf[(lam >= 1.5 * a) & (lam <= 3.5 * a)].max())
    return secondary / primary, secondary, primary


def main():
    out = {}
    print(f"{'sub':7s} {'R':5s} {'sec/prim':9s} {'prevalence':10s}")
    for name in ("square", "hex"):
        for R in R_LIST:
            ratio, sec, prim = band_ratio(name, R)
            prev = PREVALENCE[name].get(R)
            out[f"{name}_R{R}"] = {"ratio": ratio, "secondary": sec,
                                   "primary": prim, "prevalence": prev}
            print(f"{name:7s} {R:<5} {ratio:<9.4f} "
                  f"{'' if prev is None else f'{prev:.2f}'}")
    # correlation across R>=5 per substrate + pooled
    xs, ys = [], []
    for name in ("square", "hex"):
        rr = [out[f"{name}_R{R}"]["ratio"] for R in R_LIST if R >= 5]
        pp = [PREVALENCE[name][R] for R in R_LIST if R >= 5]
        c = np.corrcoef(rr, pp)[0, 1]
        out[f"corr_{name}"] = float(c)
        print(f"corr(ratio, prevalence) {name}: r = {c:+.3f}  (n=7)")
        xs += rr
        ys += pp
    c = np.corrcoef(xs, ys)[0, 1]
    out["corr_pooled"] = float(c)
    print(f"corr pooled (both substrates): r = {c:+.3f}  (n=14)")
    sq_gt = sum(1 for R in R_LIST if R >= 5
                and out[f"square_R{R}"]["ratio"] > out[f"hex_R{R}"]["ratio"])
    out["square_gt_hex_of7"] = sq_gt
    print(f"square ratio > hex ratio at {sq_gt}/7 R values")
    (ROOT / "runs" / "hexlife_kernel_spectrum.json").write_text(
        json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, AQUA = "#2a78d6", "#1baf7a"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax in (a1, a2):
        ax.grid(True, alpha=0.22, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    for name, col, mk in (("square", BLUE, "s"), ("hex", AQUA, "h")):
        rs = [R for R in R_LIST]
        a1.plot(rs, [out[f"{name}_R{R}"]["ratio"] for R in rs], color=col,
                marker=mk, ms=6, lw=2, label=name)
        rs5 = [R for R in R_LIST if R >= 5]
        a2.scatter([out[f"{name}_R{R}"]["ratio"] for R in rs5],
                   [PREVALENCE[name][R] for R in rs5], color=col,
                   marker=mk, s=45, label=name)
    a1.set_xlabel("kernel radius R (cells)")
    a1.set_ylabel("secondary/primary band ratio of |K^|")
    a1.set_title("short-wavelength (1.5–3.5a) residue in the sampled\n"
                 "kernel's transfer function", fontsize=10, loc="left")
    a1.legend(fontsize=8.5)
    a2.set_xlabel("secondary/primary band ratio")
    a2.set_ylabel("measured tight-mode prevalence")
    a2.set_title("mechanism check: linear-response channel strength\n"
                 "vs observed morphology prevalence (R≥5)",
                 fontsize=10, loc="left")
    a2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "hexlife_kernel_spectrum.png", dpi=130)
    print("-> figures/hexlife_kernel_spectrum.png")


if __name__ == "__main__":
    main()
