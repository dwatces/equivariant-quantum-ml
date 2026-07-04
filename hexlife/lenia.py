"""hexlife — Lenia-class continuous cellular automata on arbitrary Bravais
lattices, built to answer the emergence corner of the six-year question:

    Learning routes around geometry (verdict 1, closed). Self-organization
    CANNOT — there is no optimizer. Does the hexagonal substrate support
    measurably different / richer emergent dynamics than the square, at
    matched cell density and identical physical rule?

Design (the control discipline):
  * ONE implementation. A substrate is just a 2x2 basis matrix B; cells sit
    at integer combinations i*b1 + j*b2 on a torus (periodic parallelogram).
      square: b1=(1,0),       b2=(0,1)          (area/cell = 1)
      hex:    b1=(a,0),       b2=(a/2, a√3/2)   with a=√(2/√3) ≈ 1.0746
              (area/cell = a²√3/2 = 1 — density-matched to the square)
  * The kernel is a CONTINUOUS ring K(r) sampled at each lattice offset's
    true Euclidean (minimum-image) distance, normalized to sum 1. The rule
    (Lenia growth) is identical. Only the sampling lattice differs — the
    experiment isolates discretization geometry.
  * Convolution is exact on any Bravais lattice via FFT (index-space
    translation invariance holds; offset-row representations would break
    it, axial/sheared coords do not).

Self-test: python3 lenia.py --selftest  (uses CUDA if available)
"""

from __future__ import annotations

import math

import torch

SQRT3 = math.sqrt(3.0)
HEX_A = math.sqrt(2.0 / SQRT3)          # density-matched hex lattice constant

BASES = {
    "square": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    "hex":    torch.tensor([[HEX_A, 0.0], [HEX_A / 2, HEX_A * SQRT3 / 2]]),
}


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- geometry -----------------------------------

def offset_distances(n: int, basis: torch.Tensor) -> torch.Tensor:
    """(n,n) Euclidean distance from cell (0,0) to every wrapped offset.

    Minimum-image on the torus: for each index offset take the closest of
    the 9 neighbouring parallelogram images (sufficient for the shears here).
    """
    idx = torch.arange(n)
    di = ((idx + n // 2) % n - n // 2).float()          # wrapped to [-n/2, n/2)
    dj = di.clone()
    J, I = torch.meshgrid(dj, di, indexing="ij")        # row=j (b2), col=i (b1)
    best = None
    for wj in (-n, 0, n):
        for wi in (-n, 0, n):
            x = (I + wi) * basis[0, 0] + (J + wj) * basis[1, 0]
            y = (I + wi) * basis[0, 1] + (J + wj) * basis[1, 1]
            d = torch.sqrt(x * x + y * y)
            best = d if best is None else torch.minimum(best, d)
    return best


def cell_positions(n: int, basis: torch.Tensor) -> torch.Tensor:
    """(n,n,2) cartesian positions of cells (for rendering/rotation)."""
    idx = torch.arange(n).float()
    J, I = torch.meshgrid(idx, idx, indexing="ij")
    x = I * basis[0, 0] + J * basis[1, 0]
    y = I * basis[0, 1] + J * basis[1, 1]
    return torch.stack([x, y], dim=-1)


# ------------------------------ the CA ------------------------------------

class Lenia:
    """Batched Lenia on a Bravais lattice. State: (B, n, n) in [0,1]."""

    def __init__(self, substrate: str, n: int = 128, R: float = 10.0,
                 r0: float = 0.5, w: float = 0.15, dt: float = 0.1,
                 dev: torch.device | None = None):
        self.substrate, self.n, self.R, self.dt = substrate, n, R, dt
        self.dev = dev or device()
        basis = BASES[substrate]
        d = offset_distances(n, basis)
        shell = torch.exp(-0.5 * ((d - r0 * R) / (w * R)) ** 2)
        shell[d > R] = 0.0
        shell[0, 0] = 0.0                                # no self-coupling
        self.kernel = (shell / shell.sum()).to(self.dev)
        self.Kf = torch.fft.rfft2(self.kernel)
        self.basis = basis

    def step(self, A: torch.Tensor, mu: torch.Tensor,
             sigma: torch.Tensor) -> torch.Tensor:
        U = torch.fft.irfft2(torch.fft.rfft2(A) * self.Kf, s=(self.n, self.n))
        G = 2.0 * torch.exp(-0.5 * ((U - mu) / sigma) ** 2) - 1.0
        return (A + self.dt * G).clamp_(0.0, 1.0)

    def run(self, A: torch.Tensor, mu, sigma, steps: int,
            record_every: int = 0):
        """Evolve; returns (A_final, mass_history, centroid_history)."""
        mu = torch.as_tensor(mu, device=self.dev).view(-1, 1, 1).float()
        sigma = torch.as_tensor(sigma, device=self.dev).view(-1, 1, 1).float()
        masses, cents = [], []
        for t in range(steps):
            A = self.step(A, mu, sigma)
            if record_every and (t % record_every == 0 or t == steps - 1):
                masses.append(A.mean(dim=(1, 2)).cpu())
                cents.append(self.centroid(A).cpu())
        return A, (torch.stack(masses, 1) if masses else None), \
                  (torch.stack(cents, 1) if cents else None)

    # --------------------------- measurement ------------------------------

    def centroid(self, A: torch.Tensor) -> torch.Tensor:
        """(B,2) torus centroid in CARTESIAN units via first Fourier phase."""
        n = self.n
        ang = 2 * math.pi * torch.arange(n, device=A.device) / n
        zi = (A * torch.exp(1j * ang)[None, None, :]).sum(dim=(1, 2))  # over i
        zj = (A * torch.exp(1j * ang)[None, :, None]).sum(dim=(1, 2))  # over j
        fi = torch.angle(zi) / (2 * math.pi) * n     # fractional index coords
        fj = torch.angle(zj) / (2 * math.pi) * n
        b = self.basis.to(A.device)
        x = fi * b[0, 0] + fj * b[1, 0]
        y = fi * b[0, 1] + fj * b[1, 1]
        return torch.stack([x, y], dim=-1)

    def drift(self, cents: torch.Tensor) -> torch.Tensor:
        """(B,2) net physical drift from a centroid history (B,T,2),
        unwrapping torus jumps step-by-step."""
        b = self.basis
        Lx = self.n * abs(b[0, 0]) or 1.0
        steps = cents[:, 1:] - cents[:, :-1]
        # unwrap: any per-step jump > half-domain is a wrap artifact
        period_x = self.n * b[0, 0]
        period_y = self.n * b[1, 1]
        sx = steps[..., 0] - period_x * torch.round(steps[..., 0] / period_x)
        sy = steps[..., 1] - period_y * torch.round(steps[..., 1] / period_y)
        return torch.stack([sx.sum(1), sy.sum(1)], dim=-1)


# --------------------------- outcome metrics -------------------------------

def classify(mass_hist: torch.Tensor, drift: torch.Tensor,
             speed_thresh: float = 4.0) -> list[str]:
    """Per-run outcome from mass history (B,T) + net drift (B,2).

    dead: final mass ~0 · full: saturated · alive: persistent structured
    mass · traveler: alive AND net displacement > speed_thresh cell-units.
    """
    out = []
    final = mass_hist[:, -1]
    tail = mass_hist[:, -8:]
    stable = (tail.max(1).values - tail.min(1).values) < 0.02
    disp = drift.norm(dim=-1)
    for b in range(mass_hist.shape[0]):
        if final[b] < 0.005:
            out.append("dead")
        elif final[b] > 0.55:
            out.append("full")
        elif disp[b] > speed_thresh and final[b] > 0.01:
            out.append("traveler")
        elif stable[b]:
            out.append("alive")
        else:
            out.append("chaotic")
    return out


def render(A_row: torch.Tensor, sim: Lenia, px: int = 256) -> torch.Tensor:
    """Rasterize one (n,n) state to a (px,px) cartesian image (display only:
    nearest cell in fractional-lattice metric)."""
    n = sim.n
    b = sim.basis
    binv = torch.linalg.inv(b.T)
    xs = torch.linspace(0, float(n * b[0, 0]), px)
    ys = torch.linspace(0, float(n * b[1, 1]), px)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    P = torch.stack([X, Y], -1).reshape(-1, 2)
    F = (binv @ P.T).T                                  # fractional coords
    ii = torch.round(F[:, 0]).long() % n
    jj = torch.round(F[:, 1]).long() % n
    return A_row.detach().cpu()[jj, ii].reshape(px, px)


def min_image(d: torch.Tensor, sim) -> torch.Tensor:
    """Exact-enough minimum image on the SHEARED torus: wrap in fractional
    lattice coordinates (a rectangular wrap is only correct when the shear
    is zero — i.e. on the square basis; using it on hex scrambles angles).

    d: (..., 2) cartesian displacements → wrapped cartesian displacements.
    """
    n = sim.n
    B = sim.basis.to(d.device)            # rows b1, b2; position = Bᵀ(i,j)
    Binv = torch.linalg.inv(B.T)
    f = torch.einsum("ab,...b->...a", Binv, d)
    f = f - n * torch.round(f / n)
    return torch.einsum("ab,...b->...a", B.T, f)


def lattice_harmonics(A_row: torch.Tensor, sim, center=None,
                      r_min: float = 1.5, m_list=(4, 6)):
    """Angular harmonic content of a field measured on the TRUE cell
    positions (no rasterization — a square-pixel raster imprints m4 on
    everything and must never sit in the measurement path).

    Returns {m: |Σ A·e^{imθ}| / Σ A} over cells with r > r_min from center
    (min-image on the torus). center defaults to the field's torus centroid.
    """
    n = sim.n
    pos = cell_positions(n, sim.basis).to(A_row.device)
    if center is None:
        center = sim.centroid(A_row[None])[0]
    d = min_image(pos - center.view(1, 1, 2), sim)
    r = d.norm(dim=-1)
    th = torch.atan2(d[..., 1], d[..., 0])
    w = A_row * (r > r_min)
    tot = w.sum().clamp(min=1e-12)
    out = {}
    for m in m_list:
        z = (w * torch.exp(1j * m * th)).sum()
        out[f"c{m}"] = float(torch.abs(z) / tot)
    return out


def front_isotropy(A_row: torch.Tensor, sim, center, sectors: int = 24):
    """Isotropy of an expanding front: mean radius per angular sector
    (physical coordinates), returns std(r̄)/mean(r̄) — 0 = perfect circle."""
    n = sim.n
    pos = cell_positions(n, sim.basis).to(A_row.device)
    d = min_image(pos - center.view(1, 1, 2), sim)
    r = d.norm(dim=-1)
    th = torch.atan2(d[..., 1], d[..., 0])
    sec = ((th + math.pi) / (2 * math.pi) * sectors).long().clamp(0, sectors - 1)
    w = A_row.flatten()
    rf, sf = r.flatten(), sec.flatten()
    rbar = []
    for s in range(sectors):
        m = sf == s
        ws = w[m]
        if float(ws.sum()) < 1e-9:
            continue
        rbar.append(float((ws * rf[m]).sum() / ws.sum()))
    rbar = torch.tensor(rbar)
    return float(rbar.std() / rbar.mean().clamp(min=1e-9)), rbar.tolist()


def physical_bump(sim, sigma_r: float = 3.0, amp: float = 0.8):
    """Gaussian seed built from PHYSICAL distances to the domain centre —
    index-space seeds are sheared ellipses on the hex basis (E1 bug)."""
    n = sim.n
    pos = cell_positions(n, sim.basis)
    centre = torch.tensor([n * sim.basis[0, 0] / 2 + n * sim.basis[1, 0] / 2,
                           n * sim.basis[1, 1] / 2])
    r2 = ((pos - centre) ** 2).sum(-1)
    return (amp * torch.exp(-r2 / (2 * sigma_r ** 2)))[None].to(sim.dev)


def angular_spectrum(img: torch.Tensor, nbins: int = 72):
    """Angular power distribution of a rendered pattern + harmonic content.

    Returns (per-angle power [nbins], anisotropy_index, m4, m6) where m4/m6
    are the relative amplitudes of the 4-fold / 6-fold Fourier components of
    the angular distribution — the square lattice should imprint m4, the hex
    lattice m6."""
    F = torch.fft.fftshift(torch.abs(torch.fft.fft2(img - img.mean())) ** 2)
    px = img.shape[0]
    c = px // 2
    yy, xx = torch.meshgrid(torch.arange(px) - c, torch.arange(px) - c,
                            indexing="ij")
    r = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
    theta = torch.atan2(yy.float(), xx.float())
    sel = (r > 3) & (r < c * 0.9)
    tb = ((theta[sel] + math.pi) / (2 * math.pi) * nbins).long() % nbins
    pw = torch.zeros(nbins).scatter_add_(0, tb, F[sel])
    cnt = torch.zeros(nbins).scatter_add_(0, tb, torch.ones_like(F[sel]))
    pw = pw / cnt.clamp(min=1)
    pw = pw / pw.mean().clamp(min=1e-12)
    ani = float(pw.std())
    ft = torch.fft.fft(pw)
    m4 = float(torch.abs(ft[4]) / (abs(float(ft[0])) + 1e-12))
    m6 = float(torch.abs(ft[6]) / (abs(float(ft[0])) + 1e-12))
    return pw, ani, m4, m6


def soup(batch: int, n: int, frac: float = 0.12, seed: int = 0,
         dev=None) -> torch.Tensor:
    """Random blobs in the central region — IDENTICAL index-space soup is
    used for both substrates (pairing; physical layouts differ by the shear,
    which is disclosed)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.zeros(batch, n, n)
    m = int(n * 0.5)
    lo = (n - m) // 2
    blob = (torch.rand(batch, m, m, generator=g) < frac).float()
    blob *= torch.rand(batch, m, m, generator=g)
    # smooth a touch so structures aren't single-cell noise
    k = torch.ones(1, 1, 3, 3) / 9.0
    blob = torch.nn.functional.conv2d(blob[:, None], k, padding=1)[:, 0]
    A[:, lo:lo + m, lo:lo + m] = blob
    return A.to(dev or device())


# ------------------------------ self-test ----------------------------------

def selftest():
    dev = device()
    print(f"device: {dev}")
    for name in ("square", "hex"):
        sim = Lenia(name, n=96, R=8.0, dev=dev)
        assert abs(float(sim.kernel.sum()) - 1.0) < 1e-4, "kernel norm"
        # uniform state must stay uniform (translation invariance / FFT conv)
        A = torch.full((2, 96, 96), 0.3, device=dev)
        A2 = sim.step(A, torch.tensor([[0.15]], device=dev).view(-1, 1, 1),
                      torch.tensor([[0.02]], device=dev).view(-1, 1, 1))
        assert float((A2.max() - A2.min())) < 1e-5, "uniformity broken"
        # dynamics are non-trivial from soup
        A = soup(4, 96, seed=1, dev=dev)
        m0 = A.mean()
        A3, mh, ch = sim.run(A, [0.15] * 4, [0.02] * 4, 60, record_every=10)
        assert abs(float(A3.mean() - m0)) > 1e-4, "dynamics trivial"
        print(f"  {name:6s} kernel ok, uniform fixed-point ok, "
              f"soup mass {float(m0):.3f} → {float(A3.mean()):.3f}")
    # density match sanity
    a = HEX_A
    assert abs(a * a * SQRT3 / 2 - 1.0) < 1e-9
    print("density matched (1 cell / unit area on both lattices)")
    print("SELFTEST PASS")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
