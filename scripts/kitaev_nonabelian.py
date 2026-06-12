"""
Kitaev honeycomb + field (kappa) -- classical groundwork for the NON-ABELIAN (Ising) phase.
2026-06-10. The one named frontier left after the Kitaev topological-order arc
(runs/KITAEV_TOPOLOGICAL_ORDER.md): Kitaev + magnetic field -> gapped B phase with
Chern nu = +-1, chiral Majorana edge (c = 1/2), and Majorana zero modes bound to
vortices (visons) = Ising anyons.

Method: exact free-Majorana (BdG) treatment in a fixed flux sector -- this is EXACT
for the Kitaev model (Kitaev 2006). H = (i/4) sum_jk A_jk c_j c_k, A real antisymmetric.
  NN bonds: A = 2*J_alpha*u_jk (gauge u=+1; flipped on a string to make vortices).
  NNN bonds: A = +-2*kappa -- the field-induced 3-spin term = Haldane-like mass.

Experiments (all numbers written to runs/kitaev_nonabelian.json + stdout log):
  E1 phase diagram: gap + Chern number over (Jz, kappa). Expect: kappa=0 B phase
     gapless; kappa>0 B phase gapped with C=+-1; toric phase (Jz>Jx+Jy) C=0 always.
  E2 cylinder edge spectrum: in-gap chiral Majorana branch iff C=+-1 (bulk-edge),
     edge-localized, chirality (slope) flips with sign(kappa).
  E3 vortex pair on torus: smallest mode energy E0 vs vortex separation.
     B phase (C=+-1): E0 -> 0 exponentially + mode localized on the two vortices
     = MAJORANA ZERO MODES (the Ising-anyon carriers). Toric control: no such mode.

Self-checks (the sign conventions are self-diagnosing):
  S1 torus spectrum == Bloch spectrum on commensurate k-grid.
  S2 Chern is an integer; would be 0 in the B phase if the NNN sign pattern were wrong.
  S3 bulk-edge correspondence: in-gap edge branch present iff C != 0.
"""
import json
import time

import numpy as np

rng = np.random.default_rng(0)
T0 = time.time()
OUT = {}

# ---------------------------------------------------------------- geometry
N1 = np.array([1.0, 0.0])
N2 = np.array([0.5, np.sqrt(3) / 2])
POS = {0: np.array([0.0, 0.0]), 1: (N1 + N2) / 3.0}  # sublattice A=0, B=1

# NN bonds, oriented A -> B (u=+1 convention): B(r) bonded to A(r), A(r+n1), A(r+n2)
# => A(r) bonded to B(r) [z], B(r-n1) [x], B(r-n2) [y]
NN = [  # (dR, bond label) for A(0) -> B(0+dR)
    ((0, 0), "z"),
    ((-1, 0), "x"),
    ((0, -1), "y"),
]
# NNN within a sublattice: the 6 directions; Haldane-like alternating signs.
# Sign prescription chosen for sublattice A; B gets the OPPOSITE sign (this is the
# choice that produces C=+-1 -- verified by S2, which would read C=0 if wrong).
NNN_DIRS = [(1, 0), (0, -1), (-1, 1)]  # +nu for these on A; the reverses get -nu


def count_vortices(L, flip_z_bonds):
    """Hexagon h(r) contains the z-bonds of cells r and r+(n1-n2) (derived by
    walking the hexagon: bonds z(r), x(r+n1), y(r+n1), z(r+n1-n2), x(r+n1-n2), y(r)).
    Only z-bonds are flipped, so W_h(r) = u_z(r) * u_z(r+n1-n2). Returns the list
    of hexagons with W=-1 (the vortices)."""
    flip = set((x % L, y % L) for x, y in flip_z_bonds)
    vort = []
    for x in range(L):
        for y in range(L):
            w = (-1 if (x, y) in flip else 1) * (-1 if ((x + 1) % L, (y - 1) % L) in flip else 1)
            if w == -1:
                vort.append((x, y))
    return vort


def build_torus_A(L, Jx, Jy, Jz, kappa, flip_z_bonds=()):
    """Real antisymmetric A on an L x L torus. flip_z_bonds: set of cells (x,y)
    whose z-bond u is flipped to -1 (a string -> vortex pair at its ends)."""
    J = {"x": Jx, "y": Jy, "z": Jz}
    n = 2 * L * L
    A = np.zeros((n, n))
    idx = lambda x, y, s: 2 * ((x % L) * L + (y % L)) + s
    flip = set(flip_z_bonds)
    for x in range(L):
        for y in range(L):
            a = idx(x, y, 0)
            for (dx, dy), lab in NN:
                b = idx(x + dx, y + dy, 1)
                u = -1.0 if (lab == "z" and (x, y) in flip) else 1.0
                A[a, b] += 2 * J[lab] * u
                A[b, a] -= 2 * J[lab] * u
            if kappa != 0.0:
                for dx, dy in NNN_DIRS:
                    for s, sgn in ((0, +1.0), (1, -1.0)):
                        j = idx(x, y, s)
                        k = idx(x + dx, y + dy, s)
                        A[j, k] += 2 * kappa * sgn
                        A[k, j] -= 2 * kappa * sgn
    return A


def bloch_H(kvec, Jx, Jy, Jz, kappa):
    """2x2 Hermitian Bloch Hamiltonian H(k) = i*M(k), M(k)=sum_R A(R) e^{i k.R}."""
    J = {"x": Jx, "y": Jy, "z": Jz}
    M = np.zeros((2, 2), complex)
    for (dx, dy), lab in NN:
        R = dx * N1 + dy * N2
        M[0, 1] += 2 * J[lab] * np.exp(1j * kvec @ R)
        M[1, 0] -= 2 * J[lab] * np.exp(-1j * kvec @ R)
    if kappa != 0.0:
        for dx, dy in NNN_DIRS:
            R = dx * N1 + dy * N2
            for s, sgn in ((0, +1.0), (1, -1.0)):
                M[s, s] += 2 * kappa * sgn * (np.exp(1j * kvec @ R) - np.exp(-1j * kvec @ R))
    H = 1j * M
    assert np.allclose(H, H.conj().T, atol=1e-12), "Bloch H not Hermitian"
    return H


# reciprocal vectors: G_i . N_j = 2 pi delta_ij
Gmat = 2 * np.pi * np.linalg.inv(np.array([N1, N2]).T)
G1, G2 = Gmat[0], Gmat[1]


def bz_grid(Nk):
    return [(i1 / Nk, i2 / Nk) for i1 in range(Nk) for i2 in range(Nk)]


def gap_and_chern(Jx, Jy, Jz, kappa, Nk=60):
    """Min gap over BZ + Fukui-Hatsugai-Suzuki Chern number of the lower band."""
    Es = np.zeros((Nk, Nk))
    Us = np.zeros((Nk, Nk, 2), complex)  # lower-band eigenvector
    for i1 in range(Nk):
        for i2 in range(Nk):
            k = (i1 / Nk) * G1 + (i2 / Nk) * G2
            w, v = np.linalg.eigh(bloch_H(k, Jx, Jy, Jz, kappa))
            Es[i1, i2] = w[1] - w[0]
            Us[i1, i2] = v[:, 0]
    gap = Es.min() / 2.0  # half the band-to-band distance = |E| of lowest mode
    # FHS link variables
    F = 0.0
    for i1 in range(Nk):
        for i2 in range(Nk):
            u00 = Us[i1, i2]
            u10 = Us[(i1 + 1) % Nk, i2]
            u11 = Us[(i1 + 1) % Nk, (i2 + 1) % Nk]
            u01 = Us[i1, (i2 + 1) % Nk]
            t = (
                np.vdot(u00, u10) * np.vdot(u10, u11)
                * np.vdot(u11, u01) * np.vdot(u01, u00)
            )
            F += np.angle(t)
    C = F / (2 * np.pi)
    return gap, C


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- S1: torus vs Bloch
log("== S1 self-check: torus spectrum vs Bloch spectrum (L=12, iso J, kappa=0.2) ==")
L = 12
A = build_torus_A(L, 1, 1, 1, 0.2)
ev_torus = np.linalg.eigvalsh(1j * A)
ev_torus = np.sort(np.abs(ev_torus))
ev_bloch = []
for i1 in range(L):
    for i2 in range(L):
        k = (i1 / L) * G1 + (i2 / L) * G2
        w = np.linalg.eigvalsh(bloch_H(k, 1, 1, 1, 0.2))
        ev_bloch += list(np.abs(w))
ev_bloch = np.sort(np.array(ev_bloch))
s1_err = float(np.max(np.abs(ev_torus - ev_bloch)))
log(f"S1 max |torus - Bloch| over all {len(ev_torus)} levels = {s1_err:.3e}")
OUT["S1_torus_vs_bloch_maxerr"] = s1_err
assert s1_err < 1e-8, "S1 FAILED: torus and Bloch spectra disagree"

# ---------------------------------------------------------------- E1: phase diagram
log("\n== E1 phase diagram: gap + Chern over (Jz, kappa), Jx=Jy=1, Nk=60 ==")
E1 = []
for kappa in (0.0, 0.1, 0.2):
    for Jz in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        gap, C = gap_and_chern(1, 1, Jz, kappa)
        E1.append({"Jz": Jz, "kappa": kappa, "gap": float(gap), "chern": float(C)})
        log(f"  Jz={Jz:.1f} kappa={kappa:.1f}: gap={gap:.4f}  Chern={C:+.3f}")
OUT["E1_phase_diagram"] = E1
# also the kappa-sign flip at the isotropic point
gap_m, C_m = gap_and_chern(1, 1, 1, -0.2)
log(f"  iso J, kappa=-0.2: gap={gap_m:.4f}  Chern={C_m:+.3f}  (sign flip check)")
OUT["E1_kappa_flip"] = {"gap": float(gap_m), "chern": float(C_m)}

# ---------------------------------------------------------------- E2: cylinder edge
log("\n== E2 cylinder edge spectrum (periodic x, open y), iso J ==")


def cylinder_spectrum(Ny, Jx, Jy, Jz, kappa, nk=121):
    """H(kx) for a cylinder: 1D Bloch in n1, open in n2 (Ny cells). Returns
    kx array, energies [nk, 2Ny], and edge-weight of each state (bottom 3 cells)."""
    J = {"x": Jx, "y": Jy, "z": Jz}
    kxs = np.linspace(-np.pi, np.pi, nk)
    ener = np.zeros((nk, 2 * Ny))
    edgew = np.zeros((nk, 2 * Ny))
    for ik, kx in enumerate(kxs):
        M = np.zeros((2 * Ny, 2 * Ny), complex)
        for y in range(Ny):
            a = 2 * y + 0
            for (dx, dy), lab in NN:
                y2 = y + dy
                if 0 <= y2 < Ny:
                    b = 2 * y2 + 1
                    ph = np.exp(1j * kx * dx)
                    M[a, b] += 2 * J[lab] * ph
                    M[b, a] -= 2 * J[lab] * np.conj(ph)
            if kappa != 0.0:
                for dx, dy in NNN_DIRS:
                    y2 = y + dy
                    if 0 <= y2 < Ny:
                        for s, sgn in ((0, +1.0), (1, -1.0)):
                            j, k2 = 2 * y + s, 2 * y2 + s
                            ph = np.exp(1j * kx * dx)
                            M[j, k2] += 2 * kappa * sgn * ph
                            M[k2, j] -= 2 * kappa * sgn * np.conj(ph)
        H = 1j * M
        w, v = np.linalg.eigh(H)
        ener[ik] = w
        wt = np.abs(v) ** 2
        edgew[ik] = wt[:6, :].sum(axis=0)  # weight in BOTTOM 3 unit cells (6 sites)
    return kxs, ener, edgew


Ny = 40
kxs, en_topo, ew_topo = cylinder_spectrum(Ny, 1, 1, 1, 0.2)
_, en_triv, ew_triv = cylinder_spectrum(Ny, 1, 1, 2.5, 0.2)
_, en_flip, ew_flip = cylinder_spectrum(Ny, 1, 1, 1, -0.2)
bulkgap, _ = gap_and_chern(1, 1, 1, 0.2)
bulkgap_T, _ = gap_and_chern(1, 1, 2.5, 0.2)
THR, THR_T = 0.9 * bulkgap, 0.9 * bulkgap_T  # per-phase thresholds (gaps differ: 2.0 vs 1.0)
n_in_topo = int(np.sum(np.abs(en_topo) < THR))
n_in_triv = int(np.sum(np.abs(en_triv) < THR_T))
log(f"  bulk gaps (units eigvalsh(iA)): B phase {bulkgap:.4f}, toric {bulkgap_T:.4f}")
log(f"  in-gap state count over {len(kxs)} kx (each vs ITS OWN gap): "
    f"B phase (C=+-1) = {n_in_topo},  toric (C=0) = {n_in_triv}")
# edge-localization of in-gap states in the B phase (bottom-edge weight; states on the
# top edge have ~0 bottom weight, so bottom-edge states should show ~1, bulk ~0.075)
mask = np.abs(en_topo) < THR
bw = ew_topo[mask]
log(f"  in-gap B-phase states: median bottom-edge weight of the bottom-half = "
    f"{float(np.median(bw[bw > 0.5])) if (bw > 0.5).any() else float('nan'):.3f}, "
    f"fraction edge-localized (bottom>0.5 or <0.05 i.e. top) = "
    f"{float(np.mean((bw > 0.5) | (bw < 0.05))):.3f} (bulk expectation ~ {6/(2*Ny):.3f})")


# chirality: slope of the bottom-edge branch AT ITS ZERO CROSSING (wherever in the BZ)
def edge_slope(en, ew):
    pts = []
    for ik in range(len(kxs)):
        m = (np.abs(en[ik]) < THR) & (ew[ik] > 0.5)
        for b in np.where(m)[0]:
            pts.append((kxs[ik], en[ik][b]))
    if len(pts) < 4:
        return float("nan")
    pts = np.array(pts)
    kc = pts[np.argmin(np.abs(pts[:, 1])), 0]  # kx of the near-zero point
    dk = np.angle(np.exp(1j * (pts[:, 0] - kc)))  # wrap-aware distance from kc
    sel = np.abs(dk) < 0.6
    if sel.sum() < 3:
        return float("nan")
    return float(np.polyfit(dk[sel], pts[sel, 1], 1)[0])


sl_plus, sl_minus = edge_slope(en_topo, ew_topo), edge_slope(en_flip, ew_flip)
log(f"  bottom-edge branch velocity: kappa=+0.2 -> {sl_plus:+.3f}, kappa=-0.2 -> {sl_minus:+.3f}"
    f"  (opposite sign = chirality flips with kappa)")
OUT["E2_cylinder"] = {
    "bulk_gap": float(bulkgap), "ingap_states_Bphase": n_in_topo,
    "ingap_states_toric": n_in_triv,
    "frac_edge_localized": float(np.mean((bw > 0.5) | (bw < 0.05))) if mask.any() else None,
    "bulk_expectation": 6 / (2 * Ny),
    "edge_velocity_kappa_plus": sl_plus, "edge_velocity_kappa_minus": sl_minus,
}

# ---------------------------------------------------------------- E3: vortex MZMs
log("\n== E3 vortex pair: Majorana zero modes vs separation (L=20 torus) ==")
L = 20


def vortex_run(Jz, kappa, d, y0=14, x0=4):
    """Two vortices: flip z-bonds of cells (x0+i, y0-i), i=0..d-1 -- the string steps
    along n1-n2 so consecutive z-bonds SHARE a hexagon (interior fluxes cancel,
    exactly 2 vortices remain at the ends; asserted via count_vortices).
    Returns lowest |E| levels + localization of the lowest mode at the 2 vortices."""
    flips = [((x0 + i) % L, (y0 - i) % L) for i in range(d)]
    vort = count_vortices(L, flips)
    assert len(vort) == 2, f"expected 2 vortices, got {len(vort)}: {vort}"
    A = build_torus_A(L, 1, 1, Jz, kappa, flip_z_bonds=flips)
    w, v = np.linalg.eigh(1j * A)
    order = np.argsort(np.abs(w))
    E0, E1_ = np.abs(w[order[0]]), np.abs(w[order[2]])
    # localization: weight of the +-E0 pair within radius-2 cells of each vortex hexagon
    mode = np.abs(v[:, order[0]]) ** 2 + np.abs(v[:, order[1]]) ** 2  # total weight 2.0
    cellw = mode.reshape(L, L, 2).sum(axis=2)

    def diskw(xc, yc):
        tot = 0.0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                tot += cellw[(xc + dx) % L, (yc + dy) % L]
        return tot

    locw = sum(diskw(x, y) for x, y in vort)
    return E0, E1_, locw, vort


log("  B phase (iso J, kappa=0.2):")
e3_b = []
for d in (2, 3, 4, 5, 6, 8):
    E0, E1_, locw, vort = vortex_run(1.0, 0.2, d)
    e3_b.append({"d": d, "E0": float(E0), "E1": float(E1_), "loc_weight": float(locw)})
    log(f"    sep d={d:2d} (vortices {vort}): E0={E0:.2e}  next={E1_:.4f}  "
        f"mode weight on the 2 vortex disks={locw:.3f}/2.0")
# exponential-decay fit of the MZM splitting (skip d=2: disks overlap / short-string)
ds = np.array([r["d"] for r in e3_b if r["d"] >= 3 and r["E0"] > 1e-12])
es = np.array([r["E0"] for r in e3_b if r["d"] >= 3 and r["E0"] > 1e-12])
if len(ds) >= 3:
    slope, intc = np.polyfit(ds, np.log(es), 1)
    log(f"  fit ln E0 = {slope:.3f}*d + {intc:.2f}  -> decay length xi = {-1/slope:.2f} cells"
        f" (exponential = MZM hybridization)")
    OUT["E3_fit"] = {"slope": float(slope), "xi_cells": float(-1 / slope)}
log("  toric control (Jz=2.5, kappa=0.2):")
e3_t = []
for d in (4, 8):
    E0, E1_, locw, vort = vortex_run(2.5, 0.2, d)
    e3_t.append({"d": d, "E0": float(E0), "E1": float(E1_), "loc_weight": float(locw)})
    log(f"    sep d={d:2d}: E0={E0:.5f}  next={E1_:.4f}  mode weight on the 2 vortex disks={locw:.3f}/2.0")
# no-vortex reference gaps
A_ref = build_torus_A(L, 1, 1, 1.0, 0.2)
ref_b = float(np.min(np.abs(np.linalg.eigvalsh(1j * A_ref))))
A_ref = build_torus_A(L, 1, 1, 2.5, 0.2)
ref_t = float(np.min(np.abs(np.linalg.eigvalsh(1j * A_ref))))
log(f"  no-vortex lowest |E|: B phase {ref_b:.4f}, toric {ref_t:.4f}")
OUT["E3_vortex"] = {"B_phase": e3_b, "toric": e3_t,
                    "novortex_E0_B": ref_b, "novortex_E0_toric": ref_t}

OUT["runtime_s"] = time.time() - T0
with open("runs/kitaev_nonabelian.json", "w") as f:
    json.dump(OUT, f, indent=1)
log(f"\nDONE in {OUT['runtime_s']:.1f}s -> runs/kitaev_nonabelian.json")
