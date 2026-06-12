"""
NON-ABELIAN BRAIDING of vortex Majorana zero modes -- Kitaev honeycomb B phase (kappa>0).
2026-06-10. Numerical realization of Ivanov's theorem (PRL 86, 268 (2001)) on the
project's own lattice: exchanging two vortices acts on the Majorana zero modes as
gamma_i -> gamma_j, gamma_j -> -gamma_i; exchanges sharing a vortex DO NOT COMMUTE.

Method (exact, free-Majorana/BdG; follows scripts/kitaev_nonabelian.py conventions):
  * 4 vortices on an L x L torus (4 MZMs = 2 fermionic modes = the Ising fusion space).
  * Vortex motion = toggling the sign of one bond coupling (substepped continuously
    s: +1 -> -1 to keep adjacent subspaces overlapping).
  * Parallel transport of the 4-dim near-zero REAL subspace along the path: Wilson line
    W = prod O_k, O_k = SVD-unitarized overlap V_{k+1}^T V_k (quality = sigma_min).
  * Braid matrices read out in the vortex-LOCALIZED Majorana basis.

Observables (controls built in):
  O1 single exchange E12: |M| ~ permutation swapping modes 1,2 (transported mode lands
     on the other vortex) -- the swap part is gauge-free.
  O2 COMMUTATOR LOOP [E12, E23] (shared vortex 2): u returns EXACTLY to start (every
     bond toggled twice) => holonomy is gauge-free. Prediction (Ivanov): a signed
     3-cycle 1->2->3->-1 -- far from identity.
  O3 CONTROL: commutator loop [E12, E34] (disjoint pairs) => identity.
Non-Abelian = O2 far from I while O3 ~ I.
"""
import json
import sys
import time

import numpy as np

T0 = time.time()
OUT = {}
L = int(sys.argv[1]) if len(sys.argv) > 1 else 20
GAP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
KAPPA = 0.2
J = {"x": 1.0, "y": 1.0, "z": 1.0}
N1i, N2i = (1, 0), (0, 1)  # integer cell steps for n1, n2
NNN_DIRS = [(1, 0), (0, -1), (-1, 1)]
SUBSTEPS = [0.6, 0.2, -0.2, -0.6, -1.0]  # bond scale path +1 -> -1 (avoid exact 0)


def log(m):
    print(m, flush=True)


# bond (cell c, type t): z(c)=A(c)-B(c); x(c)=A(c)-B(c-n1); y(c)=A(c)-B(c-n2)
BOND_OFF = {"z": (0, 0), "x": (-1, 0), "y": (0, -1)}
# hexagon h(r) contains bonds: z(r), x(r+n1), y(r+n1), z(r+n1-n2), x(r+n1-n2), y(r)
HEX_BONDS = [((0, 0), "z"), ((1, 0), "x"), ((1, 0), "y"),
             ((1, -1), "z"), ((1, -1), "x"), ((0, 0), "y")]
# vortex motion: moving a vortex at h(r) one step flips the shared bond:
MOVE_BOND = {
    (1, 0): lambda r: ((r[0] + 1, r[1]), "y"),    # h(r) -> h(r+n1)
    (-1, 0): lambda r: ((r[0], r[1]), "y"),       # h(r) -> h(r-n1)
    (0, 1): lambda r: ((r[0] + 1, r[1]), "x"),    # h(r) -> h(r+n2)
    (0, -1): lambda r: ((r[0] + 1, r[1] - 1), "x"),  # h(r) -> h(r-n2)
    (1, -1): lambda r: ((r[0] + 1, r[1] - 1), "z"),  # h(r) -> h(r+n1-n2)
    (-1, 1): lambda r: ((r[0], r[1]), "z"),       # h(r) -> h(r-(n1-n2))
}


def wrap(c):
    return (c[0] % L, c[1] % L)


def idx(c, s):
    return 2 * ((c[0] % L) * L + (c[1] % L)) + s


def build_A(scale):
    """A real antisymmetric; scale: dict bond->s in [-1,1] (default +1)."""
    n = 2 * L * L
    A = np.zeros((n, n))
    for x in range(L):
        for y in range(L):
            a = idx((x, y), 0)
            for t, (dx, dy) in BOND_OFF.items():
                b = idx((x + dx, y + dy), 1)
                s = scale.get(((x, y), t), 1.0)
                A[a, b] += 2 * J[t] * s
                A[b, a] -= 2 * J[t] * s
            for dx, dy in NNN_DIRS:
                for s_, sgn in ((0, +1.0), (1, -1.0)):
                    j_, k_ = idx((x, y), s_), idx((x + dx, y + dy), s_)
                    A[j_, k_] += 2 * KAPPA * sgn
                    A[k_, j_] -= 2 * KAPPA * sgn
    return A


def fluxes(scale):
    """Vortex hexagons (W=-1). Valid only when all scales are +-1."""
    vort = []
    for x in range(L):
        for y in range(L):
            w = 1.0
            for (dx, dy), t in HEX_BONDS:
                w *= np.sign(scale.get((wrap((x + dx, y + dy)), t), 1.0))
            if w < 0:
                vort.append((x, y))
    return sorted(vort)


def low_subspace(A, dim=4):
    """Real orthonormal basis (n x dim) of the near-zero subspace + spectra info.
    dim real Majorana dims = dim complex near-zero eigenvectors of iA (the +-E0, +-E0'
    pairs); Re/Im of those span the dim-dimensional real space (each +- pair twice)."""
    w, v = np.linalg.eigh(1j * A)
    o = np.argsort(np.abs(w))
    cols = []
    for i in o[:dim]:
        cols += [v[:, i].real, v[:, i].imag]
    Mr = np.stack(cols, axis=1)
    U, S, _ = np.linalg.svd(Mr, full_matrices=False)
    Esorted = np.abs(w[o])
    return U[:, :dim], Esorted[dim - 1], Esorted[dim]  # basis, top |E| in subspace, next


def localized_gammas(V, vortices):
    """Per-vortex localized Majorana vectors from subspace V (n x 4), ordered by the
    given vortex list (radius-2 cell disks around each vortex hexagon)."""
    gs = []
    for (vx, vy) in vortices:
        mask = np.zeros(2 * L * L, bool)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for s in (0, 1):
                    mask[idx((vx + dx, vy + dy), s)] = True
        G = V[mask, :]
        _, _, Vt = np.linalg.svd(G, full_matrices=False)
        g = V @ Vt[0]
        gs.append(g / np.linalg.norm(g))
    Gm = np.stack(gs, axis=1)
    # orthonormalize (they are near-orthogonal when vortices are separated)
    Uq, _ = np.linalg.qr(Gm)
    # keep sign convention: align Uq columns with the raw gammas
    for i in range(Uq.shape[1]):
        if Uq[:, i] @ Gm[:, i] < 0:
            Uq[:, i] *= -1
    return Uq


class Transporter:
    def __init__(self, scale):
        self.scale = dict(scale)
        A = build_A(self.scale)
        self.V, etop, enext = low_subspace(A)
        self.W = np.eye(4)
        self.stats = {"sigma_min": 1.0, "max_E_subspace": etop, "min_E_next": enext}

    def toggle_bond(self, bond):
        """Move one vortex step: scale[bond] *= -1, substepped, transporting V."""
        s0 = self.scale.get(bond, 1.0)
        assert abs(abs(s0) - 1.0) < 1e-12
        for s in SUBSTEPS:
            self.scale[bond] = s0 * s
            A = build_A(self.scale)
            Vn, etop, enext = low_subspace(A)
            Mov = Vn.T @ self.V
            U, S, Vt = np.linalg.svd(Mov)
            O = U @ Vt
            self.W = O @ self.W
            self.V = Vn
            self.stats["sigma_min"] = min(self.stats["sigma_min"], float(S.min()))
            self.stats["max_E_subspace"] = max(self.stats["max_E_subspace"], float(etop))
            self.stats["min_E_next"] = min(self.stats["min_E_next"], float(enext))
        # final value is exactly -s0
        self.scale[bond] = -s0
        if abs(self.scale[bond] - 1.0) < 1e-12:
            del self.scale[bond]


def path_moves(start, steps):
    """List of bonds for a vortex walking from `start` by unit steps in `steps`."""
    bonds, r = [], start
    for d in steps:
        c, t = MOVE_BOND[d](r)
        bonds.append((wrap(c), t))
        r = wrap((r[0] + d[0], r[1] + d[1]))
    return bonds, r


def exchange_bonds(pa, pb):
    """Counterclockwise exchange of vortices at pa=(xa,y), pb=(xb,y), xb>xa, same row:
    a goes over the top (+n2, +n1 x d, -n2), b slides left (-n1 x d)."""
    d = (pb[0] - pa[0]) % L
    over, _ = path_moves(pa, [(0, 1)] + [(1, 0)] * d + [(0, -1)])
    slide, _ = path_moves(pb, [(-1, 0)] * d)
    return over[: 1 + d] + slide + over[1 + d:]  # a up+across, b across, a down


def exchange_bonds_vert(pa, pb):
    """Exchange for a vertical pair pa=(x,ya), pb=(x,yb), yb>ya: a right+up+left,
    b slides down."""
    d = (pb[1] - pa[1]) % L
    over, _ = path_moves(pa, [(1, 0)] + [(0, 1)] * d + [(-1, 0)])
    slide, _ = path_moves(pb, [(0, -1)] * d)
    return over[: 1 + d] + slide + over[1 + d:]


# ---------------------------------------------------------------- setup
Y0 = L // 2
P1, P2, P3 = (4, Y0), (4 + GAP, Y0), (4 + 2 * GAP, Y0)
P4 = (4 + 2 * GAP, Y0 + GAP + 2)
log(f"L={L} GAP={GAP}: vortices {P1} {P2} {P3} {P4}")
scale0 = {}
for x in range(P1[0] + 1, P2[0] + 1):                    # string P1-P2 (y-bonds)
    scale0[((x, Y0), "y")] = -1.0
for yy in range(Y0, P4[1]):                               # string P3-P4 (x-bonds)
    scale0[((P3[0] + 1, yy), "x")] = -1.0
got = fluxes(scale0)
log(f"setup: vortices at {got} (expect {sorted([P1,P2,P3,P4])})")
assert got == sorted([P1, P2, P3, P4]), "initial flux config wrong"

A0 = build_A(scale0)
V0, etop0, enext0 = low_subspace(A0)
log(f"zero-mode subspace: top |E| in subspace = {etop0:.2e}, next level = {enext0:.3f} "
    f"(bulk gap ~2.0)")
VORTS = [P1, P2, P3, P4]
G0 = localized_gammas(V0, VORTS)
insub = np.linalg.norm(V0 @ (V0.T @ G0) - G0, axis=0)
log(f"localized gammas: residual outside subspace = {np.max(insub):.2e}; "
    f"orthonormality err = {np.max(np.abs(G0.T @ G0 - np.eye(4))):.2e}")
OUT["setup"] = {"E_top_subspace": float(etop0), "E_next": float(enext0),
                "gamma_out_of_subspace": float(np.max(insub))}

E12 = exchange_bonds(P1, P2)
E23 = exchange_bonds(P2, P3)
E34 = exchange_bonds_vert(P3, P4)
log(f"path lengths: E12 {len(E12)} flips, E23 {len(E23)}, E34 {len(E34)} "
    f"(x{len(SUBSTEPS)} substeps each)")


def run_loop(name, segments):
    """Transport along segments (list of bond-lists); returns M in the gamma basis,
    asserting u returns to scale0 (gauge-free) for closed loops."""
    tr = Transporter(scale0)
    for seg in segments:
        for b in seg:
            tr.toggle_bond(b)
    closed = (tr.scale == scale0) or (set(tr.scale.items()) == set(scale0.items()))
    # express holonomy in the localized gamma basis at the FINAL config
    Gend = localized_gammas(tr.V, VORTS)
    M = Gend.T @ (tr.V @ (tr.W @ (V0.T @ G0)))
    log(f"\n[{name}] closed-loop u restored: {closed}; "
        f"sigma_min={tr.stats['sigma_min']:.3f}, "
        f"max|E_sub|={tr.stats['max_E_subspace']:.3f}, min E_next={tr.stats['min_E_next']:.3f}")
    with np.printoptions(precision=3, suppress=True):
        log(f"M =\n{M}")
    return M, tr, closed


# O1: single exchange E12 -- swap structure (gauge-free in |M|)
M_x, tr_x, _ = run_loop("O1 single exchange E12", [E12])
assert fluxes(tr_x.scale) == sorted(VORTS), "exchange should preserve vortex positions"
swap_score = float(np.mean([abs(M_x[1, 0]), abs(M_x[0, 1]), abs(M_x[2, 2]), abs(M_x[3, 3])]))
off_score = float(np.mean([abs(M_x[0, 0]), abs(M_x[1, 1])]))
log(f"O1 swap pattern: |M_21|,|M_12|,|M_33|,|M_44| mean = {swap_score:.3f} "
    f"(want ~1); |M_11|,|M_22| mean = {off_score:.3f} (want ~0)")
OUT["O1_exchange"] = {"M": M_x.tolist(), "swap_score": swap_score, "diag_leak": off_score,
                      "sigma_min": tr_x.stats["sigma_min"]}

# O2: commutator loop [E12, E23] (shared vortex) -- the non-Abelian observable
rev = lambda seg: list(reversed(seg))
M_c, tr_c, closed_c = run_loop("O2 commutator [E12,E23] (shared)", [E12, E23, rev(E12), rev(E23)])
assert closed_c, "commutator loop must restore u exactly"
dist_c = float(np.linalg.norm(M_c - np.eye(4)))
# Ivanov prediction: signed 3-cycle on modes (1,2,3); compare |M| to the 3-cycle perm
P3cyc = np.zeros((4, 4))
P3cyc[1, 0] = P3cyc[2, 1] = P3cyc[0, 2] = P3cyc[3, 3] = 1
match_c = float(np.sum(np.abs(M_c) * P3cyc) / 4.0)
log(f"O2 ||M - I|| = {dist_c:.3f} (non-Abelian if O(1)); "
    f"|M| match to predicted 3-cycle = {match_c:.3f} (want ~1)")
OUT["O2_commutator_shared"] = {"M": M_c.tolist(), "dist_from_I": dist_c,
                               "match_3cycle": match_c, "sigma_min": tr_c.stats["sigma_min"]}

# O3 control: commutator loop [E12, E34] (disjoint pairs) -- must be ~identity
M_d, tr_d, closed_d = run_loop("O3 commutator [E12,E34] (disjoint control)",
                               [E12, E34, rev(E12), rev(E34)])
assert closed_d
dist_d = float(np.linalg.norm(M_d - np.eye(4)))
log(f"O3 ||M - I|| = {dist_d:.3f} (want ~0)")
OUT["O3_commutator_disjoint"] = {"M": M_d.tolist(), "dist_from_I": dist_d,
                                 "sigma_min": tr_d.stats["sigma_min"]}

verdict = (dist_c > 0.5) and (dist_d < 0.2) and (swap_score > 0.8)
log(f"\nVERDICT: non-Abelian braiding {'DEMONSTRATED' if verdict else 'NOT clean'} -- "
    f"shared-commutator {dist_c:.2f} >> disjoint {dist_d:.2f}; exchange swaps modes "
    f"(score {swap_score:.2f}).")
OUT["verdict_nonabelian"] = bool(verdict)
OUT["L"], OUT["GAP"] = L, GAP
OUT["runtime_s"] = time.time() - T0
fn = f"runs/kitaev_braid_mzm_L{L}.json" if len(sys.argv) > 1 else "runs/kitaev_braid_mzm.json"
with open(fn, "w") as f:
    json.dump(OUT, f, indent=1)
log(f"DONE in {OUT['runtime_s']:.0f}s -> {fn}")
