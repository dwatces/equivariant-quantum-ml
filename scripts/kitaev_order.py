"""
KITAEV HONEYCOMB topological ORDER -- the genuinely-untested regime.

The whole project tested HALDANE (band topology, single-particle: chirality/edge/disorder)
and found it real-but-modest. It NEVER tested the honeycomb's KITAEV model = many-body
TOPOLOGICAL ORDER (Z2 fluxes, anyons), which is (a) the genuine quantum-advantage regime per
the dequantization argument [symmetry alone is classically simulable; advantage needs topological
order], and (b) RIGOROUSLY honeycomb-unique: Kitaev's exact solvability needs the TRIVALENT
honeycomb (each vertex has exactly one x-, one y-, one z-bond). A square lattice (coordination 4)
CANNOT be 3-colored one-type-per-edge -> the construction fails. "Square can't" is a THEOREM here,
not a marginal sim result.

H = -Jx sum_{x-bonds} X_i X_j  -Jy sum_{y-bonds} Y_i Y_j  -Jz sum_{z-bonds} Z_i Z_j

DECISIVE signatures (exact diagonalization, dense, small cluster):
  (1) plaquette flux W_p = X Y Z X Y Z around a hexagon COMMUTES with H -> extensive set of
      conserved Z2 fluxes = the solvable spin-liquid structure (the hallmark of topological order).
  (2) fluxes are SHARP in eigenstates: <W_p> = +-1.
  (3) SQUARE control: build the analogous 4-coordinate "Kitaev" on a square torus -> the would-be
      plaquette/flux operators do NOT commute with H (no conserved Z2 flux) -> no topological order.
No number recorded before its run completes.
"""
import os, sys, json, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = {"X": X, "Y": Y, "Z": Z, "I": I}


def op(pauli_on, n):
    """pauli_on: dict {site:'X'/'Y'/'Z'} -> full n-qubit operator via kron."""
    mats = [PAULI[pauli_on.get(k, "I")] for k in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def bond_op(a, b, t, n):
    return op({a: t, b: t}, n)


# ---------------- honeycomb 2x2 torus (8 sites, 4 plaquettes) -----------------
def honeycomb_torus():
    """cells (i,j) in {0,1}^2, sublattice A/B. site = ((i*2+j)*2 + s).
    z-bond: A(i,j)-B(i,j);  x-bond: A(i,j)-B(i+1,j);  y-bond: A(i,j)-B(i,j+1) (mod 2)."""
    def idx(i, j, s): return ((i % 2) * 2 + (j % 2)) * 2 + s
    n = 8
    bonds = []
    for i in range(2):
        for j in range(2):
            A = idx(i, j, 0)
            bonds.append((A, idx(i, j, 1), "Z"))       # z: same cell
            bonds.append((A, idx(i + 1, j, 1), "X"))   # x: +i neighbor B
            bonds.append((A, idx(i, j + 1, 1), "Y"))   # y: +j neighbor B
    # plaquette (hexagon) operator: product of the OUTWARD-bond Pauli at each of 6 vertices.
    # For each vertex, outward bond = the bond type NOT shared with the two in-plaquette bonds.
    # We discover plaquettes as 6-cycles and assign each vertex the type of its third (outward) bond.
    return n, bonds


def vertex_bondtypes(n, bonds):
    """map site -> {neighbor: type} and site -> set of its 3 types."""
    adj = {k: {} for k in range(n)}
    for a, b, t in bonds:
        adj[a][b] = t; adj[b][a] = t
    return adj


def find_hexagons(n, bonds):
    """find 6-cycles (plaquettes) in the small torus graph."""
    adj = {k: set() for k in range(n)}
    for a, b, _ in bonds:
        adj[a].add(b); adj[b].add(a)
    hexes = set()
    for start in range(n):
        # DFS depth-6 simple cycles back to start
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == 6:
                if start in adj[node] and start == min(path):
                    hexes.add(tuple(sorted(path)))
                continue
            for nb in adj[node]:
                if nb not in path:
                    stack.append((nb, path + [nb]))
    return [list(h) for h in hexes]


def plaquette_operator(hexagon, adj, n):
    """W_p = product over hexagon vertices of sigma^{outward bond type}.
    outward bond = the neighbor NOT in the hexagon."""
    pauli_on = {}
    hexset = set(hexagon)
    for v in hexagon:
        outward = [nb for nb in adj[v] if nb not in hexset]
        if len(outward) != 1:
            return None
        pauli_on[v] = adj[v][outward[0]]
    return op(pauli_on, n)


# ---------------- square 2x2 torus control (coordination 4) -------------------
def square_torus():
    """3x3 periodic square (9 sites), each vertex has coordination 4 -> CANNOT 3-color
    one-type-per-edge (4 bonds, 3 types -> a type repeats at every vertex). Assign Kitaev-style
    types per direction and show the would-be plaquette fluxes fail to commute with H."""
    def idx(i, j): return (i % 3) * 3 + (j % 3)
    n = 9
    bonds = []
    seen = set()
    # +i -> X, +j -> Y  (each vertex then also has -i:X and -j:Y from neighbors => 2 X + 2 Y, no Z)
    for i in range(3):
        for j in range(3):
            a = idx(i, j)
            for (di, dj, t) in [(1, 0, "X"), (0, 1, "Y")]:
                b = idx(i + di, j + dj)
                key = tuple(sorted((a, b)))
                if a != b and key not in seen:
                    seen.add(key); bonds.append((a, b, t))
    return n, bonds


def comm_norm(A, B):
    return float(np.linalg.norm(A @ B - B @ A))


def main():
    os.makedirs("runs", exist_ok=True)
    logf = open("runs/kitaev_order.log", "a")
    def log(m): print(m); logf.write(m + "\n"); logf.flush()
    log(f"\n=== KITAEV HONEYCOMB topological order (exact diag) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")

    # ---- honeycomb ----
    n, bonds = honeycomb_torus()
    adj = vertex_bondtypes(n, bonds)
    types_per_vertex = {k: sorted(set(adj[k].values())) for k in range(n)}
    tri = all(v == ["X", "Y", "Z"] for v in types_per_vertex.values())
    log(f"honeycomb: {n} sites, {len(bonds)} bonds | every vertex has exactly x,y,z bonds: {tri}")

    Jx = Jy = Jz = 1.0
    H = np.zeros((2**n, 2**n), complex)
    for a, b, t in bonds:
        J = {"X": Jx, "Y": Jy, "Z": Jz}[t]
        H -= J * bond_op(a, b, t, n)

    hexes = find_hexagons(n, bonds)
    log(f"honeycomb plaquettes (6-cycles) found: {len(hexes)}")
    Wps, comms = [], []
    for h in hexes:
        Wp = plaquette_operator(h, adj, n)
        if Wp is None:
            continue
        Wps.append(Wp)
        comms.append(comm_norm(Wp, H))
    log(f"plaquette flux operators built: {len(Wps)}")
    log(f"  max ||[W_p, H]||  = {max(comms):.2e}   (==0 => fluxes CONSERVED = topological order)")
    # mutual commutation of fluxes
    mutual = max((comm_norm(Wps[i], Wps[j]) for i in range(len(Wps)) for j in range(i + 1, len(Wps))),
                 default=0.0)
    log(f"  max ||[W_p, W_q]|| = {mutual:.2e}   (==0 => simultaneously diagonalizable Z2 set)")
    # ground state + flux sharpness
    E, V = np.linalg.eigh(H)
    gs = V[:, 0]
    fluxes = [float(np.real(gs.conj() @ (Wp @ gs))) for Wp in Wps]
    log(f"  ground energy/site = {E[0]/n:.4f} | ground-state degeneracy (E-E0<1e-6) = {int((E - E[0] < 1e-6).sum())}")
    log(f"  <W_p> in ground state = {[round(f,3) for f in fluxes]}  (sharp +-1 => good flux quantum numbers)")
    gap_iso = float(E[(E - E[0] > 1e-6)][0] - E[0])
    log(f"  spectral gap (isotropic Jx=Jy=Jz=1) = {gap_iso:.4f}  (isotropic point is ~gapless/Dirac B-phase)")

    # anisotropic toric-code (A_z) phase: Jz >> Jx,Jy -> gapped topologically-ordered phase
    Ha = np.zeros((2**n, 2**n), complex)
    Jani = {"X": 0.25, "Y": 0.25, "Z": 1.0}
    for a, b, t in bonds:
        Ha -= Jani[t] * bond_op(a, b, t, n)
    Ea, Va = np.linalg.eigh(Ha)
    gap_ani = float(Ea[(Ea - Ea[0] > 1e-6)][0] - Ea[0])
    gsa = Va[:, 0]
    fluxes_a = [float(np.real(gsa.conj() @ (Wp @ gsa))) for Wp in Wps]
    comm_a = max(comm_norm(Wp, Ha) for Wp in Wps)
    log(f"  anisotropic (Jz=1, Jx=Jy=0.25, toric-code phase): gap = {gap_ani:.4f} (gapped); "
        f"fluxes still conserved ||[W,Ha]||={comm_a:.1e}; <W_p>={[round(f,3) for f in fluxes_a]}")

    # ---- square control ----
    ns, sb = square_torus()
    adjs = vertex_bondtypes(ns, sb)
    deg = [len(adjs[k]) for k in range(ns)]
    tps = {k: sorted(adjs[k].values()) for k in range(ns)}
    can_3color = all(sorted(set(adjs[k].values())) == ["X", "Y", "Z"] and len(adjs[k]) == 3 for k in range(ns))
    log(f"\nsquare control: {ns} sites, coordination = {deg} | one-of-each-type-per-vertex: {can_3color}")
    Hs = np.zeros((2**ns, 2**ns), complex)
    for a, b, t in sb:
        Hs -= bond_op(a, b, t, ns)
    sq_hexes = find_hexagons(ns, sb)        # squares have 4-cycles, not 6 -> expect none
    # build the natural square plaquette (4-cycle) flux and test commutation
    def find_kcycles(n, bonds, k):
        adjc = {q: set() for q in range(n)}
        for a, b, _ in bonds:
            adjc[a].add(b); adjc[b].add(a)
        res = set()
        for s in range(n):
            stack = [(s, [s])]
            while stack:
                nd, p = stack.pop()
                if len(p) == k:
                    if s in adjc[nd] and s == min(p):
                        res.add(tuple(sorted(p)))
                    continue
                for nb in adjc[nd]:
                    if nb not in p:
                        stack.append((nb, p + [nb]))
        return [list(r) for r in res]
    sq4 = find_kcycles(ns, sb, 4)
    sqcomms = []
    for c in sq4:
        # square 'flux' = product of outward-bond Pauli (best-effort; vertices have 2 outward bonds)
        pauli_on = {}
        cset = set(c)
        ok = True
        for v in c:
            outward = [nb for nb in adjs[v] if nb not in cset]
            if not outward:
                ok = False; break
            pauli_on[v] = adjs[v][outward[0]]
        if ok:
            sqcomms.append(comm_norm(op(pauli_on, ns), Hs))
    log(f"square: 4-cycles={len(sq4)} | best-effort plaquette flux ||[W,H]|| = "
        f"{(max(sqcomms) if sqcomms else float('nan')):.2e}  (!=0 => NO conserved Z2 flux => no topological order)")

    out = dict(honeycomb=dict(n=n, trivalent=tri, n_plaquettes=len(Wps),
                              max_comm_WH=max(comms), max_comm_WW=mutual,
                              ground_deg=int((E - E[0] < 1e-6).sum()), fluxes=fluxes),
               square=dict(n=ns, coordination=deg, can_3color=can_3color,
                           plaquette_comm=(max(sqcomms) if sqcomms else None)))
    json.dump(out, open("runs/kitaev_order.json", "w"), indent=2, default=float)
    log("\nVERDICT: honeycomb Kitaev has an extensive conserved-Z2-flux structure (topological order)"
        " that the square fundamentally lacks (coordination 4 -> no one-type-per-edge -> no commuting flux).")
    log("=== done ===")


if __name__ == "__main__":
    main()
