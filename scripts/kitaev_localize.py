"""KITAEV localized ANYON on a 2-hexagon patch (10 sites) -- classical ground truth + QPU prep recipe.

kitaev_order/anyons showed fluxes are conserved + gapped. The next signature: an anyon (vison) is
LOCALIZED -- a local operator flips ONE plaquette's flux while the NEIGHBOUR stays untouched.

Patch = two hexagons sharing an edge (10 sites), Kitaev-coloured. Plaquette fluxes W_p1, W_p2.
  * prepare the (+1,+1) flux vacuum,
  * apply a LOCAL vison-creation operator (single Pauli on a p1-only site, anticommutes with W_p1,
    commutes with W_p2),
  * measure <W_p1> -> -1, <W_p2> -> +1  == the anyon is LOCALIZED to plaquette 1,
  * evolve under Kitaev -> the vison is STABLE (fluxes conserved).

Also derives the SHALLOW prep for the QPU port: 8 non-shared sites take a product-state prep
(each a +1 eigenstate of its W_p Pauli); only the 2 SHARED sites need an entangled (Bell) state to
satisfy both fluxes at once. Prints the shared-site Paulis so the QPU circuit can be built.
No number recorded before its run completes.
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.linalg import expm
from kitaev_order import op, bond_op


def honeycomb_points(R=3):
    a1 = np.array([1.5, math.sqrt(3) / 2]); a2 = np.array([1.5, -math.sqrt(3) / 2])
    A = np.array([0.0, 0.0]); B = np.array([1.0, 0.0])
    pts = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            o = i * a1 + j * a2
            pts.append(o + A); pts.append(o + B)
    return np.array(pts)


def bond_color(d):
    """axis of a NN bond by direction angle mod 180 -> X/Y/Z (the 3 honeycomb bond axes)."""
    ang = math.degrees(math.atan2(d[1], d[0])) % 180.0
    k = int(round(ang / 60.0)) % 3
    return "XYZ"[k]


def find_cycles(adj, k):
    res = set()
    for s in adj:
        stack = [(s, [s])]
        while stack:
            nd, p = stack.pop()
            if len(p) == k:
                if s in adj[nd] and s == min(p):
                    res.add(tuple(sorted(p)))
                continue
            for nb in adj[nd]:
                if nb not in p:
                    stack.append((nb, p + [nb]))
    return [list(r) for r in res]


def main():
    os.makedirs("runs", exist_ok=True)
    logf = open("runs/kitaev_localize.log", "a")
    def log(m): print(m); logf.write(m + "\n"); logf.flush()
    log(f"\n=== KITAEV localized ANYON (2-hexagon patch) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")

    pts = honeycomb_points(3)
    # global bonds + colors
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    gadj = {i: {} for i in range(len(pts))}
    for i in range(len(pts)):
        for j in np.where(np.abs(D[i] - 1.0) < 0.2)[0]:
            if j != i:
                gadj[i][int(j)] = bond_color(pts[j] - pts[i])
    gadj_set = {i: set(gadj[i]) for i in gadj}
    hexes = find_cycles(gadj_set, 6)
    # pick two hexagons sharing exactly an edge (2 common vertices)
    pair = None
    for a in range(len(hexes)):
        for b in range(a + 1, len(hexes)):
            common = set(hexes[a]) & set(hexes[b])
            if len(common) == 2 and common.issubset(gadj_set[list(common)[0] if False else hexes[a][0]]) or len(common) == 2:
                # require the 2 common vertices be adjacent (a shared edge)
                u, v = tuple(common)
                if v in gadj_set[u]:
                    pair = (hexes[a], hexes[b]); break
        if pair:
            break
    h1, h2 = pair
    verts = sorted(set(h1) | set(h2))
    remap = {g: k for k, g in enumerate(verts)}        # global->local 0..9
    n = len(verts)
    log(f"patch: {n} sites (two hexagons sharing an edge); shared = {sorted(set(h1)&set(h2))}")

    # local adjacency/colors among patch vertices
    ladj = {remap[g]: {remap[j]: c for j, c in gadj[g].items() if j in remap} for g in verts}
    tri_ok = all(sorted(set(ladj[q].values())) == ["X", "Y", "Z"] or len(ladj[q]) < 3 for q in ladj)
    deg = {q: len(ladj[q]) for q in ladj}
    log(f"  per-site coordination in patch: {sorted(deg.values())} (interior=3, boundary<3)")

    def plaq_op(hex_global):
        pauli = {}
        hset = set(hex_global)
        for g in hex_global:
            outward = [j for j in gadj[g] if j not in hset]      # outward bond (may be outside patch)
            if not outward:
                return None
            pauli[remap[g]] = gadj[g][outward[0]]
        return pauli

    pauli1, pauli2 = plaq_op(h1), plaq_op(h2)
    Wp1, Wp2 = op(pauli1, n), op(pauli2, n)

    # Kitaev Hamiltonian on the patch (bonds among patch vertices)
    H = np.zeros((2**n, 2**n), complex)
    seen = set()
    for g in verts:
        for j, c in gadj[g].items():
            if j in remap and tuple(sorted((remap[g], remap[j]))) not in seen:
                seen.add(tuple(sorted((remap[g], remap[j]))))
                H -= bond_op(remap[g], remap[j], c, n)

    cWH1 = float(np.linalg.norm(Wp1 @ H - H @ Wp1))
    cWH2 = float(np.linalg.norm(Wp2 @ H - H @ Wp2))
    cWW = float(np.linalg.norm(Wp1 @ Wp2 - Wp2 @ Wp1))
    log(f"  W_p1 Paulis (local idx): {pauli1}")
    log(f"  W_p2 Paulis (local idx): {pauli2}")
    log(f"  ||[W_p1,H]||={cWH1:.1e}  ||[W_p2,H]||={cWH2:.1e}  ||[W_p1,W_p2]||={cWW:.1e}  (all ~0 => conserved fluxes)")

    # (+1,+1) flux-vacuum eigenstate via projector on a random vector
    rng = np.random.default_rng(0)
    P = ((np.eye(2**n) + Wp1) / 2) @ ((np.eye(2**n) + Wp2) / 2)
    psi = P @ (rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n))
    psi /= np.linalg.norm(psi)
    e1 = float(np.real(psi.conj() @ Wp1 @ psi)); e2 = float(np.real(psi.conj() @ Wp2 @ psi))
    log(f"\n  flux vacuum: <W_p1>={e1:+.3f} <W_p2>={e2:+.3f} (want +1,+1)")

    # local vison creation: a single Pauli on a p1-ONLY site that anticommutes with W_p1, commutes with W_p2
    shared = set(remap[g] for g in (set(h1) & set(h2)))
    p1only = [remap[g] for g in h1 if remap[g] not in shared]
    creator = None
    for q in p1only:
        for pl in "XYZ":
            L = op({q: pl}, n)
            ac1 = np.linalg.norm(L @ Wp1 + Wp1 @ L) < 1e-9      # anticommute with Wp1
            co2 = np.linalg.norm(L @ Wp2 - Wp2 @ L) < 1e-9      # commute with Wp2
            if ac1 and co2:
                creator = (q, pl, L); break
        if creator:
            break
    q, pl, L = creator
    log(f"  vison creator = {pl} on site {q} (p1-only): anticommutes W_p1, commutes W_p2")
    psiv = L @ psi
    v1 = float(np.real(psiv.conj() @ Wp1 @ psiv)); v2 = float(np.real(psiv.conj() @ Wp2 @ psiv))
    log(f"  after creation: <W_p1>={v1:+.3f} <W_p2>={v2:+.3f}  ==> LOCALIZED anyon (p1 flipped, p2 intact)")

    # stability under Kitaev evolution
    U = expm(-1j * H * 1.5)
    psit = U @ psiv
    t1 = float(np.real(psit.conj() @ Wp1 @ psit)); t2 = float(np.real(psit.conj() @ Wp2 @ psit))
    log(f"  after Kitaev evolution T=1.5: <W_p1>={t1:+.3f} <W_p2>={t2:+.3f}  (vison STABLE => conserved)")

    # shared-site Paulis (for the QPU shallow prep)
    sh = sorted(shared)
    log(f"\n  shared sites (local idx) {sh}: W_p1 gives "
        f"{[pauli1.get(s,'I') for s in sh]}, W_p2 gives {[pauli2.get(s,'I') for s in sh]} "
        f"(equal => product prep; differ => 1 entangling gate on the shared pair)")

    json.dump(dict(n=n, shared=sh, pauli1=pauli1, pauli2=pauli2, comm=[cWH1, cWH2, cWW],
                   vacuum=[e1, e2], localized=[v1, v2], stable=[t1, t2],
                   creator=[q, pl]), open("runs/kitaev_localize.json", "w"), indent=2, default=str)
    log("=== done ===")


if __name__ == "__main__":
    main()
