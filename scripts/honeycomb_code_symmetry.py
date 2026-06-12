"""
HONEYCOMB FLOQUET CODE — exact circuit-level symmetry test (2026-06-11 evening).

THE QUESTION (from the day's two results colliding):
  - Surface code: exact circuit-level twisted symmetry is OBSTRUCTED — 999-affine-map
    null + 1,728-schedule co-design null (scripts/circuit_symmetry_mine.py,
    scripts/schedule_codesign.py). Mechanism: 4-step CNOT extraction hook errors;
    mirror-axis ancillas provably unfixable (no permutation equals its own E<->W swap).
  - Honeycomb (Hastings-Haah) Floquet code: checks are WEIGHT-2 because the lattice is
    TRIVALENT (the project's "square can't 3-color" theorem). Extraction = 2 gates per
    check -> schedule freedom is ONE BIT per edge (which endpoint first), and hook
    errors are weight<=1 on data.
  CONJECTURE: the symmetry that is provably broken for the surface code's extraction
  is constructively achievable for the honeycomb code — i.e. the hexagon-native code
  admits exact circuit-level equivariance where the square-native one cannot.

PREDICTIONS LOGGED BEFORE THE RUN (hard rule: no result recorded before completion):
  P1 mpp mode (direct 2-body MPP): translations AND both C2s verify (no extraction
     structure exists to break anything).
  P2 cx-naive mode (ancilla + 2 controlled gates, uniform "A-endpoint first"):
     translations verify (hooks map to hooks under sublattice-preserving maps);
     C2 maps FAIL (they swap A<->B sublattices; hooks land on the wrong endpoint).
  P3 cx-sym mode (endpoint orders chosen orbit-wise under {T11, T22, C2_face}):
     translations + C2_face verify. C2_edge fixes the central z-edge while swapping
     its endpoints -> one-bit obstruction (the surface-code mirror-axis phenomenon in
     minimal form) -> C2_edge still fails even here. That asymmetry IS the result.

Conventions: torus, L x M cells (L,M = 0 mod 3), vertices A(i,j), B(i,j).
Edges: z: A(i,j)-B(i,j); x: B(i,j)-A(i+1,j); y: B(i,j)-A(i,j+1).
Faces f(i,j) = [A(i,j), B(i,j), A(i,j+1), B(i-1,j+1), A(i-1,j+1), B(i-1,j)],
3-colored c=(i-j)%3; edge color = the color of neither adjacent face. Sub-round s
measures color-(s%3) edges with Pauli (X,Y,Z)[s%3]; init |+>^n so sub-round 0 (XX)
is deterministic. Face windows: the two consecutive sub-round phases carrying its
edge colors; detector = window product at consecutive periods (12 records).
Full-DEM verification for s-preserving maps (translations); time-bulk one-sided
verification for color-permuting maps (their windows shift s by +-1, so the first/
last layers are a genuine time boundary — reported as such, like spatial boundaries).
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import stim

PAULI = "XYZ"
PTARG = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
PGATE = {"X": "CX", "Y": "CY", "Z": "CZ"}  # ancilla |+> is the control; MX reads parity


def build_lattice(L, M):
    assert L % 3 == 0 and M % 3 == 0
    verts = [("A", i, j) for i in range(L) for j in range(M)] + \
            [("B", i, j) for i in range(L) for j in range(M)]
    vid = {v: k for k, v in enumerate(verts)}

    def A(i, j):
        return ("A", i % L, j % M)

    def B(i, j):
        return ("B", i % L, j % M)

    faces = {}
    for i in range(L):
        for j in range(M):
            fv = [A(i, j), B(i, j), A(i, j + 1), B(i - 1, j + 1),
                  A(i - 1, j + 1), B(i - 1, j)]
            fe = [frozenset({fv[k], fv[(k + 1) % 6]}) for k in range(6)]
            faces[(i, j)] = {"color": (i - j) % 3, "verts": fv, "edge_sets": fe}

    adj = defaultdict(list)
    for fk, f in faces.items():
        for es in f["edge_sets"]:
            adj[es].append(fk)
    edges, elook = {}, {}
    for es, fks in sorted(adj.items(), key=lambda kv: sorted(vid[v] for v in kv[0])):
        assert len(fks) == 2, f"edge borders {len(fks)} faces"
        c1, c2 = faces[fks[0]]["color"], faces[fks[1]]["color"]
        assert c1 != c2
        va, vb = sorted(es, key=lambda v: vid[v])
        eid = len(edges)
        edges[eid] = (va, vb, 3 - c1 - c2)
        elook[es] = eid

    deg = defaultdict(set)
    for va, vb, c in edges.values():
        deg[va].add(c)
        deg[vb].add(c)
    assert all(len(cs) == 3 for cs in deg.values()), "vertex color degree != 3"
    for f in faces.values():
        cs = [edges[elook[es]][2] for es in f["edge_sets"]]
        assert sorted(set(cs)) == sorted({(f["color"] + 1) % 3, (f["color"] + 2) % 3})
        assert all(cs[k] != cs[(k + 1) % 6] for k in range(6))
    return verts, vid, edges, elook, faces


def candidate_vertex_maps(L, M):
    def T(di, dj):
        def f(v):
            s, i, j = v
            return (s, (i + di) % L, (j + dj) % M)
        return f

    def C2_face(v):  # 180deg about the center of face f(0,0)
        s, i, j = v
        return (("B" if s == "A" else "A"), (-1 - i) % L, (1 - j) % M)

    def C2_edge(v):  # 180deg about the center of z-edge A(0,0)-B(0,0)
        s, i, j = v
        return (("B" if s == "A" else "A"), (-i) % L, (-j) % M)

    def C3_face(v):  # 120deg about the center of face f(0,0) — color-PRESERVING
        s, i, j = v
        if s == "A":
            return ("A", (-i - j) % L, (i + 1) % M)
        return ("B", (-i - j - 1) % L, (i + 1) % M)

    def C3_vertex(v):  # 120deg about vertex A(0,0) — cycles colors (spacetime screw)
        s, i, j = v
        if s == "A":
            return ("A", (-i - j) % L, i % M)
        return ("B", (-i - j - 1) % L, i % M)

    return {"T11": T(1, 1), "T22": T(2, 2), "C3_face": C3_face,
            "C3_vertex": C3_vertex, "C2_face": C2_face, "C2_edge": C2_edge}


def derive_action(vmap, edges, elook, faces):
    eperm, eswapfix = {}, []
    for eid, (va, vb, c) in edges.items():
        es2 = frozenset({vmap(va), vmap(vb)})
        if es2 not in elook:
            return None
        e2 = elook[es2]
        eperm[eid] = e2
        if e2 == eid and vmap(va) != va:
            eswapfix.append(eid)
    fvsets = {fk: frozenset(f["verts"]) for fk, f in faces.items()}
    flook = {vs: fk for fk, vs in fvsets.items()}
    fperm = {}
    for fk, vs in fvsets.items():
        vs2 = frozenset(vmap(v) for v in vs)
        if vs2 not in flook:
            return None
        fperm[fk] = flook[vs2]
    return eperm, eswapfix, fperm


def orbit_orders(edges, elook, vmaps_named):
    """Equivariant 'first endpoint' per edge under the given maps; returns
    (orders, conflicts). A conflict means no covariant assignment exists there."""
    orders, conflicts = {}, []
    todo = set(edges.keys())
    while todo:
        seed = min(todo)
        orders[seed] = edges[seed][0]
        queue = [seed]
        seen = {seed}
        while queue:
            e = queue.pop()
            ea, eb, _ = edges[e]
            first = orders[e]
            for name, vmap in vmaps_named.items():
                e2 = elook[frozenset({vmap(ea), vmap(eb)})]
                f2 = vmap(first)
                if e2 in orders:
                    if orders[e2] != f2:
                        conflicts.append((name, e2))
                else:
                    orders[e2] = f2
                    queue.append(e2)
                    seen.add(e2)
        todo -= seen
        todo -= set(orders.keys()) & todo
    return orders, conflicts


def window_phase(cf):
    want = {(cf + 1) % 3, (cf + 2) % 3}
    for q in range(3):
        if {q % 3, (q + 1) % 3} == want:
            return q
    raise AssertionError


def build_circuit(L, M, R, p, mode, edges, elook, faces, vid, orders=None):
    nv = len(vid)
    data = list(range(nv))
    by_color = defaultdict(list)
    for eid in sorted(edges):
        by_color[edges[eid][2]].append(eid)
    npc = len(by_color[0])
    assert all(len(v) == npc for v in by_color.values())
    anc0 = nv
    S = 3 * R

    c = stim.Circuit()
    for v, k in sorted(vid.items(), key=lambda kv: kv[1]):
        c.append("QUBIT_COORDS", k, ("AB".index(v[0]), v[1], v[2]))
    c.append("RX", data)

    rec = {}
    nrec = 0
    bodies = []
    for s in range(S):
        col = s % 3
        P = PAULI[col]
        body = stim.Circuit()
        if p:
            body.append("DEPOLARIZE1", data, p)
        if mode == "mpp":
            for eid in by_color[col]:
                va, vb, _ = edges[eid]
                t = [PTARG[P](vid[va]), stim.target_combiner(), PTARG[P](vid[vb])]
                if p:
                    body.append("MPP", t, p)
                else:
                    body.append("MPP", t)
                rec[(eid, s)] = nrec
                nrec += 1
        else:
            for k_e, eid in enumerate(by_color[col]):
                a = anc0 + k_e
                va, vb, _ = edges[eid]
                if orders is not None:
                    first = orders[eid]
                else:
                    first = va if va[0] == "A" else vb
                second = vb if first == va else va
                body.append("RX", [a])
                if p:
                    body.append("Z_ERROR", [a], p)
                for d in (first, second):
                    body.append(PGATE[P], [a, vid[d]])
                    if p:
                        body.append("DEPOLARIZE2", [a, vid[d]], p)
                if p:
                    body.append("Z_ERROR", [a], p)
                body.append("MX", [a])
                rec[(eid, s)] = nrec
                nrec += 1
        bodies.append(body)

    det_meta = []
    full = c
    for s in range(S):
        full += bodies[s]
        made = (s + 1) * npc  # records emitted so far
        for fk in sorted(faces):
            f = faces[fk]
            cf = f["color"]
            ph = window_phase(cf)
            if (s - ph - 1) % 3 != 0:
                continue
            k = (s - ph - 1) // 3
            if k < 1:
                continue
            recs = []
            for kk in (k - 1, k):
                s0, s1 = 3 * kk + ph, 3 * kk + ph + 1
                for es in f["edge_sets"]:
                    eid = elook[es]
                    col = edges[eid][2]
                    ss = s0 if (s0 % 3) == col else s1
                    recs.append(rec[(eid, ss)])
            full.append("DETECTOR", [stim.target_rec(r - made) for r in recs],
                        (fk[0], fk[1], cf, s))
            det_meta.append((fk, cf, s))
    return full, det_meta


def dem_groups(circ):
    dem = circ.detector_error_model(flatten_loops=True)
    groups = defaultdict(list)
    off = 0
    for inst in dem:
        if inst.type == "shift_detectors":
            t0 = inst.targets_copy()[0]
            off += t0.val if hasattr(t0, "val") else int(t0)
        elif inst.type == "error":
            pr = inst.args_copy()[0]
            dets = frozenset(t.val + off for t in inst.targets_copy()
                             if t.is_relative_detector_id())
            if dets:
                groups[dets].append(round(pr, 9))
    return {k: tuple(sorted(v)) for k, v in groups.items()}


def detector_perm(det_meta, fperm, cmap, ds_by_color, domain=None):
    index = {meta: k for k, meta in enumerate(det_meta)}
    perm = {}
    for k, (fk, cf, s) in enumerate(det_meta):
        if domain is not None and k not in domain:
            continue
        key2 = (fperm[fk], cmap[cf], s + ds_by_color[cf])
        if key2 not in index:
            return None
        perm[k] = index[key2]
    if len(set(perm.values())) != len(perm):
        return None
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.008)
    ap.add_argument("--bulk-margin", type=int, default=3,
                    help="sub-rounds excluded at each time end for screw tests; "
                    "CX extraction's init transient needs ~6")
    args = ap.parse_args()
    L, M, R, p = args.L, args.M, args.R, args.p
    T0 = time.time()

    verts, vid, edges, elook, faces = build_lattice(L, M)
    print(f"lattice OK: {len(verts)} data qubits, {len(edges)} edges, "
          f"{len(faces)} faces (S1 passed)", flush=True)

    vmaps = candidate_vertex_maps(L, M)
    actions, cmaps = {}, {}
    for name, vmap in vmaps.items():
        act = derive_action(vmap, edges, elook, faces)
        assert act is not None, f"{name} not an automorphism"
        eperm, eswapfix, fperm = act
        actions[name] = act
        cm = {}
        for fk, fk2 in fperm.items():
            c1, c2 = faces[fk]["color"], faces[fk2]["color"]
            assert cm.setdefault(c1, c2) == c2, f"{name}: inconsistent color action"
        cmaps[name] = cm
        print(f"  {name}: colors {cm}, endpoint-swapping fixed edges: "
              f"{len(eswapfix)}", flush=True)

    sym_generators = {n: vmaps[n] for n in ("T11", "T22", "C3_face", "C3_vertex")}
    orders, conflicts = orbit_orders(edges, elook, sym_generators)
    assert len(orders) == len(edges)
    print(f"orbit-equivariant orders under (T11,T22,C3_face,C3_vertex): "
          f"conflicts={len(conflicts)}", flush=True)

    results = {"params": vars(args), "modes": {}}
    for mode, osel in (("mpp", None), ("cx-naive", None), ("cx-sym", "sym")):
        stim_mode = "mpp" if mode == "mpp" else "cx"
        ords = orders if osel == "sym" else None
        circ, det_meta = build_circuit(L, M, R, p, stim_mode, edges, elook,
                                       faces, vid, orders=ords)
        circ0, _ = build_circuit(L, M, R, 0.0, stim_mode, edges, elook,
                                 faces, vid, orders=ords)
        dets0, _ = circ0.compile_detector_sampler().sample(
            256, separate_observables=True)
        g2 = bool(not dets0.any())
        groups = dem_groups(circ)
        print(f"\n== {mode}: {circ.num_detectors} detectors, G2={g2}, "
              f"{len(groups)} DEM mechanism groups ==", flush=True)
        mres = {"g2": g2, "n_detectors": circ.num_detectors,
                "n_mechanisms": len(groups)}
        if not g2:
            results["modes"][mode] = mres
            continue

        all_s = [s for _, _, s in det_meta]
        bm = args.bulk_margin
        bulk = {k for k, (_, _, s) in enumerate(det_meta)
                if min(all_s) + bm <= s <= max(all_s) - bm}

        for name, (eperm, eswapfix, fperm) in actions.items():
            cm = cmaps[name]
            verdict = {}
            # classify the color action: identity -> pure spatial (full-DEM test);
            # cyclic shift -> spacetime screw, uniform time shift (time-bulk test);
            # transposition -> only a symmetry composed with TIME REVERSAL (the
            # period-3 schedule is chiral in time) — classified, not tested here.
            shifts = {(cm[cf] - cf) % 3 for cf in range(3)}
            if shifts == {0}:
                ds = {cf: 0 for cf in range(3)}
                perm = detector_perm(det_meta, fperm, cm, ds, domain=None)
                verdict["class"] = "spatial"
                verdict["scope"] = "full"
                verdict["exact"] = perm is not None and bool(all(
                    groups.get(frozenset(perm[d] for d in dets)) == pr
                    for dets, pr in groups.items()))
                verdict["n_checked"] = len(groups)
            elif len(shifts) == 1:
                dphase = shifts.pop()
                duni = ((dphase + 1) % 3) - 1  # lift to -1/+1
                ds = {cf: duni for cf in range(3)}
                permb = detector_perm(det_meta, fperm, cm, ds, domain=bulk)
                verdict["class"] = f"screw(dt={duni})"
                if permb is None:
                    verdict["scope"] = "none"
                else:
                    gb = {dets: pr for dets, pr in groups.items()
                          if dets <= set(permb.keys())}
                    verdict["scope"] = "time-bulk"
                    mism = []
                    for dets, pr in gb.items():
                        pr2 = groups.get(frozenset(permb[d] for d in dets))
                        if pr2 != pr:
                            mism.append((sorted(det_meta[d] for d in dets), pr, pr2))
                    verdict["exact"] = not mism
                    verdict["n_checked"] = len(gb)
                    verdict["n_mismatch"] = len(mism)
                    for m in mism[:4]:
                        print(f"      mismatch: dets={m[0]} probs {m[1]} -> {m[2]}",
                              flush=True)
            else:
                verdict["class"] = "time-reversal-paired (transposes 2 colors)"
                verdict["scope"] = "not tested"
            mres[name] = verdict
            print(f"  {name}: {verdict}", flush=True)
        results["modes"][mode] = mres

    results["runtime_s"] = time.time() - T0
    with open("runs/honeycomb_code_symmetry.json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    print(f"\nDONE {results['runtime_s']:.0f}s -> runs/honeycomb_code_symmetry.json",
          flush=True)


if __name__ == "__main__":
    main()
