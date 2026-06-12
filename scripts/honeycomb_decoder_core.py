"""
HONEYCOMB FLOQUET CODE -> EQUIVARIANT DECODER (2026-06-12).

The build the symmetry result pointed at (runs/HONEYCOMB_CODE_SYMMETRY.md next-step 1):
add LOGICAL OBSERVABLES to the Hastings-Haah honeycomb circuit, solve the twisted
symmetry action including the observables, and train symmetry-tied decoders on the
code whose circuit-level DEM carries an EXACT p3 space group (the surface code
provably doesn't — 999-map + 1,728-schedule nulls).

New machinery vs the surface-code pipeline (equiv_decoder_d5b.py):
  1. TERMINATED memory experiment: init detectors (s=0 XX records are individually
     deterministic on |+>^n) + final transversal MX readout + "virtual sub-round"
     detectors (color-0 edge values recomputed from data-record pairs complete the
     in-flight color-1 face windows) — so early/late errors are not undetectable.
  2. Logical observables DERIVED, not hand-built: GF(2) null space of the noiseless
     measurement-record matrix = all deterministic record parities; quotient by the
     detector span = the trackable logicals (expect exactly 2 on the torus).
  3. GENERALIZED twist: on the torus C3 acts on H1 by an order-3 matrix, so a single
     logical is NOT mapped to itself x stabilizers — the action is
        (dets, obs) -> (P dets, A obs ⊕ X·(P dets)),  A ∈ GL(2,F2), X = 2 chi rows,
     and the decoder is a 4-CLASS (joint 2-bit) twisted-equivariant model:
        f(x)[c] = (1/|G|) Σ_g mlp(x̃_g)[ A_g c ⊕ X_g·x̃_g ],  x̃_g = scatter_g(x).
     A and X are solved from the exact DEM per det-set group (multiset method) and
     verified mechanism-by-mechanism before use. Gauge note: X is unique only modulo
     directions invisible to every mechanism det-set; therefore equivariance is exact
     on REALIZABLE syndromes (all train/test data) and the probe uses sampled shots.

PREDICTIONS LOGGED BEFORE ANY RUN (hard rule — no result recorded before completion):
  P1 gates pass: G2 determinism incl. init + virtual-round detectors and the derived
     observables; exactly 2 record logicals on the torus after init-augmentation.
  P2 translations solve with A = I (homology preserved); C3 REQUIRES A != I — the
     homology-mixing twist is the new structural fact.
  P3 ordering at scarce n: p3_9 >= c3tw >= plain (more exact symmetry, same params).
  P4 (uncertain): absolute gaps may differ from surface d=5 (~300 dets here, easier
     task per detector; honest either way).

Group for tying (L=M=6): p3_9 = {C3^a T(2,2)^b} — C3 T22 C3^-1 = T(R(2,2)) = T(2,2)
(R(2,2)=(-4,2)≡(2,2) mod 6), so this is a closed abelian order-9 subgroup of p3.
"""
import time
from collections import defaultdict

import numpy as np
import stim

PAULI = "XYZ"
PTARG = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
PGATE = {"X": "CX", "Y": "CY", "Z": "CZ"}


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------- lattice (verbatim
# from honeycomb_code_symmetry.py, S1 self-checks included)
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
        assert len(fks) == 2
        c1, c2 = faces[fks[0]]["color"], faces[fks[1]]["color"]
        assert c1 != c2
        va, vb = sorted(es, key=lambda v: vid[v])
        edges[len(edges)] = (va, vb, 3 - c1 - c2)
        elook[es] = len(edges) - 1

    deg = defaultdict(set)
    for va, vb, c in edges.values():
        deg[va].add(c)
        deg[vb].add(c)
    assert all(len(cs) == 3 for cs in deg.values())
    return verts, vid, edges, elook, faces


def vertex_maps(L, M):
    """The 8 non-identity elements of p3_9 (all color-preserving)."""
    def T(di, dj):
        def f(v):
            s, i, j = v
            return (s, (i + di) % L, (j + dj) % M)
        return f

    def C3(v):  # 120deg about face f(0,0) center — color-preserving (verified prior)
        s, i, j = v
        if s == "A":
            return ("A", (-i - j) % L, (i + 1) % M)
        return ("B", (-i - j - 1) % L, (i + 1) % M)

    def comp(f, g):
        return lambda v: f(g(v))

    C3sq = comp(C3, C3)
    els = {"C3": C3, "C3sq": C3sq, "T22": T(2, 2), "T44": T(4, 4)}
    for a, fa in (("C3", C3), ("C3sq", C3sq)):
        for b, fb in (("T22", T(2, 2)), ("T44", T(4, 4))):
            els[f"{a}.{b}"] = comp(fa, fb)
    return els


def derive_action(vmap, edges, elook, faces):
    eperm = {}
    for eid, (va, vb, c) in edges.items():
        es2 = frozenset({vmap(va), vmap(vb)})
        if es2 not in elook:
            return None
        eperm[eid] = elook[es2]
    fvsets = {fk: frozenset(f["verts"]) for fk, f in faces.items()}
    flook = {vs: fk for fk, vs in fvsets.items()}
    fperm = {}
    for fk, vs in fvsets.items():
        vs2 = frozenset(vmap(v) for v in vs)
        if vs2 not in flook:
            return None
        fperm[fk] = flook[vs2]
    return eperm, fperm


def window_phase(cf):
    want = {(cf + 1) % 3, (cf + 2) % 3}
    for q in range(3):
        if {q % 3, (q + 1) % 3} == want:
            return q
    raise AssertionError


# ---------------------------------------------------------------- circuit w/ readout
def build_circuit(L, M, R, p, mode, edges, elook, faces, vid, obs_recs=None,
                  s_extra=0, extra_det_recs=None):
    """Honeycomb memory circuit, |+>^n init, S=3R+s_extra sub-rounds, terminated
    by transversal M{X,Y,Z} readout in the basis of the NEXT (virtual) sub-round's
    color cr=S%3. Structural detectors: s=0 edge records (X-type, |+>^n); k=0
    windows of color-0 faces (X-type face stabilizers); bulk window comparisons;
    virtual-round closures; final window-vs-data comparisons for color-cr faces.
    Returns (circuit, det_meta, det_recs, ntotal_records)."""
    nv = len(vid)
    data = list(range(nv))
    by_color = defaultdict(list)
    for eid in sorted(edges):
        by_color[edges[eid][2]].append(eid)
    anc0 = nv
    S = 3 * R + s_extra
    cr = S % 3

    c = stim.Circuit()
    for v, k in sorted(vid.items(), key=lambda kv: kv[1]):
        c.append("QUBIT_COORDS", k, ("AB".index(v[0]), v[1], v[2]))
    c.append("RX", data)

    rec, nrec = {}, 0
    det_meta, det_recs = [], []

    def window_recs(fk, k, rec_data=None):
        """Records of window k of face fk; None if not fully available."""
        f = faces[fk]
        ph = window_phase(f["color"])
        s0 = 3 * k + ph
        recs = []
        for es in f["edge_sets"]:
            eid = elook[es]
            col = edges[eid][2]
            ss = s0 if (s0 % 3) == col else s0 + 1
            if ss < S:
                recs.append(rec[(eid, ss)])
            elif ss == S and col == cr and rec_data is not None:
                va, vb, _ = edges[eid]
                recs += [rec_data[va], rec_data[vb]]
            else:
                return None
        return recs

    KIND = {"edge": 0, "face": 1, "face0init": 2, "facefin": 3, "xtra": 4}

    def emit(recs, meta):
        kind, key, cf, s = meta
        ij = (key, -1) if kind in ("edge", "xtra") else key
        c.append("DETECTOR", [stim.target_rec(r - nrec) for r in recs],
                 (KIND[kind], ij[0], ij[1], cf, s))
        det_meta.append(meta)
        det_recs.append(tuple(recs))

    def emit_face_comparisons(s, rec_data=None):
        for fk in sorted(faces):
            cf = faces[fk]["color"]
            ph = window_phase(cf)
            if (s - ph - 1) % 3 != 0:
                continue
            k = (s - ph - 1) // 3
            if k < 1:
                continue
            r1 = window_recs(fk, k - 1, rec_data)
            r2 = window_recs(fk, k, rec_data)
            if r1 is None or r2 is None:
                continue
            emit(r1 + r2, ("face", fk, cf, s))

    for s in range(S):
        col = s % 3
        P = PAULI[col]
        if p:
            c.append("DEPOLARIZE1", data, p)
        if mode == "mpp":
            for eid in by_color[col]:
                va, vb, _ = edges[eid]
                t = [PTARG[P](vid[va]), stim.target_combiner(), PTARG[P](vid[vb])]
                if p:
                    c.append("MPP", t, p)
                else:
                    c.append("MPP", t)
                rec[(eid, s)] = nrec
                nrec += 1
        else:
            for k_e, eid in enumerate(by_color[col]):
                a = anc0 + k_e
                va, vb, _ = edges[eid]
                first = va if va[0] == "A" else vb  # naive uniform order
                second = vb if first == va else va
                c.append("RX", [a])
                if p:
                    c.append("Z_ERROR", [a], p)
                for d in (first, second):
                    c.append(PGATE[P], [a, vid[d]])
                    if p:
                        c.append("DEPOLARIZE2", [a, vid[d]], p)
                if p:
                    c.append("Z_ERROR", [a], p)
                c.append("MX", [a])
                rec[(eid, s)] = nrec
                nrec += 1
        if s == 0:
            # init edge detectors: each s=0 XX record deterministic on |+>^n
            for eid in by_color[0]:
                emit((rec[(eid, 0)],), ("edge", eid, 0, 0))
        if s == 2:
            # init face detectors: k=0 window of each color-0 face — the face
            # stabilizer is X-type (Y·Z ∝ X per vertex), deterministic on |+>^n
            for fk in sorted(faces):
                if faces[fk]["color"] == 0:
                    r0 = window_recs(fk, 0)
                    if r0 is not None:
                        emit(r0, ("face0init", fk, 0, 2))
        emit_face_comparisons(s)

    # final transversal readout in the virtual round's basis
    if p:
        c.append("DEPOLARIZE1", data, p)
    rec_data = {}
    for v, k in sorted(vid.items(), key=lambda kv: kv[1]):
        if p:
            c.append(f"M{PAULI[cr]}", [k], p)
        else:
            c.append(f"M{PAULI[cr]}", [k])
        rec_data[v] = nrec
        nrec += 1
    # virtual sub-round S closes in-flight windows of faces lacking color cr
    emit_face_comparisons(S, rec_data=rec_data)
    # final face detectors: color-cr faces' last window vs data product (their
    # stabilizer is readout-Pauli-type per vertex, so the comparison closes)
    for fk in sorted(faces):
        if faces[fk]["color"] == cr:
            k_last = (S - 2 - window_phase(cr)) // 3
            rl = window_recs(fk, k_last)
            if rl is None:
                continue
            dr = [rec_data[v] for v in faces[fk]["verts"]]
            emit(rl + dr, ("facefin", fk, cr, S))

    # auto-folded structural detectors (zero-data deterministic parities found
    # empirically by derive_logicals and re-declared as detectors)
    if extra_det_recs:
        for i, recs in enumerate(extra_det_recs):
            emit(tuple(recs), ("xtra", i, -1, S))

    if obs_recs is not None:
        for k, orecs in enumerate(obs_recs):
            c.append("OBSERVABLE_INCLUDE",
                     [stim.target_rec(r - nrec) for r in sorted(orecs)], k)
    aux = {"rec": rec, "rec_data": rec_data, "S": S, "nv": nv}
    return c, det_meta, det_recs, nrec, aux


# ---------------------------------------------------------------- GF(2) machinery
def gf2_echelon(rows):
    ech, piv = [], []
    for r in rows:
        r = r.copy()
        for e, pc in zip(ech, piv):
            if r[pc]:
                r ^= e
        nz = np.nonzero(r)[0]
        if len(nz):
            ech.append(r)
            piv.append(int(nz[0]))
    return ech, piv


def gf2_reduce(v, ech, piv):
    v = v.copy()
    for e, pc in zip(ech, piv):
        if v[pc]:
            v ^= e
    return v


def gf2_nullspace(D):
    """Null space basis of bool matrix D (m x n): {w : D w = 0}."""
    m, n = D.shape
    A = D.copy()
    piv_of_col = {}
    r = 0
    for col in range(n):
        if r >= m:
            break
        rows = np.nonzero(A[r:, col])[0]
        if len(rows) == 0:
            continue
        A[[r, r + rows[0]]] = A[[r + rows[0], r]]
        mask = A[:, col].copy()
        mask[r] = False
        A[mask] ^= A[r]
        piv_of_col[col] = r
        r += 1
    free = [c for c in range(n) if c not in piv_of_col]
    basis = []
    for fc in free:
        w = np.zeros(n, dtype=bool)
        w[fc] = True
        for col, rr in piv_of_col.items():
            if A[rr, fc]:
                w[col] = True
        basis.append(w)
    return basis


def solve_gf2(A, b):
    """Solve A x = b over GF(2) (vectorized elimination). None if inconsistent."""
    A = (A.copy() % 2).astype(np.int8)
    b = (b.copy() % 2).astype(np.int8)
    m, n = A.shape
    piv, r = [], 0
    for col in range(n):
        if r >= m:
            break
        rows = np.nonzero(A[r:, col])[0]
        if len(rows) == 0:
            continue
        A[[r, r + rows[0]]] = A[[r + rows[0], r]]
        b[[r, r + rows[0]]] = b[[r + rows[0], r]]
        mask = A[:, col].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        b[mask] ^= b[r]
        piv.append(col)
        r += 1
    if np.any(b[r:]):
        return None
    x = np.zeros(n, dtype=np.int64)
    for k, col in enumerate(piv):
        x[col] = b[k]
    return x


def derive_logicals(circ0, det_recs, nrec, shots=1024):
    """Deterministic record parities (empirical null space) modulo detector span.
    Returns (logical bit-vectors, det rank, deterministic dim)."""
    Mmat = circ0.compile_sampler(seed=0).sample(shots)
    D = (Mmat ^ Mmat[0])[1:]
    null = gf2_nullspace(D.astype(bool))
    det_rows = []
    for recs in det_recs:
        v = np.zeros(nrec, dtype=bool)
        for r in recs:
            v[r] ^= True
        det_rows.append(v)
    ech, piv = gf2_echelon(det_rows)
    logicals, lech, lpiv = [], [], []
    for w in null:
        rem = gf2_reduce(w, ech, piv)
        rem = gf2_reduce(rem, lech, lpiv)
        nz = np.nonzero(rem)[0]
        if len(nz):
            lech.append(rem)
            lpiv.append(int(nz[0]))
            logicals.append(rem)
    cleaned = []
    for w in logicals:
        w = w.copy()
        for _ in range(3):
            improved = False
            for e in ech:
                if (w & e).sum() * 2 > e.sum():
                    w = w ^ e
                    improved = True
            if not improved:
                break
        cleaned.append(w)
    return cleaned, len(ech), len(null)


# ---------------------------------------------------------------- DEM + (P, A, X)
def dem_errors(c):
    dem = c.detector_error_model(flatten_loops=True)
    errs, off = [], 0
    for inst in dem:
        if inst.type == "shift_detectors":
            t0 = inst.targets_copy()[0]
            off += t0.val if hasattr(t0, "val") else int(t0)
        elif inst.type == "error":
            p = inst.args_copy()[0]
            dets, ob = [], 0
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val + off)
                elif t.is_logical_observable_id():
                    ob ^= 1 << t.val
            errs.append((frozenset(dets), ob, round(p, 9)))
    return errs


GL2 = [np.array(a, dtype=np.int64) for a in
       ([[1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 1], [0, 1]],
        [[1, 0], [1, 1]], [[0, 1], [1, 1]], [[1, 1], [1, 0]])]


def apply_A(Amat, ob):
    o = np.array([ob & 1, (ob >> 1) & 1], dtype=np.int64)
    o2 = Amat.dot(o) % 2
    return int(o2[0] | (o2[1] << 1))


def solve_element(perm, by_v, nd):
    """Find (A, X) with DEM-multiset equality under
    (v, o, p) -> (perm v, A o ⊕ X·(perm v), p). Returns (A, X(2,nd)) or None."""
    def ms(g):
        return tuple(sorted(g))

    for Amat in GL2:
        eq_rows, eq_t = ([], []), ([], [])
        ok = True
        for v, group in by_v.items():
            v2 = frozenset(int(perm[d]) for d in v)
            tgt = by_v.get(v2)
            if tgt is None:
                ok = False
                break
            tgt_ms = ms(tgt)
            cons = [b for b in range(4)
                    if ms([(apply_A(Amat, o) ^ b, p) for o, p in group]) == tgt_ms]
            if not cons:
                ok = False
                break
            row = None
            for bit in (0, 1):
                vals = {(b >> bit) & 1 for b in cons}
                if len(vals) == 1:
                    if row is None:
                        row = np.zeros(nd, dtype=np.int8)
                        for d in v2:
                            row[d] = 1
                    eq_rows[bit].append(row)
                    eq_t[bit].append(vals.pop())
        if not ok:
            continue
        X, good = [], True
        for bit in (0, 1):
            if eq_rows[bit]:
                sol = solve_gf2(np.array(eq_rows[bit]),
                                np.array(eq_t[bit], dtype=np.int8))
                if sol is None:
                    good = False
                    break
                X.append(sol)
            else:
                X.append(np.zeros(nd, dtype=np.int64))
        if not good:
            continue
        X = np.stack(X)
        full = all(
            ms(by_v[frozenset(int(perm[d]) for d in v)]) ==
            ms([(apply_A(Amat, o) ^
                 int(X[0][[int(perm[d]) for d in v]].sum() % 2) ^
                 (int(X[1][[int(perm[d]) for d in v]].sum() % 2) << 1), p)
                for o, p in g])
            for v, g in by_v.items())
        if full:
            return Amat, X
    return None


def record_perm(vmap, eperm, aux, nrec):
    """Permutation of measurement-record indices induced by a lattice map."""
    rperm = np.full(nrec, -1, dtype=np.int64)
    for (eid, s), r in aux["rec"].items():
        rperm[r] = aux["rec"][(eperm[eid], s)]
    for v, r in aux["rec_data"].items():
        rperm[r] = aux["rec_data"][vmap(v)]
    assert (rperm >= 0).all() and len(set(rperm.tolist())) == nrec
    return rperm


def detector_perm_array(det_recs, rperm):
    """Detector permutation via record-set matching (a detector IS its records)."""
    index = {frozenset(recs): k for k, recs in enumerate(det_recs)}
    perm = np.zeros(len(det_recs), dtype=np.int64)
    for k, recs in enumerate(det_recs):
        key2 = frozenset(int(rperm[r]) for r in recs)
        if key2 not in index:
            return None
        perm[k] = index[key2]
    if len(set(perm.tolist())) != len(det_recs):
        return None
    return perm


def matching_from(c):
    """MWPM matcher; falls back to ignoring undecomposable hyperedges plus
    high-weight boundary escapes (prevents odd-parity failures on the torus).
    Returns (matcher, fallback_used)."""
    import pymatching
    try:
        dem = c.detector_error_model(decompose_errors=True)
        return pymatching.Matching.from_detector_error_model(dem), False
    except Exception as e:
        log(f"  decompose_errors failed ({type(e).__name__}) — "
            f"ignore_decomposition_failures + boundary escapes "
            f"(MWPM slightly pessimistic)")
        dem = c.detector_error_model(decompose_errors=True,
                                     ignore_decomposition_failures=True)
        m = pymatching.Matching.from_detector_error_model(dem)
        for i in range(m.num_detectors):
            m.add_boundary_edge(i, weight=60.0,
                                merge_strategy="smallest-weight")
        return m, True


# ---------------------------------------------------------------- build everything
def build_all(L=6, M=6, R=9, p=0.008, mode="cx", auto_p=True, s_extra=0):
    res = {"params": dict(L=L, M=M, R=R, p=p, mode=mode, s_extra=s_extra)}
    verts, vid, edges, elook, faces = build_lattice(L, M)
    log(f"lattice: {len(verts)} data qubits, {len(edges)} edges, {len(faces)} faces"
        f" | termination s_extra={s_extra} (readout {PAULI[(3 * R + s_extra) % 3]})")

    c0, det_meta, det_recs, nrec, aux = build_circuit(
        L, M, R, 0.0, mode, edges, elook, faces, vid, s_extra=s_extra)
    dets0 = c0.compile_detector_sampler(seed=1).sample(256)
    g2 = bool(not dets0.any())
    log(f"G2 (noiseless determinism incl. init+virtual detectors): {g2} "
        f"({len(det_meta)} detectors, {nrec} records)")
    assert g2, "G2 FAILED — detector construction wrong"

    nv = aux["nv"]
    logicals, dim_det, dim_all = derive_logicals(c0, det_recs, nrec)

    def datasup(w):
        return int(w[nrec - nv:].sum())

    trivial = [w for w in logicals if datasup(w) == 0]
    wraps = [w for w in logicals if datasup(w) > 0]
    log(f"deterministic space: dim {dim_all} = det rank {dim_det} + "
        f"{len(trivial)} trivial (zero-data, dropped — see comment) + "
        f"{len(wraps)} wraps; wrap weights "
        f"{[int(w.sum()) for w in wraps]}, data support "
        f"{[datasup(w) for w in wraps]}")
    assert len(wraps) == 2, \
        f"expected exactly 2 wrap logicals at this termination, got {len(wraps)}" \
        f" — wrong s_extra? (logical types rotate; only one termination carries them)"
    # trivial zero-data parities (e.g. the global last-XX-round product) are NOT
    # added as detectors: a global-support detector turns every boundary-layer
    # error into an undecomposable hyperedge and carries ~no decoding signal.
    obs_recs = [np.nonzero(w)[0].tolist() for w in wraps]
    res.update(n_detectors=len(det_meta), n_records=nrec,
               logical_weights=[int(w.sum()) for w in wraps],
               n_trivial_dropped=len(trivial), det_rank=dim_det)

    def mk(pp):
        return build_circuit(L, M, R, pp, mode, edges, elook, faces, vid,
                             obs_recs=obs_recs, s_extra=s_extra)[0]

    c0o = mk(0.0)
    d0, o0 = c0o.compile_detector_sampler(seed=2).sample(
        256, separate_observables=True)
    assert not d0.any() and not o0.any(), "obs not deterministic noiselessly"
    log("observables: deterministic noiselessly ✓")

    p_use = p
    while True:
        c = mk(p_use)
        errs = dem_errors(c)
        n_undet = sum(1 for v, o, _ in errs if not v and o)
        m, mwpm_fb = matching_from(c)
        ds, os_ = c.compile_detector_sampler(seed=3).sample(
            20000, separate_observables=True)
        pred = m.decode_batch(ds)
        acc = float(np.mean(np.all(pred == os_, axis=1)))
        flip = float(np.mean(os_.any(axis=1)))
        log(f"p={p_use}: MWPM joint acc {acc:.4f}, obs-flip rate {flip:.4f}, "
            f"undetectable-logical mechanisms: {n_undet}")
        if not auto_p or 0.85 <= acc <= 0.995 or p_use <= 0.00075:
            break
        p_use = round(p_use / 2, 6)
    res.update(p_used=p_use, mwpm_probe=acc, flip_rate=flip,
               n_undetectable=n_undet, mwpm_fallback=mwpm_fb)

    by_v = defaultdict(list)
    for v, o, pr in dem_errors(c):
        by_v[v].append((o, pr))
    nd = c.num_detectors
    elements = []
    for name, vmap in vertex_maps(L, M).items():
        act = derive_action(vmap, edges, elook, faces)
        assert act is not None, f"{name}: not an automorphism"
        rperm = record_perm(vmap, act[0], aux, nrec)
        perm = detector_perm_array(det_recs, rperm)
        assert perm is not None, f"{name}: detector permutation failed"
        sol = solve_element(perm, by_v, nd)
        if sol is None:
            log(f"  {name}: NO (A,X) verifies — recorded honestly")
            res.setdefault("failed_elements", []).append(name)
            continue
        Amat, X = sol
        a_id = bool((Amat == np.eye(2, dtype=np.int64)).all())
        log(f"  {name}: VERIFIED  A={'I' if a_id else Amat.tolist()}  "
            f"|chi|=({int(X[0].sum())},{int(X[1].sum())})  "
            f"groups={len(by_v)}")
        elements.append((name, perm, Amat, X))
    res["elements"] = {n: {"A": A.tolist(),
                           "chi_wt": [int(X[0].sum()), int(X[1].sum())]}
                       for n, _, A, X in elements}
    return c, det_meta, elements, res


# ---------------------------------------------------------------- models + training
def make_models():
    import torch
    import torch.nn as nn

    class MLP4(nn.Module):
        def __init__(self, nd, h=128):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nd, h), nn.ReLU(),
                                     nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 4))

        def forward(self, x):
            return self.net(x)

    class GroupInv4(nn.Module):
        """f(x)[c] = (1/|G|) Σ_g mlp(x̃_g)[ A_g c ⊕ X_g·x̃_g ], x̃_g[perm[d]] = x[d]."""
        def __init__(self, nd, elements, h=128):
            super().__init__()
            self.mlp = MLP4(nd, h)
            invps, chis, cmaps = [], [], []
            for perm, Amat, X in elements:
                invp = np.zeros(len(perm), dtype=np.int64)
                invp[perm] = np.arange(len(perm))
                invps.append(torch.tensor(invp, dtype=torch.long))
                chis.append(torch.tensor(np.asarray(X, dtype=np.float32)))
                cmaps.append(torch.tensor([apply_A(np.asarray(Amat), cc)
                                           for cc in range(4)], dtype=torch.long))
            self.register_buffer("IP", torch.stack(invps))
            self.register_buffer("CH", torch.stack(chis))   # (|G|, 2, nd)
            self.register_buffer("CM", torch.stack(cmaps))  # (|G|, 4)

        def forward(self, x):
            outs = []
            for k in range(self.IP.shape[0]):
                xg = x[:, self.IP[k]]
                logit = self.mlp(xg)
                b = torch.remainder(xg @ self.CH[k].T, 2.0).long()
                bidx = b[:, 0] | (b[:, 1] << 1)
                idx = self.CM[k][None, :].expand(x.shape[0], 4) ^ bidx[:, None]
                outs.append(torch.gather(logit, 1, idx))
            return torch.stack(outs).mean(dim=0)

    return MLP4, GroupInv4


def train_eval(model, Xtr, ytr, Xte, yte, dev, steps=4000, bs=256, lr=1e-3):
    import torch
    import torch.nn as nn
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev, dtype=torch.long)
    n = len(ytr)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=dev)
        opt.zero_grad()
        loss = lossf(model(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
    model.eval()
    hits = []
    with torch.no_grad():
        for i in range(0, len(yte), 50000):
            xb = torch.tensor(Xte[i:i + 50000], device=dev)
            pred = model(xb).argmax(dim=1).cpu().numpy()
            hits.append(pred == yte[i:i + 50000])
    return float(np.mean(np.concatenate(hits)))


def run_experiment(L=6, M=6, R=9, p=0.008, mode="cx", s_extra=0,
                   ntrains=(250, 1000, 4000, 16000, 64000),
                   seeds=(0, 1, 2, 3, 4), ntest=200_000, steps=4000,
                   smoke=False):
    import pymatching
    import torch
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if smoke:
        ntrains, seeds, steps, ntest = (250,), (0,), 300, 20000
    c, det_meta, elements, res = build_all(L, M, R, p, mode, s_extra=s_extra)
    nd = c.num_detectors
    MLP4, GroupInv4 = make_models()

    ident = (np.arange(nd), np.eye(2, dtype=np.int64), np.zeros((2, nd), np.int64))
    by = {n: (pm, A, X) for n, pm, A, X in elements}
    need = ["C3", "C3sq", "T22", "T44", "C3.T22", "C3.T44", "C3sq.T22", "C3sq.T44"]
    assert all(n in by for n in need), f"missing elements: {set(need) - set(by)}"
    c3_elems = [ident, by["C3"], by["C3sq"]]
    p3_elems = [ident] + [by[n] for n in need]

    def sample(n, seed):
        s = c.compile_detector_sampler(seed=seed)
        ds, os_ = s.sample(n, separate_observables=True)
        y = (os_[:, 0].astype(np.int64) | (os_[:, 1].astype(np.int64) << 1))
        return ds.astype(np.float32), y

    # invariance probe on REALIZABLE syndromes (X gauge is fixed only mod
    # mechanism-invisible directions, so random bit-vectors would be unfair)
    Xp, _ = sample(64, seed=777)
    probe = GroupInv4(nd, p3_elems).to(dev)
    xb = torch.tensor(Xp, device=dev)
    worst = 0.0
    with torch.no_grad():
        fx = probe(xb)
        for pm, A, X in [by[n] for n in need]:
            invp = np.zeros(nd, dtype=np.int64)
            invp[pm] = np.arange(nd)
            xg = xb[:, torch.tensor(invp, device=dev)]
            fg = probe(xg)
            bb = torch.remainder(
                xg @ torch.tensor(np.asarray(X, dtype=np.float32), device=dev).T,
                2.0).long()
            bidx = bb[:, 0] | (bb[:, 1] << 1)
            cm = torch.tensor([apply_A(np.asarray(A), cc) for cc in range(4)],
                              device=dev)
            idx = cm[None, :].expand(xb.shape[0], 4) ^ bidx[:, None]
            worst = max(worst, float((fx - torch.gather(fg, 1, idx)).abs().max()))
    log(f"twisted-average invariance probe on realizable syndromes: {worst:.2e}")
    res["invariance_probe"] = worst
    assert worst < 1e-4, "equivariant construction violated — convention bug"

    Xte, yte = sample(ntest, seed=999)
    m, _ = matching_from(c)
    pred = m.decode_batch(Xte.astype(bool))
    mw = float(np.mean((pred[:, 0].astype(np.int64) |
                        (pred[:, 1].astype(np.int64) << 1)) == yte))
    maj = float(np.max(np.bincount(yte, minlength=4)) / len(yte))
    nparams = sum(pp.numel() for pp in MLP4(nd).parameters())
    log(f"test: MWPM joint {mw:.4f}, majority {maj:.4f}, params/branch {nparams}, "
        f"nd={nd}, dev={dev}")
    res.update(mwpm=mw, majority=maj, params=nparams, device=dev, sweep={})

    for n in ntrains:
        row = {}
        for arm, mkm in (("plain", lambda: MLP4(nd)),
                         ("c3tw", lambda: GroupInv4(nd, c3_elems)),
                         ("p3_9", lambda: GroupInv4(nd, p3_elems))):
            accs = []
            for seed in seeds:
                torch.manual_seed(seed)
                Xtr, ytr = sample(n, seed=17 * seed + 5)
                accs.append(train_eval(mkm(), Xtr, ytr, Xte, yte, dev, steps=steps))
            row[arm] = (float(np.mean(accs)), float(np.std(accs)))
        res["sweep"][n] = row
        log(f"  n={n:6d}: " + "  ".join(
            f"{a} {row[a][0]:.4f}±{row[a][1]:.4f}" for a in row))
    res["runtime_s"] = time.time() - t0
    return res
