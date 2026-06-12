"""
HONEYCOMB-CODE EQUIVARIANT DECODER — the build on top of the symmetry discovery
(`runs/HONEYCOMB_CODE_SYMMETRY.md`, 2026-06-11). Three stages:

  1. MEMORY CIRCUIT with a logical observable: X-string logical on a non-contractible
     cycle, tracked through the measurement schedule by a Pauli-frame algorithm (each
     check that anticommutes with the current logical is multiplied in and its record
     joins the observable), closed by a final all-MX data readout + final detectors
     for the X-type (color-0) plaquettes. GATES: stim noiseless determinism of every
     detector AND the observable.
  2. SYMMETRY WITH OBSERVABLE: the color-preserving elements (translations, C3 about
     faces — exact per the symmetry run) re-verified on the full DEM including the
     observable bit; twist mask chi solved per element by GF(2) over per-group
     multiset equations (ports the verified machinery of equiv_decoder_d5b /
     schedule_codesign).
  3. DECODER ARMS (matched params): MLP | periodic 3D-CNN (translation weight-tying
     is EXACT here — torus) | CNN + C3 orbit average with chi-twists. Sample-
     efficiency sweep vs MWPM (decomposed DEM).

PREDICTIONS LOGGED BEFORE ANY RUN (hard rule):
  P1 the Pauli-frame logical returns to X-type after S = 0 mod 3 sub-rounds.
  P2 translations T(1,1),T(2,2) and C3_face verify WITH observable on the full DEM,
     each with some chi (likely nonzero: the maps move the logical string).
  P3 decoder: cnn_c3 >= cnn >= mlp at every n, gap largest at small n (the
     equivariance law on its exactness home turf). Absolute level vs MWPM: unknown.
"""
import argparse
import json
import sys
import time
from collections import defaultdict

import numpy as np
import stim

sys.path.insert(0, "scripts")
sys.path.insert(0, "/root")
from honeycomb_code_symmetry import (PAULI, PTARG, PGATE, build_lattice,
                                     candidate_vertex_maps, derive_action,
                                     window_phase)

PXZ = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


# ------------------------------------------------- symplectic GF(2) helpers
def _commutes(v1, v2, nv):
    return int(v1[:nv] @ v2[nv:] + v1[nv:] @ v2[:nv]) % 2 == 0


def _gf2_rank(rows):
    A = np.array(rows, dtype=np.int64) % 2
    r = 0
    for col in range(A.shape[1]):
        piv = np.nonzero(A[r:, col])[0]
        if len(piv) == 0:
            continue
        A[[r, r + piv[0]]] = A[[r + piv[0], r]]
        for rr in range(A.shape[0]):
            if rr != r and A[rr, col]:
                A[rr] ^= A[r]
        r += 1
        if r == A.shape[0]:
            break
    return r


def _gf2_nullspace(A):
    """Basis of {v : A v = 0} over GF(2). A: (m, n)."""
    A = np.array(A, dtype=np.int64) % 2
    m, n = A.shape
    piv_of_col = {}
    r = 0
    for col in range(n):
        piv = np.nonzero(A[r:, col])[0]
        if len(piv) == 0:
            continue
        A[[r, r + piv[0]]] = A[[r + piv[0], r]]
        for rr in range(m):
            if rr != r and A[rr, col]:
                A[rr] ^= A[r]
        piv_of_col[col] = r
        r += 1
        if r == m:
            break
    basis = []
    free = [c for c in range(n) if c not in piv_of_col]
    for fc in free:
        v = np.zeros(n, dtype=np.int64)
        v[fc] = 1
        for col, row in piv_of_col.items():
            if A[row, fc]:
                v[col] = 1
        basis.append(v)
    return basis


def _check_vec(eid, col, edges, vid, nv):
    px, pz = PXZ[PAULI[col]]
    va, vb, _ = edges[eid]
    v = np.zeros(2 * nv, dtype=np.int64)
    for vt in (va, vb):
        v[vid[vt]] ^= px
        v[nv + vid[vt]] ^= pz
    return v


def _anti(v1, v2, nv):
    return int(v1[:nv] @ v2[nv:] + v1[nv:] @ v2[:nv]) % 2


def frame_evolve(xL0, S, by_color, edges, vid):
    """Track the logical through the schedule. INVARIANT: the frame must COMMUTE
    with every check at the moment it is measured (an anticommuting measurement
    would randomize the observable). The invariant is maintained by multiplying
    in ALREADY-MEASURED checks (recorded outcomes) from the previous two
    sub-rounds, found by a GF(2) solve each sub-round. Returns (final frame as
    (x,z), absorbed (eid, s_past) set) or None if the candidate can't be tracked."""
    nv = len(vid)
    Lv = np.concatenate([xL0, np.zeros(nv, dtype=np.int64)])
    absorbed = set()
    for s in range(S):
        col = s % 3
        up = [(eid, _check_vec(eid, col, edges, vid, nv)) for eid in by_color[col]]
        t = np.array([_anti(Lv, cv, nv) for _, cv in up], dtype=np.int64)
        if t.any():
            if s == 0:
                return None
            pool = []
            for sp in (s - 1, s - 2):
                if sp < 0:
                    continue
                cp = sp % 3
                pool += [((eid, sp), _check_vec(eid, cp, edges, vid, nv))
                         for eid in by_color[cp]]
            A = np.array([[_anti(pv, cv, nv) for _, pv in pool]
                          for _, cv in up], dtype=np.int64)
            x = solve_gf2(A, t)
            if x is None:
                return None
            for k in np.nonzero(x)[0]:
                key, pv = pool[k]
                Lv ^= pv
                absorbed ^= {key}
        # safety: now strictly commuting with every upcoming check
        if any(_anti(Lv, cv, nv) for _, cv in up):
            return None
    return Lv, absorbed


def find_logical(L, M, vid, edges, faces, R):
    """Mechanical logical search: X-type operators commuting with all plaquettes
    (GF(2) nullspace of the Y/Z-plaquette indicator matrix), frame-evolved through
    the schedule; keep the first that (a) ends X-type at some S in [3R, 3R+5],
    (b) ends OUTSIDE the end-instant stabilizer span (genuinely logical)."""
    nv = len(vid)
    by_color = defaultdict(list)
    for eid in sorted(edges):
        by_color[edges[eid][2]].append(eid)

    rows = [np.array([1 if vt in set(f["verts"]) else 0 for vt in
                      sorted(vid, key=lambda v: vid[v])], dtype=np.int64)
            for f in faces.values() if f["color"] != 0]
    basis = _gf2_nullspace(np.array(rows))

    cands = sorted(basis, key=lambda v: v.sum())
    cands += sorted((b1 ^ b2 for i, b1 in enumerate(basis)
                     for b2 in basis[i + 1:]), key=lambda v: v.sum())[:60]

    for S in range(3 * R, 3 * R + 6):
        last_col = (S - 1) % 3
        gens = []
        for f in faces.values():
            v = np.zeros(2 * nv, dtype=np.int64)
            px, pz = PXZ[PAULI[f["color"]]]
            for vt in f["verts"]:
                v[vid[vt]] ^= px
                v[nv + vid[vt]] ^= pz
            gens.append(v)
        pxl, pzl = PXZ[PAULI[last_col]]
        for eid in by_color[last_col]:
            va, vb, _ = edges[eid]
            v = np.zeros(2 * nv, dtype=np.int64)
            v[vid[va]] ^= pxl
            v[nv + vid[va]] ^= pzl
            v[vid[vb]] ^= pxl
            v[nv + vid[vb]] ^= pzl
            gens.append(v)
        base_rank = _gf2_rank(gens)
        for v_x in cands:
            if not v_x.any():
                continue
            out = frame_evolve(v_x, S, by_color, edges, vid)
            if out is None:
                continue
            Lv, absorbed = out
            if Lv[nv:].any():
                continue  # final form not X-type -> MX readout can't close it
            if not all(_commutes(Lv, g, nv) for g in gens):
                continue
            if _gf2_rank(gens + [Lv]) == base_rank:
                continue  # stabilizer product -> trivial observable
            return v_x.copy(), S, sorted(absorbed), Lv[:nv].copy()
    raise AssertionError("no X-type logical found (basis exhausted)")


# ---------------------------------------------------------- memory circuit
def build_memory_circuit(L, M, R, p, mode="cx"):
    verts, vid, edges, elook, faces = build_lattice(L, M)
    nv = len(vid)
    data = list(range(nv))
    by_color = defaultdict(list)
    for eid in sorted(edges):
        by_color[edges[eid][2]].append(eid)
    npc = len(by_color[0])
    anc0 = nv
    xL0, S, absorbed, xS = find_logical(L, M, vid, edges, faces, R)
    absorbed_set = set(absorbed)

    c = stim.Circuit()
    for v, k in sorted(vid.items(), key=lambda kv: kv[1]):
        c.append("QUBIT_COORDS", k, ("AB".index(v[0]), v[1], v[2]))
    c.append("RX", data)

    rec = {}
    nrec = 0
    obs_recs = []
    for s in range(S):
        col = s % 3
        P = PAULI[col]
        if p:
            c.append("DEPOLARIZE1", data, p)
        for k_e, eid in enumerate(by_color[col]):
            va, vb, _ = edges[eid]
            if mode == "mpp":
                t = [PTARG[P](vid[va]), stim.target_combiner(), PTARG[P](vid[vb])]
                if p:
                    c.append("MPP", t, p)
                else:
                    c.append("MPP", t)
            else:
                a = anc0 + k_e
                first = va if va[0] == "A" else vb
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
            if (eid, s) in absorbed_set:
                obs_recs.append(nrec)
            nrec += 1
        # detectors completing at this s (interior, k>=1)
        # (appended after the sub-round's measurements; rel index uses nrec)
        for fk in sorted(faces):
            f = faces[fk]
            cf = f["color"]
            ph = window_phase(cf)
            if (s - ph - 1) % 3 != 0:
                continue
            kk = (s - ph - 1) // 3
            if kk < 1:
                # init closure: |+>^n is a +1 eigenstate of the X-plaquettes, so
                # the FIRST window of every color-0 face is deterministic alone —
                # the t=0 mirror of the final-MX closure. (Other colors: random.)
                if kk == 0 and cf == 0:
                    recs = []
                    s0, s1 = ph, ph + 1
                    for es in f["edge_sets"]:
                        eid = elook[es]
                        ecol = edges[eid][2]
                        ss = s0 if (s0 % 3) == ecol else s1
                        recs.append(rec[(eid, ss)])
                    c.append("DETECTOR",
                             [stim.target_rec(r - nrec) for r in recs],
                             (fk[0], fk[1], cf, s))
                continue
            recs = []
            for k2 in (kk - 1, kk):
                s0, s1 = 3 * k2 + ph, 3 * k2 + ph + 1
                for es in f["edge_sets"]:
                    eid = elook[es]
                    ecol = edges[eid][2]
                    ss = s0 if (s0 % 3) == ecol else s1
                    recs.append(rec[(eid, ss)])
            c.append("DETECTOR", [stim.target_rec(r - nrec) for r in recs],
                     (fk[0], fk[1], cf, s))

    det_meta = []
    # final readout: MX on all data (find_logical guarantees X-type at S)
    if p:
        c.append("Z_ERROR", data, p)
    c.append("MX", data)
    drec = {q: nrec + k for k, q in enumerate(data)}
    nrec += nv
    # final detectors: color-0 faces (X-plaquettes) close against final data
    for fk in sorted(faces):
        f = faces[fk]
        if f["color"] != 0:
            continue
        ph = window_phase(0)
        kk = max(k2 for k2 in range((S + 3) // 3) if 3 * k2 + ph + 1 <= S - 1)
        recs = []
        s0, s1 = 3 * kk + ph, 3 * kk + ph + 1
        for es in f["edge_sets"]:
            eid = elook[es]
            ecol = edges[eid][2]
            ss = s0 if (s0 % 3) == ecol else s1
            recs.append(rec[(eid, ss)])
        for v in f["verts"]:
            recs.append(drec[vid[v]])
        c.append("DETECTOR", [stim.target_rec(r - nrec) for r in recs],
                 (fk[0], fk[1], 0, S))

    # observable = absorbed check records + final MX on the logical's support
    obs_targets = [stim.target_rec(r - nrec) for r in obs_recs]
    obs_targets += [stim.target_rec(drec[q] - nrec) for q in np.nonzero(xS)[0]]
    c.append("OBSERVABLE_INCLUDE", obs_targets, 0)

    # rebuild det_meta in stim's detector order from coords (robust)
    coords = c.get_detector_coordinates()
    det_meta = [tuple(int(x) for x in coords[k]) for k in range(c.num_detectors)]
    det_meta = [((a, b), cc, ss) for (a, b, cc, ss) in det_meta]
    return c, det_meta, (verts, vid, edges, elook, faces)


# ---------------------------------------------------------- DEM + chi solver
def dem_groups_obs(circ):
    dem = circ.detector_error_model(flatten_loops=True)
    groups = defaultdict(list)
    off = 0
    for inst in dem:
        if inst.type == "shift_detectors":
            t0 = inst.targets_copy()[0]
            off += t0.val if hasattr(t0, "val") else int(t0)
        elif inst.type == "error":
            pr = inst.args_copy()[0]
            dets, obs = [], 0
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val + off)
                elif t.is_logical_observable_id():
                    obs ^= 1
            groups[frozenset(dets)].append((obs, round(pr, 9)))
    return dict(groups)


def solve_gf2(A, b):
    A = A.copy() % 2
    b = b.copy() % 2
    m, n = A.shape
    piv, r = [], 0
    for col in range(n):
        rows = np.nonzero(A[r:, col])[0]
        if len(rows) == 0:
            continue
        A[[r, r + rows[0]]] = A[[r + rows[0], r]]
        b[[r, r + rows[0]]] = b[[r + rows[0], r]]
        for rr in range(m):
            if rr != r and A[rr, col]:
                A[rr] ^= A[r]
                b[rr] ^= b[r]
        piv.append(col)
        r += 1
        if r == m:
            break
    if np.any(b[r:]):
        return None
    chi = np.zeros(n, dtype=np.int64)
    for k, col in enumerate(piv):
        chi[col] = b[k]
    return chi


def verify_with_obs(groups, perm, nd, strict=True):
    """Twisted verification with affine freedom: obs' = obs ^ chi.dets ^ c.
    strict=True: every group must conform (exact claim). strict=False: groups
    whose (obs,prob) multisets can't be related by any flip (the time-boundary
    degenerate sector — logical flips that are not detector-determined) are
    EXCLUDED from the equations and reported. Returns
    (chi, c, n_excluded, prob_mass_excluded) or None."""
    ms = lambda g: tuple(sorted(g))
    fl = lambda g: tuple(sorted((o ^ 1, pp) for (o, pp) in g))
    rows, tbits = [], []
    bad, badmass = 0, 0.0
    good = []
    for v, group in groups.items():
        v2 = frozenset(int(perm[d]) for d in v)
        tgt = groups.get(v2)
        if tgt is None or (ms(group) != ms(tgt) and fl(group) != ms(tgt)):
            if strict:
                return None
            bad += 1
            badmass += sum(pp for _, pp in group)
            continue
        good.append(v)
        c0 = ms(group) == ms(tgt)
        c1 = fl(group) == ms(tgt)
        if c0 and c1:
            continue
        row = np.zeros(nd + 1, dtype=np.int64)
        for d in v2:
            row[d] = 1
        row[nd] = 1
        rows.append(row)
        tbits.append(0 if c0 else 1)
    if rows:
        sol = solve_gf2(np.array(rows), np.array(tbits))
        if sol is None:
            return None
        chi, cb = sol[:nd], int(sol[nd])
    else:
        chi, cb = np.zeros(nd, dtype=np.int64), 0
    ok = all(
        ms(groups[frozenset(int(perm[d]) for d in v)]) ==
        (fl(g) if (int(sum(chi[int(perm[d])] for d in v)) + cb) % 2 else ms(g))
        for v in good for g in [groups[v]])
    return (chi, cb, bad, badmass) if ok else None


def symmetry_elements(L, M, det_meta, lat):
    """Detector perms for the color-preserving group elements (full-DEM class).
    Includes axis translations T(3,0)/T(0,3) when the torus admits them — these
    can preserve the logical representative itself (exact incl. boundary)."""
    verts, vid, edges, elook, faces = lat
    vmaps = dict(candidate_vertex_maps(L, M))

    def T(di, dj):
        def f(v):
            sl, i, j = v
            return (sl, (i + di) % L, (j + dj) % M)
        return f

    if L % 3 == 0 and L > 3:
        vmaps["T30"] = T(3, 0)
    if M % 3 == 0 and M > 3:
        vmaps["T03"] = T(0, 3)
    keep = [n for n in ("T11", "T22", "T30", "T03", "C3_face") if n in vmaps]
    index = {meta: k for k, meta in enumerate(det_meta)}
    out = {}
    for name in keep:
        act = derive_action(vmaps[name], edges, elook, faces)
        assert act is not None
        _, _, fperm = act
        perm = np.zeros(len(det_meta), dtype=np.int64)
        ok = True
        for k, (fk, cf, s) in enumerate(det_meta):
            key2 = (fperm[fk], cf, s)
            if key2 not in index:
                ok = False
                break
            perm[k] = index[key2]
        if ok and len(set(perm.tolist())) == len(det_meta):
            out[name] = perm
    return out


# ---------------------------------------------------------- models / training
def make_models():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MLP(nn.Module):
        def __init__(self, nd, h=128):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(nd, h), nn.ReLU(),
                                     nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

        def forward(self, x):
            return self.net(x)[:, 0]

    class PeriodicCNN(nn.Module):
        """3D conv over (t, i, j); circular padding in (i, j) — EXACT on the torus."""
        def __init__(self, flat_idx, shape, h=32):
            super().__init__()
            self.register_buffer("flat_idx",
                                 __import__("torch").tensor(flat_idx).long())
            self.shape = shape
            self.c1 = nn.Conv3d(1, h, 3)
            self.c2 = nn.Conv3d(h, h, 3)
            self.c3 = nn.Conv3d(h, h, 3)
            self.head = nn.Linear(h, 1)

        def _pad(self, g):
            g = F.pad(g, (1, 1, 1, 1, 0, 0), mode="circular")  # j, i circular
            return F.pad(g, (0, 0, 0, 0, 1, 1))                # t zero

        def forward(self, x):
            import torch
            B = x.shape[0]
            nt, ni, nj = self.shape
            g = torch.zeros(B, nt * ni * nj, device=x.device, dtype=x.dtype)
            g.scatter_(1, self.flat_idx.expand(B, -1), x)
            g = g.view(B, 1, nt, ni, nj)
            for conv in (self.c1, self.c2, self.c3):
                g = F.relu(conv(self._pad(g)))
            return self.head(g.mean(dim=(2, 3, 4)))[:, 0]

    class GroupInvWrap(nn.Module):
        def __init__(self, backbone, elements):
            super().__init__()
            import torch
            self.backbone = backbone
            self.register_buffer("Pm", torch.stack(
                [torch.tensor(p).long() for p, _, _ in elements]))
            self.register_buffer("Cm", torch.stack(
                [torch.tensor(ch).float() for _, ch, _ in elements]))
            self.register_buffer("cb", torch.tensor(
                [float(c) for _, _, c in elements]))

        def forward(self, x):
            import torch
            outs = []
            for k in range(self.Pm.shape[0]):
                xk = x[:, self.Pm[k]]
                s = 1.0 - 2.0 * torch.remainder(xk @ self.Cm[k] + self.cb[k], 2.0)
                outs.append(s * self.backbone(xk))
            return torch.stack(outs).mean(dim=0)

    return MLP, PeriodicCNN, GroupInvWrap


def train_eval(model, Xtr, ytr, Xte, yte, dev, steps=4000, bs=256, lr=1e-3):
    import torch
    import torch.nn as nn
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    n = len(ytr)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),), device=dev)
        opt.zero_grad()
        loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
        loss.backward()
        opt.step()
    model.eval()
    accs = []
    with torch.no_grad():
        for i in range(0, len(yte), 20000):
            xb = torch.tensor(Xte[i:i + 20000], device=dev)
            accs.append((model(xb) > 0).float().cpu().numpy() == yte[i:i + 20000])
    return float(np.mean(np.concatenate(accs)))


def run_experiment(L=6, M=6, R=6, p=0.008,
                   ntrains=(250, 1000, 4000, 16000, 64000),
                   seeds=(0, 1, 2, 3, 4), steps=4000, ntest=200_000, log=print):
    import torch
    import pymatching
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T0 = time.time()
    OUT = {"device": (torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"),
           "params_cfg": dict(L=L, M=M, R=R, p=p)}

    circ, det_meta, lat = build_memory_circuit(L, M, R, p, mode="cx")
    circ0, _, _ = build_memory_circuit(L, M, R, 0.0, mode="cx")
    d0, o0 = circ0.compile_detector_sampler().sample(256, separate_observables=True)
    assert not d0.any(), "noiseless detectors fired"
    assert (o0 == o0[0]).all(), "observable nondeterministic"
    log(f"GATES PASSED: {circ.num_detectors} detectors deterministic, "
        f"observable deterministic (noiseless value {int(o0[0,0])})")

    groups = dem_groups_obs(circ)
    nd = circ.num_detectors
    perms = symmetry_elements(L, M, det_meta, lat)
    elements = [(np.arange(nd), np.zeros(nd, dtype=np.int64), 0)]
    OUT["symmetry"] = {}
    for name, perm in list(perms.items()) + [
            ("C3_face^2", perms["C3_face"][perms["C3_face"]])
            if "C3_face" in perms else ("skip", None)]:
        if perm is None:
            continue
        res = verify_with_obs(groups, perm, nd, strict=True)
        scope = "EXACT-full"
        if res is None:
            res = verify_with_obs(groups, perm, nd, strict=False)
            scope = "exact-bulk"
        if res is None:
            OUT["symmetry"][name] = None
            log(f"  {name}: FAILED outright ({len(groups)} groups)")
            continue
        chi, cb, nbad, badmass = res
        OUT["symmetry"][name] = dict(scope=scope, chi=int(chi.sum()), c=cb,
                                     excluded=nbad, excluded_mass=badmass)
        log(f"  {name}: {scope} |chi|={int(chi.sum())} c={cb}"
            + (f" (boundary-degenerate excluded: {nbad} groups, mass {badmass:.4f})"
               if nbad else "") + f" [{len(groups)} groups]")
        elements.append((perm, chi, cb))
    OUT["n_group_elements"] = len(elements)

    # detector grid for the CNN: t = rank of s, spatial (i, j)
    ts = sorted({s for _, _, s in det_meta})
    ti = {v: k for k, v in enumerate(ts)}
    flat_idx = np.zeros(nd, dtype=np.int64)
    for k, (fk, cf, s) in enumerate(det_meta):
        flat_idx[k] = (ti[s] * L + fk[0]) * M + fk[1]
    shape = (len(ts), L, M)

    MLP, PeriodicCNN, GroupInvWrap = make_models()
    npar_cnn = sum(q.numel() for q in PeriodicCNN(flat_idx, shape).parameters())
    npar_mlp = sum(q.numel() for q in MLP(nd).parameters())
    log(f"grid {shape}, ndet={nd}, params cnn={npar_cnn} mlp={npar_mlp}")

    sampler = circ.compile_detector_sampler(seed=999)
    Xte, yte = sampler.sample(ntest, separate_observables=True)
    Xte = Xte.astype(np.float32)
    yte = yte[:, 0].astype(np.float32)
    base = max(yte.mean(), 1 - yte.mean())
    try:
        m = pymatching.Matching.from_detector_error_model(
            circ.detector_error_model(decompose_errors=True))
        mwpm = float(np.mean(m.decode_batch(Xte.astype(bool))[:, 0] == yte))
    except Exception as e:  # honest: honeycomb DEMs are not always decomposable
        mwpm = None
        log(f"  MWPM unavailable: {type(e).__name__}: {e}")
    log(f"majority={base:.4f} MWPM={mwpm}")
    OUT["majority"] = float(base)
    OUT["mwpm"] = mwpm
    OUT["sweep"] = {}

    for n in ntrains:
        accs = defaultdict(list)
        for seed in seeds:
            s2 = circ.compile_detector_sampler(seed=7 * seed + 1)
            Xtr, ytr = s2.sample(n, separate_observables=True)
            Xtr = Xtr.astype(np.float32)
            ytr = ytr[:, 0].astype(np.float32)
            torch.manual_seed(seed)
            accs["mlp"].append(train_eval(MLP(nd), Xtr, ytr, Xte, yte, dev, steps))
            torch.manual_seed(seed)
            accs["cnn"].append(train_eval(PeriodicCNN(flat_idx, shape),
                                          Xtr, ytr, Xte, yte, dev, steps))
            torch.manual_seed(seed)
            accs["cnn_c3"].append(train_eval(
                GroupInvWrap(PeriodicCNN(flat_idx, shape), elements),
                Xtr, ytr, Xte, yte, dev, steps))
        row = {k: (float(np.mean(v)), float(np.std(v))) for k, v in accs.items()}
        OUT["sweep"][n] = row
        log(f"  n={n:6d}: " + "  ".join(
            f"{k} {row[k][0]:.4f}±{row[k][1]:.4f}" for k in ("mlp", "cnn", "cnn_c3")))
    OUT["runtime_s"] = time.time() - T0
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        out = run_experiment(L=3, M=3, R=4, ntrains=(250,), seeds=(0,),
                             steps=120, ntest=20000)
        print(json.dumps({k: v for k, v in out.items() if k != "sweep"}, indent=1,
                         default=str))
        print(json.dumps(out["sweep"], indent=1))
    else:
        out = run_experiment()
        with open("runs/honeycomb_decoder.json", "w") as f:
            json.dump(out, f, indent=1, default=str)
        print(f"saved -> runs/honeycomb_decoder.json ({out['runtime_s']:.0f}s)")
