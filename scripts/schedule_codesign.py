"""
SCHEDULE CO-DESIGN search (job B, 2026-06-11): can a CNOT extraction schedule make
the surface code's mirror/C2 symmetry EXACT at circuit-level noise?

Background: stim's generated circuit-level surface code has NO twisted symmetry in
the whole affine-geometric family (999-map null, scripts/circuit_symmetry_mine.py).
The conjecture is that the obstruction is the extraction schedule, not the physics.
This script builds its own unrotated surface-code memory-Z circuit with a
PARAMETERIZED CNOT schedule and searches schedule space for one where the exact
group-based verifier passes.

Schedule families searched (d=3, rounds=3, full circuit-level noise):
  A: uniform global schedule  (perm_X, perm_Z) in S4 x S4            (576)
  B: mirror-paired halves: left half uses (perm_X, perm_Z), right half uses the
     E<->W mirrored perms, center-column plaquettes use the left perm     (576)
  C: like B but the right half also REVERSES the step order (mirror composed
     with time-reversal within the round). Fixes B's structural collision:
     center anc step-k E hits (cx+1,y) exactly when the mirrored neighbor's
     step-k W does; reversing the right half's steps makes that need
     d_k == d_{5-k}, impossible for a permutation.                        (576)

CORRECTNESS GATES per candidate circuit (a schedule only counts if ALL pass):
  G1 valid moment structure: no qubit touched twice in a CNOT timestep
  G2 noiseless circuit -> every detector deterministic (stim sampler all-zero)
  G3 distance preserved: len(shortest_graphlike_error) == d  (hook errors kill this)
  G4 the exact group-based twisted-symmetry verifier (mirror_x / mirror_y / C2)

Honest outcomes: ANY (G1-G4)-passing schedule = constructive schedule-aware
equivariance (new arm to train). A full null = strengthened impossibility evidence.
"""
import itertools
import json
import sys
import time

sys.path.insert(0, "scripts")

import numpy as np
import stim

from equiv_cnn_core import dem_errors, solve_gf2

from collections import defaultdict

D = 3
ROUNDS = 3
P = 0.008
DIRS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
MIRROR_X_DIR = {"N": "N", "S": "S", "E": "W", "W": "E"}


def build_layout(d):
    """Unrotated surface code: data at x+y even; X-ancillas at (even,odd);
    Z-ancillas at (odd,even). Grid 0..2d-2. Memory-Z: Z-logical = left column."""
    span = 2 * d - 1
    data, xanc, zanc = [], [], []
    for x in range(span):
        for y in range(span):
            if (x + y) % 2 == 0:
                data.append((x, y))
            elif x % 2 == 0:
                xanc.append((x, y))
            else:
                zanc.append((x, y))
    return data, xanc, zanc, span


def build_circuit(d, rounds, p, sched_for):
    """sched_for(kind, (x,y)) -> permutation tuple of direction names."""
    data, xanc, zanc, span = build_layout(d)
    qid = {}
    for i, c in enumerate(data + xanc + zanc):
        qid[c] = i
    nbrs = lambda c: [tuple(np.add(c, DIRS[k])) for k in DIRS]

    c = stim.Circuit()
    for coord, i in qid.items():
        c.append("QUBIT_COORDS", i, coord)
    dq = [qid[q] for q in data]
    xq = [qid[q] for q in xanc]
    zq = [qid[q] for q in zanc]
    anc = xq + zq

    def extraction_round(noisy):
        body = stim.Circuit()
        body.append("R", anc)
        if noisy:
            body.append("X_ERROR", anc, p)
        body.append("H", xq)
        if noisy:
            body.append("DEPOLARIZE1", xq, p)
        for step in range(4):
            moment = []
            for a in xanc:
                dirn = sched_for("X", a)[step]
                tgt = tuple(np.add(a, DIRS[dirn]))
                if tgt in qid and (tgt[0] + tgt[1]) % 2 == 0:
                    moment.append((qid[a], qid[tgt]))  # X-anc is control
            for a in zanc:
                dirn = sched_for("Z", a)[step]
                tgt = tuple(np.add(a, DIRS[dirn]))
                if tgt in qid and (tgt[0] + tgt[1]) % 2 == 0:
                    moment.append((qid[tgt], qid[a]))  # data is control for Z
            used = [q for pair in moment for q in pair]
            if len(used) != len(set(used)):
                raise ValueError("schedule collision")  # gate G1
            flat = [q for pair in moment for q in pair]
            if flat:
                body.append("CX", flat)
                if noisy:
                    body.append("DEPOLARIZE2", flat, p)
            body.append("TICK")
        body.append("H", xq)
        if noisy:
            body.append("DEPOLARIZE1", xq, p)
        if noisy:
            body.append("X_ERROR", anc, p)
        body.append("MR", anc)
        return body

    # round 1: only Z-ancilla detectors are deterministic (|0..0> init)
    if True:
        if p:
            c.append("X_ERROR", dq, p)
        c += extraction_round(noisy=bool(p))
        for k, a in enumerate(zanc):
            c.append("DETECTOR", [stim.target_rec(-len(anc) + len(xq) + k)],
                     (*a, 0))
    for r in range(1, rounds):
        if p:
            c.append("DEPOLARIZE1", dq, p)
        c += extraction_round(noisy=bool(p))
        for k, a in enumerate(xanc):
            c.append("DETECTOR",
                     [stim.target_rec(-len(anc) + k),
                      stim.target_rec(-2 * len(anc) + k)], (*a, r))
        for k, a in enumerate(zanc):
            c.append("DETECTOR",
                     [stim.target_rec(-len(anc) + len(xq) + k),
                      stim.target_rec(-2 * len(anc) + len(xq) + k)], (*a, r))
    # final data measurement
    if p:
        c.append("X_ERROR", dq, p)
    c.append("M", dq)
    drec = {q: -len(dq) + i for i, q in enumerate(data)}
    for k, a in enumerate(zanc):
        targs = [stim.target_rec(drec[t]) for t in nbrs(a)
                 if t in drec]
        targs.append(stim.target_rec(-len(dq) - len(anc) + len(xq) + k))
        c.append("DETECTOR", targs, (*a, rounds))
    # logical Z = left data column (x = 0)
    obs = [stim.target_rec(drec[q]) for q in data if q[0] == 0]
    c.append("OBSERVABLE_INCLUDE", obs, 0)
    return c


def gates_pass(c, d):
    # G2: noiseless determinism
    nonoise = c.without_noise()
    s = nonoise.compile_detector_sampler()
    dets, obs = s.sample(64, separate_observables=True)
    if dets.any() or obs.any():
        return "G2-nondeterministic"
    # G3: distance
    try:
        err = c.shortest_graphlike_error()
    except ValueError:
        return "G3-no-graphlike"
    if len(err) != d:
        return f"G3-distance-{len(err)}"
    return None


def verify_map(c, fmap):
    coords = c.get_detector_coordinates()
    nd = c.num_detectors
    lookup = {tuple(np.round(coords[i], 4)): i for i in range(nd)}
    perm = np.full(nd, -1, dtype=int)
    for i in range(nd):
        j = lookup.get(tuple(np.round(fmap(*coords[i]), 4)))
        if j is None:
            return None
        perm[i] = j
    if len(set(perm.tolist())) != nd:
        return None
    errs = dem_errors(c)
    by_v = defaultdict(list)
    for v, o, p in errs:
        by_v[v].append((o, p))
    ms = lambda g: tuple(sorted(g))
    fl = lambda g: tuple(sorted((o ^ 1, pp) for (o, pp) in g))
    rows, tbits = [], []
    for v, group in by_v.items():
        v2 = frozenset(int(perm[dd]) for dd in v)
        tgt = by_v.get(v2)
        if tgt is None:
            return None
        c0 = ms(group) == ms(tgt)
        c1 = fl(group) == ms(tgt)
        if not c0 and not c1:
            return None
        if c0 and c1:
            continue
        row = np.zeros(nd, dtype=np.int64)
        for dd in v2:
            row[dd] = 1
        rows.append(row)
        tbits.append(0 if c0 else 1)
    chi = solve_gf2(np.array(rows), np.array(tbits)) if rows else np.zeros(nd, dtype=np.int64)
    if chi is None:
        return None
    ok = all(
        ms(by_v[frozenset(int(perm[dd]) for dd in v)]) ==
        (fl(g) if int(sum(chi[int(perm[dd])] for dd in v)) % 2 else ms(g))
        for v, g in by_v.items())
    return int(chi.sum()) if ok else None


def symmetry_check(c, span):
    X = Y = span - 1
    cands = {"mirror_x": lambda x, y, t: (X - x, y, t),
             "mirror_y": lambda x, y, t: (x, Y - y, t),
             "C2": lambda x, y, t: (X - x, Y - y, t)}
    return {name: verify_map(c, f) for name, f in cands.items()}


def main():
    T0 = time.time()
    perms = list(itertools.permutations("NESW"))
    data, xanc, zanc, span = build_layout(D)
    cx = (span - 1) / 2
    results = {"families": {}, "verified": []}
    n_valid = n_g2 = n_g3 = 0
    for fam in ("A", "B", "C"):
        fam_stats = {"tried": 0, "collision": 0, "g2": 0, "g3": 0, "passed_gates": 0}
        for pX in perms:
            for pZ in perms:
                fam_stats["tried"] += 1

                if fam == "A":
                    def sched_for(kind, a, pX=pX, pZ=pZ):
                        return pX if kind == "X" else pZ
                elif fam == "B":
                    def sched_for(kind, a, pX=pX, pZ=pZ):
                        base = pX if kind == "X" else pZ
                        if a[0] > cx:
                            return tuple(MIRROR_X_DIR[k] for k in base)
                        return base
                else:
                    def sched_for(kind, a, pX=pX, pZ=pZ):
                        base = pX if kind == "X" else pZ
                        if a[0] > cx:
                            return tuple(MIRROR_X_DIR[k] for k in reversed(base))
                        return base

                try:
                    c = build_circuit(D, ROUNDS, P, sched_for)
                except ValueError:
                    fam_stats["collision"] += 1
                    continue
                g = gates_pass(c, D)
                if g:
                    fam_stats["g2" if g.startswith("G2") else "g3"] += 1
                    continue
                fam_stats["passed_gates"] += 1
                sym = symmetry_check(c, span)
                hits = {k: v for k, v in sym.items() if v is not None}
                if hits:
                    rec = {"family": fam, "perm_X": "".join(pX),
                           "perm_Z": "".join(pZ), "verified": hits}
                    results["verified"].append(rec)
                    print(f"  ★ FOUND fam={fam} X={''.join(pX)} Z={''.join(pZ)} "
                          f"-> {hits}", flush=True)
        results["families"][fam] = fam_stats
        print(f"family {fam}: {fam_stats}", flush=True)
    results["runtime_s"] = time.time() - T0
    with open("runs/schedule_codesign.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"DONE in {results['runtime_s']:.0f}s; verified schedules: "
          f"{len(results['verified'])} -> runs/schedule_codesign.json", flush=True)


if __name__ == "__main__":
    main()
