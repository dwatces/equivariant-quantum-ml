"""
Job 6 (local, CPU): MINE circuit-level symmetries — the schedule-aware frontier.

At circuit-level noise the naive spatial mirrors fail verification (extraction-
schedule hook errors de-symmetrize the DEM). The findings doc conjectures that
SCHEDULE-AWARE group actions exist: spatial maps composed with temporal offsets /
reversal / coordinate shifts. This searches that family exhaustively with the exact
group-based verifier. A verified element => first circuit-level equivariant decoder
arm. A clean null on the whole family is an honest, publishable bound either way.

Family searched per spatial base s in {mirror_x, mirror_y, C2, identity}:
  f(x,y,t) = s(x,y) + (dx, dy), with t -> t+dt or t -> T-t+dt, dt,dx,dy in [-2..2]
(identity-spatial excluded for the identity-time case).

Usage: .venv-gpu/bin/python scripts/circuit_symmetry_mine.py
"""
import itertools
import json
import sys

sys.path.insert(0, "scripts")

import numpy as np
import stim

from equiv_cnn_core import P, dem_errors, solve_gf2

from collections import defaultdict


def verify_map(c, fmap):
    """Group-based verification of an arbitrary coordinate map. Returns chi or None."""
    coords = c.get_detector_coordinates()
    nd = c.num_detectors
    lookup = {tuple(np.round(coords[i], 4)): i for i in range(nd)}
    perm = np.full(nd, -1, dtype=int)
    for i in range(nd):
        j = lookup.get(tuple(np.round(fmap(*coords[i]), 4)))
        if j is None:
            return None, None
        perm[i] = j
    if len(set(perm.tolist())) != nd:
        return None, None
    errs = dem_errors(c)
    by_v = defaultdict(list)
    for v, o, p in errs:
        by_v[v].append((o, p))

    def ms(g):
        return tuple(sorted(g))

    def fl(g):
        return tuple(sorted((o ^ 1, p) for (o, p) in g))

    rows, tbits = [], []
    for v, group in by_v.items():
        v2 = frozenset(int(perm[d]) for d in v)
        tgt = by_v.get(v2)
        if tgt is None:
            return None, None
        c0 = ms(group) == ms(tgt)
        c1 = fl(group) == ms(tgt)
        if not c0 and not c1:
            return None, None
        if c0 and c1:
            continue
        row = np.zeros(nd, dtype=np.int64)
        for d in v2:
            row[d] = 1
        rows.append(row)
        tbits.append(0 if c0 else 1)
    chi = solve_gf2(np.array(rows), np.array(tbits)) if rows else np.zeros(nd, dtype=np.int64)
    if chi is None:
        return None, None
    ok = all(
        ms(by_v[frozenset(int(perm[d]) for d in v)]) ==
        (fl(g) if int(sum(chi[int(perm[d])] for d in v)) % 2 else ms(g))
        for v, g in by_v.items())
    return (perm, chi) if ok else (None, None)


def mine(d):
    c = stim.Circuit.generated(
        "surface_code:unrotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=P,
        before_measure_flip_probability=P,
        after_reset_flip_probability=P,
        before_round_data_depolarization=P)
    coords = c.get_detector_coordinates()
    X = max(v[0] for v in coords.values())
    Y = max(v[1] for v in coords.values())
    T = max(v[2] for v in coords.values())
    spatial = {
        "id": lambda x, y: (x, y),
        "mirror_x": lambda x, y: (X - x, y),
        "mirror_y": lambda x, y: (x, Y - y),
        "C2": lambda x, y: (X - x, Y - y),
    }
    found = []
    tried = 0
    rng = range(-2, 3)
    for sname, s in spatial.items():
        for trev in (False, True):
            for dt, dx, dy in itertools.product(rng, rng, rng):
                if sname == "id" and not trev and dt == 0 and dx == 0 and dy == 0:
                    continue  # identity

                def fmap(x, y, t, s=s, trev=trev, dt=dt, dx=dx, dy=dy):
                    xs, ys = s(x, y)
                    tt = (T - t if trev else t) + dt
                    return (xs + dx, ys + dy, tt)

                tried += 1
                perm, chi = verify_map(c, fmap)
                if perm is not None:
                    tag = f"{sname}{'·trev' if trev else ''}+dt{dt}+dx{dx}+dy{dy}"
                    print(f"  d={d}: VERIFIED {tag} |chi|={int(chi.sum())}", flush=True)
                    found.append({"map": tag, "chi_weight": int(chi.sum())})
    print(f"  d={d}: tried {tried} candidate maps, verified {len(found)}", flush=True)
    return found


if __name__ == "__main__":
    out = {}
    for d in (3, 5):
        print(f"== mining circuit-level symmetries, d={d} ==", flush=True)
        out[f"d{d}"] = mine(d)
    with open("runs/circuit_symmetry_mine.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved -> runs/circuit_symmetry_mine.json")
