"""The commensurate sweep — the open stone of verdict 6, with the
translation lens (Daniel, 2026-07-02: "physics-origin is not a barrier to
translate to compute; it just hasn't happened yet").

Two questions, one powered run:
  Q1 MAGNITUDE — at spot spacings of ~2–6 lattice constants (n=256 domains
     give the spot counts that R=3 lacked at 128²), do paired ψ4/ψ6 packing
     orders finally differ?
  Q2 ORIENTATION LOCKING (new observable — the one a compute-translation
     needs first): does the emergent hexatic/tetratic axis ALIGN with the
     substrate's crystal axes? Locking index = mean over patterns of
     cos(m·Φ_m), where Φ_m = (1/m)·arg(Σ_bonds e^{imθ}) is the pattern's
     bond-orientation phase and the lattice's own m-fold axes sit at phase 0
     (b1 along +x on both substrates). +1 = locked on-axis, 0 = no locking,
     −1 = anti-locked (locked BETWEEN axes — also substrate memory!).
     Significance: circular — under H0 (no locking) cos(mΦ) has mean 0;
     report mean ± 1.96·sd/√n per (substrate, R).
     Magnitude nulls can hide locking; phase cannot.

Resume-safe (JSONL keyed by substrate/R/seed), runs anywhere:
  .venv-gpu/bin/python3 hexlife/sweep_commensurate.py [--n 256] [--out runs/hexlife_commensurate.jsonl]
Analysis:
  .venv-gpu/bin/python3 hexlife/sweep_commensurate.py --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lenia import Lenia, classify, device, soup  # noqa: E402

ROOT = Path(__file__).parent.parent
R_LIST = (2.0, 2.5, 3.0, 3.5, 4.0)
SEEDS = (11, 12, 13, 14, 15)
# refocused on the alive strip found by E2/E5 (low mu); paired grid 16x16
MUS = torch.linspace(0.08, 0.20, 16)
SIGMAS = torch.logspace(math.log10(0.008), math.log10(0.05), 16)
STEPS = 400


def spot_stats(A_row: torch.Tensor, sim, thresh_frac: float = 0.4,
               min_spots: int = 40):
    """Spot centroids → paired packing magnitudes AND phases + spacing.

    Returns dict or None. Lattice-correct connectivity (verdict-6 artifact
    #4); Delaunay on 3x3 periodic replication, central bonds only, spurious
    long bonds pruned at 2x median.
    """
    from scipy import ndimage
    from scipy.spatial import Delaunay
    arr = A_row.cpu().numpy()
    mx = float(arr.max())
    if mx < 0.05:
        return None
    if sim.substrate == "hex":
        structure = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    lab, nl = ndimage.label(arr > thresh_frac * mx, structure=structure)
    if nl < min_spots:
        return None
    n = sim.n
    B = sim.basis.numpy()
    coms = ndimage.center_of_mass(arr, lab, range(1, nl + 1))
    pts = np.array([[i * B[0, 0] + j * B[1, 0],
                     i * B[0, 1] + j * B[1, 1]] for j, i in coms])
    T1, T2 = n * B[0], n * B[1]
    reps, owner = [], []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            reps.append(pts + a * T1 + b * T2)
            owner += [a == 0 and b == 0] * len(pts)
    P = np.concatenate(reps)
    owner = np.array(owner)
    tri = Delaunay(P)
    from collections import defaultdict
    nbrs = defaultdict(set)
    for s in tri.simplices:
        for u in s:
            for v in s:
                if u != v:
                    nbrs[u].add(v)
    z4 = z6 = 0.0 + 0.0j
    nn_dists = []
    nb = 0
    for u in np.where(owner)[0]:
        vs = list(nbrs[u])
        if len(vs) < 3:
            continue
        d = P[vs] - P[u]
        L = np.linalg.norm(d, axis=1)
        keep = L < 2.0 * np.median(L)
        d, L = d[keep], L[keep]
        th = np.arctan2(d[:, 1], d[:, 0])
        z4 += np.exp(1j * 4 * th).sum()
        z6 += np.exp(1j * 6 * th).sum()
        nb += len(th)
        nn_dists.append(L.min())
    if nb < 3 * min_spots:
        return None
    return {"n_spots": int(nl),
            "psi4": float(abs(z4) / nb), "psi6": float(abs(z6) / nb),
            "phi4": float(np.angle(z4) / 4), "phi6": float(np.angle(z6) / 6),
            "spacing": float(np.median(nn_dists))}


def run_sweep(n: int, out_path: Path, seeds=SEEDS):
    dev = device()
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["substrate"], r["R"], r["seed"]))
    mus = MUS.repeat_interleave(len(SIGMAS))
    sigs = SIGMAS.repeat(len(MUS))
    total = len(R_LIST) * len(seeds) * 2
    k = 0
    for Rv in R_LIST:
        for seed in seeds:
            for name in ("square", "hex"):
                k += 1
                if (name, Rv, seed) in done:
                    continue
                sim = Lenia(name, n=n, R=Rv, dev=dev)
                A = soup(len(mus), n, seed=seed, dev=dev)
                A, mh, ch = sim.run(A, mus, sigs, STEPS, record_every=25)
                drift = sim.drift(ch)
                outcomes = classify(mh, drift)
                rows = []
                for b, o in enumerate(outcomes):
                    if o not in ("traveler", "alive"):
                        continue
                    st = spot_stats(A[b], sim)
                    if st is None:
                        continue
                    rows.append({"b": b, "mu": float(mus[b]),
                                 "sigma": float(sigs[b]), **st})
                rec = {"substrate": name, "R": Rv, "seed": seed, "n": n,
                       "alive": sum(1 for o in outcomes
                                    if o in ("traveler", "alive")),
                       "n_params": len(outcomes), "patterns": rows}
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{k}/{total}] {name} R={Rv} seed={seed}: "
                      f"alive {rec['alive']}/{rec['n_params']}, "
                      f"measured {len(rows)}", flush=True)
                del A, mh, ch
                if dev.type == "cuda":
                    torch.cuda.empty_cache()


def analyze(out_path: Path):
    recs = [json.loads(l) for l in out_path.read_text().splitlines()]
    lat_a = {"square": 1.0, "hex": math.sqrt(2 / math.sqrt(3))}
    print(f"{'sub':7s} {'R':4s} {'n_pat':6s} {'ψ4':7s} {'ψ6':7s} "
          f"{'lock4 (±CI)':16s} {'lock6 (±CI)':16s} {'spacing/a':9s}")
    summary = {}
    for name in ("square", "hex"):
        for Rv in R_LIST:
            pats = [p for r in recs
                    if r["substrate"] == name and r["R"] == Rv
                    for p in r["patterns"]]
            if not pats:
                continue
            psi4 = np.mean([p["psi4"] for p in pats])
            psi6 = np.mean([p["psi6"] for p in pats])
            c4 = np.array([math.cos(4 * p["phi4"]) for p in pats])
            c6 = np.array([math.cos(6 * p["phi6"]) for p in pats])
            l4, e4_ = c4.mean(), 1.96 * c4.std() / max(1, len(c4)) ** 0.5
            l6, e6_ = c6.mean(), 1.96 * c6.std() / max(1, len(c6)) ** 0.5
            sp = np.median([p["spacing"] for p in pats]) / lat_a[name]
            print(f"{name:7s} {Rv:<4} {len(pats):<6d} {psi4:.3f}   "
                  f"{psi6:.3f}   {l4:+.3f}±{e4_:.3f}    "
                  f"{l6:+.3f}±{e6_:.3f}    {sp:.2f}")
            summary[f"{name}_R{Rv}"] = {
                "n": len(pats), "psi4": float(psi4), "psi6": float(psi6),
                "lock4": [float(l4), float(e4_)],
                "lock6": [float(l6), float(e6_)],
                "spacing_over_a": float(sp)}
    # paired magnitude comparison per R (params alive+measured in both)
    print("\npaired ψ sign tests (hex − square), per R:")
    for Rv in R_LIST:
        by = {}
        for name in ("square", "hex"):
            by[name] = {}
            for r in recs:
                if r["substrate"] == name and r["R"] == Rv:
                    for p in r["patterns"]:
                        by[name][(r["seed"], p["b"])] = p
        keys = sorted(set(by["square"]) & set(by["hex"]))
        if len(keys) < 8:
            print(f"  R={Rv}: only {len(keys)} pairs — underpowered")
            continue
        d6 = [by["hex"][k]["psi6"] - by["square"][k]["psi6"] for k in keys]
        d4 = [by["hex"][k]["psi4"] - by["square"][k]["psi4"] for k in keys]
        k6 = sum(1 for x in d6 if x > 0)
        k4 = sum(1 for x in d4 if x > 0)
        print(f"  R={Rv}: n={len(keys)}  ψ6 hex> {k6}/{len(keys)} "
              f"(mean Δ {np.mean(d6):+.4f})   ψ4 hex> {k4}/{len(keys)} "
              f"(mean Δ {np.mean(d4):+.4f})")
        summary[f"paired_R{Rv}"] = {"n": len(keys), "psi6_hex_gt": k6,
                                    "psi4_hex_gt": k4,
                                    "d6_mean": float(np.mean(d6)),
                                    "d4_mean": float(np.mean(d4))}
    (ROOT / "runs" / "hexlife_commensurate_summary.json").write_text(
        json.dumps(summary, indent=1))
    print("\nsummary → runs/hexlife_commensurate_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "runs" /
                                         "hexlife_commensurate.jsonl"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(Path(args.out))
    else:
        run_sweep(args.n, Path(args.out), seeds=tuple(args.seeds))
