"""The two open stones of verdict 6 (findings: runs/HEXLIFE_FINDINGS.md
"Open" items 1 + the Q0 reconciliation), one driver, three modes.

MODE crossover — the 3–8a panel: where does locking die?
  The resolved commensurate sweep (R 2–4, spacings 1.4–3.1a) found the
  substrate OWNS the settled pattern (symmetry selection + orientation
  locking to |0.97|); the dilute regime (R=10, ~8a) is null. The regime
  between is unmapped. Same rig, same strip grid, same 400 steps, same
  artifact-hardened spot_stats; R ∈ {4.5, 5, 5.5, 6, 7, 8, 9, 10} on
  n=256 domains, seeds 11–15.
  PRE-REGISTERED (written before any run):
    P1 matched-symmetry |lock| decays from the R=4 values (square −0.90,
       hex −0.56) toward 0 with increasing spacing/a; we locate the
       spacing where the 95% CI first includes 0 and stays there.
    P2 sign flips with commensuration ratio may continue en route
       (Novaco–McTague oscillation), so the decay need not be monotone in
       magnitude — the claim tested is DECAY-TO-NULL BY ~8a, not
       monotonicity.
    P3 paired ψ-selection sign tests weaken in step with locking
       (mean Δψ6 already shrank +0.18 → ~+0.05 by R=4).
    P4 abundance (Q0) stays null at all crossover R (it was already null
       at R≥3.5).
    FALSIFIER: if locking survives at ~8a with CI excluding 0, verdict
    6's dilute-regime statement was wrong in the phase channel (its E6
    basis measured magnitudes only, which cannot see phase).

MODE strip128 + MODE broad256 — the E5 abundance reconciliation A/B.
  Q0 found hex sustains +42/+33/+13% more alive params at R=2/2.5/3
  (10/10 seeds) on the low-μ strip at n=256; E5 found abundance NULL on
  the broad 24×24 grid at n=128 (R≥2.5). Two factors differ: the (μ,σ)
  window and the domain size. Factorial completion (both at 400 steps —
  E5 itself used 500; accepted as a minor third factor, disclosed):
    cell C strip128 = strip grid, n=128, R ∈ {2, 2.5, 3}, seeds 11–15
    cell D broad256 = broad E5 grid (24×24), n=256, same R, same seeds
  PRE-REGISTERED:
    P5 if the Q0 effect is real physics of the tight-commensuration
       strip, C reproduces the hex excess (10/10-class sign counts at
       R=2/2.5) and D shows it INSIDE the strip window (μ≤0.20, σ≤0.05)
       while diluting toward E5's null in the aggregate.
    P6 if instead C is null and D's aggregate is positive, the effect
       was a domain-size artifact and the Q0 claim must be retracted to
       "n=256-specific" — reported either way.

Resume-safe JSONL (keyed substrate/R/seed). Records add `alive_idx` (the
alive param indices) so abundance can be re-windowed post hoc.

Run:     python3 hexlife/sweep_crossover.py --mode crossover [--cpu]
Analyze: python3 hexlife/sweep_crossover.py --mode crossover --analyze
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lenia import Lenia, classify, soup  # noqa: E402
from sweep_commensurate import (MUS as STRIP_MUS, SIGMAS as STRIP_SIGMAS,  # noqa: E402
                                spot_stats)

ROOT = Path(__file__).parent.parent
STEPS = 400
SEEDS = (11, 12, 13, 14, 15)

# E5's broad grid, mirrored from experiments.py (import avoided: matplotlib)
BROAD_MUS = torch.linspace(0.08, 0.42, 24)
BROAD_SIGMAS = torch.logspace(math.log10(0.008), math.log10(0.10), 24)
# the strip window inside the broad grid, for post-hoc restriction
STRIP_MU_MAX, STRIP_SIG_MAX = 0.20, 0.05

MODES = {
    "crossover": {"n": 256, "R": (4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0),
                  "mus": STRIP_MUS, "sigmas": STRIP_SIGMAS,
                  "out": "hexlife_crossover.jsonl"},
    "strip128":  {"n": 128, "R": (2.0, 2.5, 3.0),
                  "mus": STRIP_MUS, "sigmas": STRIP_SIGMAS,
                  "out": "hexlife_e5recon_strip128.jsonl"},
    "broad256":  {"n": 256, "R": (2.0, 2.5, 3.0),
                  "mus": BROAD_MUS, "sigmas": BROAD_SIGMAS,
                  "out": "hexlife_e5recon_broad256.jsonl"},
    # E5's own cell (broad grid @ n=128) re-run under this protocol
    # (400 steps, 5 seeds, alive_idx) — completes the 2x2 factorial
    "broad128":  {"n": 128, "R": (2.0, 2.5, 3.0),
                  "mus": BROAD_MUS, "sigmas": BROAD_SIGMAS,
                  "out": "hexlife_e5recon_broad128.jsonl"},
    # box-size test for the dilute-regime locking (2026-07-03 follow-up):
    # at spacing ~6-9a the periodic DOMAIN (29 rows at n=256) could lock
    # the array orientation instead of the microscopic lattice — and on
    # hex the box axes coincide with the lattice axes. If lock6 at R=7/10
    # is lattice-driven it is box-size invariant (as strip128 proved for
    # R<=3); if box-driven it weakens with more rows (192 -> 256 -> 384).
    "box192":    {"n": 192, "R": (7.0, 10.0),
                  "mus": STRIP_MUS, "sigmas": STRIP_SIGMAS,
                  "out": "hexlife_boxtest192.jsonl"},
    "box384":    {"n": 384, "R": (7.0, 10.0),
                  "mus": STRIP_MUS, "sigmas": STRIP_SIGMAS,
                  "out": "hexlife_boxtest384.jsonl"},
}


def run_mode(mode: str, seeds, dev: torch.device):
    cfg = MODES[mode]
    out_path = ROOT / "runs" / cfg["out"]
    n = cfg["n"]
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["substrate"], r["R"], r["seed"]))
    mus = cfg["mus"].repeat_interleave(len(cfg["sigmas"]))
    sigs = cfg["sigmas"].repeat(len(cfg["mus"]))
    total = len(cfg["R"]) * len(seeds) * 2
    k = 0
    for Rv in cfg["R"]:
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
                alive_idx = [b for b, o in enumerate(outcomes)
                             if o in ("traveler", "alive")]
                rows = []
                for b in alive_idx:
                    st = spot_stats(A[b], sim)
                    if st is None:
                        continue
                    rows.append({"b": b, "mu": float(mus[b]),
                                 "sigma": float(sigs[b]), **st})
                rec = {"substrate": name, "R": Rv, "seed": seed, "n": n,
                       "mode": mode, "alive": len(alive_idx),
                       "alive_idx": alive_idx, "n_params": len(outcomes),
                       "patterns": rows}
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{k}/{total}] {mode} {name} R={Rv} seed={seed}: "
                      f"alive {rec['alive']}/{rec['n_params']}, "
                      f"measured {len(rows)}", flush=True)
                del A, mh, ch
                if dev.type == "cuda":
                    torch.cuda.empty_cache()


def _lock_table(recs, R_vals, tag, summary):
    lat_a = {"square": 1.0, "hex": math.sqrt(2 / math.sqrt(3))}
    print(f"\n{tag}: per-(substrate, R) packing + locking")
    print(f"{'sub':7s} {'R':5s} {'n_pat':6s} {'ψ4':7s} {'ψ6':7s} "
          f"{'lock4 (±CI)':16s} {'lock6 (±CI)':16s} {'spacing/a':9s}")
    for name in ("square", "hex"):
        for Rv in R_vals:
            pats = [p for r in recs
                    if r["substrate"] == name and r["R"] == Rv
                    for p in r["patterns"]]
            if not pats:
                continue
            psi4 = np.mean([p["psi4"] for p in pats])
            psi6 = np.mean([p["psi6"] for p in pats])
            c4 = np.array([math.cos(4 * p["phi4"]) for p in pats])
            c6 = np.array([math.cos(6 * p["phi6"]) for p in pats])
            l4, e4 = c4.mean(), 1.96 * c4.std() / max(1, len(c4)) ** 0.5
            l6, e6 = c6.mean(), 1.96 * c6.std() / max(1, len(c6)) ** 0.5
            sp = np.median([p["spacing"] for p in pats]) / lat_a[name]
            print(f"{name:7s} {Rv:<5} {len(pats):<6d} {psi4:.3f}   "
                  f"{psi6:.3f}   {l4:+.3f}±{e4:.3f}    "
                  f"{l6:+.3f}±{e6:.3f}    {sp:.2f}")
            summary[f"{name}_R{Rv}"] = {
                "n": len(pats), "psi4": float(psi4), "psi6": float(psi6),
                "lock4": [float(l4), float(e4)], "lock6": [float(l6),
                                                           float(e6)],
                "spacing_over_a": float(sp)}


def _paired_psi(recs, R_vals, summary):
    print("\npaired ψ sign tests (hex − square), per R:")
    for Rv in R_vals:
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


def _abundance(recs, R_vals, mode, summary):
    cfg = MODES[mode]
    mus = cfg["mus"].repeat_interleave(len(cfg["sigmas"]))
    sigs = cfg["sigmas"].repeat(len(cfg["mus"]))
    in_strip = [(float(mus[b]) <= STRIP_MU_MAX + 1e-9 and
                 float(sigs[b]) <= STRIP_SIG_MAX + 1e-9)
                for b in range(len(mus))]
    n_strip = sum(in_strip)
    windows = [("aggregate", None)]
    if mode.startswith("broad"):
        windows.append((f"strip-window ({n_strip}/{len(mus)} params)",
                        in_strip))
    for wname, wmask in windows:
        print(f"\n{mode} paired abundance, {wname} (hex − square per seed):")
        for Rv in R_vals:
            by = {}
            for r in recs:
                if r["R"] != Rv:
                    continue
                a = (len(r["alive_idx"]) if wmask is None else
                     sum(1 for b in r["alive_idx"] if wmask[b]))
                by[(r["substrate"], r["seed"])] = a
            seeds = sorted({s for (_, s) in by})
            pairs = [(by.get(("square", s)), by.get(("hex", s)))
                     for s in seeds
                     if ("square", s) in by and ("hex", s) in by]
            if not pairs:
                continue
            sq = np.mean([p[0] for p in pairs])
            hx = np.mean([p[1] for p in pairs])
            wins = sum(1 for a, b in pairs if b > a)
            rel = (hx - sq) / sq * 100 if sq else float("nan")
            print(f"  R={Rv}: square {sq:.1f} vs hex {hx:.1f} "
                  f"({rel:+.0f}%), hex higher {wins}/{len(pairs)} seeds")
            key = f"abund_{'strip' if wmask else 'agg'}_R{Rv}"
            summary[key] = {"square": float(sq), "hex": float(hx),
                            "rel_pct": float(rel), "hex_gt_seeds": wins,
                            "n_seeds": len(pairs)}


def analyze(mode: str):
    out_path = ROOT / "runs" / MODES[mode]["out"]
    recs = [json.loads(l) for l in out_path.read_text().splitlines()]
    R_vals = sorted({r["R"] for r in recs})
    summary = {}
    _lock_table(recs, R_vals, mode, summary)
    _paired_psi(recs, R_vals, summary)
    _abundance(recs, R_vals, mode, summary)
    sp = ROOT / "runs" / MODES[mode]["out"].replace(".jsonl",
                                                    "_summary.json")
    sp.write_text(json.dumps(summary, indent=1))
    print(f"\nsummary → {sp.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.mode)
    else:
        dev = torch.device("cpu" if args.cpu else
                           ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"mode={args.mode} device={dev}")
        run_mode(args.mode, tuple(args.seeds), dev)
