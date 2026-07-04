"""Mode-split re-analysis of the crossover sweep (2026-07-03).

The probe (probe_fragmentation.py + figures/hexlife_frag_probe.png)
established that the square substrate's tight-spacing population is REAL
morphology (ring/annulus spots and 2a micro-texture; threshold-stable
counts), not threshold fragmentation. The pooled locking table therefore
mixes two populations on square. Split per pattern:

  tight = spacing < 3.05 a   (the lattice-scale morphology channel)
  wide  = spacing >= 3.05 a  (true kernel-scale spot arrays)

Valid for R >= 5 only (at R=4.5 the kernel scale ~2.9-3.4a overlaps the
lattice scale — the genuinely commensurate bridge; reported unsplit).

Outputs, per (substrate, R):
  * tight-mode prevalence (fraction of patterns) — the NEW morphology-
    selection observable; expect ~0 on hex if the effect is square-only.
  * per-mode lock4/lock6 (+-95% CI) and spacing — the honest "does array
    locking die" curve lives in the WIDE mode.
  * per-seed tight fractions for a sign test square vs hex.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
A_LAT = {"square": 1.0, "hex": math.sqrt(2 / math.sqrt(3))}
CUT = 3.05
R_SPLIT = 5.0


def ci(v):
    v = np.asarray(v)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(1.96 * v.std() / max(1, len(v)) ** 0.5)


def main():
    recs = [json.loads(l) for l in
            (ROOT / "runs" / "hexlife_crossover.jsonl").read_text()
            .splitlines()]
    R_vals = sorted({r["R"] for r in recs})
    summary = {}
    print(f"{'sub':7s} {'R':5s} {'mode':6s} {'n_pat':6s} {'frac':6s} "
          f"{'lock4 (±CI)':16s} {'lock6 (±CI)':16s} {'med spac/a':10s}")
    for name in ("square", "hex"):
        for Rv in R_vals:
            pats = [p for r in recs
                    if r["substrate"] == name and r["R"] == Rv
                    for p in r["patterns"]]
            if not pats:
                continue
            spac = np.array([p["spacing"] / A_LAT[name] for p in pats])
            if Rv < R_SPLIT:
                modes = {"all": np.ones(len(pats), bool)}
            else:
                modes = {"tight": spac < CUT, "wide": spac >= CUT}
            for mname, mask in modes.items():
                sel = [p for p, m in zip(pats, mask) if m]
                if not sel:
                    print(f"{name:7s} {Rv:<5} {mname:6s} 0")
                    continue
                l4, e4 = ci([math.cos(4 * p["phi4"]) for p in sel])
                l6, e6 = ci([math.cos(6 * p["phi6"]) for p in sel])
                sp = float(np.median([p["spacing"] / A_LAT[name]
                                      for p in sel]))
                frac = len(sel) / len(pats)
                print(f"{name:7s} {Rv:<5} {mname:6s} {len(sel):<6d} "
                      f"{frac:.2f}   {l4:+.3f}±{e4:.3f}    "
                      f"{l6:+.3f}±{e6:.3f}    {sp:.2f}")
                summary[f"{name}_R{Rv}_{mname}"] = {
                    "n": len(sel), "frac": frac,
                    "lock4": [l4, e4], "lock6": [l6, e6],
                    "med_spacing_a": sp}
    # per-seed tight prevalence, paired square vs hex (R >= 5)
    print("\ntight-mode prevalence per seed (R>=5): square vs hex")
    for Rv in [r for r in R_vals if r >= R_SPLIT]:
        row = {}
        for name in ("square", "hex"):
            fr = []
            for r in recs:
                if r["substrate"] == name and r["R"] == Rv:
                    sp = [p["spacing"] / A_LAT[name] for p in r["patterns"]]
                    if sp:
                        fr.append(np.mean([s < CUT for s in sp]))
            row[name] = fr
        sq, hx = np.array(row["square"]), np.array(row["hex"])
        wins = int((sq > hx).sum())
        print(f"  R={Rv}: square {sq.mean():.2f} vs hex {hx.mean():.2f} "
              f"(square higher {wins}/{len(sq)} seeds)")
        summary[f"tightprev_R{Rv}"] = {
            "square_mean": float(sq.mean()), "hex_mean": float(hx.mean()),
            "square_gt_seeds": wins, "n_seeds": len(sq)}
    (ROOT / "runs" / "hexlife_crossover_modes.json").write_text(
        json.dumps(summary, indent=1))
    print("\nsummary → runs/hexlife_crossover_modes.json")


if __name__ == "__main__":
    main()
