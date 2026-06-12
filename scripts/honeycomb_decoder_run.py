"""Full honeycomb equivariant-decoder sweep (local GPU).
L=6, R=9, MY termination (s_extra=1), auto-p (gates landed at 0.002),
arms plain / c3tw / p3_9, n in {250,1k,4k,16k,64k}, 5 seeds, 200k test shots.
Output: runs/honeycomb_decoder.{log,json}
"""
import json
import sys

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import run_experiment  # noqa: E402

out = run_experiment(L=6, M=6, R=9, p=0.008, mode="cx", s_extra=1)
with open("runs/honeycomb_decoder.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print(f"DONE {out['runtime_s']:.0f}s -> runs/honeycomb_decoder.json")
