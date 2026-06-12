"""Big-n MLP extension (local GPU, runs while Modal does the CNN sweep):
does the unsaturated p3_9 curve keep climbing toward MWPM with 4x data and
5x training steps — and does the majority-dead plain arm ever wake up?
n in {64k (re-run at 20k steps for comparability), 256k}, 3 seeds.
Output: runs/honeycomb_decoder_bign.{log,json}
"""
import json
import sys

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import run_experiment  # noqa: E402

out = run_experiment(L=6, M=6, R=9, p=0.008, mode="cx", s_extra=1,
                     ntrains=(64000, 256000), seeds=(0, 1, 2), steps=20000)
with open("runs/honeycomb_decoder_bign.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print(f"DONE {out['runtime_s']:.0f}s -> runs/honeycomb_decoder_bign.json")
