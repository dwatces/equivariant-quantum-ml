"""Local gate run for the honeycomb equivariant decoder: L=3 end-to-end smoke,
then L=6 gates + tiny train on CPU. Full sweep goes to Modal afterwards.
Usage: .venv-gpu/bin/python scripts/honeycomb_decoder_local.py
"""
import json
import sys
import time

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import run_experiment  # noqa: E402

t0 = time.time()
print("==== SMOKE L=3 (R=5, s_extra=1 -> MY readout) ====", flush=True)
out3 = run_experiment(L=3, M=3, R=5, p=0.008, mode="cx", s_extra=1, smoke=True)
print("\n==== GATES + SMOKE L=6 (R=9, s_extra=1 -> MY readout) ====", flush=True)
out6 = run_experiment(L=6, M=6, R=9, p=0.008, mode="cx", s_extra=1, smoke=True)
with open("runs/honeycomb_decoder_gates.json", "w") as f:
    json.dump({"L3": out3, "L6": out6}, f, indent=1, default=str)
print(f"\nALL GATES DONE {time.time() - t0:.0f}s "
      f"-> runs/honeycomb_decoder_gates.json", flush=True)
