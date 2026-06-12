"""CNN x p3_9 honeycomb sweep on Modal — one A10G container PER SEED (5 parallel).
Each container rebuilds the circuit (CPU ~2 min) and runs all (n, arm) configs
for its seed; the local entrypoint merges mean/std across seeds.

    modal run scripts/honeycomb_cnn_modal.py
"""
import json

import modal

app = modal.App("honeycomb-cnn")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "stim==1.16.0", "pymatching", "torch")
    .add_local_file("scripts/honeycomb_decoder_core.py",
                    "/root/honeycomb_decoder_core.py")
    .add_local_file("scripts/honeycomb_cnn_core.py",
                    "/root/honeycomb_cnn_core.py")
)


@app.function(image=image, gpu="A10G", timeout=10800)
def run_seed(seed: int):
    from honeycomb_cnn_core import run_experiment_cnn

    return run_experiment_cnn(seeds=(seed,))


@app.local_entrypoint()
def main():
    seeds = [0, 1, 2, 3, 4]
    parts = list(run_seed.map(seeds))
    merged = {k: v for k, v in parts[0].items() if k != "sweep"}
    merged["seeds"] = seeds
    merged["sweep"] = {}
    import numpy as np

    for n in parts[0]["sweep"]:
        row = {}
        for arm in parts[0]["sweep"][n]:
            accs = [p["sweep"][n][arm][0] for p in parts]
            row[arm] = (float(np.mean(accs)), float(np.std(accs)))
        merged["sweep"][n] = row
        print(f"  n={n}: " + "  ".join(
            f"{a} {row[a][0]:.4f}±{row[a][1]:.4f}" for a in row))
    merged["runtime_s_per_seed"] = [p["runtime_s"] for p in parts]
    with open("runs/honeycomb_decoder_cnn.json", "w") as f:
        json.dump(merged, f, indent=1, default=str)
    print("saved -> runs/honeycomb_decoder_cnn.json")
