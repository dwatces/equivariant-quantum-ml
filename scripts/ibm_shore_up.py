"""SHORE UP the two strongest Kitaev/topological-order QPU results to portfolio-grade:
anyon BRAIDING (-1 mutual statistics) + Kitaev FLUX conservation, across 3 INDEPENDENT backends
with DD + bootstrap shot-noise error bars + cross-device mean+-std. No twirling (budget).

Reuses the verified circuit builders from ibm_toric_braiding.py and ibm_kitaev_flux.py.
Submits one small batch per backend per test; processes backends one at a time and STOPS if the
remaining QPU budget runs low. Every number read from completed jobs; figures from the JSON.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import ibm_toric_braiding as BR
import ibm_kitaev_flux as FL

BACKENDS = ["ibm_marrakesh", "ibm_kingston", "ibm_fez"]
ROOTRUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def boot(counts, fn, nboot=300, seed=0):
    """bootstrap shot-noise std of an observable fn(counts-like) by resampling shots."""
    rng = np.random.default_rng(seed)
    keys = list(counts.keys()); w = np.array([counts[k] for k in keys], float); w /= w.sum()
    Ntot = int(sum(counts.values()))
    vals = []
    for _ in range(nboot):
        draw = rng.multinomial(Ntot, w)
        vals.append(fn({k: int(d) for k, d in zip(keys, draw) if d > 0}))
    return float(np.std(vals))


def run_test(svc, SamplerV2, transpile, name, qcs, ref_idx, observables, logf):
    """observables: dict label -> (circuit_index, fn(counts)->value). Returns per-backend dict."""
    def log(m): print(m); logf.write(m + "\n"); logf.flush()
    per_dev = {}
    for bname in BACKENDS:
        u = svc.usage()
        if u["usage_remaining_seconds"] < 25:
            log(f"  [budget] {u['usage_remaining_seconds']}s left -> stop before {bname}"); break
        try:
            backend = svc.backend(bname)
            ref = transpile(qcs[ref_idx], backend, optimization_level=3, seed_transpiler=11)
            init = ref.layout.initial_index_layout(filter_ancillas=True)
            tq = transpile(qcs, backend, optimization_level=3, initial_layout=init, seed_transpiler=11)
            n2 = max(sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in tq)
            sampler = SamplerV2(mode=backend); sampler.options.default_shots = 8192
            try:
                sampler.options.dynamical_decoupling.enable = True
                sampler.options.dynamical_decoupling.sequence_type = "XY4"
            except Exception:
                pass
            job = sampler.run(tq)
            log(f"  [{name}/{bname}] job {job.job_id()} (max 2q={n2})")
            res = job.result()
            counts = []
            for r in res:
                counts.append(r.data.c.get_counts() if hasattr(r.data, "c")
                              else r.data[list(r.data.keys())[0]].get_counts())
            dev = {}
            for lab, (ci, fn) in observables.items():
                dev[lab] = (fn(counts[ci]), boot(counts[ci], fn))
            per_dev[bname] = dev
            log(f"      " + " | ".join(f"{lab} {v:+.3f}±{e:.3f}" for lab, (v, e) in dev.items()))
        except Exception as e:
            log(f"  [{name}/{bname}] FAILED: {e!r}")
    return per_dev


def aggregate(per_dev, labels):
    out = {}
    for lab in labels:
        vals = [per_dev[b][lab][0] for b in per_dev if lab in per_dev[b]]
        out[lab] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), n_dev=len(vals),
                        per_device={b: per_dev[b][lab] for b in per_dev if lab in per_dev[b]})
    return out


def main():
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService()
    logf = open(os.path.join(ROOTRUN, "ibm_shore_up.log"), "a")
    def log(m): print(m); logf.write(m + "\n"); logf.flush()
    log(f"\n=== SHORE UP (3 backends, DD, bootstrap) ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
    log(f"  budget start: {svc.usage()['usage_remaining_seconds']}s")

    results = {}

    # ---- BRAIDING (toric code): <Zbar> for vacuum/braid/control ----
    br_qcs = [BR.build("vacuum"), BR.build("braid"), BR.build("control")]
    br_obs = {"vacuum": (0, BR.zbar_from_counts), "braid": (1, BR.zbar_from_counts),
              "control": (2, BR.zbar_from_counts)}
    log("\n[BRAIDING] <Zbar> vacuum/braid/control across devices:")
    br_dev = run_test(svc, SamplerV2, transpile, "braid", br_qcs, 0, br_obs, logf)
    results["braiding"] = aggregate(br_dev, ["vacuum", "braid", "control"])

    # ---- KITAEV FLUX: <W_p> for t0 / kitaev_s2 / field_s2 ----
    fl_specs = [dict(steps=0, field=False), dict(steps=2, field=False), dict(steps=2, field=True)]
    fl_qcs = [FL.build(measure=True, **kw) for kw in fl_specs]
    wpfn = lambda c: FL.wp_from_counts(c)
    fl_obs = {"t0": (0, wpfn), "kitaev_s2": (1, wpfn), "field_s2": (2, wpfn)}
    log("\n[KITAEV FLUX] <W_p> t0/kitaev_s2/field across devices:")
    fl_dev = run_test(svc, SamplerV2, transpile, "flux", fl_qcs, 1, fl_obs, logf)
    results["flux"] = aggregate(fl_dev, ["t0", "kitaev_s2", "field_s2"])

    json.dump(results, open(os.path.join(ROOTRUN, "ibm_shore_up.json"), "w"), indent=2)
    log(f"\n  budget end: {svc.usage()['usage_remaining_seconds']}s")
    # summary
    if "braiding" in results and results["braiding"].get("braid", {}).get("n_dev"):
        b = results["braiding"]
        log(f"\n  BRAIDING (cross-device): vacuum {b['vacuum']['mean']:+.3f}±{b['vacuum']['std']:.3f} | "
            f"braid {b['braid']['mean']:+.3f}±{b['braid']['std']:.3f} | "
            f"control {b['control']['mean']:+.3f}±{b['control']['std']:.3f}")
    if "flux" in results and results["flux"].get("kitaev_s2", {}).get("n_dev"):
        f = results["flux"]
        log(f"  FLUX (cross-device): t0 {f['t0']['mean']:+.3f}±{f['t0']['std']:.3f} | "
            f"kitaev {f['kitaev_s2']['mean']:+.3f}±{f['kitaev_s2']['std']:.3f} | "
            f"field {f['field_s2']['mean']:+.3f}±{f['field_s2']['std']:.3f}")
    log("=== done ===")


if __name__ == "__main__":
    main()
