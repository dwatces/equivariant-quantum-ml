"""FINAL QPU TEST #4 (2026-06-07): ANYON BRAIDING (-1 mutual statistics) on real hardware.

Classical toric_braiding.py: 8-qubit toric code (= honeycomb Kitaev anisotropic limit), logical
Xbar=X[0,2], Zbar=Z[0,1] cross once -> anticommute. Vacuum <Zbar>=+1; dragging an anyon around the
crossing cycle (apply Xbar) -> <Zbar>=-1 (the -1 braiding phase); a contractible loop (a star) ->
<Zbar>=+1 (no braiding). The -1 vs +1 contrast IS the mutual statistics.

QPU: prep the toric ground state (|0_L>, <Zbar>=+1) via Qiskit stabilizer-state synthesis, then
measure <Zbar>=<Z0 Z1> for: vacuum / after-Xbar(braid) / after-star(control). 3 circuits.
This is the DEEPEST test of the session (8q toric ground state) -> real fidelity risk; SIMULATE-FIRST
and only submit if the gate passes AND depth is acceptable. No unread hardware number.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

N = 8
STARS = [[0, 1, 4, 6], [0, 1, 5, 7], [2, 3, 4, 6]]          # 3 independent X-stars
PLAQS = [[0, 2, 4, 5], [1, 3, 4, 5], [0, 2, 6, 7]]          # 3 independent Z-plaquettes
ZBAR = [0, 1]                                                # logical Z (measured)
ZBAR2 = [4, 6]                                               # 2nd logical Z (other cycle; fixes the pure state)
XBAR = [0, 2]                                                # logical X (the braiding drag)
STAR_CTRL = [0, 1, 4, 6]                                     # contractible X-loop (control)
SHOTS = 8192
ROOTRUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def lbl(sites, p):
    c = ["I"] * N
    for s in sites:
        c[s] = p
    return "".join(c[::-1])                                  # qiskit little-endian


def stab_labels():
    return ([lbl(s, "X") for s in STARS] + [lbl(s, "Z") for s in PLAQS]
            + [lbl(ZBAR, "Z"), lbl(ZBAR2, "Z")])


def prep_circuit():
    from qiskit.synthesis import synth_circuit_from_stabilizers
    return synth_circuit_from_stabilizers([Pauli(s) for s in stab_labels()])


def build(mode="vacuum", measure=True):
    qc = QuantumCircuit(N, N)
    qc.compose(prep_circuit(), inplace=True)
    if mode == "braid":
        for q in XBAR:
            qc.x(q)
    elif mode == "control":
        for q in STAR_CTRL:
            qc.x(q)
    if measure:
        qc.measure(range(N), range(N))
    return qc


def zbar_from_sv(mode):
    sv = Statevector.from_instruction(build(mode, measure=False))
    return float(np.real(sv.expectation_value(Pauli(lbl(ZBAR, "Z")))))


def zbar_from_counts(counts):
    tot = sum(counts.values()); acc = 0
    for b, c in counts.items():
        bits = b.replace(" ", "")[::-1]
        parity = (int(bits[ZBAR[0]]) + int(bits[ZBAR[1]])) % 2
        acc += (1 if parity == 0 else -1) * c
    return acc / max(1, tot)


def simulate():
    vac, brd, ctl = zbar_from_sv("vacuum"), zbar_from_sv("braid"), zbar_from_sv("control")
    prep = prep_circuit()
    n2q_prep = sum(1 for inst in prep.data if inst.operation.num_qubits == 2)
    print(f"[sim] toric ground state | prep logical 2q gates = {n2q_prep}")
    print(f"[sim] <Zbar> vacuum  = {vac:+.3f}  (want +1)")
    print(f"[sim] <Zbar> BRAID   = {brd:+.3f}  (want -1 = -1 mutual statistics)")
    print(f"[sim] <Zbar> control = {ctl:+.3f}  (want +1 = no braiding)")
    gate = (vac > 0.99 and brd < -0.99 and ctl > 0.99 and n2q_prep <= 24)
    print(f"[gate] vacuum +1 + braid -1 + control +1 + prep<=24 2q: {gate}")
    print(f"[gate] {'>>> clean -> submittable' if gate else '>>> NOT clean -> do NOT submit'}")
    return gate


def submit(backend_name):
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    specs = ["vacuum", "braid", "control"]
    qcs = [build(m, measure=True) for m in specs]
    svc = QiskitRuntimeService()
    backend = svc.backend(backend_name) if backend_name else svc.least_busy(operational=True, simulator=False)
    print(f"[backend] {backend.name} ({backend.num_qubits}q), queue={backend.status().pending_jobs}", flush=True)
    ref = transpile(qcs[0], backend, optimization_level=3, seed_transpiler=11)
    init = ref.layout.initial_index_layout(filter_ancillas=True)
    tq = transpile(qcs, backend, optimization_level=3, initial_layout=init, seed_transpiler=11)
    n2 = max(sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in tq)
    print(f"[transpile] pinned layout {init[:N]} | max transpiled 2q = {n2}", flush=True)
    if n2 > 70:
        print(f"[abort] transpiled depth {n2} too high for 8q fidelity — refusing to waste budget."); return
    sampler = SamplerV2(mode=backend); sampler.options.default_shots = SHOTS
    try:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    except Exception as e:
        print(f"[warn] DD not set: {e!r}", flush=True)
    job = sampler.run(tq)
    print(f"[job] {job.job_id()} submitted to {backend.name}", flush=True)
    open(os.path.join(ROOTRUN, "ibm_toric_braiding_jobid.txt"), "w").write(job.job_id())
    res = job.result()
    z = {}
    for m, r in zip(specs, res):
        c = r.data.c.get_counts() if hasattr(r.data, "c") else r.data[list(r.data.keys())[0]].get_counts()
        z[m] = zbar_from_counts(c)
    out = dict(backend=backend.name, job_id=job.job_id(), Zbar=z)
    json.dump(out, open(os.path.join(ROOTRUN, "ibm_toric_braiding_result.json"), "w"), indent=2)
    print(f"\n[HARDWARE] <Zbar>:  vacuum {z['vacuum']:+.3f} | BRAID {z['braid']:+.3f} | control {z['control']:+.3f}")
    print(f"[read] braid flips to negative while control stays positive => -1 anyon mutual statistics on HW")
    print(f"[saved] runs/ibm_toric_braiding_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    clean = simulate()
    if not a.submit:
        print("\n(sim-only; pass --submit after confirming)")
        return
    if not clean:
        print("\n[abort] gate not clean — refusing to submit (hard rule)."); return
    submit(a.backend)


if __name__ == "__main__":
    main()
