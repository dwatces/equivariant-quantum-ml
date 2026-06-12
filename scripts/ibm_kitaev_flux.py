"""FINAL QPU TEST #2 (2026-06-07): Kitaev Z2 PLAQUETTE FLUX on real hardware = topological ORDER.

The project's QPU arc was all HALDANE band topology (single-particle chirality/edge). This is the
first KITAEV test = many-body TOPOLOGICAL ORDER. Classical ED (kitaev_order.py) showed the honeycomb
plaquette flux W_p = X0 Y1 Z2 X3 Y4 Z5 is EXACTLY conserved by the Kitaev Hamiltonian (square can't).
Hardware question: is the Z2 flux conserved on real qubits under Kitaev dynamics, and broken by a
non-Kitaev perturbation?

One hexagon, 6 qubits, ring edges coloured [Z,X,Y,Z,X,Y] for (0,1)(1,2)(2,3)(3,4)(4,5)(5,0)
=> [W_p, H_Kitaev] = 0 (verified by the sim gate). Prepare the +1 flux eigenstate
|+x>|+y>|0>|+x>|+y>|0> (each qubit a +1 eigenstate of its W_p Pauli), Trotter-evolve, measure <W_p>.
  * KITAEV evolution  -> <W_p> conserved (~+1).
  * KITAEV + transverse field (non-commuting) -> <W_p> drops (flux broken) = control.

DISCIPLINE: SIMULATE-FIRST; only --submit if the gate passes. No unread hardware number. Scope: a
single-plaquette flux-conservation signature (no anyons/degeneracy/scaling) = entry-level topological
order on hardware, genuinely new for this project.
"""
import os, sys, math, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import Statevector

N = 6
# ring edges (i, j, color): chosen so W_p = X0 Y1 Z2 X3 Y4 Z5 commutes with every bond term
EDGES = [(0, 1, "Z"), (1, 2, "X"), (2, 3, "Y"), (3, 4, "Z"), (4, 5, "X"), (5, 0, "Y")]
WP = ["X", "Y", "Z", "X", "Y", "Z"]           # plaquette Pauli per qubit
J, DT, STEPS, SHOTS = 1.0, 0.6, 2, 8192
HFIELD = 1.2                                    # transverse-field strength for the control
ROOTRUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def prep_flux_eigenstate(qc):
    """+1 eigenstate of W_p: qubit in +1 eigenstate of its WP Pauli (X->|+x>, Y->|+y>, Z->|0>)."""
    for q, p in enumerate(WP):
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.h(q); qc.s(q)                   # |+y> = S H |0>
        # Z -> |0> (nothing)


def kitaev_step(qc, field=False):
    for (i, j, c) in EDGES:
        g = {"X": RXXGate, "Y": RYYGate, "Z": RZZGate}[c](2 * J * DT)
        qc.append(g, [i, j])
    if field:
        for q in range(N):
            qc.rx(2 * HFIELD * DT, q)          # transverse field (breaks W_p)


def build(field=False, steps=STEPS, measure=True):
    qc = QuantumCircuit(N, N)
    prep_flux_eigenstate(qc)
    for _ in range(steps):
        kitaev_step(qc, field=field)
    if measure:                                 # rotate each qubit to read its WP Pauli in Z-basis
        for q, p in enumerate(WP):
            if p == "X":
                qc.h(q)
            elif p == "Y":
                qc.sdg(q); qc.h(q)
        qc.measure(range(N), range(N))
    return qc


def wp_from_sv(qc):
    sv = Statevector.from_instruction(qc)
    # measure W_p as expectation of the Pauli string
    from qiskit.quantum_info import Pauli
    label = "".join(WP[::-1])                    # qiskit little-endian
    return float(np.real(sv.expectation_value(Pauli(label))))


def wp_from_counts(counts):
    tot = sum(counts.values()); acc = 0
    for b, c in counts.items():
        bits = b.replace(" ", "")
        parity = sum(int(x) for x in bits) % 2
        acc += (1 if parity == 0 else -1) * c
    return acc / max(1, tot)


def simulate():
    wp0 = wp_from_sv(build(measure=False, steps=0))
    wpk = wp_from_sv(build(field=False, measure=False))
    wpf = wp_from_sv(build(field=True, measure=False))
    n2q = sum(1 for inst in build().data if inst.operation.num_qubits == 2)
    print(f"[sim] Kitaev hexagon N={N} | J={J} dt={DT} steps={STEPS} field={HFIELD}")
    print(f"[sim] <W_p> initial (t=0)              = {wp0:+.4f}  (prep check, want +1)")
    print(f"[sim] <W_p> after KITAEV evolution     = {wpk:+.4f}  (want ~+1 = flux CONSERVED)")
    print(f"[sim] <W_p> after KITAEV+FIELD (ctrl)  = {wpf:+.4f}  (want < kitaev = flux broken)")
    print(f"[sim] logical 2q gates/circuit = {n2q}")
    gate = (wp0 > 0.99) and (wpk > 0.95) and (wpf < wpk - 0.2) and (n2q <= 28)
    print(f"[gate] prep ok + flux conserved under Kitaev + broken by field + shallow: {gate}")
    print(f"[gate] {'>>> clean -> submittable' if gate else '>>> NOT clean -> do NOT submit'}")
    return gate


def submit(backend_name):
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    # batch: t=0 baseline, Kitaev steps 1 & 2 (conservation over time), field control at step 2
    specs = [("t0", dict(steps=0, field=False)),
             ("kitaev_s1", dict(steps=1, field=False)),
             ("kitaev_s2", dict(steps=2, field=False)),
             ("field_s2", dict(steps=2, field=True))]
    qcs = [build(measure=True, **kw) for _, kw in specs]
    labels = [s for s, _ in specs]
    svc = QiskitRuntimeService()
    backend = svc.backend(backend_name) if backend_name else svc.least_busy(operational=True, simulator=False)
    print(f"[backend] {backend.name} ({backend.num_qubits}q), queue={backend.status().pending_jobs}", flush=True)
    ref = transpile(qcs[2], backend, optimization_level=3, seed_transpiler=11)
    init = ref.layout.initial_index_layout(filter_ancillas=True)
    tq = transpile(qcs, backend, optimization_level=3, initial_layout=init, seed_transpiler=11)
    n2 = max(sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in tq)
    print(f"[transpile] pinned layout {init[:N]} | max transpiled 2q = {n2}", flush=True)
    sampler = SamplerV2(mode=backend); sampler.options.default_shots = SHOTS
    try:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    except Exception as e:
        print(f"[warn] DD not set: {e!r}", flush=True)
    job = sampler.run(tq)
    print(f"[job] {job.job_id()} submitted to {backend.name}", flush=True)
    open(os.path.join(ROOTRUN, "ibm_kitaev_flux_jobid.txt"), "w").write(job.job_id())
    res = job.result()
    wp = {}
    for lab, r in zip(labels, res):
        c = r.data.c.get_counts() if hasattr(r.data, "c") else r.data[list(r.data.keys())[0]].get_counts()
        wp[lab] = wp_from_counts(c)
    out = dict(backend=backend.name, job_id=job.job_id(), Wp=wp)
    json.dump(out, open(os.path.join(ROOTRUN, "ibm_kitaev_flux_result.json"), "w"), indent=2)
    print(f"\n[HARDWARE] <W_p>:  t0 {wp['t0']:+.3f} | Kitaev s1 {wp['kitaev_s1']:+.3f} | "
          f"Kitaev s2 {wp['kitaev_s2']:+.3f} | FIELD s2 {wp['field_s2']:+.3f}")
    print(f"[read] flux conserved under Kitaev (s1,s2 stay high) vs broken by field (drops) = topological order on HW")
    print(f"[saved] runs/ibm_kitaev_flux_result.json")


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
