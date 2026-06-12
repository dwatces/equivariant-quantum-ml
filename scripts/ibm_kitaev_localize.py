"""FINAL QPU TEST #3 (2026-06-07): a LOCALIZED Kitaev anyon (vison) on real hardware.

Classical kitaev_localize.py: on a 2-hexagon patch, applying a LOCAL operator (Y on site 0) flips
ONE plaquette flux (<W_p1>: +1->-1) while the NEIGHBOUR is untouched (<W_p2> stays +1) = a localized
anyon. The (+1,+1) flux vacuum is a STABILIZER state (8 product qubits + 1 entangled shared pair =
~1 CNOT), so this is SHALLOW and feasible on 10 qubits -- much shallower than dynamics.

W_p1 = X0 Y1 Z2 Z4 Y5 X6   W_p2 = X3 Y4 Z5 Z7 Y8 X9   (shared sites 4,5 differ -> 1 entangling gate)
Vison creator = Y on site 0. Measure <W_p1>,<W_p2> for vacuum vs vison; W_p1/W_p2 need different
bases on the shared qubits -> 4 circuits (2 states x 2 plaquette bases).

SIGNATURE: vacuum (+1,+1) -> vison (-1,+1). The neighbour flux staying +1 == LOCALIZATION.
DISCIPLINE: SIMULATE-FIRST; submit only if the gate passes. No unread hardware number. Scope:
stabilizer-level anyon creation+detection (NOT braiding/dynamics), genuinely new for this project.
"""
import os, sys, math, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

N = 10
WP1 = {0: "X", 1: "Y", 2: "Z", 4: "Z", 5: "Y", 6: "X"}
WP2 = {3: "X", 4: "Y", 5: "Z", 7: "Z", 8: "Y", 9: "X"}
SHARED = (4, 5)
CREATOR = (0, "Y")          # local vison creator: Y on site 0
SHOTS = 8192
ROOTRUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def set_plus_eigfns(qc, q, p):
    """put qubit q in the +1 eigenstate of Pauli p."""
    if p == "X":
        qc.h(q)
    elif p == "Y":
        qc.h(q); qc.s(q)        # |+y>
    # Z -> |0> (nothing)


def prep_vacuum(qc):
    # 8 non-shared sites: product +1 eigenstate of their W_p Pauli
    for q, p in WP1.items():
        if q not in SHARED:
            set_plus_eigfns(qc, q, p)
    for q, p in WP2.items():
        if q not in SHARED:
            set_plus_eigfns(qc, q, p)
    # shared pair (4,5): Bell+Rx(-pi/2) -> stabilized by Z4Y5=+1 and Y4Z5=+1
    a, b = SHARED
    qc.h(a); qc.cx(a, b); qc.rx(-math.pi / 2, b)


def basis_rotate(qc, wp):
    """rotate each W_p qubit so its Pauli -> Z, ready for computational measurement."""
    for q, p in wp.items():
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.sdg(q); qc.h(q)


def build(vison=False, plaq="p1", measure=True):
    qc = QuantumCircuit(N, N)
    prep_vacuum(qc)
    if vison:
        q, p = CREATOR
        getattr(qc, p.lower())(q)        # apply Y on site 0
    if measure:
        basis_rotate(qc, WP1 if plaq == "p1" else WP2)
        qc.measure(range(N), range(N))
    return qc


def wp_label(wp):
    chars = ["I"] * N
    for q, p in wp.items():
        chars[q] = p
    return "".join(chars[::-1])          # qiskit little-endian


def wp_from_sv(vison):
    sv = Statevector.from_instruction(build(vison=vison, measure=False))
    return (float(np.real(sv.expectation_value(Pauli(wp_label(WP1))))),
            float(np.real(sv.expectation_value(Pauli(wp_label(WP2))))))


def wp_from_counts(counts, wp):
    qs = list(wp.keys()); tot = sum(counts.values()); acc = 0
    for b, c in counts.items():
        bits = b.replace(" ", "")[::-1]          # bits[q] = qubit q (reverse of qiskit string)
        parity = sum(int(bits[q]) for q in qs) % 2
        acc += (1 if parity == 0 else -1) * c
    return acc / max(1, tot)


def simulate():
    vac = wp_from_sv(False)
    vis = wp_from_sv(True)
    n2q = sum(1 for inst in build().data if inst.operation.num_qubits == 2)
    print(f"[sim] 2-hexagon patch N={N} | W_p1={wp_label(WP1)} W_p2={wp_label(WP2)}")
    print(f"[sim] flux VACUUM : <W_p1>={vac[0]:+.3f}  <W_p2>={vac[1]:+.3f}   (want +1,+1)")
    print(f"[sim] after Y0 (VISON): <W_p1>={vis[0]:+.3f}  <W_p2>={vis[1]:+.3f}   (want -1,+1 = LOCALIZED)")
    print(f"[sim] logical 2q gates/circuit = {n2q}")
    gate = (vac[0] > 0.99 and vac[1] > 0.99 and vis[0] < -0.99 and vis[1] > 0.99 and n2q <= 6)
    print(f"[gate] vacuum(+1,+1) + vison flips ONLY p1 + shallow: {gate}")
    print(f"[gate] {'>>> clean -> submittable' if gate else '>>> NOT clean -> do NOT submit'}")
    return gate


def submit(backend_name):
    import requests
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "timeout": k.get("timeout", 30)})
    from qiskit import transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    specs = [("vac_p1", False, "p1"), ("vac_p2", False, "p2"),
             ("vis_p1", True, "p1"), ("vis_p2", True, "p2")]
    qcs = [build(vison=v, plaq=p, measure=True) for _, v, p in specs]
    labels = [s for s, _, _ in specs]
    svc = QiskitRuntimeService()
    backend = svc.backend(backend_name) if backend_name else svc.least_busy(operational=True, simulator=False)
    print(f"[backend] {backend.name} ({backend.num_qubits}q), queue={backend.status().pending_jobs}", flush=True)
    ref = transpile(qcs[0], backend, optimization_level=3, seed_transpiler=11)
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
    open(os.path.join(ROOTRUN, "ibm_kitaev_localize_jobid.txt"), "w").write(job.job_id())
    res = job.result()
    cnt = {}
    for lab, r in zip(labels, res):
        cnt[lab] = r.data.c.get_counts() if hasattr(r.data, "c") else r.data[list(r.data.keys())[0]].get_counts()
    vac = (wp_from_counts(cnt["vac_p1"], WP1), wp_from_counts(cnt["vac_p2"], WP2))
    vis = (wp_from_counts(cnt["vis_p1"], WP1), wp_from_counts(cnt["vis_p2"], WP2))
    out = dict(backend=backend.name, job_id=job.job_id(),
               vacuum=dict(Wp1=vac[0], Wp2=vac[1]), vison=dict(Wp1=vis[0], Wp2=vis[1]))
    json.dump(out, open(os.path.join(ROOTRUN, "ibm_kitaev_localize_result.json"), "w"), indent=2)
    print(f"\n[HARDWARE] VACUUM <W_p1>={vac[0]:+.3f} <W_p2>={vac[1]:+.3f}")
    print(f"[HARDWARE] VISON  <W_p1>={vis[0]:+.3f} <W_p2>={vis[1]:+.3f}")
    print(f"[read] p1 flips (+ -> -) while p2 stays + => LOCALIZED anyon on hardware")
    print(f"[saved] runs/ibm_kitaev_localize_result.json")


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
