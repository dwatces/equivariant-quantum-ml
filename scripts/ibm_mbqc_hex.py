"""
MBQC ON THE HEXAGON CLUSTER -- measurement-programmed MAGIC, on real hardware.
2026-06-10. The hardware capstone of today's two classical results: (1) the honeycomb
cluster/AKLT states are universal MBQC resources (the substrate computes; measurements
are the program); (2) magic is the quantum/classical separator (kitaev_magic.py).
Here: a 6-qubit hexagon cluster wire; the FIRST measurement's angle theta is the
program. theta=0 -> Clifford output (zero magic); theta=pi/4 -> T-type output
(single-qubit SRE log2(4/3)=0.415). Corrections are pure Pauli frame (non-Clifford
angle is FIRST in the flow -> no adaptive feed-forward; static shallow circuit).

DISCIPLINE: sim gate first (branch-by-branch Pauli-frame verification, transpile-depth
budget guard); submit ONLY if clean; one device; ~10 s of the remaining 43 s.
Usage:  python ibm_mbqc_hex.py            # sim gate only
        python ibm_mbqc_hex.py --submit   # gate, then submit + retrieve + analyze
"""
import json
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix

NQ = 6                      # hexagon: q0..q4 measured, q5 = output
THETAS = {"clifford": 0.0, "magic": np.pi / 4}
PAULIS = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
          "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.array([[1, 0], [0, -1]])}
SHOTS = 4096


def log(m):
    print(m, flush=True)


def cluster_circuit(theta, ring, tomo):
    """Hexagon cluster (path q0-..-q5, +closing CZ q5-q0 if ring); q0 measured at
    angle theta (operator cos.X+sin.Y -> basis change Rz(-t),H), q1..q4 at X; output
    q5 pre-rotated for tomography basis tomo in {X,Y,Z}. All measured at the end."""
    qc = QuantumCircuit(NQ, NQ)
    qc.h(range(NQ))
    for i in range(NQ - 1):
        qc.cz(i, i + 1)
    if ring:
        qc.cz(NQ - 1, 0)
    qc.barrier()
    qc.rz(-theta, 0)
    qc.h(0)
    for i in range(1, NQ - 1):
        qc.h(i)
    if tomo == "X":
        qc.h(NQ - 1)
    elif tomo == "Y":
        qc.sdg(NQ - 1)
        qc.h(NQ - 1)
    qc.measure(range(NQ), range(NQ))
    return qc


def branch_states(theta, ring):
    """Exact post-measurement output states for all 32 outcome branches."""
    qc = QuantumCircuit(NQ)
    qc.h(range(NQ))
    for i in range(NQ - 1):
        qc.cz(i, i + 1)
    if ring:
        qc.cz(NQ - 1, 0)
    qc.rz(-theta, 0)
    qc.h(0)
    for i in range(1, NQ - 1):
        qc.h(i)
    psi = np.asarray(Statevector(qc)).reshape([2] * NQ)  # index order q5,q4,...,q0
    out = {}
    for s in range(32):
        bits = [(s >> i) & 1 for i in range(5)]  # bits[i] = outcome of qubit i
        sub = psi
        # project qubits q0..q4 (statevector axes are reversed: axis k = qubit NQ-1-k)
        sl = [slice(None)] * NQ
        for i, b in enumerate(bits):
            sl[NQ - 1 - i] = b
        vec = psi[tuple(sl)]  # remaining axis = q5
        p = np.vdot(vec, vec).real
        if p < 1e-12:
            out[s] = None
            continue
        out[s] = vec / np.sqrt(p)
    return out


def find_frame(theta, ring):
    """For each branch find the Pauli mapping it to branch-0; None if impossible."""
    br = branch_states(theta, ring)
    ref = br[0]
    frame = {}
    for s, v in br.items():
        if v is None:
            frame[s] = None
            continue
        got = None
        for name, P in PAULIS.items():
            if state_fidelity(Statevector(P @ v), Statevector(ref)) > 1 - 1e-9:
                got = name
                break
        if got is None:
            return None, None
        frame[s] = got
    return frame, ref


def bloch(vec):
    rho = np.outer(vec, vec.conj())
    return {p: float(np.trace(rho @ PAULIS[p]).real) for p in "XYZ"}


def magic_witness(r):
    s4 = 1.0 + r["X"] ** 4 + r["Y"] ** 4 + r["Z"] ** 4
    return float(-np.log2(s4 / 2.0))


# ---------------------------------------------------------------- SIM GATE
log("== SIM GATE ==")
RESULT = {"design": {}}
cand = {}
for ring in (True, False):
    frames, targets, ok = {}, {}, True
    for label, th in THETAS.items():
        fr, ref = find_frame(th, ring)
        if fr is None:
            ok = False
            log(f"  {label} ring={ring}: Pauli frame IMPOSSIBLE")
            break
        frames[label], targets[label] = fr, bloch(ref)
        r = targets[label]
        log(f"  ring={ring} target[{label}]: r=({r['X']:+.3f},{r['Y']:+.3f},{r['Z']:+.3f}) "
            f"magic={magic_witness(r):.4f}")
    if ok:
        gap = abs(magic_witness(targets["magic"]) - magic_witness(targets["clifford"]))
        cand[ring] = (gap, frames, targets)
        log(f"  ring={ring}: programmed-magic gap = {gap:.4f}")
USE_RING = max(cand, key=lambda k: cand[k][0])
GAPSEL, FRAMES, TARGETS = cand[USE_RING]
log(f"  -> selected {'RING' if USE_RING else 'PATH (the hexagon walk)'} topology, gap {GAPSEL:.4f}")
if True in cand and cand[True][0] < 0.05:
    log("  (note: the closed hexagon RING cancels the program -- loop interference; "
        "recorded as a sim observation)")
RESULT["design"]["topology"] = "ring" if USE_RING else "path"
RESULT["design"]["ring_gap"] = cand.get(True, (None,))[0]
RESULT["design"]["targets"] = TARGETS
RESULT["design"]["target_magic"] = {l: magic_witness(TARGETS[l]) for l in THETAS}
assert GAPSEL > 0.3, "GATE FAIL: programs don't separate in magic"

# transpile gate
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backend = service.backend("ibm_marrakesh")
CIRCS, META = [], []
for label, th in THETAS.items():
    for tomo in "XYZ":
        CIRCS.append(cluster_circuit(th, USE_RING, tomo))
        META.append((label, tomo))
tc = transpile(CIRCS, backend, optimization_level=3, seed_transpiler=11)
n2q = [c.num_nonlocal_gates() for c in tc]
log(f"  transpiled 2q counts: {n2q} (budget guard <= 45 each)")
RESULT["design"]["transpiled_2q"] = n2q
assert max(n2q) <= 45, "GATE FAIL: too deep"
log("== SIM GATE PASSED ==")

if "--submit" not in sys.argv:
    log("(dry run -- rerun with --submit to fly)")
    sys.exit(0)

# ---------------------------------------------------------------- SUBMIT
from qiskit_ibm_runtime import SamplerV2

log(f"== SUBMIT to {backend.name}, {len(tc)} circuits x {SHOTS} shots ==")
sampler = SamplerV2(mode=backend)
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XY4"
job = sampler.run(tc, shots=SHOTS)
log(f"job id: {job.job_id()}")
res = job.result()
log(f"usage: {job.usage()} s")
RESULT["job_id"] = job.job_id()
RESULT["usage_s"] = job.usage()

# ---------------------------------------------------------------- ANALYZE
HW = {l: {} for l in THETAS}
for (label, tomo), pub in zip(META, res):
    counts = pub.data.c.get_counts()
    frame = FRAMES[label]
    num, tot = 0.0, 0
    for bitstr, ct in counts.items():
        bits = bitstr[::-1]  # bits[i] = qubit i
        s = sum((1 << i) for i in range(5) if bits[i] == "1")
        P = frame.get(s)
        if P is None:
            continue
        sign = 1.0
        if P != "I" and P != tomo and not (P == "Z" and tomo == "Z"):
            anti = {("X", "Y"), ("X", "Z"), ("Y", "X"), ("Y", "Z"), ("Z", "X"), ("Z", "Y")}
            if (P, tomo) in anti:
                sign = -1.0
        out = 1.0 if bits[5] == "0" else -1.0
        num += sign * out * ct
        tot += ct
    HW[label][tomo] = num / tot
for label in THETAS:
    r = HW[label]
    tgt = TARGETS[label]
    dot = sum(r[p] * tgt[p] for p in "XYZ")
    nr = np.sqrt(sum(r[p] ** 2 for p in "XYZ"))
    nt = np.sqrt(sum(tgt[p] ** 2 for p in "XYZ"))
    rhat = {p: (r[p] / nr if nr > 0 else 0.0) for p in "XYZ"}
    log(f"HW [{label}]: r=({r['X']:+.3f},{r['Y']:+.3f},{r['Z']:+.3f}) |r|={nr:.3f}  "
        f"dir-overlap with target={dot/(nr*nt+1e-12):+.3f}  "
        f"magic witness raw={magic_witness(r):.4f}  on r-hat={magic_witness(rhat):.4f}")
    HW[label]["dir_overlap"] = dot / (nr * nt + 1e-12)
    HW[label]["magic_raw"] = magic_witness(r)
    HW[label]["magic_rhat"] = magic_witness(rhat)
RESULT["hardware"] = HW
with open("runs/ibm_mbqc_hex.json", "w") as f:
    json.dump(RESULT, f, indent=1, default=float)
log("saved runs/ibm_mbqc_hex.json")
