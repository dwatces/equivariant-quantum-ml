"""SECOND-SIZE STRUCTURE CHECK (pre-figures): does the honeycomb paper's
structural story survive at L=9, or is it an L=6 accident?

At L=9 the closed abelian order-9 subgroup is {C3^a T(3,3)^b} (T(2,2) does
not close: R(2,2)=(-4,2) != (2,2) mod 9; R(3,3)=(-6,3)=(3,3) mod 9).

PREDICTIONS LOGGED BEFORE THE RUN (hard rule):
  P1 termination claim generalizes: G2 passes and EXACTLY 2 deterministic
     wraps exist at S = 3R+1 (MY readout) on the L=9 torus.
  P2 translations solve with A = I; C3 solves with A = an ORDER-3 element of
     GL(2,F2) — i.e. [[1,1],[1,0]] or its square [[0,1],[1,1]] (which of the
     two depends on the independently-derived wrap basis at L=9, so the
     prediction is the conjugacy class, not the matrix).
  P3 demsym.close({C3, T33}) closes to exactly 9 verified elements with
     A-pattern {I: 3, M: 3, M^2: 3}.
Output: runs/demsym_honeycomb_L9.{log,json}
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import (build_lattice, build_circuit,
                                    derive_action, derive_logicals,
                                    record_perm, detector_perm_array,
                                    log)  # noqa: E402

import demsym  # noqa: E402

T0 = time.time()
L = M = 9
R = 9
S_EXTRA = 1
P = 0.002

verts, vid, edges, elook, faces = build_lattice(L, M)
log(f"L={L}: {len(verts)} data qubits, {len(edges)} edges, {len(faces)} faces")

c0, det_meta, det_recs, nrec, aux = build_circuit(
    L, M, R, 0.0, "cx", edges, elook, faces, vid, s_extra=S_EXTRA)
dets0 = c0.compile_detector_sampler(seed=1).sample(256)
g2 = bool(not dets0.any())
log(f"G2 noiseless determinism: {g2} ({len(det_meta)} detectors, {nrec} records)")
assert g2

# L=9 has ~1,618 random record dimensions; the L=6 default of 1,024 shots
# under-samples the row space and reports a spuriously huge "deterministic"
# space (597 fake wraps, caught by P1's assert on the first run). Sample
# comfortably past the random rank.
logicals, dim_det, dim_all = derive_logicals(c0, det_recs, nrec, shots=4096)
nv = aux["nv"]
wraps = [w for w in logicals if int(w[nrec - nv:].sum()) > 0]
log(f"deterministic wraps at S=3R+1 (MY): {len(wraps)} "
    f"(weights {[int(w.sum()) for w in wraps]})")
assert len(wraps) == 2, f"P1 FAILS at L=9: {len(wraps)} wraps"
log("P1 ✓ termination claim generalizes to L=9")

obs_recs = [np.nonzero(w)[0].tolist() for w in wraps]
c, _, _, _, _ = build_circuit(L, M, R, P, "cx", edges, elook, faces, vid,
                              obs_recs=obs_recs, s_extra=S_EXTRA)
d0, o0 = c.compile_detector_sampler(seed=2).sample(64,
                                                   separate_observables=True)
dem = demsym.Dem.from_circuit(c)
log(f"DEM: {dem.num_detectors} detectors, {len(dem.groups)} groups, "
    f"{dem.num_mechanisms} mechanisms")


def T(di, dj):
    def f(v):
        s, i, j = v
        return (s, (i + di) % L, (j + dj) % M)
    return f


def C3(v):  # same 120-degree map as honeycomb_decoder_core.vertex_maps
    s, i, j = v
    if s == "A":
        return ("A", (-i - j) % L, (i + 1) % M)
    return ("B", (-i - j - 1) % L, (i + 1) % M)


ORDER3 = ([[1, 1], [1, 0]], [[0, 1], [1, 1]])
solved = {}
for name, vmap in (("C3", C3), ("T33", T(3, 3))):
    act = derive_action(vmap, edges, elook, faces)
    assert act is not None, f"{name}: not a lattice automorphism"
    rperm = record_perm(vmap, act[0], aux, nrec)
    perm = detector_perm_array(det_recs, rperm)
    assert perm is not None, f"{name}: detector permutation failed"
    e = demsym.solve_action(dem, perm, name=name)
    assert e is not None, f"{name}: NO (A,X) verifies at L=9"
    a_id = bool((e.A == np.eye(2, dtype=np.int64)).all())
    log(f"  {name}: A={'I' if a_id else e.A.tolist()}  |chi|={e.chi_weights()}")
    solved[name] = e

assert (solved["T33"].A == np.eye(2, dtype=np.int64)).all(), "P2 T33 A != I"
assert solved["C3"].A.tolist() in [list(map(list, m)) for m in ORDER3], \
    "P2 C3 A not order-3"
log("P2 ✓ translations A=I; C3 mixes the logicals via an order-3 GL(2,F2) "
    "element at L=9")

grp = demsym.close([solved["C3"], solved["T33"]], dem)
pat = {}
for e in grp:
    key = "I" if (e.A == np.eye(2, dtype=np.int64)).all() else str(e.A.tolist())
log_pat = {}
for e in grp:
    k = str(e.A.tolist())
    log_pat[k] = log_pat.get(k, 0) + 1
log(f"closure: {len(grp)} elements; A-pattern {log_pat}")
assert len(grp) == 9 and sorted(log_pat.values()) == [3, 3, 3], "P3 FAILS"
log("P3 ✓ 9-element closed group verified at L=9")

out = {"L": L, "R": R, "p": P, "n_detectors": dem.num_detectors,
       "n_groups": len(dem.groups), "wrap_weights": [int(w.sum()) for w in wraps],
       "C3_A": solved["C3"].A.tolist(), "T33_A": solved["T33"].A.tolist(),
       "closure_size": len(grp), "A_pattern": log_pat,
       "runtime_s": time.time() - T0}
with open("runs/demsym_honeycomb_L9.json", "w") as f:
    json.dump(out, f, indent=1)
log(f"DONE {out['runtime_s']:.0f}s -> runs/demsym_honeycomb_L9.json")
