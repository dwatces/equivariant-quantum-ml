"""CROSS-VALIDATION: the demsym package vs honeycomb_decoder_core on the
paper's main object — the terminated honeycomb Floquet memory (L=6, R=9,
MY readout, naive cx extraction, p=0.002).

The package re-solves (A, X) from the stim circuit alone (its own DEM
extraction + solver); perms come from the project's record-matching route.

PREDICTIONS LOGGED BEFORE THE RUN (hard rule):
  P1 demsym.solve_action reproduces the logged actions: C3 -> A=[[1,1],[1,0]],
     T22 -> A=I, both verify exactly.
  P2 demsym.close([C3, T22]) closes to EXACTLY the 9-element p3_9 group with
     the logged A-pattern ({I,C3,C3sq} x {I,T22,T44}).
  P3 (uncertain, gauge): chi weights may or may not equal the logged
     (196,170) / (72,168) — X is gauge-fixed only on realizable syndromes;
     equality of A + exact verification is the invariant content.
Output: runs/demsym_honeycomb_check.log
"""
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from honeycomb_decoder_core import (build_lattice, build_circuit, vertex_maps,
                                    derive_action, derive_logicals,
                                    record_perm, detector_perm_array,
                                    log)  # noqa: E402

import demsym  # noqa: E402

T0 = time.time()
L = M = 6
R = 9
S_EXTRA = 1
P = 0.002  # the gated noise level of every logged learning run

verts, vid, edges, elook, faces = build_lattice(L, M)
c0, det_meta, det_recs, nrec, aux = build_circuit(
    L, M, R, 0.0, "cx", edges, elook, faces, vid, s_extra=S_EXTRA)
logicals, _, _ = derive_logicals(c0, det_recs, nrec)
nv = aux["nv"]
wraps = [w for w in logicals if int(w[nrec - nv:].sum()) > 0]
assert len(wraps) == 2
obs_recs = [np.nonzero(w)[0].tolist() for w in wraps]
c, _, _, _, _ = build_circuit(L, M, R, P, "cx", edges, elook, faces, vid,
                              obs_recs=obs_recs, s_extra=S_EXTRA)

dem = demsym.Dem.from_circuit(c)
log(f"package DEM: {dem.num_detectors} detectors, {dem.num_observables} obs, "
    f"{len(dem.groups)} groups, {dem.num_mechanisms} mechanisms")

vmaps = vertex_maps(L, M)
solved = {}
for name in ("C3", "T22"):
    act = derive_action(vmaps[name], edges, elook, faces)
    rperm = record_perm(vmaps[name], act[0], aux, nrec)
    perm = detector_perm_array(det_recs, rperm)
    e = demsym.solve_action(dem, perm, name=name)
    assert e is not None and e.verify(dem), f"{name}: package solve FAILED"
    a_id = bool((e.A == np.eye(2, dtype=np.int64)).all())
    log(f"  {name}: A={'I' if a_id else e.A.tolist()}  |chi|={e.chi_weights()}")
    solved[name] = e

assert solved["C3"].A.tolist() == [[1, 1], [1, 0]], "P1 C3 A mismatch"
assert (solved["T22"].A == np.eye(2, dtype=np.int64)).all(), "P1 T22 A mismatch"
log("P1 ✓ package reproduces the logged (A, X) actions from the circuit alone")

grp = demsym.close([solved["C3"], solved["T22"]], dem)
log(f"closure of {{C3, T22}}: {len(grp)} elements")
assert len(grp) == 9, "P2 closure size mismatch"
a_counts = {}
for e in grp:
    a_counts[str(e.A.tolist())] = a_counts.get(str(e.A.tolist()), 0) + 1
log(f"  A-pattern: {a_counts}")
assert a_counts == {"[[1, 0], [0, 1]]": 3, "[[1, 1], [1, 0]]": 3,
                    "[[0, 1], [1, 1]]": 3}, "P2 A-pattern mismatch"
log("P2 ✓ p3_9 rebuilt from 2 generators by verified closure")
log(f"P3 note (gauge): logged run had C3 |chi|=(196,170), T22 |chi|=(72,168)")
log(f"DONE {time.time() - T0:.0f}s")
