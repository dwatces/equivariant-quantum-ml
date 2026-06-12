# demsym

**Exact twisted symmetries of stim detector error models — solved, verified,
and wrapped into PyTorch equivariant decoders.**

A symmetry of a quantum error-correcting experiment is not just a permutation
of detectors. It acts on the *labels* too: logical observables pick up
syndrome-dependent corrections (the logical representative moves), and on some
codes the symmetry *mixes the logical qubits*. `demsym` works with the full
twisted action

```
syndrome:  x  ->  x_g            (x_g[perm[d]] = x[d])
label:     c  ->  A c  ⊕  X·x_g
```

with `A` invertible over GF(2) and `X` a per-observable syndrome-linear
correction. Given any stim circuit and a candidate detector permutation, it
**solves (A, X) directly from the detector error model** and verifies the
result exactly (multiset equality of all error mechanisms — no sampling, no
tolerance). A `None` is a real answer: the permutation is *not* a symmetry of
that DEM.

## Install

```
pip install demsym            # solver only (numpy + stim)
pip install 'demsym[torch]'   # + the PyTorch equivariant wrapper
```

## Sixty seconds: the whole story on 20 detectors

```python
import stim, demsym

c = stim.Circuit.generated(
    "repetition_code:memory", distance=5, rounds=4,
    before_round_data_depolarization=0.04,
    before_measure_flip_probability=0.01)

dem  = demsym.Dem.from_circuit(c)
perm = demsym.perm_from_coords(c, lambda co: (8.0 - co[0],) + co[1:])

elem = demsym.solve_action(dem, perm, name="mirror")
print(elem)   # Element('mirror', nd=20, A=I, |chi|=20)
```

Three things just happened:

1. **The mirror verifies** — it is an exact symmetry of the circuit-level DEM.
2. **The twist is real**: `|chi| != 0`. The logical observable is the last
   data qubit; its mirror image is the first; the difference is a chain of
   checks — a syndrome-linear label correction. Drop it
   (`X = 0`) and verification fails. Naive "invariant" models mis-tie exactly
   here.
3. Change the noise to `after_clifford_depolarization=0.01` and
   `solve_action` returns `None` — **circuit-level extraction noise breaks
   the spatial symmetry**. That is not a bug; it is a structural fact about
   schedules, and finding it is the point of the tool.

Build the group and an exactly equivariant model:

```python
group = demsym.close([elem], dem)          # verified closure, identity included
model = demsym.nn.TwistedEquivariant(
    demsym.nn.mlp_backbone(dem.num_detectors, dem.num_observables), group)
# model(x): (batch, nd) -> (batch, 2^k) logits over joint observable classes
demsym.nn.equivariance_probe(model, group, sampled_shots)   # ~1e-6
```

## What this machinery found (the reason it exists)

Applied to memory experiments at circuit level (all numbers from logged runs;
preprint in preparation):

- **Surface code: obstruction.** No detector permutation implementing the
  code's spatial symmetries verifies on circuit-level DEMs across 999
  coordinate-map candidates and 1,728 extraction schedules (including
  co-designed ones). Mechanism: hook errors de-symmetrize the schedule; a
  mirror-axis ancilla cannot equal its own east-west swap.
- **Honeycomb Floquet code: exactness.** With completely naive uniform
  extraction, the circuit-level DEM of a terminated Hastings–Haah honeycomb
  memory carries an **exact p3 space group** (translations + C3, verified on
  all 4,584 mechanism groups at L=6). Trivalence does the work: weight-2
  checks make 1-bit schedules automatically covariant.
- **C3 mixes the logical qubits**: `A = [[1,1],[1,0]]`, order 3 in GL(2,F2) —
  solved blind from the DEM. The correct decoder is therefore a 4-class
  twisted-equivariant model; scalar per-observable equivariance cannot
  express this.
- **Equivariance as a learnability switch.** Matched 63k-parameter MLPs on
  the honeycomb task (p=0.002, MWPM joint accuracy 0.947, majority 0.291):

  | train samples | plain | 9-element twisted tying |
  |---|---|---|
  | 250 | 0.258 | **0.363** |
  | 1,000 | 0.267 | **0.404** |
  | 16,000 | 0.274 | **0.600** |
  | 64,000 | 0.290 | **0.744** |
  | 256,000 | 0.391 | **0.806** |

  The plain network at 64k samples — and at 22.5× the parameters (1.4M) — sits
  at the majority baseline; the tied model at n=1,000 beats plain at
  n=256,000 (≥256× sample efficiency). We claim **no** superiority over
  matched-graph MWPM anywhere; the tied model remains ~14 pp below it on this
  task.

## Validation

The solver is validated against the analytic symmetry action of
**Egorov, Bondesan & Welling, "The END: An Equivariant Neural Decoder"
(arXiv:2304.07362)** on their setting (toric code, code capacity): the blind
DEM solve recovers their action — including the 90°-rotation logical swap —
**uniquely among all 24 observable permutations**, for every element. That
bridge ships as a test (`tests/test_toric_end.py`).

## Scope and honest limits

- You supply candidate permutations (from detector coordinates via
  `perm_from_coords`, or any construction you trust); demsym decides, exactly.
  Automorphism *discovery* is out of scope for v0.1 — see AutDEC
  (arXiv:2503.01738) for DEM-automorphism search via graph tools.
- `X` is gauge-fixed only modulo directions invisible to every mechanism:
  equivariance is exact on **realizable** syndromes. Probe with sampled
  shots, not random bit-vectors.
- Auto-enumeration of `A` candidates covers k ≤ 3 observables (GL(k,F2) =
  1, 6, 168); beyond that, pass `A_candidates` (e.g. the observable
  permutation matrices, as the toric validation does for k=4).
- The wrapper costs |G| backbone passes per forward: matched parameters, not
  matched FLOPs.
- Twisted equivariant decoding itself is prior art (The END, above). What
  demsym adds is the solve-from-DEM route — the one that still exists at
  circuit level, where analytic lattice-path constructions don't apply —
  plus the verified wrapper.

## Citing

If demsym contributes to your work, cite the accompanying paper
(*Circuit-level equivariance is code-dependent: obstruction in the surface
code, exactness in the honeycomb Floquet code, and consequences for neural
decoders*, D. Olliver, 2026 — arXiv ID on announcement; `CITATION.cff`
ships with the repository).

## License

MIT. Author: Daniel Olliver (dwatces@gmail.com).
