# Equivariant Quantum Machine Learning on a Honeycomb Lattice

A self-directed research program on what a honeycomb substrate's two "gifts" —
**symmetry (equivariance)** and **topology** — are actually worth, measured
honestly: on real quantum hardware, and now at the level of quantum
error-correcting codes and the neural decoders that learn them.

---

## 📄 The paper (2026): circuit-level equivariance is code-dependent

**[`preprint/main.pdf`](preprint/main.pdf)** — *Circuit-level equivariance is
code-dependent: obstruction in the surface code, exactness in the honeycomb
Floquet code, and consequences for neural decoders.* (arXiv submission in
progress; LaTeX source and figure pipeline included.)

The short version: whether a code hands its lattice symmetry to a machine-learned
decoder at circuit level is a property of the **code**. For the surface code it
provably doesn't survive compilation (999 spacetime maps + 1,728 co-designed
schedules, zero verified, with a one-paragraph proof of why no schedule can work).
For the honeycomb Floquet code an exact p3 space group survives the most naive
compilation — and its C3 rotation **mixes the two logical qubits** (GL(2,F₂)),
so the right decoder is jointly equivariant over both. Built exactly into a
network, the symmetry flips the honeycomb decoding task from unlearnable to
learned (≥256× sample-efficiency separation; noise-rate sweep included).

Every number traces to a logged run in [`runs/`](runs/) — re-checkable via
[`scripts/preprint_audit.py`](scripts/preprint_audit.py) (116 checks, 116 pass).

## 📦 `pip install demsym`

The paper's solver, released as a package: given any [stim](https://github.com/quantumlib/Stim)
circuit, find exact twisted symmetries of its detector error model — detector
permutation + GL(k,F₂) logical action + syndrome-dependent twist — and wrap any
PyTorch backbone into an exactly equivariant decoder. Validated by blind
recovery of The END's analytic toric action (ships as a test). See
[`demsym/README.md`](demsym/README.md).

## 🐝 hexlife — does the lattice a pattern grows on matter? (new)

**[`hexlife/`](hexlife/)** — matched-density Bravais-lattice Lenia with
paired controls, built to test the substrate question for
self-organization rather than learning. One implementation, square and
hexagonal floors at exactly 1 cell/unit-area, identical rule, identical
seeds; all metrics lattice-native. Results
([`hexlife/FINDINGS.md`](hexlife/FINDINGS.md)): substrate-indifference
at dilute scales (with a five-artifact hunt log showing how easily the
opposite gets "discovered"), then — below ~3 lattice constants — the
substrate takes ownership: symmetry selection and orientation locking
to the crystal axes with commensuration sign flips
(Novaco–McTague-class physics in a continuous CA; 11,946 patterns,
replicated across GPU/CPU hardware to the 2nd–3rd decimal), a mapped
commensurate–incommensurate crossover, and a weak but box-invariant
hexagonal-floor lock that outlives the square floor's by ~3× in
spacing — because emergent patterns are themselves hexatic, and only
the hex floor speaks their symmetry.

## 🎮 Play against the trained decoder

The 63k-parameter honeycomb decoder from the paper runs **live in your
browser** (float16 weights, gate-tested against the python model):
**[the error-hunting game](https://portfolio-6f211.web.app/anyons/decoder/)** —
guess quantum-memory errors yourself, race the net against minimum-weight
matching, flip the symmetry switch, drag the training-data slider.

---

## Topological order on real quantum hardware — the Kitaev arc (2026)

Honeycomb-lattice topological order, measured on IBM superconducting processors —
every experiment simulator-gated before submission, validated across **three
independent backends** (ibm_marrakesh, ibm_kingston, ibm_fez) with dynamical
decoupling and bootstrap error bars.

| Result | Hardware numbers (cross-device mean ± std) |
|---|---|
| Z₂ flux conservation (Kitaev dynamics vs field control) | ⟨W_p⟩ +0.642 ± 0.016 vs +0.115 ± 0.030 |
| Localized anyon (vison) creation, neighbour untouched | W_p1 −0.905 ± 0.010, W_p2 +0.924 ± 0.015 |
| Anyon braiding — the −1 mutual statistics | ⟨Z̄⟩ vacuum +0.845 ± 0.030 → braid −0.843 ± 0.035, control +0.845 ± 0.033 |

Figure: `figures/kitaev_topological_order.png`. IBM job IDs in
`runs/KITAEV_TOPOLOGICAL_ORDER.md`; reproducible on the free open plan.

**Classical capstone — the non-Abelian (Ising) phase** (`scripts/kitaev_nonabelian.py`,
`scripts/kitaev_braid_mzm.py`): exact free-Majorana treatment of Kitaev + field.
Chern number ±1 (sign flips with the field; analytic gap 6√3κ reproduced to 4
digits), a single chiral Majorana edge branch, Majorana zero modes bound to
vortices (absent in the Chern-0 control), and **numerically braided Majorana
modes**: exchange commutators sharing a vortex are far from identity
(‖M−I‖ = 2.69, matching Ivanov's 3-cycle) while disjoint exchanges commute
(0.04) — non-Abelian exchange statistics, end-to-end on one lattice. A
measurement-programmed magic-state primitive on a 6-qubit hexagon cluster ran
on hardware (`scripts/ibm_mbqc_hex.py`, witness 0.4150 vs ideal log₂(4/3)).

The whole-project narrative — six years, the falsified parts included — is in
[`HEX_PROJECT_SYNTHESIS.md`](HEX_PROJECT_SYNTHESIS.md) and the public essay at
[danielolliver.com/essay](https://portfolio-6f211.web.app/essay/).

---

## Earlier arc: equivariant QML on hardware (below)

## What's here
- **`PAPER_DRAFT.md`** — full write-up: methods, results, error bars, limitations. *Preprint draft — not peer-reviewed.*
- **`figures/`** — the three headline figures.
- **`scripts/`** — the experiments that produced the results (Qiskit + IBM Quantum Runtime).

## Headline results (exact numbers + error bars in `PAPER_DRAFT.md`)
1. **Classical equivariance = a sample-efficiency gift.** An exact lattice-equivariant
   network beats a plain one at scarce data, converging to a tie with abundant data (`fig1`).
2. **Both gifts coexist on real hardware.** Equivariant + topological (Haldane) circuits,
   run on **three independent IBM quantum devices**, error-mitigated (dynamical decoupling
   + Pauli/readout twirling) with bootstrap error bars, show the **C3 lattice symmetry
   preserved** (vs a cleanly-detected broken-symmetry control) **and the chiral topological
   signature present** (vs ~0 for a trivial control) — across single-particle, two-particle
   (entangled), and size-scaled (6→12-qubit) regimes (`fig2`).
3. **Causal attribution.** The chirality scales monotonically with the topological term,
   on hardware (`fig3`).

## Honest scope
This shows known symmetry/topology **signatures coexist and survive hardware noise** — it
is *not* a claim of computational quantum advantage (that's scale-gated). Small scale
(6–12 qubits); a single self-directed research effort.

## Stack
Qiskit · IBM Quantum Runtime · NumPy · Matplotlib
