# Equivariant quantum ML on a honeycomb lattice

Research code, logged runs and a preprint from a self-directed study of what a honeycomb lattice's two "gifts", **symmetry (equivariance)** and **topology**, are actually worth: in quantum error-correcting codes and the neural decoders that learn them, on real IBM quantum hardware, and in self-organising cellular automata.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20715320.svg)](https://doi.org/10.5281/zenodo.20715320)

**Preprint:** [doi.org/10.5281/zenodo.20715320](https://doi.org/10.5281/zenodo.20715320) · **Solver:** [`pip install demsym`](https://pypi.org/project/demsym/) · **Interactive demos:** [anyon braiding](https://danielolliver.com/anyons/), [decoder game](https://danielolliver.com/anyons/decoder/)

## The decoder paper (2026)

**Circuit-level equivariance is code-dependent: obstruction in the surface code, exactness in the honeycomb Floquet code, and consequences for neural decoders.** Daniel Olliver, preprint, June 2026. DOI [10.5281/zenodo.20715320](https://doi.org/10.5281/zenodo.20715320). The PDF, LaTeX source and figures are in [`preprint/`](preprint/) ([`main.pdf`](preprint/main.pdf)); the figures are built by [`scripts/make_preprint_figures.py`](scripts/make_preprint_figures.py).

The short version: whether a code hands its lattice symmetry to a machine-learned decoder at circuit level is a property of the **code**.

- **Surface code: obstructed.** Exact twisted symmetry does not survive compilation. 999 spacetime maps and 1,728 co-designed extraction schedules give zero verified symmetries, and a one-paragraph proof shows why no schedule can work: a mirror-axis ancilla would need a CNOT order equal to its own reflection.
- **Honeycomb Floquet code: exact.** An exact p3 space group survives the most naive compilation. Its C3 rotation **mixes the two logical qubits** (GL(2,F₂)), so the right decoder is jointly equivariant over both.
- **Equivariance as a learnability switch.** Built exactly into a network, the symmetry flips the honeycomb decoding task from unlearnable to learned. A matched-parameter unconstrained network stays at the majority-class baseline through 64,000 training samples, while the exactly tied model reaches 0.806 ± 0.003 at 256,000; the tied model at 1,000 samples beats the unconstrained one at 256,000 (a ≥256× sample-efficiency separation). A noise-rate sweep is included.
- **Scope.** No decoder in the paper beats minimum-weight perfect matching (MWPM), and the paper says so.

Every number traces to a logged run in [`runs/`](runs/) and can be re-checked with [`scripts/preprint_audit.py`](scripts/preprint_audit.py) (116 checks, 116 pass). See [Quick start](#quick-start) for the exact commands.

## demsym

The paper's solver, released as a package: [`demsym` on PyPI](https://pypi.org/project/demsym/). Given a [stim](https://github.com/quantumlib/Stim) circuit and a candidate detector permutation, it solves the full twisted symmetry action (detector permutation, GL(k,F₂) logical action and a syndrome-dependent twist) directly from the detector error model, verifies it exactly against every error mechanism, and wraps any PyTorch backbone into an exactly equivariant decoder. You supply the candidate permutations (for example from detector coordinates); automatic symmetry discovery is out of scope.

It is validated by blind recovery of the analytic toric-code action from "The END" (Egorov, Bondesan and Welling, arXiv:2304.07362), which ships as a test (`demsym/tests/test_toric_end.py`). Full documentation, scope and limits: [`demsym/README.md`](demsym/README.md).

```bash
pip install demsym            # solver only (numpy + stim)
pip install 'demsym[torch]'   # adds the PyTorch equivariant wrapper
```

## Quick start

### Find a symmetry with demsym

```python
import stim, demsym

c = stim.Circuit.generated(
    "repetition_code:memory", distance=5, rounds=4,
    before_round_data_depolarization=0.04,
    before_measure_flip_probability=0.01)

dem  = demsym.Dem.from_circuit(c)
perm = demsym.perm_from_coords(c, lambda co: (8.0 - co[0],) + co[1:])
print(demsym.solve_action(dem, perm, name="mirror"))
# Element('mirror', nd=20, A=I, |chi|=[20])
```

The mirror verifies as an exact symmetry of the circuit-level detector error model, and its twist is non-zero. [`demsym/README.md`](demsym/README.md) explains what that means and how to build an equivariant model from it.

### Re-check the paper's numbers

```bash
git clone https://github.com/dwatces/equivariant-quantum-ml.git
cd equivariant-quantum-ml
pip install demsym
python3 scripts/demsym_honeycomb_check.py | tee runs/demsym_honeycomb_check.log
python3 scripts/preprint_audit.py
```

The audit compares the paper's numeric claims with the logged JSON in `runs/` and should end with `116 PASS, 0 FAIL`. One check reads `runs/demsym_honeycomb_check.log`, and `.log` files are not committed (see `.gitignore`), so the `demsym_honeycomb_check.py` command regenerates it first. It runs in seconds on a CPU.

### Re-run the experiments

There is no requirements file. Install what each group of scripts imports:

| Scripts | Packages |
|---|---|
| Decoder and code-symmetry scripts | numpy, stim, pymatching, torch, demsym |
| Kitaev and non-Abelian (classical) | numpy, scipy |
| IBM hardware (`ibm_*.py`) | qiskit, qiskit-ibm-runtime, requests |
| Figures | matplotlib |
| `hexlife/` | torch, numpy, scipy, matplotlib |
| `honeycomb_cnn_modal.py` | modal |

Run everything from the repository root: the scripts add `scripts/` to the import path and write to `runs/` by relative path. Re-running a script overwrites its committed record in `runs/`.

| Entry point | What it does | Output |
|---|---|---|
| `scripts/toric_end_bridge.py` | Blind DEM solve against The END's analytic toric action | `runs/toric_end_bridge.json` |
| `scripts/honeycomb_code_symmetry.py` | Exact circuit-level symmetry test of the honeycomb Floquet code (`--L --M --R --p`) | `runs/honeycomb_code_symmetry.json` |
| `scripts/honeycomb_decoder_run.py` | Full honeycomb decoder sweep (plain, c3tw, p3_9 arms; 5 seeds; uses CUDA if available) | `runs/honeycomb_decoder.json` |
| `scripts/honeycomb_decoder_bign.py` | Big-n extension (64k and 256k samples) | `runs/honeycomb_decoder_bign.json` |
| `scripts/honeycomb_decoder_scale.py` | Capacity ablation (wider plain networks) | `runs/honeycomb_decoder_scale.json` |
| `scripts/honeycomb_decoder_psweep.py` | Noise-rate sweep (p = 0.001, 0.002, 0.004) | `runs/honeycomb_decoder_psweep.json` |
| `scripts/demsym_honeycomb_L9.py` | Second-size structure check at L=9 via demsym | `runs/demsym_honeycomb_L9.json` |
| `scripts/equiv_decoder.py`, `equiv_decoder_d5.py`, `equiv_decoder_d5b.py` | Surface-code equivariant decoders (d=3, d=5/7, twisted and Klein-4 arms) | `runs/equiv_decoder*.json` |
| `modal run scripts/honeycomb_cnn_modal.py` | CNN × p3_9 sweep on Modal GPUs | `runs/honeycomb_decoder_cnn.json` |
| `scripts/make_preprint_figures.py` | Preprint figures, read from the run JSONs | `preprint/figures/` |

**IBM hardware scripts.** Submitting jobs needs an IBM Quantum account saved for qiskit-ibm-runtime (the code calls `QiskitRuntimeService()` with no arguments; `ibm_mbqc_hex.py` connects even for its dry run). `ibm_equivariant_topological.py`, `ibm_kitaev_flux.py`, `ibm_kitaev_localize.py`, `ibm_toric_braiding.py` and `ibm_mbqc_hex.py` run a local simulator gate first and only submit with `--submit`. `ibm_eqtopo_2particle.py` is simulation only (its `--submit` path prints a placeholder; `ibm_eqtopo_2p_robust.py` does the hardware run). `ibm_eqtopo_scale.py` runs its own simulator gate and then submits without asking. The other multi-device scripts (`ibm_eqtopo_robust.py`, `ibm_eqtopo_2p_robust.py`, `ibm_eqtopo_t2sweep.py`, `ibm_shore_up.py`) submit as soon as they run.

**Known gaps in this snapshot.** `circuit_symmetry_mine.py` and `schedule_codesign.py` import a module (`equiv_cnn_core`) that is not in the repository; their logged outputs are in `runs/`. `make_figures.py` and `make_kitaev_figures.py` read result JSONs (mostly IBM hardware results) that are not committed; the figures they produced are in `figures/`.

## hexlife: does the lattice a pattern grows on matter?

[`hexlife/`](hexlife/) takes the substrate question from learning to self-organisation: matched-density Bravais-lattice Lenia (a continuous cellular automaton) with paired controls. One implementation, square and hexagonal floors at exactly 1 cell per unit area, identical rule, identical seeds, and all metrics lattice-native. Full write-up: [`hexlife/FINDINGS.md`](hexlife/FINDINGS.md). Rig details: [`hexlife/README.md`](hexlife/README.md).

1. **Dilute scales (spacing ≳ 8 lattice constants): substrate-indifference** in existence, abundance and packing order, with a five-artifact hunt log showing how easily the opposite gets "discovered".
2. **Below ~3 lattice constants the substrate takes ownership:** symmetry selection and orientation locking to the crystal axes, with commensuration sign flips. This is Novaco-McTague-class physics in a continuous CA, over 11,946 patterns, replicated across GPU and CPU hardware to the 2nd-3rd decimal.
3. **A mapped commensurate-incommensurate crossover (3-8a):** square-floor locking rings through sign flips (−0.56 → +0.95 → +0.55) and dies by ~6.5a. The hexagonal floor keeps a weak (+0.2) but box-invariant lock that outlives the square floor's by ~3× in spacing, because emergent patterns are themselves hexatic and only the hex floor speaks their symmetry.
4. **A lattice-scale morphology channel** (ring and micro-texture patterns pinned at 2-3a) that the square floor sustains 2-3× more than hex. The obvious linear-response mechanism is tested and falsified (`kernel_spectrum.py`).

All predictions were pre-registered in the script docstrings before their runs, and falsified predictions are reported as falsified.

```bash
python3 hexlife/lenia.py --selftest                   # kernel, FFT and density gates
python3 hexlife/experiments.py all                    # E1/E2 battery
python3 hexlife/sweep_commensurate.py                 # the resolved sweep (R 2-4)
python3 hexlife/sweep_crossover.py --mode crossover   # the 3-8a panel
python3 hexlife/sweep_crossover.py --mode crossover --analyze
python3 hexlife/analyze_crossover_modes.py            # wide/tight mode split
python3 hexlife/kernel_spectrum.py                    # mechanism falsification
python3 hexlife/fig_crossover.py                      # the locking figure
```

CUDA is optional; every sweep also runs on CPU. The hexlife scripts read and write `runs/` and `figures/` at the repository root, while the committed hexlife records (JSONL, one line per record, plus summary JSONs) are in `hexlife/runs/` and `hexlife/figures/`. To run the analysis-only steps on the committed data, copy them up first with `cp hexlife/runs/* runs/`.

## Interactive demos

- [**Anyon braiding**](https://danielolliver.com/anyons/): braid anyons in your browser.
- [**The error-hunting game**](https://danielolliver.com/anyons/decoder/): the paper's 63k-parameter honeycomb decoder running live in the browser (float16 weights, gate-tested against the Python model). Guess quantum-memory errors yourself, race the network against minimum-weight matching, flip the symmetry switch and drag the training-data slider.

## Topological order on real quantum hardware: the Kitaev arc (2026)

Honeycomb-lattice topological order, measured on IBM superconducting processors. Every experiment was simulator-gated before submission and validated across **three independent backends** (ibm_marrakesh, ibm_kingston, ibm_fez) with dynamical decoupling and bootstrap error bars.

| Result | Hardware numbers (cross-device mean ± std) |
|---|---|
| Z₂ flux conservation (Kitaev dynamics vs field control) | ⟨W_p⟩ +0.642 ± 0.016 vs +0.115 ± 0.030 |
| Localised anyon (vison) creation, neighbour untouched | W_p1 −0.905 ± 0.010, W_p2 +0.924 ± 0.015 |
| Anyon braiding: the −1 mutual statistics | ⟨Z̄⟩ vacuum +0.845 ± 0.030 → braid −0.843 ± 0.035, control +0.845 ± 0.033 |

Figure: [`figures/kitaev_topological_order.png`](figures/kitaev_topological_order.png). IBM job IDs and per-device numbers are in [`runs/KITAEV_TOPOLOGICAL_ORDER.md`](runs/KITAEV_TOPOLOGICAL_ORDER.md).

**Classical capstone: the non-Abelian (Ising) phase** (`scripts/kitaev_nonabelian.py`, `scripts/kitaev_braid_mzm.py`). An exact free-Majorana treatment of Kitaev plus field gives Chern number ±1 (the sign flips with the field; the analytic gap 6√3κ is reproduced to 4 digits), a single chiral Majorana edge branch, and Majorana zero modes bound to vortices (absent in the Chern-0 control). It also gives **numerically braided Majorana modes**: exchange commutators sharing a vortex are far from identity (‖M−I‖ = 2.69, matching Ivanov's 3-cycle) while disjoint exchanges commute (0.04). That is non-Abelian exchange statistics, end to end on one lattice. Figure: [`figures/nonabelian_phase.png`](figures/nonabelian_phase.png); notes in [`runs/NONABELIAN_ROADMAP.md`](runs/NONABELIAN_ROADMAP.md).

A measurement-programmed magic-state primitive on a 6-qubit hexagon cluster also ran on hardware (`scripts/ibm_mbqc_hex.py`, witness 0.4150 vs ideal log₂(4/3)).

## Earlier work: equivariant QML on hardware

- [`PAPER_DRAFT.md`](PAPER_DRAFT.md): full write-up with methods, results, error bars and limitations. A preprint draft, not peer reviewed.
- [`figures/`](figures/): the three headline figures (`fig1_*` to `fig3_*`).
- [`scripts/`](scripts/): the experiments that produced the results (Qiskit and IBM Quantum Runtime).

Headline results (exact numbers and error bars in `PAPER_DRAFT.md`):

1. **Classical equivariance is a sample-efficiency gift.** An exact lattice-equivariant network beats a plain one at scarce data and converges to a tie with abundant data (`fig1`).
2. **Both gifts coexist on real hardware.** Equivariant plus topological (Haldane) circuits, run on **three independent IBM quantum devices**, error-mitigated (dynamical decoupling plus Pauli and readout twirling) with bootstrap error bars, show the **C3 lattice symmetry preserved** (against a cleanly detected broken-symmetry control) **and the chiral topological signature present** (against ~0 for a trivial control). This holds across single-particle, two-particle (entangled) and size-scaled (6 to 12 qubit) regimes (`fig2`). For the six-site ring: equivariance error E_C3 = 0.031 ± 0.007 vs 0.696 ± 0.014 for the broken control, and directional asymmetry +0.397 ± 0.007 vs +0.000 ± 0.003 for the trivial control.
3. **Causal attribution.** On hardware, the chirality scales monotonically with the topological term (`fig3`).

**Honest scope.** This shows that known symmetry and topology **signatures coexist and survive hardware noise**. It is *not* a claim of computational quantum advantage, which is gated by scale. Small scale (6 to 12 qubits), and a single self-directed research effort.

## The whole project

The whole-project narrative, six years with the falsified parts included, is in [`HEX_PROJECT_SYNTHESIS.md`](HEX_PROJECT_SYNTHESIS.md) and the public essay at [danielolliver.com/essay](https://danielolliver.com/essay/).

## Repository layout

```
preprint/                 decoder paper: main.pdf, main.tex, figures/
demsym/                   the solver package (published on PyPI), with tests
scripts/                  experiment, figure and audit scripts
runs/                     logged results (JSON) and findings notes behind every number
figures/                  figures for the hardware and Kitaev results
hexlife/                  Bravais-lattice Lenia study: code, runs, figures, FINDINGS.md
PAPER_DRAFT.md            earlier hardware write-up (draft)
HEX_PROJECT_SYNTHESIS.md  whole-project narrative
CITATION.cff              citation metadata
```

## Citing

If this work or demsym contributes to your research, please cite the preprint:

```bibtex
@misc{olliver2026circuit,
  author    = {Olliver, Daniel},
  title     = {Circuit-level equivariance is code-dependent: obstruction in the
               surface code, exactness in the honeycomb Floquet code, and
               consequences for neural decoders},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Preprint},
  doi       = {10.5281/zenodo.20715320},
  url       = {https://doi.org/10.5281/zenodo.20715320}
}
```

[`CITATION.cff`](CITATION.cff) names the same paper as its preferred citation.

## Licence

The demsym package is MIT-licensed ([`demsym/LICENSE`](demsym/LICENSE)). The preprint on Zenodo is released under CC BY 4.0.

## Author

Daniel Olliver, [danielolliver.com](https://danielolliver.com/)
