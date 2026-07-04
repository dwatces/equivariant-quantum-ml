# hexlife — matched-density Bravais-lattice Lenia with paired controls

Does the lattice a pattern grows on matter? A rig built to answer that
question honestly for Lenia-class continuous cellular automata:

- **One implementation, any Bravais lattice.** A substrate is a 2×2
  basis matrix; square and hexagonal floors run at exactly **1 cell per
  unit area** (density-matched), with the identical continuous kernel
  and growth rule. FFT convolution is exact on any Bravais lattice.
- **Paired design throughout.** Identical (μ,σ) parameter grids and
  identical index-space initial soups per seed across substrates.
- **Lattice-native measurement.** All metrics live on true cell
  positions — never through a square-pixel raster. The artifact log in
  `FINDINGS.md` shows why (five large "significant effects" in one
  night, all measurement artifacts, each killed by a paired control).

## Headline results (full write-up: [FINDINGS.md](FINDINGS.md))

1. **At pattern spacings ≳ 8 lattice constants: substrate-indifference**
   (existence, abundance, packing order — clean nulls with matched
   power). Emergent spots hex-pack on the square floor too: hexagonal
   order is what the physics does, on any floor.
2. **Below ~3 lattice constants the substrate owns the pattern**:
   symmetry selection (hex→ψ6, square→ψ4; paired sign tests
   near-unanimous over 11,946 patterns) and orientation **locking to
   the crystal axes** (|lock| up to 0.97) whose sign flips with the
   spacing/lattice-constant ratio — Novaco–McTague-class commensurate
   physics emerging in a continuous CA. Replicated across disjoint
   seeds and different hardware (GPU vs CPU) to the 2nd–3rd decimal.
3. **The crossover (3–8a)**: square-floor locking rings through violent
   commensuration sign flips (−0.56 → +0.95 → +0.55) and dies by
   ~6.5a; the hexagonal floor keeps a weak (+0.2), box-size-invariant
   on-axis lock to at least 8.7a — because the emergent order is
   itself hexatic, and only the hex floor shares its symmetry.
4. **A lattice-scale morphology channel** (ring/micro-texture patterns
   pinned at 2–3a) that the square floor sustains 2–3× more than hex —
   with the obvious linear-response mechanism already falsified
   (`kernel_spectrum.py`).

## Reproduce

```bash
python3 lenia.py --selftest                 # kernel/FFT/density gates
python3 experiments.py all                  # E1/E2 battery
python3 sweep_commensurate.py               # the resolved sweep (R 2-4)
python3 sweep_crossover.py --mode crossover # the 3-8a panel
python3 sweep_crossover.py --mode crossover --analyze
python3 analyze_crossover_modes.py          # wide/tight mode split
python3 probe_fragmentation.py              # the artifact-#6 probe
python3 kernel_spectrum.py                  # mechanism falsification
python3 fig_crossover.py                    # the locking figure
```

Deps: `torch numpy scipy matplotlib` (CUDA optional; every sweep also
runs on CPU). Raw run data (JSONL, one line per record) and summary
JSONs are in `runs/`; figures in `figures/`.

All predictions were pre-registered in the script docstrings before
their runs; falsified predictions are reported as falsified in
`FINDINGS.md`.
