# Equivariant Quantum Machine Learning on a Honeycomb Lattice

A self-directed research project investigating whether a honeycomb (graphene-like)
quantum substrate's two "gifts" — **symmetry (equivariance)** and **topology
(chirality)** — survive on *real* quantum hardware, and whether lattice symmetry
yields a learnable advantage classically.

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
