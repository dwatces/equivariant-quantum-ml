# Non-Abelian (Ising) phase — classical groundwork + hardware roadmap

*2026-06-10 (Fable 5). The one named frontier left after the Kitaev topological-order arc
(`runs/KITAEV_TOPOLOGICAL_ORDER.md`). All numbers below read from the completed run
`runs/kitaev_nonabelian.log` / `runs/kitaev_nonabelian.json`; figure
`figures/nonabelian_phase.png`; code `scripts/kitaev_nonabelian.py`.*

## Why this phase is the capstone — and the unification

Kitaev honeycomb + magnetic field opens the **non-Abelian B phase**: the field's leading
effect is a 3-spin term (coefficient κ) that, in the exact Majorana solution, becomes a
**Haldane-like next-nearest-neighbour hopping with Chern number ν = ±1**. Two consequences:

1. **The project's two QPU threads are literally the same physics here.** The Majorana
   sector of Kitaev+field IS a Haldane-type Chern band — the chiral edge the project measured
   on hardware (97% mirror, t2-causal-scaling) and the Kitaev flux/anyon results it measured
   on hardware are the two halves that this phase fuses. The chiral **Majorana** edge is
   "half" a Haldane edge (central charge c = 1/2 vs c = 1) — one chiral branch, not two.
2. **Odd Chern number ⇒ each vortex (vison) binds a Majorana zero mode ⇒ visons become
   ISING ANYONS (σ)** — non-Abelian: σ×σ = 1 + ψ, braiding acts on a degenerate space.
   This is the genuine universal-TQC primitive, beyond the abelian −1 braiding already
   demonstrated on 3 devices.

## Classical results (exact free-Majorana/BdG, fixed flux sector — exact for this model)

Self-checks: torus vs Bloch spectra agree to 3×10⁻¹⁴ (S1); Chern integers are
self-diagnosing for the NNN sign prescription (wrong pattern ⇒ C=0; got ±1) (S2);
bulk-edge correspondence (S3, below). Energies in eigvalsh(iA) units, J=1.

**E1 — phase diagram (gap + Chern over Jz, κ; Nk=60 FHS).**
- κ=0: B phase gapless (Jz=1 gap 0.0000; the 0.064/0.027 at Jz=0.5/1.5 are k-grid
  discretization of the Dirac points, not real gaps); toric phase gapped (Jz=2.5: 1.0,
  Jz=3.0: 2.0); Chern 0 everywhere.
- κ>0: **B phase gaps with C=+1 exactly** (Jz=0.5/1.0/1.5), toric stays C=0; the
  boundary Jz=2 still closes. **κ→−κ flips C to −1.**
- Analytic anchor: at κ=0.1 the gap is 1.0392 = **6√3·κ — Kitaev's closed-form Δ,
  matched to 4 digits.** (At κ=0.2 the BZ minimum moves; gap reads 2.0.)

**E2 — chiral Majorana edge (cylinder Ny=40, 121 kx).**
- In-gap states (each phase vs its own gap): **B phase 88, toric 0** — bulk-edge
  correspondence clean.
- Every in-gap B-phase state is edge-localized (fraction 1.000; median bottom-edge weight
  0.999 vs bulk expectation 0.075).
- **One** chiral branch per edge (c=1/2 signature), velocity at the zero crossing
  **+1.826 for κ=+0.2, −1.826 for κ=−0.2** — chirality set by the field sign, exactly
  the Haldane-thread phenomenology, now in the Majorana sector.

**E3 — Majorana zero modes bound to vortices (L=20 torus; flux-counting asserts exactly
2 vortices; string steps along n1−n2 so interior fluxes cancel).**
- B phase (κ=0.2): lowest mode **E0 = 0.30 (d=2) → 9.0×10⁻² (d=3) → 3.0×10⁻³ (d=4)
  → 2.4×10⁻² (d=5) → 2.4×10⁻² (d=6) → 7.9×10⁻³ (d=8)** while the next state stays
  0.96–1.80 and the no-vortex gap is 2.0 — a mode **2–3 orders below the gap**, with
  ±E0-pair weight 1.85–2.10 (of 2.0) on the two vortex disks = **zero modes localized
  on the vortices**. The non-monotonic decay is the known oscillatory MZM splitting
  (∝ cos(k·d)·e^(−d/ξ)); the log-fit ξ≈4.2 cells is contaminated by the oscillation —
  reported as qualitative (exponentially suppressed with oscillations), not a clean ξ.
- **Toric control (Jz=2.5, κ=0.2, C=0): no zero mode** — E0 ≈ 1.03–1.06 ≈ the bulk gap,
  vortex-disk weight 0.13–0.19. The MZM exists **only** in the C=±1 phase.

**Net: the full non-Abelian triad — ν=±1 Chern band, single chiral Majorana edge, and
vortex-bound Majorana zero modes (the Ising-anyon carrier) — demonstrated exactly, with
controls, on the classical rig. The frontier is now characterized, not just named.**

## ⭐ ADDENDUM (same day, second session): NON-ABELIAN BRAIDING — DONE numerically

`scripts/kitaev_braid_mzm.py`, `runs/kitaev_braid_mzm.{log,json}` (+ `_L26` wide-spacing
re-run). Ivanov's theorem realized on our lattice: 4 vortices (4 MZMs = the Ising fusion
space), vortex motion by continuous bond-sign interpolation, Wilson-line parallel
transport of the 4-dim zero-mode subspace, braid matrices in the vortex-localized
Majorana basis. **L=20 results (all from the completed log):**
- **O1 single exchange:** |M| = swap of the two exchanged modes (score 0.97), spectator
  modes untouched (|M_33|,|M_44| ≈ 1.00); diagonal leak 0.34 = finite-separation
  hybridization (vortices pass within ~1 cell on L=20).
- **O2 commutator of exchanges SHARING a vortex** (gauge-free: u returns exactly):
  **‖M−I‖ = 2.69**, |M| matches the predicted Ivanov 3-cycle at **0.94**.
- **O3 control — commutator of DISJOINT exchanges: ‖M−I‖ = 0.041** (identity).
- Transport quality σ_min ≥ 0.96; subspace–bulk separation held throughout
  (max in-subspace |E| 0.61 during closest approach vs next level ≥ 0.69).
**Non-commuting exchange operations with a clean null control = non-Abelian exchange
statistics demonstrated end-to-end on the project's honeycomb.**
**L=26/GAP=6 wide-spacing re-run (completed, `runs/kitaev_braid_mzm_L26.log`) SHARPENS
it: shared commutator ‖M−I‖ = 2.566 with Ivanov-3-cycle match 0.975 (was 0.942);
disjoint control 0.005 (was 0.041) → a ~500× contrast.** O1 swap 0.974 with diagonal
leak 0.319 ≈ unchanged — the leak is set by the exchange path's CLOSEST APPROACH
(~1 cell in this path geometry), not the base separation; a wider exchange loop is the
knob if a crisper single-exchange matrix is ever needed (not required for the
statistics claims, which are the gauge-free commutator contrasts).

## ⭐ ADDENDUM 2: MAGIC (non-stabilizerness) across the phase diagram — predictions 3/3

`scripts/kitaev_magic.py`, `runs/kitaev_magic.{log,json}` (11 s). Stabilizer Rényi
entropy M2 of the exact ground state vs uniform (1,1,1) field h, n=8 and n=12 tori,
predictions logged pre-run. **All three confirmed:** (a) toric corner h=0 = most
stabilizer-like (0.085/qubit n=8, 0.164 n=12); (b) magic rises steeply with h in the
isotropic/B region (0.17→0.59/qubit by h≈0.3); (c) iso > toric at EVERY h, both sizes
(ratio ≈ 2, size-stable). **Diagnostic discovery: the saturation plateau 0.585/qubit
= log₂(3/2) EXACTLY = the single-qubit SRE of a (1,1,1)-polarized spin** → the plateau
marks the small-size crossover to the trivial polarized phase, so the topological
reading rests on the h ≤ 0.2 rise + the iso-vs-toric contrast, not the plateau. Honest
limits: n=8/12 (densities not size-converged; ratio is), uniform-field ED (not the
exactly-solvable κ form). Story line gained: *the abelian corner we braided on IBM
hardware is exactly the magic-free (classically simulable) corner; magic — the quantum
resource — accumulates precisely where the non-Abelian phase opens.*

## Honest scope

- Textbook physics (Kitaev 2006; Ivanov 2001), exact at the free-Majorana level; the
  3-spin κ term is the perturbative stand-in for a real field (valid deep in the phase;
  the real-field model at finite h is not exactly solvable and its phase diagram is an
  active research area — out of scope here).
- Braiding is at the BdG/zero-mode-transport level (the standard numerical formulation
  of Ising-anyon statistics) in a fixed flux sector; fusion-channel readout in the
  many-body Fock space is the remaining classical extension. On hardware, true
  non-Abelian braiding requires adaptive circuits (see below).
- None of this is novel research; it is **readiness** — the design knowledge to recognize
  and gate a hardware version.

## Hardware roadmap (gated; budget = IBM open plan, rolling 28-day window)

Budget state at writing: 557/600 s consumed; ~419 s frees ≈ 2026-06-30, ~138 s more
≈ 2026-07-05. A kickoff routine is scheduled for June 30 (9 am NZT).

- **Tier 1 (QPU arc 2, June 30, priority 1) — honeycomb Floquet code (Hastings–Haah):**
  not non-Abelian, but the strongest feasible next hardware result: a few rounds of the
  2-qubit-check measurement schedule on a small patch (mid-circuit measurements are
  available on IBM), showing the stabilizer group cycling and a logical observable
  surviving rounds → extends the arc from "topological order exists on hardware" to
  "the honeycomb PROTECTS information on hardware." Sim-gate first; 2q-gate budget guard.
- **Tier 1b (priority 2, if sim-gate passes + budget) — twist defects in the toric code:**
  the most achievable Ising-σ-flavored hardware experiment. Twists (lattice dislocations)
  in the abelian toric code behave as Ising-like anyons and bind Majorana-type modes
  (Bombin 2010) — conceptually reachable at ~10–15 qubits on the existing 8-qubit
  toric-code tooling (`scripts/toric_braiding.py` lineage), unlike true B-phase physics.
  Design + sim-gate during arc-2 prep; submit only if the signature is clean and shallow.
- **Tier 2 (out of reach on the open plan — the long-term marker):** genuine Ising-anyon
  braiding with fusion readout needs adaptive/feed-forward circuits and larger codes
  (done by Google/Quantinuum at scale). Revisit as free-tier hardware matures; also the
  natural home for the analog neutral-atom platforms (see `runs/ANALOG_SUBSTRATES.md`).

**Decision rule for June 30:** verify real budget → Floquet-code sim-gate → if clean,
submit (3-device + DD as usual); twist-defect only with leftover budget and a clean gate.
Never record a result before reading the completed run.
