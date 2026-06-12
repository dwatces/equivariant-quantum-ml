# The honeycomb Floquet code has exact circuit-level geometric symmetry — the surface code provably doesn't (2026-06-11 evening)

*`scripts/honeycomb_code_symmetry.py`, `runs/honeycomb_code_symmetry.{log,json}` +
`_L6.log`. All numbers from completed runs. stim 1.16, from-scratch Hastings–Haah
honeycomb code on a torus (construction self-checked: S1 lattice/coloring asserts,
G2 = 256-shot noiseless determinism of every detector, in all modes and sizes).*

## Why this experiment exists (the two results that collided today)

1. **Surface code (yesterday + today):** exact circuit-level twisted symmetry is
   ROBUSTLY OBSTRUCTED — 999-affine-map null (`circuit_symmetry_mine.py`) plus a
   1,728-schedule constructive search (`schedule_codesign.py`, families A/B/C) with
   **zero verified schedules**. Identified obstruction: 4-step CNOT extraction hook
   errors; an ancilla on a mirror axis would need a CNOT order equal to its own
   E↔W swap, which no permutation can be.
2. **Honeycomb (Hastings–Haah) Floquet code:** checks are WEIGHT-2 *because the
   lattice is trivalent* — the project's oldest hexagon-only theorem ("square can't
   3-color"; coordination 3 ⇒ one edge per color per vertex ⇒ 2-body checks, one
   color per sub-round). Weight-2 checks mean: (a) extraction = 2 gates per check ⇒
   schedule freedom is ONE BIT per edge (which endpoint first), (b) hook errors are
   weight ≤ 1 on data.

**Conjecture tested:** the symmetry class that is provably unreachable for the
surface code's extraction is achievable — possibly for free — on the hexagon-native
code.

## Setup

Torus, L×M cells (18 → 72 data qubits), faces 3-colored c=(i−j) mod 3, edge color =
the color of neither adjacent face; sub-round s measures color-(s mod 3) edges with
Pauli (X,Y,Z)[s mod 3]; init |+⟩^n; detectors = face inference windows (6 checks)
compared across consecutive periods (12 records). p = 0.008 circuit-level
(DEPOLARIZE1 on data per sub-round; per-check ancilla prep/measure flips +
DEPOLARIZE2 after each entangling gate in cx modes; MPP flip in mpp mode).
Three extraction modes: **mpp** (direct 2-body measurement), **cx-naive** (ancilla +
2 controlled gates, uniform "A-sublattice endpoint first" — no engineering),
**cx-sym** (endpoint orders chosen orbit-equivariantly; turned out unnecessary).
Verifier: exact multiset equality of DEM mechanism groups under the candidate
detector permutation — same standard as the surface-code null.

## The structural discovery en route: the code is a SPACETIME crystal

The period-3 color schedule splits the lattice point group by its color action:

- **Color-preserving elements** (translations T(1,1),T(2,2); **C3 about any face
  center** — verified color-preserving: rotation by 120° maps (i,j)→(−i−j, i),
  and (−2i−j) ≡ (i−j) mod 3): pure spatial symmetries of the scheduled circuit.
- **Color-cycling elements** (C3 about a vertex; translations like T(1,0)): only
  symmetries when composed with a time shift of one sub-round — **spacetime
  screws**, the Floquet-code analogue of a screw axis.
- **Color-transposing elements** (C2 about faces/edges, mirrors): a transposition
  reverses the cyclic color order, so these pair with **time reversal**. Classified,
  not tested (would need time-symmetric noise placement). This is a property of the
  schedule's temporal chirality, not a defect of the code.

## RESULTS (L=6, R=9 periods, 276 detectors; all classes also pass at L=3)

| element | class | mpp | cx-naive | cx-sym |
|---|---|---|---|---|
| T(1,1) | spatial | **EXACT, full DEM** (3,288 groups) | **EXACT** (4,020) | **EXACT** (4,020) |
| T(2,2) | spatial | **EXACT, full DEM** | **EXACT** | **EXACT** |
| C3 (face) | spatial | **EXACT, full DEM** | **EXACT** | **EXACT** |
| C3 (vertex) | screw dt=+1 | **EXACT, time-bulk** (1,152) | **EXACT, time-bulk** (1,404) | **EXACT, time-bulk** (1,404) |
| C2 (face/edge) | time-reversal-paired | not tested | not tested | not tested |

- **The full p3 rotation space group (translations + C3 about every face) is an
  EXACT symmetry of the circuit-level detector error model — even with naive
  ancilla-compiled extraction.** No schedule co-design needed: the one-bit
  "A-endpoint first" rule is automatically covariant under every sublattice-
  preserving element, and hooks are weight-1. The geometry does the co-design.
- Screw exactness is a time-bulk statement: the CX init transient produces 12
  single-detector mechanism groups pinned at absolute s=7 (diagnosed by moving the
  bulk window: mismatches did NOT move with it) — a genuine time boundary, the
  temporal analogue of spatial boundaries, excluded by bulk margin 6. MPP mode's
  shorter transient is exact already at margin 3.
- cx-sym equals cx-naive everywhere — recorded because *that it's unnecessary* is
  part of the finding.

## The contrast (the headline)

| | square-native (surface code) | hexagon-native (honeycomb code) |
|---|---|---|
| point-group circuit symmetry | **none** — 999-map + 1,728-schedule null; mirror-axis obstruction proven | **exact p3** (C3 + translations), zero engineering |
| mechanism | 4!-per-plaquette CNOT orders; weight-2 hooks de-symmetrize the DEM | one-bit-per-edge orders, auto-covariant; weight-1 hooks merge |
| consequence for ML decoding | equivariance must be approximate/learned at circuit level | **exact equivariant decoders exist at circuit level by construction** |

Project convergence: the equivariance law ("symmetry pays iff EXACT and data
scarce") + the trivalence theorem + the decoder application meet here. The same
geometric fact that makes Kitaev solvable makes the hexagon's QEC code the one
whose circuit-level noise respects the lattice symmetry exactly — which is the
precondition for the >16×-sample-efficiency twisted-equivariant decoders measured
this week (`EQUIV_DECODER_FINDINGS.md`) to apply at circuit level, where real
devices live and where the surface code provably can't follow.

## Honest limits

- Detector-level DEM symmetry; no logical observable yet (the χ-twist machinery
  needs the honeycomb code's moving logicals — next build). The surface-code null
  was also detector-level, so the contrast is apples-to-apples.
- Torus only (no spatial boundaries); planar patches break translations as always.
- Uniform noise rates assumed (structural alignment is geometric; real-device rate
  inhomogeneity deforms probabilities for any code equally).
- The C2/mirror time-reversal pairing is classified, untested.
- Textbook ingredients (Hastings–Haah 2021 defines the code; its spacetime
  structure is known) — what was checked here and not found in memory or the
  literature sweep is the *exact circuit-level DEM symmetry verification* and its
  decoder-side contrast with the surface-code obstruction. Novelty unaudited;
  treat as strong portfolio + the foundation for the June-30 Floquet decoder arc,
  not a claimed publication.

## Next steps

1. Logical observable + χ-twist on the honeycomb code → train translation+C3-tied
   CNN decoder on honeycomb syndromes vs plain (the d=5/7 pipeline ports directly).
   **→ DONE 2026-06-12, see next section.**
2. June-30 QPU arc: the hardware Floquet demo now pairs with a decoder result.
3. Optional: time-symmetric noise variant to test the C2×time-reversal class.

---

# 2026-06-12 — Logicals + the GL(2,F2) twist: C3 MIXES the two logical qubits

*`scripts/honeycomb_decoder_core.py` (+`_local.py`, `_run.py`),
`runs/honeycomb_decoder_gates.{log,json}`. All numbers below from the completed
gates log. Training sweep: PENDING at time of writing — no sweep numbers here
(hard rule); see `runs/honeycomb_decoder.{log,json}` when complete.*

## What was built (all gates G2-style verified before any training)

1. **Terminated memory experiment** on the torus HH code: |+>^n init; structural
   detectors = s=0 edge records (X-type on |+>^n) + k=0 windows of color-0 faces
   (their stabilizer is X-type: one Y-edge and one Z-edge meet at each vertex,
   Y·Z ∝ X) + bulk window comparisons + virtual-round closures (data-pair
   reconstruction of the missing color) + final window-vs-data comparisons for
   readout-color faces. G2 = 256-shot noiseless determinism, all detector kinds.
2. **Logical observables DERIVED, not hand-built:** GF(2) null space of the
   noiseless record matrix (1024 shots) = every deterministic record parity;
   quotient by detector span = the trackable logicals. **Empirical discovery en
   route: the deterministic wrap pair exists ONLY at the S≡1 (mod 3) / MY
   termination** — S≡0/MX and S≡2/MZ carry zero wraps (the honeycomb e↔m logical-
   type rotation, found by sweeping the termination phase, not assumed). The two
   wraps at L=6: weights 116/128 records (8/14 final-data + ~110 check records) =
   the "moving logical" spacetime tube, derived automatically.
3. **The twist generalizes from a scalar χ to (A, X):** on the torus C3 acts on
   H1 by an order-3 matrix, so the DEM symmetry action is
   (dets, obs⃗) → (P·dets, A·obs⃗ ⊕ X·(P·dets)), A ∈ GL(2,F2), X = two χ rows,
   solved per det-set group (multiset method) and verified against every DEM
   mechanism group (4,584 groups at L=6, p=0.002 circuit-level, cx-naive).

## VERIFIED at L=6 (and L=3): the full p3_9 element table

| element | A (homology action) | twisted? |
|---|---|---|
| T22, T44 | **I** | yes (\|χ\| up to 168) |
| C3, C3·T22, C3·T44 | **[[1,1],[1,0]]** (order 3) | yes |
| C3², C3²·T22, C3²·T44 | **[[0,1],[1,1]]** = ([[1,1],[1,0]])² | yes |

- **Translations preserve the homology classes (A=I); C3 genuinely MIXES the two
  logical qubits** — (o₁,o₂) → (o₁⊕o₂, o₁). Both predictions logged before the
  run (P2) confirmed. Design lesson #3 of the decoder program: at circuit level
  on the hexagon-native code, the correct equivariant decoder is a **4-class
  (joint two-bit) twisted model** — scalar-label equivariance cannot express the
  C3 symmetry at all.
- Decoder construction f(x)[c] = (1/9) Σ_g mlp(x̃_g)[A_g c ⊕ X_g·x̃_g] is exactly
  invariant on realizable syndromes: probe max deviation **7.45e-09** (X is
  gauge-fixed only modulo mechanism-invisible directions, hence the realizable-
  syndrome caveat — documented in the module).
- Noise gate: p=0.008/0.004 are above-threshold-ish at L=6 (MWPM joint 0.27/0.62);
  auto-p landed at **p=0.002: MWPM joint 0.9486, majority 0.2824** (4-class task,
  obs-flip rate 0.71). MWPM decomposition clean once the global-parity trivial
  parity is dropped (a global detector turns boundary-layer errors into
  undecomposable hyperedges — caught and documented).

## ✅ TRAINING SWEEP COMPLETE (same day, 621s local GPU) — symmetry is the
## difference between not learning and learning

*`runs/honeycomb_decoder.{log,json}` — numbers read from the completed log.
L=6, R=9, MY termination, p=0.002 circuit-level cx-naive, matched 63,236-param
MLP backbone in every arm, 5 seeds, 200k test shots. MWPM joint 0.9471,
majority class 0.2908.*

| n train | plain | c3tw (3-elem) | p3_9 (9-elem) |
|---|---|---|---|
| 250 | 0.2575 ± .005 | 0.2820 ± .004 | **0.3633 ± .012** |
| 1,000 | 0.2665 ± .003 | 0.3021 ± .004 | **0.4037 ± .006** |
| 4,000 | 0.2721 ± .001 | 0.3226 ± .004 | **0.4698 ± .003** |
| 16,000 | 0.2741 ± .000 | 0.3488 ± .001 | **0.6005 ± .009** |
| 64,000 | 0.2899 ± .002 | 0.5145 ± .004 | **0.7443 ± .004** |

- **The plain MLP NEVER learns this task** — at/below the majority baseline at
  every n through 64k samples. The 4-class moving-logical honeycomb task at 360
  detectors is effectively unlearnable for the bare backbone at these sizes.
- The SAME backbone with the exact 9-element twisted tying reaches **0.7443** —
  a +45 pp gap at 64k, ordering p3_9 > c3tw > plain at every n (prediction P3 ✓).
  On the code where the symmetry is exact at circuit level, equivariance is not
  a sample-efficiency multiplier but a learnability switch — convergent with
  The END's ablation (plain pooling fails to learn; see
  `runs/DECODER_LITERATURE_POSITIONING.md`).
- Honest: p3_9@64k is still 20 pp below MWPM (capacity wall again — 2-layer MLP;
  curves still rising 16k→64k, not saturated); matched params, NOT matched FLOPs
  (9 forward passes per step); single backbone family; one code size.
- Next levers: CNN × p3_9 composition (translation weight-tying + 4-class
  twisted averaging), larger n, then the June-30 hardware pairing.

### Big-n extension (same day, local GPU 806s, `runs/honeycomb_decoder_bign.json`;
### 3 seeds, steps=20k, same test set)

| n train | plain | c3tw | p3_9 |
|---|---|---|---|
| 64,000 | 0.2861 ± .002 | 0.4971 ± .002 | 0.7306 ± .002 |
| 256,000 | **0.3907 ± .001** | 0.6256 ± .005 | **0.8060 ± .003** |

- **p3_9 keeps climbing: 0.806 at 256k** (14.1 pp below MWPM 0.9471, still
  unsaturated). c3tw 0.626.
- **plain finally wakes at 256k (0.391, above majority at last) — but
  p3_9@1,000 (0.4037) still beats plain@256,000 → ≥256× sample efficiency** on
  the honeycomb task (it was >16× on surface d=5; the exact-circuit-level-
  symmetry code pushes the gap two orders of magnitude).
- Honest: at fixed n=64k, 20k steps slightly UNDERPERFORMS 4k steps for the
  tied arms (0.7306 vs 0.7443; c3tw 0.4971 vs 0.5145) — mild overfitting with
  longer training, and 3 seeds here vs 5 there; not a contradiction, noted.

### CNN × p3_9 composition (same day, Modal 5×A10G ~45 min/seed,
### `runs/honeycomb_decoder_cnn.json`; 5 seeds, 46,644-param 3D-CNN backbone,
### circular padding on torus dims, same test set; probe 7.45e-09)

| n train | cnn (plain) | cnn_c3tw | cnn_p3_9 | (MLP p3_9 for reference) |
|---|---|---|---|---|
| 250 | 0.2555 ± .005 | 0.2810 ± .009 | **0.3647 ± .011** | 0.3633 |
| 1,000 | 0.2649 ± .003 | 0.3021 ± .008 | **0.4337 ± .012** | 0.4037 |
| 4,000 | 0.2732 ± .002 | 0.3393 ± .006 | **0.5291 ± .008** | 0.4698 |
| 16,000 | 0.2820 ± .003 | 0.4055 ± .007 | **0.6351 ± .013** | 0.6005 |
| 64,000 | 0.3184 ± .007 | 0.5380 ± .008 | **0.7502 ± .015** | 0.7443 |

Three answers (predictions P1–P3 of `honeycomb_cnn_core.py`):
1. **The translation prior alone does NOT rescue learnability** (P1 negative):
   plain CNN sits at/near majority through 16k and reaches only 0.318 at 64k.
2. **The composition pays in the scarce-mid regime**: cnn_p3_9 beats MLP-p3_9
   by +3.0/+5.9/+3.5 pp at 1k/4k/16k, converging at 64k (0.7502 vs 0.7443).
   The ~0.75 ceiling at 64k (≈20 pp below MWPM 0.9471) is NOT broken by conv
   composition — same plateau pattern as the surface d=7 CNN. Wall =
   architecture scale, not weight-tying choice.
3. ⭐ **Conv weight-sharing CANNOT absorb the twist** (P3 answered no, sharply):
   on the SAME conv backbone, going from C3-averaging to full p3_9 averaging
   adds +21 pp at 64k — even though convolutions are translation-equivariant,
   the LABEL twist under translations (A = I but χ ≠ 0) is a cocycle on the
   output, invisible to feature weight-sharing. The twisted averaging is
   irreducible. This deepens the element-choice law: weight sharing handles
   features; only twisted averaging handles the label.

### Scale ablation — the capacity referee question (2026-06-12, local GPU 491s,
### `runs/honeycomb_decoder_scale.json`; 2 seeds, steps=20k, same task/test set)

Predictions P-A/P-B/P-C logged in `scripts/honeycomb_decoder_scale.py` BEFORE
the run. Question: is "plain never learns" an artifact of the 63,236-param
backbone? Arms scale plain to h=512 (449,540 params, 7.1×) and h=1024
(1,423,364 params, 22.5×); p3_9 to h=512. MWPM 0.9471, majority 0.2908
(reproduced on a fresh 200k test set).

| arm | n train | acc |
|---|---|---|
| plain h=512 | 64,000 | 0.2858 ± .001 |
| plain h=1024 | 64,000 | 0.2862 ± .001 |
| plain h=1024 | 256,000 | 0.3602 ± .001 |
| p3_9 h=512 | 64,000 | 0.7386 ± .001 |
| p3_9 h=512 | 256,000 | **0.8007 ± .001** |

- **P-B fired: the learnability switch is robust to 22.5× parameters.** Plain
  at 1.42M params still sits AT the majority baseline at 64k (0.2862 vs
  0.2908) — width alone does not wake the bare backbone.
- At 256k the wide plain wakes to 0.3602 — no better than h=128's 0.3907
  (slightly worse; 2 seeds vs 3). The ≥256× sample-efficiency line survives
  width: p3_9@1,000 (0.4037) > plain at ANY tested width @256,000.
- **P-C resolved: width does NOT narrow the MWPM gap for the tied arm either**
  — p3_9 h=512 gives 0.7386@64k / 0.8007@256k vs h=128's 0.7443 / 0.8060
  (flat-to-slightly-down at 4× width). Combined with the CNN backbone swap
  landing at the same ~0.75@64k, the remaining ~15 pp gap to MWPM is
  **architecture-class (2-layer MLP / shallow CNN), not parameter count and
  not tying strategy**.
- Honest: 2 seeds; h>128 arms use the same 20k-step budget (the mild 20k-step
  overfitting noted above applies to all arms equally).

### Second-size structure check — L=9 (2026-06-12, 4s,
### `runs/demsym_honeycomb_L9.{log,json}`, via the demsym package)

Predictions P1–P3 logged in `scripts/demsym_honeycomb_L9.py` before the run;
all three confirmed:
- P1: exactly 2 deterministic wraps, again only at S≡1 mod 3 + MY readout
  (weights 174/200) — the termination effect is not an L=6 accident.
- P2: T(3,3) → A=I; **C3 → the identical order-3 matrix [[1,1],[1,0]]** on
  the independently re-derived L=9 wrap basis.
- P3: close({C3, T33}) = 9 verified elements, A-pattern 3·I / 3·M / 3·M²,
  exact over all 10,314 mechanism groups (810 detectors).
- Build note (caught by the P1 gate, first run): `derive_logicals`' default
  1,024 noiseless shots under-samples the L=9 record row space (~1,618
  random dims) → 597 spurious "wraps"; 4,096 shots fixes it. The L=6 default
  was silently sufficient. Lesson: the empirical-nullspace shot count must
  exceed the record matrix's random rank.

## Honest limits

- Torus; naive uniform extraction order (the point); uniform rates; MY-basis
  memory only (the termination that carries deterministic wraps with X init).
- The trivial zero-data parity (global last-XX-round product) is dropped, not
  decoded — negligible signal, big matching cost.
- Prior-art positioning (The END, DA codes) + claims-hygiene checklist:
  `runs/DECODER_LITERATURE_POSITIONING.md`. Solver validated against The END's
  analytic construction on the toric code (blind-unique among 24 label
  permutations, all elements): `scripts/toric_end_bridge.py`,
  `runs/toric_end_bridge.{log,json}`.
