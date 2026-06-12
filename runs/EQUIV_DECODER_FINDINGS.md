# Equivariant neural decoders v1 — twisted symmetry, 4× sample efficiency (2026-06-10)

*Application #1 of `APPLICATIONS_2026.md`. `scripts/equiv_decoder.py`,
`runs/equiv_decoder.{log,json}`. All numbers from the completed log; stim 1.16 +
pymatching 2.4; surface code, phenomenological noise p=0.008, matched params (21,377,
literally the same MLP), 3 seeds, 200k test shots.*

## Headline (d=3, C2 twisted-invariant vs plain, MWPM = 0.9948)

| n train | plain | twisted-inv | gap |
|---|---|---|---|
| 250 | 0.9339 ± .008 | **0.9819 ± .001** | +4.8 pp |
| 1,000 | 0.9666 | **0.9833** | +1.7 pp |
| 4,000 | 0.9763 | **0.9873** | +1.1 pp |
| 16,000 | 0.9829 | **0.9922** | +0.9 pp |
| 64,000 | 0.9905 | **0.9944** | +0.4 pp |

- **~4× sample efficiency** (inv @16k ≈ plain @64k) and inv wins at EVERY size.
- At 64k the twisted-invariant decoder sits **statistically at the MWPM baseline**
  (0.9944 vs 0.9948) while plain still lags — symmetry closes the gap to the
  near-optimal classical decoder. (Beating MWPM was not claimed; that needs
  correlated/real-device noise — the AlphaQubit regime, where sample efficiency
  matters even more because device data is scarce.)

## The two design lessons (the actual IP of the night)

1. **Lattice symmetry alone MIS-TIES the label.** A naive C2-invariant decoder was
   ~10 pp WORSE than plain (first run): the rotation maps the logical representative
   to representative × stabilizers, so the observable bit is not invariant.
   **The correct formulation is a TWISTED equivariance:** (dets, obs) →
   (P·dets, obs ⊕ χ·dets), where the mask χ encodes the stabilizers between the two
   representatives — their values telescope into detector sums, so the twist is
   exactly linear in the syndrome record. χ is solved from the exact detector error
   model by GF(2) elimination and the whole (P, χ) action is verified mechanism-by-
   mechanism against the DEM before use. Model: f(x) = ½[MLP(x) + (−1)^{χ·Px}MLP(Px)]
   — exactly invariant, identical parameter count.
2. **Circuit-level noise breaks naive spatial symmetry** — the syndrome-extraction
   schedule's CNOT ordering (hook errors) de-symmetrizes the DEM even on a symmetric
   lattice (verifier rejected all candidates at circuit level). Schedule-aware group
   actions (spatial op composed with the right temporal/basis correction) are the
   research frontier — and the reason this niche rewards a specialist.

## ✅ d=5/7 RESOLVED + the element-choice discovery (2026-06-11)

*`scripts/equiv_decoder_d5.py` + `equiv_decoder_d5b.py`, `runs/equiv_decoder_d5.{log,json}` +
`equiv_decoder_d5b.{log,json}`. All numbers from completed logs; unrotated memory_z,
phenomenological p=0.008, matched params (42,369 at d=5), 5 seeds, 200k test shots.*

**1. The χ-solver is FIXED — it was pairing, not physics.** v1 paired mechanisms
individually by (det-set, prob) with a 1e-9 tolerance; stim's DEM merging creates
probability collisions between distinct mechanisms at d≥5, generating wrong equations.
v2 extracts equations **per det-set group** (multiset comparison: each group forces
χ·v₂=0, =1, or is free — no pairing). Result: **all three symmetries verify at d=5
(754 forced groups) AND d=7 (2,260 groups)** — mirror_x is plain (χ=0), mirror_y and
C2 are twisted (|χ|=120 at d=5, 336 at d=7).

**2. ⭐ WHICH group element you average over matters enormously — the TWISTED elements
are the valuable ones.** d=5, identical MLP and params in every arm:

| n train | plain | mirror_x (χ=0) | C2 twisted | Klein-4 | MWPM |
|---|---|---|---|---|---|
| 250 | 0.8154 | 0.8151 | **0.8847** | **0.8942** | 0.9994 |
| 1,000 | 0.8193 | 0.8324 | 0.9040 | 0.9137 | |
| 4,000 | 0.8483 | 0.8624 | 0.9253 | 0.9389 | |
| 16,000 | 0.8801 | 0.9019 | 0.9538 | 0.9662 | |
| 64,000 | 0.9169 | 0.9295 | 0.9721 | **0.9797** | |

- **Klein-4 @ n=4,000 (0.9389) > plain @ n=64,000 (0.9169) → >16× sample efficiency**
  (up from ~4× at d=3); C2-twisted alone gives most of it.
- The mirror that FIXES the logical string (mirror_x, hence χ=0) is nearly worthless;
  the elements that MOVE the string (and therefore carry the stabilizer twist) do the
  work. Reading: tying weights across detector configurations related by a
  string-moving symmetry injects exactly the nonlocal structure the decoder must
  otherwise learn from data — the twist isn't a nuisance to engineer around, it IS
  the signal. (Klein-average exactness checked numerically: max branch deviation 7e-9.)
- Caveats (honest): matched params and training steps, NOT matched FLOPs (Klein-4 does
  4 forward passes per step); single MLP family; phenomenological noise.

**2b. Circuit-level NULL (2026-06-11, `scripts/circuit_symmetry_mine.py`):** exhaustive
search of the affine family (mirror_x/y/C2/identity × time-reversal × dt,dx,dy ∈ [−2,2];
999 maps per distance) finds **zero verified twisted symmetries of the circuit-level DEM
at d=3 or d=5**. The extraction schedule's hook errors break the whole geometric family,
not just the naive mirrors. Frontier sharpened: exact circuit-level equivariance needs
either **schedule co-design** (choose CNOT orders that map onto each other under the
mirror — build the symmetry into the circuit) or DEM-automorphism mining (non-geometric
relabelings, graph-isomorphism machinery). Affine maps are provably insufficient.

**3. The capacity wall is the new frontier (and the TPU justification).** Even Klein-4
@64k sits 2pp below MWPM at d=5 (0.9797 vs 0.9994); at d=7 the plain 2-layer MLP is
BELOW the majority-class baseline at 64k samples (0.7334 vs 0.7639) and order-2
averaging only just recovers it (0.7628). Bottleneck shifts data→architecture with
distance. Next steps: twisted-element averaging composed with spatial weight-tying
(CNN over detector coords), d=7 with the C2-twisted arm, n>64k, then Floquet-code
detectors (June-30 feeds real syndromes) → circuit-level schedule-aware actions →
correlated noise (the beat-MWPM regime).

## Modal compute burst (2026-06-11 afternoon) — 6 jobs, every next-step above executed

*All numbers from completed logs: `runs/equiv_decoder_d7_modal.log`, `_d5_bign.log`,
`_d911.log`, `_corr.log`, `_corr2.log`, `_cnn_modal.log` (partial — timed out),
`schedule_codesign.log`. Modal A10G unless noted; same matched-params discipline.*

### a. d=7 twisted MLP arms (438s) — symmetry rescues d=7 from below-majority

majority 0.7639, MWPM 1.0000, ndet=588. c2tw / klein4:
n=250: 0.7730±.009 / 0.7746±.015 · n=4k: 0.8060 / 0.8188 ·
n=16k: 0.8227 / 0.8420 · n=64k: 0.8531±.003 / **0.8846±.001**.
Plain was 0.7334@64k (below majority 0.7639) — Klein-4 averaging lifts the same
backbone +15pp to 0.8846. Wall persists (11.5pp below MWPM) but symmetry is doing
real architectural work, not just sample efficiency.

### b. d=5 big-n scaling (802s) — the advantage survives 1M samples

steps=4k:  64k: plain .9184 / kl4 .9791 · 256k: .9608/.9878 · 1M: .9679/.9902
steps=20k: 64k: plain .9229 / kl4 .9798 · 256k: .9739/.9898 · 1M: **.9863/.9951**
Klein-4 stays ahead at every (n, steps) including 1M samples (+0.9pp at 20k steps);
kl4@64k ≈ plain@256k–1M → ≥4–16× sample efficiency holds AT SCALE; kl4 1M/20k
(0.9951) closes to 0.4pp of MWPM (0.9994) under phenomenological noise.

### c. d=9/11 (292s) — the MLP family is capacity-dead by d=9; symmetry delays, not removes

d=9 (ndet=1296, majority .6826): plain BELOW majority at every n (best .6215@16k);
klein4 .6708@1k → .7105@16k → .7054@64k (saturates barely above majority).
d=11 (ndet=2420, majority .6160): plain ~.55 (below majority); klein4 peaks .6098@16k
— now klein4 is below majority too. Architecture, not data, is the binding constraint
from d=9 on. (MWPM 1.0000 at both.)

### d. Correlated noise — Klein-4's edge GROWS, but MWPM is not yet beaten (honest)

DEPOLARIZE2(0.01) on horizontal data pairs, d=5 (majority .7845, MWPM .9955,
decomposition still clean): plain .7554/.7713/.8238 (4k/16k/64k) vs klein4
.8667/.9057/**.9350** — the gap at 64k is +11pp (vs +6pp iid) = correlation amplifies
the symmetry advantage. Best-shot klein4 256k/20k-steps: 0.9643 — still 3pp below MWPM.
DEPOLARIZE2(0.03), d=5 (majority .5805, MWPM .7705): plain below majority (~.53);
klein4 .5775@64k; best-shot klein4-MLP 256k/20k .6069; klein4-CNN .6739 — still 10pp
below MWPM. **Verdict: pair-correlations on data qubits do NOT break MWPM's
decomposition here; the beat-MWPM regime needs noise that actually frustrates matching
(Y-biased / crossing-pair correlations) or real-device data. Do not claim otherwise.**

### e. CNN backbone + Klein-4 (job 3, 3h timeout; d=5 complete, d=7 n≤4k + re-run)

3-layer 3D CNN over the (t,x,y) detector grid, 87,641 params, group-wrap exactness
7e-9. d=5 (grid 6×9×9): cnn .8309@250 → .9560@64k; **cnn_klein4 .9198@250 → .9781@64k**
— at n=250 the CNN+Klein-4 beats the MLP+Klein-4 (.8942) by +2.6pp (best low-n d=5
arm yet); at 64k they tie (.9781 vs .9797). The two structures (translation
weight-tying + exact twisted averaging) COMPOSE cleanly.
d=7 (grid 8×13×13): cnn_klein4 **.8253@250 / .8901@1k / .9216@4k** — at 4k samples it
already exceeds the MLP-klein4 at 64k (.8846): the CNN breaks the d=7 capacity wall
with 16× less data. (n=16k/64k hit the 10800s Modal timeout; completion job
`equiv_decoder_modal_cnn7.py` → `runs/equiv_decoder_cnn7.json`.)
**✅ d=7 completion (read 2026-06-12 from `equiv_decoder_cnn7.{json,log}`, A10G 5151s,
5 seeds):** cnn .8870±.010@16k → .8828±.022@64k; **cnn_klein4 .9265±.004@16k →
.9245±.004@64k**. Both arms PLATEAU 16k→64k (flat within error) — the CNN has its own
ceiling ~0.925 at d=7, 7.5pp below MWPM (0.99996). Klein-4 keeps +4pp over plain at
both n. Honest read: CNN+Klein-4 moves the d=7 wall from "below majority" (MLP) to
~0.925, but more DATA no longer helps at this depth/width — next lever is architecture
scale or the honeycomb code's exact circuit-level symmetry (below), not n.

### f. Schedule co-design search (job B) — constructive NULL, 3 families, obstruction identified

`scripts/schedule_codesign.py`, d=3 unrotated memory-Z, rounds=3, full circuit noise,
gates G1 (no collision) / G2 (deterministic detectors) / G3 (distance preserved) /
G4 (exact twisted-symmetry verifier). Family A uniform (576): 96 collision-free, 16
pass G1–G3, **0 verify**. Family B mirror-paired halves (576): all 576 collide —
structural (center-column ancilla's step-k E target is exactly the mirrored
neighbor's step-k W target). Family C = B + time-reversed right half (576): collision
fixed as designed (32 valid, 8 pass gates), **still 0 verify**. Combined with the
999-affine-map null: exact geometric circuit-level equivariance is robustly obstructed
for this code at d=3. **Identified obstruction: any ancilla ON the mirror axis needs a
CNOT order whose hook-error prefixes are E↔W-symmetric, and no permutation equals its
own E↔W swap.** Escalations: per-plaquette schedules exploiting hook-error DEM
degeneracy, DEM-automorphism mining (graph isomorphism, non-geometric), or codes whose
symmetry axes avoid ancillas. This null is publishable-grade evidence that
schedule-aware equivariance is a genuine research problem, not an engineering tweak.
**⭐ RESOLVED SAME EVENING, constructively, by changing CODE not schedule: the
honeycomb Floquet code's circuit-level DEM carries an EXACT p3 space group
(translations + C3) plus exact time-bulk screws, even with naive ancilla extraction
— the trivalence theorem (weight-2 checks ⇒ one-bit schedules, weight-1 hooks) is
the mechanism. Full result + the square-vs-hexagon contrast table:
`runs/HONEYCOMB_CODE_SYMMETRY.md`.**

## ⚠️ Prior art (read in full 2026-06-12 — apply before ANY public claim)

**"The END: An Equivariant Neural Decoder for QEC" (Egorov, Bondesan, Welling,
ICLR 2023, arXiv:2304.07362)** already introduced twisted equivariance for QEC
decoders — syndrome-dependent label flips AND logical-permuting group actions
(translations, rotations, flips, X↔Z duality) — for the TORIC code at CODE
CAPACITY (perfect syndromes, no extraction circuits, 10⁶ training samples,
threshold focus). Their plain-pooling ablation fails to learn = independent
corroboration of lesson 1. Our lessons 1–2 are an independent reconstruction of
their structure via a different route (DEM-solved + mechanism-verified, which is
what survives at circuit level where analytic paths don't exist). OUR distinct
layers: circuit-level obstruction nulls, the honeycomb Floquet constructive
resolution incl. GL(2,F₂) logical mixing, and the scarce-data sample-efficiency
program. Full positioning + claims-hygiene checklist + new experiment queue:
`runs/DECODER_LITERATURE_POSITIONING.md`. Cite The END everywhere the twist
appears; never claim "first equivariant decoder."

## Why this matters commercially

Real-device syndrome data is the scarce resource (AlphaQubit trained on limited
Sycamore data); 4× sample efficiency at matched params, from a symmetry that is exact
BY CONSTRUCTION in every QEC code, is directly the kind of result decoder teams
(Riverlane, Google, IBM, PsiQuantum's stack) care about. This v1 + the two lessons is
the portfolio seed for application #1.
