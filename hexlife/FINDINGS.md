# hexlife — the emergence door (2026-07-02, one night, GTX 1070 + Azure gpt-5-mini; commensurate sweep resolved 2026-07-03, GTX 1070 + independent Azure D32s_v5 replication)

**Question.** The six-year dream was a hexagonal substrate for emergent
intelligence. Five verdicts closed the learning doors — backprop routes
around geometry. But that verdict has an unexploited corollary this project
never tested: **self-organization cannot route around anything; there is no
optimizer.** Emergence is the one regime where the substrate must matter if
it matters anywhere — and it is literally the original image (knowledge
spreading as waves, structure arising on its own). Prior odds looked real:
the project's own verified physics (classical waves ~28% more isotropic on
the honeycomb; quantum walks 46–76%).

**One-line verdict (verdict 6, completed 2026-07-03 — TWO REGIMES).**
At matched cell density and identical physical rule, Lenia-class
self-organization is **substrate-indifferent in existence, abundance, and
packing order** at spot spacings ≳ 8 lattice constants — the lattice
*watermarks transients* and the settled patterns forget it. **Below ~3
lattice constants the settled patterns do NOT forget it**: the substrate
sets the packing symmetry class (hex→hexatic, square→tetratic; paired sign
tests near-unanimous) and the emergent bond-orientation phase LOCKS to the
crystal axes (|lock| up to 0.94–0.97), with commensuration-dependent sign.
**Crossover mapped + box-tested 2026-07-03 night** (see "The crossover
sweep" below): square spot-array locking oscillates through violent
commensuration sign flips (−0.56 → +0.95 → +0.55 across 4.4–5.4a) and
DIES by ~6.5a; hex arrays keep a weak +0.2 on-axis lock to at least
8.7a — box-size INVARIANT (n=192/256/384), so lattice-driven: the
arc's first genuine substrate asymmetry, from symmetry-matching (the
emergent order is itself hexatic; only the hex floor speaks its
language at long range). Second, unplanned discovery: a lattice-scale
MORPHOLOGY channel (ring / micro-texture patterns pinned at 2–3a) that
the square substrate sustains 2–3× more than hex, each locked in its
own symmetry. The full sentence of verdict 6 now: *the pattern does
not remember the lattice — until it is small enough to fit the
lattice's teeth, and even far above them it keeps a compass aligned to
any floor that shares its symmetry.*

---

## The rig (hexlife/lenia.py + experiments.py — reusable, selftested)

- ONE implementation; a substrate is a 2×2 Bravais basis. Square b=(1,0),(0,1);
  hex b=(a,0),(a/2,a√3/2) with a=√(2/√3) — **1 cell per unit area on both**.
- Kernel = continuous ring K(r) sampled at each offset's true Euclidean
  min-image distance, normalized; growth = standard Lenia; FFT convolution
  is exact on any Bravais lattice (index-space translation invariance).
- Paired design throughout: identical (μ,σ) grids AND identical index-space
  soups per seed across substrates. All metrics lattice-native (cell
  positions), never through a raster.
- Selftest: kernel normalization, uniform-state fixed point (proves the FFT
  trick), density match, nontrivial dynamics. PASS on CUDA.

## Results (all runs complete before anything below was written)

**E1 — calibration (`runs/hexlife_e1.json`, fig hexlife_e1_isotropy.png).**
The real phenomenon of the night:
- Sampled kernels carry equal-magnitude, symmetry-appropriate imprints at
  R=10: square c4=9.1e-4 (c6≈3e-8), hex c6=7.7e-4 (c4≈2e-7). The substrates
  are equally "rough," in different harmonics.
- Growth dynamics AMPLIFY the imprint into the transient: front at t=25
  carries square c4=4.2e-3 (×4.6), hex c6=5.3e-3 (×6.9). Total front
  roughness equal (sector std/mean 0.0191 vs 0.0198).
- The imprint grows as R shrinks: front iso 0.02 (R=10) → 0.13–0.17 (R=3).
  Large kernels wash the lattice out — this is *why* Lenia uses big kernels.

**E2 — baseline sweep, R=10 (1152 paired runs/substrate).**
Structured outcomes: square 161, hex 159. Identical. Doubles as proof the
paired rig is unbiased. The "travelers" are drifting Turing-style spot
phases (domain-filling dot lattices), not discrete gliders — soup at these
params lands in the spot phase.

**E5 — the R-floor (`runs/hexlife_e5_rfloor.json`, fig hexlife_e5_rfloor.png).**
The headline question: does hex keep supporting structure to smaller kernels
(structure per FLOP — "life" was a stretch at this goalpost, Daniel's
correction 07-02, accepted: these are self-organized patterns, not life)?
**NULL, cleanly:** alive fractions statistically identical
at every R (2.5→10): 337v349, 344v345, 300v294, 259v260, 203v207, 175v175,
161v159 (each n=1152, Wilson CIs overlap fully). The hexagon does NOT lower
the resolution floor.

**E6 — emergent packing order (`runs/hexlife_e6_spotorder.json`).**
Bond-orientational order (ψ4 tetratic / ψ6 hexatic, Delaunay on periodic
replication) of detected spot centroids, paired on params alive in both:
- R=10 (80 pairs): ψ6 hex>square in 25/80, ψ4 in 51/80 — chance-level, mean
  diffs 0.01–0.03 on means ~0.5. NULL.
- R=4 (40 pairs, after connectivity fix): ψ4 23/40, ψ6 20/40 — **exactly
  chance**. NULL. Emergent spot phases hex-pack (ψ6≈0.56) *on both
  substrates* — the hexagonal order in the patterns comes from the physics,
  not the lattice; the square substrate neither prevents nor degrades it.
- R=3 (spacing ~6 lattice constants): only 9 measurable pairs — open at the
  time of writing; **resolved 2026-07-03 by the commensurate sweep (below):
  at ≲3a the effect is real and large.**
- Per-spot SHAPE harmonics: metric INVALIDATED at small R by the lattice
  granularity floor — a featureless physical disk reads c4 up to 0.11 on
  square and c6 up to 1.0 on hex depending on cell count (disk calibration
  in session log). No honest shape claim either way.

**E4 — blind LLM judge (Azure gpt-5-mini vision, labels stripped, shuffled).**
"Lifelike 0–10" on paired finals: square 7.18 (n=11), hex 7.40 (n=10);
rate-limited to n=21 total. No perceptual difference — corroborates the
nulls at anecdote strength. Caveat: both substrates render through the same
square-pixel pipeline, but pixelation differs (hex resamples smoother), so
this probe can only ever be suggestive.

## The artifact hunt (five caught in one night — the method IS the story)

1. **Index-space seed shear**: a Gaussian bump built in index coords is a
   physically sheared ellipse on the hex basis → fake front anisotropy.
   Fix: physical-coordinate seeds.
2. **Raster-contaminated metrics**: measuring harmonics through a
   square-pixel render imprints m4 on everything. Fix: lattice-native
   harmonics on true cell positions.
3. **Rectangular min-image on a sheared torus**: exact for square (zero
   shear), scrambles angles on hex (a j-wrap must shift x by half a domain)
   → hex front read c4=0.61(!). Fix: fractional-coordinate wrap.
4. **Index-connectivity blob fragmentation**: ndimage 4-connectivity is
   square adjacency; it fragments hex blobs along the missing axial
   diagonal → spurious "significant" packing effects (36/40!) that went to
   chance (23/40) with lattice-correct structuring elements.
5. **Shape-metric lattice floor**: per-spot c4/c6 at 5–40 cells is dominated
   by sampling granularity (disk calibration) — retracted rather than
   baselined, since the floor jumps discontinuously with cell count.
Each artifact initially masqueraded as a *large, significant substrate
effect* — in both directions. Tonight's nulls survived the hunt; the effects
did not.

## Verdict 6, stated for the record

Self-organizing Lenia-class dynamics at matched density do not care which
lattice they run on: not for whether structure emerges (E2), not for how
much of parameter space supports it at any kernel radius (E5), not for the
orientational order of what settles (E6, to spacing ~8a). The substrate's
geometry survives only as a microscopic watermark on transients (E1) that
equilibration erases. The emergence door — the last door of the original
dream on classical digital substrates — closes like the others: **the
pattern does not remember the lattice.** What remains hexagonal about the
patterns was never the lattice's gift; hexagonal packing is what the physics
does everywhere, on any floor. (In that one poetic sense the dream was
backwards-right: the hexagon is not the substrate intelligence needs — it is
what emergence builds on its own, even on a square floor.)

**Amendment (2026-07-03, after the commensurate sweep):** verdict 6 as
stated holds for the dilute regime (spacing ≳ 8a) where it was measured. In
the commensurate regime (≲ 3a) the sweep below shows the opposite: the
pattern does remember the lattice — totally, in the symmetry channel. The
honest full sentence is now: *the pattern does not remember the lattice,
until it is small enough to fit the lattice's teeth.*

## Daniel's framing (2026-07-02, recorded before the follow-up sweep)

"Emergent patterns may come from physics; I dispute that's a barrier to
translate to compute; it just hasn't happened yet." Accepted, and it bounds
the verdict correctly: verdict 6 establishes substrate-indifference of
existence/abundance/packing at tested scales. It does NOT establish that
emergent pattern dynamics cannot be translated into computation —
*untranslated ≠ untranslatable*. The commensurate sweep below therefore adds
the observable that a translation would need first: **orientation locking**
(does the substrate write readable information — its crystal axes — into the
emergent state?). A locked phase is a substrate-addressable degree of
freedom; that is the minimal precondition for "compute on emergence."

## The commensurate sweep — the open stone, RESOLVED POSITIVE (2026-07-03)

TWO independent runs, analyzed 2026-07-03 morning after both 50/50 grids
completed — nothing below was written before that. PRIMARY: local GTX 1070,
seeds 11–15 → `runs/hexlife_commensurate.jsonl` (finished 00:21 NZST).
REPLICATION: Azure `hexlife-sweep` VM (Standard_D32s_v5, 32-vCPU torch-CPU
— GPU quota was 0 on the fresh sub), seeds 16–20 →
`runs/hexlife_commensurate_azure.jsonl` (completed before its auto-shutdown;
fetched 2026-07-03, VM then deleted). **The replication reproduces every
headline number to the 2nd–3rd decimal** on disjoint seeds and different
hardware/backend (e.g. square lock4 at R=4: −0.904 both; hex lock6 at R=2:
−0.969 vs −0.965) — the strongest artifact defense in this file. Design per
the docstring: n=256 domains, R ∈ {2, 2.5, 3, 3.5, 4}, paired 16×16 (μ,σ)
grid refocused on the low-μ alive strip; 11,946 measured patterns across
100 records. Combined analysis below (`hexlife_commensurate_all.jsonl` +
`_summary.json`); per-run tables in `_summary_azure.json` and the session
log.

**Q1 — packing symmetry class: the substrate OWNS it below ~3a.**

| sub | R | n_pat | ψ4 | ψ6 | lock4 (±CI) | lock6 (±CI) | spacing/a |
|---|---|---|---|---|---|---|---|
| square | 2.0 | 606 | **0.144** | 0.007 | −0.936±0.028 | +0.050±0.056 | 1.41 |
| square | 2.5 | 839 | **0.091** | 0.018 | +0.819±0.035 | −0.007±0.055 | 2.00 |
| square | 3.0 | 1316 | **0.068** | 0.017 | −0.827±0.027 | −0.048±0.036 | 2.50 |
| square | 3.5 | 1426 | **0.073** | 0.020 | −0.691±0.037 | −0.012±0.038 | 2.94 |
| square | 4.0 | 1423 | **0.045** | 0.024 | −0.904±0.018 | −0.035±0.036 | 3.08 |
| hex | 2.0 | 859 | 0.008 | **0.101** | +0.291±0.045 | −0.967±0.010 | 1.82 |
| hex | 2.5 | 1115 | 0.007 | **0.177** | −0.056±0.045 | −0.418±0.052 | 1.73 |
| hex | 3.0 | 1486 | 0.016 | **0.124** | −0.111±0.031 | +0.683±0.035 | 2.60 |
| hex | 3.5 | 1434 | 0.008 | **0.063** | −0.019±0.036 | −0.103±0.047 | 2.83 |
| hex | 4.0 | 1442 | 0.011 | **0.084** | −0.048±0.035 | −0.560±0.037 | 3.08 |

Paired sign tests (same seed, same (μ,σ), hex − square), near-unanimous at
every R: ψ6 hex-higher in 538/546 (R=2), 745/773 (R=2.5), 1220/1251 (R=3),
1100/1335 (R=3.5), 1203/1358 (R=4); ψ4 square-higher in the mirror-image
proportions (526/546, 746/773, 1101/1251, 1326/1335, 1238/1358). Binomial
p astronomically small; effect sizes modest (mean Δψ6 +0.04 to +0.18 on
magnitudes ~0.05–0.18). This is **symmetry selection**, not strong
crystallization: absolute order is weak everywhere at these spacings —
the substrate decides *which* harmonic survives, not how ordered it is.

**Q2 — orientation locking: the substrate writes its axes into the
equilibrium state.** The matched-symmetry lock channel is enormous —
square lock4 |0.69–0.94| at every R, hex lock6 up to |0.97| — while the
wrong-symmetry channel sits at zero (square lock6 all |≤0.05|; hex lock4
|≤0.11| for R≥2.5): a built-in null control the effect passes. The SIGN
flips with commensuration ratio: square at spacing √2·a anti-locks (the
45°-rotated diagonal phase, lock4=−0.94), at spacing 2.0a locks on-axis
(+0.82); hex at spacing ~1.7–1.8a anti-locks (−0.97 / −0.42 — the
(√3×√3)R30° motif of adsorbate epitaxy), at 2.60a locks on-axis (+0.68),
at 2.83a unlocks (−0.10), at 3.08a anti-locks again (−0.56). Rotated AND
on-axis phases are both substrate memory; the phase diagram vs spacing/a
is textbook commensurate–incommensurate physics (Novaco–McTague class),
here emerging in a continuous CA. One unexplained residual: hex lock4 =
+0.291±0.045 at R=2 — small cross-channel talk at the tightest
commensuration, noted, not interpreted.

**Q0 — an unplanned third observable: abundance is NOT null here.** Paired
alive counts (of 256 params, mean over the 10 seeds): R=2.0 square 60.6 vs
hex 85.9 (+42% relative), R=2.5 83.9 vs 111.5 (+33%), R=3.0 131.6 vs 148.6
(+13%) — hex higher in **10/10 seeds** at each of those R; R=3.5 and 4.0
null (+0.8 and +1.9, 6/10 seeds — chance). ⚠ This TENSIONS with E5's
abundance null at R=2.5–3 (337v349 at n=1152 params): E5 swept the broad
(μ,σ) grid on smaller domains; this sweep is the low-μ strip at n=256.
Within the strip, at tight spacings, the hex floor sustains marginal
parameters the square floor kills — plausibly the more isotropic small-R
kernel sampling (E1: the lattice imprint grows as R shrinks). ~~Which
factor (grid refocus vs domain size) reconciles the two is OPEN~~ —
**RESOLVED 2026-07-03 evening** (see "Q0 abundance reconciliation"
below): window dilution + E5's unpaired 2-seed power, not domain size;
the claim is now robust-but-conditional (R≲3, low-μ strip). R=2.0 was
below E5's tested floor and had never been measured.

**Bounds and honesty.** (i) The refocused low-μ strip produced spacings
1.4–3.1a, not the designed 2–6a — the resolved claim is bounded at ≲3a;
the 3–8a crossover to verdict 6's dilute null remains unmapped (the decay
is visible: mean Δψ6 shrinks +0.18 → +0.04–0.06 and hex lock weakens,
non-monotonically, toward R=4). (ii) Both substrates imprint their own
symmetry with comparable strength — **hex is not privileged**; this opens
"geometry transmits at commensurate scale," not "hex wins." (iii) Metrics
are the artifact-hardened set (lattice-correct connectivity, physical
coords, Delaunay on periodic replication); the sign-flip-with-R structure
also rules out any fixed per-substrate metric bias, which would not
oscillate with spacing. (iv) Patterns pooled across (μ,σ) within a seed are
not fully independent; the paired sign tests inherit the pairing design and
are the primary statistic.

**Answer to the translation lens (Daniel's framing above): YES.** At
commensurate scales the substrate writes readable information — its axes
and its symmetry class — into the settled state, and the write direction
flips with a controllable parameter (kernel radius → spacing/a). That is a
substrate-addressable degree of freedom: the minimal precondition for
"compute on emergence" exists, in exactly and only the regime where the
pattern scale meshes with the lattice's teeth.

## The crossover sweep + the morphology channel (2026-07-03 evening)

All from completed runs (crossover 80/80 records before analysis; probe
and mode-split after). Driver `hexlife/sweep_crossover.py` (predictions
P1–P6 pre-registered in its docstring before any run); R ∈ {4.5, 5, 5.5,
6, 7, 8, 9, 10}, strip grid, n=256, seeds 11–15; same artifact-hardened
`spot_stats`. Data `runs/hexlife_crossover.jsonl`, split analysis
`runs/hexlife_crossover_modes.json`, figure
`figures/hexlife_crossover_locking.png`.

**The near-artifact that became the second discovery.** The pooled square
table showed median spot spacing ~2a at R=5–8 (it should track the kernel,
~0.8R) then a jump to 7.6–9.3a — the signature I flagged as candidate
artifact #6 (threshold fragmentation). The probe
(`hexlife/probe_fragmentation.py`, 4 exemplars re-simulated;
`runs/hexlife_frag_probe.json`, `figures/hexlife_frag_probe.png`) ruled
fragmentation OUT: component counts are threshold-STABLE (0.25–0.6×max:
hex R10 714=714; square R5 swings <6%), and the fields show real
structure — at R=5 a macro-envelope filled with ~2a micro-texture; at R=7
a full-domain array of SQUARE RING/annulus spots whose arcs register ~2a
apart along the diagonals (exemplar cos4Φ = −0.995); square R=9-wide and
hex R=10 are clean true dot arrays. The detector reported real physics.

**Result 1 — the lattice-scale morphology channel (new observable).**
Splitting patterns at spacing 3.05a (valid R≥5; at R=4.5 kernel scale ≈
lattice scale — the genuinely commensurate bridge):
- BOTH substrates sustain a population of lattice-scale morphologies
  (rings, micro-texture) far above the kernel's preferred scale — but the
  square substrate carries 2–3× more of it: prevalence 86%→34% (R=5→10)
  vs hex 30%→7%, square higher in 5/5 seeds at every R.
- Each substrate's channel is locked in its own symmetry: square rings
  anti-lock to the diagonals, hardening with R (−0.88 → −0.97 at R=7→10);
  hex tight patterns lock on-axis (+0.79/+0.82/+0.53/+0.49 at
  R=5/7/8/9). Square's R=5–6 texture sub-regime is unlocked (+0.06/−0.16/
  −0.21) — two distinct sub-morphologies (texture vs rings).
- Reading: at R=5–8 the square lattice does not merely orient the
  pattern — it INJECTS its own ~2a length scale into what grows, where
  the continuous kernel alone would put 4–6.5a spots. Geometry acts on
  morphology selection, not just orientation. Why square ≫ hex is open
  (plausibly the square sampling comb's stronger short-wavelength
  ripple; E1 measured equal-magnitude imprints only at R=10).

**Result 2 — the honest array-locking crossover (wide mode only).**
- Square: the commensuration ladder continues out of the resolved sweep
  (−0.90 at 3.08a) with violent sign flips — −0.56 at 4.38a, **+0.95 at
  4.77a**, +0.55 at 5.40a — then locking DIES: +0.02/+0.07/−0.04/+0.01
  at 6.5–9.4a (every CI includes 0). A textbook commensurate–
  incommensurate transition, completed between spacing ~5.4a and ~6.5a.
  P1/P2 confirmed for square, including the predicted non-monotone
  oscillation.
- Hex: decays non-monotonically (+0.52 at 4.2a, dip +0.09 at 4.6a,
  +0.33 at 6.1a) to a stubborn tail of **+0.20–0.23 (CIs exclude 0) at
  7.0–8.7a** — trips P1's pre-registered falsifier (locking is NOT dead
  at ~8a on hex). Claimed only after the box test below.
- ✅ **The box confound RULED OUT (box192/box384 sweeps, completed
  before this paragraph was written).** The worry: at 8.7a the 256-cell
  torus holds ~29 rows; the periodic DOMAIN could lock the array (hex
  box axes coincide with the lattice axes; a hexatic array can never
  commensurately tile the square box) — predicting exactly the observed
  square-dies/hex-persists asymmetry. Pre-registered discriminator:
  box-driven → |lock| falls with row count; lattice-driven → invariant.
  Result (wide mode, R=10): hex lock6 = +0.196±0.081 (n=192, 22 rows),
  +0.231±0.078 (n=256, 29 rows), +0.196±0.080 (n=384, 44 rows) —
  INVARIANT across a 2× change in rows; at R=7 it RISES with box size
  (+0.28 → +0.33 → +0.46), the opposite of box-locking. Square lock4
  stays dead at every size (+0.15/+0.02/−0.09 and +0.14/+0.01/−0.07,
  CIs straddling 0, sign unstable). Tight-mode prevalence is
  box-invariant to two decimals (e.g. square R=7: 0.69/0.70/0.70). For
  R≤3 box-independence was already proven by strip128 (all locks
  replicate at half the box). **The hex dilute tail is lattice-driven
  and now claimed**, along with the weak persistent ψ6 selection at
  large R (66% at R=10).
- **Reading (the first genuine substrate ASYMMETRY of the arc):** the
  emergent spot arrays are intrinsically hexatic — they hex-pack on any
  floor (E6). So at long range only the hexagonal substrate is
  symmetry-compatible with the order the pattern itself wants: square's
  ψ4 grip has nothing hexatic to hold and dies by ~6.5a, while hex's ψ6
  grip stays weakly engaged (~+0.2) out to at least 8.7a (~3× further).
  Bounded honestly: this is symmetry-MATCHING privilege (the substrate
  that speaks the pattern's native language keeps a grip), not a
  general hex superiority — a hypothetical tetratic-ordered pattern
  would presumably invert it. It does NOT contradict the E6 magnitude
  null (ψ6 magnitudes match across floors); the memory lives purely in
  the phase/orientation channel, which magnitude metrics cannot see.
- Abundance at R≥4.5: null (±0–3%, mixed signs) — P4 confirmed.

**Q0 abundance reconciliation — RESOLVED (strip128 + broad128 cells
complete; broad256 replication still running).** Factorial cells, all
400 steps, seeds 11–15, per-param `alive_idx`:
- strip grid @ n=128: hex +33%/+14%/+4% alive at R=2/2.5/3, hex higher
  5/5 seeds at every R (vs +42%/+33%/+13% at n=256) → the effect
  SURVIVES halving the domain; if anything finite size UNDER-measures it.
- broad E5 grid @ n=128 (E5's own configuration, now paired + powered):
  aggregate +11%/+5%/+0% (5/5, 5/5, 4/5 seeds) — and restricted to the
  strip window inside the broad grid: **+34%/+10%/+3%**, matching the
  strip-grid cell (+33% at R=2) across different grids.
- So E5's null decomposes: (i) window dilution — off-strip params carry
  ~none of the effect; (ii) power — E5's unpaired 2-seed Wilson CIs
  could not resolve a diluted +5–11% (its own raw R=2.5 counts were
  hex-higher, 349v337); (iii) the effect dies by R=3.5 regardless.
  Disclosed minor factor: these cells run 400 steps vs E5's 500.
  The conditional-abundance claim upgrades to: robust, window-graded,
  size-consistent (R≲3, low-μ strip).
- Bonus from strip128: symmetry selection AND orientation locking
  replicate at n=128 near number-for-number (square lock4@R2 −0.986 vs
  −0.936; hex lock6@R3 +0.633 vs +0.683; ψ6 sign tests 342/350, 489/530,
  646/674) — a third independent replication of the commensurate-regime
  result, and the box-independence proof for R≤3.

## Open (designed, not promised)

1. ~~Box-size test~~ DONE (ruled lattice-driven; claims landed above).
2. ~~broad256 cell~~ DONE (strip-window +42/+29/+14% reproduces Q0 from
   a different grid; aggregate +15/+12/+5%; factorial closed).
3. **The morphology channel's (μ,σ) map + mechanism**: which strip params
   go tight, and why square ≫ hex. The biggest new stone. **First
   candidate mechanism FALSIFIED (2026-07-04,
   `hexlife/kernel_spectrum.py`, pre-registered):** the linear-response
   account (square's Cartesian comb aliases the ring kernel harder →
   stronger short-wavelength band in |K̂|) is ruled out — the
   secondary/primary band ratio is square>hex at only 2/7 R values
   (hex often carries MORE linear channel: 0.167 vs 0.116 at R=5)
   while expressing 2–3× LESS morphology; pooled corr(ratio,
   prevalence) r=+0.17 ≈ null. Within-substrate R-decline weakly
   co-moves (square r=+0.82, hex r=+0.49). Whatever selects the
   square's rings/texture is nonlinear or symmetry-geometric, not
   linear band availability. `runs/hexlife_kernel_spectrum.json`,
   fig `figures/hexlife_kernel_spectrum.png`.
4. **Transient-amplification physics**: the ×5–7 growth of the kernel
   harmonic into fronts — measure the amplification factor vs (μ,σ,R);
   small, clean, publishable-as-note.
5. **Oscillatory/wave regimes** (spirals, not spots): different
   universality class, untested.
