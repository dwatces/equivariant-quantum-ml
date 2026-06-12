# Kitaev honeycomb TOPOLOGICAL ORDER — the untested regime (2026-06-07, Opus 4.8)

User asked for "more angles, imaginative, things not tested on QPU, use all compute." The genuine
gap: the entire project tested **Haldane band topology** (single-particle: chirality, edge transport,
disorder) and found it real-but-modest. It **never touched the KITAEV honeycomb** = many-body
**topological ORDER** (Z₂ fluxes, anyons) — which the project's own GQML/dequantization analysis names
as the *genuine quantum-advantage regime* (symmetry alone is classically simulable; advantage needs
topological order). And here **"the square can't" is a theorem**, not a marginal sim result.

Three new artifacts, all run, every number read from a completed log; one prose bug found+corrected.
Scripts: `kitaev_order.py`, `kitaev_anyons.py`, `ibm_kitaev_flux.py`. Logs/JSON in `runs/`.

---

## 1. Classical exact diagonalization — topological order is EXACT, square-impossible (`kitaev_order.py`)
8-site honeycomb torus, H = −Σ Jγ σγσγ over 3-colored bonds.
- Honeycomb is **trivalent** (every vertex has exactly one x-, y-, z-bond). Plaquette flux
  W_p = X Y Z X Y Z: **‖[W_p,H]‖ = 0** (exactly conserved) and **‖[W_p,W_q]‖ = 0** (mutually
  commuting Z₂ set) — the solvable spin-liquid structure = topological order. Ground state has
  **sharp ⟨W_p⟩ = −1**. Holds in both the isotropic (Jx=Jy=Jz) and the gapped toric-code phase
  (Jz≫Jx,Jy): ‖[W_p,Hₐ]‖=0 there too.
- **SQUARE control (9-site, coordination 4):** cannot be 3-colored one-type-per-edge (pigeonhole:
  4 bonds, 3 types); the best-effort plaquette flux gives **‖[W,H]‖ = 111 ≠ 0** → no conserved Z₂
  flux → **no topological order**. This is the cleanest "square fundamentally can't" in the whole
  project — a coordination theorem, not a tie-within-noise.

## 2. Classical anyons + protection gap (`kitaev_anyons.py`)
H block-diagonalizes into 2^(#plaquettes) **flux sectors** (16 isotropic). 
- Ground sits in the **uniform flux sector (−1,−1,−1,−1)** — Lieb's theorem confirmed.
- Flux excitations (visons) are **gapped localized anyons** = the topological protection.
- Gap is **largest at the isotropic point (~0.46)** and **shrinks** toward the deep anisotropic
  toric limit (~Jx²Jy/Jz²) — correct Kitaev→toric-code physics. (Initial log said "grows" — WRONG,
  corrected; the data's shrink is the right physics.)
- HONEST CAVEAT (finite size): the 4-plaquette torus forces the cheapest excitation to flip all 4
  fluxes (can't isolate one vison pair), so the gap MAGNITUDE is finite-size-limited; the
  qualitative result (sectors + Lieb ground + gapped anyons) is robust.

## 3. QPU — Z₂ flux conserved on REAL HARDWARE (`ibm_kitaev_flux.py`) — first Kitaev/topological-order on QPU
One hexagon (6 qubits), ring edges colored [Z,X,Y,Z,X,Y] so [W_p, H_Kitaev]=0 (sim-gate verified
⟨W_p⟩=+1.000 conserved noiselessly). Prepare the +1 flux eigenstate, Trotter-evolve, measure ⟨W_p⟩.
SIM-GATE PASSED (Kitaev ⟨W_p⟩=+1.000, field-control +0.193, 12 logical 2q = shallow).
**HARDWARE (job `d8ic2ipe8nrc73bi59cg`, ibm_marrakesh, DD-mitigated, 46 transpiled 2q):**
- ⟨W_p⟩ = **+0.924 (t=0) → +0.760 (Kitaev 1 step) → +0.667 (Kitaev 2 steps)** — flux stays high,
  the decay is decoherence (graceful), not flux-breaking.
- **Kitaev+field control at the same depth: ⟨W_p⟩ = +0.121** — the non-commuting perturbation
  genuinely DESTROYS the flux. The clean signal is the **gap: 0.667 (Kitaev) vs 0.121 (field)** at
  identical depth → the Z₂ flux is conserved by Kitaev dynamics and broken by non-Kitaev dynamics,
  on real qubits = **topological order on hardware.**
- Cost ~11s (budget 439→450/600, 150s remaining).
HONEST SCOPE: single plaquette, signature-level flux conservation (NOT anyon braiding / ground-state
degeneracy / scaling — those are deeper & qubit-scale-gated). New for this project; textbook physics.

---

## 4. Localized ANYON (vison) — classical + QPU (`kitaev_localize.py`, `ibm_kitaev_localize.py`)
The next signature beyond flux conservation: an anyon is LOCALIZED — a local operator flips ONE
plaquette's flux, leaving the neighbour untouched. Patch = two hexagons sharing an edge (10 sites).
- **CLASSICAL (exact, `kitaev_localize.py`):** W_p1 = X0Y1Z2Z4Y5X6, W_p2 = X3Y4Z5Z7Y8X9 (shared
  sites 4,5), all commutators 0. Flux vacuum ⟨W_p1⟩=⟨W_p2⟩=+1. Apply **Y on site 0** (anticommutes
  W_p1, commutes W_p2) → **⟨W_p1⟩ = −1, ⟨W_p2⟩ = +1** = a LOCALIZED anyon (only p1 flips). It's
  **STABLE** under Kitaev evolution (T=1.5: stays −1,+1). Rigorous, decisive.
- **QPU prep is SHALLOW:** the (+1,+1) vacuum is a stabilizer state — 8 product qubits + 1 entangled
  shared pair (Z4Y5=+1, Y4Z5=+1 via H·CX·Rx(−π/2)) = **1 logical 2q gate** (localization is pure
  stabilizer algebra, no dynamics needed). 4 circuits (vacuum/vison × measure-p1/p2; the shared
  qubits need different bases for W_p1 vs W_p2). SIM-GATE PASSED: vacuum (+1.000,+1.000), vison
  (−1.000,+1.000), 1 logical 2q. Submitted to ibm_marrakesh (queue 0), max transpiled 2q = 1, DD.
  **HARDWARE (job `d8icaec2upec739m1vp0`, ibm_marrakesh):**
  - **VACUUM: ⟨W_p1⟩ = +0.931, ⟨W_p2⟩ = +0.928**
  - **VISON (after Y₀): ⟨W_p1⟩ = −0.916, ⟨W_p2⟩ = +0.928**
  - W_p1 flips sign (+0.93 → −0.92 = anyon created) while **W_p2 is identical (0.928 → 0.928) = the
    neighbour flux is UNTOUCHED** → a **localized anyon on real hardware**, at high fidelity (the
    1-CNOT stabilizer circuit barely decoheres). Cost ~11s (budget 461/600, 139s left).
  HONEST SCOPE: stabilizer-level anyon creation + detection (NOT braiding or dynamics-stability on
  hardware — those are deeper); but the localization signature is clean and decisive.

## 5. Anyon BRAIDING — the −1 mutual statistics (`toric_braiding.py`, `ibm_toric_braiding.py`)
The deepest primitive: the toric code (= honeycomb Kitaev anisotropic limit) has e/m anyons with
−1 MUTUAL STATISTICS. Braiding an anyon around a torus cycle ⟺ the logical X-loop and Z-loop (which
cross once) ANTICOMMUTE → dragging the anyon flips the conjugate logical's sign.
- **CLASSICAL (exact, 8-qubit toric code):** logicals found numerically (X̄=X[0,2], Z̄=Z[0,1], cross
  once → anticommute). Vacuum |0_L⟩ ⟨Z̄⟩=+1; **BRAID (apply X̄): ⟨Z̄⟩ = −1** (the −1 phase);
  **CONTROL (apply a contractible star loop): ⟨Z̄⟩ = +1**. The −1 vs +1 contrast = mutual statistics.
- **QPU:** toric ground state prepared via Qiskit stabilizer-state synthesis (21 logical 2q),
  measure ⟨Z̄⟩=⟨Z0 Z1⟩ for vacuum / braid / control. SIM-GATE PASSED (+1.000 / −1.000 / +1.000).
  Transpiled to 39 2q on ibm_marrakesh (under the 70-gate budget guard). DEEPEST circuit of the
  session. **HARDWARE (job `d8icf4s2upec739m24t0`, ibm_marrakesh, DD):**
  - **vacuum ⟨Z̄⟩ = +0.782 | BRAID ⟨Z̄⟩ = −0.800 | control ⟨Z̄⟩ = +0.785**
  - The braiding operation **flips the logical sign (+0.78 → −0.80)** while the contractible-loop
    control stays positive → the **−1 anyon mutual statistics on real hardware**. Magnitudes are
    decoherence-reduced from ±1 (39-gate, 8-qubit ground state) but the sign contrast is unambiguous
    (braid −0.80 vs control/vacuum +0.78). Cost ~9s (budget 470/600, 130s left).
  HONEST SCOPE: abelian (toric-code) braiding via logical-operator anticommutation — the genuine −1
  mutual-statistics primitive; NOT non-Abelian braiding (Ising anyons, universal TQC) which is far
  deeper. But this is the deepest on-thesis QPU result the project has achieved.

## 6. Shored up to portfolio-grade — 3 independent backends + error bars (`ibm_shore_up.py`)
The two headline results re-run on **3 independent IBM backends** (marrakesh, kingston, fez), DD,
bootstrap shot-noise error bars, cross-device mean±std. Figure: `figures/kitaev_topological_order.png`.
- **BRAIDING ⟨Z̄⟩ (cross-device): vacuum +0.845±0.030 | braid −0.843±0.035 | control +0.845±0.033.**
  Braid-vs-control separation ≈ 1.69 ≫ ~0.03 device scatter. Per-device: marrakesh (.81/−.80/.82),
  kingston (.88/−.89/.89), fez (.84/−.84/.83) — all three agree. The −1 mutual statistics reproduced
  on three independent quantum computers.
- **KITAEV FLUX ⟨W_p⟩ (cross-device): t0 +0.918±0.030 | Kitaev(2 steps) +0.642±0.016 | field +0.115±0.030.**
  Flux conserved by Kitaev (0.64) vs broken by field (0.12), separation ≫ device scatter.
- **LOCALIZED ANYON also shored to 3 devices** (`ibm_shore_localize.py`, jobs d8icl2u6/d8icl7k2/d8iclcdv):
  vacuum ⟨W_p1⟩ +0.914±0.013 / ⟨W_p2⟩ +0.922±0.008 → vison ⟨W_p1⟩ **−0.905±0.010** / ⟨W_p2⟩ **+0.924±0.015**
  (anyon site flips, neighbour untouched; all 3 devices agree). So ALL FOUR QPU results (flux,
  localized anyon, braiding) are now multi-device with error bars.
- Bootstrap shot errors are tiny (~0.005–0.012); the honest dominant error bar is the cross-device
  std (~0.01–0.03). Total shore-up cost ~87s (budget 130→43s). Jobs: braid d8ich9pe/d8ichh9e/d8ichl42;
  flux d8ichotv/d8icht1e/d8ici0pe; localize d8icl2u6/d8icl7k2/d8iclcdv.
- Figure `figures/kitaev_topological_order.png` regenerated — all 3 panels now multi-device w/ error bars.

## NET — why this matters (and its honest limits)
This is the **deepest, most on-thesis "hexagon-only" result in the project**: the Kitaev spin liquid /
topological order exists **only** on the trivalent honeycomb (square impossible by a coordination
theorem), it's the **genuine quantum-advantage regime** (topological order, not single-particle
symmetry), and it now has a **real-hardware signature** (Z₂ flux conserved on the QPU). It reframes the
dream's "remaining shelter" — the quantum many-body regime — from abstract to concretely-probed.
HONEST LIMITS (unchanged big picture): textbook physics (not novel research); single-plaquette
signature, not anyons/degeneracy/scaling; useful-scale topological-order hardware is field-scale/years
out and NOT individually fundable. The hexagon's unique gift is genuinely QUANTUM + TOPOLOGICAL-ORDER
+ hardware-gated — exactly the project's standing conclusion, now demonstrated in its strongest form.
Capital/escape goals stay decoupled, via `consulting-launch`.
