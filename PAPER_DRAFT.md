# Two computational gifts of the honeycomb lattice: lattice-symmetry equivariance and topological chirality, on classical reservoirs and superconducting hardware

*Working draft — Opus 4.8 session, 2026-06-02. Honest scope: a focused demonstration + characterization,
aimed at a preprint / workshop level. Every reported number is read back from a completed run/job; see
`runs/` and the cited job IDs.*

---

## Abstract

The honeycomb lattice offers two distinct, frequently-conflated computational resources: (i) a
**lattice symmetry** (the C6/C3 point group) that, imposed as *equivariance*, yields sample-efficiency
and — in the quantum setting — trainability (avoidance of barren plateaus); and (ii) a **topology** (the
Haldane chiral edge / broken time-reversal) that yields robust, directional transport. We ask whether a
single honeycomb substrate can carry **both** at once. Classically, using fixed-physics quantum-walk
*reservoirs* with a trained linear readout, we find the two gifts are **separable but not jointly
exploitable**: a C3-equivariant readout gives a large sample-efficiency gain (+0.27 accuracy at scarce
data), but across five controlled experiments the reservoir's *topology* provides no measurable
computational advantage at single-particle scale, for a reason we make explicit — the symmetry gift
favors a compact rotationally-symmetric flake while the topology gift favors an extended edge, and these
geometries are mutually exclusive classically. On **superconducting hardware**, however, the two
*signatures* coexist: a six-site Haldane ring quantum walk simultaneously preserves the C3 lattice
symmetry (equivariance error E_C3 = 0.031 ± 0.007, vs 0.696 ± 0.014 for a symmetry-broken control) and
exhibits topological chirality (directional asymmetry +0.397 ± 0.007, vs +0.000 ± 0.003 for a
topologically-trivial control), consistent across three independent IBM devices with error mitigation.
The coexistence persists into the entangled regime: with two hard-core excitations, the two-body
correlation is simultaneously C3-equivariant (E_C3 = 0.043 vs 0.652 broken) and chiral (mirror TVD =
0.131 vs 0.021 trivial). The result is a hardware realization of an *equivariant circuit on a
topological substrate*; its scope is signature-coexistence and noise-robustness, not a computational
advantage, which we argue is intrinsically a larger-scale (topological-order) question.

---

## 1. Introduction

Geometric machine learning exploits the symmetries of data; geometric *quantum* machine learning (GQML)
extends this to quantum models, where symmetry is unusually load-bearing: symmetry-equivariant quantum
neural networks provably avoid **barren plateaus**, the exponentially-vanishing gradients that otherwise
render variational quantum models untrainable [PRX Quantum 5.020328; 5.030320; npj QI 2024]. The
mathematics is shared with classical equivariant networks — both are governed by the **commutant** of
the symmetry group via Schur's lemma [arXiv 2210.07980] — so symmetry is a single thread running from
classical convolution to quantum trainability.

A second, independent property of certain lattices is **topology**. The honeycomb is the canonical
example: the Haldane model gaps its Dirac cones into a Chern insulator with chiral, disorder-robust edge
modes. Topology underlies fault-tolerant proposals (Kitaev-honeycomb / honeycomb-Floquet codes) and, in
reservoir computing, has been associated with enhanced edge memory.

The honeycomb is unusual in offering *both* a rich point-group symmetry **and** nontrivial topology.
This motivates a concrete question: **can one honeycomb substrate deliver both computational gifts at
once — symmetry for efficiency/trainability and topology for robustness?** We address it in two regimes:
classical fixed-physics reservoirs (Sec. 3) and superconducting quantum hardware (Sec. 4).

Our contributions are: (1) a clean classical demonstration that a *C3-equivariant readout* on a
honeycomb quantum reservoir is sample-efficient; (2) an honest, controlled finding that the reservoir's
*topology* confers no single-particle computational advantage, with a geometric explanation for why the
two gifts resist combining classically; and (3) a multi-device, error-mitigated **hardware**
demonstration that the two *signatures* — C3 equivariance and topological chirality — coexist and
survive noise in a single six-qubit circuit.

## 2. Background: the two gifts

**Symmetry / equivariance.** A model is G-equivariant if its action commutes with the group action.
For a quantum walk whose Hamiltonian H commutes with a lattice rotation permutation P (PHP† = H), the
site-resolved output of a P-rotated input is the P-permuted output. Imposing equivariance on a (linear)
readout — averaging features over the symmetry orbits, the commutant projection — reduces parameters and
improves data efficiency when the task respects the symmetry.

**Topology / chirality.** The Haldane term (complex next-nearest-neighbor hopping with phase φ) breaks
time-reversal, opening a topological gap and inducing chiral circulation. Under φ → −φ the circulation
reverses. A trivial lattice (no Haldane term) shows no such chirality. On hardware, the *signature* of
the chiral/topological class is this directional, mirror-breaking asymmetry, which the present project
has previously shown survives noise (a 97% chirality-mirror preservation on `ibm_fez`).

## 3. Classical reservoirs: the gifts are separable but not jointly exploitable

**Setup.** A fixed honeycomb quantum-walk reservoir (the substrate computes; only a linear readout is
trained — no barren plateau by construction). We compare a C3-equivariant (orbit-pooled) readout against
a plain per-site readout, and a topological (Haldane) reservoir against a trivial one, on rotation-
invariant classification (`scripts/hex_qrc_equivariant.py`).

**Result 1 — the symmetry gift is real (Fig. 1).** On a rotation-invariant task with discrete-C3
inputs, the C3-equivariant readout is sharply sample-efficient: at n_train = 12 it reaches 0.92 vs 0.65
for the plain readout (a +0.27 gap), the gap closing to zero by n_train ≈ 48 — the textbook
commutant-projection signature (large gift when data is scarce, vanishing at abundance). The square-
lattice control *cannot host* a C3-equivariant readout (it is C4), the built-in mismatch control.
[`runs/hex_qrc_equivariant.json`; reservoir exactly C3-equivariant, [P,H] = 0 to machine precision.]

**Result 2 — the topology gift does not combine classically (controls in companion note).** Across five controlled
single-particle experiments — chirality discrimination, disorder-robustness, and open-system delayed-
recall memory at three operating points (and a non-Hermitian skin-effect variant) — the topological
reservoir provided **no** measurable computational advantage over the trivial one. The explanation is
geometric: the equivariance gift requires a **compact, rotationally-symmetric** flake (one needs rotation
orbits to pool), whereas the topological transport gift requires an **extended edge** (the chiral mode
must propagate); these geometries are mutually exclusive in a classical single-particle reservoir. We
therefore do not claim a classical joint advantage. [`runs/`; details in `GQML_FRONTIER_NOTE.md`.]

This negative is informative: it predicts that combining the gifts is intrinsically a *quantum*
(many-body / hardware) question — which motivates Sec. 4.

## 4. Hardware demonstration: the two signatures coexist on real qubits

**Circuit.** A six-site ring (the elementary honeycomb plaquette) with: nearest-neighbor hopping applied
as a two-colored Trotter step (each color a set of disjoint edges, so the step commutes *exactly* with
the 120° rotation S² — exact C3 at the Trotter level); plus a Haldane complex next-nearest-neighbor
hopping on the two triangles (the chiral/topological term). 17 circuits per device: six C3-orbit starts
(0…5), six symmetry-broken controls (a detuning on one site), and chirality (φ = ±π/2) and trivial
(t2 = 0) probes. Sim-gated before submission (noiseless statevector). A single physical layout is pinned
across all circuits per device, so the C3-orbit starts differ *only* in the walker's start qubit (no
transpilation artifact). [`scripts/ibm_equivariant_topological.py`, `ibm_eqtopo_robust.py`.]

**Metrics.** Within the conserved single-excitation sector: E_C3 = mean total-variation distance between
the rotated-start distributions and the rotation of the start-0 distribution (0 = exact equivariance);
chirality = P[clockwise neighbor] − P[counter-clockwise neighbor] (0 = no chirality); a mirror TVD
between φ = +π/2 and −π/2.

**Result 3 — both gifts survive, across three devices (Fig. 2, Table 1).** With dynamical decoupling
(XY4) and Pauli/readout twirling, on ibm_marrakesh, ibm_kingston and ibm_fez (jobs d8f5v207, d8f5v33a,
d8f5v3o7; 8192 shots; bootstrap error bars):

| | hardware (3-device mean ± std) | control | separation |
|---|---|---|---|
| Symmetry E_C3 | **0.031 ± 0.007** | 0.696 ± 0.014 (broken) | 22.7× |
| Topology chirality | **+0.397 ± 0.007** | +0.000 ± 0.003 (trivial) | clean |
| mirror TVD(φ+,φ−) | 0.516 ± 0.011 | — | 87% of ideal |

Conservation 0.624 ± 0.025. The devices agree tightly (E_C3 0.024–0.040; chirality 0.391–0.408).
Notably E_C3 = 0.031 is essentially the ideal Trotter floor (0.024) — i.e., the hardware preserves the
C3 lattice symmetry up to the circuit's own approximation, not device noise — while the chirality
recovers to 0.40 of an ideal 0.45 and vanishes (to 0.000) without the topological term. The two gifts
coexist in one real quantum circuit.

**Result 5 — coexistence in the entangled two-particle regime (S3, hardware).** To test whether the
coexistence is merely single-particle, we repeat the experiment with *two* hard-core excitations on the
ring (a genuinely many-body, entangled state) and measure the two-body pair-correlation G(i,j). Both
gifts survive (3 devices, mitigated; jobs d8f8rt1v, d8f8ru3o, d8f8ruro): the pair-correlation is
C3-equivariant — E_C3 = **0.043 ± 0.001** vs **0.652 ± 0.004** for the broken control (15× separation,
near the ideal 0.035) — and the Haldane phase steers it chirally — mirror TVD = **0.131 ± 0.006** vs
**0.021 ± 0.004** for the trivial control (near the ideal 0.152). Conservation 0.663 ± 0.025. The two
gifts thus coexist not only at single-particle level but in a two-body *correlation* — the entangled
regime that single-particle simulation cannot reach.

**Result 6 — size scaling (S4, hardware).** Repeating on a twelve-site ring (12 qubits; C3 requires N
divisible by six) confirms the coexistence is not a small-system accident: E_C3 = **0.039 ± 0.009** vs
**0.612 ± 0.010** broken (16×), chirality = **+0.291 ± 0.004** vs **−0.004 ± 0.009** trivial, on three
devices (jobs d8f9091v, d8f909o7, d8f90ajo). Conservation is lower (0.383 ± 0.019; 161 transpiled
two-qubit gates) but the contrasts persist, the chirality retaining 73% of the six-ring value at double
the depth.

## 5. Discussion and scope

This is, to our knowledge, the first single-circuit hardware demonstration that a honeycomb quantum walk
simultaneously realizes lattice-symmetry equivariance and topological chirality, verified across
multiple devices with mitigation. Its honest scope is **signature coexistence and noise-robustness, not
a computational advantage**. The latter is intrinsically out of reach at this scale for two reasons we
make explicit: (a) the classical separation of Sec. 3 (the gifts' geometric incompatibility at single-
particle scale), and (b) a *dequantization* tension — the small commutant that buys trainability can
also render a model classically simulable, so genuine quantum advantage must come from the topology /
many-body sector, not the symmetry alone. The natural path to a *joint advantage* is therefore the
many-body regime (multi-excitation, topological order), where protection is a bulk property independent
of single-particle edge geometry — the resource-gated frontier.

## Figures (generated from the run JSONs; `figures/`, `scripts/make_figures.py`)
- **Fig. 1** (`fig1_classical_equivariance.png`) — classical: a C3-equivariant readout on the honeycomb
  quantum reservoir is sample-efficient (+0.27 at n=12, converging to plain at abundance).
- **Fig. 2** (`fig2_qpu_coexistence.png`) — hardware: across single-particle, two-particle (entangled),
  and N=12 experiments (3 devices, mitigated), the symmetry is preserved (E_C3 ≪ broken control, left)
  and the topology is present (chirality ≫ trivial control, right). The headline figure.
- **Fig. 3** (`fig3_t2_causal_scaling.png`) — hardware: chirality scales monotonically with the
  topological strength t2 and vanishes at t2=0, tracking the noiseless simulation (causal attribution).

## 6. Methods
[circuit parameters: J=1, t2=0.3, φ=π/2, dt=0.6, 2 Trotter steps; XXPlusYY two-qubit primitive;
transpiled 2q depth 94; mitigation: DD XY4 + gate/measure twirling, 32 randomizations; post-selection on
the single-excitation sector; bootstrap (200 resamples) error bars; classical reservoirs: exact
diagonalization / Lindblad propagator, ridge readout. Full configs in the cited scripts.]

## Limitations
- Six qubits; one mitigated run per device per sector. (Single-particle AND two-particle/entangled
  sectors both demonstrated; larger excitation number / lattice size untested.)
- E_C3 includes a small (0.024) Trotter symmetry-breaking from the odd-cycle Haldane triangle; the
  result is *approximate* C3 + strong chirality, stated as such.
- The reported "22.7× / clean" separations are contrasts vastly exceeding device-to-device scatter
  (three devices + bootstrap), not formal p-values.
- No computational-advantage claim; the classical results include genuine negatives (Sec. 3).

**Result 4 — causal attribution, on hardware (S1; Fig. 3).** Sweeping the topological strength t2 on three
devices (mitigated; jobs d8f91qbo, d8f91rba, d8f91s87), the measured chirality rises **monotonically**
and tracks the noiseless prediction closely: +0.003 (t2=0) → +0.168 (0.1) → +0.290 (0.2) → +0.394 (0.3)
→ +0.477 (0.4), against a simulation of 0.00 / 0.18 / 0.33 / 0.45 / 0.53 (~88–90% retention). At t2=0
the chirality is +0.003 ≈ 0 — exactly non-chiral without the topological term. The chirality is thus
caused by, and scales with, the topological term, on real qubits and not only in simulation. (A
companion simulation confirms E_C3 stays small and the broken control large across the same sweep.)

## Strengthening experiments
**S1 (DONE, sim, free):** t2 sweep — chirality scales monotonically with topological strength while E_C3
stays controlled. Result 4 above. ✓
S2 (sim, free): noisy-Aer prediction matching the hardware contrasts (validates mitigation/interpretation).
**S3 (DONE, QPU, 3 devices):** two-excitation (entangled) equivariant+topological circuit — both gifts
survive in the two-body correlation (Result 5). The single-particle caveat is now addressed. ✓
S4 (QPU): ring-size scaling (6 → 8 → 10) — would show the coexistence is not a six-qubit accident
(remaining; depth grows so conservation drops — the next experiment if pursued).
S5 (classical, free): the C3-equivariance sample-efficiency curve on the *same* ring substrate, tying
the classical and quantum halves to one geometry.
