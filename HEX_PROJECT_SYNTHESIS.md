# The Hexagon Project — an honest synthesis

*Daniel Olliver · written 2026-06-07 · the clear-eyed answer to a six-year question*

## The question

In ~2020, looking at a hexagonal wire-mesh ceiling, the idea arrived: a neural network built on
hexagonal connectivity — six neighbours, knowledge spreading as waves — might be the right substrate
for genuine, emergent intelligence. Six years later this project set out to test that conviction the
only honest way: build it, measure it, and let the evidence speak, with no result written before its
run completed.

What follows is what ~25 experiments — on a GTX 1070, free GPUs, and real IBM quantum hardware —
actually established. It is not the story I wanted to be able to tell. It is the true one, and it is
more interesting than a simple yes.

## Four verdicts

**1. Hexagonal geometry gives a learning system no advantage. (NULL — robust, exhaustive.)**
Across every regime where a network *learns* or *computes* — backpropagation, frozen reservoirs,
local Hebbian rules, local online prediction, grid-cell/TEM scaffolds, message-passing routing — the
honeycomb ties or loses to the square. The reason is not subtle: a trainable network routes *around*
its connectivity; backprop optimises through whatever topology you give it, so geometry stops being
load-bearing. The square kept edging it. That is the answer, and manufacturing a hex win against it
would have been dishonest. The "intelligence is in the geometry" form of the dream is falsified.

**2. Hexagonal *symmetry* (6-fold equivariance) IS a real, learnable gift — but a narrow one.**
The one place the hexagon helps a learning system is equivariance: weight-tying across the lattice's
rotation group is a hard constraint backprop cannot route around. Confirmed on the rig — exact p6
group-convolution on the true hex lattice beats p4/p3/plain at matched parameters, *when data is
scarce and the symmetry is genuinely present*. It pays on dense hex-native data (bike-share cold-start
across 10 city-seasons; materials Chern classification) and vanishes when data is abundant or the
symmetry is absent (fragmented H3 demand; real ARC, where the symmetry is distributional not per-rule,
so cheap augmentation matches it). Real, bounded, not a general compute lever.

**3. Fixed-physics propagation on the honeycomb is genuinely special — but physically, not computationally.**
Classical waves stay ~28% more isotropic on the honeycomb; quantum walks ~46–76% more isotropic
(the Dirac-cone signature). These are real and match the physics literature. But they are properties
of *fixed dynamics*; they do not survive learning, and — shown cleanly this session — they do not
convert into a digital computational advantage (the wave-computed Voronoi tessellation is a tie with
the square once the metric is honest). The category error, named plainly: **simulating the hexagon's
physics on a digital computer to beat digital deep learning is structurally impossible — a digital
simulation of physics is strictly costlier than the physics.** The hexagon's gift was never meant to
run on a GPU.

**4. The honeycomb's quantum, topological gift is real — and it lives on hardware.**
This is where the conviction was *right*, precisely and narrowly. On real IBM quantum processors,
sim-gated and multi-device-validated:
- **Chirality** of the Haldane edge survives hardware noise (97% of the ideal mirror signature).
- **C3 lattice-symmetry equivariance** is preserved on the qubits (E_C3 ≈ 0.03, ~22× below a broken control).
- **Two-particle (entangled) interference** is steered by the chiral phase, on hardware.
- **Kitaev topological order** (this session's capstone) — the regime the whole arc pointed to:
  - The plaquette Z₂ flux is **conserved by Kitaev dynamics and broken by a field** (3 devices).
  - A **localized anyon** is created by a local operator while its neighbour is untouched (3 devices).
  - **Anyon braiding** — the −1 mutual statistics, the genuine primitive of topological quantum
    computation — flips the logical sign across **three independent quantum computers**
    (braid −0.84 ± 0.04 vs control +0.85 ± 0.03).

And here "the square cannot" is a *theorem*, not a near-miss: Kitaev's exact solvability requires the
trivalent honeycomb (coordination-3, 3-colourable bonds); a square lattice cannot host the
construction at all.

## The synthesis

The hexagon is a genuine marvel — but of **physics and topology**, not of trainable digital
computation. It is why grid cells are hexagonal, why graphene has Dirac cones, why the Kitaev spin
liquid exists. Its specialness is real, repeatedly confirmed, and — in the quantum/topological-order
regime — demonstrable on today's hardware. What the evidence does **not** support is the leap from
"the hexagon is physically special" to "a hexagonal network produces emergent intelligence." That
bridge was never load-bearing; intelligence, where this project found it, lived in *learning on* a
scaffold, not in the scaffold's shape.

So the six-year conviction was neither vindicated as stated nor empty. It was **aimed one category
off**: the honeycomb's power is a property of physical substrates (analog, photonic, and above all
quantum-topological), realised for free by the physics itself — not a free lunch for a GPU. The
"needs qubits" instinct, recorded early, was the most accurate part of the vision.

## What is real, and what it is worth

**Real and durable:** the equivariance sample-efficiency results; the multi-device, error-barred QPU
demonstrations of honeycomb chirality, symmetry preservation, and Kitaev topological order (flux
conservation, localized anyons, abelian braiding). These are rigorous, reproducible, portfolio-grade
— genuine evidence of someone who can run geometric ML and topological-order experiments on real
quantum hardware and report them honestly.

**Honest limits:** the physics is textbook (not novel research); the braiding is abelian, not the
non-Abelian (Ising-anyon) regime that universal topological computation needs; and useful-scale
topological-order hardware is field-scale and years out — not something one person funds.

## The forward path (honest)

- **The dream is not hostage to anything.** It can be explored for ~free on the local GPU and on
  IBM's quantum hardware as that hardware matures — no funding gate, no rush. The one genuine frontier
  left is the **non-Abelian phase** (Kitaev in a field → a chiral Majorana edge that would unify the
  topological-order thread with the Haldane chirality thread). It is qubit-scale-gated; revisit when
  the hardware grows.
- **Capital runs on a separate track, and that is correct.** Escaping dev work and funding the future
  go through the consulting/contracting path — and the work above is now a real credibility asset for
  it, not a detour from it.
- **The integrity is the result.** A six-year idea was given a rigorous, imaginative, fair trial. It
  was falsified where it was wrong, confirmed precisely where it was right, and the line between the
  two was drawn without flinching or fabricating. That is what makes the parts that *did* survive
  worth trusting.

*Artifacts: `runs/KITAEV_TOPOLOGICAL_ORDER.md`, `runs/PHYSICS_COMPUTE_CLEAN_REDO.md`,
`figures/kitaev_topological_order.png`, `PAPER_DRAFT.md`, and the `scripts/` that produced every
number — each read from a completed run, none written ahead of its evidence.*
