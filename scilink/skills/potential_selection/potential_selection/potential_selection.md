---
description: Choose the interatomic-potential family — a classical force field or a machine-learning potential (MLIP) — for a molecular-dynamics task, once the structure and its species are known.
---
# Potential selection — classical force field vs machine-learning potential

## overview

A molecular-dynamics run needs an interatomic potential, and there are two
families to draw it from:

- a **classical force field** (via the `force_field` skills — OpenFF/SMIRNOFF
  for organics and common ions, AMBER for biomolecules): cheap, deterministic,
  extensively validated, but only defined where transferable parameters exist,
  and **non-reactive** (bonds cannot break or form);
- a **machine-learning potential** (via the `machine_learning_potentials`
  skills — universal pretrained models such as MACE-MP, CHGNet, Orb, UMA):
  near-DFT accuracy for arbitrary chemistry with no per-system parameterization,
  reactive, but more expensive and less established for any single system.

This choice is made **after** the structure exists — so the species are known
and force-field coverage is a checkable fact rather than a guess at routing
time. It is a separate decision from *which* specific force field or *which*
specific MLIP: once the family is chosen, the corresponding skills pick the
member. The router does not make this choice; it only decides that the task is
molecular dynamics.

## planning

Decide from the system's chemistry and the research goal.

**Choose a classical force field when every species is well covered and the
physics is non-reactive** — this is the default when it applies, because a
validated force field is cheaper and more reproducible than an MLIP:

- molecular liquids, solutions, and electrolytes built from organic molecules
  and common mono-/di-atomic ions (water, alkanes, alcohols, ethers, common
  salts) — SMIRNOFF/OpenFF territory;
- biomolecules (proteins, nucleic acids, lipids) and their aqueous environments
  — AMBER territory;
- polymers of common monomers;
- any system where a standard, transferable force field exists for **all**
  species and the goal is equilibrium structure, transport (diffusion,
  viscosity), or thermodynamics without bond rearrangement.

**Choose a machine-learning potential when a classical force field is missing,
unreliable, or physically inadequate:**

- inorganic solids, metals, alloys, oxides, ceramics, and semiconductors — no
  transferable classical force field, and a hand-built one is rarely defensible;
- surfaces and interfaces involving an inorganic phase (metal–water,
  oxide–water, electrode–electrolyte);
- reactive chemistry — bond breaking/forming, catalysis, decomposition — which a
  classical force field cannot represent at all;
- novel or exotic materials with no established parameters;
- **any system where one or more species lacks classical-force-field coverage**:
  a run is only as good as its worst-parameterized atom, so a single uncovered
  species is enough to prefer an MLIP over a patched-together force field.

**When the system is mixed or the coverage is uncertain — e.g. an organic
molecule adsorbed on a metal, or an electrolyte against an oxide — prefer the
MLIP.** A universal pretrained MLIP degrades gracefully across chemistries,
whereas a classical force field with an uncovered or guessed component fails
silently: it still runs to completion and still produces a plausible-looking
trajectory, which is exactly the failure this decision exists to avoid.

Report the chosen family and a one- or two-sentence justification naming the
species or goal feature that drove the choice.
