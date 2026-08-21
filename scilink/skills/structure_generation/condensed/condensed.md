---
description: Condensed-phase / solvated-box structure generation — liquids, solutions, explicitly-solvated solutes, amorphous cells, and interfaces at a target density via Packmol (incl. pymatgen's wrapper) / OpenMM / mBuild / ASE, written as engine-neutral extended XYZ. For classical MD (LAMMPS, GROMACS, OpenMM).
output_format: extxyz
---
## Overview

Build **periodic condensed-phase** systems for classical MD (LAMMPS, GROMACS, OpenMM): bulk
liquids, solutions, explicitly-solvated solutes, amorphous cells, and liquid/solid interfaces.
Output periodic **extended XYZ** (orthorhombic/cubic cell + PBC + all atoms). The goal is the right
number of molecules at a realistic **density**, packed into a periodic box with no severe
overlaps — a sound *initial* configuration that MD equilibration will relax.

This is for many-molecule periodic boxes. For a single isolated molecule use `molecular`; for
crystalline solids use `crystal`; for proteins / nucleic acids use `biomolecular`.

**Scope boundary:** this step produces *coordinates*. Force-field assignment / atom typing /
charges are a downstream step. When the target is an AMBER-typed system (proteins, or small
molecules needing GAFF), solvation + counter-ions are usually done together with topology
building via AmberTools `tleap` — see the **`amber` force-field skill** (`force_field/amber`) —
rather than packed here. Pack explicitly here for generic liquids / solutions / amorphous cells.

## Planning

1. **Components & counts:** parse the requested species and their numbers or concentration
   (e.g. "256 water", "1 M NaCl in water"). When only a density/box is given, compute counts
   from the target density: ρ = (Σ Nᵢ · Mᵢ) / (N_A · V).
2. **Box size:** choose a cubic/orthorhombic cell giving the target density (e.g. liquid water
   ≈ 1.0 g/cm³ → ~0.033 molecules/Å³). Size it so a solute does not see its own periodic image
   (≳ 10–12 Å of solvent around a solute).
3. **Explicit solvent:** classical MD uses **explicit** solvent — actually place the solvent
   molecules (unlike molecular DFT's implicit model). Add counter-ions to neutralize a charged
   solute when relevant.
4. **Charge neutrality (get the counter-ion counts right):** the box **must be net-neutral**, and
   the count of each ion follows from *charge balance*, not from a 1:1 pairing. Balance on
   **total charge**, weighting every ion by its valence:

   > Σ (nᵢ · zᵢ) over **all** cations  =  Σ (nⱼ · |zⱼ|) over **all** anions.

   A multivalent ion needs **|z| monovalent counter-ions**, and a shared counter-ion neutralizes
   *every* cation at once — the commonest mistake is pairing it with only one species. Worked
   example (aqueous NaOH + a Mg²⁺ salt): 4 Na⁺ (+4) and 1 Mg²⁺ (+2) give **+6**; with 2 Cl⁻
   present (−2), the hydroxide count must close the rest → **4 OH⁻** (−4), for 4 OH⁻ + 2 Cl⁻ = −6.
   Two OH⁻ (a naive 1-per-Na, ignoring Mg²⁺) leaves the box at **+2** and any neutrality guard
   fails the build. **Before packing, compute the signed charge sum explicitly and adjust the most
   abundant counter-ion until it is exactly zero.**
5. **Packing:** plan a non-overlapping random packing. Equilibration (NPT/NVT) is a downstream
   MD step — here you only need a valid, non-overlapping start near the target density.

## Implementation

Prefer whatever is installed (check the AVAILABLE LIBRARIES list in the prompt). For any
**multi-component** box — solutions, ions in solvent, mixtures — strongly prefer a dedicated
packing tool (Packmol, OpenMM, mBuild) over a hand-rolled placement loop: these enforce the
minimum separation across **all** atom pairs, whereas ad-hoc loops routinely check only
heavy-atom centres and let hydrogens of neighbouring molecules overlap. Reserve the ASE/numpy
fallback for when no packing library is available.

- **Packmol** — the de-facto packing tool, and the default choice for solvated / multi-component
  boxes. Drive it from Python via **pymatgen** (`pymatgen.io.packmol.PackmolBoxGen`), which writes
  the Packmol input, runs it, and returns a structure; or write a Packmol input file and call the
  `packmol` binary via `subprocess` (place N copies per species with a `tolerance` ≈ 2.0 Å minimum
  separation). Packmol checks every atom pair, so ions and solvent are packed without the
  hydrogen-overlap clashes a manual loop produces.
- **OpenMM `Modeller.addSolvent()`** (often with **PDBFixer**) — adds a water box + neutralizing
  ions in a few lines; convenient for solvating a solute.
- **mBuild (MoSDeF)** — programmatic construction of complex / multi-component / polymeric boxes
  (`mbuild.fill_box`).
- **ASE / numpy fallback** — build each molecule once (ASE / RDKit), then insert copies at random
  positions/orientations into the cell, rejecting placements closer than a minimum distance. The
  rejection test must compare **every atom** of the trial molecule (hydrogens included) against
  **every** already-placed atom under the minimum image — checking only heavy-atom or
  molecular-centre distances lets light atoms of neighbouring molecules interpenetrate. Wrap each
  molecule into the cell as a rigid unit (shift by one reference atom's image), never per-atom
  (`positions % L` applied atom-by-atom splits a molecule across the boundary).
- **Neutrality check (compute it, don't just assert it):** derive each ion count from the
  charge-balance equation above, then, in the script, compute the signed charge sum
  `Q = Σ nᵢ zᵢ` from the final composition *before* packing. If `|Q| > 0` (e.g. a multivalent ion
  left it non-zero), **correct the counts** — add/adjust the most abundant counter-ion to reach
  `Q == 0` and proceed — rather than raising and exiting non-zero, which wastes a whole build.
  Only fail the build if neutrality cannot be reached (a genuinely ill-posed request).
- **Cell & PBC:** set an orthorhombic/cubic `cell` and `pbc=True` sized to the target density.
- **Output:** write engine-neutral periodic **extended XYZ** (carries the cell + PBC), e.g.
  `ase.io.write("structure.extxyz", atoms, format="extxyz")`, then print the exact
  line `STRUCTURE_SAVED:structure.extxyz`. Do not write a VASP POSCAR here — the engine-native
  input (e.g. a LAMMPS data file) is produced by the downstream force-field / engine step.
- **Components manifest (required):** also write `components.json` next to the structure, listing
  each distinct species and its count **in the same order the atoms appear in the coordinate
  file**, with a SMILES per species — e.g.
  `{"components": [{"name": "water", "smiles": "O", "count": 500}, {"name": "Na+", "smiles":
  "[Na+]", "count": 9}, {"name": "Cl-", "smiles": "[Cl-]", "count": 9}]}`. The downstream
  force-field step needs this to know which molecule each atom belongs to (a packed box is only
  coordinates); the component order and counts must match the packing exactly.

## Validation

A generated **condensed-phase box** is a valid MD starting configuration when:

- It parses, and the **composition / molecule counts** match the request (and the box is
  net-neutral when ions are involved).
- **Density** is within ~10–20 % of the target / a physically realistic value for the phase
  (not vacuum-sparse, not impossibly dense).
- **Periodic box:** full 3D PBC with sensible, large-enough cell dimensions (a solute has
  ≳ 10 Å of solvent to its periodic image).
- **No severe atom overlaps** (min pairwise distance ≳ 0.7 Å); molecules are intact, not
  inter-penetrating.

**Do NOT flag (MD equilibration resolves these):**

- Modest close contacts or a not-yet-equilibrated, slightly high-energy packing.
- Lack of a relaxed radial distribution / perfect spacing — that's what the MD run produces.

**Flag as substantive:** wrong composition or molecule counts, density wildly off (too dense →
unresolvable overlaps; too sparse → vacuum voids / unintended interface), missing PBC, a charged
box that should be neutral, demixed/empty regions, or broken (inter-penetrating) molecules.
