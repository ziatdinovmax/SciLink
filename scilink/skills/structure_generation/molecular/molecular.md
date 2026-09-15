---
description: Isolated-molecule / small-cluster structure generation — single molecules, ions, radicals, and non-covalent complexes via RDKit / ASE / PubChem, written as a Cartesian XYZ. For molecular DFT (PySCF, ORCA, Gaussian, NWChem).
output_format: xyz
---
## Overview

Build **isolated, non-periodic** molecular structures for molecular DFT (PySCF, ORCA,
Gaussian, NWChem): single molecules, ions, radicals, and small non-covalent complexes /
clusters. Output a Cartesian **XYZ** file (Å). The goal is a chemically correct molecule —
right formula and connectivity, all hydrogens explicit, a sensible relaxed-ish 3D conformer
with no atom clashes — suitable as an initial geometry for a gas-phase or implicitly-solvated
DFT calculation.

This is for *molecules*, not periodic solids. For bulk crystals / slabs use the `crystal`
skill; for solvated boxes / liquids use `condensed`; for proteins / nucleic acids use
`biomolecular`.

## Planning

Identify the species from the request (common name, SMILES, formula, or InChI) and:

1. **Single molecule:** build from SMILES (preferred for organics) or from a small-molecule
   database (`ase.build.molecule` G2 set for very common species like H2O, CH4, C6H6).
2. **Ion / radical:** note the **net charge and spin multiplicity** from the request — these
   do not change the geometry-building step but MUST be recorded for the downstream DFT input
   (e.g. an OH⁻ vs OH• differ in electrons, not in how you place atoms).
3. **Conformer:** generate a low-energy 3D conformer (ETKDG embedding + a quick force-field
   pre-optimization). One reasonable conformer is enough; DFT will relax it.
4. **Complex / cluster:** build each fragment, then place them at a sensible non-covalent
   separation (~2.5–3.5 Å contact) in the requested arrangement.

**Solvation:** molecular DFT almost always uses an **implicit** continuum solvent applied at
the DFT step — do NOT add explicit solvent molecules unless the request explicitly asks for a
microsolvated cluster. Build the bare solute.

## Implementation

Prefer whatever is installed (the generation loop falls back if an import is missing):

- **RDKit (preferred for organics):** `Chem.MolFromSmiles(smiles)` → `Chem.AddHs(mol)` →
  `AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())` → `AllChem.MMFFOptimizeMolecule(mol)`; then
  read the conformer coordinates and elements into an `ase.Atoms`.
- **ASE (common small molecules):** `from ase.build import molecule; atoms = molecule('H2O')`.
- **PubChemPy** (`pubchempy`) — fetch a known compound (and a 3D conformer) by name or CID when
  you don't have a SMILES handy.
- **Open Babel / `pybel`** — SMILES → 3D (`make3D`) and broad format conversion; a good fallback
  when RDKit is unavailable.
- **Always add explicit hydrogens** (RDKit `AddHs`); verify the H count matches the formula.
- **No periodic cell needed** — leave the cell unset (XYZ carries only coordinates). If a code
  later needs a box, that's a downstream concern.
- **Output:** write Cartesian XYZ, e.g. `ase.io.write("structure.xyz", atoms)`, then print the
  exact line `STRUCTURE_SAVED:structure.xyz`.

## Validation

A generated **isolated molecule** is a valid molecular-DFT starting point when:

- It parses, and the **molecular formula / atom counts match the request**, with **all
  hydrogens explicit** (e.g. ethanol must be C2H6O = 9 atoms, not the heavy-atom skeleton).
- **Connectivity / isomer** matches the requested species (right functional groups, right
  regio-/stereochemistry if specified).
- **Sensible 3D geometry:** bond lengths near typical values, no atom overlaps (min pairwise
  distance ≳ 0.7 Å), not artificially planar/collinear unless the molecule genuinely is, no
  dissociated/fragmented fragments for a single molecule.
- For **ions/radicals**, the structure is consistent with the requested charge/multiplicity
  (which is recorded for the DFT step, not encoded in the geometry).

**Do NOT flag (not applicable to an isolated molecule):**

- Absence of a periodic cell, supercell, vacuum padding, or PBC — this is a non-periodic
  molecule by design.
- Absence of explicit solvent — molecular DFT uses implicit solvation at the DFT step.
- A non-global conformer — DFT relaxation refines it; only flag a clearly strained/clashing one.

**Flag as substantive:** missing hydrogens, wrong molecular formula, wrong isomer/connectivity,
fragmented or dissociated geometry, atom overlaps, or a 2D/flattened embed that failed to
generate real 3D coordinates.
