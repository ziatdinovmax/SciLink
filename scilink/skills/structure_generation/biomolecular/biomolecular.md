---
description: Biomolecular structure generation — proteins, peptides, nucleic acids, lipid membranes, and protein–ligand complexes fetched from the PDB (RCSB) or built from sequence via biotite / Biopython / PDBFixer / PeptideBuilder, written as a PDB. For MD (GROMACS, AMBER, OpenMM) or QM/MM.
output_format: pdb
---
## Overview

Build **biomolecular** starting structures for MD (GROMACS, AMBER, OpenMM) or QM/MM: proteins,
peptides, nucleic acids (DNA/RNA), lipid membranes, and protein–ligand complexes. Output a
**PDB** — it preserves chains, residue names/numbers, and atom names, which the downstream
force-field step needs. The goal is a complete, chemically sensible, *cleaned* biomolecule with
correct chains/residues — a sound starting point, not a solvated/parameterized system.

For an isolated small molecule use `molecular`; for a generic solvent box use `condensed`; for
crystalline solids use `crystal`.

**Scope boundary (important):** this step produces the cleaned solute coordinates only.
**Explicit solvation, counter-ions, protonation/atom typing, and force-field assignment are the
downstream force-field step** — for AMBER/LAMMPS that is AmberTools (`pdb4amber`, `tleap`
`solvatebox`/`addIonsRand`, `antechamber`), covered by the **`amber` force-field skill**
(`force_field/amber`). Do NOT solvate, add ions, or add force-field hydrogens here unless the
request explicitly asks for a pre-solvated coordinate box.

## Planning

Identify the system and how to obtain it:

1. **Known structure (PDB ID):** fetch from RCSB (biotite or Biopython). Decide which chain(s)
   to keep, whether to keep/strip crystallographic waters and hetero atoms (ligands, ions), and
   how to resolve alternate locations (keep altloc 'A').
2. **From sequence:** build a peptide / protein (e.g. PeptideBuilder for a backbone, or a
   secondary-structure-aware builder) or a nucleic-acid duplex.
3. **Complex:** combine a cleaned receptor with a ligand (the ligand geometry can come from the
   `molecular` skill / RDKit).
4. **Membrane:** specify lipid composition + leaflet counts; a packed bilayer (note: CHARMM-GUI
   / packmol-memgen style builders are the usual route — produce a coordinate PDB here).
5. **Missing atoms / residues:** completing missing *heavy* atoms and short gaps is fine
   (PDBFixer); do NOT silently fabricate large missing loops — report the gap instead.
6. **Protonation / pH:** note the intended pH, but leave detailed protonation to the FF step
   (tleap / PDBFixer in the `amber` pipeline), since residue protonation states (HID/HIE/HIP,
   ASH/GLH) are assigned there.

## Implementation

Prefer whatever is installed (the generation loop falls back if an import is missing):

- **biotite** (`biotite.database.rcsb` + `biotite.structure.io.pdb`) or **Biopython**
  (`Bio.PDB.PDBList` / `MMCIFParser` / `PDBParser`, `PDBIO`) — fetch, parse, select chains,
  filter waters/hetero/altlocs, and write PDB.
- **PDBFixer** — find/add missing heavy atoms and residues, remove heterogens, build a complete
  clean PDB (its `addSolvent` exists but leave solvation to the FF step by default).
- **PeptideBuilder** — build a peptide PDB from a sequence (+ φ/ψ if specified).
- **OpenMM `Modeller`** / **MDAnalysis** — manipulation / selection helpers.
- For ligands in a complex, build the ligand via the `molecular` skill (RDKit) and merge.
- **Output:** write a PDB, e.g. `Bio.PDB.PDBIO().save("structure.pdb")` (or biotite's PDB
  writer), then print the exact line `STRUCTURE_SAVED:structure.pdb`.

## Validation

A generated **biomolecular** structure is a valid MD/QM/MM starting point when:

- It parses, and the **chain(s) / residue composition** match the request (the right sequence,
  or the requested PDB ID's biological unit; requested ligand/cofactor present).
- **Standard residues** (or clearly-explained non-standard ones); residue/atom **naming** is
  PDB-conventional so the downstream force-field step can type it.
- **No missing heavy atoms** in the residues that ARE present; no large silently-fabricated
  missing loops (gaps should be reported, not invented).
- **Reasonable backbone geometry:** a continuous chain (no broken connectivity), no gross steric
  clashes beyond what energy minimization fixes.

**Do NOT flag (these belong to the downstream force-field / prep step):**

- Absence of explicit solvent water, ions, or a periodic box — added at the FF step
  (see the `amber` skill).
- Missing hydrogens / unset protonation states — assigned by tleap / PDBFixer downstream.
- The need for energy minimization or equilibration.

**Flag as substantive:** wrong or incomplete sequence, missing chains, large missing segments
that were silently fabricated, broken chain connectivity, a requested ligand/cofactor that's
absent, or severe steric clashes that minimization won't resolve.
