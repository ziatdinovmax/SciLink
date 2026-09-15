---
description: Crystalline / periodic structure generation — bulk crystals, supercells, point defects (vacancies, substitutions, interstitials), surface slabs, and 2D monolayers via ASE / pymatgen / Materials Project, written as engine-neutral extended XYZ.
output_format: extxyz
---
## Overview

Build **periodic crystalline** structures for periodic-DFT and solid-state MD: bulk unit
cells, supercells, point defects, surface slabs, and (extracted) 2D monolayers. Prefer ASE
and pymatgen primitives, and fetch known compounds from the Materials Project when a key is
available. Output engine-neutral extended XYZ (cell + PBC + coordinates, and any
selective-dynamics constraints as a per-atom column); the downstream engine step writes the
engine-native input (POSCAR, etc.) from these coordinates.
The goal is a periodic cell that is stoichiometric, physically reasonable (no atom overlaps),
and faithful to the requested phase / supercell / defect.

This is the default structure scale. For grain boundaries use the `aimsgb` skill; for
isolated molecules, solvated boxes, or biomolecules use the `molecular` / `condensed` /
`biomolecular` skills instead.

## Planning

Identify the build type first:

1. **Known bulk compound** (named, or formula + phase): fetch from the Materials Project
   (pymatgen `MPRester`) when a key is available; otherwise build a known prototype
   (rocksalt, zincblende, wurtzite, fcc, bcc, diamond, perovskite, rutile, fluorite) with
   `ase.build.bulk` or pymatgen, using literature lattice parameters.
2. **Supercell**: replicate the conventional *or* primitive cell (match the request) by the
   requested nx×ny×nz, preserving stoichiometry.
3. **Point defect**: vacancy (remove one site), substitution (change one site's species),
   interstitial (add a site at a sensible void). Track the resulting counts and place the
   defect as requested (e.g. "adjacent" → nearest neighbor).
4. **Surface slab**: cleave the requested Miller surface, set the layer count, add vacuum
   (≥ 12–15 Å) along the surface normal (orient it along c), and optionally freeze bottom
   layers (selective dynamics) to mimic bulk.
5. **2D monolayer**: extract one layer from the bulk by locating the van-der-Waals gap along
   the stacking axis; **verify the extracted layer's stoichiometry** — layered/monoclinic
   systems (e.g. CrPS₄) are easy to mis-extract.

Decide conventional vs primitive cell from the request (default: conventional unless told
otherwise). Note the target spacegroup/phase so it can be checked later.

## Implementation

- Prefer **pymatgen** (`Structure`, `SpacegroupAnalyzer`, supercell transforms, defect
  tooling) and **ASE** (`bulk`, `surface`, `make_supercell`, replication). Use the
  **Materials Project** (`mp_api.client.MPRester`) for known compounds when an MP key is set.
- **Supercells:** `atoms *= (nx, ny, nz)` (ASE) or `structure.make_supercell([nx,ny,nz])`
  (pymatgen). Confirm atom count = formula-units × cell × replication.
- **Vacancies:** delete the chosen site. **Substitutions:** reassign the site's species.
  **Interstitials:** add a site at a tetrahedral/octahedral void. Re-check species counts after.
- **Slabs:** `ase.build.surface(...)` or pymatgen `SlabGenerator`; centre the slab with
  vacuum (≥ 12–15 Å) so ONLY the surface-normal direction is effectively non-periodic.
  Freeze bottom layers with selective-dynamics flags when a relaxation is implied.
- **Monolayers:** find the layer spacing/gap, select one layer's atoms, and **assert the
  layer composition matches the bulk formula ratio before** building the supercell.
- **Output:** write engine-neutral **extended XYZ**, which carries the cell, PBC, and any
  per-atom constraints (`FixAtoms` → a selective-dynamics column), e.g.
  `ase.io.write("structure.extxyz", atoms, format="extxyz")`, then print the exact line
  `STRUCTURE_SAVED:structure.extxyz`. Do not write a VASP POSCAR here — the engine-native input
  is produced by the downstream engine step from these coordinates.

## Validation

A generated **crystalline / periodic** structure is a valid starting point for periodic DFT
or solid-state MD when:

- It parses, and the **total atom count** matches the request (supercell size × formula units,
  adjusted for any defects).
- **Stoichiometry / composition** matches the requested compound (e.g. a CrPS₄ monolayer must
  keep Cr:P:S = 1:1:4, minus any requested vacancy).
- **No unphysical atom overlaps** — minimum interatomic distance is physical (≳ 0.7 Å and near
  typical bond lengths); no duplicated sites.
- **Periodicity is sensible** — a full 3D periodic cell for bulk; for a slab, ≥ ~12 Å vacuum
  along the surface normal (15 Å common but not strict) and along that direction ONLY.
- **Phase / spacegroup** matches if a specific polymorph was requested (`SpacegroupAnalyzer`).
- **Defects** (vacancies / substitutions / interstitials) are present in the correct number and type.

**Normal for an UNRELAXED crystal — do NOT flag these:**

- Atomic clashes / close contacts (<1.0 Å) at grain boundaries, interfaces, or surface
  terminations — these resolve during relaxation.
- Atoms placed on ideal lattice sites near defects rather than displaced toward relaxed
  positions; absence of explicit defect bond reconstruction.
- Vacuum thicknesses anywhere from ~12 Å upward (15 Å is common but not a strict requirement).
- Minor coordinate-wrap artifacts that don't change the periodic image.

**Flag as substantive:** wrong composition / stoichiometry, wrong supercell size vs the
request, missing requested defects, fundamentally wrong bonding (script bug), or severely
insufficient vacuum (<10 Å) for a slab. Watch for the classic failure: a monolayer / layer
extraction that dropped a sublattice (wrong stoichiometry) or produced a skewed / degenerate
cell.
