---
description: OpenFF (SMIRNOFF) force-field parameterization for small organic molecules, solvents, ions, and electrolytes — produces an engine-neutral OpenFF Interchange (exports natively to LAMMPS / GROMACS / OpenMM). Charges from NAGL (no AmberTools / OpenEye needed).
---
## Overview

Parameterize a system of **SMILES-defined molecules** with an OpenFF SMIRNOFF
force field (the Sage line, `openff-2.x.offxml`) and emit an engine-neutral
**OpenFF Interchange** — the canonical parameterized system that exports natively
to LAMMPS, GROMACS, and OpenMM. This is the preferred backend for small organic
molecules, solvents, ions, and **liquid electrolytes** (carbonates, glymes,
Li/Na salts), where the chemistry is well-defined by SMILES.

Use OpenFF when the components are small molecules / ions with clear SMILES. For
**proteins, nucleic acids, and lipids** prefer the `amber` backend (ff19SB /
GAFF via tleap). OpenFF does not describe **metals, oxides, or reactive
systems** — those are not classical small-molecule force fields.

Charges come from **NAGL** (OpenFF's graph-net AM1-BCC surrogate), so this path
needs no AmberTools or OpenEye and is fully pip-installable.

## Planning

1. **Components.** Identify each species and its count as SMILES — e.g. water
   `O`, Na⁺ `[Na+]`, Cl⁻ `[Cl-]`, ethylene carbonate `C1COC(=O)O1`, PF₆⁻
   `F[P-](F)(F)(F)(F)F`. The component **order and counts must match the packed
   coordinate file exactly** (the parameterized topology aligns atom-by-atom
   with the coordinates).
2. **Force field.** Default to the latest Sage (`openff-2.2.0.offxml`). For
   **water**, add the matching water model as an extra force field (e.g.
   `tip3p.offxml`) rather than charging water with NAGL. Monatomic ions need an
   ion model in the force-field set when Sage does not cover them.
3. **Periodicity.** A **periodic box is required** — LAMMPS/GROMACS
   electrostatics use PME, which an in-vacuum system cannot express. The packed
   box from structure generation supplies the cell.
4. **Charges.** NAGL by default (no QM). Reserve AM1-BCC (AmberTools) only when a
   specific published charge set must be reproduced.

## Implementation

Use the `build_interchange` tool — do not hand-write OpenFF code. It takes the
component manifest (SMILES + counts, in coordinate order) and the packed
coordinates, parameterizes each component, assigns NAGL charges, builds the
periodic topology aligned to the coordinates (with an atom-count cross-check),
and serializes an Interchange:

```python
build_interchange(
    components=[{"name": "water", "smiles": "O", "count": 500},
                {"name": "Na+", "smiles": "[Na+]", "count": 9},
                {"name": "Cl-", "smiles": "[Cl-]", "count": 9}],
    coordinates_file="structure.extxyz",        # packed box, with cell + PBC
    extra_force_fields=["tip3p.offxml"],         # water model when water present
)  # -> {"interchange_path": ".../system_interchange.json", "n_atoms", "total_charge"}
```

The returned Interchange is engine-neutral; the MD engine's `write_md_inputs`
tool exports it (LAMMPS data file, etc.). Do **not** write engine input files
here — that is the engine skill's job.

## Interpretation

- `UnassignedValenceError` / "no parameters assigned": the SMILES has chemistry
  the force field does not cover (wrong/incomplete SMILES, an ion or metal Sage
  lacks). Fix the SMILES or add the right force-field file (water/ion model).
- Atom-count mismatch between topology and coordinates: the component manifest's
  SMILES/counts/order do not match the packed box — correct the manifest.
- NAGL charge errors: the molecule is outside NAGL's organic training domain
  (e.g. an exotic ion); supply that species' charges another way.

## Validation

- The system is **net-neutral** when ions are present (`total_charge` ≈ 0).
- `n_atoms` matches the coordinate file (the tool enforces this).
- Every component was typed (no unassigned parameters), and a periodic box is set.
