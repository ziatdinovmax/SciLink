"""Structure generation emits engine-neutral coordinate formats.

The structure-generation stage produces portable coordinates; the engine-native
input (a VASP POSCAR, a LAMMPS data file, ...) is written downstream by the
engine step. So no structure_generation skill should declare a VASP-specific
POSCAR output format — periodic classes use extended XYZ, isolated molecules
plain XYZ, biomolecules PDB. Offline (reads skill frontmatter only).
"""

from __future__ import annotations

from scilink.skills.loader import load_skill


def _output_format(structure_class: str):
    meta = load_skill(structure_class, domain="structure_generation").get("meta") or {}
    return meta.get("output_format")


def test_output_formats_are_engine_neutral():
    expected = {
        "crystal": "extxyz",      # periodic; carries cell + PBC + constraints
        "condensed": "extxyz",    # periodic boxes for classical MD
        "molecular": "xyz",       # isolated molecule, non-periodic
        "biomolecular": "pdb",    # standard biomolecular format
    }
    for cls, fmt in expected.items():
        assert _output_format(cls) == fmt, f"{cls} output_format={_output_format(cls)!r}"


def test_no_class_emits_a_vasp_poscar():
    # POSCAR is VASP-specific — structure generation must stay engine-neutral.
    for cls in ("crystal", "condensed", "molecular", "biomolecular"):
        assert (_output_format(cls) or "").lower() != "poscar"
