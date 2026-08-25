"""OpenFF force-field skill: registration, discovery, guarded deps, skill prose.

Offline (no openff/interchange): the build_interchange tool's module imports
(its openff imports are lazy), it registers a TOOL_SPEC discoverable via the
registry for the ``openff`` skill, calling it without the ``[ff]`` deps raises an
actionable error, and the skill markdown parses into the fixed section
vocabulary. The actual Interchange build + LAMMPS export is verified live in the
``[ff]`` env.
"""

from __future__ import annotations

import pytest


def test_build_interchange_module_and_toolspec():
    from scilink.skills.force_field.openff.build_interchange import (
        build_interchange,  # noqa: F401
        TOOL_SPEC,
    )
    assert TOOL_SPEC.name == "build_interchange"
    assert "simulation" in TOOL_SPEC.agents
    assert "coordinates_file" in TOOL_SPEC.required


def test_registry_resolves_build_interchange():
    from scilink.skills._shared._registry import get_tool_function
    fn = get_tool_function("build_interchange", active_skills=["openff"])
    assert fn.__name__ == "build_interchange"


def test_build_interchange_guards_missing_ff_deps():
    from scilink.skills.force_field.openff.build_interchange import build_interchange
    with pytest.raises(ImportError, match=r"conda-forge"):
        build_interchange([{"name": "w", "smiles": "O", "count": 1}],
                          "/tmp/does_not_exist.extxyz")


def _write_box(tmp_path, symbols, positions, cell=20.0):
    """Write a tiny periodic extxyz for live parameterization tests."""
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from ase.io import write
    path = str(tmp_path / "structure.extxyz")
    write(path, Atoms(symbols=symbols, positions=positions,
                      cell=[cell, cell, cell], pbc=True))
    return path


def test_build_interchange_supplements_metal_ion_vdw(tmp_path):
    # Live ([ff] env): a divalent metal (Zn2+) that Sage has no vdW for is
    # parameterized via the bundled ion supplement, NAGL is skipped for the
    # monatomic ions, and the net-neutral system builds without the gate firing.
    pytest.importorskip("openff.toolkit")
    pytest.importorskip("openff.interchange")
    from scilink.skills.force_field.openff.build_interchange import build_interchange
    coords = _write_box(tmp_path, ["Zn", "Cl", "Cl"],
                        [[10, 10, 10], [7, 10, 10], [13, 10, 10]])
    res = build_interchange(
        [{"name": "Zn2+", "smiles": "[Zn+2]", "count": 1},
         {"name": "Cl-", "smiles": "[Cl-]", "count": 2}],
        coords, working_dir=str(tmp_path))
    assert res["n_atoms"] == 3
    assert abs(res["total_charge"]) < 1e-6


def test_build_interchange_gate_rejects_unparameterized_cation(tmp_path):
    # Live ([ff] env): a cation absent from the supplement (Fe2+) gets no vdW;
    # the completeness gate refuses to emit the half-parameterized system and
    # names the offender rather than producing a zero-LJ point charge.
    pytest.importorskip("openff.toolkit")
    pytest.importorskip("openff.interchange")
    from scilink.skills.force_field.openff.build_interchange import build_interchange
    coords = _write_box(tmp_path, ["Fe", "Cl", "Cl"],
                        [[10, 10, 10], [7, 10, 10], [13, 10, 10]])
    with pytest.raises(ValueError, match=r"no Lennard-Jones parameters.*Fe"):
        build_interchange(
            [{"name": "Fe2+", "smiles": "[Fe+2]", "count": 1},
             {"name": "Cl-", "smiles": "[Cl-]", "count": 2}],
            coords, working_dir=str(tmp_path))


def test_openff_skill_loads_with_sections():
    from scilink.skills.loader import load_skill
    sk = load_skill("openff", domain="force_field")
    present = [s for s in ("overview", "planning", "implementation",
                           "interpretation", "validation") if sk.get(s)]
    # All five fixed sections authored.
    assert set(present) == {"overview", "planning", "implementation",
                            "interpretation", "validation"}
