"""Periodic-DFT agent writes the engine-native coordinate file deterministically.

Structure generation emits portable coordinates (extended XYZ); the engine's
coordinate input (e.g. a VASP POSCAR) is written here, exactly, via ASE — not
transcribed by the LLM. Engines whose skill declares no separate coordinate
file (e.g. QE embeds the geometry in its main input) are left untouched.

Offline; no API key (the agent is instantiated via __new__ to bypass the
credentialed constructor — the method under test uses only self.logger).
"""

from __future__ import annotations

import logging

import pytest

from scilink.agents.sim_agents.periodic_dft_agent import PeriodicDFTAgent

pytest.importorskip("ase")
from ase.build import bulk  # noqa: E402
from ase.io import read, write  # noqa: E402


VASP_SKILL = {"skill_sections": {"meta": {"structure_file": "POSCAR",
                                          "structure_format": "vasp"}}}


def _agent():
    a = PeriodicDFTAgent.__new__(PeriodicDFTAgent)  # bypass credentialed __init__
    a.logger = logging.getLogger("test_dft_native_structure")
    return a


def test_native_structure_spec_reads_frontmatter():
    assert PeriodicDFTAgent._native_structure_spec(VASP_SKILL) == ("POSCAR", "vasp")
    assert PeriodicDFTAgent._native_structure_spec({"skill_sections": {"meta": {}}}) == (None, None)
    assert PeriodicDFTAgent._native_structure_spec(None) == (None, None)


def test_inject_overwrites_with_deterministic_poscar(tmp_path):
    si = bulk("Si", "diamond", a=5.43, cubic=True)  # 8-atom conventional cell
    p = tmp_path / "structure.extxyz"
    write(str(p), si, format="extxyz")

    # The LLM emitted a bogus POSCAR alongside a real INCAR; the deterministic
    # write must replace it with exact coordinates.
    result = {"input_files": {"INCAR": "ISMEAR = 0\n", "POSCAR": "garbage\n"}}
    _agent()._inject_native_structure(result, str(p), VASP_SKILL)

    txt = result["input_files"]["POSCAR"]
    assert "garbage" not in txt
    pp = tmp_path / "POSCAR"
    pp.write_text(txt)
    back = read(str(pp), format="vasp")
    assert len(back) == 8
    assert set(back.get_chemical_symbols()) == {"Si"}
    assert abs(back.get_volume() - si.get_volume()) < 1e-6
    assert result["input_files"]["INCAR"] == "ISMEAR = 0\n"  # other files untouched


def test_inject_noop_without_declaration(tmp_path):
    # An engine whose skill declares no separate coordinate file (e.g. QE) is a no-op.
    si = bulk("Si", "diamond", a=5.43, cubic=True)
    p = tmp_path / "structure.extxyz"
    write(str(p), si, format="extxyz")
    result = {"input_files": {"qe.in": "&control\n"}}
    _agent()._inject_native_structure(result, str(p), {"skill_sections": {"meta": {}}})
    assert list(result["input_files"].keys()) == ["qe.in"]
