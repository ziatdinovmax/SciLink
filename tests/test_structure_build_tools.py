"""Offline tests for the codegen environment-probe helpers.

`detect_structure_build_tools` reports which optional structure-building
libraries are importable so the generation prompt can tell the LLM what is
available instead of guessing. No network, no real libraries required — the
probe primitives (`importlib.util.find_spec`, `shutil.which`) are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

from scilink.agents.sim_agents.utils import (
    detect_structure_build_tools,
    format_available_tools_block,
)


# --- format_available_tools_block --------------------------------------------

def test_format_block_empty_is_blank():
    # Nothing available → no prompt block at all (prompt left unchanged).
    assert format_available_tools_block({}) == ""
    assert format_available_tools_block({"packmol": False, "rdkit": False}) == ""


def test_format_block_lists_only_available_sorted():
    block = format_available_tools_block(
        {"packmol": True, "rdkit": False, "openmm": True, "mbuild": False}
    )
    assert "openmm, packmol" in block  # sorted, only the True ones
    assert "rdkit" not in block
    assert "mbuild" not in block


def test_format_block_states_prefer_principle():
    # The one load-bearing instruction: prefer a purpose-built lib over ad-hoc code.
    block = format_available_tools_block({"packmol": True})
    assert "prefer" in block.lower()
    assert "AVAILABLE LIBRARIES" in block


# --- detect_structure_build_tools --------------------------------------------

def _fake_find_spec(present_modules):
    def _spec(name):
        return object() if name in present_modules else None
    return _spec


def test_detect_packmol_requires_wrapper_and_binary():
    # Wrapper importable but binary missing → packmol unavailable.
    with patch(
        "scilink.agents.sim_agents.utils.importlib.util.find_spec",
        _fake_find_spec({"pymatgen", "pymatgen.io.packmol"}),
    ), patch("scilink.agents.sim_agents.utils.shutil.which", return_value=None):
        avail = detect_structure_build_tools()
    assert avail["pymatgen"] is True
    assert avail["packmol"] is False  # wrapper present, binary absent


def test_detect_packmol_available_when_both_present():
    with patch(
        "scilink.agents.sim_agents.utils.importlib.util.find_spec",
        _fake_find_spec({"pymatgen", "pymatgen.io.packmol", "rdkit"}),
    ), patch(
        "scilink.agents.sim_agents.utils.shutil.which",
        return_value="/usr/bin/packmol",
    ):
        avail = detect_structure_build_tools()
    assert avail["packmol"] is True
    assert avail["rdkit"] is True
    assert avail["openmm"] is False
    assert avail["mbuild"] is False


def test_detect_fail_closed_on_probe_error():
    # A probe that raises (e.g. broken parent package) → reported unavailable,
    # never propagated.
    def _boom(name):
        raise ImportError("broken parent package")

    with patch(
        "scilink.agents.sim_agents.utils.importlib.util.find_spec", _boom
    ), patch("scilink.agents.sim_agents.utils.shutil.which", return_value=None):
        avail = detect_structure_build_tools()
    assert all(v is False for v in avail.values())
