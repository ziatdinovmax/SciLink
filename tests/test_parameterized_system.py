"""Engine-neutral force-field contract: ParameterizedSystem + the Interchange
export helper.

Offline: the contract and the export-helper module import with only stdlib (no
openff/interchange/parmed). The actual Interchange export is gated behind the
``scilink[ff]`` extra and verified live elsewhere; here we assert the contract
round-trips and the helper fails with an actionable error when the deps are
absent (or, when they are present, on the missing payload — never silently).
"""

from __future__ import annotations

from scilink.agents.sim_agents._parameterized_system import (
    ComponentSpec,
    ParameterizedSystem,
)
from scilink.agents.sim_agents import _engine_inputs


def test_parameterized_system_round_trip():
    ps = ParameterizedSystem(
        backend="openff-2.2.0",
        source_format="interchange",
        n_atoms=1518,
        total_charge=0.0,
        components=[ComponentSpec("water", "O", 500),
                    ComponentSpec("Na+", "[Na+]", 9, 1.0)],
        coordinates_file="structure.extxyz",
        box=[24.0, 24.0, 24.0],
        interchange_path="ic.json",
    )
    back = ParameterizedSystem.from_json(ps.to_json())
    assert back == ps
    assert isinstance(back.components[1], ComponentSpec)
    assert back.components[1].name == "Na+"
    assert back.amber_files == ("", "")  # tuple preserved through JSON


def test_amber_payload_round_trip():
    ps = ParameterizedSystem(
        backend="amber-ff19SB+gaff2", source_format="amber",
        n_atoms=22, total_charge=0.0, amber_files=("sys.prmtop", "sys.inpcrd"),
    )
    back = ParameterizedSystem.from_json(ps.to_json())
    assert back.source_format == "amber"
    assert back.amber_files == ("sys.prmtop", "sys.inpcrd")


def test_write_md_inputs_fails_loudly_without_payload(tmp_path):
    # Either openff is absent (ImportError naming scilink[ff]) or present but the
    # interchange payload is missing (FileNotFoundError). Never a silent success.
    ps = ParameterizedSystem(
        backend="openff-2.2.0", source_format="interchange", n_atoms=3,
        total_charge=0.0, interchange_path=str(tmp_path / "missing.json"),
    )
    import pytest
    with pytest.raises((ImportError, FileNotFoundError)):
        _engine_inputs.write_md_inputs(ps, "lammps", str(tmp_path / "out"))


def test_unknown_source_format_rejected(tmp_path):
    ps = ParameterizedSystem(backend="x", source_format="bogus", n_atoms=1,
                             total_charge=0.0)
    import pytest
    # ValueError if openff present; ImportError if absent — both are "loud".
    with pytest.raises((ValueError, ImportError)):
        _engine_inputs.write_md_inputs(ps, "lammps", str(tmp_path / "out"))
