"""Force-field step wiring in the MD pipeline.

Offline: the components-manifest loader (the bridge from a packed box to
per-species chemistry) behaves correctly, and the condensed structure skill
instructs codegen to emit the manifest. The full run_complete_workflow MD path
(structure -> FF -> typed data file -> lmp) is verified live in the [ff] env.
"""

from __future__ import annotations

import json

from scilink.agents.sim_agents.simulation_pipeline import _load_components_manifest


def test_manifest_absent_returns_none(tmp_path):
    sp = tmp_path / "structure.extxyz"
    sp.write_text("x")
    assert _load_components_manifest(str(sp)) is None


def test_manifest_loaded_when_present(tmp_path):
    sp = tmp_path / "structure.extxyz"
    sp.write_text("x")
    (tmp_path / "components.json").write_text(json.dumps(
        {"components": [{"name": "water", "smiles": "O", "count": 500}]}))
    m = _load_components_manifest(str(sp))
    assert m["components"][0] == {"name": "water", "smiles": "O", "count": 500}


def test_empty_components_treated_as_no_manifest(tmp_path):
    sp = tmp_path / "structure.extxyz"
    sp.write_text("x")
    (tmp_path / "components.json").write_text(json.dumps({"components": []}))
    assert _load_components_manifest(str(sp)) is None


def test_condensed_skill_requires_components_manifest():
    from scilink.skills.loader import load_skill
    impl = load_skill("condensed", domain="structure_generation").get("implementation", "")
    assert "components.json" in impl


def test_orthogonalize_zero_tilt_drops_line(tmp_path):
    # A zero-tilt 'xy xz yz' line (always written by Interchange) is removed so
    # LAMMPS reads an orthogonal box — avoids the triclinic/PPPM ordering trap.
    from scilink.agents.sim_agents._engine_inputs import _orthogonalize_zero_tilt
    data = tmp_path / "system.data"
    data.write_text(
        "0 36 xlo xhi\n0 36 ylo yhi\n0 36 zlo zhi\n"
        "0.0 0.0 0.0 xy xz yz\n\nMasses\n\n1 1.008\n"
    )
    _orthogonalize_zero_tilt(str(data))
    assert "xy xz yz" not in data.read_text()


def test_orthogonalize_keeps_nonzero_tilt(tmp_path):
    # A genuinely triclinic cell (nonzero tilt) must be preserved.
    from scilink.agents.sim_agents._engine_inputs import _orthogonalize_zero_tilt
    data = tmp_path / "system.data"
    data.write_text(
        "0 36 xlo xhi\n0 36 ylo yhi\n0 36 zlo zhi\n"
        "1.5 0.0 0.0 xy xz yz\n\nMasses\n\n1 1.008\n"
    )
    _orthogonalize_zero_tilt(str(data))
    assert "1.5 0.0 0.0 xy xz yz" in data.read_text()
