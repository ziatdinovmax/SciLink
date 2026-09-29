"""Potential selection runs before classical FF parameterization (issue #666).

A packed-box ``components.json`` manifest exists whether the run ends up on a
force field or an MLIP. The workflow must choose the family first and only
parameterize classically when that choice is ``force_field``; otherwise a
metal slab dies in OpenFF before the selector ever runs. No LLM, no OpenFF.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402
import scilink.agents.sim_agents.potential_selection as ps  # noqa: E402
import scilink.agents.sim_agents.force_field_agent as ffa  # noqa: E402
import scilink.agents.sim_agents._engine_inputs as ei  # noqa: E402


@pytest.fixture
def md_case(tmp_path, monkeypatch):
    """A manifest-bearing MD structure with every LLM/OpenFF seam faked.

    Returns a dict the test mutates: ``family`` (what the fake selector
    answers) and ``calls`` (ordered log of selector / FF / generation calls).
    """
    case = {"family": "force_field", "calls": []}

    structure = tmp_path / "structure.extxyz"
    structure.write_text("dummy")
    (tmp_path / "components.json").write_text(json.dumps(
        {"components": [{"name": "water", "smiles": "O", "count": 100}]}))

    def fake_select(**kw):
        case["calls"].append("select")
        return {"family": case["family"], "reasoning": "test", "source": "llm"}

    class FakePSystem:
        backend, n_atoms, total_charge = "openff", 300, 0.0

    class FakeFFAgent:
        def __init__(self, **kw):
            pass

        def parameterize(self, **kw):
            case["calls"].append("parameterize")
            return FakePSystem()

    def fake_generate_inputs(**kw):
        case["calls"].append("generate")
        case["gen_kwargs"] = kw
        return {"status": "success", "input_files": {"run.lammps": "run"},
                "entry_file": "run.lammps"}

    monkeypatch.setattr(ps, "select_potential_family", fake_select)
    monkeypatch.setattr(ffa, "ForceFieldAgent", FakeFFAgent)
    monkeypatch.setattr(ei, "write_md_inputs",
                        lambda p, s, w: {"structure_file": str(structure),
                                         "force_field_files": {}})
    monkeypatch.setattr(sp, "_generate_inputs", fake_generate_inputs)
    case["structure"] = structure
    case["out"] = tmp_path / "out"
    return case


def _run(case):
    return sp.run_complete_workflow(
        "MD of a test system", scale="molecular_dynamics", software="lammps",
        structure_file=str(case["structure"]), output_dir=str(case["out"]),
        validate=False, api_key="k",
    )


def test_selector_runs_before_parameterization(md_case):
    _run(md_case)
    assert md_case["calls"].index("select") < md_case["calls"].index("parameterize")


def test_force_field_choice_parameterizes_classically(md_case):
    result = _run(md_case)
    assert "force_field" in result["steps_completed"]
    assert result["potential_selection"]["family"] == "force_field"
    assert md_case["gen_kwargs"]["potential_selection"]["family"] == "force_field"


def test_mlip_choice_skips_classical_parameterization(md_case):
    md_case["family"] = "mlip"
    result = _run(md_case)
    assert "parameterize" not in md_case["calls"]
    assert "force_field" not in result["steps_completed"]
    assert result["potential_selection"]["family"] == "mlip"
    assert md_case["gen_kwargs"]["potential_selection"]["family"] == "mlip"
    assert result["final_status"] != "failed_force_field"
    # The MLIP path must not be mistaken for "no manifest, nothing to do".
    assert not any("no components.json" in w
                   for w in result.get("warnings", []))


def test_selector_called_once_per_workflow(md_case):
    _run(md_case)
    assert md_case["calls"].count("select") == 1


def test_generate_inputs_honours_passed_selection(tmp_path, monkeypatch):
    """A precomputed decision is used as-is; no second selection is made."""
    def boom(**kw):
        raise AssertionError("selector must not run again")
    monkeypatch.setattr(ps, "select_potential_family", boom)
    seen = {}
    monkeypatch.setattr(sp, "_generate_mlip_inputs",
                        lambda **kw: seen.setdefault("mlip", kw) or
                        {"status": "success", "input_files": {}})
    monkeypatch.setattr(sp, "_generate_classical_md_inputs",
                        lambda **kw: seen.setdefault("classical", kw) or
                        {"status": "success", "input_files": {}})

    out = sp._generate_inputs(
        scale="molecular_dynamics", software="lammps", method="llm",
        structure_file=str(tmp_path / "s.xyz"), request="r",
        output_dir=str(tmp_path), api_key="k", base_url=None, model_name="m",
        potential_selection={"family": "mlip", "source": "llm"},
    )
    assert "mlip" in seen and "classical" not in seen
    assert out["potential_selection"]["family"] == "mlip"
