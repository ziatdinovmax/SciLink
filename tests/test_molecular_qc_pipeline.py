"""Integration test: the molecular_qc scale drives the refinement loop.

Clones the pattern of ``test_pipeline_refinement_integration.py`` for the new
``molecular_qc`` scale. Input generation is monkeypatched (no LLM, no real
NWChem), but the executor and the entire refinement control path run for real
via a LocalExecutor — proving the molecular_qc branch in ``_generate_inputs``
reaches step-4 refinement exactly like the shipped scales. No API keys, no
cluster, no nwchem binary.
"""

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402
import scilink.agents.sim_agents.critics as critics_mod  # noqa: E402
from scilink.agents.sim_agents.refinement import LocalExecutor  # noqa: E402


def _stub_generation(monkeypatch):
    """Fake _generate_inputs returning a single NWChem deck + entry_file."""
    def fake_generate_inputs(**kw):
        deck = "echo running\n"
        (Path(kw["output_dir"]) / "job.nw").write_text(deck)
        return {
            "status": "success",
            "software": "nwchem",
            "input_files": {"job.nw": deck},
            "entry_file": "job.nw",
        }
    monkeypatch.setattr(sp, "_generate_inputs", fake_generate_inputs)


def test_molecular_qc_executor_path_fail_fix_succeed(tmp_path, monkeypatch):
    _stub_generation(monkeypatch)

    class FakeRunCritic:
        calls = 0

        def __init__(self, **kw):
            pass

        def assess(self, output_dir, research_goal, skill=None, domain=None,
                   fixes_mode="auto", input_files=None, check_observables=False):
            FakeRunCritic.calls += 1
            if FakeRunCritic.calls == 1:
                return {"status": "success", "run_status": "failed",
                        "verdict": "needs_fixes",
                        "suggested_fixes": {"job.nw": "echo fixed\n"}}
            return {"status": "success", "run_status": "succeeded",
                    "verdict": "good", "suggested_fixes": None}

    monkeypatch.setattr(critics_mod, "RunCritic", FakeRunCritic)

    structure = tmp_path / "mol.xyz"
    structure.write_text("3\ndummy\nO 0 0 0\nC 0 0 1.16\nO 0 0 2.32\n")
    out = tmp_path / "out"

    result = sp.run_complete_workflow(
        "optimize the carbamate ion pair and get its enthalpy",
        scale="molecular_qc", software="nwchem",
        structure_file=str(structure),
        output_dir=str(out),
        api_key="fake-do-not-bill",
        model_name="claude-opus-4-6",
        validate=False,
        executor=LocalExecutor(timeout=30),
        run_command="echo {script}",   # exec-safe stand-in for "nwchem {script}"
        autonomy="autonomous",
        max_run_cycles=3,
    )

    # The whole step-4 glue ran for molecular_qc: phases built from the deck,
    # loop drove the real executor, the critic's fix was applied and re-run.
    assert result["scale"] == "molecular_qc"
    assert result["engine"] == "nwchem"
    assert result["final_status"] == "success", result
    assert result["refinement"]["status"] == "success"
    assert FakeRunCritic.calls == 2          # fail, then good
    assert "refinement" in result["steps_completed"]
    assert (out / "job.nw").read_text() == "echo fixed\n"


def test_molecular_qc_no_executor_stops_after_generation(tmp_path, monkeypatch):
    _stub_generation(monkeypatch)
    structure = tmp_path / "mol.xyz"
    structure.write_text("dummy")
    result = sp.run_complete_workflow(
        "prep only",
        scale="molecular_qc", software="nwchem",
        structure_file=str(structure),
        output_dir=str(tmp_path / "out2"),
        api_key="fake-do-not-bill",
        validate=False,
    )
    assert result["final_status"] == "success"
    assert "refinement" not in result          # no execution attempted


def test_molecular_qc_default_engine_is_nwchem(tmp_path, monkeypatch):
    """Omitting `software` falls back to the scale's default engine."""
    seen = {}

    def fake_generate_inputs(**kw):
        seen["software"] = kw["software"]
        return {"status": "success", "input_files": {"job.nw": "x"},
                "entry_file": "job.nw"}
    monkeypatch.setattr(sp, "_generate_inputs", fake_generate_inputs)

    structure = tmp_path / "mol.xyz"
    structure.write_text("dummy")
    sp.run_complete_workflow(
        "geometry only", scale="molecular_qc",
        structure_file=str(structure), output_dir=str(tmp_path / "o3"),
        api_key="fake-do-not-bill", validate=False,
    )
    assert seen["software"] == "nwchem"


def test_molecular_qc_kwargs_bind_to_signature():
    """Cheap signature guard: the kwargs this scale uses bind to the real
    run_complete_workflow signature (catches drift / typos)."""
    sig = inspect.signature(sp.run_complete_workflow)
    sig.bind(
        "request",
        scale="molecular_qc", software="nwchem",
        structure_file="/tmp/mol.xyz", output_dir="/tmp/out",
        api_key="k", model_name="claude-opus-4-6", validate=False,
        executor=object(), run_command="nwchem {script}",
        autonomy="autonomous", max_run_cycles=3,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
