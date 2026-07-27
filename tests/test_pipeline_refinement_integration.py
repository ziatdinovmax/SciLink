"""Integration tests for the refinement loop composed into the pipeline.

These exercise the seams that the unit tests skip: the step-4 glue in
``run_complete_workflow`` (phases + RefinementContext + RunCritic +
run_refinement wired together, with a *real* LocalExecutor), and the
UI-generated remote-run script (does it compile, and do its
run_complete_workflow kwargs bind to real parameter names?).

Only the LLM-dependent bits are stubbed (structure generation is skipped via
structure_file; input generation and the critic are monkeypatched). The
executor and the entire refinement control path run for real. No API keys.
"""

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402
import scilink.agents.sim_agents.critics as critics_mod  # noqa: E402
from scilink.agents.sim_agents.refinement import LocalExecutor  # noqa: E402


class TestWorkflowComposition:
    """run_complete_workflow's executor path, end to end with a real executor."""

    def _stub_generation(self, monkeypatch):
        def fake_generate_inputs(**kw):
            script = "echo running\n"
            (Path(kw["output_dir"]) / "run.lammps").write_text(script)
            return {
                "status": "success",
                "input_files": {"run.lammps": script},
                "entry_file": "run.lammps",
            }
        monkeypatch.setattr(sp, "_generate_inputs", fake_generate_inputs)

    def _setup_md_with_manifest(self, tmp_path, monkeypatch):
        """Structure + components.json + faked FF parameterization, so the
        reference-check gate is reachable without real OpenFF."""
        import json
        self._stub_generation(monkeypatch)
        structure = tmp_path / "system.data"
        structure.write_text("dummy")
        (tmp_path / "components.json").write_text(json.dumps(
            {"components": [{"name": "water", "smiles": "O", "count": 100},
                            {"name": "EIS", "smiles": "CCS(=O)(=O)C(C)C",
                             "count": 10}]}))

        class FakePSystem:
            backend, n_atoms, total_charge = "openff", 330, 0.0

        class FakeFFAgent:
            def __init__(self, **kw):
                pass

            def parameterize(self, **kw):
                return FakePSystem()

        import scilink.agents.sim_agents._engine_inputs as ei
        import scilink.agents.sim_agents.force_field_agent as ffa
        monkeypatch.setattr(ffa, "ForceFieldAgent", FakeFFAgent)
        monkeypatch.setattr(ei, "write_md_inputs",
                            lambda ps, sw, wd: {"structure_file": str(structure),
                                                "force_field_files": {}})
        return structure

    def test_reference_check_gate_blocks_production_on_poor(self, tmp_path,
                                                            monkeypatch):
        structure = self._setup_md_with_manifest(tmp_path, monkeypatch)
        monkeypatch.setattr(sp, "_reference_check", lambda *a, **k: {
            "selections": [], "measurements": [],
            "verdict": {"verdict": "poor", "failure_class": "force_field"}})
        result = sp.run_complete_workflow(
            "aqueous sulfone electrolyte density",
            scale="molecular_dynamics", software="lammps",
            structure_file=str(structure), output_dir=str(tmp_path / "out"),
            api_key="fake-do-not-bill", validate=False,
            executor=LocalExecutor(timeout=30), run_command="true",
            reference_check=True)
        # Caught before production: gate ran, campaign did NOT.
        assert result["final_status"] == "reference_validation_failed"
        assert result.get("force_field_flagged") is True
        assert "reference_validation" in result["steps_completed"]
        assert "refinement" not in result["steps_completed"]
        assert any("reparameteriz" in w.lower()
                   for w in result.get("warnings", []))

    def test_reference_check_good_proceeds_to_production(self, tmp_path,
                                                         monkeypatch):
        structure = self._setup_md_with_manifest(tmp_path, monkeypatch)
        monkeypatch.setattr(sp, "_reference_check", lambda *a, **k: {
            "selections": [], "measurements": [],
            "verdict": {"verdict": "good", "failure_class": None}})
        import scilink.agents.sim_agents.refinement as rf
        monkeypatch.setattr(rf, "run_campaign", lambda *a, **k: {
            "status": "success", "phases": [], "stages": [], "history": []})
        result = sp.run_complete_workflow(
            "aqueous electrolyte density",
            scale="molecular_dynamics", software="lammps",
            structure_file=str(structure), output_dir=str(tmp_path / "out2"),
            api_key="fake-do-not-bill", validate=False,
            executor=LocalExecutor(timeout=30), run_command="true",
            reference_check=True)
        # Good verdict: gate ran and passed; production proceeded.
        assert result.get("force_field_flagged") is not True
        assert "reference_validation" in result["steps_completed"]
        assert "refinement" in result["steps_completed"]

    def test_executor_path_fail_fix_succeed(self, tmp_path, monkeypatch):
        self._stub_generation(monkeypatch)

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
                            "suggested_fixes": {"run.lammps": "echo fixed\n"}}
                return {"status": "success", "run_status": "succeeded",
                        "verdict": "good", "suggested_fixes": None}

        monkeypatch.setattr(critics_mod, "RunCritic", FakeRunCritic)

        structure = tmp_path / "POSCAR"
        structure.write_text("dummy structure")
        out = tmp_path / "out"

        result = sp.run_complete_workflow(
            "anneal a slab",
            scale="molecular_dynamics", software="lammps",
            structure_file=str(structure),
            output_dir=str(out),
            api_key="fake-do-not-bill",
            model_name="claude-opus-4-6",
            validate=False,
            executor=LocalExecutor(timeout=30),
            run_command="true",
            autonomy="autonomous",
            max_run_cycles=3,
        )

        # The whole step-4 glue ran: phases built, loop drove the real
        # executor, the critic's fix was applied and re-run. With the LAMMPS
        # dry-run gate active, the fail→fix happens in the cheap pre-flight
        # (critic calls 1–2), then the production run succeeds first try
        # (call 3) — so the gate is exercised end to end in the real pipeline.
        assert result["final_status"] == "success", result
        assert result["refinement"]["status"] == "success"
        assert result["refinement"]["dry_run"]["status"] == "passed"
        assert FakeRunCritic.calls == 3          # gate: fail, good; run: good
        assert "refinement" in result["steps_completed"]
        # The gate's fix is the deck the production run executed.
        assert (out / "run.lammps").read_text() == "echo fixed\n"

    def test_no_executor_leaves_behavior_unchanged(self, tmp_path, monkeypatch):
        self._stub_generation(monkeypatch)
        structure = tmp_path / "POSCAR"
        structure.write_text("dummy")
        result = sp.run_complete_workflow(
            "prep only",
            scale="molecular_dynamics", software="lammps",
            structure_file=str(structure),
            output_dir=str(tmp_path / "out2"),
            api_key="fake-do-not-bill",
            validate=False,
        )
        assert result["final_status"] == "success"
        assert "refinement" not in result          # no execution attempted
        assert result["structure_generation"]["status"] == "skipped"

    def test_executor_without_run_command_skips_refinement(self, tmp_path, monkeypatch):
        self._stub_generation(monkeypatch)
        structure = tmp_path / "POSCAR"
        structure.write_text("dummy")
        result = sp.run_complete_workflow(
            "no command",
            scale="molecular_dynamics", software="lammps",
            structure_file=str(structure),
            output_dir=str(tmp_path / "out3"),
            api_key="fake-do-not-bill",
            validate=False,
            executor=LocalExecutor(timeout=30),
            run_command=None,
        )
        assert result["final_status"] == "success"
        assert result["refinement"]["status"] == "skipped"

    def test_force_field_result_is_surfaced(self, tmp_path, monkeypatch):
        self._stub_generation(monkeypatch)
        # Stub the campaign to return a converged, force-field-flagged result
        # (the detector's output), bypassing the loop/dry-run internals.
        import scilink.agents.sim_agents.refinement as rf
        monkeypatch.setattr(rf, "run_campaign", lambda *a, **k: {
            "status": "success", "failure_class": "force_field",
            "phases": [], "stages": [], "history": [],
        })
        structure = tmp_path / "POSCAR"
        structure.write_text("dummy")
        result = sp.run_complete_workflow(
            "compute mixture density",
            scale="molecular_dynamics", software="lammps",
            structure_file=str(structure),
            output_dir=str(tmp_path / "out_ff"),
            api_key="fake-do-not-bill",
            validate=False,
            executor=LocalExecutor(timeout=30),
            run_command="true",
        )
        # The force-field cause is surfaced plainly; the run still completes
        # (detection only — the automated fix is not wired yet).
        assert result["refinement"]["failure_class"] == "force_field"
        assert result.get("force_field_flagged") is True
        assert any("reparameteriz" in w.lower()
                   for w in result.get("warnings", []))
        assert result["final_status"] == "success"


class TestGeneratedRemoteScript:
    """The UI-generated remote-run script must compile and call the pipeline
    with real parameter names."""

    def _load_generator(self):
        # sim_workflow imports streamlit (not a test dep); stub it so the
        # template generator imports.
        for name in ("streamlit",):
            sys.modules.setdefault(name, types.ModuleType(name))
        from scilink.ui.components.sim_workflow import _generate_run_script
        return _generate_run_script

    def _render(self):
        gen = self._load_generator()
        return gen(
            data_file="/work/system.data",
            research_goal="equilibrate at 300K",
            ff_files={"ff.param": "pair_style lj/cut 10.0"},
            lammps_command="lmp",
            model_name="claude-opus-4-6",
            base_url="",
            api_key_env="ANTHROPIC_API_KEY",
            max_attempts=3,
            stage_timeout=3600,
            container_work_dir="/work",
        )

    def test_generated_script_compiles(self):
        script = self._render()
        compile(script, "<generated>", "exec")  # raises SyntaxError on failure
        assert "lmp -in {script}" in script     # template placeholder preserved
        assert "LAMMPSOrchestrator" not in script  # migrated off the baseline

    def test_generated_call_kwargs_are_valid(self):
        script = self._render()
        tree = ast.parse(script)
        # run_complete_workflow is a thin wrapper — (user_request, *,
        # max_structure_retries, **kwargs) — that forwards **kwargs to
        # _run_workflow_once, so the accepted keyword names live on the
        # delegate. Validate against the union of both signatures.
        valid = (set(inspect.signature(sp.run_complete_workflow).parameters)
                 | set(inspect.signature(sp._run_workflow_once).parameters))
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_complete_workflow"
        ]
        assert len(calls) == 1, "expected exactly one run_complete_workflow call"
        kwargs = {kw.arg for kw in calls[0].keywords if kw.arg}
        unknown = kwargs - valid
        assert not unknown, f"generated script passes unknown kwargs: {unknown}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
