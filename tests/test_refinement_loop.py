"""Unit tests for the engine-neutral refinement loop.

Drives ``run_refinement`` with a fake executor and a scripted fake critic —
no LLM, no simulation engine, no API keys. Verifies the loop's control flow
(cycle counting, multi-phase chaining, fix application) and the three
autonomy policies' decisions.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.refinement import (  # noqa: E402
    AutonomousPolicy,
    AutopilotPolicy,
    CoPilotPolicy,
    CycleDecision,
    Executor,
    Phase,
    RefinementContext,
    Stage,
    run_campaign,
    run_refinement,
    policy_for,
)


# ──────────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────────

class FakeExecutor(Executor):
    """Records each run and reports completion without running anything."""

    def __init__(self):
        self.calls = []

    def run(self, input_files, run_command, run_dir):
        self.calls.append({
            "input_files": dict(input_files),
            "run_command": run_command,
            "run_dir": run_dir,
        })
        return {"status": "completed", "output_dir": run_dir, "returncode": 0}


class ScriptedCritic:
    """Returns a pre-set sequence of verdicts, one per assess() call."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0

    def assess(self, output_dir, research_goal, skill=None, domain=None,
               fixes_mode="auto", input_files=None, check_observables=False,
               required_observables=None):
        self.calls += 1
        # Repeat the last verdict if the loop asks for more than scripted.
        idx = min(self.calls - 1, len(self._verdicts) - 1)
        return dict(self._verdicts[idx])


def _good():
    return {"status": "success", "run_status": "succeeded", "verdict": "good",
            "suggested_fixes": None}


def _needs_fixes(files=None):
    return {"status": "success", "run_status": "failed", "verdict": "needs_fixes",
            "suggested_fixes": files or {"in.sim": "patched contents"}}


def _poor(files=None):
    return {"status": "success", "run_status": "succeeded", "verdict": "poor",
            "suggested_fixes": files or {"in.sim": "patched"}}


def _warning(files=None):
    # files=None → benign warning (critic offers no fix); files set → fixable.
    return {"status": "success", "run_status": "succeeded", "verdict": "warning",
            "suggested_fixes": files}


def _force_field():
    # A run that CONVERGED but is physically wrong; the cause is the force
    # field, so no deck fix is offered (like a "structure" cause).
    return {"status": "success", "run_status": "succeeded", "verdict": "poor",
            "failure_class": "force_field", "suggested_fixes": None}


def _phase(name="production", run_dir="/tmp/rf_test_phase"):
    return Phase(name=name, input_files={"in.sim": "original"},
                 run_command="true", run_dir=run_dir)


def _ctx(autonomy="autonomous", max_cycles=3, interact=None):
    return RefinementContext(
        research_goal="compute something", scale="molecular_dynamics",
        engine="lammps", skill="lammps", domain="molecular_dynamics",
        autonomy=autonomy, max_cycles=max_cycles, interact=interact,
    )


# ──────────────────────────────────────────────────────────────────────────
# Core loop behavior (autonomous)
# ──────────────────────────────────────────────────────────────────────────

class TestLoopFlow:
    def test_converges_first_try(self):
        ex = FakeExecutor()
        critic = ScriptedCritic([_good()])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert len(ex.calls) == 1
        assert critic.calls == 1
        assert result["phases"][0]["verdict"] == "good"

    def test_fail_fix_succeed(self):
        ex = FakeExecutor()
        critic = ScriptedCritic([_needs_fixes({"in.sim": "fixed"}), _good()])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        # Two executions: original, then the fixed inputs.
        assert len(ex.calls) == 2
        assert ex.calls[0]["input_files"] == {"in.sim": "original"}
        assert ex.calls[1]["input_files"] == {"in.sim": "fixed"}

    def test_stall_stops_when_not_improving(self):
        # A verdict that never improves stops on the stall check (after one
        # non-improving cycle), not by burning the whole cycle budget.
        ex = FakeExecutor()
        critic = ScriptedCritic([_needs_fixes()])  # never improves
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(),
                                _ctx(max_cycles=5))
        assert result["status"] == "failed"
        assert len(ex.calls) == 2  # cycle 0 refines, cycle 1 stalls → stop
        assert result["phases"][0]["status"] == "stopped"

    def test_fixable_warning_is_refined(self):
        # A warning the critic offers a fix for is pursued toward good, not
        # accepted as-is.
        ex = FakeExecutor()
        critic = ScriptedCritic([_warning({"in.sim": "smaller timestep"}), _good()])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert len(ex.calls) == 2  # warning → fix → good
        assert ex.calls[1]["input_files"] == {"in.sim": "smaller timestep"}

    def test_benign_warning_stops_as_success(self):
        # A warning with no proposed fix is benign — stop immediately, and it
        # counts as success (acceptable terminal state).
        ex = FakeExecutor()
        critic = ScriptedCritic([_warning(None)])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert len(ex.calls) == 1
        assert result["phases"][0]["status"] == "success"
        assert result["phases"][0]["verdict"] == "warning"

    def test_no_fixes_stops_even_if_not_good(self):
        # A poor verdict with no actionable fixes should stop, not spin.
        ex = FakeExecutor()
        critic = ScriptedCritic([{"verdict": "poor", "run_status": "succeeded",
                                  "suggested_fixes": None}])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert len(ex.calls) == 1
        assert result["phases"][0]["status"] == "stopped"

    def test_force_field_cause_propagates_regardless_of_overall(self):
        # A converged-but-wrong result: the run "succeeded", so this is not a
        # crash — yet the force-field cause must surface at the result level
        # (unlike "structure", which only propagates on a failed run), because
        # the deck loop cannot fix it and it needs reparameterization upstream.
        ex = FakeExecutor()
        critic = ScriptedCritic([_force_field()])
        result = run_refinement([_phase()], ex, critic, AutonomousPolicy(), _ctx())
        assert result["failure_class"] == "force_field"
        assert result["phases"][0]["failure_class"] == "force_field"
        assert len(ex.calls) == 1          # no deck fix to try → stops

    def test_multi_phase_chains(self):
        ex = FakeExecutor()
        # phase 1 good first try; phase 2 needs a fix then good.
        critic = ScriptedCritic([_good(), _needs_fixes({"in.sim": "f"}), _good()])
        phases = [_phase("equil", "/tmp/rf_eq"), _phase("prod", "/tmp/rf_prod")]
        result = run_refinement(phases, ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert [p["phase"] for p in result["phases"]] == ["equil", "prod"]
        # equil: 1 run; prod: 2 runs.
        assert len(ex.calls) == 3
        assert {c["run_dir"] for c in ex.calls} == {"/tmp/rf_eq", "/tmp/rf_prod"}

    def test_cycle_resets_per_phase(self):
        ex = FakeExecutor()
        critic = ScriptedCritic([_good(), _good()])
        ctx = _ctx()
        run_refinement([_phase("a", "/tmp/a"), _phase("b", "/tmp/b")], ex,
                       critic, AutonomousPolicy(), ctx)
        # History has one entry per phase, each at cycle 0.
        assert [h["cycle"] for h in ctx.history] == [0, 0]
        assert [h["phase"] for h in ctx.history] == ["a", "b"]

    def test_empty_phases_fails_cleanly(self):
        result = run_refinement([], FakeExecutor(), ScriptedCritic([_good()]),
                                AutonomousPolicy(), _ctx())
        assert result["status"] == "failed"


# ──────────────────────────────────────────────────────────────────────────
# Policies
# ──────────────────────────────────────────────────────────────────────────

class TestPolicies:
    def test_copilot_abort_via_interact(self):
        # Human says "no" at the pre-run gate → aborted, nothing runs.
        answers = iter(["no"])
        ex = FakeExecutor()
        ctx = _ctx(autonomy="co-pilot",
                   interact=lambda prompt, payload: next(answers))
        result = run_refinement([_phase()], ex, ScriptedCritic([_good()]),
                                CoPilotPolicy(), ctx)
        assert result["status"] == "aborted"
        assert len(ex.calls) == 0

    def test_copilot_approves_then_runs(self):
        answers = iter(["yes", "yes"])  # approve pre-run, then approve a fix
        ex = FakeExecutor()
        ctx = _ctx(autonomy="co-pilot",
                   interact=lambda prompt, payload: next(answers, "yes"))
        critic = ScriptedCritic([_needs_fixes({"in.sim": "f"}), _good()])
        result = run_refinement([_phase()], ex, critic, CoPilotPolicy(), ctx)
        assert result["status"] == "success"
        assert len(ex.calls) == 2

    def test_copilot_headless_degrades(self):
        # No interact handle → co-pilot must not block; behaves autonomously.
        ex = FakeExecutor()
        critic = ScriptedCritic([_needs_fixes({"in.sim": "f"}), _good()])
        result = run_refinement([_phase()], ex, critic, CoPilotPolicy(),
                                _ctx(autonomy="co-pilot"))
        assert result["status"] == "success"
        assert len(ex.calls) == 2

    def test_autopilot_stops_when_stalled(self):
        # Two non-improving cycles → autopilot stops before exhausting budget.
        ex = FakeExecutor()
        critic = ScriptedCritic([_poor()])  # always poor, never improves
        result = run_refinement([_phase()], ex, critic, AutopilotPolicy(),
                                _ctx(autonomy="autopilot", max_cycles=5))
        # Runs cycle 0 (poor), cycle 1 (poor → stalled) then stops.
        assert len(ex.calls) == 2
        assert result["phases"][0]["status"] == "stopped"

    def test_autopilot_surfaces_failing_prerun(self):
        # A "fails" pre-run verdict with an interact handle prompts the human.
        seen = {}
        def interact(prompt, payload):
            seen["asked"] = True
            return "no"
        ex = FakeExecutor()
        ctx = _ctx(autonomy="autopilot", interact=interact)
        result = run_refinement(
            [_phase()], ex, ScriptedCritic([_good()]), AutopilotPolicy(), ctx,
            pre_run_verdict={"validation_status": "fails"},
        )
        assert seen.get("asked") is True
        assert result["status"] == "aborted"
        assert len(ex.calls) == 0

    def test_policy_for_mapping(self):
        assert isinstance(policy_for("co-pilot"), CoPilotPolicy)
        assert isinstance(policy_for("autopilot"), AutopilotPolicy)
        assert isinstance(policy_for("autonomous"), AutonomousPolicy)
        # Unknown label defaults to autonomous (safe for headless callers).
        assert isinstance(policy_for("nonsense"), AutonomousPolicy)
        assert isinstance(policy_for(""), AutonomousPolicy)


class TestStages:
    """Staged + parallel campaigns via run_campaign (Stage model)."""

    def _member(self, name):
        return Phase(name=name, input_files={"in.sim": "original"},
                     run_command="true", run_dir=f"/tmp/rf_{name}")

    def test_single_sequential_stage_matches_run_refinement(self):
        # A campaign of one sequential stage behaves like the flat chain.
        ex = FakeExecutor()
        critic = ScriptedCritic([_good(), _needs_fixes({"in.sim": "f"}), _good()])
        stages = [Stage(name="run",
                        phases=[self._member("equil"), self._member("prod")])]
        result = run_campaign(stages, ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert [p["phase"] for p in result["phases"]] == ["equil", "prod"]
        assert len(result["stages"]) == 1

    def test_fanout_members_are_independent(self):
        # 3 independent members: A good, B needs a fix then good, C poor with no
        # fix (stops). B's refinement and C's failure must not disturb A.
        ex = FakeExecutor()
        critic = ScriptedCritic([
            _good(),                                # A: T300
            _needs_fixes({"in.sim": "fixed"}), _good(),  # B: T400 (fix→good)
            {"verdict": "poor", "run_status": "succeeded",
             "suggested_fixes": None},              # C: T500 (stopped)
        ])
        fanout = Stage(name="tsweep", parallel=True, phases=[
            self._member("T300"), self._member("T400"), self._member("T500")])
        result = run_campaign([fanout], ex, critic, AutonomousPolicy(), _ctx())

        # Every member ran in its own dir; the fan-out did not short-circuit.
        assert {c["run_dir"] for c in ex.calls} == {
            "/tmp/rf_T300", "/tmp/rf_T400", "/tmp/rf_T500"}
        members = {m["phase"]: m for m in result["stages"][0]["members"]}
        assert members["T300"]["status"] == "success"
        assert members["T400"]["status"] == "success"   # recovered
        assert members["T500"]["status"] == "stopped"
        # All-required by default → one stopped member fails the stage.
        assert result["stages"][0]["n_success"] == 2
        assert result["status"] == "failed"

    def test_fanout_min_success_quorum(self):
        # Same 2-of-3, but a quorum of 2 lets the stage (and campaign) succeed.
        ex = FakeExecutor()
        critic = ScriptedCritic([
            _good(), _good(),
            {"verdict": "poor", "run_status": "succeeded", "suggested_fixes": None},
        ])
        fanout = Stage(name="tsweep", parallel=True, min_success=2, phases=[
            self._member("T300"), self._member("T400"), self._member("T500")])
        result = run_campaign([fanout], ex, critic, AutonomousPolicy(), _ctx())
        assert result["stages"][0]["n_success"] == 2
        assert result["status"] == "success"

    def test_combine_runs_once_after_fanout(self):
        # optim → fan-out(2, both good) → combine. Combine runs exactly once.
        ex = FakeExecutor()
        critic = ScriptedCritic([_good(), _good(), _good(), _good()])
        stages = [
            Stage(name="optim", phases=[self._member("optim")]),
            Stage(name="windows", parallel=True,
                  phases=[self._member("w0"), self._member("w1")]),
            Stage(name="wham", kind="combine", phases=[self._member("wham")]),
        ]
        result = run_campaign(stages, ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "success"
        assert result["stages"][-1]["kind"] == "combine"
        # Combine executed once, and last.
        assert ex.calls[-1]["run_dir"] == "/tmp/rf_wham"
        assert sum(c["run_dir"] == "/tmp/rf_wham" for c in ex.calls) == 1

    def test_combine_skipped_when_fanout_fails(self):
        # A failing fan-out stops the campaign before the combine stage runs.
        ex = FakeExecutor()
        critic = ScriptedCritic([
            {"verdict": "poor", "run_status": "succeeded", "suggested_fixes": None},
        ])
        stages = [
            Stage(name="windows", parallel=True, phases=[self._member("w0")]),
            Stage(name="wham", kind="combine", phases=[self._member("wham")]),
        ]
        result = run_campaign(stages, ex, critic, AutonomousPolicy(), _ctx())
        assert result["status"] == "failed"
        # The combine never ran.
        assert all(c["run_dir"] != "/tmp/rf_wham" for c in ex.calls)
        assert [s["name"] for s in result["stages"]] == ["windows"]

    def test_empty_campaign_fails_cleanly(self):
        result = run_campaign([Stage(name="x", phases=[])], FakeExecutor(),
                              ScriptedCritic([_good()]), AutonomousPolicy(), _ctx())
        assert result["status"] == "failed"


class TestCollectPhases:
    """The pipeline's generation-result → Phase normalization (engine-neutral)."""

    def test_single_phase_from_entry_file(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        gr = {"input_files": {"run.lammps": "units metal\n"}, "entry_file": "run.lammps"}
        phases = _collect_phases(gr, "/tmp/runX", "lmp -in {script}")
        assert len(phases) == 1
        assert phases[0].run_command == "lmp -in run.lammps"
        assert phases[0].run_dir == "/tmp/runX"

    def test_entry_synthesized_from_lone_input_file(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        p = _collect_phases({"input_files": {"in.lmp": "x"}}, "/tmp/y",
                            "lmp -in {script}")[0]
        assert p.run_command == "lmp -in in.lmp"

    def test_explicit_multiphase_passthrough(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        gr = {"phases": [
            {"name": "equil", "input_files": {"eq.lmp": "a"}, "entry_file": "eq.lmp"},
            {"name": "prod", "input_files": {"pr.lmp": "b"}, "entry_file": "pr.lmp"},
        ]}
        ps = _collect_phases(gr, "/tmp/z", "lmp -in {script}")
        assert [p.name for p in ps] == ["equil", "prod"]
        assert ps[1].run_command == "lmp -in pr.lmp"

    def test_template_without_placeholder_used_verbatim(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_phases
        gr = {"input_files": {"run.lammps": "x"}, "entry_file": "run.lammps"}
        assert _collect_phases(gr, "/tmp/y", "run_md.sh")[0].run_command == "run_md.sh"


class TestCollectStages:
    """Generation-result → Stage normalization (engine-neutral, offline)."""

    def test_legacy_shape_wraps_as_one_sequential_stage(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_stages
        gr = {"input_files": {"run.lammps": "x"}, "entry_file": "run.lammps"}
        stages = _collect_stages(gr, "/tmp/base", "lmp -in {script}")
        assert len(stages) == 1
        assert stages[0].parallel is False and stages[0].kind == "run"
        assert stages[0].phases[0].run_command == "lmp -in run.lammps"
        assert stages[0].phases[0].run_dir == "/tmp/base"

    def test_sequential_steps_share_base_run_dir(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_stages
        gr = {"stages": [
            {"name": "optim", "input_files": {"o.lmp": "a"}, "entry_file": "o.lmp"},
            {"name": "prod", "input_files": {"p.lmp": "b"}, "entry_file": "p.lmp"},
        ]}
        stages = _collect_stages(gr, "/tmp/base", "lmp -in {script}")
        assert [s.name for s in stages] == ["optim", "prod"]
        # Sequential steps chain in the shared base dir.
        assert all(s.phases[0].run_dir == "/tmp/base" for s in stages)

    def test_fanout_members_get_isolated_run_dirs(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_stages
        gr = {"stages": [
            {"name": "tsweep", "parallel": True, "min_success": 2, "members": [
                {"name": "T300", "input_files": {"p.lmp": "a"}, "entry_file": "p.lmp"},
                {"name": "T400", "input_files": {"p.lmp": "b"}, "entry_file": "p.lmp"},
            ]},
        ]}
        stages = _collect_stages(gr, "/tmp/base", "lmp -in {script}")
        st = stages[0]
        assert st.parallel is True and st.min_success == 2
        assert {p.run_dir for p in st.phases} == {
            "/tmp/base/tsweep/T300", "/tmp/base/tsweep/T400"}
        assert all(p.run_command == "lmp -in p.lmp" for p in st.phases)

    def test_combine_stage_dir_and_command_override(self):
        from scilink.agents.sim_agents.simulation_pipeline import _collect_stages
        gr = {"stages": [
            {"name": "windows", "parallel": True, "members": [
                {"name": "w0", "input_files": {"w.lmp": "a"}, "entry_file": "w.lmp"},
            ]},
            {"name": "wham", "kind": "combine", "entry_file": "wham.py",
             "input_files": {"wham.py": "..."}, "run_command": "python {script}"},
        ]}
        stages = _collect_stages(gr, "/tmp/base", "lmp -in {script}")
        combine = stages[-1]
        assert combine.kind == "combine"
        assert combine.phases[0].run_dir == "/tmp/base/wham"
        # The combine declares its own interpreter, overriding the engine template.
        assert combine.phases[0].run_command == "python wham.py"


class TestSweepGeneration:
    """Parallel-sweep generation: expansion + fan-out assembly (offline)."""

    def test_expand_parameter_sweep_substitutes_placeholder(self):
        from scilink.skills.molecular_dynamics.lammps.lammps import (
            expand_parameter_sweep, SWEEP_PLACEHOLDER,
        )
        base = (f"velocity all create {SWEEP_PLACEHOLDER} 12345\n"
                f"fix 1 all nvt temp {SWEEP_PLACEHOLDER} {SWEEP_PLACEHOLDER} 0.1\n")
        members = expand_parameter_sweep(base, "temperature", [300, 400, 500])
        assert [m["name"] for m in members] == [
            "temperature_300", "temperature_400", "temperature_500"]
        # The placeholder is gone and the value is substituted everywhere.
        assert SWEEP_PLACEHOLDER not in members[1]["script"]
        assert members[1]["script"].count("400") == 3
        assert "300" not in members[1]["script"]

    def test_sweep_member_name_is_safe(self):
        from scilink.skills.molecular_dynamics.lammps.lammps import sweep_member_name
        assert sweep_member_name("temperature", 300) == "temperature_300"
        assert sweep_member_name("rest center", 1.5) == "restcenter_1.5"

    def test_assemble_fanout_stage_makes_self_contained_members(self):
        from scilink.agents.sim_agents.md_simulation_agent import (
            _assemble_fanout_stage,
        )
        members = [{"name": "temperature_300", "script": "S300"},
                   {"name": "temperature_400", "script": "S400"}]
        shared = {"system.data": "DATA", "ff.params": "FF"}
        stages = _assemble_fanout_stage(members, "run.lammps", shared)
        assert len(stages) == 1
        st = stages[0]
        assert st["name"] == "production" and st["parallel"] is True
        m0 = st["members"][0]
        # Every member carries the shared files plus its own script.
        assert m0["input_files"]["system.data"] == "DATA"
        assert m0["input_files"]["ff.params"] == "FF"
        assert m0["input_files"]["run.lammps"] == "S300"
        assert m0["entry_file"] == "run.lammps"
        # Members are independent: editing one's map must not touch the other's.
        assert st["members"][1]["input_files"]["run.lammps"] == "S400"

    def test_sweep_expansion_feeds_collect_stages(self):
        # End-to-end of the deterministic half: expand → assemble → normalize
        # into Stage objects with isolated run dirs (no agent, no LLM).
        from scilink.skills.molecular_dynamics.lammps.lammps import (
            expand_parameter_sweep, SWEEP_PLACEHOLDER,
        )
        from scilink.agents.sim_agents.md_simulation_agent import (
            _assemble_fanout_stage,
        )
        from scilink.agents.sim_agents.simulation_pipeline import _collect_stages

        base = f"fix 1 all nvt temp {SWEEP_PLACEHOLDER} {SWEEP_PLACEHOLDER} 0.1\n"
        members = expand_parameter_sweep(base, "temperature", [300, 400])
        stages_spec = _assemble_fanout_stage(
            members, "run.lammps", {"system.data": "DATA"})
        stages = _collect_stages({"stages": stages_spec}, "/tmp/base",
                                 "lmp -in {script}")
        assert len(stages) == 1 and stages[0].parallel is True
        assert {p.run_dir for p in stages[0].phases} == {
            "/tmp/base/production/temperature_300",
            "/tmp/base/production/temperature_400"}
        assert all(p.run_command == "lmp -in run.lammps" for p in stages[0].phases)


class TestStagedGeneration:
    """Sequential multi-phase generation: sequential stage assembly (offline)."""

    def test_assemble_sequential_stages_makes_self_contained_stages(self):
        from scilink.agents.sim_agents.md_simulation_agent import (
            _assemble_sequential_stages,
        )
        steps = [
            {"name": "equilibration", "entry_file": "run_equilibration.lammps",
             "script": "EQUIL"},
            {"name": "production", "entry_file": "run_production.lammps",
             "script": "PROD"},
        ]
        shared = {"system.data": "DATA", "ff.params": "FF"}
        specs = _assemble_sequential_stages(steps, shared)
        assert [s["name"] for s in specs] == ["equilibration", "production"]
        s0 = specs[0]
        # Each stage carries the shared files plus its own script — and is a
        # plain sequential step (no parallel / members).
        assert s0["input_files"]["system.data"] == "DATA"
        assert s0["input_files"]["ff.params"] == "FF"
        assert s0["input_files"]["run_equilibration.lammps"] == "EQUIL"
        assert s0["entry_file"] == "run_equilibration.lammps"
        assert "parallel" not in s0 and "members" not in s0
        # Stages are independent dicts; one stage's script doesn't leak into another.
        assert specs[1]["input_files"]["run_production.lammps"] == "PROD"
        assert "run_equilibration.lammps" not in specs[1]["input_files"]

    def test_sequential_assembly_feeds_collect_stages(self):
        # End-to-end of the deterministic half: assemble → normalize into
        # sequential Stage objects sharing the base run dir (no agent, no LLM).
        from scilink.agents.sim_agents.md_simulation_agent import (
            _assemble_sequential_stages,
        )
        from scilink.agents.sim_agents.simulation_pipeline import _collect_stages

        steps = [
            {"name": "equilibration", "entry_file": "run_equilibration.lammps",
             "script": "EQUIL"},
            {"name": "production", "entry_file": "run_production.lammps",
             "script": "PROD"},
        ]
        specs = _assemble_sequential_stages(steps, {"system.data": "DATA"})
        stages = _collect_stages({"stages": specs}, "/tmp/base", "lmp -in {script}")
        # Two sequential stages, one phase each, all sharing the base run dir so
        # restart files chain optimization → equilibration → production.
        assert len(stages) == 2
        assert [st.name for st in stages] == ["equilibration", "production"]
        assert all(st.parallel is False and len(st.phases) == 1 for st in stages)
        assert {st.phases[0].run_dir for st in stages} == {"/tmp/base"}
        assert stages[0].phases[0].run_command == "lmp -in run_equilibration.lammps"
        assert stages[1].phases[0].run_command == "lmp -in run_production.lammps"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
