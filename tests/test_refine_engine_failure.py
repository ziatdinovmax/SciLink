"""A run whose engine failed is never accepted on the critic's word.

A critic can call a crashed run a "warning" (e.g. NWChem exiting 255 after
TDDFT refused the molecule's symmetry): the phase then passed as a success
with no result. The executor's facts (non-zero exit, timeout, launch error)
now hold an acceptable verdict at needs_fixes. No LLM, engine or API key.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.refinement import (  # noqa: E402
    AutonomousPolicy, AutopilotPolicy, Executor, Phase, RefinementContext, Stage,
    run_campaign,
)


class _Exec(Executor):
    """Returns the scripted executor results, one per run (the last repeats)."""

    def __init__(self, *results):
        self.results, self.calls = list(results), 0

    def run(self, input_files, run_command, run_dir):
        r = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return {"output_dir": run_dir, **r}


class _Critic:
    def __init__(self, *verdicts):
        self.verdicts, self.calls = list(verdicts), 0

    def assess(self, *a, **k):
        v = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
        self.calls += 1
        return dict(v)


def _warning(fixes=None):
    return {"status": "success", "run_status": "incomplete", "verdict": "warning",
            "suggested_fixes": fixes}


def _good():
    return {"status": "success", "run_status": "succeeded", "verdict": "good",
            "suggested_fixes": None}


def _ctx():
    # No skill: no health gate or dry-run hooks from a real bundle.
    return RefinementContext(research_goal="g", scale="molecular_qc", engine="nwchem",
                             skill=None, domain="molecular_qc",
                             autonomy="autonomous", max_cycles=3)


def _phase(tmp_path, kind="run"):
    ph = Phase(name="job.nw", input_files={"job.nw": "x"}, run_command="nwchem job.nw",
               run_dir=str(tmp_path), entry_file="job.nw")
    return Stage(name="run", phases=[ph], kind=kind)


def _run(tmp_path, ex, critic, kind="run", policy=None):
    out = run_campaign([_phase(tmp_path, kind)], ex, critic,
                       policy or AutonomousPolicy(), _ctx())
    return out, out["phases"][0]


def test_a_crashed_run_is_not_a_success_on_a_warning(tmp_path):
    out, rec = _run(tmp_path, _Exec({"status": "completed", "returncode": 255}),
                    _Critic(_warning()))
    assert out["status"] != "success"
    assert rec["status"] == "stopped" and rec["verdict"] == "needs_fixes"
    assert rec["engine_failure"] == "the engine exited with code 255"


def test_a_crash_the_critic_can_fix_is_refined_then_accepted(tmp_path):
    ex = _Exec({"status": "completed", "returncode": 255},
               {"status": "completed", "returncode": 0})
    out, rec = _run(tmp_path, ex, _Critic(_warning({"job.nw": "fixed"}), _good()))
    assert ex.calls == 2
    assert out["status"] == "success" and rec["status"] == "success"
    assert "engine_failure" not in rec


def test_a_timed_out_run_is_not_accepted(tmp_path):
    out, rec = _run(tmp_path, _Exec({"status": "error", "returncode": None,
                                     "error": "Run timed out after 60s"}),
                    _Critic(_good()))
    assert rec["status"] == "stopped"
    assert rec["engine_failure"] == "Run timed out after 60s"


def test_an_unknown_exit_code_still_trusts_the_critic(tmp_path):
    # A scheduler that reports no exit code: nothing measured to override.
    out, rec = _run(tmp_path, _Exec({"status": "completed", "returncode": None}),
                    _Critic(_good()))
    assert out["status"] == "success" and rec["status"] == "success"


def test_a_clean_run_with_a_warning_is_still_accepted(tmp_path):
    out, rec = _run(tmp_path, _Exec({"status": "completed", "returncode": 0}),
                    _Critic(_warning()))
    assert rec["status"] == "success" and rec["verdict"] == "warning"


def test_a_crashed_combine_step_is_not_accepted(tmp_path):
    out, rec = _run(tmp_path, _Exec({"status": "completed", "returncode": 1}),
                    _Critic(_good()), kind="combine")
    assert rec["status"] == "stopped"
    assert rec["engine_failure"] == "the engine exited with code 1"


def test_the_reason_reaches_the_campaign_result(tmp_path):
    out, _ = _run(tmp_path, _Exec({"status": "completed", "returncode": 255}),
                  _Critic(_warning()))
    assert out["engine_failures"] == ["job.nw: the engine exited with code 255"]


def test_autopilot_refines_a_crash_it_used_to_accept(tmp_path):
    # Autopilot stopped on "warning" and called it a success.
    ex = _Exec({"status": "completed", "returncode": 255},
               {"status": "completed", "returncode": 0})
    out, rec = _run(tmp_path, ex, _Critic(_warning({"job.nw": "fixed"}), _good()),
                    policy=AutopilotPolicy())
    assert ex.calls == 2 and rec["status"] == "success"


def test_a_repeated_crash_stops_within_the_cycle_budget(tmp_path):
    ex = _Exec({"status": "completed", "returncode": 255})
    out, rec = _run(tmp_path, ex, _Critic(_warning({"job.nw": "again"})))
    assert ex.calls <= 3 and rec["status"] in ("stopped", "exhausted")
    assert rec["engine_failure"] == "the engine exited with code 255"


def test_a_signal_kill_is_a_failure(tmp_path):
    out, rec = _run(tmp_path, _Exec({"status": "completed", "returncode": -9}),
                    _Critic(_good()))
    assert rec["engine_failure"] == "the engine exited with code -9"


def test_a_poor_verdict_is_left_as_it_is(tmp_path):
    poor = {"status": "success", "run_status": "failed", "verdict": "poor",
            "suggested_fixes": None}
    out, rec = _run(tmp_path, _Exec({"status": "completed", "returncode": 1}),
                    _Critic(poor))
    assert rec["verdict"] == "poor" and "engine_failure" not in rec


def test_a_crashed_fanout_member_does_not_count_toward_the_quorum(tmp_path):
    members = [Phase(name=f"m{i}", input_files={"in": "x"}, run_command="run",
                     run_dir=str(tmp_path / f"m{i}"), entry_file="in") for i in range(2)]
    stage = Stage(name="sweep", phases=members, parallel=True, min_success=2)

    class _PerMember(Executor):
        def run(self, input_files, run_command, run_dir):
            rc = 255 if run_dir.endswith("m1") else 0
            return {"status": "completed", "returncode": rc, "output_dir": run_dir}

    out = run_campaign([stage], _PerMember(), _Critic(_warning()), AutonomousPolicy(), _ctx())
    assert out["status"] != "success"
    assert [p["status"] for p in out["phases"]] == ["success", "stopped"]
