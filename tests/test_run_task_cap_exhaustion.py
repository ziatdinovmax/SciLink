"""Regression tests for #211 item 1: the #208 status flip.

``chat()`` handles an exhausted iteration cap by returning the warning
string (preserving CLI/UI behavior) and setting ``_last_chat_hit_iter_cap``;
``run_task`` must surface that as ``status="error"`` to programmatic
callers instead of silently reporting success. This contract crosses the
chat loop / run_task seam via a flag, so a chat-loop refactor can break it
without any type error — these tests are the tripwire.

  python -m pytest tests/test_run_task_cap_exhaustion.py -v
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest

CAP_WARNING = ("⚠️ Reached maximum tool iterations (3). The task may be "
               "incomplete.")


def _make(mode, tmp_path):
    if mode == "analysis":
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent)
        return AnalysisOrchestratorAgent(api_key="sk-dummy",
                                         base_dir=str(tmp_path / "an"))
    if mode == "planning":
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent)
        return PlanningOrchestratorAgent(api_key="sk-dummy",
                                         data_dir=str(tmp_path),
                                         base_dir=str(tmp_path / "pl"))
    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent)
    return SimulationOrchestratorAgent(api_key="sk-dummy",
                                       base_dir=str(tmp_path / "si"))


@pytest.mark.parametrize("mode", ["analysis", "planning", "simulate"])
def test_cap_exhaustion_flips_status_to_error(mode, tmp_path):
    orch = _make(mode, tmp_path)
    original_cap = orch.max_iterations

    def fake_chat(_prompt):
        # What a real chat() turn looks like after burning its whole
        # iteration budget: warning string returned, flag set.
        orch._last_chat_hit_iter_cap = True
        return CAP_WARNING

    orch.chat = fake_chat
    result = orch.run_task("some task", max_iterations=3)

    assert result["status"] == "error", (
        f"{mode}: cap exhaustion reported as {result['status']!r} — "
        "the #208 silent-success bug is back")
    assert CAP_WARNING in (result.get("error") or "")
    # The per-call max_iterations override must not leak past the call.
    assert orch.max_iterations == original_cap


@pytest.mark.parametrize("mode", ["analysis", "planning", "simulate"])
def test_clean_completion_still_reports_success(mode, tmp_path):
    orch = _make(mode, tmp_path)

    def fake_chat(_prompt):
        orch._last_chat_hit_iter_cap = False
        return "All done."

    orch.chat = fake_chat
    result = orch.run_task("some task")
    assert result["status"] == "success"
    assert not result.get("error")
