"""Regression tests for honest delegation status.

``chat()``'s emergency catch-all returns "❌ Error: ..." as a normal reply
string, so ``run_task`` used to report ``status="success"`` for a turn that
actually died — live sessions recorded such delegations as successes with
zero files and summaries beginning ❌, and telemetry / fusion treated them
as real results. Two layers now keep the ledger honest:

  - child-side: ``chat()`` flags the failure (``_last_chat_error``) and
    ``run_task`` flips the status (mirrors the #208 iteration-cap flag);
  - meta-side: ``_close_delegation`` derives status from the outcome —
    a result carrying an error, or missing a status, is never a success.

  python -m pytest tests/test_run_task_error_status.py -v
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

import pytest


def _make(mode, tmp_path):
    if mode == "analysis":
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent)
        return AnalysisOrchestratorAgent(api_key="sk-dummy",
                                         base_dir=str(tmp_path / "an"))
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent)
    return PlanningOrchestratorAgent(api_key="sk-dummy",
                                     data_dir=str(tmp_path),
                                     base_dir=str(tmp_path / "pl"))


@pytest.mark.parametrize("mode", ["analysis", "planning"])
def test_chat_emergency_error_flips_status(mode, tmp_path):
    orch = _make(mode, tmp_path)

    def fake_chat(_prompt):
        # What a real chat() turn looks like after its emergency catch-all
        # fired: ❌ string returned as the reply, error flag set.
        orch._last_chat_hit_iter_cap = False
        orch._last_chat_error = "LLM call exploded"
        return "❌ Error: LLM call exploded\n\n(Emergency checkpoint saved)"

    orch.chat = fake_chat
    result = orch.run_task("some task")

    assert result["status"] == "error", (
        f"{mode}: emergency chat error reported as {result['status']!r} — "
        "the ❌-summary-as-success bug (#411 follow-up) is back")
    assert "LLM call exploded" in (result.get("error") or "")


@pytest.mark.parametrize("mode", ["analysis", "planning"])
def test_chat_resets_error_flag_each_turn(mode, tmp_path):
    """A stale error flag from a failed turn must not poison the next one."""
    orch = _make(mode, tmp_path)
    orch._last_chat_error = "stale failure from a previous turn"
    orch.use_openai = False
    orch._handle_litellm_chat = lambda _msg: "fine"
    reply = orch.chat("hello")
    assert "fine" in reply
    assert orch._last_chat_error is None


@pytest.mark.parametrize("mode", ["analysis", "planning"])
def test_clean_completion_still_reports_success(mode, tmp_path):
    orch = _make(mode, tmp_path)

    def fake_chat(_prompt):
        orch._last_chat_hit_iter_cap = False
        orch._last_chat_error = None
        return "All done."

    orch.chat = fake_chat
    result = orch.run_task("some task")
    assert result["status"] == "success"
    assert not result.get("error")


# ---------------------------------------------------------------- meta side


def _make_meta(tmp_path):
    from scilink.agents.meta_agent.meta_orchestrator import (
        MetaOrchestratorAgent)
    return MetaOrchestratorAgent(api_key="sk-dummy", base_dir=str(tmp_path))


def test_close_delegation_derives_status_from_outcome(tmp_path):
    meta = _make_meta(tmp_path)

    # A "success" carrying an error is a failure.
    entry = {"index": 1}
    meta._close_delegation(entry, {"status": "success",
                                   "error": "❌ Error: it broke",
                                   "summary": "❌ Error: it broke"})
    assert entry["status"] == "error"

    # A result with no status at all is not a success.
    entry = {"index": 2}
    meta._close_delegation(entry, {"summary": "??"})
    assert entry["status"] == "error"

    # A clean success stays a success; an explicit error stays an error.
    entry = {"index": 3}
    meta._close_delegation(entry, {"status": "success", "summary": "ok"})
    assert entry["status"] == "success"
    entry = {"index": 4}
    meta._close_delegation(entry, {"status": "error", "error": "x"})
    assert entry["status"] == "error"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
