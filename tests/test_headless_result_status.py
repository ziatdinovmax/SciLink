"""What a headless caller can trust: the status, the exit code and whether
the analyses behind a success were verified.

Before: a meta ``-p`` turn was a success whenever it had a reply (a turn
whose every delegation failed exited 0), and an analyze ``-p`` run exited 0
for a salvaged or unverified result with nothing at the top level to say so.
No LLM calls.
"""
import json
from types import SimpleNamespace

import pytest

from scilink.agents.exp_agents._replay import verdict_record
from scilink.agents.exp_agents._verification_record import (merge_verification,
                                                            verification_summary)
from scilink.agents.exp_agents.analysis_orchestrator import (AnalysisMode,
                                                            AnalysisOrchestratorAgent)
from scilink.agents.meta_agent.turn_result import turn_result, turn_status
from scilink.cli.shell import headless
from scilink.cli.shell.modes import MetaAdapter


# ── verification over a task's analyses ──────────────────────────

def _row(aid, status="success", verified=True, reason="approved"):
    return {"analysis_id": aid, "status": status, "verified": verified, "reason": reason}


def test_all_verified():
    v = verification_summary([_row("a"), _row("b")])
    assert v == {"analyses": 2, "verified": 2, "unverified": [], "failed": [],
                 "all_verified": True}


def test_an_unverified_success_is_counted_and_named():
    v = verification_summary([_row("a"), _row("b", verified=False, reason="salvaged fit")])
    assert v["all_verified"] is False and v["verified"] == 1
    assert v["unverified"] == [{"analysis_id": "b", "reason": "salvaged fit"}]
    assert v["failed"] == []


def test_a_failed_analysis_is_not_called_unverified():
    v = verification_summary([_row("a"), _row("b", status="error", verified=False)])
    assert v["all_verified"] is False
    assert v["failed"] == [{"analysis_id": "b", "status": "error"}] and v["unverified"] == []


def test_a_partial_result_is_a_result_and_is_judged_on_its_verdict():
    v = verification_summary([_row("cube", status="partial", verified=False, reason="status 'partial'")])
    assert v["unverified"][0]["analysis_id"] == "cube" and v["failed"] == []


def test_no_analysis_is_neither_verified_nor_not():
    assert verification_summary([])["all_verified"] is None
    assert verification_summary(None)["analyses"] == 0


def test_merge_over_delegations():
    a = verification_summary([_row("a")])
    b = verification_summary([_row("b", verified=False, reason="r")])
    merged = merge_verification([a, None, b, {"unrelated": 1}])
    assert merged["analyses"] == 2 and merged["verified"] == 1
    assert merged["all_verified"] is False and merged["unverified"][0]["analysis_id"] == "b"
    assert merge_verification([None, None]) is None
    assert merge_verification([verification_summary([])])["all_verified"] is None


# ── analyze: run_task carries it ─────────────────────────────────

def _analysis_orch(records):
    o = AnalysisOrchestratorAgent.__new__(AnalysisOrchestratorAgent)
    o.analysis_mode = AnalysisMode.AUTONOMOUS
    o.max_iterations = 5
    o.analysis_results = []
    o.default_profile = o.default_targets = o.default_time_budget_s = None
    o._last_chat_hit_iter_cap = False
    o._last_chat_error = None
    o.logger = SimpleNamespace(exception=lambda *a, **k: None, info=lambda *a, **k: None,
                               warning=lambda *a, **k: None)
    o.set_analysis_mode = lambda m: setattr(o, "analysis_mode", m)
    o._auto_checkpoint = lambda *a, **k: None

    def chat(prompt):
        o.analysis_results.extend(records)
        return "done"
    o.chat = chat
    return o


def _record(aid, *, verified, reason="approved", status="success"):
    return {"analysis_id": aid, "agent_name": "CurveFittingAgent", "status": status,
            "data_path": f"/data/{aid}.csv", "output_directory": None,
            "full_result": {"status": status, "verdict": verdict_record(
                verified=verified, reason=reason, decided_by="qc_gate")}}


def test_run_task_reports_verification_at_the_top_level():
    res = _analysis_orch([_record("a", verified=True),
                          _record("b", verified=False, reason="best available, not approved")]
                         ).run_task("fit both")
    assert res["status"] == "success"            # the task ran: unchanged
    v = res["verification"]
    assert v["analyses"] == 2 and v["verified"] == 1 and v["all_verified"] is False
    assert v["unverified"] == [{"analysis_id": "b", "reason": "best available, not approved"}]
    # the same answer the rows give
    assert [r["verified"] for r in res["analyses"]] == [True, False]


def test_run_task_with_no_analysis_has_nothing_to_verify():
    res = _analysis_orch([]).run_task("just answer")
    assert res["verification"]["all_verified"] is None


# ── the meta turn ────────────────────────────────────────────────

def _entry(index, status="success", mode="analysis", **kw):
    e = {"index": index, "mode": mode, "label": f"d{index}", "status": status, "summary": "",
         "key_findings": [f"finding {index}"], "files_produced": [f"/out/{index}.png"],
         "feature_tables": [], "suggested_followups": [], "warnings": [], "error": None}
    e.update(kw)
    return e


@pytest.mark.parametrize("statuses, expected", [
    ([], "success"),                               # a turn that needed no delegation
    (["success", "success"], "success"),
    (["success", "error"], "partial"),
    (["error", "interrupted"], "error"),
    (["error"], "error"),
    (["partial"], "success"),                      # a result was produced
])
def test_turn_status_follows_the_delegations(statuses, expected):
    entries = [_entry(i + 1, s) for i, s in enumerate(statuses)]
    assert turn_status("a reply", entries) == expected


def test_a_turn_that_failed_is_an_error_whatever_it_delegated():
    ok = [_entry(1)]
    assert turn_status("", ok) == "error"
    assert turn_status("❌ Error: boom", ok, chat_error="boom") == "error"
    assert turn_status("⚠️ iteration cap", ok, hit_iteration_cap=True) == "error"


class _Meta:
    """A meta whose chat() closes the delegations it is scripted with."""

    def __init__(self, ledger=None, script=(), reply="All done.", chat_error=None):
        self._delegation_ledger = list(ledger or [])
        self._script, self._reply, self._chat_error = list(script), reply, chat_error
        self._last_chat_error = None
        self._last_chat_hit_iter_cap = False

    def chat(self, task):
        self._delegation_ledger.extend(self._script)
        self._last_chat_error = self._chat_error
        return self._reply


def test_every_delegation_failed_is_an_error_even_with_a_reply():
    """The bug: this exited 0."""
    meta = _Meta(script=[_entry(1, "error", error="branch wall-clock budget exceeded")],
                 reply="I could not complete the analysis.")
    res = MetaAdapter().headless_run(meta, "analyze it")
    assert res["status"] == "error" and "budget exceeded" in res["error"]
    assert headless.exit_code(res) == headless.EXIT_ERROR
    assert res["files_produced"] == [] and res["key_findings"] == []
    assert res["delegation_results"][0]["status"] == "error"


def test_some_delegations_failed_is_partial_with_its_own_exit_code():
    meta = _Meta(script=[_entry(1), _entry(2, "error", error="boom")])
    res = MetaAdapter().headless_run(meta, "analyze both")
    assert res["status"] == "partial" and headless.exit_code(res) == headless.EXIT_RESERVATIONS
    assert res["files_produced"] == ["/out/1.png"]            # not the failed one's
    assert any("#2" in w and "boom" in w for w in res["warnings"])
    assert "error" not in res


def test_a_restored_sessions_earlier_delegations_are_not_this_tasks():
    meta = _Meta(ledger=[_entry(1, "error", error="old failure")], script=[_entry(2)])
    res = MetaAdapter().headless_run(meta, "next step")
    assert res["status"] == "success" and res["delegations"] == 1
    assert [d["index"] for d in res["delegation_results"]] == [2]
    assert res["files_produced"] == ["/out/2.png"]


def test_a_chat_error_is_an_error():
    meta = _Meta(reply="❌ Error: provider down", chat_error="provider down")
    res = MetaAdapter().headless_run(meta, "t")
    assert res["status"] == "error" and res["error"] == "provider down"


def test_the_meta_result_carries_the_delegations_verification():
    unverified = verification_summary([_row("a2", verified=False, reason="salvaged")])
    meta = _Meta(script=[_entry(1, verification=verification_summary([_row("a1")])),
                         _entry(2, verification=unverified),
                         _entry(3, mode="planning")])
    res = MetaAdapter().headless_run(meta, "analyze then plan")
    assert res["status"] == "success"
    assert res["verification"]["analyses"] == 2 and res["verification"]["all_verified"] is False
    assert res["delegation_results"][1]["verification"]["unverified"][0]["analysis_id"] == "a2"
    assert headless.exit_code(res) == headless.EXIT_OK
    assert headless.exit_code(res, strict=True) == headless.EXIT_RESERVATIONS
    json.dumps(res)                                           # what -p prints


def test_an_analysis_delegation_that_analysed_nothing_is_named():
    """Seen live: the specialist could not find the file, replied, and the
    delegation closed as a success with no analysis in it."""
    meta = _Meta(script=[_entry(1, verification=verification_summary([]), files_produced=[],
                                key_findings=[])],
                 reply="The file was not found.")
    res = MetaAdapter().headless_run(meta, "analyze missing.csv")
    assert res["status"] == "success" and res["verification"]["analyses"] == 0
    assert res["warnings"] == ["Delegation #1 (analysis) ran no analysis."]
    assert headless.exit_code(res) == headless.EXIT_OK
    assert headless.exit_code(res, strict=True) == headless.EXIT_RESERVATIONS


def test_a_turn_with_no_analysis_has_no_verification():
    res = turn_result(_Meta(script=[_entry(1, mode="planning")]), "plan", "Planned.", 0)
    assert "verification" not in res and headless.exit_code(res, strict=True) == headless.EXIT_OK


def test_the_ledger_keeps_the_childs_verification(tmp_path, monkeypatch):
    from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    m = MetaOrchestratorAgent(base_dir=str(tmp_path / "meta"),
                              model_name="anthropic/claude-sonnet-4-5",
                              meta_mode=MetaMode.AUTONOMOUS, launch_dir=str(tmp_path))
    entry = m._open_delegation("analysis", "fit it", None, None)
    v = verification_summary([_row("a", verified=False, reason="salvaged")])
    m._close_delegation(entry, {"status": "success", "summary": "s", "verification": v})
    assert m._delegation_ledger[-1]["verification"] == v
    res = turn_result(m, "fit it", "Fitted.", 0)
    assert res["verification"]["all_verified"] is False
    saved = json.loads((tmp_path / "meta" / "checkpoint.json").read_text())
    assert saved["delegation_ledger"][-1]["verification"]["unverified"][0]["analysis_id"] == "a"


# ── exit codes ───────────────────────────────────────────────────

def test_exit_codes():
    ok = {"status": "success"}
    assert headless.exit_code(ok) == 0 and headless.exit_code(ok, strict=True) == 0
    assert headless.exit_code({"status": "error"}) == 1
    assert headless.exit_code({"status": "partial"}) == 3
    unverified = {"status": "success", "verification": {"all_verified": False}}
    assert headless.exit_code(unverified) == 0                 # unchanged without --strict
    assert headless.exit_code(unverified, strict=True) == 3
    # no analysis ran: a reply, a success, and nothing behind it
    nothing = {"status": "success", "verification": {"analyses": 0, "all_verified": None}}
    assert headless.exit_code(nothing) == 0 and headless.exit_code(nothing, strict=True) == 3
    # a mode with no analyses to verify (plan, simulate) is not judged
    assert headless.exit_code({"status": "success", "campaign_state": {}}, strict=True) == 0
    verified = {"status": "success", "verification": {"analyses": 2, "all_verified": True}}
    assert headless.exit_code(verified, strict=True) == 0
    # an error stays an error under --strict
    assert headless.exit_code({"status": "error", "verification": {"all_verified": False}},
                              strict=True) == 1


def test_strict_flag_reaches_the_exit_code(tmp_path, capsys, monkeypatch):
    from shell_fakes import FakeAdapter, make_args
    monkeypatch.chdir(tmp_path)

    class Unverified(FakeAdapter):
        def headless_run(self, agent, task, context=None):
            return {"status": "success", "summary": "fitted",
                    "verification": verification_summary(
                        [_row("a"), _row("b", verified=False, reason="salvaged fit")])}

    adapter = Unverified()
    args = make_args(adapter, ["-p", "t", "--yes"])
    assert headless.run(adapter, args, "t") == 0
    out = capsys.readouterr().out
    assert "1 of 2 analyses verified" in out and "not verified: b (salvaged fit)" in out
    args = make_args(adapter, ["-p", "t", "--yes", "--strict", "--output-format", "json"])
    assert headless.run(adapter, args, "t") == 3
    assert json.loads(capsys.readouterr().out)["verification"]["all_verified"] is False
