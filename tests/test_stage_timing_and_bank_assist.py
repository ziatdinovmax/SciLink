"""Run instrumentation: per-stage timings and the script bank's assist log.

Two questions the analysis agents could not answer about their own runs:

  - *where did the time go?* — ``StageTimer`` wraps each pipeline controller
    and records wall-clock plus the delta of the always-on LLM counters in
    ``scilink.tracing``;
  - *did the script bank help?* — ``record_bank_assist`` stamps every QC-loop
    item with the mode that served it and how many verification iterations it
    then needed, and appends the same event to the bank's assist log, which
    ``assist_stats`` rolls up against the ``mode="none"`` baseline.

No LLM calls anywhere.
"""

import json
import time
from types import SimpleNamespace

import pytest

from scilink import tracing
from scilink.agents.exp_agents import _stage_timing
from scilink.agents.exp_agents._qc_engine import QCItemContext, record_bank_assist
from scilink.agents.exp_agents._stage_timing import StageTimer
from scilink.skills._shared import _script_bank as sb


@pytest.fixture(autouse=True)
def _isolated_bank(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path))
    monkeypatch.setenv("SCILINK_MEMORY", "1")
    monkeypatch.delenv("SCILINK_SCRIPT_BANK", raising=False)


# ──────────────────────────────────────────────────────────────
# LLM counters
# ──────────────────────────────────────────────────────────────

class TestLLMCounters:
    def test_counts_without_tracing_enabled(self):
        assert not tracing.is_enabled()
        before = tracing.llm_counters()
        tracing.note_llm_call(1.5, prompt_tokens=100, completion_tokens=20)
        tracing.note_llm_call(0.5)
        after = tracing.llm_counters()
        assert after["calls"] - before["calls"] == 2
        assert after["seconds"] - before["seconds"] == pytest.approx(2.0)
        assert after["prompt_tokens"] - before["prompt_tokens"] == 100
        assert after["completion_tokens"] - before["completion_tokens"] == 20

    def test_garbage_never_raises(self):
        before = tracing.llm_counters()["calls"]
        tracing.note_llm_call("not a number", object(), object())
        # The bad call is dropped whole rather than half-counted.
        assert tracing.llm_counters()["calls"] in (before, before + 1)

    def test_wrapper_hook_counts_when_tracing_is_off(self):
        from scilink.wrappers.litellm_wrapper import _record_trace
        resp = SimpleNamespace(
            choices=[], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
        before = tracing.llm_counters()
        _record_trace("m", [], resp, 0.25)
        after = tracing.llm_counters()
        assert after["calls"] - before["calls"] == 1
        assert after["prompt_tokens"] - before["prompt_tokens"] == 7


# ──────────────────────────────────────────────────────────────
# StageTimer
# ──────────────────────────────────────────────────────────────

class _Planning:
    def execute(self, state):
        tracing.note_llm_call(0.2, 50, 10)
        tracing.note_llm_call(0.3, 60, 10)
        return {**state, "planned": True}


class _Numerics:
    def execute(self, state):
        time.sleep(0.02)
        return {**state, "fitted": True}


class _Boom:
    def execute(self, state):
        raise RuntimeError("stage failed")


class TestStageTimer:
    def test_records_per_stage_llm_deltas(self):
        t = StageTimer()
        state = t.run(_Planning(), {})
        state = t.run(_Numerics(), state)
        assert state == {"planned": True, "fitted": True}
        s = t.summary()
        assert [r["stage"] for r in s["stages"]] == ["_Planning", "_Numerics"]
        planning, numerics = s["stages"]
        assert planning["llm_calls"] == 2
        assert planning["llm_seconds"] == pytest.approx(0.5)
        assert planning["prompt_tokens"] == 110
        assert numerics["llm_calls"] == 0
        assert numerics["seconds"] >= 0.02
        assert s["llm_calls"] == 2
        assert s["by_stage"]["_Planning"]["runs"] == 1
        json.dumps(s)  # persisted into analysis_results.json

    def test_repeated_stage_aggregates(self):
        t = StageTimer()
        for _ in range(3):
            t.run(_Planning(), {})
        s = t.summary()
        assert len(s["stages"]) == 3
        assert s["by_stage"]["_Planning"] == {
            "runs": 3, "seconds": pytest.approx(s["staged_seconds"], abs=0.01),
            "llm_calls": 6, "llm_seconds": pytest.approx(1.5)}

    def test_failing_stage_is_recorded_and_reraised(self):
        t = StageTimer()
        with pytest.raises(RuntimeError, match="stage failed"):
            t.run(_Boom(), {})
        assert t.records[0]["status"] == "error"
        assert t.records[0]["stage"] == "_Boom"

    def test_custom_name_and_extra_fields(self):
        t = StageTimer()
        t.run(_Numerics(), {}, name="tier2:_Numerics", tier=2)
        assert t.records[0]["stage"] == "tier2:_Numerics"
        assert t.records[0]["tier"] == 2

    def test_llm_fraction_capped_for_threaded_stages(self):
        t = StageTimer()
        with t.stage("fanout"):
            time.sleep(0.01)
            tracing.note_llm_call(50.0)  # worker threads sum past wall-clock
        assert t.summary()["llm_fraction"] == 1.0

    def test_concurrent_timers_flag_shared_counts(self):
        outer, inner = StageTimer(), StageTimer()
        with outer.stage("a"):
            with inner.stage("b"):
                pass
        assert inner.records[0].get("llm_counts_shared") is True
        assert outer.records[0].get("llm_counts_shared") is True
        solo = StageTimer()
        with solo.stage("c"):
            pass
        assert "llm_counts_shared" not in solo.records[0]
        assert _stage_timing._active_timers == 0

    def test_empty_summary(self):
        s = StageTimer().summary()
        assert s["stages"] == [] and s["llm_fraction"] is None

    def test_log_summary_never_raises(self):
        t = StageTimer()
        t.run(_Planning(), {})
        lines = []
        t.log_summary(SimpleNamespace(info=lines.append))
        assert any("Stage timing" in ln for ln in lines)
        t.log_summary(None)  # a broken logger must not fail the run


# ──────────────────────────────────────────────────────────────
# Bank assist
# ──────────────────────────────────────────────────────────────

def _ctx(item_idx=0, **attrs):
    ctx = QCItemContext(state={"output_dir": "/tmp/session_42"}, data=None,
                        data_path="x.csv", item_name="spec_a", item_idx=item_idx)
    for k, v in attrs.items():
        setattr(ctx, k, v)
    return ctx


def _res(iterations=2, approved=True, **extra):
    return {"success": True,
            "quality_history": {"approved": approved,
                                "verification_iterations": [{}] * iterations},
            **extra}


MATCH = {"record": {"id": "abc12345"}, "score": 0.51, "fingerprint_score": 0.6}
HOST = SimpleNamespace(output_dir="/tmp/session_42")


class TestRecordBankAssist:
    def test_baseline_mode_none(self):
        res = _res(iterations=4)
        block = record_bank_assist(HOST, _ctx(), res, domain="curve_fitting",
                                   seconds=12.345)
        assert block["mode"] == "none" and block["record_id"] is None
        assert block["iterations"] == 4 and block["approved"] is True
        assert block["seconds"] == 12.35
        assert res["bank_assist"] is block
        [ev] = sb.read_assist_log()
        assert ev["mode"] == "none" and ev["session"] == "session_42"

    def test_exemplar_mode(self):
        block = record_bank_assist(HOST, _ctx(bank_exemplar=MATCH), _res(1),
                                   domain="curve_fitting")
        assert block["mode"] == "exemplar"
        assert block["record_id"] == "abc12345" and block["score"] == 0.51

    def test_edit_adapt_survived_vs_replaced(self):
        attempt = {"n_edits": 2, "executed": True}
        kept = record_bank_assist(
            HOST, _ctx(bank_exemplar=MATCH, bank_adapt_attempt=attempt),
            _res(0, bank_edit_adapt={"id": "abc12345"}), domain="curve_fitting")
        assert kept["mode"] == "edit_adapt" and kept["survived"] is True
        assert kept["n_edits"] == 2
        # A verification-loop refit drops the provenance: the adaptation ran
        # but is not what got accepted.
        replaced = record_bank_assist(
            HOST, _ctx(bank_exemplar=MATCH, bank_adapt_attempt=attempt),
            _res(5), domain="curve_fitting")
        assert replaced["mode"] == "edit_adapt" and replaced["survived"] is False

    def test_adaptation_that_never_applied_is_still_an_attempt(self):
        # Observed live: the model's edit list did not apply, the run fell
        # back to exemplar generation — but the LLM call was already spent.
        attempt = {"n_edits": None, "applied": False, "executed": False,
                   "fell_through": "edits do not apply: edit 3/6"}
        block = record_bank_assist(
            HOST, _ctx(bank_exemplar=MATCH, bank_adapt_attempt=attempt),
            _res(1), domain="curve_fitting")
        assert block["mode"] == "edit_adapt"
        assert block["applied"] is False and block["survived"] is False
        assert "edit 3/6" in block["fell_through"]

    def test_locked_reuse_and_replay_are_not_bank_events(self):
        assert record_bank_assist(
            HOST, _ctx(), _res(reuse_validity={"verdict": "good"}),
            domain="curve_fitting") is None
        assert record_bank_assist(
            HOST, _ctx(), _res(locked_replay=True),
            domain="hyperspectral", anchor_only=False) is None
        assert sb.read_assist_log() == []

    def test_non_anchor_skipped_unless_every_item_runs_the_loop(self):
        # Curve/image: only anchors run the verification loop.
        assert record_bank_assist(HOST, _ctx(item_idx=3), _res(0),
                                  domain="curve_fitting") is None
        # Hyperspectral: every target runs the full ladder.
        assert record_bank_assist(HOST, _ctx(item_idx=3), _res(0),
                                  domain="hyperspectral",
                                  anchor_only=False)["mode"] == "none"

    def test_never_raises_on_malformed_result(self):
        assert record_bank_assist(HOST, _ctx(), None, domain="x") is None
        assert record_bank_assist(HOST, _ctx(), {"quality_history": "nope"},
                                  domain="x") is None

    def test_inert_when_bank_disabled(self, monkeypatch):
        monkeypatch.setenv("SCILINK_SCRIPT_BANK", "0")
        res = _res()
        block = record_bank_assist(HOST, _ctx(), res, domain="curve_fitting")
        assert block is not None and res["bank_assist"] is block  # still on the result
        assert sb.read_assist_log() == []                          # nothing persisted


class TestAssistStats:
    def _seed(self):
        for it in (5, 7, 6):
            sb.log_assist({"domain": "curve_fitting", "mode": "none",
                           "iterations": it, "approved": True, "seconds": 300})
        sb.log_assist({"domain": "curve_fitting", "mode": "edit_adapt",
                       "score": 0.50, "iterations": 0, "approved": True,
                       "survived": True, "seconds": 40})
        sb.log_assist({"domain": "curve_fitting", "mode": "edit_adapt",
                       "score": 0.48, "iterations": 6, "approved": False,
                       "survived": False, "seconds": 380})
        sb.log_assist({"domain": "curve_fitting", "mode": "exemplar",
                       "score": 0.80, "iterations": 1, "approved": True})
        sb.log_assist({"domain": "image_analysis", "mode": "none",
                       "iterations": 3, "approved": True})

    def test_rollup_by_mode_and_score(self):
        self._seed()
        stats = sb.assist_stats()
        cf = stats["curve_fitting"]
        assert cf["n_events"] == 6
        assert cf["by_mode"]["none"] == {
            "n": 3, "approved_rate": 1.0, "mean_iterations": 6.0,
            "mean_seconds": 300.0}
        ea = cf["by_mode"]["edit_adapt"]
        assert ea["n"] == 2 and ea["survived_rate"] == 0.5
        assert ea["approved_rate"] == 0.5 and ea["mean_iterations"] == 3.0
        assert cf["by_score"]["0.45-0.55"]["n"] == 2
        assert cf["by_score"]["0.70-1.00"]["n"] == 1
        assert stats["image_analysis"]["n_events"] == 1

    def test_domain_filter_and_empty(self):
        assert sb.assist_stats() == {}
        self._seed()
        assert list(sb.assist_stats("image_analysis")) == ["image_analysis"]

    def test_torn_line_is_skipped(self):
        self._seed()
        with open(sb.assist_log_path(), "a", encoding="utf-8") as fh:
            fh.write('{"domain": "curve_fitting", "mode": "no')  # killed mid-write
        assert sb.assist_stats()["curve_fitting"]["n_events"] == 6

    def test_log_file_does_not_appear_as_a_record(self):
        self._seed()
        assert sb.list_records() == []


def test_bank_stats_cli(capsys):
    from scilink.cli import memory as cli
    TestAssistStats()._seed()
    assert cli._cmd_bank_stats(SimpleNamespace(domain=None, json=False)) == 0
    out = capsys.readouterr().out
    assert "curve_fitting: 6 QC-loop item(s)" in out
    assert "edit_adapt" in out and "survived=50%" in out
    assert cli._cmd_bank_stats(SimpleNamespace(domain=None, json=True)) == 0
    assert json.loads(capsys.readouterr().out)["curve_fitting"]["n_events"] == 6
