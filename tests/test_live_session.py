"""Tests for the LiveSession class.

Exercises lifecycle, tick cadence, single-flight LLM dispatch,
JSONL persistence, and callback / error paths. Uses a mocked
``AnalysisOrchestratorAgent.run_task`` so no LLM is involved.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scilink.agents.exp_agents.live_data_sources import (
    CallbackSource,
    LatestData,
    MtimePollFileSource,
)
from scilink.agents.exp_agents.live_session import LiveSession
from scilink.agents.exp_agents.live_triggers import (
    ManualTrigger,
    TriggerPolicy,
    VerdictChangeTrigger,
)
from scilink.agents.exp_agents.live_types import LiveTickResult


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _MockOrch:
    """A minimal stand-in for AnalysisOrchestratorAgent.

    Records calls to ``run_task`` so tests can assert what context the
    LiveSession built. ``delay`` simulates an in-flight LLM call so
    tests can probe single-flight semantics.
    """

    def __init__(self, *, delay: float = 0.0):
        self.calls: list[dict] = []
        self.delay = delay
        self._call_lock = threading.Lock()

    def run_task(self, task: str, context: dict | None = None,
                 autonomy=None) -> dict:
        with self._call_lock:
            self.calls.append({"task": task, "context": context, "autonomy": autonomy})
        if self.delay:
            time.sleep(self.delay)
        return {
            "status": "success",
            "task": task,
            "summary": f"interp #{len(self.calls)}",
            "key_findings": [],
            "files_produced": [],
            "warnings": [],
        }


def _make_tick_fn(metric_seq: list[float] | None = None,
                  verdict_seq: list[str] | None = None,
                  features_seq: list[list[dict]] | None = None):
    """Build a tick_fn that emits a deterministic sequence of LiveTickResult."""
    idx = {"i": 0}

    def tick(latest_data: LatestData, session_state: dict, skill_state: dict) -> LiveTickResult:
        i = idx["i"]
        idx["i"] += 1
        return LiveTickResult(
            timestamp=time.time(),
            primary_metric=(metric_seq[i] if metric_seq else 0.5),
            metric_name="figure_of_merit",
            verdict=(verdict_seq[i] if verdict_seq else "marginal"),
            detected_features=(features_seq[i] if features_seq else []),
            notes=f"tick #{i}",
        )

    return tick, idx


def _wait_for(predicate, *, timeout: float = 3.0, interval: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_start_stop_idempotent(tmp_path):
    orch = _MockOrch()
    source = CallbackSource()
    tick_fn, _ = _make_tick_fn()
    s = LiveSession(orch, source, tick_fn, tick_interval_sec=0.05)
    s.start()
    s.start()  # idempotent — does not raise
    s.stop()
    s.stop()  # idempotent


def test_start_writes_session_start_line(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn()
    path = tmp_path / "live.jsonl"
    s = LiveSession(orch, src, tick_fn, history_path=path, tick_interval_sec=0.1)
    s.start()
    s.stop(timeout=1.0)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines[0]["kind"] == "session_start"
    assert lines[-1]["kind"] == "session_end"


# ---------------------------------------------------------------------------
# Tick cadence + history
# ---------------------------------------------------------------------------


def test_tick_fires_when_data_arrives(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, idx = _make_tick_fn(metric_seq=[0.5, 0.6, 0.7])
    s = LiveSession(orch, src, tick_fn, tick_interval_sec=0.05)
    s.start()
    try:
        for _ in range(3):
            src.push(text="data")
            time.sleep(0.15)
        assert _wait_for(lambda: idx["i"] >= 3, timeout=2.0)
        hist = s.history()
        assert len(hist) >= 3
        assert s.latest() is not None
        assert s.latest().primary_metric == pytest.approx(hist[-1].primary_metric)
    finally:
        s.stop()


def test_tick_skipped_when_source_returns_none(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, idx = _make_tick_fn()
    s = LiveSession(orch, src, tick_fn, tick_interval_sec=0.05)
    s.start()
    try:
        time.sleep(0.5)  # No pushes → no ticks
        assert idx["i"] == 0
        assert s.history() == []
    finally:
        s.stop()


def test_tick_fn_exception_does_not_crash_loop(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    call_count = {"n": 0}

    def flaky_tick(data, ss, skill):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first call boom")
        return LiveTickResult(
            timestamp=time.time(), primary_metric=0.9,
            metric_name="m", verdict="accept",
        )

    s = LiveSession(orch, src, flaky_tick, tick_interval_sec=0.05)
    s.start()
    try:
        src.push(text="d1")
        time.sleep(0.2)
        src.push(text="d2")
        assert _wait_for(lambda: call_count["n"] >= 2 and s.latest() is not None)
        assert s.latest().verdict == "accept"
    finally:
        s.stop()


# ---------------------------------------------------------------------------
# Trigger evaluation + JSONL persistence
# ---------------------------------------------------------------------------


def test_trigger_event_emits_jsonl_line(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn(verdict_seq=["marginal", "marginal", "accept"])
    path = tmp_path / "live.jsonl"
    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        history_path=path,
    )
    s.start()
    try:
        for _ in range(3):
            src.push(text="d")
            time.sleep(0.15)
        assert _wait_for(lambda: len(orch.calls) >= 1, timeout=3.0)
    finally:
        s.stop(timeout=2.0)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    kinds = [l["kind"] for l in lines]
    assert kinds[0] == "session_start"
    assert "tick" in kinds
    assert "trigger" in kinds
    assert "llm_response" in kinds
    assert kinds[-1] == "session_end"


def test_jsonl_is_chronological(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn(verdict_seq=["marginal", "accept", "reject"])
    path = tmp_path / "live.jsonl"
    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        history_path=path,
    )
    s.start()
    try:
        for _ in range(3):
            src.push(text="d")
            time.sleep(0.15)
        time.sleep(0.5)
    finally:
        s.stop(timeout=2.0)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    timestamps = [l.get("timestamp", 0) for l in lines]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Single-flight LLM dispatch
# ---------------------------------------------------------------------------


def test_single_flight_coalesces_concurrent_events(tmp_path):
    orch = _MockOrch(delay=0.4)  # each LLM call takes 0.4s
    src = CallbackSource()
    # Three back-to-back verdict changes — should NOT produce three LLM calls
    tick_fn, _ = _make_tick_fn(verdict_seq=["marginal", "accept", "marginal", "accept"])
    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
    )
    s.start()
    try:
        for _ in range(4):
            src.push(text="d")
            time.sleep(0.08)
        # Let everything settle
        time.sleep(2.0)
    finally:
        s.stop(timeout=3.0)
    # Three verdict transitions produce three trigger events, but the
    # dispatcher batches the ones that arrive during the first LLM call
    # → strictly fewer than 3 calls (usually 1 or 2).
    assert len(orch.calls) < 3
    assert len(orch.calls) >= 1


def test_llm_busy_flag_reflects_state(tmp_path):
    orch = _MockOrch(delay=0.5)
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn(verdict_seq=["marginal", "accept"])
    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
    )
    s.start()
    try:
        src.push(text="d1")
        time.sleep(0.1)
        src.push(text="d2")
        time.sleep(0.3)  # LLM should be busy now
        assert s.llm_busy is True
        time.sleep(1.0)  # Wait for it to finish
        assert s.llm_busy is False
    finally:
        s.stop(timeout=2.0)


def test_llm_run_task_failure_logged_not_raised(tmp_path):
    class _RaisingOrch:
        def run_task(self, task, context=None, autonomy=None):
            raise RuntimeError("API down")

    path = tmp_path / "live.jsonl"
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn(verdict_seq=["marginal", "accept"])
    s = LiveSession(
        _RaisingOrch(), src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        history_path=path,
    )
    s.start()
    try:
        for _ in range(2):
            src.push(text="d")
            time.sleep(0.15)
        time.sleep(0.5)
    finally:
        s.stop(timeout=2.0)
    # Should write an error-status llm_response line, not crash the session
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    err_lines = [l for l in lines if l["kind"] == "llm_response" and l.get("status") == "error"]
    assert err_lines


# ---------------------------------------------------------------------------
# Callbacks + manual trigger
# ---------------------------------------------------------------------------


def test_on_tick_callback_fires_for_each_tick(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn()
    received: list[LiveTickResult] = []
    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        on_tick=received.append,
    )
    s.start()
    try:
        for _ in range(3):
            src.push(text="d")
            time.sleep(0.1)
        assert _wait_for(lambda: len(received) >= 3, timeout=2.0)
    finally:
        s.stop()


def test_on_llm_response_callback_fires_with_event_and_result(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn(verdict_seq=["marginal", "accept"])
    seen: list[tuple] = []

    def callback(event, result):
        seen.append((event.name, result.get("status")))

    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        on_llm_response=callback,
    )
    s.start()
    try:
        for _ in range(2):
            src.push(text="d")
            time.sleep(0.15)
        assert _wait_for(lambda: len(seen) >= 1, timeout=2.0)
    finally:
        s.stop()
    assert seen[0] == ("verdict_change", "success")


def test_force_interpretation_fires_manual_trigger(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn()
    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[ManualTrigger()]),
    )
    s.start()
    try:
        src.push(text="d")
        time.sleep(0.15)
        s.force_interpretation()
        src.push(text="d")
        assert _wait_for(lambda: len(orch.calls) >= 1, timeout=2.0)
    finally:
        s.stop()
    # The dispatcher saw a manual event
    assert any(c["context"]["trigger"]["name"] == "manual" for c in orch.calls)


def test_force_interpretation_without_manual_trigger_is_noop(tmp_path):
    orch = _MockOrch()
    src = CallbackSource()
    tick_fn, _ = _make_tick_fn()
    s = LiveSession(
        orch, src, tick_fn,
        tick_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),  # no manual
    )
    s.start()
    try:
        s.force_interpretation()  # no-op; should not raise
        time.sleep(0.2)
    finally:
        s.stop()
    assert orch.calls == []


# ---------------------------------------------------------------------------
# End-to-end: file source + simple peak-count tick_fn
# ---------------------------------------------------------------------------


def _peak_count_tick(data: LatestData, session_state: dict, skill_state: dict) -> LiveTickResult:
    """Trivial tick_fn for the end-to-end stub test: count '|' chars as peaks."""
    text = data.text or ""
    n_peaks = text.count("|")
    verdict = "accept" if n_peaks >= 3 else "marginal"
    features = [{"position": i} for i, c in enumerate(text) if c == "|"]
    return LiveTickResult(
        timestamp=time.time(),
        primary_metric=float(n_peaks),
        metric_name="peak_count",
        verdict=verdict,
        detected_features=features,
        notes=f"detected {n_peaks} peaks",
    )


def test_end_to_end_with_file_source(tmp_path):
    """Mtime-poll file source + peak-count tick_fn. Verifies the full
    pipeline (tick → trigger → LLM dispatch → JSONL) end-to-end against
    a file that grows over time. MtimePollFileSource returns the full
    content on each change so peak count accumulates as bars are
    appended."""
    orch = _MockOrch()
    data_path = tmp_path / "live_scan.txt"
    data_path.write_text("")  # empty start
    src = MtimePollFileSource(data_path)
    s = LiveSession(
        orch, src, _peak_count_tick,
        tick_interval_sec=0.1,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        history_path=tmp_path / "live.jsonl",
    )
    s.start()
    try:
        for _ in range(4):
            with data_path.open("a") as f:
                f.write("|")
            time.sleep(0.25)
        time.sleep(0.5)
    finally:
        s.stop(timeout=2.0)
    hist = s.history()
    assert len(hist) >= 4
    # Verdict transitioned marginal → accept somewhere
    verdicts = [t.verdict for t in hist]
    assert "marginal" in verdicts and "accept" in verdicts
    # LLM was called at least once (on the verdict change)
    assert len(orch.calls) >= 1
    # JSONL has the expected line kinds
    lines = [json.loads(l) for l in (tmp_path / "live.jsonl").read_text().splitlines()]
    kinds = {l["kind"] for l in lines}
    assert {"session_start", "tick", "trigger", "llm_response", "session_end"} <= kinds
