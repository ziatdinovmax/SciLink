"""Tests for the live-session JSONL replay path."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from scilink.agents.exp_agents.live_data_sources import CallbackSource, MtimePollFileSource
from scilink.agents.exp_agents.live_replay import ReplayReport, replay_jsonl
from scilink.agents.exp_agents.live_session import LiveSession
from scilink.agents.exp_agents.live_triggers import (
    ThresholdCrossTrigger,
    TriggerPolicy,
    VerdictChangeTrigger,
    default_policy,
)
from scilink.agents.exp_agents.live_types import LiveReadingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fixture_jsonl(path: Path, readings: list[dict]) -> None:
    """Write a minimal fixture JSONL with the given reading entries."""
    lines: list[str] = []
    base = time.time() - 100
    lines.append(json.dumps({"kind": "session_start", "timestamp": base}))
    for i, t in enumerate(readings):
        lines.append(json.dumps({
            "kind": "reading",
            "timestamp": base + 1.0 * (i + 1),
            "metric": t.get("metric", 0.5),
            "metric_name": t.get("metric_name", "figure_of_merit"),
            "verdict": t.get("verdict", "marginal"),
            "detected_features": t.get("features", []),
            "notes": t.get("notes", ""),
            "raw": t.get("raw", {}),
        }))
    lines.append(json.dumps({"kind": "session_end", "timestamp": base + 1.0 * (len(readings) + 1)}))
    path.write_text("\n".join(lines) + "\n")


class _MockOrch:
    def __init__(self):
        self.calls: list[dict] = []

    def run_task(self, task: str, context: dict | None = None,
                 autonomy=None) -> dict:
        self.calls.append({"task": task, "context": context})
        return {"status": "success", "task": task, "summary": "replayed"}


# ---------------------------------------------------------------------------
# Reading stream reconstruction
# ---------------------------------------------------------------------------


def test_replay_reproduces_tick_count(tmp_path):
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [{"metric": 0.4}, {"metric": 0.5}, {"metric": 0.6}])
    report = replay_jsonl(p)
    assert report.reading_count == 3


def test_replay_preserves_tick_metadata(tmp_path):
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [
        {"metric": 0.5, "verdict": "marginal", "features": [{"position": 28.4}]},
        {"metric": 0.8, "verdict": "accept", "features": [{"position": 28.4}, {"position": 47.3}]},
    ])
    seen: list[LiveReadingResult] = []
    replay_jsonl(p, on_reading=seen.append)
    assert len(seen) == 2
    assert seen[0].verdict == "marginal"
    assert seen[1].primary_metric == 0.8
    assert any(f["position"] == 47.3 for f in seen[1].detected_features)


def test_replay_ignores_non_tick_lines(tmp_path):
    """session_start, session_end, trigger, llm_response lines must
    not be counted as readings."""
    p = tmp_path / "live.jsonl"
    lines = [
        json.dumps({"kind": "session_start", "timestamp": 100}),
        json.dumps({"kind": "reading", "timestamp": 101, "metric": 0.5, "verdict": "marginal"}),
        json.dumps({"kind": "trigger", "timestamp": 101, "name": "verdict_change", "details": {}}),
        json.dumps({"kind": "llm_response", "timestamp": 101, "status": "success", "summary": "..."}),
        json.dumps({"kind": "reading", "timestamp": 102, "metric": 0.8, "verdict": "accept"}),
        json.dumps({"kind": "session_end", "timestamp": 103}),
    ]
    p.write_text("\n".join(lines) + "\n")
    report = replay_jsonl(p)
    assert report.reading_count == 2


def test_replay_skips_malformed_json(tmp_path):
    p = tmp_path / "live.jsonl"
    p.write_text(
        json.dumps({"kind": "reading", "timestamp": 1, "metric": 0.5, "verdict": "marginal"}) + "\n"
        + "{not valid json}\n"
        + json.dumps({"kind": "reading", "timestamp": 2, "metric": 0.7, "verdict": "marginal"}) + "\n"
    )
    report = replay_jsonl(p)
    assert report.reading_count == 2


# ---------------------------------------------------------------------------
# Trigger evaluation determinism
# ---------------------------------------------------------------------------


def test_replay_reproduces_trigger_events_deterministically(tmp_path):
    """Same fixture + same policy → same trigger events on every run."""
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [
        {"verdict": "marginal"},
        {"verdict": "marginal"},
        {"verdict": "accept"},
        {"verdict": "accept"},
        {"verdict": "reject"},
    ])
    policy_factory = lambda: TriggerPolicy(triggers=[VerdictChangeTrigger()])
    r1 = replay_jsonl(p, trigger_policy=policy_factory())
    r2 = replay_jsonl(p, trigger_policy=policy_factory())
    assert r1.trigger_event_counts == r2.trigger_event_counts
    assert r1.trigger_count == r2.trigger_count == 2  # marginal→accept, accept→reject


def test_replay_custom_policy_produces_different_events(tmp_path):
    """Loosening a threshold should make a configurable trigger fire more."""
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [
        {"metric": 0.3, "verdict": "marginal"},
        {"metric": 0.5, "verdict": "marginal"},
        {"metric": 0.65, "verdict": "marginal"},
        {"metric": 0.4, "verdict": "marginal"},
        {"metric": 0.72, "verdict": "marginal"},
    ])
    strict = replay_jsonl(p, trigger_policy=TriggerPolicy(
        triggers=[ThresholdCrossTrigger(threshold=0.7, direction="above")]
    ))
    loose = replay_jsonl(p, trigger_policy=TriggerPolicy(
        triggers=[ThresholdCrossTrigger(threshold=0.5, direction="above")]
    ))
    assert loose.trigger_count >= strict.trigger_count


# ---------------------------------------------------------------------------
# LLM modes
# ---------------------------------------------------------------------------


def test_llm_mode_skip_does_not_call_orchestrator(tmp_path):
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [
        {"verdict": "marginal"}, {"verdict": "accept"},
    ])
    orch = _MockOrch()
    replay_jsonl(p, orchestrator=orch, llm_mode="skip")
    assert orch.calls == []


def test_llm_mode_skip_echoes_original_responses(tmp_path):
    """The 'skip' mode reports the historical llm_response lines."""
    p = tmp_path / "live.jsonl"
    lines = [
        json.dumps({"kind": "session_start", "timestamp": 100}),
        json.dumps({"kind": "reading", "timestamp": 101, "metric": 0.5, "verdict": "marginal"}),
        json.dumps({"kind": "reading", "timestamp": 102, "metric": 0.8, "verdict": "accept"}),
        json.dumps({"kind": "llm_response", "timestamp": 102,
                    "trigger": "verdict_change", "status": "success",
                    "summary": "Identified Si Fd-3m"}),
        json.dumps({"kind": "session_end", "timestamp": 103}),
    ]
    p.write_text("\n".join(lines) + "\n")
    report = replay_jsonl(p, llm_mode="skip")
    assert len(report.llm_responses) == 1
    assert "Si Fd-3m" in report.llm_responses[0]["summary"]


def test_llm_mode_redo_calls_orchestrator_per_trigger(tmp_path):
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [
        {"verdict": "marginal"},
        {"verdict": "accept"},   # → verdict_change event
        {"verdict": "reject"},   # → verdict_change event
    ])
    orch = _MockOrch()
    report = replay_jsonl(
        p,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        orchestrator=orch,
        llm_mode="redo",
    )
    assert len(orch.calls) == 2
    assert report.llm_responses[0]["trigger"] == "verdict_change"


def test_llm_mode_redo_requires_orchestrator(tmp_path):
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [{"verdict": "marginal"}])
    with pytest.raises(ValueError, match="orchestrator"):
        replay_jsonl(p, llm_mode="redo")


def test_invalid_llm_mode_raises(tmp_path):
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [{"verdict": "marginal"}])
    with pytest.raises(ValueError, match="llm_mode"):
        replay_jsonl(p, llm_mode="nope")


# ---------------------------------------------------------------------------
# Cadence + callbacks
# ---------------------------------------------------------------------------


def test_no_speed_runs_instantly(tmp_path):
    """Default (None) replay has no sleeping between readings."""
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [{"metric": 0.5}] * 10)
    t0 = time.monotonic()
    replay_jsonl(p)
    assert time.monotonic() - t0 < 0.2


def test_speed_controls_cadence(tmp_path):
    """At speed=2.0, replay should take ~half the original reading delta."""
    p = tmp_path / "live.jsonl"
    # Fixture has 1.0s deltas between readings; speed=10.0 → ~0.1s between readings
    _write_fixture_jsonl(p, [{"metric": float(i)} for i in range(4)])
    t0 = time.monotonic()
    replay_jsonl(p, speed=10.0)
    elapsed = time.monotonic() - t0
    assert 0.05 < elapsed < 1.5  # ~0.3s for 3 inter-reading gaps at 0.1s each


def test_on_event_callback_fires_for_each_trigger(tmp_path):
    p = tmp_path / "live.jsonl"
    _write_fixture_jsonl(p, [
        {"verdict": "marginal"},
        {"verdict": "accept"},
        {"verdict": "reject"},
    ])
    seen = []
    replay_jsonl(
        p,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        on_event=lambda ev: seen.append(ev.name),
    )
    assert seen == ["verdict_change", "verdict_change"]


# ---------------------------------------------------------------------------
# Round-trip with a real LiveSession recording
# ---------------------------------------------------------------------------


def test_round_trip_from_live_session(tmp_path):
    """Run a LiveSession, then replay its JSONL — events should match."""
    from unittest.mock import MagicMock
    orch = MagicMock()
    orch.run_task.return_value = {"status": "success", "summary": "ok"}
    src = CallbackSource()

    reading_idx = {"i": 0}
    verdict_seq = ["marginal", "marginal", "accept", "accept", "reject"]

    def reading_fn(data, ss, skill):
        v = verdict_seq[reading_idx["i"]]
        reading_idx["i"] += 1
        return LiveReadingResult(
            timestamp=time.time(),
            primary_metric=0.5,
            metric_name="m",
            verdict=v,
        )

    path = tmp_path / "live.jsonl"
    s = LiveSession(
        orch, src, reading_fn,
        reading_interval_sec=0.05,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
        history_path=path,
    )
    s.start()
    try:
        for _ in range(5):
            src.push(text="d")
            time.sleep(0.12)
        time.sleep(0.4)
    finally:
        s.stop(timeout=2.0)

    # Replay the recorded session with the same policy
    report = replay_jsonl(
        path,
        trigger_policy=TriggerPolicy(triggers=[VerdictChangeTrigger()]),
    )
    # Live session should have produced 2 verdict-change events
    # (marginal→accept, accept→reject); replay should reproduce them.
    assert report.trigger_count == 2
    assert report.trigger_event_counts == {"verdict_change": 2}
