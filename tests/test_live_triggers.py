"""Tests for the live-monitoring trigger taxonomy."""

from __future__ import annotations

import time

import pytest

from scilink.agents.exp_agents.live_triggers import (
    ConfidenceReversalTrigger,
    HeartbeatTrigger,
    ManualTrigger,
    NewFeatureTrigger,
    QualitativeProgressTrigger,
    ThresholdCrossTrigger,
    Trigger,
    TriggerPolicy,
    VerdictChangeTrigger,
    default_policy,
    from_overrides,
)
from scilink.agents.exp_agents.live_types import LiveReadingResult, TriggerEvent


# --- Helpers ------------------------------------------------------------------

def _tick(
    *,
    ts: float = None,
    metric: float = 0.5,
    verdict: str = "marginal",
    features: list[dict] | None = None,
    metric_name: str = "figure_of_merit",
) -> LiveReadingResult:
    return LiveReadingResult(
        timestamp=ts if ts is not None else time.time(),
        primary_metric=metric,
        metric_name=metric_name,
        verdict=verdict,
        detected_features=features or [],
    )


def _hist(*readings: LiveReadingResult) -> list[LiveReadingResult]:
    return list(readings)


# --- Protocol -----------------------------------------------------------------

def test_all_triggers_satisfy_protocol():
    for t in (
        VerdictChangeTrigger(),
        NewFeatureTrigger(),
        ConfidenceReversalTrigger(),
        ThresholdCrossTrigger(threshold=0.5),
        HeartbeatTrigger(),
        ManualTrigger(),
    ):
        assert isinstance(t, Trigger), type(t).__name__


# --- VerdictChangeTrigger -----------------------------------------------------

def test_verdict_change_fires_on_transition():
    t = VerdictChangeTrigger()
    # First reading — records baseline, no event
    assert t.evaluate(_hist(_tick(verdict="marginal"))) is None
    # Same verdict — no event
    assert t.evaluate(_hist(_tick(verdict="marginal"), _tick(verdict="marginal"))) is None


def test_verdict_change_emits_event_on_change():
    t = VerdictChangeTrigger()
    t.evaluate(_hist(_tick(verdict="marginal")))
    ev = t.evaluate(_hist(_tick(verdict="marginal"), _tick(verdict="accept")))
    assert ev is not None
    assert ev.name == "verdict_change"
    assert ev.details == {"from": "marginal", "to": "accept"}


def test_verdict_change_reset_re_arms():
    t = VerdictChangeTrigger()
    t.evaluate(_hist(_tick(verdict="accept")))
    t.reset()
    # After reset, first reading is again baseline — no event
    assert t.evaluate(_hist(_tick(verdict="reject"))) is None


# --- NewFeatureTrigger --------------------------------------------------------

def test_new_feature_fires_when_feature_appears():
    t = NewFeatureTrigger(lookback=3)
    h = _hist(
        _tick(features=[{"position": 28.4}]),
        _tick(features=[{"position": 28.4}]),
        _tick(features=[{"position": 28.4}, {"position": 47.3}]),
    )
    ev = t.evaluate(h)
    assert ev is not None
    assert ev.name == "new_feature"
    assert any(f["position"] == 47.3 for f in ev.details["new_features"])


def test_new_feature_no_change_no_event():
    t = NewFeatureTrigger(lookback=3)
    h = _hist(_tick(features=[{"p": 1}]) for _ in range(4))
    h = _hist(*[_tick(features=[{"p": 1}]) for _ in range(4)])
    assert t.evaluate(h) is None


def test_new_feature_needs_baseline():
    t = NewFeatureTrigger(lookback=3)
    # Single reading — no baseline window yet
    assert t.evaluate(_hist(_tick(features=[{"p": 1}]))) is None


# --- ConfidenceReversalTrigger ------------------------------------------------

def test_confidence_reversal_fires_on_decline_after_monotonic_rise():
    t = ConfidenceReversalTrigger(window=4, min_reversal=0.05)
    h = _hist(
        _tick(metric=0.5),
        _tick(metric=0.6),
        _tick(metric=0.7),
        _tick(metric=0.8),
        _tick(metric=0.65),  # reversal
    )
    ev = t.evaluate(h)
    assert ev is not None
    assert ev.name == "confidence_reversal"
    assert ev.details["reversal_magnitude"] == pytest.approx(0.15, abs=1e-9)


def test_confidence_reversal_skips_when_not_monotonic():
    t = ConfidenceReversalTrigger(window=4)
    h = _hist(
        _tick(metric=0.5),
        _tick(metric=0.4),  # broke monotone
        _tick(metric=0.6),
        _tick(metric=0.7),
        _tick(metric=0.5),
    )
    assert t.evaluate(h) is None


def test_confidence_reversal_lower_is_better_direction():
    t = ConfidenceReversalTrigger(
        window=3, min_reversal=0.05, direction="lower_is_better",
    )
    # Monotonically decreasing cost; then increases
    h = _hist(
        _tick(metric=0.8),
        _tick(metric=0.6),
        _tick(metric=0.4),
        _tick(metric=0.55),  # reversal
    )
    ev = t.evaluate(h)
    assert ev is not None


def test_confidence_reversal_validates_direction():
    with pytest.raises(ValueError, match="direction"):
        ConfidenceReversalTrigger(direction="sideways")


# --- ThresholdCrossTrigger ----------------------------------------------------

def test_threshold_cross_above_fires_only_on_below_to_above():
    t = ThresholdCrossTrigger(threshold=0.7, direction="above")
    # Start below — baseline
    assert t.evaluate(_hist(_tick(metric=0.5))) is None
    # Stay below — no event
    assert t.evaluate(_hist(_tick(metric=0.6))) is None
    # Cross above — fires
    ev = t.evaluate(_hist(_tick(metric=0.75)))
    assert ev is not None
    assert ev.details["from"] == "below" and ev.details["to"] == "above"
    # Stay above — no event
    assert t.evaluate(_hist(_tick(metric=0.8))) is None
    # Cross below — does not fire (wrong direction)
    assert t.evaluate(_hist(_tick(metric=0.4))) is None
    # Cross above again — fires
    ev = t.evaluate(_hist(_tick(metric=0.9)))
    assert ev is not None


def test_threshold_cross_below_fires_only_on_above_to_below():
    t = ThresholdCrossTrigger(threshold=0.5, direction="below")
    t.evaluate(_hist(_tick(metric=0.8)))  # baseline above
    assert t.evaluate(_hist(_tick(metric=0.3))) is not None


def test_threshold_cross_validates_direction():
    with pytest.raises(ValueError, match="direction"):
        ThresholdCrossTrigger(threshold=0.5, direction="lateral")


# --- HeartbeatTrigger ---------------------------------------------------------

def test_heartbeat_fires_after_interval():
    t = HeartbeatTrigger(interval_sec=10.0)
    # First reading — records baseline
    assert t.evaluate(_hist(_tick(ts=100.0))) is None
    # 5s later — not enough
    assert t.evaluate(_hist(_tick(ts=105.0))) is None
    # 12s after baseline — fires
    ev = t.evaluate(_hist(_tick(ts=112.0)))
    assert ev is not None
    # Re-armed
    assert t.evaluate(_hist(_tick(ts=115.0))) is None
    ev = t.evaluate(_hist(_tick(ts=125.0)))
    assert ev is not None


# --- ManualTrigger ------------------------------------------------------------

def test_manual_fires_only_after_request():
    t = ManualTrigger()
    assert t.evaluate(_hist(_tick())) is None
    t.request()
    ev = t.evaluate(_hist(_tick()))
    assert ev is not None
    assert ev.name == "manual"
    # Single-shot: clears the pending flag
    assert t.evaluate(_hist(_tick())) is None


def test_manual_re_requests_independently():
    t = ManualTrigger()
    t.request()
    t.evaluate(_hist(_tick()))
    t.request()
    ev = t.evaluate(_hist(_tick()))
    assert ev is not None


# --- TriggerPolicy ------------------------------------------------------------

def test_policy_fires_union_of_constituent_triggers():
    policy = TriggerPolicy(triggers=[VerdictChangeTrigger(), ThresholdCrossTrigger(0.7)])
    policy.evaluate(_hist(_tick(metric=0.5, verdict="marginal")))
    events = policy.evaluate(_hist(_tick(metric=0.5, verdict="marginal"),
                                    _tick(metric=0.8, verdict="accept")))
    names = {e.name for e in events}
    assert "verdict_change" in names
    assert any(n.startswith("threshold_above") for n in names)


def test_policy_swallows_trigger_exceptions():
    """A broken trigger must not kill the reading loop."""
    class _Broken:
        name = "broken"
        def evaluate(self, history):
            raise RuntimeError("boom")
        def reset(self):
            pass

    policy = TriggerPolicy(triggers=[_Broken(), VerdictChangeTrigger()])
    policy.evaluate(_hist(_tick(verdict="marginal")))
    events = policy.evaluate(_hist(_tick(verdict="marginal"), _tick(verdict="accept")))
    # The broken trigger doesn't crash; the working one still fires.
    assert any(e.name == "verdict_change" for e in events)


def test_policy_reset_resets_constituents():
    v = VerdictChangeTrigger()
    policy = TriggerPolicy(triggers=[v])
    policy.evaluate(_hist(_tick(verdict="accept")))
    policy.reset()
    assert v._last_seen is None


# --- default_policy + from_overrides ------------------------------------------

def test_default_policy_includes_core_triggers():
    p = default_policy()
    names = {type(t).__name__ for t in p.triggers}
    assert "VerdictChangeTrigger" in names
    assert "NewFeatureTrigger" in names
    assert "ConfidenceReversalTrigger" in names
    assert "HeartbeatTrigger" in names
    assert "ManualTrigger" in names


def test_default_policy_can_disable_heartbeat():
    p = default_policy(enable_heartbeat=False)
    names = {type(t).__name__ for t in p.triggers}
    assert "HeartbeatTrigger" not in names


def test_default_policy_threshold_arg_adds_trigger():
    p = default_policy(threshold=0.7)
    assert any(isinstance(t, ThresholdCrossTrigger) for t in p.triggers)


def test_from_overrides_no_block_returns_default():
    p = from_overrides(None)
    assert any(isinstance(t, HeartbeatTrigger) for t in p.triggers)


def test_from_overrides_applies_skill_settings():
    p = from_overrides({
        "heartbeat_sec": 30.0,
        "confidence_threshold": 0.65,
        "reversal_direction": "lower_is_better",
    })
    heartbeat = next((t for t in p.triggers if isinstance(t, HeartbeatTrigger)), None)
    assert heartbeat is not None and heartbeat.interval_sec == 30.0
    threshold = next((t for t in p.triggers if isinstance(t, ThresholdCrossTrigger)), None)
    assert threshold is not None and threshold.threshold == 0.65
    reversal = next((t for t in p.triggers if isinstance(t, ConfidenceReversalTrigger)), None)
    assert reversal is not None and reversal.direction == "lower_is_better"


def test_from_overrides_ignores_unknown_keys():
    p = from_overrides({"some_future_field": 42})
    assert len(p.triggers) >= 3  # default set still present


# --- QualitativeProgressTrigger -----------------------------------------------

class _FakeLLMCall:
    """Helper for patching the cheap-LLM helper deterministically."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, *, model, api_key, guidance, history_summary,
                 allow_adapt=True):
        self.calls.append({
            "model": model, "guidance": guidance,
            "history_summary": history_summary,
            "allow_adapt": allow_adapt,
        })
        if not self.responses:
            return None
        return self.responses.pop(0)


def test_qualitative_first_eval_sets_clock_no_fire():
    """First call ever just records the timestamp; never fires immediately."""
    t = QualitativeProgressTrigger(
        guidance="watch noise floor", model="m", api_key="k",
        interval_sec=10.0,
    )
    h = [_tick(metric=0.5)]
    assert t.evaluate(h) is None


def test_qualitative_interval_gate_skips_calls(monkeypatch):
    """Within the interval, .evaluate() returns None without calling the LLM."""
    fake = _FakeLLMCall([{"fire": True, "reason": "noise", "severity": "low"}])
    monkeypatch.setattr(
        "scilink.agents.exp_agents.live_triggers._call_qualitative_check",
        fake,
    )
    t = QualitativeProgressTrigger(
        guidance="g", model="m", api_key="k", interval_sec=10.0,
    )
    base = time.time()
    h = [_tick(ts=base)]
    t.evaluate(h)  # first call sets the clock
    # Second call only 1s later — should not call the LLM
    h.append(_tick(ts=base + 1.0))
    assert t.evaluate(h) is None
    assert fake.calls == []


def test_qualitative_fires_when_interval_elapsed_and_decision_fire(monkeypatch):
    fake = _FakeLLMCall([{"fire": True, "reason": "noise creep", "severity": "medium"}])
    monkeypatch.setattr(
        "scilink.agents.exp_agents.live_triggers._call_qualitative_check",
        fake,
    )
    t = QualitativeProgressTrigger(
        guidance="watch noise", model="claude-haiku", api_key="k", interval_sec=10.0,
    )
    base = time.time()
    h = [_tick(ts=base)]
    t.evaluate(h)  # first call sets the clock
    h.append(_tick(ts=base + 11.0))  # interval elapsed
    ev = t.evaluate(h)
    assert ev is not None
    assert ev.name == "qualitative_progress"
    assert ev.details["reason"] == "noise creep"
    assert ev.details["severity"] == "medium"
    # Smoke check the LLM was called with the right inputs
    assert len(fake.calls) == 1
    assert fake.calls[0]["model"] == "claude-haiku"
    assert "watch noise" in fake.calls[0]["guidance"]


def test_qualitative_does_not_fire_when_decision_is_wait(monkeypatch):
    fake = _FakeLLMCall([{"fire": False, "reason": "", "severity": "low"}])
    monkeypatch.setattr(
        "scilink.agents.exp_agents.live_triggers._call_qualitative_check",
        fake,
    )
    t = QualitativeProgressTrigger(
        guidance="g", model="m", api_key="k", interval_sec=10.0,
    )
    base = time.time()
    h = [_tick(ts=base)]
    t.evaluate(h)
    h.append(_tick(ts=base + 11.0))
    assert t.evaluate(h) is None


def test_qualitative_dedupes_repeated_reason(monkeypatch):
    """Same fire-reason twice in a row → only first one fires (avoid spam)."""
    fake = _FakeLLMCall([
        {"fire": True, "reason": "noise creep", "severity": "low"},
        {"fire": True, "reason": "noise creep", "severity": "low"},  # same
        {"fire": True, "reason": "intensity drift", "severity": "medium"},  # different
    ])
    monkeypatch.setattr(
        "scilink.agents.exp_agents.live_triggers._call_qualitative_check",
        fake,
    )
    t = QualitativeProgressTrigger(
        guidance="g", model="m", api_key="k", interval_sec=10.0,
    )
    base = time.time()
    h = [_tick(ts=base)]
    t.evaluate(h)  # clock-set
    h.append(_tick(ts=base + 11.0))
    assert t.evaluate(h) is not None  # first fire
    h.append(_tick(ts=base + 22.0))
    assert t.evaluate(h) is None     # same reason — dedupe
    h.append(_tick(ts=base + 33.0))
    assert t.evaluate(h) is not None  # different reason — fire


def test_qualitative_llm_failure_does_not_fire(monkeypatch):
    """If the cheap-LLM call fails (returns None), the trigger does not fire."""
    fake = _FakeLLMCall([None, None])
    monkeypatch.setattr(
        "scilink.agents.exp_agents.live_triggers._call_qualitative_check",
        fake,
    )
    t = QualitativeProgressTrigger(
        guidance="g", model="m", api_key="k", interval_sec=10.0,
    )
    base = time.time()
    h = [_tick(ts=base)]
    t.evaluate(h)
    h.append(_tick(ts=base + 11.0))
    assert t.evaluate(h) is None


def test_qualitative_reset_re_arms():
    t = QualitativeProgressTrigger(
        guidance="g", model="m", api_key="k", interval_sec=10.0,
    )
    h = [_tick()]
    t.evaluate(h)
    assert t._last_checked_at is not None
    t.reset()
    assert t._last_checked_at is None
    assert t._last_fired_reason is None


def test_qualitative_satisfies_trigger_protocol():
    t = QualitativeProgressTrigger(
        guidance="g", model="m", api_key="k", interval_sec=10.0,
    )
    assert isinstance(t, Trigger)
    assert t.name == "qualitative_progress"
