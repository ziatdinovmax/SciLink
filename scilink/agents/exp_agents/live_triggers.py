"""Trigger taxonomy for live-monitoring sessions.

Each trigger inspects the rolling ``LiveReadingResult`` history each reading
and emits a :class:`TriggerEvent` (or ``None``) when its condition is
met. :class:`TriggerPolicy` composes a set of triggers; one reading can
fire several events from different triggers.

The taxonomy is **what counts as "interesting"** — when the LLM
"slow loop" should be invoked to produce a textual interpretation.
The framework reads only the structured ``LiveReadingResult`` fields, so
new trigger types plug in without touching skills or the session.

Built-in triggers (v1):

  - :class:`VerdictChangeTrigger` — reject ↔ marginal ↔ accept transitions
  - :class:`NewFeatureTrigger` — feature appears that wasn't seen in
    the previous N readings
  - :class:`ConfidenceReversalTrigger` — metric was monotonically
    improving for K readings; now reverses
  - :class:`ThresholdCrossTrigger` — metric crosses a configured
    boundary in either direction
  - :class:`HeartbeatTrigger` — periodic narrative even when nothing
    changed (keeps the decision feed populated during a quiet scan)
  - :class:`ManualTrigger` — user clicked "interpret now"

Skill markdown can supply a ``trigger_overrides:`` block in frontmatter
to override the defaults (e.g. a Raman skill can set
``confidence_threshold: 0.65`` because Raman peak matches typically
score lower than XRD).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .live_types import LiveReadingResult, TriggerEvent, Verdict


# ---------------------------------------------------------------------------
# Trigger protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Trigger(Protocol):
    """A trigger evaluates the rolling history each reading."""

    name: str

    def evaluate(
        self, history: list[LiveReadingResult]
    ) -> Optional[TriggerEvent]: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Built-in triggers
# ---------------------------------------------------------------------------


class VerdictChangeTrigger:
    """Fires when the latest reading's verdict differs from the previous.

    The most generally-useful trigger for phase-ID workflows. Both
    "marginal → accept" and "accept → marginal" should prompt the LLM
    to comment, since each carries different scientific weight.
    """

    name = "verdict_change"

    def __init__(self) -> None:
        self._last_seen: Optional[Verdict] = None

    def evaluate(self, history: list[LiveReadingResult]) -> Optional[TriggerEvent]:
        if not history:
            return None
        latest = history[-1]
        if self._last_seen is None:
            # First observation: not a "change" — record and stay quiet.
            self._last_seen = latest.verdict
            return None
        if latest.verdict == self._last_seen:
            return None
        previous = self._last_seen
        self._last_seen = latest.verdict
        return TriggerEvent(
            timestamp=latest.timestamp,
            name=self.name,
            details={"from": previous, "to": latest.verdict},
            triggering_reading=latest,
        )

    def reset(self) -> None:
        self._last_seen = None


class NewFeatureTrigger:
    """Fires when an entry in ``detected_features`` is new vs. recent readings.

    Skill-specific: each modality decides what "a feature" means
    (XRD = peak position; Raman = peak position + assignment; STM
    = lattice symmetry, etc.). The trigger compares the latest reading's
    feature set to the union of the previous ``lookback`` readings. New
    elements in the set fire the trigger.

    Feature equality is via a configurable key function (default: the
    feature dict itself). For XRD a peak's 2θ position rounded to
    0.1° is the natural identity key — round noise out, catch actual
    new reflections.
    """

    name = "new_feature"

    def __init__(self, lookback: int = 5, key: Optional[callable] = None) -> None:  # type: ignore[valid-type]
        self.lookback = lookback
        self.key = key or (lambda f: tuple(sorted(f.items())))

    def evaluate(self, history: list[LiveReadingResult]) -> Optional[TriggerEvent]:
        if len(history) < 2:
            return None
        latest = history[-1]
        baseline_window = history[-1 - self.lookback : -1]
        if not baseline_window:
            return None
        baseline_keys = set()
        for past in baseline_window:
            for f in past.detected_features:
                try:
                    baseline_keys.add(self.key(f))
                except Exception:
                    continue
        new_features = []
        for f in latest.detected_features:
            try:
                k = self.key(f)
            except Exception:
                continue
            if k not in baseline_keys:
                new_features.append(f)
        if not new_features:
            return None
        return TriggerEvent(
            timestamp=latest.timestamp,
            name=self.name,
            details={
                "new_features": new_features,
                "baseline_lookback": len(baseline_window),
            },
            triggering_reading=latest,
        )

    def reset(self) -> None:
        pass


class ConfidenceReversalTrigger:
    """Fires when an improving trend reverses.

    Looks at the last ``window`` readings; if ``primary_metric`` was
    monotonically improving for at least ``window - 1`` of them and
    the latest reading reverses by more than ``min_reversal``, fires.

    "Improving" depends on the metric direction. Most scoring metrics
    (correlation, FOM) are higher-is-better; cost-style metrics
    (MIP cost, RMSE) are lower-is-better. ``direction`` defaults to
    higher-is-better; flip to ``"lower"`` for cost-style metrics.
    """

    name = "confidence_reversal"

    def __init__(
        self,
        window: int = 5,
        min_reversal: float = 0.05,
        direction: str = "higher_is_better",
    ) -> None:
        self.window = window
        self.min_reversal = float(min_reversal)
        self.direction = direction
        if direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError(
                f"direction must be 'higher_is_better' or 'lower_is_better'; "
                f"got {direction!r}"
            )

    def evaluate(self, history: list[LiveReadingResult]) -> Optional[TriggerEvent]:
        if len(history) < self.window + 1:
            return None
        recent = history[-self.window - 1 :]
        # Walk: did we improve for the first window steps then reverse on the last?
        metrics = [t.primary_metric for t in recent]
        if self.direction == "higher_is_better":
            improved_run = all(
                metrics[i] < metrics[i + 1]
                for i in range(self.window - 1)
            )
            reversal = metrics[-2] - metrics[-1]
        else:
            improved_run = all(
                metrics[i] > metrics[i + 1]
                for i in range(self.window - 1)
            )
            reversal = metrics[-1] - metrics[-2]
        if not improved_run:
            return None
        if reversal < self.min_reversal:
            return None
        return TriggerEvent(
            timestamp=recent[-1].timestamp,
            name=self.name,
            details={
                "metric_name": recent[-1].metric_name,
                "window": self.window,
                "trend_before": metrics[:-1],
                "current": metrics[-1],
                "reversal_magnitude": float(reversal),
                "direction": self.direction,
            },
            triggering_reading=recent[-1],
        )

    def reset(self) -> None:
        pass


class ThresholdCrossTrigger:
    """Fires when ``primary_metric`` crosses a configured boundary.

    Useful for "alert me when confidence first hits 0.7" or "warn me if
    it drops below 0.4." Fires once per crossing direction — won't
    spam if the metric hovers around the threshold.
    """

    def __init__(
        self,
        threshold: float,
        direction: str = "above",
        name: Optional[str] = None,
    ) -> None:
        if direction not in ("above", "below"):
            raise ValueError(f"direction must be 'above' or 'below'; got {direction!r}")
        self.threshold = float(threshold)
        self.direction = direction
        self.name = name or f"threshold_{direction}_{threshold:.2f}"
        self._last_relation: Optional[str] = None  # "above" or "below"

    def evaluate(self, history: list[LiveReadingResult]) -> Optional[TriggerEvent]:
        if not history:
            return None
        latest = history[-1]
        relation = "above" if latest.primary_metric > self.threshold else "below"
        if self._last_relation is None:
            self._last_relation = relation
            return None
        if relation == self._last_relation:
            return None
        previous = self._last_relation
        self._last_relation = relation
        # Fire only when the current relation matches our configured direction
        # (e.g. ThresholdCrossTrigger(0.7, "above") fires on below→above only).
        if relation != self.direction:
            return None
        return TriggerEvent(
            timestamp=latest.timestamp,
            name=self.name,
            details={
                "threshold": self.threshold,
                "from": previous,
                "to": relation,
                "metric_name": latest.metric_name,
                "metric": latest.primary_metric,
            },
            triggering_reading=latest,
        )

    def reset(self) -> None:
        self._last_relation = None


class HeartbeatTrigger:
    """Periodic narrative; fires when ``interval_sec`` has elapsed since the
    last LLM event of any kind. Keeps the decision feed populated during
    long quiet stretches.
    """

    name = "heartbeat"

    def __init__(self, interval_sec: float = 60.0) -> None:
        self.interval_sec = float(interval_sec)
        self._last_fired_at: Optional[float] = None

    def evaluate(self, history: list[LiveReadingResult]) -> Optional[TriggerEvent]:
        if not history:
            return None
        latest = history[-1]
        if self._last_fired_at is None:
            # Don't fire on session start — let other triggers go first.
            self._last_fired_at = latest.timestamp
            return None
        if (latest.timestamp - self._last_fired_at) < self.interval_sec:
            return None
        self._last_fired_at = latest.timestamp
        return TriggerEvent(
            timestamp=latest.timestamp,
            name=self.name,
            details={"interval_sec": self.interval_sec},
            triggering_reading=latest,
        )

    def reset(self) -> None:
        self._last_fired_at = None


@dataclass
class ManualTrigger:
    """User-initiated trigger. The UI / API sets ``pending=True``; on the
    next ``evaluate()`` the trigger fires once and clears the flag.
    """

    name: str = "manual"
    pending: bool = False
    _last_request_ts: Optional[float] = None

    def request(self) -> None:
        """Mark a manual interpretation as requested."""
        self.pending = True
        self._last_request_ts = time.time()

    def evaluate(self, history: list[LiveReadingResult]) -> Optional[TriggerEvent]:
        if not self.pending or not history:
            return None
        latest = history[-1]
        self.pending = False
        return TriggerEvent(
            timestamp=self._last_request_ts or latest.timestamp,
            name=self.name,
            details={"reason": "user_requested_interpretation"},
            triggering_reading=latest,
        )

    def reset(self) -> None:
        self.pending = False
        self._last_request_ts = None


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class TriggerPolicy:
    """A composable list of triggers.

    Evaluating the policy returns the list of events that fired this
    reading. The session writes one JSONL line per event and enqueues
    each for the LLM dispatcher. The dispatcher's single-flight
    semantics decides whether a fresh ``run_task`` runs immediately
    or coalesces into the in-flight call.

    Skill frontmatter ``trigger_overrides:`` is applied at construction
    time via :func:`from_overrides`.
    """

    triggers: list[Trigger] = field(default_factory=list)

    def evaluate(self, history: list[LiveReadingResult]) -> list[TriggerEvent]:
        events: list[TriggerEvent] = []
        for t in self.triggers:
            try:
                ev = t.evaluate(history)
            except Exception:  # noqa: BLE001 — trigger bug shouldn't crash the reading loop
                continue
            if ev is not None:
                events.append(ev)
        return events

    def reset(self) -> None:
        for t in self.triggers:
            t.reset()


def default_policy(
    *,
    enable_heartbeat: bool = True,
    heartbeat_sec: float = 60.0,
    threshold: Optional[float] = None,
    threshold_direction: str = "above",
    new_feature_lookback: int = 5,
    reversal_window: int = 5,
    reversal_direction: str = "higher_is_better",
    include_manual: bool = True,
) -> TriggerPolicy:
    """Construct the default trigger set used when a session doesn't pass one explicitly."""
    triggers: list[Trigger] = [
        VerdictChangeTrigger(),
        NewFeatureTrigger(lookback=new_feature_lookback),
        ConfidenceReversalTrigger(window=reversal_window, direction=reversal_direction),
    ]
    if threshold is not None:
        triggers.append(ThresholdCrossTrigger(threshold=threshold, direction=threshold_direction))
    if enable_heartbeat:
        triggers.append(HeartbeatTrigger(interval_sec=heartbeat_sec))
    if include_manual:
        triggers.append(ManualTrigger())
    return TriggerPolicy(triggers=triggers)


def from_overrides(overrides: Optional[dict]) -> TriggerPolicy:
    """Build a policy from a skill frontmatter ``trigger_overrides:`` block.

    Supported keys (all optional):
      - ``heartbeat_sec``: float — disables heartbeat when 0 / None
      - ``confidence_threshold``: float — adds a ThresholdCrossTrigger(above)
      - ``new_feature_lookback``: int
      - ``reversal_window``: int
      - ``reversal_direction``: 'higher_is_better' | 'lower_is_better'

    Unknown keys are ignored (forwards-compat for skill authors using
    fields we haven't added yet).
    """
    if not overrides:
        return default_policy()
    return default_policy(
        enable_heartbeat=bool(overrides.get("heartbeat_sec", 60.0)),
        heartbeat_sec=float(overrides.get("heartbeat_sec", 60.0)),
        threshold=overrides.get("confidence_threshold"),
        threshold_direction=overrides.get("confidence_threshold_direction", "above"),
        new_feature_lookback=int(overrides.get("new_feature_lookback", 5)),
        reversal_window=int(overrides.get("reversal_window", 5)),
        reversal_direction=overrides.get("reversal_direction", "higher_is_better"),
        include_manual=bool(overrides.get("include_manual", True)),
    )


__all__ = [
    "Trigger",
    "TriggerEvent",
    "TriggerPolicy",
    "VerdictChangeTrigger",
    "NewFeatureTrigger",
    "ConfidenceReversalTrigger",
    "ThresholdCrossTrigger",
    "HeartbeatTrigger",
    "ManualTrigger",
    "default_policy",
    "from_overrides",
]
