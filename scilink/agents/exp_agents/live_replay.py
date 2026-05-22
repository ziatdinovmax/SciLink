"""Replay a recorded live-monitoring session from its JSONL stream.

Use cases:

  - **Trigger tuning.** Replay an experiment under a different
    :class:`TriggerPolicy` to see what would have fired. No instrument
    needed; iterate on thresholds against the recorded data.

  - **Debugging.** A live LLM call produced a wrong interpretation —
    replay with ``llm_mode="redo"`` to reproduce the call with the
    historical context (possibly against a different model / prompt).

  - **Post-hoc audit.** Walk the timeline of detected features and
    decisions a session produced; export to a report.

The replay function reads only ``kind == "tick"`` lines (the data
record). Trigger events and LLM responses from the original run are
recomputed when ``llm_mode="redo"`` and echoed when
``llm_mode="skip"`` (default). The original session's
``session_start`` / ``session_end`` markers are passed through but
otherwise ignored.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .live_triggers import TriggerEvent, TriggerPolicy, default_policy
from .live_types import LiveTickResult

_logger = logging.getLogger(__name__)


@dataclass
class ReplayReport:
    """Structured summary of a replay run."""

    tick_count: int = 0
    trigger_count: int = 0
    trigger_event_counts: dict[str, int] = field(default_factory=dict)
    trigger_events: list[TriggerEvent] = field(default_factory=list)
    llm_responses: list[dict] = field(default_factory=list)
    duration_sec: float = 0.0


def _iter_tick_lines(path: Path) -> Iterator[dict]:
    """Yield only the ``kind == "tick"`` entries from a session JSONL."""
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as e:
                _logger.debug("Skipping malformed JSONL line: %s", e)
                continue
            if entry.get("kind") == "tick":
                yield entry


def _iter_llm_lines(path: Path) -> Iterator[dict]:
    """Yield only the ``kind == "llm_response"`` entries from a session JSONL."""
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if entry.get("kind") == "llm_response":
                yield entry


def _entry_to_tick_result(entry: dict) -> LiveTickResult:
    return LiveTickResult(
        timestamp=float(entry["timestamp"]),
        primary_metric=float(entry.get("metric", entry.get("primary_metric", 0.0))),
        metric_name=str(entry.get("metric_name", "metric")),
        verdict=str(entry.get("verdict", "unknown")),
        detected_features=list(entry.get("detected_features", [])),
        notes=str(entry.get("notes", "")),
        raw=dict(entry.get("raw", {})),
    )


def replay_jsonl(
    path: str | Path,
    *,
    trigger_policy: Optional[TriggerPolicy] = None,
    speed: Optional[float] = None,
    on_tick: Optional[Callable[[LiveTickResult], None]] = None,
    on_event: Optional[Callable[[TriggerEvent], None]] = None,
    orchestrator: Any = None,
    llm_mode: str = "skip",
) -> ReplayReport:
    """Replay a recorded session's JSONL stream.

    Args:
        path: ``live_ticks.jsonl`` path from a previous session.
        trigger_policy: Policy to evaluate against the replayed ticks.
            Defaults to :func:`default_policy` — same as a live session.
            Pass a tweaked policy to test thresholds against the same
            data without re-running the experiment.
        speed: Playback speed multiplier. ``None`` (default) replays
            as fast as possible (no sleeping). ``1.0`` matches real
            time. ``10.0`` plays 10× faster. Useful for debugging:
            ``1.0`` recreates the live cadence so you can watch the
            decision feed unfold.
        on_tick: Optional callback invoked with each rebuilt
            ``LiveTickResult`` (in chronological order). Useful for
            driving a UI replay or extracting per-tick metrics.
        on_event: Optional callback invoked with each fired
            ``TriggerEvent``. Useful for inspecting which triggers
            fired during the replay.
        orchestrator: Required when ``llm_mode="redo"`` — its
            ``run_task`` is invoked with the historical context.
        llm_mode: ``"skip"`` (default) echoes the original
            ``llm_response`` lines into the report without calling the
            LLM. ``"redo"`` calls ``orchestrator.run_task`` for each
            trigger that fires.

    Returns:
        :class:`ReplayReport` summarizing tick / trigger / LLM counts
        and the structured event timeline.
    """
    if llm_mode not in ("skip", "redo"):
        raise ValueError(f"llm_mode must be 'skip' or 'redo'; got {llm_mode!r}")
    if llm_mode == "redo" and orchestrator is None:
        raise ValueError("llm_mode='redo' requires an orchestrator")

    path = Path(path)
    policy = trigger_policy if trigger_policy is not None else default_policy()
    report = ReplayReport()

    if llm_mode == "skip":
        # Pre-load the original LLM responses so we can echo them into the report
        original_llm = list(_iter_llm_lines(path))
    else:
        original_llm = []

    rolling: list[LiveTickResult] = []
    last_ts: Optional[float] = None
    started_at = time.monotonic()

    for entry in _iter_tick_lines(path):
        result = _entry_to_tick_result(entry)
        # Cadence: sleep proportional to inter-tick delta if speed is set
        if speed is not None and last_ts is not None and result.timestamp > last_ts:
            delta = result.timestamp - last_ts
            sleep_for = delta / max(float(speed), 1e-9)
            if sleep_for > 0:
                time.sleep(min(sleep_for, 60.0))  # cap at 60s per gap
        last_ts = result.timestamp

        rolling.append(result)
        report.tick_count += 1
        if on_tick is not None:
            try:
                on_tick(result)
            except Exception:  # noqa: BLE001
                _logger.exception("on_tick callback raised")

        events = policy.evaluate(rolling)
        for ev in events:
            report.trigger_count += 1
            report.trigger_event_counts[ev.name] = \
                report.trigger_event_counts.get(ev.name, 0) + 1
            report.trigger_events.append(ev)
            if on_event is not None:
                try:
                    on_event(ev)
                except Exception:  # noqa: BLE001
                    _logger.exception("on_event callback raised")
            if llm_mode == "redo" and orchestrator is not None:
                context = {
                    "trigger": {
                        "name": ev.name,
                        "details": ev.details,
                        "timestamp": ev.timestamp,
                    },
                    "replay": True,
                    "recent_ticks": [
                        {
                            "timestamp": t.timestamp,
                            "metric": t.primary_metric,
                            "metric_name": t.metric_name,
                            "verdict": t.verdict,
                        }
                        for t in rolling[-10:]
                    ],
                }
                task = (
                    f"[Replay] The live monitoring session detected an event "
                    f"of kind '{ev.name}'. Please interpret the latest scan "
                    f"and produce an update for the decision feed."
                )
                try:
                    result_dict = orchestrator.run_task(task=task, context=context)
                except Exception as e:  # noqa: BLE001
                    _logger.exception("run_task failed during replay: %s", e)
                    result_dict = {"status": "error", "error": str(e)}
                report.llm_responses.append({
                    "trigger": ev.name,
                    "timestamp": time.time(),
                    "result": result_dict,
                })

    if llm_mode == "skip":
        report.llm_responses = original_llm

    report.duration_sec = time.monotonic() - started_at
    return report


__all__ = ["ReplayReport", "replay_jsonl"]
