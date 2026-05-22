"""LiveSession: drives the live-monitoring two-loop architecture.

A live session owns three threads:

  1. **Reading thread** — polls the data source every ``reading_interval_sec``,
     calls the skill's ``reading_fn``, appends the result to the rolling
     history, writes a JSONL line, and evaluates the trigger policy.
     Trigger events go onto an in-process queue.

  2. **LLM dispatch thread** — pulls events from the queue. When an
     event arrives, it drains any additional events that piled up
     during the wait and makes ONE call to
     :meth:`AnalysisOrchestratorAgent.run_task` (which mutates
     ``self.messages`` and is therefore not concurrent-safe — single-
     flight is required, not optional). The coalesced events form
     the context dict the LLM sees.

  3. **Main / UI thread** — reads ``latest()`` / ``history()`` /
     reads :func:`force_interpretation` calls and the
     ``on_llm_response`` callback. Does not block on either worker.

JSONL persistence is opt-in via ``history_path``. The schema is one
record per line, distinguished by a ``"kind"`` field: ``"reading"``,
``"trigger"``, ``"llm_response"``, or ``"session_end"``. Replay
(see :meth:`replay`) reads the same JSONL back reading-by-reading.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from .live_data_sources import LatestData, LiveDataSource
from .live_triggers import (
    ManualTrigger,
    TriggerEvent,
    TriggerPolicy,
    default_policy,
)
from .live_types import LiveReadingResult

_logger = logging.getLogger(__name__)


# A reading function receives the latest data, a mutable session-scoped state
# dict (whatever the function wants to carry across readings), and the active
# skill state dict (frontmatter + sections). It returns a LiveReadingResult.
ReadingFn = Callable[[LatestData, dict, dict], LiveReadingResult]


class LiveSession:
    """One running live-monitoring session.

    Construction is parameter-only; nothing starts until ``start()`` is
    called. ``stop()`` is idempotent and safe to call from any thread.
    """

    def __init__(
        self,
        orchestrator: Any,                       # AnalysisOrchestratorAgent
        data_source: LiveDataSource,
        reading_fn: ReadingFn,
        *,
        reading_interval_sec: float = 2.0,
        trigger_policy: Optional[TriggerPolicy] = None,
        history_path: Optional[Path] = None,
        history_maxlen: int = 2000,
        skill_state: Optional[dict] = None,
        on_llm_response: Optional[Callable[[TriggerEvent, dict], None]] = None,
        on_reading: Optional[Callable[[LiveReadingResult], None]] = None,
        # Adaptive-script support
        session_dir: Optional[Path] = None,
        adapt_model: Optional[str] = None,
        adapt_api_key: Optional[str] = None,
        adapt_description: Optional[str] = None,
        adapt_additional_guidance: Optional[str] = None,
        adapt_skill_context: Optional[str] = None,
    ) -> None:
        self.orch = orchestrator
        self.source = data_source
        self._reading_fn: ReadingFn = reading_fn
        self._reading_fn_lock = threading.Lock()
        self.reading_interval_sec = float(reading_interval_sec)
        self.policy = trigger_policy or default_policy()
        self.history_path = Path(history_path) if history_path else None
        self.skill_state = dict(skill_state) if skill_state else {}
        self.on_llm_response = on_llm_response
        self.on_reading = on_reading

        # Adaptive script regeneration: the supervisor can request a new
        # reading_fn mid-session. We need a place to save scripts and an
        # LLM + API key to call. When session_dir + adapt_model are
        # absent, adaptation is disabled (supervisor's "adapt" action
        # falls back to "flag").
        self.session_dir: Optional[Path] = Path(session_dir) if session_dir else None
        self._adapt_model = adapt_model
        self._adapt_api_key = adapt_api_key
        self._adapt_description = adapt_description
        self._adapt_additional_guidance = adapt_additional_guidance
        self._adapt_skill_context = adapt_skill_context
        self._adapt_version = 1   # initial script (if generated) is v1
        self._adapt_count = 0
        self._adapt_history: list[dict] = []  # one entry per regeneration

        self._history: deque[LiveReadingResult] = deque(maxlen=history_maxlen)
        self._history_lock = threading.Lock()
        self._jsonl_lock = threading.Lock()
        self._stop = threading.Event()
        self._reading_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._event_queue: "queue.Queue[TriggerEvent]" = queue.Queue()
        self._session_state: dict = {}
        self._started_at: Optional[float] = None
        self._llm_dispatcher_busy = threading.Event()

        # Cache the last K LatestData payloads for validating a
        # regenerated reading_fn before hot-swapping.
        self._validation_cache: deque[LatestData] = deque(maxlen=5)

        # Surface the ManualTrigger (if present) so force_interpretation()
        # can request it without searching the policy each call.
        self._manual: Optional[ManualTrigger] = next(
            (t for t in self.policy.triggers if isinstance(t, ManualTrigger)),
            None,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the reading and dispatch threads. Idempotent."""
        if self._reading_thread is not None and self._reading_thread.is_alive():
            return
        self._stop.clear()
        self._started_at = time.time()
        self._write_jsonl({"kind": "session_start", "timestamp": self._started_at})
        self._reading_thread = threading.Thread(
            target=self._reading_loop, name="LiveSession-reading", daemon=True,
        )
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="LiveSession-dispatch", daemon=True,
        )
        self._reading_thread.start()
        self._dispatch_thread.start()
        _logger.info("LiveSession started (interval=%.2fs)", self.reading_interval_sec)

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal both worker threads to exit and wait briefly for them.

        Safe to call from any thread, including the LLM-response callback.
        """
        if self._stop.is_set():
            return
        self._stop.set()
        # Best-effort join — daemon threads will die with the process anyway.
        for th in (self._reading_thread, self._dispatch_thread):
            if th is not None and th.is_alive():
                th.join(timeout=timeout)
        self._write_jsonl({"kind": "session_end", "timestamp": time.time()})
        _logger.info("LiveSession stopped")

    # ------------------------------------------------------------------
    # Public state inspection
    # ------------------------------------------------------------------

    def latest(self) -> Optional[LiveReadingResult]:
        with self._history_lock:
            return self._history[-1] if self._history else None

    def history(self, n: Optional[int] = None) -> list[LiveReadingResult]:
        with self._history_lock:
            items = list(self._history)
        return items if n is None else items[-n:]

    def force_interpretation(self) -> None:
        """Request a manual interpretation on the next reading.

        No-op when the active policy doesn't include a :class:`ManualTrigger`
        (operator disabled it).
        """
        if self._manual is not None:
            self._manual.request()

    @property
    def llm_busy(self) -> bool:
        """True iff a ``run_task`` is currently in flight (single-flight)."""
        return self._llm_dispatcher_busy.is_set()

    # ------------------------------------------------------------------
    # Worker bodies
    # ------------------------------------------------------------------

    def _reading_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._take_reading()
            except Exception:  # noqa: BLE001
                _logger.exception("reading loop iteration raised; continuing")
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self.reading_interval_sec - elapsed)
            # Use Event.wait so stop() can wake us early.
            self._stop.wait(timeout=sleep_for)

    @property
    def reading_fn(self) -> ReadingFn:
        """Currently-active reading function. Lock-protected (atomically
        swapped by the adaptation path)."""
        with self._reading_fn_lock:
            return self._reading_fn

    def _take_reading(self) -> None:
        data = self.source.read_latest()
        if data is None:
            return
        # Cache for validation of any future regenerated reading_fn
        self._validation_cache.append(data)
        with self._reading_fn_lock:
            active_fn = self._reading_fn
        try:
            result = active_fn(data, self._session_state, self.skill_state)
        except Exception:  # noqa: BLE001
            _logger.exception("reading_fn raised; skipping this reading")
            return
        if not isinstance(result, LiveReadingResult):
            _logger.warning(
                "reading_fn returned %r, expected LiveReadingResult — skipping",
                type(result).__name__,
            )
            return
        with self._history_lock:
            self._history.append(result)
            history_snapshot = list(self._history)

        self._write_jsonl({
            "kind": "reading",
            "timestamp": result.timestamp,
            "metric": result.primary_metric,
            "metric_name": result.metric_name,
            "verdict": result.verdict,
            "detected_features": result.detected_features,
            "notes": result.notes,
            "raw": _serializable(result.raw),
        })

        if self.on_reading is not None:
            try:
                self.on_reading(result)
            except Exception:  # noqa: BLE001
                _logger.exception("on_reading callback raised; continuing")

        events = self.policy.evaluate(history_snapshot)
        for ev in events:
            self._write_jsonl({
                "kind": "trigger",
                "timestamp": ev.timestamp,
                "name": ev.name,
                "details": _serializable(ev.details),
            })
            self._event_queue.put(ev)

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # Drain anything else that piled up during the .get() wait.
            coalesced = [first]
            while True:
                try:
                    coalesced.append(self._event_queue.get_nowait())
                except queue.Empty:
                    break
            # Route adapt events to script regeneration; everything else
            # to the regular full-model interpretation path. If a batch
            # contains BOTH, do the adapt first (it changes the active
            # reading_fn) and then run interpretation with the surviving
            # non-adapt events.
            adapt_events = [
                e for e in coalesced
                if e.name == "qualitative_progress"
                and (e.details or {}).get("action") == "adapt"
            ]
            other_events = [e for e in coalesced if e not in adapt_events]
            try:
                self._llm_dispatcher_busy.set()
                for ev in adapt_events:
                    self._adapt_reading_script(ev)
                if other_events:
                    self._invoke_llm(other_events)
            finally:
                self._llm_dispatcher_busy.clear()

    def _invoke_llm(self, events: list[TriggerEvent]) -> None:
        latest_event = events[-1]
        task = self._build_task(latest_event, events)
        context = self._build_context(latest_event, events)
        try:
            result = self.orch.run_task(task=task, context=context)
        except Exception as e:  # noqa: BLE001
            _logger.exception("run_task failed: %s", e)
            self._write_jsonl({
                "kind": "llm_response",
                "timestamp": time.time(),
                "trigger": latest_event.name,
                "status": "error",
                "error": str(e),
                "coalesced_events": [e.name for e in events],
            })
            return
        if not isinstance(result, dict):
            result = {"status": "unknown", "summary": str(result)}
        self._write_jsonl({
            "kind": "llm_response",
            "timestamp": time.time(),
            "trigger": latest_event.name,
            "status": result.get("status", "unknown"),
            "summary": result.get("summary", ""),
            "key_findings": result.get("key_findings", []),
            "files_produced": result.get("files_produced", []),
            "warnings": result.get("warnings", []),
            "coalesced_events": [e.name for e in events],
        })
        if self.on_llm_response is not None:
            try:
                self.on_llm_response(latest_event, result)
            except Exception:  # noqa: BLE001
                _logger.exception("on_llm_response callback raised")

    # ------------------------------------------------------------------
    # Adaptive-script regeneration
    # ------------------------------------------------------------------

    def _adapt_reading_script(self, event: TriggerEvent) -> None:
        """Regenerate the per-reading analysis script in response to a
        supervisor "adapt" event. Validates against the cache; only
        hot-swaps if validation passes. All outcomes are JSONL-logged
        and surfaced to the operator via the on_llm_response callback.
        """
        if not self.session_dir or not self._adapt_model or not self._adapt_api_key:
            # Configured-off: log + fall through (no hot-swap, no failure)
            self._write_jsonl({
                "kind": "script_adaptation_skipped",
                "timestamp": time.time(),
                "reason": "adaptive-script regeneration not configured "
                          "(no session_dir / adapt_model / adapt_api_key)",
                "trigger_reason": (event.details or {}).get("reason", ""),
            })
            return
        from .live_codegen import generate_and_validate_with_retry

        deviation_reason = (event.details or {}).get("reason", "")
        recent = self.history(n=10)
        prior_summary = (
            f"Most recent script was activated at version v{self._adapt_version}. "
            f"Supervisor flagged: {deviation_reason}\n\nRecent readings:\n"
        )
        for r in recent[-6:]:
            prior_summary += (
                f"  t={r.timestamp:.1f} {r.metric_name}={r.primary_metric:.3f} "
                f"verdict={r.verdict} features={len(r.detected_features)} "
                f"notes={r.notes[:80]}\n"
            )

        # Use the same generate + validate + retry loop the session-start
        # path uses. Validation samples come from the rolling cache of
        # actually-observed LatestData payloads — guaranteed-real input
        # the new script must handle.
        new_fn, new_version, attempts = generate_and_validate_with_retry(
            description=self._adapt_description or "(no description)",
            additional_guidance=self._adapt_additional_guidance,
            skill_context=self._adapt_skill_context,
            model=self._adapt_model,
            api_key=self._adapt_api_key,
            session_dir=self.session_dir,
            reference_data=list(self._validation_cache),
            skill_state=self.skill_state,
            max_attempts=3,
            start_version=self._adapt_version + 1,
            initial_prior_context=prior_summary,
        )
        if new_fn is None:
            self._write_jsonl({
                "kind": "script_adaptation_failed",
                "timestamp": time.time(),
                "reason": "all 3 regeneration attempts failed",
                "trigger_reason": deviation_reason,
                "attempts": attempts,
            })
            return
        # Hot-swap
        with self._reading_fn_lock:
            self._reading_fn = new_fn
        self._adapt_version = new_version
        self._adapt_count += 1
        reason = (
            f"validated against {len(self._validation_cache)} cached readings"
            + (f" after {len(attempts)} prior attempt(s)" if attempts else "")
        )
        self._adapt_history.append({
            "version": new_version,
            "timestamp": time.time(),
            "trigger_reason": deviation_reason,
            "validation_note": reason,
            "script_path": str(self.session_dir / f"reading_script_v{new_version}.py"),
        })
        self._write_jsonl({
            "kind": "script_adapted",
            "timestamp": time.time(),
            "version": new_version,
            "trigger_reason": deviation_reason,
            "validation_note": reason,
            "script_path": str(self.session_dir / f"reading_script_v{new_version}.py"),
        })
        # Surface to the UI via on_llm_response as a synthetic "adaptation" event
        if self.on_llm_response is not None:
            try:
                self.on_llm_response(event, {
                    "status": "success",
                    "summary": (
                        f"Analysis script adapted to v{new_version} in "
                        f"response to: {deviation_reason}. {reason}"
                    ),
                    "key_findings": [],
                    "files_produced": [str(self.session_dir /
                                            f"reading_script_v{new_version}.py")],
                    "warnings": [],
                })
            except Exception:  # noqa: BLE001
                _logger.exception("on_llm_response callback raised on adaptation")

    # ------------------------------------------------------------------
    # Task / context shaping
    # ------------------------------------------------------------------

    def _build_task(self, latest_event: TriggerEvent, events: list[TriggerEvent]) -> str:
        if len(events) == 1:
            return (
                f"The live monitoring session detected an event of kind "
                f"'{latest_event.name}'. Please interpret the latest scan "
                f"and produce a concise update for the decision feed."
            )
        names = ", ".join(e.name for e in events)
        return (
            f"The live monitoring session detected multiple coincident events "
            f"({names}). Please interpret the latest scan in light of these "
            f"and produce a concise update for the decision feed."
        )

    def _build_context(self, latest_event: TriggerEvent,
                        events: list[TriggerEvent]) -> dict:
        recent = self.history(n=10)
        return {
            "trigger": {
                "name": latest_event.name,
                "details": _serializable(latest_event.details),
                "timestamp": latest_event.timestamp,
            },
            "coalesced_event_names": [e.name for e in events],
            "latest_metric": (
                {
                    "name": recent[-1].metric_name,
                    "value": recent[-1].primary_metric,
                    "verdict": recent[-1].verdict,
                }
                if recent else None
            ),
            "recent_ticks": [
                {
                    "timestamp": t.timestamp,
                    "metric": t.primary_metric,
                    "metric_name": t.metric_name,
                    "verdict": t.verdict,
                    "n_features": len(t.detected_features),
                    "notes": t.notes,
                }
                for t in recent
            ],
        }

    # ------------------------------------------------------------------
    # JSONL writer
    # ------------------------------------------------------------------

    def _write_jsonl(self, entry: dict) -> None:
        if self.history_path is None:
            return
        with self._jsonl_lock:
            try:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                with self.history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=_jsonable) + "\n")
            except OSError as e:
                _logger.warning("Failed to append to %s: %s", self.history_path, e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """JSON default-handler for things like Path or numpy scalars."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    return str(obj)


def _serializable(obj: Any) -> Any:
    """Best-effort conversion of a nested dict/list to JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "__dict__"):
        return _serializable(asdict(obj)) if hasattr(obj, "__dataclass_fields__") else str(obj)
    return str(obj)


__all__ = ["LiveSession", "ReadingFn"]
