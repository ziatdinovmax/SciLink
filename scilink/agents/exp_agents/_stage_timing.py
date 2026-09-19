"""Per-stage wall-clock and LLM-cost accounting for the analysis pipelines.

The analysis agents run a fixed sequence of controllers, and until now nothing
recorded how long each one took: logs carry iteration counts, the opt-in tracer
carries per-call latency, and neither says "planning took 40 s, the QC loop
took 9 min, 85% of it waiting on the model". Depth presets (which stages a
quick run may skip) cannot be tuned without those numbers.

``StageTimer`` wraps each ``controller.execute(state)`` and records, per stage,
the wall-clock seconds plus the delta of the process-wide LLM counters
(``scilink.tracing.llm_counters``): calls, seconds spent inside completions,
and tokens. ``summary()`` is what the agents persist as ``stage_timings`` in
``analysis_results.json``.

Accounting only — a timer never changes what a stage does, and a failure
inside the timer never fails a stage.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

_active_lock = threading.Lock()
_active_timers = 0
# Bumped every time a stage starts while another is already running. A stage
# compares the value at its start and end, so it notices a neighbour that
# started AND finished inside it (the active count alone would miss that).
_overlaps = 0


def _counters() -> Dict[str, float]:
    try:
        from ... import tracing
        return tracing.llm_counters()
    except Exception:  # noqa: BLE001 - accounting must never break a run
        return {}


class StageTimer:
    """Collects one record per executed pipeline stage."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str, **extra: Any) -> Iterator[None]:
        """Time one stage. Records even when the stage raises."""
        global _active_timers, _overlaps
        with _active_lock:
            _active_timers += 1
            concurrent = _active_timers > 1
            if concurrent:
                _overlaps += 1
            overlaps_at_start = _overlaps
        before = _counters()
        t0 = time.perf_counter()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            seconds = time.perf_counter() - t0
            after = _counters()
            with _active_lock:
                concurrent = concurrent or _overlaps != overlaps_at_start
                _active_timers -= 1
            rec: Dict[str, Any] = {
                "stage": name,
                "seconds": round(seconds, 3),
                "llm_calls": int(after.get("calls", 0) - before.get("calls", 0)),
                "llm_seconds": round(
                    after.get("seconds", 0.0) - before.get("seconds", 0.0), 3),
                "prompt_tokens": int(after.get("prompt_tokens", 0)
                                     - before.get("prompt_tokens", 0)),
                "completion_tokens": int(after.get("completion_tokens", 0)
                                         - before.get("completion_tokens", 0)),
                "status": status,
            }
            if concurrent:
                # Another pipeline was being timed in this process at the same
                # time (a meta fan-out): the LLM counters are process-wide, so
                # this stage's llm_* figures include the neighbour's calls.
                rec["llm_counts_shared"] = True
            rec.update(extra)
            self.records.append(rec)

    def run(self, controller: Any, state: dict, *,
            name: Optional[str] = None, **extra: Any) -> dict:
        """``controller.execute(state)`` under a stage named after its class."""
        with self.stage(name or controller.__class__.__name__, **extra):
            return controller.execute(state)

    def summary(self) -> Dict[str, Any]:
        """The persisted shape: per-stage records, per-name totals, run totals.

        A stage's ``llm_seconds`` can exceed its ``seconds`` when it fans LLM
        calls out to worker threads (the counter sums latencies), so
        ``llm_fraction`` is capped at 1 and is a lower bound on how much of the
        wall-clock was model time only for serial stages.
        """
        by_stage: Dict[str, Dict[str, Any]] = {}
        for r in self.records:
            agg = by_stage.setdefault(r["stage"], {
                "runs": 0, "seconds": 0.0, "llm_calls": 0, "llm_seconds": 0.0})
            agg["runs"] += 1
            agg["seconds"] = round(agg["seconds"] + r["seconds"], 3)
            agg["llm_calls"] += r["llm_calls"]
            agg["llm_seconds"] = round(agg["llm_seconds"] + r["llm_seconds"], 3)
        staged = sum(r["seconds"] for r in self.records)
        llm_s = sum(r["llm_seconds"] for r in self.records)
        return {
            "stages": list(self.records),
            "by_stage": by_stage,
            "total_seconds": round(time.perf_counter() - self._t0, 3),
            "staged_seconds": round(staged, 3),
            "llm_calls": sum(r["llm_calls"] for r in self.records),
            "llm_seconds": round(llm_s, 3),
            "llm_fraction": (round(min(1.0, llm_s / staged), 3)
                             if staged > 0 else None),
            "prompt_tokens": sum(r["prompt_tokens"] for r in self.records),
            "completion_tokens": sum(r["completion_tokens"] for r in self.records),
        }

    def log_summary(self, logger: Any, top: int = 6) -> None:
        """One compact block in the run log: where the time went."""
        try:
            s = self.summary()
            if not s["stages"]:
                return
            logger.info(
                f"⏱️  Stage timing: {s['staged_seconds']:.1f}s staged, "
                f"{s['llm_calls']} LLM call(s), {s['llm_seconds']:.1f}s in the "
                f"model ({(s['llm_fraction'] or 0) * 100:.0f}%)")
            ranked = sorted(s["by_stage"].items(),
                            key=lambda kv: -kv[1]["seconds"])[:top]
            for name, agg in ranked:
                logger.info(
                    f"      {agg['seconds']:8.1f}s  {agg['llm_calls']:3d} call(s)  "
                    f"{name}" + (f" ×{agg['runs']}" if agg["runs"] > 1 else ""))
        except Exception:  # noqa: BLE001 - accounting must never break a run
            pass
