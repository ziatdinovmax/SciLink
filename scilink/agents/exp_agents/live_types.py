"""Shared types for live-monitoring sessions.

Kept in a separate module so :mod:`live_triggers` and
:mod:`live_session` can both import the dataclasses without a circular
dependency. Nothing here imports SciLink agent / controller code, so
skill-side ``tick_fn`` modules can also import these types without
pulling the orchestrator's transitive heavyweight deps (threading,
LiteLLM wrappers, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# Coarse phase-ID verdict bucket. Skill tick functions return one of these so
# triggers can reason about transitions without parsing free text.
Verdict = Literal["accept", "marginal", "reject", "unknown"]


@dataclass
class LiveTickResult:
    """Output of a skill's ``tick_fn``; consumed by triggers + JSONL writer + UI.

    Universal fields cover what the trigger policy and UI need. Anything
    skill-specific lives in ``raw`` — for replay (so a follow-up
    analysis can reconstruct exactly what the tick saw) and for the
    LLM context dict (so the slow loop sees the full tick payload, not
    just the summarized verdict).
    """

    timestamp: float
    primary_metric: float
    metric_name: str
    verdict: Verdict
    detected_features: list[dict] = field(default_factory=list)
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerEvent:
    """One event emitted by a trigger when its condition fires.

    Triggers populate ``details`` with whatever per-trigger context is
    most useful to the LLM (the verdict transition, the new feature
    set, the rolling-window slope, etc.). The session bundles these
    into the ``context`` arg of ``orch.run_task``.
    """

    timestamp: float
    name: str
    details: dict[str, Any] = field(default_factory=dict)
    triggering_tick: Optional[LiveTickResult] = None


__all__ = ["Verdict", "LiveTickResult", "TriggerEvent"]
