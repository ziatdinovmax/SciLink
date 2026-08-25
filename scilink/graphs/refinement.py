"""
scilink.graphs.refinement
==========================

Reusable human-feedback refinement subgraph — 1:1 replacement for the
imperative ``while iteration < self.max_iterations`` accept/refine loops
duplicated (independently of the quality-verification loop already migrated
in ``graphs/verification.py``) across:

* ``image_analysis_controllers.py:PlanningStep.execute``
* ``curve_fitting_controllers.py:PlanningStep.execute``
* ``fft_microscopy_controllers.py`` (parameter refinement)
* ``sam_microscopy_controllers.py`` (parameter refinement, both controllers)

Unlike the verification subgraph, this loop has no quality score and no
annealing: show the current payload, take one round of feedback ("accept"
or "refine"), either lock it in or apply the refinement and loop again.
Feedback can come from a human (``input()``) or from an LLM acting as an
automated decision-maker — the subgraph doesn't care which, only
``feedback_fn``'s contract matters.

For automated (no-human) sites where the loop can exhaust without ever
reaching "accept" — e.g.
``sam_microscopy_controllers.AutomatedLLMRefinementController``, which
invokes an LLM judge to pick the best of all attempted iterations rather
than just locking in the last one — pass ``judge_fn`` to
``build_refinement_subgraph``.

Graph topology
---------------

    [START]
       │
       ▼
   [render]
       │
       ▼
 [collect_feedback]
       │
       ├── accept ────────────────────────────────► [END]
       │
       └── refine ──► [apply_feedback]
                            │
                            ├── aborted ───────────────────► [END]  (apply_fn signaled failure)
                            │
                            ├── max_iterations_reached ──► [END]  (judge_fn picks the
                            │                                      winner if provided)
                            │
                            └── continue ──► [render]  (loop)

State
-----
See ``RefinementState`` in ``scilink/graphs/state.py``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from scilink.graphs.state import RefinementState

logger = logging.getLogger(__name__)

# Sentinel route values
_ROUTE_ACCEPT = "accept"
_ROUTE_REFINE = "refine"
_ROUTE_MAX = "max_iterations"
_ROUTE_ABORTED = "aborted"
_ROUTE_CONTINUE = "continue"


def sanitize_for_checkpoint(obj: Any) -> Any:
    """
    Recursively convert numpy scalar types to native Python types.

    ``MemorySaver`` checkpoints state via msgpack, which handles numpy
    ``ndarray`` natively but raises ``TypeError`` on numpy *scalar* types
    (``np.float64``, ``np.int64``, ``np.bool_``, …) — common several levels
    deep in dicts returned by scientific analysis code (e.g. atomai's SAM
    particle analyzer, ``mean()``/``std()`` results). ``payload``/``history``
    values pass through this before being checkpointed so a stray numpy
    scalar doesn't crash the write. Call this on the initial ``payload``
    too — LangGraph checkpoints the input state before any node runs.
    """
    np_mod = sys.modules.get("numpy")
    if np_mod is not None and isinstance(obj, np_mod.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: sanitize_for_checkpoint(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_checkpoint(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_checkpoint(v) for v in obj)
    return obj


def _make_nodes(
    render_fn: Optional[Callable[[RefinementState], None]],
    feedback_fn: Callable[[RefinementState], Dict[str, Any]],
    apply_fn: Callable[[RefinementState], Dict[str, Any]],
    max_iterations: int,
    judge_fn: Optional[Callable[[RefinementState, list], Dict[str, Any]]] = None,
):
    """
    Return node functions for the refinement loop.

    All nodes close over the provided callables; no controller logic moves
    into this module.
    """

    def render(state: RefinementState) -> Dict[str, Any]:
        """Display the current payload. Side-effect only; no state change."""
        if render_fn is not None:
            render_fn(state)
        return {}

    def collect_feedback(state: RefinementState) -> Dict[str, Any]:
        """
        Collect one round of feedback.

        ``feedback_fn`` must return a dict with at least ``"action"``:
        ``"accept"`` or ``"refine"``. On accept, this node locks in the
        current (unmodified) payload — mirrors the imperative loops'
        ``break`` on an empty/accept response.
        """
        logger.info(
            "   Refinement collect_feedback (iter=%d)", state.get("iteration", 0)
        )
        delta = feedback_fn(state)
        if "payload" in delta:
            delta["payload"] = sanitize_for_checkpoint(delta["payload"])
        if "history" in delta:
            delta["history"] = sanitize_for_checkpoint(delta["history"])
        action = delta.get("action", _ROUTE_ACCEPT)
        delta["action"] = action
        if action == _ROUTE_ACCEPT:
            delta["accepted"] = True
            delta["locked_payload"] = delta.get("payload", state.get("payload"))
        return delta

    def apply_feedback(state: RefinementState) -> Dict[str, Any]:
        """
        Apply the collected feedback to produce a refined payload.

        Mirrors: ``state = refine_plan(state, feedback); iteration += 1``.
        If this round exhausts ``max_iterations``, lock in the refined
        payload and stop — mirrors each controller's
        ``if iteration >= self.max_iterations: warn; proceed with current``.

        If ``apply_fn`` signals ``aborted=True`` (an unrecoverable failure
        mid-round), stop immediately without setting ``accepted`` — mirrors
        the ``except Exception: break`` path in
        ``sam_microscopy_controllers.HumanFeedbackRefinementController``,
        which does not run the max-iterations-reached branch either.

        On exhaustion, ``judge_fn`` (if provided) picks the winning payload
        from the full round-by-round ``history`` instead of just locking in
        whatever the last round produced — mirrors
        ``AutomatedLLMRefinementController``'s judge-selects-best-iteration
        fallback.
        """
        delta = apply_fn(state)
        if "payload" in delta:
            delta["payload"] = sanitize_for_checkpoint(delta["payload"])
        if "history" in delta:
            delta["history"] = sanitize_for_checkpoint(delta["history"])
        if delta.get("aborted"):
            logger.warning("   Refinement: apply_fn signaled abort — stopping")
            delta.setdefault("locked_payload", delta.get("payload", state.get("payload")))
            delta.setdefault("accepted", False)
            return delta

        new_iteration = state.get("iteration", 0) + 1
        delta["iteration"] = new_iteration
        if new_iteration >= max_iterations:
            delta["accepted"] = False
            if judge_fn is not None:
                logger.info(
                    "   Refinement: max_iterations (%d) reached — invoking judge_fn",
                    max_iterations,
                )
                full_history = state.get("history", [])
                judge_delta = judge_fn(state, full_history) or {}
                final_payload = sanitize_for_checkpoint(
                    judge_delta.get("payload", delta.get("payload", state.get("payload")))
                )
                delta["payload"] = final_payload
                delta["locked_payload"] = final_payload
            else:
                logger.warning(
                    "   Refinement: max_iterations (%d) reached — proceeding with current payload",
                    max_iterations,
                )
                delta["locked_payload"] = delta.get("payload", state.get("payload"))
        return delta

    return render, collect_feedback, apply_feedback


def _route_after_collect(state: RefinementState) -> str:
    if state.get("action") == _ROUTE_ACCEPT:
        return _ROUTE_ACCEPT
    return _ROUTE_REFINE


def _route_after_apply(state: RefinementState, max_iterations: int) -> str:
    if state.get("aborted"):
        return _ROUTE_ABORTED
    if state.get("iteration", 0) >= max_iterations:
        return _ROUTE_MAX
    return _ROUTE_CONTINUE


def build_refinement_subgraph(
    feedback_fn: Callable[[RefinementState], Dict[str, Any]],
    apply_fn: Callable[[RefinementState], Dict[str, Any]],
    render_fn: Optional[Callable[[RefinementState], None]] = None,
    judge_fn: Optional[Callable[[RefinementState, list], Dict[str, Any]]] = None,
    max_iterations: int = 3,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the generic accept/refine human-feedback subgraph.

    ``render_fn`` may be ``None`` when the display step is already folded
    into ``feedback_fn`` (e.g. the plan-refinement sites, whose existing
    ``_get_human_feedback`` methods display-then-prompt in one call).

    ``judge_fn(state, history) -> {"payload": ...}`` may be provided for
    automated (no-human) sites where exhausting ``max_iterations`` should
    pick the best of all attempted rounds rather than just locking in the
    last one. ``history`` is the full accumulated round-by-round record
    list. Ignored when the loop terminates via "accept" or "aborted".
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    render_node, collect_feedback_node, apply_feedback_node = _make_nodes(
        render_fn=render_fn,
        feedback_fn=feedback_fn,
        apply_fn=apply_fn,
        max_iterations=max_iterations,
        judge_fn=judge_fn,
    )

    def route_after_apply(state: RefinementState) -> str:
        return _route_after_apply(state, max_iterations)

    builder = StateGraph(RefinementState)

    builder.add_node("render", render_node)
    builder.add_node("collect_feedback", collect_feedback_node)
    builder.add_node("apply_feedback", apply_feedback_node)

    builder.add_edge(START, "render")
    builder.add_edge("render", "collect_feedback")

    builder.add_conditional_edges(
        "collect_feedback",
        _route_after_collect,
        {
            _ROUTE_ACCEPT: END,
            _ROUTE_REFINE: "apply_feedback",
        },
    )

    builder.add_conditional_edges(
        "apply_feedback",
        route_after_apply,
        {
            _ROUTE_ABORTED: END,
            _ROUTE_MAX: END,
            _ROUTE_CONTINUE: "render",
        },
    )

    return builder.compile(checkpointer=checkpointer)
