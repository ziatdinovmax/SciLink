"""
scilink.graphs.verification
============================

Reusable verification-retry subgraph.

This subgraph replaces the ``while`` loop that currently appears in both
``image_analysis_controllers.py:_execute_and_verify`` and
``curve_fitting_controllers.py``.  The logic (annealing, patience, best-result
tracking) is identical — it is a structural replacement, not a redesign.

Graph topology
--------------

    [START]
       │
       ▼
  [run_analysis]
       │
       ▼
  [verify_quality]
       │
       ├── approved ──────────────────────────────────► [END]
       │
       ├── needs_human_feedback ──► [human_feedback] ──► [run_analysis]
       │
       ├── max_iterations_reached ─────────────────────► [END]
       │
       └── needs_refinement ──► [apply_feedback]
                                        │
                                        ▼
                                   [anneal]
                                        │
                                        └──► [run_analysis]  (loop)

State
-----
See ``VerificationState`` in ``scilink/graphs/state.py``.

Usage
-----
The subgraph is composed into a controller like this::

    from scilink.graphs.verification import build_verification_subgraph

    subgraph = build_verification_subgraph(
        run_fn=controller._run_analysis_attempt,
        verify_fn=controller._verify_quality,
        feedback_fn=controller._apply_verification_feedback,
        human_fn=controller._get_human_feedback,   # or None for non-interactive
        max_iterations=7,
        quality_threshold=0.7,
        n_annealing_levels=3,
    )

    result = subgraph.invoke(initial_state)

The callables receive the ``VerificationState`` dict and return a state delta
(``dict``).  They are provided by the controller so no controller logic moves
into this module.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from scilink.graphs.state import VerificationState

logger = logging.getLogger(__name__)

# Sentinel route values used by the routing function
_ROUTE_APPROVED = "approved"
_ROUTE_HUMAN = "human_feedback"
_ROUTE_REFINE = "apply_feedback"
_ROUTE_MAX = "max_iterations"


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def _make_nodes(
    run_fn: Callable[[VerificationState], Dict[str, Any]],
    verify_fn: Callable[[VerificationState], Dict[str, Any]],
    feedback_fn: Callable[[VerificationState], Dict[str, Any]],
    human_fn: Optional[Callable[[VerificationState], Dict[str, Any]]],
    quality_threshold: float,
    n_annealing_levels: int,
    patience_limit: int,
):
    """
    Return node functions that close over the provided callables.

    Parameters
    ----------
    run_fn
        Callable that runs one analysis attempt.  Receives state, returns
        ``{"current_result": ..., "best_result": ..., "best_score": ...}``.
    verify_fn
        Callable that runs LLM verification on ``state["current_result"]``.
        Returns ``{"verification_result": dict, "score": float}``.
    feedback_fn
        Callable that updates ``state["analysis_config"]`` based on
        verification feedback.  Returns ``{"analysis_config": dict}``.
    human_fn
        Optional callable that prompts the user and returns
        ``{"analysis_config": dict}`` or ``{"approved": True}``.
        When ``None``, the human_feedback node is a no-op pass-through.
    quality_threshold
        Score above which a result is considered approved.
    n_annealing_levels
        Number of constraint-annealing levels (typically 3: tight/warm/hot).
    patience_limit
        Number of consecutive non-improving iterations before escalating
        the annealing level.
    """

    def run_analysis(state: VerificationState) -> Dict[str, Any]:
        """Execute one analysis attempt using the current config."""
        logger.info(
            "   Verification run_analysis (iter=%d, annealing=%d)",
            state.get("iteration", 0),
            state.get("annealing_level", 0),
        )
        delta = run_fn(state)
        return delta

    def verify_quality(state: VerificationState) -> Dict[str, Any]:
        """Run LLM verification on the current result and update history."""
        current = state.get("current_result")
        if not current or not current.get("success"):
            logger.warning("   verify_quality: no successful current_result — skipping")
            return {"iteration": state.get("iteration", 0) + 1}

        logger.info(
            "   Verification verify_quality (iter=%d)", state.get("iteration", 0)
        )
        delta = verify_fn(state)

        # Accumulate into verification_history via the reducer
        record = {
            "iteration": state.get("iteration", 0),
            "annealing_level": state.get("annealing_level", 0),
            "score": delta.get("best_score", state.get("best_score", 0.0)),
            "approved": delta.get("approved", False),
            "config_snapshot": dict(state.get("analysis_config", {})),
        }
        delta.setdefault("verification_history", [])
        if isinstance(delta.get("verification_history"), list):
            delta["verification_history"] = delta["verification_history"] + [record]
        else:
            delta["verification_history"] = [record]

        delta["iteration"] = state.get("iteration", 0) + 1
        return delta

    def apply_feedback(state: VerificationState) -> Dict[str, Any]:
        """Apply LLM verification feedback to refine the analysis config."""
        logger.info("   Verification apply_feedback")
        return feedback_fn(state)

    def anneal(state: VerificationState) -> Dict[str, Any]:
        """
        Advance the annealing level if best_score has stalled.

        Mirrors the adaptive patience-counter logic from the controllers:
        - If score improved this iteration → reset patience counter.
        - If score stalled for `patience_limit` iterations → escalate.
        - Cap at (n_annealing_levels - 1).
        """
        current_score = state.get("best_score", 0.0)
        patience = state.get("patience_counter", 0)
        annealing_level = state.get("annealing_level", 0)

        # Compare against the last recorded score in history
        history = state.get("verification_history", [])
        prev_score = history[-2]["score"] if len(history) >= 2 else -1.0

        if current_score > prev_score:
            # Score improved — reset patience
            new_patience = 0
        else:
            new_patience = patience + 1

        # Escalate if stalled long enough
        new_level = annealing_level
        if new_patience >= patience_limit and new_level < n_annealing_levels - 1:
            new_level = min(new_level + 1, n_annealing_levels - 1)
            new_patience = 0
            logger.info(
                "   Verification annealing escalated: %d → %d", annealing_level, new_level
            )

        return {
            "annealing_level": new_level,
            "patience_counter": new_patience,
        }

    def human_feedback(state: VerificationState) -> Dict[str, Any]:
        """
        Surface a human-in-the-loop checkpoint.

        In CO_PILOT / SUPERVISED modes the controller provides a ``human_fn``
        that prompts the user and returns either a config update or approval.
        When ``human_fn`` is None (AUTONOMOUS), this node is a no-op and
        the loop simply continues.
        """
        if human_fn is None:
            logger.debug("   Verification human_feedback: no human_fn — pass-through")
            return {"human_feedback_requested": False}

        logger.info("   Verification human_feedback: requesting user input")
        delta = human_fn(state)
        delta["human_feedback_requested"] = False
        return delta

    return run_analysis, verify_quality, apply_feedback, anneal, human_feedback


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_verification(
    state: VerificationState,
    quality_threshold: float,
    max_iterations: int,
) -> str:
    """
    Decide the next step after ``verify_quality``.

    Priority (matches controller logic):
    1. Approved (score ≥ threshold) → END
    2. Max iterations reached → END
    3. Human feedback requested → human_feedback
    4. Otherwise → apply_feedback (continue refining)
    """
    if state.get("approved") or state.get("best_score", 0.0) >= quality_threshold:
        logger.info(
            "   Verification approved (score=%.2f ≥ threshold=%.2f)",
            state.get("best_score", 0.0),
            quality_threshold,
        )
        return _ROUTE_APPROVED

    iteration = state.get("iteration", 0)
    if iteration >= max_iterations:
        logger.warning(
            "   Verification max_iterations (%d) reached without approval", max_iterations
        )
        return _ROUTE_MAX

    if state.get("human_feedback_requested"):
        return _ROUTE_HUMAN

    return _ROUTE_REFINE


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_verification_subgraph(
    run_fn: Callable[[VerificationState], Dict[str, Any]],
    verify_fn: Callable[[VerificationState], Dict[str, Any]],
    feedback_fn: Callable[[VerificationState], Dict[str, Any]],
    human_fn: Optional[Callable[[VerificationState], Dict[str, Any]]] = None,
    max_iterations: int = 7,
    quality_threshold: float = 0.7,
    n_annealing_levels: int = 3,
    patience_limit: int = 2,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the verification-retry subgraph.

    Parameters
    ----------
    run_fn
        One analysis execution attempt.
    verify_fn
        LLM quality verification.
    feedback_fn
        Config refinement based on verification output.
    human_fn
        Optional human-in-the-loop callback.
    max_iterations
        Upper bound on verification rounds (mirrors
        ``DEFAULT_MAX_VERIFICATION_ITERATIONS = 7`` from the controllers).
    quality_threshold
        Score above which the result is considered approved (0–1 scale).
    n_annealing_levels
        Number of constraint-annealing temperature levels.
    patience_limit
        Stall tolerance before escalating the annealing level.
    checkpointer
        LangGraph checkpointer.  Defaults to ``MemorySaver()``.

    Returns
    -------
    Compiled ``CompiledGraph`` ready for ``.invoke(initial_state)``.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    (
        run_analysis_node,
        verify_quality_node,
        apply_feedback_node,
        anneal_node,
        human_feedback_node,
    ) = _make_nodes(
        run_fn=run_fn,
        verify_fn=verify_fn,
        feedback_fn=feedback_fn,
        human_fn=human_fn,
        quality_threshold=quality_threshold,
        n_annealing_levels=n_annealing_levels,
        patience_limit=patience_limit,
    )

    def route_verification(state: VerificationState) -> str:
        return _route_verification(state, quality_threshold, max_iterations)

    builder = StateGraph(VerificationState)

    # Nodes
    builder.add_node("run_analysis", run_analysis_node)
    builder.add_node("verify_quality", verify_quality_node)
    builder.add_node("apply_feedback", apply_feedback_node)
    builder.add_node("anneal", anneal_node)
    builder.add_node("human_feedback", human_feedback_node)

    # Edges
    builder.add_edge(START, "run_analysis")
    builder.add_edge("run_analysis", "verify_quality")
    builder.add_conditional_edges(
        "verify_quality",
        route_verification,
        {
            _ROUTE_APPROVED: END,
            _ROUTE_MAX: END,
            _ROUTE_HUMAN: "human_feedback",
            _ROUTE_REFINE: "apply_feedback",
        },
    )
    builder.add_edge("apply_feedback", "anneal")
    builder.add_edge("anneal", "run_analysis")
    builder.add_edge("human_feedback", "run_analysis")

    return builder.compile(checkpointer=checkpointer)
