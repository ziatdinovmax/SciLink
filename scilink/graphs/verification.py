"""
scilink.graphs.verification
============================

Reusable verification-retry subgraph — 1:1 replacement for the imperative
``for verification_iter in range(...)`` loops in both:

* ``image_analysis_controllers.py:UnifiedImageProcessingController._execute_and_verify``
* ``curve_fitting_controllers.py:UnifiedSeriesProcessingController`` (fitting loop)

Two public builders are provided:

``build_verification_subgraph``
    Generic builder for image-analysis-style verification.  Uses LLM quality
    scores (0–1 float), patience-only annealing, and a score ≥ threshold
    approval condition.

``build_curve_fitting_verification_subgraph``
    Curve-fitting specialization.  Extends the generic builder with:
    - ``fit_acceptable`` boolean approval (``verify_fn`` sets it in state)
    - Physics-based promotion (``physically_better_than_best`` flag)
    - Rate-based annealing escalation (improvement / remaining < required_rate)
    - ``best_ever_rejected`` gate on the verifier-approval bypass
    - ``R2_FLOOR`` for catastrophic-regression rejection
    - Final extra verification pass when the loop exhausts (for-else semantics)

Graph topology (shared)
-----------------------

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
       ├── verify_failed ──────────────────────────────► [END]   (safe break)
       │
       ├── needs_human_feedback ──► [human_feedback] ──► [run_analysis]
       │
       ├── max_iterations_reached ──► [final_pass] ─────► [END]  (for-else pass)
       │
       └── needs_refinement ──► [apply_feedback]
                                        │
                                        ├── config_unchanged ──► [force_anneal] ──► (END or run_analysis)
                                        │
                                        ▼
                                   [anneal]
                                        │
                                        └──► [run_analysis]  (loop)

State
-----
See ``VerificationState`` in ``scilink/graphs/state.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from scilink.graphs.state import VerificationState

logger = logging.getLogger(__name__)

# Sentinel route values
_ROUTE_APPROVED = "approved"
_ROUTE_FAILED = "verify_failed"
_ROUTE_HUMAN = "human_feedback"
_ROUTE_REFINE = "apply_feedback"
_ROUTE_MAX = "max_iterations"
_ROUTE_UNCHANGED = "config_unchanged"
_ROUTE_CONTINUE = "continue"


# ---------------------------------------------------------------------------
# Shared node factory
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
    Return node functions for the generic (image-analysis) verification loop.

    All nodes close over the provided callables; no controller logic moves
    into this module.
    """

    def run_analysis(state: VerificationState) -> Dict[str, Any]:
        """Execute one analysis attempt using the current config."""
        logger.info(
            "   Verification run_analysis (iter=%d, annealing=%d)",
            state.get("iteration", 0),
            state.get("annealing_level", 0),
        )
        delta = run_fn(state)
        # Clear per-iteration flags
        delta.setdefault("verification_failed", False)
        delta.setdefault("config_unchanged", False)
        return delta

    def verify_quality(state: VerificationState) -> Dict[str, Any]:
        """
        Run LLM verification on the current result and update history.

        Mirrors the imperative loop body:
        - If verify_fn returns None → set verification_failed (break path).
        - Update best_score high-water mark if score improved.
        - Append rich history record (issues_found, overall_assessment, etc.).
        - Store last_verification for apply_feedback to read.
        """
        current = state.get("current_result")
        if not current or not current.get("success"):
            logger.warning("   verify_quality: no successful current_result — break")
            return {
                "iteration": state.get("iteration", 0) + 1,
                "verification_failed": True,
            }

        logger.info(
            "   Verification verify_quality (iter=%d)", state.get("iteration", 0)
        )

        try:
            verification = verify_fn(state)
        except Exception as e:
            logger.warning("   verify_quality: verify_fn raised %s — break", e)
            return {
                "iteration": state.get("iteration", 0) + 1,
                "verification_failed": True,
            }

        if verification is None:
            logger.warning("   verify_quality: verify_fn returned None — break")
            return {
                "iteration": state.get("iteration", 0) + 1,
                "verification_failed": True,
            }

        # Extract score
        v_score = verification.get("quality_score", verification.get("score", 0.0))
        if not isinstance(v_score, (int, float)):
            v_score = 0.0

        # High-water mark (mirrors: if v_score > best_score: best_score = v_score)
        current_best = state.get("best_score", 0.0)
        new_best_score = max(current_best, v_score)
        new_best_result = (
            current if v_score > current_best else state.get("best_result")
        )

        # Rich history record matching old controller fields
        record: Dict[str, Any] = {
            "iteration": state.get("iteration", 0),
            "annealing_level": state.get("annealing_level", 0),
            # Both field names: old controllers used "quality_score"; new field is "score"
            "quality_score": v_score,
            "score": v_score,
            "issues_found": verification.get("issues_found", []),
            "overall_assessment": verification.get("overall_assessment", ""),
            "recommended_action": verification.get("recommended_action", ""),
            "approved": v_score >= quality_threshold,
            "config_snapshot": dict(state.get("analysis_config", {})),
        }
        # Tag the record if the previous iteration's refinement failed
        if state.get("_last_refinement_error"):
            record["refinement_error"] = state["_last_refinement_error"]

        return {
            "iteration": state.get("iteration", 0) + 1,
            "best_score": new_best_score,
            "best_result": new_best_result,
            "prev_best_score": state.get("best_score", 0.0),
            "last_verification": verification,
            "verification_history": [record],
            "verification_failed": False,
            "approved": new_best_score >= quality_threshold,
            # Forward human_feedback_requested if verify_fn set it
            "human_feedback_requested": verification.get(
                "human_feedback_requested",
                state.get("human_feedback_requested", False),
            ),
        }

    def apply_feedback(state: VerificationState) -> Dict[str, Any]:
        """
        Apply LLM verification feedback to refine the analysis config.

        Also detects the no-config-change case (mirrors:
        ``if refined_config == state["locked_analysis_config"]: escalate``).
        """
        logger.info("   Verification apply_feedback")
        old_config = dict(state.get("analysis_config", {}))
        delta = feedback_fn(state)

        new_config = delta.get("analysis_config", old_config)
        config_unchanged = new_config == old_config

        delta["config_unchanged"] = config_unchanged
        if config_unchanged:
            logger.info(
                "   apply_feedback: config unchanged — will force-escalate annealing"
            )
        # Forward refinement errors into state so the next verify_quality
        # can tag the history record (mirrors old loop's verification_history[-1]["refinement_error"])
        refinement_error = delta.pop("_refinement_error", None)
        delta["_last_refinement_error"] = refinement_error or ""
        return delta

    def anneal(state: VerificationState) -> Dict[str, Any]:
        """
        Advance the annealing level based on the best-score high-water mark.

        Mirrors the image analysis controller exactly:
        - Compare best_score vs prev_best_score (not history[-2]).
        - If improved → reset stall counter.
        - If stalled for patience_limit → escalate +1 (capped at n-1).
        - Iteration floor: floor = min(iter // 2, n-1); lift if above current.
        """
        current_best = state.get("best_score", 0.0)
        prev_best = state.get("prev_best_score", 0.0)
        patience = state.get("patience_counter", 0)
        annealing_level = state.get("annealing_level", 0)
        iteration = state.get("iteration", 0)

        # Patience-based (mirrors _stall_count logic)
        if current_best > prev_best:
            new_patience = 0
        else:
            new_patience = patience + 1

        new_level = annealing_level
        if new_patience >= patience_limit and new_level < n_annealing_levels - 1:
            new_level = min(new_level + 1, n_annealing_levels - 1)
            new_patience = 0
            logger.info(
                "   Annealing: %d stalled iterations, escalating to level %d",
                patience_limit, new_level,
            )

        # Iteration floor (mirrors: _floor = min(verification_iter // 2, n-1))
        # iteration has already been incremented by verify_quality; subtract 1.
        iter_idx = max(iteration - 1, 0)
        floor = min(iter_idx // 2, n_annealing_levels - 1)
        if floor > new_level:
            logger.info(
                "   Annealing: iteration floor lifting level %d → %d",
                new_level, floor,
            )
            new_level = floor
            new_patience = 0

        return {
            "annealing_level": new_level,
            "patience_counter": new_patience,
        }

    def force_anneal(state: VerificationState) -> Dict[str, Any]:
        """
        Immediate escalation when apply_feedback produced no config change.

        Mirrors: ``if refined_config == locked: escalate +1; if at max: break``

        We store the pre-escalation level in ``_force_anneal_was_noop`` so
        the router can distinguish "successfully escalated → continue loop"
        from "already at max → break".
        """
        annealing_level = state.get("annealing_level", 0)
        new_level = min(annealing_level + 1, n_annealing_levels - 1)
        was_noop = (new_level == annealing_level)  # already at max before this call
        logger.info(
            "   force_anneal: no config changes, escalating %d → %d%s",
            annealing_level, new_level,
            " (at max — will break)" if was_noop else "",
        )
        # Reset config_unchanged so the next apply_feedback is evaluated fresh
        return {
            "annealing_level": new_level,
            "patience_counter": 0,
            "config_unchanged": False,
            "_force_anneal_was_noop": was_noop,
        }

    def final_pass(state: VerificationState) -> Dict[str, Any]:
        """
        One extra verification call after the loop budget is exhausted.

        Mirrors the for-else clause in both old controllers:
        image analysis: verify final current_result, promote if score > best.
        (Curve fitting has additional physics-promotion logic in its own
        final_pass node.)
        """
        current = state.get("current_result")
        if not current or not current.get("success"):
            logger.warning("   final_pass: no successful current_result — skipping")
            return {}

        logger.info("   final_pass: verifying final result...")
        try:
            final_verification = verify_fn(state)
        except Exception as e:
            logger.warning("   final_pass: verify_fn raised %s", e)
            return {}

        if not final_verification:
            return {}

        v_score = final_verification.get("quality_score", final_verification.get("score", 0.0))
        if not isinstance(v_score, (int, float)):
            v_score = 0.0

        current_best = state.get("best_score", 0.0)
        new_best_score = max(current_best, v_score)
        new_best_result = current if v_score > current_best else state.get("best_result")

        record: Dict[str, Any] = {
            "iteration": state.get("iteration", 0),
            "annealing_level": state.get("annealing_level", 0),
            "quality_score": v_score,
            "score": v_score,
            "issues_found": final_verification.get("issues_found", []),
            "overall_assessment": final_verification.get("overall_assessment", ""),
            "recommended_action": final_verification.get("recommended_action", ""),
            "approved": new_best_score >= quality_threshold,
            "config_snapshot": dict(state.get("analysis_config", {})),
            "final_pass": True,
        }

        return {
            "best_score": new_best_score,
            "best_result": new_best_result,
            "last_verification": final_verification,
            "verification_history": [record],
            "approved": new_best_score >= quality_threshold,
        }

    def human_feedback(state: VerificationState) -> Dict[str, Any]:
        """Surface a human-in-the-loop checkpoint."""
        if human_fn is None:
            logger.debug("   human_feedback: no human_fn — pass-through")
            return {"human_feedback_requested": False}
        logger.info("   human_feedback: requesting user input")
        delta = human_fn(state)
        delta["human_feedback_requested"] = False
        return delta

    return (
        run_analysis,
        verify_quality,
        apply_feedback,
        anneal,
        force_anneal,
        final_pass,
        human_feedback,
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_verify(
    state: VerificationState,
    quality_threshold: float,
    max_iterations: int,
) -> str:
    """
    Router after verify_quality — exact priority order from both controllers:
    1. Verification failed (None / exception) → END (safe break).
    2. Approved (score ≥ threshold or approved flag) → END.
    3. Max iterations reached → final_pass node (for-else semantics).
    4. Human feedback requested → human_feedback.
    5. Otherwise → apply_feedback.
    """
    if state.get("verification_failed"):
        return _ROUTE_FAILED

    if state.get("approved") or state.get("best_score", 0.0) >= quality_threshold:
        logger.info(
            "   Verification approved (score=%.2f)", state.get("best_score", 0.0)
        )
        return _ROUTE_APPROVED

    if state.get("iteration", 0) >= max_iterations:
        logger.info("   max_iterations reached — running final pass")
        return _ROUTE_MAX

    if state.get("human_feedback_requested"):
        return _ROUTE_HUMAN

    return _ROUTE_REFINE


def _route_after_force_anneal(
    state: VerificationState,
    n_annealing_levels: int,
) -> str:
    """
    After force_anneal: if the escalation was a no-op (already at max before
    the call), stop — this mirrors ``if _annealing_level == _cur_level: break``.
    Otherwise continue the loop at the new (higher) annealing level.
    """
    if state.get("_force_anneal_was_noop", False):
        logger.info("   force_anneal: already at max level — stopping loop")
        return END
    return "run_analysis"


def _route_after_apply_feedback(state: VerificationState) -> str:
    """Route to force_anneal if config unchanged, else to anneal."""
    if state.get("config_unchanged"):
        return _ROUTE_UNCHANGED
    return _ROUTE_CONTINUE


# ---------------------------------------------------------------------------
# Generic builder (image analysis)
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
    Build and compile the generic (image-analysis) verification subgraph.

    Behavioral parity with ``UnifiedImageProcessingController._execute_and_verify``:
    - Patience + iteration-floor annealing escalation.
    - No-config-change immediate escalation (or break at max level).
    - verify_fn returning None → safe break.
    - High-water-mark best_score tracking.
    - Rich history records (issues_found, overall_assessment, etc.).
    - Final extra verification pass when loop exhausts (for-else semantics).
    - Human feedback mid-loop checkpoint.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    (
        run_analysis_node,
        verify_quality_node,
        apply_feedback_node,
        anneal_node,
        force_anneal_node,
        final_pass_node,
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

    def route_after_verify(state: VerificationState) -> str:
        return _route_after_verify(state, quality_threshold, max_iterations)

    def route_after_force_anneal(state: VerificationState) -> str:
        return _route_after_force_anneal(state, n_annealing_levels)

    def route_after_apply_feedback(state: VerificationState) -> str:
        return _route_after_apply_feedback(state)

    builder = StateGraph(VerificationState)

    builder.add_node("run_analysis", run_analysis_node)
    builder.add_node("verify_quality", verify_quality_node)
    builder.add_node("apply_feedback", apply_feedback_node)
    builder.add_node("anneal", anneal_node)
    builder.add_node("force_anneal", force_anneal_node)
    builder.add_node("final_pass", final_pass_node)
    builder.add_node("human_feedback", human_feedback_node)

    builder.add_edge(START, "run_analysis")
    builder.add_edge("run_analysis", "verify_quality")

    builder.add_conditional_edges(
        "verify_quality",
        route_after_verify,
        {
            _ROUTE_APPROVED: END,
            _ROUTE_FAILED: END,
            _ROUTE_MAX: "final_pass",
            _ROUTE_HUMAN: "human_feedback",
            _ROUTE_REFINE: "apply_feedback",
        },
    )

    builder.add_conditional_edges(
        "apply_feedback",
        route_after_apply_feedback,
        {
            _ROUTE_UNCHANGED: "force_anneal",
            _ROUTE_CONTINUE: "anneal",
        },
    )

    builder.add_conditional_edges(
        "force_anneal",
        route_after_force_anneal,
        {
            END: END,
            "run_analysis": "run_analysis",
        },
    )

    builder.add_edge("anneal", "run_analysis")
    builder.add_edge("human_feedback", "run_analysis")
    builder.add_edge("final_pass", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Curve-fitting specialization
# ---------------------------------------------------------------------------


def _make_curve_fitting_nodes(
    run_fn: Callable[[VerificationState], Dict[str, Any]],
    verify_fn: Callable[[VerificationState], Dict[str, Any]],
    feedback_fn: Callable[[VerificationState], Dict[str, Any]],
    human_fn: Optional[Callable[[VerificationState], Dict[str, Any]]],
    n_annealing_levels: int,
    patience_limit: int,
):
    """
    Node factory for curve-fitting verification.

    Adds on top of the generic nodes:
    - Physics-based promotion (physically_better_than_best + R2_FLOOR).
    - fit_acceptable boolean approval.
    - Rate-based annealing escalation.
    - best_ever_rejected tracking.
    - best_verification passthrough.
    - Richer history (r_squared, physically_better_than_best, comparison_note).
    - Curve-fitting final_pass with retroactive physics promotion.
    """

    def run_analysis(state: VerificationState) -> Dict[str, Any]:
        logger.info(
            "   CF Verification run_analysis (iter=%d, annealing=%d)",
            state.get("iteration", 0),
            state.get("annealing_level", 0),
        )
        delta = run_fn(state)
        delta.setdefault("verification_failed", False)
        delta.setdefault("config_unchanged", False)
        return delta

    def verify_quality(state: VerificationState) -> Dict[str, Any]:
        """
        Curve-fitting verify node.

        Additional behaviors vs generic:
        - Reads current_r2 from current_result["fit_quality"]["r_squared"].
        - Physics-based promotion: if current != best AND current_r2 >= R2_FLOOR
          AND physically_better_than_best → promote.
        - fit_acceptable=True → sets approved=True (verifier-approval path).
        - Tracks best_verification and best_ever_rejected.
        - Stores r_squared, physically_better_than_best, comparison_note in history.
        """
        current = state.get("current_result")
        if not current or not current.get("success"):
            logger.warning("   CF verify_quality: no successful current_result — break")
            return {
                "iteration": state.get("iteration", 0) + 1,
                "verification_failed": True,
            }

        try:
            verification = verify_fn(state)
        except Exception as e:
            logger.warning("   CF verify_quality: verify_fn raised %s — break", e)
            return {
                "iteration": state.get("iteration", 0) + 1,
                "verification_failed": True,
            }

        if verification is None:
            logger.warning("   CF verify_quality: verify_fn returned None — break")
            return {
                "iteration": state.get("iteration", 0) + 1,
                "verification_failed": True,
            }

        current_r2 = current.get("fit_quality", {}).get("r_squared", 0.0)
        r2_floor = state.get("r2_floor", 0.0)
        best_result = state.get("best_result")
        best_r2 = state.get("best_r2", 0.0)

        was_rejected = not verification.get("fit_acceptable", True)

        # Physics-based promotion
        new_best_result = best_result
        new_best_r2 = best_r2
        new_best_ever_rejected = state.get("best_ever_rejected", False)
        new_best_verification = state.get("best_verification")

        if (current is not best_result
                and current_r2 >= r2_floor
                and verification.get("physically_better_than_best", False)):
            note = (verification.get("comparison_note") or "physics improvement")[:90]
            new_best_r2 = current_r2
            new_best_result = current
            new_best_ever_rejected = False
            new_best_verification = None
            logger.info(
                "   CF Retroactively promoted current (R²=%.4f) on physics — %s",
                current_r2, note,
            )

        # Track best_verification for the result that is currently best
        if current is new_best_result:
            new_best_verification = verification
            new_best_ever_rejected = new_best_ever_rejected or was_rejected

        # fit_acceptable → approval
        approved_by_verifier = not was_rejected
        if approved_by_verifier:
            new_best_r2 = current_r2
            new_best_result = current
            new_best_verification = verification
            new_best_ever_rejected = False
            logger.info("   CF ✅ Fit approved (R²=%.4f)", current_r2)

        # Numeric best_score mirrors best_r2 for routing compatibility
        new_best_score = new_best_r2

        record: Dict[str, Any] = {
            "iteration": state.get("iteration", 0),
            "annealing_level": state.get("annealing_level", 0),
            "r_squared": current_r2,
            "score": current_r2,
            "config_used": dict(state.get("analysis_config", {})),
            "config_snapshot": dict(state.get("analysis_config", {})),
            "issues_found": verification.get("issues_found", []),
            "overall_assessment": verification.get("overall_assessment", ""),
            "recommended_action": verification.get("recommended_action", ""),
            "physically_better_than_best": verification.get("physically_better_than_best", False),
            "comparison_note": verification.get("comparison_note", ""),
            "approved": approved_by_verifier,
        }
        if state.get("_last_refinement_error"):
            record["refinement_error"] = state["_last_refinement_error"]

        return {
            "iteration": state.get("iteration", 0) + 1,
            "best_score": new_best_score,
            "best_r2": new_best_r2,
            "best_result": new_best_result,
            "prev_best_score": state.get("best_score", 0.0),
            "best_verification": new_best_verification,
            "best_ever_rejected": new_best_ever_rejected,
            "last_verification": verification,
            "verification_history": [record],
            "verification_failed": False,
            "approved": approved_by_verifier,
        }

    def apply_feedback(state: VerificationState) -> Dict[str, Any]:
        old_config = dict(state.get("analysis_config", {}))
        logger.info("   CF apply_feedback")
        delta = feedback_fn(state)
        new_config = delta.get("analysis_config", old_config)
        config_unchanged = new_config == old_config
        delta["config_unchanged"] = config_unchanged
        if config_unchanged:
            logger.info("   CF apply_feedback: config unchanged — force-escalate")
        refinement_error = delta.pop("_refinement_error", None)
        delta["_last_refinement_error"] = refinement_error or ""
        return delta

    def anneal(state: VerificationState) -> Dict[str, Any]:
        """
        Curve-fitting annealing: three escalation triggers (a, b, c).

        (a) Rate-based: improvement / remaining < required_rate → escalate.
        (b) Patience-based: best stalled for patience_limit iters → escalate
            (only if rate trigger didn't already fire).
        (c) Iteration floor: floor = (iter+1) // floor_divisor → lift if above current.
        """
        current_best = state.get("best_r2", state.get("best_score", 0.0))
        prev_best = state.get("prev_best_score", 0.0)
        patience = state.get("patience_counter", 0)
        annealing_level = state.get("annealing_level", 0)
        iteration = state.get("iteration", 0)  # already incremented
        max_iterations = state.get("max_iterations", 7)
        r2_threshold = state.get("r2_threshold", 0.8)

        improvement = current_best - prev_best
        remaining = max(max_iterations - (iteration - 1) - 1, 1)
        required_rate = max(r2_threshold - current_best, 0.0) / remaining

        floor_divisor = max(max_iterations // n_annealing_levels, 1)
        iter_idx = max(iteration - 1, 0)

        new_level = annealing_level
        new_patience = patience
        rate_escalated = False

        # (a) Rate-based
        if improvement < required_rate:
            new_level = min(new_level + 1, n_annealing_levels - 1)
            rate_escalated = True
            logger.info(
                "   CF Annealing: improvement %.4f < required %.4f, level → %d",
                improvement, required_rate, new_level,
            )

        # (b) Patience-based (guarded by rate_escalated)
        if current_best > prev_best:
            new_patience = 0
        else:
            new_patience = patience + 1
            if new_patience >= patience_limit and not rate_escalated:
                candidate = min(new_level + 1, n_annealing_levels - 1)
                if candidate > new_level:
                    new_level = candidate
                    logger.info(
                        "   CF Annealing: stalled %d iters, level → %d",
                        new_patience, new_level,
                    )
                new_patience = 0

        # (c) Iteration floor
        floor = min((iter_idx + 1) // floor_divisor, n_annealing_levels - 1)
        if floor > new_level:
            logger.info(
                "   CF Annealing: floor lifting level %d → %d", new_level, floor
            )
            new_level = floor
            new_patience = 0

        return {
            "annealing_level": new_level,
            "patience_counter": new_patience,
        }

    def force_anneal(state: VerificationState) -> Dict[str, Any]:
        annealing_level = state.get("annealing_level", 0)
        new_level = min(annealing_level + 1, n_annealing_levels - 1)
        was_noop = (new_level == annealing_level)
        logger.info(
            "   CF force_anneal: no config changes, escalating %d → %d%s",
            annealing_level, new_level,
            " (at max — will break)" if was_noop else "",
        )
        return {
            "annealing_level": new_level,
            "patience_counter": 0,
            "config_unchanged": False,
            "_force_anneal_was_noop": was_noop,
        }

    def final_pass(state: VerificationState) -> Dict[str, Any]:
        """
        Curve-fitting for-else: one final verify after loop exhaustion.
        Includes retroactive physics promotion of a deferred current_result.
        """
        current = state.get("current_result")
        if not current or not current.get("success"):
            return {}

        logger.info("   CF final_pass: verifying final refit...")
        try:
            final_verification = verify_fn(state)
        except Exception as e:
            logger.warning("   CF final_pass: verify_fn raised %s", e)
            return {}

        if not final_verification:
            return {}

        current_r2 = current.get("fit_quality", {}).get("r_squared", 0.0)
        r2_floor = state.get("r2_floor", 0.0)
        best_result = state.get("best_result")
        best_r2 = state.get("best_r2", 0.0)
        best_ever_rejected = state.get("best_ever_rejected", False)

        final_rejected = not final_verification.get("fit_acceptable", True)

        new_best_result = best_result
        new_best_r2 = best_r2
        new_best_ever_rejected = best_ever_rejected
        fit_approved = False

        # Retroactive physics promotion of deferred current
        if (current is not best_result
                and current_r2 >= r2_floor
                and final_verification.get("physically_better_than_best", False)):
            note = (final_verification.get("comparison_note") or "physics improvement")[:90]
            new_best_r2 = current_r2
            new_best_result = current
            logger.info(
                "   CF final_pass: post-loop promoted current (R²=%.4f) on physics — %s",
                current_r2, note,
            )

        if current is new_best_result:
            if not final_rejected:
                logger.info("   CF ✅ Final fit approved (R²=%.4f)", new_best_r2)
                fit_approved = True
                new_best_ever_rejected = False
            else:
                new_best_ever_rejected = True

        record: Dict[str, Any] = {
            "iteration": state.get("iteration", 0),
            "annealing_level": state.get("annealing_level", 0),
            "r_squared": current_r2,
            "score": current_r2,
            "config_snapshot": dict(state.get("analysis_config", {})),
            "issues_found": final_verification.get("issues_found", []),
            "overall_assessment": final_verification.get("overall_assessment", ""),
            "recommended_action": final_verification.get("recommended_action", ""),
            "physically_better_than_best": final_verification.get("physically_better_than_best", False),
            "comparison_note": final_verification.get("comparison_note", ""),
            "approved": fit_approved,
            "final_pass": True,
        }

        return {
            "best_r2": new_best_r2,
            "best_result": new_best_result,
            "best_score": new_best_r2,
            "best_ever_rejected": new_best_ever_rejected,
            "best_verification": final_verification if current is new_best_result else state.get("best_verification"),
            "last_verification": final_verification,
            "verification_history": [record],
            "approved": fit_approved,
        }

    def human_feedback(state: VerificationState) -> Dict[str, Any]:
        if human_fn is None:
            return {"human_feedback_requested": False}
        delta = human_fn(state)
        delta["human_feedback_requested"] = False
        return delta

    return (
        run_analysis,
        verify_quality,
        apply_feedback,
        anneal,
        force_anneal,
        final_pass,
        human_feedback,
    )


def build_curve_fitting_verification_subgraph(
    run_fn: Callable[[VerificationState], Dict[str, Any]],
    verify_fn: Callable[[VerificationState], Dict[str, Any]],
    feedback_fn: Callable[[VerificationState], Dict[str, Any]],
    human_fn: Optional[Callable[[VerificationState], Dict[str, Any]]] = None,
    max_iterations: int = 7,
    r2_threshold: float = 0.8,
    n_annealing_levels: int = 3,
    patience_limit: int = 2,
    checkpointer: Any = None,
) -> Any:
    """
    Build the curve-fitting verification subgraph.

    Behavioral parity with the ``for verification_iter in range(...)`` loop
    in ``UnifiedSeriesProcessingController``.  Adds on top of the generic subgraph:

    - ``fit_acceptable`` boolean approval (verifier-approval path).
    - Physics-based promotion (``physically_better_than_best`` + R²-floor gate).
    - Rate-based annealing escalation (trigger a).
    - ``best_ever_rejected`` gate.
    - ``best_verification`` passthrough for context-rich subsequent prompts.
    - Final extra verification pass with retroactive physics promotion.
    - Richer history records (r_squared, physically_better_than_best, comparison_note).

    The initial state must include ``r2_threshold`` and ``r2_floor`` so the
    nodes can read them without closing over controller-instance state.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    (
        run_analysis_node,
        verify_quality_node,
        apply_feedback_node,
        anneal_node,
        force_anneal_node,
        final_pass_node,
        human_feedback_node,
    ) = _make_curve_fitting_nodes(
        run_fn=run_fn,
        verify_fn=verify_fn,
        feedback_fn=feedback_fn,
        human_fn=human_fn,
        n_annealing_levels=n_annealing_levels,
        patience_limit=patience_limit,
    )

    def route_after_verify(state: VerificationState) -> str:
        # Curve fitting: approved flag is set by fit_acceptable; no numeric threshold routing
        if state.get("verification_failed"):
            return _ROUTE_FAILED
        if state.get("approved"):
            return _ROUTE_APPROVED
        if state.get("iteration", 0) >= max_iterations:
            return _ROUTE_MAX
        if state.get("human_feedback_requested"):
            return _ROUTE_HUMAN
        return _ROUTE_REFINE

    def route_after_force_anneal(state: VerificationState) -> str:
        return _route_after_force_anneal(state, n_annealing_levels)

    def route_after_apply_feedback(state: VerificationState) -> str:
        return _route_after_apply_feedback(state)

    builder = StateGraph(VerificationState)

    builder.add_node("run_analysis", run_analysis_node)
    builder.add_node("verify_quality", verify_quality_node)
    builder.add_node("apply_feedback", apply_feedback_node)
    builder.add_node("anneal", anneal_node)
    builder.add_node("force_anneal", force_anneal_node)
    builder.add_node("final_pass", final_pass_node)
    builder.add_node("human_feedback", human_feedback_node)

    builder.add_edge(START, "run_analysis")
    builder.add_edge("run_analysis", "verify_quality")

    builder.add_conditional_edges(
        "verify_quality",
        route_after_verify,
        {
            _ROUTE_APPROVED: END,
            _ROUTE_FAILED: END,
            _ROUTE_MAX: "final_pass",
            _ROUTE_HUMAN: "human_feedback",
            _ROUTE_REFINE: "apply_feedback",
        },
    )

    builder.add_conditional_edges(
        "apply_feedback",
        route_after_apply_feedback,
        {
            _ROUTE_UNCHANGED: "force_anneal",
            _ROUTE_CONTINUE: "anneal",
        },
    )

    builder.add_conditional_edges(
        "force_anneal",
        route_after_force_anneal,
        {
            END: END,
            "run_analysis": "run_analysis",
        },
    )

    builder.add_edge("anneal", "run_analysis")
    builder.add_edge("human_feedback", "run_analysis")
    builder.add_edge("final_pass", END)

    return builder.compile(checkpointer=checkpointer)
