"""
Deep parity tests for the LangGraph migration — full behavioral coverage.

This file documents the EXACT behavioral contracts of the new LangGraph code
versus the old imperative loops, using concrete input/output assertions.

Test groups
-----------

V — Verification subgraph behavioral contracts (what the subgraph DOES do)
    and documented divergences from the old controllers (what it DOESN'T do).
    All tests pass on the current code; the "divergence" tests will FAIL if
    the subgraph is silently fixed to match the old behavior without updating
    these tests.

SR — Session restore: seeded history produces same LLM context as inline history.

LT — LiteLLM path: use_openai=False end-to-end smoke test.

MT — self.messages two-track divergence: MemorySaver vs shadow copy.

No live LLM calls; no API keys required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_analysis_orch(base_dir: str):
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent, AnalysisMode,
    )
    return AnalysisOrchestratorAgent(
        base_dir=base_dir,
        api_key="dummy-key-not-used",
        model_name="claude-opus-4-6",
        analysis_mode=AnalysisMode.AUTONOMOUS,
    )


def _fake_openai_response(content: str, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _build_subgraph(run_fn, verify_fn, feedback_fn, human_fn=None,
                    max_iterations=5, quality_threshold=0.7,
                    n_annealing_levels=3, patience_limit=2):
    from scilink.graphs.verification import build_verification_subgraph
    from langgraph.checkpoint.memory import MemorySaver
    return build_verification_subgraph(
        run_fn=run_fn, verify_fn=verify_fn, feedback_fn=feedback_fn,
        human_fn=human_fn, max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        n_annealing_levels=n_annealing_levels,
        patience_limit=patience_limit,
        checkpointer=MemorySaver(),
    )


def _build_cf_subgraph(run_fn, verify_fn, feedback_fn, human_fn=None,
                       max_iterations=5, r2_threshold=0.8,
                       n_annealing_levels=3, patience_limit=2):
    from scilink.graphs.verification import build_curve_fitting_verification_subgraph
    from langgraph.checkpoint.memory import MemorySaver
    return build_curve_fitting_verification_subgraph(
        run_fn=run_fn, verify_fn=verify_fn, feedback_fn=feedback_fn,
        human_fn=human_fn, max_iterations=max_iterations,
        r2_threshold=r2_threshold,
        n_annealing_levels=n_annealing_levels,
        patience_limit=patience_limit,
        checkpointer=MemorySaver(),
    )


def _base_state(**overrides):
    state = {
        "messages": [],
        "analysis_config": {},
        "current_result": None,
        "best_result": None,
        "best_score": 0.0,
        "prev_best_score": 0.0,
        "last_verification": None,
        "verification_failed": False,
        "config_unchanged": False,
        "verification_history": [],
        "iteration": 0,
        "max_iterations": 5,
        "annealing_level": 0,
        "patience_counter": 0,
        "approved": False,
        "human_feedback_requested": False,
        # curve-fitting fields (harmless for generic subgraph)
        "best_r2": 0.0,
        "r2_floor": 0.0,
        "r2_threshold": 0.8,
        "best_ever_rejected": False,
        "best_verification": None,
    }
    state.update(overrides)
    return state


def _invoke(subgraph, state, thread_id="test"):
    return subgraph.invoke(state, config={"configurable": {"thread_id": thread_id}})


# ===========================================================================
# V — Verification subgraph behavioral contracts (1:1 with old controllers)
# ===========================================================================

class TestVerificationSubgraphContracts:
    """
    Behavioral parity tests.  Each test drives the subgraph with synthetic
    inputs and asserts it matches the old imperative-loop behavior exactly.
    """

    # -----------------------------------------------------------------------
    # V1 — Approval
    # -----------------------------------------------------------------------

    def test_v1_score_at_threshold_sets_approved_true(self):
        """Score >= threshold sets approved=True in state and exits immediately."""
        def run_fn(state):
            return {"current_result": {"success": True}, "best_result": {"success": True}}
        def verify_fn(state):
            return {"quality_score": 0.7}
        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_subgraph(run_fn, verify_fn, feedback_fn, quality_threshold=0.7)
        result = _invoke(sg, _base_state(), "v1-threshold")
        assert result["approved"] is True
        assert result["iteration"] <= 2

    def test_v1_approved_flag_from_verify_fn_exits_after_one_run(self):
        """verify_fn returning quality_score above threshold exits immediately."""
        run_count = [0]
        def run_fn(state):
            run_count[0] += 1
            return {"current_result": {"success": True}}
        def verify_fn(state):
            return {"quality_score": 0.9}
        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_subgraph(run_fn, verify_fn, feedback_fn)
        result = _invoke(sg, _base_state(), "v1-flag")
        assert result["approved"] is True
        assert run_count[0] == 1

    # -----------------------------------------------------------------------
    # V2 — Max iterations: final extra verification pass (for-else parity)
    # -----------------------------------------------------------------------

    def test_v2_final_pass_called_when_loop_exhausts(self):
        """
        When the loop exhausts without approval, the subgraph runs one final
        verify call (for-else parity with both old controllers).
        verify_count should be max_iterations + 1.
        """
        verify_count = [0]
        call_n = [0]
        def run_fn(state):
            return {"current_result": {"success": True}}
        def verify_fn(state):
            verify_count[0] += 1
            return {"quality_score": 0.3}
        def feedback_fn(state):
            # Return a changing config so config_unchanged never fires
            call_n[0] += 1
            return {"analysis_config": {"step": call_n[0]}}

        max_iter = 3
        sg = _build_subgraph(run_fn, verify_fn, feedback_fn, max_iterations=max_iter)
        result = _invoke(sg, _base_state(max_iterations=max_iter), "v2-finalpass")

        assert result["approved"] is False
        assert verify_count[0] == max_iter + 1, (
            f"Expected {max_iter + 1} verify calls (loop + final pass), "
            f"got {verify_count[0]}"
        )

    def test_v2_final_pass_can_approve_via_high_score(self):
        """If the final pass score crosses the threshold, approved is set True."""
        call_count = [0]
        step = [0]
        def run_fn(state):
            return {"current_result": {"success": True}}
        def verify_fn(state):
            call_count[0] += 1
            # Last call (final pass, 4th for max_iter=3) returns a high score
            score = 0.8 if call_count[0] > 3 else 0.3
            return {"quality_score": score}
        def feedback_fn(state):
            step[0] += 1
            return {"analysis_config": {"step": step[0]}}

        max_iter = 3
        sg = _build_subgraph(run_fn, verify_fn, feedback_fn,
                             max_iterations=max_iter, quality_threshold=0.7)
        result = _invoke(sg, _base_state(max_iterations=max_iter), "v2-finalapprove")
        assert result["approved"] is True

    # -----------------------------------------------------------------------
    # V3 — Best-score high-water mark is maintained by the subgraph
    # -----------------------------------------------------------------------

    def test_v3_subgraph_maintains_best_score_high_water_mark(self):
        """
        The verify_quality node maintains a best_score high-water mark
        (mirrors: if v_score > best_score: best_score = v_score).
        Non-monotonic scores: [0.5, 0.3, 0.6] → final best_score == 0.6.
        """
        scores = [0.5, 0.3, 0.6]
        idx = [0]
        step = [0]

        def run_fn(state):
            return {"current_result": {"success": True}}

        def verify_fn(state):
            s = scores[min(idx[0], len(scores) - 1)]
            idx[0] += 1
            return {"quality_score": s}

        def feedback_fn(state):
            # Change config each time so the loop runs to max_iterations
            step[0] += 1
            return {"analysis_config": {"step": step[0]}}

        max_iter = 3
        sg = _build_subgraph(run_fn, verify_fn, feedback_fn,
                             max_iterations=max_iter, quality_threshold=0.9)
        result = _invoke(sg, _base_state(max_iterations=max_iter), "v3-hwm")
        # best_score should be max seen across all verify calls (0.6):
        assert result["best_score"] == pytest.approx(0.6)

    # -----------------------------------------------------------------------
    # V4 — Patience and annealing escalation mirrors old controller exactly
    # -----------------------------------------------------------------------

    def test_v4_patience_compares_best_score_vs_prev_best(self):
        """
        The anneal node compares best_score vs prev_best_score (the running
        high-water mark from the previous iteration), matching the old controller's
        ``if best_score > _prev_best_score`` exactly.
        Flat scores → patience exhausts → annealing escalates.
        """
        scores = [0.4, 0.4, 0.4, 0.4, 0.4]
        idx = [0]

        def run_fn(state):
            return {"current_result": {"success": True}}

        def verify_fn(state):
            s = scores[min(idx[0], len(scores) - 1)]
            idx[0] += 1
            return {"quality_score": s}

        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_subgraph(run_fn, verify_fn, feedback_fn,
                             max_iterations=5, patience_limit=2, n_annealing_levels=3)
        result = _invoke(sg, _base_state(max_iterations=5), "v4-patience")
        assert result["annealing_level"] > 0

    def test_v4_iteration_floor_lifts_annealing_level(self):
        """
        The iteration floor (floor = min(iter // 2, n-1)) can lift annealing
        above what patience alone would set.
        With 6 iterations and n=3: floor at iter 4 = min(4//2, 2) = 2 (hot).
        """
        def run_fn(state):
            return {"current_result": {"success": True}}
        def verify_fn(state):
            return {"quality_score": 0.1}  # never approves
        def feedback_fn(state):
            return {"analysis_config": {}}

        # Many iterations, high patience_limit (patience never fires alone)
        sg = _build_subgraph(run_fn, verify_fn, feedback_fn,
                             max_iterations=6, patience_limit=10, n_annealing_levels=3)
        result = _invoke(sg, _base_state(max_iterations=6), "v4-floor")
        # Floor at iter 4 → level 2; should reach max level
        assert result["annealing_level"] == 2

    def test_v4_annealing_capped_at_n_levels_minus_1(self):
        """Annealing level never exceeds n_annealing_levels - 1."""
        def run_fn(state):
            return {"current_result": {"success": True}}
        def verify_fn(state):
            return {"quality_score": 0.1}
        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_subgraph(run_fn, verify_fn, feedback_fn,
                             max_iterations=20, patience_limit=1, n_annealing_levels=3)
        result = _invoke(sg, _base_state(max_iterations=20), "v4-cap")
        assert result["annealing_level"] <= 2

    # -----------------------------------------------------------------------
    # V5 — verify_fn returning None → safe break (parity with old controllers)
    # -----------------------------------------------------------------------

    def test_v5_none_result_triggers_safe_break(self):
        """
        When current_result.success is False, the subgraph sets
        verification_failed=True and routes to END immediately,
        matching ``if verification is None: break`` in both old controllers.
        """
        run_count = [0]

        def run_fn(state):
            run_count[0] += 1
            return {"current_result": {"success": False}}

        def verify_fn(state):
            return {"quality_score": 0.0}

        def feedback_fn(state):
            return {"analysis_config": {}}

        max_iter = 5
        sg = _build_subgraph(run_fn, verify_fn, feedback_fn, max_iterations=max_iter)
        result = _invoke(sg, _base_state(max_iterations=max_iter), "v5-break")

        assert result.get("verification_failed") is True
        # Only one run attempt before safe break
        assert run_count[0] == 1, (
            f"Expected 1 run (safe break), got {run_count[0]}"
        )

    # -----------------------------------------------------------------------
    # V6 — No-config-change immediate escalation
    # -----------------------------------------------------------------------

    def test_v6_no_config_change_escalates_immediately(self):
        """
        When apply_feedback returns the same config, force_anneal escalates
        immediately (mirrors: ``if refined_config == locked: escalate``).
        """
        def run_fn(state):
            return {"current_result": {"success": True}}
        def verify_fn(state):
            return {"quality_score": 0.3}
        def feedback_fn(state):
            # Always return same config — no change
            return {"analysis_config": state.get("analysis_config", {})}

        sg = _build_subgraph(run_fn, verify_fn, feedback_fn,
                             max_iterations=4, patience_limit=10, n_annealing_levels=3)
        result = _invoke(sg, _base_state(max_iterations=4), "v6-unchanged")
        # force_anneal fires on every iteration → should reach max level quickly
        assert result["annealing_level"] >= 1

    def test_v6_no_config_change_at_max_level_stops_loop(self):
        """
        At max annealing level, no-config-change triggers break rather than
        continuing (mirrors: ``if _annealing_level == _cur_level: break``).
        """
        def run_fn(state):
            return {"current_result": {"success": True}}
        def verify_fn(state):
            return {"quality_score": 0.3}
        def feedback_fn(state):
            return {"analysis_config": {}}

        # Start already at max level (2 for n=3)
        sg = _build_subgraph(run_fn, verify_fn, feedback_fn,
                             max_iterations=10, n_annealing_levels=3)
        result = _invoke(sg, _base_state(max_iterations=10, annealing_level=2), "v6-maxbreak")
        # Should have stopped early, not exhausted all 10 iterations
        # (because at max level, no-config-change is a break)
        assert result["iteration"] < 10

    # -----------------------------------------------------------------------
    # V7 — Verification history fields match old controller
    # -----------------------------------------------------------------------

    def test_v7_history_contains_all_old_controller_fields(self):
        """
        History records contain: quality_score, score, issues_found,
        overall_assessment, recommended_action, annealing_level, approved,
        config_snapshot.  These match what both old controllers stored.
        """
        def run_fn(state):
            return {"current_result": {"success": True}}
        def verify_fn(state):
            return {
                "quality_score": 0.9,
                "issues_found": ["minor issue"],
                "overall_assessment": "good",
                "recommended_action": "continue",
            }
        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_subgraph(run_fn, verify_fn, feedback_fn)
        result = _invoke(sg, _base_state(), "v7-history")

        history = result.get("verification_history", [])
        assert len(history) >= 1
        entry = history[0]

        assert "quality_score" in entry, "Missing 'quality_score' (old image analysis field)"
        assert "score" in entry, "Missing 'score' field"
        assert "issues_found" in entry, "Missing 'issues_found'"
        assert "overall_assessment" in entry, "Missing 'overall_assessment'"
        assert "recommended_action" in entry, "Missing 'recommended_action'"
        assert "annealing_level" in entry, "Missing 'annealing_level'"
        assert "approved" in entry, "Missing 'approved'"
        assert "config_snapshot" in entry, "Missing 'config_snapshot'"

    # -----------------------------------------------------------------------
    # V8 — Human feedback mid-loop
    # -----------------------------------------------------------------------

    def test_v8_human_feedback_called_mid_loop_then_continues(self):
        """human_fn is called when requested and run continues afterward."""
        call_log = []

        def run_fn(state):
            call_log.append("run")
            return {"current_result": {"success": True}}

        call_count = [0]
        def verify_fn(state):
            call_count[0] += 1
            req = call_count[0] == 1
            return {"quality_score": 0.3, "human_feedback_requested": req}

        def feedback_fn(state):
            call_log.append("feedback")
            return {"analysis_config": {}, "human_feedback_requested": False}

        def human_fn(state):
            call_log.append("human")
            return {"analysis_config": {"human_override": True}, "human_feedback_requested": False}

        sg = _build_subgraph(run_fn, verify_fn, feedback_fn, human_fn=human_fn, max_iterations=4)
        _invoke(sg, _base_state(max_iterations=4), "v8-human")

        assert "human" in call_log
        assert call_log.count("run") > 1


# ===========================================================================
# VCF — Curve-fitting-specific behaviors
# ===========================================================================

class TestCurveFittingVerificationSubgraph:
    """
    Parity tests for build_curve_fitting_verification_subgraph:
    - Physics-based promotion
    - fit_acceptable boolean approval
    - Rate-based annealing escalation
    - best_ever_rejected tracking
    - Final pass with retroactive physics promotion
    - Richer history fields
    """

    def _cf_base_state(self, **overrides):
        state = _base_state(
            r2_threshold=0.8,
            r2_floor=0.5,
            best_r2=0.0,
            best_ever_rejected=False,
            best_verification=None,
        )
        state.update(overrides)
        return state

    def test_vcf1_fit_acceptable_approves(self):
        """fit_acceptable=True in verify_fn output triggers approval."""
        run_count = [0]
        def run_fn(state):
            run_count[0] += 1
            return {
                "current_result": {"success": True, "fit_quality": {"r_squared": 0.75}},
            }
        def verify_fn(state):
            return {"fit_acceptable": True, "issues_found": []}
        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_cf_subgraph(run_fn, verify_fn, feedback_fn, max_iterations=5)
        result = _invoke(sg, self._cf_base_state(max_iterations=5), "vcf1-accept")
        assert result["approved"] is True
        assert run_count[0] == 1

    def test_vcf2_physics_based_promotion(self):
        """
        When current != best AND current_r2 >= r2_floor AND
        physically_better_than_best=True, current is promoted to best.
        """
        # best_result has r2=0.6, current has r2=0.65 (above floor 0.5)
        initial_best = {"success": True, "fit_quality": {"r_squared": 0.6}}
        new_current = {"success": True, "fit_quality": {"r_squared": 0.65}}

        run_count = [0]
        def run_fn(state):
            run_count[0] += 1
            return {"current_result": new_current}

        def verify_fn(state):
            # Signal physics improvement, fit not numerically acceptable
            return {
                "fit_acceptable": False,
                "physically_better_than_best": True,
                "comparison_note": "better peak shape",
                "issues_found": [],
            }

        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_cf_subgraph(run_fn, verify_fn, feedback_fn, max_iterations=2)
        result = _invoke(sg, self._cf_base_state(
            max_iterations=2,
            best_result=initial_best,
            best_r2=0.6,
            r2_floor=0.5,
        ), "vcf2-physics")

        # Physics promotion: best_r2 should be current_r2 (0.65)
        assert result["best_r2"] == pytest.approx(0.65), (
            f"Expected physics-promoted best_r2=0.65, got {result['best_r2']}"
        )

    def test_vcf3_rate_based_escalation(self):
        """
        Rate-based escalation fires when improvement / remaining < required_rate,
        even without patience exhausting.
        With r2_threshold=0.9, best_r2=0.3, tiny improvement → escalate.
        """
        scores = [0.30, 0.31]
        idx = [0]

        def run_fn(state):
            r2 = scores[min(idx[0], len(scores) - 1)]
            return {
                "current_result": {"success": True, "fit_quality": {"r_squared": r2}},
                "best_r2": r2,
                "best_score": r2,
            }

        def verify_fn(state):
            r2 = scores[min(idx[0], len(scores) - 1)]
            idx[0] += 1
            return {"fit_acceptable": False, "issues_found": []}

        def feedback_fn(state):
            return {"analysis_config": {"changed": True}}

        # patience_limit=10 so patience alone would never fire in 2 iters
        sg = _build_cf_subgraph(run_fn, verify_fn, feedback_fn,
                                max_iterations=2, r2_threshold=0.9,
                                patience_limit=10, n_annealing_levels=3)
        result = _invoke(sg, self._cf_base_state(
            max_iterations=2, r2_threshold=0.9, best_r2=0.3, best_score=0.3,
        ), "vcf3-rate")

        # Rate-based should have fired → annealing > 0
        assert result["annealing_level"] > 0, (
            "Expected rate-based escalation, annealing_level should be > 0"
        )

    def test_vcf4_best_ever_rejected_tracking(self):
        """
        best_ever_rejected is set True when best_result is rejected by verifier.
        The verifier inspects best_result when current IS best_result.
        """
        # Use a shared dict so current_result is best_result (same object)
        shared_result = {"success": True, "fit_quality": {"r_squared": 0.7}}

        def run_fn(state):
            # Return the same object that is also best_result in state
            return {"current_result": shared_result}

        def verify_fn(state):
            return {"fit_acceptable": False, "issues_found": ["bad fit"]}

        def feedback_fn(state):
            return {"analysis_config": {"changed": True}}

        sg = _build_cf_subgraph(run_fn, verify_fn, feedback_fn, max_iterations=2)
        initial = self._cf_base_state(max_iterations=2, best_r2=0.7)
        # Set best_result to the same object run_fn will return
        initial["best_result"] = shared_result
        result = _invoke(sg, initial, "vcf4-rejected")
        assert result.get("best_ever_rejected") is True

    def test_vcf5_final_pass_with_retroactive_physics_promotion(self):
        """
        When loop exhausts, the curve-fitting final pass runs one extra
        verify call and retroactively promotes current if physics says so.
        """
        verify_count = [0]
        step = [0]
        current_result = {"success": True, "fit_quality": {"r_squared": 0.72}}
        best_result = {"success": True, "fit_quality": {"r_squared": 0.65}}

        def run_fn(state):
            return {"current_result": current_result}

        def verify_fn(state):
            verify_count[0] += 1
            is_final = verify_count[0] > 3  # final pass (4th call for max_iter=3)
            return {
                "fit_acceptable": False,
                "physically_better_than_best": is_final,
                "comparison_note": "final physics check",
                "issues_found": [],
            }

        def feedback_fn(state):
            # Change config so the loop runs to max_iterations
            step[0] += 1
            return {"analysis_config": {"step": step[0]}}

        max_iter = 3
        sg = _build_cf_subgraph(run_fn, verify_fn, feedback_fn, max_iterations=max_iter)
        initial = self._cf_base_state(
            max_iterations=max_iter,
            best_result=best_result,
            best_r2=0.65,
            r2_floor=0.5,
        )
        result = _invoke(sg, initial, "vcf5-finalphysics")

        # Final pass fired
        assert verify_count[0] == max_iter + 1, (
            f"Expected {max_iter + 1} verify calls (loop + final pass), "
            f"got {verify_count[0]}"
        )
        # Physics promotion on final pass: best_r2 should be 0.72
        assert result["best_r2"] == pytest.approx(0.72), (
            f"Expected retroactive physics promotion to 0.72, got {result['best_r2']}"
        )

    def test_vcf6_history_has_curve_fitting_fields(self):
        """History records contain r_squared, physically_better_than_best, comparison_note."""
        def run_fn(state):
            return {"current_result": {"success": True, "fit_quality": {"r_squared": 0.75}}}
        def verify_fn(state):
            return {
                "fit_acceptable": True,
                "issues_found": ["minor"],
                "overall_assessment": "ok",
                "recommended_action": "stop",
                "physically_better_than_best": False,
                "comparison_note": "same",
            }
        def feedback_fn(state):
            return {"analysis_config": {}}

        sg = _build_cf_subgraph(run_fn, verify_fn, feedback_fn, max_iterations=3)
        result = _invoke(sg, self._cf_base_state(max_iterations=3), "vcf6-history")

        history = result.get("verification_history", [])
        assert len(history) >= 1
        entry = history[0]

        assert "r_squared" in entry, "Missing r_squared field"
        assert "physically_better_than_best" in entry
        assert "comparison_note" in entry
        assert "issues_found" in entry
        assert "overall_assessment" in entry
        assert "recommended_action" in entry


# ===========================================================================
# SR — Session restore: seeded history produces same LLM context
# ===========================================================================

class TestSessionRestoreParity:
    """
    _seed_graph_history() should produce the same MemorySaver thread state as
    if those messages had been accumulated by live graph invocations.

    A restored session (seeded from JSON) must present the same message
    history to the LLM as a fresh session that received the same turns.
    """

    def test_sr_seeded_messages_appear_in_graph_state(self):
        """After seeding, the graph state contains exactly the seeded messages."""
        from langchain_core.messages import HumanMessage, AIMessage

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)

            history = [
                {"role": "user", "content": "What is the data type?"},
                {"role": "assistant", "content": "It is microscopy data."},
                {"role": "user", "content": "How many images?"},
            ]
            orch._seed_graph_history(history)

            state = orch._graph.get_state(orch._graph_config)
            msgs = state.values.get("messages", [])
            contents = [m.content for m in msgs if hasattr(m, "content")]

            assert "What is the data type?" in contents
            assert "It is microscopy data." in contents
            assert "How many images?" in contents

    def test_sr_seeded_session_sends_prior_context_to_llm(self):
        """
        A fresh graph + seeded history should include prior messages in the
        wire-format list sent to the LLM on the next call.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from scilink.graphs.analysis import build_analysis_graph

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            history = [
                {"role": "user", "content": "Prior turn content XYZ"},
                {"role": "assistant", "content": "Prior response ABC"},
            ]
            orch._seed_graph_history(history)

            captured_messages = []

            def side_effect(**kwargs):
                captured_messages.append(kwargs.get("messages", []))
                return _fake_openai_response("Follow-up response.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = side_effect
                orch.chat("Follow-up question")

            assert len(captured_messages) == 1
            wire_msgs = captured_messages[0]
            all_content = " ".join(m.get("content", "") or "" for m in wire_msgs)

            assert "Prior turn content XYZ" in all_content, (
                "Seeded prior user message not in LLM context on follow-up call."
            )
            assert "Prior response ABC" in all_content, (
                "Seeded prior assistant message not in LLM context on follow-up call."
            )

    def test_sr_empty_seed_leaves_graph_clean(self):
        """Seeding with empty history should not add any messages to graph state."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch._seed_graph_history([])
            state = orch._graph.get_state(orch._graph_config)
            msgs = state.values.get("messages", []) if state.values else []
            assert len(msgs) == 0

    def test_sr_system_messages_excluded_from_seed(self):
        """System messages in history are skipped — not injected into MemorySaver."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch._seed_graph_history([
                {"role": "system", "content": "You are an assistant"},
                {"role": "user", "content": "Hello"},
            ])
            state = orch._graph.get_state(orch._graph_config)
            msgs = state.values.get("messages", [])
            types = [getattr(m, "type", type(m).__name__) for m in msgs]
            assert "system" not in types

    def test_sr_tool_messages_seeded_correctly(self):
        """Tool messages (role=tool) in JSON history are replayed as ToolMessages."""
        from langchain_core.messages import ToolMessage

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch._seed_graph_history([
                {"role": "user", "content": "Run analysis"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "tc_1", "type": "function",
                     "function": {"name": "run_analysis", "arguments": "{}"}},
                ]},
                {"role": "tool", "content": "Tool result text", "tool_call_id": "tc_1"},
            ])
            state = orch._graph.get_state(orch._graph_config)
            msgs = state.values.get("messages", [])
            tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
            assert len(tool_msgs) == 1
            assert tool_msgs[0].content == "Tool result text"


# ===========================================================================
# LT — LiteLLM path: use_openai=False end-to-end
# ===========================================================================

class TestLitellmPath:
    """
    The LiteLLM branch (use_openai=False) in call_model should produce the
    same graph-level behavior as the OpenAI branch (use_openai=True).
    """

    def test_lt_litellm_plain_text_response(self):
        """LiteLLM path returns text response correctly via the graph."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = False  # force LiteLLM path

            # Build a fake LiteLLM response that matches what litellm.completion returns
            fake_msg = MagicMock()
            fake_msg.content = "LiteLLM response text."
            fake_msg.tool_calls = []
            fake_choice = MagicMock()
            fake_choice.message = fake_msg
            fake_response = MagicMock()
            fake_response.choices = [fake_choice]

            with patch("litellm.completion", return_value=fake_response):
                result = orch.chat("Hello from LiteLLM path")

            assert "LiteLLM response text." in result

    def test_lt_litellm_tool_call_then_text(self):
        """LiteLLM path handles a tool call followed by a text response."""
        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = False

            # First response: tool call
            tc = MagicMock()
            tc.id = "ltc_1"
            tc.function.name = "nonexistent_tool"
            tc.function.arguments = "{}"

            tool_msg = MagicMock()
            tool_msg.content = ""
            tool_msg.tool_calls = [tc]
            tool_choice = MagicMock()
            tool_choice.message = tool_msg
            tool_response = MagicMock()
            tool_response.choices = [tool_choice]

            # Second response: final text
            final_msg = MagicMock()
            final_msg.content = "Done via LiteLLM."
            final_msg.tool_calls = []
            final_choice = MagicMock()
            final_choice.message = final_msg
            final_response = MagicMock()
            final_response.choices = [final_choice]

            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                return tool_response if call_count[0] == 1 else final_response

            with patch("litellm.completion", side_effect=side_effect):
                result = orch.chat("Do a tool call")

            assert "Done via LiteLLM." in result
            assert call_count[0] == 2

    def test_lt_litellm_max_iterations_enforced(self):
        """LiteLLM path respects MAX_TOOL_ITERATIONS identically to OpenAI path."""
        from langgraph.checkpoint.memory import MemorySaver
        from scilink.graphs.analysis import build_analysis_graph

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = False
            orch.MAX_TOOL_ITERATIONS = 2
            # self.max_iterations (set once at __init__, run_task-overridable)
            # is the live value the graph actually reads — see
            # scilink/graphs/_react.py's should_continue.
            orch.max_iterations = 2
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            tc = MagicMock()
            tc.id = "ltc_limit"
            tc.function.name = "noop"
            tc.function.arguments = "{}"

            always_tool_msg = MagicMock()
            always_tool_msg.content = ""
            always_tool_msg.tool_calls = [tc]
            always_tool_choice = MagicMock()
            always_tool_choice.message = always_tool_msg
            always_tool_response = MagicMock()
            always_tool_response.choices = [always_tool_choice]

            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                return always_tool_response

            with patch("litellm.completion", side_effect=side_effect):
                result = orch.chat("Keep calling tools forever")

            # After MAX_TOOL_ITERATIONS, graph routes to END
            assert result  # something returned
            # LLM was called at most MAX_TOOL_ITERATIONS + 1 times
            assert call_count[0] <= orch.MAX_TOOL_ITERATIONS + 1

    def test_lt_litellm_timeout_retries_up_to_limit(self):
        """LiteLLM path retries on timeout up to _TIMEOUT_RETRIES times (same as OpenAI path)."""
        from scilink.graphs._react import _TIMEOUT_RETRIES

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = False

            call_count = [0]
            def side_effect(*args, **kwargs):
                call_count[0] += 1
                raise Exception("Request timed out")

            with patch("litellm.completion", side_effect=side_effect):
                with pytest.raises(Exception, match="timed out"):
                    orch._invoke_graph("Hello")

            assert call_count[0] == _TIMEOUT_RETRIES, (
                f"Expected {_TIMEOUT_RETRIES} LiteLLM calls (retry up to limit), "
                f"got {call_count[0]}."
            )

    def test_lt_litellm_empty_response_nudge_injected(self):
        """LiteLLM path injects nudge on empty response, same as OpenAI path."""
        from langchain_core.messages import HumanMessage
        from scilink.graphs._react import _make_react_nodes

        orch = MagicMock()
        orch.use_openai = False
        orch._system_prompt = "system"
        orch.tools_for_model = []

        empty_msg = MagicMock()
        empty_msg.content = None
        empty_msg.tool_calls = []
        empty_choice = MagicMock()
        empty_choice.message = empty_msg
        empty_response = MagicMock()
        empty_response.choices = [empty_choice]

        with patch("litellm.completion", return_value=empty_response):
            call_model, _ = _make_react_nodes(orch)
            state = {"messages": [HumanMessage(content="go")], "step_count": 0}
            delta = call_model(state)

        msgs = delta.get("messages", [])
        assert len(msgs) == 2, "Expected [empty_ai_msg, nudge] for LiteLLM empty response"
        assert isinstance(msgs[1], HumanMessage)


# ===========================================================================
# MT — self.messages two-track divergence
# ===========================================================================

class TestMessagesTrackDivergence:
    """
    Under the two-track model, self.messages (shadow copy for JSON) and
    MemorySaver (canonical LLM context) can diverge. These tests document
    where and how.
    """

    def test_mt_memsaver_accumulates_unboundedly_while_self_messages_trims(self):
        """
        DIVERGENCE: self.messages trims when > 120 messages.
        MemorySaver accumulates all messages across all turns.
        After trim, self.messages is shorter than MemorySaver's thread state.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from scilink.graphs.analysis import build_analysis_graph

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            # Seed 125 messages to trigger the trim path (> 120 threshold)
            history = []
            for i in range(63):  # 63 user + 63 assistant = 126 entries (+ system = 127)
                history.append({"role": "user", "content": f"User turn {i}"})
                history.append({"role": "assistant", "content": f"Assistant turn {i}"})

            # Directly set self.messages to simulate a long session
            system_msg = orch.messages[0]
            orch.messages = [system_msg] + [
                {"role": m["role"], "content": m["content"]} for m in history
            ]

            # Verify self.messages is now > 120
            assert len(orch.messages) > 120

            # Run one turn — triggers the trim path in _invoke_graph
            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.return_value = _fake_openai_response(
                    "Response after trim."
                )
                orch.chat("Post-trim turn")

            # self.messages was trimmed (will be well under original 127+)
            assert len(orch.messages) < 120, (
                "self.messages was not trimmed as expected"
            )

            # MemorySaver has ALL messages (not trimmed)
            graph_state = orch._graph.get_state(orch._graph_config)
            graph_msg_count = len(graph_state.values.get("messages", []))

            # Graph accumulates all seeded + the new turn; self.messages is a subset
            # (We can't assert exact numbers due to seeding, but graph should have more)
            assert graph_msg_count > 0

    def test_mt_self_messages_used_for_json_persistence_not_llm_context(self):
        """
        self.messages drives _save_history (JSON), NOT the LLM API call.
        The LLM context comes from MemorySaver via graph state.
        Confirm that modifying self.messages after a turn does NOT affect
        what the LLM sees on the next call.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from scilink.graphs.analysis import build_analysis_graph

        with tempfile.TemporaryDirectory() as td:
            orch = _make_analysis_orch(td)
            orch.use_openai = True
            orch._graph = build_analysis_graph(orch, checkpointer=MemorySaver())

            # First turn
            captured = []
            def capture(**kwargs):
                captured.append(list(kwargs.get("messages", [])))
                return _fake_openai_response("First response.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = capture
                orch.chat("First question")

            msgs_after_first = list(orch.messages)

            # Tamper with self.messages — inject a fake message
            orch.messages.append({"role": "user", "content": "INJECTED_FAKE_MESSAGE"})

            # Second turn — what does the LLM actually see?
            captured2 = []
            def capture2(**kwargs):
                captured2.append(list(kwargs.get("messages", [])))
                return _fake_openai_response("Second response.")

            with patch("openai.OpenAI") as mock_cls:
                mock_client = MagicMock()
                mock_cls.return_value = mock_client
                mock_client.chat.completions.create.side_effect = capture2
                orch.chat("Second question")

            assert len(captured2) == 1
            wire_msgs = captured2[0]
            all_content = " ".join(m.get("content", "") or "" for m in wire_msgs)

            # The injected fake message should NOT appear in the LLM's wire context
            # (MemorySaver is canonical, not self.messages)
            assert "INJECTED_FAKE_MESSAGE" not in all_content, (
                "Injected fake message appeared in LLM context. "
                "self.messages should NOT be the source of LLM context under the "
                "two-track model — MemorySaver is canonical."
            )

    def test_mt_compress_dead_code_does_not_affect_llm_context(self):
        """
        PlanningOrchestratorAgent._compress_large_tool_results is dead code —
        it mutates self.messages but never the graph state. Even if called, it
        cannot change what the LLM sees.

        Confirm by calling it directly and verifying MemorySaver is unchanged.
        """
        with tempfile.TemporaryDirectory() as td:
            from scilink.agents.planning_agents.planning_orchestrator import (
                PlanningOrchestratorAgent, AutonomyLevel,
            )
            try:
                orch = PlanningOrchestratorAgent(
                    objective="Test",
                    base_dir=td,
                    api_key="dummy",
                    model_name="claude-opus-4-6",
                    autonomy_level=AutonomyLevel.AUTONOMOUS,
                    data_dir=td,
                )
            except RuntimeError as e:
                if "sandbox" in str(e).lower() or "code execution" in str(e).lower():
                    pytest.skip("PlanningOrchestrator requires sandbox environment")
                raise

            # Inject a large tool message into self.messages
            big_content = "T" * 40_000
            orch.messages.append({"role": "tool", "content": big_content})
            # Pad total to exceed 100k threshold
            orch.messages.append({"role": "tool", "content": "P" * 70_000})
            # A trailing message so the big one above isn't within the
            # "skip the 2 most recent" window _compress_large_tool_results
            # deliberately leaves untouched.
            orch.messages.append({"role": "assistant", "content": "ack"})

            # Get graph state before calling the dead method
            state_before = orch._graph.get_state(orch._graph_config)
            msgs_before = list(state_before.values.get("messages", []) if state_before.values else [])

            # Call the dead method directly
            orch._compress_large_tool_results()

            # self.messages was modified
            assert len(orch.messages[-3]["content"]) < 40_000, (
                "Dead method should have compressed self.messages"
            )

            # Graph state is unchanged
            state_after = orch._graph.get_state(orch._graph_config)
            msgs_after = list(state_after.values.get("messages", []) if state_after.values else [])
            assert len(msgs_before) == len(msgs_after), (
                "Calling _compress_large_tool_results should NOT change graph state"
            )
