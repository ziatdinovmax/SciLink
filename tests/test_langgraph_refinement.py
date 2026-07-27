"""
Unit tests for scilink.graphs.refinement — pytest-discoverable.

Covers the generic accept/refine human-feedback subgraph that replaces the
``while iteration < self.max_iterations`` loops in:

* image_analysis_controllers.py:PlanningStep.execute
* curve_fitting_controllers.py:PlanningStep.execute
* fft_microscopy_controllers.py (parameter refinement)
* sam_microscopy_controllers.py (both refinement controllers)

* build_refinement_subgraph — accept path, max-iterations path, aborted
  path, judge_fn path, optional render_fn
* sanitize_for_checkpoint    — numpy scalar -> native conversion, nested
  containers, ndarray pass-through

No live LLM calls are made; no API keys are required.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _thread_cfg():
    return {"configurable": {"thread_id": f"t-{uuid.uuid4().hex[:8]}"}}


def _base_initial(payload=None, max_iterations=3):
    return {
        "messages": [],
        "payload": payload if payload is not None else {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "action": "",
        "accepted": False,
        "locked_payload": None,
        "aborted": False,
        "history": [],
    }


# ===========================================================================
# sanitize_for_checkpoint
# ===========================================================================

class TestSanitizeForCheckpoint:
    def test_passthrough_for_plain_types(self):
        from scilink.graphs.refinement import sanitize_for_checkpoint
        assert sanitize_for_checkpoint(3) == 3
        assert sanitize_for_checkpoint(3.5) == 3.5
        assert sanitize_for_checkpoint("x") == "x"
        assert sanitize_for_checkpoint(None) is None
        assert sanitize_for_checkpoint(True) is True

    def test_converts_numpy_scalars(self):
        np = pytest.importorskip("numpy")
        from scilink.graphs.refinement import sanitize_for_checkpoint

        result = sanitize_for_checkpoint(np.float64(3.2))
        assert isinstance(result, float)
        assert not isinstance(result, np.generic)
        assert result == pytest.approx(3.2)

        result_int = sanitize_for_checkpoint(np.int64(7))
        assert isinstance(result_int, int)
        assert result_int == 7

    def test_recurses_into_nested_containers(self):
        np = pytest.importorskip("numpy")
        from scilink.graphs.refinement import sanitize_for_checkpoint

        obj = {
            "area": np.float64(12.5),
            "particles": [{"centroid": (np.float64(1.0), np.float64(2.0))}],
            "count": np.int64(3),
        }
        result = sanitize_for_checkpoint(obj)
        assert isinstance(result["area"], float)
        assert isinstance(result["particles"][0]["centroid"][0], float)
        assert isinstance(result["count"], int)

    def test_ndarray_passes_through_unconverted(self):
        np = pytest.importorskip("numpy")
        from scilink.graphs.refinement import sanitize_for_checkpoint

        arr = np.zeros((3, 3))
        result = sanitize_for_checkpoint(arr)
        assert result is arr

    def test_checkpoint_survives_numpy_scalars(self):
        """The actual bug this exists to fix: MemorySaver raises TypeError
        on raw numpy scalars nested in checkpointed state."""
        np = pytest.importorskip("numpy")
        from scilink.graphs.refinement import build_refinement_subgraph, sanitize_for_checkpoint

        def feedback_fn(state):
            return {"action": "accept"}

        def apply_fn(state):
            return {"payload": state["payload"]}

        subgraph = build_refinement_subgraph(feedback_fn=feedback_fn, apply_fn=apply_fn, max_iterations=3)
        payload = {"area": np.float64(3.2), "mask": np.zeros((2, 2), dtype=bool)}
        initial = _base_initial(payload=sanitize_for_checkpoint(payload))
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["accepted"] is True


# ===========================================================================
# build_refinement_subgraph — accept / max-iterations / aborted / judge_fn
# ===========================================================================

class TestRefinementAcceptPath:
    def test_accepts_immediately(self):
        from scilink.graphs.refinement import build_refinement_subgraph

        def feedback_fn(state):
            return {"action": "accept"}

        def apply_fn(state):
            raise AssertionError("apply_fn must not run on immediate accept")

        subgraph = build_refinement_subgraph(feedback_fn=feedback_fn, apply_fn=apply_fn, max_iterations=3)
        initial = _base_initial(payload={"x": 1})
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["accepted"] is True
        assert result["locked_payload"] == {"x": 1}
        assert result["iteration"] == 0

    def test_accepts_after_one_refine_round(self):
        from scilink.graphs.refinement import build_refinement_subgraph

        decisions = iter(["refine", "accept"])
        apply_calls = []

        def feedback_fn(state):
            return {"action": next(decisions), "payload": state["payload"]}

        def apply_fn(state):
            apply_calls.append(state["iteration"])
            return {"payload": {**state["payload"], "refined": True}}

        subgraph = build_refinement_subgraph(feedback_fn=feedback_fn, apply_fn=apply_fn, max_iterations=3)
        initial = _base_initial(payload={"x": 1})
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["accepted"] is True
        assert result["locked_payload"] == {"x": 1, "refined": True}
        assert apply_calls == [0]


class TestRefinementMaxIterations:
    def test_exhausts_after_exact_max_rounds(self):
        """Never accepts -> exactly max_iterations refine rounds, no extra
        feedback call after exhaustion (mirrors the imperative loops, which
        check the while-condition before the next round, not after)."""
        from scilink.graphs.refinement import build_refinement_subgraph

        feedback_calls = []
        apply_calls = []

        def feedback_fn(state):
            feedback_calls.append(state["iteration"])
            return {"action": "refine", "payload": state["payload"]}

        def apply_fn(state):
            apply_calls.append(state["iteration"])
            return {"payload": {**state["payload"], "round": state["iteration"] + 1}}

        subgraph = build_refinement_subgraph(feedback_fn=feedback_fn, apply_fn=apply_fn, max_iterations=3)
        initial = _base_initial(payload={"x": 1})
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["accepted"] is False
        assert result["iteration"] == 3
        assert feedback_calls == [0, 1, 2]
        assert apply_calls == [0, 1, 2]
        assert result["locked_payload"] == {"x": 1, "round": 3}


class TestRefinementAborted:
    def test_apply_fn_abort_stops_without_max_iterations_branch(self):
        """apply_fn signaling aborted=True stops the loop immediately,
        distinct from reaching max_iterations — mirrors the
        except-Exception-break path in HumanFeedbackRefinementController."""
        from scilink.graphs.refinement import build_refinement_subgraph

        def feedback_fn(state):
            return {"action": "refine", "payload": state["payload"]}

        def apply_fn(state):
            return {"payload": {**state["payload"], "current_params": {"min_area": 100}}, "aborted": True}

        subgraph = build_refinement_subgraph(feedback_fn=feedback_fn, apply_fn=apply_fn, max_iterations=5)
        initial = _base_initial(payload={"x": 1})
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["aborted"] is True
        assert result["accepted"] is False
        assert result["iteration"] == 0  # apply_feedback's normal increment is skipped on abort
        assert result["locked_payload"]["current_params"] == {"min_area": 100}


class TestRefinementJudgeFn:
    def test_judge_fn_picks_winner_on_exhaustion(self):
        from scilink.graphs.refinement import build_refinement_subgraph

        def feedback_fn(state):
            record = {"iteration": state["iteration"] + 1, "score": state["iteration"]}
            return {"action": "refine", "payload": state["payload"], "history": [record]}

        def apply_fn(state):
            return {"payload": state["payload"]}

        judge_calls = []

        def judge_fn(state, history):
            judge_calls.append(list(history))
            best = max(history, key=lambda r: r["score"])
            return {"payload": {"winner": best["iteration"]}}

        subgraph = build_refinement_subgraph(
            feedback_fn=feedback_fn, apply_fn=apply_fn, judge_fn=judge_fn, max_iterations=3
        )
        initial = _base_initial(payload={})
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["accepted"] is False
        assert result["locked_payload"] == {"winner": 3}
        assert len(result["history"]) == 3
        assert len(judge_calls) == 1
        assert len(judge_calls[0]) == 3

    def test_judge_fn_not_called_on_accept(self):
        from scilink.graphs.refinement import build_refinement_subgraph

        def feedback_fn(state):
            return {"action": "accept", "payload": state["payload"]}

        def apply_fn(state):
            raise AssertionError("apply_fn must not run")

        def judge_fn(state, history):
            raise AssertionError("judge_fn must not run on accept")

        subgraph = build_refinement_subgraph(
            feedback_fn=feedback_fn, apply_fn=apply_fn, judge_fn=judge_fn, max_iterations=3
        )
        initial = _base_initial(payload={"x": 1})
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["accepted"] is True


class TestRefinementRenderFn:
    def test_render_fn_called_once_per_round_before_feedback(self):
        from scilink.graphs.refinement import build_refinement_subgraph

        render_calls = []
        decisions = iter(["refine", "refine", "accept"])

        def render_fn(state):
            render_calls.append(state["iteration"])

        def feedback_fn(state):
            return {"action": next(decisions), "payload": state["payload"]}

        def apply_fn(state):
            return {"payload": state["payload"]}

        subgraph = build_refinement_subgraph(
            feedback_fn=feedback_fn, apply_fn=apply_fn, render_fn=render_fn, max_iterations=5
        )
        initial = _base_initial(payload={})
        result = subgraph.invoke(initial, config=_thread_cfg())
        assert result["accepted"] is True
        # One render before each of the 3 feedback rounds; numbering is
        # 1-based at the call site (state["iteration"] + 1) — here we just
        # confirm cardinality matches the number of feedback rounds.
        assert render_calls == [0, 1, 2]
