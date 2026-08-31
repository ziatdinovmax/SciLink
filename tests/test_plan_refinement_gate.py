"""Tests for the shared plan-refinement gate

The human plan-approval loop was duplicated byte-for-byte in the curve
fitting and image analysis planning controllers (same guard, same
``while iteration < max_iterations``, same ``_refine_requested`` /
``_refine_feedback`` pops); the verification loop was already shared via
``_qc_engine.py`` and this was the leftover. It now lives once in
``base_controllers.run_plan_refinement_gate``.

  python -m pytest tests/test_plan_refinement_gate.py -v
"""
import logging
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

from scilink.agents.exp_agents.controllers.base_controllers import (
    run_plan_refinement_gate)


class FakeController:
    """Feeds a scripted sequence of user decisions to the gate.

    Each element of ``decisions`` is a feedback string (= "refine with
    this") or None (= approve). The list running out counts as approval.
    """

    def __init__(self, decisions, max_iterations=3, enable=True):
        self.enable_human_feedback = enable
        self.max_iterations = max_iterations
        self.logger = logging.getLogger("fake-controller")
        self._decisions = list(decisions)
        self.feedback_calls = 0
        self.refined_with = []

    def _get_human_feedback(self, state):
        self.feedback_calls += 1
        d = self._decisions.pop(0) if self._decisions else None
        if d is not None:
            state["_refine_requested"] = True
            state["_refine_feedback"] = d
        return state

    def _refine_plan(self, state, feedback):
        self.refined_with.append(feedback)
        return state


def _gate(ctrl, state):
    return run_plan_refinement_gate(
        ctrl, state, refine_banner="\nRefining plan...\n",
        max_banner="Max refinements reached.")


def test_immediate_approval_runs_no_refinement():
    ctrl = FakeController([None])
    state = _gate(ctrl, {})
    assert ctrl.feedback_calls == 1
    assert ctrl.refined_with == []
    assert "_refine_requested" not in state


def test_refine_until_approved_logs_feedback():
    ctrl = FakeController(["sharper peaks", "wider window", None])
    state = _gate(ctrl, {})
    assert ctrl.refined_with == ["sharper peaks", "wider window"]
    assert state["human_feedback_log"] == ["sharper peaks", "wider window"]
    # The transient markers never leak into the state.
    assert "_refine_requested" not in state
    assert "_refine_feedback" not in state


def test_max_iterations_caps_the_loop(capsys):
    ctrl = FakeController(["a", "b", "c", "d"], max_iterations=2)
    _gate(ctrl, {})
    assert ctrl.refined_with == ["a", "b"]
    assert "Max refinements reached." in capsys.readouterr().out


def test_disabled_feedback_skips_gate():
    ctrl = FakeController(["never asked"], enable=False)
    _gate(ctrl, {})
    assert ctrl.feedback_calls == 0


def test_locked_script_reuse_skips_gate():
    ctrl = FakeController(["never asked"])
    _gate(ctrl, {"reuse_locked_script": True})
    assert ctrl.feedback_calls == 0


def test_both_controllers_use_the_shared_gate():
    """The extraction is only done if both former copies now import it."""
    from scilink.agents.exp_agents.controllers import (
        base_controllers, curve_fitting_controllers, image_analysis_controllers)
    assert (curve_fitting_controllers.run_plan_refinement_gate
            is base_controllers.run_plan_refinement_gate)
    assert (image_analysis_controllers.run_plan_refinement_gate
            is base_controllers.run_plan_refinement_gate)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
