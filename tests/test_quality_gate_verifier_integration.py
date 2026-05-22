"""Regression tests: QualityGate short-circuit in `_verify_fit_with_llm`
emits the verdict-dict keys that downstream consumers actually read.

Caught on PR #193 review: my original implementation returned
`should_accept` and `issues`, but consumers at
``curve_fitting_controllers.py:2839, :2871, :3094`` read
`fit_acceptable` and `issues_found`. The mismatch meant non-R² gates
were effectively inert — a hard-reject value defaulted to
``fit_acceptable=True`` and got promoted as accepted.

These tests assert the verdict dict has the canonical keys at each of
the three branches (accept / marginal / hard-reject) of the
short-circuit, and that downstream-consumer access patterns produce
the expected control-flow outcomes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)
from scilink.agents.exp_agents.quality_gate import QualityGate


# --- Fixture helpers ----------------------------------------------------------


def _make_controller():
    """Build a minimal controller instance to invoke _verify_fit_with_llm.

    The verifier path we're testing short-circuits BEFORE the LLM call,
    so most of the controller's state is irrelevant. Use a permissive
    MagicMock for everything we don't touch.
    """
    ctrl = UnifiedSeriesProcessingController.__new__(UnifiedSeriesProcessingController)
    ctrl.logger = MagicMock()
    ctrl.model = MagicMock()
    ctrl.generation_config = MagicMock()
    ctrl.safety_settings = MagicMock()
    ctrl.parse_fn = MagicMock()
    return ctrl


def _state_with_gate(metric: str = "figure_of_merit",
                     accept: float = 0.70,
                     hard_reject: float = 0.40,
                     direction: str = "higher_is_better") -> dict:
    return {
        "quality_gate": QualityGate(
            metric=metric,
            accept_threshold=accept,
            hard_reject_threshold=hard_reject,
            direction=direction,
        ),
    }


def _fit_result_with(value: float | None, metric: str = "figure_of_merit") -> dict:
    """Build a minimal fit_result dict the verifier short-circuit will
    read via gate.extract()."""
    return {
        "fit_quality": {metric: value} if value is not None else {},
        "visualization_bytes": b"<png>",  # bypass the no-viz return None branch
        "model_type": "test",
        "parameters": {},
    }


# --- Canonical-schema regression: the three short-circuit branches ------------


def test_accept_branch_emits_fit_acceptable_true():
    """Above accept threshold → fit_acceptable=True, issues_found=[]."""
    ctrl = _make_controller()
    state = _state_with_gate()
    verdict = ctrl._verify_fit_with_llm(state, _fit_result_with(0.85))
    assert verdict is not None
    assert verdict["fit_acceptable"] is True
    assert verdict["issues_found"] == []
    assert verdict["physically_better_than_best"] is False
    # The reviewer's check that downstream consumers see the right
    # value: `verification.get("fit_acceptable", True)` reads True
    # → _was_rejected = False (correct: accept means don't reject).
    assert verdict.get("fit_acceptable", True) is True


def test_hard_reject_branch_emits_fit_acceptable_false_with_issue():
    """Below hard-reject → fit_acceptable=False, non-empty issues_found.

    This is the critical case PR #193 caught: the original code emitted
    `should_accept=False` here, but downstream read `fit_acceptable`
    which defaulted to True → the rejection was silently dropped."""
    ctrl = _make_controller()
    state = _state_with_gate()
    verdict = ctrl._verify_fit_with_llm(state, _fit_result_with(0.20))
    assert verdict is not None
    assert verdict["fit_acceptable"] is False
    assert len(verdict["issues_found"]) >= 1
    assert verdict["recommended_action"] != "none"
    # Downstream call pattern (controllers.py:2839): not get(...,True) → True
    assert not verdict.get("fit_acceptable", True)
    # Downstream call pattern (controllers.py:2871): get("issues_found", [])
    assert verdict.get("issues_found", []) != []


def test_marginal_branch_emits_fit_acceptable_true_with_caveat():
    """Between accept and hard-reject → fit_acceptable=True (don't
    trigger retry), but issues_found carries the 'marginal' note so the
    synthesis layer can qualify confidence."""
    ctrl = _make_controller()
    state = _state_with_gate()
    verdict = ctrl._verify_fit_with_llm(state, _fit_result_with(0.55))
    assert verdict is not None
    assert verdict["fit_acceptable"] is True  # don't retry on marginal
    assert "marginal" in verdict["overall_assessment"].lower()
    # Marginal carries a 'caveat' issue so the synthesis layer can
    # qualify confidence — but it does NOT trigger retry (recommended_action='none').
    assert verdict["recommended_action"] == "none"


def test_missing_metric_is_hard_reject():
    """A fit_result with no figure_of_merit at all should hard-reject —
    not silently default to accepted."""
    ctrl = _make_controller()
    state = _state_with_gate()
    # extract() returns None when the key is absent → gate.is_hard_reject(None) is True
    verdict = ctrl._verify_fit_with_llm(state, _fit_result_with(None))
    assert verdict is not None
    assert verdict["fit_acceptable"] is False


# --- Lower-is-better direction ------------------------------------------------


def test_lower_is_better_accept_branch():
    """For cost-style metrics (MIP cost, RMSE), `direction=lower_is_better`."""
    ctrl = _make_controller()
    state = _state_with_gate(
        metric="mip_cost", accept=0.25, hard_reject=0.55,
        direction="lower_is_better",
    )
    verdict = ctrl._verify_fit_with_llm(state, _fit_result_with(0.10, metric="mip_cost"))
    assert verdict["fit_acceptable"] is True


def test_lower_is_better_hard_reject_branch():
    ctrl = _make_controller()
    state = _state_with_gate(
        metric="mip_cost", accept=0.25, hard_reject=0.55,
        direction="lower_is_better",
    )
    verdict = ctrl._verify_fit_with_llm(state, _fit_result_with(0.70, metric="mip_cost"))
    assert verdict["fit_acceptable"] is False
    assert verdict["issues_found"]


# --- R² path unaffected -------------------------------------------------------


def test_r_squared_path_does_not_short_circuit():
    """When the gate's metric IS r_squared, the short-circuit is skipped
    and the existing R² verifier prompt path runs (would call the LLM —
    we don't go that far here; just confirm the short-circuit doesn't
    fire by checking that None viz returns None rather than a structured
    verdict)."""
    ctrl = _make_controller()
    state = {"quality_gate": QualityGate()}  # r_squared default
    # No viz → existing R² path returns None (line 2304)
    verdict = ctrl._verify_fit_with_llm(
        state, {"fit_quality": {"r_squared": 0.95}, "visualization_bytes": None},
    )
    assert verdict is None
