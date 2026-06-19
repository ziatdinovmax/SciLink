"""
Offline tests for parallel best-of-N anchor analysis.

Hermetic: `_execute_and_verify` is stubbed with canned results, the LLM
judge is a MagicMock returning fixed JSON. Verifies the fan-out wiring,
the quality gate, judge selection + fallbacks, winner config propagation,
and the strict N=1 / reuse-script no-op paths.
"""

import json
import logging
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
    UnifiedImageProcessingController,
)
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)
from scilink.agents.exp_agents.analysis_orchestrator_tools import (
    _resolve_n_candidates,
    _resolve_candidate_escalation,
)
from scilink.agents.exp_agents._locked_exec import atomic_np_save


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _judge_model(payload: str) -> MagicMock:
    model = MagicMock()
    resp = MagicMock()
    resp.text = payload
    model.generate_content.return_value = resp
    return model


def _parse_fn(response):
    try:
        return json.loads(response.text), None
    except Exception as e:  # pragma: no cover
        return None, {"error": str(e)}


def _controller(tmp_path, model=None,
                enable_human_feedback=False) -> UnifiedImageProcessingController:
    return UnifiedImageProcessingController(
        model=model or MagicMock(),
        logger=logging.getLogger("test_best_of_n"),
        generation_config=None,
        safety_settings=None,
        parse_fn=_parse_fn,
        executor=MagicMock(),
        script_instructions="",
        correction_instructions="",
        quality_instructions="",
        output_dir=str(tmp_path),
        image_to_bytes_fn=lambda arr: b"img",
        enable_human_feedback=enable_human_feedback,
    )


def _canned_result(score: float, approved: bool, success: bool = True,
                   tag: str = "", iterations: int = 1,
                   result_type: str = "delivered",
                   levels: list | None = None) -> dict:
    # `levels` sets the annealing level of each verification iteration — the
    # struggle signal the fast-accept gate reads. When omitted, all
    # `iterations` entries are cold (level 0).
    if levels is None:
        levels = [0] * iterations
    iters = [
        {"score": score, "annealing_level": lvl, "issues": [],
         "result_type": result_type}
        for lvl in levels
    ]
    return {
        "index": 0,
        "name": "image_0000",
        "success": success,
        "analysis_type": "test_analysis",
        "extracted_features": {"tag": tag},
        "quality_metrics": {},
        "visualization_bytes": b"\x89PNG" + tag.encode(),
        "visualization_path": f"/tmp/{tag or 'viz'}.png",
        "script": "print('hi')",
        "quality_history": {
            "final_score": score,
            "approved": approved,
            "verification_iterations": iters,
        },
    }


def _stub_execute(controller, results_by_tag: dict, record: list):
    """Install a stub _execute_and_verify returning per-candidate results."""

    def stub(state, image_data, data_path, image_name, image_idx,
             is_regime_anchor=False, reuse_script=None, reuse_source=None):
        record.append(state)
        tag = state.get("_candidate_tag", "_direct")
        result = results_by_tag[tag]
        # Simulate the QC loop refining the locked config in place.
        state["locked_analysis_config"] = {"refined_by": tag}
        return result

    controller._execute_and_verify = stub
    return stub


def _base_state(n: int) -> dict:
    return {
        "n_candidates": n,
        "locked_analysis_config": {"processing_pipeline": "original"},
        "original_image_bytes": b"\x89PNGoriginal",
    }


def _run(controller, state):
    return controller._execute_and_verify_best_of_n(
        state=state,
        image_data=np.zeros((4, 4)),
        data_path="img.npy",
        image_name="image_0000",
        image_idx=0,
    )


# ---------------------------------------------------------------------------
# Fan-out mechanics
# ---------------------------------------------------------------------------

def test_fanout_launches_n_isolated_attempts(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
        "cand_02": _canned_result(0.7, True, tag="cand_02"),
    }, seen)

    state = _base_state(3)
    result = _run(c, state)

    assert len(seen) == 3
    subdirs = {s["_candidate_subdir"] for s in seen}
    tags = {s["_candidate_tag"] for s in seen}
    assert subdirs == {f"_candidates/cand_{i:02d}" for i in range(3)}
    assert tags == {f"cand_{i:02d}" for i in range(3)}
    assert all(s["_suppress_human_feedback"] for s in seen)
    # Each attempt got its own config object (deep-copied), not the original.
    configs = [id(s["locked_analysis_config"]) for s in seen]
    assert len(set(configs)) == 3
    assert result["anchor_candidates"]
    assert len(result["anchor_candidates"]) == 3


def test_n1_is_strict_noop(tmp_path):
    c = _controller(tmp_path)
    seen = []
    _stub_execute(c, {"_direct": _canned_result(0.9, True, tag="direct")}, seen)

    state = _base_state(1)
    result = _run(c, state)

    assert len(seen) == 1
    # State passed by identity: no copy, no candidate keys injected.
    assert seen[0] is state
    assert "_candidate_subdir" not in state
    assert "anchor_candidates" not in result


def test_reuse_script_bypasses_fanout(tmp_path):
    c = _controller(tmp_path)
    seen = []

    def stub(state, image_data, data_path, image_name, image_idx,
             is_regime_anchor=False, reuse_script=None, reuse_source=None):
        seen.append((state, reuse_script))
        return _canned_result(0.9, True, tag="reuse")

    c._execute_and_verify = stub
    state = _base_state(3)
    result = c._execute_and_verify_best_of_n(
        state=state, image_data=np.zeros((4, 4)), data_path="img.npy",
        image_name="image_0000", image_idx=0,
        reuse_script="print('prior')", reuse_source="prior_run",
    )

    assert len(seen) == 1
    assert seen[0][0] is state
    assert seen[0][1] == "print('prior')"
    assert "anchor_candidates" not in result


def test_auxiliary_operands_no_longer_force_single_attempt(tmp_path):
    # v2: aux staging is atomic (atomic_np_save), so aux data fans out too.
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
        "cand_02": _canned_result(0.7, True, tag="cand_02"),
    }, seen)

    state = _base_state(3)
    state["auxiliary_items"] = [{"label": "I0", "path": "ref.npy"}]
    result = _run(c, state)

    assert len(seen) == 3
    assert len(result["anchor_candidates"]) == 3


# ---------------------------------------------------------------------------
# Gating + judge selection
# ---------------------------------------------------------------------------

def test_gate_drops_unapproved_before_judge(tmp_path):
    # Judge picks survivor index 1; the unapproved candidate must not be
    # part of the survivor list the index refers to.
    model = _judge_model('{"selected_index": 1, "reasoning": "cleaner"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.95, False, tag="cand_00"),  # unapproved
        "cand_01": _canned_result(0.80, True, tag="cand_01"),
        "cand_02": _canned_result(0.75, True, tag="cand_02"),
    }, [])

    result = _run(c, _base_state(3))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert len(selected) == 1
    # Survivors (sorted by attempt) are cand_01, cand_02 -> index 1 = cand_02.
    assert selected[0]["attempt"] == 2
    assert result["extracted_features"]["tag"] == "cand_02"
    assert result["anchor_judge"]["reasoning"] == "cleaner"
    assert result["anchor_judge"]["fallback"] is False


def test_single_survivor_skips_judge(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.5, False, tag="cand_01"),
    }, [])

    state = _base_state(2)
    result = _run(c, state)

    model.generate_content.assert_not_called()
    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 0


def test_judge_garbage_falls_back_to_argmax_score(tmp_path):
    model = _judge_model("not json at all {{{")
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.81, True, tag="cand_00"),
        "cand_01": _canned_result(0.92, True, tag="cand_01"),
    }, [])

    result = _run(c, _base_state(2))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 1  # highest score
    assert result["anchor_judge"]["fallback"] is True


def test_judge_out_of_range_index_falls_back(tmp_path):
    model = _judge_model('{"selected_index": 7, "reasoning": "??"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.7, True, tag="cand_01"),
    }, [])

    result = _run(c, _base_state(2))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 0
    assert result["anchor_judge"]["fallback"] is True


def test_all_unapproved_keeps_best_score(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    low = _canned_result(0.45, False, tag="cand_00")
    low["quality_warning"] = "below threshold"
    _stub_execute(c, {
        "cand_00": low,
        "cand_01": _canned_result(0.30, False, tag="cand_01"),
    }, [])

    result = _run(c, _base_state(2))

    model.generate_content.assert_not_called()
    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 0
    assert result["quality_warning"] == "below threshold"
    assert result["anchor_judge"]["fallback"] is True


def test_all_failed_returns_failure(tmp_path):
    c = _controller(tmp_path)
    fail = _canned_result(0.0, False, success=False, tag="cand_00")
    fail2 = _canned_result(0.0, False, success=False, tag="cand_01")
    _stub_execute(c, {"cand_00": fail, "cand_01": fail2}, [])

    result = _run(c, _base_state(2))

    assert result["success"] is False


def test_winner_config_propagates_to_outer_state(tmp_path):
    model = _judge_model('{"selected_index": 1, "reasoning": "better"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
    }, [])

    state = _base_state(2)
    _run(c, state)

    # Stub stamps each attempt's refined config; the winner's must win.
    assert state["locked_analysis_config"] == {"refined_by": "cand_01"}


def test_judge_evidence_includes_all_survivor_visualizations(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
        "cand_02": _canned_result(0.7, True, tag="cand_02"),
    }, [])

    _run(c, _base_state(3))

    (_, kwargs) = model.generate_content.call_args
    parts = kwargs.get("contents") or model.generate_content.call_args[0][0]
    image_parts = [p for p in parts if isinstance(p, dict)]
    # original + 3 candidate visualizations
    assert len(image_parts) == 4
    text = parts[0]
    assert "Candidate 0" in text and "Candidate 2" in text


# ---------------------------------------------------------------------------
# Curve fitting: same wrapper, R²-gated
# ---------------------------------------------------------------------------

def _curve_controller(tmp_path, model=None,
                      enable_human_feedback=False) -> UnifiedSeriesProcessingController:
    return UnifiedSeriesProcessingController(
        model=model or MagicMock(),
        logger=logging.getLogger("test_best_of_n_curve"),
        generation_config=None,
        safety_settings=None,
        parse_fn=_parse_fn,
        executor=MagicMock(),
        script_instructions="",
        correction_instructions="",
        quality_instructions="",
        output_dir=str(tmp_path),
        plot_fn=lambda data, info: b"plot",
        enable_human_feedback=enable_human_feedback,
    )


def _canned_fit(r2: float, approved: bool, success: bool = True,
                tag: str = "", iterations: int = 1,
                levels: list | None = None,
                fit_quality: dict | None = None) -> dict:
    # `levels` sets the annealing level of each verification iteration (the
    # struggle signal the fast-accept gate reads). When omitted, all
    # `iterations` entries are cold (level 0). `fit_quality` overrides the
    # metric dict the gate reads (for non-R² gate tests).
    if levels is None:
        levels = [0] * iterations
    iters = [
        {"r_squared": r2, "annealing_level": lvl, "issues": []}
        for lvl in levels
    ]
    return {
        "index": 0,
        "name": "spectrum_0000",
        "success": success,
        "model_type": "exp_decay",
        "parameters": {"tau": 1.0},
        "fit_quality": fit_quality if fit_quality is not None else {"r_squared": r2},
        "visualization_bytes": b"\x89PNG" + tag.encode(),
        "visualization_path": f"/tmp/{tag or 'fit'}.png",
        "script": "print('fit')",
        "quality_history": {
            "final_r2": r2,
            "approved": approved,
            "verification_iterations": iters,
        },
    }


def _stub_fit(controller, results_by_tag: dict, record: list):
    def stub(state, curve_data, data_path, spectrum_name, spectrum_idx,
             is_regime_anchor=False, reuse_script=None, reuse_source=None):
        record.append(state)
        tag = state.get("_candidate_tag", "_direct")
        state["locked_fitting_config"] = {"refined_by": tag}
        return results_by_tag[tag]

    controller._fit_with_quality_control = stub
    return stub


class _StubReplanner:
    """Records which candidates were independently replanned."""

    def __init__(self, calls: list):
        self.calls = calls

    def replan_headless(self, state: dict) -> dict:
        self.calls.append(state.get("_candidate_tag"))
        return state


def _curve_state(n: int) -> dict:
    return {
        "n_candidates": n,
        "locked_fitting_config": {"physical_model": "original"},
        "original_plot_bytes": b"\x89PNGoriginal",
    }


def _run_curve(controller, state, **kwargs):
    return controller._fit_with_quality_control_best_of_n(
        state=state,
        curve_data=np.zeros(64),
        data_path="spec.npy",
        spectrum_name="spectrum_0000",
        spectrum_idx=0,
        **kwargs,
    )


def test_curve_fanout_launches_n_isolated_attempts(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00"),
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
        "cand_02": _canned_fit(0.97, True, tag="cand_02"),
    }, seen)

    state = _curve_state(3)
    result = _run_curve(c, state)

    assert len(seen) == 3
    assert {s["_candidate_subdir"] for s in seen} == {
        f"_candidates/cand_{i:02d}" for i in range(3)
    }
    assert all(s["_suppress_human_feedback"] for s in seen)
    configs = [id(s["locked_fitting_config"]) for s in seen]
    assert len(set(configs)) == 3
    assert len(result["anchor_candidates"]) == 3
    assert sum(1 for r in result["anchor_candidates"] if r["selected"]) == 1


def test_curve_n1_is_strict_noop(tmp_path):
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {"_direct": _canned_fit(0.99, True, tag="direct")}, seen)

    state = _curve_state(1)
    result = _run_curve(c, state)

    assert len(seen) == 1
    assert seen[0] is state
    assert "anchor_candidates" not in result


def test_curve_reuse_script_bypasses_fanout(tmp_path):
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {"_direct": _canned_fit(0.99, True, tag="direct")}, seen)

    state = _curve_state(3)
    result = _run_curve(c, state, reuse_script="print('prior')",
                        reuse_source="prior_run")

    assert len(seen) == 1
    assert seen[0] is state
    assert "anchor_candidates" not in result


def test_curve_auxiliary_operands_no_longer_force_single_attempt(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00"),
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
        "cand_02": _canned_fit(0.97, True, tag="cand_02"),
    }, seen)

    state = _curve_state(3)
    state["auxiliary_items"] = [{"label": "I0", "path": "ref.npy"}]
    result = _run_curve(c, state)

    assert len(seen) == 3
    assert len(result["anchor_candidates"]) == 3


def test_curve_gate_drops_unapproved_then_judge_picks(tmp_path):
    model = _judge_model('{"selected_index": 1, "reasoning": "more physical"}')
    c = _curve_controller(tmp_path, model)
    _stub_fit(c, {
        "cand_00": _canned_fit(0.999, False, tag="cand_00"),  # unapproved
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
        "cand_02": _canned_fit(0.97, True, tag="cand_02"),
    }, [])

    result = _run_curve(c, _curve_state(3))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    # Survivors are cand_01, cand_02 -> judge index 1 = cand_02.
    assert selected[0]["attempt"] == 2
    assert result["anchor_judge"]["reasoning"] == "more physical"


def test_curve_judge_garbage_falls_back_to_argmax_r2(tmp_path):
    model = _judge_model("not json")
    c = _curve_controller(tmp_path, model)
    _stub_fit(c, {
        "cand_00": _canned_fit(0.97, True, tag="cand_00"),
        "cand_01": _canned_fit(0.99, True, tag="cand_01"),
    }, [])

    result = _run_curve(c, _curve_state(2))

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 1
    assert result["anchor_judge"]["fallback"] is True


def test_curve_winner_config_propagates(tmp_path):
    model = _judge_model('{"selected_index": 1, "reasoning": "better"}')
    c = _curve_controller(tmp_path, model)
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00"),
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
    }, [])

    state = _curve_state(2)
    _run_curve(c, state)

    assert state["locked_fitting_config"] == {"refined_by": "cand_01"}


def test_fanout_persists_attempt_result_snapshots(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
    }, [])

    _run(c, _base_state(2))

    for i in range(2):
        snap = (tmp_path / "image_0000" / "_candidates" / f"cand_{i:02d}"
                / "attempt_result.json")
        assert snap.exists()
        data = json.loads(snap.read_text())
        assert data["attempt"] == i
        assert data["extracted_features"] == {"tag": f"cand_{i:02d}"}


# ---------------------------------------------------------------------------
# Escalation (escalate-on-weak-first)
# ---------------------------------------------------------------------------

def _esc_state(n: int) -> dict:
    state = _base_state(n)
    state["candidate_escalation"] = True
    return state


def test_escalation_fast_accepts_strong_first_attempt(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    seen = []
    # 0.85 >= 0.7 + 0.05 backstop, never went hot (level 0) -> fast accept
    _stub_execute(c, {
        "cand_00": _canned_result(0.85, True, tag="cand_00"),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 1
    model.generate_content.assert_not_called()
    assert len(result["anchor_candidates"]) == 1
    assert result["anchor_judge"]["escalated"] is False
    assert "no escalation" in result["anchor_judge"]["reasoning"]


def test_escalation_weak_score_fans_out(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    # 0.72 approved but below the 0.75 backstop -> escalate.
    _stub_execute(c, {
        "cand_00": _canned_result(0.72, True, tag="cand_00", levels=[0, 0]),
        "cand_01": _canned_result(0.85, True, tag="cand_01"),
        "cand_02": _canned_result(0.80, True, tag="cand_02"),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True
    # Judge sees all three (attempt 0 included).
    assert len(result["anchor_candidates"]) == 3


def test_escalation_failed_first_fans_out(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.0, False, success=False, tag="cand_00"),
        "cand_01": _canned_result(0.85, True, tag="cand_01"),
        "cand_02": _canned_result(0.80, True, tag="cand_02"),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True


def test_escalation_decline_never_fast_accepts(tmp_path):
    # #289 interaction: a rigorous null/decline scores high and converges
    # fast — both fast-accept criteria are biased toward it. It must always
    # escalate so the judge compares it against delivered attempts.
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.95, True, tag="cand_00",
                                  result_type="null_decline"),
        "cand_01": _canned_result(0.85, True, tag="cand_01"),
        "cand_02": _canned_result(0.80, True, tag="cand_02"),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True


def test_escalation_good_score_cold_fast_accepts(tmp_path):
    # Redesign: a genuinely good first attempt (0.80, approved) that reached
    # threshold WITHOUT going hot fast-accepts, even over a few iterations.
    # This is the exact case the old iters<=2 / margin-0.15 gate over-
    # escalated — a 0.80/level-1 run fanned out 3x and still won (the live
    # session that triggered this redesign).
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.80, True, tag="cand_00", levels=[0, 0, 1]),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 1
    model.generate_content.assert_not_called()
    assert result["anchor_judge"]["escalated"] is False


def test_escalation_high_score_many_cold_iters_fast_accepts(tmp_path):
    # Redesign: iteration COUNT no longer gates. A high score reached over
    # several iterations that never left the cold/warm regime is not a
    # struggle -> fast-accept.
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00", levels=[0, 0, 1]),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 1
    model.generate_content.assert_not_called()
    assert result["anchor_judge"]["escalated"] is False


def test_escalation_hot_anneal_fans_out(tmp_path):
    # The struggle signal: even a strong score escalates if the loop had to go
    # HOT (fully relax plan constraints) to reach it — that's not a clean win.
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00", levels=[0, 1, 2]),
        "cand_01": _canned_result(0.85, True, tag="cand_01"),
        "cand_02": _canned_result(0.80, True, tag="cand_02"),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True
    # No annealing-level seeding any more (superseded by independent per-
    # candidate plans): fan-out attempts use the caller's default start.
    by_tag = {s["_candidate_tag"]: s for s in seen}
    assert all(
        s.get("_starting_annealing_level") is None for s in by_tag.values()
    )


def test_escalation_started_hot_is_not_struggle(tmp_path):
    # A fit that STARTED at hot (a caller / re-run set the starting level) and
    # stayed there did NOT climb under stall — it's a clean win, so fast-accept
    # with no escalation. Only CLIMBING into hot (start < hot) counts.
    model = _judge_model('{"selected_index": 0, "reasoning": "unused"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00", levels=[2, 2, 2]),
    }, seen)

    result = _run(c, _esc_state(3))

    assert len(seen) == 1
    assert result["anchor_judge"]["escalated"] is False


def test_no_escalation_flag_keeps_fixed_n(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.95, True, tag="cand_00"),
        "cand_01": _canned_result(0.85, True, tag="cand_01"),
        "cand_02": _canned_result(0.80, True, tag="cand_02"),
    }, seen)

    result = _run(c, _base_state(3))  # no candidate_escalation

    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is False


def test_curve_escalation_fast_accept_and_escalate(tmp_path):
    # fast_thr = 0.95 + min(0.02, 0.025) = 0.97
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {"cand_00": _canned_fit(0.98, True, tag="cand_00")}, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True
    result = _run_curve(c, state)
    assert len(seen) == 1
    assert result["anchor_judge"]["escalated"] is False

    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c2 = _curve_controller(tmp_path, model)
    seen2 = []
    _stub_fit(c2, {
        "cand_00": _canned_fit(0.955, True, tag="cand_00"),  # < 0.97
        "cand_01": _canned_fit(0.99, True, tag="cand_01"),
        "cand_02": _canned_fit(0.98, True, tag="cand_02"),
    }, seen2)
    state2 = _curve_state(3)
    state2["candidate_escalation"] = True
    result2 = _run_curve(c2, state2)
    assert len(seen2) == 3
    assert result2["anchor_judge"]["escalated"] is True


def test_curve_escalation_high_r2_many_iters_fast_accepts(tmp_path):
    # Gate redesign: iteration COUNT no longer gates. A fit that cleared the
    # bar (0.99 >= 0.97), approved, and never went hot fast-accepts no matter
    # how many iterations it took — the old `iterations <= 2` proxy would have
    # escalated this good-but-slow fit.
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00", levels=[0, 0, 0, 0, 0]),
    }, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True
    result = _run_curve(c, state)
    assert len(seen) == 1
    assert result["anchor_judge"]["escalated"] is False


def _chi2_gate():
    from scilink.agents.exp_agents.quality_gate import QualityGate
    # lower_is_better: accept χ² <= 1.5, hard-reject > 3.0, optimum 0.0.
    # band 1.5 -> fast margin 0.4*1.5 = 0.6 -> fast bar χ² <= 0.9.
    return QualityGate(metric="reduced_chi2", accept_threshold=1.5,
                       hard_reject_threshold=3.0, direction="lower_is_better",
                       best_value=0.0)


def test_curve_escalation_lower_is_better_metric_fast_accepts(tmp_path):
    # χ² gate: a comfortably-low χ² (0.8 <= 0.9) approved fit fast-accepts —
    # the gate reads χ² from fit_quality, not R².
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.5, True, tag="cand_00",
                               fit_quality={"reduced_chi2": 0.8, "r_squared": 0.5}),
    }, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True
    state["quality_gate"] = _chi2_gate()
    result = _run_curve(c, state)
    assert len(seen) == 1
    assert result["anchor_judge"]["escalated"] is False


def test_curve_escalation_metric_aware_escalates_on_high_r2_bad_chi2(tmp_path):
    # THE bug fix: R²=0.99 is high (old R²-only gate would have fast-accepted),
    # but the active metric is χ²=1.4 — approved (<=1.5) yet above the χ² fast
    # bar (0.9). The metric-aware gate correctly escalates on the real metric.
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00",
                               fit_quality={"reduced_chi2": 1.4, "r_squared": 0.99}),
        "cand_01": _canned_fit(0.99, True, tag="cand_01",
                               fit_quality={"reduced_chi2": 0.7, "r_squared": 0.99}),
        "cand_02": _canned_fit(0.99, True, tag="cand_02",
                               fit_quality={"reduced_chi2": 0.8, "r_squared": 0.99}),
    }, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True
    state["quality_gate"] = _chi2_gate()
    result = _run_curve(c, state)
    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True


def test_curve_escalation_hot_fans_out_despite_high_r2(tmp_path):
    # The struggle signal guards the SUBJECTIVE (physical-correctness) half of
    # approval: even a high, approved R² escalates if the fit had to anneal HOT
    # (full model freedom) to get there — a hot-annealed over-fit can post high
    # R² yet be physically wrong, so don't fast-accept it.
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00", levels=[0, 1, 2]),
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
        "cand_02": _canned_fit(0.985, True, tag="cand_02"),
    }, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True
    result = _run_curve(c, state)
    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True


def test_curve_escalation_started_hot_is_not_struggle(tmp_path):
    # A high-R² fit that STARTED at hot (caller / re-run set T=2) and stayed
    # there is a clean win, not a struggle -> fast-accept, no escalation.
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00", levels=[2, 2, 2]),
    }, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True
    result = _run_curve(c, state)
    assert len(seen) == 1
    assert result["anchor_judge"]["escalated"] is False


# ---------------------------------------------------------------------------
# Join approval (CO_PILOT/AUTOPILOT)
# ---------------------------------------------------------------------------

def _patch_input(monkeypatch, responses, observer=None):
    it = iter(responses)

    def fake_input(prompt=""):
        if observer:
            observer(prompt)
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)


def test_join_approval_enter_accepts_and_cleans_reviews(tmp_path, monkeypatch):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model, enable_human_feedback=True)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
    }, [])

    reviews_at_prompt = []
    prompts = []

    def observer(prompt):
        prompts.append(prompt)
        reviews_at_prompt.extend(tmp_path.glob("bestofn_candidate_*_review.png"))

    _patch_input(monkeypatch, [""], observer)
    result = _run(c, _base_state(2))

    # Review pngs existed at prompt time (UI scan filter matches "review")
    assert len(reviews_at_prompt) == 2
    assert all("review" in p.stem for p in reviews_at_prompt)
    # The input() prompt itself names the default (drives CLI/scripted use).
    assert any("accept candidate 0" in p for p in prompts)
    # ... and are cleaned up afterwards.
    assert not list(tmp_path.glob("bestofn_candidate_*_review.png"))
    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 0
    assert result["anchor_judge"]["human_override"] is False


def test_join_approval_digit_overrides_winner(tmp_path, monkeypatch):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model, enable_human_feedback=True)
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
        "cand_02": _canned_result(0.7, True, tag="cand_02"),
    }, [])

    _patch_input(monkeypatch, ["2"])
    state = _base_state(3)
    result = _run(c, state)

    selected = [r for r in result["anchor_candidates"] if r["selected"]]
    assert selected[0]["attempt"] == 2
    assert result["extracted_features"]["tag"] == "cand_02"
    # Overridden candidate's refined config propagates.
    assert state["locked_analysis_config"] == {"refined_by": "cand_02"}
    assert result["anchor_judge"]["human_override"] is True


def test_join_approval_more_triggers_escalation(tmp_path, monkeypatch):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model, enable_human_feedback=True)
    seen = []
    _stub_execute(c, {
        "cand_00": _canned_result(0.85, True, tag="cand_00"),  # fast-accept
        "cand_01": _canned_result(0.9, True, tag="cand_01"),
        "cand_02": _canned_result(0.8, True, tag="cand_02"),
    }, seen)

    _patch_input(monkeypatch, ["more", ""])
    result = _run(c, _esc_state(3))

    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True
    assert len(result["anchor_candidates"]) == 3


def test_join_approval_not_fired_when_feedback_disabled(tmp_path, monkeypatch):
    def explode(prompt=""):
        raise AssertionError("input() must not be called")

    monkeypatch.setattr("builtins.input", explode)

    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)  # feedback disabled
    _stub_execute(c, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
    }, [])
    _run(c, _base_state(2))

    # ... and suppressed even when enabled (nested best-of-N context).
    c2 = _controller(tmp_path, model, enable_human_feedback=True)
    _stub_execute(c2, {
        "cand_00": _canned_result(0.9, True, tag="cand_00"),
        "cand_01": _canned_result(0.8, True, tag="cand_01"),
    }, [])
    state = _base_state(2)
    state["_suppress_human_feedback"] = True
    _run(c2, state)


# ---------------------------------------------------------------------------
# Worker log attribution (UI verbose panel + CLI prefixes)
# ---------------------------------------------------------------------------

def test_log_context_attribution_and_prefix(tmp_path):
    import threading as th
    from scilink.utils.log_context import (
        register_worker, unregister_worker, effective_thread,
    )

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("test_log_ctx")
    logger.setLevel(logging.INFO)
    handler = Capture()
    logger.addHandler(handler)
    parent = th.get_ident()
    seen = {}

    def worker():
        register_worker(parent, "cand_07")
        try:
            seen["effective"] = effective_thread(th.get_ident())
            logger.info("Verification 2/7 (annealing level 0)...")
        finally:
            unregister_worker()
        logger.info("after unregister")

    t = th.Thread(target=worker)
    t.start(); t.join()
    logger.removeHandler(handler)

    # Worker records map back to the parent thread for UI filtering...
    assert seen["effective"] == parent
    # ...and are prefixed with the candidate tag; post-unregister ones aren't.
    assert records[0] == "[cand_07] Verification 2/7 (annealing level 0)..."
    assert records[1] == "after unregister"


def test_fanout_workers_are_log_registered(tmp_path):
    import threading as th
    from scilink.utils.log_context import effective_thread

    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    parent = th.get_ident()
    attribution = []

    def stub(state, image_data, data_path, image_name, image_idx,
             is_regime_anchor=False, reuse_script=None, reuse_source=None):
        attribution.append(effective_thread(th.get_ident()) == parent)
        tag = state.get("_candidate_tag", "_direct")
        state["locked_analysis_config"] = {"refined_by": tag}
        return _canned_result(0.9, True, tag=tag)

    c._execute_and_verify = stub
    state = _base_state(2)
    # stub keyed results irrelevant; both attempts return tag-stamped results
    _run(c, state)

    assert attribution == [True, True]


# ---------------------------------------------------------------------------
# atomic_np_save
# ---------------------------------------------------------------------------

def test_atomic_np_save_roundtrip_and_overwrite(tmp_path):
    p = tmp_path / "aux.npy"
    a = np.arange(12).reshape(3, 4)
    atomic_np_save(p, a)
    assert np.array_equal(np.load(p), a)
    b = np.ones((2, 2))
    atomic_np_save(p, b)
    assert np.array_equal(np.load(p), b)
    assert not list(tmp_path.glob("*.tmp.npy"))


def test_atomic_np_save_concurrent_hammer(tmp_path):
    from concurrent.futures import ThreadPoolExecutor as TPE

    p = tmp_path / "aux.npy"
    arr = np.random.default_rng(0).normal(size=(256, 256))

    def writer(_):
        for _ in range(5):
            atomic_np_save(p, arr)
            assert np.array_equal(np.load(p), arr)

    with TPE(max_workers=8) as pool:
        list(pool.map(writer, range(8)))
    assert np.array_equal(np.load(p), arr)
    assert not list(tmp_path.glob("*.tmp.npy"))


# ---------------------------------------------------------------------------
# Tool-side default resolution
# ---------------------------------------------------------------------------

class ImageAnalysisAgent:  # name drives the default lookup
    def analyze(self, data, n_candidates: int = 1,
                candidate_escalation: bool = False):
        pass


class CurveFittingAgent:
    def analyze(self, data, n_candidates: int = 1,
                candidate_escalation: bool = False):
        pass


class _NoSupportAgent:
    def analyze(self, data):
        pass


def test_resolve_default_image_agent_gets_three():
    assert _resolve_n_candidates(ImageAnalysisAgent(), None) == 3


def test_resolve_default_curve_agent_gets_three():
    # Curve now auto-escalates by default (skill-gated in the agent).
    assert _resolve_n_candidates(CurveFittingAgent(), None) == 3


def test_resolve_default_unsupporting_agent_gets_one():
    assert _resolve_n_candidates(_NoSupportAgent(), None) is None


def test_resolve_explicit_request_wins():
    assert _resolve_n_candidates(ImageAnalysisAgent(), 5) == 5
    assert _resolve_n_candidates(ImageAnalysisAgent(), 1) == 1


def test_resolve_unsupported_agent_returns_none():
    assert _resolve_n_candidates(_NoSupportAgent(), 3) is None
    assert _resolve_n_candidates(_NoSupportAgent(), None) is None


def test_resolve_escalation_default_path_only():
    # Default path (no explicit n) -> escalation on.
    assert _resolve_candidate_escalation(ImageAnalysisAgent(), None) is True
    # Explicit n (any value) -> exact N, no escalation.
    assert _resolve_candidate_escalation(ImageAnalysisAgent(), 3) is False
    assert _resolve_candidate_escalation(ImageAnalysisAgent(), 1) is False
    # Agent without the param -> never.
    assert _resolve_candidate_escalation(_NoSupportAgent(), None) is False


# ---------------------------------------------------------------------------
# Per-candidate independent planning (ensemble diversity)
# ---------------------------------------------------------------------------

def test_image_fanout_replans_candidates_above_zero(tmp_path):
    # Candidate 0 keeps the primary plan; candidates >=1 each replan.
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    calls = []
    c.replanner = _StubReplanner(calls)
    seen = []
    _stub_execute(c, {
        f"cand_0{i}": _canned_result(0.8, True, tag=f"cand_0{i}")
        for i in range(3)
    }, seen)

    _run(c, _base_state(3))  # fixed-N: all three run

    assert sorted(calls) == ["cand_01", "cand_02"]


def test_image_fanout_replan_disabled_by_flag(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _controller(tmp_path, model)
    calls = []
    c.replanner = _StubReplanner(calls)
    seen = []
    _stub_execute(c, {
        f"cand_0{i}": _canned_result(0.8, True, tag=f"cand_0{i}")
        for i in range(3)
    }, seen)

    state = _base_state(3)
    state["independent_candidate_plans"] = False
    _run(c, state)

    assert calls == []


def test_curve_fanout_replans_candidates_above_zero(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    calls = []
    c.replanner = _StubReplanner(calls)
    seen = []
    _stub_fit(c, {
        f"cand_0{i}": _canned_fit(0.99, True, tag=f"cand_0{i}")
        for i in range(3)
    }, seen)

    _run_curve(c, _curve_state(3))

    assert sorted(calls) == ["cand_01", "cand_02"]


def test_curve_fanout_replan_disabled_by_flag(tmp_path):
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    calls = []
    c.replanner = _StubReplanner(calls)
    seen = []
    _stub_fit(c, {
        f"cand_0{i}": _canned_fit(0.99, True, tag=f"cand_0{i}")
        for i in range(3)
    }, seen)

    state = _curve_state(3)
    state["independent_candidate_plans"] = False
    _run_curve(c, state)

    assert calls == []


# ---------------------------------------------------------------------------
# Skill-gated curve auto-escalation
# ---------------------------------------------------------------------------

def test_curve_auto_escalation_suppressed_when_skill_active(tmp_path):
    # Auto-default (candidate_escalation=True) + a loaded skill -> single
    # skill-guided fit; the fan-out is suppressed (candidates would converge).
    c = _curve_controller(tmp_path)
    seen = []
    _stub_fit(c, {"_direct": _canned_fit(0.99, True, tag="x")}, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True
    state["skill_name"] = "xps"
    result = _run_curve(c, state)
    assert len(seen) == 1
    assert seen[0].get("_candidate_tag") is None   # single path
    assert "anchor_candidates" not in result


def test_curve_auto_escalation_fires_when_no_skill(tmp_path):
    # Auto-default + NO skill -> escalation engages; a weak attempt 0 fans out.
    model = _judge_model('{"selected_index": 1, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.90, True, tag="cand_00"),  # weak -> escalate
        "cand_01": _canned_fit(0.99, True, tag="cand_01"),
        "cand_02": _canned_fit(0.98, True, tag="cand_02"),
    }, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = True   # no skill_name set
    result = _run_curve(c, state)
    assert len(seen) == 3
    assert result["anchor_judge"]["escalated"] is True


def test_curve_explicit_count_honored_despite_skill(tmp_path):
    # Explicit "run 3" (candidate_escalation=False) is honored even with a
    # skill active -> fan-out is NOT suppressed.
    model = _judge_model('{"selected_index": 0, "reasoning": "ok"}')
    c = _curve_controller(tmp_path, model)
    seen = []
    _stub_fit(c, {
        "cand_00": _canned_fit(0.99, True, tag="cand_00"),
        "cand_01": _canned_fit(0.98, True, tag="cand_01"),
        "cand_02": _canned_fit(0.97, True, tag="cand_02"),
    }, seen)
    state = _curve_state(3)
    state["candidate_escalation"] = False
    state["skill_name"] = "xps"
    result = _run_curve(c, state)
    assert len(seen) == 3
