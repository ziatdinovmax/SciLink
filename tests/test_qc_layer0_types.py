"""Unit tests for the Layer-0 QC types (issue #327 phase 1).

Covers: QualityGate.value_source + verdict extraction + IMAGE_SCORE_DEFAULT,
arithmetic equivalence of gate routing (incl. post-adjust_threshold
mutation), the shared verification-record builders, CritiquePayload, and
QCProfile presets.
"""

import math

import pytest

from scilink.agents.exp_agents.quality_gate import (
    IMAGE_SCORE_DEFAULT,
    R_SQUARED_DEFAULT,
    QualityGate,
    from_mapping,
)
from scilink.agents.exp_agents._verification_record import (
    CURVE_HISTORY_KEYMAP,
    CURVE_PROMPT_KEYMAP,
    IMAGE_HISTORY_KEYMAP,
    IMAGE_PROMPT_KEYMAP,
    build_quality_history,
    build_verification_prompt_history,
)
from scilink.agents.exp_agents._critique import SYNTHESIS, CritiquePayload
from scilink.agents.exp_agents._qc_profile import (
    REALTIME,
    THOROUGH,
    QCProfile,
    resolve_profile,
)


# ---------------------------------------------------------------------------
# QualityGate.value_source
# ---------------------------------------------------------------------------

def test_gate_value_source_default_is_result():
    assert R_SQUARED_DEFAULT.value_source == "result"
    assert QualityGate().value_source == "result"


def test_gate_value_source_validation():
    with pytest.raises(ValueError, match="value_source"):
        QualityGate(value_source="oracle")


def test_extract_result_source_ignores_verdict():
    v = R_SQUARED_DEFAULT.extract({"r_squared": 0.97}, verdict={"r_squared": 0.1})
    assert v == 0.97


def test_extract_verdict_source_reads_verdict():
    gate = IMAGE_SCORE_DEFAULT
    assert gate.extract(None, verdict={"quality_score": 0.85}) == 0.85
    # result dict is not consulted for verdict-source gates
    assert gate.extract({"quality_score": 0.2}, verdict={"quality_score": 0.85}) == 0.85
    # missing verdict → None (hard reject semantics)
    assert gate.extract({"quality_score": 0.9}) is None


def test_extract_nan_and_non_numeric_none():
    assert R_SQUARED_DEFAULT.extract({"r_squared": float("nan")}) is None
    assert R_SQUARED_DEFAULT.extract({"r_squared": "bad"}) is None
    assert IMAGE_SCORE_DEFAULT.extract(None, verdict={"quality_score": None}) is None


def test_image_default_gate_shape():
    g = IMAGE_SCORE_DEFAULT
    assert g.metric == "quality_score"
    assert g.accept_threshold == 0.7
    assert g.hard_reject_threshold == 0.7  # empty soft band
    assert g.value_source == "verdict"
    assert g.best_value == 1.0
    # empty soft band: nothing is soft
    assert not g.is_soft_band(0.699)
    assert not g.is_soft_band(0.7)


def test_from_mapping_value_source_roundtrip():
    g = from_mapping({"metric": "quality_score", "accept_threshold": 0.6,
                      "hard_reject_threshold": 0.6, "value_source": "verdict"})
    assert g.value_source == "verdict"
    # default when omitted
    assert from_mapping({"metric": "r_squared"}).value_source == "result"


def test_with_accept_threshold_preserves_value_source():
    g = IMAGE_SCORE_DEFAULT.with_accept_threshold(0.8)
    assert g.value_source == "verdict"
    assert g.accept_threshold == 0.8


# ---------------------------------------------------------------------------
# Arithmetic equivalence of the routed accept checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9, 0.95, 0.999])
def test_gate_is_accept_equals_raw_comparison(threshold):
    """gate.is_accept(v) must equal (v >= t) on a grid around the threshold —
    the exact property the phase-1 driver routing relies on."""
    gate_r2 = R_SQUARED_DEFAULT.with_accept_threshold(threshold)
    gate_img = IMAGE_SCORE_DEFAULT.with_accept_threshold(threshold)
    eps = 1e-12
    grid = [0.0, threshold - 0.1, threshold - eps, threshold,
            threshold + eps, threshold + 0.1, 1.0]
    for v in grid:
        assert gate_r2.is_accept(v) == (v >= threshold), v
        assert gate_img.is_accept(v) == (v >= threshold), v
    assert gate_r2.is_accept(None) is False


def test_gate_rebuild_observes_threshold_mutation():
    """Mirrors curve's adjust_threshold action: rebuilding the gate from the
    mutated live threshold must change the verdict accordingly."""
    live_threshold = 0.95
    gate = R_SQUARED_DEFAULT.with_accept_threshold(live_threshold)
    assert not gate.is_accept(0.91)
    live_threshold = 0.90  # human adjusted
    gate = R_SQUARED_DEFAULT.with_accept_threshold(live_threshold)
    assert gate.is_accept(0.91)


# ---------------------------------------------------------------------------
# Shared verification-record builders
# ---------------------------------------------------------------------------

_VERIFICATION_HISTORY = [
    {
        "r_squared": 0.6,
        "quality_score": 0.4,
        "result_type": "delivered",
        "annealing_level": 1,
        "tools_used": ["find_peaks"],
        "config_used": {"physical_model": "Lorentzian",
                        "processing_pipeline": "threshold -> label"},
        "issues_found": [{"location": "peak", "problem": "missed",
                          "evidence": "resid", "suggested_fix": "add peak"}],
        "recommended_action": "add a peak",
    },
]


def test_curve_history_shape_and_order():
    h = build_quality_history(
        best_value=0.97, threshold=0.95,
        all_attempts=[{"model": "A", "r2": 0.6},
                      {"model": "B", "r2": 0.97, "diagnosis": "better"},
                      {"model": "Verification-1", "r2": 0.97}],
        verification_history=_VERIFICATION_HISTORY,
        judge_result={"reasoning": "judged"},
        script_errors=[{"error": "e", "diagnosis": "d"}],
        keymap=CURVE_HISTORY_KEYMAP,
    )
    assert list(h) == ["final_r2", "threshold", "approved",
                       "verification_iterations", "alternative_models",
                       "script_errors", "judge_reasoning"]
    assert h["approved"] is True
    it = h["verification_iterations"][0]
    assert list(it) == ["r_squared", "annealing_level", "tools_used",
                        "model", "issues", "fix_applied"]
    assert it["model"] == "Lorentzian"
    assert it["issues"] == [{"location": "peak", "problem": "missed"}]
    # Verification-* attempts filtered; first attempt excluded
    assert h["alternative_models"] == [{"model": "B", "r2": 0.97,
                                        "diagnosis": "better"}]


def test_image_history_shape_and_order():
    h = build_quality_history(
        best_value=0.65, threshold=0.7,
        all_attempts=None,
        verification_history=_VERIFICATION_HISTORY,
        judge_result={"reasoning": "judged", "score_explanation": "expl"},
        script_errors=None,
        keymap=IMAGE_HISTORY_KEYMAP,
    )
    assert list(h) == ["final_score", "threshold", "approved",
                       "verification_iterations", "script_errors",
                       "judge_reasoning", "score_explanation"]
    assert h["approved"] is False
    it = h["verification_iterations"][0]
    assert list(it) == ["score", "result_type", "annealing_level",
                        "issues", "tools_used", "fix_applied"]
    assert it["score"] == 0.4
    assert h["score_explanation"] == "expl"


def test_image_history_missing_score_raises():
    """entry['quality_score'] hard-indexing is deliberate historical behavior."""
    with pytest.raises(KeyError):
        build_quality_history(
            best_value=0.9, threshold=0.7, all_attempts=None,
            verification_history=[{"annealing_level": 0}],
            judge_result=None, script_errors=None,
            keymap=IMAGE_HISTORY_KEYMAP,
        )


def test_prompt_history_empty_and_modality_wording():
    assert build_verification_prompt_history([], CURVE_PROMPT_KEYMAP) == ""
    curve_text = build_verification_prompt_history(
        _VERIFICATION_HISTORY, CURVE_PROMPT_KEYMAP)
    image_text = build_verification_prompt_history(
        _VERIFICATION_HISTORY, IMAGE_PROMPT_KEYMAP)
    assert "R² = 0.6000" in curve_text
    assert "- Config: Lorentzian" in curve_text
    assert "  • peak: missed" in curve_text
    assert "the plot, a registered tool's" in curve_text
    assert "plateau/convergence rule" in curve_text

    assert "Quality score = 0.40" in image_text
    assert "- Pipeline: threshold -> label" in image_text
    assert "  - peak: missed" in image_text
    assert "the images, a registered tool's" in image_text
    assert "2. If a fix didn't work, suggest something DIFFERENT" in image_text


# ---------------------------------------------------------------------------
# CritiquePayload
# ---------------------------------------------------------------------------

def test_critique_payload_defaults_to_synthesis():
    p = CritiquePayload(source="human", critique="the trend has two regimes")
    assert p.target == SYNTHESIS
    assert p.targets_synthesis


def test_critique_payload_unit_target():
    p = CritiquePayload(source="consistency", critique="peers use model X",
                        target=3, hints={"expected_model": "X"})
    assert not p.targets_synthesis
    assert p.hints["expected_model"] == "X"


@pytest.mark.parametrize("bad", [
    {"source": "oracle", "critique": "x"},
    {"source": "human", "critique": "   "},
    {"source": "human", "critique": "x", "target": "unit-3"},
])
def test_critique_payload_validation(bad):
    with pytest.raises(ValueError):
        CritiquePayload(**bad)


# ---------------------------------------------------------------------------
# QCProfile
# ---------------------------------------------------------------------------

def test_thorough_matches_todays_defaults():
    assert THOROUGH.max_verification_iterations == 7
    assert THOROUGH.check_plan_conformance
    assert THOROUGH.human_feedback
    assert THOROUGH.escalation_enabled
    assert not THOROUGH.voted_verification


def test_realtime_preset_is_light():
    assert REALTIME.max_verification_iterations <= 1
    assert not REALTIME.check_plan_conformance
    assert not REALTIME.best_of_n_eligible
    assert not REALTIME.human_feedback
    assert not REALTIME.literature
    assert not REALTIME.escalation_enabled


def test_resolve_profile():
    assert resolve_profile(None) is THOROUGH
    assert resolve_profile("realtime") is REALTIME
    assert resolve_profile("THOROUGH") is THOROUGH
    assert resolve_profile(REALTIME) is REALTIME
    with pytest.raises(ValueError):
        resolve_profile("turbo")
    with pytest.raises(TypeError):
        resolve_profile(3)


def test_from_agent_kwargs_bridge():
    p = QCProfile.from_agent_kwargs(
        max_verification_iterations=0,
        enable_human_feedback=False,
        use_literature=False,
        n_candidates=1,
    )
    assert p.max_verification_iterations == 0
    assert not p.human_feedback
    assert not p.literature
    assert not p.best_of_n_eligible
    # None leaves base values in place
    q = QCProfile.from_agent_kwargs()
    assert q == THOROUGH


def test_profile_validation():
    with pytest.raises(ValueError):
        QCProfile(name="")
    with pytest.raises(ValueError):
        QCProfile(name="x", max_verification_iterations=-1)
