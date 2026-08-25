"""Tests for the QualityGate dataclass and resolver."""

from __future__ import annotations

import pytest

from scilink.agents.exp_agents.quality_gate import (
    QualityGate,
    R_SQUARED_DEFAULT,
    from_mapping,
    resolve_gate,
)


# --- Dataclass invariants -----------------------------------------------------

def test_default_is_r_squared():
    g = QualityGate()
    assert g.metric == "r_squared"
    assert g.accept_threshold == 0.95
    assert g.hard_reject_threshold == 0.90
    assert g.direction == "higher_is_better"


def test_higher_is_better_threshold_ordering():
    with pytest.raises(ValueError, match="higher_is_better"):
        QualityGate(accept_threshold=0.5, hard_reject_threshold=0.8)


def test_lower_is_better_threshold_ordering():
    with pytest.raises(ValueError, match="lower_is_better"):
        QualityGate(metric="cost", accept_threshold=0.5,
                    hard_reject_threshold=0.3, direction="lower_is_better")


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        QualityGate(direction="medium")


# --- extract / is_accept / is_hard_reject -------------------------------------

def test_extract_handles_missing_and_malformed_values():
    g = QualityGate()
    assert g.extract(None) is None
    assert g.extract({}) is None
    assert g.extract({"r_squared": "not a number"}) is None
    assert g.extract({"r_squared": float("nan")}) is None
    assert g.extract({"r_squared": 0.97}) == 0.97


def test_higher_is_better_acceptance():
    g = QualityGate()  # r² ≥ 0.95
    assert g.is_accept(0.97)
    assert g.is_accept(0.95)
    assert not g.is_accept(0.94)
    assert g.is_hard_reject(0.85)
    assert not g.is_hard_reject(0.90)
    assert g.is_soft_band(0.93)
    assert not g.is_soft_band(0.99)


def test_lower_is_better_acceptance():
    g = QualityGate(metric="cost", accept_threshold=0.25,
                    hard_reject_threshold=0.55, direction="lower_is_better")
    assert g.is_accept(0.10)
    assert g.is_accept(0.25)
    assert not g.is_accept(0.30)
    assert g.is_hard_reject(0.70)
    assert not g.is_hard_reject(0.55)
    assert g.is_soft_band(0.40)


def test_none_value_is_hard_reject():
    """Missing metric → hard reject (preserves pre-existing behavior when
    fit_quality.r_squared defaulted to 0 and failed the gate)."""
    g = QualityGate()
    assert not g.is_accept(None)
    assert g.is_hard_reject(None)
    assert not g.is_soft_band(None)


# --- Resolution priority ------------------------------------------------------

def test_resolve_call_override_wins():
    call_gate = QualityGate(metric="cost", accept_threshold=0.2,
                             hard_reject_threshold=0.5, direction="lower_is_better")
    agent_gate = QualityGate(accept_threshold=0.97)
    skill_meta = {"quality_gate": {"metric": "figure_of_merit",
                                    "accept_threshold": 0.7,
                                    "hard_reject_threshold": 0.4}}
    resolved = resolve_gate(
        call_override=call_gate, agent_default=agent_gate,
        skill_meta=skill_meta, legacy_threshold=0.85,
    )
    assert resolved is call_gate


def test_resolve_agent_when_no_call_override():
    agent_gate = QualityGate(accept_threshold=0.97)
    skill_meta = {"quality_gate": {"metric": "figure_of_merit",
                                    "accept_threshold": 0.7,
                                    "hard_reject_threshold": 0.4}}
    resolved = resolve_gate(
        call_override=None, agent_default=agent_gate,
        skill_meta=skill_meta, legacy_threshold=0.85,
    )
    assert resolved is agent_gate


def test_resolve_skill_when_no_explicit_gate():
    skill_meta = {"quality_gate": {"metric": "figure_of_merit",
                                    "accept_threshold": 0.7,
                                    "hard_reject_threshold": 0.4}}
    resolved = resolve_gate(skill_meta=skill_meta, legacy_threshold=0.85)
    assert resolved.metric == "figure_of_merit"
    assert resolved.accept_threshold == 0.7


def test_resolve_legacy_threshold_when_no_skill_gate():
    resolved = resolve_gate(legacy_threshold=0.97)
    assert resolved.metric == "r_squared"
    assert resolved.accept_threshold == 0.97
    # Soft band width preserved (default is 0.05).
    assert resolved.hard_reject_threshold == pytest.approx(0.92)


def test_resolve_default_when_nothing_specified():
    resolved = resolve_gate()
    assert resolved is R_SQUARED_DEFAULT


def test_resolve_dict_call_override_coerced():
    resolved = resolve_gate(call_override={"metric": "cost",
                                            "accept_threshold": 0.2,
                                            "hard_reject_threshold": 0.5,
                                            "direction": "lower_is_better"})
    assert resolved.metric == "cost"
    assert resolved.direction == "lower_is_better"


def test_resolve_invalid_type_raises():
    with pytest.raises(TypeError):
        resolve_gate(call_override="not-a-gate")


# --- with_accept_threshold ----------------------------------------------------

def test_with_accept_threshold_preserves_soft_band_width():
    g = QualityGate(accept_threshold=0.95, hard_reject_threshold=0.90)
    new = g.with_accept_threshold(0.85)
    assert new.accept_threshold == 0.85
    assert new.hard_reject_threshold == pytest.approx(0.80)


def test_with_accept_threshold_lower_is_better():
    g = QualityGate(metric="cost", accept_threshold=0.25,
                    hard_reject_threshold=0.55, direction="lower_is_better")
    new = g.with_accept_threshold(0.40)
    assert new.accept_threshold == 0.40
    assert new.hard_reject_threshold == pytest.approx(0.70)


# --- from_mapping fills in defaults -------------------------------------------

def test_from_mapping_minimal():
    g = from_mapping({"metric": "figure_of_merit", "accept_threshold": 0.7,
                       "hard_reject_threshold": 0.4})
    assert g.metric == "figure_of_merit"
    assert g.direction == "higher_is_better"  # defaulted


# --- user_threshold override (experienced-user R²) ---------------------------

class _CaptureLogger:
    def __init__(self): self.warnings = []
    def warning(self, msg): self.warnings.append(msg)


def test_user_threshold_overrides_skill_r2_gate():
    """A user R² override beats a skill's r_squared gate, and warns."""
    skill_meta = {"quality_gate": {"metric": "r_squared", "accept_threshold": 0.90,
                                   "hard_reject_threshold": 0.75}}
    log = _CaptureLogger()
    resolved = resolve_gate(skill_meta=skill_meta, user_threshold=0.85, logger=log)
    assert resolved.metric == "r_squared"
    assert resolved.accept_threshold == pytest.approx(0.85)
    assert any("recommends 0.900" in w for w in log.warnings)


def test_user_threshold_metric_guard_keeps_skill_gate():
    """A bare R² override must NOT replace a skill's non-r_squared metric."""
    skill_meta = {"quality_gate": {"metric": "figure_of_merit", "accept_threshold": 0.7,
                                   "hard_reject_threshold": 0.4}}
    log = _CaptureLogger()
    resolved = resolve_gate(skill_meta=skill_meta, user_threshold=0.85, logger=log)
    assert resolved.metric == "figure_of_merit"          # skill gate kept
    assert resolved.accept_threshold == pytest.approx(0.7)
    assert any("ignored" in w and "figure_of_merit" in w for w in log.warnings)


def test_user_threshold_applies_with_no_skill():
    resolved = resolve_gate(user_threshold=0.88, legacy_threshold=0.95)
    assert resolved.metric == "r_squared"
    assert resolved.accept_threshold == pytest.approx(0.88)


def test_explicit_quality_gate_still_beats_user_threshold():
    """An explicit quality_gate (full gate) outranks a numeric R² override."""
    call_gate = QualityGate(metric="cost", accept_threshold=0.2,
                            hard_reject_threshold=0.5, direction="lower_is_better")
    resolved = resolve_gate(call_override=call_gate, user_threshold=0.85,
                            skill_meta={"quality_gate": {"metric": "r_squared",
                                                         "accept_threshold": 0.9,
                                                         "hard_reject_threshold": 0.75}})
    assert resolved.metric == "cost"


def test_no_user_threshold_preserves_skill_over_legacy():
    """Sanity: without a user override, skill still beats legacy (unchanged)."""
    skill_meta = {"quality_gate": {"metric": "r_squared", "accept_threshold": 0.90,
                                   "hard_reject_threshold": 0.75}}
    resolved = resolve_gate(skill_meta=skill_meta, legacy_threshold=0.95)
    assert resolved.accept_threshold == pytest.approx(0.90)  # skill, not legacy


# --- Escalation fast-accept margin (metric-agnostic) --------------------------

def test_fast_margin_r2_default_reproduces_legacy():
    # band 0.05, fraction 0.4 -> 0.02; cap (1-0.95)/2=0.025 -> 0.02; bar 0.97.
    g = R_SQUARED_DEFAULT
    assert g.fast_accept_margin(0.4) == pytest.approx(0.02)
    assert g.clears_by_fast_margin(0.98, 0.4) is True
    assert g.clears_by_fast_margin(0.955, 0.4) is False
    assert g.clears_by_fast_margin(0.97, 0.4) is True  # exactly the bar


def test_fast_margin_r2_clamped_near_ceiling():
    # accept 0.99 -> band preserved 0.05 -> raw 0.02, capped at (1-0.99)/2.
    g = R_SQUARED_DEFAULT.with_accept_threshold(0.99)
    assert g.fast_accept_margin(0.4) == pytest.approx(0.005)
    assert g.clears_by_fast_margin(0.997, 0.4) is True   # >= 0.995
    assert g.clears_by_fast_margin(0.993, 0.4) is False  # approved but < 0.995


def test_fast_margin_lower_is_better_direction():
    g = QualityGate(metric="reduced_chi2", accept_threshold=1.5,
                    hard_reject_threshold=3.0, direction="lower_is_better",
                    best_value=0.0)
    assert g.fast_accept_margin(0.4) == pytest.approx(0.6)  # 0.4 * band 1.5
    assert g.clears_by_fast_margin(0.8, 0.4) is True    # <= 0.9
    assert g.clears_by_fast_margin(1.4, 0.4) is False   # accepted but > 0.9


def test_fast_margin_unbounded_metric_no_cap():
    # No best_value -> margin is the full band fraction, no ceiling clamp.
    g = QualityGate(metric="bic", accept_threshold=100.0,
                    hard_reject_threshold=200.0, direction="lower_is_better")
    assert g.best_value is None
    assert g.fast_accept_margin(0.4) == pytest.approx(40.0)


def test_fast_margin_missing_value_does_not_clear():
    assert R_SQUARED_DEFAULT.clears_by_fast_margin(None, 0.4) is False


def test_from_mapping_infers_best_value_for_r_squared():
    g = from_mapping({"metric": "r_squared", "accept_threshold": 0.99,
                      "hard_reject_threshold": 0.94})
    assert g.best_value == 1.0
    g2 = from_mapping({"metric": "reduced_chi2", "accept_threshold": 1.5,
                       "hard_reject_threshold": 3.0, "direction": "lower_is_better"})
    assert g2.best_value is None  # unknown metric -> unbounded unless declared
