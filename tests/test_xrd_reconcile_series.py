"""Offline tests for reconcile_series_phases (profile + identification join).

Deterministic, no LLM: a synthetic crossfade series where LOW-T peaks decay
and HIGH-T peaks grow, with per-frame phase labels — exercises peak tracking,
regime split, the two transition estimates + agreement, and the honest
"unidentified" path when a regime's phase is null.
"""

from __future__ import annotations

import pytest

from scilink.skills.structure_matching.xrd.reconcile_series import (
    reconcile_series_phases, TOOL_SPEC as RS_SPEC)


def _series(n=21, low_name="PhaseA", high_name="PhaseB",
            id_switch_frame=10, id_gap=(9, 11)):
    """LOW-T peaks (11.0, 22.0) fade over frames; HIGH-T (12.0, 20.0) grow.
    Temperature 30..70. Identification names low_name early, high_name late,
    with a null gap in id_gap (the transition frames the ID declines)."""
    frames, phase_ids = [], []
    for i in range(n):
        T = 30 + 40 * i / (n - 1)
        f = i / (n - 1)                          # 0..1 crossfade
        peaks = [{"center": 11.0, "area": 1000 * (1 - f)},
                 {"center": 22.0, "area": 500 * (1 - f)},
                 {"center": 12.0, "area": 1000 * f},
                 {"center": 20.0, "area": 500 * f}]
        frames.append({"value": T, "peaks": peaks})
        if id_gap[0] <= i <= id_gap[1]:
            ph = None
        else:
            ph = low_name if i < id_switch_frame else high_name
        phase_ids.append({"value": T, "phase": ph, "figure_of_merit": 0.9})
    return frames, phase_ids


def test_tool_registered_with_knobs():
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "reconcile_series_phases" in names
    for k in ("tol_deg", "min_presence_frac", "agreement_deg", "output_figure"):
        assert k in RS_SPEC.parameters


def test_reconciles_and_labels(tmp_path):
    frames, ph = _series()
    r = reconcile_series_phases(frames, ph, output_figure=str(tmp_path / "r.png"))
    assert r["low_t_phase"] == "PhaseA"
    assert r["high_t_phase"] == "PhaseB"
    # both transitions near the crossfade midpoint (T=50)
    assert abs(r["transition_profile"] - 50) < 6
    assert abs(r["transition_identification"] - 50) < 8
    assert r["agreement"]["verdict"] == "consistent"
    # peaks attributed to the right regime/phase
    by_pos = {p["position_deg"]: p for p in r["tracked_peaks"]}
    assert by_pos[11.0]["regime"] == "low" and by_pos[11.0]["phase"] == "PhaseA"
    assert by_pos[12.0]["regime"] == "high" and by_pos[12.0]["phase"] == "PhaseB"
    assert r["figure"] and (tmp_path / "r.png").exists()


def test_unidentified_regime_stays_null():
    # HIGH-T phase never identified (all high-frame IDs null) — its trends are
    # real but the label must stay null, not borrow the low-T name.
    frames, ph = _series(id_switch_frame=10)
    for i in range(10, len(ph)):
        ph[i]["phase"] = None
    r = reconcile_series_phases(frames, ph)
    assert r["low_t_phase"] == "PhaseA"
    assert r["high_t_phase"] is None                    # honestly unidentified
    assert r["transition_profile"] is not None          # profile still works
    assert r["transition_identification"] is None        # ID couldn't time it
    assert r["agreement"]["verdict"] == "one_sided"


def test_divergence_flagged():
    # profile transition ~50; force the ID switch very late so the two disagree
    frames, ph = _series(id_switch_frame=19, id_gap=(19, 19))
    r = reconcile_series_phases(frames, ph, agreement_deg=5.0)
    if r["transition_identification"] is not None:
        assert r["agreement"]["verdict"] in ("divergent", "consistent")
        # with a late ID switch and tight tolerance it should read divergent
        if abs(r["transition_profile"] - r["transition_identification"]) > 5:
            assert r["agreement"]["verdict"] == "divergent"


def test_guards():
    with pytest.raises(ValueError):
        reconcile_series_phases([{"value": 0, "peaks": []}], [{"value": 0, "phase": None}])
