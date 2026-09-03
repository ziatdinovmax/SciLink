"""Interleaved series along a label axis: axis coherence, scout-all, prompt
block, and data-space regime-membership correction — with the conventional
monotone series left byte-identical."""
import json
import numpy as np
import pytest

from scilink.skills._shared.series_reduction import reduce_curves
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    SeriesScoutController, _correct_regime_membership)

X = np.linspace(0, 100, 200)


def _curve(step):          # a flat trace with an optional level step at x = 50
    return X, np.where(X > 50, step, 0.0) + 0.02 * np.sin(X)


def _interleaved():        # controls at 0, 3, 8; +/- steps elsewhere (file order axis)
    steps = [0, 25, -25, 0, 22, -23, 21, -15, 0]
    return [_curve(s) for s in steps], steps


def _monotone(n=12, t0=6):  # conventional: a composition switch along temperature
    curves = [_curve(0 if i < t0 else 20) for i in range(n)]
    return curves, [300 + 10 * i for i in range(n)]


def test_axis_coherence_flags_interleaved_but_not_monotone():
    curves, _ = _interleaved()
    r = reduce_curves(curves, controls=list(range(9)), control_source="role_index")
    assert r["status"] == "success" and r["axis_coherence"]["coherent"] is False
    assert r["axis_coherence"]["n_reversals"] >= 2 and len(r["scores_by_index"]) == 9
    curves, temps = _monotone()
    r2 = reduce_curves(curves, controls=temps, control_source="temperature")
    assert r2["axis_coherence"]["coherent"] is True and r2["axis_coherence"]["n_reversals"] == 0
    assert abs(r2["change_point"] - 355) < 6


def test_scores_by_index_follow_input_order_not_sorted_order():
    curves, temps = _monotone()
    shuffled = [5, 0, 11, 3, 8, 1, 9, 2, 10, 4, 7, 6]
    r = reduce_curves([curves[i] for i in shuffled], controls=[temps[i] for i in shuffled])
    sc = np.asarray(r["scores_by_index"])
    high = [k for k, i in enumerate(shuffled) if i >= 6]
    low = [k for k, i in enumerate(shuffled) if i < 6]
    assert sc[high].min() > sc[low].max()


def test_scout_all_only_when_incoherent():
    assert SeriesScoutController._select_scout_indices(9) == [0, 2, 4, 6, 8]
    assert SeriesScoutController._select_scout_indices(9, scout_all=True) == list(range(9))
    assert SeriesScoutController._select_scout_indices(40, scout_all=True) == \
        SeriesScoutController._select_scout_indices(40)          # above the cap: unchanged
    assert SeriesScoutController._select_scout_indices(12) == [0, 3, 6, 9, 11]


def test_membership_correction_moves_clear_misfit_only_when_incoherent():
    curves, steps = _interleaved()
    red = reduce_curves(curves, controls=list(range(9)), control_source="role_index")
    regimes = [{"name": "control", "spectrum_indices": [0, 1, 3, 8]},          # 1 is wrong (a +25 step)
               {"name": "insertion", "spectrum_indices": [4, 6]},
               {"name": "retraction", "spectrum_indices": [2, 5, 7]}]
    moves = _correct_regime_membership(regimes, red)
    assert moves == [(1, "control", "insertion")]
    assert regimes[0]["spectrum_indices"] == [0, 3, 8] and regimes[1]["spectrum_indices"] == [1, 4, 6]
    # coherent axis, sharp two-level series: a boundary spectrum the planner put on
    # the wrong side is an UNAMBIGUOUS misfit (score inside the other regime's range)
    curves, temps = _monotone()
    red2 = reduce_curves(curves, controls=temps, control_source="temperature")
    regimes2 = [{"name": "a", "spectrum_indices": [0, 1, 2, 3, 4, 5, 6]}, {"name": "b", "spectrum_indices": [7, 8, 9, 10, 11]}]
    assert _correct_regime_membership(regimes2, red2) == [(6, "a", "b")]
    assert regimes2[0]["spectrum_indices"] == [0, 1, 2, 3, 4, 5] and regimes2[1]["spectrum_indices"] == [6, 7, 8, 9, 10, 11]
    # coherent axis, a control filed into a sharp-step regime: moved even though the
    # control regime's two members have (near-)identical scores
    curves = [_curve(s) for s in (0, 0, 0, 25, 25, 25, -25, -25, -25)]
    red4 = reduce_curves(curves, controls=[0, 0, 0, 1, 1, 1, 2, 2, 2], control_source="magnet_action")
    regimes5 = [{"name": "ctl", "spectrum_indices": [0, 1]}, {"name": "ins", "spectrum_indices": [2, 3, 4, 5]}, {"name": "ret", "spectrum_indices": [6, 7, 8]}]
    assert _correct_regime_membership(regimes5, red4) == [(2, "ins", "ctl")]
    # coherent axis, a planner's 2-member "transition" bin of pure-phase spectra: left alone
    # (a 1-2 member regime has no reliable median; conventional plans are not restructured)
    curves, temps = _monotone()
    red5 = reduce_curves(curves, controls=temps, control_source="temperature")
    regimes6 = [{"name": "low", "spectrum_indices": [0, 1, 2, 3, 4]}, {"name": "coex", "spectrum_indices": [5, 6]},
                {"name": "high", "spectrum_indices": [7, 8, 9, 10, 11]}]
    assert _correct_regime_membership(regimes6, red5) == [] and regimes6[1]["spectrum_indices"] == [5, 6]
    # coherent axis, GRADUAL transition: nothing is moved (a ramp spectrum is never inside the other regime's range by a wide margin)
    ramp = [_curve(20.0 * i / 11) for i in range(12)]
    red3 = reduce_curves(ramp, controls=temps, control_source="temperature")
    assert red3["axis_coherence"]["coherent"] is True
    regimes3 = [{"name": "a", "spectrum_indices": [0, 1, 2, 3, 4, 5]}, {"name": "b", "spectrum_indices": [6, 7, 8, 9, 10, 11]}]
    assert _correct_regime_membership(regimes3, red3) == []
    regimes4 = [{"name": "a", "spectrum_indices": [0, 1, 2, 3, 4, 5, 6, 7]}, {"name": "b", "spectrum_indices": [8, 9, 10, 11]}]
    assert _correct_regime_membership(regimes4, red3) == []
    assert _correct_regime_membership(regimes, None) == []


def test_planner_prompt_gets_membership_block_only_when_incoherent(tmp_path):
    from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cfc
    src = open(cfc.__file__).read()
    assert "AXIS NOT COHERENT" in src and "PC1 score per spectrum index" in src
    i = src.index("AXIS NOT COHERENT"); j = src.index("Use this to place regime boundaries")
    assert i < j and 'if ac and not ac.get("coherent", True)' in src


def test_missing_indices_join_nearest_neighbour_or_nearest_score():
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import _assign_missing_indices
    # coherent (value-sorted) axis: neighbours decide
    regimes = [{"name": "ret", "spectrum_indices": [0, 2]}, {"name": "ctl", "spectrum_indices": [4]}, {"name": "ins", "spectrum_indices": [6, 8]}]
    rule = _assign_missing_indices(regimes, {1, 3, 5, 7}, {"axis_coherence": {"coherent": True}})
    assert "neighbour" in rule                      # no scores: ties (3, 5) go to the lower neighbour
    assert regimes[0]["spectrum_indices"] == [0, 1, 2, 3] and regimes[1]["spectrum_indices"] == [4, 5] and regimes[2]["spectrum_indices"] == [6, 7, 8]
    # with scores, an equidistant index follows the data: here 3 and 5 look like the control (score ~0)
    curves = [_curve(s) for s in (-25, -25, -25, 0, 0, 0, 25, 25, 25)]
    red = reduce_curves(curves, controls=[-1, -1, -1, 0, 0, 0, 1, 1, 1], control_source="magnet_condition")
    assert red["axis_coherence"]["coherent"] is True
    regimes = [{"name": "ret", "spectrum_indices": [0, 2]}, {"name": "ctl", "spectrum_indices": [4]}, {"name": "ins", "spectrum_indices": [6, 8]}]
    _assign_missing_indices(regimes, {1, 3, 5, 7}, red)
    assert regimes[0]["spectrum_indices"] == [0, 1, 2] and regimes[1]["spectrum_indices"] == [3, 4, 5] and regimes[2]["spectrum_indices"] == [6, 7, 8]
    # incoherent axis: PC1 score decides
    curves, steps = _interleaved()
    red = reduce_curves(curves, controls=list(range(9)), control_source="role_index")
    regimes = [{"name": "ctl", "spectrum_indices": [0, 8]}, {"name": "ins", "spectrum_indices": [4]}, {"name": "ret", "spectrum_indices": [2]}]
    rule = _assign_missing_indices(regimes, {1, 3, 5, 6, 7}, red)
    assert "PC1" in rule
    assert regimes[0]["spectrum_indices"] == [0, 3, 8] and regimes[1]["spectrum_indices"] == [1, 4, 6] and regimes[2]["spectrum_indices"] == [2, 5, 7]
    # nothing assigned at all: first regime, as before
    regimes = [{"name": "a", "spectrum_indices": []}, {"name": "b", "spectrum_indices": []}]
    assert "first regime" in _assign_missing_indices(regimes, {0, 1}, None) and regimes[0]["spectrum_indices"] == [0, 1]


def test_membership_correction_converges_over_passes():
    # planner put one control and two retractions into a 3-member 'insertion' bin;
    # pass 1 moves the two stepped spectra, pass 2 then sees the control as a misfit
    curves = [_curve(s) for s in (0, 0, 0, 25, 25, 25, -25, -25, -25)]
    red = reduce_curves(curves, controls=[0, 0, 0, 1, 1, 1, 2, 2, 2], control_source="magnet_action")
    regimes = [{"name": "ctl", "spectrum_indices": [0, 1]}, {"name": "ins", "spectrum_indices": [2, 4]},
               {"name": "ret", "spectrum_indices": [3, 5, 6, 7, 8]}]
    moves = _correct_regime_membership(regimes, red)
    assert set(m[0] for m in moves) == {2, 3, 5}
    assert regimes[0]["spectrum_indices"] == [0, 1, 2] and regimes[1]["spectrum_indices"] == [3, 4, 5] and regimes[2]["spectrum_indices"] == [6, 7, 8]
