"""score_xrd_match_robust(mode='component') — scoring ONE phase of a mixture.

Synthetic peak lists: a minor phase whose lines are all present but carry a
small share of the measured intensity must score well in component mode (and
poorly in single mode); a candidate whose strong lines are absent, or whose
lines only land in a dense band of measured peaks, must not.
"""
from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd.score_match_robust import (
    _poisson_binomial_tail, score_xrd_match_robust)

MAJOR = ([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], [100, 80, 60, 50, 40, 30, 20])
MINOR = ([24.0, 33.0, 45.0, 57.0, 66.0], [100, 70, 50, 40, 30])


def _peaks(*phases_scales):
    pos, inten = [], []
    for (p, i), s in phases_scales:
        pos += list(p); inten += [s * v for v in i]
    return {"positions": pos, "intensities": inten}


def test_minor_component_accepted_in_component_mode_only():
    exp = _peaks((MAJOR, 1.0), (MINOR, 0.1))
    comp = score_xrd_match_robust(MINOR[0], MINOR[1], exp_peaks=exp, mode="component")
    single = score_xrd_match_robust(MINOR[0], MINOR[1], exp_peaks=exp)
    assert comp["verdict"] == "accept", comp
    assert single["figure_of_merit"] < comp["figure_of_merit"]
    assert comp["missing_strong_lines"] == []


def test_absent_phase_rejected():
    exp = _peaks((MAJOR, 1.0))
    absent = ([22.5, 35.5, 47.5, 63.5], [100, 60, 50, 40])
    r = score_xrd_match_robust(absent[0], absent[1], exp_peaks=exp, mode="component")
    assert r["verdict"] == "reject"
    assert len(r["missing_strong_lines"]) == 4


def test_dense_measured_band_is_not_evidence():
    # measured: a forest of noise-like peaks every 0.4° in 10-18° + the major phase
    band = list(np.arange(10.0, 18.0, 0.4))
    exp = {"positions": band + MAJOR[0], "intensities": [5.0] * len(band) + MAJOR[1]}
    forest = ([10.6, 11.4, 12.2, 13.0, 13.8, 14.6, 15.4], [100, 90, 80, 70, 60, 50, 40])
    on = score_xrd_match_robust(forest[0], forest[1], exp_peaks=exp, mode="component")
    off = score_xrd_match_robust(forest[0], forest[1], exp_peaks=exp, mode="component",
                                 chance_correction=False)
    assert off["coverage"] == pytest.approx(1.0)
    assert on["figure_of_merit"] < 0.4 < off["figure_of_merit"]
    assert on["chance_p_value"] > 0.5


def test_three_line_rule_caps_verdict():
    exp = _peaks((MAJOR, 1.0))
    two = ([30.0, 60.0], [100, 60])
    r = score_xrd_match_robust(two[0], two[1], exp_peaks=exp, mode="component")
    assert r["n_strong_lines_tested"] == 2
    assert r["verdict"] != "accept"


def test_component_defaults_and_validation():
    exp = _peaks((MAJOR, 1.0), (MINOR, 0.1))
    r = score_xrd_match_robust(MINOR[0], MINOR[1], exp_peaks=exp, mode="component")
    assert r["mode"] == "component" and r["fitted_scale"] == 1.0   # no lattice-scale fit for a measured cell
    with pytest.raises(ValueError):
        score_xrd_match_robust(MINOR[0], MINOR[1], exp_peaks=exp, mode="mixture")
    with pytest.raises(ValueError):
        score_xrd_match_robust(MINOR[0], MINOR[1], exp_peaks=exp, mode="component", algorithm="mip")


def test_poisson_binomial_tail():
    assert _poisson_binomial_tail([0.5, 0.5], 2) == pytest.approx(0.25)
    assert _poisson_binomial_tail([0.2, 0.3, 0.4], 0) == pytest.approx(1.0)
    assert _poisson_binomial_tail([0.1, 0.1, 0.1], 3) == pytest.approx(0.001)
