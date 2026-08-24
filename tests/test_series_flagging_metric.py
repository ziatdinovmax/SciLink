"""Series outlier-flagging must score by the GATE's metric, not global R².

A correct low-SNR fit has a high gate metric (peak_region_r2) but a low global
R²; the series flagger must not false-flag it. Mirrors the real NLTO ²³Na
composition series, where good fits had peak_region_r2≈0.998 but global R²≈0.5.
The r_squared path is unchanged.
"""

from __future__ import annotations

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController,
)
from scilink.agents.exp_agents.quality_gate import QualityGate


def _ctrl():
    c = UnifiedSeriesProcessingController.__new__(UnifiedSeriesProcessingController)
    c.r2_threshold = 0.95
    c.outlier_sigma = 3.0
    return c


# (global_r2, peak_region_r2): good low-SNR fits + one genuinely broken (x=0.17).
_NLTO = [(0.84, 0.954), (0.63, 0.998), (0.52, 0.998), (0.56, 0.998),
         (-366407.0, -366407.0), (0.96, 0.966), (0.95, 0.955), (0.94, 0.951)]


def _series(data):
    return [{"index": i, "name": f"x{i}", "success": True,
             "fit_quality": {"r_squared": g, "peak_region_r2": p}}
            for i, (g, p) in enumerate(data)]


def test_gate_metric_avoids_false_flags():
    gate = QualityGate(metric="peak_region_r2", accept_threshold=0.85,
                       hard_reject_threshold=0.30)
    flagged = {f["index"] for f in _ctrl()._detect_outliers(_series(_NLTO), gate=gate)}
    # Only the genuinely broken spectrum (index 4) flags; the good low-SNR fits
    # (1,2,3 with peak_region_r2≈0.998) do NOT.
    assert flagged == {4}


def test_r_squared_path_unchanged():
    # No non-R² gate → legacy global-R² flagging (unchanged behavior).
    flagged = {f["index"] for f in _ctrl()._detect_outliers(_series(_NLTO))}
    assert flagged == {0, 1, 2, 3, 4, 7}


def test_fit_failure_still_flags():
    sr = _series([(0.99, 0.99), (0.98, 0.98), (0.97, 0.97)])
    sr.append({"index": 3, "name": "x3", "success": False, "fit_quality": {}})
    gate = QualityGate(metric="peak_region_r2", accept_threshold=0.85,
                       hard_reject_threshold=0.30)
    flagged = {f["index"]: f["reason"] for f in _ctrl()._detect_outliers(sr, gate=gate)}
    assert flagged.get(3) == "fit_failed"
