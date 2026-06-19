"""Fit-free planning-stage structure scout (analysis-agnostic).

Verifies it surfaces hard-to-resolve LOCAL structure (squashed shoulders,
unresolved doublets, edges) for ANY curve type, and stays quiet on flat data.
"""
import numpy as np
import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    _extract_xy, _data_structure_diagnostics, _render_data_zoom_panels,
)

rng = np.random.default_rng(0)


def _g(x, a, c, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def _windows_cover(diag, x):
    return any(w["x_lo"] <= x <= w["x_hi"] for w in (diag or {}).get("worst_windows", []))


def test_squashed_shoulder_is_flagged():
    # A tall peak at 50 (amp 100) with a SMALL shoulder at 72 (amp 6) — the
    # shoulder is squashed in the full view but is real structure.
    x = np.linspace(0, 100, 1000)
    y = _g(x, 100, 50, 4) + _g(x, 6, 72, 3) + 0.5 + rng.normal(0, 0.3, x.size)
    diag = _data_structure_diagnostics(x, y)
    assert diag and diag["worst_windows"]
    assert _windows_cover(diag, 72), "shoulder region should be flagged"
    # a flat region far from any feature is not the top window
    assert not (diag["worst_windows"][0]["x_lo"] <= 15 <= diag["worst_windows"][0]["x_hi"])


def test_unresolved_doublet_has_sign_changes():
    x = np.linspace(0, 100, 1000)
    y = _g(x, 20, 48, 3) + _g(x, 18, 56, 3) + rng.normal(0, 0.2, x.size)
    diag = _data_structure_diagnostics(x, y)
    assert _windows_cover(diag, 52)  # the doublet region
    # the doublet window shows systematic oscillation
    top = max(diag["worst_windows"], key=lambda w: w["rms_over_noise"])
    assert top["sign_changes"] >= 1


def test_edge_step_is_flagged():
    # A sigmoid edge (no peaks) — structure is the edge, not a maximum.
    x = np.linspace(0, 100, 1000)
    y = 10.0 / (1 + np.exp(-(x - 55) / 1.5)) + rng.normal(0, 0.05, x.size)
    diag = _data_structure_diagnostics(x, y)
    assert diag and diag["worst_windows"]
    assert _windows_cover(diag, 55), "edge region should be flagged"


def test_flat_noise_flags_nothing():
    x = np.linspace(0, 100, 1000)
    y = 5.0 + rng.normal(0, 0.3, x.size)
    diag = _data_structure_diagnostics(x, y)
    assert diag is not None
    assert diag["worst_windows"] == []  # no structure above the noise floor


def test_pure_line_flags_nothing():
    x = np.linspace(0, 100, 1000)
    y = 2.0 + 0.05 * x + rng.normal(0, 0.1, x.size)
    diag = _data_structure_diagnostics(x, y)
    assert diag["worst_windows"] == []  # a line is locally quadratic -> no detail


def test_extract_xy_shapes():
    assert _extract_xy(np.arange(10.0)) is not None
    assert _extract_xy(np.zeros((10, 2))) is not None
    assert _extract_xy(np.zeros((2, 10))) is not None
    assert _extract_xy(np.zeros((5, 10))) is None  # a stack -> not a single curve


def test_render_panels_smoke():
    x = np.linspace(0, 100, 1000)
    y = _g(x, 100, 50, 4) + _g(x, 6, 72, 3) + rng.normal(0, 0.3, x.size)
    diag = _data_structure_diagnostics(x, y)
    panels = _render_data_zoom_panels(x, y, diag)
    assert panels and all(isinstance(p[1], (bytes, bytearray)) and p[1][:4] == b"\x89PNG"
                          for p in panels)
