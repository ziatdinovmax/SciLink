"""Offline tests for the extract_peaks helper."""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd.extract_peaks import (
    TOOL_SPEC,
    extract_peaks,
)


def _lorentzian_pattern(peak_positions, peak_amps, grid=None, fwhm=0.15, noise=0.0):
    if grid is None:
        grid = np.arange(10.0, 90.0, 0.02)
    gamma = fwhm / 2.0
    y = np.zeros_like(grid)
    for x0, amp in zip(peak_positions, peak_amps):
        y += amp * (gamma ** 2) / ((grid - x0) ** 2 + gamma ** 2)
    if noise:
        rng = np.random.default_rng(0)
        y = y + rng.normal(scale=noise * y.max(), size=y.shape)
    return grid.tolist(), y.tolist()


def test_tool_spec_renders():
    block = TOOL_SPEC.to_prompt()
    assert "extract_peaks" in block
    assert "scipy.signal.find_peaks" in block


def test_finds_three_silicon_peaks():
    peaks = [28.44, 47.30, 56.12]
    amps = [100.0, 60.0, 30.0]
    grid, intensity = _lorentzian_pattern(peaks, amps)
    out = extract_peaks(grid, intensity, prominence_frac=0.01)
    assert len(out["positions"]) == 3
    found_sorted = sorted(out["positions"])
    for found, expected in zip(found_sorted, peaks):
        assert abs(found - expected) < 0.05


def test_sorts_by_descending_intensity():
    grid, intensity = _lorentzian_pattern([30.0, 50.0, 70.0], [40.0, 100.0, 80.0])
    out = extract_peaks(grid, intensity, prominence_frac=0.01)
    assert out["intensities"] == sorted(out["intensities"], reverse=True)
    # Strongest peak (at 50 deg, amp 100) should be first
    assert abs(out["positions"][0] - 50.0) < 0.05


def test_max_peaks_truncates():
    positions = [20.0 + 2.0 * i for i in range(20)]
    amps = [100.0 - i for i in range(20)]
    grid, intensity = _lorentzian_pattern(positions, amps)
    out = extract_peaks(grid, intensity, prominence_frac=0.005, max_peaks=5)
    assert len(out["positions"]) == 5
    # Truncation must keep the strongest five
    assert all(p < 30.0 for p in out["positions"][:5])


def test_returns_fwhm_when_refine_true():
    grid, intensity = _lorentzian_pattern([40.0], [100.0], fwhm=0.3)
    out = extract_peaks(grid, intensity, refine=True)
    assert len(out["fwhms"]) == 1
    # Fit may not be exact but should be in the ballpark
    assert 0.1 < out["fwhms"][0] < 0.6


def test_fwhms_are_zero_when_refine_false():
    grid, intensity = _lorentzian_pattern([40.0], [100.0])
    out = extract_peaks(grid, intensity, refine=False)
    assert out["fwhms"] == [0.0]


def test_sub_pixel_position_refinement():
    """Parabolic refinement should improve position estimate over the bare grid sample."""
    grid_step = 0.05
    grid = np.arange(20.0, 60.0, grid_step)
    true_pos = 30.0 + grid_step * 0.4  # peak between two grid points
    fwhm = 0.2
    gamma = fwhm / 2
    intensity = 100 * gamma ** 2 / ((grid - true_pos) ** 2 + gamma ** 2)
    out = extract_peaks(grid.tolist(), intensity.tolist())
    assert len(out["positions"]) == 1
    # Without refinement, the best we'd do is grid_step/2 = 0.025 error.
    # With refinement we expect order-of-magnitude better.
    assert abs(out["positions"][0] - true_pos) < 0.01


def test_returns_empty_when_no_peaks_above_prominence():
    grid = np.linspace(10, 90, 1000).tolist()
    intensity = (np.ones(1000) * 5.0).tolist()  # flat
    out = extract_peaks(grid, intensity, prominence_frac=0.1)
    assert out == {"positions": [], "intensities": [], "fwhms": [], "prominences": []}


def test_min_distance_suppresses_close_doublets():
    """Two peaks 0.1deg apart with min_distance_deg=0.5 should collapse to one."""
    grid, intensity = _lorentzian_pattern([40.0, 40.1], [100.0, 90.0], fwhm=0.05)
    out = extract_peaks(grid, intensity, min_distance_deg=0.5)
    assert len(out["positions"]) == 1


def test_validates_arguments():
    grid = np.linspace(10, 90, 1000).tolist()
    intensity = np.zeros(1000).tolist()

    with pytest.raises(ValueError, match="same length"):
        extract_peaks(grid, intensity[:10])
    with pytest.raises(ValueError, match="prominence_frac"):
        extract_peaks(grid, intensity, prominence_frac=1.5)
    with pytest.raises(ValueError, match="min_distance_deg"):
        extract_peaks(grid, intensity, min_distance_deg=-0.1)
    with pytest.raises(ValueError, match="max_peaks"):
        extract_peaks(grid, intensity, max_peaks=0)
    with pytest.raises(ValueError, match="background"):
        extract_peaks(grid, intensity, background="weird")
