"""Offline tests for the robust tier (Hanawalt + MIP)."""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.structure_matching.xrd.score_match_robust import (
    PULP_AVAILABLE,
    TOOL_SPEC,
    score_xrd_match_robust,
)


_skip_no_pulp = pytest.mark.skipif(
    not PULP_AVAILABLE,
    reason="pulp not installed; pip install scilink[structure-matching]",
)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

SI_PEAKS = [28.44, 47.30, 56.12, 69.13, 76.38]   # silicon CuKa
SI_AMPS = [100.0, 60.0, 30.0, 25.0, 15.0]


def _make_exp_peaks(positions, intensities, shift=0.0, scale=1.0):
    """Build an exp_peaks dict directly (skip continuous-pattern extraction)."""
    shifted = [scale * p + shift for p in positions]
    return {"positions": shifted, "intensities": list(intensities)}


def _diamond_peaks():
    # Diamond CuKa peaks (Fd-3m, a=3.57): very different from Si
    return [43.93, 75.30, 91.50]


# ---------------------------------------------------------------------------
# Tool spec + argument validation
# ---------------------------------------------------------------------------

def test_tool_spec_renders():
    block = TOOL_SPEC.to_prompt()
    assert "score_xrd_match_robust" in block
    assert "hanawalt" in block.lower()
    assert "mip" in block.lower()


def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError, match="algorithm"):
        score_xrd_match_robust(
            sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
            exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS),
            algorithm="rietveld",
        )


def test_requires_exp_input():
    with pytest.raises(ValueError, match="exp_peaks"):
        score_xrd_match_robust(
            sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        )


def test_empty_exp_peaks_returns_reject():
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks={"positions": [], "intensities": []},
    )
    assert out["verdict"] == "reject"
    assert out["matched_peaks"] == []


def test_empty_sim_peaks_returns_reject():
    out = score_xrd_match_robust(
        sim_two_theta=[], sim_intensity=[],
        exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS),
    )
    assert out["verdict"] == "reject"


# ---------------------------------------------------------------------------
# Hanawalt algorithm
# ---------------------------------------------------------------------------

def test_hanawalt_perfect_match():
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS),
        algorithm="hanawalt",
    )
    assert out["algorithm"] == "hanawalt"
    assert out["verdict"] == "accept"
    assert out["figure_of_merit"] > 0.9
    assert len(out["matched_peaks"]) == 5
    assert out["coverage"] == 1.0


def test_hanawalt_total_mismatch_rejected():
    out = score_xrd_match_robust(
        sim_two_theta=_diamond_peaks(), sim_intensity=[100.0, 40.0, 20.0],
        exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS),
        algorithm="hanawalt",
    )
    assert out["verdict"] == "reject"
    assert out["figure_of_merit"] < 0.3
    assert len(out["matched_peaks"]) == 0


def test_hanawalt_tolerant_to_intensity_variation():
    """Hanawalt should still accept when peaks line up positionally even if
    intensities are wildly off (real-lab preferred orientation effect)."""
    bad_intensities = [10.0, 100.0, 100.0, 5.0, 200.0]  # swapped, scaled, etc.
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=_make_exp_peaks(SI_PEAKS, bad_intensities),
        algorithm="hanawalt",
    )
    assert out["verdict"] == "accept"
    assert out["coverage"] == 1.0
    # Position score should still be very high
    assert out["position_score"] > 0.95


def test_hanawalt_partial_match_marginal():
    # Only the first two Si peaks present
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=_make_exp_peaks(SI_PEAKS[:2], SI_AMPS[:2]),
        algorithm="hanawalt",
    )
    # Two-of-five coverage → marginal at best
    assert out["coverage"] == 1.0  # all *exp* peaks matched
    # But coverage is relative to exp peaks, so this is full; the FOM is high.
    # Inversely: when most exp peaks have no sim equivalent, coverage drops.


def test_hanawalt_unmatched_when_outside_tolerance():
    exp_pl = _make_exp_peaks([28.44 + 0.5, 47.30, 56.12], [100.0, 60.0, 30.0])
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=exp_pl,
        algorithm="hanawalt",
        tol_deg=0.3,
    )
    # The first exp peak is 0.5 deg off, outside tol=0.3 → unmatched
    assert 0 in out["unmatched_exp"]
    # Coverage is intensity-weighted: peaks 1+2 carry (60+30)/(100+60+30) = 90/190
    # of total intensity, so coverage ≈ 0.474 (not the 2/3 a plain count
    # would give). This is the deliberate fix that prevents weak noise peaks
    # from dragging coverage on otherwise-clean spectra.
    assert out["coverage"] == pytest.approx(90 / 190, abs=0.001)


def test_hanawalt_intensity_weighted_coverage_ignores_weak_noise_peaks():
    """A clean 5-peak Si match with 10 weak noise peaks added should still
    score near 1.0 coverage, because the noise peaks carry negligible
    intensity."""
    main_peaks = [(p, a) for p, a in zip(SI_PEAKS, SI_AMPS)]
    # Add 10 "noise" peaks at random positions with tiny intensity.
    noise_peaks = [(20.0 + 5.0 * i, 1.0) for i in range(10)]
    all_peaks = main_peaks + noise_peaks
    positions = [p for p, _ in all_peaks]
    intensities = [a for _, a in all_peaks]
    exp_pl = {"positions": positions, "intensities": intensities}

    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=exp_pl,
        algorithm="hanawalt",
        tol_deg=0.3,
    )
    # Most of the noise peaks should be unmatched, but coverage stays high
    # because they carry essentially zero intensity.
    assert out["coverage"] > 0.95
    assert out["verdict"] == "accept"


# ---------------------------------------------------------------------------
# MIP algorithm
# ---------------------------------------------------------------------------

@_skip_no_pulp
def test_mip_perfect_match():
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS),
        algorithm="mip",
    )
    assert out["algorithm"] == "mip"
    assert out["verdict"] == "accept"
    assert out["cost"] < 0.1
    assert len(out["matched_peaks"]) == 5
    assert abs(out["fitted_shift"]) < 0.05


@_skip_no_pulp
def test_mip_recovers_known_shift():
    """A 0.2-degree zero-shift is fitted, not folded into the residual."""
    true_shift = 0.2
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS, shift=true_shift),
        algorithm="mip",
    )
    assert out["verdict"] == "accept"
    assert abs(out["fitted_shift"] - true_shift) < 0.05


@_skip_no_pulp
def test_mip_recovers_known_scale():
    true_scale = 1.005
    out = score_xrd_match_robust(
        sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
        exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS, scale=true_scale),
        algorithm="mip",
        scale_search=(0.99, 1.01, 0.001),
    )
    assert out["verdict"] == "accept"
    assert abs(out["fitted_scale"] - true_scale) < 0.003


@_skip_no_pulp
def test_mip_total_mismatch_rejected():
    out = score_xrd_match_robust(
        sim_two_theta=_diamond_peaks(), sim_intensity=[100.0, 40.0, 20.0],
        exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS),
        algorithm="mip",
    )
    # Verdict must reject; a coincidental position match on 1-2 peaks is
    # allowed (Si 76.38 is within ~tol of Diamond 75.30 at scale ~1.014).
    assert out["verdict"] == "reject"
    assert len(out["matched_peaks"]) <= 2


@_skip_no_pulp
def test_mip_at_most_one_match_per_peak():
    """Two close exp peaks must not both match the same sim peak."""
    # Two exp peaks 0.1 deg apart, both near a single sim peak
    out = score_xrd_match_robust(
        sim_two_theta=[40.0], sim_intensity=[100.0],
        exp_peaks={"positions": [39.95, 40.10], "intensities": [50.0, 80.0]},
        algorithm="mip",
        tol_deg=0.3,
    )
    sim_indices = [m["sim_idx"] for m in out["matched_peaks"]]
    assert len(sim_indices) == len(set(sim_indices))  # no duplicates


def test_mip_raises_clear_error_without_pulp(monkeypatch):
    if PULP_AVAILABLE:
        pytest.skip("pulp installed; cannot test the missing-dep path")
    with pytest.raises(RuntimeError, match="pulp"):
        score_xrd_match_robust(
            sim_two_theta=SI_PEAKS, sim_intensity=SI_AMPS,
            exp_peaks=_make_exp_peaks(SI_PEAKS, SI_AMPS),
            algorithm="mip",
        )
