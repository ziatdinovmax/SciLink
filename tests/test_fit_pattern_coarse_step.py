"""fit_pattern on a coarsely sampled pattern.

When 2*step exceeds the default init_fwhm_deg, the initial FWHM guess sat
below its own lower bound and least_squares rejected the fit outright
("Initial guess is outside of provided bounds") — reported as a convergence
failure, and in the SNIP 'auto' sweep swallowed into a bare "no SNIP
iteration count converged". The initial guess is now clamped into the
bounds, and the sweep error names the underlying cause.
"""
import numpy as np
import pytest

from scilink.skills.curve_fitting.xrd_profile.fit_pattern import fit_pattern

STEP = 0.1431  # > init_fwhm_deg (0.2) / 2 -> fwhm_lo = 0.286 > 0.2


def _pattern(n=700, fwhm=1.0):
    rng = np.random.default_rng(0)
    x = np.arange(n) * STEP
    sigma = fwhm / 2.3548
    y = 2.2 * np.exp(-0.5 * ((x - 31.0) / sigma) ** 2) + 0.04 * rng.normal(size=n)
    return x, y


def test_coarse_step_fits_instead_of_bounds_rejection():
    x, y = _pattern()
    r = fit_pattern(x.tolist(), y.tolist(), prominence_frac=0.2)
    assert r["r_squared"] > 0.95
    assert len(r["peaks"]) == 1
    assert abs(r["peaks"][0]["center"] - 31.0) < 0.2
    # the fitted width is inside the bounds, not pinned to a rejected p0
    assert r["peaks"][0]["fwhm"] >= 2 * STEP


def test_snip_sweep_error_names_underlying_cause():
    x, y = _pattern()
    # No detectable peaks -> every trial raises; message must carry the cause.
    with pytest.raises(RuntimeError, match="last error"):
        fit_pattern(x.tolist(), (0.01 * np.ones_like(y)).tolist(),
                    prominence_frac=0.9)


def test_step_too_coarse_for_max_fwhm_is_a_clear_error():
    x, y = _pattern()
    # Raised inside the SNIP sweep, so it surfaces through the sweep's
    # RuntimeError — with the cause in the message.
    with pytest.raises((ValueError, RuntimeError), match="too coarse"):
        fit_pattern(x.tolist(), y.tolist(), max_fwhm_deg=0.2)
