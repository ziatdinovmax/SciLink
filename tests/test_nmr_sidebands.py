"""Tests for the MAS spinning-sideband manifold tool (curve_fitting/nmr)."""

import numpy as np
import pytest

from scilink.skills.curve_fitting.nmr.multipeak import _voigt_height_norm
from scilink.skills.curve_fitting.nmr.sidebands import fit_sideband_manifold

RNG = np.random.default_rng(7)

CENTRE = 5.0        # ppm
MAS = 20.0          # ppm spacing
AMPS = {-2: 0.10, -1: 0.50, 0: 1.00, 1: 0.40, 2: 0.15}


def _manifold(noise=0.003):
    x = np.linspace(-100.0, 100.0, 4001)
    y = np.zeros_like(x)
    for k, a in AMPS.items():
        y += a * _voigt_height_norm(x, CENTRE + k * MAS, 0.8, 0.8)
    y += RNG.normal(0.0, noise, x.size)
    return x, y


def test_recovers_manifold():
    x, y = _manifold()
    res = fit_sideband_manifold(x, y, mas_rate_ppm=MAS)
    assert abs(res["isotropic_shift_ppm"] - CENTRE) < 0.5
    assert res["n_sidebands_each_side"] == 2          # auto-detected
    assert res["fit_quality"]["r_squared"] > 0.98
    # sideband-inclusive quantification: centreband is ~1.0/2.15 of the total
    expected_cb = AMPS[0] / sum(AMPS.values())
    assert abs(res["centreband_fraction"] - expected_cb) < 0.08
    # per-order pattern ordering preserved (|+1| > |+2|, |-1| > |-2|)
    oi = {int(k): abs(v) for k, v in res["order_intensities"].items()}
    assert oi[1] > oi[2] and oi[-1] > oi[-2]
    assert res["manifold_span_ppm"] == pytest.approx(2 * MAS)


def test_knob_n_sidebands_overrides_auto():
    x, y = _manifold()
    res = fit_sideband_manifold(x, y, mas_rate_ppm=MAS, n_sidebands=1)
    assert res["n_sidebands_each_side"] == 1
    assert set(res["order_intensities"]) == {"-1", "0", "1"}


def test_knob_min_amp_snr_gates_auto_detection():
    x, y = _manifold()
    strict = fit_sideband_manifold(x, y, mas_rate_ppm=MAS, min_amp_snr=1e6)
    assert strict["n_sidebands_each_side"] == 0
    assert strict["centreband_fraction"] == pytest.approx(1.0)


def test_centre_seed_when_sideband_is_tallest():
    # Large anisotropy: make the +1 sideband taller than the centreband; the
    # explicit centre_ppm knob must rescue the assignment.
    x = np.linspace(-100.0, 100.0, 4001)
    amps = {-1: 0.6, 0: 0.5, 1: 1.0}
    y = np.zeros_like(x)
    for k, a in amps.items():
        y += a * _voigt_height_norm(x, CENTRE + k * MAS, 0.8, 0.8)
    res = fit_sideband_manifold(x, y, mas_rate_ppm=MAS, centre_ppm=CENTRE)
    assert abs(res["isotropic_shift_ppm"] - CENTRE) < 0.5
    assert res["centreband_fraction"] < 0.5  # sidebands dominate, correctly counted
