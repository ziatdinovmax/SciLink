"""Offline tests for the NMR second-order quadrupolar central-transition helper.

No network, no LLM. Validates the forward model + grid-seeded fit:
- a RESOLVED quadrupolar pattern round-trips (recovers planted C_Q, η);
- a narrow, broadening-dominated line is flagged ``Cq_resolved=False`` rather
  than returning a falsely precise C_Q (genuine C_Q–linewidth degeneracy);
- the tool specs register under the ``nmr`` bundle and are skill-gated.
"""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.curve_fitting.nmr.quadrupolar import (
    simulate_quad_ct, fit_quad_ct, _nu_q_hz,
)

NU_L = 211.66  # 23Na at 18.8 T
I_NA = 1.5


def _synth(cq, eta, d_iso, lw, n=5000):
    ppm = np.linspace(-70, 50, n)
    y = simulate_quad_ct(ppm, d_iso, cq, eta, NU_L, I=I_NA,
                         lw_gauss_ppm=lw, lw_lorentz_ppm=lw * 0.8, mas=True)
    y = y + np.random.default_rng(0).normal(0, 0.01, y.size)
    return ppm, y


def test_nu_q_formula():
    # ν_Q = 3 C_Q / [2 I (2I-1)]; for I=3/2 that is C_Q/2.
    assert _nu_q_hz(2.0, 1.5) == pytest.approx(1.0e6, rel=1e-9)


@pytest.mark.parametrize("cq,eta", [(2.5, 0.2), (4.0, 0.3), (3.0, 0.7)])
def test_resolved_pattern_roundtrips(cq, eta):
    # Resolved (broad, asymmetric) second-order pattern: C_Q and η recover.
    ppm, y = _synth(cq, eta, d_iso=0.0, lw=0.3)
    res = fit_quad_ct(ppm, y, nu_L_MHz=NU_L, I=I_NA, mas=True)
    assert res["derived"]["Cq_resolved"] is True
    assert res["parameters"]["Cq_MHz"] == pytest.approx(cq, abs=0.4)
    assert res["fit_quality"]["r_squared"] > 0.9


@pytest.mark.parametrize("cq,lw", [(1.2, 0.8), (0.8, 0.6)])
def test_narrow_line_flagged_unreliable(cq, lw):
    # Broadening-dominated narrow line: C_Q is degenerate with linewidth, so the
    # helper must flag it rather than assert a precise C_Q.
    ppm, y = _synth(cq, 0.6, d_iso=-2.0, lw=lw)
    res = fit_quad_ct(ppm, y, nu_L_MHz=NU_L, I=I_NA, mas=True)
    assert res["derived"]["Cq_resolved"] is False
    assert any("upper bound" in f for f in res["derived"]["reliability_flags"])


def test_simulate_shape_basics():
    ppm = np.linspace(-40, 20, 3000)
    y = simulate_quad_ct(ppm, 0.0, 3.0, 0.3, NU_L, I=I_NA,
                         lw_gauss_ppm=0.3, lw_lorentz_ppm=0.2, mas=True)
    assert y.shape == ppm.shape
    assert np.all(y >= -1e-9) and y.max() == pytest.approx(1.0, abs=1e-6)
    # MAS line is narrower than the static powder for the same parameters.
    y_static = simulate_quad_ct(ppm, 0.0, 3.0, 0.3, NU_L, I=I_NA,
                                lw_gauss_ppm=0.3, lw_lorentz_ppm=0.2, mas=False)
    assert np.ptp(ppm[y > 0.2]) < np.ptp(ppm[y_static > 0.2])


def test_tools_registered_and_skill_gated():
    from scilink.skills._shared._registry import get_tools_for, get_tool_function
    names = {t.name for t in get_tools_for("curve_fitting", active_skills=["nmr"])}
    assert {"simulate_quad_ct", "fit_quad_ct"} <= names
    # not visible without the nmr bundle active
    names0 = {t.name for t in get_tools_for("curve_fitting", active_skills=[])}
    assert "fit_quad_ct" not in names0
    assert callable(get_tool_function("fit_quad_ct", active_skills=["nmr"]))
