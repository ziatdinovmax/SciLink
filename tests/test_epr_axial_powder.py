"""Tests for the EPR axial-powder simulator and fitter.

Narrow physical-sanity coverage. The full ``fit_axial_powder`` is also
smoke-tested against a synthetic spectrum so a refactor that breaks the
optimizer or the model shape gets caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from scilink.skills.curve_fitting.epr.axial_powder import (
    fit_axial_powder,
    simulate_axial_powder,
)


# CODATA constants used to verify the resonance condition independently.
_H = 6.62607015e-34
_MU_B = 9.2740100783e-24


def _B_for_g(g: float, nu_Hz: float) -> float:
    """B in Gauss for a given g and microwave frequency."""
    return (_H * nu_Hz) / (_MU_B * g) * 1e4


class TestForwardModelSanity:
    @pytest.fixture
    def B(self):
        return np.linspace(2400.0, 4500.0, 4096)

    def test_no_hyperfine_centers_on_g_resonance(self, B):
        # I_nuc=0 → one transition per orientation, axial powder pattern
        # peaks/valleys around g_par and g_perp. The derivative crosses
        # zero near the g-effective absorption peak.
        sim = simulate_axial_powder(
            B, g_par=2.30, g_perp=2.06, A_par_G=0.0,
            lw_G=20.0, nu_GHz=9.64, I_nuc=0.0,
        )
        # Argmin is the negative-going peak of the perpendicular component
        # (which dominates the spectral envelope for axial g_par > g_perp).
        B_perp = _B_for_g(2.06, 9.64e9)
        # The negative-going peak of the derivative sits just above B_perp
        # (within ~half the linewidth). Be generous on the tolerance.
        argmin_B = B[int(np.argmin(sim))]
        assert abs(argmin_B - B_perp) < 30.0, (
            f"argmin B={argmin_B:.1f} should be near B(g_perp)={B_perp:.1f}"
        )

    def test_hyperfine_quartet_count(self, B):
        # Cu(II) parallel hyperfine = 4 lines. Look at the parallel
        # region (2400-3200 G) and count the local maxima.
        from scipy.signal import find_peaks
        sim = simulate_axial_powder(
            B, g_par=2.30, g_perp=2.06, A_par_G=180.0,
            lw_G=15.0, nu_GHz=9.64, I_nuc=1.5,
        )
        mask = (B >= 2500) & (B <= 3200)
        peaks, _ = find_peaks(sim[mask], prominence=0.01)
        # Cu(II) at A_par = 180 G gives 4 well-resolved parallel-region
        # maxima in the derivative. Allow 3-5 to tolerate the edge of the
        # window cutting off the outermost line.
        assert 3 <= len(peaks) <= 5, f"expected ~4 hyperfine peaks, found {len(peaks)}"

    def test_hyperfine_spacing_equals_A_par(self, B):
        # Adjacent parallel maxima should be separated by A_par.
        from scipy.signal import find_peaks
        A = 170.0
        sim = simulate_axial_powder(
            B, g_par=2.30, g_perp=2.06, A_par_G=A,
            lw_G=12.0, nu_GHz=9.64, I_nuc=1.5,
        )
        mask = (B >= 2500) & (B <= 3200)
        Bw = B[mask]; yw = sim[mask]
        peaks, props = find_peaks(yw, prominence=0.01)
        if len(peaks) < 3:
            pytest.skip("Fewer than 3 hyperfine peaks resolved at these params")
        # Take the top-3 by prominence, sort by field, look at mean gap.
        top = peaks[np.argsort(props["prominences"])[-3:]]
        top = np.sort(top)
        gaps = np.diff(Bw[top])
        assert abs(np.mean(gaps) - A) < 0.10 * A, (
            f"mean hyperfine gap {np.mean(gaps):.1f} G "
            f"deviates from A_par={A} G by > 10%"
        )

    def test_isotropic_component_centers_on_iso_g(self, B):
        # With Cu turned down low and a strong isotropic component, the
        # zero-crossing of the spectrum should match B(iso_g).
        sim = simulate_axial_powder(
            B, g_par=2.30, g_perp=2.06, A_par_G=170.0,
            lw_G=200.0,                          # smear Cu component
            nu_GHz=9.64, I_nuc=1.5,
            amplitude=0.05,                       # Cu nearly invisible
            iso_amplitude=1.0, iso_g=2.0030, iso_lw_G=10.0,
        )
        B_iso = _B_for_g(2.0030, 9.64e9)
        # Derivative zero-crossing between its global max and min
        imax, imin = int(np.argmax(sim)), int(np.argmin(sim))
        lo, hi = sorted([imax, imin])
        seg_B = B[lo:hi + 1]; seg_y = sim[lo:hi + 1]
        cross_idx = np.where(np.diff(np.sign(seg_y)) != 0)[0]
        assert cross_idx.size, "isotropic line should have a derivative zero-crossing"
        # Interpolate to find the crossing
        i = cross_idx[0]
        B_cross = seg_B[i] - seg_y[i] * (seg_B[i+1] - seg_B[i]) / (seg_y[i+1] - seg_y[i])
        assert abs(B_cross - B_iso) < 5.0, (
            f"isotropic zero-crossing {B_cross:.1f} G vs expected {B_iso:.1f} G"
        )


class TestFitRoundtrip:
    def test_recovers_synthetic_cu_parameters(self):
        # Forward-simulate a clean Cu(II) spectrum, fit it back, check
        # the optimizer recovers the input parameters within ~1%.
        rng = np.random.default_rng(0)
        B = np.linspace(2400.0, 4500.0, 4096)
        truth = dict(g_par=2.32, g_perp=2.07, A_par_G=175.0, lw_G=35.0)
        y_clean = simulate_axial_powder(B, **truth, nu_GHz=9.64, amplitude=1.0)
        # Tiny Gaussian noise (~1% of peak-to-peak) so optimizer has
        # something realistic to chew on but parameters are still well-
        # determined.
        y = y_clean + 0.01 * (y_clean.max() - y_clean.min()) * rng.standard_normal(B.size)

        out = fit_axial_powder(
            B.tolist(), y.tolist(),
            nu_GHz=9.64, include_isotropic=False,
            g_par_init=2.30, g_perp_init=2.06,
            A_par_G_init=170.0, lw_G_init=40.0,
        )
        p = out["parameters"]["main"]
        assert out["fit_quality"]["r_squared"] > 0.95
        assert abs(p["g_par"]  - truth["g_par"])    < 0.01
        assert abs(p["g_perp"] - truth["g_perp"])   < 0.01
        assert abs(p["A_par_G"] - truth["A_par_G"]) < 5.0
        assert abs(p["lw_G"]   - truth["lw_G"])     < 5.0


class TestToolDiscovery:
    """The skill's tools must be discoverable by the registry, otherwise the
    agent loads the skill markdown but can never *call* fit_axial_powder and
    falls back to mis-specified codegen. The registry only recognizes the
    ``TOOL_SPEC`` / ``TOOL_SPECS`` attribute names — a module declaring specs
    under other names (e.g. ``TOOL_SPEC_FIT``) is silently skipped.
    """

    def test_epr_tools_registered(self):
        from scilink.skills._shared._registry import _per_skill_specs

        specs = _per_skill_specs().get(("curve_fitting", "epr"))
        assert specs, "EPR skill declares no discoverable ToolSpecs"
        names = {s.name for s in specs}
        assert {"simulate_axial_powder", "fit_axial_powder"} <= names

    def test_epr_tools_visible_when_skill_active(self):
        from scilink.skills._shared._registry import get_tools_for

        names = {t.name for t in get_tools_for(
            "curve_fitting", active_skills=["epr"])}
        assert {"simulate_axial_powder", "fit_axial_powder"} <= names
