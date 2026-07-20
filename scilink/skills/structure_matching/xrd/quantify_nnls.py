"""``quantify_phases_nnls`` — fast linear phase quantification by non-negative
least squares over a shortlist of candidate structures.

Where it sits: identification tells you WHICH phases are present;
``identify_mixture`` does that by a greedy peel-and-subtract over the *peak
list*. This tool does the complementary QUANTITATIVE step over the *continuous
profile*: it fits the measured pattern as a non-negative linear combination of
the candidates' simulated patterns —

    y(2θ)  ≈  Σ  cᵢ · patternᵢ(2θ),   cᵢ ≥ 0

and returns each phase's fraction. Because all candidates compete in ONE joint
fit (rather than being peeled off one at a time), overlapping reflections are
deconvolved natively — the NNLS strength over greedy matching.

It is the light, fast rung of the quantification ladder:

    identify_mixture   →   quantify_phases_nnls   →   refine_multiphase
    (which phases)         (fast fractions, screening)   (rigorous Rietveld QPA)

Two honesty properties, both learned from in-situ practice:

* **Abstention.** NNLS will always return *some* non-negative combination — even
  for a pattern whose true phase is not among the candidates (or not in any
  database), it fits the closest available columns, which can be chemically
  wrong. So the tool checks the fit residual against the data noise and returns
  ``reliable=False`` with a note when the model cannot explain the pattern,
  rather than handing over confident fractions for an incomplete model.
* **Honest units.** The primary output is an INTENSITY fraction (each phase's
  share of the fitted scattering) — a screening quantity, like
  ``identify_mixture``'s intensity_share but from a joint fit. A ZMV-corrected
  ``weight_fraction_est`` is also returned, clearly marked approximate; rigorous
  weight-fraction QPA is ``refine_multiphase`` (Rietveld).

Deterministic and offline (pymatgen kinematic patterns + scipy NNLS); no LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

# np.trapz was renamed np.trapezoid in NumPy 2.0 (old name removed).
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _pseudo_voigt(x: np.ndarray, center: float, fwhm: float, eta: float) -> np.ndarray:
    """Unit-height pseudo-Voigt (eta·Lorentzian + (1-eta)·Gaussian)."""
    hw = 0.5 * fwhm
    g = np.exp(-np.log(2.0) * ((x - center) / hw) ** 2)
    l = 1.0 / (1.0 + ((x - center) / hw) ** 2)
    return eta * l + (1.0 - eta) * g


def _profile_column(structure_path: str, grid: np.ndarray, wavelength,
                    fwhm: float, eta: float) -> np.ndarray:
    """Simulated continuous profile for one structure on the measured 2θ grid:
    broaden each kinematic reflection into a pseudo-Voigt and sum."""
    from .simulate_xrd import simulate_xrd_pattern
    tt_range = (float(grid[0]), float(grid[-1]))
    sim = simulate_xrd_pattern(structure_path, wavelength=wavelength,
                               two_theta_range=tt_range, engine="pymatgen")
    col = np.zeros_like(grid)
    for c, inten in zip(sim["two_theta"], sim["intensities"]):
        if tt_range[0] <= c <= tt_range[1]:
            col += float(inten) * _pseudo_voigt(grid, float(c), fwhm, eta)
    return col


def _zmv(structure_path: str) -> float:
    """Z·M·V proxy (formula units × cell mass × cell volume) for converting an
    intensity contribution to an approximate weight fraction — the Hill–Howard
    QPA relation w_i ∝ S_i·(ZMV)_i. Returns 1.0 on any failure (falls back to
    intensity fractions)."""
    try:
        from pymatgen.core import Structure
        s = Structure.from_file(structure_path)
        # cell mass (amu) × cell volume (Å³); Z is folded into the full-cell mass
        return float(s.composition.weight) * float(s.volume)
    except Exception:
        return 1.0


TOOL_SPEC = ToolSpec(
    name="quantify_phases_nnls",
    description=(
        "Fast quantitative phase analysis by non-negative least squares (NNLS). "
        "Fits the measured continuous pattern as a non-negative combination of a "
        "SHORTLIST of candidate structures' simulated patterns and returns each "
        "phase's fraction. Deconvolves overlapping mixtures in one joint fit "
        "(unlike identify_mixture's greedy peel-and-subtract). The light rung of "
        "quantification: identify_mixture (which phases) → quantify_phases_nnls "
        "(fast fractions) → refine_multiphase (rigorous Rietveld QPA). Run it on "
        "the handful of candidates from search-match / identify_mixture, NOT the "
        "whole database. Returns reliable=False with a note when the residual "
        "exceeds the noise (a phase is likely missing / off-database) rather than "
        "reporting confident fractions for an incomplete model."
    ),
    import_line="from scilink.skills.structure_matching.xrd.quantify_nnls import quantify_phases_nnls",
    signature=(
        "quantify_phases_nnls(two_theta, intensity, structure_paths, "
        "wavelength='CuKa', fwhm_deg=0.15, eta=0.5, background='snip', "
        "snip_iterations=24, min_fraction=0.02, residual_over_noise_max=6.0, "
        "two_theta_range=None) -> dict"
    ),
    parameters={
        "two_theta": {"type": "list[float]", "description": "Measured 2θ grid of the FULL continuous pattern (the raw scan, not a peak list)."},
        "intensity": {"type": "list[float]", "description": "Measured intensities aligned with two_theta (the full scan)."},
        "structure_paths": {"type": "list[str]", "description": "CIF paths of the candidate phases to quantify — the shortlist from search_match_pattern / identify_mixture (materialize_dir CIFs). Keep it small (2–10); NNLS is for a shortlist, not the whole library."},
        "wavelength": {"type": "str | float", "description": "Source name ('CuKa', 'MoKa', …) or wavelength in Å. MUST match the measurement."},
        "fwhm_deg": {"type": "float", "description": "Peak broadening (FWHM, 2θ°) used to build each candidate's profile column. Match the data's peak width: LOWER (e.g. 0.05) for sharp / large-crystallite / synchrotron patterns, RAISE (e.g. 0.4) for broad / nanocrystalline lines. Too-narrow columns miss overlap; too-wide smears distinct phases together. Default 0.15."},
        "eta": {"type": "float", "description": "Pseudo-Voigt mixing of the profile columns: 0 = pure Gaussian, 1 = pure Lorentzian. RAISE toward Lorentzian for long peak tails. Default 0.5."},
        "background": {"type": "str", "description": "Background handling: 'snip' (subtract a SNIP background from the data before the fit — default, robust), 'poly' (add low-order polynomial columns to the fit instead), or 'none' (data already background-free). Use 'snip' unless the data is pre-subtracted."},
        "snip_iterations": {"type": "int", "description": "SNIP clipping window (background='snip' only). RAISE to remove broader background humps (amorphous halo); LOWER if a real broad peak is being eaten as background. Default 24."},
        "min_fraction": {"type": "float", "description": "Drop phases below this fitted intensity fraction and renormalize (parsimony). RAISE (e.g. 0.05) to report only major phases; LOWER (e.g. 0.01) to keep minor phases. Default 0.02."},
        "residual_over_noise_max": {"type": "float", "description": "Abstention threshold: if the worst unexplained feature (99.5th-pct residual) exceeds this multiple of the noise, the model is judged incomplete (a phase likely missing / off-database) and reliable=False. LOWER = stricter (abstain more readily); RAISE = permissive. Default 6.0 (a clean fit sits near 3)."},
        "two_theta_range": {"type": "tuple", "description": "Optional (min, max) 2θ° fit window; default uses the full measured range."},
    },
    required=["two_theta", "intensity", "structure_paths"],
    returns=(
        "dict with 'phases' (list of {structure_path, intensity_fraction, "
        "weight_fraction_est}), 'reliable' (bool), 'residual_over_noise', "
        "'r_factor', 'background', 'note'. intensity_fraction is a screening "
        "quantity (share of fitted scattering); weight_fraction_est is a "
        "ZMV-corrected approximation — use refine_multiphase for rigorous QPA. "
        "When reliable=False, treat the fractions as untrustworthy."
    ),
    when_to_use=(
        "After identify_mixture / search_match_pattern have produced a shortlist "
        "of candidate phases, to get fast quantitative fractions and deconvolve "
        "overlapping mixtures. Escalate to refine_multiphase when rigorous "
        "weight fractions (with esds) are needed."
    ),
)


def quantify_phases_nnls(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    structure_paths: Sequence[str],
    wavelength: Any = "CuKa",
    fwhm_deg: float = 0.15,
    eta: float = 0.5,
    background: str = "snip",
    snip_iterations: int = 24,
    min_fraction: float = 0.02,
    residual_over_noise_max: float = 6.0,
    two_theta_range: Optional[tuple] = None,
) -> dict[str, Any]:
    """Quantify phase fractions by NNLS over a candidate shortlist. See TOOL_SPEC."""
    from scipy.optimize import nnls

    x = np.asarray(two_theta, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.size != y.size or x.size < 8:
        raise ValueError("two_theta and intensity must be aligned arrays of length >= 8.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if two_theta_range is not None:
        m = (x >= two_theta_range[0]) & (x <= two_theta_range[1])
        x, y = x[m], y[m]
    if not structure_paths:
        raise ValueError("structure_paths is empty — supply the candidate shortlist.")

    # --- background ---
    bg_cols: list[np.ndarray] = []
    if background == "snip":
        from ...curve_fitting.xrd_profile.background import _snip
        y_fit = y - _snip(y, iterations=int(snip_iterations))
        y_fit = np.clip(y_fit, 0.0, None)
    elif background == "poly":
        y_fit = y.copy()
        xn = (x - x.mean()) / (float(np.ptp(x)) or 1.0)
        bg_cols = [np.ones_like(x), xn, xn ** 2, xn ** 3]  # low-order polynomial
    elif background == "none":
        y_fit = y.copy()
    else:
        raise ValueError(f"background must be 'snip' | 'poly' | 'none'; got {background!r}")

    # --- design matrix: one simulated profile column per candidate (+bg) ---
    phase_cols = [_profile_column(p, x, wavelength, float(fwhm_deg), float(eta))
                  for p in structure_paths]
    for j, col in enumerate(phase_cols):
        if not np.any(col > 0):
            _logger.warning("candidate %s has no reflections in the fit window", structure_paths[j])
    A = np.column_stack(phase_cols + bg_cols) if bg_cols else np.column_stack(phase_cols)

    # --- solve NNLS ---
    coef, _ = nnls(A, y_fit)
    n_phase = len(phase_cols)
    phase_coef = coef[:n_phase]
    model = A @ coef

    # --- per-phase intensity contribution (integrated) ---
    contrib = np.array([phase_coef[j] * float(_trapz(phase_cols[j], x))
                        for j in range(n_phase)])
    total = float(contrib.sum())
    inten_frac = contrib / total if total > 0 else np.zeros(n_phase)

    # --- ZMV-corrected approximate weight fractions ---
    zmv = np.array([_zmv(p) for p in structure_paths])
    wraw = inten_frac * zmv
    wtot = float(wraw.sum())
    wfrac = wraw / wtot if wtot > 0 else inten_frac

    # --- residual vs noise → abstention ---
    resid = y_fit - model
    # Noise = MAD of the residual (robust to a minority of large residuals). The
    # abstention signal is the WORST unexplained feature (99.5th-percentile |resid|)
    # over that noise — NOT the RMS, which averages a localized missing-phase peak
    # away. A complete fit leaves only noise (~3σ extremes → ratio ≈ 3); a missing
    # phase leaves a full unexplained peak (ratio ≫ 6).
    noise = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) or 1.0
    resid_over_noise = float(np.percentile(np.abs(resid), 99.5) / noise)
    denom = float(np.sum(np.abs(y_fit))) or 1.0
    r_factor = float(np.sum(np.abs(resid)) / denom)
    reliable = resid_over_noise <= float(residual_over_noise_max)

    # --- assemble, drop trace phases, renormalize ---
    phases = []
    for j, p in enumerate(structure_paths):
        if inten_frac[j] >= float(min_fraction):
            phases.append({"structure_path": p,
                           "intensity_fraction": round(float(inten_frac[j]), 4),
                           "weight_fraction_est": round(float(wfrac[j]), 4)})
    kept_i = sum(ph["intensity_fraction"] for ph in phases) or 1.0
    kept_w = sum(ph["weight_fraction_est"] for ph in phases) or 1.0
    for ph in phases:
        ph["intensity_fraction"] = round(ph["intensity_fraction"] / kept_i, 4)
        ph["weight_fraction_est"] = round(ph["weight_fraction_est"] / kept_w, 4)
    phases.sort(key=lambda d: -d["intensity_fraction"])

    note = ("Fit explains the pattern within the noise; fractions are a reliable "
            "screening estimate (intensity-based; escalate to refine_multiphase "
            "for rigorous weight-fraction QPA with esds)."
            if reliable else
            "Fit residual exceeds the data noise — the candidate set does NOT "
            "fully explain the pattern (a phase is likely missing or off-database). "
            "Treat the fractions as unreliable; add candidates or run index_pattern "
            "on the residual before quantifying.")

    return {
        "phases": phases,
        "reliable": reliable,
        "residual_over_noise": round(resid_over_noise, 2),
        "r_factor": round(r_factor, 4),
        "background": background,
        "note": note,
    }
