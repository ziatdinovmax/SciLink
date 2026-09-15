"""``simulate_axial_powder`` and ``fit_axial_powder`` — minimal CW-EPR
axial powder simulator and fitter for S = 1/2 systems at X-band.

Scope (v0 sketch):
  - Axial g-tensor (g_par, g_perp).
  - Axial hyperfine on the parallel axis only (A_par); A_perp absorbed
    into the orientation-averaged linewidth. Default I = 3/2 (Cu).
  - Optional second species: an isotropic narrow line at user-fixed g
    (intended for organic radical / defect contamination near g ≈ 2).
  - First-derivative output (dchi"/dB) on a Gauss field grid.

Out of scope (deferred — see ``epr.md`` "Out of scope" section):
  - Rhombic g (full three-axis powder integration).
  - Resolved A_perp; quadrupolar coupling; multi-nuclear hyperfine.
  - Frequency bands other than X-band (no resonance-condition relativistic
    correction; an explicit ``nu_GHz`` kwarg lets a caller experiment).
  - Saturation modeling (power-dependent linewidth).
  - Multi-species deconvolution beyond Cu(II) + one isotropic radical.

Reuses ``load_curve_data`` from ``_shared/curve_fitting_tools.py`` upstream
of any call here — this module takes raw arrays and returns raw arrays.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

# Physical constants (CODATA)
_H = 6.62607015e-34       # J·s
_MU_B = 9.2740100783e-24  # J/T


# --------------------------------------------------------------------------
#  Forward model
# --------------------------------------------------------------------------

def _gaussian_derivative(B: np.ndarray, B0: float, sigma: float) -> np.ndarray:
    """First derivative of a Gaussian lineshape at B0 with std sigma (Gauss).

    Analytic form so we don't double-broaden by computing the absorption
    on a sub-sampled grid and finite-differencing it.
    """
    z = (B - B0) / sigma
    return -(z / sigma) * np.exp(-0.5 * z * z)


def _axial_resonance_fields(
    nu_Hz: float,
    g_par: float,
    g_perp: float,
    A_par_G: float,
    n_theta: int,
    mI_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (B_res, weight) for the axial-powder grid × hyperfine lines.

    Returns flat arrays so the caller can broadcast a derivative-line
    shape across them. The orientation weight is sin(theta) dθ (powder
    isotropy).
    """
    theta = np.linspace(0.0, 0.5 * np.pi, n_theta)
    sin_t = np.sin(theta)
    cos2 = np.cos(theta) ** 2
    sin2 = np.sin(theta) ** 2
    g_theta = np.sqrt(g_par * g_par * cos2 + g_perp * g_perp * sin2)
    # Orientation-projected A along the resonance direction (first-order;
    # A_perp absorbed into the linewidth, see module docstring).
    A_theta = A_par_G * (g_par / g_theta) ** 2 * np.cos(theta)

    # Field for g·μ_B·B = h·ν - A·m_I, in Gauss (factor 1e4 from Tesla).
    B_g = (_H * nu_Hz) / (_MU_B * g_theta) * 1e4  # shape (n_theta,)

    # Outer product across hyperfine quantum numbers
    mI = np.asarray(mI_values, dtype=float)
    B_res = B_g[:, None] - A_theta[:, None] * mI[None, :]   # (n_theta, n_mI)
    weight = np.broadcast_to(sin_t[:, None], B_res.shape)   # (n_theta, n_mI)
    return B_res.ravel(), weight.ravel()


def simulate_axial_powder(
    B: Sequence[float],
    g_par: float,
    g_perp: float,
    A_par_G: float,
    lw_G: float,
    nu_GHz: float = 9.64,
    I_nuc: float = 1.5,
    n_theta: int = 181,
    iso_amplitude: float = 0.0,
    iso_g: float = 2.0030,
    iso_lw_G: float = 8.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Simulate a first-derivative CW-EPR powder spectrum.

    The Cu(II)-class default (``I_nuc=1.5``) gives 2I+1 = 4 hyperfine
    components on the parallel axis. ``iso_amplitude > 0`` adds an
    isotropic narrow line — meant for organic-radical contamination near
    g ≈ 2, not for a second axial species.

    Args:
        B: Field grid in Gauss. Must be monotonic (either direction).
        g_par, g_perp: Axial g-tensor principal values.
        A_par_G: Parallel hyperfine coupling in Gauss.
        lw_G: Gaussian FWHM linewidth in Gauss applied to every
            orientation/transition before powder summation.
        nu_GHz: Microwave frequency. X-band default 9.64 GHz.
        I_nuc: Nuclear spin. 3/2 → Cu/Co/Mn-class quartet; 0 → no
            hyperfine; 1/2 → doublet.
        n_theta: Number of θ samples in [0, π/2]. 181 = 0.5° resolution,
            sufficient for X-band Cu spectra at typical lw ≥ 20 G. Bump
            to 361 if features look "grainy".
        iso_amplitude: Amplitude of the optional isotropic second
            component (in the same units as the main spectrum, set 0
            to disable).
        iso_g: g-value of the isotropic component (default 2.0030,
            organic radical region).
        iso_lw_G: Linewidth of the isotropic component (Gaussian FWHM).
        amplitude: Overall scale of the Cu-like main component.

    Returns:
        d_chi_over_dB on the input field grid.
    """
    B = np.asarray(B, dtype=float)
    if B.size < 2:
        raise ValueError("B must have at least 2 points")

    # FWHM → Gaussian sigma
    sigma = lw_G / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Axial powder
    n_mI = int(round(2 * I_nuc + 1))
    if n_mI < 1:
        raise ValueError(f"I_nuc={I_nuc} produces no hyperfine lines")
    mI_vals = np.arange(-I_nuc, I_nuc + 1e-9, 1.0)  # robust for half-integer
    B_res, weight = _axial_resonance_fields(
        nu_Hz=nu_GHz * 1e9,
        g_par=g_par, g_perp=g_perp,
        A_par_G=A_par_G, n_theta=n_theta,
        mI_values=mI_vals,
    )

    # Sum derivative-lineshape contributions (vectorized across grid)
    # spec[i] = Σ_k weight[k] · g'(B[i] - B_res[k]; σ)
    # Broadcast: (Ng, Nk) ~ Ng*Nk floats. For 1200 × 724 = ~10^6 — fine.
    dB = B[:, None] - B_res[None, :]
    contrib = -(dB / (sigma * sigma)) * np.exp(-0.5 * (dB / sigma) ** 2)
    spec = (contrib * weight[None, :]).sum(axis=1)

    # Normalize Cu-like component to unit peak-to-peak amplitude before
    # scaling, so `amplitude` is interpretable across calls.
    pp = float(np.max(spec) - np.min(spec))
    if pp > 0:
        spec = amplitude * spec / pp

    # Optional isotropic radical
    if iso_amplitude > 0.0:
        sigma_iso = iso_lw_G / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        B_iso = (_H * nu_GHz * 1e9) / (_MU_B * iso_g) * 1e4
        d_iso = _gaussian_derivative(B, B_iso, sigma_iso)
        pp_iso = float(np.max(d_iso) - np.min(d_iso))
        if pp_iso > 0:
            spec = spec + iso_amplitude * d_iso / pp_iso

    return spec


# --------------------------------------------------------------------------
#  Fit
# --------------------------------------------------------------------------

def fit_axial_powder(
    B: Sequence[float],
    y: Sequence[float],
    nu_GHz: float = 9.64,
    I_nuc: float = 1.5,
    g_par_init: float = 2.30,
    g_perp_init: float = 2.06,
    A_par_G_init: float = 170.0,
    lw_G_init: float = 40.0,
    include_isotropic: bool = False,
    iso_g_init: float = 2.0030,
    iso_lw_G_init: float = 10.0,
    bounds_g_par: tuple = (2.10, 2.50),
    bounds_g_perp: tuple = (2.00, 2.20),
    bounds_A_par_G: tuple = (50.0, 250.0),
    bounds_lw_G: tuple = (5.0, 200.0),
    n_theta: int = 181,
) -> dict[str, Any]:
    """Least-squares fit of an axial powder spectrum.

    Optionally adds a narrow isotropic second component (intended for an
    organic-radical contribution near g ≈ 2, not a second metal species).
    Bounds are conservative defaults for Cu(II) at X-band — override per
    sample chemistry. Free parameters and their order:

    Always: ``amplitude, g_par, g_perp, A_par_G, lw_G``
    When ``include_isotropic`` True: ``+ iso_amplitude, iso_g, iso_lw_G``

    Returns:
        dict with ``parameters`` (per-component sub-dicts), ``fit_quality``
        (``r_squared``, ``rmse``), ``y_fit``, ``derived``
        (``g_avg`` = (g_par + 2·g_perp) / 3 — the isotropic-equivalent
        g-value), and ``model_used`` (string).
    """
    B = np.asarray(B, dtype=float)
    y = np.asarray(y, dtype=float)
    if B.shape != y.shape:
        raise ValueError(f"B and y shape mismatch: {B.shape} vs {y.shape}")

    # Normalize y to a peak-to-peak amplitude of 1 so the amplitude
    # parameter is interpretable; remember the scale to restore later.
    y_pp = float(np.max(y) - np.min(y))
    if y_pp <= 0:
        raise ValueError("y has zero peak-to-peak amplitude")
    y_scaled = y / y_pp

    if include_isotropic:
        p0 = [1.0, g_par_init, g_perp_init, A_par_G_init, lw_G_init,
              0.2, iso_g_init, iso_lw_G_init]
        lo = [0.05, bounds_g_par[0], bounds_g_perp[0], bounds_A_par_G[0], bounds_lw_G[0],
              0.0, 1.98, 2.0]
        hi = [10.0, bounds_g_par[1], bounds_g_perp[1], bounds_A_par_G[1], bounds_lw_G[1],
              2.0, 2.05, 50.0]
    else:
        p0 = [1.0, g_par_init, g_perp_init, A_par_G_init, lw_G_init]
        lo = [0.05, bounds_g_par[0], bounds_g_perp[0], bounds_A_par_G[0], bounds_lw_G[0]]
        hi = [10.0, bounds_g_par[1], bounds_g_perp[1], bounds_A_par_G[1], bounds_lw_G[1]]

    def _model(p):
        if include_isotropic:
            amp, gp, gpp, Ap, lw, ia, ig, ilw = p
            return simulate_axial_powder(
                B, g_par=gp, g_perp=gpp, A_par_G=Ap, lw_G=lw,
                nu_GHz=nu_GHz, I_nuc=I_nuc, n_theta=n_theta,
                amplitude=amp,
                iso_amplitude=ia, iso_g=ig, iso_lw_G=ilw,
            )
        amp, gp, gpp, Ap, lw = p
        return simulate_axial_powder(
            B, g_par=gp, g_perp=gpp, A_par_G=Ap, lw_G=lw,
            nu_GHz=nu_GHz, I_nuc=I_nuc, n_theta=n_theta,
            amplitude=amp,
        )

    def _residual(p):
        return _model(p) - y_scaled

    result = least_squares(
        _residual, x0=p0, bounds=(lo, hi),
        method="trf", max_nfev=5000, ftol=1e-9, xtol=1e-9, gtol=1e-9,
    )

    y_fit_scaled = _model(result.x)
    y_fit = y_fit_scaled * y_pp
    resid = y - y_fit
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(resid * resid)))

    if include_isotropic:
        amp, gp, gpp, Ap, lw, ia, ig, ilw = result.x
        params = {
            "main": {"amplitude": float(amp), "g_par": float(gp),
                     "g_perp": float(gpp), "A_par_G": float(Ap),
                     "lw_G": float(lw)},
            "isotropic": {"amplitude": float(ia), "g": float(ig),
                          "lw_G": float(ilw)},
        }
    else:
        amp, gp, gpp, Ap, lw = result.x
        params = {
            "main": {"amplitude": float(amp), "g_par": float(gp),
                     "g_perp": float(gpp), "A_par_G": float(Ap),
                     "lw_G": float(lw)},
        }

    return {
        "parameters": params,
        "fit_quality": {"r_squared": r2, "rmse": rmse},
        "derived": {"g_avg": (params["main"]["g_par"] + 2.0 * params["main"]["g_perp"]) / 3.0},
        "y_fit": y_fit.tolist(),
        "model_used": ("axial Cu-class powder + isotropic line"
                       if include_isotropic else "axial Cu-class powder"),
        "n_theta": n_theta,
        "nu_GHz": nu_GHz,
        "I_nuc": I_nuc,
    }


# --------------------------------------------------------------------------
#  Tool registry
# --------------------------------------------------------------------------

TOOL_SPEC_SIMULATE = ToolSpec(
    name="simulate_axial_powder",
    description=(
        "Simulate a first-derivative CW-EPR axial-powder spectrum for an "
        "S=1/2 system with axial g-tensor and parallel hyperfine coupling. "
        "Default I_nuc=3/2 covers Cu(II); set 0 for no hyperfine. Optional "
        "isotropic narrow line for organic-radical contamination near g ≈ 2."
    ),
    import_line="from scilink.skills.curve_fitting.epr.axial_powder import simulate_axial_powder",
    signature=(
        "simulate_axial_powder(B, g_par, g_perp, A_par_G, lw_G, "
        "nu_GHz=9.64, I_nuc=1.5, n_theta=181, iso_amplitude=0.0, "
        "iso_g=2.0030, iso_lw_G=8.0, amplitude=1.0) -> ndarray"
    ),
    parameters={
        "B": {"type": "list[float]", "description": "Field grid in Gauss (monotonic)."},
        "g_par": {"type": "float", "description": "g_parallel."},
        "g_perp": {"type": "float", "description": "g_perpendicular."},
        "A_par_G": {"type": "float", "description": "Parallel hyperfine in Gauss."},
        "lw_G": {"type": "float", "description": "Gaussian FWHM in Gauss."},
        "nu_GHz": {"type": "float", "description": "Microwave frequency (default 9.64, X-band)."},
        "I_nuc": {"type": "float", "description": "Nuclear spin. 3/2=Cu/Mn quartet; 0=no hyperfine; 1/2=doublet."},
        "n_theta": {"type": "int", "description": "Orientation grid size; 181 ≈ 0.5°."},
        "iso_amplitude": {"type": "float", "description": "Amplitude of optional isotropic line (0 disables)."},
        "iso_g": {"type": "float", "description": "g-value of the isotropic line."},
        "iso_lw_G": {"type": "float", "description": "FWHM of the isotropic line in Gauss."},
        "amplitude": {"type": "float", "description": "Overall scale of the main component."},
    },
    required=["B", "g_par", "g_perp", "A_par_G", "lw_G"],
    returns="ndarray, dχ\"/dB on the input field grid.",
    when_to_use=(
        "Forward modeling — generating a trial powder spectrum for a "
        "known parameter set, or evaluating the fitted model from "
        "``fit_axial_powder`` on a finer grid for plotting."
    ),
)


TOOL_SPEC_FIT = ToolSpec(
    name="fit_axial_powder",
    description=(
        "Least-squares fit of an experimental CW-EPR derivative spectrum "
        "with the axial-powder model. Returns g_par, g_perp, A_par, "
        "linewidth, R², and an optional second isotropic component "
        "(intended for radical contamination near g ≈ 2)."
    ),
    import_line="from scilink.skills.curve_fitting.epr.axial_powder import fit_axial_powder",
    signature=(
        "fit_axial_powder(B, y, nu_GHz=9.64, I_nuc=1.5, g_par_init=2.30, "
        "g_perp_init=2.06, A_par_G_init=170.0, lw_G_init=40.0, "
        "include_isotropic=False, ...) -> dict"
    ),
    parameters={
        "B": {"type": "list[float]", "description": "Experimental field grid in Gauss."},
        "y": {"type": "list[float]", "description": "Experimental dχ\"/dB intensities."},
        "nu_GHz": {"type": "float", "description": "Microwave frequency (default 9.64)."},
        "I_nuc": {"type": "float", "description": "Nuclear spin (default 3/2 for Cu)."},
        "g_par_init": {"type": "float", "description": "Initial guess for g_parallel."},
        "g_perp_init": {"type": "float", "description": "Initial guess for g_perpendicular."},
        "A_par_G_init": {"type": "float", "description": "Initial guess for A_parallel (G)."},
        "lw_G_init": {"type": "float", "description": "Initial guess for Gaussian FWHM (G)."},
        "include_isotropic": {"type": "bool", "description": "If True, fit a second narrow isotropic component."},
        "bounds_g_par": {"type": "tuple", "description": "(lo, hi) bounds for g_par."},
        "bounds_g_perp": {"type": "tuple", "description": "(lo, hi) bounds for g_perp."},
        "bounds_A_par_G": {"type": "tuple", "description": "(lo, hi) bounds for A_par."},
        "bounds_lw_G": {"type": "tuple", "description": "(lo, hi) bounds for linewidth."},
        "n_theta": {"type": "int", "description": "Orientation grid; 181 ≈ 0.5° resolution."},
    },
    required=["B", "y"],
    returns=(
        "dict with 'parameters' (nested per-component), 'fit_quality' "
        "(r_squared, rmse), 'derived' (g_avg), 'y_fit' (model on B), "
        "and 'model_used' / 'nu_GHz' / 'I_nuc' metadata."
    ),
    when_to_use=(
        "Per-spectrum EPR powder fit for axial S=1/2 systems (Cu(II) the "
        "canonical case). For an axial Cu(II) spectrum with a sharp g≈2 "
        "shoulder, set ``include_isotropic=True``."
    ),
)


# The registry discovers ``TOOL_SPEC`` (single) or ``TOOL_SPECS`` (list); this
# module declares two tools, so expose them via the list form.
TOOL_SPECS = [TOOL_SPEC_SIMULATE, TOOL_SPEC_FIT]
