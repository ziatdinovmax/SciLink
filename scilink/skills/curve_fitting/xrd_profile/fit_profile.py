"""``fit_profile`` tool — per-peak pseudo-Voigt fit of an XRD peak.

Wraps lmfit's ``PseudoVoigtModel`` (or Lorentzian / Gaussian) with a
local linear / constant background. Supports single-peak fits and joint
fits of overlapping peaks (pass a list of starting centers).
"""

from __future__ import annotations

import logging
from typing import Any, Sequence, Union

import numpy as np

from ..._shared._spec import ToolSpec

try:
    from lmfit.models import (
        ConstantModel,
        GaussianModel,
        LinearModel,
        LorentzianModel,
        PseudoVoigtModel,
    )
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    PseudoVoigtModel = LorentzianModel = GaussianModel = None  # type: ignore
    LinearModel = ConstantModel = None  # type: ignore

_logger = logging.getLogger(__name__)


TOOL_SPEC = ToolSpec(
    name="fit_profile",
    description=(
        "Fit a single (or jointly-fit overlapping) peak in an XRD "
        "pattern with a pseudo-Voigt (default), Lorentzian, or "
        "Gaussian profile plus a local background. Returns center, "
        "FWHM, amplitude, mixing parameter, and R²."
    ),
    import_line="from scilink.skills.curve_fitting.xrd_profile.fit_profile import fit_profile",
    signature=(
        "fit_profile(exp_two_theta, exp_intensity, peak_init, "
        "model='pseudo_voigt', window_deg=1.0, background='linear') -> dict"
    ),
    parameters={
        "exp_two_theta": {
            "type": "list[float]",
            "description": "Experimental 2-theta grid (degrees).",
        },
        "exp_intensity": {
            "type": "list[float]",
            "description": "Experimental intensity. Same length as exp_two_theta. Background-subtracted recommended.",
        },
        "peak_init": {
            "type": "float | list[float]",
            "description": "Starting 2θ center (single peak) or list of starting centers (joint fit of overlapping peaks).",
        },
        "model": {
            "type": "str",
            "description": "'pseudo_voigt' (default), 'lorentzian', or 'gaussian'.",
        },
        "window_deg": {
            "type": "float",
            "description": "Half-width of the fit window around the peak(s) in degrees. Default 1.0. For multi-peak fits, the window spans min-window to max+window.",
        },
        "background": {
            "type": "str",
            "description": "Local baseline inside the window: 'linear' (default), 'constant', or 'none'.",
        },
    },
    required=["exp_two_theta", "exp_intensity", "peak_init"],
    returns=(
        "dict with top-level 'center', 'fwhm', 'amplitude', 'eta', "
        "'r_squared' (the primary peak's values; r_squared is over the "
        "fit window), plus 'peaks' (list of per-peak dicts; each entry "
        "carries its own center/fwhm/amplitude/eta AND r_squared so "
        "iterating ``peaks`` is safe), 'model_used', 'window' ((lo, hi))."
    ),
    when_to_use=(
        "After seeding peak centers (extract_peaks or manual). Use "
        "joint fitting (list of centers) when two peaks lie within "
        "1.5 × FWHM of each other."
    ),
)


def fit_profile(
    exp_two_theta: Sequence[float],
    exp_intensity: Sequence[float],
    peak_init: Union[float, Sequence[float]],
    model: str = "pseudo_voigt",
    window_deg: float = 1.0,
    background: str = "linear",
) -> dict[str, Any]:
    """Fit one or several pseudo-Voigt / Lorentzian / Gaussian peaks."""
    if not LMFIT_AVAILABLE:
        raise RuntimeError("fit_profile requires lmfit; install via 'pip install lmfit'")
    if model not in {"pseudo_voigt", "lorentzian", "gaussian"}:
        raise ValueError(f"Unknown model: {model!r}")
    if background not in {"linear", "constant", "none"}:
        raise ValueError(f"Unknown background: {background!r}")
    if window_deg <= 0:
        raise ValueError(f"window_deg must be positive; got {window_deg}")

    centers = _normalize_centers(peak_init)
    x = np.asarray(exp_two_theta, dtype=float)
    y = np.asarray(exp_intensity, dtype=float)
    if x.shape != y.shape:
        raise ValueError("exp_two_theta and exp_intensity must have the same length")

    lo = min(centers) - window_deg
    hi = max(centers) + window_deg
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 5:
        raise ValueError(
            f"Fit window [{lo:.3f}, {hi:.3f}] contains fewer than 5 points; "
            "widen window_deg or check peak_init."
        )
    xf = x[mask]
    yf = y[mask]

    composite = None
    params = None
    peak_class = _peak_class(model)
    for i, center in enumerate(centers):
        prefix = f"p{i}_"
        peak = peak_class(prefix=prefix)
        peak_params = peak.guess(yf, x=xf)
        peak_params[f"{prefix}center"].set(value=center, min=center - window_deg, max=center + window_deg)
        peak_params[f"{prefix}sigma"].set(min=1e-3)
        peak_params[f"{prefix}amplitude"].set(min=0.0)
        if composite is None:
            composite = peak
            params = peak_params
        else:
            composite = composite + peak
            params.update(peak_params)

    if background == "linear":
        bg = LinearModel(prefix="bg_")
        composite = composite + bg
        params.update(bg.make_params(intercept=float(yf.min()), slope=0.0))
    elif background == "constant":
        bg = ConstantModel(prefix="bg_")
        composite = composite + bg
        params.update(bg.make_params(c=float(yf.min())))

    try:
        result = composite.fit(yf, params, x=xf)
    except Exception as e:
        raise RuntimeError(f"lmfit failed: {e}") from e

    y_fit = result.best_fit
    ss_res = float(np.sum((yf - y_fit) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    per_peak: list[dict[str, float]] = []
    for i in range(len(centers)):
        prefix = f"p{i}_"
        per_peak.append(_extract_peak_params(result, prefix, model, r_squared))

    primary = per_peak[0]
    return {
        "center": primary["center"],
        "fwhm": primary["fwhm"],
        "amplitude": primary["amplitude"],
        "eta": primary["eta"],
        "r_squared": float(r_squared),
        "peaks": per_peak,
        "model_used": model,
        "window": (float(lo), float(hi)),
    }


def _normalize_centers(peak_init: Union[float, Sequence[float]]) -> list[float]:
    if isinstance(peak_init, (int, float)):
        return [float(peak_init)]
    centers = [float(c) for c in peak_init]
    if not centers:
        raise ValueError("peak_init must contain at least one center")
    return centers


def _peak_class(model: str):
    if model == "pseudo_voigt":
        return PseudoVoigtModel
    if model == "lorentzian":
        return LorentzianModel
    return GaussianModel


def _extract_peak_params(result, prefix: str, model: str, r_squared: float) -> dict[str, float]:
    p = result.params
    center = float(p[f"{prefix}center"].value)
    amplitude = float(p[f"{prefix}amplitude"].value)
    if f"{prefix}fwhm" in p:
        fwhm = float(p[f"{prefix}fwhm"].value)
    else:
        sigma = float(p[f"{prefix}sigma"].value)
        fwhm = _fwhm_from_sigma(sigma, model)
    if model == "pseudo_voigt" and f"{prefix}fraction" in p:
        # lmfit's PseudoVoigt fraction: 0 = pure Gaussian, 1 = pure Lorentzian.
        # We propagate that as eta unchanged.
        eta = float(p[f"{prefix}fraction"].value)
    elif model == "lorentzian":
        eta = 1.0
    elif model == "gaussian":
        eta = 0.0
    else:
        eta = float("nan")
    return {
        "center": center,
        "fwhm": fwhm,
        "amplitude": amplitude,
        "eta": eta,
        # For joint fits, lmfit produces ONE composite R² for the whole fit —
        # we copy it onto every per-peak entry so callers iterating ``peaks``
        # don't lose the R² signal.
        "r_squared": float(r_squared),
    }


def _fwhm_from_sigma(sigma: float, model: str) -> float:
    import math
    if model == "gaussian":
        return 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma
    if model == "lorentzian":
        return 2.0 * sigma
    # pseudo-Voigt typically exposes a derived fwhm parameter; fallback approx
    return 2.0 * sigma
