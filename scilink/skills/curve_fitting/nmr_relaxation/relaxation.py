"""NMR relaxation-curve fitting: spin-lattice (T1) and spin-spin (T2).

The input is an *integral vs delay* curve (x = relaxation delay in seconds), not
a spectrum. The recovery/decay model is set by the experiment:

- ``inversion_recovery`` (t1ir):   I(t) = I0·(1 − A·exp(−(t/T1)^β))   (A≈2 ideal;
  fit A for imperfect inversion; signed — negative at short t).
- ``saturation_recovery`` (satrec): I(t) = I0·(1 − exp(−(t/T1)^β)).
- ``t2_decay`` (CPMG / echo):       I(t) = I0·exp(−(t/T2)^β).

β is the stretching exponent: β=1 is mono-exponential; **β<1 is the standard
description for a disordered/glassy solid (a distribution of relaxation times)** —
common for quadrupolar nuclei in solid electrolytes. ``n_components=2`` fits two
independent environments (two T1/T2 with populations). Robust auto-seeding from
the null point (IR) or the 1/e time (SR/T2) avoids local minima.

Mirrors the curve_fitting helper shape; registered as a ``TOOL_SPEC``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..._shared._spec import ToolSpec

_MODELS = ("inversion_recovery", "saturation_recovery", "t2_decay")


def _kernel(t, tau, beta):
    return np.exp(-np.power(np.clip(t / max(tau, 1e-12), 0, None), beta))


def _seed_tau(t, y, model):
    """Cheap T1/T2 seed from the curve shape."""
    if model == "t2_decay":
        target = y[0] * np.exp(-1.0)           # 1/e of the initial amplitude
        below = np.where(y <= target)[0]
        return t[below[0]] if below.size else float(np.median(t))
    if model == "inversion_recovery":
        sign = np.sign(y)
        cross = np.where(np.diff(sign) != 0)[0]   # null point
        if cross.size:
            return float(t[cross[0]]) / np.log(2.0)
        # No null within the window → the recovery barely started, so T1 is
        # LONGER than the sampled range; seed large (don't collapse to 0).
        return float(t[-1]) * 5.0
    # saturation recovery: time to ~63% of the plateau
    plateau = y[-1]
    if plateau != 0:
        idx = np.where(np.abs(y) >= 0.63 * abs(plateau))[0]
        if idx.size:
            return float(t[idx[0]])
        # Never reaches 63% in the window → T1 longer than sampled; seed large.
        return float(t[-1]) * 5.0
    return float(np.median(t))


def fit_relaxation(
    delay: Sequence[float],
    intensity: Sequence[float],
    model: str = "inversion_recovery",
    stretched: bool = False,
    n_components: int = 1,
    tau_init: Optional[float] = None,
    bounds_tau_s: Optional[tuple] = None,
) -> dict[str, Any]:
    """Fit an NMR relaxation recovery/decay curve.

    Returns the relaxation time(s) (T1 for recovery models, T2 for decay), the
    stretching exponent β (1.0 when ``stretched=False``), I0, the imperfect-
    inversion factor A (inversion recovery), per-component populations when
    ``n_components=2``, and R² over the curve (the whole curve is signal, so
    global R² is the right metric here — unlike spectral fitting).
    """
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}")
    t = np.asarray(delay, float)
    y = np.asarray(intensity, float)
    order = np.argsort(t)
    t, y = t[order], y[order]
    scale = float(np.max(np.abs(y))) or 1.0
    tlo, thi = (bounds_tau_s if bounds_tau_s else
                (max(float(t.min()) * 0.1, 1e-7), float(t.max()) * 50))
    tau0 = tau_init if tau_init is not None else _seed_tau(t, y, model)
    tau0 = min(max(tau0, tlo), thi)
    # β is free in [0.3, 1] when stretched; otherwise pinned to ~1 (a hair below
    # so the optimizer accepts a strict lo<hi bound and β stays effectively 1).
    beta_lo = 0.3 if stretched else 1.0 - 1e-6

    def recovery(tau, beta):
        k = _kernel(t, tau, beta)
        if model == "t2_decay":
            return k
        if model == "inversion_recovery":
            return 1.0 - 2.0 * k          # A folded in below for fit
        return 1.0 - k                    # saturation recovery

    # Parameter vector by case.
    if n_components == 2:
        # two independent components sharing the model; populations f, 1-f
        def model_fn(p):
            I0, f, tau_a, tau_b, beta = p
            ya = recovery(tau_a, beta); yb = recovery(tau_b, beta)
            if model == "inversion_recovery":
                ka = _kernel(t, tau_a, beta); kb = _kernel(t, tau_b, beta)
                return I0 * (1 - 2 * (f * ka + (1 - f) * kb))
            return I0 * (f * ya + (1 - f) * yb)
        p0 = [scale * np.sign(y[-1] or 1), 0.5, tau0 * 0.3, tau0 * 3, 1.0]
        lo = [-10 * scale, 0.0, tlo, tlo, beta_lo]
        hi = [10 * scale, 1.0, thi, thi, 1.0]
    elif model == "inversion_recovery":
        # I0, A (imperfect inversion ~2), tau, beta
        def model_fn(p):
            I0, A, tau, beta = p
            return I0 * (1.0 - A * _kernel(t, tau, beta))
        p0 = [scale * np.sign(y[-1] or 1), 1.9, tau0, 1.0]
        lo = [-10 * scale, 1.0, tlo, beta_lo]; hi = [10 * scale, 2.05, thi, 1.0]
    else:
        # I0, tau, beta (saturation recovery / t2 decay)
        def model_fn(p):
            I0, tau, beta = p
            return I0 * recovery(tau, beta)
        p0 = [scale * np.sign(y[-1] or 1) or scale, tau0, 1.0]
        lo = [-10 * scale, tlo, beta_lo]; hi = [10 * scale, thi, 1.0]

    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
    res = least_squares(lambda p: model_fn(p) - y, p0, bounds=(lo, hi),
                        method="trf", x_scale="jac", max_nfev=600)
    yfit = model_fn(res.x)
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-30
    r2 = 1.0 - ss_res / ss_tot
    time_label = "T2_s" if model == "t2_decay" else "T1_s"

    if n_components == 2:
        I0, f, tau_a, tau_b, beta = res.x
        params = {"I0": float(I0), f"{time_label}_a": float(tau_a),
                  f"{time_label}_b": float(tau_b), "population_a": float(f),
                  "population_b": float(1 - f), "beta": float(beta)}
    elif model == "inversion_recovery":
        I0, A, tau, beta = res.x
        params = {"I0": float(I0), "A_inversion": float(A),
                  time_label: float(tau), "beta": float(beta)}
    else:
        I0, tau, beta = res.x
        params = {"I0": float(I0), time_label: float(tau), "beta": float(beta)}

    flags = []
    if stretched and params.get("beta", 1.0) < 0.6:
        flags.append("strongly stretched (β<0.6): broad distribution of "
                     "relaxation times — quote a mean/median relaxation time.")
    # The relaxation time is only constrained if it falls inside the sampled
    # delays. A value past the longest (or below the shortest) delay is an
    # extrapolation — report it as a bound and widen the delay list.
    tmax, tmin = float(t.max()), float(t[t > 0].min()) if np.any(t > 0) else 0.0
    taus = [v for k, v in params.items() if k.startswith(("T1_s", "T2_s"))]
    if taus and max(taus) > tmax:
        flags.append(f"relaxation time exceeds the longest delay ({tmax:.3g} s) — "
                     "unconstrained; treat as a lower bound and extend the delay list.")
    if taus and tmin > 0 and min(taus) < tmin:
        flags.append(f"relaxation time below the shortest delay ({tmin:.3g} s) — "
                     "unconstrained; treat as an upper bound and add shorter delays.")
    return {
        "model": model, "stretched": stretched, "n_components": n_components,
        "parameters": params,
        "fit_quality": {"r_squared": float(r2),
                        "rmse": float(np.sqrt(ss_res / max(len(y), 1)))},
        "y_fit": yfit.tolist(),
        "flags": flags,
    }


TOOL_SPEC = ToolSpec(
    name="fit_relaxation",
    description=(
        "Fit an NMR relaxation recovery/decay curve (integral vs delay) to "
        "extract T1 (inversion/saturation recovery) or T2 (echo decay), with an "
        "optional stretched exponent β for disordered solids and an optional "
        "two-component fit. Robust auto-seeding from the null point / 1-e time."
    ),
    import_line="from scilink.skills.curve_fitting.nmr_relaxation.relaxation import fit_relaxation",
    signature=(
        "fit_relaxation(delay, intensity, model='inversion_recovery', "
        "stretched=False, n_components=1, tau_init=None, bounds_tau_s=None) -> dict"
    ),
    parameters={
        "delay": {"type": "list[float]", "description": "Relaxation delays (seconds)."},
        "intensity": {"type": "list[float]", "description": "Peak integral per delay (signed for inversion recovery)."},
        "model": {"type": "str", "description": "'inversion_recovery' | 'saturation_recovery' | 't2_decay'. Determine it from the PULSE PROGRAM (inversion-recovery stem → inversion_recovery; saturation-recovery stem → saturation_recovery; CPMG/echo train → t2_decay), or a model-type metadata field if present — do not guess from the curve shape."},
        "stretched": {"type": "bool", "description": "Free the stretching exponent β (default False = mono-exponential). Set True for a disordered/glassy solid or quadrupolar nucleus where a single τ underfits (β<1 = distribution of times)."},
        "n_components": {"type": "int", "description": "1 (default) or 2. Use 2 when the residual of a single-component fit shows a second relaxation environment (returns two times + populations)."},
        "tau_init": {"type": "float", "description": "Initial T1/T2 guess (s); default auto-seeded from the curve. Set it when the null/1-e seed is poor (very noisy data)."},
        "bounds_tau_s": {"type": "tuple", "description": "(lo, hi) bounds on the relaxation time in seconds; default spans the delay range. Narrow it around a literature value to stabilise a noisy fit."},
    },
    required=["delay", "intensity"],
    returns=(
        "dict with 'parameters' (T1_s or T2_s, beta, I0, A_inversion or "
        "populations), 'fit_quality' (r_squared, rmse), 'y_fit', and 'flags'."
    ),
    when_to_use=(
        "A relaxation experiment staged as an (delay, integral) curve. Pick the "
        "model from the pulse program / experiment metadata. NOT for a ppm-axis "
        "spectrum (use the nmr skill there)."
    ),
)

TOOL_SPECS = [TOOL_SPEC]
