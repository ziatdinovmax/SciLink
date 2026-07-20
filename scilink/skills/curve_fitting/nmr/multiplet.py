"""Constrained first-order J-coupled multiplet fitting for 1D NMR.

A scalar-coupled resonance (a doublet, triplet, quartet, …) is a *single*
chemical environment split into ``m`` equally-spaced lines by ``J``. Fitting it
as ``m`` independent Voigts is badly under-determined: the lines share one
chemical shift, one linewidth, and a fixed (binomial, for first-order coupling
to equivalent spin-½ neighbours) intensity pattern, yet a free multi-Voigt fit
has ~4·m free parameters that can wander into split/merged/asymmetric solutions
and is sensitive to the starting guess.

``fit_jcoupled_multiplet`` imposes the physics instead: the whole manifold is
parametrized by just **(centroid, J, σ, γ, amplitude, offset)** — one shift, one
coupling, one shared Voigt lineshape, binomial (or caller-supplied) relative
intensities — regardless of how many lines there are. That collapses a quartet
from ~16 free parameters to 6, removing the instability while still returning the
quantity of interest: the coupling constant ``J`` (in Hz) and the underlying
(coupling-free) chemical shift and linewidth.

General by construction, not tuned to any sample: the multiplicity and J are
inferred from the data (or supplied by the caller), the intensity pattern is the
physical binomial law, and the lineshape is a standard Voigt. For a single
uncoupled line use a 1-component Voigt; for several *distinct chemical
environments* (different shifts) use ``fit_multipeak_voigt``; for a quadrupolar
powder lineshape use ``fit_quad_ct``.
"""

from __future__ import annotations

from math import comb
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import wofz

from ..._shared._spec import ToolSpec
from .quality import peak_region_r2


def _voigt_height_norm(x, c, sigma, gamma):
    """Voigt profile normalized to unit peak height (so amp == line height)."""
    sigma = max(abs(sigma), 1e-6)
    gamma = max(abs(gamma), 1e-9)
    z = ((x - c) + 1j * gamma) / (sigma * np.sqrt(2.0))
    z0 = (1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / np.real(wofz(z0))


def _fwhm(sigma, gamma):
    """Olivero-Longbothum Voigt FWHM approximation."""
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    return 0.5346 * fl + np.sqrt(0.2166 * fl * fl + fg * fg)


def _pattern_weights(multiplicity: int, pattern) -> np.ndarray:
    """Relative line intensities, normalized so the tallest line is 1.0.

    ``"binomial"`` gives the first-order coupling pattern (Pascal's triangle):
    a doublet 1:1, triplet 1:2:1, quartet 1:3:3:1, … i.e. C(m-1, k). A list/tuple
    is taken verbatim (must match ``multiplicity``) for non-binomial cases.
    """
    if isinstance(pattern, (list, tuple, np.ndarray)):
        w = np.asarray(pattern, float)
        if w.size != multiplicity:
            raise ValueError(
                f"pattern has {w.size} weights but multiplicity={multiplicity}")
    elif pattern == "binomial":
        n = multiplicity - 1
        w = np.array([comb(n, k) for k in range(multiplicity)], float)
    elif pattern == "uniform":
        w = np.ones(multiplicity, float)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    return w / (w.max() or 1.0)


def _line_centers(centroid: float, J_ppm: float, multiplicity: int) -> np.ndarray:
    """Equally-spaced line positions symmetric about the centroid."""
    k = np.arange(multiplicity) - (multiplicity - 1) / 2.0
    return centroid + k * J_ppm


def _model(x, centroid, J_ppm, sigma, gamma, amp, offset, weights):
    out = np.full_like(x, offset, dtype=float)
    for c, w in zip(_line_centers(centroid, J_ppm, len(weights)), weights):
        out = out + amp * w * _voigt_height_norm(x, c, sigma, gamma)
    return out


def _autodetect(x, y, noise):
    """Seed (multiplicity, centroid, J_ppm) from regularly-spaced maxima.

    Returns ``None`` when fewer than two clear lines are present (caller should
    pass ``multiplicity`` explicitly, or treat as a singlet).
    """
    # Sub-peaks that clear BOTH an absolute noise floor and a relative floor
    # (a fraction of the tallest line), AND have real PROMINENCE. The prominence
    # requirement is essential: without it, noise ripples on the rounded top of a
    # broad line register as dozens of spurious sub-maxima and wreck the spacing
    # estimate. Prominence demands each counted line stand out from its
    # surroundings, so only genuine, separated lines survive.
    ymax = float(np.max(y))
    floor = max(4.0 * noise, 0.05 * ymax)
    pk, props = find_peaks(y, height=floor, prominence=max(3.0 * noise, 0.05 * ymax),
                           distance=3)
    if pk.size < 2:
        return None
    # Order by position; estimate spacing from consecutive gaps (median is robust
    # to a missing weak outer line).
    xs = np.sort(x[pk])
    gaps = np.diff(xs)
    J_ppm = float(np.median(gaps))
    if J_ppm <= 0:
        return None
    # Multiplicity from the span / spacing (rounded); centroid = intensity-weighted.
    span = xs[-1] - xs[0]
    m = int(round(span / J_ppm)) + 1
    m = max(2, min(m, pk.size))  # don't claim more lines than detected maxima
    centroid = float(np.average(x[pk], weights=props["peak_heights"]))
    return m, centroid, J_ppm


def _envelope_width(x, y, noise, k=3.0, min_run=3):
    """ppm extent of the signal envelope (first→last *contiguous* run above
    ``k·noise``).

    For a multiplet of known line count this gives a robust J seed —
    ``≈ (m-1)·J`` from outer line to outer line — without needing to resolve the
    individual lines, which noise on broad lines can defeat. Contiguous runs
    (not bare min/max of all above-threshold points) are required so that
    isolated noise spikes far from the multiplet do not blow out the width.
    """
    mask = np.abs(y - np.median(y)) > k * noise
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    if not runs:
        return None
    return float(x[runs[-1][1]] - x[runs[0][0]])


def fit_jcoupled_multiplet(
    x: Sequence[float],
    y: Sequence[float],
    baseline: Optional[Sequence[float]] = None,
    multiplicity: Optional[int] = None,
    pattern: "str | Sequence[float]" = "binomial",
    nu_L_MHz: Optional[float] = None,
    J_hz: Optional[float] = None,
    center_ppm: Optional[float] = None,
    allow_negative: bool = False,
) -> dict[str, Any]:
    """Fit a first-order J-coupled multiplet as one constrained manifold.

    The manifold is ``multiplicity`` equally-spaced Voigt lines sharing one
    centroid, one coupling ``J``, one (σ, γ) lineshape, and fixed ``pattern``
    intensities — fit with just 6 free parameters (centroid, J, σ, γ, amplitude,
    offset). ``multiplicity``/``J``/``center`` are seeded from the data when not
    given. Crop ``x``/``y`` to the multiplet first.

    Returns the coupling constant (Hz and ppm), the coupling-free chemical shift
    and linewidth, the per-line positions/intensities, the fit, and the
    peak-region quality. ``nu_L_MHz`` is needed to report J and widths in Hz.
    """
    x = np.asarray(x, float)
    y0 = np.asarray(y, float)
    base = np.zeros_like(y0) if baseline is None else np.asarray(baseline, float)
    y = y0 - base
    order = np.argsort(x)
    x, y, base = x[order], y[order], base[order]

    noise = 1.4826 * np.median(np.abs(y - np.median(y))) or float(np.std(y)) or 1.0
    span = float(x.max() - x.min())
    sgn = -1.0 if (allow_negative and abs(y.min()) > abs(y.max())) else 1.0

    # --- seed multiplicity / centroid / J from the data when not supplied -----
    det = _autodetect(x, sgn * y, noise)
    if multiplicity is None:
        multiplicity = det[0] if det else 1
    if center_ppm is None:
        center_ppm = det[1] if det else float(x[np.argmax(np.abs(y))])
    multiplicity = max(1, int(multiplicity))
    if J_hz is not None and nu_L_MHz:
        J_ppm0 = float(J_hz) / float(nu_L_MHz)
    elif multiplicity > 1:
        # Two J estimates: the per-line spacing (det[2]) and the envelope width
        # /(m-1). The per-line value is sharp when lines are cleanly resolved but
        # is wrong when a narrow noisy line splits into several detected maxima;
        # the envelope (outer-to-outer extent over (m-1) gaps) is the robust
        # cross-check. Trust det only when the two AGREE; otherwise use the
        # envelope. This is the crucial guard against a bad seed (which lets the
        # manifold collapse onto the central line).
        W = _envelope_width(x, sgn * y, noise)
        J_env = (W / (multiplicity - 1)) if W else None
        J_det = det[2] if det else None
        if J_det and J_env and 0.4 * J_env <= J_det <= 1.6 * J_env:
            J_ppm0 = J_det
        elif J_env:
            J_ppm0 = J_env
        elif J_det:
            J_ppm0 = J_det
        else:
            J_ppm0 = span / max(multiplicity * 4, 8)
    else:
        J_ppm0 = span / max(multiplicity * 4, 8)  # singlet: J unused
    weights = _pattern_weights(multiplicity, pattern)

    # --- bounds & initial guess ----------------------------------------------
    amax = float(np.max(np.abs(y)) * 5 + 1e-30)
    wlo = max(span / len(x), 1e-4)
    whi = span / 2.0
    a0 = sgn * float(np.max(np.abs(y)))
    # J lower bound: for a resolved multiplet J ≫ linewidth, so floor J at a
    # fraction of the data-seeded spacing. This blocks the failure mode where the
    # optimizer collapses all lines onto the dominant central line (J→0) and fits
    # the manifold as one fat peak — a strong local minimum for a 1:2:1 triplet.
    J_floor = max(wlo, 0.2 * J_ppm0) if multiplicity > 1 else wlo

    def resid(p):
        return _model(x, p[0], p[1], p[2], p[3], p[4], p[5], weights) - y

    def _fit_from(sg_seed, j_anchor=False):
        jlo = (0.7 * J_ppm0 if j_anchor else J_floor)
        jhi = (1.3 * J_ppm0 if j_anchor else whi)
        p0 = [center_ppm, min(max(J_ppm0, jlo), jhi), sg_seed, sg_seed, a0, 0.0]
        lo = [x.min(), jlo, wlo, wlo, (-amax if allow_negative else 0.0), -amax]
        hi = [x.max(), jhi, whi, whi, amax, amax]
        if multiplicity == 1:  # no coupling: pin J at its floor (don't fit it)
            lo[1], hi[1] = J_floor, J_floor + 1e-9   # strict lo<hi for least_squares
            p0[1] = J_floor
        p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
        r = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                          x_scale="jac", max_nfev=800)
        yf = _model(x, *r.x[:6], weights)
        return r.x, peak_region_r2(x.tolist(), y.tolist(), yf.tolist())["peak_region_r2"]

    # Multi-start over the shared linewidth seed (the lineshape is the part the
    # data-seed is weakest on): a narrow line vs a J/4-broad start can land in
    # different basins. The final start ANCHORS J near the data-seeded spacing —
    # it cannot collapse (all lines onto the centre) or run away (lines off the
    # data), so it reliably anchors a good fit whenever the seed is sound. Keep
    # the best peak-region R² across starts, so an anchored start never hurts.
    base_w = (J_ppm0 if multiplicity > 1 else span / 100.0)
    starts = [(0.25, False), (0.1, False), (0.04, False)]
    if multiplicity > 1:
        starts.append((0.1, True))
    best = None
    for frac, anchor in starts:
        sg = max(min(base_w * frac, whi), 1.5 * wlo)
        try:
            xr_, r2_ = _fit_from(sg, j_anchor=anchor)
        except Exception:
            continue
        if best is None or r2_ > best[1]:
            best = (xr_, r2_)
    centroid, J_ppm, sigma, gamma, amp, offset = best[0]
    y_fit = _model(x, centroid, J_ppm, sigma, gamma, amp, offset, weights)
    q = peak_region_r2(x.tolist(), y.tolist(), y_fit.tolist())

    centers = _line_centers(centroid, J_ppm, multiplicity)
    fwhm_ppm = float(_fwhm(sigma, gamma))
    nu = float(nu_L_MHz) if nu_L_MHz else None
    return {
        "multiplicity": int(multiplicity),
        "pattern": (list(map(float, weights)) if not isinstance(pattern, str) else pattern),
        "center_ppm": float(centroid),
        "J_ppm": float(J_ppm) if multiplicity > 1 else 0.0,
        "J_hz": (float(J_ppm * nu) if (nu and multiplicity > 1) else None),
        "fwhm_ppm": fwhm_ppm,
        "fwhm_hz": (fwhm_ppm * nu if nu else None),
        "sigma_ppm": float(abs(sigma)),
        "gamma_ppm": float(abs(gamma)),
        "amplitude": float(amp),
        "offset": float(offset),
        "lines": [
            {"center_ppm": float(c), "rel_intensity": float(w), "height": float(amp * w)}
            for c, w in zip(centers, weights)
        ],
        "total_area": float(amp * fwhm_ppm * float(weights.sum())),
        "fit_quality": {"peak_region_r2": q["peak_region_r2"], "r_squared": q["r_squared"]},
        "y_fit": (y_fit + base).tolist(),
    }


TOOL_SPEC = ToolSpec(
    name="fit_jcoupled_multiplet",
    description=(
        "Fit a first-order J-coupled NMR multiplet (doublet, triplet, quartet, …) "
        "as ONE constrained manifold: m equally-spaced Voigt lines sharing a "
        "single chemical shift, one coupling constant J, one Voigt linewidth, and "
        "a fixed binomial intensity pattern — just 6 free parameters regardless of "
        "the number of lines. Far more stable than fitting m independent Voigts, "
        "and it returns J directly. Use for a SINGLE coupled environment; for "
        "several distinct chemical shifts use fit_multipeak_voigt, and for a "
        "quadrupolar powder lineshape use fit_quad_ct."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.multiplet import fit_jcoupled_multiplet",
    signature=(
        "fit_jcoupled_multiplet(x, y, baseline=None, multiplicity=None, "
        "pattern='binomial', nu_L_MHz=None, J_hz=None, center_ppm=None, "
        "allow_negative=False) -> dict"
    ),
    parameters={
        "x": {"type": "list[float]", "description": "Chemical-shift axis (ppm). Crop to the multiplet for a stable fit."},
        "y": {"type": "list[float]", "description": "Spectrum intensity over the multiplet."},
        "baseline": {"type": "list[float]", "description": "Fitted baseline on x (or omit if pre-subtracted)."},
        "multiplicity": {"type": "int", "description": "Number of lines (2=doublet, 3=triplet, 4=quartet, …). Omit to auto-detect from the regularly-spaced sub-peaks; pass it explicitly when you can read the line count off the spectrum."},
        "pattern": {"type": "str | list[float]", "description": "Relative line intensities. 'binomial' (default) = first-order coupling to equivalent spin-½ neighbours (1:1, 1:2:1, 1:3:3:1, …). 'uniform' for equal intensities, or pass an explicit symmetric list for an irregular pattern."},
        "nu_L_MHz": {"type": "float", "description": "Larmor frequency of the observed nucleus (MHz), from spectrometer_frequency_MHz. Required to report J and linewidths in Hz."},
        "J_hz": {"type": "float", "description": "Optional initial guess for the coupling constant (Hz); refined by the fit. Omit to seed J from the observed line spacing."},
        "center_ppm": {"type": "float", "description": "Optional initial guess for the multiplet centroid (ppm); omit to seed from the data."},
        "allow_negative": {"type": "bool", "description": "Permit an inverted (negative) multiplet for a 180°-phased spectrum (default False)."},
    },
    required=["x", "y"],
    returns=(
        "dict with 'multiplicity', 'J_hz'/'J_ppm', the coupling-free 'center_ppm' "
        "and 'fwhm_ppm'/'fwhm_hz', per-line 'lines' (center_ppm/rel_intensity), "
        "'total_area', 'fit_quality' (peak_region_r2, r_squared), and 'y_fit'."
    ),
    when_to_use=(
        "A resolved scalar-coupled multiplet from ONE chemical environment — any "
        "doublet/triplet/quartet/… of equally-spaced lines with a binomial "
        "(1:1, 1:2:1, 1:3:3:1, …) intensity ratio — when the coupling constant J "
        "is the deliverable. Use it instead of fitting the lines as independent "
        "Voigts. NOT for separate chemical environments (use fit_multipeak_voigt) "
        "or a single uncoupled line (a 1-component Voigt)."
    ),
)

TOOL_SPECS = [TOOL_SPEC]
