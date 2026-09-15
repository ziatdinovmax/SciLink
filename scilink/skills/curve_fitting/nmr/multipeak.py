"""Residual-driven multi-Voigt fitting for 1D NMR with parsimony stopping.

Multiple chemical environments (e.g. a sharp mobile species + a broad rigid one
in a MOF, or two crystallographic sites in a solid) appear as overlapping peaks.
The agent can already sum Voigts, but *deciding how many* is the unreliable part
— under-fitting a shoulder (one peak where there are two) or over-fitting noise.

``fit_multipeak_voigt`` makes that decision principled and repeatable: it adds
Voigt components one at a time, each seeded at the largest residual extremum,
refits all components jointly, and KEEPS a new component only if it materially
improves the peak-region R² and refines to a physically distinct, non-negligible
peak. It stops when the next component no longer helps — so the peak count is
chosen by the data, not guessed.

This is a general peak-deconvolution routine (Voigt sum, signed amplitudes for
inverted peaks), not tuned to any sample. Two NMR-specific guards: optional
exclusion of spinning-sideband positions (±k·MAS-rate from the strongest centre)
so sidebands are not fit as separate chemical sites, and the caller is expected
to route a single asymmetric *quadrupolar* line to ``fit_quad_ct`` instead of
mimicking it with several Voigts.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import wofz

from ..._shared._spec import ToolSpec
from .quality import peak_region_r2


def _voigt_height_norm(x, c, sigma, gamma):
    """Voigt profile normalized to unit peak height (amp == peak height)."""
    sigma = max(abs(sigma), 1e-6)
    gamma = max(abs(gamma), 1e-9)
    z = ((x - c) + 1j * gamma) / (sigma * np.sqrt(2.0))
    z0 = (1j * gamma) / (sigma * np.sqrt(2.0))
    return np.real(wofz(z)) / np.real(wofz(z0))


def _model(x, peaks, offset):
    out = np.full_like(x, offset, dtype=float)
    for amp, c, s, g in peaks:
        out = out + amp * _voigt_height_norm(x, c, s, g)
    return out


def _fwhm(sigma, gamma):
    # Olivero-Longbothum approximation for the Voigt FWHM.
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    return 0.5346 * fl + np.sqrt(0.2166 * fl * fl + fg * fg)


def _refit(x, y, peaks, offset, allow_negative, wlo, whi):
    """Joint least-squares refit of all current peaks + a constant offset."""
    p0, lo, hi = [], [], []
    span = float(x.max() - x.min())
    amax = float(np.max(np.abs(y)) * 5 + 1e-30)
    for amp, c, s, g in peaks:
        p0 += [amp, c, s, g]
        lo += [(-amax if allow_negative else 0.0), x.min(), wlo, wlo]
        hi += [amax, x.max(), whi, whi]
    p0 += [offset]; lo += [-amax]; hi += [amax]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]

    def resid(p):
        pk = [(p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p) - 1, 4)]
        return _model(x, pk, p[-1]) - y

    r = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                      x_scale="jac", max_nfev=600)
    pk = [(r.x[i], r.x[i + 1], r.x[i + 2], r.x[i + 3]) for i in range(0, len(r.x) - 1, 4)]
    return pk, float(r.x[-1])


def fit_multipeak_voigt(
    x: Sequence[float],
    y: Sequence[float],
    baseline: Optional[Sequence[float]] = None,
    max_peaks: int = 6,
    allow_negative: bool = True,
    improve_thresh: float = 0.02,
    min_amp_snr: float = 4.0,
    mas_rate_ppm: Optional[float] = None,
    n_sidebands: int = 4,
    sideband_tol_frac: float = 0.25,
) -> dict[str, Any]:
    """Fit a parsimonious sum of Voigt peaks; let the data choose the count.

    Adds components seeded at the largest residual extremum, refitting jointly,
    and keeps each only if peak-region R² improves by > ``improve_thresh`` and
    the new peak is distinct (not within ~½ FWHM of an existing one), above the
    noise (``min_amp_snr``), and of physical width. If ``mas_rate_ppm`` is given,
    seeds near ±k·MAS-rate of the strongest centre are skipped (spinning
    sidebands, not chemical sites). Returns the peak list, the fit, and metrics.
    """
    x = np.asarray(x, float)
    y0 = np.asarray(y, float)
    base = np.zeros_like(y0) if baseline is None else np.asarray(baseline, float)
    y = y0 - base
    order = np.argsort(x)
    x, y = x[order], y[order]

    noise = 1.4826 * np.median(np.abs(y - np.median(y))) or float(np.std(y)) or 1.0
    span = float(x.max() - x.min())
    wlo, whi = max(span / len(x), 1e-4), span / 2.0
    w0 = max(span / 200.0, 5 * wlo)

    # First peak: the global extremum.
    c0 = float(x[np.argmax(np.abs(y))])
    a0 = float(y[np.argmax(np.abs(y))])
    peaks, offset = _refit(x, y, [(a0, c0, w0, w0)], 0.0, allow_negative, wlo, whi)

    def _metric(pk, off):
        return peak_region_r2(x, y, _model(x, pk, off))["peak_region_r2"]

    best_r2 = _metric(peaks, offset)
    strongest_c = max(peaks, key=lambda p: abs(p[0]))[1]

    while len(peaks) < max_peaks:
        resid = y - _model(x, peaks, offset)
        # Mask out neighbourhoods of existing peaks and sideband positions.
        # Block only a tight neighbourhood of each existing apex — just enough
        # to avoid re-seeding the same maximum, NOT so wide it masks an
        # overlapping neighbouring site (the post-refit distinctness check is
        # what rejects true duplicates).
        block = np.zeros_like(x, dtype=bool)
        for amp, c, s, g in peaks:
            block |= np.abs(x - c) < 0.25 * _fwhm(s, g)
        if mas_rate_ppm:
            for k in range(1, n_sidebands + 1):
                for sb in (strongest_c + k * mas_rate_ppm, strongest_c - k * mas_rate_ppm):
                    block |= np.abs(x - sb) < sideband_tol_frac * mas_rate_ppm
        cand = np.where(~block, np.abs(resid), 0.0)
        if not np.any(cand > 0):
            break
        ci = int(np.argmax(cand))
        if abs(resid[ci]) < min_amp_snr * noise:
            break  # nothing left above the noise

        trial, toff = _refit(x, y, peaks + [(float(resid[ci]), float(x[ci]), w0, w0)],
                             offset, allow_negative, wlo, whi)
        trial_r2 = _metric(trial, toff)
        new = trial[-1]
        new_fw = _fwhm(new[2], new[3])
        # Reject only a genuine DEGENERATE split — a new peak that duplicates an
        # existing one in BOTH centre and width (the optimizer splitting one peak
        # into two). A narrow+broad pair at a similar centre (e.g. mobile + rigid
        # environments) is physically distinct and must be kept.
        def _dup(p):
            ofw = _fwhm(p[2], p[3])
            return (abs(new[1] - p[1]) < 0.5 * min(new_fw, ofw)
                    and 0.5 < new_fw / max(ofw, 1e-9) < 2.0)
        distinct = not any(_dup(p) for p in trial[:-1])
        physical = abs(new[0]) > min_amp_snr * noise and wlo * 1.5 < new_fw < whi
        if trial_r2 > best_r2 + improve_thresh and distinct and physical:
            peaks, offset, best_r2 = trial, toff, trial_r2
        else:
            break

    y_fit = _model(x, peaks, offset)
    q = peak_region_r2(x, y, y_fit)
    peak_list = sorted(
        ({"center_ppm": float(c), "amplitude": float(a),
          "fwhm_ppm": float(_fwhm(s, g)), "sigma": float(abs(s)), "gamma": float(abs(g)),
          "area": float(a * _fwhm(s, g))} for a, c, s, g in peaks),
        key=lambda d: d["center_ppm"])
    return {
        "n_peaks": len(peaks),
        "peaks": peak_list,
        "offset": float(offset),
        "fit_quality": {"peak_region_r2": q["peak_region_r2"], "r_squared": q["r_squared"]},
        "y_fit": (y_fit + base[order]).tolist(),
    }


TOOL_SPEC = ToolSpec(
    name="fit_multipeak_voigt",
    description=(
        "Fit a parsimonious sum of Voigt peaks to a 1D NMR spectrum, letting the "
        "DATA choose the number of components: it adds peaks at the largest "
        "residual, refits jointly, and keeps each only if it improves the "
        "peak-region R² and is physically distinct. Handles overlapping "
        "environments (sharp+broad), signed/inverted peaks, and skips spinning-"
        "sideband positions. NOT for a single quadrupolar lineshape — use "
        "fit_quad_ct there."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.multipeak import fit_multipeak_voigt",
    signature=(
        "fit_multipeak_voigt(x, y, baseline=None, max_peaks=6, "
        "allow_negative=True, improve_thresh=0.02, min_amp_snr=4.0, "
        "mas_rate_ppm=None, n_sidebands=4, sideband_tol_frac=0.25) -> dict"
    ),
    parameters={
        "x": {"type": "list[float]", "description": "Chemical-shift axis (ppm). Crop to the signal region first for speed."},
        "y": {"type": "list[float]", "description": "Spectrum intensity."},
        "baseline": {"type": "list[float]", "description": "Fitted baseline on x (or omit if pre-subtracted)."},
        "max_peaks": {"type": "int", "description": "Hard cap on component count (default 6). Raise it for a crowded spectrum with many resolved environments; lower it to force a more parsimonious fit."},
        "allow_negative": {"type": "bool", "description": "Permit inverted (negative) peaks (default True). Set False once the spectrum is correctly phased so all real peaks are positive — this prevents fitting noise dips."},
        "improve_thresh": {"type": "float", "description": "Minimum peak-region-R² gain required to KEEP each added peak (default 0.02). This is the parsimony knob: LOWER it (e.g. 0.005) to recover weak/subtle shoulders the default misses; RAISE it (e.g. 0.05) if the fit is adding spurious peaks."},
        "min_amp_snr": {"type": "float", "description": "Minimum amplitude of a new peak in noise units (default 4). RAISE for noisy spectra to avoid fitting noise as peaks; LOWER (e.g. 3) to catch genuine weak resonances."},
        "mas_rate_ppm": {"type": "float", "description": "MAS rate in ppm (= spinning_rate_Hz / ν_L_MHz). When given, seeds near ±k·MAS-rate of the strongest centre are skipped so spinning sidebands are not fit as chemical sites. Omit for solution-state or static spectra."},
        "n_sidebands": {"type": "int", "description": "How many sideband orders (±1…±k) to exclude when mas_rate_ppm is set (default 4). Increase for slow spinning / large anisotropy where many sideband orders appear."},
        "sideband_tol_frac": {"type": "float", "description": "Half-width of each excluded sideband window, as a fraction of the MAS rate (default 0.25). Widen if sidebands are broad and leak past the default window."},
    },
    required=["x", "y"],
    returns=(
        "dict with 'n_peaks', 'peaks' (each center_ppm/amplitude/fwhm_ppm/area), "
        "'fit_quality' (peak_region_r2, r_squared), and 'y_fit'."
    ),
    when_to_use=(
        "A spectrum with two or more overlapping chemical environments (a "
        "shoulder, an asymmetric multi-site cluster). For a single symmetric "
        "line use a 1-component Voigt; for a quadrupolar powder lineshape use "
        "fit_quad_ct."
    ),
)

TOOL_SPECS = [TOOL_SPEC]
