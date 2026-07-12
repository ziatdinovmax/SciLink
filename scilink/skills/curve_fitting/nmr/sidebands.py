"""Spinning-sideband manifold fitting for MAS NMR.

Under magic-angle spinning, an anisotropic interaction (CSA for spin-½; the
satellite/CSA pattern for quadrupolar nuclei) that is not fully averaged appears
as a centreband at the isotropic shift plus a comb of **spinning sidebands** at
exactly δ_iso ± k·ν_r (ν_r = spinning rate). Treating the sidebands as separate
chemical sites is wrong, and a quantification that sums only the centreband
undercounts — the sideband intensity belongs to the same species.

``fit_sideband_manifold`` fits the whole comb as one species: peaks LOCKED to
δ_iso ± k·(ν_r/ν_L) ppm (positions fixed by the spinning rate, not free),
sharing one Voigt lineshape, with a free amplitude per order. It returns the
isotropic shift, the per-order intensities, and — the point of the tool — the
**sideband-inclusive total integrated intensity** plus the centreband fraction,
so MAS quantification is correct. CSA tensor extraction (Herzfeld–Berger) is
deferred; the manifold span is reported as a rough anisotropy scale.

Scope: one species' manifold per call (the caller passes the centreband region
and the MAS rate). For several sideband-bearing sites, call once per centreband.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..._shared._spec import ToolSpec
from .multipeak import _voigt_height_norm, _fwhm
from .quality import peak_region_r2


def fit_sideband_manifold(
    x: Sequence[float],
    y: Sequence[float],
    mas_rate_ppm: float,
    centre_ppm: Optional[float] = None,
    n_sidebands: Optional[int] = None,
    baseline: Optional[Sequence[float]] = None,
    allow_negative: bool = True,
    centre_search_ppm: Optional[float] = None,
    max_order_cap: int = 12,
    min_amp_snr: float = 3.0,
) -> dict[str, Any]:
    """Fit a centreband + spinning-sideband comb as one species.

    Peaks are locked at ``centre ± k·mas_rate_ppm`` (k = 0…n_sidebands); only the
    isotropic ``centre``, a shared Voigt width, the per-order amplitudes, and a
    constant offset are fit. ``n_sidebands`` defaults to the highest order with
    data above ``min_amp_snr``·noise (capped at ``max_order_cap``). Returns the
    isotropic shift, per-order intensities, the sideband-inclusive total
    integrated intensity, the centreband fraction, and the fit quality.
    """
    x = np.asarray(x, float)
    y0 = np.asarray(y, float)
    base = np.zeros_like(y0) if baseline is None else np.asarray(baseline, float)
    y = y0 - base
    order = np.argsort(x)
    x, y = x[order], y[order]

    noise = 1.4826 * np.median(np.abs(y - np.median(y))) or float(np.std(y)) or 1.0
    span = float(x.max() - x.min())
    wlo, whi = max(span / len(x), 1e-4), max(mas_rate_ppm * 0.9, span / 4.0)
    if centre_ppm is None:
        centre_ppm = float(x[np.argmax(np.abs(y))])
    if centre_search_ppm is None:
        centre_search_ppm = mas_rate_ppm * 0.5

    # Auto order count: highest k whose ± position carries data above the noise.
    max_k_in_range = int(min(max_order_cap,
                             (x.max() - centre_ppm) // max(mas_rate_ppm, 1e-9),
                             (centre_ppm - x.min()) // max(mas_rate_ppm, 1e-9)))
    if n_sidebands is None:
        n_sidebands = 0
        for k in range(1, max_k_in_range + 1):
            hit = False
            for pos in (centre_ppm + k * mas_rate_ppm, centre_ppm - k * mas_rate_ppm):
                j = int(np.argmin(np.abs(x - pos)))
                lo, hi = max(0, j - 3), min(len(x), j + 4)
                if np.max(np.abs(y[lo:hi])) > min_amp_snr * noise:
                    hit = True
            if hit:
                n_sidebands = k
    n_sidebands = int(min(n_sidebands, max_k_in_range))
    orders = list(range(-n_sidebands, n_sidebands + 1))

    w0 = max(span / 200.0, 5 * wlo)
    amax = float(np.max(np.abs(y)) * 5 + 1e-30)

    def positions(centre):
        return [centre + k * mas_rate_ppm for k in orders]

    def model(p):
        centre, sigma, gamma = p[0], p[1], p[2]
        amps, offset = p[3:3 + len(orders)], p[-1]
        out = np.full_like(x, offset)
        for a, pos in zip(amps, positions(centre)):
            out = out + a * _voigt_height_norm(x, pos, sigma, gamma)
        return out

    # Seed amplitudes from the data at each locked position.
    a_seed = []
    for pos in positions(centre_ppm):
        j = int(np.argmin(np.abs(x - pos)))
        a_seed.append(float(y[j]))
    p0 = [centre_ppm, w0, w0, *a_seed, 0.0]
    lo = [centre_ppm - centre_search_ppm, wlo, wlo,
          *[(-amax if allow_negative else 0.0)] * len(orders), -amax]
    hi = [centre_ppm + centre_search_ppm, whi, whi, *[amax] * len(orders), amax]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]

    res = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi),
                        method="trf", x_scale="jac", max_nfev=800)
    centre, sigma, gamma = res.x[0], res.x[1], res.x[2]
    amps = res.x[3:3 + len(orders)]
    offset = res.x[-1]
    fwhm = _fwhm(sigma, gamma)
    areas = {k: float(a * fwhm) for k, a in zip(orders, amps)}
    total = float(sum(abs(v) for v in areas.values())) or 1e-30
    cb = abs(areas.get(0, 0.0))
    # Significant orders (>5% of centreband) set the manifold span.
    sig_k = [abs(k) for k, v in areas.items() if abs(v) > 0.05 * (cb or 1.0)]
    span_ppm = (max(sig_k) if sig_k else 0) * mas_rate_ppm

    y_fit = model(res.x)
    q = peak_region_r2(x, y, y_fit)
    return {
        "isotropic_shift_ppm": float(centre),
        "n_sidebands_each_side": n_sidebands,
        "fwhm_ppm": float(fwhm),
        "order_intensities": {str(k): areas[k] for k in orders},
        "total_integrated_intensity": total,
        "centreband_fraction": float(cb / total),
        "manifold_span_ppm": float(span_ppm),
        "fit_quality": {"peak_region_r2": q["peak_region_r2"], "r_squared": q["r_squared"]},
        "y_fit": (y_fit + base[order]).tolist(),
        "note": "CSA/quadrupolar tensor extraction (Herzfeld-Berger) deferred; "
                "manifold_span_ppm is a rough anisotropy scale.",
    }


TOOL_SPEC = ToolSpec(
    name="fit_sideband_manifold",
    description=(
        "Fit a MAS spinning-sideband manifold (centreband + sidebands locked at "
        "δ_iso ± k·MAS-rate) as ONE species and return the sideband-inclusive "
        "total integrated intensity, the per-order pattern, and the centreband "
        "fraction — so MAS quantification counts the sidebands instead of "
        "treating them as separate sites or dropping them."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.sidebands import fit_sideband_manifold",
    signature=(
        "fit_sideband_manifold(x, y, mas_rate_ppm, centre_ppm=None, "
        "n_sidebands=None, baseline=None, allow_negative=True, "
        "centre_search_ppm=None, max_order_cap=12, min_amp_snr=3.0) -> dict"
    ),
    parameters={
        "x": {"type": "list[float]", "description": "Chemical-shift axis (ppm), spanning the centreband and its sidebands."},
        "y": {"type": "list[float]", "description": "Spectrum intensity."},
        "mas_rate_ppm": {"type": "float", "description": "Spinning rate in ppm (= spinning_rate_Hz / ν_L_MHz) — sets the sideband spacing. REQUIRED."},
        "centre_ppm": {"type": "float", "description": "Isotropic-shift seed (default: strongest peak). Set it explicitly when the centreband is NOT the tallest peak (large anisotropy can make a sideband taller)."},
        "n_sidebands": {"type": "int", "description": "Sideband orders per side to fit (default: auto-detected from where data clears the noise). Set explicitly to force more/fewer orders."},
        "baseline": {"type": "list[float]", "description": "Fitted baseline on x (or omit if pre-subtracted)."},
        "allow_negative": {"type": "bool", "description": "Permit negative amplitudes (default True). Set False once correctly phased."},
        "centre_search_ppm": {"type": "float", "description": "How far the isotropic shift may float from the seed (default ½·MAS-rate). Narrow it if the seed is reliable."},
        "max_order_cap": {"type": "int", "description": "Safety cap on sideband orders (default 12). Raise for very slow spinning with many orders."},
        "min_amp_snr": {"type": "float", "description": "Noise multiple a sideband position must clear to count in auto order-detection (default 3). Raise on noisy spectra."},
    },
    required=["x", "y", "mas_rate_ppm"],
    returns=(
        "dict with 'isotropic_shift_ppm', 'order_intensities' (area per order), "
        "'total_integrated_intensity' (sideband-inclusive), 'centreband_fraction', "
        "'manifold_span_ppm', and 'fit_quality'."
    ),
    when_to_use=(
        "A MAS spectrum showing a regular comb of peaks spaced by the spinning "
        "rate around a centreband — to quantify total intensity correctly or to "
        "confirm features are sidebands, not distinct sites. Not for solution or "
        "static spectra."
    ),
)

TOOL_SPECS = [TOOL_SPEC]
