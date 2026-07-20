"""Second-order quadrupolar central-transition powder lineshape for solid-state
NMR of half-integer quadrupolar nuclei (²³Na, ²⁷Al, ¹⁷O, ⁷¹Ga, …).

The central transition (+½ ↔ −½) of a half-integer spin is unperturbed by the
*first*-order quadrupolar interaction but is broadened and shifted to second
order by an amount scaling as ν_Q²/ν_L. In a powder this produces a
characteristic asymmetric lineshape whose width and singularity ("horn"/"foot")
positions encode the quadrupolar coupling constant C_Q and asymmetry η_Q — the
physics a plain (pseudo-)Voigt fit cannot recover.

This module provides the forward model (orientation-averaged powder lineshape,
static or infinite-MAS) and a least-squares fit that returns δ_iso (chemical
shift), C_Q, η_Q, and a residual Gaussian/Lorentzian broadening. It is the NMR
analogue of ``curve_fitting/epr/axial_powder.py`` and is registered as a
``TOOL_SPEC`` so the curve-fitting agent's generated script can call it.

Scope (v0): one central-transition site under the second-order quadrupolar
interaction (static or fast-MAS limit). Out of scope: satellite transitions,
multiple sites (fit them by summing sites in the caller), CSA–quadrupolar
cross-terms, finite-spinning sideband manifolds, and dynamic/relaxation
effects. Solution-state quadrupolar nuclei tumble fast and give a plain
Lorentzian — do NOT use this model there.

Convention: frequency axis in ppm, decreasing left→right (NMR standard). The
Larmor frequency ν_L (MHz) is REQUIRED — it sets the ν_Q²/ν_L scaling and the
ppm↔Hz conversion, and must come from the spectrometer metadata, never a guess.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..._shared._spec import ToolSpec

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
#  Second-order quadrupolar angular coefficients
#
#  The central-transition frequency offset for a crystallite at polar angles
#  (θ, φ) in the EFG principal-axis frame is
#
#      Δν(θ,φ) = -(ν_Q² / ν_L) · [I(I+1) - 3/4] · (1/6)
#                 · [ A(η,φ)·cos⁴θ + B(η,φ)·cos²θ + C(η,φ) ]
#
#  with ν_Q = 3·C_Q / [2 I (2I-1)]. The (A,B,C) coefficient sets below are the
#  standard second-order results: the STATIC set (Baugher et al., 1969) and the
#  infinite-MAS set, in which the 2nd-rank term is averaged out and only the
#  reduced 4th-rank pattern survives.
# --------------------------------------------------------------------------

def _abc_static(eta: float, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2 = np.cos(2.0 * phi)
    A = -27.0 / 8.0 + (9.0 / 4.0) * eta * c2 - (3.0 / 8.0) * (eta * c2) ** 2
    B = 30.0 / 8.0 - 0.5 * eta**2 - 2.0 * eta * c2 + (3.0 / 4.0) * (eta * c2) ** 2
    C = -3.0 / 8.0 + (1.0 / 3.0) * eta**2 + 0.25 * eta * c2 - (3.0 / 8.0) * (eta * c2) ** 2
    return A, B, C


def _abc_mas(eta: float, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Infinite-spinning (fast-MAS) limit — 4th-rank-only averaged coefficients.
    c2 = np.cos(2.0 * phi)
    A = 21.0 / 16.0 - (7.0 / 8.0) * eta * c2 + (7.0 / 48.0) * (eta * c2) ** 2
    B = -9.0 / 8.0 + (1.0 / 12.0) * eta**2 + eta * c2 - (7.0 / 24.0) * (eta * c2) ** 2
    C = 5.0 / 16.0 - (1.0 / 18.0) * eta**2 - (1.0 / 8.0) * eta * c2 + (7.0 / 48.0) * (eta * c2) ** 2
    return A, B, C


def _nu_q_hz(Cq_MHz: float, I: float) -> float:
    """Quadrupolar splitting ν_Q = 3 C_Q / [2 I (2I-1)] in Hz (C_Q given in MHz)."""
    return 3.0 * (Cq_MHz * 1e6) / (2.0 * I * (2.0 * I - 1.0))


def simulate_quad_ct(
    ppm: Sequence[float],
    delta_iso: float,
    Cq_MHz: float,
    eta: float,
    nu_L_MHz: float,
    I: float = 1.5,
    lw_gauss_ppm: float = 0.5,
    lw_lorentz_ppm: float = 0.5,
    amplitude: float = 1.0,
    mas: bool = True,
    n_theta: int = 200,
    n_phi: int = 120,
) -> np.ndarray:
    """Second-order quadrupolar central-transition powder lineshape on ``ppm``.

    Orientation-averages the analytic frequency offset over a (θ, φ) grid
    (sin θ weighting), histograms onto the ppm axis, then convolves with a
    Voigt (Gaussian ⊗ Lorentzian) instrumental/residual broadening. ``delta_iso``
    is the *chemical* shift (ppm); the field-dependent quadrupole-induced shift
    is produced by the powder average itself. Returns intensity on ``ppm``.
    """
    ppm = np.asarray(ppm, dtype=float)
    nu_L = nu_L_MHz * 1e6
    nu_q = _nu_q_hz(Cq_MHz, I)
    pref = -(nu_q**2 / nu_L) * (I * (I + 1.0) - 0.75) / 6.0  # Hz

    theta = np.linspace(1e-4, np.pi - 1e-4, n_theta)
    phi = np.linspace(0.0, np.pi / 2.0, n_phi)  # η symmetry over a quadrant
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    abc = _abc_mas(eta, ph) if mas else _abc_static(eta, ph)
    A, B, C = abc
    cos2 = np.cos(th) ** 2
    shift_hz = pref * (A * cos2**2 + B * cos2 + C)          # Δν(θ,φ) in Hz
    shift_ppm = shift_hz / nu_L_MHz                          # Hz / (MHz) = ppm
    nu_ppm = delta_iso + shift_ppm
    weight = np.sin(th)

    # Histogram the powder distribution onto a fine internal grid spanning the
    # data range, then convolve. Build edges from the data axis (sorted asc).
    lo, hi = float(ppm.min()), float(ppm.max())
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, max(4096, ppm.size))
    dgrid = grid[1] - grid[0]
    hist, _ = np.histogram(nu_ppm.ravel(), bins=grid.size,
                           range=(grid[0] - dgrid / 2, grid[-1] + dgrid / 2),
                           weights=weight.ravel())

    # Voigt broadening kernel on the same grid (centered).
    kx = np.arange(-grid.size // 2, grid.size // 2) * dgrid
    sigma = max(lw_gauss_ppm, 1e-6) / 2.3548
    gamma = max(lw_lorentz_ppm, 1e-6) / 2.0
    gauss = np.exp(-0.5 * (kx / sigma) ** 2)
    lorentz = (gamma**2) / (kx**2 + gamma**2)
    kernel = np.convolve(gauss, lorentz, mode="same")
    kernel /= kernel.sum() + 1e-30
    broadened = np.convolve(hist, kernel, mode="same")

    y = np.interp(ppm, grid, broadened)
    m = y.max()
    if m > 0:
        y = y / m * amplitude
    return y


# --------------------------------------------------------------------------
#  Fit
# --------------------------------------------------------------------------

def fit_quad_ct(
    ppm: Sequence[float],
    intensity: Sequence[float],
    nu_L_MHz: float,
    I: float = 1.5,
    mas: bool = True,
    delta_iso_init: float | None = None,
    Cq_MHz_init: float = 1.5,
    eta_init: float = 0.5,
    lw_ppm_init: float = 1.0,
    bounds_Cq_MHz: tuple[float, float] = (0.0, 10.0),
    bounds_delta_iso: tuple[float, float] | None = None,
    n_theta: int = 160,
    n_phi: int = 90,
) -> dict[str, Any]:
    """Fit a single second-order quadrupolar central-transition powder pattern.

    Returns ``parameters`` (delta_iso_ppm, Cq_MHz, eta, P_Q_MHz, lw_gauss_ppm,
    lw_lorentz_ppm, amplitude, baseline), ``derived`` (the quadrupole-induced
    isotropic shift δ_QIS in ppm, and the centre of gravity), ``fit_quality``
    (r_squared, rmse), and ``y_fit``. Background should be removed by the caller
    first; a constant offset is fit here as a safety net.
    """
    x = np.asarray(ppm, dtype=float)
    y = np.asarray(intensity, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    yspan = float(np.nanmax(y) - np.nanmin(y)) or 1.0

    if delta_iso_init is None:
        delta_iso_init = float(x[np.argmax(y)])
    if bounds_delta_iso is None:
        bounds_delta_iso = (float(x.min()), float(x.max()))

    # p = [delta_iso, Cq, eta, lw_gauss, lw_lorentz, amplitude, baseline]
    lo = [bounds_delta_iso[0], bounds_Cq_MHz[0], 0.0, 1e-3, 1e-3, 0.0, -abs(yspan)]
    hi = [bounds_delta_iso[1], bounds_Cq_MHz[1], 1.0, 50.0, 50.0, 10.0 * yspan, abs(yspan)]

    def model(p):
        d, cq, eta, lg, ll, amp, base = p
        return base + simulate_quad_ct(
            x, d, cq, eta, nu_L_MHz, I=I, lw_gauss_ppm=lg, lw_lorentz_ppm=ll,
            amplitude=amp, mas=mas, n_theta=n_theta, n_phi=n_phi)

    def resid(p):
        return model(p) - y

    # --- Coarse (Cq, eta) grid pre-search: the second-order quadrupolar
    # objective is multimodal in (Cq, eta), so a single local start is
    # unreliable. For each node, ALIGN the trial lineshape's peak to the data
    # peak before scoring — the visible singularity is offset from delta_iso by
    # a (Cq, eta)-dependent amount, so a fixed delta_iso would bias the scan
    # toward small Cq. Score the best linear amp+baseline, then multi-start the
    # polish from the top few nodes and keep the best. ---
    cq_grid = np.linspace(max(bounds_Cq_MHz[0], 0.1), bounds_Cq_MHz[1], 14)
    eta_grid = np.linspace(0.0, 1.0, 6)
    base0, amp0 = float(np.nanmin(y)), float(np.nanmax(y) - np.nanmin(y)) or 1.0
    data_peak = float(x[np.argmax(y)])
    nodes = []
    for cq in cq_grid:
        for et in eta_grid:
            # simulate at delta_iso=data_peak, find the trial peak, realign.
            s0 = simulate_quad_ct(x, data_peak, cq, et, nu_L_MHz, I=I,
                                  lw_gauss_ppm=lw_ppm_init, lw_lorentz_ppm=lw_ppm_init,
                                  amplitude=1.0, mas=mas, n_theta=64, n_phi=36)
            d_node = data_peak + (data_peak - float(x[np.argmax(s0)]))
            shape = simulate_quad_ct(x, d_node, cq, et, nu_L_MHz, I=I,
                                     lw_gauss_ppm=lw_ppm_init, lw_lorentz_ppm=lw_ppm_init,
                                     amplitude=1.0, mas=mas, n_theta=64, n_phi=36)
            G = np.vstack([shape, np.ones_like(shape)]).T
            coef, *_ = np.linalg.lstsq(G, y, rcond=None)
            sse = float(np.sum((G @ coef - y) ** 2))
            nodes.append((sse, d_node, cq, et))
    nodes.sort(key=lambda t: t[0])

    best_res, best_cost = None, np.inf
    for _, d_node, cq_seed, eta_seed in nodes[:3]:   # multi-start polish
        p0 = [d_node, cq_seed, eta_seed, lw_ppm_init, lw_ppm_init, amp0, base0]
        p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
        r = least_squares(resid, p0, bounds=(lo, hi), method="trf",
                          x_scale="jac", max_nfev=400)
        if r.cost < best_cost:
            best_cost, best_res = r.cost, r
    res = best_res
    yfit = model(res.x)
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-30
    r2 = 1.0 - ss_res / ss_tot

    d, cq, eta, lg, ll, amp, base = res.x
    Pq = cq * np.sqrt(1.0 + eta**2 / 3.0)
    # Quadrupole-induced isotropic shift of the centre of gravity (ppm).
    nu_q = _nu_q_hz(cq, I)
    qis_ppm = -(nu_q**2 / (nu_L_MHz * 1e6)) * (I * (I + 1.0) - 0.75) / 30.0 \
        * (1.0 + eta**2 / 3.0) / (nu_L_MHz)

    # --- Reliability guard. The central-transition lineshape determines C_Q
    # only when the second-order quadrupolar broadening is RESOLVED above the
    # instrumental/residual broadening. When the two are comparable, C_Q and the
    # linewidth are degenerate (a smaller C_Q + larger lw reproduces the same
    # smooth line), so the returned C_Q is at best an upper bound — confirm with
    # MQMAS or the satellite transitions. Flag that regime rather than emit a
    # falsely precise number. ---
    quad_width_ppm = abs((nu_q**2 / (nu_L_MHz * 1e6)) * (I * (I + 1.0) - 0.75)
                         / nu_L_MHz)  # ~second-order CT width scale, ppm
    broad_ppm = max(lg, ll)
    resolved = quad_width_ppm > 1.5 * broad_ppm
    flags = []
    if not resolved:
        flags.append("Cq_unreliable: broadening-dominated central line; C_Q is an "
                     "upper bound (C_Q–linewidth degenerate). Confirm via MQMAS / "
                     "satellite transitions.")
    if eta < 0.02 or eta > 0.98:
        flags.append(f"eta railed to {eta:.2f}; poorly determined.")
    return {
        "parameters": {
            "delta_iso_ppm": float(d), "Cq_MHz": float(cq), "eta": float(eta),
            "P_Q_MHz": float(Pq), "lw_gauss_ppm": float(lg),
            "lw_lorentz_ppm": float(ll), "amplitude": float(amp),
            "baseline": float(base),
        },
        "derived": {
            "delta_QIS_ppm": float(qis_ppm),
            "centre_of_gravity_ppm": float(d + qis_ppm),
            "spin_I": I, "regime": "MAS" if mas else "static",
            "Cq_resolved": bool(resolved),
            "reliability_flags": flags,
        },
        "fit_quality": {"r_squared": float(r2),
                        "rmse": float(np.sqrt(ss_res / max(len(y), 1)))},
        "y_fit": yfit.tolist(),
        "nu_L_MHz": nu_L_MHz,
        "model_used": f"2nd-order quadrupolar CT ({'MAS' if mas else 'static'}), I={I}",
    }


# --------------------------------------------------------------------------
#  Czjzek (distribution of EFGs) central-transition lineshape
#
#  Disordered / amorphous / glassy solids (and many defective or solid-solution
#  crystalline phases) do not have ONE well-defined (C_Q, η); each site sees a
#  slightly different electric-field gradient, so the central-transition line is
#  the AVERAGE over a *distribution* of (C_Q, η). A single-site `fit_quad_ct`
#  (sharp horns) or a plain pseudo-Voigt cannot reproduce the resulting smooth,
#  skewed line — it leaves systematic residual structure (the failure mode the
#  quality flag `residual_structured` flags on broad ²³Na/⁶⁷Zn solids).
#
#  The Czjzek Gaussian Isotropic Model (Czjzek et al., 1981) gives the standard
#  EFG distribution, parametrized by a single width σ (MHz):
#
#      P(C_Q, η) ∝ C_Q⁴ · η · (1 − η²/9) · exp[ −C_Q²(1 + η²/3) / (2σ²) ]
#
#  The observed line is Σ over that distribution of the single-site CT lineshape.
#  Because the CT pattern translates rigidly with δ_iso, we precompute a fixed
#  (C_Q, η) lineshape basis ONCE and only re-weight it as σ varies — so the fit
#  stays cheap.
# --------------------------------------------------------------------------

def _czjzek_weights(cq_grid: np.ndarray, eta_grid: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Czjzek GIM weights on the (C_Q, η) grid for width ``sigma`` (MHz)."""
    Cq, Eta = np.meshgrid(cq_grid, eta_grid, indexing="ij")
    sig = max(float(sigma), 1e-6)
    P = (Cq ** 4) * Eta * (1.0 - Eta ** 2 / 9.0) \
        * np.exp(-(Cq ** 2) * (1.0 + Eta ** 2 / 3.0) / (2.0 * sig ** 2))
    P = np.clip(P, 0.0, None)
    s = float(P.sum())
    return P / s if s > 0 else P


def _czjzek_basis(grid_ppm, ref_delta, cq_grid, eta_grid, nu_L_MHz, I, mas,
                  lw_intrinsic, n_theta, n_phi) -> np.ndarray:
    """Area-normalized single-site CT lineshapes for every (C_Q, η), at ``ref_delta``."""
    basis = np.empty((len(cq_grid), len(eta_grid), grid_ppm.size), float)
    for i, cq in enumerate(cq_grid):
        for j, et in enumerate(eta_grid):
            L = simulate_quad_ct(grid_ppm, ref_delta, max(cq, 1e-3), et, nu_L_MHz, I=I,
                                 lw_gauss_ppm=lw_intrinsic, lw_lorentz_ppm=lw_intrinsic,
                                 amplitude=1.0, mas=mas, n_theta=n_theta, n_phi=n_phi)
            s = float(L.sum())
            basis[i, j] = L / s if s > 0 else L
    return basis


def simulate_quad_czjzek(
    ppm: Sequence[float],
    delta_iso: float,
    sigma_cz_MHz: float,
    nu_L_MHz: float,
    I: float = 1.5,
    lw_gauss_ppm: float = 0.5,
    lw_lorentz_ppm: float = 0.5,
    amplitude: float = 1.0,
    mas: bool = True,
    n_cq: int = 24,
    n_eta: int = 6,
    cq_max_MHz: float | None = None,
    n_theta: int = 48,
    n_phi: int = 24,
) -> np.ndarray:
    """Czjzek (distribution-of-EFG) central-transition lineshape on ``ppm``.

    Averages the single-site CT lineshape over the Czjzek GIM distribution of
    (C_Q, η) with width ``sigma_cz_MHz``, then applies residual Voigt broadening.
    Returns intensity on ``ppm``. (For fitting use ``fit_quad_czjzek``, which
    caches the basis.)
    """
    ppm = np.asarray(ppm, float)
    cq_max = cq_max_MHz or max(4.0 * sigma_cz_MHz, 0.5)
    cq_grid = np.linspace(0.05, cq_max, n_cq)
    eta_grid = np.linspace(0.0, 1.0, n_eta)
    lo, hi = float(ppm.min()), float(ppm.max())
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, min(8192, max(2048, ppm.size)))
    dgrid = grid[1] - grid[0]
    basis = _czjzek_basis(grid, delta_iso, cq_grid, eta_grid, nu_L_MHz, I, mas,
                          max(3 * dgrid, 1e-6), n_theta, n_phi)
    w = _czjzek_weights(cq_grid, eta_grid, sigma_cz_MHz)
    czj = np.tensordot(w, basis, axes=([0, 1], [0, 1]))
    kx = np.arange(-grid.size // 2, grid.size // 2) * dgrid
    sigma = max(lw_gauss_ppm, 1e-6) / 2.3548
    gamma = max(lw_lorentz_ppm, 1e-6) / 2.0
    kernel = np.convolve(np.exp(-0.5 * (kx / sigma) ** 2),
                         (gamma ** 2) / (kx ** 2 + gamma ** 2), mode="same")
    kernel /= kernel.sum() + 1e-30
    czj = np.convolve(czj, kernel, mode="same")
    y = np.interp(ppm, grid, czj)
    m = y.max()
    return (y / m * amplitude) if m > 0 else y


def fit_quad_czjzek(
    ppm: Sequence[float],
    intensity: Sequence[float],
    nu_L_MHz: float,
    I: float = 1.5,
    mas: bool = True,
    delta_iso_init: float | None = None,
    bounds_sigma_cz_MHz: tuple[float, float] = (0.02, 4.0),
    lw_ppm_init: float = 1.0,
    bounds_delta_iso: tuple[float, float] | None = None,
    n_cq: int = 24,
    n_eta: int = 6,
    n_theta: int = 64,
    n_phi: int = 36,
) -> dict[str, Any]:
    """Fit a Czjzek distribution-broadened central-transition lineshape.

    For a DISORDERED half-integer quadrupolar solid whose line is broad and
    smoothly skewed (no sharp single-site horns) — the model averages the CT
    lineshape over a Czjzek (C_Q, η) distribution of width σ. Returns
    ``parameters`` (delta_iso_ppm, sigma_cz_MHz, mean_Cq_MHz, broadening,
    amplitude, baseline), ``fit_quality`` (r_squared, rmse), and ``y_fit``.
    Crop to the central transition and remove the background first.
    """
    x = np.asarray(ppm, float)
    y = np.asarray(intensity, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    yspan = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    if delta_iso_init is None:
        delta_iso_init = float(x[np.argmax(y)])
    if bounds_delta_iso is None:
        bounds_delta_iso = (float(x.min()), float(x.max()))

    # Fixed (C_Q, η) basis at a reference δ — large enough in C_Q to cover the
    # upper σ bound. The CT pattern translates rigidly with δ_iso, so we shift
    # the precomputed sum rather than rebuild the basis each iteration.
    ref = delta_iso_init
    cq_max = 4.0 * bounds_sigma_cz_MHz[1]
    cq_grid = np.linspace(0.05, cq_max, n_cq)
    eta_grid = np.linspace(0.0, 1.0, n_eta)
    lo, hi = float(x.min()), float(x.max())
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, min(8192, max(2048, x.size)))
    dgrid = grid[1] - grid[0]
    basis = _czjzek_basis(grid, ref, cq_grid, eta_grid, nu_L_MHz, I, mas,
                          max(3 * dgrid, 1e-6), n_theta, n_phi)
    kx = np.arange(-grid.size // 2, grid.size // 2) * dgrid

    def _shape(delta, sigma_cz, lw_g, lw_l):
        czj = np.tensordot(_czjzek_weights(cq_grid, eta_grid, sigma_cz), basis,
                           axes=([0, 1], [0, 1]))
        if abs(delta - ref) > 1e-9:                      # rigid translation
            czj = np.interp(grid - (delta - ref), grid, czj, left=0.0, right=0.0)
        sg = max(lw_g, 1e-6) / 2.3548
        gm = max(lw_l, 1e-6) / 2.0
        ker = np.convolve(np.exp(-0.5 * (kx / sg) ** 2),
                          (gm ** 2) / (kx ** 2 + gm ** 2), mode="same")
        ker /= ker.sum() + 1e-30
        czj = np.convolve(czj, ker, mode="same")
        s = np.interp(x, grid, czj)
        m = s.max()
        return s / m if m > 0 else s

    # p = [delta_iso, sigma_cz, lw_gauss, lw_lorentz, amplitude, baseline]
    lo_b = [bounds_delta_iso[0], bounds_sigma_cz_MHz[0], 1e-3, 1e-3, 0.0, -abs(yspan)]
    hi_b = [bounds_delta_iso[1], bounds_sigma_cz_MHz[1], 50.0, 50.0, 10.0 * yspan, abs(yspan)]

    def resid(p):
        d, scz, lg, ll, amp, base = p
        return (base + amp * _shape(d, scz, lg, ll)) - y

    base0, amp0 = float(np.nanmin(y)), yspan
    best, best_cost = None, np.inf
    for scz_seed in np.linspace(bounds_sigma_cz_MHz[0] * 2, bounds_sigma_cz_MHz[1] * 0.6, 4):
        p0 = [delta_iso_init, scz_seed, lw_ppm_init, lw_ppm_init, amp0, base0]
        p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo_b, hi_b)]
        try:
            r = least_squares(resid, p0, bounds=(lo_b, hi_b), method="trf",
                              x_scale="jac", max_nfev=300)
        except Exception:
            continue
        if r.cost < best_cost:
            best_cost, best = r.cost, r
    res = best
    yfit = res.x[5] + res.x[4] * _shape(res.x[0], res.x[1], res.x[2], res.x[3])
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-30
    r2 = 1.0 - ss_res / ss_tot

    d, scz, lg, ll, amp, base = res.x
    w = _czjzek_weights(cq_grid, eta_grid, scz)
    mean_cq = float(np.tensordot(w, np.meshgrid(cq_grid, eta_grid, indexing="ij")[0],
                                 axes=([0, 1], [0, 1])))
    return {
        "parameters": {
            "delta_iso_ppm": float(d), "sigma_cz_MHz": float(scz),
            "mean_Cq_MHz": mean_cq, "lw_gauss_ppm": float(lg),
            "lw_lorentz_ppm": float(ll), "amplitude": float(amp), "baseline": float(base),
        },
        "derived": {"spin_I": I, "regime": "MAS" if mas else "static",
                    "model": "Czjzek GIM distribution of (C_Q, η)"},
        "fit_quality": {"r_squared": float(r2),
                        "rmse": float(np.sqrt(ss_res / max(len(y), 1)))},
        "y_fit": yfit.tolist(),
        "nu_L_MHz": nu_L_MHz,
        "model_used": f"Czjzek distribution CT ({'MAS' if mas else 'static'}), I={I}",
    }


# --------------------------------------------------------------------------
#  Tool registry
# --------------------------------------------------------------------------

TOOL_SPEC_SIMULATE = ToolSpec(
    name="simulate_quad_ct",
    description=(
        "Simulate a second-order quadrupolar central-transition powder "
        "lineshape (static or fast-MAS) for a half-integer spin (²³Na I=3/2, "
        "²⁷Al I=5/2, ¹⁷O I=5/2, …). Returns intensity on a ppm axis."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.quadrupolar import simulate_quad_ct",
    signature=(
        "simulate_quad_ct(ppm, delta_iso, Cq_MHz, eta, nu_L_MHz, I=1.5, "
        "lw_gauss_ppm=0.5, lw_lorentz_ppm=0.5, amplitude=1.0, mas=True) -> ndarray"
    ),
    parameters={
        "ppm": {"type": "list[float]", "description": "Chemical-shift axis (ppm)."},
        "delta_iso": {"type": "float", "description": "Isotropic CHEMICAL shift (ppm)."},
        "Cq_MHz": {"type": "float", "description": "Quadrupolar coupling constant C_Q (MHz)."},
        "eta": {"type": "float", "description": "EFG asymmetry η ∈ [0, 1]."},
        "nu_L_MHz": {"type": "float", "description": "Larmor frequency of the nucleus (MHz) — from metadata."},
        "I": {"type": "float", "description": "Nuclear spin (1.5 for ²³Na; 2.5 for ²⁷Al/¹⁷O)."},
        "lw_gauss_ppm": {"type": "float", "description": "Residual Gaussian FWHM (ppm)."},
        "lw_lorentz_ppm": {"type": "float", "description": "Residual Lorentzian FWHM (ppm)."},
        "amplitude": {"type": "float", "description": "Peak amplitude scale."},
        "mas": {"type": "bool", "description": "True = infinite-MAS lineshape; False = static powder."},
    },
    required=["ppm", "delta_iso", "Cq_MHz", "eta", "nu_L_MHz"],
    returns="ndarray, central-transition intensity on the input ppm axis.",
    when_to_use=(
        "Forward modeling a SOLID-state half-integer quadrupolar central "
        "transition. NOT for solution-state (fast tumbling → plain Lorentzian)."
    ),
)

TOOL_SPEC_FIT = ToolSpec(
    name="fit_quad_ct",
    description=(
        "Least-squares fit of a solid-state second-order quadrupolar "
        "central-transition powder lineshape. Returns δ_iso (chemical shift), "
        "C_Q, η_Q, the quadrupolar product P_Q, residual broadening, R², and the "
        "quadrupole-induced isotropic shift — the physics a (pseudo-)Voigt fit "
        "cannot recover."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.quadrupolar import fit_quad_ct",
    signature=(
        "fit_quad_ct(ppm, intensity, nu_L_MHz, I=1.5, mas=True, "
        "delta_iso_init=None, Cq_MHz_init=1.5, eta_init=0.5, lw_ppm_init=1.0, "
        "bounds_Cq_MHz=(0,10), bounds_delta_iso=None, n_theta=160, n_phi=90) -> dict"
    ),
    parameters={
        "ppm": {"type": "list[float]", "description": "Chemical-shift axis (ppm). Crop to the central-transition region before calling."},
        "intensity": {"type": "list[float]", "description": "Spectrum intensity (background pre-removed)."},
        "nu_L_MHz": {"type": "float", "description": "Larmor frequency (MHz) — REQUIRED, from metadata."},
        "I": {"type": "float", "description": "Nuclear spin: 1.5 for ²³Na/¹¹B/⁷¹Ga; 2.5 for ²⁷Al/¹⁷O/²⁵Mg; 3.5 for ⁴⁵Sc/⁵¹V. Set it to the observed nucleus."},
        "mas": {"type": "bool", "description": "True for magic-angle-spinning data (infinite-MAS lineshape); False for a static/wideline powder. Match the acquisition."},
        "delta_iso_init": {"type": "float", "description": "Initial chemical-shift guess (ppm); default = peak max. Set it near the high-frequency 'horn' of the lineshape when the default seeds poorly."},
        "Cq_MHz_init": {"type": "float", "description": "Initial C_Q guess (MHz). The grid search covers the bounds, so this mainly matters if you also narrow the bounds."},
        "eta_init": {"type": "float", "description": "Initial η guess ∈ [0,1]."},
        "lw_ppm_init": {"type": "float", "description": "Initial residual (Gaussian+Lorentzian) broadening in ppm (default 1.0). Set near the apparent extra broadening beyond the quadrupolar lineshape; larger for disordered/amorphous samples."},
        "bounds_Cq_MHz": {"type": "tuple", "description": "(lo, hi) bounds for C_Q in MHz (default (0,10)). NARROW it around a literature/expected value to stabilise the fit; RAISE the upper bound for large-C_Q nuclei (e.g. ²⁷Al up to ~20)."},
        "bounds_delta_iso": {"type": "tuple", "description": "(lo, hi) bounds for the chemical shift (ppm); default = data range. Narrow it to the plausible shift window for the nucleus/material to avoid the fit wandering onto a sideband."},
        "n_theta": {"type": "int", "description": "Polar-angle powder-averaging grid (default 160). Increase (e.g. 320) for a sharper, well-resolved lineshape; decrease for speed on broad lines."},
        "n_phi": {"type": "int", "description": "Azimuthal powder-averaging grid (default 90). Increase together with n_theta when η is large and the lineshape is structured."},
    },
    required=["ppm", "intensity", "nu_L_MHz"],
    returns=(
        "dict with 'parameters' (delta_iso_ppm, Cq_MHz, eta, P_Q_MHz, "
        "linewidths), 'derived' (delta_QIS_ppm, centre_of_gravity_ppm), "
        "'fit_quality' (r_squared, rmse), and 'y_fit'."
    ),
    when_to_use=(
        "Any SOLID-state half-integer quadrupolar central transition where C_Q/η_Q "
        "is the deliverable — run it by DEFAULT rather than pre-judging a line as "
        "'too symmetric to bother'. It returns Cq_resolved=False when the line "
        "cannot determine C_Q, which is the signal to fall back to reporting a "
        "linewidth. NOT for solution / motionally-narrowed lines (fast tumbling "
        "gives a plain Lorentzian/Voigt)."
    ),
)

TOOL_SPEC_CZJZEK = ToolSpec(
    name="fit_quad_czjzek",
    description=(
        "Fit a Czjzek distribution-broadened second-order quadrupolar "
        "central-transition lineshape — for a DISORDERED / amorphous / glassy or "
        "solid-solution half-integer quadrupolar solid whose line is broad and "
        "smoothly skewed with NO sharp single-site horns. Models the line as an "
        "average of CT lineshapes over a Czjzek (C_Q, η) distribution of width σ, "
        "which a single-site fit_quad_ct or a pseudo-Voigt cannot reproduce. "
        "Returns σ_Cz (the EFG-distribution width) and the mean C_Q."
    ),
    import_line="from scilink.skills.curve_fitting.nmr.quadrupolar import fit_quad_czjzek",
    signature=(
        "fit_quad_czjzek(ppm, intensity, nu_L_MHz, I=1.5, mas=True, "
        "delta_iso_init=None, bounds_sigma_cz_MHz=(0.02,4.0), lw_ppm_init=1.0, "
        "bounds_delta_iso=None, n_cq=24, n_eta=6, n_theta=64, n_phi=36) -> dict"
    ),
    parameters={
        "ppm": {"type": "list[float]", "description": "Chemical-shift axis (ppm). Crop to the central-transition region first."},
        "intensity": {"type": "list[float]", "description": "Spectrum intensity (background pre-removed)."},
        "nu_L_MHz": {"type": "float", "description": "Larmor frequency (MHz) — REQUIRED, from metadata."},
        "I": {"type": "float", "description": "Nuclear spin: 1.5 for ²³Na/⁶⁷Zn (⁶⁷Zn is I=5/2 — set 2.5); 2.5 for ²⁷Al/¹⁷O/²⁵Mg. Set to the observed nucleus."},
        "mas": {"type": "bool", "description": "True for MAS data (infinite-MAS lineshape); False for static/wideline. Match the acquisition."},
        "delta_iso_init": {"type": "float", "description": "Initial chemical-shift guess (ppm); default = peak max."},
        "bounds_sigma_cz_MHz": {"type": "tuple", "description": "(lo, hi) bounds for the Czjzek width σ in MHz (default (0.02, 4.0)). RAISE the upper bound for very broad lines / large-C_Q nuclei; the internal C_Q grid auto-scales to 4× this upper bound."},
        "lw_ppm_init": {"type": "float", "description": "Initial residual (instrumental/extra) broadening in ppm (default 1.0)."},
        "bounds_delta_iso": {"type": "tuple", "description": "(lo, hi) bounds for the chemical shift (ppm); default = data range."},
        "n_cq": {"type": "int", "description": "C_Q grid points for the distribution average (default 24). Increase for a sharper σ estimate; decrease for speed."},
        "n_eta": {"type": "int", "description": "η grid points for the distribution average (default 6)."},
        "n_theta": {"type": "int", "description": "Polar powder-averaging grid for each basis lineshape (default 64)."},
        "n_phi": {"type": "int", "description": "Azimuthal powder-averaging grid (default 36)."},
    },
    required=["ppm", "intensity", "nu_L_MHz"],
    returns=(
        "dict with 'parameters' (delta_iso_ppm, sigma_cz_MHz, mean_Cq_MHz, "
        "linewidths, amplitude, baseline), 'fit_quality' (r_squared, rmse), and "
        "'y_fit'."
    ),
    when_to_use=(
        "A DISORDERED-solid half-integer quadrupolar line (amorphous / glassy / "
        "solid-solution / heavily defective), where the EFG-distribution width "
        "σ_Cz is the deliverable. Fit it alongside single-site fit_quad_ct and keep "
        "whichever fits better: a single well-defined site favors fit_quad_ct, a "
        "distribution of sites (washed-out horns, or systematic residual left by a "
        "single site) favors this. NOT for solution / motionally-narrowed lines."
    ),
)

TOOL_SPECS = [TOOL_SPEC_SIMULATE, TOOL_SPEC_FIT, TOOL_SPEC_CZJZEK]
