"""Per-pixel absorption-edge STEP measurement for spectrum-imaging cubes.

Exposed to the hyperspectral code-gen sandbox as ``measure_edge_step(...)``.
It isolates the discontinuous jump in optical depth across an absorption edge
(K/L X-ray edge, XANES edge, EELS core-loss onset) using *wide, edge-offset*
continuum windows and a photon-flux gate — the two things improvised per-run
code kept getting wrong (tight adjacent windows + smoothing under-measure the
jump ~2x; no flux gate lets a starved edge fabricate a coherent-looking signal).

It returns the edge step ``Delta(mu*t)`` (a.k.a. Delta-OD) and its SNR — NOT a
thickness. Downstream converts to a physical quantity itself, e.g.
``t = edge_step / (delta_mu_over_rho * rho)`` with mu/rho from ``attenuation()``.
Keeping thickness out of this primitive is deliberate: the edge step is generic
to edge spectroscopy; the conversion is domain-specific and lives in a skill.

Lives in ``_shared`` because edge-jump extraction is generic physics reusable
across X-ray transmission, XANES, and EELS core-loss work.
"""
import numpy as np


def _linfit_extrap(E_win, OD_win, x0):
    """Vectorized per-pixel linear fit OD(E)=a+b*E over a window, extrapolated
    to energy ``x0``. ``OD_win`` is (npix, nwin); returns (npix,) at x0."""
    X = np.column_stack([np.ones_like(E_win), E_win])   # (nwin, 2)
    beta, *_ = np.linalg.lstsq(X, OD_win.T, rcond=None)  # (2, npix)
    return beta[0] + beta[1] * x0                         # a + b*x0


def measure_edge_step(
    data, i0, energy_kev, edge_kev,
    pre_gap=2.0, post_gap=1.0, win_width=4.0,
    low_e_cut_kev=5.0, flux_floor_counts=20.0,
    auto_center=True, search_tol_kev=2.0, t_floor=1e-6,
):
    """Measure the per-pixel absorption-edge step (jump in optical depth).

    Fits a local linear continuum on each side of the edge over *wide, offset*
    windows, extrapolates both to the edge energy, and returns the
    discontinuity ``Delta-OD = OD_above(edge) - OD_below(edge)`` per pixel —
    which cancels the smooth (non-edge) background. Also gates on post-edge
    photon flux so a flux-starved edge is reported as *not measurable* instead
    of returning bremsstrahlung-slope noise.

    Parameters
    ----------
    data : ndarray (H, W, E)
        Transmitted-intensity datacube (counts). Energy is the trailing axis.
    i0 : ndarray (H, W, E) or (E,)
        Incident-flux reference (flat field / I0). A 1-D spectrum is broadcast.
    energy_kev : array-like (E,)
        Energy of each channel in keV (the cube's own energy axis).
    edge_kev : float
        Nominal edge energy in keV (e.g. Au K-edge 80.8).
    pre_gap, post_gap : float
        keV offset of each continuum window from the edge. Keep the windows a
        little off the edge so finite edge width / detector broadening does not
        eat into the step. RAISE if the edge is broad or mis-centered; LOWER to
        hug the edge when the continuum is strongly curved.
    win_width : float
        Width (keV) of each continuum window. WIDEN to beat down noise on a
        clean edge; NARROW if the continuum is curved or fluorescence lines sit
        nearby (a wide window then biases the linear fit).
    low_e_cut_kev : float
        Ignore channels below this energy (noise / air-scatter).
    flux_floor_counts : float
        Measurability gate: if the median I0 counts in the post-edge window are
        below this, the above-edge region is photon-starved and the step is
        noise — the result is flagged ``measurable=False``. RAISE to be stricter
        (reject marginal edges), LOWER only if you trust low-count statistics.
    auto_center : bool
        If True, locate the actual edge channel as the steepest rise of the
        field-mean OD within ``search_tol_kev`` of ``edge_kev`` (handles small
        energy-axis miscalibration). If False, use ``edge_kev`` verbatim.
    search_tol_kev : float
        Half-width (keV) of the auto-center search window around ``edge_kev``.
        WIDEN if the energy axis may be off by more than a couple keV; NARROW
        (or set ``auto_center=False``) if a nearby fluorescence line has a
        steeper gradient than the edge and auto-center locks onto it.
    t_floor : float
        Lower clip on transmission before ``-ln`` (guards ``log(0)``). This is
        the saturation knob for THICK samples: above a strong edge, transmission
        can fall to this floor, saturating OD and biasing the step (hence the
        thickness) HIGH. RAISE it (e.g. 1e-4) if a thick coupon reads high /
        the post-edge window sits at the clip; LOWER it only if genuine deep
        absorption is being clipped on a thin sample.

    Returns
    -------
    dict with:
        edge_step : (H, W) float   -- Delta-OD per pixel (the jump; multiply by
                                      1/(delta_mu_over_rho*rho) for thickness).
        snr       : (H, W) float   -- edge_step / photon-noise sigma per pixel.
        measurable : bool          -- False if post-edge flux-starved or no
                                      field-level edge (do NOT trust the map).
        edge_kev_used : float      -- edge energy actually used (auto-centered).
        pre_window, post_window : (lo, hi) keV of the continuum windows.
        field_step : float         -- field-mean Delta-OD (edge present if >0).
        post_flux_counts : float   -- median I0 counts in the post window.
        reason : str               -- human-readable measurability verdict.
    """
    E = np.asarray(energy_kev, dtype=float)
    H, W, ne = data.shape
    npix = H * W
    D = data.reshape(npix, ne).astype(float)
    I0 = np.asarray(i0, dtype=float)
    I0 = (I0.reshape(npix, ne) if I0.ndim == 3 else np.broadcast_to(I0, (npix, ne)))

    valid_E = E >= low_e_cut_kev

    # Transmission -> optical depth, guarded.
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(I0 > 0, D / I0, 1.0)
    T = np.clip(T, t_floor, 1.0)
    OD = -np.log(T)

    # --- locate the edge (auto-center on steepest field-mean OD rise) ---
    edge_used = float(edge_kev)
    if auto_center:
        near = valid_E & (E >= edge_kev - search_tol_kev) & (E <= edge_kev + search_tol_kev)
        if near.sum() >= 3:
            od_mean = np.nanmean(OD, axis=0)
            grad = np.gradient(od_mean, E)
            cand = np.where(near)[0]
            edge_used = float(E[cand[np.argmax(grad[cand])]])

    # --- wide, offset continuum windows ---
    pre_lo, pre_hi = edge_used - pre_gap - win_width, edge_used - pre_gap
    post_lo, post_hi = edge_used + post_gap, edge_used + post_gap + win_width
    pre_mask = valid_E & (E >= pre_lo) & (E <= pre_hi)
    post_mask = valid_E & (E >= post_lo) & (E <= post_hi)

    reasons = []
    if pre_mask.sum() < 2 or post_mask.sum() < 2:
        return {
            "edge_step": np.zeros((H, W)), "snr": np.zeros((H, W)),
            "measurable": False, "edge_kev_used": edge_used,
            "pre_window": (pre_lo, pre_hi), "post_window": (post_lo, post_hi),
            "field_step": 0.0, "post_flux_counts": 0.0,
            "reason": "continuum window(s) fall outside the energy axis",
        }

    # --- flux gate: is there real photon flux above the edge? ---
    post_flux = float(np.median(np.nanmean(I0[:, post_mask], axis=1)))
    if post_flux < flux_floor_counts:
        reasons.append(
            f"post-edge flux {post_flux:.0f} < floor {flux_floor_counts:.0f} counts "
            "(photon-starved; step would be bremsstrahlung-slope noise)"
        )

    # --- per-pixel edge step via linear continuum extrapolation ---
    E_pre, E_post = E[pre_mask], E[post_mask]
    od_pre_edge = _linfit_extrap(E_pre, OD[:, pre_mask], edge_used)
    od_post_edge = _linfit_extrap(E_post, OD[:, post_mask], edge_used)
    edge_step = od_post_edge - od_pre_edge

    # --- SNR from pre-edge residual scatter (photon + continuum noise):
    # residual std on the pre window about its own linear fit ---
    X = np.column_stack([np.ones_like(E_pre), E_pre])
    beta_pre, *_ = np.linalg.lstsq(X, OD[:, pre_mask].T, rcond=None)
    fit_pre = (X @ beta_pre).T                       # (npix, nb)
    sigma = np.std(OD[:, pre_mask] - fit_pre, axis=1)
    nb, na = E_pre.size, E_post.size
    se = np.maximum(sigma, 1e-9) * np.sqrt(1.0 / nb + 1.0 / na)
    snr = edge_step / se

    field_step = float(np.nanmedian(edge_step))
    if field_step <= 0:
        reasons.append(f"field-median step {field_step:.3g} <= 0 (no coherent edge)")

    measurable = len(reasons) == 0
    return {
        "edge_step": edge_step.reshape(H, W),
        "snr": snr.reshape(H, W),
        "measurable": measurable,
        "edge_kev_used": edge_used,
        "pre_window": (pre_lo, pre_hi), "post_window": (post_lo, post_hi),
        "field_step": field_step, "post_flux_counts": post_flux,
        "reason": "measurable edge" if measurable else "; ".join(reasons),
    }


from ._spec import ToolSpec  # noqa: E402

TOOL_SPEC = ToolSpec(
    name="measure_edge_step",
    description=(
        "Per-pixel absorption-edge STEP (jump in optical depth) across a K/L "
        "X-ray edge (or XANES/EELS core-loss onset), using wide edge-offset "
        "continuum windows + a photon-flux gate. Returns Delta-OD and SNR (NOT "
        "thickness) plus a measurable flag — multiply by 1/(delta(mu/rho)*rho) "
        "for thickness. Removes the ~2x under-measurement from tight/smoothed "
        "windows and refuses flux-starved edges instead of fabricating a signal."
    ),
    signature=(
        "measure_edge_step(data, i0, energy_kev, edge_kev, pre_gap=2.0, "
        "post_gap=1.0, win_width=4.0, low_e_cut_kev=5.0, flux_floor_counts=20.0, "
        "auto_center=True, search_tol_kev=2.0) -> dict"
    ),
    import_line="from scilink.skills._shared.edge_step import measure_edge_step",
    parameters={
        "data": "Transmitted-intensity datacube (H,W,E), counts, energy trailing.",
        "i0": "Incident-flux reference (H,W,E) or 1-D (E,); flat field / I0.",
        "energy_kev": "Per-channel energy in keV (the cube's own axis).",
        "edge_kev": "Nominal edge energy in keV (e.g. Au K 80.8).",
        "pre_gap": "keV offset of the PRE-edge window from the edge. RAISE if the edge is broad or mis-centered (step reads low because the window eats into the rising edge); LOWER to hug the edge when the continuum is strongly curved over the offset span.",
        "post_gap": "keV offset of the POST-edge window from the edge. RAISE to clear edge broadening / white-line overshoot just above the edge; LOWER to stay in well-fluxed channels on a thick sample whose transmission dies fast above the edge.",
        "win_width": "keV width of each continuum window. WIDEN to cut noise on a clean edge (more channels in the linear fit); NARROW if the step reads biased because the window straddles continuum curvature or fluorescence lines.",
        "low_e_cut_kev": "Ignore channels below this energy. RAISE if low-E noise/air-scatter leaks in; rarely needs lowering.",
        "flux_floor_counts": "Field-level measurability gate: flag measurable=False if median post-edge I0 counts < this (photon-starved -> step is bremsstrahlung-slope noise). RAISE to reject marginal edges; LOWER only if you trust low-count statistics. NOTE: this is a coarse whole-field gate; per-pixel measurability is the returned snr map (threshold snr>~3).",
        "auto_center": "If True, snap the edge to the steepest field-mean OD rise within search_tol_kev (fixes small energy-axis miscalibration). Set False if a nearby fluorescence line is steeper than the true edge and auto-center locks onto it.",
        "search_tol_kev": "Half-width (keV) of the auto-center search window. WIDEN if axis calibration is uncertain; NARROW if a competing feature nearby is being picked instead of the edge.",
        "t_floor": "Transmission clip floor before -ln (saturation knob for THICK samples). RAISE (e.g. 1e-4) if a thick coupon reads HIGH or the post-edge window sits at the clip (OD saturates -> step biased high); LOWER only if genuine deep absorption on a thin sample is being clipped.",
    },
    required=["data", "i0", "energy_kev", "edge_kev"],
    agents=["hyperspectral"],
    when_to_use=(
        "Quantifying an absorption-edge jump per pixel (X-ray K-edge thickness, "
        "XANES edge height, EELS core-loss step). Use instead of hand-rolled "
        "pre/post window means — it picks robust windows, gates on flux, and "
        "reports whether the edge is measurable at all."
    ),
    returns=(
        "dict: edge_step (H,W Delta-OD), snr (H,W), measurable (bool), "
        "edge_kev_used, pre_window, post_window, field_step, post_flux_counts, reason."
    ),
    example=(
        "r = measure_edge_step(data, aux['baseline_I0'], energy_kev, 80.8)\n"
        "if r['measurable']:\n"
        "    dmu = attenuation('Au', np.array([81.6]))[0] - attenuation('Au', np.array([80.0]))[0]\n"
        "    thickness_um = r['edge_step'] / (dmu * 19.32) * 1e4"
    ),
)
