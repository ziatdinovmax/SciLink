# Auto-generated Script
# Task: For every pixel, form the transmission by dividing the sample cube by the same-grid I0 flat-field baseline, then compute the negative-log absorbance ln(I0/I). Restrict to energies above ~40 keV to avoid scatter/noise-dominated bremsstrahlung. Use the measure_edge_step helper to quantify the Au K-edge jump at 80.7 keV per pixel (it selects robust narrow pre-/post-edge windows, gates on flux, and reports whether the edge is measurable). Obtain the tabulated Au K-edge jump in the linear attenuation coefficient from the attenuation helper (element = Au) and convert the measured Delta(mu*t) edge step to areal thickness via t = Delta(mu*t) / Delta(mu_tab). Emit: an edge-jump/absorbance-step map, a per-pixel thickness map (in cm or mm), and a boolean/masked map marking where the Au edge is measurable (this locates the coupon). Report summary statistics (median thickness and spread) over the coupon region.

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    # --- Shapes ---
    H, W, E = data.shape
    energy_kev = np.asarray(axis, dtype=float)
    edge_kev = 80.7  # Au K-edge

    # --- I0 flat-field baseline ---
    i0 = None
    if auxiliary and "I0 flat-field baseline cube (no sample) on the same grid" in auxiliary:
        i0 = np.asarray(auxiliary["I0 flat-field baseline cube (no sample) on the same grid"], dtype=float)
    if i0 is None:
        # Fall back to a robust field-mean incident estimate (per-energy) if no baseline supplied.
        # Use the high-percentile transmitted intensity as a proxy for unattenuated flux.
        i0 = np.percentile(data.reshape(-1, E), 95, axis=0)[None, None, :] * np.ones((H, W, 1))

    # --- Light per-pixel denoising on the transmitted cube (preserve the edge) ---
    # Smooth only along the energy axis with a small Gaussian; clip negatives/zeros.
    dat = np.asarray(data, dtype=float)
    dat = np.clip(dat, 0.0, None)
    dat_s = gaussian_filter(dat, sigma=(0, 0, 1.5))
    i0_s = np.clip(np.asarray(i0, dtype=float), 1e-6, None)
    i0_s = gaussian_filter(i0_s, sigma=(0, 0, 1.5)) if i0_s.ndim == 3 else i0_s

    # --- Measure the Au K-edge step per pixel using the registered helper ---
    # Restrict low-energy noise via low_e_cut_kev (>~40 keV to avoid scatter/brems).
    r = measure_edge_step(
        dat_s, i0_s, energy_kev, edge_kev,
        pre_gap=2.0, post_gap=1.5, win_width=4.0,
        low_e_cut_kev=40.0, flux_floor_counts=20.0,
        auto_center=True, search_tol_kev=2.0,
    )

    edge_step = np.asarray(r["edge_step"], dtype=float)   # Delta(mu*t) = Delta-OD, (H,W)
    snr = np.asarray(r["snr"], dtype=float)
    edge_used = float(r.get("edge_kev_used", edge_kev))

    # --- Tabulated Au K-edge jump in linear attenuation coefficient ---
    # Evaluate mu just below and just above the (used) edge energy.
    rho_Au = 19.32  # g/cm^3
    e_lo = np.array([edge_used - 1.0])
    e_hi = np.array([edge_used + 1.0])
    mu_lo = float(attenuation('Au', e_lo, density=rho_Au)[0])   # 1/cm
    mu_hi = float(attenuation('Au', e_hi, density=rho_Au)[0])   # 1/cm
    dmu_tab = mu_hi - mu_lo                                     # jump in linear mu (1/cm)
    if not np.isfinite(dmu_tab) or dmu_tab <= 0:
        # Fallback to mass-attenuation-based jump if tabulated linear jump is degenerate.
        mm = attenuation('Au', np.array([edge_used - 1.0, edge_used + 1.0]))
        dmu_tab = float((mm[1] - mm[0]) * rho_Au)

    # --- Convert Delta(mu*t) -> areal thickness t = Delta(mu*t)/Delta(mu_tab) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        thickness_cm = edge_step / dmu_tab   # cm
    # Physical guard: thickness cannot be negative; NaN where non-physical.
    thickness_cm = np.where(np.isfinite(thickness_cm), thickness_cm, np.nan)
    thickness_cm = np.where(thickness_cm < 0, 0.0, thickness_cm)
    thickness_um = thickness_cm * 1e4

    # --- Measurability mask (locate the coupon) ---
    # Per-pixel: require positive edge step AND SNR above threshold.
    snr_thresh = 3.0
    measurable = np.isfinite(edge_step) & (edge_step > 0) & np.isfinite(snr) & (snr > snr_thresh)
    # If the field-level flag is False, the mask still reflects per-pixel SNR (more informative).

    # --- Masked thickness (coupon only) ---
    thickness_coupon = np.where(measurable, thickness_um, np.nan)

    # --- Summary statistics over coupon region ---
    coupon_vals = thickness_um[measurable]
    if coupon_vals.size > 0:
        med_t = float(np.nanmedian(coupon_vals))
        # Robust spread: median absolute deviation scaled to sigma-equivalent.
        mad = float(np.nanmedian(np.abs(coupon_vals - med_t))) * 1.4826
        iqr = float(np.nanpercentile(coupon_vals, 75) - np.nanpercentile(coupon_vals, 25))
        n_pix = int(coupon_vals.size)
    else:
        med_t, mad, iqr, n_pix = np.nan, np.nan, np.nan, 0

    summary = np.array([med_t, mad, iqr, float(n_pix)], dtype=float)

    maps = {
        "Au_Edge_Step": edge_step,               # Delta-OD (absorbance step / edge jump)
        "Au_Thickness": thickness_coupon,        # coupon-masked thickness in microns
        "Au_Thickness_full": thickness_um,       # unmasked thickness (microns)
        "Au_Edge_SNR": snr,                       # per-pixel edge SNR
        "Au_Coupon_Mask": measurable.astype(float),  # boolean coupon location
        "Au_Thickness_Summary": summary,          # [median_um, MAD_um, IQR_um, N_pixels]
    }

    units = {
        "Au_Edge_Step": "optical depth (Delta ln(I0/I), a.u.)",
        "Au_Thickness": "micron",
        "Au_Thickness_full": "micron",
        "Au_Edge_SNR": "a.u.",
        "Au_Coupon_Mask": "bool (1=measurable/coupon)",
        "Au_Thickness_Summary": "[median_um, MAD_um, IQR_um, N_pix]",
    }

    desc = (
        "Transmission T = I/I0 formed by dividing the (lightly energy-smoothed) sample cube by the "
        "same-grid I0 flat-field baseline; absorbance = -ln(T) = mu*t. The Au K-edge jump at 80.7 keV "
        "is quantified per pixel with measure_edge_step using robust pre/post continuum windows above "
        "40 keV (low_e_cut_kev=40) with a flux gate; it reports Delta(mu*t) (edge step) and SNR. The "
        "tabulated linear-attenuation jump Delta(mu_tab) = mu(E+1)-mu(E-1) for Au (rho=19.32 g/cm^3) is "
        "taken from the NIST attenuation helper. Areal thickness t = Delta(mu*t)/Delta(mu_tab), reported "
        "in microns. A per-pixel measurability mask (edge_step>0 and SNR>3) locates the coupon; median "
        "thickness and robust spread (MAD, IQR) are computed over the coupon region."
    )

    return {"maps": maps, "units": units, "description": desc}
