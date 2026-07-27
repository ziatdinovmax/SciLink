# Auto-generated Script
# Task: For each pixel, divide the sample cube by the provided I0 flat-field baseline and compute absorbance mu*t(E) = ln(I0/I), excluding energies below ~40 keV where bremsstrahlung scatter/noise dominates. Use the edge-step helper to measure the Au K-edge jump (discontinuity in mu*t) at the Au K-edge (~80.725 keV), taking the difference between narrow post-edge and pre-edge windows; let the helper choose robust windows and flux-gate pixels where the edge is not measurable. Convert the per-pixel edge jump Delta(mu*t) to areal/physical thickness by dividing by the tabulated Au K-edge jump in the linear attenuation coefficient (obtain mu/rho for Au across the edge from the attenuation helper and use rho = 19.3 g/cm^3): t = Delta(mu*t) / Delta_mu_tab. Produce (1) a spatial map of the Au K-edge jump magnitude to locate the coupon and (2) a per-pixel Au thickness map in mm (or cm). Report the mean/median thickness over pixels where the edge is confidently measurable.

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    data = np.asarray(data, dtype=float)
    axis = np.asarray(axis, dtype=float)
    H, W, E = data.shape

    # --- Get I0 flat-field baseline from auxiliary ---
    i0 = None
    if auxiliary and "I0 flat-field baseline cube (no sample) on the same grid" in auxiliary:
        i0 = np.asarray(auxiliary["I0 flat-field baseline cube (no sample) on the same grid"], dtype=float)

    edge_kev = 80.725

    # --- Fallback I0 if none provided: use high-flux field spectrum estimate ---
    if i0 is None:
        # crude estimate: use per-energy max across pixels as I0 (no-sample proxy)
        i0 = np.broadcast_to(np.max(data.reshape(-1, E), axis=0), (H, W, E)).copy()

    # --- Light denoising along energy to stabilize edge step (do not erase edge) ---
    # smooth only mildly; measure_edge_step does linear window fits internally
    data_s = gaussian_filter(data, sigma=(0, 0, 1.0))
    i0_s = gaussian_filter(i0, sigma=(0, 0, 1.0)) if i0.ndim == 3 else i0

    # --- Measure edge step (Delta mu*t) per pixel using robust helper ---
    r = measure_edge_step(
        data_s, i0_s, axis, edge_kev,
        pre_gap=2.0, post_gap=1.5, win_width=4.0,
        low_e_cut_kev=40.0,   # exclude bremsstrahlung-dominated low-E region
        flux_floor_counts=20.0,
        auto_center=True, search_tol_kev=2.0
    )

    edge_step = np.asarray(r["edge_step"], dtype=float)   # (H,W) Delta-OD
    snr = np.asarray(r["snr"], dtype=float)
    edge_used = r.get("edge_kev_used", edge_kev)

    # --- Tabulated Au K-edge jump in LINEAR attenuation coefficient ---
    rho_Au = 19.3  # g/cm^3
    # sample mu just below and just above the edge (linear mu, 1/cm)
    e_below = np.array([edge_used - 1.0])
    e_above = np.array([edge_used + 1.5])
    try:
        mu_below = float(attenuation('Au', e_below, density=rho_Au)[0])
        mu_above = float(attenuation('Au', e_above, density=rho_Au)[0])
        delta_mu_lin = mu_above - mu_below   # 1/cm
    except Exception:
        # fallback: mass attenuation * density
        mu_below = float(attenuation('Au', e_below)[0]) * rho_Au
        mu_above = float(attenuation('Au', e_above)[0]) * rho_Au
        delta_mu_lin = mu_above - mu_below

    if not np.isfinite(delta_mu_lin) or delta_mu_lin <= 0:
        delta_mu_lin = np.nan

    # --- Convert edge jump to thickness ---
    # edge_step = Delta(mu*t) [dimensionless OD], delta_mu_lin [1/cm]
    # t [cm] = edge_step / delta_mu_lin ; convert to mm
    thickness_cm = edge_step / delta_mu_lin
    thickness_mm = thickness_cm * 10.0

    # --- Confidence gating: SNR and non-negative jump ---
    confident = np.isfinite(snr) & (snr > 3.0) & np.isfinite(edge_step) & (edge_step > 0)

    # Clean thickness map: set non-confident/negative to NaN for reporting,
    # but keep the raw jump map for coupon localization.
    thickness_mm_clean = np.where(confident, thickness_mm, np.nan)
    # clip physically implausible negatives in jump map to 0 for display
    edge_jump_map = np.where(np.isfinite(edge_step), edge_step, np.nan)

    # --- Robust statistics over confident pixels ---
    valid_t = thickness_mm_clean[np.isfinite(thickness_mm_clean)]
    if valid_t.size > 0:
        mean_t = float(np.mean(valid_t))
        median_t = float(np.median(valid_t))
    else:
        mean_t = np.nan
        median_t = np.nan

    return {
        "maps": {
            "Au_Edge_Jump": edge_jump_map,        # Delta(mu*t) optical-depth jump
            "Au_Thickness": thickness_mm_clean,   # mm, confident pixels
            "Edge_SNR": snr,
            "Confident_Mask": confident.astype(float),
        },
        "units": {
            "Au_Edge_Jump": "a.u. (Delta optical depth)",
            "Au_Thickness": "mm",
            "Edge_SNR": "a.u.",
            "Confident_Mask": "bool",
        },
        "description": (
            "Absorbance mu*t = ln(I0/I) computed per pixel (low-E<40 keV excluded). "
            "measure_edge_step measures the Au K-edge jump Delta(mu*t) at ~%.3f keV "
            "using robust post/pre continuum windows with a flux gate. Thickness = "
            "Delta(mu*t)/Delta_mu_lin, where Delta_mu_lin (=%.3g 1/cm) is the tabulated "
            "Au K-edge jump in the LINEAR attenuation coefficient from NIST mu/rho * "
            "rho=19.3 g/cm^3. Confident pixels: SNR>3 and positive jump. "
            "Mean thickness=%.4f mm, median=%.4f mm."
            % (float(edge_used), delta_mu_lin, mean_t, median_t)
        ),
    }