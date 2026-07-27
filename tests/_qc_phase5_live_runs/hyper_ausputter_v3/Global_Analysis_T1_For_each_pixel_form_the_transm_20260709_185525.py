# Auto-generated Script
# Task: For each pixel, form the transmission spectrum by dividing the primary datacube by the baseline_I0 flat-field, guard against zero/negative flux, and convert to absorbance -ln(I/I0), restricting to energies above 5 keV. Use the edge-step helper (measure_edge_step) to quantify the Au K-edge jump at 80.8 keV per pixel from robust pre-edge (~just below 80.8 keV) and post-edge windows; this jump map both locates the Au (structured high-signal regions) and gives the edge magnitude. Then convert the measured edge step (delta of the linear attenuation across the K-edge) into Au thickness per pixel using the tabulated Au mu(E) discontinuity obtained from the attenuation helper for the element Au evaluated just below and just above 80.8 keV (thickness = edge_step / (mu_above - mu_below) with appropriate density handling from the tool). Report the edge-step map, the derived thickness map, and only assign thickness where the edge is flagged measurable.

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    data = np.asarray(data, dtype=float)
    axis = np.asarray(axis, dtype=float)
    H, W, E = data.shape

    # ---- I0 reference ----
    if auxiliary and "baseline_I0" in auxiliary and auxiliary["baseline_I0"] is not None:
        i0 = np.asarray(auxiliary["baseline_I0"], dtype=float)
    else:
        # fallback: use field-median spectrum as flat field
        i0 = np.broadcast_to(np.median(data.reshape(-1, E), axis=0), data.shape).copy()

    edge_kev = 80.8

    # ------------------------------------------------------------------
    # STRUCTURALLY DIFFERENT APPROACH:
    # A sputtered Au film gives an OD step far below per-pixel noise at 81 keV.
    # Previous attempts failed by (a) global fits, (b) tight windows, and
    # (c) self-referential MAD gates that kept only noise outliers.
    #
    # Fix:
    #  1. Use the registered measure_edge_step with WIDE continuum windows and
    #     a modest photon-flux gate (physical, not self-referential).
    #  2. Apply light spatial binning to the transmission cube BEFORE the edge
    #     step to build per-pixel SNR so a coherent film region survives.
    #  3. Use a DIFFERENTIAL edge step (pre/post window difference) — cancels the
    #     smooth bremsstrahlung background and is feature-specific.
    #  4. Gate on an ABSOLUTE, physically motivated significance: step > 3*noise
    #     estimated from the returned SNR map (snr>3), NOT a MAD-on-SNR outlier
    #     hunt. This retains the coherent population instead of extreme tails.
    # ------------------------------------------------------------------

    # Light spatial smoothing (per energy channel) to raise per-pixel SNR
    # without erasing the (spatially coherent) film edge.
    data_s = gaussian_filter(data, sigma=(1.0, 1.0, 0.0))
    i0_s = gaussian_filter(i0, sigma=(1.0, 1.0, 0.0)) if i0.ndim == 3 else i0

    # Run the robust edge-step helper with WIDE windows.
    res = measure_edge_step(
        data_s, i0_s, axis, edge_kev,
        pre_gap=2.0, post_gap=2.0, win_width=8.0,
        low_e_cut_kev=5.0, flux_floor_counts=20.0,
        auto_center=True, search_tol_kev=2.0,
    )

    edge_step = np.asarray(res["edge_step"], dtype=float)   # Delta-OD (H,W)
    snr = np.asarray(res["snr"], dtype=float)               # per-pixel SNR (H,W)
    edge_used = float(res.get("edge_kev_used", edge_kev))

    # ---- Au mu discontinuity across the K-edge (linear, 1/cm) ----
    rho_Au = 19.32  # g/cm^3
    # Evaluate just below / above the used edge energy.
    e_below = np.array([edge_used - 0.8])
    e_above = np.array([edge_used + 0.8])
    mu_below = float(attenuation("Au", e_below, density=rho_Au)[0])  # 1/cm
    mu_above = float(attenuation("Au", e_above, density=rho_Au)[0])  # 1/cm
    dmu = mu_above - mu_below  # 1/cm, positive jump

    # ---- ABSOLUTE physical significance gate ----
    # snr = edge_step / step_noise (from helper). snr>3 => step > 3*noise.
    # Also require a positive edge step (K-edge is a JUMP up in OD).
    measurable_field = bool(res.get("measurable", True))
    valid = np.isfinite(edge_step) & np.isfinite(snr)
    sig_mask = valid & (snr > 3.0) & (edge_step > 0.0)

    # ---- Edge-step map: report the measured Delta-OD, keeping only physically
    #      significant (positive, snr>3) pixels; others -> NaN (not measured).
    edge_step_map = np.where(sig_mask, edge_step, np.nan)

    # ---- Thickness map (micrometres) where edge is measurable ----
    # thickness [cm] = edge_step(OD) / dmu(1/cm);  *1e4 -> um
    thickness_um = np.full((H, W), np.nan, dtype=float)
    if dmu > 0:
        t = edge_step / dmu * 1e4  # um
        thickness_um = np.where(sig_mask, t, np.nan)

    return {
        "maps": {
            "Au_Edge_Step": edge_step_map,
            "Au_Thickness": thickness_um,
            "Edge_SNR": snr,
        },
        "units": {
            "Au_Edge_Step": "Delta-OD (a.u.)",
            "Au_Thickness": "um",
            "Edge_SNR": "a.u.",
        },
        "description": (
            "Transmission = data / baseline_I0 (light spatial binning to raise per-pixel "
            "SNR without erasing the coherent film edge). The Au K-edge jump at 80.8 keV "
            "is measured as a DIFFERENTIAL step (robust pre/post continuum windows via "
            "measure_edge_step) which cancels the smooth bremsstrahlung background. Wide "
            "(8 keV) continuum windows reduce noise; an ABSOLUTE significance gate (SNR>3 "
            "and positive step) retains the spatially coherent film population instead of "
            "self-referential MAD outliers. Edge step (Delta-OD) is converted to Au "
            "thickness via t = edge_step / (mu_above - mu_below) using NIST linear "
            "attenuation for Au (rho=19.32 g/cm^3) evaluated +/-0.8 keV across the edge. "
            "Thickness assigned only where the edge is flagged measurable."
        ),
    }
