# Auto-generated Script
# Task: Form per-pixel transmission by dividing the primary datacube by the baseline_I0 flat-field operand, then compute ln(I0/I). Quantify the Bi K-edge jump at 90.5 keV with `measure_edge_step` (let it pick the robust pre/post windows and gate on flux; exclude energies below ~40 keV). This produces the edge-step map (Delta_mu_t) which both localizes the Bi coupon and serves as the areal-thickness proxy. Convert the measured edge jump to physical thickness using the tabulated Bi K-edge attenuation discontinuity obtained via the `attenuation` tool for element Bi (thickness = Delta_mu_t / Delta_mu_tab, with Bi density). Output the edge-jump map and the thickness map (in mm or cm). Mark pixels where the edge is not measurable as background/NaN.

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    # --- Shape handling ---
    cube = np.asarray(data, dtype=float)
    if cube.ndim == 2:
        raise ValueError("Expected 3D cube (H,W,E)")
    H, W, E = cube.shape
    energy = np.asarray(axis, dtype=float).ravel()

    # --- I0 flat field ---
    if auxiliary and "baseline_I0" in auxiliary and auxiliary["baseline_I0"] is not None:
        i0 = np.asarray(auxiliary["baseline_I0"], dtype=float)
    else:
        i0 = np.median(cube.reshape(-1, E), axis=0)[None, None, :] * np.ones((H, W, 1))

    # --- Clean: clip negatives/zeros, light spectral smoothing for stability ---
    cube_clean = np.clip(cube, 0.0, None)
    i0_clean = np.clip(i0, 1e-6, None)

    # Mild spectral smoothing (broad K-edge step preserved)
    cube_sm = gaussian_filter(cube_clean, sigma=(0, 0, 1.2))
    i0_sm = gaussian_filter(i0_clean, sigma=(0, 0, 1.2))

    # --- Bi K-edge nominal energy ---
    edge_kev = 90.5

    # --- Robust edge-step measurement via registered tool ---
    # Exclude low-energy scatter/noise-dominated region below ~40 keV.
    r = measure_edge_step(
        cube_sm,
        i0_sm,
        energy,
        edge_kev,
        low_e_cut_kev=40.0,
        auto_center=True,
        search_tol_kev=2.0,
    )

    edge_step = np.asarray(r["edge_step"], dtype=float)   # Delta(mu*t) per pixel
    snr = np.asarray(r["snr"], dtype=float)
    measurable_field = bool(r["measurable"])
    edge_used = float(r.get("edge_kev_used", edge_kev))

    # --- Per-pixel measurability mask (SNR gate) ---
    snr_thresh = 3.0
    pixel_measurable = np.isfinite(snr) & (snr > snr_thresh) & np.isfinite(edge_step)

    # --- Tabulated linear-attenuation jump for Bi across its K-edge ---
    rho_bi = 9.78  # g/cm^3, standard density of bismuth
    e_lo = np.array([edge_used - 1.0])
    e_hi = np.array([edge_used + 1.0])
    mu_lo = float(attenuation('Bi', e_lo, density=rho_bi)[0])   # 1/cm below edge
    mu_hi = float(attenuation('Bi', e_hi, density=rho_bi)[0])   # 1/cm above edge
    delta_mu_tab = mu_hi - mu_lo  # linear-attenuation jump (1/cm)

    if delta_mu_tab <= 0 or not np.isfinite(delta_mu_tab):
        delta_mu_tab = np.nan

    # --- Convert edge jump (dimensionless mu*t) to thickness ---
    edge_jump_map = edge_step.copy()
    thickness_cm = edge_jump_map / delta_mu_tab
    thickness_um = thickness_cm * 1e4  # cm -> um

    # --- Mask off-coupon / non-measurable pixels as NaN ---
    edge_jump_out = np.where(pixel_measurable, edge_jump_map, np.nan)
    thickness_out = np.where(pixel_measurable, thickness_um, np.nan)

    # Guard against non-physical negative values from noise
    thickness_out = np.where(thickness_out < 0, np.nan, thickness_out)
    edge_jump_out = np.where(edge_jump_out < 0, np.nan, edge_jump_out)

    maps = {
        "Edge_Jump": edge_jump_out.astype(float),
        "Thickness": thickness_out.astype(float),
        "Edge_SNR": snr.astype(float),
        "Measurable_Mask": pixel_measurable.astype(float),
    }

    units = {
        "Edge_Jump": "a.u. (Delta mu*t, optical depth)",
        "Thickness": "um",
        "Edge_SNR": "a.u.",
        "Measurable_Mask": "bool",
    }

    description = (
        "Per-pixel transmission I/I0 was formed against the baseline_I0 flat field "
        "and the K-edge step in optical depth ln(I0/I) at the Bi K-edge (~90.5 keV) "
        "was measured per pixel with measure_edge_step (robust pre/post continuum "
        "windows, flux gating, low-energy scatter region below ~40 keV excluded via "
        "low_e_cut_kev). This yields Edge_Jump = Delta(mu*t), which localizes the Bi "
        "coupon and serves as an areal-thickness proxy. Thickness is t = "
        "Delta(mu*t)/Delta(mu_tab), where the tabulated linear-attenuation jump "
        "Delta(mu_tab) for Bi across its K-edge is obtained from the attenuation() "
        "tool at the standard Bi density (9.78 g/cm^3), converted to microns. "
        "Pixels flagged non-measurable (SNR<3 / off-coupon background) are masked as NaN."
    )

    return {"maps": maps, "units": units, "description": description}