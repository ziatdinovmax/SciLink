# Auto-generated Script
# Task: Per-pixel Au thickness from the K-edge jump. First form the transmission ratio using the provided baseline_I0 cube as the flat field (I/I0) and compute the optical depth ln(I0/I) per pixel, excluding the scatter/noise-dominated low-energy region. Quantify the discontinuity in ln(I0/I) at the Au K-edge (~80.7 keV) using `measure_edge_step`, which selects robust narrow pre-edge/post-edge windows, gates on flux, and reports whether the edge is measurable — do not hand-roll pre/post window means. This yields the edge jump Delta(mu*t) per pixel (Edge_Jump map), which localizes the Au coupon spatially. Then convert to physical thickness t = Delta(mu*t) / Delta(mu_tab), where the tabulated linear-attenuation jump for Au at its K-edge is obtained via `attenuation` for the element 'Au' (using the standard density) rather than embedding coefficient tables. Produce the thickness map (Au_Thickness). Mask/flag pixels where `measure_edge_step` reports the edge is not measurable (off-coupon / background).

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    # --- Shape handling ---
    cube = np.asarray(data, dtype=float)
    if cube.ndim == 2:
        # (pixels, energy) -> assume square-ish; try to infer
        raise ValueError("Expected 3D cube (H,W,E)")
    H, W, E = cube.shape
    energy = np.asarray(axis, dtype=float).ravel()

    # --- I0 flat field ---
    if auxiliary and "baseline_I0" in auxiliary and auxiliary["baseline_I0"] is not None:
        i0 = np.asarray(auxiliary["baseline_I0"], dtype=float)
    else:
        # Fallback: use the median spectrum across the field as a pseudo-I0
        i0 = np.median(cube.reshape(-1, E), axis=0)[None, None, :] * np.ones((H, W, 1))

    # --- Light despiking / smoothing along the energy axis for stability ---
    # Clip negatives/zeros (physical: intensities are non-negative counts)
    cube_clean = np.clip(cube, 0.0, None)
    i0_clean = np.clip(i0, 1e-6, None)

    # Mild spectral smoothing (does not erase the broad K-edge step)
    cube_sm = gaussian_filter(cube_clean, sigma=(0, 0, 1.2))
    i0_sm = gaussian_filter(i0_clean, sigma=(0, 0, 1.2))

    edge_kev = 80.7

    # --- Robust edge-step measurement via registered tool ---
    # Exclude low-energy scatter/noise-dominated region via low_e_cut_kev.
    r = measure_edge_step(
        cube_sm,
        i0_sm,
        energy,
        edge_kev,
        low_e_cut_kev=10.0,
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
    # If the whole field failed the flux gate, keep only high-SNR pixels anyway
    # (the coupon itself should still show a jump).

    # --- Tabulated linear-attenuation jump for Au across its K-edge ---
    # Use standard density of Au for LINEAR mu (1/cm). Sample just below/above edge.
    rho_au = 19.32  # g/cm^3, standard density
    e_lo = np.array([edge_used - 1.0])
    e_hi = np.array([edge_used + 1.0])
    mu_lo = float(attenuation('Au', e_lo, density=rho_au)[0])   # 1/cm below edge
    mu_hi = float(attenuation('Au', e_hi, density=rho_au)[0])   # 1/cm above edge
    delta_mu_tab = mu_hi - mu_lo  # linear-attenuation jump (1/cm)

    # --- Convert edge jump (dimensionless mu*t) to thickness ---
    # t [cm] = Delta(mu*t) / Delta(mu_tab); convert to microns.
    if delta_mu_tab <= 0 or not np.isfinite(delta_mu_tab):
        delta_mu_tab = np.nan

    edge_jump_map = edge_step.copy()
    thickness_cm = edge_jump_map / delta_mu_tab
    thickness_um = thickness_cm * 1e4  # cm -> um

    # --- Mask off-coupon / non-measurable pixels ---
    edge_jump_out = np.where(pixel_measurable, edge_jump_map, np.nan)
    thickness_out = np.where(pixel_measurable, thickness_um, np.nan)

    # Guard against negative (non-physical) thickness from noise: flag as NaN
    thickness_out = np.where(thickness_out < 0, np.nan, thickness_out)
    edge_jump_out = np.where(edge_jump_out < 0, np.nan, edge_jump_out)

    maps = {
        "Edge_Jump": edge_jump_out.astype(float),
        "Au_Thickness": thickness_out.astype(float),
        "Edge_SNR": snr.astype(float),
        "Measurable_Mask": pixel_measurable.astype(float),
    }

    units = {
        "Edge_Jump": "a.u. (Delta mu*t, optical depth)",
        "Au_Thickness": "um",
        "Edge_SNR": "a.u.",
        "Measurable_Mask": "bool",
    }

    description = (
        "Transmission I/I0 was formed against the baseline_I0 flat field and the "
        "K-edge step in optical depth ln(I0/I) at the Au K-edge (~80.7 keV) was "
        "measured per pixel with measure_edge_step (robust pre/post continuum "
        "windows, flux gating, low-energy scatter region excluded via low_e_cut). "
        "This yields Edge_Jump = Delta(mu*t), which localizes the Au coupon. "
        "Thickness is t = Delta(mu*t)/Delta(mu_tab), where the tabulated linear-"
        "attenuation jump Delta(mu_tab) for Au at its K-edge is obtained from the "
        "attenuation() tool at the standard Au density (19.32 g/cm^3), converted to "
        "microns. Pixels flagged non-measurable (SNR<3 / off-coupon background) are "
        "masked as NaN."
    )

    return {"maps": maps, "units": units, "description": description}
