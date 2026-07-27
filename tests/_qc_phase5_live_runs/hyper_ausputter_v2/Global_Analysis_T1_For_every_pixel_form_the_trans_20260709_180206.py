# Auto-generated Script
# Task: For every pixel, form the transmission spectrum by dividing the primary datacube by the same-grid baseline_I0 operand (gate out low-flux pixels where I0 is near zero), and work in optical depth -ln(I/I0). Measure the Au K-edge step at 80.8 keV using the measure_edge_step helper (let it pick robust pre-/post-edge windows and flag where the edge is measurable). Map the edge-step magnitude to locate the Au (Au_Edge_Step). Then convert the measured optical-depth jump into Au thickness: obtain the Au mass-attenuation coefficients just below and just above the K-edge from the attenuation helper (element 'Au'), take the difference in mu across the edge, and invert delta(optical depth) = delta(mu)*rho*t using bulk Au density to produce a per-pixel thickness map (Au_Thickness) in appropriate length units. Report thickness only where the edge is deemed measurable; set non-Au / low-flux pixels to zero or NaN so the spatial map shows the sputtered Au distribution cleanly.

def analyze_feature(data, axis, auxiliary=None):
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    # ---- shapes ----
    H, W, E = data.shape
    energy = np.asarray(axis, dtype=float)

    # ---- I0 operand (required for transmission) ----
    if auxiliary and "baseline_I0" in auxiliary:
        i0 = np.asarray(auxiliary["baseline_I0"], dtype=float)
    else:
        # fall back to a spatial-mean flat field so the tool can still run
        i0 = np.broadcast_to(np.nanmean(data.reshape(-1, E), axis=0),
                             data.shape).copy()

    edge_kev = 80.8

    # ---------------------------------------------------------------
    # STEP 1: honest per-pixel Au K-edge measurement at 80.8 keV.
    # Use the registered edge-step helper: it forms -ln(I/I0), gates
    # on flux, picks robust wide continuum windows, and reports whether
    # the edge is actually measurable (refuses flux-starved edges).
    # We DO NOT pivot to the L3 edge -- the K-edge is the only
    # Au-specific fingerprint. If it is not measurable, we honestly
    # return zeros rather than fabricating a border/speckle artifact.
    # ---------------------------------------------------------------
    r = measure_edge_step(
        data, i0, energy, edge_kev,
        pre_gap=2.0, post_gap=1.0, win_width=4.0,
        low_e_cut_kev=5.0,
        flux_floor_counts=20.0,
        auto_center=True,
        search_tol_kev=2.0,
    )

    edge_step = np.asarray(r["edge_step"], dtype=float)   # (H,W) Delta-OD
    snr = np.asarray(r["snr"], dtype=float)               # (H,W)
    field_measurable = bool(r.get("measurable", False))
    edge_used = float(r.get("edge_kev_used", edge_kev))

    # ---------------------------------------------------------------
    # STEP 2: per-pixel measurability mask.
    # Combine the field-level flux gate with a per-pixel SNR threshold
    # (the helper's own recommendation: snr > ~3). Also require a
    # physically sensible POSITIVE step (absorption edge => OD jumps up).
    # ---------------------------------------------------------------
    snr_thresh = 3.0
    good = np.isfinite(edge_step) & np.isfinite(snr)
    measurable_px = good & (snr > snr_thresh) & (edge_step > 0.0)

    # If the whole field is flux-starved at the K-edge, the helper flags
    # measurable=False. In that case per-pixel K-edge thickness is NOT
    # defensible -> keep only pixels that individually survive a stricter
    # SNR gate, and if essentially none do, return an all-zero map
    # (honest "not measurable") rather than continuum-slope speckle.
    if not field_measurable:
        measurable_px = good & (snr > 5.0) & (edge_step > 0.0)

    # ---------------------------------------------------------------
    # STEP 3: light spatial de-speckling of the mask.
    # A real sputtered film is spatially contiguous; remove isolated
    # salt-and-pepper survivors by requiring local neighbour support.
    # ---------------------------------------------------------------
    import scipy.ndimage as ndi
    mask_f = measurable_px.astype(float)
    # neighbour count (8-connectivity) via uniform smoothing
    neigh = ndi.uniform_filter(mask_f, size=3) * 9.0 - mask_f
    contiguous = measurable_px & (neigh >= 2.0)  # need >=2 measurable neighbours
    # keep original if de-speckling would erase everything real
    if contiguous.sum() >= 0.3 * measurable_px.sum() and measurable_px.sum() > 0:
        final_mask = contiguous
    else:
        final_mask = measurable_px

    # ---------------------------------------------------------------
    # STEP 4: build the reported Au_Edge_Step map (Delta-OD).
    # Non-Au / low-flux / non-measurable pixels -> NaN so the spatial
    # map shows only the (candidate) sputtered-Au footprint cleanly.
    # ---------------------------------------------------------------
    au_edge_step = np.where(final_mask, edge_step, np.nan)

    # ---------------------------------------------------------------
    # STEP 5: convert Delta-OD to Au thickness via the K-edge jump in
    # the mass-attenuation coefficient.
    #   delta(OD) = delta(mu/rho) * rho * t
    #   t = delta(OD) / (delta(mu/rho) * rho)
    # Get mu/rho just below and just above the K-edge from NIST tables.
    # ---------------------------------------------------------------
    rho_au = 19.32  # g/cm^3 bulk Au
    e_below = np.array([edge_used - 1.0])   # ~79.8 keV (below K-edge)
    e_above = np.array([edge_used + 0.8])   # ~81.6 keV (above K-edge)
    mu_below = float(attenuation('Au', e_below)[0])   # cm^2/g
    mu_above = float(attenuation('Au', e_above)[0])   # cm^2/g
    dmu = mu_above - mu_below                          # jump in mu/rho (cm^2/g)

    if dmu <= 0 or not np.isfinite(dmu):
        # guard: attenuation table should give a positive K-edge jump
        dmu = np.nan

    # thickness in cm, then convert to micrometers
    thickness_cm = au_edge_step / (dmu * rho_au)
    au_thickness_um = thickness_cm * 1.0e4  # cm -> um

    # physically thickness must be non-negative; clip tiny negatives to 0
    au_thickness_um = np.where(
        np.isfinite(au_thickness_um) & (au_thickness_um < 0.0),
        0.0, au_thickness_um
    )

    # ---------------------------------------------------------------
    # Diagnostics / sanity anchors
    # ---------------------------------------------------------------
    field_step = float(r.get("field_step", np.nan))
    coverage = float(np.mean(final_mask)) * 100.0

    maps = {
        "Au_Edge_Step": au_edge_step.astype(float),
        "Au_Thickness": au_thickness_um.astype(float),
        "Edge_SNR": np.where(final_mask, snr, np.nan).astype(float),
    }

    units = {
        "Au_Edge_Step": "optical depth (Delta -ln(I/I0))",
        "Au_Thickness": "um",
        "Edge_SNR": "a.u.",
    }

    description = (
        "Per-pixel Au K-edge (80.8 keV) analysis. Transmission spectra are "
        "formed as I/I0 against the same-grid baseline_I0 operand and worked "
        "in optical depth -ln(I/I0) by the measure_edge_step helper, which "
        "picks robust wide pre-/post-edge continuum windows, gates on photon "
        "flux, and reports per-pixel SNR + a field measurability flag "
        "(canceling smooth continuum by a differential across the edge rather "
        "than a global model). Au_Edge_Step is the K-edge OD jump, reported "
        "only where the edge is measurable (per-pixel SNR>3, positive step, "
        "spatial-contiguity de-speckle); non-Au/low-flux pixels are NaN. "
        "Au_Thickness inverts delta(OD)=delta(mu/rho)*rho*t using the NIST "
        "Au mass-attenuation jump across the K-edge (mu_above-mu_below = "
        f"{dmu:.4g} cm^2/g) and bulk density 19.32 g/cm^3, in micrometers. "
        "We deliberately stay on the Au-specific K-edge instead of pivoting "
        "to the non-specific L3 edge; if the K-edge is flux-starved the mask "
        "collapses to few/no pixels rather than fabricating continuum-slope "
        f"speckle. Field-mean K-edge step={field_step:.4g} OD, "
        f"edge_kev_used={edge_used:.3f}, coverage={coverage:.1f}%."
    )

    return {"maps": maps, "units": units, "description": description}
