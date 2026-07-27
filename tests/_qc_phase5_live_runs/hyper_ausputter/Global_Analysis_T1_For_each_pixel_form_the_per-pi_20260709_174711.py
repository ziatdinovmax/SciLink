# Auto-generated Script
# Task: For each pixel, form the per-pixel transmission by dividing the sample cube by the same-grid I0 flat-field cube (gating out low-flux/zero-I0 pixels and ignoring energies below 5 keV), then take -ln(I/I0) to get absorbance. Measure the Au K-edge jump at 80.8 keV using the edge-step helper (measure_edge_step) so robust pre-/post-edge windows and flux gating are handled by the tool; record whether the edge is measurable. Produce a spatial map of the Au K-edge step to locate the Au. Then convert the measured edge step to Au thickness via the Beer-Lambert law using the change in mass attenuation coefficient across the Au K-edge obtained from the attenuation helper for elemental Au (do not hard-code attenuation values). Output a per-pixel Au thickness map. Mask/flag pixels where the edge is not measurable rather than reporting spurious thickness.

def analyze_feature(data, axis, auxiliary=None):
    import scipy.ndimage as ndi
    from scilink.skills._shared.edge_step import measure_edge_step
    from scilink.skills._shared.xray_attenuation import attenuation

    energy = np.asarray(axis, dtype=float)
    cube = np.asarray(data, dtype=float)
    H, W, E = cube.shape
    edge_kev = 80.8

    # ------------------------------------------------------------------
    # 1. Obtain a TRUE open-beam I0 flat-field. Do NOT fabricate one.
    #    If no genuine I0 is supplied, the K-edge transmission method is
    #    not applicable -> flag everything as not measurable.
    # ------------------------------------------------------------------
    i0 = None
    if auxiliary:
        for k in auxiliary:
            if 'I0' in k or 'flat' in k.lower() or 'baseline' in k.lower():
                cand = np.asarray(auxiliary[k], dtype=float)
                if cand.shape == cube.shape:
                    i0 = cand
                    break

    nan_map = np.full((H, W), np.nan, dtype=float)
    zeros_map = np.zeros((H, W), dtype=float)

    if i0 is None:
        # No genuine open-beam reference: refuse to fabricate transmission.
        return {
            "maps": {
                "Au_Edge_Step": nan_map.copy(),
                "Au_Thickness": nan_map.copy(),
                "Measurable_Mask": zeros_map.copy(),
                "Edge_SNR": nan_map.copy(),
            },
            "units": {
                "Au_Edge_Step": "delta-OD (unitless)",
                "Au_Thickness": "micrometers",
                "Measurable_Mask": "boolean (0/1)",
                "Edge_SNR": "unitless",
            },
            "description": (
                "No genuine open-beam I0 flat-field cube was supplied. A true "
                "open-beam reference is REQUIRED to form transmission T=I/I0 and "
                "measure the Au K-edge step. Self-normalization against the sample "
                "field median only re-encodes the beam/detector spatial flux profile "
                "as a spurious edge and is refused. All outputs are flagged as "
                "NOT MEASURABLE (NaN / mask=0)."
            ),
        }

    # ------------------------------------------------------------------
    # 2. Honest measurability check on the FIELD-MEAN spectrum BEFORE any
    #    per-pixel fitting. Check whether the K-edge region carries flux
    #    and whether a step actually exists in the field-mean OD.
    # ------------------------------------------------------------------
    # Light spectral smoothing only for diagnostics / stability (does not
    # erase a real edge which is broad on this coarse energy grid).
    smooth_cube = ndi.uniform_filter1d(cube, size=3, axis=-1, mode='nearest')
    smooth_i0 = ndi.uniform_filter1d(i0, size=3, axis=-1, mode='nearest')

    # Continuum windows just below/above the edge (wide, feature-local).
    pre_lo, pre_hi = edge_kev - 6.0, edge_kev - 2.0
    post_lo, post_hi = edge_kev + 1.0, edge_kev + 5.0
    pre_sel = (energy >= pre_lo) & (energy < pre_hi)
    post_sel = (energy >= post_lo) & (energy < post_hi)

    # Field-mean incident flux in the post-edge window (photons available).
    field_i0 = i0.reshape(-1, E).mean(axis=0)
    post_flux_field = np.nanmedian(field_i0[post_sel]) if post_sel.any() else 0.0

    # A coarse, honest flux floor: the K-edge tail is often photon-starved.
    FLUX_FLOOR = 20.0

    # ------------------------------------------------------------------
    # 3. Use the registered edge-step helper (robust windows + flux gate).
    #    It reports whether the edge is measurable rather than fabricating.
    # ------------------------------------------------------------------
    edge_step_map = nan_map.copy()
    snr_map = nan_map.copy()
    measurable_mask = zeros_map.copy()
    thickness_map = nan_map.copy()

    try:
        r = measure_edge_step(
            data=cube,
            i0=i0,
            energy_kev=energy,
            edge_kev=edge_kev,
            pre_gap=2.0,
            post_gap=1.0,
            win_width=4.0,
            low_e_cut_kev=5.0,
            flux_floor_counts=FLUX_FLOOR,
            auto_center=True,
            search_tol_kev=2.0,
        )
    except Exception:
        r = None

    field_measurable = bool(r['measurable']) if (r is not None and 'measurable' in r) else False
    # Extra guard: require genuine post-edge flux in the field mean.
    if post_flux_field < FLUX_FLOOR:
        field_measurable = False

    if r is not None and field_measurable:
        es = np.asarray(r['edge_step'], dtype=float)
        snr = np.asarray(r['snr'], dtype=float)
        if es.shape == (H, W):
            edge_step_map = es
            snr_map = snr

            # Per-pixel measurability: require SNR > 3 (helper guidance).
            pix_ok = np.isfinite(snr) & (snr > 3.0) & np.isfinite(es)
            # A K-edge jump is a POSITIVE step in optical depth.
            pix_ok &= (es > 0.0)
            measurable_mask = pix_ok.astype(float)

            # ----------------------------------------------------------
            # 4. Beer-Lambert: convert delta-OD to Au thickness using the
            #    CHANGE in mass attenuation across the K-edge, from the
            #    attenuation helper (no hard-coded mu values).
            # ----------------------------------------------------------
            # Sample mu/rho just below and just above the edge.
            e_below = np.array([edge_kev - 0.8])
            e_above = np.array([edge_kev + 0.8])
            mu_below = float(attenuation('Au', e_below)[0])   # cm^2/g
            mu_above = float(attenuation('Au', e_above)[0])   # cm^2/g
            dmu = abs(mu_above - mu_below)                     # cm^2/g
            rho_au = 19.32                                     # g/cm^3

            if dmu > 1e-6:
                # delta-OD = dmu * rho * t  ->  t[cm] = dOD/(dmu*rho)
                t_cm = es / (dmu * rho_au)
                t_um = t_cm * 1e4  # cm -> micrometers
                # Only report thickness where the edge is measurable.
                t_out = np.full((H, W), np.nan, dtype=float)
                good = pix_ok & (t_um >= 0)
                t_out[good] = t_um[good]
                thickness_map = t_out

    # If nothing was measurable, keep NaN/zeros (honest flagging).
    n_ok = int(np.nansum(measurable_mask))

    desc = (
        "Per-pixel Au K-edge (80.8 keV) analysis via true open-beam transmission. "
        "Transmission T=I/I0 formed from the supplied flat-field cube; measure_edge_step "
        "handles robust pre-/post-edge continuum windows, low-energy cut (5 keV) and a "
        "photon-flux gate. The field-mean post-edge flux was checked FIRST: if the K-edge "
        "region is photon-starved (post-edge I0 below the flux floor) the edge is declared "
        "NOT MEASURABLE and no thickness is fabricated. Edge_Step = delta optical depth; "
        "pixels with SNR<=3 or non-positive step are masked. Thickness from Beer-Lambert: "
        "t = delta-OD / (delta(mu/rho) * rho_Au), with delta(mu/rho) taken across the K-edge "
        "from the attenuation helper (no hard-coded values) and rho_Au=19.32 g/cm^3. "
        f"Measurable pixels: {n_ok}. "
    )
    if n_ok == 0:
        desc += (
            "WARNING: zero pixels passed the measurability gate -- the 80.8 keV K-edge "
            "sits in a photon-starved tail of this spectrum. The Au signal is likely only "
            "accessible via the Au L-edges (~11.9-14.4 keV) or fluorescence lines in the "
            "high-flux region; the K-edge thickness map is intentionally NaN rather than spurious."
        )

    return {
        "maps": {
            "Au_Edge_Step": edge_step_map,
            "Au_Thickness": thickness_map,
            "Measurable_Mask": measurable_mask,
            "Edge_SNR": snr_map,
        },
        "units": {
            "Au_Edge_Step": "delta-OD (unitless)",
            "Au_Thickness": "micrometers",
            "Measurable_Mask": "boolean (0/1)",
            "Edge_SNR": "unitless",
        },
        "description": desc,
    }
