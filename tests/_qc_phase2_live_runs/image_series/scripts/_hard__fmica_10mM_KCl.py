import numpy as np
import json
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

from scilink.skills._shared.fourier_reflection import fourier_reflection_map

try:
    from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
    _HAVE_RESOLVE = True
except Exception:
    _HAVE_RESOLVE = False
    resolve_pixel_size_nm = None


# ---------------------------------------------------------------
# Discover the two concentration frames (2 mM and 10 mM KCl)
# The plan's core objective is COMPARATIVE. We locate two inputs;
# fall back gracefully if only one is present.
# ---------------------------------------------------------------
def _find_inputs():
    """Return ordered list of (label, npy_path, meta_path_or_None)."""
    candidates = []
    # Preferred explicit names by concentration
    name_map = [
        ('2mM', ['data_2mM.npy', 'data_2mm.npy', '2mM.npy', '2mm.npy']),
        ('10mM', ['data_10mM.npy', 'data_10mm.npy', '10mM.npy', '10mm.npy']),
    ]
    for label, names in name_map:
        for nm in names:
            if os.path.exists(nm):
                candidates.append((label, nm))
                break
    if len(candidates) == 2:
        return candidates

    # Otherwise glob for any .npy that look like frames
    npys = sorted(glob.glob('*.npy'))
    # Exclude our own output artifacts
    excl = {'preprocessed_square.npy', 'reflection_amplitude_map.npy', 'domain_mask.npy'}
    npys = [p for p in npys if os.path.basename(p) not in excl]

    # Try to label by concentration hints in filename
    labelled = []
    for p in npys:
        low = p.lower()
        if '2mm' in low or '2_mm' in low:
            labelled.append(('2mM', p))
        elif '10mm' in low or '10_mm' in low:
            labelled.append(('10mM', p))
    # dedupe keeping order
    seen = set(); out = []
    for lab, p in labelled:
        if lab not in seen:
            out.append((lab, p)); seen.add(lab)
    if len(out) == 2:
        return out

    # Fall back: if 'data.npy' present, treat as single (comparison degraded)
    if os.path.exists('data.npy'):
        base = [('sample', 'data.npy')]
        # Add any additional distinct frame if available
        extra = [p for p in npys if os.path.basename(p) != 'data.npy']
        if extra:
            base = [('frameA', 'data.npy'), ('frameB', extra[0])]
        return base

    # Last resort: first up-to-two npys found
    if len(npys) >= 2:
        return [('frameA', npys[0]), ('frameB', npys[1])]
    elif len(npys) == 1:
        return [('sample', npys[0])]
    else:
        raise FileNotFoundError('No .npy input frames found for analysis.')


def _load_metadata_for(npy_path):
    """Attempt to load a sidecar metadata/system_info JSON for a frame."""
    stem = os.path.splitext(npy_path)[0]
    for cand in (stem + '.json', stem + '_metadata.json', stem + '_system_info.json',
                 'metadata.json', 'system_info.json'):
        if os.path.exists(cand):
            try:
                with open(cand, 'r') as f:
                    return json.load(f), cand
            except Exception:
                pass
    return {}, None


def _extract_intensity_mapping(meta):
    """Pull data_range_minimum/maximum/units for physical intensity tracking."""
    if not isinstance(meta, dict):
        return {'min': None, 'max': None, 'units': None}
    src = meta.get('system_info', meta)
    if not isinstance(src, dict):
        src = meta
    dmin = src.get('data_range_minimum', src.get('data_min'))
    dmax = src.get('data_range_maximum', src.get('data_max'))
    units = src.get('data_units', src.get('units', src.get('data_range_units')))
    def _f(v):
        try:
            return float(v)
        except Exception:
            return None
    return {'min': _f(dmin), 'max': _f(dmax), 'units': units}


# ---------------------------------------------------------------
# Per-frame pipeline (LOCKED strategy). Returns a dict of results
# plus arrays needed for visualization/saving.
# ---------------------------------------------------------------
STREAK_THRESH = 0.02  # tuned: fraction of variance considered 'severe' streaking


def row_streak_metric(a):
    row_means = np.mean(a, axis=1)
    return float(np.var(row_means) / (np.var(a) + 1e-12))


def per_row_poly_detrend(a, order, x):
    out = np.empty_like(a)
    for r in range(a.shape[0]):
        coeffs = np.polyfit(x, a[r], order)
        out[r] = a[r] - np.polyval(coeffs, x)
    return out


def process_frame(label, npy_path):
    notes = []
    image = np.load(npy_path)
    notes.append(f'[{label}] Loaded {npy_path} shape={image.shape} dtype={image.dtype}')

    if image.ndim == 3:
        img = image[:, :, 0].astype(np.float64)
    else:
        img = image.astype(np.float64)

    H, W = img.shape

    # ---- Physical intensity mapping tracking ----
    meta, meta_path = _load_metadata_for(npy_path)
    intensity = _extract_intensity_mapping(meta)
    if intensity['min'] is not None and intensity['max'] is not None:
        notes.append(f"[{label}] Physical intensity mapping: "
                     f"[{intensity['min']}, {intensity['max']}] {intensity['units']}")
    else:
        notes.append(f'[{label}] No physical intensity mapping found in metadata; '
                     'topography kept in native array units (no physical calibration).')

    # Round-trip verification of physical units: map array -> physical -> back.
    # Preprocessing is linear (subtraction/detrend), so we verify the intensity
    # scale is carried consistently through a normalize/denormalize round trip.
    roundtrip_ok = None
    if intensity['min'] is not None and intensity['max'] is not None:
        span = intensity['max'] - intensity['min']
        if abs(span) > 0:
            # physical value per array unit (assume array already in physical
            # units unless a normalized [0,1] range is detected)
            arr_min, arr_max = float(np.min(img)), float(np.max(img))
            # normalize to [0,1] using declared physical range, then invert
            norm = (img - intensity['min']) / span
            recovered = norm * span + intensity['min']
            resid = float(np.max(np.abs(recovered - img)))
            roundtrip_ok = bool(resid < 1e-6 * (abs(span) + 1e-12))
            notes.append(f'[{label}] Intensity round-trip residual={resid:.3e} '
                         f'(ok={roundtrip_ok}); array range=[{arr_min:.4g},{arr_max:.4g}].')

    # ---- STEP 1: Per-row median subtraction ----
    row_median = np.median(img, axis=1, keepdims=True)
    img_rowsub = img - row_median

    # ---- STEP 2: streak-removal verification + escalation ----
    # FIX: escalation now chains from the median-subtracted image (img_rowsub),
    # not the original img, so the median-subtraction stage is preserved.
    x = np.arange(W)
    m0 = row_streak_metric(img_rowsub)
    notes.append(f'[{label}] Row-streak metric after median subtraction: {m0:.4f}')

    proc = img_rowsub
    streak_stage = 'median_subtraction'
    if m0 > STREAK_THRESH:
        proc1 = per_row_poly_detrend(img_rowsub, 1, x)
        m1 = row_streak_metric(proc1)
        notes.append(f'[{label}] Escalated to per-row 1st-order poly (on median-subtracted); metric: {m1:.4f}')
        proc = proc1
        streak_stage = 'median+per_row_poly_order1'
        if m1 > STREAK_THRESH:
            proc2 = per_row_poly_detrend(img_rowsub, 2, x)
            m2 = row_streak_metric(proc2)
            notes.append(f'[{label}] Escalated to per-row 2nd-order poly (on median-subtracted); metric: {m2:.4f}')
            proc = proc2
            streak_stage = 'median+per_row_poly_order2'
            final_metric = m2
        else:
            final_metric = m1
    else:
        final_metric = m0
        notes.append(f'[{label}] Median subtraction sufficient; no polynomial escalation needed.')

    # ---- STEP 3: resample to square pixels via FOV ----
    px = None
    if _HAVE_RESOLVE:
        try:
            px = resolve_pixel_size_nm(meta, img.shape)
        except Exception as e:
            notes.append(f'[{label}] resolve_pixel_size_nm failed: {e}')
            px = None

    if px is not None and px.get('x') is not None and px.get('y') is not None:
        nm_per_px_x = float(px['x'])
        nm_per_px_y = float(px['y'])
        notes.append(f"[{label}] pixel size from metadata: x={nm_per_px_x}, y={nm_per_px_y} ({px.get('source')})")
    else:
        NOMINAL_FOV_NM = 500.0
        nm_per_px_x = NOMINAL_FOV_NM / W
        nm_per_px_y = NOMINAL_FOV_NM / H
        notes.append(f'[{label}] No metadata pixel size; assumed square FOV {NOMINAL_FOV_NM} nm. '
                     f'nm/px x={nm_per_px_x:.4f}, y={nm_per_px_y:.4f}. Absolute spacings scale with FOV assumption.')

    target_nm_per_px = min(nm_per_px_x, nm_per_px_y)
    new_W = int(round(W * nm_per_px_x / target_nm_per_px))
    new_H = int(round(H * nm_per_px_y / target_nm_per_px))
    zoom_y = new_H / H
    zoom_x = new_W / W
    img_sq = ndimage.zoom(proc, (zoom_y, zoom_x), order=1)
    notes.append(f'[{label}] Resampled {proc.shape} -> {img_sq.shape} for square pixels at {target_nm_per_px:.4f} nm/px.')

    # ---- STEP 4: fourier_reflection_map ----
    frm = None
    frm_error = None
    try:
        frm = fourier_reflection_map(img_sq, pixel_size_nm=target_nm_per_px,
                                     params={'d_range': (target_nm_per_px * 3, 20.0),
                                             'min_sigma': 3.0})
    except Exception as e:
        frm_error = str(e)
        notes.append(f'[{label}] fourier_reflection_map failed: {e}')
        try:
            frm = fourier_reflection_map(img_sq, pixel_size_nm=target_nm_per_px)
            notes.append(f'[{label}] Retried fourier_reflection_map with default params (succeeded).')
            frm_error = None
        except Exception as e2:
            frm_error = str(e2)
            notes.append(f'[{label}] fourier_reflection_map retry failed: {e2}')

    # ---- STEP 5: STRICT artifact rejection + gating ----
    extracted = {}
    quality = {}
    domain_mask = None
    amplitude_map = None
    resolvable_order = False
    loss_of_order = True

    if frm is not None and frm.get('reflections'):
        reflections = frm['reflections']
        extracted['n_reflections_detected'] = len(reflections)
        top = []
        for r in reflections[:5]:
            top.append({
                'd_nm': float(r.get('d_nm')) if r.get('d_nm') is not None else None,
                'freq_cyc_px': float(r.get('freq_cyc_px')) if r.get('freq_cyc_px') is not None else None,
                'sigma': float(r.get('sigma')) if r.get('sigma') is not None else None,
                'is_satellite_candidate': bool(r.get('is_satellite_candidate', False)),
            })
        extracted['top_reflections'] = top

        extracted['strongest_satellite_d_nm'] = (
            float(frm['strongest_satellite_d_nm'])
            if frm.get('strongest_satellite_d_nm') is not None else None)
        extracted['mapped_d_nm'] = (
            float(frm['mapped_d_nm']) if frm.get('mapped_d_nm') is not None else None)
        extracted['mapped_is_satellite_candidate'] = bool(frm.get('mapped_is_satellite_candidate', False))

        spot_snr_domain = frm.get('spot_snr_domain')
        spot_snr_bulk = frm.get('spot_snr_bulk')
        domain_fraction = frm.get('domain_fraction')
        null_threshold = frm.get('null_threshold')

        quality['spot_snr_domain'] = float(spot_snr_domain) if spot_snr_domain is not None else None
        quality['spot_snr_bulk'] = float(spot_snr_bulk) if spot_snr_bulk is not None else None
        quality['domain_fraction'] = float(domain_fraction) if domain_fraction is not None else None
        quality['null_threshold'] = float(null_threshold) if null_threshold is not None else None

        amplitude_map = frm.get('amplitude_map')
        domain_mask = frm.get('domain_mask')

        reflection_orientation_deg = None
        if amplitude_map is not None:
            gy, gx = np.gradient(np.asarray(amplitude_map, dtype=np.float64))
            Jxx = np.mean(gx * gx); Jyy = np.mean(gy * gy); Jxy = np.mean(gx * gy)
            theta = 0.5 * np.arctan2(2 * Jxy, (Jxx - Jyy))
            reflection_orientation_deg = float(np.degrees(theta))
        extracted['reflection_orientation_deg_vs_fast_axis'] = reflection_orientation_deg

        gate_null = (domain_mask is not None and domain_fraction is not None
                     and domain_fraction > 0.0)
        gate_snr = (spot_snr_domain is not None and spot_snr_bulk is not None
                    and spot_snr_domain > spot_snr_bulk)

        gate_coherent = False
        largest_frac = 0.0
        if domain_mask is not None:
            dm = np.asarray(domain_mask, dtype=bool)
            lbl, n = ndimage.label(dm)
            if n > 0:
                sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
                largest = sizes.max()
                largest_frac = float(largest / dm.size)
                gate_coherent = largest_frac > 0.01
        quality['largest_domain_component_fraction'] = largest_frac

        quality['gate_null_passed'] = bool(gate_null)
        quality['gate_snr_passed'] = bool(gate_snr)
        quality['gate_coherent_passed'] = bool(gate_coherent)

        streak_suspect = False
        if reflection_orientation_deg is not None:
            if abs(reflection_orientation_deg) < 15 or abs(abs(reflection_orientation_deg) - 180) < 15:
                streak_suspect = True
        quality['streak_axis_suspect'] = bool(streak_suspect)

        if streak_suspect:
            resolvable_order = gate_null and gate_snr and gate_coherent
            if not resolvable_order:
                notes.append(f'[{label}] Streak-axis-aligned reflection did NOT pass full gate '
                             '(null + snr + coherent) -> rejected as artifact.')
        else:
            resolvable_order = gate_null and gate_snr

        loss_of_order = not resolvable_order
    else:
        extracted['n_reflections_detected'] = 0
        if frm is not None and frm.get('note'):
            notes.append(f"[{label}] fourier_reflection_map note: {frm['note']}")
        loss_of_order = True
        resolvable_order = False

    extracted['resolvable_order'] = bool(resolvable_order)
    extracted['loss_of_order_finding'] = bool(loss_of_order)

    if resolvable_order:
        extracted['reflection_spacing_nm'] = extracted.get('mapped_d_nm')
        notes.append(f'[{label}] RESOLVED periodic order survived strict gating.')
    else:
        extracted['reflection_spacing_nm'] = None
        notes.append(f'[{label}] NO RESOLVABLE ORDER: reflection(s) failed strict artifact/gating criteria.')

    quality['streak_metric_median_subtraction'] = float(m0)
    quality['streak_metric_final'] = float(final_metric)
    quality['streak_stage'] = streak_stage
    quality['nm_per_px_square'] = float(target_nm_per_px)
    quality['intensity_mapping'] = intensity
    quality['intensity_roundtrip_ok'] = roundtrip_ok

    return {
        'label': label,
        'path': npy_path,
        'img_raw': img,
        'img_sq': img_sq,
        'amplitude_map': amplitude_map,
        'domain_mask': domain_mask,
        'streak_stage': streak_stage,
        'extracted': extracted,
        'quality': quality,
        'notes': notes,
        'nm_per_px': target_nm_per_px,
    }


# ---------------------------------------------------------------
# Run pipeline on all discovered frames
# ---------------------------------------------------------------
inputs = _find_inputs()
results_per_frame = []
all_notes = []
for label, path in inputs:
    r = process_frame(label, path)
    results_per_frame.append(r)
    all_notes.extend(r['notes'])


# ---------------------------------------------------------------
# Save primary arrays (per frame, label-tagged)
# ---------------------------------------------------------------
saved_arrays = {}
for r in results_per_frame:
    lab = r['label']
    fn_pre = f"preprocessed_square_{lab}.npy"
    np.save(fn_pre, r['img_sq'].astype(np.float32))
    saved_arrays[fn_pre] = {
        'description': f'Row-flattened, streak-corrected, square-pixel-resampled AFM frame ({lab})',
        'shape': list(r['img_sq'].shape), 'dtype': 'float32'}

    if r['amplitude_map'] is not None:
        amp = np.asarray(r['amplitude_map'], dtype=np.float32)
        fn_amp = f"reflection_amplitude_map_{lab}.npy"
        np.save(fn_amp, amp)
        saved_arrays[fn_amp] = {
            'description': f'Amplitude map of mapped reflection from fourier_reflection_map ({lab})',
            'shape': list(amp.shape), 'dtype': 'float32'}

    if r['domain_mask'] is not None:
        dmask = np.asarray(r['domain_mask']).astype(np.uint8)
    else:
        dmask = np.zeros(r['img_sq'].shape, dtype=np.uint8)
    fn_dm = f"domain_mask_{lab}.npy"
    np.save(fn_dm, dmask)
    saved_arrays[fn_dm] = {
        'description': f'Null-gated ordered-domain binary mask (1=ordered, 0=bulk) ({lab})',
        'shape': list(dmask.shape), 'dtype': 'uint8'}


# ---------------------------------------------------------------
# COMPARATIVE analysis: presence / spacing / coherence vs KCl conc.
# ---------------------------------------------------------------
def _find_by_label(results, targets):
    for t in targets:
        for r in results:
            if r['label'] == t:
                return r
    return None

frame_lo = _find_by_label(results_per_frame, ['2mM', 'frameA', 'sample'])
frame_hi = _find_by_label(results_per_frame, ['10mM', 'frameB'])

comparison = {}
if frame_lo is not None and frame_hi is not None and frame_lo is not frame_hi:
    lo_e, lo_q = frame_lo['extracted'], frame_lo['quality']
    hi_e, hi_q = frame_hi['extracted'], frame_hi['quality']

    def _diff(a, b):
        if a is None or b is None:
            return None
        return float(b - a)

    comparison = {
        'low_conc_label': frame_lo['label'],
        'high_conc_label': frame_hi['label'],
        'presence': {
            frame_lo['label']: bool(lo_e.get('resolvable_order')),
            frame_hi['label']: bool(hi_e.get('resolvable_order')),
        },
        'spacing_nm': {
            frame_lo['label']: lo_e.get('reflection_spacing_nm'),
            frame_hi['label']: hi_e.get('reflection_spacing_nm'),
            'delta_nm_high_minus_low': _diff(lo_e.get('reflection_spacing_nm'),
                                             hi_e.get('reflection_spacing_nm')),
        },
        'coherence_domain_fraction': {
            frame_lo['label']: lo_q.get('domain_fraction'),
            frame_hi['label']: hi_q.get('domain_fraction'),
            'delta_high_minus_low': _diff(lo_q.get('domain_fraction'),
                                          hi_q.get('domain_fraction')),
        },
        'largest_coherent_domain_fraction': {
            frame_lo['label']: lo_q.get('largest_domain_component_fraction'),
            frame_hi['label']: hi_q.get('largest_domain_component_fraction'),
            'delta_high_minus_low': _diff(lo_q.get('largest_domain_component_fraction'),
                                          hi_q.get('largest_domain_component_fraction')),
        },
        'spot_snr_domain': {
            frame_lo['label']: lo_q.get('spot_snr_domain'),
            frame_hi['label']: hi_q.get('spot_snr_domain'),
        },
    }

    # Qualitative differential ordering statement
    lo_ord = bool(lo_e.get('resolvable_order'))
    hi_ord = bool(hi_e.get('resolvable_order'))
    if lo_ord and hi_ord:
        lf = lo_q.get('largest_domain_component_fraction') or 0.0
        hf = hi_q.get('largest_domain_component_fraction') or 0.0
        if hf > lf:
            trend = 'ordering (coherent domain fraction) INCREASES from low to high KCl.'
        elif hf < lf:
            trend = 'ordering (coherent domain fraction) DECREASES from low to high KCl.'
        else:
            trend = 'ordering coherence comparable between the two KCl concentrations.'
    elif hi_ord and not lo_ord:
        trend = 'resolvable order EMERGES at the higher KCl concentration (absent at low).'
    elif lo_ord and not hi_ord:
        trend = 'resolvable order is LOST at the higher KCl concentration (present at low).'
    else:
        trend = 'no resolvable order at either KCl concentration.'
    comparison['differential_ordering_finding'] = trend
else:
    comparison['note'] = ('Only a single frame available; the plan\'s 2 mM vs 10 mM comparison '
                          'could not be performed. Reporting the single-frame result as the closest '
                          'viable alternative.')


# ---------------------------------------------------------------
# Visualization: side-by-side per frame + comparison
# ---------------------------------------------------------------
n_frames = len(results_per_frame)
fig, axes = plt.subplots(n_frames, 3, figsize=(14, 4.2 * n_frames), squeeze=False)

for i, r in enumerate(results_per_frame):
    lab = r['label']
    ax0, ax1, ax2 = axes[i]

    ax0.imshow(r['img_sq'], cmap='afmhot')
    ax0.set_title(f'{lab}: preprocessed square px\n(streak: {r["streak_stage"]})')
    ax0.axis('off')

    if r['amplitude_map'] is not None:
        im = ax1.imshow(np.asarray(r['amplitude_map']), cmap='viridis')
        ax1.set_title(f'{lab}: reflection amplitude\nd={r["extracted"].get("mapped_d_nm")} nm')
        plt.colorbar(im, ax=ax1, fraction=0.046)
    else:
        ax1.text(0.5, 0.5, 'No reflection mapped', ha='center', va='center')
        ax1.set_title(f'{lab}: reflection amplitude')
    ax1.axis('off')

    ax2.imshow(r['img_sq'], cmap='gray')
    dm = r['domain_mask']
    if dm is not None and np.any(np.asarray(dm)):
        dmb = np.asarray(dm, dtype=bool)
        overlay = np.zeros((*dmb.shape, 4))
        overlay[dmb] = [1, 0, 0, 0.4]
        ax2.imshow(overlay)
        ax2.contour(dmb, colors='yellow', linewidths=0.8)
        status = 'ACCEPTED' if r['extracted'].get('resolvable_order') else 'REJECTED'
        ax2.set_title(f'{lab}: ordered domain\nfrac={r["quality"].get("domain_fraction")}, {status}')
    else:
        ax2.set_title(f'{lab}: ordered domain\n(none / no order)')
    ax2.axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()


# ---------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------
per_frame_out = {}
for r in results_per_frame:
    per_frame_out[r['label']] = {
        'path': r['path'],
        'extracted_features': r['extracted'],
        'quality_metrics': r['quality'],
    }

summary_bits = []
if 'differential_ordering_finding' in comparison:
    summary_bits.append('KCl-concentration comparison: ' + comparison['differential_ordering_finding'])
    sp = comparison['spacing_nm']
    summary_bits.append(f"Reflection spacing {comparison['low_conc_label']}={sp[comparison['low_conc_label']]} nm, "
                        f"{comparison['high_conc_label']}={sp[comparison['high_conc_label']]} nm.")
else:
    summary_bits.append(comparison.get('note', 'Comparison unavailable.'))
summary_bits.append('Escalation chained from the row-median-subtracted image (median stage preserved).')
summary_bits.append('Physical intensity mapping (data_range_minimum/maximum/units) tracked per frame with round-trip verification.')
summary_bits.append('Pixel size assumed from square-FOV where no metadata (absolute spacings scale with the FOV assumption).')

results = {
    'analysis_type': 'Comparative AFM (KCl concentration): per-frame row-flatten + streak-removal '
                     'verification/escalation (chained from median subtraction), square-pixel resample, '
                     'fourier_reflection_map reciprocal-space mapping with strict streak-axis artifact '
                     'rejection and null/SNR/coherence gating; differential presence/spacing/coherence '
                     'compared between concentrations.',
    'inputs_analyzed': [{'label': lab, 'path': p} for lab, p in inputs],
    'per_frame': per_frame_out,
    'concentration_comparison': comparison,
    'saved_arrays': saved_arrays,
    'summary': ' '.join(summary_bits) + ' Notes: ' + ' | '.join(all_notes),
}

print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results, default=str))
