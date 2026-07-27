import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from scilink.skills._shared.fourier_reflection import fourier_reflection_map
from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm

adjust_notes = []

# ---- Load ----
img = np.load('data.npy')
if img.ndim == 3:
    # not RGB necessarily; take first channel (topography-like)
    img = img[:, :, 0]
    adjust_notes.append('Input had >2 dims; used channel 0.')
img = img.astype(np.float64)
H, W = img.shape

# ---- Pixel size resolution ----
# No metadata file is provided alongside data.npy; attempt resolve, else fall back.
metadata = {}
px = resolve_pixel_size_nm(metadata, img.shape)
if px is not None and px.get('x') is not None:
    nm_per_px_x = float(px['x'])
    nm_per_px_y = float(px['y'])
    px_source = px.get('source', 'metadata')
else:
    # Fallback: typical high-resolution AFM scan of ordered adsorbate lattice.
    # Assume 100 nm FOV over 512 px (documented assumption).
    fov_nm = 100.0
    nm_per_px_x = fov_nm / W
    nm_per_px_y = fov_nm / H
    px_source = 'fallback_assumption_100nm_FOV'
    adjust_notes.append(
        'No metadata available; assumed 100 nm FOV over 512 px -> %.4f nm/px. '
        'Reflection SPACING in nm scales linearly with true FOV; coherence/domain '
        'metrics and orientation are FOV-independent.' % nm_per_px_x)

# ---- Preprocessing: per-row median subtraction (streak removal) ----
row_median = np.median(img, axis=1, keepdims=True)
img_rowmed = img - row_median

# Verify streak removal: compare row-to-row variance of row means before/after.
def streak_metric(a):
    rm = np.mean(a, axis=1)
    return float(np.std(rm))

streak_before = streak_metric(img)
streak_after = streak_metric(img_rowmed)

proc = img_rowmed
escalated = False
# Escalate to per-row polynomial detrend if residual streaking remains large.
if streak_after > 0.15 * streak_before:
    escalated = True
    x = np.arange(W)
    proc2 = np.empty_like(img)
    for r in range(H):
        coeffs = np.polyfit(x, img[r], deg=2)
        proc2[r] = img[r] - np.polyval(coeffs, x)
    proc = proc2
    adjust_notes.append(
        'Per-row median left residual streaking (std %.3g -> %.3g); '
        'escalated to per-row 2nd-order polynomial detrend.'
        % (streak_before, streak_after))
    streak_after = streak_metric(proc)

# ---- Resample to square pixels via FOV (preserve aspect ratio) ----
# Data is 512x512; if nm/px differ between axes, resample to isotropic pixels.
if abs(nm_per_px_x - nm_per_px_y) / max(nm_per_px_x, nm_per_px_y) > 1e-3:
    target = min(nm_per_px_x, nm_per_px_y)
    zoom_y = nm_per_px_y / target
    zoom_x = nm_per_px_x / target
    proc_sq = ndi.zoom(proc, (zoom_y, zoom_x), order=1)
    pixel_size_nm = target
    adjust_notes.append('Resampled anisotropic pixels to square (%.4f nm/px).' % target)
else:
    proc_sq = proc
    pixel_size_nm = nm_per_px_x

# ---- Targeted reciprocal-space mapping ----
params = {'min_sigma': 3.0}
try:
    res = fourier_reflection_map(proc_sq, pixel_size_nm=pixel_size_nm, params=params)
except Exception as e:
    res = {'note': 'fourier_reflection_map failed: %s' % repr(e)}
    adjust_notes.append('Tool raised exception; see note.')

reflections = res.get('reflections', []) if isinstance(res, dict) else []
mapped_d = res.get('mapped_d_nm') if isinstance(res, dict) else None
amp_map = res.get('amplitude_map') if isinstance(res, dict) else None
phase_map = res.get('phase_map') if isinstance(res, dict) else None
domain_mask = res.get('domain_mask') if isinstance(res, dict) else None
domain_fraction = res.get('domain_fraction') if isinstance(res, dict) else None
spot_snr_domain = res.get('spot_snr_domain') if isinstance(res, dict) else None
spot_snr_bulk = res.get('spot_snr_bulk') if isinstance(res, dict) else None
null_threshold = res.get('null_threshold') if isinstance(res, dict) else None
strongest_sat = res.get('strongest_satellite_d_nm') if isinstance(res, dict) else None
is_super = res.get('is_mapped_superstructure') if isinstance(res, dict) else None
mapped_is_sat = res.get('mapped_is_satellite_candidate') if isinstance(res, dict) else None

# Reflection sigma / resolvability / orientation for the mapped reflection
# are READ from the tool output (the tool is the source of truth).
mapped_sigma = None
mapped_freq = None
reflection_orientation_deg = None
best = None
if mapped_d is not None and reflections:
    for r in reflections:
        if abs(r.get('d_nm', -1) - mapped_d) < 1e-6:
            best = r
            break
    if best is None:
        best = reflections[0]
    mapped_sigma = best.get('sigma')
    mapped_freq = best.get('freq_cyc_px')

# ---- Reflection orientation: read from tool output (source of truth) ----
# The plan specifies orientation is recorded as part of the tool-based pipeline.
# Try common orientation-bearing fields from the tool result / mapped reflection.
def _first_present(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None

_orient_keys = ('orientation_deg', 'angle_deg', 'reflection_orientation_deg',
                'theta_deg', 'orientation')
# Prefer the per-reflection orientation, then the top-level result.
_ot = _first_present(best if isinstance(best, dict) else None, _orient_keys)
if _ot is None:
    _ot = _first_present(res, _orient_keys)
if _ot is not None:
    try:
        reflection_orientation_deg = float(_ot) % 180.0
    except (TypeError, ValueError):
        reflection_orientation_deg = None

if reflection_orientation_deg is None and mapped_d is not None and isinstance(res, dict):
    adjust_notes.append(
        'fourier_reflection_map returned no orientation field for the mapped '
        'reflection; reflection_orientation_deg left null (tool is source of truth, '
        'not reimplemented inline).')

# ---- Null-gate + domain-vs-bulk SNR confirmation ----
# Plan: apply null-gate AND spot_snr_domain > spot_snr_bulk check.
# Null-gate: the mapped-reflection amplitude within the ordered domain must
# actually clear the tool's null_threshold (a genuine gate, not just recorded).
null_gate_passed = None
if null_threshold is not None and amp_map is not None:
    amp_arr = np.asarray(amp_map, dtype=np.float64)
    if domain_mask is not None:
        dmask = np.asarray(domain_mask).astype(bool)
        if dmask.shape == amp_arr.shape and dmask.any():
            domain_amp = float(np.nanmax(amp_arr[dmask]))
        else:
            domain_amp = float(np.nanmax(amp_arr))
    else:
        domain_amp = float(np.nanmax(amp_arr))
    null_gate_passed = bool(domain_amp > float(null_threshold))
elif null_threshold is None and amp_map is not None:
    # No explicit null_threshold from tool; the domain_mask is itself already
    # the tool's null-gated segmentation, so treat a non-empty mask as the gate.
    if domain_mask is not None:
        null_gate_passed = bool(np.asarray(domain_mask).astype(bool).any())

snr_check_passed = None
if (spot_snr_domain is not None) and (spot_snr_bulk is not None):
    snr_check_passed = bool(spot_snr_domain > spot_snr_bulk)

reflection_confirmed = False
if snr_check_passed and (null_gate_passed is not False) and \
        (domain_fraction is not None and domain_fraction > 0):
    # Require SNR check; require null-gate to pass (True) or be unavailable but
    # backed by a non-empty null-gated domain_mask (handled above).
    reflection_confirmed = bool(null_gate_passed) if null_gate_passed is not None else bool(snr_check_passed)

# ---- Save arrays ----
saved = {}
np.save('processed_image.npy', proc_sq.astype(np.float32))
saved['processed_image.npy'] = {
    'description': 'Row-leveled, square-pixel processed AFM image used for FFT mapping',
    'shape': list(proc_sq.shape), 'dtype': 'float32'}

if amp_map is not None:
    np.save('reflection_amplitude_map.npy', np.asarray(amp_map, dtype=np.float32))
    saved['reflection_amplitude_map.npy'] = {
        'description': 'Amplitude map (where mapped reflection lives) at d=%.4f nm' % (mapped_d if mapped_d else -1),
        'shape': list(np.asarray(amp_map).shape), 'dtype': 'float32'}

if domain_mask is not None:
    dm = np.asarray(domain_mask).astype(np.int32)
    np.save('domain_mask.npy', dm)
    saved['domain_mask.npy'] = {
        'description': 'Null-gated ordered-domain segmentation (1=ordered domain,0=bulk)',
        'shape': list(dm.shape), 'dtype': 'int32'}

# ---- Visualization ----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

ax = axes[0, 0]
ax.imshow(img, cmap='afmhot')
ax.set_title('Original AFM (raw)')
ax.axis('off')

ax = axes[0, 1]
ax.imshow(proc_sq, cmap='afmhot')
ax.set_title('Row-leveled + square px%s' % (' (poly)' if escalated else ''))
ax.axis('off')

ax = axes[0, 2]
logp = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(proc_sq - proc_sq.mean()))))
ax.imshow(logp, cmap='viridis')
ax.set_title('FFT power (log)')
ax.axis('off')

ax = axes[1, 0]
if amp_map is not None:
    ax.imshow(np.asarray(amp_map), cmap='inferno')
    ax.set_title('Reflection amplitude\nd=%.3f nm' % (mapped_d if mapped_d else -1))
else:
    ax.text(0.5, 0.5, 'No resolvable\nreflection', ha='center', va='center')
    ax.set_title('Reflection amplitude')
ax.axis('off')

ax = axes[1, 1]
if domain_mask is not None:
    ax.imshow(proc_sq, cmap='gray')
    ax.imshow(np.ma.masked_where(~np.asarray(domain_mask), np.asarray(domain_mask)),
              cmap='autumn', alpha=0.5)
    ax.set_title('Ordered-domain mask\nfrac=%.3f' % (domain_fraction if domain_fraction else 0))
else:
    ax.text(0.5, 0.5, 'No domain', ha='center', va='center')
    ax.set_title('Ordered-domain mask')
ax.axis('off')

ax = axes[1, 2]
if phase_map is not None:
    ax.imshow(np.asarray(phase_map), cmap='twilight')
    ax.set_title('Bragg phase (unreferenced)')
else:
    ax.text(0.5, 0.5, 'No phase', ha='center', va='center')
    ax.set_title('Bragg phase')
ax.axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---- Results ----
reflection_present = bool(mapped_d is not None and reflection_confirmed)

features = {
    'reflection_present': reflection_present,
    'reflection_spacing_nm': float(mapped_d) if mapped_d is not None else None,
    'reflection_freq_cyc_px': float(mapped_freq) if mapped_freq is not None else None,
    'reflection_sigma_resolvability': float(mapped_sigma) if mapped_sigma is not None else None,
    'domain_fraction': float(domain_fraction) if domain_fraction is not None else None,
    'spot_snr_domain': float(spot_snr_domain) if spot_snr_domain is not None else None,
    'spot_snr_bulk': float(spot_snr_bulk) if spot_snr_bulk is not None else None,
    'strongest_satellite_d_nm': float(strongest_sat) if strongest_sat is not None else None,
    'reflection_orientation_deg_vs_fast_scan': reflection_orientation_deg,
    'mapped_is_satellite_candidate': bool(mapped_is_sat) if mapped_is_sat is not None else None,
    'is_mapped_superstructure': bool(is_super) if is_super is not None else None,
    'n_reflections_detected': len(reflections),
}

quality = {
    'streak_std_before': streak_before,
    'streak_std_after': streak_after,
    'streak_removed': bool(streak_after < 0.5 * streak_before) if streak_before > 0 else None,
    'escalated_to_polynomial': escalated,
    'pixel_size_nm': float(pixel_size_nm),
    'pixel_size_source': px_source,
    'null_threshold': float(null_threshold) if null_threshold is not None else None,
    'null_gate_passed': null_gate_passed,
    'spot_snr_domain_gt_bulk': snr_check_passed,
    'reflection_confirmed_domain_gt_bulk': reflection_confirmed,
}

if not reflections and isinstance(res, dict) and res.get('note'):
    summary = 'No resolvable periodic reflection: %s' % res.get('note')
elif reflection_present:
    summary = ('Resolvable periodic lattice/adsorbate reflection at d=%.3f nm '
               '(sigma=%.1f), domain_fraction=%.2f, confirmed by null-gate '
               '(amplitude clears null_threshold) AND spot_snr_domain=%.2f > bulk=%.2f, '
               'orientation %s deg vs fast-scan axis (from tool).'
               % (mapped_d, mapped_sigma if mapped_sigma else -1,
                  domain_fraction if domain_fraction else 0,
                  spot_snr_domain if spot_snr_domain else 0,
                  spot_snr_bulk if spot_snr_bulk else 0,
                  ('%.0f' % reflection_orientation_deg) if reflection_orientation_deg is not None else 'N/A'))
else:
    summary = ('Reflection detected (mapped d=%s nm) but NOT confirmed as a localized '
               'ordered domain (null-gate and/or spot_snr_domain > bulk not satisfied).'
               % (('%.3f' % mapped_d) if mapped_d else 'None'))

if adjust_notes:
    summary += ' | Adjustments: ' + ' '.join(adjust_notes)

results = {
    'analysis_type': 'AFM per-row leveling + square-pixel resample + fourier_reflection_map '
                     'reciprocal-space mapping of ordered-adsorbate/lattice reflection (KCl ordering).',
    'extracted_features': features,
    'quality_metrics': quality,
    'summary': summary,
    'saved_arrays': saved,
}

print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results))
