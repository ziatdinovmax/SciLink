import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.fourier_reflection import fourier_reflection_map

adjustments = []

# ---------------------------------------------------------------
# 0. Load metadata if available (for pixel size + z-calibration)
# ---------------------------------------------------------------
metadata = {}
for _mfn in ('metadata.json', 'system_info.json'):
    if os.path.exists(_mfn):
        try:
            with open(_mfn, 'r') as _f:
                metadata = json.load(_f)
            adjustments.append('Loaded metadata from %s.' % _mfn)
            break
        except Exception as _e:
            adjustments.append('Failed to read %s: %s' % (_mfn, repr(_e)))

def _meta_get(md, key):
    if not isinstance(md, dict):
        return None
    if key in md:
        return md[key]
    si = md.get('system_info')
    if isinstance(si, dict) and key in si:
        return si[key]
    return None

# ---------------------------------------------------------------
# 1. Load image (may be multi-channel, not RGB)
# ---------------------------------------------------------------
img = np.load('data.npy')
if img.ndim == 3:
    img = img[:, :, 0]
    adjustments.append('Input had >2D shape; used channel 0 as topography (height).')
raw_dtype = str(img.dtype)
img = img.astype(np.float64)

# ---------------------------------------------------------------
# 1b. z-scale calibration from height-channel metadata.
#     Convert raw counts -> nm. If unavailable, Rq stays in raw
#     z-units and is EXCLUDED from cross-concentration comparison.
# ---------------------------------------------------------------
z_min = _meta_get(metadata, 'data_range_minimum')
z_max = _meta_get(metadata, 'data_range_maximum')
z_units = _meta_get(metadata, 'data_range_units') or _meta_get(metadata, 'units')

z_scale = None  # physical-z (nm) per raw-count
if z_min is not None and z_max is not None:
    try:
        z_min = float(z_min); z_max = float(z_max)
        raw_lo = float(np.nanmin(img)); raw_hi = float(np.nanmax(img))
        raw_span = raw_hi - raw_lo
        if raw_span > 0:
            z_scale = (z_max - z_min) / raw_span
            adjustments.append(
                'z-calibration from metadata: raw [%.4g,%.4g] -> z [%.4g,%.4g] %s '
                '(z_scale=%.6g %s/count).' % (
                    raw_lo, raw_hi, z_min, z_max, str(z_units),
                    z_scale, str(z_units)))
    except Exception as _e:
        adjustments.append('Failed to build z-calibration mapping: %s' % repr(_e))

Rq_z_calibrated = z_scale is not None
if not Rq_z_calibrated:
    z_units = z_units or 'z-units'
    adjustments.append(
        'No data_range_minimum/maximum in metadata; z stored in raw '
        '%s counts. Rq reported in raw z-units, Rq_z_calibrated=false, '
        'and Rq EXCLUDED from cross-concentration comparison.' % raw_dtype)

# ---------------------------------------------------------------
# 2. AFM preprocessing: per-row median leveling
# ---------------------------------------------------------------
row_med = np.median(img, axis=1, keepdims=True)
leveled = img - row_med
leveled = leveled - np.median(leveled)

# ---------------------------------------------------------------
# 3. Resolve pixel size via FOV (verified-scale). Fall back to
#    assumed 20 nm FOV if genuinely unavailable, and FLAG scale.
# ---------------------------------------------------------------
px = resolve_pixel_size_nm(metadata, leveled.shape)
fov_recovered_flag = False
if px is None:
    assumed_fov_nm = 20.0
    nm_per_px_x = assumed_fov_nm / leveled.shape[1]
    nm_per_px_y = assumed_fov_nm / leveled.shape[0]
    px_source = 'assumed_fov_20nm_fallback'
    scale_uncertain_flag = True
    adjustments.append(
        'resolve_pixel_size_nm returned None (no metadata/FOV available); '
        'assumed FOV=20 nm -> %.4f nm/px. ALL nm-valued metrics flagged '
        'scale-uncertain and also reported in px.' % nm_per_px_x)
else:
    nm_per_px_x = float(px['x'])
    nm_per_px_y = float(px.get('y', px['x']))
    px_source = 'metadata'
    fov_recovered_flag = True
    scale_uncertain_flag = False
    adjustments.append(
        'Recovered FOV/pixel size from metadata (source=%s): x=%.4f y=%.4f nm/px.'
        % (px.get('source', 'metadata'), nm_per_px_x, nm_per_px_y))

# ---------------------------------------------------------------
# 4. Square-pixel resample via FOV
# ---------------------------------------------------------------
aniso_ratio = nm_per_px_y / nm_per_px_x if nm_per_px_x > 0 else 1.0
if abs(aniso_ratio - 1.0) > 1e-3:
    target_nm = min(nm_per_px_x, nm_per_px_y)
    ny0, nx0 = leveled.shape
    zoom_y = nm_per_px_y / target_nm
    zoom_x = nm_per_px_x / target_nm
    square = ndi.zoom(leveled, (zoom_y, zoom_x), order=1)
    nm_per_px = float(target_nm)
    adjustments.append(
        'Anisotropic pixels (x=%.4f, y=%.4f nm/px); resampled via FOV to '
        'square pixels at %.4f nm/px (shape %s -> %s).' % (
            nm_per_px_x, nm_per_px_y, nm_per_px, (ny0, nx0), square.shape))
else:
    square = leveled
    nm_per_px = float(nm_per_px_x)

# ---------------------------------------------------------------
# 4b. Mild detrend + light smoothing prior to autocorrelation.
#     Subtract a low-order polynomial background plane, then a light
#     Gaussian sized relative to pixel_size to suppress single-pixel
#     noise WITHOUT washing out corrugation (~1-2 1/nm).
# ---------------------------------------------------------------
ny, nx = square.shape
yy0, xx0 = np.mgrid[0:ny, 0:nx].astype(np.float64)
yn = (yy0 - ny / 2.0) / (ny / 2.0)
xn = (xx0 - nx / 2.0) / (nx / 2.0)
# 2nd-order polynomial background
A = np.column_stack([
    np.ones(square.size), xn.ravel(), yn.ravel(),
    (xn * xn).ravel(), (yn * yn).ravel(), (xn * yn).ravel()])
coef, *_ = np.linalg.lstsq(A, square.ravel(), rcond=None)
background = (A @ coef).reshape(square.shape)
detr = square - background
# light Gaussian: aim ~0.15 nm sigma (well below atomic spacing) but >= ~0.6 px
smooth_sigma_px = max(0.6, 0.15 / nm_per_px) if nm_per_px > 0 else 0.8
detr_sm = ndi.gaussian_filter(detr, sigma=smooth_sigma_px)
adjustments.append(
    'Detrend: subtracted 2nd-order polynomial background; light Gaussian '
    'smooth sigma=%.2f px (~%.3f nm) before autocorrelation.'
    % (smooth_sigma_px, smooth_sigma_px * nm_per_px))

# ---------------------------------------------------------------
# 5. Nyquist gate on target d_range for atomic/quasi-atomic lattice.
#    Only AUTHORIZE sub-nm reflection search when scale is metadata;
#    on fallback scale still run but tag results scale-conditional.
# ---------------------------------------------------------------
d_range = (0.15, 2.0)  # nm target periodicity
nyquist_d = 2.0 * nm_per_px
resolvable = nyquist_d < d_range[1]
sub_nm_authorized = (px_source == 'metadata')

# ---------------------------------------------------------------
# 6. Streak-collinearity check (avoid mapping scan-streak artifacts)
# ---------------------------------------------------------------
win = np.outer(np.hanning(ny), np.hanning(nx))
F = np.fft.fftshift(np.fft.fft2(detr * win))
P = np.abs(F) ** 2
cy, cx = np.array(P.shape) // 2
row_band = P[cy, :].sum() - P[cy, cx]
col_band = P[:, cx].sum() - P[cy, cx]
total_offdc = P.sum() - P[cy, cx]
streak_frac = float(max(row_band, col_band) / (total_offdc + 1e-12))
streak_collinear = streak_frac > 0.5

# ---------------------------------------------------------------
# 7. fourier_reflection_map if gates pass (do not force spots).
#    Report legitimate null if only diffuse lobe + streak.
# ---------------------------------------------------------------
refl_result = None
reflection_null_flag = None
amplitude_map = None
domain_mask = None
refl_features = {
    'reflection_d_nm': None, 'reflection_sigma': None,
    'spot_snr': None, 'spot_snr_bulk': None,
    'domain_fraction': None, 'strongest_satellite_d_nm': None,
    'n_reflections_detected': 0,
}

if resolvable and not streak_collinear:
    try:
        refl_result = fourier_reflection_map(
            square, pixel_size_nm=nm_per_px,
            params={'d_range': d_range, 'min_sigma': 3.0})
    except Exception as e:
        adjustments.append('fourier_reflection_map failed: %s' % repr(e))
        refl_result = None
else:
    reason = []
    if not resolvable:
        reason.append('target d_range below Nyquist (nm/px=%.4f)' % nm_per_px)
    if streak_collinear:
        reason.append('FFT dominated by axis-collinear scan streak (frac=%.2f)' % streak_frac)
    adjustments.append('Skipped fourier_reflection_map: ' + '; '.join(reason))

if refl_result is not None and 'note' not in refl_result and \
        refl_result.get('amplitude_map') is not None:
    amplitude_map = np.asarray(refl_result['amplitude_map'])
    domain_mask = np.asarray(refl_result.get('domain_mask'))
    reflections = refl_result.get('reflections', [])
    mapped_d = refl_result.get('mapped_d_nm')
    top_sigma = reflections[0]['sigma'] if reflections else None
    reflection_null_flag = False
    refl_features = {
        'reflection_d_nm': float(mapped_d) if mapped_d is not None else None,
        'reflection_sigma': float(top_sigma) if top_sigma is not None else None,
        'spot_snr': float(refl_result.get('spot_snr_domain'))
            if refl_result.get('spot_snr_domain') is not None else None,
        'spot_snr_bulk': float(refl_result.get('spot_snr_bulk'))
            if refl_result.get('spot_snr_bulk') is not None else None,
        'domain_fraction': float(refl_result.get('domain_fraction'))
            if refl_result.get('domain_fraction') is not None else None,
        'strongest_satellite_d_nm': (float(refl_result['strongest_satellite_d_nm'])
            if refl_result.get('strongest_satellite_d_nm') is not None else None),
        'n_reflections_detected': int(len(reflections)),
    }
    if sub_nm_authorized:
        adjustments.append(
            'fourier_reflection_map: %d reflection(s); mapped d=%s nm sigma=%s '
            '(scale from metadata).' % (
                len(reflections),
                ('%.3f' % refl_features['reflection_d_nm']) if refl_features['reflection_d_nm'] else 'NA',
                ('%.1f' % refl_features['reflection_sigma']) if refl_features['reflection_sigma'] else 'NA'))
    else:
        adjustments.append(
            'fourier_reflection_map ran on FALLBACK scale (px_source=%s): '
            'reflection results are SCALE-CONDITIONAL; d_nm values are '
            'scale-uncertain.' % px_source)
else:
    # Legitimate null: only diffuse lobe + streak, no discrete spot.
    if refl_result is not None and 'note' in refl_result:
        adjustments.append('fourier_reflection_map (legitimate null): ' + str(refl_result.get('note')))
        reflection_null_flag = True
    elif refl_result is not None:
        reflection_null_flag = True
    # if refl_result is None because gate skipped, leave null_flag None

# ---------------------------------------------------------------
# 8. ALWAYS: radial PSD. Extract BOTH the low-frequency envelope
#    length (large-scale height undulation) AND the mid-frequency
#    (~1-2 1/nm) corrugation peak wavelength / amplitude / width.
# ---------------------------------------------------------------
yy, xx = np.indices((ny, nx))
r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
r_int = r.astype(int)
psd_sum = np.bincount(r_int.ravel(), weights=P.ravel())
psd_cnt = np.bincount(r_int.ravel())
radial_psd = psd_sum / np.maximum(psd_cnt, 1)
freq_px = np.arange(len(radial_psd))
kmax = min(len(radial_psd), ny // 2)
# spatial frequency (1/nm) for each radial bin
freq_inv_nm = (freq_px / float(ny)) / nm_per_px if nm_per_px > 0 else freq_px * 0.0

# --- low-frequency envelope length: dominant peak in low-freq band
# (exclude DC r<2). Low-freq band = below ~0.5 1/nm (large undulation).
low_search = radial_psd.copy().astype(np.float64)
low_search[:2] = 0
low_band_mask = np.zeros(len(radial_psd), dtype=bool)
low_band_mask[2:kmax] = freq_inv_nm[2:kmax] < 0.5
low_search[~low_band_mask] = 0
low_freq_envelope_length_nm = None
if low_search.max() > 0:
    lpk = int(np.argmax(low_search))
    fnm = freq_inv_nm[lpk]
    if fnm > 0:
        low_freq_envelope_length_nm = float(1.0 / fnm)

# --- mid-frequency corrugation peak in ~1-2 1/nm band ---
mid_lo, mid_hi = 1.0, 2.0  # 1/nm
mid_band_mask = np.zeros(len(radial_psd), dtype=bool)
mid_band_mask[2:kmax] = (freq_inv_nm[2:kmax] >= mid_lo) & (freq_inv_nm[2:kmax] <= mid_hi)
mid_freq_peak_wavelength_nm = None
mid_freq_peak_amplitude = None
mid_freq_peak_width = None
if mid_band_mask.any():
    band_idx = np.where(mid_band_mask)[0]
    band_psd = radial_psd[band_idx]
    # local background: median of the mid band, subtracted for amplitude
    bg = float(np.median(band_psd))
    pk_rel = int(np.argmax(band_psd))
    pk_idx = int(band_idx[pk_rel])
    fnm_pk = freq_inv_nm[pk_idx]
    if fnm_pk > 0:
        mid_freq_peak_wavelength_nm = float(1.0 / fnm_pk)
    mid_freq_peak_amplitude = float(band_psd[pk_rel] - bg)
    # width: FWHM in 1/nm around the peak (half-prominence crossing)
    peak_val = float(band_psd[pk_rel])
    half = bg + 0.5 * (peak_val - bg)
    # walk left/right within full spectrum from pk_idx
    li = pk_idx
    while li > 2 and radial_psd[li] > half:
        li -= 1
    ri = pk_idx
    while ri < kmax - 1 and radial_psd[ri] > half:
        ri += 1
    if ri > li:
        mid_freq_peak_width = float(abs(freq_inv_nm[ri] - freq_inv_nm[li]))

# corrugation length = mid-frequency peak wavelength (scale-appropriate)
corrugation_wavelength_nm = mid_freq_peak_wavelength_nm

# ---------------------------------------------------------------
# 9. ALWAYS: autocorrelation correlation length (on detrended+smoothed).
#    Validate against BOTH pixel size and the mid-freq PSD peak;
#    reject correlation lengths that collapse to ~1 px as noise.
# ---------------------------------------------------------------
f = detr_sm - detr_sm.mean()
ac = np.fft.ifft2(np.abs(np.fft.fft2(f)) ** 2).real
ac = np.fft.fftshift(ac)
ac /= ac.max()
ac_sum = np.bincount(r_int.ravel(), weights=ac.ravel())
ac_rad = ac_sum / np.maximum(psd_cnt, 1)
thr = 1.0 / np.e
corr_len_px = None
for rr in range(1, min(len(ac_rad), ny // 2)):
    if ac_rad[rr] < thr:
        a0, a1 = ac_rad[rr - 1], ac_rad[rr]
        frac = (a0 - thr) / (a0 - a1 + 1e-12)
        corr_len_px = (rr - 1) + frac
        break

correlation_length_in_px = float(corr_len_px) if corr_len_px is not None else None
correlation_length_nm = (float(corr_len_px * nm_per_px)
                         if corr_len_px is not None else None)
# noise gate: collapse to ~1 px means pixel-noise-limited, not real order
correlation_length_noise_flag = bool(
    corr_len_px is not None and corr_len_px <= 1.5)
if correlation_length_noise_flag:
    adjustments.append(
        'Correlation length collapsed to ~%.2f px (<=1.5 px): flagged as '
        'pixel-noise-limited, NOT genuine atomic order.' % corr_len_px)
# cross-check against mid-freq PSD peak wavelength
if (correlation_length_nm is not None and corrugation_wavelength_nm is not None
        and not correlation_length_noise_flag):
    ratio = correlation_length_nm / corrugation_wavelength_nm
    adjustments.append(
        'Correlation length %.3f nm vs mid-freq corrugation wavelength '
        '%.3f nm (ratio=%.2f).' % (correlation_length_nm,
                                   corrugation_wavelength_nm, ratio))

# ---------------------------------------------------------------
# 10. ALWAYS: RMS roughness Rq. Calibrate z to nm if metadata present;
#     else keep raw z-units and EXCLUDE from cross-concentration.
# ---------------------------------------------------------------
rq_raw = float(np.sqrt(np.mean((square - square.mean()) ** 2)))
if Rq_z_calibrated:
    Rq_nm = float(rq_raw * abs(z_scale))
    rq_units = z_units or 'nm'
    adjustments.append('Rq z-calibrated: Rq=%.4g %s.' % (Rq_nm, rq_units))
else:
    Rq_nm = float(rq_raw)  # raw z-units
    rq_units = z_units or ('raw %s counts' % raw_dtype)
    adjustments.append('Rq in raw z-units (%s); excluded from cross-conc comparison.' % rq_units)

# ---------------------------------------------------------------
# 11. ALWAYS: coverage fraction (Otsu on leveled surface)
# ---------------------------------------------------------------
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

sm = ndi.gaussian_filter(square, sigma=1.0)
try:
    t = threshold_otsu(sm)
except Exception:
    t = np.median(sm)
patch_mask = sm > t
patch_mask = ndi.binary_opening(patch_mask, iterations=1)
coverage = float(patch_mask.mean())
lbl = label(patch_mask)
props = regionprops(lbl)
if props:
    areas_px = np.array([p.area for p in props])
    eq_diam_nm = np.array([2.0 * np.sqrt(a / np.pi) for a in areas_px]) * nm_per_px
    mean_patch_size_nm = float(np.mean(eq_diam_nm))
    n_patches = int(len(props))
else:
    mean_patch_size_nm = None
    n_patches = 0

# ---------------------------------------------------------------
# 12. Save arrays
# ---------------------------------------------------------------
np.save('leveled_image.npy', square.astype(np.float32))
np.save('detrended_smoothed.npy', detr_sm.astype(np.float32))
np.save('patch_mask.npy', patch_mask.astype(np.uint8))
saved = {
    'leveled_image.npy': {
        'description': 'Per-row median leveled AFM topography (square pixels).',
        'shape': list(square.shape), 'dtype': 'float32'},
    'detrended_smoothed.npy': {
        'description': 'Polynomial-detrended + light-Gaussian-smoothed surface used for autocorrelation.',
        'shape': list(detr_sm.shape), 'dtype': 'float32'},
    'patch_mask.npy': {
        'description': 'Binary Otsu coverage mask (1=elevated patch, 0=background).',
        'shape': list(patch_mask.shape), 'dtype': 'uint8'},
}
if amplitude_map is not None:
    np.save('reflection_amplitude_map.npy', amplitude_map.astype(np.float32))
    saved['reflection_amplitude_map.npy'] = {
        'description': 'fourier_reflection_map amplitude map of the mapped lattice reflection.',
        'shape': list(amplitude_map.shape), 'dtype': 'float32'}
if domain_mask is not None:
    np.save('reflection_domain_mask.npy', domain_mask.astype(np.uint8))
    saved['reflection_domain_mask.npy'] = {
        'description': 'Null-gated ordered-lattice domain mask from fourier_reflection_map.',
        'shape': list(domain_mask.shape), 'dtype': 'uint8'}

# ---------------------------------------------------------------
# 13. Visualization (headline result in visualization.png)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(img, cmap='afmhot')
axes[0, 0].set_title('Original AFM (raw ch0)')
axes[0, 0].axis('off')

axes[0, 1].imshow(square, cmap='afmhot')
axes[0, 1].set_title('Row-leveled topography (%.4f nm/px, %s)' % (nm_per_px, px_source))
axes[0, 1].axis('off')

axes[0, 2].imshow(np.log1p(P), cmap='inferno')
axes[0, 2].set_title('FFT power (log)\nstreak_frac=%.2f' % streak_frac)
axes[0, 2].axis('off')

# radial PSD with low-freq envelope + mid-freq corrugation markers
valid = np.arange(1, kmax)
axes[1, 0].loglog(np.maximum(freq_inv_nm[valid], 1e-6),
                  np.maximum(radial_psd[valid], 1e-12), color='k', lw=0.8)
axes[1, 0].axvspan(mid_lo, mid_hi, color='orange', alpha=0.15, label='mid band 1-2 1/nm')
if low_freq_envelope_length_nm:
    axes[1, 0].axvline(1.0 / low_freq_envelope_length_nm, color='b', ls='--',
                       label='envelope L=%.2f nm' % low_freq_envelope_length_nm)
if corrugation_wavelength_nm:
    axes[1, 0].axvline(1.0 / corrugation_wavelength_nm, color='r', ls='--',
                       label='corrugation lam=%.3f nm' % corrugation_wavelength_nm)
axes[1, 0].set_xlabel('spatial freq (1/nm)')
axes[1, 0].set_ylabel('radial PSD')
axes[1, 0].set_title('Radial PSD: envelope vs corrugation')
axes[1, 0].legend(fontsize=7)

# reflection amplitude map or legitimate-null note
if amplitude_map is not None:
    axes[1, 1].imshow(amplitude_map, cmap='viridis')
    ttl = 'Reflection amplitude\nd=%s nm, sigma=%s%s' % (
        ('%.3f' % refl_features['reflection_d_nm']) if refl_features['reflection_d_nm'] else 'NA',
        ('%.1f' % refl_features['reflection_sigma']) if refl_features['reflection_sigma'] else 'NA',
        '' if sub_nm_authorized else ' (scale-cond.)')
    axes[1, 1].set_title(ttl)
    axes[1, 1].axis('off')
else:
    msg = ('Legitimate reflection null\n(diffuse lobe + streak,\nno discrete spot)'
           if reflection_null_flag else 'Reflection search skipped\n(gate: Nyquist/streak)')
    axes[1, 1].text(0.5, 0.5, msg, ha='center', va='center')
    axes[1, 1].set_title('Reflection map')
    axes[1, 1].axis('off')

# coverage overlay
axes[1, 2].imshow(square, cmap='gray')
ax_ov = np.zeros((*patch_mask.shape, 4))
ax_ov[patch_mask] = [1, 0, 0, 0.35]
axes[1, 2].imshow(ax_ov)
axes[1, 2].contour(patch_mask, colors='yellow', linewidths=0.5)
axes[1, 2].set_title('Coverage=%.2f, n=%d' % (coverage, n_patches))
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# Full-resolution verifier panels
fig2, ax2 = plt.subplots(figsize=(6, 6))
valid = np.arange(1, kmax)
ax2.loglog(np.maximum(freq_inv_nm[valid], 1e-6),
           np.maximum(radial_psd[valid], 1e-12), color='k', lw=0.9)
ax2.axvspan(mid_lo, mid_hi, color='orange', alpha=0.15)
if low_freq_envelope_length_nm:
    ax2.axvline(1.0 / low_freq_envelope_length_nm, color='b', ls='--',
                label='envelope L=%.2f nm' % low_freq_envelope_length_nm)
if corrugation_wavelength_nm:
    ax2.axvline(1.0 / corrugation_wavelength_nm, color='r', ls='--',
                label='corrugation lam=%.3f nm' % corrugation_wavelength_nm)
ax2.set_xlabel('spatial freq (1/nm)'); ax2.set_ylabel('radial PSD')
ax2.set_title('Radial PSD (full res)'); ax2.legend(fontsize=8)
plt.tight_layout(); plt.savefig('verifier_panel_radial_psd.png', dpi=100); plt.close()

if amplitude_map is not None:
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    im3 = ax3.imshow(amplitude_map, cmap='viridis')
    ax3.set_title('Reflection amplitude map (d=%s nm)'
                  % (('%.3f' % refl_features['reflection_d_nm'])
                     if refl_features['reflection_d_nm'] else 'NA'))
    ax3.axis('off'); plt.colorbar(im3, ax=ax3, fraction=0.046)
    plt.tight_layout(); plt.savefig('verifier_panel_reflection_amplitude.png', dpi=100); plt.close()

# ---------------------------------------------------------------
# 14. Results JSON
# ---------------------------------------------------------------
extracted = {
    'pixel_size_nm': float(nm_per_px),
    'pixel_size_nm_x_raw': float(nm_per_px_x),
    'pixel_size_nm_y_raw': float(nm_per_px_y),
    'pixel_size_source': px_source,
    'fov_recovered_flag': bool(fov_recovered_flag),
    'scale_uncertain_flag': bool(scale_uncertain_flag),
    'nyquist_d_nm': float(nyquist_d),
    'target_d_range_nm': list(d_range),
    'nyquist_resolvable': bool(resolvable),
    'sub_nm_reflection_authorized': bool(sub_nm_authorized),
    'streak_collinear': bool(streak_collinear),
    'streak_fraction': float(streak_frac),
    'Rq_nm': float(Rq_nm),
    'Rq_units': rq_units,
    'Rq_z_calibrated': bool(Rq_z_calibrated),
    'reflection_d_nm': refl_features['reflection_d_nm'],
    'reflection_sigma': refl_features['reflection_sigma'],
    'spot_snr': refl_features['spot_snr'],
    'spot_snr_bulk': refl_features['spot_snr_bulk'],
    'domain_fraction': refl_features['domain_fraction'],
    'reflection_null_flag': (bool(reflection_null_flag)
                             if reflection_null_flag is not None else None),
    'strongest_satellite_d_nm': refl_features['strongest_satellite_d_nm'],
    'n_reflections_detected': refl_features['n_reflections_detected'],
    'radial_psd_low_freq_envelope_length_nm': (float(low_freq_envelope_length_nm)
        if low_freq_envelope_length_nm is not None else None),
    'radial_psd_mid_freq_peak_wavelength_nm': (float(mid_freq_peak_wavelength_nm)
        if mid_freq_peak_wavelength_nm is not None else None),
    'mid_freq_peak_amplitude': (float(mid_freq_peak_amplitude)
        if mid_freq_peak_amplitude is not None else None),
    'mid_freq_peak_width': (float(mid_freq_peak_width)
        if mid_freq_peak_width is not None else None),
    'correlation_length_nm': correlation_length_nm,
    'correlation_length_in_px': correlation_length_in_px,
    'correlation_length_noise_flag': bool(correlation_length_noise_flag),
    'coverage': coverage,
    'mean_patch_size_nm': mean_patch_size_nm,
    'n_patches': n_patches,
}

quality = {
    'reflection_map_run': bool(amplitude_map is not None),
    'reflection_null_flag': (bool(reflection_null_flag)
                             if reflection_null_flag is not None else None),
    'spot_snr_domain_gt_bulk': (
        bool(refl_features['spot_snr'] is not None and
             refl_features['spot_snr_bulk'] is not None and
             refl_features['spot_snr'] > 3.0 * refl_features['spot_snr_bulk'])
        if refl_features['spot_snr'] is not None else None),
    'n_reflections_detected': refl_features['n_reflections_detected'],
    'scale_valid_for_cross_conc': bool(px_source == 'metadata'),
    'Rq_valid_for_cross_conc': bool(Rq_z_calibrated),
    'correlation_length_noise_flag': bool(correlation_length_noise_flag),
}

lattice_present = bool(refl_features['reflection_d_nm'] is not None and
                       refl_features['reflection_sigma'] is not None and
                       refl_features['reflection_sigma'] >= 3.0)
summary = (
    'Row-leveled AFM at %.4f nm/px (source=%s, scale_uncertain=%s). '
    'Reflection: %s. Corrugation (mid-freq PSD) lam=%s nm (amp=%s, width=%s 1/nm); '
    'low-freq envelope L=%s nm. Correlation length=%s nm (%s px, noise_flag=%s). '
    'Rq=%.3g %s (z_calibrated=%s). Coverage=%.2f. '
    'Parameter adjustments: detrend=2nd-order poly + Gaussian sigma=%.2f px before '
    'autocorr; mid-freq band 1-2 1/nm; corr-noise gate at 1.5 px; reflection search '
    'tagged scale-conditional on fallback FOV. %s' % (
        nm_per_px, px_source, scale_uncertain_flag,
        ('present d=%.3f nm sigma=%.1f%s' % (
            refl_features['reflection_d_nm'], refl_features['reflection_sigma'],
            '' if sub_nm_authorized else ' [scale-conditional]'))
            if lattice_present else
            ('legitimate null (diffuse lobe + streak)' if reflection_null_flag
             else 'not resolvable/skipped'),
        ('%.3f' % corrugation_wavelength_nm) if corrugation_wavelength_nm else 'NA',
        ('%.3g' % mid_freq_peak_amplitude) if mid_freq_peak_amplitude is not None else 'NA',
        ('%.3f' % mid_freq_peak_width) if mid_freq_peak_width is not None else 'NA',
        ('%.2f' % low_freq_envelope_length_nm) if low_freq_envelope_length_nm else 'NA',
        ('%.3f' % correlation_length_nm) if correlation_length_nm else 'NA',
        ('%.2f' % correlation_length_in_px) if correlation_length_in_px else 'NA',
        correlation_length_noise_flag,
        Rq_nm, rq_units, Rq_z_calibrated, coverage, smooth_sigma_px,
        ' | '.join(adjustments) if adjustments else 'none'))

results = {
    'analysis_type': ('AFM row-leveling + z-calibration + verified-scale square-pixel '
                      'resampling + polynomial detrend/light-smooth + Nyquist/streak-gated '
                      'fourier_reflection_map (legitimate-null aware) + radial PSD split into '
                      'low-freq envelope vs mid-freq corrugation + noise-gated autocorrelation '
                      'correlation length + z-gated Rq + coverage.'),
    'extracted_features': extracted,
    'quality_metrics': quality,
    'summary': summary,
    'saved_arrays': saved,
}
print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results))
