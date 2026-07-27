import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.fourier_reflection import fourier_reflection_map

summary_notes = []

# ---------------------------------------------------------------
# 1. Load data + metadata
# ---------------------------------------------------------------
image_raw = np.load('data.npy')
if image_raw.ndim == 3:
    # not RGB per problem; take first channel
    image_raw = image_raw[:, :, 0]
    summary_notes.append('Input had extra dims; used channel 0.')
image_raw = image_raw.astype(np.float64)
N_rows, N_cols = image_raw.shape

metadata = None
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    summary_notes.append(f'metadata.json not read ({e}); using plan fallbacks.')

# ---------------------------------------------------------------
# 2. Map uint16 -> physical height (nm) using metadata data_range
# ---------------------------------------------------------------
# defaults from plan
z_min, z_max = -974.27, -916.03
z_units = 'nm'

def find_key(d, key):
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    for v in d.values():
        r = find_key(v, key)
        if r is not None:
            return r
    return None

if metadata is not None:
    dmin = find_key(metadata, 'data_range_minimum')
    dmax = find_key(metadata, 'data_range_maximum')
    dunits = find_key(metadata, 'data_range_units')
    if dmin is not None and dmax is not None:
        z_min, z_max = float(dmin), float(dmax)
        summary_notes.append(f'Used metadata data_range [{z_min},{z_max}].')
    if dunits is not None:
        z_units = str(dunits)

# physical = value * (z_max - z_min)/65535 + z_min
height = image_raw * (z_max - z_min) / 65535.0 + z_min  # in nm

# ---------------------------------------------------------------
# 3. Leveling: global plane fit
# ---------------------------------------------------------------
yy, xx = np.mgrid[0:N_rows, 0:N_cols]
A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
coef, *_ = np.linalg.lstsq(A, height.ravel(), rcond=None)
plane = (A @ coef).reshape(height.shape)
lev = height - plane
rms_before = float(np.std(height))
rms_after_plane = float(np.std(lev))

# per-row median subtraction (robust horizontal baseline removal)
row_med = np.median(lev, axis=1, keepdims=True)
lev = lev - row_med

# additional directional destripe: subtract low-order polynomial along columns
# to attenuate slow diagonal scan streaks (documented choice: 2nd-order per column)
col_idx = np.arange(N_rows)
col_bg = np.zeros_like(lev)
for c in range(N_cols):
    p = np.polyfit(col_idx, lev[:, c], 2)
    col_bg[:, c] = np.polyval(p, col_idx)
# low-pass the column background across columns to keep it a smooth destripe field
col_bg = ndimage.uniform_filter1d(col_bg, size=9, axis=1, mode='nearest')
lev = lev - col_bg
summary_notes.append('Destripe: per-row median + smoothed 2nd-order per-column polynomial subtraction.')

rms_after = float(np.std(lev))
Rq = float(np.sqrt(np.mean((lev - lev.mean())**2)))
corrugation_rms = Rq

# ---------------------------------------------------------------
# 4. Resample to square pixels using metadata FOV
# ---------------------------------------------------------------
px = resolve_pixel_size_nm(metadata, image_raw.shape) if metadata is not None else None
if px is not None and px.get('x') and px.get('y'):
    nm_per_px_x = float(px['x'])
    nm_per_px_y = float(px['y'])
    summary_notes.append(f"pixel size from metadata: x={nm_per_px_x:.4f}, y={nm_per_px_y:.4f} nm/px ({px.get('source')}).")
else:
    # plan fallback FOVs
    nm_per_px_x = 400.8 / N_cols
    nm_per_px_y = 401.6 / N_rows
    summary_notes.append('pixel size from plan FOV fallback (400.8/Nx, 401.6/Ny).')

# target square pixel = smaller of the two (finer sampling)
target_px = min(nm_per_px_x, nm_per_px_y)
out_cols = int(round(N_cols * nm_per_px_x / target_px))
out_rows = int(round(N_rows * nm_per_px_y / target_px))
square = resize(lev, (out_rows, out_cols), order=1, mode='reflect',
                anti_aliasing=True, preserve_range=True).astype(np.float64)
nm_per_px = float(target_px)
summary_notes.append(f'Resampled {N_rows}x{N_cols} -> {out_rows}x{out_cols}, {nm_per_px:.4f} nm/px (bilinear).')

# ---------------------------------------------------------------
# 5. Detect & mask fast-scan-line frequency in FFT
# ---------------------------------------------------------------
# Fast scan is along rows (x). Scan-line artifact = horizontal striping ->
# vertical frequency (varying row-to-row). Identify dominant vertical freq.
F = np.fft.fftshift(np.fft.fft2(square))
power = np.abs(F)**2
cr, cc = out_rows // 2, out_cols // 2
# column of vertical frequencies (kx=0)
vert_profile = power[:, cc].copy()
vert_profile[cr] = 0  # ignore DC
# find peak vertical frequency (row index)
peak_row = int(np.argmax(vert_profile))
fy_cyc_px = abs(peak_row - cr) / out_rows
scanline_period_nm = (1.0 / fy_cyc_px * nm_per_px) if fy_cyc_px > 0 else None
summary_notes.append(f'Detected scan-line vertical freq {fy_cyc_px:.4f} cyc/px'
                     + (f' (~{scanline_period_nm:.2f} nm period).' if scanline_period_nm else '.'))

# Build FFT with scan-line freq + harmonics masked (for a cleaned view / artifact note)
F_masked = F.copy()
for h in range(1, 5):
    off = int(round(h * fy_cyc_px * out_rows))
    if off == 0:
        continue
    for sgn in (+1, -1):
        rr = cr + sgn * off
        if 0 <= rr < out_rows:
            F_masked[max(0, rr-1):rr+2, cc-1:cc+2] = 0
square_clean = np.real(np.fft.ifft2(np.fft.ifftshift(F_masked)))

# ---------------------------------------------------------------
# 6. fourier_reflection_map restricted to peptide self-assembly band ~3-10 nm
# ---------------------------------------------------------------
fr = fourier_reflection_map(
    square_clean,
    pixel_size_nm=nm_per_px,
    params={
        'd_range': (3.0, 10.0),
        'min_sigma': 3.0,
        'bandwidth_frac': 0.12,
        'null_percentile': 99.5,
    },
)

reflections = fr.get('reflections', [])
mapped_d = fr.get('mapped_d_nm')
spot_dom = fr.get('spot_snr_domain')
spot_bulk = fr.get('spot_snr_bulk')
domain_fraction = fr.get('domain_fraction')
amp_map = fr.get('amplitude_map')
domain_mask = fr.get('domain_mask')
null_thr = fr.get('null_threshold')

# best reflection sigma
best_sigma = None
if reflections:
    best_sigma = float(reflections[0].get('sigma'))

# Validity checks:
# (a) cleared null floor -> a reflection exists / mapped
# (b) NOT the scan-line frequency
# (c) spot_snr_domain > 3x spot_snr_bulk
valid = False
reason = 'no resolvable periodic self-assembly'
if mapped_d is not None and amp_map is not None:
    mapped_freq_cyc_px = nm_per_px / mapped_d  # cyc/px
    freq_matches_scanline = (fy_cyc_px > 0 and
                             abs(mapped_freq_cyc_px - fy_cyc_px) < 0.15 * fy_cyc_px)
    snr_ok = (spot_dom is not None and spot_bulk is not None and
              spot_dom > 3.0 * max(spot_bulk, 1e-9))
    if freq_matches_scanline:
        reason = 'mapped reflection coincides with masked scan-line frequency (rejected)'
    elif not snr_ok:
        reason = 'reflection does not clear spot_snr_domain > 3x spot_snr_bulk (rejected)'
    else:
        valid = True
        reason = 'valid self-assembly reflection'

# ---------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

ax = axes[0, 0]
im = ax.imshow(height, cmap='afmhot')
ax.set_title(f'Raw height (nm)\nRMS={rms_before:.2f} nm')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 1]
im = ax.imshow(lev, cmap='afmhot')
ax.set_title(f'Leveled residual (nm)\nRq={Rq:.3f} nm')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 2]
im = ax.imshow(square_clean, cmap='afmhot', aspect='equal')
ax.set_title(f'Square px, scan-line masked\n{nm_per_px:.3f} nm/px')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 0]
logp = np.log1p(power)
im = ax.imshow(logp, cmap='viridis')
if fy_cyc_px > 0:
    off = fy_cyc_px * out_rows
    ax.axhline(cr + off, color='r', ls='--', lw=0.8)
    ax.axhline(cr - off, color='r', ls='--', lw=0.8)
ax.set_title(f'FFT log-power\nscan-line f={fy_cyc_px:.4f} cyc/px (red)')
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 1]
if amp_map is not None:
    im = ax.imshow(amp_map, cmap='inferno')
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'Reflection amplitude map\nd={mapped_d:.2f} nm')
else:
    ax.text(0.5, 0.5, 'No reflection mapped', ha='center', va='center')
    ax.set_title('Amplitude map')

ax = axes[1, 2]
if domain_mask is not None:
    ax.imshow(square_clean, cmap='gray', aspect='equal')
    ov = np.zeros((*domain_mask.shape, 4))
    ov[domain_mask.astype(bool)] = [0, 1, 0, 0.4]
    ax.imshow(ov)
    ax.set_title(f'Ordered domain (frac={domain_fraction:.3f})\nvalid={valid}')
else:
    ax.text(0.5, 0.5, reason, ha='center', va='center', wrap=True)
    ax.set_title('Domain / decision')

for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---------------------------------------------------------------
# 8. Save arrays
# ---------------------------------------------------------------
saved = {}
np.save('leveled_height_nm.npy', lev.astype(np.float32))
saved['leveled_height_nm.npy'] = {
    'description': 'Leveled height residual (nm), original grid, plane+row-median+destripe removed',
    'shape': list(lev.shape), 'dtype': 'float32'}
np.save('square_leveled_nm.npy', square_clean.astype(np.float32))
saved['square_leveled_nm.npy'] = {
    'description': 'Square-pixel leveled height (nm) with scan-line freq masked',
    'shape': list(square_clean.shape), 'dtype': 'float32'}
if amp_map is not None:
    np.save('reflection_amplitude_map.npy', np.asarray(amp_map, dtype=np.float32))
    saved['reflection_amplitude_map.npy'] = {
        'description': f'Ordered-domain amplitude map for reflection d={mapped_d} nm',
        'shape': list(np.asarray(amp_map).shape), 'dtype': 'float32'}
if domain_mask is not None:
    dm = np.asarray(domain_mask).astype(np.int32)
    np.save('domain_mask.npy', dm)
    saved['domain_mask.npy'] = {
        'description': 'Null-gated ordered-domain binary mask (1=domain,0=bulk)',
        'shape': list(dm.shape), 'dtype': 'int32'}

# ---------------------------------------------------------------
# 9. Results JSON
# ---------------------------------------------------------------
extracted = {
    'self_assembly_period_nm': (float(mapped_d) if valid else None),
    'reflection_significance_sigma': best_sigma,
    'spot_snr_domain': (float(spot_dom) if spot_dom is not None else None),
    'spot_snr_bulk': (float(spot_bulk) if spot_bulk is not None else None),
    'domain_fraction': (float(domain_fraction) if (valid and domain_fraction is not None) else None),
    'corrugation_height_nm_rms': float(corrugation_rms),
    'residual_roughness_Rq_nm': float(Rq),
    'scanline_frequency_cyc_px': float(fy_cyc_px),
    'scanline_period_nm': (float(scanline_period_nm) if scanline_period_nm else None),
    'scanline_excluded': True,
    'valid_reflection': bool(valid),
    'decision': reason,
}

quality = {
    'rms_before_leveling_nm': rms_before,
    'rms_after_plane_nm': rms_after_plane,
    'rms_after_full_leveling_nm': rms_after,
    'nm_per_px_square': nm_per_px,
    'nm_per_px_x_orig': nm_per_px_x,
    'nm_per_px_y_orig': nm_per_px_y,
    'null_threshold': (float(null_thr) if null_thr is not None else None),
    'n_reflections_found': len(reflections),
}

results = {
    'analysis_type': 'AFM peptide self-assembly: physical-height mapping, plane+row-median+destripe leveling, square resampling, scan-line FFT masking, targeted Fourier reflection mapping (3-10 nm band)',
    'extracted_features': extracted,
    'quality_metrics': quality,
    'summary': ('%s; period=%s nm. %s' % (
        reason,
        (f'{mapped_d:.2f}' if valid else 'N/A'),
        ' '.join(summary_notes))),
    'saved_arrays': saved,
}

print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results))
