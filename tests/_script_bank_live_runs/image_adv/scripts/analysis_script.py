import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.fourier_reflection import fourier_reflection_map

# ---------------------------------------------------------------
# 1) Load image
# ---------------------------------------------------------------
image = np.load("data.npy")
if image.ndim == 3:
    img = image[:, :, 0].astype(np.float32)
else:
    img = image.astype(np.float32)

H, W = img.shape

# ---------------------------------------------------------------
# 1b) Metadata / calibration (authoritative)
# ---------------------------------------------------------------
metadata = None
try:
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
except Exception:
    metadata = None

notes = []

# Per-axis pixel size via registered tool
px = None
if metadata is not None:
    try:
        px = resolve_pixel_size_nm(metadata, img.shape)
    except Exception:
        px = None

pixel_size_nm_x = None
pixel_size_nm_y = None
px_source = 'uncalibrated'
if px is not None:
    pixel_size_nm_x = float(px['x'])
    pixel_size_nm_y = float(px['y'])
    px_source = px.get('source', 'metadata')

# Also try to read raw FOV per axis from metadata for square-pixel confirmation
fov_x = fov_y = None
fov_units = None
if metadata is not None:
    md = metadata.get('system_info', metadata)
    ed = md.get('experimental_details', md)
    si = ed.get('spatial_info', ed) if isinstance(ed, dict) else {}
    if isinstance(si, dict):
        fov_x = si.get('field_of_view_x')
        fov_y = si.get('field_of_view_y')
        fov_units = si.get('field_of_view_units')
        # convert to nm if units known
        def to_nm(v, u):
            if v is None:
                return None
            v = float(v)
            if u is None:
                return v
            u = str(u).lower()
            if u in ('nm', 'nanometer', 'nanometers'):
                return v
            if u in ('um', 'micron', 'microns', 'micrometer', 'micrometers', '\u00b5m'):
                return v * 1000.0
            if u in ('a', 'ang', 'angstrom', 'angstroms', '\u00c5'):
                return v * 0.1
            if u in ('m', 'meter', 'meters'):
                return v * 1e9
            return v
        fov_x = to_nm(fov_x, fov_units)
        fov_y = to_nm(fov_y, fov_units)

calibrated = pixel_size_nm_x is not None and pixel_size_nm_x > 0

# ---------------------------------------------------------------
# 2) Per-row median subtraction (AFM baseline leveling)
#    Features are 2D dots -> row-median leveling is safe.
# ---------------------------------------------------------------
row_med = np.median(img, axis=1, keepdims=True)
leveled = img - row_med

# ---------------------------------------------------------------
# 3) Resample to square pixels if x/y pixel sizes differ
# ---------------------------------------------------------------
square = leveled
resampled = False
pixel_size_nm = pixel_size_nm_x
new_shape = (H, W)

if calibrated and pixel_size_nm_y is not None and pixel_size_nm_y > 0:
    ratio = pixel_size_nm_y / pixel_size_nm_x
    if abs(ratio - 1.0) > 1e-3:
        # choose target square pixel = smaller of the two (finer)
        target = min(pixel_size_nm_x, pixel_size_nm_y)
        # physical extent per axis
        ext_x = pixel_size_nm_x * W
        ext_y = pixel_size_nm_y * H
        new_W = int(round(ext_x / target))
        new_H = int(round(ext_y / target))
        zoom_y = new_H / H
        zoom_x = new_W / W
        square = ndimage.zoom(leveled, (zoom_y, zoom_x), order=1)
        new_shape = square.shape
        pixel_size_nm = target
        resampled = True
        notes.append(
            f"Anisotropic pixels (x={pixel_size_nm_x:.4f}, y={pixel_size_nm_y:.4f} nm/px); "
            f"resampled to square pixels at {target:.4f} nm/px -> shape {new_shape}."
        )
        # confirm nm/px == FOV / N_new
        if fov_x is not None:
            notes.append(f"Confirm: FOV_x/N_x_new = {fov_x/new_shape[1]:.4f} nm/px vs {target:.4f}.")
        if fov_y is not None:
            notes.append(f"Confirm: FOV_y/N_y_new = {fov_y/new_shape[0]:.4f} nm/px vs {target:.4f}.")
    else:
        notes.append(f"x/y pixel sizes equal ({pixel_size_nm_x:.4f} nm/px); no resampling needed.")
elif calibrated:
    notes.append(f"Single pixel size {pixel_size_nm_x:.4f} nm/px; assumed square.")
else:
    # Uncalibrated fallback: assume expected ~0.0294 nm/px per plan context
    pixel_size_nm = 0.0294
    px_source = 'assumed_default'
    notes.append("UNCALIBRATED: metadata pixel size unresolved; assumed 0.0294 nm/px (plan expectation).")

# ---------------------------------------------------------------
# 4) Fourier reflection map (registered tool)
#    Detect reflections, flag satellites, map strongest.
# ---------------------------------------------------------------
fr_params = {'d_range': (0.2, 2.0)}
try:
    fr = fourier_reflection_map(square, pixel_size_nm, params=fr_params)
except Exception as e:
    fr = {'note': f'fourier_reflection_map failed: {e}'}
    notes.append(f"fourier_reflection_map raised: {e}")

reflections = fr.get('reflections', []) or []
mapped_d_nm = fr.get('mapped_d_nm')
mapped_is_satellite = fr.get('mapped_is_satellite_candidate')
strongest_satellite_d_nm = fr.get('strongest_satellite_d_nm')
amplitude_map = fr.get('amplitude_map')
phase_map = fr.get('phase_map')
domain_mask = fr.get('domain_mask')
domain_fraction = fr.get('domain_fraction')
spot_snr_domain = fr.get('spot_snr_domain')
spot_snr_bulk = fr.get('spot_snr_bulk')
is_mapped_superstructure = fr.get('is_mapped_superstructure')
null_threshold = fr.get('null_threshold')

# If nothing cleared the floor, try lowering d_range / re-run once
if not reflections and 'note' in fr:
    notes.append(f"Initial reflection detection note: {fr.get('note')}")

# Reflection significance summary
reflections_summary = []
for r in reflections:
    reflections_summary.append({
        'd_nm': r.get('d_nm'),
        'sigma': r.get('sigma'),
        'is_satellite_candidate': r.get('is_satellite_candidate'),
        'integer_multiple_of': r.get('integer_multiple_of'),
    })

# Strongest reflection (highest sigma) spacing
strongest_d_nm = reflections_summary[0]['d_nm'] if reflections_summary else None
strongest_sigma = reflections_summary[0]['sigma'] if reflections_summary else None

# Compare to known mica basal spacing ~0.52 nm
KNOWN_MICA_NM = 0.52
mica_match = None
mica_match_d = None
if reflections_summary:
    # find reflection closest to known basal spacing
    best = min(reflections_summary, key=lambda r: abs((r['d_nm'] or 1e9) - KNOWN_MICA_NM))
    mica_match_d = best['d_nm']
    if mica_match_d is not None:
        mica_match = float(mica_match_d)

# Domain confirmation: spot_snr_domain vs spot_snr_bulk
domain_confirmed = None
if spot_snr_domain is not None and spot_snr_bulk is not None and spot_snr_bulk > 0:
    domain_confirmed = bool(spot_snr_domain > 3.0 * spot_snr_bulk)

# Band-limited amplitude homogeneity (based on periodic-band amplitude, NOT raw contrast)
amp_homogeneity = None
amp_cv = None
if amplitude_map is not None:
    amp = np.asarray(amplitude_map, dtype=float)
    finite = amp[np.isfinite(amp)]
    if finite.size > 0 and np.mean(finite) != 0:
        amp_cv = float(np.std(finite) / (np.mean(finite) + 1e-12))
        # lower CV -> more homogeneous ordered pattern
        amp_homogeneity = 'homogeneous' if amp_cv < 0.5 else ('moderate' if amp_cv < 1.0 else 'heterogeneous')

# ---------------------------------------------------------------
# 5) Visualization
# ---------------------------------------------------------------
n_panels = 4
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

vmin, vmax = np.percentile(img, [1, 99])
axes[0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title('Original image')
axes[0].axis('off')

lvmin, lvmax = np.percentile(square, [1, 99])
axes[1].imshow(square, cmap='gray', vmin=lvmin, vmax=lvmax)
title1 = 'Leveled (row-median)'
if resampled:
    title1 += ' + square-resampled'
axes[1].set_title(title1)
axes[1].axis('off')

if amplitude_map is not None:
    im2 = axes[2].imshow(np.asarray(amplitude_map, dtype=float), cmap='inferno')
    ttl = 'Band-limited amplitude map'
    if mapped_d_nm is not None:
        ttl += f'\nd={mapped_d_nm:.3f} nm'
    axes[2].set_title(ttl)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
else:
    axes[2].text(0.5, 0.5, 'No amplitude map\n(no resolvable reflection)', ha='center', va='center')
    axes[2].set_title('Amplitude map')
    axes[2].axis('off')

# Domain mask overlay or reflection sigma bar chart
if domain_mask is not None:
    axes[3].imshow(square, cmap='gray', vmin=lvmin, vmax=lvmax)
    dm = np.asarray(domain_mask, dtype=float)
    axes[3].imshow(np.ma.masked_where(dm == 0, dm), cmap='autumn', alpha=0.5)
    ttl = 'Ordered-domain mask (null-gated)'
    if domain_fraction is not None:
        ttl += f'\nfrac={domain_fraction:.2f}'
    axes[3].set_title(ttl)
    axes[3].axis('off')
elif reflections_summary:
    ds = [r['d_nm'] for r in reflections_summary]
    sg = [r['sigma'] for r in reflections_summary]
    axes[3].bar(range(len(ds)), sg, color='steelblue')
    axes[3].set_xticks(range(len(ds)))
    axes[3].set_xticklabels([f"{d:.3f}" for d in ds], rotation=45, ha='right')
    axes[3].set_ylabel('sigma')
    axes[3].set_xlabel('d (nm)')
    axes[3].set_title('Detected reflections')
else:
    axes[3].text(0.5, 0.5, 'No reflections detected', ha='center', va='center')
    axes[3].set_title('Reflections')
    axes[3].axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---------------------------------------------------------------
# 6) Save arrays
# ---------------------------------------------------------------
saved = {}
np.save('leveled_square_image.npy', square.astype(np.float32))
saved['leveled_square_image.npy'] = {
    'description': 'Row-median leveled image, resampled to square pixels if anisotropic.',
    'shape': list(square.shape),
    'dtype': 'float32',
}
if amplitude_map is not None:
    amp_arr = np.asarray(amplitude_map, dtype=np.float32)
    np.save('lattice_amplitude_map.npy', amp_arr)
    saved['lattice_amplitude_map.npy'] = {
        'description': f'Band-limited amplitude map at mapped reflection d={mapped_d_nm} nm.',
        'shape': list(amp_arr.shape),
        'dtype': 'float32',
    }
if domain_mask is not None:
    dm_arr = np.asarray(domain_mask).astype(np.uint8)
    np.save('ordered_domain_mask.npy', dm_arr)
    saved['ordered_domain_mask.npy'] = {
        'description': 'Null-gated ordered-domain binary mask (1=ordered domain).',
        'shape': list(dm_arr.shape),
        'dtype': 'uint8',
    }

# ---------------------------------------------------------------
# 7) Assemble JSON results
# ---------------------------------------------------------------
summary_parts = []
if calibrated:
    if resampled:
        summary_parts.append(f"Anisotropic pixels resampled to square {pixel_size_nm:.4f} nm/px (source: {px_source}).")
    else:
        summary_parts.append(f"Square pixels at {pixel_size_nm:.4f} nm/px (source: {px_source}).")
else:
    summary_parts.append(f"Uncalibrated; used assumed {pixel_size_nm:.4f} nm/px.")

if strongest_d_nm is not None:
    summary_parts.append(f"Strongest reflection d={strongest_d_nm:.3f} nm (sigma={strongest_sigma:.1f}).")
else:
    summary_parts.append("No resolvable reflection cleared the significance floor.")

if strongest_satellite_d_nm is not None:
    summary_parts.append(f"Satellite/superstructure at d={strongest_satellite_d_nm:.3f} nm.")
else:
    summary_parts.append("No resolvable superstructure/satellite.")

if mica_match is not None:
    summary_parts.append(f"Closest reflection to known mica basal ~0.52 nm: {mica_match:.3f} nm.")

if domain_confirmed is not None:
    summary_parts.append(
        f"Domain SNR check: spot_snr_domain={spot_snr_domain:.2f} vs bulk={spot_snr_bulk:.2f} -> "
        f"{'localized ordering confirmed' if domain_confirmed else 'not clearly localized'}."
    )
if amp_homogeneity is not None:
    summary_parts.append(f"Band amplitude homogeneity: {amp_homogeneity} (CV={amp_cv:.2f}).")

summary_parts.extend(notes)

results = {
    "analysis_type": "Row-median AFM leveling + square-pixel calibration + targeted Fourier reflection mapping (fourier_reflection_map) of mica lattice/adsorbate periodicity",
    "extracted_features": {
        "pixel_size_nm_x": pixel_size_nm_x,
        "pixel_size_nm_y": pixel_size_nm_y,
        "pixel_size_nm_used": float(pixel_size_nm) if pixel_size_nm is not None else None,
        "pixel_size_source": px_source,
        "pixels_equal_or_resampled": ('resampled' if resampled else ('equal' if calibrated else 'uncalibrated')),
        "resampled_shape": list(new_shape),
        "detected_reflections": reflections_summary,
        "strongest_reflection_d_nm": strongest_d_nm,
        "strongest_reflection_sigma": strongest_sigma,
        "mapped_d_nm": mapped_d_nm,
        "mapped_is_satellite_candidate": mapped_is_satellite,
        "strongest_satellite_d_nm": strongest_satellite_d_nm,
        "is_mapped_superstructure": is_mapped_superstructure,
        "known_mica_basal_spacing_nm": KNOWN_MICA_NM,
        "closest_reflection_to_mica_basal_nm": mica_match,
        "band_amplitude_cv": amp_cv,
        "band_amplitude_homogeneity": amp_homogeneity,
        "domain_fraction": domain_fraction,
        "spot_snr_domain": spot_snr_domain,
        "spot_snr_bulk": spot_snr_bulk,
        "localized_ordering_confirmed": domain_confirmed,
    },
    "quality_metrics": {
        "calibrated": bool(calibrated),
        "resampled_to_square": bool(resampled),
        "n_reflections_detected": len(reflections_summary),
        "null_threshold": null_threshold,
        "fourier_note": fr.get('note'),
    },
    "summary": ' '.join(summary_parts),
    "saved_arrays": saved,
}

print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}")
