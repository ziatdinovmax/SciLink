import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.log_blob import log_blob_detect

import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops

# ----------------------------------------------------------------------
# 1. Load image and metadata
# ----------------------------------------------------------------------
image = np.load('data.npy')

# Handle possible multi-channel input
if image.ndim == 3:
    # not RGB necessarily; use first channel for analysis
    analysis_img = image[:, :, 0].astype(np.float32)
else:
    analysis_img = image.astype(np.float32)

metadata = None
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    metadata = None

# ----------------------------------------------------------------------
# 2. Resolve pixel size
# ----------------------------------------------------------------------
px = None
if metadata is not None:
    try:
        px = resolve_pixel_size_nm(metadata, analysis_img.shape)
    except Exception:
        px = None

if px is not None and px.get('x') is not None:
    pixel_size_nm = float(px['x'])
    px_source = px.get('source', 'metadata')
    calibrated = True
else:
    pixel_size_nm = None
    px_source = 'unavailable'
    calibrated = False

unit = 'nm' if calibrated else 'px'

# ----------------------------------------------------------------------
# 3. PRIMARY DETECTION: scale-space LoG blob detection keyed to ~80 nm
#    dark spherical Au nanoparticles.
# ----------------------------------------------------------------------
OBJECT_DIAMETER_NM = 80.0

# threshold_rel tuned UP from the default to separate abutting cores while
# keeping one mark per visible dark sphere. Documented in summary.
THRESHOLD_REL = 0.25
OVERLAP = 0.4  # lower overlap helps split touching blobs

log_kwargs = dict(
    polarity='dark',
    threshold_rel=THRESHOLD_REL,
    overlap=OVERLAP,
    num_sigma=12,
    exclude_border=False,
)

if calibrated:
    log_res = log_blob_detect(analysis_img, object_diameter_nm=OBJECT_DIAMETER_NM,
                              pixel_size_nm=pixel_size_nm, **log_kwargs)
else:
    # Without calibration, provide a sigma range in pixels around an
    # assumed core size. Fall back to a broad range.
    # Assume ~80 px equivalent scale if uncalibrated (documented).
    log_res = log_blob_detect(analysis_img, min_sigma=8, max_sigma=40,
                              **log_kwargs)

log_objects = log_res.get('objects', []) or []
log_note = log_res.get('note', '')

# ----------------------------------------------------------------------
# 4. De-duplicate LoG detections within ~one radius (concentric fringe
#    contrast can produce multiple marks on one sphere).
# ----------------------------------------------------------------------
def dedup_objects(objs, pixsz):
    """Merge detections whose centers are closer than ~one radius."""
    if len(objs) == 0:
        return objs
    pts = np.array([[o['cy'], o['cx']] for o in objs], dtype=float)
    # diameters in pixels
    if pixsz is not None:
        diam_px = np.array([o['diameter_nm'] / pixsz for o in objs])
    else:
        diam_px = np.array([o['diameter_nm'] for o in objs])  # already px
    order = np.argsort(-diam_px)  # keep larger first
    kept = []
    kept_pts = []
    for idx in order:
        p = pts[idx]
        r = diam_px[idx] / 2.0
        merge = False
        for kp, kr in kept_pts:
            d = np.hypot(p[0] - kp[0], p[1] - kp[1])
            if d < max(r, kr) * 1.0:  # within ~one radius
                merge = True
                break
        if not merge:
            kept.append(objs[idx])
            kept_pts.append((p, r))
    return kept

log_objects_dd = dedup_objects(log_objects, pixel_size_nm)

# ----------------------------------------------------------------------
# 5. Watershed fallback for dense/touching-sphere clusters that LoG merges.
#    Distance-transform watershed on a dark-object mask.
# ----------------------------------------------------------------------
# Build dark-object mask via global Otsu (dark spheres -> below threshold)
thr = threshold_otsu(analysis_img)
dark_mask = analysis_img < thr
# clean up
dark_mask = ndi.binary_opening(dark_mask, structure=np.ones((3, 3)))
dark_mask = ndi.binary_fill_holes(dark_mask)

# Estimate particle radius in pixels
if calibrated:
    est_radius_px = (OBJECT_DIAMETER_NM / pixel_size_nm) / 2.0
else:
    # from median LoG diameter (px) if available
    if len(log_objects_dd) > 0:
        med_d = np.median([o['diameter_nm'] for o in log_objects_dd])
        est_radius_px = med_d / 2.0
    else:
        est_radius_px = 20.0

min_dist = max(3, int(round(est_radius_px)))

distance = ndi.distance_transform_edt(dark_mask)
try:
    peak_coords = peak_local_max(distance, min_distance=min_dist,
                                 labels=dark_mask, exclude_border=False)
except Exception:
    peak_coords = np.empty((0, 2), dtype=int)

markers = np.zeros(distance.shape, dtype=np.int32)
for i, (yy, xx) in enumerate(peak_coords):
    markers[yy, xx] = i + 1
markers_lab, _ = ndi.label(markers)
ws_labels = watershed(-distance, markers_lab, mask=dark_mask)

ws_props = regionprops(ws_labels, intensity_image=analysis_img)

# ----------------------------------------------------------------------
# Choose the primary result: prefer LoG (per-plan primary detection).
# Use watershed as documented cross-check / fallback.
# ----------------------------------------------------------------------
H, W = analysis_img.shape
border_margin = max(2, int(round(est_radius_px * 0.5)))

def is_interior(cy, cx, r):
    return (cx - r > 0) and (cy - r > 0) and (cx + r < W) and (cy + r < H)

# Build per-particle records from LoG (primary)
particles = []
for o in log_objects_dd:
    cy, cx = o['cy'], o['cx']
    d_nm = o['diameter_nm']
    if calibrated:
        r_px = (d_nm / pixel_size_nm) / 2.0
        d_report = d_nm
    else:
        r_px = d_nm / 2.0
        d_report = d_nm  # in px
    border = bool(o.get('border', not is_interior(cy, cx, r_px)))
    interior = is_interior(cy, cx, r_px) and not border
    particles.append({
        'cy': float(cy), 'cx': float(cx),
        'diameter': float(d_report),
        'interior': bool(interior),
    })

n_detected = len(particles)
interior_particles = [p for p in particles if p['interior']]
n_interior = len(interior_particles)

# Size distribution from interior particles
interior_diams = np.array([p['diameter'] for p in interior_particles], dtype=float)
all_diams = np.array([p['diameter'] for p in particles], dtype=float)

if interior_diams.size > 0:
    diam_mean = float(np.mean(interior_diams))
    diam_median = float(np.median(interior_diams))
    diam_std = float(np.std(interior_diams))
else:
    diam_mean = diam_median = diam_std = None

# Histogram
if interior_diams.size > 0:
    hist_counts, hist_edges = np.histogram(interior_diams, bins=20)
    hist_counts = hist_counts.tolist()
    hist_edges = hist_edges.tolist()
else:
    hist_counts, hist_edges = [], []

# ----------------------------------------------------------------------
# Packing / coverage note
# ----------------------------------------------------------------------
dark_area_frac = float(dark_mask.sum()) / float(dark_mask.size)
if calibrated:
    single_area_px = np.pi * est_radius_px ** 2
    est_particles_from_area = dark_area_frac * dark_mask.size / single_area_px
else:
    est_particles_from_area = None

if dark_area_frac > 0.4:
    packing_note = ('Dense monolayer regime: dark-object coverage {:.2f} indicates '
                    'closely-packed / touching spheres; LoG scale-space maxima used '
                    'to resolve individual cores.'.format(dark_area_frac))
else:
    packing_note = ('Coverage {:.2f}; particles moderately separated.'.format(dark_area_frac))

# ----------------------------------------------------------------------
# Save primary detection arrays
# ----------------------------------------------------------------------
# Build a label map from LoG detections (draw filled disks)
yy, xx = np.ogrid[:H, :W]
log_label_map = np.zeros((H, W), dtype=np.int32)
for i, p in enumerate(particles):
    if calibrated:
        r_px = (p['diameter'] / pixel_size_nm) / 2.0
    else:
        r_px = p['diameter'] / 2.0
    r_px = max(1.0, r_px)
    mask = (yy - p['cy']) ** 2 + (xx - p['cx']) ** 2 <= r_px ** 2
    log_label_map[mask] = i + 1

np.save('analysis_labels.npy', log_label_map)
np.save('watershed_labels.npy', ws_labels.astype(np.int32))
pos_arr = np.array([[p['cy'], p['cx'], p['diameter']] for p in particles], dtype=float) \
    if len(particles) > 0 else np.empty((0, 3))
np.save('particle_positions.npy', pos_arr)

# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

vmin, vmax = np.percentile(analysis_img, [1, 99])

# Panel 1: original
ax = axes[0, 0]
ax.imshow(analysis_img, cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title('Original image')
ax.axis('off')

# Panel 2: LoG detection overlay on raw image
ax = axes[0, 1]
ax.imshow(analysis_img, cmap='gray', vmin=vmin, vmax=vmax)
for p in particles:
    if calibrated:
        r_px = (p['diameter'] / pixel_size_nm) / 2.0
    else:
        r_px = p['diameter'] / 2.0
    color = 'lime' if p['interior'] else 'red'
    ax.add_patch(Circle((p['cx'], p['cy']), r_px, fill=False,
                        edgecolor=color, linewidth=0.8))
ax.set_title('LoG detection overlay (green=interior, red=edge)\n'
             'N_detected={}, N_interior={}'.format(n_detected, n_interior))
ax.axis('off')

# Panel 3: watershed segmentation cross-check
ax = axes[1, 0]
from skimage.color import label2rgb
ws_overlay = label2rgb(ws_labels, image=(analysis_img - vmin) / (vmax - vmin + 1e-9),
                       bg_label=0, alpha=0.4)
ax.imshow(np.clip(ws_overlay, 0, 1))
ax.set_title('Watershed fallback (dark-mask distance transform)\n'
             'n_regions={}'.format(len(ws_props)))
ax.axis('off')

# Panel 4: size distribution histogram
ax = axes[1, 1]
if interior_diams.size > 0:
    ax.hist(interior_diams, bins=20, color='steelblue', edgecolor='k', alpha=0.8)
    ax.axvline(diam_mean, color='red', linestyle='--',
               label='mean={:.1f} {}'.format(diam_mean, unit))
    ax.axvline(diam_median, color='green', linestyle=':',
               label='median={:.1f} {}'.format(diam_median, unit))
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No interior particles', ha='center', va='center')
ax.set_xlabel('Diameter ({})'.format(unit))
ax.set_ylabel('Count')
ax.set_title('Size distribution (interior particles)')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ----------------------------------------------------------------------
# Results JSON
# ----------------------------------------------------------------------
summary_bits = []
summary_bits.append('LoG scale-space blob detection (log_blob_detect, polarity=dark) '
                    'keyed to ~80 nm Au spheres was used as the primary detector.')
summary_bits.append('threshold_rel tuned to {} and overlap={} to give one mark per '
                    'dark sphere and separate abutting neighbors.'.format(THRESHOLD_REL, OVERLAP))
summary_bits.append('Detections de-duplicated within ~one radius to suppress multiple '
                    'marks from internal fringe/thickness contrast.')
summary_bits.append('Distance-transform watershed (Otsu dark mask -> EDT -> peak_local_max '
                    'min_distance~radius -> watershed) computed as documented fallback for '
                    'merged touching clusters.')
if not calibrated:
    summary_bits.append('Pixel size unavailable from metadata; diameters reported in pixels '
                        'and sigma range set directly.')
if log_note:
    summary_bits.append('log_blob_detect note: ' + str(log_note))
summary_bits.append(packing_note)

results = {
    'analysis_type': 'Spherical Au nanoparticle detection and sizing via scale-space '
                     'LoG blob detection with watershed fallback for dense packing.',
    'extracted_features': {
        'pixel_size_nm': pixel_size_nm,
        'pixel_size_source': px_source,
        'calibrated': calibrated,
        'diameter_unit': unit,
        'object_diameter_target_nm': OBJECT_DIAMETER_NM,
        'N_detected': n_detected,
        'N_interior': n_interior,
        'diameter_mean': diam_mean,
        'diameter_median': diam_median,
        'diameter_std': diam_std,
        'histogram_counts': hist_counts,
        'histogram_bin_edges': hist_edges,
        'log_blob_diameter_median_reported': log_res.get('diameter_median_nm'),
        'watershed_n_regions': len(ws_props),
        'dark_object_coverage_fraction': dark_area_frac,
        'packing_note': packing_note,
    },
    'quality_metrics': {
        'polarity_used': log_res.get('polarity_used'),
        'annotation_masked': log_res.get('annotation_masked'),
        'n_raw_log_detections': len(log_objects),
        'n_after_dedup': len(log_objects_dd),
        'threshold_rel': THRESHOLD_REL,
        'overlap': OVERLAP,
        'estimated_radius_px': float(est_radius_px),
    },
    'summary': ' '.join(summary_bits),
    'saved_arrays': {
        'analysis_labels.npy': {
            'description': 'Integer label map of LoG-detected particles (filled disks), '
                           'background=0, particles labeled 1..N.',
            'shape': list(log_label_map.shape),
            'dtype': str(log_label_map.dtype),
        },
        'watershed_labels.npy': {
            'description': 'Watershed segmentation label map (fallback for dense/touching '
                           'clusters), background=0.',
            'shape': list(ws_labels.shape),
            'dtype': 'int32',
        },
        'particle_positions.npy': {
            'description': 'Per-particle array [cy, cx, diameter] (diameter in {}); '
                           'rows = N_detected.'.format(unit),
            'shape': list(pos_arr.shape),
            'dtype': str(pos_arr.dtype),
        },
    },
}

print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results))
