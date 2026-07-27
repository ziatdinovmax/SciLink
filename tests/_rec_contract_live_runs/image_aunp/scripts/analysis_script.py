import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.log_blob import log_blob_detect

import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import measure

# ----------------------------------------------------------------------
# 1. Load image + metadata
# ----------------------------------------------------------------------
image = np.load('data.npy')
notes = []

# Handle possible multi-channel input
if image.ndim == 3:
    # not assuming RGB; use first channel for detection
    notes.append(f'Input had shape {image.shape}; using channel 0 for detection.')
    work = image[:, :, 0].astype(np.float32)
else:
    work = image.astype(np.float32)

metadata = None
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    notes.append(f'metadata.json not readable ({e}); pixel size may be unavailable.')

# ----------------------------------------------------------------------
# 2. Resolve pixel size (authoritative from metadata)
# ----------------------------------------------------------------------
px = resolve_pixel_size_nm(metadata, work.shape) if metadata is not None else None
if px is None:
    raise ValueError('pixel size unavailable; cannot convert to nm')
pixel_size_nm = float(px['x'])
notes.append(f"pixel_size_nm resolved = {pixel_size_nm:.5f} (source={px.get('source')}), "
             f"y={px.get('y'):.5f}")

OBJECT_DIAMETER_NM = 80.0
radius_px = (OBJECT_DIAMETER_NM / pixel_size_nm) / 2.0

# ----------------------------------------------------------------------
# 3. LoG blob detection (registered tool), dark polarity
#    Tune threshold_rel FROM THE OVERLAY: RAISE if speckle/false cores
#    appear, LOWER if visible spheres are unmarked. We drive the choice
#    with a count/contrast-based ground-truth check (a proxy for visual
#    overlay inspection) rather than a packing-density heuristic.
# ----------------------------------------------------------------------
POLARITY = 'dark'

def run_log(tr):
    return log_blob_detect(
        work,
        object_diameter_nm=OBJECT_DIAMETER_NM,
        pixel_size_nm=pixel_size_nm,
        polarity=POLARITY,
        threshold_rel=tr,
        overlap=0.5,
        exclude_border=False,
        mask_annotations=True,
    )

# Build a robust dark-region reference used for the overlay-style check:
# a real dark sphere centroid should sit on a genuinely dark pixel
# (well below background), whereas speckle/false cores tend to land on
# near-background intensity.
med = np.median(work)
mad = np.median(np.abs(work - med)) + 1e-9
std_robust = 1.4826 * mad
dark_core_level = med - 0.5 * std_robust  # below this = clearly dark

def overlay_inspect(res):
    """Emulate overlay inspection: score how many detected cores are
    truly dark (real spheres) vs. sitting on background (speckle/false).
    Returns (n_total, n_real_dark, frac_false)."""
    objs = res.get('objects', []) or []
    n_total = len(objs)
    if n_total == 0:
        return 0, 0, 0.0
    n_real = 0
    for o in objs:
        iy, ix = int(round(o['cy'])), int(round(o['cx']))
        if 0 <= iy < work.shape[0] and 0 <= ix < work.shape[1]:
            # sample a small neighborhood mean to be robust to a single pixel
            y0 = max(0, iy - 1); y1 = min(work.shape[0], iy + 2)
            x0 = max(0, ix - 1); x1 = min(work.shape[1], ix + 2)
            local = float(np.mean(work[y0:y1, x0:x1]))
            if local <= dark_core_level:
                n_real += 1
    frac_false = 1.0 - (n_real / n_total)
    return n_total, n_real, frac_false

# Sweep a ladder of thresholds and pick the one whose overlay is clean:
# few false (speckle) cores while still marking visible dark spheres.
# This is the RAISE-if-speckle / LOWER-if-missed rule made concrete.
threshold_ladder = [0.05, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
FALSE_FRAC_MAX = 0.25  # if more than this fraction of cores are on background -> speckle, RAISE

sweep = []
for tr in threshold_ladder:
    r = run_log(tr)
    n_total, n_real, frac_false = overlay_inspect(r)
    sweep.append((tr, r, n_total, n_real, frac_false))
    notes.append(f'overlay-check thr_rel={tr}: n={n_total}, real_dark={n_real}, '
                 f'false_frac={frac_false:.2f}')

# Selection logic mirroring overlay inspection:
#  - reject thresholds whose overlay is speckle-dominated (false_frac too high) -> RAISE
#  - among acceptable thresholds, take the LOWEST (so faint visible spheres
#    are not left unmarked), maximizing real detections.
acceptable = [s for s in sweep if s[4] <= FALSE_FRAC_MAX and s[3] >= 1]
if acceptable:
    # lowest threshold that is still clean -> catches faint spheres without speckle
    chosen = min(acceptable, key=lambda s: s[0])
    notes.append(f'Overlay inspection: chose LOWEST clean threshold_rel={chosen[0]} '
                 f'(false_frac={chosen[4]:.2f}) to avoid leaving faint spheres unmarked.')
else:
    # everything looked speckle-heavy -> RAISE to the highest threshold (cleanest cores)
    chosen = max(sweep, key=lambda s: s[0])
    notes.append(f'Overlay inspection: all thresholds showed speckle; RAISED to '
                 f'threshold_rel={chosen[0]} for cleanest cores.')

threshold_rel, res, _, _, _ = chosen

objects = res.get('objects', []) or []
n_detected = res.get('n_detected', len(objects))
polarity_used = res.get('polarity_used', POLARITY)
notes.append(f'Final threshold_rel={threshold_rel}, n_detected={n_detected}, polarity_used={polarity_used}')

# ----------------------------------------------------------------------
# 4. De-duplicate detections within ~one radius
# ----------------------------------------------------------------------
cents = np.array([[o['cy'], o['cx']] for o in objects], dtype=float) if objects else np.zeros((0, 2))
diams = np.array([o.get('diameter_nm', np.nan) for o in objects], dtype=float) if objects else np.zeros((0,))

keep = np.ones(len(cents), dtype=bool)
if len(cents) > 1:
    from scipy.spatial import cKDTree
    tree = cKDTree(cents)
    pairs = tree.query_pairs(r=radius_px)
    for i, j in pairs:
        if keep[i] and keep[j]:
            # drop the smaller-diameter / later one
            if np.nan_to_num(diams[i]) >= np.nan_to_num(diams[j]):
                keep[j] = False
            else:
                keep[i] = False
cents = cents[keep]
diams = diams[keep]
objects = [o for o, k in zip(objects, keep) if k]
notes.append(f'After de-duplication within radius {radius_px:.1f}px: {len(objects)} objects')

# ----------------------------------------------------------------------
# 5. Build dark-region mask (foreground for the spheres).
# ----------------------------------------------------------------------
dark_mask = work < (med - 0.5 * std_robust)
# Morphological cleanup
dark_mask = ndi.binary_opening(dark_mask, structure=np.ones((3, 3)))
dark_mask = ndi.binary_closing(dark_mask, structure=np.ones((3, 3)))

# ----------------------------------------------------------------------
# 5a. CONDITIONAL watershed decision.
#     Only split with a distance-transform watershed IF visible touching
#     spheres are still merged into single detections. We detect merging
#     by comparing connected dark-mask components against detections: a
#     component whose area spans clearly more than one particle but holds
#     <= 1 detection indicates merged touching cores.
# ----------------------------------------------------------------------
single_area = np.pi * radius_px ** 2
merge_detected = False
n_merged_components = 0

lbl_cc, n_cc = ndi.label(dark_mask)
if n_cc > 0 and len(cents) > 0:
    # map detections to connected components
    det_component_counts = np.zeros(n_cc + 1, dtype=int)
    for (cy, cx) in cents:
        iy, ix = int(round(cy)), int(round(cx))
        if 0 <= iy < work.shape[0] and 0 <= ix < work.shape[1]:
            c = lbl_cc[iy, ix]
            if c > 0:
                det_component_counts[c] += 1
    comp_areas = ndi.sum(np.ones_like(lbl_cc), lbl_cc, index=np.arange(1, n_cc + 1))
    for ci in range(1, n_cc + 1):
        area = comp_areas[ci - 1]
        # component large enough to hold >~1.5 particles but with <=1 detection
        if area >= 1.5 * single_area and det_component_counts[ci] <= 1:
            n_merged_components += 1
    if n_merged_components > 0:
        merge_detected = True

notes.append(f'Merge check: {n_merged_components} dark components look like merged '
             f'touching spheres (single-particle area ~{single_area:.0f}px). '
             f'Watershed {"WILL" if merge_detected else "will NOT"} be applied.')

# ----------------------------------------------------------------------
# 5b. Distance-transform watershed (ONLY if merging detected).
#     Markers come from peak_local_max on the distance transform with
#     min_distance ~ particle radius (as specified in the plan).
# ----------------------------------------------------------------------
labels = np.zeros(work.shape, dtype=np.int32)
n_labels = 0
watershed_applied = False

if merge_detected and dark_mask.any():
    distance = ndi.distance_transform_edt(dark_mask)
    coords = peak_local_max(
        distance,
        min_distance=max(1, int(round(radius_px))),
        labels=dark_mask,
    )
    markers = np.zeros(work.shape, dtype=np.int32)
    for idx, (yy, xx) in enumerate(coords, start=1):
        markers[yy, xx] = idx
    markers_lbl, _ = ndi.label(markers)
    labels = watershed(-distance, markers_lbl, mask=dark_mask)
    n_labels = int(labels.max())
    watershed_applied = True
    notes.append(f'Distance-transform watershed applied (peak_local_max '
                 f'min_distance={int(round(radius_px))}px) -> {n_labels} regions.')
else:
    notes.append('Watershed skipped: no merged touching spheres detected; '
                 'using LoG detections directly for sizing.')

# ----------------------------------------------------------------------
# 6. Per-particle diameter.
#    If watershed ran, size from split labels; otherwise (plan-conditional
#    path not triggered) size directly from the LoG detections.
# ----------------------------------------------------------------------
per_diam_nm = []
per_cent = []
if watershed_applied and n_labels > 0:
    props = measure.regionprops(labels)
    min_area_px = np.pi * (0.3 * radius_px) ** 2  # reject tiny fragments
    for p in props:
        if p.area < min_area_px:
            continue
        d_px = p.equivalent_diameter
        d_nm = d_px * pixel_size_nm
        # size filter: keep within [0.4x, 4x] object size
        if OBJECT_DIAMETER_NM * 0.4 <= d_nm <= OBJECT_DIAMETER_NM * 4.0:
            per_diam_nm.append(float(d_nm))
            per_cent.append([float(p.centroid[0]), float(p.centroid[1])])

if len(per_diam_nm) == 0:
    # use LoG-derived diameters/centroids (watershed not applied or gave no valid regions)
    per_diam_nm = [float(d) for d in diams if np.isfinite(d)]
    per_cent = [[float(c[0]), float(c[1])] for c in cents]
    if watershed_applied:
        notes.append('Watershed gave no valid regions; used LoG diameters/centroids.')
    else:
        notes.append('Used LoG diameters/centroids directly (no merging -> no watershed).')

per_diam_nm = np.array(per_diam_nm, dtype=float)
per_cent = np.array(per_cent, dtype=float) if len(per_cent) else np.zeros((0, 2))

n_final = len(per_diam_nm)
n_interior = res.get('n_interior', None)

diam_median = float(np.median(per_diam_nm)) if n_final else None
diam_mean = float(np.mean(per_diam_nm)) if n_final else None
diam_std = float(np.std(per_diam_nm)) if n_final else None

# histogram
if n_final:
    hist_counts, hist_edges = np.histogram(per_diam_nm,
                                           bins=min(20, max(5, n_final // 3 + 1)))
    hist_counts = hist_counts.tolist()
    hist_edges = hist_edges.tolist()
else:
    hist_counts, hist_edges = [], []

# ----------------------------------------------------------------------
# 7. Save arrays
# ----------------------------------------------------------------------
np.save('analysis_labels.npy', labels.astype(np.int32))
np.save('particle_centroids.npy', per_cent)
np.save('particle_diameters_nm.npy', per_diam_nm)

# ----------------------------------------------------------------------
# 8. Visualization
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

vmin, vmax = np.percentile(work, [1, 99])
axes[0, 0].imshow(work, cmap='gray', vmin=vmin, vmax=vmax)
axes[0, 0].set_title('Original image (channel 0)')
axes[0, 0].axis('off')

# Segmentation overlay
ax = axes[0, 1]
ax.imshow(work, cmap='gray', vmin=vmin, vmax=vmax)
if n_labels > 0:
    overlay = np.ma.masked_where(labels == 0, labels)
    ax.imshow(overlay, cmap='nipy_spectral', alpha=0.4)
    for c in measure.find_contours(labels > 0, 0.5):
        ax.plot(c[:, 1], c[:, 0], color='red', linewidth=0.6)
for (cy, cx) in per_cent:
    ax.plot(cx, cy, 'c+', markersize=6)
ax.set_title(f'Segmentation overlay (n={n_final}, watershed={watershed_applied})')
ax.axis('off')

# Detected blobs as circles
ax = axes[1, 0]
ax.imshow(work, cmap='gray', vmin=vmin, vmax=vmax)
for o in objects:
    r = (o.get('diameter_nm', OBJECT_DIAMETER_NM) / pixel_size_nm) / 2.0
    ax.add_patch(Circle((o['cx'], o['cy']), r, fill=False, color='yellow', linewidth=0.8))
ax.set_title(f'LoG detections (thr_rel={threshold_rel}, polarity={polarity_used})')
ax.axis('off')

# Histogram
ax = axes[1, 1]
if n_final:
    ax.hist(per_diam_nm, bins=min(20, max(5, n_final // 3 + 1)),
            color='steelblue', edgecolor='k')
    if diam_median is not None:
        ax.axvline(diam_median, color='red', linestyle='--',
                   label=f'median={diam_median:.1f} nm')
    ax.legend()
ax.set_xlabel('diameter (nm)')
ax.set_ylabel('count')
ax.set_title('Diameter distribution')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close(fig)

# ----------------------------------------------------------------------
# 9. Results JSON
# ----------------------------------------------------------------------
results = {
    'analysis_type': 'Scale-matched LoG band-pass blob detection (dark polarity) sized to ~80 nm '
                     'particles, threshold_rel tuned by overlay inspection, de-duplicated within '
                     'one radius, conditional distance-transform watershed (peak_local_max markers, '
                     'min_distance ~ radius) to split touching cores, then per-particle sizing.',
    'extracted_features': {
        'particle_count_n_detected': int(n_detected),
        'particle_count_final': int(n_final),
        'n_interior': (int(n_interior) if n_interior is not None else None),
        'diameter_median_nm': diam_median,
        'diameter_mean_nm': diam_mean,
        'diameter_std_nm': diam_std,
        'diameter_std_nm_err': diam_std,
        'diameter_histogram_counts': hist_counts,
        'diameter_histogram_edges_nm': hist_edges,
        'per_particle_diameter_nm': per_diam_nm.tolist(),
        'per_particle_centroid_yx': per_cent.tolist(),
        'polarity_used': polarity_used,
        'detector_used': 'log_blob_detect + conditional distance-transform watershed',
        'threshold_rel': float(threshold_rel),
        'watershed_applied': bool(watershed_applied),
        'watershed_min_distance_px': float(radius_px),
        'n_merged_components_detected': int(n_merged_components),
    },
    'quality_metrics': {
        'pixel_size_nm': pixel_size_nm,
        'radius_px_expected': float(radius_px),
        'n_watershed_regions': int(n_labels),
        'dark_mask_fraction': float(dark_mask.mean()),
    },
    'summary': 'Detected {} dark ~80 nm spheres (median diameter {} nm); '.format(
        n_final, f'{diam_median:.1f}' if diam_median is not None else 'NA')
        + ' | '.join(notes),
    'saved_arrays': {
        'analysis_labels.npy': {
            'description': f'Integer watershed label map, {n_labels} particles labeled 1..N, background=0 (all-zero if watershed not applied)',
            'shape': list(labels.shape),
            'dtype': str(labels.dtype),
        },
        'particle_centroids.npy': {
            'description': 'Per-particle centroid positions (row, col) in pixels',
            'shape': list(per_cent.shape),
            'dtype': str(per_cent.dtype),
        },
        'particle_diameters_nm.npy': {
            'description': 'Per-particle equivalent diameters in nm',
            'shape': list(per_diam_nm.shape),
            'dtype': str(per_diam_nm.dtype),
        },
    },
}
print(f'IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}')
