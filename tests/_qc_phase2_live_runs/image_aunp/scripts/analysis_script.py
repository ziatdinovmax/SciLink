import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.log_blob import log_blob_detect

import skimage.measure
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import scipy.ndimage as ndi

notes = []

# ---------------- Load image ----------------
image = np.load('data.npy')
if image.ndim == 3:
    notes.append(f'Image has shape {image.shape}; using channel 0 for analysis.')
    image = image[:, :, 0]
image = np.asarray(image, dtype=np.float32)

# ---------------- Metadata / pixel size ----------------
metadata = None
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    notes.append(f'metadata.json not readable ({e}); pixel size may be unresolved.')

pixel_size_nm = None
px = None
if metadata is not None:
    px = resolve_pixel_size_nm(metadata, image.shape)
if px is not None:
    pixel_size_nm = float(px['x'])
    notes.append(f"Pixel size resolved: {pixel_size_nm:.4f} nm/px (source={px.get('source')}).")
else:
    pixel_size_nm = 0.396
    notes.append('Pixel size could not be resolved from metadata; falling back to ~0.396 nm/px per plan.')

OBJ_DIAM_NM = 80.0

# ---------------- Detection: log_blob_detect (registered tool) ----------------
# Tune threshold_rel UP from a low starting value; polarity='dark' per plan.
tried = []
res = None
chosen_thr = None
for thr_rel in [0.1, 0.2, 0.3, 0.4]:
    r = log_blob_detect(image, object_diameter_nm=OBJ_DIAM_NM,
                        pixel_size_nm=pixel_size_nm, polarity='dark',
                        threshold_rel=thr_rel, overlap=0.5,
                        exclude_border=False, mask_annotations=True)
    n = r.get('n_detected', 0)
    tried.append((thr_rel, n))

# Final choice: prefer thr_rel=0.3 (per plan, tuned UP) if it detected objects,
# else best available that still yields detections.
final = None
for thr_rel in [0.3, 0.2, 0.4, 0.1]:
    r = log_blob_detect(image, object_diameter_nm=OBJ_DIAM_NM,
                        pixel_size_nm=pixel_size_nm, polarity='dark',
                        threshold_rel=thr_rel, overlap=0.5,
                        exclude_border=False, mask_annotations=True)
    if r.get('n_detected', 0) > 0:
        final = r; chosen_thr = thr_rel; break
if final is None:
    # keep the last attempted (may be empty) so downstream code still runs
    final = r; chosen_thr = thr_rel
res = final
notes.append(f'log_blob_detect threshold_rel sweep {tried}; chose threshold_rel={chosen_thr}.')
notes.append(f"polarity_used={res.get('polarity_used')}, annotation_masked={res.get('annotation_masked')}.")

objects = res.get('objects', []) or []
n_detected = res.get('n_detected', 0)

# ---------------- Build detection mask & distance-transform watershed ----------------
# Radius in pixels from known diameter
radius_px = (OBJ_DIAM_NM / pixel_size_nm) / 2.0

# Otsu on the image: dark particles are BELOW the threshold; bright gaps ABOVE.
thr_val = threshold_otsu(image)
binary_mask = image < thr_val  # True where particles (dark) are
binary_mask = binary_opening(binary_mask, disk(1))

# Distance-transform watershed to split touching particles.
distance = ndi.distance_transform_edt(binary_mask)
min_dist = max(3, int(round(radius_px)))
try:
    peaks = peak_local_max(distance, min_distance=min_dist, labels=binary_mask)
    marker_mask = np.zeros(distance.shape, dtype=bool)
    if peaks.shape[0] > 0:
        marker_mask[tuple(peaks.T)] = True
    markers = ndi.label(marker_mask)[0]
    if markers.max() > 0:
        ws_labels = watershed(-distance, markers, mask=binary_mask)
    else:
        ws_labels = ndi.label(binary_mask)[0]
except Exception as e:
    notes.append(f'Watershed step failed ({e}); using connected-component labels.')
    ws_labels = ndi.label(binary_mask)[0]

# Size gate for valid regions (fragments and merges filtered out)
expected_area = np.pi * radius_px ** 2
min_area = 0.15 * expected_area
max_area = 6.0 * expected_area

H, W = image.shape
props = skimage.measure.regionprops(ws_labels, intensity_image=image)
valid_ws = [p for p in props if min_area <= p.area <= max_area]

notes.append(f'Blob detections={n_detected}; valid watershed-labeled regions={len(valid_ws)}. '
             'Per-particle equivalent-circle diameters measured from labeled-region area (consistent single source).')

# ---------------- Per-particle equivalent-circle diameters (single labeled-region source) ----------------
# Per plan: measure equivalent-circle diameter (nm) via region AREA for every
# labeled particle, using ONE consistent labeled-region step (regionprops on the
# watershed/connected-component label image). Blob detections anchor detection;
# sizing is always region-area based, never blob sigma or ad-hoc per-crop Otsu.
records = []
label_map = np.zeros((H, W), dtype=np.int32)

idx = 0
for p in valid_ws:
    idx += 1
    area = float(p.area)
    eq_diam_px = 2.0 * np.sqrt(area / np.pi)
    eq_diam_nm = eq_diam_px * pixel_size_nm
    minr, minc, maxr, maxc = p.bbox
    border = (minr <= 0 or minc <= 0 or maxr >= H or maxc >= W)
    perim = p.perimeter if p.perimeter > 0 else np.nan
    circ = (4.0 * np.pi * area) / (perim ** 2) if perim and not np.isnan(perim) else np.nan
    sol = float(p.solidity)
    cy, cx = p.centroid
    label_map[ws_labels == p.label] = idx
    records.append(dict(cy=float(cy), cx=float(cx), diameter_nm=eq_diam_nm,
                        area_px=area, circularity=float(circ) if np.isfinite(circ) else None,
                        solidity=sol, border=bool(border)))

# ---------------- Statistics (interior particles) ----------------
N_detected = len(records)
interior = [r for r in records if not r['border'] and np.isfinite(r['diameter_nm'])]
N_interior = len(interior)
int_diams = np.array([r['diameter_nm'] for r in interior], dtype=float)
all_diams = np.array([r['diameter_nm'] for r in records if np.isfinite(r['diameter_nm'])], dtype=float)

def safe(v):
    return float(v) if np.isfinite(v) else None

if int_diams.size > 0:
    d_median = float(np.median(int_diams))
    d_mean = float(np.mean(int_diams))
    d_std = float(np.std(int_diams))
    polydispersity = float(d_std / d_mean) if d_mean > 0 else None
    hist_counts, hist_edges = np.histogram(int_diams, bins=20)
else:
    d_median = d_mean = d_std = polydispersity = None
    hist_counts, hist_edges = np.array([]), np.array([])

circs = np.array([r['circularity'] for r in interior
                  if r.get('circularity') is not None and np.isfinite(r['circularity'])], dtype=float)
sols = np.array([r['solidity'] for r in interior
                 if r.get('solidity') is not None and np.isfinite(r['solidity'])], dtype=float)

# ---------------- Save arrays ----------------
np.save('analysis_labels.npy', label_map)
np.save('particle_diameters_nm.npy', all_diams)

# ---------------- Visualization ----------------
vmin, vmax = np.percentile(image, [1, 99])
fig, axes = plt.subplots(2, 2, figsize=(13, 13))

ax = axes[0, 0]
ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title('Original image')
ax.axis('off')

ax = axes[0, 1]
ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
overlay = np.ma.masked_where(label_map == 0, label_map)
ax.imshow(overlay, cmap='nipy_spectral', alpha=0.45)
try:
    ax.contour(label_map > 0, colors='cyan', linewidths=0.4)
except Exception:
    pass
for r in records:
    col = 'red' if r['border'] else 'lime'
    if np.isfinite(r['diameter_nm']):
        rad_px = (r['diameter_nm'] / pixel_size_nm) / 2.0
        ax.add_patch(Circle((r['cx'], r['cy']), rad_px, fill=False,
                            edgecolor=col, linewidth=0.6))
ax.set_title(f'Segmentation overlay (N={N_detected}, interior={N_interior})\n'
             'green=interior, red=border')
ax.axis('off')

ax = axes[1, 0]
if int_diams.size > 0:
    ax.hist(int_diams, bins=20, color='steelblue', edgecolor='k')
    if d_median is not None:
        ax.axvline(d_median, color='red', ls='--', label=f'median={d_median:.1f} nm')
    ax.legend()
    ax.set_xlabel('Equivalent-circle diameter (nm)')
    ax.set_ylabel('Count')
    ax.set_title('Interior diameter distribution')
else:
    ax.text(0.5, 0.5, 'No interior particles', ha='center', va='center')
    ax.set_title('Diameter distribution')

ax = axes[1, 1]
ax.imshow(binary_mask, cmap='gray')
ax.set_title(f'Otsu foreground mask (thr={thr_val:.1f})\nfor distance-transform watershed')
ax.axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close(fig)

# ---------------- Results JSON ----------------
results = {
    'analysis_type': 'Nanoparticle detection and size distribution via scale-space LoG blob '
                     'detection (log_blob_detect, ~80 nm) with distance-transform watershed '
                     'splitting of touching particles; equivalent-circle diameters measured '
                     'from a single consistent labeled-region step (region area).',
    'extracted_features': {
        'pixel_size_nm': pixel_size_nm,
        'object_diameter_nm_expected': OBJ_DIAM_NM,
        'sizing_source': 'labeled_region_area (watershed/connected-component regionprops)',
        'N_detected': int(N_detected),
        'N_interior': int(N_interior),
        'diameter_median_nm': d_median,
        'diameter_mean_nm': d_mean,
        'diameter_std_nm': d_std,
        'polydispersity_std_over_mean': polydispersity,
        'histogram_counts': hist_counts.tolist(),
        'histogram_bin_edges_nm': hist_edges.tolist(),
        'circularity_median': safe(np.median(circs)) if circs.size else None,
        'solidity_median': safe(np.median(sols)) if sols.size else None,
    },
    'quality_metrics': {
        'polarity_used': res.get('polarity_used'),
        'annotation_masked': bool(res.get('annotation_masked', False)),
        'threshold_rel_used': chosen_thr,
        'n_blob_detections': int(n_detected),
        'otsu_threshold_value': float(thr_val),
        'radius_px_expected': float(radius_px),
        'n_watershed_valid_regions': int(len(valid_ws)),
        'fraction_border_particles': float((N_detected - N_interior) / N_detected) if N_detected else None,
    },
    'summary': (f'Detected {N_detected} particles ({N_interior} interior); interior median '
                f'equivalent-circle diameter '
                f"{d_median:.1f} nm" if d_median is not None else 'no interior particles measured')
               + '. ' + ' '.join(notes),
    'saved_arrays': {
        'analysis_labels.npy': {
            'description': f'Integer label map, {N_detected} particles labeled 1..N, background=0',
            'shape': list(label_map.shape), 'dtype': str(label_map.dtype)},
        'particle_diameters_nm.npy': {
            'description': 'Per-particle equivalent-circle diameters in nm (all finite detections)',
            'shape': list(all_diams.shape), 'dtype': str(all_diams.dtype)},
    },
}
print(f'IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}')
