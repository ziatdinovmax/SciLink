import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.log_blob import log_blob_detect

# ---------------------------------------------------------------
# 1) Load image
# ---------------------------------------------------------------
image = np.load("data.npy")
if image.ndim == 3:
    img = image[:, :, 0].astype(np.float32)
else:
    img = image.astype(np.float32)

# ---------------------------------------------------------------
# 1b) Resolve pixel size from metadata.json (authoritative)
# ---------------------------------------------------------------
px_source = 'uncalibrated'
pixel_size_nm = None
metadata = None
try:
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
except Exception:
    metadata = None

if metadata is not None:
    try:
        px = resolve_pixel_size_nm(metadata, img.shape)
    except Exception:
        px = None
    if px is not None:
        pixel_size_nm = float(px['x'])
        px_source = px.get('source', 'metadata')

calibrated = pixel_size_nm is not None and pixel_size_nm > 0
unit = 'nm' if calibrated else 'px'

# Known expected diameter ~12 nm (per plan)
expected_diameter_nm = 12.0

# ---------------------------------------------------------------
# 2) Scale-space LoG blob detection (registered tool)
#    Plan: log_blob_detect for dense/touching small dark cores.
#    threshold_rel tuned from overlay.
# ---------------------------------------------------------------
note_parts = []

# If uncalibrated, fall back to passthrough sigma range (px units).
if calibrated:
    def detect(tr):
        return log_blob_detect(
            img,
            object_diameter_nm=expected_diameter_nm,
            pixel_size_nm=pixel_size_nm,
            polarity='dark',
            threshold_rel=tr,
            overlap=0.5,
            exclude_border=False,
        )
else:
    # Rough px-based sigma range fallback; assume ~10 px diameter cores.
    def detect(tr):
        return log_blob_detect(
            img,
            min_sigma=2.0,
            max_sigma=8.0,
            num_sigma=10,
            polarity='dark',
            threshold_rel=tr,
            overlap=0.5,
            exclude_border=False,
        )
    note_parts.append('UNCALIBRATED: metadata.json missing or pixel size unresolved; sizes reported in pixels, sigma range guessed.')

# Tune threshold_rel: start moderate, adjust based on detection count sanity.
threshold_rel_used = 0.10
result = detect(threshold_rel_used)
n0 = result.get('n_detected', 0)

# If nothing detected, progressively lower threshold_rel to catch faint cores.
if n0 == 0:
    for tr in (0.05, 0.02, 0.01):
        result = detect(tr)
        if result.get('n_detected', 0) > 0:
            threshold_rel_used = tr
            note_parts.append(f'Lowered threshold_rel to {tr} because higher values missed all cores.')
            break
# If a huge number relative to image (over-detection of speckle), raise it.
else:
    # Heuristic upper bound on plausible count for a 2048x2048 field of 12nm cores.
    n_now = result.get('n_detected', 0)
    if n_now > 200000:
        for tr in (0.15, 0.20, 0.30):
            r2 = detect(tr)
            if r2.get('n_detected', 0) <= 200000:
                result = r2
                threshold_rel_used = tr
                note_parts.append(f'Raised threshold_rel to {tr} to suppress speckle over-detection.')
                break

n_detected = result.get('n_detected', 0)
n_interior = result.get('n_interior', 0)
polarity_used = result.get('polarity_used', 'dark')
annotation_masked = result.get('annotation_masked', False)
objects = result.get('objects', []) or []

# ---------------------------------------------------------------
# 3) Build size distribution from returned diameters_nm
# ---------------------------------------------------------------
diam_key_median = result.get('diameter_median_nm')
diam_key_mean = result.get('diameter_mean_nm')
diam_key_std = result.get('diameter_std_nm')

diameters = result.get('diameters_nm', None)
if diameters is None:
    diameters = [o.get('diameter_nm') for o in objects if o.get('diameter_nm') is not None]
diameters = np.array([d for d in diameters if d is not None], dtype=float)

if diameters.size > 0:
    d_median = float(np.median(diameters))
    d_mean = float(np.mean(diameters))
    d_std = float(np.std(diameters))
    hist_counts, hist_edges = np.histogram(diameters, bins=min(40, max(5, diameters.size // 5 + 1)))
    hist_counts = hist_counts.tolist()
    hist_edges = hist_edges.tolist()
else:
    d_median = d_mean = d_std = None
    hist_counts = []
    hist_edges = []

# ---------------------------------------------------------------
# 4) Centroid coordinates + nearest-neighbor stats + Clark-Evans
# ---------------------------------------------------------------
coords_px = np.array([[o['cy'], o['cx']] for o in objects], dtype=float) if objects else np.zeros((0, 2))

# Convert coords to nm if calibrated for NN distances
conv = pixel_size_nm if calibrated else 1.0  # nm/px or px/px

nn_distances = np.array([], dtype=float)
clark_evans = None
nn_median = nn_mean = nn_std = None

if coords_px.shape[0] >= 2:
    coords_phys = coords_px * conv  # (y,x) in nm or px
    tree = cKDTree(coords_phys)
    # k=2 -> first neighbor is self (distance 0), second is nearest neighbor
    dists, _ = tree.query(coords_phys, k=2)
    nn_distances = dists[:, 1]
    nn_distances = nn_distances[np.isfinite(nn_distances)]
    if nn_distances.size > 0:
        nn_median = float(np.median(nn_distances))
        nn_mean = float(np.mean(nn_distances))
        nn_std = float(np.std(nn_distances))

        # Clark-Evans index R = observed mean NN / expected mean NN (CSR)
        # expected = 1 / (2 * sqrt(density)), density = N / area
        H, W = img.shape
        area_phys = (H * conv) * (W * conv)  # nm^2 or px^2
        N = coords_phys.shape[0]
        density = N / area_phys if area_phys > 0 else 0.0
        if density > 0:
            expected_nn = 1.0 / (2.0 * np.sqrt(density))
            clark_evans = float(nn_mean / expected_nn) if expected_nn > 0 else None

# Interpret Clark-Evans: R<1 clustered, R~1 random, R>1 dispersed/regular
ce_interpretation = None
if clark_evans is not None:
    if clark_evans < 0.85:
        ce_interpretation = 'clustered (chain/network-like aggregation)'
    elif clark_evans > 1.15:
        ce_interpretation = 'dispersed/regular arrangement'
    else:
        ce_interpretation = 'random (near-CSR)'

# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

vmin, vmax = np.percentile(img, [1, 99])
axes[0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title('Original image')
axes[0].axis('off')

# Overlay detected blobs as circles
axes[1].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
for o in objects:
    r_px = 0.5 * (o['diameter_nm'] / conv) if (o.get('diameter_nm') is not None and conv > 0) else 3.0
    color = 'red' if o.get('border') else 'lime'
    circ = plt.Circle((o['cx'], o['cy']), r_px, fill=False, color=color, lw=0.4)
    axes[1].add_patch(circ)
axes[1].set_title(f'LoG blobs (N={n_detected}, interior={n_interior})')
axes[1].axis('off')

# Size distribution histogram
if diameters.size > 0:
    axes[2].hist(diameters, bins=min(40, max(5, diameters.size // 5 + 1)),
                 color='steelblue', edgecolor='k')
    axes[2].set_xlabel(f'Diameter ({unit})')
    axes[2].set_ylabel('Count')
    if d_median is not None:
        axes[2].axvline(d_median, color='red', ls='--', label=f'median={d_median:.2f} {unit}')
        axes[2].legend()
    axes[2].set_title('Size distribution')
else:
    axes[2].text(0.5, 0.5, 'No particles detected', ha='center', va='center')
    axes[2].set_title('Size distribution')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------
# Primary detection result: centroid coordinates (cy, cx) in pixels
np.save('centroid_coordinates_px.npy', coords_px)
if nn_distances.size > 0:
    np.save('nearest_neighbor_distances.npy', nn_distances)

# ---------------------------------------------------------------
# Assemble JSON results
# ---------------------------------------------------------------
summary_parts = []
if calibrated:
    summary_parts.append(f'Calibrated at {pixel_size_nm:.4f} nm/px (source: {px_source}).')
else:
    summary_parts.append('UNCALIBRATED (pixels).')
summary_parts.append(
    f'Scale-space LoG blob detection (log_blob_detect, polarity={polarity_used}, '
    f'threshold_rel={threshold_rel_used}, object_diameter_nm={expected_diameter_nm}) '
    f'detected {n_detected} particles ({n_interior} interior).'
)
if d_median is not None:
    summary_parts.append(f'Median diameter {d_median:.2f} {unit}.')
if clark_evans is not None:
    summary_parts.append(f'Clark-Evans index R={clark_evans:.3f} -> {ce_interpretation}.')
if result.get('note'):
    summary_parts.append(f"Tool note: {result['note']}")
if annotation_masked:
    summary_parts.append('A burned-in annotation/scale bar was masked.')
summary_parts.extend(note_parts)

results = {
    "analysis_type": "Scale-matched LoG blob detection (log_blob_detect) of dark nanoparticle cores -> size distribution + nearest-neighbor / Clark-Evans spatial statistics",
    "extracted_features": {
        "particle_count": int(n_detected),
        "particle_count_interior": int(n_interior),
        "pixel_size_nm": pixel_size_nm,
        "pixel_size_source": px_source,
        "size_unit": unit,
        "polarity_used": polarity_used,
        "threshold_rel_used": float(threshold_rel_used),
        "diameter_distribution_nm": {
            "median": d_median,
            "mean": d_mean,
            "std": d_std,
            "tool_median_nm": diam_key_median,
            "tool_mean_nm": diam_key_mean,
            "tool_std_nm": diam_key_std,
            "histogram_counts": hist_counts,
            "histogram_bin_edges": hist_edges,
        },
        "nearest_neighbor_distance_distribution": {
            "unit": unit,
            "median": nn_median,
            "mean": nn_mean,
            "std": nn_std,
            "n": int(nn_distances.size),
        },
        "clark_evans_clustering_index": clark_evans,
        "clark_evans_interpretation": ce_interpretation,
    },
    "quality_metrics": {
        "calibrated": bool(calibrated),
        "n_detected": int(n_detected),
        "n_interior": int(n_interior),
        "annotation_masked": bool(annotation_masked),
        "tool_note": result.get('note'),
    },
    "summary": ' '.join(summary_parts),
    "saved_arrays": {
        "centroid_coordinates_px.npy": {
            "description": f"Particle centroid coordinates (cy, cx) in pixels from LoG blob detection; {n_detected} rows.",
            "shape": list(coords_px.shape),
            "dtype": str(coords_px.dtype),
        }
    },
}

if nn_distances.size > 0:
    results["saved_arrays"]["nearest_neighbor_distances.npy"] = {
        "description": f"Nearest-neighbor distances between particle centroids in {unit}; {nn_distances.size} values.",
        "shape": list(nn_distances.shape),
        "dtype": str(nn_distances.dtype),
    }

print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}")
