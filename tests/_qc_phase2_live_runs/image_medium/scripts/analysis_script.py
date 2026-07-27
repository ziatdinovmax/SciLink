import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import skimage.measure
from skimage.segmentation import find_boundaries

results = {
    "analysis_type": "SAM instance segmentation of touching/overlapping triangular nanoparticles with per-object morphology",
    "extracted_features": {},
    "quality_metrics": {},
    "summary": "",
    "saved_arrays": {}
}
summary_notes = []

# ---------------------------------------------------------------
# 1. Load image
# ---------------------------------------------------------------
image = np.load("data.npy")
print("Loaded image shape:", image.shape, "dtype:", image.dtype)

# Handle possible multi-channel input
if image.ndim == 3:
    # not necessarily RGB; use first channel for analysis
    gray = image[:, :, 0].astype(np.float32)
    summary_notes.append(f"Input had {image.shape[2]} channels; used channel 0 for analysis.")
else:
    gray = image.astype(np.float32)

H, W = gray.shape[:2]

# ---------------------------------------------------------------
# 2. Resolve pixel size from metadata
# ---------------------------------------------------------------
metadata = None
try:
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
except Exception as e:
    summary_notes.append(f"Could not read metadata.json ({e}).")

nm_per_px = None
try:
    from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
    px = resolve_pixel_size_nm(metadata if metadata is not None else {}, gray.shape)
    if px is not None:
        nm_per_px = float(px['x'])
        summary_notes.append(f"Pixel size resolved: {nm_per_px:.4f} nm/px (source: {px.get('source')}).")
except Exception as e:
    summary_notes.append(f"resolve_pixel_size_nm failed ({e}).")

if nm_per_px is None or not np.isfinite(nm_per_px) or nm_per_px <= 0:
    nm_per_px = 0.355  # documented fallback from plan
    summary_notes.append("Pixel size unavailable; used fallback 0.355 nm/px per plan.")

# ---------------------------------------------------------------
# 3. Compute area limits from expected edge-length range
#    Equilateral triangle area = (sqrt(3)/4) * edge^2
# ---------------------------------------------------------------
def tri_area_px(edge_nm):
    edge_px = edge_nm / nm_per_px
    return (np.sqrt(3) / 4.0) * edge_px**2

min_edge_nm = 60.0   # ~60 nm edge lower bound (per plan)
max_edge_nm = 350.0  # ~350 nm edge upper bound (per plan)
min_area = float(tri_area_px(min_edge_nm))
max_area = float(tri_area_px(max_edge_nm))
summary_notes.append(
    f"Area filter from edge range [{min_edge_nm},{max_edge_nm}] nm: "
    f"min_area={min_area:.0f} px^2, max_area={max_area:.0f} px^2."
)

# ---------------------------------------------------------------
# 4. Prepare 3-channel uint8 image for SAM
# ---------------------------------------------------------------
g = gray.astype(np.float32)
lo, hi = np.percentile(g, 1), np.percentile(g, 99)
if hi <= lo:
    lo, hi = float(g.min()), float(g.max() if g.max() > g.min() else g.min() + 1)
g_norm = np.clip((g - lo) / (hi - lo), 0, 1)
g_u8 = (g_norm * 255).astype(np.uint8)
rgb = np.stack([g_u8, g_u8, g_u8], axis=-1)

# ---------------------------------------------------------------
# 5. Run SAM
# ---------------------------------------------------------------
sam_ok = False
result = None
try:
    from scilink.skills._shared.sam import run_sam_analysis
    result = run_sam_analysis(rgb, params={
        'sam_parameters': 'default',
        'min_area': int(min_area),
        'max_area': int(max_area),
        'pruning_iou_threshold': 0.5,
    })
    sam_ok = True
    summary_notes.append(
        f"SAM (default, iou=0.5) detected {result.get('total_count')} raw masks."
    )
except Exception as e:
    summary_notes.append(f"run_sam_analysis failed at runtime ({e}); falling back to Otsu+watershed.")

# ---------------------------------------------------------------
# 6. Build label map
# ---------------------------------------------------------------
labeled = np.zeros((H, W), dtype=np.int32)
n_detected_raw = 0

if sam_ok and result is not None and len(result.get('particles', [])) > 0:
    particles = result['particles']
    n_detected_raw = len(particles)
    for i, p in enumerate(particles):
        m = np.array(p['mask'], dtype=bool)
        if m.shape != (H, W):
            continue
        labeled[m] = i + 1
else:
    # Fallback: classical overlapping-object pipeline (watershed)
    import scipy.ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    thr = threshold_otsu(g_norm)
    binary_mask = g_norm > thr
    # triangles could be dark; pick polarity by which gives fewer huge blobs
    if binary_mask.mean() > 0.5:
        binary_mask = ~binary_mask
    binary_mask = binary_opening(binary_mask, disk(2))
    distance = ndi.distance_transform_edt(binary_mask)
    est_r = int(max(5, (min_edge_nm / nm_per_px) / 3))
    coords = peak_local_max(distance, min_distance=est_r, labels=binary_mask)
    markers = np.zeros(binary_mask.shape, dtype=np.int32)
    for j, c in enumerate(coords):
        markers[c[0], c[1]] = j + 1
    markers, _ = ndi.label(markers)
    labeled = watershed(-distance, markers, mask=binary_mask).astype(np.int32)
    n_detected_raw = int(labeled.max())
    summary_notes.append(f"Fallback watershed produced {n_detected_raw} regions.")

# ---------------------------------------------------------------
# 7. Region properties + post-filtering
# ---------------------------------------------------------------
props = skimage.measure.regionprops(labeled, intensity_image=gray)

# Area-based fragment pruning: compute typical area, drop < 1/4 typical
raw_areas = np.array([p.area for p in props], dtype=float)
typical_area = float(np.median(raw_areas)) if len(raw_areas) > 0 else 0.0
fragment_thresh = max(min_area, 0.25 * typical_area) if typical_area > 0 else min_area

# solidity coarse convexity sanity check (triangles are convex -> high solidity)
solidity_min = 0.55

def touches_border(bbox, shape):
    minr, minc, maxr, maxc = bbox
    return (minr <= 0) or (minc <= 0) or (maxr >= shape[0]) or (maxc >= shape[1])

kept = []          # analyzed (interior, passing filters)
n_edge_truncated = 0
n_fragment_dropped = 0
n_solidity_dropped = 0

# Rebuild a clean label map containing only retained-for-count objects
clean = np.zeros((H, W), dtype=np.int32)
new_label = 0

for p in props:
    if p.area < fragment_thresh:
        n_fragment_dropped += 1
        continue
    try:
        sol = p.solidity
    except Exception:
        sol = 1.0
    if sol < solidity_min:
        n_solidity_dropped += 1
        continue
    new_label += 1
    clean[labeled == p.label] = new_label
    border = touches_border(p.bbox, (H, W))
    if border:
        n_edge_truncated += 1
    else:
        kept.append(p)

labeled = clean
particle_count = new_label  # raw N retained (including edge-truncated)
particle_count_analyzed = len(kept)

summary_notes.append(
    f"Post-filter: dropped {n_fragment_dropped} fragments (<{fragment_thresh:.0f}px^2), "
    f"{n_solidity_dropped} low-solidity; {n_edge_truncated} edge-truncated excluded from stats."
)

# ---------------------------------------------------------------
# 8. Per-object morphology (analyzed set)
# ---------------------------------------------------------------
per_area_nm2 = []
equiv_edge_nm = []
centroids = []
for p in kept:
    area_nm2 = float(p.area) * (nm_per_px ** 2)
    edge_nm = float(np.sqrt(4.0 * area_nm2 / np.sqrt(3.0)))
    per_area_nm2.append(area_nm2)
    equiv_edge_nm.append(edge_nm)
    cy, cx = p.centroid
    centroids.append([float(cx), float(cy)])

per_area_nm2 = np.array(per_area_nm2, dtype=float)
equiv_edge_nm = np.array(equiv_edge_nm, dtype=float)

def stats(a):
    if len(a) == 0:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None}
    return {
        "count": int(len(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }

size_distribution = {
    "equivalent_edge_length_nm": stats(equiv_edge_nm),
    "area_nm2": stats(per_area_nm2),
    "edge_histogram": {},
}
if len(equiv_edge_nm) > 0:
    hist, edges = np.histogram(equiv_edge_nm, bins=min(20, max(5, len(equiv_edge_nm))))
    size_distribution["edge_histogram"] = {
        "counts": hist.tolist(),
        "bin_edges_nm": edges.tolist(),
    }

# ---------------------------------------------------------------
# 9. Visualization
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

ax = axes[0, 0]
ax.imshow(gray, cmap='gray')
ax.set_title('Original image')
ax.axis('off')

# Segmentation overlay
ax = axes[0, 1]
ax.imshow(gray, cmap='gray')
n_obj = int(labeled.max())
if n_obj > 0:
    rng = np.random.default_rng(0)
    cmap_colors = rng.random((n_obj + 1, 3))
    cmap_colors[0] = 0
    overlay = np.zeros((H, W, 4), dtype=float)
    for lab in range(1, n_obj + 1):
        mask = labeled == lab
        overlay[mask, :3] = cmap_colors[lab]
        overlay[mask, 3] = 0.45
    ax.imshow(overlay)
    bnd = find_boundaries(labeled, mode='outer')
    bnd_rgba = np.zeros((H, W, 4), dtype=float)
    bnd_rgba[bnd] = [1, 1, 0, 1]
    ax.imshow(bnd_rgba)
    # mark centroids
    for c in centroids:
        ax.plot(c[0], c[1], 'r+', markersize=6)
ax.set_title(f'SAM overlay: {n_obj} objects ({particle_count_analyzed} analyzed)')
ax.axis('off')

# Edge length histogram
ax = axes[1, 0]
if len(equiv_edge_nm) > 0:
    ax.hist(equiv_edge_nm, bins=min(20, max(5, len(equiv_edge_nm))),
            color='steelblue', edgecolor='k')
    ax.axvline(np.median(equiv_edge_nm), color='r', linestyle='--',
               label=f'median={np.median(equiv_edge_nm):.1f} nm')
    ax.legend()
ax.set_xlabel('Equivalent triangle edge length (nm)')
ax.set_ylabel('Count')
ax.set_title('Size distribution (analyzed particles)')

# Area distribution
ax = axes[1, 1]
if len(per_area_nm2) > 0:
    ax.hist(per_area_nm2, bins=min(20, max(5, len(per_area_nm2))),
            color='indianred', edgecolor='k')
ax.set_xlabel('Area (nm^2)')
ax.set_ylabel('Count')
ax.set_title('Area distribution (analyzed particles)')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close(fig)

# ---------------------------------------------------------------
# 10. Save arrays
# ---------------------------------------------------------------
np.save('analysis_labels.npy', labeled)
results["saved_arrays"]["analysis_labels.npy"] = {
    "description": f"Integer label map, {n_obj} particles labeled 1-{n_obj}, background=0",
    "shape": list(labeled.shape),
    "dtype": str(labeled.dtype),
}

if len(centroids) > 0:
    cent_arr = np.array(centroids, dtype=float)
    np.save('centroids.npy', cent_arr)
    results["saved_arrays"]["centroids.npy"] = {
        "description": "Centroid positions (x,y) in pixels for analyzed (interior) particles",
        "shape": list(cent_arr.shape),
        "dtype": str(cent_arr.dtype),
    }

# ---------------------------------------------------------------
# 11. Assemble results
# ---------------------------------------------------------------
results["extracted_features"] = {
    "particle_count": int(particle_count),
    "particle_count_analyzed": int(particle_count_analyzed),
    "n_edge_truncated": int(n_edge_truncated),
    "per_particle_area_nm2": per_area_nm2.tolist(),
    "equivalent_edge_length_nm": equiv_edge_nm.tolist(),
    "centroid_positions_xy_px": centroids,
    "size_distribution": size_distribution,
}
results["quality_metrics"] = {
    "pixel_size_nm_per_px": nm_per_px,
    "min_area_px": min_area,
    "max_area_px": max_area,
    "fragment_area_threshold_px": fragment_thresh,
    "solidity_min": solidity_min,
    "n_raw_masks": int(n_detected_raw),
    "n_fragment_dropped": int(n_fragment_dropped),
    "n_solidity_dropped": int(n_solidity_dropped),
    "used_sam": bool(sam_ok),
    "median_edge_length_nm": float(np.median(equiv_edge_nm)) if len(equiv_edge_nm) else None,
}
results["summary"] = (
    f"Segmented {particle_count} triangular nanoparticles "
    f"({particle_count_analyzed} interior analyzed, {n_edge_truncated} edge-truncated); "
    f"median equivalent edge length "
    f"{np.median(equiv_edge_nm):.1f} nm." if len(equiv_edge_nm) else
    f"Segmented {particle_count} particles ({particle_count_analyzed} analyzed)."
) + " NOTES: " + " ".join(summary_notes)

print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}")
