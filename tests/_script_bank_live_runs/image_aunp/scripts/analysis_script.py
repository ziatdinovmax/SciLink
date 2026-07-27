import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure, segmentation, filters, morphology, feature
from scipy import ndimage as ndi

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm

# ---------------------------------------------------------------
# 1) Load image
# ---------------------------------------------------------------
image = np.load("data.npy")

# Handle possible channel dimension
if image.ndim == 3:
    img = image[:, :, 0].astype(np.float32)
else:
    img = image.astype(np.float32)

# ---------------------------------------------------------------
# 1b) Resolve pixel size from metadata.json
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

# Known expected diameter ~80 nm
expected_diameter_nm = 80.0

calibrated = pixel_size_nm is not None and pixel_size_nm > 0
if calibrated:
    unit = 'nm'
    expected_radius_px = (expected_diameter_nm / pixel_size_nm) / 2.0
else:
    unit = 'px'
    # Estimate expected radius in px from the image as a fallback.
    # Rough guess: ~80 px diameter -> 40 px radius if truly unknown.
    expected_radius_px = 40.0

# Guard the marker min_distance against degenerate values
min_dist = int(max(3, round(expected_radius_px)))

# ---------------------------------------------------------------
# 2) Binary mask: dark particles on lighter background (Otsu)
#    Optional light Gaussian smoothing to suppress interior mottling.
# ---------------------------------------------------------------
note = None
smoothed = filters.gaussian(img, sigma=1.0, preserve_range=True)
try:
    thr = filters.threshold_otsu(smoothed)
except Exception as e:
    thr = float(np.median(smoothed))
    note = f"Otsu failed ({e}); used median threshold."

# Dark particles -> True where intensity is BELOW threshold (polarity set from image)
binary_mask = smoothed < thr

# Cleanup: remove small specks and close small interior holes
binary_mask = morphology.binary_opening(binary_mask, morphology.disk(1))
binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=int(np.pi * (0.25 * expected_radius_px) ** 2 + 1))

# ---------------------------------------------------------------
# 3) Distance transform + peak_local_max markers
#    min_distance ~= expected particle radius in px
# ---------------------------------------------------------------
distance = ndi.distance_transform_edt(binary_mask)

coords = feature.peak_local_max(
    distance,
    min_distance=min_dist,
    labels=binary_mask,
    exclude_border=False,
)
marker_mask = np.zeros(distance.shape, dtype=bool)
if coords.shape[0] > 0:
    marker_mask[tuple(coords.T)] = True
markers, _ = ndi.label(marker_mask)

# ---------------------------------------------------------------
# 4) Watershed on inverted distance transform to split touching particles
# ---------------------------------------------------------------
if markers.max() > 0:
    labeled = segmentation.watershed(-distance, markers, mask=binary_mask)
else:
    # No markers found; fall back to connected components on the mask.
    labeled, _ = ndi.label(binary_mask)
labeled = labeled.astype(np.int32)

# ---------------------------------------------------------------
# 5) Label instances; measure per-object equivalent-diameter + area
#    (regionprops on the ORIGINAL image), convert to nm if calibrated
# ---------------------------------------------------------------
conv = pixel_size_nm if calibrated else 1.0  # nm/px (or px/px)

props = measure.regionprops(labeled, intensity_image=img)

# Reject tiny fragments below ~1/4 of a typical particle area as noise
min_area_px = np.pi * (0.5 * expected_radius_px) ** 2  # 1/4 of expected disk area

objects = []
for p in props:
    area_px = float(p.area)
    if area_px < max(1.0, min_area_px):
        continue
    eq_diam_px = float(p.equivalent_diameter)
    cy, cx = p.centroid
    minr, minc, maxr, maxc = p.bbox
    border = (minr <= 0 or minc <= 0 or maxr >= img.shape[0] or maxc >= img.shape[1])
    objects.append({
        'label': int(p.label),
        'cy': float(cy),
        'cx': float(cx),
        'area_px': area_px,
        'eq_diam_px': eq_diam_px,
        'diameter': eq_diam_px * conv,   # nm if calibrated else px
        'border': bool(border),
        'bbox': (int(minr), int(minc), int(maxr), int(maxc)),
    })

# Rebuild a clean label map keeping only accepted objects
kept_labels = {o['label'] for o in objects}
clean = np.zeros_like(labeled)
for new_i, o in enumerate(objects, start=1):
    clean[labeled == o['label']] = new_i
    o['new_label'] = new_i
labeled = clean

# ---------------------------------------------------------------
# 6) Exclude border-truncated objects -> interior population
# ---------------------------------------------------------------
n_detected = len(objects)
interior_objs = [o for o in objects if not o['border']]
n_interior = len(interior_objs)

interior_diams = [o['diameter'] for o in interior_objs]
interior_areas = [o['area_px'] for o in interior_objs]
interior_diams_arr = np.array(interior_diams, dtype=float)

all_diams = [o['diameter'] for o in objects]

# ---------------------------------------------------------------
# 8) Statistics on interior objects
# ---------------------------------------------------------------
if interior_diams_arr.size > 0:
    d_median = float(np.median(interior_diams_arr))
    d_mean = float(np.mean(interior_diams_arr))
    d_std = float(np.std(interior_diams_arr))
else:
    d_median = d_mean = d_std = None

# ---------------------------------------------------------------
# 7) Over-detection cross-check: cluster of fragments below ~1/4 typical area
# ---------------------------------------------------------------
over_detection_flag = False
if interior_diams_arr.size > 3 and d_median:
    small_thresh = 0.5 * float(d_median)  # diameter ratio 0.5 -> area 1/4
    n_small = int(np.sum(interior_diams_arr < small_thresh))
    if n_small > 0.25 * interior_diams_arr.size:
        over_detection_flag = True

# ---------------------------------------------------------------
# Visualization (overlay instance boundaries on the ORIGINAL image)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

vmin, vmax = np.percentile(img, [1, 99])
axes[0].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
axes[0].set_title('Original image')
axes[0].axis('off')

axes[1].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
if labeled.max() > 0:
    boundaries = segmentation.find_boundaries(labeled, mode='outer')
    overlay = np.zeros((*img.shape, 4), dtype=float)
    overlay[boundaries] = [1.0, 1.0, 0.0, 1.0]  # yellow boundaries
    axes[1].imshow(overlay)
    for o in objects:
        color = 'red' if o['border'] else 'lime'
        r_px = o['eq_diam_px'] / 2.0
        circ = plt.Circle((o['cx'], o['cy']), r_px, fill=False, color=color, lw=0.8)
        axes[1].add_patch(circ)
axes[1].set_title(f'Watershed instances (N={n_detected}, interior={n_interior})')
axes[1].axis('off')

if interior_diams_arr.size > 0:
    axes[2].hist(interior_diams_arr,
                 bins=min(30, max(5, interior_diams_arr.size)),
                 color='steelblue', edgecolor='k')
    axes[2].set_xlabel(f'Equivalent diameter ({unit})')
    axes[2].set_ylabel('Count')
    if d_median:
        axes[2].axvline(d_median, color='red', ls='--',
                        label=f'median={d_median:.1f} {unit}')
        axes[2].legend()
else:
    axes[2].text(0.5, 0.5, 'No interior particles', ha='center', va='center')
axes[2].set_title('Interior size distribution')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------
np.save('analysis_labels.npy', labeled)

# ---------------------------------------------------------------
# Assemble JSON results
# ---------------------------------------------------------------
summary_parts = []
if not calibrated:
    summary_parts.append('UNCALIBRATED: sizes reported in pixels (metadata.json missing or pixel size unresolved).')
else:
    summary_parts.append(f'Calibrated at {pixel_size_nm:.4f} nm/px (source: {px_source}).')

summary_parts.append('Method: Otsu dark-particle mask (dark-on-lighter-background) -> distance transform -> '
                     f'peak_local_max markers (min_distance={min_dist} px ~ expected radius) -> watershed on inverted '
                     'distance -> regionprops equivalent-diameter/area.')

if n_detected == 0:
    summary_parts.append('No particles detected.')
else:
    med_txt = f'{d_median:.1f}' if d_median is not None else 'NA'
    summary_parts.append(
        f'Detected {n_detected} particles ({n_interior} interior), median diameter {med_txt} {unit}.'
    )
if over_detection_flag:
    summary_parts.append('WARNING: cluster of small fragments below ~1/4 typical area detected (possible watershed over-splitting of mottled interiors; consider increasing min_distance).')
if note:
    summary_parts.append(note)

results = {
    "analysis_type": "Distance-transform watershed nanoparticle detection & sizing (~80 nm target diameter): Otsu dark-particle mask -> distance transform -> peak_local_max markers -> watershed -> regionprops",
    "extracted_features": {
        "particle_count_N_detected": n_detected,
        "particle_count_N_interior": n_interior,
        "pixel_size_nm": pixel_size_nm,
        "pixel_size_source": px_source,
        "size_unit": unit,
        "expected_radius_px": float(expected_radius_px),
        "marker_min_distance_px": int(min_dist),
        "diameter_median": d_median,
        "diameter_mean": d_mean,
        "diameter_std": d_std,
        "interior_diameters": interior_diams,
        "interior_areas_px": interior_areas,
        "all_diameters": all_diams,
    },
    "quality_metrics": {
        "calibrated": bool(calibrated),
        "over_detection_small_fragment_flag": bool(over_detection_flag),
        "otsu_threshold": float(thr),
        "min_fragment_area_px": float(min_area_px),
        "note": note,
    },
    "summary": ' '.join(summary_parts),
    "saved_arrays": {
        "analysis_labels.npy": {
            "description": f"Integer instance label map from watershed segmentation, {n_detected} objects labeled 1-{n_detected}, background=0",
            "shape": list(labeled.shape),
            "dtype": str(labeled.dtype),
        }
    },
}

print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}")
