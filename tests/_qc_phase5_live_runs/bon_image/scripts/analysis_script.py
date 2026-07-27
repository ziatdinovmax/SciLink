import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label as sklabel, find_contours
from scipy import ndimage as ndi

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.log_blob import log_blob_detect
from scilink.skills._shared.sam import run_sam_analysis

# ------------------------------------------------------------------
# 1. Load image + metadata
# ------------------------------------------------------------------
image = np.load('data.npy')

if image.ndim == 3:
    analysis_img = image[:, :, 0].astype(np.float32)
else:
    analysis_img = image.astype(np.float32)

metadata = None
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception:
    metadata = None

# ------------------------------------------------------------------
# 2. Resolve pixel size
# ------------------------------------------------------------------
px = None
if metadata is not None:
    try:
        px = resolve_pixel_size_nm(metadata, analysis_img.shape)
    except Exception:
        px = None

if px is not None:
    pixel_size_nm = float(px['x'])
    px_source = px.get('source', 'metadata')
else:
    pixel_size_nm = None
    px_source = 'unavailable (diameters in pixels)'

summary_notes = []

H, W = analysis_img.shape

# ------------------------------------------------------------------
# 3. STEP 1 - SEED DETECTION via log_blob_detect (retain).
#    Centers are used ONLY as seeds/sanity/counting, NOT for sizing.
#    Sigma bracketed from particle RADIUS (calibrated, do not regress).
# ------------------------------------------------------------------
object_diameter_nm = 80.0

if pixel_size_nm is not None:
    r_px = (object_diameter_nm / 2.0) / pixel_size_nm
else:
    r_px = 20.0
    summary_notes.append('Pixel size unavailable; using nominal radius ~20 px for sigma bracketing (diameters in pixels).')

min_sigma = max(1.0, 0.6 * r_px)
max_sigma = max(min_sigma + 1.0, 1.3 * r_px)


def run_log(threshold_rel):
    return log_blob_detect(
        analysis_img,
        polarity='dark',
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=12,
        threshold_rel=threshold_rel,
        overlap=0.5,
        exclude_border=False,
        mask_annotations=True,
        pixel_size_nm=pixel_size_nm,
    )


# Lower threshold_rel slightly to recover faint disks in lower-left/center.
chosen_res = None
chosen_tr = None
for tr in [0.08, 0.10, 0.15, 0.20, 0.30, 0.40]:
    try:
        res = run_log(tr)
    except Exception as e:
        summary_notes.append(f'log_blob_detect failed at threshold_rel={tr}: {e}')
        res = None
        break
    n = res.get('n_detected', 0)
    chosen_res = res
    chosen_tr = tr
    if 0 < n < 800:
        break

seed_objs = chosen_res.get('objects', []) if chosen_res is not None else []

# de-duplicate detections within ~one radius
dedup_seeds = []
if seed_objs:
    min_sep = max(3.0, 0.9 * r_px)
    centers = np.array([[o['cy'], o['cx']] for o in seed_objs], dtype=float)
    taken = np.zeros(len(seed_objs), dtype=bool)
    order = np.argsort([-(o.get('diameter_nm') or 0.0) for o in seed_objs])
    for idx in order:
        if taken[idx]:
            continue
        cy, cx = centers[idx]
        for j in range(len(seed_objs)):
            if j == idx or taken[j]:
                continue
            if (centers[j][0] - cy) ** 2 + (centers[j][1] - cx) ** 2 < min_sep ** 2:
                taken[j] = True
        dedup_seeds.append(seed_objs[idx])
        taken[idx] = True
    if len(dedup_seeds) < len(seed_objs):
        summary_notes.append(f'De-duplicated seeds within ~{min_sep:.1f} px: {len(seed_objs)} -> {len(dedup_seeds)}.')
else:
    dedup_seeds = []

seed_centers = np.array([[o['cy'], o['cx']] for o in dedup_seeds], dtype=float) if dedup_seeds else np.zeros((0, 2))
# calibrated sigma-derived diameter per seed (sqrt(2)*sigma calibrated inside tool)
seed_sigma_diam = [o.get('diameter_nm', None) for o in dedup_seeds]
n_seeds = len(dedup_seeds)

summary_notes.append(f'log_blob_detect seeds: {n_seeds} (threshold_rel={chosen_tr}).')


def circularity_from_props(p):
    if p.perimeter > 0:
        return float(4.0 * np.pi * p.area / (p.perimeter ** 2))
    return 0.0


# ------------------------------------------------------------------
# 4. STEP 2 - INSTANCE SEGMENTATION via run_sam_analysis (registered tool).
#    Replaces the per-bbox Otsu mask step. Area band sized to an ~80 nm
#    sphere: area ~ pi*r_px^2 with a [0.5x, 2x] tolerance band.
# ------------------------------------------------------------------
expected_area_px = float(np.pi * (r_px ** 2))
sam_min_area = float(0.5 * expected_area_px)
sam_max_area = float(2.0 * expected_area_px)

sam_result = None
sam_ok = False
sam_note = ''

try:
    sam_result = run_sam_analysis(
        analysis_img,
        params={
            'sam_parameters': 'default',
            'min_area': int(round(sam_min_area)),
            'max_area': int(round(sam_max_area)),
            'pruning_iou_threshold': 0.5,
            'use_clahe': False,
        },
    )
    n_sam = sam_result.get('total_count', 0) if sam_result else 0
    if sam_result is not None and n_sam > 0:
        sam_ok = True
    else:
        sam_note = 'run_sam_analysis returned 0 instances at default; retrying with sensitive.'
        summary_notes.append(sam_note)
except Exception as e:
    sam_note = f'run_sam_analysis failed at default: {e}'
    summary_notes.append(sam_note)

# Retry with sensitive params if default under-segmented badly relative to seeds
if (not sam_ok) or (sam_ok and n_seeds > 0 and sam_result.get('total_count', 0) < 0.4 * n_seeds):
    try:
        sam_result2 = run_sam_analysis(
            analysis_img,
            params={
                'sam_parameters': 'sensitive',
                'min_area': int(round(sam_min_area)),
                'max_area': int(round(sam_max_area)),
                'pruning_iou_threshold': 0.6,
                'use_clahe': False,
            },
        )
        n_sam2 = sam_result2.get('total_count', 0) if sam_result2 else 0
        if n_sam2 > (sam_result.get('total_count', 0) if sam_result else 0):
            sam_result = sam_result2
            sam_ok = n_sam2 > 0
            summary_notes.append(f"Used 'sensitive' SAM params (instances {n_sam2}).")
    except Exception as e:
        summary_notes.append(f'run_sam_analysis sensitive retry failed: {e}')

# Build labeled mask from SAM instances
sam_labeled = np.zeros(analysis_img.shape, dtype=np.int32)
sam_props = []
if sam_ok and sam_result is not None:
    parts = sam_result.get('particles', [])
    lid = 0
    for p in parts:
        try:
            m = np.array(p['mask'], dtype=bool)
        except Exception:
            continue
        if m.shape != analysis_img.shape or m.sum() == 0:
            continue
        lid += 1
        sam_labeled[m] = lid
    if lid == 0:
        sam_ok = False
        summary_notes.append('SAM masks could not be rasterized to image shape; falling back to watershed.')
    else:
        sam_props = regionprops(sam_labeled, intensity_image=analysis_img)

# ------------------------------------------------------------------
# 5. FALLBACK: distance-transform watershed on Otsu mask, only if SAM
#    unavailable / unusable.
# ------------------------------------------------------------------
use_watershed = not sam_ok
method_used = None

if use_watershed:
    method_used = 'distance_transform_watershed (fallback; SAM unavailable/unusable)'
    summary_notes.append('SAM unavailable or produced unusable output; using distance-transform watershed fallback.')
    from skimage.morphology import binary_opening, disk
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    thr = threshold_otsu(analysis_img)
    binary_mask = analysis_img < thr  # dark particles -> foreground below thr
    binary_mask = binary_opening(binary_mask, disk(2))
    binary_mask = ndi.binary_fill_holes(binary_mask)

    distance = ndi.distance_transform_edt(binary_mask)
    distance_s = ndi.gaussian_filter(distance, sigma=2)

    est_radius_px = max(3, int(round(r_px)))
    min_distance = max(3, int(est_radius_px * 0.8))

    coords = peak_local_max(distance_s, min_distance=min_distance, labels=binary_mask)
    markers = np.zeros(distance_s.shape, dtype=np.int32)
    for i, (rr, cc) in enumerate(coords, start=1):
        markers[rr, cc] = i
    markers = ndi.label(markers)[0]

    ws = watershed(-distance_s, markers, mask=binary_mask)
    inst_labeled = np.zeros_like(ws, dtype=np.int32)
    props_ws = regionprops(ws.astype(np.int32), intensity_image=analysis_img)
    nid = 0
    min_area = max(20, int(0.2 * np.pi * r_px ** 2))
    for p in props_ws:
        if p.area < min_area:
            continue
        nid += 1
        inst_labeled[ws == p.label] = nid
    inst_labeled_props = regionprops(inst_labeled, intensity_image=analysis_img)
else:
    method_used = 'log_blob_detect seeds + run_sam_analysis instance masks (equivalent_diameter, no radius clamp)'
    inst_labeled = sam_labeled
    inst_labeled_props = sam_props

# ------------------------------------------------------------------
# 6. STEP: MATCH SAM/watershed instances to LoG seeds + MEASURE.
#    One instance per seed. Border handling: exclude only instances whose
#    MASK touches the frame edge (not centers merely near the edge).
#    NO radius clamp on masks.
# ------------------------------------------------------------------
particle_records = []

# map each instance label -> which seeds fall inside it
label_to_seeds = {}
if n_seeds > 0 and inst_labeled.max() > 0:
    for si in range(n_seeds):
        cy = int(round(seed_centers[si][0])); cx = int(round(seed_centers[si][1]))
        if 0 <= cy < H and 0 <= cx < W:
            lab = int(inst_labeled[cy, cx])
            if lab > 0:
                label_to_seeds.setdefault(lab, []).append(si)

seed_matched = np.zeros(n_seeds, dtype=bool)
match_status = ['recall-miss'] * n_seeds  # per seed

# process instances
for p in inst_labeled_props:
    lab = int(p.label)
    area_px = int(p.area)
    eqd_px = float(p.equivalent_diameter)
    circ = circularity_from_props(p)
    ecc = float(p.eccentricity)
    solidity = float(p.solidity)

    # border: does the mask touch the frame edge?
    minr, minc, maxr, maxc = p.bbox
    border = (minr <= 0 or minc <= 0 or maxr >= H or maxc >= W)

    seeds_here = label_to_seeds.get(lab, [])
    n_seeds_in = len(seeds_here)
    if n_seeds_in == 0:
        inst_match = 'no-seed'  # instance without any seed (possible speckle/spurious)
    elif n_seeds_in == 1:
        inst_match = 'matched'
        seed_matched[seeds_here[0]] = True
        match_status[seeds_here[0]] = 'matched'
    else:
        inst_match = 'under-segmented'
        for si in seeds_here:
            seed_matched[si] = True
            match_status[si] = 'under-segmented'

    if pixel_size_nm is not None:
        eqd = eqd_px * pixel_size_nm
        area_val = area_px * (pixel_size_nm ** 2)
    else:
        eqd = eqd_px
        area_val = float(area_px)

    particle_records.append({
        'cy': float(p.centroid[0]), 'cx': float(p.centroid[1]),
        'label': lab,
        'equivalent_diameter': float(eqd),
        'diameter': float(eqd),
        'diameter_source': 'sam-instance' if not use_watershed else 'watershed-instance',
        'area': float(area_val),
        'area_px': int(area_px),
        'circularity': float(circ),
        'eccentricity': float(ecc),
        'solidity': float(solidity),
        'border': bool(border),
        'n_seeds_in_instance': int(n_seeds_in),
        'instance_match': inst_match,
    })

# ------------------------------------------------------------------
# 7. STEP 5 - FALLBACK for seeds SAM failed to segment (recall misses):
#    use calibrated sigma-derived diameter. Report the fraction using fallback.
# ------------------------------------------------------------------
fallback_records = []
for si in range(n_seeds):
    if seed_matched[si]:
        continue
    # recall miss -> sigma fallback
    sd = seed_sigma_diam[si]
    cy, cx = seed_centers[si]
    # border check: seed within one radius of edge
    border = (cy - r_px <= 0 or cx - r_px <= 0 or cy + r_px >= H or cx + r_px >= W)
    if sd is None:
        continue
    if pixel_size_nm is not None:
        r_nm = float(sd) / 2.0
        area_val = float(np.pi * r_nm ** 2)
    else:
        r_final_px = float(sd) / 2.0
        area_val = float(np.pi * r_final_px ** 2)
    rec = {
        'cy': float(cy), 'cx': float(cx),
        'label': -1,
        'equivalent_diameter': float(sd),
        'diameter': float(sd),
        'diameter_source': 'sigma-fallback',
        'area': float(area_val),
        'area_px': None,
        'circularity': None,
        'eccentricity': None,
        'solidity': None,
        'border': bool(border),
        'n_seeds_in_instance': 0,
        'instance_match': 'recall-miss',
    }
    fallback_records.append(rec)
    match_status[si] = 'recall-miss'

all_records = particle_records + fallback_records

n_recall_miss = int(sum(1 for s in match_status if s == 'recall-miss'))
n_matched = int(sum(1 for s in match_status if s == 'matched'))
n_underseg = int(sum(1 for s in match_status if s == 'under-segmented'))

n_fallback_used = len(fallback_records)
frac_fallback = (float(n_fallback_used) / float(len(all_records))) if all_records else None

# ------------------------------------------------------------------
# 8. Statistics.
#    PRIMARY size distribution: SAM/watershed instances that MATCH a seed
#    (one seed), non-border, using equivalent_diameter (NO clamp).
#    Light QC (roundness/solidity) for reporting only, NOT for exclusion.
# ------------------------------------------------------------------
unit = 'nm' if pixel_size_nm is not None else 'px'

# primary set: matched (single-seed) instances, non-border
primary = [r for r in particle_records
           if r['instance_match'] == 'matched' and not r['border']]
# if too few matched-single instances, broaden to all non-border instances (report choice)
if len(primary) < 5:
    primary = [r for r in particle_records if not r['border']]
    summary_notes.append('Few single-seed matched instances; primary distribution broadened to all non-border SAM instances.')

diams = np.array([r['equivalent_diameter'] for r in primary], dtype=float)
areas = np.array([r['area'] for r in primary if r['area'] is not None], dtype=float)
circs = np.array([r['circularity'] for r in primary if r['circularity'] is not None], dtype=float)
eccs = np.array([r['eccentricity'] for r in primary if r.get('eccentricity') is not None], dtype=float)
solids = np.array([r['solidity'] for r in primary if r.get('solidity') is not None], dtype=float)

# sigma-derived cross-check: all seeds' sigma diameters (interior)
sigma_interior = []
for si in range(n_seeds):
    sd = seed_sigma_diam[si]
    cy, cx = seed_centers[si]
    border = (cy - r_px <= 0 or cx - r_px <= 0 or cy + r_px >= H or cx + r_px >= W)
    if sd is not None and not border:
        sigma_interior.append(float(sd))
sigma_diams = np.array(sigma_interior, dtype=float)

if diams.size > 0:
    d_mean = float(np.mean(diams)); d_med = float(np.median(diams)); d_std = float(np.std(diams))
    d_q25 = float(np.percentile(diams, 25)); d_q75 = float(np.percentile(diams, 75))
    d_iqr = float(d_q75 - d_q25)
    nb = int(min(25, max(5, diams.size)))
    hist_counts, hist_edges = np.histogram(diams, bins=nb)
else:
    d_mean = d_med = d_std = d_iqr = d_q25 = d_q75 = None
    hist_counts, hist_edges = np.array([]), np.array([])

sigma_med = float(np.median(sigma_diams)) if sigma_diams.size > 0 else None

# counts
n_border_excluded = int(sum(1 for r in all_records if r['border']))
n_interior = int(sum(1 for r in all_records if not r['border']))
n_total = len(all_records)

# ------------------------------------------------------------------
# Cross-check: SAM median vs sigma median vs nominal 80 nm.
# Investigate divergence; report which boundary is measured.
# ------------------------------------------------------------------
crosscheck_note = ''
measured_boundary_type = 'unknown'
if d_med is not None:
    parts_cc = []
    if sigma_med is not None and sigma_med > 0:
        rel_div = abs(d_med - sigma_med) / sigma_med
        parts_cc.append(f'SAM-instance median = {d_med:.1f} {unit}; sigma cross-check median = {sigma_med:.1f} {unit} (diff {rel_div*100:.0f}%).')
    if pixel_size_nm is not None:
        rel_nom = abs(d_med - object_diameter_nm) / object_diameter_nm
        parts_cc.append(f'Nominal outer diameter = {object_diameter_nm:.0f} nm (SAM median differs by {rel_nom*100:.0f}%).')
        # boundary interpretation: SAM measures the full particle-to-background boundary.
        # sigma responds to the dark core; if SAM >> sigma, SAM is capturing the outer boundary.
        if sigma_med is not None and d_med >= sigma_med:
            measured_boundary_type = 'outer (particle-to-background) boundary from SAM instance; sigma-derived tracks the dark core (smaller)'
        elif sigma_med is not None and d_med < sigma_med:
            measured_boundary_type = 'SAM instance smaller than sigma-core estimate; likely under-segmentation (investigate)'
        else:
            measured_boundary_type = 'outer (particle-to-background) boundary from SAM instance'
        # Investigate: does SAM median land near nominal ~80 nm?
        if rel_nom > 0.25:
            parts_cc.append('SAM median diverges from nominal ~80 nm by >25%: check whether 80 nm refers to outer diameter vs core, and inspect the overlay for under/over-segmentation.')
        else:
            parts_cc.append('SAM median is consistent with nominal ~80 nm (outer boundary).')
    crosscheck_note = ' '.join(parts_cc)

# ------------------------------------------------------------------
# 9. Visualization (headline result in visualization.png)
# ------------------------------------------------------------------
vmin, vmax = np.percentile(analysis_img, [1, 99])
label_map = inst_labeled

fig, axes = plt.subplots(2, 2, figsize=(14, 14))

ax = axes[0, 0]
ax.imshow(analysis_img, cmap='gray', vmin=vmin, vmax=vmax)
ax.set_title('Original image')
ax.axis('off')

ax = axes[0, 1]
ax.imshow(analysis_img, cmap='gray', vmin=vmin, vmax=vmax)
if label_map.max() > 0:
    for lab in range(1, int(label_map.max()) + 1):
        m = label_map == lab
        if m.sum() == 0:
            continue
        for cont in find_contours(m.astype(float), 0.5):
            ax.plot(cont[:, 1], cont[:, 0], color='cyan', linewidth=0.6)
# seeds
if n_seeds > 0:
    for si in range(n_seeds):
        st = match_status[si]
        if st == 'matched':
            col = 'lime'
        elif st == 'under-segmented':
            col = 'orange'
        else:
            col = 'yellow'  # recall-miss -> sigma fallback
        ax.plot(seed_centers[si][1], seed_centers[si][0], '+', color=col, markersize=6, markeredgewidth=1.2)
ax.set_title(f'SAM instances + seeds ({method_used})\n{n_total} objects, {n_interior} interior\nlime=matched, orange=under-seg, yellow=recall-miss(sigma)')
ax.axis('off')

ax = axes[1, 0]
if diams.size > 0:
    ax.hist(diams, bins=int(min(25, max(5, diams.size))), color='steelblue', edgecolor='k')
    ax.axvline(d_med, color='red', linestyle='--', label=f'SAM median={d_med:.1f} {unit}')
    if sigma_med is not None:
        ax.axvline(sigma_med, color='green', linestyle=':', label=f'sigma median={sigma_med:.1f} {unit}')
    if pixel_size_nm is not None:
        ax.axvline(object_diameter_nm, color='black', linestyle='-.', label=f'nominal={object_diameter_nm:.0f} nm')
    ax.legend()
ax.set_xlabel(f'Equivalent diameter ({unit})')
ax.set_ylabel('Count')
ax.set_title('Primary size distribution (SAM matched instances)')

ax = axes[1, 1]
if solids.size > 0 and diams.size > 0:
    ax.scatter(diams, solids, s=15, alpha=0.6, color='purple')
    ax.set_xlabel(f'Equivalent diameter ({unit})')
    ax.set_ylabel('Solidity')
    ax.set_title('QC (reporting only): diameter vs solidity')
else:
    ax.text(0.5, 0.5, 'No solidity data', ha='center', va='center')
    ax.axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# full-resolution overlay for the verifier
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
ax2.imshow(analysis_img, cmap='gray', vmin=vmin, vmax=vmax)
if label_map.max() > 0:
    for lab in range(1, int(label_map.max()) + 1):
        m = label_map == lab
        if m.sum() == 0:
            continue
        for cont in find_contours(m.astype(float), 0.5):
            ax2.plot(cont[:, 1], cont[:, 0], color='cyan', linewidth=0.5)
if n_seeds > 0:
    for si in range(n_seeds):
        st = match_status[si]
        if st == 'matched':
            col = 'lime'
        elif st == 'under-segmented':
            col = 'orange'
        else:
            col = 'yellow'
        ax2.plot(seed_centers[si][1], seed_centers[si][0], '+', color=col, markersize=5, markeredgewidth=1.0)
ax2.set_title(f'SAM instance masks + LoG seeds ({method_used})')
ax2.axis('off')
plt.tight_layout()
plt.savefig('verifier_panel_overlay.png', dpi=100)
plt.close()

# ------------------------------------------------------------------
# 10. Save arrays
# ------------------------------------------------------------------
np.save('analysis_labels.npy', label_map.astype(np.int32))

saved_arrays = {
    'analysis_labels.npy': {
        'description': f'Integer instance label map (particles 1..N, background=0). Method: {method_used}. Each label is a true per-particle instance mask from run_sam_analysis (or watershed fallback); sizes are regionprops equivalent_diameter with NO radius clamp.',
        'shape': list(label_map.shape),
        'dtype': 'int32',
    }
}

# ------------------------------------------------------------------
# 11. Report JSON
# ------------------------------------------------------------------
summary = (
    f'Detected {n_seeds} LoG seeds (polarity=dark, sigma bracketed from ~80 nm radius) and segmented '
    f'{len(particle_records)} instances via {method_used}. '
    f'PRIMARY size distribution built from seed-matched, non-border SAM instances using regionprops '
    f'equivalent_diameter with NO radius clamp: median '
    f'{("%.1f %s" % (d_med, unit)) if d_med is not None else "N/A"}, IQR '
    f'{("%.1f %s" % (d_iqr, unit)) if d_iqr is not None else "N/A"}. '
    f'Cross-check: {crosscheck_note} '
    f'Seed->instance match: {n_matched} matched, {n_underseg} under-segmented (multi-seed instance), '
    f'{n_recall_miss} recall-miss (used sigma fallback). Fraction using sigma fallback = '
    f'{("%.2f" % frac_fallback) if frac_fallback is not None else "N/A"}. '
    f'Measured boundary type: {measured_boundary_type}. '
    f'SAM area band = [{sam_min_area:.0f}, {sam_max_area:.0f}] px (0.5x-2x of pi*r_px^2). '
    f'Border handling excludes only mask-edge-touching instances ({n_border_excluded} excluded). '
    f'Pixel size source: {px_source}. ' + ' '.join(summary_notes)
)

results = {
    'analysis_type': 'Dark Au sphere sizing: log_blob_detect seeds (counting/sanity) + run_sam_analysis instance masks (primary sizing via regionprops equivalent_diameter, no radius clamp), seed->instance matching, sigma-derived fallback for recall misses, mask-edge border exclusion. Watershed documented fallback.',
    'extracted_features': {
        'pixel_size_nm_and_calibration_source': {'pixel_size_nm': pixel_size_nm, 'source': px_source},
        'diameter_unit': unit,
        'method_used': method_used,
        'threshold_rel_used': (float(chosen_tr) if chosen_tr is not None else None),
        'min_sigma_px': float(min_sigma),
        'max_sigma_px': float(max_sigma),
        'sam_area_band_px': [float(sam_min_area), float(sam_max_area)],
        'particle_count_total': int(n_total),
        'particle_count_interior': int(n_interior),
        'particle_count_border_excluded': int(n_border_excluded),
        'n_log_seeds': int(n_seeds),
        'n_sam_instances': int(len(particle_records)),
        'per_particle_equivalent_diameter_nm': [float(r['equivalent_diameter']) for r in primary],
        'median_diameter_nm': d_med,
        'mean_diameter_nm': d_mean,
        'std_diameter_nm': d_std,
        'diameter_iqr': d_iqr,
        'diameter_q25': d_q25,
        'diameter_q75': d_q75,
        'sigma_derived_diameter_nm_median_crosscheck': sigma_med,
        'per_particle_area_px': [ (int(r['area_px']) if r['area_px'] is not None else None) for r in primary],
        'per_particle_area_nm2': [float(r['area']) for r in primary],
        'per_particle_solidity': [ (float(r['solidity']) if r['solidity'] is not None else None) for r in primary],
        'per_particle_circularity': [ (float(r['circularity']) if r['circularity'] is not None else None) for r in primary],
        'per_particle_eccentricity': [ (float(r['eccentricity']) if r['eccentricity'] is not None else None) for r in primary],
        'measured_boundary_type': measured_boundary_type,
        'seed_to_instance_match_status': {
            'matched': int(n_matched),
            'recall_miss': int(n_recall_miss),
            'under_segmented': int(n_underseg),
        },
        'fraction_particles_using_sigma_fallback': frac_fallback,
        'histogram_counts': hist_counts.tolist(),
        'histogram_bin_edges': hist_edges.tolist(),
        'particle_count_interior_with_uncertainty': {
            'value': int(n_interior),
            'uncertainty_frac': 0.20,
            'range': [int(round(n_interior * 0.8)), int(round(n_interior * 1.2))],
        },
    },
    'quality_metrics': {
        'fraction_particles_using_sigma_fallback': frac_fallback,
        'fraction_border_excluded': (float(n_border_excluded) / float(n_total)) if n_total else None,
        'solidity_median': (float(np.median(solids)) if solids.size > 0 else None),
        'circularity_median': (float(np.median(circs)) if circs.size > 0 else None),
        'eccentricity_median': (float(np.median(eccs)) if eccs.size > 0 else None),
        'crosscheck_note': crosscheck_note,
        'sam_note': sam_note,
        'note': 'Report count with ~20% uncertainty relative to visual estimate. Light QC (solidity/circularity/eccentricity) is REPORTING-ONLY and not used for exclusion (avoids clean-subset bias).',
    },
    'summary': summary,
    'saved_arrays': saved_arrays,
}

print(f'IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}')
