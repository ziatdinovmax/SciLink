import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import measure
from scipy import ndimage as ndi

# ---------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------
image = np.load('data.npy')
print('image shape', image.shape, 'dtype', image.dtype)

# metadata
try:
    with open('metadata.json') as f:
        metadata = json.load(f)
except Exception as e:
    metadata = {}
    print('metadata load failed:', e)

# ---------------------------------------------------------------
# 2. CALIBRATION
# ---------------------------------------------------------------
from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
px = resolve_pixel_size_nm(metadata, image.shape)
if px is not None:
    pixel_size_nm = float(px['x'])
    calib_source = px.get('source', 'metadata')
    uncalibrated = False
else:
    pixel_size_nm = None
    calib_source = 'none'
    uncalibrated = True
print('pixel_size_nm', pixel_size_nm, 'source', calib_source)

# ---------------------------------------------------------------
# 3. GRAYSCALE (luminance)
# ---------------------------------------------------------------
if image.ndim == 3 and image.shape[2] >= 3:
    gray = rgb2gray(image[:, :, :3])  # luminance, float 0..1
elif image.ndim == 3:
    gray = image[:, :, 0].astype(float) / 255.0
else:
    gray = image.astype(float) / 255.0
gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)

H, W = gray.shape

# ---------------------------------------------------------------
# Helper for pixel->nm conversions
# ---------------------------------------------------------------
def px_to_nm(v):
    return v * pixel_size_nm if pixel_size_nm is not None else v

# Disc radius scale: radii ~40-90 px -> sigma ~ r/1.414 -> ~28-64 px
MIN_SIGMA = 28.0
MAX_SIGMA = 64.0
TYPICAL_RADIUS_PX = 55.0  # ~1 disc radius for dedup / exclusion

# ---------------------------------------------------------------
# 4. ROUND (VATERITE) DETECTION with log_blob_detect
# ---------------------------------------------------------------
from scilink.skills._shared.log_blob import log_blob_detect

round_thr_rel = 0.12
round_res = log_blob_detect(
    gray_u8,
    pixel_size_nm=pixel_size_nm,
    polarity='bright',
    min_sigma=MIN_SIGMA,
    max_sigma=MAX_SIGMA,
    num_sigma=12,
    threshold_rel=round_thr_rel,
    overlap=0.5,
    exclude_border=False,
)
round_objs = round_res.get('objects', []) or []
print('raw round detections:', len(round_objs), 'note:', round_res.get('note'))

# ---------------------------------------------------------------
# 5. SINGLE SIMPLE CENTER-DISTANCE DEDUP
# ---------------------------------------------------------------
# Sort by diameter (proxy for stronger response since blob_log threshold not returned)
# and merge any two whose centers are within ~1 disc radius, keep first (stronger).
MIN_SEP = TYPICAL_RADIUS_PX  # ~55 px

def center_dedup(objs, min_sep):
    # order by diameter descending as strength proxy
    order = sorted(range(len(objs)), key=lambda i: -(objs[i].get('diameter_nm') or 0))
    kept = []
    for idx in order:
        o = objs[idx]
        cy, cx = o['cy'], o['cx']
        ok = True
        for k in kept:
            if (cy - k['cy'])**2 + (cx - k['cx'])**2 < min_sep**2:
                ok = False
                break
        if ok:
            kept.append(o)
    return kept

round_dedup = center_dedup(round_objs, MIN_SEP)
print('after dedup:', len(round_dedup))

# ---------------------------------------------------------------
# 6. REAL INSTANCE MASKS via SAM (fallback: connected component)
# ---------------------------------------------------------------
sam_available = True
sam_result = None
try:
    from scilink.skills._shared.sam import run_sam_analysis
    # area bounds from radii 40-90 px: area ~ pi*r^2 => ~5000 .. ~25000
    min_area = int(np.pi * (30**2) * 0.5)
    max_area = int(np.pi * (110**2))
    sam_result = run_sam_analysis(gray_u8, params={
        'sam_parameters': 'default',
        'min_area': min_area,
        'max_area': max_area,
        'pruning_iou_threshold': 0.5,
    })
    print('SAM total_count:', sam_result.get('total_count'))
except Exception as e:
    sam_available = False
    print('SAM unavailable:', repr(e))

sam_masks = []
if sam_available and sam_result is not None:
    for p in sam_result.get('particles', []):
        m = np.array(p['mask'], dtype=bool)
        if m.shape == gray.shape:
            sam_masks.append(m)


def mask_for_center(cy, cx, bbox):
    """Return a real binary mask (full-image bool) containing center, or None."""
    icy, icx = int(round(cy)), int(round(cx))
    icy = min(max(icy, 0), H - 1)
    icx = min(max(icx, 0), W - 1)
    # 6a: try SAM mask containing center
    if sam_masks:
        best = None
        best_area = None
        for m in sam_masks:
            if m[icy, icx]:
                a = int(m.sum())
                if best is None or a < best_area:
                    best = m
                    best_area = a
        if best is not None:
            return best
    # 6b: fallback Otsu on local crop, connected component containing center
    y0, x0, y1, x1 = bbox
    pad = 10
    y0 = max(0, int(y0) - pad); x0 = max(0, int(x0) - pad)
    y1 = min(H, int(y1) + pad); x1 = min(W, int(x1) + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    try:
        t = threshold_otsu(crop)
    except Exception:
        return None
    bw = crop > t  # bright bodies
    lbl, n = ndi.label(bw)
    ly, lx = icy - y0, icx - x0
    if not (0 <= ly < bw.shape[0] and 0 <= lx < bw.shape[1]):
        return None
    comp_id = lbl[ly, lx]
    if comp_id == 0:
        return None
    full = np.zeros((H, W), dtype=bool)
    full[y0:y1, x0:x1] = (lbl == comp_id)
    return full


def mask_props(mask):
    lbl = mask.astype(np.int32)
    props = measure.regionprops(lbl)
    if not props:
        return None
    return props[0]


def is_degenerate(p, mask):
    # HARD DROP exactly ecc=0 AND sol=1, or fills bbox as rectangle
    ecc = p.eccentricity
    sol = p.solidity
    if abs(ecc - 0.0) < 1e-9 and abs(sol - 1.0) < 1e-9:
        return True
    minr, minc, maxr, maxc = p.bbox
    bbox_area = (maxr - minr) * (maxc - minc)
    if bbox_area > 0 and p.area / bbox_area > 0.985:
        return True
    return False


# ---------------------------------------------------------------
# Build accepted round set with SHAPE GATE (step 6 masks + step 6 gate)
# ---------------------------------------------------------------
ECC_MAX = 0.85
SOL_MIN = 0.7
if pixel_size_nm is not None:
    DIAM_MIN_NM, DIAM_MAX_NM = 6500.0, 14500.0
else:
    DIAM_MIN_NM, DIAM_MAX_NM = None, None

accepted_round = []  # dicts with mask, props, center, diameter
round_label = np.zeros((H, W), dtype=np.int32)

for o in round_dedup:
    cy, cx = o['cy'], o['cx']
    bbox = o.get('bbox', (max(0, cy - 60), max(0, cx - 60), min(H, cy + 60), min(W, cx + 60)))
    m = mask_for_center(cy, cx, bbox)
    if m is None or m.sum() == 0:
        continue  # HARD DROP no center-connected component
    p = mask_props(m)
    if p is None:
        continue
    if is_degenerate(p, m):
        continue  # HARD DROP degenerate idealized tile
    ecc = float(p.eccentricity)
    sol = float(p.solidity)
    equiv_d_px = float(p.equivalent_diameter)
    equiv_d_nm = px_to_nm(equiv_d_px)
    # shape gate
    if ecc > ECC_MAX:
        continue
    if sol < SOL_MIN:
        continue
    if DIAM_MIN_NM is not None:
        if not (DIAM_MIN_NM <= equiv_d_nm <= DIAM_MAX_NM):
            continue
    accepted_round.append({
        'cy': float(cy), 'cx': float(cx),
        'mask': m,
        'eccentricity': ecc,
        'solidity': sol,
        'equiv_diameter_px': equiv_d_px,
        'equiv_diameter_nm': equiv_d_nm,
    })

# assign labels (avoid overwriting by later overlaps: keep first)
for i, a in enumerate(accepted_round, start=1):
    round_label[a['mask'] & (round_label == 0)] = i

print('accepted round discs:', len(accepted_round))

# ---------------------------------------------------------------
# 7. DARK PASS (elongated debris) - conservative
# ---------------------------------------------------------------
dark_thr_rel = 0.30  # conservative to avoid rim-scatter over-fire
dark_res = log_blob_detect(
    gray_u8,
    pixel_size_nm=pixel_size_nm,
    polarity='dark',
    min_sigma=6.0,
    max_sigma=40.0,
    num_sigma=10,
    threshold_rel=dark_thr_rel,
    overlap=0.5,
    exclude_border=False,
)
dark_objs = dark_res.get('objects', []) or []
print('raw dark detections:', len(dark_objs), 'note:', dark_res.get('note'))

# exclusion helpers vs accepted round centers/masks
round_centers = np.array([[a['cy'], a['cx']] for a in accepted_round]) if accepted_round else np.zeros((0, 2))
EXCL_RADIUS = TYPICAL_RADIUS_PX

def center_in_round(cy, cx):
    icy, icx = int(round(cy)), int(round(cx))
    if 0 <= icy < H and 0 <= icx < W and round_label[icy, icx] > 0:
        return True
    if round_centers.shape[0] > 0:
        d2 = (round_centers[:, 0] - cy)**2 + (round_centers[:, 1] - cx)**2
        if np.any(d2 < EXCL_RADIUS**2):
            return True
    return False

accepted_debris = []
debris_label = np.zeros((H, W), dtype=np.int32)
DARK_ECC_MIN = 0.7

for o in dark_objs:
    cy, cx = o['cy'], o['cx']
    if center_in_round(cy, cx):
        continue  # exclude debris on/near round discs
    bbox = o.get('bbox', (max(0, cy - 30), max(0, cx - 30), min(H, cy + 30), min(W, cx + 30)))
    # dark object mask: Otsu on local crop, take DARK connected component containing center
    y0, x0, y1, x1 = bbox
    pad = 8
    y0 = max(0, int(y0) - pad); x0 = max(0, int(x0) - pad)
    y1 = min(H, int(y1) + pad); x1 = min(W, int(x1) + pad)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        continue
    try:
        t = threshold_otsu(crop)
    except Exception:
        continue
    bw = crop < t  # dark objects
    lbl, n = ndi.label(bw)
    icy, icx = int(round(cy)) - y0, int(round(cx)) - x0
    if not (0 <= icy < bw.shape[0] and 0 <= icx < bw.shape[1]):
        continue
    comp_id = lbl[icy, icx]
    if comp_id == 0:
        continue  # no coherent connected mask
    comp = (lbl == comp_id)
    # require single coherent connected mask (it is one component by construction)
    full = np.zeros((H, W), dtype=bool)
    full[y0:y1, x0:x1] = comp
    p = mask_props(full)
    if p is None:
        continue
    if is_degenerate(p, full):
        continue
    ecc = float(p.eccentricity)
    if ecc <= DARK_ECC_MIN:
        continue  # require elongated
    accepted_debris.append({
        'cy': float(cy), 'cx': float(cx),
        'mask': full,
        'eccentricity': ecc,
    })

for i, d in enumerate(accepted_debris, start=1):
    debris_label[d['mask'] & (debris_label == 0)] = i

# verify zero debris on round discs
debris_on_round = 0
for d in accepted_debris:
    icy, icx = int(round(d['cy'])), int(round(d['cx']))
    if 0 <= icy < H and 0 <= icx < W and round_label[icy, icx] > 0:
        debris_on_round += 1
print('accepted debris:', len(accepted_debris), 'debris_on_round:', debris_on_round)

# ---------------------------------------------------------------
# 9. STATISTICS (round discs, from real masks)
# ---------------------------------------------------------------
diams_nm = [a['equiv_diameter_nm'] for a in accepted_round]
eccs = [a['eccentricity'] for a in accepted_round]
sols = [a['solidity'] for a in accepted_round]

if diams_nm:
    diameter_median = float(np.median(diams_nm))
    diameter_mean = float(np.mean(diams_nm))
    diameter_std = float(np.std(diams_nm))
    hist_counts, hist_edges = np.histogram(diams_nm, bins=10)
    diameter_histogram = {
        'counts': hist_counts.tolist(),
        'bin_edges': hist_edges.tolist(),
    }
else:
    diameter_median = diameter_mean = diameter_std = None
    diameter_histogram = {'counts': [], 'bin_edges': []}

debris_eccs = [d['eccentricity'] for d in accepted_debris]

# ---------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0, 0]
ax.imshow(image[:, :, :3] if image.ndim == 3 else gray, cmap=None if image.ndim == 3 else 'gray')
ax.set_title('Original image')
ax.axis('off')

# Overlay: round masks + debris masks + contours
ax = axes[0, 1]
if image.ndim == 3:
    ax.imshow(image[:, :, :3])
else:
    ax.imshow(gray, cmap='gray')

# round overlay (green)
round_overlay = np.zeros((H, W, 4))
round_overlay[round_label > 0] = [0, 1, 0, 0.35]
ax.imshow(round_overlay)
for a in accepted_round:
    cs = measure.find_contours(a['mask'].astype(float), 0.5)
    for c in cs:
        ax.plot(c[:, 1], c[:, 0], '-', color='lime', linewidth=1.2)
    ax.plot(a['cx'], a['cy'], '+', color='yellow', markersize=6)
# debris overlay (red)
debris_overlay = np.zeros((H, W, 4))
debris_overlay[debris_label > 0] = [1, 0, 0, 0.4]
ax.imshow(debris_overlay)
for d in accepted_debris:
    cs = measure.find_contours(d['mask'].astype(float), 0.5)
    for c in cs:
        ax.plot(c[:, 1], c[:, 0], '-', color='red', linewidth=1.2)
ax.set_title('Segmentation overlay: %d round (green), %d debris (red)' % (len(accepted_round), len(accepted_debris)))
ax.axis('off')

# raw round detections vs dedup
ax = axes[1, 0]
if image.ndim == 3:
    ax.imshow(image[:, :, :3])
else:
    ax.imshow(gray, cmap='gray')
for o in round_objs:
    ax.plot(o['cx'], o['cy'], 'o', mfc='none', mec='cyan', markersize=8)
for a in accepted_round:
    ax.plot(a['cx'], a['cy'], 'x', color='yellow', markersize=8)
ax.set_title('Raw round detections (cyan, n=%d) vs accepted (yellow, n=%d)' % (len(round_objs), len(accepted_round)))
ax.axis('off')

# diameter histogram
ax = axes[1, 1]
if diams_nm:
    ax.hist(diams_nm, bins=10, color='steelblue', edgecolor='k')
    unit = 'nm' if pixel_size_nm is not None else 'px'
    ax.set_xlabel('Equivalent diameter (%s)' % unit)
    ax.set_ylabel('Count')
    ax.axvline(diameter_median, color='r', ls='--', label='median=%.0f' % diameter_median)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No round discs', ha='center')
ax.set_title('Round disc diameter distribution')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---------------------------------------------------------------
# SAVE ARRAYS
# ---------------------------------------------------------------
np.save('round_labels.npy', round_label)
np.save('debris_labels.npy', debris_label)

# ---------------------------------------------------------------
# RESULTS JSON
# ---------------------------------------------------------------
summary_notes = []
summary_notes.append('Reverted to log_blob_detect (bright, sigma %d-%d px ~ radii 40-90 px, threshold_rel=%.2f).' % (int(MIN_SIGMA), int(MAX_SIGMA), round_thr_rel))
summary_notes.append('Single center-distance dedup at ~%d px (1 disc radius).' % int(MIN_SEP))
if not sam_available:
    summary_notes.append('SAM unavailable; used Otsu connected-component fallback for real masks.')
else:
    summary_notes.append('SAM used for real instance masks (fallback Otsu CC where center not covered).')
summary_notes.append('Dark debris pass conservative (threshold_rel=%.2f), ecc>%.1f, excluded near/on round discs.' % (dark_thr_rel, DARK_ECC_MIN))
if uncalibrated:
    summary_notes.append('UNCALIBRATED: diameters reported in pixels.')

results = {
    'analysis_type': 'Single-image vaterite round-disc detection (log_blob_detect + real SAM/Otsu masks) with conservative dark elongated-debris pass',
    'extracted_features': {
        'pixel_size_nm': pixel_size_nm,
        'pixel_size_source': calib_source,
        'uncalibrated': uncalibrated,
        'round_vaterite_count': len(accepted_round),
        'per_particle_equivalent_diameter_nm': [float(x) for x in diams_nm],
        'per_particle_eccentricity': [float(x) for x in eccs],
        'per_particle_solidity': [float(x) for x in sols],
        'diameter_median_nm': diameter_median,
        'diameter_mean_nm': diameter_mean,
        'diameter_std_nm': diameter_std,
        'diameter_histogram': diameter_histogram,
        'dark_debris_count': len(accepted_debris),
        'per_debris_eccentricity': [float(x) for x in debris_eccs],
    },
    'quality_metrics': {
        'raw_round_detections': len(round_objs),
        'round_after_dedup': len(round_dedup),
        'round_accepted': len(accepted_round),
        'raw_dark_detections': len(dark_objs),
        'debris_accepted': len(accepted_debris),
        'debris_on_round_discs': debris_on_round,
        'sam_available': sam_available,
    },
    'summary': ' '.join(summary_notes),
    'saved_arrays': {
        'round_labels.npy': {
            'description': 'Integer label map of accepted round vaterite discs (background=0, %d discs labeled 1..N)' % len(accepted_round),
            'shape': list(round_label.shape),
            'dtype': str(round_label.dtype),
        },
        'debris_labels.npy': {
            'description': 'Integer label map of accepted elongated dark debris (background=0, %d objects labeled 1..N)' % len(accepted_debris),
            'shape': list(debris_label.shape),
            'dtype': str(debris_label.dtype),
        },
    },
}

print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results))
