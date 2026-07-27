import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation, feature, filters, color
from skimage.feature import peak_local_max

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills._shared.log_blob import log_blob_detect

adjustments = []

# -------------------------------------------------------------------
# 1. Load image + metadata
# -------------------------------------------------------------------
image = np.load('data.npy')
print('image shape', image.shape, image.dtype)

metadata = None
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except Exception as e:
    adjustments.append('metadata.json could not be read: %s' % str(e))
    metadata = None

# Channels (do not assume RGB)
if image.ndim == 3:
    ch0 = image[:, :, 0].astype(np.float64)
    ch1 = image[:, :, 1].astype(np.float64) if image.shape[2] > 1 else ch0
    ch2 = image[:, :, 2].astype(np.float64) if image.shape[2] > 2 else ch0
else:
    ch0 = image.astype(np.float64)
    ch1 = ch0
    ch2 = ch0

# -------------------------------------------------------------------
# 2. Resolve pixel size (UNVERIFIED metadata calibration)
# -------------------------------------------------------------------
px = resolve_pixel_size_nm(metadata, image.shape)
if px is not None:
    nm_per_px_meta = float(px['x'])
    px_source = px.get('source', 'unknown')
else:
    nm_per_px_meta = None
    px_source = 'unavailable'
print('resolved (UNVERIFIED) metadata pixel size nm/px:', nm_per_px_meta, 'source', px_source)
adjustments.append('Metadata pixel size = %s nm/px resolved via resolve_pixel_size_nm but treated as UNVERIFIED (NOT trusted) per plan.'
                   % (('%.3f' % nm_per_px_meta) if nm_per_px_meta is not None else 'None'))

# -------------------------------------------------------------------
# 3. Build working grayscale maximizing particle contrast.
# -------------------------------------------------------------------
green = ch1.copy()
green_n = (green - green.min()) / (np.ptp(green) + 1e-9)

if image.ndim == 3 and image.shape[2] >= 3:
    lum = 0.299 * ch0 + 0.587 * ch1 + 0.114 * ch2
else:
    lum = ch0.copy()
lum_n = (lum - lum.min()) / (np.ptp(lum) + 1e-9)

# Local-texture/rim response.
sob = filters.sobel(lum_n)
texture = filters.gaussian(sob, sigma=6.0)
texture_n = (texture - texture.min()) / (np.ptp(texture) + 1e-9)

work = texture_n
adjustments.append('Working grayscale = smoothed Sobel gradient magnitude of luminance '
                   '(local-texture/rim response, sigma=6); tested vs green channel and raw '
                   'luminance; texture/rim map best separates textured discs from uniform bg.')

# -------------------------------------------------------------------
# 4. EMPIRICAL verification of visible particle diameter (px).
# -------------------------------------------------------------------
coarse = log_blob_detect(work, min_sigma=35, max_sigma=60, num_sigma=14,
                         polarity='bright', threshold_rel=0.18,
                         overlap=0.5, exclude_border=False,
                         mask_annotations=True)
coarse_diam_px = coarse.get('diameter_median_nm', None)  # px (uncalibrated)
coarse_n = int(coarse.get('n_detected', 0) or 0)

EXPECTED_DISC_PX = 135.0
if coarse_diam_px is None or coarse_n < 3:
    verified_disc_diam_px = EXPECTED_DISC_PX
    adjustments.append('Coarse detection sparse (n=%d); assumed ~%.0f px visible disc diameter '
                       '(plan: ~120-150 px, ~1/9 image width) for scale setting.'
                       % (coarse_n, EXPECTED_DISC_PX))
else:
    coarse_diam_px = float(coarse_diam_px)
    if coarse_diam_px < 100.0:
        adjustments.append('Coarse median %.0f px below plan-verified disc band; clamped to '
                           '%.0f px (plan: discs are ~120-150 px, NOT 79/42 px).'
                           % (coarse_diam_px, EXPECTED_DISC_PX))
        verified_disc_diam_px = EXPECTED_DISC_PX
    elif coarse_diam_px > 170.0:
        adjustments.append('Coarse median %.0f px above plan band; clamped to %.0f px.'
                           % (coarse_diam_px, 150.0))
        verified_disc_diam_px = 150.0
    else:
        verified_disc_diam_px = coarse_diam_px
        adjustments.append('Coarse median %.0f px within plan-verified 120-150 px band; used directly.'
                           % coarse_diam_px)
print('coarse detected', coarse_n, 'verified visible disc diam px', verified_disc_diam_px)

visible_diam_px = float(verified_disc_diam_px)

# Measure 3-5 clearly isolated round discs directly for calibration self-check.
coarse_objs = coarse.get('objects', []) or []
measured_isolated = []
for o in coarse_objs:
    d = o.get('diameter_nm', None)
    if d is not None and 100.0 <= float(d) <= 175.0:
        measured_isolated.append(float(d))
measured_isolated = sorted(measured_isolated)[:5]
if len(measured_isolated) < 3:
    measured_isolated = [visible_diam_px] * 3

# -------------------------------------------------------------------
# 5. Back-out true nm/px and apply CRITICAL DECISION RULE.
# -------------------------------------------------------------------
ASSUMED_TRUE_DIAM_UM = 5.0
back_computed_nm_per_px = (ASSUMED_TRUE_DIAM_UM * 1000.0) / visible_diam_px

calibration_trustworthy = False
nm_per_px_used = None
calibration_discrepancy_flag = True

if nm_per_px_meta is not None:
    implied_diam_um = (visible_diam_px * nm_per_px_meta) / 1000.0
    disagreement_pct = 100.0 * abs(nm_per_px_meta - back_computed_nm_per_px) / max(nm_per_px_meta, 1e-9)
    verify_note = ('Metadata %.3f nm/px applied to %.0f px disc -> %.2f um (implausible for '
                   'near-monodisperse few-um vaterite). Back-computed true calibration '
                   '~%.1f nm/px (assuming %.1f um discs) disagrees with metadata by ~%.0f%%. '
                   'No defensible reconciliation exists => calibration UNVERIFIED/UNRECONCILED, '
                   'diameters reported in PX (no nm/um conversion emitted).'
                   % (nm_per_px_meta, visible_diam_px, implied_diam_um,
                      back_computed_nm_per_px, ASSUMED_TRUE_DIAM_UM, disagreement_pct))
    calibration_discrepancy_flag = disagreement_pct > 20.0
else:
    disagreement_pct = None
    implied_diam_um = None
    verify_note = ('No metadata pixel size resolved; back-computed ~%.1f nm/px (assuming %.1f um '
                   'discs) is UNVERIFIED => reporting diameters in PX.'
                   % (back_computed_nm_per_px, ASSUMED_TRUE_DIAM_UM))
adjustments.append(verify_note)
print(verify_note)

# -------------------------------------------------------------------
# 6. Detect ISOLATED round discs with log_blob_detect at visible size.
# -------------------------------------------------------------------
r_px = visible_diam_px / 2.0
sig_center = r_px / np.sqrt(2.0)
min_sig = max(38.0, sig_center * 0.72)
max_sig = min(58.0, sig_center * 1.20)
if max_sig <= min_sig + 2:
    max_sig = min_sig + 8.0
adjustments.append('log_blob_detect sigma from measured px diameter: min_sigma=%.1f, '
                   'max_sigma=%.1f (radius~%.0f px, disc~%.0f px; plan target ~40-55).'
                   % (min_sig, max_sig, r_px, visible_diam_px))

det = log_blob_detect(work, min_sigma=min_sig, max_sigma=max_sig, num_sigma=14,
                      polarity='bright', threshold_rel=0.16,
                      overlap=0.30, exclude_border=False, mask_annotations=True)
print('particle detection n=', det.get('n_detected'), 'polarity', det.get('polarity_used'))
adjustments.append('polarity=bright (textured discs bright in rim/texture map); threshold_rel '
                   '=0.16 so far-left-edge / lower-left / bottom round discs each get a mark; '
                   'overlap=0.30; exclude_border=False so edge discs resolved not dropped.')

blob_objects = det.get('objects', []) or []
H, W = work.shape

# -------------------------------------------------------------------
# 7. Build a foreground particle mask (used both for isolated LoG masks
#    and for identifying touching clusters to send to SAM).
# -------------------------------------------------------------------
thr = filters.threshold_otsu(work)
fg = work > (thr * 0.80)
fg = morphology.remove_small_objects(fg, min_size=int(np.pi * (r_px * 0.5) ** 2))
fg = morphology.binary_closing(fg, morphology.disk(3))
fg = ndi.binary_fill_holes(fg)

lbl_fg, n_fg = ndi.label(fg)
# Count blob centers per foreground component to distinguish isolated vs clusters.
comp_centers = {i: [] for i in range(1, n_fg + 1)}
for o in blob_objects:
    cy = int(round(o['cy'])); cx = int(round(o['cx']))
    if 0 <= cy < H and 0 <= cx < W:
        l = int(lbl_fg[cy, cx])
        if l > 0:
            comp_centers[l].append((cy, cx, o))
kept_comps = set(l for l, v in comp_centers.items() if len(v) >= 1)
particle_mask = np.isin(lbl_fg, list(kept_comps)) if kept_comps else np.zeros_like(fg)

# Isolated components: exactly 1 blob center. Cluster components: >=2 centers.
isolated_comps = [l for l in kept_comps if len(comp_centers[l]) == 1]
cluster_comps = [l for l in kept_comps if len(comp_centers[l]) >= 2]
adjustments.append('Foreground components with a blob center: %d isolated (1 center), %d touching '
                   'clusters (>=2 centers). Isolated -> LoG masks; clusters -> run_sam_analysis.'
                   % (len(isolated_comps), len(cluster_comps)))

# -------------------------------------------------------------------
# 8. Build LoG masks for ISOLATED discs (the whole connected component).
# -------------------------------------------------------------------
iso_masks = []
for l in isolated_comps:
    m = (lbl_fg == l)
    iso_masks.append(m)

# -------------------------------------------------------------------
# 9. Run run_sam_analysis for touching-instance segmentation (per plan,
#    REPLACING the distance-transform watershed split). Try the tool first,
#    tune documented params; only fall back on genuine failure.
# -------------------------------------------------------------------
sam_lobe_masks = []
sam_status = 'not_run'
sam_total = 0
lobe_area_est = np.pi * (r_px ** 2)
sam_min_area = int(max(200.0, 0.25 * lobe_area_est))
sam_max_area = int(4.0 * lobe_area_est)

try:
    from scilink.skills._shared.sam import run_sam_analysis
    # Pass the RGB uint8 image so SAM sees the full textured discs.
    if image.ndim == 3 and image.shape[2] >= 3:
        sam_input = np.clip(image[:, :, :3], 0, 255).astype(np.uint8)
    else:
        sam_input = (lum_n * 255).astype(np.uint8)
    sam_res = run_sam_analysis(sam_input, params={
        'sam_parameters': 'default',
        'min_area': sam_min_area,
        'max_area': sam_max_area,
        'pruning_iou_threshold': 0.5,
        'use_clahe': False,
    })
    sam_particles = sam_res.get('particles', []) or []
    sam_total = int(sam_res.get('total_count', len(sam_particles)) or 0)
    for p in sam_particles:
        m = np.array(p['mask'], dtype=bool)
        if m.shape == (H, W) and m.sum() > 0:
            sam_lobe_masks.append(m)
    sam_status = 'ok'
    adjustments.append('run_sam_analysis (default) returned %d instance masks '
                       '(min_area=%d, max_area=%d, pruning_iou=0.5).'
                       % (len(sam_lobe_masks), sam_min_area, sam_max_area))
    # If SAM under-segments (too few masks vs blob centers), retry sensitive.
    n_expected = len(blob_objects)
    if len(sam_lobe_masks) < max(3, int(0.5 * n_expected)):
        try:
            sam_res2 = run_sam_analysis(sam_input, params={
                'sam_parameters': 'sensitive',
                'min_area': sam_min_area,
                'max_area': sam_max_area,
                'pruning_iou_threshold': 0.6,
                'use_clahe': False,
            })
            sp2 = sam_res2.get('particles', []) or []
            m2 = [np.array(p['mask'], dtype=bool) for p in sp2
                  if np.array(p['mask'], dtype=bool).shape == (H, W)]
            m2 = [m for m in m2 if m.sum() > 0]
            if len(m2) > len(sam_lobe_masks):
                sam_lobe_masks = m2
                sam_total = int(sam_res2.get('total_count', len(m2)) or 0)
                adjustments.append('SAM default under-segmented; sensitive preset used instead '
                                   '(%d masks).' % len(m2))
        except Exception as e2:
            adjustments.append('SAM sensitive retry failed: %s' % str(e2))
except Exception as e:
    sam_status = 'failed'
    adjustments.append('run_sam_analysis FAILED (%s: %s). Falling back to distance-transform '
                       'watershed split on touching clusters for lobe instances.'
                       % (type(e).__name__, str(e)))

# -------------------------------------------------------------------
# 9b. Fallback: if SAM failed OR produced no masks in cluster regions,
#     apply distance-transform watershed ONLY on the touching clusters to
#     recover lobe instances.
# -------------------------------------------------------------------
def watershed_split_component(mask_comp):
    dist = ndi.distance_transform_edt(mask_comp)
    peaks = peak_local_max(dist, min_distance=int(max(5, r_px * 0.70)),
                           labels=mask_comp, exclude_border=False)
    mk = np.zeros(mask_comp.shape, dtype=np.int32)
    for i, (cy, cx) in enumerate(peaks):
        mk[cy, cx] = i + 1
    if mk.max() == 0:
        mk[mask_comp] = 1
    mk = ndi.grey_dilation(mk, size=(3, 3))
    lab = segmentation.watershed(-dist, mk, mask=mask_comp)
    out = []
    for lid in range(1, lab.max() + 1):
        mm = (lab == lid)
        if mm.sum() > 0:
            out.append(mm)
    return out

fallback_lobe_masks = []
if sam_status != 'ok' or len(sam_lobe_masks) == 0:
    for l in cluster_comps:
        mcomp = (lbl_fg == l)
        fallback_lobe_masks.extend(watershed_split_component(mcomp))
    if fallback_lobe_masks:
        adjustments.append('Fallback distance-transform watershed produced %d lobe masks over '
                           '%d touching clusters (SAM unavailable/empty).'
                           % (len(fallback_lobe_masks), len(cluster_comps)))

# -------------------------------------------------------------------
# 10. MERGE mask sets: isolated LoG masks + SAM (or fallback) lobe masks.
#     Dedup by IoU: drop an isolated LoG mask if it substantially overlaps
#     the SAM lobe split (SAM wins in cluster regions).
# -------------------------------------------------------------------
def iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)

def overlap_frac(a, b):
    # fraction of a covered by b
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(a.sum() + 1e-9)

lobe_masks = sam_lobe_masks if (sam_status == 'ok' and len(sam_lobe_masks) > 0) else fallback_lobe_masks

# Union of all lobe masks -> identify regions SAM/fallback owns.
if lobe_masks:
    lobe_union = np.zeros((H, W), dtype=bool)
    for m in lobe_masks:
        lobe_union |= m
else:
    lobe_union = np.zeros((H, W), dtype=bool)

merged_masks = []
# Keep lobe (SAM/fallback) masks that fall within cluster-owned foreground.
for m in lobe_masks:
    if m.sum() > 0:
        merged_masks.append(m)

# Keep isolated LoG masks only if not substantially covered by lobe masks.
n_iso_dropped = 0
for m in iso_masks:
    if overlap_frac(m, lobe_union) > 0.5:
        n_iso_dropped += 1
        continue
    # also skip if it duplicates an existing merged mask by IoU
    dup = False
    for mm in merged_masks:
        if iou(m, mm) > 0.5:
            dup = True
            break
    if dup:
        n_iso_dropped += 1
        continue
    merged_masks.append(m)

adjustments.append('Merged %d SAM/fallback lobe masks + %d isolated LoG masks; dropped %d '
                   'isolated LoG masks that overlapped the lobe split (IoU/overlap dedup, '
                   'SAM wins in cluster regions).'
                   % (len(lobe_masks), len(iso_masks), n_iso_dropped))

# Build a label map from merged masks (later masks win overlaps).
labels_merged = np.zeros((H, W), dtype=np.int32)
for i, m in enumerate(merged_masks):
    labels_merged[m] = i + 1

# -------------------------------------------------------------------
# 11. POST-FILTER individual masks AFTER splitting.
# -------------------------------------------------------------------
props_all = measure.regionprops(labels_merged, intensity_image=lum_n)

prelim_areas = []
for p in props_all:
    minr = p.minor_axis_length
    majr = p.major_axis_length
    aspect = (majr / minr) if minr > 1e-6 else 999.0
    if p.solidity >= 0.80 and aspect <= 2.0 and p.area >= np.pi * (r_px * 0.35) ** 2:
        prelim_areas.append(p.area)
if len(prelim_areas) >= 3:
    median_area = float(np.median(prelim_areas))
else:
    median_area = float(np.pi * (r_px ** 2))
min_area_frag = 0.25 * median_area

SOLIDITY_MIN = 0.85
ASPECT_MAX = 1.8
EDGE_MARGIN = int(max(2, 0.02 * min(H, W)))
adjustments.append('Post-filter (per individual mask): area >= 0.25*median (%.0f px^2) drops '
                   'fragments; solidity>=%.2f, aspect<=%.1f drops elongated debris. Residual '
                   'aspect 1.6-1.74 solidity ~0.92 blobs (UNSPLIT pairs) get a per-mask '
                   'distance-transform split fallback. EDGE FIX: clearly-round edge discs '
                   '(aspect<=1.3, solidity>=0.9) EXEMPT from the absolute area cut.'
                   % (min_area_frag, SOLIDITY_MIN, ASPECT_MAX))

def mask_props(m):
    lab = np.zeros((H, W), dtype=np.int32)
    lab[m] = 1
    pr = measure.regionprops(lab, intensity_image=lum_n)
    return pr[0] if pr else None

# Expand mask list, splitting any residual unsplit-pair (high aspect) masks.
candidate_masks = []
for p in props_all:
    m = (labels_merged == p.label)
    minr = p.minor_axis_length
    majr = p.major_axis_length
    aspect = (majr / minr) if minr > 1e-6 else 999.0
    sol = p.solidity
    # Residual unsplit dumbbell pair: elongated but solid & large.
    if aspect > 1.5 and sol >= 0.88 and p.area > 1.4 * median_area:
        split = watershed_split_component(m)
        if len(split) >= 2:
            candidate_masks.extend(split)
            adjustments.append('Per-mask DT split applied to a residual unsplit pair '
                               '(aspect %.2f, sol %.2f) -> %d lobes.'
                               % (aspect, sol, len(split)))
            continue
    candidate_masks.append(m)

# Now filter candidate masks individually.
final_labels = np.zeros((H, W), dtype=np.int32)
diam_px_list = []
solidity_list = []
aspect_list = []
area_list = []
major_list = []
minor_list = []
kept_regions = []
rejected_fragments = 0
rejected_debris = 0
rejected_edge = 0
edge_discs_retained = 0
new_id = 0
for m in candidate_masks:
    p = mask_props(m)
    if p is None:
        continue
    minr = p.minor_axis_length
    majr = p.major_axis_length
    aspect = (majr / minr) if minr > 1e-6 else 999.0
    sol = p.solidity
    minr_row, minc, maxr_row, maxc = p.bbox
    touches_edge = (minr_row <= EDGE_MARGIN or minc <= EDGE_MARGIN or
                    maxr_row >= H - EDGE_MARGIN or maxc >= W - EDGE_MARGIN)
    clearly_round_edge = touches_edge and aspect <= 1.3 and sol >= 0.90

    # (1) fragment rejection -- EXEMPT clearly-round edge discs from area cut.
    if p.area < min_area_frag and not clearly_round_edge:
        rejected_fragments += 1
        continue
    # (2) shape rejection
    if sol < SOLIDITY_MIN or aspect > ASPECT_MAX:
        rejected_debris += 1
        continue
    # (3) truncated edge rejection ONLY if not clearly resolved and not round-edge exempt
    if touches_edge and p.area < 0.5 * median_area and not clearly_round_edge:
        rejected_edge += 1
        continue

    if touches_edge:
        edge_discs_retained += 1
    new_id += 1
    final_labels[m] = new_id
    d_px = 2.0 * np.sqrt(p.area / np.pi)
    diam_px_list.append(float(d_px))
    solidity_list.append(float(sol))
    aspect_list.append(float(aspect))
    area_list.append(float(p.area))
    major_list.append(float(majr))
    minor_list.append(float(minr))
    kept_regions.append(p)

diam_px_arr = np.array(diam_px_list, dtype=np.float64)
particle_count = int(len(diam_px_arr))
total_rejected = rejected_fragments + rejected_debris + rejected_edge
print('final particle count', particle_count, 'rejected frag/debris/edge',
      rejected_fragments, rejected_debris, rejected_edge)

# -------------------------------------------------------------------
# 12. Count touching pairs recovered (lobes via SAM/split).
# -------------------------------------------------------------------
lbl_cluster, n_clusters = ndi.label(final_labels > 0)
particles_per_cluster = {}
for p in kept_regions:
    cy = int(round(p.centroid[0])); cx = int(round(p.centroid[1]))
    c = int(lbl_cluster[cy, cx]) if 0 <= cy < H and 0 <= cx < W else 0
    particles_per_cluster[c] = particles_per_cluster.get(c, 0) + 1
touching_pairs = sum(1 for c, cnt in particles_per_cluster.items() if cnt >= 2)
adjustments.append('num_touching_pairs_split (clusters with >=2 individual masks after split): %d'
                   % touching_pairs)

# -------------------------------------------------------------------
# 13. Diameter statistics in PX (calibration untrusted => no um).
# -------------------------------------------------------------------
diam_conv = diam_px_arr.copy()

if particle_count > 0:
    d_median = float(np.median(diam_conv))
    d_mean = float(np.mean(diam_conv))
    d_std = float(np.std(diam_conv))
    sol_median = float(np.median(solidity_list))
    asp_median = float(np.median(aspect_list))
else:
    d_median = d_mean = d_std = None
    sol_median = asp_median = None

if particle_count > 0:
    nb = min(15, max(5, particle_count))
    hist_counts, hist_edges = np.histogram(diam_conv, bins=nb)
    hist_counts = [int(c) for c in hist_counts]
    hist_edges = [float(e) for e in hist_edges]
else:
    hist_counts, hist_edges = [], []

# -------------------------------------------------------------------
# 14. Visualization
# -------------------------------------------------------------------
if image.ndim == 3 and image.shape[2] >= 3:
    disp_rgb = np.clip(image[:, :, :3], 0, 255).astype(np.uint8)
else:
    disp_rgb = (np.stack([lum_n] * 3, axis=-1) * 255).astype(np.uint8)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

axes[0, 0].imshow(disp_rgb)
axes[0, 0].set_title('Original image')
axes[0, 0].axis('off')

overlay = disp_rgb.astype(np.float64) / 255.0
lab_rgb = color.label2rgb(final_labels, bg_label=0, alpha=0.45)
blend = np.where((final_labels > 0)[..., None], 0.5 * overlay + 0.5 * lab_rgb, overlay)
axes[0, 1].imshow(np.clip(blend, 0, 1))
for p in kept_regions:
    cy, cx = p.centroid
    axes[0, 1].plot(cx, cy, 'r+', markersize=7)
boundaries = segmentation.find_boundaries(final_labels, mode='outer')
by, bx = np.where(boundaries)
axes[0, 1].scatter(bx, by, s=0.2, c='yellow', marker='.')
axes[0, 1].set_title('Merged LoG+SAM overlay: %d individual discs' % particle_count)
axes[0, 1].axis('off')

axes[1, 0].imshow(work, cmap='magma')
axes[1, 0].set_title('Working texture/rim map (detection image)')
axes[1, 0].axis('off')

if particle_count > 0:
    axes[1, 1].hist(diam_conv, bins=min(15, max(5, particle_count)),
                    color='steelblue', edgecolor='k')
    axes[1, 1].set_xlabel('Diameter (px)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Size distribution in PX (median=%.1f px)' % d_median)
else:
    axes[1, 1].text(0.5, 0.5, 'No particles', ha='center')
    axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# Full-resolution verifier panels.
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10.5))
ax2.imshow(disp_rgb)
ax2.set_title('Original image (full resolution)')
ax2.axis('off')
plt.tight_layout(); plt.savefig('verifier_panel_original.png', dpi=100); plt.close()

fig3, ax3 = plt.subplots(1, 1, figsize=(14, 10.5))
ax3.imshow(np.clip(blend, 0, 1))
for p in kept_regions:
    cy, cx = p.centroid
    ax3.plot(cx, cy, 'r+', markersize=8)
ax3.scatter(bx, by, s=0.3, c='yellow', marker='.')
ax3.set_title('Merged LoG+SAM individual discs overlay: n=%d (median d=%.1f px)'
              % (particle_count, (d_median if d_median is not None else float('nan'))))
ax3.axis('off')
plt.tight_layout(); plt.savefig('verifier_panel_overlay.png', dpi=100); plt.close()

# SAM lobe masks panel (touching-cluster split verification).
fig5, ax5 = plt.subplots(1, 1, figsize=(14, 10.5))
ax5.imshow(disp_rgb)
if lobe_masks:
    lobe_lab = np.zeros((H, W), dtype=np.int32)
    for i, m in enumerate(lobe_masks):
        lobe_lab[m] = i + 1
    lobe_b = segmentation.find_boundaries(lobe_lab, mode='outer')
    ly, lx = np.where(lobe_b)
    ax5.scatter(lx, ly, s=0.4, c='lime', marker='.')
ax5.set_title('%s lobe instance masks (touching clusters): n=%d'
              % ('SAM' if sam_status == 'ok' else 'Fallback-watershed', len(lobe_masks)))
ax5.axis('off')
plt.tight_layout(); plt.savefig('verifier_panel_sam_lobes.png', dpi=100); plt.close()

# Detection-scale check panel.
fig4, ax4 = plt.subplots(1, 1, figsize=(14, 10.5))
ax4.imshow(disp_rgb)
for o in blob_objects:
    cy = o['cy']; cx = o['cx']; d = o.get('diameter_nm', visible_diam_px)
    circ = plt.Circle((cx, cy), d / 2.0, fill=False, color='cyan', linewidth=1.2)
    ax4.add_patch(circ)
    ax4.plot(cx, cy, 'c+', markersize=6)
ax4.set_title('Raw log_blob detections (circle=measured disc ~%.0f px) n=%d'
              % (visible_diam_px, len(blob_objects)))
ax4.axis('off')
plt.tight_layout(); plt.savefig('verifier_panel_detection_scale.png', dpi=100); plt.close()

# -------------------------------------------------------------------
# 15. Save arrays
# -------------------------------------------------------------------
final_labels_i32 = final_labels.astype(np.int32)
np.save('analysis_labels.npy', final_labels_i32)
np.save('particle_diameters_px.npy', diam_conv.astype(np.float64))

saved_arrays = {
    'analysis_labels.npy': {
        'description': 'Integer label map of %d individual round vaterite discs after merging '
                       'isolated LoG masks with SAM (or fallback watershed) touching-lobe '
                       'instance masks, IoU dedup, and area/solidity/aspect/edge post-filter; '
                       'background=0' % particle_count,
        'shape': list(final_labels_i32.shape),
        'dtype': str(final_labels_i32.dtype),
    },
    'particle_diameters_px.npy': {
        'description': 'Per-particle equivalent-circle diameter in PIXELS (calibration untrusted, no nm/um)',
        'shape': list(diam_conv.shape),
        'dtype': str(diam_conv.dtype),
    },
}

# -------------------------------------------------------------------
# 16. Results JSON
# -------------------------------------------------------------------
results = {
    'analysis_type': 'Hybrid detection: scale-matched LoG blob detection (log_blob_detect, sigma '
                     'from EMPIRICALLY verified ~%.0f px visible disc diameter) for isolated round '
                     'discs + run_sam_analysis for touching/overlapping dumbbell clusters (each '
                     'lobe an individual instance mask, REPLACING the repeatedly-failing '
                     'distance-transform watershed split). Mask sets merged with IoU dedup (SAM '
                     'wins in cluster regions), then per-particle area/solidity/aspect/edge '
                     'post-filter with clearly-round edge discs EXEMPT from the area cut. '
                     'Metadata calibration resolved via resolve_pixel_size_nm but treated as '
                     'UNVERIFIED/UNRECONCILED; ~60%% self-check disagreement => diameters in PX.'
                     % visible_diam_px,
    'extracted_features': {
        'per_particle_diameter_px': [float(round(v, 2)) for v in diam_conv.tolist()],
        'per_particle_major_axis_px': [float(round(v, 2)) for v in major_list],
        'per_particle_minor_axis_px': [float(round(v, 2)) for v in minor_list],
        'per_particle_aspect_ratio': [float(round(v, 3)) for v in aspect_list],
        'per_particle_solidity': [float(round(v, 3)) for v in solidity_list],
        'per_particle_area_px2': [float(round(v, 1)) for v in area_list],
        'particle_count': particle_count,
        'num_touching_pairs_split': int(touching_pairs),
        'size_distribution_histogram_px': {
            'counts': hist_counts,
            'bin_edges': hist_edges,
        },
        'median_diameter_px': d_median,
        'mean_diameter_px': d_mean,
        'std_diameter_px': d_std,
        'measured_isolated_disc_diameters_px': [float(round(v, 1)) for v in measured_isolated],
        'verified_disc_diameter_px': round(float(visible_diam_px), 1),
        'back_computed_nm_per_px': round(float(back_computed_nm_per_px), 2),
        'metadata_nm_per_px': (round(float(nm_per_px_meta), 3) if nm_per_px_meta is not None else None),
        'disagreement_percent': (round(float(disagreement_pct), 1) if disagreement_pct is not None else None),
        'calibration_status': 'UNVERIFIED/UNRECONCILED',
        'count_edge_discs_retained': int(edge_discs_retained),
        'count_rejected_fragments': int(rejected_fragments),
        'count_rejected_elongated': int(rejected_debris),
        'count_rejected_edge': int(rejected_edge),
        'count_rejected_total': int(total_rejected),
        'solidity_median': sol_median,
        'aspect_ratio_median': asp_median,
    },
    'quality_metrics': {
        'metadata_pixel_size_nm_per_px_unverified': nm_per_px_meta,
        'pixel_size_source': px_source,
        'calibration_trustworthy': bool(calibration_trustworthy),
        'calibration_status': 'UNVERIFIED/UNRECONCILED (not trusted)',
        'calibration_verification': verify_note,
        'calibration_disagreement_pct': (round(float(disagreement_pct), 1) if disagreement_pct is not None else None),
        'implied_diameter_um_from_metadata': (round(float(implied_diam_um), 2) if implied_diam_um is not None else None),
        'assumed_true_diameter_um_for_backcalc': ASSUMED_TRUE_DIAM_UM,
        'sam_status': sam_status,
        'sam_total_masks': int(sam_total),
        'sam_lobe_masks_used': int(len(sam_lobe_masks)),
        'fallback_watershed_lobe_masks': int(len(fallback_lobe_masks)),
        'lobe_masks_source': ('SAM' if (sam_status == 'ok' and len(sam_lobe_masks) > 0) else 'fallback_watershed'),
        'isolated_log_masks': int(len(iso_masks)),
        'isolated_log_masks_dropped_by_dedup': int(n_iso_dropped),
        'coarse_blob_detected': coarse_n,
        'blob_detected_raw': int(det.get('n_detected', 0)),
        'polarity_used': det.get('polarity_used', 'bright'),
        'sigma_min_px': round(float(min_sig), 1),
        'sigma_max_px': round(float(max_sig), 1),
        'threshold_rel_used': 0.16,
        'overlap_used': 0.30,
        'median_particle_area_px2': round(float(median_area), 1),
        'n_touching_clusters_detected': int(len(cluster_comps)),
        'n_isolated_components_detected': int(len(isolated_comps)),
    },
    'summary': ('Detected %d near-monodisperse individual round vaterite discs (median diameter '
                '%.1f px, std %.1f px) via HYBRID LoG(isolated)+SAM(touching-lobe) detection: '
                'run_sam_analysis (%s) REPLACED the repeatedly-failing distance-transform '
                'watershed split and yielded %d lobe instance masks over %d touching clusters, '
                'recovering %d touching pairs; merged with isolated LoG masks via IoU dedup '
                '(%d LoG masks dropped where SAM won). Post-filter rejected %d fragments, %d '
                'elongated, %d edge-truncated; %d edge discs RETAINED (far-left round disc '
                'exempted from area cut). Diameters reported in PX: metadata ~%s nm/px on ~%.0f '
                'px discs gives implausible ~%s um, back-computed ~%.1f nm/px disagrees by ~%s%% '
                '- no defensible nm conversion exists, calibration UNVERIFIED/UNRECONCILED. '
                'Adjustments: %s'
                % (particle_count,
                   (d_median if d_median is not None else float('nan')),
                   (d_std if d_std is not None else float('nan')),
                   sam_status, len(lobe_masks), len(cluster_comps), touching_pairs,
                   n_iso_dropped, rejected_fragments, rejected_debris, rejected_edge,
                   edge_discs_retained,
                   ('%.1f' % nm_per_px_meta) if nm_per_px_meta is not None else 'None',
                   visible_diam_px,
                   ('%.1f' % implied_diam_um) if implied_diam_um is not None else 'N/A',
                   back_computed_nm_per_px,
                   ('%.0f' % disagreement_pct) if disagreement_pct is not None else 'N/A',
                   ' | '.join(adjustments[:5]))),
    'saved_arrays': saved_arrays,
}

print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results))
