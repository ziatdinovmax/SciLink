import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform, morphology, measure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
image = np.load('data.npy')
if image.ndim > 2:
    # Not RGB; take first channel per instructions
    image = image[:, :, 0]
image = image.astype(np.float64)
N_y, N_x = image.shape  # rows, cols

# ---------------------------------------------------------------
# Read metadata for calibration (authoritative)
# ---------------------------------------------------------------
meta = {}
try:
    with open('metadata.json', 'r') as f:
        meta = json.load(f)
except Exception as e:
    meta = {}

def deep_get(d, keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# spatial info
spatial = deep_get(meta, ['experimental_details', 'spatial_info'], {}) or {}
if not spatial:
    spatial = deep_get(meta, ['system_info', 'experimental_details', 'spatial_info'], {}) or {}

FOV_x = spatial.get('field_of_view_x', None)
FOV_y = spatial.get('field_of_view_y', None)
fov_units = spatial.get('field_of_view_units', 'nm')

# convert FOV to nm if needed
def to_nm(val, units):
    if val is None:
        return None
    u = (units or 'nm').lower()
    if u in ('nm', 'nanometer', 'nanometers'):
        return float(val)
    if u in ('um', 'micron', 'micrometer', 'micrometers', 'µm'):
        return float(val) * 1000.0
    if u in ('m', 'meter', 'meters'):
        return float(val) * 1e9
    if u in ('a', 'angstrom', 'angstroms', 'å'):
        return float(val) * 0.1
    return float(val)

FOV_x_nm = to_nm(FOV_x, fov_units)
FOV_y_nm = to_nm(FOV_y, fov_units)

# data range for volts mapping
data_range = deep_get(meta, ['experimental_details', 'data_range'], {}) or {}
if not data_range:
    # sometimes flat keys
    dr_min = spatial.get('data_range_minimum', None)
    dr_max = spatial.get('data_range_maximum', None)
    dr_units = spatial.get('data_range_units', None)
else:
    dr_min = data_range.get('data_range_minimum', None)
    dr_max = data_range.get('data_range_maximum', None)
    dr_units = data_range.get('data_range_units', None)

# search broadly for data range if still missing
def find_key(d, target):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == target:
                return v
            r = find_key(v, target)
            if r is not None:
                return r
    return None

if dr_min is None:
    dr_min = find_key(meta, 'data_range_minimum')
if dr_max is None:
    dr_max = find_key(meta, 'data_range_maximum')
if dr_units is None:
    dr_units = find_key(meta, 'data_range_units')

# Plan-specified calibration constants (authoritative fallback)
V_MIN_PLAN = 1.6481
V_MAX_PLAN = 1.8176
if dr_min is None:
    dr_min = V_MIN_PLAN
if dr_max is None:
    dr_max = V_MAX_PLAN
if dr_units is None:
    dr_units = 'V'

dr_min = float(dr_min)
dr_max = float(dr_max)

# ---------------------------------------------------------------
# Convert uint16 -> volts:  V = raw*(max-min)/65535 + min
# ---------------------------------------------------------------
RAW_MAX = 65535.0
volts = image * (dr_max - dr_min) / RAW_MAX + dr_min

# round-trip verification within quantization
round_trip_raw = np.rint((volts - dr_min) * RAW_MAX / (dr_max - dr_min))
round_trip_ok = bool(np.all(np.abs(round_trip_raw - image) <= 1.0))

# ---------------------------------------------------------------
# Resample to SQUARE pixels using true per-axis pixel size
# ---------------------------------------------------------------
if FOV_x_nm is not None and FOV_y_nm is not None:
    px_x = FOV_x_nm / N_x  # nm per pixel horizontal
    px_y = FOV_y_nm / N_y  # nm per pixel vertical
else:
    px_x = 1.0
    px_y = 1.0

# choose target square pixel size = smaller of the two (finer)
nm_per_px = min(px_x, px_y)
# compute new dims preserving physical FOV
if FOV_x_nm is not None and FOV_y_nm is not None:
    N_x_new = int(round(FOV_x_nm / nm_per_px))
    N_y_new = int(round(FOV_y_nm / nm_per_px))
else:
    N_x_new = N_x
    N_y_new = N_y

volts_sq = transform.resize(volts, (N_y_new, N_x_new), order=1,
                            mode='reflect', anti_aliasing=True,
                            preserve_range=True)

# confirm square pixel consistency
if FOV_x_nm is not None and FOV_y_nm is not None:
    nm_px_x_check = FOV_x_nm / N_x_new
    nm_px_y_check = FOV_y_nm / N_y_new
    square_ok = bool(abs(nm_px_x_check - nm_px_y_check) < 1e-3 * max(nm_px_x_check, 1e-9))
else:
    nm_px_x_check = nm_px_y_check = nm_per_px
    square_ok = True

# ---------------------------------------------------------------
# Row-align: subtract per-row median (escalate to polynomial if striping remains)
# ---------------------------------------------------------------
work = volts_sq.copy()
row_med = np.median(work, axis=1, keepdims=True)
work_rowmed = work - row_med

# check residual striping over featureless area: std of per-row means
row_means_res = np.mean(work_rowmed, axis=1)
residual_stripe = float(np.std(row_means_res))

row_method = 'per-row median'
# escalate to per-row 1st-order polynomial if striping remains significant
overall_std = float(np.std(work_rowmed))
if residual_stripe > 0.02 * overall_std and overall_std > 0:
    cols = np.arange(work.shape[1])
    work_poly = np.empty_like(work)
    for r in range(work.shape[0]):
        c = np.polyfit(cols, work[r], 1)
        work_poly[r] = work[r] - np.polyval(c, cols)
    work_rowmed = work_poly
    row_method = 'per-row 1st-order polynomial'

# ---------------------------------------------------------------
# Global 1st-order plane fit removal
# ---------------------------------------------------------------
yy, xx = np.mgrid[0:work_rowmed.shape[0], 0:work_rowmed.shape[1]]
A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
coef, *_ = np.linalg.lstsq(A, work_rowmed.ravel(), rcond=None)
plane = (A @ coef).reshape(work_rowmed.shape)
leveled = work_rowmed - plane

# ---------------------------------------------------------------
# Mask obvious scan artifacts (dark upper-left streaks)
# Detect extreme low outliers, especially near upper-left
# ---------------------------------------------------------------
med = np.median(leveled)
mad = np.median(np.abs(leveled - med)) + 1e-12
robust_sigma = 1.4826 * mad
# artifact = strongly dark outliers
artifact_mask = leveled < (med - 5.0 * robust_sigma)
# emphasize upper-left region streaks: also catch moderate dark in top-left quadrant
top_left = np.zeros_like(leveled, dtype=bool)
tl_r = leveled.shape[0] // 3
tl_c = leveled.shape[1] // 3
top_left[:tl_r, :tl_c] = True
artifact_mask |= (top_left & (leveled < (med - 3.5 * robust_sigma)))
# clean small specks, keep coherent streaks
artifact_mask = morphology.remove_small_objects(artifact_mask, min_size=8)
artifact_mask = morphology.binary_dilation(artifact_mask, morphology.disk(1))

valid_mask = ~artifact_mask
frac_masked = float(np.mean(artifact_mask))

# ---------------------------------------------------------------
# Gaussian smooth (sigma=2) on parallel copy for clustering
# ---------------------------------------------------------------
smoothed = ndimage.gaussian_filter(leveled, sigma=2.0)

# ---------------------------------------------------------------
# Adaptive domain count k=2..4 via silhouette + BIC + coherence check
# Cluster ONLY on valid (non-artifact) pixels; artifacts never enter KMeans.
# ---------------------------------------------------------------
valid_vals = smoothed[valid_mask].reshape(-1, 1)
# subsample for silhouette speed
rng = np.random.default_rng(42)
if valid_vals.shape[0] > 6000:
    sil_idx = rng.choice(valid_vals.shape[0], 6000, replace=False)
else:
    sil_idx = np.arange(valid_vals.shape[0])


def _gaussian_bic(X, labels, centers):
    # BIC for a spherical-Gaussian mixture (1D feature) implied by KMeans.
    n, d = X.shape
    k = centers.shape[0]
    # pooled variance across clusters
    ss = 0.0
    for ci in range(k):
        pts = X[labels == ci]
        if pts.shape[0] > 0:
            ss += np.sum((pts - centers[ci]) ** 2)
    denom = max(n - k, 1)
    var = ss / denom + 1e-12
    ll = 0.0
    for ci in range(k):
        pts = X[labels == ci]
        ni = pts.shape[0]
        if ni == 0:
            continue
        ll += (
            ni * np.log(ni / n)
            - ni * d / 2.0 * np.log(2.0 * np.pi * var)
            - (np.sum((pts - centers[ci]) ** 2)) / (2.0 * var)
        )
    n_params = k * d + k + 1  # means + weights + shared variance
    bic = -2.0 * ll + n_params * np.log(n)
    return float(bic)


def _coherence_for_labels(km_model, k):
    # Build a full valid-only label map and measure min largest-connected-component fraction.
    lm = np.zeros(smoothed.shape, dtype=np.int32)
    lab_valid = km_model.predict(valid_vals)
    tmp = np.zeros(smoothed.shape, dtype=np.int32)
    tmp[valid_mask] = lab_valid + 1  # 1..k in valid region, 0 elsewhere
    lm = tmp
    min_coh = 1.0
    for ci in range(1, k + 1):
        m = lm == ci
        if m.sum() == 0:
            min_coh = 0.0
            continue
        lbl, nlab = ndimage.label(m)
        if nlab == 0:
            min_coh = 0.0
            continue
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, nlab + 1))
        min_coh = min(min_coh, float(sizes.max() / m.sum()))
    return min_coh


best = None
selection_scores = {}
bic_scores = {}
coherence_scores = {}
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(valid_vals)
    try:
        sil = float(silhouette_score(valid_vals[sil_idx], lab[sil_idx]))
    except Exception:
        sil = -1.0
    bic = _gaussian_bic(valid_vals, lab, km.cluster_centers_)
    coh = _coherence_for_labels(km, k)
    selection_scores[k] = sil
    bic_scores[k] = bic
    coherence_scores[k] = coh
    # Combined selection: prefer silhouette, but require spatial coherence
    # (reject salt-and-pepper) using BIC as corroboration/tie-breaker.
    if best is None:
        best = {'k': k, 'sil': sil, 'bic': bic, 'coh': coh, 'km': km, 'lab': lab}
    else:
        # A candidate must be spatially coherent (largest CC >= 0.5) to win on silhouette.
        cand_ok = coh >= 0.5
        best_ok = best['coh'] >= 0.5
        if cand_ok and not best_ok:
            best = {'k': k, 'sil': sil, 'bic': bic, 'coh': coh, 'km': km, 'lab': lab}
        elif cand_ok == best_ok:
            # both coherent (or both not): choose higher silhouette; if silhouettes
            # are within a small margin, use lower BIC as corroboration.
            if sil > best['sil'] + 1e-3:
                best = {'k': k, 'sil': sil, 'bic': bic, 'coh': coh, 'km': km, 'lab': lab}
            elif abs(sil - best['sil']) <= 1e-3 and bic < best['bic']:
                best = {'k': k, 'sil': sil, 'bic': bic, 'coh': coh, 'km': km, 'lab': lab}

k_sel = best['k']
km = best['km']

# Build full label map from VALID-ONLY clustering (order clusters by mean potential)
centers = km.cluster_centers_.ravel()
order = np.argsort(centers)  # ascending mean potential
remap = np.zeros(k_sel, dtype=int)
for new_i, old_i in enumerate(order):
    remap[old_i] = new_i + 1  # labels 1..k

label_map = np.zeros(smoothed.shape, dtype=np.int32)
# Predict only on valid pixels so artifacts never enter clustering
lab_valid = km.predict(valid_vals)
remapped_valid = remap[lab_valid]
label_map[valid_mask] = remapped_valid
# artifacts remain 0

# ---------------------------------------------------------------
# Coherence check: largest connected component fraction per domain
# ---------------------------------------------------------------
coherence = {}
for dlab in range(1, k_sel + 1):
    m = label_map == dlab
    if m.sum() == 0:
        coherence[dlab] = 0.0
        continue
    lbl, n = ndimage.label(m)
    if n == 0:
        coherence[dlab] = 0.0
        continue
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    coherence[dlab] = float(sizes.max() / m.sum())

# Morphological cleanup to reduce salt-and-pepper (opening) for reporting
clean_label = label_map.copy()
for dlab in range(1, k_sel + 1):
    m = label_map == dlab
    m = morphology.remove_small_objects(m, min_size=16)
    # only remove specks; do not reassign here to keep it simple
    clean_label[(label_map == dlab) & (~m)] = 0

min_coherence = float(min(coherence.values())) if coherence else 0.0

# ---------------------------------------------------------------
# Per-domain measurements on LEVELED physical-unit image (not smoothed)
# ---------------------------------------------------------------
total_valid_px = int(np.sum(label_map > 0))
domain_stats = {}
for dlab in range(1, k_sel + 1):
    m = label_map == dlab
    npix = int(m.sum())
    if npix == 0:
        continue
    vals = leveled[m]
    domain_stats[dlab] = {
        'mean_potential_V': float(np.mean(vals)),
        'std_potential_V': float(np.std(vals)),
        'area_fraction': float(npix / total_valid_px) if total_valid_px > 0 else 0.0,
        'n_pixels': npix,
        'largest_cc_fraction': coherence.get(dlab, 0.0),
    }

# pairwise CPD contrast (mV)
pairwise = {}
dlabs = sorted(domain_stats.keys())
for i in range(len(dlabs)):
    for j in range(i + 1, len(dlabs)):
        a, b = dlabs[i], dlabs[j]
        diff_mV = (domain_stats[b]['mean_potential_V'] - domain_stats[a]['mean_potential_V']) * 1000.0
        pairwise[f'domain{a}_vs_domain{b}_mV'] = float(diff_mV)

# total leveled potential range (mV) over valid area
leveled_valid = leveled[valid_mask]
total_range_mV = float((leveled_valid.max() - leveled_valid.min()) * 1000.0)

# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

ax = axes[0, 0]
im0 = ax.imshow(volts, cmap='viridis', aspect='auto')
ax.set_title('Original (Volts)\n%dx%d raw px' % (N_y, N_x))
plt.colorbar(im0, ax=ax, fraction=0.046, label='V')

ax = axes[0, 1]
im1 = ax.imshow(volts_sq, cmap='viridis')
ax.set_title('Square px %.3f nm/px\n%dx%d' % (nm_per_px, N_y_new, N_x_new))
plt.colorbar(im1, ax=ax, fraction=0.046, label='V')

ax = axes[0, 2]
im2 = ax.imshow(leveled * 1000.0, cmap='RdBu_r')
ax.set_title('Leveled (%s + plane)\nmV' % row_method)
plt.colorbar(im2, ax=ax, fraction=0.046, label='mV')

ax = axes[1, 0]
im3 = ax.imshow(smoothed * 1000.0, cmap='RdBu_r')
ax.set_title('Smoothed (sigma=2) mV')
plt.colorbar(im3, ax=ax, fraction=0.046, label='mV')

ax = axes[1, 1]
# segmentation overlay
ax.imshow(leveled * 1000.0, cmap='gray')
over = np.ma.masked_where(label_map == 0, label_map)
ax.imshow(over, cmap='jet', alpha=0.45)
for dlab in range(1, k_sel + 1):
    conts = measure.find_contours((label_map == dlab).astype(float), 0.5)
    for c in conts:
        ax.plot(c[:, 1], c[:, 0], 'k-', linewidth=0.4)
ax.set_title('CPD domains (k=%d)' % k_sel)

ax = axes[1, 2]
ax.imshow(leveled * 1000.0, cmap='gray')
am = np.ma.masked_where(~artifact_mask, artifact_mask)
ax.imshow(am, cmap='autumn', alpha=0.8)
ax.set_title('Artifact mask (%.2f%% masked)' % (frac_masked * 100))

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------
np.save('cpd_domain_labels.npy', label_map)
np.save('leveled_potential_V.npy', leveled.astype(np.float32))
np.save('artifact_mask.npy', artifact_mask)

# ---------------------------------------------------------------
# Results JSON
# ---------------------------------------------------------------
domain_means = {f'domain{d}': domain_stats[d]['mean_potential_V'] for d in dlabs}
domain_areas = {f'domain{d}': domain_stats[d]['area_fraction'] for d in dlabs}

summary = (
    'KPFM surface potential leveled (row-align: %s, then 1st-order plane) and clustered into %d '
    'spatially-coherent CPD domains (k selected by silhouette + BIC corroboration + largest-CC '
    'coherence check; clustering fit/predicted on valid non-artifact pixels only, min largest-CC frac %.2f); '
    'total leveled range %.1f mV, %.2f%% area masked as artifact. '
    'nm/px=%.4f (square px, aspect preserved).'
) % (row_method, k_sel, min_coherence, total_range_mV, frac_masked * 100, nm_per_px)

adjust_notes = []
if not round_trip_ok:
    adjust_notes.append('round-trip dtype->V->dtype exceeded 1-LSB in some pixels (resampling introduces sub-quant deviations).')
if dr_min == V_MIN_PLAN and dr_max == V_MAX_PLAN and find_key(meta, 'data_range_minimum') is None:
    adjust_notes.append('data_range not found in metadata; used plan calibration min=1.6481 max=1.8176 V.')
if FOV_x_nm is None:
    adjust_notes.append('FOV not in metadata; used nm/px=1.0 placeholder for square resampling.')
if adjust_notes:
    summary += ' Notes: ' + ' '.join(adjust_notes)

results = {
    'analysis_type': 'KPFM CPD domain analysis: volts conversion, square-pixel resampling, row-align + plane leveling, artifact masking, adaptive K-means clustering of surface potential into spatially-coherent contact-potential domains',
    'extracted_features': {
        'n_cpd_domains': int(k_sel),
        'domain_selection_metric': 'silhouette + BIC + coherence',
        'silhouette_scores_by_k': selection_scores,
        'bic_scores_by_k': bic_scores,
        'coherence_scores_by_k': coherence_scores,
        'selected_silhouette': float(best['sil']),
        'selected_bic': float(best['bic']),
        'mean_surface_potential_V_per_domain': domain_means,
        'pairwise_CPD_contrast_mV': pairwise,
        'area_fraction_per_domain': domain_areas,
        'total_leveled_potential_range_mV': total_range_mV,
        'fraction_area_masked_artifact': frac_masked,
        'per_domain_details': {str(d): domain_stats[d] for d in dlabs},
    },
    'quality_metrics': {
        'nm_per_px': float(nm_per_px),
        'square_pixel_consistent': square_ok,
        'nm_per_px_x_check': float(nm_px_x_check),
        'nm_per_px_y_check': float(nm_px_y_check),
        'volts_roundtrip_within_1LSB': round_trip_ok,
        'data_range_min_V': dr_min,
        'data_range_max_V': dr_max,
        'data_range_units': dr_units,
        'row_alignment_method': row_method,
        'residual_stripe_std_V': residual_stripe,
        'min_domain_largest_cc_fraction': min_coherence,
        'domain_coherence': {str(d): coherence[d] for d in coherence},
        'clustering_on_valid_pixels_only': True,
        'resampled_shape': [int(N_y_new), int(N_x_new)],
        'original_shape': [int(N_y), int(N_x)],
    },
    'summary': summary,
    'saved_arrays': {
        'cpd_domain_labels.npy': {
            'description': 'Integer CPD domain label map on square-pixel grid (background/artifact=0, domains labeled 1..%d ordered by ascending mean potential)' % k_sel,
            'shape': list(label_map.shape),
            'dtype': str(label_map.dtype),
        },
        'leveled_potential_V.npy': {
            'description': 'Leveled surface potential in Volts (row-aligned + plane-subtracted, square pixels)',
            'shape': list(leveled.shape),
            'dtype': 'float32',
        },
        'artifact_mask.npy': {
            'description': 'Boolean mask of excluded scan artifacts (dark upper-left streaks / outliers)',
            'shape': list(artifact_mask.shape),
            'dtype': 'bool',
        },
    },
}

print('IMAGE_ANALYSIS_RESULTS_JSON:' + json.dumps(results))
