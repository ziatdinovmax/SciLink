import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm

# ---------------------------------------------------------------
# 0) Load metadata + resolve pixel size
# ---------------------------------------------------------------
image = np.load("data.npy")
if image.ndim == 3:
    img_shape = image.shape[:2]
else:
    img_shape = image.shape

metadata = None
try:
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
except Exception:
    metadata = None

pixel_size_nm = None
px_source = 'fallback'
if metadata is not None:
    try:
        px = resolve_pixel_size_nm(metadata, img_shape)
    except Exception:
        px = None
    if px is not None:
        pixel_size_nm = float(px['x'])
        px_source = px.get('source', 'metadata')

# Plan-stated fallback calibration
if pixel_size_nm is None or pixel_size_nm <= 0:
    pixel_size_nm = 0.6828
    px_source = 'plan_default(0.6828)'

note_parts = []

# ---------------------------------------------------------------
# 1) Load Tier-1 interior centroids (reuse, do NOT re-detect)
# ---------------------------------------------------------------
centroids_px = None
cent_source = None
for cand in ["centroid_coordinates_px.npy", "interior_centroids_px.npy", "centroids_px.npy"]:
    try:
        arr = np.load(cand)
        centroids_px = np.asarray(arr, dtype=float)
        cent_source = cand
        break
    except Exception:
        continue

if centroids_px is None:
    note_parts.append("centroid_coordinates_px.npy not found; attempted fallbacks (analysis_labels.npy centroids).")
    # Fallback: derive centroids from a saved label map if present
    try:
        from skimage import measure
        lab = np.load("analysis_labels.npy")
        props = measure.regionprops(lab.astype(np.int32))
        centroids_px = np.array([[p.centroid[1], p.centroid[0]] for p in props], dtype=float)  # (x, y)
        cent_source = "analysis_labels.npy (regionprops centroids)"
    except Exception:
        centroids_px = None

if centroids_px is None or centroids_px.size == 0:
    # Cannot proceed without points; emit a minimal failure result.
    results = {
        "analysis_type": "Edge-corrected Ripley's L(r) and pair correlation g(r) spatial point-pattern analysis",
        "extracted_features": {},
        "quality_metrics": {"n_points": 0},
        "summary": "FAILED: no Tier-1 centroid file (centroid_coordinates_px.npy) or usable fallback found; cannot compute spatial statistics.",
        "saved_arrays": {},
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "No centroids available", ha='center', va='center')
    ax.axis('off')
    plt.savefig('visualization.png', dpi=100)
    plt.close()
    print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}")
    raise SystemExit(0)

# Ensure shape (N,2). Detect (x,y) vs (row,col) convention.
centroids_px = np.atleast_2d(centroids_px)
if centroids_px.shape[1] != 2 and centroids_px.shape[0] == 2:
    centroids_px = centroids_px.T

# ---------------------------------------------------------------
# Convert to nm and define observation window
# ---------------------------------------------------------------
pts_nm = centroids_px * pixel_size_nm  # columns are (x_nm, y_nm) if input was (x,y)

# Field of view (nm square). Prefer metadata spatial_info, else plan value.
fov_nm = None
try:
    si = metadata['experimental_details']['spatial_info']
    fovx = si.get('field_of_view_x')
    fovy = si.get('field_of_view_y')
    units = str(si.get('field_of_view_units', 'nm')).lower()
    scale = 1.0
    if units in ('um', 'µm', 'micron', 'microns', 'micrometer', 'micrometers'):
        scale = 1000.0
    elif units in ('nm', 'nanometer', 'nanometers'):
        scale = 1.0
    if fovx is not None and fovy is not None:
        fov_nm = (float(fovx) * scale, float(fovy) * scale)
except Exception:
    fov_nm = None

if fov_nm is None:
    # plan-stated FOV
    fov_nm = (1398.34, 1398.34)
    note_parts.append("FOV from plan default (1398.34 nm square).")

# Define window bounds. Use FOV if points fit inside it; otherwise fall back
# to the point bounding box (robust to unknown centroid convention).
x = pts_nm[:, 0]
y = pts_nm[:, 1]

xmin_data, xmax_data = float(x.min()), float(x.max())
ymin_data, ymax_data = float(y.min()), float(y.max())

use_fov = (xmax_data <= fov_nm[0] * 1.05) and (ymax_data <= fov_nm[1] * 1.05) and \
          (xmin_data >= -0.05 * fov_nm[0]) and (ymin_data >= -0.05 * fov_nm[1])

if use_fov:
    win = (0.0, fov_nm[0], 0.0, fov_nm[1])  # xmin, xmax, ymin, ymax
    win_source = 'metadata/plan FOV'
else:
    win = (xmin_data, xmax_data, ymin_data, ymax_data)
    win_source = 'point bounding box (points exceeded FOV)'
    note_parts.append("Points exceeded FOV bounds; used point bounding box as window.")

xmin, xmax, ymin, ymax = win
Lx = xmax - xmin
Ly = ymax - ymin
win_area = Lx * Ly

# Restrict to interior/valid points strictly inside the window
inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
pts = pts_nm[inside]
N = pts.shape[0]
intensity = N / win_area if win_area > 0 else 0.0  # points per nm^2

# ---------------------------------------------------------------
# 2) Edge-corrected Ripley's K/L and g(r)
# ---------------------------------------------------------------
NN_SPACING_NM = 37.6

r_max = 400.0
r_min = 10.0
dr = 2.5
r = np.arange(r_min, r_max + dr, dr)


def translational_edge_weight(px, py, dvec_x, dvec_y, Lx, Ly):
    """Translational (isotropic-approx) edge correction weight per pair.
    Weight = area(window) / area(window intersect window translated by pair vector).
    For a rectangle: overlap = (Lx-|dx|)*(Ly-|dy|)."""
    ox = Lx - np.abs(dvec_x)
    oy = Ly - np.abs(dvec_y)
    ox = np.clip(ox, 1e-9, None)
    oy = np.clip(oy, 1e-9, None)
    overlap = ox * oy
    return (Lx * Ly) / overlap


def ripley_K_and_g(points, r, Lx, Ly, area, sigma_frac=0.10):
    """Compute translational edge-corrected K(r), L(r)-r, and g(r).
    g(r) via kernel (Gaussian) density estimate of edge-corrected pair distances."""
    n = points.shape[0]
    if n < 2:
        return (np.full_like(r, np.nan), np.full_like(r, np.nan), np.full_like(r, np.nan))
    lam = n / area
    px = points[:, 0]
    py = points[:, 1]

    # Pairwise vectors (i != j), both directions counted
    dx = px[:, None] - px[None, :]
    dy = py[:, None] - py[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    iu = ~np.eye(n, dtype=bool)
    dvx = dx[iu]
    dvy = dy[iu]
    dd = dist[iu]

    w = translational_edge_weight(None, None, dvx, dvy, Lx, Ly)

    # K(r): sum of weights for pairs within r, normalized
    K = np.empty_like(r)
    for k, rr in enumerate(r):
        m = dd <= rr
        K[k] = w[m].sum()
    K = K / (lam * n)

    L = np.sqrt(np.clip(K, 0, None) / np.pi)
    L_minus_r = L - r

    # g(r): Gaussian kernel estimator of edge-corrected pair distances
    # g(r) = (1/(2*pi*r*lam*n)) * sum_pairs w * kernel(r - d_ij)
    h = max(sigma_frac * NN_SPACING_NM, dr)  # kernel bandwidth in nm
    g = np.empty_like(r)
    norm = 1.0 / (np.sqrt(2 * np.pi) * h)
    for k, rr in enumerate(r):
        kern = norm * np.exp(-0.5 * ((rr - dd) / h) ** 2)
        s = np.sum(w * kern)
        denom = 2.0 * np.pi * rr * lam * n
        g[k] = s / denom if denom > 0 else np.nan
    return K, L_minus_r, g


K_obs, Lmr_obs, g_obs = ripley_K_and_g(pts, r, Lx, Ly, win_area)

# ---------------------------------------------------------------
# 3) CSR Monte Carlo envelope
# ---------------------------------------------------------------
n_mc = 150
rng = np.random.default_rng(42)
Lmr_mc = np.empty((n_mc, r.size))
g_mc = np.empty((n_mc, r.size))

for i in range(n_mc):
    rx = rng.uniform(xmin, xmax, size=N)
    ry = rng.uniform(ymin, ymax, size=N)
    csr_pts = np.column_stack([rx, ry])
    _, lmr_i, g_i = ripley_K_and_g(csr_pts, r, Lx, Ly, win_area)
    Lmr_mc[i] = lmr_i
    g_mc[i] = g_i

Lmr_lo = np.nanpercentile(Lmr_mc, 2.5, axis=0)
Lmr_hi = np.nanpercentile(Lmr_mc, 97.5, axis=0)
g_lo = np.nanpercentile(g_mc, 2.5, axis=0)
g_hi = np.nanpercentile(g_mc, 97.5, axis=0)

# ---------------------------------------------------------------
# 4) Identify regimes / characteristic scales
# ---------------------------------------------------------------
# Short-range exclusion radius: largest r where L(r)-r is below envelope (regularity)
# and/or g(r) below envelope (exclusion) at small r.
below_env = Lmr_obs < Lmr_lo
above_env = Lmr_obs > Lmr_hi

# short-range exclusion: contiguous below-envelope region starting near r_min
short_range_exclusion_radius_nm = None
if below_env.any():
    # find contiguous run from the start (smallest r) that is below envelope
    idx = 0
    while idx < below_env.size and below_env[idx]:
        idx += 1
    if idx > 0:
        short_range_exclusion_radius_nm = float(r[idx - 1])
    else:
        # exclusion not at the very start; take first below-env radius
        first_below = np.argmax(below_env)
        short_range_exclusion_radius_nm = float(r[first_below])

# Crossover radius: where L(r)-r changes sign from negative to positive
cross_radius = None
sign = np.sign(Lmr_obs)
for k in range(1, r.size):
    if np.isfinite(Lmr_obs[k]) and np.isfinite(Lmr_obs[k - 1]):
        if Lmr_obs[k - 1] < 0 <= Lmr_obs[k]:
            # linear interpolate zero crossing
            r0, r1 = r[k - 1], r[k]
            v0, v1 = Lmr_obs[k - 1], Lmr_obs[k]
            if v1 != v0:
                cross_radius = float(r0 + (0 - v0) * (r1 - r0) / (v1 - v0))
            else:
                cross_radius = float(r1)
            break
L_r_crossover_radius_nm = cross_radius

# First g(r) peak: local maximum in g_obs (after the exclusion notch)
first_g_peak_radius_nm = None
first_g_peak_val = None
g_clean = np.where(np.isfinite(g_obs), g_obs, -np.inf)
for k in range(1, r.size - 1):
    if g_clean[k] > g_clean[k - 1] and g_clean[k] >= g_clean[k + 1] and g_clean[k] > 1.0:
        first_g_peak_radius_nm = float(r[k])
        first_g_peak_val = float(g_obs[k])
        break

# Mesoscale clustering length: the r at which L(r)-r reaches its maximum
# in the clustering (positive, above-envelope) regime.
mesoscale_clustering_length_nm = None
if above_env.any():
    masked = np.where(above_env, Lmr_obs, -np.inf)
    if np.isfinite(masked).any() and masked.max() > -np.inf:
        mesoscale_clustering_length_nm = float(r[int(np.argmax(masked))])
if mesoscale_clustering_length_nm is None:
    # fallback: argmax of L(r)-r overall if positive
    if np.nanmax(Lmr_obs) > 0:
        mesoscale_clustering_length_nm = float(r[int(np.nanargmax(Lmr_obs))])

# NN spacing observed (from data) as cross-check
nn_obs_median = None
if N >= 2:
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn = dists[:, 1]
    nn_obs_median = float(np.median(nn))

# ---------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------
has_short_regularity = bool(below_env[:max(1, r.size // 3)].any())
has_mesoscale_clustering = bool(above_env.any())

if has_short_regularity and has_mesoscale_clustering:
    regime_classification = "short-range exclusion/regularity transitioning to mesoscale clustering"
elif has_mesoscale_clustering:
    regime_classification = "clustered (aggregated) across scales"
elif has_short_regularity:
    regime_classification = "regular/dispersed (exclusion dominated)"
else:
    regime_classification = "consistent with complete spatial randomness (CSR)"

# ---------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------
np.save('spatial_stats_r_nm.npy', r)
np.save('ripley_L_minus_r.npy', Lmr_obs)
np.save('pair_correlation_g.npy', g_obs)
csr_bounds = np.column_stack([r, Lmr_lo, Lmr_hi, g_lo, g_hi])
np.save('csr_envelope_bounds.npy', csr_bounds)
np.save('analysis_points_nm.npy', pts)

# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: point pattern in window
ax = axes[0]
ax.scatter(pts[:, 0], pts[:, 1], s=6, c='navy')
ax.add_patch(plt.Rectangle((xmin, ymin), Lx, Ly, fill=False, ec='red', lw=1.2))
ax.set_aspect('equal')
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_title(f'Interior centroids (N={N})\nwindow: {Lx:.0f}x{Ly:.0f} nm')
ax.invert_yaxis()

# Panel 2: L(r)-r with CSR envelope
ax = axes[1]
ax.fill_between(r, Lmr_lo, Lmr_hi, color='lightgray', alpha=0.8, label='CSR 95% env')
ax.plot(r, Lmr_obs, 'b-', lw=1.8, label='observed L(r)-r')
ax.axhline(0, color='k', lw=0.7)
ax.axvline(NN_SPACING_NM, color='orange', ls=':', lw=1.2, label=f'NN {NN_SPACING_NM:.1f} nm')
if L_r_crossover_radius_nm is not None:
    ax.axvline(L_r_crossover_radius_nm, color='green', ls='--', lw=1.2,
               label=f'crossover {L_r_crossover_radius_nm:.0f} nm')
if mesoscale_clustering_length_nm is not None:
    ax.axvline(mesoscale_clustering_length_nm, color='purple', ls='-.', lw=1.0,
               label=f'cluster {mesoscale_clustering_length_nm:.0f} nm')
ax.set_xlabel('r (nm)')
ax.set_ylabel('L(r) - r (nm)')
ax.set_title('Centered Ripley L-function')
ax.legend(fontsize=7, loc='best')

# Panel 3: g(r) with CSR envelope
ax = axes[2]
ax.fill_between(r, g_lo, g_hi, color='lightgray', alpha=0.8, label='CSR 95% env')
ax.plot(r, g_obs, 'b-', lw=1.8, label='observed g(r)')
ax.axhline(1, color='k', lw=0.7)
ax.axvline(NN_SPACING_NM, color='orange', ls=':', lw=1.2, label=f'NN {NN_SPACING_NM:.1f} nm')
if first_g_peak_radius_nm is not None:
    ax.axvline(first_g_peak_radius_nm, color='red', ls='--', lw=1.2,
               label=f'1st peak {first_g_peak_radius_nm:.0f} nm')
ax.set_xlabel('r (nm)')
ax.set_ylabel('g(r)')
ax.set_title('Pair correlation function')
ax.legend(fontsize=7, loc='best')

plt.tight_layout()
plt.savefig('visualization.png', dpi=100)
plt.close()

# ---------------------------------------------------------------
# Assemble JSON results
# ---------------------------------------------------------------
summary_parts = []
summary_parts.append(
    f"Loaded {N} interior centroids from {cent_source}; calibrated {pixel_size_nm:.4f} nm/px (source: {px_source})."
)
summary_parts.append(
    f"Window {Lx:.1f}x{Ly:.1f} nm ({win_source}), intensity {intensity*1e4:.3f} pts/(100nm)^2."
)
summary_parts.append(
    "Computed translational edge-corrected Ripley K->L(r)-r and Gaussian-kernel g(r) over "
    f"r={r_min:.0f}-{r_max:.0f} nm (dr={dr} nm) with a {n_mc}-run CSR Monte Carlo 95% envelope."
)
summary_parts.append(f"Regime: {regime_classification}.")
if nn_obs_median is not None:
    summary_parts.append(f"Observed median NN spacing {nn_obs_median:.1f} nm (marker {NN_SPACING_NM} nm).")
if note_parts:
    summary_parts.append(' '.join(note_parts))

results = {
    "analysis_type": "Edge-corrected Ripley's L(r) and pair correlation g(r) spatial point-pattern analysis of Tier-1 interior centroids with CSR Monte Carlo envelope",
    "extracted_features": {
        "ripley_L_minus_r_curve": [float(v) for v in Lmr_obs],
        "pair_correlation_g_of_r": [float(v) for v in g_obs],
        "r_values_nm": [float(v) for v in r],
        "csr_envelope_bounds": {
            "L_minus_r_lo": [float(v) for v in Lmr_lo],
            "L_minus_r_hi": [float(v) for v in Lmr_hi],
            "g_lo": [float(v) for v in g_lo],
            "g_hi": [float(v) for v in g_hi],
        },
        "L_r_crossover_radius_nm": L_r_crossover_radius_nm,
        "short_range_exclusion_radius_nm": short_range_exclusion_radius_nm,
        "first_g_peak_radius_nm": first_g_peak_radius_nm,
        "first_g_peak_value": first_g_peak_val,
        "mesoscale_clustering_length_nm": mesoscale_clustering_length_nm,
        "regime_classification": regime_classification,
        "nn_spacing_marker_nm": NN_SPACING_NM,
        "observed_median_nn_spacing_nm": nn_obs_median,
    },
    "quality_metrics": {
        "n_points": int(N),
        "pixel_size_nm": float(pixel_size_nm),
        "pixel_size_source": px_source,
        "window_nm": [float(Lx), float(Ly)],
        "window_source": win_source,
        "intensity_pts_per_nm2": float(intensity),
        "n_monte_carlo": int(n_mc),
        "r_min_nm": float(r_min),
        "r_max_nm": float(r_max),
        "dr_nm": float(dr),
        "centroid_source": cent_source,
    },
    "summary": ' '.join(summary_parts),
    "saved_arrays": {
        "spatial_stats_r_nm.npy": {
            "description": "Radial distance values r (nm) used for L(r) and g(r)",
            "shape": list(r.shape), "dtype": str(r.dtype),
        },
        "ripley_L_minus_r.npy": {
            "description": "Edge-corrected centered Ripley L(r)-r curve (nm) vs r",
            "shape": list(Lmr_obs.shape), "dtype": str(Lmr_obs.dtype),
        },
        "pair_correlation_g.npy": {
            "description": "Edge-corrected pair correlation function g(r) vs r",
            "shape": list(g_obs.shape), "dtype": str(g_obs.dtype),
        },
        "csr_envelope_bounds.npy": {
            "description": "Columns: r_nm, L-r 2.5pct, L-r 97.5pct, g 2.5pct, g 97.5pct from CSR Monte Carlo",
            "shape": list(csr_bounds.shape), "dtype": str(csr_bounds.dtype),
        },
        "analysis_points_nm.npy": {
            "description": "Interior centroid coordinates in nm (x, y) used for the analysis",
            "shape": list(pts.shape), "dtype": str(pts.dtype),
        },
    },
}

print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}")
