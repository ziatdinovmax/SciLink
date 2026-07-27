import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm
from scilink.skills.image_analysis.atomic_stem.lattice import measure_lattice_constant
from scilink.skills._shared.fourier_reflection import fourier_reflection_map


def to_native(o):
    if isinstance(o, dict):
        return {str(k): to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


adjustments = []

# ---------------------------------------------------------------------------
# Load image
# ---------------------------------------------------------------------------
image = np.load("data.npy")
if image.ndim == 3:
    # pick first channel (data is nominally 2D; guard for multi-channel)
    if image.shape[2] <= 4:
        image = image[:, :, 0]
        adjustments.append("Input had >2 dims; used channel 0.")
    else:
        image = image[:, :, 0]
image = np.asarray(image, dtype=np.float32)

# ---------------------------------------------------------------------------
# Metadata / pixel size
# ---------------------------------------------------------------------------
metadata = None
try:
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
except Exception as e:
    adjustments.append(f"metadata.json not readable ({e}); pixel size may be unavailable.")

pixel_size_nm = None
px_source = None
px_x = px_y = None
if metadata is not None:
    px = resolve_pixel_size_nm(metadata, image.shape)
    if px is not None:
        px_x = float(px.get('x')) if px.get('x') is not None else None
        px_y = float(px.get('y')) if px.get('y') is not None else None
        px_source = px.get('source')
        if px_x is not None:
            pixel_size_nm = px_x

if pixel_size_nm is None:
    # Cannot resolve calibration from metadata; abort gracefully with report.
    results = {
        "analysis_type": "Lattice metrology (FFT Bragg geometry) + film/substrate discrimination",
        "extracted_features": {},
        "quality_metrics": {},
        "summary": "Pixel size could not be resolved from metadata.json; lattice metrology in nm is not possible.",
        "saved_arrays": {},
    }
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    ax.set_title('Raw image (no calibration)')
    fig.savefig('visualization.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(to_native(results))}")
    raise SystemExit(0)

# Note pixel anisotropy (measure_lattice_constant assumes square pixels)
if px_x is not None and px_y is not None and abs(px_x - px_y) / max(px_x, px_y) > 0.02:
    adjustments.append(
        f"Pixels anisotropic (x={px_x:.5f}, y={px_y:.5f} nm); using x for square-pixel tools per calibration."
    )

# ---------------------------------------------------------------------------
# STEP 1: Lattice metrology on RAW full frame
# ---------------------------------------------------------------------------
lat = measure_lattice_constant(image, pixel_size_nm)

# If flagged multi_lattice / low_confidence, crop to a single domain (center)
# and re-run.
cropped_lat = None
crop_used = None
if lat.get('multi_lattice') or lat.get('low_confidence'):
    h, w = image.shape
    cy, cx = h // 2, w // 2
    half = min(h, w) // 4  # central half
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half
    crop = image[y0:y1, x0:x1]
    adjustments.append(
        f"Step1 flagged (multi_lattice={lat.get('multi_lattice')}, "
        f"low_confidence={lat.get('low_confidence')}); re-ran measure_lattice_constant "
        f"on central crop [{y0}:{y1},{x0}:{x1}]."
    )
    cropped_lat = measure_lattice_constant(crop, pixel_size_nm)
    crop_used = [int(y0), int(y1), int(x0), int(x1)]

# Choose the reported lattice result (prefer crop if it improved confidence)
lat_report = lat
lat_source = "full_frame"
if cropped_lat is not None:
    if (not cropped_lat.get('multi_lattice') and not cropped_lat.get('low_confidence')
            and cropped_lat.get('lattice_constant_nm') is not None):
        lat_report = cropped_lat
        lat_source = "central_crop"

# Build reflection census (from the reported lattice measurement)
refl_census = []
for r in (lat_report.get('reflections') or []):
    refl_census.append({
        'd_nm': r.get('d_nm'),
        'sigma': r.get('sigma'),
        'order': r.get('order'),
        'on_lattice': r.get('on_lattice'),
        'is_fundamental': r.get('is_fundamental'),
    })

# Geometric NN relationship string
nn_basis = lat_report.get('nn_basis')
centered = lat_report.get('centered_sublattice')
a_nm = lat_report.get('lattice_constant_nm')
nn_nm = lat_report.get('nn_distance_nm')
if nn_basis is not None:
    nn_relation = f"NN = {nn_basis}"
elif centered:
    nn_relation = "NN = a/sqrt2 (centered/perovskite sublattice)"
else:
    nn_relation = "NN = a (primitive)"

# ---------------------------------------------------------------------------
# STEP 2a: Fourier reflection census + superstructure/satellite check
# ---------------------------------------------------------------------------
superstructure = {
    'strongest_satellite_d_nm': None,
    'reflections': [],
    'domain_fraction': None,
    'spot_snr_domain': None,
    'spot_snr_bulk': None,
    'is_mapped_superstructure': None,
    'confirmed_real': None,
    'verdict': None,
}
amp_map = None
domain_mask = None
try:
    det = fourier_reflection_map(image, pixel_size_nm)
    for r in (det.get('reflections') or []):
        superstructure['reflections'].append({
            'd_nm': r.get('d_nm'),
            'sigma': r.get('sigma'),
            'integer_multiple_of': r.get('integer_multiple_of'),
            'is_satellite_candidate': r.get('is_satellite_candidate'),
        })
    sat = det.get('strongest_satellite_d_nm')
    superstructure['strongest_satellite_d_nm'] = sat
    if sat is not None:
        res_map = fourier_reflection_map(image, pixel_size_nm, d_nm=sat)
        amp_map = res_map.get('amplitude_map')
        domain_mask = res_map.get('domain_mask')
        snr_dom = res_map.get('spot_snr_domain')
        snr_bulk = res_map.get('spot_snr_bulk')
        superstructure['domain_fraction'] = res_map.get('domain_fraction')
        superstructure['spot_snr_domain'] = snr_dom
        superstructure['spot_snr_bulk'] = snr_bulk
        superstructure['is_mapped_superstructure'] = res_map.get('is_mapped_superstructure')
        confirmed = (snr_dom is not None and snr_bulk is not None
                     and snr_dom > 3.0 * snr_bulk)
        superstructure['confirmed_real'] = bool(confirmed)
        if confirmed:
            superstructure['verdict'] = (
                "confirmed ordered superstructure domain "
                f"(d={sat:.4f} nm; spot_snr_domain={snr_dom:.2f} >> "
                f"3x bulk={snr_bulk:.2f})"
            )
        else:
            superstructure['verdict'] = (
                "satellite present in PSD but NOT confirmed as a localized ordered domain "
                f"(spot_snr_domain={snr_dom} vs bulk={snr_bulk}); ambiguous / possible edge artifact"
            )
    else:
        superstructure['verdict'] = "no resolvable superstructure/satellite reflection"
except Exception as e:
    superstructure['verdict'] = f"fourier_reflection_map failed: {e}"
    adjustments.append(f"fourier_reflection_map raised: {e}")

# ---------------------------------------------------------------------------
# STEP 2b: per-row mean column-intensity profile on RAW image
#   distinguish smooth shading ramp (artifact) vs sustained per-column
#   Z-contrast step (candidate film/substrate interface).
# ---------------------------------------------------------------------------
# Per-column mean (vertical film/substrate interface would appear as a step
# along the column axis); also per-row mean for a horizontal interface.
col_profile = image.mean(axis=0)   # mean over rows -> one value per column
row_profile = image.mean(axis=1)   # mean over cols -> one value per row


def ramp_vs_step(profile):
    x = np.arange(profile.size, dtype=np.float64)
    # linear fit (smooth global ramp model)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, profile, rcond=None)
    linfit = A @ coef
    resid = profile - linfit
    ramp_span = float(coef[0] * (profile.size - 1))
    resid_std = float(np.std(resid))
    # step detection: best single split minimizing within-segment variance
    n = profile.size
    cs = np.cumsum(profile)
    cs2 = np.cumsum(profile.astype(np.float64) ** 2)
    best_i = None
    best_gain = -np.inf
    total_var = float(np.var(profile)) * n
    margin = max(10, n // 20)
    for i in range(margin, n - margin):
        n1 = i
        n2 = n - i
        s1 = cs[i - 1]
        s2 = cs[-1] - s1
        q1 = cs2[i - 1]
        q2 = cs2[-1] - q1
        var1 = q1 - s1 * s1 / n1
        var2 = q2 - s2 * s2 / n2
        within = var1 + var2
        gain = total_var - within
        if gain > best_gain:
            best_gain = gain
            best_i = i
    if best_i is not None:
        m1 = float(profile[:best_i].mean())
        m2 = float(profile[best_i:].mean())
        step_amp = m2 - m1
    else:
        m1 = m2 = step_amp = None
    global_range = float(profile.max() - profile.min())
    return {
        'ramp_slope_per_px': float(coef[0]),
        'ramp_span_over_profile': ramp_span,
        'residual_std_after_linear': resid_std,
        'best_split_index': int(best_i) if best_i is not None else None,
        'segment_mean_low': m1,
        'segment_mean_high': m2,
        'step_amplitude': step_amp,
        'global_range': global_range,
    }


col_stats = ramp_vs_step(col_profile)
row_stats = ramp_vs_step(row_profile)


def interpret_profile(stats):
    # A sustained Z-contrast step: step amplitude large relative to residual
    # scatter AND relative to the smooth linear ramp span.
    step = stats['step_amplitude']
    resid = stats['residual_std_after_linear']
    ramp = abs(stats['ramp_span_over_profile'])
    grange = stats['global_range']
    if step is None or grange == 0:
        return "flat / no structure"
    step_abs = abs(step)
    # step must exceed residual noise clearly and be a meaningful fraction of range
    strong_step = (resid > 0 and step_abs > 4.0 * resid) and (step_abs > 0.3 * grange)
    if strong_step and step_abs > 1.2 * ramp:
        return "sustained Z-contrast step (candidate film/substrate interface)"
    if ramp > step_abs and ramp > 0.3 * grange:
        return "smooth shading ramp (artifact / thickness gradient)"
    return "no clear step; mild variation consistent with shading/noise"


col_verdict = interpret_profile(col_stats)
row_verdict = interpret_profile(row_stats)

# ---------------------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------------------
saved = {}
if lat_report.get('lattice_constant_nm') is not None:
    pass

np.save('col_intensity_profile.npy', col_profile.astype(np.float32))
saved['col_intensity_profile.npy'] = {
    'description': 'Per-column mean intensity profile (raw image, mean over rows) for interface/ramp discrimination',
    'shape': list(col_profile.shape), 'dtype': 'float32'}
np.save('row_intensity_profile.npy', row_profile.astype(np.float32))
saved['row_intensity_profile.npy'] = {
    'description': 'Per-row mean intensity profile (raw image, mean over cols) for interface/ramp discrimination',
    'shape': list(row_profile.shape), 'dtype': 'float32'}

if amp_map is not None:
    np.save('superstructure_amplitude.npy', np.asarray(amp_map, dtype=np.float32))
    saved['superstructure_amplitude.npy'] = {
        'description': 'Amplitude map of strongest satellite reflection (where superstructure lives)',
        'shape': list(np.asarray(amp_map).shape), 'dtype': 'float32'}
if domain_mask is not None:
    dm = np.asarray(domain_mask).astype(np.uint8)
    np.save('superstructure_domain_mask.npy', dm)
    saved['superstructure_domain_mask.npy'] = {
        'description': 'Null-gated ordered-domain segmentation mask for strongest satellite',
        'shape': list(dm.shape), 'dtype': 'uint8'}

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
n_panels = 4
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ttl = 'Raw image'
if crop_used is not None and lat_source == 'central_crop':
    y0, y1, x0, x1 = crop_used
    from matplotlib.patches import Rectangle
    ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='cyan',
                              facecolor='none', lw=1.5))
    ttl += ' (cyan = lattice ROI)'
ax[0].set_title(ttl)
ax[0].axis('off')

# Lattice summary panel (text)
ax[1].axis('off')
lines = []
lines.append(f"lattice_constant_nm = {a_nm}")
lines.append(f"a1_nm = {lat_report.get('a1_nm')}")
lines.append(f"a2_nm = {lat_report.get('a2_nm')}")
lines.append(f"gamma_deg = {lat_report.get('gamma_deg')}")
lines.append(f"nn_distance_nm = {nn_nm}")
lines.append(f"nn_basis = {nn_basis}  ({nn_relation})")
lines.append(f"centered_sublattice = {centered}")
lines.append(f"explained_fraction = {lat_report.get('explained_fraction')}")
lines.append(f"multi_lattice = {lat_report.get('multi_lattice')}")
lines.append(f"low_confidence = {lat_report.get('low_confidence')}")
lines.append(f"n_reflections = {lat_report.get('n_reflections')}")
lines.append(f"source = {lat_source}")
ax[1].text(0.02, 0.98, "LATTICE METROLOGY\n" + "\n".join(lines),
           va='top', ha='left', fontsize=10, family='monospace',
           transform=ax[1].transAxes)

# Superstructure amplitude map or PSD-note
if amp_map is not None:
    im2 = ax[2].imshow(np.asarray(amp_map), cmap='inferno')
    ax[2].set_title('Strongest satellite amplitude\n' +
                    ('CONFIRMED' if superstructure.get('confirmed_real') else 'unconfirmed'))
    ax[2].axis('off')
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    if domain_mask is not None:
        ax[2].contour(np.asarray(domain_mask).astype(float), levels=[0.5],
                      colors='cyan', linewidths=1.0)
else:
    ax[2].axis('off')
    ax[2].text(0.02, 0.98,
               "SUPERSTRUCTURE / SATELLITE\n\n" +
               f"strongest_satellite_d_nm = {superstructure['strongest_satellite_d_nm']}\n\n" +
               f"verdict:\n{superstructure['verdict']}",
               va='top', ha='left', fontsize=10, family='monospace',
               transform=ax[2].transAxes, wrap=True)

# Intensity profiles
ax[3].plot(col_profile, label='per-column mean (raw)', color='C0')
ax[3].plot(row_profile, label='per-row mean (raw)', color='C1', alpha=0.7)
if col_stats['best_split_index'] is not None:
    ax[3].axvline(col_stats['best_split_index'], color='C0', ls='--', alpha=0.5)
if row_stats['best_split_index'] is not None:
    ax[3].axvline(row_stats['best_split_index'], color='C1', ls=':', alpha=0.5)
ax[3].set_title('Raw mean intensity profiles\n(ramp vs Z-contrast step)')
ax[3].set_xlabel('pixel index (col / row)')
ax[3].set_ylabel('mean intensity')
ax[3].legend(fontsize=8)

fig.suptitle('Lattice metrology + film/substrate discrimination', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig('visualization.png', dpi=100, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------------------
# Assemble results JSON
# ---------------------------------------------------------------------------
summary_bits = []
if a_nm is not None:
    summary_bits.append(
        f"Lattice constant {a_nm:.4f} nm (a1={lat_report.get('a1_nm')}, "
        f"a2={lat_report.get('a2_nm')}, gamma={lat_report.get('gamma_deg')} deg); "
        f"NN={nn_nm} nm ({nn_relation})."
    )
else:
    summary_bits.append("No lattice resolved by measure_lattice_constant.")
summary_bits.append(f"Superstructure: {superstructure['verdict']}.")
summary_bits.append(
    f"Column-profile: {col_verdict}; row-profile: {row_verdict}.")
if adjustments:
    summary_bits.append("Adjustments: " + " ".join(adjustments))

results = {
    "analysis_type": "Deterministic lattice metrology (FFT Bragg geometry via measure_lattice_constant) "
                     "plus film/substrate discrimination (fourier_reflection_map satellite census + "
                     "raw per-row/column intensity profiling).",
    "extracted_features": {
        "pixel_size_nm": pixel_size_nm,
        "pixel_size_x_nm": px_x,
        "pixel_size_y_nm": px_y,
        "pixel_size_source": px_source,
        "lattice_source": lat_source,
        "lattice_constant_nm": lat_report.get('lattice_constant_nm'),
        "lattice_constant_std_nm": lat_report.get('lattice_constant_std_nm'),
        "a1_nm": lat_report.get('a1_nm'),
        "a2_nm": lat_report.get('a2_nm'),
        "gamma_deg": lat_report.get('gamma_deg'),
        "nn_distance_nm": nn_nm,
        "nn_basis": nn_basis,
        "nn_geometric_relation": nn_relation,
        "centered_sublattice": centered,
        "fundamental_d_nm": lat_report.get('fundamental_d_nm'),
        "one_direction_only": lat_report.get('one_direction_only'),
        "explained_fraction": lat_report.get('explained_fraction'),
        "multi_lattice": lat_report.get('multi_lattice'),
        "low_confidence": lat_report.get('low_confidence'),
        "n_reflections": lat_report.get('n_reflections'),
        "reflection_census": refl_census,
        "secondary_lattice": lat_report.get('secondary_lattice'),
        "strongest_satellite_d_nm": superstructure['strongest_satellite_d_nm'],
        "fourier_reflections": superstructure['reflections'],
        "superstructure_domain_fraction": superstructure['domain_fraction'],
        "superstructure_spot_snr_domain": superstructure['spot_snr_domain'],
        "superstructure_spot_snr_bulk": superstructure['spot_snr_bulk'],
        "superstructure_confirmed_real": superstructure['confirmed_real'],
        "superstructure_verdict": superstructure['verdict'],
        "col_profile_stats": col_stats,
        "col_profile_verdict": col_verdict,
        "row_profile_stats": row_stats,
        "row_profile_verdict": row_verdict,
    },
    "quality_metrics": {
        "explained_fraction": lat_report.get('explained_fraction'),
        "multi_lattice_flag": lat_report.get('multi_lattice'),
        "low_confidence_flag": lat_report.get('low_confidence'),
        "n_reflections": lat_report.get('n_reflections'),
        "full_frame_multi_lattice": lat.get('multi_lattice'),
        "full_frame_low_confidence": lat.get('low_confidence'),
        "reran_on_crop": cropped_lat is not None,
    },
    "summary": " ".join(summary_bits),
    "saved_arrays": saved,
}

results = to_native(results)
print(f"IMAGE_ANALYSIS_RESULTS_JSON:{json.dumps(results)}")
