import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import least_squares
from scipy.integrate import trapezoid

# ----------------------------------------------------------------------
# 1. Load RAW data
# ----------------------------------------------------------------------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2:
        x_raw, y_raw = data[0].astype(float), data[1].astype(float)
    elif data.shape[1] == 2:
        x_raw, y_raw = data[:, 0].astype(float), data[:, 1].astype(float)
    else:
        x_raw, y_raw = data[0].astype(float), data[1].astype(float)
else:
    raise ValueError('Unexpected data shape')

# sort by x
order = np.argsort(x_raw)
x = x_raw[order]
y_raw = y_raw[order]

# ----------------------------------------------------------------------
# 2. Preprocessing: intensity-type Raman spectrum.
#    - Remove single-pixel cosmic spikes via median filter comparison
#    - Clip negatives to zero (noise)
# ----------------------------------------------------------------------
from scipy.signal import medfilt
y_med = medfilt(y_raw, kernel_size=5)
resid_med = y_raw - y_med
noise_est = np.median(np.abs(resid_med - np.median(resid_med))) * 1.4826
spike_mask = resid_med > 10 * max(noise_est, 1.0)
y_proc = y_raw.copy()
y_proc[spike_mask] = y_med[spike_mask]
y_proc = np.clip(y_proc, 0, None)

# ----------------------------------------------------------------------
# Background assessment (Planning step 1): rolling-minimum baseline estimate
# to confirm baseline is negligible (<<40% of area -> no ALS needed).
# ----------------------------------------------------------------------
from scipy.ndimage import minimum_filter1d, uniform_filter1d
win = max(51, (len(x) // 20) | 1)
rmin = minimum_filter1d(y_proc, size=win)
rmin = uniform_filter1d(rmin, size=win)
total_area = trapezoid(y_proc, x)
base_area = trapezoid(rmin, x)
base_frac = base_area / total_area if total_area > 0 else 0.0
# Per plan: expected <<40% -> flat (constant, near-zero) baseline suffices. No ALS.

# ----------------------------------------------------------------------
# 3. Model: sum of 7 symmetric pseudo-Voigt profiles + constant baseline
#    Reseeded per updated plan:
#      p1 ~128 (low-freq shoulder of 141 band)   [was spurious peak_2 @196]
#      p2 ~141 (main sharp band)
#      p3 ~375 (rising-flank component of 394)    [was spurious peak_3 @301]
#      p4 ~394 (main strong band)
#      p5 ~515, p6 ~636, p7 ~792
# ----------------------------------------------------------------------
def pseudo_voigt(x, amp, center, fwhm, eta):
    s = fwhm / 2.35482
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

N_PEAKS = 7
# component order: [128-shoulder, 141-main, 375-flank, 394-main, 515, 636, 792]
centers0 = np.array([128., 141., 377., 394., 515., 636., 792.])
amps0    = np.array([50000., 347000., 40000., 205000., 152000., 24000., 4000.])
fwhm0    = np.array([20., 11., 12., 18., 20., 25., 30.])
eta0     = np.array([0.5] * N_PEAKS)
base0    = 0.0

# center bounds: shoulder/flank loosened to +/-15; main strong bands +/-10
center_halfwidth = np.array([15., 10., 15., 10., 10., 12., 15.])

def pack(amps, centers, fwhms, etas, base):
    p = []
    for i in range(N_PEAKS):
        p += [amps[i], centers[i], fwhms[i], etas[i]]
    p.append(base)
    return np.array(p)

def model(p, x):
    out = np.full_like(x, p[-1])
    for i in range(N_PEAKS):
        a, c, w, e = p[4*i:4*i+4]
        out = out + pseudo_voigt(x, a, c, w, e)
    return out

p0 = pack(amps0, centers0, fwhm0, eta0, base0)

# bounds
lo, hi = [], []
for i in range(N_PEAKS):
    lo += [0.0, centers0[i] - center_halfwidth[i], 0.5, 0.0]
    hi += [np.inf, centers0[i] + center_halfwidth[i], 200.0, 1.0]
lo.append(-1e4)
hi.append(1e5)
lo = np.array(lo); hi = np.array(hi)
p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

# ----------------------------------------------------------------------
# Minimum-separation constraint (>=9 cm^-1) between the reseeded pairs
# 128/141 and 375/394 to prevent re-degeneration. Enforced as a soft
# penalty added to the residual vector (per plan: keep new components from
# collapsing onto their neighbors).
# ----------------------------------------------------------------------
MIN_SEP = 9.0
SEP_W = 5000.0  # penalty weight (counts) per cm^-1 of violation

def resid(p):
    r = model(p, x) - y_proc
    c128, c141 = p[1*0+1], p[4*1+1]  # centers of comp 0 and comp 1
    c375, c394 = p[4*2+1], p[4*3+1]  # centers of comp 2 and comp 3
    pen = []
    v1 = MIN_SEP - (c141 - c128)
    pen.append(SEP_W * max(v1, 0.0))
    v2 = MIN_SEP - (c394 - c375)
    pen.append(SEP_W * max(v2, 0.0))
    return np.concatenate([r, np.array(pen)])

# ----------------------------------------------------------------------
# Staged fit (per plan): fix strong main bands, fit shoulder/flank comps,
# then release all for a final joint refinement.
# ----------------------------------------------------------------------
# Stage A: fix main bands (141=comp1, 394=comp3, 515=comp4) centers/fwhm by
# narrowing their bounds tightly around seeds; let shoulder/flank move.
lo_a = lo.copy(); hi_a = hi.copy()
for mi in (1, 3, 4):
    lo_a[4*mi+1] = centers0[mi] - 1.0; hi_a[4*mi+1] = centers0[mi] + 1.0
    lo_a[4*mi+2] = max(0.5, fwhm0[mi] - 3.0); hi_a[4*mi+2] = fwhm0[mi] + 3.0
p0a = np.clip(p0, lo_a + 1e-9, hi_a - 1e-9)
try:
    resA = least_squares(resid, p0a, bounds=(lo_a, hi_a), method='trf',
                         max_nfev=20000, x_scale='jac')
    p_start = resA.x
except Exception:
    p_start = p0
p_start = np.clip(p_start, lo + 1e-9, hi - 1e-9)

# Stage B: release all parameters for joint refinement
res = least_squares(resid, p_start, bounds=(lo, hi), method='trf',
                    max_nfev=40000, x_scale='jac')
pf = res.x

# parameter errors from Jacobian
try:
    J = res.jac
    dof = max(1, len(x) - len(pf))
    s2 = 2.0 * res.cost / dof
    JTJ = J.T @ J
    cov = np.linalg.pinv(JTJ) * s2
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))
except Exception:
    perr = np.full_like(pf, np.nan)

# ----------------------------------------------------------------------
# 4. Fit quality (over full modelled domain, saved-array space)
# ----------------------------------------------------------------------
fit_y = model(pf, x)
residuals = y_proc - fit_y
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_proc - np.mean(y_proc)) ** 2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
rmse = np.sqrt(np.mean(residuals ** 2))
noise_level = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
if noise_level <= 0:
    noise_level = rmse

# residual diagnostics (per plan)
r_over_n = residuals / noise_level
lag1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if len(residuals) > 2 else 0.0
frac_3sig = float(np.mean(np.abs(r_over_n) > 3.0))

def sign_changes_in(region_lo, region_hi):
    m = (x >= region_lo) & (x <= region_hi)
    rr = residuals[m]
    if len(rr) < 2:
        return 0
    s = np.sign(rr)
    s = s[s != 0]
    return int(np.sum(np.diff(s) != 0))

sc_peak1 = sign_changes_in(91.5, 166.2)
sc_peak4 = sign_changes_in(315.4, 464.7)

# ----------------------------------------------------------------------
# 5. Save fit.npy (evaluated at same x-points, in raw x order)
# ----------------------------------------------------------------------
inv = np.argsort(order)
fit_original_order = fit_y[inv]
np.save('fit.npy', fit_original_order.astype(float))

# ----------------------------------------------------------------------
# 6. Visualization
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1.2, 1.2], hspace=0.4)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax2 = fig.add_subplot(gs[2])  # independent x-axis (peak 1 zoom)
ax3 = fig.add_subplot(gs[3])  # independent x-axis (peak 4 zoom)

cmap = plt.cm.viridis(np.linspace(0, 0.9, N_PEAKS))

# main panel
ax0.plot(x, y_raw, color='0.75', lw=0.8, alpha=0.5, label='Raw data')
ax0.plot(x, y_proc, 'k.', ms=2, label='Data')
ax0.plot(x, fit_y, 'r-', lw=1.3, label='Fit')
for i in range(N_PEAKS):
    a, c, w, e = pf[4*i:4*i+4]
    comp = pseudo_voigt(x, a, c, w, e) + pf[-1]
    ax0.plot(x, comp, '-', color=cmap[i], lw=0.9, alpha=0.8,
             label=f'Component {i+1}')
ax0.set_yscale('symlog', linthresh=1000)
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(fontsize=7, ncol=2, loc='upper right')
ax0.set_xlim(x.min(), x.max())

# residual panel (normalized)
ax1.plot(x, r_over_n, 'b-', lw=0.7)
ax1.axhline(0, color='k', lw=0.5)
ax1.axhline(3, color='r', ls='--', lw=0.5)
ax1.axhline(-3, color='r', ls='--', lw=0.5)
ax1.set_ylabel('Resid / noise')
ax1.set_xlabel('X')
ax1.set_xlim(x.min(), x.max())

# zoom panel: peak 1 region (91.5-166.2)
z1lo, z1hi = 91.5, 200.0
z1 = (x >= z1lo) & (x <= z1hi)
ax2.plot(x[z1], y_proc[z1], 'k.', ms=2, label='Data')
ax2.plot(x[z1], fit_y[z1], 'r-', lw=1.2, label='Fit')
for i in range(N_PEAKS):
    a, c, w, e = pf[4*i:4*i+4]
    if z1lo <= c <= z1hi:
        comp = pseudo_voigt(x, a, c, w, e) + pf[-1]
        ax2.plot(x[z1], comp[z1], '-', color=cmap[i], lw=0.9, alpha=0.8)
ax2.set_xlim(z1lo, z1hi)
ax2.set_ylabel('Y')
ax2.set_xlabel('X (zoom)')
ax2.legend(fontsize=7, loc='upper right')

# zoom panel: peak 4 region (315.4-464.7)
z4lo, z4hi = 315.4, 464.7
z4 = (x >= z4lo) & (x <= z4hi)
ax3.plot(x[z4], y_proc[z4], 'k.', ms=2, label='Data')
ax3.plot(x[z4], fit_y[z4], 'r-', lw=1.2, label='Fit')
for i in range(N_PEAKS):
    a, c, w, e = pf[4*i:4*i+4]
    if z4lo <= c <= z4hi:
        comp = pseudo_voigt(x, a, c, w, e) + pf[-1]
        ax3.plot(x[z4], comp[z4], '-', color=cmap[i], lw=0.9, alpha=0.8)
ax3.set_xlim(z4lo, z4hi)
ax3.set_ylabel('Y')
ax3.set_xlabel('X (zoom)')
ax3.legend(fontsize=7, loc='upper right')

fig.savefig('visualization.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ----------------------------------------------------------------------
# 7. Results JSON
# ----------------------------------------------------------------------
max_amp = np.max(pf[0:4*N_PEAKS:4]) if N_PEAKS > 0 else 1.0
params = {}
for i in range(N_PEAKS):
    a, c, w, e = pf[4*i:4*i+4]
    ae, ce, we, ee = perr[4*i:4*i+4]
    area_l = a * (np.pi * w / 2.0)
    s = w / 2.35482
    area_g = a * s * np.sqrt(2 * np.pi)
    area = e * area_l + (1 - e) * area_g
    params[f'peak_{i+1}'] = {
        'center': float(c), 'center_err': float(ce),
        'amplitude': float(a), 'amplitude_err': float(ae),
        'fwhm': float(w), 'fwhm_err': float(we),
        'eta': float(e), 'eta_err': float(ee),
        'integrated_area': float(area),
        'relative_intensity': float(a / max_amp) if max_amp > 0 else 0.0,
    }

results = {
    'model_type': 'Sum of 7 symmetric pseudo-Voigt profiles on a flat (constant) baseline; full-range fit (91-1286 cm^-1) via staged Trust Region Reflective least squares. Reseeded 128/141 shoulder+main pair and 375/394 flank+main pair with min-separation >=9 cm^-1 soft constraint. Baseline area fraction ~%.3f (rolling-min estimate) -> flat baseline used, no ALS.' % base_frac,
    'parameters': params,
    'baseline_offset': float(pf[-1]),
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'residual_diagnostics': {
        'residual_rms_over_noise': float(rmse / noise_level) if noise_level > 0 else None,
        'lag1_autocorrelation': lag1,
        'fraction_beyond_3sigma': frac_3sig,
        'sign_changes_peak1_region': sc_peak1,
        'sign_changes_peak4_region': sc_peak4,
    },
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
