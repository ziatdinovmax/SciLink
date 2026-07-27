import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import trapezoid

# ---------------------------------------------------------------
# 1. Load RAW data
# ---------------------------------------------------------------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2:
        x = data[0].astype(float)
        y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
    else:
        # assume first two columns
        x = data[:, 0].astype(float)
        y = data[:, 1].astype(float)
else:
    y = data.astype(float)
    x = np.linspace(134.647, 1271.48, y.size)

# sort by x
idx = np.argsort(x)
x = x[idx]
y = y[idx]
N = y.size

# ---------------------------------------------------------------
# 2. Preprocessing: intensity-type Raman -> clip negatives (noise).
#    Baseline is near-flat per plan; fit a LINEAR baseline as a model
#    parameter (light correction). No smoothing (data assumed clean).
# ---------------------------------------------------------------
y_raw = y.copy()
y = np.clip(y, 0, None)

# ---------------------------------------------------------------
# Model: sum of 13 pseudo-Voigt profiles + linear baseline
# ---------------------------------------------------------------
def pseudo_voigt(x, amp, center, fwhm, eta):
    fwhm = np.maximum(fwhm, 1e-6)
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1.0 - eta) * g)

NCOMP = 13
# seed centers per plan
seed_centers = np.array([142.8, 161.0, 189.0, 213.0, 272.0, 284.0,
                         701.0, 706.0, 1085.0, 1088.0, 1100.0,
                         1207.0, 1238.0])

def local_amp(c):
    m = np.abs(x - c) < 6.0
    if not np.any(m):
        return 100.0
    return max(float(np.max(y[m])), 10.0)

# seed amplitudes
seed_amps = np.array([local_amp(c) for c in seed_centers])
# v1 core (idx 8) carries sharp apex ~22000; wing (idx 9) broader/lower
seed_amps[8] = 20000.0
seed_amps[9] = 4000.0
seed_amps[10] = 2000.0  # ~1100 shoulder

# seed FWHM
seed_fwhm = np.array([6.0, 6.0, 10.0, 8.0, 8.0, 8.0,
                      4.0, 4.0, 3.0, 9.0, 12.0, 8.0, 8.0])
# seed eta (mixing)
seed_eta = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5])

# linear baseline seeds
b_lo = np.median(y[:50])
b_hi = np.median(y[-50:])
slope0 = (b_hi - b_lo) / (x[-1] - x[0])
intercept0 = b_lo - slope0 * x[0]

# ---------------------------------------------------------------
# Parameter packing: [amp,center,fwhm,eta]*13 + [slope,intercept]
# ---------------------------------------------------------------
def pack(amps, centers, fwhms, etas, slope, intercept):
    p = np.empty(NCOMP * 4 + 2)
    p[0:NCOMP*4:4] = amps
    p[1:NCOMP*4:4] = centers
    p[2:NCOMP*4:4] = fwhms
    p[3:NCOMP*4:4] = etas
    p[-2] = slope
    p[-1] = intercept
    return p

def unpack(p):
    amps = p[0:NCOMP*4:4]
    centers = p[1:NCOMP*4:4]
    fwhms = p[2:NCOMP*4:4]
    etas = p[3:NCOMP*4:4]
    slope = p[-2]
    intercept = p[-1]
    return amps, centers, fwhms, etas, slope, intercept

def model_full(p, xx):
    amps, centers, fwhms, etas, slope, intercept = unpack(p)
    out = slope * xx + intercept
    for i in range(NCOMP):
        out = out + pseudo_voigt(xx, amps[i], centers[i], fwhms[i], etas[i])
    return out

# ---------------------------------------------------------------
# Region weighting to protect weak lattice modes (135-206) and
# stabilize other regions.
# ---------------------------------------------------------------
def make_weights(xx):
    w = np.ones_like(xx)
    # boost weak lattice region 135-206
    w[(xx >= 135) & (xx <= 206)] = 4.0
    # boost v4 doublet region
    w[(xx >= 690) & (xx <= 715)] = 2.5
    # boost high-x doublet
    w[(xx >= 1195) & (xx <= 1250)] = 2.0
    return w

weights = make_weights(x)

# ---------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------
lo = np.empty(NCOMP * 4 + 2)
hi = np.empty(NCOMP * 4 + 2)
for i in range(NCOMP):
    lo[4*i+0] = 0.0                       # amp >= 0
    hi[4*i+0] = np.inf
    lo[4*i+1] = seed_centers[i] - 5.0     # center +/- 5
    hi[4*i+1] = seed_centers[i] + 5.0
    lo[4*i+2] = 0.5                       # fwhm > 0
    hi[4*i+2] = 60.0
    lo[4*i+3] = 0.0                       # eta in [0,1]
    hi[4*i+3] = 1.0
# v4 doublet (idx 6,7): bound narrow widths to avoid ringing
hi[4*6+2] = 8.0
hi[4*7+2] = 8.0
lo[4*6+2] = 2.0
lo[4*7+2] = 2.0
# v1 core (idx 8): narrow, allow toward Lorentzian
hi[4*8+2] = 6.0
lo[4*8+3] = 0.3
# v1 wing (idx 9): broader
lo[4*9+2] = 4.0
hi[4*9+2] = 30.0
# baseline
lo[-2] = -50.0; hi[-2] = 50.0
lo[-1] = -1e5; hi[-1] = 1e5

# ---------------------------------------------------------------
# Staged local fits to get stable seeds
# ---------------------------------------------------------------
def fit_region_local(comp_idx, xmask, p_current):
    """Fit only the amp/center/fwhm/eta of comp_idx components over xmask."""
    sub_lo = []
    sub_hi = []
    p0 = []
    slots = []
    for ci in comp_idx:
        for k in range(4):
            slot = 4*ci + k
            slots.append(slot)
            p0.append(p_current[slot])
            sub_lo.append(lo[slot])
            sub_hi.append(hi[slot])
    p0 = np.array(p0)
    sub_lo = np.array(sub_lo)
    sub_hi = np.array(sub_hi)
    xx = x[xmask]
    yy = y[xmask]
    ww = np.sqrt(weights[xmask])
    # fixed baseline from current
    base_slope = p_current[-2]
    base_int = p_current[-1]

    def resid(sp):
        pfull = p_current.copy()
        for s, v in zip(slots, sp):
            pfull[s] = v
        return (model_full(pfull, xx) - yy) * ww

    try:
        res = least_squares(resid, p0, bounds=(sub_lo, sub_hi),
                            method='trf', max_nfev=4000)
        for s, v in zip(slots, res.x):
            p_current[s] = v
    except Exception:
        pass
    return p_current

p = pack(seed_amps, seed_centers, seed_fwhm, seed_eta, slope0, intercept0)

# clip seeds into bounds
p = np.minimum(np.maximum(p, lo), hi)

# Stage 1: local region fits
# lattice region 135-300
p = fit_region_local([0,1,2,3,4,5], (x>=135)&(x<=300), p)
# v4 doublet 690-715
p = fit_region_local([6,7], (x>=685)&(x<=720), p)
# v1 complex 1070-1115
p = fit_region_local([8,9,10], (x>=1065)&(x<=1120), p)
# high-x doublet 1190-1260
p = fit_region_local([11,12], (x>=1185)&(x<=1265), p)

# ---------------------------------------------------------------
# Stage 2: global simultaneous fit, all 13 components + baseline
# ---------------------------------------------------------------
sqrtw = np.sqrt(weights)

def resid_global(p):
    return (model_full(p, x) - y) * sqrtw

res = least_squares(resid_global, p, bounds=(lo, hi),
                    method='trf', max_nfev=20000)
p = res.x

# ---------------------------------------------------------------
# Parameter uncertainties from Jacobian (unweighted residual scaling)
# ---------------------------------------------------------------
fit_y = model_full(p, x)
residual = y - fit_y
try:
    J = res.jac
    dof = max(1, len(y) - len(p))
    # weighted residual variance
    wres = res.fun
    s_sq = np.sum(wres**2) / dof
    JTJ = J.T @ J
    cov = np.linalg.pinv(JTJ) * s_sq
    perr = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
except Exception:
    perr = np.full_like(p, np.nan)

# ---------------------------------------------------------------
# Metrics computed from SAVED arrays (y_raw vs fit_y)
# fit_y includes the baseline -> same space as raw data
# ---------------------------------------------------------------
ss_res = np.sum((y_raw - fit_y) ** 2)
ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
rmse = float(np.sqrt(np.mean((y_raw - fit_y) ** 2)))

# ---------------------------------------------------------------
# Per-region residual diagnostics: sign changes and RMS
# ---------------------------------------------------------------
def region_diag(xmask):
    r = (y_raw - fit_y)[xmask]
    if r.size < 2:
        return {'sign_changes': 0, 'rms': 0.0}
    sgn = np.sign(r)
    sgn[sgn == 0] = 1
    sc = int(np.sum(sgn[1:] != sgn[:-1]))
    return {'sign_changes': sc, 'rms': float(np.sqrt(np.mean(r**2)))}

regions = {
    'lattice_135_300': region_diag((x>=135)&(x<=300)),
    'v4_685_720': region_diag((x>=685)&(x<=720)),
    'v1_1065_1120': region_diag((x>=1065)&(x<=1120)),
    'highx_1185_1265': region_diag((x>=1185)&(x<=1265)),
}

# ---------------------------------------------------------------
# Save fit.npy (evaluated at all N x-points, includes baseline)
# ---------------------------------------------------------------
np.save('fit.npy', fit_y.astype(float))

# ---------------------------------------------------------------
# Assemble parameters
# ---------------------------------------------------------------
amps, centers, fwhms, etas, slope, intercept = unpack(p)
ea, ec, ef, ee = perr[0:NCOMP*4:4], perr[1:NCOMP*4:4], perr[2:NCOMP*4:4], perr[3:NCOMP*4:4]

params = {}
for i in range(NCOMP):
    params[f'peak_{i+1}'] = {
        'center': float(centers[i]), 'center_err': float(ec[i]),
        'amplitude': float(amps[i]), 'amplitude_err': float(ea[i]),
        'fwhm': float(fwhms[i]), 'fwhm_err': float(ef[i]),
        'eta': float(etas[i]), 'eta_err': float(ee[i]),
    }
params['baseline'] = {'slope': float(slope), 'slope_err': float(perr[-2]),
                      'intercept': float(intercept), 'intercept_err': float(perr[-1])}

# derived quantities
nu1_core_area = float(trapezoid(pseudo_voigt(x, amps[8], centers[8], fwhms[8], etas[8]), x))
nu1_wing_area = float(trapezoid(pseudo_voigt(x, amps[9], centers[9], fwhms[9], etas[9]), x))
derived = {
    'nu1_core_wing_amp_ratio': float(amps[8] / amps[9]) if amps[9] != 0 else None,
    'nu1_combined_area': nu1_core_area + nu1_wing_area,
    'nu4_splitting': float(centers[7] - centers[6]),
    'nu4_amp_ratio': float(amps[7] / amps[6]) if amps[6] != 0 else None,
    'highx_splitting': float(centers[12] - centers[11]),
    'highx_amp_ratio': float(amps[12] / amps[11]) if amps[11] != 0 else None,
    'region_residuals': regions,
}

# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
noise = np.std((y_raw - fit_y)[(x>=134)&(x<=140)]) if np.any((x>=134)&(x<=140)) else np.std(y_raw - fit_y)
if not np.isfinite(noise) or noise <= 0:
    noise = np.std(y_raw - fit_y) or 1.0

fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(4, 3, height_ratios=[3, 1.2, 1.2, 1.6], hspace=0.45, wspace=0.3)

ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(x, y_raw, color='0.6', lw=0.8, alpha=0.5, label='Raw data')
ax_main.plot(x, y, color='k', lw=0.8, label='Data')
ax_main.plot(x, fit_y, color='r', lw=1.2, label='Fit')
baseln = slope * x + intercept
for i in range(NCOMP):
    comp = baseln + pseudo_voigt(x, amps[i], centers[i], fwhms[i], etas[i])
    ax_main.plot(x, comp, lw=0.6, alpha=0.7, label=f'Component {i+1}')
ax_main.set_yscale('symlog', linthresh=100)
ax_main.set_xlabel('X'); ax_main.set_ylabel('Y')
ax_main.set_title('Data and Fit')
ax_main.set_xlim(x.min(), x.max())
ax_main.legend(fontsize=6, ncol=4, loc='upper left')

# residual panel (raw + normalized)
ax_res = fig.add_subplot(gs[1, :])
resid_raw = y_raw - fit_y
ax_res.plot(x, resid_raw, color='b', lw=0.6, label='Residuals')
ax_res.axhline(0, color='k', lw=0.5)
ax_res.set_xlim(x.min(), x.max())
ax_res.set_xlabel('X'); ax_res.set_ylabel('Residual')
ax_res.legend(fontsize=7, loc='upper right')

ax_nres = fig.add_subplot(gs[2, :])
ax_nres.plot(x, resid_raw / noise, color='m', lw=0.6, label='Residuals / noise')
ax_nres.axhline(0, color='k', lw=0.5)
ax_nres.set_xlim(x.min(), x.max())
ax_nres.set_xlabel('X'); ax_nres.set_ylabel('Norm. resid.')
ax_nres.legend(fontsize=7, loc='upper right')

# zoomed region sub-panels (independent x-axes)
zoom_specs = [(135, 300, 'lattice'), (685, 720, 'v4'), (1065, 1120, 'v1'), (1185, 1265, 'high-x')]
for j, (a, b, _lab) in enumerate(zoom_specs):
    col = j % 3
    if j < 3:
        axz = fig.add_subplot(gs[3, col])
    else:
        axz = fig.add_subplot(gs[3, 2])
    m = (x >= a) & (x <= b)
    axz.plot(x[m], y_raw[m], color='k', lw=0.8)
    axz.plot(x[m], fit_y[m], color='r', lw=1.0)
    axz.plot(x[m], resid_raw[m], color='b', lw=0.6)
    axz.axhline(0, color='0.5', lw=0.4)
    axz.set_xlim(a, b)
    axz.set_xlabel('X'); axz.set_ylabel('Y')
    axz.tick_params(labelsize=6)

fig.savefig('visualization.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ---------------------------------------------------------------
# Output JSON
# ---------------------------------------------------------------
results = {
    'model_type': 'Sum of 13 pseudo-Voigt profiles on a linear baseline (nu1 core/wing split, nu4 doublet, lattice modes, high-x doublet)',
    'parameters': params,
    'derived': derived,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': rmse},
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
