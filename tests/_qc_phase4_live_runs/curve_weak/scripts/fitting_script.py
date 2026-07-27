import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.integrate import trapezoid
from lmfit import Parameters, minimize

# ---------------- Load data ----------------
data = np.load('data.npy')
if data.ndim == 2:
    if data.shape[0] == 2:
        x = data[0].astype(float); y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float); y = data[:, 1].astype(float)
    else:
        x = np.arange(data.shape[-1]).astype(float); y = data.reshape(-1).astype(float)
else:
    y = data.astype(float)
    x = np.linspace(134.647, 1271.48, len(y))

order = np.argsort(x)
x = x[order]; y_raw = y[order].copy()

# Intensity-type spectrum: clip negatives (noise)
y = y_raw.copy()
y[y < 0] = 0.0

# ---------------- Baseline assessment (Rule 1) ----------------
def rolling_min(arr, w):
    n = len(arr); half = w // 2; out = np.empty(n)
    for i in range(n):
        lo = max(0, i - half); hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].min()
    return out

rmin = rolling_min(y, 101)
baseline_frac = trapezoid(rmin, x) / max(trapezoid(y, x), 1e-12)

# ---------------- Mild ALS baseline (weak) ----------------
def als_baseline(yv, lam=1e5, p=0.001, niter=6):
    L = len(yv)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for _ in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * yv)
        w = p * (yv > z) + (1 - p) * (yv < z)
    return z

baseline = als_baseline(y, lam=1e5, p=0.001, niter=6)
y_corr = y - baseline

# Noise estimate from corrected signal (robust)
noise = 1.4826 * np.median(np.abs(y_corr - np.median(y_corr)))
if noise <= 0:
    noise = np.std(y_corr) * 0.1 + 1e-9

# ---------------- Pseudo-Voigt model ----------------
def pseudo_voigt(xv, amp, center, fwhm, eta):
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((xv - center) / s) ** 2)
    l = 1.0 / (1.0 + ((xv - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

# --------------------------------------------------------------------
# REALLOCATED 10-component budget per updated locked plan:
#  1: ~143 lattice mode
#  2: ~190 lattice mode
#  3: ~213 lattice mode (eta/width refined for shape mismatch)
#  4: ~1210 broad band (REALLOCATED from collapsed peak_4; seed A~1080)
#  5: ~272 doublet partner
#  6: ~284 doublet partner
#  7: ~1086 SHARP nu1 core (narrow, high-eta Lorentzian, heavy tails)
#  8: ~1240 broad band (REALLOCATED from collapsed peak_8; seed A~1400)
#  9: ~701 doublet partner
# 10: ~706 doublet partner
# (center_seed, amp_seed, fwhm_seed, eta_seed, is_reclaimed)
# --------------------------------------------------------------------
seeds = [
    (143.0,  1000.0, 8.0,  0.5, False),   # peak_1 lattice
    (190.0,  2000.0, 8.0,  0.5, False),   # peak_2 lattice
    (213.0,  2000.0, 10.0, 0.5, False),   # peak_3 lattice (refined eta/width)
    (1210.0, 1080.0, 28.0, 0.5, True),    # peak_4 REALLOCATED -> 1210 band
    (272.0,  3000.0, 8.0,  0.5, False),   # peak_5 doublet
    (284.0,  3000.0, 8.0,  0.5, False),   # peak_6 doublet
    (1086.0, 22000.0, 6.0, 0.85, False),  # peak_7 SHARP nu1 core, high eta
    (1240.0, 1400.0, 28.0, 0.5, True),    # peak_8 REALLOCATED -> 1240 band
    (701.0,  2000.0, 8.0,  0.5, False),   # peak_9 doublet
    (706.0,  2000.0, 8.0,  0.5, False),   # peak_10 doublet
]

# Improve amplitude seeds from local data (except reclaimed bands which use
# the plan's observed-maximum seed so they don't collapse toward local noise)
def local_amp(c, win=8.0, floor=100.0):
    m = (x > c - win) & (x < c + win)
    if np.any(m):
        return max(float(y_corr[m].max()), floor)
    return 500.0

refined = []
for (c, a, f, e, reclaimed) in seeds:
    if reclaimed:
        # seed at observed local maximum but keep the plan's minimum meaningful A
        la = local_amp(c, win=12.0, floor=a)
        refined.append((c, max(la, a), f, e, reclaimed))
    else:
        refined.append((c, local_amp(c), f, e, reclaimed))
seeds = refined

params = Parameters()
for i, (c, a, f, e, reclaimed) in enumerate(seeds, start=1):
    if reclaimed:
        # amplitude/FWHM lower bounds prevent re-collapse to zero
        params.add(f'amp{i}', value=a, min=300.0, max=1e6)
        params.add(f'fwhm{i}', value=f, min=12.0, max=80.0)
    else:
        params.add(f'amp{i}', value=a, min=0.0, max=1e6)
        params.add(f'fwhm{i}', value=f, min=1.0, max=60.0)
    params.add(f'cen{i}', value=c, min=c - 5.0, max=c + 5.0)
    params.add(f'eta{i}', value=e, min=0.0, max=1.0)
# constant residual offset
params.add('offset', value=0.0, min=-500.0, max=500.0)

N = 10

def model_eval(pars, xv):
    p = pars.valuesdict()
    m = np.full_like(xv, p['offset'])
    for i in range(1, N + 1):
        m = m + pseudo_voigt(xv, p[f'amp{i}'], p[f'cen{i}'], p[f'fwhm{i}'], p[f'eta{i}'])
    return m

def residual(pars, xv, yv):
    return model_eval(pars, xv) - yv

out = minimize(residual, params, args=(x, y_corr), method='leastsq')
bp = out.params

# ---------------- Build fit in RAW space (add baseline back) ----------------
fit_corr = model_eval(bp, x)
fit_raw = fit_corr + baseline  # overlay on raw data

# Output-space contract: save fit that overlays the data array used for R2.
data_for_metric = y  # clipped intensity (raw-space, negatives->0)
fit_for_metric = fit_raw
np.save('fit.npy', fit_for_metric)

# ---------------- R2 / RMSE from the two saved-space arrays ----------------
resid = data_for_metric - fit_for_metric
ss_res = float(np.sum(resid ** 2))
ss_tot = float(np.sum((data_for_metric - np.mean(data_for_metric)) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
rmse = float(np.sqrt(np.mean(resid ** 2)))

# ---------------- Integrated areas per component ----------------
integrated_areas = {}
for i in range(1, N + 1):
    comp = pseudo_voigt(x, bp[f'amp{i}'].value, bp[f'cen{i}'].value,
                        bp[f'fwhm{i}'].value, bp[f'eta{i}'].value)
    integrated_areas[f'peak_{i}'] = float(trapezoid(comp, x))

# residual_max_over_noise
residual_max_over_noise = float(np.max(np.abs(resid)) / noise) if noise > 0 else None
nfree = max(len(resid) - out.nvarys, 1)
reduced_chi_squared = float(ss_res / nfree)

# ---------------- Parameters output ----------------
def perr(name):
    p = bp[name]
    return float(p.stderr) if p.stderr is not None else None

amps = [float(bp[f'amp{i}'].value) for i in range(1, N + 1)]
max_amp = max(amps) if amps else 1.0

parameters = {}
for i in range(1, N + 1):
    parameters[f'peak_{i}'] = {
        'center': float(bp[f'cen{i}'].value),
        'center_err': perr(f'cen{i}'),
        'amplitude': float(bp[f'amp{i}'].value),
        'amplitude_err': perr(f'amp{i}'),
        'fwhm': float(bp[f'fwhm{i}'].value),
        'fwhm_err': perr(f'fwhm{i}'),
        'eta': float(bp[f'eta{i}'].value),
        'relative_intensity': float(bp[f'amp{i}'].value / max_amp) if max_amp > 0 else 0.0,
        'integrated_area': integrated_areas[f'peak_{i}'],
    }

# ---------------- Visualization ----------------
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(4, 2, height_ratios=[3, 1.2, 1.2, 1.5])
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, :])
axz1 = fig.add_subplot(gs[3, 0])
axz2 = fig.add_subplot(gs[3, 1])

ax0.plot(x, y_raw, color='lightgrey', alpha=0.5, lw=0.8, label='Raw')
ax0.plot(x, data_for_metric, color='black', lw=0.9, label='Data')
ax0.plot(x, fit_raw, color='red', lw=1.2, label='Fit')
colors = plt.cm.viridis(np.linspace(0, 0.9, N))
for i in range(1, N + 1):
    comp = pseudo_voigt(x, bp[f'amp{i}'].value, bp[f'cen{i}'].value,
                        bp[f'fwhm{i}'].value, bp[f'eta{i}'].value) + baseline
    ax0.plot(x, comp, color=colors[i - 1], lw=0.7, alpha=0.7,
             label=f'Component {i}')
ax0.plot(x, baseline, color='blue', lw=0.7, ls='--', alpha=0.6, label='Baseline')
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(fontsize=6, ncol=3, loc='upper left')
ax0.set_xlim(x.min(), x.max())

# raw residual
ax1.plot(x, resid, color='purple', lw=0.7)
ax1.axhline(0, color='k', lw=0.5)
ax1.set_ylabel('Residual')
ax1.set_xlim(x.min(), x.max())

# normalized residual
ax2.plot(x, resid / noise, color='teal', lw=0.7)
ax2.axhline(0, color='k', lw=0.5)
ax2.axhline(3, color='r', lw=0.5, ls='--')
ax2.axhline(-3, color='r', lw=0.5, ls='--')
ax2.set_ylabel('Residual / noise')
ax2.set_xlabel('X')
ax2.set_xlim(x.min(), x.max())

# zoom: 1200-1271 region (1210 + 1240 bands)
mz1 = (x >= 1150) & (x <= 1271.48)
axz1.plot(x[mz1], data_for_metric[mz1], color='black', lw=0.9, label='Data')
axz1.plot(x[mz1], fit_raw[mz1], color='red', lw=1.2, label='Fit')
for i in range(1, N + 1):
    comp = pseudo_voigt(x, bp[f'amp{i}'].value, bp[f'cen{i}'].value,
                        bp[f'fwhm{i}'].value, bp[f'eta{i}'].value) + baseline
    axz1.plot(x[mz1], comp[mz1], color=colors[i - 1], lw=0.7, alpha=0.7)
axz1.set_xlim(1150, 1271.48)
axz1.set_title('Zoom 1150-1271')
axz1.set_xlabel('X'); axz1.set_ylabel('Y')

# zoom: nu1 region ~1086
mz2 = (x >= 1050) & (x <= 1140)
axz2.plot(x[mz2], data_for_metric[mz2], color='black', lw=0.9)
axz2.plot(x[mz2], fit_raw[mz2], color='red', lw=1.2)
for i in range(1, N + 1):
    comp = pseudo_voigt(x, bp[f'amp{i}'].value, bp[f'cen{i}'].value,
                        bp[f'fwhm{i}'].value, bp[f'eta{i}'].value) + baseline
    axz2.plot(x[mz2], comp[mz2], color=colors[i - 1], lw=0.7, alpha=0.7)
axz2.set_xlim(1050, 1140)
axz2.set_title('Zoom 1050-1140')
axz2.set_xlabel('X'); axz2.set_ylabel('Y')

fig.tight_layout()
fig.savefig('visualization.png', dpi=130)
plt.close(fig)

# ---------------- Results JSON ----------------
results = {
    'model_type': 'Sum of 10 pseudo-Voigt components + constant offset on weak-ALS baseline-corrected Raman spectrum (baseline added back for raw-space overlay); two components reallocated to real bands at ~1210 and ~1240 cm-1 with amplitude/FWHM lower bounds; nu1 modeled by single sharp high-eta Lorentzian-leaning core',
    'parameters': parameters,
    'fit_quality': {'r_squared': r2, 'rmse': rmse},
    'baseline_fraction': float(baseline_frac),
    'noise_level': float(noise),
    'residual_max_over_noise': residual_max_over_noise,
    'reduced_chi_squared': reduced_chi_squared,
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
