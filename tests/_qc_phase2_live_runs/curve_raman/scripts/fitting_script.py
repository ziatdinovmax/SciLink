import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import medfilt
from scipy.integrate import trapezoid
import lmfit
from lmfit import Parameters, minimize

# ---------------- Load ----------------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2:
        x = data[0].astype(float); y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float); y = data[:, 1].astype(float)
    else:
        x = np.arange(data.shape[0], dtype=float); y = data[:, 0].astype(float)
else:
    # 1-D: need x. Reconstruct from metadata range.
    y = data.astype(float)
    x = np.linspace(91.5332, 1285.74, len(y))

# sort by x
ordr = np.argsort(x)
x = x[ordr]; y = y[ordr]

y_raw = y.copy()

# ---------------- Preprocess ----------------
# Cosmic spike removal (single-pixel) via light median filter comparison
ymed = medfilt(y, kernel_size=5)
noise_est = np.median(np.abs(y - ymed)) * 1.4826
spike_mask = np.abs(y - ymed) > 8 * (noise_est + 1e-9)
y_desp = y.copy()
y_desp[spike_mask] = ymed[spike_mask]

# Intensity-type data: clip negatives to zero
y_desp = np.clip(y_desp, 0, None)

# ---------------- ALS baseline ----------------
def als_baseline(y, lam=1e6, p=0.001, niter=15):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for _ in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

baseline = als_baseline(y_desp, lam=1e6, p=0.001, niter=15)
y_corr = y_desp - baseline

# ALS baseline fraction of total area (fluorescence check)
total_area = trapezoid(np.clip(y_desp, 0, None), x)
base_area = trapezoid(np.clip(baseline, 0, None), x)
als_fraction = float(base_area / total_area) if total_area > 0 else 0.0

# noise floor estimate from corrected signal (robust)
resid_hi = y_corr - medfilt(y_corr, 7)
noise = np.median(np.abs(resid_hi)) * 1.4826
if noise <= 0:
    noise = np.std(y_corr) * 0.01 + 1e-6

# ---------------- Model: pseudo-Voigt ----------------
def pseudo_voigt(x, amp, center, fwhm, eta):
    s = fwhm / 2.3548200450309493
    g = np.exp(-0.5 * ((x - center) / s) ** 2)
    l = 1.0 / (1.0 + ((x - center) / (fwhm / 2.0)) ** 2)
    return amp * (eta * l + (1 - eta) * g)

# Seed bands per plan (revised step 6).
# Plan-permitted shoulder components are ADDED to address verifier findings:
#  - shoulder near 130 for the asymmetric rising flank of the dominant ~141 band
#  - reinitialized ~197 component (non-zero amp/reasonable FWHM) instead of collapsed peak
#  - shoulder near 372 for the under-resolved leading edge of the ~394 band
#  - shoulder near 538 for the oscillatory residual of the ~514 band
# All use the plan's existing shoulder-on-asymmetry/bimodality mechanism; no new model family.
# Each seed: (center, fwhm)
seeds = [
    (130.0, 12.0),   # shoulder on rising flank of dominant band (plan: peak_1_shoulder_center ~130)
    (141.0, 18.0),   # dominant band re-seeded center ~141, broader FWHM (plan: peak_1_center_refined)
    (197.0, 14.0),   # reinitialized real bump ~197 (plan: peak_2_reinitialized_parameters)
    (305.0, 40.0),   # broad; may split into 298/310
    (372.0, 15.0),   # shoulder on low-x leading edge of 394 band (plan: peak_5 shoulder ~372)
    (394.0, 20.0),   # main 394 band
    (514.0, 22.0),   # 514 band
    (538.0, 18.0),   # optional shoulder near 538 (plan: peak_6 optional shoulder)
    (638.0, 25.0),
    (793.0, 25.0),
]

xmin, xmax = 91.5332, 1285.74
fitmask = (x >= xmin) & (x <= xmax)
xf = x[fitmask]; yf = y_corr[fitmask]

def amp_at(c):
    idx = np.argmin(np.abs(xf - c))
    lo = max(0, idx - 5); hi = min(len(xf), idx + 6)
    return max(float(np.max(yf[lo:hi])), noise)

def build_params(seed_list):
    p = Parameters()
    for i, (c, fw) in enumerate(seed_list):
        pre = f'p{i}_'
        p.add(pre + 'center', value=c, min=c - 10, max=c + 10)
        p.add(pre + 'fwhm', value=fw, min=2.0, max=200.0)
        p.add(pre + 'amp', value=amp_at(c), min=0.0)
        p.add(pre + 'eta', value=0.5, min=0.0, max=1.0)
    return p

def model_eval(params, x, n):
    tot = np.zeros_like(x)
    for i in range(n):
        pre = f'p{i}_'
        tot += pseudo_voigt(x, params[pre + 'amp'].value,
                            params[pre + 'center'].value,
                            params[pre + 'fwhm'].value,
                            params[pre + 'eta'].value)
    return tot

def residual(params, x, data, n):
    return model_eval(params, x, n) - data

def do_fit(seed_list):
    n = len(seed_list)
    params = build_params(seed_list)
    out = minimize(residual, params, args=(xf, yf, n), method='leastsq', max_nfev=40000)
    return out, n

# Initial fit with single 305 component (plus plan-permitted shoulders)
out, n = do_fit(seeds)

# ---- Check 305 region residual for bimodality ----
# find the index of the ~305 broad component in current seed list
center305_idx = 3
center305 = out.params[f'p{center305_idx}_center'].value
reg = (xf > center305 - 45) & (xf < center305 + 45)
res_region = yf[reg] - model_eval(out.params, xf, n)[reg]

add_second = False
if res_region.size > 10:
    above = res_region > 3 * noise
    groups = 0
    prev = False
    for a in above:
        if a and not prev:
            groups += 1
        prev = a
    if groups >= 2:
        add_second = True

if add_second:
    # replace the single 305 seed with two components (298/310), keep everything else
    seeds2 = [
        (130.0, 12.0),
        (141.0, 18.0),
        (197.0, 14.0),
        (298.0, 25.0),
        (310.0, 25.0),
        (372.0, 15.0),
        (394.0, 20.0),
        (514.0, 22.0),
        (538.0, 18.0),
        (638.0, 25.0),
        (793.0, 25.0),
    ]
    out, n = do_fit(seeds2)

# ---------------- Evaluate over full array ----------------
fit_corr_full = model_eval(out.params, x, n)
# Save in RAW space: fit = peaks + baseline (overlays raw data)
fit_full = fit_corr_full + baseline
np.save('fit.npy', fit_full.astype(float))

# R^2 / RMSE from the two saved-space arrays (raw data vs fit_full)
residuals_raw = y_raw - fit_full
ss_res = float(np.sum(residuals_raw ** 2))
ss_tot = float(np.sum((y_raw - np.mean(y_raw)) ** 2))
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
rmse = float(np.sqrt(np.mean(residuals_raw ** 2)))

# ---------------- Parameters output ----------------
amps = [out.params[f'p{i}_amp'].value for i in range(n)]
max_amp = max(amps) if amps else 1.0
params_out = {}
for i in range(n):
    pre = f'p{i}_'
    c = out.params[pre + 'center']
    fw = out.params[pre + 'fwhm']
    am = out.params[pre + 'amp']
    et = out.params[pre + 'eta']
    area = trapezoid(pseudo_voigt(x, am.value, c.value, fw.value, et.value), x)
    params_out[f'peak_{i+1}'] = {
        'center': float(c.value),
        'center_err': float(c.stderr) if c.stderr is not None else None,
        'fwhm': float(fw.value),
        'fwhm_err': float(fw.stderr) if fw.stderr is not None else None,
        'amplitude': float(am.value),
        'amplitude_err': float(am.stderr) if am.stderr is not None else None,
        'eta': float(et.value),
        'eta_err': float(et.stderr) if et.stderr is not None else None,
        'integrated_area': float(area),
        'relative_intensity': float(am.value / max_amp) if max_amp > 0 else 0.0,
    }

# ---------------- Visualization ----------------
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False,
                         gridspec_kw={'height_ratios': [3, 1, 1.3]})
ax0, ax1, ax2 = axes

ax0.plot(x, y_raw, color='0.6', alpha=0.4, lw=0.8, label='Raw')
ax0.plot(x, y_desp, color='black', lw=0.9, label='Data')
ax0.plot(x, fit_full, color='red', lw=1.2, label='Fit')
ax0.plot(x, baseline, color='green', ls='--', lw=1.0, label='Baseline')
for i in range(n):
    pre = f'p{i}_'
    comp = pseudo_voigt(x, out.params[pre + 'amp'].value,
                        out.params[pre + 'center'].value,
                        out.params[pre + 'fwhm'].value,
                        out.params[pre + 'eta'].value) + baseline
    ax0.plot(x, comp, lw=0.7, alpha=0.7, label=f'Component {i+1}')
ax0.set_xlim(xmin, xmax)
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(fontsize=7, ncol=2, loc='upper right')

# raw residuals
ax1.plot(x, residuals_raw, color='purple', lw=0.7)
ax1.axhline(0, color='k', lw=0.5)
ax1.set_xlim(xmin, xmax)
ax1.set_ylabel('Residual')

# normalized residual (residual / noise)
ax2.plot(x, residuals_raw / noise, color='teal', lw=0.7)
ax2.axhline(3, color='r', ls='--', lw=0.5)
ax2.axhline(-3, color='r', ls='--', lw=0.5)
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlim(xmin, xmax)
ax2.set_ylabel('Residual / noise')
ax2.set_xlabel('X')

plt.tight_layout()
plt.savefig('visualization.png', dpi=130)
plt.close()

# ---------------- Results JSON ----------------
results = {
    'model_type': f'Sum of {n} pseudo-Voigt peaks on light ALS-corrected baseline (raw-space fit = peaks + ALS baseline)',
    'parameters': params_out,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'als_baseline_fraction_of_area': als_fraction,
    'noise_level': float(noise),
    'number_of_retained_components': int(n),
    'deviation_note': ''
}
print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
