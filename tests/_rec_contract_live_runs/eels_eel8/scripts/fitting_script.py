import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from scipy.integrate import trapezoid

# ---------------- Load data ----------------
data = np.load('data.npy')
data = np.asarray(data)
if data.ndim == 2:
    if data.shape[0] == 2 and data.shape[1] != 2:
        x = data[0].astype(float); y = data[1].astype(float)
    elif data.shape[1] == 2:
        x = data[:, 0].astype(float); y = data[:, 1].astype(float)
    else:
        x = data[0].astype(float); y = data[1].astype(float)
else:
    # 1-D: assume y, build x from metadata range
    y = data.astype(float)
    x = np.linspace(683.9, 730.15, y.size)

# sort by x
sidx = np.argsort(x)
x = x[sidx]; y = y[sidx]

# Pre-edge flat at zero per plan; no baseline subtraction, no smoothing (clean edge data).

# ---------------- Model definitions ----------------
def gaussian(x, amp, cen, fwhm):
    sig = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amp * np.exp(-(x - cen)**2 / (2.0 * sig**2))

def arctan_step(x, height, edge, width):
    # arctan continuum step: rises from 0 to `height`
    return height * (0.5 + (1.0/np.pi) * np.arctan((x - edge) / width))

def full_model(x, amp_l3, cen_l3, fwhm_l3,
               amp_l3s, cen_l3s, fwhm_l3s,
               amp_l2, cen_l2, fwhm_l2,
               amp_l2s, cen_l2s, fwhm_l2s,
               step_l3_h, step_l3_e, step_l3_w,
               step_l2_h, step_l2_e, step_l2_w):
    g_l3 = gaussian(x, amp_l3, cen_l3, fwhm_l3)
    g_l3s = gaussian(x, amp_l3s, cen_l3s, fwhm_l3s)
    g_l2 = gaussian(x, amp_l2, cen_l2, fwhm_l2)
    g_l2s = gaussian(x, amp_l2s, cen_l2s, fwhm_l2s)
    s_l3 = arctan_step(x, step_l3_h, step_l3_e, step_l3_w)
    s_l2 = arctan_step(x, step_l2_h, step_l2_e, step_l2_w)
    return g_l3 + g_l3s + g_l2 + g_l2s + s_l3 + s_l2

model = Model(full_model)

# ---------------- Fit domain ----------------
fit_lo, fit_hi = 684.0, 730.0
mask = (x >= fit_lo) & (x <= fit_hi)
xf = x[mask]; yf = y[mask]

# ---------------- Initial parameters ----------------
params = Parameters()
# L3 main
params.add('amp_l3', value=800.0, min=0.0)
params.add('cen_l3', value=710.5, min=710.0, max=711.0)
params.add('fwhm_l3', value=1.2, min=0.5, max=3.0)
# L3 shoulder / pre-peak
params.add('amp_l3s', value=350.0, min=0.0)
params.add('cen_l3s', value=708.9, min=708.4, max=709.4)
params.add('fwhm_l3s', value=0.8, min=0.5, max=3.0)
# L2 main
params.add('amp_l2', value=130.0, min=0.0)
params.add('cen_l2', value=723.7, min=723.2, max=724.2)
params.add('fwhm_l2', value=2.0, min=0.5, max=3.0)
# L2 secondary / multiplet
params.add('amp_l2s', value=90.0, min=0.0)
params.add('cen_l2s', value=722.0, min=721.5, max=722.5)
params.add('fwhm_l2s', value=1.2, min=0.5, max=3.0)
# arctan steps
params.add('step_l3_h', value=150.0, min=0.0)
params.add('step_l3_e', value=708.0, min=707.0, max=709.0)
params.add('step_l3_w', value=0.8, min=0.1, max=3.0)
params.add('step_l2_h', value=80.0, min=0.0)
params.add('step_l2_e', value=721.0, min=720.0, max=722.0)
params.add('step_l2_w', value=0.8, min=0.1, max=3.0)

# ---------------- Fit (Levenberg-Marquardt) ----------------
result = model.fit(yf, params, x=xf, method='leastsq')

pv = result.params

# ---------------- Evaluate full model at ALL x-points ----------------
fit_full = full_model(x, **{k: pv[k].value for k in pv})
np.save('fit.npy', np.asarray(fit_full, dtype=float))

# ---------------- Fit quality over full modelled domain ----------------
yfit_dom = fit_full[mask]
resid = yf - yfit_dom
ss_res = np.sum(resid**2)
ss_tot = np.sum((yf - np.mean(yf))**2)
r_squared = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')
rmse = np.sqrt(np.mean(resid**2))

# ---------------- Gaussian areas (analytic) & L3/L2 ratio ----------------
def gauss_area(amp, fwhm):
    sig = fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))
    return amp * sig * np.sqrt(2.0*np.pi)

area_l3 = gauss_area(pv['amp_l3'].value, pv['fwhm_l3'].value)
area_l3s = gauss_area(pv['amp_l3s'].value, pv['fwhm_l3s'].value)
area_l2 = gauss_area(pv['amp_l2'].value, pv['fwhm_l2'].value)
area_l2s = gauss_area(pv['amp_l2s'].value, pv['fwhm_l2s'].value)

l3_total_area = area_l3 + area_l3s
l2_total_area = area_l2 + area_l2s
l3_l2_ratio = l3_total_area / l2_total_area if l2_total_area > 0 else float('nan')

def gv(name):
    return float(pv[name].value)
def ge(name):
    e = pv[name].stderr
    return float(e) if e is not None else None

# ---------------- Component curves for plotting ----------------
comp_l3 = gaussian(x, gv('amp_l3'), gv('cen_l3'), gv('fwhm_l3'))
comp_l3s = gaussian(x, gv('amp_l3s'), gv('cen_l3s'), gv('fwhm_l3s'))
comp_l2 = gaussian(x, gv('amp_l2'), gv('cen_l2'), gv('fwhm_l2'))
comp_l2s = gaussian(x, gv('amp_l2s'), gv('cen_l2s'), gv('fwhm_l2s'))
comp_s3 = arctan_step(x, gv('step_l3_h'), gv('step_l3_e'), gv('step_l3_w'))
comp_s2 = arctan_step(x, gv('step_l2_h'), gv('step_l2_e'), gv('step_l2_w'))
comp_steps = comp_s3 + comp_s2

# ---------------- Visualization ----------------
noise = np.std(resid) if np.std(resid) > 0 else 1.0
fig, axes = plt.subplots(3, 1, figsize=(10, 10),
                         gridspec_kw={'height_ratios': [3, 1, 1.2]})
ax0, ax1, ax2 = axes

ax0.plot(x, y, color='0.35', lw=1.0, label='Data')
ax0.plot(x, fit_full, 'r-', lw=1.5, label='Fit')
ax0.plot(x, comp_l3, '--', lw=1.0, label='Component 1')
ax0.plot(x, comp_l3s, '--', lw=1.0, label='Component 2')
ax0.plot(x, comp_l2, '--', lw=1.0, label='Component 3')
ax0.plot(x, comp_l2s, '--', lw=1.0, label='Component 4')
ax0.plot(x, comp_steps, ':', lw=1.2, color='purple', label='Background')
ax0.axvspan(fit_lo, fit_hi, color='yellow', alpha=0.05)
ax0.set_xlim(x.min(), x.max())
ax0.set_ylabel('Y')
ax0.set_title('Data and Fit')
ax0.legend(fontsize=8, ncol=2)

# raw residual panel (shares x with ax0)
ax1.plot(xf, resid, color='0.2', lw=0.9)
ax1.axhline(0, color='r', lw=0.8)
ax1.set_xlim(x.min(), x.max())
ax1.set_ylabel('Residuals')

# normalized residual + zoom on L2 window (own x-axis, not shared)
ax2.plot(xf, resid/noise, color='navy', lw=0.9, label='Norm. residual (res/noise)')
ax2.axhline(0, color='r', lw=0.8)
ax2.set_xlim(x.min(), x.max())
ax2.set_ylabel('Residuals / noise')
ax2.set_xlabel('X')
ax2.legend(fontsize=8, loc='upper left')

# inset zoom over L2 region 721-726 eV to verify two-component L2
try:
    axins = ax2.inset_axes([0.62, 0.55, 0.36, 0.42])
    zmask = (xf >= 721) & (xf <= 726)
    axins.plot(xf[zmask], resid[zmask], color='0.2', lw=0.9)
    axins.axhline(0, color='r', lw=0.8)
    axins.set_title('L2 window 721-726 eV', fontsize=7)
    axins.tick_params(labelsize=6)
except Exception:
    pass

plt.tight_layout()
plt.savefig('visualization.png', dpi=140)
plt.close()

# ---------------- Results JSON ----------------
parameters = {
    'L3_main': {
        'center': gv('cen_l3'), 'center_err': ge('cen_l3'),
        'amplitude': gv('amp_l3'), 'amplitude_err': ge('amp_l3'),
        'fwhm': gv('fwhm_l3'), 'fwhm_err': ge('fwhm_l3'),
        'area': float(area_l3)
    },
    'L3_shoulder': {
        'center': gv('cen_l3s'), 'center_err': ge('cen_l3s'),
        'amplitude': gv('amp_l3s'), 'amplitude_err': ge('amp_l3s'),
        'fwhm': gv('fwhm_l3s'), 'fwhm_err': ge('fwhm_l3s'),
        'area': float(area_l3s)
    },
    'L2_main': {
        'center': gv('cen_l2'), 'center_err': ge('cen_l2'),
        'amplitude': gv('amp_l2'), 'amplitude_err': ge('amp_l2'),
        'fwhm': gv('fwhm_l2'), 'fwhm_err': ge('fwhm_l2'),
        'area': float(area_l2)
    },
    'L2_secondary': {
        'center': gv('cen_l2s'), 'center_err': ge('cen_l2s'),
        'amplitude': gv('amp_l2s'), 'amplitude_err': ge('amp_l2s'),
        'fwhm': gv('fwhm_l2s'), 'fwhm_err': ge('fwhm_l2s'),
        'area': float(area_l2s)
    },
    'step_L3': {
        'edge': gv('step_l3_e'), 'edge_err': ge('step_l3_e'),
        'height': gv('step_l3_h'), 'height_err': ge('step_l3_h'),
        'width': gv('step_l3_w'), 'width_err': ge('step_l3_w')
    },
    'step_L2': {
        'edge': gv('step_l2_e'), 'edge_err': ge('step_l2_e'),
        'height': gv('step_l2_h'), 'height_err': ge('step_l2_h'),
        'width': gv('step_l2_w'), 'width_err': ge('step_l2_w')
    },
    'L3_L2_ratio': {
        'value': float(l3_l2_ratio),
        'L3_total_area': float(l3_total_area),
        'L2_total_area': float(l2_total_area)
    }
}

results = {
    'model_type': 'Fe L2,3 core-loss edge: 4 Gaussian white lines (L3 main, L3 shoulder, L2 main, L2 secondary) + double arctan continuum step (L3 & L2), no polynomial baseline',
    'parameters': parameters,
    'fit_quality': {'r_squared': float(r_squared), 'rmse': float(rmse)},
    'deviation_note': ''
}

print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
