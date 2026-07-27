import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

# ------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------
data = np.load('data.npy')
data = np.asarray(data, dtype=float)
if data.ndim == 2:
    # assume shape (2, N) or (N, 2)
    if data.shape[0] == 2:
        x = data[0]
        y = data[1]
    elif data.shape[1] == 2:
        x = data[:, 0]
        y = data[:, 1]
    else:
        raise ValueError('Unexpected 2D data shape')
else:
    # 1-D intensity only: build x from provided range
    y = data
    x = np.linspace(683.9, 730.15, y.size)

N = y.size

# No smoothing / clipping: EELS core-loss intensity, data are clean and
# negatives near zero are just noise but we do NOT clip since we fit a
# continuum step; keep raw.
y_raw = y.copy()

# ------------------------------------------------------------------
# 2. Fit domain per plan: model the edge region. Pre-edge below ~706 eV
#    is flat at zero; noisy 719/728 windows are within tail. Fit full
#    measured range (continuum + 4 Lorentzians) but focus on edges.
#    We fit the entire measured x-range as the model domain.
# ------------------------------------------------------------------
fit_mask = np.ones(N, dtype=bool)  # full domain
xf = x[fit_mask]
yf = y[fit_mask]

# ------------------------------------------------------------------
# 3. Model: 4 Lorentzians + 2 arctangent steps (continuum) + baseline
# ------------------------------------------------------------------
def lorentzian(x, amp, center, fwhm):
    # amplitude = peak height
    g = fwhm / 2.0
    return amp * g**2 / ((x - center)**2 + g**2)

def arctan_step(x, height, onset, width):
    # smooth hydrogenic/Hartree-Slater step; width ~ broadening
    return height * (0.5 + (1.0/np.pi) * np.arctan((x - onset) / width))

def full_model(x, amp_L3, cen_L3, fwhm_L3,
               amp_L3s, cen_L3s, fwhm_L3s,
               amp_L2, cen_L2, fwhm_L2,
               amp_L2s, cen_L2s, fwhm_L2s,
               step1_h, step1_on, step1_w,
               step2_h, step2_on, step2_w,
               baseline):
    m = baseline
    m = m + lorentzian(x, amp_L3, cen_L3, fwhm_L3)
    m = m + lorentzian(x, amp_L3s, cen_L3s, fwhm_L3s)
    m = m + lorentzian(x, amp_L2, cen_L2, fwhm_L2)
    m = m + lorentzian(x, amp_L2s, cen_L2s, fwhm_L2s)
    m = m + arctan_step(x, step1_h, step1_on, step1_w)
    m = m + arctan_step(x, step2_h, step2_on, step2_w)
    return m

model = Model(full_model)
params = Parameters()

# L3 main white line
params.add('amp_L3', value=1000.0, min=0)
params.add('cen_L3', value=710.5, min=709.5, max=711.5)
params.add('fwhm_L3', value=1.5, min=0.5, max=4.0)
# L3 pre-peak / crystal-field shoulder
params.add('amp_L3s', value=380.0, min=0)
params.add('cen_L3s', value=709.0, min=708.0, max=710.0)
params.add('fwhm_L3s', value=1.0, min=0.5, max=4.0)
# L2 main white line
params.add('amp_L2', value=140.0, min=0)
params.add('cen_L2', value=723.7, min=722.7, max=724.7)
params.add('fwhm_L2', value=2.5, min=0.5, max=4.0)
# L2 low-energy shoulder
params.add('amp_L2s', value=120.0, min=0)
params.add('cen_L2s', value=722.0, min=721.0, max=723.0)
params.add('fwhm_L2s', value=1.5, min=0.5, max=4.0)
# Continuum step 1 (L3 onset ~709 eV)
params.add('step1_h', value=80.0, min=0, max=400)
params.add('step1_on', value=709.0, min=707.0, max=711.0)
params.add('step1_w', value=1.5, min=0.3, max=3.0)
# Continuum step 2 (L2 onset ~722 eV)
params.add('step2_h', value=40.0, min=0, max=300)
params.add('step2_on', value=722.0, min=720.0, max=724.0)
params.add('step2_w', value=1.5, min=0.3, max=3.0)
# Constant baseline (pre-edge ~ 0)
params.add('baseline', value=0.0, min=-20, max=20)

result = model.fit(yf, params, x=xf, method='least_squares',
                   max_nfev=20000)

# ------------------------------------------------------------------
# 4. Evaluate fit over ALL N x-points
# ------------------------------------------------------------------
fit_full = full_model(x, **{k: result.params[k].value for k in result.params})
residual = y - fit_full

# R^2 and RMSE over the full modelled domain
ss_res = np.sum((yf - result.best_fit)**2)
ss_tot = np.sum((yf - np.mean(yf))**2)
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
rmse = np.sqrt(np.mean((yf - result.best_fit)**2))

# noise estimate from pre-edge flat region for normalized residual
pre_mask = x < 705.0
if np.count_nonzero(pre_mask) > 10:
    noise = np.std(y[pre_mask])
else:
    noise = np.std(residual)
if not np.isfinite(noise) or noise <= 0:
    noise = rmse if rmse > 0 else 1.0

# ------------------------------------------------------------------
# 5. Save fit array
# ------------------------------------------------------------------
np.save('fit.npy', fit_full)

# ------------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------------
p = result.params
c_L3  = lorentzian(x, p['amp_L3'].value,  p['cen_L3'].value,  p['fwhm_L3'].value)
c_L3s = lorentzian(x, p['amp_L3s'].value, p['cen_L3s'].value, p['fwhm_L3s'].value)
c_L2  = lorentzian(x, p['amp_L2'].value,  p['cen_L2'].value,  p['fwhm_L2'].value)
c_L2s = lorentzian(x, p['amp_L2s'].value, p['cen_L2s'].value, p['fwhm_L2s'].value)
c_step = (arctan_step(x, p['step1_h'].value, p['step1_on'].value, p['step1_w'].value)
          + arctan_step(x, p['step2_h'].value, p['step2_on'].value, p['step2_w'].value)
          + p['baseline'].value)

fig = plt.figure(figsize=(10, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax2 = fig.add_subplot(gs[2], sharex=ax0)

ax0.plot(x, y_raw, color='0.6', lw=0.8, alpha=0.5, label='Data (raw)')
ax0.plot(x, y, 'k.', ms=2.5, label='Data')
ax0.plot(x, fit_full, 'r-', lw=1.5, label='Fit')
ax0.plot(x, c_L3 + c_step, '--', color='tab:blue', lw=1.0, label='Component 1')
ax0.plot(x, c_L3s + c_step, '--', color='tab:green', lw=1.0, label='Component 2')
ax0.plot(x, c_L2 + c_step, '--', color='tab:orange', lw=1.0, label='Component 3')
ax0.plot(x, c_L2s + c_step, '--', color='tab:purple', lw=1.0, label='Component 4')
ax0.plot(x, c_step, ':', color='0.4', lw=1.0, label='Background')
ax0.set_ylabel('Y')
ax0.legend(fontsize=8, ncol=2)
ax0.set_title('Data and Fit')
ax0.set_xlim(x.min(), x.max())

ax1.plot(x, residual, 'k-', lw=0.8)
ax1.axhline(0, color='r', lw=0.8)
ax1.set_ylabel('Residual')

ax2.plot(x, residual / noise, 'b-', lw=0.8)
ax2.axhline(0, color='r', lw=0.8)
ax2.axhline(3, color='0.5', ls=':', lw=0.8)
ax2.axhline(-3, color='0.5', ls=':', lw=0.8)
ax2.set_ylabel('Residual / noise')
ax2.set_xlabel('X')

fig.savefig('visualization.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# 7. Results JSON
# ------------------------------------------------------------------
def gv(name):
    return float(result.params[name].value)
def ge(name):
    e = result.params[name].stderr
    return float(e) if e is not None else None

# Integrated Lorentzian areas (analytic: pi*amp*fwhm/2) for L3/L2 ratio
def larea(amp, fwhm):
    return np.pi * amp * fwhm / 2.0
area_L3 = larea(gv('amp_L3'), gv('fwhm_L3')) + larea(gv('amp_L3s'), gv('fwhm_L3s'))
area_L2 = larea(gv('amp_L2'), gv('fwhm_L2')) + larea(gv('amp_L2s'), gv('fwhm_L2s'))
L3_L2_ratio = float(area_L3 / area_L2) if area_L2 > 0 else None
spin_orbit = gv('cen_L2') - gv('cen_L3')

parameters = {
    "L3_white_line": {"center": gv('cen_L3'), "center_err": ge('cen_L3'),
                       "amplitude": gv('amp_L3'), "amplitude_err": ge('amp_L3'),
                       "fwhm": gv('fwhm_L3'), "fwhm_err": ge('fwhm_L3')},
    "L3_shoulder": {"center": gv('cen_L3s'), "center_err": ge('cen_L3s'),
                     "amplitude": gv('amp_L3s'), "amplitude_err": ge('amp_L3s'),
                     "fwhm": gv('fwhm_L3s'), "fwhm_err": ge('fwhm_L3s')},
    "L2_white_line": {"center": gv('cen_L2'), "center_err": ge('cen_L2'),
                       "amplitude": gv('amp_L2'), "amplitude_err": ge('amp_L2'),
                       "fwhm": gv('fwhm_L2'), "fwhm_err": ge('fwhm_L2')},
    "L2_shoulder": {"center": gv('cen_L2s'), "center_err": ge('cen_L2s'),
                     "amplitude": gv('amp_L2s'), "amplitude_err": ge('amp_L2s'),
                     "fwhm": gv('fwhm_L2s'), "fwhm_err": ge('fwhm_L2s')},
    "continuum_step_L3": {"step": gv('step1_h'), "step_err": ge('step1_h'),
                           "onset": gv('step1_on'), "onset_err": ge('step1_on'),
                           "width": gv('step1_w'), "width_err": ge('step1_w')},
    "continuum_step_L2": {"step": gv('step2_h'), "step_err": ge('step2_h'),
                           "onset": gv('step2_on'), "onset_err": ge('step2_on'),
                           "width": gv('step2_w'), "width_err": ge('step2_w')},
    "baseline": {"value": gv('baseline'), "value_err": ge('baseline')},
    "derived": {"L3_L2_intensity_ratio": L3_L2_ratio,
                 "spin_orbit_splitting": float(spin_orbit)}
}

results = {
    "model_type": "4 Lorentzian white lines (Fe L3, L3 shoulder, L2, L2 shoulder) on a double-arctangent continuum step plus constant baseline",
    "parameters": parameters,
    "fit_quality": {"r_squared": float(r_squared), "rmse": float(rmse)},
    "deviation_note": ""
}

print(f"FIT_RESULTS_JSON:{json.dumps(results)}")
