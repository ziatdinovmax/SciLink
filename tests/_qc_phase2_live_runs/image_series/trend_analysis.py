import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - REQUIRED
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
with open('series_analysis_results.json', 'r') as f:
    data = json.load(f)

results = data.get('results', [])
series_metadata = data.get('series_metadata', {})

var_name = series_metadata.get('variable', 'control variable')
var_unit = series_metadata.get('unit', '')
var_values = series_metadata.get('values', [])

flagged = set(data.get('flagged_images', []) or [])

# ----------------------------------------------------------------------
# Align primary variable to results by index
# ----------------------------------------------------------------------
def primary_value(res):
    idx = res.get('index')
    if idx is not None and idx < len(var_values):
        return var_values[idx]
    return None

x_vals = []
names = []
feat = {}
qual = {}
flag_mask = []

# Features / metrics to track
feature_keys = [
    'reflection_spacing_nm',
    'domain_fraction',
    'spot_snr_domain',
    'spot_snr_bulk',
    'reflection_sigma_resolvability',
    'n_reflections_detected',
]
bool_keys = ['reflection_present']
qual_keys = [
    'streak_std_before',
    'streak_std_after',
    'pixel_size_nm',
    'null_threshold',
]
qual_bool_keys = ['reflection_confirmed_domain_gt_bulk', 'null_gate_passed']

for k in feature_keys + bool_keys:
    feat[k] = []
for k in qual_keys + qual_bool_keys:
    qual[k] = []

for res in results:
    xv = primary_value(res)
    x_vals.append(xv)
    names.append(res.get('name', 'idx%s' % res.get('index')))
    flag_mask.append(res.get('index') in flagged)

    ef = res.get('extracted_features', {}) or {}
    qm = res.get('quality_metrics', {}) or {}

    for k in feature_keys:
        v = ef.get(k, None)
        feat[k].append(v if isinstance(v, (int, float)) else None)
    for k in bool_keys:
        v = ef.get(k, None)
        feat[k].append(v)
    for k in qual_keys:
        v = qm.get(k, None)
        qual[k].append(v if isinstance(v, (int, float)) else None)
    for k in qual_bool_keys:
        v = qm.get(k, None)
        qual[k].append(v)

x_arr = np.array([np.nan if v is None else float(v) for v in x_vals], dtype=float)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def clean_xy(xarr, yseq):
    xs, ys = [], []
    for xi, yi in zip(xarr, yseq):
        if yi is None:
            continue
        if isinstance(yi, bool):
            yi = 1.0 if yi else 0.0
        try:
            yf = float(yi)
        except (TypeError, ValueError):
            continue
        if np.isnan(xi) or np.isnan(yf):
            continue
        xs.append(xi)
        ys.append(yf)
    return np.array(xs), np.array(ys)

def regress(xs, ys):
    """Return slope, intercept if >=2 unique x, else None."""
    if len(xs) >= 2 and len(np.unique(xs)) >= 2:
        m, b = np.polyfit(xs, ys, 1)
        return m, b
    return None

def plot_feature(ax, xarr, yseq, label, flag_mask, ylabel=None, is_bool=False):
    xs, ys = clean_xy(xarr, yseq)
    if len(xs) == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=11)
        ax.set_title(label, fontsize=10)
        return
    order = np.argsort(xs)
    xs_s, ys_s = xs[order], ys[order]
    ax.plot(xs_s, ys_s, '-o', color='#1f77b4', ms=7, lw=1.5, zorder=3)

    # Regression
    reg = regress(xs, ys)
    if reg is not None:
        m, b = reg
        xr = np.linspace(np.nanmin(xarr), np.nanmax(xarr), 50)
        ax.plot(xr, m * xr + b, '--', color='#ff7f0e', lw=1.2, alpha=0.8,
                label='slope=%.3g /%s' % (m, var_unit))
        ax.legend(fontsize=7, loc='best')

    # Flag markers (red X)
    for xi, yi, fm in zip(xarr, [ (1.0 if isinstance(v,bool) and v else (0.0 if isinstance(v,bool) else v)) for v in yseq], flag_mask):
        if fm and yi is not None and not (isinstance(xi,float) and np.isnan(xi)):
            try:
                ax.plot(xi, float(yi), 'x', color='red', ms=12, mew=3, zorder=5)
            except (TypeError, ValueError):
                pass

    ax.set_xlabel('%s (%s)' % (var_name, var_unit), fontsize=9)
    ax.set_ylabel(ylabel or label, fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.3)
    if is_bool:
        ax.set_ylim(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['False', 'True'])

# ----------------------------------------------------------------------
# Build dashboard (2x3)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('KCl on Mica: Ordered-Adsorbate / Lattice-Reflection Trend vs KCl Concentration',
             fontsize=14, fontweight='bold')

# Panel 1: Reflection presence + confirmation (booleans overlaid)
ax = axes[0, 0]
xs_p, ys_p = clean_xy(x_arr, feat['reflection_present'])
xs_c, ys_c = clean_xy(x_arr, qual['reflection_confirmed_domain_gt_bulk'])
plotted = False
if len(xs_p):
    o = np.argsort(xs_p)
    ax.plot(xs_p[o], ys_p[o], '-o', color='#1f77b4', ms=8, label='reflection_present')
    plotted = True
if len(xs_c):
    o = np.argsort(xs_c)
    ax.plot(xs_c[o], ys_c[o], '-s', color='#2ca02c', ms=8, label='confirmed (domain>bulk)')
    plotted = True
if plotted:
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['False', 'True'])
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, color='gray')
ax.set_xlabel('%s (%s)' % (var_name, var_unit), fontsize=9)
ax.set_title('Reflection Detection / Confirmation', fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: reflection spacing
plot_feature(axes[0, 1], x_arr, feat['reflection_spacing_nm'],
             'Reflection Spacing', flag_mask, ylabel='spacing (nm)')

# Panel 3: domain fraction
plot_feature(axes[0, 2], x_arr, feat['domain_fraction'],
             'Ordered-Domain Fraction', flag_mask, ylabel='domain fraction')

# Panel 4: SNR domain vs bulk
ax = axes[1, 0]
xs_d, ys_d = clean_xy(x_arr, feat['spot_snr_domain'])
xs_b, ys_b = clean_xy(x_arr, feat['spot_snr_bulk'])
any_snr = False
if len(xs_d):
    o = np.argsort(xs_d)
    ax.plot(xs_d[o], ys_d[o], '-o', color='#1f77b4', ms=7, label='spot_snr_domain')
    any_snr = True
if len(xs_b):
    o = np.argsort(xs_b)
    ax.plot(xs_b[o], ys_b[o], '-s', color='#d62728', ms=7, label='spot_snr_bulk')
    any_snr = True
if any_snr:
    ax.legend(fontsize=7)
    ax.set_xlabel('%s (%s)' % (var_name, var_unit), fontsize=9)
    ax.set_ylabel('spot SNR', fontsize=9)
else:
    ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, color='gray')
ax.set_title('Spot SNR: Domain vs Bulk', fontsize=10)
ax.grid(alpha=0.3)

# Panel 5: resolvability sigma + n_reflections (twin axis)
ax = axes[1, 1]
xs_s, ys_s = clean_xy(x_arr, feat['reflection_sigma_resolvability'])
if len(xs_s):
    o = np.argsort(xs_s)
    ax.plot(xs_s[o], ys_s[o], '-o', color='#9467bd', ms=7, label='sigma_resolvability')
    ax.set_ylabel('sigma resolvability', color='#9467bd', fontsize=9)
    ax.tick_params(axis='y', labelcolor='#9467bd')
else:
    ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, color='gray')
xs_n, ys_n = clean_xy(x_arr, feat['n_reflections_detected'])
if len(xs_n):
    ax2 = ax.twinx()
    o = np.argsort(xs_n)
    ax2.plot(xs_n[o], ys_n[o], '-^', color='#8c564b', ms=7, label='n_reflections')
    ax2.set_ylabel('n_reflections_detected', color='#8c564b', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#8c564b')
ax.set_xlabel('%s (%s)' % (var_name, var_unit), fontsize=9)
ax.set_title('Resolvability & Reflection Count', fontsize=10)
ax.grid(alpha=0.3)

# Panel 6: leveling / streak-removal quality (log scale)
ax = axes[1, 2]
xs_sb, ys_sb = clean_xy(x_arr, qual['streak_std_before'])
xs_sa, ys_sa = clean_xy(x_arr, qual['streak_std_after'])
any_q = False
if len(xs_sb):
    o = np.argsort(xs_sb)
    ax.semilogy(xs_sb[o], np.clip(ys_sb[o], 1e-20, None), '-o', color='#1f77b4', ms=7, label='streak_std_before')
    any_q = True
if len(xs_sa):
    o = np.argsort(xs_sa)
    ax.semilogy(xs_sa[o], np.clip(ys_sa[o], 1e-20, None), '-s', color='#2ca02c', ms=7, label='streak_std_after')
    any_q = True
if any_q:
    ax.legend(fontsize=7)
    ax.set_ylabel('streak std (log)', fontsize=9)
else:
    ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, color='gray')
ax.set_xlabel('%s (%s)' % (var_name, var_unit), fontsize=9)
ax.set_title('Leveling / Streak-Removal Quality', fontsize=10)
ax.grid(alpha=0.3, which='both')

# ----------------------------------------------------------------------
# Regression summary text + missing-data note
# ----------------------------------------------------------------------
summary_lines = []
for k in feature_keys:
    xs, ys = clean_xy(x_arr, feat[k])
    reg = regress(xs, ys)
    if reg is not None:
        summary_lines.append('%s: slope=%.4g /%s' % (k, reg[0], var_unit))
    elif len(xs) == 1:
        summary_lines.append('%s: single point (%.3g) - no regression' % (k, ys[0]))
    else:
        summary_lines.append('%s: no data' % k)

n_missing = sum(1 for r in results if not (r.get('extracted_features')))
if n_missing:
    summary_lines.append('NOTE: %d/%d frames missing extracted features' % (n_missing, len(results)))

fig.text(0.005, 0.005, '  |  '.join(summary_lines), fontsize=7, color='#333333', wrap=True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('feature_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')  # REQUIRED - prevent memory leaks and display

print('Saved feature_trends.png')
print('Regression / trend summary:')
for line in summary_lines:
    print('  ' + line)
