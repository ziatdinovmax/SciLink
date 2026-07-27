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

results = data.get('results', data if isinstance(data, list) else [])
series_metadata = data.get('series_metadata', {}) if isinstance(data, dict) else {}

var_name = series_metadata.get('variable', 'KCl concentration')
var_unit = series_metadata.get('unit', 'mM')
var_values = series_metadata.get('values', [])


def primary_value(res):
    idx = res.get('index', None)
    if idx is not None and isinstance(var_values, list) and idx < len(var_values):
        return var_values[idx]
    return None


# ----------------------------------------------------------------------
# Extract aligned feature arrays
# ----------------------------------------------------------------------
def getf(res, key, default=np.nan):
    feats = res.get('extracted_features', {})
    v = feats.get(key, default)
    if v is None:
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def getq(res, key, default=None):
    q = res.get('quality_metrics', {})
    return q.get(key, default)


rows = []
for res in results:
    x = primary_value(res)
    if x is None:
        continue
    rows.append({
        'x': float(x),
        'name': res.get('name', 'img'),
        'coverage': getf(res, 'coverage'),
        'corr_len': getf(res, 'correlation_length_nm'),
        'mid_wl': getf(res, 'radial_psd_mid_freq_peak_wavelength_nm'),
        'low_env': getf(res, 'radial_psd_low_freq_envelope_length_nm'),
        'patch': getf(res, 'mean_patch_size_nm'),
        'Rq': getf(res, 'Rq_nm'),
        'scale_uncertain': bool(res.get('extracted_features', {}).get('scale_uncertain_flag', False)),
        'refl_null': bool(res.get('extracted_features', {}).get('reflection_null_flag', False)),
        'corr_noise': bool(res.get('extracted_features', {}).get('correlation_length_noise_flag', False)),
        'scale_valid_cc': bool(getq(res, 'scale_valid_for_cross_conc', False)),
        'Rq_valid_cc': bool(getq(res, 'Rq_valid_for_cross_conc', False)),
    })

rows.sort(key=lambda r: r['x'])
if len(rows) == 0:
    # Nothing to plot; create placeholder figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, 'No aligned data points available', ha='center', va='center')
    ax.axis('off')
    plt.savefig('feature_trends.png', dpi=150, bbox_inches='tight')
    plt.close('all')
    raise SystemExit

xs = np.array([r['x'] for r in rows])


def arr(key):
    return np.array([r[key] for r in rows], dtype=float)


def regression(x, y):
    """Return slope, intercept, xfit, yfit for finite points (>=2)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return None
    xf = x[m]
    yf = y[m]
    if np.ptp(xf) == 0:
        return None
    coef = np.polyfit(xf, yf, 1)
    xr = np.linspace(xf.min(), xf.max(), 50)
    return coef[0], coef[1], xr, np.polyval(coef, xr)


# ----------------------------------------------------------------------
# Panel plotting helper
# ----------------------------------------------------------------------
regression_report = {}


def plot_feature(ax, key, title, ylabel, cc_valid_key=None):
    y = arr(key)
    ax.plot(xs, y, 'o-', color='#1f77b4', ms=8, lw=1.5, label='observed')
    # regression
    reg = regression(xs, y)
    if reg is not None:
        slope, intercept, xr, yr = reg
        ax.plot(xr, yr, '--', color='#ff7f0e', lw=1.5,
                label='linear fit')
        # delta over full range
        finite = np.isfinite(y)
        delta = None
        if finite.sum() >= 2:
            delta = y[finite][-1] - y[finite][0]
        regression_report[key] = {'slope': slope, 'intercept': intercept,
                                  'delta': delta}
        ax.text(0.03, 0.95,
                'slope={:.3g}/{}'.format(slope, var_unit),
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                bbox=dict(boxstyle='round', fc='w', ec='0.7', alpha=0.8))
    # mark invalid / flagged points with red X
    for i, r in enumerate(rows):
        invalid = (not np.isfinite(y[i]))
        if cc_valid_key is not None and not r[cc_valid_key]:
            invalid = True
        if invalid:
            yy = y[i] if np.isfinite(y[i]) else ax.get_ylim()[0]
            ax.plot(xs[i], yy, 'x', color='red', ms=14, mew=3, zorder=5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('{} ({})'.format(var_name, var_unit))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    # ensure both concentration ticks visible
    ax.set_xticks(sorted(set(xs.tolist())))


# ----------------------------------------------------------------------
# Build dashboard (2x3)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Mica adsorbate / surface-structure trends vs {} ({})'.format(var_name, var_unit),
             fontsize=14, fontweight='bold')

plot_feature(axes[0, 0], 'coverage', 'Adsorbate coverage', 'coverage (fraction)')
plot_feature(axes[0, 1], 'corr_len', 'Correlation length', 'nm')
plot_feature(axes[0, 2], 'mid_wl', 'Mid-freq corrugation wavelength', 'nm',
             cc_valid_key='scale_valid_cc')
plot_feature(axes[1, 0], 'patch', 'Mean patch size', 'nm')
plot_feature(axes[1, 1], 'Rq', 'Rq roughness (scale-caveated)', 'nm / z-units',
             cc_valid_key='Rq_valid_cc')

# Quality / flag evolution panel
qax = axes[1, 2]
flag_keys = [('scale_uncertain', 'scale uncertain'),
             ('refl_null', 'reflection null'),
             ('corr_noise', 'corr-len noise'),
             ('scale_valid_cc', 'scale valid (cross-conc)'),
             ('Rq_valid_cc', 'Rq valid (cross-conc)')]
for j, (k, lbl) in enumerate(flag_keys):
    yvals = np.array([1 if r[k] else 0 for r in rows]) + j * 0.05  # tiny offset
    qax.plot(xs, yvals, 'o-', ms=7, label=lbl)
qax.set_ylim(-0.2, 1.4)
qax.set_yticks([0, 1])
qax.set_yticklabels(['False', 'True'])
qax.set_xticks(sorted(set(xs.tolist())))
qax.set_title('Quality / validity flags', fontsize=10)
qax.set_xlabel('{} ({})'.format(var_name, var_unit))
qax.grid(alpha=0.3)
qax.legend(fontsize=7, loc='center left', bbox_to_anchor=(0.0, 0.5))

# annotate caveat if only one concentration present
n_conc = len(set(xs.tolist()))
if n_conc < 2:
    fig.text(0.5, 0.005,
             'NOTE: only {} concentration level present in results; '
             'cross-concentration trend/regression is provisional.'.format(n_conc),
             ha='center', fontsize=9, color='darkred')

fig.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('feature_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')  # REQUIRED

# Emit regression summary to stdout for the objective
print('Regression summary (feature: slope per {}, intercept, delta over range):'.format(var_unit))
for k, v in regression_report.items():
    print('  {}: slope={:.4g}, intercept={:.4g}, delta={}'.format(
        k, v['slope'], v['intercept'],
        'n/a' if v['delta'] is None else '{:.4g}'.format(v['delta'])))
