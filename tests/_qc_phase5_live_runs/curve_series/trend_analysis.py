import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
with open('series_fit_results.json', 'r') as f:
    data = json.load(f)

# Support both {'results':...} and bare list
if isinstance(data, dict):
    results = data.get('results', data.get('series', []))
    series_metadata = data.get('series_metadata', {})
else:
    results = data
    series_metadata = {}

# Fallback metadata (matches provided SERIES_METADATA)
if not series_metadata:
    series_metadata = {
        'variable': 'temperature',
        'unit': 'K',
        'values': [243.2, 258.2, 278.2, 293.5, 313.2]
    }

var_name = series_metadata.get('variable', 'temperature')
var_unit = series_metadata.get('unit', 'K')
temp_values = series_metadata.get('values', [])

# ---------------------------------------------------------------
# Helper: extract the dominant 31P peak from a heterogeneous fit dict
# ---------------------------------------------------------------
def extract_metrics(res):
    params = res.get('parameters', {})
    peaks = []
    for k, v in params.items():
        if k.startswith('peak_') and isinstance(v, dict):
            center = v.get('center', v.get('delta_iso_averaged', np.nan))
            fwhm = v.get('fwhm_ppm', v.get('FWHM_ppm', np.nan))
            inten = v.get('integrated_intensity', v.get('amplitude', np.nan))
            popf = v.get('population_fraction', np.nan)
            cerr = v.get('center_err', v.get('delta_iso_averaged_err', np.nan))
            peaks.append({'center': center, 'fwhm': fwhm,
                          'inten': inten, 'popf': popf, 'cerr': cerr})
    if not peaks:
        return None

    # Dominant peak = highest population fraction if available, else intensity
    def score(p):
        if not np.isnan(p['popf']):
            return p['popf']
        return p['inten'] if not np.isnan(p['inten']) else -np.inf
    dominant = max(peaks, key=score)

    # component count
    ccount = params.get('component_count', None)
    if ccount is None:
        ccount = len(peaks)

    total_inten = np.nansum([p['inten'] for p in peaks])

    fq = res.get('fit_quality', {})
    r2 = fq.get('r_squared', fq.get('peak_region_r2', np.nan))

    return {
        'center': dominant['center'],
        'cerr': dominant['cerr'],
        'fwhm': dominant['fwhm'],
        'popf': dominant['popf'],
        'ccount': ccount,
        'total_inten': total_inten,
        'r2': r2,
        'n_peaks': len(peaks),
    }

# ---------------------------------------------------------------
# Build aligned arrays (sorted by temperature)
# ---------------------------------------------------------------
rows = []
for res in results:
    idx = res.get('index', None)
    if idx is not None and idx < len(temp_values):
        T = temp_values[idx]
    else:
        T = np.nan
    m = extract_metrics(res)
    if m is None:
        continue
    m['T'] = T
    m['name'] = res.get('name', '')
    # flag detection (generic)
    m['flagged'] = bool(res.get('flagged', False))
    rows.append(m)

rows = [r for r in rows if not np.isnan(r['T'])]
rows.sort(key=lambda r: r['T'])

T   = np.array([r['T'] for r in rows], float)
cen = np.array([r['center'] for r in rows], float)
cerr= np.array([r['cerr'] if not np.isnan(r['cerr']) else 0.0 for r in rows], float)
fw  = np.array([r['fwhm'] for r in rows], float)
inten = np.array([r['total_inten'] for r in rows], float)
popf  = np.array([r['popf'] for r in rows], float)
cc    = np.array([r['ccount'] for r in rows], float)
r2    = np.array([r['r2'] for r in rows], float)
flag  = np.array([r['flagged'] for r in rows], bool)
multi = cc > 1  # slow-exchange / resolved-species spectra

invT = 1000.0 / T  # for van't Hoff style regression

# ---------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------
def linfit(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return None
    xg, yg = x[good], y[good]
    A = np.vstack([xg, np.ones_like(xg)]).T
    coef, *_ = np.linalg.lstsq(A, yg, rcond=None)
    slope, intercept = coef
    yhat = slope * xg + intercept
    ss_res = np.sum((yg - yhat) ** 2)
    ss_tot = np.sum((yg - yg.mean()) ** 2)
    r2v = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'slope': slope, 'intercept': intercept, 'r2': r2v,
            'x': xg, 'y': yg}

# Shift vs T (linear) and shift vs 1000/T
fit_cenT   = linfit(T, cen)
fit_ceninv = linfit(invT, cen)
fit_fwhm   = linfit(T, fw)

# ---------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------
plt.rcParams.update({'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3})
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('31P NMR trends vs temperature — G2/TEP Mg(OTf)$_2$ electrolyte\n'
             '(dominant P species; competitive-solvation speciation)',
             fontsize=13, fontweight='bold')

def mark_flags(ax, x, y):
    if flag.any():
        ax.scatter(x[flag], y[flag], s=180, facecolors='none',
                   edgecolors='red', marker='X', linewidths=2.5,
                   label='flagged', zorder=6)
    if multi.any():
        ax.scatter(x[multi], y[multi], s=140, facecolors='none',
                   edgecolors='darkorange', marker='s', linewidths=2.0,
                   label='multi-species (slow exch.)', zorder=5)

# (0,0) Dominant peak center vs T with regression
ax = axes[0, 0]
ax.errorbar(T, cen, yerr=cerr, fmt='o-', color='C0', capsize=3,
            label='dominant $\\delta$', zorder=4)
if fit_cenT:
    xx = np.linspace(T.min(), T.max(), 100)
    ax.plot(xx, fit_cenT['slope']*xx + fit_cenT['intercept'], '--',
            color='gray',
            label=f"fit: {fit_cenT['slope']*1000:.2f} ppb/K\nR$^2$={fit_cenT['r2']:.3f}")
mark_flags(ax, T, cen)
ax.set_xlabel(f'{var_name} ({var_unit})')
ax.set_ylabel('dominant peak center (ppm)')
ax.set_title('Chemical shift vs T (coalescence)')
ax.legend(fontsize=8)
ax.invert_yaxis()

# (0,1) FWHM vs T
ax = axes[0, 1]
ax.plot(T, fw, 's-', color='C3', label='dominant FWHM')
if fit_fwhm:
    xx = np.linspace(T.min(), T.max(), 100)
    ax.plot(xx, fit_fwhm['slope']*xx + fit_fwhm['intercept'], '--',
            color='gray',
            label=f"slope={fit_fwhm['slope']*1000:.2f} ppb/K\nR$^2$={fit_fwhm['r2']:.3f}")
mark_flags(ax, T, fw)
ax.set_xlabel(f'{var_name} ({var_unit})')
ax.set_ylabel('dominant peak FWHM (ppm)')
ax.set_title('Linewidth vs T (exchange broadening)')
ax.legend(fontsize=8)

# (0,2) Number of resolved species vs T
ax = axes[0, 2]
ax.step(T, cc, where='mid', color='C2', marker='o', label='resolved P species')
mark_flags(ax, T, cc)
ax.set_xlabel(f'{var_name} ({var_unit})')
ax.set_ylabel('component count')
ax.set_title('Speciation: # resolved P environments')
ax.set_yticks(np.arange(0, np.nanmax(cc)+2))
ax.axhspan(0.5, 1.5, color='C0', alpha=0.07)
ax.axhspan(1.5, np.nanmax(cc)+1, color='C1', alpha=0.07)
ax.text(T.min(), 1.02, ' fast exchange', fontsize=8, color='C0')
if np.nanmax(cc) > 1:
    ax.text(T.min(), 2.02, ' slow exchange / resolved', fontsize=8, color='C1')
ax.legend(fontsize=8)

# (1,0) Shift vs 1000/T (van't Hoff style) with regression
ax = axes[1, 0]
ax.errorbar(invT, cen, yerr=cerr, fmt='D', color='C4', capsize=3,
            label='data')
if fit_ceninv:
    xx = np.linspace(invT.min(), invT.max(), 100)
    ax.plot(xx, fit_ceninv['slope']*xx + fit_ceninv['intercept'], '--',
            color='k',
            label=(f"$\\delta$ = {fit_ceninv['slope']:.4f}(1000/T) "
                   f"+ {fit_ceninv['intercept']:.3f}\nR$^2$={fit_ceninv['r2']:.3f}"))
mark_flags(ax, invT, cen)
ax.set_xlabel('1000/T (K$^{-1}$)')
ax.set_ylabel('dominant peak center (ppm)')
ax.set_title('Shift vs 1000/T (regression model)')
ax.legend(fontsize=8)

# (1,1) Total integrated intensity & dominant population fraction
ax = axes[1, 1]
l1 = ax.plot(T, inten/np.nanmax(inten), 'o-', color='C5',
             label='total integ. intensity (norm.)')
ax.set_xlabel(f'{var_name} ({var_unit})')
ax.set_ylabel('normalized total intensity')
ax2 = ax.twinx()
l2 = ax2.plot(T, popf, '^--', color='C6', label='dominant pop. fraction')
ax2.set_ylabel('dominant population fraction')
ax2.set_ylim(0, 1.05)
mark_flags(ax, T, inten/np.nanmax(inten))
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center left')
ax.set_title('Intensity & dominant speciation vs T')

# (1,2) Fit quality R^2 vs T
ax = axes[1, 2]
ax.plot(T, r2, 'o-', color='C7', label='R$^2$')
mark_flags(ax, T, r2)
ax.set_xlabel(f'{var_name} ({var_unit})')
ax.set_ylabel('fit R$^2$')
ax.set_title('Fit quality vs T')
ax.set_ylim(min(0.99, np.nanmin(r2)-0.001), 1.0005)
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')

# ---------------------------------------------------------------
# Console summary (regression outputs required by objective)
# ---------------------------------------------------------------
print('=== 31P NMR temperature-trend summary ===')
for r in rows:
    print(f"T={r['T']:.1f}K  delta={r['center']:.3f}ppm  "
          f"FWHM={r['fwhm']:.3f}ppm  n_species={r['ccount']}  R2={r['r2']:.4f}")
if fit_cenT:
    print(f"\nShift vs T:      slope={fit_cenT['slope']*1000:.3f} ppb/K, "
          f"intercept={fit_cenT['intercept']:.3f} ppm, R2={fit_cenT['r2']:.4f}")
if fit_ceninv:
    print(f"Shift vs 1000/T: slope={fit_ceninv['slope']:.4f} ppm/(1000/T), "
          f"intercept={fit_ceninv['intercept']:.3f} ppm, R2={fit_ceninv['r2']:.4f}")
if fit_fwhm:
    print(f"FWHM vs T:       slope={fit_fwhm['slope']*1000:.3f} ppb/K, R2={fit_fwhm['r2']:.4f}")
print('\nInterpretation: transition from multi-component (slow-exchange, '
      'resolved TEP solvation environments) at low T toward a single '
      'motionally-averaged peak at high T indicates fast competitive '
      'solvation exchange between G2 and TEP as temperature increases.')
