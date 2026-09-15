---
description: X-ray photoelectron spectroscopy peak fitting (Shirley/Tougaard backgrounds, asymmetric line shapes) for chemical-state analysis from binding energies.
technique: [XPS, "X-ray photoelectron spectroscopy", "photoelectron spectroscopy", ESCA]
---
# XPS Curve Fitting Skill

## overview

X-ray Photoelectron Spectroscopy (XPS) curve fitting for chemical state
analysis. Binding energies are characteristic of elemental oxidation states
and chemical environments. This skill covers core-level peak fitting with
Shirley/Tougaard backgrounds and asymmetric line shapes.

## planning

**Background subtraction:** Always compute and SUBTRACT the Shirley background
BEFORE peak fitting. Do NOT fit the background as part of the peak model.
The workflow is: (1) sort data by ascending binding energy, (2) compute
Shirley background, (3) subtract it, (4) fit Voigt peaks to the subtracted
data, (5) report R² on full spectrum (background + peaks vs original).
For metallic samples near the Fermi edge, consider Tougaard background
instead. A simple linear background is only acceptable for well-isolated
peaks on a flat baseline.

**Line shapes:** Use Voigt profiles (Gaussian–Lorentzian convolution) as the
default peak shape. For metallic states, use asymmetric Doniach–Sunjic or
LF (Lorentzian-asymmetric with tail) line shapes. The Gaussian width is
dominated by instrumental broadening and should be constrained to be similar
across peaks in the same spectral region.

**Spin-orbit splitting:** Core levels with orbital angular momentum l > 0
show spin-orbit doublets with fixed area ratios:
- p levels (l=1): 2p3/2 and 2p1/2 with area ratio 2:1, e.g. Ti 2p splitting ~5.7 eV
- d levels (l=2): 3d5/2 and 3d3/2 with area ratio 3:2, e.g. Ag 3d splitting ~6.0 eV
- f levels (l=3): 4f7/2 and 4f5/2 with area ratio 4:3, e.g. Au 4f splitting ~3.7 eV

Constrain the doublet separation and area ratio to theoretical values.

**FWHM constraints:** Typical FWHM for XPS peaks is 0.8–3.0 eV. Peaks
broader than 3.0 eV likely indicate unresolved chemical components. Peaks
narrower than 0.7 eV may suggest overfitting or unrealistic constraints.

**Number of components:** Start with the minimum number of chemically
justifiable peaks. Each component must correspond to a distinct chemical
state. Do not add peaks solely to improve R². If a new component is needed,
justify it with known chemistry of the sample.

**Charge referencing:** If the survey includes C 1s from adventitious carbon,
reference all binding energies to C 1s = 284.8 eV. Note the correction
applied.

## analysis

**CRITICAL: XPS fitting workflow.** Always follow this exact sequence:
1. Sort data so binding energy is in INCREASING order (low to high).
2. Compute and subtract Shirley background.
3. Fit Voigt peaks to the background-subtracted data.
4. Report R² on the FULL spectrum (background + peaks vs original data).

**Complete XPS fitting template** — adapt this pattern for any XPS region:

```python
import numpy as np
from lmfit.models import VoigtModel

# ---- Step 1: Load and sort by increasing binding energy ----
data = np.loadtxt(DATA_PATH, delimiter=',', skiprows=1)
x_raw, y_raw = data[:, 0], data[:, 1]
sort_idx = np.argsort(x_raw)
x = x_raw[sort_idx]
y = y_raw[sort_idx]

# ---- Step 2: Shirley background subtraction ----
def shirley_background(x, y, tol=1e-5, max_iter=50):
    """Iterative Shirley background. x must be sorted ascending."""
    n = len(y)
    bg = np.full(n, y[0])  # start from low-BE endpoint
    for _ in range(max_iter):
        diff = y - bg
        # Integrate from high BE toward low BE
        cumul = np.zeros(n)
        for i in range(n - 2, -1, -1):
            cumul[i] = cumul[i + 1] + (diff[i] + diff[i + 1]) * 0.5 * (x[i + 1] - x[i])
        total = cumul[0] if cumul[0] != 0 else 1.0
        new_bg = y[0] + (y[-1] - y[0]) * cumul / total
        if np.max(np.abs(new_bg - bg)) < tol:
            break
        bg = new_bg
    return bg

bg = shirley_background(x, y)
y_sub = y - bg  # background-subtracted spectrum

# ---- Step 3: Build peak model (example: two doublets) ----
def make_doublet(prefix_main, prefix_sat, center_init, splitting, area_ratio):
    """Create a spin-orbit doublet with constrained splitting and area ratio."""
    main = VoigtModel(prefix=prefix_main)
    sat = VoigtModel(prefix=prefix_sat)
    params = main.make_params() + sat.make_params()
    params[f'{prefix_main}center'].set(value=center_init, min=center_init - 1.5, max=center_init + 1.5)
    params[f'{prefix_main}sigma'].set(value=0.5, min=0.2, max=1.5)
    params[f'{prefix_main}gamma'].set(value=0.3, min=0.05, max=1.0)
    params[f'{prefix_sat}center'].expr = f'{prefix_main}center + {splitting}'
    params[f'{prefix_sat}amplitude'].expr = f'{prefix_main}amplitude * {1.0 / area_ratio}'
    params[f'{prefix_sat}sigma'].expr = f'{prefix_main}sigma'
    params[f'{prefix_sat}gamma'].expr = f'{prefix_main}gamma'
    return main + sat, params

# Dominant species doublet
model1, params1 = make_doublet('A_3_2_', 'A_1_2_', center_init=458.8, splitting=5.7, area_ratio=2)
# Minor species doublet (if needed)
model2, params2 = make_doublet('B_3_2_', 'B_1_2_', center_init=457.0, splitting=5.5, area_ratio=2)

composite_model = model1 + model2
all_params = params1
all_params.update(params2)

# Initial amplitude guesses from data
all_params['A_3_2_amplitude'].set(value=float(np.max(y_sub)) * 0.7, min=0)
all_params['B_3_2_amplitude'].set(value=float(np.max(y_sub)) * 0.2, min=0)

# ---- Step 4: Fit ----
result = composite_model.fit(y_sub, all_params, x=x, method='leastsq')

# ---- Step 5: Compute R² on full spectrum (bg + peaks vs original) ----
y_fit_full = bg + result.best_fit
ss_res = np.sum((y - y_fit_full) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1.0 - ss_res / ss_tot
```

**Doniach-Sunjic line shape for metallic states:**

When symmetric Voigt profiles show systematic residuals (underfitting at
peak top, overfitting on flanks), use lmfit's built-in `DoniachModel`.
The asymmetry parameter `alpha` is typically 0.02–0.2 for metals.

```python
from lmfit.models import DoniachModel

# Doniach-Sunjic for metallic component
peak = DoniachModel(prefix='met_')
params = peak.make_params()
params['met_amplitude'].set(value=5000, min=0)
params['met_center'].set(value=454.0, min=452, max=456)
params['met_sigma'].set(value=0.4, min=0.1, max=1.5)
# alpha controls asymmetry: 0 = symmetric, higher = more asymmetric
params['met_alpha'].set(value=0.1, min=0.01, max=0.3)
```

For doublets with Doniach-Sunjic, constrain the satellite the same way
as with VoigtModel (splitting, area ratio, shared width and alpha).

**NumPy compatibility:** `np.trapz` was removed in NumPy 2.0. Always use
`np.trapezoid` for trapezoidal integration. Never use `np.trapz`.

**Key points for the LLM script generator:**
- Always sort by ascending BE before fitting. XPS CSV files often have
  descending BE.
- The Shirley background uses endpoint intensities: `y[0]` (low-BE side)
  and `y[-1]` (high-BE side). The step goes from low to high intensity.
- Fit peaks to `y_sub` (background-subtracted), but compute R² on the
  full spectrum `y` vs `bg + result.best_fit`.
- Use `method='leastsq'` (Levenberg-Marquardt) for speed and reliability.
- Set `min=0` on all amplitude parameters (peaks cannot be negative).

## interpretation

**Binding energy reference values** (referenced to C 1s = 284.8 eV):

Common chemical states and their expected binding energies:

Carbon C 1s:
- C-C / C-H: 284.8 eV (reference)
- C-O (ether, alcohol): 286.1–286.5 eV
- C=O (carbonyl): 287.5–288.0 eV
- O-C=O (carboxyl, ester): 288.5–289.5 eV
- CF2: 291.0–292.0 eV
- pi-pi* shake-up: ~291.5 eV (broad)

Oxygen O 1s:
- Metal oxide (lattice O²⁻): 529.0–530.5 eV
- Hydroxide / C=O: 531.0–532.0 eV
- C-O / adsorbed water: 532.5–533.5 eV

Nitrogen N 1s:
- Metal nitride: 396.5–397.5 eV
- Amine / amide: 399.0–400.5 eV
- Protonated amine (NH₃⁺): 401.0–402.0 eV
- Nitrate (NO₃⁻): 406.5–407.5 eV

Titanium Ti 2p3/2:
- Ti metal: 453.9–454.1 eV
- TiO (Ti²⁺): 455.0–455.5 eV
- Ti₂O₃ (Ti³⁺): 456.5–457.5 eV
- TiO₂ (Ti⁴⁺): 458.5–459.0 eV

Silicon Si 2p:
- Si metal: 99.0–99.5 eV
- SiO₂: 103.0–103.8 eV
- Si₃N₄: 101.5–102.0 eV

**Common surface contaminants:** Nearly all ex-situ samples show adventitious
carbon (C 1s at 284.8 eV, used as charge reference) and adsorbed oxygen/water
(O 1s at 532–533 eV). Sodium (Na 1s ~1071 eV) and calcium (Ca 2p ~347 eV)
are common handling contaminants. Silicon (Si 2p ~102 eV) may appear from
silicone grease or glass substrates. Do not over-interpret these as part
of the sample chemistry unless the sample is expected to contain them.

**Shake-up and plasmon satellite structures:** Many elements show intrinsic
satellite features that are chemically diagnostic:
- Cu 2p: Strong shake-up satellites ~9 eV above main peaks are diagnostic
  of Cu²⁺ (CuO). Cu⁰ and Cu¹⁺ show NO shake-up satellites — their
  absence confirms reduced copper.
- Ti 2p: Weak shake-up satellites ~13 eV above 2p3/2 in TiO₂. Usually
  broad and low intensity (~5% of main peak).
- Ni 2p: Intense multiplet splitting and shake-up satellites. Ni²⁺ (NiO)
  shows satellites ~6 eV above main peaks; Ni⁰ shows asymmetric peaks
  but weak satellites.
- Ce 3d: Complex multiplet structure with 3 doublet pairs from final-state
  effects. Satellite pattern distinguishes Ce³⁺ from Ce⁴⁺.
- Fe 2p: Shake-up satellites ~8 eV above main peaks for Fe³⁺ (Fe₂O₃),
  weaker for Fe²⁺ (FeO), absent for Fe⁰.

Do not add satellite peaks unless the element and chemical state are known
to produce them. Spurious peaks at unexpected positions more likely indicate
fitting artifacts or Shirley background errors.

**Quantification:** Peak areas after Shirley background subtraction are
proportional to atomic concentration when divided by relative sensitivity
factors (RSF). Report atomic percentages when multiple components are
resolved within a region. Note that XPS quantification is semi-quantitative
(typical accuracy ±10–20%).

**Chemical state assignment:** Assignments must be justified by both
binding energy position AND expected chemistry of the sample. Consider
the sample preparation, expected composition, and possible surface
contamination.

## validation

**Quality checks specific to XPS fitting:**
- FWHM must be between 0.8 and 3.0 eV for each component. Flag if outside range.
- Spin-orbit splitting must match literature values within ±0.3 eV.
- Area ratios for spin-orbit doublets must match theoretical values within ±10%.
- Peak positions should match known chemical states within ±0.5 eV (after charge correction).
- R² > 0.99 is expected for well-resolved XPS peaks; R² < 0.95 indicates a poor model.
- Residuals should show no systematic structure (check for missed components or incorrect background).
- If more than 5 components are needed in a single spectral region, verify physical justification.
- Gaussian width should be consistent (within ±0.3 eV) across components in the same region.
- **Differential charging:** On insulating samples, peaks may appear
  asymmetrically broadened toward higher BE. This is NOT multiple chemical
  states — it is a measurement artifact from non-uniform surface charging.
  Suspect differential charging when: (1) FWHM is unusually large (>2.5 eV)
  for a single expected chemical state, (2) peak shape is asymmetric toward
  high BE only, (3) the broadening disappears with a flood gun. Do not
  deconvolve charging-broadened peaks into multiple chemical components.
