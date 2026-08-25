---
description: EELS spectrum-image analysis — core-loss edges for elemental and chemical-state mapping, low-loss plasmons for dielectric response, on STEM datacubes.
---
# EELS Spectrum Imaging Skill

## overview

Electron Energy Loss Spectroscopy (EELS) spectrum imaging analysis. EELS-SI
datasets are 3D datacubes (x, y, energy-loss) acquired in a scanning
transmission electron microscope (STEM). Core-loss edges provide elemental
and chemical-state information; low-loss plasmons probe dielectric response.
NMF decomposition separates overlapping spectral signatures into physically
meaningful endmember spectra and spatial abundance maps for elemental
mapping, phase identification, and chemical-state analysis.

## planning

**Decomposition method selection:**
- Use NMF when the expected phases/edges are broadly known and spectra are
  non-negative (core-loss EELS after background subtraction). NMF yields
  directly interpretable endmembers that approximate real spectral shapes.
- Use PCA for fully exploratory analysis of unknown systems or when negative
  spectral features are expected (e.g., difference spectra, derivative-like
  fine structure). PCA captures maximum variance but components are abstract
  linear combinations, not physical spectra.

**Component count estimation:**
- Start with the number of distinct core-loss edges present in the energy
  range, plus one for the background/plural-scattering component.
- Add one component per distinct chemical environment / oxidation state
  you expect for each element (e.g. mixed-valence phases, interface
  states). *Example:* a dataset spanning two adjacent edges in a
  mixed-valence transition-metal oxide (such as Ti L2,3 + O K in a
  TiO2/SrTiO3 system) typically calls for ~3–5 components — one per
  oxidation state, one per ligand environment, plus a background. Other
  systems scale by the same logic.
- Datasets with a single edge region: 2–4 components typically suffice.
- If the energy range includes both low-loss and core-loss: add 1–2
  components for plasmon/zero-loss tail contributions.

**Energy range considerations:**
- Core-loss edges sit on a power-law background that must be captured by
  at least one NMF component or pre-subtracted.
- Narrow energy windows around a single edge family (e.g., 440-540 eV for
  Ti L + O K) give cleaner decomposition than wide ranges.
- If the dataset spans >200 eV, consider whether multiple decompositions
  on sub-ranges would be more informative.

**Preprocessing considerations:**
- Zero-loss peak alignment should be performed before NMF if energy drift
  is present (check for spatial variation in edge onset positions).
- Fourier-log or Fourier-ratio deconvolution removes plural scattering
  and sharpens edges, but introduces noise. For noisy data (SNR < 10),
  skip deconvolution and let NMF handle the plural scattering as a
  separate component.
- Negative values from background subtraction or deconvolution violate
  NMF constraints. Either clip to zero or switch to PCA.

## implementation

**Per-pixel fitting recipes for EELS features.**

Core-loss edges: subtract a power-law background (`A * E^-r`, r typically
2.5–5) fit to a pre-edge window, then fit the edge step plus any ELNES
peaks with `lmfit.models.StepModel` + `GaussianModel` / `LorentzianModel`.

Low-loss plasmons: a Drude-Lorentz oscillator in loss space is the most
physically faithful model. A Lorentzian-on-linear-background is an
acceptable approximation when the peak sits well inside the measurement
window.

A peak whose center lies outside the measurement window cannot be
honestly measured from a tail-only fit — mathematically the lineshape
admits a center value, but the uncertainty is large and the result is
extrapolation, not measurement. The correct strategy is to **bound the
center to the measurement window** and **NaN out pixels whose fit
converges at the bound** (those pixels' peak is outside the window;
the result is "no measurement available", not "peak at the bound").
Do not widen bounds beyond the window to chase those pixels.

Initialization and bounds:
- Initialize peak center at the argmax of a lightly smoothed (Savitzky-
  Golay, window 5–9, polyorder 2) spectrum, not a fixed value.
- Bound the peak center to the measurement window (`[axis_start,
  axis_end]`).
- Mark per-pixel fits with R² < ~0.5 or parameters railed at a bound
  as NaN. The dashboard histogram will then show the real interior
  distribution; boundary pile-ups indicate fit failures or features
  outside the measurement window — both honestly reported as NaN.

## interpretation

**Reference edge energies (core-loss onset values).** The entries below
are illustrative examples covering commonly encountered edges; the same
chemical-shift / fine-structure principles apply to any core-loss edge.
For elements / edges not listed, infer the initial edge energy from the
data itself (the spectrum's rising-step or peak position) and keep fit
bounds generous around that initial guess.

Titanium Ti L2,3:
- Ti L3 onset: ~456 eV (metal), ~458 eV (TiO2)
- Ti L2 onset: ~462 eV (metal), ~464 eV (TiO2)
- L3/L2 white-line ratio: ~0.8 for Ti4+ (TiO2), ~1.0 for Ti3+, >1.2 for Ti2+/Ti0
- Crystal field splitting of L3 into t2g/eg peaks (~2 eV): signature of octahedral coordination in rutile/anatase TiO2

Oxygen O K:
- Onset: ~532 eV
- Pre-peak at ~530 eV in transition metal oxides (hybridized O 2p - metal 3d states)
- Shape is highly sensitive to local bonding: SrTiO3 shows distinct a/b/c features at 531, 536, 543 eV
- Intensity of the pre-peak correlates with metal-oxygen covalency

Iron Fe L2,3:
- Fe L3 onset: ~708 eV (Fe2+), ~710 eV (Fe3+)
- White-line ratio L3/L2 distinguishes oxidation states
- Fe0 (metal): broader, lower white lines at ~707 eV

Silicon Si L2,3:
- Si (elemental): onset ~99 eV with sharp edge
- SiO2: onset ~106 eV with delayed maximum
- Si3N4: onset ~102 eV

Carbon C K:
- Graphite/sp2: sharp pi* peak at 285 eV, broad sigma* at 292 eV
- Diamond/sp3: no pi* peak, sigma* onset at 289 eV
- Amorphous C: broadened pi* and sigma* features

Nitrogen N K:
- Onset: ~401 eV
- BN: distinct pi* at 401 eV, sigma* at 409 eV (similar to graphite pattern)
- Metal nitrides: onset shifted and broadened

**ELNES fingerprints:**
- Crystal field splitting visible in L2,3 edges of 3d transition metals
  provides direct information on site symmetry (octahedral vs tetrahedral).
- The O K pre-peak intensity in perovskites (SrTiO3, BaTiO3) is a probe
  of the B-site cation valence and oxygen vacancy concentration.

**Quantification:**
- Relative elemental concentrations from NMF component abundances require
  correction by partial ionization cross-sections (Hartree-Slater model).
- Absolute quantification requires known specimen thickness (from
  low-loss log-ratio method: t/lambda = ln(I_total/I_zero-loss)).

## validation

**Reconstruction error:**
- NMF reconstruction error (Frobenius norm) should decrease monotonically
  with increasing component count. A residual error > 10% of total signal
  variance suggests insufficient components or data quality issues.
- Compare reconstruction RMSE across component counts: the optimal number
  shows diminishing returns (elbow in error curve).

**Component physicality checks:**
- All NMF component spectra must be non-negative. Any negative values
  indicate numerical issues or data that violates NMF assumptions.
- Each component spectrum should have recognizable spectral features
  (edges, peaks) or be identifiable as a smooth background.
- Spatial maps should show coherent regions, not salt-and-pepper noise.
  A map dominated by single-pixel fluctuations likely represents a noise
  component that should be removed by reducing n_components.

**Noise-floor component detection:**
- If a component's spatial map shows no spatial structure (flat or
  random), and its spectrum resembles scaled noise or a featureless
  continuum, it is likely capturing noise. Remove it by reducing the
  component count.
- Check the explained variance of each component: components explaining
  <1% of total variance are candidates for noise.

**Cross-validation with known chemistry:**
- Edge onset energies in NMF components should match literature values
  within the energy resolution of the instrument (typically 0.5-1.5 eV
  for standard EELS, <0.1 eV for monochromated systems).
- Spatial distributions should correlate with known sample morphology
  (e.g., a component associated with a given element should be
  concentrated in regions of the sample that contain that element).
- If HAADF-STEM or EDS data is available, verify spatial consistency.
