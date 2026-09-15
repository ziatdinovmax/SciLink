---
description: STM and low-bias conductive-AFM image analysis — interprets bias-dependent features as local density of states or conductance rather than directly as atomic positions.
---
# STM & Conductive AFM (low-bias) Imaging Skill

## overview

Scanning tunneling microscopy (STM) and conductive atomic force
microscopy at low bias (cAFM) produce images whose features reflect
local density of states or local conductance. Apparent "atoms" can
shift with bias voltage, tip condition, or contact geometry. Use this
skill for LDOS heterogeneity, conductance domain mapping, and
pattern-level analysis of electronic imaging data.

## planning

### foundational
The primary signal is electronic. Set the plan's `quality_criteria`
to match electronic features — LDOS heterogeneity, conductance
variation, domain structure, coherence of abundance maps. Atomic-like
periodic patterns may still be present, but include lattice-related
criteria only when the objective explicitly calls for lattice
characterization.

The right Tier 1 tool depends on what the image shows. For
heterogeneous textures or phase domains (quantum materials,
crystalline patches, spatially varying electronic order),
`run_fft_nmf_analysis` with a feature-scale window is typically
appropriate. For discrete features on a surface (molecular
adsorbates, clusters, individual defects), peak/blob detection
(`skimage.feature`) or instance segmentation (SAM) is more natural —
count, measure, and map the spatial distribution of individual
features rather than looking for periodicity that isn't there.

Classical atom-detection tools (`detect_atoms`, `detect_atoms_dcnn`)
are designed for STEM-style atomic columns and typically not the
right default for electronic imaging — apparent atoms can shift with
bias, tip state, and contact geometry. Use them when the objective
explicitly calls for lattice characterization and the image is
genuinely atomic-resolution and stable.

When the objective *is* point defects in a stable periodic lattice
(adsorbates/protrusions or vacancies on an atomic surface, defects in
a charge-ordered or moire superlattice), `fft_defect_map`
(`scilink.skills._shared.fft_defect`) maps both signs — `excess`
(adatom/adsorbate) and `deficit` (vacancy) — against a noise null
without atom finding. It requires square pixels and a leveled
background first (it does neither itself); pass the square-pixel size
in nm. Heed its dense-disorder / single-cluster warnings — a charge
domain or cluster is not a set of point defects.

**Do not try to pre-filter LDOS modulations out of the lattice
signal.** In STM/cAFM the lattice contrast is genuinely modulated by
the LDOS envelope — that modulation is real signal, not noise. Avoid
aggressive high-pass filtering, mean subtraction within windows, or
bandpass design aimed at "isolating the pure lattice": these remove
physically meaningful variation and may leave only noise where the
lattice contrast was weakest. Let the decomposition see both scales;
interpret the resulting components rather than engineering the
pipeline to force a separation the physics doesn't support.

**Visualizing FFT-NMF results.** When `n_components` ≤ 3, display all
FFT components and all abundance maps. For `n_components` > 3, show
the major ones (top by total abundance). Pair each abundance map with
its corresponding FFT component so the viewer can match them — an
abundance map alone is uninterpretable. Preserve true aspect ratios;
don't stretch or squash images to fill subplot shapes.

## validation

### foundational
Validate against the electronic signal — spatial coherence of
detected domains, physical plausibility of conductance / LDOS values,
agreement between independent electronic descriptors.

For FFT-NMF-based analyses: acceptable output has non-noise frequency
content in the components and spatially coherent abundance maps.
Semantic labeling of components (e.g. "this is the crystalline
component") requires post-hoc interpretation and depends on the data —
don't treat it as a hard quality criterion.

Do not penalize a decomposition for "not isolating the pure lattice"
or for being dominated by the LDOS envelope rather than lattice-scale
peaks: envelope and lattice contrast are physically coupled in these
techniques, and the envelope is real signal — not contamination the
decomposition should have suppressed. Accept coherent, data-faithful
outputs whether or not they produce a pure-lattice component.
