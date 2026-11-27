"""``refine_rietveld`` tool — Rietveld refinement of a candidate structure against
a measured XRD pattern (GSAS-II ``gsas`` backend).

This is the tier-3 follow-up to phase identification: once ``search_structures`` +
``simulate_xrd_pattern`` + ``score_xrd_match`` have picked the phase, Rietveld
refinement fits the *whole* measured profile to extract accurate lattice
parameters (+ esd), crystallite size, and microstrain, and to quantify the fit.

The heavy lifting lives in the self-contained GSAS-II wrapper
(``_gsas_engine.rietveld_refine``); this module is the thin, registry-visible
TOOL_SPEC surface. GSAS-II is the optional ``gsas`` extra (see ``_gsas_engine``
for the install recipe)."""

from __future__ import annotations

from typing import Any, Optional, Union

from ..._shared._spec import ToolSpec


TOOL_SPEC = ToolSpec(
    name="refine_rietveld",
    description=(
        "Rietveld-refine a candidate crystal structure against a measured powder "
        "XRD pattern (GSAS-II). The follow-up to phase identification: extracts "
        "accurate lattice parameters with esd, crystallite size and microstrain, "
        "and the fit quality, by least-squares fitting the whole profile. Uses a "
        "robust staged protocol (background+scale -> 2theta zero -> sample "
        "broadening -> unit cell). Requires the optional 'gsas' extra."
    ),
    import_line="from scilink.skills.structure_matching.xrd.refine_rietveld import refine_rietveld",
    signature=(
        "refine_rietveld(structure_path, two_theta, intensity, wavelength='CuKa', "
        "two_theta_range=None, refine_cell=True, refine_profile=True, "
        "refine_atoms=False, n_background_terms=6, kalpha2_ratio=None, "
        "displacement='cif', geometry=None, instrument_preset=None, "
        "sample_displacement=None, preferred_orientation=None) -> dict"
    ),
    parameters={
        "structure_path": {"type": "str", "description": "CIF of the identified candidate phase (e.g. the best match from search_structures/score_xrd_match)."},
        "two_theta": {"type": "list[float]", "description": "Measured 2θ axis (degrees)."},
        "intensity": {"type": "list[float]", "description": "Measured intensity at each 2θ (raw, with background; do NOT background-subtract — Rietveld fits the background)."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or wavelength in Å. Must match how the pattern was collected. Pass resolve_wavelength(system_info) unchanged: an anode K-alpha name ('CuKa') is refined as the K-alpha1/K-alpha2 doublet, 'CuKa1' or a number as a single line."},
        "two_theta_range": {"type": "tuple", "description": "Optional (min,max) crop of the measured pattern before refining — exclude a noisy low-angle upturn or an empty high-angle tail. None = full range."},
        "refine_cell": {"type": "bool", "description": "Refine the unit-cell parameters (default True). Turn OFF to hold a trusted reference cell fixed, or when the data is too sparse to constrain it."},
        "refine_profile": {"type": "bool", "description": "Refine isotropic sample broadening — microstrain (default True). This matches the peak WIDTHS and yields the microstrain output. Turn OFF only if the instrument profile already matches the data. Only microstrain is refined (not a separate crystallite size): the two are correlated broadening terms and refining both destabilizes sparse/high-symmetry patterns (a cubic pattern can collapse the cell)."},
        "refine_atoms": {"type": "bool", "description": "Also refine atomic coordinates + isotropic displacement parameters (default False). RISKY: on noisy or low-resolution data it can diverge or overfit. Enable ONLY for a high-quality pattern once the cell+profile fit is already good."},
        "n_background_terms": {"type": "int", "description": "Chebyshev background terms (default 6). RAISE (10–14) for a strongly curved or humped background; LOWER for a flat one."},
        "displacement": {"type": "str | float", "description": "Atomic displacement (thermal) parameters used in the structure factors. Default \x27cif\x27: each element\x27s isotropic U from the structure file (multiplicity-weighted mean over its sites; anisotropic values converted to U_eq) wherever the file gives a plausible value (non-H: 0.0005-0.025 Å²), GSAS-II\x27s 0.01 Å² otherwise — values outside that range are usually transcription or refinement artefacts and are listed in \x27displacement_U_replaced\x27. \x27cif_raw\x27: the file\x27s values unbounded. \x27default\x27: 0.01 Å² for every atom. A number: that U (Å²) for every atom. Displacement parameters damp high-angle intensity, so values that do not fit the phase move weight between phases with different fall-off. The result\x27s \x27displacement_U\x27 lists the values used."},
        "geometry": {"type": "str | None", "description": "Diffractometer geometry: 'bragg-brentano' (flat-plate reflection) or 'debye-scherrer' (capillary / transmission). Pass resolve_geometry(system_info). Default None infers Bragg-Brentano for a lab-anode wavelength and Debye-Scherrer otherwise. It decides the sample-displacement term and whether preferred_orientation='auto' applies; the result's 'geometry' records it."},
        "instrument_preset": {"type": "str | None", "description": "Starting instrument profile and polarization: 'lab' (Caglioti U,V,W = 2,-2,5, polarization 0.7) or 'synchrotron' (GSAS-II's high-resolution beamline default, U,V,W = 1.163,-0.126,0.063, polarization 0.99). Default None picks by wavelength (a lab-anode line -> lab, anything else -> synchrotron). A start far broader than the data makes microstrain refine NEGATIVE (unphysical; the result lists such phases in negative_microstrain — then start narrower). Override for a high-resolution lab instrument or a synchrotron at a lab-like wavelength."},
        "sample_displacement": {"type": "bool | None", "description": "Refine the flat-plate sample displacement (Bragg-Brentano 'Shift', um) after the zero offset: a sample surface off the goniometer axis shifts peaks by ~cos(theta), which a constant zero cannot follow. Default off (None/False): without an internal standard it correlates with the unit cells and can absorb a real cell difference. Turn on (True) for flat-plate reflection data whose residual shows peak-position misfit growing toward low angle, or when an internal standard fixes the cells; kept only if the fit improves ('sample_displacement' in the result)."},
        "preferred_orientation": {"type": "str | dict | None", "description": "March-Dollase correction as in refine_rietveld_multiphase, but default None here (one phase: texture and the structure model trade off). Pass 'auto' or {0: [h, k, l]} when one family of reflections is systematically too strong/weak in the residual."},
        "kalpha2_ratio": {"type": "float | None", "description": "I(K-alpha2)/I(K-alpha1) to model. Default None = decided by wavelength: 0.5 (the natural, fixed ratio of an X-ray tube's K-alpha doublet) for an anode K-alpha name such as 'CuKa'; no K-alpha2 for 'CuKa1' or a number. Set 0 to force a single line (K-alpha2 stripped or monochromated although the source is labelled 'CuKa'); set a value to force the doublet at that ratio (e.g. the ratio stated in the data file header). A positive value needs a lab-anode wavelength."},
    },
    required=["structure_path", "two_theta", "intensity"],
    returns=(
        "dict with 'lattice' (a,b,c,α,β,γ,volume) and 'lattice_esd', "
        "'input_lattice' (the starting cell) and 'converged' (bool), 'Rwp', "
        "'profile_corr' (Yobs-vs-Ycalc correlation — the robust fit metric when "
        "the data has no counting statistics, e.g. arbitrary intensity units, "
        "where Rwp is inflated), 'gof', 'microstrain' (refined isotropic "
        "broadening; 'crystallite_size_um' is None — not separately refined), "
        "'convergence_trace' (per-stage Rwp/corr), and 'profile' "
        "(two_theta / y_obs / y_calc / y_background / residual for plotting), "
        "'geometry', 'instrument_preset', 'sample_displacement', "
        "'preferred_orientation' and 'displacement_U' / 'displacement_U_replaced' "
        "(what the refinement used). "
        "IMPORTANT: Rietveld is a LOCAL refinement — it needs a starting cell "
        "within ~1% of the truth. If 'converged' is False the cell ran away "
        "(starting model too far off or wrong phase); do NOT trust 'lattice' — "
        "compare it to 'input_lattice'. 'converged' is necessary but NOT "
        "sufficient: a wrong-but-plausible low-symmetry (triclinic) structure can "
        "reach a moderate profile_corr (~0.85) with a wrong cell and pass — so "
        "corroborate a low-symmetry / modest-corr result against the identification "
        "score and the expected cell before trusting it."
    ),
    when_to_use=(
        "After a phase is identified (its cell is already approximately right), to "
        "refine its lattice parameters and extract crystallite size / microstrain "
        "from the measured pattern, or to quantify how well the structure fits the "
        "whole profile. Read 'profile_corr' (not the absolute Rwp) as the fit "
        "quality when the data is in arbitrary units, and check 'converged'. Do "
        "NOT run it on a far-off or unidentified structure — it will diverge."
    ),
)


def refine_rietveld(
    structure_path: str,
    two_theta,
    intensity,
    wavelength: Union[str, float] = "CuKa",
    two_theta_range: tuple = None,
    refine_cell: bool = True,
    refine_profile: bool = True,
    refine_atoms: bool = False,
    n_background_terms: int = 6,
    kalpha2_ratio: float = None,
    displacement: Union[str, float] = "cif",
    geometry: Optional[str] = None,
    instrument_preset: Optional[str] = None,
    sample_displacement: Optional[bool] = None,
    preferred_orientation=None,
) -> dict[str, Any]:
    """Rietveld-refine a structure against a measured pattern. See ``TOOL_SPEC``.

    Thin wrapper over the GSAS-II engine; lazy-imported so the module stays
    importable without the optional ``gsas`` dependency."""
    from ._gsas_engine import rietveld_refine
    return rietveld_refine(
        structure_path, two_theta, intensity, wavelength=wavelength,
        two_theta_range=two_theta_range, refine_cell=refine_cell,
        refine_profile=refine_profile, refine_atoms=refine_atoms,
        n_background_terms=n_background_terms, kalpha2_ratio=kalpha2_ratio,
        displacement=displacement, geometry=geometry, instrument_preset=instrument_preset,
        sample_displacement=sample_displacement, preferred_orientation=preferred_orientation,
    )
