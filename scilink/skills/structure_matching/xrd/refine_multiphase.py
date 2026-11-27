"""``refine_rietveld_multiphase`` / ``refine_rietveld_series`` tools —
quantitative phase fractions from multi-phase Rietveld (GSAS-II ``gsas``
backend).

The quantitative follow-up to mixture identification: ``identify_mixture`` /
``track_phase_series`` deliver WHICH phases (and screening intensity shares);
these tools deliver WEIGHT fractions — the standard in-situ deliverable — by
refining all phases jointly against the whole measured profile. Heavy lifting
lives in ``_gsas_engine`` (``rietveld_refine_multiphase`` /
``rietveld_refine_series``); this module is the thin registry-visible
TOOL_SPEC surface."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from ..._shared._spec import ToolSpec


_SPEC_MULTI = ToolSpec(
    name="refine_rietveld_multiphase",
    description=(
        "Multi-phase Rietveld: refine ALL identified phases jointly against one "
        "measured pattern (GSAS-II) → per-phase WEIGHT FRACTIONS (the "
        "quantitative upgrade of identify_mixture's screening intensity_share), "
        "per-phase lattice (+esd) and microstrain, and whole-profile fit "
        "quality. Staged protocol (background+scale → zero → sample "
        "displacement (Bragg-Brentano) → phase fractions → mustrain (negative "
        "values reset) → instrument profile (guarded) → cells → preferred "
        "orientation (Bragg-Brentano, guarded)). Validated round-trip: fractions to ~1e-3, cells to "
        "<1e-4 Å on synthetic two-phase data. Requires the optional 'gsas' "
        "extra."
    ),
    import_line=("from scilink.skills.structure_matching.xrd.refine_multiphase "
                 "import refine_rietveld_multiphase"),
    signature=(
        "refine_rietveld_multiphase(structure_paths, two_theta, intensity, "
        "wavelength='CuKa', two_theta_range=None, refine_cell=True, "
        "refine_profile=True, n_background_terms=6, "
        "initial_fractions=None, refine_instrument=True, instrument_terms=None, "
        "kalpha2_ratio=None, displacement='cif', geometry=None, "
        "instrument_preset=None, sample_displacement=None, "
        "preferred_orientation='auto') -> dict"
    ),
    parameters={
        "structure_paths": {"type": "list[str]", "description": "CIFs of ALL phases in the mixture (≥2) — identify_mixture's structure_paths, or CIFs from search/simulate. Missing a real phase biases every fraction; include even minor confirmed phases."},
        "two_theta": {"type": "list[float]", "description": "Measured 2θ grid (FULL profile, not a peak list — Rietveld fits the whole pattern)."},
        "intensity": {"type": "list[float]", "description": "Measured intensities aligned with two_theta."},
        "wavelength": {"type": "str | float", "description": "Pass resolve_wavelength(system_info) unchanged. An anode K-alpha name ('CuKa', 'MoKa', …) is refined as the tube's K-alpha1 + K-alpha2 doublet (ratio 0.5) — the usual lab source; a K-alpha1 name ('CuKa1') or a number in Å (monochromated K-alpha1, synchrotron) as a single line. The result's 'kalpha_doublet' records which was modelled."},
        "two_theta_range": {"type": "list[float]", "description": "Optional [min, max] crop before refining — crop low-angle air-scatter upturns for in-situ lab data."},
        "refine_cell": {"type": "bool", "description": "Refine each phase's unit cell (default True). Turn OFF to hold trusted reference cells fixed (sparse data, heavily overlapped patterns)."},
        "refine_profile": {"type": "bool", "description": "Refine per-phase isotropic microstrain (default True) — matches peak widths. Turn OFF only when the instrument profile already matches."},
        "n_background_terms": {"type": "int", "description": "Chebyshev background terms (default 6). RAISE (10-14) for strongly curved/humped backgrounds; LOWER for flat ones."},
        "refine_instrument": {"type": "bool", "description": "Refine the instrument profile (Caglioti V, W and Lorentzian X — the terms microstrain cannot mimic) after microstrain (default True) — the built-in profile is a generic lab template, and a profile that does not match the diffractometer moves intensity between overlapping phases. Guarded: kept only if the fit does not get worse and the widths stay physical (convergence_trace records it as instrument_profile, with rejected=True when undone). Turn OFF for an instrument whose profile is known to match the template."},
        "instrument_terms": {"type": "list[str]", "description": "Which instrument-profile terms refine_instrument refines. Default ['V','W','X']: Caglioti V and W (Gaussian width) and Lorentzian X — the terms isotropic microstrain cannot mimic. 'U' (Gaussian tan^2) and 'Y' (Lorentzian tan) duplicate microstrain broadening and can drive widths negative (the guard then undoes the stage); add them only with refine_profile=False."},
        "kalpha2_ratio": {"type": "float | None", "description": "I(K-alpha2)/I(K-alpha1) to model. Default None = decided by wavelength: 0.5 (the natural, fixed ratio of an X-ray tube's K-alpha doublet) for an anode K-alpha name such as 'CuKa'; no K-alpha2 for 'CuKa1' or a number. Set 0 to force a single line (K-alpha2 stripped or monochromated although the source is labelled 'CuKa'); set a value to force the doublet at that ratio (e.g. the ratio stated in the data file header). A positive value needs a lab-anode wavelength."},
        "displacement": {"type": "str | float", "description": "Atomic displacement (thermal) parameters used in the structure factors. Default \x27cif\x27: each element\x27s isotropic U from the structure file (multiplicity-weighted mean over its sites; anisotropic values converted to U_eq) wherever the file gives a plausible value (non-H: 0.0005-0.025 Å²), GSAS-II\x27s 0.01 Å² otherwise — values outside that range are usually transcription or refinement artefacts and are listed in \x27displacement_U_replaced\x27. \x27cif_raw\x27: the file\x27s values unbounded. \x27default\x27: 0.01 Å² for every atom. A number: that U (Å²) for every atom. Displacement parameters damp high-angle intensity, so values that do not fit the phase move weight between phases with different fall-off. The result\x27s \x27displacement_U\x27 lists the values used."},
        "geometry": {"type": "str | None", "description": "Diffractometer geometry: 'bragg-brentano' (flat-plate reflection) or 'debye-scherrer' (capillary / transmission). Pass resolve_geometry(system_info). Default None infers Bragg-Brentano for a lab-anode wavelength and Debye-Scherrer otherwise. It decides the sample-displacement term and whether preferred_orientation='auto' applies; the result's 'geometry' records it."},
        "instrument_preset": {"type": "str | None", "description": "Starting instrument profile and polarization: 'lab' (Caglioti U,V,W = 2,-2,5, polarization 0.7) or 'synchrotron' (GSAS-II's high-resolution beamline default, U,V,W = 1.163,-0.126,0.063, polarization 0.99). Default None picks by wavelength (a lab-anode line -> lab, anything else -> synchrotron). A start far broader than the data makes microstrain refine NEGATIVE (unphysical; the result lists such phases in negative_microstrain — then start narrower). Override for a high-resolution lab instrument or a synchrotron at a lab-like wavelength."},
        "sample_displacement": {"type": "bool | None", "description": "Refine the flat-plate sample displacement (Bragg-Brentano 'Shift', um) after the zero offset: a sample surface off the goniometer axis shifts peaks by ~cos(theta), which a constant zero cannot follow. Default off (None/False): without an internal standard it correlates with the unit cells and can absorb a real cell difference. Turn on (True) for flat-plate reflection data whose residual shows peak-position misfit growing toward low angle, or when an internal standard fixes the cells; kept only if the fit improves ('sample_displacement' in the result)."},
        "preferred_orientation": {"type": "str | dict | None", "description": "March-Dollase preferred-orientation correction. Default 'auto' (applied for Bragg-Brentano only): for each phase above 5 wt% it tries the a, b, c axes and the normals of the phase's two strongest reflections, and keeps the best only if Rwp drops by >= 3 % (relative) with 0.4 <= MD <= 2.5; MD < 1 = plates lying flat, > 1 = needles. A dict {phase index or name: [h, k, l]} refines MD along that axis unconditionally; None turns it off. Platy (gypsum, micas, clays, hydroxides) and needle phases in a flat-plate holder routinely need it; an uncorrected texture leaves one family of reflections badly misfit and inflates Rwp. The result's 'preferred_orientation' (and each phase's) records axis, MD and the axes tried. Report MD with the fractions."},
        "initial_fractions": {"type": "list[float]", "description": "Starting phase fractions aligned with structure_paths (default: all equal). SET THESE when abundances are very lopsided — a ~95% dominant phase with trace minors can diverge from an all-equal start (fractions pinned at 0/100, profile_corr well below a good fit's ~0.97). Seed from identify_mixture's intensity_share, or e.g. [1.0, 0.05, 0.05] for dominant-plus-traces, and RETRY with a corrected start on that signature."},
    },
    required=["structure_paths", "two_theta", "intensity"],
    returns=(
        "dict. EXACT SHAPES: 'weight_fractions' is a FLAT DICT "
        "{phase_name(str): fraction(float)} — read fractions from here; "
        "'weight_fraction_esds' is the matching FLAT DICT {phase_name: esd "
        "(fraction units) or None}, with 'weight_fraction_esd_note' saying why "
        "an esd is None (e.g. a phase collapsed to zero). These esds are "
        "STATISTICAL (counting noise, scaled by GOF): they measure precision, "
        "not accuracy, and are typically several times smaller than the real "
        "error, which is dominated by systematic effects. Report them as the "
        "refinement esd alongside the fractions; do not reject a converged "
        "refinement because an esd is small, and do not present them as the "
        "total uncertainty. "
        "'phases' is a LIST of per-phase dicts (keys: phase_name, "
        "weight_fraction, weight_fraction_esd, lattice, input_lattice, lattice_esd, "
        "microstrain, cell_runaway, displacement_U, displacement_U_replaced, "
        "preferred_orientation) — do NOT call .items() on it. Also: 'geometry', "
        "'instrument_preset', 'sample_displacement', 'preferred_orientation' "
        "{phase: {axis, MD, tried}}, 'converged', "
        "'profile_corr' (READ THIS as fit quality for arbitrary-unit data; "
        "'Rwp' is inflated by nominal weights), 'gof', 'convergence_trace', "
        "'profile' (obs/calc/background/residual). QUANTITATIVE CAVEATS: "
        "weight fractions are meaningful only when every crystalline phase is "
        "included (a missed phase inflates the others), amorphous content is "
        "INVISIBLE to Rietveld, and strong microabsorption contrast biases "
        "fractions. If 'converged' is False, do not trust fractions or cells — "
        "check each phase's lattice vs input_lattice for runaways."
    ),
    when_to_use=(
        "AFTER identify_mixture (or any multi-phase identification) confirmed "
        "the phase set and materialized structure_paths: quantitative phase "
        "fractions for a single mixture pattern. For a whole in-situ series "
        "use refine_rietveld_series. For a single phase use refine_rietveld."
    ),
)


_SPEC_SERIES = ToolSpec(
    name="refine_rietveld_series",
    description=(
        "SEQUENTIAL multi-phase Rietveld across an in-situ / operando series: "
        "per-frame WEIGHT FRACTIONS vs time/temperature — the standard "
        "in-situ deliverable — plus per-frame refined cells (thermal "
        "expansion) and per-frame fit quality. Each frame runs the full "
        "staged multi-phase protocol, warm-started from its predecessor's "
        "refined fractions and lattices, and is independently "
        "convergence-checked (a diverged frame is reported and NOT carried "
        "forward). Requires the optional 'gsas' extra."
    ),
    import_line=("from scilink.skills.structure_matching.xrd.refine_multiphase "
                 "import refine_rietveld_series"),
    signature=(
        "refine_rietveld_series(structure_paths, frames, wavelength='CuKa', "
        "two_theta_range=None, refine_cell=True, refine_profile=True, "
        "n_background_terms=6, warm_start=True, refine_instrument=True, "
        "instrument_terms=None, kalpha2_ratio=None, displacement='cif', "
        "geometry=None, instrument_preset=None, sample_displacement=None, "
        "preferred_orientation=None) -> dict"
    ),
    parameters={
        "structure_paths": {"type": "list[str]", "description": "CIFs of the phase set present anywhere in the series (from the endpoints-first identification). Phases absent in a given frame refine to ~0 fraction there — include the union, not per-frame subsets."},
        "frames": {"type": "list[dict]", "description": "Series in order: [{'two_theta': [...], 'intensity': [...], 'label': <T/time>}] — FULL profiles per frame, not peak lists. Crop low-angle air scatter via two_theta_range."},
        "wavelength": {"type": "str | float", "description": "Pass resolve_wavelength(system_info) unchanged — K-alpha names are refined as the K-alpha1/K-alpha2 doublet, 'CuKa1' or a number as one line (see refine_rietveld_multiphase)."},
        "two_theta_range": {"type": "list[float]", "description": "Optional [min, max] crop applied to every frame."},
        "refine_cell": {"type": "bool", "description": "Refine each phase's cell per frame (default True) — this is what traces thermal expansion. OFF holds all frames at the reference cells."},
        "refine_profile": {"type": "bool", "description": "Refine per-phase isotropic microstrain per frame (default True)."},
        "n_background_terms": {"type": "int", "description": "Chebyshev background terms per frame (default 6)."},
        "refine_instrument": {"type": "bool", "description": "Refine the instrument profile per frame, guarded (default True; see refine_rietveld_multiphase)."},
        "instrument_terms": {"type": "list[str]", "description": "As refine_rietveld_multiphase (default ['V','W','X'])."},
        "kalpha2_ratio": {"type": "float | None", "description": "I(K-alpha2)/I(K-alpha1) to model. Default None = decided by wavelength: 0.5 (the natural, fixed ratio of an X-ray tube's K-alpha doublet) for an anode K-alpha name such as 'CuKa'; no K-alpha2 for 'CuKa1' or a number. Set 0 to force a single line (K-alpha2 stripped or monochromated although the source is labelled 'CuKa'); set a value to force the doublet at that ratio (e.g. the ratio stated in the data file header). A positive value needs a lab-anode wavelength."},
        "displacement": {"type": "str | float", "description": "As refine_rietveld_multiphase (default 'cif': U from the structure files)."},
        "geometry": {"type": "str | None", "description": "As refine_rietveld_multiphase (pass resolve_geometry(system_info))."},
        "instrument_preset": {"type": "str | None", "description": "As refine_rietveld_multiphase."},
        "sample_displacement": {"type": "bool | None", "description": "As refine_rietveld_multiphase."},
        "preferred_orientation": {"type": "str | dict | None", "description": "As refine_rietveld_multiphase but default None (off): 'auto' re-runs the axis trials on every frame, which multiplies the run time. To correct texture across a series, find the axis on one representative frame with refine_rietveld_multiphase(preferred_orientation='auto') and pass it here as {phase index: [h, k, l]}."},
        "warm_start": {"type": "bool", "description": "Start each frame from the LAST CONVERGED frame's refined fractions and lattices (default True) — the standard sequential-refinement trick: cells evolve smoothly with T, so the warm start keeps the local optimizer in the right basin. Set False for non-contiguous or shuffled frames where carrying state across would mislead."},
    },
    required=["structure_paths", "frames"],
    returns=(
        "dict: 'frames' — per frame {label, weight_fractions {phase: w}, "
        "weight_fraction_esds {phase: statistical esd or None}, "
        "converged, profile_corr, Rwp, cells {phase: refined lattice}}. "
        "Smooth cell drift with T = thermal expansion; a frame with "
        "converged=False (or an 'error' entry) is poorly described by the "
        "phase set — do not trust its fractions, and cross-check "
        "track_phase_series residual alerts for a transient intermediate. "
        "Same quantitative caveats as refine_rietveld_multiphase (complete "
        "phase set, no amorphous, microabsorption)."
    ),
    when_to_use=(
        "The quantification step for an in-situ series AFTER "
        "track_phase_series established which phases exist and roughly where: "
        "phase-fraction evolution curves w(T)/w(t) and lattice-parameter "
        "evolution. Not a screening tool — it needs the confirmed phase set "
        "and full profiles; for fast screening use track_phase_series."
    ),
)

TOOL_SPECS = [_SPEC_MULTI, _SPEC_SERIES]


def refine_rietveld_multiphase(
    structure_paths: Sequence[str],
    two_theta: Sequence[float],
    intensity: Sequence[float],
    wavelength: Union[str, float] = "CuKa",
    two_theta_range: Optional[Sequence[float]] = None,
    refine_cell: bool = True,
    refine_profile: bool = True,
    n_background_terms: int = 6,
    initial_fractions: Optional[Sequence[float]] = None,
    refine_instrument: bool = True,
    instrument_terms: Optional[Sequence[str]] = None,
    kalpha2_ratio: Optional[float] = None,
    displacement: Union[str, float] = "cif",
    geometry: Optional[str] = None,
    instrument_preset: Optional[str] = None,
    sample_displacement: Optional[bool] = None,
    preferred_orientation: Any = "auto",
) -> dict[str, Any]:
    """Multi-phase Rietveld → weight fractions. See ``_SPEC_MULTI``."""
    from ._gsas_engine import rietveld_refine_multiphase
    return rietveld_refine_multiphase(
        structure_paths, two_theta, intensity, wavelength=wavelength,
        two_theta_range=tuple(two_theta_range) if two_theta_range else None,
        refine_cell=refine_cell, refine_profile=refine_profile,
        n_background_terms=n_background_terms,
        initial_fractions=list(initial_fractions) if initial_fractions else None,
        refine_instrument=refine_instrument,
        instrument_terms=list(instrument_terms) if instrument_terms else None,
        kalpha2_ratio=kalpha2_ratio, displacement=displacement, geometry=geometry,
        instrument_preset=instrument_preset, sample_displacement=sample_displacement,
        preferred_orientation=preferred_orientation)


def refine_rietveld_series(
    structure_paths: Sequence[str],
    frames: Sequence[dict],
    wavelength: Union[str, float] = "CuKa",
    two_theta_range: Optional[Sequence[float]] = None,
    refine_cell: bool = True,
    refine_profile: bool = True,
    n_background_terms: int = 6,
    warm_start: bool = True,
    refine_instrument: bool = True,
    instrument_terms: Optional[Sequence[str]] = None,
    kalpha2_ratio: Optional[float] = None,
    displacement: Union[str, float] = "cif",
    geometry: Optional[str] = None,
    instrument_preset: Optional[str] = None,
    sample_displacement: Optional[bool] = None,
    preferred_orientation: Any = None,
) -> dict[str, Any]:
    """Sequential multi-phase Rietveld over a series. See ``_SPEC_SERIES``."""
    from ._gsas_engine import rietveld_refine_series
    return rietveld_refine_series(
        structure_paths, frames, wavelength=wavelength,
        two_theta_range=tuple(two_theta_range) if two_theta_range else None,
        refine_cell=refine_cell, refine_profile=refine_profile,
        n_background_terms=n_background_terms, warm_start=warm_start,
        refine_instrument=refine_instrument,
        instrument_terms=list(instrument_terms) if instrument_terms else None,
        kalpha2_ratio=kalpha2_ratio, displacement=displacement, geometry=geometry,
        instrument_preset=instrument_preset, sample_displacement=sample_displacement,
        preferred_orientation=preferred_orientation)
