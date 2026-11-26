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
        "quality. Staged protocol (background+scale → zero → phase fractions → "
        "mustrain → instrument profile (guarded) → cells). Validated round-trip: fractions to ~1e-3, cells to "
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
        "kalpha2_ratio=None) -> dict"
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
        "initial_fractions": {"type": "list[float]", "description": "Starting phase fractions aligned with structure_paths (default: all equal). SET THESE when abundances are very lopsided — a ~95% dominant phase with trace minors can diverge from an all-equal start (fractions pinned at 0/100, profile_corr well below a good fit's ~0.97). Seed from identify_mixture's intensity_share, or e.g. [1.0, 0.05, 0.05] for dominant-plus-traces, and RETRY with a corrected start on that signature."},
    },
    required=["structure_paths", "two_theta", "intensity"],
    returns=(
        "dict. EXACT SHAPES: 'weight_fractions' is a FLAT DICT "
        "{phase_name(str): fraction(float)} — read fractions from here; "
        "'phases' is a LIST of per-phase dicts (keys: phase_name, "
        "weight_fraction, lattice, input_lattice, lattice_esd, microstrain, "
        "cell_runaway) — do NOT call .items() on it. Also: 'converged', "
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
        "instrument_terms=None, kalpha2_ratio=None) -> dict"
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
        "warm_start": {"type": "bool", "description": "Start each frame from the LAST CONVERGED frame's refined fractions and lattices (default True) — the standard sequential-refinement trick: cells evolve smoothly with T, so the warm start keeps the local optimizer in the right basin. Set False for non-contiguous or shuffled frames where carrying state across would mislead."},
    },
    required=["structure_paths", "frames"],
    returns=(
        "dict: 'frames' — per frame {label, weight_fractions {phase: w}, "
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
        kalpha2_ratio=kalpha2_ratio)


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
        kalpha2_ratio=kalpha2_ratio)
