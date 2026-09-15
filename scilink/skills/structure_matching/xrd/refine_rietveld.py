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

from typing import Any, Union

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
        "refine_atoms=False, n_background_terms=6) -> dict"
    ),
    parameters={
        "structure_path": {"type": "str", "description": "CIF of the identified candidate phase (e.g. the best match from search_structures/score_xrd_match)."},
        "two_theta": {"type": "list[float]", "description": "Measured 2θ axis (degrees)."},
        "intensity": {"type": "list[float]", "description": "Measured intensity at each 2θ (raw, with background; do NOT background-subtract — Rietveld fits the background)."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or wavelength in Å. Must match how the pattern was collected."},
        "two_theta_range": {"type": "tuple", "description": "Optional (min,max) crop of the measured pattern before refining — exclude a noisy low-angle upturn or an empty high-angle tail. None = full range."},
        "refine_cell": {"type": "bool", "description": "Refine the unit-cell parameters (default True). Turn OFF to hold a trusted reference cell fixed, or when the data is too sparse to constrain it."},
        "refine_profile": {"type": "bool", "description": "Refine isotropic sample broadening — microstrain (default True). This matches the peak WIDTHS and yields the microstrain output. Turn OFF only if the instrument profile already matches the data. Only microstrain is refined (not a separate crystallite size): the two are correlated broadening terms and refining both destabilizes sparse/high-symmetry patterns (a cubic pattern can collapse the cell)."},
        "refine_atoms": {"type": "bool", "description": "Also refine atomic coordinates + isotropic displacement parameters (default False). RISKY: on noisy or low-resolution data it can diverge or overfit. Enable ONLY for a high-quality pattern once the cell+profile fit is already good."},
        "n_background_terms": {"type": "int", "description": "Chebyshev background terms (default 6). RAISE (10–14) for a strongly curved or humped background; LOWER for a flat one."},
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
        "(two_theta / y_obs / y_calc / y_background / residual for plotting). "
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
) -> dict[str, Any]:
    """Rietveld-refine a structure against a measured pattern. See ``TOOL_SPEC``.

    Thin wrapper over the GSAS-II engine; lazy-imported so the module stays
    importable without the optional ``gsas`` dependency."""
    from ._gsas_engine import rietveld_refine
    return rietveld_refine(
        structure_path, two_theta, intensity, wavelength=wavelength,
        two_theta_range=two_theta_range, refine_cell=refine_cell,
        refine_profile=refine_profile, refine_atoms=refine_atoms,
        n_background_terms=n_background_terms,
    )
