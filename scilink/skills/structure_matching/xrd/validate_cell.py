"""``validate_cell_lebail`` tool — structure-free whole-pattern validation of a
candidate unit cell (GSAS-II Le Bail fit; ``gsas`` extra).

The arbiter of a candidate CELL in the blind-identification chain. Scoring a
simulated pattern conflates the cell with a specific structure — a correct cell
paired with a wrong same-cell structure scores badly and gets discarded. The
Le Bail fit frees every reflection intensity and asks only "does this CELL
account for the whole measured profile?", which is the standard practice step
between indexing and structure work.

The heavy lifting lives in the self-contained GSAS-II wrapper
(``_gsas_engine.lebail_fit``); this module is the thin, registry-visible
TOOL_SPEC surface."""

from __future__ import annotations

from typing import Any, Union

from ..._shared._spec import ToolSpec


TOOL_SPEC = ToolSpec(
    name="validate_cell_lebail",
    description=(
        "Validate a candidate unit cell against the measured powder pattern with "
        "a structure-free Le Bail whole-pattern fit (no atoms needed — reflection "
        "intensities are free). THE arbiter of an indexed cell: run it on "
        "index_pattern's candidates BEFORE any database search or structure "
        "scoring. A wrong cell or SUBCELL leaves peaks unaccounted (low "
        "profile_corr); the true cell fits (corr ≥ ~0.9) and its lattice refines "
        "to precision with no structure. A SUPERCELL always fits at least as well "
        "as the true cell — among cells that fit, prefer the SMALLEST. Requires "
        "the optional 'gsas' extra."
    ),
    import_line="from scilink.skills.structure_matching.xrd.validate_cell import validate_cell_lebail",
    signature=(
        "validate_cell_lebail(two_theta, intensity, cell, bravais='Cubic-P', "
        "wavelength='CuKa', two_theta_range=None, refine_cell=True, "
        "refine_zero=True, n_background_terms=6, extra_cycles=2, "
        "observed_peaks=None, peak_tol_deg=0.15) -> dict"
    ),
    parameters={
        "two_theta": {"type": "list[float]", "description": "Measured 2θ axis (degrees)."},
        "intensity": {"type": "list[float]", "description": "Measured intensity (raw, with background — the fit models the background)."},
        "cell": {"type": "dict | list", "description": "Candidate cell: an index_pattern candidate_cells entry (dict with a,b,c,alpha,beta,gamma) or a 6-sequence."},
        "bravais": {"type": "str", "description": "GSAS Bravais name from index_pattern ('Tetragonal-P', 'Cubic-F', …) — mapped to a generic space group carrying the lattice extinctions — or an explicit H-M space-group symbol when known ('P 42/m n m')."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or wavelength in Å; must match the measurement."},
        "two_theta_range": {"type": "tuple", "description": "Optional (min,max) crop — exclude a noisy low-angle upturn or empty tail."},
        "refine_cell": {"type": "bool", "description": "Refine the cell during the fit (default True): a correct approximate cell polishes to the precise one, structure-free. Turn OFF to test a cell exactly as given."},
        "refine_zero": {"type": "bool", "description": "Refine the 2θ zero offset (default True)."},
        "n_background_terms": {"type": "int", "description": "Chebyshev background terms (default 6). RAISE (10–14) for strongly curved backgrounds."},
        "extra_cycles": {"type": "int", "description": "Extra intensity-extraction cycles after the staged fit (default 2) — Le Bail intensities converge iteratively; RAISE if profile_corr is still climbing in the convergence_trace."},
        "observed_peaks": {"type": "list[float]", "description": "Optional measured peak positions (extract_peaks' 'positions'). When given, each is checked against the reflections the fitted cell generates: the returned 'unaccounted_peaks' are direct evidence of an impurity phase (a few, on a good fit) or a wrong/subcell (many)."},
        "peak_tol_deg": {"type": "float", "description": "Match window for the peak accounting (default 0.15°). RAISE for broad peaks or residual zero error."},
    },
    required=["two_theta", "intensity", "cell"],
    returns=(
        "dict with 'profile_corr' (the verdict metric), 'cell_fits' (corr ≥ 0.8), "
        "'lattice' (refined cell — precise when the cell fits), 'input_lattice', "
        "'space_group_used', 'convergence_trace' (per-stage corr), 'profile' "
        "(two_theta / y_obs / y_calc for plotting), and — when observed_peaks "
        "was given — 'accounted_peaks' / 'unaccounted_peaks' (orphans: a few on "
        "a good fit = impurity lines to identify separately; many = wrong cell "
        "or subcell). Verdicts: corr ≥ ~0.9 the cell accounts for the pattern; "
        "≤ ~0.6 wrong cell or subcell. A supercell fits as well as the true "
        "cell — prefer the smallest cell that fits."
    ),
    when_to_use=(
        "Immediately after index_pattern, on each plausible candidate cell — "
        "BEFORE database searches or structure scoring — to kill wrong cells and "
        "subcells cheaply and to rank alias families (smallest fitting cell "
        "wins). Also to double-check a final identification: the winning phase's "
        "cell must Le-Bail-fit the pattern."
    ),
)


def validate_cell_lebail(
    two_theta,
    intensity,
    cell,
    bravais: str = "Cubic-P",
    wavelength: Union[str, float] = "CuKa",
    two_theta_range: tuple = None,
    refine_cell: bool = True,
    refine_zero: bool = True,
    n_background_terms: int = 6,
    extra_cycles: int = 2,
    observed_peaks=None,
    peak_tol_deg: float = 0.15,
) -> dict[str, Any]:
    """Le Bail whole-pattern validation of a candidate cell. See ``TOOL_SPEC``.

    Thin wrapper over the GSAS-II engine; lazy-imported so the module stays
    importable without the optional ``gsas`` dependency."""
    from ._gsas_engine import lebail_fit
    return lebail_fit(
        two_theta, intensity, cell, bravais=bravais, wavelength=wavelength,
        two_theta_range=two_theta_range, refine_cell=refine_cell,
        refine_zero=refine_zero, n_background_terms=n_background_terms,
        extra_cycles=extra_cycles, observed_peaks=observed_peaks,
        peak_tol_deg=peak_tol_deg,
    )
