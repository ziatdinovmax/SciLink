"""``simulate_xrd_pattern`` tool — kinematic XRD pattern from a crystal structure.

Thin wrapper around :class:`pymatgen.analysis.diffraction.xrd.XRDCalculator`.
The wrapper exists so the LLM-generated script imports a single, named tool
(via skill-registry registration) instead of constructing the calculator
directly — the skill bundle owns the convention.
"""

from __future__ import annotations

import logging
from typing import Any, Union

from ..._shared._spec import ToolSpec

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    PYMATGEN_XRD_AVAILABLE = True
except ImportError:
    PYMATGEN_XRD_AVAILABLE = False
    Structure = None  # type: ignore
    XRDCalculator = None  # type: ignore

_logger = logging.getLogger(__name__)


TOOL_SPEC = ToolSpec(
    name="simulate_xrd_pattern",
    description=(
        "Compute an XRD pattern from a crystal structure. Default engine is "
        "pymatgen's kinematic XRDCalculator; a pluggable 'engine' selects the "
        "simulator so a heavier backend (e.g. GSAS-II) can be swapped in later "
        "with no change to downstream scoring. Returns 2-theta peak positions, "
        "relative intensities (max = 100), Miller indices, and d-spacings."
    ),
    import_line="from scilink.skills.structure_matching.xrd.simulate_xrd import simulate_xrd_pattern",
    signature=(
        "simulate_xrd_pattern(structure_path: str, wavelength: str | float = "
        "'CuKa', two_theta_range: tuple = (10, 90), engine: str = 'pymatgen') -> dict"
    ),
    parameters={
        "structure_path": {
            "type": "str",
            "description": "Path to a CIF file (typically returned by search_structures).",
        },
        "wavelength": {
            "type": "str | float",
            "description": (
                "X-ray source name ('CuKa', 'CuKa1', 'MoKa', 'CoKa', 'FeKa', "
                "'AgKa', 'CrKa') or wavelength in angstroms."
            ),
        },
        "two_theta_range": {
            "type": "tuple",
            "description": "(min, max) in degrees. Default (10, 90).",
        },
        "engine": {
            "type": "str",
            "description": (
                "Simulation backend. 'pymatgen' (default, kinematic peak list, "
                "always available; sufficient for peak-list identification). "
                "'gsas' — GSAS-II full-profile simulation with realistic "
                "instrument/sample broadening (optional 'gsas' extra); returns "
                "the same core keys PLUS 'profile_two_theta'/'profile_intensities' "
                "(the continuous pattern), for full-pattern matching. Scoring on "
                "the shared core keys is unaffected by the choice."
            ),
        },
        "crystallite_um": {
            "type": "float",
            "description": (
                "GSAS-II engine only (ignored by pymatgen). Isotropic crystallite "
                "size in micrometres for size broadening. LOWER it toward a "
                "nanocrystalline sample to broaden the peaks (e.g. 0.05 = 50 nm "
                "gives visibly broad lines); RAISE it for sharp lines. Default "
                "~10 µm (essentially sharp). Set this whenever you want the "
                "profile to reflect a realistic finite crystallite size."
            ),
        },
        "peak_rel_height": {
            "type": "float",
            "description": (
                "GSAS-II engine only (ignored by pymatgen). Peak-pick threshold "
                "for the returned discrete peak list, as a fraction of the max "
                "profile intensity (default 0.01). LOWER to report weaker "
                "reflections in 'two_theta'/'intensities'; RAISE to keep only "
                "strong peaks. Does not affect the continuous profile."
            ),
        },
    },
    required=["structure_path"],
    returns=(
        "dict with 'two_theta' (list[float]), 'intensities' (list[float]; "
        "normalized so max = 100), 'hkls' (list[list[int]]; primary "
        "reflection per peak), 'd_spacings' (list[float]), 'wavelength' "
        "(Å), 'structure_path' (echo), 'engine' (which simulator produced it). "
        "The 'gsas' engine additionally returns 'profile_two_theta' and "
        "'profile_intensities' (the continuous broadened pattern)."
    ),
    when_to_use=(
        "After search_structures, to generate a simulated pattern from each "
        "candidate structure. Chain with score_xrd_match to quantify the "
        "match against experimental data."
    ),
)


def simulate_xrd_pattern(
    structure_path: str,
    wavelength: Union[str, float] = "CuKa",
    two_theta_range: tuple = (10, 90),
    engine: str = "pymatgen",
    crystallite_um: float = None,
    peak_rel_height: float = None,
) -> dict[str, Any]:
    """Compute an XRD pattern from a crystal structure. See ``TOOL_SPEC``.

    ``engine`` selects the simulation backend ('pymatgen' default). All engines
    return the SAME core dict contract (two_theta / intensities / hkls /
    d_spacings / wavelength), so everything downstream — scoring, lattice-scale
    alignment, matching — is engine-agnostic. Swapping in a different simulator
    (e.g. GSAS-II) is localized to this module: implement a function with the
    ``(structure_path, wavelength, two_theta_range, **kwargs) -> dict`` signature
    and register it in ``_ENGINES`` (optional dependency, lazy-imported).

    ``crystallite_um`` / ``peak_rel_height`` are engine-specific knobs applied
    only by engines that use them (the 'gsas' full-profile engine); the pymatgen
    peak-list engine ignores them. Left as ``None`` they take the engine
    default."""
    try:
        backend = _ENGINES[engine]
    except KeyError:
        raise ValueError(
            f"Unknown XRD simulation engine {engine!r}. Available: "
            f"{sorted(_ENGINES)}. Register a new engine in "
            f"scilink.skills.structure_matching.xrd.simulate_xrd._ENGINES."
        )
    # forward only the engine knobs the caller actually set (None -> engine default)
    kwargs = {k: v for k, v in
              {"crystallite_um": crystallite_um, "peak_rel_height": peak_rel_height}.items()
              if v is not None}
    result = backend(structure_path, wavelength, two_theta_range, **kwargs)
    result["engine"] = engine
    return result


def _simulate_pymatgen(structure_path, wavelength, two_theta_range, **_ignored) -> dict[str, Any]:
    """Default engine: pymatgen's kinematic ``XRDCalculator`` (always available
    with the structure-matching extra; sufficient for peak-list identification)."""
    if not PYMATGEN_XRD_AVAILABLE:
        raise RuntimeError(
            "simulate_xrd_pattern requires pymatgen with XRDCalculator support; "
            "install pymatgen >= 2022.x"
        )

    structure = Structure.from_file(structure_path)
    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=tuple(two_theta_range))

    hkls = [_primary_hkl(entry) for entry in pattern.hkls]

    return {
        "two_theta": [float(x) for x in pattern.x],
        "intensities": [float(y) for y in pattern.y],
        "hkls": hkls,
        "d_spacings": [float(d) for d in pattern.d_hkls],
        "wavelength": float(calc.wavelength),
        "structure_path": structure_path,
    }


def _simulate_gsas(structure_path, wavelength, two_theta_range, **kwargs) -> dict[str, Any]:
    """Optional engine: GSAS-II full-profile simulation (``gsas`` extra).

    Lazy-imported so the module stays importable without GSAS-II. Returns the
    same core contract keys as the pymatgen engine (peak-picked from the profile)
    plus the continuous ``profile_two_theta`` / ``profile_intensities`` that
    GSAS-II uniquely provides — see ``_gsas_engine`` for the design rationale.
    ``kwargs`` forwards engine knobs (crystallite_um, peak_rel_height)."""
    from ._gsas_engine import simulate_gsas
    return simulate_gsas(structure_path, wavelength, tuple(two_theta_range), **kwargs)


# XRD simulation-engine registry — the swap point. Each entry maps an engine
# name to a callable with the same signature and return contract as
# ``_simulate_pymatgen``. The GSAS-II engine (optional ``gsas`` extra) is
# lazy-imported inside ``_simulate_gsas`` so this module stays importable
# without it; peak-based downstream code is unaffected because the core output
# keys are identical (GSAS only *adds* the continuous-profile keys).
_ENGINES = {
    "pymatgen": _simulate_pymatgen,
    "gsas": _simulate_gsas,
}


def _primary_hkl(entry: Any) -> list[int]:
    """Extract the first (h, k, l) triplet from pymatgen's per-peak hkl payload.

    ``pattern.hkls`` is a list[list[dict]]; each inner dict has key 'hkl'
    pointing to a length-3 tuple. Symmetry-equivalent reflections share a
    peak — we return the first for brevity.
    """
    if not entry:
        return [0, 0, 0]
    first = entry[0]
    if isinstance(first, dict):
        hkl = first.get("hkl") or first.get("HKL")
    else:
        hkl = first
    if hkl is None:
        return [0, 0, 0]
    return [int(c) for c in hkl]
