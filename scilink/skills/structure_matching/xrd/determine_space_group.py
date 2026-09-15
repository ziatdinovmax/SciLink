"""``determine_space_group`` tool — systematic-absence analysis of an indexed cell.

The step between cell validation and structure work in the discovery workflow:
once a cell Le-Bail-fits the pattern, WHICH reflections are absent narrows the
space group. Reflection conditions (International Tables) are tested against
the observed peaks: lattice centerings (I, F, A, B, C, R), axial screw axes
(h00/0k0/00l parity), and zonal glide planes (0kl/h0l/hk0 parities). Each
condition is graded on evidence — how many testable observed reflections obey
it, and whether any observed reflection violates it outright.

Honest scope: powder data cannot always decide a space group (different groups
share extinction symbols, and sparse patterns leave conditions untestable), so
the output is EVIDENCE plus a list of common space groups consistent with it —
ranked by how frequently each occurs in real crystals, and explicitly not
exhaustive. Pure numpy — works without the ``gsas`` extra; use the cell exactly
as validated by ``validate_cell_lebail``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

_ALIASES = {"cuka": 1.5406, "cuka1": 1.54056, "moka": 0.71073, "coka": 1.78897,
            "feka": 1.93604, "crka": 2.28970, "agka": 0.55941}

# Reflection conditions (International Tables). Each: (name, class-selector,
# parity rule). A condition is TESTABLE on a peak only when every plausible hkl
# assignment of that peak falls in the condition's class.
_CENTERINGS = {
    "P": lambda h, k, l: True,
    "I": lambda h, k, l: (h + k + l) % 2 == 0,
    "F": lambda h, k, l: (h % 2 == k % 2 == l % 2),
    "A": lambda h, k, l: (k + l) % 2 == 0,
    "B": lambda h, k, l: (h + l) % 2 == 0,
    "C": lambda h, k, l: (h + k) % 2 == 0,
    "R": lambda h, k, l: (-h + k + l) % 3 == 0,   # obverse, hexagonal axes
}
_AXIAL = [  # screw axes
    ("h00: h=2n (2_1 along a)", lambda h, k, l: k == 0 and l == 0, lambda h, k, l: h % 2 == 0),
    ("0k0: k=2n (2_1 along b)", lambda h, k, l: h == 0 and l == 0, lambda h, k, l: k % 2 == 0),
    ("00l: l=2n (2_1 along c)", lambda h, k, l: h == 0 and k == 0, lambda h, k, l: l % 2 == 0),
    ("00l: l=4n (4_1/4_3)",     lambda h, k, l: h == 0 and k == 0, lambda h, k, l: l % 4 == 0),
    ("00l: l=3n (3_1/3_2/6_2)", lambda h, k, l: h == 0 and k == 0, lambda h, k, l: l % 3 == 0),
]
_ZONAL = [  # glide planes
    ("0kl: k=2n (b glide ⊥a)",   lambda h, k, l: h == 0, lambda h, k, l: k % 2 == 0),
    ("0kl: l=2n (c glide ⊥a)",   lambda h, k, l: h == 0, lambda h, k, l: l % 2 == 0),
    ("0kl: k+l=2n (n glide ⊥a)", lambda h, k, l: h == 0, lambda h, k, l: (k + l) % 2 == 0),
    ("h0l: h=2n (a glide ⊥b)",   lambda h, k, l: k == 0, lambda h, k, l: h % 2 == 0),
    ("h0l: l=2n (c glide ⊥b)",   lambda h, k, l: k == 0, lambda h, k, l: l % 2 == 0),
    ("h0l: h+l=2n (n glide ⊥b)", lambda h, k, l: k == 0, lambda h, k, l: (h + l) % 2 == 0),
    ("hk0: h=2n (a glide ⊥c)",   lambda h, k, l: l == 0, lambda h, k, l: h % 2 == 0),
    ("hk0: k=2n (b glide ⊥c)",   lambda h, k, l: l == 0, lambda h, k, l: k % 2 == 0),
    ("hk0: h+k=2n (n glide ⊥c)", lambda h, k, l: l == 0, lambda h, k, l: (h + k) % 2 == 0),
]

# Common space groups per crystal system with the conditions they REQUIRE
# (centering letter + names from the tables above). Frequency-ordered (these
# few groups cover the large majority of real crystals). NOT exhaustive — the
# output says so.
_COMMON_SGS = {
    "triclinic": [("P-1", "P", []), ("P1", "P", [])],
    "monoclinic": [
        ("P2_1/c", "P", ["0k0: k=2n (2_1 along b)", "h0l: l=2n (c glide ⊥b)"]),
        ("C2/c",   "C", ["h0l: l=2n (c glide ⊥b)"]),
        ("P2_1",   "P", ["0k0: k=2n (2_1 along b)"]),
        ("C2/m",   "C", []),
        ("P2/m",   "P", []),
    ],
    "orthorhombic": [
        ("Pnma",     "P", ["0kl: k+l=2n (n glide ⊥a)", "hk0: h=2n (a glide ⊥c)"]),
        ("P2_12_12_1", "P", ["h00: h=2n (2_1 along a)", "0k0: k=2n (2_1 along b)", "00l: l=2n (2_1 along c)"]),
        ("Pbca",     "P", ["0kl: k=2n (b glide ⊥a)", "h0l: l=2n (c glide ⊥b)", "hk0: h=2n (a glide ⊥c)"]),
        ("Cmcm",     "C", ["h0l: l=2n (c glide ⊥b)"]),
        ("Pnnm",     "P", ["0kl: k+l=2n (n glide ⊥a)", "h0l: h+l=2n (n glide ⊥b)"]),
        ("Immm",     "I", []),
        ("Fmmm",     "F", []),
        ("Cmmm",     "C", []),
        ("Pmmm",     "P", []),
    ],
    "tetragonal": [
        ("P4_2/mnm", "P", ["0kl: k+l=2n (n glide ⊥a)"]),
        ("I4/mmm",   "I", []),
        ("I4_1/amd", "I", ["00l: l=4n (4_1/4_3)"]),
        ("P4/mmm",   "P", []),
        ("P4/nmm",   "P", ["hk0: h+k=2n (n glide ⊥c)"]),
    ],
    "trigonal/hexagonal": [
        ("R-3m (hex)", "R", []),
        ("R-3c (hex)", "R", ["00l: l=3n (3_1/3_2/6_2)"]),  # hex axes: 00l l=6n implied with R
        ("P6_3/mmc",  "P", []),
        ("P6/mmm",    "P", []),
        ("P3_121",    "P", ["00l: l=3n (3_1/3_2/6_2)"]),
        ("P-3m1",     "P", []),
    ],
    "cubic": [
        ("Fm-3m", "F", []), ("Fd-3m", "F", []), ("F-43m", "F", []),
        ("Im-3m", "I", []), ("Ia-3d", "I", []),
        ("Pm-3m", "P", []), ("Pa-3", "P", ["0kl: k=2n (b glide ⊥a)"]),
    ],
}


def _lam(wavelength: Any) -> float:
    if isinstance(wavelength, (int, float)):
        return float(wavelength)
    key = str(wavelength).strip().lower().replace(" ", "").replace("-", "")
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unrecognized wavelength {wavelength!r}")


def _dhkl(cell: Sequence[float], hkls: np.ndarray) -> np.ndarray:
    """d-spacings from the reciprocal metric tensor (any crystal system)."""
    a, b, c, al, be, ga = (float(v) for v in cell)
    al, be, ga = np.radians([al, be, ga])
    ca, cb, cg = np.cos([al, be, ga])
    sa, sb, sg = np.sin([al, be, ga])
    V = a * b * c * np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    # reciprocal cell
    ast, bst, cst = b * c * sa / V, a * c * sb / V, a * b * sg / V
    cast = (cb * cg - ca) / (sb * sg)
    cbst = (ca * cg - cb) / (sa * sg)
    cgst = (ca * cb - cg) / (sa * sb)
    h, k, l = hkls[:, 0], hkls[:, 1], hkls[:, 2]
    inv_d2 = (h**2 * ast**2 + k**2 * bst**2 + l**2 * cst**2
              + 2 * k * l * bst * cst * cast
              + 2 * h * l * ast * cst * cbst
              + 2 * h * k * ast * bst * cgst)
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(inv_d2)


TOOL_SPEC = ToolSpec(
    name="determine_space_group",
    description=(
        "Systematic-absence analysis: given a VALIDATED unit cell (from "
        "validate_cell_lebail) and the measured peak positions, tests the "
        "International-Tables reflection conditions — lattice centerings "
        "(I/F/A/B/C/R), screw axes (h00/0k0/00l), glide planes (0kl/h0l/hk0) — "
        "against which reflections are observed vs absent, and returns the "
        "evidence plus COMMON space groups consistent with it (frequency-"
        "ranked, explicitly not exhaustive — powder extinctions often cannot "
        "decide a unique group). The step between cell validation and any "
        "structure work in the new-phase workflow. Pure computation, no "
        "optional dependencies."
    ),
    import_line="from scilink.skills.structure_matching.xrd.determine_space_group import determine_space_group",
    signature=(
        "determine_space_group(observed_peaks, cell, crystal_system, "
        "wavelength='CuKa', tol_deg=0.1, min_evidence=2) -> dict"
    ),
    parameters={
        "observed_peaks": {"type": "list[float]", "description": "Measured peak positions (2θ°). Use calibrated/corrected positions — absences are parity statements about WHICH lines exist, so a zero error corrupts assignments."},
        "cell": {"type": "dict | list", "description": "The validated cell (a,b,c,alpha,beta,gamma) — use validate_cell_lebail's refined 'lattice', not the raw indexed cell."},
        "crystal_system": {"type": "str", "description": "One of 'cubic','tetragonal','trigonal/hexagonal','orthorhombic','monoclinic','triclinic' (index_pattern's candidate crystal_system)."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa',…) or Å."},
        "tol_deg": {"type": "float", "description": "Assignment window between an observed peak and a generated reflection (default 0.1° — the cell is already refined, so keep this TIGHT; RAISE only for broad peaks). A loose window multiplies ambiguous assignments and destroys the evidence."},
        "min_evidence": {"type": "int", "description": "Minimum testable reflections for a condition to be reported as 'obeyed' rather than 'insufficient evidence' (default 2)."},
    },
    required=["observed_peaks", "cell", "crystal_system"],
    returns=(
        "dict with 'centering' ({letter: verdict}) — the inferred lattice "
        "centering is the most restrictive one not violated; 'conditions' "
        "(per screw/glide condition: obeyed / violated / insufficient, with "
        "n_testable and the violating peaks); 'consistent_space_groups' "
        "(common groups whose required conditions are all non-violated, "
        "frequency-ranked, NOT exhaustive — groups sharing an extinction "
        "symbol are indistinguishable from powder absences alone); "
        "'n_peaks_assigned', 'warnings'."
    ),
    when_to_use=(
        "In the new-phase (discovery) workflow, after validate_cell_lebail "
        "confirms a cell: narrows the space group before any structure "
        "solution or Rietveld with a specific group. Skip when the phase was "
        "identified from a database — its space group is already known."
    ),
)


def determine_space_group(
    observed_peaks: Sequence[float],
    cell,
    crystal_system: str,
    wavelength: Any = "CuKa",
    tol_deg: float = 0.1,
    min_evidence: int = 2,
) -> dict[str, Any]:
    """Systematic-absence evidence + consistent common space groups.

    See ``TOOL_SPEC``. Logic: peaks are assigned to P-lattice reflections of
    the given cell; a condition is VIOLATED when some observed peak's every
    plausible assignment is a reflection the condition forbids, and OBEYED when
    >= min_evidence testable reflections follow it with none violating."""
    system = crystal_system.strip().lower()
    if system in ("trigonal", "hexagonal"):
        system = "trigonal/hexagonal"
    if system not in _COMMON_SGS:
        raise ValueError(f"unknown crystal_system {crystal_system!r}; choose "
                         f"from {sorted(_COMMON_SGS)}")
    if isinstance(cell, dict):
        cv = [cell[k] for k in ("a", "b", "c", "alpha", "beta", "gamma")]
    else:
        cv = list(cell)
    lam = _lam(wavelength)
    obs = np.asarray(sorted(float(t) for t in observed_peaks), dtype=float)
    if obs.size < 5:
        raise ValueError("need at least 5 observed peaks for absence analysis")

    # generate P-lattice reflections to just past the last observed peak
    d_min = lam / (2.0 * np.sin(np.radians((obs.max() + 1.0) / 2.0)))
    hmax = int(np.ceil(max(cv[0], cv[1], cv[2]) / d_min)) + 1
    rng = np.arange(-hmax, hmax + 1)
    H, K, L = np.meshgrid(rng, rng, rng, indexing="ij")
    hkl = np.column_stack([H.ravel(), K.ravel(), L.ravel()])
    hkl = hkl[~np.all(hkl == 0, axis=1)]
    d = _dhkl(cv, hkl)
    keep = d >= d_min * 0.98
    hkl, d = hkl[keep], d[keep]
    with np.errstate(invalid="ignore"):
        tt = 2.0 * np.degrees(np.arcsin(np.clip(lam / (2.0 * d), 0, 1)))

    # assign: for each observed peak, ALL plausible hkl within tol
    assignments = []          # list of (peak, hkl_array)
    unassigned = []
    for p in obs:
        m = np.abs(tt - p) <= float(tol_deg)
        if m.any():
            assignments.append((float(p), hkl[m]))
        else:
            unassigned.append(round(float(p), 3))

    warnings = []
    if unassigned:
        warnings.append(
            f"{len(unassigned)} peaks not assignable to this cell "
            f"({unassigned[:6]}{'…' if len(unassigned) > 6 else ''}) — impurity "
            "lines or a wrong cell; absence evidence ignores them.")

    def _grade(selector, rule):
        n_testable = 0
        violators = []
        for p, hs in assignments:
            in_class = np.array([selector(*row) for row in hs])
            if not in_class.any():
                continue
            if in_class.all():          # unambiguous: peak belongs to the class
                n_testable += 1
                if not any(rule(*row) for row in hs[in_class]):
                    violators.append(round(p, 3))
        if violators:
            return {"verdict": "violated", "n_testable": n_testable,
                    "violating_peaks": violators}
        if n_testable >= int(min_evidence):
            return {"verdict": "obeyed", "n_testable": n_testable}
        return {"verdict": "insufficient evidence", "n_testable": n_testable}

    # centerings: graded over ALL assigned peaks (the class is 'every hkl')
    centering = {}
    for letter, rule in _CENTERINGS.items():
        if letter == "P":
            continue
        violators = []
        for p, hs in assignments:
            if not any(rule(*row) for row in hs):
                violators.append(round(p, 3))
        centering[letter] = ({"verdict": "violated", "violating_peaks": violators[:8]}
                             if violators else {"verdict": "consistent"})

    conditions = {}
    for name, selector, rule in _AXIAL + _ZONAL:
        conditions[name] = _grade(selector, rule)

    # consistent common space groups: required centering not violated AND every
    # required condition not violated
    consistent = []
    for sgname, cen, reqs in _COMMON_SGS[system]:
        if cen != "P" and centering.get(cen, {}).get("verdict") == "violated":
            continue
        if any(conditions.get(r, {}).get("verdict") == "violated" for r in reqs):
            continue
        support = sum(1 for r in reqs if conditions.get(r, {}).get("verdict") == "obeyed")
        consistent.append({"space_group": sgname, "centering": cen,
                           "required_conditions": reqs,
                           "n_conditions_obeyed": support})
    # richest evidence first, then original (frequency) order
    consistent.sort(key=lambda r: -r["n_conditions_obeyed"])

    return {
        "centering": centering,
        "conditions": conditions,
        "consistent_space_groups": consistent,
        "n_peaks_assigned": len(assignments),
        "unassigned_peaks": unassigned,
        "warnings": warnings,
        "note": (
            "Evidence, not a verdict: powder absences often cannot decide a "
            "unique group (groups sharing an extinction symbol are "
            "indistinguishable here), and 'obeyed' with few testable "
            "reflections is weak. The list is frequency-ranked COMMON groups "
            "consistent with the evidence — not exhaustive. Corroborate the "
            "final choice by a Le Bail fit with that group's extinctions."
        ),
    }
