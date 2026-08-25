"""``calibrate_zero`` tool — 2θ calibration from an internal standard.

Real powder work mixes a known standard (Si, LaB₆, corundum) into the sample so
every pattern carries reference lines of exactly known position; the measured
offsets of those lines calibrate the diffractometer errors. Two aberrations
dominate a lab pattern and have distinct angular signatures:

* **zero error** — constant Δ2θ across the pattern;
* **specimen displacement** — Δ2θ ∝ cos θ (largest at low angle).

The tool matches the standard's reference lines (computed from NIST SRM lattice
constants at the query wavelength) to the measured peaks, fits
``Δ2θ = zero + disp·cosθ``, and returns both the fitted terms and the sample's
peak list with the calibration applied. Feed ``zero_offset`` to
``index_pattern`` / ``refine_rietveld``, or use ``corrected_peaks`` directly —
autoindexing is exquisitely sensitive to zero error, so calibrating first is
the single cheapest way to improve blind identification on lab data.

Positions-only: intensities of the standard are irrelevant here, so the
reference lines are generated from the lattice + extinctions alone."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

# NIST SRM lattice constants (Å, ~22 °C). Positions-only reference data.
# Si: SRM 640f (diamond cubic, Fd-3m); LaB6: SRM 660c (Pm-3m, all hkl allowed);
# corundum: SRM 676a (R-3c, hex setting).
_STANDARDS = {
    "Si":       {"lattice": ("cubic", 5.431144), "extinction": "F_diamond"},
    "LaB6":     {"lattice": ("cubic", 4.156826), "extinction": "P"},
    "corundum": {"lattice": ("hex", 4.759091, 12.991779), "extinction": "R-3c"},
}

_ALIASES = {"cuka": 1.5406, "cuka1": 1.54056, "moka": 0.71073, "coka": 1.78897,
            "feka": 1.93604, "crka": 2.28970, "agka": 0.55941}


def _lam(wavelength: Any) -> float:
    if isinstance(wavelength, (int, float)):
        return float(wavelength)
    key = str(wavelength).strip().lower().replace(" ", "").replace("-", "")
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unrecognized wavelength {wavelength!r}")


def _reference_two_theta(standard: str, lam: float, tt_max: float) -> np.ndarray:
    """Reference line positions (2θ°) for a standard from lattice + extinctions."""
    spec = _STANDARDS[standard]
    kind = spec["lattice"][0]
    ds = set()
    N = 12
    if kind == "cubic":
        a = spec["lattice"][1]
        for h in range(N):
            for k in range(N):
                for l in range(N):
                    if h == k == l == 0:
                        continue
                    if spec["extinction"] == "F_diamond":
                        # F-centering: h,k,l all even or all odd; diamond glide
                        # additionally kills all-even with h+k+l = 4n+2.
                        par = {h % 2, k % 2, l % 2}
                        if len(par) != 1:
                            continue
                        if par == {0} and (h + k + l) % 4 != 0:
                            continue
                    ds.add(round(a / np.sqrt(h * h + k * k + l * l), 6))
    else:  # hexagonal setting (corundum R-3c: -h+k+l = 3n)
        a, c = spec["lattice"][1], spec["lattice"][2]
        for h in range(-N, N):
            for k in range(-N, N):
                for l in range(N):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    if (-h + k + l) % 3 != 0:
                        continue
                    inv_d2 = 4.0 / 3.0 * (h * h + h * k + k * k) / a**2 + l * l / c**2
                    if inv_d2 <= 0:
                        continue
                    ds.add(round(1.0 / np.sqrt(inv_d2), 6))
    ds = np.array(sorted(ds, reverse=True))
    with np.errstate(invalid="ignore"):
        s = lam / (2.0 * ds)
    tt = 2.0 * np.degrees(np.arcsin(s[s <= 1.0]))
    return tt[(tt > 5.0) & (tt <= tt_max)]


TOOL_SPEC = ToolSpec(
    name="calibrate_zero",
    description=(
        "Calibrate the 2θ scale from an INTERNAL STANDARD (Si, LaB6, or "
        "corundum mixed into the sample): matches the standard's exactly-known "
        "reference lines (NIST SRM lattice constants) to the measured peaks and "
        "fits the two dominant diffractometer aberrations — constant zero error "
        "and specimen displacement (∝ cos θ). Returns the fitted offsets and the "
        "measured peak list with the calibration applied (standard lines "
        "removed). Autoindexing and lattice refinement are exquisitely sensitive "
        "to zero error, so when a standard is present, calibrate FIRST and feed "
        "'zero_offset' / 'corrected_peaks' to index_pattern, search_match_pattern "
        "and refine_rietveld."
    ),
    import_line="from scilink.skills.structure_matching.xrd.calibrate_zero import calibrate_zero",
    signature=(
        "calibrate_zero(two_theta_peaks, standard='Si', wavelength='CuKa', "
        "tol_deg=0.25, fit_displacement=True, min_lines=3) -> dict"
    ),
    parameters={
        "two_theta_peaks": {"type": "list[float]", "description": "ALL measured peak positions (2θ°, sample + standard together) — extract_peaks' 'positions'."},
        "standard": {"type": "str", "description": "Which internal standard is in the sample: 'Si' (SRM 640f), 'LaB6' (SRM 660c — most lines, best for calibration), or 'corundum' (SRM 676a)."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or Å; sets where the standard's lines fall."},
        "tol_deg": {"type": "float", "description": "Match window between a reference line and a measured peak (default 0.25°). RAISE if the zero error may be large; LOWER on already-good data to avoid grabbing sample peaks."},
        "fit_displacement": {"type": "bool", "description": "Also fit the specimen-displacement term (∝ cos θ, default True). Turn OFF with few matched lines (<4) — two terms need more constraints than one."},
        "min_lines": {"type": "int", "description": "Minimum matched standard lines to accept a calibration (default 3); fewer → error rather than a garbage fit."},
    },
    required=["two_theta_peaks"],
    returns=(
        "dict with 'zero_offset_deg' (constant term — pass to index_pattern / "
        "refine_rietveld; SUBTRACT from measured 2θ to correct), "
        "'displacement_coeff' (cosθ term, 0 when not fitted), 'residual_rms_deg' "
        "(post-fit line-position residual — ≥ tol/2 means a poor calibration), "
        "'n_lines_matched', 'matched_lines' ([measured, reference] pairs), "
        "'corrected_peaks' (the NON-standard measured peaks with the calibration "
        "applied — feed these to indexing / search-match)."
    ),
    when_to_use=(
        "Whenever the metadata says an internal standard was mixed into the "
        "sample, BEFORE any indexing, search-match, or refinement. Not usable "
        "when no standard is present — then rely on the Zero refinement inside "
        "refine_rietveld / validate_cell_lebail instead."
    ),
)


def calibrate_zero(
    two_theta_peaks: Sequence[float],
    standard: str = "Si",
    wavelength: Any = "CuKa",
    tol_deg: float = 0.25,
    fit_displacement: bool = True,
    min_lines: int = 3,
) -> dict[str, Any]:
    """Fit zero error (+ specimen displacement) from internal-standard lines.

    See ``TOOL_SPEC``. Model: measured = reference + zero + disp·cos(θ_ref)."""
    if standard not in _STANDARDS:
        raise ValueError(f"Unknown standard {standard!r}; choose from "
                         f"{sorted(_STANDARDS)}")
    lam = _lam(wavelength)
    tt = np.asarray(sorted(float(t) for t in two_theta_peaks), dtype=float)
    if tt.size < min_lines:
        raise ValueError(f"need at least {min_lines} measured peaks")
    ref = _reference_two_theta(standard, lam, tt_max=float(tt.max()) + 1.0)

    # match each reference line to its nearest measured peak within tol
    pairs = []          # (measured, reference)
    used = set()
    for r in ref:
        i = int(np.argmin(np.abs(tt - r)))
        if abs(tt[i] - r) <= float(tol_deg) and i not in used:
            pairs.append((float(tt[i]), float(r)))
            used.add(i)
    if len(pairs) < int(min_lines):
        raise ValueError(
            f"only {len(pairs)} standard lines matched within {tol_deg} deg — "
            "is the standard really present (and the wavelength right)? RAISE "
            "tol_deg if the zero error may be larger."
        )

    meas = np.array([p[0] for p in pairs])
    refv = np.array([p[1] for p in pairs])
    delta = meas - refv
    if fit_displacement and len(pairs) >= 4:
        A = np.column_stack([np.ones_like(refv), np.cos(np.radians(refv / 2.0))])
        coef, *_ = np.linalg.lstsq(A, delta, rcond=None)
        zero, disp = float(coef[0]), float(coef[1])
        model = A @ coef
    else:
        zero, disp = float(np.mean(delta)), 0.0
        model = np.full_like(delta, zero)
    rms = float(np.sqrt(np.mean((delta - model) ** 2)))

    # corrected SAMPLE peaks: remove the matched standard lines, undo the model
    corrected = []
    for i, t in enumerate(tt):
        if i in used:
            continue
        corr = t - (zero + disp * np.cos(np.radians(t / 2.0)))
        corrected.append(round(float(corr), 4))

    return {
        "zero_offset_deg": round(zero, 4),
        "displacement_coeff": round(disp, 4),
        "residual_rms_deg": round(rms, 4),
        "n_lines_matched": len(pairs),
        "matched_lines": [[round(m, 3), round(r, 3)] for m, r in pairs],
        "corrected_peaks": corrected,
        "standard": standard,
        "note": (
            "Measured = true + zero + disp*cos(theta): 'corrected_peaks' have "
            "the model SUBTRACTED (standard lines removed) — use them directly "
            "for indexing/search-match. CAVEAT: over a typical angular range "
            "cos(theta) is nearly constant, so the zero and displacement terms "
            "trade off — their SUM (the applied correction) is accurate but the "
            "split is not. Prefer 'corrected_peaks'; pass 'zero_offset_deg' "
            "alone to other tools only from a fit_displacement=False "
            "calibration. residual_rms_deg >= tol/2 or few matched lines means "
            "a poor calibration; check the standard choice and wavelength."
        ),
    }
