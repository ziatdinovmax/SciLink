"""``score_xrd_match_robust`` tool — peak-list-based scoring with two algorithms.

The robust tier of the structure-matching workflow. Operates on peak
lists (extracted from continuous patterns via :func:`extract_peaks`)
rather than continuous data, so it factors out background, scale, and
preferred-orientation effects that profile-level metrics fold into the
residual.

Two algorithms:

- ``"hanawalt"`` — classical search-match. For each top-N experimental
  peak, find the closest simulated peak within a tolerance window;
  score on coverage + position residual + (lightly) intensity residual.
  Pure numpy / scipy. Default algorithm; matches the long-standing
  Hanawalt-Fink approach implemented in commercial XRD packages.

- ``"mip"`` — mixed-integer linear programming via PuLP. Jointly fits
  peak assignment + zero-shift + lattice scale by gridding scale and
  solving one MILP per scale. Provably optimal under the formulation;
  natural to extend to multi-phase later (one assignment matrix per
  candidate phase, phase-fraction continuous variables). Requires
  ``pulp`` (declared in ``scilink[structure-matching]``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec
from .extract_peaks import extract_peaks

try:
    import pulp  # type: ignore
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    pulp = None  # type: ignore

_logger = logging.getLogger(__name__)


# Hanawalt verdict thresholds on the figure-of-merit (0..1, higher is better)
_HANAWALT_ACCEPT_MIN = 0.70
_HANAWALT_MARGINAL_MIN = 0.40

# MIP verdict thresholds — cost is normalized to tolerance, so a "match" with
# 0 residual contributes 0 and an unmatched peak contributes 1.
_MIP_ACCEPT_MAX = 0.25
_MIP_MARGINAL_MAX = 0.55

# MIP cost weights (intensity-weighted form). The cost penalises the UNEXPLAINED
# INTENSITY fraction (weak noise peaks the extractor picked up barely count; a
# missing STRONG reflection still drives cost up) plus the mean matched-peak
# residual. Replaces the old count-based "each unmatched peak costs a tolerance
# unit", which let ~10 noise peaks reject an otherwise-correct multi-phase match.
_COST_UNCOVERED_W = 0.8
_COST_RESIDUAL_W = 0.2

# Multi-phase MIP: per-phase activation penalty in the MILP objective. Each
# active phase costs this many "implicit unmatched peaks" — so a phase is
# only kept when it accounts for more matches than that. Tuned to ~3 peaks:
# a phase that explains fewer than 3 peaks above noise probably isn't a
# real component.
_MULTIPHASE_ACTIVATION_PENALTY = 3.0

# Multi-phase MIP: predicted-but-absent penalty (BIDIRECTIONAL matching). The
# bare "maximize matched exp peaks" objective rewards a peak-RICH phase for
# covering experimental peaks while never charging it for its OWN strong
# reflections that are absent from the data — so a Magnéli-type suboxide with
# ~30 reflections can out-cover the true anatase+rutile by overlap. We add a
# penalty: for each ACTIVE phase, every strong predicted peak (relative
# intensity >= _STRONG_SIM_FRAC) that is NOT matched to an experimental peak
# costs `_MULTIPHASE_ABSENT_PENALTY * (its relative intensity)`. A phase whose
# strong predicted peaks are mostly missing then loses, even if it covers a few
# experimental peaks. Weak predicted reflections (the long tail, often below
# detection) are NOT penalised — only the strong ones a real phase must show.
_MULTIPHASE_ABSENT_PENALTY = 3.0
_STRONG_SIM_FRAC = 0.15

# Single-phase Hanawalt predicted-coverage gate. A phase whose OWN strong
# reflections are mostly absent from the data is a coincidental few-peak false
# match (e.g. a wrong-chemistry phase grabbing 2-3 peaks by overlap, or a
# lattice-scaled near-miss). Down-weight its FoM by predicted_coverage once that
# drops below _PRED_COV_FULL; at/above it the gate is 1.0 so a real phase (which
# shows nearly all its strong peaks) is unaffected. This is the single-phase
# analogue of the multiphase predicted-but-absent penalty. Calibrated to 0.65:
# real phases (predicted_coverage ~0.89-1.0 even nanocrystalline) keep full FoM,
# while a wrong-chemistry match (predicted_coverage ~0.46) is cut below the
# reject threshold instead of scoring a false "marginal".
_PRED_COV_FULL = 0.65


TOOL_SPEC = ToolSpec(
    name="score_xrd_match_robust",
    description=(
        "Peak-list-based scoring of a simulated XRD pattern against an "
        "experimental one, with selectable algorithm. 'hanawalt' (default) "
        "is classical figure-of-merit search-match; both algorithms fit a "
        "single lattice scale so a DFT-relaxed reference cell (Materials "
        "Project structures sit a few % off the experimental lattice) is "
        "aligned before matching instead of being falsely rejected. 'mip' is "
        "mixed-integer linear programming with joint peak-assignment + "
        "zero-shift + lattice-scale optimization. Use after the fast tier "
        "identifies a short list of candidates; this tier is for confident "
        "identification on real-lab patterns."
    ),
    import_line="from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_robust",
    signature=(
        "score_xrd_match_robust(exp_two_theta=None, exp_intensity=None, "
        "exp_peaks=None, sim_two_theta, sim_intensity, algorithm='hanawalt', "
        "tol_deg=0.3, max_exp_peaks=20, max_sim_peaks=30, "
        "scale_search=(0.96, 1.04, 0.002), shift_search=(-0.4, 0.4), "
        "fit_lattice_scale=True) -> dict"
    ),
    parameters={
        "exp_two_theta": {
            "type": "list[float]",
            "description": "Experimental 2-theta grid. Required unless exp_peaks is provided.",
        },
        "exp_intensity": {
            "type": "list[float]",
            "description": "Experimental intensity. Required unless exp_peaks is provided.",
        },
        "exp_peaks": {
            "type": "dict",
            "description": "Pre-extracted experimental peak list from extract_peaks. If provided, skips internal peak extraction.",
        },
        "sim_two_theta": {
            "type": "list[float]",
            "description": "Simulated peak positions (degrees) from simulate_xrd_pattern.",
        },
        "sim_intensity": {
            "type": "list[float]",
            "description": "Simulated peak intensities (relative).",
        },
        "algorithm": {
            "type": "str",
            "description": "'hanawalt' (default, fast, pure-Python) or 'mip' (slower, provably optimal under the formulation; requires pulp).",
        },
        "tol_deg": {
            "type": "float",
            "description": "Position tolerance for matching peaks (degrees). Default 0.3.",
        },
        "max_exp_peaks": {
            "type": "int",
            "description": "Cap on experimental peaks considered (strongest kept). Default 20.",
        },
        "max_sim_peaks": {
            "type": "int",
            "description": "Cap on simulated peaks considered (strongest kept). Default 30.",
        },
        "scale_search": {
            "type": "tuple",
            "description": "(min, max, step) lattice-scale grid used by BOTH algorithms. Default (0.96, 1.04, 0.002) — covers ±4% lattice-parameter mismatch (DFT-relaxed MP cells are typically 1-3% larger than the experimental cell; thermal expansion adds more). The scale is applied via Bragg's law (sin-theta scaling), not a constant 2-theta shift.",
        },
        "shift_search": {
            "type": "tuple",
            "description": "(min, max) zero-shift bounds for MIP (degrees). Default (-0.4, 0.4). Ignored by hanawalt.",
        },
        "fit_lattice_scale": {
            "type": "bool",
            "description": "hanawalt only. Default True: fit a single lattice scale (over scale_search) before matching, adopted only if it aligns >=2 reflections so a wrong phase is not force-aligned. Set False to match at the reference lattice as-is (e.g. when the reference is already at experimental conditions).",
        },
    },
    required=["sim_two_theta", "sim_intensity"],
    returns=(
        "dict with 'algorithm', 'figure_of_merit' (or 'cost' for MIP), "
        "'verdict' ('accept' | 'marginal' | 'reject'), 'matched_peaks' "
        "(list of {exp_idx, sim_idx, exp_pos, sim_pos, residual_deg}), "
        "'unmatched_exp' (list of exp peak indices), 'fitted_shift' "
        "(degrees; 0 for hanawalt), 'fitted_scale' (the fitted lattice scale; "
        "1.0 = no scaling / reference already at the experimental lattice, "
        ">1.0 = reference cell larger than experimental, e.g. ~1.02 for a "
        "DFT-relaxed metal — a value near the search bound warrants review), "
        "'n_exp_peaks', 'n_sim_peaks'."
    ),
    when_to_use=(
        "After the fast tier narrows to a few candidates with promising "
        "correlation, run the robust tier for confident identification. "
        "Use 'hanawalt' by default; switch to 'mip' for multi-phase or "
        "constrained matching, or when zero-shift and lattice scale need "
        "to be reported as fitted parameters."
    ),
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_xrd_match_robust(
    sim_two_theta: Sequence[float],
    sim_intensity: Sequence[float],
    *,
    exp_two_theta: Optional[Sequence[float]] = None,
    exp_intensity: Optional[Sequence[float]] = None,
    exp_peaks: Optional[dict] = None,
    algorithm: str = "hanawalt",
    tol_deg: float = 0.3,
    max_exp_peaks: int = 20,
    max_sim_peaks: int = 30,
    scale_search: tuple = (0.96, 1.04, 0.002),
    shift_search: tuple = (-0.4, 0.4),
    fit_lattice_scale: bool = True,
) -> dict[str, Any]:
    """Robust peak-list scoring. See ``TOOL_SPEC`` for full contract."""
    if algorithm not in {"hanawalt", "mip"}:
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. Must be 'hanawalt' or 'mip'."
        )

    exp_pl = _resolve_exp_peaks(exp_peaks, exp_two_theta, exp_intensity, max_exp_peaks)
    sim_pl = _build_sim_peak_list(sim_two_theta, sim_intensity, max_sim_peaks)

    if not exp_pl.positions:
        return _empty_result(algorithm, exp_pl, sim_pl, "no experimental peaks")
    if not sim_pl.positions:
        return _empty_result(algorithm, exp_pl, sim_pl, "no simulated peaks")

    if algorithm == "hanawalt":
        return _score_hanawalt(
            exp_pl, sim_pl, tol_deg=tol_deg,
            scale_search=scale_search if fit_lattice_scale else None)
    return _score_mip(
        exp_pl, sim_pl,
        tol_deg=tol_deg,
        scale_search=scale_search,
        shift_search=shift_search,
    )


# ---------------------------------------------------------------------------
# Internal peak-list representation
# ---------------------------------------------------------------------------

@dataclass
class _PeakList:
    positions: list[float]
    intensities: list[float]
    intensities_norm: list[float]  # max-normalized to 1.0

    @property
    def n(self) -> int:
        return len(self.positions)


def _resolve_exp_peaks(
    exp_peaks: Optional[dict],
    exp_two_theta: Optional[Sequence[float]],
    exp_intensity: Optional[Sequence[float]],
    max_exp_peaks: int,
) -> _PeakList:
    if exp_peaks is not None:
        positions = list(exp_peaks.get("positions", []))
        intensities = list(exp_peaks.get("intensities", []))
    elif exp_two_theta is not None and exp_intensity is not None:
        extracted = extract_peaks(
            exp_two_theta, exp_intensity, max_peaks=max_exp_peaks,
        )
        positions = extracted["positions"]
        intensities = extracted["intensities"]
    else:
        raise ValueError(
            "Provide either exp_peaks or both exp_two_theta and exp_intensity"
        )
    return _to_peak_list(positions, intensities, max_exp_peaks)


def _build_sim_peak_list(
    sim_two_theta: Sequence[float],
    sim_intensity: Sequence[float],
    max_sim_peaks: int,
) -> _PeakList:
    positions = list(sim_two_theta)
    intensities = list(sim_intensity)
    return _to_peak_list(positions, intensities, max_sim_peaks)


def _to_peak_list(
    positions: list[float],
    intensities: list[float],
    max_keep: int,
) -> _PeakList:
    if len(positions) != len(intensities):
        raise ValueError("positions and intensities must have the same length")
    if not positions:
        return _PeakList([], [], [])
    pairs = sorted(zip(positions, intensities), key=lambda p: -p[1])[:max_keep]
    pos_sorted = [float(p) for p, _ in pairs]
    int_sorted = [float(i) for _, i in pairs]
    max_i = max(int_sorted) if int_sorted else 1.0
    if max_i <= 0:
        max_i = 1.0
    norm = [i / max_i for i in int_sorted]
    return _PeakList(pos_sorted, int_sorted, norm)


def _empty_result(algorithm: str, exp_pl: _PeakList, sim_pl: _PeakList, why: str) -> dict[str, Any]:
    score_key = "figure_of_merit" if algorithm == "hanawalt" else "cost"
    return {
        "algorithm": algorithm,
        score_key: 0.0 if algorithm == "hanawalt" else float("inf"),
        "verdict": "reject",
        "matched_peaks": [],
        "unmatched_exp": list(range(exp_pl.n)),
        "fitted_shift": 0.0,
        "fitted_scale": 1.0,
        "n_exp_peaks": exp_pl.n,
        "n_sim_peaks": sim_pl.n,
        "note": why,
    }


# ---------------------------------------------------------------------------
# Hanawalt search-match
# ---------------------------------------------------------------------------

def _apply_lattice_scale(positions, scale: float):
    """Re-position peaks for a lattice-parameter scale `a` via Bragg's law.

    A uniform lattice scaling multiplies every d-spacing, so sin(theta) scales:
    sin(theta_scaled) = a * sin(theta_orig). This is the physically-correct
    transform for a DFT-relaxed reference cell (typically a few % larger than
    the experimental cell) — NOT a constant 2-theta shift, which is wrong away
    from low angle."""
    positions = np.asarray(positions, dtype=float)
    sin_scaled = scale * np.sin(np.radians(positions / 2.0))
    sin_scaled = np.clip(sin_scaled, -1.0, 1.0)
    return 2.0 * np.degrees(np.arcsin(sin_scaled))


def _scaled_peaklist(pl: "_PeakList", scale: float) -> "_PeakList":
    """A copy of a peak list with positions re-scaled for a lattice parameter."""
    return _PeakList(
        positions=[float(p) for p in _apply_lattice_scale(pl.positions, scale)],
        intensities=list(pl.intensities),
        intensities_norm=list(pl.intensities_norm),
    )


def _fit_lattice_scale(exp_pl, sim_pl, tol_deg, scale_search, min_matches: int = 2,
                       weight_by: str = "exp") -> float:
    """Find the single lattice scale that best aligns the simulated peaks to the
    experimental ones (intensity-weighted matched coverage). Bounded to a few %
    so a wrong phase can't be force-aligned.

    A non-unity scale is only adopted if it aligns at least ``min_matches`` peaks
    — a real DFT-relaxed phase snaps *multiple* reflections into place at the
    right scale, whereas a wrong phase produces at most a lone coincidental
    alignment that must not be allowed to drive the scale (which would erode
    discrimination). Falls back to 1.0 (no scaling) otherwise.

    ``weight_by`` — which intensity weights a match. ``'exp'`` (single-phase
    default: downweights sim lines landing on weak noise peaks). ``'sim'`` for
    the MIXTURE per-phase pre-fit: a minority phase's true lines are weak in
    the mixture, so exp-weighting lets a few strong lines of the MAJORITY
    phase pull the minority's scale off its own exact matches (observed: NaCl
    in a 70/30 Si mix stretched 3.8% onto Si lines, deactivating it in the
    joint MILP). The candidate's own relative intensities are
    fraction-independent — the right anchor when other phases share the
    pattern."""
    lo, hi, step = scale_search
    scales = np.arange(lo, hi + step / 2.0, step)
    sim_pos = np.asarray(sim_pl.positions)
    exp_pos = np.asarray(exp_pl.positions)
    exp_w = np.sqrt(np.asarray(exp_pl.intensities_norm))
    sim_w = np.sqrt(np.asarray(sim_pl.intensities_norm))

    def _eval(a: float):
        sp = _apply_lattice_scale(sim_pos, a)
        used: set[int] = set()
        score = 0.0
        n = 0
        for ep, w in zip(exp_pos, exp_w):
            res = np.abs(sp - ep)
            for j in used:
                res[j] = np.inf
            j = int(np.argmin(res))
            if res[j] <= tol_deg:
                used.add(j)
                n += 1
                weight = w if weight_by == "exp" else float(sim_w[j])
                score += weight * (1.0 - res[j] / tol_deg)
        return score, n

    base_score, _ = _eval(1.0)
    best_scale, best_score = 1.0, base_score
    for a in scales:
        score, n = _eval(float(a))
        if n < min_matches:
            continue  # a lone coincidental alignment must not select a scale
        if score > best_score + 1e-9 or (
            abs(score - best_score) <= 1e-9 and abs(a - 1.0) < abs(best_scale - 1.0)
        ):
            best_score, best_scale = score, float(a)
    return best_scale


def _weighted_cost(exp_pl: "_PeakList", unmatched_exp, total_residual: float, tol_deg: float) -> float:
    """Intensity-weighted joint cost in [0, 1] (figure_of_merit = 1 - cost).
    Penalises the UNEXPLAINED INTENSITY fraction (so weak noise peaks barely
    count, while a missing strong reflection still drives cost up) plus the mean
    matched-peak residual. Replaces the old count-based form where each unmatched
    peak — noise or not — cost a full tolerance unit, which rejected otherwise-
    correct matches whenever the extractor returned a handful of noise peaks."""
    total_int = sum(exp_pl.intensities)
    if total_int > 0:
        unmatched_int = sum(exp_pl.intensities[i] for i in unmatched_exp)
        uncovered = unmatched_int / total_int
    else:
        uncovered = len(unmatched_exp) / max(exp_pl.n, 1)
    n_matched = exp_pl.n - len(unmatched_exp)
    mean_res = (total_residual / n_matched) / tol_deg if n_matched > 0 else 1.0
    return float(_COST_UNCOVERED_W * uncovered + _COST_RESIDUAL_W * min(mean_res, 1.0))


def _score_hanawalt(
    exp_pl: _PeakList,
    sim_pl: _PeakList,
    *,
    tol_deg: float,
    scale_search: tuple | None = None,
) -> dict[str, Any]:
    """Classical Hanawalt-style figure-of-merit with intensity weighting.

    When ``scale_search`` is given, a single lattice scale is fit first so a
    DFT-relaxed reference cell (peaks shifted a few % in sin-theta) is aligned
    to the experimental pattern before matching — otherwise such a reference is
    falsely rejected even when the phase is clearly present."""
    fitted_scale = (
        _fit_lattice_scale(exp_pl, sim_pl, tol_deg, scale_search)
        if scale_search else 1.0
    )
    sim_positions = _apply_lattice_scale(sim_pl.positions, fitted_scale)
    sim_norm = np.asarray(sim_pl.intensities_norm)
    used_sim = set()
    matched_peaks = []
    unmatched_exp = []
    matched_exp_intensities: list[float] = []

    pos_scores = []
    int_scores = []
    weights = []

    for i, (ep, ei_norm, ei_abs) in enumerate(
        zip(exp_pl.positions, exp_pl.intensities_norm, exp_pl.intensities)
    ):
        residuals = np.abs(sim_positions - ep)
        # Skip already-used sim peaks
        for j_used in used_sim:
            residuals[j_used] = np.inf
        j = int(np.argmin(residuals))
        if residuals[j] > tol_deg:
            unmatched_exp.append(i)
            continue
        used_sim.add(j)
        pos_score = 1.0 - residuals[j] / tol_deg
        int_score = max(0.0, 1.0 - abs(ei_norm - float(sim_norm[j])))
        weight = math.sqrt(ei_norm)  # weight by experimental intensity
        matched_peaks.append({
            "exp_idx": i,
            "sim_idx": j,
            "exp_pos": ep,
            "sim_pos": float(sim_positions[j]),
            "residual_deg": float(residuals[j]),
        })
        matched_exp_intensities.append(float(ei_abs))
        pos_scores.append(pos_score * weight)
        int_scores.append(int_score * weight)
        weights.append(weight)

    # Intensity-weighted coverage: matching the strong peaks matters more than
    # matching weak noise spikes that extract_peaks may have picked up. The
    # original unweighted (matched / total) form penalized clean spectra with
    # any noise peaks even when every meaningful peak was matched.
    total_intensity = sum(exp_pl.intensities)
    matched_intensity = sum(matched_exp_intensities)
    if total_intensity > 0:
        coverage = matched_intensity / total_intensity
    elif exp_pl.n > 0:
        coverage = len(matched_peaks) / exp_pl.n
    else:
        coverage = 0.0

    if weights:
        mean_pos = sum(pos_scores) / sum(weights)
        mean_int = sum(int_scores) / sum(weights)
    else:
        mean_pos = 0.0
        mean_int = 0.0

    fom = 0.55 * coverage + 0.35 * mean_pos + 0.10 * mean_int

    # Bidirectional gate: of THIS phase's strong reflections, how many appear in
    # the data? A real phase shows nearly all of them (gate ~1.0); a coincidental
    # false match leaves most of its own strong peaks unobserved -> gate < 1 ->
    # FoM cut, so a wrong-chemistry phase can no longer score "marginal" off a
    # few overlapping peaks.
    imax = max(sim_pl.intensities) if sim_pl.intensities else 0.0
    if imax > 0:
        strong_total = sum(
            sim_pl.intensities[j] / imax for j in range(sim_pl.n)
            if sim_pl.intensities[j] / imax >= _STRONG_SIM_FRAC
        )
        strong_matched = sum(
            sim_pl.intensities[j] / imax for j in used_sim
            if sim_pl.intensities[j] / imax >= _STRONG_SIM_FRAC
        )
        predicted_coverage = strong_matched / strong_total if strong_total > 0 else 1.0
    else:
        predicted_coverage = 1.0
    fom = fom * min(1.0, predicted_coverage / _PRED_COV_FULL)

    if fom >= _HANAWALT_ACCEPT_MIN:
        verdict = "accept"
    elif fom >= _HANAWALT_MARGINAL_MIN:
        verdict = "marginal"
    else:
        verdict = "reject"

    return {
        "algorithm": "hanawalt",
        "figure_of_merit": float(fom),
        "coverage": float(coverage),
        "predicted_coverage": float(predicted_coverage),
        "position_score": float(mean_pos),
        "intensity_score": float(mean_int),
        "verdict": verdict,
        "matched_peaks": matched_peaks,
        "unmatched_exp": unmatched_exp,
        "fitted_shift": 0.0,
        "fitted_scale": float(fitted_scale),
        "n_exp_peaks": exp_pl.n,
        "n_sim_peaks": sim_pl.n,
    }


# ---------------------------------------------------------------------------
# MIP peak-matching
# ---------------------------------------------------------------------------

def _score_mip(
    exp_pl: _PeakList,
    sim_pl: _PeakList,
    *,
    tol_deg: float,
    scale_search: tuple,
    shift_search: tuple,
) -> dict[str, Any]:
    """Joint shift + scale + assignment via MILP, gridded over scale."""
    if not PULP_AVAILABLE:
        raise RuntimeError(
            "MIP algorithm requires pulp; install via "
            "'pip install scilink[structure-matching]'"
        )

    scales = _scale_grid(scale_search)
    shift_lo, shift_hi = float(shift_search[0]), float(shift_search[1])

    best = None
    for scale in scales:
        result = _solve_mip_for_scale(
            exp_pl, sim_pl,
            scale=scale,
            tol_deg=tol_deg,
            shift_lo=shift_lo,
            shift_hi=shift_hi,
        )
        if result is None:
            continue
        if best is None or result["cost"] < best["cost"]:
            best = result
            best["fitted_scale"] = scale

    if best is None:
        return _empty_result("mip", exp_pl, sim_pl, "MILP found no feasible assignment")

    cost = best["cost"]
    if cost <= _MIP_ACCEPT_MAX:
        verdict = "accept"
    elif cost <= _MIP_MARGINAL_MAX:
        verdict = "marginal"
    else:
        verdict = "reject"

    return {
        "algorithm": "mip",
        "cost": float(cost),
        "verdict": verdict,
        "matched_peaks": best["matched_peaks"],
        "unmatched_exp": best["unmatched_exp"],
        "fitted_shift": float(best["fitted_shift"]),
        "fitted_scale": float(best["fitted_scale"]),
        "n_exp_peaks": exp_pl.n,
        "n_sim_peaks": sim_pl.n,
    }


def _scale_grid(scale_search: tuple) -> np.ndarray:
    lo, hi, step = float(scale_search[0]), float(scale_search[1]), float(scale_search[2])
    if step <= 0 or hi < lo:
        raise ValueError(f"Invalid scale_search: {scale_search!r}")
    return np.arange(lo, hi + step * 0.5, step)


def _solve_mip_for_scale(
    exp_pl: _PeakList,
    sim_pl: _PeakList,
    *,
    scale: float,
    tol_deg: float,
    shift_lo: float,
    shift_hi: float,
) -> Optional[dict[str, Any]]:
    """Solve one MILP at a fixed lattice scale; returns assignment + shift + cost.

    Two-pass approach to avoid a CBC interaction with the bilinear residual
    constraint ``r <= tol * x`` that produced bound-violating "solutions" in
    an earlier formulation.

    Pass 1 (MILP) — pure assignment with feasibility cone on shift:

        Variables:  x[i,j] in {0,1} — 1 if exp i matched to sim j
                    shift in [shift_lo, shift_hi] — zero-shift (degrees)
        Constraints:
            sum_j x[i,j] <= 1                          (each exp matched at most once)
            sum_i x[i,j] <= 1                          (each sim matched at most once)
            exp_i - scale*sim_j - shift <= tol + M(1 - x[i,j])
            -(exp_i - scale*sim_j - shift) <= tol + M(1 - x[i,j])
        Objective: maximize sum x[i,j]

    Pass 2 (post-solve scoring) — evaluate ``|exp_i - scale*sim_j - shift|``
    at the optimum for each matched pair; aggregate to a normalized cost in
    [0, 1] where every unmatched peak contributes a full tolerance unit.
    """
    nE = exp_pl.n
    nS = sim_pl.n
    if nE == 0 or nS == 0:
        return None

    # Pre-prune (i,j) pairs whose minimum possible residual exceeds tol for
    # any plausible shift — keeps the MILP small. A pair is feasible when
    # some shift in [shift_lo, shift_hi] reduces |exp_i - α·sim_j - shift|
    # below tol, equivalently raw = exp_i - α·sim_j lies in
    # [shift_lo - tol, shift_hi + tol].
    candidate_pairs: list[tuple[int, int]] = []
    raw_lookup: dict[tuple[int, int], float] = {}
    for i in range(nE):
        for j in range(nS):
            raw = exp_pl.positions[i] - scale * sim_pl.positions[j]
            if shift_lo - tol_deg <= raw <= shift_hi + tol_deg:
                candidate_pairs.append((i, j))
                raw_lookup[(i, j)] = raw
    if not candidate_pairs:
        # Nothing in this scale can match within tolerance under any shift.
        return {
            "cost": 1.0,
            "matched_peaks": [],
            "unmatched_exp": list(range(nE)),
            "fitted_shift": 0.0,
        }

    M = max(
        (max(exp_pl.positions) - min(exp_pl.positions)) if nE > 1 else 0.0,
        (max(sim_pl.positions) - min(sim_pl.positions)) if nS > 1 else 0.0,
        1.0,
    ) + max(abs(shift_lo), abs(shift_hi)) + tol_deg + 1.0

    prob = pulp.LpProblem("xrd_match", pulp.LpMaximize)
    x = {
        p: pulp.LpVariable(f"x_{p[0]}_{p[1]}", cat=pulp.LpBinary)
        for p in candidate_pairs
    }
    shift = pulp.LpVariable("shift", lowBound=shift_lo, upBound=shift_hi)

    # Objective: maximize the number of matched pairs. (Intensity-weighted
    # ties could be broken with a small bonus term; omitted for v1.)
    prob += pulp.lpSum(x.values())

    # Assignment constraints
    by_i: dict[int, list[tuple[int, int]]] = {i: [] for i in range(nE)}
    by_j: dict[int, list[tuple[int, int]]] = {j: [] for j in range(nS)}
    for p in candidate_pairs:
        by_i[p[0]].append(p)
        by_j[p[1]].append(p)
    for i, pairs in by_i.items():
        if pairs:
            prob += pulp.lpSum(x[p] for p in pairs) <= 1, f"once_per_exp_{i}"
    for j, pairs in by_j.items():
        if pairs:
            prob += pulp.lpSum(x[p] for p in pairs) <= 1, f"once_per_sim_{j}"

    # Tolerance cone: when x[i,j] = 1, |raw - shift| <= tol.
    for p in candidate_pairs:
        raw = raw_lookup[p]
        prob += raw - shift <= tol_deg + M * (1 - x[p]), f"tol_pos_{p[0]}_{p[1]}"
        prob += -(raw - shift) <= tol_deg + M * (1 - x[p]), f"tol_neg_{p[0]}_{p[1]}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        _logger.debug("MILP non-optimal at scale=%.5f: %s", scale, pulp.LpStatus[status])
        return None

    # Pass 2: with the assignment fixed, refine shift analytically. The shift
    # that minimizes Σ |raw - shift| over the matched pairs is the L1 median
    # (clipped to [shift_lo, shift_hi]). The MILP objective alone doesn't
    # discriminate among shifts inside the tolerance cone — any feasible
    # shift gives the same match count — so CBC may return a corner-of-the-
    # cone shift that inflates the cost. The median-refinement step picks
    # the best representative.
    matched_raws = [
        raw_lookup[p] for p in candidate_pairs
        if pulp.value(x[p]) is not None and pulp.value(x[p]) >= 0.5
    ]
    if matched_raws:
        median_shift = float(np.median(matched_raws))
        fitted_shift = float(np.clip(median_shift, shift_lo, shift_hi))
    else:
        fitted_shift = float(pulp.value(shift)) if pulp.value(shift) is not None else 0.0

    matched_peaks: list[dict[str, Any]] = []
    matched_exp_set: set[int] = set()
    total_residual = 0.0
    for p in candidate_pairs:
        val = pulp.value(x[p])
        if val is None or val < 0.5:
            continue
        i, j = p
        residual = abs(raw_lookup[p] - fitted_shift)
        # If median-refined shift pushed a marginal match outside the cone,
        # drop it from the matched set.
        if residual > tol_deg + 1e-9:
            continue
        matched_exp_set.add(i)
        total_residual += residual
        matched_peaks.append({
            "exp_idx": i,
            "sim_idx": j,
            "exp_pos": exp_pl.positions[i],
            "sim_pos": float(scale * sim_pl.positions[j] + fitted_shift),
            "residual_deg": float(residual),
        })
    unmatched_exp = [i for i in range(nE) if i not in matched_exp_set]
    # Intensity-weighted cost (see _weighted_cost): unexplained INTENSITY plus
    # mean residual, so noise peaks don't reject a correct match.
    cost = _weighted_cost(exp_pl, unmatched_exp, total_residual, tol_deg)

    return {
        "cost": float(cost),
        "matched_peaks": matched_peaks,
        "unmatched_exp": unmatched_exp,
        "fitted_shift": fitted_shift,
    }


# ---------------------------------------------------------------------------
# Joint multi-phase MIP — assigns experimental peaks across N candidate phases
# ---------------------------------------------------------------------------

TOOL_SPEC_MULTIPHASE = ToolSpec(
    name="score_xrd_match_multiphase",
    description=(
        "Joint multi-phase scoring: given an experimental peak list and "
        "N candidate phase patterns, solves one MILP that assigns each "
        "experimental peak to at most one (phase, simulated peak) pair, "
        "with per-phase activation binaries that let the solver leave "
        "phases out entirely. Returns per-phase coverage and an overall "
        "joint cost. Use for suspected mixtures; for single-phase "
        "identification, score_xrd_match_robust with algorithm='hanawalt' "
        "or 'mip' remains the right tool."
    ),
    import_line="from scilink.skills.structure_matching.xrd.score_match_robust import score_xrd_match_multiphase",
    signature=(
        "score_xrd_match_multiphase(exp_peaks, candidates, tol_deg=0.3, "
        "scale_search=(0.99, 1.01, 0.005), shift_search=(-0.4, 0.4), "
        "max_exp_peaks=30) -> dict"
    ),
    parameters={
        "exp_peaks": {
            "type": "dict",
            "description": "Pre-extracted experimental peak list from extract_peaks (positions + intensities).",
        },
        "candidates": {
            "type": "list[dict]",
            "description": (
                "List of candidate phases. Each entry: {'id': str, "
                "'formula': str, 'sim_two_theta': list[float], "
                "'sim_intensity': list[float]}. The same shape "
                "simulate_xrd_pattern returns, plus an id and formula."
            ),
        },
        "tol_deg": {
            "type": "float",
            "description": "Position tolerance for matching peaks (degrees). Default 0.3.",
        },
        "scale_search": {
            "type": "tuple",
            "description": "(min, max, step) shared lattice-scale grid. Default (0.99, 1.01, 0.005) — coarser than single-phase MIP to keep multi-phase MILP runtime bounded.",
        },
        "shift_search": {
            "type": "tuple",
            "description": "(min, max) zero-shift bounds (degrees). Shared across phases (one instrument, one zero-shift). Default (-0.4, 0.4).",
        },
        "max_exp_peaks": {
            "type": "int",
            "description": "Cap on experimental peaks considered (strongest kept). Default 30 (higher than single-phase since the assignment problem allocates across phases).",
        },
        "lattice_scale_search": {
            "type": "tuple",
            "description": "(min, max, step) PER-PHASE lattice-scale grid (default (0.96, 1.04, 0.002), ±4%). Each phase is aligned independently before the joint assignment, because lattice mismatch is per-phase — e.g. a DFT-relaxed metal (+2%) and an experimental molecular reference (0%) in the same pattern need different scales. This is separate from scale_search/shift_search, which absorb the SHARED instrument zero-shift.",
        },
        "fit_lattice_scale": {
            "type": "bool",
            "description": "Default True: fit a per-phase lattice scale before the joint assignment (adopted only if it aligns >=2 reflections). Set False to match each phase at its reference lattice as-is.",
        },
    },
    required=["exp_peaks", "candidates"],
    returns=(
        "dict with 'algorithm' ('mip_multiphase'), 'cost' (overall joint "
        "cost in [0, 1], lower better), 'figure_of_merit' (1 - cost, "
        "higher-better mirror so the same quality gate as the single-phase "
        "scorer reads a multi-phase result), 'verdict', 'active_phases' (list of {id, "
        "formula, coverage, matched_peaks, mean_residual_deg, lattice_scale "
        "— the per-phase fitted lattice scale, ~1.02 for a DFT-relaxed metal}), "
        "'unmatched_exp' (peak indices not explained by any phase), "
        "'fitted_shift', 'fitted_scale' (shared instrument terms), "
        "'n_exp_peaks', 'n_phases_considered'."
    ),
    when_to_use=(
        "Suspected mixtures (system_info / notes mention 'mixture', "
        "'multi-phase', 'two-phase', or chemistry_hint contains two "
        "non-compound elements). For confirmed single-phase, "
        "score_xrd_match_robust per candidate is faster and sufficient."
    ),
)
TOOL_SPECS = [TOOL_SPEC_MULTIPHASE]


def score_xrd_match_multiphase(
    exp_peaks: dict,
    candidates: list[dict],
    *,
    tol_deg: float = 0.3,
    scale_search: tuple = (0.99, 1.01, 0.005),
    shift_search: tuple = (-0.4, 0.4),
    max_exp_peaks: int = 30,
    lattice_scale_search: tuple = (0.96, 1.04, 0.002),
    fit_lattice_scale: bool = True,
) -> dict[str, Any]:
    """Joint multi-phase MIP. See ``TOOL_SPEC_MULTIPHASE`` for full contract."""
    if not PULP_AVAILABLE:
        raise RuntimeError(
            "score_xrd_match_multiphase requires pulp; install via "
            "'pip install scilink[structure-matching]'"
        )
    if not candidates:
        raise ValueError("candidates must contain at least one phase")

    exp_pl = _to_peak_list(
        list(exp_peaks.get("positions", [])),
        list(exp_peaks.get("intensities", [])),
        max_exp_peaks,
    )
    if exp_pl.n == 0:
        return _empty_multiphase_result(candidates, "no experimental peaks")

    sim_pls: list[_PeakList] = []
    for cand in candidates:
        sim_pls.append(_to_peak_list(
            list(cand.get("sim_two_theta", [])),
            list(cand.get("sim_intensity", [])),
            30,
        ))
    if all(pl.n == 0 for pl in sim_pls):
        return _empty_multiphase_result(candidates, "no simulated peaks")

    # Lattice mismatch is PER-PHASE (a DFT-relaxed metal and an experimental
    # molecular reference in the same pattern need different scales), so align
    # each phase independently before the joint assignment. The MILP's shared
    # scale/shift below then only absorbs the common instrument zero-shift, which
    # genuinely IS shared (one detector). Without this, a single shared scale
    # can align at most one phase of a mixture.
    per_phase_scale = [1.0] * len(sim_pls)
    if fit_lattice_scale:
        for i, pl in enumerate(sim_pls):
            if pl.n:
                # weight_by='sim': in a mixture the exp-intensity weighting
                # would pull a minority phase's scale onto the majority
                # phase's strong lines (see _fit_lattice_scale docstring).
                a = _fit_lattice_scale(exp_pl, pl, tol_deg,
                                       lattice_scale_search, weight_by="sim")
                per_phase_scale[i] = a
                sim_pls[i] = _scaled_peaklist(pl, a)

    scales = _scale_grid(scale_search)
    shift_lo, shift_hi = float(shift_search[0]), float(shift_search[1])

    best = None
    for scale in scales:
        result = _solve_multiphase_mip(
            exp_pl, sim_pls,
            scale=scale,
            tol_deg=tol_deg,
            shift_lo=shift_lo,
            shift_hi=shift_hi,
        )
        if result is None:
            continue
        if best is None or result["cost"] < best["cost"]:
            best = result
            best["fitted_scale"] = scale

    if best is None:
        return _empty_multiphase_result(candidates, "no feasible joint assignment")

    cost = best["cost"]
    if cost <= _MIP_ACCEPT_MAX:
        verdict = "accept"
    elif cost <= _MIP_MARGINAL_MAX:
        verdict = "marginal"
    else:
        verdict = "reject"

    active_phases = []
    for p_idx, cand in enumerate(candidates):
        per_phase = best["per_phase"][p_idx]
        if not per_phase["active"]:
            continue
        active_phases.append({
            "id": cand.get("id", str(p_idx)),
            "formula": cand.get("formula", ""),
            "coverage": per_phase["coverage"],
            "predicted_coverage": per_phase.get("predicted_coverage", 1.0),
            "matched_peaks": per_phase["matched_peaks"],
            "mean_residual_deg": per_phase["mean_residual_deg"],
            "lattice_scale": float(per_phase_scale[p_idx]),
        })

    return {
        "algorithm": "mip_multiphase",
        "cost": float(cost),
        # Higher-is-better mirror of `cost`, so the same quality gate the
        # single-phase scorer feeds (metric='figure_of_merit') can read a
        # multi-phase result. cost in [0,1] (lower better) -> fom = 1 - cost;
        # the multi-phase accept (cost <= 0.25) maps to fom >= 0.75, above the
        # 0.70 gate. The authoritative accept/marginal/reject is `verdict`.
        "figure_of_merit": float(max(0.0, 1.0 - cost)),
        "verdict": verdict,
        "active_phases": active_phases,
        "unmatched_exp": best["unmatched_exp"],
        "fitted_shift": float(best["fitted_shift"]),
        "fitted_scale": float(best["fitted_scale"]),
        "n_exp_peaks": exp_pl.n,
        "n_phases_considered": len(candidates),
    }


def _empty_multiphase_result(candidates: list[dict], why: str) -> dict[str, Any]:
    return {
        "algorithm": "mip_multiphase",
        "cost": float("inf"),
        "figure_of_merit": 0.0,
        "verdict": "reject",
        "active_phases": [],
        "unmatched_exp": [],
        "fitted_shift": 0.0,
        "fitted_scale": 1.0,
        "n_exp_peaks": 0,
        "n_phases_considered": len(candidates),
        "note": why,
    }


def _strong_sim_weights(sim_pls: list["_PeakList"]) -> dict[int, list[tuple[int, float]]]:
    """Per phase, the (sim peak index, relative-intensity weight) list for peaks
    at >= _STRONG_SIM_FRAC of that phase's maximum intensity — the strong
    reflections a real phase must show, used by the predicted-but-absent
    penalty. Weak peaks (the long tail, often below detection) are excluded."""
    out: dict[int, list[tuple[int, float]]] = {}
    for p_idx, pl in enumerate(sim_pls):
        items: list[tuple[int, float]] = []
        if pl.n and pl.intensities:
            imax = max(pl.intensities)
            if imax > 0:
                for j in range(pl.n):
                    w = pl.intensities[j] / imax
                    if w >= _STRONG_SIM_FRAC:
                        items.append((j, float(w)))
        out[p_idx] = items
    return out


def _solve_multiphase_mip(
    exp_pl: _PeakList,
    sim_pls: list[_PeakList],
    *,
    scale: float,
    tol_deg: float,
    shift_lo: float,
    shift_hi: float,
) -> Optional[dict[str, Any]]:
    """One MILP across N phases at a fixed shared scale.

    Variables:
        x[i, j, p] in {0, 1} — exp peak i matched to sim peak j of phase p.
        y[p]      in {0, 1} — phase p activated.
        shift     in [shift_lo, shift_hi] — shared zero-shift (one
                                              diffractometer, one zero).

    Constraints:
        Σ_{j, p} x[i, j, p] ≤ 1                    (exp i matched at most once)
        Σ_i x[i, j, p] ≤ 1   ∀ p, j                (each sim peak used once)
        x[i, j, p] ≤ y[p]                          (phase active when used)
        |exp_i - scale·sim_{j,p} - shift| ≤ tol + M(1 − x[i, j, p])

    Objective:
        maximize  Σ x[i, j, p] − activation_penalty · Σ y[p]
    """
    nE = exp_pl.n
    nP = len(sim_pls)
    if nE == 0 or nP == 0:
        return None

    candidate_triples: list[tuple[int, int, int]] = []
    raw_lookup: dict[tuple[int, int, int], float] = {}
    for p_idx, sim_pl in enumerate(sim_pls):
        for i in range(nE):
            for j in range(sim_pl.n):
                raw = exp_pl.positions[i] - scale * sim_pl.positions[j]
                if shift_lo - tol_deg <= raw <= shift_hi + tol_deg:
                    candidate_triples.append((i, j, p_idx))
                    raw_lookup[(i, j, p_idx)] = raw
    if not candidate_triples:
        return {
            "cost": 1.0,
            "per_phase": [
                {"active": False, "coverage": 0.0, "matched_peaks": [], "mean_residual_deg": 0.0}
                for _ in sim_pls
            ],
            "unmatched_exp": list(range(nE)),
            "fitted_shift": 0.0,
        }

    M = max(
        (max(exp_pl.positions) - min(exp_pl.positions)) if nE > 1 else 0.0,
        max(
            (max(pl.positions) - min(pl.positions)) if pl.n > 1 else 0.0
            for pl in sim_pls
        ),
        1.0,
    ) + max(abs(shift_lo), abs(shift_hi)) + tol_deg + 1.0

    prob = pulp.LpProblem("xrd_multiphase", pulp.LpMaximize)
    x = {
        t: pulp.LpVariable(f"x_{t[0]}_{t[1]}_{t[2]}", cat=pulp.LpBinary)
        for t in candidate_triples
    }
    y = {
        p_idx: pulp.LpVariable(f"y_{p_idx}", cat=pulp.LpBinary)
        for p_idx in range(nP)
    }
    shift = pulp.LpVariable("shift", lowBound=shift_lo, upBound=shift_hi)

    # Constraint groupings (built before the objective so the predicted-absent
    # penalty can reference per-(sim-peak, phase) match sums).
    by_i: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(nE)}
    by_jp: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    by_p: dict[int, list[tuple[int, int, int]]] = {p_idx: [] for p_idx in range(nP)}
    for t in candidate_triples:
        i, j, p_idx = t
        by_i[i].append(t)
        by_jp.setdefault((j, p_idx), []).append(t)
        by_p[p_idx].append(t)

    # Strong predicted peaks per phase (relative intensity >= _STRONG_SIM_FRAC),
    # weighted by relative intensity — the reflections a real phase must show.
    strong_sim = _strong_sim_weights(sim_pls)

    # Objective: maximize matched exp peaks, minus per-phase activation cost,
    # minus the predicted-but-absent penalty (BIDIRECTIONAL matching). For an
    # active phase (y=1) a strong predicted peak j contributes its weight unless
    # it is matched (Σ_i x[i,j,p] = 1); an inactive phase contributes nothing
    # (y=0 forces every x[*,j,p]=0, so y - Σx = 0). This is linear in x, y.
    absent_penalty = pulp.lpSum(
        w * (y[p_idx] - pulp.lpSum(x[t] for t in by_jp.get((j, p_idx), [])))
        for p_idx, js in strong_sim.items()
        for j, w in js
    )
    prob += (
        pulp.lpSum(x.values())
        - _MULTIPHASE_ACTIVATION_PENALTY * pulp.lpSum(y.values())
        - _MULTIPHASE_ABSENT_PENALTY * absent_penalty
    )

    for i, triples in by_i.items():
        if triples:
            prob += pulp.lpSum(x[t] for t in triples) <= 1, f"once_per_exp_{i}"
    for (j, p_idx), triples in by_jp.items():
        if triples:
            prob += pulp.lpSum(x[t] for t in triples) <= 1, f"once_per_sim_{p_idx}_{j}"
    for p_idx, triples in by_p.items():
        for t in triples:
            prob += x[t] <= y[p_idx], f"requires_active_{t[0]}_{t[1]}_{t[2]}"

    # Tolerance cone
    for t in candidate_triples:
        raw = raw_lookup[t]
        prob += raw - shift <= tol_deg + M * (1 - x[t]), f"tol_pos_{t[0]}_{t[1]}_{t[2]}"
        prob += -(raw - shift) <= tol_deg + M * (1 - x[t]), f"tol_neg_{t[0]}_{t[1]}_{t[2]}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        _logger.debug("Multiphase MILP non-optimal at scale=%.5f: %s", scale, pulp.LpStatus[status])
        return None

    # Median-shift refinement over the matched triples (same rationale as
    # single-phase MIP — CBC may pick a corner-of-the-cone shift).
    matched_raws = [
        raw_lookup[t] for t in candidate_triples
        if pulp.value(x[t]) is not None and pulp.value(x[t]) >= 0.5
    ]
    if matched_raws:
        median_shift = float(np.median(matched_raws))
        fitted_shift = float(np.clip(median_shift, shift_lo, shift_hi))
    else:
        fitted_shift = float(pulp.value(shift)) if pulp.value(shift) is not None else 0.0

    per_phase: list[dict[str, Any]] = []
    matched_exp_set: set[int] = set()
    total_residual = 0.0
    total_matched = 0
    for p_idx, sim_pl in enumerate(sim_pls):
        active = pulp.value(y[p_idx]) is not None and pulp.value(y[p_idx]) >= 0.5
        matched_for_phase: list[dict[str, Any]] = []
        phase_matched_exp: set[int] = set()
        phase_matched_sim: set[int] = set()
        phase_residuals: list[float] = []
        for t in by_p[p_idx]:
            val = pulp.value(x[t])
            if val is None or val < 0.5:
                continue
            i, j, _ = t
            residual = abs(raw_lookup[t] - fitted_shift)
            if residual > tol_deg + 1e-9:
                continue
            matched_for_phase.append({
                "exp_idx": i,
                "sim_idx": j,
                "exp_pos": exp_pl.positions[i],
                "sim_pos": float(scale * sim_pl.positions[j] + fitted_shift),
                "residual_deg": float(residual),
            })
            phase_matched_exp.add(i)
            phase_matched_sim.add(j)
            phase_residuals.append(residual)
            matched_exp_set.add(i)
            total_residual += residual
            total_matched += 1
        coverage = (
            sum(exp_pl.intensities[i] for i in phase_matched_exp) / sum(exp_pl.intensities)
            if sum(exp_pl.intensities) > 0 else 0.0
        )
        # Predicted-coverage: of this phase's STRONG predicted peaks, the
        # intensity-weighted fraction actually observed. Low predicted-coverage
        # flags an over-predicting (Magnéli-type) false match.
        strong = strong_sim.get(p_idx, [])
        strong_total = sum(w for _, w in strong)
        absent_w = sum(w for j, w in strong if j not in phase_matched_sim)
        predicted_coverage = (1.0 - absent_w / strong_total) if strong_total > 0 else 1.0
        per_phase.append({
            "active": bool(active and matched_for_phase),
            "coverage": float(coverage),
            "predicted_coverage": float(predicted_coverage),
            "absent_weight": float(absent_w),
            "matched_peaks": matched_for_phase,
            "mean_residual_deg": float(np.mean(phase_residuals)) if phase_residuals else 0.0,
        })

    unmatched_exp = [i for i in range(nE) if i not in matched_exp_set]
    # Intensity-weighted cost (see _weighted_cost): unexplained INTENSITY plus
    # mean residual, so the handful of weak noise peaks the extractor returns
    # no longer rejects an otherwise-correct multi-phase match. The predicted-
    # but-absent penalty governs SELECTION in the MILP OBJECTIVE (not the cost);
    # `predicted_coverage` is surfaced per phase so an over-predicting phase that
    # slips through is still visible downstream.
    cost = _weighted_cost(exp_pl, unmatched_exp, total_residual, tol_deg)

    return {
        "cost": float(cost),
        "per_phase": per_phase,
        "unmatched_exp": unmatched_exp,
        "fitted_shift": fitted_shift,
    }
