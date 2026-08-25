"""``identify_mixture`` tool — multi-phase identification by sequential subtraction.

The classical search-match loop for a mixture pattern: identify the dominant
phase by fingerprint search, remove the lines it accounts for, re-search the
residual, repeat. In-situ series are *by definition* two-phase during any
transformation, so this is the identification workhorse for operando data.

Two failure modes of naive subtraction are handled explicitly:

* **Over-removal of shared lines.** Phases in a real mixture share peak
  positions (e.g. a cubic line coinciding with a tetragonal one). Removing
  every measured peak within tolerance of the accepted phase would delete the
  second phase's evidence. Instead the accepted phase's *predicted intensity*
  (scaled to the measurement on its matched lines) is subtracted from each
  matched peak; a peak keeps its unexplained remainder when that remainder is
  a substantial fraction of the peak (``shared_line_floor``).
* **Greedy misassignment.** Sequential acceptance is greedy; a line grabbed
  by phase 1 may belong to phase 2. The accepted ensemble is therefore
  re-scored jointly with ``score_xrd_match_multiphase`` (one MILP across all
  phases, per-phase activation) — its verdict and per-phase coverage are the
  authoritative confirmation, the sequential FOMs are the discovery path.

The per-phase ``intensity_share`` is a *screening* proxy for abundance
(accounted measured intensity, normalized) — scattering power differs between
phases, so it is NOT a weight fraction. Rietveld on the identified ensemble
is the quantitative step."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


def _d_to_tt(d: np.ndarray, lam: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        s = lam / (2.0 * np.asarray(d, dtype=float))
    s = np.clip(s, -1.0, 1.0)
    return 2.0 * np.degrees(np.arcsin(s))


TOOL_SPEC = ToolSpec(
    name="identify_mixture",
    description=(
        "Identify ALL phases in a multi-phase powder pattern by sequential "
        "subtraction over the fingerprint library: search-match the peak list, "
        "accept the best phase, subtract its scaled predicted intensities from "
        "the matched peaks (shared lines keep their unexplained remainder), "
        "re-search the residual, repeat; then confirm the whole ensemble with "
        "one joint multi-phase MILP (score_xrd_match_multiphase). THE blind "
        "route for suspected mixtures and for in-situ / operando frames, which "
        "are two-phase during any transformation. Deterministic, offline, no "
        "chemistry needed. Requires the fingerprint library "
        "(scilink fetch-xrd-library)."
    ),
    import_line=("from scilink.skills.structure_matching.xrd.identify_mixture "
                 "import identify_mixture"),
    signature=(
        "identify_mixture(two_theta, intensity, wavelength='CuKa', "
        "library_path=None, max_phases=3, fom_threshold=0.5, tol_deg=0.3, "
        "min_residual_peaks=4, shared_line_floor=0.3, absent_tiebreak_band=0.1, "
        "n_key_lines=3, n_query_lines=10, confirm=True, "
        "materialize_dir='./fp_matches') -> dict"
    ),
    parameters={
        "two_theta": {"type": "list[float]", "description": "Measured peak positions (2θ°) of the FULL mixture pattern — extract_peaks' 'positions'. Feed ALL confident peaks (lower extract_peaks' prominence_frac for mixtures — each phase's key lines hide among more peaks)."},
        "intensity": {"type": "list[float]", "description": "Peak intensities aligned with two_theta — extract_peaks' 'intensities'."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or Å — synchrotron wavelengths pass as the float. The library is wavelength-free (d-spacings)."},
        "library_path": {"type": "str", "description": "Fingerprint parquet (default resolution: $SCILINK_XRD_FINGERPRINT_DB, then the per-user store)."},
        "max_phases": {"type": "int", "description": "Iteration cap (default 3). RAISE for known complex mixtures; each extra phase costs one library search on the residual."},
        "fom_threshold": {"type": "float", "description": "Per-iteration acceptance: the residual's best figure_of_merit must reach this to accept another phase (default 0.5 — deliberately LAXER than the single-phase ≥0.7 rubric, because the FOM's dominant term is coverage of the CURRENT peak list, and a mixture's other phases depress it even for a correct match: the true dominant phase of an equal three-phase mix scores only ~0.6). The joint multiphase_confirmation is the guard against the lax gate. RAISE (e.g. 0.65) to avoid spurious trace phases when confirmation is unavailable; LOWER (e.g. 0.4) only to chase a weak minority phase, and then trust ONLY confirmation-active phases."},
        "tol_deg": {"type": "float", "description": "Position tolerance (2θ°) for search, line accounting, and subtraction (default 0.3). RAISE for uncalibrated/shifted patterns; LOWER for calibrated synchrotron data (sharper peaks — tighter tolerance sharpens the shared-line split)."},
        "min_residual_peaks": {"type": "int", "description": "Stop when fewer residual peaks remain (default 4; the search itself needs ≥3). RAISE to avoid identifying phases from noise stragglers."},
        "shared_line_floor": {"type": "float", "description": "Over-subtraction guard: a matched peak SURVIVES into the residual (with its unexplained remainder as intensity) when that remainder is ≥ this fraction of the peak (default 0.3). LOWER to keep more shared-line evidence for later phases (risk: the same phase's duplicate library entries re-match); RAISE toward 1.0 to approach hard removal (classical, but deletes coincident lines of the next phase)."},
        "absent_tiebreak_band": {"type": "float", "description": "Acceptance tie-break: among candidates within this FOM band of the residual's best, accept the one with the FEWEST absent strong predicted lines (default 0.1). This is the deterministic form of the single-phase near-tie rule — validated on real mixtures, where dense-line junk entries (organics/exotics whose line forests blanket any residual) out-FOM the true minority phase by ~0.05 while showing 0.1-0.25 of their own strong lines absent vs the truth's 0.0. WIDEN (e.g. 0.15) if a true phase keeps losing to junk that out-scores it by more; set 0 to disable (pure FOM order)."},
        "n_key_lines": {"type": "int", "description": "Hanawalt key lines required per candidate (default 3). LOWER to 2 for textured samples — passed through to each search."},
        "n_query_lines": {"type": "int", "description": "Query's strongest lines the key lines match against (default 10, higher than single-phase search's 8 — a mixture's phase key lines hide among more measured lines)."},
        "confirm": {"type": "bool", "description": "Run the joint multi-phase MILP confirmation over the accepted ensemble (default True; needs pulp). The MILP's verdict/coverage is the authoritative accept — a phase the joint solver leaves inactive or with low coverage is a subtraction artifact."},
        "materialize_dir": {"type": "str", "description": "Directory each accepted phase's full CIF is resolved into ('structure_path' — the bridge to simulation confirmation and multi-phase Rietveld)."},
    },
    required=["two_theta", "intensity"],
    returns=(
        "dict: 'phases' — accepted in discovery order, each {source_id, "
        "formula, space_group, cell, figure_of_merit (on ITS residual), "
        "iteration, n_lines_accounted, intensity_share (screening abundance "
        "proxy — accounted intensity fraction, NOT a weight fraction; use "
        "Rietveld for quantification), structure_path, confirmed (kept "
        "active by the joint MILP — treat confirmed=False as a subtraction "
        "artifact, not a component)}; 'residual_peaks' "
        "(positions/intensities left unaccounted — substantial strong "
        "residuals mean a phase not in the library: fall back to "
        "index_pattern on the residual list); 'residual_intensity_frac'; "
        "'stopped_because'; 'multiphase_confirmation' — the joint MILP "
        "result (verdict accept/marginal/reject, per-phase active/coverage, "
        "unmatched_exp) or None when confirm=False/pulp missing. KNOWN "
        "LIMIT: a sparse-line minority phase (few strong lines, e.g. cubic "
        "fluorite-types) can be shadowed by a dense-line position-degenerate "
        "library entry — when a phase is unconfirmed or confirmation leaves "
        "peaks unmatched, re-examine those peaks and apply chemistry "
        "plausibility before reporting."
    ),
    when_to_use=(
        "Any suspected multi-phase pattern with no chemistry hypothesis: "
        "'mixture' / 'two-phase' / 'co-existing phases' in the notes, an "
        "in-situ frame mid-transformation, or a single-phase search-match "
        "that leaves many strong measured peaks unmatched. For a KNOWN "
        "candidate list use score_xrd_match_multiphase directly; this tool "
        "is for discovering the candidates."
    ),
)


def identify_mixture(
    two_theta: Sequence[float],
    intensity: Sequence[float],
    wavelength: Any = "CuKa",
    library_path: Optional[str] = None,
    max_phases: int = 3,
    fom_threshold: float = 0.5,
    tol_deg: float = 0.3,
    min_residual_peaks: int = 4,
    shared_line_floor: float = 0.3,
    absent_tiebreak_band: float = 0.1,
    n_key_lines: int = 3,
    n_query_lines: int = 10,
    confirm: bool = True,
    materialize_dir: str = "./fp_matches",
) -> dict[str, Any]:
    """Sequential-subtraction mixture identification. See ``TOOL_SPEC``."""
    from .fingerprint import _load_library, _resolve_lam, search_match_pattern

    lam = _resolve_lam(wavelength)
    tt = np.asarray(two_theta, dtype=float)
    yy = np.asarray(intensity, dtype=float)
    if tt.size < 3:
        raise ValueError("identify_mixture needs at least 3 measured peaks.")
    if tt.size != yy.size:
        raise ValueError("two_theta and intensity must have the same length.")
    df = _load_library(library_path)
    total_intensity = float(yy.sum())

    res_tt, res_yy = tt.copy(), yy.copy()
    phases: list[dict] = []
    accepted_ids: set = set()
    stopped_because = f"max_phases ({int(max_phases)}) reached"

    for iteration in range(1, int(max_phases) + 1):
        if res_tt.size < int(min_residual_peaks):
            stopped_because = (f"{res_tt.size} residual peaks < "
                               f"min_residual_peaks ({int(min_residual_peaks)})")
            break
        r = search_match_pattern(
            res_tt.tolist(), res_yy.tolist(), wavelength=wavelength,
            library_path=library_path, tol_deg=float(tol_deg),
            n_key_lines=int(n_key_lines), n_query_lines=int(n_query_lines),
            materialize_top=0)
        cand = [m for m in r["matches"] if m["source_id"] not in accepted_ids]
        if not cand or cand[0]["figure_of_merit"] < float(fom_threshold):
            best = cand[0]["figure_of_merit"] if cand else 0.0
            stopped_because = (f"no convincing match on the residual (best "
                               f"fom {best:.2f} < threshold {float(fom_threshold):.2f})")
            break
        # Absent-strong-lines tie-break (deterministic form of the skill's
        # near-tie rule): dense-line junk entries out-FOM a true minority
        # phase by a few hundredths on a residual while leaving a visible
        # fraction of their own strong lines unmeasured; the true phase shows
        # ~0 absent. Among candidates within the band, fewest-absent wins.
        top_fom = cand[0]["figure_of_merit"]
        floor = max(top_fom - float(absent_tiebreak_band), float(fom_threshold))
        band = [m for m in cand if m["figure_of_merit"] >= floor]
        best = min(band, key=lambda m: (m.get("frac_strong_lines_absent") or 0.0,
                                        -m["figure_of_merit"]))
        row = df[df["source_id"] == best["source_id"]].iloc[0]
        ph_tt = _d_to_tt(np.asarray(row["ds"], dtype=float), lam)
        ph_ii = np.asarray(row["intensities"], dtype=float)
        fin = np.isfinite(ph_tt)
        ph_tt, ph_ii = ph_tt[fin], ph_ii[fin]

        # Subtract the accepted phase: scale its normalized predicted
        # intensities to the measurement via the median ratio over matched
        # lines (median — robust to preferred orientation distorting a few
        # lines), then remove each matched peak's explained part. min() caps
        # the accounted credit at the measured intensity so an
        # over-predicted line cannot claim intensity that is not there.
        dist = np.abs(res_tt[:, None] - ph_tt[None, :])
        nearest = dist.argmin(axis=1)
        matched = dist.min(axis=1) <= float(tol_deg)
        pred = ph_ii[nearest]
        ok = matched & (pred > 0)
        if not ok.any():
            stopped_because = (f"match {best['formula']} accounts for no "
                               "residual peaks (inconsistent scoring)")
            break
        scale = float(np.median(res_yy[ok] / pred[ok]))
        explained = np.where(matched, scale * pred, 0.0)
        accounted = float(np.minimum(explained, res_yy)[matched].sum())
        remainder = res_yy - explained
        keep = ~matched | (remainder >= float(shared_line_floor) * res_yy)

        phases.append({
            "source_id": best["source_id"],
            "formula": best["formula"],
            "space_group": best["space_group"],
            "cell": best["cell"],
            "figure_of_merit": best["figure_of_merit"],
            "iteration": iteration,
            "n_lines_accounted": int(matched.sum()),
            "intensity_share": round(accounted / max(total_intensity, 1e-9), 3),
            "cif_path": best.get("cif_path"),
        })
        accepted_ids.add(best["source_id"])
        res_tt = res_tt[keep]
        res_yy = np.where(matched, np.maximum(remainder, 0.0), res_yy)[keep]

    # Structure bridge for every ACCEPTED phase (same resolution as
    # search_match_pattern: local library CIF, else fetch by COD ID).
    for ph in phases:
        cif = ph.pop("cif_path", None)
        if cif and os.path.exists(str(cif)):
            ph["structure_path"] = str(cif)
            continue
        try:
            from .._backends.cod import fetch_cod_cif
            ph["structure_path"] = fetch_cod_cif(ph["source_id"], materialize_dir)
        except Exception as exc:                       # offline / non-COD id
            _logger.debug("materialize failed for %s: %s", ph["source_id"], exc)
            ph["structure_path"] = None

    confirmation = None
    if confirm and phases:
        try:
            from .score_match_robust import score_xrd_match_multiphase
            cands = []
            for ph in phases:
                row = df[df["source_id"] == ph["source_id"]].iloc[0]
                ctt = _d_to_tt(np.asarray(row["ds"], dtype=float), lam)
                cii = np.asarray(row["intensities"], dtype=float)
                fin = np.isfinite(ctt)
                cands.append({"id": ph["source_id"], "formula": ph["formula"],
                              "sim_two_theta": ctt[fin].tolist(),
                              "sim_intensity": cii[fin].tolist()})
            confirmation = score_xrd_match_multiphase(
                exp_peaks={"positions": tt.tolist(), "intensities": yy.tolist()},
                candidates=cands, tol_deg=float(tol_deg))
            active = {p["id"] for p in confirmation["active_phases"]}
            for ph in phases:
                ph["confirmed"] = ph["source_id"] in active
        except Exception as exc:
            _logger.warning("multiphase confirmation unavailable: %s", exc)
            confirmation = None

    return {
        "phases": phases,
        "n_phases": len(phases),
        "residual_peaks": {"positions": [round(float(v), 3) for v in res_tt],
                           "intensities": [round(float(v), 3) for v in res_yy]},
        "residual_intensity_frac": round(float(res_yy.sum())
                                         / max(total_intensity, 1e-9), 3),
        "stopped_because": stopped_because,
        "multiphase_confirmation": confirmation,
        "note": (
            "phases are in DISCOVERY order (dominant first); intensity_share "
            "is a screening proxy, not a weight fraction — quantify with "
            "Rietveld on the structure_paths. Trust multiphase_confirmation "
            "over the per-iteration FOMs: an inactive/low-coverage phase "
            "there is a subtraction artifact. Strong peaks left in "
            "residual_peaks = a phase missing from the library (indexing "
            "route on the residual list)."
        ),
    }
