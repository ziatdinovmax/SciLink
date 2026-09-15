"""``track_phase_series`` tool — phase evolution across an in-situ series.

Identification of a series is NOT N independent identifications: identify
each phase at its ESTABLISHING frame (where it is purest — the first/last
frames for a simple transformation ramp; the first crystalline frame for a
crystallization; a mid-series frame for a multi-step path), then track —
every frame is a mixture of the established endmembers until residual
evidence says otherwise, and an alerted residual is identified, added to
the set, and re-tracked. This driver executes the tracking half
deterministically: one joint multi-phase MILP per frame against a FIXED
endmember set, no database search and no LLM in the per-frame loop.

Endmember references take three forms, because a real series often contains a
phase no database holds (validated on an in-situ dehydration series whose
product phase is absent from COD):

* a library entry (``{"source_id": ...}``) — resolved from the fingerprint
  library;
* a simulated pattern (``{"sim_two_theta": [...], "sim_intensity": [...]}``)
  — e.g. from ``simulate_xrd_pattern`` on an identified ``structure_path``;
* an EMPIRICAL pattern (``{"positions": [...], "intensities": [...]}``) —
  the extracted peak list of a pure frame itself, no structure needed.

Per-phase ``share`` is matched-intensity coverage — a *screening* fraction
proxy (scattering power differs between phases); multi-phase Rietveld is the
quantitative step. The per-frame MILP refits each phase's lattice scale, so
thermal expansion is followed (and reported) rather than mistaken for a
phase change."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


TOOL_SPEC = ToolSpec(
    name="track_phase_series",
    description=(
        "Track phase evolution across an in-situ / operando series (T-ramp, "
        "time series): one joint multi-phase MILP per frame against a FIXED "
        "endmember set → per-frame phase shares, per-phase lattice-scale "
        "drift (thermal expansion), onset/completion frames, coexistence "
        "window, and residual alerts flagging frames that the endmembers "
        "cannot explain (transient intermediate phase → run identify_mixture "
        "on THAT frame, add the discovered phase, re-track). Endmembers come "
        "from identifying each phase at its ESTABLISHING frame — first/last "
        "frames for a simple transformation ramp, the first crystalline "
        "frame for a crystallization, a mid-series frame for a multi-step "
        "path; a phase missing from every database is passed as an EMPIRICAL "
        "reference (its establishing frame's own peak list). Deterministic, "
        "no per-frame search or LLM."
    ),
    import_line=("from scilink.skills.structure_matching.xrd.track_phase_series "
                 "import track_phase_series"),
    signature=(
        "track_phase_series(frames, endmembers, wavelength='CuKa', "
        "library_path=None, tol_deg=0.3, onset_threshold=0.05, "
        "residual_alert_frac=0.25, frame_stride=1, fit_lattice_scale=True) "
        "-> dict"
    ),
    parameters={
        "frames": {"type": "list[dict]", "description": "Per-frame extracted peak lists, in series order: [{'positions': [...], 'intensities': [...], 'label': <optional T/time>}]. Run extract_peaks per frame (crop low-angle air-scatter upturns FIRST for in-situ lab data — a 2θ<5° ramp starves peak detection)."},
        "endmembers": {"type": "list[dict]", "description": "The fixed phase set to track, each ONE of: {'source_id': <library id>} (resolved from the fingerprint library), {'sim_two_theta': [...], 'sim_intensity': [...]} (simulated from an identified structure), or {'positions': [...], 'intensities': [...]} (EMPIRICAL — a pure frame's peak list, for phases absent from databases). Optional 'label'/'formula' for reporting. Identify endpoints first; 2-3 endmembers is typical."},
        "wavelength": {"type": "str | float", "description": "Source ('CuKa','MoKa',…) or Å — needed only to convert library-entry d-spacings; synchrotron wavelengths pass as the float."},
        "library_path": {"type": "str", "description": "Fingerprint parquet for source_id endmembers (default resolution: $SCILINK_XRD_FINGERPRINT_DB, then the per-user store)."},
        "tol_deg": {"type": "float", "description": "Peak-match tolerance (2θ°, default 0.3). LOWER for calibrated synchrotron data; RAISE if peaks drift more than this between the reference and hot frames beyond what the ±4% lattice-scale refit absorbs."},
        "onset_threshold": {"type": "float", "description": "A phase counts as PRESENT in a frame when its share ≥ this (default 0.05). RAISE to ignore trace-level flicker when peak lists are noisy; LOWER to catch the earliest nucleation frames."},
        "residual_alert_frac": {"type": "float", "description": "Flag a frame when the intensity fraction unexplained by ALL endmembers ≥ this (default 0.25). Flagged frames are where a transient intermediate phase would appear — run identify_mixture on them. LOWER for cleaner data / higher sensitivity to intermediates; RAISE if texture/background artifacts trip false alerts."},
        "frame_stride": {"type": "int", "description": "Process every Nth frame (default 1 = all). For very long series (1000s of frames, ~0.1-0.5 s/frame MILP) screen coarsely first (e.g. stride 10), then re-run stride 1 on the detected transition window."},
        "fit_lattice_scale": {"type": "bool", "description": "Refit each phase's lattice scale (±4%) per frame (default True) — follows thermal expansion so peak drift is not mistaken for phase loss; the reported per-frame 'lattice_scales' drift IS the expansion signal. Disable only for fixed-temperature series where drift itself would indicate a problem."},
    },
    required=["frames", "endmembers"],
    returns=(
        "dict: 'shares' — per processed frame, {endmember_label: "
        "matched-intensity share} (screening fraction proxy, NOT weight "
        "fractions — quantify with multi-phase Rietveld); 'lattice_scales' — "
        "per frame per phase (drift traces thermal expansion); "
        "'residual_frac' per frame (intensity no endmember explains); "
        "'frame_labels', 'frame_indices'; 'phase_events' — per endmember "
        "{onset_frame, final_frame, n_frames_present, max_share}; "
        "'coexistence_frames' — frames with ≥2 phases present (the "
        "transition window); 'residual_alert_frames' — frames exceeding "
        "residual_alert_frac: run identify_mixture on these (possible "
        "transient intermediate). Onset/final are frame INDICES into the "
        "original series; use frame_labels for T/time."
    ),
    when_to_use=(
        "Any in-situ / operando series AFTER its establishing frames are "
        "identified (search_match_pattern / identify_mixture — first/last "
        "frames for a simple ramp; adapt for amorphous starts, multi-step "
        "paths, or ever-present spectator phases): "
        "dehydration/decomposition ramps, phase-transition "
        "scans, operando cycling. Complements the per-frame profile-fitting "
        "series loop (xrd_profile skill): this tool answers WHICH phases and "
        "HOW MUCH per frame; profile fitting answers peak shapes/positions."
    ),
)


def _d_to_tt(d: np.ndarray, lam: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        s = lam / (2.0 * np.asarray(d, dtype=float))
    s = np.clip(s, -1.0, 1.0)
    return 2.0 * np.degrees(np.arcsin(s))


def _resolve_endmember(ref: dict, lam: float, library_path: Optional[str],
                       idx: int) -> dict[str, Any]:
    """Normalize any accepted reference form into MILP candidate keys."""
    label = ref.get("label") or ref.get("formula") or ref.get("id")
    if "source_id" in ref:
        from .fingerprint import _load_library
        df = _load_library(library_path)
        rows = df[df["source_id"] == ref["source_id"]]
        if not len(rows):
            raise ValueError(f"endmember source_id {ref['source_id']!r} not "
                             "found in the fingerprint library")
        row = rows.iloc[0]
        tt = _d_to_tt(np.asarray(row["ds"], dtype=float), lam)
        ii = np.asarray(row["intensities"], dtype=float)
        fin = np.isfinite(tt)
        return {"id": str(ref["source_id"]), "formula": row["formula"],
                "label": label or row["formula"],
                "sim_two_theta": tt[fin].tolist(),
                "sim_intensity": ii[fin].tolist()}
    if "sim_two_theta" in ref:
        tt, ii = ref["sim_two_theta"], ref["sim_intensity"]
    elif "positions" in ref:                      # empirical (pure frame)
        tt, ii = ref["positions"], ref["intensities"]
    else:
        raise ValueError(
            f"endmember #{idx} has none of the accepted forms: 'source_id', "
            "'sim_two_theta'+'sim_intensity', or 'positions'+'intensities'")
    if len(tt) != len(ii):
        raise ValueError(f"endmember #{idx}: positions/intensities length mismatch")
    return {"id": str(label or f"phase_{idx}"),
            "formula": str(ref.get("formula", "")),
            "label": str(label or f"phase_{idx}"),
            "sim_two_theta": [float(v) for v in tt],
            "sim_intensity": [float(v) for v in ii]}


def track_phase_series(
    frames: Sequence[dict],
    endmembers: Sequence[dict],
    wavelength: Any = "CuKa",
    library_path: Optional[str] = None,
    tol_deg: float = 0.3,
    onset_threshold: float = 0.05,
    residual_alert_frac: float = 0.25,
    frame_stride: int = 1,
    fit_lattice_scale: bool = True,
) -> dict[str, Any]:
    """Per-frame joint MILP tracking of a fixed endmember set. See ``TOOL_SPEC``."""
    from .fingerprint import _resolve_lam
    from .score_match_robust import score_xrd_match_multiphase

    if not frames:
        raise ValueError("frames is empty")
    if not endmembers:
        raise ValueError("endmembers is empty — identify the series endpoints "
                         "first and pass the identified/empirical references")
    lam = _resolve_lam(wavelength)
    cands = [_resolve_endmember(dict(r), lam, library_path, i)
             for i, r in enumerate(endmembers)]
    labels = []
    for c in cands:                                # de-duplicate display labels
        base = c["label"]
        lab, k = base, 2
        while lab in labels:
            lab, k = f"{base}#{k}", k + 1
        labels.append(lab)

    stride = max(1, int(frame_stride))
    idxs = list(range(0, len(frames), stride))
    shares, scales, residuals, frame_labels = [], [], [], []
    for fi in idxs:
        fr = frames[fi]
        pos = [float(v) for v in fr["positions"]]
        inten = [float(v) for v in fr["intensities"]]
        frame_labels.append(fr.get("label", fi))
        try:
            r = score_xrd_match_multiphase(
                exp_peaks={"positions": pos, "intensities": inten},
                candidates=cands, tol_deg=float(tol_deg),
                fit_lattice_scale=bool(fit_lattice_scale))
        except Exception as exc:
            _logger.warning("frame %d MILP failed: %s", fi, exc)
            shares.append({lab: None for lab in labels})
            scales.append({lab: None for lab in labels})
            residuals.append(None)
            continue
        by_id = {p["id"]: p for p in r["active_phases"]}
        sh = {lab: round(float(by_id[c["id"]]["coverage"]), 3)
              if c["id"] in by_id else 0.0
              for lab, c in zip(labels, cands)}
        sc = {lab: round(float(by_id[c["id"]]["lattice_scale"]), 4)
              if c["id"] in by_id else None
              for lab, c in zip(labels, cands)}
        shares.append(sh)
        scales.append(sc)
        # exp peaks matched at most once across phases -> matched fractions
        # are disjoint; the unexplained remainder is 1 - sum(shares)
        residuals.append(round(max(0.0, 1.0 - sum(v for v in sh.values()
                                                  if v is not None)), 3))

    events = {}
    for lab in labels:
        present = [k for k, sh in enumerate(shares)
                   if (sh[lab] or 0.0) >= float(onset_threshold)]
        events[lab] = {
            "onset_frame": idxs[present[0]] if present else None,
            "final_frame": idxs[present[-1]] if present else None,
            "n_frames_present": len(present),
            "max_share": max((sh[lab] or 0.0) for sh in shares),
        }
    coexist = [idxs[k] for k, sh in enumerate(shares)
               if sum((v or 0.0) >= float(onset_threshold)
                      for v in sh.values()) >= 2]
    alerts = [idxs[k] for k, rf in enumerate(residuals)
              if rf is not None and rf >= float(residual_alert_frac)]

    return {
        "n_frames": len(frames),
        "n_processed": len(idxs),
        "frame_indices": idxs,
        "frame_labels": frame_labels,
        "endmembers": labels,
        "shares": shares,
        "lattice_scales": scales,
        "residual_frac": residuals,
        "phase_events": events,
        "coexistence_frames": coexist,
        "residual_alert_frames": alerts,
        "note": (
            "shares are matched-intensity fractions (screening proxy, not "
            "weight fractions — quantify with multi-phase Rietveld). "
            "coexistence_frames bound the transition window. Frames in "
            "residual_alert_frames contain intensity NO endmember explains — "
            "run identify_mixture on those frames (transient intermediate "
            "phase); a residual rising at the series END means a product "
            "phase is missing from the endmember set (identify the final "
            "frame). lattice_scales drifting smoothly with T is thermal "
            "expansion, not a phase change."
        ),
    }
