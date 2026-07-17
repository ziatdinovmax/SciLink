"""``reconcile_series_phases`` tool — couple a profile-fit series with an
identification series.

In-situ XRD analysis has two complementary views that answer different
questions and depend on different things:

* **Profile fitting** (the ``xrd_profile`` skill) answers HOW the structure
  evolves — per-frame peak positions, widths, and areas, and the transition
  read from peak appearance/disappearance/shift. It fits a lineshape model
  but is database-INDEPENDENT: it needs no reference and works even when the
  phase is in no database.
* **Identification** (this ``xrd`` skill) answers WHICH phases — but only
  where a reference exists in a database.

Neither alone is the full picture: profile fitting gives rich trends with no
names; identification gives names but hits a wall on phases absent from
databases (organics, novel products). This tool RECONCILES the two —
attributing the profile peak-evolution trends to the identified phases, and
cross-checking the transition temperature the two methods find independently.
Where identification could not name a phase, the trend stays honestly
``unidentified`` rather than force-labeled.

Run it AFTER both passes over the same series: profile-fit the frames
(``fit_pattern`` per frame) to get the per-frame peaks, and identify the
establishing frames (``search_match_pattern`` / ``identify_mixture``) to get
the per-frame phase. This tool does no fitting or searching itself — it is the
deterministic join over their outputs."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


TOOL_SPEC = ToolSpec(
    name="reconcile_series_phases",
    description=(
        "Couple an in-situ PROFILE-FIT series (per-frame peak positions/areas, "
        "from xrd_profile fit_pattern) with an IDENTIFICATION series (per-frame "
        "phase, from search_match_pattern / identify_mixture) over the SAME "
        "frames: attribute the profile-fit peak-evolution trends to the "
        "identified phases and CROSS-CHECK the transition temperature the two "
        "methods find independently. The pragmatic 'profile + identification "
        "together' for a series — profile answers HOW the structure evolves "
        "(database-independent), identification answers WHICH phases (where a "
        "database allows); this joins them. Trends whose phase could not be "
        "identified stay 'unidentified' rather than force-labeled. Purely "
        "deterministic post-processing over the two passes' outputs."
    ),
    import_line=("from scilink.skills.structure_matching.xrd.reconcile_series "
                 "import reconcile_series_phases"),
    signature=(
        "reconcile_series_phases(frames, phase_ids, tol_deg=0.25, "
        "min_presence_frac=0.2, agreement_deg=15.0, regime_window_frac=0.33, "
        "crossover_threshold=0.5, output_figure=None) -> dict"
    ),
    parameters={
        "frames": {"type": "list[dict]", "description": "Per-frame profile-fit output in series order: [{'value': <T or time>, 'peaks': [{'center': deg, 'area': float}, ...]}]. 'peaks' is each frame's fitted peak list (fit_pattern / fit_profile parameters). 'value' is the series variable (temperature/time)."},
        "phase_ids": {"type": "list[dict]", "description": "Per-frame identification output, aligned with frames (same order/length): [{'value': <T>, 'phase': <formula or name, or null if not identified>, 'figure_of_merit': <float, optional>}]. Frames where identification declined carry phase=null (they stay 'unidentified' in the labeling)."},
        "tol_deg": {"type": "float", "description": "Position tolerance (2θ°) for tracking a peak across frames and for clustering peaks into reflections (default 0.25). RAISE for peaks that shift a lot with T (thermal expansion) or noisier centers; LOWER for sharp well-calibrated data."},
        "min_presence_frac": {"type": "float", "description": "A tracked reflection is kept only if it appears in at least this fraction of frames (default 0.2) — filters transient noise peaks. RAISE to keep only persistent reflections; LOWER to retain short-lived ones (a transient intermediate's peaks)."},
        "agreement_deg": {"type": "float", "description": "Profile and identification transition estimates within this many series-VARIABLE units (°C, time) are called 'consistent' (default 15). RAISE for a slow ramp / coarse frame spacing; LOWER for a fine scan. A larger gap is flagged DIVERGENT (investigate: mis-tracked peaks, a mid-series false ID, or a genuine two-step process the single-crossover model misses)."},
        "regime_window_frac": {"type": "float", "description": "Fraction of frames at EACH end used to classify which reflections belong to the start- vs end-phase family (default 0.33 = first/last third). LOWER (e.g. 0.15) when the endpoint frames are pure; RAISE toward 0.5 for a gradual transformation whose extremes barely differ."},
        "crossover_threshold": {"type": "float", "description": "End-phase area-share level that defines the transition (default 0.5 = ~50% conversion). Change only to mark the transition at a different conversion fraction."},
        "output_figure": {"type": "str", "description": "Optional path to save the reconciled figure (phase-labeled peak-area evolution + both transition estimates). None skips plotting."},
    },
    required=["frames", "phase_ids"],
    returns=(
        "dict: 'low_t_phase' / 'high_t_phase' (the identified phase dominant "
        "in each regime, or null=unidentified), 'tracked_peaks' (each: "
        "position, regime low|high, phase label, area series), "
        "'transition_profile' (from the peak-area-share crossover) and "
        "'transition_identification' (midpoint of the identified phase "
        "switch, ignoring unidentified transition frames), 'agreement' "
        "(deg apart + consistent|divergent|one_sided), 'figure' (path if "
        "output_figure given), 'note'. When a regime's phase is null, its "
        "trends are real but UNNAMED — the phase is not in the searched "
        "database (organics/novel products): report the peak evolution and "
        "recommend an empirical reference or the indexing route, do not "
        "invent a name."
    ),
    when_to_use=(
        "AFTER profile-fitting an in-situ series (xrd_profile) AND identifying "
        "its establishing frames (xrd) — to produce the combined view: "
        "phase-labeled structural-evolution trends with a cross-validated "
        "transition. The recommended in-situ deliverable when both a database "
        "match and database-independent profile-fit trends are wanted."
    ),
)


def _frame_peaks(fr: dict) -> list[tuple]:
    out = []
    for p in fr.get("peaks", []) or []:
        try:
            out.append((float(p["center"]), float(p.get("area", p.get("amplitude", 0.0)))))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def reconcile_series_phases(
    frames: Sequence[dict],
    phase_ids: Sequence[dict],
    tol_deg: float = 0.25,
    min_presence_frac: float = 0.2,
    agreement_deg: float = 15.0,
    regime_window_frac: float = 0.33,
    crossover_threshold: float = 0.5,
    output_figure: Optional[str] = None,
) -> dict[str, Any]:
    """Join a profile-fit series with an identification series. See ``TOOL_SPEC``.

    Thin XRD wrapper over the technique-agnostic core in
    ``scilink.skills._shared._reconcile``: maps 2θ peaks → features and
    phases → labels, adds the XRD figure and vocabulary. The tracking /
    regime-split / attribution / transition-cross-check math is the shared
    core, reused by any spectroscopy that has a profile-fitting and an
    identification pass over a series."""
    from ..._shared._reconcile import reconcile_series as _core

    # XRD vocabulary -> generic core vocabulary
    feature_frames = [{"value": f.get("value", i),
                       "features": [{"position": c, "weight": a}
                                    for c, a in _frame_peaks(f)]}
                      for i, f in enumerate(frames)]
    label_frames = [{"value": p.get("value"), "label": p.get("phase")}
                    for p in phase_ids]
    r = _core(feature_frames, label_frames, tol=tol_deg,
              min_presence_frac=min_presence_frac, agreement_units=agreement_deg,
              regime_window_frac=regime_window_frac,
              crossover_threshold=crossover_threshold)

    T = np.array(r["values"])
    weight = np.array(r["_weight"])
    low_t, high_t = np.array(r["_low_idx"], int), np.array(r["_high_idx"], int)
    lo_phase, hi_phase = r["low_regime_label"], r["high_regime_label"]
    t_profile, t_id = r["transition_profile"], r["transition_identification"]

    fig_path = None
    if output_figure:
        try:
            fig_path = _plot(T, weight, low_t, high_t,
                             np.array(r["_low_share"]), np.array(r["_high_share"]),
                             r["_refs"], lo_phase, hi_phase, t_profile, t_id,
                             output_figure)
        except Exception as exc:
            _logger.warning("reconcile figure failed: %s", exc)

    tracked = [{"position_deg": tf["position"], "regime": tf["regime"],
                "phase": tf["label"], "area_series": tf["weight_series"]}
               for tf in r["tracked_features"]]
    agree = {"deg_apart": r["agreement"]["units_apart"],
             "verdict": r["agreement"]["verdict"]}

    return {
        "series_variable_values": [float(v) for v in T],
        "low_t_phase": lo_phase,
        "high_t_phase": hi_phase,
        "tracked_peaks": tracked,
        "transition_profile": t_profile,
        "transition_identification": t_id,
        "agreement": agree,
        "figure": fig_path,
        "note": (
            "profile-fit trends are database-independent (they fit peak "
            "shapes but need no reference); the phase "
            "labels come from identification and are only as good as the "
            "database coverage. A null low/high phase means that regime's "
            "phase is NOT in the searched database (common for organics / "
            "novel products) — the peak evolution is real but unnamed; "
            "recommend an empirical reference (a pure frame) or the indexing "
            "route, do not invent a name. When the two transitions DIVERGE, "
            "suspect mis-tracked peaks, a mid-series false ID, or a two-step "
            "process the single-crossover model misses."
        ),
    }


def _plot(T, area, low_t, high_t, low_share, high_share, refs,
          lo_phase, hi_phase, t_profile, t_id, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2]})
    for j in low_t:
        ax[0].plot(T, area[:, j], "-", color="tab:blue", alpha=0.6, lw=1)
    for j in high_t:
        ax[0].plot(T, area[:, j], "-", color="tab:red", alpha=0.6, lw=1)
    ax[0].plot([], [], color="tab:blue",
               label=f"LOW-T: {lo_phase or 'unidentified'}")
    ax[0].plot([], [], color="tab:red",
               label=f"HIGH-T: {hi_phase or 'unidentified'}")
    ax[0].set_ylabel("integrated peak area")
    ax[0].set_title("Profile + Identification, reconciled\n"
                    "peak-area evolution (profile), phase labels (identification)")
    ax[0].legend(fontsize=9, loc="upper right")
    ax[1].plot(T, low_share, "o-", color="tab:blue", label="LOW-T area share")
    ax[1].plot(T, high_share, "s-", color="tab:red", label="HIGH-T area share")
    if t_profile is not None:
        ax[1].axvline(t_profile, color="k", ls="--",
                      label=f"profile transition ≈ {t_profile:.1f}")
    if t_id is not None:
        ax[1].axvline(t_id, color="tab:green", ls=":",
                      label=f"identification transition ≈ {t_id:.1f}")
    ax[1].set_xlabel("series variable"); ax[1].set_ylabel("area share")
    ax[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path
