"""Will the locked recipe travel? — a deterministic check at setup.

A live recipe is replayed verbatim on frames whose features are larger, smaller
or elsewhere than in the reference. The code-generation prompt already demands
data-relative bounds; observed on a real STEM-EELS line scan, a reference taken
on a weak pixel still produced ``max=600`` on the amplitude, and every frame
over a bright crystal (true height 1300) came back pinned at 600 with R² 0.85.
The gates caught it, but only after the stream had reached those frames and at
the price of a re-anchor.

The check asks the recipe directly: replay it (strict, zero-LLM) on the
reference with its signal scaled up and down. R² is invariant under that
transform for a recipe whose constants are expressions of the data — noise
scales with the signal — so a material drop means a number read off the
reference is baked in. It costs two replays and no model call, and its verdict
goes into the setup record.

A third replay shifts the x axis by a few percent of its span. That one is
INFORMATION, not a verdict: a recipe that finds its features follows them
anywhere, while one that fixes positions does not, and fixing positions is
sometimes exactly right (XPS binding energies, the D and G bands, a known
reflection). What the operator learns is how far a feature may move before this
recipe stops fitting and the loop has to rebuild it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

SCALES = (3.0, 0.35)
R2_TOLERANCE = 0.03
#: x-axis shift of the position replay, as a fraction of the measured span.
SHIFT_FRACTION = 0.03


def check_portability(replay_r2: Callable[[str, str], Optional[float]], reference_data: str,
                      reference_r2: Optional[float], work_dir: str,
                      scales=SCALES, tolerance: float = R2_TOLERANCE,
                      shift_fraction: Optional[float] = SHIFT_FRACTION) -> Dict[str, Any]:
    """``replay_r2(data_path, tag)`` returns the locked recipe's R² on a file
    (None when it failed to run). Returns ``{"portable": bool, "trials": [...]}``;
    an empty dict when the reference cannot be read or has no R² to compare with."""
    from .instruments import read_curve
    if reference_r2 is None:
        return {}
    try:
        x, y, x_label, y_label = read_curve(reference_data)
    except (OSError, ValueError):
        return {}
    d = Path(work_dir)
    d.mkdir(parents=True, exist_ok=True)
    trials: List[Dict[str, Any]] = []
    for scale in scales:
        path = d / f"reference_x{scale:g}.csv"
        np.savetxt(path, np.column_stack([x, y * scale]), delimiter=",",
                   header=f"{x_label},{y_label}", comments="")
        r2 = replay_r2(str(path), f"x{scale:g}")
        ok = r2 is not None and r2 >= reference_r2 - tolerance
        trials.append({"signal_scale": scale, "r_squared": None if r2 is None else round(r2, 4),
                       "ok": bool(ok)})
    report = {"portable": all(t["ok"] for t in trials),
              "reference_r_squared": round(reference_r2, 4), "trials": trials}
    # Positions: the same curve, its x axis moved. Reported, never judged.
    span = float(np.nanmax(x) - np.nanmin(x))
    if shift_fraction and span > 0:
        shift = shift_fraction * span
        path = d / "reference_shifted.csv"
        np.savetxt(path, np.column_stack([x + shift, y]), delimiter=",",
                   header=f"{x_label},{y_label}", comments="")
        r2 = replay_r2(str(path), "shifted")
        report["positions"] = {
            "shift": round(shift, 6), "shift_fraction": shift_fraction,
            "r_squared": None if r2 is None else round(r2, 4),
            "follows_features": bool(r2 is not None and r2 >= reference_r2 - tolerance)}
    return report


def describe_positions(report: Dict[str, Any]) -> str:
    """One sentence on how the recipe treats feature positions ('' if untested)."""
    pos = report.get("positions")
    if not pos:
        return ""
    if pos["follows_features"]:
        return "It finds its features where they are, so it follows them if they move."
    how = "it did not run" if pos["r_squared"] is None else f"R² fell to {pos['r_squared']}"
    return (f"It fixes feature positions. With the axis moved by {pos['shift']:g} "
            f"({pos['shift_fraction']:.0%} of the span) {how}, so a larger move than that "
            "means a rebuild. That is expected when positions are known in advance.")


def describe(report: Dict[str, Any]) -> str:
    bad = [t for t in report.get("trials", []) if not t["ok"]]
    if not bad:
        return "The recipe travels. Same fit quality with the signal scaled up and down."
    parts = [f"at ×{t['signal_scale']:g} signal " + ("it did not run" if t["r_squared"] is None
                                                    else f"R² {t['r_squared']}") for t in bad]
    return (f"The recipe may not travel. Reference R² {report['reference_r_squared']}, "
            + ", ".join(parts) + ". A constant read off the reference is probably baked in.")


def agent_replay_r2(agent_factory: Callable[[str], Any], anchor_dir: str, system_info: Any,
                    work_dir: str, edits: Optional[List[Dict[str, Any]]] = None
                    ) -> Callable[[str, str], Optional[float]]:
    """``replay_r2`` through the curve agent's strict zero-LLM replay."""
    def replay(data_path: str, tag: str) -> Optional[float]:
        try:
            agent = agent_factory(str(Path(work_dir) / f"run_{tag}"))
            kwargs: Dict[str, Any] = dict(
                system_info=system_info, prior_analysis_paths=[str(anchor_dir)],
                reuse_locked_script=True, profile="realtime", strict_replay=True)
            if edits:
                kwargs["script_edits"] = list(edits)
            res = agent.analyze(data_path, **kwargs) or {}
            if res.get("status") != "success":
                return None
            r2 = (res.get("fit_quality") or {}).get("r_squared")
            if r2 is None:
                r2 = (res.get("reuse_validity") or {}).get("r_squared")
            return None if r2 is None else float(r2)
        except Exception:  # noqa: BLE001 - a recipe that cannot run is the finding
            return None
    return replay


# ── datacubes ────────────────────────────────────────────────────────────────

#: A quantity that neither stays put nor follows the signal by more than this.
CUBE_TOLERANCE = 0.05


def check_cube_portability(replay: Callable[[str, str], Optional[Dict[str, float]]],
                           reference_cube: str, work_dir: str, tracked: Optional[List[str]] = None,
                           scales=SCALES, tolerance: float = CUBE_TOLERANCE,
                           load: Optional[Callable[[Any], np.ndarray]] = None) -> Dict[str, Any]:
    """The same question for a locked CUBE recipe, asked of its outputs.

    ``replay(cube_path, tag)`` runs the locked script on a cube and returns its
    flat features, or ``None`` when the replay did not pass the agent's own gate.
    The reference cube is replayed as it is and with every count multiplied by
    each of ``scales`` (noise included, so nothing about the data changes but its
    level). A per-pixel analysis that does not depend on the signal level then
    returns each quantity either UNCHANGED (an energy, a width, a fit quality, a
    pixel count) or SCALED with the counts (an amplitude, an area). A quantity
    that does neither, or a replay that stops passing, points at a constant read
    off the reference: an amplitude bound, a threshold in counts, a mask level.
    ``tracked`` limits the verdict to those outputs (the others are reported).
    Zero model calls. ``{}`` when the reference itself cannot be replayed."""
    from .modality import load_cube
    try:
        cube = (load or load_cube)(reference_cube)        # an image is asked the same question
    except (OSError, ValueError):
        return {}
    base = replay(str(reference_cube), "x1")
    if not base:
        return {}
    d = Path(work_dir)
    d.mkdir(parents=True, exist_ok=True)
    side = Path(str(reference_cube)).with_suffix(".json")
    watched = [k for k in (tracked or list(base)) if k in base]
    trials: List[Dict[str, Any]] = []
    for scale in scales:
        path = d / f"reference_x{scale:g}.npy"
        np.save(path, (cube * scale).astype(np.float32))
        if side.is_file():                        # the axis and the rest of the metadata travel
            path.with_suffix(".json").write_text(side.read_text())
        feats = replay(str(path), f"x{scale:g}")
        trial: Dict[str, Any] = {"scale": scale, "passed": bool(feats), "depends_on_level": {}}
        for key, ref in base.items():
            if not feats or key not in feats:
                if feats is not None and key in watched:
                    trial["depends_on_level"][key] = None          # the output disappeared
                continue
            new = feats[key]
            size = max(abs(ref), abs(new), 1e-12)
            unchanged = abs(new - ref) <= tolerance * size
            follows = abs(new - scale * ref) <= tolerance * max(abs(scale * ref), abs(new), 1e-12)
            if not (unchanged or follows):
                trial["depends_on_level"][key] = round(new / ref, 4) if ref else None
        trial["ok"] = trial["passed"] and not any(k in watched for k in trial["depends_on_level"])
        trials.append(trial)
    return {"kind": "datacube", "portable": all(t["ok"] for t in trials), "trials": trials,
            "tracked": watched}


def describe_cube(report: Dict[str, Any]) -> str:
    if not report:
        return ""
    if report["portable"]:
        extra = sorted({k for t in report["trials"] for k in t["depends_on_level"]})
        return ("The recipe does not depend on the signal level"
                + (f" (untracked {', '.join(extra[:4])} does)." if extra else "."))
    parts = []
    for t in report["trials"]:
        if not t["passed"]:
            parts.append(f"at {t['scale']:g}x the counts the replay no longer passes its gate")
        elif any(k in report["tracked"] for k in t["depends_on_level"]):
            bad = [k for k in t["depends_on_level"] if k in report["tracked"]]
            parts.append(f"at {t['scale']:g}x the counts {', '.join(bad[:3])} neither stays nor scales")
    return ("The recipe depends on the signal level of the reference: " + "; ".join(parts)
            + ". It probably carries a constant read off that cube (a bound, a threshold in counts).")
