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
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

SCALES = (3.0, 0.35)
R2_TOLERANCE = 0.03


def check_portability(replay_r2: Callable[[str, str], Optional[float]], reference_data: str,
                      reference_r2: Optional[float], work_dir: str,
                      scales=SCALES, tolerance: float = R2_TOLERANCE) -> Dict[str, Any]:
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
    return {"portable": all(t["ok"] for t in trials), "reference_r_squared": round(reference_r2, 4),
            "trials": trials}


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
