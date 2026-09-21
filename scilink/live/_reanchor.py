"""The re-anchor worker: a new recipe for a stream whose old one stopped
fitting, built in its own interpreter.

One frame, or a WINDOW of the most recent frames. A window is planned as a
series — the plan sees the change happening (what fades, what grows) instead of
one snapshot of it, so the recipe it writes still fits when the change has
completed — and the recipe is locked on the newest frame. Before that the bank
is asked about the newest frame alone (no model call): a regime seen before is
served in seconds, as with a single frame.

``python -m scilink.live._reanchor`` reads a JSON spec from stdin and writes
``result.json`` into the spec's ``out_dir``. See
``measurement_loop._ProcessEscalation`` for why this is a ``-m`` subprocess and
not a ``multiprocessing`` child.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict


def reanchor(spec: Dict[str, Any]) -> None:
    """Re-anchor on one frame. Plain-data spec in, ``result.json`` out. Never
    raises."""
    import os
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"status": "error"}
    t0 = time.perf_counter()
    try:
        handler = logging.FileHandler(out_dir / "escalation.log", encoding="utf-8")
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        # The parent already holds the sandbox approval; a child process
        # cannot answer an interactive prompt, so the approval travels.
        if spec.get("sandbox_approved"):
            os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
        if spec.get("modality") == "hyperspectral":
            payload = _hyperspectral(spec, out_dir)
            raise _Done()
        from ..agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = CurveFittingAgent(output_dir=str(out_dir / "run"),
                                  enable_human_feedback=False, **spec["agent_kwargs"])
        data, kwargs = spec["data_path"], dict(spec["analyze_kwargs"])
        window = [str(p) for p in data] if isinstance(data, (list, tuple)) else [str(data)]
        pin_data, window_info = window[-1], None
        if len(window) == 1:
            res = agent.analyze(window[0], **kwargs) or {}
        else:
            res = agent.analyze(window[-1], **{**kwargs, "bank_only": True}) or {}
            if res.get("status") != "success":
                from .measurement_loop import MeasurementLoop
                agent = CurveFittingAgent(output_dir=str(out_dir / "run_window"),
                                          enable_human_feedback=False, **spec["agent_kwargs"])
                res = agent.analyze(window, **{
                    **kwargs,
                    "profile": {"base": kwargs.get("profile") or "thorough", "trend": False,
                                "synthesis": "none", "adaptive_refit": False},
                    "series_metadata": {"variable": "frame", "values": list(range(len(window)))},
                }) or {}
                if res.get("status") == "success":
                    anchor, window_info = MeasurementLoop._single_frame_anchor(
                        Path(res.get("output_directory") or str(out_dir / "run_window")),
                        out_dir / "anchor", window)
                    res = {**res, "output_directory": anchor}
                    pin_data = window_info.pop("data")
        payload = {
            "status": res.get("status"),
            "output_directory": res.get("output_directory") or str(out_dir / "run"),
            "cold_start": res.get("cold_start"),
            "llm_calls": (res.get("stage_timings") or {}).get("llm_calls"),
            "error": res.get("error"),
            "window": window_info,
        }
        if res.get("status") == "success" and spec.get("pin_outputs"):
            # The new recipe must report the same pinned names as the old one.
            try:
                from .measurement_loop import MeasurementLoop
                from .pinning import agent_replay, pin_outputs

                def factory(d):
                    return CurveFittingAgent(output_dir=d, enable_human_feedback=False,
                                             **spec["agent_kwargs"])
                from .modality import CurveModality
                script, anchor_dir = CurveModality().anchor_script(payload["output_directory"])
                pinned = pin_outputs(
                    script=script, outputs=spec["pin_outputs"], model=agent.model,
                    replay=agent_replay(factory, str(anchor_dir), pin_data,
                                        spec.get("system_info"), str(out_dir / "pinning")))
                payload["pin_edits"] = pinned["edits"]
                payload["pin_features"] = pinned["features"]
                payload["pin_review"] = pinned.get("review")
                payload["llm_calls"] = (payload.get("llm_calls") or 0) + (
                    pinned.get("llm_calls") or pinned["attempts"])
            except Exception as e:  # noqa: BLE001 - reported to the parent
                payload["pin_error"] = f"{type(e).__name__}: {e}"
    except _Done:
        pass
    except BaseException as e:  # noqa: BLE001 - reported, never raised
        payload = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    payload["seconds"] = round(time.perf_counter() - t0, 2)
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    tmp.replace(out_dir / "result.json")


class _Done(Exception):
    """The payload is complete (a modality that needs none of the curve steps)."""


def _hyperspectral(spec: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """A datacube's rebuild or audit: fresh code for the SAME targets and output
    names (``locked_targets`` in the spec's analyze kwargs), so nothing is pinned.
    ``pin_features`` carries what the new run reports, which is what an audit
    compares and what a rebuilt recipe starts its plausible ranges from."""
    from ..agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    from .modality import HyperspectralModality
    agent = HyperspectralAnalysisAgent(output_dir=str(out_dir / "run"),
                                       enable_human_feedback=False, **spec["agent_kwargs"])
    data = spec["data_path"]
    data = data[-1] if isinstance(data, (list, tuple)) else data
    res = agent.analyze(str(data), **dict(spec["analyze_kwargs"])) or {}
    modality = HyperspectralModality()
    usable = res.get("status") in modality.usable_status
    return {
        "status": "success" if usable else (res.get("status") or "error"),
        "output_directory": res.get("output_directory") or str(out_dir / "run"),
        "llm_calls": (res.get("stage_timings") or {}).get("llm_calls"),
        "error": res.get("error"),
        "window": None,
        "pin_features": modality.features(res) if usable else {},
        "partial": res.get("status") == "partial",
    }


def main() -> int:
    reanchor(json.loads(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
