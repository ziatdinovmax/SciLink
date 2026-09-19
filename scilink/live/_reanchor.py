"""The re-anchor worker: one analysis of one frame, in its own interpreter.

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
        from ..agents.exp_agents.curve_fitting_agent import CurveFittingAgent
        agent = CurveFittingAgent(output_dir=str(out_dir / "run"),
                                  enable_human_feedback=False, **spec["agent_kwargs"])
        res = agent.analyze(spec["data_path"], **spec["analyze_kwargs"]) or {}
        payload = {
            "status": res.get("status"),
            "output_directory": res.get("output_directory") or str(out_dir / "run"),
            "cold_start": res.get("cold_start"),
            "llm_calls": (res.get("stage_timings") or {}).get("llm_calls"),
            "error": res.get("error"),
        }
        if res.get("status") == "success" and spec.get("pin_outputs"):
            # The new recipe must report the same pinned names as the old one.
            try:
                from .measurement_loop import MeasurementLoop
                from .pinning import agent_replay, pin_outputs

                def factory(d):
                    return CurveFittingAgent(output_dir=d, enable_human_feedback=False,
                                             **spec["agent_kwargs"])
                script, anchor_dir = MeasurementLoop._anchor_script(payload["output_directory"])
                pinned = pin_outputs(
                    script=script, outputs=spec["pin_outputs"], model=agent.model,
                    replay=agent_replay(factory, str(anchor_dir), spec["data_path"],
                                        spec.get("system_info"), str(out_dir / "pinning")))
                payload["pin_edits"] = pinned["edits"]
                payload["pin_features"] = pinned["features"]
                payload["llm_calls"] = (payload.get("llm_calls") or 0) + pinned["attempts"]
            except Exception as e:  # noqa: BLE001 - reported to the parent
                payload["pin_error"] = f"{type(e).__name__}: {e}"
    except BaseException as e:  # noqa: BLE001 - reported, never raised
        payload = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    payload["seconds"] = round(time.perf_counter() - t0, 2)
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
    tmp.replace(out_dir / "result.json")


def main() -> int:
    reanchor(json.loads(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
