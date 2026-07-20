"""Offline tests for full-series change detection in the curve-series scout.

The scout's visual subsample is capped at 7 spectra; the added
`reduce_curves` pass runs on the FULL series and feeds the planner a
computed change point next to the overlay. Synthetic series with planted
ground truth: a transition placed BETWEEN scout indices (invisible to the
subsample), control axis from series_metadata, fallbacks, failure
isolation, and the prompt injection.

  conda run -n scilink python tests/test_scout_series_reduction.py
"""
import logging
import os
import tempfile

import numpy as np

from scilink.skills._shared.series_reduction import reduce_curves
from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    CurveFittingPlanningController,
    SeriesScoutController,
)

RNG = np.random.default_rng(7)
results = {}
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("scout-reduction-test")


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _g(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)


def _make_series(temps, t0=85.0, width=2.0, n_channels=600):
    """Composition switch at t0: peak at 900 dies, peak at 1300 grows."""
    x = np.linspace(400, 1800, n_channels)
    curves = []
    for T in temps:
        w = 1 / (1 + np.exp(-(T - t0) / width))
        y = ((1 - w) * _g(x, 900, 40) + w * _g(x, 1300, 40)
             + RNG.normal(0, 0.004, x.size))
        curves.append(np.column_stack([x, y]))
    return curves


def _stub_plot_fn(curve_data, system_info, title_suffix=""):
    return b"\x89PNG-stub"


def _scout_state(num, stack=None, paths=None, values=None,
                 variable="temperature", unit="C"):
    state = {
        "is_single_spectrum": False,
        "num_spectra": num,
        "system_info": {},
        "series_metadata": {"variable": variable, "unit": unit,
                            "values": values if values is not None else []},
    }
    if stack is not None:
        state["spectrum_stack"] = stack
    if paths is not None:
        state["spectrum_paths"] = paths
    return state


def main():
    # 1) reduce_curves core: explicit controls, named axis, figure bytes.
    temps = list(range(30, 170, 10))
    curves = [(c[:, 0], c[:, 1]) for c in _make_series(temps)]
    out = reduce_curves(curves, controls=temps, control_source="temperature",
                        return_figure=True)
    print("1) reduce_curves core (planted change at 85):")
    check("success", out["status"] == "success")
    check("change point within half a step of 85",
          abs(out["change_point"] - 85.0) <= 5.0)
    check("control source passed through",
          out["control_variable"]["source"] == "temperature")
    check("figure returned as PNG bytes",
          out.get("score_curve_png", b"")[:4] == b"\x89PNG")
    check("no artifact paths without out_dir",
          "score_curve_path" not in out and "reduction_json_path" not in out)

    out = reduce_curves(curves)
    check("index fallback when controls omitted",
          out["status"] == "success"
          and out["control_variable"]["source"] == "index")
    check("no figure unless requested", "score_curve_png" not in out)

    print("2) reduce_curves failure paths:")
    check("too few curves -> error",
          reduce_curves(curves[:3])["status"] == "error")
    check("control-length mismatch -> error",
          reduce_curves(curves, controls=temps[:-2])["status"] == "error")

    # 3) Controller on an in-memory stack: transition planted BETWEEN scout
    #    indices. n=40 scouts {0,7,13,20,26,33,39}; transition at index 23.
    temps = [20.0 + 2.5 * i for i in range(40)]  # 20 .. 117.5
    t0 = temps[23] + 1.0  # between scout indices 20 and 26
    stack = np.stack(_make_series(temps, t0=t0, width=0.8))
    scout = SeriesScoutController(logger=logger, plot_fn=_stub_plot_fn)
    state = scout.execute(_scout_state(40, stack=stack, values=temps))
    red = state.get("series_reduction")
    print("3) scout controller, stack path (transition between scouts):")
    check("reduction present and successful",
          isinstance(red, dict) and red["status"] == "success")
    check("ran on the full series, not the 7 scouts", red["n_points"] == 40)
    check("change point within one step of planted t0",
          abs(red["change_point"] - t0) <= 2.5)
    check("control axis from series_metadata",
          red["control_variable"]["source"] == "temperature")
    check("score curve figure attached",
          red.get("score_curve_png", b"")[:4] == b"\x89PNG")
    check("additive: scout_data intact", len(state["scout_data"]) == 7)
    check("additive: overlay intact", bool(state.get("scout_overlay_plot")))

    # 4) Controller on a file-path series (no stack).
    d = tempfile.mkdtemp()
    paths = []
    for i, c in enumerate(_make_series(temps, t0=t0, width=0.8)):
        p = os.path.join(d, f"spec_{i:03d}.txt")
        np.savetxt(p, c, delimiter=",")
        paths.append(p)
    state = scout.execute(_scout_state(40, paths=paths, values=temps))
    red = state.get("series_reduction")
    print("4) scout controller, file-path series:")
    check("paths: reduction successful",
          isinstance(red, dict) and red["status"] == "success")
    check("paths: change point located", abs(red["change_point"] - t0) <= 2.5)

    # 5) Non-numeric series values -> index axis fallback.
    state = scout.execute(_scout_state(
        40, stack=stack, values=[f"run_{i}" for i in range(40)]))
    red = state.get("series_reduction")
    print("5) non-numeric control values:")
    check("index fallback", isinstance(red, dict)
          and red["control_variable"]["source"] == "index")

    # 6) No-op and failure isolation.
    print("6) no-op + failure isolation:")
    state = scout.execute({"is_single_spectrum": True})
    check("single spectrum: scout untouched", "series_reduction" not in state)
    state = scout.execute(_scout_state(
        6, paths=[os.path.join(d, f"missing_{i}.txt") for i in range(6)],
        values=list(range(6))))
    check("all loads fail: reduction None, no raise",
          state.get("series_reduction") is None)

    # 7) Long-series cap: reduction subsampled, marker set.
    temps_long = [float(i) for i in range(300)]
    stack_long = np.stack(_make_series(temps_long, t0=150.0, width=2.0,
                                       n_channels=200))
    state = scout.execute(_scout_state(300, stack=stack_long,
                                       values=temps_long))
    red = state.get("series_reduction")
    print("7) long-series cap:")
    check("capped at 256 rows", isinstance(red, dict)
          and red["n_points"] <= 256 and "of 300 spectra" in
          red.get("subsampled", ""))
    check("capped run still locates change",
          abs(red["change_point"] - 150.0) <= 3.0)

    # 8) Prompt injection next to the overlay (method uses no self state).
    stack = np.stack(_make_series(temps, t0=t0, width=0.8))
    state = scout.execute(_scout_state(40, stack=stack, values=temps))
    prompt = []
    CurveFittingPlanningController._append_scout_context(
        None, prompt, state, state["scout_data"])
    text = "\n".join(p for p in prompt if isinstance(p, str))
    images = [p for p in prompt if isinstance(p, dict)]
    print("8) planning-prompt injection:")
    check("change-detection block present",
          "Full-Series Change Detection" in text)
    check("change point stated with axis name",
          "temperature" in text and f"{state['series_reduction']['change_point']:g}" in text)
    check("score curve embedded as image",
          any(p.get("data") == state["series_reduction"]["score_curve_png"]
              for p in images))
    check("existing scout images still present",
          sum(1 for p in images if p.get("data") == b"\x89PNG-stub") == 7)
    # Reduction absent -> block absent (backcompat with pre-existing states).
    state.pop("series_reduction")
    prompt = []
    CurveFittingPlanningController._append_scout_context(
        None, prompt, state, state["scout_data"])
    text = "\n".join(p for p in prompt if isinstance(p, str))
    check("no block when reduction missing",
          "Full-Series Change Detection" not in text)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"SCOUT SERIES REDUCTION: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
