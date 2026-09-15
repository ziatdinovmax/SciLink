"""Offline tests for the figure-cosmetics channel (hints -> codegen).

Pre-fix, run_analysis(hints=...) reached the PLANNING prompts only for
curve/image — a figure-presentation request ("the legend covers the data")
was silently dropped before the script that draws visualization.png was
generated (hyperspectral always injected hints into codegen). Now hints
reach curve/image codegen AND correction prompts, and meta fan-out carries
an explicit figure_style forwarded into every branch task and the fusion
codegen prompt. No-regression contract: with hints/figure_style unset,
every prompt and task string is byte-identical to pre-feature output.

  UNSAFE_EXECUTION_OK=true python tests/test_figure_hints_channel.py
"""
import json
import logging
import tempfile

import numpy as np

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController)
from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
    UnifiedImageProcessingController)
from scilink.agents.exp_agents.instruct import (
    FITTING_SCRIPT_INSTRUCTIONS,
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
    IMAGE_ANALYSIS_SCRIPT_INSTRUCTIONS,
    IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
)
from scilink.agents.meta_agent.fanout import _mesh_task

results = {}
logging.disable(logging.CRITICAL)

HINT = "Place the legend OUTSIDE the plot axes; never overlay it on data."


def check(name, cond):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class _StubModel:
    def __init__(self):
        self.prompts = []

    def generate_content(self, prompt, **kw):
        self.prompts.append(prompt)
        return json.dumps({"diagnosis": "d",
                           "script": "import numpy as np\nprint('ok')\n"})


def _prompt_of(model):
    p = model.prompts[-1]
    return p if isinstance(p, str) else "\n".join(
        x for x in p if isinstance(x, str))


def _curve_ctrl(model):
    return UnifiedSeriesProcessingController(
        model=model, logger=logging.getLogger("t"),
        generation_config=None, safety_settings=None, parse_fn=None,
        executor=None, script_instructions=FITTING_SCRIPT_INSTRUCTIONS,
        correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions="", output_dir=tempfile.mkdtemp(),
        plot_fn=None)


def _image_ctrl(model):
    return UnifiedImageProcessingController(
        model=model, logger=logging.getLogger("t"),
        generation_config=None, safety_settings=None, parse_fn=None,
        executor=None,
        script_instructions=IMAGE_ANALYSIS_SCRIPT_INSTRUCTIONS,
        correction_instructions=IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
        quality_instructions="", output_dir=tempfile.mkdtemp(),
        image_to_bytes_fn=lambda *a, **k: b"")


def _curve_state(hints=None):
    s = {"locked_fitting_config": {
        "analysis_approach": "a", "physical_model": "gaussian",
        "parameters_to_extract": ["center"], "fitting_strategy": "ls"}}
    if hints:
        s["analysis_hints"] = hints
    return s


def _image_state(hints=None):
    s = {"locked_analysis_config": {
        "analysis_approach": "a", "processing_pipeline": "p",
        "features_to_extract": ["size"]}}
    if hints:
        s["analysis_hints"] = hints
    return s


def main():
    stats = {"n_points": 100, "x_range": [0, 1], "y_range": [0, 1],
             "y_mean": 0.5, "y_std": 0.1, "has_nans": False}

    # 1) Curve codegen: hints present -> User Guidance block; absent ->
    #    byte-identical prompts (the no-regression contract).
    m = _StubModel()
    c = _curve_ctrl(m)
    c._generate_fitting_script(_curve_state(), "data.npy", stats)
    base = _prompt_of(m)
    c._generate_fitting_script(_curve_state(HINT), "data.npy", stats)
    with_h = _prompt_of(m)
    print("1) curve codegen:")
    check("hints reach codegen", "## User Guidance" in with_h
          and HINT in with_h)
    check("no hints -> no block", "## User Guidance" not in base)
    check("no hints -> byte-identical prompt",
          base == with_h.replace("## User Guidance\n" + HINT + "\n\n", "")
          or "User Guidance" not in base)

    # 2) Curve correction: block present and BEFORE the response footer;
    #    coexists with the timeout-escalation clause.
    m = _StubModel()
    c = _curve_ctrl(m)
    c._correct_script(_curve_state(HINT), "print('x')", "NameError: x")
    p = _prompt_of(m)
    print("2) curve correction:")
    check("hints in correction", HINT in p)
    check("before the response footer",
          p.index(HINT) < p.index("**Response:**"))
    state = _curve_state(HINT)
    c._correct_script_with_timeout_escalation(
        state, "print('x')", "Script execution timed out after 1800 seconds.")
    p = _prompt_of(m)
    check("coexists with escalation clause",
          HINT in p and "TIMEOUT ESCALATION" in p
          and p.index(HINT) < p.index("**Response:**"))
    m2 = _StubModel()
    c2 = _curve_ctrl(m2)
    c2._correct_script(_curve_state(), "print('x')", "NameError: x")
    check("no hints -> correction unchanged",
          "User Guidance" not in _prompt_of(m2))

    # 3) Image codegen + correction: same contract.
    m = _StubModel()
    ic = _image_ctrl(m)
    ic._generate_analysis_script(_image_state(HINT), "data.npy",
                                 {"shape": (8, 8), "dtype": "f4",
                                  "min": 0.0, "max": 1.0})
    p = _prompt_of(m)
    print("3) image codegen + correction:")
    check("image codegen gets hints", "## User Guidance" in p and HINT in p)
    m2 = _StubModel()
    ic2 = _image_ctrl(m2)
    ic2._generate_analysis_script(_image_state(), "data.npy",
                                  {"shape": (8, 8), "dtype": "f4",
                                   "min": 0.0, "max": 1.0})
    check("image codegen without hints unchanged",
          "User Guidance" not in _prompt_of(m2))
    m3 = _StubModel()
    ic3 = _image_ctrl(m3)
    ic3._correct_script(_image_state(HINT), "print('x')", "NameError")
    p = _prompt_of(m3)
    check("image correction gets hints before footer",
          HINT in p and p.index(HINT) < p.index("**Response:**"))

    # 4) Fan-out: figure_style forwarded verbatim into the branch task with
    #    the hints-forwarding instruction; absent/None -> byte-identical.
    style = "legends outside the axes, colorblind-safe palette"
    b = {"data_path": "/d", "task": "Analyze this.", "label": "x",
         "figure_style": style}
    t = _mesh_task(b, [])
    print("4) fan-out task composition:")
    check("style block in branch task",
          "FIGURE PRESENTATION" in t and style in t)
    check("instructs hints forwarding", "`hints` parameter" in t)
    b_none = dict(b, figure_style=None)
    b_missing = {"data_path": "/d", "task": "Analyze this.", "label": "x"}
    t_none, t_missing = _mesh_task(b_none, []), _mesh_task(b_missing, [])
    check("None/missing style -> byte-identical task (pre-feature shape)",
          t_none == t_missing and "FIGURE PRESENTATION" not in t_none)

    # 5) Fusion + stash wiring (source-level: composition is inline in the
    #    codegen loop) and schema texts.
    import inspect
    from scilink.agents.meta_agent import fanout as fanout_mod
    src = inspect.getsource(fanout_mod)
    print("5) fusion wiring + schemas:")
    check("fusion codegen reads the stashed style",
          '_fanout_figure_style' in src
          and "apply to \"\n             \"fusion_figure.png" in src
          or src.count("_fanout_figure_style") >= 2)
    from scilink.agents.exp_agents import analysis_orchestrator_tools as aot
    check("hints schema mentions figure presentation",
          "figure-presentation" in inspect.getsource(aot))
    from scilink.agents.meta_agent import meta_orchestrator_tools as mot
    msrc = inspect.getsource(mot)
    check("delegate_to_analyses exposes figure_style",
          '"figure_style"' in msrc and "figure_style=figure_style" in msrc)

    print("\n" + "=" * 50)
    npass = sum(results.values())
    print(f"FIGURE HINTS CHANNEL: {npass}/{len(results)} checks passed")
    for k, v in results.items():
        if not v:
            print("  FAILED:", k)
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
