"""Offline tests for `literature_context` injection into planner prompts.

Three contracts under test:

1. `ImagePlanningController._build_planning_prompt` appends a `## Literature`
   section when `state["literature_context"]` is set (no task_mode field on
   the image agent — always inject if present).

2. `HumanFeedbackRefinementController._plan_analysis` (curve fitting) appends
   `## Literature` when `literature_context` is set AND
   `task_mode != "identification"`.

3. The same curve-fitting planner WITHHOLDS `## Literature` when
   `task_mode == "identification"` — the load-bearing guard that protects the
   unbiased-fit semantics of identification mode.

4. `UnifiedCurveSynthesisController._build_interpretation_prompt` (Stage 2)
   appends `## Literature` unconditionally — i.e., identification-mode
   candidate enumeration still receives the lit context. (Smoke-tested by
   grepping the source rather than constructing the full state; full
   pipeline instantiation is too heavy for an offline test and the
   conditional has no task_mode branch.)
"""
from __future__ import annotations

import inspect
import io
import json
import logging
from types import SimpleNamespace

import numpy as np


# --- Image planner -----------------------------------------------------------

def _make_image_planner():
    """Construct an ImagePlanningController instance with stub dependencies."""
    from scilink.agents.exp_agents.controllers.image_analysis_controllers import (
        ImagePlanningController,
    )

    return ImagePlanningController(
        model=SimpleNamespace(),  # not called by _build_planning_prompt
        logger=logging.getLogger("test"),
        generation_config=None,
        safety_settings=None,
        parse_fn=lambda r: (None, None),
        instructions="STUB_INSTRUCTIONS",
        output_dir="/tmp",
    )


def _minimal_image_state(literature_context: str | None = None) -> dict:
    return {
        "original_image_bytes": b"\xff\xd8\xff",  # JPEG-ish stub
        "image_statistics": {"mean": 0.5, "std": 0.1},
        "system_info": {"material": "stub"},
        "is_single_image": True,
        "num_images": 1,
        "literature_context": literature_context,
    }


def _stringify_parts(parts: list) -> str:
    """Flatten a multimodal prompt parts list to a single searchable string."""
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            out.append(f"<{p.get('mime_type', 'blob')}>")
    return "\n".join(out)


def test_image_planner_injects_literature_when_present():
    planner = _make_image_planner()
    state = _minimal_image_state(literature_context="DOMAIN_LIT_CONTENT")
    parts = planner._build_planning_prompt(state)
    flat = _stringify_parts(parts)
    assert "## Literature" in flat
    assert "DOMAIN_LIT_CONTENT" in flat


def test_image_planner_omits_literature_when_absent():
    planner = _make_image_planner()
    state = _minimal_image_state(literature_context=None)
    parts = planner._build_planning_prompt(state)
    flat = _stringify_parts(parts)
    assert "## Literature" not in flat


# --- Curve planner — identification-mode guard -------------------------------

def _make_curve_planner():
    """Construct a HumanFeedbackRefinementController with stub deps + a model
    that captures the prompt parts and returns a parseable JSON response."""
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        HumanFeedbackRefinementController,
    )

    captured: dict = {"parts": None}

    class _CaptureModel:
        def generate_content(self, parts, generation_config=None):
            captured["parts"] = parts
            payload = json.dumps({
                "observations": "stub",
                "analysis_approach": "stub approach",
                "physical_model": "stub model",
                "parameters_to_extract": [],
                "fitting_strategy": "stub strategy",
                "literature_query": None,
            })
            return SimpleNamespace(text=payload)

    controller = HumanFeedbackRefinementController(
        model=_CaptureModel(),
        logger=logging.getLogger("test"),
        generation_config=None,
        safety_settings=None,
        parse_fn=lambda r: (json.loads(r.text), None),
        instructions="STUB_INSTRUCTIONS",
        output_dir="/tmp",
    )
    return controller, captured


def _minimal_curve_state(
    literature_context: str | None,
    task_mode: str | None,
) -> dict:
    # Render a tiny real PNG so the bytes field is plausibly valid.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)

    return {
        "original_plot_bytes": buf.getvalue(),
        "data_statistics": {"n_points": 200},
        "system_info": {"xlabel": "E", "ylabel": "I"},
        "is_single_spectrum": True,
        "num_spectra": 1,
        "literature_context": literature_context,
        "task_mode": task_mode,
    }


def test_curve_planner_injects_literature_in_fitting_mode():
    controller, captured = _make_curve_planner()
    state = _minimal_curve_state(
        literature_context="DOMAIN_LIT_CONTENT",
        task_mode="fitting",
    )
    controller._plan_analysis(state)
    flat = _stringify_parts(captured["parts"])
    assert "## Literature" in flat
    assert "DOMAIN_LIT_CONTENT" in flat


def test_curve_planner_withholds_literature_in_identification_mode():
    """Load-bearing guard: identification-mode planner must NOT see lit
    context, even when `state["literature_context"]` is populated."""
    controller, captured = _make_curve_planner()
    state = _minimal_curve_state(
        literature_context="DOMAIN_LIT_CONTENT_THAT_MUST_BE_HIDDEN",
        task_mode="identification",
    )
    controller._plan_analysis(state)
    flat = _stringify_parts(captured["parts"])
    assert "## Literature" not in flat
    assert "DOMAIN_LIT_CONTENT_THAT_MUST_BE_HIDDEN" not in flat


def test_curve_planner_omits_literature_when_absent():
    """No `literature_context` set → no section regardless of mode."""
    for mode in ("fitting", "identification", None):
        controller, captured = _make_curve_planner()
        state = _minimal_curve_state(literature_context=None, task_mode=mode)
        controller._plan_analysis(state)
        flat = _stringify_parts(captured["parts"])
        assert "## Literature" not in flat, f"unexpected lit injection in mode={mode!r}"


# --- Curve synthesis — unconditional lit injection ---------------------------

def test_curve_synthesis_gates_literature_in_id_mode():
    """`UnifiedCurveSynthesisController` Stage-1 appends `## Literature` for
    non-ID runs (opportunistic Channel-A reuse, issue #323 D1) but MUST gate
    it out in identification mode (D2: ID runs are literature-free in-run;
    literature enters only post-fit via `refine_interpretation`).

    Construct-then-call is prohibitively heavy for this controller (many
    coupled state keys); a source-level check is the right granularity for
    a regression guard on a one-line conditional.
    """
    from scilink.agents.exp_agents.controllers import curve_fitting_controllers
    src = inspect.getsource(curve_fitting_controllers.UnifiedCurveSynthesisController)
    lit_gates = [
        line for line in src.splitlines()
        if 'state.get("literature_context")' in line
    ]
    assert lit_gates, "expected at least one lit-context conditional in synthesis"
    for line in lit_gates:
        assert 'task_mode' in line and '"identification"' in line, (
            f"synthesis lit injection missing the identification-mode gate: {line!r}"
        )
