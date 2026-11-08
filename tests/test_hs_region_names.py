"""Hyperspectral regions are named by evidence, not by their contrast.

The planner that writes the analysis targets and every code-generation
attempt (first and retry, built on one base prompt) carry the principle:
what each actually sends to the model is checked, not only the constant.

  conda run -n scilink python -m pytest tests/test_hs_region_names.py -q
"""
import json
import logging
import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents._region_names import REGION_NAMES_PRINCIPLE
from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
    RunDynamicAnalysisController, SelectRefinementTargetController)

import test_hs_scalar_checks as tsc
import test_ungated_outputs as tu


class _Captured(Exception):
    pass


def test_the_planner_sends_the_principle_up_front():
    from scilink.agents.exp_agents.pipelines.hyperspectral_pipelines import (
        create_hyperspectral_iteration_pipeline)
    sent = []

    class Model:
        def generate_content(self, contents, **kw):
            sent.append("\n".join(p for p in contents if isinstance(p, str))
                        if isinstance(contents, list) else str(contents))
            raise _Captured()
    pipe = create_hyperspectral_iteration_pipeline(
        model=Model(), logger=logging.getLogger("t"), generation_config=None,
        safety_settings=None, parse_fn=lambda r: (json.loads(r), None), settings={},
        preprocessor=None)
    planner = next(c for c in pipe if isinstance(c, SelectRefinementTargetController))
    cube, _ = tu._cube()
    try:
        planner.execute({"skip_decomposition": True, "hspy_data": cube, "analysis_images": [],
                         "system_info": {"experiment": {"technique": "spectrum image"}},
                         "iteration_title": "T"})
    except _Captured:
        pass
    assert sent, "the planner made no model call"
    assert REGION_NAMES_PRINCIPLE in sent[0]
    assert sent[0].index(REGION_NAMES_PRINCIPLE) < 400


def test_every_code_generation_attempt_sends_the_principle(tmp_path):
    """The first attempt returns a required map all-NaN, so the retry runs;
    both prompts carry the principle."""
    import numpy as np
    prompts = []
    first = ("def analyze_feature(data, energy_axis):\n    import numpy as np\n"
             "    d = np.asarray(data)\n"
             "    return {'maps': {'Depth_Map': np.full(d.shape[:2], np.nan)}, 'units': 'a.u.', 'description': 'x'}\n")
    scripts = iter([first, tsc.SCRIPT_ALL_PASS])

    class Model:
        def generate_content(self, contents, **kw):
            prompts.append(contents if isinstance(contents, str) else json.dumps(contents, default=str))
            return json.dumps({"code": next(scripts)})
    cube, E = tu._cube()
    ctrl = RunDynamicAnalysisController(model=Model(), logger=logging.getLogger("t"),
                                        generation_config=None, safety_settings=None,
                                        parse_fn=lambda r: (json.loads(r), None), executor_timeout=60)
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    ctrl.execute({
        "refinement_decision": {"refinement_needed": True, "requires_custom_code": True,
                                "targets": [{"type": "custom_code", "description": "fit the band",
                                             "required_outputs": ["Depth_Map"]}]},
        "hspy_data": cube, "original_hspy_data": cube, "energy_axis": E,
        "system_info": {"axis_spec": {"axis_2": {"name": "wavelength", "units": "nm", "start": 400, "end": 900}}},
        "settings": {"output_dir": str(tmp_path)}, "iteration_title": "T", "analysis_images": [],
        "error_dict": None, "max_verification_iterations": 1})
    assert len(prompts) == 2 and "PREVIOUS ATTEMPT FAILED" in prompts[1]
    principle = json.dumps(REGION_NAMES_PRINCIPLE)[1:-1]     # as it appears in a JSON-dumped prompt
    assert all(REGION_NAMES_PRINCIPLE in p or principle in p for p in prompts)


def test_the_principle_names_no_technique():
    """A principle, not a trace: no technique, signal or sample is named."""
    text = REGION_NAMES_PRINCIPLE.lower()
    for word in ("eels", "plasmon", "raman", "haadf", "nanocrystal", "cube", "otsu"):
        assert word not in text


def _planner_prompt(state):
    """The planner's actual prompt (text parts and image parts) for ``state``."""
    from scilink.agents.exp_agents.pipelines.hyperspectral_pipelines import (
        create_hyperspectral_iteration_pipeline)
    sent = []

    class Model:
        def generate_content(self, contents, **kw):
            sent.append(contents)
            raise _Captured()
    pipe = create_hyperspectral_iteration_pipeline(
        model=Model(), logger=logging.getLogger("t"), generation_config=None,
        safety_settings=None, parse_fn=lambda r: (json.loads(r), None), settings={},
        preprocessor=None)
    planner = next(c for c in pipe if isinstance(c, SelectRefinementTargetController))
    try:
        planner.execute(state)
    except _Captured:
        pass
    parts = sent[0]
    text = "\n".join(p for p in parts if isinstance(p, str))
    images = [p for p in parts if isinstance(p, dict) and p.get("mime_type", "").startswith("image/")]
    return text, images


def test_a_planner_without_decomposition_sees_the_field():
    cube, _ = tu._cube()
    info = {"axis_spec": {"axis_2": {"name": "wavelength", "units": "nm", "start": 400, "end": 900}}}
    text, images = _planner_prompt({"skip_decomposition": True, "hspy_data": cube,
                                    "analysis_images": [], "system_info": info,
                                    "iteration_title": "T"})
    assert "FIELD OVERVIEW" in text and "(No visual results available)" not in text
    assert len(images) == 1 and images[0]["data"][:2] == b"\xff\xd8"          # one JPEG
    assert "DATA FACTS" in text                                               # the code writer's facts too


def test_a_planner_with_decomposition_maps_is_unchanged():
    cube, _ = tu._cube()
    shown = {"label": "Component 1", "bytes": b"\xff\xd8fake"}
    text, images = _planner_prompt({"hspy_data": cube, "analysis_images": [shown],
                                    "system_info": {"experiment": {"technique": "x"}},
                                    "iteration_title": "T"})
    assert "FIELD OVERVIEW" not in text and "DATA FACTS" not in text
    assert [i["data"] for i in images] == [b"\xff\xd8fake"]


def test_the_overview_never_breaks_planning():
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import _field_overview
    assert _field_overview({}) == {}
    assert _field_overview({"hspy_data": [[1.0]]}) == {}
