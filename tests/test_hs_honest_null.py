"""An honest null stands beside diagnostic maps (#723, item 1).

The hyperspectral controller honoured a ``not_measurable`` declaration only
when the script returned NO maps. Live, every unit of a series returned its
required maps all-NaN, a declaration with numeric evidence and one diagnostic
mask: the declaration was ignored, the judge never called, and the all-NaN
critique ("nothing was fitted ... remove try/except") pushed the model away
from the honest null for five rounds per unit. Through the real
``RunDynamicAnalysisController``: the declaration now stands when every
REQUIRED output is absent or entirely NaN; the diagnostics are recorded with
the determination and not committed; the contradiction repair still applies;
and a declaration beside a VALUED required output is critiqued as such.

  conda run -n scilink python -m pytest tests/test_hs_honest_null.py -q
"""
import json
import logging
import os

import numpy as np

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents.controllers.hyperspectral_controllers import RunDynamicAnalysisController

def _null_with_mask(window=None):
    """All-NaN required map + a diagnostic mask + a declaration examining
    ``window`` (no window key when None)."""
    win = f"'window': {list(window)}, " if window is not None else ""
    return (
        "def analyze_feature(data, energy_axis):\n"
        "    import numpy as np\n"
        "    d = np.asarray(data)\n"
        "    return {'maps': {'Edge_Position': np.full(d.shape[:2], np.nan), 'Fit_Mask': np.ones(d.shape[:2])},\n"
        f"            'not_measurable': {{'feature': 'edge position', {win}'evidence': 'prominence 0.4 sigma of the mean',\n"
        "                               'description': 'no edge above the noise'}}\n")


NULL_WITH_MASK = _null_with_mask((540, 565))
VALUED_WITH_DECLARATION = (
    "def analyze_feature(data, energy_axis):\n"
    "    import numpy as np\n"
    "    d = np.asarray(data)\n"
    "    return {'maps': {'Edge_Position': d.mean(axis=2)},\n"
    "            'not_measurable': {'feature': 'edge position', 'evidence': 'weak', 'description': 'unclear'}}\n")


def _flat_cube():
    rng = np.random.default_rng(0)
    E = np.linspace(450.0, 570.0, 96)
    return (10.0 + rng.normal(0, 1.0, (6, 6, E.size))).astype(np.float32), E


def _peaked_cube():
    rng = np.random.default_rng(1)
    E = np.linspace(450.0, 570.0, 96)
    pk = 80.0 * np.exp(-0.5 * ((E - 510.0) / 4.0) ** 2)
    return (10.0 + pk[None, None, :] + rng.normal(0, 1.0, (6, 6, E.size))).astype(np.float32), E


def _run(tmp_path, cube, E, scripts, review=(True, ""), judge=None, max_iter=0):
    prompts = []
    answers = iter(scripts)
    last = {"code": scripts[-1]}

    class _Model:
        def generate_content(self, contents, **kw):
            text = contents if isinstance(contents, str) else "\n".join(c for c in contents if isinstance(c, str))
            prompts.append(text)
            try:
                last["code"] = next(answers)
            except StopIteration:
                pass
            return json.dumps({"code": last["code"]})
    ctrl = RunDynamicAnalysisController(model=_Model(), logger=logging.getLogger("t"),
                                        generation_config=None, safety_settings=None,
                                        parse_fn=lambda r: (json.loads(r), None), executor_timeout=60)
    ctrl._review_required_output = lambda *a, **k: review
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    judged = []
    if judge is not None:
        ctrl._judge_not_measurable = lambda nm, ctx: judged.append(nm) or judge
    state = ctrl.execute({
        "refinement_decision": {"refinement_needed": True, "requires_custom_code": True,
                                "targets": [{"type": "custom_code", "description": "map the edge position",
                                             "required_outputs": ["Edge_Position"]}]},
        "hspy_data": cube, "original_hspy_data": cube, "energy_axis": E,
        "system_info": {"axis_spec": {"axis_2": {"name": "E", "units": "eV", "start": 450, "end": 570}}},
        "settings": {"output_dir": str(tmp_path)}, "iteration_title": "T", "analysis_images": [],
        "error_dict": None, "max_verification_iterations": max_iter})
    return state, prompts, judged


def test_a_declaration_beside_a_diagnostic_mask_is_judged_and_accepted(tmp_path):
    cube, E = _flat_cube()
    state, prompts, judged = _run(tmp_path, cube, E, [NULL_WITH_MASK], judge=(True, ""))
    assert len(judged) == 1                                   # the judge was asked
    assert [d["name"] for d in judged[0]["diagnostic_maps"]] == ["Edge_Position", "Fit_Mask"]
    rec = state["dynamic_analysis_records"][0]
    assert rec["task_success"] is True
    meta = state["custom_analysis_metadata_list"]
    # the determination is reported with its diagnostics named; the mask is
    # not committed as a feature (no map review looked at it)
    assert [m.get("determination") for m in meta] == ["not measurable in this dataset (judged honest null)"]
    assert meta[0]["diagnostic_maps"] == ["Edge_Position", "Fit_Mask"]
    assert not any(m.get("name") == "Fit_Mask" for m in meta)
    assert len(prompts) == 1                                  # no retry


def test_a_strong_feature_inside_the_declared_window_is_repaired_in_place(tmp_path):
    """The facts show a >= 5 sigma peak at 510 eV INSIDE the window the
    declaration examined: a wrong gate, repaired in place (GATE ERROR) with the
    feature and the window named, the judge never asked."""
    cube, E = _peaked_cube()
    good = ("def analyze_feature(data, energy_axis):\n"
            "    import numpy as np\n"
            "    return {'maps': {'Edge_Position': np.asarray(data).mean(axis=2)}, 'units': 'eV', 'description': 'd'}\n")
    state, prompts, judged = _run(tmp_path, cube, E, [_null_with_mask((500, 520)), good],
                                  judge=(AssertionError("judge must not be called"), ""))
    assert judged == [] and len(prompts) == 2 and "GATE ERROR" in prompts[1]
    assert "INSIDE that window" in prompts[1] and "[500, 520]" in prompts[1]
    assert state["dynamic_analysis_records"][0]["task_success"] is True


def test_a_strong_feature_outside_the_declared_window_is_judged_not_repaired(tmp_path):
    """The #735 live case: a band elsewhere in the cube (510 eV) does not
    contradict "the feature I looked for at 540-565 is absent". The judge
    decides; the repair, which once steered the model onto the other band,
    is not issued."""
    cube, E = _peaked_cube()
    state, prompts, judged = _run(tmp_path, cube, E, [_null_with_mask((540, 565))], judge=(True, ""))
    assert len(judged) == 1 and len(prompts) == 1 and not any("GATE ERROR" in p for p in prompts)
    assert state["dynamic_analysis_records"][0]["task_success"] is True


def test_a_declaration_without_a_window_goes_to_the_judge(tmp_path):
    """Nothing to hold it to: no repair, the judge decides."""
    cube, E = _peaked_cube()
    state, prompts, judged = _run(tmp_path, cube, E, [_null_with_mask(None)], judge=(True, ""))
    assert len(judged) == 1 and not any("GATE ERROR" in p for p in prompts)


def test_a_declaration_beside_a_valued_required_output_is_critiqued_as_such(tmp_path):
    """A required output came back WITH values and failed its review: the
    retry critique says the declaration was not honoured and why — not the
    all-NaN "remove try/except" text."""
    cube, E = _flat_cube()
    state, prompts, judged = _run(tmp_path, cube, E, [VALUED_WITH_DECLARATION, NULL_WITH_MASK],
                                  review=(False, "Edge_Position is noise"), judge=(True, ""), max_iter=3)
    retry = next(p for p in prompts[1:] if "Required outputs failed" in p)
    assert "NOT honoured" in retry and "['Edge_Position'] came back with values" in retry
    assert "remove blanket exception handling" not in retry.lower()
