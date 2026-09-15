"""Golden pin for the hyperspectral dynamic-analysis loop (#327 phase 5).

Runs ``RunDynamicAnalysisController.execute`` with a ScriptedModel and pins,
per scenario: every prompt sent to the model (in order), the per-target
``dynamic_analysis_records``, the committed feature metadata, the saved
image labels/filenames (timestamps normalized), and the degradation notes.

Captured on the PRE-engine-port controller; the phase-5 port must reproduce
these byte-identically. Regenerate: ``QC_GOLDEN_UPDATE=1 pytest tests/qc_golden``.
"""

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

# UNSAFE_EXECUTION_OK is set per-test by the package conftest's _golden_env
# fixture (monkeypatch) — no module-level env writes that leak process-wide.
sys.path.insert(0, str(Path(__file__).parent.parent))

from qc_golden.harness import (  # noqa: E402
    Rule,
    ScriptedModel,
    check_golden,
    make_normalizer,
    normalize_obj,
)

from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (  # noqa: E402
    RunDynamicAnalysisController,
)

logging.basicConfig(level=logging.INFO)

# Prompt-routing markers (distinctive phrases from instruct.py templates).
_VISUAL_QC_MARKER = "You are a Quality Assurance Scientist"
_COMBINED_REVIEW_MARKER = "SINGLE combined review"
_SALVAGE_JUDGE_MARKER = "FINAL salvage decision"
_CODEGEN_MARKER = "analyze_feature"

_TS_RE = re.compile(r"\d{8}_\d{6}")

# Retry-feedback prompts embed traceback.format_exc(), whose frame names and
# line numbers shift with ANY edit to the controller file — volatile by
# nature, so goldens mask them (flagged per plan §8.1: the masked span is
# exactly `File "...", line N, in <frame>`; the exception text itself stays
# pinned).
_TB_FRAME_RE = re.compile(r'File "[^"]+", line \d+, in \S+')

GOOD_CODE = """\
def analyze_feature(data, energy_axis):
    import numpy as np
    return {
        "maps": {"mean_map": np.asarray(data).mean(axis=2)},
        "units": "a.u.",
        "description": "per-pixel mean intensity",
    }
"""

BROKEN_CODE = """\
def analyze_feature(data, energy_axis):
    raise ValueError("global fit saturated - method inadequate")
"""

# Required output plus a passing diagnostic map — for the salvage scenario the
# combined review keeps rejecting `edge_jump` while visual QC passes `mean_map`.
TWO_MAP_CODE = """\
def analyze_feature(data, energy_axis):
    import numpy as np
    arr = np.asarray(data)
    return {
        "maps": {
            "edge_jump": arr[:, :, -1] - arr[:, :, 0],
            "mean_map": arr.mean(axis=2),
        },
        "units": {"edge_jump": "a.u.", "mean_map": "a.u."},
        "description": "edge jump and mean intensity",
    }
"""


def _parse(resp):
    return json.loads(resp.text), None


def _make_state(tmp_path, target):
    cube = np.random.default_rng(0).random((6, 5, 12))
    return {
        "refinement_decision": {
            "refinement_needed": True,
            "requires_custom_code": True,
            "targets": [target],
        },
        "hspy_data": cube,
        "original_hspy_data": cube,
        "system_info": {"technique": "synthetic test cube"},
        "settings": {"output_dir": str(tmp_path)},
        "iteration_title": "Golden_Iter",
        "analysis_images": [],
        "error_dict": None,
        "analysis_objective": "map the diagnostic feature per pixel",
    }


def _run(tmp_path, target, rules):
    base_norm = make_normalizer({str(tmp_path): "<OUTDIR>"})

    def norm(text):
        text = _TB_FRAME_RE.sub('File "<SRC>", line <N>, in <FRAME>', text)
        return _TS_RE.sub("<TS>", base_norm(text))

    model = ScriptedModel(rules, normalizer=norm)
    ctrl = RunDynamicAnalysisController(
        model=model, logger=logging.getLogger("hs_golden"),
        generation_config=None, safety_settings=None, parse_fn=_parse,
        executor_timeout=60,
    )
    state = _make_state(tmp_path, target)
    state = ctrl.execute(state)
    payload = {
        "calls": model.calls,
        "records": normalize_obj(state.get("dynamic_analysis_records"), norm),
        "meta": normalize_obj(state.get("custom_analysis_metadata_list"), norm),
        "images": [
            {"label": it.get("label"), "filename": norm(it.get("filename", ""))}
            for it in state.get("analysis_images", [])
        ],
        "degradation_notes": normalize_obj(state.get("degradation_notes"), norm),
        "failed_flag": state.get("dynamic_analysis_failed"),
    }
    return payload


def test_hs_golden_happy(tmp_path):
    rules = [
        Rule("visual_qc", _VISUAL_QC_MARKER, [{"valid": True}], repeat_last=True),
        Rule("codegen", _CODEGEN_MARKER, [{"code": GOOD_CODE}], repeat_last=True),
    ]
    target = {"type": "custom_code",
              "description": "map the mean intensity per pixel"}
    payload = _run(tmp_path, target, rules)
    assert payload["records"][0]["task_success"] is True
    check_golden("hs_happy", payload)


def test_hs_golden_retry_ladder(tmp_path):
    # 3 QC-rejected attempts (levels 0, 1, 1) then acceptance at level 2
    # (hot); pins the annealed retry-feedback prompts (critique + script
    # anchor at warm, no anchor at hot) and the record levels. QC
    # rejections, not execution errors, drive the ladder: execution errors
    # are now repaired by the in-attempt mechanical-correction loop
    # (curve/image parity) and never advance the annealing level — see
    # test_hs_golden_exec_repair.
    rules = [
        Rule("visual_qc", _VISUAL_QC_MARKER,
             [{"valid": False,
               "critique": "map is spatially incoherent noise - "
                           "the estimator is not extracting the feature"}] * 3
             + [{"valid": True}],
             repeat_last=True),
        Rule("codegen", _CODEGEN_MARKER, [{"code": GOOD_CODE}],
             repeat_last=True),
    ]
    target = {"type": "custom_code",
              "description": "map the mean intensity per pixel"}
    payload = _run(tmp_path, target, rules)
    rec = payload["records"][0]
    assert rec["task_success"] is True
    levels = [it["annealing_level"]
              for it in rec["quality_history"]["verification_iterations"]]
    assert levels == [0, 1, 1, 2]
    check_golden("hs_ladder", payload)


def test_hs_golden_exec_repair(tmp_path):
    # An execution error is repaired by the in-attempt mechanical
    # correction: one ladder attempt, level 0, exec_corrections recorded —
    # pins the MECHANICAL CORRECTION prompt (traceback + repair-not-
    # redesign instruction).
    rules = [
        Rule("visual_qc", _VISUAL_QC_MARKER, [{"valid": True}],
             repeat_last=True),
        Rule("codegen", _CODEGEN_MARKER,
             [{"code": BROKEN_CODE}, {"code": GOOD_CODE}],
             repeat_last=True),
    ]
    target = {"type": "custom_code",
              "description": "map the mean intensity per pixel"}
    payload = _run(tmp_path, target, rules)
    rec = payload["records"][0]
    assert rec["task_success"] is True
    its = rec["quality_history"]["verification_iterations"]
    assert [it["annealing_level"] for it in its] == [0]
    assert its[0]["exec_corrections"] == 1
    check_golden("hs_exec_repair", payload)


def test_hs_golden_salvage_approximate(tmp_path):
    # The required output never passes the combined review (2-of-3 reject
    # votes each attempt); the diagnostic map keeps passing visual QC. All
    # attempts fail the required-outputs gate; the salvage judge presents
    # the best partial as an APPROXIMATE result with a caveat.
    rules = [
        Rule("salvage", _SALVAGE_JUDGE_MARKER,
             [{"present": True, "confidence": "medium",
               "caveat": "edge amplitude unverified; treat as approximate"}]),
        Rule("combined_review", _COMBINED_REVIEW_MARKER,
             [{"valid": False, "critique": "edge_jump map is indistinguishable from noise"}],
             repeat_last=True),
        Rule("visual_qc", _VISUAL_QC_MARKER, [{"valid": True}], repeat_last=True),
        Rule("codegen", _CODEGEN_MARKER, [{"code": TWO_MAP_CODE}], repeat_last=True),
    ]
    target = {"type": "custom_code",
              "description": "quantify the edge jump per pixel",
              "required_outputs": ["edge_jump"]}
    payload = _run(tmp_path, target, rules)
    rec = payload["records"][0]
    assert rec["task_success"] is False
    assert rec["salvaged"] is True
    assert payload["degradation_notes"][0]["kind"] == "approximate"
    # only the passing diagnostic map was committed, marked approximate
    assert payload["meta"][0]["name"] == "mean_map"
    assert payload["meta"][0]["description"].startswith("[APPROXIMATE")
    check_golden("hs_salvage", payload)
