"""#569 — capability availability of the script-execution environment: the
probe runs THROUGH the executor (the subprocess interpreter, not the parent
process), is cached, degrades to "unknown" instead of raising, renders a
prompt block with the advisory-with-fallback rule, and the SAM tool reports
structured unavailability instead of a raw ImportError."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scilink.agents.exp_agents import _exec_env as ee


class _Exec:
    """A fake ScriptExecutor: records the script it ran, returns canned stdout."""
    def __init__(self, stdout=None, status="success", message=None):
        self.scripts = []
        self.stdout, self.status, self.message = stdout, status, message

    def execute_script(self, script, working_dir=None, timeout=None):
        self.scripts.append(script)
        if self.status != "success":
            return {"status": "error", "message": self.message or "boom"}
        return {"status": "success", "stdout": self.stdout, "stderr": ""}


def _stdout(packages, cuda=False, mps=True, ckpts=()):
    return "noise\n" + ee._MARKER + json.dumps({
        "python": "/envs/exec/bin/python", "python_version": "3.12.1",
        "packages": packages, "devices": {"cuda": cuda, "mps": mps},
        "sam_checkpoints": list(ckpts), "checkpoint_dir": "/c"}) + "\n"


@pytest.fixture(autouse=True)
def _fresh_cache():
    ee.reset_cache()
    yield
    ee.reset_cache()


def test_probe_runs_through_the_executor_and_is_cached():
    ex = _Exec(_stdout({"segment_anything": None, "torch": "2.1", "scipy": "1.11"}))
    env = ee.probe_execution_environment(ex)
    assert env["status"] == "ok" and env["python"] == "/envs/exec/bin/python"
    assert env["packages"]["segment_anything"] is None and env["packages"]["torch"] == "2.1"
    # the probe is a script the executor ran, importing in the subprocess
    assert len(ex.scripts) == 1 and "importlib.import_module" in ex.scripts[0]
    assert ee.capability_available(env, "segment_anything") is False
    assert ee.capability_available(env, "torch") is True
    assert ee.capability_available(env, "mps") is True and ee.capability_available(env, "gpu") is True
    assert ee.capability_available(env, "sam_checkpoint") is False
    # cached: a second call does not re-run
    ee.probe_execution_environment(ex)
    assert len(ex.scripts) == 1
    ee.probe_execution_environment(ex, force=True)
    assert len(ex.scripts) == 2


def test_probe_failure_degrades_to_unknown():
    env = ee.probe_execution_environment(_Exec(status="error", message="timed out"))
    assert env["status"] == "unknown" and "timed out" in env["error"]
    assert ee.capability_available(env, "torch") is None
    block = ee.exec_env_block(env)
    assert "Unknown" in block and "do not hard-mandate" in block
    # a probe that printed nothing usable
    env = ee.probe_execution_environment(_Exec("hello"), force=True)
    assert env["status"] == "unknown"


def test_block_lists_present_absent_devices_and_the_rule():
    env = ee.probe_execution_environment(_Exec(_stdout(
        {"segment_anything": "present", "torch": "2.1", "hyperspy": None, "cv2": "4.9"},
        cuda=False, mps=False, ckpts=["sam_vit_h.pth"])))
    block = ee.exec_env_block(env, for_verifier=True)
    assert "Compute: CPU only" in block
    assert "ABSENT optional packages: HyperSpy" in block
    assert "SAM (segment_anything)" in block and "sam_vit_h.pth" in block
    assert "only prescribe or mandate a capability listed as available" in block
    assert "status='unavailable' has said all it can" in block
    assert "said all it can" not in ee.exec_env_block(env)  # verifier-only sentence


def test_sam_tool_reports_structured_unavailability(monkeypatch):
    from scilink.skills._shared import sam
    def _raise(params):
        raise ImportError("No module named 'segment_anything'")
    monkeypatch.setattr(sam, "get_or_create_sam_model", _raise)
    out = sam.run_sam_analysis(np.zeros((16, 16), dtype=np.uint8), {"model_type": "vit_h"})
    assert out["status"] == "unavailable" and out["capability"] == "segment_anything"
    assert "segment_anything" in out["reason"] and out["total_count"] == 0 and out["masks"] == []
    assert "status='unavailable'" in sam.TOOL_SPEC.returns


def test_image_prompts_carry_the_environment_block(monkeypatch):
    """Planner, codegen-refinement and verifier prompts all include it."""
    from scilink.agents.exp_agents.controllers import image_analysis_controllers as ic
    env = ee.probe_execution_environment(_Exec(_stdout({"segment_anything": None, "torch": "2.1"})))
    monkeypatch.setattr(ic, "exec_env_block", lambda **kw: ee.exec_env_block(env, for_verifier=kw.get("for_verifier", False)))
    host = ic.UnifiedImageProcessingController.__new__(ic.UnifiedImageProcessingController)
    import logging
    host.logger = logging.getLogger("t")
    prompts = []
    host.model = SimpleNamespace(generate_content=lambda **kw: prompts.append(kw["contents"][0]) or SimpleNamespace(text="{}"))
    host.generation_config = None; host.safety_settings = None
    host._parse = lambda r: ({}, "no json")
    state = {"locked_analysis_config": {"processing_pipeline": "p", "analysis_approach": "a"}, "_annealing_level": 0}
    host._apply_verification_feedback(state, {"recommended_action": "use SAM", "issues_found": []}, history=[])
    assert "ABSENT optional packages: SAM (segment_anything)" in prompts[0]
    assert "said all it can" in prompts[0]
    planner = ic.ImagePlanningController.__new__(ic.ImagePlanningController)
    planner._get_instructions = lambda st: "instructions"
    planner.logger = logging.getLogger("t")
    prompt = planner._build_planning_prompt({"original_image_bytes": b"x", "image_statistics": {},
                                             "system_info": {}, "is_single_image": True})
    assert any(isinstance(p, str) and "Execution environment" in p for p in prompt)
