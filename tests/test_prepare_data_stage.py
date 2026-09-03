"""Orchestrator data-preparation stage: raw-instrument detection, the
generate-run-verify engine with a fake model, and the prepare_data tool."""
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["UNSAFE_EXECUTION_OK"] = "true"

from scilink.agents.exp_agents.data_preparation import (  # noqa: E402
    detect_raw_instrument, build_prep_context, run_preparation, static_guard,
    parse_result_marker, check_products, RESULT_MARKER)
from scilink.executors import ScriptExecutor  # noqa: E402


def _bundle(tmp_path: Path) -> Path:
    b = tmp_path / "bundle"; (b / "raw").mkdir(parents=True)
    (b / "reconstruction_manifest.json").write_text(json.dumps({
        "authoritative_boundary": "raw_temporal_hologram", "records": [{"workflow": "WF001", "raw_hdf5": "raw/a.h5"}]}))
    (b / "raw" / "a.json").write_text(json.dumps({"measurement_type": "off_axis_temporal_hologram_stack",
                                                  "generic_image_routing_permitted": False}))
    (b / "raw" / "a.h5").write_bytes(b"")
    return b


def test_detect_raw_instrument_dir_file_and_negative(tmp_path):
    b = _bundle(tmp_path)
    d = detect_raw_instrument(b)
    assert d and d["data_type"] == "raw_instrument" and d["manifest"].endswith("reconstruction_manifest.json")
    f = detect_raw_instrument(b / "raw" / "a.h5")
    assert f and "generic_image_routing_permitted" in " ".join(f["evidence"])
    plain = tmp_path / "plain"; plain.mkdir(); np.save(plain / "img.npy", np.zeros((64, 64)))
    (plain / "img.json").write_text(json.dumps({"technique": "SEM", "measurement_type": "microscopy"}))
    assert detect_raw_instrument(plain) is None and detect_raw_instrument(plain / "img.npy") is None


def test_context_lists_files_and_manifest(tmp_path):
    b = _bundle(tmp_path)
    ctx = build_prep_context(b, detect_raw_instrument(b))
    assert "reconstruction_manifest.json" in ctx and "raw/a.h5" in ctx and "RAW-INSTRUMENT EVIDENCE" in ctx


def test_static_guard_and_marker():
    assert static_guard("import subprocess\n_PREP") and static_guard("x=1") and static_guard("import os\nos.system('x')\n_PREP")
    assert static_guard("import numpy as np\np=_PREP['out_dir']") is None
    assert parse_result_marker("noise")[0] is None
    assert parse_result_marker(RESULT_MARKER + '{"a": 1}')[0] == {"a": 1}


GOOD_SCRIPT = '''
import os, json, numpy as np
out = _PREP["out_dir"]
m = np.zeros((20, 30), dtype=np.float32); p = os.path.join(out, "map.npy"); np.save(p, m)
json.dump({"kind": "image", "units": "rad"}, open(os.path.join(out, "map.json"), "w"))
c = os.path.join(out, "trace.csv"); open(c, "w").write("elapsed_s,phase\\n0,0\\n1,1\\n")
print("PREP_RESULT_JSON:" + json.dumps({"products": [
    {"path": p, "kind": "image", "sidecar": os.path.join(out, "map.json"), "description": "map"},
    {"path": c, "kind": "curve", "sidecar": None, "description": "trace"}],
    "qc": {"passed": True, "metrics": {"coherence": 0.97}, "notes": []}, "receipts": [], "summary": "ok"}))
'''
BAD_SCRIPT = "import numpy as np\nprint(_PREP['out_dir'])\n"   # no marker


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies); self.calls = []

    def generate_content(self, prompt, **kw):
        self.calls.append(prompt)
        class R: pass
        r = R(); r.text = self.replies.pop(0) if self.replies else '{"verdict": "pass", "reasons": []}'
        return r


def test_engine_retries_then_succeeds(tmp_path):
    model = FakeModel(["```python\n" + BAD_SCRIPT + "```", "```python\n" + GOOD_SCRIPT + "```",
                       '{"verdict": "pass", "reasons": ["gate met"]}'])
    out = tmp_path / "prep"
    skill = {"name": "dummy", "meta": {"description": "d"}, "planning": "p", "implementation": "i",
             "validation": "products must exist"}
    res = run_preparation(model=model, executor=ScriptExecutor(timeout=60), data_path=str(tmp_path),
                          task="prepare", out_dir=out, scratch_dir=out / "_scratch", context="ctx",
                          skill=skill, tool_inventory="", max_attempts=3, llm_verify=True)
    assert res["status"] == "success" and res["attempts"] == 2
    assert {p["kind"] for p in res["products"]} == {"image", "curve"}
    assert Path(res["products"][1]["sidecar"]).is_file()        # synthesized sidecar for the curve
    assert (out / "scripts" / "prepare_script.py").is_file() and (out / "analysis_results.json").is_file()
    ar = json.loads((out / "analysis_results.json").read_text())
    assert ar["agent_type"] == "data_preparation" and ar["extracted_features"] == {"coherence": 0.97}
    assert "PREVIOUS ATTEMPT FAILED" in model.calls[1]


def test_engine_rejects_failed_qc(tmp_path):
    bad_qc = GOOD_SCRIPT.replace('"passed": True', '"passed": False')
    model = FakeModel(["```python\n" + bad_qc + "```"] * 2)
    res = run_preparation(model=model, executor=ScriptExecutor(timeout=60), data_path=str(tmp_path),
                          task="t", out_dir=tmp_path / "p", scratch_dir=tmp_path / "s", context="c",
                          skill=None, max_attempts=2, llm_verify=False)
    assert res["status"] == "error" and "qc.passed" in res["message"]


def test_engine_fails_llm_verification_then_gives_up(tmp_path):
    model = FakeModel(["```python\n" + GOOD_SCRIPT + "```", '{"verdict": "fail", "required_fixes": ["report coverage"]}',
                       '{"verdict": "fail", "required_fixes": ["report coverage"]}',
                       "```python\n" + GOOD_SCRIPT + "```", '{"verdict": "fail", "required_fixes": ["still"]}',
                       '{"verdict": "fail", "required_fixes": ["still"]}'])
    skill = {"name": "d", "meta": {}, "validation": "must report coverage"}
    res = run_preparation(model=model, executor=ScriptExecutor(timeout=60), data_path=str(tmp_path),
                          task="t", out_dir=tmp_path / "p", scratch_dir=tmp_path / "s", context="c",
                          skill=skill, max_attempts=2, llm_verify=True)
    assert res["status"] == "error" and "verification failed" in res["message"]


def test_check_products_rejects_outside_dir_and_wrong_ndim(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    np.save(out / "cube.npy", np.zeros((2, 3, 4))); np.save(tmp_path / "ext.npy", np.zeros((2, 2)))
    problems, clean = check_products({"products": [
        {"path": str(out / "cube.npy"), "kind": "image"}, {"path": str(tmp_path / "ext.npy"), "kind": "image"}],
        "qc": {"passed": True, "metrics": {"x": "no"}}}, out)
    assert any("ndim" in p for p in problems) and any("outside" in p for p in problems) and any("not a number" in p for p in problems)


@pytest.fixture
def orch(tmp_path):
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent, AnalysisMode
    with contextlib.redirect_stdout(io.StringIO()):
        ag = AnalysisOrchestratorAgent(base_dir=str(tmp_path / "session"), api_key="sk-dummy",
                                       model_name="claude-opus-4-6", analysis_mode=AnalysisMode.AUTONOMOUS,
                                       restore_checkpoint=False)
    return ag


def test_examine_data_routes_raw_instrument(orch, tmp_path):
    b = _bundle(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        r = json.loads(orch.tools.execute_tool("examine_data", data_path=str(b)))
    assert r["data_type"] == "raw_instrument" and r["preparation_required"] is True and r["next_tool"] == "prepare_data"
    assert r["suggested_agents"] == [] and orch.current_data_type == "raw_instrument"
    names = [s["function"]["name"] for s in orch.tools.openai_schemas]
    assert "prepare_data" in names
    prompt = orch.get_system_prompt() if callable(getattr(orch, "get_system_prompt", None)) else orch.messages[0]["content"]
    assert "raw_instrument" in prompt and "prepare_data" in prompt


def test_prepare_data_tool_end_to_end(orch, tmp_path, monkeypatch):
    b = _bundle(tmp_path)
    fake = FakeModel(["```python\n" + GOOD_SCRIPT + "```", '{"verdict": "pass", "reasons": []}'])
    monkeypatch.setattr(orch.tools, "_internal_model", lambda: fake)
    with contextlib.redirect_stdout(io.StringIO()):
        r = json.loads(orch.tools.execute_tool("prepare_data", data_path=str(b), task="reconstruct it",
                                               skill="mmzi_hologram_reconstruction", max_attempts=2))
    assert r["status"] == "success", r
    assert r["skill_used"] == "mmzi_hologram_reconstruction" and len(r["files_produced"]) == 2
    assert Path(r["output_directory"]).name.startswith("prepare_")
    assert orch.analysis_results[-1]["agent_id"] == "prepare" and orch.analysis_results[-1]["status"] == "success"
    # the skill's recipe and both helper tools reached the code-writing prompt
    assert "reconstruct_offaxis_hologram_stack" in fake.calls[0] and "derive_phase_products" in fake.calls[0]
    assert "Implementation recipe" in fake.calls[0] and "_PREP" in fake.calls[0]


def test_run_analysis_skill_menu_excludes_preparation_skills():
    from scilink.agents.exp_agents.analysis_orchestrator_tools import _build_skill_description
    assert "mmzi_hologram_reconstruction" not in _build_skill_description()


class MangledModel(FakeModel):
    """Mimics the LiteLLM wrapper: `.text` is the JSON-extracted view (sliced
    from the first '{' to the last '}'), `raw_text` is the full reply (#238)."""
    def generate_content(self, prompt, **kw):
        r = super().generate_content(prompt, **kw)
        full = r.text
        i, j = full.find("{"), full.rfind("}")
        r.raw_text = full
        r.text = full[i:j + 1] if 0 <= i < j else full
        return r


def test_engine_reads_raw_text_not_cleaned_text(tmp_path):
    model = MangledModel(["```python\n" + GOOD_SCRIPT + "```"])
    res = run_preparation(model=model, executor=ScriptExecutor(timeout=60), data_path=str(tmp_path),
                          task="t", out_dir=tmp_path / "p", scratch_dir=tmp_path / "s", context="c",
                          skill=None, max_attempts=1, llm_verify=False)
    assert res["status"] == "success", res


def test_check_products_accepts_provenance_stack(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    np.save(out / "stack.npy", np.zeros((3, 4, 5), dtype=np.float32))
    problems, clean = check_products({"products": [{"path": str(out / "stack.npy"), "kind": "stack"}],
                                      "qc": {"passed": True, "metrics": {}}}, out)
    assert problems == [] and clean[0]["kind"] == "stack" and Path(clean[0]["sidecar"]).is_file()


def test_fanout_refuses_raw_instrument_branches(tmp_path):
    from types import SimpleNamespace
    from scilink.agents.meta_agent.fanout import raw_instrument_branches, run_fanout
    b = _bundle(tmp_path)
    plain = tmp_path / "plain"; plain.mkdir(); np.save(plain / "img.npy", np.zeros((64, 64)))
    branches = [{"data_path": str(b), "label": "raw holograms", "task": "t"},
                {"data_path": str(plain / "img.npy"), "label": "image", "task": "t"}]
    hits = raw_instrument_branches(branches)
    assert [h["label"] for h in hits] == ["raw holograms"]
    out = json.loads(run_fanout(SimpleNamespace(), branches))
    assert out["status"] == "declined" and out["reason"] == "raw_instrument_branch"
    assert "prepare_data" in out["message"] and out["raw_branches"][0]["label"] == "raw holograms"
    assert raw_instrument_branches([branches[1]]) == []


def test_single_vote_verification_failure_is_discarded(tmp_path):
    model = FakeModel(["```python\n" + GOOD_SCRIPT + "```", '{"verdict": "fail", "reasons": ["hallucinated"]}',
                       '{"verdict": "pass", "reasons": ["re-check: values meet the gate"]}'])
    skill = {"name": "d", "meta": {}, "validation": "gate >= 0.95"}
    res = run_preparation(model=model, executor=ScriptExecutor(timeout=60), data_path=str(tmp_path),
                          task="t", out_dir=tmp_path / "p", scratch_dir=tmp_path / "s", context="c",
                          skill=skill, max_attempts=1, llm_verify=True)
    assert res["status"] == "success" and res["attempts"] == 1 and len(model.calls) == 3


def test_user_stop_is_terminal_not_retried(tmp_path):
    class StoppingExecutor:
        calls = 0
        def execute_script(self, script, working_dir=None, timeout=None):
            StoppingExecutor.calls += 1
            return {"status": "error", "message": "Script execution was stopped by the user."}
    model = FakeModel(["```python\n" + GOOD_SCRIPT + "```"] * 3)
    res = run_preparation(model=model, executor=StoppingExecutor(), data_path=str(tmp_path), task="t",
                          out_dir=tmp_path / "p", scratch_dir=tmp_path / "s", context="c", skill=None,
                          max_attempts=3, llm_verify=False)
    assert res["status"] == "cancelled" and StoppingExecutor.calls == 1 and res["attempts"] == 1
