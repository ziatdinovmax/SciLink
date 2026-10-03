"""A claim is verified only when the outputs it may rest on passed a gate (#722).

A hyperspectral task's gate checks its MAPS (the per-map review on fresh code,
``map_health`` on a replay); the ``scalars`` it also reports pass no gate. A
live swarm run posted a claim as verified that rested on railed, fit-failed
scalars (positions at window edges, depths of 1e-12), a subscription fired on
it and fusion counted it. Here, through the real path — the dynamic-analysis
controller marks what it did not check, the agent stamps it, ``run_task``
carries it on the ``analyses`` row, the board reads it — a claim of such a run
is provisional with the reason, its recipe stays verified, and a run with
nothing ungated is unchanged. A derivation that failed its checks lists what
it left behind as unverified and is not in ``files_produced``.

  conda run -n scilink python -m pytest tests/test_ungated_outputs.py -q
"""
import contextlib
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents import hyperspectral_analysis_agent as hsa
from scilink.agents.exp_agents.controllers.hyperspectral_controllers import RunDynamicAnalysisController
from scilink.agents.meta_agent import board as board_mod
from scilink.agents.meta_agent import reactions

# a subscription as the meta declares one: on a claim, status left to its default
ON_A_CLAIM = reactions.normalize_subscriptions(
    [{"on": {"kind": "claim"}, "enqueue": {"mode": "planning", "task": "design a test of {finding.text}"}}])[0][0]

AXIS = {"technique": "UV-vis transmission", "sample": "film",
        "energy_range": {"start": 400.0, "end": 900.0, "units": "nm"}}

# A required map the gate reviews, and per-band fit results through the
# scalars channel — the live run's shape: railed at the window edge, depth ~0.
SCRIPT_WITH_SCALARS = (
    "def analyze_feature(data, energy_axis):\n"
    "    import numpy as np\n"
    "    d = np.asarray(data)\n"
    "    return {'maps': {'Depth_Map': d.mean(axis=2)}, 'units': 'a.u.', 'description': 'depth',\n"
    "            'scalars': {'Band1_Position_nm': 435.0003, 'Band1_Depth': 1e-12}}\n")
SCRIPT_MAPS_ONLY = (
    "def analyze_feature(data, energy_axis):\n"
    "    import numpy as np\n"
    "    return {'maps': {'Depth_Map': np.asarray(data).mean(axis=2)}, 'units': 'a.u.', 'description': 'depth'}\n")


def _cube(seed=0):
    rng = np.random.default_rng(seed)
    E = np.linspace(400.0, 900.0, 64)
    yy = np.linspace(0, 1, 6)[:, None, None]
    c = 1.0 - 0.3 * np.exp(-0.5 * ((E - 600.0) / 20.0) ** 2)[None, None, :] * (1 + yy) \
        + rng.normal(0, 0.005, (6, 6, 64))
    return c.astype(np.float32), E


def _controller_run(tmp_path, script):
    """The real dynamic-analysis controller: one task, the model answering with
    ``script``, the map review passing — what it commits is what the agent
    reports as ``extracted_features``."""
    class _Model:
        def generate_content(self, contents, **kw):
            return json.dumps({"code": script})
    cube, E = _cube()
    ctrl = RunDynamicAnalysisController(model=_Model(), logger=logging.getLogger("t"),
                                        generation_config=None, safety_settings=None,
                                        parse_fn=lambda r: (json.loads(r), None), executor_timeout=60)
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    state = ctrl.execute({
        "refinement_decision": {"refinement_needed": True, "requires_custom_code": True,
                                "targets": [{"type": "custom_code", "description": "fit each band",
                                             "required_outputs": ["Depth_Map"]}]},
        "hspy_data": cube, "original_hspy_data": cube, "energy_axis": E,
        "system_info": {"axis_spec": {"axis_2": {"name": "wavelength", "units": "nm", "start": 400, "end": 900}}},
        "settings": {"output_dir": str(tmp_path)}, "iteration_title": "T", "analysis_images": [],
        "error_dict": None, "max_verification_iterations": 0})
    assert state["dynamic_analysis_records"][0]["task_success"] is True
    return state["custom_analysis_metadata_list"], state["dynamic_analysis_records"]


def test_the_controller_marks_what_no_gate_checked(tmp_path):
    meta, _ = _controller_run(tmp_path, SCRIPT_WITH_SCALARS)
    by_name = {m["name"]: m for m in meta}
    assert set(by_name) == {"Depth_Map", "Band1_Position_nm", "Band1_Depth"}
    assert "gated" not in by_name["Depth_Map"]                     # the map passed the review
    assert by_name["Band1_Position_nm"]["gated"] is False and by_name["Band1_Depth"]["gated"] is False


def _hs_agent(out_dir, meta, records):
    """The real hyperspectral agent, its single-cube pipeline returning what
    the controller committed (the seam tests/test_hs_series.py stubs)."""
    agent = hsa.HyperspectralAnalysisAgent(api_key="sk-dummy", output_dir=str(out_dir),
                                           enable_human_feedback=False, executor_timeout=60)

    def pipeline(self, data_path, system_info, instruction_prompt, reuse_records=None, **kw):
        (self.output_dir / "dynamic_analysis_records.json").write_text(json.dumps(records, default=str))
        return {"detailed_analysis": "bands fitted", "extracted_features": meta,
                "dynamic_analysis_records": records,
                "scientific_claims": [{"claim": "No change in band positions, depths or widths across the series.",
                                       "spectroscopic_evidence": "Band1_Position_nm 435.0003",
                                       "scientific_impact": "x", "has_anyone_question": "?",
                                       "keywords": ["UV-vis"]}]}, None
    agent._run_analysis_pipeline = pipeline.__get__(agent)
    agent._maybe_bank_scripts = lambda *a, **k: []
    agent._maybe_stage_t2_solutions = lambda *a, **k: []
    agent._auto_select_skills = lambda *a, **k: []
    return agent


def _run_task_through_the_board(tmp_path, script, monkeypatch):
    """controller → agent (stamp) → orchestrator.run_task (analyses row) →
    board.records_for: the path a swarm item's claim takes."""
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent, AnalysisMode
    meta, records = _controller_run(tmp_path / "ctrl", script)
    cube, _ = _cube(1)
    data = tmp_path / "film.npy"
    np.save(data, cube)
    (tmp_path / "film.json").write_text(json.dumps(AXIS))
    with contextlib.redirect_stdout(io.StringIO()):
        orch = AnalysisOrchestratorAgent(base_dir=str(tmp_path / "s"), api_key="sk-dummy",
                                         model_name="claude-opus-4-6", analysis_mode=AnalysisMode.AUTONOMOUS)
        orch.create_agent_for_analysis = lambda agent_id, out_dir, **kw: _hs_agent(out_dir, meta, records)

        def chat(prompt):
            out = json.loads(orch.tools.execute_tool("run_analysis", data_path=str(data), agent_id=2,
                                                     analysis_goal="fit each absorption band"))
            assert out.get("status") == "success", out
            return "analysed"
        orch.chat = chat
        result = orch.run_task("analyse the film")
    entry = {"index": 1, "label": "film", "mode": "analysis", "status": "success"}
    return result, board_mod.records_for(entry, result)


def test_a_claim_resting_on_ungated_scalars_is_provisional_and_its_recipe_verified(tmp_path, monkeypatch):
    result, recs = _run_task_through_the_board(tmp_path, SCRIPT_WITH_SCALARS, monkeypatch)
    row = result["analyses"][0]
    assert row["verified"] is True                                 # the gate passed what it checks
    assert row["ungated_outputs"] == ["Band1_Depth", "Band1_Position_nm"]
    claims = [r for r in recs if r["kind"] == "claim"]
    recipes = [r for r in recs if r["kind"] == "recipe"]
    assert claims and all(c["status"] == "provisional" for c in claims)
    assert "no gate checked (Band1_Depth, Band1_Position_nm)" in claims[0]["evidence"]["gate"]
    assert recipes and all(r["status"] == "verified" for r in recipes)
    # a subscription on claims (status defaults to verified) does not fire
    assert not any(reactions.matches(ON_A_CLAIM, c) for c in claims)


def test_a_run_with_nothing_ungated_posts_as_before(tmp_path, monkeypatch):
    result, recs = _run_task_through_the_board(tmp_path, SCRIPT_MAPS_ONLY, monkeypatch)
    assert result["analyses"][0]["ungated_outputs"] == []
    claims = [r for r in recs if r["kind"] == "claim"]
    assert [r["status"] for r in claims] == ["verified"]
    assert all(reactions.matches(ON_A_CLAIM, c) for c in claims)


def test_every_reason_a_claim_is_held_is_given():
    """A replay whose interpretation is not certified AND that reported
    ungated outputs: the claim's reason names both."""
    row = {"analysis_id": "r1", "status": "success", "verified": True, "reason": "replay gate passed",
           "decided_by": "replay_gate", "interpretation_checked": False, "ungated_outputs": ["n_fitted_pixels"]}
    verified, why = board_mod._claim_verified(row)
    assert not verified
    assert "interpretation is not certified" in why and "no gate checked (n_fitted_pixels)" in why
    verified, why = board_mod._claim_verified({**row, "interpretation_checked": True})
    assert not verified and "certified" not in why and "n_fitted_pixels" in why


def test_a_series_carries_its_units_ungated_outputs(tmp_path, monkeypatch):
    """A hyperspectral series: each unit's stamp names its ungated outputs and
    the series' stamp their union — the series synthesis reads every unit's
    scalars, so its claims rest on them."""
    from scilink.agents.exp_agents._verification_record import ungated_outputs_of
    meta, records = _controller_run(tmp_path / "ctrl", SCRIPT_WITH_SCALARS)
    paths = []
    for i in range(3):
        cube, _ = _cube(i)
        p = tmp_path / f"film_{i}.npy"
        np.save(p, cube)
        paths.append(str(p))

    def pipeline(self, data_path, system_info, instruction_prompt, reuse_records=None, **kw):
        recs = [{**records[0], "locked_replay": bool(reuse_records), "replay_verbatim": True}]
        (self.output_dir / "dynamic_analysis_records.json").write_text(json.dumps(recs, default=str))
        return {"detailed_analysis": "bands", "extracted_features": meta,
                "dynamic_analysis_records": recs, "scientific_claims": []}, None
    monkeypatch.setenv("SCILINK_HS_SERIES_POOL", "thread")
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_run_analysis_pipeline", pipeline)
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_maybe_bank_scripts", lambda *a, **k: [])
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_maybe_stage_t2_solutions", lambda *a, **k: [])
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_auto_select_skills", lambda *a, **k: [])
    from test_hs_series import _Calls, _FakeModel
    agent = hsa.HyperspectralAnalysisAgent(api_key="sk-dummy", output_dir=str(tmp_path / "series"),
                                           enable_human_feedback=False, executor_timeout=120)
    agent.model = _FakeModel(_Calls())
    res = agent.analyze(paths, system_info=dict(AXIS),
                        series_metadata={"variable": "dose", "values": [1, 2, 3], "unit": "mC"})
    units = [r for r in res["individual_results"] if r.get("success")]
    assert units and all(r["unit_verdict"]["ungated"] == ["Band1_Depth", "Band1_Position_nm"] for r in units)
    assert res["verdict"]["verified"] is True
    assert res["verdict"]["ungated"] == ungated_outputs_of(res) == ["Band1_Depth", "Band1_Position_nm"]


def test_a_failed_derivation_is_not_a_result(tmp_path, monkeypatch):
    """A derivation whose products fail its checks leaves them on disk: the
    tool result lists them as unverified, and run_task leaves the failed run's
    files out of files_produced and says where they are."""
    from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent, AnalysisMode
    src = tmp_path / "run"
    src.mkdir()
    np.save(src / "depth.npy", np.ones((4, 4)))
    # writes a product, then reports a product that does not exist: the
    # derivation's own product check fails on every attempt
    code = ("import json, numpy as np, os\n"
            "out = _DERIVE['out_dir']\n"
            "np.savetxt(os.path.join(out, 'depth_table.csv'), np.ones((4, 4)), delimiter=',')\n"
            "print('DERIVE_RESULT_JSON ' + json.dumps({'products': [{'path': os.path.join(out, 'missing.csv'),"
            " 'description': 'table'}], 'summary': 'depth table'}))\n")
    with contextlib.redirect_stdout(io.StringIO()):
        orch = AnalysisOrchestratorAgent(base_dir=str(tmp_path / "s"), api_key="sk-dummy",
                                         model_name="claude-opus-4-6", analysis_mode=AnalysisMode.AUTONOMOUS)
        seen = {}

        def chat(prompt):
            seen["tool"] = json.loads(orch.tools.execute_tool(
                "derive_from_outputs", task="a table of the depth map", source_paths=[str(src)],
                code=code, max_attempts=1, llm_verify=False))
            return "derived"
        orch.chat = chat
        result = orch.run_task("tabulate the depth map")
    tool = seen["tool"]
    assert tool["status"] == "error"
    assert any(f.endswith("depth_table.csv") for f in tool["unverified_files"])
    assert "not results" in tool["note"]
    assert not any("derive_" in f for f in result["files_produced"])
    assert any("are not results and are not in files_produced" in w for w in result["warnings"])
