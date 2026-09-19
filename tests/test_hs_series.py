"""Offline tests: hyperspectral SERIES analysis (anchor + locked replay).

The hyperspectral agent used to refuse a list of datacubes ("Batch processing
not supported") while the image and curve agents run a locked-recipe series.
Now ``analyze([cube_0, cube_1, ...], series_metadata=...)`` analyses the first
dataset in full, locks its approved dynamic-analysis scripts, replays them
verbatim on every later dataset (the #172 locked-replay path), flags failed
datasets and statistical outliers, re-analyses failures within
``max_series_refits``, runs the trend codegen over the per-dataset feature
table and synthesizes the series.

The single-cube pipeline is stubbed at ``_run_analysis_pipeline`` (the seam
below the series driver); the trend script really executes in the sandbox
so the ``series_analysis_results.json`` contract is exercised end to end.

  conda run -n scilink python -m pytest tests/test_hs_series.py -q
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents import hyperspectral_analysis_agent as hsa
from scilink.agents.exp_agents.controllers import hyperspectral_series as hs
from scilink.agents.exp_agents.feature_table import write_feature_table

HyperspectralAnalysisAgent = hsa.HyperspectralAnalysisAgent

SCRIPT = "def analyze_feature(data, axis):\n    return {'maps': {'Mean_Map': data.mean(2)}}\n"
AXIS = {"technique": "EELS", "sample": "TiOx",
        "energy_range": {"start": 450.0, "end": 550.0, "units": "eV"}}
MEANS = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]      # index 5 is a 2-sigma outlier


# ---------------------------------------------------------------------------
# Fixtures: cubes on disk, a stubbed single-cube pipeline, a fake LLM
# ---------------------------------------------------------------------------

def _cubes(tmp_path):
    d = tmp_path / "cubes"
    d.mkdir()
    paths = []
    for i, m in enumerate(MEANS):
        p = d / f"cube_T{300 + 50 * i}.npy"
        np.save(p, np.full((4, 4, 8), m, dtype=np.float32))
        paths.append(str(p))
    return paths


class _Calls:
    def __init__(self):
        self.pipeline = []      # one entry per single-cube pipeline run
        self.llm = []           # one entry per parent-level LLM prompt


def _install_fake_pipeline(monkeypatch, calls, fail_on_replay=(), salvage_on_anchor=()):
    """Stub the single-cube pipeline: features = the cube's mean; writes the
    dynamic_analysis_records.json the locked-replay loader reads; fails the
    named datasets when they are REPLAYED (so the refit stage has work)."""

    def fake_pipeline(self, data_path, system_info, instruction_prompt,
                      reuse_records=None, **kw):
        name = Path(data_path).stem
        locked = kw.get("locked_targets")
        calls.pipeline.append({
            "role": self._series_role, "name": name,
            "reuse": bool(reuse_records), "locked": bool(locked), "hints": kw.get("hints"),
            "extra_outputs": (locked[0].get("extra_outputs") if locked else None),
            "skills": [s.get("name") for s in (kw.get("skill_state") or {}).get("skills_loaded", [])],
            "series_context": (system_info or {}).get("series_context")
            if isinstance(system_info, dict) else None,
        })
        if name in fail_on_replay and reuse_records:
            return None, {"error": "replay failed", "details": f"boom on {name}"}
        value = float(np.load(data_path).mean())
        if name in salvage_on_anchor and self._series_role == "anchor":
            # salvaged attempt: features committed, but no required output approved
            rec = {"target": "mean map", "task_success": False, "required_outputs": ["Mean_Map"],
                   "script": SCRIPT, "quality_history": {"approved": False}}
            (self.output_dir / "dynamic_analysis_records.json").write_text(json.dumps([rec]))
            return {"detailed_analysis": "salvaged", "scientific_claims": [],
                    "extracted_features": [{"name": "Field_Mean", "units": "", "scalar": value}],
                    "dynamic_analysis_records": [rec],
                    "degradation_notes": [{"confidence": "low", "caveat": "salvaged"}]}, None
        rec = {"target": (locked[0]["target"] if locked else "mean map"), "task_success": True,
               "required_outputs": ["Mean_Map"], "script": SCRIPT,
               "quality_history": {"approved": True},
               "locked_replay": bool(reuse_records), "replay_verbatim": True}
        (self.output_dir / "dynamic_analysis_records.json").write_text(json.dumps([rec]))
        # Fresh code under locked targets keeps the required output NAME but,
        # like a real regenerated script, drifts the units and adds its own
        # diagnostic map under a different prefix.
        feats = ([{"name": "Mean_Map", "units": "counts",
                   "stats": {"min": value - 1, "max": value + 1, "mean": value}},
                  {"name": "Fit_R2", "units": "", "stats": {"mean": 0.99}}]
                 if locked else
                 [{"name": "Mean_Map", "units": "a.u.",
                   "stats": {"min": value - 1, "max": value + 1, "mean": value}},
                  {"name": "R2", "units": "", "stats": {"mean": 0.98}}])
        return {
            "detailed_analysis": f"analysis of {name}",
            "scientific_claims": [],
            "extracted_features": feats,
            "dynamic_analysis_records": [rec],
        }, None

    monkeypatch.setattr(HyperspectralAnalysisAgent, "_run_analysis_pipeline", fake_pipeline)
    monkeypatch.setattr(HyperspectralAnalysisAgent, "_maybe_bank_scripts",
                        lambda self, *a, **k: [])
    monkeypatch.setattr(HyperspectralAnalysisAgent, "_maybe_stage_t2_solutions",
                        lambda self, *a, **k: [])


TREND_SCRIPT = """
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d = json.load(open('series_analysis_results.json'))
rows = [r for r in d['results'] if r['success']]
vals = d['series_metadata']['values']
assert isinstance(vals, list), 'values must be a list aligned by index'
x = [vals[r['index']] for r in rows]
y = [r['extracted_features']['Mean_Map_mean'] for r in rows]
plt.plot(x, y, 'o-'); plt.savefig('feature_trends.png'); plt.close('all')
print('TREND_OK', len(rows))
"""


class _FakeModel:
    """Answers the trend-codegen prompt with a real script and the synthesis
    prompt with a series interpretation; records what it was asked."""

    def __init__(self, calls, plan=None):
        self.calls = calls
        self.plan = plan            # regime plan to return, None = one regime

    def generate_content(self, contents, **kw):
        text = "\n".join(c for c in contents if isinstance(c, str))
        self.calls.llm.append(text)
        if "Series Regime Planning" in text:
            return json.dumps({"observations": "scouted",
                               "series_analysis_plan": self.plan} if self.plan
                              else {"observations": "uniform series"})
        if "Return JSON with" in text and '"script"' in text:
            return json.dumps({"analysis_approach": "mean vs T", "key_metrics": ["Mean_Map_mean"],
                               "flagged_handling": "red x", "expected_outputs": ["feature_trends.png"],
                               "script": TREND_SCRIPT})
        return json.dumps({
            "detailed_analysis": "Mean_Map_mean rises linearly with temperature until a jump at 550 K.",
            "scientific_claims": [{"claim": "Mean intensity increases with temperature",
                                   "spectroscopic_evidence": "Mean_Map_mean 1→5 then 100",
                                   "scientific_impact": "x", "has_anyone_question": "Has anyone seen this?",
                                   "keywords": ["EELS", "temperature", "series", "TiOx"]}],
            "feature_trends": {"Mean_Map_mean": {"trend": "increasing", "interpretation": "thermal"}},
            "flagged_analysis": {"summary": "one outlier", "possible_causes": ["transition"],
                                 "scientific_significance": "maybe"},
            "caveats": ["synthetic"],
        })


def _agent(tmp_path, calls, monkeypatch, plan=None):
    out = tmp_path / "series_out"
    agent = HyperspectralAnalysisAgent(api_key="sk-dummy", output_dir=str(out),
                                       enable_human_feedback=False, executor_timeout=120)
    agent.model = _FakeModel(calls, plan=plan)
    # Children must not re-select skills; the parent does it once.
    autoselect_calls = []
    monkeypatch.setattr(HyperspectralAnalysisAgent, "_auto_select_skills",
                        lambda self, *a, **k: autoselect_calls.append(self._series_role) or [])
    agent._autoselect_calls = autoselect_calls
    return agent, out


# ---------------------------------------------------------------------------
# End to end (stubbed pipeline, real series driver, real trend execution)
# ---------------------------------------------------------------------------

def test_series_anchor_replay_refit_flag_trend_synthesis(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls, fail_on_replay={"cube_T400"})
    agent, out = _agent(tmp_path, calls, monkeypatch)
    paths = _cubes(tmp_path)
    meta = {"variable": "temperature", "values": [300, 350, 400, 450, 500, 550], "unit": "K"}

    res = agent.analyze(paths, system_info=dict(AXIS), series_metadata=meta,
                        objective="How does the signal evolve with temperature?")

    # --- roles: one anchor (full), five verbatim replays, one refit ----------
    roles = [(c["role"], c["reuse"]) for c in calls.pipeline]
    assert roles[0] == ("anchor", False)
    assert roles[1:6] == [("replay", True)] * 5
    assert roles[6] == ("refit", False)
    refit = calls.pipeline[6]
    assert refit["locked"] is True                      # locked targets, fresh code
    assert not any(c["locked"] for c in calls.pipeline[:6])
    assert refit["name"] == "cube_T400" and "Series context" in refit["hints"]
    # the scout + planning call happened once, before any dataset ran
    assert "Series Regime Planning" in calls.llm[0] and "Change point" in calls.llm[0]
    assert refit["series_context"]["value"] == 400
    assert calls.pipeline[3]["series_context"] == {"index": 3, "n_datasets": 6, "variable": "temperature",
                                                   "unit": "K", "value": 450}
    # skills resolved once by the parent (role None), never by a child
    assert agent._autoselect_calls == [None]

    # --- result shape mirrors the image/curve series result ------------------
    assert res["status"] == "success", res.get("warnings")
    s = res["summary"]
    assert (s["total_datasets"], s["successful_analyses"], s["anchor_index"]) == (6, 6, 0)
    assert s["locked_targets"] == ["mean map"] and s["refitted_count"] == 1
    assert len(res["individual_results"]) == 6
    r2 = res["individual_results"][2]
    assert r2["adaptively_refitted"] is True and r2["role"] == "refit" and r2["success"]
    # schema completion: the refit's drifted names were aliased onto the locked columns
    assert r2["locked_schema_gap"] == []
    assert r2["schema_aliases"] == {"Mean_Map_min": "Mean_Map_min_counts", "Mean_Map_max": "Mean_Map_max_counts",
                                    "Mean_Map_mean": "Mean_Map_mean_counts", "R2_mean": "Fit_R2_mean"}
    assert res["summary"]["regimes"] == 1 and res["summary"]["regime_anchors"] == {"series": 0}
    assert res["series_analysis_plan"] is None
    assert res["individual_results"][1]["reuse_validity"]["verdict"] == "good"
    assert res["refit_summary"] == [{"index": 2, "name": "cube_T400", "regime": "series",
                                     "original_error": "boom on cube_T400", "new_status": "success",
                                     "n_features": 4, "improved": True, "locked_schema_gap": []}]
    assert res["refit_skipped_by_budget"] == []

    # --- flagging: the 100-mean dataset is a statistical outlier, not refit --
    flagged = {f["index"]: f["reason"] for f in res["flagged_datasets"]}
    assert flagged == {5: "statistical_outlier"}
    assert res["individual_results"][5]["flagged"] is True
    assert res["series_features"]["Mean_Map_mean"] == MEANS     # aligned across the refit

    # --- trend codegen ran for real over the series JSON ----------------------
    assert res["trend_analysis"]["success"] is True
    assert "TREND_OK 6" in res["trend_analysis"]["stdout"]
    assert (out / "feature_trends.png").is_file()
    assert (out / "trend_analysis.py").is_file()

    # --- synthesis + report ---------------------------------------------------
    assert res["scientific_claims"][0]["claim"].startswith("Mean intensity")
    assert res["feature_trends"]["Mean_Map_mean"]["trend"] == "increasing"
    assert "550 K" in res["detailed_analysis"]
    assert Path(res["report_path"]).is_file()
    synth_prompt = calls.llm[-1]
    assert "control_value" in synth_prompt and "temperature=450 K" in synth_prompt
    assert "How does the signal evolve" in synth_prompt

    # --- on-disk layout: per-dataset dirs, series JSON, feature table --------
    for i in range(6):
        assert (out / f"dataset_{i:04d}").is_dir()
    assert (out / "dataset_0002_refit").is_dir()
    sar = json.loads((out / "series_analysis_results.json").read_text())
    assert sar["locked_config"]["anchor_index"] == 0 and sar["successful"] == 6
    assert sar["results"][2]["adaptively_refitted"] is True     # re-written after refit
    assert sar["series_metadata"]["values"] == meta["values"]
    assert json.loads((out / "flagged_datasets.json").read_text())["flagged_count"] == 1
    assert (out / "analysis_results.json").is_file()

    csv_path = write_feature_table(out)                # the orchestrator's adapter, unchanged
    lines = Path(csv_path).read_text().splitlines()
    header = lines[0].split(",")
    assert "temperature" in header and "Mean_Map_mean" in header
    assert len(lines) == 7                              # header + one row per dataset


def test_refit_budget_zero_leaves_failure_and_reports_partial(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls, fail_on_replay={"cube_T400"})
    agent, out = _agent(tmp_path, calls, monkeypatch)
    paths = _cubes(tmp_path)
    res = agent.analyze(paths, system_info={**AXIS, "series": {
        "variable": "temperature", "unit": "K",
        # filename-keyed values are aligned to the list order
        "values": {Path(p).name: 300 + 50 * i for i, p in enumerate(paths)}}},
        max_series_refits=0)
    assert [c["role"] for c in calls.pipeline] == ["anchor"] + ["replay"] * 5
    assert res["status"] == "partial"
    assert res["summary"]["successful_analyses"] == 5
    assert res["individual_results"][2]["success"] is False
    assert res["individual_results"][2]["flag_reason"] == "analysis_failed"
    assert [x["index"] for x in res["refit_skipped_by_budget"]] == [2]
    assert res["series_metadata"]["values"] == [300, 350, 400, 450, 500, 550]
    assert "series" not in calls.pipeline[0]["series_context"]      # popped from system_info
    sar = json.loads((out / "series_analysis_results.json").read_text())
    assert sar["successful"] == 5


def test_anchor_failure_promotes_next_dataset(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    agent, out = _agent(tmp_path, calls, monkeypatch)
    paths = _cubes(tmp_path)[:4]
    orig = HyperspectralAnalysisAgent._run_analysis_pipeline

    def flaky(self, data_path, *a, **k):
        # the full-analysis anchor attempt on cube 0 explodes; its later
        # locked-targets refit (a different attempt) goes through
        if Path(data_path).stem == "cube_T300" and self._series_role == "anchor":
            return None, {"error": "anchor exploded", "details": "no axis"}
        return orig(self, data_path, *a, **k)
    monkeypatch.setattr(HyperspectralAnalysisAgent, "_run_analysis_pipeline", flaky)

    res = agent.analyze(paths, series_metadata={"variable": "t", "values": [0, 1, 2, 3], "unit": "s"},
                        max_series_refits=None)
    roles = [(c["role"], c["reuse"], c["locked"]) for c in calls.pipeline]
    # 0 fails as anchor candidate (before the recorder), 1 becomes the anchor
    # and schema source, 2-3 replay; 0 then gets a refit in locked-targets
    # mode (a different attempt from its failed independent run).
    assert roles == [("anchor", False, False), ("replay", True, False), ("replay", True, False),
                     ("refit", False, True)]
    assert res["summary"]["anchor_index"] == 1
    r0 = res["individual_results"][0]
    assert r0["role"] == "refit" and r0["success"] and r0["adaptively_refitted"]
    assert res["refit_summary"][0]["index"] == 0
    assert res["status"] == "success"


def test_salvaged_anchor_is_unverified_and_refit_with_locked_targets(tmp_path, monkeypatch):
    """Dataset 0 commits features from a salvaged attempt (no approved
    required output): it must not lock, dataset 1 becomes the schema source,
    dataset 0 is flagged 'unverified' and re-analysed in locked-targets mode."""
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls, salvage_on_anchor={"cube_T300"})
    agent, out = _agent(tmp_path, calls, monkeypatch)
    paths = _cubes(tmp_path)[:4]
    res = agent.analyze(paths, series_metadata={"variable": "t", "values": [0, 1, 2, 3], "unit": "s"})
    roles = [(c["role"], c["reuse"], c["locked"]) for c in calls.pipeline]
    assert roles == [("anchor", False, False), ("anchor", False, False), ("replay", True, False),
                     ("replay", True, False), ("refit", False, True)]
    assert "was not verified" in calls.pipeline[4]["hints"]
    assert res["summary"]["anchor_index"] == 1 and res["summary"]["unverified_count"] == 0
    r0 = res["individual_results"][0]
    assert r0["adaptively_refitted"] and r0["verified"] and r0["locked_schema_gap"] == []
    assert res["refit_summary"][0]["original_error"] == "unverified (salvaged attempt)"
    assert res["series_features"]["Mean_Map_mean"] == MEANS[:4]         # aligned after the refit
    assert res["status"] == "success"


def test_unverified_row_stays_flagged_and_schema_completed_when_budget_is_zero(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls, salvage_on_anchor={"cube_T300"})
    agent, out = _agent(tmp_path, calls, monkeypatch)
    res = agent.analyze(_cubes(tmp_path)[:4], max_series_refits=0,
                        series_metadata={"variable": "t", "values": [0, 1, 2, 3], "unit": "s"})
    r0 = res["individual_results"][0]
    assert r0["role"] == "independent" and r0["success"] and r0["verified"] is False
    assert r0["flag_reason"] == "unverified"
    assert "locked_schema_gap" in r0 and "Mean_Map_mean" in r0["locked_schema_gap"]   # completed, gap reported
    assert [x["index"] for x in res["refit_skipped_by_budget"]] == [0]
    assert res["summary"]["unverified_count"] == 1 and res["status"] == "partial"
    # unverified rows are excluded from the outlier statistics
    assert not [f for f in res["flagged_datasets"] if f["reason"] == "statistical_outlier"]


def test_failed_anchor_candidate_not_refit_without_schema(tmp_path, monkeypatch):
    """No dataset ever locked a recipe: a failed anchor candidate already had
    the only kind of run available, so it is not run again."""
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    agent, out = _agent(tmp_path, calls, monkeypatch)
    monkeypatch.setattr(HyperspectralAnalysisAgent, "_run_analysis_pipeline",
                        lambda self, *a, **k: (None, {"error": "x", "details": "always"}))
    res = agent.analyze(_cubes(tmp_path)[:3], series_metadata={"variable": "t", "values": [0, 1, 2], "unit": "s"})
    assert res["status"] == "error" and res["summary"]["successful_analyses"] == 0
    assert res["refit_summary"] == [] and not (out / "dataset_0000_refit").exists()


def test_single_element_list_is_a_single_analysis(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    agent, out = _agent(tmp_path, calls, monkeypatch)
    res = agent.analyze(_cubes(tmp_path)[:1], system_info=dict(AXIS))
    assert res["status"] == "success" and "summary" not in res
    assert calls.pipeline[0]["role"] is None and len(calls.pipeline) == 1
    assert not (out / "dataset_0000").exists()


def test_4d_stack_is_a_series(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    agent, out = _agent(tmp_path, calls, monkeypatch)
    stack = np.stack([np.full((3, 3, 6), m, np.float32) for m in (1, 2, 3)])
    res = agent.analyze(stack, system_info=dict(AXIS))
    assert sorted(p.name for p in (out / "series_input").iterdir()) == [
        "cube_0000.npy", "cube_0001.npy", "cube_0002.npy"]
    assert res["summary"]["total_datasets"] == 3
    assert res["series_metadata"] == {"variable": "index", "values": [0, 1, 2], "unit": ""}
    # a bare 3D array is still refused (unchanged contract)
    r3 = agent.analyze(stack[0], system_info=dict(AXIS))
    assert r3["status"] == "error" and "Direct array" in r3["error"]["error"]


def test_two_regimes_lock_one_recipe_each_with_shared_schema(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    plan = {"rationale": "edge shifts", "regimes": [
        {"name": "low_T", "dataset_indices": [0, 1, 2], "description": "L3 near 458 eV"},
        {"name": "high_T", "dataset_indices": [3, 4, 5], "description": "L3 shifted +3 eV"}],
        "transition_points": [{"between_indices": [2, 3], "description": "jump"}]}
    agent, out = _agent(tmp_path, calls, monkeypatch, plan=plan)
    paths = _cubes(tmp_path)
    res = agent.analyze(paths, system_info=dict(AXIS),
                        series_metadata={"variable": "temperature", "values": [300, 350, 400, 450, 500, 550],
                                         "unit": "K"})
    roles = [(c["role"], c["reuse"], c["locked"]) for c in calls.pipeline]
    assert roles == [("anchor", False, False), ("replay", True, False), ("replay", True, False),
                     ("regime_anchor", False, True), ("replay", True, False), ("replay", True, False)]
    assert "Series regime 'high_T'" in calls.pipeline[3]["hints"]
    assert res["status"] == "success"
    assert res["summary"]["regimes"] == 2
    assert res["summary"]["regime_anchors"] == {"low_T": 0, "high_T": 3}
    assert [r["regime"] for r in res["individual_results"]] == ["low_T"] * 3 + ["high_T"] * 3
    # the second regime's anchor was asked for the schema source's extra maps by name
    assert calls.pipeline[3]["extra_outputs"] == ["R2"]
    # the second regime's anchor reports the schema source's column names
    r3 = res["individual_results"][3]
    assert r3["role"] == "regime_anchor" and r3["locked_schema_gap"] == []
    assert "Mean_Map_mean_counts" in r3["schema_aliases"].values()
    assert res["series_features"]["Mean_Map_mean"] == MEANS
    assert res["locked_config"]["regimes"]["high_T"]["anchor_index"] == 3
    sar = json.loads((out / "series_analysis_results.json").read_text())
    assert [r["name"] for r in sar["series_analysis_plan"]["regimes"]] == ["low_T", "high_T"]
    # replays of the second regime pointed at ITS anchor, not the first
    assert res["individual_results"][4]["reuse_validity"]["verdict"] == "good"
    # the synthesis prompt carried the regimes
    assert "high_T" in calls.llm[-1]
    assert "Series regimes" in Path(res["report_path"]).read_text()


def test_regime_plan_with_missing_indices_is_repaired(tmp_path, monkeypatch):
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    plan = {"regimes": [{"name": "A", "dataset_indices": [0, 1]},
                        {"name": "B", "dataset_indices": [4, 5, 99]},
                        {"name": "empty", "dataset_indices": []}]}
    agent, out = _agent(tmp_path, calls, monkeypatch, plan=plan)
    res = agent.analyze(_cubes(tmp_path), system_info=dict(AXIS),
                        series_metadata={"variable": "t", "values": [0, 1, 2, 3, 4, 5], "unit": "s"})
    regs = {r["name"]: r["dataset_indices"] for r in res["series_analysis_plan"]["regimes"]}
    assert set(regs) == {"A", "B"}                       # empty regime dropped, 99 clamped
    assert sorted(regs["A"] + regs["B"]) == [0, 1, 2, 3, 4, 5]   # 2 and 3 assigned to a neighbour
    assert res["summary"]["successful_analyses"] == 6


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_flatten_feature_records_matches_single_cube_columns():
    recs = [
        {"name": "Peak Pos", "units": "eV", "stats": {"min": 1, "max": 2, "mean": 1.5, "bad": "x"}},
        {"name": "Ratio", "units": "a.u.", "scalar": 0.4},
        {"not_measurable": {"feature": "Plasmon shift"}},
        "junk", {"name": "no stats"},
    ]
    assert hs.flatten_feature_records(recs) == {
        "Peak_Pos_min_eV": 1.0, "Peak_Pos_max_eV": 2.0, "Peak_Pos_mean_eV": 1.5,
        "Ratio": 0.4, "Plasmon_shift_not_measurable": 1}


def _row(i, success=True, **feats):
    return {"index": i, "name": f"u{i}", "success": success,
            "extracted_features": feats, "error": None if success else "err"}


def test_detect_outliers_flags_failures_and_sigma_outliers():
    # With n near-identical values a single outlier tops out at sqrt(n-1)
    # sigma (population std), so nine rows are needed to clear 2 sigma.
    rows = [_row(0, a=1.0, b_not_measurable=1), _row(1, a=1.1), _row(2, a=0.9),
            _row(3, a=1.0), _row(4, a=20.0), _row(5, success=False),
            _row(6, a=1.05), _row(7, a=0.95), _row(8, a=1.0)]
    flagged = {f["index"]: f["reason"] for f in hs.detect_outliers(rows, 2.0)}
    assert flagged == {5: "analysis_failed", 4: "statistical_outlier"}
    # fewer than three successes: only failures are flagged
    assert [f["reason"] for f in hs.detect_outliers(rows[:2] + [rows[5]], 2.0)] == ["analysis_failed"]


def test_detect_outliers_leave_one_out_flags_small_series():
    """Five datasets, a monotonic trend and one planted jump: a population
    z-score caps at sqrt(4) = 2 sigma so it could never flag; the leave-one-out
    score does (observed live: an unflagged +2.5 eV white-line jump)."""
    rows = [_row(i, pos=458.0 + 0.4 * i) for i in range(4)] + [_row(4, pos=462.1)]
    flagged = {f["index"]: f for f in hs.detect_outliers(rows, 2.0)}
    assert list(flagged) == [4]
    assert "trend" in flagged[4]["details"]
    # the trend's own endpoints are not outliers (linear trend, no noise)
    rows = [_row(i, pos=458.0 + 0.4 * i) for i in range(5)]
    assert hs.detect_outliers(rows, 2.0) == []
    # ... nor with the control variable supplied and a mild curvature
    rows = [_row(i, pos=458.0 + 0.4 * i + 0.02 * i * i) for i in range(6)]
    assert hs.detect_outliers(rows, 2.0, control_values=[300, 350, 400, 450, 500, 550]) == []
    # exactly three successes: population score, as in the image agent
    rows = [_row(0, a=1.0), _row(1, a=1.0), _row(2, a=5.0)]
    fl = hs.detect_outliers(rows, 1.2)
    assert fl and "series mean" in fl[0]["details"]


def test_detect_outliers_primary_prefixes_ignore_diagnostics():
    """With locked required outputs, only their columns are scored: a refit's
    different R² is not an outlier, the primary jump is."""
    rows = [_row(i, Pos_mean=458.0 + 0.4 * i, Pos_min=457.0 + 0.4 * i, R2_mean=0.98) for i in range(6)]
    rows[2]["extracted_features"]["R2_mean"] = 0.995          # refit with a different method
    rows[5]["extracted_features"]["Pos_mean"] = 463.0         # the physics
    rows[5]["extracted_features"]["Pos_min"] = 462.0
    flagged = hs.detect_outliers(rows, 2.0, feature_prefixes=["Pos"])
    assert [f["index"] for f in flagged] == [5]
    assert "R2" not in flagged[0]["details"]
    # a noisy extreme-value column alone does not flag: min/max are not scored
    rows2 = [_row(i, Pos_mean=458.0 + 0.4 * i, Pos_min=457.0 + 0.4 * i) for i in range(6)]
    rows2[3]["extracted_features"]["Pos_min"] = 440.0
    assert hs.detect_outliers(rows2, 2.0, feature_prefixes=["Pos"]) == []
    assert "Pos_min" in hs.detect_outliers(rows2, 2.0)[0]["details"]     # unrestricted scan still sees it
    # prefixes that match nothing fall back to every column
    assert {f["index"] for f in hs.detect_outliers(rows, 2.0, feature_prefixes=["Nope"])} >= {5}


def test_detect_outliers_dominant_anomaly_suppresses_noise_columns():
    """One real jump plus near-constant diagnostic columns with noise-level
    scatter: only the dataset carrying the dominant anomaly is flagged."""
    rows = [_row(i, pos=458.0 + 0.4 * i, r2=0.977 + 3e-4 * ((i * 7) % 3), fwhm=3.2 + 1e-3 * (i % 2))
            for i in range(5)]
    rows[4]["extracted_features"]["pos"] = 462.8
    rows[0]["extracted_features"]["r2"] = 0.9783          # ~4-sigma in its own column
    flagged = hs.detect_outliers(rows, 2.0)
    assert [f["index"] for f in flagged] == [4]
    assert flagged[4 - 4]["deviation_sigma"] > 5


def test_select_refit_candidates_budget():
    flagged = [{"index": 1, "reason": "analysis_failed"}, {"index": 2, "reason": "statistical_outlier"},
               {"index": 3, "reason": "analysis_failed"}, {"index": 4, "reason": "analysed_failed_typo"}]
    now, skipped = hs.select_refit_candidates(flagged, None)
    assert [c["index"] for c in now] == [1, 3] and skipped == []
    now, skipped = hs.select_refit_candidates(flagged, 1)
    assert [c["index"] for c in now] == [1] and [c["index"] for c in skipped] == [3]
    assert "max_series_refits=1" in skipped[0]["skip_reason"]
    now, skipped = hs.select_refit_candidates(flagged, 0)
    assert now == [] and [c["index"] for c in skipped] == [1, 3]
    # one dataset carrying two reasons is one candidate, at its highest priority
    both = [{"index": 5, "reason": "unverified"}, {"index": 5, "reason": "analysis_failed"},
            {"index": 2, "reason": "unverified"}]
    now, skipped = hs.select_refit_candidates(both, 1)
    assert [(c["index"], c["reason"]) for c in now] == [(5, "analysis_failed")]
    assert [c["index"] for c in skipped] == [2]


def test_build_series_row_success_semantics():
    ok = hs.build_series_row(0, "/x/c.npy", {
        "status": "partial", "confidence": "low", "warnings": ["approx"],
        "extracted_features": [{"name": "M", "stats": {"mean": 1.0}}],
        "dynamic_analysis_records": [{"task_success": True}, {"task_success": False}],
        "script_reuse": {"verbatim": False, "n_replayed": 2, "scope_degraded": True}}, "replay", "/o")
    assert ok["success"] and ok["quality_metrics"]["approved_fraction"] == 0.5
    assert ok["reuse_validity"] == {"reused": True, "verbatim": False, "n_replayed": 2,
                                    "verdict": "degraded", "scope_degraded": True}
    nothing = hs.build_series_row(1, "/x/d.npy", {"status": "partial", "extracted_features": [],
                                                  "warnings": ["all targets failed"]}, "replay", "/o")
    assert nothing["success"] is False and nothing["error"] == "all targets failed"
    err = hs.build_series_row(2, "/x/e.npy", {"status": "error", "error": {"error": "E", "details": "D"}},
                              "anchor", "/o")
    assert err["error"] == "D"


def test_trend_instructions_keep_placeholders():
    t = hs.HyperspectralSeriesTrendController.TREND_ANALYSIS_INSTRUCTIONS
    for k in ("{objective}", "{series_summary}", "{series_metadata}", "{flagged_info}"):
        assert k in t
    assert "hyperspectral datacube" in t and "FLAGGED DATASETS" in t
    assert "image analysis results" not in t


def test_select_scout_indices_rule():
    assert hs.select_scout_indices(3) == [0, 1, 2]
    assert hs.select_scout_indices(5) == [0, 2, 4]
    assert hs.select_scout_indices(12) == [0, 3, 6, 9, 11]
    assert len(hs.select_scout_indices(40)) == 7
    assert hs.select_scout_indices(9, scout_all=True) == list(range(9))
    assert len(hs.select_scout_indices(40, scout_all=True)) == 7   # over the scout-all cap


def test_scout_series_finds_the_planted_change_point(tmp_path):
    """Nine cubes with a spectral peak that jumps between the 4th and 5th:
    the full-series SVD change detection locates it and the overlay renders."""
    d = tmp_path / "s"; d.mkdir()
    E = np.linspace(450, 550, 40)
    paths = []
    for i in range(9):
        c = 470.0 + (12.0 if i >= 4 else 0.0) + 0.2 * i
        spec = np.exp(-0.5 * ((E - c) / 3.0) ** 2)
        cube = (spec[None, None, :] * np.random.RandomState(i).uniform(0.8, 1.2, (3, 3, 1))).astype(np.float32)
        p = d / f"c{i}.npy"; np.save(p, cube); paths.append(str(p))
    log = __import__("logging").getLogger("t")
    meta = {"variable": "T", "values": [300 + 10 * i for i in range(9)], "unit": "K"}
    scout = hs.scout_series(paths, np.load, lambda n: (E, "energy (eV)", True), meta, log)
    # evenly spaced {0,2,4,6,8} plus the pair bracketing the sharp change (3, 4)
    assert scout["n_loaded"] == 9 and [s["index"] for s in scout["scout_data"]] == [0, 2, 3, 4, 6, 8]
    assert scout["overlay_png"][:4] == b"\x89PNG"
    red = scout["reduction"]
    assert red["status"] == "success" and red["control_variable"]["source"] == "T"
    assert 330 < red["change_point"] < 340 and red["change_sharpness"] > 0.5
    prompt = hs.build_regime_plan_prompt({"num_images": 9, "series_metadata": meta, "system_info": {},
                                          "skills_loaded": []}, scout)
    text = "\n".join(p for p in prompt if isinstance(p, str))
    assert "Change point: T ≈ 335" in text and "Series Regime Planning" in text
    assert sum(1 for p in prompt if isinstance(p, dict)) == 2      # overlay + score curve
    # fewer than four datasets: no reduction, still scouted
    small = hs.scout_series(paths[:3], np.load, lambda n: (E, "eV", True), meta, log)
    assert small["reduction"] is None and len(small["scout_data"]) == 3


def test_extract_series_plan_validation():
    log = __import__("logging").getLogger("t")
    assert hs.extract_series_plan({"foo": 1}, 5, None, log) is None
    assert hs.extract_series_plan({"series_analysis_plan": {"regimes": ["junk"]}}, 5, None, log) is None
    plan = hs.extract_series_plan({"series_analysis_plan": {"regimes": [
        {"name": "a", "dataset_indices": [0, "1", 7]}, {"dataset_indices": [4]}, {"name": "z", "dataset_indices": []}]}},
        5, None, log)
    names = [r["name"] for r in plan["regimes"]]
    assert names == ["a", "regime_2"]
    cover = sorted(i for r in plan["regimes"] for i in r["dataset_indices"])
    assert cover == [0, 1, 2, 3, 4]
    assert plan["regimes"][0]["dataset_indices"][:2] == [0, 1]
    # a dataset claimed twice stays with the first regime
    plan = hs.extract_series_plan({"series_analysis_plan": {"regimes": [
        {"name": "a", "dataset_indices": [0, 1, 2]}, {"name": "b", "dataset_indices": [2, 3]}]}}, 4, None, log)
    assert [r["dataset_indices"] for r in plan["regimes"]] == [[0, 1, 2], [3]]


def test_complete_locked_schema_aliases_and_gaps():
    locked = ["L3_Position_mean_eV", "L3_Position_min_eV", "Ratio_mean", "R2_mean", "Width_mean_eV"]
    row = {"index": 3, "extracted_features": {
        "L3_Position_mean": 462.0, "L3_Position_min": 461.0,        # units suffix dropped
        "Ratio_mean": 1.1,                                            # exact
        "Fit_R2_mean": 0.99,                                          # prefix drift
        "Sigma_mean_eV": 1.0, "Width_mean": 2.0, "Width_mean_eV_smoothed": 3.0,  # ambiguous → gap
        "Extra_mean": 7.0}}
    out = hs.complete_locked_schema(row, locked)
    f = out["extracted_features"]
    assert f["L3_Position_mean_eV"] == 462.0 and "L3_Position_mean" not in f
    assert f["L3_Position_min_eV"] == 461.0 and f["R2_mean"] == 0.99 and "Fit_R2_mean" not in f
    assert f["Ratio_mean"] == 1.1 and f["Extra_mean"] == 7.0
    assert f["Width_mean"] == 2.0 and f["Width_mean_eV_smoothed"] == 3.0     # both kept, unaliased
    assert out["locked_schema_gap"] == ["Width_mean_eV"]
    assert out["schema_aliases"] == {"L3_Position_mean_eV": "L3_Position_mean",
                                     "L3_Position_min_eV": "L3_Position_min", "R2_mean": "Fit_R2_mean"}
    # nothing to do when the row already conforms
    ok = hs.complete_locked_schema({"index": 0, "extracted_features": {"Ratio_mean": 1}}, ["Ratio_mean"])
    assert ok["locked_schema_gap"] == [] and "schema_aliases" not in ok


def test_locked_targets_plan_asks_for_extra_maps(tmp_path):
    """The locked-targets plan fixes the required names and asks for the
    schema source's extra maps by name; no planning LLM call, no script."""
    from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
    import logging

    class _Boom:
        def generate_content(self, *a, **k):
            raise AssertionError("planner must not be called in locked-targets mode")
    ctrl = hc.SelectRefinementTargetController(_Boom(), logging.getLogger("t"), None, None,
                                               parse_fn=lambda r: ({}, None))
    state = ctrl.execute({"locked_targets": [{"target": "L3 position and ratio",
                                              "required_outputs": ["L3_Position", "L2_L3_Ratio"],
                                              "extra_outputs": ["L3_Intensity", "Fit_R2"]}],
                          "skip_decomposition": True})
    dec = state["refinement_decision"]
    assert dec["requires_custom_code"] and len(dec["targets"]) == 1
    tgt = dec["targets"][0]
    assert tgt["required_outputs"] == ["L3_Position", "L2_L3_Ratio"]
    assert "supplied_script" not in tgt
    assert "L3_Intensity, Fit_R2" in tgt["description"] and "SERIES SCHEMA" in tgt["description"]
    # decomposition is bypassed like a replay
    d = hc.DecompositionController(_Boom(), logging.getLogger("t"), None, None,
                                   settings={"output_dir": str(tmp_path)}, preprocessor=None,
                                   parse_fn=lambda r: ({}, None))
    st = d.execute({"locked_targets": [{"target": "x", "required_outputs": ["A"]}],
                    "hspy_data": np.zeros((2, 2, 8))})
    assert st["skip_decomposition"] is True and st["preprocessing_mask"].shape == (2, 2)


def test_build_series_row_verified_flag_and_unverified_outlier_exclusion():
    salvaged = hs.build_series_row(0, "/x/a.npy", {
        "status": "partial", "extracted_features": [{"name": "M", "stats": {"mean": 1.0}}],
        "dynamic_analysis_records": [{"task_success": False}],
        "script_reuse": {"verbatim": True, "n_replayed": 1}}, "replay", "/o")
    assert salvaged["success"] and salvaged["verified"] is False
    assert salvaged["reuse_validity"]["verdict"] == "degraded"      # verbatim but unverified
    clean = hs.build_series_row(1, "/x/b.npy", {
        "status": "success", "extracted_features": [{"name": "M", "stats": {"mean": 1.0}}],
        "dynamic_analysis_records": [{"task_success": True}]}, "replay", "/o")
    assert clean["verified"] is True
    rows = [dict(_row(i, a=1.0 + 0.1 * i), verified=True) for i in range(4)] + [dict(_row(4, a=50.0), verified=False)]
    flagged = hs.detect_outliers(rows, 2.0)
    assert [(f["index"], f["reason"]) for f in flagged] == [(4, "unverified")]
    now, skipped = hs.select_refit_candidates(flagged, None)
    assert [c["index"] for c in now] == [4]


def test_detect_outliers_groups_score_each_regime_separately():
    """Interleaved regimes: scored as one series, every dataset of the minority
    state looks anomalous; scored per regime, only a true within-regime
    outlier is flagged."""
    rows = []
    for i in range(8):
        pos = 458.0 + (3.0 if i % 2 else 0.0) + 0.05 * i
        rows.append(_row(i, pos=pos))
    groups = {"A": [0, 2, 4, 6], "B": [1, 3, 5, 7]}
    # pure alternation: per-regime scoring flags nothing
    assert hs.detect_outliers(rows, 2.0, groups=groups) == []
    rows[6]["extracted_features"]["pos"] = 470.0          # real outlier inside regime A
    per = hs.detect_outliers(rows, 2.0, groups=groups)
    assert [f["index"] for f in per] == [6] and per[0]["details"].startswith("[regime A]")
    # failures and unverified rows are still emitted once, regardless of grouping
    rows[3]["success"] = False; rows[3]["error"] = "x"
    rows[7]["verified"] = False
    reasons = [(f["index"], f["reason"]) for f in hs.detect_outliers(rows, 2.0, groups=groups)]
    assert reasons.count((3, "analysis_failed")) == 1 and (7, "unverified") in reasons


def test_outlier_groups_only_for_interleaved_regimes():
    plan = {"regimes": [{"name": "A", "dataset_indices": [0, 2]}, {"name": "B", "dataset_indices": [1, 3]}]}
    inco = {"reduction": {"axis_coherence": {"coherent": False}}}
    co = {"reduction": {"axis_coherence": {"coherent": True}}}
    assert HyperspectralAnalysisAgent._outlier_groups(plan, inco) == {"A": [0, 2], "B": [1, 3]}
    assert HyperspectralAnalysisAgent._outlier_groups(plan, co) is None
    assert HyperspectralAnalysisAgent._outlier_groups(None, inco) is None
    assert HyperspectralAnalysisAgent._outlier_groups({"regimes": [{"name": "only", "dataset_indices": [0, 1]}]}, inco) is None


def test_plan_series_regimes_retries_once_on_unparseable_answer():
    import logging
    calls = []

    class _Model:
        def generate_content(self, contents, **kw):
            calls.append("\n".join(c for c in contents if isinstance(c, str)))
            if len(calls) == 1:
                return "{\"observations\": \"cut off mid"          # truncated JSON
            return json.dumps({"observations": "ok", "series_analysis_plan": {"regimes": [
                {"name": "a", "dataset_indices": [0, 1]}, {"name": "b", "dataset_indices": [2, 3]}]}})

    def parse(resp):
        try:
            return json.loads(resp), None
        except Exception as e:  # noqa: BLE001
            return None, {"error": "Failed to parse valid JSON", "details": str(e)}

    state = {"num_images": 4, "series_metadata": {"variable": "t", "values": [0, 1, 2, 3], "unit": ""},
             "system_info": {}, "skills_loaded": []}
    plan = hs.plan_series_regimes(_Model(), None, None, parse, state, {"scout_data": []}, logging.getLogger("t"))
    assert len(calls) == 2 and "cut off" in calls[1] and "ONLY the JSON object" in calls[1]
    assert [r["dataset_indices"] for r in plan["regimes"]] == [[0, 1], [2, 3]]

    # two failures → one regime, no exception
    calls.clear()
    class _Bad:
        def generate_content(self, contents, **kw):
            calls.append(1); return "not json"
    assert hs.plan_series_regimes(_Bad(), None, None, parse, state, {"scout_data": []}, logging.getLogger("t")) is None
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Deterministic replay gate + parallel replays
# ---------------------------------------------------------------------------

def test_replay_map_gate_rules():
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import _replay_map_gate
    good = np.random.RandomState(0).normal(459.0, 0.3, (8, 8))
    ref = {"min": 458.0, "max": 460.0, "mean": 459.0}
    assert _replay_map_gate(good, None, ref, True) == (True, "")
    # low coverage
    holes = good.copy(); holes[:6, :] = np.nan
    ok, why = _replay_map_gate(holes, None, ref, True)
    assert not ok and "coverage" in why
    # coverage judged within the fit mask when scoped
    mask = np.zeros((8, 8), bool); mask[6:, :] = True
    assert _replay_map_gate(holes, mask, ref, True)[0]
    # collapsed map
    ok, why = _replay_map_gate(np.full((8, 8), 461.1), None, None, True)
    assert not ok and "constant" in why
    # out of the plausible range: anchor [458, 460] widened by one span → [456, 462]
    ok, why = _replay_map_gate(good + 4.0, None, ref, True)
    assert not ok and "plausible range" in why
    assert _replay_map_gate(good + 1.5, None, ref, True)[0]          # inside the widened range
    # the range rule applies to required outputs only; diagnostics need coverage + non-collapse
    assert _replay_map_gate(good + 4.0, None, ref, False)[0]
    # narrow anchor range: widened by at least 0.5 % of the magnitude (459 ± 2.3)
    tight = {"min": 459.0, "max": 459.0, "mean": 459.0}
    assert _replay_map_gate(good + 1.5, None, tight, True)[0]
    assert not _replay_map_gate(good + 4.0, None, tight, True)[0]


def test_locked_replay_uses_deterministic_gate_and_no_llm_review(tmp_path, monkeypatch):
    """A locked replay never calls the LLM reviewers; its maps pass or fail
    the deterministic gate against the anchor's reference stats."""
    from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
    import logging
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    log = logging.getLogger("t")
    script = "def analyze_feature(data, axis):\n    return {'maps': {'Mean_Map': data.mean(axis=2)}, 'units': 'a.u.', 'description': 'd'}\n"
    axis_ok = {"axis_spec": {"axis_2": {"name": "E", "units": "eV", "start": 400, "end": 900}}}

    def run(reference):
        st = {"hspy_data": np.random.rand(5, 5, 8) + 10.0, "original_hspy_data": np.random.rand(5, 5, 8) + 10.0,
              "system_info": dict(axis_ok), "energy_axis": np.linspace(400, 900, 8),
              "settings": {"output_dir": str(tmp_path)}, "max_verification_iterations": 0,
              "iteration_title": "T", "analysis_objective": "obj",
              "reuse_records": [{"target": "mean", "task_success": True, "required_outputs": ["Mean_Map"],
                                 "script": script, "quality_history": {"approved": True}}],
              "replay_reference": reference}

        class _NoLLM:
            def generate_content(self, *a, **k):
                raise AssertionError("no LLM call expected on a locked replay")
        st = hc.SelectRefinementTargetController(_NoLLM(), log, None, None, parse_fn=lambda r: ({}, None)).execute(st)
        ctrl = hc.RunDynamicAnalysisController(_NoLLM(), log, None, None, parse_fn=lambda r: ({}, None))
        ctrl._review_required_output = lambda *a, **k: (_ for _ in ()).throw(AssertionError("reviewer called"))
        ctrl._check_result_visually = lambda *a, **k: (_ for _ in ()).throw(AssertionError("visual QC called"))
        return ctrl.execute(st)

    ok = run({"Mean_Map": {"min": 10.2, "max": 10.8, "mean": 10.5}})
    assert [m["name"] for m in ok["custom_analysis_metadata_list"]] == ["Mean_Map"]
    assert ok["dynamic_analysis_records"][0]["task_success"] is True
    bad = run({"Mean_Map": {"min": 458.0, "max": 460.0, "mean": 459.0}})   # method breakdown
    assert not (bad.get("custom_analysis_metadata_list") or [])
    assert bad["dynamic_analysis_records"][0]["task_success"] is False


def test_resolve_series_workers(monkeypatch):
    monkeypatch.delenv("SCILINK_HS_SERIES_WORKERS", raising=False)
    assert hs.resolve_series_workers(None) == 1 and hs.resolve_series_workers(0) == 1
    assert hs.resolve_series_workers(4) == 4
    monkeypatch.setenv("SCILINK_HS_SERIES_WORKERS", "3")
    assert hs.resolve_series_workers(None) == 3 and hs.resolve_series_workers(2) == 2
    monkeypatch.setenv("SCILINK_HS_SERIES_WORKERS", "junk")
    assert hs.resolve_series_workers(None) == 1


def test_light_synthesis_skips_critic_for_replay_children(tmp_path):
    agent = HyperspectralAnalysisAgent(api_key="sk-dummy", output_dir=str(tmp_path / "a"),
                                       enable_human_feedback=False)
    full = [c.__class__.__name__ for c in agent._synthesis_controllers()]
    assert "RunSelfReflectionController" in full and "ApplyReflectionUpdatesController" in full
    child = agent._make_unit_agent(tmp_path / "u", "replay", human_feedback=False)
    light = [c.__class__.__name__ for c in child._synthesis_controllers()]
    assert "RunSelfReflectionController" not in light and "ApplyReflectionUpdatesController" not in light
    assert light[0] == "BuildHolisticSynthesisPromptController" and "GenerateHTMLReportController" in light
    anchor = agent._make_unit_agent(tmp_path / "v", "anchor", human_feedback=False)
    assert len(anchor._synthesis_controllers()) == len(full)


def test_parallel_replays_run_on_worker_pool(tmp_path, monkeypatch):
    """series_workers > 1: replays are queued after the anchors and run on
    the pool (threads under SCILINK_HS_SERIES_POOL=thread so the stubbed
    pipeline applies); rows come back in series order with the anchor's
    reference stats handed to each replay."""
    monkeypatch.setenv("SCILINK_HS_SERIES_POOL", "thread")
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls, fail_on_replay={"cube_T400"})
    orig = HyperspectralAnalysisAgent._run_analysis_pipeline
    seen_refs = {}

    def spy(self, data_path, *a, **k):
        seen_refs[Path(data_path).stem] = k.get("replay_reference")
        return orig(self, data_path, *a, **k)
    monkeypatch.setattr(HyperspectralAnalysisAgent, "_run_analysis_pipeline", spy)
    agent, out = _agent(tmp_path, calls, monkeypatch)
    paths = _cubes(tmp_path)
    res = agent.analyze(paths, system_info=dict(AXIS), series_workers=3,
                        series_metadata={"variable": "temperature", "values": [300, 350, 400, 450, 500, 550], "unit": "K"})
    roles = {c["name"]: c["role"] for c in calls.pipeline if c["role"] != "refit"}
    assert roles["cube_T300"] == "anchor" and all(roles[f"cube_T{t}"] == "replay" for t in (350, 400, 450, 500, 550))
    assert res["summary"]["series_workers"] == 3
    assert [r["index"] for r in res["individual_results"]] == [0, 1, 2, 3, 4, 5]
    assert res["summary"]["successful_analyses"] == 6 and res["summary"]["refitted_count"] == 1
    assert res["series_features"]["Mean_Map_mean"] == MEANS
    # every replay received the anchor's per-map stats as the gate reference
    assert seen_refs["cube_T350"] == {"Mean_Map": {"min": 0.0, "max": 2.0, "mean": 1.0},
                                      "R2": {"mean": 0.98}}
    for i in (1, 3, 4, 5):
        assert (out / f"dataset_{i:04d}" / "replay.log").is_file()
    assert res["individual_results"][1]["reuse_validity"]["verdict"] == "good"


# ---------------------------------------------------------------------------
# Regime-plan human gate + anchor codegen data facts
# ---------------------------------------------------------------------------

def test_regime_plan_gate_accepts_on_enter_and_revises_on_feedback(tmp_path, monkeypatch):
    """Co-pilot/autopilot: the plan is shown; Enter accepts; text is fed back
    to the planner, which revises; the revised plan is shown again."""
    import scilink.hitl as hitl
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    plan = {"rationale": "edge shifts", "regimes": [
        {"name": "low_T", "dataset_indices": [0, 1, 2, 3], "description": "L3 ~458"},
        {"name": "high_T", "dataset_indices": [4, 5], "description": "shifted"}]}
    agent, out = _agent(tmp_path, calls, monkeypatch, plan=plan)
    agent.enable_human_feedback = True                      # co-pilot / autopilot

    answers = iter(["treat the whole series as one regime", ""])
    asked = []

    def fake_ask(prompt, **kw):
        asked.append(kw.get("origin"))
        return next(answers)
    monkeypatch.setattr(hitl, "request_human_feedback", fake_ask)

    # the fake model returns a single-regime plan once it sees analyst feedback
    orig_gen = agent.model.generate_content

    def gen(contents, **kw):
        text = "\n".join(c for c in contents if isinstance(c, str))
        if "Analyst feedback on the previous plan" in text:
            assert "treat the whole series as one regime" in text and "low_T" in text
            return json.dumps({"observations": "ok", "series_analysis_plan": {
                "rationale": "per analyst", "regimes": [{"name": "all", "dataset_indices": [0, 1, 2, 3, 4, 5]}]}})
        return orig_gen(contents, **kw)
    agent.model.generate_content = gen

    res = agent.analyze(_cubes(tmp_path), system_info=dict(AXIS),
                        series_metadata={"variable": "temperature", "values": [300, 350, 400, 450, 500, 550], "unit": "K"})
    assert [a["stage"] for a in asked] == ["series_regime_plan", "series_regime_plan"]
    assert [a["round"] for a in asked] == [1, 2]
    assert res["summary"]["regimes"] == 1 and res["summary"]["regime_anchors"] == {"all": 0}
    assert agent.state["human_feedback_log"] == [{"stage": "series_regime_plan",
                                                  "feedback": "treat the whole series as one regime"}]
    # anchors of the (single) regime and replays as usual
    assert [c["role"] for c in calls.pipeline] == ["anchor"] + ["replay"] * 5


def test_regime_plan_gate_skipped_when_feedback_off_or_eof(tmp_path, monkeypatch):
    import scilink.hitl as hitl
    calls = _Calls()
    _install_fake_pipeline(monkeypatch, calls)
    plan = {"regimes": [{"name": "a", "dataset_indices": [0, 1, 2]}, {"name": "b", "dataset_indices": [3, 4, 5]}]}
    agent, out = _agent(tmp_path, calls, monkeypatch, plan=plan)
    asked = []
    monkeypatch.setattr(hitl, "request_human_feedback", lambda *a, **k: asked.append(1) or "")
    res = agent.analyze(_cubes(tmp_path), system_info=dict(AXIS),
                        series_metadata={"variable": "t", "values": list(range(6)), "unit": ""})
    assert asked == [] and res["summary"]["regimes"] == 2          # autonomous: no gate
    # EOF on the prompt keeps the current plan
    agent2, out2 = _agent(tmp_path / "b", calls, monkeypatch, plan=plan)
    agent2.enable_human_feedback = True
    def boom(*a, **k):
        raise EOFError
    monkeypatch.setattr(hitl, "request_human_feedback", boom)
    res2 = agent2.analyze(_cubes(tmp_path / "b"), system_info=dict(AXIS),
                          series_metadata={"variable": "t", "values": list(range(6)), "unit": ""})
    assert res2["summary"]["regimes"] == 2


def test_render_regime_plan_lists_regimes_anchors_and_values():
    plan = {"rationale": "r", "regimes": [{"name": "A", "dataset_indices": [0, 2], "description": "d"},
                                          {"name": "B", "dataset_indices": [1]}],
            "transition_points": [{"between_indices": [0, 1], "description": "jump"}]}
    txt = hs.render_regime_plan(plan, {"variable": "T", "values": [300, 350, 400], "unit": "K"},
                                {"reduction": {"change_point": 325.0, "change_sharpness": 0.9,
                                               "axis_coherence": {"coherent": False}}}, 3)
    assert "1. A" in txt and "0 (T=300 K), 2 (T=400 K)" in txt and "anchor: dataset 0" in txt
    assert "2. B" in txt and "NOT coherent" in txt and "jump" in txt
    assert "1 regime" in hs.render_regime_plan(None, {}, None, 4)


def test_data_facts_block_reports_measured_peaks_and_noise():
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        _render_data_facts, build_code_generation_prompt)
    E = np.linspace(450, 570, 120)
    rng = np.random.RandomState(0)
    spec = np.exp(-0.5 * ((E - 462.1) / 1.2) ** 2) + 0.7 * np.exp(-0.5 * ((E - 468.0) / 1.4) ** 2) + 0.08
    cube = spec[None, None, :] + rng.normal(0, 0.03, (12, 12, 120))
    txt = _render_data_facts(cube, E, "eV")
    assert "DATA FACTS" in txt and "144 spectra" in txt
    assert "462." in txt and "468" in txt and "sigma of the mean" in txt
    import re
    fwhm = [float(x) for x in re.findall(r"width ≲ ([0-9.]+) eV", txt)]
    assert fwhm and all(1.5 < f < 9.0 for f in fwhm), fwhm     # bounded, not spanning both peaks
    assert "measurable in aggregate" in txt and "not on literature values" in txt
    # featureless cube: no peaks, honest wording
    flat = 0.1 + rng.normal(0, 0.03, (10, 10, 120))
    assert "featureless" in _render_data_facts(flat, E, "eV")
    # the block lands in the codegen prompt ahead of the measurability gate
    prompt = build_code_generation_prompt(target_desc="t", h=12, w=12, e=120, axis_units="eV",
                                          axis_start=450, axis_end=570, processing_note="raw",
                                          data_facts=txt)
    assert prompt.index("DATA FACTS") < prompt.index("MEASURABILITY GATE")
    assert "DATA FACTS" not in build_code_generation_prompt(target_desc="t", h=1, w=1, e=8, axis_units="eV",
                                                            axis_start=0, axis_end=1, processing_note="raw")
