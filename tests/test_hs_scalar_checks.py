"""A hyperspectral scalar that comes from a fit is checked (#722 B1).

#733 kept every claim of a run that reported scalars provisional, because no
gate looked at them; the #722 swarm run had posted "no change" read off fits
railed at their window edge with depths of 1e-12. A script now returns a fitted
number as ``{"value", "role", "bounds"}``. A deterministic fit-health check runs
on it: a value at the bound it was fitted within fails (the curve agent's
pinned-at-bound rule), and so does an amplitude at zero (on a zero lower bound,
or far below the field mean's noise — the floor that rule deliberately skips).
A number that passes is gated, so a run whose numbers all pass earns its claims
back; a number that fails is reported as a failed fit — no value in the feature
table, named on the board and in the synthesis; a plain number stays unchecked,
exactly as before.

  conda run -n scilink python -m pytest tests/test_hs_scalar_checks.py -q
"""
import json
import logging
import os

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
from scilink.agents.exp_agents.controllers import hyperspectral_series as hs
from scilink.agents.exp_agents.controllers.hyperspectral_controllers import RunDynamicAnalysisController
from scilink.agents.meta_agent import reactions

import test_ungated_outputs as tu

# One band fitted per region, the numbers returned as declared fits: a healthy
# position and depth, a position railed at its window edge, a depth on its zero
# floor, a ratio resting on its bound (legitimate), and a plain number.
SCRIPT_DECLARED = (
    "def analyze_feature(data, energy_axis):\n"
    "    import numpy as np\n"
    "    d = np.asarray(data)\n"
    "    return {'maps': {'Depth_Map': d.mean(axis=2)}, 'units': 'a.u.', 'description': 'depth',\n"
    "            'scalars': {\n"
    "                'Band1_Position_nm': {'value': 600.4, 'role': 'position', 'bounds': [570, 630]},\n"
    "                'Band1_Depth': {'value': 0.31, 'role': 'amplitude', 'bounds': [0, 2]},\n"
    "                'Band2_Position_nm': {'value': 435.0003, 'role': 'position', 'bounds': [435, 470]},\n"
    "                'Band2_Depth': {'value': 1e-12, 'role': 'amplitude', 'bounds': [0, 2]},\n"
    "                'Mix_Fraction': {'value': 1.0, 'role': 'ratio', 'bounds': [0, 1]},\n"
    "                'N_Pixels': 36}}\n")
SCRIPT_ALL_PASS = (
    "def analyze_feature(data, energy_axis):\n"
    "    import numpy as np\n"
    "    return {'maps': {'Depth_Map': np.asarray(data).mean(axis=2)}, 'units': 'a.u.', 'description': 'depth',\n"
    "            'scalars': {'Band1_Position_nm': {'value': 600.4, 'role': 'position', 'bounds': [570, 630]},\n"
    "                        'Band1_Depth': {'value': 0.31, 'role': 'amplitude', 'bounds': [0, 2]}}}\n")
SCRIPT_ONE_FAILS = SCRIPT_ALL_PASS.replace("'value': 600.4", "'value': 629.9999")


def test_check_scalar_rules():
    facts = {"sigma_mean": 0.001}
    assert hc._check_scalar(600.4, "position", [570, 630], facts)[0] == "passed"
    v, why = hc._check_scalar(629.9999, "position", [570, 630], facts)
    assert v == "failed" and "upper bound 630" in why
    assert hc._check_scalar(1e-12, "amplitude", [0, 2], facts)[0] == "failed"       # zero floor, by the noise
    # a huge declared range never makes a real height "zero" (seen on real data: bounds [0, ~1e6])
    assert hc._check_scalar(0.617, "amplitude", [0, 1e6], facts)[0] == "passed"
    v, why = hc._check_scalar(1e-9, "amplitude", None, facts)                        # no bounds: the noise scale
    assert v == "failed" and "sigma" in why
    assert hc._check_scalar(0.3, "amplitude", None, facts)[0] == "passed"
    assert hc._check_scalar(1.0, "ratio", [0, 1], facts)[0] is None                  # a ratio may rest on its bound
    # no noise estimate (the facts can be empty): bounds alone never certify an amplitude
    assert hc._check_scalar(1e-12, "amplitude", [0, 2], {}) == (None, "no noise estimate for the zero check")
    # between the failed and the certified bars: unchecked, not passed (a fit to noise lands here)
    assert hc._check_scalar(1e-5, "amplitude", None, facts)[0] is None
    assert hc._check_scalar(1e-5, "amplitude", [0, 2], facts)[0] is None
    assert hc._check_scalar(6e-4, "amplitude", None, facts)[0] == "passed"           # 0.6 sigma
    assert hc._check_scalar(12.0, "count", None, facts)[0] is None                   # no check applies
    assert hc._check_scalar(5.0, "width", [5.0, 5.0], facts)[0] is None              # degenerate bounds ignored


def test_the_controller_checks_declared_fits_and_reports_failures_as_no_value(tmp_path):
    meta, records = tu._controller_run(tmp_path, SCRIPT_DECLARED)
    by = {m["name"]: m for m in meta}
    assert by["Band1_Position_nm"]["gated"] is True and by["Band1_Position_nm"]["scalar"] == 600.4
    assert by["Band1_Depth"]["gated"] is True and by["Band1_Depth"]["check"].startswith("passed")
    for name, why in (("Band2_Position_nm", "lower bound 435"), ("Band2_Depth", "no feature was fitted")):
        m = by[name]
        assert m["gated"] is False and m["scalar"] is None and why in m["check"]
        assert "FAILED FIT" in m["description"] and isinstance(m["raw_value"], float)
    assert by["Mix_Fraction"]["gated"] is False and by["Mix_Fraction"]["check"].startswith("unchecked")
    assert by["N_Pixels"]["gated"] is False and by["N_Pixels"]["scalar"] == 36.0     # a plain number, as before
    # a failed fit never reaches the feature table as a number
    flat = hs.flatten_feature_records(meta)
    assert "Band1_Position_nm" in flat and "Band2_Position_nm" not in flat and "Band2_Depth" not in flat


def test_a_run_whose_numbers_all_pass_earns_its_claims_back(tmp_path, monkeypatch):
    result, recs = tu._run_task_through_the_board(tmp_path, SCRIPT_ALL_PASS, monkeypatch)
    row = result["analyses"][0]
    assert row["ungated_outputs"] == [] and row["failed_outputs"] == []
    claims = [r for r in recs if r["kind"] == "claim"]
    assert claims and all(c["status"] == "verified" for c in claims)
    assert all(reactions.matches(tu.ON_A_CLAIM, c) for c in claims)


def test_a_failed_number_is_named_as_failed_not_unchecked(tmp_path, monkeypatch):
    result, recs = tu._run_task_through_the_board(tmp_path, SCRIPT_ONE_FAILS, monkeypatch)
    row = result["analyses"][0]
    assert row["failed_outputs"] == ["Band1_Position_nm"]
    claims = [r for r in recs if r["kind"] == "claim"]
    assert claims and all(c["status"] == "provisional" for c in claims)
    why = claims[0]["evidence"]["gate"]
    assert "FAILED their fit-health check (Band1_Position_nm)" in why and "no gate checked" not in why


def test_the_series_synthesis_names_checked_failed_and_unchecked_numbers(tmp_path):
    meta, _ = tu._controller_run(tmp_path, SCRIPT_DECLARED)
    unit = {"index": 0, "name": "film_0", "success": True, "feature_records": meta,
            "unit_verdict": {"verified": True}}
    prompt = hs.build_series_synthesis_prompt({"series_results": [unit], "flagged_images": [],
                                               "locked_config": {"targets": [{"target": "t"}]}})
    text = "\n".join(p for p in prompt if isinstance(p, str))
    assert "passed their fit-health check (inside their bounds, non-zero): Band1_Depth, Band1_Position_nm" in text
    assert "FAILED their check" in text and "Band2_Depth on film_0" in text and "Band2_Position_nm on film_0" in text
    assert "NO gate checked: Mix_Fraction, N_Pixels" in text


def test_the_contract_reaches_the_retry_prompt(tmp_path):
    """The declared form is in the code-generation contract, and every retry
    is built on that same base prompt: the first attempt returns a required
    map all-NaN, the retry carries the contract too."""
    prompts = []
    first = ("def analyze_feature(data, energy_axis):\n    import numpy as np\n"
             "    d = np.asarray(data)\n"
             "    return {'maps': {'Depth_Map': np.full(d.shape[:2], np.nan)}, 'units': 'a.u.', 'description': 'x'}\n")
    scripts = iter([first, SCRIPT_ALL_PASS])

    class _Model:
        def generate_content(self, contents, **kw):
            prompts.append(contents if isinstance(contents, str) else json.dumps(contents, default=str))
            return json.dumps({"code": next(scripts)})
    cube, E = tu._cube()
    ctrl = RunDynamicAnalysisController(model=_Model(), logger=logging.getLogger("t"),
                                        generation_config=None, safety_settings=None,
                                        parse_fn=lambda r: (json.loads(r), None), executor_timeout=60)
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")
    state = ctrl.execute({
        "refinement_decision": {"refinement_needed": True, "requires_custom_code": True,
                                "targets": [{"type": "custom_code", "description": "fit the band",
                                             "required_outputs": ["Depth_Map"]}]},
        "hspy_data": cube, "original_hspy_data": cube, "energy_axis": E,
        "system_info": {"axis_spec": {"axis_2": {"name": "wavelength", "units": "nm", "start": 400, "end": 900}}},
        "settings": {"output_dir": str(tmp_path)}, "iteration_title": "T", "analysis_images": [],
        "error_dict": None, "max_verification_iterations": 1})
    assert len(prompts) == 2 and state["dynamic_analysis_records"][0]["task_success"] is True
    contract = "is checked only when you return it as"
    assert all(contract in p for p in prompts)
    assert "PREVIOUS ATTEMPT FAILED" in prompts[1]


def test_a_value_outside_its_declared_bounds_is_never_passed():
    """Seen on the real white-light cubes: a script declared a Gaussian's SIGMA
    bounds [0.3, 17] for the FWHM it returned (40.03 = 2.355 x 17.0, i.e.
    pinned at its sigma bound). A value cannot leave the bounds it was fitted
    within, so those bounds are another parameterisation's: not checkable,
    reported as unchecked — never as passed, which certified a railed width."""
    v, why = hc._check_scalar(40.03, "width", [0.3, 17], {"sigma_mean": 0.001})
    assert v is None and "outside its declared bounds" in why
    v, why = hc._check_scalar(1e-12, "amplitude", [0.3, 17], {"sigma_mean": 0.001})
    assert v == "failed"                                   # the zero-amplitude rule still applies
    assert hc._check_scalar(16.9, "width", [0.3, 17], {"sigma_mean": 0.001})[0] == "failed"   # at the bound


def test_a_failed_number_on_a_fresh_code_series_row_is_not_aliased(tmp_path, monkeypatch):
    """Through the real series driver: the schema source reports Band2_Depth;
    a later regime's anchor (fresh code, locked targets) reports it as a FAILED
    fit beside a plain Band2_Depth_uncertainty. The locked-schema completion
    once aliased the uncertainty into the depth's column; the failed column now
    stays empty, and the failure is named, not filled."""
    import test_hs_series as ths
    from scilink.agents.exp_agents import hyperspectral_analysis_agent as hsa
    ok = ("def analyze_feature(data, energy_axis):\n    import numpy as np\n"
          "    return {'maps': {'Depth_Map': np.asarray(data).mean(axis=2)}, 'units': 'a.u.', 'description': 'd',\n"
          "            'scalars': {'Band2_Depth': {'value': 0.31, 'role': 'amplitude', 'bounds': [0, 2]}}}\n")
    bad = ("def analyze_feature(data, energy_axis):\n    import numpy as np\n"
           "    return {'maps': {'Depth_Map': np.asarray(data).mean(axis=2)}, 'units': 'a.u.', 'description': 'd',\n"
           "            'scalars': {'Band2_Depth': {'value': 1e-12, 'role': 'amplitude', 'bounds': [0, 2]},\n"
           "                        'Band2_Depth_uncertainty': 0.04}}\n")
    meta_ok, rec_ok = tu._controller_run(tmp_path / "ok", ok)
    meta_bad, rec_bad = tu._controller_run(tmp_path / "bad", bad)
    assert any(str(m.get("check", "")).startswith("failed") for m in meta_bad)

    def pipeline(self, data_path, system_info, instruction_prompt, reuse_records=None, **kw):
        meta, recs = (meta_bad, rec_bad) if self._series_role == "regime_anchor" else (meta_ok, rec_ok)
        recs = [{**recs[0], "locked_replay": bool(reuse_records), "replay_verbatim": True}]
        (self.output_dir / "dynamic_analysis_records.json").write_text(json.dumps(recs, default=str))
        return {"detailed_analysis": "d", "extracted_features": meta, "dynamic_analysis_records": recs,
                "scientific_claims": []}, None
    monkeypatch.setenv("SCILINK_HS_SERIES_POOL", "thread")
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_run_analysis_pipeline", pipeline)
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_maybe_bank_scripts", lambda *a, **k: [])
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_maybe_stage_t2_solutions", lambda *a, **k: [])
    monkeypatch.setattr(hsa.HyperspectralAnalysisAgent, "_auto_select_skills", lambda *a, **k: [])
    plan = {"rationale": "x", "regimes": [{"name": "A", "dataset_indices": [0, 1]},
                                          {"name": "B", "dataset_indices": [2, 3]}]}
    agent = hsa.HyperspectralAnalysisAgent(api_key="sk-dummy", output_dir=str(tmp_path / "series"),
                                           enable_human_feedback=False, executor_timeout=120)
    agent.model = ths._FakeModel(ths._Calls(), plan=plan)
    paths = ths._cubes(tmp_path)[:4]
    res = agent.analyze(paths, system_info=dict(ths.AXIS),
                        series_metadata={"variable": "dose", "values": [1, 2, 3, 4], "unit": "mC"})
    rows = json.loads(open(res["series_results_path"]).read())["results"]     # the full per-unit rows
    row = next(r for r in rows if r["index"] == 2)                             # regime B's anchor
    assert row.get("role") == "regime_anchor"
    assert "Band2_Depth" not in row["extracted_features"]                       # no value, not the uncertainty
    assert row["extracted_features"].get("Band2_Depth_uncertainty") == 0.04
    assert "Band2_Depth" not in (row.get("schema_aliases") or {})
    assert row["unit_verdict"]["failed_checks"] == ["Band2_Depth"]


def test_the_live_frame_names_a_tracked_number_that_failed_its_check():
    """A hyperspectral live frame whose tracked scalar failed its check: the
    number drops out of the frame's features (no value), and the frame record
    names it in ``withheld`` rather than dropping it silently."""
    from scilink.live.modality import HyperspectralModality
    records = [{"name": "Peak_Position", "stats": {"min": 1, "max": 2, "mean": 1.5}},
               {"name": "Band_Depth", "scalar": None, "raw_value": 1e-12, "gated": False,
                "check": "failed: 1.0e-09 sigma of the field mean's noise: no feature was fitted"}]
    mod = HyperspectralModality()
    result = {"status": "success", "feature_records": records}
    assert "Band_Depth" not in mod.features(result)
    v = mod.validity(result)
    assert v["withheld"] == ["Band_Depth"] and v["verdict"] == "good"
