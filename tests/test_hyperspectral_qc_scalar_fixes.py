"""QC scale-invariance + scalar-deliverables channel (TaS2 session round 2).

A. COVERAGE — the required-output review's valid-coverage check used an
   absolute `> 1e-9` floor, reporting "0.0% coverage" to the judge for dense,
   physically real amperes-scale maps (a phantom collapse the judge then
   correctly failed). Now exact-zero-only, scale-free.
B. TASK CONTRACT — the reviewer never saw the plan's per-target instruction,
   so a declared expectation ("near-zero ZBC is physically expected") could
   not inform the verdict. The target description now travels into the
   review prompt.
C. SCALARS — global numeric deliverables (correlation coefficients,
   region-integrated values) had no channel: computed values died in
   unconsumed return keys. `scalars` is now part of the code contract,
   committed with the attempt, reported to synthesis, and written to
   features.csv.
D. FLUX TABLE — fixed-point ".1f" rendering showed every band of a
   tiny-native-scale signal as 0.0 ("no signal" misinformation). Now %g.
"""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from scilink.agents.exp_agents.controllers import hyperspectral_controllers as hc
from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
    BuildHolisticSynthesisPromptController,
    _extract_scalar_records,
    _map_valid_coverage,
    _render_band_flux_table,
)
from scilink.agents.exp_agents.instruct import (
    SPECTROSCOPY_REFINEMENT_INSTRUCTIONS,
    SPECTROSCOPY_RESULT_REVIEW_INSTRUCTIONS,
)


LOGGER = logging.getLogger("test")
RNG = np.random.RandomState(0)

AXIS_OK = {
    "axis_spec": {
        "axis_2": {"name": "Bias", "units": "V", "start": 1.0, "end": -1.0},
        "signal_is_nonnegative": False,
    }
}


# ---------------------------------------------------------------------------
# A. Coverage is scale-invariant
# ---------------------------------------------------------------------------

def test_coverage_full_on_amperes_scale_map():
    # The session's failure: dense ~1e-12 maps reported as 0.0% coverage.
    m = (RNG.rand(64, 64) + 0.5) * 1e-12
    cov, n = _map_valid_coverage(m)
    assert cov == 100.0 and n == m.size


def test_coverage_scale_invariant():
    m = RNG.rand(32, 32) + 0.5
    m[:8, :8] = 0.0  # genuine masked region
    cov1, _ = _map_valid_coverage(m)
    cov2, _ = _map_valid_coverage(m * 1e-13)
    assert cov1 == pytest.approx(cov2)
    assert cov1 == pytest.approx(100.0 * (1 - 64 / 1024))


def test_coverage_counts_zeros_and_nans_as_invalid():
    m = np.ones((10, 10))
    m[0, :] = 0.0
    m[1, :] = np.nan
    cov, n = _map_valid_coverage(m)
    assert n == 80 and cov == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# B. The plan's per-target contract reaches the reviewer
# ---------------------------------------------------------------------------

def test_review_template_has_task_contract_block():
    assert "{target_context}" in SPECTROSCOPY_RESULT_REVIEW_INSTRUCTIONS
    assert "TASK CONTRACT" in SPECTROSCOPY_RESULT_REVIEW_INSTRUCTIONS
    # the judging principle: declared expectations beat generic priors
    assert "DECLARED expectations" in SPECTROSCOPY_RESULT_REVIEW_INSTRUCTIONS
    assert ("never excuses a genuine methodological flaw"
            in SPECTROSCOPY_RESULT_REVIEW_INSTRUCTIONS)


def test_reviewer_receives_target_context():
    captured = {}

    class _Model:
        def generate_content(self, contents, **kw):
            captured["prompt"] = contents[0]
            return "resp"

    ctrl = hc.RunDynamicAnalysisController(
        _Model(), LOGGER, generation_config=None, safety_settings=None,
        parse_fn=lambda r: ({"valid": True}, None),
    )
    plan_text = ("Emit the ZBC map; do NOT gate it on magnitude — "
                 "near-zero inside the gap is physically expected.")
    ok, _ = ctrl._review_required_output_one(
        b"img", "code", None, {}, "objective text", "Zbc_Map",
        "value range [...]", "", target_context=plan_text)
    assert ok is True
    assert plan_text in captured["prompt"]
    assert "objective text" in captured["prompt"]


def test_reviewer_defaults_without_target_context():
    captured = {}

    class _Model:
        def generate_content(self, contents, **kw):
            captured["prompt"] = contents[0]
            return "resp"

    ctrl = hc.RunDynamicAnalysisController(
        _Model(), LOGGER, generation_config=None, safety_settings=None,
        parse_fn=lambda r: ({"valid": True}, None),
    )
    ctrl._review_required_output_one(
        b"img", "code", None, {}, "obj", "M", "summary", "")
    assert "(not provided)" in captured["prompt"]


# ---------------------------------------------------------------------------
# C. Scalars channel
# ---------------------------------------------------------------------------

def test_extract_scalar_records_validates_and_units():
    result = {"scalars": {
        "corr_NGS_topo": -0.42,
        "n_pixels": np.int64(28224),          # numpy scalar → coerced
        "bad_nan": float("nan"),              # dropped
        "bad_array": np.arange(3),            # dropped
        "bad_str": "high",                    # dropped
    }, "units": {"corr_NGS_topo": "dimensionless"}}
    recs = _extract_scalar_records(result, result["units"])
    by_name = {r["name"]: r for r in recs}
    assert set(by_name) == {"corr_NGS_topo", "n_pixels"}
    assert by_name["corr_NGS_topo"]["scalar"] == pytest.approx(-0.42)
    assert by_name["corr_NGS_topo"]["units"] == "dimensionless"
    assert all("scalar" in r for r in recs)


def test_extract_scalar_records_absent_or_invalid_channel():
    assert _extract_scalar_records({}, "a.u.") == []
    assert _extract_scalar_records({"scalars": [1, 2]}, "a.u.") == []


def test_extract_scalar_records_capped():
    result = {"scalars": {f"s{i}": float(i) for i in range(100)}}
    assert len(_extract_scalar_records(result, "a.u.")) == hc._MAX_SCALARS_PER_TASK


def test_codegen_contract_documents_scalars():
    # The single-source builder must advertise the channel to the code LLM.
    import inspect
    src = inspect.getsource(hc)
    block = src[src.index("### REQUIRED RETURN FORMAT"):]
    assert '"scalars"' in block[:1500]
    assert "feature table" in block[:2200]


def test_planner_prompt_routes_globals_to_scalars_channel():
    block = SPECTROSCOPY_REFINEMENT_INSTRUCTIONS[
        SPECTROSCOPY_REFINEMENT_INSTRUCTIONS.index("Required outputs"):]
    assert "`scalars` channel" in block
    assert "Never promise a global number as a map key" in block


def test_write_results_file_emits_scalar_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    from scilink.agents.exp_agents.feature_table import write_feature_table

    agent = HyperspectralAnalysisAgent(
        api_key="dummy", base_url="http://localhost:1",
        output_dir=str(tmp_path), enable_human_feedback=False,
    )
    response = {"status": "success", "extracted_features": [
        {"name": "Gap_Width", "units": "V",
         "stats": {"min": 0.01, "max": 0.09, "mean": 0.04}},
        {"name": "corr_NGS_topography", "units": "dimensionless",
         "description": "Global scalar", "scalar": -0.37},
    ]}
    agent._write_results_file(response)

    feats = json.loads((tmp_path / "analysis_results.json").read_text())[
        "extracted_features"]
    assert feats["corr_NGS_topography_dimensionless"] == pytest.approx(-0.37)
    assert feats["Gap_Width_mean_V"] == pytest.approx(0.04)

    csv_path = write_feature_table(tmp_path)
    text = open(csv_path).read()
    assert "corr_NGS_topography_dimensionless" in text
    assert "-0.37" in text


def test_synthesis_prompt_renders_scalar_value():
    ctrl = BuildHolisticSynthesisPromptController(LOGGER)
    state = {
        "all_iteration_results": [{
            "iteration_title": "Global_Analysis",
            "iteration_analysis_text": "text",
            "analysis_images": [],
            "custom_analysis_metadata_list": [
                {"name": "corr_NGS_topography", "units": "dimensionless",
                 "description": "Global scalar", "scalar": -0.37},
                {"name": "Gap_Width", "units": "V", "description": "map",
                 "stats": {"min": 0.0, "max": 0.1, "mean": 0.05}},
            ],
        }],
        "system_info": {}, "instruction_prompt": "x",
        "result_json": None, "error_dict": None,
    }
    state = ctrl.execute(state)
    text = "".join(p for p in state["final_prompt_parts"] if isinstance(p, str))
    assert "Value: -0.37" in text
    assert "corr_NGS_topography" in text
    assert "Statistics: Min 0.00, Max 0.10, Mean 0.05" in text


def test_scalars_never_satisfy_required_outputs_gate():
    # The required-outputs gate counts MAP records only; a scalar with the
    # same name as a required map must not sneak the gate open. The gate
    # reads names from current_run_valid_meta (maps); scalars are staged in
    # a separate list — verify the source enforces that separation.
    import inspect
    src = inspect.getsource(hc.RunDynamicAnalysisController._run_attempt)
    assert "current_run_scalar_meta = _extract_scalar_records" in src
    # valid_names (the gate input) is built from map meta only
    assert "valid_names = {m['name'] for m in current_run_valid_meta}" in src


# ---------------------------------------------------------------------------
# Retry economy: identity-skip + passed-outputs pin
# ---------------------------------------------------------------------------

def test_retry_feedback_pins_passed_outputs():
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        _codegen_retry_feedback,
    )
    fb = _codegen_retry_feedback(1, "bad map", passed_names=["B_Map", "A_Map"])
    assert "PASSED review in the failed attempt: ['A_Map', 'B_Map']" in fb
    assert "UNCHANGED" in fb
    assert "PASSED review" not in _codegen_retry_feedback(1, "bad map")


def test_identity_skip_and_pin_across_attempts(tmp_path, monkeypatch):
    """Attempt 1: Good_Map passes, Bad_Map rejected -> retry. Attempt 2
    reproduces Good_Map identically -> its review verdict is reused (no
    second review), and the retry prompt carries the passed-outputs pin."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")

    SCRIPT = '''
def analyze_feature(data, axis):
    m = data.mean(axis=2)
    return {"maps": {"Good_Map": m, "Bad_Map": m * 2},
            "units": "a.u.", "description": "test"}
'''
    codegen_prompts = []

    class _Model:
        def generate_content(self, contents, **kw):
            codegen_prompts.append(contents if isinstance(contents, str)
                                   else str(contents))
            return json.dumps({"code": SCRIPT})

    ctrl = hc.RunDynamicAnalysisController(
        _Model(), LOGGER, generation_config=None, safety_settings=None,
        parse_fn=lambda r: ({}, None),
    )

    review_calls = []

    def fake_review(dashboard_bytes, code_str, mean_spec, system_info,
                    objective, feature_name, summary, tools_desc, **kw):
        review_calls.append(feature_name)
        if feature_name == "Bad_Map" and review_calls.count("Bad_Map") == 1:
            return False, "wrong estimator. Corrective direction: fix it"
        return True, ""

    ctrl._review_required_output = fake_review
    ctrl._check_result_visually = lambda *a, **k: (True, "")

    state = {
        "hspy_data": np.random.rand(5, 5, 8) * 1e-12,
        "original_hspy_data": np.random.rand(5, 5, 8) * 1e-12,
        "system_info": dict(AXIS_OK),
        "energy_axis": np.linspace(1, -1, 8),
        "settings": {"output_dir": str(tmp_path)},
        "refinement_decision": {
            "requires_custom_code": True,
            "targets": [{
                "description": "two maps", "type": "custom_code",
                "required_outputs": ["Good_Map", "Bad_Map"],
            }],
        },
        "iteration_title": "T",
        "analysis_objective": "obj",
    }
    state = ctrl.execute(state)

    names = [m["name"] for m in state.get("custom_analysis_metadata_list") or []]
    assert sorted(names) == ["Bad_Map", "Good_Map"]
    # Good_Map reviewed ONCE (attempt 2 reused the verdict); Bad_Map twice.
    assert review_calls.count("Good_Map") == 1
    assert review_calls.count("Bad_Map") == 2
    # The second codegen prompt pinned the passing output AND anchored the
    # prior script (warm level).
    assert len(codegen_prompts) == 2
    assert "PASSED review in the failed attempt: ['Good_Map']" in codegen_prompts[1]
    assert "ADAPT IT, DO NOT REWRITE FROM SCRATCH" in codegen_prompts[1]


def test_warm_retry_anchors_script_and_carries_history():
    """Curve/image parity: warm retries adapt the prior script and see the
    earlier-attempt trajectory; hot retries drop the anchor."""
    ctrl = hc.RunDynamicAnalysisController(
        SimpleNamespace(), LOGGER, generation_config=None,
        safety_settings=None, parse_fn=lambda r: ({}, None),
    )
    entries = [
        {"annealing_level": 0, "passed_fraction": 0.25,
         "issues_found": [{"location": "Gap_Width",
                           "problem": "edge pinned at window bound"}]},
        {"annealing_level": 1, "passed_fraction": 0.0,
         "issues_found": [{"location": "Gap_Width",
                           "problem": "over-strict gate, 0% coverage"}]},
    ]
    ctx = SimpleNamespace(retries=2, base_prompt="BASE", annealing_level=1,
                          last_passed_names=[], last_code="def analyze_feature(): pass",
                          attempt_entries=entries)
    out = ctrl.qc_refine(ctx, {"error_msg": "latest critique"})
    p = out["prompt"]
    # warm (retries=2 -> level 1): script anchored + history of EARLIER attempts
    assert "ADAPT IT, DO NOT REWRITE FROM SCRATCH" in p
    assert "def analyze_feature(): pass" in p
    assert "PRIOR ATTEMPTS ON THIS TASK" in p
    assert "edge pinned at window bound" in p
    assert "over-strict gate" not in p  # latest failure lives in the Critique, not history

    # hot (retries=3 -> level 2): anchor dropped, history retained
    ctx.retries = 3
    p_hot = ctrl.qc_refine(ctx, {"error_msg": "latest critique"})["prompt"]
    assert "ADAPT IT, DO NOT REWRITE FROM SCRATCH" not in p_hot
    assert "PRIOR ATTEMPTS ON THIS TASK" in p_hot
    assert "ABANDON" in p_hot


GOOD_SCRIPT = '''
def analyze_feature(data, axis):
    m = data.mean(axis=2)
    return {"maps": {"Mean_Map": m}, "units": "a.u.", "description": "d"}
'''
BROKEN_SCRIPT = '''
def analyze_feature(data, axis):
    m = data.mean(axis=2
    return {"maps": {"Mean_Map": m}}
'''


def _exec_state(tmp_path):
    return {
        "hspy_data": np.random.rand(5, 5, 8) * 1e-12,
        "original_hspy_data": np.random.rand(5, 5, 8) * 1e-12,
        "system_info": dict(AXIS_OK),
        "energy_axis": np.linspace(1, -1, 8),
        "settings": {"output_dir": str(tmp_path)},
        "refinement_decision": {
            "requires_custom_code": True,
            "targets": [{"description": "mean map", "type": "custom_code",
                         "required_outputs": ["Mean_Map"]}],
        },
        "iteration_title": "T",
        "analysis_objective": "obj",
    }


def test_exec_error_repaired_without_ladder_spend(tmp_path, monkeypatch):
    """A syntax error is fixed by a mechanical correction inside the SAME
    ladder attempt: the correction prompt carries the traceback, the ladder
    never advances, and the record notes the repair."""
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    prompts = []

    class _Model:
        def generate_content(self, contents, **kw):
            prompts.append(contents if isinstance(contents, str) else str(contents))
            script = BROKEN_SCRIPT if len(prompts) == 1 else GOOD_SCRIPT
            return json.dumps({"code": script})

    ctrl = hc.RunDynamicAnalysisController(
        _Model(), LOGGER, generation_config=None, safety_settings=None,
        parse_fn=lambda r: ({}, None),
    )
    ctrl._review_required_output = lambda *a, **k: (True, "")
    ctrl._check_result_visually = lambda *a, **k: (True, "")

    state = ctrl.execute(_exec_state(tmp_path))
    names = [m["name"] for m in state.get("custom_analysis_metadata_list") or []]
    assert names == ["Mean_Map"]                       # task succeeded
    assert len(prompts) == 2                           # initial + 1 repair
    assert "MECHANICAL CORRECTION" in prompts[1]
    # the broken code reaches the repair prompt (via the raw-response head,
    # since the compile-checking parser rejects syntax errors at parse time)
    assert "m = data.mean(axis=2" in prompts[1]
    assert "PREVIOUS ATTEMPT FAILED" not in prompts[1]  # NOT a ladder retry
    rec = (state.get("dynamic_analysis_records") or [])[0]
    entries = rec["quality_history"]["verification_iterations"]
    assert entries[-1].get("exec_corrections") == 1    # honest record


def test_exec_corrections_exhausted_becomes_ladder_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    calls = []

    class _Model:
        def generate_content(self, contents, **kw):
            calls.append(contents if isinstance(contents, str) else str(contents))
            return json.dumps({"code": BROKEN_SCRIPT})

    ctrl = hc.RunDynamicAnalysisController(
        _Model(), LOGGER, generation_config=None, safety_settings=None,
        parse_fn=lambda r: ({}, None),
    )
    state = _exec_state(tmp_path)
    state["max_verification_iterations"] = 0           # single ladder attempt
    state = ctrl.execute(state)
    assert state.get("dynamic_analysis_failed") is True
    # 1 initial + (MAX_EXEC_ATTEMPTS - 1) repairs, then the ladder attempt dies
    assert len(calls) == hc.RunDynamicAnalysisController.MAX_EXEC_ATTEMPTS
    assert all("MECHANICAL CORRECTION" in c for c in calls[1:])


def test_retry_banner_replaces_misplaced_verification_line(caplog):
    """The engine's "Verification k/N" banner is suppressed for hyperspectral
    (verification runs inside the attempt) and replaced by an honest
    "Preparing attempt k/N" line at the retry-preparation site."""
    ctrl = hc.RunDynamicAnalysisController(
        SimpleNamespace(), logging.getLogger("test.banner"),
        generation_config=None, safety_settings=None,
        parse_fn=lambda r: ({}, None),
    )
    assert ctrl.qc_iteration_banner(SimpleNamespace(), 1, 4) is None

    ctx = SimpleNamespace(retries=1, base_prompt="BASE", annealing_level=0,
                          last_passed_names=["Good_Map"])
    with caplog.at_level(logging.INFO, logger="test.banner"):
        out = ctrl.qc_refine(ctx, {"error_msg": "critique text"})
    text = "\n".join(r.message for r in caplog.records)
    assert "Preparing attempt 2/5" in text
    assert "regenerating with critique feedback" in text
    assert out["prompt"].startswith("BASE")
    assert "PASSED review in the failed attempt: ['Good_Map']" in out["prompt"]


# ---------------------------------------------------------------------------
# Fit examples: validation, rendering, reviewer + commit integration
# ---------------------------------------------------------------------------

def test_validate_fit_examples():
    good = {"fit_examples": [
        {"pixel": [2, 3], "fitted": np.zeros(8), "label": "map max"},
        {"pixel": [0, 0]},                              # no fitted: kept
        {"pixel": [99, 0], "fitted": np.zeros(8)},      # out of bounds: dropped
        {"pixel": [1, 1], "fitted": np.zeros(5)},       # wrong length: fitted->None
        "junk",                                         # malformed: dropped
    ]}
    out = hc._validate_fit_examples(good, 5, 5, 8)
    assert [ex["pixel"] for ex in out] == [(2, 3), (0, 0), (1, 1)]
    assert out[0]["fitted"] is not None
    assert out[1]["fitted"] is None and out[2]["fitted"] is None
    # cap
    many = {"fit_examples": [{"pixel": [0, 0]}] * 30}
    assert len(hc._validate_fit_examples(many, 5, 5, 8)) == hc._MAX_FIT_EXAMPLES
    assert hc._validate_fit_examples({}, 5, 5, 8) == []


def test_fit_examples_panel_renders():
    from scilink.skills.hyperspectral.eels.eels import create_fit_examples_panel
    data = RNG.rand(5, 5, 16) * 1e-12
    axis = np.linspace(1, -1, 16)
    examples = [
        {"pixel": (0, 0), "fitted": data[0, 0] * 1.05, "label": "best R2"},
        {"pixel": (4, 4), "fitted": None, "label": "map min"},
    ]
    maps = {"Gap_Width": RNG.rand(5, 5)}
    out = create_fit_examples_panel(data, axis, "Bias (V)", examples, maps, LOGGER)
    assert isinstance(out, bytes) and len(out) > 1000
    assert create_fit_examples_panel(data, axis, "Bias (V)", [], maps, LOGGER) is None


def test_fit_examples_reach_reviewer_and_committed_images(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    SCRIPT = '''
def analyze_feature(data, axis):
    m = data.mean(axis=2)
    return {"maps": {"Mean_Map": m}, "units": "a.u.", "description": "d",
            "fit_examples": [
                {"pixel": [0, 0], "fitted": data[0, 0, :], "label": "corner"},
                {"pixel": [2, 2], "fitted": data[2, 2, :], "label": "center"},
            ]}
'''

    class _Model:
        def generate_content(self, contents, **kw):
            return json.dumps({"code": SCRIPT})

    ctrl = hc.RunDynamicAnalysisController(
        _Model(), LOGGER, generation_config=None, safety_settings=None,
        parse_fn=lambda r: ({}, None),
    )
    seen_panels = []

    def fake_review(*a, **kw):
        seen_panels.append(kw.get("fit_panel_bytes"))
        return True, ""

    ctrl._review_required_output = fake_review
    ctrl._check_result_visually = lambda *a, **k: (True, "")

    state = {
        "hspy_data": np.random.rand(5, 5, 8) * 1e-12,
        "original_hspy_data": np.random.rand(5, 5, 8) * 1e-12,
        "system_info": dict(AXIS_OK),
        "energy_axis": np.linspace(1, -1, 8),
        "settings": {"output_dir": str(tmp_path)},
        "refinement_decision": {
            "requires_custom_code": True,
            "targets": [{"description": "mean map", "type": "custom_code",
                         "required_outputs": ["Mean_Map"]}],
        },
        "iteration_title": "T",
        "analysis_objective": "obj",
    }
    state = ctrl.execute(state)
    # reviewer received the rendered panel
    assert seen_panels and isinstance(seen_panels[0], bytes)
    # the panel rode the commit alongside the dashboards
    labels = [im["label"] for im in state.get("analysis_images", [])]
    assert any(l.startswith("Fit Examples") for l in labels)


def test_robust_hist_window_excludes_and_counts_outliers():
    from scilink.skills.hyperspectral.eels.eels import _robust_hist_window
    vals = np.concatenate([RNG.randn(28000),           # real distribution
                           [-25000.0, 40000.0, 50000.0]])  # spike pixels
    lo, hi, n_below, n_above = _robust_hist_window(vals)
    assert -4 < lo < -2 and 2 < hi < 4      # window hugs the real distribution
    assert n_below >= 1 and n_above >= 2
    # no-outlier data: window covers essentially everything
    clean = RNG.randn(10000)
    lo, hi, nb, na = _robust_hist_window(clean)
    assert (nb + na) / clean.size <= 0.011  # only the percentile tails


def test_robust_hist_window_constant_data():
    from scilink.skills.hyperspectral.eels.eels import _robust_hist_window
    lo, hi, nb, na = _robust_hist_window(np.full(100, 7.0))
    assert lo == hi == 7.0 and nb == 0 and na == 0


def test_dashboard_renders_with_extreme_outliers():
    from scilink.skills.hyperspectral.eels.eels import create_feature_dashboard
    m = RNG.randn(64, 64)
    m[0, 0], m[1, 1], m[2, 2] = -25000.0, 40000.0, 50000.0
    out = create_feature_dashboard(m, "Gap_Depth_Residual_DOS", "dimensionless")
    assert isinstance(out, bytes) and len(out) > 1000


def test_codegen_contract_documents_fit_examples():
    import inspect
    src = inspect.getsource(hc)
    block = src[src.index("### REQUIRED RETURN FORMAT"):]
    assert '"fit_examples"' in block[:4000]
    assert "representative pixels" in block[:4000]


def test_codegen_contract_guards_divisions():
    # Ratio-map division blowups (a few unbounded pixels wrecking the map's
    # statistics) are addressed at the cause: the contract demands
    # scale-RELATIVE denominator masking, never a fixed absolute epsilon —
    # the same absolute-threshold trap this branch fixes elsewhere.
    import inspect
    src = inspect.getsource(hc)
    block = src[src.index("### 3. CODING CONSTRAINTS"):]
    assert "Guard divisions" in block[:2500]
    assert "RELATIVE to its own scale" in block[:2500]
    assert "never a fixed absolute epsilon" in block[:2500]


# ---------------------------------------------------------------------------
# D. Flux table renders any native scale
# ---------------------------------------------------------------------------

def _tiny_cube(scale):
    x = np.linspace(-1, 1, 32)
    base = np.abs(np.sin(np.pi * x)) + 0.1
    return (base[None, None, :] * (1 + 0.05 * RNG.rand(8, 8, 1))) * scale


def test_flux_table_nonzero_on_amperes_scale():
    cube = _tiny_cube(1e-12)
    table = _render_band_flux_table(cube, np.linspace(1, -1, 32), "V")
    assert "MEASURED FLUX BY BAND" in table
    # every band of the old ".1f" rendering read exactly "0.0"
    assert "e-1" in table  # scientific notation present
    body = [l for l in table.splitlines() if "-" in l and "band" not in l]
    assert not all("0.0 " in l for l in body)


def test_flux_table_counts_scale_still_readable():
    cube = _tiny_cube(200.0)
    table = _render_band_flux_table(cube, np.linspace(0, 700, 32), "eV")
    assert "MEASURED FLUX BY BAND" in table
    # counts-scale magnitudes render as plain numbers of the right order
    assert any(tok.strip().startswith(("1", "2", "3"))
               for l in table.splitlines() for tok in l.split()[-1:]
               if "band" not in l and "OVERRIDE" not in l)


def test_flux_table_includes_aligned_auxiliary():
    cube = _tiny_cube(1e-12)
    aux = {"I0 reference": np.full(32, 5e-11)}
    table = _render_band_flux_table(cube, np.linspace(1, -1, 32), "V", aux=aux)
    assert "I0 reference" in table
    assert "5e-11" in table


# ---------------------------------------------------------------------------
# A2. Tiny-native-scale images: display renders with contrast, operands stay raw
# ---------------------------------------------------------------------------

def _stm_like_topo():
    """STM z-height in meters: offset ~-1.6e-7, corrugation ~1e-9 — the shape
    that collapsed to a constant-zero uint8 under the old 1e-6 epsilon."""
    base = np.sin(np.linspace(0, 6 * np.pi, 64))[:, None] * np.ones((1, 64))
    return (-1.58e-7 + 8e-10 * (base + 0.05 * RNG.rand(64, 64))).astype(
        np.float32)


def test_normalize_to_uint8_renders_tiny_range_with_contrast():
    from scilink.skills._shared.image_processor import _normalize_to_uint8
    out = _normalize_to_uint8(_stm_like_topo())
    assert out.dtype == np.uint8
    assert len(np.unique(out)) > 10  # old code: exactly 1 (all zeros)


def test_normalize_to_uint8_constant_image_is_zero_without_crash():
    from scilink.skills._shared.image_processor import _normalize_to_uint8
    out = _normalize_to_uint8(np.full((8, 8), -1.58e-7, dtype=np.float32))
    assert out.dtype == np.uint8 and (out == 0).all()


def test_load_image_raw_preserves_native_values(tmp_path):
    from scilink.skills._shared.image_processor import load_image
    topo = _stm_like_topo()
    p = tmp_path / "topo.npy"
    np.save(p, topo)
    raw = load_image(str(p), raw=True)
    assert np.allclose(raw, topo)
    disp = load_image(str(p))
    assert disp.dtype == np.uint8 and len(np.unique(disp)) > 10


def test_qc_rejection_logs_structured_problem_fix(caplog):
    critique = ("The peak is pinned to the edge of the search band; the "
                "histogram piles up at the clip values. CORRECTIVE DIRECTION: "
                "(1) restrict the search window; (2) require a true local "
                "maximum with prominence.")
    with caplog.at_level(logging.WARNING, logger="test.qc"):
        hc._log_qc_rejection(logging.getLogger("test.qc"),
                             "Coherence_Peak_Plus_Energy", critique,
                             "Combined review")
    text = "\n".join(r.message for r in caplog.records)
    assert "❌ Combined review rejected [Coherence_Peak_Plus_Energy]" in text
    assert "Problem:" in text and "Fix:" in text
    assert "restrict the search window" in text
    # wrapped: no single console line carries the whole critique
    assert all(len(r.message) < 100 for r in caplog.records)


def test_qc_rejection_without_marker_is_problem_only(caplog):
    with caplog.at_level(logging.WARNING, logger="test.qc2"):
        hc._log_qc_rejection(logging.getLogger("test.qc2"), "M",
                             "Map is pure noise with no structure.", "Visual QC")
    text = "\n".join(r.message for r in caplog.records)
    assert "Problem:" in text and "Fix:" not in text


def test_qc_rejection_logged_once_not_three_times():
    # The old flow triple-printed each critique: a combined-review warning,
    # the shared visual-QC warning with the same text, then the attempt
    # failure re-dumping every critique. Source-assert the dedup: rejections
    # render only through the structured helper, and the attempt-failure
    # echo is a one-line digest for QC failures.
    import inspect
    src = inspect.getsource(hc.RunDynamicAnalysisController._run_attempt)
    assert "Combined review rejected" not in src        # inner duplicate gone
    assert src.count("_log_qc_rejection") == 1          # single render site
    assert "critiques above; full text passed to the retry" in src


def test_auxiliary_image_operand_carries_raw_values(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent,
    )
    agent = HyperspectralAnalysisAgent(
        api_key="dummy", base_url="http://localhost:1",
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
    )
    topo = _stm_like_topo()
    p = tmp_path / "topo.npy"
    np.save(p, topo)
    state = agent._load_auxiliary_items(str(p), "co-registered topography")
    item = state["auxiliary_items"][0]
    arr = item["array"]
    # Pre-fix: arr was the uint8 display rendering — here all-zero, std 0,
    # silently NaN-ing any correlation computed against it.
    assert arr.std() > 0
    assert np.allclose(arr, topo.astype(float))
    assert item["plot_bytes"]  # display path still works
    assert "raw value range" in (item["summary"] or "")
