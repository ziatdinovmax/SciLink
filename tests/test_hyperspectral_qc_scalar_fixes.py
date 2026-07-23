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
