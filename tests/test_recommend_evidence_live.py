"""Live tests (Bedrock) for the recommend_measurements evidence contract
(#383): machine-readable `target` adoption and `measurement_history`
adherence, across curve (EELS + Raman) and image modalities.

Run manually:
    AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION_NAME=us-east-1 UNSAFE_EXECUTION_OK=true \
        python -m pytest tests/test_recommend_evidence_live.py -s -q

Hard assertions cover the contract guarantees (success, target dict-or-None,
format adoption). History adherence is printed for manual grading — LLM
behavior, not a schema guarantee.
"""

import json
import os
from pathlib import Path

import pytest

MODEL = "bedrock/us.anthropic.claude-opus-4-8"
BENCH = Path.home() / "Code/benchmarking_for_paper2"
RUNS = Path(__file__).parent / "_rec_contract_live_runs"

pytestmark = pytest.mark.skipif(
    not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"),
    reason="live test: needs Bedrock credentials")


def _contract_checks(out, label):
    assert out.get("status") == "success", f"{label}: {out.get('error')}"
    recs = out["measurement_recommendations"]
    assert recs, f"{label}: no recommendations"
    for r in recs:
        assert r.get("target") is None or isinstance(r["target"], dict)
    with_target = [r for r in recs if isinstance(r.get("target"), dict)
                   and r["target"].get("setting")]
    assert with_target, f"{label}: no machine-readable targets adopted"
    print(f"\n=== {label}: {len(recs)} recs, "
          f"{len(with_target)} with targets ===")
    for r in recs:
        print(f"  p{r.get('priority')} target={json.dumps(r.get('target'))}")
        print(f"     {r.get('description', '')[:140]}")
    return recs


def _fit_and_recommend(agent_cls, data, system_info, history, label, **akw):
    outdir = RUNS / label
    agent = agent_cls(model_name=MODEL, api_key=None,
                      enable_human_feedback=False, output_dir=str(outdir), **akw)
    res = agent.analyze(str(data), system_info=system_info)
    # partial (e.g. one HS sub-task failed) still carries detailed_analysis
    # and is a legitimate recommend_measurements input
    assert res.get("status") in ("success", "partial") and \
        res.get("detailed_analysis"), f"{label}: analyze failed"
    if res.get("status") == "partial":
        # rec generator rejects status=="error" or an "error" key; a partial
        # result may carry per-task errors irrelevant to recommendations
        res = {k: v for k, v in res.items() if k != "error"}
        res["status"] = "success"
    # fresh agent for recommendations (no stored figures: #382 workaround)
    rec_agent = agent_cls(model_name=MODEL, api_key=None,
                          enable_human_feedback=False,
                          output_dir=str(outdir / "_rec"))
    base = rec_agent.recommend_measurements(analysis_result=res,
                                            system_info=system_info)
    recs_base = _contract_checks(base, f"{label} (no history)")
    hist = rec_agent.recommend_measurements(analysis_result=res,
                                            system_info=system_info,
                                            measurement_history=history)
    recs_hist = _contract_checks(hist, f"{label} (with history)")
    print(f"  history given: {history}")
    (outdir / "live_outputs.json").write_text(json.dumps(
        {"base": base, "hist": hist, "history": history}, indent=2,
        default=str))
    return recs_base, recs_hist


def test_eels_curve_live():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    d = BENCH / "EELS/EELS_staged/strong/EEL_8"
    meta = json.loads((d / "metadata.json").read_text())
    _fit_and_recommend(
        CurveFittingAgent, d / "data1.csv", meta,
        ["EELS core-loss 683-735 eV (Fe L2,3 white lines) — the analyzed "
         "spectrum", "EELS low-loss -30 to +150 eV (ZLP + plasmon)"],
        "eels_eel8", max_verification_iterations=1)


def test_raman_curve_live():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    d = BENCH / "RamanIR/Raman_staged"
    csv = d / "01_easy_Diamond_R150088.csv"
    side = json.loads(csv.with_suffix(".json").read_text())
    info = {k: v for k, v in side.items() if k not in ("ground_truth",
                                                       "descriptors")}
    _fit_and_recommend(
        CurveFittingAgent, csv, info,
        ["Raman, 532 nm excitation, 100-1500 cm-1 — the analyzed spectrum"],
        "raman_diamond", max_verification_iterations=1)


def test_hyperspectral_live():
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import (
        HyperspectralAnalysisAgent)
    d = Path.home() / "Code/SciLink/meta_session_20260722_194112/uploads"
    info = {
        "technique": "Scanning tunneling spectroscopy (grid STS)",
        "signal": "dI/dV (lock-in X, LIX 1-omega) datacube",
        "sample": "TaS2",
        "spatial": "168x168 px over 15x15 nm",
        "energy_range": {"start": -1.0, "end": 1.0, "units": "V"},
        "spectral_axis": "bias sweep -1.0 to +1.0 V, 151 points",
    }
    _fit_and_recommend(
        HyperspectralAnalysisAgent, d / "LIX_1_omega__A.npy", info,
        ["Grid STS dI/dV cube, 168x168 px over 15x15 nm, bias -1.0 to +1.0 V "
         "(151 pts) — the analyzed cube",
         "Constant-current topography of the same 15x15 nm region"],
        "sts_tas2", max_verification_iterations=1)


def test_image_live():
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    d = BENCH / "TEM_staged/01_easy_AuNP"
    img = next(p for p in sorted(d.iterdir())
               if p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg"))
    meta_f = d / "metadata.json"
    info = json.loads(meta_f.read_text()) if meta_f.exists() else None
    _fit_and_recommend(
        ImageAnalysisAgent, img, info,
        ["HAADF-STEM overview image at the current field of view — the "
         "analyzed image"],
        "image_aunp")
