"""Golden pin for the #172 locked-script reuse fast path (engine qc_try_reuse).

Each test runs the happy scenario once to produce a real prior-run artifact
set (series_fit_results.json + scripts/), then a SECOND agent on the same
data with ``prior_analysis_paths`` + ``reuse_locked_script=True`` and pins
the reuse run's prompts, result (incl. ``reuse_validity``), and history.

Deliberate asymmetry pinned here: curve reuse makes NO verification call and
attaches NO quality_history; image reuse runs ONE soft verification pass and
DOES attach a quality_history. Regenerate with QC_GOLDEN_UPDATE=1.
"""

import pytest

from .fixtures import write_blob_image, write_gaussian_spectrum
from .harness import (
    ScriptedModel,
    check_golden,
    make_normalizer,
    normalize_obj,
)
from .scenarios import curve_rules, image_rules


def test_curve_reuse_golden(tmp_path):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    data_dir = tmp_path / "data"
    prior_dir = tmp_path / "prior"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")

    norm = make_normalizer({
        str(out_dir): "<OUTDIR>", str(prior_dir): "<PRIORDIR>",
        str(data_dir): "<DATADIR>",
    })

    # Run 1 — produce the prior artifacts.
    prior_agent = CurveFittingAgent(
        output_dir=str(prior_dir), enable_human_feedback=False,
        use_literature=False,
    )
    prior_agent.model = ScriptedModel(curve_rules("happy"), normalizer=norm)
    prior_result = prior_agent.analyze(str(spectrum))
    assert prior_result["status"] == "success"
    assert (prior_dir / "scripts" / "fitting_script.py").is_file()
    assert (prior_dir / "series_fit_results.json").is_file()

    # Run 2 — reuse the locked script verbatim.
    agent = CurveFittingAgent(
        output_dir=str(out_dir), enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel(curve_rules("happy"), normalizer=norm)
    agent.model = model
    result = agent.analyze(
        str(spectrum),
        prior_analysis_paths=[str(prior_dir)],
        reuse_locked_script=True,
    )

    assert result["status"] == "success", result.get("error")
    rv = result.get("reuse_validity")
    assert rv and rv["reused"] is True and rv["verdict"] == "good", rv
    # Curve asymmetry: reuse attaches NO quality_history and calls NO verifier.
    assert "quality_history" not in result
    assert not any(c["rule"] == "verify" for c in model.calls)

    payload = {
        "llm_calls": [c["rule"] for c in model.calls],
        "prompts": model.calls,
        "result": normalize_obj(result, norm),
        "reuse_validity": normalize_obj(rv, norm),
    }
    check_golden("curve_reuse", payload)


def test_image_reuse_golden(tmp_path):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    data_dir = tmp_path / "data"
    prior_dir = tmp_path / "prior"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    image = write_blob_image(data_dir / "blobs.npy")

    norm = make_normalizer({
        str(out_dir): "<OUTDIR>", str(prior_dir): "<PRIORDIR>",
        str(data_dir): "<DATADIR>",
    })

    prior_agent = ImageAnalysisAgent(
        output_dir=str(prior_dir), enable_human_feedback=False,
        use_literature=False,
    )
    prior_agent.model = ScriptedModel(image_rules("happy"), normalizer=norm)
    prior_result = prior_agent.analyze(str(image))
    assert prior_result["status"] == "success"

    agent = ImageAnalysisAgent(
        output_dir=str(out_dir), enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel(image_rules("happy"), normalizer=norm)
    agent.model = model
    result = agent.analyze(
        str(image),
        prior_analysis_paths=[str(prior_dir)],
        reuse_locked_script=True,
    )

    assert result["status"] == "success", result.get("error")
    rv = result.get("reuse_validity")
    assert rv and rv["reused"] is True, rv
    # Image asymmetry: reuse runs ONE soft verification pass and attaches
    # a quality_history (ledger item pinned in the plan §2.2).
    assert result.get("quality_history") is not None

    payload = {
        "llm_calls": [c["rule"] for c in model.calls],
        "prompts": model.calls,
        "result": normalize_obj(result, norm),
        "reuse_validity": normalize_obj(rv, norm),
    }
    check_golden("image_reuse", payload)
