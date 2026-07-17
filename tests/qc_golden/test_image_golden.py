"""Golden tests: ImageAnalysisAgent end-to-end under a scripted LLM.

Pins prompts / result / quality_history / saved scripts for the single-image
happy path and the verification-failure→annealing path.
Regenerate with QC_GOLDEN_UPDATE=1.
"""

import pytest

from .fixtures import write_blob_image
from .harness import (
    ScriptedModel,
    check_golden,
    collect_saved_scripts,
    make_normalizer,
    normalize_obj,
)
from .scenarios import image_rules


def _run_image(tmp_path, mode: str):
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    image = write_blob_image(data_dir / "blobs.npy")

    norm = make_normalizer({str(out_dir): "<OUTDIR>", str(data_dir): "<DATADIR>"})

    agent = ImageAnalysisAgent(
        output_dir=str(out_dir),
        enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel(image_rules(mode), normalizer=norm)
    agent.model = model

    result = agent.analyze(str(image))
    return agent, model, result, out_dir, norm


@pytest.mark.parametrize("mode", ["happy", "anneal"])
def test_image_golden(tmp_path, mode):
    agent, model, result, out_dir, norm = _run_image(tmp_path, mode)

    assert result["status"] == "success", result.get("error")
    assert result["quality_history"]["approved"] is True
    # Tier-2 decision must have been consulted and declined (default depth
    # is "auto"); a tier2_results key would mean the scenario ran tier 2.
    assert "tier2_results" not in result

    if mode == "anneal":
        iters = result["quality_history"]["verification_iterations"]
        assert len(iters) >= 2
        assert max(i.get("annealing_level", 0) for i in iters) > 0

    payload = {
        "llm_calls": [c["rule"] for c in model.calls],
        "prompts": model.calls,
        "result": normalize_obj(result, norm),
        "quality_history": normalize_obj(result.get("quality_history"), norm),
        "scripts": collect_saved_scripts(out_dir, norm),
    }
    check_golden(f"image_{mode}", payload)
