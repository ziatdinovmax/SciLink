"""Golden tests: CurveFittingAgent end-to-end under a scripted LLM.

Pins prompts / result / quality_history / saved scripts for the single-
spectrum happy path and the verification-failure→annealing path.
Regenerate with QC_GOLDEN_UPDATE=1.
"""

import pytest

from .fixtures import write_gaussian_spectrum
from .harness import (
    ScriptedModel,
    check_golden,
    collect_saved_scripts,
    make_normalizer,
    normalize_obj,
)
from .scenarios import curve_rules


def _run_curve(tmp_path, mode: str):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")

    norm = make_normalizer({str(out_dir): "<OUTDIR>", str(data_dir): "<DATADIR>"})

    agent = CurveFittingAgent(
        output_dir=str(out_dir),
        enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel(curve_rules(mode), normalizer=norm)
    agent.model = model

    result = agent.analyze(str(spectrum))
    return agent, model, result, out_dir, norm


@pytest.mark.parametrize("mode", ["happy", "anneal"])
def test_curve_golden(tmp_path, mode):
    agent, model, result, out_dir, norm = _run_curve(tmp_path, mode)

    assert result["status"] == "success", result.get("error")
    assert result["quality_history"]["approved"] is True

    if mode == "anneal":
        # The point of the scenario: verification rejected at least once and
        # the annealing ladder actually moved before approval.
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
    check_golden(f"curve_{mode}", payload)
