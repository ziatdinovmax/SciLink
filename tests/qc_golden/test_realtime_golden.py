"""Golden pin for the realtime profile (#346 step 3).

The contract this pins: a curve frame analyzed under ``profile="realtime"``
with a reusable prior makes **ZERO LLM calls** — every pipeline stage around
the reuse fast path (skill suggestion, planning, plan validation, literature,
verification, synthesis) is skipped — and the result carries the provenance
stamp plus the fingerprint-drift channel. The model object would fail loudly
on any call (ScriptedModel with no rules), so a regression that reintroduces
an LLM call fails here immediately. Regenerate with QC_GOLDEN_UPDATE=1.
"""

import numpy as np

from .fixtures import write_gaussian_spectrum
from .harness import (
    ScriptedModel,
    check_golden,
    make_normalizer,
    normalize_obj,
)
from .scenarios import curve_rules


def test_curve_realtime_zero_llm_golden(tmp_path):
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

    # Anchor — one thorough run producing the locked recipe.
    prior_agent = CurveFittingAgent(
        output_dir=str(prior_dir), enable_human_feedback=False,
        use_literature=False,
    )
    prior_agent.model = ScriptedModel(curve_rules("happy"), normalizer=norm)
    prior_result = prior_agent.analyze(str(spectrum))
    assert prior_result["status"] == "success"

    # Frame — realtime: the model has NO rules, so ANY LLM call fails loudly.
    agent = CurveFittingAgent(
        output_dir=str(out_dir), enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel([], normalizer=norm)
    agent.model = model
    result = agent.analyze(
        str(spectrum),
        profile="realtime",
        prior_analysis_paths=[str(prior_dir)],
        reuse_locked_script=True,
    )

    assert result["status"] == "success", result.get("error")
    assert model.calls == [], f"realtime frame made LLM calls: {model.calls}"
    assert result.get("profile") == "realtime"
    rv = result.get("reuse_validity")
    assert rv and rv["reused"] is True and rv["verdict"] == "good", rv
    # Drift channel present. The anchor run had the bank disabled (no
    # persisted fingerprint), so the lazy fallback re-reads the anchor's
    # data file — same file here, so similarity is exactly 1.0.
    assert rv.get("drift") == "none", rv
    assert rv.get("fingerprint_similarity") == 1.0, rv
    # Provenance reaches quality_history for the post-hoc sweep.
    assert (result.get("quality_history") or {}).get(
        "produced_under_profile") == "realtime"

    payload = {
        "llm_calls": [c["rule"] for c in model.calls],  # pinned empty
        "result": normalize_obj(result, norm),
        "reuse_validity": normalize_obj(rv, norm),
    }
    check_golden("curve_realtime_reuse", payload)


def test_realtime_requires_locked_recipe(tmp_path):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")
    agent = CurveFittingAgent(
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
        use_literature=False,
    )
    agent.model = ScriptedModel([])
    import pytest
    with pytest.raises(ValueError, match="realtime"):
        agent.analyze(str(spectrum), profile="realtime")


def test_realtime_drift_flags_changed_data(tmp_path):
    """A realtime frame whose data no longer resembles the anchor must flag
    drift='suspected' even when the reused script still fits (the R² gate
    alone is blind to this — live-proven on the dehydration series)."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    data_dir = tmp_path / "data"
    prior_dir = tmp_path / "prior"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")

    prior_agent = CurveFittingAgent(
        output_dir=str(prior_dir), enable_human_feedback=False,
        use_literature=False,
    )
    prior_agent.model = ScriptedModel(curve_rules("happy"))
    assert prior_agent.analyze(str(spectrum))["status"] == "success"

    # New phase: different peak set in the same window (three peaks vs one).
    x = np.linspace(0, 100, 500)
    y = (3.0 * np.exp(-(x - 20) ** 2 / 8)
         + 2.0 * np.exp(-(x - 65) ** 2 / 12)
         + 1.5 * np.exp(-(x - 85) ** 2 / 6)
         + np.random.RandomState(0).normal(0, 0.02, x.size))
    new_frame = data_dir / "new_phase.csv"
    np.savetxt(new_frame, np.column_stack([x, y]), delimiter=",",
               header="x,y", comments="")

    agent = CurveFittingAgent(
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
        use_literature=False,
    )
    agent.model = ScriptedModel([])
    result = agent.analyze(
        str(new_frame),
        profile="realtime",
        prior_analysis_paths=[str(prior_dir)],
        reuse_locked_script=True,
    )
    assert result["status"] == "success", result.get("error")
    rv = result.get("reuse_validity") or {}
    assert rv.get("drift") == "suspected", rv
    assert rv.get("fingerprint_similarity") < 0.92, rv
