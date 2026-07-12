"""Golden pin for the realtime profile (#346 step 3).

The contract this pins: a curve frame analyzed under ``profile="realtime"``
with a reusable prior makes **ZERO LLM calls** — every pipeline stage around
the reuse fast path (skill suggestion, planning, plan validation, literature,
verification, synthesis) is skipped — and the result carries the provenance
stamp plus the fingerprint-drift channel. The model object would fail loudly
on any call (ScriptedModel with no rules), so a regression that reintroduces
an LLM call fails here immediately. Regenerate with QC_GOLDEN_UPDATE=1.
"""

import json

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


def _enable_bank(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_HOME", str(tmp_path / "scilink_home"))
    monkeypatch.setenv("SCILINK_SCRIPT_BANK", "1")


def test_realtime_cold_start_zero_llm(tmp_path, monkeypatch):
    """Verbatim cold start (#346 step 4): a realtime run with NO prior
    auditions bank candidates against the frame; the winner becomes the
    locked recipe — still zero LLM calls — with source=script_bank:<id>,
    cold_start surfaced, and the record's cross-session stats bumped once."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    from scilink.skills._shared import _script_bank as sb

    _enable_bank(tmp_path, monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")

    # Seed the bank via a normal thorough run (write hook does the banking).
    seed_agent = CurveFittingAgent(
        output_dir=str(tmp_path / "seed"), enable_human_feedback=False,
        use_literature=False,
    )
    seed_agent.model = ScriptedModel(curve_rules("happy"))
    seed_result = seed_agent.analyze(str(spectrum))
    assert seed_result["status"] == "success"
    assert seed_result.get("banked_scripts"), "seed run must bank its script"
    rid = seed_result["banked_scripts"][0]
    assert sb.get_record("curve_fitting", rid)["stats"]["n_successes"] == 1

    # Cold start — no prior, fail-loud model.
    agent = CurveFittingAgent(
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel([])
    agent.model = model
    result = agent.analyze(str(spectrum), profile="realtime")

    assert result["status"] == "success", result.get("error")
    assert model.calls == [], f"cold start made LLM calls: {model.calls}"
    assert result.get("profile") == "realtime"
    rv = result.get("reuse_validity") or {}
    assert rv.get("reused") is True and rv.get("verdict") == "good", rv
    assert rv.get("source") == f"script_bank:{rid}", rv
    assert rv.get("drift") == "none", rv  # same data as the record's origin
    cs = result.get("cold_start") or {}
    assert cs.get("id") == rid and cs.get("n_auditioned") == 1, cs
    # One cross-session success recorded for the winner; the realtime run
    # itself banked nothing.
    rec = sb.get_record("curve_fitting", rid)
    assert rec["stats"]["n_successes"] == 2
    assert len(sb.list_records("curve_fitting")) == 1


def test_realtime_cold_start_no_match_demotes_to_thorough(tmp_path, monkeypatch):
    """Empty bank → the cold start finds nothing and the run loudly falls
    back to a thorough anchor (LLM calls happen; no realtime provenance)."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    _enable_bank(tmp_path, monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")

    agent = CurveFittingAgent(
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel(curve_rules("happy"))
    agent.model = model
    result = agent.analyze(str(spectrum), profile="realtime")

    assert result["status"] == "success", result.get("error")
    assert result.get("profile") != "realtime"
    assert model.calls, "thorough fallback must run the normal LLM pipeline"
    assert "reuse_validity" not in result


def test_realtime_cold_start_broken_candidate_skipped(tmp_path, monkeypatch):
    """A higher-ranked candidate whose script fails the audition is skipped;
    the next candidate wins (n_auditioned=2)."""
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    from scilink.skills._shared import _script_bank as sb

    _enable_bank(tmp_path, monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")

    # Seed a WORKING record via a thorough run.
    seed_agent = CurveFittingAgent(
        output_dir=str(tmp_path / "seed"), enable_human_feedback=False,
        use_literature=False,
    )
    seed_agent.model = ScriptedModel(curve_rules("happy"))
    seed_result = seed_agent.analyze(str(spectrum))
    good_id = seed_result["banked_scripts"][0]
    good_rec = sb.get_record("curve_fitting", good_id)

    # Seed a BROKEN record with the same fingerprint and a usage bonus that
    # ranks it FIRST (find_exemplar sorts by score).
    sb.add_record("curve_fitting", {
        "working_script": "raise RuntimeError('rotted script')",
        "data_fingerprint": good_rec["data_fingerprint"],
        "measurement_context": good_rec.get("measurement_context") or {},
        "technique_signals": good_rec.get("technique_signals") or {},
        "outcome": {}, "provenance": {"session": "old"},
    })
    broken_id = [r["id"] for r in sb.list_records("curve_fitting")
                 if r["id"] != good_id][0]
    for s in ("s2", "s3", "s4", "s5"):
        sb.record_success("curve_fitting", broken_id, session=s)

    agent = CurveFittingAgent(
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
        use_literature=False,
    )
    agent.model = ScriptedModel([])
    result = agent.analyze(str(spectrum), profile="realtime")

    assert result["status"] == "success", result.get("error")
    cs = result.get("cold_start") or {}
    assert cs.get("id") == good_id, cs
    assert cs.get("n_auditioned") == 2, cs
    assert (result.get("reuse_validity") or {}).get("source") == \
        f"script_bank:{good_id}"


def test_degenerate_data_check_unit():
    """The pre-flight gate is conservative: glitch shapes fail, real and
    partially-corrupted spectra pass (the correction path salvages those)."""
    from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
        _degenerate_data_check,
    )
    x = np.linspace(0, 100, 500)
    y = 5 * np.exp(-(x - 50) ** 2 / 9) + np.random.RandomState(0).normal(0, 0.05, 500)

    assert _degenerate_data_check(np.vstack([x, y])) is None  # real spectrum
    y_nan = y.copy()
    y_nan[::3] = np.nan  # a third corrupted — recoverable, must pass
    assert _degenerate_data_check(np.vstack([x, y_nan])) is None

    assert "dynamic range" in _degenerate_data_check(
        np.vstack([x, np.zeros_like(x)]))          # all-zero
    assert "finite" in _degenerate_data_check(
        np.vstack([x[:10], y[:10]]))                # runt
    assert "finite" in _degenerate_data_check(
        np.vstack([x, np.full_like(x, np.nan)]))    # all-NaN
    assert "dynamic range" in _degenerate_data_check(
        np.vstack([x, np.full_like(x, 7.3)]))       # constant


def test_realtime_glitch_frame_gated_zero_llm(tmp_path):
    """An all-zeros frame under realtime fails in the pre-flight gate:
    zero LLM calls, no script execution attempts, honest error."""
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

    x = np.linspace(0, 100, 500)
    glitch = data_dir / "glitch_zeros.csv"
    np.savetxt(glitch, np.column_stack([x, np.zeros_like(x)]),
               delimiter=",", header="x,y", comments="")

    agent = CurveFittingAgent(
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
        use_literature=False,
    )
    model = ScriptedModel([])  # fail-loud: ANY LLM call raises
    agent.model = model
    result = agent.analyze(
        str(glitch),
        profile="realtime",
        prior_analysis_paths=[str(prior_dir)],
        reuse_locked_script=True,
    )
    assert model.calls == [], f"gated frame made LLM calls: {model.calls}"
    sr = json.loads(
        (tmp_path / "out" / "series_fit_results.json").read_text())
    frame = sr["results"][0]
    assert frame["success"] is False
    assert "Pre-flight degenerate-data gate" in frame["error"]
    assert frame.get("script") is None  # nothing was executed or corrected


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
