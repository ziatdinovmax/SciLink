"""#753 / #758 review: the curve agent's single-run certification reference,
through the real writer — a thorough run's ``analysis_results.json`` carries
it, a live frame (the realtime profile replaying that run) writes none."""

import json

from .fixtures import write_gaussian_spectrum
from .harness import ScriptedModel, make_normalizer
from .scenarios import curve_rules


def test_a_thorough_run_stamps_its_reference_and_a_live_frame_does_not(tmp_path):
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    data_dir, prior_dir, out_dir = tmp_path / "data", tmp_path / "prior", tmp_path / "out"
    data_dir.mkdir()
    spectrum = write_gaussian_spectrum(data_dir / "gaussian_peak.csv")
    norm = make_normalizer({str(out_dir): "<OUTDIR>", str(prior_dir): "<PRIORDIR>", str(data_dir): "<DATADIR>"})

    prior = CurveFittingAgent(output_dir=str(prior_dir), enable_human_feedback=False, use_literature=False)
    prior.model = ScriptedModel(curve_rules("happy"), normalizer=norm)
    assert prior.analyze(str(spectrum))["status"] == "success"
    saved = json.loads((prior_dir / "analysis_results.json").read_text())
    ref = saved.get("certification_reference") or {}
    assert ref.get("kind") == "curve" and ref.get("drift_state") and ref.get("identity", {}).get("n_units") == 1

    frame = CurveFittingAgent(output_dir=str(out_dir), enable_human_feedback=False, use_literature=False)
    frame.model = ScriptedModel([], normalizer=norm)                      # no model call allowed
    res = frame.analyze(str(spectrum), profile="realtime", prior_analysis_paths=[str(prior_dir)],
                        reuse_locked_script=True)
    assert res["status"] == "success"
    assert "certification_reference" not in json.loads((out_dir / "analysis_results.json").read_text())
