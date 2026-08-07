"""script_edits — surgical single-knob follow-ups on the #172 reuse path.

Phase A of analysis_script_surgical_refinement_plan.md: "same analysis,
but with one knob changed" re-ran full codegen and produced a script
differing in a dozen incidental ways, breaking one-variable-at-a-time
comparability. script_edits applies exact old/new pairs to the prior
run's saved script BEFORE the verbatim reuse run — the rerun differs in
exactly the knob, and the sandbox run remains the verification
(execute-don't-gate semantics unchanged from #172).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    _apply_reuse_script_edits)

SCRIPT = ("import numpy as np\n"
          "THRESH = 0.02\n"
          "WINDOW = (1.0, 9.0)\n"
          "print('CUSTOM_SCRIPT_SUCCESS')\n")


def make_prior_run(tmp_path):
    run = tmp_path / "prior_run"
    (run / "scripts").mkdir(parents=True)
    (run / "scripts" / "fitting_script.py").write_text(SCRIPT)
    (run / "series_fit_results.json").write_text(json.dumps({
        "results": [{"model_type": "pseudo_voigt", "success": True}],
        "total_spectra": 1, "successful": 1,
    }))
    return run


# ---------------------------------------------- controller helper


def test_edits_applied_and_source_labeled(tmp_path):
    state = {"script_edits": [
        {"old_text": "THRESH = 0.02", "new_text": "THRESH = 0.05"}]}
    text, src = _apply_reuse_script_edits(state, SCRIPT, "prior_run_001")
    assert "THRESH = 0.05" in text and "THRESH = 0.02" not in text
    assert text.replace("THRESH = 0.05", "THRESH = 0.02") == SCRIPT
    assert src == "prior_run_001 + 1 edit(s)"


def test_no_edits_or_no_script_is_a_no_op():
    assert _apply_reuse_script_edits({}, SCRIPT, "s") == (SCRIPT, "s")
    assert _apply_reuse_script_edits(
        {"script_edits": [{"old_text": "a", "new_text": "b"}]},
        None, None) == (None, None)


def test_non_applying_edit_refuses_loudly():
    """Post-validation failure means the prior run changed on disk — refuse
    rather than silently run the UNEDITED script."""
    state = {"script_edits": [{"old_text": "NOT THERE", "new_text": "x"}]}
    with pytest.raises(RuntimeError, match="no longer apply"):
        _apply_reuse_script_edits(state, SCRIPT, "s")


# ---------------------------------------------- analyze() entry validation


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    return CurveFittingAgent(
        api_key="offline-dummy", model_name="claude-opus-5",
        output_dir=str(tmp_path / "out"), enable_human_feedback=False,
        use_literature=False)


DATA = np.column_stack([np.linspace(0, 10, 50),
                        np.exp(-((np.linspace(0, 10, 50) - 5) ** 2))])


def test_edits_without_reuse_flag_refused(agent, tmp_path):
    run = make_prior_run(tmp_path)
    out = agent.analyze(
        DATA, system_info="curve",
        prior_analysis_paths=[str(run)],
        script_edits=[{"old_text": "THRESH = 0.02",
                       "new_text": "THRESH = 0.05"}])
    assert out["status"] == "error"
    assert "reuse_locked_script" in out["error"]["details"]


def test_edits_with_no_prior_script_refused(agent, tmp_path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    out = agent.analyze(
        DATA, system_info="curve",
        prior_analysis_paths=[str(empty)], reuse_locked_script=True,
        script_edits=[{"old_text": "a", "new_text": "b"}])
    assert out["status"] == "error"
    assert "prior script" in out["error"]["error"]


def test_non_matching_edits_refused_before_any_work(agent, tmp_path):
    run = make_prior_run(tmp_path)
    out = agent.analyze(
        DATA, system_info="curve",
        prior_analysis_paths=[str(run)], reuse_locked_script=True,
        script_edits=[
            {"old_text": "THRESH = 0.02", "new_text": "THRESH = 0.05"},
            {"old_text": "NOT IN SCRIPT", "new_text": "x"}])
    assert out["status"] == "error"
    assert out["error"]["error"] == "script_edits do not apply"
    assert out["error"]["failed_edit"] == 2


# ---------------------------------------------- orchestrator surface


def test_orchestrator_forwards_and_documents_script_edits():
    src = Path("scilink/agents/exp_agents/analysis_orchestrator_tools.py"
               ).read_text()
    assert '"script_edits"' in src
    i = src.index('"script_edits": {')
    desc = src[i:i + 1800]
    assert "reuse_locked_script" in desc          # names its preconditions
    assert "VERBATIM" in desc                     # copy-from-saved-script rule
    assert "signal, not gate" in desc             # execute-don't-gate semantics
    # forwarding is signature-gated so non-curve agents never see it
    j = src.index('analyze_kwargs["script_edits"]')
    assert "_inspect.signature" in src[j - 300:j]
