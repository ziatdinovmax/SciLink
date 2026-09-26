"""Bank minimal-edit adaptation (Phase B of the surgical-refinement plan).

Today's "adapt" regenerates the whole script with the banked one as a
prompt exemplar — eroding exactly the provenance that made it worth
banking. The edit-adapt mode asks the LLM for an edit LIST against the
banked script, applies it mechanically, and lets the UNCHANGED
verification loop judge the result. The ladder can only add: every
failure falls through (with its reason logged) to today's exemplar path.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    UnifiedSeriesProcessingController as C)

BANKED = ("import numpy as np\n"
          "CENTER_GUESS = 5.0\n"
          "print('FIT_RESULTS_JSON: {}')\n"
          "print('CUSTOM_SCRIPT_SUCCESS')\n")


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts = []

    def generate_content(self, prompt, generation_config=None):
        self.calls += 1
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.replies.pop(0))


def make_self(model_replies, fit_result=None):
    captured = {}

    def _fit_single_spectrum(**kw):
        captured.update(kw)
        return fit_result if fit_result is not None else {
            "success": True, "fit_quality": {"r_squared": 0.99}}

    s = SimpleNamespace(
        logger=SimpleNamespace(info=lambda *a, **k: None,
                               warning=lambda *a, **k: None),
        model=FakeModel(model_replies),
        output_dir="/tmp/sess",
        generation_config=None,
        _fit_single_spectrum=_fit_single_spectrum,
        _process_single_image=lambda **kw: (
            captured.update(kw) or (fit_result if fit_result is not None
                                    else {"success": True})),
    )
    return s, captured


def make_ctx(score=0.8, script=BANKED):
    return SimpleNamespace(
        state={"_bank_exemplar": {
            "score": score,
            "record": {"id": "rec_001", "working_script": script}},
            "locked_fitting_config": {"physical_model": "gaussian"},
            "system_info": "test curve"},
        data=None, data_path="d.csv", item_name="s", item_idx=0,
        initial_label=None)


GOOD_REPLY = json.dumps({
    "edits": [{"old_text": "CENTER_GUESS = 5.0",
               "new_text": "CENTER_GUESS = 7.2"}],
    "rationale": "peak sits at 7.2 in this dataset"})


def test_happy_path_applies_edits_and_records_provenance():
    fake, captured = make_self([GOOD_REPLY])
    ctx = make_ctx()
    res = C._try_bank_edit_adapt(fake, ctx)
    assert res is not None and res["success"]
    assert "CENTER_GUESS = 7.2" in captured["base_script"]
    assert captured["base_script"].replace("7.2", "5.0") == BANKED.strip()
    bea = res["bank_edit_adapt"]
    assert bea["id"] == "rec_001" and bea["n_edits"] == 1
    assert "edit-adapt of rec_001" in ctx.initial_label


def test_empty_edit_list_means_verbatim():
    fake, captured = make_self([json.dumps({"edits": [], "rationale": "fits"})])
    res = C._try_bank_edit_adapt(fake, make_ctx())
    assert res is not None
    assert captured["base_script"] == BANKED.strip()
    assert res["bank_edit_adapt"]["n_edits"] == 0


def test_weak_match_skips_without_an_llm_call():
    fake, _ = make_self([GOOD_REPLY])
    assert C._try_bank_edit_adapt(fake, make_ctx(score=0.40)) is None
    assert fake.model.calls == 0


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("SCILINK_BANK_EDIT_ADAPT", "0")
    fake, _ = make_self([GOOD_REPLY])
    assert C._try_bank_edit_adapt(fake, make_ctx()) is None
    assert fake.model.calls == 0


def test_score_override(monkeypatch):
    monkeypatch.setenv("SCILINK_BANK_EDIT_ADAPT_SCORE", "0.95")
    fake, _ = make_self([GOOD_REPLY])
    assert C._try_bank_edit_adapt(fake, make_ctx(score=0.8)) is None


def test_garbage_json_falls_through_after_one_retry():
    fake, _ = make_self(["not json at all", "still not json"])
    assert C._try_bank_edit_adapt(fake, make_ctx()) is None
    assert fake.model.calls == 2          # one corrective retry, then out


def test_non_applying_edits_fall_through():
    reply = json.dumps({"edits": [{"old_text": "NOT PRESENT",
                                   "new_text": "x"}], "rationale": "r"})
    fake, captured = make_self([reply])
    assert C._try_bank_edit_adapt(fake, make_ctx()) is None
    assert "base_script" not in captured   # nothing executed


def test_execution_failure_falls_through():
    fake, _ = make_self([GOOD_REPLY], fit_result={"success": False})
    assert C._try_bank_edit_adapt(fake, make_ctx()) is None


def test_success_bump_only_with_surviving_provenance(monkeypatch):
    from scilink.skills._shared import _script_bank
    bumped = []
    monkeypatch.setattr(_script_bank, "record_success",
                        lambda d, rid, session=None, **kw: bumped.append(rid))
    fake, _ = make_self([])
    C._bump_bank_adapt_success(fake, {
        "success": True, "bank_edit_adapt": {"id": "rec_001"}})
    C._bump_bank_adapt_success(fake, {"success": True})   # refit: no prov
    C._bump_bank_adapt_success(fake, {
        "success": False, "bank_edit_adapt": {"id": "rec_002"}})
    # kept-but-flagged poor fit must NOT count as proven (live: R²=0.42
    # NMR adaptation executed fine and would have inflated proven-N)
    C._bump_bank_adapt_success(fake, {
        "success": True, "quality_warning": "R² below threshold",
        "bank_edit_adapt": {"id": "rec_003"}})
    assert bumped == ["rec_001"]


def test_template_contract():
    from scilink.agents.exp_agents.instruct import BANK_EDIT_ADAPT_INSTRUCTIONS
    filled = BANK_EDIT_ADAPT_INSTRUCTIONS.format(
        script_kind="curve-fitting", banked_script="s", locked_config="{}",
        data_context="d", output_contract="the FIT_RESULTS_JSON print")
    assert '"edits"' in filled and "ONLY a JSON object" in filled
    assert "EXACTLY ONCE" in filled
    assert "FIT_RESULTS_JSON" in filled    # contract text is caller-supplied


# ------------------------------------------- image twin (shared core)


def make_image_ctx(score=0.8):
    ctx = make_ctx(score=score)
    ctx.state["locked_analysis_config"] = {"processing_pipeline": "blobs"}
    ctx.data = SimpleNamespace(shape=(64, 64))
    return ctx


def test_image_wrapper_shares_the_core():
    from scilink.agents.exp_agents.controllers.image_analysis_controllers \
        import UnifiedImageProcessingController as IC
    fake, captured = make_self([GOOD_REPLY])
    ctx = make_image_ctx()
    res = IC._try_bank_edit_adapt(fake, ctx)
    assert res is not None and res["bank_edit_adapt"]["id"] == "rec_001"
    assert "CENTER_GUESS = 7.2" in captured["base_script"]
    assert captured["image_name"] == "s"       # image runner was used
    assert "edit-adapt of rec_001" in ctx.initial_label


def test_image_bump_uses_image_domain(monkeypatch):
    from scilink.agents.exp_agents.controllers.image_analysis_controllers \
        import UnifiedImageProcessingController as IC
    from scilink.skills._shared import _script_bank
    bumped = []
    monkeypatch.setattr(_script_bank, "record_success",
                        lambda d, rid, session=None, **kw: bumped.append((d, rid)))
    fake, _ = make_self([])
    IC._bump_bank_adapt_success(fake, {
        "success": True, "bank_edit_adapt": {"id": "rec_009"}})
    assert bumped == [("image_analysis", "rec_009")]


# --------------------------------------- hyperspectral twin (H1)


def make_hs_ctx(score=0.8, with_exemplar=True):
    return SimpleNamespace(
        state={},
        item_name="Ni K-edge jump map",
        required_outputs=["edge_jump_map"],
        session={"processing_note": "binned 2x",
                 "optimal_data": SimpleNamespace(shape=(32, 32, 200))},
        attempt_entries=[], retries=0,
        bank_exemplar=({"score": score,
                        "record": {"id": "hs_rec_1",
                                   "working_script": BANKED}}
                       if with_exemplar else None),
        initial_label=None, current_result=None, last_code=None,
        supplied_script=None)


def hs_fake(model_replies, task_success=True):
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers \
        import RunDynamicAnalysisController as HC
    captured = {}

    def _run_attempt(ctx):
        captured["supplied"] = ctx.supplied_script
        ctx.attempt_entries.append({"passed_fraction": 1.0})
        ctx.retries += 1
        return {"task_success": task_success}

    s = SimpleNamespace(
        logger=SimpleNamespace(info=lambda *a, **k: None,
                               warning=lambda *a, **k: None,
                               error=lambda *a, **k: None),
        model=FakeModel(model_replies), generation_config=None,
        _run_attempt=_run_attempt)
    return HC, s, captured


def test_hs_adapt_executes_edited_script_and_keeps_provenance():
    HC, fake, captured = hs_fake([GOOD_REPLY])
    ctx = make_hs_ctx()
    res = HC._try_bank_edit_adapt(fake, ctx)
    assert res is not None and res["success"] is True
    assert "CENTER_GUESS = 7.2" in captured["supplied"]
    assert res["_from_bank_adapt"] is True
    assert ctx.bank_edit_adapt["id"] == "hs_rec_1"
    assert "_bank_exemplar" not in ctx.state       # cleaned up
    assert len(ctx.attempt_entries) == 1           # counts as attempt 1


def test_hs_task_failure_is_budget_neutral():
    """A task-failed adaptation must roll back its attempt entry and
    retry tick — the escalation ladder starts fresh."""
    HC, fake, captured = hs_fake([GOOD_REPLY], task_success=False)
    ctx = make_hs_ctx()
    res = HC._try_bank_edit_adapt(fake, ctx)
    assert res is None
    assert ctx.attempt_entries == [] and ctx.retries == 0
    assert "_bank_exemplar" not in ctx.state


def test_hs_no_exemplar_no_llm_call():
    HC, fake, _ = hs_fake([GOOD_REPLY])
    assert HC._try_bank_edit_adapt(fake, make_hs_ctx(
        with_exemplar=False)) is None
    assert fake.model.calls == 0


def test_hs_record_provenance_gated_on_adapted_acceptance():
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers \
        import RunDynamicAnalysisController as HC
    fake = SimpleNamespace(
        logger=SimpleNamespace(warning=lambda *a, **k: None),
        SUCCESS_THRESHOLD=0.6)
    ctx = make_hs_ctx()
    ctx.attempt_entries = [{"passed_fraction": 1.0}]
    ctx.best_attempt = {"valid_count": 1}
    ctx.last_code = "code"
    ctx.bank_edit_adapt = {"id": "hs_rec_1", "n_edits": 1}

    ctx.current_result = {"_from_bank_adapt": True}
    rec = HC._build_target_record(fake, ctx, task_success=True)
    assert rec["bank_edit_adapt"]["id"] == "hs_rec_1"

    ctx.current_result = {}                        # ladder refit replaced it
    rec = HC._build_target_record(fake, ctx, task_success=True)
    assert "bank_edit_adapt" not in rec


def test_hs_supplied_script_mode_in_run_attempt_source():
    src = Path("scilink/agents/exp_agents/controllers/"
               "hyperspectral_controllers.py").read_text()
    i = src.index('getattr(ctx, "supplied_script", None)')
    assert "Executing supplied" in src[i:i + 600]


def test_single_shared_implementation():
    from pathlib import Path
    curve = Path("scilink/agents/exp_agents/controllers/"
                 "curve_fitting_controllers.py").read_text()
    image = Path("scilink/agents/exp_agents/controllers/"
                 "image_analysis_controllers.py").read_text()
    hs = Path("scilink/agents/exp_agents/controllers/"
              "hyperspectral_controllers.py").read_text()
    engine = Path("scilink/agents/exp_agents/_qc_engine.py").read_text()
    assert engine.count("def try_bank_edit_adapt") == 1
    for src in (curve, image, hs):
        assert "from .._qc_engine import try_bank_edit_adapt" in src
        assert "def try_bank_edit_adapt" not in src.replace(
            "from .._qc_engine import try_bank_edit_adapt", "")


# ── hardening (live audit 2026-09-26) ──────────────────────────────────
# Four out of four live adaptations of a column-0 script came back with a
# four-space indent and were refused by the exact applier; a damped
# oscillation anchored to a peak script and was "adapted" into a different
# model, then credited as a peak-fit success.

INDENTED_BANKED = ("import numpy as np\n"
                   "params = Parameters()\n"
                   "params.add('center', value=5.0, min=4.5, max=5.5)\n"
                   "def helper():\n"
                   "    return 1\n"
                   "print('FIT_RESULTS_JSON: {}')\n"
                   "print('CUSTOM_SCRIPT_SUCCESS')\n")


def test_leading_whitespace_drift_is_re_anchored():
    reply = json.dumps({"edits": [
        {"old_text": "    params.add('center', value=5.0, min=4.5, max=5.5)",
         "new_text": "    params.add('center', value=6.5, min=6.0, max=7.0)"}],
        "model_family_kept": True, "rationale": "shifted peak"})
    fake, captured = make_self([reply])
    ctx = make_ctx(script=INDENTED_BANKED)
    res = C._try_bank_edit_adapt(fake, ctx)
    assert res is not None and fake.model.calls == 1
    assert "params.add('center', value=6.5, min=6.0, max=7.0)\n" in captured["base_script"]
    assert "    params.add" not in captured["base_script"]
    assert ctx.bank_adapt_attempt["n_relaxed"] == 1 and res["bank_edit_adapt"]["n_edits"] == 1


def test_non_applying_edits_get_one_corrected_attempt():
    bad = json.dumps({"edits": [{"old_text": "CENTER_GUESS = 9.9", "new_text": "x"}],
                      "model_family_kept": True, "rationale": "r"})
    fake, captured = make_self([bad, GOOD_REPLY])
    ctx = make_ctx()
    res = C._try_bank_edit_adapt(fake, ctx)
    assert res is not None and fake.model.calls == 2
    assert "CENTER_GUESS = 7.2" in captured["base_script"]
    assert ctx.bank_adapt_attempt["retried"] is True
    # the corrected attempt carries the applier's message back to the model
    assert "did not apply" in fake.model.prompts[1] and "leading whitespace" in fake.model.prompts[1]


def test_second_bad_edit_list_falls_through_without_a_third_call():
    bad = json.dumps({"edits": [{"old_text": "NOPE", "new_text": "x"}],
                      "model_family_kept": True, "rationale": "r"})
    fake, captured = make_self([bad, bad])
    assert C._try_bank_edit_adapt(fake, make_ctx()) is None
    assert fake.model.calls == 2 and "base_script" not in captured


def test_adapter_declines_a_different_model_family_without_executing():
    reply = json.dumps({"edits": [], "model_family_kept": False,
                        "rationale": "oscillation data needs a damped cosine, not a peak"})
    fake, captured = make_self([reply])
    ctx = make_ctx()
    assert C._try_bank_edit_adapt(fake, ctx) is None
    assert "base_script" not in captured
    assert ctx.bank_adapt_attempt["model_mismatch"] is True
    assert "model family mismatch" in ctx.bank_adapt_attempt["fell_through"]


def test_family_guard_withholds_credit_for_a_rewritten_model(monkeypatch):
    from scilink.skills._shared import _script_bank
    bumped = []
    monkeypatch.setattr(_script_bank, "record_success",
                        lambda d, rid, session=None, **kw: bumped.append(rid))
    fake, _ = make_self([])
    rec = {"id": "rec_001", "technique_signals": {
        "model_type": "Linear baseline + single exponentially-modified Gaussian peak"}}
    ctx = SimpleNamespace(bank_exemplar={"record": rec}, bank_adapt_attempt={})
    res = {"success": True, "bank_edit_adapt": {"id": "rec_001", "n_edits": 8},
           "model_type": "Constant baseline + damped cosine ring-down"}
    C._bump_bank_adapt_success(fake, res, ctx)
    assert bumped == [] and res["bank_edit_adapt"]["model_changed"] is True
    assert ctx.bank_adapt_attempt["model_changed"] is True
    # a profile swap within the same kind of signal still counts
    res2 = {"success": True, "bank_edit_adapt": {"id": "rec_001", "n_edits": 3},
            "model_type": "Sigmoidal step baseline + pseudo-Voigt peak"}
    C._bump_bank_adapt_success(fake, res2, ctx)
    assert bumped == ["rec_001"] and "model_changed" not in res2["bank_edit_adapt"]


def test_family_words_map_to_signal_kinds():
    from scilink.agents.exp_agents._qc_engine import (
        model_families_disjoint, model_family_tokens, relax_snippet_edits)
    assert model_family_tokens("EMG asymmetric peak on a linear baseline") == {"peak"}
    assert model_families_disjoint("Gaussian peak", "damped cosine")
    assert not model_families_disjoint("Gaussian peak", "Lorentzian doublet")
    assert not model_families_disjoint("", "damped cosine")          # nothing to compare
    # an ambiguous block (two identical stripped matches) is left alone
    text = "a = 1\n    x = 2\nb = 3\n    x = 2\n"
    out, n = relax_snippet_edits(text, [{"old_text": "        x = 2", "new_text": "x = 9"}])
    assert n == 0 and out[0]["old_text"] == "        x = 2"


def test_only_applied_adaptations_charge_the_record(monkeypatch):
    from scilink.agents.exp_agents._qc_engine import record_bank_assist
    from scilink.skills._shared import _script_bank
    failures, logged = [], []
    monkeypatch.setattr(_script_bank, "record_failure",
                        lambda d, rid, reason, session=None, **kw: failures.append(reason))
    monkeypatch.setattr(_script_bank, "log_assist", lambda block: logged.append(block))
    host = SimpleNamespace(output_dir="/tmp/sess")
    rec = {"record": {"id": "rec_001"}, "score": 0.5}

    def ctx(attempt):
        return SimpleNamespace(bank_exemplar=rec, bank_adapt_attempt=attempt,
                               item_name="s", is_anchor=True, state={})
    res = {"success": True, "quality_history": {"approved": True}}
    record_bank_assist(host, ctx({"applied": False, "executed": False, "n_edits": None,
                                  "fell_through": "edits do not apply: ..."}),
                       dict(res), domain="curve_fitting")
    record_bank_assist(host, ctx({"applied": False, "executed": False, "model_mismatch": True,
                                  "fell_through": "model family mismatch"}),
                       dict(res), domain="curve_fitting")
    assert failures == [] and logged[1]["model_mismatch"] is True
    record_bank_assist(host, ctx({"applied": True, "executed": False, "n_edits": 2}),
                       dict(res), domain="curve_fitting")
    record_bank_assist(host, ctx({"applied": True, "executed": True, "n_edits": 2}),
                       dict(res), domain="curve_fitting")   # executed, then replaced by a refit
    assert failures == ["edit_adapt_not_executed", "edit_adapt_replaced"]
