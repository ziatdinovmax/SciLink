"""Bank minimal-edit adaptation (Phase B of the surgical-refinement plan).

Today's "adapt" regenerates the whole script with the banked one as a
prompt exemplar — eroding exactly the provenance that made it worth
banking. The edit-adapt mode asks the LLM for an edit LIST against the
banked script, applies it mechanically, and lets the UNCHANGED
verification loop judge the result. The ladder can only add: every
failure falls through (with its reason logged) to today's exemplar path.
"""

import json
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

    def generate_content(self, prompt, generation_config=None):
        self.calls += 1
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
                        lambda d, rid, session=None: bumped.append(rid))
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
                        lambda d, rid, session=None: bumped.append((d, rid)))
    fake, _ = make_self([])
    IC._bump_bank_adapt_success(fake, {
        "success": True, "bank_edit_adapt": {"id": "rec_009"}})
    assert bumped == [("image_analysis", "rec_009")]


def test_single_shared_implementation():
    from pathlib import Path
    curve = Path("scilink/agents/exp_agents/controllers/"
                 "curve_fitting_controllers.py").read_text()
    image = Path("scilink/agents/exp_agents/controllers/"
                 "image_analysis_controllers.py").read_text()
    engine = Path("scilink/agents/exp_agents/_qc_engine.py").read_text()
    assert engine.count("def try_bank_edit_adapt") == 1
    for src in (curve, image):
        assert "from .._qc_engine import try_bank_edit_adapt" in src
        assert "def try_bank_edit_adapt" not in src.replace(
            "from .._qc_engine import try_bank_edit_adapt", "")
