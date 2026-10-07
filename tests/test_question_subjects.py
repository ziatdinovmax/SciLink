"""The gates that declare what is under review (scilink.hitl ``subject``):
their block builders, and that the request they ask carries it."""

from types import SimpleNamespace

import pytest

from scilink import hitl
from scilink.agents.exp_agents.controllers import base_controllers as base
from scilink.agents.exp_agents.controllers import curve_fitting_controllers as cfc
from scilink.agents.exp_agents.controllers import hyperspectral_series as hs
from scilink.agents.exp_agents.controllers import image_analysis_controllers as iac
from scilink.agents.planning_agents import user_interface as ui


@pytest.fixture(autouse=True)
def _reset():
    yield
    hitl.set_default_channel(None)


class Capture:
    def __init__(self, answer=""):
        self.answer = answer
        self.req = None

    def ask(self, req):
        self.req = req
        return self.answer


def test_subject_block_vocabulary_is_checked():
    b = hitl.subject_block("chips", label="P", items=["a"])
    assert b == {"type": "chips", "label": "P", "items": ["a"]}
    with pytest.raises(ValueError):
        hitl.subject_block("hologram")
    s = hitl.make_subject("T", [b, None, {}])
    assert s == {"title": "T", "blocks": [b]}


def test_numbered_steps_split_only_after_a_sentence_end():
    assert base.numbered_steps("1. Baseline with 8.7 cm-1 window. 2. Fit two peaks") == \
        ["Baseline with 8.7 cm-1 window.", "Fit two peaks."]
    assert base.numbered_steps("Fit one Gaussian.") == ["Fit one Gaussian."]
    assert base.numbered_steps("") == []
    # an arrow chain is a pipeline; a lone step is not a list
    assert base.numbered_steps("Denoise (sigma~1) -> Sobel gradient -> watershed") == \
        ["Denoise (sigma~1).", "Sobel gradient.", "watershed."]
    assert base.steps_block("⚙️ Pipeline", "Threshold and label.") == \
        {"type": "text", "label": "⚙️ Pipeline", "markdown": "Threshold and label."}
    assert base.steps_block("⚙️ Pipeline", "1. a. 2. b")["type"] == "steps"
    # an arrow inside prose means "implies" (live: a fitting strategy split
    # into a paragraph and two dangling fragments) — the paragraph stays whole
    prose = ("TRF least-squares over the full range. Inspect residuals: an S-shaped "
             "residual on the dominant band -> switch that band to an asymmetric "
             "profile; a bimodal residual -> keep both components. Verify the areas.")
    assert base.numbered_steps(prose) == [prose]


def test_fitting_plan_subject_single_and_series():
    state = {"is_single_spectrum": True, "observations": "two peaks",
             "analysis_approach": "peak fit", "physical_model": "2 Lorentzians",
             "parameters_to_extract": ["center", "fwhm"],
             "fitting_strategy": "1. Subtract baseline. 2. Fit both peaks."}
    s = cfc.fitting_plan_subject(state)
    assert s["title"] == "📋 Proposed fitting plan — single spectrum"
    assert [(b["type"], b["label"]) for b in s["blocks"]] == [
        ("text", "🔍 Observations"), ("text", "📊 Approach"), ("text", "📐 Physical model"),
        ("text", "🎯 Parameters to extract"), ("steps", "⚙️ Fitting strategy")]
    assert s["blocks"][2]["markdown"] == "2 Lorentzians"
    assert s["blocks"][3]["markdown"] == "center, fwhm"
    assert s["blocks"][4]["items"] == ["Subtract baseline.", "Fit both peaks."]

    series = dict(state, is_single_spectrum=False, num_spectra=6,
                  series_metadata={"values": [100, 200, 300, 400, 500, 600], "unit": "K"},
                  series_analysis_plan={
                      "rationale": "a phase change",
                      # a regime left sparse falls back to the series plan's
                      # fields, as the printer does
                      "physical_model": "2 Lorentzians",
                      "parameters_to_extract": ["center", "fwhm"],
                      "regimes": [{"name": "low", "spectrum_indices": [0, 1, 2]},
                                  {"name": "high", "spectrum_indices": [3, 4, 5],
                                   "physical_model": "1 Lorentzian"}],
                      "transition_points": [{"between_indices": [2, 3], "description": "split"}]})
    s = cfc.fitting_plan_subject(series)
    table = next(b for b in s["blocks"] if b["type"] == "table")
    assert table["label"] == "📦 Series fitting regimes (2)"
    assert table["rows"][0] == [1, "low", "[0, 1, 2] (100–300 K)", "2 Lorentzians", "center, fwhm"]
    assert table["rows"][1][3] == "1 Lorentzian"
    assert [b["type"] for b in s["blocks"]][-3:] == ["table", "text", "table"]
    locked = cfc.fitting_plan_subject(dict(state, is_single_spectrum=False, num_spectra=3))
    assert locked["blocks"][-1]["type"] == "notice"


def test_fitting_plan_gate_asks_with_the_subject(capsys):
    cap = Capture(answer="use Voigt")
    hitl.set_default_channel(cap)
    owner = SimpleNamespace(_display_plan=lambda state: None)
    state = {"is_single_spectrum": True, "physical_model": "G"}
    out = cfc.CurveFittingPlanningController._get_human_feedback(owner, state)
    assert cap.req.kind == "review_plan" and cap.req.origin == {"stage": "fitting_plan"}
    assert cap.req.subject["title"].startswith("📋 Proposed fitting plan")
    assert out["_refine_feedback"] == "use Voigt"


# ── stage 2: the other plan gates ────────────────────────────────

def test_image_analysis_plan_subject_single_and_series():
    state = {"is_single_image": True, "observations": "grains", "analysis_approach": "segment",
             "processing_pipeline": "1. Flatten. 2. Threshold. 3. Label.",
             "features_to_extract": ["grain_size", "count"], "quality_criteria": "no merged grains",
             "expected_outputs": ["mask.png"]}
    s = iac.analysis_plan_subject(state)
    assert s["title"] == "📋 Proposed analysis plan — single image"
    assert [(b["type"], b["label"]) for b in s["blocks"]] == [
        ("text", "🔍 Observations"), ("text", "📊 Approach"), ("steps", "⚙️ Pipeline"),
        ("text", "🎯 Features to extract"), ("text", "✅ Quality criteria"),
        ("text", "📄 Expected outputs")]
    assert s["blocks"][2]["items"] == ["Flatten.", "Threshold.", "Label."]
    assert s["blocks"][3]["markdown"] == "grain_size, count"
    series = dict(state, is_single_image=False, num_images=4,
                  series_metadata={"values": [1, 2, 3, 4], "unit": "h"},
                  series_analysis_plan={"regimes": [
                      {"name": "early", "image_indices": [0, 1], "processing_pipeline": "p1",
                       "features_to_extract": ["a"]},
                      {"name": "late", "image_indices": [2, 3]}]})
    table = next(b for b in iac.analysis_plan_subject(series)["blocks"] if b["type"] == "table")
    assert table["label"] == "📦 Image analysis regimes (2)"
    assert table["rows"][0] == [1, "early", "[0, 1] (1–2 h)", "p1", "a"]
    assert table["rows"][1][3] == "N/A"
    locked = iac.analysis_plan_subject(dict(state, is_single_image=False, num_images=3))
    assert locked["blocks"][-1]["type"] == "notice"


def test_image_plan_gate_asks_with_the_subject():
    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    owner = SimpleNamespace(_display_plan=lambda state: None)
    out = iac.ImagePlanningController._get_human_feedback(owner, {"is_single_image": True})
    assert cap.req.origin == {"stage": "analysis_plan"}
    assert cap.req.subject["title"].startswith("📋 Proposed analysis plan")
    assert "_refine_requested" not in out


def test_refinement_plan_subject_and_gate():
    state = {"skip_decomposition": True, "iteration_title": "Iteration 0",
             "refinement_decision": {"refinement_needed": True, "reasoning": "two phases",
                                     "targets": [{"type": "nmf", "value": 3, "description": "3 comps"},
                                                 {"type": "custom_code", "value": None,
                                                  "description": "mask the vacuum"}]}}
    s = base.refinement_plan_subject(state)
    assert s["title"] == "🎯 Analysis plan review — Iteration 0"
    assert s["blocks"][0]["label"] == "Summary of current analysis"
    assert "Skip-decomposition" in s["blocks"][0]["markdown"]
    assert "Analysis plan ready = **True**" in s["blocks"][1]["markdown"]
    table = s["blocks"][2]
    assert table["label"] == "🎯 Targeted actions (2)"
    assert table["rows"] == [[1, "nmf", "3", "3 comps"], [2, "custom_code", "", "mask the vacuum"]]
    step = base.refinement_plan_subject({"result_json": {"detailed_analysis": "seen"},
                                         "refinement_decision": {}})
    assert step["title"] == "🎯 Analysis step review — Current Analysis"
    assert step["blocks"][0]["markdown"] == "seen"
    assert step["blocks"][-1]["markdown"] == "No specific targets were generated."

    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    ctrl = base.IterativeFeedbackController(None, SimpleNamespace(info=lambda *a, **k: None,
                                                                    warning=lambda *a, **k: None),
                                            None, None, parse_fn=None, settings={},
                                            refinement_instruction="")
    full = dict(state, settings={"enable_human_feedback": True}, current_depth=0)
    assert ctrl.execute(full) is full
    assert cap.req.origin == {"stage": "preprocess_plan"} and cap.req.subject == s


def test_regime_plan_subject():
    meta = {"values": [300, 400, 500], "variable": "T", "unit": "K"}
    scout = {"reduction": {"change_point": 450.0, "change_sharpness": 0.8,
                           "axis_coherence": {"coherent": False}}}
    plan = {"rationale": "a phase change", "regimes": [
        {"name": "low", "dataset_indices": [0, 1], "description": "one phase"},
        {"name": "high", "dataset_indices": [2]}],
        "transition_points": [{"between_indices": [1, 2], "description": "melt"}]}
    s = hs.regime_plan_subject(plan, meta, scout, 3)
    assert s["title"] == "📋 Proposed series regime plan"
    assert [b["label"] for b in s["blocks"]] == ["🔎 Change detection", "💡 Rationale",
                                                 "Regimes (2)", "↕ Transition points"]
    assert "NOT coherent" in s["blocks"][0]["markdown"]
    assert s["blocks"][2]["rows"][0] == ["low", "0 (300 K), 1 (400 K)", "dataset 0", "one phase"]
    assert s["blocks"][2]["columns"][1] == "Datasets (T)"
    assert s["blocks"][3]["rows"] == [["[1, 2]", "melt"]]
    one = hs.regime_plan_subject(None, meta, None, 3)
    assert one["blocks"] == [{"type": "text", "label": "Regimes",
                              "markdown": "1 regime — all 3 datasets share one locked script."}]


def _experiment_plan():
    return {"proposed_experiments": [{
        "experiment_name": "Anneal series", "hypothesis": "grains grow",
        "experimental_steps": ["1. Cut coupons", "", "2) Anneal 1 h", "=== DOMAIN 2 ===", "Image"],
        "required_equipment": ["furnace", "SEM"], "expected_outcome": "bigger grains",
        "justification": "Ostwald", "source_documents": ["paper.pdf"],
        "implementation_code": "print(1)"}],
        "critic_findings": [{"dimension": "safety", "issue": "no PPE", "severity": "minor"},
                            {"dimension": "budget", "issue": "800 C > furnace max",
                             "severity": "blocking"}]}


def test_plan_subject_experiment():
    s = ui.plan_subject(_experiment_plan(), report_path="/r/plan.html")
    assert s["title"] == "✅ Proposed experimental plan" and "report" not in s
    labels = [b.get("label") or b.get("title") for b in s["blocks"]]
    assert labels == ["🔬 Experiment: Anneal series", "🧪 Experimental steps",
                      "🛠️ Required equipment", "📈 Expected outcome", "💡 Justification",
                      "📄 Source documents", "💻 Implementation code",
                      "⚠️ Caveats & potential limitations"]
    assert s["blocks"][0]["markdown"] == "🎯 **Hypothesis.** grains grow"
    assert s["blocks"][1] == {"type": "steps", "label": "🧪 Experimental steps",
                              "items": ["Cut coupons", "Anneal 1 h", "▸ DOMAIN 2", "Image"]}
    assert s["blocks"][2]["markdown"] == "furnace, SEM"
    assert s["blocks"][5]["markdown"] == "- paper.pdf"
    assert s["blocks"][7]["lines"] == ["Minor: [safety] no PPE", "BLOCKING: [budget] 800 C > furnace max"]
    # two experiments are numbered; more than five pieces of equipment are a list
    two = _experiment_plan()
    two["proposed_experiments"].append(dict(two["proposed_experiments"][0],
                                            experiment_name="B", required_equipment=list("abcdef")))
    s = ui.plan_subject(two)
    assert s["blocks"][0]["label"] == "🔬 Experiment 1: Anneal series"
    assert next(b for b in s["blocks"] if b.get("label") == "🔬 Experiment 2: B")
    assert [b for b in s["blocks"] if b.get("label") == "🛠️ Required equipment"][1]["markdown"] \
        == "\n".join(f"- {c}" for c in "abcdef")


def test_plan_subject_ideation_error_and_empty():
    plan = {"proposed_experiments": [{
        "experiment_name": "Portfolio", "hypothesis": "h", "expected_outcome": "o",
        "justification": "j", "experimental_steps": ["PS-1 do x"],
        "concepts": [{"id": "D1", "title": "Catch it", "tier": 1, "hypothesis": "hh",
                      "details": ["d1", "d2"], "b_operando_only_question": "why"}],
        "required_equipment": []}]}
    s = ui.plan_subject(plan, ideation=True)
    assert s["title"] == "✅ Proposed research directions"
    labels = [b["label"] for b in s["blocks"]]
    assert labels == ["🧭 Portfolio: Portfolio", "🧠 Research directions (1)",
                      "🧭 Shared protocol", "🛠️ Key capabilities", "📄 Source documents"]
    assert "report" not in s
    assert s["blocks"][0]["markdown"].startswith("🎯 **Thesis.** h")
    assert "📈 **Expected outcomes.** o" in s["blocks"][0]["markdown"]
    # an ideation entry without concepts is a direction in its own right
    lone = ui.plan_subject({"proposed_experiments": [{"experiment_name": "D", "hypothesis": "h"}]},
                           ideation=True)
    assert lone["blocks"][0]["label"] == "💡 Research direction: D"
    md = s["blocks"][1]["markdown"]
    assert md.startswith("**D1: Catch it** · tier 1") and "  - d1" in md
    assert "*Operando only question.* why" in md
    assert s["blocks"][2]["markdown"] == "- PS-1 do x"       # ideation keeps the author's labels
    assert s["blocks"][3]["markdown"] == "None specified."
    err = ui.plan_subject({"error": "boom"})
    assert err["blocks"][0]["type"] == "notice" and err["blocks"][0]["lines"] == ["boom"]
    assert ui.plan_subject({})["blocks"][0]["title"].startswith("⚠️ No experiments")


def test_planning_review_gate_passes_the_subject(capsys):
    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    subject = ui.plan_subject(_experiment_plan())
    assert ui.get_user_feedback(auto_repair={"status": "applied", "notes": [
        {"was": "950 C", "now": "850 C", "why": "limit"}]}, subject=subject) is None
    assert cap.req.kind == "approve_or_revise" and cap.req.subject == subject
    assert cap.req.origin["auto_repair"] == ["950 C -> 850 C (limit)"]
    assert ui.get_user_feedback() is None and cap.req.subject is None   # the code-review callers


# ── stage 4: the pickers and comparisons ────────────────────────

def _class_with(module, method):
    return next(c for c in vars(module).values() if isinstance(c, type) and method in vars(c))


def _bestofn_candidates():
    return [{"attempt": 0, "score": 0.9812, "approved": True, "iterations": 2, "success": True,
             "result": {"visualization_bytes": b"png"}},
            {"attempt": 1, "score": 0.9534, "approved": False, "iterations": 4, "success": True,
             "result": {}}]


def test_bestofn_join_subject_and_gates(tmp_path):
    cands = _bestofn_candidates()
    s = base.bestofn_join_subject(cands, cands[0], {"reasoning": "cleanest residual"}, True,
                                  "R²", tmp_path)
    assert s["title"].startswith("🏁 Best-of-N candidates")
    block = s["blocks"][0]
    assert block["pick"] == 0 and block["reasoning"] == "cleanest residual"
    assert block["items"][0] == {"idx": 0, "name": "2 iterations", "metric": "R²", "value": "0.9812",
                                 "approved": True,
                                 "figure": str(tmp_path / "bestofn_candidate_00_review.png")}
    assert block["items"][1]["figure"] is None and block["items"][1]["approved"] is False
    assert block["free_text"]["input"].startswith("Or type 'more'")
    image = base.bestofn_join_subject(cands, cands[1], {}, False, "score", tmp_path)
    assert image["blocks"][0]["items"][0]["value"] == "0.98" and "free_text" not in image["blocks"][0]

    for module, metric in ((cfc, "R²"), (iac, "score")):
        cap = Capture(answer="1")
        hitl.set_default_channel(cap)
        cls = _class_with(module, "_get_bestofn_join_approval")
        owner = SimpleNamespace(output_dir=tmp_path)
        assert cls._get_bestofn_join_approval(owner, cands, cands[0], {"reasoning": "r"}, False) == 1
        assert cap.req.kind == "bestofn_select" and cap.req.origin == {"stage": "bestofn_join"}
        assert cap.req.subject["blocks"][0]["items"][0]["metric"] == metric


def test_consensus_subject_and_gates():
    improved = [{"index": 2, "new_model": "Voigt", "new_r2": 0.99},
                {"index": 5, "new_model": "Gaussian", "new_r2": 0.97},
                {"index": 7, "new_model": "Voigt", "new_r2": 0.985}]
    counts = {"Gaussian": 1, "Voigt": 2}
    s = base.consensus_subject(improved, counts, "spectra", "model", "new_model", "new_r2", "R²")
    assert s["title"] == "🔄 Adaptive refit — no model consensus among the re-fitted spectra"
    block = s["blocks"][0]
    assert block["pick"] is None
    assert block["items"] == [
        {"idx": 1, "name": "Voigt", "judge_comment": "spectra [2, 7] · R²: 0.9900, 0.9850"},
        {"idx": 2, "name": "Gaussian", "judge_comment": "spectra [5] · R²: 0.9700"}]
    assert block["free_text"] == {"input": "Or suggest a different model:", "submit": "Use this model"}

    cap = Capture(answer="1")
    hitl.set_default_channel(cap)
    cls = _class_with(cfc, "_ask_user_for_consensus")
    assert cls._ask_user_for_consensus(SimpleNamespace(), improved, counts) == "Voigt"
    assert cap.req.kind == "consensus_select" and cap.req.subject == s
    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    cls = _class_with(iac, "_ask_user_for_consensus")
    imgs = [{"index": 1, "new_pipeline": "watershed", "new_score": 0.8}]
    assert cls._ask_user_for_consensus(SimpleNamespace(), imgs, {"watershed": 1}) is None
    assert cap.req.subject["blocks"][0]["items"][0]["judge_comment"] == "images [1] · score: 0.80"


def test_consistency_subject_and_gates():
    s = base.consistency_subject("spectrum", 3, "T400K.csv", "model", "Voigt", 0.95, "Gaussian",
                                 0.98, "R²")
    assert s["title"] == "⚠️ Spectrum [3] T400K.csv: the consensus model has a lower R²"
    cmp = s["blocks"][0]
    assert cmp["left"]["label"] == "Consensus" and cmp["right"]["label"] == "Independent"
    assert cmp["left"]["blocks"][0]["items"] == [{"label": "Model", "value": "Voigt"},
                                                 {"label": "R²", "value": "0.9500", "flag": "bad"}]
    assert cmp["right"]["blocks"][0]["items"][1] == {"label": "R²", "value": "0.9800", "flag": "ok"}

    cap = Capture(answer="consensus")
    hitl.set_default_channel(cap)
    cls = _class_with(cfc, "_ask_keep_consistency_result")
    assert cls._ask_keep_consistency_result(SimpleNamespace(), "T400K.csv", 3, "Voigt", 0.95,
                                            "Gaussian", 0.98) is True
    assert cap.req.options == ["consensus", ""] and cap.req.subject == s
    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    cls = _class_with(iac, "_ask_keep_consistency_result")
    assert cls._ask_keep_consistency_result(SimpleNamespace(), "img_02.npy", 2, "watershed", 0.7,
                                            "threshold", 0.9) is False
    assert cap.req.subject["blocks"][0]["left"]["blocks"][0]["items"][0] == \
        {"label": "Pipeline", "value": "watershed"}


def test_plan_candidates_subject_and_gates():
    cands = [{"proposed_experiments": [{"experiment_name": "Anneal series", "hypothesis": "h1",
                                        "expected_outcome": "o1", "justification": "j1"}]},
             {"proposed_experiments": [{"experiment_name": "Quench series", "hypothesis": "h2"}]}]
    judge = {"selected_candidate": 2, "reasoning": "quench isolates the variable",
             "scores": [{"candidate": 1, "groundedness": 4, "testability": 3, "actionability": 4,
                         "feasibility": 5, "information_gain": 2, "comment": "weak gain"},
                        {"candidate": 2, "groundedness": 5, "testability": 5, "actionability": 4,
                         "feasibility": 4, "information_gain": 4}]}
    s = ui.plan_candidates_subject(cands, judge, 2, report_paths=["/r/c1.html", "/r/c2.html"],
                                   pick_caveats=["Minor: [safety] no PPE"])
    assert s["title"] == "🧭 Plan candidates — 2 distinct strategies"
    block = s["blocks"][0]
    assert block["pick"] == 2 and block["reasoning"] == "quench isolates the variable"
    assert block["caveats"] == ["Minor: [safety] no PPE"]
    first = block["items"][0]
    assert first["idx"] == 1 and first["name"] == "Anneal series"
    assert first["judge_comment"].startswith("groundedness 4/5 · testability 3/5")
    assert first["judge_comment"].endswith("— weak gain")
    assert "🎯 **Hypothesis.** h1" in first["body"] and first["report"] == "/r/c1.html"
    assert "report" not in ui.plan_candidates_subject(cands, judge, 2)["blocks"][0]["items"][0]
    assert block["items"][1]["body"].count("N/A") == 2

    cap = Capture(answer="1")
    hitl.set_default_channel(cap)
    assert ui.get_candidate_selection(2, 2, subject=s) == 1
    assert cap.req.kind == "plan_candidate_select" and cap.req.subject == s
    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    plan = ui.plan_subject(_experiment_plan())
    assert ui.get_reopen_decision("limit exceeded", subject=plan)[0] == "keep"
    assert cap.req.kind == "keep_or_revert" and cap.req.options == ["keep", "revert"]
    assert cap.req.subject == plan and cap.req.origin["reason"] == "limit exceeded"


# ── stage 5: metadata, fan-out confirm, code review ─────────────

def test_dataset_description_subject_and_gate(tmp_path):
    from scilink.agents.planning_agents.user_interface import (
        dataset_description_subject, get_dataset_description)
    s = dataset_description_subject("yields.csv")
    assert s["title"] == "⚠️ Missing metadata for yields.csv"
    assert s["blocks"][0]["label"] == "Why" and "Enter skips" in s["blocks"][0]["markdown"]
    cap = Capture(answer="Suzuki coupling yields")
    hitl.set_default_channel(cap)
    assert get_dataset_description("yields.csv") == "Suzuki coupling yields"
    assert cap.req.kind == "dataset_description" and cap.req.subject == s
    assert cap.req.origin["filename"] == "yields.csv"


def test_code_review_subject_and_gate(tmp_path):
    from scilink.agents.planning_agents.user_interface import code_review_subject
    s = code_review_subject(tmp_path / "temp_code_review",
                            [tmp_path / "temp_code_review" / "exp_1.py",
                             tmp_path / "temp_code_review" / "exp_2.py"])
    assert s["title"] == "👀 Code review required"
    assert s["blocks"][0]["items"][1] == {"label": "Scripts", "value": "exp_1.py, exp_2.py"}
    assert code_review_subject("/r", [], iteration=3)["title"].endswith("— iteration 3")
    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    assert ui.get_user_feedback(subject=s, stage="code_review") is None
    assert cap.req.kind == "approve_or_revise" and cap.req.origin["stage"] == "code_review"
    assert cap.req.subject == s


def test_fanout_confirm_subject_and_gate(monkeypatch):
    import scilink.agents.meta_agent.fanout as fo
    verdict = {"verdict": "complementary", "confidence": 0.9, "join_axis": "temperature",
               "join_type": "outer", "rationale": "same coupon", "unrelated": ["c"]}
    branches = {"a": {"label": "XRD", "data_path": "/d/xrd.csv"},
                "b": {"label": "Raman", "data_path": "/d/raman.csv", "steer": True}}
    s = fo.fanout_confirm_subject(verdict, ["a", "b"], branches, True, 2, False)
    assert s["title"].startswith("🔀 Parallel multi-dataset analysis")
    assert [(b["type"], b.get("label")) for b in s["blocks"]] == [
        ("fields", None), ("text", "Rationale"), ("text", "🔀 Branches (2)"),
        ("fields", None), ("text", "Branch approvals")]
    assert s["blocks"][0]["items"][0]["value"] == "complementary (confidence 0.9)"
    assert "- XRD  (xrd.csv)" in s["blocks"][2]["markdown"] and "operand mesh" in s["blocks"][2]["markdown"]
    assert [i["label"] for i in s["blocks"][3]["items"]] == ["Pruned as unrelated", "Steering opt-in"]
    assert "autonomously" in s["blocks"][4]["markdown"]
    big = fo.fanout_confirm_subject(verdict, list("abcdefg"), {}, False, 0, True)
    assert big["blocks"][-2]["type"] == "notice" and "soft cap" in big["blocks"][-2]["title"]
    assert "pause for approvals" in big["blocks"][-1]["markdown"]

    seen = {}
    def fake_ask(prompt, **kw):
        seen.update(kw); return "y"
    monkeypatch.setattr(fo, "request_human_feedback", fake_ask)
    orch = SimpleNamespace(_enable_human_feedback=True, fanout_branch_hitl=False)
    proceed, _ = fo._confirm_fanout(orch, verdict, ["a", "b"], branches)
    assert proceed is True
    assert seen["kind"] == "confirm" and seen["origin"] == {"stage": "fanout_confirm"}
    assert seen["subject"]["blocks"][2]["label"] == "🔀 Branches (2)"


def test_numbered_steps_recognise_the_formats_plans_are_written_in():
    """#701: the pipeline printed as one paragraph — "N)" numbering, an
    unnumbered first step, "(N)" with semicolons, arrows of either kind —
    while decimals, "sigma=2." and "n_cage=4)" are never split, and a
    number out of sequence is not a step."""
    one_line = ("Calibrate at 0.15 nm/px. 2) measure_lattice_constant(window=0.2) with sigma=2. "
                "3) detect_atoms_dcnn(n_cage=4) then refine.")
    assert base.numbered_steps(one_line) == [
        "Calibrate at 0.15 nm/px.", "measure_lattice_constant(window=0.2) with sigma=2.",
        "detect_atoms_dcnn(n_cage=4) then refine."]
    assert base.numbered_steps("(1) flatten; (2) segment with threshold 0.5; (3) measure areas") == [
        "flatten.", "segment with threshold 0.5.", "measure areas."]
    assert base.numbered_steps("flatten → segment → measure") == ["flatten.", "segment.", "measure."]
    assert base.numbered_steps("Set n_cage=4) and sigma=2. Then fit.") == ["Set n_cage=4) and sigma=2. Then fit."]
    # an unnumbered first step counts only when a 2 AND a 3 follow it; a lone
    # "2." (and a 4 out of sequence) leaves the paragraph whole
    assert base.numbered_steps("Flatten the image. 2. Segment. 4. Measure.") == ["Flatten the image. 2. Segment. 4. Measure."]
    assert base.numbered_steps("1) load 2) crop 3) fit") == ["1) load 2) crop 3) fit"]   # no boundary: whole
    # a lone "2)" after an abbreviation is a citation, not an unnumbered first step's second
    for cite in ("Calibrate the detector gain (see Ref. 2) before fitting the peaks.",
                 "Compare with Fig. 2) and Eq. 2) as needed."):
        assert base.numbered_steps(cite) == [cite], cite
    # the console form: one step per line under the label, as the gate's steps block
    assert base.steps_text(one_line) == ("1. Calibrate at 0.15 nm/px.\n   2. measure_lattice_constant(window=0.2) "
                                         "with sigma=2.\n   3. detect_atoms_dcnn(n_cage=4) then refine.")
    assert base.steps_text("Threshold and label.") == "Threshold and label."
    assert base.steps_block("⚙️ Pipeline", one_line)["items"] == base.numbered_steps(one_line)
