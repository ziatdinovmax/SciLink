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
    assert s["blocks"][2]["rows"][0] == ["low", "0 (T=300 K), 1 (T=400 K)", "dataset 0", "one phase"]
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
    assert s["title"] == "✅ Proposed experimental plan"
    labels = [b.get("label") or b.get("title") for b in s["blocks"]]
    assert labels == ["📄 Full report", "🔬 Experiment: Anneal series", "🧪 Experimental steps",
                      "🛠️ Required equipment", "📈 Expected outcome", "💡 Justification",
                      "📄 Source documents", "💻 Implementation code",
                      "⚠️ Caveats & potential limitations"]
    assert s["blocks"][1]["markdown"] == "🎯 **Hypothesis.** grains grow"
    assert s["blocks"][2] == {"type": "steps", "label": "🧪 Experimental steps",
                              "items": ["Cut coupons", "Anneal 1 h", "▸ DOMAIN 2", "Image"]}
    assert s["blocks"][3]["markdown"] == "furnace, SEM"
    assert s["blocks"][6]["markdown"] == "- paper.pdf"
    assert s["blocks"][8]["lines"] == ["Minor: [safety] no PPE", "BLOCKING: [budget] 800 C > furnace max"]
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
    assert labels == ["💡 Research direction: Portfolio", "🧠 Research directions (1)",
                      "🧭 Shared protocol", "🛠️ Key capabilities", "📄 Source documents"]
    assert "📈 **Expected outcomes.** o" in s["blocks"][0]["markdown"]
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


# ── stage 3: the fit and result gates ───────────────────────────

def _class_with(module, method):
    return next(c for c in vars(module).values() if isinstance(c, type) and method in vars(c))


def test_fit_review_subject_and_gate(tmp_path):
    fit = {"model_type": "2 Gaussians", "parameters": {
        "peak_1": {"center": 302.0123, "fwhm": 14.0, "center_err": 0.03, "eta": "n/a"},
        "baseline": "flat"}}
    s = cfc.fit_review_subject(fit, 0.9912, 0.95, 5, "/s/first_spectrum_fit_review.png")
    assert s["title"].startswith("📊 First spectrum fit result")
    assert [b["type"] for b in s["blocks"]] == ["figure", "fields", "table", "notice"]
    assert s["blocks"][1]["items"][1] == {"label": "📊 R²", "value": "0.9912 (threshold 0.95)",
                                          "flag": "ok"}
    assert s["blocks"][2]["rows"] == [["peak_1", "center", "302"], ["peak_1", "fwhm", "14"],
                                      ["peak_1", "eta", "n/a"]]     # _err skipped, no dict skipped
    assert "all 5 spectra" in s["blocks"][3]["lines"][0]
    assert cfc.fit_review_subject(fit, 0.5, 0.95, 1, None)["blocks"][0]["type"] == "fields"
    assert cfc.fit_review_subject(fit, 0.5, 0.95, 1, None)["blocks"][0]["items"][1]["flag"] == "bad"

    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    cls = _class_with(cfc, "_get_user_feedback_on_fit")
    owner = SimpleNamespace(output_dir=tmp_path, r2_threshold=0.95)
    fit["visualization_bytes"] = b"png"
    assert cls._get_user_feedback_on_fit(owner, {"num_spectra": 3}, fit, 0.97) is None
    assert cap.req.kind == "review_fit" and cap.req.origin == {"stage": "fit_review"}
    assert cap.req.subject["blocks"][0]["path"] == str(tmp_path / "first_spectrum_fit_review.png")


def test_poor_fit_subject_and_gate(tmp_path):
    best = {"fit_quality": {"r_squared": 0.81}, "visualization_bytes": b"png"}
    attempts = [{"model": "1 Gaussian", "r2": 0.7}, {"model": "2 Gaussians", "r2": 0.81}]
    s = cfc.poor_fit_subject(best, attempts, 0.95, 0.9, "/s/quality_review_fit.png")
    assert s["title"] == "⚠️ Fit quality below threshold"
    assert [b["type"] for b in s["blocks"]] == ["figure", "text", "fields", "table", "text"]
    assert s["blocks"][2]["items"][0]["value"] == "R² = 0.8100"
    assert s["blocks"][3]["rows"] == [["1 Gaussian", "0.7000"], ["2 Gaussians", "0.8100"]]
    assert 'threshold 0.90' in s["blocks"][4]["markdown"]

    cap = Capture(answer="accept")
    hitl.set_default_channel(cap)
    cls = _class_with(cfc, "_get_human_feedback_for_poor_fit")
    owner = SimpleNamespace(output_dir=tmp_path, r2_threshold=0.95, _r2_soft_margin=lambda t: 0.05,
                            HUMAN_FEEDBACK_PROMPT=cls.HUMAN_FEEDBACK_PROMPT,
                            logger=SimpleNamespace(warning=lambda *a, **k: None))
    assert cls._get_human_feedback_for_poor_fit(owner, {}, best, attempts) is None
    assert cap.req.origin == {"stage": "poor_fit_review"}
    assert cap.req.subject["blocks"][0]["path"] == str(tmp_path / "quality_review_fit.png")


def test_result_review_subject_and_gate(tmp_path):
    result = {"analysis_type": "grain segmentation",
              "extracted_features": {"grain_count": 42, "mean_diameter_um": 7.123456}}
    s = iac.result_review_subject({"is_single_image": True}, result, 0.87, "/s/review.png")
    assert s["title"] == "Analysis result — review before synthesis"
    assert [b["type"] for b in s["blocks"]] == ["figure", "fields", "table"]
    assert s["blocks"][1]["items"] == [{"label": "Analysis", "value": "grain segmentation"},
                                       {"label": "Quality score", "value": "0.87"}]
    assert s["blocks"][2]["rows"] == [["grain_count", 42], ["mean_diameter_um", "7.123"]]
    series = iac.result_review_subject({"is_single_image": False, "num_images": 6,
                                        "_current_regime_name": "late"}, result, 0.5, None)
    assert series["title"].startswith("First image result")
    assert series["blocks"][-1] == {"type": "notice", "title": "⚠️ Locked pipeline", "lines": [
        "This analysis pipeline will be applied to all images in regime 'late'."]}

    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    cls = _class_with(iac, "_get_user_feedback_on_result")
    owner = SimpleNamespace(output_dir=tmp_path)
    result["visualization_bytes"] = b"png"
    assert cls._get_user_feedback_on_result(owner, {"is_single_image": True}, result, 0.87) is None
    assert cap.req.kind == "review_result" and cap.req.origin == {"stage": "result_review"}
    assert cap.req.subject["blocks"][0]["path"] == str(tmp_path / "first_image_analysis_review.png")


def test_poor_quality_subject_and_gate(tmp_path):
    best = {"_quality_score": 0.42, "visualization_bytes": b"png"}
    attempts = [{"pipeline": "threshold", "score": 0.3},
                {"pipeline": "Verification 1", "score": 0.42}]
    s = iac.poor_quality_subject(best, attempts, None)
    assert s["title"] == "⚠️ Analysis quality below threshold"
    assert [b["type"] for b in s["blocks"]] == ["text", "fields", "table", "text"]
    assert s["blocks"][1]["items"][0]["value"] == "Quality score = 0.42"
    assert s["blocks"][2]["rows"] == [["threshold", "0.30"], ["Verification 1", "0.42"]]

    cap = Capture(answer="")
    hitl.set_default_channel(cap)
    cls = _class_with(iac, "_get_human_feedback_for_poor_quality")
    owner = SimpleNamespace(output_dir=tmp_path, HUMAN_FEEDBACK_PROMPT=cls.HUMAN_FEEDBACK_PROMPT,
                            logger=SimpleNamespace(warning=lambda *a, **k: None))
    assert cls._get_human_feedback_for_poor_quality(owner, {}, best, attempts) is None
    assert cap.req.origin == {"stage": "poor_quality_review"}
    assert cap.req.subject["blocks"][0]["path"] == str(tmp_path / "quality_review_analysis.png")


def test_scalarizer_review_subject():
    from scilink.agents.planning_agents.scalarizer_agent import scalarizer_review_subject
    rows = [{"yield": 0.5, "temp": 300}, {"yield": 0.6, "temp": 310},
            {"yield": 0.7, "temp": 320}, {"yield": 0.8, "temp": 330}]
    s = scalarizer_review_subject("results.csv", ["yield", "temp"], rows, "/s/plot.png")
    assert s["title"] == "👀 Scalarizer review — results.csv"
    assert s["blocks"][0]["markdown"] == "Extracted 2 column(s) from 4 data point(s); the first 3 shown."
    assert s["blocks"][1]["rows"] == [["0.5", "300"], ["0.6", "310"], ["0.7", "320"]]
    assert s["blocks"][2] == {"type": "figure", "path": "/s/plot.png", "caption": "Extraction plot"}
    one = scalarizer_review_subject("r.csv", ["a"], [{"a": 1}], None)
    assert one["blocks"][0]["markdown"].endswith("1 data point(s).") and len(one["blocks"]) == 2
