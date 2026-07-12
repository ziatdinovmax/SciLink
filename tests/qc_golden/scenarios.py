"""Canned LLM responses + fake analysis scripts for the golden scenarios.

Routing markers and response schemas follow the mock-authoring contract for
the two pipelines (see the design note, issue #327). Markers for prompts that
live in ``instruct.py`` track the constants; the verifier / refine-config
prompts are inline controller literals, so their distinctive substrings are
hardcoded here (a change to those literals is a behavior change the goldens
are *supposed* to catch).
"""

from __future__ import annotations

import json

from .harness import Rule

# ---------------------------------------------------------------------------
# Canned analysis scripts (executed for real in the sandbox)
# ---------------------------------------------------------------------------

_CURVE_SCRIPT = """\
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# canned harness fit script: %(tag)s
arr = np.asarray(np.load("data.npy"), dtype=float)
if arr.ndim == 2 and arr.shape[0] == 2:
    x, y = arr[0], arr[1]
elif arr.ndim == 2 and arr.shape[1] == 2:
    x, y = arr[:, 0], arr[:, 1]
else:
    x = np.arange(arr.size); y = arr.ravel()

fit = 5.0 * np.exp(-0.5 * ((x - 4.0) / 0.8) ** 2) + 0.3 * x + 1.0

plt.figure(figsize=(6, 4))
plt.plot(x, y, "k.", label="data")
plt.plot(x, fit, "r-", label="fit (%(tag)s)")
plt.legend()
plt.savefig("visualization.png", dpi=80)
plt.close()

results = {
    "model_type": "%(model_type)s",
    "parameters": {"amplitude": 5.0, "center": 4.0, "sigma": 0.8},
    "fit_quality": {"r_squared": %(r2)s, "rmse": 0.01},
    "deviation_note": "",
}
print("FIT_RESULTS_JSON:" + json.dumps(results))
"""


def curve_script(tag: str, model_type: str, r2: float) -> str:
    return _CURVE_SCRIPT % {"tag": tag, "model_type": model_type, "r2": r2}


_IMAGE_SCRIPT = """\
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# canned harness analysis script: %(tag)s
img = np.asarray(np.load("data.npy"), dtype=float)
mask = img > (img.mean() + 2.0 * img.std())

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray"); plt.title("input")
plt.subplot(1, 2, 2); plt.imshow(mask, cmap="viridis"); plt.title("%(tag)s")
plt.savefig("visualization.png", dpi=80)
plt.close()

results = {
    "analysis_type": "blob segmentation",
    # attempt_tag rides in extracted_features because the verification
    # prompt embeds this dict (not the summary string) — the scripted
    # verifier keys acceptance on it.
    "extracted_features": {"blob_count": 3,
                           "mask_fraction": round(float(mask.mean()), 6),
                           "attempt_tag": "%(tag)s"},
    "quality_metrics": {"threshold_sigma": 2.0},
    "summary": "Segmented 3 bright blobs (%(tag)s)",
    "saved_arrays": {},
}
print("IMAGE_ANALYSIS_RESULTS_JSON:" + json.dumps(results))
"""


def image_script(tag: str) -> str:
    return _IMAGE_SCRIPT % {"tag": tag}


# ---------------------------------------------------------------------------
# Shared response fragments
# ---------------------------------------------------------------------------

def _claims(topic: str) -> list[dict]:
    return [{
        "claim": f"The {topic} is consistent with the canned harness model.",
        "scientific_impact": "Golden-harness fixture claim.",
        "has_anyone_question": f"Has anyone characterized the {topic} this way?",
        "keywords": ["golden", "harness", topic.split()[0]],
    }]


def _synthesis(topic: str) -> dict:
    return {
        "detailed_analysis": f"Canned synthesis of the {topic} for the golden harness.",
        "scientific_claims": _claims(topic),
        "caveats": "",
        "suggested_followup": "",
    }


NO_SKILLS = {"skills": []}
PLAN_VALID = {"valid": True}
CONFORMANT = {"conformant": True}

# The final "good" scripts embed these tags; the verifier callables key on
# them appearing in the verification prompt (which quotes the fit results /
# analysis summary), so acceptance is content-driven, not call-count-driven.
CURVE_GOOD_R2 = 0.995
# Plain-substring token so it matches whether the prompt embeds the raw JSON
# value (0.995) or a rounded rendering (0.9950).
CURVE_GOOD_TOKEN = "0.995"
IMAGE_GOOD_TAG = "attempt-final"


# ---------------------------------------------------------------------------
# Curve rules
# ---------------------------------------------------------------------------

def _curve_verify_accept() -> dict:
    return {
        "fit_acceptable": True,
        "issues_found": [],
        "recommended_action": "none",
        "overall_assessment": "Clean fit; residuals structureless.",
    }


def _curve_verify_reject() -> dict:
    return {
        "fit_acceptable": False,
        "issues_found": [{
            "location": "peak region 3-5",
            "problem": "model misses the peak entirely",
            "evidence": "large systematic residual at the peak",
            "suggested_fix": "add a Gaussian component",
        }],
        "recommended_action": "add a Gaussian peak component to the model",
        "overall_assessment": "Fit inadequate.",
    }


def curve_rules(mode: str) -> list[Rule]:
    """Rules for CurveFittingAgent. mode: 'happy' | 'anneal'."""
    assert mode in ("happy", "anneal")

    if mode == "happy":
        def codegen(call_no, text):
            return {"script": curve_script("attempt-1", "Gaussian + linear background",
                                           CURVE_GOOD_R2)}
    else:
        def codegen(call_no, text):
            # Poor fits while the loop refines the existing script; the good
            # script arrives only on a FRESH regeneration after the first
            # attempt — i.e. the hot-escalation script-drop path (the prompt
            # then has no "## REFINEMENT MODE" block). This pins the
            # _just_escalated_to_hot behavior semantically, not by call count.
            if call_no == 1:
                return {"script": curve_script("attempt-1",
                                               "linear background only", 0.60)}
            if "## REFINEMENT MODE" in text:
                r2 = round(min(0.60 + 0.02 * call_no, 0.80), 2)
                return {"script": curve_script(f"attempt-{call_no}",
                                               "single Lorentzian", r2)}
            return {"script": curve_script("attempt-hot-regen",
                                           "Gaussian + linear background",
                                           CURVE_GOOD_R2)}

    def verify(call_no, text):
        if CURVE_GOOD_TOKEN in text:
            return _curve_verify_accept()
        return _curve_verify_reject()

    def refine_config(call_no, text):
        return {
            "physical_model": f"Gaussian + linear background (refined v{call_no})",
            "fitting_strategy": f"seed peak from apex, refine v{call_no}",
            "parameters_to_extract": ["amplitude", "center", "sigma"],
            "analysis_approach": "single peak on linear background",
        }

    return [
        Rule("skill_suggestion",
             "Decide which (if any) of these domain skills are relevant",
             [NO_SKILLS], repeat_last=True),
        Rule("plan_validate",
             "You are validating a curve fitting plan BEFORE it is executed.",
             [PLAN_VALID], repeat_last=True),
        Rule("planning",
             "You are an expert spectroscopist analyzing experimental data.",
             [{
                 "observations": "Single symmetric peak near x=4 on a linear background.",
                 "analysis_approach": "single peak on linear background",
                 "physical_model": "Gaussian + linear background",
                 "parameters_to_extract": ["amplitude", "center", "sigma"],
                 "fitting_strategy": "seed peak from apex, fit peak + background jointly",
                 "literature_query": None,
             }], repeat_last=True),
        Rule("conformance",
             "You are verifying that a Python script correctly implements a scientific analysis plan.",
             [CONFORMANT], repeat_last=True),
        Rule("verify",
             "## STEP 1: CHECK FOR BROKEN FITS",
             [verify], repeat_last=True),
        Rule("refine_config",
             "Refine the fitting approach based on automated verification feedback.",
             [refine_config], repeat_last=True),
        Rule("codegen",
             "Write a curve fitting script for spectroscopic data.",
             [codegen], repeat_last=True),
        Rule("synthesis",
             "You will interpret this curve fitting analysis in three stages.",
             [_synthesis("Gaussian peak")], repeat_last=True),
    ]


# ---------------------------------------------------------------------------
# Image rules
# ---------------------------------------------------------------------------

def _image_verify_accept() -> dict:
    return {
        "quality_score": 0.9,
        "is_acceptable": True,
        "result_type": "delivered",
        "completeness": 0.9,
        "correctness": 0.9,
        "relevance": 0.9,
        "issues_found": [],
        "missed_features": [],
        "false_positives": [],
        "recommended_action": "none",
        "overall_assessment": "Segmentation matches the visible blobs.",
    }


def _image_verify_reject() -> dict:
    return {
        "quality_score": 0.4,
        "is_acceptable": False,
        "result_type": "delivered",
        "completeness": 0.4,
        "correctness": 0.5,
        "relevance": 0.6,
        "issues_found": [{
            "location": "lower-right blob",
            "problem": "threshold too aggressive, blob missed",
            "evidence": "mask covers only two of three blobs",
            "suggested_fix": "lower the threshold",
        }],
        "missed_features": ["third blob"],
        "false_positives": [],
        "recommended_action": "lower the segmentation threshold",
        "overall_assessment": "Segmentation incomplete.",
    }


TIER2_SKIP = {"tier2_needed": False,
              "reasoning": "Foundational analysis is sufficient for the objective.",
              "suggested_focus": ""}


def image_rules(mode: str) -> list[Rule]:
    """Rules for ImageAnalysisAgent. mode: 'happy' | 'anneal'."""
    assert mode in ("happy", "anneal")

    if mode == "happy":
        def codegen(call_no, text):
            return {"script": image_script(IMAGE_GOOD_TAG)}
    else:
        # One counter shared by BOTH codegen rules (fresh + refinement
        # template). Refinement-template calls return poor attempts; a fresh
        # regeneration after the first call is the hot-escalation script-drop
        # path and returns the good script.
        _gen = {"n": 0}

        def codegen(call_no, text):
            _gen["n"] += 1
            if "Refine an existing image analysis script" in text:
                return {"script": image_script(f"attempt-{_gen['n']}")}
            return {"script": image_script(
                "attempt-1" if _gen["n"] == 1 else IMAGE_GOOD_TAG)}

    def verify(call_no, text):
        if IMAGE_GOOD_TAG in text:
            return _image_verify_accept()
        return _image_verify_reject()

    def refine_config(call_no, text):
        return {
            "processing_pipeline": f"gaussian blur -> adaptive threshold v{call_no} -> label -> regionprops",
            "analysis_approach": "blob segmentation with adaptive threshold",
            "features_to_extract": ["blob_count", "mask_fraction"],
            "quality_criteria": "all three blobs segmented",
        }

    return [
        Rule("skill_suggestion",
             "Decide which (if any) of these domain skills are relevant",
             [NO_SKILLS], repeat_last=True),
        Rule("plan_validate",
             "You are validating an image analysis plan BEFORE it is executed.",
             [PLAN_VALID], repeat_last=True),
        Rule("planning",
             "You are an expert image analyst working with scientific microscopy and imaging data.",
             [{
                 "observations": "Three bright blobs on a gentle gradient background.",
                 "analysis_approach": "blob segmentation with global threshold",
                 "processing_pipeline": "global threshold -> label -> regionprops",
                 "features_to_extract": ["blob_count", "mask_fraction"],
                 "quality_criteria": "all visible blobs segmented, no spurious regions",
                 "expected_outputs": ["visualization.png"],
                 "literature_query": None,
             }], repeat_last=True),
        Rule("conformance",
             "You are verifying that a Python script correctly implements a scientific image analysis plan.",
             [CONFORMANT], repeat_last=True),
        Rule("verify",
             "## SCORING RUBRIC",
             [verify], repeat_last=True),
        Rule("refine_config",
             "Refine the image analysis approach based on automated verification feedback.",
             [refine_config], repeat_last=True),
        Rule("codegen_refit",
             "Refine an existing image analysis script to match a refined plan.",
             [codegen], repeat_last=True),
        Rule("codegen",
             "Write a Python script to analyze a SINGLE image.",
             [codegen], repeat_last=True),
        Rule("tier2_decision",
             "You are evaluating whether a foundational image analysis warrants deeper follow-up analysis.",
             [TIER2_SKIP], repeat_last=True),
        Rule("synthesis",
             "Interpret these image analysis results.",
             [_synthesis("blob population")], repeat_last=True),
    ]
