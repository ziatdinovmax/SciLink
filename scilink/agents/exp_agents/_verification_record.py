"""Shared verification-record builders for the analysis QC loops.

One implementation of the two structures the curve-fitting and image-analysis
controllers used to duplicate (~90% isomorphic twins):

- ``build_quality_history`` — the per-item ``quality_history`` dict attached
  to every result record (consumed by T=2 staging, the orchestrator, and the
  self-evolution tooling);
- ``build_verification_prompt_history`` — the "PREVIOUS VERIFICATION
  ATTEMPTS" context block injected into the verifier prompt.

Modality differences are captured in a *keymap* (``CURVE_HISTORY_KEYMAP`` /
``IMAGE_HISTORY_KEYMAP``) so each controller keeps emitting **exactly** its
historical output — key names, key order, formatting, and edge-case semantics
(e.g. image's ``entry["quality_score"]`` hard-KeyError on a malformed history
entry) are all preserved. The golden suite (``tests/qc_golden``) pins this.

Part of the Layer-0 types of the QC unification (issue #327,
``analysis_qc_unification_plan.md`` §2.2). Hyperspectral adopts the same
builder in the HS-1 expansion via its own keymap.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def _issues(entry: dict) -> list:
    return [
        {
            "location": iss.get("location", ""),
            "problem": iss.get("problem", ""),
        }
        for iss in entry.get("issues_found", [])
    ]


# ---------------------------------------------------------------------------
# quality_history
# ---------------------------------------------------------------------------

CURVE_HISTORY_KEYMAP: Dict[str, Any] = {
    "final_key": "final_r2",
    # (output key, extractor) in the exact historical insertion order.
    "iteration_fields": [
        ("r_squared", lambda e: e.get("r_squared")),
        ("annealing_level", lambda e: e.get("annealing_level", 0)),
        ("tools_used", lambda e: e.get("tools_used", [])),
        # The model in force at this iteration — lets a consumer (e.g. the
        # self-evolution figure) show the per-attempt model alongside its
        # R² and issues.
        ("model", lambda e: (e.get("config_used") or {}).get("physical_model", "")),
        ("issues", _issues),
        ("fix_applied", lambda e: e.get("recommended_action", "")),
    ],
    "include_alternative_models": True,
    "include_score_explanation": False,
}

# Hyperspectral dynamic-analysis (per-target codegen retry loop). The metric
# is the fraction of output maps that passed QC on an attempt; the annealing
# level maps the 3-stage retry ladder (patch → question method → abandon
# method family) onto the shared 0/1/2 scale. ``approved`` is overwritten by
# the caller with the task-success verdict (fraction + required-outputs gate),
# mirroring curve's verifier-approved overwrite.
HS_HISTORY_KEYMAP: Dict[str, Any] = {
    "final_key": "final_passed_fraction",
    "iteration_fields": [
        ("passed_fraction", lambda e: e.get("passed_fraction")),
        ("annealing_level", lambda e: e.get("annealing_level", 0)),
        ("issues", _issues),
        ("fix_applied", lambda e: e.get("recommended_action", "")),
    ],
    "include_alternative_models": False,
    "include_score_explanation": False,
}

IMAGE_HISTORY_KEYMAP: Dict[str, Any] = {
    "final_key": "final_score",
    "iteration_fields": [
        # Deliberate hard indexing: a history entry without quality_score is
        # malformed and has always raised — kept as-is.
        ("score", lambda e: e["quality_score"]),
        ("result_type", lambda e: e.get("result_type")),
        ("annealing_level", lambda e: e.get("annealing_level", 0)),
        ("issues", _issues),
        ("tools_used", lambda e: e.get("tools_used", [])),
        ("fix_applied", lambda e: e.get("recommended_action", "")),
    ],
    "include_alternative_models": False,
    "include_score_explanation": True,
}


def build_quality_history(
    *,
    best_value: float,
    threshold: float,
    all_attempts: Optional[list],
    verification_history: list,
    judge_result: Optional[dict],
    script_errors: Optional[list],
    keymap: Dict[str, Any],
) -> dict:
    """Build the per-item ``quality_history`` dict.

    Captures problem→solution pairs at every level: script errors,
    verification iterations, alternative approaches, and judge reasoning.
    """
    history: dict = {
        keymap["final_key"]: best_value,
        "threshold": threshold,
        "approved": best_value >= threshold,
        "verification_iterations": [
            {name: extract(entry) for name, extract in keymap["iteration_fields"]}
            for entry in verification_history
        ],
    }
    if keymap.get("include_alternative_models"):
        history["alternative_models"] = [
            {
                "model": a.get("model", ""),
                "r2": a.get("r2", 0),
                "diagnosis": a.get("diagnosis", ""),
            }
            for a in (all_attempts or [])[1:]
            if not str(a.get("model", "")).startswith("Verification")
        ]
    history["script_errors"] = script_errors or []
    history["judge_reasoning"] = (judge_result or {}).get("reasoning")
    if keymap.get("include_score_explanation"):
        history["score_explanation"] = (judge_result or {}).get("score_explanation")
    return history


# ---------------------------------------------------------------------------
# verification-prompt history block
# ---------------------------------------------------------------------------

def _curve_metric_lines(prev: dict) -> List[str]:
    label = prev.get("metric_label", "R²")
    mv = prev.get("metric_value", prev.get("r_squared"))
    bm = prev.get("best_metric_value", prev.get("best_so_far"))
    parts = [f"{label} = {mv:.4f}" if mv is not None else f"{label} = N/A"]
    if bm is not None:
        parts.append(f"best-so-far = {bm:.4f}")
    return ["- " + " | ".join(parts)]


def _image_metric_lines(prev: dict) -> List[str]:
    score = prev.get("quality_score")
    return [f"- Quality score = {score:.2f}" if score is not None
            else "- Quality score = N/A"]


CURVE_PROMPT_KEYMAP: Dict[str, Any] = {
    "metric_lines": _curve_metric_lines,
    "config_line": lambda prev: (
        f"- Config: {prev.get('config_used', {}).get('physical_model', 'N/A')}"
    ),
    "issue_bullet": "•",
    "rule_2": (
        "2. If a fix didn't work AND the best metric is still below the accept "
        "threshold, suggest something DIFFERENT. But if the best is already above "
        "the accept threshold and has not improved for the last 2 iterations, do "
        "NOT propose another change — accept and record any remaining concern as a "
        "caveat (the plateau/convergence rule takes precedence)."
    ),
    "rule_4_evidence": "the plot",
}

IMAGE_PROMPT_KEYMAP: Dict[str, Any] = {
    "metric_lines": _image_metric_lines,
    "config_line": lambda prev: (
        f"- Pipeline: {prev.get('config_used', {}).get('processing_pipeline', 'N/A')}"
    ),
    "issue_bullet": "-",
    "rule_2": "2. If a fix didn't work, suggest something DIFFERENT",
    "rule_4_evidence": "the images",
}


def build_verification_prompt_history(
    previous_iterations: List[dict],
    keymap: Dict[str, Any],
) -> str:
    """Build the "PREVIOUS VERIFICATION ATTEMPTS" verifier-prompt block."""
    if not previous_iterations:
        return ""

    lines = [
        "\n\n## PREVIOUS VERIFICATION ATTEMPTS",
        "Review what was tried before. Don't suggest fixes that already failed.\n"
    ]

    metric_lines: Callable[[dict], List[str]] = keymap["metric_lines"]
    config_line: Callable[[dict], str] = keymap["config_line"]
    bullet = keymap["issue_bullet"]

    for i, prev in enumerate(previous_iterations, 1):
        lines.append(f"\n### Attempt {i}")
        lines.extend(metric_lines(prev))
        lines.append(config_line(prev))
        lines.append(f"- Assessment: {prev.get('overall_assessment', 'N/A')}")

        issues = prev.get('issues_found', [])
        if issues:
            lines.append(f"- Issues ({len(issues)}):")
            for issue in issues:
                lines.append(f"  {bullet} {issue.get('location', '?')}: {issue.get('problem', '?')}")

        if prev.get('recommended_action'):
            lines.append(f"- Action taken: {prev['recommended_action']}")

        if prev.get('refinement_error'):
            lines.append(
                f"- **NOTE: The recommended fix was NOT applied** because "
                f"the refinement LLM call failed ({prev['refinement_error']}). "
                f"The results below are UNCHANGED from this attempt — "
                f"do not penalize for identical output. Re-evaluate the "
                f"recommended action and suggest concrete fixes."
            )

    lines.extend([
        "\n\n## IMPORTANT",
        "1. Check if previous issues were RESOLVED or still PERSIST",
        keymap["rule_2"],
        "3. If a previous fix was NOT applied due to an API error, "
        "re-suggest it or propose an alternative",
        "4. A previously-raised issue may have been MISTAKEN. RETRACT it (drop it; "
        "stop demanding fixes) when STRONG evidence shows the concern was unfounded "
        f"— {keymap['rule_4_evidence']}, a registered tool's documented behaviour/guarantees, clear "
        "physics, or an independent cross-check. Absent strong evidence, keep "
        "scrutinizing: 'persists' means still demonstrably real, not merely "
        "un-disproven.",
    ])

    return "\n".join(lines)
