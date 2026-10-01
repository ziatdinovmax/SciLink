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

import logging

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
        # In-attempt mechanical repairs of execution errors (curve/image-
        # parity inner loop) — recorded so the persisted history is honest
        # about what the attempt cost. 0 for the common clean case.
        ("exec_corrections", lambda e: e.get("exec_corrections", 0)),
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


# ---------------------------------------------------------------------------
# the verdict a caller may rely on
# ---------------------------------------------------------------------------

def _has_record(qh: Any) -> bool:
    """A real verification record, as the QC engine writes it (``approved``
    present) — not the bare ``{"produced_under_profile": …}`` stamp a series
    follower carries."""
    return isinstance(qh, dict) and "approved" in qh


def _unit_verdict(item: dict, *, where: str) -> Optional[Dict[str, Any]]:
    """Why one verified-by-record unit (a single run, a series anchor, a
    regime anchor, a refit) is NOT verified, else None."""
    qh = item.get("quality_history") or {}
    if qh.get("unverified"):
        return {"verified": False, "reason": f"verification did not finish{where}"
                + (f": {qh.get('stopped_by')}" if qh.get("stopped_by") else "")}
    if item.get("quality_warning"):
        return {"verified": False, "reason": f"salvaged best-available result{where}"}
    if item.get("judge_warning"):
        return {"verified": False, "reason": f"the judge found no acceptable fit{where}"}
    if qh.get("verifier_rejected"):
        return {"verified": False, "reason": f"the verifier still rejected the result at the cap{where}"}
    if not qh.get("approved"):
        # a locked replay is judged by the replay gate, not a verifier (the
        # image agent stamps approved_by only when the gate passed)
        replay = qh.get("approved_by") == "replay_gate" or (item.get("reuse_validity") or {}).get("reused")
        return {"verified": False, "reason": (f"the replay gate rejected the result{where}" if replay
                                              else f"the verifier did not approve the result{where}")}
    # A verifier may approve a fit below the numeric threshold on physics
    # grounds (inside the gate's soft band): that is the pipeline's gate and
    # it counts. A run whose verification was bypassed
    # (max_verification_iterations=0: the image agent stamps "bypass", the
    # curve agent "verifier" with no iteration) passed no gate unless the
    # metric itself met the threshold.
    metric = next((qh.get(k) for k in ("final_r2", "final_score", "final_passed_fraction")
                   if isinstance(qh.get(k), (int, float))), None)
    no_pass = qh.get("approved_by") == "bypass" or (
        qh.get("approved_by") == "verifier" and not qh.get("verification_iterations"))
    if no_pass and metric is not None and qh.get("threshold") is not None and metric < qh["threshold"]:
        return {"verified": False, "reason": f"verification bypassed and the metric is below its threshold{where}"}
    return None


def unit_verdict_for(unit: Dict[str, Any], *, recipe: Optional[Dict[str, Any]] = None,
                     regime: Optional[str] = None) -> Dict[str, Any]:
    """The verdict on ONE series unit, stamped by the series driver at the
    moment it knows everything — the gate result of a unit it verified, or
    the recipe a follower replayed and that recipe's verdict. The driver
    writes it as ``unit["unit_verdict"]``; ``analysis_verdict`` then only
    aggregates, and no later refit of another unit can change it.

    ``recipe`` is the regime anchor's record at the time the follower ran:
    ``{"unit": name, "verdict": {verified, reason}}`` or None when the
    follower was fitted with no base script.
    """
    name = unit.get("name") or unit.get("index")
    if not unit.get("success"):
        return {"verified": False, "reason": f"unit failed: {str(unit.get('error') or '')[:120]}",
                "regime": regime}
    if _has_record(unit.get("quality_history")):
        # verified by its own gate: an anchor, a regime anchor, a refit
        bad = _unit_verdict(unit, where="")
        return {"verified": bad is None, "reason": (bad or {}).get("reason") or "approved by its own gate",
                "regime": regime, "own_gate": True}
    if (unit.get("quality_history") or {}).get("unverified"):
        return {"verified": False, "reason": "follower unverified (budget)", "regime": regime}
    if unit.get("fitted_from") == "fresh_code" or recipe is None:
        return {"verified": False, "reason": "fitted without a locked recipe (its regime's anchor "
                                             "produced none)", "regime": regime}
    rv = recipe.get("verdict") or {}
    if rv.get("verified"):
        return {"verified": True, "reason": f"replayed the locked recipe of unit {recipe.get('unit')}, "
                                            "whose gate passed", "regime": regime, "recipe_of": recipe.get("unit")}
    return {"verified": False, "reason": f"replayed the locked recipe of unit {recipe.get('unit')}, which was "
                                         f"not approved: {rv.get('reason')}", "regime": regime,
            "recipe_of": recipe.get("unit")}


def stamp_unit_verdict(unit: Dict[str, Any], *, recipe: Optional[Dict[str, Any]] = None,
                       regime: Optional[str] = None) -> None:
    """``unit["unit_verdict"] = unit_verdict_for(...)``, never raising: a
    verdict that cannot be formed leaves the unit unstamped (the aggregator
    then falls back to its reconstruction) and must not fail a fit."""
    try:
        unit["unit_verdict"] = unit_verdict_for(unit, recipe=recipe, regime=regime)
    except Exception as exc:  # noqa: BLE001 - a stamp is a side note on a fit
        logging.getLogger(__name__).warning(f"unit verdict not stamped: {exc}")


def analysis_verdict(full_result: Optional[dict]) -> Dict[str, Any]:
    """Did this analysis pass its own pipeline's checks?

    A run's ``status`` says whether it produced a result, not whether the
    result was approved: the curve and image agents return ``success`` for a
    salvaged best-available fit (``quality_warning``), for a run whose
    verification did not finish (``quality_history.unverified``), for a
    result the verifier never approved (``approved`` false) and for one the
    judge picked as best available (``judge_warning``); hyperspectral reports
    ``partial``. This reads those signals in the shapes the agents write and
    gives one answer with the reason — for the board, which posts only what
    an agent verified as verified, and for any caller that must not mistake
    a produced result for an approved one.

    Shapes:

    - a single curve or image run: the top-level ``quality_history`` (and
      ``quality_warning`` / ``judge_warning``);
    - a curve or image series (``individual_results``): the units that went
      through the QC engine — the anchor, a regime anchor, a refit — carry a
      verification record and must be approved with no salvage marker; the
      followers (a locked replay, no record beyond the profile stamp) must
      have succeeded and not be ``unverified``. A unit that FAILED is not in
      the feature table (the agent flags it and says so), so it does not
      block; a unit that succeeded unverified IS in the table, so it does.
    - a hyperspectral cube (``dynamic_analysis_records``): ``success``
      status, and every target that produced a script approved
      (``task_success``), none salvaged;
    - a hyperspectral series: every successful row's own ``verified``.
    """
    full = full_result or {}
    status = full.get("status")
    if status != "success":
        return {"verified": False, "reason": f"status {status!r}"}
    rv = full.get("reuse_validity") or {}
    if rv.get("reused") and rv.get("verdict") not in (None, "good"):
        return {"verified": False, "reason": f"reused script verdict {rv.get('verdict')!r}"}

    items = full.get("individual_results")
    hs_records = full.get("dynamic_analysis_records")

    if isinstance(items, list) and items:
        ok_items = [it for it in items if isinstance(it, dict) and it.get("success")]
        if not ok_items:
            return {"verified": False, "reason": "no unit succeeded"}
        failed = len(items) - len(ok_items)
        if all(isinstance(it.get("unit_verdict"), dict) for it in ok_items):
            # The driver stamped every unit when it knew the recipe and the
            # gate result (unit_verdict_for): aggregate, nothing to reconstruct.
            for it in ok_items:
                uv = it["unit_verdict"]
                if not uv.get("verified"):
                    return {"verified": False,
                            "reason": f"{uv.get('reason')} (unit {it.get('name') or it.get('index')})"}
            return {"verified": True, "reason": "every unit verified by the series driver"
                    + (f" ({failed} failed unit(s) excluded by the agent)" if failed else "")}
        # A shape from before the stamp: reconstruct from the markers.
        # The recipe a follower replayed is its regime's anchor AS IT WAS when
        # the follower ran. A refit anchor carries a summary of the unit it
        # replaced (``replaced_unit``); a follower still on the locked script
        # is judged by that, never by a refit it never re-ran.
        recipe_by_regime: Dict[Any, Dict[str, Any]] = {}
        for it in ok_items:
            if it.get("role") == "anchor" and it.get("adaptively_refitted") and it.get("replaced_unit"):
                recipe_by_regime[it.get("regime")] = {**it["replaced_unit"], "role": "anchor"}
        anchors = 0
        for it in ok_items:
            name = it.get("name") or it.get("index")
            where = f" (unit {name})"
            if "verified" in it and "quality_history" not in it:
                # a hyperspectral series row: the driver's own verdict, held
                # to the single-cube rule — status "success" (a salvaged or
                # degraded cube is "partial"), something extracted, and
                # EVERY target approved (the row's own n_approved == n_targets;
                # the driver's `verified` asks for one)
                qm = it.get("quality_metrics") or {}
                n_t, n_a = qm.get("n_targets"), qm.get("n_approved")
                if (it.get("verified") is False or it.get("status") != "success"
                        or not it.get("n_features")):
                    return {"verified": False, "reason": f"unit not verified by the series driver{where}"}
                if not (isinstance(n_t, int) and isinstance(n_a, int) and n_t > 0 and n_a == n_t):
                    if not n_t:
                        return {"verified": False, "reason": f"the unit has no dynamic-analysis record{where}"}
                    return {"verified": False, "reason": f"not every target of the unit was approved{where}: "
                                                          f"{n_a} of {n_t}"}
                anchors += 1 if it.get("role") == "anchor" else 0
                continue
            rv = it.get("reuse_validity") or {}
            if rv.get("reused") and not _has_record(it.get("quality_history")):
                # a reused anchor: the replay gate is its verification
                anchors += 1
                if rv.get("verdict") != "good":
                    return {"verified": False, "reason": f"reused script verdict {rv.get('verdict')!r}{where}"}
                continue
            if _has_record(it.get("quality_history")):
                if it.get("adaptively_refitted") and it.get("role") != "anchor":
                    # a FOLLOWER refit the series driver accepted by its
                    # consistency rule: held like a follower (finished, not
                    # unverified), not to the anchor's salvage markers —
                    # otherwise a refit that improved a unit could unverify a
                    # series the unrefit unit would have passed. An anchor
                    # keeps its role through a refit and keeps the anchor's
                    # bar: a salvaged anchor is not laundered by refitting it.
                    if (it.get("quality_history") or {}).get("unverified"):
                        return {"verified": False, "reason": f"refit unverified{where}"}
                    continue
                anchors += 1
                bad = _unit_verdict(it, where=where)
                if bad:
                    return bad
            else:
                if it.get("fitted_from") == "fresh_code":
                    # its regime's anchor failed: fitted from scratch, no verifier
                    return {"verified": False, "reason": f"follower fitted without a locked recipe{where}"}
                if (it.get("quality_history") or {}).get("unverified"):
                    return {"verified": False, "reason": f"follower unverified{where}"}
                recipe = recipe_by_regime.get(it.get("regime"))
                if recipe is not None:
                    bad = _unit_verdict(recipe, where="")
                    if bad:
                        return {"verified": False, "reason": (
                            f"follower replays a recipe that was not approved{where}: "
                            f"{bad['reason']} (recipe from unit {recipe.get('name')}, since refit)")}
        if anchors == 0 and not any("verified" in it for it in ok_items):
            return {"verified": False, "reason": "no unit carries a verification record"}
        return {"verified": True, "reason": "series anchors approved and every follower verified"
                + (f" ({failed} failed unit(s) excluded by the agent)" if failed else "")}

    if isinstance(hs_records, list) and hs_records:
        scripted = 0
        for r in hs_records:
            if not isinstance(r, dict):
                continue
            where = f" (target {r.get('target')})"
            if r.get("not_measurable") and not r.get("task_success"):
                continue                      # answered through the honest channel
            if r.get("salvaged"):
                return {"verified": False, "reason": f"salvaged target{where}"}
            if not r.get("script") and not r.get("task_success"):
                return {"verified": False, "reason": f"target failed before any code ran{where}"}
            scripted += 1
            if not r.get("task_success") or not (r.get("quality_history") or {}).get("approved", True):
                return {"verified": False, "reason": f"target did not pass verification{where}"}
        if not scripted:
            return {"verified": False, "reason": "no target produced an approved script"}
        return {"verified": True, "reason": "every target passed verification"}

    if full.get("quality_warning") and not _has_record(full.get("quality_history")):
        return {"verified": False, "reason": "salvaged best-available result (quality_warning)"}
    qh = full.get("quality_history")
    if not _has_record(qh):
        if rv.get("reused") and rv.get("verdict") == "good":
            return {"verified": True, "reason": "locked-script reuse passed the replay gate"}
        return {"verified": False, "reason": "no verification record"}
    bad = _unit_verdict(full, where="")
    if bad:
        return bad
    return {"verified": True, "reason": "approved by the analysis verifier"
            if qh.get("verification_iterations") else "met the acceptance threshold"}


def series_anchor_unit(full_result: Optional[dict]) -> Optional[str]:
    """The name of a series' anchor unit — the first successful unit that
    went through the QC engine and is not a refit — whose ``scripts/<name>.py``
    is the locked recipe. None for a single run."""
    items = (full_result or {}).get("individual_results")
    if not isinstance(items, list) or not items:
        return None
    roled = [it for it in items if isinstance(it, dict) and it.get("success") and it.get("name")
             and it.get("role") == "anchor"]
    if roled:
        return str(roled[0]["name"])          # the first anchor, refit or not
    for it in items:                          # a shape from before the role stamp
        if not (isinstance(it, dict) and it.get("success") and it.get("name")) or it.get("adaptively_refitted"):
            continue
        if _has_record(it.get("quality_history")) or (it.get("reuse_validity") or {}).get("reused"):
            return str(it["name"])
    return None

