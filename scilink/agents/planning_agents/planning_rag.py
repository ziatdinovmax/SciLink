import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    from PIL import Image as _PIL_Image
except Exception:  # Pillow optional — critic degrades to text-only evidence
    _PIL_Image = None

from scilink.parsers import parse_adaptive_excel
from scilink.knowledge import run_rag, parse_json_from_response
from .parser_utils import plan_is_portfolio
from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS,
    HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK,
    TEA_INSTRUCTIONS_FALLBACK,
    HYPOTHESIS_DISTINCTNESS_CONDITIONING,
    HYPOTHESIS_BEST_OF_N_SELECTION_INSTRUCTIONS,
    HYPOTHESIS_BESTOFN_AUTHOR_NOTE,
    BESTOFN_SELECTION_PROFILE_LAB,
    BESTOFN_SELECTION_PROFILE_IDEATION,
    IDEATION_AUTHOR_OVERRIDE,
    IDEATION_OUTPUT_RULES,
    TECHNICAL_DOCUMENT_INSTRUCTIONS,
    IDEATION_PORTFOLIO_INSTRUCTIONS,
    PORTFOLIO_REFINEMENT_RULES,
    IDEATION_PORTFOLIO_INSTRUCTIONS_FALLBACK,
    TECHNICAL_DOCUMENT_INSTRUCTIONS_FALLBACK,
    TECHNICAL_DOCUMENT_REVISION_RULES,
    CONSTRAINT_COVERAGE_NOTE
)


def summarize_experiment(exp: Dict[str, Any], index: int) -> str:
    """One experiment rendered for the reasoning passes (conformance, critic).

    Enumerates a `concepts` portfolio when the plan carries one. Without
    this, a plan holding N research directions is summarized by its single
    experiment entry, so the conformance check cannot see that a
    "propose 4-6 directions" objective was in fact satisfied — the failure
    that pushed portfolios into `experimental_steps` in the first place.
    Concept bodies are clipped: the passes need coverage and identity, not
    the full text.
    """
    lines = [f"Experiment {index}: {exp.get('experiment_name', 'N/A')}",
             f"  Hypothesis: {exp.get('hypothesis', 'N/A')}",
             f"  Justification: "
             f"{exp.get('justification', 'No justification provided.')}"]
    concepts = exp.get("concepts")
    if isinstance(concepts, list) and concepts:
        lines.append(f"  Research directions in this plan "
                     f"({len(concepts)} total):")
        for c in concepts:
            if not isinstance(c, dict):
                lines.append(f"    - {str(c)[:200]}")
                continue
            label = " ".join(str(c.get(k)) for k in ("id", "tier")
                             if c.get(k))
            head = f"[{label}] " if label else ""
            lines.append(f"    - {head}{str(c.get('title', 'Untitled'))[:160]}")
            for key in ("hypothesis", "novelty"):
                if c.get(key):
                    lines.append(f"        {key}: {str(c[key])[:300]}")
    lines.append("---")
    return "\n".join(lines)


# The critic's dimensions. Each is a lens the reviewer of a proposal actually
# applies; findings carry one of these as ``dimension`` and are rendered
# generically downstream (format_caveats), so adding a lens is prompt-only.
CRITIC_DIMENSIONS = ("physics", "consistency", "design", "statistics",
                     "method", "evidence")

_CRITIC_DIRECTION_CLIP = 2500   # chars of `details` per direction shown to the critic


def summarize_plan_for_critic(result: Dict[str, Any]) -> str:
    """The plan as the CRITIC should see it — everything the author decided.

    ``summarize_experiment`` (used by the conformance pass) is deliberately a
    coverage-and-identity view: it clips a portfolio direction to title /
    hypothesis / novelty. That view hides exactly what a reviewer checks —
    arms and replicate counts, control construction, the continuity mode,
    the named support, the estimator's reference — which for portfolios
    lives in ``directions[*].details`` and ``shared_protocol``. Live: an
    "every policy paired with a yoked control" + "5 policies × 6–8
    trajectories" arithmetic contradiction and a "quenched coupons" Track-2
    mode both sat in ``details`` and never reached the critic.

    Returns the conformance summary plus, when present, each direction's
    ``details`` (clipped per direction, marked), the ``shared_protocol``, and
    the author's ``open_questions`` (so the critic does not re-raise what the
    author already flagged, and can judge whether a flagged risk is framed
    the right way round).
    """
    experiments = result.get("proposed_experiments", []) or []
    parts = [summarize_experiment(exp, i + 1) for i, exp in enumerate(experiments)]
    directions = result.get("directions")
    if isinstance(directions, list) and directions:
        parts.append("DIRECTION DETAILS (the author's full design per direction):")
        for d in directions:
            if not isinstance(d, dict):
                continue
            head = " ".join(str(d.get(k)) for k in ("id", "tier") if d.get(k))
            parts.append(f"[{head}] {str(d.get('title', 'Untitled'))[:200]}")
            det = d.get("details")
            if isinstance(det, list):
                det = "\n".join(f"  - {str(x)}" for x in det)
            det = str(det or "").strip()
            if det:
                if len(det) > _CRITIC_DIRECTION_CLIP:
                    det = det[:_CRITIC_DIRECTION_CLIP] + "\n  [clipped]"
                parts.append(det)
    sp = result.get("shared_protocol")
    if sp:
        body = "\n".join(f"  - {x}" for x in sp) if isinstance(sp, list) else str(sp)
        parts.append("SHARED PROTOCOL (applies to every direction):\n" + body)
    oq = result.get("open_questions")
    if oq:
        body = "\n".join(f"  - {x}" for x in oq) if isinstance(oq, list) else str(oq)
        parts.append("OPEN QUESTIONS THE AUTHOR ALREADY RAISED:\n" + body)
    return "\n".join(parts)


def verify_plan_relevance(objective: str,
                          result: Dict[str, Any],
                          model: Any,
                          generation_config: Any) -> Tuple[bool, str]:
    """
    Objective-conformance check (the enforcing self-reflection step). Returns
    (True, "") if the plan conforms, or (False, "Reason") if not — a False
    triggers an automatic plan adjustment in the caller.

    Logic:
    1. Checks if the plan was generated via Fallback (General Knowledge).
    2. If Fallback: Verifies only scientific soundness (Relaxed).
    3. If Strict: Verifies document grounding and specific constraint adherence (Strict).

    This is intentionally scoped to relevance / objective-conformance only.
    Physical realism and internal consistency are handled separately and
    advisorily by ``critique_plan`` (run after this check + any adjustment).
    """
    experiments = result.get("proposed_experiments", [])
    if not experiments:
        return False, "No experiments generated."

    # 1. Detect Fallback Mode
    # We check if ANY experiment contains the mandatory fallback warning defined in instruct.py
    is_fallback = False
    for exp in experiments:
        justification = exp.get('justification', '').lower()
        if "general scientific knowledge" in justification or "documents lacked specific context" in justification:
            is_fallback = True
            break

    # 2. Build Plan Summary for the Verifier
    plan_summary = "\n".join(summarize_experiment(exp, i + 1)
                             for i, exp in enumerate(experiments))

    # 3. Construct Context-Aware Prompt
    if is_fallback:
        print("    - ℹ️  Verifying Fallback Plan (Relaxed Constraints)...")
        eval_prompt = f"""
        You are a scientific research evaluator.

        **CONTEXT:** The system failed to find specific documents for the User Objective in the Knowledge Base.
        Therefore, it generated a plan based on **General Scientific Knowledge**.

        1. User Objective: "{objective}"
        2. Proposed Plan (General Knowledge):
        {plan_summary}

        **TASK:**
        Determine if the Proposed Plan makes scientific sense for the Objective, acknowledging that it CANNOT cite specific documents.

        **CRITERIA FOR PASS:**
        - The plan addresses the objective using standard, correct scientific principles.
        - The logic is sound and actionable.
        - **DO NOT FAIL** the plan simply because it uses general knowledge or lacks specific context (this is expected in fallback mode).

        **Output:**
        Respond with a single JSON object: {{ "is_relevant": boolean, "reason": "string explanation" }}
        """
    else:
        print("    - ℹ️  Verifying Strict Plan (Document Constraints)...")
        eval_prompt = f"""
        You are a scientific research evaluator.

        1. User Objective: "{objective}"
        2. Proposed Plan:
        {plan_summary}

        **TASK:**
        Review the "Hypothesis" and "Justification" for each experiment.
        Determine if the Proposed Plan is directly relevant to the User Objective AND supported by the cited context.

        **CRITERIA FOR FAIL:**
        - The plan ignores specific constraints in the objective (e.g., "Use X method" but the plan uses "Y").
        - The justification contradicts the hypothesis.
        - The plan is logically incoherent.

        **Output:**
        Respond with a single JSON object: {{ "is_relevant": boolean, "reason": "string explanation" }}
        """

    # 4. Execute Verification
    try:
        response = model.generate_content([eval_prompt], generation_config=generation_config)
        eval_result, _ = parse_json_from_response(response)

        if eval_result and not eval_result.get("is_relevant"):
            reason = eval_result.get('reason', 'Unknown irrelevance.')
            print(f"    - ⚠️  Plan Verification Failed: {reason}")
            return False, reason

        print(f"    - ✅ Plan Verification Passed.")
        return True, ""

    except Exception as e:
        logging.error(f"Verification step failed: {e}")
        # Fail open: If the verifier crashes, we assume the plan is okay to avoid blocking the user.
        return True, ""


def critique_plan(objective: str,
                  result: Dict[str, Any],
                  model: Any,
                  generation_config: Any,
                  retrieved_context: Optional[str] = None,
                  primary_data: Optional[str] = None,
                  images: Optional[List[Any]] = None,
                  image_descriptions: Optional[List[str]] = None,
                  additional_context: Optional[str] = None,
                  skill_context: Optional[str] = None,
                  prior_plan: Optional[Dict[str, Any]] = None,
                  prior_findings: Optional[List[Dict[str, Any]]] = None,
                  human_feedback: Optional[str] = None) -> Dict[str, Any]:
    """
    Advisory critic for PHYSICAL REALISM and INTERNAL CONSISTENCY — a separate
    LLM call run AFTER ``verify_plan_relevance`` (and any conformance-driven
    adjustment), on the final plan. It NEVER rewrites the plan; it returns
    caveats for a human / consumer to weigh.

    Objective-relevance and document grounding are NOT re-litigated here — that
    is ``verify_plan_relevance``'s enforcing job. This critic only flags physics
    and consistency caveats.

    Returns {"findings": [{"dimension","severity","experiment","issue"}, ...]}
    with ``dimension`` in {physics, consistency}. Empty findings == clean. Fails
    open ({"findings": []}) on error so a critic crash never blocks the user.

    Optional evidence args mirror what the plan author saw, so the critic checks
    against the same material rather than reasoning in a vacuum (the robustness
    ingredient borrowed from the hyperspectral reviewer, which reads its plots):
    ``retrieved_context``, ``primary_data``, ``images``/``image_descriptions``,
    ``additional_context``, ``skill_context``. All optional.

    Revision context (all optional, for re-critiquing a just-revised plan): pass
    ``prior_plan`` (the version before the edit), ``prior_findings`` (the caveats
    it carried), and ``human_feedback`` (what the human asked for). The critic is
    handed the full picture and reasons about the CURRENT plan itself — the human
    may have addressed the caveats, changed something unrelated, or deliberately
    overridden the critic, so nothing is assumed: prior caveats that no longer
    apply are dropped, ones that still apply are kept, and new issues introduced
    by the revision are flagged.
    """
    experiments = result.get("proposed_experiments", [])
    if not experiments:
        return {"findings": []}

    plan_summary = summarize_plan_for_critic(result)

    # --- Evidence: mirror what the plan author saw so checks are grounded. ---
    evidence_parts = []
    if primary_data:
        evidence_parts.append(f"## 📊 Primary Experimental Data:\n{primary_data}")
    if additional_context:
        evidence_parts.append(f"## Additional Context:\n{additional_context}")
    if retrieved_context:
        evidence_parts.append(f"## Retrieved Context (KB + literature):\n{retrieved_context}")
    if skill_context:
        evidence_parts.append(skill_context)

    loaded_images = []
    if images and _PIL_Image:
        for img in images:
            if isinstance(img, str):
                try:
                    loaded_images.append(_PIL_Image.open(img))
                except Exception as e:
                    print(f"    - ⚠️ Critic could not load image {img}: {e}")
            else:
                loaded_images.append(img)  # assume already a PIL image
    if loaded_images:
        note = "## Provided Images: (See attached)"
        if image_descriptions:
            note += f"\n## Image Descriptions:\n{json.dumps(image_descriptions, indent=2)}"
        evidence_parts.append(note)

    if evidence_parts:
        evidence_section = (
            "\n────────────────────────────────────\n"
            "EVIDENCE THE PLAN AUTHOR USED — anchor your physics checks to the\n"
            "data values here, and read it for stated limitations, artifacts or\n"
            "instabilities of the techniques and materials the plan names:\n"
            + "\n\n".join(evidence_parts) + "\n"
        )
    else:
        evidence_section = ""

    # Revision context: when the CURRENT plan is a revision, hand the critic the
    # before/criticism/request/after picture and let it reason about the current
    # plan — don't presume the human acted on the caveats (they may have changed
    # something unrelated or overridden the critic entirely).
    revision_parts = []
    if prior_plan and prior_plan.get("proposed_experiments"):
        prior_summary = "\n".join(
            f"Experiment {i+1}: {e.get('experiment_name', 'N/A')}\n"
            f"  Hypothesis: {e.get('hypothesis', 'N/A')}\n"
            f"  Justification: {e.get('justification', 'No justification provided.')}\n---"
            for i, e in enumerate(prior_plan.get("proposed_experiments", []))
        )
        revision_parts.append("PRIOR PLAN (before this revision):\n" + prior_summary)
    if prior_findings:
        pl = "\n".join(
            f"  - [{f.get('dimension')}/{f.get('severity')}] {f.get('issue')}"
            for f in prior_findings if f.get("issue")
        )
        if pl:
            revision_parts.append("CAVEATS PREVIOUSLY RAISED on the prior plan:\n" + pl)
    if human_feedback:
        revision_parts.append(f'REVISION REQUEST:\n"{human_feedback}"')

    if revision_parts:
        prior_section = (
            "\n────────────────────────────────────\n"
            "REVISION CONTEXT — the CURRENT plan above is a revision of an earlier one.\n"
            "Using the prior plan, its caveats, and the human's request below, judge the\n"
            "CURRENT plan:\n"
            "  • a prior caveat the revision RESOLVED -> drop it.\n"
            "  • a prior caveat that still applies and was left unaddressed -> report it.\n"
            "  • a prior caveat the human EXPLICITLY ACCEPTED as a tradeoff -> RETAIN it\n"
            "    but mark it accepted: phrase the issue as an accepted limitation and set\n"
            "    its severity to 'minor' (documented for the record, not a blocker).\n"
            "  • any NEW physics/consistency issue the revision introduced -> report it.\n\n"
            + "\n\n".join(revision_parts) + "\n"
        )
    else:
        prior_section = ""

    eval_prompt = f"""
You are reviewing an experimental plan the way an expert referee reviews a proposal:
for physical realism, internal consistency, design and statistical rigor, and method
feasibility. A separate check has already confirmed the plan is relevant to the
objective, so do NOT re-litigate objective-relevance or document grounding here.

OBJECTIVE: "{objective}"

PROPOSED PLAN:
{plan_summary}
{evidence_section}{prior_section}
Assume the plan may be flawed and try to break it on these axes:
  • physics — parameters or conditions that are physically impossible or implausible;
              a technique that cannot measure the stated quantity or resolve the
              claimed scale; violated conservation laws or instrument limits.
  • consistency — a justification that contradicts its own hypothesis; steps that do
              not actually test the stated hypothesis; equipment that does not match
              the steps; two experiments that contradict each other.
  • design — the arms, controls, replicates and totals must add up and every control
              must be constructible; each gate must protect the claim it is said to
              protect; a confirmatory comparison must be fixed before its data are
              taken — no unresolved either/or choices of material, support or
              endpoint — and not adapted while under test.
  • statistics — the independent experimental unit must be named; nested repeats on
              one sample are not replicates; power is estimated from a pilot, never
              asserted from a replicate count.
  • method — each technique must be able to operate under the stated conditions and
              measure the stated quantity; a calibration reference must be independent
              of the estimator's inputs; the endpoint must be a named quantity with one
              primary normalization, with transient or inventory contributions excluded.
  • evidence — a limitation, artifact or instability of a named technique or material
              that the evidence above states and the plan relies on without addressing.
Work axis by axis: for EACH axis, name the single most material flaw you can find on
it (or nothing if that axis is clean), then add further findings on any axis. Report
only a flaw you can name concretely. Do NOT invent problems to appear thorough. Do not
repeat a risk the author already raised under OPEN QUESTIONS unless it is framed the
wrong way round. Report at most 10 findings, most material first.

SEVERITY:
  • critical — would make the plan infeasible or scientifically wrong.
  • minor    — worth noting, but the plan still stands.

OUTPUT — a single JSON object:
{{"findings": [
   {{"dimension": "physics|consistency|design|statistics|method|evidence",
     "severity": "critical|minor",
     "experiment": "<experiment name or 'plan-wide'>",
     "issue": "<one concrete sentence>"}}
]}}
If the plan is clean, return {{"findings": []}}.
"""

    try:
        prompt_parts = [eval_prompt]
        prompt_parts.extend(loaded_images)
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        verdict, _ = parse_json_from_response(response)
        findings = (verdict or {}).get("findings", []) or []
        crit = [f for f in findings if f.get("severity") == "critical"]
        if crit:
            print(f"    - ⚠️  Critic noted {len(crit)} significant caveat(s).")
        else:
            print(f"    - ✅ Critic: {len(findings)} minor caveat(s).")
        return {"findings": findings}

    except Exception as e:
        logging.error(f"Critic step failed: {e}")
        # Fail open: if the critic crashes, assume the plan is okay to avoid blocking the user.
        return {"findings": []}


def perform_science_rag(objective: str,
                        instructions: str,
                        task_name: str,
                        kb_docs: Any,  # Pass the KB object here
                        model: Any,    # Pass the LLM object here
                        generation_config: Any,
                        primary_data_set: Optional[Dict[str, str]] = None,
                        image_paths: Optional[List[str]] = None,
                        image_descriptions: Optional[List[str]] = None,
                        additional_context: Optional[str] = None,
                        external_context: Optional[str] = None,
                        skill_context: Optional[str] = None,
                        return_context: bool = False,
                        mode_key: Optional[str] = None,
                        fallback_instructions: Optional[str] = None) -> Any:
    """
    Executes the Scientific/TEA RAG loop over the Docs KnowledgeBase.

    Thin planning-side wrapper over the shared ``scilink.knowledge.run_rag``
    engine: it resolves planning's primary-data Excel summary and selects the
    matching fallback instruction set, then delegates retrieval + generation.

    When ``return_context`` is True, returns ``(result, author_context)`` where
    ``author_context`` is ``{"retrieved_context", "primary_data"}`` — the
    grounding evidence the generation saw, so a downstream critic can verify
    against the same material. Default False preserves the result-only return.
    """

    # --- Resolve primary data (Excel) into a summary string ---
    primary_data_str = None
    if primary_data_set:
        try:
            chunks = parse_adaptive_excel(
                primary_data_set['file_path'], primary_data_set['metadata_path']
            )
            if chunks:
                summary = next(
                    (c for c in chunks if c['metadata'].get('content_type')
                     in ('dataset_summary', 'dataset_package')),
                    chunks[0],
                )
                primary_data_str = summary['text']
        except Exception as e:
            print(f"  - ⚠️ Warning: Failed to parse primary data set: {e}")

    # Literature context crowds out the less glamorous constraints: plans stay
    # on-topic but silently drop individual requirements, and the misses are
    # omissions rather than violations. Demanding an explicit constraint->step
    # mapping restores coverage at no measurable cost to plan quality. Scoped
    # to the condition it was measured in — constraints AND literature both
    # present; a constraints-only run already complies.
    if additional_context and external_context:
        additional_context = f"{additional_context}\n{CONSTRAINT_COVERAGE_NOTE}"

    # --- Select the fallback instruction set matching the planning task ---
    # Matched by PREFIX, not identity: callers append overrides to the
    # canonical block (ideation adds IDEATION_AUTHOR_OVERRIDE), and an
    # equality test silently left those runs with no fallback set — so an
    # "Insufficient context" author response aborted the run instead of
    # regenerating from general knowledge. An explicit argument still wins.
    if fallback_instructions is None:
        if instructions.startswith(HYPOTHESIS_GENERATION_INSTRUCTIONS):
            fallback_instructions = HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
        elif instructions.startswith(TEA_INSTRUCTIONS):
            fallback_instructions = TEA_INSTRUCTIONS_FALLBACK

    rag_out = run_rag(
        query=objective,
        instructions=instructions,
        kb=kb_docs,
        model=model,
        generation_config=generation_config,
        images=image_paths,
        image_descriptions=image_descriptions,
        external_context=external_context,
        additional_context=additional_context,
        primary_data_str=primary_data_str,
        skill_context=skill_context,
        fallback_instructions=fallback_instructions,
        task_name=task_name,
        return_context=return_context,
        mode_key=mode_key,
    )

    if return_context:
        result, retrieved_context = rag_out
        return result, {"retrieved_context": retrieved_context,
                        "primary_data": primary_data_str}
    return rag_out


def generate_plan_candidates(objective: str,
                             kb_docs: Any,
                             model: Any,
                             generation_config: Any,
                             n_candidates: int,
                             primary_data_set: Optional[Dict[str, str]] = None,
                             image_paths: Optional[List[str]] = None,
                             image_descriptions: Optional[List[str]] = None,
                             additional_context: Optional[str] = None,
                             external_context: Optional[str] = None,
                             skill_context: Optional[str] = None,
                             selection_profile: str = "lab",
                             contract: Optional[Dict[str, Any]] = None
                             ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Sequential, diversity-conditioned best-of-N candidate generation.

    Candidate 1 runs the normal science RAG (with its fallback swap); its
    instruction tier — "strict" or "fallback" — is then pinned for the whole
    run, so a fallback run authors ALL candidates under the fallback
    instructions and a strict run never silently mixes in general-knowledge
    plans. Each later candidate sees the prior candidates' hypothesis
    sentences plus the distinctness conditioning, and may DECLINE (error JSON)
    when the evidence supports no further distinct approach — so
    ``n_candidates`` is a cap, not a quota.

    ``selection_profile="ideation"`` has author-side effect only in STRICT
    runs: a fallback run swaps in the fallback instructions for every
    candidate, which already license general knowledge, so
    ``IDEATION_AUTHOR_OVERRIDE`` does not apply there and only the judge
    weighting differs.

    Returns ``(candidates, author_context, tier)`` where ``candidates`` holds
    only successful (non-declined) plans, in generation order, and
    ``author_context`` is the shared grounding evidence
    (``{"retrieved_context", "primary_data"}``) all candidates were authored
    against.
    """
    # Every author — candidate 1 included — is told it is writing ONE of N.
    # Without this, an objective phrased as "explore several alternatives"
    # makes the (otherwise unconditioned) first author pack all the
    # alternative strategies into a single plan, which then wins at the
    # judge on scope-matching (observed live via the meta delegation path).
    author_ctx = (f"{additional_context}\n\n{HYPOTHESIS_BESTOFN_AUTHOR_NOTE}"
                  if additional_context else HYPOTHESIS_BESTOFN_AUTHOR_NOTE) \
        if n_candidates > 1 else additional_context

    # Ideation profile relaxes the derivability rule at AUTHORING time: the
    # override rides the instruction block itself (adjacent to the safety
    # rule it supersedes). Lab profile authors exactly as before — the
    # benchmark showed the strict derivability discipline structurally caps
    # rediscovery when the key inspiration is absent from the context.
    # Grounding latitude is tier-dependent (the fallback set already grants
    # it); the OUTPUT rules are not — they ride both tiers, else a fallback
    # ideation run goes back to cramming its portfolio into steps.
    # A CONTRACT swaps what a candidate IS — a portfolio of research
    # directions rather than an experiment — while the tier pinning,
    # distinctness conditioning and early-stop below are shape-agnostic and
    # stay shared. Absent a contract this is byte-for-byte the experiment
    # path it has always been.
    _key = (contract or {}).get("key", "proposed_experiments")
    _label = (contract or {}).get("label", "Candidate Plan")
    _summarise = (contract or {}).get("summarise")

    _ideation = selection_profile == "ideation"
    author_instructions = HYPOTHESIS_GENERATION_INSTRUCTIONS
    fallback_set = HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
    if contract:
        author_instructions = contract["strict"]
        fallback_set = contract["fallback"]
    elif _ideation:
        author_instructions = (HYPOTHESIS_GENERATION_INSTRUCTIONS
                               + IDEATION_AUTHOR_OVERRIDE
                               + IDEATION_OUTPUT_RULES)
        fallback_set = (HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
                        + IDEATION_OUTPUT_RULES)

    first, author_context = perform_science_rag(
        objective=objective,
        instructions=author_instructions,
        task_name=f"{_label} 1",
        kb_docs=kb_docs,
        model=model,
        generation_config=generation_config,
        primary_data_set=primary_data_set,
        image_paths=image_paths,
        image_descriptions=image_descriptions,
        additional_context=author_ctx,
        external_context=external_context,
        skill_context=skill_context,
        return_context=True,
        mode_key="_rag_mode",
        # Stated explicitly rather than inferred: the ideation override is
        # concatenated onto the canonical block above, and the fallback set
        # carries the OUTPUT rules but not the grounding override (it already
        # licenses general knowledge — see this function's docstring).
        fallback_instructions=fallback_set,
    )
    tier = first.pop("_rag_mode", "strict") if isinstance(first, dict) else "strict"
    candidates = [first]
    if first.get("error") or not first.get(_key):
        return candidates, author_context, tier

    if tier == "fallback":
        print("  - ℹ️  Fallback tier pinned for all candidates in this run.")

    instructions = author_instructions if tier == "strict" else fallback_set

    for k in range(2, n_candidates + 1):
        prior = "\n".join(
            f"{i}. {_summarise(c) if _summarise else (c['proposed_experiments'][0].get('hypothesis', 'N/A'))}"
            for i, c in enumerate(candidates, 1)
        )
        conditioning = HYPOTHESIS_DISTINCTNESS_CONDITIONING.format(
            prior_hypotheses=prior
        )
        note_and_conditioning = f"{HYPOTHESIS_BESTOFN_AUTHOR_NOTE}\n\n{conditioning}"
        add_ctx = (f"{additional_context}\n\n{note_and_conditioning}"
                   if additional_context else note_and_conditioning)
        # No fallback_instructions here: within the pinned tier, a decline is
        # a decline — the early stop, not a trigger to change tiers mid-run.
        res = run_rag(
            query=objective,
            instructions=instructions,
            kb=kb_docs,
            model=model,
            generation_config=generation_config,
            images=image_paths,
            image_descriptions=image_descriptions,
            external_context=external_context,
            additional_context=add_ctx,
            primary_data_str=author_context.get("primary_data"),
            skill_context=skill_context,
            fallback_instructions=None,
            task_name=f"{_label} {k}",
        )
        if not isinstance(res, dict) or res.get("error") or not res.get(_key):
            reason = (res or {}).get("error", f"no {_key} returned") \
                if isinstance(res, dict) else "unparseable response"
            print(f"  - 🛑 Candidate {k} declined ({reason}) — "
                  f"stopping at {len(candidates)} distinct candidate(s).")
            break
        candidates.append(res)

    return candidates, author_context, tier


def judge_plan_candidates(objective: str,
                          candidates: List[Dict[str, Any]],
                          model: Any,
                          generation_config: Any,
                          retrieved_context: Optional[str] = None,
                          primary_data: Optional[str] = None,
                          images: Optional[List[Any]] = None,
                          image_descriptions: Optional[List[str]] = None,
                          additional_context: Optional[str] = None,
                          skill_context: Optional[str] = None,
                          fallback_tier: bool = False,
                          selection_profile: str = "lab") -> Dict[str, Any]:
    """
    Comparative LLM judge over best-of-N plan candidates.

    A selector, not an editor: scores each candidate on five criteria against
    the SAME evidence the authors saw and picks one; it emits no caveats (the
    advisory critic owns that channel) and never modifies a plan. Fails open
    to candidate 1 — a judge crash must not block planning — with the failure
    recorded in the returned dict.

    Returns ``{"scores": [...], "selected_candidate": <1-based>,
    "reasoning": str}`` (+ ``"error"`` if the judge failed).
    """
    fail_open = {"scores": [], "selected_candidate": 1,
                 "reasoning": "Judge unavailable — defaulted to candidate 1."}
    if len(candidates) < 2:
        return {"scores": [], "selected_candidate": 1,
                "reasoning": "Single candidate — no comparison needed."}

    cand_blocks = []
    for i, cand in enumerate(candidates, 1):
        cand_blocks.append(
            f"### CANDIDATE {i}\n"
            + json.dumps({"proposed_experiments":
                          cand.get("proposed_experiments", [])}, indent=2)
        )

    evidence_parts = []
    if primary_data:
        evidence_parts.append(f"## 📊 Primary Experimental Data:\n{primary_data}")
    if additional_context:
        # Lab constraints / equipment live here — feasibility is judged
        # against this, so it must reach the judge, not just the authors.
        evidence_parts.append(f"## Additional Context:\n{additional_context}")
    if retrieved_context:
        evidence_parts.append(f"## Retrieved Context (KB + literature):\n{retrieved_context}")
    if skill_context:
        evidence_parts.append(skill_context)

    loaded_images = []
    if images and _PIL_Image:
        for img in images:
            if isinstance(img, str):
                try:
                    loaded_images.append(_PIL_Image.open(img))
                except Exception as e:
                    print(f"    - ⚠️ Judge could not load image {img}: {e}")
            else:
                loaded_images.append(img)
    if loaded_images:
        note = "## Provided Images: (See attached)"
        if image_descriptions:
            note += f"\n## Image Descriptions:\n{json.dumps(image_descriptions, indent=2)}"
        evidence_parts.append(note)

    tier_note = ""
    if fallback_tier:
        tier_note = (
            "\nNOTE: ALL candidates were authored in fallback mode (general "
            "scientific knowledge; the local knowledge base lacked specific "
            "context). Judge groundedness against the evidence that IS "
            "provided (data, literature, images) and general plausibility — "
            "do not penalize candidates for the missing knowledge base.\n"
        )

    # Selection profile: same criteria and scores either way (the human sees
    # an identical card format); only the WEIGHTING of the pick changes.
    # "lab" codifies the execution-first behavior; "ideation" weights
    # information gain first — benchmark evidence showed the lab weighting
    # leaves the boldest (most rediscovery-shaped) candidate unpicked.
    profile_note = (BESTOFN_SELECTION_PROFILE_IDEATION
                    if selection_profile == "ideation"
                    else BESTOFN_SELECTION_PROFILE_LAB)

    prompt = (
        f"{HYPOTHESIS_BEST_OF_N_SELECTION_INSTRUCTIONS}\n{profile_note}\n"
        f"## OBJECTIVE:\n{objective}\n{tier_note}\n"
        f"## EVIDENCE ALL CANDIDATES WERE AUTHORED AGAINST:\n"
        + ("\n\n".join(evidence_parts) if evidence_parts
           else "(no shared evidence beyond the objective)")
        + "\n\n## THE CANDIDATES:\n" + "\n\n".join(cand_blocks)
    )

    try:
        prompt_parts = [prompt]
        prompt_parts.extend(loaded_images)
        response = model.generate_content(prompt_parts,
                                          generation_config=generation_config)
        verdict, err = parse_json_from_response(response)
        if err or not isinstance(verdict, dict):
            logging.error(f"Best-of-N judge parse failure: {err}")
            return {**fail_open, "error": str(err)}
        sel = verdict.get("selected_candidate")
        if not isinstance(sel, int) or not (1 <= sel <= len(candidates)):
            logging.error(f"Best-of-N judge returned invalid selection: {sel}")
            return {**fail_open, "error": f"invalid selected_candidate: {sel}"}
        return {"scores": verdict.get("scores", []),
                "selected_candidate": sel,
                "reasoning": verdict.get("reasoning", "")}
    except Exception as e:
        logging.error(f"Best-of-N judge failed: {e}")
        return {**fail_open, "error": str(e)}


def normalize_code(code: str) -> str:
    """Normalizes code by collapsing all whitespace to single spaces."""
    if not code: return ""
    return " ".join(code.split())


def perform_code_rag(
    result: Dict[str, Any],
    kb_code: Any,
    model: Any,
    generation_config: Any,
    previous_implementations: Optional[List[Dict[str, Any]]] = None,
    skill_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves API syntax from the Code KB and generates implementation scripts.
    If previous code implementations are provided, lets the LLM decide whether to:
    - Preserve existing code (no changes needed)
    - Update existing code (incremental edits)
    - Rewrite from scratch (major procedural changes)
    """
    
    experiments = result.get("proposed_experiments", [])
    if not experiments:
        return result
    
    # 1. Retrieve API documentation from Code KB
    all_steps_text = " ".join([
        " ".join(e.get('experimental_steps', [])) 
        for e in experiments
    ])
    
    print(f"  - 🔍 Retrieving API syntax for implementation...")
    hits = kb_code.retrieve(f"python implementation for {all_steps_text}", top_k=5)
    
    repo_map_context = kb_code.get_relevant_maps(hits) if hits else ""
    code_ctx = "\n\n".join([
        f"FILE: {c['metadata']['source']}\n{c['text']}" 
        for c in hits
    ]) if hits else "No API examples found in Code KB."
    
    code_files = list(set([Path(c['metadata']['source']).name for c in hits])) if hits else []
    
    # 2. Build mapping of previous implementations by experiment name
    previous_code_map = {}
    if previous_implementations:
        for impl in previous_implementations:
            exp_name = impl.get('experiment_name', '')
            if exp_name:
                previous_code_map[exp_name] = impl
    
    # 3. Generate/Update code for each experiment
    for exp in experiments:
        steps = exp.get("experimental_steps", [])
        exp_name = exp.get("experiment_name", "Experiment")
        hypothesis = exp.get("hypothesis", "N/A")
        
        # Find matching previous implementation
        prev_impl = previous_code_map.get(exp_name)
        
        # Build the master prompt
        prompt = f"""
You are an expert Research Software Engineer working on an iterative scientific project.

**EXPERIMENT OVERVIEW:**
Name: {exp_name}
Hypothesis: {hypothesis}

**NEW EXPERIMENTAL STEPS:**
{json.dumps(steps, indent=2)}

"""

        # Add previous implementation context if it exists
        if prev_impl:
            prev_code = prev_impl.get('code', '')
            prev_iteration = prev_impl.get('iteration', 'unknown')
            
            prompt += f"""
**PREVIOUS IMPLEMENTATION (Iteration {prev_iteration}):**
```python
{prev_code}
```

**YOUR DECISION:**
You must choose one of three strategies:

1. **PRESERVE** - If the new steps are identical or the change is only a parameter/value:
   - Return the exact same code unchanged
   - Example: "Increase temperature from 50°C to 60°C" → just parameter change

2. **UPDATE** - If the procedure changed but the overall structure is similar:
   - Keep the working framework (imports, error handling, setup)
   - Modify only the changed sections
   - Add comments marking what changed
   - Example: "Add a centrifugation step after mixing" → insert new function call

3. **REWRITE** - If this is a fundamentally different approach:
   - Start fresh using the API Reference below
   - Example: "Switch from batch processing to real-time streaming"

"""
        else:
            prompt += f"""
**PREVIOUS IMPLEMENTATION:**
None - this is the first implementation for this experiment.

**YOUR TASK:**
Write a complete Python script from scratch using the API Reference below.

"""

        # Add skill context if available
        if skill_context:
            prompt += f"\n{skill_context}\n"

        # Add API context
        prompt += f"""
**REPOSITORY STRUCTURES (for correct import paths):**
{repo_map_context}

**API SYNTAX REFERENCE (Official Documentation/Examples):**
{code_ctx}

**INSTRUCTIONS:**
- Use the "API Syntax Reference" to find the correct functions.
- Map the scientific intent of the Steps to the code.
- You must prioritize using classes and functions from the API Reference over generic external libraries.
- If updating existing code, preserve working patterns
- Return ONLY valid JSON.

**OUTPUT FORMAT:**
Respond with a JSON object:
{{"implementation_code": "COMPLETE_PYTHON_CODE_HERE"}}
"""
        
        try:
            print(f"    - 🤖 Analyzing '{exp_name}'...")
            resp = model.generate_content([prompt], generation_config=generation_config)
            code_res, parse_error = parse_json_from_response(resp)
            
            if parse_error:
                print(f"    - ⚠️ JSON parsing error for '{exp_name}': {parse_error}")
                continue
            
            if code_res and "implementation_code" in code_res:
                new_code = code_res["implementation_code"]
                exp["implementation_code"] = new_code
                exp["code_source_files"] = code_files
                
                if prev_impl:
                    old_code = prev_impl.get('code', '')
                    
                    # Compare normalized versions to ignore harmless whitespace/indentation differences
                    if normalize_code(new_code) == normalize_code(old_code):
                        print(f"    - ⏹️  Preserved (No logic changes): {exp_name}")
                    else:
                        print(f"    - 🔄 Updated: {exp_name}")

                else:
                    print(f"    - ✨ Generated: {exp_name}")
                            
            else:
                print(f"    - ⚠️ LLM did not return code for '{exp_name}'")
                
        except Exception as e:
            print(f"    - ❌ Failed to process '{exp_name}': {e}")
    
    return result


def refine_plan_with_feedback(original_result: Dict[str, Any],
                              feedback: str,
                              objective: str,
                              model: Any,
                              generation_config: Any,
                              new_context: Optional[str] = None,
                              result_images: Optional[List[Any]] = None,
                              skill_context: Optional[str] = None
                              ) -> Dict[str, Any]:
    """
    Refines the experimental plan based on user input or experimental results.
    Now supports injecting fresh RAG context relevant to the feedback/results.

    Feedback here is authoritative (human review, experimental results, or a
    discovered constraint) and is incorporated directly. The advisory critic does
    NOT route through this function — its caveats are surfaced for a human /
    consumer to weigh, not auto-applied.
    """

    # Construct the context block if available
    context_block = ""
    if new_context:
        context_block = (
            f"\n**📚 RELEVANT LITERATURE FOR OBSERVED RESULTS:**\n"
            f"{new_context}\n"
            f"(Use this literature to interpret the results and adjust the plan accordingly.)\n"
        )

    # Strip source_documents from plan so the LLM only cites references
    # it actually uses during refinement (from KB RAG or external context)
    plan_for_prompt = {k: v for k, v in original_result.items() if k != "source_documents"}

    refinement_prompt = f"""
    You are an expert Research Strategist acting as an editor.

    **Original Objective:** {objective}

    **Current Plan (JSON):**
    {json.dumps(plan_for_prompt, indent=2)}

    **Experimental Results / Feedback:** "{feedback}"
    {context_block}

    **Task:**
    Update the "Current Plan" to strictly address the Feedback and Results.
    - If the results indicate failure, use the Literature Context to propose a fix.
    - If the results indicate success, move to the next logical step.

    **Constraints:**
    - You MUST return the exact same JSON structure (keys: "proposed_experiments", etc.).
    - Update "experimental_steps", "hypothesis", "required_equipment", or "concepts" as requested.
    - If the current plan carries a "concepts" list, it is the portfolio of research directions: KEEP it as a list of concept objects, revising, adding or removing entries as the feedback requires. Never collapse it into prose or into "experimental_steps" rows.
    - Do NOT add explanations outside the JSON.
    - Do NOT carry forward quantitative claims from the original plan that contradict the experimental results.
    - For "source_documents", list ONLY references you actually used from the provided Literature Context. Do NOT invent or carry forward references not present in the context.

    **Output:**
    A single valid JSON object containing the updated plan.
    """

    # A portfolio refines against judgement, not measurement — see
    # PORTFOLIO_REFINEMENT_RULES.
    if plan_is_portfolio(original_result) or any(
            (e or {}).get("concepts")
            for e in (original_result.get("proposed_experiments") or [])):
        refinement_prompt += PORTFOLIO_REFINEMENT_RULES

    prompt_parts = [refinement_prompt]

    if skill_context:
        prompt_parts.append(skill_context)

    if result_images:
        print(f"    + 📎 Attaching {len(result_images)} images to refinement prompt.")
        prompt_parts.extend(result_images)

    try:
        # Generate Content (Sending List of Text + Images)
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        refined_result, error_msg = parse_json_from_response(response)
        
        if error_msg:
            print(f"    - ⚠️ JSON Parsing Failed: {error_msg}. Retrying...")

            raw_text = ""
            if hasattr(response, 'text'):
                raw_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                raw_text = response.parts[0].text

            repair_prompt = (
                "The following text was intended to be valid JSON but has a formatting error.\n\n"
                f"**Error:** {error_msg}\n\n"
                f"**Raw text:**\n{raw_text}\n\n"
                "Fix ONLY the JSON formatting issues (missing commas, unescaped characters, "
                "trailing commas, etc.). Do NOT change any content or values. "
                "Return ONLY the corrected JSON object with no explanation."
            )

            try:
                retry_response = model.generate_content(
                    [repair_prompt], generation_config=generation_config
                )
                refined_result, retry_error = parse_json_from_response(retry_response)

                if retry_error:
                    print(f"    - ⚠️ JSON Retry Also Failed: {retry_error}")
                    return {
                        "error": "JSON_PARSE_ERROR",
                        "message": f"LLM output invalid after retry: {retry_error}",
                        "raw_output": str(raw_text)[:500]
                    }
                else:
                    print(f"    - ✅ JSON repair succeeded on retry.")
            except Exception as retry_exc:
                print(f"    - ⚠️ JSON retry call failed: {retry_exc}")
                return {
                    "error": "JSON_PARSE_ERROR",
                    "message": f"LLM output invalid: {error_msg}",
                    "raw_output": str(raw_text)[:500]
                }
        
        # Structure Validation
        if "proposed_experiments" not in refined_result:
            return {
                "error": "INVALID_STRUCTURE",
                "message": "JSON parsed but missing 'proposed_experiments' key.",
                "raw_output": str(refined_result)[:200]
            }
            
        return refined_result
        
    except Exception as e:
        print(f"    - ⚠️ Error during refinement: {e}")
        return original_result
    

def refine_code_with_feedback(result: Dict[str, Any], 
                              feedback: str, 
                              model: Any, 
                              generation_config: Any) -> Dict[str, Any]:
    """
    Refines the implementation code based on user feedback.
    """
    experiments = result.get("proposed_experiments", [])
    if not experiments:
        return result

    # Context construction: We dump the current code so the LLM knows what to fix
    current_code_state = ""
    for i, exp in enumerate(experiments):
        name = exp.get('experiment_name', f'Experiment {i+1}')
        code = exp.get("implementation_code", "# No code generated")
        current_code_state += f"--- CODE FOR: {name} ---\n{code}\n\n"

    prompt = f"""
    You are a Senior Research Software Engineer.
    
    **TASK:** Refine the Python implementation code based on User Feedback.
    
    **CURRENT CODE STATE:**
    {current_code_state}
    
    **USER FEEDBACK / ERROR REPORT:**
    "{feedback}"
    
    **INSTRUCTIONS:**
    1. Apply the user's fixes to the relevant code blocks.
    2. If the user refers to a specific experiment, only update that one.
    3. You must return a JSON object with a list of "updated_codes". 
       Each item in the list must match the order of the experiments above.
    4. Provide the FULL updated code for each script, not just the diffs.
    
    **OUTPUT FORMAT:**
    {{
        "updated_codes": [
            "FULL_PYTHON_SCRIPT_1...",
            "FULL_PYTHON_SCRIPT_2..."
        ]
    }}
    """
    
    print(f"    - ↻ Refine Code RAG: Generating updates based on feedback...")
    try:
        response = model.generate_content([prompt], generation_config=generation_config)
        updates, error = parse_json_from_response(response)
        
        if updates and "updated_codes" in updates:
            new_codes = updates["updated_codes"]
            # Map back to the result structure
            if len(new_codes) == len(experiments):
                for i, code in enumerate(new_codes):
                    experiments[i]["implementation_code"] = code
                print("    - ✅ Code successfully refined.")
            else:
                print("    - ⚠️ Warning: LLM returned wrong number of code blocks. Skipping update.")
        elif error:
            print(f"    - ⚠️ JSON Error during refinement: {error}")
        
        return result
        
    except Exception as e:
        print(f"    - ❌ Error during code refinement: {e}")
        return result

def author_technical_document(request: str,
                              kb_docs: Any,
                              model: Any,
                              generation_config: Any,
                              *,
                              external_context: Optional[str] = None,
                              source_documents: Optional[str] = None,
                              additional_context: Optional[str] = None,
                              skill_context: Optional[str] = None,
                              revise_document: Optional[str] = None,
                              task_name: str = "Technical Document"
                              ) -> Dict[str, Any]:
    """Author a grounded technical document (roadmap, estimate, memo, brief).

    Same retrieval path as plan generation, different contract: the model
    returns SECTIONS, not an experiment. Sections rather than one markdown
    blob because these documents run to thousands of words, and a single
    JSON string that long is the shape that has truncated on us before.

    Returns ``{"sections": [...]}`` or ``{"error": ...}``; the caller
    assembles the markdown.
    """
    parts = []
    if revise_document:
        parts.append("## THE CURRENT DOCUMENT (revise THIS, in full):\n"
                     + revise_document)
    if source_documents:
        parts.append("## PRIOR SESSION DOCUMENTS (build on these, do not "
                     "restate them wholesale):\n" + source_documents)
    if additional_context:
        parts.append(additional_context)

    _instr = TECHNICAL_DOCUMENT_INSTRUCTIONS
    _fallback = TECHNICAL_DOCUMENT_INSTRUCTIONS_FALLBACK
    if revise_document:
        _instr += TECHNICAL_DOCUMENT_REVISION_RULES
        _fallback += TECHNICAL_DOCUMENT_REVISION_RULES
    result = run_rag(
        query=request,
        instructions=_instr,
        fallback_instructions=_fallback,
        kb=kb_docs,
        model=model,
        generation_config=generation_config,
        external_context=external_context,
        additional_context="\n\n".join(parts) if parts else None,
        skill_context=skill_context,
        task_name=task_name,
    )
    if not isinstance(result, dict):
        return {"error": f"Document generation returned {type(result).__name__}"}
    return result


def document_to_markdown(title: str, sections: List[Dict[str, Any]]) -> str:
    """Assemble authored sections into a markdown document.

    The title is owned HERE, not by the author. On a revision the model is
    shown a document that already opens with its title and told to return
    everything verbatim, so it faithfully returns the title as a section —
    and prepending it again added one copy per revision (live: six stacked
    titles, readable as the title's own edit history). A leading section that
    merely repeats the title is therefore dropped.
    """
    def _is_title_echo(sec) -> bool:
        if not isinstance(sec, dict) or (sec.get("body") or "").strip():
            return False
        h = str(sec.get("heading") or "").strip().lstrip("#").strip()
        return h.casefold() == str(title).strip().casefold()

    sections = list(sections or [])
    while sections and _is_title_echo(sections[0]):
        sections.pop(0)

    out = [f"# {title}", ""]
    for sec in sections or []:
        if not isinstance(sec, dict):
            out += [str(sec), ""]
            continue
        heading = str(sec.get("heading") or "").strip()
        body = str(sec.get("body") or "").strip()
        if heading:
            out += [f"## {heading}", ""]
        if body:
            out += [body, ""]
    return "\n".join(out).rstrip() + "\n"


# The portfolio contract for generate_plan_candidates. The tier pinning,
# distinctness conditioning and early stop are shape-agnostic and shared; only
# what a candidate IS changes.
def portfolio_contract() -> Dict[str, Any]:
    def _summarise(cand: Dict[str, Any]) -> str:
        dirs = cand.get("directions") or []
        titles = "; ".join(str(d.get("title") or d.get("id") or "")
                           for d in dirs[:6] if isinstance(d, dict))
        return f"{cand.get('thesis', 'N/A')} — directions: {titles}"

    return {"key": "directions",
            "label": "Candidate Portfolio",
            "strict": IDEATION_PORTFOLIO_INSTRUCTIONS,
            "fallback": IDEATION_PORTFOLIO_INSTRUCTIONS_FALLBACK,
            "summarise": _summarise}


def author_portfolio(objective: str,
                     kb_docs: Any,
                     model: Any,
                     generation_config: Any,
                     **kw) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Author ONE portfolio (the n_candidates=1 path)."""
    return perform_science_rag(
        objective=objective,
        instructions=IDEATION_PORTFOLIO_INSTRUCTIONS,
        fallback_instructions=IDEATION_PORTFOLIO_INSTRUCTIONS_FALLBACK,
        task_name="Research Portfolio",
        kb_docs=kb_docs,
        model=model,
        generation_config=generation_config,
        return_context=True,
        **kw,
    )
