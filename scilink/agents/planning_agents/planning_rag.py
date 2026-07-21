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
    IDEATION_AUTHOR_OVERRIDE
)


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
    plan_summary_lines = []
    for i, exp in enumerate(experiments):
        name = exp.get('experiment_name', 'N/A')
        hyp = exp.get('hypothesis', 'N/A')
        justification = exp.get('justification', 'No justification provided.')

        plan_summary_lines.append(f"Experiment {i+1}: {name}")
        plan_summary_lines.append(f"  Hypothesis: {hyp}")
        plan_summary_lines.append(f"  Justification: {justification}")
        plan_summary_lines.append("---")

    plan_summary = "\n".join(plan_summary_lines)

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

    plan_summary = "\n".join(
        f"Experiment {i+1}: {exp.get('experiment_name', 'N/A')}\n"
        f"  Hypothesis: {exp.get('hypothesis', 'N/A')}\n"
        f"  Justification: {exp.get('justification', 'No justification provided.')}\n---"
        for i, exp in enumerate(experiments)
    )

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
            "data values here:\n"
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
You are reviewing an experimental plan for PHYSICAL REALISM and INTERNAL CONSISTENCY.
A separate check has already confirmed the plan is relevant to the objective, so do
NOT re-litigate objective-relevance or document grounding here.

OBJECTIVE: "{objective}"

PROPOSED PLAN:
{plan_summary}
{evidence_section}{prior_section}
Assume the plan may be flawed and try to break it on two axes:
  • physics — parameters or conditions that are physically impossible or implausible;
              a technique that cannot measure the stated quantity or resolve the
              claimed scale; violated conservation laws or instrument limits.
  • consistency — a justification that contradicts its own hypothesis; steps that do
              not actually test the stated hypothesis; equipment that does not match
              the steps; two experiments that contradict each other.
Report only a flaw you can name concretely. Do NOT invent problems to appear
thorough; if an axis is clean, report nothing for it.

SEVERITY:
  • critical — would make the plan infeasible or scientifically wrong.
  • minor    — worth noting, but the plan still stands.

OUTPUT — a single JSON object:
{{"findings": [
   {{"dimension": "physics|consistency",
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
                        mode_key: Optional[str] = None) -> Any:
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

    # --- Select the fallback instruction set matching the planning task ---
    fallback_instructions = None
    if instructions == HYPOTHESIS_GENERATION_INSTRUCTIONS:
        fallback_instructions = HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
    elif instructions == TEA_INSTRUCTIONS:
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
                             selection_profile: str = "lab"
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
    author_instructions = HYPOTHESIS_GENERATION_INSTRUCTIONS
    if selection_profile == "ideation":
        author_instructions = (HYPOTHESIS_GENERATION_INSTRUCTIONS
                               + IDEATION_AUTHOR_OVERRIDE)

    first, author_context = perform_science_rag(
        objective=objective,
        instructions=author_instructions,
        task_name="Candidate Plan 1",
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
    )
    tier = first.pop("_rag_mode", "strict") if isinstance(first, dict) else "strict"
    candidates = [first]
    if first.get("error") or not first.get("proposed_experiments"):
        return candidates, author_context, tier

    if tier == "fallback":
        print("  - ℹ️  Fallback tier pinned for all candidates in this run.")

    instructions = (author_instructions if tier == "strict"
                    else HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK)

    for k in range(2, n_candidates + 1):
        prior = "\n".join(
            f"{i}. {c['proposed_experiments'][0].get('hypothesis', 'N/A')}"
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
            task_name=f"Candidate Plan {k}",
        )
        if not isinstance(res, dict) or res.get("error") or not res.get("proposed_experiments"):
            reason = (res or {}).get("error", "no experiments returned") \
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
    - Update "experimental_steps", "hypothesis", or "required_equipment" as requested.
    - Do NOT add explanations outside the JSON.
    - Do NOT carry forward quantitative claims from the original plan that contradict the experimental results.
    - For "source_documents", list ONLY references you actually used from the provided Literature Context. Do NOT invent or carry forward references not present in the context.

    **Output:**
    A single valid JSON object containing the updated plan.
    """

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