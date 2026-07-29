import logging
import os
import textwrap
from typing import Any, Callable, Optional
import json


# Generic Tier-A synthesis re-entry instructions (issue #322). Deliberately
# modality-neutral: the payload critique + surfaced features carry the
# specifics. Revision only — the original analysis is never overwritten.
SYNTHESIS_REENTRY_INSTRUCTIONS = """You are revising the final interpretation of a completed scientific analysis.

You are given: the ORIGINAL interpretation (detailed analysis text and scientific claims), the quantitative FEATURES the analysis extracted, and a CRITIQUE/CONTEXT payload from a reviewer (a human expert, an automated verifier, or a literature search).

Revise the interpretation in light of the payload:
- Incorporate only changes the payload and the extracted features actually support; do not fabricate new measurements.
- If the payload contradicts the original interpretation, say what changed and why.
- If the payload adds context (e.g. literature) that refines identification or mechanism, integrate it and cite it as provided context.
- Keep everything that remains valid; mark genuine uncertainty as uncertainty.

Return JSON with:
{
    "detailed_analysis": "the full REVISED interpretation text",
    "scientific_claims": [
        {"claim": "...", "scientific_impact": "...", "has_anyone_question": "Has anyone ...?", "keywords": ["...", "..."]}
    ],
    "revision_summary": "1-3 sentences: what changed relative to the original and why"
}
"""


class LiteratureSearchController:
    """Search literature if enabled and query provided.

    Shared across the analysis modalities (one implementation; the curve /
    image controller modules re-export it under their historical name).

    DEPRECATED as an in-pipeline step: prefer the orchestrator-level
    `search_literature` tool, which fetches lit context BEFORE planning so
    the planner can produce a literature-informed plan. Retained as a
    fallback for direct-Python-API callers using `use_literature=True`.
    """

    def __init__(
        self,
        logger: logging.Logger,
        literature_agent: Any | None = None,
        output_dir: str = "",
    ):
        self.logger = logger
        self.literature_agent = literature_agent
        self.output_dir = output_dir

    def _save_results(self, query: str, report: str) -> dict:
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature")
            os.makedirs(lit_dir, exist_ok=True)

            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, "w") as f:
                f.write(query)
            saved_files["query_file"] = query_path

            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, "w") as f:
                f.write(report)
            saved_files["report_file"] = report_path
        except Exception as e:
            self.logger.warning(f"Failed to save literature: {e}")
        return saved_files

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if state.get("literature_context"):
            self.logger.info("\n📚 --- Skipping Literature (pre-fetched via search_literature tool) ---\n")
            return state

        if self.literature_agent is None:
            self.logger.info("\n📚 --- Skipping Literature (disabled) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        query = state.get("literature_query")
        if not query:
            self.logger.info("\n📚 --- Skipping Literature (no query needed) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        self.logger.info("\n📚 --- Searching Literature ---\n")
        self.logger.info(f"  Query: {query}")

        try:
            result = self.literature_agent.query_for_models(query)
            if result.get("status") == "success":
                state["literature_context"] = result["formatted_answer"]
                self.logger.info("  ✅ Success")
            else:
                state["literature_context"] = None
                self.logger.warning("  ⚠️ No results")

            state["literature_files"] = self._save_results(
                query, state["literature_context"] or f"No results: {result.get('message')}"
            )
        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}")
            state["literature_context"] = None
            state["literature_files"] = self._save_results(query, f"Error: {e}")

        return state


class RunFinalInterpretationController:
    """
    [🧠 LLM Step]
    A generic controller that takes the 'final_prompt_parts' from the state,
    runs the LLM, and stores the 'result_json' and 'error_dict' back in the state.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn  # Pass in the agent's parser

    def execute(self, state: dict) -> dict:
        # In skip-decomposition mode the iteration-stage interpretation
        # has nothing to interpret — no NMF/PCA/ICA results, no dynamic
        # analysis output yet — so it would generate a pre-emptive
        # narrative that misleads the human-feedback step and downstream
        # readers. The synthesis stage runs its own interpretation after
        # dynamic analysis completes, so skipping here loses nothing.
        # Detect the iteration-stage skip by the presence of
        # state["skip_decomposition"]; the synthesis stage builds its
        # state dict without that key, so the synthesis interpretation
        # runs normally.
        if state.get("skip_decomposition"):
            self.logger.info(
                "Skip-decomposition gate active — bypassing iteration-stage "
                "interpretation. The synthesis stage will interpret after "
                "dynamic analysis completes."
            )
            return state

        self.logger.info("🧠 LLM Step: Generating scientific interpretation...")
        prompt = state.get("final_prompt_parts")

        if not prompt:
            self.logger.error("Interpretation step invoked with no 'final_prompt_parts' in state.")
            state["error_dict"] = {"error": "Pipeline failed to build interpretation prompt"}
            return state

        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)

            state["result_json"] = result_json
            state["error_dict"] = error_dict

            if not error_dict:
                self.logger.info("✅ LLM Step Complete: Interpretation generated.")
            else:
                self.logger.error(f"❌ LLM Step Failed: {error_dict.get('details')}")

        except Exception as e:
            self.logger.exception(f"❌ LLM Step Failed: {e}")
            state["result_json"] = None
            state["error_dict"] = {"error": "LLM interpretation step failed", "details": str(e)}

        return state

class StoreAnalysisResultsController:
    """
    [🛠️ Tool Step]
    A generic controller that takes the 'analysis_images' from the state
    and saves them using the agent's 'store_fn' for the feedback loop.
    """
    def __init__(self, logger: logging.Logger, store_fn: Callable):
        self.logger = logger
        self._store_analysis_images = store_fn

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Tool Step: Storing analysis images for feedback...")
        
        if state.get("error_dict"):
            self.logger.warning("Skipping storage: An error occurred in the pipeline.")
            return state

        try:
            analysis_metadata = {
                "image_path": state.get("image_path"),
                "system_info": state.get("system_info"),
                "num_stored_images": len(state.get("analysis_images", []))
                # You can add more metadata from the 'state' dict here
            }
            self._store_analysis_images(state.get("analysis_images", []), analysis_metadata)
            self.logger.info("✅ Tool Step Complete: Analysis images stored.")
        except Exception as e:
            self.logger.error(f"❌ Tool Step Failed: Could not store analysis images: {e}")
            
        return state
    

class IterativeFeedbackController:
    """
    [🧠 LLM/User Step] 
    Facilitates human-in-the-loop validation and refinement.
        """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable, settings: dict, refinement_instruction: str):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        
        # The pipeline MUST provide the logic prompt
        self.refinement_instruction = refinement_instruction 
            
        self.feedback_depths = settings.get('feedback_depths', [0])

    def execute(self, state: dict) -> dict:
        # Check if human feedback is globally enabled (via agent settings)
        if not state.get('settings', {}).get('enable_human_feedback', False):
             self.logger.info("Feedback skipped: Human feedback not enabled for this agent.")
             return state
        
        current_depth = state.get("current_depth", -1) 
        
        if current_depth not in self.feedback_depths:
            self.logger.info(f"Feedback skipped: Current depth ({current_depth}) is not in allowed list {self.feedback_depths}.")
            return state

        decision = state.get("refinement_decision")
        if not decision:
            self.logger.warning("Feedback skipped: 'refinement_decision' missing from state.")
            return state

        self.logger.info("\n\n👤 --- USER STEP: REVIEW ANALYSIS PLAN --- 👤\n")

        # --- 1. Display Current Decision to User and Collect Feedback ---
        # In skip-decomposition mode no iteration-stage analysis has run
        # yet, so there's no current-analysis summary to show — the
        # "summary" would be the pre-emptive LLM narrative that we now
        # skip in RunFinalInterpretationController. Suppress the summary
        # block when there is no real analysis text to display.
        skip_mode = bool(state.get("skip_decomposition"))
        iteration_title = state.get("iteration_title", "Current Analysis")
        result_json = state.get("result_json") or {}
        analysis_text = result_json.get("detailed_analysis", "").strip()
        targets = decision.get("targets", [])

        print("\n" + "="*80)
        header = "ANALYSIS PLAN REVIEW" if skip_mode else "ANALYSIS STEP REVIEW"
        print(f"🎯 {header}: {iteration_title}")
        print("="*80)

        if analysis_text:
            print("\n**SUMMARY OF CURRENT ANALYSIS:**")
            print(analysis_text)
            print("-" * 80)
        elif skip_mode:
            print("\n(Skip-decomposition mode — no iteration-stage analysis "
                  "to summarize; the synthesis stage will interpret after the "
                  "dynamic-analysis step runs.)")
            print("-" * 80)

        plan_label = "Analysis Plan Ready" if skip_mode else "Refinement Needed"
        print(f"🧠 LLM's Proposed Plan: {plan_label} = **{decision.get('refinement_needed', False)}**")
        print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        print()
        print("-" * 80)
        print(f"\U0001f3af Targeted Actions ({len(targets)} found)")
        print("-" * 80)

        if not targets:
            print("  (No specific targets were generated.)")

        for i, t in enumerate(targets, 1):
            t_type = t.get('type', 'N/A')
            t_value = t.get('value', None)
            t_desc = t.get('description', 'No description provided.')

            # custom_code targets carry value=None by schema design —
            # the description is the payload. Skip the value annotation
            # in that case to avoid the meaningless "(value: None)" noise.
            header = f"  [{i}] {t_type}"
            if t_value is not None:
                header += f"  (value: {t_value})"
            print(f"\n{header}")
            for line in textwrap.wrap(t_desc, width=70):
                print(f"      {line}")

        print("-" * 80)
        
        try:
            user_feedback = input("\n🤔 Your feedback to adjust the targets/plan (or press Enter to accept): ").strip()
        except KeyboardInterrupt:
            self.logger.warning("User interrupted feedback. Accepting original decision.")
            return state

        if not user_feedback:
            decision_label = "analysis plan" if skip_mode else "refinement decision"
            self.logger.info(f"✅ User accepted original {decision_label}.")
            return state
        
        # --- 3. Run LLM Refinement with Full Context ---
        self.logger.info("🔄 Refining decision using full scientific context...")
        
        # Prepare context for the LLM
        system_info_json = json.dumps(state.get("system_info", {}), indent=2)

        prompt_parts = [
            f"You are an expert reviewer. Use the HUMAN EXPERT FEEDBACK to produce a **REVISED** and definitive version of the analysis plan JSON object. The human input overrides the initial automated logic.",
            
            f"\n\n--- LLM'S ORIGINAL DECISION JSON ---\n{json.dumps(decision, indent=2)}",
            f"\n\n--- CURRENT ITERATION'S DETAILED ANALYSIS ---\n{analysis_text}",
            f"\n\n--- CURRENT SYSTEM METADATA ---\n{system_info_json}",
            f"\n\n--- HUMAN EXPERT FEEDBACK ---\n\"{user_feedback}\"",
            "\n\n--- VISUAL CONTEXT (Plots from Current Analysis) ---\n"
        ]

        # Add all analysis images for visual context
        for img in state.get("analysis_images", []):
            image_bytes = img.get('data') or img.get('bytes')
            if image_bytes:
                prompt_parts.append(f"\n{img['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
        
        # Inject mandatory instructions here
        prompt_parts.append(f"\n\n### DECISION RULES\n{self.refinement_instruction}")

        prompt_parts.append("""

### REVISION REQUIREMENTS
You MUST re-analyze the original targets and the human feedback, then generate a single, complete, and definitive JSON object.

Your task is to provide the FINAL list of executable tasks. Do NOT embed descriptions or reasonings outside of the specified keys.
                            
You are FORBIDDEN from returning `refinement_needed: true` with an empty `targets` list. If refinement is needed, at least one target is required.
                            
Output must strictly adhere to the JSON format defined above.
""")

        
        # Call LLM for structured revision
        param_gen_config = None#GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            refined_json, error_dict = self._parse_llm_response(response)

            if refined_json and not error_dict:
                state["refinement_decision"] = refined_json
                self.logger.info(f"✅ Refinement success. Final decision: {refined_json.get('reasoning', 'No reasoning').strip()}")
                
                # Debug print to confirm targets were created
                new_targets = refined_json.get("targets", [])
                print(f"\n✅ REFINED: New plan established based on feedback. ({len(new_targets)} targets created)")
            else:
                self.logger.error("❌ LLM failed to produce a valid refinement JSON. Retaining original decision.")
                print("\n❌ Refinement failed due to bad LLM output. Retaining original plan.")

        except Exception as e:
            self.logger.error(f"❌ Error during LLM refinement call: {e}")
            print("\n❌ Critical error during refinement. Retaining original plan.")
            
        return state

class RunSelfReflectionController:
    """
    [🧠 CRITIC Step]
    Reviews the Draft 1 analysis against the images to catch hallucinations.

    Modality-agnostic (reads only ``result_json`` + ``analysis_images``);
    moved verbatim from the hyperspectral controllers (issue #327 phase 3).
    ``instructions`` defaults to the hyperspectral reflection prompt so the
    original pipeline behavior is unchanged.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn,
                 instructions: Optional[str] = None):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        if instructions is None:
            from ..instruct import SPECTROSCOPY_REFLECTION_INSTRUCTIONS
            instructions = SPECTROSCOPY_REFLECTION_INSTRUCTIONS
        self.instructions = instructions

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- SELF-REFLECTION: REVIEWING ANALYSIS --- 🧠\n")

        # 1. Get the Draft 1 Analysis
        current_result = state.get("result_json")
        if not current_result:
            self.logger.warning("No analysis found to review.")
            return state

        draft_text = current_result.get("detailed_analysis", "")
        claims = current_result.get("scientific_claims", [])

        # 2. Build the Review Prompt
        prompt_parts = [self.instructions]
        prompt_parts.append("\n\n### DRAFT ANALYSIS TO REVIEW:")
        prompt_parts.append(f"{draft_text}")
        prompt_parts.append(f"\n\n### GENERATED CLAIMS:\n{json.dumps(claims, indent=2)}")

        # 2b. Measurement metadata — lets the critic weigh instrument-level
        # explanations (modulation broadening, drift over a long acquisition,
        # axis conventions) against electronic-structure claims. Deliberately
        # NOT added: the flux table (measurability is the upstream QC judges'
        # jurisdiction), the QC attempt trajectory (double-jeopardy bias on
        # already-accepted maps; salvage markers already reach the draft),
        # and the objective (scope-fit belongs to synthesis and the meta).
        sys_info = state.get("system_info")
        if sys_info:
            prompt_parts.append(
                "\n\n### MEASUREMENT METADATA:\n"
                + json.dumps(sys_info, default=str)[:1500])

        # 2c. Recorded global scalars — the numbers actually delivered, so
        # quoted values in the draft can be cross-checked against the record.
        _scalars = []
        for it in state.get("all_iteration_results") or []:
            for m in (it or {}).get("custom_analysis_metadata_list") or []:
                if isinstance(m, dict) and isinstance(m.get("scalar"), (int, float)):
                    _scalars.append(
                        f"- {m.get('name')} = {m['scalar']:.6g} "
                        f"{m.get('units', '')}".rstrip())
        if _scalars:
            prompt_parts.append(
                "\n\n### RECORDED GLOBAL SCALARS (the delivered numbers):\n"
                + "\n".join(_scalars))

        # 3. Add Evidence (Images)
        # The critic needs to see the data to know if the text is lying.
        prompt_parts.append("\n\n### VISUAL EVIDENCE:")
        analysis_images = state.get("analysis_images", [])
        if not analysis_images:
            prompt_parts.append("(No images available for verification)")

        for img in analysis_images:
            image_bytes = img.get('data') or img.get('bytes')
            label = img.get('label', 'Unknown Plot')
            if image_bytes:
                prompt_parts.append(f"\n**{label}**")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})

        # 3b. Auxiliary companions (topography, reference channels): claims
        # about registry/correlation with these are only checkable if the
        # critic can see them.
        for aux in state.get("auxiliary_items") or []:
            label = aux.get("label") or "auxiliary"
            summary = aux.get("summary") or ""
            blob = aux.get("plot_bytes")
            prompt_parts.append(
                f"\n**Auxiliary: {label}**" + (f" — {summary}" if summary else ""))
            if blob:
                prompt_parts.append({
                    "mime_type": aux.get("mime_type") or "image/jpeg",
                    "data": blob})

        # 4. Run Model
        try:
            param_gen_config = None
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            review_json, error = self._parse_llm_response(response)

            if error:
                self.logger.warning("Reflection failed to parse. Assuming approval.")
                state["reflection_result"] = {"status": "approved"}
            else:
                state["reflection_result"] = review_json
                self.logger.info(f"✅ Reflection Complete. Status: {review_json.get('status')}")
                if review_json.get('status') != 'approved':
                    self.logger.info(f"   Critique: {review_json.get('critique')}")

        except Exception as e:
            self.logger.error(f"Reflection step crashed: {e}")
            state["reflection_result"] = {"status": "approved"} # Fail open

        return state


class ApplyReflectionUpdatesController:
    """
    [🧠 EDITOR Step]
    Applies the changes suggested by the critic, if any.

    Modality-agnostic; moved verbatim from the hyperspectral controllers
    (issue #327 phase 3). ``instructions`` defaults to the hyperspectral
    reflection-update prompt so the original pipeline behavior is unchanged.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn,
                 instructions: Optional[str] = None):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        if instructions is None:
            from ..instruct import SPECTROSCOPY_REFLECTION_UPDATE_INSTRUCTIONS
            instructions = SPECTROSCOPY_REFLECTION_UPDATE_INSTRUCTIONS
        self.instructions = instructions

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state

        review = state.get("reflection_result", {})
        if review.get("status") == "approved":
            self.logger.info("⏩ No revisions needed. Proceeding to report generation.")
            return state

        self.logger.info("\n\n🧠 --- REFINEMENT: APPLYING CRITICAL UPDATES --- 🧠\n")

        # 1. Setup Context
        original_result = state.get("result_json")
        critique_text = review.get("critique", "No critique provided.")

        prompt_parts = [self.instructions]
        prompt_parts.append(f"\n\n### CRITICAL REVIEW:\n{critique_text}")
        prompt_parts.append(f"\n\n### ORIGINAL DRAFT:\n{json.dumps(original_result, indent=2)}")

        # We re-attach images so the editor can verify what needs to be changed
        # (e.g., "Remove discussion of Component 3")
        prompt_parts.append("\n\n### VISUAL CONTEXT (For Reference):")
        for img in state.get("analysis_images", []):
            image_bytes = img.get('data') or img.get('bytes')
            label = img.get('label', 'Unknown Plot')
            if image_bytes:
                prompt_parts.append(f"\n**{label}**")
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})

        # 2. Run Model
        try:
            param_gen_config = None
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            updated_json, error = self._parse_llm_response(response)

            if not error and updated_json:
                # OVERWRITE the result
                state["result_json"] = updated_json
                self.logger.info("✅ Analysis updated based on self-reflection.")
            else:
                self.logger.warning("Failed to parse updated analysis. Keeping original draft.")

        except Exception as e:
            self.logger.error(f"Refinement step crashed: {e}")
            # Do not overwrite state['result_json'], just keep the old one

        return state


class SynthesisReEntryController:
    """
    [🧠 EDITOR Step] Tier-A synthesis re-entry (issue #322).

    Re-runs ONLY the interpretation over a completed analysis result, with an
    injected :class:`~scilink.agents.exp_agents._critique.CritiquePayload`.
    The payload source is pluggable — a human critique, an automated
    verifier/critic, or a literature search (#323) are interchangeable
    producers of the same payload.

    Unlike :class:`ApplyReflectionUpdatesController` (the in-pipeline editor,
    which overwrites ``state["result_json"]`` mid-run), this controller
    produces a *revision*: the original result is never mutated. Callers
    (``BaseAnalysisAgent.reenter_interpretation``, the orchestrator's
    ``refine_interpretation`` tool) append the revision to an append-only
    ``interpretation_revisions`` list.
    """

    def __init__(self, model, logger, generation_config, safety_settings, parse_fn,
                 instructions: Optional[str] = None):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.instructions = instructions or SYNTHESIS_REENTRY_INSTRUCTIONS

    def revise(self, prior_result: dict, payload, *, features_block: str = "",
               images: Optional[list] = None,
               system_info: Optional[dict] = None) -> tuple:
        """Produce a revised interpretation. Returns ``(revision, error)``.

        ``revision`` is ``{detailed_analysis, scientific_claims,
        revision_summary, source, critique}`` — the caller owns storage.
        The prior result is read-only here.
        """
        source = getattr(payload, "source", "unknown")
        critique = getattr(payload, "critique", str(payload))
        hints = getattr(payload, "hints", None)

        self.logger.info(
            f"\n🔁 --- SYNTHESIS RE-ENTRY (source: {source}) --- 🔁\n"
        )

        prompt_parts = [self.instructions]
        prompt_parts.append("\n\n### ORIGINAL INTERPRETATION:\n"
                            f"{prior_result.get('detailed_analysis', '')}")
        claims = prior_result.get("scientific_claims", [])
        if claims:
            prompt_parts.append(
                f"\n\n### ORIGINAL CLAIMS:\n{json.dumps(claims, indent=2)}"
            )
        if features_block:
            prompt_parts.append(
                f"\n\n### EXTRACTED FEATURES (measured by the analysis):\n{features_block}"
            )
        prompt_parts.append(
            f"\n\n### CRITIQUE / CONTEXT PAYLOAD (source: {source}):\n{critique}"
        )
        if hints:
            prompt_parts.append(
                f"\n\n### STRUCTURED HINTS:\n{json.dumps(hints, indent=2, default=str)}"
            )
        if system_info:
            prompt_parts.append(
                f"\n\n### SYSTEM INFORMATION:\n{json.dumps(system_info, indent=2, default=str)}"
            )
        for img in images or []:
            image_bytes = img.get("data") or img.get("bytes")
            if image_bytes:
                prompt_parts.append(f"\n**{img.get('label', 'Analysis figure')}**")
                prompt_parts.append({"mime_type": img.get("mime_type", "image/jpeg"),
                                     "data": image_bytes})
        prompt_parts.append("\n\nProvide the revised interpretation in the requested JSON format.")

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error = self._parse_llm_response(response)
            if error or not result_json:
                return None, (error or {"error": "Empty re-entry response"})
            revision = {
                "detailed_analysis": result_json.get("detailed_analysis", ""),
                "scientific_claims": result_json.get("scientific_claims", []),
                "revision_summary": result_json.get("revision_summary", ""),
                "source": source,
                "critique": critique,
            }
            if not revision["detailed_analysis"]:
                return None, {"error": "Re-entry returned no detailed_analysis"}
            self.logger.info(
                f"✅ Re-entry complete: {revision['revision_summary'][:120]}"
            )
            return revision, None
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Synthesis re-entry failed: {e}")
            return None, {"error": "Synthesis re-entry failed", "details": str(e)}
