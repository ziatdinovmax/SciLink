import logging
import os
import textwrap
from typing import Any, Callable
import json


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