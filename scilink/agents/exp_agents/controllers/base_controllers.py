import logging
import textwrap
from typing import Callable
import json


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
        self.logger.info("🧠 LLM Step: Generating final scientific interpretation...")
        prompt = state.get("final_prompt_parts")
        
        if not prompt:
            self.logger.error("Pipeline reached final step, but no 'final_prompt_parts' in state.")
            state["error_dict"] = {"error": "Pipeline failed to build final prompt"}
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
                self.logger.info("✅ LLM Step Complete: Final analysis generated.")
            else:
                self.logger.error(f"❌ LLM Step Failed: {error_dict.get('details')}")

        except Exception as e:
            self.logger.exception(f"❌ LLM Step Failed: {e}")
            state["result_json"] = None
            state["error_dict"] = {"error": "Final LLM analysis failed", "details": str(e)}

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
        iteration_title = state.get("iteration_title", "Current Analysis")
        analysis_text = state.get("result_json", {}).get("detailed_analysis", "No analysis text provided.")
        targets = decision.get("targets", [])
        
        print("\n" + "="*80)
        print(f"🎯 ANALYSIS STEP REVIEW: {iteration_title}")
        print("="*80)
        print("\n**SUMMARY OF CURRENT ANALYSIS:**")
        print(analysis_text)
        print("-" * 80)
        
        print(f"🧠 LLM's Proposed Plan: Refinement Needed = **{decision.get('refinement_needed', False)}**")
        print(f"Reasoning: {decision.get('reasoning', 'N/A')}")
        print()
        print("-" * 80)
        print(f"\U0001f3af Targeted Actions ({len(targets)} found)")
        print("-" * 80)

        if not targets:
            print("  (No specific targets were generated.)")

        for i, t in enumerate(targets, 1):
            t_type = t.get('type', 'N/A')
            t_value = t.get('value', 'N/A')
            t_desc = t.get('description', 'No description provided.')

            print(f"\n  [{i}] {t_type}  (value: {t_value})")
            for line in textwrap.wrap(t_desc, width=70):
                print(f"      {line}")

        print("-" * 80)
        
        try:
            user_feedback = input("\n🤔 Your feedback to adjust the targets/plan (or press Enter to accept): ").strip()
        except KeyboardInterrupt:
            self.logger.warning("User interrupted feedback. Accepting original decision.")
            return state

        if not user_feedback:
            self.logger.info("✅ User accepted original refinement decision.")
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