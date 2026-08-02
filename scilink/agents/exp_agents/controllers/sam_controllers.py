import json
import logging
from ....tools.sam import (
    run_sam_analysis, 
    visualize_sam_results, 
    calculate_sam_statistics,
    save_sam_visualization
)
from ....tools.image_processor import convert_numpy_to_jpeg_bytes
from ..instruct import SAM_ANALYSIS_REFINE_INSTRUCTIONS


class RunSAMRefinementLoopController:
    """
    [🛠️ Tool Step + 🧠 LLM Step]
    Runs the entire SAM analysis and refinement loop. 
    It contains its own internal LLM calls.
    """
    def __init__(self, model, logger, generation_config, safety_settings, settings: dict, parse_fn: callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.settings = settings
        self._parse_llm_response = parse_fn
        self.refinement_cycles = self.settings.get('refinement_cycles', 0)
        self.save_visualizations = self.settings.get('save_visualizations', True)

    def _llm_get_refinement_params(self, original_image_bytes, overlay_image_bytes, particle_count, current_params) -> dict | None:
        """
        (Private) This is the LLM call, now correctly
        encapsulated *inside the controller*.
        """
        try:
            self.logger.info("   (Loop 🧠: Calling LLM for refinement parameters...)")
            
            prompt_parts = [SAM_ANALYSIS_REFINE_INSTRUCTIONS]
            prompt_parts.append(f"\n\nCurrent Analysis Results:")
            prompt_parts.append(f"- Particle count: {particle_count}")
            prompt_parts.append(f"- Current parameters: {json.dumps(current_params, indent=2)}")
            prompt_parts.append(f"\n\n**ORIGINAL MICROSCOPY IMAGE (for reference):**")
            prompt_parts.append({"mime_type": "image/jpeg", "data": original_image_bytes})
            prompt_parts.append(f"\n\n**CURRENT SEGMENTATION RESULT:**")
            prompt_parts.append({"mime_type": "image/jpeg", "data": overlay_image_bytes})
            prompt_parts.append("\n\n**ANALYSIS TASK:**")
            prompt_parts.append("Compare the segmentation result against the original image. Provide refined parameters to improve accuracy.")
            
            # Using self.generation_config passed from Agent (which is None, effectively default settings)
            # If you want specific params (like low temp), use a dict:
            # config = {"temperature": 0.2, "response_mime_type": "application/json"}
            
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config, 
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"   (Loop 🧠: ❌ LLM refinement call failed: {error_dict})")
                return None
            
            reasoning = result_json.get("reasoning", "No reasoning provided")
            new_parameters = result_json.get("parameters", {})
            
            self.logger.info(f"   (Loop 🧠: LLM refinement reasoning: {reasoning})")
            
            expected_keys = {"use_clahe", "sam_parameters", "min_area", "max_area", "pruning_iou_threshold"}
            if not all(key in new_parameters for key in expected_keys):
                self.logger.warning(f"   (Loop 🧠: ❌ Invalid parameter set from LLM, missing keys.")
                return None
            
            new_parameters["use_pruning"] = True # Always enable this
            return new_parameters
            
        except Exception as e:
            self.logger.error(f"   (Loop 🧠: ❌ Error in LLM refinement: {e})")
            return None

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n🛠️ --- CALLING TOOL: PARTICLE ANALYZER WITH SEGMENT ANYTHING MODEL --- 🛠️\n")
        
        try:
            image_array = state["preprocessed_image_array"]
            original_image_bytes = state["image_blob"]["data"] # Use the same preprocessed blob
            
            # 1. Get initial parameters from agent settings
            current_params = {
                "checkpoint_path": self.settings.get('checkpoint_path', None),
                "model_type": self.settings.get('model_type', 'vit_h'),
                "device": self.settings.get('device', 'auto'),
                "use_clahe": self.settings.get('use_clahe', False),
                "sam_parameters": self.settings.get('sam_parameters', 'default'),
                "min_area": self.settings.get('min_area', 500),
                "max_area": self.settings.get('max_area', 50000),
                "use_pruning": self.settings.get('use_pruning', True),
                "pruning_iou_threshold": self.settings.get('pruning_iou_threshold', 0.5)
            }
            
            # 2. Run initial analysis
            self.logger.info(f"   (Loop 🛠️: Running initial SAM analysis...)")
            sam_result = run_sam_analysis(image_array, params=current_params)
            state["sam_result"] = sam_result # Store the first result
            
            if self.save_visualizations:
                initial_overlay = visualize_sam_results(sam_result)
                save_sam_visualization(initial_overlay, "initial", 0, sam_result['total_count'], current_params, self.logger)
            
            # 3. --- Refinement Loop ---
            for cycle in range(self.refinement_cycles):
                self.logger.info(f"   (Loop 🔄: Starting refinement cycle {cycle + 1}/{self.refinement_cycles}...)")
                
                current_overlay_img = visualize_sam_results(sam_result)
                current_overlay_bytes = convert_numpy_to_jpeg_bytes(current_overlay_img)
                
                # Call this controller's *own* private LLM method
                new_params = self._llm_get_refinement_params(
                    original_image_bytes,
                    current_overlay_bytes,
                    sam_result['total_count'],
                    current_params
                )
                
                if new_params is None or new_params == current_params:
                    self.logger.info("   (Loop 🔄: No valid parameter changes suggested. Stopping refinement.)")
                    break
                
                # Merge new params with persistent tool params
                current_params.update(new_params) 
                
                self.logger.info(f"   (Loop 🛠️: Re-running analysis with new params...)")
                sam_result = run_sam_analysis(image_array, params=current_params)
                state["sam_result"] = sam_result # Overwrite with the latest result
                
                if self.save_visualizations:
                    refined_overlay = visualize_sam_results(sam_result)
                    save_sam_visualization(refined_overlay, "refined", cycle + 1, sam_result['total_count'], current_params, self.logger)
            
            self.logger.info("✅ SAM Workflow Complete.")

        except Exception as e:
            self.logger.error(f"❌ SAM Workflow Failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "SAM Analysis Workflow failed", "details": str(e)}

        return state

class CalculateSAMStatsController:
    """
    [🛠️ Tool Step]
    Takes the final 'sam_result' from the state, calculates stats,
    and puts them in 'summary_stats'.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("🛠️ Tool Step: Calling 'calculate_sam_statistics'...")
        sam_result = state.get("sam_result")
        
        if sam_result is None:
            self.logger.warning("   (Tool Info: No 'sam_result' in state. Skipping stats.)")
            state["summary_stats"] = {"total_particles": 0, "error": "Analysis failed in previous step."}
            return state

        try:
            summary_stats = calculate_sam_statistics(
                sam_result=sam_result,
                image_path=state["image_path"],
                preprocessed_image_shape=state["preprocessed_image_array"].shape,
                nm_per_pixel=state.get("nm_per_pixel")
            )
            state["summary_stats"] = summary_stats
            self.logger.info("✅ Tool Step Complete: Statistics calculated.")
        except Exception as e:
            self.logger.error(f"❌ Tool Step Failed: Stats calculation failed: {e}")
            state["summary_stats"] = {"total_particles": 0, "error": str(e)}
            
        return state

class BuildSAMPromptController:
    """
    [📝 Prep Step]
    Builds the final prompt, adding the SAM overlay and stats.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("📝 Prep Step: Building final prompt with SAM results...")
        
        prompt_parts = [state["instruction_prompt"]]
        
        if state.get("additional_top_level_context"):
            prompt_parts.append(f"\n\n## Special Considerations:\n{state['additional_top_level_context']}\n")
            
        prompt_parts.append("\n\nPrimary Microscopy Image:\n")
        prompt_parts.append(state["image_blob"])

        sam_result = state.get("sam_result")
        sam_stats = state.get("summary_stats")

        if sam_result is not None and sam_stats is not None:
            # Generate the *final* overlay for the prompt
            final_overlay_img = visualize_sam_results(sam_result)
            overlay_bytes = convert_numpy_to_jpeg_bytes(final_overlay_img)
            
            prompt_parts.append("\n\nSupplemental SAM Particle Segmentation Analysis:")
            prompt_parts.append(f"Detected {sam_stats.get('total_particles', 0)} particles.")
            
            prompt_parts.append("\n**Morphological Statistics Summary:**")
            for key, value in sam_stats.items():
                if isinstance(value, (int, float, str, list)):
                    prompt_parts.append(f"- {key}: {value}")
                elif isinstance(value, dict):
                    prompt_parts.append(f"- {key}: {json.dumps(value)}")

            prompt_parts.append("\nSAM Particle Segmentation Overlay (particles outlined in red):")
            prompt_parts.append({"mime_type": "image/jpeg", "data": overlay_bytes})
            
            state["analysis_images"].append({
                "label": "SAM Particle Segmentation Overlay",
                "data": overlay_bytes
            })
        else:
            prompt_parts.append("\n\n(No supplemental SAM analysis was run or it failed)")

        prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(state['system_info'], indent=2)}")
        prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        self.logger.info("✅ Prep Step Complete: Final prompt is ready.")
        return state