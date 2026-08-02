"""
SAM Microscopy Analysis Controllers

This module contains unified controllers that handle both single image (n=1)
and batch (n>1) analysis identically. The key principle is:

    Single image = Batch of 1

All controllers adapt their behavior based on state["is_single_image"]
and state["num_images"], but use the same code paths.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Any, Dict
import numpy as np

from PIL import Image

from ..instruct import (
    SAM_BATCH_REFINEMENT_INSTRUCTIONS,
    SAM_BATCH_SYNTHESIS_INSTRUCTIONS,
    SAM_BATCH_CUSTOM_ANALYSIS_INSTRUCTIONS,
    SAM_SINGLE_IMAGE_SYNTHESIS_INSTRUCTIONS,
)

from ....tools.image_processor import (
    load_image,
    preprocess_image,
    convert_numpy_to_jpeg_bytes
)
from ....tools.sam import (
    get_or_create_sam_model,
    run_sam_analysis,
    visualize_sam_results,
    calculate_sam_statistics,
    save_sam_visualization
)


# ============================================================================
# AUTOMATED LLM REFINEMENT CONTROLLER
# (Runs when human feedback is disabled - LLM acts as the reviewer)
# ============================================================================

class AutomatedLLMRefinementController:
    """
    [🧠 LLM Step]
    Automated quality gate that runs when human feedback is DISABLED.
    
    The LLM evaluates the initial segmentation and decides whether to:
    - Accept the current results and proceed
    - Refine parameters and re-run SAM analysis
    
    KEY FEATURE: Iteration history + LLM Judge
    
    Every iteration's results (params, SAM output, stats, LLM evaluation)
    are stored in a history. If refinement doesn't converge to an accepted
    result within max iterations, an LLM "judge" reviews ALL iterations
    and selects the best one. This prevents two failure modes:
    
    1. Max iterations reached with poor results → judge picks the best
    2. Non-monotonic improvement (iter 1 good, iter 2 worse) → judge picks iter 1
    
    Works identically for single images and batches - always evaluates
    the first image as a representative sample.
    """
    
    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        settings: dict
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.max_auto_iterations = settings.get('max_auto_refinement_iterations', 2)
    
    def _get_llm_evaluation(self, state: dict, iteration_history: list) -> dict:
        """
        Ask the LLM to evaluate segmentation quality and decide whether
        to accept or refine parameters.
        
        Args:
            state: Current pipeline state
            iteration_history: List of previous iteration records, so the LLM
                               can see what has already been tried and avoid
                               repeating failed strategies.
        
        Returns:
            dict with keys:
                - decision: "accept" | "refine"
                - reasoning: str
                - recommended_parameters: dict (only if decision == "refine")
                - evaluation: dict with quality scores
        """
        self.logger.info("   🧠 LLM evaluating segmentation quality...")
        
        original_image_bytes = state["image_blob"]["data"]
        sam_result = state.get("sam_result")
        sam_stats = state.get("summary_stats", {})
        current_params = state.get("current_params", {})
        
        # Generate overlay visualization for LLM review
        overlay_img = visualize_sam_results(sam_result)
        overlay_bytes = convert_numpy_to_jpeg_bytes(overlay_img)
        
        # Build history context so LLM knows what was already tried
        history_context = ""
        if iteration_history:
            history_lines = []
            for record in iteration_history:
                eval_info = record.get("evaluation", {})
                history_lines.append(
                    f"  - Iteration {record['iteration']}: "
                    f"params={json.dumps(record['params_used'], indent=None)}, "
                    f"particles={record.get('particle_count', '?')}, "
                    f"coverage={eval_info.get('coverage_score', '?')}/10, "
                    f"accuracy={eval_info.get('accuracy_score', '?')}/10, "
                    f"quality={eval_info.get('overall_quality', '?')}, "
                    f"decision={record.get('decision', '?')}"
                )
            history_context = (
                "\n\n**PREVIOUS ITERATIONS (do NOT repeat parameters that already failed):**\n"
                + "\n".join(history_lines)
            )
        
        prompt_parts = [
            SAM_BATCH_REFINEMENT_INSTRUCTIONS,
            "\n\n**CONTEXT:** You are the automated quality reviewer. There is no "
            "human in the loop. You must decide whether the segmentation is acceptable "
            "or needs refinement. Be especially vigilant about:\n"
            "- Zero or very low particle counts (may indicate wrong parameters)\n"
            "- Extremely high counts (may indicate over-segmentation)\n"
            "- Particles that are clearly artifacts or noise\n"
            "- Large regions of the image with obvious features that were missed\n",
            "\n\n**ORIGINAL MICROSCOPY IMAGE:**",
            {"mime_type": "image/jpeg", "data": original_image_bytes},
            "\n\n**CURRENT SEGMENTATION RESULT:**",
            {"mime_type": "image/jpeg", "data": overlay_bytes},
            f"\n\n**MORPHOLOGICAL STATISTICS:**\n{json.dumps(sam_stats, indent=2)}",
            f"\n\n**CURRENT PARAMETERS:**\n{json.dumps(current_params, indent=2)}",
            *([history_context] if history_context else []),
            "\n\nRespond with JSON containing:\n"
            "- \"decision\": \"accept\" or \"refine\"\n"
            "- \"reasoning\": your analysis of the segmentation quality\n"
            "- \"evaluation\": {\"coverage_score\": 1-10, \"accuracy_score\": 1-10, "
            "\"overall_quality\": \"good\"/\"acceptable\"/\"poor\"}\n"
            "- \"recommended_parameters\": {param changes} (only if decision is \"refine\")\n"
        ]
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM evaluation failed: {error_dict}")
                return {"decision": "accept", "reasoning": "Evaluation failed, proceeding with current results"}
            
            # Validate the response has required fields
            if "decision" not in result_json:
                if result_json.get("needs_refinement"):
                    result_json["decision"] = "refine"
                else:
                    result_json["decision"] = "accept"
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error in LLM evaluation: {e}")
            return {"decision": "accept", "reasoning": f"Evaluation error: {str(e)}"}
    
    def _judge_select_best_iteration(self, state: dict, iteration_history: list) -> int:
        """
        LLM Judge: Reviews all iteration results and selects the best one.
        
        Called when:
        - Max iterations exhausted without an "accept" decision
        - Acts as a safety net against non-monotonic refinement
        
        Args:
            state: Current pipeline state (has original image for reference)
            iteration_history: Complete list of iteration records
        
        Returns:
            Index into iteration_history of the best iteration (0-based)
        """
        self.logger.info("\n   ⚖️  LLM JUDGE: Reviewing all iterations to select the best result...\n")
        
        original_image_bytes = state["image_blob"]["data"]
        
        # Build comparison data for all iterations
        iterations_summary = []
        overlay_parts = []
        
        for record in iteration_history:
            iter_num = record["iteration"]
            eval_info = record.get("evaluation", {})
            
            iterations_summary.append({
                "iteration": iter_num,
                "particle_count": record.get("particle_count", 0),
                "parameters": record["params_used"],
                "coverage_score": eval_info.get("coverage_score"),
                "accuracy_score": eval_info.get("accuracy_score"),
                "overall_quality": eval_info.get("overall_quality"),
                "reasoning": record.get("reasoning", ""),
            })
            
            # Include overlay images for visual comparison
            overlay_bytes = record.get("overlay_bytes")
            if overlay_bytes:
                overlay_parts.append(f"\n\n**ITERATION {iter_num} SEGMENTATION:**")
                overlay_parts.append({"mime_type": "image/jpeg", "data": overlay_bytes})
        
        prompt_parts = [
            "You are a quality judge for microscopy image segmentation. "
            "Multiple parameter configurations have been tried. Your job is to "
            "select the BEST iteration — the one that most accurately segments "
            "the particles/features in the original image.\n\n"
            "Consider:\n"
            "- Coverage: Are all real particles detected?\n"
            "- Accuracy: Are detected regions actual particles (not noise/artifacts)?\n"
            "- Particle count plausibility: Does the count make sense for this image?\n"
            "- Area statistics: Are the detected sizes reasonable?\n\n"
            "If ALL iterations are poor quality, still pick the least-bad one.\n",
            "\n\n**ORIGINAL MICROSCOPY IMAGE:**",
            {"mime_type": "image/jpeg", "data": original_image_bytes},
        ]
        
        # Add all overlay images
        prompt_parts.extend(overlay_parts)
        
        prompt_parts.append(
            f"\n\n**ITERATION DETAILS:**\n{json.dumps(iterations_summary, indent=2)}"
        )
        
        prompt_parts.append(
            "\n\nRespond with JSON containing:\n"
            "- \"selected_iteration\": the iteration number (1-based) of the best result\n"
            "- \"reasoning\": why this iteration is the best\n"
            "- \"confidence\": \"high\"/\"medium\"/\"low\" — how confident you are in this selection\n"
            "- \"quality_warning\": optional string if even the best result has significant issues\n"
        )
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict or not result_json:
                self.logger.warning(f"Judge evaluation failed: {error_dict}. Falling back to highest-scored iteration.")
                return self._fallback_select_best(iteration_history)
            
            selected = result_json.get("selected_iteration")
            reasoning = result_json.get("reasoning", "No reasoning provided")
            confidence = result_json.get("confidence", "unknown")
            quality_warning = result_json.get("quality_warning")
            
            self.logger.info(f"   ⚖️  Judge selected iteration {selected} (confidence: {confidence})")
            self.logger.info(f"   💬 Reasoning: {reasoning}")
            
            if quality_warning:
                self.logger.warning(f"   ⚠️  Quality warning: {quality_warning}")
            
            # Convert 1-based iteration number to 0-based index
            if isinstance(selected, int) and 1 <= selected <= len(iteration_history):
                return selected - 1
            else:
                self.logger.warning(f"   ⚠️  Invalid selection '{selected}'. Falling back to scoring.")
                return self._fallback_select_best(iteration_history)
                
        except Exception as e:
            self.logger.error(f"Judge error: {e}. Falling back to highest-scored iteration.")
            return self._fallback_select_best(iteration_history)
    
    def _fallback_select_best(self, iteration_history: list) -> int:
        """
        Deterministic fallback: select the iteration with the highest
        combined coverage + accuracy score. Used when the LLM judge fails.
        """
        best_idx = 0
        best_score = -1
        
        for i, record in enumerate(iteration_history):
            eval_info = record.get("evaluation", {})
            coverage = eval_info.get("coverage_score", 0)
            accuracy = eval_info.get("accuracy_score", 0)
            
            # Ensure numeric
            try:
                combined = float(coverage) + float(accuracy)
            except (TypeError, ValueError):
                combined = 0
            
            if combined > best_score:
                best_score = combined
                best_idx = i
        
        self.logger.info(f"   📊 Fallback selected iteration {best_idx + 1} (score: {best_score})")
        return best_idx
    
    def _restore_iteration(self, state: dict, record: dict) -> dict:
        """Restore state to a specific iteration's results."""
        state["current_params"] = record["params_used"].copy()
        state["sam_result"] = record["sam_result"]
        state["summary_stats"] = record["summary_stats"]
        state["final_params_for_batch"] = record["params_used"].copy()
        return state
    
    def execute(self, state: dict) -> dict:
        """
        Execute automated LLM refinement loop with iteration tracking
        and judge-based best-selection.
        
        Flow:
        1. Evaluate current segmentation
        2. If accepted → use it, done
        3. If refine → update params, re-run SAM, record iteration, loop
        4. If max iterations reached without accept → LLM judge picks best
        
        Only runs when human feedback is DISABLED.
        """
        if state.get('enable_human_feedback', False):
            self.logger.info("Human feedback enabled — skipping automated LLM refinement.")
            return state
        
        is_single = state.get("is_single_image", False)
        mode_str = "SINGLE IMAGE" if is_single else "BATCH"
        self.logger.info(f"\n\n🤖 --- AUTOMATED LLM QUALITY GATE ({mode_str}) --- 🤖\n")
        
        sam_result = state.get("sam_result")
        if sam_result and sam_result.get("total_count", 0) == 0:
            self.logger.warning("⚠️  Initial SAM detected 0 particles — LLM will evaluate.")
        
        # =====================================================================
        # Track all iterations for the judge
        # =====================================================================
        iteration_history = []
        accepted = False
        
        iteration = 0
        while iteration < self.max_auto_iterations:
            iteration += 1
            self.logger.info(f"   📋 Evaluation iteration {iteration}/{self.max_auto_iterations}")
            
            # Snapshot current state BEFORE evaluation
            current_sam_result = state.get("sam_result")
            current_stats = state.get("summary_stats", {})
            current_params = state.get("current_params", {}).copy()
            
            # Generate overlay for this iteration (stored for judge)
            overlay_img = visualize_sam_results(current_sam_result)
            overlay_bytes = convert_numpy_to_jpeg_bytes(overlay_img)
            
            # Get LLM evaluation (with history context)
            evaluation = self._get_llm_evaluation(state, iteration_history)
            
            decision = evaluation.get("decision", "accept")
            reasoning = evaluation.get("reasoning", "No reasoning provided")
            quality = evaluation.get("evaluation", {})
            
            self.logger.info(f"   📊 Quality: coverage={quality.get('coverage_score', '?')}/10, "
                           f"accuracy={quality.get('accuracy_score', '?')}/10, "
                           f"overall={quality.get('overall_quality', '?')}")
            self.logger.info(f"   💬 Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
            self.logger.info(f"   🎯 Decision: {decision.upper()}")
            
            # Record this iteration
            iteration_record = {
                "iteration": iteration,
                "params_used": current_params,
                "sam_result": current_sam_result,
                "summary_stats": current_stats,
                "particle_count": current_sam_result.get("total_count", 0) if current_sam_result else 0,
                "evaluation": quality,
                "reasoning": reasoning,
                "decision": decision,
                "overlay_bytes": overlay_bytes,
            }
            iteration_history.append(iteration_record)
            
            if decision == "accept":
                self.logger.info("   ✅ LLM accepted segmentation quality.")
                state["llm_refinement_iterations"] = iteration
                state["llm_quality_evaluation"] = quality
                state["final_params_for_batch"] = current_params
                state["refinement_history"] = self._sanitize_history_for_state(iteration_history)
                accepted = True
                break
            
            elif decision == "refine":
                recommended_params = evaluation.get("recommended_parameters", {})
                
                if not recommended_params:
                    self.logger.warning("   ⚠️ LLM recommended refinement but provided no parameters.")
                    # Don't break — let the judge decide if this was the best iteration
                    continue
                
                self.logger.info(f"   🔧 Applying recommended parameters: {json.dumps(recommended_params, indent=2)}")
                
                # Update parameters
                updated_params = state.get("current_params", {}).copy()
                updated_params.update(recommended_params)
                state["current_params"] = updated_params
                
                # Re-run SAM analysis
                try:
                    image_array = state["preprocessed_image_array"]
                    new_sam_result = run_sam_analysis(image_array, params=updated_params)
                    state["sam_result"] = new_sam_result
                    
                    summary_stats = calculate_sam_statistics(
                        sam_result=new_sam_result,
                        image_path=state["image_path"],
                        preprocessed_image_shape=image_array.shape,
                        nm_per_pixel=state.get("nm_per_pixel")
                    )
                    state["summary_stats"] = summary_stats
                    
                    old_count = current_sam_result.get("total_count", 0) if current_sam_result else 0
                    new_count = new_sam_result.get("total_count", 0)
                    self.logger.info(f"   🔄 Re-analysis: {old_count} → {new_count} particles")
                    
                except Exception as e:
                    self.logger.error(f"   ❌ Re-analysis failed: {e}")
                    # Revert to pre-refinement params so next iteration or judge
                    # can still work with the last good result
                    state["current_params"] = current_params
                    state["sam_result"] = current_sam_result
                    state["summary_stats"] = current_stats
                    continue
            else:
                self.logger.warning(f"   ⚠️ Unknown decision '{decision}'.")
                continue
        
        # =====================================================================
        # JUDGE: If we exhausted iterations without an accepted result
        # =====================================================================
        if not accepted:
            self.logger.warning(
                f"\n   ⚠️ Max auto-refinement iterations ({self.max_auto_iterations}) "
                f"reached without acceptance. Invoking LLM judge..."
            )
            
            if len(iteration_history) == 1:
                # Only one iteration — no need for a judge, just use it
                self.logger.info("   📋 Only one iteration attempted — using it directly.")
                best_idx = 0
            else:
                best_idx = self._judge_select_best_iteration(state, iteration_history)
            
            best_record = iteration_history[best_idx]
            self.logger.info(
                f"\n   🏆 Judge selected iteration {best_record['iteration']} "
                f"({best_record['particle_count']} particles)"
            )
            
            # Restore state to the best iteration's results
            state = self._restore_iteration(state, best_record)
            state["llm_refinement_iterations"] = len(iteration_history)
            state["llm_quality_evaluation"] = best_record.get("evaluation", {})
            state["judge_selected_iteration"] = best_record["iteration"]
            state["judge_invoked"] = True
            state["refinement_history"] = self._sanitize_history_for_state(iteration_history)
        else:
            state["judge_invoked"] = False
        
        return state
    
    def _sanitize_history_for_state(self, iteration_history: list) -> list:
        """
        Create a serializable version of iteration history for state/results.
        Removes large binary data (overlay_bytes, sam_result masks) that
        shouldn't be serialized to JSON.
        """
        sanitized = []
        for record in iteration_history:
            sanitized.append({
                "iteration": record["iteration"],
                "params_used": record["params_used"],
                "particle_count": record.get("particle_count", 0),
                "evaluation": record.get("evaluation", {}),
                "reasoning": record.get("reasoning", ""),
                "decision": record.get("decision", ""),
            })
        return sanitized


# ============================================================================
# HUMAN FEEDBACK REFINEMENT CONTROLLER
# (Same for single and batch - refines parameters on first image)
# ============================================================================

class HumanFeedbackRefinementController:
    """
    [👤 Human Step + 🧠 LLM Step]
    Facilitates human-in-the-loop parameter refinement for the first image.
    
    Works identically for single images and batches:
    - Single image: Refine params, then process that one image
    - Batch: Refine params on first image, then apply to all
    
    NOTE: Only runs when enable_human_feedback=True.
    When disabled, AutomatedLLMRefinementController handles quality evaluation.
    """
    
    def __init__(
        self, 
        model, 
        logger: logging.Logger, 
        generation_config, 
        safety_settings, 
        parse_fn: Callable, 
        settings: dict
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.max_refinement_iterations = settings.get('max_feedback_iterations', 3)
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _display_analysis_for_review(self, state: dict, iteration: int) -> None:
        """Display current analysis results for human review."""
        sam_result = state.get("sam_result")
        sam_stats = state.get("summary_stats", {})
        is_single = state.get("is_single_image", False)

        overlay_img = visualize_sam_results(sam_result)
        
        review_viz_path = self.output_dir / f"review_iteration_{iteration}.png"
        review_viz_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay_img).save(review_viz_path)

        print("\n" + "=" * 80)
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({state.get('num_images', 1)} images)"
        print(f"🔬 SAM ANALYSIS REVIEW - {mode_str} - Iteration {iteration}")
        print("=" * 80)

        print(f"\n🖼️  **Visualization saved to:** {review_viz_path}")
        print(f"   Open this file to see the segmentation overlay.\n")
        
        print(f"\n📊 **Detection Summary (First Image):**")
        print(f"   - Total particles detected: {sam_stats.get('total_particles', 0)}")
        
        mean_area = sam_stats.get('mean_area_pixels', 'N/A')
        std_area = sam_stats.get('std_area_pixels', 'N/A')
        if isinstance(mean_area, (int, float)):
            print(f"   - Mean area: {mean_area:.2f} px²")
        if isinstance(std_area, (int, float)):
            print(f"   - Area std: {std_area:.2f} px²")
        
        print(f"\n⚙️ **Current Parameters:**")
        current_params = state.get("current_params", {})
        for key, value in current_params.items():
            if key not in ['checkpoint_path', 'device']:
                print(f"   - {key}: {value}")
        
        if not is_single:
            print(f"\n📦 **Note:** These parameters will be applied to all {state.get('num_images', 1)} images.")
        
        print("-" * 80)
    
    def _get_llm_assessment(self, state: dict, iteration_history: list = None) -> dict:
        """
        Get LLM's assessment of the current segmentation quality.

        Args:
            state: Current pipeline state
            iteration_history: List of previous iteration records, so the LLM
                               can see what has already been tried and avoid
                               repeating failed strategies.
        """
        self.logger.info("   🧠 Getting LLM assessment of segmentation quality...")

        original_image_bytes = state["image_blob"]["data"]
        sam_result = state.get("sam_result")
        sam_stats = state.get("summary_stats", {})

        overlay_img = visualize_sam_results(sam_result)
        overlay_bytes = convert_numpy_to_jpeg_bytes(overlay_img)

        # Build history context so LLM knows what was already tried
        history_context = ""
        if iteration_history:
            history_lines = []
            for record in iteration_history:
                eval_info = record.get("evaluation", {})
                history_lines.append(
                    f"  - Iteration {record['iteration']}: "
                    f"params={json.dumps(record['params_used'], indent=None)}, "
                    f"particles={record.get('particle_count', '?')}, "
                    f"coverage={eval_info.get('coverage_score', '?')}/10, "
                    f"accuracy={eval_info.get('accuracy_score', '?')}/10, "
                    f"quality={eval_info.get('overall_quality', '?')}, "
                    f"decision={record.get('decision', '?')}"
                )
            history_context = (
                "\n\n**PREVIOUS ITERATIONS (do NOT repeat parameters that already failed):**\n"
                + "\n".join(history_lines)
            )

        prompt_parts = [
            SAM_BATCH_REFINEMENT_INSTRUCTIONS,
            "\n\n**ORIGINAL MICROSCOPY IMAGE:**",
            {"mime_type": "image/jpeg", "data": original_image_bytes},
            "\n\n**CURRENT SEGMENTATION RESULT:**",
            {"mime_type": "image/jpeg", "data": overlay_bytes},
            f"\n\n**MORPHOLOGICAL STATISTICS:**\n{json.dumps(sam_stats, indent=2)}",
            f"\n\n**CURRENT PARAMETERS:**\n{json.dumps(state.get('current_params', {}), indent=2)}",
            history_context,
        ]

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM assessment failed: {error_dict}")
                return {"needs_refinement": False, "reasoning": "Could not assess"}
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error in LLM assessment: {e}")
            return {"needs_refinement": False, "reasoning": str(e)}
    
    def _collect_human_feedback(self, llm_assessment: dict) -> dict:
        """Collect human feedback on the analysis.

        Uses a single ``input()`` call so it maps directly to the
        Streamlit UI's "Accept as-is" / "Submit feedback" buttons:
        - Empty response  → accept LLM recommendation
        - Non-empty text  → treat as natural-language parameter suggestion
        """
        print("\n🤖 **LLM Assessment:**")
        evaluation = llm_assessment.get("evaluation", {})
        print(f"   - Coverage: {evaluation.get('coverage_score', 'N/A')}/10")
        print(f"   - Accuracy: {evaluation.get('accuracy_score', 'N/A')}/10")
        print(f"   - Overall Quality: {evaluation.get('overall_quality', 'N/A')}")
        print(f"\n   Reasoning: {llm_assessment.get('reasoning', 'N/A')}")

        needs_refinement = llm_assessment.get("needs_refinement", False)
        if needs_refinement:
            print("\n   📝 LLM Recommendation: **Refine** with these parameters:")
            rec_params = llm_assessment.get("recommended_parameters", {})
            for key, value in rec_params.items():
                print(f"      - {key}: {value}")
        else:
            print("\n   ✅ LLM Recommendation: **Results look good — proceed to processing.**")

        print("\n" + "-" * 80)
        print("Press Enter to accept LLM recommendation, or type feedback to suggest changes.")

        try:
            user_feedback = input("\n🤔 Your feedback (or press Enter to accept): ").strip()
        except KeyboardInterrupt:
            self.logger.warning("User interrupted. Accepting LLM recommendation.")
            user_feedback = ""

        # Empty → accept LLM recommendation
        if not user_feedback:
            if needs_refinement:
                return {"action": "use_llm", "params": llm_assessment.get("recommended_parameters", {})}
            return {"action": "accept", "params": None}

        # Non-empty → convert natural language to parameters
        params = self._convert_feedback_to_params(user_feedback)
        if params:
            return {"action": "custom", "params": params}
        else:
            print("Could not interpret feedback. Accepting LLM recommendation.")
            if needs_refinement:
                return {"action": "use_llm", "params": llm_assessment.get("recommended_parameters", {})}
            return {"action": "accept", "params": None}
    
    def _convert_feedback_to_params(self, user_feedback: str) -> dict:
        """Use LLM to convert natural language feedback to SAM parameters."""
        self.logger.info("   🧠 Converting feedback to parameters...")
        
        current_params = {
            "use_clahe": self.settings.get('use_clahe', False),
            "sam_parameters": self.settings.get('sam_parameters', 'default'),
            "min_area": self.settings.get('min_area', 500),
            "max_area": self.settings.get('max_area', 50000),
            "pruning_iou_threshold": self.settings.get('pruning_iou_threshold', 0.5)
        }
        
        prompt = f"""Convert user feedback into SAM parameter adjustments.

**Current Parameters:**
{json.dumps(current_params, indent=2)}

**Available Parameters:**
- use_clahe (true/false): Enable contrast enhancement
- sam_parameters ("default"/"sensitive"/"ultra-permissive"): Detection sensitivity
- min_area (integer): Minimum particle size in pixels
- max_area (integer): Maximum particle size in pixels
- pruning_iou_threshold (0.0-1.0): Overlap threshold for removing duplicates

**User Feedback:**
"{user_feedback}"

Return JSON with ONLY the parameters to change:
{{"min_area": 200, "sam_parameters": "sensitive"}}
"""

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict or not result_json:
                return None
            
            print(f"\n   ✅ Interpreted as: {json.dumps(result_json, indent=2)}")
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error converting feedback: {e}")
            return None
    
    def execute(self, state: dict) -> dict:
        """Execute the human feedback refinement loop with iteration tracking."""
        if not state.get('enable_human_feedback', False):
            self.logger.info("Human feedback disabled — skipping (automated LLM handles this).")
            return state

        is_single = state.get("is_single_image", False)
        mode_str = "SINGLE IMAGE" if is_single else "BATCH"
        self.logger.info(f"\n\n👤 --- HUMAN FEEDBACK REFINEMENT ({mode_str}) --- 👤\n")

        iteration_history = []
        iteration = 0
        while iteration < self.max_refinement_iterations:
            iteration += 1

            # Snapshot current state before evaluation
            current_sam_result = state.get("sam_result")
            current_stats = state.get("summary_stats", {})
            current_params = state.get("current_params", {}).copy()

            self._display_analysis_for_review(state, iteration)
            llm_assessment = self._get_llm_assessment(state, iteration_history)

            # Record this iteration
            quality = llm_assessment.get("evaluation", {})
            iteration_record = {
                "iteration": iteration,
                "params_used": current_params,
                "particle_count": current_sam_result.get("total_count", 0) if current_sam_result else 0,
                "evaluation": quality,
                "reasoning": llm_assessment.get("reasoning", ""),
                "decision": "",  # filled below based on human choice
            }

            feedback = self._collect_human_feedback(llm_assessment)
            iteration_record["decision"] = feedback["action"]
            iteration_history.append(iteration_record)

            if feedback["action"] == "accept":
                self.logger.info("✅ User accepted current results.")
                state["refinement_complete"] = True
                state["final_params_for_batch"] = state.get("current_params", {})
                break

            elif feedback["action"] in ["use_llm", "custom"]:
                new_params = feedback["params"]
                if new_params:
                    current_params = state.get("current_params", {})
                    current_params.update(new_params)
                    state["current_params"] = current_params

                    self.logger.info(f"🔄 Re-running SAM analysis with updated parameters...")

                    try:
                        image_array = state["preprocessed_image_array"]
                        sam_result = run_sam_analysis(image_array, params=current_params)
                        state["sam_result"] = sam_result

                        summary_stats = calculate_sam_statistics(
                            sam_result=sam_result,
                            image_path=state["image_path"],
                            preprocessed_image_shape=image_array.shape,
                            nm_per_pixel=state.get("nm_per_pixel")
                        )
                        state["summary_stats"] = summary_stats

                        self.logger.info(f"✅ Re-analysis complete. {sam_result['total_count']} particles detected.")

                    except Exception as e:
                        self.logger.error(f"❌ Re-analysis failed: {e}")
                        break

        if iteration >= self.max_refinement_iterations:
            self.logger.warning(f"⚠️ Max iterations reached. Using current parameters.")
            state["final_params_for_batch"] = state.get("current_params", {})

        # Store sanitized history for downstream controllers (synthesis, reporting)
        state["refinement_history"] = [
            {
                "iteration": r["iteration"],
                "params_used": r["params_used"],
                "particle_count": r.get("particle_count", 0),
                "evaluation": r.get("evaluation", {}),
                "reasoning": r.get("reasoning", ""),
                "decision": r.get("decision", ""),
            }
            for r in iteration_history
        ]
        state["llm_refinement_iterations"] = len(iteration_history)
        if iteration_history:
            state["llm_quality_evaluation"] = iteration_history[-1].get("evaluation", {})

        return state


# ============================================================================
# UNIFIED BATCH PROCESSING CONTROLLER
# (Processes all images - whether n=1 or n>1)
# ============================================================================

class UnifiedBatchProcessingController:
    """
    [🛠️ Tool Step]
    Processes ALL images using the refined parameters.
    
    Key features:
    - Loads SAM model ONCE and reuses for all images
    - Works identically for n=1 or n>1
    - Supports both file paths and numpy array stack inputs
    - OPTIMIZATION: Skips re-analysis for single images when params unchanged
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.save_visualizations = settings.get('save_visualizations', True)
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
    
    def _get_image_and_name(self, state: dict, idx: int) -> tuple:
        """Get image array and name for a given index."""
        image_stack = state.get("image_stack")
        image_paths = state.get("image_paths")
        
        if image_stack is not None:
            image_array = image_stack[idx]
            image_name = f"frame_{idx:04d}"
            return image_array, image_name
        elif image_paths:
            image_path = image_paths[idx]
            loaded_image = load_image(image_path)
            image_name = Path(image_path).stem
            return loaded_image, image_name
        else:
            raise ValueError("No image data found in state")
    
    def _can_skip_single_image_reanalysis(self, state: dict) -> bool:
        """
        Check if we can skip re-analysis for a single image.
        
        Conditions:
        1. Single image mode (n=1)
        2. We have existing SAM results from initial analysis
        3. Parameters were not changed during feedback refinement
        """
        is_single = state.get("is_single_image", False)
        if not is_single:
            return False
        
        # Check if we have existing results
        existing_result = state.get("sam_result")
        existing_stats = state.get("summary_stats")
        if not existing_result or not existing_stats:
            return False
        
        # Check if parameters changed
        initial_params = state.get("current_params", {})
        final_params = state.get("final_params_for_batch", {})
        
        # Compare relevant parameters (exclude non-analysis params)
        analysis_keys = [
            'use_clahe', 'sam_parameters', 'min_area', 
            'max_area', 'use_pruning', 'pruning_iou_threshold'
        ]
        
        for key in analysis_keys:
            if initial_params.get(key) != final_params.get(key):
                self.logger.info(f"   Parameter '{key}' changed: {initial_params.get(key)} → {final_params.get(key)}")
                return False
        
        return True
    
    def _build_result_from_cached(self, state: dict) -> dict:
        """
        Build a batch_results entry from cached initial analysis.
        Also saves visualization if configured.
        """
        sam_result = state["sam_result"]
        summary_stats = state["summary_stats"]
        image_paths = state.get("image_paths")
        first_image_name = state.get("first_image_name", "frame_0000")
        
        viz_path = None
        if self.save_visualizations:
            # Generate and save visualization from cached result
            preprocessed_img = state.get("preprocessed_image_array")
            overlay_img = visualize_sam_results(sam_result, preprocessed_img)
            
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            viz_path = viz_dir / f"overlay_0000_{first_image_name}.png"
            
            save_sam_visualization(
                overlay_img, 
                str(viz_path), 
                0,
                sam_result['total_count'], 
                state.get("final_params_for_batch", {}), 
                self.logger
            )
        
        return {
            "index": 0,
            "image_path": str(image_paths[0]) if image_paths else None,
            "image_name": first_image_name,
            "visualization_path": str(viz_path) if viz_path else None,
            "particle_count": sam_result['total_count'],
            "statistics": summary_stats,
            "success": True,
            "error": None,
            "cached": True  # Flag indicating this used cached results
        }
    
    def execute(self, state: dict) -> dict:
        """Process all images using refined parameters."""
        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", False)
        input_type = state.get("input_type", "unknown")
        
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({num_images} images)"
        self.logger.info(f"\n\n🔄 --- PROCESSING: {mode_str} --- 🔄\n")
        
        if num_images == 0:
            state["batch_results"] = []
            return state
        
        # =====================================================================
        # OPTIMIZATION: Skip re-analysis for single image if params unchanged
        # =====================================================================
        if self._can_skip_single_image_reanalysis(state):
            self.logger.info("⏭️  Skipping re-analysis (single image, parameters unchanged)")
            self.logger.info("   Using cached results from initial analysis.\n")
            
            cached_result = self._build_result_from_cached(state)
            
            # Save batch results JSON (maintains consistency with full processing)
            results_path = self.output_dir / "batch_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "total_images": 1,
                    "is_single_image": True,
                    "successful": 1,
                    "parameters_used": state.get("final_params_for_batch", {}),
                    "input_type": input_type,
                    "used_cached_result": True,
                    "results": [cached_result]
                }, f, indent=2, default=str)
            
            state["batch_results"] = [cached_result]
            state["batch_results_path"] = str(results_path)
            
            self.logger.info(f"   ✅ {cached_result['particle_count']} particles (from cache)")
            self.logger.info(f"\n✅ Processing complete: 1/1 successful (cached).")
            
            return state
        
        # =====================================================================
        # FULL PROCESSING: Either batch mode OR single image with changed params
        # =====================================================================
        
        # Get refined parameters
        batch_params = state.get("final_params_for_batch", state.get("current_params", {}))
        
        self.logger.info(f"📦 Processing with parameters:")
        for key, value in batch_params.items():
            if key not in ['checkpoint_path', 'device']:
                self.logger.info(f"   - {key}: {value}")
        
        # Load SAM model ONCE
        self.logger.info("\n🧠 Loading SAM model...")
        try:
            sam_analyzer = get_or_create_sam_model(batch_params)
            self.logger.info("✅ SAM model loaded and cached.")
        except Exception as e:
            self.logger.error(f"❌ Failed to load SAM model: {e}")
            state["batch_results"] = []
            state["error_dict"] = {"error": "SAM model loading failed", "details": str(e)}
            return state
        
        nm_per_pixel = state.get("nm_per_pixel")
        batch_results = []
        
        for idx in range(num_images):
            try:
                raw_image, image_name = self._get_image_and_name(state, idx)
                
                if is_single:
                    self.logger.info(f"   Processing: {image_name}")
                else:
                    self.logger.info(f"   [{idx + 1}/{num_images}] Processing: {image_name}")
                
                preprocessed_img, _ = preprocess_image(raw_image)
                
                # Reuse cached model
                sam_result = run_sam_analysis(
                    preprocessed_img, 
                    params=batch_params,
                    analyzer=sam_analyzer
                )
                
                # Get image path for stats
                image_paths = state.get("image_paths")
                image_path_for_stats = image_paths[idx] if image_paths else f"frame_{idx:04d}"
                
                summary_stats = calculate_sam_statistics(
                    sam_result=sam_result,
                    image_path=str(image_path_for_stats),
                    preprocessed_image_shape=preprocessed_img.shape,
                    nm_per_pixel=nm_per_pixel
                )
                
                # Save visualization
                viz_path = None
                if self.save_visualizations:
                    overlay_img = visualize_sam_results(sam_result, preprocessed_img)
                    viz_dir = self.output_dir / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    viz_path = viz_dir / f"overlay_{idx:04d}_{image_name}.png"
                    save_sam_visualization(
                        overlay_img, str(viz_path), idx,
                        sam_result['total_count'], batch_params, self.logger
                    )
                
                result_entry = {
                    "index": idx,
                    "image_path": str(image_paths[idx]) if image_paths else None,
                    "image_name": image_name,
                    "visualization_path": str(viz_path) if viz_path else None,
                    "particle_count": sam_result['total_count'],
                    "statistics": summary_stats,
                    "success": True,
                    "error": None,
                    "cached": False
                }
                
                batch_results.append(result_entry)
                self.logger.info(f"      ✅ Detected {sam_result['total_count']} particles")
                
            except Exception as e:
                self.logger.error(f"      ❌ Failed: {e}")
                batch_results.append({
                    "index": idx,
                    "image_path": None,
                    "image_name": f"frame_{idx:04d}",
                    "visualization_path": None,
                    "particle_count": 0,
                    "statistics": {},
                    "success": False,
                    "error": str(e),
                    "cached": False
                })
        
        # Save batch results JSON
        results_path = self.output_dir / "batch_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_images": num_images,
                "is_single_image": is_single,
                "successful": sum(1 for r in batch_results if r["success"]),
                "parameters_used": batch_params,
                "input_type": input_type,
                "used_cached_result": False,
                "results": batch_results
            }, f, indent=2, default=str)
        
        state["batch_results"] = batch_results
        state["batch_results_path"] = str(results_path)
        
        successful = sum(1 for r in batch_results if r["success"])
        self.logger.info(f"\n✅ Processing complete: {successful}/{num_images} successful.")
        
        return state

# ============================================================================
# CONDITIONAL CUSTOM ANALYSIS CONTROLLER
# (Only runs for n>=2, skipped for single images)
# ============================================================================

class ConditionalCustomAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates and executes custom Python script for trend analysis.
    
    CONDITIONAL:
    - For n>=2: Generates trend analysis script
    - For n=1: Skipped (no trends to analyze)
    """
    
    def __init__(
        self, 
        model, 
        logger: logging.Logger, 
        generation_config, 
        safety_settings, 
        parse_fn: Callable, 
        settings: dict,
        executor
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
        self.max_correction_attempts = settings.get('max_script_corrections', 3)
        self.executor = executor
    
    def execute(self, state: dict) -> dict:
        """Execute custom analysis - conditional on batch size."""
        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", False)
        
        if is_single or num_images < 2:
            self.logger.info("\n📊 Custom trend analysis skipped (single image mode).\n")
            state["custom_analysis_results"] = {
                "success": True,
                "skipped": True,
                "reason": "Single image - no trend analysis applicable"
            }
            return state
        
        self.logger.info("\n\n🧠 --- CUSTOM ANALYSIS SCRIPT GENERATION --- 🧠\n")
        
        # Full custom analysis logic for batches
        batch_results = state.get("batch_results", [])
        
        script_result = self._generate_analysis_script(state)
        
        if not script_result or "script" not in script_result:
            self.logger.error("Failed to generate analysis script.")
            state["custom_analysis_results"] = {"success": False, "error": "Script generation failed"}
            return state
        
        approach = script_result.get('analysis_approach', 'unknown')
        metrics = script_result.get('key_metrics_to_track', [])
        self.logger.info(f"   📊 Analysis approach: {approach}")
        self.logger.info(f"   📈 Key metrics: {metrics}")
        
        script = script_result["script"]
        success, stdout, stderr = False, "", ""
        
        for attempt in range(self.max_correction_attempts + 1):
            if attempt > 0:
                self.logger.info(f"   🔄 Execution attempt {attempt + 1}")
            
            success, stdout, stderr = self._execute_script(script)
            
            if success:
                self.logger.info("   ✅ Script executed successfully!")
                break
            
            error_preview = stderr[:200] + "..." if len(stderr) > 200 else stderr
            self.logger.warning(f"   ⚠️ Script failed: {error_preview}")
            
            if attempt < self.max_correction_attempts:
                corrected = self._correct_script(script, stderr, attempt + 1)
                if corrected:
                    script = corrected
                else:
                    break
        
        generated_files = []
        for ext in ['*.png', '*.csv']:
            generated_files.extend(self.output_dir.glob(ext))
        
        state["custom_analysis_results"] = {
            "success": success,
            "skipped": False,
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics_to_track"),
            "reasoning": script_result.get("reasoning"),
            "stdout": stdout,
            "stderr": stderr if not success else None,
            "generated_files": [str(f) for f in generated_files],
            "script_path": str(self.output_dir / "custom_analysis.py")
        }
        
        return state
    
    def _generate_analysis_script(self, state: dict) -> Optional[Dict]:
        """Generate custom analysis script using LLM."""
        self.logger.info("   🧠 Generating custom analysis script...")
        
        batch_results = state.get("batch_results", [])
        series_metadata = state.get("series_metadata", {})
        
        particle_counts = []
        mean_areas = []
        
        for r in batch_results:
            if r["success"]:
                particle_counts.append(r["particle_count"])
                stats = r.get("statistics", {})
                mean_area = stats.get("mean_area_pixels")
                mean_areas.append(round(mean_area, 2) if mean_area else 0)
        
        time_points = series_metadata.get("time_points") or list(range(len(batch_results)))
        
        summary = {
            "total_images": len(batch_results),
            "successful": sum(1 for r in batch_results if r["success"]),
            "particle_counts": particle_counts,
            "mean_areas": mean_areas,
            "variable": series_metadata.get("variable", "unknown"),
            "time_points": time_points,
        }
        
        prompt_parts = [
            SAM_BATCH_CUSTOM_ANALYSIS_INSTRUCTIONS,
            f"\n\n**BATCH SUMMARY:**\n```json\n{json.dumps(summary, indent=2)}\n```",
        ]
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict and not (result_json and 'script' in result_json):
                return None
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error generating script: {e}")
            return None
    
    def _execute_script(self, script: str) -> tuple:
        # Save script for debugging
        script_path = self.output_dir / "custom_analysis.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        result = self.executor.execute_script(script, working_dir=str(self.output_dir))
        
        if result["status"] == "success":
            return True, result["stdout"], result["stderr"]
        else:
            return False, "", result["message"]
    
    def _correct_script(self, original_script: str, error_message: str, attempt: int) -> Optional[str]:
        """Use LLM to correct a failed script."""
        self.logger.info(f"   🔧 Attempting script correction (attempt {attempt})...")
        
        if len(error_message) > 1000:
            error_message = error_message[:500] + "\n...[truncated]...\n" + error_message[-500:]
        
        prompt = f"""Fix this Python script that failed:

**SCRIPT:**
```python
{original_script}
```

**ERROR:**
```
{error_message}
```

Return JSON with: {{"diagnosis": "...", "script": "corrected script"}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, _ = self._parse_llm_response(response)
            
            if result_json:
                self.logger.info(f"   📋 Diagnosis: {result_json.get('diagnosis', 'N/A')}")
                return result_json.get("script")
            return None
            
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None


# ============================================================================
# UNIFIED SYNTHESIS CONTROLLER
# (Adapts behavior based on single vs batch)
# ============================================================================

class UnifiedSynthesisController:
    """
    [🧠 LLM Step]
    Synthesizes findings into scientific claims.
    
    ADAPTIVE:
    - For n>=2: Cross-image synthesis with trend analysis
    - For n=1: Single-image scientific interpretation
    """
    
    def __init__(
        self, 
        model, 
        logger: logging.Logger, 
        generation_config, 
        safety_settings, 
        parse_fn: Callable, 
        settings: dict,
        store_fn: Callable = None
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self._store_analysis_images = store_fn or (lambda *args, **kwargs: None)
    
    def execute(self, state: dict) -> dict:
        """Execute synthesis - adapts to single vs batch."""
        is_single = state.get("is_single_image", False)
        
        if is_single:
            return self._synthesize_single_image(state)
        else:
            return self._synthesize_batch(state)
    
    def _build_segmentation_quality_context(self, state: dict) -> str:
        """
        Build a context block describing segmentation quality, refinement
        history, and any caveats the synthesis LLM should factor into its
        scientific claims.
        
        This ensures the LLM knows about:
        - Whether the judge had to intervene (refinement never converged)
        - Quality scores from the evaluation
        - What parameter changes were tried and why
        - Any quality warnings from the judge
        
        The LLM is instructed to incorporate appropriate caveats and
        confidence qualifiers into its scientific claims.
        """
        lines = []
        
        # Quality evaluation scores
        llm_eval = state.get("llm_quality_evaluation", {})
        if llm_eval:
            coverage = llm_eval.get("coverage_score", "?")
            accuracy = llm_eval.get("accuracy_score", "?")
            overall = llm_eval.get("overall_quality", "unknown")
            lines.append(f"- Segmentation quality assessment: coverage={coverage}/10, accuracy={accuracy}/10, overall={overall}")
        
        # Refinement history
        refinement_iters = state.get("llm_refinement_iterations", 0)
        refinement_history = state.get("refinement_history", [])
        
        if refinement_iters > 1:
            lines.append(f"- Parameter refinement required {refinement_iters} iteration(s) to optimize segmentation")
            for record in refinement_history:
                iter_num = record.get("iteration")
                decision = record.get("decision", "?")
                particles = record.get("particle_count", "?")
                eval_info = record.get("evaluation", {})
                lines.append(
                    f"  - Iteration {iter_num}: {particles} particles, "
                    f"coverage={eval_info.get('coverage_score', '?')}/10, "
                    f"accuracy={eval_info.get('accuracy_score', '?')}/10, "
                    f"decision={decision}"
                )
        
        # Judge intervention
        judge_invoked = state.get("judge_invoked", False)
        judge_selected = state.get("judge_selected_iteration")
        
        if judge_invoked:
            lines.append(
                f"- ⚠️ IMPORTANT: Automated refinement did NOT converge to an accepted result. "
                f"A judge selected iteration {judge_selected} as the best available, but even "
                f"this result may have significant segmentation issues."
            )
        
        # Compose the full context block
        if not lines:
            return ""
        
        quality_context = "\n".join(lines)
        
        caveat_instruction = (
            "\n\n**SEGMENTATION QUALITY CONTEXT:**\n"
            f"{quality_context}\n\n"
            "**IMPORTANT:** Factor the above segmentation quality information into your "
            "scientific analysis. Specifically:\n"
            "- If overall quality is 'poor' or scores are below 5/10, explicitly state that "
            "quantitative claims have limited reliability due to segmentation uncertainty.\n"
            "- If the judge had to intervene (refinement did not converge), note that the "
            "segmentation parameters may not be optimal for this sample and results should "
            "be interpreted with caution.\n"
            "- Scale the confidence of your scientific claims to match the segmentation quality. "
            "High-quality segmentation (8+/10) supports strong claims; poor segmentation "
            "(below 5/10) should only support tentative observations.\n"
            "- If particle counts or morphological statistics may be unreliable, say so "
            "explicitly rather than presenting them as definitive.\n"
        )
        
        return caveat_instruction
    
    def _synthesize_single_image(self, state: dict) -> dict:
        """Generate scientific interpretation for a single image."""
        self.logger.info("\n\n🔬 --- SINGLE IMAGE SYNTHESIS --- 🔬\n")
        
        batch_results = state.get("batch_results", [])
        if not batch_results or not batch_results[0].get("success"):
            state["synthesis_result"] = {"error": "No successful analysis to synthesize"}
            return state
        
        result = batch_results[0]
        stats = result.get("statistics", {})
        
        # Build segmentation quality context
        quality_context = self._build_segmentation_quality_context(state)
        
        # Build prompt for single-image analysis
        prompt_parts = [
            SAM_SINGLE_IMAGE_SYNTHESIS_INSTRUCTIONS,
            f"\n\n**IMAGE ANALYSIS RESULTS:**",
            f"- Image: {result.get('image_name', 'unknown')}",
            f"- Particle count: {result.get('particle_count', 0)}",
            f"\n\n**MORPHOLOGICAL STATISTICS:**\n{json.dumps(stats, indent=2)}",
            f"\n\n**SYSTEM INFORMATION:**\n{json.dumps(state.get('system_info', {}), indent=2)}",
        ]
        
        # Inject quality context before images
        if quality_context:
            prompt_parts.append(quality_context)
        
        # Add visualization if available
        viz_path = result.get("visualization_path")
        if viz_path and Path(viz_path).exists():
            with open(viz_path, 'rb') as f:
                prompt_parts.append("\n\n**SEGMENTATION VISUALIZATION:**")
                prompt_parts.append({"mime_type": "image/png", "data": f.read()})
        
        # Add original image
        if state.get("image_blob"):
            prompt_parts.append("\n\n**ORIGINAL IMAGE:**")
            prompt_parts.append(state["image_blob"])
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Single image synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state
    
    def _synthesize_batch(self, state: dict) -> dict:
        """Synthesize findings across multiple images."""
        self.logger.info("\n\n🔬 --- BATCH SYNTHESIS --- 🔬\n")
        
        batch_results = state.get("batch_results", [])
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        
        stats_summary = []
        for r in batch_results:
            if r["success"]:
                stats_summary.append({
                    "index": r["index"],
                    "name": r["image_name"],
                    "particles": r["particle_count"],
                    "mean_area": r["statistics"].get("mean_area_pixels"),
                })
        
        # Build segmentation quality context
        quality_context = self._build_segmentation_quality_context(state)
        
        prompt_parts = [
            SAM_BATCH_SYNTHESIS_INSTRUCTIONS,
            f"\n\n**INDIVIDUAL ANALYSIS SUMMARY:**\n{json.dumps(stats_summary, indent=2)}",
            f"\n\n**CUSTOM ANALYSIS RESULTS:**\n{json.dumps(custom_results, indent=2)}",
            f"\n\n**SERIES METADATA:**\n{json.dumps(series_metadata, indent=2)}"
        ]
        
        # Inject quality context
        if quality_context:
            prompt_parts.append(quality_context)
        
        # Add generated plots
        if custom_results.get("success") and custom_results.get("generated_files"):
            prompt_parts.append("\n\n**TREND ANALYSIS VISUALIZATIONS:**")
            for file_path in custom_results["generated_files"][:5]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        prompt_parts.append(f"\n{Path(file_path).name}:")
                        prompt_parts.append({"mime_type": "image/png", "data": f.read()})
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Batch synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state


# ============================================================================
# UNIFIED REPORT GENERATION CONTROLLER
# (Generates appropriate report for single vs batch)
# ============================================================================

class UnifiedReportGenerationController:
    """
    [📄 Report Step]
    Generates final HTML report and JSON summary.
    
    ADAPTIVE:
    - For n>=2: Full batch report with trends
    - For n=1: Simplified single-image report
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'sam_output'))
    
    def execute(self, state: dict) -> dict:
        """Generate final reports."""
        is_single = state.get("is_single_image", False)
        
        self.logger.info("\n\n📄 --- GENERATING REPORT --- 📄\n")
        
        if is_single:
            self._generate_single_image_report(state)
        else:
            self._generate_batch_report(state)
        
        return state
    
    def _generate_single_image_report(self, state: dict) -> None:
        """Generate report for single image analysis."""
        import base64
        
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        result = batch_results[0] if batch_results else {}
        
        detailed_analysis = synthesis.get("detailed_analysis", "No analysis available.")
        scientific_claims = synthesis.get("scientific_claims", [])
        
        # Get visualization image
        viz_path = result.get("visualization_path")
        embedded_image = None
        
        if viz_path and Path(viz_path).exists():
            with open(viz_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                embedded_image = {"name": Path(viz_path).stem, "data": b64}
        else:
            # Fallback: search for visualization in output directory
            viz_dir = self.output_dir / "visualizations"
            if viz_dir.exists():
                viz_files = sorted(viz_dir.glob("overlay_*.png"))
                if viz_files:
                    with open(viz_files[0], 'rb') as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                        embedded_image = {"name": viz_files[0].stem, "data": b64}
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Include LLM quality evaluation info if available
        quality_info = ""
        llm_eval = state.get("llm_quality_evaluation", {})
        refinement_iters = state.get("llm_refinement_iterations", 0)
        judge_invoked = state.get("judge_invoked", False)
        judge_selected = state.get("judge_selected_iteration")
        
        if llm_eval:
            quality_info = f"""
        <p><strong>Automated Quality Check:</strong> 
            Coverage {llm_eval.get('coverage_score', '?')}/10, 
            Accuracy {llm_eval.get('accuracy_score', '?')}/10
            ({llm_eval.get('overall_quality', 'N/A')})
            — {refinement_iters} refinement iteration(s)</p>"""
            if judge_invoked:
                quality_info += f"""
        <p><strong>⚖️ Judge Override:</strong> Selected iteration {judge_selected} 
            out of {refinement_iters} (best quality among all attempts)</p>"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SAM Microscopy Analysis Report - Single Image</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f4f4f9; }}
        .container {{ background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .metadata-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin: 20px 0; }}
        .analysis-text {{ white-space: pre-wrap; background: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; }}
        .claim-card {{ background: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin: 15px 0; border-radius: 0 5px 5px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center; }}
        .image-card img {{ max-width: 100%; border-radius: 4px; }}
        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 SAM Microscopy Analysis Report</h1>
    
    <div class="metadata-box">
        <p><strong>Date:</strong> {timestamp}</p>
        <p><strong>Image:</strong> {result.get('image_name', 'unknown')}</p>
        <p><strong>Particles Detected:</strong> {result.get('particle_count', 0)}</p>
        <p><strong>Mean Area:</strong> {result.get('statistics', {}).get('mean_area_pixels', 'N/A')} px²</p>{quality_info}
    </div>

    <h2>Scientific Analysis</h2>
    <div class="analysis-text">{detailed_analysis}</div>
"""
        
        if embedded_image:
            html += """
    <h2>Visualization</h2>
    <div class="image-grid">
"""
            html += f"""        <div class="image-card">
            <img src="data:image/png;base64,{embedded_image['data']}" alt="Segmentation Result">
            <p>Particle Detection Overlay</p>
        </div>
"""
            html += "    </div>\n"
        
        if scientific_claims:
            html += "    <h2>Key Scientific Claims</h2>\n"
            for i, claim in enumerate(scientific_claims, 1):
                claim_text = claim.get('claim', 'N/A') if isinstance(claim, dict) else str(claim)
                impact_text = claim.get('scientific_impact', '') if isinstance(claim, dict) else ''
                has_anyone_text = claim.get('has_anyone_question', '') if isinstance(claim, dict) else ''
                
                html += f"""    <div class="claim-card">
        <strong>Claim {i}:</strong> {claim_text}<br>
"""
                if impact_text:
                    html += f"""        <em>Impact:</em> {impact_text}<br>
"""
                if has_anyone_text:
                    html += f"""        <em>Literature Query:</em> {has_anyone_text}
"""
                html += "    </div>\n"
        
        html += """
    <div class="footer">Generated by SAM Microscopy Analysis Agent</div>
</div>
</body>
</html>"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated report: {report_path}")
    
    def _generate_batch_report(self, state: dict) -> None:
        """Generate report for batch analysis."""
        import base64
        
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        
        detailed_analysis = synthesis.get("detailed_analysis", "No synthesis available.")
        scientific_claims = synthesis.get("scientific_claims", [])
        
        # Collect PNG files from main output directory (trend plots from custom analysis)
        # Exclude review_iteration images and overlay images (individual segmentations)
        png_files = sorted(self.output_dir.glob("*.png"))
        embedded_images = []
        for png_path in png_files:
            # Skip review iterations and individual overlay images
            if png_path.name.startswith("review_iteration"):
                continue
            if png_path.name.startswith("overlay_"):
                continue
            if png_path.exists():
                with open(png_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    embedded_images.append({"name": png_path.stem, "data": b64})
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_images = len(batch_results)
        successful = sum(1 for r in batch_results if r.get("success"))
        
        # Include LLM quality evaluation info if available
        quality_info = ""
        llm_eval = state.get("llm_quality_evaluation", {})
        refinement_iters = state.get("llm_refinement_iterations", 0)
        judge_invoked = state.get("judge_invoked", False)
        judge_selected = state.get("judge_selected_iteration")
        
        if llm_eval:
            quality_info = f"""
        <p><strong>Automated Quality Check:</strong> 
            Coverage {llm_eval.get('coverage_score', '?')}/10, 
            Accuracy {llm_eval.get('accuracy_score', '?')}/10
            ({llm_eval.get('overall_quality', 'N/A')})
            — {refinement_iters} refinement iteration(s)</p>"""
            if judge_invoked:
                quality_info += f"""
        <p><strong>⚖️ Judge Override:</strong> Selected iteration {judge_selected} 
            out of {refinement_iters} (best quality among all attempts)</p>"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SAM Microscopy Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f4f4f9; }}
        .container {{ background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .metadata-box {{ background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin: 20px 0; }}
        .analysis-text {{ white-space: pre-wrap; background: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; }}
        .claim-card {{ background: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin: 15px 0; border-radius: 0 5px 5px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 8px; text-align: center; }}
        .image-card img {{ max-width: 100%; border-radius: 4px; }}
        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 SAM Microscopy Analysis Report</h1>
    
    <div class="metadata-box">
        <p><strong>Date:</strong> {timestamp}</p>
        <p><strong>Images Processed:</strong> {successful}/{num_images}</p>
        <p><strong>Analysis Approach:</strong> {custom_results.get("approach", "time_series")}</p>
        <p><strong>Series Variable:</strong> {series_metadata.get("variable", "unknown")}</p>{quality_info}
    </div>

    <h2>Scientific Analysis</h2>
    <div class="analysis-text">{detailed_analysis}</div>
"""
        
        if embedded_images:
            html += """
    <h2>Visualizations</h2>
    <div class="image-grid">
"""
            for idx, img in enumerate(embedded_images[:6], 1):
                # Determine caption based on image type
                name_lower = img['name'].lower()
                if 'overlay' in name_lower:
                    caption = f"Segmentation Result {idx}"
                elif 'trend' in name_lower or 'time' in name_lower:
                    caption = f"Trend Analysis {idx}"
                elif 'histogram' in name_lower or 'distribution' in name_lower:
                    caption = f"Distribution Plot {idx}"
                else:
                    caption = f"Analysis Output {idx}"
                
                html += f"""        <div class="image-card">
            <img src="data:image/png;base64,{img['data']}" alt="{caption}">
            <p>{caption}</p>
        </div>
"""
            html += "    </div>\n"
        
        if scientific_claims:
            html += "    <h2>Key Scientific Claims</h2>\n"
            for i, claim in enumerate(scientific_claims, 1):
                claim_text = claim.get('claim', 'N/A') if isinstance(claim, dict) else str(claim)
                impact_text = claim.get('scientific_impact', '') if isinstance(claim, dict) else ''
                has_anyone_text = claim.get('has_anyone_question', '') if isinstance(claim, dict) else ''
                
                html += f"""    <div class="claim-card">
        <strong>Claim {i}:</strong> {claim_text}<br>
"""
                if impact_text:
                    html += f"""        <em>Impact:</em> {impact_text}<br>
"""
                if has_anyone_text:
                    html += f"""        <em>Literature Query:</em> {has_anyone_text}
"""
                html += "    </div>\n"
        
        html += """
    <div class="footer">Generated by SAM Microscopy Analysis Agent</div>
</div>
</body>
</html>"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated batch report: {report_path}")