import json
import os
from PIL import Image
import logging
import numpy as np

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .instruct import (
    SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    PARTICLE_ANALYSIS_REFINE_INSTRUCTIONS,
)
from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes

from ...auth import get_api_key, APIKeyNotFoundError

from atomai.models import ParticleAnalyzer


class GeminiSAMMicroscopyAnalysisAgent:
    """
    Agent for analyzing microscopy images using Segment Anything Model (SAM) and Gemini models.
    Follows the same pattern as the standard microscopy agent but uses SAM for particle segmentation.
    """

    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05", sam_settings: dict | None = None):
        # Auto-discover API key
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        genai.configure(api_key=google_api_key)
        
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        self.logger = logging.getLogger(__name__)
        self.sam_settings = sam_settings if sam_settings else {}
        self.RUN_SAM = self.sam_settings.get('SAM_ENABLED', True)
        self.refinement_cycles = self.sam_settings.get('refinement_cycles', 0)
        self.save_visualizations = self.sam_settings.get('save_visualizations', True)

    def _parse_llm_response(self, response) -> tuple[dict | None, dict | None]:
        """
        Parses the LLM response, expecting JSON.
        """
        result_json = None
        error_dict = None
        raw_text = None
        json_string = None

        try:
            raw_text = response.text
            first_brace_index = raw_text.find('{')
            last_brace_index = raw_text.rfind('}')
            if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                json_string = raw_text[first_brace_index : last_brace_index + 1]
                result_json = json.loads(json_string)
            else:
                raise ValueError("Could not find valid JSON object delimiters '{' and '}' in the response text.")

        except (json.JSONDecodeError, AttributeError, IndexError, ValueError) as e:
            error_details = str(e)
            error_raw_response = raw_text if raw_text is not None else getattr(response, 'text', 'N/A')
            self.logger.error(f"Error parsing Gemini JSON response: {e}")
            parsed_substring_for_log = json_string if json_string else 'N/A'
            self.logger.debug(f"Attempted to parse substring: {parsed_substring_for_log[:500]}...")
            self.logger.debug(f"Original Raw response text: {error_raw_response[:500]}...")

            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                self.logger.error(f"Request blocked due to: {block_reason}")
                error_dict = {"error": f"Content blocked by safety filters", "details": f"Reason: {block_reason}"}
            elif response.candidates and response.candidates[0].finish_reason != 1:
                finish_reason = response.candidates[0].finish_reason
                self.logger.error(f"Generation finished unexpectedly: {finish_reason}")
                error_dict = {"error": f"Generation finished unexpectedly: {finish_reason}", "details": error_details, "raw_response": error_raw_response}
            else:
                error_dict = {"error": "Failed to parse valid JSON from LLM response", "details": error_details, "raw_response": error_raw_response}
        except Exception as e:
            self.logger.exception(f"Unexpected error processing response: {e}")
            error_dict = {"error": "Unexpected error processing LLM response", "details": str(e)}
        
        return result_json, error_dict

    def _generate_json_from_text_parts(self, prompt_parts: list) -> tuple[dict | None, dict | None]:
        """
        Internal helper to generate JSON from a list of textual prompt parts.
        """
        try:
            self.logger.debug(f"Sending text-only prompt to LLM. Total parts: {len(prompt_parts)}")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return self._parse_llm_response(response)
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during text-based LLM call: {e}")
            return None, {"error": "An unexpected error occurred during text-based LLM call", "details": str(e)}

    def _save_visualization(self, overlay_image: np.ndarray, stage: str, cycle: int, particle_count: int, params: dict):
        """
        Save visualization images for each refinement step.
        """
        try:
            from datetime import datetime
            
            # Create output directory
            output_dir = "sam_analysis_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp and info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stage}_cycle{cycle:02d}_{particle_count}particles_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            Image.fromarray(overlay_image).save(filepath)
            
            # Save parameters as text file
            params_filename = f"{stage}_cycle{cycle:02d}_params_{timestamp}.txt"
            params_filepath = os.path.join(output_dir, params_filename)
            with open(params_filepath, 'w') as f:
                f.write(f"Stage: {stage}\n")
                f.write(f"Cycle: {cycle}\n")
                f.write(f"Particle Count: {particle_count}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Parameters:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
            
            print(f"📸 Saved {stage} visualization: {filename} ({particle_count} particles)")
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization: {e}")

    def _run_sam_analysis(self, image_path: str) -> tuple[np.ndarray | None, dict | None]:
        """
        Run SAM particle segmentation analysis with optional iterative refinement.
        """
        try:
            self.logger.info("--- Starting SAM Particle Segmentation Analysis ---")
            
            # Initialize ParticleAnalyzer with settings from config
            model_type = self.sam_settings.get('model_type', 'vit_h')
            checkpoint_path = self.sam_settings.get('checkpoint_path', None)
            device = self.sam_settings.get('device', 'auto')
            
            analyzer = ParticleAnalyzer(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                device=device
            )
        
            # Load and preprocess image
            raw_image = load_image(image_path)
            if len(raw_image.shape) == 3:
                raw_image = np.mean(raw_image, axis=2).astype(np.uint8)

            # Use preprocessed (resized) image for BOTH SAM and LLM
            image_array = preprocess_image(raw_image)
            self.original_preprocessed_image = image_array

                    
            # Initial analysis parameters from config or defaults
            current_params = {
                "use_clahe": self.sam_settings.get('use_clahe', False),
                "sam_parameters": self.sam_settings.get('sam_parameters', 'default'),
                "min_area": self.sam_settings.get('min_area', 500),
                "max_area": self.sam_settings.get('max_area', 50000),
                "use_pruning": self.sam_settings.get('use_pruning', True),
                "pruning_iou_threshold": self.sam_settings.get('pruning_iou_threshold', 0.5)
            }
            
            # Run initial analysis
            self.logger.info(f"Running initial SAM analysis with params: {current_params}")
            sam_result = analyzer.analyze(image_array, params=current_params)
            
            # Save initial visualization if requested
            if self.save_visualizations:
                initial_overlay = ParticleAnalyzer.visualize_particles(sam_result, show_plot=False, show_labels=True, show_centroids=True)
                self._save_visualization(initial_overlay, "initial", 0, sam_result['total_count'], current_params)
            
            # Run refinement cycles if requested
            for cycle in range(self.refinement_cycles):
                self.logger.info(f"--- Starting refinement cycle {cycle + 1}/{self.refinement_cycles} ---")
                
                # Create visualization of current result
                current_overlay = ParticleAnalyzer.visualize_particles(
                    sam_result, 
                    show_plot=False, 
                    show_labels=True, 
                    show_centroids=True
                )
                
                # Get refinement suggestions from LLM
                new_params = self._get_refinement_parameters(current_overlay, sam_result['total_count'], current_params)
                
                if new_params is None:
                    self.logger.warning(f"Failed to get refinement parameters for cycle {cycle + 1}, stopping refinement")
                    break
                
                # Check if parameters actually changed
                if new_params == current_params:
                    print(f"📋 No parameter changes suggested for cycle {cycle + 1}, stopping refinement")
                    self.logger.info(f"No parameter changes suggested for cycle {cycle + 1}, stopping refinement")
                    break
                
                # Show what changed
                changed_params = {k: v for k, v in new_params.items() if current_params.get(k) != v}
                if changed_params:
                    print(f"🔄 PARAMETER CHANGES for cycle {cycle + 1}:")
                    for key, value in changed_params.items():
                        print(f"   {key}: {current_params.get(key)} → {value}")
                    print()
                
                # Run analysis with refined parameters
                self.logger.info(f"Running refined analysis with params: {new_params}")
                refined_result = analyzer.analyze(image_array, params=new_params)
                
                # Save refined visualization if requested
                if self.save_visualizations:
                    refined_overlay = ParticleAnalyzer.visualize_particles(refined_result, show_plot=False, show_labels=True, show_centroids=True)
                    self._save_visualization(refined_overlay, "refined", cycle + 1, refined_result['total_count'], new_params)
                
                # Update current result and parameters
                sam_result = refined_result
                current_params = new_params
                
                self.logger.info(f"Cycle {cycle + 1} completed: {sam_result['total_count']} particles detected")
            
            # Create final visualization overlay
            final_overlay = ParticleAnalyzer.visualize_particles(
                sam_result, 
                show_plot=False, 
                show_labels=True, 
                show_centroids=True
            )
            
            # Extract comprehensive morphological statistics
            particles_df = ParticleAnalyzer.particles_to_dataframe(sam_result)
            
            if not particles_df.empty:
                summary_stats = {
                    'total_particles': sam_result['total_count'],
                    
                    # Size statistics
                    'mean_area': float(particles_df['area'].mean()),
                    'std_area': float(particles_df['area'].std()),
                    'area_range': [float(particles_df['area'].min()), float(particles_df['area'].max())],
                    'area_percentiles': [float(particles_df['area'].quantile(q)) for q in [0.25, 0.5, 0.75]],
                    
                    # Shape statistics
                    'mean_circularity': float(particles_df['circularity'].mean()),
                    'std_circularity': float(particles_df['circularity'].std()),
                    'circularity_range': [float(particles_df['circularity'].min()), float(particles_df['circularity'].max())],
                    
                    'mean_aspect_ratio': float(particles_df['aspect_ratio'].mean()),
                    'std_aspect_ratio': float(particles_df['aspect_ratio'].std()),
                    'aspect_ratio_range': [float(particles_df['aspect_ratio'].min()), float(particles_df['aspect_ratio'].max())],
                    
                    'mean_solidity': float(particles_df['solidity'].mean()),
                    'std_solidity': float(particles_df['solidity'].std()),
                    'solidity_range': [float(particles_df['solidity'].min()), float(particles_df['solidity'].max())],
                    
                    # Derived size statistics
                    'mean_equiv_diameter': float(particles_df['equiv_diameter'].mean()),
                    'std_equiv_diameter': float(particles_df['equiv_diameter'].std()),
                    'equiv_diameter_range': [float(particles_df['equiv_diameter'].min()), float(particles_df['equiv_diameter'].max())],
                    
                    'mean_perimeter': float(particles_df['perimeter'].mean()),
                    'std_perimeter': float(particles_df['perimeter'].std()),
                    'perimeter_range': [float(particles_df['perimeter'].min()), float(particles_df['perimeter'].max())],
                    
                    # Analysis metadata
                    'final_parameters': current_params,
                    'refinement_cycles_completed': min(cycle + 1 if 'cycle' in locals() else 0, self.refinement_cycles)
                }
            else:
                # Handle empty results
                summary_stats = {
                    'total_particles': 0,
                    'mean_area': 0, 'std_area': 0, 'area_range': [0, 0], 'area_percentiles': [0, 0, 0],
                    'mean_circularity': 0, 'std_circularity': 0, 'circularity_range': [0, 0],
                    'mean_aspect_ratio': 0, 'std_aspect_ratio': 0, 'aspect_ratio_range': [0, 0],
                    'mean_solidity': 0, 'std_solidity': 0, 'solidity_range': [0, 0],
                    'mean_equiv_diameter': 0, 'std_equiv_diameter': 0, 'equiv_diameter_range': [0, 0],
                    'mean_perimeter': 0, 'std_perimeter': 0, 'perimeter_range': [0, 0],
                    'final_parameters': current_params,
                    'refinement_cycles_completed': min(cycle + 1 if 'cycle' in locals() else 0, self.refinement_cycles)
                }
            
            total_cycles = summary_stats['refinement_cycles_completed']
            self.logger.info(f"SAM analysis complete after {total_cycles} refinement cycles. Final count: {sam_result['total_count']} particles.")
            
            # Save final visualization if requested
            if self.save_visualizations:
                self._save_visualization(final_overlay, "final", total_cycles, sam_result['total_count'], current_params)
                print(f"💾 Visualizations saved to: sam_analysis_visualizations/")
            
            return final_overlay, summary_stats
            
        except Exception as sam_e:
            self.logger.error(f"SAM particle segmentation analysis failed: {sam_e}", exc_info=True)
            return None, None

    def _get_refinement_parameters(self, overlay_image: np.ndarray, particle_count: int, current_params: dict) -> dict | None:
        """
        Get refinement parameters from LLM based on visual analysis of current results.
        """
        try:
            self.logger.info(f"Requesting parameter refinement for {particle_count} particles")
            
            # Convert both images to bytes for LLM
            original_bytes = convert_numpy_to_jpeg_bytes(self.original_preprocessed_image)
            overlay_bytes = convert_numpy_to_jpeg_bytes(overlay_image)
            
            # Build prompt for refinement with BOTH images
            prompt_parts = [PARTICLE_ANALYSIS_REFINE_INSTRUCTIONS]
            prompt_parts.append(f"\n\nCurrent Analysis Results:")
            prompt_parts.append(f"- Particle count: {particle_count}")
            prompt_parts.append(f"- Current parameters: {json.dumps(current_params, indent=2)}")
            
            # Add ORIGINAL image first for comparison
            prompt_parts.append(f"\n\n**ORIGINAL MICROSCOPY IMAGE (for reference):**")
            prompt_parts.append("This is the original microscopy image showing the actual particles to be segmented.")
            prompt_parts.append({"mime_type": "image/jpeg", "data": original_bytes})
            
            # Add SEGMENTATION overlay second for evaluation
            prompt_parts.append(f"\n\n**CURRENT SEGMENTATION RESULT:**")
            prompt_parts.append("This shows the current segmentation with detected particles outlined in red, centroids as green dots, and particle IDs labeled.")
            prompt_parts.append({"mime_type": "image/jpeg", "data": overlay_bytes})
            
            prompt_parts.append("\n\n**ANALYSIS TASK:**")
            prompt_parts.append("Compare the segmentation result against the original image. Are the detected particles the right size? Are particles being missed or incorrectly detected? Are the red outlines accurately capturing the particle boundaries visible in the original image?")
            prompt_parts.append("\nProvide refined parameters to improve the segmentation accuracy.")
            
            # Query LLM for refinement
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM refinement request failed: {error_dict}")
                return None
            
            reasoning = result_json.get("reasoning", "No reasoning provided")
            new_parameters = result_json.get("parameters", {})
            
            # Log AND print reasoning for visibility
            self.logger.info(f"LLM refinement reasoning: {reasoning}")
            self.logger.info(f"LLM suggested parameters: {new_parameters}")
            
            print(f"\n🧠 LLM REFINEMENT REASONING:")
            print(f"   {reasoning}")
            print(f"\n🔧 SUGGESTED PARAMETERS:")
            for key, value in new_parameters.items():
                print(f"   {key}: {value}")
            print()
            
            # Validate new parameters
            expected_keys = {"use_clahe", "sam_parameters", "min_area", "max_area", "pruning_iou_threshold"}
            if not all(key in new_parameters for key in expected_keys):
                self.logger.warning(f"Invalid parameter set from LLM, missing keys: {expected_keys - set(new_parameters.keys())}")
                return None
            
            # Add use_pruning=True since it's always enabled
            new_parameters["use_pruning"] = True
            
            # Basic validation
            if not isinstance(new_parameters.get("use_clahe"), bool):
                self.logger.warning("Invalid use_clahe parameter type")
                return None
            
            if new_parameters.get("sam_parameters") not in ["default", "sensitive", "ultra-permissive"]:
                self.logger.warning("Invalid sam_parameters value")
                return None
            
            if not (isinstance(new_parameters.get("min_area"), int) and new_parameters.get("min_area") > 0):
                self.logger.warning("Invalid min_area parameter")
                return None
            
            if not (isinstance(new_parameters.get("max_area"), int) and new_parameters.get("max_area") > new_parameters.get("min_area")):
                self.logger.warning("Invalid max_area parameter")
                return None
            
            pruning_threshold = new_parameters.get("pruning_iou_threshold")
            if not (isinstance(pruning_threshold, (int, float)) and 0.0 <= pruning_threshold <= 1.0):
                self.logger.warning("Invalid pruning_iou_threshold parameter")
                return None
            
            return new_parameters
            
        except Exception as e:
            self.logger.error(f"Error getting refinement parameters: {e}")
            return None

    def _analyze_image_base(self, image_path: str, system_info: dict | str | None, 
                            instruction_prompt: str) -> tuple[dict | None, dict | None]:
        """
        Internal helper for image-based analysis, including optional SAM segmentation.
        """
        try:
            loaded_image = load_image(image_path)
            preprocessed_img_array = preprocess_image(loaded_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            sam_overlay, sam_stats = None, None
            if self.RUN_SAM:
                sam_overlay, sam_stats = self._run_sam_analysis(image_path)
            
            prompt_parts = [instruction_prompt]
            
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nPrimary Microscopy Image:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)

            if sam_overlay is not None and sam_stats is not None:
                # Convert SAM overlay to bytes for LLM
                sam_overlay_bytes = convert_numpy_to_jpeg_bytes(sam_overlay)
                prompt_parts.append("\n\nSupplemental SAM Particle Segmentation Analysis:")
                prompt_parts.append(f"Detected {sam_stats['total_particles']} particles with comprehensive morphological analysis:")
                
                # Size statistics
                prompt_parts.append(f"\n**Size Analysis:**")
                prompt_parts.append(f"- Mean area: {sam_stats['mean_area']:.1f} ± {sam_stats['std_area']:.1f} pixels")
                prompt_parts.append(f"- Area range: {sam_stats['area_range'][0]:.0f} to {sam_stats['area_range'][1]:.0f} pixels")
                prompt_parts.append(f"- Area quartiles (Q1, Q2, Q3): {sam_stats['area_percentiles'][0]:.0f}, {sam_stats['area_percentiles'][1]:.0f}, {sam_stats['area_percentiles'][2]:.0f} pixels")
                prompt_parts.append(f"- Mean equivalent diameter: {sam_stats['mean_equiv_diameter']:.1f} ± {sam_stats['std_equiv_diameter']:.1f} pixels")
                prompt_parts.append(f"- Mean perimeter: {sam_stats['mean_perimeter']:.1f} ± {sam_stats['std_perimeter']:.1f} pixels")
                
                # Shape statistics
                prompt_parts.append(f"\n**Shape Analysis:**")
                prompt_parts.append(f"- Mean circularity: {sam_stats['mean_circularity']:.3f} ± {sam_stats['std_circularity']:.3f} (range: {sam_stats['circularity_range'][0]:.3f} to {sam_stats['circularity_range'][1]:.3f})")
                prompt_parts.append(f"- Mean aspect ratio: {sam_stats['mean_aspect_ratio']:.2f} ± {sam_stats['std_aspect_ratio']:.2f} (range: {sam_stats['aspect_ratio_range'][0]:.2f} to {sam_stats['aspect_ratio_range'][1]:.2f})")
                prompt_parts.append(f"- Mean solidity: {sam_stats['mean_solidity']:.3f} ± {sam_stats['std_solidity']:.3f} (range: {sam_stats['solidity_range'][0]:.3f} to {sam_stats['solidity_range'][1]:.3f})")
                
                # Include refinement info if available
                if sam_stats.get('refinement_cycles_completed', 0) > 0:
                    prompt_parts.append(f"\n**Analysis Parameters:**")
                    prompt_parts.append(f"- Refinement cycles completed: {sam_stats['refinement_cycles_completed']}")
                    prompt_parts.append(f"- Final analysis parameters: {json.dumps(sam_stats['final_parameters'], indent=2)}")
                
                prompt_parts.append("\nSAM Particle Segmentation Overlay (particles outlined in red with centroids and labels):")
                prompt_parts.append({"mime_type": "image/jpeg", "data": sam_overlay_bytes})
            else:
                prompt_parts.append("\n\n(No supplemental SAM particle segmentation analysis results are provided or SAM was disabled/failed)")

            if system_info:
                system_info_text = "\n\nAdditional System Information (Metadata):\n"
                if isinstance(system_info, str):
                    try: system_info_text += json.dumps(json.loads(system_info), indent=2)
                    except json.JSONDecodeError: system_info_text += system_info
                elif isinstance(system_info, dict): system_info_text += json.dumps(system_info, indent=2)
                else: system_info_text += str(system_info)
                prompt_parts.append(system_info_text)
            
            prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
            
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return self._parse_llm_response(response)

        except FileNotFoundError:
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except ImportError as e:
            self.logger.error(f"Missing dependency for SAM analysis: {e}")
            return None, {"error": "Missing SAM dependency", "details": str(e)}
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during SAM image analysis setup: {e}")
            return None, {"error": "An unexpected error occurred during analysis setup", "details": str(e)}

    def analyze_microscopy_image_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims for literature comparison.
        This path always uses image-based analysis with SAM segmentation.
        """
        result_json, error_dict = self._analyze_image_base(
            image_path, system_info, SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict:
            return error_dict
        if result_json is None:
            return {"error": "SAM analysis for claims failed unexpectedly after LLM processing."}

        detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided by LLM.")
        scientific_claims = result_json.get("scientific_claims", [])
        valid_claims = []

        if not isinstance(scientific_claims, list):
            self.logger.warning(f"'scientific_claims' from LLM was not a list: {scientific_claims}")
            scientific_claims = []

        for claim in scientific_claims:
            if isinstance(claim, dict) and all(k in claim for k in ["claim", "scientific_impact", "has_anyone_question", "keywords"]):
                if isinstance(claim.get("keywords"), list):
                    valid_claims.append(claim)
                else:
                    self.logger.warning(f"Claim skipped due to 'keywords' not being a list: {claim}")
            else:
                self.logger.warning(f"Claim skipped due to missing keys or incorrect dict format: {claim}")
        
        if not valid_claims and not detailed_analysis == "Analysis not provided by LLM.":
            self.logger.warning("SAM analysis for claims successful ('detailed_analysis' provided) but no valid claims found or parsed.")
        elif not valid_claims:
             self.logger.warning("LLM call did not yield valid claims or analysis text for SAM claims workflow.")

        return {"full_analysis": detailed_analysis, "claims": valid_claims}