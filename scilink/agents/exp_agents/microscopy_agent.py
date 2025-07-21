import json
import os
import logging
import numpy as np

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent

from .human_feedback import SimpleFeedbackMixin


from .instruct import (
    MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    MICROSCOPY_CLAIMS_INSTRUCTIONS,
    FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS,
    MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)
from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes, normalize_and_convert_to_image_bytes

from atomai.stat import SlidingFFTNMF


class MicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing microscopy images using generative AI models
    """

    def __init__(self,
                 google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 fft_nmf_settings: dict | None = None, enable_human_feedback: bool = True):
        super().__init__(google_api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        
        self.fft_nmf_settings = fft_nmf_settings if fft_nmf_settings else {}
        self.RUN_FFT_NMF = self.fft_nmf_settings.get('FFT_NMF_ENABLED', False)
        self.FFT_NMF_AUTO_PARAMS = self.fft_nmf_settings.get('FFT_NMF_AUTO_PARAMS', False)
        self._recommendation_agent = None

    def _get_fft_nmf_params_from_llm(self, image_blob, system_info) -> tuple[int | None, int | None, str | None]:
        """Get FFT/NMF parameters from LLM based on image analysis."""
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: sFFT/NMF PARAMETER ESTIMATION -------------------- 🤖\n")
        self.logger.info("Attempting to get FFT/NMF parameters from LLM...")
        
        prompt_parts = [FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)
        
        if system_info:
            system_info_text = "\n\nAdditional System Information:\n"
            if isinstance(system_info, str):
                try: 
                    system_info_text += json.dumps(json.loads(system_info), indent=2)
                except json.JSONDecodeError: 
                    system_info_text += system_info
            elif isinstance(system_info, dict): 
                system_info_text += json.dumps(system_info, indent=2)
            else: 
                system_info_text += str(system_info)[:1000]
            prompt_parts.append(system_info_text)
            
        prompt_parts.append("\n\nOutput ONLY the JSON object with 'window_size', 'n_components', and 'explanation'")
        
        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            
            self.logger.debug(f"LLM parameter estimation raw response: {response}")
            
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                self.logger.error(f"LLM parameter estimation blocked: {response.prompt_feedback.block_reason}")
                return None, None, None
                
            if response.candidates and response.candidates[0].finish_reason != 1:
                self.logger.warning(f"LLM parameter estimation finished unexpectedly: {response.candidates[0].finish_reason}")

            raw_text_params = response.text
            result_json_params = json.loads(raw_text_params)
            
            window_size_nm = result_json_params.get("window_size_nm")
            n_components = result_json_params.get("n_components")
            explanation = result_json_params.get("explanation", "No explanation provided.")
            
            # Validate parameters
            params_valid = True
            if not isinstance(window_size_nm, (float, int)) or window_size_nm <= 0:
                self.logger.warning(f"LLM invalid window_size_nm: {window_size_nm}")
                params_valid = False
                window_size_nm = None
                
            if not isinstance(n_components, int) or not (1 <= n_components <= 16):
                self.logger.warning(f"LLM invalid n_components: {n_components}")
                params_valid = False
                n_components = None
                
            if not isinstance(explanation, str) or not explanation.strip():
                explanation = "Invalid/empty explanation from LLM."
                
            if params_valid:
                self.logger.info(f"LLM suggested params: window_size_nm={window_size_nm}, n_components={n_components}")
                return window_size_nm, n_components, explanation
                
            return None, None, explanation
            
        except Exception as e:
            self.logger.error(f"LLM call for FFT/NMF params failed: {e}", exc_info=True)
            return None, None, None

    def _run_fft_nmf_analysis(self, image_path: str, window_size: int, n_components: int, window_step: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run sliding FFT + NMF analysis using AtomAI."""
        try:
            self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: SLIDING FFT + NMF ANALYSIS -------------------- 🤖\n")
            self.logger.info("--- Starting Sliding FFT + NMF Analysis (AtomAI) ---")
            
            fft_output_dir = self.fft_nmf_settings.get('output_dir', 'microscopy_analysis')
            os.makedirs(fft_output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
            fft_output_base = os.path.join(fft_output_dir, f"{safe_base_name}_output")
            
            # Create AtomAI analyzer with auto-parameter calculation support
            analyzer = SlidingFFTNMF(
                # Use None to trigger auto-calculation, or use LLM-provided values
                window_size_x=window_size if window_size and window_size > 0 else None,
                window_size_y=window_size if window_size and window_size > 0 else None,
                window_step_x=window_step if window_step and window_step > 0 else None,
                window_step_y=window_step if window_step and window_step > 0 else None,
                
                # Use settings from config
                interpolation_factor=self.fft_nmf_settings.get('interpolation_factor', 2),
                zoom_factor=self.fft_nmf_settings.get('zoom_factor', 2),
                hamming_filter=self.fft_nmf_settings.get('hamming_filter', True),
                components=n_components
            )
            
            # AtomAI's analyze_image method handles both file paths and arrays
            components, abundances = analyzer.analyze_image(image_path, output_path=fft_output_base)
            
            self.logger.info(f"AtomAI FFT+NMF analysis complete. Components: {components.shape}, Abundances: {abundances.shape}")
            self._save_fft_nmf_plots(components, abundances, image_path)

            return components, abundances
            
        except Exception as fft_e:
            self.logger.error(f"AtomAI Sliding FFT + NMF analysis failed: {fft_e}", exc_info=True)
            return None, None

    def _analyze_image_base(self, image_path: str, system_info: dict | str | None, 
                            instruction_prompt: str, 
                            additional_top_level_context: str | None = None) -> tuple[dict | None, dict | None]:
        """
        Internal helper for image-based analysis, including optional FFT/NMF.
        Now uses base class methods for common functionality.
        """
        try:
            # Clear any previous stored images
            self._clear_stored_images()
            # Use base class methods for common operations
            system_info = self._handle_system_info(system_info)
            loaded_image = load_image(image_path)
            nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(system_info, loaded_image.shape)

            preprocessed_img_array, _ = preprocess_image(loaded_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            analysis_images = [
                {"label": "Primary Microscopy Image", "data": image_bytes}
            ]

            components_array, abundances_array = None, None
            if self.RUN_FFT_NMF:
                ws_nm, nc, fft_explanation = None, None, None
                if self.FFT_NMF_AUTO_PARAMS:
                    auto_params = self._get_fft_nmf_params_from_llm(image_blob, system_info)
                    if auto_params: 
                        ws_nm, nc, fft_explanation = auto_params
                        
                if fft_explanation: 
                    self.logger.info(f"LLM Explanation for FFT/NMF params: {fft_explanation}")
                
                # Determine window size in pixels. Prioritize LLM physical size, then fall back to config.
                ws_pixels = self.fft_nmf_settings.get('window_size_x') # Default from config (could be None)

                if ws_nm is not None:
                    if nm_per_pixel is not None and nm_per_pixel > 0:
                        calculated_ws_pixels = int(round(ws_nm / nm_per_pixel))
                        # For optimal FFT performance, find the smallest "good" FFT size
                        # that is greater than or equal to the calculated physical size.
                        good_fft_sizes = [16, 24, 32, 48, 64, 80, 96, 120, 128, 
                                          160, 180, 192, 240, 256, 360, 384, 480, 512]
                        
                        # Find the first size in the list that is >= our calculated size
                        ws_pixels = next((size for size in good_fft_sizes if size >= calculated_ws_pixels), 512) # Default to 512 if too large
                        
                        self.logger.info(f"Using LLM-suggested physical window size: {ws_nm:.2f} nm -> {calculated_ws_pixels} pixels. "
                                         f"Selected optimal FFT size: {ws_pixels} pixels.")
                        
                    else:
                        self.logger.warning(f"LLM suggested a physical window size ({ws_nm} nm), but image scale (nm/pixel) is unknown. Falling back to configured pixel window size.")
                elif ws_pixels:
                    self.logger.info(f"Using default/configured window size of {ws_pixels} pixels.")

                if nc is None: 
                    nc = self.fft_nmf_settings.get('components', 4)
                
                # Calculate step size based on window size (if provided) or use config
                if ws_pixels is not None:
                    step = max(1, ws_pixels // 4) # Standard practice: step = window_size / 4
                else:
                    step = self.fft_nmf_settings.get('window_step_x') # Could be None for auto-calc
                
                components_array, abundances_array = self._run_fft_nmf_analysis(image_path, ws_pixels, nc, step)
            
            # Build prompt
            prompt_parts = [instruction_prompt]
            
            if additional_top_level_context:
                prompt_parts.append("\n\n## Special Considerations for This Analysis (Based on Prior Review):\n")
                prompt_parts.append(additional_top_level_context)
                prompt_parts.append("\nPlease ensure your DFT structure recommendations and scientific justifications specifically address or investigate these special considerations alongside your general analysis of the image features. The priority should be given to structures that elucidate these highlighted aspects.\n")
            
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: 
                analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nPrimary Microscopy Image:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)

            # Add FFT/NMF results if available
            if components_array is not None and abundances_array is not None:
                prompt_parts.append("\n\nSupplemental Analysis Data (Sliding FFT + NMF Grayscale Images):")
                num_components = min(components_array.shape[0], abundances_array.shape[0])
                img_format = 'JPEG'
                self.logger.info(f"Adding {num_components} NMF components/abundances as {img_format} to prompt.")
                
                for i in range(num_components):
                    try:
                        comp_bytes = normalize_and_convert_to_image_bytes(components_array[i], log_scale=True, format=img_format)
                        prompt_parts.append(f"\nNMF Component {i+1} (Frequency Pattern - Grayscale):")
                        prompt_parts.append({"mime_type": f"image/{img_format.lower()}", "data": comp_bytes})
                        
                        abun_bytes = normalize_and_convert_to_image_bytes(abundances_array[i], log_scale=False, format=img_format)
                        prompt_parts.append(f"\nNMF Abundance Map {i+1} (Spatial Distribution - Grayscale):")
                        prompt_parts.append({"mime_type": f"image/{img_format.lower()}", "data": abun_bytes})
                    
                        analysis_images.append({
                            "label": f"NMF Abundance Map {i+1} (Spatial Distribution - Grayscale)",
                            "data": abun_bytes
                        })
                    
                    except Exception as convert_e:
                        self.logger.error(f"Failed to convert NMF result {i+1} to image bytes: {convert_e}")
                        prompt_parts.append(f"\n(Error converting NMF result {i+1} image for prompt)")
            else:
                prompt_parts.append("\n\n(No supplemental FFT/NMF image analysis results are provided or FFT/NMF was disabled/failed)")

            analysis_metadata = {
                "image_path": image_path,
                "system_info": system_info,
                "fft_nmf_enabled": self.RUN_FFT_NMF,
                "num_stored_images": len(analysis_images)
            }
            self._store_analysis_images(analysis_images, analysis_metadata)
            
            # Use base class method for system info prompt section
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)
            
            prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
            
            self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: INTERPRETING RESULTS -------------------- 🤖\n")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return self._parse_llm_response(response)  # Using base class method

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except ImportError as e:
            self._clear_stored_images()
            self.logger.error(f"Missing dependency for image processing: {e}")
            return None, {"error": "Missing image processing dependency", "details": str(e)}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"An unexpected error occurred during image analysis setup or FFT/NMF: {e}")
            return None, {"error": "An unexpected error occurred during analysis setup", "details": str(e)}

    def _save_fft_nmf_plots(self, components: np.ndarray, abundances: np.ndarray, image_path: str):
        """Creates and saves nice plots for each NMF component and its abundance map."""
        import matplotlib.pyplot as plt
        from datetime import datetime

        try:
            # You can define a separate directory for these visualizations
            output_dir = self.fft_nmf_settings.get('visualization_dir', 'fft_nmf_visualizations')
            os.makedirs(output_dir, exist_ok=True)
            
            num_components = components.shape[0]
            self.logger.info(f"Creating and saving {num_components} NMF visualization plots...")

            # --- Create a clean base name and unique timestamp ---
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for i in range(num_components):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'NMF Result {i+1}/{num_components}', fontsize=16)

                # --- Left Plot: NMF Component (FFT pattern) ---
                # Using a log scale makes the FFT features much more visible
                comp_img = np.log1p(components[i])
                ax1.imshow(comp_img, cmap='inferno')
                ax1.set_title(f'Component {i+1} (FFT Pattern)')
                ax1.axis('off')

                # --- Right Plot: NMF Abundance Map ---
                im = ax2.imshow(abundances[i], cmap='inferno')
                ax2.set_title(f'Abundance Map {i+1} (Spatial Location)')
                ax2.axis('off')
                fig.colorbar(im, ax=ax2, label="Abundance", fraction=0.046, pad=0.04)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for the main title

                # --- Save the figure to a file ---
                plot_filename = f"{safe_base_name}_nmf_plot_{i+1}_{timestamp}.png"
                plot_filepath = os.path.join(output_dir, plot_filename)
                plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
                plt.close(fig) # Close the figure to free up memory

            self.logger.info(f"✅ Saved NMF visualizations to: {output_dir}")

        except Exception as e:
            self.logger.error(f"Failed to create or save NMF plots: {e}", exc_info=True)
    
    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        """
        Analyze microscopy image to generate DFT structure recommendations.
        Supports both image-based and text-based analysis paths.
        Now uses base class validation methods for recommendations.
        """
        result_json, error_dict = None, None
        output_analysis_key = "detailed_analysis" 

        # Text-Only path
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")
            
            # Lazy initialization: create the agent only when it's first needed.
            if not self._recommendation_agent:
                self._recommendation_agent = RecommendationAgent(self.google_api_key, self.model_name)
            
            # Correctly delegate the call with all necessary arguments
            return self._recommendation_agent.generate_dft_recommendations_from_text(
                cached_detailed_analysis=cached_detailed_analysis,
                additional_prompt_context=additional_prompt_context,
                system_info=system_info
            )

        # Image-based path
        elif image_path:
            self.logger.info("Generating DFT recommendations from image analysis.")
            instruction_prompt_text = MICROSCOPY_ANALYSIS_INSTRUCTIONS # Standard image analysis prompt
            # additional_prompt_context (novelty string) is passed to _analyze_image_base to be appended
            result_json, error_dict = self._analyze_image_base(
                image_path, system_info, instruction_prompt_text, additional_top_level_context=additional_prompt_context
            )
        else:
            # Neither path is viable
            return {"error": "Either image_path or (cached_detailed_analysis AND additional_prompt_context) must be provided for DFT recommendations."}

        if error_dict:
            return error_dict # Return error if LLM call or parsing failed
        if result_json is None: # Safeguard, should be covered by error_dict
            return {"error": "Analysis failed unexpectedly after LLM processing."}

        # Use the determined key to fetch the main textual output from LLM
        analysis_output_text = result_json.get(output_analysis_key, "Analysis/Reasoning not provided by LLM.")
        recommendations = result_json.get("structure_recommendations", [])
        
        # Use base class validation method
        sorted_recommendations = self._validate_structure_recommendations(recommendations)

        if not sorted_recommendations and not analysis_output_text == "Analysis/Reasoning not provided by LLM.":
             self.logger.warning(f"LLM call successful ('{output_analysis_key}' provided) but no valid recommendations found or parsed.")
        elif not sorted_recommendations:
            self.logger.warning("LLM call did not yield valid recommendations or analysis text.")

        # Return a consistent key for the main textual output for the calling script
        return {"analysis_summary_or_reasoning": analysis_output_text, "recommendations": sorted_recommendations}

    def analyze_microscopy_image_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims for literature comparison.
        This path always uses image-based analysis.
        Now uses base class validation methods.
        """
        result_json, error_dict = self._analyze_image_base(
            image_path, system_info, MICROSCOPY_CLAIMS_INSTRUCTIONS, additional_top_level_context=None # No special context needed here usually
        )

        if error_dict:
            return error_dict
        if result_json is None: # Safeguard
            return {"error": "Analysis for claims failed unexpectedly after LLM processing."}

        detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided by LLM.")
        scientific_claims = result_json.get("scientific_claims", [])
        
        # Use base class validation method
        valid_claims = self._validate_scientific_claims(scientific_claims)

        if not valid_claims and not detailed_analysis == "Analysis not provided by LLM.":
            self.logger.warning("Analysis for claims successful ('detailed_analysis' provided) but no valid claims found or parsed.")
        elif not valid_claims:
             self.logger.warning("LLM call did not yield valid claims or analysis text for claims workflow.")

        initial_result = {"detailed_analysis": detailed_analysis, "scientific_claims": valid_claims}
        return self._apply_feedback_if_enabled(
            initial_result, 
            image_path=image_path, 
            system_info=system_info
        )
    
    def _get_claims_instruction_prompt(self) -> str:
        return MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
