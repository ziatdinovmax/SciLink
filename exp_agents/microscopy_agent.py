import json
import os
from io import BytesIO
from PIL import Image
import logging
import numpy as np

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

# Import all necessary instructions
from .instruct import (
    MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    MICROSCOPY_CLAIMS_INSTRUCTIONS,
    FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS,
    TEXT_ONLY_DFT_RECOMMENDATION_INSTRUCTIONS # Ensure this is defined in instruct.py
)
from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes, normalize_and_convert_to_image_bytes
from .fft_nmf_analyzer import SlidingFFTNMF


class GeminiMicroscopyAnalysisAgent:
    """
    Agent for analyzing microscopy images using Gemini models.
    Refactored to support both image-based and text-based DFT recommendations.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-1.5-pro-preview-0514", fft_nmf_settings: dict | None = None): # Updated model name as an example
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        self.logger = logging.getLogger(__name__)
        self.fft_nmf_settings = fft_nmf_settings if fft_nmf_settings else {} # Ensure it's a dict
        self.RUN_FFT_NMF = self.fft_nmf_settings.get('FFT_NMF_ENABLED', False)
        self.FFT_NMF_AUTO_PARAMS = self.fft_nmf_settings.get('FFT_NMF_AUTO_PARAMS', False)

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
            elif response.candidates and response.candidates[0].finish_reason != 1: # 1 == Stop
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
            # for i, part in enumerate(prompt_parts): # For detailed logging if needed
            # self.logger.debug(f"Part {i}: {str(part)[:200]}...")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return self._parse_llm_response(response)
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during text-based LLM call: {e}")
            return None, {"error": "An unexpected error occurred during text-based LLM call", "details": str(e)}

    def _get_fft_nmf_params_from_llm(self, image_blob, system_info) -> tuple[int | None, int | None, str | None]:
        self.logger.info("Attempting to get FFT/NMF parameters from LLM...")
        prompt_parts = [FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)
        if system_info:
            system_info_text = "\n\nAdditional System Information:\n"
            if isinstance(system_info, str):
                try: system_info_text += json.dumps(json.loads(system_info), indent=2)
                except json.JSONDecodeError: system_info_text += system_info
            elif isinstance(system_info, dict): system_info_text += json.dumps(system_info, indent=2)
            else: system_info_text += str(system_info)[:1000]
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
            window_size = result_json_params.get("window_size")
            n_components = result_json_params.get("n_components")
            explanation = result_json_params.get("explanation", "No explanation provided.")
            params_valid = True
            if not isinstance(window_size, int) or window_size <= 0:
                self.logger.warning(f"LLM invalid window_size: {window_size}")
                params_valid = False; window_size = None
            if not isinstance(n_components, int) or not (1 <= n_components <= 16):
                self.logger.warning(f"LLM invalid n_components: {n_components}")
                params_valid = False; n_components = None
            if not isinstance(explanation, str) or not explanation.strip():
                explanation = "Invalid/empty explanation from LLM."
            if params_valid:
                self.logger.info(f"LLM suggested params: window_size={window_size}, n_components={n_components}")
                return window_size, n_components, explanation
            return None, None, explanation
        except Exception as e:
            self.logger.error(f"LLM call for FFT/NMF params failed: {e}", exc_info=True)
            return None, None, None


    def _run_fft_nmf_analysis(self, image_path: str, window_size: int, n_components: int, window_step: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        try:
            self.logger.info("--- Starting Sliding FFT + NMF Analysis ---")
            fft_output_dir = self.fft_nmf_settings.get('output_dir', 'fft_nmf_results')
            os.makedirs(fft_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
            fft_output_base = os.path.join(fft_output_dir, f"{safe_base_name}_analysis")
            analyzer = SlidingFFTNMF(
                window_size_x=window_size, window_size_y=window_size,
                window_step_x=window_step, window_step_y=window_step,
                interpolation_factor=self.fft_nmf_settings.get('interpolation_factor', 2),
                zoom_factor=self.fft_nmf_settings.get('zoom_factor', 2),
                hamming_filter=self.fft_nmf_settings.get('hamming_filter', True),
                components=n_components
            )
            components, abundances = analyzer.analyze_image(image_path, output_path=fft_output_base)
            self.logger.info(f"FFT+NMF analysis complete. Components: {components.shape}, Abundances: {abundances.shape}")
            return components, abundances
        except Exception as fft_e:
            self.logger.error(f"Sliding FFT + NMF analysis failed: {fft_e}", exc_info=True)
        return None, None

    def _analyze_image_base(self, image_path: str, system_info: dict | str | None, 
                            instruction_prompt: str, 
                            additional_top_level_context: str | None = None) -> tuple[dict | None, dict | None]:
        """
        Internal helper for image-based analysis, including optional FFT/NMF.
        """
        try:
            loaded_image = load_image(image_path)
            preprocessed_img_array = preprocess_image(loaded_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            components_array, abundances_array = None, None
            if self.RUN_FFT_NMF:
                ws, nc, fft_explanation = None, None, None
                if self.FFT_NMF_AUTO_PARAMS:
                    auto_params = self._get_fft_nmf_params_from_llm(image_blob, system_info) # Pass image_blob here
                    if auto_params: ws, nc, fft_explanation = auto_params
                if fft_explanation: self.logger.info(f"LLM Explanation for FFT/NMF params: {fft_explanation}")
                if ws is None: ws = self.fft_nmf_settings.get('window_size_x', preprocessed_img_array.shape[0] // 16 if preprocessed_img_array is not None else 64)
                if nc is None: nc = self.fft_nmf_settings.get('components', 4)
                step = max(1, ws // 4)
                components_array, abundances_array = self._run_fft_nmf_analysis(image_path, ws, nc, step)

            prompt_parts = [instruction_prompt]
            if additional_top_level_context:
                prompt_parts.append("\n\n## Special Considerations for This Analysis (Based on Prior Review):\n")
                prompt_parts.append(additional_top_level_context)
                prompt_parts.append("\nPlease ensure your DFT structure recommendations and scientific justifications specifically address or investigate these special considerations alongside your general analysis of the image features. The priority should be given to structures that elucidate these highlighted aspects.\n")
            
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nPrimary Microscopy Image:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)

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
                    except Exception as convert_e:
                        self.logger.error(f"Failed to convert NMF result {i+1} to image bytes: {convert_e}")
                        prompt_parts.append(f"\n(Error converting NMF result {i+1} image for prompt)")
            else:
                prompt_parts.append("\n\n(No supplemental FFT/NMF image analysis results are provided or FFT/NMF was disabled/failed)")

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
        except ImportError as e: # Should not happen if dependencies are met
            self.logger.error(f"Missing dependency for image processing: {e}")
            return None, {"error": "Missing image processing dependency", "details": str(e)}
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during image analysis setup or FFT/NMF: {e}")
            return None, {"error": "An unexpected error occurred during analysis setup", "details": str(e)}

    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        result_json, error_dict = None, None
        # Determine the key for the main textual output from LLM based on the path taken
        output_analysis_key = "full_analysis" # Default for image-based path

        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Generating DFT recommendations from cached analysis and novelty context.")
            instruction_prompt_text = TEXT_ONLY_DFT_RECOMMENDATION_INSTRUCTIONS # Use the new text-only prompt
            output_analysis_key = "detailed_reasoning_for_recommendations" # Expected from new prompt

            prompt_list_for_llm = [instruction_prompt_text]
            prompt_list_for_llm.append("\n\n--- Start of Cached Initial Image Analysis ---\n")
            prompt_list_for_llm.append(cached_detailed_analysis)
            prompt_list_for_llm.append("\n--- End of Cached Initial Image Analysis ---\n")

            prompt_list_for_llm.append("\n\n--- Start of Special Considerations (e.g., Novelty Insights) ---\n")
            prompt_list_for_llm.append(additional_prompt_context)
            prompt_list_for_llm.append("\n--- End of Special Considerations ---\n")

            if system_info:
                system_info_text_part = "\n\nAdditional System Information (Metadata):\n"
                if isinstance(system_info, str):
                    try: system_info_text_part += json.dumps(json.loads(system_info), indent=2)
                    except json.JSONDecodeError: system_info_text_part += system_info
                elif isinstance(system_info, dict): system_info_text_part += json.dumps(system_info, indent=2)
                else: system_info_text_part += str(system_info)
                prompt_list_for_llm.append(system_info_text_part)
            
            prompt_list_for_llm.append("\n\nProvide your DFT structure recommendations strictly in the requested JSON format based on the above text.")
            result_json, error_dict = self._generate_json_from_text_parts(prompt_list_for_llm)

        elif image_path:
            self.logger.info("Generating DFT recommendations from image analysis.")
            instruction_prompt_text = MICROSCOPY_ANALYSIS_INSTRUCTIONS # Standard image analysis prompt
            # additional_prompt_context (novelty string) is passed to _analyze_image_base to be appended
            result_json, error_dict = self._analyze_image_base(
                image_path, system_info, instruction_prompt_text, additional_top_level_context=additional_prompt_context
            )
            # output_analysis_key remains "full_analysis"
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
        
        valid_recommendations = []
        if not isinstance(recommendations, list):
            self.logger.warning(f"'structure_recommendations' from LLM was not a list: {recommendations}")
            recommendations = [] 

        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in ["description", "scientific_interest", "priority"]):
                if isinstance(rec.get("priority"), int):
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Recommendation skipped due to invalid priority type (expected int): {rec.get('priority')}. Recommendation: {rec}")
            else:
                self.logger.warning(f"Recommendation skipped due to missing keys or incorrect dict format: {rec}")
        
        sorted_recommendations = sorted(valid_recommendations, key=lambda x: x.get("priority", 99))

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
        valid_claims = []

        if not isinstance(scientific_claims, list):
            self.logger.warning(f"'scientific_claims' from LLM was not a list: {scientific_claims}")
            scientific_claims = []

        for claim in scientific_claims:
            if isinstance(claim, dict) and all(k in claim for k in ["claim", "scientific_impact", "has_anyone_question", "keywords"]):
                # Optionally add validation for keywords being a list of strings etc.
                if isinstance(claim.get("keywords"), list):
                    valid_claims.append(claim)
                else:
                    self.logger.warning(f"Claim skipped due to 'keywords' not being a list: {claim}")
            else:
                self.logger.warning(f"Claim skipped due to missing keys or incorrect dict format: {claim}")
        
        if not valid_claims and not detailed_analysis == "Analysis not provided by LLM.":
            self.logger.warning("Analysis for claims successful ('detailed_analysis' provided) but no valid claims found or parsed.")
        elif not valid_claims:
             self.logger.warning("LLM call did not yield valid claims or analysis text for claims workflow.")


        return {"full_analysis": detailed_analysis, "claims": valid_claims}