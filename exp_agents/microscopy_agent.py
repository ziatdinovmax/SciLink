import json
import os
from io import BytesIO
from PIL import Image
import logging
import numpy as np

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .instruct import MICROSCOPY_ANALYSIS_INSTRUCTIONS, MICROSCOPY_CLAIMS_INSTRUCTIONS, FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS
from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes, normalize_and_convert_to_image_bytes
from .fft_nmf_analyzer import SlidingFFTNMF


class GeminiMicroscopyAnalysisAgent:
    """
    Agent for analyzing microscopy images using Gemini models
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-pro-exp-03-25", fft_nmf_settings: dict | None = None):
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
        self.fft_nmf_settings = fft_nmf_settings
        self.RUN_FFT_NMF = fft_nmf_settings.get('FFT_NMF_ENABLED', False) if fft_nmf_settings else False
        self.FFT_NMF_AUTO_PARAMS = fft_nmf_settings.get('FFT_NMF_AUTO_PARAMS', False) if fft_nmf_settings else False

    def analyze_microscopy_image(self, image_path: str, system_info: dict | str | None = None):
        try:
            pil_image = load_image(image_path)
            preprocessed_img_array = preprocess_image(pil_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            # Run optional FFT+NMF analysis
            components_array, abundances_array = None, None
            if self.RUN_FFT_NMF:
                ws = None
                nc = None
                if self.FFT_NMF_AUTO_PARAMS:
                    ws, nc, explanation = self._get_fft_nmf_params_from_llm(image_blob, system_info) 
                if explanation:
                    self.logger.info(f"LLM Explanation for FFT/NMF params: {explanation}")
                # Use config defaults if LLM fails or auto-params is off
                if ws is None:
                    ws = self.fft_nmf_settings.get('FFT_NMF_WINDOW_SIZE_X', preprocessed_img_array.shape[0]//16)
                if nc is None:
                    nc = self.fft_nmf_settings.get('FFT_NMF_COMPONENTS', 4)
                # Calculate step size (ensure minimum of 1)
                step = max(1, ws // 4)
                # Run analysis with determined params
                components_array, abundances_array = self._run_fft_nmf_analysis(image_path, ws, nc, step)

            prompt_parts = [MICROSCOPY_ANALYSIS_INSTRUCTIONS]
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nImage:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)

            if components_array is not None and abundances_array is not None:
                prompt_parts.append("\n\nSupplemental Analysis Data (Sliding FFT + NMF Grayscale Images):")
                num_components = min(components_array.shape[0], abundances_array.shape[0]) # Use min in case of mismatch
                img_format = 'JPEG'
                self.logger.info(f"Adding {num_components} NMF components/abundances as {img_format} images to prompt.")

                for i in range(num_components):
                    try:
                        # Convert component i (log scale useful for FFT patterns)
                        comp_bytes = normalize_and_convert_to_image_bytes(
                            components_array[i], log_scale=True, format=img_format
                        )
                        prompt_parts.append(f"\nNMF Component {i+1} (Frequency Pattern - Grayscale):")
                        prompt_parts.append({"mime_type": f"image/{img_format.lower()}", "data": comp_bytes})

                        # Convert abundance map i (linear scale usually better)
                        abun_bytes = normalize_and_convert_to_image_bytes(
                            abundances_array[i], log_scale=False, format=img_format
                        )
                        prompt_parts.append(f"\nNMF Abundance Map {i+1} (Spatial Distribution - Grayscale):")
                        prompt_parts.append({"mime_type": f"image/{img_format.lower()}", "data": abun_bytes})

                    except Exception as convert_e:
                            self.logger.error(f"Failed to convert NMF result {i+1} to image bytes: {convert_e}")
                            prompt_parts.append(f"\n(Error converting NMF result {i+1} image for prompt)")

            else:
                prompt_parts.append("\n\n(No supplemental image analysis results are provided)")
            
            if system_info:
                system_info_text = "\n\nAdditional System Information:\n"
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

            try:
                raw_text = response.text
                first_brace_index = raw_text.find('{')
                last_brace_index = raw_text.rfind('}')
                if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                    json_string = raw_text[first_brace_index : last_brace_index + 1]
                    result_json = json.loads(json_string)
                else:
                    raise ValueError("Could not find valid JSON object delimiters '{' and '}' in the response text.")
                if "detailed_analysis" not in result_json or "structure_recommendations" not in result_json:
                    raise ValueError("JSON response missing required keys ('detailed_analysis', 'structure_recommendations')")
            except (json.JSONDecodeError, AttributeError, IndexError, ValueError) as e:
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason; print(f"Request blocked due to: {block_reason}")
                     return {"error": f"Content blocked by safety filters", "details": f"Reason: {block_reason}"}
                 if response.candidates and response.candidates[0].finish_reason != 1:
                     finish_reason = response.candidates[0].finish_reason; print(f"Generation finished unexpectedly: {finish_reason}")
                 error_raw_text = response.text if hasattr(response, 'text') else 'N/A'
                 print(f"Error parsing Gemini JSON response: {e}")
                 parsed_substring = json_string if 'json_string' in locals() else 'N/A'
                 print(f"Attempted to parse substring: {parsed_substring[:500]}...")
                 print(f"Original Raw response text: {error_raw_text[:500]}...")
                 return {"error": "Failed to parse valid JSON from LLM response", "details": str(e), "raw_response": error_raw_text}
            except Exception as e:
                 print(f"Unexpected error processing response: {e}"); print(f"Full response object: {response}")
                 return {"error": "Unexpected error processing LLM response", "details": str(e)}

            detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided.")
            recommendations = result_json.get("structure_recommendations", [])
            valid_recommendations = []
            if isinstance(recommendations, list):
                for rec in recommendations:
                    if isinstance(rec, dict) and all(k in rec for k in ["description", "scientific_interest", "priority"]):
                         if isinstance(rec.get("priority"), int): valid_recommendations.append(rec)
                         else: print(f"Warning: Recommendation skipped due to invalid priority type: {rec}")
                    else: print(f"Warning: Recommendation skipped due to missing keys or incorrect format: {rec}")
                sorted_recommendations = sorted(valid_recommendations, key=lambda x: x.get("priority", 99))
            else:
                print(f"Warning: 'structure_recommendations' was not a list: {recommendations}"); sorted_recommendations = []
            return {"full_analysis": detailed_analysis, "recommendations": sorted_recommendations}
        except FileNotFoundError: return {"error": "Image file not found", "details": f"Path: {image_path}"}
        except ImportError as e: return {"error": "Missing dependency", "details": str(e)}
        except Exception as e: print(f"An unexpected error occurred in analyze_microscopy_image: {e}"); return {"error": "An unexpected error occurred during analysis", "details": str(e)}


    def analyze_microscopy_image_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims for literature comparison.
        Uses the claims-focused instructions rather than the structure recommendations.
        """
        try:
            pil_image = load_image(image_path)
            preprocessed_img_array = preprocess_image(pil_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            # Run optional FFT+NMF analysis
            components_array, abundances_array = None, None
            if self.RUN_FFT_NMF:
                ws = None
                nc = None
                if self.FFT_NMF_AUTO_PARAMS:
                    ws, nc, explanation = self._get_fft_nmf_params_from_llm(image_blob, system_info) 
                if explanation:
                    self.logger.info(f"LLM Explanation for FFT/NMF params: {explanation}")
                # Use config defaults if LLM fails or auto-params is off
                if ws is None:
                    ws = self.fft_nmf_settings.get('FFT_NMF_WINDOW_SIZE_X', preprocessed_img_array.shape[0]//16)
                if nc is None:
                    nc = self.fft_nmf_settings.get('FFT_NMF_COMPONENTS', 4)
                # Calculate step size (ensure minimum of 1)
                step = max(1, ws // 4)
                # Run analysis with determined params
                components_array, abundances_array = self._run_fft_nmf_analysis(image_path, ws, nc, step)

            # Use the claims-focused instructions
            prompt_parts = [MICROSCOPY_CLAIMS_INSTRUCTIONS]
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nImage:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)

            if components_array is not None and abundances_array is not None:
                prompt_parts.append("\n\nSupplemental Analysis Data (Sliding FFT + NMF Grayscale Images):")
                num_components = min(components_array.shape[0], abundances_array.shape[0]) # Use min in case of mismatch
                img_format = 'JPEG'
                self.logger.info(f"Adding {num_components} NMF components/abundances as {img_format} images to prompt.")

                for i in range(num_components):
                    try:
                        # Convert component i (log scale useful for FFT patterns)
                        comp_bytes = normalize_and_convert_to_image_bytes(
                            components_array[i], log_scale=True, format=img_format
                        )
                        prompt_parts.append(f"\nNMF Component {i+1} (Frequency Pattern - Grayscale):")
                        prompt_parts.append({"mime_type": f"image/{img_format.lower()}", "data": comp_bytes})

                        # Convert abundance map i (linear scale usually better)
                        abun_bytes = normalize_and_convert_to_image_bytes(
                            abundances_array[i], log_scale=False, format=img_format
                        )
                        prompt_parts.append(f"\nNMF Abundance Map {i+1} (Spatial Distribution - Grayscale):")
                        prompt_parts.append({"mime_type": f"image/{img_format.lower()}", "data": abun_bytes})

                    except Exception as convert_e:
                            self.logger.error(f"Failed to convert NMF result {i+1} to image bytes: {convert_e}")
                            prompt_parts.append(f"\n(Error converting NMF result {i+1} image for prompt)")

            else:
                prompt_parts.append("\n\n(No supplemental image analysis results are provided)")

            if system_info:
                system_info_text = "\n\nAdditional System Information:\n"
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

            try:
                raw_text = response.text
                first_brace_index = raw_text.find('{')
                last_brace_index = raw_text.rfind('}')
                if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                    json_string = raw_text[first_brace_index : last_brace_index + 1]
                    result_json = json.loads(json_string)
                else:
                    raise ValueError("Could not find valid JSON object delimiters '{' and '}' in the response text.")
                if "detailed_analysis" not in result_json or "scientific_claims" not in result_json:
                    raise ValueError("JSON response missing required keys ('detailed_analysis', 'scientific_claims')")
            except (json.JSONDecodeError, AttributeError, IndexError, ValueError) as e:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason; print(f"Request blocked due to: {block_reason}")
                    return {"error": f"Content blocked by safety filters", "details": f"Reason: {block_reason}"}
                if response.candidates and response.candidates[0].finish_reason != 1:
                    finish_reason = response.candidates[0].finish_reason; print(f"Generation finished unexpectedly: {finish_reason}")
                error_raw_text = response.text if hasattr(response, 'text') else 'N/A'
                print(f"Error parsing Gemini JSON response: {e}")
                parsed_substring = json_string if 'json_string' in locals() else 'N/A'
                print(f"Attempted to parse substring: {parsed_substring[:500]}...")
                print(f"Original Raw response text: {error_raw_text[:500]}...")
                return {"error": "Failed to parse valid JSON from LLM response", "details": str(e), "raw_response": error_raw_text}
            except Exception as e:
                print(f"Unexpected error processing response: {e}"); print(f"Full response object: {response}")
                return {"error": "Unexpected error processing LLM response", "details": str(e)}

            detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided.")
            scientific_claims = result_json.get("scientific_claims", [])
            valid_claims = []
            if isinstance(scientific_claims, list):
                for claim in scientific_claims:
                    if isinstance(claim, dict) and all(k in claim for k in ["claim", "scientific_impact", "has_anyone_question", "keywords"]):
                        valid_claims.append(claim)
                    else:
                        print(f"Warning: Claim skipped due to missing keys or incorrect format: {claim}")
            else:
                print(f"Warning: 'scientific_claims' was not a list: {scientific_claims}")
                
            return {"full_analysis": detailed_analysis, "claims": valid_claims}
        except FileNotFoundError:
            return {"error": "Image file not found", "details": f"Path: {image_path}"}
        except ImportError as e:
            return {"error": "Missing dependency", "details": str(e)}
        except Exception as e:
            print(f"An unexpected error occurred in analyze_microscopy_image_for_claims: {e}")
            return {"error": "An unexpected error occurred during analysis", "details": str(e)}
        

    def _run_fft_nmf_analysis(self, image_path: str, window_size: int, n_components: int, window_step: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Runs the Sliding FFT + NMF analysis if enabled and available.
        Returns tuple (components_array, abundances_array) or (None, None).
        Optionally saves .npy files and visualization plots based on SlidingFFTNMF implementation.
        """
        try:
            self.logger.info("--- Starting Sliding FFT + NMF Analysis ---")
            fft_output_dir = self.fft_nmf_settings.get('FFT_NMF_OUTPUT_DIR', 'fft_nmf_results')
            os.makedirs(fft_output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
            fft_output_base = os.path.join(fft_output_dir, f"{safe_base_name}_analysis")

            analyzer = SlidingFFTNMF(
                    window_size_x=window_size,
                    window_size_y=window_size,
                    window_step_x=window_step,
                    window_step_y=window_step,
                    interpolation_factor=self.fft_nmf_settings.get('FFT_NMF_INTERPOLATION_FACTOR', 2),
                    zoom_factor=self.fft_nmf_settings.get('FFT_NMF_ZOOM_FACTOR', 2),
                    hamming_filter=self.fft_nmf_settings.get('FFT_NMF_HAMMING_FILTER', True),
                    components=n_components
            )
            components, abundances = analyzer.analyze_image(image_path, output_path=fft_output_base)

            self.logger.info(f"FFT+NMF analysis complete. Components shape: {components.shape}, Abundances shape: {abundances.shape}")
            return components, abundances

        except Exception as fft_e:
            self.logger.error(f"Sliding FFT + NMF analysis failed: {fft_e}", exc_info=True)

        return None, None # Return None if disabled or failed
    

    def _get_fft_nmf_params_from_llm(self, image_blob, system_info) -> tuple[int | None, int | None]:
        """
        Makes a separate LLM call to estimate FFT/NMF parameters (window_size, n_components).
        """
        self.logger.info("Attempting to get FFT/NMF parameters from LLM...")

        # 1. Construct the Prompt
        prompt_parts = [FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)

        if system_info:
            system_info_text = "\n\nAdditional System Information:\n"
            if isinstance(system_info, str):
                try:
                    # Try parsing if it's a JSON string, otherwise use raw
                    system_info_text += json.dumps(json.loads(system_info), indent=2)
                except json.JSONDecodeError:
                    system_info_text += system_info
            elif isinstance(system_info, dict):
                system_info_text += json.dumps(system_info, indent=2)
            else:
                system_info_text += str(system_info)
            prompt_parts.append(system_info_text)

        # Final instruction reinforcing the output format
        prompt_parts.append("\n\nOutput ONLY the JSON object with 'window_size', 'n_components', and 'explanation'")

        # 2. Configure API Call for JSON Response
        # Explicitly ask the API to return JSON. This improves reliability.
        param_gen_config = GenerationConfig(response_mime_type="application/json")

        # 3. Call the LLM
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings, # Reuse safety settings
            )
            self.logger.debug(f"LLM parameter estimation raw response: {response}") # Log raw for debugging

            # 4. Parse and Validate the Response
            # Check for safety blocks first
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 self.logger.error(f"LLM parameter estimation request blocked by safety filters: {block_reason}")
                 return None, None # Fallback

            # Check for unexpected finish reasons
            if response.candidates and response.candidates[0].finish_reason != 1: # Assuming 1 is STOP/normal
                finish_reason = response.candidates[0].finish_reason
                self.logger.warning(f"LLM parameter estimation finished unexpectedly: {finish_reason}. Trying to parse anyway.")
                # Don't necessarily return None yet, might still have parsable text

            # Attempt to parse the JSON payload
            raw_text = response.text
            result_json = json.loads(raw_text) # This should work if API respects mime type

            window_size = result_json.get("window_size")
            n_components = result_json.get("n_components")
            explanation = result_json.get("explanation")

            # --- Validation ---
            params_valid = True
            if not isinstance(window_size, int) or window_size <= 0:
                self.logger.warning(f"LLM returned invalid window_size: {window_size} (Type: {type(window_size)})")
                params_valid = False

            if not isinstance(n_components, int) or not (2 <= n_components <= 10):
                self.logger.warning(f"LLM returned invalid n_components: {n_components} (Type: {type(n_components)}). Expected int between 2-10.")
                params_valid = False

            explanation_valid = isinstance(explanation, str) and explanation.strip() != ""
            if not explanation_valid:
                self.logger.warning(f"LLM did not provide a valid explanation string: {explanation}")
                explanation = None # Set explanation to None if invalid, but params might still be okay
            # --- End Validation ---

            if params_valid:
                 self.logger.info(f"LLM successfully suggested parameters: window_size={window_size}, n_components={n_components}")
                 return window_size, n_components, explanation
            else:
                 self.logger.warning("LLM parameter suggestions failed validation. Falling back to defaults.")
                 return None, None, None # Fallback if essential params are invalid

        # Handle specific errors during parsing or API call
        except json.JSONDecodeError as json_e:
            self.logger.error(f"LLM parameter estimation response was not valid JSON: {json_e}. Raw text: '{raw_text[:500]}...'")
            return None, None
        except (AttributeError, IndexError, ValueError, TypeError) as parse_e:
             self.logger.error(f"Error parsing or validating LLM parameter response: {parse_e}", exc_info=True)
             return None, None
        # Handle broader API or network errors
        except Exception as e:
            self.logger.error(f"LLM call for FFT/NMF parameters failed unexpectedly: {e}", exc_info=True)
            return None, None # Fallback on any error