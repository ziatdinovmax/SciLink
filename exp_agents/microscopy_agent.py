import json
import os
from io import BytesIO
from PIL import Image

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .instruct import MICROSCOPY_ANALYSIS_INSTRUCTIONS, MICROSCOPY_CLAIMS_INSTRUCTIONS
from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes

class GeminiMicroscopyAnalysisAgent:
    """
    Agent for analyzing microscopy images using Gemini models
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-pro-exp-03-25"):
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.instructions = MICROSCOPY_ANALYSIS_INSTRUCTIONS
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.safety_settings = {
             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

    def analyze_microscopy_image(self, image_path: str, system_info: dict | str | None = None):
        try:
            pil_image = load_image(image_path)
            preprocessed_img_array = preprocess_image(pil_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            prompt_parts = [self.instructions]
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nImage:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)
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

            # Use the claims-focused instructions
            prompt_parts = [MICROSCOPY_CLAIMS_INSTRUCTIONS]
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nImage:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)
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