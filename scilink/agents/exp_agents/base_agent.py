import json
import logging

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from ...auth import get_api_key, APIKeyNotFoundError


class BaseAnalysisAgent:
    """
    Base class for analysis agents
    """
    
    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05"):
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
        self.google_api_key = google_api_key 
        self.model_name = model_name

    def _parse_llm_response(self, response) -> tuple[dict | None, dict | None]:
        """
        Parses the LLM response, expecting JSON.
        Shared implementation used by all agents.
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
            self.logger.error(f"Error parsing LLM JSON response: {e}")
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
        Shared implementation used by multiple agents.
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

    def _handle_system_info(self, system_info: dict | str | None) -> dict:
        """
        Handle system_info input (can be dict, file path, or None).
        Converts to dict format for consistent processing.
        """
        if isinstance(system_info, str):
            try:
                with open(system_info, 'r') as f:
                    system_info = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self.logger.error(f"Error loading system_info from {system_info}: {e}")
                system_info = {} # Proceed without system info if loading fails
        elif system_info is None:
            system_info = {} # Ensure it's a dict for easier access later
        
        return system_info

    def _calculate_spatial_scale(self, system_info: dict, image_shape: tuple) -> tuple[float | None, float | None]:
        """
        Calculate nm/pixel and field of view from system metadata.
        
        Args:
            system_info: Dictionary containing spatial metadata
            image_shape: (height, width) of the loaded image
            
        Returns:
            tuple: (nm_per_pixel, fov_in_nm) or (None, None) if calculation fails
        """
        # Determine physical scale (nm/pixel) from metadata, mirroring spectroscopy agent's approach
        nm_per_pixel = None
        fov_in_nm = None
        
        if isinstance(system_info, dict) and 'spatial_info' in system_info and isinstance(system_info.get('spatial_info'), dict):
            spatial = system_info['spatial_info']
            fov_x = spatial.get("field_of_view_x")
            units = spatial.get("field_of_view_units", "nm") # Default to nm

            if fov_x is not None and isinstance(fov_x, (int, float)) and fov_x > 0:
                h, w = image_shape[:2]
                
                # Convert provided units to nm for consistent calculations
                scale_to_nm = 1.0
                if units.lower() in ['um', 'micrometer', 'micrometers']:
                    scale_to_nm = 1000.0
                elif units.lower() in ['a', 'angstrom', 'angstroms']:
                    scale_to_nm = 0.1
                
                fov_in_nm = fov_x * scale_to_nm
                
                if w > 0:
                    nm_per_pixel = fov_in_nm / w
                    self.logger.info(f"Calculated nm/pixel from FOV: {fov_x} {units} / {w} px = {nm_per_pixel:.4f} nm/pixel")
                else:
                    self.logger.warning("Cannot calculate scale from FOV because image width is 0.")
            else:
                self.logger.warning(f"Invalid or missing 'field_of_view_x' in spatial_info: {fov_x}. Physical scale not applied.")
        
        return nm_per_pixel, fov_in_nm

    def _validate_scientific_claims(self, scientific_claims: list) -> list:
        """
        Validate scientific claims structure and content.
        Shared validation logic used by all agents that generate claims.
        """
        valid_claims = []

        if not isinstance(scientific_claims, list):
            self.logger.warning(f"'scientific_claims' from LLM was not a list: {scientific_claims}")
            return valid_claims

        for claim in scientific_claims:
            if isinstance(claim, dict) and all(k in claim for k in ["claim", "scientific_impact", "has_anyone_question", "keywords"]):
                if isinstance(claim.get("keywords"), list):
                    valid_claims.append(claim)
                else:
                    self.logger.warning(f"Claim skipped due to 'keywords' not being a list: {claim}")
            else:
                self.logger.warning(f"Claim skipped due to missing keys or incorrect dict format: {claim}")
        
        return valid_claims

    def _validate_structure_recommendations(self, recommendations: list) -> list:
        """
        Validate and sort structure recommendations.
        Shared validation logic used by all agents that generate recommendations.
        """
        valid_recommendations = []
        
        if not isinstance(recommendations, list):
            self.logger.warning(f"'structure_recommendations' from LLM was not a list: {recommendations}")
            return valid_recommendations

        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in ["description", "scientific_interest", "priority"]):
                if isinstance(rec.get("priority"), int):
                    valid_recommendations.append(rec)
                else:
                    self.logger.warning(f"Recommendation skipped due to invalid priority type (expected int): {rec.get('priority')}. Recommendation: {rec}")
            else:
                self.logger.warning(f"Recommendation skipped due to missing keys or incorrect dict format: {rec}")
        
        # Sort by priority (1 = highest priority)
        sorted_recommendations = sorted(valid_recommendations, key=lambda x: x.get("priority", 99))
        return sorted_recommendations

    def _build_system_info_prompt_section(self, system_info: dict) -> str:
        """
        Build the system information section for LLM prompts.
        """
        if not system_info:
            return ""
            
        system_info_text = "\n\nAdditional System Information (Metadata):\n"
        if isinstance(system_info, dict):
            system_info_text += json.dumps(system_info, indent=2)
        else:
            system_info_text += str(system_info)
        
        return system_info_text