import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .instruct import TEXT_ONLY_DFT_RECOMMENDATION_INSTRUCTIONS
from ...auth import get_api_key, APIKeyNotFoundError


class RecommendationAgent:
    """
    An agent specialized in synthesizing textual analysis from various sources 
    (e.g., experimental data analysis, literature novelty reviews) to generate 
    targeted Density Functional Theory structure recommendations.
    
    This agent does not analyze raw experimental data directly. Instead, it operates
    on the textual output of other agents to provide the final recommendations.
    """

    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05"):
        """
        Initializes the RecommendationAgent.

        Args:
            google_api_key (str | None, optional): The Google API key. If not provided,
                it will be discovered from the environment. Defaults to None.
            model_name (str, optional): The name of the generative AI model to use. 
                Defaults to "gemini-2.5-pro-preview-06-05".
        """
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        genai.configure(api_key=google_api_key)

        self.model_name = model_name 

        self.model = genai.GenerativeModel(self.model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RecommendationAgent initialized with model: {self.model_name}")

    def _parse_llm_response(self, response) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Parses the LLM response, expecting a valid JSON object.

        Args:
            response: The response object from the generative AI model.

        Returns:
            A tuple containing the parsed JSON dictionary (or None) and an error dictionary (or None).
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
            
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                error_dict = {"error": f"Content blocked by safety filters", "details": f"Reason: {block_reason}"}
            else:
                error_dict = {"error": "Failed to parse valid JSON from LLM response", "details": error_details, "raw_response": error_raw_response}
        
        except Exception as e:
            self.logger.exception(f"Unexpected error processing response: {e}")
            error_dict = {"error": "Unexpected error processing LLM response", "details": str(e)}
            
        return result_json, error_dict

    def _generate_json_from_text_parts(self, prompt_parts: list) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Internal helper to make a text-only call to the LLM and parse the JSON response.

        Args:
            prompt_parts (list): A list of strings and other prompt components.

        Returns:
            A tuple containing the parsed JSON dictionary (or None) and an error dictionary (or None).
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

    def generate_dft_recommendations_from_text(
        self,
        cached_detailed_analysis: str,
        additional_prompt_context: str,
        system_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates DFT structure recommendations by synthesizing a prior analysis
        with novelty context from a literature review.

        Args:
            cached_detailed_analysis (str): The detailed textual analysis from a prior step (e.g., from an image agent).
            additional_prompt_context (str): Textual context outlining novel findings or other special considerations.
            system_info (dict | str | None, optional): System metadata (dict or JSON string). Defaults to None.

        Returns:
            A dictionary containing the reasoning and a sorted list of recommendations.
        """
        self.logger.info("Generating DFT recommendations from cached analysis and novelty context.")
        
        # Build the prompt for the LLM
        prompt_list = [TEXT_ONLY_DFT_RECOMMENDATION_INSTRUCTIONS]
        prompt_list.append("\n\n--- Start of Cached Initial Experimental Data Analysis ---\n")
        prompt_list.append(cached_detailed_analysis)
        prompt_list.append("\n--- End of Cached Initial Experimental Data Analysis ---\n")

        prompt_list.append("\n\n--- Start of Special Considerations (e.g., Novelty Insights) ---\n")
        prompt_list.append(additional_prompt_context)
        prompt_list.append("\n--- End of Special Considerations ---\n")

        if system_info:
            system_info_text = "\n\nAdditional System Information (Metadata):\n"
            if isinstance(system_info, dict):
                system_info_text += json.dumps(system_info, indent=2)
            else:
                # Handle case where system_info might be a pre-formatted string
                system_info_text += str(system_info)
            prompt_list.append(system_info_text)

        prompt_list.append("\n\nProvide your DFT structure recommendations strictly in the requested JSON format.")
        
        # Get and parse the LLM response
        result_json, error_dict = self._generate_json_from_text_parts(prompt_list)

        if error_dict:
            return error_dict
        if result_json is None:
            return {"error": "Recommendation generation failed unexpectedly after LLM processing."}

        # Extract and validate the results from the JSON
        reasoning = result_json.get("detailed_reasoning_for_recommendations", "Reasoning not provided by LLM.")
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
                    self.logger.warning(f"Recommendation skipped due to invalid priority type (expected int): {rec}")
            else:
                self.logger.warning(f"Recommendation skipped due to missing keys or incorrect dict format: {rec}")

        sorted_recommendations = sorted(valid_recommendations, key=lambda x: x.get("priority", 99))

        if not sorted_recommendations and reasoning != "Reasoning not provided by LLM.":
            self.logger.warning("LLM call successful ('detailed_reasoning' provided) but no valid recommendations found.")
        
        # Return a consistently formatted dictionary
        return {
            "analysis_summary_or_reasoning": reasoning,
            "recommendations": sorted_recommendations
        }