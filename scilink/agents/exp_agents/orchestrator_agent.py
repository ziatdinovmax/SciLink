import logging
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .microscopy_agent import GeminiMicroscopyAnalysisAgent
from .sam_microscopy_agent import GeminiSAMMicroscopyAnalysisAgent
from .atomistic_microscopy_agent import GeminiAtomisticMicroscopyAnalysisAgent
from .spectroscopy_agent import GeminiSpectroscopyAnalysisAgent
from .instruct import ORCHESTRATOR_INSTRUCTIONS
from ...auth import get_api_key, APIKeyNotFoundError

# Mapping from integer ID to the corresponding agent class
AGENT_MAP = {
    0: GeminiMicroscopyAnalysisAgent,
    1: GeminiSAMMicroscopyAnalysisAgent,
    2: GeminiAtomisticMicroscopyAnalysisAgent,
    3: GeminiSpectroscopyAnalysisAgent,
}

class OrchestratorAgent:
    """
    An LLM-powered agent that selects the most appropriate experimental analysis agent
    based on the user's scientific intent.
    """
    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-flash-preview-05-20"):
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

    def _parse_llm_response(self, response) -> tuple[dict | None, dict | None]:
        """Parses the LLM response, expecting JSON."""
        try:
            raw_text = response.text
            first_brace_index = raw_text.find('{')
            last_brace_index = raw_text.rfind('}')
            if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                json_string = raw_text[first_brace_index : last_brace_index + 1]
                return json.loads(json_string), None
            else:
                raise ValueError("Could not find valid JSON object in response.")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing Gemini JSON response: {e}")
            error_dict = {"error": "Failed to parse JSON from LLM response", "details": str(e), "raw_response": response.text}
            return None, error_dict

    def select_agent(self, data_type: str, system_info: dict | str | None = None) -> int:
        """
        Selects the appropriate experimental agent by asking the Gemini LLM.

        Args:
            data_type: The primary type of data, e.g., 'microscopy' or 'spectroscopy'.
            system_info: Additional information about the system or analysis goal.

        Returns:
            The integer key for the selected agent in AGENT_MAP, or -1 on failure.
        """
        self.logger.info(f"Orchestrator LLM selecting agent for data_type: '{data_type}'")

        # Require system_info to be provided for an informed decision.
        if not system_info or not str(system_info).strip():
            self.logger.error("system_info must be provided for the orchestrator to select an agent.")
            return -1

        # If the data type is spectroscopy, the choice is clear.
        if data_type.lower() == 'spectroscopy':
            self.logger.info("Data type is 'spectroscopy'. Selecting Spectroscopy agent directly.")
            return 3

        prompt_parts = [
            ORCHESTRATOR_INSTRUCTIONS,
            "\n--- User Request ---",
            f"data_type: {data_type}",
            f"system_info: {str(system_info)}"
        ]

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)

            if error_dict:
                self.logger.error(f"Orchestrator LLM call failed: {error_dict}")
                return -1

            agent_id = result_json.get('agent_id')
            reasoning = result_json.get('reasoning', 'No reasoning provided.')
            self.logger.info(f"LLM selected agent ID: {agent_id}. Reasoning: {reasoning}")

            if isinstance(agent_id, int) and agent_id in AGENT_MAP:
                return agent_id
            else:
                self.logger.warning(f"LLM returned an invalid agent ID: {agent_id}. Returning -1.")
                return -1

        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during agent selection: {e}")
            return -1