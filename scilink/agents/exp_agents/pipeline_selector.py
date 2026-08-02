import logging
import json
from typing import Dict

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params


class PipelineSelector:
    def __init__(self, 
                 api_key: str | None = None, 
                 model_name: str = "gemini-3-flash-preview", 
                 base_url: str = None,
                 # Deprecated
                 google_api_key: str | None = None, 
                 local_model: str = None):
        
        self.logger = logging.getLogger(__name__)
        
        self.api_key, self.base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="PipelineSelector"
        )
        
        if self.base_url:
            if 'gguf' in self.base_url:
                 self.logger.info(f"💻 Using local agent as pipeline selector.")
                 from ...wrappers.llama_wrapper import LocalLlamaModel
                 self.model = LocalLlamaModel(self.base_url)
            else:
                 self.logger.info(f"🏛️ Using OpenAI-compatible agent as pipeline selector.")
                 if self.api_key is None:
                     self.api_key = get_internal_proxy_key()
                 
                 if not self.api_key:
                     raise ValueError("API key required for internal proxy.")
                     
                 self.model = OpenAIAsGenerativeModel(model=model_name, api_key=self.api_key, base_url=self.base_url)
        else:
             self.logger.info(f"☁️ Using LiteLLM as pipeline selector.")
             self.model = LiteLLMGenerativeModel(model=model_name, api_key=self.api_key)
        
        self.generation_config = None
        self.safety_settings = None
    
    def select_pipeline(self, 
                        available_pipelines: Dict[str, Dict], 
                        image_blob: Dict | None = None, 
                        system_info: Dict | None = None) -> tuple[str, str]:
        """
        Select the most appropriate pipeline for the given input.
        
        Args:
            available_pipelines: Dict mapping pipeline_id -> {description, ...}
            image_blob: Optional image data for visual analysis
            system_info: Optional system metadata
            
        Returns:
            tuple: (selected_pipeline_id, reasoning_string)
            Returns (None, error_message) on failure
        """
        self.logger.info("Pipeline selector: Analyzing input to choose best pipeline...")
    
        selection_instructions = available_pipelines.get('_meta', {}).get('selection_instructions')
        
        if not selection_instructions:
            return None, "No selection instructions found in pipeline registry"
        
        # Build pipeline descriptions (skip metadata entries)
        pipeline_desc_text = ""
        for pid, info in available_pipelines.items():
            if pid.startswith('_'):  # Skip metadata keys like '_meta'
                continue
            pipeline_desc_text += f"- **ID '{pid}'**: {info.get('description', 'No description')}\n"
        
        # Build prompt with domain-specific instructions
        prompt_parts = [selection_instructions.replace('**Available Pipelines:**\n(These will be inserted automatically)', 
                                                       f'**Available Pipelines:**\n{pipeline_desc_text}')]

        
        # Add image if available
        if image_blob:
            prompt_parts.append("\n--- Image for Context ---")
            prompt_parts.append("This is the input image to be analyzed:")
            prompt_parts.append(image_blob)
        
        # Add system info
        if system_info:
            prompt_parts.append("\n--- System Information ---")
            prompt_parts.append(json.dumps(system_info, indent=2))
        
        prompt_parts.append("\nBased on the provided information, select the most appropriate pipeline.")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings, # Now this attribute exists
            )
            
            result_json = self._parse_llm_response(response)
            
            if not result_json:
                return None, "Failed to parse LLM response"
            
            pipeline_id = result_json.get('pipeline_id')
            reasoning = result_json.get('reasoning', 'No reasoning provided.')
            
            self.logger.info(f"\n\n🧠 Pipeline Selector Reasoning: {reasoning}\n")
            
            valid_pipeline_ids = [pid for pid in available_pipelines.keys() if not pid.startswith('_')]

            if pipeline_id not in valid_pipeline_ids:
                error_msg = f"LLM selected invalid pipeline: '{pipeline_id}'. Available: {valid_pipeline_ids}"
                self.logger.warning(error_msg)
                return None, error_msg
            
            return pipeline_id, reasoning
            
        except Exception as e:
            error_msg = f"Pipeline selection failed: {e}"
            self.logger.exception(error_msg)
            return None, error_msg
    
    def _parse_llm_response(self, response) -> dict | None:
        """Parse JSON response from LLM."""
        try:
            # Handle LiteLLM/OpenAI wrapper response object
            if hasattr(response, 'text'):
                raw_text = response.text
            elif hasattr(response, 'choices'): 
                 raw_text = response.choices[0].message.content
            else:
                 raw_text = str(response)

            first_brace = raw_text.find('{')
            last_brace = raw_text.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_string = raw_text[first_brace:last_brace + 1]
                return json.loads(json_string)
            else:
                raise ValueError("Could not find valid JSON in response.")
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM JSON response: {e}")
            return None