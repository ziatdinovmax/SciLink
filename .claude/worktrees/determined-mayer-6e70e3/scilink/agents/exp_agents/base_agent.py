# base_agent.py

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params

HIGH_NOVELTY_THRESHOLD = 4   # Score 4-5: Highly novel, needs validation
MEDIUM_NOVELTY_THRESHOLD = 3  # Score 3: Somewhat novel, needs differentiation


# =============================================================================
# MIXIN: Shared LLM functionality
# =============================================================================

class LLMAgentMixin:
    """
    Mixin providing common LLM functionality for all agents.
    
    This includes model initialization, response parsing, and utility methods
    shared between full analysis agents and utility/preprocessing agents.
    """
    
    logger: logging.Logger
    api_key: str | None
    base_url: str | None
    model_name: str
    model: Any
    generation_config: Any
    safety_settings: Any
    output_dir: Path
    
    def _initialize_model(self) -> None:
        """Initialize the LLM model based on configuration."""
        if self.base_url:
            # A. GGUF / Local File Mode
            if 'gguf' in self.base_url:
                logging.info(f"💻 Using local agent (GGUF path detected): {self.base_url}")
                from ...wrappers.llama_wrapper import LocalLlamaModel
                self.model = LocalLlamaModel(self.base_url)

            # B. Internal Proxy / Network Mode
            else:
                logging.info(f"🏛️ Using OpenAI-compatible agent: {self.base_url}")
                
                if self.api_key is None:
                    self.api_key = get_internal_proxy_key()
                
                if not self.api_key:
                    raise ValueError(
                        "API key required for internal proxy.\n"
                        "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                    )
                
                self.model = OpenAIAsGenerativeModel(
                    model=self.model_name,
                    api_key=self.api_key,
                    base_url=self.base_url
                )
        else:
            # C. Public / LiteLLM Mode
            logging.info(f"☁️ Using LiteLLM agent: {self.model_name}")
            self.model = LiteLLMGenerativeModel(
                model=self.model_name,
                api_key=self.api_key
            )

    # =========================================================================
    # LLM RESPONSE PARSING
    # =========================================================================
    
    def _parse_llm_response(self, response: Any) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Parse LLM response to extract JSON, with multiple fallback strategies.
        
        Returns:
            Tuple of (result_json, error_dict) - one will be None
        """
        try:
            raw_text = self._extract_text_from_response(response)
            
            if not raw_text or not raw_text.strip():
                return None, {
                    "error": "Empty response from LLM",
                    "details": "Response text was empty"
                }
            
            # Strategy 1: Direct JSON parse
            result = self._try_direct_json_parse(raw_text)
            if result is not None:
                return result, None
            
            # Strategy 2: Find JSON in markdown code blocks
            result = self._try_extract_json_from_code_blocks(raw_text)
            if result is not None:
                return result, None
            
            # Strategy 3: Find any JSON object in text
            result = self._try_extract_json_from_text(raw_text)
            if result is not None:
                self.logger.warning("JSON extracted via regex fallback - may be incomplete")
                return result, None
            
            # Strategy 4: For script responses, try to extract Python code
            result = self._try_extract_python_script(raw_text)
            if result is not None:
                self.logger.warning("JSON parsing failed, but found Python code block")
                return result, None
            
            # All strategies failed
            snippet = raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
            self.logger.error(f"All JSON parsing strategies failed. Response snippet: {snippet}")
            
            return None, {
                "error": "Failed to parse valid JSON from LLM response",
                "details": "No valid JSON found using any extraction strategy",
                "raw_response": raw_text[:2000]
            }
            
        except Exception as e:
            self.logger.exception(f"Unexpected error parsing LLM response: {e}")
            return None, {
                "error": "Exception during response parsing",
                "details": str(e)
            }
    
    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text content from various LLM response formats."""
        if hasattr(response, 'text'):
            return response.text
        if hasattr(response, 'content'):
            return response.content
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content
            if hasattr(choice, 'text'):
                return choice.text
        return str(response)
    
    def _try_direct_json_parse(self, raw_text: str) -> Optional[dict]:
        try:
            text_clean = raw_text.strip()
            
            if text_clean.startswith('```json'):
                text_clean = text_clean[7:]
            elif text_clean.startswith('```'):
                text_clean = text_clean[3:]
            if text_clean.endswith('```'):
                text_clean = text_clean[:-3]
            
            text_clean = text_clean.strip()
            
            # First attempt: direct parse
            try:
                result = json.loads(text_clean)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
            
            # Second attempt: fix common issues (unescaped newlines in strings)
            # Replace newlines that appear between quotes
            import re
            
            def escape_newlines_in_strings(match):
                content = match.group(1)
                # Escape newlines, carriage returns, tabs
                content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                return '"' + content + '"'
            
            # Match strings: "..." (non-greedy, handles most cases)
            # This is a simplified approach - handles most LLM outputs
            fixed_text = re.sub(r'"((?:[^"\\]|\\.)*?)"', escape_newlines_in_strings, text_clean, flags=re.DOTALL)
            
            result = json.loads(fixed_text)
            if isinstance(result, dict):
                return result
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _try_extract_json_from_code_blocks(self, raw_text: str) -> Optional[dict]:
        """Extract JSON from markdown code blocks."""
        patterns = [
            r'```json\s*\n?([\s\S]*?)\n?```',
            r'```\s*\n?(\{[\s\S]*?\})\n?```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, raw_text, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match.strip())
                    if isinstance(result, dict) and len(result) > 0:
                        return result
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _try_extract_json_from_text(self, raw_text: str) -> Optional[dict]:
        """Extract JSON objects from anywhere in the text."""
        start_idx = None
        brace_count = 0
        
        for i, char in enumerate(raw_text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    potential_json = raw_text[start_idx:i+1]
                    try:
                        result = json.loads(potential_json)
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except json.JSONDecodeError:
                        pass
                    start_idx = None
        
        return None
    
    def _try_extract_python_script(self, raw_text: str) -> Optional[dict]:
        """Extract Python code blocks and wrap in synthetic JSON structure."""
        python_block_pattern = r'```python\s*([\s\S]*?)\s*```'
        matches = re.findall(python_block_pattern, raw_text, re.DOTALL)
        
        if matches:
            script = matches[0].strip()
            if script:
                return {
                    "script": script,
                    "analysis_approach": "extracted_from_code_block",
                    "key_metrics_to_track": [],
                    "_extraction_method": "python_code_block_fallback"
                }
        
        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _handle_system_info(self, system_info: Dict[str, Any] | str | None) -> Dict[str, Any]:
        """
        Handle system_info input (can be dict, file path, or None).
        
        Supports multiple input formats for backward compatibility:
        - None -> empty dict
        - dict with top-level keys (e.g., {"sample": "...", "technique": "..."}) -> returned as-is
        - dict with nested "system_info" key -> extracts the nested dict
        - str (file path to JSON) -> loads and processes as above
        
        This allows users to pass either:
        - A flat system_info dict directly
        - A full metadata file that contains a "system_info" key
        - A path to either type of JSON file
        """
        if system_info is None:
            return {}
        
        # Load from file if string path provided
        if isinstance(system_info, str):
            try:
                with open(system_info, 'r') as f:
                    system_info = json.load(f)
            except FileNotFoundError:
                self.logger.error(f"system_info file not found: {system_info}")
                return {}
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in system_info file {system_info}: {e}")
                return {}
            except Exception as e:
                self.logger.error(f"Error loading system_info from {system_info}: {e}")
                return {}
        
        # Ensure we have a dict at this point
        if not isinstance(system_info, dict):
            self.logger.warning(f"system_info is not a dict: {type(system_info)}. Returning empty dict.")
            return {}
        
        # Handle nested "system_info" key (common in full metadata files)
        # Only extract if the nested value is a dict and contains typical system_info keys
        if "system_info" in system_info:
            nested = system_info["system_info"]
            if isinstance(nested, dict) and len(nested) > 0:
                self.logger.debug("Extracted nested 'system_info' from metadata structure")
                return nested
        
        # Return as-is for flat structures or if no valid nested system_info found
        return system_info

    @staticmethod
    def _extract_series_metadata(
        system_info: Dict[str, Any],
        series_metadata: dict | None,
    ) -> tuple[Dict[str, Any], dict | None]:
        """Pop ``"series"`` from *system_info* when no explicit *series_metadata* is given.

        Returns the (possibly modified) *system_info* and the resolved
        *series_metadata*.  If *series_metadata* was already provided it
        takes precedence and *system_info* is returned unchanged.

        The expected series_metadata structure is::

            {
                "variable": "temperature",  # independent variable name
                "values": [300, 350, 400],  # one value per data point
                "unit": "K"                 # unit for values
            }
        """
        if not series_metadata and isinstance(system_info, dict) and "series" in system_info:
            system_info = dict(system_info)          # shallow copy to avoid mutating caller's dict
            series_metadata = system_info.pop("series")
        return system_info, series_metadata

    def _build_system_info_prompt_section(self, system_info: Dict[str, Any]) -> str:
        """Build the system information section for LLM prompts."""
        if not system_info:
            return ""
        
        return f"\n\n## System Information\n{json.dumps(system_info, indent=2)}"

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


# =============================================================================
# TYPE ALIAS for analyze() data parameter
# =============================================================================

# Data can be: single path, list of paths, or numpy array
AnalysisInput = Union[str, List[str], np.ndarray]


# =============================================================================
# BASE UTILITY AGENT (for preprocessing/helper agents)
# =============================================================================

class BaseUtilityAgent(LLMAgentMixin):
    """
    Base class for utility/preprocessing agents.
    
    These agents provide helper functionality (preprocessing, parameter estimation, etc.)
    but don't implement the full analyze() -> recommend_measurements() workflow.
    
    Use this base class for:
        - Preprocessing agents (HyperspectralPreprocessingAgent, CurvePreprocessingAgent)
        - Parameter estimation agents
        - Other helper/utility agents
    
    This class provides:
        - LLM model initialization
        - Response parsing utilities
        - Basic state management
        - Common utility methods
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-opus-4-6",
        base_url: str | None = None,
        output_dir: str = ".",
        # Deprecated arguments
        google_api_key: str | None = None,
        local_model: str | None = None,
        **kwargs
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Output directory setup
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent identification
        self.agent_type = "utility"  # Subclasses should override
        
        # Normalize parameters
        self.api_key, self.base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source=self.__class__.__name__
        )
        
        self.model_name = model_name
        self.generation_config = None
        self.safety_settings = None
        
        # Initialize LLM model
        self._initialize_model()
        
        # Basic state for utility agents
        self.state: Dict[str, Any] = {}

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Override in subclasses to add agent-specific state fields."""
        return {}
    
    def _init_state(self, **context) -> None:
        """Initialize state for a new session."""
        if self.state.get("session_id") is None:
            self.state = {
                "session_id": str(uuid.uuid4()),
                "start_time": datetime.now().isoformat(),
                "agent_type": self.agent_type,
                "status": "initialized"
            }
            self.state.update(self._get_initial_state_fields())
        
        for key, value in context.items():
            self.state[key] = value
        
        self.state["status"] = "active"


# =============================================================================
# BASE ANALYSIS AGENT (for full analysis agents)
# =============================================================================

class BaseAnalysisAgent(LLMAgentMixin, ABC):
    """
    Base class for full analysis agents.
    
    Analysis agents follow a consistent pattern:
    
    1. Primary analysis via `analyze()` method
       - Accepts flexible input: str (path), List[str] (batch), or np.ndarray
       - Returns detailed_analysis, scientific_claims, and agent-specific fields
       - Standardized return format with "status" field
    
    2. Optional measurement recommendations via `recommend_measurements()`
       - Can be called after analyze() with the results
       - Or called directly (will run analyze() internally)
    
    Subclasses must implement:
        - analyze(): Primary analysis logic
        - _get_claims_instruction_prompt(): Instruction prompt for analysis
        - _get_measurement_recommendations_prompt(): Instruction prompt for recommendations
    
    Use BaseUtilityAgent instead for preprocessing/helper agents that don't need
    the full analyze() workflow.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-opus-4-6",
        base_url: str | None = None,
        output_dir: str = ".",
        enable_human_feedback: bool = False,
        # Deprecated arguments
        google_api_key: str | None = None,
        local_model: str | None = None,
        **kwargs
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        # State management
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_type = "base_analysis"  # Subclasses should override
        self.state: Dict[str, Any] = {}
        self.enable_human_feedback = enable_human_feedback
        
        # Normalize parameters
        self.api_key, self.base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source=self.__class__.__name__
        )
        
        self.model_name = model_name
        self.generation_config = None
        self.safety_settings = None
        
        # Initialize LLM model
        self._initialize_model()

        # Storage for analysis artifacts
        self._stored_analysis_images: List[Dict[str, Any]] = []
        self._stored_analysis_metadata: Dict[str, Any] = {}

    # =========================================================================
    # PRIMARY ENTRY POINTS
    # =========================================================================
    
    @abstractmethod
    def analyze(
        self,
        data: AnalysisInput,
        system_info: Dict[str, Any] | str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Primary analysis entry point.
        
        All agents must implement this method with a consistent return format.
        
        Args:
            data: Input data. Can be:
                - str: Path to single data file
                - List[str]: List of paths for batch processing
                - np.ndarray: Direct array input
            system_info: Metadata dictionary or path to metadata file
            **kwargs: Agent-specific options
        
        Returns:
            dict containing at minimum:
                - "status": "success" | "error" | "cancelled"
                - "detailed_analysis": str (when successful)
                - "scientific_claims": list[dict] (when successful)
                - "output_directory": str
                - "error": dict (when status="error")
        
        Examples:
            # Single file
            result = agent.analyze("spectrum.csv")
            
            # Multiple files (batch)
            result = agent.analyze(["img1.tif", "img2.tif"])
            
            # Numpy array
            result = agent.analyze(my_array)
            
            # With metadata
            result = agent.analyze("data.npy", system_info={"sample": "TiO2"})
        """
        raise NotImplementedError
    
    def _parse_data_input(
        self,
        data: AnalysisInput
    ) -> Tuple[Optional[str], Optional[List[str]], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Parse the flexible data input into specific types.
        
        Args:
            data: Input data (str, List[str], or np.ndarray)
        
        Returns:
            Tuple of (data_path, data_paths, data_array, error_dict)
            - Only one of the first three will be non-None
            - error_dict is non-None if input is invalid
        """
        if data is None:
            return None, None, None, {
                "error": "No input provided",
                "details": "data parameter is required"
            }
        
        if isinstance(data, str):
            # Single file path
            return data, None, None, None
        
        elif isinstance(data, list):
            # List of file paths
            if not data:
                return None, None, None, {
                    "error": "Empty input",
                    "details": "data list is empty"
                }
            if not all(isinstance(p, str) for p in data):
                return None, None, None, {
                    "error": "Invalid input",
                    "details": "All items in data list must be strings (file paths)"
                }
            return None, data, None, None
        
        elif isinstance(data, np.ndarray):
            # Numpy array
            return None, None, data, None
        
        else:
            return None, None, None, {
                "error": "Invalid input type",
                "details": f"Expected str, List[str], or np.ndarray, got {type(data).__name__}"
            }
    
    def recommend_measurements(
        self,
        analysis_result: Dict[str, Any] | None = None,
        data: AnalysisInput | None = None,
        system_info: Dict[str, Any] | str | None = None,
        novelty_context: str | None = None,
        novelty_assessment: Dict[str, Any] | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate measurement recommendations based on analysis.
        
        This is an optional post-processing step that builds on analyze() results.
        If no analysis_result is provided, runs analyze() first.
        
        Args:
            analysis_result: Output from analyze(). If None, runs analyze() first.
            data: Input data (required if analysis_result is None)
            system_info: Metadata (passed to analyze() if running it)
            novelty_context: Optional context from literature review
            **kwargs: Additional arguments passed to analyze() if needed
        
        Returns:
            dict containing:
                - "status": "success" | "error"
                - "analysis_integration": str (how analysis informed recommendations)
                - "measurement_recommendations": list[dict]
        """
        # If no analysis provided, run it first
        if analysis_result is None:
            if data is None:
                return {
                    "status": "error",
                    "error": {
                        "error": "No input provided",
                        "details": "Must provide either analysis_result or data"
                    }
                }
            
            self.logger.info("No analysis_result provided, running analyze() first...")
            analysis_result = self.analyze(
                data=data,
                system_info=system_info,
                **kwargs
            )
            
            if analysis_result.get("status") == "error":
                return {
                    "status": "error",
                    "error": analysis_result.get("error", {"error": "Analysis failed"})
                }
        
        # Generate recommendations from analysis
        return self._generate_recommendations(analysis_result, system_info, novelty_context, novelty_assessment)

    # =========================================================================
    # BACKWARD COMPATIBLE ALIASES
    # =========================================================================
    
    def analyze_for_claims(
        self,
        data_path: str,
        system_info: Dict[str, Any] | str | None = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze data to generate scientific claims.
        
        BACKWARD COMPATIBLE: Delegates to analyze().
        """
        result = self.analyze(data=data_path, system_info=system_info, **kwargs)
        
        if result.get("status") == "success":
            return {
                "detailed_analysis": result.get("detailed_analysis", ""),
                "scientific_claims": result.get("scientific_claims", [])
            }
        else:
            return result.get("error", result)
    
    def generate_measurement_recommendations(
        self,
        analysis_result: Dict[str, Any],
        system_info: Dict[str, Any] | None = None,
        novelty_context: str | None = None
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use recommend_measurements() instead.
        """
        self.logger.warning(
            "generate_measurement_recommendations() is deprecated. "
            "Use recommend_measurements(analysis_result=...) instead."
        )
        result = self._generate_recommendations(analysis_result, system_info, novelty_context)
        
        if result.get("status") == "success":
            return {
                "analysis_integration": result.get("analysis_integration", ""),
                "measurement_recommendations": result.get("measurement_recommendations", []),
                "total_recommendations": len(result.get("measurement_recommendations", []))
            }
        else:
            return {"error": result.get("error", "Recommendation generation failed")}

    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================
    
    def _get_claims_instruction_prompt(self) -> str:
        """Return the instruction prompt for claims generation.

        Override in subclasses to provide domain-specific guidance for the
        LLM when generating scientific claims and refining analysis with
        human feedback. Returning "" is valid — the LLM will still produce
        claims based on the data alone.
        """
        return ""

    def _get_measurement_recommendations_prompt(self) -> str:
        """Return the instruction prompt for measurement recommendations.

        Override in subclasses to provide domain-specific guidance for the
        LLM when generating follow-up measurement suggestions. Returning ""
        is valid — the LLM will generate generic recommendations from the
        analysis results.
        """
        return ""

    # =========================================================================
    # INTERNAL RECOMMENDATION GENERATION
    # =========================================================================
    
    def _generate_recommendations(
        self,
        analysis_result: Dict[str, Any],
        system_info: Dict[str, Any] | str | None = None,
        novelty_context: str | None = None,
        novelty_assessment: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Internal method to generate recommendations from analysis results."""
        system_info = self._handle_system_info(system_info)
        
        if analysis_result.get("status") == "error" or "error" in analysis_result:
            return {
                "status": "error",
                "error": {
                    "error": "Cannot generate recommendations from failed analysis",
                    "details": analysis_result.get("error", {})
                }
            }
        
        try:
            instruction_prompt = self._get_measurement_recommendations_prompt()
            
            prompt_parts = [instruction_prompt]
            prompt_parts.append("\n\n## Analysis Results")
            
            detailed_analysis = analysis_result.get("detailed_analysis", "")
            if detailed_analysis:
                prompt_parts.append(f"\n**Detailed Analysis:**\n{detailed_analysis}")
            
            scientific_claims = analysis_result.get("scientific_claims", [])
            if scientific_claims:
                prompt_parts.append(f"\n\n**Scientific Claims:**\n{json.dumps(scientific_claims, indent=2)}")
            
            stored_images = self._get_stored_analysis_images()
            if stored_images:
                prompt_parts.append("\n\n## Analysis Images")
                for img_data in stored_images[:5]:
                    if isinstance(img_data, dict) and 'label' in img_data and 'data' in img_data:
                        prompt_parts.append(f"\n**{img_data['label']}:**")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": img_data['data']})
            
            if novelty_context:
                prompt_parts.append(f"\n\n## Novelty Context\n{novelty_context}")

            if novelty_assessment:
                novelty_section = self._build_novelty_prompt_section(novelty_assessment)
                prompt_parts.append(novelty_section)
            
            if system_info:
                prompt_parts.append(f"\n\n## System Information\n{json.dumps(system_info, indent=2)}")
            
            prompt_parts.append("\n\nProvide measurement recommendations in JSON format.")
            
            self.logger.info("📋 Generating measurement recommendations...")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                return {"status": "error", "error": error_dict}
            
            recommendations = result_json.get("measurement_recommendations", [])
            valid_recommendations = self._validate_measurement_recommendations(recommendations)
            
            self.logger.info(f"✅ Generated {len(valid_recommendations)} measurement recommendations")
            
            return self._enhance_with_novelty(
                base_result={
                    "status": "success",
                    "analysis_integration": result_json.get("analysis_integration", ""),
                    "measurement_recommendations": valid_recommendations
                },
                novelty_assessment=novelty_assessment,
                analysis_result=analysis_result
            )
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return {
                "status": "error",
                "error": {"error": "Recommendation generation failed", "details": str(e)}
            }
        

    def _build_novelty_prompt_section(self, novelty_assessment: Dict[str, Any]) -> str:
        """Build prompt section from novelty assessment for LLM context."""
        assessments = novelty_assessment.get("assessments", [])
        high_novelty = novelty_assessment.get("high_novelty_claims", [])
        summary_stats = novelty_assessment.get("summary_stats", {})
        
        section = "\n\n## Novelty Assessment Results\n"
        
        section += f"**Summary:** {len(assessments)} claims assessed, "
        section += f"average novelty score: {summary_stats.get('average_score', 0):.1f}/5\n\n"
        
        if high_novelty:
            section += "**HIGH-NOVELTY FINDINGS (Score 4-5) - Prioritize validation:**\n"
            for claim in high_novelty[:5]:
                section += f"- [Score {claim.get('novelty_score', '?')}] {claim.get('original_claim', '')[:150]}...\n"
                section += f"  Literature gap: {claim.get('novelty_explanation', '')[:100]}...\n"
            section += "\n"
        
        medium_novelty = [a for a in assessments if a.get("novelty_score") == MEDIUM_NOVELTY_THRESHOLD]
        if medium_novelty:
            section += "**MEDIUM-NOVELTY FINDINGS (Score 3) - Consider differentiation:**\n"
            for claim in medium_novelty[:3]:
                section += f"- {claim.get('original_claim', '')[:100]}...\n"
            section += "\n"
        
        section += """
    **Recommendation Guidance Based on Novelty:**
    - For HIGH-NOVELTY claims: Suggest validation experiments (replication, complementary techniques)
    - For MEDIUM-NOVELTY claims: Suggest differentiation experiments (what makes this unique?)
    - For LOW-NOVELTY claims: Lower priority, suggest only if scientifically valuable
    """
        
        return section


    def _enhance_with_novelty(
        self,
        base_result: Dict[str, Any],
        novelty_assessment: Optional[Dict[str, Any]],
        analysis_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Enhance base recommendations with novelty-informed prioritization."""
        if novelty_assessment is None:
            base_result["novelty_informed"] = False
            base_result["novelty_recommendations"] = []
            return base_result
        
        self.logger.info("🔬 Enhancing recommendations with novelty assessment...")
        
        assessments = novelty_assessment.get("assessments", [])
        high_novelty_claims = novelty_assessment.get("high_novelty_claims", [])
        summary_stats = novelty_assessment.get("summary_stats", {})
        
        novelty_recommendations = []
        
        # High-novelty claims → Validation experiments
        for claim in high_novelty_claims:
            novelty_recommendations.append(
                self._generate_validation_recommendation(claim)
            )
        
        # Medium-novelty claims → Differentiation experiments
        medium_novelty_claims = [
            a for a in assessments 
            if a.get("novelty_score") == MEDIUM_NOVELTY_THRESHOLD
        ]
        for claim in medium_novelty_claims:
            novelty_recommendations.append(
                self._generate_differentiation_recommendation(claim)
            )
        
        # Prioritize base recommendations by novelty relevance
        base_recommendations = base_result.get("measurement_recommendations", [])
        prioritized_recommendations = self._prioritize_by_novelty(
            base_recommendations, assessments
        )
        
        # Build integration summary
        integration_summary = self._build_novelty_integration_summary(
            base_result.get("analysis_integration", ""),
            summary_stats,
            len(high_novelty_claims),
            len(medium_novelty_claims),
            len(assessments)
        )
        
        self.logger.info(
            f"✅ Added {len(novelty_recommendations)} novelty-specific recommendations"
        )
        
        return {
            "status": "success",
            "analysis_integration": integration_summary,
            "measurement_recommendations": prioritized_recommendations,
            "novelty_recommendations": novelty_recommendations,
            "novelty_informed": True,
            "novelty_summary": {
                "total_claims_assessed": len(assessments),
                "high_novelty_count": len(high_novelty_claims),
                "medium_novelty_count": len(medium_novelty_claims),
                "average_score": summary_stats.get("average_score", 0)
            }
        }


    def _generate_validation_recommendation(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation experiment recommendation for high-novelty claim."""
        return {
            "type": "validation",
            "priority": 1,
            "description": f"Validate high-novelty finding: {claim.get('original_claim', '')[:150]}",
            "scientific_justification": (
                f"Novelty score {claim.get('novelty_score', '?')}/5. "
                f"{claim.get('novelty_explanation', 'Novel finding requires independent validation.')}"
            ),
            "suggested_approaches": [
                "Replicate measurement with different sample preparation",
                "Use complementary technique to confirm finding",
                "Perform statistical analysis across multiple samples",
                "Compare with reference materials or standards"
            ],
            "novelty_context": {
                "original_claim": claim.get("original_claim"),
                "novelty_score": claim.get("novelty_score"),
                "literature_gap": claim.get("search_answer", "")[:300] if claim.get("search_answer") else None
            }
        }


    def _generate_differentiation_recommendation(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Generate differentiation experiment for medium-novelty claim."""
        return {
            "type": "differentiation", 
            "priority": 2,
            "description": f"Differentiate finding from prior work: {claim.get('original_claim', '')[:150]}",
            "scientific_justification": (
                f"Novelty score {claim.get('novelty_score', '?')}/5. "
                "Finding has partial overlap with existing literature. "
                "Differentiation experiments needed to establish uniqueness."
            ),
            "suggested_approaches": [
                "Identify specific conditions/parameters that differ from prior work",
                "Quantify improvements or differences systematically",
                "Explore edge cases or boundary conditions",
                "Document unique aspects of sample or methodology"
            ],
            "novelty_context": {
                "original_claim": claim.get("original_claim"),
                "novelty_score": claim.get("novelty_score"),
                "existing_work": claim.get("search_answer", "")[:200] if claim.get("search_answer") else None
            }
        }


    def _prioritize_by_novelty(
        self,
        recommendations: List[Dict[str, Any]],
        novelty_assessments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-prioritize recommendations based on novelty relevance."""
        if not novelty_assessments:
            return recommendations
        
        high_novelty_keywords = set()
        for assessment in novelty_assessments:
            if assessment.get("novelty_score", 0) >= HIGH_NOVELTY_THRESHOLD:
                claim = assessment.get("original_claim", "").lower()
                words = claim.split()
                high_novelty_keywords.update(
                    w for w in words if len(w) > 4 and w.isalpha()
                )
        
        if not high_novelty_keywords:
            return recommendations
        
        scored = []
        for rec in recommendations:
            rec_text = str(rec).lower()
            match_count = sum(1 for kw in high_novelty_keywords if kw in rec_text)
            
            enhanced_rec = rec.copy()
            if match_count > 0:
                enhanced_rec["novelty_relevance"] = f"Related to {match_count} high-novelty finding(s)"
                if "priority" in enhanced_rec:
                    enhanced_rec["priority"] = max(1, enhanced_rec["priority"] - 1)
            
            scored.append((match_count, enhanced_rec))
        
        scored.sort(key=lambda x: (-x[0], x[1].get("priority", 99)))
        
        return [rec for _, rec in scored]


    def _build_novelty_integration_summary(
        self,
        base_integration: str,
        summary_stats: Dict[str, Any],
        high_count: int,
        medium_count: int,
        total_count: int
    ) -> str:
        """Build summary of how novelty assessment informed recommendations."""
        avg_score = summary_stats.get("average_score", 0)
        
        if high_count > 0:
            novelty_msg = (
                f"🌟 {high_count} HIGH-NOVELTY finding(s) identified! "
                f"Validation experiments have been prioritized."
            )
        elif medium_count > 0:
            novelty_msg = (
                f"🤔 {medium_count} finding(s) with partial novelty. "
                f"Differentiation experiments suggested."
            )
        else:
            novelty_msg = (
                "📚 Findings largely align with existing literature. "
                "Recommendations focus on incremental improvements."
            )
        
        summary = base_integration + "\n\n" if base_integration else ""
        summary += (
            f"**Novelty-Informed Prioritization:**\n"
            f"- Claims assessed: {total_count}\n"
            f"- Average novelty score: {avg_score:.1f}/5\n"
            f"- High novelty (4-5): {high_count}\n"
            f"- Medium novelty (3): {medium_count}\n\n"
            f"{novelty_msg}"
        )
        
        return summary

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    
    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Override in subclasses to add agent-specific state fields."""
        return {}
    
    def _init_state(self, **context) -> None:
        """Initialize state for a new session."""
        if self.state.get("session_id") is None:
            self.state = {
                "session_id": str(uuid.uuid4()),
                "start_time": datetime.now().isoformat(),
                "agent_type": self.agent_type,
                "action_history": [],
                "status": "initialized"
            }
            self.state.update(self._get_initial_state_fields())
        
        for key, value in context.items():
            self.state[key] = value
        
        self.state["status"] = "active"
        self._save_state()

    def _log_action(
        self,
        action: str,
        input_ctx: Dict[str, Any],
        result: Dict[str, Any],
        rationale: Optional[str] = None,
        feedback: Optional[str] = None
    ) -> None:
        """Record an atomic action to state history and save."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "input": input_ctx,
            "rationale": rationale,
            "result": self._normalize_result(result),
            "feedback": feedback
        }
        
        if "action_history" not in self.state:
            self.state["action_history"] = []
        
        self.state["action_history"].append(entry)
        self._save_state()

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """Normalize result for JSON serialization."""
        if isinstance(result, dict):
            return result.copy()
        return {"raw_result": str(result)}
    
    def _get_state_filename(self) -> str:
        return f"{self.agent_type}_state.json"

    def _save_state(self) -> None:
        """Persist state to disk."""
        state_file = self.output_dir / self._get_state_filename()
        try:
            with open(state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save {self.agent_type} state: {e}")

    def load_state(self, state_path: str) -> bool:
        """Restore state from disk."""
        path = Path(state_path)
        if not path.exists():
            self.logger.warning(f"State file not found: {state_path}")
            return False
        
        try:
            with open(path, 'r') as f:
                self.state = json.load(f)
            
            if "action_history" not in self.state:
                self.state["action_history"] = []
            
            self.logger.info(f"Restored {self.agent_type} state: session {self.state.get('session_id')}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to load {self.agent_type} state: {e}")
            return False

    # =========================================================================
    # IMAGE STORAGE
    # =========================================================================
    
    def _store_analysis_images(self, images: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> None:
        """Store analysis images for potential reuse in recommendations."""
        self._stored_analysis_images = images.copy() if images else []
        self._stored_analysis_metadata = metadata or {}
        self.logger.debug(f"Stored {len(self._stored_analysis_images)} analysis images")

    def _get_stored_analysis_images(self) -> List[Dict[str, Any]]:
        """Retrieve stored analysis images."""
        return self._stored_analysis_images.copy()

    def _clear_stored_images(self) -> None:
        """Clear stored images to free memory."""
        self._stored_analysis_images = []
        self._stored_analysis_metadata = {}

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================
    
    def _validate_scientific_claims(self, scientific_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate scientific claims structure and content."""
        valid_claims = []

        if not isinstance(scientific_claims, list):
            self.logger.warning(f"'scientific_claims' was not a list: {type(scientific_claims)}")
            return valid_claims

        required_keys = ["claim", "scientific_impact", "has_anyone_question", "keywords"]
        
        for claim in scientific_claims:
            if isinstance(claim, dict) and all(k in claim for k in required_keys):
                if isinstance(claim.get("keywords"), list):
                    valid_claims.append(claim)
                else:
                    self.logger.warning("Claim skipped: 'keywords' not a list")
            else:
                self.logger.warning("Claim skipped: missing required keys")
        
        return valid_claims

    def _validate_measurement_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate measurement recommendations structure."""
        valid_recommendations = []
        
        if not isinstance(recommendations, list):
            return valid_recommendations
        
        required_keys = ["description", "scientific_justification", "priority"]
        
        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in required_keys):
                priority = rec.get("priority")
                if isinstance(priority, int) and 1 <= priority <= 5:
                    valid_recommendations.append(rec)
        
        return sorted(valid_recommendations, key=lambda x: x.get("priority", 5))

    def _validate_structure_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and sort structure recommendations."""
        valid_recommendations = []
        
        if not isinstance(recommendations, list):
            self.logger.warning(f"'structure_recommendations' was not a list")
            return valid_recommendations

        for rec in recommendations:
            if isinstance(rec, dict) and all(k in rec for k in ["description", "scientific_interest", "priority"]):
                if isinstance(rec.get("priority"), int):
                    valid_recommendations.append(rec)
        
        return sorted(valid_recommendations, key=lambda x: x.get("priority", 99))

    # =========================================================================
    # SPATIAL SCALE CALCULATION
    # =========================================================================
    
    def _calculate_spatial_scale(
        self,
        system_info: Dict[str, Any],
        image_shape: tuple
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate spatial scale from system metadata."""
        if not system_info:
            self.logger.info("No system_info provided. Physical scale not applied.")
            return None, None
        
        pixel_size = None
        
        if 'spatial_info' in system_info:
            spatial = system_info['spatial_info']
            if isinstance(spatial, dict):
                pixel_size = spatial.get('nm_per_pixel') or spatial.get('pixel_size_nm')
        
        if pixel_size is None:
            key_candidates = [
                'pixel_size_nm', 'nm_per_pixel', 'scale_nm_per_pixel',
                'pixel_size', 'pixelSize', 'pixel_scale', 'scale', 'resolution_nm'
            ]
            for key in key_candidates:
                if key in system_info and system_info[key] is not None:
                    pixel_size = system_info[key]
                    break
        
        if pixel_size is None:
            self.logger.info("No spatial calibration found in system metadata.")
            return None, None
        
        try:
            pixel_size = float(pixel_size)
        except (TypeError, ValueError):
            self.logger.warning(f"Invalid pixel_size value: {pixel_size}")
            return None, None
        
        if pixel_size <= 0:
            self.logger.warning(f"Invalid pixel_size value: {pixel_size}. Must be positive.")
            return None, None
        
        unit = system_info.get('pixel_size_unit', 'nm').lower()
        if unit in ['um', 'µm', 'micron', 'microns', 'micrometer']:
            pixel_size *= 1000
        elif unit in ['pm', 'picometer']:
            pixel_size /= 1000
        elif unit in ['a', 'angstrom', 'å']:
            pixel_size /= 10
        
        h, w = image_shape[:2]
        fov = max(h, w) * pixel_size
        
        self.logger.info(f"Spatial calibration: {pixel_size:.3f} nm/pixel, FOV: {fov:.1f} nm")
        
        return pixel_size, fov

    # =========================================================================
    # REFINEMENT WITH FEEDBACK
    # =========================================================================
    
    def _refine_analysis_with_feedback(
        self,
        original_analysis: str,
        original_claims: List[Dict[str, Any]],
        user_feedback: str,
        instruction_prompt: str,
        stored_images: List[Dict[str, Any]] = None,
        system_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Use LLM to refine analysis and claims based on user feedback."""
        try:
            refinement_prompt = f"""
{instruction_prompt}

## REFINEMENT TASK
You previously generated this analysis and claims:

ORIGINAL DETAILED ANALYSIS:
{original_analysis}

ORIGINAL CLAIMS:
{json.dumps(original_claims, indent=2)}

A human expert provided this feedback:
"{user_feedback}"

Use this feedback *thoughtfully* to refine both the detailed analysis and scientific claims. 
Maintain the same JSON output format with "detailed_analysis" and "scientific_claims" keys.
"""
            
            prompt_parts = [refinement_prompt]
            
            if stored_images:
                for img_data in stored_images:
                    if isinstance(img_data, dict) and 'label' in img_data and 'data' in img_data:
                        prompt_parts.append(f"\n{img_data['label']}:")
                        prompt_parts.append({"mime_type": "image/jpeg", "data": img_data['data']})
            
            if system_info:
                prompt_parts.append(self._build_system_info_prompt_section(system_info))
            
            prompt_parts.append("\nProvide the refined analysis in JSON format.")
            
            self.logger.info("🔄 Refining analysis based on feedback...")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                return {"error": "Refinement failed", "details": error_dict}
            
            refined_claims = result_json.get("scientific_claims", [])
            validated_claims = self._validate_scientific_claims(refined_claims)
            
            self.logger.info(f"✅ Refinement complete: {len(validated_claims)} validated claims")
            
            return {
                "detailed_analysis": result_json.get("detailed_analysis", original_analysis),
                "scientific_claims": validated_claims
            }
            
        except Exception as e:
            self.logger.error(f"Analysis refinement failed: {e}")
            return {"error": "Refinement failed", "details": str(e)}