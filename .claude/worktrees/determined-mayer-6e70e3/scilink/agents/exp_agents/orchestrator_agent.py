import logging
import json

from .fft_microscopy_agent import FFTMicroscopyAnalysisAgent
from .sam_microscopy_agent import SAMMicroscopyAnalysisAgent
from .hyperspectral_analysis_agent import HyperspectralAnalysisAgent
from .curve_fitting_agent import CurveFittingAgent

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params


# =============================================================================
# ORCHESTRATOR INSTRUCTIONS
# =============================================================================
ORCHESTRATOR_INSTRUCTIONS = """
You are an AI orchestrator that selects the most appropriate analysis agent for experimental data.

## Available Agents

| ID | Agent | Use Case |
|----|-------|----------|
| 0 | FFTMicroscopyAnalysisAgent | Microstructure analysis via FFT/NMF - grains, phases, domains, periodic structures, AND atomic-resolution images (STEM, TEM, AFM at atomic scale) |
| 1 | SAMMicroscopyAnalysisAgent | Particle/object segmentation - counting particles, size distributions, object detection |
| 2 | HyperspectralAnalysisAgent | Spectroscopic and hyperspectral data - 3D datacubes with spatial + spectral dimensions |
| 3 | CurveFittingAgent | 1D curve/spectrum fitting - peaks, band gaps, kinetics, DSC, XRD patterns, any x-y data |

## Selection Guidelines

**FFTMicroscopyAnalysisAgent (ID: 0)** - Choose when:
- Analyzing microstructure (grains, grain boundaries, phases, domains)
- Looking for periodic patterns or crystallographic features
- Working with atomic-resolution images (HAADF-STEM, ABF-STEM, HRTEM, atomic AFM)
- Performing Fourier analysis or NMF decomposition
- General microscopy image analysis

**SAMMicroscopyAnalysisAgent (ID: 1)** - Choose when:
- Need to count or segment discrete particles/objects
- Measuring particle size distributions
- Detecting and outlining distinct features (nanoparticles, cells, defects as objects)
- The image contains clearly separable objects to be counted

**HyperspectralAnalysisAgent (ID: 2)** - Choose when:
- Data is 3D with spatial (x,y) and spectral (wavelength/energy) dimensions
- EELS spectrum imaging, EDS mapping, Raman imaging
- Need spectral unmixing or component analysis across spatial regions

**CurveFittingAgent (ID: 3)** - Choose when:
- Data is 1D (x-y pairs): spectra, curves, time series
- DSC, TGA, XRD, UV-Vis, PL, Raman spectra (single point)
- Need to fit peaks, extract parameters, find transitions
- Any tabular data with independent and dependent variables

## Response Format

You MUST respond with valid JSON only, no other text:

```json
{
  "agent_id": <integer 0-3>,
  "reasoning": "<brief explanation of why this agent is appropriate>"
}
```

## Examples

User: data_type: microscopy, system_info: {"technique": "HAADF-STEM", "material": "MoS2 monolayer"}
Response: {"agent_id": 0, "reasoning": "HAADF-STEM of 2D material requires FFTMicroscopyAnalysisAgent for atomic-resolution microstructure analysis."}

User: data_type: microscopy, system_info: {"technique": "SEM", "goal": "count nanoparticles"}
Response: {"agent_id": 1, "reasoning": "Particle counting task requires SAMMicroscopyAnalysisAgent for segmentation."}

User: data_type: spectroscopy, system_info: {"technique": "EELS-SI", "material": "oxide interface"}
Response: {"agent_id": 2, "reasoning": "EELS spectrum imaging is hyperspectral data requiring HyperspectralAnalysisAgent."}

User: data_type: curve, system_info: {"technique": "DSC", "material": "metal alloy"}
Response: {"agent_id": 3, "reasoning": "DSC is 1D thermal analysis data requiring CurveFittingAgent for peak fitting."}
"""

# Mapping from integer ID to the corresponding agent class
# Updated: Removed AtomisticMicroscopyAnalysisAgent, added CurveFittingAgent
AGENT_MAP = {
    0: FFTMicroscopyAnalysisAgent,      # Microstructure via FFT/NMF (including atomic-resolution)
    1: SAMMicroscopyAnalysisAgent,       # Particle/object segmentation
    2: HyperspectralAnalysisAgent,       # Spectroscopic/hyperspectral data
    3: CurveFittingAgent                 # 1D curve/spectrum fitting
}

# Human-readable descriptions for logging
AGENT_DESCRIPTIONS = {
    0: "FFTMicroscopyAnalysisAgent - Microstructure analysis (grains, phases, domains, atomic-resolution)",
    1: "SAMMicroscopyAnalysisAgent - Particle/object segmentation and counting",
    2: "HyperspectralAnalysisAgent - Spectroscopic and hyperspectral data analysis",
    3: "CurveFittingAgent - 1D curve and spectrum fitting"
}


class OrchestratorAgent:
    """
    An LLM-powered agent that selects the most appropriate experimental analysis agent.
    
    Agent IDs:
        0: FFTMicroscopyAnalysisAgent - For microstructure analysis including atomic-resolution
        1: SAMMicroscopyAnalysisAgent - For particle/object segmentation
        2: HyperspectralAnalysisAgent - For spectroscopic/hyperspectral data
        3: CurveFittingAgent - For 1D curve/spectrum fitting
    """
    def __init__(self, 
                 api_key: str | None = None, 
                 model_name: str = "claude-opus-4-6",
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
            source="OrchestratorAgent"
        )
        
        # Initialize Model
        if self.base_url:
            if 'gguf' in self.base_url:
                logging.info(f"💻 Using local agent (GGUF path detected): {self.base_url}")
                from ...wrappers.llama_wrapper import LocalLlamaModel
                self.model = LocalLlamaModel(self.base_url)
            else:
                logging.info(f"🏛️ Using OpenAI-compatible agent as orchestrator: {self.base_url}")
                if self.api_key is None:
                    self.api_key = get_internal_proxy_key()
                
                if not self.api_key:
                    raise ValueError("API key required for internal proxy.")

                self.model = OpenAIAsGenerativeModel(model=model_name, api_key=self.api_key, base_url=self.base_url)
        else:
            logging.info(f"☁️ Using LiteLLM agent as the orchestrator: {model_name}")
            # LiteLLM finds API keys in env vars automatically
            self.model = LiteLLMGenerativeModel(
                model=model_name, 
                api_key=self.api_key
            )

        self.generation_config = None

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
            self.logger.error(f"Error parsing LLM JSON response: {e}")
            error_dict = {"error": "Failed to parse JSON from LLM response", "details": str(e), "raw_response": response.text}
            return None, error_dict

    def select_agent(self, data_type: str, system_info: dict | str | None = None, image_path: str | None = None) -> tuple[int, str | None]:
        """
        Selects the appropriate experimental agent by asking the LLM.

        Args:
            data_type: The primary type of data, e.g., 'microscopy', 'spectroscopy', 'curve', '1D curve'.
            system_info: Additional information about the system or analysis goal (metadata).
            image_path: Optional path to an image for visual context (only for image data types).

        Returns:
            A tuple containing the integer key for the selected agent and the reasoning string.
            Returns (-1, error_message) on failure.
        """
        self.logger.info(f"Orchestrator LLM selecting agent for data_type: '{data_type}'")

        # Require system_info to be provided for an informed decision.
        if not system_info or not str(system_info).strip():
            error_msg = "system_info (metadata) must be provided for the orchestrator to select an agent."
            self.logger.error(error_msg)
            return -1, error_msg

        prompt_parts = [ORCHESTRATOR_INSTRUCTIONS]
        
        # Add image analysis if available (for microscopy/image data)
        if image_path:
            try:
                from .utils import load_image, preprocess_image, convert_numpy_to_jpeg_bytes
                
                # Load and preprocess image (resize to a max dimension to save tokens)
                image = load_image(image_path)
                resized_image, _ = preprocess_image(image, max_dim=512)  # Downsample for the orchestrator
                image_bytes = convert_numpy_to_jpeg_bytes(resized_image, quality=80)
                
                prompt_parts.append("\n--- Image for Analysis ---")
                prompt_parts.append(
                    "Analyze this image to help determine the appropriate agent. "
                    "Consider: Is this atomic-resolution? Are there discrete particles to count? "
                    "Is it a microstructure with grains/domains? What analysis would be most appropriate?"
                )
                prompt_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
                
            except Exception as e:
                self.logger.warning(f"Failed to load image for orchestrator: {e}")
                prompt_parts.append("\n(Image could not be loaded - selecting based on metadata only)")
        
        # Add metadata and data type
        prompt_parts.append("\n--- Metadata & Data Type ---")
        prompt_parts.append(f"data_type: {data_type}")
        prompt_parts.append(f"metadata/system_info: {json.dumps(system_info) if isinstance(system_info, dict) else str(system_info)}")
        
        prompt_parts.append("\n--- Your Task ---")
        prompt_parts.append(
            "Based on the metadata (and image if provided), select the most appropriate agent. "
            "Respond with JSON only: {\"agent_id\": <0-3>, \"reasoning\": \"<explanation>\"}"
        )

        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
            )
            result_json, error_dict = self._parse_llm_response(response)

            if error_dict:
                self.logger.error(f"Orchestrator LLM call failed: {error_dict}")
                return -1, f"Orchestrator LLM call failed: {error_dict.get('details', 'Unknown error')}"

            agent_id = result_json.get('agent_id')
            reasoning = result_json.get('reasoning', 'No reasoning provided.')
            self.logger.info(f"\n\n🧠 Orchestrator Reasoning: {reasoning}\n")

            if isinstance(agent_id, int) and agent_id in AGENT_MAP:
                return agent_id, reasoning
            else:
                error_msg = f"LLM returned an invalid agent ID: {agent_id}. Valid IDs: {list(AGENT_MAP.keys())}"
                self.logger.warning(error_msg)
                return -1, error_msg

        except Exception as e:
            error_msg = f"An unexpected error occurred during agent selection: {e}"
            self.logger.exception(error_msg)
            return -1, error_msg