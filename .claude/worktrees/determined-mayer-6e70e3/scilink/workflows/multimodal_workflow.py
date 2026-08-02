import logging
import json
import sys
from io import StringIO
import inspect
from typing import Dict, Any, List
from pathlib import Path

from ..agents.exp_agents import (
    AtomisticMicroscopyAnalysisAgent,
    MicroscopyAnalysisAgent,
    SAMMicroscopyAnalysisAgent,
    HyperspectralAnalysisAgent,
    CurveFittingAgent
    
)
from ..agents.exp_agents.instruct import HOLISTIC_EXPERIMENTAL_SYNTHESIS_INSTRUCTIONS


class MultiModalExperimentWorkflow:
    """
    A high-level workflow to orchestrate a multi-modal experimental analysis.

    This workflow coordinates multiple specialist agents (for microscopy, spectroscopy, etc.)
    to analyze their respective data and then synthesizes all findings into a single,
    cohesive scientific narrative and a set of high-level claims.
    """
    def __init__(self, google_api_key: str, model_name: str = "gemini-2.5-pro-preview-06-05",
                 output_dir: str = "holistic_output", **kwargs):
        """
        Initializes the workflow and all the necessary specialist agents.

        Args:
            google_api_key (str): The Google API key.
            model_name (str): The name of the Gemini model for the final synthesis step.
            output_dir (str): The directory to save the final results.
            **kwargs: Additional keyword arguments passed to the specialist agents.
        """
        self.log_capture = StringIO()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(name)s: %(message)s',
            force=True,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.StreamHandler(self.log_capture)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Base arguments required by all agents
        common_args = {'google_api_key': google_api_key, 'model_name': model_name}

        def get_agent_kwargs(agent_class):
            """Helper function to filter kwargs for a specific agent's constructor."""
            # Get the names of all valid parameters for the agent's __init__ method
            sig = inspect.signature(agent_class.__init__)
            valid_params = set(sig.parameters.keys())
            
            # Filter the provided kwargs to include only the ones this agent accepts
            agent_specific_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            
            # Combine the common args with the agent-specific ones
            return {**common_args, **agent_specific_kwargs}

        # Instantiate each agent with only the arguments it can accept
        self.atomistic_agent = AtomisticMicroscopyAnalysisAgent(**get_agent_kwargs(AtomisticMicroscopyAnalysisAgent))
        self.general_microscopy_agent = MicroscopyAnalysisAgent(**get_agent_kwargs(MicroscopyAnalysisAgent))
        self.sam_agent = SAMMicroscopyAnalysisAgent(**get_agent_kwargs(SAMMicroscopyAnalysisAgent))
        self.hyperspectral_agent = HyperspectralAnalysisAgent(**get_agent_kwargs(HyperspectralAnalysisAgent))
        self.curve_agent = CurveFittingAgent(**get_agent_kwargs(CurveFittingAgent))
        
        # The synthesis step uses a generative model directly
        self.synthesis_model = self.atomistic_agent.model # Reuse the configured model instance
        self.generation_config = self.atomistic_agent.generation_config

    def run(self, data_inputs: Dict[str, Dict], common_system_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes the full multi-modal analysis and synthesis pipeline.

        Args:
            data_inputs (Dict[str, Dict]): Maps analysis types to a dict containing
                'data_path' and an optional 'metadata_path'.
            common_system_info (Dict[str, Any], optional): A dictionary of metadata
                that applies to the entire sample (e.g., synthesis details).
        """
        all_results = []
        workflow_status = {"final_status": "started", "steps_completed": []}

        self.logger.info("Starting specialist analyses for each data modality...")

        # 1. Define the mapping from the input key to the agent instance.
        agent_mapping = {
            'atomistic_microscopy': self.atomistic_agent,
            'general_microscopy': self.general_microscopy_agent,
            'partciles_microscopy': self.sam_agent,
            'hyperspectral': self.hyperspectral_agent,
            '1Dspectroscopy': self.curve_agent
        }

        # --- Step 1: Run All Specialist Analyses ---
        for analysis_type, paths in data_inputs.items():
            agent = agent_mapping.get(analysis_type)
            if agent:
                data_path = paths.get('data_path')
                metadata_path = paths.get('metadata_path')

                if not data_path:
                    self.logger.warning(f"Skipping '{analysis_type}' because 'data_path' is missing.")
                    continue

                # 2. Load the specific metadata for this analysis.
                specific_info = {}
                if metadata_path and Path(metadata_path).exists():
                    with open(metadata_path, 'r') as f:
                        specific_info = json.load(f)
                
                # 3. Merge common and specific metadata. Specific info takes precedence.
                final_system_info = {**(common_system_info or {}), **specific_info}

                self.logger.info(f"Running {analysis_type} analysis on {data_path}...")
                result = agent.analyze_for_claims(data_path, final_system_info)
                
                if "detailed_analysis" in result:
                    all_results.append({'source': analysis_type, 'data': result, 'agent': agent})
                else:
                    self.logger.warning(f"{analysis_type} analysis failed or returned no valid data.")

        if not all_results:
            workflow_status["final_status"] = "error"
            workflow_status["message"] = "All specialist analyses failed."
            return workflow_status

        workflow_status["steps_completed"].append("specialist_analyses")

        # --- Step 2: Synthesize Results ---
        self.logger.info("Synthesizing results from all modalities...")
        
        # 4. Create a single, comprehensive metadata dictionary for the final synthesis step.
        comprehensive_info = common_system_info or {}
        for analysis_type, paths in data_inputs.items():
            metadata_path = paths.get('metadata_path')
            if metadata_path and Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    comprehensive_info.update(json.load(f))
        
        # 5. Call the synthesis method with the complete, merged info.
        synthesis_result = self._synthesize_results(all_results, comprehensive_info)
        
        workflow_status["synthesis_result"] = synthesis_result
        if synthesis_result.get("status") != "success":
            workflow_status["final_status"] = "error"
            workflow_status["message"] = "Synthesis of multi-modal data failed."
            return workflow_status
        
        workflow_status["steps_completed"].append("synthesis")
        workflow_status["final_status"] = "success"
        
        self._save_results(workflow_status)
        return workflow_status

    def _synthesize_results(self, results: List[Dict], system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses an LLM to create a single narrative from multiple analysis results.
        """
        prompt_parts = [HOLISTIC_EXPERIMENTAL_SYNTHESIS_INSTRUCTIONS]

        for res in results:
            agent_instance = res['agent']
            prompt_parts.append(f"\n--- Analysis from {res['source']} ---")
            prompt_parts.append(f"## Text Summary from {res['source']} Agent ##")
            prompt_parts.append(res['data']['detailed_analysis'])
            
            # Retrieve and add any visual evidence generated by the specialist agent
            images = agent_instance._get_stored_analysis_images()
            if images:
                prompt_parts.append(f"## Visual Evidence from {res['source']} Agent ##")
                for img in images:
                    label = img.get('label', f"{res['source']} Image")
                    prompt_parts.append(f"{label}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": img['data']})
        
        prompt_parts.append(agent_instance._build_system_info_prompt_section(system_info))

        # Final LLM call to get the synthesized report
        response = self.synthesis_model.generate_content(prompt_parts, generation_config=self.generation_config)
        final_json, error = agent_instance._parse_llm_response(response) # Reuse parsing logic

        if error:
            return {"status": "error", "details": error}
        
        final_json['status'] = 'success'
        return final_json

    def _save_results(self, result: Dict[str, Any]):
        """Saves the final workflow output to a JSON file."""
        output_file = self.output_dir / "holistic_analysis_summary.json"
        try:
            with open(output_file, 'w') as f:
                # A custom serializer might be needed if agent objects are included
                json.dump(result, f, indent=2, default=lambda o: '<object>')
            self.logger.info(f"Holistic workflow results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save holistic workflow results: {e}")