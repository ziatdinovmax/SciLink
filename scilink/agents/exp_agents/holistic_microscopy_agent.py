import logging
from typing import Dict, Any, List

from .base_agent import BaseAnalysisAgent
from .atomistic_microscopy_agent import AtomisticMicroscopyAnalysisAgent
from .microscopy_agent import MicroscopyAnalysisAgent

from .human_feedback import SimpleFeedbackMixin
from .instruct import MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS, HOLISTIC_SYNTHESIS_INSTRUCTIONS


class HolisticMicroscopyAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    A composite agent that runs a staged, interactive analysis using atomistic
    and general microscopy agents, allowing for feedback at each step before
    a final synthesis.
    """
    def __init__(self, google_api_key: str = None, model_name: str = "gemini-2.5-pro-preview-06-05", 
                 local_model: str = None, enable_human_feedback: bool = True, **kwargs):
        """
        Initializes the agent and its sub-agents, passing the feedback flag to them.
        """
        super().__init__(
            google_api_key=google_api_key,
            model_name=model_name,
            local_model=local_model,
            enable_human_feedback=enable_human_feedback
        )
        self.logger = logging.getLogger(__name__)

        common_agent_args = {
            'google_api_key': google_api_key,
            'model_name': model_name,
            'local_model': local_model,
            'enable_human_feedback': enable_human_feedback 
        }
        
        self.atomistic_agent = AtomisticMicroscopyAnalysisAgent(**common_agent_args)

        general_agent_args = common_agent_args.copy()
        general_agent_args['fft_nmf_settings'] = {
            'FFT_NMF_ENABLED': True, 'FFT_NMF_AUTO_PARAMS': True
        }
        self.general_agent = MicroscopyAnalysisAgent(**general_agent_args)
        self.logger.info("HolisticMicroscopyAgent initialized for staged feedback.")

    def synthesize_analyses(self, atomistic_text: str, general_text: str, 
                            atomistic_images: List[Dict], general_images: List[Dict], 
                            system_info: Dict) -> Dict[str, Any]:
        """
        Takes analysis texts and images from multiple sources and synthesizes them.
        """
        self.logger.info("🧠 Synthesizing multi-modal results with text and images...")
        
        prompt_parts = [HOLISTIC_SYNTHESIS_INSTRUCTIONS]
        
        prompt_parts.append("--- Atomistic Analysis Results ---")
        prompt_parts.append("## Text Summary from Atomistic Agent ##")
        prompt_parts.append(atomistic_text)
        prompt_parts.append("## Visual Evidence from Atomistic Agent ##")
        for img in atomistic_images:
            prompt_parts.append(img.get('label', 'Analysis Image') + ":")
            prompt_parts.append({"mime_type": "image/jpeg", "data": img['data']})

        prompt_parts.append("\n--- General (FFT-NMF) Analysis Results ---")
        prompt_parts.append("## Text Summary from General Agent ##")
        prompt_parts.append(general_text)
        prompt_parts.append("## Visual Evidence from General Agent ##")
        for img in general_images:
            prompt_parts.append(img.get('label', 'Analysis Image') + ":")
            prompt_parts.append({"mime_type": "image/jpeg", "data": img['data']})

        prompt_parts.append(self._build_system_info_prompt_section(system_info))
        prompt_parts.append("Synthesize ALL provided information into a final report.")
        
        response = self.model.generate_content(
            prompt_parts, 
            generation_config=self.generation_config
        )
        final_json, error = self._parse_llm_response(response)
        
        if error:
            return {"status": "error", "message": "Synthesis failed.", "details": error}
        
        final_json['status'] = 'success'
        self.logger.info("✅ Successfully synthesized holistic results.")
        return final_json

    def analyze_for_claims(self, image_path: str, system_info: Dict, **kwargs) -> Dict[str, Any]:
        """
        Runs a sequential, interactive analysis and synthesis workflow.
        """
        print("\n" + "="*80)
        print("🔬 STAGE 1 of 3: Atomistic Analysis")
        print("="*80)
        atomistic_result = self.atomistic_agent.analyze_microscopy_image_for_claims(image_path, system_info)
        
        if "error" in atomistic_result:
            self.logger.error("Atomistic analysis failed. Aborting holistic workflow.")
            return atomistic_result
            
        atomistic_text = atomistic_result.get("detailed_analysis", "")
        atomistic_images = self.atomistic_agent._get_stored_analysis_images()

        print("\n" + "="*80)
        print("🗺️ STAGE 2 of 3: General Microscopy Analysis (FFT-NMF)")
        print("="*80)
        general_result = self.general_agent.analyze_microscopy_image_for_claims(image_path, system_info)

        if "error" in general_result:
            self.logger.error("General analysis failed. Aborting holistic workflow.")
            return general_result

        general_text = general_result.get("detailed_analysis", "")
        general_images = self.general_agent._get_stored_analysis_images()

        print("\n" + "="*80)
        print("🧠 STAGE 3 of 3: Final Synthesis")
        print("="*80)
        synthesized_result = self.synthesize_analyses(
            atomistic_text=atomistic_text,
            general_text=general_text,
            atomistic_images=atomistic_images,
            general_images=general_images,
            system_info=system_info
        )

        if synthesized_result.get('status') != 'success':
            return synthesized_result

        final_result = self._apply_feedback_if_enabled(
            claims_result=synthesized_result,
            image_path=image_path,
            system_info=system_info
        )

        final_result['source_analyses'] = {
            'atomistic_microscopy': atomistic_result,
            'general_microscopy': general_result
        }
        
        return final_result
    
    def _get_claims_instruction_prompt(self) -> str:
        return HOLISTIC_SYNTHESIS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        """
        Returns the instruction prompt for generating next-step measurement recommendations.
        For a holistic analysis, we can use the general microscopy recommendations prompt.
        """
        return MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS