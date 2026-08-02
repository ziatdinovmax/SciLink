import logging
from typing import Dict, Any, Optional

from .experiment_novelty_workflow import ExperimentNoveltyAssessment 

class MicroscopyAnalysisWorkflow(ExperimentNoveltyAssessment):
    """
    A specialized workflow for analyzing microscopy image data (e.g., TEM, SEM,
    STEM, AFM) and assessing its novelty against the literature.

    This workflow inherits from ExperimentNoveltyAssessment, pre-setting the
    data_type to 'microscopy'. It allows manual agent selection via `agent_id`
    or automatic selection by the orchestrator.
    It supports optional steps like DFT and measurement recommendations.
    """

    def __init__(self,
                 agent_id: Optional[int] = None, # Allow specific agent selection
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 output_dir: str = "microscopy_analysis_output",
                 max_wait_time: int = 1000,
                 dft_recommendations: bool = False,
                 measurement_recommendations: bool = False,
                 enable_human_feedback: bool = True,
                 display_agent_logs: bool = True,
                 **analyzer_kwargs):
        """
        Initialize the Microscopy Novelty Assessment workflow.

        Args:
            agent_id (int, optional): Manually specify the microscopy agent ID
                                      (0=General, 1=SAM, 2=Atomistic, 4=Holistic).
                                      If None, the orchestrator selects automatically.
            google_api_key (str, optional): Google API key. Defaults to env.
            futurehouse_api_key (str, optional): FutureHouse API key. Defaults to env.
            analysis_model (str, optional): Model name for analysis/scoring.
            local_model (str, optional): URL for local OpenAI-compatible endpoint.
            output_dir (str, optional): Directory for outputs.
            max_wait_time (int, optional): Max literature search wait time.
            dft_recommendations (bool, optional): Generate DFT recommendations. Defaults to True.
            measurement_recommendations (bool, optional): Generate measurement recommendations. Defaults to False.
            enable_human_feedback (bool, optional): Enable human feedback loop. Defaults to True.
            display_agent_logs (bool, optional): Show detailed agent logs. Defaults to True.
            **analyzer_kwargs: Additional keyword arguments passed to the underlying MicroscopyAnalyzer.
        """
        self.logger = logging.getLogger(__name__) # Logger needed before super().__init__

        super().__init__(
            data_type='microscopy', # Explicitly set for this workflow
            agent_id=agent_id, # Pass agent_id to the base class constructor
            google_api_key=google_api_key,
            futurehouse_api_key=futurehouse_api_key,
            analysis_model=analysis_model,
            local_model=local_model,
            output_dir=output_dir,
            max_wait_time=max_wait_time,
            dft_recommendations=dft_recommendations,
            measurement_recommendations=measurement_recommendations,
            enable_human_feedback=enable_human_feedback,
            display_agent_logs=display_agent_logs,
            # Pass any extra kwargs specific to the MicroscopyAnalyzer
            **analyzer_kwargs
        )
        self.logger.info("MicroscopyAnalysisWorkflow (subclass) initialized.")

    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary specific to the microscopy workflow."""
        base_summary = super().get_summary(workflow_result)
        summary = base_summary.replace("Experiment Novelty Assessment Summary",
                                       "Microscopy Novelty Assessment Summary")
        return summary

