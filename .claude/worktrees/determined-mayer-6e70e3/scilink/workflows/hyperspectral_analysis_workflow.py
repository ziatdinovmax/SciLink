import logging
from typing import Dict, Any

from .experiment_novelty_workflow import ExperimentNoveltyAssessment


class HyperspectralAnalysisWorkflow(ExperimentNoveltyAssessment):
    """
    A specialized workflow for analyzing hyperspectral data (e.g., EELS, EDS maps)
    and assessing its novelty against the literature.

    This workflow inherits from ExperimentNoveltyAssessment, pre-setting the
    data_type to 'spectroscopy'. It utilizes the HyperspectralAnalysisAgent
    internally via the SpectroscopyAnalyzer.
    It supports optional steps like DFT and measurement recommendations.
    """

    def __init__(self,
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 output_dir: str = "hyperspectral_analysis_output",
                 max_wait_time: int = 1000,
                 spectral_unmixing_enabled: bool = True,
                 dft_recommendations: bool = False,
                 measurement_recommendations: bool = False,
                 enable_human_feedback: bool = True,
                 display_agent_logs: bool = True,
                 run_preprocessing: bool = True,
                 **analyzer_kwargs):
        """
        Initialize the Hyperspectral Novelty Assessment workflow.

        Args:
            google_api_key (str, optional): Google API key. Defaults to env.
            futurehouse_api_key (str, optional): FutureHouse API key. Defaults to env.
            analysis_model (str, optional): Model name for analysis/scoring.
            local_model (str, optional): URL for local OpenAI-compatible endpoint.
            output_dir (str, optional): Directory for outputs.
            max_wait_time (int, optional): Max literature search wait time.
            spectral_unmixing_enabled (bool, optional): Enable spectral unmixing. Defaults to True.
            dft_recommendations (bool, optional): Generate DFT recommendations. Defaults to False.
            measurement_recommendations (bool, optional): Generate measurement recommendations. Defaults to False.
            enable_human_feedback (bool, optional): Enable human feedback loop. Defaults to True.
            display_agent_logs (bool, optional): Show detailed agent logs. Defaults to True.
            **analyzer_kwargs: Additional keyword arguments passed to the underlying SpectroscopyAnalyzer/HyperspectralAnalysisAgent.
        """
        self.logger = logging.getLogger(__name__) # Logger needed before super().__init__

        # Call the parent class's __init__ method, explicitly setting data_type
        super().__init__(
            data_type='spectroscopy',
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
            run_preprocessing=run_preprocessing,
            # Pass specific flags and any extra kwargs for the SpectroscopyAnalyzer
            spectral_unmixing_enabled=spectral_unmixing_enabled,
            **analyzer_kwargs
        )
        self.logger.info("HyperspectralAnalysisWorkflow (subclass) initialized.")

    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary specific to the hyperspectral workflow."""
        base_summary = super().get_summary(workflow_result)
        summary = base_summary.replace("Experiment Novelty Assessment Summary",
                                       "Hyperspectral Novelty Assessment Summary")

        # Add unmixing info if available
        analysis_res = workflow_result.get("claims_generation", {})
        if analysis_res.get("status") == "success":
             unmixing_results = analysis_res.get("unmixing_results")
             if unmixing_results is not None:
                 summary += f"\nSpectral Unmixing Performed: Yes (see JSON output)"

        return summary

