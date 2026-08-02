import logging
from typing import Dict, Any

from .experiment_novelty_workflow import ExperimentNoveltyAssessment

class Spectroscopy1DAnalysisWorkflow(ExperimentNoveltyAssessment):
    """
    A specialized workflow for analyzing 1D spectroscopy or curve data
    (e.g., PL, Raman, XRD) and assessing its novelty against the literature.

    This workflow inherits from ExperimentNoveltyAssessment, pre-setting the
    data_type to 'spectroscopy' which utilizes the CurveFittingAgent for 1D data.
    It supports optional steps like DFT and measurement recommendations.
    """

    def __init__(self,
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 output_dir: str = "1d_spectroscopy_analysis_output",
                 max_wait_time: int = 1000,
                 dft_recommendations: bool = False,
                 measurement_recommendations: bool = False,
                 enable_human_feedback: bool = True,
                 display_agent_logs: bool = True,
                 run_preprocessing: bool = True,
                 **analyzer_kwargs):
        """
        Initialize the 1D Spectroscopy Novelty Assessment workflow.

        Args:
            google_api_key (str, optional): Google API key. Defaults to env.
            futurehouse_api_key (str, optional): FutureHouse API key. Defaults to env.
            analysis_model (str, optional): Model name for analysis/scoring.
            local_model (str, optional): URL for local OpenAI-compatible endpoint.
            output_dir (str, optional): Directory for outputs.
            max_wait_time (int, optional): Max literature search wait time.
            dft_recommendations (bool, optional): Generate DFT recommendations. Defaults to False.
            measurement_recommendations (bool, optional): Generate measurement recommendations. Defaults to False.
            enable_human_feedback (bool, optional): Enable human feedback loop. Defaults to True.
            display_agent_logs (bool, optional): Show detailed agent logs. Defaults to True.
            **analyzer_kwargs: Additional keyword arguments passed to the underlying CurveAnalyzer/CurveFittingAgent.
        """
        self.logger = logging.getLogger(__name__) # Logger needed before super().__init__ potentially logs

        super().__init__(
            data_type='curve',
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
            # Pass any extra kwargs specific to the CurveAnalyzer (which handles CurveFittingAgent)
            **analyzer_kwargs
        )
        self.logger.info("1DSpectroscopyAnalysisWorkflow (subclass) initialized.")

    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary specific to the 1D spectroscopy workflow."""
        base_summary = super().get_summary(workflow_result)
        # Modify the title or add specific details if needed
        summary = base_summary.replace("Experiment Novelty Assessment Summary",
                                       "1D Spectroscopy/Curve Novelty Assessment Summary")

        # Add fitting parameters info if available
        analysis_res = workflow_result.get("claims_generation", {})
        if analysis_res.get("status") == "success":
             fitting_params = analysis_res.get("fitting_parameters")
             if fitting_params is not None:
                 summary += f"\nFitting Parameters Extracted: Yes (see JSON output)"

        return summary

