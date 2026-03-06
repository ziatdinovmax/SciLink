"""
CurveFittingAgent: Curve Fitting Agent for Spectroscopic Analysis

This module provides a curve fitting agent that handles both single spectrum
analysis and spectral series analysis using the same unified architecture.

Quality control features:
- Automatic model retry when R² is inadequate (configurable threshold)
- Statistical outlier detection for series (may indicate interesting physics)
- Adaptive refit of flagged spectra with independent model selection
- Consistency pass to align refitted models across spectra
- Human feedback integration for unresolved quality issues

For series analysis:
1. Carefully fit the first spectrum with full LLM planning and quality control
2. Lock the fitting model and strategy for remaining spectra
3. Detect and flag spectra where the locked model fails (low R²)
4. Adaptive refit: re-analyze flagged spectra independently with full QC,
   injecting experimental context (technique, sample, series position) and
   series context (successful fits, neighbor info) into the refit prompt
5. Consistency pass: if a majority of refitted spectra converge on the same
   model, re-refit outliers using the consensus model as peer evidence
6. Generate custom analysis code for trend visualization
7. Synthesize findings across the series, including refit analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from .base_agent import BaseAnalysisAgent, AnalysisInput
from .human_feedback import SimpleFeedbackMixin
from ...executors import ScriptExecutor, require_sandbox_approval
from ..lit_agents.literature_agent import FittingModelLiteratureAgent
from .preprocess import CurvePreprocessingAgent
from .pipelines.curve_fitting_pipelines import create_unified_curve_fitting_pipeline
from ...tools.curve_fitting_tools import load_curve_data, plot_curve_to_bytes
from ._deprecation import normalize_params
from ...skills.loader import load_skill

from .instruct import (
    FITTING_INTERPRETATION_INSTRUCTIONS,
    CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS,
)


logger = logging.getLogger(__name__)


class CurveFittingAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Unified Curve Fitting Agent for spectroscopic analysis.

    ALL analysis follows the series processing pattern:
    - Single spectrum analysis = series of 1
    - Multiple spectra = standard series processing
    - Numpy array stack = series processing

    Quality control:
    - LLM verification loop (n iterations) to catch and fix fit issues automatically
    - Human feedback for additional refinement (if enabled)
    - Automatic model retry when R² < threshold
    - Statistical outlier detection for series (may indicate interesting physics)
    - Adaptive refit of flagged spectra with independent model selection
    - Consistency pass to align refitted models when a majority agrees
    - Human feedback for no-consensus and R²-drop scenarios (if enabled)

    For series analysis, the fitting model is carefully selected on the
    first spectrum and then LOCKED for consistent analysis across all spectra.
    When the locked model fails on some spectra (e.g., due to a phase
    transition), the adaptive refit step re-analyzes those spectra
    independently. The refit prompt includes experimental context (technique,
    sample, series position) and series context (successful fits summary,
    failed indices, neighbor info) so the LLM doesn't analyze in isolation.

    After independent refits, a consistency pass checks whether a majority of
    refitted spectra converged on the same model. If so, minority spectra are
    re-refitted with the consensus model as peer evidence. If no majority
    exists and human feedback is enabled, the user is asked to select a model.
    If a consensus refit produces lower R² than the independent result, the
    user is asked which to keep (when human feedback is enabled).

    Security:
    - This agent executes LLM-generated Python code for curve fitting
    - A sandbox check is performed at initialization
    - If no sandbox (Docker/VM/Colab) is detected, user is prompted to confirm
    - Use UNSAFE_EXECUTION_OK=true environment variable to bypass in CI/CD

    Args:
        api_key: LLM API key
        model_name: LLM model name
        base_url: LLM API base URL
        output_dir: Output directory
        futurehouse_api_key: FutureHouse API key for literature
        use_literature: Enable literature search (default: False)
        run_preprocessing: Enable data preprocessing
        enable_human_feedback: Enable feedback loop (controls human-in-the-loop
            for fitting approach refinement, no-consensus model selection, and
            R²-drop decisions during adaptive refit)
        executor_timeout: Script timeout in seconds
        r2_threshold: Minimum acceptable R² value (default: 0.95). Spectra
            below this threshold are flagged for adaptive refit.
        max_model_retries: Max alternative models to try (default: 3)
        outlier_sigma: Sigma threshold for outlier detection (default: 2.0)
        max_verification_iterations: Max LLM verification iterations (default: 3)

    Example:
        agent = CurveFittingAgent(api_key="...", use_literature=True)

        # Single spectrum
        result = agent.analyze("spectrum.csv")

        # Multiple spectra (series)
        result = agent.analyze(["spec1.csv", "spec2.csv", "spec3.csv"])

        # Numpy stack
        result = agent.analyze(my_spectra_stack)

        # With metadata and hints
        result = agent.analyze(
            "spectrum.csv",
            system_info={"sample": "TiO2"},
            hints="Focus on the band gap"
        )

        # Series with metadata
        result = agent.analyze(
            spectra_paths,
            series_metadata={
                "variable": "temperature",
                "values": [300, 350, 400, 450, 500],
                "unit": "K"
            }
        )

        # Custom quality settings
        agent = CurveFittingAgent(
            api_key="...",
            r2_threshold=0.90,              # Accept lower quality fits
            max_model_retries=5,            # Try more alternatives
            outlier_sigma=3.0,              # Less aggressive outlier detection
            max_verification_iterations=5   # More verification passes
        )

        # Get measurement recommendations
        recommendations = agent.recommend_measurements(analysis_result=result)

    Raises:
        RuntimeError: If sandbox check fails and user declines to proceed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.1-pro-preview",
        base_url: str | None = None,
        output_dir: str = "curve_analysis_output",
        # Deprecated parameters
        google_api_key: str | None = None,
        local_model: str | None = None,
        # Agent configuration
        futurehouse_api_key: str | None = None,
        use_literature: bool = False,
        run_preprocessing: bool = True,
        enable_human_feedback: bool = True,
        executor_timeout: int = 300,
        max_wait_time: int = 1000,
        # Quality control settings
        r2_threshold: float = 0.95,
        max_model_retries: int = 3,
        outlier_sigma: float = 2.0,
        max_verification_iterations: int = 5,
        **kwargs,
    ):
        # ====================================================================
        # SANDBOX CHECK - Must happen first, before any expensive operations
        # ====================================================================
        # This agent executes LLM-generated code, so we verify the environment
        # is sandboxed (Docker/VM/Colab) or get explicit user approval.
        # The global cache in require_sandbox_approval() ensures users are
        # only prompted once per session, even if multiple agents are created.
        
        if not require_sandbox_approval(
            context="CurveFittingAgent (curve fitting analysis)"
        ):
            raise RuntimeError(
                "CurveFittingAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )

        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="CurveFittingAgent"
        )

        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
        )

        self.agent_type = "curve_fitting"
        self.use_literature = use_literature
        self.output_dir = Path(self.output_dir).resolve()

        # Quality control settings
        self.r2_threshold = r2_threshold
        self.max_model_retries = max_model_retries
        self.outlier_sigma = outlier_sigma
        self.max_verification_iterations = max_verification_iterations

        self.executor = ScriptExecutor(timeout=executor_timeout)

        # Optional preprocessor
        self.run_preprocessing = run_preprocessing
        self.preprocessor = None
        if run_preprocessing:
            self.preprocessor = CurvePreprocessingAgent(
                api_key=self.api_key,
                model_name=model_name,
                base_url=self.base_url,
                output_dir=os.path.join(self.output_dir, "preprocessing"),
                executor_timeout=executor_timeout,
            )

        # Optional literature agent
        self.literature_agent = None
        if use_literature:
            lit_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")
            if lit_key:
                try:
                    self.literature_agent = FittingModelLiteratureAgent(
                        api_key=lit_key, max_wait_time=max_wait_time
                    )
                    logger.info("Literature agent initialized")
                except Exception as e:
                    logger.error(f"Literature agent failed: {e}")
            else:
                logger.warning("use_literature=True but no API key provided")

    def _get_initial_state_fields(self) -> dict:
        """Return initial state fields for the agent."""
        return {
            "current_spectrum": None,
            "pipeline_type": "curve_fitting_unified",
            "is_series": False
        }

    def analyze(
        self,
        data: AnalysisInput,
        system_info: Dict[str, Any] | str | None = None,
        # Curve fitting specific
        objective: str | None = None,
        hints: str | None = None,
        series_metadata: Optional[dict] = None,
        auxiliary_data: Optional[str] = None,
        auxiliary_label: Optional[str] = None,
        # Domain skill
        skill: Optional[str] = None,
        # Prior knowledge from reference analyses
        prior_knowledge: Optional[List[Dict[str, Any]]] = None,
        # Quality control overrides (optional)
        r2_threshold: Optional[float] = None,
        max_model_retries: Optional[int] = None,
        outlier_sigma: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified analysis method - handles single spectra and series identically.
        
        Single spectrum analysis is internally converted to a series of 1.
        For series, the fitting model is locked after the first spectrum.

        Args:
            data: Input data. Can be:
                - str: Single spectrum path (.npy, .csv, .txt)
                - List[str]: Multiple spectrum paths (series)
                - np.ndarray: 1D/2D (single) or 3D (series stack) array
            system_info: Sample/experiment metadata. May include a ``"series"``
                key with series metadata (see below); it will be extracted
                automatically.
            objective: Optional high-level scientific objective that frames
                the entire analysis (e.g., "Determine whether the sample
                underwent a phase transition", "Quantify the relative
                concentration of anatase vs rutile"). Unlike hints which
                guide *how* to analyze, objective specifies *why* you are
                analyzing and *what question* to answer.
            hints: Optional tactical guidance for the analysis (e.g.,
                "Try a Voigt model", "Focus on the band gap region")
            series_metadata: Optional metadata about the series. Can also
                be provided inside ``system_info["series"]``. Explicit
                ``series_metadata`` takes precedence. Expected structure::

                    {
                        "variable": "temperature",   # or "time", "concentration", …
                        "values": [300, 350, 400],      # one value per spectrum, in file order
                        "unit": "K"                      # unit for values
                    }

                When calling through the orchestrator, ``values`` can be a dict
                mapping filenames to values instead of a list. The orchestrator
                sorts files by value for correct physical ordering and converts
                the dict to a list before passing to the agent::

                    {
                        "variable": "temperature",
                        "values": {"spec_5K.csv": 5, "spec_20K.csv": 20, "spec_10K.csv": 10},
                        "unit": "K"
                    }
                    # → files sorted to [spec_5K, spec_10K, spec_20K]
                    # → values converted to [5, 10, 20]
            auxiliary_data: Optional path to an auxiliary dataset (1D curve file
                or 2D image) from the same sample/experiment. Not fitted or
                analyzed in detail, but provided to the LLM as context for
                planning and interpreting the main analysis. Supports .csv,
                .txt, .npy, .png, .jpg, .tif, etc.
            auxiliary_label: Description of the auxiliary data, e.g. "TGA curve
                collected simultaneously during the DSC measurement" or
                "SEM image of sample surface". Defaults to filename stem.
            skill: Optional domain skill name or path to a .md skill file.
                Built-in skills (e.g. "xps") are resolved from the package.
                Custom skills: provide a path to a .md file with sections
                ## planning, ## fitting, ## interpretation, ## validation.
                The skill injects domain-specific guidance at each pipeline
                stage.
            r2_threshold: Override default R² threshold for this analysis
            max_model_retries: Override default max retries for this analysis
            outlier_sigma: Override default outlier sigma for this analysis

        Returns:
            Dict with status, detailed_analysis, scientific_claims,
            model_type, fitting_parameters, fit_quality, output_directory,
            and for series: individual_results, trend_analysis, parameter_evolution,
            flagged_spectra (outliers that may indicate interesting physics)
        
        Examples:
            # Single spectrum
            result = agent.analyze("spectrum.csv")

            # Series with series metadata as a separate argument
            result = agent.analyze(
                ["temp_300K.csv", "temp_350K.csv", "temp_400K.csv"],
                series_metadata={
                    "variable": "temperature",
                    "values": [300, 350, 400],
                    "unit": "K"
                }
            )

            # Series with series metadata inside system_info
            result = agent.analyze(
                ["conc_01.csv", "conc_02.csv", "conc_03.csv"],
                system_info={
                    "sample": "Boron-doped silicon",
                    "instrument": "Raman spectrometer, 532 nm",
                    "series": {
                        "variable": "concentration",
                        "values": [0.1, 0.2, 0.3],
                        "unit": "mol/L"
                    }
                }
            )

            # With relaxed quality threshold
            result = agent.analyze("noisy_spectrum.csv", r2_threshold=0.85)

            # Numpy stack (3D array: n_spectra x 2 x n_points)
            result = agent.analyze(my_spectra_stack)

            # With auxiliary data for context
            result = agent.analyze(
                "dsc_curve.csv",
                auxiliary_data="tga_curve.csv",
                auxiliary_label="TGA curve collected simultaneously during DSC"
            )

            # With domain skill
            result = agent.analyze("xps_ti2p.csv", skill="xps")

            # With custom skill file
            result = agent.analyze("data.csv", skill="/path/to/my_skill.md")
        """
        # Use provided overrides or fall back to instance defaults
        effective_r2_threshold = r2_threshold if r2_threshold is not None else self.r2_threshold
        effective_max_retries = max_model_retries if max_model_retries is not None else self.max_model_retries
        effective_outlier_sigma = outlier_sigma if outlier_sigma is not None else self.outlier_sigma
        
        # Convert DataFrame to numpy array (first two numeric columns)
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include="number")
            if numeric_cols.shape[1] < 2:
                return {
                    "status": "error",
                    "error": {"error": "Invalid DataFrame",
                              "details": f"Expected at least 2 numeric columns, got {numeric_cols.shape[1]}"},
                    "output_directory": str(self.output_dir)
                }
            data = numeric_cols.iloc[:, :2].to_numpy()

        # Parse input
        data_path, data_paths, data_array, error = self._parse_data_input(data)
        
        if error:
            return {
                "status": "error",
                "error": error,
                "output_directory": str(self.output_dir)
            }
        
        # Normalize to internal variables
        spectrum_path = data_path
        spectrum_paths = data_paths
        spectrum_stack = data_array
        
        # Convert single spectrum to series of 1
        if spectrum_path is not None:
            spectrum_paths = [spectrum_path]
            self.logger.info("Single spectrum mode: treating as series of 1")
        
        # Determine input type and count
        if spectrum_stack is not None:
            # Handle numpy array input
            if spectrum_stack.ndim == 1:
                # 1D array: single spectrum y-values
                spectrum_stack = spectrum_stack[np.newaxis, np.newaxis, :]
                self.logger.info("1D array provided, converted to shape (1, 1, n)")
            elif spectrum_stack.ndim == 2:
                # 2D array: single spectrum [x, y] or [y] with multiple points
                if spectrum_stack.shape[0] == 2:
                    # Shape (2, n): single spectrum with x and y
                    spectrum_stack = spectrum_stack[np.newaxis, :, :]
                else:
                    # Shape (n, 2): single spectrum, transpose
                    spectrum_stack = spectrum_stack.T[np.newaxis, :, :]
                self.logger.info(f"2D array provided, converted to shape {spectrum_stack.shape}")
            elif spectrum_stack.ndim != 3:
                return {
                    "status": "error",
                    "error": {"error": "Invalid shape", "details": f"Array must be 1D, 2D, or 3D, got {spectrum_stack.ndim}D"},
                    "output_directory": str(self.output_dir)
                }
            
            num_spectra = spectrum_stack.shape[0]
            input_type = "numpy_array"
        else:
            num_spectra = len(spectrum_paths)
            input_type = "file_paths"
        
        is_single_spectrum = (num_spectra == 1)
        
        self.logger.info("")
        self.logger.info(f"📈 CURVE FITTING ANALYSIS - {num_spectra} spectrum{'s' if num_spectra > 1 else ''}")
        self.logger.info(f"   Quality: R² threshold={effective_r2_threshold}, max_retries={effective_max_retries}")
        if not is_single_spectrum:
            self.logger.info(f"   Outlier detection: {effective_outlier_sigma}σ")
        
        # Load first spectrum for initial analysis
        if spectrum_stack is not None:
            first_spectrum = spectrum_stack[0]
            first_spectrum_name = "spectrum_0000"
        else:
            try:
                first_spectrum = load_curve_data(spectrum_paths[0])
                first_spectrum_name = Path(spectrum_paths[0]).stem
            except Exception as e:
                return {
                    "status": "error",
                    "error": {"error": "Failed to load spectrum", "details": str(e)},
                    "output_directory": str(self.output_dir)
                }
        
        # Optional preprocessing of first spectrum
        processed_first_spectrum = first_spectrum
        first_spectrum_preprocess_quality = None
        if self.preprocessor is not None:
            try:
                processed_first_spectrum, first_spectrum_preprocess_quality = (
                    self.preprocessor.run_preprocessing(
                        first_spectrum, self._handle_system_info(system_info)
                    )
                )
            except Exception as e:
                self.logger.warning(f"Preprocessing failed: {e}, using raw data")
        
        # Generate initial plot
        original_plot_bytes = plot_curve_to_bytes(
            processed_first_spectrum, 
            self._handle_system_info(system_info)
        )
        
        # Compute statistics for first spectrum
        data_statistics = self._compute_statistics(processed_first_spectrum)
        
        # Load auxiliary data if provided
        aux_state = {
            "auxiliary_plot_bytes": None,
            "auxiliary_label": None,
            "auxiliary_summary": None,
            "auxiliary_mime_type": None,
        }
        if auxiliary_data:
            aux_state = self._load_auxiliary_data(auxiliary_data, auxiliary_label)
            if aux_state.get("auxiliary_plot_bytes"):
                self.logger.info(f"   Auxiliary data loaded: {aux_state['auxiliary_label']}")

        # Load skill if provided
        skill_state = {"skill_name": None, "skill_sections": None}
        if skill:
            try:
                parsed = load_skill(skill, domain="curve_fitting")
                skill_state = {"skill_name": parsed["name"], "skill_sections": parsed}
                self.logger.info(f"   Skill loaded: {parsed['name']}")
            except FileNotFoundError:
                self.logger.warning(
                    f"   Skill '{skill}' not found — proceeding without domain skill"
                )

        # Extract series metadata from system_info if not provided explicitly
        handled_system_info = self._handle_system_info(system_info)
        handled_system_info, series_metadata = self._extract_series_metadata(
            handled_system_info, series_metadata
        )

        # Build initial state
        state = {
            # Input data
            "spectrum_paths": spectrum_paths,
            "spectrum_stack": spectrum_stack,
            "input_type": input_type,
            "num_spectra": num_spectra,
            "is_single_spectrum": is_single_spectrum,

            # System info
            "system_info": handled_system_info,
            "series_metadata": series_metadata or {},
            "analysis_hints": hints,
            "analysis_objective": objective,

            # Auxiliary reference data
            **aux_state,

            # Domain skill
            **skill_state,

            # Prior knowledge from reference analyses
            "prior_knowledge": prior_knowledge or [],

            # First spectrum (for planning)
            "data_path": spectrum_paths[0] if spectrum_paths else first_spectrum_name,
            "curve_data": processed_first_spectrum,
            "original_plot_bytes": original_plot_bytes,
            "data_statistics": data_statistics,
            "first_spectrum_preprocessed": first_spectrum_preprocess_quality is not None,
            "first_spectrum_preprocess_quality": first_spectrum_preprocess_quality,

            # Pipeline state
            "analysis_images": [{"label": "First Spectrum", "data": original_plot_bytes}],
            "result_json": {},
            "error_dict": None,
        }
        
        # Create unified pipeline with quality settings
        pipeline = create_unified_curve_fitting_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            plot_fn=plot_curve_to_bytes,
            executor=self.executor,
            output_dir=str(self.output_dir),
            preprocessor=self.preprocessor,
            literature_agent=self.literature_agent,
            enable_human_feedback=self.enable_human_feedback,
            r2_threshold=effective_r2_threshold,
            max_model_retries=effective_max_retries,
            outlier_sigma=effective_outlier_sigma,
            max_verification_iterations=self.max_verification_iterations,
        )
        
        # Execute pipeline
        for i, controller in enumerate(pipeline, 1):
            step_name = controller.__class__.__name__
            self.logger.info(f"\n📍 STEP {i}: {step_name}\n")
            
            try:
                state = controller.execute(state)
                
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at {step_name}: {state['error_dict']}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Pipeline step {step_name} raised exception: {e}")
                state["error_dict"] = {"error": f"Pipeline step failed: {step_name}", "details": str(e)}
                break
        
        # Handle errors
        if state.get("error_dict"):
            return {
                "status": "error",
                "error": state["error_dict"],
                "output_directory": str(self.output_dir)
            }
        
        # Compile results
        final_results = self._compile_results(state)
        
        # Save final results
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, 'w') as f:
            # Make serializable
            serializable = self._make_serializable(final_results)
            json.dump(serializable, f, indent=2, default=str)
        
        self.logger.info("")
        self.logger.info("✅ ANALYSIS COMPLETE")
        self.logger.info(f"   Results: {results_path}")
        if state.get("report_path"):
            self.logger.info(f"   Report: {state['report_path']}")

        flagged = final_results.get("flagged_spectra", [])
        if flagged:
            self.logger.warning(f"   ⚠️ {len(flagged)} spectra flagged for review")
        
        # Log action
        self._log_action(
            action="curve_fit",
            input_ctx={
                "num_spectra": num_spectra,
                "input_type": input_type,
                "series_metadata": series_metadata,
                "r2_threshold": effective_r2_threshold,
            },
            result=final_results.get("summary") if not is_single_spectrum else final_results,
            rationale=f"Model: {final_results.get('model_type', 'unknown')}"
        )
        
        return final_results

    def _compute_statistics(self, curve_data: np.ndarray) -> dict:
        """Compute statistics for a spectrum."""
        if curve_data.ndim == 1:
            x = np.arange(len(curve_data))
            y = curve_data
        elif curve_data.shape[0] == 2:
            x, y = curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            x, y = curve_data[:, 0], curve_data[:, 1]
        else:
            raise ValueError(f"Unexpected data shape: {curve_data.shape}")

        return {
            "n_points": len(x),
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
            "has_nans": bool(np.any(np.isnan(curve_data))),
        }

    def _load_auxiliary_data(
        self, auxiliary_data: str, auxiliary_label: Optional[str]
    ) -> dict:
        """
        Load auxiliary data and return state fields for pipeline injection.

        Supports 1D curve files (.csv, .txt, .dat, .tsv) and images
        (.png, .jpg, .tif, etc.). For .npy files, inspects array shape
        to distinguish curves from images.

        Returns dict with auxiliary_plot_bytes, auxiliary_label,
        auxiliary_summary, and auxiliary_mime_type (all None on failure).
        """
        result = {
            "auxiliary_plot_bytes": None,
            "auxiliary_label": auxiliary_label or Path(auxiliary_data).stem,
            "auxiliary_summary": None,
            "auxiliary_mime_type": None,
        }

        if not os.path.exists(auxiliary_data):
            self.logger.warning(f"Auxiliary data file not found: {auxiliary_data}")
            return result

        ext = Path(auxiliary_data).suffix.lower()
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        curve_extensions = {'.csv', '.txt', '.dat', '.tsv'}

        try:
            is_curve = False
            is_image = False

            if ext == '.npy':
                arr = np.load(auxiliary_data)
                if arr.ndim == 1:
                    is_curve = True
                elif arr.ndim == 2 and min(arr.shape) <= 2:
                    is_curve = True
                else:
                    is_image = True
            elif ext in curve_extensions:
                is_curve = True
            elif ext in image_extensions:
                is_image = True
            else:
                self.logger.warning(
                    f"Unrecognized auxiliary file extension: {ext}"
                )
                return result

            if is_curve:
                if ext == '.npy':
                    curve = np.load(auxiliary_data)
                    if curve.ndim == 1:
                        curve = np.column_stack(
                            [np.arange(len(curve)), curve]
                        )
                    elif curve.shape[0] == 2:
                        curve = curve.T
                    # else shape (n, 2) already correct
                else:
                    curve = load_curve_data(auxiliary_data)
                    if curve.ndim == 2 and curve.shape[0] == 2:
                        curve = curve.T

                # Ensure shape (n, 2) for plotting
                if curve.ndim == 2 and curve.shape[1] == 2:
                    x, y = curve[:, 0], curve[:, 1]
                elif curve.ndim == 2 and curve.shape[0] == 2:
                    x, y = curve[0], curve[1]
                else:
                    x = np.arange(curve.shape[-1])
                    y = curve.flatten()

                result["auxiliary_summary"] = (
                    f"1D curve with {len(x)} points. "
                    f"X range: [{float(np.nanmin(x)):.4g}, {float(np.nanmax(x)):.4g}]. "
                    f"Y range: [{float(np.nanmin(y)):.4g}, {float(np.nanmax(y)):.4g}]."
                )

                plot_info = {"title": result["auxiliary_label"]}
                plot_data = np.column_stack([x, y])
                result["auxiliary_plot_bytes"] = plot_curve_to_bytes(
                    plot_data, plot_info
                )
                result["auxiliary_mime_type"] = "image/png"

            elif is_image:
                from ...tools.image_processor import (
                    load_image,
                    convert_numpy_to_jpeg_bytes,
                )

                img = load_image(auxiliary_data)
                result["auxiliary_summary"] = (
                    f"Image with shape {img.shape} "
                    f"(dtype: {img.dtype})."
                )
                if img.ndim == 3:
                    import cv2
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    result["auxiliary_plot_bytes"] = (
                        convert_numpy_to_jpeg_bytes(img_gray)
                    )
                else:
                    result["auxiliary_plot_bytes"] = (
                        convert_numpy_to_jpeg_bytes(img)
                    )
                result["auxiliary_mime_type"] = "image/jpeg"

        except Exception as e:
            self.logger.warning(f"Failed to load auxiliary data: {e}")

        return result

    def _compile_results(self, state: dict) -> Dict[str, Any]:
        """Compile results into a consistent output structure."""
        is_single = state.get("is_single_spectrum", True)
        num_spectra = state.get("num_spectra", 1)
        series_results = state.get("series_results", [])
        synthesis = state.get("synthesis_result", {})
        flagged_spectra = state.get("flagged_spectra", [])
        
        # Base result structure
        results = {
            "status": "success",
            "output_directory": str(self.output_dir)
        }
        
        if is_single:
            # Single spectrum: backward-compatible structure
            fit_results = state.get("fit_results", {})
            
            results["detailed_analysis"] = synthesis.get("detailed_analysis")
            results["scientific_claims"] = self._validate_scientific_claims(
                synthesis.get("scientific_claims", [])
            )
            results["model_type"] = fit_results.get("model_type")
            results["fitting_parameters"] = fit_results.get("parameters", {})
            results["fit_quality"] = fit_results.get("fit_quality", {})
            results["literature_files"] = state.get("literature_files")
            
            # Include quality warning if present
            if series_results and series_results[0].get("quality_warning"):
                results["quality_warning"] = series_results[0]["quality_warning"]
                results["attempted_models"] = series_results[0].get("attempted_models", [])


        else:
            # Series: full structure with trends and flagged spectra
            successful = sum(1 for r in series_results if r.get("success", False))
            
            refit_summary = state.get("refit_summary", [])
            results["summary"] = {
                "total_spectra": num_spectra,
                "successful_fits": successful,
                "flagged_count": len(flagged_spectra),
                "refitted_count": sum(1 for r in refit_summary if r.get("improved")),
                "input_type": state.get("input_type"),
                "locked_model": state.get("locked_fitting_config", {}).get("physical_model"),
                "is_single_spectrum": False
            }
            
            results["detailed_analysis"] = synthesis.get("detailed_analysis", "")
            results["scientific_claims"] = self._validate_scientific_claims(
                synthesis.get("scientific_claims", [])
            )
            
            # Series-specific results
            results["individual_results"] = [
                {
                    "index": r["index"],
                    "name": r["name"],
                    "success": r["success"],
                    "model_type": r.get("model_type"),
                    "parameters": r.get("parameters", {}),
                    "fit_quality": r.get("fit_quality", {}),
                    "visualization_path": r.get("visualization_path"),
                    "error": r.get("error"),
                    "flagged": r.get("flagged", False),
                    "flag_reason": r.get("flag_reason"),
                    "flag_recommendation": r.get("flag_recommendation"),
                    "adaptively_refitted": r.get("adaptively_refitted", False),
                    "original_r2": r.get("original_r2"),
                    "locked_model_type": r.get("locked_model_type"),
                }
                for r in series_results
            ]

            results["flagged_spectra"] = flagged_spectra
            results["flagged_spectra_analysis"] = synthesis.get("flagged_spectra_analysis", {})
            results["refit_summary"] = refit_summary
            results["refit_analysis"] = synthesis.get("refit_analysis", {})
            results["trend_analysis"] = state.get("trend_analysis_results", {})
            results["parameter_trends"] = synthesis.get("parameter_trends", {})
            results["caveats"] = synthesis.get("caveats", "")
            results["literature_files"] = state.get("literature_files")
            results["locked_fitting_config"] = state.get("locked_fitting_config")
        
        return results

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, bytes):
            return None  # Skip bytes
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    # =========================================================================
    # BACKWARD COMPATIBLE METHODS
    # =========================================================================

    def analyze_spectrum_series(
        self,
        spectrum_paths: Optional[List[str]] = None,
        spectrum_stack: Optional[np.ndarray] = None,
        system_info: Optional[Union[dict, str]] = None,
        series_metadata: Optional[dict] = None,
        objective: str | None = None,
        hints: str | None = None,
    ) -> Dict[str, Any]:
        """
        Analyze a series of spectra.

        BACKWARD COMPATIBLE: Delegates to unified analyze() method.

        Args:
            spectrum_paths: List of file paths to spectra
            spectrum_stack: 3D numpy array (n_spectra x 2 x n_points)
            system_info: System/sample metadata
            series_metadata: Dict with ``"variable"`` (str), ``"values"``
                (list), and ``"unit"`` (str) describing the independent
                variable across the series
            objective: High-level scientific objective
            hints: Analysis guidance

        Returns:
            Analysis results dictionary
        """
        if spectrum_paths is not None:
            return self.analyze(
                spectrum_paths,
                system_info=system_info,
                series_metadata=series_metadata,
                hints=hints,
                objective=objective,
            )
        elif spectrum_stack is not None:
            return self.analyze(
                spectrum_stack,
                system_info=system_info,
                series_metadata=series_metadata,
                hints=hints,
                objective=objective,
            )
        else:
            return {
                "status": "error",
                "error": {"error": "No input", "details": "Must provide spectrum_paths or spectrum_stack"},
                "output_directory": str(self.output_dir)
            }

    def _get_claims_instruction_prompt(self) -> str:
        return FITTING_INTERPRETATION_INSTRUCTIONS

    def _get_measurement_recommendations_prompt(self) -> str:
        return CURVE_FITTING_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS