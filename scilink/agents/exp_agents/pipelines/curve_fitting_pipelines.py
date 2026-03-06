# pipelines/curve_fitting_pipelines.py

"""
Unified Curve Fitting Pipeline Factory

Factory functions for creating curve fitting pipelines.
All analysis now uses a single unified pipeline that handles both
single spectra (n=1) and series (n>1) identically.

Key principle: Single spectrum = Series of 1

Quality control features:
- Automatic model retry when R² is inadequate
- Statistical outlier detection for series
- Human feedback integration for unresolved quality issues
"""

import logging
from typing import Callable, List, Any

from ..controllers.curve_fitting_controllers import (
    # Original controllers
    AnalyzeDataController,
    SeriesScoutController,
    LiteratureSearchController,
    GenerateCurveFittingReportController,
    # Unified controllers for series support
    HumanFeedbackRefinementController,
    UnifiedSeriesProcessingController,
    AdaptiveRefitController,
    ConditionalTrendAnalysisController,
    UnifiedCurveSynthesisController,
    UnifiedCurveReportController,
)
from ..controllers.base_controllers import (
    StoreAnalysisResultsController,
)
from ..instruct import (
    CURVE_ANALYSIS_INSTRUCTIONS,
    FITTING_SCRIPT_INSTRUCTIONS,
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
    FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
    FITTING_INTERPRETATION_INSTRUCTIONS,
    PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
)


def create_unified_curve_fitting_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    parse_fn: Callable,
    store_fn: Callable,
    plot_fn: Callable,
    executor: Any,
    output_dir: str,
    preprocessor: Any | None = None,
    literature_agent: Any | None = None,
    enable_human_feedback: bool = False,
    r2_threshold: float = 0.95,
    max_model_retries: int = 3,
    outlier_sigma: float = 2.0,
    max_verification_iterations: int = 3,
) -> List:
    """
    Factory function to create the unified curve fitting pipeline.
    
    This pipeline handles BOTH single spectra and series with quality control:

    1. Analyze First Spectrum Data
       - Compute statistics, create initial visualization

    2. Human Feedback Refinement (optional)
       - LLM plans fitting approach
       - Human can refine the plan
       - Configuration is LOCKED for series processing

    3. Literature Search (if enabled)
       - Search for relevant fitting models
       - Runs only once (on first spectrum context)

    4. Unified Series Processing with Quality Control
       - Fits ALL spectra using locked configuration
       - Single spectrum = series of 1
       - LLM verification loop (n iterations) to catch and fix issues
       - Human feedback for additional refinement (if enabled)
       - Automatic model retry if R² below threshold
       - Statistical outlier detection for series

    5. Adaptive Refit (automatic)
       - Re-analyzes flagged spectra independently with full quality control
       - Uses different models when the locked config fails
       - Skipped for single spectrum or when no spectra are flagged

    6. Conditional Trend Analysis
       - For n>=2: Generates and executes trend analysis
       - Highlights flagged spectra in visualizations
       - For n=1: Skipped

    7. Synthesis
       - For n>=2: Cross-spectrum synthesis with outlier analysis
       - For n=1: Single-spectrum interpretation

    8. Store Results
       - Save analysis images and artifacts

    9. Report Generation
       - Adapts format based on single vs series
       - Includes flagged spectra and refit sections for series
    
    Args:
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation configuration
        safety_settings: LLM safety settings
        parse_fn: Function to parse LLM responses
        store_fn: Function to store analysis images
        plot_fn: Function to plot curve data
        executor: Script executor instance
        output_dir: Output directory path
        preprocessor: Optional preprocessor agent
        literature_agent: Optional literature search agent
        enable_human_feedback: Enable human-in-the-loop refinement
        r2_threshold: Minimum acceptable R² value (default: 0.95)
        max_model_retries: Max alternative models to try if R² inadequate (default: 3)
        outlier_sigma: Sigma threshold for outlier detection in series (default: 2.0)
        max_verification_iterations: Max LLM verification iterations for first spectrum (default: 3)
    
    Returns:
        List of controller instances to execute in sequence
    """
    pipeline = []

    # Step 1: Analyze first spectrum data (compute stats, initial plot)
    pipeline.append(AnalyzeDataController(logger, plot_fn))

    # Step 1.5: Scout representative spectra for series planning
    pipeline.append(
        SeriesScoutController(
            logger=logger,
            plot_fn=plot_fn,
        )
    )

    # Step 2: Human feedback refinement on fitting approach
    pipeline.append(
        HumanFeedbackRefinementController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            instructions=CURVE_ANALYSIS_INSTRUCTIONS,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
            max_iterations=5
        )
    )

    # Step 3: Literature search (runs once, uses first spectrum context)
    pipeline.append(
        LiteratureSearchController(
            logger=logger,
            literature_agent=literature_agent,
            output_dir=output_dir
        )
    )

    # Step 4: Unified series processing with quality control
    pipeline.append(
        UnifiedSeriesProcessingController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            script_instructions=FITTING_SCRIPT_INSTRUCTIONS,
            correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
            quality_instructions=FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
            output_dir=output_dir,
            plot_fn=plot_fn,
            r2_threshold=r2_threshold,
            max_model_retries=max_model_retries,
            enable_human_feedback=enable_human_feedback,
            outlier_sigma=outlier_sigma,
            max_verification_iterations=max_verification_iterations,
            preprocessor=preprocessor,
            conformance_instructions=PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
        )
    )

    # Step 5: Adaptive refit of flagged spectra (post-processing recovery)
    pipeline.append(
        AdaptiveRefitController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            script_instructions=FITTING_SCRIPT_INSTRUCTIONS,
            correction_instructions=FITTING_SCRIPT_CORRECTION_INSTRUCTIONS,
            quality_instructions=FIT_QUALITY_ASSESSMENT_INSTRUCTIONS,
            output_dir=output_dir,
            plot_fn=plot_fn,
            r2_threshold=r2_threshold,
            max_model_retries=max_model_retries,
            max_verification_iterations=max_verification_iterations,
            preprocessor=preprocessor,
            enable_human_feedback=enable_human_feedback,
            conformance_instructions=PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
        )
    )

    # Step 6: Conditional trend analysis (only for n>=2)
    pipeline.append(
        ConditionalTrendAnalysisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            output_dir=output_dir,
            max_corrections=3
        )
    )

    # Step 7: Synthesis (adapts to single vs series)
    pipeline.append(
        UnifiedCurveSynthesisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            single_spectrum_instructions=FITTING_INTERPRETATION_INSTRUCTIONS,
            output_dir=output_dir
        )
    )

    # Step 8: Store analysis results/images
    pipeline.append(
        StoreAnalysisResultsController(logger, store_fn)
    )

    # Step 9a: Single spectrum report
    pipeline.append(
        GenerateCurveFittingReportController(logger, output_dir, r2_threshold=r2_threshold)
    )

    # Step 9b: Series report (only generates for n>=2)
    pipeline.append(
        UnifiedCurveReportController(logger, output_dir)
    )

    logger.info(f"Unified curve fitting pipeline created: {len(pipeline)} steps")
    logger.info(f"  Quality settings: R² threshold={r2_threshold}, max_retries={max_model_retries}, outlier_sigma={outlier_sigma}")
    logger.info(f"  Verification iterations: {max_verification_iterations}")
    
    return pipeline


# =============================================================================
# LEGACY PIPELINE FACTORY (for backward compatibility)
# =============================================================================

def create_curve_fitting_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    parse_fn: Callable,
    store_fn: Callable,
    plot_fn: Callable,
    executor: Any,
    output_dir: str,
    preprocessor: Any | None = None,
    literature_agent: Any | None = None,
    enable_human_feedback: bool = False,
    settings: dict | None = None,  # Deprecated
    r2_threshold: float = 0.95,
    max_model_retries: int = 3,
    outlier_sigma: float = 2.0,
    max_verification_iterations: int = 3,
) -> List:
    """
    BACKWARD COMPATIBLE: Creates curve fitting pipeline.
    
    Now returns the unified pipeline that handles both single spectra
    and series analysis with quality control.
    
    For explicit series analysis, use create_unified_curve_fitting_pipeline().
    """
    if settings is not None:
        import warnings
        warnings.warn(
            "The 'settings' parameter is deprecated and will be ignored.",
            DeprecationWarning
        )
    
    return create_unified_curve_fitting_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        parse_fn=parse_fn,
        store_fn=store_fn,
        plot_fn=plot_fn,
        executor=executor,
        output_dir=output_dir,
        preprocessor=preprocessor,
        literature_agent=literature_agent,
        enable_human_feedback=enable_human_feedback,
        r2_threshold=r2_threshold,
        max_model_retries=max_model_retries,
        outlier_sigma=outlier_sigma,
        max_verification_iterations=max_verification_iterations,
    )