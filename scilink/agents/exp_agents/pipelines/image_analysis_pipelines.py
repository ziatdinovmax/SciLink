# pipelines/image_analysis_pipelines.py

"""
Unified Image Analysis Pipeline Factory

Factory function for creating image analysis pipelines.
All analysis uses a single unified pipeline that handles both
single images (n=1) and image series (n>1) identically.

Key principle: Single image = Series of 1

Quality control features:
- LLM-driven quality assessment (no single numeric threshold)
- Statistical outlier detection for series
- Automatic approach retry when quality is inadequate
- Human feedback integration for unresolved quality issues
"""

import logging
from typing import Callable, List, Any

from ..controllers.image_analysis_controllers import (
    AnalyzeImageController,
    ImageSeriesScoutController,
    SkillSuggestionController,
    ImagePlanningController,
    LiteratureSearchController,
    UnifiedImageProcessingController,
    ImageAdaptiveRefitController,
    ConditionalImageTrendController,
    UnifiedImageSynthesisController,
    GenerateImageReportController,
)
from ..controllers.base_controllers import (
    StoreAnalysisResultsController,
)
from ..instruct import (
    IMAGE_ANALYSIS_PLANNING_INSTRUCTIONS,
    IMAGE_ANALYSIS_SCRIPT_INSTRUCTIONS,
    IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
    IMAGE_ANALYSIS_SCRIPT_REFINEMENT_PROMPT,
    IMAGE_ANALYSIS_QUALITY_ASSESSMENT_INSTRUCTIONS,
    IMAGE_ANALYSIS_INTERPRETATION_INSTRUCTIONS,
    IMAGE_ANALYSIS_PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
)


def create_unified_image_analysis_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    parse_fn: Callable,
    store_fn: Callable,
    load_skills_fn: Callable,
    image_to_bytes_fn: Callable,
    montage_fn: Callable,
    executor: Any,
    output_dir: str,
    literature_agent: Any | None = None,
    enable_human_feedback: bool = False,
    outlier_sigma: float = 2.0,
    max_verification_iterations: int = 7,
    num_plan_candidates: int = 1,
) -> List:
    """
    Factory function to create the unified image analysis pipeline.

    This pipeline handles BOTH single images and series with quality control:

    1. Analyze First Image
       - Compute statistics, create thumbnail for LLM

    2. Series Scout (for n>1)
       - Sample representative images, create montage

    3. Planning with Human Feedback (optional)
       - LLM plans analysis approach
       - Human can refine the plan
       - Configuration is LOCKED for series processing

    4. Literature Search (if enabled)
       - Search for relevant analysis techniques
       - Runs only once

    5. Unified Series Processing with Quality Control
       - Processes ALL images using locked configuration
       - Single image = series of 1
       - LLM verification loop to catch and fix issues
       - Automatic approach retry if quality inadequate
       - Statistical outlier detection for series

    6. Adaptive Refit (automatic)
       - Re-analyzes flagged images independently
       - Uses different approaches when the locked config fails
       - Skipped for single image or when no images are flagged

    7. Conditional Trend Analysis
       - For n>=2: Generates and executes feature trend analysis
       - For n=1: Skipped

    8. Synthesis
       - For n>=2: Cross-image synthesis with outlier analysis
       - For n=1: Single-image interpretation

    9. Store Results + Report Generation

    Args:
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation configuration
        safety_settings: LLM safety settings
        parse_fn: Function to parse LLM responses
        store_fn: Function to store analysis images
        image_to_bytes_fn: Function to convert numpy image to JPEG bytes
        montage_fn: Function to create labeled image montage
        executor: Script executor instance
        output_dir: Output directory path
        literature_agent: Optional literature search agent
        enable_human_feedback: Enable human-in-the-loop refinement
        outlier_sigma: Sigma threshold for outlier detection in series (default: 2.0)
        max_verification_iterations: Max LLM verification iterations (default: 7)

    Returns:
        List of controller instances to execute in sequence
    """
    pipeline = []

    # Step 1: Analyze first image (compute stats, create thumbnail)
    pipeline.append(AnalyzeImageController(logger, image_to_bytes_fn))

    # Step 2: Scout representative images for series planning
    pipeline.append(
        ImageSeriesScoutController(
            logger=logger,
            image_to_bytes_fn=image_to_bytes_fn,
            montage_fn=montage_fn,
        )
    )

    # Step 3: Auto-suggest a domain skill if none was provided
    pipeline.append(
        SkillSuggestionController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            load_skills_fn=load_skills_fn,
        )
    )

    # Step 4: LLM planning with optional human feedback
    planning_controller = ImagePlanningController(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        parse_fn=parse_fn,
        instructions=IMAGE_ANALYSIS_PLANNING_INSTRUCTIONS,
        output_dir=output_dir,
        enable_human_feedback=enable_human_feedback,
        max_iterations=5,
        num_plan_candidates=num_plan_candidates,
    )
    pipeline.append(planning_controller)

    # Step 5: Literature search (runs once)
    pipeline.append(
        LiteratureSearchController(
            logger=logger,
            literature_agent=literature_agent,
            output_dir=output_dir,
        )
    )

    # Step 6: Unified series processing with quality control
    pipeline.append(
        UnifiedImageProcessingController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            script_instructions=IMAGE_ANALYSIS_SCRIPT_INSTRUCTIONS,
            correction_instructions=IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
            quality_instructions=IMAGE_ANALYSIS_QUALITY_ASSESSMENT_INSTRUCTIONS,
            output_dir=output_dir,
            image_to_bytes_fn=image_to_bytes_fn,
            enable_human_feedback=enable_human_feedback,
            outlier_sigma=outlier_sigma,
            max_verification_iterations=max_verification_iterations,
            conformance_instructions=IMAGE_ANALYSIS_PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
            refinement_instructions=IMAGE_ANALYSIS_SCRIPT_REFINEMENT_PROMPT,
            # Lets each best-of-N fan-out candidate (>=1) plan its own
            # independent approach instead of sharing the locked plan.
            replanner=planning_controller,
        )
    )

    # Step 7: Adaptive refit of flagged images
    pipeline.append(
        ImageAdaptiveRefitController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            script_instructions=IMAGE_ANALYSIS_SCRIPT_INSTRUCTIONS,
            correction_instructions=IMAGE_ANALYSIS_SCRIPT_CORRECTION_INSTRUCTIONS,
            quality_instructions=IMAGE_ANALYSIS_QUALITY_ASSESSMENT_INSTRUCTIONS,
            output_dir=output_dir,
            image_to_bytes_fn=image_to_bytes_fn,
            max_verification_iterations=max_verification_iterations,
            enable_human_feedback=enable_human_feedback,
            conformance_instructions=IMAGE_ANALYSIS_PLAN_CONFORMANCE_CHECK_INSTRUCTIONS,
            refinement_instructions=IMAGE_ANALYSIS_SCRIPT_REFINEMENT_PROMPT,
        )
    )

    # Step 8: Conditional trend analysis (only for n>=2)
    pipeline.append(
        ConditionalImageTrendController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            output_dir=output_dir,
            max_corrections=3,
        )
    )

    # Step 9: Synthesis (adapts to single vs series)
    pipeline.append(
        UnifiedImageSynthesisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            single_image_instructions=IMAGE_ANALYSIS_INTERPRETATION_INSTRUCTIONS,
            output_dir=output_dir,
        )
    )

    # Step 10: Store analysis results/images
    pipeline.append(
        StoreAnalysisResultsController(logger, store_fn)
    )

    # Step 11: Report generation (adapts to single vs series)
    pipeline.append(
        GenerateImageReportController(logger, output_dir)
    )

    logger.info(f"Unified image analysis pipeline created: {len(pipeline)} steps")
    logger.info(f"  Outlier sigma: {outlier_sigma}")
    logger.info(f"  Verification iterations: {max_verification_iterations}")

    return pipeline
