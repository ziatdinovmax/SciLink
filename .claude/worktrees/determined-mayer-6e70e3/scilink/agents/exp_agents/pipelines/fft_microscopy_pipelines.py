"""
Microscopy Analysis Pipelines - Unified Architecture

Factory functions for creating microscopy analysis pipelines.
All analysis now uses a single unified pipeline that handles both
single images (n=1) and batches (n>1) identically.
"""

import logging
from typing import Callable, List, Optional

from ..controllers.fft_microscopy_controllers import (
    # Unified pipeline controllers
    InitialFFTAnalysisController,
    HumanFeedbackRefinementController,
    UnifiedBatchProcessingController,
    ConditionalCustomAnalysisController,
    UnifiedSynthesisController,
    UnifiedReportGenerationController,
)


def create_unified_microscopy_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable,
    preset_params: Optional[dict] = None
) -> List:
    """
    Factory function to create the unified microscopy analysis pipeline.
    
    This pipeline handles BOTH single images and batches:
    
    1. Initial FFT Analysis (on first image)
       - Gets LLM parameter suggestions
       - Runs FFT/NMF on first frame
       - Skipped if preset_params provided
       
    2. Human Feedback Refinement (optional)
       - Refines FFT/NMF parameters on the first image
       - Skipped if enable_human_feedback=False or preset_params provided
       
    3. Batch Processing
       - Processes ALL images (including single images as n=1)
       - Caches FFT/NMF analyzer for efficiency
       
    4. Conditional Custom Analysis
       - For n>=2: Generates and executes trend analysis script
       - For n=1: Skipped (no trends to analyze)
       
    5. Synthesis
       - For n>=2: Cross-image synthesis of findings
       - For n=1: Single-image scientific interpretation
       
    6. Report Generation
       - Generates HTML report and JSON summary
       - Adapts format based on single vs batch
    
    Args:
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation configuration
        safety_settings: LLM safety settings
        settings: Pipeline settings dict
        parse_fn: Function to parse LLM responses
        store_fn: Function to store analysis images
        preset_params: If provided, skip initial analysis and use these params
    
    Returns:
        List of controller instances to execute in sequence
    """
    pipeline = []
    
    # Step 1: Initial FFT Analysis (skip if preset_params)
    if preset_params is None:
        pipeline.append(
            InitialFFTAnalysisController(
                model=model,
                logger=logger,
                generation_config=generation_config,
                safety_settings=safety_settings,
                settings=settings
            )
        )
    else:
        # Inject preset params directly
        pipeline.append(
            _PresetParamsController(preset_params, logger)
        )
    
    # Step 2: Human feedback refinement on first image
    pipeline.append(
        HumanFeedbackRefinementController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        )
    )
    
    # Step 3: Process all images with refined parameters
    pipeline.append(
        UnifiedBatchProcessingController(
            logger=logger,
            settings=settings
        )
    )
    
    # Step 4: Custom analysis script (conditional on n>=2)
    pipeline.append(
        ConditionalCustomAnalysisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        )
    )
    
    # Step 5: Scientific synthesis
    pipeline.append(
        UnifiedSynthesisController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings,
            store_fn=store_fn
        )
    )
    
    # Step 6: Report generation
    pipeline.append(
        UnifiedReportGenerationController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            settings=settings
        )
    )
    
    return pipeline


class _PresetParamsController:
    """Injects preset parameters without LLM estimation."""
    
    def __init__(self, params: dict, logger: logging.Logger):
        self.params = params
        self.logger = logger
    
    def execute(self, state: dict) -> dict:
        self.logger.info("📋 Using preset parameters (skipping LLM estimation)")
        self.logger.info(f"   Parameters: {self.params}")
        
        state["locked_params"] = self.params
        state["llm_params"] = self.params
        state["first_frame_results"] = {"llm_params": self.params}
        state["current_params"] = self.params
        
        return state


# =============================================================================
# LEGACY PIPELINE FACTORIES (for backward compatibility)
# =============================================================================

def create_fftnmf_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    DEPRECATED: Use create_unified_microscopy_pipeline instead.
    
    This factory is preserved for backward compatibility but now
    returns the unified pipeline.
    """
    logger.warning(
        "create_fftnmf_pipeline() is deprecated. "
        "Use create_unified_microscopy_pipeline() instead."
    )
    return create_unified_microscopy_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        settings=settings,
        parse_fn=parse_fn,
        store_fn=store_fn
    )


def create_series_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable,
    feedback_callback=None
) -> List:
    """
    DEPRECATED: Use create_unified_microscopy_pipeline instead.
    
    This factory is preserved for backward compatibility.
    """
    logger.warning(
        "create_series_pipeline() is deprecated. "
        "Use create_unified_microscopy_pipeline() instead."
    )
    return create_unified_microscopy_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        settings=settings,
        parse_fn=parse_fn,
        store_fn=lambda *args, **kwargs: None
    )


def create_batch_only_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable,
    locked_params: dict
) -> List:
    """
    DEPRECATED: Use create_unified_microscopy_pipeline with preset_params instead.
    """
    logger.warning(
        "create_batch_only_pipeline() is deprecated. "
        "Use create_unified_microscopy_pipeline(preset_params=...) instead."
    )
    return create_unified_microscopy_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        settings=settings,
        parse_fn=parse_fn,
        store_fn=lambda *args, **kwargs: None,
        preset_params=locked_params
    )


def create_quick_series_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    parse_fn: Callable
) -> List:
    """
    DEPRECATED: Use create_unified_microscopy_pipeline instead.
    
    For quick processing, disable human feedback in settings.
    """
    logger.warning(
        "create_quick_series_pipeline() is deprecated. "
        "Use create_unified_microscopy_pipeline() with enable_human_feedback=False."
    )
    # Temporarily disable feedback
    settings_copy = settings.copy()
    settings_copy['enable_human_feedback'] = False
    
    return create_unified_microscopy_pipeline(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        settings=settings_copy,
        parse_fn=parse_fn,
        store_fn=lambda *args, **kwargs: None
    )