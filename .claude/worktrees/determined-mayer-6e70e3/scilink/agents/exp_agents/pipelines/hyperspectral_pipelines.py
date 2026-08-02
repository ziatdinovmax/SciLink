import logging
from typing import Callable, List
from ..controllers.hyperspectral_controllers import (
    RunPreprocessingController,
    GetInitialComponentParamsController,
    RunComponentTestLoopController,
    CreateElbowPlotController,
    GetFinalComponentSelectionController,
    RunFinalSpectralUnmixingController,
    CreateAnalysisPlotsController,
    BuildHyperspectralPromptController,
    RunDynamicAnalysisController,
    SelectRefinementTargetController,
    GenerateRefinementTasksController,
    BuildHolisticSynthesisPromptController,
    GenerateHTMLReportController,
    RunSelfReflectionController,      
    ApplyReflectionUpdatesController   
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController,
    IterativeFeedbackController
)
from ..preprocess import HyperspectralPreprocessingAgent

from ..instruct import SPECTROSCOPY_REFINEMENT_INSTRUCTIONS

def create_hyperspectral_iteration_pipeline(
    model,
    logger: logging.Logger,
    generation_config,
    safety_settings,
    settings: dict,
    preprocessor: HyperspectralPreprocessingAgent,
    parse_fn: Callable
) -> List:
    """
    Assembles the pipeline that runs *per-iteration* of the recursive analysis.
    This includes: NMF -> Plotting -> Refinement Decision -> Data Slicing.
    """

    pipeline = []

    # --- 1. PREPROCESSING (Only on first iteration) ---
    if settings.get('run_preprocessing', True):
        pipeline.append(RunPreprocessingController(logger, preprocessor))

    # --- 2. SPECTRAL UNMIXING ---
    if settings.get('enabled', True):

        # 2a. Auto-component workflow
        if settings.get('auto_components', True):
            # [🧠 LLM] Get initial component guess
            pipeline.append(GetInitialComponentParamsController(
                model, logger, generation_config, safety_settings, parse_fn
            ))
            # [🛠️ Tool] Run NMF loop to get errors
            pipeline.append(RunComponentTestLoopController(logger, settings))
            # [🛠️ Tool] Create elbow plot from errors
            pipeline.append(CreateElbowPlotController(logger, settings))
            # [🛠️ LLM] Select final n_components from elbow plot
            pipeline.append(GetFinalComponentSelectionController(
                model, logger, generation_config, safety_settings, parse_fn
            ))

        # 2b. [🛠️ Tool] Run final NMF
        pipeline.append(RunFinalSpectralUnmixingController(logger, settings))

        # 2c. [🛠️ Tool] Create all plots for analysis
        pipeline.append(CreateAnalysisPlotsController(logger, settings))

    # --- 3. ITERATION ANALYSIS & REFINEMENT DECISION ---

    # 3a. [📝 Prep] Build the prompt for *this iteration*
    pipeline.append(BuildHyperspectralPromptController(logger))

    # 3b. [🧠 LLM] Run interpretation for *this iteration*
    pipeline.append(RunFinalInterpretationController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 3c. [🧠 LLM] Decide if we need to zoom (Draft the plan)
    pipeline.append(SelectRefinementTargetController(
        model, logger, generation_config, safety_settings, parse_fn
    ))
    
    # 3d. [🧠/👤 User] FEEDBACK STEP
    pipeline.append(IterativeFeedbackController(
        model, logger, generation_config, safety_settings, 
        parse_fn, settings, refinement_instruction=SPECTROSCOPY_REFINEMENT_INSTRUCTIONS
    ))

    # 3e. [🧠/💻] Dynamic Analysis
    pipeline.append(RunDynamicAnalysisController(
        model, logger, generation_config, safety_settings, parse_fn
    ))
    
    # 3f. [🛠️ Tool] Prepare data for next loop (Standard NMF tasks)
    pipeline.append(GenerateRefinementTasksController(logger))

    logger.info(f"Hyperspectral *iteration* pipeline created with {len(pipeline)} steps.")
    return pipeline

def create_hyperspectral_synthesis_pipeline(
    model,  
    logger: logging.Logger,  
    generation_config,  
    safety_settings,  
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:

    pipeline = []

    # 1. [📝 Prep] Build the holistic synthesis prompt
    pipeline.append(BuildHolisticSynthesisPromptController(logger))

    # 2. [🧠 LLM] Run final synthesis interpretation (DRAFT 1)
    pipeline.append(RunFinalInterpretationController(
        model, logger, generation_config, safety_settings, parse_fn
    ))
    
    # 3. [🧠 Critic] Review for hallucinations/overfitting
    pipeline.append(RunSelfReflectionController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 4. [🧠 Editor] Apply fixes if needed
    pipeline.append(ApplyReflectionUpdatesController(
        model, logger, generation_config, safety_settings, parse_fn
    ))
    
    # 5. [📄 Report] Generate HTML Report
    pipeline.append(GenerateHTMLReportController(logger, settings))

    # 6. [🛠️ Tool] Store all images
    pipeline.append(StoreAnalysisResultsController(logger, store_fn))
    
    logger.info(f"Hyperspectral *synthesis* pipeline created with {len(pipeline)} steps.")
    return pipeline