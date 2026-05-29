import logging
from typing import Callable, List
from ..controllers.hyperspectral_controllers import (
    DecompositionController,
    BuildHyperspectralPromptController,
    RunDynamicAnalysisController,
    SelectRefinementTargetController,
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
    parse_fn: Callable,
    executor_timeout: int = 600,
) -> List:
    """
    Assembles the pipeline that runs *per-iteration* of the recursive analysis.
    This includes: NMF -> Plotting -> Refinement Decision -> Data Slicing.
    """

    pipeline = []

    # --- SPECTRAL DECOMPOSITION (one cohesive, model-bearing step) ---
    # The decomposition is a single composite controller that owns its own input
    # prep, component estimation, elbow scan, LLM selection, final unmixing, and
    # validation plots, and leaves the decomposition contract in `state`. It
    # cleans the cube *for decomposition* internally; the per-pixel codegen
    # downstream receives the RAW cube and owns its own fittability denoising.
    # See issue #220 / docs/hyperspectral_codegen_relocation.md.
    if settings.get('enabled', True):
        pipeline.append(DecompositionController(
            model, logger, generation_config, safety_settings,
            settings, preprocessor, parse_fn
        ))

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
        model, logger, generation_config, safety_settings, parse_fn,
        executor_timeout=executor_timeout,
    ))

    logger.info(f"Hyperspectral iteration pipeline created with {len(pipeline)} steps.")
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