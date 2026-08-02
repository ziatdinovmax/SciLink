from ..controllers.atomistic_microscopy_controllers import (
    RunAtomDetectionController,
    RunIntensityAnalysisController,
    GetIntensityGMMParamsController,
    RunIntensityGMMController,
    GetLocalEnvParamsController,
    RunLocalEnvAnalysisController,
    RunNNAnalysisController,
    BuildAtomisticPromptController
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController
)
from typing import Callable, List
import logging

def create_atomistic_pipeline(
    model, 
    logger: logging.Logger, 
    generation_config, 
    safety_settings, 
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    Assembles the full, multi-step pipeline for the AtomisticMicroscopyAnalysisAgent.
    """
    
    pipeline = []

    # 1. 🛠️ Tool: Run NN ensemble to find atoms
    pipeline.append(RunAtomDetectionController(logger, settings))

    # 2. 🛠️ Tool: Extract intensities and create histogram
    pipeline.append(RunIntensityAnalysisController(logger, settings))

    # 3. 🧠 LLM: Select n_components for intensity GMM
    pipeline.append(GetIntensityGMMParamsController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 4. 🛠️ Tool: Run 1D intensity GMM and create plots
    pipeline.append(RunIntensityGMMController(logger, settings))

    # 5. 🧠 LLM: Select n_components for local env GMM
    pipeline.append(GetLocalEnvParamsController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 6. 🛠️ Tool: Run local environment analysis (atomai.stat.imlocal)
    # This step is non-critical and will not halt the pipeline on failure.
    pipeline.append(RunLocalEnvAnalysisController(logger, settings))

    # 7. 🛠️ Tool: Run nearest-neighbor distance analysis
    # This step is non-critical and will not halt the pipeline on failure.
    pipeline.append(RunNNAnalysisController(logger, settings))

    # 8. 📝 Prep: Combine all results into a final prompt
    pipeline.append(BuildAtomisticPromptController(logger, settings))

    # 9. 🧠 LLM: Run final interpretation (Reusable)
    pipeline.append(RunFinalInterpretationController(
        model, logger, generation_config, safety_settings, parse_fn
    ))

    # 10. 🛠️ Tool: Store final images for feedback (Reusable)
    pipeline.append(StoreAnalysisResultsController(logger, store_fn))
    
    logger.info(f"Atomistic pipeline created with {len(pipeline)} steps.")
    return pipeline