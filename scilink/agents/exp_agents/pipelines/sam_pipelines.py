from ..controllers.sam_controllers import (
    RunSAMRefinementLoopController,
    CalculateSAMStatsController,
    BuildSAMPromptController
)
from ..controllers.base_controllers import (
    RunFinalInterpretationController,
    StoreAnalysisResultsController
)
from typing import Callable, List

def create_sam_pipeline(
    model, 
    logger, 
    generation_config, 
    safety_settings, 
    settings: dict,
    parse_fn: Callable,
    store_fn: Callable
) -> List:
    """
    Assembles and returns a list of controller instances 
    for the standard SAM analysis pipeline.
    """
    
    pipeline = []

    # 1. 🛠️/🧠 Tool/LLM Step (This controller runs the whole loop)
    pipeline.append(
        RunSAMRefinementLoopController(
            model, logger, generation_config, safety_settings, settings, parse_fn
        )
    )
    
    # 2. 🛠️ Tool Step (Calculate final stats)
    pipeline.append(
        CalculateSAMStatsController(logger)
    )

    # 3. 📝 Prep Step (Builds prompt)
    pipeline.append(
        BuildSAMPromptController(logger)
    )
    
    # 4. 🧠 LLM Step (Interpret)
    pipeline.append(
        RunFinalInterpretationController(
            model, logger, generation_config, safety_settings, parse_fn
        )
    )
    
    # 5. 🛠️ Tool Step (Store results)
    pipeline.append(
        StoreAnalysisResultsController(logger, store_fn)
    )
    
    return pipeline