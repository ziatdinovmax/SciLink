"""
Central registry for all available analysis pipelines.

This module maintains a catalog of all pipelines that can be used
by agents. Each pipeline is registered with metadata including:
- A unique ID
- A human-readable description for the LLM
- A factory function to create the pipeline
- The instruction prompt to use with this pipeline
"""

from typing import Dict, Callable, List
import logging

# Import all pipeline factory functions
from .pipelines.microscopy_pipelines import create_fftnmf_pipeline
from .pipelines.sam_pipelines import create_sam_pipeline
from .pipelines.atomistic_pipelines import create_atomistic_pipeline

# Import instruction prompts
from .instruct import (
    MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    MICROSCOPY_CLAIMS_INSTRUCTIONS,
    MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS,
    SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS,
    ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS,
    MICROSCOPY_PIPELINE_SELECTION_INSTRUCTIONS
)


# Pipeline Registry Structure:
# {
#     'pipeline_id': {
#         'description': 'Human-readable description for LLM selection',
#         'factory': function_that_creates_pipeline,
#         'prompts': {
#             'analysis': 'prompt for standard analysis',
#             'claims': 'prompt for claims generation',
#             'recommendations': 'prompt for measurement recommendations'
#         }
#     }
# }

MICROSCOPY_PIPELINE_REGISTRY = {
    '_meta': { 
        'selection_instructions': MICROSCOPY_PIPELINE_SELECTION_INSTRUCTIONS
    },
    
    'general': {
        'description': (
            'Standard microstructure analysis using FFT and NMF decomposition. '
            'Use for images where atoms are NOT individually resolved, OR for atomic-resolution '
            'images that are severely disordered (amorphous, very noisy, fragmented). '
            'Analyzes grains, phases, domains, and periodic structures at the meso-scale.'
        ),
        'factory': create_fftnmf_pipeline,
        'prompts': {
            'analysis': MICROSCOPY_ANALYSIS_INSTRUCTIONS,
            'claims': MICROSCOPY_CLAIMS_INSTRUCTIONS,
            'recommendations': MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
        }
    },
    
    'sam': {
        'description': (
            'Particle/object segmentation using the Segment Anything Model (SAM). '
            'Use for images containing large, distinct, countable objects like nanoparticles, '
            'cells, pores, or other discrete entities. Provides size distribution, shape analysis, '
            'and spatial arrangement statistics.'
        ),
        'factory': create_sam_pipeline,
        'prompts': {
            'analysis': SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS,  # SAM doesn't have separate analysis mode
            'claims': SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS,
            'recommendations': SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
        }
    },
    
    'atomistic': {
        'description': (
            'Atomic-resolution analysis using deep learning for atom detection. '
            'Use ONLY for high-quality images where individual atoms or atomic columns '
            'are clearly visible in a crystalline lattice. Analyzes atomic positions, '
            'intensities, local environments, defects, and interfaces at the atomic scale.'
        ),
        'factory': create_atomistic_pipeline,
        'prompts': {
            'analysis': ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS,
            'claims': ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS,
            'recommendations': ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
        }
    }
}


def get_available_pipelines(agent_type: str = 'microscopy') -> Dict:
    """
    Get all available pipelines for a given agent type.
    
    Args:
        agent_type: Type of agent ('microscopy', 'spectroscopy', etc.)
        
    Returns:
        Dictionary of available pipelines for that agent type
    """
    if agent_type == 'microscopy':
        return MICROSCOPY_PIPELINE_REGISTRY
    else:
        # Future: Add other agent types here
        return {}


def register_pipeline(agent_type: str,
                     pipeline_id: str,
                     description: str,
                     factory: Callable,
                     prompts: Dict[str, str]) -> None:
    """
    Register a new pipeline (for future extensibility).
    
    Args:
        agent_type: Type of agent this pipeline belongs to
        pipeline_id: Unique identifier for the pipeline
        description: Human-readable description for LLM
        factory: Function that creates the pipeline
        prompts: Dict mapping prompt types to instruction strings
    """
    if agent_type == 'microscopy':
        MICROSCOPY_PIPELINE_REGISTRY[pipeline_id] = {
            'description': description,
            'factory': factory,
            'prompts': prompts
        }
        logging.getLogger(__name__).info(f"Registered new microscopy pipeline: '{pipeline_id}'")
    else:
        # Future: Handle other agent types
        logging.getLogger(__name__).warning(f"Cannot register pipeline for unknown agent type: '{agent_type}'")


def create_pipeline_for_agent(pipeline_id: str,
                              agent_type: str,
                              model,
                              logger: logging.Logger,
                              generation_config,
                              safety_settings,
                              settings: dict,
                              parse_fn: Callable,
                              store_fn: Callable,
                              **extra_dependencies) -> List:
    """
    Factory function to create a pipeline instance.
    
    Args:
        pipeline_id: ID of the pipeline to create
        agent_type: Type of agent
        model: LLM model instance
        logger: Logger instance
        generation_config: LLM generation config
        safety_settings: LLM safety settings
        settings: Agent-specific settings
        parse_fn: Function to parse LLM responses
        store_fn: Function to store analysis results
        **extra_dependencies: Additional dependencies needed by specific pipelines
        
    Returns:
        List of controller instances (the pipeline)
    """
    available = get_available_pipelines(agent_type)
    
    if pipeline_id not in available:
        raise ValueError(f"Unknown pipeline ID '{pipeline_id}' for agent type '{agent_type}'")
    
    factory_fn = available[pipeline_id]['factory']
    
    # Call the factory function with all necessary arguments
    return factory_fn(
        model=model,
        logger=logger,
        generation_config=generation_config,
        safety_settings=safety_settings,
        settings=settings,
        parse_fn=parse_fn,
        store_fn=store_fn,
        **extra_dependencies
    )


def get_prompt_for_pipeline(pipeline_id: str,
                            agent_type: str,
                            prompt_type: str = 'claims') -> str:
    """
    Get the appropriate instruction prompt for a pipeline.
    
    Args:
        pipeline_id: ID of the pipeline
        agent_type: Type of agent
        prompt_type: Type of prompt ('analysis', 'claims', 'recommendations')
        
    Returns:
        Instruction prompt string
    """
    available = get_available_pipelines(agent_type)
    
    if pipeline_id not in available:
        raise ValueError(f"Unknown pipeline ID '{pipeline_id}' for agent type '{agent_type}'")
    
    prompts = available[pipeline_id]['prompts']
    
    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type '{prompt_type}' for pipeline '{pipeline_id}'")
    
    return prompts[prompt_type]