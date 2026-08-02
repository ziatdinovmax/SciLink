# scilink/agents/sim_agents/dft_orchestrator.py
"""Deprecated DFT one-shot orchestrator — delegates to the scale-agnostic pipeline.

``DFTOrchestrator`` predates the scale-agnostic
:func:`scilink.agents.sim_agents.simulation_pipeline.run_complete_workflow`
and is retained only as a thin back-compat shim. New code should call
``run_complete_workflow`` directly with ``scale="periodic_dft"``.

The pipeline routes input generation through ``PeriodicDFTAgent`` (engine
selected by ``software``) and validates with the engine-neutral
``InputValidator``, so this shim carries no VASP-specific result shaping —
it returns the pipeline result unchanged.
"""

import logging
import warnings
from typing import Optional, Dict, Any

from ._deprecation import normalize_params
from .simulation_pipeline import run_complete_workflow

logger = logging.getLogger(__name__)


class DFTOrchestrator:
    """Back-compat shim over the scale-agnostic simulation pipeline.

    Preserves the historical constructor and ``run_complete_workflow``
    signature; forwards to
    :func:`scilink.agents.sim_agents.simulation_pipeline.run_complete_workflow`
    with ``scale="periodic_dft"`` and returns its result unchanged.
    """

    def __init__(self,
                 api_key: str = None,
                 base_url: Optional[str] = None,
                 futurehouse_api_key: str = None,
                 mp_api_key: str = None,
                 generator_model: str = "gemini-3-pro-preview",
                 validator_model: str = "gemini-3-pro-preview",
                 output_dir: str = "dft_workflow_output",
                 max_refinement_cycles: int = 4,
                 script_timeout: int = 300,
                 vasp_generator_method: str = "llm",
                 # Deprecated aliases
                 google_api_key: str = None,
                 local_model: str = None):
        warnings.warn(
            "DFTOrchestrator is deprecated; call "
            "scilink.agents.sim_agents.simulation_pipeline.run_complete_workflow"
            "(scale='periodic_dft', ...) directly.",
            DeprecationWarning, stacklevel=2,
        )
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="DFTOrchestrator",
        )
        self.api_key = api_key
        self.base_url = base_url
        self.futurehouse_api_key = futurehouse_api_key
        self.mp_api_key = mp_api_key
        self.generator_model = generator_model
        self.output_dir = output_dir
        self.max_refinement_cycles = max_refinement_cycles
        self.script_timeout = script_timeout
        # Historical name for the input-generation method ("llm"/"atomate2").
        self.vasp_generator_method = vasp_generator_method

    def run_complete_workflow(self, user_request: str,
                              structure_class: str = "crystal") -> Dict[str, Any]:
        """Run the periodic-DFT pipeline and return its result unchanged."""
        return run_complete_workflow(
            user_request,
            scale="periodic_dft",
            software="vasp",
            method=self.vasp_generator_method,
            structure_class=structure_class,
            output_dir=self.output_dir,
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.generator_model,
            futurehouse_api_key=self.futurehouse_api_key,
            mp_api_key=self.mp_api_key,
            max_refinement_cycles=self.max_refinement_cycles,
            script_timeout=self.script_timeout,
        )
