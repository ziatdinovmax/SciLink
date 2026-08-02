# scilink/workflows/dft_workflow.py
#
# Backward-compatibility shim. The class moved to scilink.agents.sim_agents
# and was renamed DFTOrchestrator to match LAMMPSOrchestrator and reflect
# its actual role (a sim-agents orchestrator, not a high-level workflow).

import warnings

from ..agents.sim_agents.dft_orchestrator import DFTOrchestrator as DFTWorkflow

warnings.warn(
    "scilink.workflows.dft_workflow.DFTWorkflow is deprecated; "
    "import DFTOrchestrator from scilink.agents.sim_agents instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DFTWorkflow"]
