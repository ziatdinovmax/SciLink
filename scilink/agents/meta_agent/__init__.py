"""Meta-agent package — the orchestrator-of-orchestrators.

Presents a single chat surface and delegates to the analysis and planning
mode orchestrators through their ``run_task`` contract. See CLAUDE.md
"The meta agent".

This package must stay importable without optional dependencies (notably
ASE): the simulation orchestrator is reached only through a guarded,
in-function import in the deferred ``delegate_to_simulation`` seam.
"""

from .meta_orchestrator import MetaOrchestratorAgent, MetaMode

__all__ = ["MetaOrchestratorAgent", "MetaMode"]
