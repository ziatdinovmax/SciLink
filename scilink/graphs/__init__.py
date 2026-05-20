"""
scilink.graphs
==============

LangGraph graph definitions for SciLink orchestrators.

Phase 1 — Backbone
------------------

All three orchestrators now use a LangGraph ``StateGraph`` (ReAct topology)
as their runtime backbone.  The hand-rolled ``while iteration < MAX_TOOL_ITERATIONS``
loops have been replaced by compiled graphs backed by ``MemorySaver``.

Modules
-------

state.py
    TypedDict state schemas for all three orchestrators and the verification
    subgraph.  Three-level hierarchy::

        OrchestratorState
            └── AnalysisOrchestratorState
            └── PlanningOrchestratorState
            └── SimulationOrchestratorState
        VerificationState  (self-contained, used by verification.py)

analysis.py
    ReAct graph for ``AnalysisOrchestratorAgent``.
    Entry point: ``build_analysis_graph(orch)``

planning.py
    ReAct graph for ``PlanningOrchestratorAgent``.
    Entry point: ``build_planning_graph(orch)``

simulation.py
    ReAct graph for ``SimulationOrchestratorAgent``.
    Entry point: ``build_simulation_graph(orch)``

verification.py
    Reusable verification-retry subgraph that replaces the duplicated
    ``while`` loops in ``image_analysis_controllers.py`` and
    ``curve_fitting_controllers.py``.
    Entry point: ``build_verification_subgraph(run_fn, verify_fn, ...)``

Phase 2 (planned)
-----------------

parallel_analysis.py — multi-strategy fan-out using the Send API

Phase 3 (planned)
-----------------

fusion.py — multi-modal heterogeneous parallel analysis
"""

from scilink.graphs.state import (
    OrchestratorState,
    AnalysisOrchestratorState,
    PlanningOrchestratorState,
    SimulationOrchestratorState,
    VerificationState,
)
from scilink.graphs.analysis import build_analysis_graph
from scilink.graphs.planning import build_planning_graph
from scilink.graphs.simulation import build_simulation_graph
from scilink.graphs.verification import build_verification_subgraph

__all__ = [
    # State schemas
    "OrchestratorState",
    "AnalysisOrchestratorState",
    "PlanningOrchestratorState",
    "SimulationOrchestratorState",
    "VerificationState",
    # Graph builders
    "build_analysis_graph",
    "build_planning_graph",
    "build_simulation_graph",
    "build_verification_subgraph",
]
