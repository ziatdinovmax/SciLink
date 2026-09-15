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

_react.py
    Shared ReAct backbone — all LLM routing, tool dispatch, context
    compression, and step-limit logic in one deep module.
    Entry point: ``build_react_graph(orch, state_type, checkpointer)``
    Not exported from this package; use the mode-specific builders below.

state.py
    TypedDict state schemas for all three orchestrators.  Three-level
    hierarchy::

        OrchestratorState
            └── AnalysisOrchestratorState
            └── PlanningOrchestratorState
            └── SimulationOrchestratorState

analysis.py
    ReAct graph for ``AnalysisOrchestratorAgent``.
    Entry point: ``build_analysis_graph(orch)``

planning.py
    ReAct graph for ``PlanningOrchestratorAgent``.
    Entry point: ``build_planning_graph(orch)``

simulation.py
    ReAct graph for ``SimulationOrchestratorAgent``.
    Entry point: ``build_simulation_graph(orch)``

meta.py
    ReAct graph for ``MetaOrchestratorAgent``.
    Entry point: ``build_meta_graph(orch)``

Note: an earlier ``verification.py`` (a LangGraph verification-retry
subgraph meant to replace the per-item verify/refine loop duplicated in
``image_analysis_controllers.py`` / ``curve_fitting_controllers.py``) was
removed — it was never wired into either controller. Both, plus
hyperspectral, were unified instead behind a shared, non-LangGraph
``CodegenQCEngine`` (issue #327), which gained substantial exclusive
capability (locked-script reuse, wall-clock budgets, hyperspectral support)
that the subgraph never had. See TODO.md round-2 item 4.

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
    MetaOrchestratorState,
)
from scilink.graphs.analysis import build_analysis_graph
from scilink.graphs.planning import build_planning_graph
from scilink.graphs.simulation import build_simulation_graph
from scilink.graphs.meta import build_meta_graph

__all__ = [
    # State schemas
    "OrchestratorState",
    "AnalysisOrchestratorState",
    "PlanningOrchestratorState",
    "SimulationOrchestratorState",
    "MetaOrchestratorState",
    # Graph builders
    "build_analysis_graph",
    "build_planning_graph",
    "build_simulation_graph",
    "build_meta_graph",
]
