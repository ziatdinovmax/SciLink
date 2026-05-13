"""
scilink.graphs.state
====================

LangGraph state schemas for SciLink orchestrators.

These TypedDicts mirror the runtime state that currently lives as instance
variables on each orchestrator.  They are defined here so the graph
definitions stay clean, and so state shape is a single authoritative source
when the graphs are wired up.

Nothing in the live orchestrators imports or uses these yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import MessagesState


class AnalysisState(MessagesState):
    """
    Complete state for the AnalysisOrchestratorAgent graph.

    Extends LangGraph's ``MessagesState`` which already declares::

        messages: Annotated[list[AnyMessage], add_messages]

    The ``add_messages`` reducer means nodes return only new messages;
    LangGraph appends them — no need to return the full history.

    All other fields are plain-replace (last-writer-wins), matching how the
    orchestrator currently mutates its instance variables directly.
    """

    # --- Active data context ---
    current_data_path: Optional[str]
    current_data_type: Optional[str]  # e.g. "microscopy", "1d_data"
    current_metadata: Optional[Dict[str, Any]]

    # --- Agent selection ---
    # Index into the agent registry (0=CurveFitting, 1=ImageAnalysis, 2=Hyperspectral)
    selected_agent_id: Optional[int]

    # --- Results accumulation ---
    analysis_results: List[Dict[str, Any]]

    # --- Knowledge / skill context ---
    active_knowledge: List[Dict[str, Any]]

    # --- Session bookkeeping ---
    message_count: int
    analysis_run_counter: int
