"""
scilink.graphs.state
====================

LangGraph state schemas for SciLink orchestrators.

Three-level hierarchy
---------------------

    OrchestratorState               (shared by all three orchestrators)
        └── AnalysisOrchestratorState
        └── PlanningOrchestratorState
        └── SimulationOrchestratorState

Each layer extends the one above by adding domain-specific fields that mirror
the mutable instance variables currently held on the orchestrator objects.

Verification subgraph state
---------------------------

    VerificationState               (self-contained; used by graphs/verification.py)

All state TypedDicts are defined here so the graph definitions stay clean and
state shape is a single authoritative source.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, Union

from langgraph.graph import MessagesState


# ---------------------------------------------------------------------------
# Base layer — shared by all three orchestrators
# ---------------------------------------------------------------------------


class OrchestratorState(MessagesState):
    """
    Shared fields for all three orchestrator graphs.

    Extends LangGraph's ``MessagesState`` which already declares::

        messages: Annotated[list[AnyMessage], add_messages]

    The ``add_messages`` reducer means nodes return only new messages; LangGraph
    appends them.  All other fields here are plain-replace (last-writer-wins).

    Fields
    ------
    autonomy_mode
        String tag for the autonomy level in effect. Mirrors the
        ``AnalysisMode``, ``AutonomyLevel``, and ``SimulationMode`` enums:
        one of ``"co-pilot"``, ``"supervised"``, ``"autonomous"``.

    active_skill
        Skill name(s) currently active for the session, or ``None``.

    session_dir
        Absolute path to the session's working directory (base_dir).

    checkpoint_data
        Snapshot of session state serialised for MemorySaver persistence.
        Empty dict when nothing has been checkpointed yet.

    mcp_connections
        List of connected MCP server names (informational; actual connection
        objects live on the orchestrator instance).

    step_count
        Monotonically increasing counter incremented by every call to
        ``execute_tools``.  Used to enforce ``MAX_TOOL_ITERATIONS`` without a
        ``while`` loop: the routing function routes to ``END`` when this
        reaches the configured maximum.
    """

    autonomy_mode: str
    active_skill: Union[str, List[str], None]
    session_dir: str
    checkpoint_data: Dict[str, Any]
    mcp_connections: List[str]
    step_count: int


# ---------------------------------------------------------------------------
# Analysis orchestrator
# ---------------------------------------------------------------------------


class AnalysisOrchestratorState(OrchestratorState):
    """
    State for ``AnalysisOrchestratorAgent``.

    Mirrors all mutable instance variables currently held on the orchestrator.

    Fields
    ------
    current_data_path
        Path to the data file or directory currently under analysis.

    current_data_type
        Detected data type string (e.g. ``"microscopy"``, ``"1d_data"``).

    current_metadata
        Structured metadata dict loaded by ``load_metadata`` / ``convert_metadata``.

    selected_agent_id
        Integer index into the agent registry (0=CurveFitting, 1=ImageAnalysis,
        2=Hyperspectral).

    analysis_results
        Accumulated list of analysis result dicts from completed ``run_analysis`` calls.

    active_knowledge
        List of active knowledge-synthesis entries injected into subsequent analyses.

    message_count
        Running count of user turns (used for checkpoint-interval logic).

    analysis_run_counter
        Monotonically increasing counter used to generate unique analysis IDs
        within the same second.
    """

    current_data_path: Optional[str]
    current_data_type: Optional[str]
    current_metadata: Optional[Dict[str, Any]]
    selected_agent_id: Optional[int]
    analysis_results: List[Dict[str, Any]]
    active_knowledge: List[Dict[str, Any]]
    message_count: int
    analysis_run_counter: int


# ---------------------------------------------------------------------------
# Planning orchestrator
# ---------------------------------------------------------------------------


class PlanningOrchestratorState(OrchestratorState):
    """
    State for ``PlanningOrchestratorAgent``.

    Fields
    ------
    objective
        Research objective description for the current campaign.

    active_scalarizer_script
        Path to the active scalarizer script, or ``None``.

    expected_input_columns
        List of expected input column names for the BO loop, or ``None``.

    expected_target_columns
        List of expected target/output column names.

    target_directions
        Dict mapping column name → ``"maximize"`` or ``"minimize"``.

    expected_input_types
        Dict mapping column name → ``"continuous"`` or ``"categorical"``.

    expected_input_levels
        Dict mapping categorical column name → list of allowed levels.

    latest_tea_results
        Most recent TEA (techno-economic analysis) results, or ``None``.

    active_knowledge
        Active knowledge-synthesis entries (mirrors analysis orchestrator).

    message_count
        Running count of user turns.
    """

    objective: str
    active_scalarizer_script: Optional[str]
    expected_input_columns: Optional[List[str]]
    expected_target_columns: List[str]
    target_directions: Dict[str, str]
    expected_input_types: Optional[Dict[str, str]]
    expected_input_levels: Optional[Dict[str, List[Any]]]
    latest_tea_results: Optional[Any]
    active_knowledge: List[Dict[str, Any]]
    message_count: int


# ---------------------------------------------------------------------------
# Simulation orchestrator
# ---------------------------------------------------------------------------


class SimulationOrchestratorState(OrchestratorState):
    """
    State for ``SimulationOrchestratorAgent``.

    Fields
    ------
    generated_structures
        List of structure records produced during the session.
        Each record is a dict with keys like ``slug``, ``poscar_path``,
        ``incar_path``, ``kpoints_path``, ``description``, ``validation``.

    default_calc_params
        Sticky VASP calculation parameters (ENCUT, k-mesh density, functional, …)
        that carry across structures within the session.

    message_count
        Running count of user/task turns.
    """

    generated_structures: List[Dict[str, Any]]
    default_calc_params: Dict[str, Any]
    message_count: int


# ---------------------------------------------------------------------------
# Verification subgraph
# ---------------------------------------------------------------------------


class VerificationRecord(Dict[str, Any]):
    """Type alias — a single entry in the verification history list.

    Each record contains at minimum:
        iteration        int
        score            float
        issues           list[str]
        annealing_level  int
        approved         bool
        config_snapshot  dict
    """


class VerificationState(MessagesState):
    """
    Self-contained state for the verification-retry subgraph.

    This subgraph replaces the ``while`` loop that currently appears in both
    ``image_analysis_controllers.py:_execute_and_verify`` and
    ``curve_fitting_controllers.py``.  The loop logic (annealing, patience,
    best-result tracking) moves from imperative iteration variables into this
    TypedDict so LangGraph can checkpoint individual verification passes and
    the loop can be paused / resumed.

    Fields
    ------
    analysis_config
        Mutable configuration dict for the current analysis attempt.
        Updated by ``apply_feedback`` node after each verification round.

    current_result
        Result dict from the most recent analysis execution, or ``None``.

    best_result
        Best result seen so far (highest quality score), or ``None``.

    best_score
        Quality score of ``best_result`` (0.0–1.0 scale used by verifiers).

    verification_history
        Ordered list of ``VerificationRecord`` entries — one per round.
        Uses ``Annotated[list, operator.add]`` so sub-branches can append
        without clobbering each other in parallel contexts.

    iteration
        Current verification iteration index (0-based).

    max_iterations
        Upper bound on iterations (mirrors ``DEFAULT_MAX_VERIFICATION_ITERATIONS``
        from the controllers).

    annealing_level
        Current constraint-annealing level (0=tight, 1=warm, 2=hot).

    patience_counter
        Number of consecutive iterations without a score improvement.

    approved
        ``True`` once a result meets the quality threshold.

    human_feedback_requested
        ``True`` if the current state is waiting for human input (used by
        CO_PILOT / SUPERVISED modes to surface the interrupt).
    """

    # --- Analysis configuration ---
    analysis_config: Dict[str, Any]

    # --- Current / best results ---
    current_result: Optional[Dict[str, Any]]
    best_result: Optional[Dict[str, Any]]
    best_score: float

    # --- History ---
    verification_history: Annotated[List[Dict[str, Any]], operator.add]

    # --- Loop control ---
    iteration: int
    max_iterations: int
    annealing_level: int
    patience_counter: int

    # --- Terminal flags ---
    approved: bool
    human_feedback_requested: bool
