"""One place that builds a mode orchestrator for the meta: its persistent
children and its ephemeral workers alike.

A persistent child accumulates context across a conversation, and its
``run_task`` reports what one call produced as a before/after window over that
shared state, so two calls on one child at once would each report the other's
output as their own. An ephemeral **worker** is built for one work item, in its
own directory, and is never registered in ``orch._children``: nothing it holds
is shared with another thread, so its window is correct by construction.
Fan-out branches are analysis workers; a swarm runs workers of every mode.

Building both kinds here keeps a mode's construction (keys, knowledge base,
file fence, shared extensions) written once, so a worker is never configured
differently from the specialist the user talks to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

MODES = ("analysis", "planning", "simulation")


def _worker_knowledge_dir(orch: Any, base_dir: Path, persistent: bool) -> Path:
    """The knowledge directory a planning child is built with.

    The persistent specialist uses what the meta attached (or its own
    folder). A worker gets the same store KB (the planner copies a store KB
    into the worker's own ``kb_cache``), but a PLAIN folder KB — a
    ``--knowledge-dir`` path, a launch folder's ``kb_storage`` — is an index
    the planner appends to in place, so two workers would write one faiss
    file at once. Its ``default_kb_*`` files are copied into the worker's
    own ``knowledge/`` instead, one consistent generation.
    """
    attached = orch.knowledge_dir
    own = Path(base_dir) / "knowledge"
    if not attached:
        return own
    if persistent:
        return Path(attached)
    from ...knowledge.kb_store import read_manifest, snapshot_kb
    attached = Path(attached)
    if read_manifest(attached):                 # a store KB: the planner copies it itself
        return attached
    if attached.is_dir() and any(attached.glob("default_kb_*")):
        snapshot_kb(attached, own)
    return own


def build_child(orch: Any, mode: str, base_dir: Path, *, restore: bool = False,
                persistent: bool = False, label: Optional[str] = None) -> Any:
    """A mode orchestrator in ``base_dir``, sharing the meta's credentials,
    file fence and extensions.

    ``persistent`` children rest in CO_PILOT, as the meta's specialists always
    have. Workers rest in AUTONOMOUS, except planning: its constructor requires
    a ``data_dir`` under AUTOPILOT / AUTONOMOUS and a delegated planner has
    none, so it rests in CO_PILOT too. The resting mode never reaches a task:
    every ``run_task`` pins the autonomy for its own call.

    ``restore=True`` rebuilds the orchestrator from the checkpoint in
    ``base_dir`` (the meta's restore path, and a worker resumed in place).
    Simulation imports ``ase``; a missing install raises ``ImportError`` for
    the caller to report, and the meta module stays importable without it.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    base_dir = Path(base_dir)
    if not persistent:
        base_dir.mkdir(parents=True, exist_ok=True)
    roots = ([str(r) for r in orch.path_fence.roots]
             if getattr(orch, "path_fence", None) is not None else None)

    if mode == "analysis":
        from ..exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent, AnalysisMode
        child = AnalysisOrchestratorAgent(
            base_dir=str(base_dir),
            api_key=orch.api_key,
            model_name=orch.model_name,
            base_url=orch.base_url,
            embedding_model=orch.embedding_model,
            embedding_api_key=orch.embedding_api_key,
            futurehouse_api_key=orch.futurehouse_api_key,
            restore_checkpoint=restore,
            analysis_mode=AnalysisMode.CO_PILOT if persistent else AnalysisMode.AUTONOMOUS,
            file_roots=roots,
        )
    elif mode == "planning":
        from ..planning_agents.planning_orchestrator import (
            DELEGATED_OBJECTIVE, PlanningOrchestratorAgent, AutonomyLevel,
        )
        child = PlanningOrchestratorAgent(
            objective=DELEGATED_OBJECTIVE,
            base_dir=str(base_dir),
            api_key=orch.api_key,
            model_name=orch.model_name,
            base_url=orch.base_url,
            embedding_model=orch.embedding_model,
            embedding_api_key=orch.embedding_api_key,
            embedding_base_url=orch.embedding_base_url,
            futurehouse_api_key=orch.futurehouse_api_key,
            restore_checkpoint=restore,
            autonomy_level=AutonomyLevel.CO_PILOT,
            data_dir=None,
            file_roots=roots,
            # Explicit stable KB when the caller opted in (CLI
            # --knowledge-dir / chat-approved attach_knowledge_base),
            # else session-scoped. Without
            # the session-scoped default the child inherits
            # PlanningAgent's cwd-relative default (./kb_storage),
            # silently loading whatever stale KB the launch directory
            # holds — and a non-empty index forces query embedding on
            # every plan, which hard-fails when the embedding provider's
            # key is absent. The stable-cwd default stays intentional for
            # standalone use; meta children isolate per session unless
            # the user explicitly points them at a KB. Workers never share
            # index files: a store KB is copied into each worker's
            # ``kb_cache`` by the planner itself, a plain folder KB is
            # copied by ``_worker_knowledge_dir`` here.
            knowledge_dir=str(_worker_knowledge_dir(orch, base_dir, persistent)),
        )
    else:
        from ..sim_agents.simulation_orchestrator import (
            SimulationOrchestratorAgent, SimulationMode,
        )
        child = SimulationOrchestratorAgent(
            base_dir=str(base_dir),
            api_key=orch.api_key,
            model_name=orch.model_name,
            base_url=orch.base_url,
            futurehouse_api_key=orch.futurehouse_api_key,
            restore_checkpoint=restore,
            simulation_mode=SimulationMode.CO_PILOT if persistent else SimulationMode.AUTONOMOUS,
            file_roots=roots,
            # mp_api_key not threaded from the meta (its constructor has
            # none); MPRester falls back to the MP_API_KEY env var when a
            # crystal-from-Materials-Project structure is requested.
        )

    # Label its answers as the specialist's: in the meta's verbose stream a
    # child's final answer is a delegated deliverable, not the meta's own
    # user-facing response.
    child._agent_label = label or f"{mode.capitalize()} {'specialist' if persistent else 'worker'}"
    # Share skills / custom tools / MCP servers registered on the meta.
    orch._propagate_extensions_to_child(child)
    return child


def release_child(child: Any) -> None:
    """What a finished worker must let go of: the MCP servers it connected
    (each worker opens its own; nothing else closes them)."""
    for name in list(getattr(child, "_mcp_connections", {}) or {}):
        try:
            child.disconnect_mcp_server(name)
        except Exception:  # noqa: BLE001 - cleanup never fails a result
            pass
