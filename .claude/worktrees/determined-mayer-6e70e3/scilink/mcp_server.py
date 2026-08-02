"""MCP server that exposes SciLink's analysis and planning tools.

Allows any MCP client (Claude Desktop, Cursor, etc.) to use SciLink as a
tool provider.  Start with::

    scilink serve --model gemini-3.1-pro-preview

Supports three autonomy modes:
- **autonomous** (default) — tools execute without human approval.
- **supervised** — tools run but return key decisions for review.
- **co-pilot** — tools that need approval return a ``needs_input``
  response; the MCP client must call ``scilink_respond`` to continue.

The ``mcp`` package is installed by default with SciLink.
"""

import asyncio
import concurrent.futures
import contextlib
import io
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def _require_mcp():
    if not HAS_MCP:
        raise ImportError(
            "MCP server support requires the 'mcp' package. "
            "Install with: pip install scilink"
        )


# Tools that require user approval in co-pilot / supervised modes.
# In co-pilot mode ALL of these pause; in supervised mode only the
# high-impact subset (run_analysis, run_optimization) pauses.
_COPILOT_APPROVAL_TOOLS = {
    "run_analysis", "select_agent", "assess_novelty",
    "get_recommendations", "run_optimization", "generate_initial_plan",
    "generate_implementation_code", "run_economic_analysis",
    "discard_plan",
}
_SUPERVISED_APPROVAL_TOOLS = {
    "run_analysis", "run_optimization", "discard_plan",
}

# Tools that support optional background execution via ``background=true``.
# These are the long-running tools where a blocking call can realistically
# exceed the ~4 minute tool-call timeout in clients like Claude Desktop:
#   - run_analysis / run_optimization: full agent analysis or BO loop
#   - assess_novelty: FutureHouse literature search per claim (max_wait 600s each)
#   - get_recommendations: LLM synthesis over a full analysis record
#   - generate_initial_plan / generate_implementation_code: RAG + LLM generation
#     over potentially large docs/code knowledge bases
#   - run_economic_analysis: TEA with knowledge retrieval + LLM synthesis
#   - orchestrate_analysis / orchestrate_planning: full orchestrator chat loop
_BACKGROUND_CAPABLE_TOOLS = {
    "run_analysis",
    "run_optimization",
    "assess_novelty",
    "get_recommendations",
    "generate_initial_plan",
    "generate_implementation_code",
    "run_economic_analysis",
    "orchestrate_analysis",
    "orchestrate_planning",
}


# ── Schema conversion ───────────────────────────────────────────────────

def _openai_to_mcp_tool(schema: dict, prefix: str = "scilink") -> types.Tool:
    """Convert an OpenAI function-calling schema to an MCP Tool object."""
    fn = schema.get("function", schema)
    name = fn.get("name", "")
    input_schema = fn.get("parameters", {"type": "object", "properties": {}})

    # Add optional ``background`` parameter for long-running tools.
    if name in _BACKGROUND_CAPABLE_TOOLS:
        schema_copy = json.loads(json.dumps(input_schema))
        schema_copy.setdefault("properties", {})["background"] = {
            "type": "boolean",
            "description": (
                "If true, run in the background and return a job_id "
                "immediately. Use scilink_job_status and scilink_job_result "
                "to poll and retrieve results. Default: false (blocking)."
            ),
        }
        input_schema = schema_copy

    return types.Tool(
        name=f"{prefix}_{name}" if prefix else name,
        description=fn.get("description", ""),
        inputSchema=input_schema,
    )


# ── Stdout capture ──────────────────────────────────────────────────────

def _execute_tool_captured(tools, tool_name: str, kwargs: dict) -> str:
    """Execute a tool while capturing stdout so it doesn't corrupt stdio MCP transport."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = tools.execute_tool(tool_name, **kwargs)

    log_output = captured.getvalue().strip()
    if log_output:
        logging.info(f"[tool:{tool_name}] {log_output}")

    return result


# ── Pending action support (co-pilot / supervised) ──────────────────────

class PendingAction:
    """Holds a pending tool call that needs user approval before execution."""

    def __init__(self, tool_name: str, kwargs: dict, prompt: str, context: dict = None):
        self.tool_name = tool_name
        self.kwargs = kwargs
        self.prompt = prompt
        self.context = context or {}


def _needs_approval(tool_name: str, mode: str) -> bool:
    """Check whether a tool requires user approval given the autonomy mode."""
    if mode == "autonomous":
        return False
    if mode in ("co-pilot", "co_pilot", "copilot"):
        return tool_name in _COPILOT_APPROVAL_TOOLS
    if mode == "supervised":
        return tool_name in _SUPERVISED_APPROVAL_TOOLS
    return False


def _build_approval_prompt(tool_name: str, kwargs: dict) -> str:
    """Build a human-readable description of the pending action."""
    parts = [f"SciLink wants to execute: {tool_name}"]
    if kwargs:
        for k, v in kwargs.items():
            val = str(v)
            if len(val) > 100:
                val = val[:100] + "..."
            parts.append(f"  {k}: {val}")
    parts.append("\nCall scilink_respond with 'yes' to approve or 'no' to cancel.")
    return "\n".join(parts)


# ── Server factory ──────────────────────────────────────────────────────

def create_server(
    *,
    api_key: str = None,
    model_name: str = "gemini-3.1-pro-preview",
    base_url: str = None,
    mode: str = "both",
    session_dir: str = None,
    analysis_mode: str = "autonomous",
    futurehouse_api_key: str = None,
) -> "Server":
    """Create and return a configured MCP Server instance.

    Args:
        api_key: LLM API key.
        model_name: Model identifier.
        base_url: Optional OpenAI-compatible endpoint.
        mode: ``"analyze"``, ``"plan"``, or ``"both"``.
        session_dir: Directory for session outputs.
        analysis_mode: ``"autonomous"``, ``"supervised"``, or ``"co-pilot"``.
        futurehouse_api_key: Optional FutureHouse/Edison API key.

    Returns:
        A configured ``mcp.server.lowlevel.Server``.
    """
    _require_mcp()

    server = Server("scilink")

    # ── State (initialized lazily on first tool list) ────────────────
    # Thread pool for background jobs
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    state: Dict[str, Any] = {
        "analysis_orch": None,
        "planning_orch": None,
        "pending_action": None,
        "initialized": False,
        "jobs": {},
        "job_counter": 0,
        "config": {
            "api_key": api_key,
            "model_name": model_name,
            "base_url": base_url,
            "mode": mode,
            "session_dir": session_dir,
            "analysis_mode": analysis_mode,
            "futurehouse_api_key": futurehouse_api_key,
        },
    }

    # Tool name → (orchestrator_key, original_name) mapping
    tool_map: Dict[str, tuple] = {}

    def _ensure_initialized():
        if state["initialized"]:
            return

        config = state["config"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            _init_orchestrators(state, config)

        log = captured.getvalue().strip()
        if log:
            logging.info(f"[init] {log}")

        # Build tool map — analysis tools first, then planning with
        # collision avoidance (prefix colliding names with "plan_").
        if state["analysis_orch"]:
            for schema in state["analysis_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                if fn_name:
                    tool_map[f"scilink_{fn_name}"] = ("analysis_orch", fn_name)

        if state["planning_orch"]:
            for schema in state["planning_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                if fn_name:
                    mcp_name = f"scilink_{fn_name}"
                    if mcp_name in tool_map:
                        mcp_name = f"scilink_plan_{fn_name}"
                    tool_map[mcp_name] = ("planning_orch", fn_name)

        state["initialized"] = True

    # ── Eager init (call before starting transport) ────────────────

    def _eager_init():
        _ensure_initialized()

    server.eager_init = _eager_init

    # ── tools/list ───────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        _ensure_initialized()
        tools = []

        if state["analysis_orch"]:
            for schema in state["analysis_orch"].tools.openai_schemas:
                tools.append(_openai_to_mcp_tool(schema, prefix="scilink"))

        if state["planning_orch"]:
            for schema in state["planning_orch"].tools.openai_schemas:
                fn_name = schema.get("function", {}).get("name", "")
                mcp_name = f"scilink_{fn_name}"
                if mcp_name in tool_map and tool_map[mcp_name][0] != "planning_orch":
                    prefix = "scilink_plan"
                else:
                    prefix = "scilink"
                tools.append(_openai_to_mcp_tool(schema, prefix=prefix))

        # Always include scilink_respond so it's available if the user
        # switches to co-pilot/supervised mode mid-session via
        # scilink_set_autonomy (MCP clients only call tools/list once).
        tools.append(types.Tool(
            name="scilink_respond",
            description=(
                "Send a response to a pending SciLink action that requires "
                "human approval. Only needed in co-pilot or supervised mode. "
                "Call this after receiving a 'needs_input' status from another tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": (
                            "User's response: 'yes'/'approve' to proceed, "
                            "'no'/'reject' to cancel, or free-text feedback."
                        ),
                    },
                },
                "required": ["response"],
            },
        ))

        # Background job management tools
        tools.append(types.Tool(
            name="scilink_job_status",
            description=(
                "Check the status of a background job started with "
                "background=true. Returns 'running', 'completed', or 'failed'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by the original tool call.",
                    },
                },
                "required": ["job_id"],
            },
        ))
        tools.append(types.Tool(
            name="scilink_job_result",
            description=(
                "Retrieve the full result of a completed background job. "
                "Returns an error if the job is still running."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by the original tool call.",
                    },
                },
                "required": ["job_id"],
            },
        ))

        # Always add the set_autonomy tool
        tools.append(types.Tool(
            name="scilink_set_autonomy",
            description=(
                "Change the autonomy mode at runtime. In 'autonomous' mode "
                "all tools execute immediately. In 'supervised' mode high-impact "
                "tools (run_analysis, run_optimization) pause for approval. "
                "In 'co-pilot' mode most action tools pause for approval. "
                "Returns the new mode and whether scilink_respond is now needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["autonomous", "supervised", "co-pilot"],
                        "description": "The autonomy mode to switch to.",
                    },
                },
                "required": ["mode"],
            },
        ))

        # Orchestrator-level tools — delegate entire workflows instead
        # of calling individual tools one by one.
        if state["analysis_orch"]:
            tools.append(types.Tool(
                name="scilink_orchestrate_analysis",
                description=(
                    "Delegate a complete analysis workflow to SciLink's analysis "
                    "orchestrator. Instead of calling individual tools (examine_data, "
                    "select_agent, run_analysis, etc.) one by one, send a natural-"
                    "language prompt and the orchestrator handles the entire flow: "
                    "data examination, metadata handling, agent selection, analysis "
                    "execution, and results compilation. Best for complex multi-step "
                    "analyses where SciLink's domain expertise should drive the "
                    "workflow. Use background=true for non-trivial requests, as the "
                    "orchestrator may take several minutes."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Natural-language instruction for the analysis "
                                "orchestrator, e.g. 'Analyze the XPS data at "
                                "/path/to/data.csv using the xps skill and assess "
                                "novelty of the findings.'"
                            ),
                        },
                        "background": {
                            "type": "boolean",
                            "description": (
                                "If true, run in the background and return a job_id "
                                "immediately. Recommended for most requests. Use "
                                "scilink_job_status and scilink_job_result to poll "
                                "and retrieve results. Default: false."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            ))

        if state["planning_orch"]:
            tools.append(types.Tool(
                name="scilink_orchestrate_planning",
                description=(
                    "Delegate a complete planning workflow to SciLink's planning "
                    "orchestrator. Send a natural-language prompt and the "
                    "orchestrator handles the entire flow: objective understanding, "
                    "knowledge base retrieval, experimental plan generation, "
                    "implementation code, and Bayesian optimization. Best for "
                    "complex experimental design tasks where SciLink's domain "
                    "expertise should drive the workflow. Use background=true for "
                    "non-trivial requests."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "Natural-language instruction for the planning "
                                "orchestrator, e.g. 'Generate an experimental plan "
                                "for optimizing lithium extraction from brine using "
                                "papers in ./literature/ as the knowledge base.'"
                            ),
                        },
                        "background": {
                            "type": "boolean",
                            "description": (
                                "If true, run in the background and return a job_id "
                                "immediately. Recommended for most requests. Use "
                                "scilink_job_status and scilink_job_result to poll "
                                "and retrieve results. Default: false."
                            ),
                        },
                    },
                    "required": ["prompt"],
                },
            ))

        return tools

    # ── tools/call ───────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> List[types.TextContent]:
        _ensure_initialized()

        # Handle the respond tool
        if name == "scilink_respond":
            return await _handle_respond(state, arguments)

        # Handle autonomy mode switch
        if name == "scilink_set_autonomy":
            return _handle_set_autonomy(state, arguments)

        # Handle background job status/result
        if name == "scilink_job_status":
            return _handle_job_status(state, arguments)
        if name == "scilink_job_result":
            return _handle_job_result(state, arguments)

        # Handle orchestrator-level chat tools
        if name in ("scilink_orchestrate_analysis", "scilink_orchestrate_planning"):
            return await _handle_orchestrate(
                state, name, arguments, _executor
            )

        # Look up which orchestrator owns this tool
        if name not in tool_map:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Unknown tool: {name}",
                }),
            )]

        orch_key, original_name = tool_map[name]
        orch = state[orch_key]
        autonomy = state["config"]["analysis_mode"]

        # Co-pilot / supervised: intercept tools that need approval
        if _needs_approval(original_name, autonomy):
            prompt = _build_approval_prompt(original_name, arguments)
            state["pending_action"] = PendingAction(
                tool_name=original_name,
                kwargs=arguments,
                prompt=prompt,
                context={"orch_key": orch_key},
            )
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "needs_input",
                    "message": prompt,
                    "tool": original_name,
                    "arguments": arguments,
                }),
            )]

        # Background execution: if background=true and tool supports it,
        # submit to thread pool and return job_id immediately.
        run_in_background = arguments.pop("background", False)
        if run_in_background and original_name in _BACKGROUND_CAPABLE_TOOLS:
            state["job_counter"] += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = f"job_{ts}_{state['job_counter']:03d}"

            future = _executor.submit(
                _execute_tool_captured, orch.tools, original_name, arguments
            )
            state["jobs"][job_id] = {
                "future": future,
                "tool": original_name,
                "started_at": ts,
                "status": "running",
                "result": None,
            }

            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "started",
                    "job_id": job_id,
                    "tool": original_name,
                    "message": (
                        f"Analysis running in background (job {job_id}). "
                        "Use scilink_job_status to check progress, "
                        "then scilink_job_result to retrieve the result."
                    ),
                }),
            )]

        result = await asyncio.to_thread(
            _execute_tool_captured, orch.tools, original_name, arguments
        )

        return [types.TextContent(type="text", text=result)]

    # ── resources/list ───────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> List[types.Resource]:
        _ensure_initialized()
        resources = []

        if state["analysis_orch"]:
            resources.extend([
                types.Resource(
                    uri="scilink://analysis/status",
                    name="Analysis Session Status",
                    description="Current analysis session state",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="scilink://analysis/metadata",
                    name="Current Metadata",
                    description="Loaded sample/experiment metadata",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="scilink://analysis/agents",
                    name="Available Agents",
                    description="Registered analysis agents",
                    mimeType="application/json",
                ),
            ])

        if state["planning_orch"]:
            resources.extend([
                types.Resource(
                    uri="scilink://planning/status",
                    name="Planning Session Status",
                    description="Current planning session state",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="scilink://planning/plan",
                    name="Current Plan",
                    description="Active experimental plan",
                    mimeType="application/json",
                ),
            ])

        return resources

    # ── resources/read ───────────────────────────────────────────────

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        _ensure_initialized()

        # Analysis resources
        a_orch = state.get("analysis_orch")
        if uri == "scilink://analysis/status" and a_orch:
            return json.dumps({
                "current_data_path": getattr(a_orch, "current_data_path", None),
                "current_data_type": getattr(a_orch, "current_data_type", None),
                "selected_agent_id": getattr(a_orch, "selected_agent_id", None),
                "analysis_count": len(getattr(a_orch, "analysis_results", [])),
                "message_count": getattr(a_orch, "message_count", 0),
                "autonomy_mode": state["config"]["analysis_mode"],
            }, indent=2)

        if uri == "scilink://analysis/metadata" and a_orch:
            return json.dumps(
                getattr(a_orch, "current_metadata", None) or {}, indent=2
            )

        if uri == "scilink://analysis/agents" and a_orch:
            registry = getattr(a_orch, "_agent_registry", {})
            agents = {}
            for aid, entry in registry.items():
                agents[str(aid)] = {
                    "name": entry.get("name", ""),
                    "description": entry.get("description", ""),
                    "short_name": entry.get("short_name", ""),
                }
            return json.dumps(agents, indent=2)

        # Planning resources
        p_orch = state.get("planning_orch")
        if uri == "scilink://planning/status" and p_orch:
            return json.dumps({
                "has_plan": getattr(p_orch, "current_plan", None) is not None,
                "iteration": getattr(p_orch, "current_iteration", 0),
                "message_count": len(getattr(p_orch, "messages", [])),
                "autonomy_level": str(getattr(p_orch, "autonomy_level", "unknown")),
            }, indent=2)

        if uri == "scilink://planning/plan" and p_orch:
            plan = getattr(p_orch, "current_plan", None)
            if plan and hasattr(plan, "to_dict"):
                return json.dumps(plan.to_dict(), indent=2)
            return json.dumps(plan or {}, indent=2)

        # Backward compatibility with Phase 1 URIs
        if uri == "scilink://session/status" and a_orch:
            return await read_resource("scilink://analysis/status")
        if uri == "scilink://session/metadata" and a_orch:
            return await read_resource("scilink://analysis/metadata")
        if uri == "scilink://session/agents" and a_orch:
            return await read_resource("scilink://analysis/agents")

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


# ── Orchestrator initialization ──────────────────────────────────────────

def _init_orchestrators(state: dict, config: dict) -> None:
    """Initialize orchestrator(s) based on config."""
    import os
    from datetime import datetime
    from pathlib import Path

    # MCP server manages trust via client — skip interactive sandbox prompt
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

    session_dir = config["session_dir"]
    if not session_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = str(Path.home() / "scilink_mcp_sessions" / f"session_{ts}")
    Path(session_dir).mkdir(parents=True, exist_ok=True)

    api_key = config["api_key"]
    model_name = config["model_name"]
    base_url = config["base_url"]
    fh_key = config["futurehouse_api_key"] or os.environ.get("FUTUREHOUSE_API_KEY")

    if config["mode"] in ("analyze", "both"):
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent, AnalysisMode,
        )
        mode_map = {
            "co-pilot": AnalysisMode.CO_PILOT,
            "co_pilot": AnalysisMode.CO_PILOT,
            "copilot": AnalysisMode.CO_PILOT,
            "supervised": AnalysisMode.SUPERVISED,
            "autonomous": AnalysisMode.AUTONOMOUS,
        }
        analysis_mode = mode_map.get(
            config["analysis_mode"].lower(), AnalysisMode.AUTONOMOUS
        )
        state["analysis_orch"] = AnalysisOrchestratorAgent(
            base_dir=session_dir,
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            analysis_mode=analysis_mode,
            futurehouse_api_key=fh_key,
        )

    if config["mode"] in ("plan", "both"):
        try:
            from scilink.agents.planning_agents.planning_orchestrator import (
                PlanningOrchestratorAgent, AutonomyLevel,
            )
            autonomy_map = {
                "co-pilot": AutonomyLevel.CO_PILOT,
                "co_pilot": AutonomyLevel.CO_PILOT,
                "copilot": AutonomyLevel.CO_PILOT,
                "supervised": AutonomyLevel.SUPERVISED,
                "autonomous": AutonomyLevel.AUTONOMOUS,
            }
            autonomy_level = autonomy_map.get(
                config["analysis_mode"].lower(), AutonomyLevel.AUTONOMOUS
            )
            # Planning orchestrator needs data_dir and knowledge_dir
            # to avoid creating directories with relative paths
            # (fails when Claude Desktop runs from /).
            data_dir = str(Path(session_dir) / "data")
            knowledge_dir = str(Path(session_dir) / "kb_storage")
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            Path(knowledge_dir).mkdir(parents=True, exist_ok=True)
            state["planning_orch"] = PlanningOrchestratorAgent(
                base_dir=session_dir,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                futurehouse_api_key=fh_key,
                autonomy_level=autonomy_level,
                data_dir=data_dir,
                knowledge_dir=knowledge_dir,
            )
        except Exception as exc:
            logging.warning(f"Planning orchestrator not available: {exc}")


# ── Background job handlers ──────────────────────────────────────────────

def _handle_job_status(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Check the status of a background job."""
    job_id = arguments.get("job_id", "")
    job = state["jobs"].get(job_id)
    if job is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Unknown job_id: {job_id}",
            }),
        )]

    future = job["future"]
    if future.done():
        try:
            result = future.result()
            job["status"] = "completed"
            job["result"] = result
        except Exception as exc:
            job["status"] = "failed"
            job["result"] = json.dumps({
                "status": "error", "message": str(exc),
            })

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "job_id": job_id,
            "status": job["status"],
            "tool": job["tool"],
            "started_at": job["started_at"],
        }),
    )]


def _handle_job_result(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Retrieve the result of a completed background job."""
    job_id = arguments.get("job_id", "")
    job = state["jobs"].get(job_id)
    if job is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Unknown job_id: {job_id}",
            }),
        )]

    future = job["future"]
    if not future.done():
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "running",
                "job_id": job_id,
                "message": "Job is still running. Check again later with scilink_job_status.",
            }),
        )]

    # Ensure result is captured
    if job["result"] is None:
        try:
            job["result"] = future.result()
            job["status"] = "completed"
        except Exception as exc:
            job["result"] = json.dumps({
                "status": "error", "message": str(exc),
            })
            job["status"] = "failed"

    return [types.TextContent(type="text", text=job["result"])]


# ── Orchestrator chat handlers ──────────────────────────────────────────

def _execute_chat_captured(orch, prompt: str) -> str:
    """Execute orch.chat() while capturing stdout."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = orch.chat(prompt)

    log_output = captured.getvalue().strip()
    if log_output:
        logging.info(f"[orchestrate] {log_output}")

    return result


async def _handle_orchestrate(
    state: dict, name: str, arguments: dict, executor
) -> List["types.TextContent"]:
    """Handle orchestrator-level chat tools."""
    from datetime import datetime as _dt

    orch_key = (
        "analysis_orch" if name == "scilink_orchestrate_analysis"
        else "planning_orch"
    )
    orch = state.get(orch_key)
    if orch is None:
        mode_label = "analysis" if "analysis" in name else "planning"
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": (
                    f"{mode_label.title()} orchestrator is not available. "
                    f"Start the server with --mode {mode_label} or --mode both."
                ),
            }),
        )]

    prompt = arguments.get("prompt", "")
    if not prompt.strip():
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": "Empty prompt. Provide a natural-language instruction.",
            }),
        )]

    run_in_background = arguments.pop("background", False)
    tool_label = name.replace("scilink_", "")

    if run_in_background:
        state["job_counter"] += 1
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{ts}_{state['job_counter']:03d}"

        future = executor.submit(_execute_chat_captured, orch, prompt)
        state["jobs"][job_id] = {
            "future": future,
            "tool": tool_label,
            "started_at": ts,
            "status": "running",
            "result": None,
        }

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "started",
                "job_id": job_id,
                "tool": tool_label,
                "message": (
                    f"Orchestrator running in background (job {job_id}). "
                    "Use scilink_job_status to check progress, "
                    "then scilink_job_result to retrieve the result."
                ),
            }),
        )]

    result = await asyncio.to_thread(_execute_chat_captured, orch, prompt)
    return [types.TextContent(type="text", text=result)]


# ── Autonomy mode switch ─────────────────────────────────────────────────

def _handle_set_autonomy(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Switch autonomy mode at runtime."""
    new_mode = arguments.get("mode", "").strip().lower()
    valid = {"autonomous", "supervised", "co-pilot"}
    if new_mode not in valid:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": f"Invalid mode '{new_mode}'. Use: {', '.join(sorted(valid))}",
            }),
        )]

    old_mode = state["config"]["analysis_mode"]
    state["config"]["analysis_mode"] = new_mode

    # Clear any pending action from the previous mode
    state["pending_action"] = None

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "success",
            "previous_mode": old_mode,
            "current_mode": new_mode,
            "approval_required_for": sorted(
                _COPILOT_APPROVAL_TOOLS if new_mode == "co-pilot"
                else _SUPERVISED_APPROVAL_TOOLS if new_mode == "supervised"
                else set()
            ),
            "scilink_respond_needed": new_mode != "autonomous",
        }),
    )]


# ── Respond handler (co-pilot / supervised) ──────────────────────────────

async def _handle_respond(
    state: dict, arguments: dict
) -> List["types.TextContent"]:
    """Handle user response to a pending action."""
    pending = state.get("pending_action")
    if pending is None:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": "No pending action to respond to.",
            }),
        )]

    response = arguments.get("response", "").strip().lower()
    state["pending_action"] = None

    if response in ("yes", "y", "approve", "ok", "proceed"):
        orch_key = pending.context.get("orch_key", "analysis_orch")
        orch = state.get(orch_key)
        if orch is None:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": "Orchestrator not found.",
                }),
            )]

        result = await asyncio.to_thread(
            _execute_tool_captured, orch.tools, pending.tool_name, pending.kwargs
        )
        return [types.TextContent(type="text", text=result)]

    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "cancelled",
                "message": f"Action cancelled by user: {response}",
                "tool": pending.tool_name,
            }),
        )]


# ── Server runner ────────────────────────────────────────────────────────

async def run_stdio(server: Server, real_stdout=None) -> None:
    """Run the MCP server over stdio transport.

    Args:
        server: The configured MCP Server.
        real_stdout: The original ``sys.stdout`` before redirection.
            Needed because the CLI redirects ``sys.stdout`` to stderr
            to protect the JSON-RPC stream from stray ``print()`` calls.
    """
    _require_mcp()
    import anyio
    from io import TextIOWrapper

    stdout_arg = None
    if real_stdout is not None:
        stdout_arg = anyio.wrap_file(
            TextIOWrapper(real_stdout.buffer, encoding="utf-8")
        )

    async with stdio_server(stdout=stdout_arg) as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream,
            server.create_initialization_options(),
        )


def run_sse(server: Server, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the MCP server over SSE (Server-Sent Events) transport.

    Starts an HTTP server with two endpoints:

    - ``GET /sse`` — SSE stream for client connections
    - ``POST /messages/`` — message submission endpoint

    Args:
        server: The configured MCP Server.
        host: Bind address (default: ``127.0.0.1``).
        port: Bind port (default: ``8000``).
    """
    _require_mcp()

    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import Response
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            f"SSE transport requires additional packages: {exc}\n"
            "Install with: pip install uvicorn starlette sse-starlette"
        ) from exc

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )
        return Response()

    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ],
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ],
    )

    logging.info(f"SciLink MCP server (SSE) at http://{host}:{port}/sse")
    uvicorn.run(app, host=host, port=port, log_level="warning")
