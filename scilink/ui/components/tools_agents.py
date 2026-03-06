"""Tools & Agents tab — upload custom tools and view registered agents/tools."""

import importlib.util
import inspect
from pathlib import Path

import streamlit as st


def render_tools_agents_tab() -> None:
    """Render the Tools & Agents tab content."""
    agent = st.session_state.get("agent")
    app_mode = st.session_state.get("app_mode", "analyze")

    if agent is None:
        st.info("Start a session to view tools and agents.")
        return

    tools_col, agents_col = st.columns(2)

    with tools_col:
        _render_tools_section(agent, app_mode)

    with agents_col:
        _render_agents_section(agent, app_mode)


def _render_tools_section(agent, app_mode: str) -> None:
    """Upload custom tool files and list registered tools."""
    # Custom tool upload — only for analyze mode (planning doesn't support it yet)
    if app_mode == "analyze":
        st.subheader("Custom Tools")
        uploaded = st.file_uploader(
            "Upload a custom tool file (.py)",
            type=["py"],
            key="tool_file_uploader",
            accept_multiple_files=True,
            help=(
                "Python file with tool_schemas (OpenAI-format list) "
                "and create_tool_functions(data) factory."
            ),
        )

        if uploaded:
            for f in uploaded:
                upload_key = ("custom_tool", f.name)
                if upload_key in st.session_state._processed_uploads:
                    continue
                _load_tool_file(agent, f)
                st.session_state._processed_uploads.add(upload_key)

        # List registered external tools
        external_tools = getattr(agent, "_external_tools", [])
        if external_tools:
            st.markdown("**Registered custom tools:**")
            for t in external_tools:
                with st.expander(t["name"], expanded=False):
                    st.markdown(t.get("description", "No description."))
        else:
            st.caption("No custom tools registered yet.")

        st.divider()

    # List built-in orchestrator tools
    st.subheader("Built-in Tools")
    external_names = {
        t["name"] for t in getattr(agent, "_external_tools", [])
    }
    if hasattr(agent, "tools") and hasattr(agent.tools, "openai_schemas"):
        builtin = [
            td for td in agent.tools.openai_schemas
            if td.get("function", {}).get("name") not in external_names
        ]
        for td in builtin:
            fn = td.get("function", {})
            with st.expander(fn.get("name", "unknown"), expanded=False):
                st.markdown(fn.get("description", "No description."))
    else:
        st.caption("Agent tools not available.")


def _render_agents_section(agent, app_mode: str) -> None:
    """Display registered agents or sub-agents."""
    st.subheader("Available Agents")

    if app_mode == "analyze":
        _render_analysis_agents(agent)
    elif app_mode == "plan":
        _render_planning_agents(agent)


def _render_analysis_agents(agent) -> None:
    """Show analysis agent registry."""
    registry = getattr(agent, "_agent_registry", {})
    if not registry:
        st.caption("No agents registered.")
        return

    selected_id = getattr(agent, "selected_agent_id", None)

    for agent_id in sorted(registry.keys()):
        entry = registry[agent_id]
        name = entry.get("name", f"Agent {agent_id}")
        desc = entry.get("description", "")
        short = entry.get("short_name", "")
        is_selected = agent_id == selected_id

        with st.expander(f"[{agent_id}] {short} — {name}", expanded=is_selected):
            if is_selected:
                st.success("Currently active")
            st.markdown(desc if desc else "No description available.")


def _render_planning_agents(agent) -> None:
    """Show planning sub-agents."""
    sub_agents = [
        ("PlanningAgent", "planner",
         "Hypothesis generation, literature-backed experimental design, and RAG-powered knowledge retrieval."),
        ("ScalarizerAgent", "scalarizer",
         "Converts raw experimental results into scalar objective values for optimization."),
        ("BOAgent", "bo",
         "Bayesian optimization for iterative experimental parameter tuning."),
    ]

    for name, attr, desc in sub_agents:
        instance = getattr(agent, attr, None)
        status = "Initialized" if instance is not None else "Not initialized"
        with st.expander(name, expanded=False):
            st.markdown(desc)
            st.caption(f"Status: {status}")


def _load_tool_file(agent, uploaded_file) -> None:
    """Load a custom tool .py file and register it with the agent."""
    # Save to session dir
    session_dir = st.session_state.get("session_dir")
    if session_dir is None:
        st.error("No active session.")
        return

    tools_dir = Path(session_dir) / "custom_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    dest = tools_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())

    # Load module
    try:
        spec = importlib.util.spec_from_file_location(dest.stem, dest)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        st.error(f"Failed to load {uploaded_file.name}: {e}")
        return

    # Discover schemas
    schemas = (
        getattr(module, "tool_schemas", None)
        or getattr(module, "openai_schemas", None)
    )
    if schemas is None:
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if (
                isinstance(obj, list)
                and obj
                and isinstance(obj[0], dict)
                and obj[0].get("type") == "function"
            ):
                schemas = obj
                break

    if not schemas:
        st.error(
            f"No tool schemas found in {uploaded_file.name}. "
            "Define `tool_schemas` as a list of OpenAI-format tool dicts."
        )
        return

    # Discover factory
    factory = getattr(module, "create_tool_functions", None)
    if factory is None:
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.endswith("_tool_functions") and fn.__module__ == module.__name__:
                factory = fn
                break

    if factory is None:
        st.error(
            f"No factory function found in {uploaded_file.name}. "
            "Define `create_tool_functions(data)` returning "
            "a dict mapping tool names to callables."
        )
        return

    # Register with the orchestrator
    try:
        agent.register_tools(schemas, factory)
        count = sum(1 for s in schemas if s.get("type") == "function")
        st.success(f"Registered {count} tool(s) from {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to register tools: {e}")
