"""Tools tab — upload custom tools, connect MCP servers, and view registered tools."""

import importlib.util
import inspect
import sys
from pathlib import Path

import streamlit as st


def render_tools_agents_tab() -> None:
    """Render the Tools tab content."""
    agent = st.session_state.get("agent")

    if agent is None:
        st.info("Start a session to view tools.")
        return

    left_col, _, right_col = st.columns([10, 1, 10])

    with left_col:
        _render_upload_section(agent)
        st.divider()
        _render_mcp_section(agent)

    with right_col:
        _render_custom_tools(agent)
        st.divider()
        _render_builtin_tools(agent)


def _render_upload_section(agent) -> None:
    """Upload custom tool files."""
    st.subheader("Upload Tools")
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


def _render_custom_tools(agent) -> None:
    """Show registered external tools."""
    st.subheader("Registered External Tools")
    external_tools = getattr(agent, "_external_tools", [])
    if external_tools:
        for t in external_tools:
            with st.expander(t["name"], expanded=False):
                st.markdown(t.get("description", "No description."))
    else:
        st.caption("No custom tools registered yet.")


def _render_builtin_tools(agent) -> None:
    """Show built-in orchestrator tools."""
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


def _load_tool_file(agent, uploaded_file) -> None:
    """Load a custom tool .py file and register it with the agent."""
    session_dir = st.session_state.get("session_dir")
    if session_dir is None:
        st.error("No active session.")
        return

    tools_dir = Path(session_dir) / "custom_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    dest = tools_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())

    # Load module (suppress __pycache__ in session directory)
    try:
        spec = importlib.util.spec_from_file_location(dest.stem, dest)
        module = importlib.util.module_from_spec(spec)
        _prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = _prev
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


def _render_mcp_section(agent) -> None:
    """Connect/disconnect MCP servers and list their tools."""
    st.subheader("MCP Servers")

    mcp_connections = getattr(agent, "_mcp_connections", {})

    # Connection form
    transport = st.radio(
        "Transport", ["stdio", "sse"], horizontal=True, key="mcp_transport"
    )
    col_name, col_addr = st.columns([1, 2])
    with col_name:
        mcp_name = st.text_input("Server name", key="mcp_name")
    with col_addr:
        if transport == "stdio":
            mcp_addr = st.text_input(
                "Command",
                placeholder="npx -y @modelcontextprotocol/server-name /path",
                key="mcp_addr",
            )
        else:
            mcp_addr = st.text_input(
                "URL",
                placeholder="http://localhost:8080/sse",
                key="mcp_addr",
            )

    col_connect, _ = st.columns([1, 2])
    with col_connect:
        connect_clicked = st.button("Connect", key="mcp_connect_btn", type="primary", width="stretch")
    if connect_clicked:
        if not mcp_name or not mcp_addr:
            st.warning("Provide both a server name and a command/URL.")
        elif mcp_name in mcp_connections:
            st.warning(f"'{mcp_name}' is already connected.")
        else:
            try:
                with st.spinner(f"Connecting to '{mcp_name}'..."):
                    if transport == "stdio":
                        count = agent.connect_mcp_server(
                            mcp_name, command=mcp_addr.split()
                        )
                    else:
                        count = agent.connect_mcp_server(
                            mcp_name, url=mcp_addr
                        )
                st.success(f"Connected to '{mcp_name}': {count} tool(s)")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to connect: {e}")

    # Show connected servers
    if mcp_connections:
        for name, conn in list(mcp_connections.items()):
            with st.expander(f"MCP: {name}", expanded=False):
                tool_count = len(conn.tool_schemas) if hasattr(conn, "tool_schemas") else 0
                col_info, col_btn = st.columns([5, 1])
                with col_info:
                    st.caption(
                        f"Transport: {'stdio' if conn.command else 'sse'} · "
                        f"{tool_count} tool(s)"
                    )
                with col_btn:
                    if st.button(
                        "\u2715", key=f"mcp_disconnect_{name}",
                        help=f"Disconnect {name}",
                    ):
                        agent.disconnect_mcp_server(name)
                        st.rerun()
    else:
        st.caption("No MCP servers connected.")
