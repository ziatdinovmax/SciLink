# Connecting SciLink to External MCP Servers

SciLink can connect to external MCP (Model Context Protocol) servers and use their tools alongside its built-in analysis and planning tools. This lets you extend SciLink with capabilities like literature search, database queries, file system access, or domain-specific tools — without writing any code.

## How it works

When you connect an MCP server, SciLink:

1. Discovers all tools the server exposes
2. Registers them in the orchestrator's tool loop (prefixed with `[MCP:server_name]`)
3. Updates the system prompt so the LLM knows the new tools exist
4. Routes tool calls to the MCP server transparently

The external tools are available to the LLM alongside built-in tools like `examine_data` and `run_analysis`.

## Three ways to connect

### 1. CLI (`--mcp` flag)

```bash
# stdio transport (spawns a subprocess)
scilink analyze --mcp stdio:arxiv:python,-m,arxiv_mcp_server,--storage-path,/tmp/papers

# SSE transport (connects to a running server)
scilink analyze --mcp sse:myserver:http://localhost:8000/sse

# JSON config file
scilink analyze --mcp /path/to/mcp_config.json

# Multiple servers
scilink analyze --mcp stdio:arxiv:python,-m,arxiv_mcp_server stdio:fs:npx,-y,@modelcontextprotocol/server-filesystem,/tmp
```

**Shorthand format:**
- stdio: `stdio:<name>:<command>,<arg1>,<arg2>`
- SSE: `sse:<name>:<url>`

**JSON config file format:**
```json
{
  "name": "arxiv",
  "command": ["python", "-m", "arxiv_mcp_server", "--storage-path", "/tmp/papers"],
  "env": {"SOME_API_KEY": "..."}
}
```

Or for SSE:
```json
{
  "name": "myserver",
  "url": "http://localhost:8000/sse"
}
```

### 2. UI (Tools & Agents tab)

In the Streamlit UI, go to the **Tools & Agents** tab and find the **MCP Servers** section:

1. Select transport type (**stdio** or **sse**)
2. Enter a **server name** (any label you choose)
3. Enter the **command** (for stdio) or **URL** (for SSE)
4. Click **Connect**

Connected servers and their tool counts appear below. Click **Disconnect** to remove a server.

### 3. Programmatic

```python
from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent

orch = AnalysisOrchestratorAgent(...)

# stdio transport
count = orch.connect_mcp_server(
    "arxiv",
    command=["python", "-m", "arxiv_mcp_server", "--storage-path", "/tmp/papers"],
)

# SSE transport
count = orch.connect_mcp_server(
    "myserver",
    url="http://localhost:8000/sse",
)

# With environment variables (stdio only)
count = orch.connect_mcp_server(
    "chembl",
    command=["python", "-m", "chembl_mcp"],
    env={"CHEMBL_API_KEY": "..."},
)

print(f"Registered {count} tools")

# Disconnect when done
orch.disconnect_mcp_server("arxiv")

# Disconnect all
orch.disconnect_all_mcp_servers()
```

## Example: arXiv paper search

```bash
pip install arxiv-mcp-server
scilink analyze --mcp stdio:arxiv:python,-m,arxiv_mcp_server,--storage-path,/tmp/papers
```

Then in the chat:
```
Search arXiv for recent papers on XPS peak fitting of transition metal oxides
```

The LLM will use the arXiv MCP tools to search and retrieve papers, then use SciLink's built-in tools to analyze your data — combining literature context with experimental analysis.

## Example: OpentronsAI protocol generation

Connect to the OpentronsAI MCP server (hosted on Hugging Face) to generate Opentrons protocols:

```bash
# CLI
scilink plan --mcp stdio:OpentronsAI:npx,mcp-remote,https://opentrons-opentronsai-mcp-server.hf.space/gradio_api/mcp/
```

```python
# Python API
agent.connect_mcp_server(
    "OpentronsAI",
    command=["npx", "mcp-remote",
             "https://opentrons-opentronsai-mcp-server.hf.space/gradio_api/mcp/"]
)
```

In the UI, use the **Tools & Agents** tab:
1. Transport: **stdio**
2. Server name: **OpentronsAI**
3. Command: `npx mcp-remote https://opentrons-opentronsai-mcp-server.hf.space/gradio_api/mcp/`

## How tools appear to the LLM

External tools are registered with a `[MCP:name]` prefix in their description so the LLM can distinguish them from built-in tools. For example:

```
Built-in:  examine_data — Examine a data file to determine its type...
External:  [MCP:arxiv] search_papers — Search arXiv for papers matching a query...
```

If an external tool name collides with a built-in tool, it's automatically prefixed with the server name (e.g., `arxiv_search` instead of `search`).

## Requirements

```bash
pip install scilink
```

The `mcp` package is included in the default SciLink installation.
