# Using SciLink as an MCP Server

SciLink can run as an MCP (Model Context Protocol) server, making its analysis and planning tools available to any MCP client — Claude Code, Claude Desktop, Cursor, or any other compatible tool.

## Setup

### 1. Install SciLink with MCP support

```bash
pip install -e .                 # includes MCP support
pip install -e ".[ui]"           # MCP + Streamlit UI
pip install -e ".[all]"          # everything (UI + simulation)
```

### 2a. Claude Code — register the MCP server

```bash
claude mcp add scilink -s user \
  -e GEMINI_API_KEY=your-gemini-key \
  -e UNSAFE_EXECUTION_OK=true \
  -e "PATH=$(dirname $(which python)):$PATH" \
  -- $(which scilink) serve --mode analyze
```

**Flags explained:**
- `-s user` — available from any directory (use `-s project` to limit to one repo)
- `-e GEMINI_API_KEY=...` — API key for the LLM backend (also supports `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- `-e UNSAFE_EXECUTION_OK=true` — allows code execution without a sandbox prompt (required since there's no terminal to approve interactively)
- `-e PATH=...` — ensures the `python` command is found during code execution
- `--mode analyze` — expose analysis tools (use `plan` for planning tools, or `both` for all)

Restart Claude Code after adding.

### 2b. Claude Desktop — edit config

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "scilink": {
      "command": "/path/to/scilink",
      "args": ["serve", "--mode", "analyze"],
      "env": {
        "GEMINI_API_KEY": "your-key",
        "UNSAFE_EXECUTION_OK": "true",
        "PATH": "/path/to/python/bin:/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```

Replace `/path/to/scilink` with the full path (find it with `which scilink`).
Restart Claude Desktop after editing.

### 2c. SSE transport — any MCP client

Start the server as a long-lived HTTP service:

```bash
scilink serve --mode analyze --transport sse --port 8000
```

Then connect your MCP client to `http://127.0.0.1:8000/sse`.

### 2d. Direct command

SciLink also installs a `scilink-mcp` entry point that goes directly to the server:

```bash
scilink-mcp --mode analyze                         # stdio (default)
scilink-mcp --mode both --transport sse --port 8000  # SSE
```

## Available tools

### Analysis tools

| Tool | Description |
|------|-------------|
| `scilink_examine_data` | Inspect a data file (type, shape, suggested agents) |
| `scilink_convert_metadata` | Convert a text description to structured metadata |
| `scilink_load_metadata` | Load metadata from a JSON file or directory |
| `scilink_select_agent` | Choose an analysis agent (FFT, SAM, Hyperspectral, CurveFitting) |
| `scilink_preview_image` | Preview a microscopy image |
| `scilink_run_analysis` | Run analysis with the selected agent |
| `scilink_list_results` | List completed analyses in the session |
| `scilink_get_recommendations` | Get follow-up experiment suggestions |
| `scilink_assess_novelty` | Literature search for novelty of findings |
| `scilink_synthesize_knowledge` | Build reusable knowledge from results |
| `scilink_save_checkpoint` | Save session state |
| `scilink_show_available_agents` | List available analysis agents |
| `scilink_set_preprocessing_instruction` | Add custom preprocessing steps |
| `scilink_get_metadata_schema` | View required/optional metadata fields |
| `scilink_list_knowledge` | List active knowledge entries |
| `scilink_clear_knowledge` | Remove knowledge entries |
| `scilink_graduate_to_skill` | Convert a knowledge entry into a reusable skill |
| `scilink_update_skill` | Update an existing graduated skill with new knowledge |
| `scilink_save_file` | Save text content to a file in the session directory |

### Planning tools

Available with `--mode plan` or `--mode both`:

| Tool | Description |
|------|-------------|
| `scilink_list_workspace_files` | List files in the session directory |
| `scilink_generate_initial_plan` | Generate an experimental plan |
| `scilink_generate_implementation_code` | Add implementation code to a plan |
| `scilink_run_economic_analysis` | Techno-economic analysis |
| `scilink_refine_plan_with_results` | Refine plan based on results |
| `scilink_refine_implementation_code` | Update implementation code |
| `scilink_analyze_file` | Analyze a result file for optimization |
| `scilink_analyze_batch` | Batch analysis for optimization |
| `scilink_reset_analysis_logic` | Reset the scalarizer |
| `scilink_run_optimization` | Run Bayesian optimization |
| `scilink_plan_save_checkpoint` | Save planning session state |
| `scilink_discard_plan` | Discard the current plan |
| `scilink_show_directory_guide` | Show project directory structure |
| `scilink_plan_read_file` | Read a file from the workspace |
| `scilink_adjust_plan_for_constraints` | Adjust plan for implementation constraints |
| `scilink_plan_save_file` | Save text content to the session directory |
| `scilink_plan_synthesize_knowledge` | Distill findings from planning iterations into knowledge |
| `scilink_plan_list_knowledge` | List active knowledge entries |
| `scilink_plan_clear_knowledge` | Remove knowledge entries |
| `scilink_plan_graduate_to_skill` | Convert knowledge into a reusable skill |
| `scilink_plan_update_skill` | Update an existing graduated skill |

### Session management tools

| Tool | Description |
|------|-------------|
| `scilink_set_autonomy` | Switch between autonomous/supervised/co-pilot modes at runtime |
| `scilink_respond` | Approve or reject a pending action (co-pilot/supervised modes) |
| `scilink_job_status` | Check status of a background job |
| `scilink_job_result` | Retrieve result of a completed background job |

### Orchestrator tools

| Tool | Description |
|------|-------------|
| `scilink_orchestrate_analysis` | Delegate a complete analysis workflow to SciLink's analysis orchestrator via natural language |
| `scilink_orchestrate_planning` | Delegate a complete planning workflow to SciLink's planning orchestrator via natural language |

These tools wrap the full orchestrator chat loop. Instead of calling individual tools one by one (examine_data, select_agent, run_analysis, etc.), send a single natural-language prompt and the orchestrator handles the entire multi-step workflow using its domain-specific system prompt. Use `background=true` for non-trivial requests — the orchestrator may chain several internal tool calls and take minutes to complete.

## Usage examples

Just chat naturally. The LLM calls SciLink tools automatically.

### Analyze a spectrum

```
Examine /path/to/xps_data.csv and tell me what kind of data it is
```

### Set metadata and run analysis

```
This is XPS Ti 2p data from a TiO2 thin film collected with Al K-alpha at 1486.6 eV.
Load the metadata, select the curve fitting agent, and analyze with the xps skill.
```

### Run analysis in background (avoids timeouts)

```
Run analysis on the current data with background=true
```

Then the LLM will poll with `scilink_job_status` and retrieve with `scilink_job_result`.

### Analyze a microscopy image

```
Examine /path/to/sem_image.tif, preview it, and run particle segmentation
```

### Batch analysis

```
Examine the directory /path/to/spectra/ and run a series analysis
with temperature as the control variable
```

### Get follow-up suggestions

```
List my analysis results and get measurement recommendations for the most recent one
```

## Server options

```bash
scilink serve --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gemini-3.1-pro-preview` | LLM model for analysis agents |
| `--mode` | `both` | `analyze`, `plan`, or `both` |
| `--autonomy` | `autonomous` | `autonomous`, `supervised`, or `co-pilot` |
| `--transport` | `stdio` | `stdio` or `sse` |
| `--host` | `127.0.0.1` | Bind address (SSE only) |
| `--port` | `8000` | Bind port (SSE only) |
| `--session-dir` | auto-generated | Directory for session outputs |
| `--api-key` | from env vars | Override API key |
| `--base-url` | none | OpenAI-compatible endpoint |
| `--futurehouse-key` | from env vars | FutureHouse/Edison API key |

## Autonomy modes

Control how much approval SciLink requires:

```bash
scilink serve --autonomy autonomous   # default — all tools run immediately
scilink serve --autonomy supervised   # high-impact tools pause for approval
scilink serve --autonomy co-pilot     # most tools pause for approval
```

You can also switch at runtime by asking the LLM to call `scilink_set_autonomy`.

In **supervised** and **co-pilot** modes, high-impact tools return a `needs_input` response instead of executing. The MCP client calls `scilink_respond` with `"yes"` to approve or `"no"` to cancel.

Tools that require approval:
- **Co-pilot**: `run_analysis`, `select_agent`, `assess_novelty`, `get_recommendations`, `run_optimization`, `generate_initial_plan`, `generate_implementation_code`, `run_economic_analysis`, `discard_plan`
- **Supervised**: `run_analysis`, `run_optimization`, `discard_plan`

## Background execution

Long-running tools support an optional `background=true` parameter that returns a job ID immediately instead of blocking. This avoids timeouts in clients like Claude Desktop.

Tools that support background execution:
- `run_analysis`, `run_optimization` — full agent analysis or Bayesian optimization
- `assess_novelty` — FutureHouse literature search per scientific claim
- `get_recommendations` — measurement recommendations over a full analysis record
- `generate_initial_plan`, `generate_implementation_code` — RAG + LLM generation
- `run_economic_analysis` — technoeconomic analysis with knowledge retrieval
- `orchestrate_analysis`, `orchestrate_planning` — full orchestrator chat loops

```
run_analysis(data_path="...", background=true)
→ {"status": "started", "job_id": "job_20260308_130923_001"}

job_status(job_id="job_20260308_130923_001")
→ {"status": "running"} or {"status": "completed"}

job_result(job_id="job_20260308_130923_001")
→ full analysis result
```

## Session outputs

Results are saved to `~/scilink_mcp_sessions/session_<timestamp>/`. Each analysis run creates a subdirectory with:
- Fitted curves and plots
- `analysis_results.json` with detailed findings
- Scientific claims and recommendations

## Troubleshooting

### Server not connecting
Verify the server works standalone:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | scilink serve --mode analyze 2>/dev/null
```

### `python` not found during analysis
Ensure the `PATH` env var in the MCP config includes the directory containing `python`:
```bash
-e "PATH=$(dirname $(which python)):$PATH"
```

### Analysis times out (Claude Desktop)
Claude Desktop has a ~4 minute timeout on tool calls. Use `background=true` for long-running analyses, or use Claude Code which has no timeout.

### Tools not appearing
- Run `claude mcp list` to verify the server is registered
- Make sure `scilink serve` is available (`pip install -e .`)
- Restart the client after adding the MCP server

### Managing the server (Claude Code)
```bash
claude mcp list              # list configured servers
claude mcp get scilink       # show config details
claude mcp remove scilink    # remove the server
```
