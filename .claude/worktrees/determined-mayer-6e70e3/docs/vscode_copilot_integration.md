# Using SciLink from VS Code (GitHub Copilot)

SciLink can run as an MCP server inside VS Code, making its full set of analysis and planning tools available directly from GitHub Copilot Chat. This guide walks through the setup for a user who has installed SciLink and wants to use it from any VS Code workspace.

## Prerequisites

- **VS Code** with **GitHub Copilot** installed and signed in
- **Python 3.12+** with SciLink installed
- An LLM API key (Anthropic, Google, or OpenAI) for SciLink's internal agents

## Step 1: install SciLink

```bash
pip install scilink
```

Or from source (editable, for development):

```bash
pip install -e .
```

MCP support is included in the base install — no extras needed.

## Step 2: find your scilink binary path

```bash
which scilink
```

You will get something like:

```
/Users/yourname/miniconda3/envs/scilink/bin/scilink
```

Copy this path — you will need it in the next step. Using the absolute path (rather than just `scilink`) is important because VS Code launched from Finder or the Dock does not reliably inherit your shell's PATH.

## Step 3: configure the MCP server

Open the command palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux) and search for `MCP: Open User Configuration`. This opens your user-scope `mcp.json`, which applies to every VS Code workspace.

Add the following configuration (replace the `command` path with the output from Step 2):

```json
{
    "servers": {
        "scilink": {
            "type": "stdio",
            "command": "/absolute/path/to/your/env/bin/scilink",
            "args": [
                "serve",
                "--mode", "both",
                "--autonomy", "autonomous",
                "--model", "claude-opus-4-6"
            ],
            "env": {
                "ANTHROPIC_API_KEY": "${input:anthropic_api_key}"
            }
        }
    },
    "inputs": [
        {
            "id": "anthropic_api_key",
            "type": "promptString",
            "description": "Anthropic API key for SciLink MCP server",
            "password": true
        }
    ]
}
```

### Configuration explained

- **`command`**: The absolute path to your `scilink` binary. Using the full path avoids PATH-resolution issues when VS Code is launched from the OS app launcher rather than a terminal.
- **`args`**: The `serve` subcommand starts SciLink's MCP server. `--mode both` exposes both analysis and planning tools (use `analyze` or `plan` for just one set). `--autonomy autonomous` lets the LLM chain tools without pausing for approval. `--model` sets the LLM that SciLink's internal agents use (see "Choosing a model" below).
- **`env`**: API keys for SciLink's LLM backend. The `${input:...}` syntax tells VS Code to prompt you for the key on first use and store it securely in the system keychain — the key is never written to disk. Swap `ANTHROPIC_API_KEY` for `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `SCILINK_API_KEY` depending on your provider.
- **`inputs`**: Defines the secure prompt that `${input:anthropic_api_key}` references. Set `password: true` so VS Code masks the input field.

### Why not use the "Add MCP Server" button?

VS Code's "Add MCP Server" UI generates a placeholder config and writes it by replacing the entire `servers` block rather than merging. If you already have other MCP servers registered, it will remove them. Edit `mcp.json` directly through `MCP: Open User Configuration` instead.

### User scope vs workspace scope

The configuration above is user-scoped — it applies to every VS Code window you open. This is what you want for day-to-day use: open a data folder in VS Code, fire up Copilot Chat, and SciLink's tools are already there.

If you need a different SciLink configuration for a specific project (e.g., a different model, a different autonomy level), create a `.vscode/mcp.json` in that project's root with the project-specific overrides. Workspace-scoped configs take precedence when that workspace is open.

## Step 4: start the server and verify

1. Reload VS Code (`Cmd+Shift+P` → `Developer: Reload Window`) so it picks up the new config.
2. The first time the `scilink` server starts, VS Code pops a secure input field at the top of the window asking for your API key. Paste it in. VS Code stores it in the system keychain and reuses it on subsequent launches.
3. Check the server status: `Cmd+Shift+P` → `MCP: List Servers`. The `scilink` entry should show status **Running** and a tool count (typically 40-50 tools depending on the mode).
4. If the server shows an error, expand its entry to see the stderr output. Common issues are covered in Troubleshooting below.

### First-start timing

The first server start can take 30-40 seconds while SciLink initializes its orchestrators and sub-agents. VS Code will show "Waiting for server to respond to `initialize` request..." during this time — this is normal. The server will respond once initialization completes.

## Step 5: use SciLink from Copilot Chat

1. Open **Copilot Chat** (the chat icon in the sidebar, or `Cmd+Shift+I`).
2. **Switch the chat mode to Agent, Claude, or Codex.** This is the critical step. MCP tools are only available in these modes — in Ask or Edit modes, Copilot cannot call SciLink tools and will not know they exist.
3. Chat naturally. Copilot will decide on its own when to call SciLink tools based on your request.

### Example prompts

```
Examine /path/to/my_data/afm_scan.npy and tell me what kind of data it is.
```

```
This is XPS Ti 2p data from a TiO2 thin film. Load the metadata and run
curve fitting analysis with the xps skill.
```

```
Generate an experimental plan for optimizing lithium extraction from brine,
using the papers in ./literature/ as the knowledge base.
```

Copilot will call the appropriate SciLink tools (`scilink_examine_data`, `scilink_run_analysis`, `scilink_generate_initial_plan`, etc.) and weave the results into its response. You will see each tool call appear in the chat with its arguments and return values.

### Individual tools vs orchestrator

SciLink exposes its tools at two levels:

- **Individual tools** (`scilink_examine_data`, `scilink_run_analysis`, etc.) — Copilot calls them one by one and drives the workflow itself. Good for simple tasks or when you want to see and control each step.
- **Orchestrator tools** (`scilink_orchestrate_analysis`, `scilink_orchestrate_planning`) — send a single natural-language prompt and SciLink's own orchestrator handles the entire multi-step workflow internally, using its domain-specific system prompt to decide which agents and tools to use. Best for complex analyses where SciLink's domain expertise should drive the flow.

Example using the orchestrator:

```
Use scilink_orchestrate_analysis to analyze the XPS data at ./data/ti2p.csv
with the xps skill, then assess novelty of the findings. Use background=true.
```

### Long-running analyses

SciLink's heavy tools (`run_analysis`, `run_optimization`, `assess_novelty`, `generate_initial_plan`, `generate_implementation_code`, `get_recommendations`, `run_economic_analysis`) support an optional `background=true` parameter. When used, the tool returns a job ID immediately instead of blocking, and Copilot polls with `scilink_job_status` / `scilink_job_result` to retrieve the result. This avoids tool-call timeouts in VS Code. You can nudge this explicitly:

```
Run analysis on the current data with background=true
```

## Choosing a model

The `--model` flag in the `args` array controls which LLM SciLink's internal agents use. This is independent of Copilot's own model (which you select in the Copilot Chat model picker). Common choices:

| Model | Notes |
|---|---|
| `claude-opus-4-6` | Most capable, best for complex scientific reasoning. |
| `claude-sonnet-4-6` | Faster and cheaper, good for routine analyses. |
| `gemini-3.1-pro-preview` | Default if `--model` is omitted. Requires `GEMINI_API_KEY`. |

The model name must match what your API key supports. If you set `--model claude-opus-4-6`, your `env` block must include `ANTHROPIC_API_KEY`. If you set a Gemini model, provide `GEMINI_API_KEY` instead.

To change the model, edit `mcp.json` via `MCP: Open User Configuration` and restart the server.

## Session outputs

Each server session creates a timestamped directory under `~/scilink_mcp_sessions/`. Analysis runs produce:

- Fitted curves and plots (PNG)
- `analysis_results.json` with detailed findings
- `metadata_used.json` with the metadata snapshot
- Scientific claims and HTML reports

## Troubleshooting

### Server shows "Stopped" and won't start

Open the command palette → `MCP: List Servers` → click `scilink` → Start. If it immediately stops again, expand the entry to see its stderr output for the actual error.

### "Failed to parse message" warnings with the SciLink logo

The `args` array is missing `"serve"`. Without it, `scilink` prints its CLI help (including the ASCII logo) to stdout, which corrupts the MCP transport. Make sure `"serve"` is the first entry in the `args` array.

### Server takes 30-40 seconds to initialize

This is normal on first start — SciLink initializes both orchestrators and all sub-agents during this time. VS Code will show "Waiting for server to respond to `initialize` request..." while it waits. If it exceeds 60 seconds and VS Code gives up, try restarting the server — subsequent starts are typically faster.

### API key prompt doesn't appear

The `${input:...}` mechanism requires a matching entry in the `inputs` array. Verify that the `id` in your `inputs` entry matches the name inside `${input:...}` in the `env` block. To force a re-prompt after the key has been cached: `Cmd+Shift+P` → `MCP: Reset Cached Inputs`.

### "No SciLink tools available" in Copilot Chat

Copilot Chat must be in **Agent**, **Claude**, or **Codex** mode. MCP tools are not visible in Ask or Edit modes. Check the mode selector in the chat input area and switch to one of the supported modes.

Also verify the server is actually running (`MCP: List Servers` → check status). If the server is still initializing, Copilot won't see the tools yet — wait for init to complete.

### Tools work but analysis feels too fast or uses the wrong model

SciLink echoes which model each agent is using during server init (visible in the server's stderr output):

```
🌐 Orchestrator using LiteLLM: claude-opus-4-6
🌐 PlanningAgent using LiteLLM: claude-opus-4-6
```

Verify these match your intended model. If you suspect a mismatch, add `"LITELLM_LOG": "DEBUG"` to the `env` block temporarily — LiteLLM will log the exact model string on every API call. Remove it after verifying, as the output is verbose.

## Further reading

- [Full MCP server guide](claude_code_integration.md) — complete tool reference, autonomy modes, Claude Code / Claude Desktop setup
- [Custom tools guide](custom_tools_integration.md) — adding your own tools to SciLink
- [MCP client guide](mcp_client_integration.md) — connecting SciLink to external MCP servers (arXiv, OpentronsAI, etc.)
