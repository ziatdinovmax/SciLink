# Connecting SciLink to agent clients — one guide for all of them

SciLink serves its orchestrators and specialist agents over the
[Model Context Protocol](https://modelcontextprotocol.io). Any MCP-capable
client — Claude Code, Claude Desktop, Deep Agents Code (`dcode`), VS Code
Copilot, or your own agent built on `langchain-mcp-adapters` / any MCP SDK —
connects the same way. This page is the unified instruction: three ways to
run the server, one config recipe per client, and the container layouts.

## The invariants (true in every setup)

- **Credentials live with the server, once.** Put `KEY=VALUE` lines (e.g.
  `ANTHROPIC_API_KEY=...`, or `AWS_BEARER_TOKEN_BEDROCK=...` +
  `AWS_REGION_NAME=...`) in `~/.scilink/credentials.env` *of the machine or
  container that runs `scilink serve`*. The server loads it at startup
  (explicitly set environment variables win). Client configs never need
  secrets.
- **Paths are server-side.** Every `file_path` / `data_path` you pass to a
  `scilink_*` tool must be a path the *server* process can open. Same
  machine: any absolute path. Server in a container: put data on the shared
  volume and pass `/data/...`. Read results through the schema fields in
  tool responses (`feature_columns`, `feature_tables_schema`) instead of
  opening server files.
- **One server = one campaign at a time.** The orchestrators are stateful.
  Give each campaign its own `--session-dir`; state (and background jobs)
  survive server restarts — a fixed session dir resumes the campaign.
- **SSE has no auth.** Bind to `127.0.0.1` or keep it on a private
  network / behind an authenticating proxy.
- **Generated code runs on the server.** SciLink executes LLM-written
  analysis code; run the server inside a container or VM when that
  capability needs bounding (`UNSAFE_EXECUTION_OK=true` acknowledges the
  environment is the sandbox).
- **While tools run**, the server streams narration to the client as MCP
  log notifications (lines marked `🔬 SciLink │ …`); GUI clients that
  support MCP icons also show the SciLink logo on the server and every
  tool. Terminal clients that render neither can watch the server's own
  terminal/log.

## Three ways to run the server

**A. Spawned per session (stdio).** The client launches `scilink serve`
itself. Generate the config instead of writing it:

```bash
scilink serve --print-mcp-json --mode both --session-dir ~/scilink_sessions/mycampaign
```

This prints a secret-free `mcpServers` entry — a **zero-install** `uvx`
spec when [uv](https://docs.astral.sh/uv/) is on PATH (the machine needs
nothing else), or a path to this installation's `scilink` otherwise.

`--mode` selects which tool set the server exposes: `analyze` (data
analysis), `plan` (optimization + experiment planning), `both` (the
default — analysis + planning together), `simulate` (structure building,
engine inputs, and running DFT / MD / ML-potential simulations), or `meta`
(a single orchestrator that routes across all of them). Pick the narrowest
mode your campaign needs; `meta` is the widest surface.

**B. Long-lived service on the host (SSE).** Start it in a terminal you
keep visible (all narration appears there) and point clients at the URL:

```bash
scilink serve --mode both --transport sse --host 127.0.0.1 --port 8000 \
              --session-dir ~/scilink_sessions/mycampaign
# clients use: {"type": "sse", "url": "http://127.0.0.1:8000/sse"}
```

`scilink serve --print-mcp-json --transport sse --port 8000` prints that
client entry too.

**C. In a container (SSE sidecar).** Because the server executes generated
code, a container is the recommended production shape:

```dockerfile
# Dockerfile.scilink
FROM python:3.12-slim
RUN useradd --create-home scilink
USER scilink
RUN pip install --no-cache-dir --user "scilink>=0.0.65" uvicorn starlette sse-starlette
ENV PATH="/home/scilink/.local/bin:${PATH}" UNSAFE_EXECUTION_OK=true
VOLUME /data
EXPOSE 8000
ENTRYPOINT ["scilink", "serve"]
CMD ["--mode", "both", "--transport", "sse", "--host", "0.0.0.0", \
     "--port", "8000", "--session-dir", "/data/session"]
```

Mount the credentials file at `/home/scilink/.scilink/credentials.env`
(or pass provider variables as container env), and mount `./data:/data`
for datasets + session state.

## Per-client recipes

All of these consume the output of `--print-mcp-json` (option A) or the
SSE URL (options B/C); differences are only where the config lives.

| Client | Where the config goes |
|---|---|
| **Claude Code** (CLI) | `claude mcp add scilink -- <command …>` or paste the entry into `.mcp.json` at the project root |
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) / `%APPDATA%\Claude\claude_desktop_config.json` (Windows); renders the SciLink logo |
| **Deep Agents Code** (`dcode`) | `~/.deepagents/.mcp.json` (user-level, trusted) or `<project>/.mcp.json` (needs one-time approval for stdio) |
| **VS Code Copilot** | `MCP: Open User Configuration` → `mcp.json` (outer key is `servers`, not `mcpServers`); see `docs/vscode_copilot_integration.md` for the keychain-input pattern |
| **deepagents / LangChain (library)** | `MultiServerMCPClient({"scilink": <entry>})` + `load_mcp_tools(session)` — hold one session per campaign; see the deepagents-scilink examples repo |
| **Deep Agents CLI (managed cloud)** | `deepagents mcp-servers add --url <PUBLIC-URL> --name scilink` — the URL must be reachable from the hosted runtime and MUST sit behind auth (`--header KEY=VALUE`); never expose the raw SSE port |

Give the client agent domain instructions (an `AGENTS.md` / system prompt)
covering the campaign protocol — background jobs + `scilink_job_status`
polling + `scilink_respond` for questions, explicit
`directions` / `input_types` / `input_bounds` on optimization calls, and
"never author data files; re-ingest with `force_regenerate` instead".
The deepagents-scilink repo ships a worked example.

## Container layouts (who runs where)

|  | SciLink spawned (stdio) | SciLink on host (SSE) | SciLink in container (SSE) |
|---|---|---|---|
| **Client on host** | default — recipe A | recipe B | recipe C with `ports: 127.0.0.1:8000:8000`; client config same as B |
| **Client in a container** | install `scilink` in the *same* image and spawn over stdio — one shared boundary, simplest | `http://host.docker.internal:8000/sse` — works, but the client's container isolates it *from* the host while depending on a host service; rarely the right shape | **recommended**: both services on one compose network |

The recommended cross-container shape:

```yaml
services:
  scilink:
    build: {context: ., dockerfile: Dockerfile.scilink}
    volumes: ["./data:/data", "~/.scilink:/home/scilink/.scilink:ro"]
    # no ports: — reachable only on this network
  agent:
    build: {context: ., dockerfile: Dockerfile.agent}
    environment: [SCILINK_URL=http://scilink:8000/sse]
    volumes: ["./data:/data:ro"]     # read datasets; write nothing of the server's
    depends_on: [scilink]
```

The agent reaches SciLink at the service name (`http://scilink:8000/sse`);
datasets go on the shared `/data` volume and are passed as `/data/...`
paths; the agent's mount is read-only so it cannot author or alter data —
if an ingest looks wrong, the fix is re-calling `scilink_analyze_file`
with `force_regenerate=true`, never editing files. The server's session
state persists on the volume, so redeploying either container resumes the
campaign.

## Which setup should I use?

- Trying it out on a laptop → **A** (one `--print-mcp-json`, done).
- Interactive daily use, want to watch SciLink work → **B** (or the
  `scilink-dcode`-style launcher pattern).
- Production, shared lab server, CI, or anything multi-user → **C**, one
  scilink service per concurrent campaign.
- Client itself containerized → same-image stdio for simplicity, compose
  network for isolation.
