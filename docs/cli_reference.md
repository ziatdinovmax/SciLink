# CLI reference

Every command supports `--help` for its full flag list; this page covers the
commands and the flags you'll actually reach for.

## Commands

| Command | What it starts |
|---|---|
| `scilink` | **Mission control** (the meta agent) — routes tasks across plan / analyze / simulate. Explicit form: `scilink explore` (alias `meta`) |
| `scilink ui` | Mission control and all modes in the browser (React web UI; `--streamlit` for the classic app) |
| `scilink plan` | Planning session — experimental design, ideation, optimization |
| `scilink analyze` | Analysis session — images, spectra, datacubes, curve series |
| `scilink simulate` | Simulation session — structures, DFT, classical MD |
| `scilink serve` | MCP server — expose SciLink's tools to another agent |
| `scilink kb` | Manage named knowledge bases (see [knowledge_and_data.md](knowledge_and_data.md)) |
| `scilink memory` | Manage persistent learned skills under `~/.scilink/` |
| `scilink fetch-xrd-library` | Download the COD fingerprint library for XRD phase ID |

## Mission control — `scilink` / `scilink explore`

```bash
scilink                                        # interactive, autopilot
scilink explore --mode autonomous \
    --message "Analyze ./stem.tif, then design a follow-up campaign"   # one-shot brief
scilink explore --knowledge-dir produced-water # attach a named KB or path
scilink explore --restore --session-dir ./meta_session_...             # resume
```

Key flags: `--mode {autopilot,autonomous}` (the meta agent has two levels, not
three — a delegation must complete within a turn), `--message` (seed the first
turn — the one-shot-brief entry point), `--knowledge-dir`, `--session-dir` /
`--restore`, `--model` / `--base-url` / `--api-key`, `--embedding-model`,
`--tools` / `--skills` / `--mcp`.

## Specialist modes

```bash
scilink plan --autonomy autopilot --data-dir ./results --knowledge-dir ./papers
scilink analyze --data ./sample.tif --metadata ./metadata.json
scilink analyze --skills ./raman_skill.md --tools ./my_image_tools.py
scilink simulate --mode autopilot --request "rutile TiO2 supercell with one O vacancy"
```

- **`plan`**: `--autonomy {co-pilot,autopilot,autonomous}`, `--data-dir`,
  `--knowledge-dir` (path or KB name), `--code-dir`, `--embedding-model`,
  `--tools`, `--skills`, `--mcp`.
- **`analyze`**: `--data`, `--metadata`, `--mode {co-pilot,autopilot,autonomous}`,
  `--session-dir`, `--agents`, `--skills`, `--tools`, `--mcp`.
- **`simulate`**: `--mode {co-pilot,autopilot,autonomous}`, `--request`
  (one-shot), `--session-dir`, `--tools`, `--skills`, `--mp-api-key`,
  `--futurehouse-api-key`.

Custom extensions: `--tools` (Python tool files) and `--skills` (markdown
skill bundles) are accepted by all chat modes; `--mcp` (external MCP
servers) by `analyze`, `plan`, and mission control. See
[custom_tools_integration.md](custom_tools_integration.md) and
[mcp_client_integration.md](mcp_client_integration.md).

## Web UI — `scilink ui`

```bash
scilink ui                        # React web UI (default) → http://127.0.0.1:8422
scilink ui --port 9000            # flags pass through to the web server
scilink ui --streamlit            # the classic Streamlit app instead
```

`scilink ui` launches the React web UI — same server as `scilink-web`, so it
takes that command's flags (`--host`, `--port`, `--session-root`, `--token`,
`--users`, TLS flags; see [react_web_ui.md](react_web_ui.md)). The web
backend is a core dependency, and release wheels ship the built bundle, so a
plain `pip install scilink` gets it with no extra step. `--streamlit` forces
the classic Streamlit app; a source checkout without a built bundle also
falls back to Streamlit automatically (with a note).

## Embeddings — plan and mission control

Omit `--embedding-model` for **keyword-only (BM25)** grounding — no
embedding provider or key needed; a knowledge base is still built and
searched, just by keyword. Name a model (`gemini-embedding-001`,
`text-embedding-3-small`, a Bedrock embedder, …) for dense retrieval.
`--embedding-api-key` sets its key, and `--embedding-base-url` an
OpenAI-compatible endpoint for the embeddings only — the chat model keeps
its own route (vendor, or `--base-url`). Without `--embedding-base-url`,
embeddings follow `--base-url` when given (with its key) or go to the
embedder's vendor.

## MCP server — `scilink serve`

```bash
scilink serve --model claude-opus-4-6                 # stdio, autonomous
scilink serve --mode analyze --autonomy co-pilot      # one mode, human-gated
scilink serve --transport sse --host 127.0.0.1 --port 8000
scilink serve --print-mcp-json                        # ready-to-paste client config
```

`--mode {analyze,plan,both,meta}` selects the exposed surface — `meta` serves
mission control itself as a single delegation tool. See
[connecting_agent_clients.md](connecting_agent_clients.md).

The SSE transport is plain HTTP by default, meant to sit behind a
TLS-terminating reverse proxy when exposed off-box. To serve HTTPS directly:

```bash
scilink serve --transport sse --host 0.0.0.0 --port 8000 \
    --ssl-certfile /etc/scilink/cert.pem --ssl-keyfile /etc/scilink/key.pem
# clients use: {"type": "sse", "url": "https://host:8000/sse"}
```

(`--ssl-keyfile-password` for an encrypted key; `--print-mcp-json` reflects
the scheme.) `scilink-web` takes the same three flags.

## The terminal shell

Bare `scilink`, `scilink analyze`, `scilink plan` and `scilink simulate` all
run one terminal shell: line editing with history (`~/.scilink/history/`),
Tab completion of slash commands and paths, Alt+Enter for a newline, a status
row while the agent works (spinner, what it is doing, elapsed time), the
agents' narration filtered to tool calls, reasoning and handoffs (Ctrl+O or
`/verbose` shows everything), the answer rendered as markdown, and one line of
accounting per turn (LLM calls, tokens, seconds). Ctrl+C stops a running turn
the way the web UI's ■ button does; at the prompt it clears the line. Typing
while a turn runs drafts the next message under the status row; Enter queues
it, and queued messages run in order once the turn ends (after Ctrl+C they go
back into the prompt instead). Ctrl+D or `/quit` saves a checkpoint and exits.

Human-in-the-loop questions use the same widgets and labels as the web UI:
free-text feedback with `Enter = Approve plan` (the empty answer accepts),
numbered candidate pickers with the judge's pick as the default, keep/revert
and launch/cancel choices. Generated-code execution is confirmed once at
startup with the web UI's consent sentence (`--yes` or
`SCILINK_ACCEPT_CODE_EXECUTION=1` pre-approves it; a detected sandbox needs
no confirmation).

Sessions stay where they are created (next to your data), and every one is
registered in a central index (`~/.scilink/sessions.jsonl`, or under
`$SCILINK_HOME`), so `--resume` lists and resumes sessions from any folder;
the web UI's "Resume past session" reads the same index. The shell prints the
resume command when you quit. Set `SCILINK_SESSION_ROOT` to put new sessions
in one folder instead of the current one.

Shell flags, on every mode:

```bash
scilink -p "Analyze ./grains.tif" --output-format json --yes   # headless: one task, result on stdout
scilink plan --resume                                        # pick a past session in this folder
scilink analyze --resume analysis_session_20260919_101500    # ... or name it
scilink --verbose                                            # full narration from the start
```

### Slash commands

Shared by every mode: `/help`, `/status` (`/state`), `/mode [level]`
(`/autonomy`), `/tools`, `/mcp <config>`, `/skill <path>`, `/tool <path>`,
`/skills`, `/memory [on|off]`, `/files [subdir]`, `/verbose`, `/cost`,
`/sessions`, `/resume [id]`, `/checkpoint`, `/clear`, `/quit`.
Per mode: `/delegations` (meta), `/agents` and `/schema` (analyze),
`/objective` (plan), `/structures` (simulate).

## Persistent memory — `scilink memory`

Learned skills (graduated or distilled from sessions) live under `~/.scilink/`
(override with `$SCILINK_HOME`), outside the installed package, so they
survive upgrades and load on every future run:

```bash
scilink memory status | enable | disable    # opt-in switch
scilink memory list                         # persisted skills
scilink memory staged                       # raw solutions awaiting distillation
scilink memory show <domain>/<name>         # print a skill's markdown
scilink memory upgrade <domain>/<id> --into <domain>/<name>
scilink memory consolidate <domain>/<technique>   # distill N staged into a new skill
scilink memory promote <domain>/<name>      # make a provisional skill auto-routable
scilink memory bank                         # proven-script bank; also bank-show
```

> **Docker:** `~/.scilink` inside a container is ephemeral — mount a volume
> (`-v ~/.scilink:/home/scilinkuser/.scilink`, or set `SCILINK_HOME` to a
> mounted path) or learned skills are lost when the container exits.

## Sessions

Every chat mode writes a timestamped session directory (override with
`--session-dir`) holding the artifacts it produced plus `chat_history.json`,
`checkpoint.json`, and a session log; checkpoints make sessions resumable, and
a fixed `--session-dir` is how a restarted MCP server resumes its campaign.
Mission-control sessions nest their specialists' sessions
(`<meta_session>/analysis/`, `planning/`, `simulation/`) so each delegation's
outputs stay isolated.

## API keys

Set the key matching your model provider — `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, or `GOOGLE_API_KEY` — or, for an OpenAI-compatible internal
proxy, `SCILINK_API_KEY` together with `--base-url`. The proxy key is not a
vendor credential; vendor endpoints reject it. MCP-server deployments can keep
all of these in `~/.scilink/credentials.env` instead of the client config.
