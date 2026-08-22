# CLI reference

Every command supports `--help` for its full flag list; this page covers the
commands and the flags you'll actually reach for.

## Commands

| Command | What it starts |
|---|---|
| `scilink` | **Mission control** (the meta agent) — routes tasks across plan / analyze / simulate. Explicit form: `scilink explore` (alias `meta`) |
| `scilink ui` | Mission control and all modes in the browser |
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

## In-session slash commands

Chat sessions accept slash commands alongside natural language. Common set:
`/help`, `/tools`, `/files` (plan) or `/agents` (analyze), `/state` or
`/status`, `/autonomy [level]` (plan) or `/mode [level]` (analyze),
`/checkpoint`, `/schema` (analyze — metadata JSON schema), `/quit`.

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
