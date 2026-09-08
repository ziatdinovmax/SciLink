# React Web UI (`scilink-web`)

A React single-page app + FastAPI backend that provides the SciLink chat
experience with a real client–server split: live streaming instead of
Streamlit reruns, and a UI that can be served remotely. The Streamlit app
(`scilink-ui`) is unchanged and remains available; the two share the same
orchestrators, session directories, and checkpoint format — a session
started in one can be resumed in the other.

## Quick start

```bash
pip install "scilink[web]"      # fastapi, uvicorn, python-multipart

cd /path/to/your/data           # session dirs are created here
scilink-web                     # http://127.0.0.1:8422
```

Release wheels ship the built React bundle. From a **repository checkout**
the bundle is not in git — build it once (Node 20+):

```bash
scripts/build_webui.sh          # npm ci + build → scilink/server/static/
```

The server serves `webui/dist` when present (freshest, from `npm run build`
in `webui/`), else the bundle in `scilink/server/static/`; with neither it
answers `/` with a 503 that says how to build, while the API stays up.
Options:

```
scilink-web --host 127.0.0.1 --port 8422 --session-root .
```

`--session-root` is where sessions are created, where the resume list is
discovered, and the fence for file serving. The default is the local
single-user tool: loopback bind, no sign-in. To reach it from another
machine either tunnel (`ssh -L 8422:127.0.0.1:8422 host`) or share it
properly — see "Sharing on a lab server" below. A non-loopback `--host`
without authentication is refused.

## Sharing on a lab server

```bash
# per-user tokens: each user gets <session-root>/users/<name>/
cat > users.json <<'JSON'
{"alice": "<token>", "bob": "<token>"}
JSON
python -c 'import secrets; print(secrets.token_urlsafe(32))'   # make tokens
scilink-web --host 0.0.0.0 --port 8422 --session-root /data/scilink --users users.json

# or one shared token (single user, sessions stay in --session-root)
scilink-web --host 0.0.0.0 --token "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# (SCILINK_WEB_TOKEN in the environment works too)
```

Give each person a sign-in link — `https://host/?token=<their token>` — or
they paste the token on the sign-in screen. The page exchanges it for an
HttpOnly cookie and scrubs it from the URL; the sidebar shows who is
signed in and offers sign-out. Scripts use `Authorization: Bearer <token>`.

What a shared server changes, and only a shared server:

- every `/api/v1` call needs a token (bearer header or the login cookie);
  `/auth/*` is public so the sign-in screen can load;
- with `--users`, each user has an isolated session root and live-session
  registry — another user's session id is a 404, and the resume list only
  shows their own;
- "Use a folder on this machine…" disappears from the upload menu and the
  endpoints behind it (`/folders`, `/plan_dirs`) answer 403: the server is
  not on the browser's machine, so upload the folder instead;
- "Quit App" is disabled with `--users` (one person must not stop the
  server for everyone);
- cookie sessions live in memory, so a restart signs everyone out.

TLS is the reverse proxy's job. Caddy:

```
scilink.lab.example.org {
    reverse_proxy 127.0.0.1:8422
}
```

nginx (the `X-Forwarded-Proto` header is what makes the cookie `Secure`;
`proxy_buffering off` keeps the SSE stream live):

```
location / {
    proxy_pass         http://127.0.0.1:8422;
    proxy_set_header   Host $host;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;
    proxy_buffering    off;
    proxy_read_timeout 3600s;
    client_max_body_size 2g;
}
```

If the proxy itself authenticates every request (SSO, mTLS), run with
`--insecure-no-auth` behind it. Not covered yet: per-user LLM credentials
(entered per session in the sidebar as before; the server stores none),
rate limiting on sign-in (put the proxy's in front if internet-facing).

## What's included (first cut)

- **Modes**: meta (mission control), analyze, plan. Simulate/HPC is not in
  the web UI yet.
- **Sidebar**: model + provider fields (e.g. Bedrock region), API key /
  base URL (with "✓ available from ENV" captions — the server reports
  credential *availability*, never values), FutureHouse / Materials Project
  keys, embedding model (plan/meta), autonomy, the code-execution consent
  checkbox, session start/resume/rename, theme toggle.
- **Chat**: markdown + LaTeX rendering, image attachments, HTML report
  cards (sandboxed iframe + download), markdown deliverable cards (rendered
  inline, images resolved through the API), per-turn collapsed verbose log.
- **Live turn**: agent-working spinner, stop button, and a colorized
  streaming narration pane (meta reasoning cyan, delegated-specialist amber,
  handoff banners gold — same scheme as the Streamlit verbose panel).
- **Human-in-the-loop**: all the Streamlit approval surfaces — free-text
  feedback with context box, dataset-description prompt, code review with
  the generated scripts inline, keep/revert, best-of-N candidate selection
  (with preview images and the judge's pick), plan-candidate selection, and
  the fan-out launch confirmation panel.
- **Uploads**: per-mode pre-chat heroes (analyze data+metadata, plan
  knowledge/code/data, meta combined dropzone) writing to the same session
  subdirectories as Streamlit, composing the same dispatch prompts; a
  paperclip button on the chat input uploads mid-conversation (files are
  routed to the mode's category by extension and the saved paths dropped
  into the draft); pasted folder paths on the plan/meta heroes are
  validated server-side and enumerated into the dispatch prompt, with the
  plan agent's resource dirs repointed at the stable folders so KB indexes
  are reused across sessions.
- **Attach a script** (beyond Streamlit): a `.py` (or `.yaml` / `.md`)
  dropped on the paperclip in analyze mode lands in the session's
  `scripts/` folder and the draft says "use it as the reference
  implementation". The orchestrator passes it to `run_analysis(
  reference_scripts=[...])`; the analysis agent treats it like a script-bank
  hit — adapts it to the data (method and parameters kept, I/O conformed
  to the run's contract), verifies the result in its normal QC loop, and
  reports what it kept and changed. It is never run verbatim; for verbatim
  execution use MCP. Plan mode already reads and edits uploaded code
  (`code/`); meta hands the path through to the analysis child.
- **Folder uploads** (beyond Streamlit): every dropzone takes a dropped
  directory (walked recursively via the FileSystem entry API), and a click
  on the dropzone or the chat paperclip opens a two-item menu, "Upload
  files" / "Upload folder" — one control, two hidden inputs, because no
  native dialog picks both. On the plan and meta heroes the menu has a third
  item, "Use a folder on this machine…", which reveals the pasted-path input
  (the folder is used in place, nothing copied — the route for large or
  shared data and for plan-mode KB reuse; only meaningful on a local bind).
  The layout is preserved under the category root
  (`uploads/<folder>/<sub>/…`, `knowledge/<folder>/…`), files the category
  does not accept are skipped and reported rather than failing the drop,
  hidden entries are dropped, and a 2000-file cap applies. A flat folder
  is a series (same as a multi-file drop); a nested one is described to
  the agent subfolder by subfolder with absolute paths, since the agents'
  own directory listings are one level deep. The meta's `inspect_uploads`
  now reports subfolders and takes `recursive=true`.
- **Mission-control tree** (meta sessions, sidebar — as in Streamlit): a
  compact live tree of the delegation ledger under the session section —
  specialist branches, one line per delegation with status glyph, index,
  short label and context-flow edges ("←#1"), colored by status, pulsing
  while running, in a bounded scroll box. Always visible while you chat.
- **Telemetry tab** (meta sessions): the detail behind the tree — grouped
  by specialist with the worker agents each used, rows with start time and
  elapsed/duration, fan-out / timed-out / resumed tags, and a click (or a
  click on a sidebar tree row) that expands the task, the specialist's
  summary and findings, the produced files (opening in the Files tab),
  warnings and error. Both surfaces are fed by the session snapshot and
  `delegations` SSE events as the ledger changes mid-turn, and survive
  refresh and resume. Below the ledger, from the `/telemetry` snapshot:
  the **tool sequence** (every tool call each layer's LLM made — meta,
  analysis specialist, planning specialist — with the input/output shape
  in the table and the actual arguments and result on click — a failed
  call shows its error message inline in red instead of the shape — plus
  a link to the full chat history in Files), the **worker agents** (each
  sub-agent's action history with outcomes; a row expands to its actions,
  an action to its input, result and rationale) and the **analysis
  reports** (each analysis's claims and reasoning). Polled every 3 s while
  a turn runs, since the tool sequence reads the agents' live message
  lists, and refreshed on every ledger change otherwise. Each layer's
  full chat history can be opened in Files or downloaded as JSON. Above
  the ledger sits the **dependency graph** from the Streamlit tab, drawn
  as plain SVG (no graphviz): the meta-agent root, one box per delegation
  colored by status and annotated with its sub-agents, grey dispatch
  edges, blue context edges; boxes are layered by context depth, a layer
  wider than six wraps into rows, the parallel branches of one fan-out
  collapse into a single stacked box with their outcome counts (30
  branches are one node, not 30 boxes and 60 edges), every edge is routed
  around the boxes it does not connect (straight when nothing is in the
  way, else a sweep inside the free band between rows into the nearest
  vertical lane that clears every box — a column gap, the strip beside a
  row, or the canvas edge; sources in a wrapped layer sit in its last row
  so their edges never cross a sibling row), and clicking a box expands
  its ledger row (a fan-out box opens its first failed branch). Layout
  and routing are a pure module (`webui/src/delegationGraph.ts`) checked
  by `npm run check:graph` (`webui/scripts/graph_check.ts`): sixteen
  ledger scenarios — chains, diamonds, skip edges, fan-outs with fusion
  and a non-member consumer, two fan-outs, wide layers, a 40-deep chain,
  20 sources into one fusion, out-of-order indices, unknown / self /
  duplicate sources, a cycle, 45 delegations — each asserting the drawn
  context edges equal the ledger's `context_from` relations, one
  dispatch edge per node, endpoints on the right boxes, no edge through
  a third box, downward flow, no overlapping boxes. Edge paths carry
  `data-from` / `data-to` and node groups `data-node`, so the same audit
  can be run against the rendered SVG in a browser.
- **Skills tab** (all modes): upload custom skill `.md` files (saved under
  the session's `custom_skills/`, registered with the agent for this
  session, auto-selectable like built-ins) and browse the catalog — every
  built-in bundle by domain with its one-line description, plus a
  markdown viewer (frontmatter shown as a caption). Persistent memory
  (graduated / auto-distilled skills under `~/.scilink`) is not here yet.
- **MCP tab** (all modes): connect MCP servers — a `stdio` command, an
  SSE URL, or a streamable-HTTP URL with optional JSON headers — and
  disconnect them; each server card lists the tools it registered. The
  agent's own built-in tools are not listed: in a meta session that would
  be only the meta's routing tools, not what the specialists can do. There is deliberately no tool-file
  uploader (the Streamlit `tool_schemas` / `create_tool_functions` contract
  is not ported): users hand code to the agents as scripts attached in
  chat, adapted by codegen, or as MCP servers, run verbatim in any
  language. A `stdio` command runs on the server's machine — the same
  trust as the agents' own code execution, which every session consents
  to; on a shared server `${VAR}` in header values is not expanded from
  the server environment.
- **Sessions**: the server holds many live sessions; the sidebar lists the
  others with one-click switching, Detach leaves a session running while
  you start or join another, and the welcome screen offers reattach when
  several are live (exactly one live session reattaches automatically).
- **Live analysis inset**: a draggable, dismissible picture-in-picture
  panel that shows result figures (fit overlays, dashboards, trend plots)
  as the agent writes them mid-turn, with a filmstrip to page back and a
  branch label during fan-outs. Driven by `analysis_image` SSE events from
  the same filesystem watcher as the explorer; intermediates (candidate
  attempts, elbow plots, preview grids) are filtered server-side.
- **File Explorer** (Files tab) — beyond Streamlit parity:
  - live tree: no Refresh button — the server emits `files_changed` while a
    turn runs, so artifacts appear as the agent writes them;
  - "new" badges on files produced since the current turn started, a
    filename filter, and a recent-first flat view;
  - previews: images with zoom, NPY/TIFF heatmaps with a colormap selector
    (server-rendered), sortable/filterable CSV/TSV/XLSX tables, markdown
    with a Rendered|Source toggle, JSON pretty-print, sandboxed HTML,
    browser-native PDF, code/VASP text;
  - chat ↔ explorer cross-links: artifact cards jump to the file in the
    explorer; any file's "Attach to chat" drops its path into the draft;
  - gallery view (thumbnail grid) and per-folder / whole-session zip
    download;
  - provenance: a timeline view of the session event log
    (`events.jsonl`) and a "produced by <tool>" line on previews;
  - deep links: the tab and selected file live in the URL hash, so a
    refresh (or shared link) lands on the same file.

Not yet ported: simulate mode, the Skills tab's persistent-memory section, vibes.

## Architecture

```
webui/ (Vite + React + TS)  ──REST + SSE──►  scilink/server/ (FastAPI)
                                                │  in-process, same as Streamlit
                                                ▼
                                agent.chat() / restore_from_checkpoint()
```

- One background thread per chat turn (port of the Streamlit runner),
  stdout/logging teed through `OutputCapture`; a watcher emits incremental
  `log` SSE events.
- HITL prompts route through `scilink.hitl.set_thread_channel` into an
  HTTP-parking channel; the server converts each `FeedbackRequest` into a
  structured "presented question" (widget type, labels, candidates, preview
  images, code files) so the frontend renders without prompt sniffing.
- Artifacts are per-turn filesystem sweeps with the same rules as Streamlit
  (HTML report suppresses raw images; deliverable manifest decides which
  markdown embeds; path+mtime identity).
- SSE events carry monotonic ids in a bounded ring, so `EventSource`
  reconnects replay via `Last-Event-ID`; a fresh page falls back to the
  snapshot endpoint.
- Stop is the print/log-driven `AgentStoppedError` path, identical to
  Streamlit (a run inside a long silent native call stops at its next
  print; generated-script subprocesses are killed immediately).

### API surface (`/api/v1`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/auth/me` | public: whether sign-in is required, who is signed in, `local_files` |
| POST | `/auth/login` | access token → HttpOnly session cookie |
| POST | `/auth/logout` | drop the cookie session |
| GET | `/config` | modes, models, autonomy options, provider fields, credential availability, consent text, `auth`, `local_files` |
| GET | `/sessions?mode=` | live sessions + resumable session dirs |
| POST | `/sessions` | create, or resume with `resume_dir` |
| GET/PATCH | `/sessions/{id}` | snapshot / rename |
| POST | `/sessions/{id}/messages` | start a turn (409 while one runs) |
| GET | `/sessions/{id}/events` | SSE: `log`, `status`, `question`, `question_cleared`, `assistant_message`, `session_named`, `files_changed`, `analysis_image`, `delegations`, `error` |
| POST | `/sessions/{id}/feedback` | answer the parked HITL question |
| POST | `/sessions/{id}/stop` | stop the running turn |
| POST | `/sessions/{id}/uploads` | multipart, `category` = data/metadata/knowledge/code/planning_data/meta/scripts; optional `paths` (JSON list of relative paths, one per file) makes it a layout-preserving folder upload |
| POST | `/sessions/{id}/folders` | validate pasted local folder paths + enumerate tabular contents and immediate subfolders |
| POST | `/sessions/{id}/plan_dirs` | repoint the plan agent's knowledge/code/data dirs at stable folders |
| GET | `/sessions/{id}/files?path=` | serve a session file (traversal-fenced) |
| GET | `/sessions/{id}/tree` | recursive listing with sizes/mtimes and per-file "new" flags |
| GET | `/sessions/{id}/thumb?path=&size=&cmap=` | PNG thumbnail; NPY/TIFF rendered as normalized heatmaps |
| GET | `/sessions/{id}/table?path=&limit=` | CSV/TSV/XLSX head as JSON columns+rows |
| GET | `/sessions/{id}/zip?path=` | zip a session subdirectory (or the whole session) |
| GET | `/sessions/{id}/skills` | catalog: built-in bundles by domain with descriptions, the session's custom skills |
| GET | `/sessions/{id}/skills/{domain}/{name}` | a skill's markdown (`domain` = catalog domain or `custom`) |
| POST | `/sessions/{id}/skills` | multipart `.md` uploads → `custom_skills/`, registered with the agent |
| GET | `/sessions/{id}/tools` | connected MCP servers with their tools, other external tools, `mcp_supported` |
| POST | `/sessions/{id}/mcp` | connect an MCP server (`name`, `transport` stdio/sse/http, `command` or `url`, `headers`) |
| DELETE | `/sessions/{id}/mcp/{name}` | disconnect it |
| GET | `/sessions/{id}/delegations` | meta: the delegation ledger shaped for the sidebar tree and Telemetry tab (empty for other modes) |
| GET | `/sessions/{id}/telemetry` | meta: full read-only telemetry snapshot (ledger, worker action histories, analysis reasoning, tool sequence) |
| GET | `/sessions/{id}/provenance` | tool-call timeline from every `events.jsonl` under the session |
| DELETE | `/sessions/{id}` | reset: stop and drop the live session (dir stays resumable) |
| POST | `/quit` | shut the server down |

## Development

```bash
scilink-web --port 8422          # backend
cd webui && npm run dev          # Vite dev server on :5173, proxies /api
```

### Releasing

`.github/workflows/release.yml` runs on a `v*` tag (or manually): it builds
the bundle with `scripts/build_webui.sh`, then `python -m build`, verifies
`scilink/server/static/index.html` is inside the wheel, uploads `dist/` as a
run artifact and attaches it to the GitHub release. Publishing to PyPI is a
separate job that only runs when the repository variable `PYPI_PUBLISH` is
`true` and PyPI trusted publishing is configured for the `pypi` environment;
until then, upload the downloaded artifacts with `twine`. CI (`test.yml`)
also typechecks and builds the frontend on every push and PR, so a broken
`webui/` fails fast without anyone committing a bundle.

Backend tests: `pytest tests/test_web_server.py` (presenter classification,
artifact sweeps, upload conventions, traversal guard, discovery, SSE ring,
turn/feedback/stop plumbing with fake agents).

## Known limitations

- No token streaming: the agents expose a blocking `chat()`, so the live
  stream is console narration, with the full answer at turn end.
- Single-process, in-memory session registry: a server restart drops live
  agents (resume from checkpoint recovers, as in Streamlit).
