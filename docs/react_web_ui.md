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

The server serves the pre-built React bundle when `webui/dist` exists (repo
checkouts: `cd webui && npm install && npm run build`). Options:

```
scilink-web --host 127.0.0.1 --port 8422 --session-root .
```

`--session-root` is where sessions are created, where the resume list is
discovered, and the fence for file serving. Binding a non-loopback `--host`
prints a loud warning: there is **no authentication** in this first cut —
anyone who can reach the port can execute code through the agents. For
remote use, tunnel (`ssh -L 8422:127.0.0.1:8422 host`) or front it with an
authenticating reverse proxy.

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
  subdirectories as Streamlit, composing the same dispatch prompts.

Not yet ported: simulate mode, the File Explorer / Tools / Skills /
Telemetry tabs, vibes, pasted-folder-path inputs on the plan/meta heroes.

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
| GET | `/config` | modes, models, autonomy options, provider fields, credential availability, consent text |
| GET | `/sessions?mode=` | live sessions + resumable session dirs |
| POST | `/sessions` | create, or resume with `resume_dir` |
| GET/PATCH | `/sessions/{id}` | snapshot / rename |
| POST | `/sessions/{id}/messages` | start a turn (409 while one runs) |
| GET | `/sessions/{id}/events` | SSE: `log`, `status`, `question`, `question_cleared`, `assistant_message`, `session_named`, `error` |
| POST | `/sessions/{id}/feedback` | answer the parked HITL question |
| POST | `/sessions/{id}/stop` | stop the running turn |
| POST | `/sessions/{id}/uploads` | multipart, `category` = data/metadata/knowledge/code/planning_data/meta |
| GET | `/sessions/{id}/files?path=` | serve a session file (traversal-fenced) |

## Development

```bash
scilink-web --port 8422          # backend
cd webui && npm run dev          # Vite dev server on :5173, proxies /api
```

Backend tests: `pytest tests/test_web_server.py` (presenter classification,
artifact sweeps, upload conventions, traversal guard, discovery, SSE ring,
turn/feedback/stop plumbing with fake agents).

## Known limitations

- Concurrent turns in *different* sessions can cross-bleed `print()`
  narration in the live log (process-global stdout capture — inherited from
  the Streamlit design; the logging-path narration is per-session clean).
- No token streaming: the agents expose a blocking `chat()`, so the live
  stream is console narration, with the full answer at turn end.
- Single-process, in-memory session registry: a server restart drops live
  agents (resume from checkpoint recovers, as in Streamlit).
