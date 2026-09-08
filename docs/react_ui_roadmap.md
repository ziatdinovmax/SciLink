# React Web UI — Roadmap

State and forward plan for the `scilink-web` React UI. Companion to
`react_web_ui.md` (architecture + usage, kept current); this file is
about *what's next and why*. Updated 2026-09-07.

## Where things stand

- **PR #530 (MERGED to main)** — the core: FastAPI + SSE backend
  (`scilink/server/`) over the untouched orchestrators, React frontend
  (`webui/`), chat with live narration + activity line, all seven HITL
  approval widgets, File Explorer (live tree, previews, provenance,
  gallery, zip), floating figure inset, session lifecycle
  (create/resume/rename/reset/quit), wheel-packaged bundle
  (`scilink[web]` → `scilink-web`).
- **PR #531 (MERGED)** — daily-driver batch: mid-chat uploads, pasted
  folder paths (with plan-agent KB-dir reassignment), multi-session UX
  (switcher, detach, per-session close from sidebar and landing page),
  plus the fixes daily use surfaced: SSE heartbeat lock bug (wedged
  event streams), per-turn stdout routing (cross-session narration
  bleed; terminal lines tagged `[HHMMSS]` when several sessions run),
  session-title conversational-output gate, favicon, landing-page mode
  dropdown, collapsible sidebar, inset filmstrip de-dup.
- **2026-09-07 batch (PRs #536–#539, #544, #545, delegation view)** —
  folder uploads (recursive drop / directory picker, layout preserved
  server-side, strays skipped) behind a single upload menu that also holds
  the pasted-path route; `inspect_uploads` reports subfolders and can
  recurse; resume card ranks the headline deliverable over a late-refined
  ideation report (#533); activity line narrates plan refinement instead
  of "-"; locked sidebar fields greyed and showing the live session's
  model/autonomy; Files-tab documents newest-first; **Delegations tab**
  (live mission-control view of the meta ledger, the last high-value
  Streamlit panel); bundle no longer committed — built at release by
  `release.yml` (see "Standing decisions").

Validation posture: ~60 offline tests across
`tests/test_web_server.py`, `test_artifact_image_rewrite.py`,
`test_capabilities_cache.py`, `test_session_title.py`; live E2E on
Bedrock (analyze co-pilot end-to-end incl. HITL over the API, stop,
restart, resume).

## Next steps, in order

1. **Merge the 2026-09-07 batch, then bake.** Open at time of writing:
   #544 (listing order + test path), #545 (bundle at release), #546
   (Delegations tab). Real sessions remain the best bug-finder.
2. **First tagged release through `release.yml`.** Tag `v0.0.68` (or run
   the workflow by hand) and confirm the wheel on the release page carries
   the bundle; decide whether to enable PyPI trusted publishing
   (`PYPI_PUBLISH=true`) or keep uploading with twine.
3. **Multi-user hardening** — the gate to sharing a lab-server URL:
   token auth or documented reverse-proxy setup, per-user session
   roots, and hide the "Use a folder on this machine…" menu item on a
   non-loopback bind (it cannot work for a remote user). Per-session
   isolation is already done; only authn/authz is missing. Until then:
   SSH tunnel.
4. **Remaining panels, on demand** — Telemetry (the `/telemetry` endpoint
   already serves the full reader; the Delegations tab covers the ledger,
   so what is left is the per-agent tool sequence and worker action
   histories), Skills (browse/upload first; the persistent-memory
   pipeline UI is a separate, bigger design) and Tools/MCP (connect
   servers, list registered tools). Build when actually missed.
5. **Parity for the switch to default** (see "Standing decisions"):
   simulate mode in the web UI (HPC connection, wizards) is the largest
   gap; after it the web UI covers everything Streamlit does.
6. **Big rocks** (each needs its own design pass; they touch agent
   internals, not just the web layer):
   - True token streaming (agents expose blocking `chat()` only — the
     live stream is console narration by design until this changes).
   - Real cancellation token (stop is print/log-driven +
     subprocess-kill; a silent LLM call delays the abort).

## Standing decisions

- Web-UI fixes go straight onto the current feature branch / main —
  no per-fix PRs into an unmerged branch.
- **The web UI becomes the default.** Decided 2026-09-07: once the web
  UI reaches daily-use parity (simulate mode being the main gap), the
  `web` extra's dependencies move into the core install and `scilink-web`
  supersedes `scilink-ui` as the default; Streamlit is then retired.
  Until then the Streamlit app stays untouched and functional and the
  server keeps reusing its streamlit-free modules (`hitl`, `session_meta`,
  `ui/config`). Consequence: parity gaps are on the critical path, not
  optional, and nothing new is built Streamlit-only.
- **The built bundle is not in git.** `scripts/build_webui.sh` produces
  `scilink/server/static/`; `release.yml` builds it before the wheel and
  verifies it is inside; CI builds the frontend on every push. A checkout
  builds once. Rationale: two open frontend PRs always conflicted on the
  same hashed asset.
- **One upload control.** Files, folders and "use a folder on this
  machine" live behind a single menu; the pasted-path route is the local
  power-user path (no copy, plan-mode KB index reuse), upload is the
  general mechanism and the only one that works remotely.
- **Fixes go straight onto `main` via small PRs**, one concern per
  branch, each verified live where the change is user-visible; commits
  carry only the generic co-author trailer, no session identifiers.
- Known accepted limits (also in docs/react_web_ui.md): no token
  streaming; single-process in-memory session registry (restart →
  resume from checkpoint); ~40 concurrently open tabs (Starlette sync
  threadpool) before SSE starvation; figure inset samples bursts
  (4 figures / 2s tick), chat + Files always get everything.
