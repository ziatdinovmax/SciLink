# React Web UI — Roadmap

State and forward plan for the `scilink-web` React UI. Companion to
`react_web_ui.md` (architecture + usage, kept current); this file is
about *what's next and why*. Updated 2026-09-04.

## Where things stand

- **PR #530 (MERGED to main)** — the core: FastAPI + SSE backend
  (`scilink/server/`) over the untouched orchestrators, React frontend
  (`webui/`), chat with live narration + activity line, all seven HITL
  approval widgets, File Explorer (live tree, previews, provenance,
  gallery, zip), floating figure inset, session lifecycle
  (create/resume/rename/reset/quit), wheel-packaged bundle
  (`scilink[web]` → `scilink-web`).
- **PR #531 (OPEN)** — daily-driver batch: mid-chat uploads, pasted
  folder paths (with plan-agent KB-dir reassignment), multi-session UX
  (switcher, detach, per-session close from sidebar and landing page),
  plus the fixes daily use surfaced: SSE heartbeat lock bug (wedged
  event streams), per-turn stdout routing (cross-session narration
  bleed; terminal lines tagged `[HHMMSS]` when several sessions run),
  session-title conversational-output gate, favicon, landing-page mode
  dropdown, collapsible sidebar, inset filmstrip de-dup.

Validation posture: ~40 offline tests across
`tests/test_web_server.py`, `test_artifact_image_rewrite.py`,
`test_capabilities_cache.py`, `test_session_title.py`; live E2E on
Bedrock (analyze co-pilot end-to-end incl. HITL over the API, stop,
restart, resume).

## Next steps, in order

1. **Merge #531**, then a deliberate bake period — real sessions have
   been the best bug-finder (every architectural defect so far was
   caught in daily use, fixed same-day).
2. **Meta delegation view** — the last high-value Streamlit panel.
   Surface the meta's delegation ledger live (specialist, delegation
   number, status, context-flow edges) in the sidebar or a third tab.
   Data already exists on the orchestrator; precedent says
   live-visibility features (activity line, figure inset) pay for
   themselves.
3. **Multi-user hardening** — the gate to sharing a lab-server URL:
   token auth or documented reverse-proxy setup, per-user session
   roots. Per-session isolation is already done (stdout routing, SSE
   per session); only authn/authz is missing. Until then: SSH tunnel.
4. **Remaining panels, on demand** — Skills (browse/upload first; the
   persistent-memory pipeline UI is a separate, bigger design) and
   Tools/MCP (connect servers, list registered tools). Build when
   actually missed.
5. **Big rocks** (each needs its own design pass; they touch agent
   internals, not just the web layer):
   - Simulate mode UI (HPC connection, wizards).
   - True token streaming (agents expose blocking `chat()` only — the
     live stream is console narration by design until this changes).
   - Real cancellation token (stop is print/log-driven +
     subprocess-kill; a silent LLM call delays the abort).

## Standing decisions

- Web-UI fixes go straight onto the current feature branch / main —
  no per-fix PRs into an unmerged branch.
- The Streamlit app (`scilink-ui`) stays untouched and functional; the
  server reuses its streamlit-free modules (`hitl`, `session_meta`,
  `ui/config`). Open question, no urgency: a deprecation horizon for
  Streamlit once the web UI covers everything in daily use —
  double-maintenance is the cost of keeping both.
- Known accepted limits (also in docs/react_web_ui.md): no token
  streaming; single-process in-memory session registry (restart →
  resume from checkpoint); ~40 concurrently open tabs (Starlette sync
  threadpool) before SSE starvation; figure inset samples bursts
  (4 figures / 2s tick), chat + Files always get everything.
