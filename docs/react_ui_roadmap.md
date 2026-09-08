# React Web UI — Roadmap

State and forward plan for the `scilink-web` React UI. Companion to
`react_web_ui.md` (architecture + usage, kept current); this file is
about *what's next and why*. Updated 2026-09-08.

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
  model/autonomy; Files-tab documents newest-first; **mission-control
  tree** in the sidebar (live, as in Streamlit) with a Telemetry tab for
  the per-delegation detail (the last high-value Streamlit panel); bundle no longer committed — built at release by
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
3. ~~**Multi-user hardening**~~ — DONE 2026-09-08: `--token` /
   `--users FILE` bearer + cookie auth, per-user session roots and
   registries, a refused unauthenticated public bind, the local-folder
   route hidden and refused on a remote bind, quit disabled on a shared
   server, reverse-proxy docs. The single-user default is unchanged.
   Remaining, on demand: per-user LLM credentials, sign-in rate limiting.
4. ~~**Shared file I/O + analysis-mode adoption (#481)**~~ — DONE
   2026-09-08: `scilink/utils/file_io.py` is the one reader / writer; the
   three modes are thin wrappers; analysis mode has the full read / append
   / edit / rename surface; JSON cap and backup-on-overwrite landed once;
   the meta's capability cache now keys on the tool modules' source. Was:
   Extract
   `read_file` / `save_file` / `append_file` / `read_document` into one
   engine in `scilink/utils` (the `file_edit.py` pattern), land the JSON
   cap and backup-on-overwrite once, and give analysis mode the full
   read / edit / write surface it lacks. Prerequisite for anything
   script-shaped in analyze mode (next item) and it folds three
   diverging copies back onto one.
5. ~~**Attach a script in chat**~~ — DONE 2026-09-08: `.py` via the
   paperclip → `scripts/`; `run_analysis(reference_scripts=[...])`;
   `_reference_scripts.py` renders the script wherever the agents show the
   user's guidance (planning, codegen, refinement, correction) with
   script-bank semantics (adapt, never verbatim). Was: Accept `.py` through the paperclip in
   every mode. Plan mode then already closes the loop (read → edit →
   `generate_implementation_code`). Analyze mode gets a
   `reference_script` argument on `run_analysis` that injects the file's
   content into the codegen's implementation slot — the code-in-markdown
   rung: the agent adapts the script, the sandbox + verification loop
   backstops it. Draft text and narration say "adapted from your script";
   no skill object, no user-facing format. Verbatim execution of a user
   script in analyze mode is a separate decision (it would change the
   foundation agents' generate-verify-lock contract); until then the
   verbatim routes are MCP or a package `TOOL_SPEC` contribution.
6. **Remaining panels, on demand** — ~~Tools~~ DONE 2026-09-08 (MCP
   connect + the read-only tool inventory, without the Streamlit
   tool-file uploader, whose `tool_schemas` / `create_tool_functions`
   contract is dropped: the attach-a-script path replaces it for
   adaptation, MCP for verbatim); Telemetry (the tab exists with the
   per-delegation detail; the `/telemetry` endpoint already serves the
   full reader, so what is left is the per-agent tool sequence and worker
   action histories); Skills (browse/upload first; the persistent-memory
   pipeline UI is a separate, bigger design). Build when actually missed.
7. **Parity for the switch to default** (see "Standing decisions"):
   simulate mode in the web UI (HPC connection, wizards) is the largest
   gap; after it the web UI covers everything Streamlit does.
8. **Big rocks** (each needs its own design pass; they touch agent
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
- **No bespoke tool format.** Users hand the agents code as ordinary
  scripts attached in chat (adapted by codegen) or as MCP servers (run
  verbatim, any language). The Streamlit tool-file contract is not
  ported to the web UI.
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
