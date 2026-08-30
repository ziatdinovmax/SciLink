# Programmatic session memory — `events.jsonl` + bounded history search

**Issue:** https://github.com/ziatdinovmax/SciLink/issues/462
**Branch (when work starts):** `feature-session-event-log` off `main`
**Status:** PLANNED — no code written yet.

Inspired by the PRO-LONG result (arXiv:2607.20064): one append-only
structured log of every action/outcome, queried *programmatically*
(grep-style, never read whole), improved the same long-horizon agents by
~18 points on average. We adopt the pattern, not the harness.

## The problem, precisely

Long chat sessions lose their own middle and cannot get it back:

- `_trim_history` caps in-context history at
  `MAX_HISTORY_MESSAGES = 100` (`scilink/agents/exp_agents/analysis_orchestrator.py:538`,
  trim at `:1327`, mid-loop trim at `:1630`; the planning / simulation /
  meta orchestrators carry the same copied shape). It slices out the
  middle and inserts a summary marker. Evicted turns are unrecoverable
  by the LLM.
- The full record survives in `chat_history.json`, but that is the wrong
  recovery surface: a single JSON array containing full tool results —
  not line-addressable, and a naive read pulls the entire evicted
  history back into context (re-saturating the window the trim was
  protecting).
- `read_file` / `list_workspace_files` exist only on the planning
  orchestrator (`scilink/agents/planning_agents/orchestrator_tools.py:5650`,
  `:1333`) — and even there, nothing tells the agent its own history is
  on disk.
- The meta's delegation ledger (`scilink/agents/meta_agent/meta_orchestrator.py:472`,
  `get_delegation_history` at `meta_orchestrator_tools.py:828`) holds
  delegation-level *summaries*; specialist-level actions inside child
  sessions are out of reach.

Observed consequences: a fit parameter the user pinned 60 turns ago is
gone; a long BO campaign can re-suggest a condition that already failed;
a meta investigation can't recall what a specialist actually did three
delegations back.

## Design

### 1. The log: per-session `events.jsonl`

One compact JSON line per tool call, appended at each tools class's
single dispatch chokepoint — `execute_tool`:

| Orchestrator | Chokepoint |
|---|---|
| analysis | `scilink/agents/exp_agents/analysis_orchestrator_tools.py:5834` |
| planning | `scilink/agents/planning_agents/orchestrator_tools.py:8019` |
| simulation | `scilink/agents/sim_agents/simulation_orchestrator_tools.py:2857` |
| meta | `scilink/agents/meta_agent/meta_orchestrator_tools.py:305` |

Record shape (every field bounded at write time — summaries by
construction, so grep hits stay small regardless of session length):

```json
{"n": 41, "ts": "2026-08-14T22:03:11Z", "tool": "run_analysis",
 "args": "data=ring_spectrum.csv skill=xrd_profile", "status": "success",
 "summary": "R²=0.991, 3 peaks; figure fit_overlay.png",
 "files": ["scripts/fitting_script.py", "fit_overlay.png"]}
```

- `n` — event index (monotonic counter owned by the log, not the
  message list; message indices shift under trimming).
- `args` — gist, ≤ ~200 chars. **Redact** anything matching
  `api_key` / `token` / `secret` kwarg names.
- `summary` — ≤ ~300 chars. Derivation: if the tool result parses as
  JSON, prefer `status` + `message`/`summary`-like keys; else head of
  the string. Never embed base64/binary (`utils/tool_media.py` already
  strips images from history; the event writer takes the *post-strip*
  text).
- Written via `scilink/utils/text_io.py` helpers (explicit UTF-8 —
  the Windows cp1252 lesson from PR #455 applies; markdown/text writes
  are the ones that break).

New module **`scilink/session_events.py`**, deliberately mirroring
`scilink/hitl.py`'s thread-local binding (`set_thread_feedback_log`
at `hitl.py:184`, `use_feedback_log` at `:203`):

```python
append_event(tool, args, result, files=None)   # no-op if unbound
set_thread_event_log(path) / get_thread_event_log()
use_event_log(path)                            # context manager, restore-on-exit
search_events(path, pattern, max_hits, ...)    # pure function; the tool wraps it
read_events(path, start_n, end_n, max_chars)   # pure function
```

Binding lifecycle copies the HITL shape exactly:
- `chat()` entry binds `<base_dir>/events.jsonl` (alongside the
  `feedback_log` binding at `analysis_orchestrator.py:1355`).
- `run_task` saves the caller's binding and restores it in `finally`
  (shape at `analysis_orchestrator.py:1445–1475`) — so a meta
  delegation logs child events to the *child's* session, and the meta's
  own log resumes afterward. This is exactly the property the
  feedback-log arc already proved out.
- Unbound thread ⇒ `append_event` is a silent no-op (library callers,
  bare agents, tests unaffected).
- **Failure isolation:** the writer never raises into the tool path —
  any exception is caught and logged at debug. A broken event log must
  not break a run.

### 2. The access tools: two-step, hard-capped

Registered on **all four** orchestrators (same declaration in each —
copied, per the no-base-class rule):

- `search_session_history(pattern, max_hits=10)` — case-insensitive
  substring/regex grep over `events.jsonl`; returns matching lines with
  their `n`. Caps: `max_hits` ≤ 50, per-hit chars already bounded by
  construction, total return ≤ ~6 KB. Zero hits returns a clear
  "no matches" (honest-null, not an error).
- `get_history_events(start_n, end_n)` — the drill-down: verbatim event
  lines for a small index range. Cap: ≤ 40 events / ~8 KB per call.

Tool descriptions must say *when* to reach for them (one sentence, per
the prompt-patch convention): "Earlier turns may have been trimmed from
context; search here before asking the user to repeat information or
redoing prior work."

The shape mirrors the proven `list_workspace_files` → `read_file` pair:
cheap wide search over summaries, expensive narrow read of specifics.
Context cost of consulting memory is proportional to the question, not
to session length.

### 3. Trim integration

The `_trim_history` summary marker (`analysis_orchestrator.py:1346`)
and the mid-loop trim warning (`:1630`) gain one sentence: *"N older
turns were trimmed; use search_session_history to recover details from
them."* Applied in all four orchestrators, gated on the tools actually
being registered.

### 4. Meta reach-through

`search_session_history` on the **meta only** gains
`scope: "own" | "children" | "all"` (default `"own"`). Children scope
globs `<meta_session>/{analysis,planning,simulation}/**/events.jsonl`
(including per-delegation `delegations/<NN>_<slug>/` dirs) and prefixes
each hit with the child mode + session dir, so a long investigation can
recover specialist-level actions, not just `get_delegation_history`
summaries. Same total-size caps apply across the union.

**Fan-out branches:** branches run in worker threads, so thread-local
binding means branch events are currently lost unless bound. Bind each
branch thread to the meta's own `events.jsonl` with a
`"branch": "<branch_id>"` field (mirroring how `_BranchChannel` labels
prompts). This rides the existing per-branch setup in
`scilink/agents/meta_agent/fanout.py::_run_one_branch`.

## Non-goals (settled)

- **No embedding/RAG over session history.** PRO-LONG's evidence is
  that grep over a well-structured log beats elaborate memory for this
  use case; matches our retrieval philosophy (explicit grounding,
  lexical fallback). Revisit only on demonstrated lexical failure —
  same trigger discipline as the `BaseChatOrchestrator` rule.
- **No format changes** to `chat_history.json`, checkpoints, delegation
  ledger, or `feedback_log.jsonl`. `chat_history.json` stays the
  replay/restore artifact and never becomes an LLM-facing surface.
- **Not** the analysis/sim `read_file` adoption follow-up from the
  file-tools arc — complementary, tracked separately.
- No retention/rotation policy in v1 (bounded lines ⇒ file grows
  slowly; revisit if a real session produces a problematic file).

## Phases

Each phase: offline tests (script-style — run `python tests/test_X.py`
individually, never whole-dir pytest) + a live test with a real model,
and **no regression to current functionality** before moving on.

### Phase 0 — writer (passive)
`scilink/session_events.py` + `append_event` wired into the four
`execute_tool` chokepoints + bindings in `chat()` / `run_task`.
No prompt or tool-surface changes of any kind.
- Offline: `tests/test_session_events.py` — bounded fields, redaction,
  huge/binary result safety, unbound no-op, `use_event_log`
  save/restore, UTF-8 (non-ASCII summary round-trip), writer-exception
  isolation (a monkeypatched failing write must not fail the tool call).
- Live: short analysis session; assert `events.jsonl` exists, one line
  per tool call, valid JSON per line, no base64.
- Regression guarantee: request bodies byte-identical to `main`
  (capture-and-diff, per the shared-infra scoping practice).

### Phase 1 — access tools
`search_session_history` + `get_history_events` on all four
orchestrators (meta gets `scope` later, Phase 3).
- Offline: `tests/test_history_search_tools.py` — caps enforced
  (hits/chars/range), honest-null on zero matches, regex + plain
  substring, malformed-line tolerance (skip, don't crash).
- Live: seed a session past the trim threshold (temporarily lower
  `MAX_HISTORY_MESSAGES`), then ask about an early-turn detail; the
  agent must recover it via search rather than asking the user.
- Regression: prompts change only by the two new tool declarations.

### Phase 2 — trim marker integration
One-sentence pointer in the trim summary marker + mid-loop warning,
all four orchestrators.
- Offline: marker text asserted; marker unchanged when tools absent.
- Live: repeat the Phase-1 scenario without hinting the tool exists —
  the marker alone should route the agent to search.

### Phase 3 — meta reach-through + fan-out branch binding
`scope` param on the meta's search; branch-thread binding with
`branch` field.
- Offline: child-glob discovery incl. `delegations/<NN>_<slug>/`;
  hit labeling; union caps; branch records carry `branch`.
- Live: meta investigation with ≥2 delegations, then "what exact
  script did the analysis specialist produce in the first
  delegation?" — answered from child events, not from the ledger
  summary. Fan-out run (opt-in HITL off) shows branch-tagged events.

### Phase 4 (optional, measure first) — BO campaign check
Before building anything BO-specific, run a long BO campaign live and
check whether the generic search already prevents re-suggesting failed
conditions. Only if it doesn't, consider a campaign-facing convention
(e.g. the BO tools writing richer `summary` lines) — still no new
subsystem.

## Open questions (decide at implementation time)

1. Event `summary` derivation per tool family — a generic head-of-result
   is fine for v1; per-tool summarizers only if live tests show the
   generic gist is too lossy to search (decide from evidence, not
   upfront).
2. Should `scilink serve` (MCP) sessions bind the event log too? The
   `_execute_run_task_captured` path already runs through `run_task`,
   so it should come for free — verify in Phase 0's live test rather
   than special-casing.
3. UI: surface `events.jsonl` in the session panel? Backend first
   (hard-features-first sequencing); UI exposure is a follow-up PR.

## Figure (DONE 2026-08-14)

Updated `make_system_structure_figure.py` (Liu Fig-1 analog): the event
log belongs in the existing **"Sessions & checkpoints"** box — it is
session-scoped persistent state, not a skill, agent, or execution
resource. No new box/arrow. Edits made: title → "Sessions, memory &
checkpoints" (script ~lines 184–185); loop↔sessions arrow labels
"restore" → "restore / recall", "save" → "save / log" (~lines 225–228),
which is what marks the box as active, queryable memory rather than a
passive snapshot store. Both light/dark PNGs re-rendered.
