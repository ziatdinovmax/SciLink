# PR draft — `live-monitoring` branch

Copy-paste the body below when ready to open a PR against `main`. Branch is `live-monitoring`; 7 commits, all pushed.

Recommended PR title:

```
live-monitoring: real-time agent interpretation of in-progress measurements
```

---

## Summary

Adds a new **Live** mode to SciLink that lets an agent interpret a measurement *as it happens*. Today's workflow is batch: upload a finished scan, click Analyze, wait ~30–90 seconds for the planner / script / verifier cycle to finish. Live mode is for the case where the diffractometer is *still scanning* and you want the agent to:

- Track the pattern as new points arrive
- Tell you when the verdict changes (e.g. "the FOM just crossed into accept — looks like Si Fd-3m")
- Flag new peaks appearing that the current best candidate doesn't predict
- Stay quiet otherwise — no LLM call per data point

The user-visible flow:

1. Click **Live** in the mode selector.
2. Type the path to a file your instrument is writing to (CSV / TXT / etc.); pick a skill; set the tick interval.
3. Click **Start Live Session**.
4. Watch the dashboard update every couple of seconds. The decision feed fills in only when something interesting happens — verdict change, new peak, confidence reversal, or your manual "Interpret Now" click.

Architecturally, two loops do the work:

- **Fast tick loop** (every 2 seconds by default, pure Python, no LLM): polls the data source, runs the skill's tick function, updates the metric chart. Tens of milliseconds per tick — runs while your scan is still in progress.
- **LLM loop** (event-triggered): invoked only when a trigger fires — verdict change, new feature, confidence reversal, periodic heartbeat (default every 60 s), or manual request. Coalesces back-to-back events into one call so the LLM isn't spamming the feed.

The skill decides what "interesting" means for its modality. XRD: a new Bragg peak appearing outside the candidate's predicted set. Raman: a new band. EELS: a new edge. The framework only reads structured fields (`primary_metric`, `verdict`, `detected_features`) from each tick — no skill-specific logic in the live infrastructure.

### Wiring a skill into live mode

Any analyze-mode skill becomes live-mode-enabled by adding one block to its frontmatter:

```yaml
---
description: XRD structure matching ...
live_tick:
  enabled: true
  tick_fn: my_package.live_tick:my_tick
  trigger_overrides:                  # optional — skill-specific defaults
    heartbeat_sec: 30
    confidence_threshold: 0.65
---
```

And implementing the tick function (signature lives in `scilink/agents/exp_agents/live_session.py`):

```python
from scilink.agents.exp_agents.live_types import LiveTickResult

def my_tick(latest_data, session_state, skill_state) -> LiveTickResult:
    # parse the latest data chunk, score it, return a structured result
    return LiveTickResult(
        timestamp=time.time(),
        primary_metric=0.78,          # FOM, correlation, χ², peak count, …
        metric_name="figure_of_merit",
        verdict="accept",             # 'accept' | 'marginal' | 'reject' | 'unknown'
        detected_features=[{"position": 28.4}, {"position": 47.3}],
        notes="best match: Si Fd-3m",
    )
```

That's the entire skill-side contract. The dashboard, triggers, JSONL stream, LLM dispatch — all generic.

### Replay

Every tick + every trigger event + every LLM response gets written to `live_ticks.jsonl` under the session directory. You can replay a finished session against a different trigger policy to test thresholds without re-running the experiment, or re-fire the LLM call to debug a wrong interpretation:

```python
from scilink.agents.exp_agents.live_replay import replay_jsonl
from scilink.agents.exp_agents.live_triggers import default_policy, ThresholdCrossTrigger, TriggerPolicy

# What would have happened with a tighter confidence threshold?
report = replay_jsonl(
    "live_session_20260521_103045/live_ticks.jsonl",
    trigger_policy=TriggerPolicy(triggers=[
        *default_policy().triggers,
        ThresholdCrossTrigger(threshold=0.80, direction="above"),
    ]),
)
print(report.trigger_event_counts)
```

### Scope

v1 ships the infrastructure and a stub end-to-end test path; no real skill is wired in this PR. The XRD live tick function is a small follow-up (≈30 lines of `score_xrd_match_fast` + `extract_peaks` chaining + a one-line frontmatter block in `xrd.md`) that lands once the `structure-matching` branch is on `main`.

## Install

No new dependencies — the live infrastructure uses only the existing Streamlit + standard library; the parquet / pyarrow extras for structure-matching are unaffected.

## What to test (reviewer checklist)

- [ ] Offline tests pass: `python -m pytest tests/test_live_data_sources.py tests/test_live_triggers.py tests/test_live_session.py tests/test_live_replay.py tests/test_resolve_tick_fn.py -v` — 89 tests, 0 skipped.
- [ ] UI smoke (no LLM key needed for the setup view): `streamlit run scilink/ui/app.py`, click **Live**. The setup form should render; the skill picker is empty until a skill declares `live_tick` (expected — no skill does yet on this branch).
- [ ] Setup → dashboard transition (needs a tick function, so layer this on top of structure-matching once it merges): point at a synthetic file that a second script is appending to; verify the metric chart accumulates and the verdict pill changes colour as the appended data crosses thresholds.

---

## Technical details (for reviewers / framework maintainers)

### Module structure

New modules under `scilink/agents/exp_agents/`:

- `live_types.py` — `LiveTickResult`, `TriggerEvent` dataclasses. Standalone module so `live_session` and `live_triggers` can both import without a circular dep; also importable by skill `tick_fn` modules without pulling the orchestrator's heavyweight transitive deps (threading, LiteLLM, …).
- `live_data_sources.py` — `LatestData`, `LiveDataSource` protocol, and three concrete sources: `MtimePollFileSource`, `AppendOnlyFileSource`, `CallbackSource`.
- `live_triggers.py` — Six built-in triggers (`VerdictChangeTrigger`, `NewFeatureTrigger`, `ConfidenceReversalTrigger`, `ThresholdCrossTrigger`, `HeartbeatTrigger`, `ManualTrigger`) + `TriggerPolicy` composer + `default_policy()` and `from_overrides()` factories.
- `live_session.py` — `LiveSession` class + `TickFn` type alias.
- `live_replay.py` — `replay_jsonl()` + `ReplayReport`.

New UI component:

- `scilink/ui/components/live_panel.py` — Streamlit panel with `render_live_panel()` entry point. Includes inline `_render_setup` and `_render_dashboard` functions and an `_discover_live_enabled_skills()` helper that walks `list_all_skills()` and filters to those whose frontmatter declares `live_tick.tick_fn`.

Modified files:

- `scilink/ui/config.py` — adds `"live"` to `APP_MODES` (beta-tagged) and `"live": "live_session"` to `SESSION_DIR_PREFIXES`.
- `scilink/ui/app.py` — one early-exit branch at the top of the post-sandbox region: `if app_mode == "live": render_live_panel(); st.stop()`. The live panel handles both setup and dashboard inline; no integration with the analyze-style welcome screen / session-init machinery.
- `scilink/skills/loader.py` — `resolve_tick_fn(meta) -> Callable | None` helper added at module bottom. Strict on malformed dotted paths (raises with a clear message) so a skill that claims live-mode but ships a broken `tick_fn` fails loudly at session-construction time.

### LiveSession threading model

Three threads:

1. **Tick thread** — daemon, runs `_tick_loop()`. Each iteration: `source.read_latest()` → optional `tick_fn(latest, session_state, skill_state)` → append to in-memory deque + write JSONL line → `policy.evaluate(history)` → enqueue any fired `TriggerEvent`s. Sleeps `(interval - elapsed)` between iterations via `Event.wait()` so `stop()` wakes it immediately.

2. **Dispatch thread** — daemon, runs `_dispatch_loop()`. Blocks on `queue.get(timeout=0.5)`; on event arrival, drains any additional events that piled up during the wait (`get_nowait()` loop), then calls `orch.run_task(task=..., context=...)` ONCE with the coalesced events as context. **Single-flight is mandatory**: `AnalysisOrchestratorAgent.run_task` mutates `self.messages` (`analysis_orchestrator.py:1293–1392`) and is not concurrent-safe — the dispatcher's serial-by-construction loop is the only thing standing between live mode and message-list corruption.

3. **Main / UI thread** — reads `latest()` / `history()` / `llm_busy`, invokes `force_interpretation()`, receives `on_tick` and `on_llm_response` callbacks. Never blocks on either worker.

Locking:

- `_history_lock` (Lock) — protects the rolling deque against tick-thread writer + UI-thread readers.
- `_jsonl_lock` (Lock) — serializes JSONL writes between the tick thread (ticks + triggers) and the dispatch thread (llm_response). One line per `write()` to avoid partial lines.
- `_stop` (Event) — graceful shutdown; both worker loops poll it.
- `_llm_dispatcher_busy` (Event) — read-only flag exposed as `session.llm_busy` for the UI.

Robustness:

- `tick_fn` exceptions are logged and the loop continues (so a transient parse error on a malformed CSV line doesn't kill the session).
- `tick_fn` returning a non-`LiveTickResult` is logged and skipped (not crashed).
- `source.read_latest()` returning `None` is the no-new-data path; tick is skipped silently.
- `run_task` exceptions are logged and surfaced as a JSONL `llm_response` with `status="error"`. Session keeps running.
- `on_tick` / `on_llm_response` callbacks that raise don't crash the worker — wrapped in `try/except` and logged.
- A buggy trigger in the policy is isolated at the `TriggerPolicy` layer (`live_triggers.py`) — one broken third-party trigger can't take down the whole session.

### JSONL schema

One line per record. `kind` discriminates:

```json
{"kind": "session_start", "timestamp": 1737485845.0}
{"kind": "tick", "timestamp": ..., "metric": 0.78, "metric_name": "figure_of_merit",
 "verdict": "marginal", "detected_features": [{"position": 28.4}], "notes": "...", "raw": {...}}
{"kind": "trigger", "timestamp": ..., "name": "verdict_change",
 "details": {"from": "marginal", "to": "accept"}}
{"kind": "llm_response", "timestamp": ..., "trigger": "verdict_change",
 "status": "success", "summary": "...", "key_findings": [...],
 "files_produced": [...], "warnings": [...], "coalesced_events": ["verdict_change"]}
{"kind": "session_end", "timestamp": ...}
```

Lines are timestamp-ordered (validated by `test_jsonl_is_chronological`). `_serializable()` / `_jsonable()` normalize nested objects (Path, numpy scalars, dataclasses) so `json.dumps` doesn't crash on whatever the skill puts in `raw`.

### Replay determinism

`replay_jsonl(path, *, trigger_policy=None, speed=None, on_tick=None, on_event=None, orchestrator=None, llm_mode="skip")`:

- Reads only `kind="tick"` lines, rebuilds `LiveTickResult` in chronological order.
- Evaluates the (possibly-different) trigger policy → deterministic event sequence for the same fixture + same policy (validated by `test_replay_reproduces_trigger_events_deterministically`).
- `llm_mode="skip"` (default) echoes the original `llm_response` lines into the report without calling the LLM.
- `llm_mode="redo"` calls `orchestrator.run_task` per trigger with `context["replay"] = True` so downstream code can distinguish.
- `speed=None` runs instantly (no sleeping); `speed=1.0` matches real time; `speed=10.0` runs 10× faster. Per-gap sleep capped at 60 s to avoid surprises from sparse data.

### Trigger taxonomy

Each trigger implements `evaluate(history) -> Optional[TriggerEvent]` + `reset()`. Built-ins:

| Trigger | Fires on |
|---|---|
| `VerdictChangeTrigger` | First differing verdict in history (re-arms after each transition) |
| `NewFeatureTrigger(lookback=5, key=...)` | Element in `detected_features` not present in the previous N ticks. Default identity key is the full feature dict; XRD will typically pass `key=lambda f: round(f["position"], 1)` to round noise. |
| `ConfidenceReversalTrigger(window=5, min_reversal=0.05, direction="higher_is_better")` | Metric was monotonically improving for `window-1` ticks; reverses by ≥ `min_reversal`. Supports `direction="lower_is_better"` for MIP-cost-style metrics. |
| `ThresholdCrossTrigger(threshold, direction="above")` | First crossing in the configured direction; re-arms on the reverse crossing. No-spam at the boundary (hysteresis baked in). |
| `HeartbeatTrigger(interval_sec=60)` | Periodic. Doesn't fire on session start (other triggers go first). |
| `ManualTrigger` | `request()` sets a flag; the next `evaluate()` fires once and clears. UI calls `request()` from the "Interpret Now" button. |

`TriggerPolicy` composes a list and returns the union of events on each `evaluate()`. `default_policy()` builds the common set; `from_overrides(meta_dict)` applies a skill's `trigger_overrides:` block to the defaults.

### UI integration

`live_panel.render_live_panel()` is the single entry point. Two views:

- **Setup**: a `st.form` collecting data path, source kind, tick interval, skill (filtered to those declaring `live_tick`), and trigger toggles. On submit, instantiates `AnalysisOrchestratorAgent` + `LiveSession` and stashes both in `st.session_state.live_orch` / `st.session_state.live_session`.

- **Dashboard**: `st.fragment(run_every="2s")` (the same primitive used at `scilink/ui/app.py:539` for chat-task polling) containing:
  - Status header: verdict pill (🟢/🟡/🔴), metric value, `llm_busy` indicator, tick count.
  - Left column: `st.line_chart` of `primary_metric` over time; expander showing `detected_features` of the latest tick.
  - Right column: decision feed reading `st.session_state.live_session_feed` (populated by an `on_llm_response` callback the dispatcher thread invokes).
  - Controls: "Interpret Now" (calls `session.force_interpretation()`) and "Stop Session" (`session.stop(timeout=3)`, clears state, reruns).

App-level integration is one if-branch at the top of the post-sandbox section (`scilink/ui/app.py`) that short-circuits the welcome screen + tab UI when `app_mode == "live"`. No other mode's path is touched.

### Test coverage (offline, no LLM)

- `test_live_data_sources.py` — 18 tests: protocol conformance for all three sources, mtime / append / callback happy paths, no-change-returns-None, file-appears-after-construction, truncation, binary append, multi-append offset tracking, reset semantics.
- `test_live_triggers.py` — 26 tests: protocol conformance, each trigger's fire / no-fire / re-arm behavior, both directions for the cost-style reversal trigger, threshold-cross hysteresis, manual single-shot semantics, policy union, policy exception isolation, `default_policy` / `from_overrides` shape.
- `test_live_session.py` — 15 tests: lifecycle (start/stop idempotency, session_start / session_end lines), tick cadence, None-source path, tick_fn exception isolation, trigger → JSONL, chronological ordering, single-flight coalescing (3 rapid triggers → strictly < 3 LLM calls), `llm_busy` flag, `run_task` exception handling, `on_tick` + `on_llm_response` callbacks, manual trigger, no-op when policy lacks `ManualTrigger`, end-to-end with `MtimePollFileSource` + peak-count `tick_fn`.
- `test_live_replay.py` — 15 tests: tick-count reproduction, metadata preservation, ignoring non-tick lines, malformed-JSON resilience, deterministic trigger evaluation, custom-policy sensitivity, `llm_mode` skip / redo / validation, cadence respected at configurable speed, callbacks, full round-trip from a real `LiveSession` recording.
- `test_resolve_tick_fn.py` — 12 tests: all return-None paths, validation errors for malformed dotted paths, ImportError / AttributeError / TypeError for broken / non-callable targets, happy path including submodule packages.

89 tests, 0 skipped on a standard environment.

### Out of scope (follow-up branches)

- **XRD live tick function** — lands after `structure-matching` merges to `main`. ~30-line wrapper around `score_xrd_match_fast` + `extract_peaks` + a `live_tick:` block in `xrd.md`. No live-monitoring infra changes.
- **Multi-skill concurrent ticking** — v1 has one `tick_fn` per session. Users can still load multiple skills (analyze-mode behavior preserved); they pick the one driving ticks in the live setup form. Multi-instrument concurrent ticking (XRD + Raman simultaneously on the same sample) is a real lab value but doubles state + UI complexity; defer until single-skill usage clears.
- **WebSocket / vendor-API data sources** — the `LiveDataSource` protocol designed for it (`CallbackSource` is the bridge), but only mtime-poll / append-only / callback ship in v1.
- **Instrument-control hooks** — alert-only for v1. A future `on_action` callback can be added as another `LiveSession.__init__` kwarg without touching the rest of the architecture.
- **Adaptive trigger tuning** — the LLM doesn't auto-tune thresholds from observed false-positive rates. Triggers are operator-configurable per session.

🤖 Co-authored with Claude Code
