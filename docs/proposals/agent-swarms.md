# Proposal: swarms — concurrent agents that share what they find

Status: design notes, drafted 2026-09-28 against `main` at b988c7cd
(Release 0.0.83). Nothing implemented. Based on a source audit of the meta
agent (`meta_orchestrator.py`, `meta_orchestrator_tools.py`, `fanout.py`,
`telemetry.py`), the three mode orchestrators' `run_task`, the executors, the
human-feedback layer, the LLM wrappers and the persistent stores under
`~/.scilink`. Line numbers are as of that commit.

## Context — the ask, and what the word means here

The ask: run *swarms* of agents with SciLink, in the usual sense of the word —
"multiple autonomous or specialized agents coordinate, delegate tasks, and
share mutable context to solve complex problems in parallel rather than
relying on a single linear prompt."

That definition has four parts. SciLink has three of them today:

| Part | Today | Where |
|---|---|---|
| Specialized agents | the three mode orchestrators; foundation agents under them, specialized by skills | CLAUDE.md, "Foundation agents" |
| Delegation | meta → `run_task` on a child | `meta_orchestrator.py:1404-1496` |
| Parallel work | analysis fan-out (threads), hyperspectral series replays (processes), best-of-N candidates (threads), the live loop's background escalation (one subprocess) | `fanout.py:1919`, `hyperspectral_series.py:1322` |
| Shared mutable context | **none** — agents pass messages (the ledger, the `context` dict); branches exchange nothing at runtime | — |

It is missing the fourth part, and the missing part is not an oversight. The
fan-out's docstring states the reason: *"independence is what makes fusion's
agreement claims mean anything"* (`fanout.py:1-12`). So this proposal is
mostly about how to add shared context **without losing the ability to tell
an independent confirmation from an echo** — and about keeping a system of N
concurrent LLM agents stable, which is the harder half.

What a swarm is *not* here:

- **Not a fourth mode and not a new agent class.** It is a coordinator on
  the meta (the "meta-agent on top" CLAUDE.md allows) plus a shared record.
  Every worker is one of the existing three orchestrators.
- **Not a replacement for the same-task shapes that already work.** Many
  datasets of one kind stay in series mode (anchor + locked recipe; N agents
  each deriving their own method is exactly the drift series mode exists to
  prevent). Several analyses of the same data stay as best-of-N and audits,
  whose value is that the analyses cannot see each other. A swarm can run
  *those* as its work items; it does not re-implement them.
- **Not agent-to-agent messaging.** No agent addresses another. Agents post
  findings to one record and the coordinator decides what work follows. The
  "Alternatives considered" section explains why.

## What exists that a swarm builds on

- **Ephemeral workers.** Fan-out already builds a fresh, unregistered
  `AnalysisOrchestratorAgent` per branch in its own directory
  (`_make_ephemeral_analysis_child`, `fanout.py:836-865`), so concurrent
  branches share no orchestrator state. It is analysis-only; planning and
  simulation delegations go only through the serial `_delegate` path.
- **A coordinator loop.** The fan-out coordinator polls its futures every 5 s,
  prints a heartbeat, cancels overdue branches, and serves branch questions
  (`fanout.py:1946-1971`). It is a tool call that blocks, with no LLM in the
  loop — the right shape for a swarm coordinator too.
- **Admission and budgets.** Memory-aware admission (`_admit_branch`,
  `fanout.py:111-130`: hold a branch until its estimated working set fits,
  always admit when nothing runs), per-branch wall-clock budgets persisted on
  the ledger (`_budget_s`), caps (soft 5 / hard 8, `fanout.py:223-224`).
- **Cooperative cancellation.** A per-branch stop event raised as
  `AgentStoppedError` (a `BaseException`) on the branch's next print, plus a
  kill of the thread's subprocesses (`fanout.py:988-1034`); a user Stop
  reaches attributed worker threads the same way
  (`server/stdout_router.py:130-176`, `utils/log_context.py:124-146`).
- **Resume.** Stale branches are reopened under `_fanout_lock` and re-run from
  their own checkpoints (`resume_fanout`, `fanout.py:1279-1416`); every turn
  first sweeps `running` entries to `interrupted` (`meta_orchestrator.py:1564-1590`).
- **Provenance, partly.** Each ledger entry records `context_from` (declared,
  plus inferred from the task text, `meta_orchestrator.py:1623-1669`) and,
  where independence was spent, `informed_by` / `informed_via`
  (`co_registered_operands`, `steering`, `fusion_feedback`). Fusion reads
  these and tells the LLM to discount agreement (`fanout.py:2855-2886`).
- **A queued question channel.** `QueueChannel` (`hitl.py:142-193`) parks
  questions from worker threads and lets a coordinator serve them one at a
  time with a `[branch: label]` prefix. Fan-out wires it up
  (`fanout.py:1911-1917`) but only when `orch.fanout_branch_hitl` is set, and
  nothing in the repo sets it (`fanout.py:813`).
- **Usage accounting, coarse.** Every LLM call is counted (`tracing.note_llm_call`,
  `tracing.py:174-197`) into process-wide counters and an optional per-session
  `UsageLedger` (`usage.py:25-126`); `SCILINK_TOKEN_BUDGET` refuses new web
  work over budget (`server/ops.py:135-141`).

## The design constraint: independence is load-bearing

Five places in SciLink draw their meaning from agents *not* seeing each other:

1. **Fan-out fusion.** Agreement across branches is evidence only if the
   branches reached it separately.
2. **Best-of-N.** Candidates must differ for the judge's pick to matter;
   candidate jobs already suppress human feedback so they stay separate.
3. **Live-loop audits.** "An audit never recalls: it must be independent"; "a
   vote is not the truth" (CLAUDE.md, live loops).
4. **Judges and critics.** A critic that has read the author's reasoning is
   a second author.
5. **Replay gates.** They judge on evidence against the anchor, not on a new
   opinion.

Shared context between these would give N agreeing copies of one answer.
So the rule this proposal is built on:

> **Shared context flows into work, never into checks.** Every read of the
> shared record is recorded, and any count of agreement counts only sources
> whose recorded reads do not include each other.

Today the discount is prose: fusion is *told* about `informed_by`
(`fanout.py:2855-2886`). A swarm makes it structural: the read graph is data,
so "how many independent sources support this claim" is computed, not asked.

## Where concurrency breaks today

The audit found the code is correct for what it does now — one delegation at
a time, plus analysis-only fan-out on ephemeral children — and wrong in
predictable ways for anything more. Grouped by what would fail:

### A. A child orchestrator is single-threaded

- **`run_task` deltas are before/after windows over shared state.** Analysis:
  `n_before = len(self.analysis_results)` (`analysis_orchestrator.py:1535`),
  `new_analyses = self.analysis_results[n_before:]` (`:1602`). Planning: a
  filesystem snapshot of `base_dir` (`planning_orchestrator.py:1478`, `:1547`).
  Simulation: `generated_structures[n_before:]`
  (`simulation_orchestrator.py:570`, `:622`). Two concurrent calls on one
  instance each report the other's output as their own.
- **`run_task` saves, mutates and restores instance settings around
  `chat()`** — autonomy mode, `max_iterations`, default profile/targets/time
  budget (`analysis_orchestrator.py:1540-1590`; planning `:1498-1535`;
  simulation rewrites `self.messages[0]`, `:585-608`). Overlapping calls
  restore each other's values.
- **The children's checkpoints are not atomic.** All three do
  `open(checkpoint_path, 'w')` + `json.dump` with no lock
  (`analysis_orchestrator.py:1714-1734`, `planning_orchestrator.py:1664-1696`,
  `simulation_orchestrator.py:834-852`; the chat histories likewise). The
  meta's checkpoint is atomic and locked (`meta_orchestrator.py:1997-2009`).
  A crash mid-write leaves a truncated checkpoint that restore cannot read.
- **"Latest" defaults.** Six analysis tools default `analysis_index=-1`
  (`get_recommendations`, `refine_interpretation`, `reenter_interpretation`,
  `assess_novelty`, `recommend_simulations`, `run_dft_workflow`,
  `analysis_orchestrator_tools.py:4234-5430`), `derive_from_outputs` uses the
  newest record (`:2011-2016`), and `run_simulation` reuses the newest
  structure (`simulation_orchestrator_tools.py:1027-1037`). With concurrent
  writers, "latest" is whoever finished last.

**Consequence for the design:** a swarm worker is always an ephemeral
orchestrator on its own directory, never the persistent child. The persistent
children stay for the conversational path, where context accumulating across
turns is the point.

### B. The meta assumes one delegation at a time

- The chat loops run tool calls in a plain `for` loop
  (`meta_orchestrator.py:2292-2305`), so no two `_delegate` calls overlap
  today. Nothing locks them either: `_open_delegation` computes
  `index = len(self._delegation_ledger) + 1` (`:1505`) with no lock; only the
  fan-out caller holds `_fanout_lock` around it.
- `get_delegation_history` returns the whole ledger, private fields
  (`_budget_s`, `_started_at`, `_branch_tid`) included (`:1897-1902`).
- Telemetry reads only the analysis and planning children
  (`telemetry.py:251-252`, `:297`); simulation and fan-out workers are
  invisible there, and nothing records per-agent tokens or durations.

### C. State shared by the whole process

- **Sandbox approval** is one module bool, never reset
  (`_GLOBAL_SANDBOX_APPROVED`, `executors.py:66`, set at `:237-271`); three
  sites set `UNSAFE_EXECUTION_OK` in the process env
  (`mcp_server.py:1343`, `hyperspectral_series.py:1306`, `live/_reanchor.py:43`).
  One approval approves every agent.
- **Credentials** are written into `os.environ` per session
  (`server/session_manager.py:85`; also `litellm_wrapper.py:163`,
  `cli/shell/bootstrap.py:73`), and `APIKeyManager` is a module singleton
  with no lock (`auth.py:230`). `sandbox_env()` builds a generated script's
  environment from an allowlist over the *live* `os.environ`
  (`executors.py:325-342`), so it sees whatever another thread last wrote.
  (Correction to the hosted-campaigns proposal, item 1: the
  `env = os.environ.copy()` it cites is gone; the allowlist replaced it.
  The live-read remains.)
- **`builtins.input` is swapped process-wide for the duration of a turn**
  (`server/runner.py:366`, `cli/shell/turn.py:233`, `ui/app.py:639`), and
  `set_default_channel(AutoAcceptChannel)` in headless mode switches every
  thread without an override (`cli/shell/headless.py:79`). Any worker that
  falls back to the default channel or to raw `input()` goes to whichever turn
  swapped last.
- **`ExecutionTimeout`** uses SIGALRM on the main thread
  (`executors.py:666-668`), which is process-global.

### D. Human feedback has one slot

- `ParkingChannel.ask` stores one question in `turn.pending_question`
  (`server/hitl_channel.py:74`) and blocks with no timeout (`:77`);
  `TurnState.pending_question` is a single field (`server/runner.py:135`).
  A second concurrent question overwrites the first.
- `pending_question.json` is one sidecar per log directory (`hitl.py:274-283`).
- `QueueChannel` has a `timeout_s` but fan-out creates it without one
  (`fanout.py:1916`), so a branch waiting on a human waits indefinitely.
- EOF on a question propagates (`hitl.py:337-343`) and the 26 call sites
  handle it inconsistently (a few catch `EOFError`; e.g.
  `base_controllers.py:480` catches only `KeyboardInterrupt`).

### E. Shared stores are last-writer-wins

`atomic_write_text` (`utils/text_io.py:39`) prevents torn files, and its
docstring says plainly that it is not a lock. Per store:

| Store | Today | Concurrent failure |
|---|---|---|
| Script bank | per-domain `fcntl` lock on the mutating calls (`_script_bank.py:97-130`) | `log_assist` append and the `auto_archive` stamp are outside it (`:1132-1191`) — minor |
| Distill staging, graduation | atomic writes, unlocked read-modify-write (`_staging.py:147-166`, `_graduation.py:363-425`) | lost updates |
| `kb_store` | fixed `.staging_<name>` dir, `rmtree`'d at the start of every build (`kb_store.py:325-588`); fixed `<name>.bak` in the swap | two builds of one KB delete each other's staging |
| Instrument home | plain `write_text`, `used()` counter read-modify-write (`instrument_home.py:73-168`) | torn records, lost counts |
| `sessions.jsonl` | in-process lock, fixed tmp name (`sessions.py:29`, `:57-66`) | two processes collide on the tmp |
| SAM weights | `urlretrieve` straight to the final path, guarded by `os.path.exists` (`particle_analyzer.py:125-130`) | a second worker loads a half-downloaded `.pth` |
| DCNN weights | fixed `dest + ".part"` then `os.replace` (`atomistic_model_manager.py:113-133`) | two downloaders write one `.part` |

The SAM race is reachable **today**: two image branches of one fan-out on a
fresh machine.

### F. The LLM layer has no concurrency control

- Retries are LiteLLM's (`num_retries=4`, `litellm_wrapper.py:298`, `:439`),
  with no SciLink-side rate limiter, semaphore or 429 handling
  (only the KB embedding path handles `RateLimitError`,
  `knowledge_base.py:160-180`). `LiteLLMChatSession.send_message` passes
  neither `num_retries` nor `timeout` (`litellm_wrapper.py:799-806`);
  `litellm_completion` sets no timeout. N workers hitting one rate limit
  retry in step.
- Token counters are process-wide (`tracing.py:103-106`), attributed at
  best per session through a thread-local tag (`tracing.py:133-151`), and
  calls from spawned processes are not seen (`tracing.py:200-207`).
  Nothing is attributed per agent; nothing computes cost.

### G. Cancellation is cooperative and shallow

- A stop fires on a worker's next print. A worker inside a long LLM call
  (the generative wrapper's default timeout is 1200 s,
  `litellm_wrapper.py:368`) does not notice for up to that long.
- `ScriptExecutor` kills only its direct child on timeout — no process
  group (`executors.py:448-452`), so a script's own subprocesses survive.
- A single (non-fan-out) delegation has no cancel of its own; the meta has
  no cancel API.

## Design

### 1. The unit: an ephemeral worker per work item

A **work item** is `{mode, task, context, subject, budget, reads_board}`. The
coordinator runs each on a fresh orchestrator of its mode in
`<meta_session>/swarm/<NN>_<slug>/`, generalising
`_make_ephemeral_analysis_child` to planning (`data_dir=None`, the
construction that does not require one) and simulation (the `ase` import stays
inside the function, as `_get_simulation_child` does). Each worker has its
own instance, so the `run_task` windows in A are correct by construction and
no child state needs a lock. Each gets a ledger entry opened under a lock,
exactly as fan-out branches do.

Workers are **threads** by default, like fan-out branches. A work item moves
to a spawned **process** (the `ReplayPool` shape,
`hyperspectral_series.py:1322-1368`) when its memory estimate is large or it
must set process-global state — the fan-out docstring records the meta being
memory-killed twice by two concurrent large-cube branches
(`fanout.py:62-66`), and in one process an OOM kills every worker.

`subject` is the thing the item is about (a sample, a dataset, a structure,
a hypothesis) — the partition key the hosted proposal calls a *workstream*.
Findings, "latest" defaults and follow-up limits are all scoped by it.

### 2. The board: an append-only record of findings

The shared context is a **board**: `<meta_session>/swarm/board.jsonl`, one
record per line, with an in-memory index in the coordinator.

```
finding_id    stable id
author        worker id + ledger index + mode
subject       the partition key
kind          claim | measurement | recipe | structure | parameter_point |
              hazard | task_request | retraction | supersedes
payload       small and typed; files by path (never inline arrays)
evidence      analysis_ids, files, the gate that passed it
status        provisional | verified | retracted | superseded | tainted
reads         finding_ids the author had read when it produced this
board_version the board length the author's snapshot was taken at
created_at
```

**"Mutable" means superseded, not edited.** A record is never changed in
place; a correction is a `supersedes` or `retraction` record, and the current
view is a fold over the log. That gives the definition's *mutable* shared
context with a complete history, one writer, and no last-writer-wins.

**One writer.** Only the coordinator appends; workers post through an
in-process call (threads) or a queue (processes). One writer rules out the
store races in E, and the append is flushed before the coordinator
acknowledges it.

**Reads happen at defined points, and are recorded.** A worker opts in with
`reads_board` and reads at the start of its task and, optionally, at stage
boundaries its orchestrator exposes — never mid-LLM-call. A read returns a
filtered snapshot (by subject and kind) rendered into the prompt as one block,
the way `_steering_block` is today (`fanout.py:882-919`), and stamps the ids
it returned on the worker's read set. What an agent saw is therefore on
record, even though the interleaving of a concurrent run is not reproducible.

**Only verified findings propagate by default.** `verified` means the author's
own pipeline passed it: the analysis QC and replay gates, a human-approved
plan, a validated structure. The board adds **no new LLM judge** — that would
be one more source of the judge variance the series work had to engineer
around. A provisional finding is readable only by an item that asks for it
explicitly, and the read is marked as such.

**Checks never read.** A work item that is a check — a best-of-N candidate,
an audit, a judge, a critic, fusion verification — runs board-blind. This is
enforced structurally: the job carries a `check` tag and the read API refuses
it, rather than relying on each prompt to say so.

**Board context is additive.** A board read may add a hypothesis to consider, a
region to look at, a parameter point to try. It never sets a gate, a fit
window, a threshold or a target. Today the additive-only guardrail is prompt
text (`fanout.py:909-918`). On the board it is a type rule: those payload kinds
are delivered as hints, and gate parameters are never sourced from them.

**Independence is computed.** `independent_support(claim)` counts the authors
posting agreeing findings whose read sets, closed over transitively, do not
include another supporter's finding. Fusion and any consensus summary report
that count beside the raw count. The existing `informed_by` stamps become
reads on the board (a steering payload and a fused result are findings), so
the prompt-level discount in fusion becomes a number the prompt only renders.

### 3. The coordinator: deterministic reactions, not agents reacting

In the definition, agents "coordinate". Here only the coordinator schedules
work, and it does so by rules the meta declares when the swarm launches.

- **`run_swarm(items, subscriptions, budget)`** is a meta tool that blocks
  like fan-out's coordinator: it admits items, polls, serves questions,
  appends board records, applies subscriptions, enforces budgets, and returns
  a summary when the queue drains or the budget is spent. **There is no LLM in
  the coordinator loop.**
- **A subscription** is a declared rule: *when a verified finding of kind K on
  subject S appears, enqueue item template T with that finding in context.*
  For example: a phase-transition claim from an analysis enqueues a
  simulation of the two phases. Matching is deterministic.
- **Workers ask, the coordinator decides.** A worker that wants more work
  posts a `task_request`. It becomes an item only if a subscription or the
  budget allows it. Workers never spawn workers, so the depth is 1.
- **A running worker is never interrupted by a finding.** It reads the board
  at its next defined read point if it opted in. Otherwise the reaction is new
  work. This is what keeps the system analysable: an agent's inputs are fixed
  for the duration of its run, apart from its recorded reads.
- **Between swarm runs, the meta LLM reads the board** (a
  `get_board(subject, kind)` tool, filtered, private fields dropped) and may
  launch more. That is the "delegate tasks" part, and it stays at turn
  granularity.

### 4. Human in the loop

- Every worker thread gets its own thread channel (fan-out's `_BranchChannel`
  pattern), so no worker ever falls back to the process default or to
  `builtins.input`.
- The questions go to one **multi-slot queue** keyed by worker. This is
  `QueueChannel` with the coordinator serving it, extended to hold several
  questions at once. `ParkingChannel`/`TurnState` grow from one
  `pending_question` to a list tagged by `origin`, and the presenter shows
  which worker asks and about which subject. One `pending_question.json` per
  worker directory.
- Every question has a timeout that resolves to the gate's own default and is
  recorded as `timed_out` — the behaviour `QueueChannel` already has but
  fan-out does not use. A worker must not hold a slot and a budget forever
  on an unanswered prompt.
- Autopilot shows the swarm plan up front (items, subscriptions, budget),
  like the fan-out confirmation. At most `max_open_questions` questions are
  surfaced at a time; the others wait in the queue. Human attention is the
  scarcest resource in the swarm, and N workers each asking "approve this
  plan?" is the fastest way to have every plan waved through.

### 5. Scheduling and budgets

- **Admission** generalises `_admit_branch`: slots per mode, a memory
  estimate per item, and a hold on anything over the machine's headroom. It
  still always admits when nothing is running, so there is no deadlock.
- **An LLM concurrency limiter** sits in the wrappers (process-wide, per
  provider/model): a semaphore on in-flight calls, plus backoff with jitter
  on 429s so N workers do not retry in step. A circuit breaker pauses
  *admission* (not running work) when errors from the provider across
  workers pass a threshold.
- **The swarm budget** covers tokens, wall-clock, item count and follow-ups
  per subject. Tokens are reserved at admission from a per-mode estimate and
  reconciled on completion; an item that would exceed the remainder waits or
  is dropped with a reason. The meta owns the envelope, and the per-item
  budgets (`time_budget_s`, the QC profile) sit inside it.

## Robustness and stability

A swarm fails differently from one agent: the failures are interactions. Each
one below has a mechanism, and most are cheap.

**1. One worker's failure stays in that worker.** Exceptions become an error
result per item (as `_delegate` and fan-out already do). A timeout cancels
only that item. An item likely to exhaust memory runs in its own process,
and on cloud resources in its own task (see "Running on cloud resources"):
on 2026-09-28 two concurrent atomic-resolution image analyses on an 8 GB
laptop reached 8.3 and 6.4 GB and froze the whole machine, where a
per-worker memory limit would have ended one worker.
A degraded item is reported as degraded and excluded from consensus, as
fan-out already excludes a degraded branch from fusion. The swarm's result
lists what failed beside what succeeded.

**2. Everything is bounded.** Bounded: concurrent workers, total items per
swarm, depth (1), follow-ups per subject, open questions, tokens, wall-clock,
and board reads per item. SciLink learned this in plan mode: an advisory
critic that is asked again returns a different set of findings every time, so
chasing it never converges (CLAUDE.md, "A plan the human approved is
settled"). A swarm of agents reacting to each other is the same loop with
more participants. The budget is the termination proof, not a safety net.

**3. Reaction cycles are detected, not hoped against.** Every item carries its
causal chain: the finding ids that triggered it and theirs. The coordinator
refuses a follow-up whose chain already contains the same
`(mode, subject, kind)`. It also caps how many times one subject can be
re-triggered. So A's finding can trigger B, and B's can trigger C, but B's
cannot trigger A on the same question again. Oscillation (two agents
alternately superseding each other's claim) shows up as a supersede chain on
one subject. Past a length of two, the coordinator stops and reports the
disagreement instead of scheduling a third round.

**4. Errors do not cascade silently.** Only verified findings propagate by
default. When a finding is retracted or superseded, the read graph gives its
dependents, which are marked `tainted`. The coordinator re-runs them if the
budget allows; otherwise it lists them in the result as resting on a
withdrawn finding. A `hazard` finding (a blocking plan conflict, an unsafe
parameter) is always delivered to readers on its subject, whatever their
filters.

**5. Agreement cannot be inflated.** `independent_support` is what consensus
reports. Checks are board-blind. A majority of LLM verdicts is never an
acceptance rule: "a vote is not the truth", and the live loop's
`audit_split` / `contested` outcomes are the model. When independent sources
disagree, that is a result to show, not a tie to break.

**6. The provider is a shared resource.** The limiter and jittered backoff are
covered in §5. Before any swarm, the chat-session path needs retries and a
request timeout (`litellm_wrapper.py:799`), and so does `litellm_completion`.
A hung call must end on its own, because nothing else will end it.

**7. Liveness.** Stop is checked before and after every LLM call, not only on
print. Every LLM path has a timeout. Scripts run in their own process group
(`start_new_session=True`) and are killed as a group. A watchdog flags a
worker that has emitted no event within N minutes. Questions time out (§4).
Together these bound how late a stop can land.

**8. Durability and resume.**
- The board is append-only and flushed per record. A torn last line is
  skipped on load, the way `recipes()` skips unparsable records.
- The coordinator's state (queue, budget spent, subscriptions) goes into the
  meta's atomic checkpoint.
- Workers checkpoint atomically (fixing A).
- Item ids are stable, so a resumed swarm re-runs only the items that were
  `running`/`interrupted` (the fan-out resume path) and never repeats a
  completed one.
- A worker posts findings only at commit points, so a crash leaves nothing
  half-posted.

**9. Shared stores get locks.** A small `store_lock(path)` helper in the
script bank's `fcntl` pattern goes around the read-modify-write sites in E:
staging, graduation, instrument home, the `kb_store` manifest and staging
(with a per-build unique staging dir), and `sessions.jsonl` (unique tmp). A
`download_once(url, dest)` helper writes to a unique temp file and holds a
lock while downloading, and SAM, DCNN and the COD cache all use it.

**10. No state crosses workers through the process.** Sandbox approval
becomes a value on the session (or coordinator) that the workers inherit, not
a module bool. The `UNSAFE_EXECUTION_OK` env writes are replaced by passing
it through. Credentials are passed as client kwargs, not written to
`os.environ`. `sandbox_env()` builds from a snapshot the session owns. Items
that must still touch process state run as processes. This overlaps the
hosted proposal's must-fix list and should be done once, for both.

**11. What happened can be reconstructed.** Every item records its model,
profile, board version and read set, and every board record its author and
evidence. The interleaving of a concurrent run cannot be replayed, but what
each agent knew when it acted can always be explained, which is what a
scientist reviewing the result needs.

**12. Observability.**
- Every LLM call is attributed to a worker: a thread-local agent tag next to
  the session tag, and spawned processes report their counts back on exit.
- Telemetry covers every worker, not just the two persistent children.
- The coordinator's heartbeat names what each worker is doing (the existing
  activity label).
- The board is the single record of the run.

**13. Graceful degradation.**
- A swarm of one item is an ordinary delegation.
- With the board unavailable, items run on their `context` alone and the
  result says so.
- With the provider throttling, admission slows while running work continues.
- With the budget spent, the swarm returns what it has.

## Extension: several instruments

The swarm above is shaped for work items that finish. The live layer can join
it later, one loop per instrument, without changing the design; it needs one
new worker kind and a few rules. Two facts from the code shape this:

- **The live layer already stands apart from sessions.** `scilink.live` has
  no chat, session or orchestrator imports, and a loop's contract with the
  outside is `loop_log.jsonl` and its events (`novelty`, `breach`,
  `discovery`, recommendations; CLAUDE.md, live loops). A swarm can watch a
  loop without the loop knowing it exists.
- **Today it is one instrument per session.** The web tab keeps one live run
  per session (`live_api._RUNS[session.id]`, `server/live_api.py:26`,
  `:899-951`), and a loop has one escalation slot (`measurement_loop.py:1323`).
  Several instruments at once is new work, not a switch.

### A long-running worker

Every item in §1 finishes and returns; a measurement loop runs until it is
stopped. An **instrument worker** holds one `MeasurementLoop` for one
instrument, for the length of the experiment, plus a bridge that reads the
loop's log and posts its events to the board as findings. A `novelty` event
becomes a finding on the sample, carrying where on the axis the change is and
its onset (abrupt or gradual). A `discovery` event becomes a claim, with its
novelty score and the literature question it was scored against. A
recommendation becomes a `parameter_point`. The loop's two clocks, its gates,
drift monitor and audits are unchanged: the bridge only reads the log.

### The frame path never waits on the swarm

The per-frame fast path makes zero model calls, and it must stay that way.
Board reads and coordinator work happen only on the slow clock: re-anchors,
audits, recommendations and what runs during a pause. A slow or unavailable
coordinator or board delays cross-instrument reactions; it never delays a
frame. The bridge posts without blocking: if the coordinator is behind, events
queue in the log, which is where they are recorded anyway.

### Cross-instrument subscriptions produce recommendations, never actions

SciLink recommends and never actuates (`live/instruments.py:24`,
`measurement_loop.py:26`). That holds across instruments. A subscription such
as *novelty on the SEM at region R → recommend an EELS acquisition at R*
produces a recommendation on the second instrument's side, in the schema it
already uses. A person, or a driver the person wrote, decides whether to act
on it. Pausing one instrument because of another's finding goes through the
same decision a pause needs today (`on_pause` is required, and a pause is
recorded as `paused` / `resumed`).

### The subject is the sample, with coordinates

Linking two instruments' findings needs to know they saw the same place at the
same time. Two additions:

- **A shared time base.** Each loop counts its own steps. Board records from
  instruments carry wall-clock timestamps from one clock, and the instrument's
  own timestamp when it supplies one.
- **A registration between coordinate frames.** This is fan-out's
  `co_registered` join, carried from finished datasets to running streams.
  It is declared per instrument pair (a transform, or "unknown"). Without it
  a subscription can match "same sample" but not "same region", and the
  coordinator must not claim more than that.

A finding's subject therefore becomes `(sample, region?)`, with region in the
frame of the instrument that posted it and translated only through a declared
registration.

### Priority

Instrument time is the expensive resource. A loop's slow-clock work has a
deadline: a paused instrument is waiting on it, or the next frame will arrive
before the answer is useful. A batch analysis item has no deadline. The LLM
limiter and admission (§5) need at least two priority classes, so ten batch
workers cannot make an instrument wait. Within the live class, a paused
instrument goes before a running one.

### One SciLink per instrument

The expected direction is one SciLink per instrument, the "lab of labs"
(CLAUDE.md: "The live layer is written for an instrument-centric SciLink").
The board is then shared across processes, and possibly across machines, so
exposing it over MCP stops being optional (see "Alternatives considered").
Each instrument's SciLink posts and reads through one board server, with the
same single-writer rule: the board server is the writer, and instances are
its clients.

### Robustness specific to instruments

- **Coupled pauses can deadlock.** Instrument A is paused waiting on an
  analysis of B's data while B is paused waiting on A. Rule: a pause may
  not wait on work that is itself waiting on another paused instrument. The
  coordinator checks the wait graph when a pause is requested and, on a
  cycle, hands the decision to the person instead of holding both.
- **Independent monitors are the strongest evidence.** Each loop's change
  monitor reads only its own data, with no model, and never reads the board.
  So two instruments that flag the same region at the same time are
  independent by construction. The board records that agreement at full
  weight, the best-founded kind of agreement it can hold.
- **Instrument memory stays per instrument.** Recipes and accepted states
  live in `~/.scilink/instruments/<id>/` (`live/instrument_home.py`), outside
  the board. The board holds what happened in this experiment. The instrument
  home holds what the instrument has learned over its life. They stay
  separate so that one experiment's board cannot overwrite what an
  instrument knows. Graduating a finding into instrument memory stays a
  reviewed step, as the hosted proposal's open question on shared
  instrument memory suggests.
- **One lost instrument stops only its own worker.** If an instrument's MCP
  server disconnects, that worker ends and the result lists the instrument
  as lost at that step. The other loops continue. A subscription that waited
  on the lost instrument resolves to "not measured", never to a default.
- **An unregistered pair never guesses.** If two instruments have no
  declared registration, a region-level subscription between them does not
  fire. A sample-level one still does, and its recommendation says the
  region is unknown.

## Running on cloud resources

The swarm is meant to run on cloud resources, AWS first. The hosted-campaigns
proposal already fixes the outer shape: one ECS/Fargate task per campaign, the
campaign's files on its own EFS access point, and model weights on a shared
read-only one. A swarm lives inside one campaign. What changes is where a
worker runs and what a worker may assume.

### Two placements for a worker

- **In the campaign task.** The coordinator and its workers share the
  campaign's task, as threads and processes. This is the simple shape, and
  right for items bound by LLM calls: planning, curve fitting, a
  simulation's input generation. It is bounded by the task size, and the
  spike's tasks were 2 vCPU / 8 GB, the same memory as the laptop that froze.
- **In a worker task.** A heavy item (an image or datacube analysis, a
  series replay pool, an MLIP run) runs as its own ECS task. It uses the same
  image and the campaign's EFS access point, and its size is chosen for the
  item's class. The coordinator stays in the campaign task and starts the
  worker task with the item spec, the way the live loop's re-anchor hands
  its spec to a subprocess today (`live/_reanchor.py`).

Worker tasks buy the isolation this proposal asks for. An out-of-memory kill
is confined to the task's cgroup, so it ends one worker and not the campaign.
The coordinator reads the task's stop reason and records the item as
`out_of_memory`. The item budget then decides: retry once at the next task
size, or report the item as failed.

### Memory is declared per item class, not guessed from the input

Fan-out's admission estimates memory from input bytes (`_branch_mem_estimate`:
6x the input, with a 0.5 GB floor). The 1024 x 1024 image that froze the
laptop is about 8 MB, so the estimate was 0.5 GB. The generated scripts that
ran on it peaked at 6 to 8 GB, and what they loaded was the DCNN ensemble
and its working arrays, not the input. The estimate cannot see that, and on
a cloud task the error is the difference between a run and an OOM kill.

- **Measure every item.** Record the peak resident memory of each item's
  scripts: `ru_maxrss` of the sandbox subprocess, or the task's memory
  metric. Keep it against `(mode, skill, data shape)` in a small table on
  the campaign volume.
- **Size from what was measured.** Admission and task sizing read that
  table. An unknown class gets a conservative default, and the first run
  of a class is its measurement.
- **Let the cgroup be the cap.** `SCILINK_SANDBOX_MEM_MB` (`RLIMIT_AS`)
  stays off by default, as `_sandbox_limits` documents: it breaks CUDA and
  Metal, which reserve more address space than they use. On a CPU task the
  cgroup is the cap that holds. On a laptop nothing caps a script today.

### The board across tasks

A worker task shares no memory with the coordinator, so it cannot append to
the board in-process. The first version keeps the single-writer rule without
a network API:

- Each worker writes only its own `findings.jsonl` in its item directory
  on EFS.
- The coordinator polls those files and appends them to the board.
- The board file itself is written by the coordinator alone.

Reads work the same way in reverse: a worker gets its board snapshot in its
item spec at start, and at a stage boundary it reads a snapshot file the
coordinator refreshes. A board service (behind MCP, as the instruments
section needs) replaces the files only when a worker must read the board
mid-run.

### Shared files on EFS

- **Locks.** `path_lock` is `flock`. On Linux, the NFS client emulates
  `flock` with the POSIX byte-range locks that EFS supports. The spike saw
  no lock errors on the script bank, but one run is not a load test. Stage
  4's tests include concurrent `path_lock` holders on EFS from two tasks.
- **Model weights.** The shared models access point is read-only in the
  hosted design, so `download_once` cannot fill it at runtime. Weights are
  loaded into it when the image is built or by an admin job. A worker that
  finds a weight file missing fails with a clear message instead of
  downloading into its own volume.

### Provider quotas are shared by every task

Bedrock limits requests and tokens per minute per account and region.
Every worker in every campaign on that account draws on the same quota, and
the LLM limiter in §5 is per process, so no single task can see the whole
load. Two things hold regardless:

- **Backoff survives shared throttling.** Throttling comes back as a
  retryable `RateLimitError`, with jittered backoff (stage 0), so tasks
  throttled together do not retry together.
- **The circuit breaker works per task.** It pauses admission when errors
  from the provider pass a threshold, and each task can see its own error
  rate.

The quota share is a control-plane setting: a per-campaign concurrency and
tokens-per-minute allowance, passed to the campaign task, which its
limiter enforces.

### Cost and liveness

- **Compute is a third budget.** The swarm budget adds task-seconds per
  item class to tokens and wall-clock. A worker task is billed for as long
  as it runs.
- **The coordinator reconciles against ECS.** A worker task can end
  without writing a result: an OOM, a Spot reclaim, a lost host. The
  coordinator checks task status, not only a thread. An item whose task
  stopped is `interrupted`, and the resume path re-queues it, as it does
  today for fan-out branches after a restart.
- **Spot is for idempotent items only.** Replays and re-runnable analyses
  can go on Fargate Spot. Anything holding a human question stays on
  on-demand.
- **Credentials.** A worker task uses the campaign's task role (hosted
  phase two, item 4), and its item spec carries no secrets.

## Build order

Backend first, UI last, per CLAUDE.md's sequencing rule. Stage 0 is worth
doing whether or not the swarm is built. It fixes defects the current fan-out
can already hit.

0. **Fixes for today's parallelism.**
   - Atomic child checkpoints and chat histories.
   - `download_once` for SAM, DCNN and COD.
   - A lock on `_open_delegation`.
   - SciLink-side LLM retries with backoff and a timeout on every path
     (LiteLLM's default retry waits nothing between attempts).
   - Process-group kill in `ScriptExecutor`.
   - A lock per KB name in `kb_store`, and a locked, atomic
     `sessions.jsonl`.
   - A default timeout on the fan-out `QueueChannel`.
1. **A concurrent meta.**
   - Ephemeral workers for all three modes.
   - A per-worker thread channel and a multi-slot question queue (the web
     UI and shell show one question at a time from a list).
   - Process-global state moved onto the session (§10).
   - The LLM limiter and per-worker usage attribution.
   - A `run_swarm` with items only: no board, no subscriptions.

   This alone gives parallel cross-mode work, such as an analysis next to a
   simulation.
2. **The board.**
   - The record schema, the single writer and recorded reads.
   - Verified-only propagation and board-blind checks.
   - `independent_support` in fusion.
   - Steering, `fusion_feedback` and `informed_by` rebased onto board reads.
3. **Reactions.** Subscriptions, `task_request`, causal chains and cycle
   refusal, supersede-chain stops, retraction and taint.
4. **Scheduling.**
   - Swarm budgets with reservation and the circuit breaker.
   - Process workers for heavy items.
   - Measured peak memory per item class, and admission sized from it.
   - On AWS: worker tasks for heavy item classes, OOM and Spot
     reconciliation against ECS, and per-campaign provider quotas.
5. **Several instruments.**
   - The instrument worker and its log-to-board bridge.
   - Several live runs per session (`_RUNS` keyed by run, not session).
   - Priority classes in the limiter and admission.
   - Sample subjects with declared registrations.
   - The pause wait-graph check.
   - The board behind MCP, for one SciLink per instrument.

   It needs stages 2 and 3 (the board and subscriptions) and the priority
   classes it adds to stage 4.
6. **UI.** The swarm plan at the autopilot gate, the question queue, a
   board view per subject in Mission Control, and several instruments side
   by side in the Live tab.

Each stage lands with offline tests, using `tests/test_fanout*` and
`tests/test_web_auth.py` as the model:
- Stage 0: concurrent checkpoint and download races.
- Stage 1: two concurrent workers of every mode pair, and two questions
  outstanding at once.
- Stage 2: independence counting on constructed read graphs, and a check
  refused a read.
- Stage 3: a two-item cycle is refused, and a retraction taints exactly its
  dependents.
- Stage 4: admission under a budget.
- Stage 5: two replay instruments posting to one board; a novelty on one
  produces a recommendation on the other; a frame is never slowed by a
  stalled coordinator; a coupled-pause cycle is handed to the person.

A live run closes stages 1 and 3: an analysis whose claim triggers a
simulation, on Bedrock, with the board inspected afterwards. Stage 5 closes
with two MCP demo instruments (`python -m scilink.live.mcp_demo_server`)
driven through the Live tab.

## What stays as it is

- The three modes.
- The persistent children for conversation.
- Series mode for many datasets of one kind.
- Best-of-N and audits as independent checks.
- The fan-out complementarity gate before any fusion.
- The settled-plan rules: a swarm follow-up that would rewrite an approved
  plan is a `refine_*` call with a `trigger` like any other, and
  `new_results` from a worker is a real trigger.
- Process-per-campaign hosting: a swarm lives inside one campaign.
- The live loop's frame path (zero model calls), its "recommend, never
  actuate" boundary, and instrument memory kept per instrument.

## Alternatives considered

- **Agent-to-agent messaging.** It needs N² channels, carries no
  provenance, and makes cycles the default. The board plus coordinator rules
  give the same reach with one record and a termination argument.
- **A shared mutable dict.** Last-writer-wins is the store bug in E at the
  scale of the whole system, and it cannot say who knew what.
- **An LLM coordinator deciding every reaction.** It puts judge variance in
  the control loop and makes termination depend on a model. The LLM decides
  between swarm runs; rules decide within one.
- **Process-per-agent behind MCP with an external coordinator.** Valid today
  for throughput, since each instance has its own `SCILINK_HOME` and one
  server is one campaign (`docs/connecting_agent_clients.md`). Without a
  shared record it is parallel delegation, not a swarm. It becomes a swarm if
  the board is later exposed as an MCP server that every instance is a
  client of. That is a transport choice for stage 2, not a different design.
- **Recursive spawning (workers delegating to workers).** Deferred: depth 1
  keeps the budget and cycle arguments simple. Raise it only if a real task
  needs it.

## Open questions

- **Should provisional findings ever propagate by default?** Discovery work
  wants speed and correctness wants verification. The default here is
  correctness, with per-item opt-in.
- **Board scope.** Per meta session or per subject (the hosted proposal's
  workstream)? For instruments the extension answers part of it: the board
  is per experiment and sample, and instrument memory stays per instrument.
  Does an experiment that spans several meta sessions need a board that
  outlives them?
- **Who declares a registration between instruments?** A person at setup,
  the instruments' own metadata, or a fiducial measured at the start? A
  wrong registration makes every region-level link wrong, so it may need
  its own check.
- **Cross-agent claims.** When two verified findings from different modes
  conflict (an analysis claim against a simulation result), who decides?
  This proposal reports the conflict and adds no judge. Is that enough in
  autonomous runs?
- **IPC for process workers.** A queue to the coordinator is simplest. Is a
  board behind MCP needed before hosting needs it?
- **When does an item leave the campaign task?** A fixed list of heavy
  classes, or a threshold on its measured peak memory against the task's
  headroom? A worker task costs minutes of cold start (the image pull), so
  a short item may be cheaper queued in place.
- **Default budgets.** Measure first: run a representative cross-mode swarm
  and read `usage.jsonl` before choosing defaults, as the fast-and-live work
  did with `stage_timings`.

## What was checked against the code

- **Read directly on 2026-09-28:**
  - The module docstring, admission constants, `_make_ephemeral_analysis_child`
    and the `fanout_branch_hitl` sites in `fanout.py`.
  - `sandbox_env` and `_GLOBAL_SANDBOX_APPROVED` in `executors.py`.
  - The analysis checkpoint write and the `run_task` `n_before` snapshot in
    `analysis_orchestrator.py`.
  - The single-slot `ParkingChannel.ask`.
  - `num_retries` in `litellm_completion`.
  - `_RUNS` keyed by session id in `server/live_api.py`, and the "never
    actuates" statements in `live/instruments.py` and
    `live/measurement_loop.py`.
- **Taken from three read-only code audits the same day** (meta concurrency;
  process-wide state and stores; LLM layer, human feedback and budgets):
  every other line reference above. Treat those as a starting point to
  re-read before implementing, not as citations.
