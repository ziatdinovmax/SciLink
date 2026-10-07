# Proposal: swarms — concurrent agents that share what they find

Status: stage 0 merged to `main` on 2026-09-29 (head 328bd2bb); stage 1
merged on 2026-09-30 (#697, head bed2f7f8) after three review rounds, built
as "Starting stage 1" below describes; what changed on the way is recorded
under the stage in "Build order". Stage 2 (the board) merged on 2026-10-01 (#702, head
e1c0fc72) after eight review rounds, built as "Starting stage 2" describes,
with what changed on the way under the stage in "Build order" and its open
items in "After stage 2". The recipe-reuse dependency (#704, #705) merged
on 2026-10-01 (#707, head c2c95f71) after two review rounds. Stage 3
(reactions) merged on 2026-10-01 (#708, head a7663b2b) after three review
rounds, built as "Starting stage 3" describes, with what changed on the way
under the stage in "Build order" and its open items in "After stage 3".
Between stage 3 and stage 4 the replay policies the board's "verified"
rests on were made shared and settled (#712: PR A #713, PR B #714, #717,
#715 — all on `main` by 2026-10-03; see "Between stage 3 and stage 4"
below). Stage 4 (scheduling) is next, local first. The design notes were
drafted 2026-09-28 against `main` at b988c7cd (Release 0.0.83), based on a source audit of the meta
agent (`meta_orchestrator.py`, `meta_orchestrator_tools.py`, `fanout.py`,
`telemetry.py`), the three mode orchestrators' `run_task`, the executors, the
human-feedback layer, the LLM wrappers and the persistent stores under
`~/.scilink`. Line numbers in the sections up to "Build order" are as of
b988c7cd; "Starting stage 1" cites 328bd2bb.

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

- **A capacity plan decides whether and how to run the swarm, before it
  starts.** The coordinator adds up the items' expected peak memory
  (measured per item class; see "Running on cloud resources") and compares
  it with what the host can give: available memory minus a margin on a
  laptop, the task size on a cloud task.
  - If everything fits, the items run concurrently.
  - If the heavy items do not fit together, they run one at a time and the
    light ones run beside them.
  - If one item does not fit the host at all, it is not started. On cloud
    resources it goes to a worker task sized for it. On a laptop the result
    says why it did not run.

  In autopilot the plan is part of the swarm-plan gate ("2 of 4 items are
  heavy, about 7 GB each; this machine has 5 GB free; they will run one at
  a time"). Autonomous runs apply it without asking and never overcommit.
  Today's `_admit_branch` always admits a branch when nothing else is
  running, so on its own it would still have started the 8 GB analysis
  that froze an 8 GB laptop. The plan is what can refuse that one.
- **Admission** generalises `_admit_branch` for what the plan admitted:
  slots per mode, the item's memory from the plan, and a hold on anything
  over the current headroom. It admits a first item only when that item
  fits the host, so there is still no deadlock and no overcommit.
- **A memory guard watches while the swarm runs.** Estimates can be wrong,
  and other programs on a laptop take memory too. The coordinator samples
  free memory (psutil, or `memory_pressure` on macOS) every few seconds.
  - Below a soft threshold it stops admitting.
  - Below a hard one it cancels the most recently started heavy worker
    (stop event, then a process-group kill) and records the item as
    `memory_pressure`. The item is re-queued to run alone, once.

  Ending one worker on purpose is better than what happened without a
  guard: the OS killed system services at random and the machine froze.
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

- **Measure every item.** Record the peak resident memory of each item and
  keep it against `(mode, skill, data shape)` in a small table on the
  campaign volume.
  - **A process worker** reports its whole peak through `wait4` /
    `ru_maxrss`: the item's own arrays as well as its scripts.
  - **A thread item** can report only its tracked subprocesses. The
    process's own `RUSAGE_CHILDREN` is one maximum over every child of every
    item, so it cannot be split per item.
  - **A worker task** reports the task's memory metric.

  So the table is filled by process workers, which is one more reason
  heavy items run as processes.
- **Size from what was measured.** Admission and task sizing read that
  table. A class with no measurement yet falls back to the input-based
  estimate (`fanout._branch_mem_estimate`, which since #724 / #750 follows
  the largest unit, sees nested data and estimates a raw-instrument folder
  by its preparation). The first run of a class is its measurement.
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

0. **Fixes for today's parallelism. Done: merged 2026-09-29.**
   - Atomic child checkpoints and chat histories; a lock on
     `_open_delegation` (#679). Every ledger write under that lock, and
     `_ledger_snapshot()` for readers (#693).
   - `path_lock` (`utils/file_lock.py`) and `download_once`
     (`utils/download.py`) for SAM, DCNN and COD (#678).
   - SciLink-side LLM retries: transient errors only, jittered backoff,
     `Retry-After` honoured, and a timeout on every path (#681). LiteLLM's
     default retry waited nothing and retried 400s. The chat loops' outer
     timeout loops are gone (#691).
   - Process-group kill in `ScriptExecutor`, plus cleanup on SIGHUP/SIGTERM
     (#684); Windows Job objects (#692, not yet run on a real Windows
     machine).
   - A lock per KB name in `kb_store`, a locked, atomic `sessions.jsonl`,
     and snapshot-first KB attach (#683). A session's KB copy refreshes when
     the store is published again (#689).
   - Found while live-testing, fixed in their own PRs: the planning prompt
     kept its workspace block across autonomy changes (#686); the meta names
     the attached KB (#687); headless `-p` passes `--data`/`--metadata` (#688);
     the Live tab's Stop ends the running frame (#690). Filed, not fixed:
     #685, generated code that runs outside the executor.
   - **Not done, carried into stage 1:** the default timeout on fan-out's
     `QueueChannel`, skipped because `fanout_branch_hitl` is never set today
     and the multi-slot queue replaces it anyway; and the store locks of
     robustness item 9 for distill staging, graduation and instrument home.
1. **A concurrent meta.**
   - Ephemeral workers for all three modes.
   - A per-worker thread channel and a multi-slot question queue (the web
     UI and shell show one question at a time from a list).
   - Process-global state moved onto the session (§10).
   - The LLM limiter and per-worker usage attribution.
   - A `run_swarm` with items only: no board, no subscriptions.

   This alone gives parallel cross-mode work, such as an analysis next to a
   simulation.

   **Merged in #697 (2026-09-30).** What changed from the plan:
   - *Process-global state* needed no change for a swarm. A swarm's workers
     all run in one user's session, under that user's consent and
     credentials, so sharing them is correct. The cross-session leak is a
     multi-user-server concern, left to process-per-campaign hosting. Each
     worker installs its own thread channel, so no question reaches
     `builtins.input`, and `ExecutionTimeout` already uses a per-thread
     watchdog off the main thread.
   - *The question queue* got its timeout, withdrawal and `pending()`
     listing in the backend. The web `ParkingChannel` keeps one slot: the
     coordinator serves questions one at a time through it, so a multi-slot
     panel moves to stage 6.
   - *The memory guard* cancels the running item expected to hold the most
     memory (the newest of equals), not simply the newest. It acts one
     cancellation at a time, reading memory again only after the cancelled
     worker has ended.
   - *The LLM cap* defaults to 16 calls in flight per model, above what
     today's runs reach. It covers completions, embeddings and the internal
     proxy client.
   - The tool parameter is `work_items`, not `items`: a property named like
     the JSON-schema keyword is ambiguous to schema tooling.
   - *From the reviews:* a gate nobody answers is never a human decision
     (`hitl.last_question_timed_out`; channels with a timeout of their own,
     the MCP server's, mark it with `hitl.mark_timed_out`, and the planner
     writes `unattended_gate` instead of `human_review`); questions are shown
     from a thread of their own (`hitl.QuestionServer`) so the coordinator
     keeps enforcing budgets and the memory guard; a worker's timeout clock
     restarts when its question is shown, and time waiting on a person is
     left out of its budget; cancels reach a parked question, an LLM-slot
     wait, the backoff sleep and best-of-N candidates
     (`log_context.register_cancel`, `inherited_context`); the memory guard
     never cancels an item running alone and waits for a cancelled worker to
     end (with a deadline) before the rerun; a reviewed skill upgrade refuses
     a skill that changed during the review (`base_hash`).
   - *Not done:* swarm workers in the Telemetry tab; the web UI and shell
     driven live with worker questions; and HPC submission from meta-driven
     simulations (#696), which a swarm of simulation items needs.
2. **The board.**
   - The record schema, the single writer and recorded reads.
   - Verified-only propagation and board-blind checks.
   - `independent_support` in fusion.
   - Steering, `fusion_feedback` and `informed_by` rebased onto board reads.

   **Built (PR "Swarm stage 2").** `scilink/agents/meta_agent/board.py`.
   What changed from the design above:
   - *A correction is the replacement itself.* The schema listed `supersedes`
     as a kind; on the board it is a field: the corrected claim is a `claim`
     that `supersedes` the old id. A bare "supersedes" kind with the content
     in its payload would make every reader unwrap it. A `retraction` is a
     record of its own (`target`). The fold marks the old record
     `superseded` / `retracted` and hides it from reads; nothing revives a
     record (retracting a correction does not bring the corrected one back).
   - *Every delegation posts, not only swarm items.* Posting lives in
     `_close_delegation`, so a direct delegation, a fan-out branch, a swarm
     item and a fusion all leave records, and the board is the session's
     record rather than a swarm's. Posting never fails the delegation.
   - *Workers post through the writer's lock, not a queue.* In one process
     a lock-serialised `Board.post` that flushes (and fsyncs) before
     returning is the "one writer, acknowledged after the flush" of the
     design; the queue-and-drain shape is for process workers (stage 4).
   - *What each mode verifies:* an analysis claim is verified when its
     analysis record's status is success (its `[analysis_id]` prefix is
     evidence, not text), and the approved `scripts/analysis_script.py` is
     posted as a `recipe`; a plan's findings are verified only under a
     `human_review` stamp — an unattended gate leaves them provisional — with
     BO-engine points as verified `parameter_point`s (computed, not
     authored) and a standing blocking finding as a `hazard` (a hint by
     type, so it may propagate on the critic's word); a structure is
     verified when the validator's status is `success`. The planning result
     now carries `plan_review` and each simulation structure its
     `validation_status`; nothing else in the modes changed.
   - *A steering payload is a finding of the companion.* It is filed under
     the companion's ledger index (author "fan-out steering (reduction of
     …)", mode `steering`), provisional — a deterministic reduction, but no
     gate passed it — and the steered branch's entry reads it, which is what
     makes fusion count 1 of 2. Co-registered operands stay a ledger stamp:
     a shared dataset is not a finding, and the count skips that stamp.
   - *Fusion's claims are provisional* (a synthesis passes no gate of its
     own) and read every fused finding, so a re-analysis citing the fusion
     inherits them as reads and the next fusion counts it dependent.
   - *A read is once, at the item's start, after admission* — so an item
     admitted later in a swarm sees what earlier items of the same swarm
     already posted. The read block is rendered like fan-out steering, with
     the additive-only rule; the ledger keeps the task as sent.
   - *Found by the tests:* a post after a torn last line was appended onto
     the torn text. The writer now starts on a fresh line when the file does
     not end with one.
   - *From the review (PR #702):* an analysis claim is verified on the
     agent's own verdict, not on `status` — the curve and image agents
     return `success` for a salvaged best-available fit (`quality_warning`),
     an unverified run (`quality_history.unverified`) and a result the
     verifier never approved (`approved` false); `analysis_verdict`
     (`_verification_record.py`) reads those, per item for a series, and the
     `analyses` rows of `run_task` carry `verified` and `reason`. Fusion's
     `independent_support` is keyed by delegation index (labels repeat) and
     joins the board's read graph with the ledger's `context_from` (declared
     or inferred) and `informed_by` edges, transitively
     (`fanout.independent_support_of`); the prompt says what it cannot see.
     Only the delegation that wrote or settled a plan posts its hypotheses
     (`plan_review.written_here`); configuration and TEA findings, BO points
     and steering reductions are provisional (no gate); a blocking finding a
     human approved the plan over is settled and not posted. A reader is
     shown the newest 24 records, clipped and under a budget, between data
     markers, and is stamped with exactly those ids. Records are written
     JSON-clean (a `Path` or numpy scalar in `evidence` no longer breaks
     every later read). A supersede or retraction takes effect only from the
     original's author, a verified record, or the coordinator. `get_board`
     falls back to every subject only for a subject the board has never
     seen; subjects are NFKC-normalised. A re-analysis citing a fusion
     inherits the fusion's reads as well as its claims. The hyperspectral
     recipe is `dynamic_analysis_records.json`.
   - *Round 2 of the review:* `analysis_verdict` reads the shapes the
     agents write, not one field: a series' anchors, regime anchors and
     refits (the units with a QC-engine record) must be approved with no
     salvage marker (`quality_warning` / `judge_warning`, now carried on the
     curve and image `individual_results`), its followers must have
     succeeded and not be `unverified`, and a FAILED unit does not block
     (the agent excludes it from the table and flags it; an unverified
     success is in the table, so it does); a hyperspectral cube needs
     `success` and every scripted target `task_success` and not `salvaged`;
     a hyperspectral series reads each row's `verified`; a good-verdict
     locked reuse is verified by the replay gate. Fusion's count is the
     largest set of supporters with no coupling between any two (three
     meshed or mutually informed branches count once, never zero; two
     branches plus a re-analysis of both count two); a co-registered-operand
     stamp is not an edge; a label binds only to an earlier entry of the same
     group; `_analysis_ids_of` accepts only an analysis folder name. A plan
     is "written here" when its hypotheses, iteration or review stamp
     changed, not on any edit; a blocking finding is provisional (the critic
     is advisory) and posted once. The series recipe is the anchor's unit
     script (`recipe_unit` on the row); a hyperspectral series posts none.
     Fence markers inside a record are neutralised.
   - *Round 3 of the review:* steering is recorded on its own
     (`steered_by`), so a meshed-and-steered branch keeps its steering edge
     while the mesh stamp is skipped; inside one fan-out any sibling may be
     the steering source (the slots are created together), outside it only
     an earlier entry. The count is one routine (`board.independent_set_size`):
     the largest set of branches none of which is coupled to another, exact
     to 12 and a stated lower bound beyond, and the prompt says exactly
     that. A follower fitted with no base script (its regime's anchor failed)
     is fresh code with no verifier: the curve and image follower results
     now carry `fitted_from` (`locked_script` / `fresh_code`) and
     `replay_verbatim`; `individual_results` carries `fitted_from` (a
     model-repaired follower is still a follower by policy, so the verdict
     does not read `replay_verbatim`), and a `fresh_code` follower blocks the
     series. A series whose anchor is a
     good-verdict locked reuse is verified by the replay gate; a hyperspectral
     series row is held to the single-cube rule (status `success`, something
     extracted); a hyperspectral target that failed before any code ran
     blocks; a refit the driver accepted by its consistency rule is held like
     a follower (so a refit cannot unverify a series the unrefit unit would
     have passed); a cut anchor reads as cut, not salvaged. A plan's identity
     includes its steps and blocking issues, so an autonomous protocol
     revision is "written here". A series with no anchor unit posts no
     recipe, and a regime series' recipe record says it is the first regime's.
   - *Round 4 of the review:* a salvaged anchor was laundered by refitting
     it (the relaxed refit rule had skipped the anchor's bar for ANY refit).
     The controllers now stamp `role: "anchor"` on first-in-regime units and
     carry it through every refit replacement; `individual_results` carries
     it; a refit with the anchor role keeps the anchor's bar, a follower
     refit stays on the follower rule, and an anchor refit without a role
     (a checkpoint from before the stamp) is not counted as an anchor. A
     hyperspectral series row needs every target approved
     (`quality_metrics.n_approved == n_targets > 0`), as a single cube does.
     Steering is stamped by sibling index too (`steered_by_index`; labels
     repeat within a group), and a pre-`steered_by` stamp that says
     `+steering` counts every label as an edge (the count errs low). The
     fusion prompt no longer calls meshed agreement "one joint measurement":
     a meshed branch is a separate observation, and only a number computed
     from both datasets at once is one computation; "pairs not listed here
     are independent" now points at the INDEPENDENT SUPPORT block for
     couplings from reads and citations. Known and left: a failed reuse
     whose anchor was re-derived and approved is blocked by the schema-drift
     `quality_warning` with the reason "salvaged" (pre-existing wording).
   - *Round 5 of the review:* a follower is judged by the recipe it
     REPLAYED. An anchor refit to an approved model says nothing about
     followers still on the original script, so every refit replacement
     site carries a summary of the unit it replaced (`replaced_unit`:
     approval and salvage markers, chained to the earliest when a unit is
     replaced twice), `individual_results` carries it, and a `locked_script`
     follower whose regime anchor was refit is judged by that summary
     ("follower replays a recipe that was not approved … since refit"). The
     opposite direction — an approved anchor refit to a salvaged unit — is
     accepted as conservative: the refit anchor is a salvaged row in the
     table, whatever the followers replayed. `series_anchor_unit` reads the
     role. The steering payload carries the source's slot (`source_slot`),
     so the index stamp and the board post never match labels (two series
     in one upload directory share a stem). The companion-contact block of
     the fusion prompt no longer opens with "NOT fully independent" (a
     mesh-only run has no independence spent), the steering caveat names
     the steering sources only, and the reference to the INDEPENDENT
     SUPPORT block is guarded for a caller without a board. A record-less
     hyperspectral row reads "the unit has no dynamic-analysis record"; a
     `not_measurable` target with `task_success: False` would count as
     unapproved in a row and be skipped in a cube — no path produces it,
     and the row carries no per-target records to align on.
   - *Round 6 of the review, and the end of a cycle:* five rounds each
     closed one hole in `analysis_verdict` and opened another, because the
     verdict RECONSTRUCTED after the fact which recipe each unit replayed
     and whether its gate passed, from markers scattered across units. The
     root cause was where the decision lived. Now the curve and image
     series drivers stamp `unit_verdict` on each unit at fit time
     (`_verification_record.unit_verdict_for`): an anchor or refit by its
     own gate, a follower by the recipe it replayed and that recipe's
     verdict as it was then, fresh code as unverified — at the anchor fit,
     each follower fit (serial and the parallel drain) and every refit
     replacement, never failing a fit (`stamp_unit_verdict`). The
     aggregator only aggregates; the marker reconstruction remains for
     checkpoints from before the stamp (with `regime` now carried). A refit
     anchor keeps the script its followers replayed as
     `scripts/<unit>_locked.py`, which the board's recipe and the series
     reuse path prefer. `tests/test_series_verdict_path.py` drives the real
     series and refit controllers with only the QC loop and the executor
     stubbed; fixtures written from a reading of the code are what hid the
     series cases for five rounds.
   - *Round 7 of the review:* the structural change had added behaviour to
     the analysis agents; that is taken back so the PR's effect on them is
     the stamps alone. No `<unit>_locked.py` is written and the prior-run
     reuse pick is exactly what it was. The series driver records each
     regime's recipe ONCE, when the anchor's script is locked
     (`state["locked_recipes"]`, on the result as `locked_recipes`: unit,
     verdict then, script text), after every caveat is on the anchor (a
     reuse that failed and was re-derived is salvaged); the `run_task` rows
     carry it as `recipes` (left off the delegation summary the model
     sees, see round 8), and the board writes its own copy under
     `swarm/recipes/<NN>_<label>/<analysis_id>/` and points its record there (a single run's approved script is copied
     the same way, `source` recorded). Nothing under an agent's folder is
     added or read again for a recipe, and no later refit or reuse can
     change what a record points at. A reused unit with no QC record is
     verified iff the replay gate's verdict is `good`; a run with some
     stamps but not all is not verified ("unit X has no stamp"), and the
     legacy reconstruction is used only for runs with no stamp at all.
     `tests/test_series_verdict_path.py` adds a good-reuse series, a failed
     reuse re-derived (salvaged), a failed follower refit that leaves the
     reuse pick unchanged, and a PARITY test: the stamped verdict and the
     legacy reconstruction over every real-path shape, allowed to differ
     only on an explicit list. They differ in one case, and it is intended:
     a FOLLOWER refit that stayed salvaged — the legacy rule held a follower
     refit only to "finished, not unverified" (a round-3 relaxation), the
     stamp judges every refit by its own gate, so a salvaged refit is a
     salvaged row in the table, as a salvaged anchor refit is.
     `tests/test_image_series_verdict_path.py` is the image twin of the
     harness (the real `UnifiedImageProcessingController` and
     `ImageAdaptiveRefitController`, the real `_process_single_image` for
     every follower): clean, salvaged anchor, failed anchor with fresh-code
     followers, two regimes, a failed follower refit (approved and salvaged),
     and the same parity.
     Open for stage 3, filed as issues: which script a reuse of a refit
     series should replay (the original the followers ran, or the approved
     refit), and a way for an analysis worker to take a script file as its
     recipe.
   - *Round 8 of the review (board and meta side only; the agents are
     done):* the board's recipe copy was keyed by `<NN>_<label>/<name>`
     alone, so two analyses of one delegation anchored on the same unit
     name (or two single runs, both `fitting_script.py`), or an entry
     posted twice, overwrote one copy and a verified record could point at
     an unverified script. The path now carries the analysis id and a copy
     is written ONCE (`_write_once`: identical content reuses the file,
     different content takes the next free name) — a record's file never
     changes under it. The model's view: `_summarize_delegation_result`
     copied the `analyses` rows verbatim, so a series row's `recipes` put
     the full script text (3 regimes: ~21 K characters) in the tool
     result; the model now gets the rows without `recipes` (on a copy —
     `post_delegation` reads the same rows after), and the fan-out event
     log was never at risk (`append_event` keeps a 300-character gist and
     the files, pinned by a test). A 300-character label raised "File name
     too long" and `post_delegation`'s `break` then dropped every remaining
     recipe: directory names are clipped (`RECIPE_DIRNAME_MAX`) and a
     record that cannot be written is skipped alone. Noted, not changed:
     a single run posts a recipe only for the agents' own script names
     (`fitting_script.py`, `analysis_script.py`,
     `dynamic_analysis_records.json`), never "the first `.py`" — a
     preparation run (`prepare_script.py`) posts none; `locked_recipes`
     is one extra copy of each regime's script in `analysis_results.json`
     and the checkpoint.
   - *Found live:* the curve agent's approved script is
     `scripts/fitting_script.py` (a series: one per spectrum), the image
     agent's `analysis_script.py` — the recipe takes the folder's
     representative script (`_recipe_script`). `reads_board: {}`, the tool
     schema's plain opt-in, is falsy in Python and was read as "nothing".
     A planning `run_task`'s `key_findings` are the campaign configuration
     (targets, TEA), so a short plan posted nothing: `plan_review` now
     carries one line per proposed experiment or portfolio direction
     (`hypotheses`), and those are the plan's claims on the board.
3. **Reactions.** Subscriptions, `task_request`, causal chains and cycle
   refusal, supersede-chain stops, retraction and taint.
   - *Built (PR "Swarm stage 3"):* `meta_agent/reactions.py` holds the pure
     functions — `normalize_subscriptions`, `matches` (equality on kind,
     normalised subject and status), `fill` (one regex pass over a fixed
     field vocabulary: `{finding.text}`, `{finding.path}`, `{finding.name}`,
     `{finding.value}`, `{finding.unit}`, `{finding.id}`, `{finding.kind}`,
     `{subject}`, `{analysis_id}`, `{from.label}`, `{from.index}`,
     `{from.mode}`; anything else stays literal, a value is never
     re-expanded), `hop`, `supersede_depth` and `decide`, which returns the
     item to launch with `caused_by`, `chain` and `subscription` already on
     it, or the reason it is refused. `run_swarm(work_items,
     item_time_budget_s, subscriptions, budget)` applies them in the
     coordinator loop each time an item closes and has posted: the fired
     item goes through `launch` like any other (its own ledger entry, the
     capacity ceiling, memory admission, the item budget, the question
     queue), and `_COORDINATOR_FIELDS` are stripped from a caller's initial
     items so a chain nobody enqueued cannot defeat the cycle check.
     Refusals are structural first (a cycle: the same `(mode, subject,
     kind)` hop twice in one chain; a record at supersede depth >= 2 on its
     subject), then the counters (`max_fires`, the per-subject cap, the
     swarm's `max_reactions` and `max_items`), and each is recorded on the
     triggering entry (`refused_reactions`) and in the result with the
     chain. `budget` is `{max_items <= 8, max_reactions,
     max_triggers_per_subject (2)}`. A worker's `suggested_followups` are
     posted by `records_for` as `task_request` records (provisional, not in
     `READ_KINDS`, at most four, 400 characters) for every mode; they become
     items only through a subscription on that kind, and the result's
     `task_requests` lists each with the item it became or `null`.
     `Board.fold` derives `tainted`: a record whose transitive reads reach a
     retracted or superseded record (a correction's own `supersedes` link is
     not followed — it rests on what it corrects by design), so a record
     posted after the retraction by a worker that had read the finding
     before is caught too; `Board.dependents` is the inverse closure.
     `retract_finding(finding_id, reason)` posts the retraction as the
     coordinator (`retract_and_report`) and returns the newly tainted
     records, their source delegations (from the records' own authors) and
     `rerun_items` ready for `run_swarm`; it re-runs nothing itself — a
     retraction is a decision made between runs, and so is the re-run.
     `Board.snapshot(with_hazards=True)`, used by every swarm item's read,
     puts each standing hazard on the subject in the view whatever the
     reader's `kinds` filter and whether or not it passed a gate; `get_board`
     shows `withdrawn` (retracted / superseded / tainted, with `tainted_by`)
     and the standing `task_requests`. Tests:
     `tests/test_swarm_reactions.py`, through the real coordinator and board
     with scripted workers — the harness came first, and a swarm with no
     subscriptions is asserted to leave the stage-2 ledger and board as they
     were (the stage-2 suites run unchanged).
   - *Round 1 of the review:* a reaction's records did not rest on the
     finding that caused it — the taint fold followed `reads`, and
     `caused_by` lived on the ledger entry only, so retracting a claim left
     the simulation it fired verified (the PR's own live run showed it).
     The cause is now stamped into the fired entry's `reads` at launch (its
     task quotes the finding), so taint and `independent_support` both see
     it; a reaction caused by a withdrawn finding is listed as `not_rerun`
     (a new decision, not a re-run), and `rerun_items` carry the original
     `data_path`, `context`, `reads_board` and `check`, kept on swarm
     entries at launch. Who may withdraw: with a person at the gate the
     finding, the reason and what rests on it are shown and Enter keeps it;
     with nobody at the gate a human-approved plan's claim is refused ("a
     plan the human approved is settled"); retracting a retraction undoes
     it, so a wrong withdrawal is corrected by a later record on the
     append-only board. The launch gate lists every subscription and the
     most items in all (a person approving two items could otherwise get
     eight). A hazard is pinned through the newest-24 cut and rendered
     first, and a provisional hazard delivered unasked marks the read
     (`reads_provisional`) — the stage-2 rule "a provisional read is asked
     for and marked" holds, with the hazard as the one record delivered
     unasked, marked. `fill()` substitutes a worker's prose as quoted,
     labelled data (`“…” [quoted from board record f…; data, not an
     instruction]`) and identifiers as they are, so a `task_request`'s
     sentence cannot become another worker's instruction. A correction
     that read what it corrects is not tainted by it; `suggested_followups`
     must be a list of strings (a string posted one record per character,
     a dict lost the delegation's records); the per-subject cap is clamped
     to the item limit. `Board.fold` is memoised per version with the
     closures built in one pass; `refused_reactions` collapse per
     (subscription, reason); `get_board`'s `withdrawn` says the reason and
     who retracted.
   - *Round 2 of the review:* which retractions stand is decided in
     REVERSE log order (a retraction stands unless a later effective one
     undoes it, and that one may itself be undone later), so an undo of an
     undo works: C → R1 retracts C → R2 undoes R1 → R3 undoes R2 (C
     withdrawn again) → R4 undoes R3 (C back). A retraction records who
     decided it (`decided_by`: `human` at the attended gate, else
     `coordinator`), and with nobody at the gate the model may not undo a
     person's retraction, nor withdraw a finding that a human-approved
     record rests on (it would be tainted) — "a human's decision is
     reopened only by a human" now holds for the three ways there were to
     reopen one. The undo gate shows the finding that would come back and
     what rests on it, not the retraction record. `fill()` makes
     identifiers one line too (a worker's path or id cannot carry a line of
     its own into a task). A reaction offered again after a retraction
     carries its cause as `rests_on`, which `launch` records as reads (a
     caller-declared read can only add a coupling). `max_reactions` is
     clamped to `max_items`, and the gate says "on any task_request".
   - *Round 3 of the review:* an undo of an undo WITHDRAWS the finding
     again, and the gate, the refusals and the report had read the record
     named (a retraction → "an undo") instead of what the act does — a
     fourth way round "a human's decision is reopened only by a human", and
     a report saying the opposite of what happened. Every retraction is now
     decided from its EFFECT: `Board.preview(extra)` folds the log as it
     would read with the retraction appended (the pure fold is
     `_fold_records`; `fold()` memoises it), and `_act_effects` gives the
     findings withdrawn, newly tainted and brought back, and the retractions
     whose standing flips; the attended gate shows exactly those
     ("WITHDRAWS …", "TAINTS …", "BRINGS BACK …", Enter keeps things as
     they are), the autonomous refusals test them (a human-approved finding
     among the withdrawn or tainted; a flipped retraction that is a
     person's — a retraction with no `decided_by` stamp, from a board made
     before the stamp, counts as a person's), the report lists them
     (`withdrawn`, `restored`, `tainted`, `effect`) and `rerun_items` /
     `not_rerun` follow the affected set; an act that would change nothing
     (undoing a retraction that never took effect) is refused. `rests_on`
     is a list of ids that are on the board, at most 24, and ignored on a
     `check` (a check reads nothing); identifiers in `fill()` are bounded
     and `finding.value` is quoted too.
   - *Deviations from "Starting stage 3", each deliberate:* no worker
     supersedes its own earlier claim — no `run_task` result says "this
     replaces that", and guessing it from kind and subject would be the
     reconstruction the stage-2 reviews taught against; the fold supports it
     for a caller that knows. No automatic re-run of tainted sources inside
     a swarm — nothing inside a swarm retracts, so the re-run is the next
     swarm, with the items prepared. The result's list is
     `refused_reactions` (reason, chain, finding) rather than
     `refused_cycles`, since the cap and the chain stop are refusals of the
     same shape; no would-be entry is opened for a refused reaction, the
     refusal sits on the triggering entry.
4. **Scheduling.** (Updated 2026-10-06: see "Since the stage-4 scoping"
   for what has already landed and the worker contract.)
   - Swarm budgets with reservation (tokens; the per-item wall-clock budget
     is on `main` since #755) and the circuit breaker.
   - Process workers for heavy items, behind the worker contract, launched
     through `run_in_child` and stoppable.
   - Measured peak memory per item class (read from the process worker),
     the pre-launch capacity plan sized from it (the #750 estimate for a
     class not yet measured), and the runtime memory guard (which cancels
     through the contract).
   - Into today's fan-out, still open: the plan's refusal of an item
     larger than the host, and the guard. The swarm has both; fan-out does
     not.
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

## Starting stage 1

Kept as the plan stage 1 was built from (#697); see the notes under stage 1
in "Build order" for what changed. Line references are to `main` at 328bd2bb. Each step is one small PR off
`main`, with offline tests and, where it changes a run, a live check. The
order puts what the later steps rely on first.

**Step 1: the stores that concurrent workers will share.** Carried over from
stage 0. Wrap the read-modify-write sites in distill staging and graduation
(`skills/_shared/_staging.py`, `_graduation.py`) and the instrument home
(`live/instrument_home.py`: plain `write_text` and the `used()` counter) in
`path_lock`, the way `kb_store` and `sessions.py` use it. Test: two threads
and two processes hitting one record lose no update.

**Step 2: ephemeral workers for all three modes.**
- Generalise `_make_ephemeral_analysis_child` (`fanout.py:836`) to
  `_make_ephemeral_child(orch, mode, base_dir, restore=False)`. Keep the old
  name as a wrapper so the fan-out resume path is untouched.
- Planning is built like `_get_planning_child` (`meta_orchestrator.py:864`):
  `DELEGATED_OBJECTIVE`, `data_dir=None`, and the meta's `knowledge_dir`.
  Each worker copies a store KB into its own `kb_cache` (`_refresh_store_kb`,
  `planning_orchestrator.py:1372`), so workers never share index files. The
  cost is one copy per worker.
- Simulation is built like `_get_simulation_child` (`:919`), with the import
  inside the function. A missing `ase` makes that item an error, not the
  swarm.
- None is registered in `orch._children`. Each worker gets its own
  directory, so the `run_task` windows (analysis `:1464`, planning `:1450`,
  simulation `simulation_orchestrator.py:530`) are correct by construction.
- Test: two workers of every mode pair on one meta at once, stubbed LLM.
  Each `run_task` reports only its own outputs, and each ledger entry is
  its own.

**Step 3: process-global state onto the session (robustness item 10).**
The sites:
- Sandbox approval: `_GLOBAL_SANDBOX_APPROVED` (`executors.py:509-543`).
  It becomes an approval the session or coordinator owns and workers
  inherit. The module bool remains the standalone fallback.
- `os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")` at
  `mcp_server.py:1343`, `hyperspectral_series.py:1306` and
  `live/_reanchor.py:43`: pass the approval instead.
- Credentials written to the environment: `server/session_manager.py:85`,
  `litellm_wrapper.py:163`, `cli/shell/bootstrap.py:73`, and the
  `_api_manager` singleton (`auth.py:235`). Pass them as client kwargs.
- `sandbox_env(extra, source)` (`executors.py:597`) already takes a
  `source`; the session passes its own snapshot of the environment.
- The `builtins.input` swaps (`server/runner.py:366`, `cli/shell/turn.py:233`,
  `ui/app.py:639`) and `hitl.set_default_channel` in headless mode
  (`cli/shell/headless.py:79`). These stay, but every worker thread sets its
  own thread channel before it runs, so no worker ever reaches them.
- `ExecutionTimeout` (`executors.py:902`) uses SIGALRM only on the main
  thread. Confirm that worker threads take its other path.
- Decide whether issue #685 (generated code outside the executor) lands
  here, since it is the same boundary.

**Step 4: a channel per worker and a multi-slot question queue.**
- `_BranchChannel` (`fanout.py:817`) becomes the worker channel. It tags
  `origin` with the worker and its subject.
- `QueueChannel` (`hitl.py:142`) holds several outstanding questions. It
  gets a default `timeout_s` (the stage 0 carry-over); on timeout it resolves
  to the gate's own default and records `timed_out`.
- `ParkingChannel.ask` (`server/hitl_channel.py:63-79`) and
  `TurnState.pending_question` (`server/runner.py:135`) change from one
  slot to a list keyed by origin. `pending_question.json` (`hitl.py:333`)
  becomes one file per worker directory.
- The web UI and the shell show one question at a time from the list,
  with who is asking. Labels go in `scilink/ui/vocabulary.py` and the
  generated TS twin, per CLAUDE.md. Structured subjects already carry
  `origin`.
- Test: two questions outstanding at once, answered out of order, each
  reaching its own worker, and one timing out to its default.

**Step 5: an LLM limiter and usage per worker.**
- A semaphore per (provider, model) inside `_completion_with_retries`
  (`litellm_wrapper.py:438`). Every completion goes through it since #681.
  The embedding calls (`:1132-1231`) do not, and need the same cap. It is
  sized by an env var, with a conservative default. Backoff and
  `Retry-After` already exist.
- A worker tag beside the session tag (`tracing.py:118`), written by
  `record` into `usage.jsonl`. Telemetry reads every worker, not just the
  two persistent children.
- Test: N threads against a fake provider never exceed the cap, and usage
  is split correctly by worker.

**Step 6: `run_swarm(items, budget)`, items only.**
- An item is `{mode, task, context, subject, budget}`. Each runs on a
  step 2 worker in `<meta_session>/swarm/<NN>_<slug>/`, with a ledger entry
  opened through `_open_delegation`.
- Reuse fan-out's coordinator machinery: its pool, the stop guard,
  `_admit_branch` (`fanout.py:111`) and the resume path (`:1349`).
- Autopilot shows the swarm plan first: items, budget and memory. It is a
  new gate, so it declares a `subject=` from the start.
- With it, take the stage 4 item marked "worth taking early": refuse an
  item whose estimate exceeds what the host can give, and the runtime
  memory guard. `_admit_branch` alone still admits anything when nothing
  else runs.

**Live checks that close stage 1**, one heavy run at a time:
1. `run_swarm` with a curve-fitting analysis and a simulation side by side.
2. Two planning items against one attached KB.
3. An autopilot swarm with two questions outstanding, answered through the
   web UI and through the terminal shell.
4. After each run, confirm that `usage.jsonl` attributes calls per worker
   and each worker's directory holds only its own outputs.

**Baseline for regressions.** The full suite on `main` at 328bd2bb gave 24
failed, 3721 passed and 29 errors, all environment and `logging.disable`
ordering noise. Compare each branch against a `main` worktree run by the set
of failing test ids, not by the counts. `test_grain_twin_proxy::
test_threshold_is_tunable` and the wall-clock cases in
`test_per_tool_checkpoint` are flaky.

## After stage 1: what is on main, and what is open

Everything a fresh session needs to start stage 2 is here and in the two
sections above ("Starting stage 1" for the shape of a stage's work, and the
notes under stage 1 in "Build order" for what the reviews changed). Line
references are to `main` at bed2f7f8.

**What is on main (stage 1, #697):**
- `scilink/agents/meta_agent/swarm.py`: `run_swarm` (:311), `normalize_items`
  (:90), `capacity_plan` (:119), `_run_item` (:209), `_guard_memory` (:273);
  the meta tool `run_swarm(work_items, item_time_budget_s)` in
  `meta_orchestrator_tools.py`, and one bullet in the meta prompt (a swarm
  item is a fresh agent).
- `scilink/agents/meta_agent/workers.py`: `build_child` (:51, the one
  constructor for persistent specialists and workers), `release_child` (:150).
- `scilink/hitl.py`: `QueueChannel` (:183, timeouts, withdrawal, `pending()`,
  `close()`), `QuestionServer` (:341), `WorkerChannel` (:418),
  `mark_timed_out` / `last_question_timed_out` / `unattended_questions`
  (:388-), `question_timeout_s`. The planner's `_stamp_human_review` skips
  the stamp after a timeout and writes `unattended_gate`.
- `scilink/utils/log_context.py`: `register_cancel` / `raise_if_cancelled` /
  `cancel_watched`, `inherited_context`; `attributed_to_current` carries the
  session and worker usage tags and the cancel.
- `scilink/wrappers/llm_limiter.py`: `llm_slot`, `llm_max_inflight`;
  `litellm_wrapper.call_with_retries` (shared by the LiteLLM path and the
  proxy client); `tracing.attributed`, `usage.UsageLedger.by_worker`.
- Locks: `_staging` (domain lock), `_graduation.skill_lock`, `_memory`
  mutators, `live/instrument_home.py` (`instruments/.locks/<id>.lock`),
  `base_hash` from proposal to apply.
- Tests: `test_run_swarm.py`, `test_worker_questions.py`, `test_llm_limiter.py`,
  `test_ephemeral_workers.py`, `test_store_locks_concurrency.py`, plus
  `test_best_of_n_anchor.py::test_candidates_inherit_the_items_cancel_and_usage_tags`.
- Knobs: `SCILINK_SWARM_MAX_WORKERS` (3), `SCILINK_QUESTION_TIMEOUT_S` (1800;
  `0`/`none` = no bound), `SCILINK_LLM_MAX_INFLIGHT` (16; `0` lifts it);
  constants `SWARM_MAX_ITEMS` (8), `SWARM_ITEM_TIME_BUDGET_S` (3600),
  `SWARM_MEMORY_FLOOR_BYTES` (0.75 GB), `SWARM_DRAIN_TIMEOUT_S` (600).

**Open after stage 1** (none blocks stage 2; each says where it belongs):
- *Telemetry tab does not show swarm workers* (`telemetry.py` reads the two
  persistent children). Stage 6, or a small PR any time.
- *Multi-slot question panel* — the web `ParkingChannel` and
  `TurnState.pending_question` keep one slot; the coordinator serves one
  question at a time through it; the panel shows who asks (`asker`). Stage 6.
- *A question already on screen when a swarm returns stays until answered*
  (its answer is then discarded with the "not used" notice); clearing the
  web's `pending_question` from the swarm would reach into the turn. Stage 6.
- *Web UI and shell were never driven live with worker questions*; the
  backend paths were (scripted person, MCP client stand-in).
- *Two timing-dependent behaviours were never exercised live*, only offline:
  two workers' questions waiting at the same moment, and the timeout clock
  restarting when a queued question is shown (both live runs had the
  workers reach their gates minutes apart).
- *Memory is estimated, not measured*: an analysis item from its data file's
  size ×6, else a per-mode floor (0.5 GB). A heavy new modality (4D
  tomography) may be refused or held on a laptop. Stage 4 measures per class.
- *Planning workers copy a plain-folder KB in full*, up to 8× per swarm; a
  store KB is the recommended shape (copied once by the planner). Accepted.
- *Queue time is inside the reported LLM latency*. Accepted.
- *No swarm resume*: after a Stop, queued items stay `running` on the ledger
  until the next turn's sweep marks them interrupted. Documented; fan-out has
  a resume, the swarm does not.
- *HPC from meta-driven simulations* — #696 (plumbing, a way to connect, and
  long jobs in a swarm). A swarm of simulation items prepares inputs only.
- *Generated code outside the executor* — #685 (predates the swarm).

## After stage 2: what is on main, and what is open

**What stage 2 adds** (line references to the stage-2 PR head):
- `scilink/agents/meta_agent/board.py`: `Board` (`post`, `retract`, `fold`,
  `snapshot`, `read_closure`, `independent_support`, `public`),
  `BoardReadRefused`, `render`, `records_for` / `post_delegation` (the
  per-mode translation of a result into records). `KINDS`, `READ_KINDS`.
- `meta_orchestrator.py`: `self.board`; `_close_delegation` posts;
  `_open_delegation_locked` turns `fusion_feedback` into `reads`; the
  checkpoint carries `board_version` and the restore reports it (the file is
  the record; a shorter file than the checkpoint's version is warned about).
- `swarm.py`: items take `reads_board` (`{}` or `{subject, kinds,
  include_provisional}`) and `check`; `_read_board` at the item's start;
  results carry `reads`, `posted`, `board_read_refused`, `board_version`.
- `fanout.py`: steering measurements under the companion's index, read by
  the steered branch; `fuse_delegations` computes `independent_support`,
  renders it (`INDEPENDENT SUPPORT … n of m`), stores it on the report and
  the fusion entry, and posts the fused claims with `reads`.
- `meta_orchestrator_tools.py`: `get_board(subject, kind,
  include_provisional, limit)`; `run_swarm`'s item schema.
- `planning_orchestrator.run_task` → `plan_review` (`human_review`,
  `unattended_gate`, `blocking_findings`, `hypotheses`, `iteration`);
  `simulation_orchestrator.run_task` → `structures[].validation_status`.
- Tests: `tests/test_board.py`; board checks in `tests/test_fanout_steering.py`.

**Open after stage 2:**
- *No retraction or supersede from the meta's tools.* `Board.retract` and
  `supersedes` exist and fold correctly, but nothing calls them yet: they
  are stage 3's (retraction and taint, supersede-chain stops).
- *Taint is not propagated.* A retracted finding's dependents keep their
  status; `read_closure` gives stage 3 the graph to taint over.
- *Reads are once per item, the newest 24 records.* Stage boundaries inside
  a mode (the design's optional mid-run read points) are not exposed.
- *Independence counts reads and citations, not common ancestry.* Two
  fusions over the same inputs count as two supporters; a finding pasted into
  a task by hand with no `context_from` is invisible to the count (the prompt
  says so).
- *Subjects are strings matched case-insensitively.* Two spellings of one
  sample are two subjects; the meta's prompt asks for one spelling per item.
  A registry of subjects is stage 5's.
- *A recipe record is a board-owned copy.* Resolved before stage 3 (#704,
  #705, one PR): `prior_analysis_paths` takes a script FILE (a `.py`, or a
  `dynamic_analysis_records.json` for a cube) as the recipe — the board's
  copy under `swarm/recipes/` replays through the same gate as a run
  folder — and a reuse of a SERIES run replays the series' locked recipes
  (from `locked_recipes` in its `analysis_results.json`: the scripts its
  table rests on), not a later refit of its anchor; the agents' reuse pick
  and the board's recipe agree, and `reuse_validity.source` says which
  script was picked and why (`_verification_record.prior_recipe_scripts`,
  shared by the curve and image pickers; the hyperspectral agent already
  took the records file). A script file is a recipe and nothing more: it is
  not a run, so the loader returns no `anchor_dir` for it, and the realtime
  profile (locked config, drift fingerprint) and the live loop's anchor
  refuse it as before; a file inside a run names that run for the loop,
  which arms on the script the run replays. A curve reuse of a series that
  locked several regimes replays the recipes in lock order and keeps the
  first the R² gate calls good (`qc_try_reuse`), so a measurement above a
  transition is not held to the model locked below it; an image reuse's
  verdict is one vision review, not a deterministic gate, so it replays the
  first regime's recipe and its label says so. An image strict replay needs
  the run (its reference features), and refuses a bare file. The regime
  recipes each run VERBATIM in a candidate folder of their own
  (`_candidates/recipe_NN`, the best-of-N layout) and the kept one is
  promoted, so the figure and `fit.npy` on disk are the kept result's
  whichever ran last; the correction ladder is paid once, on the first
  recipe, only when none executed verbatim; on the fast clock a raising
  recipe moves on to the next regime as a poor one does. The live loop
  locks ONE recipe, so `CurveModality.anchor_script` refuses a series
  anchor that locked several regimes with a message saying what to point
  it at. An image series used directly as a live anchor whose first image
  has empty `extracted_features` is refused too (a replay gate with no
  reference is not a verdict).
- *Persistent-specialist delegations read nothing.* `delegate_to_*` posts but
  has no `reads_board`; the meta threads findings into `context` by hand
  (`get_board`), which is the design's turn-granularity path.
- *The Mission Control UI has no board view* (stage 6).
- Everything open after stage 1 still stands (telemetry, the one-slot web
  panel, no swarm resume, memory estimated, HPC #696, #685).

## After stage 3: what is on main, and what is open

**What stage 3 adds** (#708, a7663b2b): `meta_agent/reactions.py`; `run_swarm`'s
`subscriptions` and `budget` and the `react` step of its loop; `Board.fold`'s
`tainted` status, `Board.dependents`, `Board.snapshot(with_hazards)`,
`records_for`'s `task_request` records, `retract_and_report`; the
`retract_finding` tool and `get_board`'s `withdrawn` / `task_requests`;
`tests/test_swarm_reactions.py`.

**Open after stage 3:**
- *Reactions are declared per swarm.* A subscription lives for one
  `run_swarm` call; a standing rule across turns ("whenever a verified recipe
  appears on this sample, replay it on new data") is stage 5's instrument
  bridge, where the subscriber is a running process, not a turn.
- *Mid-run read points inside a mode* are still not exposed (unchanged from
  stage 2); a reaction is new work that reads at its start.
- *No worker supersedes a claim; no automatic re-run.* See the deviations
  under stage 3 in "Build order".
- *Templates are text.* A fired task carries the finding's text or path by
  substitution; a reaction that needs a structured context (a parameter point
  as numbers) passes `context` as the enqueue's own object, not from the
  record.
- *Taint stays on the board.* `independent_support_of`, `fuse_delegations`,
  `get_delegation_history` and steering read the ledger, not the fold, so a
  tainted supporter still counts there. (Stage 4's scheduling did not consume
  the fold either — admission reads memory and tokens, not findings; the
  first consumer is whichever of fusion or the instrument bridge needs it.)
- *Matching uses the posted status*, not the folded one — unreachable
  within one swarm today (nothing retracts inside a run).
- *`fired[].delegation_index`* names the first launch; a memory-cancelled
  reaction's rerun is a later entry with the same cause.
- *Nits from the approval of #708, not fixed:* the effect is computed
  before the retraction is posted, outside the board's lock — a post
  landing in between would be tainted unreported; it cannot happen today
  (tool calls are sequential, `run_swarm` blocks), so it is a note for the
  day a retraction runs during a live swarm. A superseded finding that an
  undo restores is reported as "brought back" although it stays superseded
  (cosmetic). An already-tainted human-approved finding is a gap: the model
  may retract a second basis Y without refusal because P is not *newly*
  tainted, so if a person later undoes X, P stays tainted by Y. Fixed with
  the status flip (#709): `rests_on_dropped` counts the ids over the cap,
  the undo gate names the finding's own withdrawal reason, and an act that
  only flips a retraction's standing says so instead of an empty effect.
- *A retraction is not gated by "a user message since".* With nobody at
  the gate the model may withdraw the agents' own findings (never a human's
  decision, nor what one rests on, nor a person's retraction); the planner's
  finer rule (a `user_request` needs a message since the approval) would
  need the meta's turn history on the board's clock.
- Everything open after stage 2 still stands (no retraction from a worker,
  common-ancestry independence, string subjects, persistent specialists read
  nothing, no Mission Control board view).

## Between stage 3 and stage 4: the replay policies (#712)

Stage 2 made the board's "verified" rest on the agents' own gates, and its
review (eight rounds) kept finding the three analysis agents judging a replay,
saying what "verified" means and choosing among a series' regime recipes each
in their own way. #712 settled those as shared policies before stage 4
serialises the records across processes. Four PRs, all on `main`:

- **PR A (#713, 4177b01d):** `exp_agents/_replay.py` by composition — the
  `verdict_record` every agent STAMPS where it decides (`decided_by` ∈ qc_gate
  · replay_gate · recipe · excluded · none, `interpretation_checked`); three
  replay gates with one verdict shape; `select_recipe`; the parity test that
  holds the stamps to the reconstruction from result shapes, no allow-list.
- **PR B (#714, d64ffc5d, five rounds):** which regime a reuse replays is read
  from the DATA (a `live/drift.py` `DriftMonitor` seeded with each regime's
  own units; the nearest first; ambiguity said where the attribution is read);
  the identity and state checks that only WITHHOLD CERTIFICATION
  (`interpretation_checked`, never a verdict — three rounds of a deciding
  check moved its false positives threshold to threshold); the certification
  bar tighter than the flag bar; one-unit references and series followers
  certified on clean evidence; the regime's stamp as its units' curves on the
  monitor's own grid; a replay-anchored series on the replay gate; names as
  token sets (settings, subscripts, screw axes, bars for space groups, Greek
  letters). The board posts a replay's CLAIMS verified only when the
  interpretation was certified; its recipe stays verified by the gate.
- **#717 (1fb449b3, four rounds):** a replay is held to the gate its RECIPE
  was approved under (`gate_record` on the recipe at lock time, on every
  run's results, in the board copy's `<stem>.recipe.json` sidecar, on the live
  anchor) unless the caller asked for a gate on the reuse run; ONE predicate —
  "this regime's anchor replayed the recipe" — drives every reader of the gate
  (the replay gate, the outlier pass, the refit's re-scan and its
  scoring-gated skip), and off it everything is `main`'s path, pinned by a
  guard test; a replay whose certificate is withheld for a stated reason is
  explained by a JUDGE (one model call, the findings as quoted data between
  markers, an opinion on the record and a provisional claim on the board,
  `reuse_caveat` to the orchestrator — never a verdict).
- **#715 (78f0814d, three rounds):** the hyperspectral timeout escalation
  (#699): one `SandboxTimeout` type on both threads (the message test missed
  every worker-thread surface), a failed attempt's arrays released before the
  repair (frames cleared along `__cause__` / `__context__`), the first limit
  never clamped, retries bounded by the deadline and the loop budget.

**What this gives stage 4.** The records the scheduler will serialise and
reconcile are settled: a unit's verdict is stamped where the information is
and never reconstructed; "verified" is the numbers, "certified" is the
interpretation, and a provisional claim is the default for anything a gate did
not pass; a recipe carries its gate; the judge is an opinion a worker task can
emit and a coordinator can ignore. Nothing in stage 4 should add a judge or a
new way to decide a verdict.

**Open on #712 (the step-2 piece and smaller):** a measured NEGATIVE verdict on
interpretation against a fixed corpus (the candidate rule: the monitor's
distance after a bounded alignment, in d or Q for XRD, one table for the regime
choice and the verdict); the certification bar scaled to the reference's own
noise and followers checked against the regime's units seen so far (a fixed
0.10 under-certifies one-unit references); hysteresis on the 10 % strong-peak
bar; number ↔ symbol space-group names; positions in flat XPS/EPR layouts;
an impurity ≥ 12 % on same-phase data as a flag; a recipe approved on physics
grounds inside the soft band having no deterministic replay gate; the
`_prior_regimes` re-parse and `DriftMonitor.locate` for the judge; per-metric
outlier pools when a mixed replayed/fresh regime becomes reachable. From
#715: #718 (a fan-out branch's own depth keys never reach the child, a
pre-existing bug) and #719 (re-escalation on every ladder attempt, test
hygiene, a broad guard pinning `main`).

**Stage 4 scoping, decided 2026-10-03.** The first stage-4 PR is the LOCAL
scheduler only — swarm budgets with reservation, the circuit breaker, process
workers for heavy items, the memory plan sized from measured peaks per item
class and the runtime guard — verified on this machine. The AWS items (worker
tasks per item class, OOM and Spot reconciliation against ECS, per-campaign
provider quotas) follow as their own PR with the hosted-campaigns work: Bedrock
via IAM is blocked until the account is verified (nothing in an ECS task can
call the model, so it cannot be live-checked), they are infrastructure rather
than scheduling logic, and they reuse the local coordinator's decisions, which
must settle first. The ECS worker contract (task spec, item class → task size,
the reconciliation events) may be written into this proposal before the local
PR so that PR already stamps what the AWS layer will read.

## Since the stage-4 scoping: what landed, and what it changes (2026-10-06)

Several PRs merged after the scoping touch what stage 4 was to build.

**Already on `main`, no longer stage-4 work:**
- **The input-based memory estimate (#724, #750).**
  - It follows the largest unit, not the sum, and walks nested folders.
  - It reads an `.npy` from its header and an HDF5 file from its largest
    dataset uncompressed.
  - It estimates a raw-instrument folder by its preparation, and multiplies
    by `series_workers`.
  - It is what a class with no measured peak falls back to.
- **Per-item wall-clock budgets (#700, #755).** A swarm analysis item gets
  the fan-out branch's budget rule (a datacube series or a raw instrument
  container gets its multiple), and a fan-out branch keeps its own depth
  (#718). The swarm budget's open part is tokens with reservation.
- **Cancellation reaches executor work (#685 / #760, #768).**
  - Model-written code and external engines run through the executor's
    tracked runner (`executors._run_tracked`): own session, Stop
    registration, whole-tree kill on a timeout or a cancel.
  - A cancelled worker (Stop, a time budget, the memory guard) ends the
    engine it runs, and no further engine run starts.
  - A turn's Stop now adds to a thread's own cancel instead of replacing it.
- **Fresh-interpreter workers (#721, #730).** `utils.child_process.run_in_child`
  is the one way SciLink starts a worker process, never a `multiprocessing`
  spawn pool, which re-imports the caller's script. Every SciLink entry point
  refuses to start inside a spawn bootstrap.
- **A submit / poll seam on the cluster executor (#766).** `ClusterExecutor`
  splits `run()` into `submit(...) → handle` and `poll(handle)`, with a
  cancel-aware wait.

**What this changes in the stage-4 PR:**
1. **Process workers are stoppable, or they are not done.** `run_in_child`
   still starts its child with a plain `Popen`. Neither Stop nor the memory
   guard reaches it (the gap noted in #730), and that would hold for every
   process worker. It must:
   - run through the tracked runner (own session, the item's cancel
     registered, whole-tree kill);
   - be waited on by a thread that carries the item's cancel
     (`inherited_context`).
2. **The table of measured peaks comes from process workers** (see "Memory
   is declared per item class"). A threaded item's peak cannot be split from
   the process's.
3. **One worker contract for every placement.** Below, in the shape #766
   gave the cluster executor.
4. **A cancelled item cancels its cluster jobs.** `ClusterExecutor`'s
   `cancel_check` is whatever the caller passes, and nothing passes the
   worker's cancel. A swarm simulation item cancelled for time or memory
   would leave its scheduler jobs running. The default should be the
   submitting thread's cancel (`log_context.cancel_requested`).
5. **Fan-out gets the refusal and the guard.** It still admits a branch
   whenever nothing else is running (`_admit_branch`), so the 8 GB analysis
   that froze an 8 GB laptop would still start as a fan-out branch. The
   swarm's `capacity_plan` refusal and `_guard_memory` should be shared, not
   copied.

**The worker contract.** One interface, three placements, so the local
scheduler already stamps what the AWS layer will read:

```
submit(spec) -> handle
    spec: plain data, no secrets — mode, task, context, data paths, the
    item's budgets (time, tokens), the board snapshot it may read, its item
    directory. A worker writes its findings to <item_dir>/findings.jsonl;
    the coordinator stays the board's only writer.
poll(handle) -> {state, result?, peak_rss_bytes?, stop_reason?}
    state: queued | running | done | failed | cancelled | out_of_memory | interrupted
cancel(handle) -> None          # idempotent
```

| placement | submit | poll / peak | cancel | `out_of_memory` / `interrupted` from |
|---|---|---|---|---|
| local process | `run_in_child` through the tracked runner | the result file; `wait4` / `ru_maxrss` | whole-tree kill | killed by SIGKILL with no result (the guard or the OS); lost on a restart |
| HPC job | `ClusterExecutor.submit` (#766) | `ClusterExecutor.poll`; the scheduler's accounting | scheduler cancel | the scheduler's exit state |
| ECS task | `RunTask` with the spec | `DescribeTasks`; the task's memory metric | `StopTask` | the task's stop reason (the container killed for memory, a Spot reclaim, a lost host) |

- **The coordinator's loop** polls handles instead of thread futures.
- **The memory guard** cancels through the contract.
- **An item `out_of_memory`** is retried once at the next size (a task) or
  alone (a local process).
- **Thread items** (light, LLM-bound) keep running as threads in the
  coordinator's process; the contract is for what is heavy.

**Unchanged:** token budgets with reservation, the circuit breaker, the
measured-peak table, the scoping (local scheduler first, the AWS placement
with the hosted-campaigns work), and "nothing in stage 4 adds a judge or a
new way to decide a verdict".

## Stage 4 on `main`: the local scheduler (2026-10-06)

Built from "Since the stage-4 scoping", as one PR ("Swarm stage 4: the local
scheduler"). What it is, by the five points of that section:

1. **Process workers are stoppable.** `utils.child_process.run_in_child`
   runs its child through the executor's tracked runner
   (`executors._run_tracked`, with an `on_start` hook): own session,
   registered to the waiting thread, whole tree killed on a cancel, and the
   wait ends with the thread's stop once the child is gone. `run_child`
   returns the value with what the child cost. The hyperspectral replay pool
   submits through `attributed_to_current`, so a series' own cancel reaches
   its replay children, and an interrupt during `collect` ends them (a child
   in its own session no longer gets the terminal's signal).
2. **The measured table** (`meta_agent/peaks.py`, `measured_items.json`
   under the SciLink home, under `path_lock`). Per item CLASS —
   `mode:kind:largest-unit-bucket:units`, computable from the spec alone,
   so it is known before the item runs — the MAX peak and tokens seen and
   the number of runs; only a run that did its class's work is recorded.
   A process worker's peak is the SUM over its tree sampled every 0.5 s
   (what the item holds on the machine; `ru_maxrss` is one process's own
   peak) or the child's own `ru_maxrss` when sampling is not possible. A
   thread item records tokens only. `fanout.estimate_item` sizes an item at
   its class's measured peak × 1.2, else the #750 input-based estimate; the
   plan gate shows "measured" against "~".
3. **The worker contract** (`meta_agent/placements.py`), as written in the
   scoping, with two placements: `thread` (today's path) and `process`
   (`LocalProcess`: a thread attributed to the item's own runs `run_child`
   on `placements:run_item`; the handle carries state, result, sampled and
   final peak, stop reason, the child's usage by model, its unanswered
   questions). The child's side rebuilds the host from the spec (model,
   endpoints, file roots, knowledge dir, the skill and MCP extensions; the
   keys come from the environment it inherits, the sandbox approval travels
   as a flag), answers every question with its default and counts it, and
   returns plain data. The item's own thread is the waiter: it submits,
   polls until terminal (its own cancel cancels the worker) and closes the
   delegation; the parent charges the child's usage to the item's worker
   tag (one record per model). State mapping: a cancel asked → `cancelled`;
   SIGKILL with nothing returned and no cancel → `out_of_memory` (rerun
   alone once by the coordinator); any other loss → `failed`.
   *The rule:* an analysis item with data runs as a process when nobody
   attends the swarm; an attended swarm keeps threads (its questions need
   the person's channel — a question channel across processes is not built);
   planning and simulation items are threads. `SCILINK_SWARM_PLACEMENT`
   overrides. A host whose key is not in the environment, or that shares a
   callable tool extension, keeps the item a thread, reason on the entry.
4. **A cancelled item cancels its cluster jobs.** `ClusterExecutor`'s
   `cancel_check` defaults to `log_context.is_cancelled` (the thread's own
   cancel or its turn's Stop).
5. **Fan-out gets the refusal and the guard, shared.** `fanout.plan_capacity`
   (the swarm's `capacity_plan` body) refuses a branch larger than the host
   before the confirmation (listed as `not_started`; fewer than two left is
   a `does_not_fit_host` decline), and `fanout.guard_memory` + `Drain` are
   the one guard both loops run: the swarm cancels the item that HOLDS the
   most (the sampled RSS of a process worker, else the estimate) through the
   placement's `cancel`; the fan-out cancels the heaviest branch, closes it
   `memory_pressure`, and reruns it alone once in its own session (the
   resume path's retry shape), in `run_fanout` and `resume_fanout` alike.

**Also from §5, now on `main`:** the token budget — `budget.max_tokens`,
reserved per item at admission from the class's measured spend (else a
per-mode placeholder), reconciled on completion from the per-worker usage
counter (`tracing.worker_usage`), an item or a reaction the remainder cannot
cover refused with the figures; a cancelled item is charged for what it
spends while it winds down (a thread item, from its own thread when it
ends; a process item, from the usage file its child keeps on every call, so
a killed worker is charged too) — and the circuit breaker
(`llm_limiter.note_provider_failure/ok`, fed by `call_with_retries`), per
model: a model whose last minute holds at least six retryable failures that
are at least half of its calls is tripped, and admission of new work on
that model is held for 30 s (a success counts toward the ratio and does not
end a hold early — in a brown-out the running work keeps succeeding now and
then, and a breaker any success closed never tripped); running work keeps
its own retries, and one model's throttling never holds another's items.

**What the first review changed (round 1).** A cancel kills the
descendants that lead their own sessions too (`_kill_process_tree`
snapshots the tree and kills the survivors: a worker in one long C call
runs no SIGTERM handler, so its generated script outlived it). A process
item's console goes to `<item dir>/worker.log` and is relayed line by line
into the turn (the web UI and the shell see it; before, it went to the raw
console under the shell's display). A key the child would not find keeps
the item a thread: on the proxy path the child reads `SCILINK_API_KEY`
only, so the meta's key must be that variable's value; the embedding and
FutureHouse keys are held to the same rule. A measured class is refused on
its RAW peak (the ×1.2 headroom is for admission: a class that ran here
must not be refused here from then on), and the class key names the
series' replay workers (`:wN`), so a four-worker measurement never sizes a
one-worker run or the reverse. A fan-out branch is refused only on a
measured peak — its input-based estimate is coarse by design and must
never refuse what the machine can run; the swarm refuses on the estimate
as it did before. The cluster executor's default cancel ends the caller
(`raise_if_cancelled` before a submit, the thread's stop after cancelling
a job), so a cancelled campaign item does not critique and resubmit; the
poll sleep reads the cancel every second. A run that ends on memory
records its peak. `peaks.forget(class)` resets a class. A question in a
worker process is marked unanswered (`mark_timed_out`), so no gate records
it as a decision.

**Left for the AWS PR, as scoped:** the ECS placement (`RunTask` /
`DescribeTasks` / `StopTask`, the task's memory metric, Spot and OOM
reconciliation), per-campaign quotas, the board across tasks. **Not built,
deliberately:** a question channel across processes (an attended swarm's
heavy items stay threads until one exists); a per-item live RSS for thread
items (the process's own cannot be split). **Measured:** a fresh worker
interpreter on this package is about 200 MB before it does anything, which
is the floor every process item pays.

**What "the core is done" means after stage 4, and what it does not.** On
one machine, §1–§5 of the design are on `main`: ephemeral workers of every
mode, the board with computed independence, deterministic reactions with
taint and retraction, one question queue whose unattended gates never
count as a decision, and a scheduler that refuses, admits, measures,
guards, budgets and stops. The limits that remain, in one place:

- *Placement.* Only `thread` and `process` exist; the HPC and ECS rows of
  the contract are later PRs' (the ECS row with per-campaign quotas and the
  board across tasks, in the AWS PR). **Simulation swarms** are complete
  only for their LLM-bound part (input generation, validation, reading
  outputs): a simulation item is a thread sized at the mode floor, a heavy
  local engine run inside it is neither measured nor placed in a process,
  and the compute part — one scheduler job per member, admitted by the
  scheduler's resources rather than the coordinator's host, polled through
  `ClusterExecutor.submit/poll` (#766) with `run_many` as its batch form —
  is the HPC-job placement of the contract, which answers #767 and waits on
  #696 (the connection) and #745 (concurrent dispatch). What stage 4 gives
  that path today: the row of the contract it will implement, a cancelled
  item cancelling its jobs (`cancel_check` defaults to the thread's own
  cancel, and a turn's Stop now cancels a running cluster job, within a
  second), the tracked engine runner, the token budget and the breaker. An
  open decision for that placement: whether queue wait counts against the
  item's wall-clock budget — today a job still queued when the budget runs
  out is cancelled with the item.
- *Attended swarms.* Heavy items stay threads, because a worker process
  has no channel to the person; a Stop reaches them cooperatively (on a
  print or a wait), not by a kill.
- *Measurement.* A thread item's memory is not measured; a class's first
  run is sized by the input-based estimate, and a class is as coarse as
  `mode:kind:size:units[:wN]` (two very different analyses of same-sized
  cubes share a row, sized by the larger). The guard compares a process
  item's LIVE RSS with a thread item's ESTIMATE, so it can cancel a process
  worker while a thread item holds more. A tree-summed peak counts shared
  libraries once per process, so it leans high. `Drain` gives up a hung
  branch's admission after its timeout while the thread may still hold its
  memory.
- *Tokens.* The per-mode reservations are placeholders until a class is
  measured; the counter is exact for tagged threads and for a process
  worker's own calls (its final report, or the usage file it keeps when it
  is killed), not for an untagged helper thread an agent starts without
  `attributed_to_current` / `inherited_context`, and a thread item that
  keeps calling after the swarm returned is charged to the ledger when its
  thread ends, after the result the caller already has.
- *Independence and taint.* Independence is the read graph (no common
  ancestry); taint stays on the board, unread by fusion and scheduling.
- *Subjects* are strings; *persistent specialists* read nothing from the
  board; there is no board view in Mission Control and no swarm plan at
  the web gate (stage 6).
- *Several instruments* (stage 5): the instrument worker, the log-to-board
  bridge, priority classes, standing subscriptions across turns.

## Starting stage 2 (the board)

The design is §2 above; the tests to write are listed under "Build order".
The board is the first thing that makes items collaborate, so it is also
where the independence rule gets its structural form. One PR for the stage
(the user's rule), verified like stage 1: the full suite against a `main`
worktree by the set of failing test ids, and live checks on Bedrock from a
frozen snapshot, one heavy run at a time.

**Step 1: the record and the writer.** A `scilink/agents/meta_agent/board.py`:
the record schema of §2, `Board.append(record)` as the only writer (the
coordinator's thread; workers post through an in-process queue the
coordinator drains on each poll), `board.jsonl` under the meta session with
an in-memory index, a torn last line skipped on load (as `recipes()` does),
and `Board.snapshot(subject=, kind=, include_provisional=False)` returning a
filtered, immutable view with the ids it returned. `supersedes` and
`retraction` records, and the fold that gives the current view.

**Step 2: posting.** `_run_item` posts a worker's verified findings at its
commit point (after `run_task` returns): one `claim` per `key_findings` entry
with the analysis id as evidence, `recipe` / `structure` / `parameter_point`
where the mode produces them, `hazard` for a blocking plan conflict. What
"verified" means per mode is what the mode already checks (QC and replay
gates, a human-approved or unattended plan is NOT verified, a validated
structure). A worker's own result stays the ledger entry; the board holds
the typed, small records with files by path.

**Step 3: reading.** An item opts in with `reads_board`; the read happens once
at the start of the item (a rendered block in the task, the shape of
`_steering_block`, `fanout.py:853`) and the returned ids are stamped on the
ledger entry as `reads`. `check` items (best-of-N candidates, audits, fusion
verification) are refused a read by the API, not by prompt text. Board
context is additive by type: hint kinds only, never a gate parameter.

**Step 4: independence.** `independent_support(claim, board)`: the count of
agreeing authors whose transitive read sets do not include another
supporter's finding. Fusion (`fuse_delegations`,
`meta_orchestrator_tools.py:1084`) reports it beside the raw count; the
existing `informed_by` / `informed_via` stamps
(`meta_orchestrator.py:1544`, `fanout.py:1812`) become reads on the board, so
the prompt-level discount in fusion renders a number instead of guessing.

**Step 5: the meta reads the board between runs.** A `get_board(subject,
kind)` tool (filtered, private fields dropped, provisional excluded unless
asked) so the meta's model can launch the next swarm from what the last one
found. This is where "delegate tasks" lives; it stays at turn granularity.

**Tests:** independence counting on constructed read graphs; a check
refused a read; verified-only propagation; a superseded record folded out;
a torn last line skipped; the writer serialised under concurrent posts;
the board in the meta checkpoint and restore.

**Live checks that close stage 2:** a swarm of two analyses on one subject
followed by a swarm that reads their claims (the second run's ledger entries
show the reads); fusion of two independent analyses reporting
`independent_support == 2`, and of one informed by the other reporting 1.

## Starting stage 3 (reactions)

Drafted while #702 was in review, from §3 ("The coordinator"), robustness
items 2–4 and the stage-3 entry in "Build order". One PR, "Swarm stage 3",
verified like stages 1 and 2. Two rules from the stage-2 reviews come first,
because they would have saved five rounds there:

- **A decision is made where its information is, never reconstructed.** The
  coordinator records an item's cause and chain at the moment it enqueues
  the item, marks taint at the moment a finding is withdrawn, and refuses a
  cycle at the moment a subscription would fire. Nothing reads the ledger
  afterwards to work out what caused what.
- **The harness before the code.** A real-path test drives `run_swarm` with
  fake workers that post findings through the real board and the real
  coordinator loop (the shape of `tests/test_run_swarm.py`), and the
  stage-2 swarm tests run unchanged under the new coordinator: a swarm with
  no subscriptions must behave exactly as it does today. Shapes come from
  the code (as in `tests/test_series_verdict_path.py`), not from a reading
  of it.

**Dependencies, each its own small PR or decision before the stage:**
- #705 — an analysis worker takes a script FILE as its recipe. Without it a
  "replay the verified recipe on the new dataset" subscription cannot name
  the board's copy under `swarm/recipes/`. *Done (see "After stage 2").*
- #704 — which script a reuse of a refit series replays. A subscription
  makes that choice with nobody looking. *Decided: the series' locked
  recipe, the script its table rests on (see "After stage 2").*
- Provisional findings do not propagate by default (open question, kept).

**Step 1: the subscription, declared and matched without a model.**
`run_swarm(work_items, subscriptions=None, budget=None)`. A subscription is
`{on: {kind, subject?, status: "verified"}, enqueue: {mode, label, task,
reads_board?, check?}, max_fires: 1}`; `task` is a template with a few named
fields (`{finding.text}`, `{finding.path}`, `{subject}`, `{analysis_id}`),
filled by the coordinator, no code and no model. Matching is equality on
`kind`, normalised `subject` and `status`; a finding fires a subscription at
most once, and a subscription fires at most `max_fires` times per swarm.
Pure functions, tested on their own.

**Step 2: the coordinator reacts.** After an item closes and posts, the loop
evaluates every subscription against the records it just posted and
enqueues the fired items under the existing rules — capacity, memory
admission, the item budget, `SWARM_MAX_ITEMS` counting fired items, and a
per-subject re-trigger cap (`SWARM_MAX_TRIGGERS_PER_SUBJECT`, 2). A fired
item is an ordinary ledger entry with `caused_by` (the finding ids) and
`chain` (the triggering item's chain plus `(mode, subject, kind,
finding_id)`), stamped when it is enqueued. Workers still never start
workers: every reaction is the coordinator's, so the depth stays 1 in the
proposal's sense while a chain may be longer than one hop.

**Step 3: `task_request`.** A worker's `suggested_followups` become
`task_request` records (provisional, never readable by default, author =
the worker). A request becomes an item only through a subscription on
`kind: task_request` or an explicit allowance in the budget; otherwise it is
listed in the swarm result as asked and not done. "Workers ask, the
coordinator decides."

**Step 4: cycles and oscillation.** A subscription that would fire an item
whose `chain` already holds the same `(mode, subject, kind)` is refused and
the refusal recorded (`refused_cycles` in the result, `refused` on the
would-be entry). A subject re-triggered past its cap is refused the same
way. A supersede chain longer than two on one subject stops the coordinator
scheduling a third round and reports the disagreement (robustness item 3);
the fold already exposes the chain.

**Step 5: retraction and taint.** `Board.retract` and `supersedes` get
their callers: a meta tool `retract_finding(finding_id, reason)` (the
person's or the meta's act, author mode `coordinator`), and a worker
superseding its own earlier claim on the same subject. When a finding is
retracted or superseded, the coordinator marks every dependent in
`read_closure` `tainted` (a fold status, as today's `superseded`), re-runs
the tainted items' sources if the budget allows, else lists them as resting
on a withdrawn finding. No automatic retraction: a check that disagrees is a
result to show (`independent_support`, `audit_split`), not a withdrawal.

**Step 6: hazards reach everyone.** A `hazard` on a subject is delivered to
every reader on that subject whatever its `kinds` filter (robustness item
4); today a filter can drop it.

**Step 7: what the result and `get_board` show.** Fired items with their
cause, refused cycles and capped subjects, tainted findings and what was
re-run, supersede chains. Nothing new in the prompt beyond naming these.

**Tests** (the harness first): a two-item cycle is refused and recorded; a
subscription fires once per finding and stops at `max_fires` and at the
subject cap; a retraction taints exactly its dependents and nothing else,
and the re-run happens under budget and is listed without; a supersede
chain of three on one subject stops the coordinator; a hazard reaches a
reader that filtered it out; a `task_request` becomes an item only through
a subscription; chains and causes survive a checkpoint; a swarm with no
subscriptions produces the same ledger and board as stage 2 (the existing
`test_run_swarm.py` and `test_board.py` unchanged).

**Live checks that close stage 3:** the proposal's canonical chain on
Bedrock — an analysis whose verified claim fires a simulation item (inputs
only, #696), the board inspected afterwards for the cause and chain; after
#705, a verified recipe firing a replay on a new dataset of the same subject;
and a planted cycle (two subscriptions that would fire each other) refused
live with the refusal in the result.

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
