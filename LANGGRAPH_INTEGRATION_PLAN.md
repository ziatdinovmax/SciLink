# LangGraph Integration Plan

## Overview

This document describes the plan to make LangGraph the runtime backbone of SciLink's orchestration layer, and to leverage that backbone to enable parallel agentic analysis pipelines.

The work has two inseparable parts:

1. **Backbone refactor** — replace the hand-rolled `while iteration < MAX_TOOL_ITERATIONS` loops in all three orchestrators with LangGraph graph execution. This is the prerequisite for everything else.
2. **Parallel pipelines** — once the backbone is in place, build multi-strategy image analysis as the first parallel graph, which becomes the natural foundation for multi-modal fusion.

---

## Motivation

### Why the backbone needs to change

The three orchestrators (`AnalysisOrchestratorAgent`, `PlanningOrchestratorAgent`, `SimulationOrchestratorAgent`) each run a hand-rolled chat loop:

```python
while iteration < MAX_TOOL_ITERATIONS:
    response = llm(messages)
    if tool_call:
        result = execute_tool(...)
        messages.append(result)
    else:
        break
```

This works for sequential single-agent flows, but it cannot express:

- **Parallel branches** — fan out to N strategies, run each concurrently, join the best result
- **Per-branch checkpoints** — pause and resume individual branches independently
- **Per-branch human-in-the-loop** — prompt the user inside one branch without blocking others
- **Subgraph composition** — nest the verification-retry loop as a reusable subgraph rather than duplicating it across `image_analysis_controllers.py` and `curve_fitting_controllers.py`
- **Streaming node outputs** — emit partial results as each branch completes rather than waiting for all

LangGraph provides all of this natively. The upgrade cost is paid once; the parallel pipeline work and everything that follows builds on it without further hand-rolling.

### Why parallel pipelines

Parallel agentic analysis pipelines are a natural next step for SciLink, and where LangGraph's graph runtime is most clearly the right tool.

**Multi-strategy spectroscopy fitting and image analysis** is the first obvious case. The verification-retry loop today locks in one analysis strategy early (in `ImagePlanningController`) and iterates on *execution* quality within that strategy. If the chosen strategy is wrong, no amount of verification iterations recovers cleanly. Multi-strategy execution fixes this structurally: instead of locking in one model or plan, we kick off several candidate strategies in parallel, run each through the existing verification-retry loop, and let a join step pick the winner by quality score.

**Multi-modal fusion** is the second case and extends the same pattern. Spectral, image, and point-cloud data from multiple instruments — XRD alongside TEM alongside EELS-SI, for example — run as parallel subgraphs; the join node produces the final unified interpretation and dispatches recommendations back to each individual instrument.

Bolting either of these onto the existing sequential pipelines would mean hand-rolling a small, shakier version of a graph runtime: async orchestration, parallel-aware checkpoints, per-branch human-in-the-loop, and a merge step. LangGraph gives us all of that natively, with each branch as an isolated subgraph that composes cleanly with the rest of the pipeline.

Within this project, the prototype target is **multi-strategy image analysis**. It is the most contained first case (homogeneous branches, single data type, well-defined quality metric) and serves as a natural foundation for the multi-modal fusion work that follows.

---

## Current State

### What exists

| File | Status |
|---|---|
| `scilink/graphs/__init__.py` | Package scaffold, documentation only |
| `scilink/graphs/state.py` | `AnalysisState(MessagesState)` TypedDict — fields mirror orchestrator instance variables. **Not imported anywhere live.** |
| `scilink/graphs/analysis.py` | `build_analysis_graph()` wiring a ReAct topology. `call_model` and `execute_tools` nodes both raise `NotImplementedError`. Module-level `graph = build_analysis_graph()` instantiated at import. |

All three files landed in commit `58f849e` ("Adding LangGraph backbone") on branch `feat/langgraph-skeleton`. They define the shape of the graph and establish that LangGraph is a core dependency (`langgraph` in `pyproject.toml`), but no live orchestrator code imports or uses them yet.

### What the live orchestrators do today

- Three independent orchestrator classes sharing ~600 lines of near-identical structure (chat loop, checkpoint, MCP bridge, autonomy mode, tool registry, history trimming).
- `AnalysisOrchestratorAgent` manages: `current_data_path`, `current_data_type`, `selected_agent_id`, `analysis_results`, `active_knowledge`, `message_count`, `analysis_run_counter` as mutable instance variables.
- `AnalysisState` in `graphs/state.py` already mirrors these fields — the state schema is the designed seam between the current instance-variable model and the graph-native model.
- Series processing in both `UnifiedImageProcessingController` and `UnifiedSeriesProcessingController` is a sequential `for` loop. No parallel execution exists anywhere in the analysis pipeline.
- The verification-retry loop is duplicated in `image_analysis_controllers.py` (~lines 3037–3219) and `curve_fitting_controllers.py`. Same annealing logic (three temperature levels: tight/warm/hot), same patience counter, independent code.
- Async exists only at the MCP boundary (`mcp_server.py`, `mcp_client.py`). The orchestrators are synchronous.

### State schema notes

The current `AnalysisState` in `state.py` uses a flat TypedDict extending `MessagesState`. The target architecture (see below) introduces a three-level hierarchy (`OrchestratorState` → `AnalysisOrchestratorState` → parallel/fusion states). `AnalysisState` will be renamed to `AnalysisOrchestratorState` during Phase 1 as part of that refactor; `state.py` is the only consumer today so the rename is safe.

---

## Target Architecture

```
┌──────────────────────────────────────────────────────────┐
│  CLI / UI / Meta Agent                                   │
└───────────────┬──────────────────────────────────────────┘
                │  graph.stream() / graph.invoke()
┌───────────────▼──────────────────────────────────────────┐
│  Orchestrator Graph  (one per mode: analyze/plan/sim)    │
│                                                          │
│  START → route_intent → [tool nodes] → END              │
│                   ↑_________↓                            │
│           MemorySaver checkpointer                       │
└───────────────┬──────────────────────────────────────────┘
                │  subgraph.invoke() / Send API (parallel)
┌───────────────▼──────────────────────────────────────────┐
│  Analysis Subgraph   (multi-strategy fan-out)            │
│                                                          │
│  plan → [strategy_A, strategy_B, strategy_N] → join     │
│            each branch: verify_loop subgraph             │
└──────────────────────────────────────────────────────────┘
```

### State layers

Three state TypedDicts, each extending the one above:

```python
class OrchestratorState(MessagesState):
    """Shared by all three orchestrator graphs."""
    autonomy_mode: str           # co_pilot | supervised | autonomous
    active_skill: str | list[str] | None
    session_dir: str
    checkpoint_data: dict
    mcp_connections: list[str]

class AnalysisOrchestratorState(OrchestratorState):
    """Analysis-specific fields (mirrors current instance variables)."""
    current_data_path: str | None
    current_data_type: str | None
    selected_agent_id: int | None
    analysis_results: dict
    active_knowledge: dict
    message_count: int
    analysis_run_counter: int

class ParallelAnalysisState(TypedDict):
    """State for the multi-strategy parallel subgraph."""
    data_path: str
    data_type: str
    skill: str | list[str] | None
    strategies: list[StrategyConfig]          # populated by plan node
    branch_results: Annotated[list, operator.add]  # reducer: collect from all branches
    best_result: AnalysisResult | None
    join_rationale: str
```

The `Annotated[list, operator.add]` reducer on `branch_results` is the LangGraph mechanism that lets parallel branches write to the same state key without conflicts.

---

## Phase 1 — Backbone Refactor

**Goal:** Replace the hand-rolled loops in all three orchestrators with LangGraph execution. Behavior is identical to today; no parallel logic yet. Establishes the foundation.

### 1.1 Implement `graphs/analysis.py` nodes

The scaffold already has the right topology. Fill in the two stub nodes:

**`call_model` node**

```python
def call_model(state: AnalysisOrchestratorState, config: RunnableConfig):
    """Invoke the LLM with the current message history and bound tools."""
    llm = _get_llm(config)                 # pulls model from configurable
    bound = llm.bind_tools(state["registered_tools"])
    response = bound.invoke(state["messages"])
    return {"messages": [response]}
```

**`execute_tools` node**

```python
def execute_tools(state: AnalysisOrchestratorState, config: RunnableConfig):
    """Execute tool calls from the last AI message."""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for tc in tool_calls:
        result = _dispatch_tool(tc["name"], tc["args"], state, config)
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}
```

**Routing function** (replaces the `if tool_call: ... else: break` logic):

```python
def should_continue(state: AnalysisOrchestratorState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tools"
    return END
```

**Graph wiring** (already present in scaffold, just connect live nodes):

```python
graph = (
    StateGraph(AnalysisOrchestratorState)
    .add_node("call_model", call_model)
    .add_node("execute_tools", execute_tools)
    .add_edge(START, "call_model")
    .add_conditional_edges("call_model", should_continue)
    .add_edge("execute_tools", "call_model")
    .compile(checkpointer=MemorySaver())
)
```

`MAX_TOOL_ITERATIONS` enforcement moves to an interrupt condition or a step counter in state rather than a `while` loop counter.

### 1.2 Migrate `AnalysisOrchestratorAgent`

- Replace `_handle_openai_chat()` and `_handle_litellm_chat()` with a single `_invoke_graph()` method that calls `graph.stream()` (streaming) or `graph.invoke()` (non-interactive).
- Move mutable instance variables (`current_data_path`, `selected_agent_id`, etc.) into `AnalysisOrchestratorState`. The orchestrator instance becomes a thin wrapper that holds the graph and thread config.
- `_auto_checkpoint()` is replaced by MemorySaver; explicit JSON checkpoints become optional secondary persistence.
- `_trim_history()` becomes a node or a state reducer rather than imperative list slicing.

**Autonomy mode → interrupt conditions:**

| Mode | LangGraph mechanism |
|---|---|
| `CO_PILOT` | `interrupt_before=["execute_tools"]` — user approves every tool call |
| `SUPERVISED` | `interrupt_before=["execute_tools"]` with a filter: only interrupt on `run_analysis` and destructive tools |
| `AUTONOMOUS` | No interrupts; `graph.invoke()` runs to completion |

**Human-in-the-loop pattern** (replacing the `input()` calls scattered through controllers):

```python
# In CO_PILOT mode, before execute_tools:
human_response = interrupt("Proposed tool call: ...")  # LangGraph interrupt
# Resume:
graph.invoke(Command(resume=human_response), config=thread_config)
```

### 1.3 Migrate `PlanningOrchestratorAgent` and `SimulationOrchestratorAgent`

Same pattern as 1.2. Each gets:
- Its own `StateGraph` with the same `call_model → execute_tools` ReAct topology
- Its own state TypedDict extending `OrchestratorState`
- Separate graph files: `graphs/planning.py`, `graphs/simulation.py`

`SimulationOrchestratorAgent.run_task()` maps to `graph.invoke()` with `autonomy_mode="autonomous"` in config.

### 1.4 Verification-retry loop as a reusable subgraph

The duplicated verification-retry loop in `image_analysis_controllers.py` and `curve_fitting_controllers.py` becomes a single composable subgraph in `graphs/verification.py`:

```python
class VerificationState(TypedDict):
    analysis_config: dict
    current_result: AnalysisResult | None
    best_result: AnalysisResult | None
    best_score: float
    verification_history: list[VerificationRecord]
    iteration: int
    annealing_level: int          # 0=tight, 1=warm, 2=hot
    patience_counter: int
    approved: bool
    human_feedback_requested: bool

def build_verification_subgraph() -> CompiledGraph:
    return (
        StateGraph(VerificationState)
        .add_node("run_analysis", run_analysis_node)
        .add_node("verify_quality", verify_quality_node)
        .add_node("apply_feedback", apply_feedback_node)
        .add_node("anneal", anneal_node)
        .add_node("human_feedback", human_feedback_node)
        .add_edge(START, "run_analysis")
        .add_edge("run_analysis", "verify_quality")
        .add_conditional_edges("verify_quality", route_verification)
        .add_edge("apply_feedback", "anneal")
        .add_edge("anneal", "run_analysis")
        .add_edge("human_feedback", "run_analysis")
        .compile(checkpointer=MemorySaver())
    )
```

`route_verification` encodes the annealing + patience logic that today lives in the `while` loop body. The subgraph is then composed into both the image analysis pipeline and the curve fitting pipeline.

### 1.5 Deliverables

- `scilink/graphs/analysis.py` — fully implemented (not stubbed)
- `scilink/graphs/planning.py` — new
- `scilink/graphs/simulation.py` — new
- `scilink/graphs/verification.py` — reusable verification-retry subgraph
- `scilink/graphs/state.py` — expanded with `OrchestratorState`, `AnalysisOrchestratorState` (renamed from `AnalysisState`), `PlanningOrchestratorState`, `SimulationOrchestratorState`, `VerificationState`
- `AnalysisOrchestratorAgent.chat()` — delegates to `graph.stream()`
- `PlanningOrchestratorAgent.chat()` — delegates to `graph.invoke()`
- `SimulationOrchestratorAgent.chat()` and `run_task()` — delegate to `graph.invoke()`
- `UnifiedImageProcessingController` and `UnifiedSeriesProcessingController` — verification loops replaced with `verification_subgraph.invoke()`
- All existing tests passing; new integration tests for the graph execution path

---

## Phase 2 — Multi-Strategy Image Analysis

**Goal:** Fan out N candidate analysis strategies in parallel, run each through the verification subgraph, join on quality score. This is the first parallel pipeline and the prototype for everything that follows.

The target case is **multi-strategy image analysis**: instead of locking in one model or plan, we kick off several candidates in parallel, run each through the existing verification-retry loop, and let a join step pick the winner. This is also where per-branch human-in-the-loop becomes meaningful — in CO_PILOT and SUPERVISED modes, the user can approve, reject, or redirect individual strategies mid-flight without blocking others.

### 2.1 Strategy generation node

A new `generate_strategies` node replaces the current `ImagePlanningController` lock-in behavior. Instead of committing to one plan, it produces a ranked list of `StrategyConfig` objects:

```python
def generate_strategies(state: ParallelAnalysisState) -> dict:
    """LLM generates N candidate analysis strategies for the data."""
    strategies = llm_plan_strategies(
        data_path=state["data_path"],
        skill=state["skill"],
        n=3,   # configurable; default 3
    )
    return {"strategies": strategies}
```

Each `StrategyConfig` captures everything `ImagePlanningController` currently locks: model choice, preprocessing flags, fitting approach, analysis depth.

### 2.2 Fan-out with the Send API

LangGraph's `Send` API dispatches one `strategy_branch` node invocation per strategy, each with its own isolated copy of state:

```python
def fan_out(state: ParallelAnalysisState) -> list[Send]:
    return [
        Send("strategy_branch", {**state, "strategy": s})
        for s in state["strategies"]
    ]
```

Each `strategy_branch` node invokes the verification subgraph for its assigned strategy:

```python
def strategy_branch(state: ParallelAnalysisState) -> dict:
    result = verification_subgraph.invoke({
        "analysis_config": state["strategy"].to_config(),
        "data_path": state["data_path"],
        ...
    })
    return {"branch_results": [result]}   # reducer appends to shared list
```

Because `branch_results` uses `Annotated[list, operator.add]`, each branch's return merges into the parent state without coordination.

### 2.3 Join node

```python
def join_strategies(state: ParallelAnalysisState) -> dict:
    """Select the best result and generate a rationale."""
    best = max(state["branch_results"], key=lambda r: r.quality_score)
    rationale = llm_compare_results(state["branch_results"])
    return {
        "best_result": best,
        "join_rationale": rationale,
    }
```

The join LLM call compares all branch results and produces a human-readable explanation of why the winning strategy was chosen. This becomes part of the analysis report.

### 2.4 Graph wiring

```python
def build_parallel_image_analysis_graph() -> CompiledGraph:
    return (
        StateGraph(ParallelAnalysisState)
        .add_node("generate_strategies", generate_strategies)
        .add_conditional_edges("generate_strategies", fan_out, ["strategy_branch"])
        .add_node("strategy_branch", strategy_branch)
        .add_node("join_strategies", join_strategies)
        .add_edge("strategy_branch", "join_strategies")
        .add_edge(START, "generate_strategies")
        .add_edge("join_strategies", END)
        .compile(checkpointer=MemorySaver())
    )
```

### 2.5 Integration with the orchestrator

`AnalysisOrchestratorTools.run_analysis()` gains a `parallel_strategies: int = 1` parameter. When `> 1`, it invokes `parallel_image_analysis_graph` instead of `ImageAnalysisAgent.analyze()` directly. When `= 1`, behavior is identical to today (single strategy, no fan-out overhead).

### 2.6 Per-branch human-in-the-loop

In `CO_PILOT` and `SUPERVISED` modes, each branch can independently surface a checkpoint to the user. Because each branch runs as an independent subgraph invocation with its own thread config, `interrupt()` inside one branch does not block others. The user can approve, reject, or redirect individual strategies mid-flight.

### 2.7 Deliverables

- `scilink/graphs/parallel_analysis.py` — `ParallelAnalysisState`, `build_parallel_image_analysis_graph()`
- `scilink/graphs/state.py` — `ParallelAnalysisState`, `StrategyConfig`, `AnalysisResult` types
- `scilink/agents/exp_agents/analysis_orchestrator_tools.py` — `run_analysis()` updated with `parallel_strategies` parameter
- New skill section hint: `ImagePlanningController` updated to emit strategy list rather than locking a single plan (backward-compatible: `n=1` path unchanged)
- Integration test: 3-strategy fan-out on a reference image, assert `best_result.quality_score >= single_strategy_score`

---

## Phase 3 — Multi-Modal Fusion

**Goal:** Spectral, image, and point-cloud data from multiple instruments run as parallel subgraphs; a join node produces the unified interpretation and dispatches recommendations per instrument.

This phase is enabled directly by Phase 2's parallel infrastructure. The case motivating it: spectral data (XRD, Raman, XPS), microscopy images (SEM, TEM, STM), and hyperspectral data (EELS-SI, EDS) from the same sample run side-by-side and the join node produces a single cross-modal interpretation plus per-instrument follow-up recommendations. Each branch internally reuses the verification subgraph from Phase 1.

The structural difference from Phase 2 is that branches are **heterogeneous** (different data types and agent types) rather than homogeneous (same agent, different strategies).

### 3.1 Modal branch nodes

Each modality runs as a `Send`-dispatched branch invoking the appropriate agent:

| Branch node | Agent | Data type |
|---|---|---|
| `spectral_branch` | `CurveFittingAgent` | XRD, Raman, PL, UV-Vis, etc. |
| `image_branch` | `ImageAnalysisAgent` | Microscopy, SEM, TEM, etc. |
| `hyperspectral_branch` | `HyperspectralAnalysisAgent` | EELS-SI, EDS, Raman imaging |

Each branch internally uses the verification subgraph from Phase 1.

### 3.2 Fusion state

```python
class FusionState(TypedDict):
    data_sources: list[DataSource]           # each with path, type, instrument metadata
    modal_results: Annotated[list, operator.add]   # reducer: collect from all branches
    fusion_interpretation: str | None
    per_instrument_recommendations: dict[str, str]
    open_questions: list[str]
```

### 3.3 Fusion join node

```python
def fusion_join(state: FusionState) -> dict:
    """
    Cross-modal LLM synthesis: produce a unified interpretation and
    per-instrument recommendations from all modal results.
    """
    interpretation = llm_fuse_modalities(state["modal_results"])
    recommendations = {
        r.instrument_id: llm_recommend_for_instrument(r, interpretation)
        for r in state["modal_results"]
    }
    return {
        "fusion_interpretation": interpretation,
        "per_instrument_recommendations": recommendations,
        "open_questions": extract_open_questions(state["modal_results"]),
    }
```

### 3.4 Deliverables

- `scilink/graphs/fusion.py` — `FusionState`, `build_fusion_graph()`
- `scilink/agents/exp_agents/analysis_orchestrator_tools.py` — new `run_multimodal_analysis(data_sources: list[DataSource])` tool
- New `DataSource` type in `scilink/graphs/state.py`
- Skill context injection: the fusion join node loads skill context for each modality independently and merges before the synthesis LLM call

---

## Migration Strategy

### Branch strategy

Work happens on dedicated branches off `main`. The current active branch is `feat/langgraph-skeleton` (contains the initial `scilink/graphs/` scaffold). Phase 1 starts from there.

```
main
├── feat/langgraph-skeleton   Initial scaffold (graphs/__init__.py, state.py, analysis.py)
│                             → becomes feat/backbone as Phase 1 work proceeds
├── feat/backbone             Phase 1 (all of it: verification subgraph,
│                             orchestrator migration, state refactor)
└── feat/parallel             Phase 2 + Phase 3 (multi-strategy image analysis,
                              multi-modal fusion) — branches off feat/backbone,
                              not off main
```

`feat/parallel` branches from `feat/backbone` (not `main`) because it depends on the graph runtime established in Phase 1. When `feat/backbone` is reviewed and merged, `feat/parallel` rebases onto the new `main` before its own PR is opened.

Each branch ships as a single PR. If Phase 1 review surface area is too large, it can be split into sub-PRs (`feat/backbone-verification`, `feat/backbone-orchestrators`) that merge sequentially into `feat/backbone` — but `feat/backbone` itself does not merge to `main` until all Phase 1 sub-work is complete and tests pass.

### Sequencing

```
feat/backbone
  Phase 1a: graphs/verification.py (verification subgraph)
      ↓
  Phase 1b: graphs/analysis.py (implement stub nodes; wire AnalysisOrchestratorAgent)
      ↓
  Phase 1c: graphs/planning.py, graphs/simulation.py (same pattern)
      ↓  (feat/backbone → main via PR)

feat/parallel  (branched from feat/backbone before merge, rebased onto main after)
  Phase 2:  graphs/parallel_analysis.py (fan-out on image analysis)
      ↓
  Phase 3:  graphs/fusion.py (multi-modal)
      ↓  (feat/parallel → main via PR)
```

Start with the verification subgraph (1a) because it is self-contained, has clear inputs/outputs, and immediately eliminates the biggest code duplication. It also lets the team validate the LangGraph subgraph composition pattern before touching the orchestrators.

### Backward compatibility

- `ImageAnalysisAgent.analyze()` and `CurveFittingAgent.analyze()` keep their current signatures throughout. The graph migration is internal to those agents.
- `AnalysisOrchestratorAgent.chat()` keeps its current signature. The `graph.stream()` call is behind the same public interface.
- All existing CLI commands (`scilink analyze`, `scilink plan`, `scilink simulate`) work identically throughout the migration.
- `parallel_strategies=1` (Phase 2) reproduces single-strategy behavior exactly.

### Testing approach

- **Unit tests per node**: each graph node function is a pure function of state → state delta; test without graph execution.
- **Subgraph integration tests**: `verification_subgraph.invoke()` on synthetic `VerificationState`; assert it terminates in ≤ `max_iterations` and returns `approved=True` for a clearly good result.
- **Orchestrator smoke tests**: `graph.invoke({"messages": [HumanMessage("analyze X")]})` on a minimal mock tool registry; assert the graph reaches `END` without raising.
- **Parallel regression test**: 3-strategy run on a reference image; assert the winning result quality ≥ single-strategy baseline.
- **Checkpoint round-trip**: interrupt a graph mid-run, serialize MemorySaver state, restore, resume; assert final result matches non-interrupted run.

### Risk areas

| Risk | Mitigation |
|---|---|
| LangGraph async vs. synchronous orchestrators | Use `graph.invoke()` (synchronous) throughout Phase 1. Async migration is a separate follow-on, not part of this plan. |
| MemorySaver in-memory vs. existing JSON checkpoints | Keep JSON checkpoint writing as a secondary side-effect in `_auto_checkpoint()` during Phase 1. Remove in Phase 2 once MemorySaver is proven. |
| `interrupt()` UX in CLI vs. UI | CLI uses `input()` bridge; UI uses existing streaming SSE channel. Both implement the same `Command(resume=...)` pattern. |
| Controller classes becoming nodes | Controllers are stateful objects today. Nodes must be pure functions. Stateless behavior moves to node functions; any genuinely stateful resources (e.g., open file handles) move to `RunnableConfig` (injected, not in state). |
| `MAX_TOOL_ITERATIONS` enforcement | Replace with a `step_count` field in state and a conditional edge that routes to END after N steps with a warning message appended. |

---

## File Map

### New files

```
scilink/graphs/
├── __init__.py          (update)
├── state.py             (expand: OrchestratorState, AnalysisOrchestratorState (rename from AnalysisState),
│                                  PlanningOrchestratorState, SimulationOrchestratorState,
│                                  VerificationState, ParallelAnalysisState,
│                                  FusionState, StrategyConfig, AnalysisResult, DataSource)
├── analysis.py          (implement stub nodes)
├── planning.py          (new)
├── simulation.py        (new)
├── verification.py      (new — reusable verification-retry subgraph)
├── parallel_analysis.py (new — multi-strategy fan-out)
└── fusion.py            (new — multi-modal fusion)
```

### Modified files

```
scilink/agents/exp_agents/
├── analysis_orchestrator.py          (replace _handle_*_chat with graph.stream())
├── analysis_orchestrator_tools.py    (run_analysis: add parallel_strategies param;
│                                      add run_multimodal_analysis tool)
├── controllers/image_analysis_controllers.py
│                                     (UnifiedImageProcessingController:
│                                      replace verification while-loop with
│                                      verification_subgraph.invoke())
└── controllers/curve_fitting_controllers.py
                                      (UnifiedSeriesProcessingController: same)

scilink/agents/planning_agents/
└── planning_orchestrator.py          (replace chat loop with graph.invoke())

scilink/agents/sim_agents/
└── simulation_orchestrator.py        (replace chat loop and run_task with graph.invoke())
```

---

## Open Questions

These are not blockers but should be resolved before Phase 2 starts:

1. **Strategy count default**: 3 parallel strategies is the proposed default. What is the right number given typical compute budget and latency tolerance? Should it be configurable per skill?

2. **Join LLM cost**: The join node makes an additional LLM call to compare branch results. Is this always warranted, or should it be conditional on the score gap between branches being below a threshold?

3. **Branch failure handling**: If one branch raises an exception (e.g., the chosen model fails on an edge-case image), should the join node skip it and report the failure, or should the whole graph fail? Proposal: skip and log; require at least one successful branch.

4. **Checkpointer backend for production**: MemorySaver is in-process and not persistent across restarts. For long-running parallel analyses, a `SqliteSaver` or Postgres-backed checkpointer may be needed. Phase 1 uses MemorySaver; Phase 2 should at minimum prototype with `SqliteSaver`.

5. **Skill injection in parallel branches**: Each branch currently inherits the same skill context. Should branches be allowed to override the skill independently (e.g., branch A uses `raman` skill, branch B uses `general_spectroscopy`)? This would increase strategy diversity.
