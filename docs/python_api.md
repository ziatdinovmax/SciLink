# Python API

## One call per task: `run_task`

Each mode orchestrator (plan / analyze / simulate) exposes the same
programmatic contract — one call, no conversation — which is also how mission
control delegates to them:

```python
from scilink.agents.exp_agents import AnalysisOrchestratorAgent

result = AnalysisOrchestratorAgent().run_task(
    "Analyze ./stem_image.tif — identify the defects and generate scientific claims"
)
```

`run_task(task, context=None, autonomy=None)` runs autonomously by default (it
never waits for a person who isn't there); pass a co-pilot or autopilot level
when a human is attached. The returned dict always has `status`, `task`,
`summary`, `files_produced`, `key_findings`, `suggested_followups` and
`warnings` (plus `error` on failure), and one mode-specific field: `analyses`
(analyze), `campaign_state` (plan) or `structures` (simulate). Use `context` to
hand one task's findings and files to the next.

> In a script, keep the code that runs SciLink under
> `if __name__ == "__main__":`. A process pool using the `spawn` start method
> (the default on macOS and Windows) re-imports your script in every worker, and
> SciLink refuses to start inside such a worker rather than run a second copy of
> your session there.

## Planning

```python
from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel
)
from scilink.agents.planning_agents import PlanningAgent, BOAgent

# The orchestrator, driven in natural language
orchestrator = PlanningOrchestratorAgent(
    objective="Optimize reaction yield",
    autonomy_level=AutonomyLevel.AUTOPILOT,
    data_dir="./experimental_results",
    knowledge_dir="./papers",
)
response = orchestrator.chat("Generate initial plan and analyze batch_001.csv")

# An agent directly
plan = PlanningAgent(model_name="claude-opus-4-6").propose_experiments(
    objective="Screen precipitation conditions",
    knowledge_paths=["./literature/"],
    primary_data_set={"file_path": "./composition_data.xlsx"},
)

# Bayesian optimization
result = BOAgent(model_name="claude-opus-4-6").run_optimization_loop(
    data_path="./optimization_data.csv",
    objective_text="Maximize yield while minimizing cost",
    input_cols=["Temperature", "pH", "Concentration"],
    input_bounds=[[20, 80], [6, 10], [0.1, 2.0]],
    target_cols=["Yield"],
    batch_size=1,
)
```

## Analysis

```python
from scilink.agents.exp_agents import (
    AnalysisOrchestratorAgent, AnalysisMode,
    ImageAnalysisAgent, HyperspectralAnalysisAgent, CurveFittingAgent,
)

# The orchestrator
orchestrator = AnalysisOrchestratorAgent(base_dir="./my_analysis",
                                         analysis_mode=AnalysisMode.AUTOPILOT)
response = orchestrator.chat("Examine ./data/sample.tif")

# Image analysis with the two-tier pipeline
agent = ImageAnalysisAgent(analysis_depth="auto")
result = agent.analyze(
    "stem_image.tif",
    system_info={"experiment": {"technique": "HAADF-STEM"}},
    objective="Identify crystal phases and defects",
)

# A series: the first unit is analysed in full, its verified recipe is locked
# and replayed on the rest, failures are re-analysed, outliers flagged, and the
# trends synthesized. Curves, images and datacubes all run a series this way.
result = agent.analyze(
    ["img_001.tif", "img_002.tif", "img_003.tif"],
    series_metadata={"variable": "dose", "values": [1e14, 1e15, 1e16], "unit": "ions/cm²"},
)

agent = CurveFittingAgent(output_dir="./curve_output", use_literature=True)
result = agent.analyze(
    ["pl_300K.csv", "pl_350K.csv", "pl_400K.csv"],
    series_metadata={"variable": "temperature", "values": [300, 350, 400], "unit": "K"},
)

agent = HyperspectralAnalysisAgent(output_dir="./eels_series")
result = agent.analyze(
    ["eels_300K.npy", "eels_350K.npy", "eels_400K.npy"],
    system_info={"technique": "EELS", "energy_range": {"start": 450, "end": 570, "units": "eV"}},
    series_metadata={"variable": "temperature", "values": [300, 350, 400], "unit": "K"},
    max_series_refits=1,
    series_workers=3,          # locked replays run concurrently
)
```

## Simulation

```python
from scilink.agents.sim_agents import (
    StructurePipeline, PeriodicDFTAgent, run_complete_workflow,
)

# Structure only (engine-agnostic, class-aware)
sp = StructurePipeline(generator_model="claude-opus-4-6")
res = sp.build_structure("a single water molecule", structure_class="molecular")
# res["final_structure_path"] points at structure.xyz

# The full one-shot pipeline (structure + inputs), any scale/engine
run_complete_workflow(
    "diamond Si, 2-atom primitive cell, ground-state SCF",
    scale="periodic_dft", software="vasp",
)

# Quantum ESPRESSO inputs through the engine-agnostic agent
PeriodicDFTAgent().generate_inputs(
    structure_file="POSCAR",
    request="vc-relax of Cu fcc",
    software="qe",
)
```

## What a session leaves on disk

```
planning_session/
├── optimization_data.csv      # accumulated experimental data
├── plan.json                  # current experimental plan
├── plan.html                  # rendered plan
├── checkpoint.json            # session state for restoration
└── output_scripts/            # generated automation code

analysis_session/
├── results/
│   └── analysis_{dataset}_{agent}_{timestamp}/
│       ├── metadata_used.json
│       ├── analysis_results.json
│       ├── visualizations/
│       └── report.html
├── chat_history.json
└── checkpoint.json
```
