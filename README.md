# SciLink

**AI-Powered Scientific Research Automation Platform**

![SciLink Logo](misc/scilink_logo_v3_dark.svg)

SciLink employs a system of intelligent agents to automate experimental design, data analysis, and iterative optimization workflows. Built around large language models with domain-specific tools, these agents act as AI research partners that can plan experiments, analyze results across multiple modalities, and suggest optimal next steps.

---

## Overview

SciLink provides three complementary agent systems that cover the full scientific research cycle:

| System | Purpose | Key Capabilities |
|--------|---------|------------------|
| **Planning Agents** | Experimental design & optimization | Hypothesis generation, Bayesian optimization, literature-aware planning |
| **Analysis Agents** | Multi-modal data analysis | Microscopy, spectroscopy, particle segmentation, curve fitting |
| **Simulation Agents** | Computational modeling | DFT calculations, classical MD (LAMMPS), structure recommendations |

All systems support three autonomy levels:

- **Co-Pilot** (default) — Human leads, AI assists. Reviews every step.
- **Supervised** — AI leads, human reviews major decisions.
- **Autonomous** — Full autonomy, no human review.

---

## Installation

```bash
pip install scilink

# With web UI
pip install scilink[ui]

# With simulation dependencies (ASE, atomate2, etc.)
pip install scilink[sim]
```

### Environment Variables

Set API keys for your preferred LLM provider:

```bash
# Google Gemini (default)
export GEMINI_API_KEY="your-key"

# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# OpenAI-compatible proxy (if applicable)
export SCILINK_API_KEY="your-key"
```

When using `SCILINK_API_KEY`, also provide a `--base-url` pointing to your OpenAI-compatible endpoint.

---

## Quick Start

SciLink can be used via the **CLI**, **web UI**, **MCP server**, or **Python API**.

### CLI

```bash
# Planning session
scilink plan
scilink plan --autonomy supervised --data-dir ./results --knowledge-dir ./papers

# Analysis session
scilink analyze
scilink analyze --data ./sample.tif --metadata ./metadata.json
```

### Web UI

```bash
scilink ui
```

Requires `pip install scilink[ui]`.

### MCP Server

```bash
scilink serve --model gemini-3.1-pro-preview
```

See [MCP Integration](#mcp-integration) for details.

### Python API

```python
from scilink.agents.planning_agents import PlanningAgent
from scilink.agents.exp_agents import AnalysisOrchestratorAgent, AnalysisMode

# Generate an experimental plan
planner = PlanningAgent(model_name="gemini-3.1-pro-preview")
plan = planner.propose_experiments(
    objective="Optimize lithium extraction yield",
    knowledge_paths=["./literature/"],
    primary_data_set={"file_path": "./composition_data.xlsx"}
)

# Analyze microscopy data
analyzer = AnalysisOrchestratorAgent(analysis_mode=AnalysisMode.SUPERVISED)
result = analyzer.chat("Analyze ./stem_image.tif and generate scientific claims")
```

---

![SciLink Reports](misc/report_snapshots.jpg)

---

## MCP Integration

SciLink supports the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) as both a **server** (exposing its tools/agents to external clients like Claude Code) and a **client** (connecting to external MCP servers for additional capabilities).

### As an MCP Server

Expose SciLink's analysis and planning tools to any MCP-compatible client:

```bash
# Default (stdio transport, autonomous mode)
scilink serve --model gemini-3.1-pro-preview

# Analysis only, with human approval for major actions
scilink serve --mode analyze --autonomy co-pilot

# HTTP transport (SSE)
scilink serve --transport sse --host 127.0.0.1 --port 8000
```

The server exposes all orchestrator tools (prefixed `scilink_` for analysis, `scilink_plan_` for planning), plus job management tools for long-running operations. Autonomy modes control which tools require human approval before execution. See [docs/claude_code_integration.md](docs/claude_code_integration.md) for the full MCP server guide.

### As an MCP Client

Connect external MCP servers to extend SciLink with additional tools:

```bash
# Python MCP server (e.g., arXiv paper search)
scilink analyze --mcp stdio:arxiv:python,-m,arxiv_mcp_server,--storage-path,/tmp/papers
```

Programmatically:

```python
orchestrator = AnalysisOrchestratorAgent()
tool_count = orchestrator.connect_mcp_server(
    server_name="arxiv",
    command=["python", "-m", "arxiv_mcp_server", "--storage-path", "/tmp/papers"]
)
```

In the **web UI**, go to the **Tools** tab > **MCP Servers** section, select a transport (stdio/SSE), enter the server name and command, and click **Connect**.

See [docs/mcp_client_integration.md](docs/mcp_client_integration.md) for the full MCP guide.

---

## Extensibility

SciLink supports custom tools, skills, and agents that can be added via CLI flags, the web UI, or programmatically.

### Custom Tools

Provide a Python file with `tool_schemas` (list of OpenAI-format tool dicts) and a `create_tool_functions(data)` factory:

```bash
scilink analyze --tools ./my_image_tools.py
```

### Custom Skills

Add domain-specific analysis guidance via Markdown skill files:

```bash
scilink analyze --skills ./raman_skill.md ./ftir_skill.md
```

Built-in skills are available for curve fitting (XPS, Raman, etc.) and hyperspectral analysis (EELS, etc.).

### Custom Agents

Register additional `BaseAnalysisAgent` subclasses:

```bash
scilink analyze --agents ./my_xrd_agent.py
```

---

# Planning Agents

<img src="misc/scilink_plan.png" alt="SciLink Planning Agent" width="50%">

The Planning Agents module automates experimental design, data analysis, and iterative optimization workflows.

## Architecture

```
PlanningOrchestratorAgent (main coordinator)
├── PlanningAgent (scientific strategy)
│   ├── Dual KnowledgeBase (Docs KB + Code KB)
│   ├── RAG Engine (retrieval-augmented generation)
│   └── Literature Agent (external search)
├── ScalarizerAgent (raw data → scalar metrics)
└── BOAgent (Bayesian optimization)
```

| Agent | Purpose |
|-------|---------|
| **PlanningOrchestratorAgent** | Coordinates the full experimental workflow via natural language |
| **PlanningAgent** | Generates experimental strategies using dual knowledge bases |
| **ScalarizerAgent** | Converts raw data (CSV, Excel) into optimization-ready metrics |
| **BOAgent** | Suggests optimal parameters via Bayesian Optimization |

## CLI Usage

```bash
scilink plan
scilink plan --autonomy supervised --data-dir ./results --knowledge-dir ./papers
scilink plan --model claude-opus-4-5
```

### Interactive Session Example

```
$ scilink plan

📋 What's your research objective?
Your objective: Optimize lithium extraction from brine

👤 You: Generate a plan using papers in ./literature/

🤖 Agent: ⚡ Generating Initial Plan...
    📚 Retrieved 8 document chunks.

🔬 EXPERIMENT 1: pH-Controlled Selective Precipitation
> 🎯 Hypothesis: Adjusting pH to 10-11 will selectively precipitate Mg(OH)₂ while retaining Li⁺

👤 You: Analyze ./results/batch_001.csv and run optimization

🤖 Agent: [calls analyze_file → {"metrics": {"yield": 78.5}}]
  [calls run_optimization → {"recommended_parameters": {"temp": 85.2, "pH": 6.8}}]
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List all available agent tools |
| `/files` | List files in workspace |
| `/state` | Show current agent state |
| `/autonomy [level]` | Show or change autonomy level |
| `/checkpoint` | Save session checkpoint |
| `/quit` | Exit session |

## Python API

```python
from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel
)
from scilink.agents.planning_agents import PlanningAgent, ScalarizerAgent, BOAgent

# Using the orchestrator
orchestrator = PlanningOrchestratorAgent(
    objective="Optimize reaction yield",
    autonomy_level=AutonomyLevel.SUPERVISED,
    data_dir="./experimental_results",
    knowledge_dir="./papers"
)
response = orchestrator.chat("Generate initial plan and analyze batch_001.csv")

# Direct agent usage
agent = PlanningAgent(model_name="gemini-3.1-pro-preview")
plan = agent.propose_experiments(
    objective="Screen precipitation conditions",
    knowledge_paths=["./literature/"],
    primary_data_set={"file_path": "./composition_data.xlsx"}
)

# Bayesian optimization
bo = BOAgent(model_name="gemini-3.1-pro-preview")
result = bo.run_optimization_loop(
    data_path="./optimization_data.csv",
    objective_text="Maximize yield while minimizing cost",
    input_cols=["Temperature", "pH", "Concentration"],
    input_bounds=[[20, 80], [6, 10], [0.1, 2.0]],
    target_cols=["Yield"],
    batch_size=1
)
```

---

# Experimental Analysis Agents

<img src="misc/scilink_analyze.png" alt="SciLink Analysis Agent" width="50%">

The Analysis Agents module provides automated scientific data analysis across multiple modalities.

## Architecture

```
AnalysisOrchestratorAgent (main coordinator)
├── FFTMicroscopyAnalysisAgent (ID: 0)
├── SAMMicroscopyAnalysisAgent (ID: 1)
├── HyperspectralAnalysisAgent (ID: 2)
└── CurveFittingAgent (ID: 3)
```

| ID | Agent | Use Case |
|----|-------|----------|
| 0 | **FFTMicroscopyAnalysisAgent** | Microstructure via FFT/NMF — grains, phases, atomic-resolution |
| 1 | **SAMMicroscopyAnalysisAgent** | Particle segmentation — counting, size distributions |
| 2 | **HyperspectralAnalysisAgent** | Spectroscopic datacubes — EELS-SI, EDS, Raman imaging |
| 3 | **CurveFittingAgent** | 1D fitting — XRD, UV-Vis, PL, DSC, kinetics |

## CLI Usage

```bash
scilink analyze
scilink analyze --data ./sample.tif --metadata ./metadata.json
scilink analyze --mode autonomous --data ./spectrum.npy
```

### Interactive Session Example

```
$ scilink analyze --data ./stem_image.tif

👤 You: Examine my data and suggest an analysis approach

🤖 Agent: ⚡ Examining data at ./stem_image.tif...
  • Type: microscopy, Shape: 2048 x 2048
  • Suggested agents: FFTMicroscopyAnalysisAgent (0) or SAMMicroscopyAnalysisAgent (1)

👤 You: Run the analysis

🤖 Agent: ⚡ Running analysis...
  The HAADF-STEM image reveals MoS2 with predominantly 2H phase structure.
  FFT analysis identified four distinct spatial frequency patterns...
  **Scientific Claims Generated:** 3
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/tools` | List orchestrator tools |
| `/agents` | List analysis agents with descriptions |
| `/status` | Show session state |
| `/mode [level]` | Show or change analysis mode |
| `/schema` | Show metadata JSON schema |
| `/quit` | Exit session |

## Python API

```python
from scilink.agents.exp_agents import (
    AnalysisOrchestratorAgent, AnalysisMode,
    FFTMicroscopyAnalysisAgent, SAMMicroscopyAnalysisAgent,
    HyperspectralAnalysisAgent, CurveFittingAgent
)

# Using the orchestrator
orchestrator = AnalysisOrchestratorAgent(
    base_dir="./my_analysis",
    analysis_mode=AnalysisMode.SUPERVISED
)
response = orchestrator.chat("Examine ./data/sample.tif")

# Direct agent usage
agent = CurveFittingAgent(output_dir="./curve_output", use_literature=True)
result = agent.analyze(
    "pl_spectrum.csv",
    system_info={"experiment": {"technique": "Photoluminescence"}},
    hints="Focus on band-edge emission"
)

# Series with trend analysis
result = agent.analyze(
    ["pl_300K.csv", "pl_350K.csv", "pl_400K.csv"],
    series_metadata={"variable": "temperature", "values": [300, 350, 400], "unit": "K"}
)
```

### Metadata Conversion

```python
from scilink.agents.exp_agents import generate_metadata_json_from_text

# "HAADF-STEM of MoS2 monolayer, 50nm FOV, 300kV"
# → {"experiment_type": "Microscopy", "experiment": {"technique": "HAADF-STEM"}, ...}
metadata = generate_metadata_json_from_text("./experiment_notes.txt")
```

---

## Novelty Assessment

SciLink can automatically check experimental findings against the scientific literature to identify what's genuinely new. This is powered by integration with [FutureHouse](https://www.futurehouse.org/) AI agents.

```
👤 You: Assess novelty of these claims

🤖 Agent: ⚡ Searching literature via FutureHouse...

  📚 [Score 2/5] Mixed 2H/1T phase coexistence → Well-documented
  🤔 [Score 3/5] Sulfur vacancy density of 3.2 × 10¹³ cm⁻² → Similar measurements exist
  🌟 [Score 4/5] 1T phase localized within 5nm of grain boundaries → Limited prior reports

  Summary: 1 HIGH-NOVELTY finding identified
```

The discovery loop: **Analysis** generates scientific claims → **Novelty Assessment** scores each against literature → **Recommendations** prioritize validation experiments for novel findings.

---

## Output Structure

### Planning Session

```
campaign_session/
├── optimization_data.csv      # Accumulated experimental data
├── plan.json                  # Current experimental plan
├── plan.html                  # Rendered plan visualization
├── checkpoint.json            # Session state for restoration
└── output_scripts/            # Generated automation code
```

### Analysis Session

```
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

---

# Simulation Agents *(Coming Soon)*

The Simulation Agents module will provide AI-powered computational modeling, bridging experimental observations with atomistic simulations.

| Agent | Purpose |
|-------|---------|
| **DFTAgent** | Density Functional Theory workflow automation |
| **MDAgent** | Classical molecular dynamics simulations via LAMMPS |
| **SimulationRecommendationAgent** | Recommends structures and simulation objectives based on experimental analysis |

Key planned features include experiment-to-simulation pipelines, defect modeling, and direct integration with the Analysis Agents.

> **Note**: This module is currently being refactored. Check back for updates.
