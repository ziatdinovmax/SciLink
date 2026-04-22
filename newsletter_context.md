# Newsletter Ground Truth — March 2026

## 1. ImageAnalysisAgent (NEW agent added this month)

A brand new agent for automated microscopy image analysis. Supports TEM, SEM, AFM, optical microscopy, and other image types. Runs a two-tier pipeline with quality verification, optional SAM segmentation, and domain-specific analysis skills.

### Python API
```python
from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent

agent = ImageAnalysisAgent(
    model_name="gemini-3.1-pro-preview",
    analysis_depth="auto",          # "basic" (Tier 1 only), "deep" (both tiers), "auto" (LLM decides)
    enable_human_feedback=True,     # prompts user before Tier 2
    output_dir="image_analysis_output",
)

result = agent.analyze(
    data="tem_grain_boundary.tif",  # str path, list of paths, or numpy array
    system_info={"sample": "polycrystalline Al thin film"},
    objective="Grain size distribution and boundary analysis",
    hints="Focus on grain boundaries",
)

print(result["status"])              # "success" or "error"
print(result["detailed_analysis"])   # LLM-generated interpretation
print(result["scientific_claims"])   # ranked list of findings
print(result["extracted_features"])  # measured quantities
```

Input types: single image path (str), list of paths for series, or numpy array (2D/3D/4D).
Supported formats: .npy, .png, .tif, .jpg, .bmp

### Two-tier pipeline
- **Tier 1 (basic):** Segmentation, feature detection, summary statistics using scikit-image/OpenCV/scipy
- **Tier 2 (deep):** Advanced analysis (sublattice separation, strain mapping, phase quantification) — builds on Tier 1 results, does not repeat work
- **auto mode:** LLM evaluates Tier 1 results and recommends whether Tier 2 is needed; user approves if `enable_human_feedback=True`

### SAM integration
Uses Segment Anything Model (SAM) via atomai for particle/object segmentation. Model is cached after first load for batch processing efficiency.

### CLI
```
scilink analyze
```
Then in the interactive chat, describe your image and objective. The orchestrator routes to ImageAnalysisAgent (agent ID 1).

### Streamlit UI
Upload image via the file uploader in the chat area. The analysis_depth is not a separate dropdown — it's controlled programmatically. Agent selection happens automatically based on data type. Results (visualizations, claims, features) appear in the chat.

---

## 2. Directory Database Querying

Lets users query directories of uniformly structured files (e.g., thousands of JSON records from CoREMOF) as a database, with auto-generated Python filtering code.

### Python API
```python
from scilink.agents.planning_agents.planning_orchestrator import PlanningOrchestratorAgent

agent = PlanningOrchestratorAgent(
    base_dir="./session",
    knowledge_dir="./coremof_records/",   # directory with 10+ files of same type
    api_key="...",
    model_name="gemini-3.1-pro-preview",
)
```
The orchestrator auto-detects directories with 10+ files of the same extension (.json, .csv, .xlsx, .tsv) as queryable databases. It generates and executes Python code (pandas for CSV, json module for JSON) to filter and screen records. Screening is done BEFORE generating analysis plans.

### CLI
```bash
scilink plan --knowledge-dir ./coremof_records/ \
    --objective "Find Cu-containing MOFs with pore diameter > 12 angstroms"
```

### Streamlit UI
In planning mode, expand the "Knowledge (papers, images)" section. Either upload knowledge files via the file uploader, or paste a folder path in the "or paste folder path" text input (placeholder: "/path/to/papers/ or /path/to/database/"). The orchestrator then auto-discovers queryable file collections.

---

## 3. MCP Server Integration

SciLink exposes 40+ analysis and planning tools via Model Context Protocol (MCP), so external AI clients can call SciLink capabilities directly.

### Starting the server
```bash
scilink serve --mode analyze --model gemini-3.1-pro-preview
```
Options: `--mode analyze|plan|both`, `--autonomy autonomous|supervised|co-pilot`, `--transport stdio|sse`

### Registering with Claude Code
```bash
claude mcp add scilink -s user \
    -e GEMINI_API_KEY=your-key \
    -- $(which scilink) serve --mode analyze
```

### Registering with Claude Desktop
Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "scilink": {
      "command": "/path/to/scilink",
      "args": ["serve", "--mode", "analyze"],
      "env": {
        "GEMINI_API_KEY": "your-key"
      }
    }
  }
}
```

### Connecting SciLink to external MCP servers
SciLink can also consume tools from other MCP servers:
```bash
scilink analyze --mcp stdio:arxiv:python,-m,arxiv_mcp_server
```

### Streamlit UI
In the "Tools" tab, under "MCP Servers": select transport (stdio/sse), enter server name and command/URL, click "Connect". Connected servers show tool count and a disconnect button.

### Key tools exposed
- `scilink_run_analysis` — run analysis (supports background=true for long jobs)
- `scilink_select_agent` — choose analysis agent
- `scilink_examine_data` — inspect data files
- `scilink_generate_initial_plan` — generate experimental plans
- `scilink_run_optimization` — Bayesian optimization
- `scilink_set_autonomy` — switch autonomy mode at runtime
- Plus 30+ more for metadata, knowledge management, checkpoints, etc.
