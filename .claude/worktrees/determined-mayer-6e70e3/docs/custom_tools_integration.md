# Using SciLink with Your Own Tools

If you already have Python code for a task — a denoiser, a particle detector, a custom morphology metric, a calibration routine — you can plug it into SciLink as a tool the analysis orchestrator can call during a chat session. Your function shows up alongside built-ins like `examine_data` and `run_analysis`, and the LLM will use it when the conversation calls for it.

This guide walks through how to wrap a function as a SciLink tool, how to register it (CLI, UI, or Python), and how its output flows into the built-in agents.

## When to use this

- You have a trusted routine you'd rather run directly than have the LLM regenerate.
- You want a preprocessing step (denoise, flat-field, crop) applied before a built-in agent analyzes the data.
- You have a domain-specific metric the LLM should include in its interpretation.

For heavier extensions — a full analysis agent with its own workflow, or a new skill file — see the `--agents` and `--skills` flags at the end.

## Step 1: wrap your function as a tool

A custom tool file needs two things: a list of tool schemas and a factory function that returns the callables.

```python
# my_tools.py
import numpy as np

def measure_coverage(image: np.ndarray, threshold: float = 0.5) -> dict:
    mask = image > threshold
    return {
        "status": "success",
        "coverage_fraction": float(mask.mean()),
        "n_pixels_above": int(mask.sum()),
    }

# Tool schema — what the LLM sees
tool_schemas = [{
    "type": "function",
    "function": {
        "name": "measure_coverage",
        "description": "Measure surface coverage fraction above an intensity threshold.",
        "parameters": {
            "type": "object",
            "properties": {"threshold": {"type": "number"}},
            "required": [],
        },
    },
}]

# Factory — returns bound callables the orchestrator will invoke
def create_tool_functions(image: np.ndarray) -> dict:
    return {"measure_coverage": lambda threshold=0.5: measure_coverage(image, threshold)}
```

Two things to know about the factory:

- **The first parameter name decides what the orchestrator passes you.** Names like `image`, `data`, or `array` mean "load the current file and pass it as a NumPy array (or a pandas DataFrame for CSV)." Names like `data_path`, `path`, or `file` mean "just pass the file path as a string."
- **Add an `output_dir` keyword argument** if your tool wants to save plots or data files. The orchestrator auto-injects the session's `results/custom_tools/` directory so your outputs land next to the other session results.

Working examples live in `examples/custom_image_tools.py` and `examples/custom_stats_tools.py`.

## Step 2: register your tool

### CLI

```bash
scilink analyze --tools ./my_tools.py

# Multiple files are fine
scilink analyze --tools ./preprocess.py ./morphology.py
```

### Web UI

In the Streamlit UI, open the **Tools & Agents** tab and upload your `.py` file under **Custom Tools**. Registered tools appear with their descriptions and remain available for the session.

### Programmatic

```python
from scilink.agents.exp_agents.analysis_orchestrator import AnalysisOrchestratorAgent
from my_tools import tool_schemas, create_tool_functions

orch = AnalysisOrchestratorAgent(base_dir="./session")
orch.register_tools(tool_schemas, create_tool_functions)

# Drive the conversation by calling orch.chat() with each user turn.
print(orch.chat("load sample.npy and measure its coverage"))
print(orch.chat("now run image analysis on it"))
```

For an interactive REPL, wrap `orch.chat()` in your own input loop — that's exactly what the `scilink analyze` CLI does internally.

## Step 3: use it in a session

Once registered, the LLM knows your tool exists and picks it up naturally when the conversation calls for it. You don't have to name it explicitly, but you can.

```
> load sample.npy, measure its coverage, then run image analysis
```

The orchestrator will:

1. Call `examine_data` on `sample.npy`.
2. Call your `measure_coverage` tool — your function runs, returns a JSON dict.
3. Call `run_analysis` on the image agent, weaving your coverage number into the hints it passes to the agent automatically.

You don't need to copy values between steps. The LLM reads your tool's JSON output from the conversation and composes the handoff on its own.

## Feeding a preprocessed file back into a built-in agent

The example above used your tool for *findings* — numbers and flags that enrich the built-in agent's interpretation. A different case is when your tool produces a **new data array** (denoised, cropped, flat-fielded) and you want the built-in agent to analyze the new version instead of the original.

Built-in agents always read from the orchestrator's current data file, so the trick is to have your tool save the preprocessed array and tell the LLM to switch data files.

```python
# preprocess.py
import numpy as np
from pathlib import Path

def _denoise(image: np.ndarray) -> np.ndarray:
    # ... your real preprocessing
    return image

def preprocess_image(image: np.ndarray, output_dir: str) -> dict:
    cleaned = _denoise(image)
    out_path = Path(output_dir) / "preprocessed.npy"
    np.save(out_path, cleaned)
    return {
        "status": "success",
        "preprocessed_path": str(out_path.absolute()),
        "next_step": (
            f"Call examine_data('{out_path.absolute()}') to switch the "
            "active data file, then run_analysis on the image agent."
        ),
    }

tool_schemas = [{
    "type": "function",
    "function": {
        "name": "preprocess_image",
        "description": (
            "Denoise the current image, save it as a new .npy file, and return "
            "the path. Call examine_data on the returned path to make the "
            "preprocessed file the active dataset."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}]

def create_tool_functions(image: np.ndarray, output_dir: str) -> dict:
    return {"preprocess_image": lambda: preprocess_image(image, output_dir)}
```

Usage:

```bash
scilink analyze --tools ./preprocess.py
```

```
> load afm_scan.npy, preprocess it with my tool, then analyze the cleaned version
```

What happens in order:

1. `examine_data("afm_scan.npy")` — original file becomes the active dataset.
2. `preprocess_image()` — your tool saves `results/custom_tools/preprocessed.npy` and returns the path.
3. `examine_data("…/preprocessed.npy")` — the LLM follows the `next_step` hint and switches the active dataset.
4. `run_analysis(agent_id=<Image>, …)` — the image agent receives the denoised file.

If you'd rather be explicit than trust the hint, just say so in chat: *"now examine and analyze the preprocessed file."*

### Things to know

- Metadata does not automatically transfer when you switch files. If your preprocessing changes something the metadata tracks (pixel size after a crop, energy axis after calibration), return the corrected fields from your tool and either ask the LLM to update the metadata or pass the corrections as hints on the next `run_analysis`.
- Custom tools see only the data (array or path) and `output_dir` — they do not see the orchestrator's metadata dict. If your tool needs metadata, expose the relevant fields as schema parameters so the LLM passes them in explicitly.

## Also available: custom agents and custom skills

Custom tools are the lightest extension. SciLink also supports two heavier ones:

- **`--agents ./my_agent.py`** — register a full analysis agent with its own `analyze()` method. Useful when the workflow is more involved than a single function. See `examples/custom_peak_agent.py` and `examples/custom_outlier_agent.py`.
- **`--skills ./my_skill.md`** — register a markdown skill file that built-in agents can load via the `skill` parameter of `run_analysis`. Useful for encoding domain-specific fitting or analysis recipes as guidance.

All three flags combine in one command:

```bash
scilink analyze --tools ./preprocess.py --agents ./my_agent.py --skills ./raman.md
```

The **Tools & Agents** tab in the web UI supports all three as well.
