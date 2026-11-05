# Foundation agents and skills

Instead of one agent per technique, SciLink ships a small set of **foundation
agents**, one per broad modality (1D curves, images, datacubes, simulation
scales), each specialized at runtime by pluggable **skill bundles**. A skill is
a markdown file of domain guidance, optionally with vetted Python helpers.
Adding a technique (a new spectroscopy, a new simulation engine) is a new
bundle, not a new agent.

## The agents in each mode

### Planning

| Agent | Purpose |
|-------|---------|
| **PlanningOrchestratorAgent** | Coordinates the full experimental workflow via natural language |
| **PlanningAgent** | Generates experimental strategies using dual knowledge bases |
| **ScalarizerAgent** | Converts raw data (CSV, Excel) into optimization-ready metrics |
| **BOAgent** | Suggests optimal parameters via Bayesian Optimization |

### Analysis

| ID | Agent | Use Case |
|----|-------|----------|
| 0 | **CurveFittingAgent** | 1D fitting — XRD, UV-Vis, PL, DSC, TGA, kinetics |
| 1 | **ImageAnalysisAgent** | All image types — microscopy, SEM, TEM, AFM, optical. Handles atomic resolution, grains, particles, textures, defects, morphology |
| 2 | **HyperspectralAnalysisAgent** | Spectroscopic datacubes — EELS-SI, EDS, Raman imaging |

> **Beta:** ImageAnalysisAgent is under active development. Expect rough edges:
> verification scores and planner choices can vary across runs, and some
> domain-specific defaults are still being tuned. Feedback welcome.

### Simulation

> **Under development:** the simulation agents are under active development.
> Expect interfaces and supported engines to change, and check generated
> structures and inputs before running them on HPC.

The interactive surface is `scilink simulate`, backed by the engine-neutral
`SimulationOrchestratorAgent`; the same pipeline is available programmatically
as `run_complete_workflow` (structure → inputs → optional validation → optional
run). The routing decision selects the *scale* (periodic DFT / molecular QC /
classical MD / MLIP) and the engine is supplied by a skill bundle, so no engine
filenames are hardcoded in agent code.

- **Structure generation.** `StructurePipeline` runs the build → validate →
  refine loop for five structure classes (`crystal`, `molecular`, `condensed`,
  `biomolecular`, `aimsgb`), each driven by a skill bundle that supplies the
  build guidance, the output format (POSCAR / xyz / pdb / extxyz) and the
  class-specific validation rubric. `StructurePlanner` maps a free-text request
  onto the right `structure_class` and downstream `simulation_scale`.
- **Periodic DFT.** `PeriodicDFTAgent` is engine-agnostic. **VASP**,
  **Quantum ESPRESSO** and **CP2K** ship under
  `scilink/skills/periodic_dft/`; adding another engine is a markdown bundle,
  no agent code changes.
- **Classical MD and MLIPs.** `MDSimulationAgent` generates classical-MD inputs
  (LAMMPS ships today), with `ForceFieldAgent` assigning the potential and
  `StructurePipeline` packing the condensed-phase box. `MLIPAgent` deploys
  machine-learned potentials as an engine-neutral runner. Force field vs MLIP is
  chosen once the structure exists: an MLIP is a potential source, not a
  separate scale.
- **Post-run analysis.** `SimulationAnalysisAgent` computes properties from a
  finished run's output via verified code generation (scalars and non-scalar
  observables — curves, images, datacubes), engine-agnostic.

## Skills

Bundles live at `scilink/skills/<domain>/<name>/`: a `<name>.md` plus optional
sibling `.py` helpers. Bring your own with `--skills`:

```bash
scilink analyze --skills ./raman_skill.md ./ftir_skill.md
```

Built-in skills cover image analysis (atomic-resolution STEM, …), curve fitting
(XPS, Raman, NMR, …), hyperspectral analysis (EELS, Raman maps, …), and the
simulation engines.

A skill is written under a fixed set of section headings, and each pipeline
stage reads the section it needs. For an analysis skill use this order:

```
## Overview         what the technique is
## Planning         how to plan a use of it
## Implementation   how to write the code (prose, or a code snippet to follow)
## Interpretation   how to read the output
## Validation       how to verify it
```

An optional YAML front-matter `description` is what the orchestrator sees when
it chooses a skill. The `implementation` section can be prose, which the code
generator follows loosely, or a concrete snippet, which it adapts to the data —
the stronger option for most authors. A helper that must run byte-exact
(numerically sensitive or reused everywhere) becomes a sibling `.py` registered
as a `TOOL_SPEC`, visible to the model only while its skill is active; that rung
needs a package contribution, since uploaded skills are markdown-only.

## Custom tools

Provide a Python file with `tool_schemas` (a list of OpenAI-format tool dicts)
and a `create_tool_functions(data)` factory:

```bash
scilink analyze --tools ./my_image_tools.py
```

See [custom_tools_integration.md](custom_tools_integration.md) for how custom
tool outputs flow into the built-in agents and how to feed a preprocessed file
back into the analysis pipeline.

## Custom agents

Register additional `BaseAnalysisAgent` subclasses:

```bash
scilink analyze --agents ./my_xrd_agent.py
```

A package can also advertise agents through the `scilink.agents` entry-point
group, which is discovered at startup.

## Agents that keep what they learn

The chat orchestrators (plan, analyze, simulate) don't just consume skills —
they grow them. During a session the agent records observations into a
session-scoped store; one that looks like a recurring pattern (a failure mode,
a non-obvious parameter choice, a domain rule) can be **graduated** into a
markdown skill that is auto-loaded on every later run (`graduate_to_skill`, in
`analyze` and `plan`). In autopilot and autonomous modes the agent decides when
an observation is worth graduating; in co-pilot it asks first.

Graduated and auto-distilled skills are stored under **`~/.scilink/`**
(override with `$SCILINK_HOME`), outside the installed package, so they survive
a `pip` upgrade. The store is **off by default**: run `scilink memory enable`
(or use the Memory panel) before anything is banked, staged or loaded.

```bash
scilink memory list                 # graduated/auto-distilled skills
scilink memory staged               # raw T=2 solutions awaiting distillation
scilink memory upgrade <domain>/<id> --into <domain>/<name>   # enrich an existing skill
scilink memory consolidate <domain>/<technique>              # distill N into a new skill
scilink memory promote <domain>/<name>                       # make a provisional skill auto-routable
```

> **Docker:** the store lives in the container's home (`~/.scilink`), which is
> ephemeral — learned skills are lost when the container exits unless you mount
> a volume. The image declares it as a `VOLUME`; persist it with, e.g.:
>
> ```bash
> docker run -v ~/.scilink:/home/scilinkuser/.scilink scilink ...
> # or: docker run -e SCILINK_HOME=/data -v scilink-mem:/data scilink ...
> ```
>
> Without a mount, a one-time warning says memory won't persist.

See [cli_reference.md](cli_reference.md) for the full `memory` command set.
