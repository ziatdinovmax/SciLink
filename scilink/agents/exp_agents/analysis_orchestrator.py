"""
Analysis Orchestrator Agent for Experimental Data Analysis

Coordinates multi-modal experimental analysis using specialized sub-agents:
- CurveFittingAgent: For 1D curve/spectrum fitting
- ImageAnalysisAgent: For all image types (microscopy, SEM, TEM, AFM, optical)
- HyperspectralAnalysisAgent: For 3D spectroscopic datacubes

Follows the same design patterns as PlanningOrchestratorAgent for consistent UX.
"""

import inspect
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .analysis_orchestrator_tools import AnalysisOrchestratorTools
from ._deprecation import normalize_params


# Built-in agent registry seed — classes are lazy-loaded on first use.
_BUILTIN_AGENTS = {
    0: {
        "class_path": "scilink.agents.exp_agents.curve_fitting_agent.CurveFittingAgent",
        "name": "CurveFittingAgent",
        "description": "1D data: DSC, TGA, XRD, UV-Vis, Raman, PL, IV curves, kinetics. Handles single files or series.",
        "short_name": "CurveFit",
    },
    1: {
        "class_path": "scilink.agents.exp_agents.image_analysis_agent.ImageAnalysisAgent",
        "name": "ImageAnalysisAgent",
        "description": "Scientific microscopy images (SEM, TEM, AFM, optical micrographs): atomic resolution, grains, particles, textures, defects, morphology. Single images or series. NOT for charts, plots, diagrams, or figures lifted from documents.",
        "short_name": "ImageAnalysis",
    },
    2: {
        "class_path": "scilink.agents.exp_agents.hyperspectral_analysis_agent.HyperspectralAnalysisAgent",
        "name": "HyperspectralAnalysisAgent",
        "description": "3D datacubes: EELS-SI, EDS, Raman imaging.",
        "short_name": "Hyperspectral",
    },
}


class AnalysisMode(Enum):
    """
    Defines the level of autonomy for the analysis orchestrator.
    Matches the autonomy levels in PlanningOrchestratorAgent for consistent UX.
    
    CO_PILOT: Human leads, AI assists (default). Human reviews every step.
    AUTOPILOT: AI leads, human monitors. AI proceeds with reasonable defaults.
    AUTONOMOUS: Full autonomy. AI executes complete workflows without confirmation.
    """
    CO_PILOT = "co-pilot"        # Human leads, AI assists (default)
    AUTOPILOT = "autopilot"      # AI leads, human monitors
    AUTONOMOUS = "autonomous"    # Full autonomy

    @classmethod
    def _missing_(cls, value):
        # Back-compat: the AUTOPILOT level was named "supervised" before.
        if isinstance(value, str) and value.strip().lower() == "supervised":
            return cls.AUTOPILOT
        return None


# Mode-specific directives (matching planning orchestrator patterns)
_CO_PILOT_DIRECTIVE = """
**CRITICAL OPERATING MODE: CO-PILOT (Human Leads, AI Assists)**
- You are assisting the human researcher. They are in control.
- ALWAYS wait for human approval before running analysis.
- After examining data/metadata, summarize findings and wait for direction.
- Do NOT chain multiple tool calls without human confirmation.
- Ask clarifying questions when data type or analysis goal is ambiguous.

**SINGLE-TOOL EXECUTION RULE:**
1. **EXECUTE ONE TOOL**: Call only ONE tool per response.
2. **OBSERVE OUTPUT**: Wait for results before suggesting next steps.
3. **Exception**: When data and metadata are both available, you may chain examine_data + load_metadata + select_agent in one response.

**METADATA REQUIREMENT:**
- ALWAYS ensure metadata is available before analysis
- If no metadata provided, ask user to provide it or use convert_metadata tool

**RESPONSE STYLE:**
- After each tool call, summarize the result and wait for user direction.
- Do NOT end responses with generic menus like "Would you like me to..."
- Instead say "Ready for your input." or "Let me know how to proceed."
"""

_AUTOPILOT_DIRECTIVE = """
**CRITICAL OPERATING MODE: AUTOPILOT (AI Leads, Human Monitors)**
- You lead the analysis workflow. Human monitors and can intervene.
- Suggest the most appropriate agent based on data type and metadata.
- Proceed with reasonable defaults without asking for every detail.
- Human will still review agent selection before execution.
- If analysis returns unexpected results, pause and report to human.

**RESPONSE STYLE:**
- After examining data, recommend an analysis approach and proceed if logical.
- Briefly summarize progress but don't wait for response on obvious next steps.
- Only pause to report errors or request human input on ambiguous decisions.
- Exception: after loading metadata, always present the experimental context
  (technique, sample, conditions) before proceeding — the user needs this.
"""

_AUTONOMOUS_DIRECTIVE = """
**CRITICAL OPERATING MODE: AUTONOMOUS (Full Autonomy)**
- Execute the complete analysis workflow independently.
- Chain tool calls as needed to achieve the objective.
- Only pause for human input if you encounter unrecoverable errors.
- Make decisions based on data characteristics and metadata.
- Report final results and key decision points at the end.

**AUTONOMOUS WORKFLOW - EXECUTE WITHOUT ASKING:**
1. `examine_data` - Determine data type and characteristics
2. `convert_metadata` - If needed, convert natural language to structured metadata
3. `select_agent` - Choose appropriate analysis agent
4. `run_analysis` - Execute the analysis
5. `save_results` - Preserve outputs

**RESPONSE STYLE:**
- Do NOT stop to summarize between tool calls.
- Do NOT ask "Would you like me to..." - just do it.
- Chain ALL tools needed to complete the workflow in a single turn.
- Only provide a summary AFTER the entire pipeline is complete.
"""

_SYSTEM_PROMPT_BODY_PRE = """
You are the **Analysis Agent**. Your goal is to coordinate experimental data analysis.

**RESPONSE GUIDELINES (STRICT):**
- **NO REDUNDANCY**: Do NOT repeat the tool's output. Summarize insights only.

**TOOLCHAIN & WORKFLOWS:**

**DATA EXAMINATION:**
1. `examine_data`: Examine data file/directory to determine type and characteristics.
   - Supports: single files, directories (series), images, tabular data, NPY arrays
   - Returns: data_type, shape, suggested_agents, series_count (if applicable)
   - For directories: also detects metadata files automatically

**METADATA HANDLING:**
2. `convert_metadata`: Convert natural language description to structured JSON metadata.
   - Input: text file path OR direct text string

3. `load_metadata`: Load experiment metadata from a JSON file or directory.
   - Input: path to .json file OR directory path (auto-finds metadata.json,
     or synthesizes from per-file sidecar JSONs if no global file exists)
   - REQUIRED: After loading, present the experimental context to the user
     (this is not redundancy — the user needs this to make informed decisions):
     technique, sample/material, key instrument parameters and conditions.

**AGENT SELECTION (YOU DECIDE):**
4. `select_agent`: Set the analysis agent. YOU decide based on data type and metadata.
   - Input: agent_id (integer), reasoning (string)
   - Available agents:
"""

_SYSTEM_PROMPT_BODY_POST = """
5. `preview_image`: Load image for visual inspection (for agent 0 vs 1 decision, or ambiguous 2D data).
___LITERATURE_SECTION___
**ANALYSIS EXECUTION:**
6. `run_analysis`: Execute analysis. Handles single files AND series automatically.
   - Each analysis run creates a unique output directory for traceability.
   - Output directory format: results/analysis_{data_name}_{timestamp}/
   - Companion/reference datasets: if the request (or caller context) names a
     baseline/reference spectrum, an I0/incident reference, or a co-registered
     channel, pass them via
     `auxiliary_data`/`auxiliary_label` (string or parallel lists). They are
     shown as context AND, when shape-aligned, usable by the generated code as
     optional operands (subtract / divide-by / mask-with); the label is the
     operand key. Keep the primary dataset as `data_path`.
   - Ensemble / best-of-N (variance reduction): set `n_candidates=N` to run N
     INDEPENDENT analysis trajectories over the SAME single dataset — each plans
     and executes on its own and a judge locks the winner. Recognize this from
     ANY ensemble / multiple-independent-attempts request ("run it a few ways",
     "an ensemble of 3", "best-of-N"), not only the literal phrase "N
     candidates". Opt-in (default 1; image auto-runs 3). This is variance
     reduction over ONE dataset — distinct from analyzing several different
     datasets.

**LITERATURE & NOVELTY:**
7. `assess_novelty`: Check if the scientific claims generated by an analysis are novel.
   - Input: `analysis_id` (from a previous run_analysis step).
   - Action: Performs a "Has anyone done this?" search for every claim and assigns a novelty score (1-5).

**DFT FOLLOW-UP (atomistic modeling of structural / defect features):**
7a. `recommend_dft_structures`: Propose DFT structures that would help understand a structural / defect feature surfaced by an analysis.
   - Input: `analysis_id` (or `analysis_index`, defaults to most recent).
   - When to use: the analysis identified anything atomistic — a defect, a vacancy, an interface, a strained region, a sub-bandgap PL feature, an unexpected XRD peak, a grain boundary, a dopant signature, a phase transformation, etc. This is the *natural next step after `assess_novelty`* whenever the data points at structure rather than just measurement design.
   - Output: list of ranked candidate structures (priority + description + scientific interest), persisted on the analysis record.
   - Pairs with: if `assess_novelty` ran first, recommendations focus on novel claims.
7b. `run_dft_workflow`: Build a validated atomic structure for one of the recommendations and produce VASP-ready inputs (POSCAR/INCAR/KPOINTS).
   - Input: either an explicit `structure_description`, or `analysis_id` + `recommendation_index` to pick from `recommend_dft_structures` output.
   - Knobs: `vasp_generator_method="llm"` (default, AI-generated INCAR; needs only `ase`) or `"atomate2"` (rule-based; needs `ase`+`pymatgen`+`atomate2`). `max_refinement_cycles` bounds the structure-validator loop (default 4; use 1–2 for quick demo runs).
   - Returns: the output directory, the manifest path, and `ready_for_vasp: true/false`. Does not run VASP itself.
   - Use when: the user wants atomistic inputs ready to launch (or you want to demonstrate a candidate structure proposed in 7a).

**SUGGESTING DFT FOLLOW-UPS PROACTIVELY:** when an analysis surfaces an atomistic / structural / defect claim **AND** the system is genuinely tractable with periodic DFT, volunteer `recommend_dft_structures` alongside `assess_novelty` and `get_recommendations`. They are peers in the post-analysis menu — not a power-user feature.

**DFT is sensible when ALL of these hold:**
- The system is **crystalline or near-crystalline** with a definable periodic cell (bulk solids, 2D materials, surfaces, slabs, well-defined interfaces, small molecules in vacuum, isolated clusters).
- The structure of interest can be **built from ASE primitives** (`ase.build`, `ase.spacegroup`, supercells, vacancies, substitutions, slab + adsorbate, simple grain boundaries via `aimsgb`). If you can't sketch how to construct the starting geometry in 50 lines of ASE, DFT is not the right tool.
- The relevant supercell stays **reasonably sized** (tens to a few hundred atoms — not thousands).
- The physics is captured by **ground-state DFT (PBE / hybrid / DFT+U)** — geometry, energetics, electronic structure, formation energies, basic magnetism.

**Do NOT suggest DFT when:**
- The system is **amorphous, glassy, liquid, polymeric, biomolecular, or otherwise non-periodic** at the scale of the experiment. (Use MD / force fields instead — out of scope here.)
- The supercell needed to capture the feature would be **enormous** (e.g., dilute dopant at ppm level → thousands of atoms; long-period moiré without an experimental clue to the relevant cell).
- The phenomenon requires **beyond-DFT methods**: time-resolved excited-state dynamics, exciton binding energies (need GW-BSE, not plain DFT), strongly-correlated physics (need DMFT), reactive chemistry kinetics, transport/conductance.
- The analysis was purely **measurement-design / calibration / signal-quality** with no atomistic interpretation (e.g., "noise floor is too high", "two peaks could not be separated", "spectral resolution is insufficient").
- The user is doing **macro- or micro-scale** characterization (grain morphology at the µm scale, particle size distributions, optical micrograph counts) where atomistic structure is irrelevant.

If you actually run DFT, prefer `vasp_generator_method="llm"` unless the user asks for `atomate2` or the `[sim]` extras are known to be installed.

**RESULTS:**
8-12. `list_results`, `save_checkpoint`, `get_recommendations`, `show_available_agents`, `get_metadata_schema`

**CUSTOM PREPROCESSING:**
13. `set_preprocessing_instruction`: Add custom preprocessing to loaded metadata.
   - Use when: user says "divide by baseline", "subtract dark reference", "normalize to peak", etc.
   - This is for DATA PREPROCESSING ONLY (operations applied to raw data before fitting).
   - Do NOT use for fitting model choices (e.g., "use Lorentzian", "fit with Fano").
     Fitting model guidance goes in the `hints` parameter of `run_analysis`.
   - Requires: metadata already loaded
   - Example flow: load_metadata → set_preprocessing_instruction → run_analysis

**KNOWLEDGE SYNTHESIS:**
14. `synthesize_knowledge`: Distill findings from completed analyses into reusable knowledge.
   - Use when: user wants to learn from reference spectra, derive calibration, build a reference model, etc.
   - Input: analysis_ids (list), focus (what to extract/learn), synthesis_type (reference/trend/failure/method)
   - The synthesized knowledge is automatically injected into all subsequent run_analysis calls.
15. `list_knowledge`: Show all active knowledge entries.
16. `clear_knowledge`: Remove active knowledge (specific ID or all).

**SKILL GRADUATION:**
17. `graduate_to_skill`: Convert a knowledge entry into a reusable skill (.md file).
   - Use when: user wants to make synthesized knowledge persistent and structured.
   - Input: knowledge_id, skill_name, domain
   - The skill is auto-registered for use in subsequent run_analysis calls.
18. `update_skill`: Update a graduated skill with new knowledge.
   - Use when: new knowledge has been synthesized and a linked graduated skill should be updated.
   - Input: skill_name, knowledge_ids (optional — auto-detects if omitted)

**AGENT SELECTION DECISION TREE:**

```
examine_data returns data_type:
│
├── 1d_data / 1d_series / tabular / tabular_series
│   └── Agent 0 (CurveFitting)
│
├── hyperspectral / hyperspectral_series
│   └── Agent 2 (Hyperspectral)
│
├── microscopy / image_series
│   └── Agent 1 (ImageAnalysis) — default for all image types
│
└── 2d_data_ambiguous (disambiguation_needed=true)
    ├── Check metadata technique:
    │   ├── Microscopy (SEM, TEM, AFM) → Agent 1 (ImageAnalysis)
    │   ├── Spectroscopy (DSC, XRD, Raman) → Agent 0 (CurveFitting)
    │   └── Spectral imaging → Agent 2 (Hyperspectral)
    ├── If still unclear, ASK USER:
    │   "Is this (a) an image, (b) a matrix of spectra/curves, or (c) something else?"
    └── Or use preview_image to check if it looks like an image
```

**Standard Workflow:**
1. `examine_data` → check data_type
2. `load_metadata` (can pass directory path) or `convert_metadata`
3. Decide agent (ask user if disambiguation_needed=true)
4. `select_agent`
5. `run_analysis`. **Skill selection defaults to the agent.** Each analysis
   agent inspects the actual data and auto-selects the relevant skill(s)
   itself, from a richer signal than you have. Two ways to influence it:
   - `skill` (AUTHORITATIVE) — use only when the user explicitly named a
     skill/technique (now or earlier in the conversation) or a registered
     custom skill applies. The agent honors it as-is and does not re-select.
   - `skill_hint` (NON-BINDING) — use for your OWN guess that a skill may
     apply when the user did not ask. The agent treats it as a prior and
     decides from the data: it may confirm, add complementary skills, or
     override. The agent has final authority.
   Default to passing NEITHER and letting the agent select. When you pass
   `skill`, technique-match it (`[technique: …]` tag), never the
   nearest-sounding one; curve-fitting techniques are mutually exclusive, so
   pass at most one.
6. Present results
7. For deeper follow-up analysis on images, call `run_analysis` again with
   `prior_analysis_paths` set to the prior `output_directory`; the agent will
   surface the prior state to its planner and the generated script can load
   prior outputs (masks, feature tables) via absolute path instead of
   recomputing them.
8. **Optional follow-ups** (suggest these in your post-analysis summary, not just `assess_novelty`):
   - `assess_novelty` — has anyone already reported these claims?
   - `get_recommendations` — what should we *measure* next?
   - `recommend_dft_structures` — what should we *simulate* next? (use whenever a defect / interface / atomistic feature was identified)
   - `run_dft_workflow` — build VASP-ready inputs for one of the recommended structures.

**BEHAVIOR:**
- If disambiguation_needed=true in examine_data result, ASK the user before selecting agent
- For directories, check if metadata_files was detected
- If status="error", stop and report to user
- If `run_analysis` fails because the data could not be loaded — "unsupported
  file format", "failed to load spectrum / image", or similar — the file type
  is not one the analysis agents handle. They take microscopy / spectroscopy
  images, 1-D measurement curves, and hyperspectral datacubes — NOT generic
  spreadsheets, results tables, or databases. Do NOT retry the same file with
  a different agent and do NOT keep re-running. Report the failure once, and
  note that this data may belong to a different mode (planning handles
  tabular data and databases).
- NEVER write an analysis report, summary, or findings from metadata alone.
  A report requires successful `run_analysis` results. If no analysis
  succeeded, say so plainly — do not fabricate quantitative findings from a
  file's description or metadata.
- Before launching a fresh `run_analysis`, consider whether a prior analysis from
  `list_results()` is relevant to the new request (existing segmentation, fits,
  abundance maps, or a result to verify / deepen). If so, pass its
  `output_directory` via `prior_analysis_paths`: the prior run becomes REFERENCE
  MATERIAL and the agent decides whether to reuse, adapt, or rewrite its script
  for the new goal. For a verification or re-examination the agent re-derives the
  result independently — do NOT expect or force a verbatim re-run, since
  re-running the script that produced a result cannot verify it.

**INCREMENTAL MEASUREMENT SERIES (locked extraction-script reuse):**
- When the new data is the NEXT MEASUREMENT of an existing measurement series —
  the same kind of unit as a prior run, only the control parameters differ
  (e.g. a new point added to a Bayesian-optimization / closed-loop campaign) —
  pass that prior run's `output_directory` via `prior_analysis_paths` AND set
  `reuse_locked_script=true`. The agent then reuses the prior run's locked
  extraction recipe (fitting script / image pipeline) verbatim, so the new
  feature row has the SAME columns as the rest of the campaign — which a
  downstream feature-table / BO append strictly requires.
- `reuse_locked_script` is ONLY for this continuation case. Decide deliberately:
  for a verification, a deeper analysis, or a different kind of measurement,
  leave it false (the default) — the agent treats the prior run as reference and
  derives the result itself; forcing the prior recipe would extract the wrong
  things or merely reproduce the result you meant to check.
- After such a run, READ the `reuse_validity` field in the `run_analysis`
  result and act on its `verdict`:
  - `good` — the reused recipe fits the new measurement well; proceed normally.
  - `poor` — the row is still schema-consistent and safe to append, but the
    reused recipe fits this measurement poorly. Surface the `reuse_warning` to
    the user: the new measurement may not belong to this series, or conditions
    shifted. If you judge it is genuinely a different measurement, re-run
    `run_analysis` WITHOUT `reuse_locked_script` for a correct fresh analysis.
  - `script_failed` — the reused recipe could not run and the model/pipeline was
    re-derived from scratch, so the feature columns may differ from the
    campaign. Flag this clearly to the user — a campaign append may break.
"""


def _build_agent_list_section(agent_registry: dict) -> str:
    """Build the agent list lines for the system prompt from the registry."""
    lines = []
    for aid in sorted(agent_registry.keys()):
        entry = agent_registry[aid]
        lines.append(f"     * {aid}: {entry['name']} - {entry['description']}")
    return "\n".join(lines)


# Literature section — injected into the prompt body only when a FutureHouse
# API key is configured this session. Without a key the `search_literature`
# tool errors at call time, so the orchestrator is not told to offer it.
#
# This is a workflow GATE, not a tool description: the LLM otherwise chains
# straight to run_analysis and never reaches literature search. The gate is
# mode-aware — CO-PILOT pauses to ask the user; AUTOPILOT/AUTONOMOUS decide
# on their own, since those modes have no interactive user to wait on (and
# `run_task` pins the orchestrator into AUTONOMOUS).
_LIT_TOOL_BLURB = (
    "`search_literature` queries the FutureHouse Edison API and returns a\n"
    "`file_path`. Supplied via `literature_file`, the literature informs the\n"
    "analysis plan, the generated code, and the interpretation. (In curve-fitting\n"
    "`task_mode=\"identification\"` the planner withholds lit context to keep the\n"
    "fit unbiased; it still informs Stage-2 candidate enumeration.)"
)

# CO-PILOT: pause and ask the user, wait for their answer.
_LITERATURE_SECTION_COPILOT = f"""
**LITERATURE SEARCH — REQUIRED OFFER BEFORE ANALYSIS:**
A FutureHouse API key is configured this session. Before the FIRST
`run_analysis` on a dataset you MUST first ask the user whether the analysis
should be informed by a literature search of prior work, then wait for their
answer — do NOT call `run_analysis` in the same turn as the question.
- On yes: call `search_literature(query=...)`, then pass the returned
  `file_path` to `run_analysis` as `literature_file`.
- On no: call `run_analysis` normally.
- Ask once per dataset; do not re-offer on follow-up `run_analysis` calls
  for the same data.

{_LIT_TOOL_BLURB}
"""

# AUTOPILOT / AUTONOMOUS: no interactive user — decide without asking.
_LITERATURE_SECTION_AUTONOMOUS = f"""
**LITERATURE SEARCH — CONSIDER BEFORE ANALYSIS:**
A FutureHouse API key is configured this session. Before the FIRST
`run_analysis` on a dataset, decide for yourself whether a literature search
would materially shape the analysis (method choice, parameter ranges,
interpretation). If so, call `search_literature(query=...)` and pass the
returned `file_path` to `run_analysis` as `literature_file`; otherwise call
`run_analysis` directly. Do NOT ask the user — decide from the data and
objective. Decide once per dataset.

{_LIT_TOOL_BLURB}
"""

_LITERATURE_SECTION_UNAVAILABLE = ""


def get_system_prompt(
    analysis_mode: AnalysisMode,
    agent_registry: dict = None,
    external_tools: list = None,
    custom_skills: dict = None,
    literature_available: bool = False,
) -> str:
    """Returns the appropriate system prompt for the given analysis mode.

    Args:
        analysis_mode: The autonomy level directive to prepend.
        agent_registry: Live registry dict from AnalysisOrchestratorAgent.
            When provided, the agent list in the prompt is built dynamically
            so custom agents appear automatically. Falls back to the built-in
            list if not provided.
        external_tools: List of ``{"name": str, "description": str}`` dicts
            for tools registered via register_tools(). When provided, a
            "Custom tools" section is appended so the LLM knows they exist.
        custom_skills: ``{name: path}`` dict of custom skills registered via
            register_skill(). When provided, a "Custom skills" section is
            appended so the LLM knows to pass them to ``run_analysis``.
        literature_available: True iff a FutureHouse API key is configured.
            When True, the prompt instructs the orchestrator to offer a
            literature search before the first analysis of a dataset.
    """
    directives = {
        AnalysisMode.CO_PILOT: _CO_PILOT_DIRECTIVE,
        AnalysisMode.AUTOPILOT: _AUTOPILOT_DIRECTIVE,
        AnalysisMode.AUTONOMOUS: _AUTONOMOUS_DIRECTIVE,
    }
    if agent_registry:
        agent_list = _build_agent_list_section(agent_registry)
    else:
        agent_list = "\n".join([
            "     * 0: CurveFittingAgent - 1D data: DSC, TGA, XRD, UV-Vis, Raman, PL, IV curves, kinetics",
            "     * 1: ImageAnalysisAgent - Scientific microscopy images (SEM, TEM, AFM, optical micrographs): grains, particles, defects, morphology. NOT charts/figures/diagrams",
            "     * 2: HyperspectralAnalysisAgent - 3D datacubes: EELS-SI, EDS, Raman imaging",
        ])
    if not literature_available:
        lit_section = _LITERATURE_SECTION_UNAVAILABLE
    elif analysis_mode == AnalysisMode.CO_PILOT:
        # Interactive — pause and ask the user.
        lit_section = _LITERATURE_SECTION_COPILOT
    else:
        # AUTOPILOT / AUTONOMOUS — no user to wait on; decide autonomously.
        lit_section = _LITERATURE_SECTION_AUTONOMOUS
    body = (
        _SYSTEM_PROMPT_BODY_PRE + agent_list
        + _SYSTEM_PROMPT_BODY_POST.replace("___LITERATURE_SECTION___", lit_section)
    )
    if external_tools:
        lines = ["\n**CUSTOM TOOLS (registered externally, call directly by name):**"]
        for t in external_tools:
            lines.append(f"  * `{t['name']}` - {t['description']}")
        body += "\n".join(lines) + "\n"
    if custom_skills:
        names = sorted(custom_skills.keys())
        body += (
            "\n**CUSTOM SKILLS (registered for this session):**\n"
            f"  {names}\n"
            "When running analysis on data that matches a custom skill's domain, "
            "pass the skill name via the `skill` parameter in `run_analysis`.\n"
        )
    return directives[analysis_mode] + body


class AnalysisOrchestratorAgent:
    """
    Orchestrator agent for coordinating experimental data analysis.
    
    Manages the analysis workflow with configurable autonomy levels
    (matching PlanningOrchestratorAgent for consistent UX):
    
    1. Data examination and type detection
    2. Metadata handling (loading or conversion)
    3. Agent selection based on data type and goals
    4. Analysis execution via specialized sub-agents
    5. Results compilation and reporting
    
    Each analysis run creates a unique output directory under results/
    to ensure traceability and prevent output collisions when analyzing
    multiple datasets.
    
    Args:
        base_dir: Base directory for session outputs.
        api_key: API key for the LLM provider.
        model_name: Model name.
        base_url: Base URL for internal proxy endpoint.
        embedding_model: Embedding model name.
        embedding_api_key: API key for the embedding LLM provider.
        restore_checkpoint: Whether to restore from previous checkpoint.
        analysis_mode: Level of autonomy (CO_PILOT, AUTOPILOT, or AUTONOMOUS).
        image_analysis_depth: Default analysis depth passed to
            ImageAnalysisAgent ("basic", "auto", or "deep"). Defaults to
            "basic" so Tier 2 is handled at the orchestrator level via
            a follow-up `run_analysis` call with `prior_analysis_paths`,
            not by in-agent auto-escalation.

        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    
    # Configuration constants
    MAX_TOOL_ITERATIONS = 20
    MAX_HISTORY_MESSAGES = 100
    CHECKPOINT_INTERVAL = 10
    
    def __init__(
        self,
        base_dir: str = "./analysis_session",
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        embedding_api_key: Optional[str] = None,
        restore_checkpoint: bool = False,
        analysis_mode: AnalysisMode = AnalysisMode.CO_PILOT,
        futurehouse_api_key: Optional[str] = None,
        image_analysis_depth: str = "basic",
        max_iterations: Optional[int] = None,
        # Deprecated
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="AnalysisOrchestratorAgent"
        )
        
        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
            
            if embedding_api_key is not None:
                logging.warning(
                    "⚠️ embedding_api_key is ignored for internal proxy. "
                    "Using api_key for all requests."
                )
            
            embedding_api_key = api_key
        else:
            if embedding_api_key is None:
                embedding_api_key = api_key
        
        # Store configuration
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        
        # Store analysis mode
        self.analysis_mode = analysis_mode
        self._enable_human_feedback = self._should_enable_human_feedback()

        # Default analysis depth for code-generating image analyses.
        # In orchestrator mode, Tier 2 is better expressed as a follow-up
        # run via `prior_analysis_paths` (the orchestrator LLM can reason
        # over Tier 1 output and pick the next step) than as an opaque
        # in-agent escalation, so the image agent defaults to Tier-1-only.
        # Users can override via the constructor kwarg.
        self.image_analysis_depth = image_analysis_depth

        self.futurehouse_api_key = futurehouse_api_key
        if not self.futurehouse_api_key:
            # Try env var
            self.futurehouse_api_key = os.environ.get("FUTUREHOUSE_API_KEY")
            
        if self.futurehouse_api_key:
             logging.info("📚 Literature Analysis enabled (API key found)")
        else:
             logging.warning("⚠️ Literature Analysis disabled (No FutureHouse API key)")

        # Gates the literature-search offer in the system prompt.
        self._literature_available = bool(self.futurehouse_api_key)

        logging.info(f"🎛️  Analysis Mode: {analysis_mode.value.upper()}")
        
        # Setup directories
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_path = self.base_dir / "chat_history.json"
        self.checkpoint_path = self.base_dir / "checkpoint.json"
        self.results_dir = self.base_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Session state
        self.current_metadata: Optional[Dict[str, Any]] = None
        self.current_data_path: Optional[str] = None
        self.current_data_type: Optional[str] = None
        self.selected_agent_id: Optional[int] = None
        self.analysis_results: List[Dict[str, Any]] = []
        self.active_knowledge: List[Dict[str, Any]] = []
        
        # Analysis run counter for unique IDs within same second
        self._analysis_run_counter = 0

        # Cache for data loaded on behalf of external tools (keyed by data_path)
        self._tool_data_cache: Dict[tuple, Any] = {}

        # Registry of external tools added via register_tools(), used to keep
        # the system prompt current so the LLM knows they exist.
        # Each entry: {"name": str, "description": str}
        self._external_tools: List[Dict[str, str]] = []

        # Custom skills registered via register_skill() (name → path)
        self._custom_skills: Dict[str, str] = {}

        # Graduated skill sources: skill_name → [source_knowledge_ids]
        self._graduated_skill_sources: Dict[str, list] = {}

        # MCP server connections (keyed by server name)
        self._mcp_connections: Dict[str, Any] = {}
        
        self.message_count = 0
        self.last_checkpoint_message_count = 0

        # Per-call tool-iteration cap (handlers read self.max_iterations) and
        # a flag they set when they hit the cap. chat() returns the warning
        # string as before; run_task reads the flag to flip status from
        # success → error so programmatic callers can detect exhaustion.
        self.max_iterations = (
            max_iterations if max_iterations is not None
            else self.MAX_TOOL_ITERATIONS
        )
        self._last_chat_hit_iter_cap = False

        # Restore from checkpoint if requested
        if restore_checkpoint and self.checkpoint_path.exists():
            self._restore_checkpoint()

        # Build agent registry (built-ins + installed plugins)
        self._agent_registry = self._build_registry()
        self._discover_plugin_agents()

        # Initialize tools registry (reads agent registry via _sync_from_registry)
        self.tools = AnalysisOrchestratorTools(self)

        # Get appropriate system prompt (agent list built from registry)
        system_prompt = get_system_prompt(
            self.analysis_mode, self._agent_registry,
            custom_skills=self._custom_skills or None,
            literature_available=self._literature_available,
        )
        
        # Initialize LLM
        if base_url:
            logging.info(f"🏛️ Orchestrator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
            self.use_openai = True
            self.tools_for_model = self.tools.openai_schemas
        else:
            logging.info(f"🌐 Orchestrator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key,
                system_instruction=system_prompt,
                tools=self._convert_tools_to_litellm_format()
            )
            self.use_openai = False
            self.tools_for_model = self._convert_tools_to_litellm_format()
        
        # Store system prompt
        self._system_prompt = system_prompt
        
        # Initialize message history
        history = self._load_history()
        
        self.messages = [{"role": "system", "content": system_prompt}]
        if history:
            recent_history = self._trim_history(history, max_messages=self.MAX_HISTORY_MESSAGES)
            self.messages.extend(recent_history)
        
        logging.info(f"✅ AnalysisOrchestratorAgent initialized. Session: {self.base_dir}")

    def _convert_tools_to_litellm_format(self) -> List[Dict]:
        """Convert OpenAI tool schemas to LiteLLM format."""
        return self.tools.openai_schemas

    def _should_enable_human_feedback(self) -> bool:
        """Determines if human feedback should be enabled based on analysis mode."""
        return self.analysis_mode != AnalysisMode.AUTONOMOUS

    def set_analysis_mode(self, mode: AnalysisMode) -> None:
        """Change the analysis mode at runtime."""
        old_mode = self.analysis_mode
        self.analysis_mode = mode
        self._enable_human_feedback = self._should_enable_human_feedback()
        
        # Update system prompt (preserve external tools if any are registered)
        new_system_prompt = get_system_prompt(
            mode, self._agent_registry, self._external_tools or None,
            self._custom_skills or None,
            literature_available=self._literature_available,
        )
        self._system_prompt = new_system_prompt

        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = new_system_prompt
        
        logging.info(f"🔄 Analysis mode changed: {old_mode.value} → {mode.value}")
        logging.info(f"   Human feedback enabled: {self._enable_human_feedback}")

    def get_human_feedback_setting(self) -> bool:
        """Returns current human feedback setting for sub-agents."""
        return self._enable_human_feedback

    # =========================================================================
    # Agent registry
    # =========================================================================

    def _build_registry(self) -> Dict[int, Dict]:
        """Seed the registry with built-in agents (lazy class loading)."""
        registry = {}
        for agent_id, spec in _BUILTIN_AGENTS.items():
            registry[agent_id] = {
                "class_path": spec["class_path"],
                "class": None,  # loaded on first use
                "name": spec["name"],
                "description": spec["description"],
                "short_name": spec["short_name"],
            }
        return registry

    def _discover_plugin_agents(self) -> None:
        """Auto-register agents advertised via the 'scilink.agents' entry point group."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group="scilink.agents")
        except Exception:
            return

        for ep in eps:
            try:
                cls = ep.load()
                next_id = max(self._agent_registry.keys()) + 1 if self._agent_registry else 4
                self.register_agent(next_id, cls)
                logging.info(f"🔌 Plugin agent '{ep.name}' registered as ID {next_id}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to load plugin agent '{ep.name}': {e}")

    def register_agent(
        self,
        agent_id: int,
        agent_class: type,
        name: str = None,
        description: str = None,
        short_name: str = None,
    ) -> None:
        """Register a custom analysis agent with the orchestrator.

        The agent class must inherit from BaseAnalysisAgent and implement
        analyze(). The other BaseAnalysisAgent methods have sensible defaults
        and do not need to be overridden.

        Optional class-level attributes are used as fallbacks when name /
        description / short_name are not passed explicitly:
            AGENT_NAME        (str) display name
            AGENT_DESCRIPTION (str) one-line capability description
            AGENT_SHORT_NAME  (str) abbreviation used in output directory names

        Args:
            agent_id:    Integer ID used in select_agent / run_analysis calls.
                         Must not collide with an existing ID you want to keep.
            agent_class: The agent class (not an instance).
            name:        Display name. Falls back to AGENT_NAME or class name.
            description: Capability description. Falls back to AGENT_DESCRIPTION.
            short_name:  Short label for directory names. Falls back to
                         AGENT_SHORT_NAME or the first 8 chars of the class name.
        """
        resolved_name = name or getattr(agent_class, "AGENT_NAME", agent_class.__name__)
        resolved_desc = description or getattr(agent_class, "AGENT_DESCRIPTION", "")
        resolved_short = short_name or getattr(agent_class, "AGENT_SHORT_NAME", agent_class.__name__[:8])

        self._agent_registry[agent_id] = {
            "class_path": None,
            "class": agent_class,
            "name": resolved_name,
            "description": resolved_desc,
            "short_name": resolved_short,
        }

        # Keep the tools dicts and system prompt in sync.
        if hasattr(self, "tools"):
            self.tools._sync_from_registry()
        self._system_prompt = get_system_prompt(
            self.analysis_mode, self._agent_registry,
            self._external_tools or None, self._custom_skills or None,
            literature_available=self._literature_available,
        )
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._system_prompt

        logging.info(f"✅ Registered agent {agent_id}: {resolved_name}")

    def register_tools(self, schemas: list, factory: callable) -> None:
        """Register external tool functions into the orchestrator's LLM loop.

        The ``factory`` callable is invoked lazily at tool-call time so it can
        receive the loaded data (e.g. a NumPy image array) rather than a file
        path.  The orchestrator inspects the *name* of the factory's first
        positional parameter to decide what to pass:

        * ``data_path`` / ``path`` / ``file`` / ``filepath`` / ``filename``
          → the current data path is passed as a plain string.
        * Any other name (e.g. ``image``, ``data``, ``array``)
          → the file at the current data path is loaded (NumPy array for images
          and .npy; pandas DataFrame for CSV; raw path string as fallback) and
          that object is passed to the factory.

        The factory must return a ``dict[str, callable]`` mapping tool names to
        callables whose keyword arguments match the tool schema parameters.
        Results are JSON-serialized if not already a string.

        A per-path cache avoids reloading the file on every tool call.  The
        cache is keyed by ``(data_path, id(factory))`` so it is invalidated
        automatically when the user switches to a different data file.

        Args:
            schemas: List of OpenAI-format tool schemas
                     (``[{"type": "function", "function": {...}}, ...]``).
            factory: Callable that accepts data and returns bound tool functions.
        """
        import inspect

        # Inspect factory signature once to decide calling convention.
        sig = inspect.signature(factory)
        params = list(sig.parameters.keys())
        first_param = params[0] if params else None
        _path_param_names = {'data_path', 'path', 'file', 'filepath', 'file_path', 'filename'}
        factory_takes_path = first_param in _path_param_names
        # If the factory declares an ``output_dir`` parameter, pass the session's
        # custom-tools output directory so tools can save files and plots.
        factory_takes_output_dir = 'output_dir' in params

        registered = 0
        for schema in schemas:
            if schema.get("type") != "function":
                continue
            fn_info = schema["function"]
            tool_name = fn_info["name"]
            description = fn_info.get("description", "")
            params_spec = fn_info.get("parameters", {})
            properties = params_spec.get("properties", {})
            required = params_spec.get("required", [])

            def _make_wrapper(name, _factory, _takes_path, _takes_output_dir):
                def wrapper(**kwargs):
                    data_path = self.current_data_path
                    if data_path is None:
                        return json.dumps({
                            "status": "error",
                            "message": "No data loaded. Use examine_data first.",
                        })
                    # Resolve output directory for tools that save files / plots.
                    output_dir = self.results_dir / "custom_tools"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cache_key = (data_path, id(_factory), str(output_dir))
                    if cache_key not in self._tool_data_cache:
                        if _takes_path:
                            data = data_path
                        else:
                            data = self._load_data_as_array(data_path)
                        if _takes_output_dir:
                            self._tool_data_cache[cache_key] = _factory(data, output_dir=str(output_dir))
                        else:
                            self._tool_data_cache[cache_key] = _factory(data)
                    bound_fns = self._tool_data_cache[cache_key]
                    fn = bound_fns.get(name)
                    if fn is None:
                        return json.dumps({
                            "status": "error",
                            "message": (
                                f"Tool '{name}' not found in factory output. "
                                f"Available: {list(bound_fns.keys())}"
                            ),
                        })
                    print(f"  ⚡ Tool: {name}...")
                    result = fn(**kwargs)
                    return result if isinstance(result, str) else json.dumps(result)
                return wrapper

            self.tools._register_tool(
                func=_make_wrapper(tool_name, factory, factory_takes_path, factory_takes_output_dir),
                name=tool_name,
                description=description,
                parameters=properties,
                required=required,
            )
            self._external_tools.append({"name": tool_name, "description": description})
            registered += 1

        logging.info(f"✅ Registered {registered} external tool(s)")

        # Keep the system prompt current so the LLM knows the new tools exist.
        self._system_prompt = get_system_prompt(
            self.analysis_mode, self._agent_registry,
            self._external_tools, self._custom_skills or None,
            literature_available=self._literature_available,
        )
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._system_prompt

    def register_skill(self, skill_path: str) -> str:
        """Register a custom skill file (.md) for use in analysis.

        Custom skills appear alongside built-in skills in the ``run_analysis``
        tool description and in ``show_available_agents`` output, so the LLM
        can select them by name.

        Args:
            skill_path: Path to a ``.md`` skill file.

        Returns:
            The skill name (file stem) used to reference it.

        Raises:
            FileNotFoundError: If *skill_path* does not exist.
            ValueError: If the file is empty or not a ``.md`` file.
        """
        path = Path(skill_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")
        if path.suffix.lower() != ".md":
            raise ValueError(f"Skill file must be a .md file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Skill file is empty: {path}")

        name = path.stem
        self._custom_skills[name] = str(path)

        # Update the run_analysis tool description so the LLM sees the new skill.
        self.tools._update_skill_description(self._custom_skills)

        # Rebuild system prompt so the LLM knows the skill exists.
        self._system_prompt = get_system_prompt(
            self.analysis_mode, self._agent_registry,
            self._external_tools, self._custom_skills,
            literature_available=self._literature_available,
        )
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._system_prompt

        logging.info(f"✅ Registered custom skill '{name}' from {path}")
        return name

    # ── MCP server integration ─────────────────────────────────────────

    def connect_mcp_server(
        self,
        server_name: str,
        *,
        command: list = None,
        url: str = None,
        env: dict = None,
    ) -> int:
        """Connect to an MCP server and register its tools.

        Args:
            server_name: Human-readable label for this server.
            command: Command + args for stdio transport,
                e.g. ``["npx", "-y", "@mcp/server-filesystem", "/tmp"]``.
            url: URL for SSE transport.
            env: Optional environment variables for the subprocess.

        Returns:
            Number of tools registered from this server.
        """
        from ...mcp_client import MCPConnection

        if server_name in self._mcp_connections:
            logging.warning(
                f"MCP server '{server_name}' already connected — "
                "disconnect first to reconnect."
            )
            return 0

        conn = MCPConnection(server_name, command=command, url=url, env=env)
        schemas = conn.connect()

        existing_names = {t["name"] for t in self._external_tools}
        registered = 0
        for schema in schemas:
            fn_info = schema.get("function", {})
            tool_name = fn_info.get("name", "")
            if not tool_name:
                continue

            # Prefix with server name if there's a collision.
            display_name = tool_name
            if tool_name in self.tools.functions_map or tool_name in existing_names:
                tool_name = f"{server_name}_{tool_name}"
                logging.info(
                    f"MCP tool renamed to '{tool_name}' to avoid collision"
                )

            description = fn_info.get("description", "")
            params_spec = fn_info.get("parameters", {})
            properties = params_spec.get("properties", {})
            required = params_spec.get("required", [])

            def _make_mcp_wrapper(_conn, _name):
                def wrapper(**kwargs):
                    return _conn.call_tool(_name, kwargs)
                return wrapper

            self.tools._register_tool(
                func=_make_mcp_wrapper(conn, display_name),
                name=tool_name,
                description=f"[MCP:{server_name}] {description}",
                parameters=properties,
                required=required,
            )
            self._external_tools.append({
                "name": tool_name,
                "description": f"[MCP:{server_name}] {description}",
            })
            registered += 1

        self._mcp_connections[server_name] = conn

        # Rebuild system prompt so the LLM sees the new tools.
        self._system_prompt = get_system_prompt(
            self.analysis_mode, self._agent_registry,
            self._external_tools, self._custom_skills or None,
            literature_available=self._literature_available,
        )
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._system_prompt

        logging.info(
            f"✅ MCP '{server_name}': registered {registered} tool(s)"
        )
        return registered

    def disconnect_mcp_server(self, server_name: str) -> None:
        """Disconnect from an MCP server and unregister its tools."""
        conn = self._mcp_connections.pop(server_name, None)
        if conn is None:
            logging.warning(f"MCP server '{server_name}' not found.")
            return

        conn.disconnect()

        # Remove tools that came from this server.
        prefix = f"[MCP:{server_name}]"
        # Collect names to remove before filtering.
        names_to_remove = {
            s.get("function", {}).get("name")
            for s in self.tools.openai_schemas
            if s.get("function", {}).get("description", "").startswith(prefix)
        }
        self._external_tools = [
            t for t in self._external_tools
            if not t["description"].startswith(prefix)
        ]
        self.tools.openai_schemas = [
            s for s in self.tools.openai_schemas
            if s.get("function", {}).get("name") not in names_to_remove
        ]
        for name in names_to_remove:
            self.tools.functions_map.pop(name, None)

        # Rebuild system prompt.
        self._system_prompt = get_system_prompt(
            self.analysis_mode, self._agent_registry,
            self._external_tools, self._custom_skills or None,
            literature_available=self._literature_available,
        )
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._system_prompt

        logging.info(f"🔌 MCP '{server_name}' disconnected.")

    def disconnect_all_mcp_servers(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self._mcp_connections):
            self.disconnect_mcp_server(name)

    def _load_data_as_array(self, data_path: str):
        """Load a data file for use by external tool factories.

        Returns a NumPy array for images and .npy files, a pandas DataFrame for
        tabular files, or the path string itself as a fallback.
        """
        path = Path(data_path)
        ext = path.suffix.lower()
        if ext in {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}:
            from ...skills._shared.image_processor import load_image
            return load_image(str(path))
        elif ext == '.npy':
            import numpy as np
            return np.load(str(path))
        elif ext in {'.csv', '.txt', '.tsv'}:
            try:
                import pandas as pd
                return pd.read_csv(str(path))
            except ImportError:
                import numpy as np
                return np.genfromtxt(str(path), delimiter=',')
        else:
            self.logger.warning(
                f"_load_data_as_array: unknown extension '{ext}' — "
                "passing path string to factory."
            )
            return data_path

    def create_agent_for_analysis(self, agent_id: int, output_dir: str) -> Any:
        """
        Create an agent instance configured for a specific analysis run.
        
        Each analysis run gets a fresh agent instance with its own output
        directory, ensuring outputs from different analyses don't collide.
        
        Args:
            agent_id: The agent type ID (0-2)
            output_dir: Unique output directory for this analysis run
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent_id is invalid
        """
        entry = self._agent_registry.get(agent_id)
        if entry is None:
            raise ValueError(
                f"Unknown agent ID: {agent_id}. "
                f"Valid IDs: {sorted(self._agent_registry.keys())}"
            )

        common_kwargs = {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "output_dir": output_dir,
            "enable_human_feedback": self._enable_human_feedback,
            # Consumed only by ImageAnalysisAgent. Filtered out below for any
            # agent whose __init__ neither declares it nor accepts **kwargs
            # (hyperspectral, FFT, SAM, atomistic) — passing it would raise.
            "analysis_depth": self.image_analysis_depth,
        }

        # Lazy-load built-in agents; custom agents already carry their class.
        cls = entry.get("class")
        if cls is None:
            import importlib
            module_path, class_name = entry["class_path"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            entry["class"] = cls  # cache for subsequent calls

        # Keep only the kwargs this agent's __init__ actually accepts. An agent
        # that declares **kwargs gets everything; one that does not gets only
        # its explicitly-named parameters.
        try:
            params = inspect.signature(cls.__init__).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD
                       for p in params.values()):
                common_kwargs = {
                    k: v for k, v in common_kwargs.items() if k in params
                }
        except (ValueError, TypeError):
            pass  # unintrospectable __init__ — pass kwargs through unchanged

        agent = cls(**common_kwargs)
        logging.info(f"   Created agent {agent_id}: {type(agent).__name__}")
        logging.info(f"   Output directory: {output_dir}")
        return agent

    def generate_analysis_id(self, data_path: str, agent_id: int) -> str:
        """
        Generate a unique analysis ID based on dataset path, agent, and timestamp.
        
        Format: {dataset_name}_{agent_short_name}_{timestamp}_{counter}
        
        Note: We use "dataset" rather than "sample" because the same physical
        sample may produce multiple datasets (different imaging conditions,
        time points, techniques, etc.).
        
        Args:
            data_path: Path to the dataset being analyzed
            agent_id: ID of the agent performing the analysis
            
        Returns:
            Unique analysis ID string
        """
        # Extract meaningful name from dataset path
        data_path_obj = Path(data_path)
        if data_path_obj.is_dir():
            dataset_name = data_path_obj.name
        else:
            dataset_name = data_path_obj.stem
        
        # Sanitize the name (remove special characters)
        import re
        dataset_name = re.sub(r'[^\w\-]', '_', dataset_name)
        
        # Truncate if too long
        if len(dataset_name) > 30:
            dataset_name = dataset_name[:30]
        
        # Get short agent name from registry
        agent_short = self._agent_registry.get(agent_id, {}).get("short_name", f"agent{agent_id}")
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Increment counter for uniqueness within same second
        self._analysis_run_counter += 1
        
        return f"{dataset_name}_{agent_short}_{timestamp}_{self._analysis_run_counter:03d}"

    def _restore_checkpoint(self):
        """Restore session state from checkpoint."""
        print(f"  📂 Restoring checkpoint from: {self.checkpoint_path}")
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            self.current_metadata = state.get("current_metadata")
            self.current_data_path = state.get("current_data_path")
            self.current_data_type = state.get("current_data_type")
            self.selected_agent_id = state.get("selected_agent_id")
            self.analysis_results = state.get("analysis_results", [])
            self.active_knowledge = state.get("active_knowledge", [])
            self._graduated_skill_sources = state.get("graduated_skill_sources", {})
            self._analysis_run_counter = state.get("analysis_run_counter", 0)
            
            # Restore analysis mode if saved
            if "analysis_mode" in state:
                try:
                    self.analysis_mode = AnalysisMode(state["analysis_mode"])
                    self._enable_human_feedback = self._should_enable_human_feedback()
                except ValueError:
                    pass
            
            print(f"    ✅ Restored state:")
            print(f"       - Data path: {self.current_data_path}")
            print(f"       - Data type: {self.current_data_type}")
            print(f"       - Selected agent: {self.selected_agent_id}")
            print(f"       - Analysis count: {len(self.analysis_results)}")
            print(f"       - Analysis mode: {self.analysis_mode.value}")
            
        except Exception as e:
            logging.warning(f"Failed to restore checkpoint: {e}")

    def _trim_history(self, history: List[Dict], max_messages: int = None) -> List[Dict]:
        """Keep only recent messages to avoid context window overflow."""
        if max_messages is None:
            max_messages = self.MAX_HISTORY_MESSAGES
            
        if len(history) <= max_messages:
            return history
        
        print(f"  ⚠️  Trimming history: {len(history)} → {max_messages} messages")
        
        context_window = 10
        recent_window = max_messages - context_window
        
        trimmed = history[:context_window] + history[-recent_window:]
        
        summary_marker = {
            "role": "system",
            "content": f"[{len(history) - max_messages} messages omitted for context management]"
        }
        trimmed.insert(context_window, summary_marker)
        
        return trimmed

    def chat(self, user_input: str) -> str:
        """Main chat interface with robust function calling support."""
        self.message_count += 1
        self._last_chat_hit_iter_cap = False

        # AUTO-CHECKPOINT: Every N messages
        if self.message_count - self.last_checkpoint_message_count >= self.CHECKPOINT_INTERVAL:
            print(f"  💾 Auto-checkpoint triggered (every {self.CHECKPOINT_INTERVAL} messages)...")
            self._auto_checkpoint()
            self.last_checkpoint_message_count = self.message_count
        
        try:
            if self.use_openai:
                response_text = self._handle_openai_chat(user_input)
            else:
                response_text = self._handle_litellm_chat(user_input)
            
            self._print_agent_answer(response_text)
            self._save_history()
            
            if self.message_count > 80:
                warning = "\n\n⚠️ Note: Conversation is getting long. Consider calling save_checkpoint and restarting."
                response_text += warning
            
            return response_text
            
        except Exception as e:
            logging.error(f"Chat Error: {e}", exc_info=True)
            
            print("  💾 Error detected - saving emergency checkpoint...")
            self._auto_checkpoint()
            
            return f"❌ Error: {e}\n\n(Emergency checkpoint saved to {self.checkpoint_path})"

    def run_task(self, task: str, context: Optional[Dict[str, Any]] = None,
                 autonomy: Optional[AnalysisMode] = None,
                 max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Non-interactive entry point — used by the meta agent.

        Runs the task and returns a structured summary that's easy to
        consume programmatically:

            {
                "status": "success" | "error",
                "task": str,                       # echoed input
                "summary": str,                    # the agent's final reply
                "files_produced": List[str],       # absolute paths
                "feature_tables": List[str],       # per-analysis feature CSVs
                "key_findings": List[str],         # extracted scientific claims
                "suggested_followups": List[str],
                "analyses": List[dict],            # session record snapshot
                "warnings": List[str],
            }

        Mirrors SimulationOrchestratorAgent.run_task — see CLAUDE.md
        "Two surfaces, one agent".

        ``autonomy`` selects the AnalysisMode for this call. Defaults to
        AUTONOMOUS — the safe choice for a headless/programmatic caller, so
        the agent never pauses for a nonexistent user. A caller attached to a
        human (the meta agent, driven via CLI/UI) passes AUTOPILOT so the
        sub-agents' human-feedback prompts reach that human.
        The original mode is restored on exit, even if chat() raises.
        """
        # Build a self-contained prompt that includes the optional context.
        prompt = task
        if context:
            try:
                ctx_str = json.dumps(context, indent=2, default=str)
            except (TypeError, ValueError):
                ctx_str = repr(context)
            prompt = (
                f"{task}\n\n"
                f"Context provided by the caller (e.g., upstream agent's findings):\n"
                f"```\n{ctx_str}\n```\n\n"
                "Use this context together with your tools to complete the task."
            )

        # Snapshot prior state so we report "what was produced *during* this
        # call" rather than "everything in the session."
        n_before = len(self.analysis_results)

        # Run under the requested autonomy mode — AUTONOMOUS by default (the
        # safe headless choice). The meta agent passes its own mode through,
        # so a co-pilot / autopilot delegation still raises the sub-agents'
        # human-feedback prompts to the user driving the session.
        run_mode = autonomy if autonomy is not None else AnalysisMode.AUTONOMOUS
        original_mode = self.analysis_mode
        original_max_iter = self.max_iterations
        try:
            self.set_analysis_mode(run_mode)
            if max_iterations is not None:
                self.max_iterations = max_iterations
            try:
                summary_text = self.chat(prompt)
                # chat() handles the iteration-cap case internally and returns
                # the warning string (preserving CLI/UI behavior); a flag on
                # self tells us whether that happened so we can surface it as
                # an error to programmatic callers instead of silently
                # reporting success.
                if self._last_chat_hit_iter_cap:
                    status = "error"
                    error_msg: Optional[str] = summary_text
                else:
                    status = "success"
                    error_msg = None
            except Exception as e:
                self.logger.exception(f"run_task failed: {e}")
                summary_text = ""
                status = "error"
                error_msg = str(e)
        finally:
            # Always restore the original mode, even if chat() raised.
            self.set_analysis_mode(original_mode)
            self.max_iterations = original_max_iter

        # Derive the structured summary from the session-state delta.
        new_analyses = self.analysis_results[n_before:]

        # files_produced: every file written under each new analysis's
        # output directory (visualizations, reports, results JSON, ...).
        files_produced: List[str] = []
        for rec in new_analyses:
            out_dir = rec.get("output_directory")
            if out_dir and Path(out_dir).is_dir():
                for p in sorted(Path(out_dir).rglob("*")):
                    if p.is_file():
                        files_produced.append(str(p.resolve()))

        # key_findings: the scientific claims extracted by the sub-agents.
        key_findings: List[str] = []
        for rec in new_analyses:
            full = rec.get("full_result") or {}
            for claim in full.get("scientific_claims", []) or []:
                text = claim.get("claim") if isinstance(claim, dict) else claim
                if text:
                    key_findings.append(f"[{rec.get('analysis_id')}] {text}")

        # Heuristic: a successful analysis with no novelty assessment yet is
        # a natural next step — claims can be checked against the literature.
        suggested_followups: List[str] = []
        for rec in new_analyses:
            if rec.get("status") == "success" and not rec.get("novelty_assessment"):
                suggested_followups.append(
                    f"Assess novelty / get recommendations for analysis "
                    f"{rec.get('analysis_id')}."
                )

        warnings: List[str] = []
        for rec in new_analyses:
            if rec.get("status") != "success":
                warnings.append(
                    f"Analysis {rec.get('analysis_id')} did not complete "
                    f"successfully (status={rec.get('status')})."
                )

        # feature_tables: per-analysis flat CSVs (conditions + extracted scalar
        # features) for downstream planning / BO — see feature_table.py.
        feature_tables: List[str] = []
        for rec in new_analyses:
            out_dir = rec.get("output_directory")
            if out_dir:
                ft = Path(out_dir) / "features.csv"
                if ft.is_file():
                    feature_tables.append(str(ft.resolve()))

        # staged_solutions: raw T=2 (hot-annealing) solutions filed in the
        # staging buffer during this run. Surfaced so the meta agent can offer
        # the user a review — upgrade an existing skill from one, or consolidate
        # several into a new skill (see review_distilled_skills). Ids into the
        # staging buffer (~/.scilink/distill_staging).
        staged_solutions: List[str] = []
        for rec in new_analyses:
            full = rec.get("full_result") or {}
            for sid in full.get("staged_solutions", []) or []:
                if sid not in staged_solutions:
                    staged_solutions.append(sid)

        result = {
            "status": status,
            "task": task,
            "summary": summary_text,
            "files_produced": files_produced,
            "feature_tables": feature_tables,
            "staged_solutions": staged_solutions,
            "key_findings": key_findings,
            "suggested_followups": suggested_followups,
            "analyses": [
                {
                    "analysis_id": rec.get("analysis_id"),
                    "agent_name": rec.get("agent_name"),
                    "status": rec.get("status"),
                    "data_path": rec.get("data_path"),
                    "output_directory": rec.get("output_directory"),
                } for rec in new_analyses
            ],
            "warnings": warnings,
        }
        if error_msg:
            result["error"] = error_msg
        return result

    def _auto_checkpoint(self):
        """Internal auto-checkpoint without LLM interaction."""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "current_metadata": self.current_metadata,
                "current_data_path": self.current_data_path,
                "current_data_type": self.current_data_type,
                "selected_agent_id": self.selected_agent_id,
                "analysis_results": self.analysis_results,
                "analysis_run_counter": self._analysis_run_counter,
                "message_count": self.message_count,
                "analysis_mode": self.analysis_mode.value,
                "active_knowledge": self.active_knowledge,
                "graduated_skill_sources": self._graduated_skill_sources,
            }

            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            print(f"    ✅ Auto-checkpoint saved")
            
        except Exception as e:
            logging.warning(f"Auto-checkpoint failed: {e}")

    def _handle_openai_chat(self, user_input: str) -> str:
        """Handle chat with OpenAI-compatible models with manual function calling loop."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url,
            timeout=120.0  # 2 minute timeout
        )
        
        self.messages.append({"role": "user", "content": user_input})
        
        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=self.MAX_HISTORY_MESSAGES)
            self.messages = [system_msg] + recent_msgs
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            print(f"  ⏳ Waiting for orchestrator response ...")
            
            try:
                response = client.chat.completions.create(
                    model=self.model.model,
                    messages=self.messages,
                    tools=self.tools_for_model,
                    tool_choice="auto"
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"  ⚠️ API timeout on iteration {iteration}. Retrying...")
                    if iteration < 3:  # Retry up to 3 times on timeout
                        continue
                raise
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                text = message.content
                if not text and iteration > 0:
                    # LLM returned empty text after tool calls — ask for a summary
                    self.messages.append({"role": "user", "content": "Please briefly summarize what you just did and suggest next steps."})
                    followup = self.client.chat.completions.create(
                        model=self.model.model,
                        messages=self.messages,
                        tools=self.tools_for_model,
                        tool_choice="none"
                    )
                    text = followup.choices[0].message.content or ""
                self.messages.append({
                    "role": "assistant",
                    "content": text
                })
                return text
            
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in message.tool_calls
                ]
            })

            self._print_assistant_reasoning(message.content)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"  🔧 Calling tool: {func_name}")
                
                result = self.tools.execute_tool(func_name, **args)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        self._last_chat_hit_iter_cap = True
        return "⚠️ Maximum tool iterations reached. Please simplify your request."

    def _print_assistant_reasoning(self, content) -> None:
        """Surface the LLM's interim reasoning that accompanies a tool call.

        Without this the console jumps straight to "🔧 Calling tool: …", so a
        deliberate step — e.g. re-running an analysis to verify a methodological
        concern — reads like a silent loop. The text is already in history; this
        just shows it so the user can see *why* the next tool is being called.
        """
        if not content:
            return
        text = content.strip() if isinstance(content, str) else str(content).strip()
        if not text:
            return
        import sys
        # Dim + italic (muted cyan) so the prose is visually distinct from the
        # structural tool-call log — but only on a real terminal, else the raw
        # escape codes would litter captured/redirected output. Blank lines
        # before and after set the thought apart from the tool call it triggers.
        # A meta-delegated run reasons in AMBER (33), the meta's own in CYAN (36),
        # so the CLI distinguishes them the way the UI does (dim + italic either way).
        specialist = getattr(self, "_agent_label", "Agent") != "Agent"
        color = "33" if specialist else "36"
        style, reset = ((f"\033[2;3;{color}m", "\033[0m")
                        if sys.stdout.isatty() else ("", ""))
        body = text.replace("\n", "\n     ")  # indent continuation lines
        # Tag a meta-delegated run's reasoning with an INVISIBLE marker (U+2063)
        # right after 💭 so the UI renders it a distinct color from the meta's
        # own 💭 while the visible glyph stays identical. ANSI color can't carry
        # this: the UI captures non-tty output (no ANSI emitted) and re-colors by
        # the 💭 marker, so the source must ride in the text. Gated on
        # `_agent_label` (the 🤖-answer attribution mechanism); a standalone
        # session keeps a plain 💭, unchanged.
        mark = "\u2063" if specialist else ""
        print(f"\n  {style}💭{mark} {body}{reset}\n")

    def _print_agent_answer(self, text) -> None:
        """Print the agent's final answer — a *deliverable*, the third output
        class alongside structural logs and `_print_assistant_reasoning`'s 💭
        asides. Reasoning is de-emphasized (dim); an answer is the payload, so
        its header is emphasized (bold + bright). `_agent_label` (default
        "Agent") lets the meta name the delegated specialist, e.g. "Analysis
        specialist", so a child's answer is not mistaken for the meta's own.
        """
        import sys
        label = getattr(self, "_agent_label", "Agent")
        # A delegated specialist's answer header takes the specialist color (bold
        # AMBER, 33) so it matches that specialist's reasoning; the meta's own
        # answer stays bold BRIGHT CYAN (96). The UI mirrors this by the invisible
        # U+2063 marker after 🤖 (it captures non-tty output, so it can't read the
        # ANSI and re-colors by the marker instead).
        specialist = label != "Agent"
        code = "1;33" if specialist else "1;96"
        mark = "\u2063" if specialist else ""
        bold, reset = (f"\033[{code}m", "\033[0m") if sys.stdout.isatty() else ("", "")
        print(f"\n{bold}🤖{mark} {label}:{reset}")
        print(text if text is not None else "")

    def _handle_litellm_chat(self, user_input: str) -> str:
        """Handle chat with LiteLLM models with manual function calling loop."""
        import litellm
        from ...wrappers.litellm_wrapper import litellm_completion

        self.messages.append({"role": "user", "content": user_input})
        
        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=self.MAX_HISTORY_MESSAGES)
            self.messages = [system_msg] + recent_msgs
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            print(f"  ⏳ Waiting for orchestrator response ...")
            
            try:
                response = litellm_completion(
                    model=self.model.model,
                    messages=self.messages,
                    tools=self.tools_for_model,
                    tool_choice="auto",
                    api_key=self.model.api_key,
                    api_base=self.model.base_url,
                    timeout=120,  # Connection timeout in seconds
                    request_timeout=120,  # Alternative timeout parameter
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"  ⚠️ API timeout on iteration {iteration}. Retrying...")
                    if iteration < 3:  # Retry up to 3 times on timeout
                        continue
                raise
            
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            content = getattr(message, "content", None)
            
            if not tool_calls:
                if not content and iteration > 0:
                    # LLM returned empty text after tool calls — ask for a summary
                    self.messages.append({"role": "user", "content": "Please briefly summarize what you just did and suggest next steps."})
                    followup = litellm_completion(
                        model=self.model.model,
                        messages=self.messages,
                        tools=self.tools_for_model,
                        tool_choice="none",
                    )
                    content = getattr(followup.choices[0].message, "content", None) or ""
                self.messages.append({
                    "role": "assistant",
                    "content": content or ""
                })
                return content or ""
            
            assistant_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in tool_calls
                ]
            }
            self.messages.append(assistant_msg)

            self._print_assistant_reasoning(content)

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  🔧 Calling tool: {func_name}")
                
                result = self.tools.execute_tool(func_name, **args)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        self._last_chat_hit_iter_cap = True
        return "⚠️ Maximum tool iterations reached. Please simplify your request."

    def _load_history(self) -> List[Dict]:
        """Load conversation history from disk."""
        if not self.history_path.exists():
            return []
        print("  🧠 Memory: Loading previous conversation...")
        try:
            with open(self.history_path, 'r') as f:
                saved = json.load(f)
            return saved
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")
            return []

    def _save_history(self):
        """Save conversation history to disk."""
        try:
            history_data = [m for m in self.messages if m["role"] != "system"]
            with open(self.history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save history: {e}")

    @classmethod
    def restore_from_checkpoint(cls, base_dir: str, **kwargs):
        """Factory method to create an AnalysisOrchestratorAgent from a checkpoint."""
        return cls(base_dir=base_dir, restore_checkpoint=True, **kwargs)