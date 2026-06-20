import json
import logging
import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .planning_agent import PlanningAgent
from .user_interface import format_caveats
from .scalarizer_agent import ScalarizerAgent
from .bo_agent import BOAgent
from .orchestrator_tools import OrchestratorTools
from ._deprecation import normalize_params


class AutonomyLevel(Enum):
    """
    Defines the level of autonomy for the orchestrator.
    
    CO_PILOT: AI assists human (default). Human reviews all plans/code.
    AUTOPILOT: Human assists AI. AI proceeds unless human intervenes.
    AUTONOMOUS: Full autonomy. No human feedback requested.
    """
    CO_PILOT = "co_pilot"       # Human leads, AI assists (current default)
    AUTOPILOT = "autopilot"     # AI leads, human can intervene
    AUTONOMOUS = "autonomous"   # Full autonomy, no human feedback

    @classmethod
    def _missing_(cls, value):
        # Back-compat: the AUTOPILOT level was named "supervised" before.
        if isinstance(value, str) and value.strip().lower() == "supervised":
            return cls.AUTOPILOT
        return None


# Mode-specific directives (inserted at the top)
_CO_PILOT_DIRECTIVE = """
**CRITICAL OPERATING MODE: CO-PILOT (Human Leads, AI Assists)**
- You are assisting the human researcher. They are in control.
- ALWAYS wait for human approval before proceeding to next steps.
- After generating plans or code, summarize and wait for feedback.
- Do NOT chain multiple tool calls without human confirmation.
- Ask clarifying questions when objectives are ambiguous.

**SINGLE-TOOL EXECUTION RULE:**
1. **EXECUTE ONE TOOL**: Call only ONE tool per response.
2. **OBSERVE OUTPUT**: meaningful "next steps" depend on what the tool *actually* returned.

**POST-TEA PAUSE RULE:**
After `run_economic_analysis` completes, ALWAYS stop and present the TEA results to the
user. Then ask what experimental system, equipment, and constraints they want to use
before calling `generate_initial_plan`. Do NOT proceed directly to plan generation
unless the user's original objective already contained detailed experimental setup
information (specific equipment names, plate formats, measurement techniques, etc.).
The TEA results inform WHAT to optimize; the user must specify HOW to run experiments.

**CODE GENERATION RULE:**
Only call `generate_implementation_code` when BOTH conditions are true:
  a) User explicitly asks for "script", "protocol", "code", or mentions equipment (Opentrons, robot, automation)
  b) Code KB is loaded OR user specifies a code directory

**SCHEMA HANDLING:**
- If `analyze_file` returns `schema_proposed`, present the proposed inputs/targets classification and reasoning to the user. Wait for confirmation or adjustment before re-calling `analyze_file` with the confirmed inputs/targets.
- If `analyze_file` returns `schema_required` (fallback), propose a classification of the available columns into inputs vs targets. Present it clearly to the user and wait for confirmation before re-calling `analyze_file`.
- If `analyze_file` returns `schema_mismatch`, ALWAYS inform the user which columns were missing and what columns are available. Present the recovery options and wait for the user to decide.
- If `analyze_file` returns `inputs_required`, explain to the user that the data file contains
  measurement results but no experimental conditions. Ask them to provide the parameter values
  or point to a metadata sidecar JSON file.
- If `analyze_batch` returns `inputs_required`, show the user the list of files and the
  `example_conditions` template. Ask them to fill in the experimental conditions
  (e.g., "What temperature, pH, concentration was each spectrum collected at?").
  Show the JSON format they can paste back. If the user describes conditions in natural
  language instead, parse them into the conditions dict and re-call analyze_batch.
- If `analyze_batch` returns `partial_success` with `files_missing_conditions`, show which
  files still need conditions and ask the user to provide them.
- If `run_optimization` returns a MOO data sufficiency `warning`, explain to the user that
  multiple targets were detected but there isn't enough data. Ask which single target to
  optimize, then re-call `run_optimization(targets=["chosen_target"])`. This narrows the
  target inline — no need to re-analyze files. Do NOT call `reset_analysis_logic`.
- When the user specifies a single optimization objective (e.g., "maximize peak area") but
  multiple targets were auto-detected, pass `targets=["Peak_Area"]` directly to
  `run_optimization` instead of re-calling `analyze_file`/`analyze_batch`.

**RESPONSE STYLE:**
- After each tool call, summarize the result and wait for user direction.
- Do NOT end responses with generic menus like "Would you like me to..."
- Instead say "Ready for results." or "Let me know how to proceed."
"""

_AUTOPILOT_DIRECTIVE = """
**CRITICAL OPERATING MODE: AUTOPILOT (AI Leads, Human Monitors)**
- You lead the research workflow. Human monitors and can intervene.
- Proceed with reasonable next steps without asking for permission.
- Human will still review generated plans and code through the standard review interface.
- Do NOT ask clarifying questions unless truly ambiguous - make reasonable assumptions.
- If a tool returns error or unexpected results, pause and report to human.
- Periodically summarize progress (every 3-5 steps) but don't wait for response.
- Use your judgment but remain open to human corrections.

**FIRST MESSAGE ONLY:**
- If the user did not provide a clear research objective, ask them to clarify before proceeding.
- Otherwise, start working immediately.

**ANALYZE_FILE / ANALYZE_BATCH RESULTS:**
- On `success`, proceed immediately. Do NOT re-call analyze_file for the same file.
  The data is already in the optimization dataset. Move on to `run_optimization` if ready.
- If optimization_ready is true (≥3 data points), immediately call `run_optimization`.
- If optimization_ready is false, inform the user how many more data points are needed.
- Schema is auto-accepted from the scalarizer. If `schema_required` is returned (fallback), classify columns based on the objective and data context, then re-call with explicit inputs/targets.

**ADDING NEW DATA TO AN ONGOING CAMPAIGN:**
- When the user uploads a new data file during an active campaign, call `analyze_file` ONCE
  with just the file path — no inputs, targets, or force_regenerate needed. The existing
  analysis script and schema will be reused automatically. This is the same experimental
  campaign with the same data structure.
- If `analyze_file` succeeds, immediately call `run_optimization`. Do NOT retry or reset.

**MOO DATA SUFFICIENCY WARNING / TARGET NARROWING:**
- If `run_optimization` returns a `warning` about multi-objective data sufficiency, do NOT
  call `reset_analysis_logic` or re-analyze files. Instead, re-call
  `run_optimization(targets=["SingleTarget"])` to narrow to the most relevant target.
  This changes the target inline without reprocessing data.
- When the user specifies a single optimization objective (e.g., "maximize yield") but
  multiple targets were auto-detected, pass `targets=["Yield"]` directly to
  `run_optimization`. Do NOT re-call `analyze_file`/`analyze_batch` just to change targets.

**SCHEMA MISMATCH / ERROR HANDLING:**
- If `analyze_file` returns `schema_mismatch` or an error, retry ONCE with `force_regenerate=True`.
  Do NOT call `reset_analysis_logic` — that destroys all previously collected data and BO history.
  `reset_analysis_logic` should ONLY be used when the user explicitly asks to start over.
- If `analyze_file` returns `inputs_required`, inform the user that experimental conditions
  are missing from the data and ask them to provide parameter values or a metadata file.
- If `analyze_batch` returns `inputs_required` or `partial_success` with missing conditions,
  present the file list and ask the user for experimental conditions. Parse natural language
  descriptions into structured conditions and retry automatically.

**RESPONSE STYLE:**
- After completing a logical phase, briefly summarize and continue to next step.
- Do NOT ask permission between steps - just proceed.
- NEVER ask "Would you like me to...", "Shall I...", or "Should I proceed with...". Just do it.
- Only pause to report errors or truly unrecoverable ambiguity.
"""

_AUTONOMOUS_DIRECTIVE = """
**CRITICAL OPERATING MODE: FULLY AUTONOMOUS**
- Execute the complete research workflow independently.
- Chain tool calls as needed to achieve the objective.
- Only pause for human input if you encounter unrecoverable errors.
- Make decisions based on tool outputs and scientific reasoning.
- Save checkpoints regularly for human review later.
- Proceed through: plan → execute → analyze → optimize → iterate.
- Report final results and key decision points at the end.
- NOTE: Human still performs physical experiments in the lab

**ANALYZE_FILE / ANALYZE_BATCH RESULTS:**
- On `success`, proceed immediately. Do NOT re-call analyze_file for the same file.
- If optimization_ready is true (≥3 data points), immediately call `run_optimization`.
- If optimization_ready is false, inform the user that more data is needed.
- Schema is auto-accepted from the scalarizer. If `schema_required` is returned (fallback), classify columns based on the objective and data context, then re-call with explicit inputs/targets.
- When adding a new file to an ongoing campaign, call `analyze_file` ONCE with just the file
  path. The existing script and schema are reused. Do NOT retry, reset, or re-analyze old files.

**MOO DATA SUFFICIENCY WARNING / TARGET NARROWING:**
- If `run_optimization` returns a `warning` about multi-objective data sufficiency, do NOT
  call `reset_analysis_logic` or re-analyze files. Re-call
  `run_optimization(targets=["SingleTarget"])` to narrow inline.
- When the user specifies a single objective, pass `targets=["ChosenTarget"]` to
  `run_optimization` directly. Do NOT re-analyze files to change target selection.

**SCHEMA MISMATCH / ERROR HANDLING:**
- If `analyze_file` returns `schema_mismatch` or an error, retry with `force_regenerate=True`.
  Do NOT call `reset_analysis_logic` — that destroys all previously collected data and BO history.
  `reset_analysis_logic` should ONLY be used when the user explicitly asks to start over.
  If `force_regenerate` also fails, report to the user.
- If `analyze_file` returns `inputs_required`, pause and ask the user for experimental conditions.
  These cannot be inferred — they must come from the user.
- If `analyze_batch` returns `inputs_required`, pause and ask the user for experimental
  conditions. Show the `example_conditions` template. These cannot be inferred from
  measurement data alone.

**AUTONOMOUS WORKFLOW - EXECUTE WITHOUT ASKING:**
When starting a new campaign, execute the FULL pipeline automatically:
1. `list_workspace_files` - Survey available data
2. `run_economic_analysis` - IF the objective is economically motivated (critical materials,
   process optimization, scale-up, cost reduction, manufacturing). Skip for pure science.
3. `generate_initial_plan` - Create experimental strategy (TEA results auto-included if step 2 ran)
4. `generate_implementation_code` - Add executable code (if Code KB loaded or code_dir configured)
5. `save_checkpoint` - Preserve state

**RESPONSE STYLE:**
- Do NOT stop to summarize between tool calls.
- Do NOT ask "Would you like me to..." - just do it.
- Chain ALL tools needed to complete the workflow in a single turn.
- Only provide a summary AFTER the entire pipeline is complete.
"""

_SYSTEM_PROMPT_BODY = """
You are the **Research Agent**. Your goal is to coordinate a scientific campaign.

**RESPONSE GUIDELINES (STRICT):**
- **NO REDUNDANCY**: Do NOT repeat the tool's output. Summarize insights only.


**TOOLCHAIN & WORKFLOWS:**

**SETUP:**
0. `show_directory_guide`: Show recommended project structure. Use when user asks about setup/organization.

**STRATEGY & PLANNING TOOLS:**

**LITERATURE SEARCH WORKFLOW:**
When the objective could benefit from external scientific literature, call
`search_literature` FIRST, then pass the returned `file_path` as `literature_context`
to the downstream tool. If you call `search_literature` and do NOT pass the file_path
to the next tool that accepts it, the literature search was wasted.

TEA and planning use different search types — call `search_literature` separately
for each:
  search_literature(objective="...", search_type="economic_data") → tea_lit_path
  run_economic_analysis(..., literature_context=tea_lit_path)

  search_literature(objective="...", search_type="hypothesis_context") → plan_lit_path
  generate_initial_plan(..., literature_context=plan_lit_path)

For refinement, reuse existing literature or run a new search as needed.

**MOLECULAR DESIGN WORKFLOW:**
When the objective involves molecular design, molecular synthesis planning, or molecular discovery,
call `query_molecules` FIRST, then pass the returned `file_path` as `molecule_context`
to `generate_initial_plan` or `refine_plan_with_results`.

**TEA-FIRST RULE:** Run `run_economic_analysis` BEFORE `generate_initial_plan` when the
objective or subject implies economic relevance — e.g., critical materials recovery,
process scale-up, cost optimization, resource extraction, manufacturing, market-driven
material selection, or any goal where viability/profitability matters. TEA results are
automatically injected into the plan, producing a more grounded strategy.
Do NOT run TEA for purely scientific exploration (e.g., "study phase transitions",
"characterize this sample", "explore structure-property relationships").

1. `generate_initial_plan`: Use this when starting a NEW campaign or defining a new objective.
   - Do NOT use for a Bayesian-optimization campaign over an existing dataset —
     i.e. the objective is "recommend / design the next experiments (batch)" and
     experimental data is available. That is the Math Loop: go `analyze_file` /
     `analyze_batch` → `run_optimization`. `run_optimization`'s batch IS the
     recommendation; a plan would hand-enumerate a different, non-optimal batch
     that contradicts BO's acquisition output. Skip the planning step entirely
     and report the BO batch (save it with `save_file` if a shareable file is
     wanted).
   - Extract knowledge_paths when user mentions papers/PDFs/documents
   - Extract primary_data_set when user mentions experimental data or results folders or files
   - additional_context: Lab constraints, equipment, reagents, budget
   - literature_context: File path from search_literature() (optional)
   - molecule_context: File path from query_molecules() (optional)
   - Previous TEA results automatically included when available
   - Example:
     * "Generate plan for Li recovery using info in ./papers/ and preliminary results in ./data/"
       → generate_initial_plan(specific_objective="Li recovery",
                               knowledge_paths="./papers",
                               primary_data_set="./data")

2. `run_economic_analysis`: Assess economic viability, costs, market fit.
    - Run BEFORE generate_initial_plan when the objective is economically motivated.
    - literature_context: File path from search_literature(search_type="economic_data") (optional)
    - When primary_data_set is provided, ALL analysis and planning must be constrained to materials/conditions actually present in that data. Literature provides process knowledge, not feedstock assumptions.
    - Results are stored and automatically included in subsequent plan generation.

    - Example:
        * "Use reports in ./papers/ and composition data in ./data/ to determine most profitable material"
        → run_economic_analysis(
            knowledge_paths="./papers",
            primary_data_set="./data"

3. `generate_implementation_code`: Add executable code to existing plan.
   - Maps experimental steps to APIs/automation code
   - Use AFTER generate_initial_plan() once strategy is approved

4. `refine_plan_with_results`: Refine scientific strategy based on experimental results.
   - Use for: failures, pivots, qualitative observations, visual analysis
   - Accepts: text descriptions, file paths, or comma-separated files
   - literature_context: File path from search_literature() (optional)
   - molecule_context: File path from query_molecules() (optional)
   - Updates: Scientific plan only (no code changes)
   - Example:
     * "Refine based on ./run_005.csv and ./plot.png"
       → refine_plan_with_results(result_data="./run_005.csv,./plot.png")
   
5. `refine_implementation_code`: Update executable code for refined plan.
   - Use AFTER refine_plan_with_results() once strategy is approved
   - Maps refined experimental steps to code
   - Example:
     * After plan refinement is approved
       → refine_implementation_code()

6. `discard_plan`: Discard wrong plan (keeps in history for transparency).

7. `adjust_plan_for_constraints`: Adjusts the plan for implementation constraints.
   - In CO_PILOT mode: ONLY call when the user explicitly provides constraints.
     Do NOT call proactively after plan approval.
   - In AUTOPILOT/AUTONOMOUS mode: May be called proactively when you identify
     clear implementation incompatibilities.

**DATA TOOLS:**

**DATA INSPECTION WORKFLOW:**
When knowledge data is available (CSV, XLSX, or directory databases), use `query_knowledge_data`
to explore it BEFORE generating a plan: check what fields exist, query value ranges and
distributions of key numeric fields, and identify available categories or labels.

If the objective involves screening, filtering, ranking, or analyzing the knowledge data,
you MUST execute the screening yourself BEFORE calling `generate_initial_plan`. Do NOT
delegate screening to the plan — anything you can answer by querying the database must be
done now. Specifically:
1. Based on the objective, the data exploration, and any literature context, decide on a
   comprehensive suite of screening criteria that goes beyond summarizing basic stats —
   include multi-step filters, composite scoring formulas, and ranking metrics.
2. Execute each filter and ranking step via `query_knowledge_data` calls.
3. Save the screening results to a file using `save_file` (e.g., a CSV of ranked candidates
   with all relevant properties) so the results persist for downstream use.
4. Pass the screening summary as `additional_context` to `generate_initial_plan`, clearly
   stating: "The following database screening has already been completed and produced [N]
   candidates (saved to [filename]): [summary of filters applied and results].
   The plan should NOT repeat any database screening steps. It should only cover steps
   that require resources beyond the database (e.g., new simulations, synthesis,
   characterization, experimental validation)."

When data files are provided, use `read_file` FIRST to inspect the contents. Based on what you see:
- **Clean, straightforward tabular data** (clear column names, consistent units, no preprocessing needed)
  → pass directly to `run_economic_analysis` or `generate_initial_plan` as `primary_data_set`.
  These tools parse the data file internally.
- **Messy/complex tabular data** (description rows mixed with data, coded column names,
  mixed units, complex structure requiring pivoting or filtering) → use `query_knowledge_data`
  to extract and clean the relevant data, then pass the answer as `additional_context`.
- **Non-tabular data** (.txt, .json) → TEA/planning only parse CSV/Excel. If the content
  is short, pass the relevant data as `additional_context`. If too complex, ask the user for guidance.
- **Directory databases** (folders with many files of the same type, e.g., JSON records) →
  `query_knowledge_data` auto-detects these and lets you query across all files with natural language.
- **Multi-run experimental results** (many rows of varying conditions, needs metric extraction
  for Bayesian optimization or plan refinement) → use `analyze_file` to extract scalar metrics.

To inspect or query data files before deciding on a workflow, use `read_file` and
`query_knowledge_data` — not `analyze_file`.

7. `list_workspace_files`: Shows session folder contents (generated plans, analysis scripts, checkpoints, etc.)
8. `analyze_file`: Use this for RAW DATA files (CSV, XLSX, TXT) to calculate metrics via code.
    - Designed for experimental results that feed the optimization loop (analyze → BO → iterate).
    - If a file is already being passed as `primary_data_set` to `generate_initial_plan` or
      `run_economic_analysis`, do NOT also call `analyze_file` on it — that would be redundant.
    - First use: Generates analysis script automatically
    - Subsequent uses: Reuses script for consistency
    - force_regenerate=True: Use when analysis needs change
    - On the first call for a new file, omit `inputs` and `targets`. The scalarizer will automatically
      classify columns into inputs (controllable parameters) and targets (measured outcomes).
    - If `schema_proposed` is returned (co-pilot mode), present the proposed classification to the user for confirmation.
      If the user adjusts, re-call `analyze_file` with the corrected inputs/targets.
    - If `schema_required` is returned (fallback), choose inputs and targets from `available_columns`
      and re-call with explicit values. Do NOT guess column names — only use names from `available_columns`.
    - If `inputs_required` is returned, the data contains only measurement results (e.g., spectra)
      with no experimental conditions. Ask the user to provide the conditions for this data
      (e.g., "What temperature/pH/concentration was this spectrum collected at?").
    - On success: report data_points_collected and optimization_ready. Do NOT repeat the schema back to the user.
8b. `analyze_batch`: Process multiple data files (e.g., spectra at different conditions) in one call.
    - Runs the scalarizer once per file, reusing the same script for consistency.
    - Experimental conditions can come from:
      a) Sidecar JSONs (auto-discovered: spectrum_300C.json next to spectrum_300C.csv)
      b) A conditions file: `analyze_batch(file_paths=[...], conditions_file="conditions.json")`
      c) Inline JSON: `analyze_batch(file_paths=[...], conditions='{"file.csv": {"temp": 300}}')`
    - If no conditions provided, returns `inputs_required` with the file list, extracted targets,
      and an `example_conditions` template the user can fill in.
    - If conditions are provided for only some files, processes those and returns `partial_success`
      with `files_missing_conditions` and an example template for the remaining files.
    - Use instead of calling `analyze_file` in a loop when files share the same data structure.
9. `reset_analysis_logic`: Destroys all analysis data, scripts, and BO history. Use ONLY when the
    user explicitly asks to start over. Do NOT use as error recovery — use `force_regenerate=True` instead.
9b. `query_knowledge_data`: Query knowledge data with natural language.
    - Works with single data files (CSV, XLSX) AND directory databases (folders with many
      uniformly-structured files, e.g., JSON record collections).
    - Generates a Python script, executes it, returns the answer.
    - NOT for experimental results feeding optimization — use analyze_file for that.
    - If file_name omitted, lists available queryable files and detected directory databases.
    - **IMPORTANT**: Query answers are only visible to YOU (the orchestrator).
      Downstream tools (TEA, planning, refinement) cannot see them unless you pass
      them via `additional_context`.
    - Examples:
      * query_knowledge_data(query="average Li concentration in Permian Basin", file_name="PWSdatabase.xlsx")
      * query_knowledge_data(query="find all MOFs with void fraction > 0.8")
      * query_knowledge_data(query="how many unique basins are represented?")

**OPTIMIZATION TOOLS:**
10. `run_optimization`: Mathematical parameter suggestions via Bayesian Optimization.
    
    **Modes:**
    - Sequential: `run_optimization()`
    - Parallel: `run_optimization(parallel_capable=True, batch_size=N)`
      * Infer N from context or ask user. Retry if "batch_size_required" returned.
    - Constraint-aware: `run_optimization(parallel_capable=True, batch_size=N, physical_constraints="...")`
      * Use when the setup has physical limitations (plate layouts, shared channels, discrete stocks).
      * Extract constraints from the plan or user description.
    - Budget-aware: `run_optimization(experimental_budget=K)`
      * K = optimization iterations remaining (including this one). 1 = final shot.
      * Pass when user mentions remaining experiments, budget, or "last round".
      * Combinable with all other modes.
    - Target-narrowing: `run_optimization(targets=["Peak_Area"])`
      * Use when multiple targets were auto-detected but user wants single-objective.
      * Narrows targets inline — no need to re-analyze files.
      * User says "maximize peak area" → targets=["Peak_Area"]

    **After run_optimization succeeds:**
    - ALWAYS display the diagnostics plot to the user using the `plot_path` from the result.
    - Summarize the `inspection` field (convergence trends, model quality, anomalies) in your response.

    **Constraint examples:**
    - User says "96-well plate where rows share temperature"
      → physical_constraints="96-well plate: 8 rows share temperature, 12 columns share pH"
    - User says "we only have 5 catalyst stocks"
      → physical_constraints="Discrete catalyst concentrations: 0.1, 0.5, 1.0, 2.0, 5.0 mM"
    - User says "reactor zones share cooling"
      → physical_constraints="Reactor: zones A,B share cooling, zones C,D share heating"
    
    **Budget examples:**
    - User says "this is our last round"
      → experimental_budget=1
    - User says "we have 3 more runs"
      → experimental_budget=3
    - User says nothing about budget
      → omit experimental_budget (default behavior)
      
11. `save_checkpoint`: Save campaign state. Use after every 3-5 experiments.
12. `read_file`: Read and preview any file (JSON, text, CSV, Excel, scripts, logs, protocols).
    Use to inspect data files BEFORE deciding which tool to use next.
    Renders Excel/CSV as formatted tables (max 50 rows × 40 columns).
    Do not re-read files whose contents were just returned by another tool.

**KNOWLEDGE & SKILL TOOLS:**
12. `synthesize_knowledge`: Distill findings from completed planning iterations into reusable knowledge.
13. `list_knowledge`: Show all active knowledge entries.
14. `clear_knowledge`: Remove active knowledge entries.
15. `graduate_to_skill`: Convert knowledge into a reusable domain skill (.md file) applied to future plans.
16. `update_skill`: Update a graduated skill with new knowledge.

**FILE PATH RULES:**
Assume user runs agent from project directory. For example, when user says "file.csv in data", use "./data/file.csv"

**When user mentions a SPECIFIC filename:**
1. Extract the filename (with or without extension)
2. Pass it to the tool
3. Tool will automatically:
   - Try exact match
   - Try common extensions (.csv, .xlsx, .xls) if no extension provided
   - Search in ./experimental_results, ./data, ./results, ./
   - Suggest corrections for typos using fuzzy matching

**CRITICAL WORKFLOW RULES:**
**Use `run_optimization` (The Math Loop) IF:**
- You are optimizing a well-defined property for the current experimental setup.
- The experiments are running successfully (no failures), and you just need to tune parameters.
- **At least 3 data files have been successfully analyzed** (check by calling list_workspace_files).
- **The objective itself is a Bayesian-optimization / "recommend the next
  experiments" campaign.** Then `run_optimization` IS the deliverable — do NOT
  also call `generate_initial_plan`. BO is a built-in capability: run it
  directly. A separately generated plan would hand-design a batch that
  contradicts (and is inferior to) BO's acquisition-function output.

**Use `iterate_with_results` (The Cognitive Loop) IF:**
- You need to propose a NEW strategy or experimental setup (e.g., "Change catalyst").
- The experiment FAILED (e.g., "Precipitate formed", "Equipment error").
- The result is qualitative (e.g., **Images**, visual observations, logs).
- There are NOT enough data points for numerical optimization yet.

**If user indicates the generated plan is wrong:**
- Common patterns: "That's not what I asked for", "Wrong material", "Focus on X instead"
- Actions:
  - Ask user to confirm: "I generated a plan for [X], but you mentioned [Y]. Should I correct this?" 
  - If user confirms it's wrong:
        a. Call discard_plan(reason="Specific explanation of what was wrong") 
        b. Call generate_initial_plan(...) again with corrected parameters 
- The discarded plan stays in history for transparency but won't appear in reports

**LONG CAMPAIGN MANAGEMENT:**
- Call `save_checkpoint` after every 3-5 experiments
- If conversation becomes very long (>50 messages), suggest user restart with checkpoint

**CODE PROVENANCE TRANSPARENCY:**
- When presenting implementation code to the user, ALWAYS state how it was produced:
  - Via `generate_implementation_code` / `refine_implementation_code` → grounded in the user's Code KB.
  - Via a third-party MCP tool or external service → state the tool name.
  - Written by you directly → state that you wrote it from general knowledge and it was NOT
    grounded in the user's codebase.
- If `generate_implementation_code` or `refine_implementation_code` returns status="error",
  clearly inform the user of the failure and the reason before proceeding with any alternative.

**MCP OUTPUT PERSISTENCE:**
- When an MCP tool returns generated code, protocols, or other text artifacts, call `save_file`
  to persist the output BEFORE calling any other tool.
- Write large artifacts in chunks (`save_file` for the first chunk, `append_file` for the rest);
  a single oversized tool argument can be truncated and lost.

**BEHAVIOR:**
- Extract ALL paths mentioned by user (papers, data, code, reports)
- Extract specific_objective from user's goal/intent
- Combine lab constraints into additional_context (equipment, reagents, pH, budget, etc.)
- Parse tool JSON responses before calling dependent tools
- If status="error", stop and report to user
- Save checkpoint periodically during long campaigns
"""


def get_system_prompt(
    autonomy_level: AutonomyLevel,
    external_tools: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Returns the appropriate system prompt for the given autonomy level.

    Args:
        autonomy_level: Current autonomy level.
        external_tools: Optional list of ``{"name": ..., "description": ...}``
            dicts describing custom/MCP tools registered at runtime.
    """
    directives = {
        AutonomyLevel.CO_PILOT: _CO_PILOT_DIRECTIVE,
        AutonomyLevel.AUTOPILOT: _AUTOPILOT_DIRECTIVE,
        AutonomyLevel.AUTONOMOUS: _AUTONOMOUS_DIRECTIVE,
    }
    prompt = directives[autonomy_level] + _SYSTEM_PROMPT_BODY

    if external_tools:
        lines = ["\n\n**ADDITIONAL TOOLS (user-registered):**"]
        for t in external_tools:
            lines.append(f"- **{t['name']}**: {t.get('description', '')}")
        prompt += "\n".join(lines)

    return prompt



class PlanningOrchestratorAgent:
    """
    Orchestrator agent for coordinating multi-iteration research campaigns.
    
    Manages the full experimental loop with configurable autonomy:
    1. Hypothesis generation (PlanningAgent)
    2. Experiment execution (external)
    3. Result analysis (ScalarizerAgent)
    4. Parameter optimization (BOAgent)
    5. Iteration decisions
    
    Args:
        objective: Research objective description.
        base_dir: Base directory for campaign outputs.
        api_key: API key for the LLM provider.
        model_name: Model name.
        base_url: Base URL for internal proxy endpoint.
        embedding_model: Embedding model name.
        embedding_api_key: API key for the embedding LLM provider.
        futurehouse_api_key: Optional FutureHouse API key for literature search.
        restore_checkpoint: Whether to restore from previous checkpoint.
        autonomy_level: Level of autonomy (CO_PILOT, AUTOPILOT, or AUTONOMOUS).
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        objective: str = "Undefined Research Goal",
        base_dir: str = "./campaign_outputs",
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        embedding_api_key: Optional[str] = None,
        futurehouse_api_key: Optional[str] = None,
        restore_checkpoint: bool = False,
        autonomy_level: AutonomyLevel = AutonomyLevel.CO_PILOT,
        data_dir: Optional[str] = None,
        knowledge_dir: Optional[str] = None,
        code_dir: Optional[str] = None,
        max_iterations: int = 20,
        # Deprecated
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="PlanningOrchestratorAgent"
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
            # LiteLLM mode: ensure embedding_api_key is set
            if embedding_api_key is None:
                embedding_api_key = api_key

        # Store autonomy level
        self.autonomy_level = autonomy_level
        self._enable_human_feedback = self._should_enable_human_feedback()
        logging.info(f"🎛️  Autonomy Level: {autonomy_level.value.upper()}")

        # Validate and store workspace directories
        if autonomy_level in (AutonomyLevel.AUTOPILOT, AutonomyLevel.AUTONOMOUS):
            if data_dir is None:
                raise ValueError(
                    f"data_dir is required for {autonomy_level.value} mode.\n"
                    f"Specify the directory containing experimental results."
                )
            if not Path(data_dir).exists():
                raise ValueError(f"data_dir does not exist: {data_dir}")

        self.data_dir = Path(data_dir) if data_dir else None
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else None
        self.code_dir = Path(code_dir) if code_dir else None

        if self.data_dir:
            logging.info(f"   Data directory: {self.data_dir}")

        self.objective = objective
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.analyzed_files_path = self.base_dir / "analyzed_files.json"
        self.analyzed_files = {}
        
        if self.analyzed_files_path.exists():
            try:
                with open(self.analyzed_files_path, 'r') as f:
                    self.analyzed_files = json.load(f)
            except Exception as e:
                logging.warning(f"Could not load analyzed_files.json: {e}")
                self.analyzed_files = {}

        self.bo_data_path = self.base_dir / "optimization_data.csv"
        self.history_path = self.base_dir / "chat_history.json"
        self.checkpoint_path = self.base_dir / "checkpoint.json"
        
        self.active_scalarizer_script = None
        self.expected_input_columns = None
        self.expected_target_columns = []
        self.target_directions = {}  # e.g. {"Yield": "maximize", "Defect_Density": "minimize"}
        self.expected_input_types = None  # {col: "continuous" | "categorical"} from scalarizer
        self.expected_input_levels = None  # {col: [level0, level1, ...]} for categorical inputs
        self.latest_tea_results = None

        # Per-delegation output isolation. Used only by run_task: the meta
        # agent reuses one planning child across delegations, and the plan
        # tools write fixed filenames (plan.json, tea_analysis.json, ...).
        # Without a per-delegation sub-directory, each delegation would
        # overwrite the previous one's artifacts. Direct `scilink plan` use
        # never calls run_task, so _active_output_subdir stays None and
        # writes land in base_dir exactly as before.
        self._delegation_counter = 0
        self._active_output_subdir = None

        # Per-call tool-iteration cap (handlers read self.max_iterations) and
        # a flag the handlers raise when they hit the cap. chat() returns the
        # warning string as before; run_task reads the flag to flip the
        # delegation result's status from success → error.
        self.max_iterations = max_iterations
        self._last_chat_hit_iter_cap = False

        # Custom tools / MCP state
        self._tool_data_cache: Dict[tuple, Any] = {}
        self._external_tools: List[Dict[str, str]] = []
        self._mcp_connections: Dict[str, Any] = {}

        # Knowledge synthesis state (mirrors AnalysisOrchestratorAgent)
        self.active_knowledge: List[Dict[str, Any]] = []
        self._custom_skills: Dict[str, str] = {}  # name → path
        self._graduated_skill_sources: Dict[str, list] = {}  # skill_name → [knowledge_ids]

        self.message_count = 0
        self.last_checkpoint_message_count = 0
        
        if restore_checkpoint and self.checkpoint_path.exists():
            self._restore_checkpoint()
        
        # --- Init Sub-Agents ---
        print("🤖 Agent: Hiring sub-agents...")
        planner_kwargs = dict(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            futurehouse_api_key=futurehouse_api_key,
            output_dir=str(self.base_dir),
        )
        if self.knowledge_dir:
            planner_kwargs["kb_base_path"] = str(self.knowledge_dir / "default_kb")
        self.planner = PlanningAgent(**planner_kwargs)

        # Literature & Molecules agents (orchestrator-level tools)
        self.lit_agent = None
        self.mol_agent = None
        if futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY"):
            from ..lit_agents.literature_agent import LiteratureSearchAgent
            from ..lit_agents.molecules_agent import MoleculesAgent
            from ..lit_agents.optimize_query import optimize_search_query
            fh_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")
            try:
                self.lit_agent = LiteratureSearchAgent(fh_key, max_wait_time=3000)
                logging.info("✅ Orchestrator: Literature Search Agent initialized.")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialize Literature Agent: {e}")
            try:
                self.mol_agent = MoleculesAgent(fh_key, max_wait_time=3000)
                logging.info("✅ Orchestrator: Molecules Agent initialized.")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialize Molecules Agent: {e}")

        self.scalarizer = ScalarizerAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            output_dir=str(self.base_dir / "scalarizer_outputs")
        )
        self.bo = BOAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            output_dir=str(self.base_dir / "bo_artifacts")
        )

        # --- Initialize Tools Registry ---
        self.tools = OrchestratorTools(self)
        
        # --- Get appropriate system prompt based on autonomy level ---
        system_prompt = get_system_prompt(
            self.autonomy_level, self._external_tools or None
        )
        system_prompt += self._build_workspace_context()
        
        # --- LLM Initialization ---
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
        
        # Store system prompt for OpenAI mode
        self._system_prompt = system_prompt
        
        # --- MEMORY INITIALIZATION ---
        history = self._load_history()
        
        if self.use_openai:
            self.messages = [{"role": "system", "content": system_prompt}]
            if history:
                recent_history = self._trim_history(history, max_messages=100)
                self.messages.extend(recent_history)
        else:
            # LiteLLM: Initialize messages list similar to OpenAI mode
            # We'll handle tool calls manually instead of using chat_session with AFC
            self.messages = [{"role": "system", "content": system_prompt}]
            if history:
                recent_history = self._trim_history(history, max_messages=100)
                self.messages.extend(recent_history)

    def _convert_tools_to_litellm_format(self) -> List[Dict]:
        """
        Convert OpenAI tool schemas to LiteLLM format.
        LiteLLM uses the same format as OpenAI for tools.
        """
        return self.tools.openai_schemas

    def _build_workspace_context(self) -> str:
        """Build workspace context string for system prompt."""
        if self.autonomy_level == AutonomyLevel.CO_PILOT:
            return ""  # Not needed, human will guide
        
        context_parts = ["\n\n**WORKSPACE CONFIGURATION:**"]
        context_parts.append(f"- Research objective: {self.objective}")

        if self.data_dir:
            context_parts.append(f"- Data directory: {self.data_dir}")
        if self.knowledge_dir:
            context_parts.append(f"- Knowledge directory: {self.knowledge_dir}")
        if self.code_dir:
            context_parts.append(f"- Code directory: {self.code_dir}")

        context_parts.append("\nUse these paths directly without asking for confirmation.")
        
        return "\n".join(context_parts)
    
    def _should_enable_human_feedback(self) -> bool:
        """Determines if human feedback should be enabled based on autonomy level."""
        # CO_PILOT and AUTOPILOT surface generated plans/code for human
        # review (accept or request changes) — the AUTOPILOT directive
        # promises this review interface. Only AUTONOMOUS runs without it.
        return self.autonomy_level != AutonomyLevel.AUTONOMOUS

    def set_autonomy_level(self, level: AutonomyLevel) -> None:
        """
        Change the autonomy level at runtime.
        
        Args:
            level: New autonomy level to set.
        """
        old_level = self.autonomy_level
        self.autonomy_level = level
        self._enable_human_feedback = self._should_enable_human_feedback()
        
        # Update system prompt
        new_system_prompt = get_system_prompt(
            level, self._external_tools or None
        )
        self._system_prompt = new_system_prompt

        # Update system message in messages list (works for both OpenAI and LiteLLM now)
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = new_system_prompt
        
        logging.info(f"🔄 Autonomy level changed: {old_level.value} → {level.value}")
        logging.info(f"   Human feedback enabled: {self._enable_human_feedback}")

    def get_human_feedback_setting(self) -> bool:
        """Returns current human feedback setting for sub-agents."""
        return self._enable_human_feedback

    # ── Skill registration ─────────────────────────────────────────────

    def register_skill(self, skill_path: str) -> str:
        """Register a graduated skill for use in subsequent plan generation.

        Args:
            skill_path: Path to the skill ``.md`` file.

        Returns:
            The skill name (file stem).
        """
        path = Path(skill_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")

        name = path.stem
        self._custom_skills[name] = str(path)

        # Make it the active skill for subsequent generate_initial_plan calls
        self._active_skill = str(path)

        # Surface the new skill to the orchestrator LLM's generate_initial_plan tool
        self.tools._update_skill_description(self._custom_skills)

        logging.info(f"📖 Registered planning skill: {name} → {path}")
        return name

    # ── Custom tools ──────────────────────────────────────────────────

    def _rebuild_system_prompt(self) -> None:
        """Rebuild the system prompt and update the message history."""
        self._system_prompt = get_system_prompt(
            self.autonomy_level, self._external_tools or None
        )
        self._system_prompt += self._build_workspace_context()
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = self._system_prompt

    def _load_data_for_factory(self, data_path: str):
        """Load a data file for use by external tool factories.

        Returns a pandas DataFrame for tabular files, a NumPy array for .npy,
        or the path string itself as a fallback.
        """
        path = Path(data_path)
        ext = path.suffix.lower()
        if ext in {'.csv', '.txt', '.tsv', '.xlsx', '.xls'}:
            try:
                if ext in {'.xlsx', '.xls'}:
                    return pd.read_excel(str(path))
                return pd.read_csv(str(path))
            except Exception:
                return data_path
        elif ext == '.npy':
            import numpy as np
            return np.load(str(path))
        else:
            logging.warning(
                f"_load_data_for_factory: unknown extension '{ext}' — "
                "passing path string to factory."
            )
            return data_path

    def register_tools(self, schemas: list, factory: callable) -> None:
        """Register external tool functions into the orchestrator's LLM loop.

        The ``factory`` callable is invoked lazily at tool-call time.  The
        orchestrator inspects the *name* of the factory's first positional
        parameter to decide what to pass:

        * ``data_path`` / ``path`` / ``file`` / ``filepath`` / ``filename``
          → the current optimization CSV path (or data directory) as a string.
        * Any other name (e.g. ``data``, ``df``, ``dataframe``)
          → the optimization CSV is loaded as a pandas DataFrame and passed.

        If the factory declares an ``output_dir`` parameter the orchestrator
        passes ``<base_dir>/custom_tools/``.

        Args:
            schemas: List of OpenAI-format tool schemas.
            factory: Callable that accepts data and returns a dict mapping tool
                names to callables.
        """
        import inspect

        sig = inspect.signature(factory)
        params = list(sig.parameters.keys())
        first_param = params[0] if params else None
        _path_param_names = {
            'data_path', 'path', 'file', 'filepath', 'file_path', 'filename',
        }
        factory_takes_path = first_param in _path_param_names
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
                    # Determine what data source to pass.
                    data_path = str(self.bo_data_path) if self.bo_data_path.exists() else None
                    if data_path is None and self.data_dir is not None:
                        data_path = str(self.data_dir)
                    if data_path is None:
                        return json.dumps({
                            "status": "error",
                            "message": (
                                "No data available. Load experimental data "
                                "or provide --data-dir first."
                            ),
                        })
                    output_dir = self.base_dir / "custom_tools"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cache_key = (data_path, id(_factory), str(output_dir))
                    if cache_key not in self._tool_data_cache:
                        if _takes_path:
                            data = data_path
                        else:
                            data = self._load_data_for_factory(data_path)
                        if _takes_output_dir:
                            self._tool_data_cache[cache_key] = _factory(
                                data, output_dir=str(output_dir)
                            )
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
                func=_make_wrapper(
                    tool_name, factory,
                    factory_takes_path, factory_takes_output_dir,
                ),
                name=tool_name,
                description=description,
                parameters=properties,
                required=required,
            )
            self._external_tools.append({
                "name": tool_name, "description": description,
            })
            registered += 1

        logging.info(f"✅ Registered {registered} external tool(s)")
        self._rebuild_system_prompt()

    # ── MCP server integration ────────────────────────────────────────

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
            command: Command + args for stdio transport.
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
        logging.info(f"✅ MCP '{server_name}': registered {registered} tool(s)")
        self._rebuild_system_prompt()
        return registered

    def disconnect_mcp_server(self, server_name: str) -> None:
        """Disconnect from an MCP server and unregister its tools."""
        conn = self._mcp_connections.pop(server_name, None)
        if conn is None:
            logging.warning(f"MCP server '{server_name}' not found.")
            return

        conn.disconnect()

        prefix = f"[MCP:{server_name}]"
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

        logging.info(f"🔌 Disconnected MCP server '{server_name}'")
        self._rebuild_system_prompt()

    def _restore_checkpoint(self):
        """Restore campaign state from checkpoint."""
        print(f"  📂 Restoring checkpoint from: {self.checkpoint_path}")
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            self.active_scalarizer_script = state.get("active_scalarizer_script")
            self.expected_input_columns = state.get("expected_input_columns")

            if "expected_target_columns" in state:
                self.expected_target_columns = state.get("expected_target_columns")
            else:
                self.expected_target_columns = []

            self.target_directions = state.get("target_directions", {})
            self.expected_input_types = state.get("expected_input_types")
            self.expected_input_levels = state.get("expected_input_levels")
            self.latest_tea_results = state.get("latest_tea_results")
            self._delegation_counter = state.get("delegation_counter", 0)

            # Restore autonomy level if saved
            if "autonomy_level" in state:
                try:
                    self.autonomy_level = AutonomyLevel(state["autonomy_level"])
                    self._enable_human_feedback = self._should_enable_human_feedback()
                except ValueError:
                    pass  # Keep default if invalid value

            if "data_dir" in state and state["data_dir"]:
                self.data_dir = Path(state["data_dir"])
            if "knowledge_dir" in state and state["knowledge_dir"]:
                self.knowledge_dir = Path(state["knowledge_dir"])
            if "code_dir" in state and state["code_dir"]:
                self.code_dir = Path(state["code_dir"])

            # Restore knowledge/skill state
            self.active_knowledge = state.get("active_knowledge", [])
            self._graduated_skill_sources = state.get("graduated_skill_sources", {})
            restored_skills = state.get("custom_skills", {})
            for skill_name, skill_path in restored_skills.items():
                if Path(skill_path).exists():
                    self._custom_skills[skill_name] = skill_path

            print(f"    ✅ Restored state:")
            print(f"       - Analysis script: {Path(self.active_scalarizer_script).name if self.active_scalarizer_script else 'None'}")
            print(f"       - Schema: {self.expected_input_columns} → {self.expected_target_columns}")
            print(f"       - Data points: {state.get('data_points_collected', 0)}")
            print(f"       - Autonomy level: {self.autonomy_level.value}")
            
        except Exception as e:
            logging.warning(f"Failed to restore checkpoint: {e}")

    def _trim_history(self, history: List[Dict], max_messages: int = 100) -> List[Dict]:
        """Keep only recent messages to avoid context window overflow."""
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
        
        # AUTO-CHECKPOINT: Every 10 messages
        if self.message_count - self.last_checkpoint_message_count >= 10:
            print("  💾 Auto-checkpoint triggered (every 10 messages)...")
            self._auto_checkpoint()
            self.last_checkpoint_message_count = self.message_count
        
        try:
            if self.use_openai:
                response_text = self._handle_openai_chat(user_input)
            else:
                # Use the same manual tool handling approach for LiteLLM
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
                 autonomy: Optional[AutonomyLevel] = None,
                 max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Non-interactive entry point — used by the meta agent.

        Runs the task and returns a structured summary that's easy to
        consume programmatically:

            {
                "status": "success" | "error",
                "task": str,                       # echoed input
                "summary": str,                    # the agent's final reply
                "files_produced": List[str],       # absolute paths
                "key_findings": List[str],         # campaign configuration
                "suggested_followups": List[str],
                "campaign_state": dict,            # objective / targets / data
                "warnings": List[str],
            }

        Mirrors SimulationOrchestratorAgent.run_task — see CLAUDE.md
        "Two surfaces, one agent".

        ``autonomy`` selects the AutonomyLevel for this call. Defaults to
        AUTONOMOUS — the safe choice for a headless/programmatic caller, so
        the agent never pauses for a nonexistent user. A caller attached to a
        human (the meta agent, driven via CLI/UI) passes AUTOPILOT so the
        sub-agents' human-feedback prompts reach that human.
        The original autonomy level is restored on exit, even if chat()
        raises.

        Planning has no analysis-style record list, so files_produced is a
        before/after diff of the campaign directory (session-churn files
        like chat_history.json are excluded).
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

        def _snapshot_files() -> set:
            """Resolved paths of campaign artifacts, minus session churn."""
            skip_names = {
                "chat_history.json", "checkpoint.json",
                "analyzed_files.json", "session_log.txt",
            }
            snap = set()
            if self.base_dir.is_dir():
                for p in self.base_dir.rglob("*"):
                    if (p.is_file() and p.name not in skip_names
                            and p.suffix not in (".log", ".pyc")):
                        snap.add(str(p.resolve()))
            return snap

        # Snapshot prior state so we report "what was produced *during* this
        # call" rather than "everything in the session."
        files_before = _snapshot_files()

        # Isolate this delegation's plan artifacts in their own sub-directory
        # so a persistent planning child (reused by the meta agent) does not
        # overwrite an earlier delegation's plan.json / tea_analysis /
        # output_scripts. The plan tools read self._active_output_subdir.
        self._delegation_counter += 1
        slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:40] or "task"
        self._active_output_subdir = (
            self.base_dir / "delegations" / f"{self._delegation_counter:02d}_{slug}"
        )
        self._active_output_subdir.mkdir(parents=True, exist_ok=True)

        # Run under the requested autonomy level — AUTONOMOUS by default (the
        # safe headless choice). The meta agent passes its own mode through,
        # so a co-pilot / autopilot delegation still raises the sub-agents'
        # human-feedback prompts to the user driving the session.
        run_level = autonomy if autonomy is not None else AutonomyLevel.AUTONOMOUS
        original_level = self.autonomy_level
        original_max_iter = self.max_iterations
        try:
            self.set_autonomy_level(run_level)
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
                logging.exception(f"run_task failed: {e}")
                summary_text = ""
                status = "error"
                error_msg = str(e)
        finally:
            # Always restore the original level, even if chat() raised.
            self.set_autonomy_level(original_level)
            self.max_iterations = original_max_iter
            # _active_output_subdir is scoped to this call; clear it so a
            # later direct chat() on the same instance writes to base_dir.
            self._active_output_subdir = None

        files_produced = sorted(_snapshot_files() - files_before)

        # data_points_collected: rows in the BO optimization table, if any.
        data_points = 0
        try:
            if self.bo_data_path.exists():
                data_points = len(pd.read_csv(self.bo_data_path))
        except Exception:
            data_points = 0

        # key_findings: the campaign configuration the orchestrator settled
        # on. Planning's substantive output is the plan text itself, which
        # the caller reads from `summary`.
        key_findings: List[str] = []
        for col in self.expected_target_columns or []:
            direction = self.target_directions.get(col, "optimize")
            key_findings.append(f"Optimization target: {col} ({direction}).")
        if self.latest_tea_results:
            key_findings.append("Techno-economic analysis results are available.")

        # Heuristic: collected BO data with no further direction is a natural
        # cue to propose the next experiment batch.
        suggested_followups: List[str] = []
        if data_points > 0:
            suggested_followups.append(
                f"Run the next BO iteration ({data_points} data point(s) "
                f"collected so far)."
            )

        warnings: List[str] = []
        if status == "success" and not files_produced:
            warnings.append("Task completed but produced no campaign artifacts.")

        # Surface the advisory critic's caveats so a headless / meta consumer
        # sees them even though no human reviewed them in-session. The plan is
        # not modified — these are limitations for the caller to weigh.
        _plan = (self.planner.state or {}).get("current_plan") or {}
        for _c in format_caveats(_plan.get("critic_findings")):
            warnings.append(f"Plan caveat — {_c}")

        result = {
            "status": status,
            "task": task,
            "summary": summary_text,
            "files_produced": files_produced,
            "key_findings": key_findings,
            "suggested_followups": suggested_followups,
            "campaign_state": {
                "objective": self.objective,
                "expected_input_columns": self.expected_input_columns,
                "expected_target_columns": self.expected_target_columns,
                "target_directions": self.target_directions,
                "data_points_collected": data_points,
            },
            "warnings": warnings,
        }
        if error_msg:
            result["error"] = error_msg
        return result

    def _compress_large_tool_results(self):
        """Compress large tool results in chat history to prevent context overflow.

        Only triggers when total history exceeds 100K chars. Truncates tool
        results larger than 30K chars to 5K chars, preserving the file_path
        for re-reading. Skips the 2 most recent messages to avoid truncating
        results the LLM hasn't responded to yet.
        """
        total_chars = sum(len(m.get("content", "") or "") for m in self.messages)
        if total_chars > 100000:
            compressed = 0
            for msg in self.messages[:-2]:
                if msg["role"] == "tool" and len(msg.get("content", "")) > 30000:
                    original_len = len(msg["content"])
                    msg["content"] = (
                        msg["content"][:5000]
                        + f"\n\n... ({original_len - 5000} chars truncated from history. "
                        f"Use read_file to re-read the full content only if "
                        f"the truncated portion above is insufficient for your current task.)"
                    )
                    compressed += 1
            if compressed:
                new_total = sum(len(m.get("content", "") or "") for m in self.messages)
                print(f"  📦 Compressed {compressed} large tool result(s) in history "
                      f"({total_chars:,} → {new_total:,} chars)")

    def _auto_checkpoint(self):
        """Internal auto-checkpoint without LLM interaction."""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "objective": self.objective,
                "active_scalarizer_script": self.active_scalarizer_script,
                "expected_input_columns": self.expected_input_columns,
                "expected_target_columns": self.expected_target_columns,
                "target_directions": self.target_directions,
                "expected_input_types": self.expected_input_types,
                "expected_input_levels": self.expected_input_levels,
                "data_points_collected": len(pd.read_csv(self.bo_data_path)) if self.bo_data_path.exists() else 0,
                "planner_state": self.planner.state,
                "message_count": self.message_count,
                "latest_tea_results": self.latest_tea_results,
                "delegation_counter": self._delegation_counter,
                "autonomy_level": self.autonomy_level.value,
                "data_dir": str(self.data_dir) if self.data_dir else None,
                "knowledge_dir": str(self.knowledge_dir) if self.knowledge_dir else None,
                "code_dir": str(self.code_dir) if self.code_dir else None,
                "active_knowledge": self.active_knowledge,
                "graduated_skill_sources": self._graduated_skill_sources,
                "custom_skills": self._custom_skills,
            }
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            print(f"    ✅ Auto-checkpoint saved")
            
        except Exception as e:
            logging.warning(f"Auto-checkpoint failed: {e}")

    @staticmethod
    def _parse_tool_args(tool_call, finish_reason=None):
        """Parse a tool call's JSON arguments, failing loud on bad input.

        Returns (args, None) on success, or (None, error_json) when the
        arguments string is malformed or truncated. The error_json is handed
        back to the model as the tool result so it can recover (#270) —
        silently substituting ``args = {}`` surfaces as a raw TypeError from
        the tool itself, which hides the real cause and sends the model into
        an unrecoverable retry loop.
        """
        try:
            return json.loads(tool_call.function.arguments), None
        except json.JSONDecodeError:
            if finish_reason == "length":
                cause = ("the arguments JSON was truncated — the response "
                         "hit the output-token limit")
            else:
                cause = ("the arguments string was not valid JSON — "
                         "typically broken escaping of quotes/newlines "
                         "inside a large string value")
            return None, json.dumps({
                "status": "error",
                "message": (
                    f"Tool call discarded: {cause}. The tool was NOT "
                    "executed. Do not retry with one large argument. For "
                    "large text content, write the file in chunks: "
                    "save_file with the first chunk, then append_file for "
                    "each remaining chunk."
                ),
            })

    def _handle_openai_chat(self, user_input: str) -> str:
        """Handle chat with OpenAI-compatible models with manual function calling loop."""
        from openai import OpenAI
        
        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url
        )
        
        self.messages.append({"role": "user", "content": user_input})
        
        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=100)
            self.messages = [system_msg] + recent_msgs

        self._compress_large_tool_results()

        max_iterations = self.max_iterations
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Also compress within the loop — tool calls can grow history mid-conversation
            if iteration > 1:
                self._compress_large_tool_results()

            print(f"  ⏳ Waiting for orchestrator response ...")

            response = client.chat.completions.create(
                model=self.model.model,
                messages=self.messages,
                tools=self.tools_for_model,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if not message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content

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

            finish_reason = getattr(response.choices[0], "finish_reason", None)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args, args_error = self._parse_tool_args(tool_call, finish_reason)

                print(f"  🔧 Calling tool: {func_name}")

                if args_error is not None:
                    print("    ⚠️ Malformed/truncated tool arguments — "
                          "returning recovery hint to the model")
                    result = args_error
                else:
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
        deliberate step reads like a silent loop. The text is already in
        history; this just shows it so the user can see *why* the next tool is
        being called. Rendered dim+italic and set off by blank lines so the
        prose is visually distinct from the structural tool-call log.
        """
        if not content:
            return
        text = content.strip() if isinstance(content, str) else str(content).strip()
        if not text:
            return
        import sys
        style, reset = (("\033[2;3;36m", "\033[0m")
                        if sys.stdout.isatty() else ("", ""))
        body = text.replace("\n", "\n     ")  # indent continuation lines
        # 💭 stays the reserved reasoning marker; a meta-delegated run tags its
        # reasoning with the SAME emoji the delegation banner uses (📋 planning)
        # so the specialist's thoughts are distinguishable from the meta's own
        # 💭 where they interleave in a meta session. A standalone session
        # (default label) prints a plain 💭, unchanged. The tag rides INSIDE the
        # 💭 line so 📋 — heavily used elsewhere as a plan/report glyph — never
        # itself triggers thought-styling. Mirrors `_print_agent_answer`'s
        # label-based 🤖 attribution.
        tag = "📋 " if getattr(self, "_agent_label", "Agent") != "Agent" else ""
        print(f"\n  {style}💭 {tag}{body}{reset}\n")

    def _print_agent_answer(self, text) -> None:
        """Print the agent's final answer — a deliverable, emphasized (bold +
        bright) to stand apart from 💭 reasoning (a dim aside) and structural
        logs. `_agent_label` (default "Agent") lets the meta name the delegated
        specialist (e.g. "Planning specialist") so a child's answer is not
        mistaken for the meta's own user-facing response."""
        import sys
        label = getattr(self, "_agent_label", "Agent")
        bold, reset = ("\033[1;96m", "\033[0m") if sys.stdout.isatty() else ("", "")
        print(f"\n{bold}🤖 {label}:{reset}")
        print(text if text is not None else "")

    def _handle_litellm_chat(self, user_input: str) -> str:
        """Handle chat with LiteLLM models with manual function calling loop."""
        import litellm
        from ...wrappers.litellm_wrapper import litellm_completion
        
        self.messages.append({"role": "user", "content": user_input})
        
        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=100)
            self.messages = [system_msg] + recent_msgs

        self._compress_large_tool_results()

        max_iterations = self.max_iterations
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Also compress within the loop — tool calls can grow history mid-conversation
            if iteration > 1:
                self._compress_large_tool_results()

            print(f"  ⏳ Waiting for orchestrator response ...")

            response = litellm_completion(
                model=self.model.model,
                messages=self.messages,
                tools=self.tools_for_model,
                tool_choice="auto",
                api_key=self.model.api_key,
                api_base=self.model.base_url
            )
            
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            content = getattr(message, "content", None)
            
            if not tool_calls:
                # No tool calls - return the text response
                self.messages.append({
                    "role": "assistant",
                    "content": content or ""
                })
                return content or ""
            
            # Has tool calls - add assistant message with tool calls
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

            finish_reason = getattr(response.choices[0], "finish_reason", None)

            # Execute each tool call
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                args, args_error = self._parse_tool_args(tool_call, finish_reason)

                print(f"  🔧 Calling tool: {func_name}")

                if args_error is not None:
                    print("    ⚠️ Malformed/truncated tool arguments — "
                          "returning recovery hint to the model")
                    result = args_error
                else:
                    result = self.tools.execute_tool(func_name, **args)
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        self._last_chat_hit_iter_cap = True
        return "⚠️ Maximum tool iterations reached. Please simplify your request."

    def _extract_response_text(self, response) -> str:
        """Robustly extract text from different response formats."""
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            text_parts = [p.text for p in response.parts if hasattr(p, 'text')]
            return ' '.join(text_parts)
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def _load_history(self) -> List[Dict]:
        """Load conversation history from disk."""
        if not self.history_path.exists(): 
            return []
        print("  🧠 Memory: Loading previous conversation...")
        try:
            with open(self.history_path, 'r') as f: 
                saved = json.load(f)
            
            # Both OpenAI and LiteLLM now use the same message format
            return saved
            
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")
            return []

    def _save_history(self):
        """Save conversation history to disk."""
        try:
            # Filter out system messages for saved history
            history_data = [m for m in self.messages if m["role"] != "system"]
            
            with open(self.history_path, 'w') as f: 
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logging.warning(f"Failed to save history: {e}")

    @classmethod
    def restore_from_checkpoint(cls, base_dir: str, **kwargs):
        """Factory method to create an OrchestratorAgent from a checkpoint."""
        return cls(base_dir=base_dir, restore_checkpoint=True, **kwargs)