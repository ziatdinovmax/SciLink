import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .planning_agent import PlanningAgent
from .scalarizer_agent import ScalarizerAgent
from .bo_agent import BOAgent
from .orchestrator_tools import OrchestratorTools
from ._deprecation import normalize_params


class AutonomyLevel(Enum):
    """
    Defines the level of autonomy for the orchestrator.
    
    CO_PILOT: AI assists human (default). Human reviews all plans/code.
    SUPERVISED: Human assists AI. AI proceeds unless human intervenes.
    AUTONOMOUS: Full autonomy. No human feedback requested.
    """
    CO_PILOT = "co_pilot"       # Human leads, AI assists (current default)
    SUPERVISED = "supervised"   # AI leads, human can intervene
    AUTONOMOUS = "autonomous"   # Full autonomy, no human feedback


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

_SUPERVISED_DIRECTIVE = """
**CRITICAL OPERATING MODE: SUPERVISED (AI Leads, Human Supervises)**
- You lead the research workflow. Human supervises and can intervene.
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

**TEA-FIRST RULE:** Run `run_economic_analysis` BEFORE `generate_initial_plan` when the
objective or subject implies economic relevance — e.g., critical materials recovery,
process scale-up, cost optimization, resource extraction, manufacturing, market-driven
material selection, or any goal where viability/profitability matters. TEA results are
automatically injected into the plan, producing a more grounded strategy.
Do NOT run TEA for purely scientific exploration (e.g., "study phase transitions",
"characterize this sample", "explore structure-property relationships").

1. `generate_initial_plan`: Use this when starting a NEW campaign or defining a new objective.
   - Extract knowledge_paths when user mentions papers/PDFs/documents
   - Extract primary_data_set when user mentions experimental data or results folders or files
   - additional_context: Lab constraints, equipment, reagents, budget
   - Previous TEA results automatically included when available
   - Example:
     * "Generate plan for Li recovery using info in ./papers/ and preliminary results in ./data/"
       → generate_initial_plan(specific_objective="Li recovery",
                               knowledge_paths="./papers",
                               primary_data_set="./data")

2. `run_economic_analysis`: Assess economic viability, costs, market fit.
    - Run BEFORE generate_initial_plan when the objective is economically motivated.
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


**DATA TOOLS:**
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
        AutonomyLevel.SUPERVISED: _SUPERVISED_DIRECTIVE,
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
        autonomy_level: Level of autonomy (CO_PILOT, SUPERVISED, or AUTONOMOUS).
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        objective: str = "Undefined Research Goal",
        base_dir: str = "./campaign_outputs",
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.1-pro-preview",
        base_url: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        embedding_api_key: Optional[str] = None,
        futurehouse_api_key: Optional[str] = None,
        restore_checkpoint: bool = False,
        autonomy_level: AutonomyLevel = AutonomyLevel.CO_PILOT,
        data_dir: Optional[str] = None,
        knowledge_dir: Optional[str] = None,
        code_dir: Optional[str] = None,
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
        if autonomy_level in (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS):
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
        self.latest_tea_results = None

        # Custom tools / MCP state
        self._tool_data_cache: Dict[tuple, Any] = {}
        self._external_tools: List[Dict[str, str]] = []
        self._mcp_connections: Dict[str, Any] = {}

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
        # Only CO_PILOT pauses for human review
        # SUPERVISED and AUTONOMOUS proceed without asking
        return self.autonomy_level == AutonomyLevel.CO_PILOT

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
            self.latest_tea_results = state.get("latest_tea_results")
            
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
            
            print(f"🤖 Agent: {response_text}")
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
                "data_points_collected": len(pd.read_csv(self.bo_data_path)) if self.bo_data_path.exists() else 0,
                "planner_state": self.planner.state,
                "message_count": self.message_count,
                "latest_tea_results": self.latest_tea_results,
                "autonomy_level": self.autonomy_level.value,
                "data_dir": str(self.data_dir) if self.data_dir else None,
                "knowledge_dir": str(self.knowledge_dir) if self.knowledge_dir else None,
                "code_dir": str(self.code_dir) if self.code_dir else None,
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
            base_url=self.model.base_url
        )
        
        self.messages.append({"role": "user", "content": user_input})
        
        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=100)
            self.messages = [system_msg] + recent_msgs
        
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1

            print(f"  ⏳ Waiting for LLM response ...") 
            
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
        
        return "⚠️ Maximum tool iterations reached. Please simplify your request."

    def _handle_litellm_chat(self, user_input: str) -> str:
        """Handle chat with LiteLLM models with manual function calling loop."""
        import litellm
        
        self.messages.append({"role": "user", "content": user_input})
        
        if len(self.messages) > 120:
            print("  ⚠️  Context window getting full - trimming history...")
            system_msg = self.messages[0]
            recent_msgs = self._trim_history(self.messages[1:], max_messages=100)
            self.messages = [system_msg] + recent_msgs
        
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1

            print(f"  ⏳ Waiting for LLM response ...") 
            
            response = litellm.completion(
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
            
            # Execute each tool call
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