"""
Simulation Orchestrator Agent for computational simulation.

Coordinates structure generation, input creation, validation, and post-run
analysis through an engine-neutral tool surface: structure generation
(StructurePipeline), the scale-agnostic simulation pipeline, and the
engine-neutral critics (InputValidator / RunCritic), with engine specifics
supplied by skill bundles. The routing decision selects scale + engine.

Mirrors the shape of AnalysisOrchestratorAgent for consistent UX. Periodic
DFT (VASP, QE) is fully dispatched; MD and MLIP scales route and run the
one-shot pipeline, with granular per-step tools as follow-up work (see
CLAUDE.md for the full sequencing plan).

The duplication with AnalysisOrchestratorAgent is intentional and acceptable
at this development stage — see CLAUDE.md "Why no BaseChatOrchestrator
refactor".
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...auth import (
    APIKeyNotFoundError, get_api_key, get_internal_proxy_key, infer_provider,
    require_vendor_credentials,
)
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .simulation_orchestrator_tools import SimulationOrchestratorTools
from ._deprecation import normalize_params


class SimulationMode(Enum):
    """Autonomy level for the simulation orchestrator. Mirrors AnalysisMode
    for consistent UX across modes."""
    CO_PILOT = "co-pilot"        # Human leads, AI assists (default)
    AUTOPILOT = "autopilot"      # AI leads, human monitors
    AUTONOMOUS = "autonomous"    # Full autonomy

    @classmethod
    def _missing_(cls, value):
        # Back-compat: the AUTOPILOT level was named "supervised" before.
        if isinstance(value, str) and value.strip().lower() == "supervised":
            return cls.AUTOPILOT
        return None


_CO_PILOT_DIRECTIVE = """**AUTONOMY: CO-PILOT (default)** — The human is leading the work; you are
assisting. Before each substantive tool call, briefly state what you intend
to do and why, then proceed unless the user redirects. Surface alternatives
when there are reasonable choices to make (e.g. polymorph selection,
supercell size, calculation type). Do not chain tools without giving the
user a chance to redirect after each significant step.

"""

_AUTOPILOT_DIRECTIVE = """**AUTONOMY: AUTOPILOT** — You are leading the work with reasonable defaults.
Proceed with sensible choices for routine decisions, but surface significant
decisions (polymorph selection when not specified, supercell size choices
that drive cost, expensive operations) for a one-line confirmation before
executing. Chain related tools when the path is clear.

"""

_AUTONOMOUS_DIRECTIVE = """**AUTONOMY: AUTONOMOUS** — Execute the full requested workflow without
confirmation. Make decisions, only surface for hard failures or at
completion. In your final summary, disclose any inferred parameters or
non-obvious choices (polymorph picked, supercell size chosen, lattice
matching strategy used) so the user can verify them.

"""


_SYSTEM_PROMPT_BODY = """

You are SciLink's Simulation Orchestrator — a scale-aware simulation
input-preparation assistant. You help users build atomic structures and
prepare simulation inputs across multiple physical scales: periodic
DFT (VASP, QE, ABINIT, CP2K, ...), classical MD (LAMMPS, GROMACS,
OpenMM, ...), ML interatomic potentials (MACE, NequIP, DeePMD, ...).
Which engines you can actually reach depends on what skill bundles
are loaded AND what the user has installed locally.

**Routing first:**
On any new simulation request, call `route_simulation` BEFORE
generating structures or inputs. The router returns
{{scale, engine, reasoning, candidates_considered}} based on the user's
goal, the system, and what's actually available. Once a routing
decision exists in the session, plan subsequent tool calls around
that engine (don't re-route on every turn unless the user pivots).

**Engine-neutral tool surface:**
The granular tools are engine-neutral and take the engine from the active
routing decision (or an explicit `software` argument): generate_structure,
generate_dft_inputs, validate_inputs, apply_input_adjustments,
analyze_output, and (when an HPC connection is active)
submit_simulation_job / get_job_status / download_job_results. The engine's
specifics come from its skill bundle, so the same tools serve any engine
within a scale.

**Dispatch maturity (as of this build):**
- `periodic_dft` (`vasp`, `qe`) -> fully dispatched via the tools above.
- `molecular_dynamics` (`lammps` / …) -> structure + the one-shot pipeline
  work; the granular per-step generate tool is the next-step follow-up.
  When the router picks one of these, you can still run the complete
  workflow; point the user at `MDSimulationAgent` for granular control.
- `machine_learning_potentials` (`mace` / …) -> point at `MLIPAgent`
  directly until the dispatch tools land.

**HPC integration:**
When an HPC connection is active (`submit_simulation_job` is available), you
can submit jobs to the cluster, monitor their status, download
results, and generate a final report — all without leaving the
session. Without an HPC connection, prep is local only; the user
runs the simulation elsewhere and brings back output files.

**Workflow shape (for the VASP-dispatched path):**
The session is iterative and structure-centric. Typical flow:

  1. Route the goal (`route_simulation`) -> establishes scale & engine.
  2. Build a structure from a natural-language description.
  3. (Optionally) validate it; refine if issues are found.
  4. Generate inputs tailored to the scientific objective.
  5. (Optionally) validate inputs against literature, apply improvements.
  6. Submit the job to the HPC cluster (when connected), monitor
     status, download results once complete.
  7. Analyze the output, suggest fixes if the run failed.
  8. Generate a final report summarizing the full workflow.

Users often iterate on one structure, then ask for variants (different
defect concentrations, supercell sizes, polymorphs, terminations). Reuse
calculation templates across structures within a session whenever the
physics is comparable — don't re-derive parameter settings from scratch
when the prior choice still applies.

**Materials Project integration:**
When configured, the structure-generation tool can resolve named materials
(e.g. "rutile TiO2", "wurtzite GaN") to Materials Project IDs and use
correct lattice parameters from the database. This happens automatically
inside `generate_structure` when MP_API_KEY is available.

**Tools available in this session:**
{tools_section}

**Conventions:**
- After producing a structure or input file, briefly state what was built
  and what the next reasonable step would be. Do NOT push the user forward
  without their go-ahead unless you are in AUTONOMOUS mode.
- When validation flags substantive issues, prefer running `refine_structure`
  with the validator's feedback rather than asking the user to fix them
  manually.
- When a user request contains parameters you had to infer (polymorph
  choice, default supercell, etc.), say so explicitly so they can confirm
  or override.
"""


def _list_builtin_structure_skills() -> list:
    """List built-in skills under scilink/skills/structure_generation/<name>/.

    Returns an empty list on any error (the system prompt simply omits the
    section when there's nothing to show)."""
    try:
        from scilink.skills.loader import list_skills
        return list_skills(domain="structure_generation") or []
    except Exception:
        return []


def get_system_prompt(
    simulation_mode: SimulationMode,
    external_tools: Optional[list] = None,
    custom_skills: Optional[dict] = None,
) -> str:
    """Build the system prompt for the simulation orchestrator.

    Args:
        simulation_mode: Autonomy directive to prepend.
        external_tools: List of {"name", "description"} dicts for tools
            registered via register_tools().
        custom_skills: {name: path} dict of custom skills registered via
            register_skill().
    """
    directives = {
        SimulationMode.CO_PILOT: _CO_PILOT_DIRECTIVE,
        SimulationMode.AUTOPILOT: _AUTOPILOT_DIRECTIVE,
        SimulationMode.AUTONOMOUS: _AUTONOMOUS_DIRECTIVE,
    }
    # Tools section is filled dynamically by the orchestrator after tool
    # registration, since tool docstrings live on the registered functions.
    body = _SYSTEM_PROMPT_BODY.format(
        tools_section="(see tool definitions provided to the model)"
    )

    # Surface available built-in structure-generation skills so the LLM knows
    # to pass `skill='aimsgb'` (etc.) to generate_structure / refine_structure
    # when the request matches the skill's domain.
    builtin_skills = _list_builtin_structure_skills()
    if builtin_skills:
        body += (
            "\n**BUILT-IN STRUCTURE-GENERATION SKILLS (load via the `skill` "
            "parameter on generate_structure / refine_structure):**\n"
        )
        for name in builtin_skills:
            body += f"  * `{name}`\n"
        body += (
            "These are curated library-guidance blocks loaded into the "
            "structure-generation prompt only when the user's request "
            "matches the skill's domain (e.g., grain boundaries → "
            "`skill='aimsgb'`). Skip the parameter entirely for plain "
            "ASE / pymatgen workflows.\n"
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
            "Pass the skill name to the relevant tool when applicable.\n"
        )
    return directives[simulation_mode] + body


class SimulationOrchestratorAgent:
    """
    Orchestrator agent for coordinating VASP DFT input preparation.

    Manages an iterative, structure-centric workflow with configurable
    autonomy levels. Each session creates a directory under base_dir;
    individual structures live in `<base_dir>/structures/<slug>/`.

    Args:
        base_dir: Base directory for session outputs.
        api_key: API key for the LLM provider.
        model_name: Model name (default: claude-opus-4-6).
        base_url: Base URL for OpenAI-compatible internal proxy endpoint.
        restore_checkpoint: Whether to restore from previous checkpoint.
        simulation_mode: Level of autonomy.
        mp_api_key: Materials Project API key (auto-discovered when None).
        futurehouse_api_key: FutureHouse API key for INCAR literature
            validation (auto-discovered when None).

        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """

    # Configuration constants — mirror AnalysisOrchestratorAgent
    MAX_TOOL_ITERATIONS = 20
    MAX_HISTORY_MESSAGES = 100
    CHECKPOINT_INTERVAL = 10

    def __init__(
        self,
        base_dir: str = "./simulate_session",
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        restore_checkpoint: bool = False,
        simulation_mode: SimulationMode = SimulationMode.CO_PILOT,
        mp_api_key: Optional[str] = None,
        futurehouse_api_key: Optional[str] = None,
        hpc_connection=None,
        hpc_scheduler=None,
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
            source="SimulationOrchestratorAgent",
        )

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )

        # Store configuration
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

        # Autonomy
        self.simulation_mode = simulation_mode
        self._enable_human_feedback = self._should_enable_human_feedback()

        # Optional service keys (auto-discovered when None; the wrapped
        # agents do their own auto-discovery, so storing here is just for
        # transparency / restore_from_checkpoint).
        self.mp_api_key = mp_api_key
        self.futurehouse_api_key = futurehouse_api_key

        # HPC backend — optional; tools degrade gracefully when None
        self.hpc_connection = hpc_connection
        self.hpc_scheduler = hpc_scheduler

        # Routing decision from route_simulation tool. Populated once
        # per session by the first route_simulation call; subsequent
        # tool calls / the chat loop can consult it without re-routing.
        self.routing_decision = None

        logging.info(f"🎛️  Simulation Mode: {simulation_mode.value.upper()}")

        # Setup directories
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.base_dir / "chat_history.json"
        self.checkpoint_path = self.base_dir / "checkpoint.json"
        self.structures_dir = self.base_dir / "structures"
        self.structures_dir.mkdir(parents=True, exist_ok=True)

        # Session state — structure-centric (vs analysis-centric in analyze mode)
        # generated_structures: list of {slug, structure_path, description,
        #     created_at, vasp_inputs_path?, vasp_output_dir?}
        self.generated_structures: List[Dict[str, Any]] = []

        # Sticky calc params: defaults that carry across structures within
        # the session (e.g. ENCUT, k-mesh density, functional). Tools that
        # generate VASP inputs should consult these and apply them when set.
        self.default_calc_params: Dict[str, Any] = {}

        # Counter for unique slugs within the same second
        self._structure_counter = 0

        # External tool / skill / MCP support — mirrors analyze mode
        self._external_tools: List[Dict[str, str]] = []
        self._custom_skills: Dict[str, str] = {}
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

        # Initialize tools registry
        self.tools = SimulationOrchestratorTools(self)

        # System prompt
        system_prompt = get_system_prompt(
            self.simulation_mode,
            external_tools=self._external_tools or None,
            custom_skills=self._custom_skills or None,
        )

        # Initialize LLM
        if base_url:
            logging.info(f"🏛️ Simulation orchestrator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            )
            self.use_openai = True
            self.tools_for_model = self.tools.openai_schemas
        else:
            # Public / LiteLLM — delegate model→provider→env-var resolution
            # to LiteLLM (works for any model LiteLLM supports; raises a
            # message naming the missing vendor env var if not).
            if api_key is None:
                require_vendor_credentials(model_name)
            logging.info(f"🌐 Simulation orchestrator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key,
                system_instruction=system_prompt,
                tools=self._convert_tools_to_litellm_format(),
            )
            self.use_openai = False
            self.tools_for_model = self._convert_tools_to_litellm_format()

        self._system_prompt = system_prompt

        # Initialize message history
        history = self._load_history()
        self.messages = [{"role": "system", "content": system_prompt}]
        if history:
            recent_history = self._trim_history(history, max_messages=self.MAX_HISTORY_MESSAGES)
            self.messages.extend(recent_history)

        logging.info(f"✅ SimulationOrchestratorAgent initialized. Session: {self.base_dir}")

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def active_skill_and_domain(self) -> tuple[Optional[str], Optional[str]]:
        """Return ``(skill, domain)`` derived from the current routing decision.

        Maps the router's ``(engine, scale)`` vocabulary onto the skill
        infrastructure's ``(skill, domain)`` vocabulary so downstream tools
        — in particular the engine-neutral critic agents — can derive their
        ``skill=`` and ``domain=`` parameters from session state instead of
        requiring the LLM to pass them on every call.

        Returns:
            ``(engine, scale)`` from ``routing_decision`` when it has been
            populated by a successful ``route_simulation`` call. Returns
            ``(None, None)`` when the orchestrator has not routed yet or
            the prior routing call failed (the LLM should call
            ``route_simulation`` before invoking tools that need this).
        """
        decision = self.routing_decision or {}
        scale = decision.get("scale")
        engine = decision.get("engine")
        if not scale or not engine:
            return None, None
        return engine, scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """Interactive chat — used by the CLI and UI."""
        # ── Console display features not yet wired here (parity TODO) ──────────
        # Analysis, meta, and planning distinguish three console output classes;
        # this orchestrator currently emits only structural logs. To bring it to
        # parity, mirror AnalysisOrchestratorAgent:
        #   1. 💭 Reasoning — copy `_print_assistant_reasoning(self, content)` and
        #      call it after appending each assistant message and BEFORE the tool
        #      loop in BOTH handlers (the two `for tool_call in …` sites in
        #      `_handle_openai_chat` / `_handle_litellm_chat`). Shows the interim
        #      reasoning dim+italic so a deliberate step doesn't read as a silent
        #      jump to "🔧 Calling tool". Follow the per-mode convention: print a
        #      plain 💭 standalone, and when meta-delegated (label != "Agent") tag
        #      the reasoning as `💭 <delegation-emoji> ` so the source is visible
        #      without the bare emoji triggering thought-styling — gated on
        #      `_agent_label` exactly as analysis (💭 🧪) and planning (💭 📋) do.
        #   2. 🤖 Answer — copy `_print_agent_answer(self, text)` and call
        #      `self._print_agent_answer(response)` just before `return response`
        #      below. NB: unlike analysis/planning (which print "🤖 Agent:" in
        #      chat()), this chat() returns the answer un-printed, so this is an
        #      ADD, not a replace.
        #   3. Meta attribution — when the deferred `delegate_to_simulation` seam
        #      creates the sim child, set
        #      `child._agent_label = "Simulation specialist"` (mirrors the
        #      analysis/planning children in MetaOrchestratorAgent).
        #   4. UI — no change needed; `ui/app.py::_log_to_html` keys thought
        #      blocks off the leading 💭, and the `💭 <emoji>` tag rides inside
        #      that line, so it is styled dim+italic like the others for free.
        self._last_chat_hit_iter_cap = False
        if self.use_openai:
            response = self._handle_openai_chat(user_input)
        else:
            response = self._handle_litellm_chat(user_input)
        self.message_count += 1
        self._auto_checkpoint()
        self._save_history()
        return response

    def run_task(self, task: str, context: Optional[Dict[str, Any]] = None,
                 autonomy: Optional[SimulationMode] = None,
                 max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Non-interactive entry point — used by the meta agent.

        Runs the task and returns a structured summary that's easy to
        consume programmatically:

            {
                "status": "success" | "error",
                "summary": str,                    # the agent's final reply
                "files_produced": List[str],       # absolute paths
                "key_findings": List[str],         # extracted from validations
                "suggested_followups": List[str],
                "structures": List[dict],          # session record snapshot
                "warnings": List[str],
                "task": str,                       # echoed input
            }

        ``autonomy`` selects the SimulationMode for this call. Defaults to
        AUTONOMOUS — the safe choice for a headless/programmatic caller, so
        the agent never pauses for a nonexistent user. A caller attached to a
        human (the meta agent, driven via CLI/UI) passes AUTOPILOT so the
        sub-agents' human-feedback prompts reach that human.
        The structured summary is derived from the post-call session-state
        delta; the original mode is restored on exit, even if chat() raises.
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

        # Snapshot prior state so we can compute "what was produced *during*
        # this call" rather than "everything in the session."
        structures_before = list(self.generated_structures or [])
        n_before = len(structures_before)

        # Run under the requested autonomy mode — AUTONOMOUS by default (the
        # safe headless choice). The meta agent passes its own mode through,
        # so a co-pilot / autopilot delegation still raises the sub-agents'
        # human-feedback prompts to the user driving the session.
        run_mode = autonomy if autonomy is not None else SimulationMode.AUTONOMOUS
        original_mode = self.simulation_mode
        original_max_iter = self.max_iterations
        try:
            self.set_simulation_mode(run_mode)
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
            self.set_simulation_mode(original_mode)
            self.max_iterations = original_max_iter

        # Derive the structured summary from session state.
        new_structures = (self.generated_structures or [])[n_before:]

        files_produced: List[str] = []
        key_findings: List[str] = []
        for s in new_structures:
            for key in ("structure_path", "script_path"):
                p = s.get(key)
                if p:
                    files_produced.append(p)
            for p in (s.get("input_files") or {}).values():
                if p:
                    files_produced.append(p)
            val = s.get("validation") or {}
            for issue in val.get("all_identified_issues", []) or []:
                key_findings.append(f"[{s.get('slug')}] {issue}")
            assessment = val.get("overall_assessment")
            if assessment:
                key_findings.append(f"[{s.get('slug')}] {assessment}")

        warnings: List[str] = []
        for s in new_structures:
            val = s.get("validation") or {}
            if val.get("status") == "needs_correction":
                warnings.append(
                    f"Structure {s.get('slug')} has unresolved validation issues."
                )

        # Heuristic: a "follow-up" is a structure that has been built but has
        # no generated inputs yet — natural next step is to generate inputs.
        suggested_followups: List[str] = []
        for s in new_structures:
            if s.get("structure_path") and not s.get("input_files"):
                suggested_followups.append(
                    f"Generate inputs for {s.get('slug')} "
                    f"(structure at {s.get('structure_path')})."
                )

        result = {
            "status": status,
            "task": task,
            "summary": summary_text,
            "files_produced": files_produced,
            "key_findings": key_findings,
            "suggested_followups": suggested_followups,
            "structures": [
                {
                    "slug": s.get("slug"),
                    "description": s.get("description"),
                    "structure_path": s.get("structure_path"),
                    "input_files": s.get("input_files") or {},
                } for s in new_structures
            ],
            "warnings": warnings,
        }
        if error_msg:
            result["error"] = error_msg
        return result

    def set_simulation_mode(self, mode: SimulationMode) -> None:
        """Change autonomy at runtime."""
        old_mode = self.simulation_mode
        self.simulation_mode = mode
        self._enable_human_feedback = self._should_enable_human_feedback()
        new_system_prompt = get_system_prompt(
            mode,
            external_tools=self._external_tools or None,
            custom_skills=self._custom_skills or None,
        )
        self._system_prompt = new_system_prompt
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = new_system_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": new_system_prompt})
        logging.info(f"🎛️  Simulation mode: {old_mode.value.upper()} → {mode.value.upper()}")

    def get_human_feedback_setting(self) -> bool:
        return self._enable_human_feedback

    # ------------------------------------------------------------------
    # MCP / external tools / custom skills (mirrors analyze mode)
    # ------------------------------------------------------------------

    def register_tools(self, schemas: list, factory: callable) -> None:
        """Register external tools that the LLM can call directly.

        Args:
            schemas: List of OpenAI-format tool schemas.
            factory: Callable(orch) → {tool_name: callable} mapping.
        """
        functions = factory(self)
        for schema in schemas:
            name = schema.get("function", {}).get("name") or schema.get("name")
            if not name:
                self.logger.warning("Skipping tool schema without name")
                continue
            description = schema.get("function", {}).get("description", "")
            self._external_tools.append({"name": name, "description": description})
            if name in functions:
                self.tools.functions_map[name] = functions[name]
                self.tools.openai_schemas.append(schema)
                self.logger.info(f"Registered external tool: {name}")
            else:
                self.logger.warning(f"Schema for '{name}' has no matching factory function")
        # Refresh system prompt to surface the new tools
        self._refresh_system_prompt()

    def register_skill(self, skill_path: str) -> str:
        """Register a custom skill (.md file) for this session."""
        path = Path(skill_path)
        if not path.exists() or not path.suffix == ".md":
            raise ValueError(f"Skill must be a .md file that exists: {skill_path}")
        name = path.stem
        self._custom_skills[name] = str(path.resolve())
        self._refresh_system_prompt()
        return name

    def connect_mcp_server(self, server_config: Dict[str, Any]) -> str:
        """Stub for MCP server connection. Mirrors AnalysisOrchestratorAgent's
        signature; concrete implementation is a follow-up step (mirror the
        analyze-mode code when needed)."""
        raise NotImplementedError(
            "MCP server support is wired through but not yet activated for "
            "the simulation orchestrator. Mirror analyze mode's pattern when "
            "implementing."
        )

    def disconnect_mcp_server(self, server_name: str) -> None:
        if server_name in self._mcp_connections:
            del self._mcp_connections[server_name]

    def disconnect_all_mcp_servers(self) -> None:
        for name in list(self._mcp_connections.keys()):
            self.disconnect_mcp_server(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_tools_to_litellm_format(self) -> List[Dict]:
        """Convert OpenAI tool schemas to LiteLLM format (passthrough)."""
        return self.tools.openai_schemas

    def _should_enable_human_feedback(self) -> bool:
        return self.simulation_mode != SimulationMode.AUTONOMOUS

    def _refresh_system_prompt(self) -> None:
        """Rebuild the system prompt after registering tools/skills so the
        model sees the updated list."""
        new_prompt = get_system_prompt(
            self.simulation_mode,
            external_tools=self._external_tools or None,
            custom_skills=self._custom_skills or None,
        )
        self._system_prompt = new_prompt
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = new_prompt

    def _trim_history(self, history: List[Dict], max_messages: int = None) -> List[Dict]:
        """Keep only the most recent N messages from history."""
        max_messages = max_messages or self.MAX_HISTORY_MESSAGES
        if len(history) <= max_messages:
            return history
        return history[-max_messages:]

    def _load_history(self) -> List[Dict]:
        """Load conversation history from disk."""
        if not self.history_path.exists():
            return []
        print("  🧠 Memory: Loading previous conversation...")
        try:
            with open(self.history_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load history: {e}")
            return []

    def _save_history(self) -> None:
        """Persist conversation history to disk."""
        try:
            history_data = [m for m in self.messages if m["role"] != "system"]
            with open(self.history_path, "w") as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save history: {e}")

    def _restore_checkpoint(self) -> None:
        """Restore session state from checkpoint (structures, sticky params)."""
        try:
            with open(self.checkpoint_path, "r") as f:
                ck = json.load(f)
            self.generated_structures = ck.get("generated_structures", []) or []
            self.default_calc_params = ck.get("default_calc_params", {}) or {}
            self.message_count = ck.get("message_count", 0)
            self.last_checkpoint_message_count = self.message_count
            self.logger.info(
                f"Restored checkpoint: {len(self.generated_structures)} "
                f"structures, message_count={self.message_count}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to restore checkpoint: {e}")

    def _auto_checkpoint(self) -> None:
        """Save checkpoint periodically (every CHECKPOINT_INTERVAL messages)."""
        if self.message_count - self.last_checkpoint_message_count < self.CHECKPOINT_INTERVAL:
            return
        try:
            ck = {
                "generated_structures": self.generated_structures,
                "default_calc_params": self.default_calc_params,
                "message_count": self.message_count,
                "saved_at": datetime.now().isoformat(),
            }
            with open(self.checkpoint_path, "w") as f:
                json.dump(ck, f, indent=2, default=str)
            self.last_checkpoint_message_count = self.message_count
            print(f"    ✅ Auto-checkpoint saved")
        except Exception as e:
            self.logger.warning(f"Auto-checkpoint failed: {e}")

    # ------------------------------------------------------------------
    # Chat handlers — manual tool-call loops, mirrors analyze mode
    # ------------------------------------------------------------------

    def _handle_openai_chat(self, user_input: str) -> str:
        """Chat with OpenAI-compatible models with manual function calling loop."""
        from openai import OpenAI

        client = OpenAI(
            api_key=self.model.api_key,
            base_url=self.model.base_url,
            timeout=120.0,
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
                    tool_choice="auto",
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"  ⚠️ API timeout on iteration {iteration}. Retrying...")
                    if iteration < 3:
                        continue
                raise

            message = response.choices[0].message

            if not message.tool_calls:
                text = message.content
                if not text and iteration > 0:
                    self.messages.append({
                        "role": "user",
                        "content": "Please briefly summarize what you just did and suggest next steps.",
                    })
                    followup = client.chat.completions.create(
                        model=self.model.model,
                        messages=self.messages,
                        tools=self.tools_for_model,
                        tool_choice="none",
                    )
                    text = followup.choices[0].message.content or ""
                self.messages.append({"role": "assistant", "content": text})
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
                            "arguments": tc.function.arguments,
                        },
                    } for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"  🔧 Calling tool: {func_name}")
                result = self.tools.execute_tool(func_name, **args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        self._last_chat_hit_iter_cap = True
        return "⚠️ Maximum tool iterations reached. Please simplify your request."

    def _handle_litellm_chat(self, user_input: str) -> str:
        """Chat with LiteLLM models with manual function calling loop."""
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
                    timeout=120,
                    request_timeout=120,
                )
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"  ⚠️ API timeout on iteration {iteration}. Retrying...")
                    if iteration < 3:
                        continue
                raise

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            content = getattr(message, "content", None)

            if not tool_calls:
                if not content and iteration > 0:
                    self.messages.append({
                        "role": "user",
                        "content": "Please briefly summarize what you just did and suggest next steps.",
                    })
                    followup = litellm_completion(
                        model=self.model.model,
                        messages=self.messages,
                        tools=self.tools_for_model,
                        tool_choice="none",
                    )
                    content = getattr(followup.choices[0].message, "content", None) or ""
                self.messages.append({"role": "assistant", "content": content or ""})
                return content or ""

            self.messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in tool_calls
                ],
            })

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
                    "content": result,
                })

        self._last_chat_hit_iter_cap = True
        return "⚠️ Maximum tool iterations reached. Please simplify your request."

    # ------------------------------------------------------------------
    # Restoration
    # ------------------------------------------------------------------

    @classmethod
    def restore_from_checkpoint(cls, base_dir: str, **kwargs):
        """Factory method: restore a SimulationOrchestratorAgent from a
        checkpoint directory."""
        return cls(base_dir=base_dir, restore_checkpoint=True, **kwargs)
