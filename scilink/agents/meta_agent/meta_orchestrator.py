"""Meta-Agent Orchestrator — the orchestrator-of-orchestrators.

Presents a single conversational surface and routes each piece of work to the
right specialist mode orchestrator (analysis or planning), consuming them
through their non-interactive ``run_task`` contract. See CLAUDE.md
"The meta agent".

It is **not a fourth mode** — it sits on top of the three modes with a
different role (router + context bridge). v1 covers analysis + planning;
simulation delegation is a deferred lazy seam (see meta_orchestrator_tools.py).

The chat-loop / checkpoint / history shape is copied from
AnalysisOrchestratorAgent. The duplication is intentional and acceptable at
this development stage — see CLAUDE.md "Why no BaseChatOrchestrator refactor".
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from .meta_orchestrator_tools import MetaOrchestratorTools


class MetaMode(Enum):
    """Autonomy level for the meta orchestrator — **two levels, not three**.

    The individual modes use a three-level paradigm (co-pilot / autopilot /
    autonomous). The meta agent cannot: a delegation runs the child through
    its one-shot ``run_task``, which gives the child a single turn. Co-pilot's
    model — pause after *every* step and wait for the user's next message —
    needs many turns, so it cannot complete a delegated task. AUTOPILOT and
    AUTONOMOUS each finish a task within one turn.

    Each delegated child runs under this same mode (mapped by enum name):
    AUTOPILOT keeps the child's human-feedback prompts — it pauses at
    decision points for the user to approve / edit plans and outputs;
    AUTONOMOUS runs end to end without pausing.
    """
    AUTOPILOT = "autopilot"      # Children pause for feedback at decision points
    AUTONOMOUS = "autonomous"    # Children run end to end without pausing

    @classmethod
    def _missing_(cls, value):
        # Back-compat: the AUTOPILOT level was named "supervised" before.
        if isinstance(value, str) and value.strip().lower() == "supervised":
            return cls.AUTOPILOT
        return None


_AUTOPILOT_DIRECTIVE = """**OPERATING MODE: AUTOPILOT (default)**
- Delegate on your own judgement; briefly announce each delegation before
  making it, and ask a clarifying question when the goal or the right
  specialist is genuinely ambiguous.
- Each delegated specialist runs in autopilot mode — it pauses at its own
  decision points for the user to approve or edit plans and outputs. Let
  that happen.
- Chain delegations when the path is clear; pause to report if a child
  returns an error or an ambiguous result.

"""

_AUTONOMOUS_DIRECTIVE = """**OPERATING MODE: AUTONOMOUS**
- Fulfil the user's goal end to end. Chain `delegate_to_*` calls as needed.
- Thread results forward: pass a child's relevant `key_findings` /
  `files_produced` into the next delegation's `context`.
- Only stop for an unrecoverable error. Summarize the whole effort at the end.

"""

_SYSTEM_PROMPT_BODY = """You are the **Meta-Agent** — a research orchestrator that coordinates
SciLink's specialist mode orchestrators. You do not analyze data or design
experiments yourself; you route each piece of work to the right specialist
and weave their results into one coherent response for the user.

__SPECIALIST_CAPABILITIES__

Computational simulation (DFT / MD) is NOT available in this build — do not
attempt to delegate simulation work.

**INSPECTING UPLOADED FILES:**
- When the user refers to uploaded files — or points you at a folder — call
  `inspect_uploads` FIRST, before delegating. It returns a content probe of
  each file (array shape/dtype, table columns, document text, JSON keys) so
  you route from evidence rather than guessing from filenames.
- Match each piece of work to the specialist whose capabilities — listed
  under SPECIALIST CAPABILITIES above — cover it. That inventory, generated
  from each mode's live tool registry, is the source of truth: reason from
  the tool and sub-agent descriptions, not from fixed file-type rules.
- Route from the probe (data type, shape, columns, document text), not the
  filename. When a file does not clearly match one mode, weigh it against
  the capability descriptions; if still unclear, ask the user.
- Two distinctions the descriptions assume you already make: (1) a 1-D
  measurement *curve* (a spectrum / scan — x vs y) is analysis data, but a
  *results table* (rows = samples or runs, columns = properties) is planning
  data, though both look "tabular"; (2) a chart, plot, or figure lifted from
  a document is reference context to read — not scientific image data.
- Papers / reports / notes route by INTENT, not file type. A few documents
  that are reference context for interpreting the data — a methods paper, a
  prior analysis report — go WITH the data to `delegate_to_analysis`, which
  can read documents directly. Literature for experiment design, hypothesis
  generation, or building a knowledge base → `delegate_to_planning`; a large
  corpus of papers always goes to planning (it builds a searchable index),
  while analysis reads only a handful straight into context. Some documents
  fit either side — a protocol is analysis reference context or planning
  experimental-design input depending on the user's goal.
- Several probed files may form a single experimental series or dataset —
  matching column schemas, sequential / parametric filenames (e.g.
  `spec_5K`, `spec_10K`, ...), or a shared sidecar-JSON pattern. Recognize
  this from the probes and the user's goal, and delegate the whole set as
  ONE batched task (pass the file list or their shared directory) so the
  specialist's batch tools engage — never one delegation per file.
- `inspect_uploads` is for routing only — do not use its output to interpret
  or analyze the data yourself; hand that to the specialist.
- When an uploaded file holds BOTH the data and its metadata mixed together
  (an HDF5/NeXus with attributes, a `.npz`/`.mat` with data+meta keys, a
  CSV/text with a header/comment metadata block, a TIFF with tags/
  ImageDescription), call `prepare_inputs` on it to split it losslessly into a
  data file + a metadata JSON, then delegate with those two paths. When SEVERAL
  such combined files are the same kind (or the user says they're a batch/
  series), pass them all as a LIST to `prepare_inputs` in ONE call — it splits
  them with a shared, reused recipe and returns one (data, metadata) pair per
  file — then make ONE batched delegation (never one prep or delegation per
  file). `prepare_inputs` is your ONLY code-generation tool and it is STRICTLY
  limited to lossless file repackaging (round-trip verified) — NEVER analysis,
  transformation, or computation. You do not write or run any other code; all
  data processing is delegated. If it errors, do NOT silently delegate — tell
  the user you could not safely split the file(s) and ask how to proceed (e.g.
  analyze as-is, or supply the data and metadata separately); for a `partial`
  batch result, surface which files failed and ask. Only fall back to
  delegating as-is if no user is available.

**EXPERIMENTAL METADATA:**
- Experimental data needs metadata — measurement technique, instrument,
  sample, and conditions — for a meaningful analysis, and planning data
  needs the experimental conditions behind each measurement.
- Before delegating an analysis or planning task, check whether the user
  supplied it: in their message, an uploaded metadata JSON, or per-file
  sidecar files. `inspect_uploads` shows each JSON's keys and filename — a
  JSON whose name matches a data file's stem (`spec_5K.json` ↔
  `spec_5K.csv`) is that file's sidecar metadata.
- If it is missing, ask the user for it conversationally FIRST, then put it
  into the delegation's `task` / `context`. Do not delegate a data task with
  no metadata and let the specialist stop midway to ask for it.
- Per-file conditions given as a MANIFEST (a .txt/.csv/README mapping each data
  file to its conditions, e.g. "filename, temperature, pH"), when the data files
  have NO matching sidecar JSONs, do NOT become feature-table columns if you only
  quote them in the task text — and the specialist will fall back to a positional
  index. Read the mapping and call `materialize_sidecars({filename: {conditions}},
  data_dir)` to write per-file sidecars FIRST, then delegate the folder; the
  conditions then flow into the feature table as columns (which downstream
  optimization needs as inputs).

**EQUIPMENT & SETUP — A HARD PRE-DELEGATION GATE:**
- This gate applies to any task whose output guides what the user does next
  in the lab: an experimental plan, a protocol, an optimization campaign, or
  a recommendation / assessment of what to measure or which steps to take
  next. All of these are only actionable when grounded in what the user can
  actually run — their measurement instruments (and what those instruments
  can and cannot detect), their processing / fabrication equipment, any
  automation, and any setup constraints. A plan or recommendation to measure
  or do something the user has no instrument or equipment for is not
  actionable.
- The gate fires BEFORE the FIRST such delegation of the session. If the
  user's goal will involve designing experiments at all, gather the
  equipment up front — before any plan-generating delegation. Do NOT
  delegate an initial / phase-1 plan and then ask for equipment when
  designing a later phase: the first plan is itself an experiment-design
  artifact and depends on the user's instruments just as much. A plan that
  recommends additional or follow-up measurements needs to know what the
  user's instruments can detect.
- So, before that first delegation, check whether the user has already
  stated their available instruments, equipment, and setup in their
  messages so far. If they have NOT, STOP — do not delegate — and ask for it
  conversationally first. Once the user answers, put their equipment and
  constraints into the delegation's `task` / `context`, then delegate.
- This gate is not optional and is not waived in autonomous mode — equipment
  is input you cannot infer or invent.

**COMPLEMENTARY MEASUREMENTS OF ONE SYSTEM:**
- Uploads may be different modalities of the SAME physical system (e.g. a
  STEM image, an XPS survey, and a Raman spectrum of one sample) — not a
  series. Clues: a shared filename stem / prefix, a shared sample name in
  metadata, or the user stating it.
- You cannot be sure they share a system. In AUTOPILOT mode, when it is not
  clear, ask the user before treating them as one. In AUTONOMOUS mode, make
  your best inference and state that assumption in your synthesis so the
  user can catch a wrong call.
- For a confirmed shared system with TWO OR MORE datasets that each deserve
  their own analysis, PREFER the parallel fan-out path over sequential
  delegations: call `assess_complementarity` to confirm they are genuinely
  complementary (same system, different information, a shared join axis), then
  `delegate_to_analyses` to analyze them CONCURRENTLY — each branch routed to
  its proper specialist and seeing the others as auxiliary — and finally
  `fuse_delegations` to reconcile their findings into ONE correlated
  interpretation. This both parallelizes the work and lets each analysis be
  informed by the others.
- Two cases do NOT take the fan-out path: (a) if `assess_complementarity` /
  `delegate_to_analyses` DECLINES (the datasets are redundant or unrelated),
  analyze them as independent `delegate_to_analysis` calls — do not force a
  fused interpretation; (b) a pure REFERENCE / baseline / calibration dataset
  (an I0, an empty-cell background, a dark frame) is an auxiliary, NOT its own
  branch — pass it as a companion to a single `delegate_to_analysis` of the
  measurement it calibrates, not as a fan-out branch.
- Sequential `delegate_to_analysis` calls threaded through `context` remain the
  fallback when fan-out is not appropriate; still end with ONE correlated
  interpretation across modalities — not N separate reports.

**THE DELEGATION CONTRACT:**
- `delegate_to_analysis(task, context)` and `delegate_to_planning(task,
  context)` run the specialist and return a structured JSON result: status,
  summary, key_findings, files_produced, suggested_followups, warnings,
  delegation_index.
- The specialist runs in the SAME autonomy mode as you. In autopilot mode it
  pauses at its decision points for the user to approve or edit plans and
  outputs (via its own human-feedback prompts) — that is expected and good;
  let it happen. In autonomous mode it runs the task end to end.
- Each specialist is ONE persistent agent for the whole session: a second
  `delegate_to_planning` reaches the SAME planning agent that handled the
  first, and likewise for analysis. It remembers its own prior delegations —
  the analyses it ran, the plan it produced, its campaign state. So a
  follow-up or refinement delegation can simply reference that prior work
  ("refine the plan you produced last step, but ...", "extend your previous
  analysis to ...") instead of re-deriving it from scratch.
- `task` is still a complete, self-contained instruction: the specialist
  remembers its OWN past delegations, but it cannot see THIS — the meta's —
  conversation. So anything that lives only here must go into `task` /
  `context`: a new data file's absolute path, a constraint the user just
  gave you, an upstream specialist's finding.
- Carry the user's objective into `task` in their OWN words — quote the
  scientific ask verbatim rather than paraphrasing it. You may add the data
  path, upstream findings, and framing around that quote, but do not reword,
  normalize, or reinterpret the quantity the user asked for: a paraphrase can
  silently change the meaning (e.g. turning "X or Y" into "X, i.e. Y"), and
  the specialist cannot see the original message to catch the drift.
- Pass upstream findings via the `context` dict, not by re-typing them into
  `task`.
- Give each call a short `label` (required) — a 2-5 word noun phrase, NOT a
  sentence: the data type for an analysis ("1-D Raman spectra", "STEM
  image"), or the focus for a planning task ("follow-up BO campaign"). It is
  how the delegation is shown in the UI.

**BRIDGING CONTEXT BETWEEN SPECIALISTS:**
- Every delegation's result is kept in a delegation ledger. Call
  `get_delegation_history` to retrieve an earlier result, then pass the
  relevant `key_findings` / `files_produced` as the `context` argument of the
  next `delegate_to_*` call. That is how analysis findings inform planning.
- When the `context` you pass draws on earlier delegations, also list those
  delegations' `delegation_index` numbers in the `context_from` argument —
  this records the provenance of the threaded findings.
- `summarize_session_state` reports what each specialist has done so far.

**ANALYSIS RESULTS -> PLANNING / BO:**
- An analysis delegation result may include `feature_tables` — CSV files of the
  extracted per-unit features (one row per spectrum / image, columns =
  experimental conditions + extracted scalar features). When a follow-up
  planning or Bayesian-optimization task needs those quantitative results, pass
  the feature-table file PATH to `delegate_to_planning` (in `context`, and name
  it in `task`).
- Do NOT re-summarize the numbers as prose for the planning specialist to
  retype — that loses precision and risks transcription errors. The planning
  specialist ingests the file directly with its `analyze_file` tool.
- When the user wants Bayesian optimization — "optimize X", "what should I
  measure / do next", "recommend the next experiments" over an existing
  dataset — delegate a concise run-it task: "Run Bayesian optimization on
  `<feature-table path>`, inputs=[...], targets=[...], to recommend the next
  measurement(s)." Do NOT phrase it as "design a campaign" and do NOT
  pre-author a multi-phase experimental plan in the `task`. The optimizer's
  recommended batch IS the deliverable; a hand-written plan alongside it
  would contradict the optimizer's output.

**RESPONSE STYLE:**
- Do not dump raw tool JSON back to the user — synthesize it into plain
  language.
- Make clear which specialist produced which result.
- Surface the specialists' `suggested_followups` when proposing next steps.
- Report produced files accurately: name each artifact and where it lives.
  Do NOT claim one file contains another's content unless you actually
  checked it (e.g. by reading the file) — e.g. do not state that a plan's
  HTML report "includes" the optimizer's diagnostic plots when those are
  separate image files in their own directory.
"""


def get_system_prompt(meta_mode: MetaMode) -> str:
    """Return the system prompt for the given meta autonomy mode.

    The prompt still contains the ``__SPECIALIST_CAPABILITIES__`` placeholder;
    it is filled in on the meta's first chat turn by ``_inject_capabilities``
    (the live tool inventory can only be read once the child orchestrators
    exist).
    """
    directives = {
        MetaMode.AUTOPILOT: _AUTOPILOT_DIRECTIVE,
        MetaMode.AUTONOMOUS: _AUTONOMOUS_DIRECTIVE,
    }
    return directives[meta_mode] + _SYSTEM_PROMPT_BODY


def _tool_lines(schemas) -> str:
    """Bulleted ``name``: first-line-description list from OpenAI tool
    schemas (``[{"function": {"name", "description", ...}}, ...]``)."""
    lines = []
    for s in schemas or []:
        fn = s.get("function") if isinstance(s, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        desc = " ".join(str(fn.get("description") or "").split())
        if len(desc) > 160:
            desc = desc[:159] + "…"
        lines.append(f"    - `{fn['name']}`: {desc}" if desc
                     else f"    - `{fn['name']}`")
    return "\n".join(lines)


class MetaOrchestratorAgent:
    """Orchestrator-of-orchestrators.

    Presents one chat surface; delegates analysis and planning sub-tasks to
    persistent child orchestrators (one per mode, reused across delegations)
    nested under the meta-session directory. Children are consumed only
    through their public ``run_task`` contract — duck-typed, no base class.

    Args:
        base_dir: Base directory for the meta session.
        api_key: API key for the LLM provider (forwarded to children).
        model_name: Model name (forwarded to children).
        base_url: Base URL for an internal proxy endpoint (forwarded).
        embedding_model: Embedding model name (forwarded).
        embedding_api_key: API key for the embedding provider (forwarded).
        futurehouse_api_key: Optional FutureHouse API key (forwarded).
        restore_checkpoint: Whether to restore from a previous checkpoint.
        meta_mode: Autonomy level (AUTOPILOT or AUTONOMOUS). The meta has
            only these two — see MetaMode.
    """

    # Configuration constants (match the child orchestrators).
    MAX_TOOL_ITERATIONS = 20
    MAX_HISTORY_MESSAGES = 100
    CHECKPOINT_INTERVAL = 10

    def __init__(
        self,
        base_dir: str = "./meta_session",
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        embedding_model: str = "gemini-embedding-001",
        embedding_api_key: Optional[str] = None,
        futurehouse_api_key: Optional[str] = None,
        restore_checkpoint: bool = False,
        meta_mode: MetaMode = MetaMode.AUTOPILOT,
        max_iterations: Optional[int] = None,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

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

        # Store configuration — these are forwarded verbatim to child
        # orchestrators when they are lazily created.
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.futurehouse_api_key = (
            futurehouse_api_key or os.environ.get("FUTUREHOUSE_API_KEY")
        )

        self.meta_mode = meta_mode
        self._enable_human_feedback = self._should_enable_human_feedback()
        logging.info(f"🎛️  Meta Mode: {meta_mode.value.upper()}")

        # Setup directories. Children live in fixed sub-directories so a
        # persistent child can be re-created (with restore_checkpoint=True)
        # after a meta restore simply by probing for its checkpoint.json.
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.base_dir / "chat_history.json"
        self.checkpoint_path = self.base_dir / "checkpoint.json"
        self.analysis_dir = self.base_dir / "analysis"
        self.planning_dir = self.base_dir / "planning"
        # Fan-out workers (ephemeral, one isolated analysis session per branch)
        # and fused cross-dataset reports live in their own sub-trees, distinct
        # from the persistent analysis/ and planning/ children. See fanout.py.
        self.fanout_dir = self.base_dir / "fanout"
        self.fusion_dir = self.base_dir / "fusion"

        # Session state — kept shallow; children own their deep state.
        self._children: Dict[str, Any] = {}          # "analysis"/"planning" -> agent
        self._delegation_ledger: List[Dict[str, Any]] = []
        # Complementarity verdicts cached by frozenset of dataset paths, so the
        # standalone assess_complementarity tool and the internal gate in
        # run_fanout share one LLM call. Serializes ledger preallocation and
        # fusion-entry appends across fan-out worker threads.
        self._complementarity_cache: Dict[frozenset, Dict[str, Any]] = {}
        self._fanout_lock = threading.Lock()
        # Auto-generated specialist capability inventory — built once on the
        # first chat turn (children must exist to read their tool registries).
        self._capabilities_block: Optional[str] = None

        # External tools registered on the meta itself (MCP server tools) —
        # callable alongside the delegate_to_* tools. MCP connections are
        # live subprocesses; not checkpointed, reconnect after a restore.
        self._external_tools: List[Dict[str, str]] = []
        self._mcp_connections: Dict[str, Any] = {}
        # Custom skills registered for the session (name → path) — kept for
        # display; the meta has no skill-running tools of its own.
        self._custom_skills: Dict[str, str] = {}
        # Shared extensions — skills, custom tools, MCP servers — registered
        # on the meta and propagated to every specialist child, including a
        # child created lazily after registration. One mechanism for all
        # three: see _register_shared_extension.
        self._shared_extensions: List[Dict[str, Any]] = []

        self.message_count = 0
        self.last_checkpoint_message_count = 0

        # Per-call tool-iteration cap. Mirrors the child orchestrators so a
        # meta caller can raise it for long-running multi-delegation turns.
        self.max_iterations = (
            max_iterations if max_iterations is not None
            else self.MAX_TOOL_ITERATIONS
        )
        self._last_chat_hit_iter_cap = False

        if restore_checkpoint and self.checkpoint_path.exists():
            self._restore_checkpoint()

        # Tools registry.
        self.tools = MetaOrchestratorTools(self)

        system_prompt = get_system_prompt(self.meta_mode)

        # Initialize LLM (dual path, copied from AnalysisOrchestratorAgent).
        if base_url:
            logging.info(f"🏛️ Meta orchestrator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name, api_key=api_key, base_url=base_url
            )
            self.use_openai = True
            self.tools_for_model = self.tools.openai_schemas
        else:
            logging.info(f"🌐 Meta orchestrator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key,
                system_instruction=system_prompt,
                tools=self._convert_tools_to_litellm_format(),
            )
            self.use_openai = False
            self.tools_for_model = self._convert_tools_to_litellm_format()

        self._system_prompt = system_prompt

        # Initialize message history.
        history = self._load_history()
        self.messages = [{"role": "system", "content": system_prompt}]
        if history:
            self.messages.extend(
                self._trim_history(history, max_messages=self.MAX_HISTORY_MESSAGES)
            )

        logging.info(f"✅ MetaOrchestratorAgent initialized. Session: {self.base_dir}")

    # =========================================================================
    # Mode / configuration
    # =========================================================================

    def _convert_tools_to_litellm_format(self) -> List[Dict]:
        """Convert OpenAI tool schemas to LiteLLM format (passthrough)."""
        return self.tools.openai_schemas

    def _should_enable_human_feedback(self) -> bool:
        """Human feedback is on unless the meta is fully autonomous."""
        return self.meta_mode != MetaMode.AUTONOMOUS

    def set_meta_mode(self, mode: MetaMode) -> None:
        """Change the meta autonomy mode at runtime."""
        old_mode = self.meta_mode
        self.meta_mode = mode
        self._enable_human_feedback = self._should_enable_human_feedback()

        new_system_prompt = get_system_prompt(mode)
        self._system_prompt = new_system_prompt
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = new_system_prompt

        logging.info(f"🔄 Meta mode changed: {old_mode.value} → {mode.value}")

    def get_human_feedback_setting(self) -> bool:
        """Returns the current human feedback setting."""
        return self._enable_human_feedback

    # =========================================================================
    # Child orchestrators (lazy, persistent, one per mode)
    # =========================================================================

    def _get_analysis_child(self):
        """Lazily create (or restore) the persistent analysis child.

        Imported lazily inside the method so importing the meta-agent module
        does not pull the analysis stack until a delegation actually happens.
        """
        if "analysis" not in self._children:
            from ..exp_agents.analysis_orchestrator import (
                AnalysisOrchestratorAgent, AnalysisMode,
            )
            restore = (self.analysis_dir / "checkpoint.json").exists()
            self.logger.info(
                f"🧩 {'Restoring' if restore else 'Creating'} analysis child "
                f"at {self.analysis_dir}"
            )
            # Resting mode CO_PILOT; each delegation's run_task sets the
            # autonomy mode for that call to match the meta's.
            self._children["analysis"] = AnalysisOrchestratorAgent(
                base_dir=str(self.analysis_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                embedding_model=self.embedding_model,
                embedding_api_key=self.embedding_api_key,
                futurehouse_api_key=self.futurehouse_api_key,
                restore_checkpoint=restore,
                analysis_mode=AnalysisMode.CO_PILOT,
            )
            # Label its answers as the specialist's — in the meta's verbose
            # stream a child's final answer is a delegated deliverable, not the
            # meta's own user-facing response.
            self._children["analysis"]._agent_label = "Analysis specialist"
            # Share skills / custom tools / MCP servers registered on the meta.
            self._propagate_extensions_to_child(self._children["analysis"])
        return self._children["analysis"]

    def _get_planning_child(self):
        """Lazily create (or restore) the persistent planning child.

        Built in CO_PILOT with data_dir=None: PlanningOrchestratorAgent only
        requires data_dir under AUTOPILOT/AUTONOMOUS at *construction* time.
        Each delegation's run_task sets the autonomy level via
        set_autonomy_level, which does not re-validate data_dir — so a
        autopilot/autonomous delegation is still safe. Per-task data files
        arrive as absolute paths in the delegation `task`.
        """
        if "planning" not in self._children:
            from ..planning_agents.planning_orchestrator import (
                PlanningOrchestratorAgent, AutonomyLevel,
            )
            restore = (self.planning_dir / "checkpoint.json").exists()
            self.logger.info(
                f"🧩 {'Restoring' if restore else 'Creating'} planning child "
                f"at {self.planning_dir}"
            )
            self._children["planning"] = PlanningOrchestratorAgent(
                objective="Delegated by meta-agent",
                base_dir=str(self.planning_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                embedding_model=self.embedding_model,
                embedding_api_key=self.embedding_api_key,
                futurehouse_api_key=self.futurehouse_api_key,
                restore_checkpoint=restore,
                autonomy_level=AutonomyLevel.CO_PILOT,
                data_dir=None,
            )
            # Label its answers as the specialist's — a delegated child's final
            # answer is a deliverable in the meta's verbose stream, not the
            # meta's own user-facing response.
            self._children["planning"]._agent_label = "Planning specialist"
            # Share skills / custom tools / MCP servers registered on the meta.
            self._propagate_extensions_to_child(self._children["planning"])
        return self._children["planning"]

    # =========================================================================
    # Shared extensions: skills, custom tools, MCP servers
    # =========================================================================

    def connect_mcp_server(
        self,
        server_name: str,
        *,
        command: list = None,
        url: str = None,
        env: dict = None,
    ) -> int:
        """Connect to an MCP server and register its tools on the meta and
        every specialist child.

        The tools become callable by the meta LLM (alongside the
        ``delegate_to_*`` tools) AND by each analysis / planning child during
        a delegation. The server config is recorded so a child created later
        picks it up too. Each agent holds its own connection — an MCP stdio
        server is spawned once per agent (meta + each created child).

        Args:
            server_name: Human-readable label for this server.
            command: Command + args for stdio transport,
                e.g. ``["npx", "-y", "@mcp/server-filesystem", "/tmp"]``.
            url: URL for SSE transport.
            env: Optional environment variables for the subprocess.

        Returns:
            Number of tools registered on the meta from this server.
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

            # Prefix with the server name on a collision with an existing tool.
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

        # Share with every specialist child (now and lazily-created later).
        self._register_shared_extension({
            "kind": "mcp", "server_name": server_name,
            "command": command, "url": url, "env": env,
        })
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
        # Slice-assign so the list identity is preserved: tools_for_model
        # aliases this list and is bound once at construction.
        self.tools.openai_schemas[:] = [
            s for s in self.tools.openai_schemas
            if s.get("function", {}).get("name") not in names_to_remove
        ]
        for name in names_to_remove:
            self.tools.functions_map.pop(name, None)

        # Drop the shared record and tear the server down on every child.
        self._shared_extensions = [
            e for e in self._shared_extensions
            if not (e.get("kind") == "mcp"
                    and e.get("server_name") == server_name)
        ]
        for child in self._children.values():
            try:
                child.disconnect_mcp_server(server_name)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"Could not disconnect MCP '{server_name}' on "
                    f"{type(child).__name__}: {e}"
                )

        logging.info(f"🔌 MCP '{server_name}' disconnected.")

    def disconnect_all_mcp_servers(self) -> None:
        """Disconnect from all connected MCP servers."""
        for name in list(self._mcp_connections):
            self.disconnect_mcp_server(name)

    def register_skill(self, skill_path: str) -> str:
        """Register a custom skill (.md) and share it with every specialist.

        The meta runs no analyses itself, so the skill is not registered on
        the meta — it is propagated to the analysis and planning children,
        where the skill-aware tools (run_analysis, planning) can select it.

        Returns:
            The skill name (the file stem) used to reference it.
        """
        path = Path(skill_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")
        if path.suffix.lower() != ".md":
            raise ValueError(f"Skill file must be a .md file: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Skill file is empty: {path}")

        self._custom_skills[path.stem] = str(path)
        self._register_shared_extension(
            {"kind": "skill", "skill_path": str(path)}
        )
        logging.info(f"✅ Skill '{path.stem}' shared with specialists")
        return path.stem

    def register_tools(self, schemas: list, factory: callable) -> None:
        """Register custom tool functions and share them with every specialist.

        Custom tools bind to a loaded data file (the factory receives the
        child's current data), which the meta — a router — has no concept
        of, so they are registered on the children, not the meta itself.
        """
        self._register_shared_extension(
            {"kind": "tools", "schemas": schemas, "factory": factory}
        )
        for schema in schemas:
            if schema.get("type") != "function":
                continue
            fn = schema.get("function", {})
            if fn.get("name"):
                self._external_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                })
        n = sum(1 for s in schemas if s.get("type") == "function")
        logging.info(f"✅ {n} custom tool(s) shared with specialists")

    # ── Shared-extension propagation ───────────────────────────────────

    def _register_shared_extension(self, ext: Dict[str, Any]) -> None:
        """Record a shared extension and apply it to every already-created
        child. A child created later replays it in _get_*_child. This is the
        one mechanism behind register_skill / register_tools / MCP sharing."""
        self._shared_extensions.append(ext)
        for child in self._children.values():
            self._apply_extension_to_child(child, ext)

    def _apply_extension_to_child(self, child, ext: Dict[str, Any]) -> None:
        """Apply one shared extension (skill / tools / MCP) to a child.

        Failures are logged, not raised — a child stays usable, and one bad
        child must not break the meta-side registration.
        """
        kind = ext.get("kind")
        try:
            if kind == "skill":
                child.register_skill(ext["skill_path"])
            elif kind == "tools":
                child.register_tools(ext["schemas"], ext["factory"])
            elif kind == "mcp":
                child.connect_mcp_server(
                    ext["server_name"],
                    command=ext.get("command"),
                    url=ext.get("url"),
                    env=ext.get("env"),
                )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(
                f"Could not apply {kind} extension to "
                f"{type(child).__name__}: {e}"
            )

    def _propagate_extensions_to_child(self, child) -> None:
        """Replay every shared extension onto a freshly created child, so a
        child instantiated after registration still shares skills/tools."""
        for ext in self._shared_extensions:
            self._apply_extension_to_child(child, ext)

    # =========================================================================
    # Specialist capability inventory
    # =========================================================================

    def _build_capabilities_block(self) -> str:
        """Build the routing inventory by reading each specialist's LIVE tool
        registry — so a new tool / sub-agent in any mode appears here with no
        meta-agent edit. Constructs the child orchestrators to read them."""
        sections: List[str] = []

        try:
            analysis = self._get_analysis_child()
            agents = getattr(analysis, "_agent_registry", {}) or {}
            agent_lines = "\n".join(
                f"    - {s.get('name')}: {s.get('description')}"
                for s in agents.values() if isinstance(s, dict)
            )
            a_tools = _tool_lines(
                getattr(getattr(analysis, "tools", None), "openai_schemas", []))
            sections.append(
                "`delegate_to_analysis` — interpret EXISTING experimental "
                "measurements. Each `run_analysis` picks ONE of these "
                "sub-agents by data type:\n"
                f"{agent_lines}\n  Orchestrator tools:\n{a_tools}"
            )
        except Exception as e:  # noqa: BLE001 - probe must not break chat()
            self.logger.warning(f"analysis capability probe failed: {e}")

        try:
            planning = self._get_planning_child()
            p_tools = _tool_lines(
                getattr(getattr(planning, "tools", None), "openai_schemas", []))
            sections.append(
                "`delegate_to_planning` — decide what to do next AND handle "
                "all tabular / knowledge data. Tools:\n"
                f"{p_tools}"
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"planning capability probe failed: {e}")

        if not sections:
            raise RuntimeError("no specialist capabilities could be read")
        return (
            "**SPECIALIST CAPABILITIES** — auto-generated from each "
            "specialist's live tool registry; route against these:\n\n"
            + "\n\n".join(sections)
        )

    def _inject_capabilities(self) -> None:
        """Splice the capability inventory into the system prompt. Built once
        and cached; re-applied if the prompt is later rebuilt. Never raises."""
        if not self.messages or self.messages[0].get("role") != "system":
            return
        if "__SPECIALIST_CAPABILITIES__" not in self.messages[0]["content"]:
            return  # placeholder already consumed
        if self._capabilities_block is None:
            try:
                self._capabilities_block = self._build_capabilities_block()
            except Exception as e:  # noqa: BLE001 - must never break chat()
                self.logger.warning(f"capability inventory unavailable: {e}")
                self._capabilities_block = (
                    "**SPECIALIST CAPABILITIES**: (inventory unavailable — "
                    "route by the principles below.)"
                )
        self.messages[0]["content"] = self.messages[0]["content"].replace(
            "__SPECIALIST_CAPABILITIES__", self._capabilities_block
        )

    # =========================================================================
    # Delegation + ledger (used by MetaOrchestratorTools)
    # =========================================================================

    def _delegate(self, mode: str, task: str, context: Optional[dict] = None,
                  context_from: Optional[list] = None,
                  label: Optional[str] = None) -> str:
        """Run a task on a child orchestrator, record it, return a JSON summary.

        The child runs under the meta's own autonomy mode (mapped by enum
        name), so an autopilot delegation keeps the specialist's
        human-feedback prompts — they reach the user driving the meta exactly
        as in a direct single-mode session. Called by the delegate_to_* tools.
        Never raises — child/setup failures are captured into an error result.

        A provisional 'running' ledger entry is opened before the child runs,
        so the UI delegation tree shows the delegation live; it is finalized
        with the result on completion.
        """
        if mode == "analysis":
            from ..exp_agents.analysis_orchestrator import AnalysisMode
            get_child, autonomy_enum = self._get_analysis_child, AnalysisMode
        elif mode == "planning":
            from ..planning_agents.planning_orchestrator import AutonomyLevel
            get_child, autonomy_enum = self._get_planning_child, AutonomyLevel
        else:
            return json.dumps({
                "status": "error",
                "message": f"Unknown delegation target: {mode}",
            })

        entry = self._open_delegation(mode, task, context, context_from, label)
        try:
            child = get_child()
            result = child.run_task(
                task, context=context,
                autonomy=autonomy_enum[self.meta_mode.name],
            )
        except Exception as e:
            self.logger.exception(f"Delegation to {mode} failed: {e}")
            result = {
                "status": "error", "error": str(e), "summary": "",
                "key_findings": [], "files_produced": [],
                "suggested_followups": [], "warnings": [],
            }
        self._close_delegation(entry, result)
        return self._summarize_delegation_result(mode, result, entry["index"])

    def _open_delegation(self, mode, task, context, context_from,
                         label=None) -> Dict[str, Any]:
        """Append a provisional 'running' ledger entry before the child runs.

        Finalized later by ``_close_delegation``. Having the entry present up
        front lets the UI delegation tree show the delegation while it runs.
        """
        index = len(self._delegation_ledger) + 1
        # Normalize context_from to valid prior delegation indices (the LLM may
        # pass ints, strings, or "#n" forms; drop anything out of range).
        raw = context_from
        if isinstance(raw, (int, str)):
            raw = [raw]
        elif not isinstance(raw, (list, tuple)):
            raw = []
        sources = sorted({
            int(s) for s in (str(v).strip().lstrip("#") for v in raw)
            if s.isdigit() and 0 < int(s) < index
        })
        entry = {
            "index": index,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "task": task,
            "label": (label or "").strip(),
            "context_keys": sorted(context.keys()) if isinstance(context, dict) else [],
            "context_from": sources,
            "status": "running",
            "summary": "",
            "key_findings": [],
            "files_produced": [],
            "feature_tables": [],
            "suggested_followups": [],
            "warnings": [],
            "error": None,
        }
        self._delegation_ledger.append(entry)
        return entry

    def _close_delegation(self, entry: Dict[str, Any], result: dict) -> None:
        """Finalize a provisional ledger entry with the child's result."""
        entry.update({
            "status": result.get("status"),
            "summary": result.get("summary", ""),
            "key_findings": result.get("key_findings", []),
            "files_produced": result.get("files_produced", []),
            "feature_tables": result.get("feature_tables", []),
            "staged_solutions": result.get("staged_solutions", []),
            "suggested_followups": result.get("suggested_followups", []),
            "warnings": result.get("warnings", []),
            "error": result.get("error"),
            "completed_at": datetime.now().isoformat(),
        })

    def _summarize_delegation_result(self, mode, result, index) -> str:
        """Build the JSON string a delegate_to_* tool returns to the meta LLM."""
        summary = {
            "delegation_index": index,
            "mode": mode,
            "status": result.get("status"),
            "summary": result.get("summary", ""),
            "key_findings": result.get("key_findings", []),
            "files_produced": result.get("files_produced", []),
            "feature_tables": result.get("feature_tables", []),
            "suggested_followups": result.get("suggested_followups", []),
            "warnings": result.get("warnings", []),
        }
        if result.get("error"):
            summary["error"] = result["error"]
        # Guardrail against the "empty-but-successful" state: a child can report
        # status="success" yet return no extractable findings (e.g. a synthesis
        # parse failure left detailed_analysis/scientific_claims blank). Left
        # unflagged, the meta LLM reads the empty result and re-delegates the
        # identical task — re-running the same data the same way reproduces the
        # same empty result, i.e. a loop. Surface it explicitly so the LLM
        # changes course instead of retrying blindly. Findings presence is keyed
        # on key_findings/summary, NOT files_produced — the run can emit files
        # (fit scripts, plots) while the interpretation came back empty.
        has_findings = bool(summary["key_findings"]) or bool((summary["summary"] or "").strip())
        if summary["status"] == "success" and not has_findings:
            summary["findings_status"] = "empty_but_successful"
            note = (
                "Delegation reported success but returned no extractable findings "
                "(empty key_findings and summary). Do NOT re-delegate the identical "
                "task expecting a different result — re-running the same data the "
                "same way reproduces the same empty result. Inspect files_produced, "
                "report the gap to the user, or change the approach."
            )
            warnings = list(summary.get("warnings") or [])
            if note not in warnings:
                warnings.append(note)
            summary["warnings"] = warnings
        # Raw T=2 solutions staged during this delegation. Surface them with a
        # mode-appropriate nudge so the meta LLM offers the user a review
        # (AUTOPILOT) or simply notes them (AUTONOMOUS) — acting via the
        # review_distilled_skills tool. Staged solutions don't change any skill
        # until explicitly upgraded/consolidated, so doing nothing is safe.
        staged = result.get("staged_solutions") or []
        if staged:
            summary["staged_solutions"] = staged
            if self.meta_mode == MetaMode.AUTOPILOT:
                summary["staged_solutions_action"] = (
                    "This run solved a hard problem from scratch (T=2) and STAGED the "
                    "solution for learning. Tell the user briefly, then ask whether to "
                    "upgrade an existing skill from it, consolidate it (with any earlier "
                    "staged solutions of the same technique) into a new skill, or leave "
                    "it staged — and call review_distilled_skills (action=list_staged / "
                    "upgrade / consolidate) accordingly."
                )
            else:
                summary["staged_solutions_action"] = (
                    "A hard (T=2) solution was staged for later distillation (no skill "
                    "changed). Note it in your summary; the user can review it later via "
                    "`scilink memory staged` or review_distilled_skills."
                )
        # Domain-specific field, passed through lightly.
        if "analyses" in result:
            summary["analyses"] = result["analyses"]
        if "campaign_state" in result:
            summary["campaign_state"] = result["campaign_state"]
        return json.dumps(summary, indent=2, default=str)

    def _session_state_summary(self) -> str:
        """JSON snapshot of cross-specialist session state."""
        children = {}
        for name in ("analysis", "planning"):
            child = self._children.get(name)
            if child is None:
                children[name] = {"instantiated": False}
                continue
            info = {
                "instantiated": True,
                "base_dir": str(getattr(child, "base_dir", "")),
                "message_count": getattr(child, "message_count", 0),
            }
            if name == "analysis":
                info["analyses_run"] = len(getattr(child, "analysis_results", []) or [])
            else:
                info["optimization_targets"] = (
                    getattr(child, "expected_target_columns", []) or []
                )
                bo_path = getattr(child, "bo_data_path", None)
                try:
                    import pandas as pd
                    info["bo_data_points"] = (
                        len(pd.read_csv(bo_path))
                        if bo_path and Path(bo_path).exists() else 0
                    )
                except Exception:
                    info["bo_data_points"] = 0
            children[name] = info

        last = self._delegation_ledger[-1] if self._delegation_ledger else None
        return json.dumps({
            "meta_mode": self.meta_mode.value,
            "session_dir": str(self.base_dir),
            "delegations_total": len(self._delegation_ledger),
            "last_delegation": (
                f"{last['mode']} / {last['status']}" if last else None
            ),
            "children": children,
        }, indent=2, default=str)

    def _delegation_history(self, limit: Optional[int] = None) -> str:
        """JSON dump of the delegation ledger (optionally the most recent N)."""
        ledger = self._delegation_ledger
        if limit is not None and limit > 0:
            ledger = ledger[-limit:]
        return json.dumps(ledger, indent=2, default=str)

    # =========================================================================
    # Parallel multi-dataset analysis (fan-out) — see fanout.py
    # =========================================================================

    def _assess_complementarity(self, datasets: list) -> str:
        """Partition datasets into complementary / redundant / unrelated."""
        from .fanout import assess_complementarity
        return json.dumps(assess_complementarity(self, datasets),
                          indent=2, default=str)

    def _run_fanout(self, branches: list) -> str:
        """Gate, confirm, then run analysis branches concurrently (full mesh)."""
        from .fanout import run_fanout
        return run_fanout(self, branches)

    def _fuse_delegations(self, indices: list, focus: Optional[str] = None) -> str:
        """Reconcile finished complementary branch findings into one narrative."""
        from .fanout import fuse_delegations
        return fuse_delegations(self, indices, focus)

    # =========================================================================
    # Checkpoint / history
    # =========================================================================

    def _restore_checkpoint(self):
        """Restore shallow meta state from checkpoint. Children are re-created
        lazily on the next delegation (their own checkpoint.json survives in
        the fixed sub-directory)."""
        print(f"  📂 Restoring checkpoint from: {self.checkpoint_path}")
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)

            if "meta_mode" in state:
                try:
                    self.meta_mode = MetaMode(state["meta_mode"])
                    self._enable_human_feedback = self._should_enable_human_feedback()
                except ValueError:
                    pass

            self.message_count = state.get("message_count", 0)
            self._delegation_ledger = state.get("delegation_ledger", [])

            print(f"    ✅ Restored state:")
            print(f"       - Meta mode: {self.meta_mode.value}")
            print(f"       - Delegations: {len(self._delegation_ledger)}")

        except Exception as e:
            logging.warning(f"Failed to restore checkpoint: {e}")

    def _auto_checkpoint(self):
        """Internal auto-checkpoint without LLM interaction."""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "meta_mode": self.meta_mode.value,
                "message_count": self.message_count,
                "children_instantiated": sorted(self._children.keys()),
                "delegation_ledger": self._delegation_ledger,
            }
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            print(f"    ✅ Auto-checkpoint saved")
        except Exception as e:
            logging.warning(f"Auto-checkpoint failed: {e}")

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
            "content": f"[{len(history) - max_messages} messages omitted for context management]",
        }
        trimmed.insert(context_window, summary_marker)

        return trimmed

    def _load_history(self) -> List[Dict]:
        """Load conversation history from disk."""
        if not self.history_path.exists():
            return []
        print("  🧠 Memory: Loading previous conversation...")
        try:
            with open(self.history_path, 'r') as f:
                return json.load(f)
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
        """Factory method to create a MetaOrchestratorAgent from a checkpoint."""
        return cls(base_dir=base_dir, restore_checkpoint=True, **kwargs)

    # =========================================================================
    # Chat
    # =========================================================================

    def chat(self, user_input: str) -> str:
        """Main chat interface with robust function calling support."""
        self.message_count += 1
        self._last_chat_hit_iter_cap = False

        # First-turn: fill the system prompt's specialist-capability inventory
        # from the children's live tool registries (idempotent thereafter).
        self._inject_capabilities()

        # Auto-checkpoint every N messages.
        if self.message_count - self.last_checkpoint_message_count >= self.CHECKPOINT_INTERVAL:
            print(f"  💾 Auto-checkpoint triggered (every {self.CHECKPOINT_INTERVAL} messages)...")
            self._auto_checkpoint()
            self.last_checkpoint_message_count = self.message_count

        try:
            if self.use_openai:
                response_text = self._handle_openai_chat(user_input)
            else:
                response_text = self._handle_litellm_chat(user_input)

            # The response is returned to the caller (CLI / UI), which is
            # responsible for displaying it — chat() does not print it.
            self._save_history()

            if self.message_count > 80:
                response_text += (
                    "\n\n⚠️ Note: Conversation is getting long. "
                    "Consider starting a fresh meta session."
                )

            return response_text

        except Exception as e:
            logging.error(f"Chat Error: {e}", exc_info=True)
            print("  💾 Error detected - saving emergency checkpoint...")
            self._auto_checkpoint()
            return f"❌ Error: {e}\n\n(Emergency checkpoint saved to {self.checkpoint_path})"

    def _handle_openai_chat(self, user_input: str) -> str:
        """Handle chat with OpenAI-compatible models — manual tool-calling loop."""
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
            print(f"  ⏳ Waiting for meta-orchestrator response ...")

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

            self._print_assistant_reasoning(message.content)

            for tool_call in message.tool_calls:
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

    def _print_assistant_reasoning(self, content) -> None:
        """Surface the LLM's interim reasoning that accompanies a tool call.

        Without this the console jumps straight to "🔧 Calling tool: …", so a
        deliberate step reads like a silent loop. The text is already in
        history; this just shows it so the user can see *why* the next tool is
        being called.
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
        style, reset = (("\033[2;3;36m", "\033[0m")
                        if sys.stdout.isatty() else ("", ""))
        body = text.replace("\n", "\n     ")  # indent continuation lines
        print(f"\n  {style}💭 {body}{reset}\n")

    def _handle_litellm_chat(self, user_input: str) -> str:
        """Handle chat with LiteLLM models — manual tool-calling loop."""
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
            print(f"  ⏳ Waiting for meta-orchestrator response ...")

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
                    "content": result,
                })

        self._last_chat_hit_iter_cap = True
        return "⚠️ Maximum tool iterations reached. Please simplify your request."
