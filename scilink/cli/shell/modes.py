"""What each mode contributes to the shared shell.

A ``ModeAdapter`` owns the mode-specific parts the four old ``*Playground``
classes carried: its command-line flags (kept verbatim), how to build or
restore its orchestrator, how autonomy is read and set on it, the status
fields it reports, its extra slash commands, the turns it runs before the
first prompt (``--data`` seeding, ``--message``), and how it runs one task
headlessly. Labels, placeholders, session prefixes and autonomy options are
NOT here — they come from ``scilink.ui.vocabulary``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bootstrap import Ask, Credentials, resolve_optional_key
from .commands import Command

# vocabulary autonomy string <-> orchestrator enum member name
_ENUM_NAME = {"co-pilot": "CO_PILOT", "autopilot": "AUTOPILOT", "autonomous": "AUTONOMOUS"}
_LEVEL_NAME = {v: k for k, v in _ENUM_NAME.items()}


def _level_from_enum(member) -> str:
    return _LEVEL_NAME.get(getattr(member, "name", str(member)), str(getattr(member, "value", member)))


class ModeAdapter:
    key: str = ""
    epilog: str = ""
    description: str = ""

    # ── flags ──────────────────────────────────────────────────
    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def autonomy_from_args(self, args) -> str:
        return getattr(args, "mode", None) or getattr(args, "autonomy", None) or "co-pilot"

    def validate_args(self, parser: argparse.ArgumentParser, args) -> None:
        for attr, suffix, flag in (("skill_files", ".md", "--skills"),
                                   ("tool_files", ".py", "--tools"),
                                   ("agent_files", ".py", "--agents")):
            for f in getattr(args, attr, None) or []:
                if not Path(f).exists():
                    parser.error(f"{flag} path does not exist: {f}")
                if not f.endswith(suffix):
                    parser.error(f"{flag} file must be a {suffix} file: {f}")

    # ── lifecycle ──────────────────────────────────────────────
    def prepare(self, args, ask: Ask) -> Dict[str, Any]:
        """Mode-only questions before the agent exists (plan's objective)."""
        return {}

    def build(self, args, creds: Credentials, session_dir: Path, *, restore: bool,
              extras: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def get_autonomy(self, agent) -> str:
        raise NotImplementedError

    def set_autonomy(self, agent, level: str) -> None:
        raise NotImplementedError

    def status_fields(self, agent) -> List[Tuple[str, str]]:
        return []

    def extra_commands(self) -> List[Command]:
        return []

    def initial_turns(self, args, agent) -> List[str]:
        return []

    def headless_run(self, agent, task: str) -> Dict[str, Any]:
        result = agent.run_task(task)
        return result if isinstance(result, dict) else {"status": "success", "summary": str(result)}


# ── shared flag groups ───────────────────────────────────────────

def _model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", type=str, default="claude-opus-4-6",
                   help="Model name (default: claude-opus-4-6)")
    p.add_argument("--base-url", type=str, dest="base_url",
                   help="Base URL for OpenAI-compatible endpoint")
    p.add_argument("--api-key", type=str, dest="api_key",
                   help="API key for LLM provider (overrides environment variables)")


def _embedding_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embedding-model", type=str, dest="embedding_model", default=None,
                   help="Embedding model for retrieval. Omit for KEYWORD-ONLY (BM25) "
                        "grounding — no embedding provider or key needed; name a model "
                        "(e.g. gemini-embedding-001, text-embedding-3-small) for dense.")
    p.add_argument("--embedding-api-key", type=str, dest="embedding_api_key",
                   help="API key for the embedding model (defaults to --api-key / env). "
                        "Ignored when --base-url is set without --embedding-base-url "
                        "(the proxy key is reused).")
    p.add_argument("--embedding-base-url", type=str, dest="embedding_base_url",
                   help="OpenAI-compatible endpoint for the EMBEDDINGS only, authenticated "
                        "with --embedding-api-key (or --api-key). Lets the chat model run "
                        "direct (or through --base-url) while embeddings go elsewhere.")


def _extras_flags(p: argparse.ArgumentParser, prog: str, *, mcp: bool = True,
                  agents: bool = False) -> None:
    if agents:
        p.add_argument("--agents", type=str, nargs="+", dest="agent_files", metavar="AGENT_FILE",
                       help="Path(s) to Python files containing custom BaseAnalysisAgent "
                            "subclasses. All subclasses found in each file are registered "
                            f"automatically. Example: {prog} --agents ./my_xrd_agent.py")
    p.add_argument("--skills", type=str, nargs="+", dest="skill_files", metavar="SKILL_FILE",
                   help="Path(s) to custom skill .md files providing domain-specific guidance "
                        "(fitting strategy, interpretation rules). Skills are made available "
                        f"by name to the agents. Example: {prog} --skills ./raman_skill.md")
    p.add_argument("--tools", type=str, nargs="+", dest="tool_files", metavar="TOOL_FILE",
                   help="Path(s) to Python files of domain-specific tool functions to expose "
                        "to the orchestrator's LLM loop. Each file must define: (1) a list of "
                        "OpenAI-format tool schemas named 'tool_schemas', 'openai_schemas', or "
                        "any top-level list of OpenAI function dicts; (2) a factory function "
                        "named 'create_tool_functions' (or any function ending in "
                        "'_tool_functions') that accepts data and returns a dict mapping tool "
                        f"names to callables. Example: {prog} --tools ./image_tools.py")
    if mcp:
        p.add_argument("--mcp", type=str, nargs="+", dest="mcp_servers", metavar="MCP_CONFIG",
                       help="MCP server configurations. Each entry can be:\n"
                            "  - A JSON config file ({\"name\":\"...\", \"command\":[\"...\"], "
                            "\"url\":\"...\", \"transport\":\"sse|http\", \"headers\":{...}, "
                            "\"env\":{}}; ${VAR} in header values is expanded from the environment)\n"
                            "  - stdio shorthand:  stdio:name:command,arg1,arg2\n"
                            "  - SSE shorthand:    sse:name:http://host:port/sse\n"
                            "  - Streamable HTTP shorthand: http:name:https://host/mcp\n"
                            f"Example: {prog} --mcp stdio:fs:npx,-y,@modelcontextprotocol/"
                            "server-filesystem,/tmp")


def _session_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--session-dir", type=str, dest="session_dir",
                   help="Session directory for outputs (default: auto-generated)")
    p.add_argument("--restore", action="store_true",
                   help="Restore from a previous checkpoint: with --session-dir that "
                        "session, otherwise a picker over the sessions in this folder")


def _deprecated_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--local-model", type=str, dest="local_model", help=argparse.SUPPRESS)
    p.add_argument("--google-api-key", type=str, dest="google_api_key", help=argparse.SUPPRESS)


def _apply_deprecations(args) -> None:
    if getattr(args, "local_model", None):
        print("⚠️  Warning: '--local-model' is deprecated. Use '--base-url' instead.")
        if not args.base_url:
            args.base_url = args.local_model
    if getattr(args, "google_api_key", None):
        print("⚠️  Warning: '--google-api-key' is deprecated. Use '--api-key' instead.")
        if not args.api_key:
            args.api_key = args.google_api_key


def _register_mp_key(mp_api_key: Optional[str]) -> None:
    if mp_api_key:
        import scilink
        scilink.set_api_key("materials_project", mp_api_key)


_ENV_EPILOG = """
Environment Variables:
  SCILINK_API_KEY          API key for internal proxy (pairs with --base-url)
  ANTHROPIC_API_KEY        Anthropic API key (CLAUDE_API_KEY is an alias)
  OPENAI_API_KEY           OpenAI API key
  GEMINI_API_KEY           Google Gemini API key (GOOGLE_API_KEY is an alias)
  FUTUREHOUSE_API_KEY      FutureHouse API key for literature search (optional)
"""


# ── Meta (Mission Control) ───────────────────────────────────────

class MetaAdapter(ModeAdapter):
    key = "meta"
    description = ("SciLink Mission Control — one chat surface that routes your research "
                   "goal to the analyze, plan and simulate specialists")
    epilog = """
Examples:
  scilink                                   # launch (same as: scilink explore)
  scilink --message "Analyze grains.tif then plan a follow-up campaign"
  scilink --mode autonomous                 # specialists run without pausing
  scilink -p "Analyze ./grains.tif" --output-format json   # headless, one task
  scilink --resume                          # pick a past session in this folder
  scilink --skills ./raman_skill.md --mcp stdio:fs:npx,-y,@modelcontextprotocol/server-filesystem,/tmp

Modes — the meta has two levels (a delegation runs a specialist through its
one-shot run_task, so the specialists' step-by-step co-pilot mode does not
apply here):
  autopilot (default)  Specialists pause at decision points for you to
                        approve / edit plans and outputs.
  autonomous            Specialists run end to end without pausing.
""" + _ENV_EPILOG

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        _model_flags(p)
        _embedding_flags(p)
        p.add_argument("--futurehouse-api-key", type=str, dest="futurehouse_api_key",
                       help="FutureHouse API key (or set FUTUREHOUSE_API_KEY env var). "
                            "Enables literature search in delegated analysis.")
        p.add_argument("--mode", type=str, dest="mode", choices=["autopilot", "autonomous"],
                       default="autopilot",
                       help="Autonomy mode (default: autopilot). The meta has two levels, "
                            "not the modes' three.")
        p.add_argument("--message", type=str, dest="initial_message",
                       help="Optional initial message to seed the session.")
        p.add_argument("--knowledge-dir", type=str, dest="knowledge_dir",
                       help="Stable knowledge/KB for planning delegations: a directory OR "
                            "the name of a knowledge base from the persistent store "
                            "(scilink kb list).")
        _extras_flags(p, "scilink")
        _session_flags(p)

    def validate_args(self, parser, args) -> None:
        super().validate_args(parser, args)
        if args.knowledge_dir and not Path(args.knowledge_dir).exists():
            from scilink.knowledge.kb_store import resolve_knowledge_source
            try:
                resolve_knowledge_source(args.knowledge_dir)
            except FileNotFoundError as e:
                parser.error(str(e))

    def build(self, args, creds, session_dir, *, restore, extras):
        from scilink.agents.meta_agent.meta_orchestrator import MetaMode, MetaOrchestratorAgent
        return MetaOrchestratorAgent(
            base_dir=str(session_dir),
            api_key=creds.api_key,
            model_name=args.model,
            base_url=creds.base_url,
            embedding_model=args.embedding_model,
            embedding_api_key=args.embedding_api_key,
            embedding_base_url=args.embedding_base_url,
            futurehouse_api_key=resolve_optional_key(args.futurehouse_api_key, "FUTUREHOUSE_API_KEY"),
            restore_checkpoint=restore,
            meta_mode=MetaMode[_ENUM_NAME[self.autonomy_from_args(args)]],
            knowledge_dir=args.knowledge_dir,
        )

    def get_autonomy(self, agent) -> str:
        return _level_from_enum(agent.meta_mode)

    def set_autonomy(self, agent, level: str) -> None:
        from scilink.agents.meta_agent.meta_orchestrator import MetaMode
        agent.set_meta_mode(MetaMode[_ENUM_NAME[level]])

    def status_fields(self, agent):
        children = getattr(agent, "_children", None) or {}
        return [("Specialists active", ", ".join(sorted(children)) or "none"),
                ("Delegations", str(len(getattr(agent, "_delegation_ledger", None) or [])))]

    def extra_commands(self):
        return [Command("/delegations", "The delegation ledger", _delegations,
                        aliases=("/children",))]

    def initial_turns(self, args, agent):
        return [args.initial_message] if getattr(args, "initial_message", None) else []

    def headless_run(self, agent, task):
        reply = agent.chat(task)
        ledger = getattr(agent, "_delegation_ledger", None) or []
        files = [f for e in ledger for f in (e.get("files_produced") or [])]
        findings = [k for e in ledger for k in (e.get("key_findings") or [])]
        failed = [e for e in ledger if str(e.get("status", "")).lower() in ("error", "failed")]
        return {"status": "error" if failed and not reply else "success", "task": task,
                "summary": reply or "", "files_produced": files, "key_findings": findings,
                "delegations": len(ledger),
                "warnings": [e.get("error") for e in failed if e.get("error")]}


def _delegations(shell, arg: str) -> None:
    from scilink.server.delegations import delegation_view
    view = delegation_view(shell.agent, str(shell.session_dir))
    rows = view.get("delegations") or []
    if not rows:
        shell.console.print("[dim]No delegations yet — the meta routes your goal to a specialist.[/]")
        return
    for d in rows:
        mark = {"success": "[green]✓[/]", "error": "[red]✗[/]", "running": "[cyan]…[/]"}.get(
            str(d.get("status")), "[dim]·[/]")
        ctx = f"  [dim]←#{','.join(map(str, d['context_from']))}[/]" if d.get("context_from") else ""
        shell.console.print(f"{mark} [bold]#{d.get('index')}[/] {d.get('mode')} — "
                            f"{(d.get('label') or d.get('task') or '')[:90]}{ctx}")
        if d.get("summary"):
            shell.console.print(f"      [dim]{d['summary'][:160]}[/]")


# ── Analyze ──────────────────────────────────────────────────────

class AnalyzeAdapter(ModeAdapter):
    key = "analyze"
    description = "SciLink Analysis Orchestrator - Interactive AI Analysis Agent"
    epilog = """
Examples:
  scilink analyze                                        # co-pilot, empty chat
  scilink analyze --data ./sample.tif --metadata ./metadata.json
  scilink analyze --mode autopilot --data ./data/
  scilink analyze --mode autonomous --data ./sample.npy --metadata ./description.txt
  scilink analyze -p "Analyze ./grains.tif" --output-format json   # headless

Analysis Modes:
  co-pilot (default)   Human leads, AI assists. Reviews all agent selections.
  autopilot            AI leads, human approves. AI proceeds with reasonable defaults.
  autonomous           Full autonomy. AI selects agents and runs without confirmation.

Supported Data Types:
  Microscopy:    .tif, .tiff, .png, .jpg, .jpeg, .bmp
  Spectroscopy:  .npy (3D hyperspectral)
  Curves:        .npy (1D/2D), .csv, .txt

Metadata Options:
  --metadata file.json     Load structured JSON metadata
  --metadata file.txt      Convert natural language to metadata
  (or provide metadata interactively during session)
""" + _ENV_EPILOG

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        _model_flags(p)
        p.add_argument("--mp-api-key", type=str, dest="mp_api_key",
                       help="Materials Project API key (or set MP_API_KEY env var). "
                            "Enables the MP tool-resolver in structure generation.")
        p.add_argument("--futurehouse-api-key", type=str, dest="futurehouse_api_key",
                       help="FutureHouse API key for novelty assessment / literature search "
                            "(or set FUTUREHOUSE_API_KEY env var). Optional.")
        p.add_argument("--mode", type=str, choices=["co-pilot", "autopilot", "autonomous"],
                       default="co-pilot", help="Analysis mode (default: co-pilot)")
        p.add_argument("--data", type=str, dest="data_path", help="Path to data file or directory")
        p.add_argument("--metadata", type=str, dest="metadata_path",
                       help="Path to metadata file (.json or .txt)")
        _extras_flags(p, "scilink analyze", agents=True)
        _session_flags(p)
        _deprecated_flags(p)

    def validate_args(self, parser, args) -> None:
        _apply_deprecations(args)
        _register_mp_key(args.mp_api_key)
        if args.data_path and not Path(args.data_path).exists():
            parser.error(f"--data path does not exist: {args.data_path}")
        if args.metadata_path and not Path(args.metadata_path).exists():
            parser.error(f"--metadata path does not exist: {args.metadata_path}")
        super().validate_args(parser, args)

    def build(self, args, creds, session_dir, *, restore, extras):
        from scilink.agents.exp_agents.analysis_orchestrator import (AnalysisMode,
                                                                     AnalysisOrchestratorAgent)
        return AnalysisOrchestratorAgent(
            base_dir=str(session_dir),
            api_key=creds.api_key,
            model_name=args.model,
            base_url=creds.base_url,
            analysis_mode=AnalysisMode[_ENUM_NAME[self.autonomy_from_args(args)]],
            restore_checkpoint=restore,
            futurehouse_api_key=resolve_optional_key(args.futurehouse_api_key, "FUTUREHOUSE_API_KEY"),
        )

    def get_autonomy(self, agent) -> str:
        return _level_from_enum(agent.analysis_mode)

    def set_autonomy(self, agent, level: str) -> None:
        from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
        agent.set_analysis_mode(AnalysisMode[_ENUM_NAME[level]])

    def status_fields(self, agent):
        return [("Current data", str(getattr(agent, "current_data_path", None) or "none")),
                ("Data type", str(getattr(agent, "current_data_type", None) or "none")),
                ("Selected agent", str(getattr(agent, "selected_agent_id", None))),
                ("Metadata", "yes" if getattr(agent, "current_metadata", None) else "no"),
                ("Analyses completed", str(len(getattr(agent, "analysis_results", None) or [])))]

    def extra_commands(self):
        return [Command("/agents", "Available analysis agents", _agents),
                Command("/schema", "The metadata JSON schema", _schema)]

    def initial_turns(self, args, agent):
        """The old ``_process_initial_inputs``: seed the chat from --data /
        --metadata according to the autonomy level."""
        data, meta = getattr(args, "data_path", None), getattr(args, "metadata_path", None)
        if not data and not meta:
            return []
        level = self.get_autonomy(agent)
        turns: List[str] = []
        if level == "autonomous" and data and meta:
            verb = "load" if Path(meta).suffix.lower() == ".json" else "convert"
            return [f"Analyze the data at {Path(data).absolute()}. First examine the data, "
                    f"then {verb} the metadata from {Path(meta).absolute()}, then select the "
                    "appropriate agent based on the data type and metadata, and finally run "
                    "the analysis. Execute the complete workflow."]
        if data:
            turns.append(f"Examine the data at {Path(data).absolute()}")
        if meta:
            if Path(meta).suffix.lower() == ".json":
                turns.append(f"Load the metadata from {Path(meta).absolute()}")
            else:
                turns.append(f"Convert the text description to metadata from {Path(meta).absolute()}")
        if level == "autopilot" and data and meta:
            turns.append("Based on the data type and metadata, recommend the appropriate "
                         "analysis agent and explain your reasoning.")
        return turns

    def headless_run(self, agent, task):
        from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
        return agent.run_task(task, autonomy=AnalysisMode.AUTONOMOUS)


def _agents(shell, arg: str) -> None:
    tools = shell.agent.tools
    for agent_id, name in tools.AGENT_NAMES.items():
        sel = "  [green]← selected[/]" if agent_id == shell.agent.selected_agent_id else ""
        shell.console.print(f"[bold]{agent_id}[/]: {name}{sel}")
        shell.console.print(f"    [dim]{tools.AGENT_DESCRIPTIONS[agent_id]}[/]")


def _schema(shell, arg: str) -> None:
    from scilink.agents.exp_agents.metadata_converter import METADATA_SCHEMA_DICT
    shell.console.print_json(json.dumps(METADATA_SCHEMA_DICT))


# ── Plan ─────────────────────────────────────────────────────────

class PlanAdapter(ModeAdapter):
    key = "plan"
    description = "SciLink Planning Orchestrator - Interactive AI Research Agent"
    epilog = """
Examples:
  scilink plan                                            # co-pilot, empty chat
  scilink plan --autonomy autopilot --data-dir ./experimental_results
  scilink plan --autonomy autonomous --data-dir ./data --knowledge-dir ./papers --code-dir ./code
  scilink plan --session-dir ./my_campaign                # named session directory
  scilink plan --restore                                  # pick a past session
  scilink plan -p "Design a screening campaign for ..." --output-format json

Autonomy Levels:
  co-pilot (default)  Human leads, AI assists. Reviews every step.
  autopilot           AI leads, human monitors. Human reviews plans/code only.
  autonomous          Full autonomy. No human review, AI chains all tools.
  Note: autopilot and autonomous modes require --data-dir to be specified.

Recommended project layout (co-pilot):
  papers/               PDFs, scientific literature
  experimental_results/ CSV/XLSX data files
  code/                 (optional) scripts, API docs
""" + _ENV_EPILOG

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        _model_flags(p)
        _embedding_flags(p)
        p.add_argument("--futurehouse-api-key", type=str, dest="futurehouse_api_key",
                       help="FutureHouse API key for literature search (or set "
                            "FUTUREHOUSE_API_KEY env var). Optional.")
        p.add_argument("--autonomy", type=str, choices=["co-pilot", "autopilot", "autonomous"],
                       default="co-pilot",
                       help="Autonomy level (default: co-pilot). Higher levels require --data-dir.")
        p.add_argument("--objective", type=str, dest="objective",
                       help="Research objective (asked interactively when omitted)")
        p.add_argument("--data-dir", type=str, dest="data_dir",
                       help="Path to experimental data directory (required for autopilot/autonomous)")
        p.add_argument("--knowledge-dir", type=str, dest="knowledge_dir",
                       help="Papers/literature directory, OR the name of a knowledge base "
                            "from the persistent store (see: scilink kb list)")
        p.add_argument("--code-dir", type=str, dest="code_dir",
                       help="Path to code/API documentation directory (optional)")
        _extras_flags(p, "scilink plan")
        _session_flags(p)
        _deprecated_flags(p)

    def validate_args(self, parser, args) -> None:
        _apply_deprecations(args)
        if args.autonomy in ("autopilot", "autonomous") and not args.data_dir \
                and getattr(args, "print_task", None) is None:
            parser.error(f"--data-dir is required for {args.autonomy} mode.\n"
                         f"Example: scilink plan --autonomy {args.autonomy} "
                         "--data-dir ./experimental_results")
        if args.data_dir and not Path(args.data_dir).exists():
            parser.error(f"--data-dir path does not exist: {args.data_dir}")
        if args.knowledge_dir:
            from scilink.knowledge.kb_store import resolve_knowledge_source
            try:
                resolve_knowledge_source(args.knowledge_dir)
            except FileNotFoundError as e:
                parser.error(str(e))
        if args.code_dir and not Path(args.code_dir).exists():
            parser.error(f"--code-dir path does not exist: {args.code_dir}")
        super().validate_args(parser, args)

    def prepare(self, args, ask: Ask) -> Dict[str, Any]:
        objective = getattr(args, "objective", None)
        if not objective and not getattr(args, "restore", False):
            objective = ask("Research objective (e.g. optimize reaction yield): ").strip()
        return {"objective": objective or "Optimize experimental conditions"}

    def build(self, args, creds, session_dir, *, restore, extras):
        from scilink.agents.planning_agents.planning_orchestrator import (
            AutonomyLevel, PlanningOrchestratorAgent)
        return PlanningOrchestratorAgent(
            objective=extras.get("objective") or "Optimize experimental conditions",
            base_dir=str(session_dir),
            api_key=creds.api_key,
            model_name=args.model,
            base_url=creds.base_url,
            embedding_model=args.embedding_model,
            embedding_api_key=args.embedding_api_key,
            embedding_base_url=args.embedding_base_url,
            futurehouse_api_key=resolve_optional_key(args.futurehouse_api_key, "FUTUREHOUSE_API_KEY"),
            autonomy_level=AutonomyLevel[_ENUM_NAME[self.autonomy_from_args(args)]],
            data_dir=args.data_dir,
            knowledge_dir=args.knowledge_dir,
            code_dir=args.code_dir,
            restore_checkpoint=restore,
        )

    def get_autonomy(self, agent) -> str:
        return _level_from_enum(agent.autonomy_level)

    def set_autonomy(self, agent, level: str) -> None:
        from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
        agent.set_autonomy_level(AutonomyLevel[_ENUM_NAME[level]])

    def status_fields(self, agent):
        rows = [("Objective", str(getattr(agent, "objective", ""))),
                ("Active script", Path(agent.active_scalarizer_script).name
                 if getattr(agent, "active_scalarizer_script", None) else "none"),
                ("Input columns", str(getattr(agent, "expected_input_columns", None))),
                ("Target columns", str(getattr(agent, "expected_target_columns", None)))]
        bo = getattr(agent, "bo_data_path", None)
        if bo is not None and Path(bo).exists():
            try:
                import pandas as pd
                rows.append(("Data points", str(len(pd.read_csv(bo)))))
            except Exception:  # noqa: BLE001
                rows.append(("Data points", "unreadable"))
        else:
            rows.append(("Data points", "0"))
        return rows

    def extra_commands(self):
        return [Command("/objective", "Show the research objective", _objective)]

    def initial_turns(self, args, agent):
        """The old ``_process_initial_inputs``: survey / run the workspace
        according to the autonomy level (co-pilot starts empty). A restored
        session continues where it left off — the survey / pipeline is not
        re-run over the loaded campaign (seen live: a restored autopilot
        session started by recommending next steps for the default
        objective)."""
        if getattr(args, "restore", False):
            return []
        level = self.get_autonomy(agent)
        data, kb, code = args.data_dir, args.knowledge_dir, args.code_dir
        if level == "co-pilot" or not (data or kb):
            return []
        objective = getattr(agent, "objective", "")
        if level == "autonomous":
            parts = [f"Execute the complete research workflow for objective: '{objective}'.",
                     "Step 1: Survey the workspace using list_workspace_files to understand "
                     "available data."]
            parts.append(f"Step 2: Run economic analysis using knowledge from {kb} and "
                         f"experimental data from {data} to assess viability." if kb
                         else "Step 2: Skip economic analysis (no knowledge directory provided).")
            parts.append(f"Step 3: Generate an initial experimental plan based on the objective, "
                         f"available data in {data}" + (f", and literature in {kb}" if kb else "") + ".")
            parts.append(f"Step 4: Generate implementation code using the code knowledge base in {code}."
                         if code else "Step 4: Skip code generation (no code directory provided).")
            parts.append("Step 5: Save a checkpoint to preserve the campaign state.")
            parts.append("Execute ALL steps without stopping for confirmation. Chain tool calls "
                         "as needed to complete the entire workflow.")
            return [" ".join(parts)]
        rec = (f"Based on the workspace contents, recommend the best next steps for achieving "
               f"the objective: '{objective}'. Consider the following options and recommend "
               "which to do first:\n")
        if kb:
            rec += f"- Run economic/TEA analysis using papers in {kb}\n"
        rec += ("- Generate an experimental plan based on available data\n"
                "- Analyze existing experimental results\n\nProvide a clear recommendation "
                "with reasoning, then proceed with the recommended action.")
        return ["Survey the workspace using list_workspace_files. Report what data files, "
                "papers, and other resources are available.", rec]

    def headless_run(self, agent, task):
        from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
        return agent.run_task(task, autonomy=AutonomyLevel.AUTONOMOUS)


def _objective(shell, arg: str) -> None:
    shell.console.print(f"[bold]Objective:[/] {getattr(shell.agent, 'objective', '')}")


# ── Simulate ─────────────────────────────────────────────────────

class SimulateAdapter(ModeAdapter):
    key = "simulate"
    description = "SciLink Simulation Orchestrator - Interactive VASP DFT input prep"
    epilog = """
Examples:
  scilink simulate                                        # co-pilot, empty chat
  scilink simulate --request "Build a rutile TiO2 supercell with one O vacancy"
  scilink simulate --mode autopilot
  scilink simulate -p "Generate VASP inputs for bulk Si" --output-format json

Modes:
  co-pilot (default)   Human leads, AI assists. Confirms before each tool call.
  autopilot            AI leads with defaults; surfaces significant decisions.
  autonomous           Full autonomy. AI executes without confirmation.

Scope (for now):
  • VASP DFT input prep only (LAMMPS support is planned follow-up).
  • Local prep only — you run VASP elsewhere and bring back outputs for analysis.
""" + _ENV_EPILOG + "  MP_API_KEY               Materials Project API key (enables MP tool-resolver)\n"

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        _model_flags(p)
        p.add_argument("--mp-api-key", type=str, dest="mp_api_key",
                       help="Materials Project API key (or set MP_API_KEY env var). "
                            "Enables the MP tool-resolver in structure generation.")
        p.add_argument("--futurehouse-api-key", type=str, dest="futurehouse_api_key",
                       help="FutureHouse API key (or set FUTUREHOUSE_API_KEY env var). "
                            "Enables INCAR literature validation.")
        p.add_argument("--mode", type=str, dest="mode",
                       choices=["co-pilot", "autopilot", "autonomous"], default="co-pilot",
                       help="Autonomy mode (default: co-pilot)")
        p.add_argument("--request", type=str, dest="initial_request",
                       help="Optional initial request to seed the session "
                            "(e.g., \"Build a rutile TiO2 supercell\").")
        _extras_flags(p, "scilink simulate", mcp=False)
        _session_flags(p)

    def validate_args(self, parser, args) -> None:
        _register_mp_key(args.mp_api_key)
        super().validate_args(parser, args)

    def build(self, args, creds, session_dir, *, restore, extras):
        from scilink.agents.sim_agents.simulation_orchestrator import (SimulationMode,
                                                                      SimulationOrchestratorAgent)
        return SimulationOrchestratorAgent(
            base_dir=str(session_dir),
            api_key=creds.api_key,
            model_name=args.model,
            base_url=creds.base_url,
            simulation_mode=SimulationMode[_ENUM_NAME[self.autonomy_from_args(args)]],
            restore_checkpoint=restore,
            mp_api_key=resolve_optional_key(args.mp_api_key, "MP_API_KEY", "MATERIALS_PROJECT_API_KEY"),
            futurehouse_api_key=resolve_optional_key(args.futurehouse_api_key, "FUTUREHOUSE_API_KEY"),
        )

    def get_autonomy(self, agent) -> str:
        return _level_from_enum(agent.simulation_mode)

    def set_autonomy(self, agent, level: str) -> None:
        from scilink.agents.sim_agents.simulation_orchestrator import SimulationMode
        agent.set_simulation_mode(SimulationMode[_ENUM_NAME[level]])

    def status_fields(self, agent):
        return [("Structures", str(len(getattr(agent, "generated_structures", None) or []))),
                ("Default calc params", str(getattr(agent, "default_calc_params", None) or "none"))]

    def extra_commands(self):
        return [Command("/structures", "Structures generated this session", _structures)]

    def initial_turns(self, args, agent):
        return [args.initial_request] if getattr(args, "initial_request", None) else []

    def headless_run(self, agent, task):
        from scilink.agents.sim_agents.simulation_orchestrator import SimulationMode
        return agent.run_task(task, autonomy=SimulationMode.AUTONOMOUS)


def _structures(shell, arg: str) -> None:
    structures = getattr(shell.agent, "generated_structures", None) or []
    shell.console.print(f"[bold]Structures generated this session:[/] {len(structures)}")
    for s in structures:
        shell.console.print(f"  • [bold]{s.get('slug')}[/]: {s.get('description')}")
        shell.console.print(f"      [dim]structure:[/] {s.get('structure_path')}")
        files = s.get("input_files") or {}
        if files:
            shell.console.print(f"      [dim]inputs:[/] {', '.join(sorted(files))}")


ADAPTERS = {
    "meta": MetaAdapter,
    "analyze": AnalyzeAdapter,
    "plan": PlanAdapter,
    "simulate": SimulateAdapter,
}
