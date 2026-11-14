"""``scilink <mode> -p "task"`` — one task, no prompt, machine-readable output.

Runs the mode's ``run_task`` under full autonomy with every human-feedback
question auto-accepted, narrates on stderr, and prints the result on
stdout as text, markdown or JSON. Exit status follows the task's status
(``exit_code``): 0 success, 1 error, and with ``--strict`` 3 for a result
to read before using.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

from rich.console import Console

from scilink import hitl

from . import bootstrap
from .channel import AutoAcceptChannel

OUTPUT_FORMATS = ("text", "markdown", "json")


def _no_ask(prompt: str = "", *, secret: bool = False, default: str = "") -> str:
    return default


def render_result(result: Dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(result, indent=2, default=str)
    lines = [result.get("summary") or ""]
    findings = result.get("key_findings") or []
    if findings:
        lines += ["", "Key findings:" if fmt == "text" else "## Key findings"]
        lines += [f"- {f}" for f in findings]
    files = result.get("files_produced") or []
    if files:
        lines += ["", "Files:" if fmt == "text" else "## Files"]
        lines += [f"- {f}" for f in files]
    verification = result.get("verification") or {}
    if verification.get("analyses"):
        lines += ["", "Verification:" if fmt == "text" else "## Verification",
                  f"- {verification.get('verified', 0)} of {verification['analyses']} "
                  "analyses verified"]
        lines += [f"- not verified: {u.get('analysis_id')} ({u.get('reason')})"
                  for u in verification.get("unverified") or []]
        lines += [f"- failed: {f.get('analysis_id')} (status={f.get('status')})"
                  for f in verification.get("failed") or []]
    warnings = result.get("warnings") or []
    if warnings:
        lines += ["", "Warnings:" if fmt == "text" else "## Warnings"]
        lines += [f"- {w}" for w in warnings]
    if result.get("error"):
        lines += ["", f"Error: {result['error']}"]
    return "\n".join(lines).rstrip() + "\n"


# Exit statuses of a ``-p`` run. 2 is a start-up failure (credentials, a
# bad flag), 130 an interrupt.
EXIT_OK, EXIT_ERROR, EXIT_RESERVATIONS = 0, 1, 3


def exit_code(result: Dict[str, Any], *, strict: bool = False) -> int:
    """0 when the task succeeded, 1 when it failed. ``--strict`` adds 3,
    "finished, read the result before using it":

    * a ``partial`` status — a meta turn in which some delegations failed
      and some did not (the reply may account for it, e.g. a retry that
      worked, which the ledger cannot tell);
    * a success that is not backed by verified analyses — one was not
      verified, one failed, or none ran at all (a task that could not find
      its data still ends in a reply, and a reply is a success). This is
      judged only on a result that carries a ``verification``: analyze mode
      always does, the meta when the turn delegated an analysis, plan and
      simulate never.

    Without ``--strict`` both exit 0, as they always have; the status and
    the count are in the result either way."""
    status = str(result.get("status", "")).lower()
    if status == "partial":
        return EXIT_RESERVATIONS if strict else EXIT_OK
    if status not in ("success", "ok", "completed", "done"):
        return EXIT_ERROR
    verification = result.get("verification")
    if strict and isinstance(verification, dict) and verification.get("all_verified") is not True:
        return EXIT_RESERVATIONS
    return EXIT_OK


def run(adapter, args, task: str, *, console: Console = None) -> int:
    err = console or Console(file=sys.stderr, highlight=False)
    fmt = getattr(args, "output_format", "text") or "text"
    try:
        creds = bootstrap.resolve_llm_credentials(args.model, getattr(args, "base_url", None),
                                                  getattr(args, "api_key", None), _no_ask)
        session_dir = Path(args.session_dir) if getattr(args, "session_dir", None) \
            else bootstrap.new_session_dir(adapter.key)
        session_dir.mkdir(parents=True, exist_ok=True)
        from scilink.sessions import register_session
        register_session(session_dir, adapter.key)
        bootstrap.ensure_sandbox_consent(assume_yes=bool(getattr(args, "yes", False)),
                                         ask=_no_ask, console=err)
        extras = adapter.prepare(args, _no_ask)
        # Narration goes to stderr so stdout carries only the result.
        with contextlib.redirect_stdout(sys.stderr):
            agent = adapter.build(args, creds, session_dir,
                                  restore=bool(getattr(args, "restore", False)), extras=extras)
            bootstrap.register_extras(
                agent, skill_files=getattr(args, "skill_files", None) or (),
                tool_files=getattr(args, "tool_files", None) or (),
                agent_files=getattr(args, "agent_files", None) or (),
                mcp_entries=getattr(args, "mcp_servers", None) or (), console=err)
            adapter.set_autonomy(agent, "autonomous")
            hitl.set_default_channel(AutoAcceptChannel())
            try:
                result = adapter.headless_run(agent, task, context=adapter.headless_context(args))
            finally:
                hitl.set_default_channel(None)
    except bootstrap.BootstrapError as e:
        err.print(f"[red]{e}[/]")
        return 2
    except KeyboardInterrupt:
        err.print("[yellow]interrupted[/]")
        return 130
    sys.stdout.write(render_result(result, fmt))
    sys.stdout.flush()
    return exit_code(result, strict=bool(getattr(args, "strict", False)))
