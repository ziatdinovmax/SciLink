"""Slash commands — one set across the four modes, plus each mode's extras.

Names are unified: ``/status`` and ``/mode`` everywhere (plan's old
``/state`` and ``/autonomy`` are aliases). The bodies read the same
agent-level helpers the web UI's panels read (skills, memory, tools,
files, delegations), so the two surfaces list the same things.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from rich.table import Table

from scilink.ui import vocabulary as V


@dataclass
class Command:
    name: str                              # "/status"
    help: str
    run: Callable[["object", str], None]   # run(shell, arg)
    aliases: tuple = ()
    arg_hint: str = ""                     # "<path>" for the help table
    completes_paths: bool = False


class Registry:
    def __init__(self) -> None:
        self._commands: Dict[str, Command] = {}
        self._aliases: Dict[str, str] = {}

    def add(self, cmd: Command) -> None:
        self._commands[cmd.name] = cmd
        for a in cmd.aliases:
            self._aliases[a] = cmd.name

    def extend(self, cmds) -> None:
        for c in cmds:
            self.add(c)

    def get(self, name: str) -> Optional[Command]:
        name = self._aliases.get(name, name)
        return self._commands.get(name)

    def names(self) -> List[str]:
        return sorted(self._commands)

    def commands(self) -> List[Command]:
        return [self._commands[n] for n in self.names()]

    def dispatch(self, shell, line: str) -> bool:
        """Run the command on ``line``; False when it is not a command."""
        head, _, arg = line.strip().partition(" ")
        cmd = self.get(head.lower())
        if cmd is None:
            return False
        cmd.run(shell, arg.strip())
        return True


# ── core commands ────────────────────────────────────────────────

def _help(shell, arg: str) -> None:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column()
    for c in shell.commands.commands():
        name = f"{c.name} {c.arg_hint}".strip()
        alias = f"  [dim]({', '.join(c.aliases)})[/]" if c.aliases else ""
        t.add_row(name, c.help + alias)
    shell.console.print(t)
    shell.console.print(
        "\n[dim]Keys: Enter sends · Alt+Enter newline · Ctrl+C stops a running turn "
        "(clears the line at the prompt) · Ctrl+O toggles verbose output, also mid-turn · "
        "Ctrl+D quits[/]")


def _status(shell, arg: str) -> None:
    m = V.mode(shell.mode)
    rows = [
        ("Mode", f"{m['emoji']}  {m['name']}"),
        ("Autonomy", shell.adapter.get_autonomy(shell.agent)),
        ("Session", str(shell.session_dir)),
        ("Model", shell.model_label()),
        ("Messages", str(getattr(shell.agent, "message_count", "?"))),
        ("Human feedback", "on" if _feedback_on(shell.agent) else "off"),
        ("Verbose output", "on" if shell.renderer.verbose else "off"),
    ]
    from .shell import context_usage
    usage = context_usage(shell.agent)
    if usage:
        rows.append(("Context", f"{usage[0]} of {usage[1]} messages before history is trimmed "
                                f"({100 * usage[0] // usage[1]}%)"))
    rows += shell.adapter.status_fields(shell.agent)
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    for k, v in rows:
        t.add_row(k, str(v))
    shell.console.print(t)


def _feedback_on(agent) -> bool:
    if hasattr(agent, "get_human_feedback_setting"):
        try:
            return bool(agent.get_human_feedback_setting())
        except Exception:  # noqa: BLE001
            pass
    return bool(getattr(agent, "_enable_human_feedback", False))


def _mode(shell, arg: str) -> None:
    options = V.autonomy_options(shell.mode)
    current = shell.adapter.get_autonomy(shell.agent)
    if not arg:
        shell.console.print(f"Autonomy: [bold]{current}[/]  [dim](/mode {'|'.join(options)})[/]")
        return
    level = arg.strip().lower()
    if level not in options:
        shell.console.print(f"[red]Unknown autonomy level {arg!r}[/] — one of {', '.join(options)}")
        return
    shell.adapter.set_autonomy(shell.agent, level)
    shell.console.print(f"[green]✓[/] autonomy set to [bold]{level}[/] · "
                        f"human feedback {'on' if _feedback_on(shell.agent) else 'off'}")


def _tools(shell, arg: str) -> None:
    from scilink.server.tools_api import tool_inventory
    builtin = list(getattr(getattr(shell.agent, "tools", None), "functions_map", {}) or {})
    shell.console.print(f"[bold]Built-in tools[/] ({len(builtin)}): " + ", ".join(builtin))
    inv = tool_inventory(shell.agent)
    if inv.get("external"):
        shell.console.print("[bold]Custom tools[/]: " + ", ".join(t["name"] for t in inv["external"]))
    for srv in inv.get("mcp_servers") or []:
        shell.console.print(f"[bold]MCP {srv['name']}[/] ({srv.get('transport')}): "
                            + ", ".join(srv.get("tools") or []))


def _mcp(shell, arg: str) -> None:
    from .bootstrap import connect_mcp_entries
    if not arg:
        shell.console.print("Usage: /mcp <config>  — a JSON file, stdio:name:cmd,arg..., "
                            "sse:name:url or http:name:url")
        return
    connect_mcp_entries(shell.agent, [arg], shell.console)


def _skill(shell, arg: str) -> None:
    from .bootstrap import register_skill_files
    if not arg:
        shell.console.print("Usage: /skill <path-to-skill.md>")
        return
    register_skill_files(shell.agent, [arg], shell.console)


def _tool(shell, arg: str) -> None:
    from .bootstrap import register_tool_files
    if not arg:
        shell.console.print("Usage: /tool <path-to-tools.py>")
        return
    register_tool_files(shell.agent, [arg], shell.console)


def _skills(shell, arg: str) -> None:
    from scilink.server.skills_api import skill_catalog
    cat = skill_catalog(shell.agent)
    if not cat.get("skills_supported", True):
        shell.console.print("[dim]This mode's agent does not take skills.[/]")
        return
    for dom in cat.get("builtin") or []:
        t = Table(title=f"{dom.get('label') or dom.get('domain')}", show_header=False,
                  title_justify="left", box=None, padding=(0, 2))
        t.add_column(style="bold")
        t.add_column()
        for s in dom.get("skills") or []:
            t.add_row(s.get("name"), s.get("description") or "")
        shell.console.print(t)
    custom = cat.get("custom") or []
    if custom:
        shell.console.print("[bold]Custom skills[/]: " + ", ".join(s["name"] for s in custom))


def _memory(shell, arg: str) -> None:
    from scilink.server.memory_api import memory_overview, set_enabled
    if arg.lower() in ("on", "off"):
        r = set_enabled(arg.lower() == "on")
        state = "on" if r.get("enabled") else "off"
        note = "  [dim](SCILINK_MEMORY env override in effect)[/]" if r.get("env_override") else ""
        shell.console.print(f"[green]✓[/] persistent memory {state}{note}")
        return
    ov = memory_overview()
    p = ov.get("pipeline") or {}
    shell.console.print(f"[bold]Persistent memory[/]: {'on' if ov.get('enabled') else 'off'}  "
                        f"[dim]{ov.get('home')}[/]")
    shell.console.print(f"  script bank: {p.get('bank_total', 0)} ({p.get('bank_proven', 0)} proven)  "
                        f"· review inbox: {p.get('inbox_total', 0)} ({p.get('inbox_ready', 0)} ready)  "
                        f"· skills: {p.get('skills_total', 0)} ({p.get('skills_provisional', 0)} provisional)")
    shell.console.print("  [dim]/memory on|off toggles; `scilink memory` manages the store[/]")


def _files(shell, arg: str) -> None:
    from scilink.server.tree import build_tree
    tree = build_tree(str(shell.session_dir))
    entries = tree.get("entries") or []
    if not entries:
        shell.console.print("[dim](empty session directory)[/]")
        return
    root = Path(arg) if arg else None
    for e in entries:
        if root and not e["name"].startswith(str(root)):
            continue
        _print_entry(shell, e, 0)
    if tree.get("truncated"):
        shell.console.print("[dim]… (truncated)[/]")


def _print_entry(shell, e: dict, depth: int) -> None:
    pad = "  " * depth
    if e.get("is_dir"):
        shell.console.print(f"{pad}[bold]{e['name']}/[/]")
        for c in (e.get("children") or [])[:40]:
            _print_entry(shell, c, depth + 1)
    else:
        new = " [green]new[/]" if e.get("new") else ""
        shell.console.print(f"{pad}{e['name']}  [dim]{_size(e.get('size') or 0)}[/]{new}")


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _verbose(shell, arg: str) -> None:
    shell.renderer.verbose = not shell.renderer.verbose
    shell.console.print(f"{V.NAMES['verbose_section']}: "
                        f"[bold]{'on' if shell.renderer.verbose else 'off'}[/]")


def _cost(shell, arg: str) -> None:
    t = shell.totals
    shell.console.print(
        f"This session: {int(t.get('calls', 0))} LLM calls · "
        f"{int(t.get('prompt_tokens', 0)):,} tokens in · {int(t.get('completion_tokens', 0)):,} out · "
        f"{t.get('seconds', 0.0):.0f}s in model calls")


def _sessions(shell, arg: str) -> None:
    from .sessions import print_sessions
    print_sessions(shell.console, Path.cwd(), shell.mode)


def _resume(shell, arg: str) -> None:
    from .sessions import pick_session
    target = arg or pick_session(shell.console, shell.prompt_session, Path.cwd(), shell.mode)
    if not target:
        return
    shell.resume(target)


def _checkpoint(shell, arg: str) -> None:
    from .turn import save_checkpoint_quietly
    path = save_checkpoint_quietly(shell.agent)
    shell.console.print(f"[green]✓[/] checkpoint saved" + (f": {path}" if path else ""))


def _clear(shell, arg: str) -> None:
    shell.console.clear()


def _quit(shell, arg: str) -> None:
    shell.request_quit()


def core_commands() -> List[Command]:
    return [
        Command("/help", "Show this help", _help),
        Command("/status", "Session state", _status, aliases=("/state",)),
        Command("/mode", "Show or set the autonomy level", _mode, aliases=("/autonomy",),
                arg_hint="[level]"),
        Command("/tools", "List built-in, custom and MCP tools", _tools),
        Command("/mcp", "Connect an MCP server", _mcp, arg_hint="<config>", completes_paths=True),
        Command("/skill", "Register a custom skill (.md)", _skill, arg_hint="<path>",
                completes_paths=True),
        Command("/tool", "Register a custom tool file (.py)", _tool, arg_hint="<path>",
                completes_paths=True),
        Command("/skills", "Skill catalog (built-in and custom)", _skills),
        Command("/memory", "Persistent memory overview; on/off toggles", _memory,
                arg_hint="[on|off]"),
        Command("/files", "Files in the session directory", _files, arg_hint="[subdir]"),
        Command("/verbose", "Toggle verbose narration (Ctrl+O, also mid-turn)", _verbose),
        Command("/cost", "LLM calls and tokens this session", _cost),
        Command("/sessions", "Past sessions here that can be resumed", _sessions),
        Command("/resume", "Resume a past session", _resume, arg_hint="[id]"),
        Command("/checkpoint", "Save the session state now", _checkpoint),
        Command("/clear", "Clear the screen", _clear),
        Command("/quit", "Exit (saves a checkpoint)", _quit, aliases=("/exit", "/q")),
    ]
