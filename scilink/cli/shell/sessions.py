"""The resume picker — the terminal twin of the web's "Resume past session".

Both list the same directories through ``scilink.sessions.discover_resumable``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from rich.table import Table

from scilink.sessions import discover_resumable
from scilink.ui import vocabulary as V


def _summary(s: dict) -> str:
    sm = s.get("summary") or {}
    bits = []
    if "analysis_count" in sm:
        bits.append(f"{sm['analysis_count']} analyses")
    if sm.get("data_file"):
        bits.append(sm["data_file"])
    if "message_count" in sm:
        bits.append(f"{sm['message_count']} messages")
    if not s.get("has_checkpoint"):
        bits.append("no checkpoint")
    return " · ".join(bits)


def print_sessions(console, root: Path, mode: str) -> List[dict]:
    sessions = discover_resumable(root, mode)
    if not sessions:
        console.print(f"[dim]No {V.mode(mode)['name']} sessions to resume under {root}[/]")
        return []
    t = Table(box=None, padding=(0, 2), show_header=True, header_style="dim")
    t.add_column("#", justify="right", style="bold")
    t.add_column("session")
    t.add_column("id", style="dim")
    t.add_column("", style="dim")
    for i, s in enumerate(sessions, 1):
        t.add_row(str(i), s["label"], s["id"], _summary(s))
    console.print(t)
    return sessions


def pick_session(console, prompt_session, root: Path, mode: str) -> Optional[str]:
    """Show the table and read a number; Enter picks the newest, empty
    input on an empty list returns None."""
    sessions = print_sessions(console, root, mode)
    if not sessions:
        return None
    try:
        ans = prompt_session.prompt(f"{V.NAMES['resume_session']} # (Enter = 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not ans:
        return sessions[0]["id"]
    if ans.isdigit() and 1 <= int(ans) <= len(sessions):
        return sessions[int(ans) - 1]["id"]
    for s in sessions:
        if s["id"] == ans:
            return s["id"]
    console.print(f"[red]No session {ans!r}[/]")
    return None
