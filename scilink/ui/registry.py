"""Process-global registry of live UI sessions — survives browser disconnects.

Streamlit ties ``st.session_state`` to the browser's WebSocket session: when
the viewer's connection drops (laptop offline, lid closed), the server keeps
running the agent thread, but the reconnected page gets a FRESH session_state
with no handle to it — the run looks lost even though it is alive (observed
with the server under ``screen`` on another machine).

This module keeps the handles at PROCESS scope. Modules are imported once per
Streamlit server process, so ``_SESSIONS`` outlives any browser session. The
app syncs its state here on every run while attached; the welcome screen
offers to REATTACH when it finds an entry whose work is still running or
whose finished result was never shown.

Entries hold live references (agent, ChatTask, the chat_messages list), so a
reattached page continues the same objects the background thread mutates —
including a pending human-feedback request. Nothing is persisted; a server
restart is the (separate) checkpoint-resume path.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

# The session_state keys that constitute an attachable session. chat_messages
# is shared BY REFERENCE so appends after reattach mutate the same list.
SESSION_KEYS = ("agent", "agent_config", "chat_messages", "chat_task",
                "known_images", "app_mode")

_SESSIONS: Dict[str, Dict[str, Any]] = {}


def sync_from(state: Mapping[str, Any]) -> None:
    """Mirror the attached session's handles into the registry.

    Called once per script run while a session is attached — reruns are
    frequent, so this also keeps reassigned objects (a fresh ChatTask each
    turn) current. Cheap: reference copies only.
    """
    sdir = state.get("session_dir")
    if not sdir:
        return
    entry = _SESSIONS.setdefault(str(sdir), {})
    for k in SESSION_KEYS:
        entry[k] = state.get(k)
    entry["updated_at"] = time.time()


def attach_to(state, session_dir: str) -> bool:
    """Copy a registry entry's handles into ``state`` (a new browser session).

    Returns False when the entry is gone (e.g. the server restarted)."""
    entry = _SESSIONS.get(str(session_dir))
    if entry is None:
        return False
    for k in SESSION_KEYS:
        state[k] = entry.get(k)
    state["session_dir"] = str(session_dir)
    state["agent_initialized"] = True
    return True


def unregister(session_dir: Optional[str]) -> None:
    if session_dir:
        _SESSIONS.pop(str(session_dir), None)


def status_of(entry: Mapping[str, Any]) -> str:
    """'awaiting input' | 'running' | 'finished' (result never shown) | 'idle'."""
    task = entry.get("chat_task")
    if task is not None:
        if getattr(task, "is_running", False):
            if getattr(task, "feedback_request", None) is not None:
                return "awaiting input"
            return "running"
        if (getattr(task, "result", None) is not None
                or getattr(task, "error", None) is not None):
            return "finished"
    return "idle"


def live_entries() -> List[Dict[str, Any]]:
    """Entries worth offering a reattach for, most recent first.

    'idle' entries are included too: after a completed turn the task is
    consumed and replaced, but the session (agent + history) is still only
    alive in this process — reattaching beats resuming from checkpoint."""
    out = []
    for sdir, entry in _SESSIONS.items():
        out.append({"session_dir": sdir, "status": status_of(entry),
                    "updated_at": entry.get("updated_at", 0.0),
                    "config": entry.get("agent_config") or {},
                    "app_mode": entry.get("app_mode"),
                    "n_messages": len(entry.get("chat_messages") or [])})
    out.sort(key=lambda e: e["updated_at"], reverse=True)
    return out
