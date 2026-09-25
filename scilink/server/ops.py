"""What a control plane needs from one SciLink server: is it up, is it busy,
stop taking work, and which workspace is this.

One server hosts one workspace (a campaign's sessions, memory and data), so
these answers are per process. ``busy`` is computed, never cached: a turn
running in any session, a live run that is arming, running, paused or still
finishing, or a memory job still going. ``idle_for_s`` counts from the last
request that started work or the last status poll that found some, so a
control plane that polls stops a server only after a quiet stretch it
observed itself, never in the middle of a turn it did not see start.

The workspace manifest is ``workspace.json``: ``SCILINK_WORKSPACE`` names the
file, else it is looked for in the session root and then one level up (the
proposal's layout puts it beside ``sessions/``). Its content is the
deployment's; only ``id`` and ``name`` are read here.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_workspace(session_root: Path) -> Optional[Dict[str, Any]]:
    named = os.environ.get("SCILINK_WORKSPACE")
    candidates = ([Path(named).expanduser()] if named else
                  [Path(session_root) / "workspace.json",
                   Path(session_root).parent / "workspace.json"])
    for path in candidates:
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("_path", str(path))
                    return data
        except (OSError, ValueError):
            continue
    return None


class Workload:
    """The server's activity and drain state; one per app."""

    LIVE_BUSY_STATES = ("arming", "running", "finishing", "paused")

    def __init__(self, session_root: Path) -> None:
        self.session_root = Path(session_root)
        self.workspace = load_workspace(self.session_root)
        self.started_at = time.time()
        self.draining = False
        self._last_activity = time.time()
        self._lock = threading.Lock()

    # -- activity ---------------------------------------------------------
    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.time()

    def busy_reasons(self, managers: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        for mgr in list(managers.values()):
            for s in list(getattr(mgr, "_sessions", {}).values()):
                turn = getattr(s, "turn", None)
                if turn is not None and getattr(turn, "is_running", False):
                    reasons.append(f"turn:{s.id}")
        try:
            from . import live_api
            for sid, run in list(live_api._RUNS.items()):
                if getattr(run, "state", None) in self.LIVE_BUSY_STATES:
                    reasons.append(f"live:{sid}")
        except Exception:  # noqa: BLE001 - the live layer is optional
            pass
        try:
            from . import memory_api
            with memory_api._jobs_lock:
                jobs = list(memory_api._jobs.values())
            reasons.extend(f"memory_job:{j.get('id')}" for j in jobs
                           if j.get("status") == "running")
        except Exception:  # noqa: BLE001
            pass
        if reasons:
            self.touch()
        return reasons

    # -- answers ----------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        from scilink import __version__
        out = {"ok": True, "version": __version__, "uptime_s": round(time.time() - self.started_at)}
        if self.workspace:
            out["workspace"] = self.workspace.get("id") or self.workspace.get("name")
        return out

    def status(self, managers: Dict[str, Any]) -> Dict[str, Any]:
        busy = self.busy_reasons(managers)
        with self._lock:
            idle_for = 0.0 if busy else time.time() - self._last_activity
        live = sum(len(getattr(m, "_sessions", {})) for m in managers.values())
        out = self.health()
        out.update({
            "state": "draining" if self.draining else ("busy" if busy else "idle"),
            "draining": self.draining,
            "busy": busy,
            "idle_for_s": round(idle_for),
            "sessions_live": live,
        })
        return out

    def refuse_if_draining(self) -> None:
        if self.draining:
            from fastapi import HTTPException
            raise HTTPException(503, "This server is draining: it finishes the work "
                                     "it has and takes no new turns. Try again later.")
