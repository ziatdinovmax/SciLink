"""Finding past sessions.

Sessions stay where they are created (project-local: the outputs live next
to the data), but every chat surface registers each one in a central
index — ``<SCILINK_HOME>/sessions.jsonl`` — so a session can be listed and
resumed from anywhere: the terminal's ``--resume`` picker, the web UI's
"Resume past session", the exit hint. This module is the index plus the
older folder scan (``discover_resumable``), and imports nothing from the
server package so the shell can use it without FastAPI.

``SCILINK_SESSION_ROOT`` relocates NEW sessions of the terminal shell to
one folder for those who prefer a central store; the index does not care
where directories are.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional

from scilink.skills.loader import scilink_home
from scilink.ui.session_meta import load_session_name, session_label
from scilink.ui.vocabulary import session_prefixes

_lock = threading.Lock()


# ── the index ────────────────────────────────────────────────────

def index_path() -> Path:
    return scilink_home() / "sessions.jsonl"


def _read_index() -> Dict[str, Dict[str, Any]]:
    """path -> record (the last line for a path wins)."""
    records: Dict[str, Dict[str, Any]] = {}
    try:
        with open(index_path(), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("path"):
                    records[rec["path"]] = rec
    except OSError:
        pass
    return records


def _write_index(records: Dict[str, Dict[str, Any]]) -> None:
    path = index_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for rec in records.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except OSError:
        pass   # the index is a convenience; never break a session over it


def register_session(session_dir, mode: str, *, launcher_cwd=None) -> Dict[str, Any]:
    """Record (or refresh) a session in the index. Idempotent per path."""
    sd = Path(session_dir).resolve()
    now = time.time()
    with _lock:
        records = _read_index()
        rec = records.get(str(sd)) or {"created": now}
        rec.update({
            "id": sd.name,
            "path": str(sd),
            "mode": mode,
            "cwd": str(Path(launcher_cwd or Path.cwd()).resolve()),
            "updated": now,
        })
        name = load_session_name(sd)
        if name:
            rec["name"] = name
        records[str(sd)] = rec
        _write_index(records)
    return rec


def touch_session(session_dir) -> None:
    """Refresh a session's ``updated`` time and display name (no-op if the
    session was never registered)."""
    sd = str(Path(session_dir).resolve())
    with _lock:
        records = _read_index()
        rec = records.get(sd)
        if rec is None:
            return
        rec["updated"] = time.time()
        name = load_session_name(sd)
        if name:
            rec["name"] = name
        _write_index(records)


def _entry(sd: Path, mode: str, rec: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The listing entry for a session directory, or None if it cannot be
    resumed (no checkpoint, no chat history)."""
    has_checkpoint = (sd / "checkpoint.json").exists()
    has_chat = (sd / "chat_history.json").exists()
    if not has_checkpoint and not has_chat:
        return None
    summary: Dict[str, Any] = {}
    if has_checkpoint:
        try:
            ckpt = json.loads((sd / "checkpoint.json").read_text())
            summary["analysis_count"] = len(ckpt.get("analysis_results", []))
            dp = ckpt.get("current_data_path")
            if dp:
                summary["data_file"] = Path(dp).name
        except Exception:
            pass
    if has_chat and "analysis_count" not in summary:
        try:
            hist = json.loads((sd / "chat_history.json").read_text())
            summary["message_count"] = sum(1 for m in hist if m.get("role") == "user")
        except Exception:
            pass
    prefix = next((p for p in session_prefixes(mode) if sd.name.startswith(p + "_")),
                  session_prefixes(mode)[0])
    return {
        "id": sd.name,
        "path": str(sd),
        "folder": str(sd.parent),
        "label": session_label(sd, prefix),
        "has_checkpoint": has_checkpoint,
        "has_chat_history": has_chat,
        "summary": summary,
        "updated": (rec or {}).get("updated") or sd.stat().st_mtime,
    }


def list_sessions(mode: str, *, root=None,
                  exclude: Collection[str] = ()) -> List[Dict[str, Any]]:
    """Resumable sessions of ``mode``, newest first: everything the index
    knows (any folder) plus a scan of ``root`` for sessions that predate the
    index. Index entries whose directory is gone are dropped from the file.
    ``exclude`` names ids to skip (a web manager's live sessions)."""
    excluded = set(exclude)
    entries: Dict[str, Dict[str, Any]] = {}
    with _lock:
        records = _read_index()
        stale = [p for p in records if not Path(p).is_dir()]
        if stale:
            for p in stale:
                records.pop(p, None)
            _write_index(records)
    for p, rec in records.items():
        if rec.get("mode") != mode:
            continue
        sd = Path(p)
        if sd.name in excluded:
            continue
        e = _entry(sd, mode, rec)
        if e:
            entries[str(sd)] = e
    if root is not None:
        for e in discover_resumable(root, mode, exclude=excluded):
            entries.setdefault(e["path"], e)
    return sorted(entries.values(), key=lambda e: e["updated"], reverse=True)


def resolve_session(ref: str, mode: Optional[str] = None, *, root=None) -> Optional[Path]:
    """The directory for a session reference: an existing path, a name
    under ``root`` (the current folder by default), or an id in the index."""
    p = Path(ref).expanduser()
    if p.is_dir():
        return p.resolve()
    local = Path(root or Path.cwd()) / ref
    if local.is_dir():
        return local.resolve()
    with _lock:
        records = _read_index()
    matches = [Path(r["path"]) for r in records.values()
               if r.get("id") == ref and (mode is None or r.get("mode") == mode)
               and Path(r["path"]).is_dir()]
    if matches:
        return max(matches, key=lambda d: records[str(d)].get("updated", 0)).resolve()
    return None


# ── the folder scan (predates the index) ─────────────────────────

def discover_resumable(root, mode: str,
                       exclude: Collection[str] = ()) -> List[Dict[str, Any]]:
    """Resumable sessions of ``mode`` under ``root``, newest first.

    A session directory is listed when its name starts with one of the
    mode's session prefixes (the canonical one or a legacy CLI spelling —
    ``campaign_session`` for plan, ``simulate_session`` for simulate) and it
    contains ``checkpoint.json`` or ``chat_history.json``. ``exclude`` names
    directories to skip (the web manager's live sessions).
    """
    root = Path(root)
    excluded = set(exclude)
    found: List[Path] = []
    for prefix in session_prefixes(mode):
        found.extend(p for p in root.glob(f"{prefix}_*") if p.is_dir())
    result = []
    for s in sorted(set(found), key=lambda p: _timestamp_key(p.name), reverse=True):
        if s.name in excluded:
            continue
        e = _entry(s.resolve(), mode)
        if e:
            result.append(e)
    return result


def _timestamp_key(name: str) -> str:
    """Sort key: the ``YYYYmmdd_HHMMSS`` tail, so a legacy-prefixed session
    and a canonical one interleave by time rather than by spelling."""
    parts = name.rsplit("_", 2)
    return "_".join(parts[-2:]) if len(parts) == 3 else name
