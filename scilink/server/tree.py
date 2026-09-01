"""Session file-tree listing, provenance index, and zip export.

The tree is the File Explorer's data source (the React twin of the
recursive expander tree at scilink/ui/app.py:1489-1639), with two things
Streamlit could not offer: per-entry ``new`` flags (modified since the
current/last turn started, for badges) and a provenance index derived from
the session event log (``events.jsonl``, scilink/session_events.py) mapping
files to the tool call that produced them.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keep listings bounded: sessions can accumulate thousands of intermediate
# files; the explorer paginates by directory expansion anyway.
_MAX_ENTRIES = 4000
_MAX_DEPTH = 12

_SKIP_DIRS = {"__pycache__", ".git", "node_modules"}


def build_tree(session_dir: str, new_since: Optional[float] = None
               ) -> Dict[str, Any]:
    """Recursive listing of the session dir.

    Entries: ``{name, path (relative, POSIX), is_dir, size, mtime, new,
    children}`` sorted directories-first then by name. ``new`` marks files
    modified at or after ``new_since`` (the running/last turn's start).
    """
    root = Path(session_dir)
    count = 0

    def walk(d: Path, depth: int) -> List[Dict[str, Any]]:
        nonlocal count
        if depth > _MAX_DEPTH or count > _MAX_ENTRIES:
            return []
        try:
            children = sorted(d.iterdir(),
                              key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return []
        out: List[Dict[str, Any]] = []
        for p in children:
            if p.name.startswith(".") or p.name in _SKIP_DIRS:
                continue
            if count > _MAX_ENTRIES:
                break
            count += 1
            try:
                stat = p.stat()
            except OSError:
                continue
            entry: Dict[str, Any] = {
                "name": p.name,
                "path": p.relative_to(root).as_posix(),
                "is_dir": p.is_dir(),
                "size": 0 if p.is_dir() else stat.st_size,
                "mtime": stat.st_mtime,
                "new": bool(new_since and not p.is_dir()
                            and stat.st_mtime >= new_since),
            }
            if p.is_dir():
                entry["children"] = walk(p, depth + 1)
            out.append(entry)
        return out

    return {"entries": walk(root, 0), "truncated": count > _MAX_ENTRIES}


def load_provenance(session_dir: str) -> List[Dict[str, Any]]:
    """Tool-call timeline from every ``events.jsonl`` under the session
    (meta children keep their own logs in their subdirectories).

    Returns events newest-first: ``{n, ts, tool, status, summary, files,
    log}`` where ``files`` are session-relative POSIX paths (event logs
    record absolute paths; entries outside the session dir are kept as
    recorded) and ``log`` is the log file's directory relative to the
    session root ("" for the session's own log).
    """
    root = Path(session_dir).resolve()
    marker = root.name + "/"

    def to_rel(f: Any) -> str:
        s = str(f)
        try:
            return Path(s).resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            pass
        # The event log clips long paths ("…runcated/prefix/…"); recover the
        # session-relative part by finding the session dir name in the string.
        idx = s.find(marker)
        if idx >= 0:
            return s[idx + len(marker):]
        return s

    events: List[Dict[str, Any]] = []
    for log_path in sorted(root.rglob("events.jsonl")):
        try:
            rel_log = log_path.parent.relative_to(root).as_posix()
        except ValueError:
            rel_log = ""
        if rel_log == ".":
            rel_log = ""
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            files = [to_rel(f) for f in rec.get("files") or []]
            events.append({
                "n": rec.get("n"),
                "ts": rec.get("ts"),
                "tool": rec.get("tool"),
                "status": rec.get("status"),
                "summary": rec.get("summary", ""),
                "files": files,
                "log": rel_log,
            })
    events.sort(key=lambda e: (e.get("ts") or "", e.get("n") or 0),
                reverse=True)
    return events


_ZIP_MAX_BYTES = 500 * 1024 * 1024


def zip_directory(session_dir: str, rel_path: str) -> bytes:
    """Zip a session subdirectory (or the whole session for ``""``)."""
    root = Path(session_dir).resolve()
    target = (root / rel_path).resolve() if rel_path else root
    if target != root and root not in target.parents:
        raise PermissionError(f"Path escapes the session directory: {rel_path}")
    if not target.is_dir():
        raise FileNotFoundError(rel_path)
    buf = io.BytesIO()
    total = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(target.rglob("*")):
            if p.is_dir() or p.name.startswith("."):
                continue
            if _SKIP_DIRS & set(p.parts):
                continue
            total += p.stat().st_size
            if total > _ZIP_MAX_BYTES:
                raise ValueError("Directory too large to zip (>500 MB).")
            zf.write(p, p.relative_to(target).as_posix())
    return buf.getvalue()
