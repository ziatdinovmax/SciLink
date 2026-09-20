"""Finding past sessions on disk.

Every chat surface — the web backend, the Streamlit UI and the terminal
shell — lists resumable sessions the same way: the directories under a
root whose name carries a mode's session prefix and that hold a checkpoint
or a chat history. This is that one listing (moved out of the web
``SessionManager``, which now delegates to it); it imports nothing from
the server package so the shell can use it without FastAPI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection, Dict, List

from scilink.ui.session_meta import session_label
from scilink.ui.vocabulary import session_prefixes


def discover_resumable(root, mode: str,
                       exclude: Collection[str] = ()) -> List[Dict[str, Any]]:
    """Resumable sessions of ``mode`` under ``root``, newest first.

    A session directory is listed when its name starts with one of the
    mode's session prefixes (the canonical one or a legacy CLI spelling —
    ``campaign_session`` for plan, ``simulate_session`` for simulate) and it
    contains ``checkpoint.json`` or ``chat_history.json``. ``exclude`` names
    directories to skip (the web manager's live sessions).

    Each entry: ``{id, label, has_checkpoint, has_chat_history, summary}``
    where ``summary`` carries ``analysis_count`` / ``data_file`` from the
    checkpoint or ``message_count`` from the chat history.
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
        has_checkpoint = (s / "checkpoint.json").exists()
        has_chat = (s / "chat_history.json").exists()
        if not has_checkpoint and not has_chat:
            continue
        summary: Dict[str, Any] = {}
        if has_checkpoint:
            try:
                ckpt = json.loads((s / "checkpoint.json").read_text())
                summary["analysis_count"] = len(ckpt.get("analysis_results", []))
                dp = ckpt.get("current_data_path")
                if dp:
                    summary["data_file"] = Path(dp).name
            except Exception:
                pass
        if has_chat and "analysis_count" not in summary:
            try:
                hist = json.loads((s / "chat_history.json").read_text())
                summary["message_count"] = sum(
                    1 for m in hist if m.get("role") == "user")
            except Exception:
                pass
        prefix = next((p for p in session_prefixes(mode)
                       if s.name.startswith(p + "_")), session_prefixes(mode)[0])
        result.append({
            "id": s.name,
            "label": session_label(s, prefix),
            "has_checkpoint": has_checkpoint,
            "has_chat_history": has_chat,
            "summary": summary,
        })
    return result


def _timestamp_key(name: str) -> str:
    """Sort key: the ``YYYYmmdd_HHMMSS`` tail, so a legacy-prefixed session
    and a canonical one interleave by time rather than by spelling."""
    parts = name.rsplit("_", 2)
    return "_".join(parts[-2:]) if len(parts) == 3 else name
