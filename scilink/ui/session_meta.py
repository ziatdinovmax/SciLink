"""Session display names: sidecar label + LLM auto-titling.

The directory name (``<prefix>_<timestamp>``) is load-bearing — deliverables
manifests and checkpoints store absolute paths into it — so a session's
human name lives in a sidecar ``session_meta.json`` and is display-only.
The agent titles an unnamed session after its first completed turn; a
user-set name always wins and is never overwritten by the agent.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

_META_FILE = "session_meta.json"


def _meta_path(session_dir) -> Path:
    return Path(session_dir) / _META_FILE


def load_session_meta(session_dir) -> dict:
    try:
        return json.loads(_meta_path(session_dir).read_text())
    except Exception:
        return {}


def load_session_name(session_dir) -> Optional[str]:
    name = load_session_meta(session_dir).get("name")
    return name if isinstance(name, str) and name.strip() else None


def save_session_name(session_dir, name: str, named_by: str) -> bool:
    """Persist a display name. ``named_by`` is "user" or "agent"; an
    agent title never overwrites a user-chosen name. Returns True if
    written."""
    name = (name or "").strip()
    if not name:
        return False
    meta = load_session_meta(session_dir)
    if named_by == "agent" and meta.get("named_by") == "user":
        return False
    meta.update({"name": name[:80], "named_by": named_by})
    try:
        _meta_path(session_dir).write_text(json.dumps(meta, indent=1))
        return True
    except OSError:
        return False


def timestamp_label(dir_name: str, prefix: str) -> str:
    """'<prefix>_20260730_135341' -> '2026-07-30 13:53:41' (best effort)."""
    parts = dir_name.removeprefix(f"{prefix}_")
    try:
        return (f"{parts[:4]}-{parts[4:6]}-{parts[6:8]} "
                f"{parts[9:11]}:{parts[11:13]}:{parts[13:15]}")
    except (IndexError, ValueError):
        return dir_name


def session_label(session_dir, prefix: str) -> str:
    """Display label for pickers: 'Name — 2026-07-30 13:53:41', or the
    bare timestamp when the session is unnamed."""
    ts = timestamp_label(Path(session_dir).name, prefix)
    name = load_session_name(session_dir)
    return f"{name} — {ts}" if name else ts


def generate_session_title(model, first_user_msg: str,
                           first_reply: str = "") -> Optional[str]:
    """One small LLM call -> a concise session title, or None on any
    failure (callers fall back to the timestamp label silently).

    Takes the agent's already-constructed model object — the one whose
    transport is set up for wherever this deployment actually talks.
    Calling litellm directly instead would re-derive the credentials and
    get the internal proxy wrong: with a base_url set and a project-suffixed
    model, litellm reads a claude-* name as the Anthropic provider and never
    speaks the proxy's schema, so titling silently no-ops on the deployment
    most sessions run on.
    """
    if not (model and first_user_msg):
        return None
    try:
        prompt = (
            "Name this scientific-assistant session in 3-6 plain words "
            "(no quotes, no trailing period). Base it on the topic, not "
            "the phrasing.\n\n"
            f"First request: {first_user_msg[:600]}\n"
            f"Start of reply: {first_reply[:300]}"
        )
        resp = model.generate_content(
            [prompt], generation_config=SimpleNamespace(max_output_tokens=24))
        title = re.sub(r'["\'\n\r.]+', " ", getattr(resp, "text", "") or "")
        title = " ".join(title.split()).strip()
        if len(title) > 60:
            # Cut on a word boundary — a sidebar label ending mid-word
            # ("...from amorphous precurs") reads as a broken title.
            title = title[:60].rsplit(" ", 1)[0].rstrip(" ,;:-") or title[:60]
        return title or None
    except Exception:
        return None
