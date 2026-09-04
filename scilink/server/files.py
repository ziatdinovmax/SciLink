"""Upload saving (Streamlit conventions) + traversal-guarded file serving.

Destinations mirror the Streamlit helpers exactly so the agents find files
where they expect them:
  analyze data/metadata, one file  -> <session>/uploads/<name>       (sidebar.py:1302)
  analyze data/metadata, several   -> <session>/uploads/series/<name> (sidebar.py:1333,1365)
  plan knowledge|code|data         -> <session>/<category>/<name>    (chat_uploads.py:388)
  meta (combined dropzone)         -> <session>/uploads/<name>       (chat_uploads.py:411)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from scilink.ui.config import (
    SUPPORTED_CODE_EXTENSIONS,
    SUPPORTED_DATA_EXTENSIONS,
    SUPPORTED_KNOWLEDGE_EXTENSIONS,
    SUPPORTED_META_EXTENSIONS,
    SUPPORTED_METADATA_EXTENSIONS,
    SUPPORTED_PLANNING_DATA_EXTENSIONS,
)

_GLOBAL_META_NAMES = {"metadata.json", "meta.json", "info.json", "experiment.json"}

_CATEGORY_EXTENSIONS: Dict[str, tuple] = {
    "data": SUPPORTED_DATA_EXTENSIONS,
    "metadata": SUPPORTED_METADATA_EXTENSIONS,
    "knowledge": SUPPORTED_KNOWLEDGE_EXTENSIONS,
    "code": SUPPORTED_CODE_EXTENSIONS,
    "planning_data": SUPPORTED_PLANNING_DATA_EXTENSIONS,
    "meta": SUPPORTED_META_EXTENSIONS,
}


class UploadError(Exception):
    pass


def _check_name(name: str, category: str) -> str:
    base = Path(name).name  # strip any client-supplied directory parts
    if not base or base.startswith("."):
        raise UploadError(f"Invalid filename: {name!r}")
    allowed = _CATEGORY_EXTENSIONS.get(category)
    if allowed is None:
        raise UploadError(f"Unknown upload category: {category!r}")
    if Path(base).suffix.lower() not in allowed:
        raise UploadError(
            f"Extension {Path(base).suffix!r} not accepted for {category!r}")
    return base


def save_uploads(session_dir: str, category: str,
                 files: List[Tuple[str, bytes]]) -> Dict[str, object]:
    """Save uploaded (name, bytes) pairs; returns saved paths + routing info.

    The returned ``paths`` are ABSOLUTE (the dispatch prompts quote absolute
    paths, matching Streamlit); ``series`` flags the multi-file analyze case;
    ``global_metadata`` is the recognized global metadata file, if any.
    """
    root = Path(session_dir)
    names = [_check_name(n, category) for n, _ in files]

    if category in ("data", "metadata"):
        if len(files) > 1:
            target = root / "uploads" / "series"
        else:
            target = root / "uploads"
    elif category == "meta":
        target = root / "uploads"
    else:  # knowledge | code | planning_data
        sub = "data" if category == "planning_data" else category
        target = root / sub
    target.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    global_meta = None
    for base, (_, blob) in zip(names, files):
        dest = target / base
        dest.write_bytes(blob)
        saved.append(str(dest))
        if category == "metadata" and base.lower() in _GLOBAL_META_NAMES:
            global_meta = str(dest)

    return {
        "paths": saved,
        "series_dir": str(target) if target.name == "series" else None,
        "global_metadata": global_meta,
        "category": category,
    }


def resolve_safe(session_dir: str, rel_path: str) -> Path:
    """Resolve ``rel_path`` inside the session dir or raise PermissionError."""
    base = Path(session_dir).resolve()
    candidate = (base / rel_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise PermissionError(f"Path escapes the session directory: {rel_path}")
    return candidate
