"""Upload saving (Streamlit conventions) + traversal-guarded file serving.

Destinations mirror the Streamlit helpers exactly so the agents find files
where they expect them:
  analyze data/metadata, one file  -> <session>/uploads/<name>       (sidebar.py:1302)
  analyze data/metadata, several   -> <session>/uploads/series/<name> (sidebar.py:1333,1365)
  plan knowledge|code|data         -> <session>/<category>/<name>    (chat_uploads.py:388)
  meta (combined dropzone)         -> <session>/uploads/<name>       (chat_uploads.py:411)
  analyze scripts (.py etc.)       -> <session>/scripts/<name>       (attach-a-script)

Folder uploads (``preserve_paths=True``) keep the client's relative layout
under the same category root — ``uploads/<folder>/<sub>/<name>`` — so a
nested drop arrives as the tree the user has on disk. Files whose extension
the category does not accept are SKIPPED (reported, not fatal): a real folder
holds READMEs, vendor sidecars and the like alongside the data, and rejecting
the whole drop on the first stray file would make folder upload useless.
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
    # A script the user attaches in chat to be ADAPTED by the analysis
    # agents (run_analysis reference_scripts) — analyze mode's home for
    # code, which its data / metadata categories rightly reject.
    "scripts": SUPPORTED_CODE_EXTENSIONS,
}


# A folder upload can hold this many files at most (browser-side cap matches).
MAX_FOLDER_FILES = 2000

# Sidecar JSONs live beside their data in a series folder, so an analyze-mode
# folder upload accepts the metadata extensions too (a flat data upload keeps
# its strict list — the metadata dropzone is a separate surface there).
_FOLDER_EXTRA_EXTENSIONS: Dict[str, tuple] = {
    "data": SUPPORTED_METADATA_EXTENSIONS,
}


class UploadError(Exception):
    pass


def _check_relpath(rel: str, category: str) -> Path:
    """Validate a client-supplied relative path for a folder upload.

    Returns the sanitized relative ``Path``; raises ``UploadError`` for a
    path that could escape the category root (absolute, ``..``), and for an
    extension the category does not accept (callers turn that into a skip).
    """
    raw = rel.replace("\\", "/").strip("/")
    parts = [c for c in raw.split("/") if c not in ("", ".")]
    if not parts or rel.startswith("/") or ".." in parts:
        raise UploadError(f"Invalid path in folder upload: {rel!r}")
    if any(c.startswith(".") for c in parts):
        raise UploadError(f"Hidden entry: {rel!r}")
    allowed = _CATEGORY_EXTENSIONS.get(category)
    if allowed is None:
        raise UploadError(f"Unknown upload category: {category!r}")
    allowed = tuple(allowed) + _FOLDER_EXTRA_EXTENSIONS.get(category, ())
    if Path(parts[-1]).suffix.lower() not in allowed:
        raise UploadError(
            f"Extension {Path(parts[-1]).suffix!r} not accepted for {category!r}")
    return Path(*parts)


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


def _category_root(root: Path, category: str) -> Path:
    if category in ("data", "metadata", "meta"):
        return root / "uploads"
    sub = "data" if category == "planning_data" else category
    return root / sub          # knowledge | code | scripts


def save_folder_upload(session_dir: str, category: str,
                       files: List[Tuple[str, bytes]]) -> Dict[str, object]:
    """Save a folder upload — ``(relative_path, bytes)`` pairs sharing one
    top-level folder — preserving the layout under the category root.

    Returns the same keys as ``save_uploads`` plus:
      root      absolute path of the saved top-level folder
      dirs      [{"path": <abs dir>, "n_files": n}] for the root and every
                subfolder that received a file (top-down), so the dispatch
                prompt can describe the layout and the agents get concrete
                subfolder paths to examine
      skipped   [{"path": rel, "reason": ...}] entries not saved
    ``series_dir`` is the root when the folder is FLAT (every file at the
    top level, more than one) — the analyze agent treats such a directory as
    a series exactly as it treats ``uploads/series/``; a nested folder has
    no single series dir and the prompt lists its subfolders instead.
    """
    if len(files) > MAX_FOLDER_FILES:
        raise UploadError(
            f"Folder upload has {len(files)} files; the limit is "
            f"{MAX_FOLDER_FILES}.")
    if category not in _CATEGORY_EXTENSIONS:
        raise UploadError(f"Unknown upload category: {category!r}")
    accepted: List[Tuple[Path, bytes]] = []
    skipped: List[Dict[str, str]] = []
    for rel, blob in files:
        try:
            accepted.append((_check_relpath(rel, category), blob))
        except UploadError as exc:
            # A traversal attempt is fatal; a stray extension / hidden entry
            # is skipped so the rest of the folder still lands.
            if "Invalid path" in str(exc):
                raise
            skipped.append({"path": rel, "reason": str(exc)})
    if not accepted:
        raise UploadError(
            "No accepted files in the folder"
            + (f" ({skipped[0]['reason']})" if skipped else "") + ".")
    tops = {p.parts[0] for p, _ in accepted}
    if len(tops) != 1:
        raise UploadError("A folder upload must contain a single top-level "
                          f"folder (got {sorted(tops)}).")
    if any(len(p.parts) < 2 for p, _ in accepted):
        raise UploadError("Folder upload entries must be <folder>/<file>.")

    base = _category_root(Path(session_dir), category)
    folder_root = base / next(iter(tops))
    saved: List[str] = []
    global_meta = None
    per_dir: Dict[Path, int] = {}
    for relp, blob in accepted:
        dest = base / relp
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        saved.append(str(dest))
        per_dir[dest.parent] = per_dir.get(dest.parent, 0) + 1
        if category == "metadata" and dest.name.lower() in _GLOBAL_META_NAMES:
            global_meta = str(dest)
    per_dir.setdefault(folder_root, 0)
    dirs = [{"path": str(d), "n_files": n}
            for d, n in sorted(per_dir.items(), key=lambda kv: str(kv[0]))]
    flat = len(per_dir) == 1 and len(saved) > 1
    return {
        "paths": saved,
        "series_dir": str(folder_root) if flat else None,
        "global_metadata": global_meta,
        "category": category,
        "root": str(folder_root),
        "dirs": dirs,
        "skipped": skipped,
    }


def save_uploads(session_dir: str, category: str,
                 files: List[Tuple[str, bytes]],
                 preserve_paths: bool = False) -> Dict[str, object]:
    """Save uploaded (name, bytes) pairs; returns saved paths + routing info.

    The returned ``paths`` are ABSOLUTE (the dispatch prompts quote absolute
    paths, matching Streamlit); ``series`` flags the multi-file analyze case;
    ``global_metadata`` is the recognized global metadata file, if any.
    With ``preserve_paths`` the names are relative paths of a folder upload —
    see ``save_folder_upload``.
    """
    if preserve_paths:
        return save_folder_upload(session_dir, category, files)
    root = Path(session_dir)
    names = [_check_name(n, category) for n, _ in files]

    if category in ("data", "metadata"):
        if len(files) > 1:
            target = root / "uploads" / "series"
        else:
            target = root / "uploads"
    elif category == "meta":
        target = root / "uploads"
    else:  # knowledge | code | planning_data | scripts
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
