"""Surgical in-place file edits — the shared core of the `edit_file` tool.

One policy, many orchestrators: exact-snippet replacement with a uniqueness
check, a size cap, a sandbox root check, and counter-suffixed pre-edit
backups. The guards here are load-bearing (the cap is what keeps the tool
from becoming a piecewise document rewriter that dodges the whole-document
revision guards), so every mode must share ONE implementation rather than
drifting copies. Mode-specific concerns stay in the orchestrator tool
wrappers: relative-path resolution, deliverable recording, transcript
prints, and the routing text in error hints (each mode names ITS OWN
alternative tool for oversized edits).

Consumers today: the planning orchestrator's `edit_file`. Analysis and
simulation orchestrators are the intended follow-ups — note that VASP
inputs (POSCAR / INCAR / KPOINTS) are extensionless, which is why the
suffix policy is a parameter: include "" in `allowed_suffixes` to permit
extensionless files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

# Text formats editable by default. A mode may pass its own set; include ""
# to allow extensionless files (VASP inputs).
DEFAULT_EDITABLE_SUFFIXES: Set[str] = {
    ".md", ".html", ".mmd", ".txt", ".json", ".py", ".csv", ".yaml", ".yml",
}

# Surgical means SMALL — see module docstring.
DEFAULT_MAX_SNIPPET_LEN = 2000

DEFAULT_TOO_LARGE_MESSAGE = (
    "Edit too large for edit_file — split the change at a unique boundary "
    "into consecutive smaller edit_file calls."
)


def apply_surgical_edit(
    path: Path,
    old_text: str,
    new_text: str,
    *,
    root: Path,
    backup_dir: Path,
    replace_all: bool = False,
    max_len: int = DEFAULT_MAX_SNIPPET_LEN,
    allowed_suffixes: Set[str] = DEFAULT_EDITABLE_SUFFIXES,
    too_large_message: str = DEFAULT_TOO_LARGE_MESSAGE,
    root_label: str = "session directory",
) -> Dict[str, Any]:
    """Replace one exact snippet in `path`, guarded and backed up.

    `path` must already be resolved (absolute); relative-name resolution
    against a delegation/output directory is the caller's job. Returns a
    dict with ``status: "success"`` (plus ``path``, ``replacements``,
    ``previous_version``) or ``status: "error"`` (plus ``message``). Never
    raises for expected failures; unexpected ones propagate to the caller's
    error handling.

    Guards, in order: sandbox (inside `root`), existence, suffix allowlist,
    non-empty and non-identical snippets, size cap (`too_large_message` is
    the mode's routing hint — it should name that mode's alternative tool),
    exactly-one occurrence unless `replace_all`.

    The pre-edit copy lands in `backup_dir` as
    ``<stem>.before_edit[N]<suffix>`` — counter-suffixed, never clobbering
    an earlier backup. Backup failure is logged and never blocks the edit.
    """
    rp = Path(path)
    root = Path(root).resolve()
    if root not in rp.parents:
        return {"status": "error",
                "message": f"path must be inside the {root_label} ({root})."}
    if not rp.exists():
        return {"status": "error", "message": f"No such file: {rp}"}
    if rp.suffix.lower() not in allowed_suffixes:
        shown = ", ".join(sorted(s for s in allowed_suffixes if s))
        return {"status": "error",
                "message": (f"'{rp.suffix}' is not an editable text "
                            f"format ({shown}).")}
    if not old_text:
        return {"status": "error",
                "message": "old_text must be a non-empty snippet."}
    if old_text == new_text:
        return {"status": "error",
                "message": "old_text and new_text are identical."}
    if max(len(old_text), len(new_text or "")) > max_len:
        return {"status": "error", "message": too_large_message}

    current = rp.read_text(errors="replace")
    n = current.count(old_text)
    if n == 0:
        return {"status": "error",
                "message": ("old_text not found — read_file the document and "
                            "copy the snippet verbatim, whitespace included.")}
    if n > 1 and not replace_all:
        return {"status": "error",
                "message": (f"old_text matches {n} places — extend the "
                            "snippet until it is unique, or pass "
                            "replace_all=true to change every occurrence.")}

    bak: Optional[Path] = None
    try:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        i, bak = 1, backup_dir / f"{rp.stem}.before_edit{rp.suffix}"
        while bak.exists():
            i += 1
            bak = backup_dir / f"{rp.stem}.before_edit{i}{rp.suffix}"
        bak.write_text(current)
    except Exception as e:  # noqa: BLE001 - never block the edit
        logging.warning(f"Pre-edit copy failed: {e}")
        bak = None

    rp.write_text(current.replace(old_text, new_text)
                  if replace_all else
                  current.replace(old_text, new_text, 1))
    return {
        "status": "success",
        "path": str(rp),
        "replacements": n if replace_all else 1,
        "previous_version": str(bak) if bak else None,
    }


def rename_or_copy_file(
    src: Path,
    dest: Path,
    *,
    root: Path,
    copy: bool = False,
    root_label: str = "session directory",
) -> Dict[str, Any]:
    """Byte-exact rename (or copy) of a session file — content never
    passes through a model.

    Exists because the alternative was observed live: with no rename
    tool, an agent reconstructed a 30 KB document through save_file +
    append_file chunks to change its filename, dropping content on the
    way and leaving divergent duplicates. Both paths must be resolved
    (absolute) and inside `root`; `dest` must not exist (never clobber).

    A markdown file's PDF twin follows the move/copy automatically (when
    one exists and the destination twin is free), preserving the
    twins-never-go-stale invariant across renames.
    """
    import shutil

    src, dest, root = Path(src), Path(dest), Path(root).resolve()
    for label, p in (("path", src), ("new path", dest)):
        if root not in p.parents:
            return {"status": "error",
                    "message": (f"{label} must be inside the {root_label} "
                                f"({root}).")}
    if not src.is_file():
        return {"status": "error", "message": f"No such file: {src}"}
    if src == dest:
        return {"status": "error",
                "message": "path and new path are identical."}
    if dest.exists():
        return {"status": "error",
                "message": (f"Destination already exists: {dest} — "
                            "refusing to overwrite it.")}

    dest.parent.mkdir(parents=True, exist_ok=True)
    op = shutil.copy2 if copy else shutil.move
    op(str(src), str(dest))

    twin_followed = False
    src_pdf, dest_pdf = src.with_suffix(".pdf"), dest.with_suffix(".pdf")
    if (src.suffix.lower() == ".md" and src_pdf.exists()
            and not dest_pdf.exists()):
        try:
            op(str(src_pdf), str(dest_pdf))
            twin_followed = True
        except Exception as e:  # noqa: BLE001 - never undo the main move
            logging.warning(f"PDF twin did not follow the rename: {e}")

    return {
        "status": "success",
        "path": str(dest),
        "previous_path": str(src),
        "copied": bool(copy),
        "pdf_twin_followed": twin_followed,
    }


def refresh_pdf_twin(md_path: Path) -> Tuple[bool, Optional[str]]:
    """Re-export a markdown document's PDF twin, if one exists.

    Deterministic markdown_to_pdf re-export — no LLM, no content change —
    so an in-place edit or revision never leaves a stale PDF beside a
    fresh .md. Returns ``(refreshed, error)``: ``(True, None)`` on refresh,
    ``(False, None)`` when there is no twin to refresh, ``(False, msg)``
    when the twin exists but re-export failed. Callers own the transcript
    prints; a failure must never block the write that triggered it.
    """
    md_path = Path(md_path)
    pdf = md_path.with_suffix(".pdf")
    if md_path.suffix.lower() != ".md" or not pdf.exists():
        return False, None
    try:
        from .md_to_pdf import markdown_to_pdf
        markdown_to_pdf(md_path)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
