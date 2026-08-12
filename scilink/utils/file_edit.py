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

Consumers today: the planning and simulation orchestrators' `edit_file`;
the analysis orchestrator is the intended follow-up. VASP inputs (POSCAR /
INCAR / KPOINTS) are extensionless, which is why the suffix policy is a
parameter: include "" in `allowed_suffixes` to permit extensionless files.
Doing so widens the gate past what a name can decide, so the size and
binary/UTF-8 content guards below carry the rest of that decision.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

# Text formats editable by default. A mode may pass its own set; include ""
# to allow extensionless files (VASP inputs).
DEFAULT_EDITABLE_SUFFIXES: Set[str] = {
    ".md", ".html", ".mmd", ".txt", ".json", ".py", ".csv", ".yaml", ".yml",
}

# Surgical means SMALL — see module docstring.
DEFAULT_MAX_SNIPPET_LEN = 2000

# The suffix allowlist cannot carry the whole "is this editable" decision
# once "" is in it: VASP writes its INPUTS extensionless (INCAR, POSCAR,
# KPOINTS) and its OUTPUTS extensionless too (WAVECAR, CHGCAR, WAVEDER),
# and the two are indistinguishable by name. So content and size are
# checked as well — a NUL in the first few KB means binary, and a file
# past the byte cap is a run artifact rather than a deck under edit.
# CHGCAR is the case that needs both: it is ASCII, so only the cap stops
# it. Cap chosen to clear the largest plausible text deck (a LAMMPS
# .data for a few hundred thousand atoms) by a wide margin.
DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
_BINARY_SNIFF_BYTES = 8192

DEFAULT_TOO_LARGE_MESSAGE = (
    "Edit too large for edit_file — split the change at a unique boundary "
    "into consecutive smaller edit_file calls."
)

# The recovery route when a verbatim snippet guess misses. Parameterized
# for the same reason as `too_large_message`: each mode names a surface it
# actually has (planning has read_file; simulate does not).
DEFAULT_NOT_FOUND_MESSAGE = (
    "read_file the document and copy the snippet verbatim, "
    "whitespace included."
)


def apply_snippet_edits(
    text: str,
    edits: list,
    *,
    max_len: int = DEFAULT_MAX_SNIPPET_LEN,
    too_large_message: str = DEFAULT_TOO_LARGE_MESSAGE,
    not_found_message: str = DEFAULT_NOT_FOUND_MESSAGE,
) -> Dict[str, Any]:
    """Atomically apply a SEQUENCE of exact-snippet replacements to text.

    Each edit is ``{old_text, new_text, replace_all?}`` with the single-edit
    semantics (non-empty, non-identical, per-snippet cap, exactly one match
    unless replace_all). Edits apply IN ORDER, each against the text as
    already edited by its predecessors. All-or-nothing: the first failing
    edit aborts the whole list — the returned error names it (1-based
    ``failed_edit``) and nothing is considered applied. On success returns
    ``{"status": "success", "text", "replacements", "n_edits"}``.

    This is the batching primitive: a document-wide set of small changes is
    ONE call carrying the whole list, not one tool call per sentence (live:
    an 18-call edit chain burned the delegation's tool-iteration budget and
    left an 18-deep backup pile).
    """
    if not edits:
        return {"status": "error", "message": "edits must be a non-empty list."}
    cur, total = text, 0
    # Single-edit calls keep the original (live-tested) message wording;
    # the "edit i/N" label and batch caveats appear only in true batches.
    batch = len(edits) > 1
    for i, e in enumerate(edits):
        label = f"edit {i + 1}/{len(edits)}: " if batch else ""
        if not isinstance(e, dict):
            return {"status": "error", "failed_edit": i + 1,
                    "message": f"{label}each edit must be an object "
                               "with old_text and new_text."}
        old = e.get("old_text") or ""
        new = e.get("new_text") or ""
        ra = bool(e.get("replace_all"))
        if not old:
            return {"status": "error", "failed_edit": i + 1,
                    "message": f"{label}old_text must be a non-empty snippet."}
        if old == new:
            return {"status": "error", "failed_edit": i + 1,
                    "message": f"{label}old_text and new_text are identical."}
        if max(len(old), len(new)) > max_len:
            return {"status": "error", "failed_edit": i + 1,
                    "message": f"{label}{too_large_message}"}
        n = cur.count(old)
        if n == 0:
            batch_note = (" Note edits apply IN ORDER: a later edit must "
                          "match the text as already changed by earlier "
                          "edits. Nothing was applied.") if batch else ""
            return {"status": "error", "failed_edit": i + 1,
                    "message": (
                        f"{label}old_text not found — "
                        f"{not_found_message}{batch_note}")}
        if n > 1 and not ra:
            batch_note = (" Nothing was applied." if batch else "")
            return {"status": "error", "failed_edit": i + 1,
                    "message": (
                        f"{label}old_text matches {n} places — extend the "
                        "snippet until it is unique, or pass "
                        f"replace_all=true to change every occurrence."
                        f"{batch_note}")}
        cur = cur.replace(old, new) if ra else cur.replace(old, new, 1)
        total += n if ra else 1
    return {"status": "success", "text": cur,
            "replacements": total, "n_edits": len(edits)}


def apply_surgical_edit(
    path: Path,
    old_text: str,
    new_text: str,
    *,
    root: Path,
    backup_dir: Path,
    replace_all: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Single-edit convenience wrapper over :func:`apply_surgical_edits`."""
    return apply_surgical_edits(
        path,
        [{"old_text": old_text, "new_text": new_text,
          "replace_all": replace_all}],
        root=root, backup_dir=backup_dir, **kwargs)


def apply_surgical_edits(
    path: Path,
    edits: list,
    *,
    root: Path,
    backup_dir: Path,
    max_len: int = DEFAULT_MAX_SNIPPET_LEN,
    allowed_suffixes: Set[str] = DEFAULT_EDITABLE_SUFFIXES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    too_large_message: str = DEFAULT_TOO_LARGE_MESSAGE,
    not_found_message: str = DEFAULT_NOT_FOUND_MESSAGE,
    root_label: str = "session directory",
) -> Dict[str, Any]:
    """Apply an atomic batch of exact-snippet edits to `path`, guarded and
    backed up.

    `path` must already be resolved (absolute); relative-name resolution
    against a delegation/output directory is the caller's job. Returns a
    dict with ``status: "success"`` (plus ``path``, ``replacements``,
    ``n_edits``, ``previous_version``, ``backup_created``) or
    ``status: "error"`` (plus ``message``, and ``failed_edit`` when a
    specific edit failed). Never raises for expected failures.

    Path guards: sandbox (inside `root`), existence, suffix allowlist,
    byte cap, and a binary/UTF-8 content check. The last two exist because
    a mode that allows "" (extensionless VASP inputs) thereby also matches
    extensionless run OUTPUTS — see `DEFAULT_MAX_FILE_BYTES`. Per-edit
    guards live in :func:`apply_snippet_edits` (all-or-nothing);
    `too_large_message` and `not_found_message` are the mode's routing
    hints — each should name a surface that mode actually has.

    Backup: ONE per file per `backup_dir` — ``<stem>.before_edit<suffix>``
    holds the state before this backup dir's FIRST edit of the file, and
    later edits (or batches) from the same dir do not add more. The chain
    origin is the meaningful audit artifact; per-edit intermediates were
    noise at scale (live: an 18-edit chain recorded 18 near-identical
    backups, each re-embedded by the UI). ``previous_version`` always names
    the chain-origin copy; ``backup_created`` says whether THIS call wrote
    it. Caveat: the name is keyed by filename, so two same-named files
    edited from one backup_dir share a slot — first wins. Backup failure is
    logged and never blocks the edit.
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

    size = rp.stat().st_size
    if size > max_file_bytes:
        return {"status": "error",
                "message": (
                    f"{rp.name} is {size / (1024 * 1024):.1f} MB, over the "
                    f"{max_file_bytes // (1024 * 1024)} MB in-place edit cap "
                    "— a file this size is a run output, not an editable "
                    "input.")}
    # Bytes, then a STRICT decode. read_text(errors="replace") here does not
    # merely mangle the edit — it rewrites every undecodable byte in the
    # whole file as U+FFFD and reports success, so refusing is the only safe
    # answer once the file is not text.
    try:
        raw = rp.read_bytes()
    except OSError as e:
        return {"status": "error",
                "message": f"Could not read {rp.name}: {e}"}
    if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
        return {"status": "error",
                "message": (f"{rp.name} is a binary file — refusing to "
                            "edit it as text.")}
    try:
        current = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"status": "error",
                "message": (f"{rp.name} is not valid UTF-8 text — refusing "
                            "to edit it as text.")}

    res = apply_snippet_edits(current, edits, max_len=max_len,
                              too_large_message=too_large_message,
                              not_found_message=not_found_message)
    if res["status"] != "success":
        return res

    bak: Optional[Path] = None
    created = False
    try:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        bak = backup_dir / f"{rp.stem}.before_edit{rp.suffix}"
        if not bak.exists():
            # copyfile, not write_text(current): the backup is the only
            # route back from a bad edit, so it must be the file's bytes
            # rather than a re-encoding of what we decoded from them.
            shutil.copyfile(rp, bak)
            created = True
    except Exception as e:  # noqa: BLE001 - never block the edit
        logging.warning(f"Pre-edit copy failed: {e}")
        bak = None

    rp.write_text(res["text"], encoding="utf-8")
    return {
        "status": "success",
        "path": str(rp),
        "replacements": res["replacements"],
        "n_edits": res["n_edits"],
        "previous_version": str(bak) if bak else None,
        "backup_created": created,
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
