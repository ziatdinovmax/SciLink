"""Generic file I/O engine shared by the orchestrators' file tools (#481).

One implementation per operation; everything mode-specific arrives as a
parameter. The orchestrators keep thin wrappers that own tool registration,
transcript prints, relative-path resolution, the mode-flavored routing text,
and (planning) deliverable recording + the PDF-twin invariant. This is the
same split ``file_edit.py`` already uses for ``edit_file`` / ``rename_file``.

Why one engine: the planning, analysis and simulation copies had started to
need the *same* fixes — the JSON pretty-print with no cap, and
backup-on-overwrite for ``save_file`` — and each mode's copy had drifted in
small accidental ways. Both fixes land here, once:

* ``read_file_content`` windows a pretty-printed JSON exactly like text
  (``max_lines`` / ``offset`` / ``tail`` / ``search``), so a large JSON no
  longer lands whole in context; a small one is returned whole as before.
* ``write_text_file`` backs up an existing file before overwriting it
  (``<stem>.before_overwrite<suffix>``, counter-suffixed, never clobbering),
  the way ``edit_file`` always backed up before an edit.

Every function returns a plain dict the wrapper serializes; nothing here
prints or raises for the ordinary failure modes (a missing file, a bad
pattern) — those come back as ``{"status": "error", "message": ...}``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── read_file ────────────────────────────────────────────────────────

# Size guards: documents get more headroom because extraction is
# page-based and a figure-heavy PDF is megabytes of images, not text.
# Tabular files are previewed (rows × cols), so they carry no byte cap.
DEFAULT_TEXT_CAP_MB = 5
DEFAULT_DOC_CAP_MB = 25
_TABULAR = (".xlsx", ".xls", ".csv")
_DOCUMENT = (".pdf", ".docx")
MAX_PREVIEW_ROWS = 100
MAX_PREVIEW_COLS = 40
MAX_PREVIEW_CHARS = 30_000
SEARCH_HIT_CAP = 40
DEFAULT_FULL_READ_MAX_CHARS = 250_000


def _size_error(path: Path, cap_mb: float) -> Optional[Dict[str, Any]]:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > cap_mb:
        return {"status": "error",
                "message": f"File too large ({size_mb:.1f} MB)."}
    return None


def _tabular_preview(path: Path, ext: str) -> str:
    import pandas as pd

    if ext == ".csv":
        df_preview = pd.read_csv(path, nrows=MAX_PREVIEW_ROWS)
        with open(path) as _f:
            total_rows = sum(1 for _ in _f) - 1
    else:
        df_preview = pd.read_excel(path, nrows=MAX_PREVIEW_ROWS)
        try:
            import openpyxl
            _wb = openpyxl.load_workbook(path, read_only=True)
            total_rows = _wb.active.max_row - 1
            _wb.close()
        except Exception:  # noqa: BLE001 - fall back to what pandas read
            total_rows = len(df_preview)
    total_cols = len(df_preview.columns)
    display_df = df_preview.iloc[:, :MAX_PREVIEW_COLS]
    preview_text = display_df.to_string()
    # Adaptive row reduction if the rendering exceeds the char budget.
    if len(preview_text) > MAX_PREVIEW_CHARS and len(display_df) > 5:
        ratio = MAX_PREVIEW_CHARS / len(preview_text)
        fewer_rows = max(5, int(len(display_df) * ratio))
        display_df = display_df.iloc[:fewer_rows]
        preview_text = display_df.to_string()
        if len(preview_text) > MAX_PREVIEW_CHARS:
            preview_text = preview_text[:MAX_PREVIEW_CHARS] + "\n... (truncated)"
    shown_rows, shown_cols = len(display_df), len(display_df.columns)
    trunc_parts = []
    if shown_rows < total_rows:
        trunc_parts.append(f"first {shown_rows} rows")
    if shown_cols < total_cols:
        trunc_parts.append(f"first {shown_cols} columns")
    trunc = f" (showing {', '.join(trunc_parts)})" if trunc_parts else ""
    return f"Shape: {total_rows} rows × {total_cols} columns{trunc}\n\n{preview_text}"


def _outline(lines: Sequence[str]) -> str:
    """Section headings with line numbers, so a truncated read says where
    the rest is instead of only that it is missing."""
    heads = [(i + 1, ln.strip()) for i, ln in enumerate(lines)
             if ln.startswith("#")]
    if len(heads) < 2:
        return ""
    return "\nSections: " + " · ".join(
        f"{h.lstrip('# ')[:44]} @ line {n}" for n, h in heads[:12]) + (
        " …" if len(heads) > 12 else "")


def window_lines(lines: List[str], *, max_lines: int = 200,
                 tail: bool = False, search: Optional[str] = None,
                 offset: Optional[int] = None,
                 whole: bool = False) -> Dict[str, Any]:
    """The windowing half of ``read_file`` over an already-loaded list of
    lines (``keepends`` style): search / offset / tail / head, with the
    truncation notices that name the way to the rest. Returns the result
    fields (``mode``, ``content``, ...) without ``status`` / ``file_path``.
    """
    total = len(lines)
    if search:
        # The real question behind most repeat reads is "is X in here, and
        # where" — a search, not a read. Answer it in one cheap call.
        try:
            rx = re.compile(search, re.I)
        except re.error as e:
            return {"status": "error",
                    "message": f"Invalid search pattern: {e}"}
        hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
        shown, out = hits[:SEARCH_HIT_CAP], []
        for i in shown:
            lo, hi = max(0, i - 1), min(total, i + 2)
            out.append(f"@@ line {i + 1}\n" + "".join(lines[lo:hi]).rstrip("\n"))
        body = "\n\n".join(out) if out else "(no matches)"
        note = (f"{len(hits)} matching line(s) in {total} total"
                + (f"; showing the first {SEARCH_HIT_CAP}"
                   if len(hits) > SEARCH_HIT_CAP else ""))
        return {"mode": "search", "pattern": search, "matches": len(hits),
                "match_lines": [i + 1 for i in shown], "total_lines": total,
                "content": f"{note}\n\n{body}"}

    truncated = True
    if whole or total <= max_lines:
        first, last, content = 1, total, "".join(lines)
        truncated = False
    elif offset is not None:
        start = max(0, offset - 1)
        shown = lines[start:start + max_lines]
        first, last = start + 1, min(total, start + max_lines)
        content = "".join(shown) + (
            f"\n... (showing lines {first}-{last} of {total}."
            f"{_outline(lines)})")
    elif tail:
        shown = lines[-max_lines:]
        first, last = total - max_lines + 1, total
        more = (f"... ({first - 1} earlier lines not shown; "
                f"omit tail to read from the top)")
        content = more + "\n" + "".join(shown)
    else:
        shown = lines[:max_lines]
        first, last = 1, max_lines
        content = "".join(shown) + (
            f"\n... ({total - max_lines} more lines not shown — this is a "
            f"TRUNCATED READ, not the whole file. Jump to any part with "
            f"offset=<line>; read the END with tail=true; find something "
            f"with search='<pattern>'; or raise max_lines."
            f"{_outline(lines)})")
    return {"mode": "tail" if tail else "head", "total_lines": total,
            "shown_lines": f"{first}-{last}", "truncated": truncated,
            "content": content}


DIRECTORY_LISTING_MAX_ENTRIES = 300


def list_directory(path: Path, *, display_path: Optional[str] = None,
                   max_entries: int = DIRECTORY_LISTING_MAX_ENTRIES) -> Dict[str, Any]:
    """What ``read_file`` returns for a directory (#591): its contents, so an
    agent holding a folder path can reach the file inside instead of
    concluding the file is elsewhere. Sub-directories come first (with their
    file counts), then files with sizes; ``scripts/*.py`` and other saved
    scripts are called out because they are the artifact most often looked
    for in an analysis output directory.
    """
    path = Path(path)
    try:
        children = sorted(path.iterdir(), key=lambda c: (c.is_file(), c.name.lower()))
    except OSError as e:
        return {"status": "error", "message": f"Cannot list {display_path or path}: {e}"}
    entries: List[Dict[str, Any]] = []
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_dir():
            try:
                n = sum(1 for c in child.iterdir() if not c.name.startswith("."))
            except OSError:
                n = None
            entries.append({"name": child.name + "/", "type": "directory", "entries": n})
        else:
            try:
                size = child.stat().st_size
            except OSError:
                size = None
            entries.append({"name": child.name, "type": "file", "size_bytes": size})
    truncated = len(entries) > max_entries
    entries = entries[:max_entries]
    scripts = [e["name"] for e in entries if e["name"].endswith(".py")]
    scripts_dir = path / "scripts"
    if scripts_dir.is_dir():
        scripts += sorted(f"scripts/{c.name}" for c in scripts_dir.iterdir() if c.suffix == ".py")
    out: Dict[str, Any] = {
        "status": "success", "file_path": str(path), "is_directory": True,
        "entries": entries, "truncated": truncated,
        "hint": (f"'{display_path or path}' is a directory, not a file. Call read_file on one "
                 "of the entries above (a file's path is this directory joined with its name)."),
    }
    if scripts:
        out["scripts"] = scripts
        out["hint"] += (" Saved scripts: " + ", ".join(scripts[:8])
                        + (" …" if len(scripts) > 8 else "") + ".")
    return out


def read_file_content(path: Path, *, max_lines: int = 200,
                      tail: bool = False, search: Optional[str] = None,
                      offset: Optional[int] = None,
                      full_read_stems: Iterable[str] = (),
                      full_read_max_chars: int = DEFAULT_FULL_READ_MAX_CHARS,
                      ocr_model: Any = None,
                      text_cap_mb: float = DEFAULT_TEXT_CAP_MB,
                      doc_cap_mb: float = DEFAULT_DOC_CAP_MB,
                      display_path: Optional[str] = None) -> Dict[str, Any]:
    """The whole ``read_file`` reader over a resolved, existing file.

    JSON is pretty-printed then windowed like text (the cap #481 asked for:
    a small JSON is returned whole, a large one truncates with the same
    offset / tail / search escape hatches); CSV / Excel are previewed as a
    rows × cols table; PDF / DOCX are extracted through the shared parser
    (OCR fallback via ``ocr_model``) and windowed like text; everything else
    is read as UTF-8 text. ``full_read_stems`` names files (by substring of
    the name) that are returned whole up to ``full_read_max_chars``.
    """
    path = Path(path)
    if path.is_dir():
        return list_directory(path, display_path=display_path)
    if not path.is_file():
        return {"status": "error",
                "message": f"Not a file: {display_path or path}"}
    shown_path = str(path)
    try:
        ext = path.suffix.lower()
        if ext not in _TABULAR:
            err = _size_error(path, doc_cap_mb if ext in _DOCUMENT else text_cap_mb)
            if err:
                return err

        if ext in _TABULAR:
            return {"status": "success", "file_path": shown_path,
                    "content": _tabular_preview(path, ext)}

        doc_meta: Dict[str, Any] = {}
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            lines = json.dumps(data, indent=2).splitlines(keepends=True)
        elif ext in _DOCUMENT:
            # Opened as text a PDF returns its compressed byte streams (#397);
            # extraction is shared infrastructure — route through it.
            from scilink.parsers.extract import extract_text
            info = extract_text(str(path), ocr_model=ocr_model)
            raw = info.get("text") or ""
            if not raw.strip():
                return {"status": "error",
                        "message": (f"No extractable text in {path.name} "
                                    "(empty or image-only document).")}
            lines = raw.splitlines(keepends=True)
            doc_meta = {k: info[k] for k in ("n_pages", "n_ocr_pages", "n_paragraphs")
                        if info.get(k) is not None}
            doc_meta["extracted"] = ext.lstrip(".")
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

        name = path.name.lower()
        whole = (any(s in name for s in full_read_stems)
                 and offset is None and not tail
                 and len("".join(lines)) <= full_read_max_chars)
        win = window_lines(lines, max_lines=max_lines, tail=tail,
                           search=search, offset=offset, whole=whole)
        if win.get("status") == "error":
            return win
        return {"status": "success", "file_path": shown_path, **win, **doc_meta}
    except Exception as e:  # noqa: BLE001 - the tool reports, never raises
        return {"status": "error", "message": f"Failed to read file: {e}"}


# ── save_file / append_file ─────────────────────────────────────────

def _next_backup_path(dest: Path) -> Path:
    """``<stem>.before_overwrite<suffix>``, then ``.before_overwrite.2``,
    ``.3``, … — never clobbering an earlier backup."""
    base = dest.with_name(f"{dest.stem}.before_overwrite{dest.suffix}")
    if not base.exists():
        return base
    n = 2
    while True:
        cand = dest.with_name(f"{dest.stem}.before_overwrite.{n}{dest.suffix}")
        if not cand.exists():
            return cand
        n += 1


def write_text_file(base_dir: Path, filename: str, content: str, *,
                    subfolder: str = "", append: bool = False,
                    backup_on_overwrite: bool = True) -> Dict[str, Any]:
    """Write (or append) UTF-8 text under ``base_dir``.

    Traversal-proof: only the last path component of ``filename`` and of
    ``subfolder`` is used, so ``../../x`` lands inside the session. On an
    overwrite (``append=False``, file exists, ``backup_on_overwrite``) the
    previous content is copied to a counter-suffixed ``.before_overwrite``
    sibling first — a generated INCAR or a report can no longer be silently
    clobbered while ``edit_file`` on the same file would have backed up.
    """
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", ".."):
        return {"status": "error", "message": "Invalid filename."}
    target_dir = Path(base_dir)
    if subfolder:
        safe_sub = Path(subfolder).name
        if safe_sub and safe_sub not in (".", ".."):
            target_dir = target_dir / safe_sub
    dest = target_dir / safe_name
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        existed = dest.exists()
        backup: Optional[str] = None
        if append:
            with open(dest, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            if existed and backup_on_overwrite and dest.is_file():
                bak = _next_backup_path(dest)
                try:
                    shutil.copyfile(dest, bak)
                    backup = str(bak)
                except OSError as exc:  # keep writing: a lost backup beats a lost write
                    logger.warning(f"backup before overwrite failed for {dest}: {exc}")
            dest.write_text(content, encoding="utf-8")
        out: Dict[str, Any] = {"status": "success", "path": str(dest),
                               "size_bytes": dest.stat().st_size,
                               "created": not existed}
        if not append:
            out["overwritten"] = existed
            if backup:
                out["backup"] = backup
        return out
    except Exception as e:  # noqa: BLE001
        logger.error(f"{'append_file' if append else 'save_file'} failed: {e}")
        return {"status": "error", "message": str(e)}


# ── read_document ───────────────────────────────────────────────────

DEFAULT_READ_DOC_MAX_CHARS = 200_000  # ~50k tokens; longer is truncated


def extract_document_text(path: Path, ocr_model: Any = None,
                          max_chars: int = DEFAULT_READ_DOC_MAX_CHARS
                          ) -> Dict[str, Any]:
    """Extract plain text from a PDF / DOCX / Markdown / text file.

    Thin wrapper over the shared ``scilink.parsers.extract_text`` (table-
    aware PDF extraction, vision-OCR fallback for scanned pages when
    ``ocr_model`` is given) that applies the per-document character cap.
    Returns ``text`` plus metadata (page / paragraph counts, ``n_chars``,
    ``truncated``, ``n_ocr_pages``). Raises ValueError for an unsupported
    extension; reader errors propagate to the caller.
    """
    from scilink.parsers import extract_text

    info = extract_text(path, ocr_model=ocr_model)
    text = info.get("text", "")
    info["truncated"] = len(text) > max_chars
    if info["truncated"]:
        text = text[:max_chars]
        info["text"] = text
        info["n_chars"] = len(text)
    return info


def read_documents_combined(paths: Sequence[str], *, base_dir: Optional[Path] = None,
                            ocr_model: Any = None,
                            max_chars: int = DEFAULT_READ_DOC_MAX_CHARS
                            ) -> Dict[str, Any]:
    """Extract several documents and combine them into one text block.

    Relative paths resolve against ``base_dir`` when given. Returns
    ``{"status": "error", ...}`` when nothing could be read, otherwise the
    combined text with per-document metadata, the errors for the ones that
    failed, and the OCR page count — the wrapper adds its mode's extras
    (analysis persists a literature file; sim / planning do not).
    """
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        return {"status": "error", "message": "No document path provided."}
    docs: List[Tuple[Path, Dict[str, Any]]] = []
    errors: List[str] = []
    for p in paths:
        dp = Path(p)
        if base_dir is not None and not dp.is_absolute():
            dp = Path(base_dir) / dp
        if not dp.is_file():
            errors.append(f"Not a file: {p}")
            continue
        try:
            docs.append((dp, extract_document_text(dp, ocr_model=ocr_model,
                                                   max_chars=max_chars)))
        except ValueError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            logger.error(f"read_document failed for {p}: {e}")
            errors.append(f"Could not read {dp.name}: {e}")
    if not docs:
        return {"status": "error", "message": "No documents could be read.",
                "errors": errors}
    combined = "\n\n---\n\n".join(f"## {dp.name}\n\n{info['text']}"
                                   for dp, info in docs)
    combined_truncated = len(combined) > max_chars
    if combined_truncated:
        combined = combined[:max_chars]
    n_ocr = sum(info.get("n_ocr_pages", 0) for _, info in docs)
    return {
        "status": "success",
        "n_documents": len(docs),
        "n_ocr_pages": n_ocr,
        "ocr_note": (f"{n_ocr} scanned page(s) had no text layer and were "
                     "transcribed by vision-OCR — verify any figures/numerics."
                     ) if n_ocr else None,
        "documents": [{"name": dp.name,
                       **{k: v for k, v in info.items() if k != "text"}}
                      for dp, info in docs],
        "errors": errors or None,
        "combined_truncated": combined_truncated,
        "text": combined,
    }
