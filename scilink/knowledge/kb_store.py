"""Named, persistent knowledge-base store.

Promotes knowledge bases from anonymous ``kb_storage`` piles coupled to a
launch directory into first-class named artifacts under the persistent
SciLink home (``~/.scilink/knowledge_bases/<name>/`` by default;
``SCILINK_HOME`` relocates the whole store):

    <store>/<name>/
    ├── manifest.json            # embedding model, dates, counts, sources
    ├── sources/                 # copied originals (enables rebuild)
    ├── default_kb_docs.faiss    # persisted index — the ``default_kb`` base
    ├── default_kb_docs.json     # name means a KB dir is directly usable as
    └── default_kb_docs.sources.json   # an orchestrator ``knowledge_dir``

Because the index files use the same ``default_kb`` base name the planning
stack derives from a ``knowledge_dir``, resolving a store name to its
directory composes with every existing consumer (orchestrator constructors,
``--knowledge-dir``, the meta's ``attach_knowledge_base``) with no loader
changes.

The manifest records which embedding model built the index — the provenance
whose absence made "KB built with Gemini, queried without a Gemini key"
fail opaquely at query time. ``embedding_compat_warning`` turns that into
an upfront, actionable message.

This module must stay importable without the agent stack: it may import
from ``scilink.knowledge`` and (lazily) ``scilink.parsers``, never from
``scilink.agents``.
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..skills.loader import scilink_home

_logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
KB_BASE_NAME = "default_kb"  # matches the planning stack's kb_base_path stem
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_DOCS_FILES = (
    f"{KB_BASE_NAME}_docs.faiss",
    f"{KB_BASE_NAME}_docs.json",
    f"{KB_BASE_NAME}_docs.sources.json",
)
_CODE_FILES = (
    f"{KB_BASE_NAME}_code.faiss",
    f"{KB_BASE_NAME}_code.json",
    f"{KB_BASE_NAME}_code.maps.json",
    f"{KB_BASE_NAME}_code.sources.json",
)


def kb_store_dir() -> Path:
    """Root of the named-KB store (created on demand)."""
    return scilink_home() / "knowledge_bases"


def _validate_name(name: str) -> str:
    if not name or not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid KB name {name!r}: use 1-64 letters, digits, dots, "
            "underscores or dashes (must start alphanumeric)."
        )
    return name


def kb_path(name: str) -> Path:
    """Path of a named KB's directory (whether or not it exists yet)."""
    return kb_store_dir() / _validate_name(name)


def read_manifest(kb_dir: Path) -> Optional[Dict[str, Any]]:
    """Return the manifest dict of a KB directory, or None if absent/bad."""
    mf = Path(kb_dir) / MANIFEST_NAME
    if not mf.is_file():
        return None
    try:
        return json.loads(mf.read_text())
    except Exception as e:  # noqa: BLE001 - a corrupt manifest is reported, not fatal
        _logger.warning(f"Unreadable KB manifest at {mf}: {e}")
        return None


def list_kbs() -> List[Dict[str, Any]]:
    """All named KBs, each as its manifest plus ``name`` and ``path``."""
    store = kb_store_dir()
    if not store.is_dir():
        return []
    out = []
    for child in sorted(store.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        manifest = read_manifest(child) or {}
        manifest.setdefault("name", child.name)
        manifest["path"] = str(child)
        out.append(manifest)
    return out


def resolve_knowledge_source(value: str, strict: bool = True
                             ) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Resolve a ``knowledge_dir``-style string to a directory.

    Resolution order (documented contract):

    1. An existing directory PATH always wins.
    2. Otherwise a bare name (no path separator) is tried as a store NAME.
    3. Otherwise: with ``strict=True`` raise ``FileNotFoundError`` listing
       the available KB names (CLI flags / explicit attach — a typo should
       error); with ``strict=False`` return the value as a plain path — the
       legacy create-on-demand semantics orchestrator constructors rely on
       (e.g. the meta's session-scoped default, which doesn't exist yet at
       construction).

    Returns ``(path, manifest)`` — manifest is None for plain directories
    without one.
    """
    p = Path(value).expanduser()
    if p.is_dir():
        return p.resolve(), read_manifest(p)
    if os.sep not in str(value) and _NAME_RE.match(str(value)):
        cand = kb_store_dir() / value
        if cand.is_dir():
            return cand.resolve(), read_manifest(cand)
    if not strict:
        return p, None
    names = [k["name"] for k in list_kbs()]
    raise FileNotFoundError(
        f"'{value}' is neither an existing directory nor a named knowledge "
        f"base. Available KBs: {names or 'none'} (create one with "
        f"'scilink kb create')."
    )


def embedding_compat_warning(manifest: Optional[Dict[str, Any]],
                             session_embedding_model: str) -> Optional[str]:
    """Warning text when a KB's build-time embedding model doesn't match the
    session's, or is unrecorded. None when compatible."""
    if not manifest:
        return None
    built_with = manifest.get("embedding_model")
    if built_with in (None, "", "unknown"):
        return (
            f"KB '{manifest.get('name', '?')}' does not record its embedding "
            f"model; retrieval quality with '{session_embedding_model}' is "
            "unverified."
        )
    if built_with != session_embedding_model:
        return (
            f"KB '{manifest.get('name', '?')}' was built with "
            f"'{built_with}' but this session embeds queries with "
            f"'{session_embedding_model}' — retrieval will be degraded or "
            f"fail. Use --embedding-model {built_with}, or rebuild: "
            f"scilink kb rebuild {manifest.get('name', '<name>')} "
            f"--embedding-model {session_embedding_model}"
        )
    return None


def _write_manifest(kb_dir: Path, **fields) -> Dict[str, Any]:
    manifest = read_manifest(kb_dir) or {}
    manifest.update(fields)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    (kb_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def _source_names(kb_dir: Path) -> List[str]:
    """Basenames of the documents under ``sources/`` (recursive)."""
    src = kb_dir / "sources"
    if not src.is_dir():
        return []
    return sorted(f.name for f in src.rglob("*") if f.is_file())


def _build_index_into(target: Path, embedding_model: str,
                      api_key: Optional[str], base_url: Optional[str],
                      ocr_model: Any,
                      record_as: Optional[Path] = None) -> Tuple[int, int]:
    """Ingest ``target/sources`` and persist the docs index into ``target``.

    ``record_as`` is the sources path written into the KB's source history —
    pass the FINAL location when building in a staging dir, so later
    source-difference checks compare against the path that will exist.

    Returns (n_chunks, n_vectors). Raises on empty ingestion or embedding
    failure — callers stage ``target`` so a failure leaves no partial KB.
    """
    from ..parsers import ingest_files
    from .knowledge_base import KnowledgeBase

    chunks = ingest_files([str(target / "sources")], is_code_mode=False,
                          ocr_model=ocr_model)
    if not chunks:
        raise ValueError(
            "No ingestible content found in the provided sources "
            "(supported: PDF, text/markdown, Excel/CSV, images with OCR)."
        )

    kb = KnowledgeBase(
        embedding_model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        use_litellm=not base_url,
    )
    # Record the sources/ dir as a manifest-style source entry (same shape
    # the planning loader writes: {"path", "files"}), against its FINAL
    # location, so later source-difference checks recognise the documents
    # as already embedded instead of re-ingesting them.
    kb.sources.append({
        "path": str(record_as or (target / "sources")),
        "files": sorted(f.name for f in (target / "sources").rglob("*")
                        if f.is_file()),
    })
    kb.build(chunks)
    prefix = target / f"{KB_BASE_NAME}_docs"
    kb.save(
        str(prefix.with_suffix(".faiss")),
        str(prefix.with_suffix(".json")),
        sources_path=str(prefix.with_suffix(".sources.json")),
    )
    return len(chunks), int(kb.index.ntotal)


def create_kb(name: str,
              source_paths: List[str],
              embedding_model: str = "gemini-embedding-001",
              api_key: Optional[str] = None,
              base_url: Optional[str] = None,
              description: str = "",
              ocr_model: Any = None,
              overwrite: bool = False) -> Dict[str, Any]:
    """Build a named KB from documents. Atomic: staged in a temp dir and
    moved into place only after a successful build, so an embedding failure
    leaves no partial KB behind.

    Returns:
        The written manifest.
    """
    _validate_name(name)
    final = kb_path(name)
    if final.exists() and not overwrite:
        raise FileExistsError(
            f"KB '{name}' already exists. Pass overwrite=True "
            f"(CLI: --overwrite) to replace it."
        )
    missing = [p for p in source_paths if not Path(p).expanduser().exists()]
    if missing:
        raise FileNotFoundError(f"Source path(s) not found: {missing}")

    store = kb_store_dir()
    store.mkdir(parents=True, exist_ok=True)
    staging = store / f".staging_{name}"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "sources").mkdir(parents=True)

    try:
        for p in source_paths:
            src = Path(p).expanduser()
            dest = staging / "sources" / src.name
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dest)

        n_chunks, n_vectors = _build_index_into(
            staging, embedding_model, api_key, base_url, ocr_model,
            record_as=final / "sources",
        )
        _write_manifest(
            staging,
            name=name,
            description=description,
            embedding_model=embedding_model,
            origin="built",
            created_at=datetime.now().isoformat(timespec="seconds"),
            n_chunks=n_chunks,
            n_vectors=n_vectors,
            sources=_source_names(staging),
        )
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _logger.info(f"📚 KB '{name}' created: {n_vectors} vectors at {final}")
    return read_manifest(final)


def import_kb(name: str,
              from_dir: str,
              embedding_model: str = "unknown",
              description: str = "",
              overwrite: bool = False) -> Dict[str, Any]:
    """Adopt an existing anonymous KB directory (e.g. a launch-folder
    ``kb_storage``) into the store, copying its persisted index files.

    No original documents are copied, so an imported KB is not rebuildable;
    pass the embedding model that built it if known — 'unknown' produces a
    compatibility warning at attach time.
    """
    _validate_name(name)
    src = Path(from_dir).expanduser()
    docs_present = all((src / f).is_file() for f in _DOCS_FILES)
    if not docs_present:
        raise FileNotFoundError(
            f"{src} does not look like a KB dir: expected {_DOCS_FILES}."
        )
    final = kb_path(name)
    if final.exists() and not overwrite:
        raise FileExistsError(
            f"KB '{name}' already exists. Pass overwrite=True to replace it."
        )

    store = kb_store_dir()
    store.mkdir(parents=True, exist_ok=True)
    staging = store / f".staging_{name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for f in _DOCS_FILES + _CODE_FILES:
            if (src / f).is_file():
                shutil.copy2(src / f, staging / f)
        try:
            sources = json.loads((staging / _DOCS_FILES[2]).read_text())
            names: List[str] = []
            for entry in sources:
                if isinstance(entry, dict):
                    names.extend(entry.get("files") or
                                 [Path(entry.get("path", "")).name])
                else:
                    names.append(Path(str(entry)).name)
            source_names = sorted({n for n in names
                                   if n and not n.startswith(KB_BASE_NAME)})
        except Exception:  # noqa: BLE001 - listing is best-effort metadata
            source_names = []
        try:
            import faiss
            n_vectors = int(faiss.read_index(str(staging / _DOCS_FILES[0])).ntotal)
        except Exception:  # noqa: BLE001 - count is best-effort metadata
            n_vectors = None
        _write_manifest(
            staging,
            name=name,
            description=description,
            embedding_model=embedding_model,
            origin=f"imported from {src}",
            created_at=datetime.now().isoformat(timespec="seconds"),
            n_vectors=n_vectors,
            sources=source_names,
        )
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _logger.info(f"📚 KB '{name}' imported from {src} → {final}")
    return read_manifest(final)


def rebuild_kb(name: str,
               embedding_model: str,
               api_key: Optional[str] = None,
               base_url: Optional[str] = None,
               ocr_model: Any = None) -> Dict[str, Any]:
    """Re-embed a named KB's stored sources with a (new) embedding model.

    Requires the KB to have been created with sources (origin 'built');
    imported KBs carry no documents and cannot be rebuilt.
    """
    final = kb_path(name)
    if not final.is_dir():
        raise FileNotFoundError(f"KB '{name}' does not exist.")
    if not (final / "sources").is_dir():
        raise ValueError(
            f"KB '{name}' has no stored sources (imported KBs are not "
            "rebuildable) — recreate it from the original documents."
        )

    store = kb_store_dir()
    staging = store / f".staging_{name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copytree(final / "sources", staging / "sources")
        n_chunks, n_vectors = _build_index_into(
            staging, embedding_model, api_key, base_url, ocr_model,
            record_as=final / "sources",
        )
        old = read_manifest(final) or {}
        _write_manifest(
            staging,
            **{**old,
               "embedding_model": embedding_model,
               "n_chunks": n_chunks,
               "n_vectors": n_vectors},
        )
        shutil.rmtree(final)
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _logger.info(f"📚 KB '{name}' rebuilt with {embedding_model}")
    return read_manifest(final)


def delete_kb(name: str) -> None:
    """Remove a named KB from the store."""
    final = kb_path(name)
    if not final.is_dir():
        raise FileNotFoundError(f"KB '{name}' does not exist.")
    shutil.rmtree(final)
    _logger.info(f"🗑️ KB '{name}' deleted.")
