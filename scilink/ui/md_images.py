"""Inline local images referenced by a markdown document.

Generated documents reference their figures by bare relative filename —
``![Campaign workflow diagram](campaign_workflow.png)`` saved beside the
markdown — which only resolves for a renderer running in the document's
own directory. Streamlit's markdown treats image targets as web URLs, so
the chat embed and the file-preview pane silently dropped such figures.
Rewriting local references to base64 data URIs makes the text
self-contained wherever it is displayed.

Streamlit-free on purpose so the UI-contract tests can exercise it
headless.
"""

import base64
import re
from pathlib import Path

# ![alt](target) with an optional "title" — target stops at whitespace or
# the closing paren, so titled refs keep their title untouched.
_IMG_REF = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp"}

# Inlining is per-rerun work in the chat loop; a figure past this size
# stays a link rather than slowing every repaint.
_MAX_INLINE_BYTES = 4_000_000


def inline_local_images(text: str, base_dir,
                        max_bytes: int = _MAX_INLINE_BYTES) -> str:
    """Return ``text`` with resolvable local image refs inlined as data URIs.

    Remote (http/https) and already-inlined (data:) targets pass through
    unchanged, as does any reference that does not resolve to a readable
    image file under ``max_bytes`` — a broken ref renders exactly as before
    rather than raising.
    """
    base = Path(base_dir)

    def swap(m: re.Match) -> str:
        target = m.group(2)
        if target.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = Path(target)
        if not p.is_absolute():
            p = base / target
        mime = _MIME.get(p.suffix.lower())
        try:
            if not mime or not p.is_file() or p.stat().st_size > max_bytes:
                return m.group(0)
            payload = base64.b64encode(p.read_bytes()).decode("ascii")
        except OSError:
            return m.group(0)
        return f"{m.group(1)}data:{mime};base64,{payload}{m.group(3)}"

    return _IMG_REF.sub(swap, text)


def pdf_twin(md_path) -> Path | None:
    """The authoring-time PDF of a markdown document, when still current.

    Documents are written with a PDF twin beside them whose conversion ran
    in the document's directory (so relative figures resolved). Serving it
    beats regenerating from in-memory text in a temp dir, where relative
    figures cannot resolve. Returns None when the twin is missing or older
    than the markdown (a revision without a fresh conversion).
    """
    md_path = Path(md_path)
    twin = md_path.with_suffix(".pdf")
    try:
        if twin.is_file() and twin.stat().st_mtime >= md_path.stat().st_mtime - 1:
            return twin
    except OSError:
        pass
    return None
