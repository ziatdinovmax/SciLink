"""UTF-8 text I/O — the encoding this codebase assumes everywhere but states
nowhere.

`open()`, `Path.read_text()` and `Path.write_text()` all fall back to the
*locale* encoding when no `encoding=` is given. On Linux and macOS that is
UTF-8, so the omission is invisible. On a Windows machine with a Western
European locale it is cp1252, which cannot encode the characters scientific
prose is actually made of — subscripts, arrows, Greek letters, the minus
sign. Persisting a literature answer that contains a single "→" therefore
raises `UnicodeEncodeError: 'charmap' codec can't encode character` and, in
the reported case, discarded a completed 13-minute Edison search.

These helpers state the encoding. On any UTF-8 system they are byte-for-byte
identical to the bare calls they replace — including newline translation,
which is left at the platform default so Windows keeps writing CRLF — and
only change behaviour where the bare call would have raised.

Reads are deliberately tolerant. A markdown file written by an EARLIER
SciLink run on a cp1252 machine is not valid UTF-8, so a strict UTF-8 read
would newly fail on artifacts that used to load fine. `read_text_utf8`
therefore tries UTF-8, falls back to the locale encoding, and only then
degrades to a lossy read — warning at each step down rather than failing.

Consumers today: the literature text path (search results, molecule design,
white paper, ideation report, provided documents) across the planning and
analysis orchestrators and the three analysis agents. Any other site that
persists or reloads model-generated prose should move here too.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def write_text_utf8(path: Any, text: str, *, append: bool = False) -> Path:
    """Write ``text`` to ``path`` as UTF-8, creating or truncating it.

    Args:
        path: Destination file.
        text: Content to write.
        append: Append instead of truncating.

    Returns:
        The destination as a ``Path``.
    """
    p = Path(path)
    # newline is left at the default so line-ending behaviour is unchanged
    # on every platform; only the codec is being pinned here.
    with open(p, "a" if append else "w", encoding="utf-8") as f:
        f.write(text)
    return p


def read_text_utf8(path: Any) -> str:
    """Read ``path`` as UTF-8, degrading rather than failing on legacy files.

    Falls back to the locale encoding for files an older SciLink wrote under
    a non-UTF-8 locale, then to a lossy UTF-8 read; each step down is logged.

    Args:
        path: File to read.

    Returns:
        The file's text.
    """
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass

    try:
        text = p.read_text()
        logging.warning(
            f"{p.name} is not valid UTF-8 — read using the locale encoding. "
            f"It was likely written by an older SciLink run on a non-UTF-8 "
            f"system; re-generating it will store UTF-8."
        )
        return text
    except UnicodeDecodeError:
        logging.warning(
            f"{p.name} decodes under neither UTF-8 nor the locale encoding — "
            f"reading it lossily; some characters will be replaced."
        )
        return p.read_text(encoding="utf-8", errors="replace")
