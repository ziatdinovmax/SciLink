"""User-provided reference scripts for the analysis agents.

A scientist attaches a script in chat and says "use it". Semantically that
is what a script-bank hit or a ``prior_analysis_paths`` run already is —
a proven implementation to ADAPT rather than reimplement — but it arrives
as a bare file rather than a prior run's artifact bundle, so it needs its
own channel: ``run_analysis(reference_scripts=[...])`` → ``analyze(
reference_scripts=...)`` → ``state["reference_scripts"]`` → a prompt block
wherever the agents show the user's guidance (planning, code generation,
refinement and correction prompts). The generated code is then verified by
the agent's normal sandbox / QC loop, exactly as a bank exemplar is — so a
mangled adaptation is corrected, not silently wrong.

Package-neutral: stdlib only, no agent imports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MAX_SCRIPT_BYTES = 64_000
_TEXT_SUFFIXES = {".py", ".txt", ".md", ".r", ".jl", ".m"}


def load_reference_scripts(paths: Optional[Sequence[str]], *,
                           max_bytes: int = MAX_SCRIPT_BYTES) -> List[Dict[str, Any]]:
    """Read the user's scripts into ``[{"label", "path", "text", "truncated"}]``.

    A missing, binary, or non-text file is skipped with a warning rather
    than failing the run; a long script is cut at ``max_bytes`` with the
    cut noted, so the prompt stays bounded.
    """
    out: List[Dict[str, Any]] = []
    for raw in paths or []:
        p = Path(str(raw)).expanduser()
        if not p.is_file():
            logger.warning(f"reference script not found, skipped: {raw}")
            continue
        if p.suffix.lower() not in _TEXT_SUFFIXES:
            logger.warning(f"reference script is not a text/script file, skipped: {p.name}")
            continue
        try:
            data = p.read_bytes()
        except OSError as exc:
            logger.warning(f"reference script unreadable, skipped: {p.name} ({exc})")
            continue
        if b"\x00" in data[:4096]:
            logger.warning(f"reference script looks binary, skipped: {p.name}")
            continue
        truncated = len(data) > max_bytes
        text = data[:max_bytes].decode("utf-8", errors="replace")
        out.append({"label": p.name, "path": str(p), "text": text,
                    "truncated": truncated})
    return out


def reference_script_block(state: Dict[str, Any], *, heading: str = "##") -> str:
    """The prompt block for ``state['reference_scripts']`` (empty string when
    none, so callers append unconditionally). ``heading`` is the markdown
    level the surrounding prompt uses ("##" or "---" style handled by the
    caller through this prefix)."""
    scripts = state.get("reference_scripts") or []
    if not scripts:
        return ""
    lines = [
        f"\n{heading} User-Provided Reference Script"
        + ("s" if len(scripts) > 1 else ""),
        "The user attached the script(s) below and asked that they be USED. "
        "Treat each like a proven script from the script bank: ADAPT it "
        "rather than reimplementing from scratch — keep its method, model "
        "choices, parameters and thresholds unless this data demonstrably "
        "needs a change; conform its input/output to this run's contract "
        "(read the staged input the instructions name, write the required "
        "outputs and markers) and drop command-line or plotting glue that "
        "conflicts with it; keep its variable and function names where "
        "practical so the user recognizes their code. State in the plan and "
        "in the script's header comment what was kept and what was changed, "
        "and why. Do not run it verbatim: it is a recipe, not the deliverable.",
    ]
    for s in scripts:
        note = " (truncated to the first 64 kB)" if s.get("truncated") else ""
        lines.append(f"\n### {s['label']}{note}\n```python\n{s['text']}\n```")
    return "\n".join(lines)
