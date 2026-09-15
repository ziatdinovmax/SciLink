"""VASP INCAR syntax validation.

Deterministic, LLM-free checks for VASP INCAR files: detection of
unrecognized tags (typically one-character typos such as ``ISPN`` for
``ISPIN``) against pymatgen's canonical tag list, with closest-match
suggestions and an in-place high-confidence auto-rename.

VASP accepts unknown INCAR keys silently, so a typo like ``ISPN = 2``
disables spin polarization and yields a physically wrong result that
converges by every other metric. Catching this is a lookup problem with
a definitive answer, so it stays deterministic rather than going to an
LLM. Discovered via the skill registry when the ``vasp`` skill is active
and called through
:func:`scilink.skills._shared._registry.get_tool_function`.
"""

from __future__ import annotations

import json
import os
import re
import warnings as _warnings
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Dict, List, Optional, Tuple

from ..._shared._spec import ToolSpec


# Tokens that appear inside legitimate INCAR values (e.g. ``LDAUL = 3 3
# -1``, ``LSORBIT = .TRUE.``) rather than as standalone tags. Filtered out
# of the suggestion pool so a value token is never proposed as a rename.
_VASP_SUGGESTION_BLOCKLIST = {"TRUE", "FALSE"}


def _load_valid_vasp_tags() -> List[str]:
    """Return the canonical VASP INCAR-tag list bundled with pymatgen.

    Returns:
        The list of recognized INCAR tag names. Empty if pymatgen is
        unavailable or the bundled tag JSON cannot be located, in which
        case callers degrade gracefully (no rename suggestions).
    """
    try:
        from pymatgen.io.vasp import inputs as _vasp_inputs
    except Exception:
        return []
    tags_path = os.path.join(
        os.path.dirname(_vasp_inputs.__file__), "incar_parameters.json"
    )
    if not os.path.exists(tags_path):
        return []
    try:
        with open(tags_path) as f:
            return list(json.load(f).keys())
    except Exception:
        return []


def check_incar_syntax(incar_content: str) -> List[Dict[str, Any]]:
    """Check a VASP INCAR for unrecognized tags via pymatgen.

    Args:
        incar_content: Raw INCAR text (not a path). Pass the file
            contents, e.g. ``open(path).read()``.

    Returns:
        A list of issue dicts, one per unrecognized tag (possibly empty).
        Each dict carries:

            severity     ``"warning"``
            category     ``"incar_tag"``
            tag          The offending tag as written (str or None)
            suggested    Closest valid VASP tag (str or None)
            confidence   ``"high"`` or ``"low"``
            description  Human-readable summary
            source       ``"pymatgen Incar.check_params"``

        ``confidence="high"`` requires the closest match to be at least
        0.85 similar to the unrecognized tag and clearly better than the
        runner-up (margin > 0.05). Only high-confidence entries are
        consumed by :func:`apply_incar_syntax_fixes`; low-confidence
        entries are returned as context for an LLM regenerator.
    """
    try:
        from pymatgen.io.vasp.inputs import Incar
    except Exception:
        return []

    try:
        if hasattr(Incar, "from_str"):
            incar = Incar.from_str(incar_content)
        else:
            incar = Incar.from_string(incar_content)  # older pymatgen
    except Exception:
        # Malformed INCAR — let VASP itself complain. This pass targets
        # the "syntactically valid but contains a fake tag" failure mode.
        return []

    valid_tags = [
        t for t in _load_valid_vasp_tags()
        if t not in _VASP_SUGGESTION_BLOCKLIST
    ]

    issues: List[Dict[str, Any]] = []
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            incar.check_params()
        except Exception:
            return []

    for w in caught:
        msg = str(w.message)
        m = re.search(r"Cannot find\s+(\S+)", msg)
        bad_tag = m.group(1).strip().strip(",.") if m else None

        suggested: Optional[str] = None
        confidence = "low"
        if bad_tag and valid_tags:
            matches = get_close_matches(
                bad_tag.upper(), valid_tags, n=2, cutoff=0.7
            )
            if matches:
                suggested = matches[0]
                top_sim = SequenceMatcher(
                    None, bad_tag.upper(), matches[0]
                ).ratio()
                runner_sim = (
                    SequenceMatcher(None, bad_tag.upper(), matches[1]).ratio()
                    if len(matches) > 1 else 0.0
                )
                if top_sim >= 0.85 and (top_sim - runner_sim) >= 0.05:
                    confidence = "high"

        issues.append({
            "severity": "warning",
            "category": "incar_tag",
            "tag": bad_tag,
            "suggested": suggested,
            "confidence": confidence,
            "description": (
                f"INCAR tag '{bad_tag}' is not recognised by VASP. "
                f"Closest match: {suggested}."
                if bad_tag and suggested else msg
            ),
            "source": "pymatgen Incar.check_params",
        })

    return issues


def apply_incar_syntax_fixes(
    incar_content: str,
    issues: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply high-confidence tag renames in place to an INCAR string.

    Args:
        incar_content: Raw INCAR text.
        issues: Issues from a prior :func:`check_incar_syntax` call. When
            omitted, the check is run inline.

    Returns:
        A ``(fixed_content, applied)`` tuple. ``fixed_content`` is the
        possibly-modified INCAR text. ``applied`` is the subset of issues
        whose rename actually fired, each augmented with ``renamed_from``
        and ``renamed_to`` keys. Low-confidence issues are left untouched
        for the caller to forward to an LLM regenerator.
    """
    if issues is None:
        issues = check_incar_syntax(incar_content)

    fixed = incar_content
    applied: List[Dict[str, Any]] = []
    for issue in issues:
        if issue.get("category") != "incar_tag":
            continue
        if issue.get("confidence") != "high":
            continue
        bad = issue.get("tag")
        good = issue.get("suggested")
        if not bad or not good:
            continue
        # Only rename when ``bad`` is on the LHS of an assignment. The
        # word-boundary regex + line anchor avoids touching value tokens
        # that happen to share the spelling.
        pattern = re.compile(rf"(?im)^(\s*){re.escape(bad)}(\s*=)")
        new_fixed, n = pattern.subn(rf"\1{good}\2", fixed)
        if n > 0:
            fixed = new_fixed
            applied.append({**issue, "renamed_from": bad, "renamed_to": good})
    return fixed, applied


def check_input_syntax(input_files: Dict[str, str]) -> List[Dict[str, Any]]:
    """Engine-neutral pre-run syntax check entry point for VASP inputs.

    This is the conventionally-named tool that an engine-neutral input
    reviewer invokes (mirroring ``snapshot_run`` on the post-run side):
    it accepts the full input-files mapping, selects the file this engine
    knows how to check (the INCAR), and delegates to
    :func:`check_incar_syntax`.

    Args:
        input_files: Mapping of input filename to file contents. The
            entry whose name matches ``INCAR`` (case-insensitive) is
            checked; other entries are ignored.

    Returns:
        The issue list from :func:`check_incar_syntax`, or an empty list
        when no INCAR is present in ``input_files``.
    """
    for name, content in input_files.items():
        if name.upper() == "INCAR" or name.upper().endswith("/INCAR"):
            return check_incar_syntax(content or "")
    return []


def apply_input_syntax_fixes(
    input_files: Dict[str, str],
    issues: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Engine-neutral pre-submit auto-fix entry point for VASP inputs.

    The fix-side counterpart to :func:`check_input_syntax`: accepts the
    full input-files mapping, applies high-confidence tag renames to the
    file this engine knows how to fix (the INCAR), and returns the
    updated mapping. The conventional auto-fix tool a generator invokes
    to clean obvious typos before handing inputs off.

    Args:
        input_files: Mapping of input filename to file contents.
        issues: Issues from a prior :func:`check_input_syntax` call. When
            omitted, the check is run inline on the INCAR.

    Returns:
        A ``(fixed_files, applied)`` tuple. ``fixed_files`` is a shallow
        copy of ``input_files`` with the INCAR entry replaced by its
        fixed text (unchanged when no INCAR is present or no fix fires).
        ``applied`` is the list of renames that fired, each with
        ``renamed_from`` / ``renamed_to`` keys.
    """
    fixed_files = dict(input_files)
    for name, content in input_files.items():
        if name.upper() == "INCAR" or name.upper().endswith("/INCAR"):
            fixed_text, applied = apply_incar_syntax_fixes(content or "", issues)
            fixed_files[name] = fixed_text
            return fixed_files, applied
    return fixed_files, []


TOOL_SPEC_INPUT_SYNTAX = ToolSpec(
    name="check_input_syntax",
    description=(
        "Engine-neutral pre-run syntax check for VASP inputs: selects the "
        "INCAR from the input-files mapping and deterministically flags "
        "unrecognized tags. The conventional pre-run tool an input "
        "reviewer invokes; no LLM call."
    ),
    parameters={
        "input_files": {
            "type": "object",
            "description": (
                "Mapping of input filename to contents; the INCAR entry "
                "is selected and checked."
            ),
        },
    },
    required=["input_files"],
    signature="check_input_syntax(input_files: dict) -> list[dict]",
    import_line="from scilink.skills.periodic_dft.vasp.vasp_syntax import check_input_syntax",
    agents=["simulation"],
    returns="list of INCAR syntax issue dicts (empty when no INCAR present).",
)

TOOL_SPEC_CHECK = ToolSpec(
    name="check_incar_syntax",
    description=(
        "Deterministically check a VASP INCAR for unrecognized tags "
        "(typically typos) against pymatgen's canonical tag list. Returns "
        "issue dicts with closest-match suggestions and a high/low "
        "confidence rating. No LLM call."
    ),
    parameters={
        "incar_content": {
            "type": "string",
            "description": "Raw INCAR text (file contents, not a path).",
        },
    },
    required=["incar_content"],
    signature="check_incar_syntax(incar_content: str) -> list[dict]",
    import_line="from scilink.skills.periodic_dft.vasp.vasp_syntax import check_incar_syntax",
    agents=["simulation"],
    returns=(
        "list of issue dicts: severity, category, tag, suggested, "
        "confidence, description, source. Empty when no unrecognized tags."
    ),
)

TOOL_SPEC_FIX = ToolSpec(
    name="apply_incar_syntax_fixes",
    description=(
        "Apply high-confidence tag renames in place to a VASP INCAR "
        "string (e.g. ISPN -> ISPIN). Low-confidence issues are left "
        "untouched for LLM review. No LLM call."
    ),
    parameters={
        "incar_content": {
            "type": "string",
            "description": "Raw INCAR text to fix.",
        },
        "issues": {
            "type": "array",
            "description": (
                "Optional issues from a prior check_incar_syntax call. "
                "When omitted, the check is run inline."
            ),
        },
    },
    required=["incar_content"],
    signature=(
        "apply_incar_syntax_fixes(incar_content: str, issues: list | None) "
        "-> tuple[str, list[dict]]"
    ),
    import_line="from scilink.skills.periodic_dft.vasp.vasp_syntax import apply_incar_syntax_fixes",
    agents=["simulation"],
    returns="(fixed_incar_text, applied_renames) tuple.",
)

TOOL_SPEC_INPUT_FIX = ToolSpec(
    name="apply_input_syntax_fixes",
    description=(
        "Engine-neutral pre-submit auto-fix for VASP inputs: applies "
        "high-confidence tag renames to the INCAR in the input-files "
        "mapping and returns the updated mapping. The conventional "
        "auto-fix tool a generator invokes; no LLM call."
    ),
    parameters={
        "input_files": {
            "type": "object",
            "description": "Mapping of input filename to contents.",
        },
        "issues": {
            "type": "array",
            "description": (
                "Optional issues from a prior check_input_syntax call. "
                "When omitted, the check is run inline."
            ),
        },
    },
    required=["input_files"],
    signature=(
        "apply_input_syntax_fixes(input_files: dict, issues: list | None) "
        "-> tuple[dict, list[dict]]"
    ),
    import_line="from scilink.skills.periodic_dft.vasp.vasp_syntax import apply_input_syntax_fixes",
    agents=["simulation"],
    returns="(fixed_input_files, applied_renames) tuple.",
)

TOOL_SPECS = [
    TOOL_SPEC_INPUT_SYNTAX,
    TOOL_SPEC_INPUT_FIX,
    TOOL_SPEC_CHECK,
    TOOL_SPEC_FIX,
]
