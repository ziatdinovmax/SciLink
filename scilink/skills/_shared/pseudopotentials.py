"""Engine-neutral pseudopotential selection (issue #786).

The LLM makes the scientific choice (which variant per element, and why);
code grounds it in what the local library actually holds, resolves it against
the user's overrides and the skill's recommended defaults, and records where
each choice came from. Engine specifics (library layout, file format, how the
choice is materialized) live in the engine's skill bundle, which exposes:

    default_pseudopotentials(elements)            -> {element: variant}
    list_pseudopotentials(elements, library=None) -> {element: [{variant, ...}]} | None
    render_pseudopotential_prep(choice, species)  -> {filename: contents}
    pseudopotential_set(input_files)              -> set key for list_pseudopotentials | None
    describe_pseudopotentials(run_dir)            -> {element: {variant, ...}}

An engine opts in by declaring a ``pseudopotentials:`` frontmatter block in its
skill markdown; engines without it are unaffected.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


def element_of(variant: str) -> Optional[str]:
    """Element symbol a variant name belongs to (``"Na_pv"`` -> ``"Na"``,
    ``"H.75"`` -> ``"H"``), or ``None`` if the name does not start with one."""
    m = _ELEMENT_RE.match(variant or "")
    return m.group(1) if m else None


def resolve_pseudopotentials(
    elements: Iterable[str],
    defaults: Dict[str, str],
    agent_choice: Optional[Dict[str, Any]] = None,
    user_map: Optional[Dict[str, str]] = None,
    inventory: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], List[str]]:
    """Pick one variant per element: user map > agent choice > skill default.

    Args:
        elements: Element symbols present in the structure.
        defaults: The skill's recommended ``{element: variant}`` table.
        agent_choice: The LLM's deviations, ``{element: variant}`` or
            ``{element: {"variant": ..., "reason": ...}}``.
        user_map: Caller overrides ``{element: variant}``; always win.
        inventory: ``{element: [available variant names]}`` from the local
            library, or ``None`` when the library is not readable here (the
            choice is then taken on trust and checked when the run starts).

    Returns:
        ``(choice, warnings)`` where ``choice`` maps each element to
        ``{"variant", "source", "reason"}`` (``source`` is ``user``, ``agent``,
        ``default`` or ``fallback``) and ``warnings`` lists every substitution
        made because a requested variant was unusable.
    """
    agent_choice = agent_choice or {}
    user_map = user_map or {}
    choice: Dict[str, Dict[str, Optional[str]]] = {}
    warnings: List[str] = []

    def available(el: str) -> Optional[List[str]]:
        return None if inventory is None else list(inventory.get(el) or [])

    for el in dict.fromkeys(elements):
        avail = available(el)

        if el in user_map:
            v = user_map[el]
            if not isinstance(v, str) or element_of(v) != el:
                warnings.append(f"{el}: user-requested {v!r} is not a variant of {el}; ignored")
            else:
                if avail is not None and v not in avail:
                    warnings.append(f"{el}: user-requested {v} is not in the library")
                choice[el] = {"variant": v, "source": "user", "reason": None}
                continue

        raw = agent_choice.get(el)
        if raw:
            v = raw.get("variant") if isinstance(raw, dict) else raw
            reason = raw.get("reason") if isinstance(raw, dict) else None
            if not isinstance(v, str) or element_of(v) != el:
                warnings.append(f"{el}: agent choice {v!r} is not a variant of {el}; using default")
            elif avail is not None and v not in avail:
                warnings.append(f"{el}: agent choice {v} is not in the library; using default")
            else:
                choice[el] = {"variant": v, "source": "agent", "reason": reason}
                continue

        d = defaults.get(el, el)
        if avail is None or d in avail:
            choice[el] = {"variant": d, "source": "default", "reason": None}
        elif el in avail:
            warnings.append(f"{el}: default {d} is not in the library; using {el}")
            choice[el] = {"variant": el, "source": "fallback", "reason": None}
        elif avail:
            warnings.append(f"{el}: neither {d} nor {el} is in the library; using {avail[0]}")
            choice[el] = {"variant": avail[0], "source": "fallback", "reason": None}
        else:
            warnings.append(f"{el}: no pseudopotential for {el} in the library")
            choice[el] = {"variant": d, "source": "default", "reason": None}

    return choice, warnings
