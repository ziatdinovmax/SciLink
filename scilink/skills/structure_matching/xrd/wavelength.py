"""``resolve_wavelength`` tool — pick an XRD source wavelength from metadata.

Looks at structured `experiment.*` fields first, then falls back to a
free-text scan for canonical source names. Returns either a named source
string (passable to ``simulate_xrd_pattern`` as ``wavelength``) or a
float wavelength in angstroms.

Eliminates the silent-CuKa-default footgun: when metadata clearly names
a non-Cu source (e.g. MoKa, CoKa), the LLM script should call this
function instead of hard-coding ``wavelength='CuKa'``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Union

from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)


# Recognized named sources. Keys are the canonical names accepted by
# pymatgen's XRDCalculator; values are aliases that may appear in free text.
_NAMED_SOURCES: dict[str, tuple[str, ...]] = {
    "CuKa": ("cuka", "cu-ka", "cu ka", "cu kα", "copper kα", "copper k-alpha"),
    "CuKa1": ("cuka1", "cu-ka1", "cu kα1", "copper kα1"),
    "MoKa": ("moka", "mo-ka", "mo ka", "mo kα", "molybdenum kα"),
    "CoKa": ("coka", "co-ka", "co ka", "co kα", "cobalt kα"),
    "FeKa": ("feka", "fe-ka", "fe ka", "fe kα", "iron kα"),
    "CrKa": ("crka", "cr-ka", "cr ka", "cr kα", "chromium kα"),
    "AgKa": ("agka", "ag-ka", "ag ka", "ag kα", "silver kα"),
}


TOOL_SPEC = ToolSpec(
    name="resolve_wavelength",
    description=(
        "Pick an XRD source wavelength from experiment metadata. Reads "
        "structured fields (experiment.wavelength, experiment.source, "
        "experiment.x_ray_source) and falls back to a free-text scan "
        "for canonical source names (CuKa, MoKa, CoKa, FeKa, CrKa, "
        "AgKa, CuKa1). Returns the resolved wavelength as a string "
        "(named source) or float (angstroms)."
    ),
    import_line="from scilink.skills.structure_matching.xrd.wavelength import resolve_wavelength",
    signature=(
        "resolve_wavelength(system_info: dict | str | None, "
        "default: str | float = 'CuKa') -> str | float"
    ),
    parameters={
        "system_info": {
            "type": "dict | str | None",
            "description": (
                "System info dict (with 'experiment' subkey) or a free-text "
                "metadata string. Pass None to get the default."
            ),
        },
        "default": {
            "type": "str | float",
            "description": "Fallback when no source can be inferred. Default 'CuKa'.",
        },
    },
    required=["system_info"],
    returns=(
        "str (named source, e.g. 'MoKa') or float (wavelength in angstroms). "
        "Pass directly to simulate_xrd_pattern's wavelength parameter."
    ),
    when_to_use=(
        "Before calling simulate_xrd_pattern, to avoid silently defaulting "
        "to CuKa on a Mo/Co/Cr-source measurement. The wrong wavelength "
        "gives uniformly bad scores for every candidate."
    ),
)


def resolve_wavelength(
    system_info: Union[dict, str, None],
    default: Union[str, float] = "CuKa",
) -> Union[str, float]:
    """Resolve an XRD source wavelength from metadata. See ``TOOL_SPEC``."""
    if system_info is None:
        return default

    # 1. Structured fields take priority.
    if isinstance(system_info, dict):
        experiment = system_info.get("experiment")
        if isinstance(experiment, dict):
            for key in ("wavelength", "x_ray_wavelength", "source", "x_ray_source"):
                value = experiment.get(key)
                resolved = _coerce_field(value)
                if resolved is not None:
                    return resolved

        # Fall back to top-level wavelength keys (some converters flatten).
        for key in ("wavelength", "x_ray_wavelength"):
            resolved = _coerce_field(system_info.get(key))
            if resolved is not None:
                return resolved

        # 2. Free-text scan across all string-valued metadata.
        text_blobs = _collect_text(system_info)
    else:
        text_blobs = [str(system_info)]

    for blob in text_blobs:
        named = _scan_text_for_named_source(blob)
        if named is not None:
            return named
        angstroms = _scan_text_for_angstroms(blob)
        if angstroms is not None:
            return angstroms

    return default


def _coerce_field(value: Any) -> Union[str, float, None]:
    """Convert a metadata field to a wavelength string or float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if 0.1 < v < 5.0:
            return v
        return None
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        # Exact named-source match (case insensitive, hyphen / space tolerant).
        named = _match_named_source(v)
        if named is not None:
            return named
        # Numeric string in angstroms.
        try:
            num = float(v)
            if 0.1 < num < 5.0:
                return num
        except ValueError:
            pass
        # Substring scan for the field's own text content.
        named = _scan_text_for_named_source(v)
        if named is not None:
            return named
        angstroms = _scan_text_for_angstroms(v)
        if angstroms is not None:
            return angstroms
    return None


def _match_named_source(text: str) -> Union[str, None]:
    """Exact match (after normalization) against canonical source names."""
    norm = _normalize(text)
    for canonical, aliases in _NAMED_SOURCES.items():
        if norm == _normalize(canonical):
            return canonical
        for alias in aliases:
            if norm == _normalize(alias):
                return canonical
    return None


def _scan_text_for_named_source(text: str) -> Union[str, None]:
    """Substring scan for canonical source names. Most-specific match wins."""
    norm = _normalize(text)
    best: tuple[str, int] | None = None  # (canonical, alias_length)
    for canonical, aliases in _NAMED_SOURCES.items():
        for alias in (canonical, *aliases):
            alias_norm = _normalize(alias)
            if alias_norm and alias_norm in norm:
                if best is None or len(alias_norm) > best[1]:
                    best = (canonical, len(alias_norm))
    return best[0] if best else None


_ANGSTROM_RE = re.compile(
    r"(?P<value>\d+\.\d+)\s*(?:[åÅ]|angstrom|A\b)",
    re.IGNORECASE,
)


def _scan_text_for_angstroms(text: str) -> Union[float, None]:
    """Pull a 'N.NNN Å' / 'N.NNN angstrom' style number from free text."""
    match = _ANGSTROM_RE.search(text)
    if not match:
        return None
    try:
        v = float(match.group("value"))
    except ValueError:
        return None
    if 0.1 < v < 5.0:
        return v
    return None


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _collect_text(d: Any, _seen: set | None = None) -> list[str]:
    """Recursively collect all string values from a dict-like metadata blob."""
    if _seen is None:
        _seen = set()
    if id(d) in _seen:
        return []
    _seen.add(id(d))

    blobs: list[str] = []
    if isinstance(d, str):
        blobs.append(d)
    elif isinstance(d, dict):
        for v in d.values():
            blobs.extend(_collect_text(v, _seen))
    elif isinstance(d, (list, tuple)):
        for item in d:
            blobs.extend(_collect_text(item, _seen))
    return blobs
