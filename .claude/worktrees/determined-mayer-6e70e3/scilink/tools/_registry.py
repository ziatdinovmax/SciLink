"""Auto-discovery registry for ``scilink.tools`` modules.

Walks ``scilink/tools/`` once, collects ``TOOL_SPEC`` (single) or
``TOOL_SPECS`` (list) attributes from each public module, filters by agent
tag, and caches the result.

Adding a new tool:
    1. Write the function in a module under ``scilink/tools/``.
    2. Add ``TOOL_SPEC = ToolSpec(...)`` (or append to ``TOOL_SPECS``) with
       ``agents=["image_analysis", ...]``.
    3. Done — no central list to edit.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from functools import lru_cache

from ._spec import ToolSpec

_logger = logging.getLogger(__name__)


# Libraries available in the execution sandbox (verified against the host
# Python environment, which the sandbox shares via subprocess.Popen in
# scilink.executors).
IMAGE_ANALYSIS_LIBRARIES: list[tuple[str, str]] = [
    ("numpy", "array ops, FFT, linear algebra"),
    ("scipy", "signal / image processing (ndimage), optimize, stats, spatial"),
    ("skimage", "segmentation, morphology, feature, measure, filters, transform"),
    ("cv2", "OpenCV — thresholding, contours, morphology, template matching"),
    ("sklearn", "clustering, PCA, classifiers for pixel / region classification"),
    ("matplotlib", "plotting"),
    ("pandas", "tabular output (particle lists, atom position tables)"),
]


def format_library_inventory() -> str:
    """Render ``IMAGE_ANALYSIS_LIBRARIES`` as a compact markdown list."""
    lines = [f"- `{name}` — {desc}" for name, desc in IMAGE_ANALYSIS_LIBRARIES]
    return "\n".join(lines)


def format_tool_inventory(agent: str = "image_analysis") -> str:
    """Render the full tool + library inventory for ``agent`` as one string.

    Used by prompts that need a drop-in ``{tool_inventory}`` substitution
    (code-gen, script-correction) — complements ``_append_tool_inventory``
    which is list-based for multi-part prompts. Both draw from the same
    registry so there is a single source of truth.
    """
    parts: list[str] = []
    specs = get_tools_for(agent)
    if specs:
        parts.append("## Available Tools")
        parts.append(
            "The following tools are registered and callable from generated scripts. "
            "Prefer a tool when it fits; combine with custom numpy/scipy/skimage/cv2 "
            "code for post-processing. A tool call anchoring the hard step followed "
            "by custom code is usually more reliable than an all-custom pipeline."
        )
        for spec in specs:
            parts.append(spec.to_prompt())

    parts.append("## Available Libraries")
    parts.append(
        "These libraries are importable in the execution sandbox. Use them for "
        "custom code when no registered tool fits."
    )
    parts.append(format_library_inventory())

    return "\n\n".join(parts)


def _collect_specs_from_module(mod) -> list[ToolSpec]:
    """Return ToolSpec list declared by a module (supports TOOL_SPEC and TOOL_SPECS)."""
    specs: list[ToolSpec] = []
    single = getattr(mod, "TOOL_SPEC", None)
    if isinstance(single, ToolSpec):
        specs.append(single)
    multi = getattr(mod, "TOOL_SPECS", None)
    if isinstance(multi, (list, tuple)):
        specs.extend(s for s in multi if isinstance(s, ToolSpec))
    return specs


@lru_cache(maxsize=None)
def _all_specs() -> tuple[ToolSpec, ...]:
    """Walk scilink.tools once and collect every declared ToolSpec.

    Cached for the life of the process. Module import failures are logged
    and skipped — an optional heavy dependency should not prevent discovery
    of the other tools.
    """
    from . import __path__ as _tools_path  # local import to avoid cycles

    collected: list[ToolSpec] = []
    for info in pkgutil.iter_modules(_tools_path):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"scilink.tools.{info.name}")
        except Exception as exc:
            _logger.debug("Skipping scilink.tools.%s: %s", info.name, exc)
            continue
        collected.extend(_collect_specs_from_module(mod))
    return tuple(collected)


def get_tools_for(agent: str) -> list[ToolSpec]:
    """Return every registered ToolSpec tagged for ``agent``."""
    return [spec for spec in _all_specs() if agent in spec.agents]
