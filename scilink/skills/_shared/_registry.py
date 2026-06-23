"""Auto-discovery registry for skill helper modules.

Walks ``scilink/skills/_shared/`` and every per-skill bundle once, collects
``TOOL_SPEC`` (single) or ``TOOL_SPECS`` (list) attributes from each public
module, filters by agent tag, and caches the result.

Adding a new tool:
    1. Write the function as a module either under ``scilink/skills/_shared/``
       (cross-skill helper) or inside the relevant ``scilink/skills/<domain>/<name>/``
       skill bundle (skill-specific helper).
    2. Add ``TOOL_SPEC = ToolSpec(...)`` (or append to ``TOOL_SPECS``) with
       ``agents=["image_analysis", ...]``.
    3. Done — no central list to edit.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from functools import lru_cache
from typing import Any, Callable

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


def format_tool_inventory(
    agent: str = "image_analysis",
    active_skills: list[str] | None = None,
) -> str:
    """Render the full tool + library inventory for ``agent`` as one string.

    Used by prompts that need a drop-in ``{tool_inventory}`` substitution
    (code-gen, script-correction) — complements ``_append_tool_inventory``
    which is list-based for multi-part prompts. Both draw from the same
    registry so there is a single source of truth.

    ``active_skills`` controls which per-skill bundle tools are visible;
    see :func:`get_tools_for`.
    """
    parts: list[str] = []
    specs = get_tools_for(agent, active_skills=active_skills)
    if specs:
        parts.append("## Available Tools")
        parts.append(
            "The following tools are registered and callable from generated scripts. "
            "Prefer a tool when it fits; combine with custom numpy/scipy/skimage/cv2 "
            "code for post-processing. A tool call anchoring the hard step followed "
            "by custom code is usually more reliable than an all-custom pipeline."
        )
        parts.append(
            "USE WHAT A TOOL RETURNS — DO NOT RE-DERIVE IT. Each tool returns its "
            "computed results in the result dict (see its **Returns**) and often a "
            "finished figure. Read those returned fields, and save the tool's figure "
            "as the visualization when it has one. Do NOT recompute, re-segment, "
            "re-render, rescale, or re-bin a quantity a tool you called already "
            "returns: re-deriving a tool's own output is the most common source of "
            "units / scaling / normalization bugs (e.g. recomputing a distance the "
            "tool already reports in nm, or re-clustering domains the tool already "
            "labelled). Post-process ONLY what no tool provides, and operate on the "
            "tool's returned values rather than reimplementing the tool."
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
def _shared_specs() -> tuple[ToolSpec, ...]:
    """Specs declared by modules under ``scilink/skills/_shared/``.

    These are cross-skill helpers; visibility is gated by the ``agents=`` tag.
    Cached for the life of the process. Module import failures are logged
    and skipped — an optional heavy dependency should not prevent discovery
    of the other tools.
    """
    from . import __path__ as _shared_path  # scilink/skills/_shared/

    collected: list[ToolSpec] = []
    for info in pkgutil.iter_modules(_shared_path):
        if info.name.startswith("_"):
            continue
        module_path = f"scilink.skills._shared.{info.name}"
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            _logger.debug("Skipping %s: %s", module_path, exc)
            continue
        collected.extend(_collect_specs_from_module(mod))
    return tuple(collected)


@lru_cache(maxsize=None)
def _per_skill_specs() -> dict[tuple[str, str], tuple[ToolSpec, ...]]:
    """Specs declared inside skill bundles, keyed by ``(domain, skill_name)``.

    Visibility is gated by *bundle membership*, not the ``agents=`` tag —
    a tool living inside a skill folder is implicitly scoped to that skill.
    """
    from scilink.skills.loader import list_all_skills, _SKILLS_DIR

    result: dict[tuple[str, str], list[ToolSpec]] = {}
    for domain, names in list_all_skills().items():
        for name in names:
            skill_dir = _SKILLS_DIR / domain / name
            specs: list[ToolSpec] = []
            for py_file in sorted(skill_dir.glob("*.py")):
                if py_file.stem.startswith("_"):
                    continue
                module_path = f"scilink.skills.{domain}.{name}.{py_file.stem}"
                try:
                    mod = importlib.import_module(module_path)
                except Exception as exc:
                    _logger.debug("Skipping %s: %s", module_path, exc)
                    continue
                specs.extend(_collect_specs_from_module(mod))
            if specs:
                result[(domain, name)] = specs
    return {k: tuple(v) for k, v in result.items()}


def _all_specs() -> tuple[ToolSpec, ...]:
    """Every declared ToolSpec, shared + per-skill. Used by tests / introspection."""
    out: list[ToolSpec] = list(_shared_specs())
    for specs in _per_skill_specs().values():
        out.extend(specs)
    return tuple(out)


def get_tools_for(agent: str, active_skills: list[str] | None = None) -> list[ToolSpec]:
    """Return ToolSpecs visible to ``agent`` given the currently-active skills.

    Visibility rules:
      * Specs from ``_shared/`` are always considered, filtered by the spec's
        ``agents=`` tag.
      * Specs from a skill bundle are considered only when the skill's name
        appears in ``active_skills``. The bundle's location implies scope —
        the spec's ``agents=`` field is ignored for per-skill specs.

    Passing ``active_skills=None`` returns only shared specs (no skill is
    loaded). Pass an empty list for the same behavior.
    """
    visible = [s for s in _shared_specs() if agent in s.agents]
    if active_skills:
        active_set = set(active_skills)
        for (_domain, name), specs in _per_skill_specs().items():
            if name in active_set:
                visible.extend(specs)
    return visible


@lru_cache(maxsize=None)
def _per_skill_callables() -> dict[tuple[str, str], dict[str, Callable[..., Any]]]:
    """Map ``(domain, skill_name)`` to ``{spec.name: callable}``.

    Walks the same skill bundles as :func:`_per_skill_specs`. For each
    discovered ``TOOL_SPEC``, looks up the matching callable in the
    declaring module by name: a spec with ``name="snapshot_run"`` is
    paired with a function named ``snapshot_run`` in the same module.
    Specs whose corresponding function cannot be located are logged
    and skipped.

    Cached for the life of the process; the skill tree is static within
    a run.
    """
    from scilink.skills.loader import list_all_skills, _SKILLS_DIR

    result: dict[tuple[str, str], dict[str, Callable[..., Any]]] = {}
    for domain, names in list_all_skills().items():
        for name in names:
            skill_dir = _SKILLS_DIR / domain / name
            callables: dict[str, Callable[..., Any]] = {}
            for py_file in sorted(skill_dir.glob("*.py")):
                if py_file.stem.startswith("_"):
                    continue
                module_path = f"scilink.skills.{domain}.{name}.{py_file.stem}"
                try:
                    mod = importlib.import_module(module_path)
                except Exception as exc:
                    _logger.debug("Skipping %s: %s", module_path, exc)
                    continue
                for spec in _collect_specs_from_module(mod):
                    fn = getattr(mod, spec.name, None)
                    if callable(fn):
                        callables[spec.name] = fn
                    else:
                        _logger.warning(
                            "TOOL_SPEC '%s' declared in %s but no "
                            "callable with that name found; skipping.",
                            spec.name, module_path,
                        )
            if callables:
                result[(domain, name)] = callables
    return result


def get_tool_function(
    name: str,
    active_skills: list[str] | None = None,
) -> Callable[..., Any]:
    """Resolve a tool name to its callable from the active skill bundles.

    Looks up ``name`` in every skill bundle listed in ``active_skills``
    and returns the first matching callable. Used by Python code that
    needs to invoke a skill-registered tool directly (as opposed to
    surfacing its spec to an LLM via :func:`get_tools_for`).

    Args:
        name: The ``TOOL_SPEC.name`` to resolve.
        active_skills: Skill names whose bundles should be searched.
            ``None`` or an empty list raises ``LookupError`` — at least
            one skill must be specified.

    Returns:
        The callable registered for ``name`` in one of the active skill
        bundles. The callable accepts whatever ``**kwargs`` its
        ``TOOL_SPEC.parameters`` describe; callers may invoke it as
        ``get_tool_function(name, active_skills=[...])(**kwargs)``.

    Raises:
        LookupError: If no matching callable is found in any active
            skill bundle. The message names the searched skills so the
            caller can verify the bundle declares the expected spec.
    """
    if not active_skills:
        raise LookupError(
            f"Cannot resolve tool {name!r}: no active skills supplied. "
            "Pass active_skills=[<skill_name>, ...] to enable lookup."
        )
    active_set = set(active_skills)
    for (_domain, skill_name), callables_map in _per_skill_callables().items():
        if skill_name in active_set and name in callables_map:
            return callables_map[name]
    raise LookupError(
        f"Tool {name!r} not found in any of the active skill bundles "
        f"{sorted(active_set)!r}. Verify that the matching bundle "
        f"declares a TOOL_SPEC with name={name!r} and that a callable "
        "with the same name exists in the spec's declaring module."
    )
