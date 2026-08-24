"""LLM-friendly forwarding shim for skill-bundled tools.

Live testing with claude-opus-4-6 showed the model routinely synthesizes
``from scilink.agents.skills.<...> import ...`` imports even when the
canonical path is ``scilink.skills.<...>`` — a natural hallucination
given the model's "skills live under the agent namespace" intuition.
Prompt hardening empirically didn't fix it (the LLM accepts the
correction in principle, then re-hallucinates with a slightly different
sub-path on the next turn).

This shim eagerly walks the ``scilink.skills`` package tree at import
time and registers every submodule under the equivalent
``scilink.agents.skills.*`` alias in ``sys.modules``. Both paths
resolve to the same Python objects — verified by ``is`` comparison in
tests. New skills picked up automatically on next Python process start.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import scilink.skills as _root


_PREFIX = "scilink.agents.skills."
_ROOT_PKG = "scilink.skills"


def _walk_and_alias() -> None:
    # Alias the root itself first so 'scilink.agents.skills' is the same
    # module as 'scilink.skills'.
    sys.modules.setdefault("scilink.agents.skills", _root)

    for module_info in pkgutil.walk_packages(_root.__path__, prefix=_ROOT_PKG + "."):
        canonical = module_info.name
        try:
            mod = importlib.import_module(canonical)
        except ImportError:
            # Optional heavy dependencies (e.g. pymatgen analysis modules
            # not installed) — skip the alias quietly; the canonical path
            # still works when the dep is present.
            continue
        alias = _PREFIX + canonical[len(_ROOT_PKG) + 1:]
        sys.modules.setdefault(alias, mod)

    # Skip-the-domain aliases: ``scilink.agents.skills.<skill>`` →
    # ``scilink.skills.<domain>.<skill>``. The LLM frequently elides the
    # domain layer (e.g. it tries ``scilink.agents.skills.xrd.<tool>``
    # rather than ``scilink.agents.skills.structure_matching.xrd.<tool>``).
    # Iterate built-in domains and register each skill name as a short
    # alias plus the alias for each of its modules.
    try:
        from scilink.skills.loader import list_all_skills
    except ImportError:
        return
    for domain, skill_names in list_all_skills().items():
        for skill in skill_names:
            canonical_skill = f"{_ROOT_PKG}.{domain}.{skill}"
            short_alias = _PREFIX + skill
            if canonical_skill in sys.modules:
                sys.modules.setdefault(short_alias, sys.modules[canonical_skill])
            # Each sub-module of the skill (tools): also aliased at the short path.
            for name, mod in list(sys.modules.items()):
                if name.startswith(canonical_skill + "."):
                    suffix = name[len(canonical_skill):]
                    sys.modules.setdefault(short_alias + suffix, mod)


_walk_and_alias()
