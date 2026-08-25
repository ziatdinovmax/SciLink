"""Backend dispatch for structure-database queries.

This subpackage is private to the ``structure_matching`` skill domain. It is
not visible to the skill discovery walker (the leading underscore in
``_backends`` excludes it; see ``scilink/skills/loader.py:117-118``).

Backends implement the :class:`StructureBackend` protocol declared in
``_base``. Skill tools (e.g. ``xrd/search_structures.py``) dispatch over the
registered backend list, filtered by ``is_available()``.

Adding a new backend
====================

The protocol is small (``is_available()`` + ``query(spec) -> list[Candidate]``).
There are three ways to wire a new backend into the dispatcher without
forking SciLink:

1. **Imperative** — call :func:`register_backend` from your code before
   invoking the agent::

       from scilink.skills.structure_matching._backends import register_backend
       from my_package.icsd_backend import ICSDBackend
       register_backend("icsd", ICSDBackend)

       agent.analyze(..., skill="xrd")
       # Now {"sources": ["icsd"]} is a valid query, and the auto-detect
       # dispatch picks up ICSDBackend whenever is_available() is True.

2. **Entry point** — declare the backend in your package's
   ``pyproject.toml``; SciLink discovers it on first import::

       [project.entry-points."scilink.structure_backends"]
       icsd = "my_package.icsd_backend:ICSDBackend"
       oqmd = "my_package.oqmd_backend:OQMDBackend"

3. **Built-in** — add to :data:`_BUILTIN_BACKENDS` in this file. Reserved
   for backends shipping with SciLink itself.

Builtins are loaded first so user-registered backends with the same
name can override.
"""

from __future__ import annotations

import logging
from typing import Callable

from ._base import QuerySpec, StructureBackend, StructureCandidate
from .cod import CODBackend
from .local_cif import LocalCIFBackend
from .materials_project import MaterialsProjectBackend

_logger = logging.getLogger(__name__)


# Public type alias: a callable returning a freshly-constructed backend.
BackendFactory = Callable[[], StructureBackend]


# Built-in backends ship with SciLink. User-registered backends are added
# to ``_REGISTRY`` separately and can shadow built-ins by name.
_BUILTIN_BACKENDS: dict[str, BackendFactory] = {
    "mp": MaterialsProjectBackend,
    "local": LocalCIFBackend,
    "cod": CODBackend,
}

_REGISTRY: dict[str, BackendFactory] = dict(_BUILTIN_BACKENDS)


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register a structure-database backend under ``name``.

    The factory is a zero-argument callable returning a fresh
    :class:`StructureBackend` instance (typically just the class itself,
    since the protocol's ``__init__`` is parameterless). Re-registering
    an existing name silently overrides the previous binding — this is
    intentional so users can shadow built-ins (e.g. a local mirror of
    Materials Project that doesn't hit the network).

    Raises:
        ValueError: when ``name`` is empty or already taken by a built-in
            and the override is rejected by the calling environment.
            (No environment currently rejects overrides; placeholder for
            future restriction.)
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Backend name must be a non-empty string; got {name!r}")
    if name in _REGISTRY and _REGISTRY[name] is not factory:
        _logger.info("Backend %s re-registered (previous factory shadowed)", name)
    _REGISTRY[name] = factory


def unregister_backend(name: str) -> None:
    """Remove a previously-registered backend.

    Built-ins can be unregistered; call ``_reload_builtins()`` to restore
    them if needed. Tests use this to keep the registry clean.
    """
    _REGISTRY.pop(name, None)


def registered_backends() -> dict[str, BackendFactory]:
    """Return a shallow copy of the current backend registry."""
    return dict(_REGISTRY)


def get_backend_factory(name: str) -> BackendFactory | None:
    """Look up a backend factory by name; returns ``None`` when unknown."""
    return _REGISTRY.get(name)


def _reload_builtins() -> None:
    """Restore the built-in backends. Used by tests to isolate state."""
    for name, factory in _BUILTIN_BACKENDS.items():
        _REGISTRY[name] = factory


def _load_entry_points() -> None:
    """Discover backends declared via the ``scilink.structure_backends`` entry-point group.

    A third-party package can register a backend without code changes
    inside SciLink by declaring it in its ``pyproject.toml``::

        [project.entry-points."scilink.structure_backends"]
        icsd = "my_package.icsd_backend:ICSDBackend"

    Failures (missing module, malformed entry-point target, etc.) are
    logged and skipped — one broken entry must not break discovery for
    every other backend.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # Python < 3.10 (unlikely, SciLink requires 3.12+)
        return

    try:
        # Python 3.10+ accepts group= kwarg.
        eps = entry_points(group="scilink.structure_backends")
    except TypeError:
        # Older importlib.metadata: returns a dict keyed by group.
        eps = entry_points().get("scilink.structure_backends", [])
    except Exception as e:
        _logger.debug("entry_points() failed for scilink.structure_backends: %s", e)
        return

    for ep in eps:
        try:
            factory = ep.load()
        except Exception as e:
            _logger.warning(
                "Failed to load structure-backend entry-point %s = %s: %s",
                ep.name, getattr(ep, "value", "?"), e,
            )
            continue
        register_backend(ep.name, factory)


# Discover external backends on import — once per process. The user can
# call register_backend() again later if they want to add more.
_load_entry_points()


__all__ = [
    "StructureBackend",
    "StructureCandidate",
    "QuerySpec",
    "BackendFactory",
    "MaterialsProjectBackend",
    "LocalCIFBackend",
    "CODBackend",
    "register_backend",
    "unregister_backend",
    "registered_backends",
    "get_backend_factory",
]
