"""Where an agent may read and write on a hosted server.

On a laptop a user names files anywhere on the machine and that is right. On a
server one process serves one workspace, and a path the model or a client
names must stay inside it: the session directory, the mode's own directories
(uploads, knowledge, code, data), the persistent store (``SCILINK_HOME``), the
model cache and the temp dir. A :class:`PathFence` holds those roots and says
yes or no; it is OFF (``None``) unless roots were given or
``SCILINK_FILE_ROOTS`` is set, so nothing changes on a laptop.

Two layers, because there are four tool dispatchers and dozens of tools:

1. **At dispatch**, by parameter name (:meth:`PathFence.refuse_tool_args`): an
   absolute path, a ``..`` escape or a glob whose fixed prefix leaves the
   roots is refused before the tool runs, with a message naming the roots.
2. **Inside the resolvers that can wander** (a relative name tried against
   the process cwd, a bare-name search, a glob), the fence bound on the
   thread for the duration of the call (:func:`bound` / :func:`current`)
   is consulted, so a path that only becomes absolute inside a helper is
   still held.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_GLOB_CHARS = "*?["


class PathFenceError(PermissionError):
    """A path outside the workspace's roots."""


class PathFence:
    def __init__(self, roots: Iterable[Any]) -> None:
        seen: List[Path] = []
        for r in roots:
            if not r:
                continue
            try:
                p = Path(str(r)).expanduser().resolve()
            except (OSError, ValueError):
                continue
            if p not in seen:
                seen.append(p)
        self.roots: List[Path] = seen

    # -- construction ---------------------------------------------------
    @classmethod
    def build(cls, base_dir, roots: Optional[Sequence[Any]] = None,
              extra: Sequence[Any] = ()) -> Optional["PathFence"]:
        """The fence for an orchestrator, or ``None`` (open) when neither
        explicit ``roots`` nor ``SCILINK_FILE_ROOTS`` are given. The base dir,
        the mode's own dirs, the persistent store, the model cache, the
        package's own skills and the temp dir are always allowed."""
        env = os.environ.get("SCILINK_FILE_ROOTS", "")
        if roots is None and not env:
            return None
        from scilink.skills.loader import _SKILLS_DIR, scilink_home
        home = scilink_home()
        models = os.environ.get("SCILINK_MODELS") or (home / "models")
        return cls([*(roots or []), *env.split(os.pathsep), base_dir, *extra,
                    home, models, _SKILLS_DIR, tempfile.gettempdir()])

    # -- questions ------------------------------------------------------
    def allows(self, path) -> bool:
        try:
            p = Path(str(path)).expanduser().resolve()
        except (OSError, ValueError):
            return False
        return any(p == r or p.is_relative_to(r) for r in self.roots)

    def check(self, path, what: str = "path") -> Path:
        if not self.allows(path):
            raise PathFenceError(self.message(path, what))
        return Path(str(path)).expanduser().resolve()

    def check_pattern(self, pattern: str) -> None:
        """A glob is judged by its fixed prefix (up to the first wildcard)."""
        s = str(pattern)
        cut = min((s.find(c) for c in _GLOB_CHARS if c in s), default=len(s))
        prefix = s[:cut] or "."
        self.check(prefix if prefix.endswith(os.sep) or cut == len(s) else os.path.dirname(prefix) or ".",
                   "pattern")

    def message(self, path, what: str = "path") -> str:
        roots = ", ".join(str(r) for r in self.roots[:4])
        return (f"The {what} {str(path)!r} is outside this workspace. Files must "
                f"be under the session directory or the workspace roots ({roots}); "
                "upload or copy the data there and use that path.")

    # -- the by-name gate ---------------------------------------------
    # Parameters that carry a path in one of the tool registries (union over
    # the meta, analysis, planning and simulation tools).
    PATH_PARAMS = frozenset({
        "file_path", "path", "paths", "data_path", "source_paths", "prior_analysis_paths",
        "reference_scripts", "literature_file", "auxiliary_data", "series_metadata",
        "json_path", "image_path", "text_file_path", "file_paths", "conditions_file",
        "candidate_pool", "knowledge_paths", "code_paths", "primary_data_set",
        "result_data", "file_name", "database_path", "revise_path", "source_files",
        "structure_path", "structure_file", "output_dir", "local_dir", "output_path",
        "data_dir", "directory", "pattern", "metadata", "skill_path", "replay_dir",
    })
    # Free text that MAY contain paths: only absolute tokens are judged.
    TEXTY_PARAMS = frozenset({"literature_context", "molecule_context", "skill", "metadata",
                              "result_data"})

    def refuse_tool_args(self, kwargs: Dict[str, Any], base_dir) -> Optional[str]:
        """The refusal message for the first argument outside the roots, or
        ``None`` when every path-like argument is inside them."""
        for name, value in kwargs.items():
            texty = name in self.TEXTY_PARAMS
            if name in self.PATH_PARAMS or texty:
                bad = self._first_bad(value, base_dir, texty=texty, pattern=(name == "pattern"))
                if bad is not None:
                    return self.message(bad, name)
            elif name == "branches" and isinstance(value, list):
                for b in value:
                    if isinstance(b, dict):
                        for k in ("data_path", "pattern", "metadata"):
                            bad = self._first_bad(b.get(k), base_dir, texty=(k == "metadata"),
                                                  pattern=(k == "pattern"))
                            if bad is not None:
                                return self.message(bad, f"branches.{k}")
            elif name == "datasets" and isinstance(value, list):
                for d in value:
                    if isinstance(d, dict):
                        bad = self._first_bad(d.get("path"), base_dir)
                        if bad is not None:
                            return self.message(bad, "datasets.path")
            elif name == "conditions" and isinstance(value, dict):
                for fname in value:
                    bad = self._first_bad(fname, base_dir)
                    if bad is not None:
                        return self.message(bad, "conditions")
        return None

    def _first_bad(self, value, base_dir, *, texty: bool = False, pattern: bool = False):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (list, tuple)):
            for v in value:
                bad = self._first_bad(v, base_dir, texty=texty, pattern=pattern)
                if bad is not None:
                    return bad
            return None
        if not isinstance(value, str) or not value.strip():
            return None
        tokens = [t.strip() for t in value.split(",")] if (texty or "," in value) else [value.strip()]
        for tok in tokens:
            if not tok:
                continue
            absolute = tok.startswith(("/", "~")) or (len(tok) > 2 and tok[1] == ":" and tok[2] in "\\/")
            if texty and not absolute:
                continue                      # prose, or a name a resolver will place
            if pattern or any(c in tok for c in _GLOB_CHARS):
                target = tok if absolute else os.path.join(str(base_dir), tok)
                try:
                    self.check_pattern(target)
                except PathFenceError:
                    return tok
                continue
            if absolute:
                if not self.allows(tok):
                    return tok
            elif ".." in Path(tok).parts:
                if not self.allows(Path(str(base_dir)) / tok):
                    return tok
        return None


# ── the fence bound on the current thread ────────────────────────────
_bound = threading.local()


def current() -> Optional[PathFence]:
    return getattr(_bound, "fence", None)


@contextlib.contextmanager
def bound(fence: Optional[PathFence]):
    prev = current()
    _bound.fence = fence
    try:
        yield fence
    finally:
        _bound.fence = prev


def held(path, what: str = "path") -> Path:
    """``path`` checked against the thread's fence when one is bound."""
    fence = current()
    if fence is not None:
        return fence.check(path, what)
    return Path(str(path))
