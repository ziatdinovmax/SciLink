"""
Available-software discovery and persistence.

The simulation orchestrator routes user goals to engines by intersecting:

    agent_supports × user_available × goal_appropriateness

The first comes from skill-bundle discovery
(``Agent.supported_software()``); the third comes from the LLM.
This module owns the middle factor — what the user actually has
installed / has access to on their cluster or laptop.

**Detection guidance lives in the skill markdown's YAML frontmatter,
not in a separate Python file**, so adding a new engine never requires
boilerplate code — a markdown-only skill bundle is sufficient:

    ---
    description: VASP DFT input generation
    detect:
      binaries:        [vasp_std, vasp_gam, vasp_ncl, vasp]
      env_vars:        [VASP_HOME, VASP_DIR]
      python_modules:  []
      guidance: |
        VASP requires a binary on PATH. On HPC clusters, may need
        `module load vasp/6.4.3` first.
    ---
    ## overview
    ...

Detection proceeds in fast-path order:
  1. ``binaries``        : each is checked via ``shutil.which``
  2. ``env_vars``        : if set and point to a binary, that wins
  3. ``python_modules``  : ``importlib.util.find_spec``

If structured hints fail / are absent, the ``guidance`` field is
context for a future LLM-driven probe path (separate commit).

Modes:
    AvailableSoftware.load()      # YAML only (raises if missing)
    AvailableSoftware.detect()    # probes only (no YAML I/O)
    AvailableSoftware.auto()      # YAML if exists, else detect+save
                                  # (non-interactive default)
    AvailableSoftware.refresh()   # re-probe; merge into existing
                                  # YAML; preserve user_confirmed
"""

import importlib.util
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..skills.loader import list_all_skills, load_skill

_logger = logging.getLogger(__name__)

_DEFAULT_YAML_PATH = Path.home() / ".scilink" / "available_software.yaml"


def _yaml_path() -> Path:
    """Resolve the path for the persisted YAML config."""
    override = os.environ.get("SCILINK_AVAILABLE_SOFTWARE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_YAML_PATH


# ─── Fast-path probes (no LLM) ─────────────────────────────────────

def _probe_binaries(names: List[str]) -> Optional[Dict[str, Any]]:
    """Look for any of ``names`` on $PATH. Returns probe result or None."""
    for name in names:
        path = shutil.which(name)
        if path:
            return {
                "available": True,
                "binary": path,
                "binary_name": name,
                "source": "shutil.which",
            }
    return None


def _probe_env_vars(env_vars: List[str], binary_names: List[str]) -> Optional[Dict[str, Any]]:
    """Check each env var for a directory containing one of the binaries."""
    for var in env_vars:
        base = os.environ.get(var)
        if not base:
            continue
        base_path = Path(base).expanduser()
        if not base_path.is_dir():
            continue
        for name in binary_names or [var.lower()]:
            candidate = base_path / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return {
                    "available": True,
                    "binary": str(candidate),
                    "binary_name": name,
                    "source": f"${var}",
                }
    return None


def _probe_python_modules(modules: List[str]) -> Optional[Dict[str, Any]]:
    """Look for any of ``modules`` as importable Python packages.

    ``find_spec`` does not merely return ``None`` for a missing module: probing
    a submodule (e.g. ``fairchem.core``) imports its parent package first, so an
    absent parent raises ``ModuleNotFoundError`` — and a broken/partial install
    can raise other ``ImportError``s. Any such failure means "not importable
    here", so treat it exactly like a ``None`` spec rather than letting it
    propagate and abort detection of every other engine.
    """
    for mod in modules:
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            # ImportError covers ModuleNotFoundError (missing parent package);
            # ValueError guards the __spec__-is-None edge case find_spec raises.
            spec = None
        if spec is not None:
            return {
                "available": True,
                "python_module": mod,
                "module_origin": spec.origin or "",
                "source": "importlib.util.find_spec",
            }
    return None


def _probe_from_frontmatter(domain: str, engine: str) -> Dict[str, Any]:
    """
    Run the fast-path probe for one skill bundle, using the structured
    ``detect:`` block in its frontmatter.

    Returns a probe-result dict with at least ``available`` (bool). On
    failure the dict carries ``reason`` and the ``guidance`` (if any)
    that an LLM-driven probe could use as context.
    """
    try:
        parsed = load_skill(engine, domain=domain)
    except FileNotFoundError:
        return {
            "available": False,
            "source": "no-skill",
            "reason": f"no skill bundle at {domain}/{engine}/{engine}.md",
        }
    except Exception as exc:
        return {
            "available": False,
            "source": "skill-load-error",
            "reason": repr(exc),
        }

    meta = parsed.get("meta") or {}
    detect = meta.get("detect")
    if not isinstance(detect, dict):
        return {
            "available": False,
            "source": "no-detect-block",
            "reason": (
                f"skill {domain}/{engine} has no `detect:` block in its "
                "frontmatter; add one or rely on LLM probe fallback"
            ),
            "guidance": None,
        }

    binaries = detect.get("binaries") or []
    env_vars = detect.get("env_vars") or []
    py_modules = detect.get("python_modules") or []
    guidance = detect.get("guidance")

    if not isinstance(binaries, list):
        binaries = [str(binaries)]
    if not isinstance(env_vars, list):
        env_vars = [str(env_vars)]
    if not isinstance(py_modules, list):
        py_modules = [str(py_modules)]

    if binaries:
        hit = _probe_binaries(binaries)
        if hit is not None:
            return hit

    if env_vars:
        hit = _probe_env_vars(env_vars, binaries)
        if hit is not None:
            return hit

    if py_modules:
        hit = _probe_python_modules(py_modules)
        if hit is not None:
            return hit

    return {
        "available": False,
        "source": "frontmatter-probe",
        "reason": (
            f"no match: tried binaries {binaries}, env vars {env_vars}, "
            f"python modules {py_modules}"
        ),
        "guidance": guidance,
    }


class AvailableSoftware:
    """Container for available-software state across scale domains.

    Internal shape::

        {
            "periodic_dft":  {"vasp":  {"available": True, ...}},
            "molecular_dynamics":  {"lammps": {...}},
            "machine_learning_potentials": {"mace": {...}},
            ...
        }

    Engine entries store at least ``available`` (bool); probes may
    include ``binary``, ``binary_name``, ``python_module``,
    ``module_origin``, ``source``, ``reason``, ``guidance``,
    ``user_confirmed``.
    """

    def __init__(self, data: Optional[Dict[str, Dict[str, dict]]] = None):
        self._data: Dict[str, Dict[str, dict]] = data or {}

    # ── classmethod constructors ───────────────────────────────────

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AvailableSoftware":
        """Load from YAML. Raises ``FileNotFoundError`` if missing."""
        p = Path(path) if path else _yaml_path()
        if not p.is_file():
            raise FileNotFoundError(f"No available-software YAML at {p}")
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML at {p} is not a mapping (got {type(raw).__name__})"
            )
        return cls(data=raw)

    @classmethod
    def detect(cls) -> "AvailableSoftware":
        """Run every discoverable skill bundle's frontmatter probe.

        No YAML I/O. Returns a fresh AvailableSoftware reflecting only
        what was probed in this process via the bundle's
        ``detect:`` block.
        """
        data: Dict[str, Dict[str, dict]] = {}
        for domain, engines in list_all_skills().items():
            for engine in engines:
                info = _probe_from_frontmatter(domain, engine)
                data.setdefault(domain, {})[engine] = info
        return cls(data=data)

    @classmethod
    def auto(cls, path: Optional[Path] = None) -> "AvailableSoftware":
        """Load YAML if it exists; otherwise detect and save.

        Non-interactive default. Suitable for headless / scripted use.
        """
        p = Path(path) if path else _yaml_path()
        if p.is_file():
            return cls.load(p)
        cfg = cls.detect()
        cfg.save(p)
        return cfg

    @classmethod
    def refresh(cls, path: Optional[Path] = None) -> "AvailableSoftware":
        """Force a re-probe; merge into existing YAML; save.

        Preserves ``user_confirmed: true`` entries from the existing
        YAML so manual overrides survive a re-probe.
        """
        p = Path(path) if path else _yaml_path()
        existing = cls()
        if p.is_file():
            try:
                existing = cls.load(p)
            except Exception as exc:
                _logger.warning("Could not load existing YAML: %s", exc)
        fresh = cls.detect()
        merged = existing.merge(fresh, prefer="newer-truthy")
        merged.save(p)
        return merged

    # ── query ──────────────────────────────────────────────────────

    def has(self, domain: str, engine: str) -> bool:
        """Whether ``engine`` is marked available for ``domain``."""
        return bool(self._data.get(domain, {}).get(engine, {}).get("available"))

    def list_available(self, domain: Optional[str] = None) -> List[str]:
        """Engines marked available, optionally filtered by domain."""
        if domain is not None:
            return sorted(
                e for e, info in self._data.get(domain, {}).items()
                if info.get("available")
            )
        out: List[str] = []
        for engines in self._data.values():
            out.extend(e for e, info in engines.items() if info.get("available"))
        return sorted(set(out))

    def domains(self) -> List[str]:
        return sorted(self._data.keys())

    def get(self, domain: str, engine: str) -> dict:
        """Return the raw metadata dict for one engine (empty if absent)."""
        return dict(self._data.get(domain, {}).get(engine, {}))

    def as_dict(self) -> Dict[str, Dict[str, dict]]:
        """Deep-ish snapshot of the underlying state."""
        return {d: {e: dict(info) for e, info in eng.items()}
                for d, eng in self._data.items()}

    # ── update ─────────────────────────────────────────────────────

    def set(self, domain: str, engine: str, available: bool,
            **metadata: Any) -> None:
        """Set or update one engine entry.

        Convention: pass ``user_confirmed=True`` when the value comes
        from explicit user input (interactive confirmation or manual
        YAML edit), so ``refresh()`` knows not to clobber it.
        """
        entry = self._data.setdefault(domain, {}).setdefault(engine, {})
        entry["available"] = bool(available)
        for k, v in metadata.items():
            entry[k] = v

    def merge(self, other: "AvailableSoftware",
              prefer: str = "other") -> "AvailableSoftware":
        """Merge ``other`` into a fresh AvailableSoftware.

        ``prefer``:
          - ``"other"``       : other's entries override self's
          - ``"self"``        : self's entries override other's
          - ``"newer-truthy"``: prefer available=True over False on
                                conflict; preserve user_confirmed
                                flags from self
        """
        merged: Dict[str, Dict[str, dict]] = self.as_dict()
        for domain, engines in other._data.items():
            for engine, info in engines.items():
                if prefer == "self" and engine in merged.get(domain, {}):
                    continue
                if prefer == "newer-truthy":
                    cur = merged.setdefault(domain, {}).get(engine, {})
                    new = dict(info)
                    if cur.get("user_confirmed"):
                        new["user_confirmed"] = True
                    if cur.get("available") and not new.get("available"):
                        new["available"] = True
                    merged[domain][engine] = new
                else:
                    merged.setdefault(domain, {})[engine] = dict(info)
        return AvailableSoftware(data=merged)

    # ── persistence ────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Write the state to YAML, returning the path written."""
        p = Path(path) if path else _yaml_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        from .text_io import atomic_write_text
        atomic_write_text(p, yaml.safe_dump(
            self._data, sort_keys=True, default_flow_style=False
        ))
        return p
