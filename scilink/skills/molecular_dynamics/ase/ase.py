"""ASE runner skill bundle — engine surface for the MLIP execution path.

The ASE runner (``scilink/agents/sim_agents/_ase_runner.py``) drives any
deployed MLIP potential through ASE for MD or relaxation, emitting a runnable
Python driver script (``run_md.py`` / ``run_relax.py``). No classical MD engine
is involved: the script is plain Python, so it is launched with the current
interpreter and needs no external binary.

This bundle is that path's engine surface for the one-shot pipeline, mirroring
the ``lammps`` bundle so the refinement loop can re-point the health gate and
RunCritic grounding at ``ase`` for an MLIP run (MLIP-execution-in-loop):

  * :func:`default_run_command` — how the generated script is launched.
  * :func:`read_health_observable` — how a finished run's gross physical-sanity
    observable is read back for the engine-neutral health gate.

The registry auto-discovers this module (``_SKILLS_DIR/<domain>/<name>/*.py``)
and pairs each ``TOOL_SPEC``/``TOOL_SPECS`` entry with the same-named callable;
no central wiring is needed.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..._shared._spec import ToolSpec

logger = logging.getLogger(__name__)


def default_run_command(script: str = "{script}") -> Optional[str]:
    """Conventional run-command template for an ASE driver script.

    The ASE runner emits a self-contained Python script (``run_md.py`` /
    ``run_relax.py``), so the run command is simply ``<python> <script>`` — no
    engine binary, no new tool (``LocalExecutor`` runs it as a shell string).
    ``sys.executable`` (not a bare ``"python"``) keeps the run in the SAME
    environment SciLink is running in, so the MLIP backend the script imports
    is the one installed here.

    Returns:
        The run-command template, or ``None`` if ``ase`` is not importable —
        the bundle cannot run a driver script without it, and the caller should
        then fall back to a user-supplied ``run_command``.
    """
    try:
        if importlib.util.find_spec("ase") is None:
            return None
    except (ImportError, ValueError):
        return None
    return f"{sys.executable} {script}"


# ─── Health reader ───────────────────────────────────────────────────

_THERMO_LOG_NAMES = ("thermo.log",)

# The ASE MD driver writes a single ``# step time(ps) PE(eV) KE(eV) T(K)``
# header followed by whitespace-delimited rows (see _ase_runner.py). Map only
# the observables the runner actually logs: temperature and the two energies.
# It logs NO density / pressure / volume column, so those are deliberately
# absent here — the gate then SKIPS them rather than compare against a column
# that does not exist. Header tokens are matched case-insensitively.
_HEALTH_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "temperature": ("T(K)", "temperature", "temp"),
    "potential_energy": ("PE(eV)", "poteng", "pe"),
    "kinetic_energy": ("KE(eV)", "ke"),
}


def _find_thermo_log(output_dir: str) -> Optional[Path]:
    """Return the top-level ``thermo.log`` in ``output_dir``, or ``None``.

    Deliberately not recursive: a phase owns its run directory, mirroring the
    lammps reader so the gate never guesses at a nested log.
    """
    d = Path(output_dir)
    if not d.is_dir():
        return None
    for name in _THERMO_LOG_NAMES:
        hit = d / name
        if hit.is_file():
            return hit
    return None


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def _equilibrated_mean(values: List[float]) -> Optional[float]:
    """Mean of the back half of a thermo series (drops startup transients)."""
    if not values:
        return None
    tail = values[len(values) // 2:] or values
    return sum(tail) / len(tail)


def _read_named_column(log_file: Path, candidates: Tuple[str, ...]
                       ) -> List[float]:
    """Read the numeric column whose header matches one of ``candidates``.

    The ASE thermo log has one ``#``-prefixed header line naming the columns,
    then whitespace-delimited numeric rows. A short/garbled row (e.g. a
    truncated final line) is skipped, never fatal. Returns the column's values
    in order, or ``[]`` if the column (or a header) is absent. Never raises.
    """
    low_candidates = [c.lower() for c in candidates]
    idx: Optional[int] = None
    ncols: Optional[int] = None
    values: List[float] = []
    try:
        with open(log_file, "r", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    header = s.lstrip("#").split()
                    low = [h.lower() for h in header]
                    for cand in low_candidates:
                        if cand in low:
                            idx = low.index(cand)
                            ncols = len(header)
                            break
                    continue
                if idx is None:
                    continue
                toks = s.split()
                # Only accept a full numeric data row of the header's width, so
                # a stray log line can't be read as thermo data.
                if ncols is not None and len(toks) != ncols:
                    continue
                if not all(_is_number(t) for t in toks):
                    continue
                values.append(float(toks[idx]))
    except Exception as e:
        logger.debug("could not parse ASE thermo log %s: %s", log_file, e)
    return values


def read_health_observable(output_dir: str, observable: str,
                           since: Optional[float] = None) -> Optional[float]:
    """Read one gross physical-sanity observable from a finished ASE MD run.

    Engine hook for the engine-neutral health gate (mirrors the lammps hook).
    Reads the matching column from the ASE ``thermo.log`` and returns its
    back-half (equilibrated) mean. Which observables and plausible bands to
    check live in ``ase.md``'s ``health:`` frontmatter, not here.

    The ASE MD driver logs temperature + energies only — there is NO density /
    pressure / volume column — so a request for one of those returns ``None``
    (unknown observable) and the gate skips it rather than misfire.

    ``since`` is the wall-clock time just before the run; the log is treated as
    stale (``None`` returned) if it is not newer than that, so a phase that
    produced no log of its own in a shared run directory is not judged on an
    earlier phase's log.

    Returns the value as a float, or ``None`` if it cannot be read (no log,
    stale log, unknown observable, or column absent). Never raises.
    """
    cols = _HEALTH_COLUMNS.get(observable.lower())
    if not cols:
        logger.debug("unknown health observable %r for ASE", observable)
        return None
    log = _find_thermo_log(output_dir)
    if log is None:
        return None
    # A log older than the run start is a leftover from an earlier phase in a
    # shared run directory; treat it as absent rather than judge on it.
    if since is not None:
        try:
            if log.stat().st_mtime + 1.0 < since:
                logger.debug("ASE thermo log %s predates the run; "
                             "treating as absent", log)
                return None
        except OSError:
            return None
    values = _read_named_column(log, cols)
    return _equilibrated_mean(values)


TOOL_SPECS = [
    ToolSpec(
        name="read_health_observable",
        description=(
            "Read a gross physical-sanity observable (temperature, "
            "potential_energy, kinetic_energy) from a finished ASE MD run "
            "directory by parsing its thermo.log. Engine hook for the health "
            "gate; returns None if unreadable. Plausible bands live in ase.md's "
            "`health:` frontmatter. The ASE MD driver logs no density / "
            "pressure / volume, so those are not served."
        ),
        parameters={
            "output_dir": "Finished ASE run directory (string).",
            "observable": (
                "Observable name: temperature | potential_energy | "
                "kinetic_energy."
            ),
        },
        agents=["simulation"],
    ),
]


TOOL_SPEC = ToolSpec(
    name="default_run_command",
    description=(
        "Return the conventional run-command template for an ASE driver script "
        "('<python> {script}', using the current interpreter), or None if ase "
        "is not importable. Lets the one-shot workflow execute an MLIP run "
        "through ASE with no engine binary or launch command hardcoded in "
        "shared code."
    ),
    parameters={},
    agents=["simulation"],
)
