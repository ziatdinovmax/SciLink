"""VASP hooks for the engine-neutral parameter-convergence sweep.

The convergence driver (`scilink.agents.sim_agents.convergence`) is engine-
neutral; the engine specifics live here and are resolved by name through the
skill registry (like ``default_run_command`` / ``snapshot_run``):

- ``set_convergence_param`` writes one ladder value into a deck (ENCUT into
  INCAR; k-point density via INCAR ``KSPACING``, dropping any KPOINTS file so
  the spacing takes effect).
- ``read_convergence_observable`` reads the convergence observable
  (energy/atom, lattice constant, band gap) from a finished run.

Which parameters, ladders, observables, and tolerances to use is declared in
the ``convergence:`` block of ``vasp.md``'s frontmatter — not here.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from ..._shared._spec import ToolSpec

logger = logging.getLogger(__name__)

_INCAR = "INCAR"
_KPOINTS = "KPOINTS"

# frontmatter `parameter` name -> INCAR key it maps to.
_PARAM_TO_INCAR_KEY = {
    "ENCUT": "ENCUT",
    "KSPACING": "KSPACING",
    "k-points": "KSPACING",
    "kpoints": "KSPACING",
}


def _set_incar_key(incar_text: str, key: str, value: Any) -> str:
    """Return INCAR text with ``key`` set to ``value`` (replaced or appended).

    VASP allows several tags on one line separated by ``;`` and trailing ``#`` /
    ``!`` comments, so the replacement matches only the key's own value and
    stops at the first ``;``, ``#`` or ``!`` — anything after it (other tags, a
    comment) is preserved. Matching to end of line would drop them.
    """
    newval = f"{key} = {value}"
    pattern = re.compile(
        rf"^(?P<indent>\s*){re.escape(key)}\s*=\s*[^;#!\n]*(?P<rest>[;#!].*)?$",
        re.MULTILINE | re.IGNORECASE)
    if pattern.search(incar_text):
        return pattern.sub(
            lambda m: f"{m.group('indent')}{newval}{m.group('rest') or ''}",
            incar_text)
    sep = "" if incar_text.endswith("\n") or not incar_text else "\n"
    return f"{incar_text}{sep}{newval}\n"


def set_convergence_param(
    input_files: Dict[str, str], param: str, value: Any,
) -> Dict[str, str]:
    """Return a copy of ``input_files`` with one convergence parameter set.

    Args:
        input_files: The base deck as ``{filename: contents}`` (must contain
            INCAR).
        param: The frontmatter ``parameter`` name (e.g. ``"ENCUT"`` or
            ``"k-points"``).
        value: The ladder value to write.

    Returns:
        A new ``{filename: contents}`` map with the parameter applied. For a
        k-point sweep the INCAR ``KSPACING`` is set and any ``KPOINTS`` file is
        removed so the spacing governs the mesh.

    Raises:
        KeyError: If INCAR is absent.
        ValueError: If ``param`` is not a known VASP convergence parameter.
    """
    incar_key = _PARAM_TO_INCAR_KEY.get(param)
    if incar_key is None:
        raise ValueError(
            f"unknown VASP convergence parameter {param!r}; "
            f"known: {sorted(set(_PARAM_TO_INCAR_KEY))}")
    if _INCAR not in input_files:
        raise KeyError("INCAR not present in input_files")

    out = dict(input_files)
    out[_INCAR] = _set_incar_key(out[_INCAR], incar_key, value)
    if incar_key == "KSPACING":
        out.pop(_KPOINTS, None)  # KSPACING is ignored while a KPOINTS file exists
    return out


def get_convergence_param(
    input_files: Dict[str, str], param: str,
) -> Optional[float]:
    """Return the base deck's current value for a convergence parameter.

    The sweep uses this to *floor* its ladder at the value input validation
    already approved (e.g. ENCUT >= 1.3x max ENMAX, vasp.md:293), so a plateau
    that the observable happens to reach at a lower rung cannot pull the
    production run below the validated setting.

    Args:
        input_files: The base deck as ``{filename: contents}``.
        param: The frontmatter ``parameter`` name (e.g. ``"ENCUT"``).

    Returns:
        The current value as a float, or ``None`` when it is not present in a
        comparable form — a missing INCAR, no such key, or a k-point sweep whose
        base deck uses an explicit KPOINTS mesh rather than INCAR ``KSPACING``
        (no scalar to compare). On ``None`` the driver applies no floor for that
        parameter rather than guessing. Never raises for a readable deck.

    Raises:
        ValueError: If ``param`` is not a known VASP convergence parameter.
    """
    incar_key = _PARAM_TO_INCAR_KEY.get(param)
    if incar_key is None:
        raise ValueError(
            f"unknown VASP convergence parameter {param!r}; "
            f"known: {sorted(set(_PARAM_TO_INCAR_KEY))}")
    incar_text = input_files.get(_INCAR)
    if not incar_text:
        return None
    # A k-point floor is comparable only when the base deck expresses density as
    # KSPACING; with an explicit KPOINTS file present there is no scalar to floor.
    if incar_key == "KSPACING" and _KPOINTS in input_files:
        return None
    m = re.search(rf"^\s*{re.escape(incar_key)}\s*=\s*([0-9.eE+-]+)",
                  incar_text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def read_convergence_observable(
    output_dir: str, observable: str,
) -> Optional[float]:
    """Read one convergence observable from a finished VASP run.

    Args:
        output_dir: The run directory (must contain vasprun.xml).
        observable: One of ``"energy_per_atom"`` (eV/atom),
            ``"lattice_constant_a"`` (Å), ``"band_gap"`` (eV).

    Returns:
        The observable as a float, or ``None`` if it cannot be read (missing or
        unparseable output, unknown observable). Never raises — a failed rung
        becomes a ``None`` the comparator skips.
    """
    vasprun_path = Path(output_dir) / "vasprun.xml"
    if not vasprun_path.exists():
        return None
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except ImportError as e:  # pragma: no cover
        logger.warning("pymatgen not available: %s", e)
        return None

    need_eigen = observable == "band_gap"
    try:
        vr = Vasprun(
            str(vasprun_path), parse_dos=False, parse_eigen=need_eigen,
            parse_potcar_file=False, exception_on_bad_xml=False,
        )
    except Exception as e:
        logger.warning("vasprun.xml parse failed in %s: %s", output_dir, e)
        return None

    try:
        if observable == "energy_per_atom":
            # A run that hit NELM without electronic convergence still reports a
            # final energy; accepting it would let the comparator fake or block a
            # plateau. An unconverged rung reads as None (per this hook's contract).
            if not vr.converged_electronic:
                logger.warning(
                    "SCF not electronically converged in %s; energy_per_atom=None",
                    output_dir)
                return None
            n = len(vr.final_structure)
            return float(vr.final_energy) / n if n else None
        if observable == "lattice_constant_a":
            # A geometry observable — require ionic relaxation (and its SCF
            # steps) to have converged, not just a parsable final structure.
            if not vr.converged:
                logger.warning(
                    "relaxation not converged in %s; lattice_constant_a=None",
                    output_dir)
                return None
            return float(vr.final_structure.lattice.abc[0])
        if observable == "band_gap":
            if not vr.converged_electronic:
                logger.warning(
                    "SCF not electronically converged in %s; band_gap=None",
                    output_dir)
                return None
            return float(vr.eigenvalue_band_properties[0])
    except Exception as e:
        logger.warning("could not read %s from %s: %s", observable, output_dir, e)
        return None

    logger.warning("unknown convergence observable %r", observable)
    return None


def kspacing_to_mesh(output_dir: str) -> Optional[str]:
    """Return the k-mesh a KSPACING run generated, as ``"n1×n2×n3"``.

    Computes the mesh VASP derives from KSPACING and the cell —
    ``N_i = max(1, ceil(|b_i| / KSPACING))`` with reciprocal vectors ``b_i``
    including the 2π factor — so a human-facing report can speak in meshes even
    though the ladder is declared in KSPACING. Reads the run's INCAR (for
    KSPACING) and POSCAR (for the cell). Returns ``None`` when the run has no
    KSPACING (e.g. an ENCUT rung) or the cell cannot be read.
    """
    d = Path(output_dir)
    incar, poscar = d / "INCAR", d / "POSCAR"
    if not incar.is_file() or not poscar.is_file():
        return None
    m = re.search(r"^\s*KSPACING\s*=\s*([0-9.eE+-]+)", incar.read_text(),
                  re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    try:
        import math
        import numpy as np
        from ase.io import read as _read
        kspacing = float(m.group(1))
        atoms = _read(str(poscar), format="vasp")
        recip = 2 * math.pi * np.linalg.inv(np.array(atoms.cell[:])).T  # rows b_i
        dims = [max(1, math.ceil(float(np.linalg.norm(recip[i])) / kspacing))
                for i in range(3)]
        return "×".join(str(n) for n in dims)
    except Exception as e:  # display-only; never break on it
        logger.debug("kspacing_to_mesh failed in %s: %s", output_dir, e)
        return None


TOOL_SPECS = [
    ToolSpec(
        name="set_convergence_param",
        description=(
            "Write one convergence-ladder value into a VASP deck (ENCUT into "
            "INCAR, or k-point density via INCAR KSPACING). Engine hook for the "
            "convergence sweep; declared parameters/ladders live in vasp.md's "
            "`convergence:` frontmatter."
        ),
        parameters={
            "input_files": {"type": "object",
                            "description": "Base deck {filename: contents}."},
            "param": {"type": "string",
                      "description": "Frontmatter parameter name, e.g. ENCUT."},
            "value": {"type": ["number", "string"],
                      "description": "Ladder value to write."},
        },
        required=["input_files", "param", "value"],
        signature="set_convergence_param(input_files: dict, param: str, value) -> dict",
        import_line=(
            "from scilink.skills.periodic_dft.vasp.vasp_convergence import "
            "set_convergence_param"),
        agents=["simulation"],
        returns="dict {filename: contents} with the parameter applied.",
    ),
    ToolSpec(
        name="get_convergence_param",
        description=(
            "Read a VASP deck's current convergence-parameter value (ENCUT, or "
            "k-point density via INCAR KSPACING) so the sweep can floor its "
            "ladder at the validated setting. Returns None if not present in a "
            "comparable form (e.g. an explicit KPOINTS mesh)."
        ),
        parameters={
            "input_files": {"type": "object",
                            "description": "Base deck {filename: contents}."},
            "param": {"type": "string",
                      "description": "Frontmatter parameter name, e.g. ENCUT."},
        },
        required=["input_files", "param"],
        signature="get_convergence_param(input_files: dict, param: str) -> float | None",
        import_line=(
            "from scilink.skills.periodic_dft.vasp.vasp_convergence import "
            "get_convergence_param"),
        agents=["simulation"],
        returns="float current value, or None if not comparably present.",
    ),
    ToolSpec(
        name="read_convergence_observable",
        description=(
            "Read a convergence observable (energy_per_atom, lattice_constant_a, "
            "band_gap) from a finished VASP run directory. Engine hook for the "
            "convergence sweep; returns None if unreadable."
        ),
        parameters={
            "output_dir": {"type": "string",
                           "description": "Finished VASP run directory."},
            "observable": {"type": "string",
                           "description": "energy_per_atom | lattice_constant_a "
                                          "| band_gap."},
        },
        required=["output_dir", "observable"],
        signature="read_convergence_observable(output_dir: str, observable: str) -> float | None",
        import_line=(
            "from scilink.skills.periodic_dft.vasp.vasp_convergence import "
            "read_convergence_observable"),
        agents=["simulation"],
        returns="float observable value, or None if unreadable.",
    ),
]
