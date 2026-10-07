"""NWChem hooks for the engine-neutral parameter-convergence sweep.

Mirrors the VASP hooks (`scilink/skills/periodic_dft/vasp/vasp_convergence.py`):
the engine-neutral driver resolves these by name through the skill registry.

- ``set_convergence_param`` writes one basis-set ladder value into the deck
  (every ``* library <basis>`` line in the ``basis … end`` block).
- ``read_convergence_observable`` reads the convergence observable
  (total energy, HOMO–LUMO gap, dipole) from a finished run via cclib.

Which basis ladder, observable, and tolerance to use is declared in the
``convergence:`` block of ``nwchem.md``'s frontmatter — not here.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from ..._shared._spec import ToolSpec

logger = logging.getLogger(__name__)

# `library <name>` inside a basis block — the token a basis sweep rewrites.
_LIBRARY_RE = re.compile(r"(\blibrary\s+)(\S+)", re.IGNORECASE)

_HARTREE_EV = 27.211386245988

# NWChem's SCF/DFT reference energy line: "Total DFT energy = -76.35..." (DFT)
# or "Total SCF energy = ..." (HF). Parsed directly so the primary observable
# needs no cclib (which may be absent).
_ENERGY_RE = re.compile(
    r"Total\s+(?:DFT|SCF)\s+energy\s*=\s*(-?\d+\.\d+)", re.IGNORECASE)

# Post-HF correlated total-energy lines, highest level of theory first. For an
# MP2 or CCSD(T) (TCE) run the basis-dependent result is the correlated total,
# not the SCF/DFT reference _ENERGY_RE matches — judging the sweep on the
# reference would converge the wrong quantity and adopt too small a basis. The
# presence of a correlated total is the signal the run was post-HF, so prefer
# whichever the log holds. Formats vary: TCE uses "= " with an optional
# "/ hartree"; the direct MP2 module uses a bare value or ":".
_CORRELATED_ENERGY_RES = (
    re.compile(r"CCSD\(T\)\s+total\s+energy(?:\s*/\s*hartree)?\s*[=:]?\s+(-?\d+\.\d+)",
               re.IGNORECASE),
    re.compile(r"\bCCSD\s+total\s+energy(?:\s*/\s*hartree)?\s*[=:]?\s+(-?\d+\.\d+)",
               re.IGNORECASE),
    re.compile(r"(?:Total\s+MP2|MP2\s+total)\s+energy(?:\s*/\s*hartree)?\s*[=:]?\s+(-?\d+\.\d+)",
               re.IGNORECASE),
)


def _find_nwchem_log(output_dir: str):
    """The NWChem output log in a run dir (LocalExecutor writes run_stdout.log)."""
    from pathlib import Path
    d = Path(output_dir)
    for name in ("run_stdout.log", "stdout", "nwchem.out"):
        if (d / name).is_file():
            return d / name
    for pattern in ("*.out", "*.nwo", "*.log"):
        hits = sorted(d.glob(pattern))
        if hits:
            return hits[-1]
    return None


def _read_total_energy_eV(output_dir: str):
    """Highest-level total energy from the NWChem log, in eV (or None).

    For a post-HF run (MP2, CCSD(T) via TCE) the basis-dependent result is the
    correlated total energy; prefer whichever correlated total the log holds
    (CCSD(T) > CCSD > MP2), falling back to the SCF/DFT reference for an HF/DFT
    run. Takes the last occurrence so a multi-step deck reads its final result.
    """
    log = _find_nwchem_log(output_dir)
    if log is None:
        return None
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    for rx in _CORRELATED_ENERGY_RES:
        matches = rx.findall(text)
        if matches:
            return float(matches[-1]) * _HARTREE_EV
    matches = _ENERGY_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1]) * _HARTREE_EV


def _set_orbital_basis(deck: str, value: Any) -> tuple:
    """Rewrite ``library <x>`` → ``library <value>`` only in the orbital basis.

    A basis sweep must touch the AO (orbital) basis and nothing else. NWChem
    decks also carry ``library`` lines the sweep must leave alone: an ``ecp``
    block (``* library def2-ecp``) and named auxiliary/fitting bases
    (``basis "cd basis" ... * library def2-universal-jkfit``). Rewriting those
    (as a deck-wide substitution does) turns an ECP into a non-ECP basis and
    corrupts the fitting set, so every rung past Kr fails or gives wrong
    energies. Scope the substitution to the orbital basis block — opened by a
    bare ``basis`` or ``basis "ao basis"`` and closed by ``end``.

    Returns ``(new_deck, n_substitutions)``.
    """
    out_lines = []
    in_orbital_basis = False
    n = 0
    for line in deck.split("\n"):
        toks = line.strip().lower().split()
        if toks and toks[0] == "basis":
            # Orbital basis unless it is a *named* auxiliary basis (e.g.
            # "cd basis", "fitting basis"); unnamed or "ao basis" is the orbital one.
            m = re.search(r'"([^"]*)"', line)
            name = (m.group(1).lower() if m else "ao basis")
            in_orbital_basis = name in ("ao basis", "")
            out_lines.append(line)
        elif in_orbital_basis and toks == ["end"]:
            in_orbital_basis = False
            out_lines.append(line)
        elif in_orbital_basis:
            new_line, k = _LIBRARY_RE.subn(
                lambda mm: f"{mm.group(1)}{value}", line)
            n += k
            out_lines.append(new_line)
        else:
            out_lines.append(line)
    return "\n".join(out_lines), n


def _deck_key(input_files: Dict[str, str]) -> str:
    """Return the NWChem deck filename (the single `.nw` file)."""
    nw = [k for k in input_files if k.lower().endswith(".nw")]
    if not nw:
        raise KeyError("no NWChem deck (*.nw) in input_files")
    return nw[0]


def set_convergence_param(
    input_files: Dict[str, str], param: str, value: Any,
) -> Dict[str, str]:
    """Return a copy of ``input_files`` with the basis set to ``value``.

    Args:
        input_files: The base deck as ``{filename: contents}`` (one ``*.nw``).
        param: The frontmatter parameter name — only ``"basis"`` is supported.
        value: The basis-library name to write (e.g. ``"def2-tzvp"``).

    Returns:
        A new ``{filename: contents}`` map with every ``library <x>`` in the
        deck replaced by ``library <value>`` — a basis sweep uses one basis for
        the whole molecule per rung.

    Raises:
        ValueError: If ``param`` is not ``"basis"``.
        KeyError: If there is no ``*.nw`` deck.
        ValueError: If the deck has no ``library`` line to set.
    """
    if param != "basis":
        raise ValueError(
            f"unknown NWChem convergence parameter {param!r}; only 'basis'")
    key = _deck_key(input_files)
    deck = input_files[key]
    new_deck, n = _set_orbital_basis(deck, value)
    if n == 0:
        raise ValueError(
            "no `library <basis>` line found in the NWChem orbital basis block")
    out = dict(input_files)
    out[key] = new_deck
    return out


def read_convergence_observable(
    output_dir: str, observable: str,
) -> Optional[float]:
    """Read one convergence observable from a finished NWChem run.

    Args:
        output_dir: The run directory.
        observable: ``"total_energy"`` (eV), ``"homo_lumo_gap"`` (eV), or
            ``"dipole"`` (Debye).

    Returns:
        The observable as a float, or ``None`` if it cannot be read (missing or
        unparseable output, unknown observable). Never raises.
    """
    if observable == "total_energy":
        # Regex the energy straight from the log — robust and cclib-free, since
        # the total energy is the primary (universal) convergence observable.
        try:
            return _read_total_energy_eV(output_dir)
        except Exception as e:
            logger.warning("total_energy read failed in %s: %s", output_dir, e)
            return None

    # HOMO-LUMO gap and dipole go through cclib (install cclib to use them).
    try:
        import cclib
        import numpy as np
        # Use the same log resolver as total_energy — an unsorted glob can pick
        # run_stderr.log (the executor writes both run_stdout.log and
        # run_stderr.log), which cclib reads as empty -> None every rung.
        log = _find_nwchem_log(output_dir)
        if log is None:
            return None
        data = cclib.io.ccread(str(log))
        if data is None:
            return None
        if observable == "homo_lumo_gap":
            mo = data.moenergies[0]          # eV, first spin channel
            h = data.homos[0]
            return float(mo[h + 1] - mo[h])
        if observable == "dipole":
            return float(np.linalg.norm(data.moments[1]))  # Debye
    except Exception as e:
        logger.warning("could not read %s from %s: %s", observable, output_dir, e)
        return None

    logger.warning("unknown convergence observable %r", observable)
    return None


TOOL_SPECS = [
    ToolSpec(
        name="set_convergence_param",
        description=(
            "Write one basis-set ladder value into an NWChem deck (replaces "
            "`* library <basis>` in the basis block). Engine hook for the "
            "convergence sweep; the basis ladder lives in nwchem.md's "
            "`convergence:` frontmatter."
        ),
        parameters={
            "input_files": {"type": "object",
                            "description": "Base deck {filename: contents}."},
            "param": {"type": "string", "description": "Must be 'basis'."},
            "value": {"type": "string",
                      "description": "Basis library name, e.g. def2-tzvp."},
        },
        required=["input_files", "param", "value"],
        signature="set_convergence_param(input_files: dict, param: str, value) -> dict",
        import_line=(
            "from scilink.skills.molecular_qc.nwchem.nwchem_convergence import "
            "set_convergence_param"),
        agents=["simulation"],
        returns="dict {filename: contents} with the basis applied.",
    ),
    ToolSpec(
        name="read_convergence_observable",
        description=(
            "Read a convergence observable (total_energy, homo_lumo_gap, "
            "dipole) from a finished NWChem run directory via cclib. Engine "
            "hook for the convergence sweep; returns None if unreadable."
        ),
        parameters={
            "output_dir": {"type": "string",
                           "description": "Finished NWChem run directory."},
            "observable": {"type": "string",
                           "description": "total_energy | homo_lumo_gap | dipole."},
        },
        required=["output_dir", "observable"],
        signature="read_convergence_observable(output_dir: str, observable: str) -> float | None",
        import_line=(
            "from scilink.skills.molecular_qc.nwchem.nwchem_convergence import "
            "read_convergence_observable"),
        agents=["simulation"],
        returns="float observable value, or None if unreadable.",
    ),
]
