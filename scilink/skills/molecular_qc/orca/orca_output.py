"""ORCA output snapshot parser.

Reads a completed (or failed) ORCA molecular-QC run and produces a structured
summary that simulate-mode critics hand to the LLM when assessing the run.
Discovered via the skill registry when the ``orca`` skill is active and called
through :func:`scilink.skills._shared._registry.get_tool_function`.

The parse goes through **cclib**, which reads ORCA / NWChem / Gaussian / PySCF
output uniformly — so this ``snapshot_run`` shares the exact contract of the
NWChem twin (``molecular_qc/nwchem/nwchem_output.py``); only the engine-specific
error signatures and output-filename conventions differ. cclib is imported
lazily inside the function so this module (and its ``TOOL_SPEC``) import cleanly
even where cclib is not installed.

All code paths return a dict with a ``status`` key; failures are reported in the
returned structure rather than raised, so callers can hand the result to the LLM
uniformly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec

logger = logging.getLogger(__name__)


# Known ORCA failure signatures surfaced via the output log so the summary can
# flag common failure modes even when cclib cannot parse a value.
_ORCA_ERROR_HINTS = [
    ("SCF NOT CONVERGED AFTER",
     "SCF did not converge — raise `%scf maxiter`, try `SlowConv`/`VerySlowConv`, "
     "a different guess (`%scf guess ...`), or add diffuse functions for anions"),
    ("The optimization did not converge",
     "geometry optimization did not converge — loosen nothing; check the geometry, "
     "raise `%geom maxiter`, or revisit the method/basis"),
    ("ORCA finished by error termination",
     "ORCA aborted — see the lines above the termination banner for the cause"),
    ("UNRECOGNIZED OR DUPLICATED KEYWORD",
     "deck syntax error — an unknown/duplicated keyword on the `!` line or in a "
     "`%...end` block"),
    ("is odd. Please specify the correct multiplicity",
     "charge/multiplicity inconsistent with the electron count — fix the "
     "`* xyz <charge> <mult>` line"),
    ("Error (ORCA_MAIN): Multiplicity",
     "multiplicity inconsistent with the electron count — fix charge/mult on the "
     "coordinate line"),
    ("could not find the basis",
     "basis set missing for an element — pick a covering basis or add it"),
    ("Error : Basis set", "basis set problem — check the basis keyword covers every element"),
    ("There are no atoms",
     "empty/unreadable coordinate block — check the `* xyz ... *` geometry"),
    ("aborting the run", "ORCA aborted the run — inspect the log for the cause"),
]

# ORCA prints this banner on a clean finish.
_ORCA_SUCCESS_MARK = "ORCA TERMINATED NORMALLY"


def _classify_log_errors(log_text: str) -> List[str]:
    """Return human-readable hints matching known ORCA error patterns."""
    hits: List[str] = []
    low = log_text.lower()
    for pattern, hint in _ORCA_ERROR_HINTS:
        if pattern.lower() in low:
            hits.append(f"{pattern}: {hint}")
    return hits


def _read_tail(path: Path, max_chars: int = 80_000) -> Optional[str]:
    """Read a file, returning at most ``max_chars`` of its tail."""
    try:
        text = path.read_text(errors="replace")
    except Exception as e:  # pragma: no cover - unreadable file
        logger.warning("Could not read %s: %s", path, e)
        return None
    return text[-max_chars:] if len(text) > max_chars else text


def _summarize_with_cclib(log_path: Path) -> Dict[str, Any]:
    """Parse an ORCA output via cclib and extract convergence + energetics.

    Returns a dict with some subset of ``converged``, ``scf_energy_eV``,
    ``n_imaginary_freqs``, ``enthalpy_hartree``, ``free_energy_hartree``.
    On parse failure returns ``{"error": "..."}``. cclib is imported here so
    the module stays importable without it.
    """
    try:
        import cclib
    except ImportError as e:
        return {"error": f"cclib not available: {e}"}

    try:
        data = cclib.io.ccread(str(log_path))
    except Exception as e:  # pragma: no cover - cclib parse failure
        return {"error": f"cclib parse failed: {e}"}
    if data is None:
        return {"error": "cclib could not parse the output"}

    out: Dict[str, Any] = {}

    # optdone: True when a geometry optimization converged. For a single-point
    # there is no optimization, so absence is not a failure.
    optdone = getattr(data, "optdone", None)
    out["converged"] = bool(optdone) if optdone is not None else None

    scf = getattr(data, "scfenergies", None)
    if scf is not None and len(scf):
        out["scf_energy_eV"] = float(scf[-1])

    vibfreqs = getattr(data, "vibfreqs", None)
    if vibfreqs is not None:
        out["n_imaginary_freqs"] = int(sum(1 for f in vibfreqs if f < 0))

    enthalpy = getattr(data, "enthalpy", None)
    if enthalpy is not None:
        out["enthalpy_hartree"] = float(enthalpy)

    freeenergy = getattr(data, "freeenergy", None)
    if freeenergy is not None:
        out["free_energy_hartree"] = float(freeenergy)

    return out


def snapshot_run(output_dir: str) -> Dict[str, Any]:
    """Summarize an ORCA run directory into a structured snapshot.

    Inspects the directory for an ORCA output log (``*.out`` / ``*.log`` /
    ``stdout``), parses it with cclib for convergence and energetics, and
    tail-matches the log for known ORCA failure patterns and the normal
    termination banner.

    Args:
        output_dir: Path to the directory containing the run's output files.

    Returns:
        A dict with fields:

            status              ``"ok"`` or ``"error"``
            output_directory    The directory inspected
            files_found         List of recognized output files present
            parse               Dict from :func:`_summarize_with_cclib`
                                (convergence + energetics), or ``None`` when
                                no log was found
            scf_energy          Convenience alias for the SCF energy (eV)
            free_energy         Convenience alias for the free energy (Hartree)
            imaginary_freqs     Convenience alias for the imaginary-mode count
            converged           Convenience alias for optimization convergence
            terminated_normally Whether the ORCA success banner was found
            log_error_hints     List of ``"<pattern>: <hint>"`` strings
            convergence_status  ``"converged"`` / ``"not_converged"`` /
                                ``"failed"`` / ``"unknown"``
            headline            One-sentence top-line assessment
    """
    out_dir = Path(output_dir)
    if not out_dir.exists():
        return {"status": "error", "message": f"Directory not found: {output_dir}"}
    if not out_dir.is_dir():
        return {"status": "error", "message": f"Not a directory: {output_dir}"}

    summary: Dict[str, Any] = {
        "status": "ok",
        "output_directory": str(out_dir),
        "files_found": [],
        "parse": None,
        "scf_energy": None,
        "free_energy": None,
        "imaginary_freqs": None,
        "converged": None,
        "terminated_normally": None,
        "log_error_hints": [],
        "convergence_status": "unknown",
    }

    # ORCA output filenames are not fixed (usually the input stem + ``.out``);
    # gather the usual candidates.
    logs = sorted(out_dir.glob("*.out")) + sorted(out_dir.glob("*.log"))
    for extra in ("stdout", "stdout.log", "orca.out"):
        p = out_dir / extra
        if p.exists() and p not in logs:
            logs.append(p)
    summary["files_found"] = [p.name for p in logs]

    if not logs:
        summary["headline"] = (
            "No ORCA output found in the directory. The run may not have "
            "started, or output files use an unexpected name."
        )
        return summary

    log_path = logs[-1]
    parsed = _summarize_with_cclib(log_path)
    summary["parse"] = parsed
    if "error" not in parsed:
        summary["scf_energy"] = parsed.get("scf_energy_eV")
        summary["free_energy"] = parsed.get("free_energy_hartree")
        summary["imaginary_freqs"] = parsed.get("n_imaginary_freqs")
        summary["converged"] = parsed.get("converged")
        if parsed.get("converged") is True:
            summary["convergence_status"] = "converged"
        elif parsed.get("converged") is False:
            summary["convergence_status"] = "not_converged"
        elif parsed.get("scf_energy_eV") is not None:
            # Single-point with an energy but no optimization to converge.
            summary["convergence_status"] = "converged"

    log_text = _read_tail(log_path)
    if log_text:
        summary["terminated_normally"] = _ORCA_SUCCESS_MARK.lower() in log_text.lower()
        summary["log_error_hints"] = _classify_log_errors(log_text)
        if summary["convergence_status"] == "unknown" and summary["log_error_hints"]:
            summary["convergence_status"] = "failed"
        # A run that never printed the normal-termination banner and has no
        # parsed energy has almost certainly crashed.
        if (summary["terminated_normally"] is False
                and summary["convergence_status"] == "unknown"):
            summary["convergence_status"] = "failed"

    status = summary["convergence_status"]
    if status == "converged":
        summary["headline"] = "Run completed; SCF/geometry converged."
    elif status == "not_converged":
        summary["headline"] = (
            "Optimization did NOT converge — check the geometry, increase "
            "iterations, or revisit the method/basis."
        )
    elif status == "failed":
        summary["headline"] = (
            "Run appears to have failed (errors found, or ORCA did not "
            "terminate normally). See log_error_hints for known patterns."
        )
    else:
        summary["headline"] = (
            "Convergence status could not be determined (cclib unavailable or "
            "output incomplete). Install cclib and confirm the output log."
        )
    return summary


TOOL_SPEC = ToolSpec(
    name="snapshot_run",
    description=(
        "Read an ORCA run directory and return a structured snapshot of "
        "convergence status, energetics (SCF energy, enthalpy, free energy), "
        "imaginary-frequency count, whether ORCA terminated normally, and any "
        "matched error patterns. Parsed via cclib. Discovered and dispatched "
        "when the ``orca`` skill is active."
    ),
    parameters={
        "output_dir": {
            "type": "string",
            "description": (
                "Absolute path to the directory containing the ORCA run's "
                "output files (*.out / *.log / stdout)."
            ),
        },
    },
    required=["output_dir"],
    signature="snapshot_run(output_dir: str) -> dict",
    import_line="from scilink.skills.molecular_qc.orca.orca_output import snapshot_run",
    agents=["simulation"],
    returns=(
        "dict with status, files_found, parse (convergence + energetics), "
        "convenience aliases (scf_energy, free_energy, imaginary_freqs, "
        "converged), terminated_normally, log_error_hints, convergence_status, "
        "and a one-line headline."
    ),
)
