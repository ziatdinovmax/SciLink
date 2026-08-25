"""VASP output snapshot parser.

Reads OUTCAR / vasprun.xml from a completed (or failed) VASP run and
produces a structured summary that simulate-mode critics hand to the LLM
when assessing the run. Discovered via the skill registry when the
``vasp`` skill is active and called through
:func:`scilink.skills._shared._registry.get_tool_function`.

All code paths return a dict with a ``status`` key; failures are reported
in the returned structure rather than raised, so callers can hand the
result to the LLM uniformly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..._shared._spec import ToolSpec

logger = logging.getLogger(__name__)


# Common VASP error patterns surfaced via the stdout/stderr log so the
# summary can flag known failure modes even when vasprun.xml is missing
# or unparseable.
_VASP_ERROR_HINTS = [
    ("ZBRENT", "ionic optimization stuck — try IBRION=1 or tighten POTIM"),
    ("NELM", "electronic SCF failed to converge — increase NELM, tighten EDIFF, try ALGO=All"),
    ("VERY BAD NEWS", "fatal numerical failure — usually indicates a bad starting geometry or symmetry mismatch"),
    ("Therefore set LREAL=.FALSE.", "real-space projection unstable — set LREAL=.FALSE."),
    ("internal error in subroutine SGRCON", "symmetry detection failure — try ISYM=0 or perturb the structure slightly"),
    ("inverse of rotation matrix", "symmetry / lattice issue — try ISYM=0"),
    ("SBESSELITER", "SCF instability — try ALGO=Normal/All or smaller AMIX"),
    ("EDDDAV: Call to ZHEGV failed", "Davidson algorithm failed — try ALGO=All or increase NBANDS"),
    ("SGCC: nion", "ion mismatch between INCAR and POSCAR — check selective dynamics block"),
    ("WARNING: stress tensor not correct", "stress incomplete — usually because ISIF or NSW is wrong for what you're computing"),
]


def _classify_log_errors(log_text: str) -> List[str]:
    """Return human-readable hints matching known VASP error patterns.

    Args:
        log_text: Tail-read text of stdout / stderr / OUTCAR.

    Returns:
        List of ``"<pattern>: <hint>"`` strings for each pattern matched
        case-insensitively in ``log_text``.
    """
    hits: List[str] = []
    for pattern, hint in _VASP_ERROR_HINTS:
        if pattern.lower() in log_text.lower():
            hits.append(f"{pattern}: {hint}")
    return hits


def _read_first_existing(*paths: Path, max_chars: int = 80_000) -> Optional[str]:
    """Read the first existing path, returning at most ``max_chars`` of its tail.

    Used for stdout / stderr style logs where only the trailing region
    typically contains the failure signature.

    Args:
        *paths: Candidate file paths, tried in order.
        max_chars: Maximum number of characters to return; the tail of
            the file is preferred when the file exceeds this size.

    Returns:
        The file's text (tail-trimmed) for the first readable path, or
        ``None`` if none exist or all reads fail.
    """
    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")
            continue
        if len(text) > max_chars:
            return text[-max_chars:]
        return text
    return None


def _summarize_vasprun(vasprun_path: Path) -> Dict[str, Any]:
    """Parse ``vasprun.xml`` and extract convergence + energetics.

    Args:
        vasprun_path: Path to a vasprun.xml file.

    Returns:
        A dict containing some subset of: ``converged_electronic``,
        ``converged_ionic``, ``converged``, ``final_energy``,
        ``n_ionic_steps``, ``n_electronic_steps_last_ionic``,
        ``max_force_eV_per_A``, ``incar_snapshot``. On parse failure
        returns ``{"error": "..."}``. Partial parse failures populate
        the available fields and surface the missing ones under
        per-field ``*_error`` keys.
    """
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except ImportError as e:
        return {"error": f"pymatgen not available: {e}"}

    try:
        vr = Vasprun(
            str(vasprun_path),
            parse_dos=False,
            parse_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=False,
        )
    except Exception as e:
        return {"error": f"vasprun.xml parse failed: {e}"}

    out: Dict[str, Any] = {}
    try:
        out["converged_electronic"] = bool(vr.converged_electronic)
        out["converged_ionic"] = bool(vr.converged_ionic)
        out["converged"] = bool(vr.converged)
    except Exception as e:
        out["convergence_check_error"] = str(e)

    try:
        out["final_energy"] = float(vr.final_energy)
    except Exception:
        out["final_energy"] = None

    try:
        out["n_ionic_steps"] = len(vr.ionic_steps)
        if vr.ionic_steps:
            last = vr.ionic_steps[-1]
            scs = last.get("electronic_steps", []) if isinstance(last, dict) else []
            out["n_electronic_steps_last_ionic"] = len(scs)
        else:
            out["n_electronic_steps_last_ionic"] = 0
    except Exception as e:
        out["ionic_step_error"] = str(e)

    try:
        if vr.ionic_steps:
            forces = vr.ionic_steps[-1].get("forces")
            if forces is not None:
                import numpy as np
                f = np.asarray(forces)
                out["max_force_eV_per_A"] = float(np.linalg.norm(f, axis=1).max())
    except Exception as e:
        out["force_check_error"] = str(e)

    try:
        params = vr.incar.as_dict() if vr.incar else {}
        keys = ["IBRION", "NSW", "ISIF", "EDIFF", "EDIFFG", "ENCUT",
                "ISMEAR", "SIGMA", "ALGO", "PREC", "NELM", "ISYM",
                "LREAL", "ISPIN"]
        out["incar_snapshot"] = {k: params[k] for k in keys if k in params}
    except Exception as e:
        out["incar_snapshot_error"] = str(e)

    return out


def snapshot_run(output_dir: str) -> Dict[str, Any]:
    """Summarize a VASP run directory into a structured snapshot.

    Inspects the directory for ``vasprun.xml`` (preferred, structured
    parse via pymatgen) and falls back to OUTCAR / OSZICAR / stdout /
    stderr tail-matching for known VASP error patterns when the
    structured output is missing or unparseable.

    Args:
        output_dir: Path to the directory containing the VASP run's
            output files.

    Returns:
        A dict with fields:

            status              ``"ok"`` or ``"error"``
            output_directory    The directory inspected
            files_found         List of recognized output files present
            vasprun             Dict from :func:`_summarize_vasprun`
                                when ``vasprun.xml`` was parseable;
                                ``None`` otherwise
            log_error_hints     List of ``"<pattern>: <hint>"`` strings
                                matched in log files
            convergence_status  One of ``"converged"``, ``"not_converged"``,
                                ``"failed"``, ``"unknown"``
            headline            One-sentence top-line assessment for
                                quick LLM consumption
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
        "vasprun": None,
        "log_error_hints": [],
        "convergence_status": "unknown",
    }

    candidates = ["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR",
                  "vasp.out", "stdout", "stdout.log", "stderr",
                  "stderr.log"]
    for name in candidates:
        if (out_dir / name).exists():
            summary["files_found"].append(name)

    vasprun_path = out_dir / "vasprun.xml"
    if vasprun_path.exists():
        vr_summary = _summarize_vasprun(vasprun_path)
        summary["vasprun"] = vr_summary
        if "error" not in vr_summary:
            converged = vr_summary.get("converged")
            if converged is True:
                summary["convergence_status"] = "converged"
            elif converged is False:
                summary["convergence_status"] = "not_converged"

    log_text = _read_first_existing(
        out_dir / "vasp.out",
        out_dir / "stdout",
        out_dir / "stdout.log",
        out_dir / "stderr",
        out_dir / "stderr.log",
        out_dir / "OUTCAR",
    )
    if log_text:
        summary["log_error_hints"] = _classify_log_errors(log_text)
        if summary["convergence_status"] == "unknown" and summary["log_error_hints"]:
            summary["convergence_status"] = "failed"

    if summary["convergence_status"] == "converged":
        summary["headline"] = "Run converged successfully."
    elif summary["convergence_status"] == "not_converged":
        summary["headline"] = (
            "Run did NOT reach convergence within the requested limits. "
            "Check NELM/NSW and the convergence criteria; consider "
            "tightening EDIFF / loosening EDIFFG, or trying ALGO=All."
        )
    elif summary["convergence_status"] == "failed":
        summary["headline"] = (
            "Run appears to have failed (errors found in the log). "
            "See log_error_hints for known patterns."
        )
    else:
        summary["headline"] = (
            "Convergence status could not be determined from the available "
            "outputs. The run may still be in progress, or critical files "
            "(vasprun.xml, OUTCAR) may be missing."
        )

    return summary


TOOL_SPEC = ToolSpec(
    name="snapshot_run",
    description=(
        "Read a VASP run directory and return a structured snapshot of "
        "convergence status, energetics, and any matched error patterns. "
        "Discovered and dispatched when the ``vasp`` skill is active."
    ),
    parameters={
        "output_dir": {
            "type": "string",
            "description": (
                "Absolute path to the directory containing the VASP run's "
                "output files (vasprun.xml, OUTCAR, OSZICAR, stdout, "
                "stderr)."
            ),
        },
    },
    required=["output_dir"],
    signature="snapshot_run(output_dir: str) -> dict",
    import_line="from scilink.skills.periodic_dft.vasp.vasp_output import snapshot_run",
    agents=["simulation"],
    returns=(
        "dict with status, files_found, vasprun (convergence + energetics), "
        "log_error_hints (matched VASP failure patterns), "
        "convergence_status, and a one-line headline."
    ),
)
