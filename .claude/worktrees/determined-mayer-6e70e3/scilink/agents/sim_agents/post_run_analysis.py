"""
Post-run VASP output analysis.

Reads OUTCAR / vasprun.xml from a completed (or failed) VASP run and
produces a structured summary the simulate orchestrator can hand back
to the user. This is the part of the orchestrator that closes the loop
after the user runs VASP elsewhere — analyze mode never does this.

Designed to be defensive: VASP runs fail in many ways and pymatgen's
parser can raise on partial / corrupt outputs. Every code path returns
a dict with a `status` key; failures are reported, not raised.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Return human-readable hints matching known VASP error patterns."""
    hits: List[str] = []
    for pattern, hint in _VASP_ERROR_HINTS:
        if pattern.lower() in log_text.lower():
            hits.append(f"{pattern}: {hint}")
    return hits


def _read_first_existing(*paths: Path, max_chars: int = 80_000) -> Optional[str]:
    """Read the first path that exists, truncated to max_chars.

    Used for stdout/stderr style logs where we just want the tail of a
    text file. Returns None if none exist or all fail to read.
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
    """Parse vasprun.xml via pymatgen and pull out convergence + energetics.

    Returns a dict with:
        converged_electronic, converged_ionic, converged (overall),
        final_energy, n_ionic_steps, n_electronic_steps_last,
        max_force, parameter notes — or an `error` key on parse failure.
    """
    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except ImportError as e:
        return {"error": f"pymatgen not available: {e}"}

    try:
        # parse_dos=False, parse_eigen=False for speed; we only need scalars.
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

    # Max force on last ionic step — needs forces from the last step's
    # output; pymatgen surfaces this on Vasprun.ionic_steps[-1]["forces"]
    try:
        if vr.ionic_steps:
            forces = vr.ionic_steps[-1].get("forces")
            if forces is not None:
                import numpy as np
                f = np.asarray(forces)
                out["max_force_eV_per_A"] = float(np.linalg.norm(f, axis=1).max())
    except Exception as e:
        out["force_check_error"] = str(e)

    # Parameter snapshot — the run's actual INCAR settings as VASP saw
    # them (after defaults), useful when the user wants to reconcile
    # what they asked for vs what ran.
    try:
        params = vr.incar.as_dict() if vr.incar else {}
        # Trim huge auto-filled keys; surface a stable subset
        keys = ["IBRION", "NSW", "ISIF", "EDIFF", "EDIFFG", "ENCUT",
                "ISMEAR", "SIGMA", "ALGO", "PREC", "NELM", "ISYM",
                "LREAL", "ISPIN"]
        out["incar_snapshot"] = {k: params[k] for k in keys if k in params}
    except Exception as e:
        out["incar_snapshot_error"] = str(e)

    return out


def analyze_run_directory(output_dir: str) -> Dict[str, Any]:
    """Top-level entry: summarize a VASP run directory.

    Returns a structured dict with status + diagnostics. Always returns
    something parseable; never raises out of expected failure modes.

    Looks for (in order):
        - vasprun.xml  → parsed via pymatgen for convergence/energetics
        - OUTCAR / OSZICAR / vasp.out / stdout / stderr → tail-read for
          error pattern matching when the structured output is missing

    Args:
        output_dir: directory containing the VASP run's outputs.
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

    # Inventory key files
    candidates = ["vasprun.xml", "OUTCAR", "OSZICAR", "CONTCAR",
                  "vasp.out", "stdout", "stdout.log", "stderr",
                  "stderr.log"]
    for name in candidates:
        if (out_dir / name).exists():
            summary["files_found"].append(name)

    # Structured parse via vasprun.xml
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

    # Pattern-match the stdout / stderr / OUTCAR for known failure modes
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
        # If vasprun told us nothing and the log looks fatal, flag that
        if summary["convergence_status"] == "unknown" and summary["log_error_hints"]:
            summary["convergence_status"] = "failed"

    # Coarse top-line judgment for the LLM to read at a glance
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
