"""GROMACS tools module — deterministic `.mdp` parsing and validation.

Registered in ``MDSimulationAgent.TOOL_REGISTRY`` under ``"gromacs"`` (the LAMMPS
twin: ``molecular_dynamics/lammps/lammps.py``). The MD agent calls
``validate_script(script_path, system_info)`` after generating the run-control
file and hands the result to the refinement loop, exactly as for LAMMPS.

GROMACS's primary generated artifact is the ``.mdp`` run-control file; the
topology (.top) and coordinates (.gro) are supplied alongside it. So the checks
here are ``.mdp``-shaped: a well-formed integrator/timestep, a thermostat, a
consistent NVT-vs-NPT pressure-coupling block, and sane cutoffs/electrostatics.
No GROMACS binary is needed — the deck is checked, not run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

# GROMACS treats '-' and '_' in .mdp option names as equivalent; we normalize
# both to '-' so `nstxout_compressed` and `nstxout-compressed` are one key.
_THERMOSTATS = {"berendsen", "nose-hoover", "v-rescale", "andersen",
                "andersen-massive"}
_BAROSTATS = {"berendsen", "parrinello-rahman", "c-rescale", "mttk"}
_OFF = {"no", "none", ""}


def parse_mdp(text: str) -> Dict[str, str]:
    """Parse a `.mdp` into a normalized ``{option: value}`` dict.

    Keys are lowercased with '_' → '-'; comments (``;`` to end of line) are
    stripped; the last assignment of a repeated key wins (GROMACS semantics).
    """
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip().lower().replace("_", "-")] = val.strip()
    return out


def _num(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val.split()[0])
    except (ValueError, IndexError):
        return None


def _n_fields(val: Optional[str]) -> int:
    return len(val.split()) if val else 0


def validate_script(script_path: str,
                    system_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate a GROMACS `.mdp` run-control file structurally.

    Checks: integrator present; timestep present, numeric, and consistent with
    the constraint setting; a thermostat for a dynamics run; a well-formed
    NVT-vs-NPT pressure-coupling block; ref-t / tc-grps field-count agreement;
    and cutoff/electrostatics presence. Returns the same shape as the LAMMPS
    validator so the refinement loop consumes it identically.
    """
    result: Dict[str, Any] = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "integrator": None,
        "dt": None,
        "nsteps": None,
        "ensemble": None,
        "thermostat": None,
        "barostat": None,
    }
    try:
        content = Path(script_path).read_text()
    except Exception as e:
        return {**result, "valid": False, "errors": [f"Cannot read: {e}"]}

    mdp = parse_mdp(content)
    errors = result["errors"]
    warnings = result["warnings"]

    integrator = mdp.get("integrator")
    result["integrator"] = integrator
    if not integrator:
        errors.append("no `integrator` set (expected md / md-vv / steep / cg)")
    is_minimizer = (integrator or "").lower() in ("steep", "cg", "l-bfgs")

    dt = _num(mdp.get("dt"))
    result["dt"] = dt
    if not is_minimizer:
        if "dt" not in mdp:
            errors.append("no `dt` (timestep) set for a dynamics run")
        elif dt is None:
            errors.append(f"`dt` is not numeric: {mdp.get('dt')!r}")
        elif dt > 0.005:
            errors.append(f"`dt` = {dt} ps is unphysically large for atomistic MD")
        else:
            constraints = mdp.get("constraints", "none").lower()
            if dt > 0.002 and constraints in _OFF:
                warnings.append(
                    f"`dt` = {dt} ps without bond constraints — use "
                    "`constraints = h-bonds` for 2 fs, or reduce dt to 0.001")

    if "nsteps" not in mdp:
        warnings.append("no `nsteps` set (run length)")
    else:
        result["nsteps"] = mdp.get("nsteps")

    if not is_minimizer:
        tcoupl = mdp.get("tcoupl", "no").lower()
        result["thermostat"] = tcoupl if tcoupl not in _OFF else None
        if tcoupl in _OFF:
            warnings.append("no thermostat (`tcoupl`) on a dynamics run — the "
                            "ensemble is unthermostatted (NVE)")
        elif tcoupl not in _THERMOSTATS:
            warnings.append(f"`tcoupl = {tcoupl}` is not a recognized thermostat")
        # ref-t / tau-t must have one value per tc-grps group.
        n_grps = _n_fields(mdp.get("tc-grps"))
        n_reft = _n_fields(mdp.get("ref-t"))
        if n_grps and n_reft and n_grps != n_reft:
            errors.append(
                f"`ref-t` has {n_reft} value(s) but `tc-grps` has {n_grps} "
                "group(s) — they must match")
        elif tcoupl not in _OFF and n_reft == 0:
            warnings.append("thermostat on but no `ref-t` (target temperature)")

    pcoupl = mdp.get("pcoupl", "no").lower()
    result["barostat"] = pcoupl if pcoupl not in _OFF else None
    if pcoupl not in _OFF:
        result["ensemble"] = "NPT"
        if pcoupl not in _BAROSTATS:
            warnings.append(f"`pcoupl = {pcoupl}` is not a recognized barostat")
        for req in ("ref-p", "tau-p", "compressibility"):
            if req not in mdp:
                errors.append(f"pressure coupling on but `{req}` is missing")
    elif not is_minimizer:
        result["ensemble"] = "NVT" if result["thermostat"] else "NVE"

    if not is_minimizer:
        if mdp.get("cutoff-scheme", "").lower() != "verlet":
            warnings.append("`cutoff-scheme` should be `Verlet` on modern GROMACS")
        if "coulombtype" not in mdp:
            warnings.append("no `coulombtype` set (expected PME for "
                            "explicit-solvent systems)")
        if "rvdw" not in mdp or "rcoulomb" not in mdp:
            warnings.append("missing `rvdw` / `rcoulomb` cutoff(s)")

    result["valid"] = len(errors) == 0
    return result
