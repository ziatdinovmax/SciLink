"""
Remote HPC environment probing.

Detects available simulation software, schedulers, container runtimes,
and filesystem layout on a connected cluster.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .connection import HPCConnection


# Software families we know how to detect (module keywords → canonical name)
_MODULE_PATTERNS: dict[str, list[str]] = {
    "vasp": ["vasp"],
    "lammps": ["lammps"],
    "gromacs": ["gromacs", "gmx"],
    "quantum_espresso": ["quantum-espresso", "qe", "espresso"],
    "cp2k": ["cp2k"],
    "nwchem": ["nwchem"],
    "orca": ["orca"],
    "gaussian": ["gaussian", "g16", "g09"],
}

# Binaries to look for in PATH
_BINARY_CHECKS: dict[str, list[str]] = {
    "vasp": ["vasp_std", "vasp_gam", "vasp_ncl"],
    "lammps": ["lmp", "lmp_mpi", "lmp_serial", "lammps"],
    "gromacs": ["gmx", "gmx_mpi", "mdrun"],
    "quantum_espresso": ["pw.x", "ph.x"],
    "cp2k": ["cp2k.psmp", "cp2k.popt", "cp2k.sopt"],
    "nwchem": ["nwchem"],
    "orca": ["orca"],
}

_CONTAINER_RUNTIMES = ["podman", "docker", "singularity", "apptainer"]


@dataclass
class HPCEnvironment:
    """Snapshot of what's available on a remote cluster."""

    home: str = ""
    scratch: Optional[str] = None
    python_path: Optional[str] = None
    scheduler_type: Optional[str] = None

    # General software inventory: {canonical_name: [module_names]}
    available_software: dict[str, list[str]] = field(default_factory=dict)
    # Binaries found in PATH: {canonical_name: [binary_paths]}
    available_binaries: dict[str, list[str]] = field(default_factory=dict)

    container_runtimes: list[str] = field(default_factory=list)

    # Legacy fields (kept for backward compat with sim_workflow.py)
    lammps_binaries: list[str] = field(default_factory=list)
    lammps_modules: list[str] = field(default_factory=list)


def probe_remote(conn: HPCConnection) -> HPCEnvironment:
    """
    Probe a connected cluster for simulation software and environment details.

    Runs a handful of non-destructive commands over SSH and returns an
    ``HPCEnvironment`` summarizing what's available.
    """
    env = HPCEnvironment()

    # ── Home directory ────────────────────────────────────────
    env.home = conn.home_dir()

    # ── Scratch directory (common conventions) ────────────────
    for scratch_var in ("SCRATCH", "SCRATCHDIR", "TMPDIR", "MEMBERWORK"):
        stdout, _, rc = conn.run(f"echo ${scratch_var}")
        val = stdout.strip()
        if rc == 0 and val and val != f"${scratch_var}":
            env.scratch = val
            break

    # ── Python ────────────────────────────────────────────────
    stdout, _, rc = conn.run("which python3 2>/dev/null || which python 2>/dev/null")
    if rc == 0 and stdout.strip():
        env.python_path = stdout.strip().split("\n")[0]

    # ── Module-based software detection ───────────────────────
    # 'module avail' writes to stderr on most systems
    _, stderr, rc = conn.run("module avail 2>&1 1>/dev/null", timeout=15)
    # Some systems write to stdout instead
    stdout_avail, _, _ = conn.run("module avail 2>/dev/null", timeout=15)
    module_text = (stderr + "\n" + stdout_avail).lower()

    for software, keywords in _MODULE_PATTERNS.items():
        matches = []
        for kw in keywords:
            # Match module names like "vasp/6.3.2" or "lammps/20230802"
            found = re.findall(
                rf"({re.escape(kw)}[\w./-]*)", module_text, re.I,
            )
            matches.extend(found)
        if matches:
            # Deduplicate and sort
            unique = sorted(set(matches))
            env.available_software[software] = unique

    # ── Binary detection ──────────────────────────────────────
    all_binaries = []
    for bins in _BINARY_CHECKS.values():
        all_binaries.extend(bins)
    # Batch into a single 'which' call for efficiency
    which_cmd = "which " + " ".join(all_binaries) + " 2>/dev/null"
    stdout, _, _ = conn.run(which_cmd, timeout=10)
    found_paths = {
        line.strip().split("/")[-1]: line.strip()
        for line in stdout.strip().split("\n")
        if line.strip() and "/" in line
    }

    for software, bins in _BINARY_CHECKS.items():
        found = [found_paths[b] for b in bins if b in found_paths]
        if found:
            env.available_binaries[software] = found

    # ── Container runtimes ────────────────────────────────────
    which_cmd = "which " + " ".join(_CONTAINER_RUNTIMES) + " 2>/dev/null"
    stdout, _, _ = conn.run(which_cmd, timeout=5)
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line and "/" in line:
            name = line.split("/")[-1]
            if name in _CONTAINER_RUNTIMES:
                env.container_runtimes.append(name)

    # ── Scheduler detection ───────────────────────────────────
    for sched_name, cmd in [
        ("SLURM", "sinfo --version"),
        ("PBS", "qstat --version"),
        ("LSF", "bsub -V"),
    ]:
        _, _, rc = conn.run(f"{cmd} 2>/dev/null", timeout=5)
        if rc == 0:
            env.scheduler_type = sched_name
            break

    # ── Backward-compat: populate legacy LAMMPS fields ────────
    env.lammps_binaries = env.available_binaries.get("lammps", [])
    env.lammps_modules = env.available_software.get("lammps", [])

    return env
