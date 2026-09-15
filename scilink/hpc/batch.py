"""Build a scheduler batch script for a single run command.

Engine-neutral and scheduler-neutral: the scheduler-specific directive lines
come from ``scheduler.batch_directives(resources)``, so this module knows
nothing about SLURM vs PBS vs LSF, and the run command and resources arrive as
data. The wrapper captures stdout, stderr, and the exit code into fixed,
engine-neutral filenames so a downstream consumer (e.g. the refinement loop's
post-run critic) reads a run's outputs the same way no matter where it ran.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from scilink.hpc.scheduler import Scheduler


def build_batch_script(
    scheduler: "Scheduler",
    run_command: str,
    *,
    resources: Optional[dict] = None,
    setup: Optional[List[str]] = None,
    stdout_file: str = "run_stdout.log",
    stderr_file: str = "run_stderr.log",
    rc_file: str = "run_returncode.txt",
) -> str:
    """Assemble a submit-ready batch script.

    Args:
        scheduler: The scheduler whose directive syntax to use; supplies the
            ``#SBATCH`` / ``#PBS`` / ``#BSUB`` header from ``resources``.
        run_command: The command to run in the job's working directory.
        resources: Engine-neutral resource request forwarded to
            ``scheduler.batch_directives`` (partition, nodes, time, …).
        setup: Shell lines run before the command (e.g. ``module load`` lines,
            ``conda activate``, env exports).
        stdout_file, stderr_file, rc_file: Filenames the wrapper writes the
            command's stdout, stderr, and exit code to, in the job's working
            directory. Default to the engine-neutral names the local executor
            uses, so the post-run snapshot is identical across executors.

    Returns:
        The batch script text (newline-terminated).
    """
    lines: List[str] = ["#!/bin/bash"]
    lines.extend(scheduler.batch_directives(resources or {}))
    lines.append("")
    # Move to the submit directory before anything writes relative paths —
    # a no-op for SLURM/LSF, a `cd $PBS_O_WORKDIR` for PBS/Torque.
    prelude = scheduler.workdir_prelude()
    if prelude:
        lines.extend(prelude)
        lines.append("")
    for line in setup or []:
        lines.append(line)
    if setup:
        lines.append("")
    # Capture stdout/stderr/returncode into fixed filenames so the post-run
    # critic reads outputs identically regardless of where the run happened.
    lines.append(f"{run_command} > {stdout_file} 2> {stderr_file}")
    lines.append(f"echo $? > {rc_file}")
    return "\n".join(lines) + "\n"
