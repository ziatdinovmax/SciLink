"""Run refinement-loop phases on an HPC scheduler.

``ClusterExecutor`` is the cluster counterpart to ``LocalExecutor``: it
satisfies the same :class:`~scilink.agents.sim_agents.refinement.Executor`
contract, so the refinement loop drives it without knowing a run happens on a
remote scheduler rather than a local subprocess. A ``run`` uploads the phase's
input files, submits a batch job, polls until the job reaches a terminal state,
downloads the outputs back into the local ``run_dir``, and persists the same
engine-neutral ``stdout`` / ``stderr`` / ``returncode`` files the local executor
writes — so the post-run critic's snapshot is identical regardless of executor.

This lives with the loop (not in ``scilink.hpc``) because it implements the
loop's ``Executor`` contract; ``scilink.hpc`` stays self-contained and is
imported lazily, so importing the loop does not pull in paramiko.

v1 runs one job per ``run`` call and blocks until it finishes. The refinement
loop calls ``run`` serially per phase (and per fan-out member), so a campaign's
parallel members are submitted one after another. Concurrent submission across
members is a future extension (a batch ``run_many`` seam) and does not change
this contract.
"""
from __future__ import annotations

import logging
import stat as _stat
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .refinement import Executor, LocalExecutor

if TYPE_CHECKING:
    from scilink.hpc.connection import HPCConnection
    from scilink.hpc.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Reuse the local executor's engine-neutral output filenames so a cluster run's
# snapshot is byte-for-byte the same shape the critic already reads.
_STDOUT = LocalExecutor.STDOUT_FILE
_STDERR = LocalExecutor.STDERR_FILE
_RC = LocalExecutor.RETURNCODE_FILE


class ClusterExecutor(Executor):
    """Execute inputs as a scheduler batch job on a remote host.

    Attributes:
        conn: A connected (or connectable) ``HPCConnection``.
        remote_root: Base remote directory under which per-run directories are
            created. Defaults to the remote home directory.
        resources: Engine-neutral resource request for the batch directives
            (partition, nodes, ntasks, time, account, gres, extra_directives).
        setup: Shell lines run before the command (module loads, env setup).
        poll_interval: Seconds between job-status polls.
        timeout: Overall wall-clock cap (seconds) before the job is cancelled.
    """

    def __init__(
        self,
        connection: "HPCConnection",
        *,
        scheduler: "Scheduler | None" = None,
        remote_root: Optional[str] = None,
        resources: Optional[dict] = None,
        setup: Optional[List[str]] = None,
        poll_interval: int = 30,
        timeout: int = 86400,
        job_script_name: str = "scilink_job.sh",
        cancel_check: Optional[Callable[[], bool]] = None,
    ):
        self.conn = connection
        self._scheduler = scheduler
        self.remote_root = remote_root
        self.resources = resources or {}
        self.setup = setup or []
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.job_script_name = job_script_name
        # "Should I stop?" hook. When it returns True mid-poll, the submitted
        # job is cancelled on the scheduler instead of being left to run — so
        # a Stop does not orphan the allocation. A KeyboardInterrupt during
        # the wait is handled the same way. By default it is the waiting
        # thread's own cancel (``log_context.is_cancelled``: a swarm item's
        # budget or memory cancel, a fan-out branch's, the turn's Stop), so
        # a cancelled item never leaves its jobs running on the cluster; a
        # caller's own hook replaces it.
        if cancel_check is None:
            from scilink.utils.log_context import is_cancelled
            cancel_check = is_cancelled
        self.cancel_check = cancel_check

    @classmethod
    def connect(
        cls,
        *,
        hostname: str,
        username: str,
        password: str = "",
        key_path: str = "",
        key_passphrase: str = "",
        auth_method: Optional[str] = None,
        proxy_jump: str = "",
        port: int = 22,
        **executor_kwargs: Any,
    ) -> "ClusterExecutor":
        """Open an HPC connection and return a ready ``ClusterExecutor``.

        Convenience constructor that assembles the ``HPCProfile`` /
        ``HPCConnection``, opens the connection, and wraps it — so a caller
        passes one call to ``run_complete_workflow(..., executor=...)`` instead
        of hand-assembling the connection. ``auth_method`` defaults to
        ``"password"`` when a password is given, else ``"key"``. Remaining
        keyword arguments (``remote_root``, ``resources``, ``setup``,
        ``poll_interval``, ``timeout``) are forwarded to the constructor.

        Args:
            hostname: SSH host of the cluster login node.
            username: SSH username.
            password: Password for password auth (omit for key auth).
            key_path: Path to a private key for key auth.
            key_passphrase: Passphrase for an encrypted key.
            auth_method: ``"password"`` or ``"key"``; inferred when omitted.
            proxy_jump: Optional ``user@bastion`` jump host.
            port: SSH port.

        Returns:
            A connected ``ClusterExecutor``.
        """
        from scilink.hpc.connection import HPCConnection, HPCProfile
        method = auth_method or ("password" if password else "key")
        profile = HPCProfile(
            name=hostname, hostname=hostname, username=username, port=port,
            auth_method=method, key_path=key_path, proxy_jump=proxy_jump,
        )
        conn = HPCConnection(profile)
        conn.connect(password=password, key_passphrase=key_passphrase)
        return cls(conn, **executor_kwargs)

    # ── internals ─────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self.conn.is_connected:
            return
        # A key/agent session can re-establish from an arg-less connect(); a
        # password session cannot (the password is intentionally not stored).
        # Surface that as a clear error rather than a bare paramiko auth failure.
        try:
            self.conn.connect()
        except Exception as exc:  # noqa: BLE001 — re-raise with guidance
            raise ConnectionError(
                "HPC connection is down and could not be re-established "
                "non-interactively (password sessions cannot self-recover). "
                "Pass an already-connected HPCConnection, e.g. via "
                "ClusterExecutor.connect(...)."
            ) from exc

    def _resolve_scheduler(self) -> "Scheduler":
        if self._scheduler is None:
            from scilink.hpc import detect_scheduler
            self._scheduler = detect_scheduler(self.conn)
            if self._scheduler is None:
                raise RuntimeError(
                    "No supported scheduler (SLURM/PBS/LSF) detected on the "
                    "remote host."
                )
        return self._scheduler

    def _remote_dir_for(self, run_dir: str) -> str:
        base = (self.remote_root or self.conn.home_dir()).rstrip("/")
        return f"{base}/{Path(run_dir).name}"

    def _download_outputs(self, remote_dir: str, local: Path) -> None:
        """Download every regular file in the remote run dir to ``local``.

        Subdirectories are skipped — a phase's run directory is flat (the loop
        gives each phase / fan-out member its own directory).
        """
        try:
            entries = self.conn.listdir(remote_dir)
        except Exception as exc:  # noqa: BLE001 — best-effort retrieval
            logger.warning("ClusterExecutor: could not list %s: %s", remote_dir, exc)
            return
        for attr in entries:
            name = attr.filename
            if name in (".", ".."):
                continue
            if _stat.S_ISDIR(attr.st_mode):
                continue
            try:
                self.conn.download(f"{remote_dir}/{name}", str(local / name))
            except Exception as exc:  # noqa: BLE001 — one bad file shouldn't abort
                logger.warning("ClusterExecutor: could not download %s: %s", name, exc)

    # ── submit / poll (the non-blocking seam) ─────────────────

    def submit(
        self, input_files: Dict[str, str], run_command: str, run_dir: str
    ) -> Dict[str, Any]:
        """Materialize + upload inputs, submit the batch job, return a handle.

        The handle carries what :meth:`poll` / :meth:`run` need to follow the
        job up later: ``{"job_id", "remote_dir", "run_path"}``. On a submission
        failure it returns ``{"error", "remote_dir", "run_path"}`` with no
        ``job_id`` and has already persisted the engine-neutral stderr/rc files.
        This is the seam a non-blocking caller (one scheduler, many in-flight
        members) submits through; :meth:`run` keeps the submit-then-wait
        contract on top of it.
        """
        from scilink.hpc.batch import build_batch_script

        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        # Materialize inputs locally too — gives the local snapshot the inputs
        # and serves as the upload source.
        for name, contents in (input_files or {}).items():
            (run_path / name).write_text(contents, encoding="utf-8")

        self._ensure_connected()
        sched = self._resolve_scheduler()
        remote_dir = self._remote_dir_for(run_dir)
        self.conn.mkdir_p(remote_dir)

        for name in (input_files or {}):
            self.conn.upload(str(run_path / name), f"{remote_dir}/{name}")

        script = build_batch_script(
            sched, run_command,
            resources=self.resources, setup=self.setup,
            stdout_file=_STDOUT, stderr_file=_STDERR, rc_file=_RC,
        )
        (run_path / self.job_script_name).write_text(script, encoding="utf-8")
        self.conn.upload(
            str(run_path / self.job_script_name),
            f"{remote_dir}/{self.job_script_name}",
        )

        try:
            job_id = sched.submit(self.job_script_name, work_dir=remote_dir)
        except Exception as exc:  # noqa: BLE001 — surface as an executor error
            (run_path / _STDERR).write_text(f"Job submission failed: {exc}", encoding="utf-8")
            (run_path / _RC).write_text("submit_error", encoding="utf-8")
            return {
                "error": f"Job submission failed: {exc}",
                "remote_dir": remote_dir, "run_path": run_path,
            }
        logger.info("ClusterExecutor: submitted job %s in %s", job_id, remote_dir)
        return {"job_id": job_id, "remote_dir": remote_dir, "run_path": run_path}

    def poll(self, handle: Dict[str, Any]) -> Dict[str, Any]:
        """One status check for a submitted job — no sleep, no download.

        Returns ``{"terminal": bool, "job": <job status object>}``. The caller
        decides cadence and when to download; :meth:`run` polls in a loop.
        """
        self._ensure_connected()
        sched = self._resolve_scheduler()
        job = sched.status(handle["job_id"])
        return {"terminal": job.status.is_terminal, "job": job}

    def _finalize(self, remote_dir: str, run_path: Path, job: Any) -> Dict[str, Any]:
        """Download outputs and build the completed-run record for a terminal job."""
        self._download_outputs(remote_dir, run_path)
        # Prefer the returncode the wrapper recorded; fall back to the
        # scheduler's reported exit code.
        returncode: Optional[int] = None
        rc_path = run_path / _RC
        if rc_path.exists():
            try:
                returncode = int(rc_path.read_text().strip())
            except (ValueError, OSError):
                returncode = None
        if returncode is None:
            returncode = job.exit_code
        return {
            "status": "completed",
            "output_dir": str(run_path),
            "returncode": returncode,
        }

    def _cancel_and_report(
        self, job_id: str, remote_dir: str, run_path: Path, why: str, rc: str,
    ) -> Dict[str, Any]:
        """Cancel the scheduler job, retrieve what landed, and report an error."""
        try:
            self._resolve_scheduler().cancel(job_id)
        except Exception as exc:  # noqa: BLE001 — best-effort cancel
            logger.warning("ClusterExecutor: cancel of %s failed: %s", job_id, exc)
        self._download_outputs(remote_dir, run_path)
        (run_path / _STDERR).write_text(why, encoding="utf-8")
        (run_path / _RC).write_text(rc, encoding="utf-8")
        return {
            "status": "error", "output_dir": str(run_path),
            "returncode": None, "error": why,
        }

    # ── Executor contract ─────────────────────────────────────

    def run(
        self, input_files: Dict[str, str], run_command: str, run_dir: str
    ) -> Dict[str, Any]:
        handle = self.submit(input_files, run_command, run_dir)
        run_path: Path = handle["run_path"]
        if "job_id" not in handle:        # submission failed in submit()
            return {
                "status": "error", "output_dir": str(run_path),
                "returncode": None, "error": handle["error"],
            }
        job_id = handle["job_id"]
        remote_dir = handle["remote_dir"]

        elapsed = 0
        try:
            while True:
                status = self.poll(handle)
                job = status["job"]
                if status["terminal"]:
                    break
                # A Stop mid-wait cancels the job instead of orphaning it.
                if self.cancel_check is not None and self.cancel_check():
                    return self._cancel_and_report(
                        job_id, remote_dir, run_path,
                        f"Job {job_id} cancelled on request.", "cancelled",
                    )
                if elapsed >= self.timeout:
                    return self._cancel_and_report(
                        job_id, remote_dir, run_path,
                        f"Job {job_id} timed out after {self.timeout}s wall-clock; cancelled.",
                        "timeout",
                    )
                time.sleep(self.poll_interval)
                elapsed += self.poll_interval
        except KeyboardInterrupt:
            # Ctrl-C / Stop during the wait must not leave the allocation
            # running on the cluster.
            self._cancel_and_report(
                job_id, remote_dir, run_path,
                f"Job {job_id} cancelled (interrupted).", "cancelled",
            )
            raise

        return self._finalize(remote_dir, run_path, job)
