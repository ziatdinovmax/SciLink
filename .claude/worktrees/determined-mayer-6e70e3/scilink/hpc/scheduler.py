"""Job-scheduler abstraction: SLURM, PBS/Torque, LSF."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scilink.hpc.connection import HPCConnection


# ── data model ────────────────────────────────────────────────

class JobStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    TIMEOUT = "Timeout"
    UNKNOWN = "Unknown"

    @property
    def emoji(self) -> str:
        return {
            JobStatus.PENDING: "🟡",
            JobStatus.RUNNING: "🔵",
            JobStatus.COMPLETED: "🟢",
            JobStatus.FAILED: "🔴",
            JobStatus.CANCELLED: "⚫",
            JobStatus.TIMEOUT: "🟠",
            JobStatus.UNKNOWN: "⚪",
        }[self]

    @property
    def is_terminal(self) -> bool:
        return self in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        )

@dataclass
class HPCJob:
    job_id: str
    name: str = ""
    status: JobStatus = JobStatus.UNKNOWN
    raw_status: str = ""
    partition: str = ""
    nodes: int = 1
    ntasks: int = 1
    time_limit: str = ""
    time_used: str = ""
    work_dir: str = ""
    stdout_file: str = ""
    stderr_file: str = ""
    script_path: str = ""
    submit_time: str = ""
    start_time: str = ""
    end_time: str = ""
    exit_code: Optional[int] = None
    node_list: str = ""


# ── abstract base ─────────────────────────────────────────────

class Scheduler(ABC):
    name: str = "unknown"

    def __init__(self, conn: HPCConnection) -> None:
        self.conn = conn

    @abstractmethod
    def detect(self) -> bool: ...

    @abstractmethod
    def submit(self, script: str, work_dir: str = "") -> str: ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool: ...

    @abstractmethod
    def status(self, job_id: str) -> HPCJob: ...

    @abstractmethod
    def queue(self, user: Optional[str] = None) -> list[HPCJob]: ...

    @abstractmethod
    def partitions(self) -> list[dict]: ...

    def tail_output(
        self,
        job: HPCJob,
        stream: str = "stdout",
        lines: int = 200,
    ) -> str:
        path = job.stdout_file if stream == "stdout" else job.stderr_file
        if not path:
            # SLURM default naming
            default = f"slurm-{job.job_id}.out" if stream == "stdout" else ""
            path = f"{job.work_dir}/{default}" if job.work_dir and default else ""
        if not path:
            return ""
        try:
            return self.conn.read_text(path, tail=lines)
        except Exception as exc:
            return f"(cannot read {path}: {exc})"

# ══════════════════════════════════════════════════════════════
# SLURM
# ══════════════════════════════════════════════════════════════

class SlurmScheduler(Scheduler):
    name = "SLURM"

    _MAP: dict[str, JobStatus] = {
        "PD": JobStatus.PENDING,
        "PENDING": JobStatus.PENDING,
        "CF": JobStatus.PENDING,
        "CONFIGURING": JobStatus.PENDING,
        "R": JobStatus.RUNNING,
        "RUNNING": JobStatus.RUNNING,
        "CG": JobStatus.RUNNING,
        "COMPLETING": JobStatus.RUNNING,
        "CD": JobStatus.COMPLETED,
        "COMPLETED": JobStatus.COMPLETED,
        "F": JobStatus.FAILED,
        "FAILED": JobStatus.FAILED,
        "NF": JobStatus.FAILED,
        "NODE_FAIL": JobStatus.FAILED,
        "OOM": JobStatus.FAILED,
        "OUT_OF_MEMORY": JobStatus.FAILED,
        "CA": JobStatus.CANCELLED,
        "CANCELLED": JobStatus.CANCELLED,
        "TO": JobStatus.TIMEOUT,
        "TIMEOUT": JobStatus.TIMEOUT,
    }

    def detect(self) -> bool:
        _, _, rc = self.conn.run("command -v squeue", timeout=10)
        return rc == 0

    def submit(self, script: str, work_dir: str = "") -> str:
        pre = f"cd {_q(work_dir)} && " if work_dir else ""
        out, err, rc = self.conn.run(f"{pre}sbatch {_q(script)}")
        if rc != 0:
            raise RuntimeError(f"sbatch failed (rc={rc}):\n{err}")
        m = re.search(r"Submitted batch job (\d+)", out)
        if not m:
            raise RuntimeError(f"Cannot parse job ID:\n{out}")
        return m.group(1)

    def cancel(self, job_id: str) -> bool:
        _, _, rc = self.conn.run(f"scancel {job_id}")
        return rc == 0

    # ── status ────────────────────────────────────────────────

    def status(self, job_id: str) -> HPCJob:
        # scontrol for live jobs
        out, _, rc = self.conn.run(
            f"scontrol show job {job_id}", timeout=15,
        )
        if rc == 0 and out.strip():
            return self._parse_scontrol(out, job_id)
        # sacct for finished jobs
        out, _, rc = self.conn.run(
            f"sacct -j {job_id} -P --noheader "
            f"--format=JobID,JobName,State,ExitCode,Start,End,"
            f"Elapsed,Partition,NodeList,WorkDir",
            timeout=15,
        )
        if rc == 0 and out.strip():
            return self._parse_sacct(out, job_id)
        return HPCJob(job_id=job_id)

    def _parse_scontrol(self, raw: str, jid: str) -> HPCJob:
        kv: dict[str, str] = {}
        for tok in raw.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                kv[k] = v
        rs = kv.get("JobState", "UNKNOWN").split()[0]
        ec = _parse_exit_code(kv.get("ExitCode", ""))
        return HPCJob(
            job_id=jid,
            name=kv.get("JobName", ""),
            status=self._MAP.get(rs, JobStatus.UNKNOWN),
            raw_status=rs,
            partition=kv.get("Partition", ""),
            nodes=_int(kv.get("NumNodes"), 1),
            ntasks=_int(kv.get("NumCPUs"), 1),
            time_limit=kv.get("TimeLimit", ""),
            time_used=kv.get("RunTime", ""),
            work_dir=kv.get("WorkDir", ""),
            stdout_file=kv.get("StdOut", ""),
            stderr_file=kv.get("StdErr", ""),
            submit_time=kv.get("SubmitTime", ""),
            start_time=kv.get("StartTime", ""),
            end_time=kv.get("EndTime", ""),
            node_list=kv.get("NodeList", ""),
            exit_code=ec,
        )

    def _parse_sacct(self, raw: str, jid: str) -> HPCJob:
        # First non-empty line = main job record (skip .batch, .extern)
        for line in raw.strip().splitlines():
            p = line.split("|")
            if len(p) >= 10 and not ("." in p[0]):
                break
        else:
            return HPCJob(job_id=jid)
        rs = p[2].split()[0]
        return HPCJob(
            job_id=jid,
            name=p[1],
            status=self._MAP.get(rs, JobStatus.UNKNOWN),
            raw_status=rs,
            exit_code=_parse_exit_code(p[3]),
            start_time=p[4],
            end_time=p[5],
            time_used=p[6],
            partition=p[7],
            node_list=p[8],
            work_dir=p[9],
        )

# ── queue ─────────────────────────────────────────────────

    def queue(self, user: Optional[str] = None) -> list[HPCJob]:
        uf = f"-u {user}" if user else "-u $USER"
        out, _, rc = self.conn.run(
            f'squeue {uf} -o "%i|%j|%T|%P|%D|%C|%l|%M|%V|%S|%R" --noheader',
            timeout=15,
        )
        if rc != 0:
            return []
        jobs: list[HPCJob] = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            p = [x.strip() for x in line.split("|")]
            if len(p) < 11:
                continue
            rs = p[2]
            jobs.append(HPCJob(
                job_id=p[0],
                name=p[1],
                status=self._MAP.get(rs, JobStatus.UNKNOWN),
                raw_status=rs,
                partition=p[3],
                nodes=_int(p[4], 1),
                ntasks=_int(p[5], 1),
                time_limit=p[6],
                time_used=p[7],
                submit_time=p[8],
                start_time=p[9],
                node_list=p[10],
            ))
        return jobs

 # ── partitions ────────────────────────────────────────────

    def partitions(self) -> list[dict]:
        out, _, rc = self.conn.run(
            'sinfo -o "%P|%a|%l|%D|%T|%c|%m|%G" --noheader',
            timeout=15,
        )
        if rc != 0:
            return []
        parts: list[dict] = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split("|")]
            if len(p) < 8:
                continue
            parts.append(dict(
                name=p[0].rstrip("*"),
                avail=p[1],
                timelimit=p[2],
                nodes=p[3],
                state=p[4],
                cpus=p[5],
                mem_MB=p[6],
                gres=p[7],
                default="*" in p[0],
            ))
        return parts


# ══════════════════════════════════════════════════════════════
# PBS / Torque
# ══════════════════════════════════════════════════════════════

class PBSScheduler(Scheduler):
    name = "PBS"

    _MAP: dict[str, JobStatus] = {
        "Q": JobStatus.PENDING,
        "H": JobStatus.PENDING,
        "W": JobStatus.PENDING,
        "R": JobStatus.RUNNING,
        "E": JobStatus.RUNNING,
        "C": JobStatus.COMPLETED,
        "F": JobStatus.FAILED,
    }

    def detect(self) -> bool:
        _, _, rc = self.conn.run("command -v qsub", timeout=10)
        return rc == 0

    def submit(self, script: str, work_dir: str = "") -> str:
        pre = f"cd {_q(work_dir)} && " if work_dir else ""
        out, err, rc = self.conn.run(f"{pre}qsub {_q(script)}")
        if rc != 0:
            raise RuntimeError(f"qsub failed:\n{err}")
        return out.strip().split(".")[0]

    def cancel(self, job_id: str) -> bool:
        _, _, rc = self.conn.run(f"qdel {job_id}")
        return rc == 0

    def status(self, job_id: str) -> HPCJob:
        out, _, rc = self.conn.run(f"qstat -f {job_id}", timeout=15)
        if rc != 0:
            return HPCJob(job_id=job_id)
        kv: dict[str, str] = {}
        for line in out.splitlines():
            if " = " in line:
                k, _, v = line.strip().partition(" = ")
                kv[k.strip()] = v.strip()
        rs = kv.get("job_state", "U")
        return HPCJob(
            job_id=job_id,
            name=kv.get("Job_Name", ""),
            status=self._MAP.get(rs, JobStatus.UNKNOWN),
            raw_status=rs,
            partition=kv.get("queue", ""),
            stdout_file=kv.get("Output_Path", "").rsplit(":", 1)[-1],
            stderr_file=kv.get("Error_Path", "").rsplit(":", 1)[-1],
            node_list=kv.get("exec_host", ""),
        )

    def queue(self, user: Optional[str] = None) -> list[HPCJob]:
        uf = f"-u {user}" if user else ""
        out, _, rc = self.conn.run(f"qstat {uf}", timeout=15)
        if rc != 0:
            return []
        jobs: list[HPCJob] = []
        for line in out.strip().splitlines()[2:]:  # skip header
            p = line.split()
            if len(p) < 6:
                continue
            rs = p[4]
            jobs.append(HPCJob(
                job_id=p[0].split(".")[0],
                name=p[1],
                status=self._MAP.get(rs, JobStatus.UNKNOWN),
                raw_status=rs,
                time_used=p[3],
                partition=p[5],
            ))
        return jobs

    def partitions(self) -> list[dict]:
        out, _, rc = self.conn.run("qstat -Q", timeout=15)
        if rc != 0:
            return []
        return [
            {"name": line.split()[0], "avail": "yes"}
            for line in out.strip().splitlines()[2:]
            if line.split()
        ]


# ══════════════════════════════════════════════════════════════
# IBM Spectrum LSF
# ══════════════════════════════════════════════════════════════

class LSFScheduler(Scheduler):
    name = "LSF"

    _MAP: dict[str, JobStatus] = {
        "PEND": JobStatus.PENDING,
        "RUN": JobStatus.RUNNING,
        "DONE": JobStatus.COMPLETED,
        "EXIT": JobStatus.FAILED,
        "USUSP": JobStatus.PENDING,
        "SSUSP": JobStatus.PENDING,
    }

    def detect(self) -> bool:
        _, _, rc = self.conn.run("command -v bsub", timeout=10)
        return rc == 0

    def submit(self, script: str, work_dir: str = "") -> str:
        pre = f"cd {_q(work_dir)} && " if work_dir else ""
        out, err, rc = self.conn.run(f"{pre}bsub < {_q(script)}")
        if rc != 0:
            raise RuntimeError(f"bsub failed:\n{err}")
        m = re.search(r"Job <(\d+)>", out)
        if not m:
            raise RuntimeError(f"Cannot parse job ID:\n{out}")
        return m.group(1)

    def cancel(self, job_id: str) -> bool:
        _, _, rc = self.conn.run(f"bkill {job_id}")
        return rc == 0

    def status(self, job_id: str) -> HPCJob:
        out, _, rc = self.conn.run(
            f"bjobs -o 'jobid stat job_name queue exec_host' "
            f"-noheader {job_id}",
            timeout=15,
        )
        if rc != 0:
            return HPCJob(job_id=job_id)
        p = out.strip().split()
        rs = p[1] if len(p) > 1 else "UNKNOWN"
        return HPCJob(
            job_id=job_id,
            status=self._MAP.get(rs, JobStatus.UNKNOWN),
            raw_status=rs,
            name=p[2] if len(p) > 2 else "",
            partition=p[3] if len(p) > 3 else "",
            node_list=p[4] if len(p) > 4 else "",
        )

    def queue(self, user: Optional[str] = None) -> list[HPCJob]:
        uf = f"-u {user}" if user else ""
        out, _, rc = self.conn.run(
            f"bjobs {uf} -o 'jobid stat job_name queue' -noheader",
            timeout=15,
        )
        if rc != 0:
            return []
        jobs: list[HPCJob] = []
        for line in out.strip().splitlines():
            p = line.split()
            if len(p) < 2:
                continue
            rs = p[1]
            jobs.append(HPCJob(
                job_id=p[0],
                status=self._MAP.get(rs, JobStatus.UNKNOWN),
                raw_status=rs,
                name=p[2] if len(p) > 2 else "",
                partition=p[3] if len(p) > 3 else "",
            ))
        return jobs

    def partitions(self) -> list[dict]:
        out, _, rc = self.conn.run(
            "bqueues -o 'queue_name status' -noheader", timeout=15,
        )
        if rc != 0:
            return []
        return [
            {"name": line.split()[0], "avail": line.split()[1] if len(line.split()) > 1 else ""}
            for line in out.strip().splitlines()
            if line.strip()
        ]

# ── helpers ───────────────────────────────────────────────────

def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _int(s: Optional[str], default: int = 0) -> int:
    try:
        return int(s)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def _parse_exit_code(raw: str) -> Optional[int]:
    if not raw:
        return None
    try:
        return int(raw.split(":")[0])
    except (ValueError, IndexError):
        return None


_SCHEDULER_CLASSES: list[type[Scheduler]] = [
    SlurmScheduler,
    PBSScheduler,
    LSFScheduler,
]


def detect_scheduler(conn: HPCConnection) -> Optional[Scheduler]:
    """Probe the remote host and return the first detected scheduler."""
    for cls in _SCHEDULER_CLASSES:
        sched = cls(conn)
        try:
            if sched.detect():
                return sched
        except Exception:
            continue
    return None
