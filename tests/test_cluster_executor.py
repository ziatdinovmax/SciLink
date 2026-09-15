"""Unit tests for ClusterExecutor and the scheduler batch builder.

No real SSH, no paramiko, no cluster: a fake connection simulates a remote
filesystem and a fake scheduler simulates submit/poll, so the upload → submit →
poll → download → persist flow and the engine-neutral snapshot are verified
offline.
"""
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# scilink.hpc.* imports paramiko (an optional dependency) via hpc/__init__, so
# skip the whole module cleanly where it isn't installed rather than erroring.
pytest.importorskip("paramiko")

from scilink.agents.sim_agents.cluster_executor import (  # noqa: E402
    ClusterExecutor, _RC, _STDOUT, _STDERR,
)
from scilink.hpc.batch import build_batch_script  # noqa: E402
from scilink.hpc.scheduler import (  # noqa: E402
    JobStatus, HPCJob, SlurmScheduler, PBSScheduler, LSFScheduler,
)


# ── test doubles ──────────────────────────────────────────────

class FakeConn:
    """In-memory stand-in for HPCConnection with a simulated remote FS."""

    def __init__(self, connected=True):
        self._connected = connected
        self._remote: dict[str, str] = {}
        self.uploads, self.downloads, self.mkdirs = [], [], []
        self.connect_called = 0

    @property
    def is_connected(self):
        return self._connected

    def connect(self, *a, **k):
        self.connect_called += 1
        self._connected = True

    def home_dir(self):
        return "/remote/home"

    def mkdir_p(self, path):
        self.mkdirs.append(path)

    def upload(self, local, remote):
        self.uploads.append((local, remote))
        self._remote[remote] = Path(local).read_text()

    def download(self, remote, local):
        self.downloads.append((remote, local))
        Path(local).write_text(self._remote[remote])

    def listdir(self, path):
        prefix = path.rstrip("/") + "/"
        out = []
        for rp in self._remote:
            rest = rp[len(prefix):] if rp.startswith(prefix) else None
            if rest and "/" not in rest:
                out.append(SimpleNamespace(
                    filename=rest, st_mode=stat.S_IFREG | 0o644))
        return out


class FakeScheduler:
    """Simulates submit + status; writes the wrapper's output files on submit."""

    name = "FAKE"

    def __init__(self, conn, *, terminal_after=1,
                 final_status=JobStatus.COMPLETED, exit_code=0,
                 rc_written="0", submit_raises=False):
        self.conn = conn
        self.terminal_after = terminal_after
        self.final_status = final_status
        self.exit_code = exit_code
        self.rc_written = rc_written
        self.submit_raises = submit_raises
        self.submitted, self.cancelled = [], []
        self._polls = 0

    def batch_directives(self, resources):
        return [f"#FAKE job={resources.get('job_name', 'scilink')}"]

    def workdir_prelude(self):
        return []

    def submit(self, script, work_dir=""):
        if self.submit_raises:
            raise RuntimeError("sbatch boom")
        self.submitted.append((script, work_dir))
        if self.rc_written is not None:
            self.conn._remote[f"{work_dir}/{_RC}"] = self.rc_written
            self.conn._remote[f"{work_dir}/{_STDOUT}"] = "thermo...\n"
            self.conn._remote[f"{work_dir}/{_STDERR}"] = ""
            self.conn._remote[f"{work_dir}/final.data"] = "RESULT"
        return "12345"

    def status(self, job_id):
        self._polls += 1
        st = (self.final_status if self._polls >= self.terminal_after
              else JobStatus.RUNNING)
        return HPCJob(job_id=job_id, status=st, exit_code=self.exit_code)

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return True


# ── ClusterExecutor.run ───────────────────────────────────────

class TestClusterExecutorRun:
    def test_full_flow_uploads_submits_polls_downloads(self, tmp_path):
        conn = FakeConn()
        sched = FakeScheduler(conn, terminal_after=2)
        ex = ClusterExecutor(conn, scheduler=sched, remote_root="/scratch",
                             resources={"partition": "normal"},
                             poll_interval=0, timeout=100)
        run_dir = tmp_path / "production"
        result = ex.run({"in.lj": "SCRIPT"}, "lmp -in in.lj", str(run_dir))

        assert result["status"] == "completed"
        assert result["returncode"] == 0
        assert result["output_dir"] == str(run_dir)
        # Inputs materialized locally.
        assert (run_dir / "in.lj").read_text() == "SCRIPT"
        # Remote dir = remote_root / basename(run_dir); created and used.
        assert "/scratch/production" in conn.mkdirs
        # Inputs + job script uploaded.
        uploaded = {r for _, r in conn.uploads}
        assert "/scratch/production/in.lj" in uploaded
        assert "/scratch/production/scilink_job.sh" in uploaded
        # Submitted with the job script + remote work dir.
        assert sched.submitted == [("scilink_job.sh", "/scratch/production")]
        # Polled until terminal (terminal_after=2).
        assert sched._polls == 2
        # Outputs downloaded into the local run dir, including the rc file.
        assert (run_dir / _RC).read_text() == "0"
        assert (run_dir / "final.data").read_text() == "RESULT"

    def test_remote_root_defaults_to_home(self, tmp_path):
        conn = FakeConn()
        sched = FakeScheduler(conn)
        ex = ClusterExecutor(conn, scheduler=sched, poll_interval=0)
        ex.run({"in.lj": "S"}, "lmp -in in.lj", str(tmp_path / "run"))
        assert "/remote/home/run" in conn.mkdirs

    def test_connects_if_not_connected(self, tmp_path):
        conn = FakeConn(connected=False)
        sched = FakeScheduler(conn)
        ex = ClusterExecutor(conn, scheduler=sched, poll_interval=0)
        ex.run({"in.lj": "S"}, "lmp -in in.lj", str(tmp_path / "run"))
        assert conn.connect_called == 1

    def test_returncode_falls_back_to_scheduler_exit_code(self, tmp_path):
        # Wrapper wrote no rc file → use the scheduler's reported exit code.
        conn = FakeConn()
        sched = FakeScheduler(conn, rc_written=None,
                              final_status=JobStatus.FAILED, exit_code=1)
        ex = ClusterExecutor(conn, scheduler=sched, poll_interval=0)
        result = ex.run({"in.lj": "S"}, "lmp -in in.lj", str(tmp_path / "run"))
        assert result["status"] == "completed"   # ran to a terminal state
        assert result["returncode"] == 1

    def test_submit_failure_returns_error(self, tmp_path):
        conn = FakeConn()
        sched = FakeScheduler(conn, submit_raises=True)
        ex = ClusterExecutor(conn, scheduler=sched, poll_interval=0)
        run_dir = tmp_path / "run"
        result = ex.run({"in.lj": "S"}, "lmp -in in.lj", str(run_dir))
        assert result["status"] == "error"
        assert result["returncode"] is None
        assert (run_dir / _RC).read_text() == "submit_error"

    def test_dead_session_that_cant_recover_raises_clear_error(self, tmp_path):
        class DeadConn(FakeConn):
            @property
            def is_connected(self):
                return False

            def connect(self, *a, **k):
                raise RuntimeError("password required")

        ex = ClusterExecutor(DeadConn(), scheduler=FakeScheduler(FakeConn()),
                             poll_interval=0)
        with pytest.raises(ConnectionError, match="could not be re-established"):
            ex.run({"in.lj": "S"}, "lmp -in in.lj", str(tmp_path / "run"))

    def test_timeout_cancels_and_errors(self, tmp_path):
        conn = FakeConn()
        # Never reaches terminal within the budget.
        sched = FakeScheduler(conn, terminal_after=9999)
        ex = ClusterExecutor(conn, scheduler=sched, poll_interval=0, timeout=0)
        run_dir = tmp_path / "run"
        result = ex.run({"in.lj": "S"}, "lmp -in in.lj", str(run_dir))
        assert result["status"] == "error"
        assert "timed out" in result["error"]
        assert sched.cancelled == ["12345"]
        assert (run_dir / _RC).read_text() == "timeout"

    def test_connect_factory_builds_profile_and_connects(self, monkeypatch):
        # The factory assembles the profile, opens the connection, and forwards
        # executor kwargs — without any real SSH.
        import scilink.hpc.connection as conn_mod
        captured = {}

        class FakeHPCConn:
            def __init__(self, profile):
                captured["profile"] = profile

            def connect(self, password="", key_passphrase=""):
                captured["password"] = password
                captured["passphrase"] = key_passphrase

        monkeypatch.setattr(conn_mod, "HPCConnection", FakeHPCConn)
        ex = ClusterExecutor.connect(
            hostname="deception.pnl.gov", username="alle927", password="secret",
            resources={"account": "ACC"}, remote_root="/scratch",
            poll_interval=5,
        )
        assert isinstance(ex, ClusterExecutor)
        assert captured["profile"].hostname == "deception.pnl.gov"
        assert captured["profile"].username == "alle927"
        assert captured["profile"].auth_method == "password"  # inferred
        assert captured["password"] == "secret"
        assert ex.resources == {"account": "ACC"}
        assert ex.remote_root == "/scratch"
        assert ex.poll_interval == 5

    def test_connect_factory_infers_key_auth_without_password(self, monkeypatch):
        import scilink.hpc.connection as conn_mod
        captured = {}

        class FakeHPCConn:
            def __init__(self, profile):
                captured["profile"] = profile

            def connect(self, password="", key_passphrase=""):
                pass

        monkeypatch.setattr(conn_mod, "HPCConnection", FakeHPCConn)
        ClusterExecutor.connect(hostname="h", username="u", key_path="/k")
        assert captured["profile"].auth_method == "key"
        assert captured["profile"].key_path == "/k"

    def test_uploaded_script_has_directives_and_wrapper(self, tmp_path):
        conn = FakeConn()
        sched = FakeScheduler(conn)
        ex = ClusterExecutor(conn, scheduler=sched, poll_interval=0,
                             setup=["module load lammps"])
        ex.run({"in.lj": "S"}, "lmp -in in.lj", str(tmp_path / "run"))
        script = conn._remote["/remote/home/run/scilink_job.sh"]
        assert script.startswith("#!/bin/bash")
        assert "#FAKE job=scilink" in script
        assert "module load lammps" in script
        assert f"lmp -in in.lj > {_STDOUT} 2> {_STDERR}" in script
        assert f"echo $? > {_RC}" in script


# ── batch builder + scheduler directives ──────────────────────

class TestBatchBuilder:
    def test_slurm_directives(self):
        sched = SlurmScheduler(conn=None)
        d = sched.batch_directives({
            "job_name": "md", "partition": "normal", "nodes": 2,
            "ntasks": 8, "time": "00:30:00", "gres": "gpu:1",
            "account": "proj", "extra_directives": ["#SBATCH --exclusive"],
        })
        assert "#SBATCH --job-name=md" in d
        assert "#SBATCH --nodes=2" in d
        assert "#SBATCH --ntasks=8" in d
        assert "#SBATCH --time=00:30:00" in d
        assert "#SBATCH --partition=normal" in d
        assert "#SBATCH --gres=gpu:1" in d
        assert "#SBATCH --account=proj" in d
        assert "#SBATCH --exclusive" in d

    def test_slurm_omits_optional_when_absent(self):
        d = SlurmScheduler(conn=None).batch_directives({})
        assert not any("partition" in line for line in d)
        assert not any("gres" in line for line in d)

    def test_pbs_directives(self):
        d = PBSScheduler(conn=None).batch_directives(
            {"job_name": "md", "nodes": 1, "ntasks": 4, "time": "01:00:00",
             "partition": "batch"})
        assert "#PBS -N md" in d
        assert "#PBS -l nodes=1:ppn=4" in d
        assert "#PBS -l walltime=01:00:00" in d
        assert "#PBS -q batch" in d

    def test_lsf_directives(self):
        d = LSFScheduler(conn=None).batch_directives(
            {"job_name": "md", "ntasks": 4, "time": "01:00", "partition": "q"})
        assert "#BSUB -J md" in d
        assert "#BSUB -n 4" in d
        assert "#BSUB -W 01:00" in d
        assert "#BSUB -q q" in d

    def test_pbs_prelude_cds_to_workdir_others_dont(self):
        # PBS/Torque starts in $HOME → the wrapper must cd to the submit dir;
        # SLURM/LSF start there already, so no cd.
        assert PBSScheduler(conn=None).workdir_prelude() == ["cd $PBS_O_WORKDIR"]
        assert SlurmScheduler(conn=None).workdir_prelude() == []
        assert LSFScheduler(conn=None).workdir_prelude() == []
        pbs = build_batch_script(PBSScheduler(conn=None), "nwchem in.nw")
        slurm = build_batch_script(SlurmScheduler(conn=None), "nwchem in.nw")
        assert "cd $PBS_O_WORKDIR" in pbs
        # And it lands before the command, so relative output paths resolve.
        assert pbs.index("cd $PBS_O_WORKDIR") < pbs.index("nwchem in.nw")
        assert "$PBS_O_WORKDIR" not in slurm

    def test_build_batch_script_assembles_header_setup_wrapper(self):
        sched = SlurmScheduler(conn=None)
        script = build_batch_script(
            sched, "lmp -in in.lj",
            resources={"partition": "normal", "time": "00:30:00"},
            setup=["module load lammps", "conda activate md"],
        )
        lines = script.splitlines()
        assert lines[0] == "#!/bin/bash"
        assert "#SBATCH --partition=normal" in script
        assert "module load lammps" in script
        assert "conda activate md" in script
        assert "lmp -in in.lj > run_stdout.log 2> run_stderr.log" in script
        assert "echo $? > run_returncode.txt" in script


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
