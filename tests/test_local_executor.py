"""Unit tests for LocalExecutor.

Runs trivial shell commands in a temp directory — no simulation engine
needed. Verifies that input files are materialized and that stdout, stderr,
and the return code are persisted so the post-run critic's snapshot can read
them.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.refinement import LocalExecutor  # noqa: E402


class TestLocalExecutor:
    def test_materializes_inputs(self, tmp_path):
        ex = LocalExecutor()
        run_dir = tmp_path / "run"
        result = ex.run({"in.txt": "hello"}, "true", str(run_dir))
        assert (run_dir / "in.txt").read_text() == "hello"
        assert result["status"] == "completed"
        assert result["output_dir"] == str(run_dir)

    def test_persists_stdout_and_returncode(self, tmp_path):
        ex = LocalExecutor()
        run_dir = tmp_path / "run"
        result = ex.run({}, "echo marker-line", str(run_dir))
        assert result["returncode"] == 0
        stdout = (run_dir / LocalExecutor.STDOUT_FILE).read_text()
        assert "marker-line" in stdout
        assert (run_dir / LocalExecutor.RETURNCODE_FILE).read_text() == "0"

    def test_nonzero_exit_is_reported_not_judged(self, tmp_path):
        ex = LocalExecutor()
        run_dir = tmp_path / "run"
        # A failing command still "completes" — the executor reports the exit
        # code; it does not decide the run failed.
        result = ex.run({}, "exit 3", str(run_dir))
        assert result["status"] == "completed"
        assert result["returncode"] == 3
        assert (run_dir / LocalExecutor.RETURNCODE_FILE).read_text() == "3"

    def test_stderr_captured(self, tmp_path):
        ex = LocalExecutor()
        run_dir = tmp_path / "run"
        ex.run({}, "echo oops 1>&2", str(run_dir))
        assert "oops" in (run_dir / LocalExecutor.STDERR_FILE).read_text()

    def test_timeout_returns_error(self, tmp_path):
        ex = LocalExecutor(timeout=1)
        run_dir = tmp_path / "run"
        result = ex.run({}, "sleep 5", str(run_dir))
        assert result["status"] == "error"
        assert result["returncode"] is None
        assert "timed out" in result["error"].lower()

    def test_creates_missing_run_dir(self, tmp_path):
        ex = LocalExecutor()
        run_dir = tmp_path / "nested" / "deep" / "run"
        result = ex.run({"a.txt": "x"}, "true", str(run_dir))
        assert run_dir.exists()
        assert result["status"] == "completed"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
