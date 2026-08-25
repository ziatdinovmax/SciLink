"""Unit tests for the engine-neutral run-output snapshot fallback.

When the active skill registers no engine-specific ``snapshot_run`` parser
(the prose-only case, e.g. LAMMPS), ``_snapshot_run_outputs`` must still hand
the critic real output — a file listing plus text/log tails — rather than a
bare note. These tests exercise that fallback with no engine and no LLM.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.critics import (  # noqa: E402
    _generic_snapshot,
    _snapshot_run_outputs,
)


class TestGenericSnapshot:
    def test_lists_files_and_tails_logs(self, tmp_path):
        (tmp_path / "log.lammps").write_text(
            "step 1\nstep 2\nERROR: Bond atoms missing\n"
        )
        (tmp_path / "data.txt").write_text("some data\n")
        snap = _generic_snapshot(str(tmp_path))
        names = {f["name"] for f in snap["files"]}
        assert {"log.lammps", "data.txt"} <= names
        # Log content surfaces so the LLM critic can spot the ERROR line.
        assert "ERROR: Bond atoms missing" in snap["file_tails"]["log.lammps"]
        assert snap["snapshot_kind"] == "generic"

    def test_tails_extensionless_log_files(self, tmp_path):
        # Names like "stdout" / "log" (no suffix) are still treated as text.
        (tmp_path / "stdout").write_text("line a\nline b\n")
        snap = _generic_snapshot(str(tmp_path))
        assert "stdout" in snap["file_tails"]

    def test_tail_is_bounded(self, tmp_path):
        big = "\n".join(f"line {i}" for i in range(1000))
        (tmp_path / "run.log").write_text(big)
        snap = _generic_snapshot(str(tmp_path))
        tail = snap["file_tails"]["run.log"]
        # Only the last lines are kept, and the final line is present.
        assert tail.count("\n") < 200
        assert "line 999" in tail
        assert "line 0\n" not in tail

    def test_missing_dir_reports_error(self, tmp_path):
        snap = _generic_snapshot(str(tmp_path / "does-not-exist"))
        assert "error" in snap


class TestSnapshotDispatch:
    def test_no_skill_uses_generic(self, tmp_path):
        (tmp_path / "out.log").write_text("hello\n")
        snap = _snapshot_run_outputs(str(tmp_path), skill=None)
        assert snap.get("snapshot_kind") == "generic"
        assert "out.log" in snap["file_tails"]

    def test_unknown_skill_falls_back_to_generic(self, tmp_path):
        # A skill with no snapshot_run tool registered must not blind the
        # critic — it falls back to the generic snapshot.
        (tmp_path / "out.log").write_text("hello\n")
        snap = _snapshot_run_outputs(str(tmp_path), skill="lammps")
        assert snap.get("snapshot_kind") == "generic"
        assert "out.log" in snap["file_tails"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
