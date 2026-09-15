"""Tests for the ORCA deterministic critic hook (``snapshot_run`` via cclib).

Layers, by dependency:
  1. TOOL_SPEC shape + module import — no cclib, no fixture.
  2. Graceful behavior on missing / empty run dirs — no cclib, no fixture.
  3. Log classification (error hints, normal-termination banner) from a
     synthetic ``.out`` — no cclib (the tail-text pass runs regardless).
  4. Real-output parse — needs cclib AND a captured ORCA ``.out`` fixture;
     auto-skips until one is dropped into ``tests/fixtures/orca/``.
"""

from pathlib import Path

import pytest

from scilink.skills.molecular_qc.orca.orca_output import snapshot_run, TOOL_SPEC

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "orca"


# ── Layer 1: spec + import (no cclib, no fixture) ──────────────────

def test_tool_spec_shape():
    assert TOOL_SPEC.name == "snapshot_run"
    assert "simulation" in TOOL_SPEC.agents
    assert "output_dir" in TOOL_SPEC.parameters
    assert TOOL_SPEC.required == ["output_dir"]
    assert TOOL_SPEC.import_line.endswith(
        "molecular_qc.orca.orca_output import snapshot_run"
    )


# ── Layer 2: graceful no-output handling (no cclib, no fixture) ─────

def test_snapshot_missing_dir():
    out = snapshot_run("/no/such/run/dir")
    assert out["status"] == "error"


def test_snapshot_empty_dir(tmp_path):
    out = snapshot_run(str(tmp_path))
    assert out["status"] == "ok"
    assert out["files_found"] == []
    assert out["convergence_status"] == "unknown"
    assert out["scf_energy"] is None
    assert out["terminated_normally"] is None


# ── Layer 3: log classification from a synthetic .out (no cclib) ────

def test_scf_not_converged_flagged(tmp_path):
    (tmp_path / "orca.out").write_text(
        "some header\nSCF NOT CONVERGED AFTER  125 CYCLES\nmore text\n"
    )
    out = snapshot_run(str(tmp_path))
    assert out["convergence_status"] == "failed"
    assert any("SCF NOT CONVERGED" in h for h in out["log_error_hints"])


def test_normal_termination_detected(tmp_path):
    (tmp_path / "orca.out").write_text(
        "run output ...\n****ORCA TERMINATED NORMALLY****\nTOTAL RUN TIME: ...\n"
    )
    out = snapshot_run(str(tmp_path))
    assert out["terminated_normally"] is True
    assert out["log_error_hints"] == []


def test_no_banner_no_energy_reads_as_failed(tmp_path):
    """A log that never reached the normal-termination banner (and cclib finds
    no energy) is treated as a crash, not an unknown."""
    (tmp_path / "orca.out").write_text("partial output, process killed\n")
    out = snapshot_run(str(tmp_path))
    assert out["terminated_normally"] is False
    # With no parseable energy and no banner, status resolves to failed.
    assert out["convergence_status"] == "failed"


# ── Layer 4: real parse (needs cclib + a captured fixture) ─────────

@pytest.mark.skipif(
    not FIXTURE_DIR.is_dir() or not any(FIXTURE_DIR.glob("*.out")),
    reason=("no real ORCA .out fixture captured yet — drop one into "
            "tests/fixtures/orca/ to enable"),
)
def test_snapshot_real_output():
    pytest.importorskip("cclib")
    out = snapshot_run(str(FIXTURE_DIR))
    assert out["status"] == "ok"
    assert out["files_found"], "expected the .out file to be discovered"
    assert out["scf_energy"] is not None
    assert out["convergence_status"] in {"converged", "not_converged", "failed"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
