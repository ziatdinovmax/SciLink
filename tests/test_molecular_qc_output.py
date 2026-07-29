"""Tests for the NWChem deterministic critic hook (``snapshot_run`` via cclib).

Three layers, by dependency:
  1. TOOL_SPEC shape + module import — no cclib, no fixture.
  2. Graceful behavior on an empty/started run dir — no cclib, no fixture.
  3. Real-output parse — needs cclib AND a captured NWChem ``.out`` fixture.

Layer 3 auto-skips until a real fixture is dropped into
``tests/fixtures/nwchem/``. NOTE: the project archive's ``3_DFT/nwchem_jobs``
holds only ``_rdkit.xyz`` INPUTS — NWChem was never run — so no fixture exists
yet. Capture one short run (e.g. di-n-propylamine) into the fixtures dir to
activate layer 3, and install cclib in the test env.
"""

from pathlib import Path

import pytest

from scilink.skills.molecular_qc.nwchem.nwchem_output import snapshot_run, TOOL_SPEC

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "nwchem"


# ── Layer 1: spec + import (no cclib, no fixture) ──────────────────

def test_tool_spec_shape():
    assert TOOL_SPEC.name == "snapshot_run"
    assert "simulation" in TOOL_SPEC.agents
    assert "output_dir" in TOOL_SPEC.parameters
    assert TOOL_SPEC.required == ["output_dir"]
    # import_line must point back at this module so the registry can dispatch.
    assert TOOL_SPEC.import_line.endswith(
        "molecular_qc.nwchem.nwchem_output import snapshot_run"
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


# ── Layer 3: real parse (needs cclib + a captured fixture) ─────────

@pytest.mark.skipif(
    not FIXTURE_DIR.is_dir() or not any(FIXTURE_DIR.glob("*.out")),
    reason=("no real NWChem .out fixture captured yet — DFT was never run in "
            "the archive; drop one into tests/fixtures/nwchem/ to enable"),
)
def test_snapshot_real_output():
    pytest.importorskip("cclib")
    out = snapshot_run(str(FIXTURE_DIR))
    assert out["status"] == "ok"
    assert out["files_found"], "expected the .out file to be discovered"
    # A converged single point/opt should yield an SCF energy and a verdict.
    assert out["scf_energy"] is not None
    assert out["convergence_status"] in {"converged", "not_converged", "failed"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
