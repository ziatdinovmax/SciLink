"""Ground the LAMMPS health reader in REAL engine output.

Runs the actual ``read_health_observable`` + health gate against trimmed-but-
real ``log.lammps`` fixtures captured from production runs (tests/fixtures/
lammps/). This guards against writing a parser and its test from the same
unchecked mental model of the log format: the fixtures are real LAMMPS bytes
(real header, real multi-column thermo, and real end-of-file noise — a
truncated final row, a `3d grid and FFT values/proc = ...` line, a LAMMPS
ERROR line), not a reconstruction.

Fixtures:
  log_healthy.lammps    a completed run, density ~0.83 g/cm^3   -> PASS
  log_exploded.lammps   a barostat-blown box, density ~0.003    -> FAIL
  log_no_density.lammps a deck that logs the stress tensor, not
                        density (viscosity run)                 -> SKIP (None)
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.health import evaluate_health  # noqa: E402
from scilink.skills.molecular_dynamics.lammps.lammps import (  # noqa: E402
    read_health_observable,
)

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "lammps"
# the loose density band lammps.md declares
DENSITY_SPEC = [{"observable": "density", "min": 0.02, "max": 30.0}]


def _run_dir(fixture_name):
    """A temp dir containing the fixture renamed to log.lammps."""
    td = Path(tempfile.mkdtemp())
    shutil.copy(FIXTURES / fixture_name, td / "log.lammps")
    return str(td)


def test_healthy_real_log_reads_physical_density():
    rho = read_health_observable(_run_dir("log_healthy.lammps"), "density")
    assert rho == pytest.approx(0.83, abs=0.05)
    assert evaluate_health({"density": rho}, DENSITY_SPEC) == []  # PASS


def test_exploded_real_log_is_caught():
    rho = read_health_observable(_run_dir("log_exploded.lammps"), "density")
    assert rho < 0.02                       # a near-vacuum
    violations = evaluate_health({"density": rho}, DENSITY_SPEC)
    assert len(violations) == 1             # the gate FAILS it
    assert "density" in violations[0].reason


def test_stress_only_deck_returns_none_and_gate_skips():
    # This real deck logs Pxy/Pxz/Pyz (for viscosity), not density, so the
    # density gate cannot judge it — it must skip, not fabricate a failure.
    rho = read_health_observable(_run_dir("log_no_density.lammps"), "density")
    assert rho is None
    assert evaluate_health({"density": rho}, DENSITY_SPEC) == []


def test_reader_handles_real_eof_noise_without_raising():
    # The real logs end in a truncated row / FFT-grid line / ERROR line; none
    # should fool the parser or raise.
    for fx in ("log_healthy.lammps", "log_exploded.lammps", "log_no_density.lammps"):
        for obs in ("density", "temperature", "volume", "pressure"):
            read_health_observable(_run_dir(fx), obs)  # no exception


def test_temperature_and_volume_also_readable():
    d = _run_dir("log_healthy.lammps")
    assert read_health_observable(d, "temperature") == pytest.approx(298, abs=20)
    assert read_health_observable(d, "volume") > 0
