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


def test_uses_last_block_not_earlier_ramp():
    # Two thermo blocks: an equilibration ramp (density ~0.1) then production
    # (density ~0.83). The reader must report the FINAL block only, so the
    # equilibration transient can't bleed into the mean.
    log = (
        "units real\n"
        "LAMMPS\n"
        "   Step          Temp          Density   \n"
        "         0   300.0         0.10\n"
        "       100   300.0         0.20\n"
        "Loop time of 1 on 1 procs\n"
        "   Step          Temp          Density   \n"
        "         0   298.0         0.830\n"
        "       100   298.0         0.832\n"
        "       200   298.0         0.831\n"
        "Loop time of 2 on 1 procs\n"
    )
    td = Path(tempfile.mkdtemp())
    (td / "log.lammps").write_text(log)
    assert read_health_observable(str(td), "density") == pytest.approx(0.831, abs=0.01)


def test_stray_line_mid_block_does_not_truncate(tmp_path):
    # A `fix print` / WARNING line mid-block must be skipped, not end the block —
    # otherwise the mean becomes the startup rows and a healthy run FAILS.
    log = (
        "units real\n"
        "   Step   Temp   Density\n"
        "      0   300.0   0.010\n"
        "    100   300.0   0.015\n"
        "Water box: 0.5 ns elapsed\n"          # stray fix print
        "WARNING: bond atoms missing (src/ntopo.cpp:1)\n"   # stray warning
        "    200   298.0   0.820\n"
        "    300   298.0   0.830\n"
        "    400   298.0   0.840\n"
        "Loop time of 2 on 1 procs\n"
    )
    (tmp_path / "log.lammps").write_text(log)
    rho = read_health_observable(str(tmp_path), "density")
    assert rho == pytest.approx(0.835, abs=0.02)   # the settled rows, not 0.015


def test_time_first_header_is_detected(tmp_path):
    # `thermo_style custom time temp density` -> header starts with Time, not Step.
    log = ("units real\n"
           "   Time   Temp   Density\n"
           "    0.0   298.0   0.83\n"
           "    1.0   298.0   0.84\n"
           "Loop time of 1 on 1 procs\n")
    (tmp_path / "log.lammps").write_text(log)
    assert read_health_observable(str(tmp_path), "density") == pytest.approx(0.835, abs=0.02)


def test_density_unit_conversion(tmp_path):
    # si logs density in kg/m^3; must convert to g/cm^3 before the band check.
    si = ("units si\n   Step Temp Density\n   0 298 1000.0\n   100 298 1000.0\n"
          "Loop time of 1 on 1 procs\n")
    (tmp_path / "log.lammps").write_text(si)
    assert read_health_observable(str(tmp_path), "density") == pytest.approx(1.0, abs=0.01)


def test_unsupported_units_return_none(tmp_path):
    # lj reduced density has no g/cm^3 equivalent -> skip rather than misjudge.
    lj = ("units lj\n   Step Temp Density\n   0 1.0 0.80\n   100 1.0 0.80\n"
          "Loop time of 1 on 1 procs\n")
    (tmp_path / "log.lammps").write_text(lj)
    assert read_health_observable(str(tmp_path), "density") is None


def test_missing_dir_and_empty_dir_return_none():
    assert read_health_observable("/no/such/dir", "density") is None
    empty = tempfile.mkdtemp()
    assert read_health_observable(empty, "density") is None
