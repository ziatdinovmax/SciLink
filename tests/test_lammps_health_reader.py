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
    parse_thermo_log,
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


def test_fix_print_step_line_is_not_a_header(tmp_path):
    # A `fix print "Step $s ..."` line begins with "Step" but contains numbers;
    # it must not be taken as a thermo header (which would drop the rows after
    # it and return the mean of only the first rows).
    log = (
        "units real\n"
        "   Step   Temp   Density\n"
        "      0   300.0   0.010\n"
        "    100   300.0   0.015\n"
        "Step 200 reached Temp 300 rho 0.02\n"   # fix print idiom, not a header
        "    200   298.0   0.820\n"
        "    300   298.0   0.830\n"
        "    400   298.0   0.840\n"
        "Loop time of 2 on 1 procs\n"
    )
    (tmp_path / "log.lammps").write_text(log)
    rho = read_health_observable(str(tmp_path), "density")
    assert rho == pytest.approx(0.835, abs=0.02)   # settled rows, not 0.0125


def test_all_words_fix_print_step_line_is_not_a_header(tmp_path):
    # An all-WORDS line beginning with "Step" (no numbers) must not be taken as
    # a header: it would otherwise steal the rows after it and leave the mean on
    # the startup rows, failing a healthy run.
    log = (
        "units real\n"
        "   Step   Temp   Density\n"
        "      0   300.0   0.010\n"
        "    100   300.0   0.015\n"
        "Step reseed complete; continuing production\n"   # all words, starts "Step"
        "    200   298.0   0.820\n"
        "    300   298.0   0.830\n"
        "    400   298.0   0.840\n"
        "Loop time of 2 on 1 procs\n"
    )
    (tmp_path / "log.lammps").write_text(log)
    rho = read_health_observable(str(tmp_path), "density")
    assert rho == pytest.approx(0.835, abs=0.02)   # settled rows, not 0.0125


def test_header_confirmed_only_by_following_numeric_row(tmp_path):
    # Two real blocks separated by an all-words stray "Step" line; the parser
    # must keep exactly the two real blocks, not spawn a phantom one.
    log = (
        "   Step Temp Density\n   0 300 0.10\n   100 300 0.20\n"
        "Loop time of 1 on 1 procs\n"
        "Step change: switching ensemble\n"              # not a header
        "   Step Temp Density\n   0 298 0.83\n   100 298 0.84\n"
        "Loop time of 2 on 1 procs\n"
    )
    (tmp_path / "log.lammps").write_text(log)
    blocks = parse_thermo_log(tmp_path / "log.lammps")
    assert len(blocks) == 2
    assert blocks[-1]["Density"] == [0.83, 0.84]


def test_stale_log_from_earlier_phase_is_ignored(tmp_path):
    # A log older than the run start (shared run_dir, this phase wrote no log)
    # must be treated as absent, not judged.
    import os
    import time
    log = ("units real\n   Step Temp Density\n   0 298 0.83\n   100 298 0.83\n"
           "Loop time of 1 on 1 procs\n")
    (tmp_path / "log.lammps").write_text(log)
    old = time.time() - 3600
    os.utime(tmp_path / "log.lammps", (old, old))
    # since = now: the hour-old log predates the run -> None
    assert read_health_observable(str(tmp_path), "density", since=time.time()) is None
    # without a since, the log is read normally
    assert read_health_observable(str(tmp_path), "density") == pytest.approx(0.83, abs=0.01)


def test_parse_thermo_log_column_filter_and_last_block():
    log = (
        "   Step   Temp   Density\n   0 300 0.10\n   100 300 0.20\n"
        "Loop time of 1 on 1 procs\n"
        "   Step   Temp   Density\n   0 298 0.83\n   100 298 0.84\n"
        "Loop time of 2 on 1 procs\n"
    )
    import tempfile as _t
    p = Path(_t.mkdtemp()) / "log.lammps"
    p.write_text(log)
    # column filter keeps only Density; two blocks returned
    blocks = parse_thermo_log(p, columns={"density"})
    assert len(blocks) == 2
    assert set(blocks[0]) == {"Density"}
    assert blocks[-1]["Density"] == [0.83, 0.84]


def test_parse_thermo_log_parity_on_real_fixture():
    # The canonical parser (shared with the MLIP agent) reads the real fixture's
    # final-block Density; preserves original-case keys.
    blocks = parse_thermo_log(FIXTURES / "log_healthy.lammps")
    assert blocks, "expected at least one thermo block"
    assert "Density" in blocks[-1]
    tail = blocks[-1]["Density"]
    assert sum(tail[len(tail) // 2:]) / (len(tail) - len(tail) // 2) == pytest.approx(0.83, abs=0.05)


def test_mlip_wrapper_merges_blocks_on_real_fixture():
    # Exercise the MLIP agent's _parse_lammps_thermo wrapper itself (not just
    # parse_thermo_log) so the merge-and-original-case behaviour it relies on is
    # covered. The wrapper uses no instance state, so a bare instance is fine.
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent
    agent = MLIPAgent.__new__(MLIPAgent)
    merged = agent._parse_lammps_thermo(str(FIXTURES / "log_healthy.lammps"))
    assert "Density" in merged and "Temp" in merged     # original-case keys
    tail = merged["Density"][-100:]
    assert sum(tail) / len(tail) == pytest.approx(0.83, abs=0.05)


def test_missing_dir_and_empty_dir_return_none():
    assert read_health_observable("/no/such/dir", "density") is None
    empty = tempfile.mkdtemp()
    assert read_health_observable(empty, "density") is None
