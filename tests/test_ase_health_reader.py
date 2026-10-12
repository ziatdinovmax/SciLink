"""Ground the ASE health reader in the REAL ASE thermo-log format.

The ASE MD runner writes a `# step time(ps) PE(eV) KE(eV) T(K)` header and
whitespace-delimited rows (scilink/agents/sim_agents/_ase_runner.py). These
fixtures mirror that format — including a truncated final row — so the reader
and its test can't share one unchecked mental model of the log. The reader
serves temperature (the one physical-sanity observable the driver logs) and
returns None for anything it can't read, so the gate SKIPS rather than
misfires (MLIP-execution-in-loop, Blocker 3).

Fixtures:
  thermo_healthy.log    equilibrated at ~300 K            -> PASS
  thermo_exploded.log   thermostat blow-up ~88000 K       -> FAIL
  thermo_no_temp.log    a log with no T(K) column         -> SKIP (None)
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.health import evaluate_health  # noqa: E402
from scilink.skills.molecular_dynamics.ase.ase import (  # noqa: E402
    read_health_observable,
)

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "ase"
# the loose temperature band ase.md declares
TEMP_SPEC = [{"observable": "temperature", "min": 1.0, "max": 50000.0}]


def _run_dir(fixture_name):
    """A temp dir containing the fixture renamed to thermo.log."""
    td = Path(tempfile.mkdtemp())
    shutil.copy(FIXTURES / fixture_name, td / "thermo.log")
    return str(td)


def test_healthy_log_reads_physical_temperature():
    t = read_health_observable(_run_dir("thermo_healthy.log"), "temperature")
    assert t == pytest.approx(300.0, abs=2.0)
    assert evaluate_health({"temperature": t}, TEMP_SPEC) == []  # PASS


def test_exploded_log_trips_the_gate():
    t = read_health_observable(_run_dir("thermo_exploded.log"), "temperature")
    assert t > 50000.0                       # a thermostat blow-up
    violations = evaluate_health({"temperature": t}, TEMP_SPEC)
    assert len(violations) == 1              # the gate FAILS it
    assert "temperature" in violations[0].reason


def test_truncated_final_row_is_ignored_not_fatal():
    # thermo_exploded.log ends in a short (4-token) truncated row; it must be
    # skipped, not read as data or raise.
    d = _run_dir("thermo_exploded.log")
    # back-half mean of the four full rows, truncated row excluded
    t = read_health_observable(d, "temperature")
    assert t == pytest.approx((84000.55 + 93000.77) / 2, abs=1.0)


def test_unlogged_observable_returns_none_and_gate_skips():
    # density is not something the ASE MD driver logs (and isn't declared);
    # the reader returns None, so a density band would simply skip.
    d = _run_dir("thermo_healthy.log")
    assert read_health_observable(d, "density") is None
    assert read_health_observable(d, "pressure") is None
    assert evaluate_health({"density": None},
                           [{"observable": "density", "min": 0.02, "max": 30.0}]) == []


def test_missing_temperature_column_returns_none():
    # A log whose header has no T(K) column can't be judged on temperature ->
    # None (skip), not a fabricated value.
    assert read_health_observable(_run_dir("thermo_no_temp.log"),
                                  "temperature") is None


def test_energies_are_readable():
    d = _run_dir("thermo_healthy.log")
    assert read_health_observable(d, "potential_energy") < 0      # eV, bound
    assert read_health_observable(d, "kinetic_energy") > 0


def test_stale_log_from_earlier_phase_is_ignored():
    import os
    import time
    d = _run_dir("thermo_healthy.log")
    old = time.time() - 3600
    os.utime(Path(d) / "thermo.log", (old, old))
    # since = now: the hour-old log predates the run -> None
    assert read_health_observable(d, "temperature", since=time.time()) is None
    # without a since, the log is read normally
    assert read_health_observable(d, "temperature") == pytest.approx(300.0, abs=2.0)


def test_missing_and_empty_dir_return_none():
    assert read_health_observable("/no/such/dir", "temperature") is None
    empty = tempfile.mkdtemp()
    assert read_health_observable(empty, "temperature") is None
