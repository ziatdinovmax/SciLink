"""Unit tests for the engine-neutral convergence comparator."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scilink.agents.sim_agents.convergence import (  # noqa: E402
    converged_setting, ConvergenceResult, run_convergence_sweep, SweepResult,
    converge_parameters, ParameterConvergence, _floor_ladder,
)


def test_clear_plateau_adopts_cheapest_trustworthy_setting():
    # Energy/atom (eV) vs ENCUT (eV): flat from 400 up, within 1 meV.
    obs = [(300, -5.20), (400, -5.401), (500, -5.4015), (600, -5.4012)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is True
    assert r.setting == 400            # cheapest within tol of the top
    assert r.value == -5.4012          # value from the most-accurate setting


def test_not_converged_when_still_drifting_at_top():
    obs = [(300, -5.0), (400, -5.2), (500, -5.35), (600, -5.47)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is False
    assert r.setting is None
    assert r.value == -5.47            # best estimate still reported
    assert "extend" in r.reason


def test_single_setting_cannot_demonstrate_plateau():
    r = converged_setting([(500, -5.4)], tolerance=0.01)
    assert r.converged is False
    assert r.setting is None
    assert r.value == -5.4


def test_empty_observations():
    r = converged_setting([], tolerance=0.01)
    assert r.converged is False and r.value is None
    assert "no readable" in r.reason


def test_none_values_are_skipped():
    # The 500 run failed; the plateau is still demonstrable from 400 vs 600.
    obs = [(300, -5.2), (400, -5.401), (500, None), (600, -5.4012)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is True
    assert r.setting == 400


def test_only_top_agrees_is_not_a_plateau():
    # Every cheaper setting is outside tol of the top; only the top agrees with
    # itself — not a demonstrated plateau.
    obs = [(300, -5.0), (400, -5.30), (500, -5.42)]
    r = converged_setting(obs, tolerance=0.001)
    assert r.converged is False
    assert r.value == -5.42


def test_tolerance_boundary_is_inclusive():
    obs = [(400, -5.400), (500, -5.401)]   # delta exactly 0.001
    assert converged_setting(obs, tolerance=0.001).converged is True
    assert converged_setting(obs, tolerance=0.0009).converged is False


def test_lattice_constant_ladder_over_kpoints():
    # k-mesh density ladder; lattice constant (Å) plateaus at 6x6x6.
    obs = [("2x2x2", 3.68), ("4x4x4", 3.615), ("6x6x6", 3.611), ("8x8x8", 3.612)]
    r = converged_setting(obs, tolerance=0.005)
    assert r.converged is True
    assert r.setting == "4x4x4"        # within 0.005 Å of the top from here up
    assert r.value == 3.612


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError):
        converged_setting([(1, 1.0), (2, 1.0)], tolerance=-0.1)


def test_result_is_dataclass_with_deltas():
    r = converged_setting([(400, -5.401), (500, -5.4012)], tolerance=0.001)
    assert isinstance(r, ConvergenceResult)
    assert [s for s, _ in r.deltas] == [400, 500]


# ---------------------------------------------------------------------------
# run_convergence_sweep — engine-neutral driver, tested with fakes
# ---------------------------------------------------------------------------

def _fake_energies(mapping):
    """read_observable that maps a run_dir string to a canned energy."""
    return lambda run_dir: mapping.get(run_dir)


def test_sweep_converges_and_reports_cheapest_setting():
    ladder = [300, 400, 500, 600]
    built = {}

    def build_member(setting):
        m = {"INCAR": f"ENCUT = {setting}", "POSCAR": "..."}
        built[setting] = m
        return m

    def run_ladder(members):
        # Verify each rung got its own param-set deck, then "run" it.
        assert set(members) == set(ladder)
        return {s: f"/run/{s}" for s in members}

    energies = {"/run/300": -5.20, "/run/400": -5.401,
                "/run/500": -5.4015, "/run/600": -5.4012}

    res = run_convergence_sweep(
        ladder=ladder, build_member=build_member, run_ladder=run_ladder,
        read_observable=_fake_energies(energies), tolerance=0.001,
        param_name="ENCUT",
    )
    assert isinstance(res, SweepResult)
    assert res.param_name == "ENCUT"
    assert res.convergence.converged is True
    assert res.convergence.setting == 400
    assert res.convergence.value == -5.4012
    assert built[600]["INCAR"] == "ENCUT = 600"          # param-setter ran per rung
    assert res.observations[0] == (300, -5.20)


def test_sweep_marks_failed_rung_as_none():
    ladder = [300, 400, 500]

    def run_ladder(members):
        # The 400 rung failed to produce a run dir.
        return {300: "/run/300", 500: "/run/500"}

    res = run_convergence_sweep(
        ladder=ladder, build_member=lambda s: {"INCAR": str(s)},
        run_ladder=run_ladder,
        read_observable=_fake_energies({"/run/300": -5.0, "/run/500": -5.4}),
        tolerance=0.001,
    )
    assert res.observations == [(300, -5.0), (400, None), (500, -5.4)]
    assert res.convergence.converged is False   # only two points, still drifting


def test_sweep_unreadable_observable_is_none():
    ladder = [400, 500]

    res = run_convergence_sweep(
        ladder=ladder, build_member=lambda s: {"INCAR": str(s)},
        run_ladder=lambda m: {s: f"/run/{s}" for s in m},
        read_observable=lambda run_dir: None,   # parser found nothing
        tolerance=0.001,
    )
    assert [v for _, v in res.observations] == [None, None]
    assert res.convergence.converged is False


# ---------------------------------------------------------------------------
# converge_parameters — sequential multi-parameter orchestration (fakes)
# ---------------------------------------------------------------------------

def _fake_set_param(inputs, param, value):
    # Store the parameter's current value in the deck dict for inspection.
    return {**inputs, param: value}


def test_converge_parameters_adopts_in_sequence():
    # ENCUT plateaus at 400; k-points (KSPACING, descending) plateaus at 0.3.
    energies = {
        # ENCUT ladder run dirs
        "/run/ENCUT/300": -5.20, "/run/ENCUT/400": -5.401,
        "/run/ENCUT/500": -5.4015, "/run/ENCUT/600": -5.4012,
        # k-points ladder run dirs
        "/run/k-points/0.5": -5.30, "/run/k-points/0.4": -5.401,
        "/run/k-points/0.3": -5.4013, "/run/k-points/0.2": -5.4012,
    }
    seen_members = {}

    def run_ladder(param, members):
        seen_members[param] = members
        return {s: f"/run/{param}/{s}" for s in members}

    specs = [
        {"parameter": "ENCUT", "ladder": [300, 400, 500, 600],
         "observable": "e", "tolerance": 0.001},
        {"parameter": "k-points", "ladder": [0.5, 0.4, 0.3, 0.2],
         "observable": "e", "tolerance": 0.001},
    ]
    pc = converge_parameters(
        base_inputs={"INCAR": "base"}, specs=specs,
        set_param=_fake_set_param,
        read_observable=lambda d, o: energies.get(d),
        run_ladder=run_ladder,
    )
    assert isinstance(pc, ParameterConvergence)
    assert pc.all_converged is True
    assert pc.final_inputs["ENCUT"] == 400
    assert pc.final_inputs["k-points"] == 0.4    # cheapest KSPACING within tol
    # Sequential adopt: the k-points members were built on the adopted ENCUT.
    assert all(m["ENCUT"] == 400 for m in seen_members["k-points"].values())


def test_converge_parameters_leaves_unconverged_param_unadopted():
    # ENCUT never plateaus; k-points does. ENCUT stays at base (not adopted).
    energies = {
        "/run/ENCUT/300": -5.0, "/run/ENCUT/400": -5.2, "/run/ENCUT/500": -5.4,
        "/run/k-points/0.5": -5.30, "/run/k-points/0.4": -5.401,
        "/run/k-points/0.3": -5.4013,
    }
    specs = [
        {"parameter": "ENCUT", "ladder": [300, 400, 500],
         "observable": "e", "tolerance": 0.001},
        {"parameter": "k-points", "ladder": [0.5, 0.4, 0.3],
         "observable": "e", "tolerance": 0.001},
    ]
    pc = converge_parameters(
        base_inputs={"INCAR": "base"}, specs=specs,
        set_param=_fake_set_param,
        read_observable=lambda d, o: energies.get(d),
        run_ladder=lambda p, members: {s: f"/run/{p}/{s}" for s in members},
    )
    assert pc.all_converged is False
    assert "ENCUT" not in pc.final_inputs           # not adopted
    assert pc.final_inputs["k-points"] == 0.4       # cheapest within tol
    assert len(pc.sweeps) == 2


# ---------------------------------------------------------------------------
# Flooring the ladder at the base deck's validated value (bug #1)
# ---------------------------------------------------------------------------

def test_floor_ladder_ascending_base_between_rungs():
    # ENCUT 520 (validated) between 500 and 600: drop 300/400/500, base is cheapest.
    eff, floored = _floor_ladder([300, 400, 500, 600, 700], 520, "ascending")
    assert eff == [520, 600, 700] and floored is True


def test_floor_ladder_ascending_base_on_rung():
    eff, floored = _floor_ladder([300, 400, 500, 600, 700], 400, "ascending")
    assert eff == [400, 500, 600, 700] and floored is True   # 300 dropped, no prepend


def test_floor_ladder_descending_kspacing():
    # KSPACING base 0.23 (validated): keep denser-or-equal, base is cheapest.
    eff, floored = _floor_ladder([0.5, 0.4, 0.3, 0.25, 0.2, 0.15], 0.23, "descending")
    assert eff == [0.23, 0.2, 0.15] and floored is True


def test_floor_ladder_no_change_when_base_at_bottom():
    eff, floored = _floor_ladder([300, 400, 500], 300, "ascending")
    assert eff == [300, 400, 500] and floored is False


def test_converge_parameters_never_adopts_below_base():
    # Energy is flat from 400 up, so WITHOUT a floor the sweep would adopt 400.
    # The base deck validated ENCUT at 520 (>= 1.3x ENMAX), so the sweep must
    # run only 520/600/700 and adopt no lower than 520.
    energies = {
        "/run/ENCUT/520": -5.401, "/run/ENCUT/600": -5.4012,
        "/run/ENCUT/700": -5.4013,
    }
    seen = {}

    def run_ladder(param, members):
        seen[param] = set(members)
        return {s: f"/run/{param}/{s}" for s in members}

    specs = [{"parameter": "ENCUT", "ladder": [300, 400, 500, 600, 700],
              "direction": "ascending", "observable": "e", "tolerance": 0.001}]
    pc = converge_parameters(
        base_inputs={"INCAR": "ENCUT = 520"}, specs=specs,
        set_param=_fake_set_param,
        read_observable=lambda d, o: energies.get(d),
        run_ladder=run_ladder,
        get_param=lambda inputs, param: 520,   # the validated base value
    )
    assert seen["ENCUT"] == {520, 600, 700}        # 300/400/500 never run
    assert pc.final_inputs["ENCUT"] == 520         # cheapest VALID rung, not 400
    assert pc.floors == {"ENCUT": 520}


def test_converge_parameters_skips_sweep_when_floor_required_but_unreadable():
    # ENCUT with skip_without_floor: if the base value can't be read (deck left
    # ENCUT unset) running the 300 eV ladder blind could adopt below ENMAX, so
    # the sweep is skipped rather than run.
    seen = {}

    def run_ladder(param, members):
        seen[param] = set(members)
        return {s: f"/run/{param}/{s}" for s in members}

    specs = [{"parameter": "ENCUT", "ladder": [300, 400, 500],
              "observable": "e", "tolerance": 0.001, "skip_without_floor": True}]
    pc = converge_parameters(
        base_inputs={"INCAR": "PREC = Accurate\n"}, specs=specs,
        set_param=_fake_set_param,
        read_observable=lambda d, o: -5.0,
        run_ladder=run_ladder,
        get_param=lambda inputs, param: None,   # ENCUT unreadable
    )
    assert "ENCUT" not in seen                   # ladder never ran
    assert pc.sweeps[0].convergence.converged is False
    assert "ENCUT" not in pc.final_inputs        # nothing adopted
    assert pc.all_converged is False


def test_converge_parameters_no_floor_when_base_unreadable():
    # get_param returns None (no skip_without_floor) -> full ladder.
    energies = {
        "/run/ENCUT/300": -5.20, "/run/ENCUT/400": -5.401,
        "/run/ENCUT/500": -5.4012,
    }
    seen = {}

    def run_ladder(param, members):
        seen[param] = set(members)
        return {s: f"/run/{param}/{s}" for s in members}

    specs = [{"parameter": "ENCUT", "ladder": [300, 400, 500],
              "observable": "e", "tolerance": 0.001}]
    pc = converge_parameters(
        base_inputs={"INCAR": "base"}, specs=specs,
        set_param=_fake_set_param,
        read_observable=lambda d, o: energies.get(d),
        run_ladder=run_ladder,
        get_param=lambda inputs, param: None,
    )
    assert seen["ENCUT"] == {300, 400, 500}        # nothing floored
    assert pc.floors == {}
    assert pc.final_inputs["ENCUT"] == 400


# ---------------------------------------------------------------------------
# Regression: real VASP Cu energies from a live convergence run
# ---------------------------------------------------------------------------

# energy/atom (eV) at each KSPACING rung, ENCUT already converged to 400 eV.
_CU_KPOINTS = [
    (0.5, -3.75676533), (0.4, -3.68430451), (0.3, -3.72132136),
    (0.25, -3.7166153), (0.2, -3.71651444), (0.15, -3.71432999),
    (0.12, -3.71703282), (0.1, -3.71568771),
]
_CU_ENCUT = [
    (300, -3.72864119), (400, -3.72132136), (500, -3.72140217),
    (600, -3.72103981), (700, -3.72091285),
]


def test_cu_encut_converges_at_400():
    r = converged_setting(_CU_ENCUT, tolerance=0.001)
    assert r.converged is True and r.setting == 400


def test_cu_kpoints_tolerance_too_strict_does_not_converge():
    # 1 meV/atom is below the metal's k-sampling wobble — correctly not converged.
    assert converged_setting(_CU_KPOINTS, tolerance=0.001).converged is False


def test_cu_kpoints_converges_at_meaningful_tolerance():
    # 5 meV/atom (the skill's k-point tolerance) converges at KSPACING 0.25
    # (13x13x13 for this cell) — in the 11-16 mesh range a practitioner accepts.
    r = converged_setting(_CU_KPOINTS, tolerance=0.005)
    assert r.converged is True
    assert r.setting == 0.25


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
