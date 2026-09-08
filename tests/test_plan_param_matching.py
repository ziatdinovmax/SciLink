"""#67: the plan's optimization_params name knobs in the planner's words
('Temperature', 'Reaction time (min)'); the data names them as columns
('temperature_C', 'time_min'). run_optimization used an exact lookup, so the
plan's bounds and level universes were silently ignored and the search box
came from the observed data range. Now the names are resolved onto the input
columns by normalized tokens, unambiguously both ways, and unresolved plan
parameters are reported with their ranges.

Offline: scalarizer stubbed to a pass-through, BOAgent's loop stubbed to
capture its kwargs."""
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from scilink.agents.planning_agents.orchestrator_tools import _resolve_plan_params
from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent, AutonomyLevel,
)


# ------------------------------------------------------------ resolver --

def _resolve(names, cols):
    r, un, _ = _resolve_plan_params(names, cols)
    return r, un


def test_exact_and_case_insensitive_win_first():
    r, un = _resolve(["temperature_C", "Time_Min"], ["temperature_C", "time_min"])
    assert r == {"temperature_C": "temperature_C", "Time_Min": "time_min"} and un == []


def test_ambiguous_candidates_are_reported():
    r, un, cands = _resolve_plan_params(
        ["Temperature", "Pressure"], ["temperature_inlet_C", "temperature_outlet_C"])
    assert r == {} and un == ["Temperature", "Pressure"]
    assert cands == {"Temperature": ["temperature_inlet_C", "temperature_outlet_C"]}


def test_unit_suffix_and_phrasing_variants():
    r, un = _resolve(
        ["Temperature", "Reaction time (min)", "Catalyst loading (mol%)", "pH"],
        ["temperature_C", "time_min", "catalyst_loading", "pH", "yield"])
    assert r == {"Temperature": "temperature_C", "Reaction time (min)": "time_min",
                 "Catalyst loading (mol%)": "catalyst_loading", "pH": "pH"}
    assert un == []


def test_equal_tokens_disambiguate_from_a_longer_sibling():
    r, un = _resolve(["Temperature"], ["temperature_C", "temperature_ramp_C"])
    assert r == {"Temperature": "temperature_C"} and un == []


def test_ambiguous_name_stays_unmatched():
    r, un = _resolve(["Temperature"], ["temperature_inlet_C", "temperature_outlet_C"])
    assert r == {} and un == ["Temperature"]


def test_two_names_wanting_one_column_both_unmatched():
    # 'time' and 'Time (min)' are the same name once the unit word drops:
    # genuinely ambiguous, neither is guessed.
    r, un = _resolve(["time", "Time (min)"], ["time_min", "T"])
    assert r == {} and set(un) == {"time", "Time (min)"}
    r, un = _resolve(["reaction time", "residence time"], ["time_min"])
    assert r == {} and set(un) == {"reaction time", "residence time"}


def test_single_letter_and_unit_only_names_do_not_token_match():
    r, un = _resolve(["T", "C"], ["temperature_C", "time_min"])
    assert r == {} and un == ["T", "C"]
    r, _ = _resolve(["T"], ["T", "t"])
    assert r == {"T": "T"}                      # case-sensitive exact wins
    r, un = _resolve(["t"], ["T", "Time"])
    assert r == {"t": "T"} and un == []         # then case-insensitive


def test_unrelated_name_unmatched():
    r, un = _resolve(["Pressure"], ["temperature_C", "time_min"])
    assert r == {} and un == ["Pressure"]


# ---------------------------------------------------------- integration --

L6 = dict(temperature_C=[30.0, 30.0, 90.0, 90.0, 50.0, 78.0],
          time_min=[10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
          solvent=["DMF", "DMSO", "DMF", "DMSO", "DMF", "DMSO"],
          yield_pct=[8.56, 3.88, 22.47, 10.81, 48.83, 61.1])


@pytest.fixture
def orch(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        o = PlanningOrchestratorAgent(
            base_dir=str(tmp_path / "session"), api_key="sk-dummy",
            autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(data_dir))
    o.scalarizer.scalarize = lambda **kw: {
        "status": "success", "metrics": dict(L6), "source_script": None,
        "column_roles": {"inputs": ["temperature_C", "time_min", "solvent"],
                         "targets": ["yield_pct"],
                         "input_types": {"solvent": "categorical"}},
        "passthrough": True, "error": None}
    csv = data_dir / "seed.csv"
    csv.write_text("temperature_C,time_min,solvent,yield_pct\n" + "\n".join(
        f"{a},{b},{c},{d}" for a, b, c, d in zip(*L6.values())) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        o.tools.execute_tool("analyze_file", file_path=str(csv), extraction_goal="x",
                             inputs=["temperature_C", "time_min", "solvent"],
                             targets=["yield_pct"],
                             input_types={"solvent": "categorical"})
    o._captured = {}

    def fake_loop(**kw):
        o._captured.clear(); o._captured.update(kw)
        return {"status": "success", "next_parameters": {}, "strategy": {}}
    o.bo.run_optimization_loop = fake_loop
    return o


def _plan(*params):
    return {"proposed_experiments": [{"optimization_params": list(params)}]}


def _run(o, **kw):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        out = json.loads(o.tools.execute_tool("run_optimization", **kw))
    return out, buf.getvalue()


def test_plan_bounds_apply_through_name_variants(orch):
    orch.planner.state["current_plan"] = _plan(
        {"parameter_name": "Temperature", "parameter_type": "continuous",
         "min_value": 20, "max_value": 120},
        {"parameter_name": "Reaction time (min)", "min_value": 5, "max_value": 60},
        {"parameter_name": "Solvent", "parameter_type": "categorical",
         "levels": ["DMF", "DMSO", "MeCN"]})
    out, log = _run(orch)
    assert out["status"] == "success"
    assert orch._captured["input_bounds"] == [[20.0, 120.0], [5.0, 60.0], [0.0, 2.0]]
    assert out["input_bounds_source"] == {"temperature_C": "planner", "time_min": "planner",
                                          "solvent": "categorical"}
    assert out["plan_param_matches"] == {"temperature_C": "Temperature",
                                         "time_min": "Reaction time (min)",
                                         "solvent": "Solvent"}
    # Planner is authoritative on the level universe, including unobserved MeCN.
    assert orch.expected_input_levels["solvent"] == ["DMF", "DMSO", "MeCN"]
    assert "plan parameter 'Temperature'" in log
    assert "input_bounds_warnings" not in out or not any(
        "derived from the observed data" in w for w in out["input_bounds_warnings"])


def test_unmatched_plan_param_is_reported_with_its_range(orch):
    orch.planner.state["current_plan"] = _plan(
        {"parameter_name": "Pressure (bar)", "min_value": 1, "max_value": 10},
        {"parameter_name": "Temperature", "min_value": 20, "max_value": 120})
    out, _ = _run(orch)
    assert orch._captured["input_bounds"][0] == [20.0, 120.0]
    assert out["input_bounds_source"]["time_min"] == "data"
    w = " ".join(out["input_bounds_warnings"])
    assert "Plan parameter 'Pressure (bar)' (range [1, 10]) matched no input column" in w
    assert "derived from the observed data" in w and "['time_min']" in w


def test_caller_bounds_still_beat_matched_plan_bounds(orch):
    orch.planner.state["current_plan"] = _plan(
        {"parameter_name": "Temperature", "min_value": 20, "max_value": 120})
    out, _ = _run(orch, input_bounds={"temperature_C": [40, 80]})
    assert orch._captured["input_bounds"][0] == [40.0, 80.0]
    assert out["input_bounds_source"]["temperature_C"] == "caller"


def test_exact_names_unchanged_and_no_matches_key(orch):
    orch.planner.state["current_plan"] = _plan(
        {"parameter_name": "temperature_C", "min_value": 20, "max_value": 120})
    out, log = _run(orch)
    assert orch._captured["input_bounds"][0] == [20.0, 120.0]
    assert "plan_param_matches" not in out
    assert "(Source: PLANNER)" in log


def test_ambiguous_plan_name_falls_to_data_with_warning(orch):
    orch.planner.state["current_plan"] = _plan(
        {"parameter_name": "Temperature", "min_value": 0, "max_value": 100},
        {"parameter_name": "temp (C)", "min_value": 5, "max_value": 60})
    out, _ = _run(orch)
    # Both reduce to 'temp*'-ish names? No: 'Temperature' -> {temperature},
    # 'temp (C)' -> {temp}; only the first equals temperature_C's tokens.
    assert orch._captured["input_bounds"][0] == [0.0, 100.0]
    assert any("Plan parameter 'temp (C)'" in w and "matched no input column" in w
               for w in out["input_bounds_warnings"])
    # Two names that are the same after normalization are never guessed.
    orch.planner.state["current_plan"] = _plan(
        {"parameter_name": "time", "min_value": 0, "max_value": 100},
        {"parameter_name": "Time (min)", "min_value": 5, "max_value": 60})
    out, _ = _run(orch)
    assert out["input_bounds_source"]["time_min"] == "data"
    amb = [w for w in out["input_bounds_warnings"] if "ambiguous between" in w]
    assert len(amb) == 2 and all("'time_min': [" in w for w in amb)
