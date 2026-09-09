"""#579 — the constrained planner works in ONE space. An index-encoded
categorical input reaches the optimizer as codes, but the planner (an LLM
reasoning over physical constraints) is shown the level names — in the
bounds, the acquisition landscape, the data summary, the current best and
the unconstrained picks — and every value it returns is mapped back to a
code before it flows downstream (where the orchestrator decodes once)."""
import json
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scilink.agents.planning_agents.bo_agent import BOAgent

LEVELS = {"catalyst_conc_mM": ["0.1", "0.5", "1", "2", "5"]}


def test_code_and_level_mapping():
    lv = LEVELS["catalyst_conc_mM"]
    assert BOAgent._level_of(lv, 3) == "2" and BOAgent._level_of(lv, 2.6) == "2"
    assert BOAgent._level_of(lv, -1) == "0.1" and BOAgent._level_of(lv, 9) == "5"
    # exact level names (numeric-looking values compare as numbers)
    assert BOAgent._code_of(lv, "2") == 3 and BOAgent._code_of(lv, 2.0) == 3 and BOAgent._code_of(lv, "2.0") == 3
    assert BOAgent._code_of(lv, 0.1) == 0 and BOAgent._code_of(lv, "5") == 4
    # a nearby value snaps only within 1 % of the level spacing
    assert BOAgent._code_of(lv, 2.005) == 3
    assert BOAgent._code_of(lv, 1.95) is None and BOAgent._code_of(lv, 3.0) is None
    # string levels: exact only
    assert BOAgent._code_of(["A", "B"], "B") == 1 and BOAgent._code_of(["A", "B"], "b") is None


def _agent(planner_json):
    """A BOAgent shell whose model returns the planner's JSON verbatim."""
    a = BOAgent.__new__(BOAgent)
    a.model = SimpleNamespace(generate_content=lambda *args, **kw: SimpleNamespace(text=json.dumps(planner_json)))
    a.generation_config = None
    a.prompts = []
    real = a.model.generate_content

    def capture(*args, **kw):
        a.prompts.append(args[0] if args else kw.get("contents"))
        return real(*args, **kw)
    a.model.generate_content = capture
    return a


def test_planner_sees_levels_and_its_physical_answer_becomes_a_code():
    planner = {"batch": [{"experiment_id": 1, "params": {"catalyst_conc_mM": 2.0, "temperature_C": 80.0}}],
               "allocation_strategy": "s", "coverage_summary": "c", "trade_offs": "t"}
    a = _agent(planner)
    recs, meta, err = a._plan_constrained_batch(
        objective_text="max yield", input_cols=["catalyst_conc_mM", "temperature_C"],
        input_bounds=[[0.0, 4.0], [40.0, 80.0]], batch_size=1, acq_summary="landscape",
        physical_constraints="only 0.1, 0.5, 1.0, 2.0, 5.0 mM stocks",
        unconstrained_recommendations=[{"catalyst_conc_mM": 1.0, "temperature_C": 80.0}],
        data_summary_str="| stats |", current_best={"catalyst_conc_mM": 2.0, "temperature_C": 73.0},
        current_best_value={"yield": 52.0}, budget_ctx={"budget_phase": "unlimited"},
        is_moo=False, pareto_front=None, input_levels=LEVELS)
    assert err is None and meta["valid_count"] == 1
    # the planner said 2.0 mM (physical) → code 3 downstream
    assert recs == [{"catalyst_conc_mM": 3.0, "temperature_C": 80.0}]
    prompt = "\n".join(p if isinstance(p, str) else str(p) for p in a.prompts[0])
    assert 'catalyst_conc_mM: CATEGORICAL — one of ["0.1", "0.5", "1", "2", "5"]' in prompt
    assert "[0.0, 4.0]" not in prompt          # no code-space bound for the categorical input
    assert "temperature_C: [40.0, 80.0]" in prompt


def test_planner_value_off_the_levels_is_a_validation_error():
    planner = {"batch": [{"experiment_id": 1, "params": {"catalyst_conc_mM": 3.0, "temperature_C": 80.0}},
                         {"experiment_id": 2, "params": {"catalyst_conc_mM": "5", "temperature_C": 60.0}}]}
    a = _agent(planner)
    recs, meta, err = a._plan_constrained_batch(
        objective_text="o", input_cols=["catalyst_conc_mM", "temperature_C"],
        input_bounds=[[0.0, 4.0], [40.0, 80.0]], batch_size=2, acq_summary="l",
        physical_constraints="stocks", unconstrained_recommendations=[], data_summary_str="s",
        current_best={}, current_best_value={}, budget_ctx={"budget_phase": "unlimited"},
        input_levels=LEVELS)
    assert err is None
    assert recs == [{"catalyst_conc_mM": 4.0, "temperature_C": 60.0}]   # only the valid one, as a code
    assert any("not one of the levels" in e for e in meta["validation_errors"])


def test_landscape_and_summary_render_level_names():
    a = BOAgent.__new__(BOAgent)
    a._cluster_acquisition_regions = lambda grid, acq, cols, bounds, n: [
        {"center": np.array([3.0, 80.0]), "acq_value": 0.9, "spread": [0.2, 1.0], "notes": ""},
        {"center": np.array([0.7, 60.0]), "acq_value": 0.4, "spread": [0.2, 1.0], "notes": ""}]
    optimizer = SimpleNamespace(evaluate_acquisition=lambda pts: np.ones(len(pts)))
    table = a._summarize_acquisition_landscape(optimizer, ["catalyst_conc_mM", "temperature_C"],
                                               [[0.0, 4.0], [40.0, 80.0]], input_levels=LEVELS)
    assert "| 1 | 2 | 80.0000 |" in table and "| 2 | 0.5 | 60.0000 |" in table   # codes 3 → "2", 0.7 → index 1 → "0.5"
    assert "3.0000" not in table
    df = pd.DataFrame({"catalyst_conc_mM": [0.0, 3.0, 4.0], "temperature_C": [40.0, 60.0, 80.0], "yield": [1, 2, 3]})
    phys = a._physical_frame(df, LEVELS)
    assert phys["catalyst_conc_mM"].tolist() == [0.1, 2.0, 5.0]
    assert a._decode_point({"catalyst_conc_mM": 4.0, "temperature_C": 80.0}, LEVELS) == {"catalyst_conc_mM": 5.0, "temperature_C": 80.0}
    # non-numeric levels stay strings
    assert a._physical_frame(pd.DataFrame({"solvent": [0.0, 1.0]}), {"solvent": ["water", "DMSO"]})["solvent"].tolist() == ["water", "DMSO"]


def test_stage_hands_the_planner_physical_inputs_and_keeps_codes_downstream():
    seen = {}

    def planner(**kw):
        seen.update(kw)
        return [{"catalyst_conc_mM": 3.0, "temperature_C": 75.0}], {"coverage_summary": "ok"}, None

    a = SimpleNamespace(_summarize_acquisition_landscape=lambda **kw: seen.setdefault("landscape_kw", kw) and "L",
                        _plan_constrained_batch=planner, _log_action=lambda **kw: None, _run_seed=None)
    a._physical_frame = types.MethodType(BOAgent._physical_frame, a)
    a._decode_point = types.MethodType(BOAgent._decode_point, a)
    a._level_of = BOAgent._level_of
    a._norm_level_label = BOAgent._norm_level_label
    c = SimpleNamespace(
        input_cols=["catalyst_conc_mM", "temperature_C"], target_cols=["yield"],
        X=np.array([[0.0, 40.0], [3.0, 60.0]]), y=np.array([[0.4], [0.9]]), optimizer=object(),
        minimize_mask=[], is_moo=False, input_bounds=[[0.0, 4.0], [40.0, 80.0]], objective_text="o",
        budget_ctx={"budget_phase": "unlimited"}, batch_size=1, physical_constraints="stocks",
        unconstrained_recommendations=[{"catalyst_conc_mM": 2.0, "temperature_C": 82.0}],
        constrained_metadata=None, next_x_batch=None, recommendations=None, input_levels=LEVELS,
        df=pd.DataFrame({"catalyst_conc_mM": [0.0, 3.0], "temperature_C": [40.0, 60.0], "yield": [0.4, 0.9]}),
        valid_config={"acquisition_strategy": {"type": "log_ei"}, "model_config": {}}, plot_path="p",
        inspection={}, acq_plot_path=None, acq_data_path=None, candidate_pool_info=None, data_path="d",
        experimental_budget=None, save_acq=False, plot_acq=False, cat_dims=[0], output_dir="/tmp/x")
    BOAgent._stage_constrained_batch(a, c)
    # the planner saw physical values: current best 2 mM (code 3), unconstrained pick 1 mM (code 2)
    assert seen["current_best"] == {"catalyst_conc_mM": 2.0, "temperature_C": 60.0}
    assert seen["unconstrained_recommendations"] == [{"catalyst_conc_mM": 1.0, "temperature_C": 82.0}]
    assert seen["input_levels"] == LEVELS and seen["landscape_kw"]["input_levels"] == LEVELS
    assert "0.1" in seen["data_summary_str"] or "0.1" in str(seen["data_summary_str"])
    # downstream stays in code space (the orchestrator decodes once)
    assert c.recommendations == [{"catalyst_conc_mM": 3.0, "temperature_C": 75.0}]
    assert c.selection_method == "constrained_planner"
    res = BOAgent._stage_output(a, c)
    assert res["acqf_optimum"] == {"catalyst_conc_mM": 2.0, "temperature_C": 82.0}  # codes, as the recommendation
