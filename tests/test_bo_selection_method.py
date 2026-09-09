"""#564 — every BO result says how its recommendation was selected:
``acqf_optimizer`` (the acquisition optimum) or ``constrained_planner`` (a
deliberate, constraint-aware deviation chosen by the LLM planner), and a
planner-chosen point carries the optimum it deviated from. Exercises the
two stages directly with a stub agent, no LLM."""
import types
from types import SimpleNamespace

import numpy as np

from scilink.agents.planning_agents.bo_agent import BOAgent


def _Frame():
    import pandas as pd
    return pd.DataFrame({"T": [1.0, 3.0], "pH": [2.0, 4.0], "yield": [0.5, 0.9]})


def _ctx(batch_size=1, constraints=None):
    return SimpleNamespace(
        input_cols=["T", "pH"], target_cols=["yield"], X=np.array([[1.0, 2.0], [3.0, 4.0]]),
        y=np.array([[0.5], [0.9]]), optimizer=object(), minimize_mask=[], is_moo=False,
        df=_Frame(), input_bounds=[[0, 10], [0, 14]], objective_text="max yield",
        budget_ctx={"budget_phase": "n/a"}, batch_size=batch_size,
        physical_constraints=constraints,
        unconstrained_recommendations=[{"T": 5.0, "pH": 7.0}] if batch_size == 1
        else [{"T": 5.0, "pH": 7.0}, {"T": 6.0, "pH": 8.0}],
        constrained_metadata=None, next_x_batch=None, recommendations=None,
        # stage 8 fields
        valid_config={"acquisition_strategy": {"type": "log_ei"}, "model_config": {}},
        plot_path="p.png", inspection={}, acq_plot_path=None, acq_data_path=None,
        candidate_pool_info=None, data_path="d.csv", experimental_budget=None,
        save_acq=False, plot_acq=False, cat_dims=None, output_dir="/tmp/x",
    )


def _agent(planner):
    a = SimpleNamespace(
        _summarize_acquisition_landscape=lambda **kw: "landscape",
        _plan_constrained_batch=planner,
        _log_action=lambda **kw: None,
        _run_seed=None,
    )
    # the stage renders the planner's inputs in physical terms (#579)
    a._physical_frame = types.MethodType(BOAgent._physical_frame, a)
    a._decode_point = types.MethodType(BOAgent._decode_point, a)
    return a


def _run(agent, c):
    BOAgent._stage_constrained_batch(agent, c)
    return BOAgent._stage_output(agent, c)


def test_unconstrained_is_the_acquisition_optimum():
    res = _run(_agent(lambda **kw: (None, None, "unused")), _ctx())
    assert res["selection_method"] == "acqf_optimizer"
    assert "acqf_optimum" not in res and "constraint_aware" not in res
    assert res["next_parameters"] == {"T": 5.0, "pH": 7.0}


def test_planner_choice_is_marked_and_carries_the_optimum():
    planner = lambda **kw: ([{"T": 2.0, "pH": 7.0}], {"coverage_summary": "one level"}, None)  # noqa: E731
    res = _run(_agent(planner), _ctx(constraints="only T in {2, 4, 6}"))
    assert res["selection_method"] == "constrained_planner"
    assert res["next_parameters"] == {"T": 2.0, "pH": 7.0}
    assert res["acqf_optimum"] == {"T": 5.0, "pH": 7.0}
    assert res["deviation_from_acqf_optimum"] == {"T": -3.0, "pH": 0.0}
    assert res["constraint_aware"] is True and res["constrained_planning"]["coverage_summary"] == "one level"


def test_planner_failure_falls_back_to_the_optimum_and_says_so():
    res = _run(_agent(lambda **kw: (None, {"validation_errors": ["x"]}, "boom")),
               _ctx(constraints="plate layout"))
    assert res["selection_method"] == "acqf_optimizer"
    assert res["next_parameters"] == {"T": 5.0, "pH": 7.0}
    assert "acqf_optimum" not in res  # nothing deviated from


def test_batch_planner_choice_carries_the_whole_unconstrained_batch(tmp_path):
    planner = lambda **kw: ([{"T": 2.0, "pH": 7.0}, {"T": 4.0, "pH": 7.0}], {}, None)  # noqa: E731
    c = _ctx(batch_size=2, constraints="rows share T")
    c.output_dir = str(tmp_path); c.step_num = 1
    res = _run(_agent(planner), c)
    assert res["selection_method"] == "constrained_planner"
    assert res["acqf_optimum"] == [{"T": 5.0, "pH": 7.0}, {"T": 6.0, "pH": 8.0}]
    assert "deviation_from_acqf_optimum" not in res  # per-point deltas only for a single point
