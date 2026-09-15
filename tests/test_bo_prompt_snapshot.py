"""Offline prompt-assembly snapshot test for the OptimizationAgent
foundationalization PR 1 (stage extraction, issue #196).

PR 1 must be a behavior-preserving refactor: extracting the stages of
`run_optimization_loop` into methods and threading a `_skill_context(section)`
seam (returns "" -- no skills yet) must leave the assembled LLM prompts
BYTE-IDENTICAL. This test proves that empirically rather than by inspection.

It captures the fully-assembled strategy-configuration prompt (the large,
pre-fit, fully-deterministic prompt at the top of the loop) across every
branch -- SOO/MOO, budget, physical constraints, strategy hint, categorical
dims, minimize direction, and history-trend -- by mocking the model to record
the prompt and short-circuit before any GP fit. The captured prompts are
compared against a checked-in golden; regenerate with UPDATE_GOLDEN=1 (only
legitimate when the baseline prompts are intentionally changed).

No network, no API key, no torch in the captured path.
"""
import os
import json
import tempfile
from pathlib import Path

import pandas as pd

from scilink.agents.planning_agents.bo_agent import BOAgent

GOLDEN = Path(__file__).parent / "golden" / "bo_strategy_prompts.json"


class _StopAfterCapture(Exception):
    pass


class _CaptureModel:
    """Records the first generate_content prompt and aborts the run."""
    def __init__(self):
        self.captured = None

    def generate_content(self, parts, generation_config=None):
        # prompt_parts is a list of strings for the strategy-config call.
        self.captured = [p if isinstance(p, str) else f"<{type(p).__name__}>" for p in parts]
        raise _StopAfterCapture()


def _write_csv(path, n_targets=1, cat=False):
    """Deterministic small dataset so df.describe() is stable."""
    data = {
        "temp": [300.0, 350.0, 400.0, 450.0, 500.0, 550.0],
        "conc": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
        "y1": [10.0, 12.0, 9.0, 15.0, 11.0, 14.0],
    }
    if cat:
        data["conc"] = [0, 1, 0, 1, 0, 1]  # integer-encoded categorical
    if n_targets == 2:
        data["y2"] = [1.0, 2.0, 1.5, 2.5, 1.2, 2.2]
    pd.DataFrame(data).to_csv(path, index=False)


def _capture(scenario, tmp):
    """Run one scenario far enough to capture the strategy prompt."""
    data_path = Path(tmp) / "data.csv"
    out_dir = Path(tmp) / "out"
    out_dir.mkdir(exist_ok=True)
    _write_csv(data_path, n_targets=scenario.get("n_targets", 1), cat=scenario.get("cat", False))

    # Optional pre-seeded history to exercise the trend branch.
    if scenario.get("history"):
        hist = [{
            "step": 1, "data_points": 3,
            "config": {"rationale": "matern_2.5 + log_ei; broad explore"},
            "inspection": {"status": "ok", "reason": "fit looks fine",
                           "suggested_adjustments": {"noise": "min_noise_high"}},
        }]
        (out_dir / "bo_history.json").write_text(json.dumps(hist))

    agent = BOAgent(api_key="test", model_name="claude-opus-4-6", output_dir=str(out_dir))
    model = _CaptureModel()
    agent.model = model

    cols = ["temp", "conc"]
    bounds = [[300.0, 600.0], [0.0, 1.2]]
    targets = ["y1", "y2"] if scenario.get("n_targets") == 2 else ["y1"]
    kwargs = dict(
        data_path=str(data_path), objective_text="Maximize y1",
        input_cols=cols, input_bounds=bounds, target_cols=targets,
        output_dir=str(out_dir), batch_size=scenario.get("batch_size", 1),
    )
    for k in ("experimental_budget", "physical_constraints", "strategy_hint"):
        if k in scenario:
            kwargs[k] = scenario[k]
    if scenario.get("cat"):
        kwargs["cat_dims"] = [1]
    if scenario.get("minimize"):
        kwargs["target_directions"] = {"y1": "minimize"}

    try:
        agent.run_optimization_loop(**kwargs)
    except _StopAfterCapture:
        pass
    assert model.captured is not None, f"no prompt captured for {scenario['name']}"
    return model.captured


SCENARIOS = [
    {"name": "soo_basic"},
    {"name": "moo", "n_targets": 2},
    {"name": "soo_budget", "experimental_budget": 3},
    {"name": "soo_constraints", "physical_constraints": "96-well plate: rows share temperature."},
    {"name": "soo_hint", "strategy_hint": "use RBF kernel, high exploration"},
    {"name": "soo_catdims", "cat": True},
    {"name": "soo_minimize", "minimize": True},
    {"name": "soo_history", "history": True},
]


def _collect():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for sc in SCENARIOS:
            sub = Path(tmp) / sc["name"]
            sub.mkdir()
            out[sc["name"]] = _capture(sc, str(sub))
    return out


def test_strategy_prompt_snapshot():
    current = _collect()
    if os.environ.get("UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2))
        print(f"WROTE golden ({len(current)} scenarios) -> {GOLDEN}")
        return
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(current), "scenario set changed"
    mism = [name for name in golden if golden[name] != current[name]]
    assert not mism, f"prompt drift in: {mism}"
    print(f"PASS strategy prompt snapshot ({len(current)} scenarios byte-identical)")


INSPECTION_GOLDEN = Path(__file__).parent / "golden" / "bo_inspection_prompts.json"

# (is_moo, model_cfg, acq_cfg, sensitivity_data) per scenario.
INSPECTION_SCENARIOS = {
    "soo_no_sobol": (False, {"surrogate": "single_task", "kernel": "matern_2.5"},
                     {"type": "log_ei"}, {}),
    "soo_with_sobol": (False, {"surrogate": "dkl", "kernel": "matern_1.5",
                               "noise": "min_noise_high", "input_transform": "warp"},
                       {"type": "ucb", "params": {"beta": 2.0}},
                       {"temp": 0.71, "conc": 0.12}),
    "moo": (True, {"surrogate": "single_task"}, {"type": "pareto"}, {}),
}


def _build_agent():
    with tempfile.TemporaryDirectory() as tmp:
        return BOAgent(api_key="test", model_name="claude-opus-4-6", output_dir=tmp)


def test_skill_context_empty():
    """The seam returns "" for every optimization section until PR 2 wires
    skills in — the invariant that keeps the builders byte-identical."""
    agent = _build_agent()
    for section in ("overview", "setup", "surrogate", "acquisition",
                    "diagnostics", "interpretation", "implementation"):
        assert agent._skill_context(section) == "", section


def test_inspection_prompt_snapshot():
    agent = _build_agent()
    current = {
        name: agent._build_inspection_prompt(
            is_moo=is_moo, model_cfg=mc, acq_cfg=ac, sensitivity_data=sd)
        for name, (is_moo, mc, ac, sd) in INSPECTION_SCENARIOS.items()
    }
    if os.environ.get("UPDATE_GOLDEN") == "1" or not INSPECTION_GOLDEN.exists():
        INSPECTION_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        INSPECTION_GOLDEN.write_text(json.dumps(current, indent=2))
        print(f"WROTE inspection golden -> {INSPECTION_GOLDEN}")
        return
    golden = json.loads(INSPECTION_GOLDEN.read_text())
    mism = [n for n in golden if golden.get(n) != current.get(n)]
    assert set(golden) == set(current) and not mism, f"inspection prompt drift: {mism}"
    print(f"PASS inspection prompt snapshot ({len(current)} scenarios byte-identical)")


if __name__ == "__main__":
    test_strategy_prompt_snapshot()
    test_skill_context_empty()
    test_inspection_prompt_snapshot()
    print("all snapshot tests passed")
