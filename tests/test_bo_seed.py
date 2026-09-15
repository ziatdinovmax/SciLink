"""run_optimization(seed=...): numeric reproducibility of a BO step.

GP fitting (random restarts) and acquisition optimisation (random raw
samples) are stochastic; with the same seed applied the way
BOAgent.run_optimization_loop applies it (torch.manual_seed + np seed
before the numeric work), two runs on identical data recommend the same
batch. The tool forwards the seed and echoes it in the response.
"""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")
os.environ.setdefault("MPLBACKEND", "Agg")


def _one_step(seed):
    import torch
    from scilink.agents.planning_agents.bo_tools import get_optimizer
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(0)                    # fixed data
    X = rng.uniform(0, 1, size=(12, 2))
    y = -((X[:, 0] - 0.6) ** 2 + (X[:, 1] - 0.3) ** 2) + 0.01 * rng.normal(size=12)
    opt = get_optimizer(is_moo=False)
    opt.fit(X, y, bounds=[(0, 1), (0, 1)],
            model_config={"kernel": "matern_2.5", "noise": "min_noise_low",
                          "surrogate": "single_task"},
            feature_names=["a", "b"])
    return np.asarray(opt.recommend(n_candidates=3, strategy="log_ei"))


def test_same_seed_same_batch_numeric():
    a = _one_step(7)
    b = _one_step(7)
    assert np.allclose(a, b), (a, b)


def test_tool_forwards_seed_and_reports_it():
    from scilink.agents.planning_agents.planning_orchestrator import (
        PlanningOrchestratorAgent, AutonomyLevel,
    )
    L6 = dict(T=[30.0, 30.0, 90.0, 90.0, 50.0, 78.0],
              t=[10.0, 50.0, 10.0, 50.0, 35.0, 18.0],
              Y=[8.56, 3.88, 22.47, 10.81, 48.83, 61.1])
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"; data_dir.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            orch = PlanningOrchestratorAgent(
                base_dir=str(Path(tmp) / "s"), api_key="sk-dummy",
                autonomy_level=AutonomyLevel.AUTONOMOUS, data_dir=str(data_dir))
        orch.scalarizer.scalarize = lambda **kw: {
            "status": "success", "metrics": dict(L6), "source_script": None,
            "column_roles": {"inputs": ["T", "t"], "targets": ["Y"]},
            "passthrough": True, "error": None}
        captured = {}

        def fake_loop(**kw):
            captured.update(kw)
            return {"status": "success", "next_parameters": {"T": 60.0, "t": 25.0},
                    "strategy": {}, "seed": kw.get("seed")}
        orch.bo.run_optimization_loop = fake_loop
        csv = Path(tmp) / "seed.csv"
        csv.write_text("T,t,Y\n" + "\n".join(
            f"{a},{b},{c}" for a, b, c in zip(L6["T"], L6["t"], L6["Y"])) + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            orch.tools.execute_tool("analyze_file", file_path=str(csv),
                                    extraction_goal="x", inputs=["T", "t"], targets=["Y"])
            out = json.loads(orch.tools.execute_tool("run_optimization", seed=123))
            out2 = json.loads(orch.tools.execute_tool("run_optimization"))
        assert captured.get("seed") is None and out2.get("seed") is None
        assert out.get("seed") == 123
