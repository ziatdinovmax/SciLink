"""Offline tests for OptimizationAgent skill machinery (issue #196, PR 2).

Standard single-/multi-objective BO is baseline competency (not skill-gated).
Skills are *specializations beyond the baseline*, activated only when explicitly
requested (or, later, agent-selected). These tests verify:
  - no skill is auto-loaded (MOO and SOO both run on the baseline);
  - an explicitly-requested skill is loaded and its guidance is spliced into the
    stage prompt via the `_skill_context` seam;
  - the BOAgent back-compat alias.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd

from scilink.agents.planning_agents import OptimizationAgent, BOAgent
from scilink.agents.planning_agents.bo_agent import OptimizationAgent as _OA


class _StopAfterCapture(Exception):
    pass


class _CaptureModel:
    def __init__(self):
        self.captured = None

    def generate_content(self, parts, generation_config=None):
        self.captured = "\n".join(p for p in parts if isinstance(p, str))
        raise _StopAfterCapture()


def _run(n_targets=1, skill=None, tmp=None):
    data = Path(tmp) / "d.csv"
    cols = {"x1": [1.0, 2, 3, 4], "x2": [0.1, 0.2, 0.3, 0.4], "y1": [5.0, 6, 7, 8]}
    if n_targets == 2:
        cols["y2"] = [1.0, 2, 1.5, 2.5]
    pd.DataFrame(cols).to_csv(data, index=False)
    agent = BOAgent(api_key="test", model_name="claude-opus-4-6", output_dir=tmp)
    agent.model = _CaptureModel()
    targets = ["y1", "y2"] if n_targets == 2 else ["y1"]
    kw = {"skill": skill} if skill is not None else {}
    try:
        agent.run_optimization_loop(
            data_path=str(data), objective_text="opt",
            input_cols=["x1", "x2"], input_bounds=[[1, 4], [0.1, 0.4]],
            target_cols=targets, output_dir=tmp, **kw,
        )
    except _StopAfterCapture:
        pass
    return agent.model.captured, agent.state.get("skills_loaded", [])


def _write_temp_skill(tmp, name="mf_demo"):
    p = Path(tmp) / f"{name}.md"
    p.write_text(
        "---\ndescription: demo specialization\ncategory: modifier\n---\n"
        "## Acquisition\nUSE_COST_AWARE_FIDELITY_REASONING_HERE\n"
    )
    return str(p)


def test_moo_no_skill_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        prompt, loaded = _run(n_targets=2, tmp=tmp)
        assert loaded == [], "MOO must not auto-load any skill (baseline competency)"


def test_soo_no_skill_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        prompt, loaded = _run(n_targets=1, tmp=tmp)
        assert loaded == []


def test_explicit_skill_is_loaded_and_spliced():
    with tempfile.TemporaryDirectory() as tmp:
        skill_path = _write_temp_skill(tmp)
        prompt, loaded = _run(n_targets=1, skill=skill_path, tmp=tmp)
        assert [s["name"] for s in loaded] == ["mf_demo"]
        # the seam splices the skill's acquisition guidance into the prompt
        assert "USE_COST_AWARE_FIDELITY_REASONING_HERE" in prompt


def test_backcompat_alias():
    assert issubclass(BOAgent, OptimizationAgent)
    assert _OA is OptimizationAgent


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all passed")
