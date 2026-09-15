"""Offline tests for the scalarizer fidelity-role wiring (multi-fidelity through
the planning orchestrator). Covers the deterministic regression surface: the
optional `column_roles.fidelity` parse/validation and that normal (non-fidelity)
scalarizer output is unaffected.
"""
import types

from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
from scilink.agents.planning_agents.instruct import SCALARIZER_PROMPT


def _tools():
    # Bypass the heavy tool registration; the capture methods only use self.orch.
    ot = OrchestratorTools.__new__(OrchestratorTools)
    ot.orch = types.SimpleNamespace(fidelity_spec="STALE", expected_input_types=None)
    return ot


def test_no_fidelity_resets_to_none():
    ot = _tools()
    ot._capture_fidelity_spec({"inputs": ["a", "b"], "targets": ["y"]}, ["a", "b"])
    assert ot.orch.fidelity_spec is None


def test_valid_fidelity_captured():
    ot = _tools()
    cr = {"inputs": ["x", "fid"], "targets": ["y"],
          "fidelity": {"column": "fid", "target_fidelity": 1.0, "costs": {"0": 1, "1": 10}}}
    ot._capture_fidelity_spec(cr, ["x", "fid"])
    assert ot.orch.fidelity_spec == {"column": "fid", "target_fidelity": 1.0,
                                     "costs": {"0": 1, "1": 10}}


def test_fidelity_minimal_no_costs_no_target():
    ot = _tools()
    ot._capture_fidelity_spec({"fidelity": {"column": "a"}}, ["a"])
    assert ot.orch.fidelity_spec == {"column": "a"}  # optional keys omitted


def test_fidelity_column_not_in_inputs_is_ignored():
    ot = _tools()
    ot._capture_fidelity_spec({"fidelity": {"column": "NOT_AN_INPUT"}}, ["x"])
    assert ot.orch.fidelity_spec is None


def test_malformed_fidelity_is_ignored():
    ot = _tools()
    ot._capture_fidelity_spec({"fidelity": "garbage"}, ["x"])  # not a dict
    assert ot.orch.fidelity_spec is None


def test_capture_input_types_still_works_and_folds_fidelity():
    # Regression: input_types capture unchanged; fidelity captured in the same call.
    ot = _tools()
    cr = {"input_types": {"a": "continuous", "b": "categorical"},
          "fidelity": {"column": "a"}}
    ot._capture_input_types(cr, ["a", "b"])
    assert ot.orch.expected_input_types == {"a": "continuous", "b": "categorical"}
    assert ot.orch.fidelity_spec == {"column": "a"}


def test_backward_compat_old_scalarizer_output():
    # Old output (no fidelity, no input_types) -> no crash, fidelity None.
    ot = _tools()
    ot._capture_input_types({"inputs": ["a"], "targets": ["y"]}, ["a"])
    assert ot.orch.fidelity_spec is None
    assert ot.orch.expected_input_types is None  # unchanged


def test_prompt_declares_optional_fidelity():
    assert "fidelity" in SCALARIZER_PROMPT
    assert "OPTIONAL" in SCALARIZER_PROMPT
    assert "MUST also appear in `inputs`" in SCALARIZER_PROMPT  # the validity rule


def test_fidelity_config_index_mapping():
    # Mirrors run_optimization's fidelity_config construction.
    inputs = ["temp", "conc", "fid"]
    spec = {"column": "fid", "target_fidelity": 1.0, "costs": {"0": 1, "1": 10}}
    fc = {"fidelity_col": inputs.index(spec["column"])}
    if spec.get("target_fidelity") is not None:
        fc["target_fidelity"] = spec["target_fidelity"]
    if spec.get("costs"):
        fc["fidelity_costs"] = spec["costs"]
    assert fc == {"fidelity_col": 2, "target_fidelity": 1.0, "fidelity_costs": {"0": 1, "1": 10}}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all passed")
