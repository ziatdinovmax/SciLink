"""ForceFieldAgent.parameterize() routing — engine-neutral entry to a
ParameterizedSystem.

Offline (no [ff] deps): routing by input is correct, and the OpenFF path's
dependency guard fires. The actual OpenFF parameterization (build_interchange ->
Interchange) is verified live in the [ff] env. The agent is instantiated via
__new__ to skip its credentialed constructor.
"""

from __future__ import annotations

import logging

import pytest

from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent


def _agent(tmp_path):
    a = ForceFieldAgent.__new__(ForceFieldAgent)
    a.working_dir = str(tmp_path)
    a.logger = logging.getLogger("test_ff_parameterize")
    return a


def test_parameterize_requires_an_input(tmp_path):
    with pytest.raises(ValueError, match="components"):
        _agent(tmp_path).parameterize()


def test_parameterize_amber_path_pending(tmp_path):
    with pytest.raises(NotImplementedError, match="AMBER"):
        _agent(tmp_path).parameterize(pdb_file="protein.pdb")


def test_parameterize_openff_path_guards_missing_ff_deps(tmp_path):
    # components + coordinates routes to OpenFF; without [ff] the build_interchange
    # tool raises an actionable scilink[ff] error (never a silent failure).
    with pytest.raises(ImportError, match=r"scilink\[ff\]"):
        _agent(tmp_path).parameterize(
            components=[{"name": "water", "smiles": "O", "count": 1}],
            coordinates_file=str(tmp_path / "structure.extxyz"),
        )
