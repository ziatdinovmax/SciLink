"""The structure pipeline must leave the process's logging to its host.

Building one used to call ``logging.basicConfig(force=True)`` with a stdout
handler: every root handler the host had installed was removed and every
record of every agent was echoed to stdout from then on, which a host that
captures stdout shows a second time beside its own log handler.
"""
import logging
from unittest import mock

import pytest

from scilink.agents.sim_agents import structure_pipeline as sp


@pytest.fixture
def pipeline(tmp_path):
    with mock.patch.object(sp, "StructureGenerator"), \
         mock.patch.object(sp, "StructureValidatorAgent"):
        yield sp.StructurePipeline(api_key="test", mp_api_key="test",
                                   output_dir=str(tmp_path))


@pytest.fixture
def root_state():
    root = logging.getLogger()
    host = logging.StreamHandler()
    root.addHandler(host)
    level = root.level
    try:
        yield root, host
    finally:
        root.removeHandler(host)
        root.setLevel(level)


def test_construction_leaves_root_handlers_alone(root_state, tmp_path):
    root, host = root_state
    before = list(root.handlers)
    level = root.level
    with mock.patch.object(sp, "StructureGenerator"), \
         mock.patch.object(sp, "StructureValidatorAgent"):
        sp.StructurePipeline(api_key="test", mp_api_key="test",
                             output_dir=str(tmp_path))
    assert root.handlers == before
    assert host in root.handlers
    assert root.level == level


def test_build_structure_captures_its_run_and_restores_root(
        root_state, pipeline, tmp_path):
    root, _ = root_state
    root.setLevel(logging.WARNING)
    before = list(root.handlers)

    def fake_generate(*_a, **_k):
        logging.getLogger("scilink.agents.sim_agents.structure_agent").info(
            "inside the run")
        return {"status": "error", "message": "stub"}

    with mock.patch.object(pipeline, "generate_and_validate",
                           side_effect=fake_generate):
        pipeline.build_structure("bulk Si", structure_class="crystal")
    logging.getLogger("scilink.test").info("after the run")

    assert root.handlers == before
    assert root.level == logging.WARNING
    log = (tmp_path / "workflow_log.txt").read_text()
    assert "inside the run" in log
    assert "after the run" not in log


def test_build_structure_restores_root_when_the_run_raises(root_state, pipeline):
    root, _ = root_state
    before = list(root.handlers)
    with mock.patch.object(pipeline, "generate_and_validate",
                           side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            pipeline.build_structure("bulk Si", structure_class="crystal")
    assert root.handlers == before
