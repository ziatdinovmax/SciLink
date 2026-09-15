"""structure_class is derived from scale when unset (issue #474).

A classical-MD run must build/validate as a condensed system, not a crystal.
We stub StructurePipeline to capture the structure_class the pipeline passes it,
returning a non-success result so the pipeline stops right after structure gen.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scilink.agents.sim_agents.simulation_pipeline as sp  # noqa: E402
import scilink.agents.sim_agents.structure_pipeline as spmod  # noqa: E402


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    class FakeStructurePipeline:
        def __init__(self, **kw):
            self.api_key = kw.get("api_key")
            self.base_url = kw.get("base_url")

        def generate_and_validate(self, request, structure_class=None):
            seen["structure_class"] = structure_class
            return {"status": "stopped"}   # non-success -> pipeline returns early

    monkeypatch.setattr(spmod, "StructurePipeline", FakeStructurePipeline)
    return seen


def _run(captured, tmp_path, **kw):
    sp._run_workflow_once(
        user_request="an aqueous NaCl electrolyte box",
        output_dir=str(tmp_path), api_key="k", base_url="http://localhost:0",
        **kw)
    return captured["structure_class"]


def test_md_derives_condensed(captured, tmp_path):
    assert _run(captured, tmp_path, scale="molecular_dynamics") == "condensed"


def test_periodic_dft_derives_crystal(captured, tmp_path):
    assert _run(captured, tmp_path, scale="periodic_dft") == "crystal"


def test_molecular_qc_derives_molecular(captured, tmp_path):
    assert _run(captured, tmp_path, scale="molecular_qc") == "molecular"


def test_explicit_class_wins(captured, tmp_path):
    # A biomolecular MD system can still opt out of the condensed default.
    got = _run(captured, tmp_path, scale="molecular_dynamics",
               structure_class="biomolecular")
    assert got == "biomolecular"


def test_explicit_crystal_wins_for_crystalline_md(captured, tmp_path):
    # Crystalline-solid / slab MD (which the MD scale absorbs since #432) is
    # built as a crystal when the caller says so, not the condensed default.
    got = _run(captured, tmp_path, scale="molecular_dynamics",
               structure_class="crystal")
    assert got == "crystal"


def test_mlip_shim_derives_condensed(captured, tmp_path):
    # The deprecated machine_learning_potentials scale is treated as an MD task,
    # so it deliberately shares MD's condensed default (not a crystal fall-through).
    assert _run(captured, tmp_path,
                scale="machine_learning_potentials") == "condensed"


def test_unknown_scale_defaults_crystal(captured, tmp_path):
    assert _run(captured, tmp_path, scale="something_new") == "crystal"


def test_run_complete_workflow_threads_class(captured, tmp_path):
    # The public entry point passes structure_class through to the derivation.
    sp.run_complete_workflow(
        "an aqueous NaCl electrolyte box", scale="molecular_dynamics",
        output_dir=str(tmp_path), api_key="k", base_url="http://localhost:0")
    assert captured["structure_class"] == "condensed"
