"""
Hermetic tests for the SimulationRouter.

No live LLM calls — the model is mocked. The routing-decision contract
is exercised across:
  - candidate computation (agent_supports ∩ user_available)
  - decision validation (rejects LLM picks outside the candidate set)
  - prompt construction (scales appear, candidates appear, no leakage)
  - parsing tolerance (json with / without code fence / preamble)
  - failure modes (no candidates, LLM error, invalid pick)
"""

import json
from unittest.mock import MagicMock

import pytest

from scilink.agents.sim_agents.simulation_router import (
    SimulationRouter,
    discover_scale_agents,
    DEFAULT_SCALE_DESCRIPTIONS,
)
from scilink.utils.available_software import AvailableSoftware


def _model_returning(payload: str):
    """Build a MagicMock model whose generate_content returns ``payload``."""
    model = MagicMock()
    resp = MagicMock()
    resp.text = payload
    model.generate_content.return_value = resp
    return model


def _avail_with(*entries):
    """Build an AvailableSoftware fixture from (domain, engine) pairs.

    Each entry can be a (domain, engine) tuple — marked available — or
    a (domain, engine, available_bool) triple to set explicit state.
    """
    cfg = AvailableSoftware()
    for entry in entries:
        if len(entry) == 2:
            domain, engine = entry
            cfg.set(domain, engine, True, source="test-fixture")
        else:
            domain, engine, avail = entry
            cfg.set(domain, engine, avail, source="test-fixture")
    return cfg


# ─── discover_scale_agents ─────────────────────────────────────────


def test_discover_scale_agents_includes_periodic_dft():
    """PeriodicDFTAgent should always be discovered."""
    scales = discover_scale_agents()
    assert "periodic_dft" in scales
    assert "vasp" in scales["periodic_dft"]["supported"]


def test_discover_scale_agents_includes_molecular_qc():
    """MolecularQCAgent should be discovered with the nwchem bundle.

    Canary that the agent registered in discover_scale_agents() and that
    supported_software() finds the skills/molecular_qc/nwchem bundle.
    """
    scales = discover_scale_agents()
    assert "molecular_qc" in scales
    assert "nwchem" in scales["molecular_qc"]["supported"]


# ─── candidate_engines: intersection ───────────────────────────────


def test_candidate_engines_returns_intersection():
    avail = _avail_with(
        ("periodic_dft", "vasp"),
        ("periodic_dft", "qe", False),     # user has it... actually no
        ("molecular_dynamics", "lammps"),
    )
    router = SimulationRouter(model=MagicMock(), available_software=avail)
    cands = router.candidate_engines()

    # VASP should be there (agent supports it AND user has it)
    assert "periodic_dft" in cands
    assert "vasp" in cands["periodic_dft"]

    # QE NOT marked available -> shouldn't appear in periodic_dft candidates
    # (and isn't in agent_supports anyway since no skill bundle)
    assert "qe" not in cands.get("periodic_dft", [])


def test_candidate_engines_omits_unavailable_scales():
    """If user has no engines available for a scale, that scale is omitted."""
    avail = _avail_with(("periodic_dft", "vasp"))  # nothing else
    router = SimulationRouter(model=MagicMock(), available_software=avail)
    cands = router.candidate_engines()

    assert list(cands.keys()) == ["periodic_dft"]


# ─── route(): success path ─────────────────────────────────────────


def test_route_returns_valid_decision():
    avail = _avail_with(("periodic_dft", "vasp"))
    decision_json = json.dumps({
        "scale": "periodic_dft",
        "engine": "vasp",
        "reasoning": "Cu slab calls for periodic DFT; VASP is the only "
                     "available engine in that scale.",
        "alternatives": [],
    })
    router = SimulationRouter(
        model=_model_returning(decision_json),
        available_software=avail,
    )

    out = router.route(
        user_goal="Relax a Cu(111) slab and compute work function",
        system_description="metallic slab, 16 atoms",
    )

    assert out["scale"] == "periodic_dft"
    assert out["engine"] == "vasp"
    assert "reasoning" in out
    assert "candidates_considered" in out
    assert out["candidates_considered"]["periodic_dft"] == ["vasp"]


def test_route_tolerates_markdown_code_fence_response():
    """LLM responses often arrive wrapped in ```json ... ``` fences."""
    avail = _avail_with(("periodic_dft", "vasp"))
    decision_json = (
        "Here is my decision:\n\n"
        "```json\n"
        '{"scale": "periodic_dft", "engine": "vasp", '
        '"reasoning": "test", "alternatives": []}\n'
        "```\n"
    )
    router = SimulationRouter(
        model=_model_returning(decision_json),
        available_software=avail,
    )
    out = router.route("test goal")
    assert out["scale"] == "periodic_dft"
    assert out["engine"] == "vasp"


# ─── route(): failure modes ────────────────────────────────────────


def test_route_no_candidates_returns_error():
    """If user has nothing installed, route should error rather than hallucinate."""
    avail = AvailableSoftware()    # empty
    router = SimulationRouter(model=MagicMock(), available_software=avail)
    out = router.route("anything")
    assert out["scale"] is None
    assert out["engine"] is None
    assert "error" in out
    assert "no engines" in out["error"].lower()


def test_route_rejects_llm_picking_invalid_engine():
    """LLM picking an engine NOT in the candidate set is treated as a failure."""
    avail = _avail_with(("periodic_dft", "vasp"))
    bad_decision = json.dumps({
        "scale": "periodic_dft",
        "engine": "qe",                # not a candidate
        "reasoning": "...", "alternatives": [],
    })
    router = SimulationRouter(
        model=_model_returning(bad_decision),
        available_software=avail,
    )
    out = router.route("test goal")
    assert out["scale"] is None
    assert "error" in out
    assert "not in the candidate set" in out["error"]
    assert out["raw_decision"]["engine"] == "qe"


def test_route_rejects_llm_picking_invalid_scale():
    avail = _avail_with(("periodic_dft", "vasp"))
    bad_decision = json.dumps({
        "scale": "molecular_dft",       # not in candidates
        "engine": "vasp",
        "reasoning": "...", "alternatives": [],
    })
    router = SimulationRouter(
        model=_model_returning(bad_decision),
        available_software=avail,
    )
    out = router.route("test goal")
    assert out["scale"] is None
    assert "error" in out


def test_route_handles_model_exception():
    avail = _avail_with(("periodic_dft", "vasp"))
    model = MagicMock()
    model.generate_content.side_effect = RuntimeError("network down")
    router = SimulationRouter(model=model, available_software=avail)
    out = router.route("test goal")
    assert out["scale"] is None
    assert "network down" in out["error"]


# ─── prompt construction ──────────────────────────────────────────


def test_prompt_includes_user_goal_and_candidates():
    avail = _avail_with(("periodic_dft", "vasp"))
    router = SimulationRouter(model=MagicMock(), available_software=avail)
    prompt = router._build_prompt(
        user_goal="Relax a Cu(111) slab",
        system_description="metallic surface",
        candidates={"periodic_dft": ["vasp"]},
    )

    assert "Relax a Cu(111) slab" in prompt
    assert "metallic surface" in prompt
    assert "periodic_dft" in prompt
    assert "vasp" in prompt
    # Scale-description guidance for the present scale is included
    assert "Planewave" in prompt or "planewave" in prompt


def test_prompt_omits_scales_without_candidates():
    """A scale with no available engines shouldn't appear in the prompt."""
    avail = _avail_with(("periodic_dft", "vasp"))
    router = SimulationRouter(model=MagicMock(), available_software=avail)
    prompt = router._build_prompt(
        user_goal="x", system_description="y",
        candidates={"periodic_dft": ["vasp"]},
    )
    assert "molecular_dft" not in prompt
    assert "molecular_dynamics" not in prompt
    assert "machine_learning_potentials" not in prompt
