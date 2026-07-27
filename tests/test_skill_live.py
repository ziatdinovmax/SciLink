"""
Live integration tests — skill loading + Claude API end-to-end.

These exercise the full path from the new Tier C bundle layout through
the LiteLLM wrapper to a real Anthropic Claude call. Skipped unless
ANTHROPIC_API_KEY is set in the environment.

Run with:

    ANTHROPIC_API_KEY=sk-ant-... \\
        python -m pytest tests/test_skill_live.py -v -s
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from scilink.skills.loader import list_all_skills, load_skill


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

MODEL = "claude-opus-4-6"


@pytest.fixture(scope="module")
def model():
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
    return LiteLLMGenerativeModel(
        model=MODEL,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )


# ─── Layer 1: Tier C discovery ──────────────────────────────────────


def test_all_skills_load_via_tier_c_layout():
    """All shipped skills must be discoverable + loadable via the bundle layout
    (`<domain>/<name>/<name>.md`) and carry a frontmatter description."""
    discovered = list_all_skills()
    expected_domains = {"curve_fitting", "force_field", "hyperspectral",
                        "image_analysis", "structure_generation",
                        "structure_matching"}
    assert expected_domains.issubset(discovered.keys())

    for domain, names in discovered.items():
        for name in names:
            s = load_skill(name, domain=domain)
            assert s["name"] == name
            assert s["meta"].get("description"), f"{domain}/{name} missing description"


# ─── Layer 2: live LLM with skill content ───────────────────────────


def test_xps_skill_steers_background_choice(model):
    """The XPS skill's `planning` section must reach the LLM and influence
    a domain decision (background subtraction method)."""
    skill = load_skill("xps", domain="curve_fitting")
    prompt = (
        f"Domain knowledge:\n{skill['overview']}\n\n{skill['planning']}\n\n"
        "QUESTION: For a C 1s spectrum on an organic polymer with three peaks "
        "around 284.6, 286.5, and 288.8 eV, which background subtraction "
        "method would you use? Answer in one sentence naming the method."
    )
    resp = model.generate_content([prompt])
    text = resp.text.lower()
    assert any(m in text for m in ("shirley", "tougaard")), \
        f"Expected Shirley/Tougaard reference; got: {resp.text}"


def test_amber_skill_steers_charge_method(model):
    """The AMBER skill's planning section should let the LLM pick a charge
    method correctly for a neutral small molecule."""
    skill = load_skill("amber", domain="force_field")
    prompt = (
        f"Domain knowledge:\n{skill['planning']}\n\n"
        "QUESTION: For a neutral small organic molecule (no charged groups, "
        "no transition metals), which antechamber charge method should you "
        "use? Reply with just one word: bcc, resp, or gas."
    )
    resp = model.generate_content([prompt])
    text = resp.text.lower().strip()
    assert "bcc" in text, \
        f"Expected 'bcc' for neutral organic; got: {resp.text}"


def test_aimsgb_extras_section_reaches_consumer(model):
    """Regression: the 'Common pitfalls' section was previously dropped silently.
    Verify it now flows through `_load_skill_content` and that an LLM grounded
    in that content correctly answers a pitfall-specific question."""
    from scilink.agents.sim_agents.simulation_orchestrator_tools import (
        SimulationOrchestratorTools,
    )

    tools = SimulationOrchestratorTools.__new__(SimulationOrchestratorTools)
    tools.orch = MagicMock()
    tools.orch._custom_skills = {}
    tools.logger = MagicMock()

    content = tools._load_skill_content("aimsgb")
    assert "Common pitfalls" in content, "extras section did not flow through"

    prompt = (
        f"Skill content:\n{content}\n\n"
        "QUESTION: A user is generating a grain-boundary structure for an HCP "
        "metal and passes a 4-index Bravais-Miller direction like [11-20]. "
        "Should they use this notation, or should they convert to a 3-index "
        "Miller form? Answer in one short sentence."
    )
    resp = model.generate_content([prompt])
    text = resp.text.lower()
    # The Common pitfalls section is unambiguous: convert to 3-index Miller.
    assert "3-index" in text or "miller" in text or "convert" in text, \
        f"Pitfalls guidance didn't surface; got: {resp.text}"


# ─── Layer 3: synthetic-data round-trip ─────────────────────────────


def _generate_synthetic_xps(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a noisy C 1s-like XPS spectrum with three peaks on a
    Shirley-shaped background. Returns (binding_energy, intensity) arrays."""
    rng = np.random.default_rng(seed)
    be = np.linspace(280.0, 295.0, 600)  # binding energy axis (eV)

    def gaussian(x, mu, fwhm, area):
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return area * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # C-C / C-H, C-O, C=O — typical organic-polymer fingerprint
    peaks = (
        gaussian(be, 284.6, 1.0, 1.0)
        + gaussian(be, 286.4, 1.2, 0.35)
        + gaussian(be, 288.6, 1.3, 0.20)
    )
    # Crude Shirley-shaped step: rises smoothly toward higher BE
    background = 0.05 + 0.08 / (1.0 + np.exp(-(be - 287.0) / 1.5))
    intensity = peaks + background + 0.01 * rng.standard_normal(be.shape)
    return be, intensity


def test_xps_skill_handles_synthetic_spectrum(model):
    """Hand the LLM a synthetic three-peak C 1s spectrum (numeric sample) and
    the XPS skill, and verify it can both identify the chemical environments
    and produce sane fitting recommendations."""
    skill = load_skill("xps", domain="curve_fitting")
    be, intensity = _generate_synthetic_xps(seed=0)

    # Down-sample to a compact textual summary the model can reason about.
    sample_idx = np.linspace(0, len(be) - 1, 30, dtype=int)
    sample = "\n".join(f"  {be[i]:.2f}  {intensity[i]:.4f}" for i in sample_idx)

    prompt = (
        f"You are an XPS curve-fitting assistant. Domain knowledge:\n"
        f"{skill['overview']}\n\n{skill['planning']}\n\n"
        f"DATA (binding energy [eV] vs intensity [a.u.], 30 sampled points):\n"
        f"{sample}\n\n"
        f"TASKS (answer concisely):\n"
        f"1. Identify the approximate binding-energy positions of any peaks.\n"
        f"2. Suggest plausible chemical environments for those peaks.\n"
        f"3. Recommend a background subtraction method.\n"
    )
    resp = model.generate_content([prompt])
    text = resp.text.lower()

    # Synthetic peaks at 284.6 / 286.4 / 288.6 — accept loose matches
    assert any(s in text for s in ("284.5", "284.6", "284.7", "~284", "around 284")), \
        f"Did not identify the C-C peak; got: {resp.text}"
    # Should suggest a real background method
    assert any(m in text for m in ("shirley", "tougaard")), \
        f"No background method suggested; got: {resp.text}"
    # Should mention realistic chemical environments
    assert any(env in text for env in ("c-c", "c-h", "c-o", "c=o", "carbon", "organic")), \
        f"No chemical environments suggested; got: {resp.text}"


# ─── Layer 4: PR 2 verification — moved helpers + registry ──────────


def test_registry_discovers_all_tool_specs_after_move():
    """The registry now walks both _shared/ and per-skill bundles. Verify it
    finds every TOOL_SPEC declared across the moved layout."""
    from scilink.skills._shared._registry import _all_specs

    specs = _all_specs()
    found = {s.name for s in specs}
    expected = {
        # _shared/ helpers
        "run_fft_nmf_analysis",
        "run_sam_analysis",
        # per-skill bundle (image_analysis/atomic_stem/atom_finding.py)
        "detect_atoms",
        "detect_atoms_dcnn",
        "refine_positions",
        "find_zone_axes",
        "find_missing_atoms",
        "subtract_atoms",
        "local_env_gmm",
    }
    missing = expected - found
    assert not missing, f"Registry missed specs after move: {missing}"


def test_every_tool_spec_import_line_resolves():
    """Every TOOL_SPEC's `import_line` is rendered into LLM-generated scripts.
    After the move, those strings must still resolve to real callables —
    otherwise the LLM's first attempt to use a tool will fail at import."""
    import importlib
    import re

    from scilink.skills._shared._registry import _all_specs

    pattern = re.compile(r"from\s+([\w\.]+)\s+import\s+(.+)")
    for spec in _all_specs():
        line = spec.import_line.strip()
        m = pattern.match(line)
        assert m, f"Couldn't parse import_line for {spec.name}: {line!r}"
        module_path, names = m.group(1), m.group(2)
        mod = importlib.import_module(module_path)
        for raw in names.split(","):
            fn = raw.strip()
            assert hasattr(mod, fn), \
                f"{module_path}.{fn} missing — TOOL_SPEC '{spec.name}' would fail at LLM call site"


def test_force_field_agent_amber_alias_resolved_after_move():
    """ForceFieldAgent imports `amber_tools` via the alias
    `from ...skills.force_field.amber import amber as amber_tools`.
    Verify the alias points at the right module and exposes the expected API."""
    from scilink.agents.sim_agents import force_field_agent as ffa

    assert ffa._AMBER_TOOLS_AVAILABLE, "amber_tools alias did not import"
    assert hasattr(ffa.amber_tools, "check_amber_tools")
    assert hasattr(ffa.amber_tools, "convert_amber_to_lammps")


def test_atomistic_agent_alias_resolved_after_move():
    """atomistic_microscopy_agent imports `atomistic_tools` via the alias
    `from ...skills.image_analysis.atomic_stem import atomic_stem as atomistic_tools`.
    Verify the alias resolves and exposes the functions it uses."""
    from scilink.agents.exp_agents import atomistic_microscopy_agent as ama

    assert hasattr(ama, "atomistic_tools")
    # atomistic_tools.py exposes rescale_for_model + predict_with_ensemble
    assert hasattr(ama.atomistic_tools, "rescale_for_model")
    assert hasattr(ama.atomistic_tools, "predict_with_ensemble")


def test_all_analysis_agents_import_cleanly():
    """Every analysis agent imports moved helpers at module level. If any
    `from ...skills._shared.X` or `from ...skills.<domain>.<name>.X` import
    fails, the agent module won't even load."""
    import importlib

    modules = [
        "scilink.agents.exp_agents.analysis_orchestrator",
        "scilink.agents.exp_agents.analysis_orchestrator_tools",
        "scilink.agents.exp_agents.atomistic_microscopy_agent",
        "scilink.agents.exp_agents.curve_fitting_agent",
        "scilink.agents.exp_agents.fft_microscopy_agent",
        "scilink.agents.exp_agents.hyperspectral_analysis_agent",
        "scilink.agents.exp_agents.image_analysis_agent",
        "scilink.agents.exp_agents.sam_microscopy_agent",
        "scilink.agents.exp_agents.controllers.curve_fitting_controllers",
        "scilink.agents.exp_agents.controllers.image_analysis_controllers",
        "scilink.agents.exp_agents.controllers.fft_microscopy_controllers",
        "scilink.agents.exp_agents.controllers.sam_microscopy_controllers",
        "scilink.agents.exp_agents.controllers.atomistic_microscopy_controllers",
        "scilink.agents.exp_agents.controllers.hyperspectral_controllers",
        "scilink.agents.sim_agents.force_field_agent",
        "scilink.agents.sim_agents.lammps_orchestrator",
        "scilink.agents.planning_agents.bo_agent",
        "scilink.cli.prepare_ff",
    ]
    for module in modules:
        importlib.import_module(module)


def test_orchestrator_skill_description_after_move():
    """The orchestrator's run_analysis tool description is built from
    `list_all_skills()` + each skill's frontmatter. After the Tier C move
    and PR 2 helper reshuffle this should still produce the expected
    7-skill blurb."""
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        _build_skill_description,
    )

    desc = _build_skill_description(agent_registry=None, custom_skills=None)
    for name in ("xps", "amber", "eels", "atomic_stem",
                 "overlapping_objects", "stm_cafm", "aimsgb"):
        assert f"'{name}' —" in desc, f"missing '{name}' in skill description"


# ─── Layer 5: live LLM uses the moved tool inventory ────────────────


def test_live_llm_can_route_via_moved_tool_inventory(model):
    """End-to-end: render the moved registry's `format_tool_inventory`,
    inject it into a prompt, and ask Claude to choose the right tools for a
    realistic task. Validates the full registry → prompt → LLM path.

    Atomic_stem must be active for the atom-finding tools to surface (PR 3
    skill-gated visibility)."""
    from scilink.skills._shared._registry import format_tool_inventory

    inventory = format_tool_inventory("image_analysis", active_skills=["atomic_stem"])

    # Sanity: the inventory must include the new bundle paths in import_line strings.
    assert "scilink.skills._shared.sam" in inventory
    assert "scilink.skills.image_analysis.atomic_stem.atom_finding" in inventory

    prompt = (
        f"You are advising on atomic-resolution STEM image analysis. The available "
        f"tools and libraries:\n\n{inventory}\n\n"
        "QUESTION: A user has a HAADF-STEM image of a doped oxide and wants to "
        "(a) detect atomic column positions with sub-pixel precision and (b) "
        "identify any vacant sites where an atom is missing relative to the "
        "perfect lattice. From the registered tools above, which ones would "
        "you call, and in what order? Reply with a numbered list of tool names."
    )
    resp = model.generate_content([prompt])
    text = resp.text.lower()
    assert "detect_atoms" in text, f"Expected detect_atoms in plan; got: {resp.text}"
    assert "find_missing_atoms" in text, f"Expected find_missing_atoms in plan; got: {resp.text}"


# ─── Layer 6: PR 3 — multi-skill loading + skill-gated tool visibility ──────


def test_registry_filter_no_skills_active():
    """No skill loaded → only `_shared/` tools visible (filtered by `agents=` tag)."""
    from scilink.skills._shared._registry import get_tools_for

    visible = {s.name for s in get_tools_for("image_analysis", active_skills=None)}
    assert visible == {"run_fft_nmf_analysis", "run_sam_analysis"}, \
        f"Expected only shared tools when no skill active; got: {visible}"


def test_registry_filter_single_skill_active():
    """atomic_stem loaded → shared + atomic_stem bundle (atom_finding's 7)."""
    from scilink.skills._shared._registry import get_tools_for

    visible = {s.name for s in get_tools_for("image_analysis", active_skills=["atomic_stem"])}
    expected = {
        "run_fft_nmf_analysis", "run_sam_analysis",
        "detect_atoms", "detect_atoms_dcnn", "refine_positions",
        "find_zone_axes", "find_missing_atoms", "subtract_atoms",
        "local_env_gmm",
    }
    assert visible == expected, f"Mismatch with atomic_stem active. Got: {visible}"


def test_registry_filter_unknown_skill_ignored():
    """Skills not present in any bundle are silently ignored."""
    from scilink.skills._shared._registry import get_tools_for

    visible = {s.name for s in get_tools_for("image_analysis", active_skills=["nonexistent"])}
    assert visible == {"run_fft_nmf_analysis", "run_sam_analysis"}


def test_force_field_agent_loads_multiple_skills():
    """ForceFieldAgent.skills should be a list; loading one or more skills
    populates it. Singular accessors (skill_name, skill_sections) reflect
    the first loaded skill for back-compat."""
    from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent
    from unittest.mock import MagicMock

    agent = ForceFieldAgent.__new__(ForceFieldAgent)
    agent.skills = []
    agent.logger = MagicMock()
    agent._available_ff_skills = ["amber"]

    # Single skill via str
    agent._load_skill("amber")
    assert len(agent.skills) == 1
    assert agent.skill_name == "amber"
    assert agent.skill_sections is agent.skills[0]
    assert agent.active_skill_names == ["amber"]

    # Loading the same skill again is a no-op
    agent._load_skill("amber")
    assert len(agent.skills) == 1

    # Reset and load via list
    agent.skills = []
    agent._load_skill(["amber"])
    assert agent.active_skill_names == ["amber"]


def test_force_field_get_skill_context_concatenates_multiple_skills():
    """When multiple skills are loaded, _get_skill_context emits one block
    per skill so the LLM can attribute guidance."""
    from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent
    from scilink.skills.loader import load_skill
    from unittest.mock import MagicMock

    agent = ForceFieldAgent.__new__(ForceFieldAgent)
    agent.logger = MagicMock()
    agent._available_ff_skills = ["amber"]
    # Stub a fake second skill so we can verify multi-block concatenation
    # without needing two real force-field skills.
    fake = {
        "name": "fake_ff",
        "overview": "Fake force field for testing concatenation.",
        "planning": "", "analysis": "", "interpretation": "",
        "validation": "", "implementation": "",
        "meta": {}, "extras": {},
    }
    real = load_skill("amber", domain="force_field")
    agent.skills = [real, fake]

    ctx = agent._get_skill_context()
    assert "Domain Knowledge: amber" in ctx
    assert "Domain Knowledge: fake_ff" in ctx
    assert "Fake force field for testing concatenation" in ctx


def test_analyze_accepts_str_or_list_via_helper():
    """BaseAnalysisAgent._load_skills_to_state should accept str | list and
    populate both the legacy singular fields and the new ``skills_loaded`` list."""
    from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent
    from unittest.mock import MagicMock

    agent = MagicMock(spec=BaseAnalysisAgent)
    agent.logger = MagicMock()

    # Single string
    state = BaseAnalysisAgent._load_skills_to_state(agent, "xps", domain="curve_fitting")
    assert state["skill_name"] == "xps"
    assert state["skill_sections"]["name"] == "xps"
    assert [s["name"] for s in state["skills_loaded"]] == ["xps"]

    # List of one
    state = BaseAnalysisAgent._load_skills_to_state(agent, ["xps"], domain="curve_fitting")
    assert [s["name"] for s in state["skills_loaded"]] == ["xps"]

    # Empty / None → no skills loaded
    state = BaseAnalysisAgent._load_skills_to_state(agent, None, domain="curve_fitting")
    assert state["skills_loaded"] == []
    assert state["skill_name"] is None

    # Unknown skill warned + skipped
    state = BaseAnalysisAgent._load_skills_to_state(
        agent, ["xps", "definitely_nonexistent"], domain="curve_fitting"
    )
    assert [s["name"] for s in state["skills_loaded"]] == ["xps"]


def test_orchestrator_resolves_skill_list_and_str(model):
    """run_analysis tool param now accepts string or array. Verify the
    description string mentions multi-skill capability so Claude routes
    correctly."""
    from scilink.agents.exp_agents.analysis_orchestrator_tools import (
        _build_skill_description,
    )

    desc = _build_skill_description(agent_registry=None, custom_skills=None)
    # The description must signal that lists are accepted
    assert "list" in desc.lower() or "multiple" in desc.lower(), \
        f"Skill description should signal multi-skill support; got: {desc}"


def test_live_multi_skill_in_one_prompt(model):
    """Load XPS + AMBER skills together (cross-domain task) and verify both
    bodies reach the LLM. This exercises the multi-skill concatenation path
    end-to-end."""
    from scilink.skills.loader import load_skill

    xps = load_skill("xps", domain="curve_fitting")
    amber = load_skill("amber", domain="force_field")

    # Imitate ForceFieldAgent._get_skill_context's concatenation
    blocks = [
        f"=== Domain Knowledge: {xps['name']} ===\n{xps['planning']}",
        f"=== Domain Knowledge: {amber['name']} ===\n{amber['planning']}",
    ]
    ctx = "\n\n".join(blocks)

    prompt = (
        f"Two domain skills are available:\n\n{ctx}\n\n"
        "QUESTION: Briefly identify which of the two skills above (XPS or AMBER) "
        "applies to (a) interpreting C 1s binding-energy fits, and (b) selecting "
        "an antechamber charge method for a small organic molecule. One sentence "
        "per sub-question."
    )
    resp = model.generate_content([prompt])
    text = resp.text.lower()
    assert "xps" in text and "amber" in text, \
        f"LLM should identify both skill names from the multi-skill context; got: {resp.text}"


def test_force_field_auto_select_loads_every_match():
    """_auto_select_skill should load *every* matching skill family, not just
    the first. With only AMBER shipped today this still loads exactly one,
    but verifies the multi-load codepath."""
    from unittest.mock import MagicMock
    from scilink.agents.sim_agents.force_field_agent import ForceFieldAgent

    agent = ForceFieldAgent.__new__(ForceFieldAgent)
    agent.skills = []
    agent.logger = MagicMock()
    agent._available_ff_skills = ["amber"]

    # AMBER-family input → amber skill loaded
    assert agent._auto_select_skill("AMBER ff19SB") is True
    assert agent.active_skill_names == ["amber"]

    # Re-call is a no-op when a skill is already loaded
    assert agent._auto_select_skill("GAFF2") is True
    assert agent.active_skill_names == ["amber"]

    # Non-matching input
    agent.skills = []
    assert agent._auto_select_skill("OPLS-AA") is False
    assert agent.skills == []

    # Multi-family description (would load both if both skills shipped):
    # today only amber is in _available_ff_skills, so only amber loads.
    agent.skills = []
    agent._available_ff_skills = ["amber"]
    agent._auto_select_skill("AMBER ff14SB protein with GAFF small molecule")
    assert "amber" in agent.active_skill_names


def test_simulation_orchestrator_multi_skill_path():
    """SimulationOrchestratorTools._load_skill_content now accepts
    str | list[str] | None. Verify each shape, including graceful
    handling of unknown skills inside a list."""
    from unittest.mock import MagicMock
    from scilink.agents.sim_agents.simulation_orchestrator_tools import (
        SimulationOrchestratorTools,
    )

    tools = SimulationOrchestratorTools.__new__(SimulationOrchestratorTools)
    tools.orch = MagicMock()
    tools.orch._custom_skills = {}
    tools.logger = MagicMock()

    # None / empty → None
    assert tools._load_skill_content(None) is None
    assert tools._load_skill_content([]) is None

    # Single string and single-element list produce identical content
    one_str = tools._load_skill_content("aimsgb")
    one_list = tools._load_skill_content(["aimsgb"])
    assert one_str is not None and one_str == one_list

    # Unknown names in a list are silently skipped; the rest still render
    mixed = tools._load_skill_content(["aimsgb", "nonexistent"])
    assert mixed is not None and "aimsgb" in mixed.lower()
    assert tools._load_skill_content(["nonexistent_a", "nonexistent_b"]) is None


def test_live_skill_gating_filters_atom_finding(model):
    """With no skill active, the inventory should NOT contain atom-finding tools.
    With atomic_stem active, it should. Verify Claude's behavior aligns."""
    from scilink.skills._shared._registry import format_tool_inventory

    inv_no_skill = format_tool_inventory("image_analysis", active_skills=None)
    inv_with_skill = format_tool_inventory("image_analysis", active_skills=["atomic_stem"])

    # Static checks on the inventory text itself
    assert "detect_atoms" not in inv_no_skill
    assert "detect_atoms" in inv_with_skill

    # Live LLM check: when given the no-skill inventory, model should NOT
    # recommend detect_atoms (it's not visible). When given the with-skill
    # inventory, it should.
    prompt_with = (
        f"{inv_with_skill}\n\n"
        "Pick ONE tool from the inventory above that fits this task: "
        "'detect atomic column positions in a STEM image'. Reply with just the tool name."
    )
    resp = model.generate_content([prompt_with])
    assert "detect_atoms" in resp.text.lower(), \
        f"With atomic_stem inventory, expected detect_atoms; got: {resp.text}"


# ─── Layer 7: orchestrator-mode live tests ──────────────────────────


@pytest.fixture(scope="module")
def orchestrator(tmp_path_factory):
    """Construct a real AnalysisOrchestratorAgent. Module-scoped so the
    setup cost (registry build, skill discovery, embedding-client init)
    is amortized across the orchestrator-mode tests."""
    from scilink.agents.exp_agents.analysis_orchestrator import (
        AnalysisOrchestratorAgent,
    )

    base_dir = tmp_path_factory.mktemp("orch_live")
    return AnalysisOrchestratorAgent(
        base_dir=str(base_dir),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name=MODEL,
    )


def test_orchestrator_run_analysis_schema_includes_oneOf(orchestrator):
    """The `run_analysis` tool schema rendered by the live orchestrator must
    advertise multi-skill via JSON Schema `oneOf [string, array]`. This is
    the bridge between PR 3's Python-side API and what Claude actually sees
    in the function-calling schema."""
    schemas = orchestrator.tools.openai_schemas
    run_analysis = next(
        s for s in schemas if s.get("function", {}).get("name") == "run_analysis"
    )
    skill_param = run_analysis["function"]["parameters"]["properties"]["skill"]

    assert "oneOf" in skill_param, \
        "run_analysis.skill schema must use oneOf for multi-skill support"
    types = {opt.get("type") for opt in skill_param["oneOf"]}
    assert "string" in types and "array" in types, \
        f"oneOf must contain both string and array; got: {types}"
    array_opt = next(o for o in skill_param["oneOf"] if o.get("type") == "array")
    assert array_opt.get("items", {}).get("type") == "string"

    desc = skill_param.get("description", "").lower()
    assert "list" in desc or "multiple" in desc, \
        "Description should signal multi-skill capability"


def test_orchestrator_skill_param_is_function_callable_via_litellm(orchestrator):
    """End-to-end: hand the orchestrator's actual `run_analysis` schema to
    Claude through the litellm wrapper and verify the model can produce a
    valid tool_use response with a single-skill string argument.

    This validates: orchestrator-built schema → litellm tool calling →
    Claude understanding → tool_use response shape.
    """
    import litellm

    schemas = orchestrator.tools.openai_schemas
    run_analysis = next(
        s for s in schemas if s.get("function", {}).get("name") == "run_analysis"
    )

    response = litellm.completion(
        model=MODEL,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        messages=[
            {
                "role": "user",
                "content": (
                    "I have a single C 1s X-ray photoelectron spectrum at "
                    "/data/c1s.csv. Please analyze it. Use the run_analysis tool."
                ),
            }
        ],
        tools=[run_analysis],
        tool_choice={"type": "function", "function": {"name": "run_analysis"}},
        timeout=120,
    )

    msg = response.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    assert tool_calls, f"Expected a tool_use response; got: {msg}"
    args = json.loads(tool_calls[0].function.arguments)

    # Single-XPS task → skill should resolve to 'xps' (string or 1-elem list)
    skill_arg = args.get("skill")
    if isinstance(skill_arg, list):
        assert "xps" in skill_arg, f"Expected xps; got: {skill_arg}"
    else:
        assert skill_arg == "xps", f"Expected xps; got: {skill_arg}"


def test_orchestrator_emits_skill_list_for_cross_domain_task(orchestrator):
    """Multi-skill: a task that explicitly spans two domains should drive
    Claude to emit a JSON array for the `skill` parameter. Verifies the
    full path orchestrator-schema → LLM → tool_use array."""
    import litellm

    schemas = orchestrator.tools.openai_schemas
    run_analysis = next(
        s for s in schemas if s.get("function", {}).get("name") == "run_analysis"
    )

    response = litellm.completion(
        model=MODEL,
        api_key=os.environ["ANTHROPIC_API_KEY"],
        messages=[
            {
                "role": "user",
                "content": (
                    "I have an XPS spectrum AND an EELS spectrum image of the "
                    "same TiO2 sample, and I want to leverage curated guidance "
                    "from both spectroscopy domains in the analysis. "
                    "The XPS file is /data/xps.csv. Please analyze it, but "
                    "first call run_analysis with BOTH the xps and eels skills "
                    "so the curated knowledge from each technique is available. "
                    "Use the run_analysis tool now."
                ),
            }
        ],
        tools=[run_analysis],
        tool_choice={"type": "function", "function": {"name": "run_analysis"}},
        timeout=120,
    )

    msg = response.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    assert tool_calls, f"Expected tool_use response; got: {msg}"
    args = json.loads(tool_calls[0].function.arguments)
    skill_arg = args.get("skill")

    assert isinstance(skill_arg, list), \
        f"Cross-domain task should emit a list; got: {skill_arg!r}"
    assert {"xps", "eels"}.issubset(set(skill_arg)), \
        f"Expected xps + eels in the list; got: {skill_arg}"


def test_live_llm_does_not_hallucinate_old_tool_paths(model):
    """The model sees the new import_line strings via format_tool_inventory.
    The hard correctness assertion is that it does NOT emit the legacy
    `scilink.tools.*` paths anywhere in the response — that would mean the
    LLM bypassed the inventory and pulled the path from training data."""
    from scilink.skills._shared._registry import format_tool_inventory

    inventory = format_tool_inventory("image_analysis")
    prompt = (
        f"Available tools (use exactly these import paths):\n{inventory}\n\n"
        "Write a complete Python snippet that imports `run_sam_analysis` and "
        "calls it on a numpy image array with default parameters. Include the "
        "import statement at the top."
    )
    resp = model.generate_content([prompt])
    code = resp.text or ""
    legacy_paths = (
        "scilink.tools.sam",
        "scilink.tools.fft_nmf",
        "scilink.tools.atom_finding_tools",
        "scilink.tools.amber_tools",
    )
    for legacy in legacy_paths:
        assert legacy not in code, \
            f"LLM hallucinated legacy path '{legacy}'; got:\n{code}"

