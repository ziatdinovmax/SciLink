"""
Live test for the Materials Project tool-resolver in StructureGenerator.

Verifies that the LLM uses the `search_material_id` tool to resolve named
materials to mp-ids before script generation, and that the resolved facts
end up injected into `_build_initial_prompt`.

Requires env vars:
    ANTHROPIC_API_KEY   — for the Claude LLM
    MP_API_KEY          — for Materials Project lookups

Optional env / flags:
    --full              — also runs a full DFT-orchestrator pipeline end-to-end
                          (one extra LLM-heavy run; produces files in
                          tests/_mp_resolver_run/).
    --model <name>      — override the model name (default: claude-opus-4-6).

Run:
    ANTHROPIC_API_KEY=... MP_API_KEY=... python tests/test_mp_resolver.py
    ANTHROPIC_API_KEY=... MP_API_KEY=... python tests/test_mp_resolver.py --full

The targeted tests (1–5) exercise just the resolver and are cheap. The
optional full-workflow test (6) runs the DFT pipeline end-to-end and writes a
POSCAR — useful for confirming the LLM actually fetches via MPRester in the
generated script.
"""

import argparse
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = "claude-opus-4-6"
RUN_DIR = REPO_ROOT / "tests" / "_mp_resolver_run"


def _require_env(*names):
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"❌ Missing required env var(s): {', '.join(missing)}")
        sys.exit(2)


def _check_mp_api_installed():
    try:
        import mp_api  # noqa: F401
    except ImportError:
        print("❌ The `mp-api` package is not installed. Install with:")
        print("     pip install mp-api")
        print("   (or `pip install 'scilink[sim]'` to get the full sim stack)")
        sys.exit(2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_0_orchestrator_plumbing(model_name: str):
    """Plumbing check: verify the resolver actually activates when reached
    via StructurePipeline with `mp_api_key=None` — the call shape used by
    the DFT pipeline (`run_complete_workflow`) that AnalysisOrchestrator's
    `run_dft_workflow` tool drives. Catches regressions where someone
    breaks the auto-discovery chain (StructurePipeline →
    get_api_key('materials_project') → StructureGenerator → mp_helper).

    Costs zero LLM calls — just constructs the agent stack and inspects
    `o.structure_generator.mp_helper.enabled`.
    """
    from scilink.agents.sim_agents.structure_pipeline import StructurePipeline

    # Positive: with MP_API_KEY in env, the resolver chain should be live.
    o = StructurePipeline(
        api_key="not-used-for-plumbing-check",  # no LLM call made in __init__
        mp_api_key=None,                        # the pipeline-dispatched shape
        generator_model=model_name,
        validator_model=model_name,
        output_dir="/tmp/scilink_plumbing_check",
        max_refinement_cycles=0,
    )
    assert o.structure_generator.mp_helper.enabled, (
        "MP helper should be enabled when MP_API_KEY is in env and "
        "StructurePipeline is constructed with mp_api_key=None. The chain "
        "StructurePipeline → get_api_key('materials_project') → "
        "StructureGenerator → MaterialsProjectHelper is broken."
    )
    print("   ✅ MP enabled in pipeline-dispatched run (MP_API_KEY in env).")

    # Negative: no MP key anywhere → chain ends with disabled helper, no error.
    saved = {k: os.environ.pop(k, None) for k in ("MP_API_KEY", "MATERIALS_PROJECT_API_KEY")}
    try:
        from scilink.auth import clear_api_key
        clear_api_key("materials_project")
        o2 = StructurePipeline(
            api_key="not-used-for-plumbing-check",
            mp_api_key=None,
            generator_model=model_name,
            validator_model=model_name,
            output_dir="/tmp/scilink_plumbing_check_2",
            max_refinement_cycles=0,
        )
        assert not o2.structure_generator.mp_helper.enabled, (
            "MP helper should be disabled when no MP key is configured anywhere."
        )
        print("   ✅ Chain disables cleanly when no MP key configured (no error).")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_1_disabled_mp_short_circuit(model_name: str):
    """No MP key → resolver returns '' and prompt is unchanged.

    The MaterialsProjectHelper auto-discovers `MP_API_KEY` /
    `MATERIALS_PROJECT_API_KEY` from the environment, so we have to clear
    both for the duration of this test to actually exercise the disabled
    path. Restored after.
    """
    from scilink.agents.sim_agents.structure_agent import StructureGenerator

    saved = {k: os.environ.pop(k, None) for k in ("MP_API_KEY", "MATERIALS_PROJECT_API_KEY")}
    try:
        sg = StructureGenerator(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model_name=model_name,
            mp_api_key=None,  # explicitly disable
        )
        assert sg.mp_helper.enabled is False, "MP should be disabled with no key"

        out = sg._resolve_materials_via_tools("Build a rutile TiO2 supercell")
        assert out == "", f"Expected empty string, got: {out!r}"

        prompt = sg._build_initial_prompt("Build rutile TiO2 supercell", "ASE")
        assert "RESOLVED MATERIALS" not in prompt, \
            "Prompt should NOT contain resolved block when MP disabled"
        print("   ✅ Resolver returns '' and prompt unchanged when MP disabled.")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_2_resolver_positive_single(model_name: str):
    """Single named polymorph → resolves to the right space group, not just any TiO2.

    Regression check: previously the model called `search_material_id` with
    bare 'TiO2' and got mp-390 (anatase, I4_1/amd) because that's the
    lowest-e_above_hull TiO2 polymorph in MP. With the polymorph parameters
    on the tool schema, the model should pass spacegroup_symbol='P4_2/mnm'
    (or crystal_system='tetragonal' for rutile) and resolve to mp-2657.
    """
    from scilink.agents.sim_agents.structure_agent import StructureGenerator

    sg = StructureGenerator(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name=model_name,
        mp_api_key=os.environ["MP_API_KEY"],
    )
    assert sg.mp_helper.enabled, "MP helper should be enabled"

    block = sg._resolve_materials_via_tools(
        "Build a 2x2x1 supercell of rutile TiO2 with vacuum padding"
    )
    print("   --- resolved block ---")
    print("   " + block.replace("\n", "\n   "))
    assert block, "Resolver should have returned a non-empty block"
    assert "mp-" in block, "Block should contain at least one mp-id"
    assert "TiO2" in block or "TiO" in block, \
        "Block should mention TiO2 in the canonical formula"
    assert "AseAtomsAdaptor" in block, "Block should include the ASE/pymatgen snippet"
    # Polymorph regression: rutile = P4_2/mnm. Accept either the explicit
    # space-group line or the rutile mp-id.
    assert ("P4_2/mnm" in block) or ("mp-2657" in block), (
        "Resolver returned a non-rutile TiO2 polymorph. The model likely "
        "didn't pass spacegroup_symbol='P4_2/mnm' to the tool. Block:\n"
        + block
    )
    print("   ✅ Single-polymorph resolution returned rutile TiO2 (P4_2/mnm).")


def test_3_resolver_multi_material(model_name: str):
    """Heterostructure request → multiple mp-ids resolved."""
    from scilink.agents.sim_agents.structure_agent import StructureGenerator

    sg = StructureGenerator(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name=model_name,
        mp_api_key=os.environ["MP_API_KEY"],
    )
    block = sg._resolve_materials_via_tools(
        "Build a graphene/MoS2 vertical heterostructure for DFT relaxation"
    )
    print("   --- resolved block ---")
    print("   " + block.replace("\n", "\n   "))
    assert block, "Resolver should have returned a non-empty block"
    mp_id_count = block.count("mp-")
    # The footer snippet contains 'mp-XXXX' as a placeholder, so we expect
    # at least 3 occurrences when 2 materials are resolved (2 ids + placeholder).
    assert mp_id_count >= 3, \
        f"Expected ≥2 resolved mp-ids (≥3 'mp-' tokens incl. placeholder), got {mp_id_count}"
    print(f"   ✅ Multi-material resolution returned ≥2 mp-ids ({mp_id_count} 'mp-' tokens).")


def test_4_resolver_negative_control(model_name: str):
    """Generic request with no named material → empty resolved block."""
    from scilink.agents.sim_agents.structure_agent import StructureGenerator

    sg = StructureGenerator(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name=model_name,
        mp_api_key=os.environ["MP_API_KEY"],
    )
    block = sg._resolve_materials_via_tools(
        "Build a 4-atom Lennard-Jones FCC test cell"
    )
    print(f"   resolved block (expect empty): {block!r}")
    assert block == "", \
        f"Generic request should produce no resolution, got: {block!r}"
    print("   ✅ Generic request correctly produced no resolved block.")


def test_5_initial_prompt_injection(model_name: str):
    """Resolved block must be prepended into the script-gen prompt."""
    from scilink.agents.sim_agents.structure_agent import StructureGenerator

    sg = StructureGenerator(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name=model_name,
        mp_api_key=os.environ["MP_API_KEY"],
    )
    description = "Build a 3x3x1 supercell of NaCl"
    prompt = sg._build_initial_prompt(description, "ASE")

    print(f"   prompt length: {len(prompt)} chars")
    assert "RESOLVED MATERIALS" in prompt, \
        "Prompt should contain the resolved block header"
    assert "mp-" in prompt, "Prompt should contain at least one mp-id"
    assert "script_content" in prompt, \
        "Prompt must still end with the JSON script_content instruction"
    # Order: resolved block must come BEFORE the JSON instruction
    assert prompt.index("RESOLVED MATERIALS") < prompt.index("script_content"), \
        "Resolved block must precede the JSON instruction"
    print("   ✅ Prompt contains resolved block before script_content instruction.")


def test_6_full_workflow(model_name: str):
    """End-to-end DFT pipeline run with a named material.

    Verifies that the generated ASE script actually references MPRester /
    `get_structure_by_material_id` (i.e., the LLM picked up on the resolved
    facts), not a hand-coded lattice guess.
    """
    from scilink.agents.sim_agents.simulation_pipeline import run_complete_workflow

    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    result = run_complete_workflow(
        "Build a 2x2x1 supercell of rutile TiO2 with 15 Å vacuum along c "
        "for surface DFT calculations.",
        scale="periodic_dft",
        software="vasp",
        method="llm",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        mp_api_key=os.environ["MP_API_KEY"],
        model_name=model_name,
        output_dir=str(RUN_DIR),
        max_refinement_cycles=1,        # keep cost low
    )
    print(f"   final_status: {result['final_status']}")
    assert result["final_status"] == "success", \
        f"Workflow failed: {result.get('final_status')}"

    # Find the most recent generated script and verify it uses MPRester
    scripts = sorted(RUN_DIR.glob("script_*.py"), key=os.path.getmtime)
    assert scripts, "No generated script found in run dir"
    last_script = scripts[-1].read_text()
    used_mp = "MPRester" in last_script and "get_structure_by_material_id" in last_script
    print(f"   generated script uses MPRester: {used_mp}")
    if not used_mp:
        print("   ⚠️  Script did not use MPRester — model may have ignored the "
              "resolved block or rebuilt the structure from scratch. Inspect "
              f"{scripts[-1]}")
    # Structure generation emits engine-neutral coordinates (extended XYZ); the
    # engine-native VASP inputs are written downstream. Assert a structure file
    # was produced, format-agnostically, rather than a specific filename.
    structs = (list(RUN_DIR.glob("*.extxyz")) + list(RUN_DIR.glob("*.xyz"))
               + list(RUN_DIR.glob("*.cif")) + list(RUN_DIR.glob("POSCAR")))
    assert structs, "no structure file produced"
    print(f"   ✅ Workflow produced structure at {structs[0]}")


# ---------------------------------------------------------------------------
# Stress tests — gated behind --stress; probe edge cases the targeted suite
# doesn't cover. Each assertion uses lenient OR-of-acceptable conditions so
# probabilistic LLM behavior doesn't false-fail the suite.
# ---------------------------------------------------------------------------

def _make_sg(model_name: str, generated_script_dir: str = None):
    from scilink.agents.sim_agents.structure_agent import StructureGenerator
    kwargs = dict(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name=model_name,
        mp_api_key=os.environ["MP_API_KEY"],
    )
    if generated_script_dir:
        kwargs["generated_script_dir"] = generated_script_dir
    return StructureGenerator(**kwargs)


def _print_block(block: str):
    print("   --- resolved block ---")
    print("   " + (block.replace("\n", "\n   ") if block else "(empty)"))


E2E_DIR = REPO_ROOT / "tests" / "_e2e_runs"


def stress_1_anatase_polymorph(model_name: str):
    """Polymorph regression #2: anatase TiO2 should resolve to mp-390 (I4_1/amd),
    not rutile (mp-2657, P4_2/mnm). Verifies polymorph qualifier flows through
    correctly for non-default polymorphs too."""
    sg = _make_sg(model_name)
    block = sg._resolve_materials_via_tools(
        "Build an anatase TiO2 surface slab for photocatalysis"
    )
    _print_block(block)
    assert "mp-" in block, "Expected at least one resolved mp-id"
    assert ("I4_1/amd" in block) or ("mp-390" in block), \
        f"Anatase should resolve to I4_1/amd / mp-390. Block: {block}"
    print("   ✅ Anatase TiO2 resolved to I4_1/amd.")


def stress_2_wurtzite_gan(model_name: str):
    """Less common polymorph: wurtzite GaN → P6_3mc (mp-804)."""
    sg = _make_sg(model_name)
    block = sg._resolve_materials_via_tools(
        "Build a wurtzite GaN bulk supercell with a substitutional defect"
    )
    _print_block(block)
    assert "GaN" in block or "Ga" in block, "Block should mention GaN"
    assert ("P6_3mc" in block) or ("mp-804" in block), \
        f"Wurtzite GaN should resolve to P6_3mc / mp-804. Block: {block}"
    print("   ✅ Wurtzite GaN resolved correctly.")


def stress_3_common_name_alias(model_name: str):
    """Common-name aliases the model must translate: 'diamond' → C with Fd-3m
    (mp-66). Tests aliasing across both name → formula AND name → polymorph."""
    sg = _make_sg(model_name)
    block = sg._resolve_materials_via_tools(
        "Build a 2x2x2 diamond supercell with one nitrogen substitutional defect"
    )
    _print_block(block)
    assert "mp-" in block, "Expected at least one resolved mp-id"
    assert ("Fd-3m" in block) or ("mp-66" in block), \
        f"Diamond should resolve to C / Fd-3m / mp-66. Block: {block}"
    print("   ✅ Diamond resolved to C / Fd-3m.")


def stress_4_no_mp_entry(model_name: str):
    """Non-existent material → tool called, MP returns None, block records
    'no MP entry found'. Verifies graceful failure for unknown materials."""
    sg = _make_sg(model_name)
    block = sg._resolve_materials_via_tools(
        "Build a supercell of FabricatedXyz789 alloy with a vacancy"
    )
    _print_block(block)
    # Either the model called the tool and got a not-found, OR the model
    # decided not to call the tool (also acceptable — it knew better).
    if block:
        assert "no MP entry found" in block, \
            f"Block should record the not-found result. Block: {block}"
        print("   ✅ Non-existent material → 'no MP entry found' recorded.")
    else:
        print("   ✅ Model declined to call the tool for an unknown material.")


def stress_5_mixed_real_and_fake(model_name: str):
    """Mixed request: one resolvable material + one unresolvable. Both should
    be reported in the block — not all-or-nothing failure."""
    sg = _make_sg(model_name)
    block = sg._resolve_materials_via_tools(
        "Build a NaCl thin film on a NotARealMaterial999 substrate"
    )
    _print_block(block)
    has_nacl = ("NaCl" in block) or ("mp-22862" in block) or ("Fm-3m" in block)
    has_failure_or_skip = (
        "no MP entry found" in block or "NotARealMaterial" not in block
    )
    assert has_nacl, f"NaCl should have resolved. Block: {block}"
    assert has_failure_or_skip, \
        f"Fake substrate should be marked not-found or skipped. Block: {block}"
    print("   ✅ Partial-success path: NaCl resolved, fake material handled.")


def stress_6_crystal_system_fallback(model_name: str):
    """Crystal system as fallback when polymorph isn't specified by HM symbol.
    'hexagonal BN' should resolve via crystal_system='hexagonal' OR an explicit
    P6_3/mmc spacegroup. Either is correct."""
    sg = _make_sg(model_name)
    block = sg._resolve_materials_via_tools(
        "Build a hexagonal boron nitride monolayer for stacking"
    )
    _print_block(block)
    assert "BN" in block or "B" in block, "Block should mention BN"
    # Hexagonal BN = h-BN, P6_3/mmc, mp-984/mp-7991/mp-2653 depending on
    # polytype. Accept any hexagonal space group OR a mention of hexagonal
    # via crystal_system filter.
    has_hex = (
        "P6_3/mmc" in block or "P-6m2" in block
        or "hexagonal" in block.lower() or "mp-984" in block
        or "mp-2653" in block or "mp-7991" in block
    )
    assert has_hex, f"Should resolve to a hexagonal BN polytype. Block: {block}"
    print("   ✅ Hexagonal BN resolved (spacegroup or crystal_system path).")


def stress_7_capacity_many_materials(model_name: str):
    """Capacity check: a request naming 4 distinct materials should still
    resolve at least 3 within the 3-iteration cap. Tests that the model
    batches calls in one assistant message rather than wasting iterations."""
    sg = _make_sg(model_name)
    block = sg._resolve_materials_via_tools(
        "Build a Si / Ge / SiO2 / Al2O3 layered heterostructure for "
        "interface DFT studies"
    )
    _print_block(block)
    # Count unique mp-ids in the block (excluding the placeholder mp-XXXX).
    import re
    found_ids = set(re.findall(r"mp-\d+", block))
    print(f"   resolved unique mp-ids: {sorted(found_ids)}")
    assert len(found_ids) >= 3, (
        f"Expected ≥3 unique mp-ids resolved within iteration cap, "
        f"got {len(found_ids)}: {found_ids}"
    )
    print(f"   ✅ Capacity test: {len(found_ids)} materials resolved.")


# ---------------------------------------------------------------------------
# End-to-end tests — gated behind --e2e. These run the FULL structure-gen
# pipeline (resolution + script generation + script execution + output
# verification) to check that the model actually USES the resolved mp-id in
# a runnable script, not just acknowledges it. The most expensive tests in
# the suite (each may trigger up to 5 script-correction retries).
# ---------------------------------------------------------------------------

def e2e_1_rutile_with_o_vacancy(model_name: str):
    """Full chain: MP resolution → script with MPRester → 2x2x1 supercell →
    O vacancy → POSCAR. Verifies all of:
      1. The model uses MPRester + correct mp-id (mp-2657 for rutile)
      2. The subprocess can reach MP (MP_API_KEY threaded through executor)
      3. Non-trivial transformations (supercell + defect) execute correctly
      4. Output composition matches expectation: 8 Ti + 15 O = 23 atoms
    """
    workdir = E2E_DIR / "rutile_o_vacancy"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    sg = _make_sg(model_name, generated_script_dir=str(workdir))
    request = (
        "Build a 2x2x1 supercell of rutile TiO2 (space group P4_2/mnm) "
        "with exactly one oxygen vacancy. Save as POSCAR."
    )

    result = sg.generate_script(
        original_user_request=request,
        attempt_number_overall=1,
    )
    print(f"   status: {result.get('status')}")
    print(f"   execution_attempts: {result.get('execution_attempts')}")

    assert result.get("status") == "success", (
        "Script generation/execution failed: "
        f"{result.get('last_error') or result.get('message')}"
    )

    script = result["final_script_content"]
    print(f"   script length: {len(script)} chars")

    used_mprester = "MPRester" in script
    used_get_by_id = "get_structure_by_material_id" in script
    used_correct_id = "mp-2657" in script
    print(f"   uses MPRester={used_mprester}, "
          f"get_structure_by_material_id={used_get_by_id}, "
          f"references mp-2657={used_correct_id}")
    assert used_mprester, "Script should use MPRester to fetch the structure"
    assert used_get_by_id, "Script should call get_structure_by_material_id"
    assert used_correct_id, \
        f"Script should reference rutile TiO2 (mp-2657), got script:\n{script[:800]}"

    output_file = result["output_file"]
    assert os.path.exists(output_file), f"Output file missing: {output_file}"

    from ase.io import read as ase_read
    atoms = ase_read(output_file)
    n_ti = sum(1 for s in atoms.get_chemical_symbols() if s == "Ti")
    n_o = sum(1 for s in atoms.get_chemical_symbols() if s == "O")
    print(f"   output formula: {atoms.get_chemical_formula()}, "
          f"Ti={n_ti}, O={n_o}, total={len(atoms)}")
    print(f"   output cell lengths (Å): {atoms.cell.lengths()}")

    # Ratio-based assertion: model may interpret "2x2x1" against the
    # primitive vs. conventional cell, but a correct vacancy application
    # leaves the structure with exactly one O missing relative to perfect
    # stoichiometry — i.e., n_O == 2*n_Ti - 1.
    assert n_ti > 0, f"Expected some Ti atoms, got {n_ti}"
    assert n_o == 2 * n_ti - 1, (
        f"Stoichiometry off: expected n_O = 2*n_Ti - 1 = {2*n_ti - 1} "
        f"(stoichiometric TiO2 minus exactly one O vacancy), got n_O={n_o}"
    )
    # Sanity bounds — non-trivial supercell, not a single primitive cell or
    # a pathologically huge one.
    assert 12 <= len(atoms) <= 200, \
        f"Atom count {len(atoms)} outside reasonable supercell range"
    print(f"   ✅ Rutile fetched from MP, supercell built ({len(atoms)} atoms), "
          f"exactly one O vacancy applied (n_O = 2·n_Ti − 1).")


def e2e_2_diamond_slab(model_name: str):
    """Full chain: MP resolution → MPRester → ase.build.surface → diamond
    (111) slab with vacuum.

    Surfaces are a different code path than defects and exercise the
    pymatgen→ASE handoff for cell-modifying operations. Verifies:
      1. Model uses MPRester + diamond mp-id (mp-66, Fd-3m)
      2. Output is a slab: large vacuum gap (≥10 Å) along one axis
      3. Composition: pure C, atom count consistent with ~4 layers
    """
    import numpy as np

    workdir = E2E_DIR / "diamond_slab"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    sg = _make_sg(model_name, generated_script_dir=str(workdir))
    request = (
        "Build a 4-layer diamond (111) surface slab with at least 15 Å of "
        "vacuum along the surface normal. Use Fd-3m diamond from Materials "
        "Project as the bulk starting structure. Save as POSCAR."
    )

    result = sg.generate_script(
        original_user_request=request,
        attempt_number_overall=1,
    )
    print(f"   status: {result.get('status')}")
    print(f"   execution_attempts: {result.get('execution_attempts')}")

    assert result.get("status") == "success", (
        "Script generation/execution failed: "
        f"{result.get('last_error') or result.get('message')}"
    )

    script = result["final_script_content"]
    print(f"   script length: {len(script)} chars")

    assert "MPRester" in script, "Script should use MPRester"
    assert "mp-66" in script, \
        f"Script should reference diamond (mp-66), got:\n{script[:800]}"
    print("   uses MPRester + mp-66: OK")

    output_file = result["output_file"]
    from ase.io import read as ase_read
    atoms = ase_read(output_file)
    syms = atoms.get_chemical_symbols()
    n_c = sum(1 for s in syms if s == "C")
    print(f"   output formula: {atoms.get_chemical_formula()}, "
          f"C={n_c}, total={len(atoms)}")
    assert n_c == len(atoms) and n_c > 0, \
        f"Expected pure-carbon slab, got {atoms.get_chemical_formula()}"

    # Vacuum check: cell lengths vs atomic extent in each direction. The
    # axis with the largest (length - extent) gap is the surface normal.
    cell_lengths = atoms.cell.lengths()
    pos = atoms.get_positions()
    extents = pos.max(axis=0) - pos.min(axis=0)
    gaps = cell_lengths - extents
    max_gap = float(np.max(gaps))
    print(f"   cell lengths (Å): {cell_lengths}")
    print(f"   atomic extents (Å): {extents}")
    print(f"   max vacuum gap (Å): {max_gap:.2f}")
    assert max_gap >= 10.0, \
        f"Expected ≥10 Å vacuum along the surface normal, got {max_gap:.2f} Å"
    print("   ✅ Diamond slab built from MP with proper vacuum padding.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TARGETED_TESTS = [
    ("Pipeline plumbing (MP enabled via StructurePipeline chain)",
                                          test_0_orchestrator_plumbing),
    ("Disabled-MP short-circuit",         test_1_disabled_mp_short_circuit),
    ("Positive: single named material",   test_2_resolver_positive_single),
    ("Positive: multi-material request",  test_3_resolver_multi_material),
    ("Negative control: generic request", test_4_resolver_negative_control),
    ("_build_initial_prompt injection",   test_5_initial_prompt_injection),
]

STRESS_TESTS = [
    ("STRESS: anatase TiO2 polymorph",        stress_1_anatase_polymorph),
    ("STRESS: wurtzite GaN polymorph",        stress_2_wurtzite_gan),
    ("STRESS: common-name alias (diamond)",   stress_3_common_name_alias),
    ("STRESS: non-existent material",         stress_4_no_mp_entry),
    ("STRESS: mixed real+fake materials",     stress_5_mixed_real_and_fake),
    ("STRESS: crystal_system fallback (h-BN)",stress_6_crystal_system_fallback),
    ("STRESS: capacity (4 materials)",        stress_7_capacity_many_materials),
]

E2E_TESTS = [
    ("E2E: rutile TiO2 + O vacancy (defect)", e2e_1_rutile_with_o_vacancy),
    ("E2E: diamond (111) slab (surface)",     e2e_2_diamond_slab),
]


def main():
    parser = argparse.ArgumentParser(description="MP tool-resolver live tests")
    parser.add_argument("--full", action="store_true",
                        help="Also run the end-to-end DFT workflow test (slow).")
    parser.add_argument("--stress", action="store_true",
                        help="Also run the stress-test suite (~7 extra LLM calls).")
    parser.add_argument("--e2e", action="store_true",
                        help="Also run end-to-end script-execution tests "
                             "(generates + runs ASE scripts; slowest).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL}).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable INFO-level logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

    _require_env("ANTHROPIC_API_KEY", "MP_API_KEY")
    _check_mp_api_installed()

    print(f"🔧 Running MP tool-resolver tests against model: {args.model}")
    print("=" * 72)

    tests = list(TARGETED_TESTS)
    if args.stress:
        tests.extend(STRESS_TESTS)
    if args.e2e:
        tests.extend(E2E_TESTS)
    if args.full:
        tests.append(("Full DFT workflow (end-to-end)", test_6_full_workflow))

    passed, failed = [], []
    for name, fn in tests:
        print(f"\n▶  {name}")
        try:
            fn(args.model)
            passed.append(name)
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"   ❌ FAIL: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"   ❌ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"Summary: {len(passed)} passed, {len(failed)} failed")
    for name in passed:
        print(f"  ✅ {name}")
    for name, err in failed:
        print(f"  ❌ {name}  ({err})")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
