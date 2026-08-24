"""
Local smoke tests for MLIPAgent assembly + the potential/runner split.

These cover the relocation/wiring contract without making any LLM
calls and without invoking real MACE / CHGNet / NequIP / DeePMD
backends — every test that needs a potential builds a fake
``DeployedPotential`` by hand.

What they exercise:
  - MLIPAgent constructs and auto-loads the `general` skill from the
    machine_learning_potentials bundle directory, and merges backend
    bundles via _load_backend_skill.
  - The engine-neutral `DeployedPotential` contract flows through both
    runners: `_ase_runner.generate_ase_script` (universal) and
    `lammps.run_with_potential` (engine-side, raises for backends with
    no pair_style).
  - MLIPAgent.deploy_pretrained delegates run generation rather than
    generating inputs itself — so its only pre-LLM validation is the
    structure_file requirement.
"""

import tempfile
from pathlib import Path


def _agent_kwargs():
    return dict(api_key="sk-smoke-not-real", model_name="gpt-4o-mini")


def _fake_potential(backend="mace", model_file="/path/to/mace-mp-0.model",
                    elements=("Cu",)):
    """Build a DeployedPotential by hand — no real backend needed.

    The ASECalculatorSpec strings are intentionally identifiable so
    tests can assert they flow through the runner verbatim.
    """
    from scilink.agents.sim_agents._potential import (
        DeployedPotential, ASECalculatorSpec,
    )
    specs = {
        "mace": ASECalculatorSpec(
            import_line="from mace.calculators import mace_mp",
            construct_expr=(
                "mace_mp(model='medium', device=DEVICE, "
                "default_dtype='float64')"
            ),
            device_env_var="MACE_DEVICE",
        ),
        "chgnet": ASECalculatorSpec(
            import_line="from chgnet.model.dynamics import CHGNetCalculator",
            construct_expr="CHGNetCalculator(use_device=DEVICE)",
            device_env_var="CHGNET_DEVICE",
        ),
    }
    return DeployedPotential(
        kind="mlip",
        backend=backend,
        model_name=backend,
        model_file=model_file,
        elements=list(elements),
        ase_calculator=specs[backend],
        notes=f"fake {backend} potential for smoke test",
    )


def test_mlip_agent_assembly():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, **_agent_kwargs())

        assert agent.skill_name == "general", (
            f"Expected `general` skill to auto-load, got {agent.skill_name!r}. "
            f"If this fails, the skill loader's domain mapping has drifted."
        )
        assert agent.skill_sections is not None
        for section in ("planning", "validation", "implementation"):
            content = agent.skill_sections.get(section, "")
            assert content, (
                f"`general` skill is missing the {section!r} section -- "
                f"the agent will hand empty context to the LLM."
            )

        ctx = agent._get_skill_context(section="validation")
        assert "MLIP" in ctx or "Energy MAE" in ctx, (
            "validation context did not include the MLIP rules"
        )


def test_mlip_backend_skill_merge():
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, skill="mace", **_agent_kwargs())

        impl = agent.skill_sections.get("implementation", "")
        assert "MACE SPECIFIC" in impl, (
            "Backend skill merge marker missing. _load_backend_skill should "
            "append a `--- MACE SPECIFIC ---` block onto each section."
        )
        assert "pair_style" in impl and "mace" in impl, (
            "MACE LAMMPS pair_style guidance missing from merged context"
        )

        planning = agent.skill_sections.get("planning", "")
        assert "mace-mp-0" in planning or "mace-off23" in planning, (
            "MACE foundation-model selection guidance not merged into planning"
        )


def test_lammps_run_with_potential():
    """LAMMPS engine consumes a DeployedPotential and emits in.lammps."""
    from scilink.skills.molecular_dynamics.lammps import lammps

    pot = _fake_potential("mace")
    with tempfile.TemporaryDirectory() as td:
        path = lammps.run_with_potential(
            pot,
            structure_file="cu_bulk.data",
            working_dir=td,
            task="md",
            timestep=0.5,
            temperature=300.0,
            pressure=None,
        )
        assert Path(path).name == "in.lammps"
        content = Path(path).read_text()

        for required in (
            "units          metal",
            "atom_style     atomic",
            "pair_style     mace no_domain_decomposition",
            "pair_coeff     * * /path/to/mace-mp-0.model Cu",
            "fix            1 all nvt",
            "read_data      cu_bulk.data",
        ):
            assert required in content, (
                f"Generated LAMMPS input missing required line: {required!r}\n"
                f"--- content ---\n{content}"
            )
        assert "npt" not in content, (
            "NVT ensemble requested (pressure=None) but NPT slipped in"
        )

        npt_path = lammps.run_with_potential(
            pot, structure_file="cu_bulk.data", working_dir=td,
            task="md", temperature=300.0, pressure=1.0,
        )
        npt_content = Path(npt_path).read_text()
        assert "fix            1 all npt" in npt_content, (
            "pressure=1.0 should produce an NPT fix line"
        )


def test_lammps_relax_with_potential():
    """task='relax' produces a box/relax + minimize input, no dynamics."""
    from scilink.skills.molecular_dynamics.lammps import lammps

    pot = _fake_potential("mace")
    with tempfile.TemporaryDirectory() as td:
        path = lammps.run_with_potential(
            pot, structure_file="cu.data", working_dir=td, task="relax",
        )
        content = Path(path).read_text()
        assert "box/relax" in content and "minimize" in content
        assert "nvt" not in content and "npt" not in content, (
            "relax task should not emit a dynamics ensemble fix"
        )


def test_lammps_unsupported_backend_raises():
    """CHGNet has no LAMMPS pair_style — run_with_potential must raise."""
    from scilink.skills.molecular_dynamics.lammps import lammps

    pot = _fake_potential("chgnet", model_file="")
    with tempfile.TemporaryDirectory() as td:
        try:
            lammps.run_with_potential(
                pot, structure_file="x.data", working_dir=td,
            )
        except NotImplementedError as exc:
            assert "ASE" in str(exc) or "ase" in str(exc), (
                "error should point users at the ASE runner"
            )
        else:
            raise AssertionError(
                "CHGNet via LAMMPS should raise NotImplementedError"
            )


def test_ase_runner_nvt():
    """ASE runner is universal — drives any DeployedPotential."""
    from scilink.agents.sim_agents._ase_runner import generate_ase_script

    pot = _fake_potential("mace")
    with tempfile.TemporaryDirectory() as td:
        path = generate_ase_script(
            pot,
            working_dir=td,
            structure_file="cu_bulk.data",
            task="md",
            timestep=1.0,
            temperature=300.0,
            pressure=None,
            n_steps=100,
            output_interval=10,
        )
        assert Path(path).name == "run_md.py"
        content = Path(path).read_text()

        for required in (
            "from mace.calculators import mace_mp",        # spec.import_line
            "mace_mp(model='medium', device=DEVICE",       # spec.construct_expr
            "MACE_DEVICE",                                 # spec.device_env_var
            "from ase.md.langevin import Langevin",
            "MaxwellBoltzmannDistribution",
            'Trajectory("traj.traj"',
            "thermo.log",
            "dyn.run(100)",
            "read_lammps_data('cu_bulk.data'",
            "ELEMENTS = ['Cu']",
        ):
            assert required in content, (
                f"ASE script missing expected token: {required!r}"
            )
        assert "from ase.md.npt import NPT" not in content, (
            "NVT requested (pressure=None) but NPT import slipped in"
        )


def test_ase_runner_npt():
    from scilink.agents.sim_agents._ase_runner import generate_ase_script

    pot = _fake_potential("mace")
    with tempfile.TemporaryDirectory() as td:
        path = generate_ase_script(
            pot, working_dir=td, task="md", pressure=1.0,
        )
        content = Path(path).read_text()
        assert "from ase.md.npt import NPT" in content
        assert "externalstress=1.0 * units.bar" in content


def test_ase_runner_relax():
    """task='relax' emits run_relax.py with BFGS + cell filter."""
    from scilink.agents.sim_agents._ase_runner import generate_ase_script

    pot = _fake_potential("mace")
    with tempfile.TemporaryDirectory() as td:
        path = generate_ase_script(
            pot, working_dir=td, task="relax", fmax=0.01,
        )
        assert Path(path).name == "run_relax.py"
        content = Path(path).read_text()
        for required in (
            "from ase.optimize import BFGS",
            "from ase.filters import FrechetCellFilter",
            "relax_result.json",
            "relaxed.xyz",
            "fmax=0.01",
        ):
            assert required in content, (
                f"relax script missing expected token: {required!r}"
            )


def test_ase_runner_rejects_bad_task():
    """The ASE runner is backend-agnostic; what it validates is `task`."""
    from scilink.agents.sim_agents._ase_runner import generate_ase_script

    pot = _fake_potential("mace")
    with tempfile.TemporaryDirectory() as td:
        try:
            generate_ase_script(pot, working_dir=td, task="banana")
        except ValueError as exc:
            assert "task" in str(exc)
        else:
            raise AssertionError("Invalid task should raise ValueError")


def test_ase_runner_spec_passthrough():
    """Backend-specific bits are *data* on the spec, not code in the
    runner — whatever the spec says shows up verbatim in the script.
    This is the contract that lets a new backend ship without touching
    the runner.
    """
    from scilink.agents.sim_agents._ase_runner import generate_ase_script

    pot = _fake_potential("chgnet")
    with tempfile.TemporaryDirectory() as td:
        path = generate_ase_script(pot, working_dir=td, task="md")
        content = Path(path).read_text()
        assert "from chgnet.model.dynamics import CHGNetCalculator" in content
        assert "CHGNetCalculator(use_device=DEVICE)" in content
        assert "CHGNET_DEVICE" in content
        # the mace spec strings must not leak in
        assert "mace_mp" not in content
        assert "from mace.calculators" not in content


def test_agent_requires_structure_file():
    """deploy_pretrained delegates run generation, so its only pre-LLM
    validation is that a structure_file was supplied."""
    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    with tempfile.TemporaryDirectory() as td:
        agent = MLIPAgent(working_dir=td, **_agent_kwargs())
        try:
            agent.deploy_pretrained(
                system_info={"elements": {"Cu": 4}, "n_atoms": 4},
                research_goal="test",
                runner="ase",
            )
        except ValueError as exc:
            assert "structure_file" in str(exc)
        else:
            raise AssertionError(
                "deploy_pretrained without structure_file should raise"
            )


def test_chgnet_skill_discoverable():
    """CHGNet skill bundle should appear in supported_software."""
    from scilink.skills.loader import list_skills, load_skill

    engines = list_skills(domain="machine_learning_potentials")
    assert "chgnet" in engines, (
        f"chgnet bundle not discovered (got {engines}). The skill bundle "
        "at scilink/skills/machine_learning_potentials/chgnet/chgnet.md "
        "should be on disk."
    )

    skill = load_skill("chgnet", domain="machine_learning_potentials")
    for section in ("planning", "implementation", "validation"):
        assert skill.get(section), (
            f"chgnet skill missing the {section!r} section"
        )

    detect = skill.get("meta", {}).get("detect", {})
    assert detect.get("python_modules") == ["chgnet"], (
        "chgnet frontmatter must declare python_modules: [chgnet] so "
        "AvailableSoftware.detect() picks it up via importlib"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"=== {name} ===")
            fn()
            print("  OK")
    print("\nAll smokes passed.")
