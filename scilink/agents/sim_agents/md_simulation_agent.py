# agents/sim_agents/md_simulation_agent.py
"""
MD simulation agent. Handles classical atomistic dynamics
across engines (LAMMPS, GROMACS, OpenMM, etc.) via skills.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from .base_agent import SimulationAgent
from ._potential import DeployedPotential

_TOOL_REGISTRY: Dict[str, Any] = {}
try:
    # Tools live alongside the skill bundle (cf. force_field/amber/amber.py).
    from ...skills.molecular_dynamics.lammps import lammps as lammps_tools
    _TOOL_REGISTRY["lammps"] = lammps_tools
except ImportError:
    pass


def _assemble_fanout_stage(
    members: List[Dict[str, str]],
    entry_file: str,
    shared_files: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Assemble a one-stage fan-out campaign from expanded member scripts.

    Each member becomes a self-contained run: the shared files (structure data,
    force fields) plus its own substituted script under ``entry_file``. The
    result is the engine-neutral ``stages`` structure ``_collect_stages``
    consumes — pure and engine-agnostic, so it is unit-testable without an
    agent, an LLM, or an engine.

    Args:
        members: ``{"name", "script"}`` dicts from ``expand_parameter_sweep``.
        entry_file: Filename each member's script is written under.
        shared_files: Files every member needs (filename → contents).

    Returns:
        A single-element list holding the production fan-out stage.
    """
    member_specs = []
    for member in members:
        input_files = dict(shared_files)
        input_files[entry_file] = member["script"]
        member_specs.append({
            "name": member["name"],
            "input_files": input_files,
            "entry_file": entry_file,
        })
    return [{"name": "production", "parallel": True, "members": member_specs}]


def _assemble_sequential_stages(
    steps: List[Dict[str, str]],
    shared_files: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Assemble an ordered sequential campaign from staged step scripts.

    Each step becomes one sequential stage. The stages share a run directory
    (``_collect_stages`` does not isolate sequential steps), so a stage's
    restart files are available to the next — the optimization → equilibration
    → production chain. Every stage carries the shared files (structure data,
    force fields) plus its own script under its ``entry_file``. The result is
    the engine-neutral ``stages`` structure ``_collect_stages`` consumes — pure
    and engine-agnostic, so it is unit-testable without an agent, an LLM, or an
    engine.

    Args:
        steps: Ordered ``{"name", "entry_file", "script"}`` dicts, one per
            phase, in execution order.
        shared_files: Files every stage needs (filename → contents).

    Returns:
        A list of sequential stage specs in execution order.
    """
    specs = []
    for step in steps:
        input_files = dict(shared_files)
        input_files[step["entry_file"]] = step["script"]
        specs.append({
            "name": step["name"],
            "input_files": input_files,
            "entry_file": step["entry_file"],
        })
    return specs


class MDSimulationAgent(SimulationAgent):
    """
    MD-specific simulation agent.

    Adds MD concepts: ensemble, temperature, pressure, timestep,
    equilibration/production phases, force field integration, staged runs.

    The base class provides: skill loading, LLM helpers, validation,
    refinement, output cleaning.
    """

    SKILL_DOMAIN = "molecular_dynamics"
    EXTENSION_MAP = {
        "lammps": [".data", ".lmp"],
        "gromacs": [".gro", ".g96"],
        "openmm":  [".pdb", ".cif"],
    }
    TOOL_REGISTRY = _TOOL_REGISTRY

    def __init__(self, working_dir: str, **kwargs):
        super().__init__(working_dir=working_dir, **kwargs)
        self._last_plan: Optional[Dict[str, Any]] = None
        self._last_system_info: Optional[Dict[str, Any]] = None

    # ================================================================
    # SYSTEM ANALYSIS
    # ================================================================

    def analyze_system(self, structure_file: str) -> Dict[str, Any]:
        self.logger.info(f"Analyzing: {structure_file}")

        if self.tools_module and hasattr(self.tools_module, "parse_data_file"):
            try:
                info = self.tools_module.parse_data_file(structure_file)
                if info.get("atom_count", 0) > 0:
                    self.logger.info(
                        f"Analysis (tools): {info['atom_count']} atoms, "
                        f"{info.get('system_category', 'unknown')}"
                    )
                    return info
            except Exception as e:
                self.logger.warning(f"Tool analysis failed: {e}")

        # Generic ASE read first — handles the engine-neutral coordinate formats
        # structure generation emits (extxyz / xyz / cif / pdb) as well as POSCAR.
        try:
            from ase.io import read as _ase_read
            atoms = _ase_read(structure_file)
            if len(atoms) > 0:
                return self._info_from_atoms(atoms)
        except Exception:
            pass

        # LAMMPS data files (ASE can't auto-detect these by extension).
        try:
            from ase.io.lammpsdata import read_lammps_data
            atoms = read_lammps_data(structure_file, style="full", units="real")
            return self._info_from_atoms(atoms)
        except Exception:
            pass

        return self._llm_analyze_structure(structure_file)

    @staticmethod
    def _info_from_atoms(atoms) -> Dict[str, Any]:
        """Summarize an ase.Atoms into the system-info dict (engine-neutral)."""
        ec: Dict[str, int] = {}
        for s in atoms.get_chemical_symbols():
            ec[s] = ec.get(s, 0) + 1
        return {
            "atom_count": len(atoms),
            "elements": sorted(ec.keys()),
            "element_counts": ec,
            "box_dimensions": atoms.get_cell().diagonal().tolist(),
            "has_water": (
                "O" in ec and "H" in ec
                and ec.get("H", 0) >= 2 * ec.get("O", 0)
            ),
            "has_ions": any(x in ec for x in ["Na", "Cl", "K", "Ca", "Mg"]),
            "has_organic": "C" in ec,
            "system_category": "unknown",
        }

    def _llm_analyze_structure(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            header = f.read(5000)
        ctx = self._get_skill_context(section="analysis")
        prompt = (
            "Analyze this structure file.\n\n"
            f"{ctx}\n\n"
            "FILE (first 5000 chars):\n"
            f"{header}\n\n"
            "Return JSON:\n"
            '{"atom_count": int, "elements": [...], "element_counts": {},'
            ' "box_dimensions": [x,y,z], "has_water": bool, "has_ions": bool,'
            ' "has_organic": bool, "has_metal": bool, "system_category": str}'
        )
        try:
            return self._generate_json(prompt)
        except Exception:
            return {
                "atom_count": 0,
                "elements": [],
                "element_counts": {},
                "system_category": "unknown",
            }

    # ================================================================
    # PLANNING
    # ================================================================

    def plan_simulation(
        self,
        research_goal: str,
        system_info: Dict[str, Any],
        temperature: float = 300.0,
        pressure: float = 1.0,
        required_observables: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self.logger.info(f"Planning: {research_goal}")

        elements_str = ", ".join(
            f"{e}: {c}"
            for e, c in system_info.get("element_counts", {}).items()
        )
        planning = self._get_skill_context(section="planning")

        # Target observables are a first-class co-input: the run's length and
        # what it logs/dumps (and how often) are chosen to satisfy them, not
        # guessed from the goal prose alone.
        from .contradictions import format_requirements_for_prompt
        obs_lines = format_requirements_for_prompt(required_observables or [])
        obs_section = (
            "TARGET OBSERVABLES — the run MUST be configured so each is\n"
            "recoverable from its output. Choose the ensemble, production_time,\n"
            "and especially what to log/dump and how often so every observable\n"
            "below is computable, and reflect them in required_outputs. Do not\n"
            "under-sample a requested observable.\n"
            f"{obs_lines}\n\n"
        ) if obs_lines else ""

        prompt = (
            "Recommend MD simulation parameters for this research goal.\n\n"
            f'GOAL: "{research_goal}"\n\n'
            "SYSTEM:\n"
            f"- Elements: {elements_str}\n"
            f"- Atoms: {system_info.get('atom_count', 0)}\n"
            f"- Category: {system_info.get('system_category', 'unknown')}\n"
            f"- Water: {'Yes' if system_info.get('has_water') else 'No'}\n"
            f"- Ions: {'Yes' if system_info.get('has_ions') else 'No'}\n"
            f"- Metal: {'Yes' if system_info.get('has_metal') else 'No'}\n"
            f"- Has bonds: {'Yes' if system_info.get('has_bonds') else 'No'}\n"
            f"- Has vacuum: {'Yes' if system_info.get('has_vacuum') else 'No'}\n\n"
            f"{obs_section}"
            f"{planning}\n\n"
            "Use the tables above to select the correct unit system, timestep, and\n"
            "damping constants for this system type.\n\n"
            "If the goal calls for a parameter sweep — several independent runs that\n"
            "vary one quantity (e.g. a set of temperatures, pressures, strain rates,\n"
            "or restraint positions) — set requires_multiple_simulations true, set\n"
            "variable_parameter to that quantity's name, and set variable_values to\n"
            "the list of values. Otherwise leave them false/null.\n\n"
            "Return JSON:\n"
            "{\n"
            '    "simulation_technique": "standard_md",\n'
            '    "ensemble": "NPT",\n'
            '    "temperature": 300.0,\n'
            '    "pressure": 1.0,\n'
            '    "timestep": 2.0,\n'
            '    "equilibration_time": 0.5,\n'
            '    "production_time": 1.5,\n'
            '    "requires_multiple_simulations": false,\n'
            '    "number_of_simulations": 1,\n'
            '    "variable_parameter": null,\n'
            '    "variable_values": null,\n'
            '    "required_outputs": ["energy", "trajectory"],\n'
            '    "methodology_description": "brief explanation"\n'
            "}"
        )
        try:
            params = self._generate_json(prompt)
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            params = {}

        params.setdefault("simulation_technique", "standard_md")
        params.setdefault("ensemble", "NPT")
        params.setdefault("temperature", temperature)
        params.setdefault("pressure", pressure)
        params.setdefault("timestep", 2.0)
        params.setdefault("equilibration_time", 0.5)
        params.setdefault("production_time", 1.5)
        params.setdefault("requires_multiple_simulations", False)
        params.setdefault("number_of_simulations", 1)
        params.setdefault("variable_parameter", None)
        params.setdefault("variable_values", None)
        params.setdefault("required_outputs", ["energy", "trajectory"])

        for k, v in kwargs.items():
            params[k] = v

        return params

    # ================================================================
    # GENERATION
    # ================================================================

    def generate_simulation(
        self,
        structure_file: str,
        research_goal: str,
        system_description: Optional[str] = None,
        temperature: float = 300.0,
        pressure: Optional[float] = 1.0,
        force_field_files: Optional[Dict[str, str]] = None,
        potential: Optional[DeployedPotential] = None,
        runner: str = "lammps",
        task: str = "md",
        required_observables: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Multi-agent collaboration path: a potential-producing agent
        # (today: MLIPAgent) hands us a DeployedPotential and we own the
        # run generation. The MD agent is the place that knows how to
        # *run* a simulation with any potential — classical FF or MLIP.
        if potential is not None:
            return self._run_with_potential(
                structure_file=structure_file,
                research_goal=research_goal,
                potential=potential,
                runner=runner,
                task=task,
                temperature=temperature,
                pressure=pressure,
                **kwargs,
            )

        # Select the engine skill by runner name first — the runner names the
        # engine directly, so skill selection must not hinge on the structure
        # file's extension (an extensionless POSCAR from structure generation
        # would otherwise match no skill and degrade generation). Extension-
        # based detection stays as the fallback. "ase" is the universal runner
        # and carries no skill bundle.
        if runner and runner != "ase" and runner in self._available_skills:
            self._load_skill(runner)
        self._auto_select_skill(structure_file)

        system_info = self.analyze_system(structure_file)
        self._last_system_info = system_info

        if not system_description:
            system_description = self._describe_system(system_info)

        plan = self.plan_simulation(
            research_goal=research_goal,
            system_info=system_info,
            temperature=temperature,
            pressure=pressure,
            required_observables=required_observables,
            **kwargs,
        )
        self._last_plan = plan

        script = self._generate_md_input(
            structure_file,
            research_goal,
            system_description,
            system_info,
            plan,
            required_observables=required_observables,
        )

        # Deck-gen boundary: the analysis's declared species-resolved selections
        # must be distinctly expressible. A resolvable collision (two selections
        # sharing an atom type) splits the type and regenerates the deck; a no-op
        # otherwise.
        script, system_info = self._enforce_analysis_realizability(
            script, structure_file, research_goal, system_description,
            system_info, plan,
        )
        self._last_system_info = system_info

        if force_field_files:
            script = self._integrate_force_fields(script, force_field_files)

        script = self._clean_and_fix(script, plan)

        # Parallel-sweep campaign: the script was authored with the sweep
        # placeholder, so expand it into one independent run per value and
        # return a normalized fan-out campaign instead of a single script.
        sweep = self._sweep_spec(plan)
        if sweep:
            return self._finalize_campaign(
                script, sweep, structure_file, force_field_files,
                research_goal, system_description, system_info, plan,
            )

        script_path = self.working_dir / "run.lammps"
        script_path.write_text(script)

        validation = self._validate(str(script_path), system_info, plan)

        if not validation.get("valid", True) and validation.get("errors"):
            self.logger.warning("Validation failed, fixing...")
            script = self._attempt_fix(script, validation["errors"], plan)
            script_path.write_text(script)
            validation = self._validate(str(script_path), system_info, plan)

        readme = self._generate_readme(
            research_goal,
            system_description,
            system_info,
            plan,
            str(script_path),
        )

        return {
            "script_path": str(script_path),
            "readme_path": readme,
            "data_path": structure_file,
            "system_info": system_info,
            "simulation_parameters": plan,
            "validation": validation,
            "skill_used": self.skill_name,
        }

    # ================================================================
    # POTENTIAL-DRIVEN RUNS  (multi-agent collaboration entry point)
    # ================================================================

    def _run_with_potential(
        self,
        structure_file: str,
        research_goal: str,
        potential: DeployedPotential,
        runner: str,
        task: str,
        temperature: float,
        pressure: Optional[float],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a run for a pre-deployed interatomic potential.

        A potential-producing agent (today: MLIPAgent) deploys the
        potential and hands the DeployedPotential descriptor here; the
        MD agent owns the run generation. This keeps both agents
        general — the MLIP agent doesn't reimplement MD orchestration,
        and the MD agent doesn't care whether the potential is an MLIP
        or (eventually) a classical force field.

        Runner dispatch is extensible by construction:
          - ``"ase"`` is the **universal** runner, built into the MD
            agent — every DeployedPotential has an ASE calculator, so
            this always works. Supports ``task="md"`` and
            ``task="relax"``.
          - any other runner dispatches to that engine's tools module
            via ``TOOL_REGISTRY[runner].run_with_potential(...)``.
            There is no per-engine branching here: adding GROMACS is
            dropping a ``gromacs`` tools module with a
            ``run_with_potential`` function — this method never
            changes. The engine raises ``NotImplementedError`` if it
            has no integration for the potential's backend (e.g. LAMMPS
            for CHGNet), and the caller is expected to use ``"ase"``.

        kwargs carries the remaining simulation params unpacked by the
        caller (timestep, n_steps, output_interval, device, fmax, ...).
        """
        sim = kwargs
        # pressure: None -> NVT, a value -> NPT. An explicit
        # pressure in sim overrides the positional arg.
        pressure = sim.get("pressure", pressure)

        if runner == "ase":
            from ._ase_runner import generate_ase_script
            run_path = generate_ase_script(
                potential=potential,
                working_dir=str(self.working_dir),
                structure_file=structure_file,
                task=task,
                timestep=sim.get("timestep", 1.0),
                temperature=temperature,
                pressure=pressure,
                n_steps=sim.get("n_steps", 1000),
                output_interval=sim.get("output_interval", 50),
                device=sim.get("device", "cuda"),
                fmax=sim.get("fmax", 0.02),
            )
            self.logger.info(
                f"Generated ASE {task} script for {potential.backend}: "
                f"{run_path}"
            )
        else:
            engine = self.TOOL_REGISTRY.get(runner)
            if engine is None:
                raise ValueError(
                    f"no tools module for runner {runner!r}; loaded "
                    f"engines: {sorted(self.TOOL_REGISTRY)}. The "
                    f"universal 'ase' runner is always available."
                )
            if not hasattr(engine, "run_with_potential"):
                raise NotImplementedError(
                    f"the {runner!r} engine's tools module has no "
                    f"run_with_potential() — it cannot run a deployed "
                    f"potential. Use runner='ase'."
                )
            run_path = engine.run_with_potential(
                potential,
                structure_file=structure_file,
                working_dir=str(self.working_dir),
                task=task,
                timestep=sim.get("timestep", 0.5),
                temperature=temperature,
                pressure=pressure,
                n_steps=sim.get("n_steps", 100000),
            )
            self.logger.info(
                f"Generated {runner} input for {potential.backend}: "
                f"{run_path}"
            )

        return {
            "run_path": run_path,
            "runner": runner,
            "task": task,
            "potential_backend": potential.backend,
            "potential_model": potential.model_name,
            "structure_file": structure_file,
            "research_goal": research_goal,
            "notes": potential.notes,
        }

    def _generate_md_input(
        self, structure_file, goal, desc, info, plan, resolved_selections=None,
        required_observables=None,
    ) -> str:
        data_filename = os.path.basename(structure_file)

        type_info = ""
        if self.tools_module and hasattr(self.tools_module, "format_type_info"):
            try:
                type_info = self.tools_module.format_type_info(structure_file)
            except Exception:
                pass

        # When a boundary check split shared atom types so species-resolved
        # selections became distinct, tell the agent which types now carry which
        # selection so it writes the corresponding groups/computes.
        resolved_block = ""
        if resolved_selections:
            rows = "\n".join(
                f"- {sel}: atom type(s) {types}"
                for sel, types in resolved_selections.items()
            )
            resolved_block = (
                "\n## Species-resolved selections (distinct atom types)\n"
                "These selections were given distinct atom types so they can be "
                "analyzed separately; use exactly these types in the matching "
                "groups/computes:\n"
                f"{rows}\n"
            )

        ts = plan.get("timestep", 2.0)
        equil_steps = int(
            (plan.get("equilibration_time", 0.5) * 1e6) / ts
        )
        prod_steps = int(
            (plan.get("production_time", 1.5) * 1e6) / ts
        )

        elements_str = ", ".join(
            f"{e}: {c}"
            for e, c in info.get("element_counts", {}).items()
        )

        implementation = self._get_skill_context(section="implementation")
        planning = self._get_skill_context(section="planning")
        validation_rules = self._get_skill_context(section="validation")
        has_skill = bool(implementation)

        multi_block = ""
        if (
            plan.get("requires_multiple_simulations")
            and plan.get("number_of_simulations", 1) > 1
        ):
            multi_block = self._build_multi_sim_block(plan)

        # Declared target observables → the deck must emit the computes / dumps /
        # thermo cadence that make each one recoverable.
        from .contradictions import format_requirements_for_prompt
        obs_lines = format_requirements_for_prompt(required_observables or [])
        obs_block_deck = (
            "## Target observables (the deck MUST make each computable — write "
            "the matching computes/dumps and thermo/dump cadence)\n"
            f"{obs_lines}\n\n"
        ) if obs_lines else ""

        if has_skill:
            prompt = (
                "You are an expert molecular dynamics simulation engineer.\n"
                "Generate a complete, runnable input file.\n\n"
                "## Goal\n"
                f"{goal}\n\n"
                "## System\n"
                f"{desc}\n"
                f"- Elements: {elements_str}\n"
                f"- Atoms: {info.get('atom_count', 0)}\n"
                f"- Category: {info.get('system_category', 'unknown')}\n"
                f"- Box: {info.get('box_dimensions', [])}\n"
                f"- Vacuum: {'Yes -- slab' if info.get('has_vacuum') else 'No'}\n\n"
                f"{type_info}\n"
                f"{resolved_block}\n"
                "## Plan\n"
                f"- Technique: {plan.get('simulation_technique')}\n"
                f"- Ensemble: {plan.get('ensemble')}\n"
                f"- Temperature: {plan.get('temperature')} K\n"
                f"- Pressure: {plan.get('pressure')} atm\n"
                f"- Timestep: {ts}\n"
                f"- Equilibration: {plan.get('equilibration_time')} ns ({equil_steps} steps)\n"
                f"- Production: {plan.get('production_time')} ns ({prod_steps} steps)\n"
                f"- Outputs: {plan.get('required_outputs', [])}\n"
                f"{multi_block}\n\n"
                f"{obs_block_deck}"
                "## Implementation Templates\n"
                f"{implementation}\n\n"
                "## Parameter Guidelines\n"
                f"{planning}\n\n"
                "## Validation Rules (must satisfy)\n"
                f"{validation_rules}\n\n"
                "RULES:\n"
                "1. Match unit system and pair_style to system category per the tables.\n"
                "2. Use literal values -- no unresolved template variables.\n"
                f"3. Structure file: {data_filename}\n"
                "4. If data file has embedded Pair Coeffs, do not add pair_coeff commands.\n"
                "5. For external potentials, add pair_coeff referencing the file with a comment.\n\n"
                "Return ONLY the input file. No markdown."
            )
        else:
            prompt = (
                "You are an MD simulation expert. No engine-specific skill loaded.\n\n"
                f"GOAL: {goal}\n"
                f"SYSTEM: {desc} ({info.get('atom_count', 0)} atoms)\n"
                f"PLAN: {plan.get('ensemble')} at {plan.get('temperature')} K, dt={ts},\n"
                f"      equil {equil_steps} steps, prod {prod_steps} steps\n"
                f"FILE: {data_filename}\n\n"
                "# GENERATED WITHOUT SKILL -- VERIFY BEFORE RUNNING\n\n"
                "Return ONLY the input file. No markdown."
            )

        return self._generate_text(prompt)

    def _declare_analysis_selections(self, research_goal, structure_file):
        """Ask the agent which species-resolved atom selections its analysis must
        resolve separately, given the goal and the components manifest.

        This is the declared contract for the ``selection_realizable`` boundary
        check: the agent states its intent (engine-neutral selection strings,
        ``"<species>"`` or ``"<species>:<Element>"``), which the contradiction
        framework then verifies and, if needed, resolves. Returns ``[]`` when
        there is no manifest, fewer than two species, or the analysis needs no
        species-resolved selection.
        """
        import json
        comp_path = os.path.join(
            os.path.dirname(structure_file) or ".", "components.json")
        if not os.path.isfile(comp_path):
            return []
        try:
            comps = json.load(open(comp_path)).get("components", [])
        except Exception:
            return []
        if len(comps) < 2:
            return []
        species = "\n".join(
            f"- {c.get('name')}  (SMILES {c.get('smiles')})" for c in comps)
        prompt = (
            "A downstream analysis of this simulation may need to distinguish "
            "atoms of the same element that belong to different molecular "
            "species.\n\n"
            f"Research goal:\n{research_goal}\n\n"
            f"Molecular species present:\n{species}\n\n"
            "List the atom selections the analysis must be able to resolve "
            "SEPARATELY, each as '<species>' or '<species>:<Element>' using the "
            "exact species names above. Include a selection only if the goal's "
            "analysis genuinely needs it on its own; return an empty list if no "
            "species-resolved selection is needed.\n"
            'Return JSON: {"selections": ["...", "..."]}'
        )
        try:
            out = self._generate_json(prompt) or {}
            return [s for s in (out.get("selections") or []) if isinstance(s, str)]
        except Exception:
            return []

    def _enforce_analysis_realizability(
        self, script, structure_file, research_goal, desc, info, plan
    ):
        """Deck-gen boundary check: the analysis's declared species-resolved
        selections must be distinctly expressible in the generated input.

        Runs the engine-neutral ``selection_realizable`` check; on a resolvable
        collision (two required selections share an atom type) it applies the
        engine's ``split_shared_types`` resolution, rewrites the data file, and
        regenerates the deck against the now-distinct types. Fail-open — any
        error leaves the deck and data file unchanged. Returns
        ``(script, system_info)``.
        """
        try:
            from .contradictions import Requirement, check_requirements
            if self.tools_module is None or not hasattr(
                self.tools_module, "split_shared_types"
            ):
                return script, info
            selections = self._declare_analysis_selections(
                research_goal, structure_file)
            if len(selections) < 2:
                return script, info
            comp_path = os.path.join(
                os.path.dirname(structure_file) or ".", "components.json")
            req = Requirement(
                "analysis selections", "selection_realizable",
                params={"selections": selections})
            arts = {
                "data_file": structure_file, "components_json": comp_path,
                "engine_tools": self.tools_module,
            }
            cons = check_requirements(
                [req], arts,
                active_skills=[self.skill_name] if self.skill_name else [])
            if not cons:
                return script, info
            c = cons[0]
            if not c.resolvable:
                self.logger.warning(
                    "Analysis realizability: %s (no resolution)", c.message)
                return script, info
            res = self.tools_module.split_shared_types(**c.resolution["kwargs"])
            with open(structure_file, "w") as fh:
                fh.write(res["data_file_text"])
            self.logger.info(
                "Analysis realizability: split shared types so %s are distinct",
                list(res["type_map"]),
            )
            new_info = self.analyze_system(structure_file)
            new_script = self._generate_md_input(
                structure_file, research_goal, desc, new_info, plan,
                resolved_selections=res["type_map"])
            return new_script, new_info
        except Exception as e:
            self.logger.warning("Analysis realizability check skipped: %s", e)
            return script, info

    def _build_multi_sim_block(self, plan):
        n = plan.get("number_of_simulations", 1)
        tech = plan.get("simulation_technique", "")
        var = plan.get("variable_parameter", "")
        vals = plan.get("variable_values", [])
        vals_str = ", ".join(str(v) for v in vals[:5])
        if len(vals) > 5:
            vals_str += f"... ({len(vals)} total)"
        block = (
            "\n## Multi-Simulation (parallel sweep)\n"
            f"- Technique: {tech}\n"
            f"- Runs: {n}\n"
            f"- Variable: {var}\n"
            f"- Values: {vals_str}\n"
        )
        # Author the sweep as ONE parameterized script: the swept quantity is
        # marked with the engine's placeholder and expanded into one run per
        # value downstream. The placeholder token belongs to the engine skill,
        # so the prompt and the expander agree on it without the agent hardcoding
        # a token.
        placeholder = (getattr(self.tools_module, "SWEEP_PLACEHOLDER", None)
                       if self.tools_module else None)
        if placeholder and var:
            block += (
                f"- Write ONE script for the whole sweep: use the literal token "
                f"{placeholder} in place of every {var} value the script sets or "
                f"references, and use literal values for everything else. Each "
                f"run is produced by substituting one value for {placeholder}, so "
                f"the placeholder must appear wherever a {var} value would.\n"
            )
        return block

    # ================================================================
    # PARALLEL-SWEEP CAMPAIGN
    # ================================================================

    def _sweep_spec(self, plan):
        """Return ``{var_name, values}`` if the plan calls for a parallel sweep.

        A sweep needs at least two values and an engine that knows how to expand
        one (``expand_parameter_sweep``). Otherwise ``None`` — the caller emits
        a single simulation.
        """
        if not (plan.get("requires_multiple_simulations")
                and plan.get("number_of_simulations", 1) > 1):
            return None
        values = plan.get("variable_values") or []
        if len(values) < 2:
            return None
        if not (self.tools_module
                and hasattr(self.tools_module, "expand_parameter_sweep")):
            return None
        return {"var_name": plan.get("variable_parameter") or "parameter",
                "values": values}

    def _finalize_campaign(self, base_script, sweep, structure_file,
                           force_field_files, goal, desc, info, plan):
        """Expand a parameterized base script into a normalized fan-out campaign.

        Each member is a self-contained run: the same structure data file and
        force-field files plus its own substituted script, so members execute
        independently in isolated directories. The expansion itself is the
        engine skill's (``expand_parameter_sweep``); this method only assembles
        the engine-neutral ``stages`` structure the refinement loop consumes.
        """
        entry = "run.lammps"
        shared = self._campaign_shared_files(structure_file, force_field_files)
        members = self.tools_module.expand_parameter_sweep(
            base_script, sweep["var_name"], sweep["values"])
        stages = _assemble_fanout_stage(members, entry, shared)

        # Validate a representative member (the placeholder is already resolved).
        rep_files = stages[0]["members"][0]["input_files"]
        rep_path = self.working_dir / entry
        rep_path.write_text(rep_files[entry])
        validation = self._validate(str(rep_path), info, plan)
        readme = self._generate_readme(goal, desc, info, plan, str(rep_path))

        return {
            "script_path": str(rep_path),
            "readme_path": readme,
            "data_path": structure_file,
            "system_info": info,
            "simulation_parameters": plan,
            "validation": validation,
            "skill_used": self.skill_name,
            "stages": stages,
            "input_files": rep_files,
            "entry_file": entry,
            "is_campaign": True,
            "campaign_kind": "parameter_sweep",
        }

    def _campaign_shared_files(self, structure_file, force_field_files):
        """Read the files every member needs (structure data + force fields)."""
        shared = {}
        try:
            shared[Path(structure_file).name] = Path(structure_file).read_text(
                errors="replace")
        except OSError:
            self.logger.warning("Could not read structure file for campaign")
        for name, path in (force_field_files or {}).items():
            try:
                shared[name] = Path(path).read_text(errors="replace")
            except OSError:
                self.logger.warning("Could not read force-field file %s", path)
        return shared

    # ================================================================
    # FORCE FIELD INTEGRATION
    # ================================================================

    def _integrate_force_fields(self, script, ff_files):
        if self.tools_module and hasattr(
            self.tools_module, "integrate_force_field_files"
        ):
            return self.tools_module.integrate_force_field_files(
                script, ff_files, str(self.working_dir)
            )
        self.logger.warning("No FF integration tool")
        header = "\n".join(
            f"# FF: {n} = {p}" for n, p in ff_files.items()
        )
        return header + "\n\n" + script

    # ================================================================
    # CLEANING
    # ================================================================

    def _clean_and_fix(self, script, plan):
        script = self._clean_output(script)
        if self.tools_module and hasattr(
            self.tools_module, "substitute_variables"
        ):
            script = self.tools_module.substitute_variables(
                script,
                temperature=plan.get("temperature", 300.0),
                pressure=plan.get("pressure", 1.0),
                timestep=plan.get("timestep", 2.0),
            )
        return script

    # ================================================================
    # STAGED SIMULATION
    # ================================================================

    def generate_staged_simulation(
        self, structure_file, research_goal, **kw
    ):
        result = self.generate_simulation(
            structure_file=structure_file,
            research_goal=research_goal,
            **kw,
        )
        full_script = Path(result["script_path"]).read_text()
        plan = result["simulation_parameters"]
        impl = self._get_skill_context(section="implementation")

        prompt = (
            "Split this simulation into 2-4 checkpointed stages.\n\n"
            "SCRIPT:\n"
            f"{full_script}\n\n"
            f"{impl}\n\n"
            "Each stage: complete, standalone, runnable. First reads data file,\n"
            "later stages read restart. All include force field commands.\n"
            "Use literal values. Write restart at end of each stage.\n\n"
            'Return JSON: {"equilibration": "script...", "production": "script..."}'
        )
        try:
            stages = self._generate_json(prompt)
        except Exception:
            stages = {"production": full_script}

        stage_scripts = {}
        steps = []
        for name, content in stages.items():
            if not isinstance(content, str):
                continue
            content = self._clean_and_fix(content, plan)
            entry = f"run_{name}.lammps"
            path = self.working_dir / entry
            path.write_text(content)
            stage_scripts[name] = str(path)
            steps.append({"name": name, "entry_file": entry, "script": content})

        shared = self._campaign_shared_files(
            structure_file, kw.get("force_field_files"))
        stage_specs = _assemble_sequential_stages(steps, shared)

        result.update(
            staged_scripts=stage_scripts,
            stages=stage_specs,
            is_staged=True,
            is_campaign=True,
            campaign_kind="staged",
        )
        # Representative phase for back-compat consumers that read a single
        # input set (the pipeline's input_files normalization, etc.).
        if stage_specs:
            result["input_files"] = stage_specs[0]["input_files"]
            result["entry_file"] = stage_specs[0]["entry_file"]
        return result

    # ================================================================
    # SYSTEM DESCRIPTION
    # ================================================================

    def _describe_system(self, info):
        cat = info.get("system_category", "unknown")
        parts = []
        non_molecular = {"O", "H", "C", "N", "S", "P", "F", "Cl"}

        if cat == "metal":
            metals = [
                e for e in info.get("elements", [])
                if e not in non_molecular
            ]
            parts.append(" ".join(metals) + " metal")
        elif cat == "semiconductor":
            parts.append(
                " ".join(info.get("elements", [])) + " semiconductor"
            )
        elif cat == "oxide":
            parts.append("metal oxide")
        elif cat == "ionic":
            parts.append("ionic crystal")
        else:
            if info.get("has_water"):
                parts.append("water")
            if info.get("has_ions"):
                ions = [
                    e for e in info.get("elements", [])
                    if e in ["Na", "K", "Cl", "Ca", "Mg"]
                ]
                if ions:
                    parts.append("+".join(ions) + " ions")
                else:
                    parts.append("ions")
            if info.get("has_organic"):
                parts.append("organic molecules")

        if not parts:
            parts.append("molecular system")

        desc = " with ".join(parts)

        if info.get("has_vacuum"):
            axis = info.get("vacuum_axis", "z")
            desc += f" (slab, vacuum {axis})"

        return f"{desc} ({info.get('atom_count', 0)} atoms)"

    # ================================================================
    # README
    # ================================================================

    def _generate_readme(self, goal, desc, info, plan, script_path):
        p = self.working_dir / "README.md"
        with open(p, "w") as f:
            f.write(f"# MD Simulation: {desc}\n\n")
            f.write(f"**Skill**: {self.skill_name or 'none'}\n\n")
            f.write(f"## Goal\n{goal}\n\n")
            f.write("## System\n")
            for el, c in info.get("element_counts", {}).items():
                f.write(f"- {el}: {c}\n")
            f.write(
                f"- Category: {info.get('system_category', 'unknown')}\n\n"
            )
            f.write("## Parameters\n")
            param_keys = [
                "ensemble",
                "temperature",
                "pressure",
                "timestep",
                "equilibration_time",
                "production_time",
            ]
            for k in param_keys:
                f.write(f"- {k}: {plan.get(k)}\n")
            f.write(f"\n## Run\n")
            f.write(f"cd {self.working_dir}\n")
            f.write(f"lmp -in {os.path.basename(script_path)}\n")
        return str(p)
