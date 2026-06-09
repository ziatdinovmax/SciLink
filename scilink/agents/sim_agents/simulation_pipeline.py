"""Scale-agnostic deterministic simulation pipeline.

A single one-shot pipeline that turns a natural-language request into a
validated structure plus ready-to-run inputs, for any simulation scale.
One scale-agnostic entry point (``run_complete_workflow``) serves every
engine; the scale selects the foundation agent and the engine selects the
skill bundle.

The pipeline is deterministic — it runs a fixed step sequence rather than
letting an orchestration LLM choose steps — which is what makes its output
reproducible for benchmarking. The chat / LLM-driven path lives on
``SimulationOrchestratorAgent`` (``chat`` / ``run_task``); this is the
headless sequence both that orchestrator and analyze-mode call.

Steps:
    1. Structure   — StructurePipeline (scale-agnostic) builds and
                     validates the atomic structure.
    2. Inputs      — the routed scale's foundation agent generates inputs,
                     returning a normalized ``input_files`` map (engine
                     selected by ``software``; an optional named ``method``
                     selects a deterministic generation backend registered
                     in the engine's skill bundle).
    3. Validation  — InputValidator reviews the generated inputs (skill
                     guidance + deterministic syntax check + literature
                     grounding when a FutureHouse key is present).

Adding a new scale (e.g. molecular DFT) is a new foundation agent plus a
skill bundle and one dispatch branch in ``_generate_inputs`` — no new
orchestrator class, and no hardcoded engine filenames anywhere.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Default engine per scale, used when the caller does not name one. Each
# scale's foundation agent resolves the engine to a skill bundle.
_DEFAULT_ENGINE = {
    "periodic_dft": "vasp",
    "molecular_qc": "nwchem",
    "molecular_dynamics": "lammps",
}


def _generate_inputs(
    *,
    scale: str,
    software: str,
    method: str,
    structure_file: str,
    request: str,
    output_dir: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model_name: str,
    force_field_files: Optional[Dict[str, str]] = None,
    staged: bool = False,
) -> Dict[str, Any]:
    """Generate inputs for ``scale``, returning a normalized result.

    Every branch returns a result dict carrying an ``input_files`` mapping
    (filename → contents), so downstream steps never guess engine
    filenames. When ``method`` names a deterministic backend (anything
    other than ``"llm"``), inputs come from a ``generate_inputs_<method>``
    tool in the engine's skill bundle; otherwise the routed scale's
    foundation agent produces them with its default LLM path.

    Args:
        scale: Simulation scale (e.g. ``"periodic_dft"``).
        software: Engine name within the scale (e.g. ``"vasp"``).
        method: ``"llm"`` for the agent's baseline generation, or a named
            backend resolved from the skill bundle.
        structure_file: Path to the built structure.
        request: The scientific request driving parameter choices.
        output_dir: Where inputs should be written.
        api_key, base_url, model_name: LLM credentials forwarded to the
            foundation agent.

    Returns:
        A dict with at least ``status`` and, on success, an ``input_files``
        mapping (filename → contents).

    Raises:
        ValueError: If the scale is not supported by the pipeline.
    """
    # Named deterministic backend: a skill-bundle generation tool. The tool
    # is responsible for returning a normalized input_files map.
    if method and method != "llm":
        from ...skills._shared._registry import get_tool_function
        gen = get_tool_function(f"generate_inputs_{method}", active_skills=[software])
        return gen(structure_file=structure_file, request=request,
                   output_dir=output_dir)

    if scale == "periodic_dft":
        from .periodic_dft_agent import PeriodicDFTAgent
        agent = PeriodicDFTAgent(
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        result = agent.generate_inputs(
            structure_file=structure_file, request=request, software=software,
        )
        # PeriodicDFTAgent already returns input_files as {filename: contents}.
        if result.get("status") == "success":
            agent.save_inputs(result, output_dir)
        return result

    if scale == "molecular_qc":
        from .molecular_qc_agent import MolecularQCAgent
        agent = MolecularQCAgent(
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        result = agent.generate_inputs(
            structure_file=structure_file, request=request, software=software,
        )
        # MolecularQCAgent returns input_files as {filename: contents} and,
        # for single-deck engines, an entry_file the refinement loop runs.
        if result.get("status") == "success":
            agent.save_inputs(result, output_dir)
        return result

    if scale == "molecular_dynamics":
        from .md_simulation_agent import MDSimulationAgent
        agent = MDSimulationAgent(
            working_dir=output_dir,
            api_key=api_key, base_url=base_url, model_name=model_name,
        )
        # Staged generation emits an optimization → equilibration → production
        # chain as a normalized sequential campaign; one-shot generation emits a
        # single phase (or a parallel sweep when the plan calls for one). Both
        # return the same normalized result shape the pipeline consumes.
        gen = (agent.generate_staged_simulation if staged
               else agent.generate_simulation)
        result = gen(
            structure_file=structure_file, research_goal=request, runner=software,
            force_field_files=force_field_files,
        )
        # Normalize the MD agent's single script_path into the common
        # input_files map so the pipeline stays engine-neutral downstream,
        # and record the entry script so the refinement loop knows what to run.
        script_path = result.get("script_path")
        if "input_files" not in result and script_path and Path(script_path).exists():
            result["input_files"] = {
                Path(script_path).name: Path(script_path).read_text()
            }
        if script_path:
            result["entry_file"] = Path(script_path).name
        result.setdefault("status", "success")
        return result

    raise ValueError(
        f"Unsupported simulation scale: {scale!r}. "
        f"Supported: {sorted(_DEFAULT_ENGINE)}. Adding a scale means a new "
        "foundation agent + skill bundle and one branch here."
    )


def run_complete_workflow(
    user_request: str,
    *,
    scale: str = "periodic_dft",
    software: Optional[str] = None,
    method: str = "llm",
    structure_class: str = "crystal",
    output_dir: str = "simulation_workflow_output",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: str = "claude-opus-4-6",
    futurehouse_api_key: Optional[str] = None,
    mp_api_key: Optional[str] = None,
    max_refinement_cycles: int = 4,
    script_timeout: int = 300,
    validate: bool = True,
    executor: "Executor | None" = None,
    run_command: Optional[str] = None,
    autonomy: str = "autonomous",
    max_run_cycles: int = 3,
    structure_file: Optional[str] = None,
    force_field_files: Optional[Dict[str, str]] = None,
    staged: bool = False,
) -> Dict[str, Any]:
    """Run the full structure → inputs → validation pipeline for any scale.

    Args:
        user_request: Natural-language description of the calculation.
        scale: Simulation scale (``"periodic_dft"``, ``"molecular_dynamics"``,
            …). Selects the foundation agent.
        software: Engine within the scale (e.g. ``"vasp"``, ``"lammps"``).
            Defaults to the scale's conventional engine.
        method: Input-generation backend. ``"llm"`` (default) uses the
            foundation agent's generation; a named backend (e.g.
            ``"atomate2"``) resolves to a skill-bundle generation tool.
        structure_class: Structure-class hint forwarded to structure
            generation.
        output_dir: Directory for all generated files.
        api_key, base_url, model_name: LLM credentials.
        futurehouse_api_key: Optional FutureHouse key enabling
            literature-grounded validation.
        mp_api_key: Optional Materials Project key for structure lookups.
        max_refinement_cycles: Structure validator-guided refinement cap.
        script_timeout: Timeout for executing generated structure scripts.
        validate: When True, run the pre-run InputValidator on the generated
            inputs (skipped for non-LLM methods, which are expert-defined).
        executor: Optional execution backend. When provided, the workflow runs
            the generated inputs and refines them to convergence via the
            engine-neutral refinement loop. ``LocalExecutor`` runs a local
            subprocess; ``ClusterExecutor`` (or ``ClusterExecutor.connect(...)``)
            submits to an HPC scheduler — the loop drives either through the same
            ``Executor`` contract. When ``None`` (the default, used for DFT), the
            workflow stops after generation + validation and the user runs the
            calculation externally.
        run_command: Command template the executor runs, with ``{script}``
            filled from each phase's entry file (e.g. ``"lmp -in {script}"``).
            User/config — required when ``executor`` is provided. The engine
            binary lives here, never in this module.
        autonomy: Autonomy level for the refinement loop (``"co-pilot"`` /
            ``"autopilot"`` / ``"autonomous"``); selects the built-in policy.
        max_run_cycles: Maximum run → assess → fix cycles per phase.
        structure_file: Optional path to an already-built structure. When
            provided, structure generation is skipped and this file is used
            directly — for callers that already have a structure and only want
            input generation + (optional) execution.
        force_field_files: Optional mapping of force-field filename to contents,
            forwarded to MD input generation.
        staged: When True, MD generation emits a multi-phase (optimization →
            equilibration → production) sequential campaign instead of a single
            run, so the refinement loop runs the per-phase loop over a restart-
            chained sequence. MD only; ignored by other scales.

    Returns:
        A workflow-result dict with ``final_status``, ``scale``, ``engine``,
        ``steps_completed``, ``output_directory``, and the per-step results
        (``structure_generation``, ``input_generation``, ``input_validation``).
    """
    software = software or _DEFAULT_ENGINE.get(scale)
    os.makedirs(output_dir, exist_ok=True)
    result: Dict[str, Any] = {
        "user_request": user_request,
        "scale": scale,
        "engine": software,
        "steps_completed": [],
        "final_status": "started",
        "output_directory": output_dir,
    }

    # ── Step 1: structure generation + validation (scale-agnostic) ──
    # Skipped when the caller supplies an already-built structure.
    if structure_file is not None:
        structure_path = structure_file
        result["structure_generation"] = {
            "status": "skipped",
            "message": "caller-supplied structure",
            "final_structure_path": structure_file,
        }
    else:
        from .structure_pipeline import StructurePipeline
        structure = StructurePipeline(
            api_key=api_key, base_url=base_url, mp_api_key=mp_api_key,
            generator_model=model_name, validator_model=model_name,
            output_dir=output_dir, max_refinement_cycles=max_refinement_cycles,
            script_timeout=script_timeout,
        )
        # Reuse the structure pipeline's resolved credentials downstream.
        api_key = structure.api_key
        base_url = structure.base_url

        structure_result = structure.generate_and_validate(
            user_request, structure_class=structure_class,
        )
        result["structure_generation"] = structure_result
        if structure_result.get("status") != "success":
            result["final_status"] = "failed_structure_generation"
            return result
        result["steps_completed"].append("structure_generation")
        structure_path = structure_result["final_structure_path"]

    # ── Step 2: input generation (routed to the scale's foundation agent) ──
    try:
        gen_result = _generate_inputs(
            scale=scale, software=software, method=method,
            structure_file=structure_path, request=user_request,
            output_dir=output_dir, api_key=api_key, base_url=base_url,
            model_name=model_name, force_field_files=force_field_files,
            staged=staged,
        )
    except Exception as e:
        result["final_status"] = "failed_input_generation"
        result["input_generation"] = {"status": "error", "message": str(e)}
        return result
    result["input_generation"] = gen_result
    if gen_result.get("status") not in (None, "success"):
        result["final_status"] = "failed_input_generation"
        return result
    result["steps_completed"].append("input_generation")

    # ── Step 3: pre-run input validation (engine-neutral critic) ──
    # Skipped for named (deterministic, expert-defined) backends and when
    # the caller opts out.
    if validate and method == "llm":
        input_files = _collect_input_files(gen_result)
        if input_files:
            from .critics import InputValidator
            validator = InputValidator(
                api_key=api_key, base_url=base_url, model_name=model_name,
                futurehouse_api_key=futurehouse_api_key,
            )
            result["input_validation"] = validator.validate(
                input_files=input_files, system_description=user_request,
                skill=software, domain=scale,
            )
            result["steps_completed"].append("input_validation")
    else:
        reason = ("non-LLM method uses expert-defined inputs"
                  if method != "llm" else "validation disabled by caller")
        result["input_validation"] = {"status": "skipped", "message": reason}

    # ── Step 4: supervised execution + refinement (only when an executor is
    # supplied; DFT's default executor=None stops here and runs externally) ──
    if executor is None:
        result["final_status"] = "success"
        return result

    if not run_command:
        result["refinement"] = {
            "status": "skipped",
            "message": "executor provided without a run_command template",
        }
        result["final_status"] = "success"
        return result

    from .refinement import RefinementContext, policy_for, run_campaign
    from .critics import RunCritic

    stages = _collect_stages(gen_result, output_dir, run_command)
    ctx = RefinementContext(
        research_goal=user_request, scale=scale, engine=software,
        skill=software, domain=scale, autonomy=autonomy,
        max_cycles=max_run_cycles,
    )
    run_critic = RunCritic(
        api_key=api_key, base_url=base_url, model_name=model_name,
    )
    refinement = run_campaign(
        stages, executor, run_critic, policy_for(autonomy), ctx,
        pre_run_verdict=result.get("input_validation"),
    )
    result["refinement"] = refinement
    result["steps_completed"].append("refinement")
    result["final_status"] = (
        "success" if refinement.get("status") == "success"
        else f"refinement_{refinement.get('status', 'failed')}"
    )
    return result


def _collect_phases(
    gen_result: Dict[str, Any], run_dir: str, run_command_template: str
) -> list:
    """Build refinement ``Phase`` objects from a generation result.

    Reads only the normalized phase fields a foundation agent emits
    (``phases``, or an ``entry_file`` + ``input_files`` for single-phase
    engines), so no engine-specific keys appear here. The run command is the
    caller-provided template with ``{script}`` filled from each phase's entry
    file, so the engine binary is never assembled in this module.

    Args:
        gen_result: The input-generation result.
        run_dir: Directory the phases execute in (shared across phases so
            staged runs can read each other's restart files).
        run_command_template: Command template with an optional ``{script}``
            placeholder for the per-phase entry file.

    Returns:
        A list of ``Phase`` objects in execution order.
    """
    from .refinement import Phase

    phases_spec = gen_result.get("phases")
    if not phases_spec:
        entry = gen_result.get("entry_file")
        input_files = gen_result.get("input_files") or {}
        if entry is None and len(input_files) == 1:
            entry = next(iter(input_files))
        phases_spec = [{
            "name": "production",
            "input_files": input_files,
            "entry_file": entry,
        }]

    phases = []
    for spec in phases_spec:
        entry = spec.get("entry_file") or ""
        cmd = (
            run_command_template.format(script=entry)
            if "{script}" in run_command_template
            else run_command_template
        )
        phases.append(Phase(
            name=spec.get("name", "run"),
            input_files=spec.get("input_files") or {},
            run_command=cmd,
            run_dir=str(run_dir),
        ))
    return phases


def _collect_stages(
    gen_result: Dict[str, Any], run_dir: str, run_command_template: str
) -> list:
    """Build refinement ``Stage`` objects from a generation result.

    Reads only normalized, engine-neutral campaign fields. A generation result
    may carry a ``stages`` list describing a staged/parallel campaign; each
    entry is one of:

    * a **sequential step** — ``{name, input_files, entry_file}``. Steps share
      ``run_dir`` so restart files chain.
    * a **parallel fan-out** — ``{name, parallel: true, members: [...],
      min_success?}`` where each member is ``{name, input_files, entry_file}``.
      Members run in their own ``run_dir/<stage>/<member>`` directory.
    * a **combine** step — ``{name, kind: "combine", input_files, entry_file,
      run_command?}`` in ``run_dir/<stage>``; ``run_command`` may override the
      template (e.g. a Python post-processing script).

    When no ``stages`` field is present, the legacy single-chain shape is read
    via :func:`_collect_phases` and wrapped as one sequential stage, so older
    generation results behave exactly as before.

    Args:
        gen_result: The input-generation result.
        run_dir: Base directory the campaign executes in.
        run_command_template: Command template with an optional ``{script}``
            placeholder for a phase's entry file.

    Returns:
        A list of ``Stage`` objects in execution order.
    """
    import os

    from .refinement import Phase, Stage

    stages_spec = gen_result.get("stages")
    if not stages_spec:
        phases = _collect_phases(gen_result, run_dir, run_command_template)
        return [Stage(name="run", phases=phases, parallel=False)]

    def _command(entry: str, override) -> str:
        template = override or run_command_template
        if "{script}" in template:
            return template.format(script=entry or "")
        return template

    def _phase(spec: Dict[str, Any], rdir: str) -> "Phase":
        entry = spec.get("entry_file") or ""
        return Phase(
            name=spec.get("name", "run"),
            input_files=spec.get("input_files") or {},
            run_command=_command(entry, spec.get("run_command")),
            run_dir=str(rdir),
        )

    stages = []
    for spec in stages_spec:
        name = spec.get("name", "run")
        if spec.get("kind") == "combine":
            stages.append(Stage(
                name=name, kind="combine", parallel=False,
                phases=[_phase(spec, os.path.join(str(run_dir), name))],
            ))
        elif spec.get("parallel") or spec.get("members"):
            members = [
                _phase(m, os.path.join(str(run_dir), name,
                                       m.get("name", "member")))
                for m in (spec.get("members") or [])
            ]
            stages.append(Stage(
                name=name, parallel=True, phases=members,
                min_success=spec.get("min_success"),
            ))
        else:
            # Sequential step: share the base run_dir so restart files chain.
            stages.append(Stage(
                name=name, parallel=False, phases=[_phase(spec, run_dir)],
            ))
    return stages


def _collect_input_files(gen_result: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{filename: contents}`` from a generation result.

    Reads the normalized ``input_files`` map every ``_generate_inputs``
    branch produces. Values may be inlined contents or paths; paths are
    read so the InputValidator always receives contents. No engine-specific
    filenames are assumed.
    """
    contents: Dict[str, str] = {}
    files = gen_result.get("input_files")
    if not isinstance(files, dict):
        return contents
    for name, val in files.items():
        if not isinstance(val, str):
            continue
        try:
            p = Path(val)
            if p.exists():
                contents[name] = p.read_text()
                continue
        except (OSError, ValueError):
            pass
        contents[name] = val
    return contents
