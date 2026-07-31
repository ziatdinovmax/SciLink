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

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Default engine per scale, used when the caller does not name one. Each
# scale's foundation agent resolves the engine to a skill bundle.
_DEFAULT_ENGINE = {
    "periodic_dft": "vasp",
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
    required_observables: Optional[list] = None,
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
            required_observables=required_observables,
        )
        # PeriodicDFTAgent already returns input_files as {filename: contents}.
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
            required_observables=required_observables,
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


def _load_components_manifest(structure_path: str) -> Optional[Dict[str, Any]]:
    """Load a ``components.json`` manifest sitting next to a generated structure.

    Condensed structure generation writes this alongside the coordinate file:
    ``{"components": [{"name", "smiles", "count"}, ...]}`` in coordinate order.
    It is the force-field step's bridge from a packed box to per-species
    chemistry. Returns the manifest dict, or None when absent / unreadable (a
    crystal/molecular structure, an MLIP-MD run, or a caller-supplied data file
    has none — the FF step is then skipped).
    """
    if not structure_path:
        return None
    manifest = os.path.join(os.path.dirname(os.path.abspath(structure_path)),
                            "components.json")
    if not os.path.isfile(manifest):
        return None
    try:
        with open(manifest) as fh:
            data = json.load(fh)
        return data if data.get("components") else None
    except Exception:
        return None


def _measure_property_via_short_run(psystem, property, working_dir, engine,
                                    executor, run_command):
    """Measure a pure component's reference ``property`` with a short run.

    Returns ``{"value": float, "units": str}`` or None. Property-general: the
    property is a goal, not a hardcoded routine — a density is a short NPT, a
    lattice constant is a relaxation, etc.

    LIVE-ENV SEAM (not yet implemented): write engine inputs via
    ``write_md_inputs``, generate a short run to measure ``property`` through the
    normal input-generation path (so it works for any property/engine — no
    per-property code), run it through ``executor`` with ``run_command``, and
    read the value from the run snapshot. Until implemented it returns None, so
    the measurement is recorded as unmeasured and the gate fails open.
    """
    _ = (psystem, property, working_dir, engine, executor, run_command)
    return None


def _reference_check(components, system_description, psystem, ff_agent, *,
                     api_key, base_url, model_name, executor, run_command,
                     engine, working_dir, ff_kwargs=None):
    """Run the pre-run reference-property validation; return its result dict.

    Builds the selector + critic and a density measurer bound to the SAME
    force-field backend the production run uses (through ``ff_agent``), then runs
    the composed :func:`reference_validation.run_reference_check`. Module-level so
    tests and callers can substitute it. ``ff_kwargs`` are extra parameterization
    arguments (e.g. ``extra_force_fields``) applied to the pure-component
    parameterization too — so re-checking a candidate fix actually measures it.
    """
    from .critics import ReferencePropertyCritic, ReferencePropertySelector
    from .reference_measurement import measure_pure_component_property
    from .reference_validation import run_reference_check

    selector = ReferencePropertySelector(api_key=api_key, base_url=base_url,
                                         model_name=model_name)
    critic = ReferencePropertyCritic(api_key=api_key, base_url=base_url,
                                     model_name=model_name)

    def _measure(component, prop):
        name = component.get("name") or component.get("smiles") or "component"
        slug = "".join(c if c.isalnum() else "_" for c in str(name))
        wd = os.path.join(working_dir, "reference_check", slug)
        return measure_pure_component_property(
            component, prop or "density", wd,
            parameterize_fn=lambda comps, coords, w: ff_agent.parameterize(
                components=comps, coordinates_file=coords, working_dir=w,
                **(ff_kwargs or {})),
            run_measure_fn=lambda ps, p, w: _measure_property_via_short_run(
                ps, p, w, engine, executor, run_command),
        )

    return run_reference_check(
        components, system_description,
        select_fn=lambda comps, sd: selector.select(comps, system_description=sd),
        measure_fn=_measure,
        judge_fn=lambda meas, sd: critic.assess(meas, system_description=sd),
    )


def _search_force_field_parameters(recommendation, tried):
    """Find a candidate force-field correction for a flagged component.

    LIVE-ENV SEAM (not yet implemented): search the literature (SciLink's
    literature agents) for validated parameters for the offending chemistry and
    return them as parameterization kwargs (e.g.
    ``{"extra_force_fields": [<path>]}``), skipping anything in ``tried``. Until
    implemented it returns None, so the fixer loop reports ``no_candidate`` and
    escalates to the human rather than inventing parameters.
    """
    _ = (recommendation, tried)
    return None


def _reparameterize(flagged, system_description, backend, components,
                    coordinates_file, ff_agent, engine, working_dir, *,
                    api_key, base_url, model_name, executor, run_command,
                    confirm_fn=None):
    """Drive the autonomous fix: advise -> search -> apply candidate + re-check.

    SciLink sources the fix and the pure-component re-check validates it; the
    human only approves (``confirm_fn``). Module-level so tests can substitute
    it. Returns :func:`reference_validation.run_reparameterization`'s dict.
    """
    from .critics import ReparameterizationAdvisor
    from .reference_validation import run_reparameterization

    advisor = ReparameterizationAdvisor(api_key=api_key, base_url=base_url,
                                        model_name=model_name)

    def _apply_and_recheck(candidate):
        return _reference_check(
            components, system_description, None, ff_agent,
            api_key=api_key, base_url=base_url, model_name=model_name,
            executor=executor, run_command=run_command, engine=engine,
            working_dir=working_dir, ff_kwargs=(candidate or {}))

    return run_reparameterization(
        flagged, system_description, backend,
        advise_fn=lambda fl, sd, bk: advisor.advise(
            fl, system_description=sd, backend=bk),
        search_fn=_search_force_field_parameters,
        apply_and_recheck_fn=_apply_and_recheck,
        confirm_fn=confirm_fn or (lambda candidate: True),
    )


def _run_workflow_once(
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
    coverage_votes: int = 1,
    structure_file: Optional[str] = None,
    force_field_files: Optional[Dict[str, str]] = None,
    staged: bool = False,
    reference_check: bool = False,
    auto_fix: bool = True,
    required_observables: Optional[list] = None,
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
        coverage_votes: Number of independent observable-coverage checks the
            pre-run gate majority-votes before blocking a deck that omits a
            required output (e.g. the stress log a viscosity goal needs). ``1``
            is a single check; larger values damp the stochastic coverage
            decision on borderline transport-property cases.
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

    # ── Step 1.5: force-field parameterization (MD, force-field-based only) ──
    # Turn a packed box of coordinates into an engine-native, parameterized
    # input (e.g. a typed LAMMPS data file) via the engine-neutral FF stack:
    # ForceFieldAgent.parameterize -> ParameterizedSystem -> write_md_inputs.
    # Gated on a components.json manifest, so MLIP-driven MD (potential-based,
    # no manifest), pre-built data files, and non-MD scales are untouched. When
    # the caller already supplied force_field_files, respect them.
    if (scale == "molecular_dynamics" and force_field_files is None):
        manifest = _load_components_manifest(structure_path)
        if manifest:
            try:
                from .force_field_agent import ForceFieldAgent
                from ._engine_inputs import write_md_inputs
                ff_agent = ForceFieldAgent(
                    working_dir=output_dir, api_key=api_key,
                    base_url=base_url, model_name=model_name,
                )
                # Keep the packed coordinates — write_md_inputs overwrites
                # structure_path with the engine data file, but a fix has to
                # re-parameterize the mixture from the original coordinates.
                orig_coordinates = structure_path
                psystem = ff_agent.parameterize(
                    components=manifest["components"],
                    coordinates_file=structure_path,
                    working_dir=output_dir,
                )
                written = write_md_inputs(psystem, software, output_dir)
                structure_path = written["structure_file"]
                force_field_files = written["force_field_files"] or None
                result["force_field"] = {
                    "status": "success", "backend": psystem.backend,
                    "n_atoms": psystem.n_atoms,
                    "total_charge": psystem.total_charge,
                }
                result["steps_completed"].append("force_field")

                # Pre-run reference-property validation — the pre-PRODUCTION
                # catch. Validate the force field against known constituent
                # properties now, before spending production compute. A 'poor'
                # verdict means the model is untrustworthy, so stop here rather
                # than run (and later discard) an expensive campaign.
                if reference_check:
                    ref = _reference_check(
                        manifest["components"], user_request, psystem, ff_agent,
                        api_key=api_key, base_url=base_url, model_name=model_name,
                        executor=executor, run_command=run_command,
                        engine=software, working_dir=output_dir,
                    )
                    result["reference_validation"] = ref
                    result["steps_completed"].append("reference_validation")
                    verdict = (ref.get("verdict") or {})
                    if verdict.get("verdict") == "poor":
                        cause = (verdict.get("failure_class")
                                 or "force_field").replace("_", " ")
                        flagged = [m for m in verdict.get("per_measurement", [])
                                   if m.get("consistent") is False]

                        # Catch -> FIX: SciLink sources a correction and the
                        # pure-component re-check validates it; the human only
                        # approves. On success, re-parameterize the mixture with
                        # the corrected model and proceed to production.
                        fix = None
                        if auto_fix:
                            fix = _reparameterize(
                                flagged, user_request, psystem.backend,
                                manifest["components"], orig_coordinates,
                                ff_agent, software, output_dir,
                                api_key=api_key, base_url=base_url,
                                model_name=model_name, executor=executor,
                                run_command=run_command,
                            )
                            result["reparameterization"] = fix
                            result["steps_completed"].append("reparameterization")

                        if fix and fix.get("status") == "fixed":
                            psystem = ff_agent.parameterize(
                                components=manifest["components"],
                                coordinates_file=orig_coordinates,
                                working_dir=output_dir,
                                **(fix.get("candidate") or {}),
                            )
                            written = write_md_inputs(psystem, software, output_dir)
                            structure_path = written["structure_file"]
                            force_field_files = written["force_field_files"] or None
                            result["force_field"]["reparameterized"] = True
                            result.setdefault("warnings", []).append(
                                "Pre-run validation flagged the force field; it was "
                                "reparameterized and re-validated. Production uses "
                                "the corrected model."
                            )
                            # fall through to production with the corrected FF
                        else:
                            result["force_field_flagged"] = True
                            result.setdefault("warnings", []).append(
                                f"Pre-run reference-property validation flagged the "
                                f"{cause}: a pure-component property contradicts "
                                "known behaviour, and no validated fix was applied. "
                                "Production was NOT run — the model needs "
                                "reparameterization first."
                            )
                            result["final_status"] = "reference_validation_failed"
                            return result
            except Exception as e:
                result["final_status"] = "failed_force_field"
                result["force_field"] = {"status": "error", "message": str(e)}
                return result
        else:
            result.setdefault("warnings", []).append(
                "molecular_dynamics run with no components.json manifest next to "
                "the structure and no force_field_files supplied — skipping "
                "force-field parameterization; the deck may read raw coordinates "
                "and fail to run."
            )

    # ── Step 2: input generation (routed to the scale's foundation agent) ──
    try:
        gen_result = _generate_inputs(
            scale=scale, software=software, method=method,
            structure_file=structure_path, request=user_request,
            output_dir=output_dir, api_key=api_key, base_url=base_url,
            model_name=model_name, force_field_files=force_field_files,
            staged=staged, required_observables=required_observables,
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
        max_cycles=max_run_cycles, coverage_votes=coverage_votes,
        required_observables=required_observables,
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
    # Detection is wired; the automated reparameterization fix is not. When the
    # critic attributes a converged-but-wrong result to the force field, surface
    # it plainly so the result is not read as trustworthy.
    if refinement.get("failure_class") == "force_field":
        result["force_field_flagged"] = True
        result.setdefault("warnings", []).append(
            "A computed property contradicts the known physical behaviour of the "
            "system — the force field appears miscalibrated. No input-deck change "
            "can fix this; the force field needs reparameterization (not yet "
            "automated). Treat this result as unreliable."
        )
        logger.warning(
            "Run flagged force-field-limited (failure_class='force_field'); "
            "reparameterization needed — not yet automated."
        )
    result["final_status"] = (
        "success" if refinement.get("status") == "success"
        else f"refinement_{refinement.get('status', 'failed')}"
    )
    return result


def run_complete_workflow(
    user_request: str,
    *,
    max_structure_retries: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the full pipeline, regenerating the structure on a structure-caused failure.

    Thin wrapper over :func:`_run_workflow_once`. The refinement loop can only
    rewrite the run inputs, so it cannot recover from a broken *initial
    structure* (a bad packmol pack that blows up at step 0 — the critic marks it
    ``failure_class="structure"``), nor from structure generation failing
    outright. When either happens and ``max_structure_retries`` remain, the
    structure is regenerated (a fresh stochastic pack) and the whole workflow is
    retried, since no deck edit can fix a broken configuration. ``0`` (default)
    preserves the single-attempt behavior. All other arguments are forwarded
    unchanged — see :func:`_run_workflow_once`.
    """
    caller_supplied_structure = kwargs.get("structure_file") is not None
    result: Dict[str, Any] = {}
    for attempt in range(max(0, max_structure_retries) + 1):
        result = _run_workflow_once(user_request, **kwargs)
        structure_caused = (
            result.get("final_status") == "failed_structure_generation"
            or (result.get("refinement") or {}).get("failure_class") == "structure"
        )
        if (structure_caused and not caller_supplied_structure
                and attempt < max_structure_retries):
            logger.info(
                "structure-caused failure (%s); regenerating structure and "
                "retrying (%d/%d)",
                result.get("final_status"), attempt + 1, max_structure_retries,
            )
            continue
        return result
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
            entry_file=entry or "",
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
            entry_file=entry or "",
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
