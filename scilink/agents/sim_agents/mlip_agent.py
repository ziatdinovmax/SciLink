"""
MLIPAgent — AI-driven agent for ML interatomic potentials.

Workflow (pretrained-first by design):

    ┌─────────────────────┐
    │ Deploy pretrained    │ ← default, zero training data needed
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Run simulation      │  (external — LAMMPS)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Evaluate quality    │ ← uncertainty + observables + LLM
    └─────────┬───────────┘
              │
         ┌────┴────┐
         │ OK?     │
         └────┬────┘
          yes │      no
              │       │
              ▼       ▼
           Done   ┌──────────────────┐
                  │ Identify failures │
                  └────────┬─────────┘
                           │
                  ┌────────┴────────┐
                  │                 │
                  ▼                 ▼
           ┌────────────┐   ┌─────────────┐
           │ Alert user │   │ Generate DFT│ ← opt-in
           │ with frames│   │ + fine-tune │
           └────────────┘   └─────────────┘
"""

import os
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional

from ...auth import (
    APIKeyNotFoundError, get_api_key, get_internal_proxy_key, infer_provider,
    require_vendor_credentials,
)
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...skills.loader import load_skill, list_skills
from ._deprecation import normalize_params

try:
    # mlip_tools is a multi-backend dispatcher (MACE / NequIP / DeePMD)
    # used across the machine_learning_potentials skill bundles, so it
    # lives in skills/_shared/ rather than under any one skill bundle.
    from ...skills._shared import mlip_tools
    _MLIP_TOOLS_AVAILABLE = True
except ImportError:
    _MLIP_TOOLS_AVAILABLE = False


class MLIPAgent:
    """
    Agent for ML interatomic potential deployment, evaluation, and refinement.

    The design is pretrained-first: deploy_pretrained() is the primary
    entry point and requires no training data.  Fine-tuning is only
    triggered when evaluate_simulation_quality() indicates the pretrained
    model is insufficient.
    """

    def __init__(
        self,
        working_dir: str,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        base_url: Optional[str] = None,
        skill: Optional[str] = None,
        local_model: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(handler)

        api_key, base_url = normalize_params(
            api_key=api_key, google_api_key=google_api_key,
            base_url=base_url, local_model=local_model,
            source="MLIPAgent",
        )

        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            self.model = OpenAIAsGenerativeModel(
                model=model_name, api_key=api_key, base_url=base_url,
            )
        else:
            # Public / LiteLLM — delegate model→provider→env-var resolution
            # to LiteLLM (works for any model LiteLLM supports; raises a
            # message naming the missing vendor env var if not).
            if api_key is None:
                require_vendor_credentials(model_name)
            self.model = LiteLLMGenerativeModel(
                model=model_name, api_key=api_key,
            )

        self.generation_config = None

        # Auth params retained so deploy_pretrained can construct the
        # MDSimulationAgent it delegates run-generation to (see
        # _get_md_agent). MLIPAgent's job ends at "here is a deployed
        # potential" — the MD agent owns the actual run.
        self._api_key = api_key
        self._model_name = model_name
        self._base_url = base_url
        self._md_agent = None   # lazily constructed in _get_md_agent

        # Skill state
        self.skill_name: Optional[str] = None
        self.skill_sections: Optional[Dict[str, str]] = None
        try:
            self._available_skills = list_skills(domain="machine_learning_potentials")
        except Exception:
            self._available_skills = []

        # Always try to load the general MLIP skill
        self._load_skill("general")
        if skill and skill != "general":
            self._load_backend_skill(skill)

        # Backend cache
        self._backends: Optional[Dict[str, Any]] = None

        # Refinement history — tracks the active learning loop
        self._refinement_history: List[Dict[str, Any]] = []

    # ================================================================
    # SKILLS
    # ================================================================

    def _load_skill(self, skill: str) -> bool:
        try:
            parsed = load_skill(skill, domain="machine_learning_potentials")
            self.skill_name = parsed["name"]
            self.skill_sections = parsed
            self.logger.info(f"📖 Loaded MLIP skill: {self.skill_name}")
            return True
        except FileNotFoundError:
            self.logger.debug(f"Skill '{skill}' not found in mlip domain")
            return False

    def _load_backend_skill(self, backend: str) -> bool:
        """Load backend-specific skill and merge into existing sections."""
        try:
            parsed = load_skill(backend, domain="machine_learning_potentials")
            if self.skill_sections is None:
                self.skill_sections = parsed
            else:
                # Merge: backend-specific content appended to each section
                for key, content in parsed.items():
                    if key == "name":
                        continue
                    existing = self.skill_sections.get(key, "")
                    if content:
                        self.skill_sections[key] = (
                            f"{existing}\n\n"
                            f"--- {backend.upper()} SPECIFIC ---\n{content}"
                            if existing else content
                        )
            self.logger.info(f"📖 Loaded backend skill: {backend}")
            return True
        except FileNotFoundError:
            self.logger.debug(f"Backend skill '{backend}' not found")
            return False

    def _get_skill_context(self, section: Optional[str] = None) -> str:
        if not self.skill_sections:
            return ""
        if section:
            content = self.skill_sections.get(section, "")
            return (
                f"=== MLIP Knowledge ({section}) ===\n{content}"
                if content else ""
            )
        return self.skill_sections.get("overview", "")

    # ================================================================
    # BACKENDS
    # ================================================================

    def _get_backends(self) -> Dict[str, Any]:
        if self._backends is None:
            if _MLIP_TOOLS_AVAILABLE:
                self._backends = mlip_tools.check_backends()
            else:
                self._backends = {}
        return self._backends

    def _select_pretrained_model(
        self, elements: List[str], research_goal: str,
    ) -> Dict[str, Any]:
        """
        Use LLM + skill knowledge to select the best pretrained model
        for the given system.
        """
        backends = self._get_backends()
        pretrained_options = []
        for backend_name, info in backends.items():
            if not info.get("available"):
                continue
            for model_info in info.get("pretrained", []):
                pretrained_options.append({
                    "backend": backend_name,
                    **model_info,
                })

        if not pretrained_options:
            raise RuntimeError(
                "No MLIP backends with pretrained models are available. "
                "Install mace-torch: pip install mace-torch"
            )

        # If only one option, skip the LLM call
        if len(pretrained_options) == 1:
            choice = pretrained_options[0]
            self.logger.info(
                f"Single pretrained option: {choice['name']} ({choice['backend']})"
            )
            return choice

        skill_context = self._get_skill_context("planning")
        options_str = json.dumps(pretrained_options, indent=2)

        prompt = f"""
Select the best pretrained ML potential for this system.

ELEMENTS: {elements}
RESEARCH GOAL: {research_goal}

AVAILABLE PRETRAINED MODELS:
{options_str}

{skill_context}

Selection criteria:
- Element coverage: does the model's training data include these elements?
- Domain match: inorganic model for metals/oxides, organic for molecules
- If elements include C, H, N, O and the system is molecular → prefer organic models
- If elements include metals, oxides → prefer inorganic models

Return JSON:
{{
    "backend": "mace",
    "model_name": "mace-mp-0",
    "justification": "why this model is appropriate",
    "cautions": "any limitations to be aware of"
}}
"""
        try:
            result = self._generate_json(prompt)
            # Validate the selection
            valid_names = {m["name"] for m in pretrained_options}
            if result.get("model_name") not in valid_names:
                result["model_name"] = pretrained_options[0]["name"]
                result["backend"] = pretrained_options[0]["backend"]
            return result
        except Exception:
            return pretrained_options[0]

    # ================================================================
    # PUBLIC API — PRETRAINED DEPLOYMENT (the default path)
    # ================================================================

    def deploy_pretrained(
        self,
        system_info: Dict[str, Any],
        research_goal: str,
        simulation_params: Optional[Dict[str, Any]] = None,
        runner: str = "lammps",
        structure_file: Optional[str] = None,
        backend: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deploy a pretrained foundation model.  This is the primary entry
        point — no training data or GPU time needed.

        Args:
            system_info: From ForceFieldAgent._analyze_system_composition()
                         or any dict with "elements" and "n_atoms" keys
            research_goal: What the simulation should achieve
            simulation_params: Optional run settings, forwarded to the MD
                agent: {"task": "md"|"relax", "temperature": 300,
                "pressure": 1.0, "timestep": ..., "n_steps": ...,
                "output_interval": ..., "device": "cuda", "fmax": ...}.
                Omitted keys fall back to the MD agent / runner defaults.
            runner: Which engine the MD agent drives the potential
                through — "lammps" (default), "ase", or any other engine
                registered in MDSimulationAgent.TOOL_REGISTRY that
                implements run_with_potential. "ase" is the universal
                fallback (works for every backend).
            structure_file: Path to a LAMMPS data file describing the
                system. Required — the MD agent's run reads from it.

        This method deploys the model, then delegates run generation to
        MDSimulationAgent. MLIPAgent's job ends at "here is a deployed
        potential"; the MD agent owns the actual run.

        Returns dict: backend, model_name, model_file, elements,
        selection, plus the MD agent's run result (run_path, runner,
        task, notes, ...).
        """
        if not _MLIP_TOOLS_AVAILABLE:
            raise ImportError(
                "mlip_tools required. Install mace-torch: pip install mace-torch"
            )
        if not structure_file:
            raise ValueError(
                "deploy_pretrained requires a structure_file "
                "(LAMMPS data file path) — the MD agent's run reads from it"
            )

        self.logger.info("=" * 60)
        self.logger.info(f"DEPLOYING PRETRAINED MLIP (runner={runner})")
        self.logger.info("=" * 60)

        elements = sorted(system_info.get("elements", {}).keys())

        # If the caller explicitly forces a backend, skip the LLM
        # selection step entirely — useful for benchmarks, regression
        # tests, and demos that want to exercise a specific engine.
        if backend is not None:
            selection = {
                "backend":       backend,
                "model_name":    model_name or backend,
                "justification": f"caller-forced via backend={backend!r}",
            }
        else:
            selection = self._select_pretrained_model(elements, research_goal)
        backend = selection.get("backend", "mace")
        model_name = selection.get("model_name") or backend

        self.logger.info(f"Selected: {model_name} ({backend})")
        self.logger.info(f"Reason: {selection.get('justification', '')}")

        # Load backend-specific skill
        self._load_backend_skill(backend)

        sim = simulation_params or {}

        # Deploy -- constructs the calculator once (validates the
        # install, locates the model file), returns an engine-neutral
        # DeployedPotential descriptor.
        potential = mlip_tools.deploy(
            backend=backend,
            model=model_name,
            elements=elements,
            working_dir=self.working_dir,
            device=sim.get("device", "cpu"),
        )

        # Delegate run generation to MDSimulationAgent. MLIPAgent's job
        # ends here -- "here is a deployed potential". The MD agent
        # owns the actual run (relax / MD / whatever the goal asks for)
        # and is the place that knows how to drive any potential
        # through any runner. See project_mlip_md_delegation memory.
        #
        # Forward only the run params the caller actually set -- the MD
        # agent / runner supply defaults for the rest, and passing an
        # explicit None would clobber those defaults.
        run_params = {
            k: sim[k]
            for k in ("timestep", "n_steps", "output_interval",
                      "device", "fmax")
            if sim.get(k) is not None
        }
        md_agent = self._get_md_agent()
        run = md_agent.generate_simulation(
            structure_file=structure_file,
            research_goal=research_goal,
            potential=potential,
            runner=runner,
            task=sim.get("task", "md"),
            temperature=sim.get("temperature", 300.0),
            pressure=sim.get("pressure"),   # None -> NVT
            **run_params,
        )

        result: Dict[str, Any] = {
            "backend":    potential.backend,
            "model_name": potential.model_name,
            "model_file": potential.model_file,
            "elements":   potential.elements,
            "selection":  selection,
            **run,        # run_path, runner, task, notes, ...
        }

        with open(os.path.join(self.working_dir, "deployment.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in result.items()
                 if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
                f, indent=2,
            )

        self.logger.info(f"Model: {potential.model_file or '(bundled)'}")
        self.logger.info(f"Run:   {run.get('run_path')}")
        self.logger.info("=" * 60)
        return result

    def _get_md_agent(self):
        """Lazily construct (and cache) the MDSimulationAgent this
        agent delegates run generation to.

        Constructed once per MLIPAgent and reused. The potential-driven
        path through the MD agent does not call the LLM, but the MD
        agent's __init__ builds one anyway, so we pass MLIPAgent's own
        auth params through for consistency.
        """
        if self._md_agent is None:
            from .md_simulation_agent import MDSimulationAgent
            self._md_agent = MDSimulationAgent(
                working_dir=str(self.working_dir),
                api_key=self._api_key,
                model_name=self._model_name,
                base_url=self._base_url,
            )
        return self._md_agent

    # ================================================================
    # PUBLIC API — FEEDBACK AND EVALUATION
    # ================================================================

    def evaluate_simulation_quality(
        self,
        deployment: Dict[str, Any],
        trajectory_file: Optional[str] = None,
        thermo_log: Optional[str] = None,
        expected_properties: Optional[Dict[str, float]] = None,
        research_goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether the MLIP simulation is scientifically reliable.

        Combines three signals:
        1. MLIP uncertainty on trajectory frames
        2. Physical observables vs expectations (density, etc.)
        3. LLM interpretation of all evidence

        Call this AFTER running the LAMMPS simulation.

        Args:
            deployment: Output of deploy_pretrained()
            trajectory_file: LAMMPS trajectory (lammpstrj or xyz)
            thermo_log: LAMMPS log file for thermo data
            expected_properties: Optional dict of expected values:
                {"density": 2.3, "lattice_constant": 4.05}
            research_goal: What the simulation should achieve

        Returns:
            {
                "acceptable": bool,
                "confidence": "high" | "medium" | "low",
                "uncertainty_report": {...},
                "observable_report": {...},
                "llm_assessment": str,
                "recommendation": "proceed" | "fine-tune" | "change-model",
                "problematic_frames": [int],
            }
        """
        self.logger.info("Evaluating MLIP simulation quality...")

        model_file = deployment.get("model_file")
        backend = deployment.get("backend", "mace")
        result: Dict[str, Any] = {
            "acceptable": True,
            "confidence": "high",
            "recommendation": "proceed",
            "problematic_frames": [],
        }

        # ── Signal 1: MLIP uncertainty on trajectory ──────────────
        uncertainty_report: Dict[str, Any] = {}
        if trajectory_file and os.path.exists(trajectory_file):
            try:
                import ase.io
                frames = list(ase.io.read(trajectory_file, index="::10"))  # subsample
                self.logger.info(f"Evaluating uncertainty on {len(frames)} frames")

                uncertainty_report = mlip_tools.evaluate_uncertainty(
                    backend=backend,
                    model_file=model_file,
                    structures=frames,
                )

                if uncertainty_report["n_extrapolating"] > 0:
                    frac = uncertainty_report["n_extrapolating"] / len(frames)
                    self.logger.warning(
                        f"{uncertainty_report['n_extrapolating']}/{len(frames)} "
                        f"frames flagged as extrapolation ({frac:.0%})"
                    )
                    if frac > 0.1:
                        result["acceptable"] = False
                        result["confidence"] = "low"

                result["problematic_frames"] = uncertainty_report.get(
                    "extrapolation_indices", []
                )

            except Exception as e:
                self.logger.warning(f"Uncertainty evaluation failed: {e}")
                uncertainty_report = {"error": str(e)}

        result["uncertainty_report"] = uncertainty_report

        # ── Signal 2: Observable comparison ───────────────────────
        observable_report = self._check_observables(
            thermo_log, expected_properties
        )
        result["observable_report"] = observable_report

        if observable_report.get("has_issues"):
            result["acceptable"] = False
            if result["confidence"] == "high":
                result["confidence"] = "medium"

        # ── Signal 3: LLM interpretation ──────────────────────────
        llm_assessment = self._llm_assess_quality(
            uncertainty_report=uncertainty_report,
            observable_report=observable_report,
            deployment=deployment,
            research_goal=research_goal,
        )
        result["llm_assessment"] = llm_assessment.get("assessment", "")

        # LLM can override our heuristic judgment
        llm_recommendation = llm_assessment.get("recommendation", "proceed")
        if llm_recommendation in ("fine-tune", "change-model"):
            result["acceptable"] = False
            result["recommendation"] = llm_recommendation
        elif result["acceptable"]:
            result["recommendation"] = "proceed"
        else:
            result["recommendation"] = "fine-tune"

        # Log summary
        self.logger.info(
            f"Quality assessment: acceptable={result['acceptable']}, "
            f"confidence={result['confidence']}, "
            f"recommendation={result['recommendation']}"
        )

        # Save
        report_file = os.path.join(self.working_dir, "quality_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    # ================================================================
    # PUBLIC API — REFINEMENT (triggered by evaluation)
    # ================================================================

    def refine_potential(
        self,
        deployment: Dict[str, Any],
        quality_report: Dict[str, Any],
        trajectory_file: str,
        mode: str = "alert",
        dft_code: str = "vasp",
        dft_settings: Optional[Dict[str, Any]] = None,
        existing_dft_data: Optional[str] = None,
        timeout_hours: float = 12.0,
    ) -> Dict[str, Any]:
        """
        Refine the MLIP based on quality evaluation feedback.

        Operates in two modes controlled by the `mode` parameter:

        mode="alert" (default, safe):
            - Extracts high-uncertainty frames from trajectory
            - Writes DFT input files
            - Returns instructions for the user to run DFT manually
            - Does NOT run training

        mode="autonomous":
            - Extracts high-uncertainty frames
            - Writes DFT input files
            - If existing_dft_data is provided, proceeds to fine-tune
            - Returns the fine-tuned model

        Args:
            deployment: Output of deploy_pretrained()
            quality_report: Output of evaluate_simulation_quality()
            trajectory_file: LAMMPS trajectory used in evaluation
            mode: "alert" | "autonomous"
            dft_code: "vasp" | "cp2k" | "generic" (for input generation)
            dft_settings: DFT calculation settings
            existing_dft_data: Path to extXYZ with DFT results
                (for autonomous mode — skip DFT generation)
            timeout_hours: Training time limit (autonomous mode)

        Returns:
            {
                "mode": str,
                "n_problematic_frames": int,
                "dft_inputs": {...} | None,
                "instructions": str,
                "fine_tuned_model": str | None,
                "new_deployment": DeployedPotential | None,
            }
        """
        self.logger.info(f"Refining potential (mode={mode})...")

        model_file = deployment.get("model_file")
        backend = deployment.get("backend", "mace")

        # ── Extract problematic frames ────────────────────────────
        problematic_frames = mlip_tools.extract_problematic_frames(
            trajectory_file=trajectory_file,
            model_file=model_file,
            backend=backend,
            top_n=20,
        )

        result: Dict[str, Any] = {
            "mode": mode,
            "n_problematic_frames": len(problematic_frames),
            "fine_tuned_model": None,
            "new_deployment": None,
        }

        # ── Generate DFT inputs ──────────────────────────────────
        dft_dir = os.path.join(self.working_dir, "dft_calculations")
        dft_inputs = mlip_tools.write_dft_inputs(
            structures=problematic_frames,
            working_dir=dft_dir,
            dft_code=dft_code,
            dft_settings=dft_settings,
        )
        result["dft_inputs"] = dft_inputs

        # ── Mode: alert ──────────────────────────────────────────
        if mode == "alert":
            result["instructions"] = (
                f"The MLIP identified {len(problematic_frames)} problematic "
                f"configurations. DFT input files have been written to:\n"
                f"  {dft_dir}\n\n"
                f"{dft_inputs['instructions']}\n\n"
                f"After DFT calculations complete, call refine_potential() again "
                f"with mode='autonomous' and existing_dft_data=<path to collected extXYZ>."
            )
            self.logger.info(result["instructions"])

            # Track in refinement history
            self._refinement_history.append({
                "iteration": len(self._refinement_history),
                "mode": "alert",
                "n_frames": len(problematic_frames),
                "dft_dir": dft_dir,
            })

            return result

        # ── Mode: autonomous ─────────────────────────────────────
        if existing_dft_data is None:
            result["instructions"] = (
                "Autonomous mode requires existing_dft_data. "
                "Run DFT on the structures in {dft_dir} first, then "
                "provide the collected extXYZ."
            )
            self.logger.warning(result["instructions"])
            return result

        # Build dataset from DFT results
        self.logger.info("Building fine-tuning dataset from DFT data...")
        dataset_dir = os.path.join(self.working_dir, "finetune_data")
        dataset_info = mlip_tools.build_training_dataset(
            structures=[],
            working_dir=dataset_dir,
            existing_data_path=existing_dft_data,
        )

        # Fine-tune
        self.logger.info("Fine-tuning from pretrained model...")
        finetune_dir = os.path.join(self.working_dir, "finetune")
        train_result = mlip_tools.train(
            backend=backend,
            dataset_info=dataset_info,
            working_dir=finetune_dir,
            foundation_model=model_file,
            hyperparameters={
                "max_num_epochs": 100,    # fewer epochs for fine-tuning
                "learning_rate": 0.001,   # lower LR for fine-tuning
            },
            timeout_hours=timeout_hours,
        )

        # Validate
        if dataset_info.get("val_file"):
            validation = mlip_tools.validate_model(
                backend=backend,
                model_file=train_result["model_file"],
                val_file=dataset_info["val_file"],
                working_dir=finetune_dir,
            )
            train_result["validation"] = validation
            self.logger.info(
                f"Validation: E_MAE={validation['energy_mae_meV']:.2f} meV/atom, "
                f"F_MAE={validation['force_mae_meV_A']:.1f} meV/Å"
            )

        result["fine_tuned_model"] = train_result["model_file"]
        result["training_result"] = train_result
        result["instructions"] = (
            f"Fine-tuned model saved to: {train_result['model_file']}\n"
            f"Hand result['new_deployment'] (a DeployedPotential) to the "
            f"MD agent, or pass the model file to deploy_pretrained()."
        )

        # Wrap the refined model as a DeployedPotential — refine_potential
        # produces a potential; the MD agent owns running it.
        elements = deployment.get("elements", [])
        result["new_deployment"] = mlip_tools.deploy(
            backend=backend,
            model=train_result["model_file"],
            elements=elements,
            working_dir=os.path.join(self.working_dir, "refined"),
        )

        # Track
        self._refinement_history.append({
            "iteration": len(self._refinement_history),
            "mode": "autonomous",
            "n_frames": len(problematic_frames),
            "model_file": train_result["model_file"],
            "validation": train_result.get("validation"),
        })

        return result

    # ================================================================
    # CONVENIENCE: FULL WORKFLOW
    # ================================================================

    def complete_workflow(
        self,
        system_info: Dict[str, Any],
        research_goal: str,
        structure_file: str,
        simulation_params: Optional[Dict[str, Any]] = None,
        runner: str = "lammps",
    ) -> Dict[str, Any]:
        """
        Deploy a pretrained model and generate a runnable simulation.

        This is the simplest entry point.  After the user runs the
        simulation, they call evaluate_simulation_quality() and
        refine_potential() as needed.

        Returns:
            Same as deploy_pretrained().
        """
        return self.deploy_pretrained(
            system_info=system_info,
            research_goal=research_goal,
            structure_file=structure_file,
            simulation_params=simulation_params,
            runner=runner,
        )

    # ================================================================
    # PRIVATE — OBSERVABLE CHECKING
    # ================================================================

    def _check_observables(
        self,
        thermo_log: Optional[str],
        expected_properties: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        """Parse LAMMPS thermo log and compare against expectations."""
        report: Dict[str, Any] = {
            "has_issues": False,
            "checks": [],
        }

        if not thermo_log or not os.path.exists(thermo_log):
            return report

        # Parse thermo data from LAMMPS log
        thermo_data = self._parse_lammps_thermo(thermo_log)
        if not thermo_data:
            report["checks"].append({
                "property": "thermo_parsing",
                "status": "warning",
                "message": "Could not parse thermo data from log file",
            })
            return report

        # Check energy stability (drift)
        if "TotEng" in thermo_data:
            energies = thermo_data["TotEng"]
            if len(energies) > 10:
                # Check for monotonic drift
                first_quarter = energies[:len(energies)//4]
                last_quarter = energies[-len(energies)//4:]
                drift = abs(
                    float(np.mean(last_quarter)) - float(np.mean(first_quarter))
                )
                mean_e = abs(float(np.mean(energies)))
                relative_drift = drift / max(mean_e, 1e-10)

                status = "ok" if relative_drift < 0.01 else "warning"
                if relative_drift > 0.05:
                    status = "error"
                    report["has_issues"] = True

                report["checks"].append({
                    "property": "energy_drift",
                    "status": status,
                    "value": relative_drift,
                    "message": f"Relative energy drift: {relative_drift:.4f}",
                })

        # Check density against expectation
        if expected_properties and "density" in expected_properties:
            if "Density" in thermo_data:
                sim_density = float(np.mean(thermo_data["Density"][-100:]))
                exp_density = expected_properties["density"]
                rel_error = abs(sim_density - exp_density) / exp_density

                status = "ok" if rel_error < 0.05 else "warning"
                if rel_error > 0.15:
                    status = "error"
                    report["has_issues"] = True

                report["checks"].append({
                    "property": "density",
                    "status": status,
                    "expected": exp_density,
                    "simulated": sim_density,
                    "relative_error": rel_error,
                })

        # Check temperature stability
        if "Temp" in thermo_data:
            temps = thermo_data["Temp"]
            if len(temps) > 10:
                temp_std = float(np.std(temps[-100:]))
                temp_mean = float(np.mean(temps[-100:]))

                # For NVT, temperature fluctuations should be reasonable
                cv = temp_std / max(temp_mean, 1.0)
                status = "ok" if cv < 0.1 else "warning"

                report["checks"].append({
                    "property": "temperature_stability",
                    "status": status,
                    "mean": temp_mean,
                    "std": temp_std,
                })

        return report

    def _parse_lammps_thermo(self, log_file: str) -> Dict[str, List[float]]:
        """Extract thermo columns from a LAMMPS log file."""
        import numpy as np

        data: Dict[str, List[float]] = {}
        headers: List[str] = []

        try:
            with open(log_file) as f:
                in_thermo = False
                for line in f:
                    stripped = line.strip()

                    # Detect thermo header
                    if stripped.startswith("Step "):
                        headers = stripped.split()
                        for h in headers:
                            data.setdefault(h, [])
                        in_thermo = True
                        continue

                    if in_thermo:
                        if stripped.startswith("Loop time") or not stripped:
                            in_thermo = False
                            continue
                        parts = stripped.split()
                        if len(parts) == len(headers):
                            try:
                                values = [float(x) for x in parts]
                                for h, v in zip(headers, values):
                                    data[h].append(v)
                            except ValueError:
                                in_thermo = False
        except Exception as e:
            self.logger.warning(f"Error parsing LAMMPS log: {e}")

        return data

    # ================================================================
    # PRIVATE — LLM HELPERS
    # ================================================================

    def _generate_json(self, prompt: str) -> dict:
        response = self.model.generate_content(
            prompt, generation_config=self.generation_config
        )
        if not response or not response.text:
            raise ValueError("Empty response from LLM")
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = "\n".join(
                l for l in raw.split("\n")
                if not l.strip().startswith("```")
            )
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        return json.loads(raw)

    def _generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt, generation_config=self.generation_config
        )
        if not response or not response.text:
            raise ValueError("Empty response from LLM")
        return response.text.strip()

    def _llm_assess_quality(
        self,
        uncertainty_report: Dict[str, Any],
        observable_report: Dict[str, Any],
        deployment: Dict[str, Any],
        research_goal: Optional[str],
    ) -> Dict[str, Any]:
        """LLM interprets all quality signals and makes a recommendation."""
        skill_context = self._get_skill_context("validation")

        prompt = f"""
You are evaluating an ML interatomic potential simulation.

MODEL: {deployment.get('model_name', '?')} ({deployment.get('backend', '?')})
RESEARCH GOAL: {research_goal or 'Not specified'}

UNCERTAINTY ANALYSIS:
  Mean uncertainty: {uncertainty_report.get('mean_energy_uncertainty', '?')} meV/atom
  Max uncertainty: {uncertainty_report.get('max_energy_uncertainty', '?')} meV/atom
  Extrapolating frames: {uncertainty_report.get('n_extrapolating', '?')}

OBSERVABLE CHECKS:
{json.dumps(observable_report.get('checks', []), indent=2)}

{skill_context}

Assess whether this simulation is scientifically reliable for the
stated research goal.

Return JSON:
{{
    "assessment": "concise explanation of quality",
    "recommendation": "proceed|fine-tune|change-model",
    "key_concerns": ["concern 1", "concern 2"],
    "suggested_actions": ["action 1"]
}}
"""
        try:
            return self._generate_json(prompt)
        except Exception:
            return {
                "assessment": "Could not generate LLM assessment",
                "recommendation": "proceed",
            }

    # ================================================================
    # PUBLIC API — DIRECT FINE-TUNING (skip pretrained deployment)
    # ================================================================

    def fine_tune(
        self,
        training_data: str,
        system_info: Dict[str, Any],
        research_goal: str,
        backend: Optional[str] = None,
        foundation_model: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        simulation_params: Optional[Dict[str, Any]] = None,
        timeout_hours: float = 12.0,
    ) -> Dict[str, Any]:
        """
        Fine-tune an MLIP directly from user-provided DFT data.
        Skips the pretrained deployment + evaluation loop entirely.

        This is the entry point for users who already have training data
        and want to go straight to a fine-tuned model.

        Args:
            training_data: Path to extXYZ file with energy+forces.
                Can also be a directory of extXYZ files (they get merged).
            system_info: System composition info (from ForceFieldAgent or
                manually constructed: {"elements": {"Si": 64, "O": 128}})
            research_goal: What the simulation should achieve
            backend: MLIP backend ("mace", "nequip", "deepmd").
                If None, auto-selected based on data and research goal.
            foundation_model: Foundation model to fine-tune from.
                If None, auto-selected.  Pass "none" or "scratch" to
                train from scratch.
            hyperparameters: Training config overrides.
            simulation_params: Optional {"device": ...} for wrapping the
                trained model as a DeployedPotential.
            timeout_hours: Training wall-clock limit.

        Returns:
            {
                "model_file": str,
                "backend": str,
                "validation": dict,
                "potential": DeployedPotential,   # hand to the MD agent
                "dataset_info": dict,
            }
        """
        if not _MLIP_TOOLS_AVAILABLE:
            raise ImportError("mlip_tools required for fine-tuning")

        self.logger.info("=" * 60)
        self.logger.info("DIRECT FINE-TUNING")
        self.logger.info("=" * 60)

        elements = sorted(system_info.get("elements", {}).keys())

        # ── Resolve training data path ────────────────────────────
        training_data = str(training_data)
        if os.path.isdir(training_data):
            training_data = self._merge_xyz_directory(training_data)

        # ── Build dataset ─────────────────────────────────────────
        dataset_dir = os.path.join(self.working_dir, "dataset")
        dataset_info = mlip_tools.build_training_dataset(
            structures=[],
            working_dir=dataset_dir,
            existing_data_path=training_data,
        )

        self.logger.info(
            f"Dataset: {dataset_info['n_train']} train, "
            f"{dataset_info['n_val']} val, "
            f"elements: {dataset_info['elements']}"
        )

        # ── Select backend + foundation model ─────────────────────
        if backend is None:
            selection = self._select_backend_for_finetuning(
                dataset_info, elements, research_goal
            )
            backend = selection["backend"]
            if foundation_model is None:
                foundation_model = selection.get("foundation_model")

        self._load_backend_skill(backend)

        # Handle explicit "train from scratch"
        from_scratch = foundation_model in (None, "none", "scratch", "")
        if from_scratch:
            foundation_model = None
            self.logger.info(f"Training {backend} from scratch")
        else:
            self.logger.info(f"Fine-tuning {backend} from {foundation_model}")

        # ── Resolve foundation model to a file path ───────────────
        resolved_foundation = None
        if foundation_model:
            resolved_foundation = self._resolve_foundation_model(foundation_model)
            if resolved_foundation is None:
                self.logger.warning(
                    f"Could not resolve {foundation_model} to a local path. "
                    f"Passing name directly — backend will handle resolution."
                )
                resolved_foundation = foundation_model

        # ── Merge hyperparameters with skill-informed defaults ────
        hparams = self._get_finetuning_hparams(
            backend, dataset_info, from_scratch, hyperparameters
        )

        # ── Train ─────────────────────────────────────────────────
        train_dir = os.path.join(self.working_dir, "training")
        train_result = mlip_tools.train(
            backend=backend,
            dataset_info=dataset_info,
            working_dir=train_dir,
            foundation_model=resolved_foundation,
            hyperparameters=hparams,
            timeout_hours=timeout_hours,
        )

        # ── Validate ──────────────────────────────────────────────
        validation = {}
        if dataset_info.get("val_file"):
            validation = mlip_tools.validate_model(
                backend=backend,
                model_file=train_result["model_file"],
                val_file=dataset_info["val_file"],
                working_dir=train_dir,
            )
            train_result["validation"] = validation

            self.logger.info(
                f"Validation: E_MAE={validation['energy_mae_meV']:.2f} meV/atom, "
                f"F_MAE={validation['force_mae_meV_A']:.1f} meV/Å, "
                f"passed={validation['passed']}"
            )

            if not validation["passed"]:
                diagnosis = self._diagnose_training_quality(train_result, {
                    "backend": backend,
                    "strategy": "scratch" if from_scratch else "finetune",
                })
                train_result["diagnosis"] = diagnosis
                self.logger.warning(
                    f"Validation below thresholds: {diagnosis.get('likely_cause')}"
                )

        # ── Wrap the trained model as a DeployedPotential ─────────
        # fine_tune's job ends at producing a potential. Running it is
        # the MD agent's job — hand the returned `potential` (or its
        # model_file) to deploy_pretrained / MDSimulationAgent when a
        # structure is in hand.
        sim = simulation_params or {}
        potential = mlip_tools.deploy(
            backend=backend,
            model=train_result["model_file"],
            elements=elements,
            working_dir=self.working_dir,
            device=sim.get("device", "cpu"),
        )

        result = {
            "model_file":   potential.model_file,
            "backend":      backend,
            "foundation":   foundation_model,
            "from_scratch":  from_scratch,
            "validation":   validation,
            "dataset_info": dataset_info,
            "potential":    potential,
            "hyperparameters": hparams,
        }

        # Save
        with open(os.path.join(self.working_dir, "finetune_result.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in result.items()
                 if isinstance(v, (str, int, float, bool, list, dict, type(None)))},
                f, indent=2,
            )

        self.logger.info("=" * 60)
        self.logger.info(f"Model: {potential.model_file}")
        self.logger.info("=" * 60)
        return result

    # ── Fine-tuning helpers ───────────────────────────────────────

    def _merge_xyz_directory(self, directory: str) -> str:
        """Merge all extXYZ files in a directory into one."""
        import ase.io

        merged_file = os.path.join(self.working_dir, "merged_training_data.xyz")
        all_frames = []
        for fname in sorted(os.listdir(directory)):
            if fname.endswith((".xyz", ".extxyz")):
                path = os.path.join(directory, fname)
                try:
                    all_frames.extend(ase.io.read(path, index=":"))
                except Exception as e:
                    self.logger.warning(f"Could not read {path}: {e}")

        if not all_frames:
            raise ValueError(f"No valid extXYZ files found in {directory}")

        ase.io.write(merged_file, all_frames, format="extxyz")
        self.logger.info(f"Merged {len(all_frames)} frames from {directory}")
        return merged_file

    def _select_backend_for_finetuning(
        self, dataset_info, elements, research_goal,
    ) -> Dict[str, Any]:
        """LLM selects backend + foundation model for fine-tuning."""
        backends = self._get_backends()
        available = {
            k: v for k, v in backends.items()
            if v.get("available") and k in ("mace", "nequip", "deepmd")
        }

        if not available:
            raise RuntimeError("No MLIP backends available.")

        # If only MACE is available, skip LLM call
        if list(available.keys()) == ["mace"]:
            # Pick foundation model based on elements
            organic_elements = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
            if set(elements).issubset(organic_elements):
                model = "mace-off23"
            else:
                model = "mace-mp-0"
            return {"backend": "mace", "foundation_model": model}

        skill_context = self._get_skill_context("planning")

        prompt = f"""
Select the best MLIP backend and foundation model for fine-tuning.

ELEMENTS: {elements}
DATASET: {dataset_info['n_train']} training structures
RESEARCH GOAL: {research_goal}

AVAILABLE BACKENDS: {json.dumps({k: v.get('pretrained', []) for k, v in available.items()}, indent=2)}

{skill_context}

For fine-tuning, prefer backends with relevant foundation models.
For training from scratch, any backend works.

Return JSON:
{{
    "backend": "mace|nequip|deepmd",
    "foundation_model": "model-name or null for from-scratch",
    "justification": "..."
}}
"""
        try:
            return self._generate_json(prompt)
        except Exception:
            return {"backend": "mace", "foundation_model": "mace-mp-0"}

    def _get_finetuning_hparams(
        self,
        backend: str,
        dataset_info: Dict[str, Any],
        from_scratch: bool,
        user_overrides: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build hyperparameters informed by skill knowledge.
        Fine-tuning uses lower LR and fewer epochs than from-scratch.
        """
        if from_scratch:
            defaults = {
                "max_num_epochs": 300,
                "learning_rate": 0.01,
                "batch_size": 4,
                "r_max": 5.0,
                "energy_weight": 1.0,
                "forces_weight": 100.0,
            }
        else:
            defaults = {
                "max_num_epochs": 100,
                "learning_rate": 0.001,
                "batch_size": 4,
                "r_max": 5.0,
                "energy_weight": 1.0,
                "forces_weight": 100.0,
            }

        # Scale batch size to dataset size
        n_train = dataset_info.get("n_train", 100)
        if n_train < 50:
            defaults["batch_size"] = 2
        elif n_train > 1000:
            defaults["batch_size"] = 8

        # Apply user overrides
        if user_overrides:
            defaults.update(user_overrides)

        return defaults

    def _resolve_foundation_model(self, foundation_model: str) -> Optional[str]:
        """
        Resolve a foundation model name to a local file path.

        MACE foundation models are downloaded and cached automatically
        by mace_mp() / mace_off().  This method triggers that download
        (if needed) and returns the cached file path.
        """
        if not _MLIP_TOOLS_AVAILABLE:
            return None

        try:
            result = mlip_tools.deploy(
                backend="mace",
                model=foundation_model,
                elements=[],          # not needed for path resolution
                working_dir=self.working_dir,
                device="cpu",
            )
            model_path = result.model_file
            if model_path and os.path.exists(str(model_path)):
                self.logger.info(f"Foundation model resolved: {model_path}")
                return str(model_path)
        except Exception as e:
            self.logger.debug(f"Could not resolve foundation model: {e}")

        # Return the name as-is — the training backend may resolve it
        return foundation_model
