"""
Backend-agnostic tools for ML interatomic potentials.

Detection, the pretrained-model list and deployment are driven by each
backend's skill bundle (its ``detect:``, ``pretrained:`` and ``deploy:``
frontmatter), so adding a backend is adding a skill bundle. Uncertainty
estimation and training still dispatch on ``backend`` to private
implementations; only MACE has them today, and the others raise
NotImplementedError with actionable messages so failures are obvious
rather than silent.
"""

from scilink.executors import run_engine  # stoppable, whole-tree kill (#685)
import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# DeployedPotential is the cross-agent contract (see
# agents/sim_agents/_potential.py). It's a pure-dataclass leaf module
# with no agent imports, so importing it here creates no cycle —
# mlip_tools.deploy() produces a DeployedPotential that
# MDSimulationAgent later consumes.
from ...agents.sim_agents._potential import (
    ASECalculatorSpec, DeployedPotential,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  BACKENDS COME FROM SKILL BUNDLES
# ═══════════════════════════════════════════════════════════════════
#
# Each MLIP backend is described by its skill bundle's frontmatter
# (skills/machine_learning_potentials/<backend>/<backend>.md):
#
#   detect:      is it installed?  (read by AvailableSoftware, as for
#                every other engine)
#   deploy:      templates for building its ASE calculator
#   pretrained:  the models deploy() can load, and the elements each
#                one was trained on
#
# So a new backend is a new skill bundle; no code here changes.

_MLIP_DOMAIN = "machine_learning_potentials"


def _skill_meta(backend: str) -> Dict[str, Any]:
    """Frontmatter of a backend's skill bundle (raises FileNotFoundError)."""
    from ..loader import load_skill
    parsed = load_skill(f"{_MLIP_DOMAIN}/{backend}", domain=_MLIP_DOMAIN)
    return parsed.get("meta") or {}


def _backend_names() -> List[str]:
    from ..loader import list_skills
    return [s for s in list_skills(domain=_MLIP_DOMAIN) if s != "general"]


def _element_set(spec: Any) -> Optional[set]:
    """Turn an ``elements`` spec into a set of symbols.

    A spec is a list of symbols, or ``{z: [lo, hi], exclude: [...]}``.
    Returns None when no spec is given (coverage unknown).
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return {str(s) for s in spec}
    if isinstance(spec, dict) and "z" in spec:
        from ase.data import chemical_symbols
        lo, hi = spec["z"]
        out = {chemical_symbols[z] for z in range(int(lo), int(hi) + 1)}
        return out - {str(s) for s in spec.get("exclude", [])}
    raise ValueError(f"Bad elements spec: {spec!r}")


def find_pretrained(backend: str, model: str) -> Optional[Dict[str, Any]]:
    """The ``pretrained`` entry for ``model`` (by name or alias), or None."""
    try:
        meta = _skill_meta(backend)
    except FileNotFoundError:
        return None
    for entry in meta.get("pretrained") or []:
        if model == entry.get("name") or model in (entry.get("aliases") or []):
            return entry
    return None


def model_elements(
    backend: str, model: str, task_name: Optional[str] = None,
) -> Optional[set]:
    """Elements a pretrained model was trained on, or None if unknown.

    Multi-task models (UMA) declare ``elements_by_task``; the head in
    use (``task_name``, else the skill's ``deploy.default_task``) picks
    the set.
    """
    entry = find_pretrained(backend, model)
    if entry is None:
        return None
    if "elements_by_task" in entry:
        task = task_name or (_skill_meta(backend).get("deploy") or {}).get(
            "default_task")
        return _element_set((entry["elements_by_task"] or {}).get(task))
    return _element_set(entry.get("elements"))


def uncovered_elements(
    backend: str, model: str, elements: List[str],
    task_name: Optional[str] = None,
) -> Optional[List[str]]:
    """Elements of the system that the model was not trained on.

    Returns ``[]`` when every element is covered, and None when the
    model declares no coverage (unknown).
    """
    covered = model_elements(backend, model, task_name)
    if covered is None:
        return None
    return sorted(set(elements) - covered)


def check_backends(
    backends: Optional[List[str]] = None,
    available_software: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Which MLIP backends are installed, and which models each offers.

    Installation is read through ``AvailableSoftware`` from each skill's
    ``detect:`` block, the same as every other engine. The model list is
    each skill's ``pretrained:`` entries (minus ``offer: false``). A
    backend whose skill has no ``deploy:`` block offers no models, since
    deploy() could not load them.

    Args:
        backends: Backend names to report. Default: every MLIP skill.
        available_software: An ``AvailableSoftware`` to read from.
            Default: a live ``AvailableSoftware.detect()``. Not the
            cached ``auto()``: its YAML lives in the home directory and
            is shared by every Python environment, and whether a Python
            package imports is a fact about this interpreter.

    Returns:
        ``{backend: {"available": bool, "pretrained": [
            {"name", "domain", "description"}, ...]}}``
    """
    if available_software is None:
        from ...utils.available_software import AvailableSoftware
        available_software = AvailableSoftware.detect()

    out: Dict[str, Dict[str, Any]] = {}
    for name in (backends if backends is not None else _backend_names()):
        try:
            meta = _skill_meta(name)
        except FileNotFoundError:
            continue
        info = available_software.get(_MLIP_DOMAIN, name)
        offered = []
        if meta.get("deploy"):
            for entry in meta.get("pretrained") or []:
                if entry.get("offer", True):
                    offered.append({k: entry[k] for k in
                                    ("name", "domain", "description")
                                    if k in entry})
        out[name] = {"available": bool(info.get("available")),
                     "pretrained": offered}
    return out


# ═══════════════════════════════════════════════════════════════════
#  MODEL DEPLOYMENT  (the default path)
# ═══════════════════════════════════════════════════════════════════

def _model_file_none(calc: Any, name: str, working_dir: str) -> str:
    return ""


def _model_file_torch_save(calc: Any, name: str, working_dir: str) -> str:
    """Find, or write, an on-disk copy of the loaded torch model.

    Engines such as LAMMPS load the model from a file. Uses the
    calculator's own path attribute when it has one, else saves
    ``calc.models[0]`` into ``working_dir``. Returns "" on failure.
    """
    for attr in ("model_path", "model_paths"):
        val = getattr(calc, attr, None)
        if val:
            path = str(val[0]) if isinstance(val, list) else str(val)
            if os.path.exists(path):
                return path
    path = os.path.join(working_dir, f"{name}.model")
    try:
        import torch
        torch.save(calc.models[0], path)
        return path
    except Exception as e:
        logger.warning(f"Could not save model file for {name}: {e}")
        return ""


# Named strategies a skill's ``model_file:`` field can choose.
_MODEL_FILE_STRATEGIES = {
    "none":       _model_file_none,
    "torch_save": _model_file_torch_save,
}


def _resolve_device(device: str, devices: Optional[List[str]]) -> str:
    """``device`` if the backend supports its kind, else "cpu".

    Matches on the part before any index, so "cuda:1" counts as "cuda".
    """
    if not devices or (device or "").split(":")[0] in devices:
        return device
    return "cpu"


def _build_calculator(import_line: str, construct_expr: str, device: str):
    """Construct the calculator once, exactly as the run script will.

    Validates the install and triggers any weight download. The
    templates come from a skill bundle; skill bundles already supply
    code that runs (this same expression goes into the run script).
    """
    ns: Dict[str, Any] = {"DEVICE": device}
    exec(import_line, ns)
    return eval(construct_expr, ns)


def deploy(
    backend: str,
    model: str,
    elements: List[str],
    working_dir: str,
    device: str = "cpu",
    task_name: Optional[str] = None,
) -> DeployedPotential:
    """
    Deploy an MLIP model as an engine-neutral DeployedPotential.

    ``model`` is a pretrained model name or alias from the backend
    skill's ``pretrained:`` list (``"mace-mp-0"``, ``"chgnet"``,
    ``"uma"``), or an on-disk path to a trained/fine-tuned model (the
    output of ``train()`` / ``fine_tune()``). Everything backend-specific
    comes from the skill's ``deploy:`` block, so a new backend needs no
    code here.

    A pretrained model is refused if the system has an element the model
    was not trained on. Several backends (Orb, UMA, DeePMD) would
    otherwise run such a system silently with meaningless forces.

    Args:
        backend:     Name of an MLIP skill bundle ("mace", "chgnet", ...)
        model:       Pretrained model name/alias, or path to a trained model
        elements:    Chemical elements in the system ([] skips the check)
        working_dir: Output directory
        device:      "cpu", "cuda", ...; falls back to "cpu" if the
                     backend does not list it
        task_name:   Prediction head for multi-task models (UMA: "omat",
                     "omol", ...). Defaults to the skill's
                     ``deploy.default_task``.

    Returns:
        A DeployedPotential descriptor — the engine-neutral contract
        MDSimulationAgent consumes to generate the actual run. See
        agents/sim_agents/_potential.py.
    """
    os.makedirs(working_dir, exist_ok=True)

    try:
        meta = _skill_meta(backend)
    except FileNotFoundError:
        raise ValueError(f"Unknown backend: {backend} "
                         f"(known: {', '.join(_backend_names())})")
    spec = meta.get("deploy")
    if not spec:
        raise NotImplementedError(
            f"The {backend} skill has no `deploy:` block, so SciLink cannot "
            f"load {backend} models yet. Add one to "
            f"skills/{_MLIP_DOMAIN}/{backend}/{backend}.md."
        )

    use_device = _resolve_device(device, spec.get("devices"))
    task = task_name or spec.get("default_task")

    if model and os.path.exists(model):
        trained = spec.get("trained_file")
        if not trained:
            raise NotImplementedError(
                f"The {backend} skill has no `deploy.trained_file` block, so "
                f"trained {backend} model files cannot be loaded yet."
            )
        name = os.path.basename(model)
        ctx = {**(spec.get("defaults") or {}),
               "name": name, "task": task, "model_file": model}
        import_line = trained.get("import_line", spec["import_line"]).format(**ctx)
        construct_expr = trained["construct_expr"].format(**ctx)
        notes = trained.get("notes", "").format(**ctx)
        calc = _build_calculator(import_line, construct_expr, use_device)
        model_file = model
    else:
        wanted = model or spec.get("default_model")
        entry = find_pretrained(backend, wanted)
        if entry is None:
            known = [e.get("name") for e in meta.get("pretrained") or []]
            raise ValueError(
                f"Unknown {backend} model {wanted!r}. Known: {known}"
            )
        name = entry["name"]
        if elements:
            missing = uncovered_elements(backend, name, elements, task)
            if missing:
                raise ValueError(
                    f"{name} was not trained on {missing} (system elements: "
                    f"{sorted(set(elements))}). Choose a model that covers "
                    f"every element."
                )
            if missing is None:
                logger.warning(
                    f"{name} declares no element coverage"
                    f"{f' for task {task!r}' if task else ''}; "
                    f"cannot check {sorted(set(elements))}."
                )
        ctx = {**(spec.get("defaults") or {}), **(entry.get("params") or {}),
               "name": name, "domain": entry.get("domain", ""), "task": task}
        import_line = spec["import_line"].format(**ctx)
        construct_expr = spec["construct_expr"].format(**ctx)
        notes = spec.get("notes", "").format(**ctx)
        calc = _build_calculator(import_line, construct_expr, use_device)
        strategy = entry.get("model_file", spec.get("model_file", "none"))
        if strategy not in _MODEL_FILE_STRATEGIES:
            raise ValueError(f"Unknown model_file strategy {strategy!r} "
                             f"for {name}; use one of "
                             f"{sorted(_MODEL_FILE_STRATEGIES)}")
        model_file = _MODEL_FILE_STRATEGIES[strategy](calc, name, working_dir)

    logger.info(f"Deployed {backend} {name} (device={use_device})")
    return DeployedPotential(
        kind="mlip",
        backend=backend,
        model_name=name,
        model_file=model_file,
        elements=list(elements),
        ase_calculator=ASECalculatorSpec(
            import_line=import_line,
            construct_expr=construct_expr,
            device_env_var=spec["device_env_var"],
        ),
        notes=notes,
    )


# ═══════════════════════════════════════════════════════════════════
#  UNCERTAINTY ESTIMATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_uncertainty(
    backend: str,
    model_file: str,
    structures: List[Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate model uncertainty on a set of structures.

    For committee/ensemble models, uses prediction variance.
    For single models, uses energy-based heuristics.

    Args:
        backend:    MLIP backend name
        model_file: Path to model artifact
        structures: List of ase.Atoms objects

    Returns:
        {
            "per_structure": [
                {
                    "index": int,
                    "energy_uncertainty": float,    # meV/atom
                    "max_force_uncertainty": float,  # meV/Å
                    "is_extrapolation": bool,
                },
                ...
            ],
            "mean_energy_uncertainty": float,
            "max_energy_uncertainty": float,
            "n_extrapolating": int,
            "extrapolation_indices": [int],
        }
    """
    if backend == "mace":
        return _mace_evaluate_uncertainty(model_file, structures, device)
    else:
        raise NotImplementedError(
            f"Uncertainty estimation for {backend} not yet implemented."
        )


def _mace_evaluate_uncertainty(
    model_file: str,
    structures: List[Any],
    device: str,
) -> Dict[str, Any]:
    """
    MACE uncertainty via per-atom energy variance heuristic.

    For true uncertainty, use a committee of models.  This single-model
    approach flags structures where per-atom energies have unusual
    distributions compared to typical bulk configurations.
    """
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=model_file,
        device=device,
        default_dtype="float64",
    )

    per_structure = []
    energy_uncertainties = []

    for i, atoms in enumerate(structures):
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        try:
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()

            # Per-atom energy is available in MACE via the calculator
            e_per_atom = energy / len(atoms_copy)

            # Force magnitude distribution — high max forces suggest
            # the model is in an unfamiliar region
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = float(np.max(force_norms))

            # Heuristic uncertainty: large force variance + extreme
            # per-atom energy → likely extrapolation
            force_std = float(np.std(force_norms))

            # Convert to meV for consistency
            energy_unc = force_std * 1000      # rough proxy
            force_unc = max_force * 1000

            # Flag as extrapolation if max force > 10 eV/Å
            is_extrap = max_force > 10.0

            per_structure.append({
                "index": i,
                "energy_per_atom": float(e_per_atom),
                "energy_uncertainty": energy_unc,
                "max_force": max_force,
                "max_force_uncertainty": force_unc,
                "force_std": force_std,
                "is_extrapolation": is_extrap,
            })
            energy_uncertainties.append(energy_unc)

        except Exception as e:
            logger.warning(f"Uncertainty evaluation failed for structure {i}: {e}")
            per_structure.append({
                "index": i,
                "energy_uncertainty": float("inf"),
                "max_force_uncertainty": float("inf"),
                "is_extrapolation": True,
                "error": str(e),
            })
            energy_uncertainties.append(float("inf"))

    extrap_indices = [s["index"] for s in per_structure if s["is_extrapolation"]]

    return {
        "per_structure": per_structure,
        "mean_energy_uncertainty": float(np.mean(
            [e for e in energy_uncertainties if np.isfinite(e)]
        )) if energy_uncertainties else 0.0,
        "max_energy_uncertainty": float(np.max(
            [e for e in energy_uncertainties if np.isfinite(e)]
        )) if energy_uncertainties else 0.0,
        "n_extrapolating": len(extrap_indices),
        "extrapolation_indices": extrap_indices,
    }


# ═══════════════════════════════════════════════════════════════════
#  TRAINING DATASET
# ═══════════════════════════════════════════════════════════════════

def build_training_dataset(
    structures: List[Any],
    working_dir: str,
    existing_data_path: Optional[str] = None,
    max_structures: int = 500,
    val_fraction: float = 0.1,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Build an extXYZ training dataset.  Backend-agnostic — all backends
    can read extXYZ.

    Returns:
        {
            "train_file": str, "val_file": str,
            "n_train": int, "n_val": int,
            "elements": [str],
            "energy_mean": float, "energy_std": float,
        }
    """
    import ase.io

    os.makedirs(working_dir, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    # Collect frames
    frames = []
    if existing_data_path:
        logger.info(f"Loading dataset from {existing_data_path}")
        frames.extend(ase.io.read(existing_data_path, index=":"))
    for item in (structures or []):
        if isinstance(item, (str, Path)):
            frames.extend(ase.io.read(str(item), index=":"))
        else:
            frames.append(item)

    # Keep only frames with energy + forces
    valid = [
        f for f in frames
        if (f.calc is not None
            and "energy" in (f.calc.results or {})
            and "forces" in (f.calc.results or {}))
    ]
    logger.info(f"Valid frames (with energy+forces): {len(valid)}/{len(frames)}")

    if not valid:
        raise ValueError("No frames with energy+forces found.")

    # Subsample
    if len(valid) > max_structures:
        idx = rng.choice(len(valid), max_structures, replace=False)
        valid = [valid[i] for i in sorted(idx)]

    # Split
    rng.shuffle(valid)
    n_val = max(1, int(val_fraction * len(valid)))
    train, val = valid[n_val:], valid[:n_val]

    train_file = os.path.join(working_dir, "train.xyz")
    val_file = os.path.join(working_dir, "val.xyz")
    ase.io.write(train_file, train, format="extxyz")
    ase.io.write(val_file, val, format="extxyz")

    energies = [f.calc.results["energy"] / len(f) for f in valid]
    elements = sorted({s for f in valid for s in f.get_chemical_symbols()})

    return {
        "train_file": train_file,
        "val_file": val_file,
        "n_train": len(train),
        "n_val": len(val),
        "elements": elements,
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
    }


# ═══════════════════════════════════════════════════════════════════
#  TRAINING  (backend dispatch)
# ═══════════════════════════════════════════════════════════════════

def train(
    backend: str,
    dataset_info: Dict[str, Any],
    working_dir: str,
    foundation_model: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    timeout_hours: float = 12.0,
) -> Dict[str, Any]:
    """
    Train or fine-tune an MLIP.

    Args:
        backend: "mace" | "nequip" | "deepmd"
        dataset_info: Output of build_training_dataset()
        working_dir: Output directory
        foundation_model: If set, fine-tune from this checkpoint
        hyperparameters: Backend-specific training config overrides
        timeout_hours: Hard wall-clock limit

    Returns:
        { "model_file": str, "validation": dict, "status": str }
    """
    hparams = hyperparameters or {}

    if backend == "mace":
        return _mace_train(dataset_info, working_dir, foundation_model,
                           hparams, timeout_hours)
    elif backend == "nequip":
        raise NotImplementedError(
            "NequIP training: implement _nequip_train() using nequip-train CLI "
            "and a YAML config.  The interface mirrors _mace_train()."
        )
    elif backend == "deepmd":
        raise NotImplementedError(
            "DeePMD training: implement _deepmd_train() using dp train CLI "
            "and a JSON config.  The interface mirrors _mace_train()."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _mace_train(
    dataset_info: Dict[str, Any],
    working_dir: str,
    foundation_model: Optional[str],
    hparams: Dict[str, Any],
    timeout_hours: float,
) -> Dict[str, Any]:
    """MACE training via mace_run_train CLI."""
    model_dir = os.path.join(working_dir, "mace_model")
    os.makedirs(model_dir, exist_ok=True)

    model_name = hparams.get("name", "mace_finetuned")

    config = {
        "name":           model_name,
        "train_file":     dataset_info["train_file"],
        "valid_file":     dataset_info["val_file"],
        "model":          "MACE",
        "r_max":          hparams.get("r_max", 5.0),
        "num_channels":   hparams.get("num_channels", 128),
        "max_L":          hparams.get("max_L", 1),
        "correlation":    hparams.get("correlation", 3),
        "max_num_epochs": hparams.get("max_num_epochs", 200),
        "batch_size":     hparams.get("batch_size", 4),
        "lr":             hparams.get("learning_rate", 0.01),
        "energy_weight":  hparams.get("energy_weight", 1.0),
        "forces_weight":  hparams.get("forces_weight", 100.0),
        "ema":            True,
        "ema_decay":      0.99,
        "amsgrad":        True,
        "default_dtype":  "float64",
        "device":         hparams.get("device", "cuda"),
        "save_cpu":       True,
        "results_dir":    model_dir,
        "log_dir":        os.path.join(model_dir, "logs"),
    }

    if foundation_model:
        config["foundation_model"] = foundation_model

    # Write config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Build CLI
    cli = ["mace_run_train"]
    for k, v in config.items():
        if isinstance(v, bool):
            if v:
                cli.append(f"--{k}")
        else:
            cli += [f"--{k}", str(v)]

    stdout_log = os.path.join(config["log_dir"], "stdout.log")
    stderr_log = os.path.join(config["log_dir"], "stderr.log")
    os.makedirs(config["log_dir"], exist_ok=True)

    logger.info(f"Starting MACE training ({model_name})...")

    with open(stdout_log, "w", encoding="utf-8") as out, open(stderr_log, "w", encoding="utf-8") as err:
        proc = run_engine(
            cli, stdout=out, stderr=err,
            cwd=working_dir,
            timeout=int(timeout_hours * 3600),
        )

    if proc.returncode != 0:
        with open(stderr_log) as f:
            tail = "".join(f.readlines()[-40:])
        raise RuntimeError(f"MACE training failed (exit {proc.returncode}):\n{tail}")

    model_file = os.path.join(model_dir, f"{model_name}.model")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found after training: {model_file}")

    return {"model_file": model_file, "config": config, "status": "success"}


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION  (backend dispatch)
# ═══════════════════════════════════════════════════════════════════

def validate_model(
    backend: str,
    model_file: str,
    val_file: str,
    working_dir: str,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """
    Compute energy/force MAE on held-out validation data.

    Returns:
        {
            "energy_mae_meV": float,
            "force_mae_meV_A": float,
            "max_force_error_meV_A": float,
            "n_evaluated": int,
            "passed": bool,
        }
    """
    if backend == "mace":
        return _mace_validate(model_file, val_file, working_dir, n_samples)
    else:
        raise NotImplementedError(
            f"Validation for {backend} not yet implemented."
        )


def _mace_validate(
    model_file: str, val_file: str,
    working_dir: str, n_samples: int,
) -> Dict[str, Any]:
    """Validate a MACE model on extXYZ reference data."""
    import ase.io
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=model_file, device="cpu", default_dtype="float64"
    )
    frames = list(ase.io.read(val_file, index=":"))[:n_samples]

    e_err, f_err = [], []

    for frame in frames:
        ref_e = frame.calc.results["energy"] / len(frame)
        ref_f = frame.calc.results["forces"]

        test = frame.copy()
        test.calc = calc
        pred_e = test.get_potential_energy() / len(test)
        pred_f = test.get_forces()

        e_err.append(abs(pred_e - ref_e) * 1000)
        f_err.extend(np.linalg.norm(pred_f - ref_f, axis=1) * 1000)

    energy_mae = float(np.mean(e_err))
    force_mae = float(np.mean(f_err))
    max_f_err = float(np.max(f_err))

    passed = energy_mae < 5.0 and force_mae < 100.0

    return {
        "energy_mae_meV": energy_mae,
        "force_mae_meV_A": force_mae,
        "max_force_error_meV_A": max_f_err,
        "n_evaluated": len(frames),
        "passed": passed,
    }


# ═══════════════════════════════════════════════════════════════════
#  DFT INPUT GENERATION
# ═══════════════════════════════════════════════════════════════════

def extract_problematic_frames(
    trajectory_file: str,
    model_file: str,
    backend: str = "mace",
    max_frames: int = 500,
    top_n: int = 20,
    device: str = "cpu",
) -> List[Any]:
    """
    Read a trajectory, score each frame by uncertainty, return the
    top-N most uncertain frames as ase.Atoms objects.

    These are the frames that should be sent to DFT for active learning.
    """
    import ase.io

    frames = list(ase.io.read(trajectory_file, index=":"))
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step]

    unc = evaluate_uncertainty(backend, model_file, frames, device)

    # Sort by uncertainty descending
    scored = sorted(
        zip(unc["per_structure"], frames),
        key=lambda x: x[0].get("max_force_uncertainty", 0),
        reverse=True,
    )

    selected = [frame for _, frame in scored[:top_n]]
    logger.info(
        f"Selected {len(selected)} high-uncertainty frames "
        f"from {len(frames)} trajectory frames"
    )
    return selected


def write_dft_inputs(
    structures: List[Any],
    working_dir: str,
    dft_code: str = "vasp",
    dft_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Write DFT input files for a list of structures.

    Supports VASP, CP2K, and a generic extXYZ dump for other codes.

    Returns:
        {
            "dft_code": str,
            "directories": [str],    # one per structure
            "n_structures": int,
            "instructions": str,     # human-readable next steps
        }
    """
    import ase.io

    os.makedirs(working_dir, exist_ok=True)
    settings = dft_settings or {}
    directories = []

    for i, atoms in enumerate(structures):
        calc_dir = os.path.join(working_dir, f"frame_{i:04d}")
        os.makedirs(calc_dir, exist_ok=True)

        if dft_code == "vasp":
            _write_vasp_inputs(atoms, calc_dir, settings)
        elif dft_code == "cp2k":
            _write_cp2k_inputs(atoms, calc_dir, settings)
        else:
            # Generic: just write the structure
            ase.io.write(
                os.path.join(calc_dir, "structure.xyz"),
                atoms, format="extxyz",
            )

        directories.append(calc_dir)

    instructions = {
        "vasp": (
            f"Run VASP in each of the {len(directories)} directories. "
            f"After completion, collect OUTCAR files and convert to extXYZ "
            f"with: ase convert */OUTCAR collected_dft.xyz"
        ),
        "cp2k": (
            f"Run CP2K in each of the {len(directories)} directories. "
            f"After completion, parse forces from output files."
        ),
    }.get(dft_code, (
        f"Run your DFT code on the {len(directories)} structures in "
        f"{working_dir}/frame_XXXX/structure.xyz. "
        f"Collect results into a single extXYZ file with energy and forces."
    ))

    return {
        "dft_code": dft_code,
        "directories": directories,
        "n_structures": len(structures),
        "instructions": instructions,
    }


def _write_vasp_inputs(atoms, calc_dir, settings):
    """Write POSCAR + template INCAR + KPOINTS for a single structure."""
    import ase.io

    ase.io.write(os.path.join(calc_dir, "POSCAR"), atoms, format="vasp")

    encut = settings.get("encut", 520)
    kpoints = settings.get("kpoints", [3, 3, 3])

    incar = f"""SYSTEM = MLIP active learning
ENCUT = {encut}
PREC = Accurate
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
IBRION = -1
NSW = 0
LREAL = Auto
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
    with open(os.path.join(calc_dir, "INCAR"), "w", encoding="utf-8") as f:
        f.write(incar)

    with open(os.path.join(calc_dir, "KPOINTS"), "w", encoding="utf-8") as f:
        f.write(f"Automatic\n0\nGamma\n{kpoints[0]} {kpoints[1]} {kpoints[2]}\n0 0 0\n")


def _write_cp2k_inputs(atoms, calc_dir, settings):
    """Write structure + template CP2K input."""
    import ase.io

    ase.io.write(os.path.join(calc_dir, "structure.xyz"), atoms, format="xyz")
    # Minimal template — user should customize
    with open(os.path.join(calc_dir, "cp2k.inp"), "w", encoding="utf-8") as f:
        f.write("# CP2K input template — customize for your system\n")
        f.write("# Structure: structure.xyz\n")

