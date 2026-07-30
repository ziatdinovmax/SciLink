"""
Backend-agnostic tools for ML interatomic potentials.

Design: every public function accepts a `backend` parameter and dispatches
to private backend-specific implementations.  This keeps the agent code
clean while making it easy to add NequIP / DeePMD / Allegro later without
touching the agent.

The heavy MACE coverage reflects reality: MACE currently has the only
production-ready foundation model ecosystem.  NequIP and DeePMD dispatch
points raise NotImplementedError with actionable messages so failures are
obvious rather than silent.
"""

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
#  BACKEND AVAILABILITY
# ═══════════════════════════════════════════════════════════════════

def _probe_mace() -> Dict[str, Any]:
    try:
        import mace
        return {
            "available": True,
            "version": getattr(mace, "__version__", "unknown"),
            "lammps_pair_style": "mace",
            "pretrained": [
                {
                    "name": "mace-mp-0",
                    "domain": "inorganic",
                    "description": (
                        "Universal potential trained on MPtrj dataset "
                        "(~150k inorganic materials). Good for metals, "
                        "oxides, ceramics."
                    ),
                },
                {
                    "name": "mace-mp-0b",
                    "domain": "inorganic",
                    "description": "Medium-accuracy variant, faster inference.",
                },
                {
                    "name": "mace-off23",
                    "domain": "organic",
                    "description": (
                        "Trained on SPICE dataset. Covers organic molecules, "
                        "peptides, drug-like compounds."
                    ),
                },
            ],
        }
    except ImportError:
        return {"available": False, "version": None, "pretrained": []}


def _probe_nequip() -> Dict[str, Any]:
    try:
        import nequip
        return {
            "available": True,
            "version": getattr(nequip, "__version__", "unknown"),
            "lammps_pair_style": "nequip",
            "pretrained": [],
        }
    except ImportError:
        return {"available": False, "version": None, "pretrained": []}


def _probe_deepmd() -> Dict[str, Any]:
    try:
        import deepmd
        return {
            "available": True,
            "version": getattr(deepmd, "__version__", "unknown"),
            "lammps_pair_style": "deepmd",
            "pretrained": [],
        }
    except ImportError:
        return {"available": False, "version": None, "pretrained": []}


def _probe_chgnet() -> Dict[str, Any]:
    # ASE-only — no LAMMPS pair_style.
    try:
        import chgnet
        return {
            "available": True,
            "version": getattr(chgnet, "__version__", "unknown"),
            "lammps_pair_style": None,
            "pretrained": [
                {
                    "name": "chgnet",
                    "domain": "universal-inorganic",
                    "description": (
                        "Universal MLIP pretrained on MPtrj (~1.5M DFT "
                        "relaxations). Predicts energies, forces, "
                        "stresses, and magnetic moments. ASE-only — "
                        "no LAMMPS pair_style; use runner='ase'."
                    ),
                },
            ],
        }
    except ImportError:
        return {"available": False, "version": None, "pretrained": []}


def _probe_uma() -> Dict[str, Any]:
    try:
        import fairchem.core
        return {
            "available": True,
            "version": getattr(fairchem.core, "__version__", "unknown"),
            "lammps_pair_style": None,
            "pretrained": [
                {
                    "name": "uma-s-1p2",
                    "domain": "universal",
                    "description": "UMA small model — fastest, good for screening.",
                },
                {
                    "name": "uma-m-1p1",
                    "domain": "universal",
                    "description": "UMA medium model — best accuracy.",
                },
            ],
        }
    except ImportError:
        return {"available": False, "version": None, "pretrained": []}


def _probe_orb() -> Dict[str, Any]:
    try:
        import orb_models
        return {
            "available": True,
            "version": getattr(orb_models, "__version__", "unknown"),
            "lammps_pair_style": None,
            "pretrained": [
                {
                    "name": "orb-v3-conservative-inf-omat",
                    "domain": "universal-inorganic",
                    "description": "Orb v3 conservative model — fast universal MLIP.",
                },
            ],
        }
    except ImportError:
        return {"available": False, "version": None, "pretrained": []}


_BACKEND_PROBES: Dict[str, Any] = {
    "mace":    _probe_mace,
    "nequip":  _probe_nequip,
    "deepmd":  _probe_deepmd,
    "chgnet":  _probe_chgnet,
    "uma":     _probe_uma,
    "orb":     _probe_orb,
}


def check_backends(
    backends: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Check which MLIP backends are installed.

    Args:
        backends: Names of backends to probe. When ``None`` (default) all
            known backends are probed. Pass a single-element list, e.g.
            ``["mace"]``, to probe only that backend and avoid importing
            unrelated packages.

    Returns:
        Dict keyed by backend name:
        ``{ "mace": {"available": bool, "version": str, "pretrained": [...]}, ... }``
    """
    targets = backends if backends is not None else list(_BACKEND_PROBES)
    return {name: _BACKEND_PROBES[name]() for name in targets if name in _BACKEND_PROBES}


# ═══════════════════════════════════════════════════════════════════
#  MODEL DEPLOYMENT  (the default path)
# ═══════════════════════════════════════════════════════════════════

def deploy(
    backend: str,
    model: str,
    elements: List[str],
    working_dir: str,
    device: str = "cpu",
) -> DeployedPotential:
    """
    Deploy an MLIP model as an engine-neutral DeployedPotential.

    ``model`` is a model *identifier* — each backend's deploy function
    decides whether it's a foundation-model keyword (``"mace-mp-0"``,
    ``"chgnet"``) or an on-disk path to a trained/fine-tuned model
    (the output of ``train()`` / ``fine_tune()``). This single entry
    point covers both cases so adding a backend touches exactly one
    deploy function — no separate pretrained/trained code paths.

    Args:
        backend:     "mace" | "chgnet" | "nequip" | "deepmd"
        model:       Foundation-model keyword or path to a trained model
        elements:    Chemical elements in the system
        working_dir: Output directory
        device:      "cpu" or "cuda"

    Returns:
        A DeployedPotential descriptor — the engine-neutral contract
        MDSimulationAgent consumes to generate the actual run. See
        agents/sim_agents/_potential.py.
    """
    os.makedirs(working_dir, exist_ok=True)

    if backend == "mace":
        return _mace_deploy(model, elements, working_dir, device)
    elif backend == "chgnet":
        return _chgnet_deploy(model, elements, working_dir, device)
    elif backend in ("nequip", "deepmd"):
        # These have no foundation models. A trained model file is the
        # only way to deploy them — but the deploy function (with its
        # ASECalculatorSpec) isn't written yet.
        if os.path.exists(model):
            raise NotImplementedError(
                f"_{backend}_deploy not yet implemented — add it here "
                f"with an ASECalculatorSpec, mirroring _mace_deploy."
            )
        raise NotImplementedError(
            f"{backend} has no foundation models; pass a trained model "
            f"file. Use backend='mace' for pretrained deployment."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _chgnet_deploy(
    model: str,
    elements: List[str],
    working_dir: str,
    device: str,
) -> DeployedPotential:
    """Deploy a CHGNet model — bundled universal model or a trained file.

    CHGNet ships a single universal model with the package; the common
    case (``model`` is the ``"chgnet"`` keyword) deploys that with no
    file on disk (``model_file=""``). If ``model`` is a path to a
    trained CHGNet checkpoint, that file is loaded instead. Either way
    the calculator is constructed once here to validate the install,
    then discarded — the run script reconstructs it from the
    ASECalculatorSpec.
    """
    from chgnet.model.dynamics import CHGNetCalculator

    use_device = device if device in ("cuda", "cpu", "mps") else "cpu"
    is_file = bool(model) and os.path.exists(model)

    if is_file:
        from chgnet.model import CHGNet
        CHGNetCalculator(model=CHGNet.from_file(model), use_device=use_device)
        construct_expr = (
            f"CHGNetCalculator(model=__import__('chgnet.model', "
            f"fromlist=['CHGNet']).CHGNet.from_file({model!r}), "
            f"use_device=DEVICE)"
        )
        model_file, model_name = model, os.path.basename(model)
        notes = f"Trained CHGNet model ({model_name}). ASE-only — no LAMMPS pair_style."
    else:
        # Construct-and-discard: validates chgnet is importable and the
        # bundled weights load. The generated script rebuilds it.
        CHGNetCalculator(use_device=use_device)
        construct_expr = "CHGNetCalculator(use_device=DEVICE)"
        model_file, model_name = "", model or "chgnet"
        notes = (
            "CHGNet universal MPtrj-pretrained model. ASE-only — no "
            "LAMMPS pair_style; the MD agent must use the ASE runner."
        )

    logger.info(f"Deployed CHGNet {model_name} (ASE-only, device={use_device})")
    return DeployedPotential(
        kind="mlip",
        backend="chgnet",
        model_name=model_name,
        model_file=model_file,
        elements=list(elements),
        ase_calculator=ASECalculatorSpec(
            import_line="from chgnet.model.dynamics import CHGNetCalculator",
            construct_expr=construct_expr,
            device_env_var="CHGNET_DEVICE",
        ),
        notes=notes,
    )


# MACE foundation-model keywords -> (loader, API size keyword, domain).
_MACE_FOUNDATION_MODELS = {
    # Bare backend name -> the default MACE foundation model. Lets
    # deploy("mace", model="mace", ...) work, symmetric with CHGNet
    # (whose only model is also named after the backend) — this is the
    # path taken when a caller forces backend="mace" without naming a
    # specific model.
    "mace":             ("mace_mp", "medium", "inorganic"),
    "mace-mp-0":        ("mace_mp", "medium", "inorganic"),
    "mace-mp-0b":       ("mace_mp", "small",  "inorganic"),
    "mace-mp-0-large":  ("mace_mp", "large",  "inorganic"),
    "small":            ("mace_mp", "small",  "inorganic"),
    "medium":           ("mace_mp", "medium", "inorganic"),
    "large":            ("mace_mp", "large",  "inorganic"),
    "mace-off23":       ("mace_off", "medium", "organic"),
    "mace-off23-small": ("mace_off", "small",  "organic"),
    "mace-off23-large": ("mace_off", "large",  "organic"),
}


def _mace_deploy(
    model: str,
    elements: List[str],
    working_dir: str,
    device: str,
) -> DeployedPotential:
    """Deploy a MACE model — foundation keyword or trained file.

    If ``model`` is a path to a trained/fine-tuned model on disk it is
    loaded via ``MACECalculator(model_paths=...)``. Otherwise it's
    treated as a foundation-model keyword resolved through the
    ``mace_mp`` / ``mace_off`` loaders. In both cases the calculator is
    constructed once to validate the install and locate the on-disk
    model file (needed for the LAMMPS pair_coeff), and a
    DeployedPotential is returned whose ASECalculatorSpec lets the run
    script reconstruct the same calculator.
    """
    if model and os.path.exists(model):
        # Trained / fine-tuned model file.
        from mace.calculators import MACECalculator
        MACECalculator(model_paths=[model], device=device,
                       default_dtype="float64")
        logger.info(f"Deployed trained MACE model → {model}")
        return DeployedPotential(
            kind="mlip",
            backend="mace",
            model_name=os.path.basename(model),
            model_file=model,
            elements=list(elements),
            ase_calculator=ASECalculatorSpec(
                import_line="from mace.calculators import MACECalculator",
                construct_expr=(
                    f"MACECalculator(model_paths=[{model!r}], "
                    f"device=DEVICE, default_dtype='float64')"
                ),
                device_env_var="MACE_DEVICE",
            ),
            notes=(
                f"Trained MACE model ({os.path.basename(model)}). "
                f"LAMMPS-capable via pair_style mliap unified."
            ),
        )

    # Foundation-model keyword.
    from mace.calculators import mace_mp, mace_off

    loader, mace_size, domain = _MACE_FOUNDATION_MODELS.get(
        model, ("mace_off" if "off" in model.lower() else "mace_mp", model,
                "organic" if "off" in model.lower() else "inorganic")
    )
    loaders = {"mace_mp": mace_mp, "mace_off": mace_off}
    calc = loaders[loader](model=mace_size, device=device,
                           default_dtype="float64")

    # Locate the cached model file (needed for the LAMMPS pair_coeff).
    model_file = None
    for attr in ("model_path", "model_paths"):
        val = getattr(calc, attr, None)
        if val:
            model_file = str(val) if not isinstance(val, list) else str(val[0])
            break

    if model_file is None or not os.path.exists(model_file):
        model_file = os.path.join(working_dir, f"{model}.model")
        try:
            import torch
            torch.save(calc.models[0], model_file)
        except Exception as e:
            logger.warning(f"Could not save model file: {e}")
            model_file = model

    logger.info(
        f"Deployed pretrained {model} → {mace_size} ({domain}) → {model_file}"
    )

    return DeployedPotential(
        kind="mlip",
        backend="mace",
        model_name=model,
        model_file=model_file,
        elements=list(elements),
        ase_calculator=ASECalculatorSpec(
            import_line=f"from mace.calculators import {loader}",
            construct_expr=(
                f"{loader}(model={mace_size!r}, device=DEVICE, "
                f"default_dtype='float64')"
            ),
            device_env_var="MACE_DEVICE",
        ),
        notes=(
            f"MACE {model} ({mace_size}, {domain}). LAMMPS-capable "
            f"via pair_style mliap unified."
        ),
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
    with open(config_path, "w") as f:
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

    with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
        proc = subprocess.run(
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
    with open(os.path.join(calc_dir, "INCAR"), "w") as f:
        f.write(incar)

    with open(os.path.join(calc_dir, "KPOINTS"), "w") as f:
        f.write(f"Automatic\n0\nGamma\n{kpoints[0]} {kpoints[1]} {kpoints[2]}\n0 0 0\n")


def _write_cp2k_inputs(atoms, calc_dir, settings):
    """Write structure + template CP2K input."""
    import ase.io

    ase.io.write(os.path.join(calc_dir, "structure.xyz"), atoms, format="xyz")
    # Minimal template — user should customize
    with open(os.path.join(calc_dir, "cp2k.inp"), "w") as f:
        f.write("# CP2K input template — customize for your system\n")
        f.write("# Structure: structure.xyz\n")

