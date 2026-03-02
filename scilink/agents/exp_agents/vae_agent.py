"""
VAEAgent — LLM-powered Variational Autoencoder training and analysis agent.

Follows all five principles from scilink_dev_manifesto.md:
  1. Stateful Memory    - every atomic action is logged and persisted to JSON.
  2. Atomic Tooling     - capabilities are discrete, composable methods.
  3. Standardised I/O   - typed input/output dicts with explicit units/context.
  4. Human-in-the-Loop  - dry-run and enable_human_feedback hooks.
  5. Decoupled          - imports only from scilink.tools; no exp-agent domain logic.
"""

import json
import logging
import math
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scilink.agents.exp_agents.base_agent import BaseUtilityAgent
from scilink.tools.nn_tools import Trainer, TrainerConfig
from scilink.tools.vae_tools import VAE_REGISTRY, VAE_REGISTRY_DESC, get_latent_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_tensor(data: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a float32 torch Tensor."""
    return torch.from_numpy(data.astype(np.float32))


def _build_dataloader(data: np.ndarray, batch_size: int,
                      val_fraction: float = 0.1, seed: int = 42
                      ) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train and optional validation DataLoaders from a numpy array.

    Each sample is one row / image slice; the target is the input itself
    (unsupervised reconstruction).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    n_val = max(1, int(len(data) * val_fraction)) if len(data) > 10 else 0

    train_idx = idx[n_val:]
    val_idx   = idx[:n_val]

    t = _as_tensor(data)
    train_ds = TensorDataset(t[train_idx])
    val_ds   = TensorDataset(t[val_idx]) if n_val > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False) if val_ds else None
    return train_loader, val_loader


def _safe_float(v) -> Optional[float]:
    """Convert tensor / numeric to Python float, returning None on failure."""
    try:
        if torch.is_tensor(v):
            return float(v.detach().cpu().item())
        return float(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# VAEAgent
# ---------------------------------------------------------------------------

class VAEAgent(BaseUtilityAgent):
    """
    LLM-powered agent that selects, trains, monitors, and analyses
    VAE/AE models from the SciLink VAE model registry.

    Atomic tools (Principle 2):
        select_model()           - LLM chooses model type from data shape + task
        configure_hyperparameters() - LLM proposes training config
        train_vae()              - builds DataLoader, trains model, returns history
        analyze_latent_space()   - computes per-dim latent statistics
        detect_training_failure() - rule-based failure diagnosis
        adjust_hyperparameters() - LLM proposes improved config after failure

    Main entry point:
        analyze()  - orchestrates the above tools with up to max_retries attempts
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-06-05",
        base_url: Optional[str] = None,
        output_dir: str = "vae_output",
        enable_human_feedback: bool = False,
        # Deprecated pass-throughs kept for API consistency
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            output_dir=output_dir,
            google_api_key=google_api_key,
            local_model=local_model,
        )

        self.agent_type = "vae"
        self.enable_human_feedback = enable_human_feedback

        # Create output sub-directories
        self._checkpoint_dir  = self.output_dir / "checkpoints"
        self._latent_viz_dir  = self.output_dir / "latent_viz"
        self._reports_dir     = self.output_dir / "reports"
        for d in (self._checkpoint_dir, self._latent_viz_dir, self._reports_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Initialise persistent state (Principle 1)
        self._init_state()

    # =========================================================================
    # Principle 1 — State helpers (mirror planning_agents.BaseAgent pattern)
    # =========================================================================

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        return {
            "action_history":       [],
            "model_type":           None,
            "training_attempts":    [],
            "best_metrics":         {},
            "latent_space_summary": {},
            "failure_history":      [],
        }

    def _log_action(self, action: str, input_ctx: Dict[str, Any],
                    result: Dict[str, Any], rationale: Optional[str] = None,
                    feedback: Optional[str] = None) -> None:
        """Append an action record to state and persist to disk."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action":    action,
            "input":     input_ctx,
            "rationale": rationale,
            "result":    {
                "status":    result.get("status", "completed"),
                "error":     result.get("error"),
                "iteration": result.get("iteration"),
                "stage":     result.get("stage"),
            },
            "feedback": feedback,
        }
        if "action_history" not in self.state:
            self.state["action_history"] = []
        self.state["action_history"].append(entry)
        self._save_state()

    def _save_state(self) -> None:
        """Persist state to <output_dir>/vae_state.json."""
        state_file = self.output_dir / "vae_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save VAE state: {e}")

    # =========================================================================
    # Principle 2 — Atomic tools
    # =========================================================================

    def select_model(self, data_shape: Tuple[int, ...],
                     task_description: str = "") -> Dict[str, Any]:
        """
        Tool 1 — Ask the LLM to select the most appropriate model type.

        Args:
            data_shape:        Shape of a single sample, e.g. (128,) or (32, 32).
            task_description:  Free-text description of the scientific task.

        Returns:
            {"model_type": str, "rationale": str}
            model_type is one of: linear_ae, conv_ae, linear_vae, conv_vae
        """
        ndim = len(data_shape)
        spatial = "1-D (spectral / signal)" if ndim == 1 else f"{ndim}-D (image/map)"
        available_model_desc = '\n'.join([f'- {k} : {v}' for k,v in VAE_REGISTRY_DESC.items()])

        prompt = f"""You are an expert in machine learning for experimental science.
Select the most appropriate VAE model for the following data.

## Available models
  {available_model_desc}

## Data
- Shape (single sample): {data_shape}
- Data type: {spatial}

## Task description
{task_description if task_description else "(not specified)"}

## Decision rules
- 1-D data  → prefer linear variants; 2-D data → prefer conv variants.
- Exploration / clustering / generation → variational; pure reconstruction → vanilla.

Respond in JSON only:
{{"model_type": "<one of the four keys above>", "rationale": "<one sentence>"}}"""

        result_json, error = self._generate_json_from_text_parts([prompt])

        if error or not result_json:
            # Safe fallback: choose based on dimensionality
            fallback = "linear_vae" if ndim == 1 else "conv_vae"
            self.logger.warning(
                f"LLM model selection failed ({error}). Defaulting to '{fallback}'."
            )
            result_json = {"model_type": fallback,
                           "rationale": "LLM unavailable; fallback by data dimensionality."}

        if result_json.get("model_type") not in VAE_REGISTRY:
            result_json["model_type"] = "linear_vae" if ndim == 1 else "conv_vae"

        self._log_action(
            action="select_model",
            input_ctx={"data_shape": list(data_shape), "task": task_description},
            result={"status": "completed", "model_type": result_json["model_type"]},
            rationale=result_json.get("rationale"),
        )
        return result_json

    # ------------------------------------------------------------------

    def configure_hyperparameters(self, model_type: str,
                                  data_shape: Tuple[int, ...],
                                  n_samples: int) -> Dict[str, Any]:
        """
        Tool 2 — Ask the LLM to propose training hyperparameters.

        Returns a HyperparamConfig dict used by train_vae().
        """
        is_conv       = "conv" in model_type
        is_variational = "vae" in model_type

        prompt = f"""You are an expert in training VAEs for scientific microscopy data.
Propose training hyperparameters for the following configuration.

## Model type: {model_type}
## Data
- Single-sample shape: {list(data_shape)}
- Number of training samples: {n_samples}
- Architecture: {"convolutional" if is_conv else "fully-connected"}
- Variational (has KL term): {is_variational}

## Constraints
- Prefer latent_dims in [4, 8, 16, 32] (start small).
- Prefer batch_size in [16, 32, 64] unless n_samples < 50 (then use 8).
- Start with lr=1e-3; use gradient_clip_val=1.0 for stability.
- max_epochs should be enough for convergence: 50-200 for small data, 20-50 for large.
- kld_weight (variational only): start at 1.0; lower to 0.1 if collapse is likely.

Respond in JSON only (no markdown):
{{
  "latent_dims": <int>,
  "lr": <float>,
  "weight_decay": <float>,
  "kld_weight": <float or null if non-variational>,
  "max_epochs": <int>,
  "batch_size": <int>,
  "gradient_clip_val": <float or null>,
  "encoder_configs": null,
  "decoder_configs": null,
  "rationale": "<one sentence>"
}}"""

        result_json, error = self._generate_json_from_text_parts([prompt])

        if error or not result_json:
            self.logger.warning("LLM hyperparameter config failed. Using sensible defaults.")
            result_json = {
                "latent_dims":       8 if not is_conv else 16,
                "lr":                1e-3,
                "weight_decay":      0.0,
                "kld_weight":        1.0 if is_variational else None,
                "max_epochs":        50,
                "batch_size":        min(32, max(8, n_samples // 8)),
                "gradient_clip_val": 1.0,
                "encoder_configs":   None,
                "decoder_configs":   None,
                "rationale":         "Defaults due to LLM failure.",
            }

        # Ensure numeric types are correct
        result_json["latent_dims"] = int(result_json.get("latent_dims", 8))
        result_json["lr"]          = float(result_json.get("lr", 1e-3))
        result_json["max_epochs"]  = int(result_json.get("max_epochs", 50))
        result_json["batch_size"]  = int(result_json.get("batch_size", 32))

        self._log_action(
            action="configure_hyperparameters",
            input_ctx={"model_type": model_type, "n_samples": n_samples},
            result={"status": "completed", **{k: v for k, v in result_json.items()
                                              if k != "rationale"}},
            rationale=result_json.get("rationale"),
        )
        return result_json

    # ------------------------------------------------------------------

    def train_vae(self, data: np.ndarray, model_type: str,
                  hyperparam_config: Dict[str, Any],
                  attempt: int = 1) -> Dict[str, Any]:
        """
        Tool 3 — Instantiate, train, and return the model with full history.

        Args:
            data:             numpy array of shape (N, ...) — one sample per row.
            model_type:       key in VAE_REGISTRY.
            hyperparam_config: dict from configure_hyperparameters().
            attempt:          training attempt index (used for checkpoint naming).

        Returns:
            {
                "status": "success|error",
                "train_history": list of per-epoch metric dicts,
                "final_metrics": dict,
                "model": nn.Module,
                "trainer": Trainer,
                "model_path": str,
            }
        """
        in_dims: Tuple[int, ...] = data.shape[1:]
        latent_dims    = hyperparam_config["latent_dims"]
        lr             = hyperparam_config["lr"]
        weight_decay   = hyperparam_config.get("weight_decay", 0.0)
        kld_weight     = hyperparam_config.get("kld_weight") or 1.0
        batch_size     = hyperparam_config["batch_size"]
        max_epochs     = hyperparam_config["max_epochs"]
        clip_val       = hyperparam_config.get("gradient_clip_val")
        enc_cfgs       = hyperparam_config.get("encoder_configs")
        dec_cfgs       = hyperparam_config.get("decoder_configs")

        model_cls = VAE_REGISTRY[model_type]
        is_variational = "vae" in model_type

        # Build model kwargs
        kwargs: Dict[str, Any] = dict(
            in_dims=in_dims,
            latent_dims=latent_dims,
            lr=lr,
            weight_decay=weight_decay,
        )
        if enc_cfgs:
            kwargs["encoder_configs"] = enc_cfgs
        if dec_cfgs:
            kwargs["decoder_configs"] = dec_cfgs
        if is_variational:
            kwargs["kld_weight"] = kld_weight

        try:
            model = model_cls(**kwargs)
        except Exception as e:
            self.logger.error(f"Model instantiation failed: {e}")
            return {"status": "error", "error": str(e)}

        train_loader, val_loader = _build_dataloader(data, batch_size)

        ckpt_dir = str(self._checkpoint_dir / f"attempt_{attempt}")
        os.makedirs(ckpt_dir, exist_ok=True)

        trainer_cfg = TrainerConfig(
            max_epochs=max_epochs,
            batch_size=batch_size,
            gradient_clip_val=clip_val,
            checkpoint_dir=ckpt_dir,
            checkpoint_monitor="val_loss" if val_loader else "loss",
            checkpoint_mode="min",
            log_every_n_steps=max(1, len(train_loader) // 2),
            seed=42,
        )

        # Provide a seed optimizer; configure_optimizers() will override it in fit()
        init_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model_path = str(self._checkpoint_dir / f"vae_attempt_{attempt}")
        trainer = Trainer(trainer_cfg, optimizer=init_optimizer, save_path=model_path)

        try:
            train_history = trainer.fit(model, train_loader, val_loader)
        except Exception as e:
            self.logger.error(f"Training failed at attempt {attempt}: {e}")
            return {"status": "error", "error": str(e)}

        final_metrics = train_history[-1] if train_history else {}

        self._log_action(
            action="train_vae",
            input_ctx={"model_type": model_type, "attempt": attempt,
                       "n_samples": len(data), "in_dims": list(in_dims)},
            result={"status": "success", "final_metrics": {
                k: _safe_float(v) for k, v in final_metrics.items()
            }},
            rationale=f"Training attempt {attempt} of {model_type}.",
        )

        return {
            "status":        "success",
            "train_history": train_history,
            "final_metrics": final_metrics,
            "model":         model,
            "trainer":       trainer,
            "model_path":    model_path + ".pt",
        }

    # ------------------------------------------------------------------

    def analyze_latent_space(self, model, data: np.ndarray,
                             batch_size: int = 64) -> Dict[str, Any]:
        """
        Tool 4 — Compute per-dimension latent space statistics and ask the LLM
        to provide a scientific interpretation.

        Returns:
            {
                "active_dims": int,
                "kld_per_dim": list | None,
                "mu_mean": list,
                "mu_std": list,
                "interpretation": str,
            }
        """
        device = next(model.parameters()).device
        data_loader = DataLoader(
            TensorDataset(_as_tensor(data)),
            batch_size=batch_size,
            shuffle=False,
        )

        stats = get_latent_stats(model, data_loader, device)

        # LLM interpretation
        kld_info = ""
        if stats["kld_per_dim"] is not None:
            kld_rounded = [round(v, 4) for v in stats["kld_per_dim"]]
            kld_info = f"\nKLD per latent dim: {kld_rounded}"

        prompt = f"""You are a materials scientist analysing the latent space of a
trained VAE applied to microscopy / experimental data.

## Latent space statistics
- Latent dimensions: {len(stats['mu_mean'])}
- Active dims (std > 0.1): {stats['active_dims']}
- Per-dim mean: {[round(v, 4) for v in stats['mu_mean']]}
- Per-dim std:  {[round(v, 4) for v in stats['mu_std']]}{kld_info}

Provide a concise scientific interpretation (2-3 sentences):
1. Comment on how many dimensions are truly active / informative.
2. If KLD per dim is available, identify which dims are collapsed (KLD≈0).
3. Suggest what the active dimensions might encode (shape, size, orientation, etc.).

Respond in JSON: {{"interpretation": "<2-3 sentence interpretation>"}}"""

        result_json, _ = self._generate_json_from_text_parts([prompt])
        interpretation = (result_json or {}).get(
            "interpretation",
            f"{stats['active_dims']} of {len(stats['mu_mean'])} latent dims are active."
        )

        result = {**stats, "interpretation": interpretation}

        self._log_action(
            action="analyze_latent_space",
            input_ctx={"n_samples": len(data), "latent_dims": len(stats["mu_mean"])},
            result={"status": "completed", "active_dims": stats["active_dims"]},
            rationale="Post-training latent space characterisation.",
        )
        return result

    # ------------------------------------------------------------------

    def detect_training_failure(self, train_history: List[Dict],
                                latent_stats: Optional[Dict],
                                model_type: str) -> Dict[str, Any]:
        """
        Tool 5 — Rule-based failure diagnosis.

        Checks:
            numerical_instability  - NaN/Inf in any loss entry
            latent_collapse        - mean KLD across all dims < 0.001
            posterior_collapse     - >50% of dims have kld_per_dim[i] < 0.01
            divergence             - final loss > 90% of initial loss (not converging)
            stagnation             - <0.1% improvement in the last 20% of epochs

        Returns:
            {"failure_type": str | None, "severity": "ok|warning|critical",
             "details": dict}
        """
        if not train_history:
            return {"failure_type": "no_history", "severity": "critical",
                    "details": {"reason": "Empty training history."}}

        is_variational = "vae" in model_type

        # -- Numerical instability
        for rec in train_history:
            for k, v in rec.items():
                if k == "epoch":
                    continue
                fv = _safe_float(v)
                if fv is None or math.isnan(fv) or math.isinf(fv):
                    return {"failure_type": "numerical_instability",
                            "severity": "critical",
                            "details": {"bad_key": k, "value": str(v)}}

        # -- Latent collapse (variational only)
        if is_variational and latent_stats and latent_stats.get("kld_per_dim") is not None:
            kld_vals = latent_stats["kld_per_dim"]
            mean_kld = sum(kld_vals) / len(kld_vals) if kld_vals else 0.0

            if mean_kld < 0.001:
                return {"failure_type": "latent_collapse",
                        "severity": "critical",
                        "details": {"mean_kld": mean_kld, "kld_per_dim": kld_vals}}

            n_dead = sum(1 for v in kld_vals if v < 0.01)
            if n_dead / max(len(kld_vals), 1) > 0.5:
                return {"failure_type": "posterior_collapse",
                        "severity": "warning",
                        "details": {"dead_dims": n_dead,
                                    "total_dims": len(kld_vals),
                                    "kld_per_dim": kld_vals}}

        # -- Divergence: final loss > 90% of initial loss
        loss_key = "loss"
        initial_loss = _safe_float(train_history[0].get(loss_key))
        final_loss   = _safe_float(train_history[-1].get(loss_key))
        if initial_loss and final_loss and final_loss > 0.9 * initial_loss:
            return {"failure_type": "divergence",
                    "severity": "critical",
                    "details": {"initial_loss": initial_loss,
                                "final_loss": final_loss}}

        # -- Stagnation: <0.1% improvement in last 20% of epochs
        if len(train_history) >= 5:
            tail_start = max(0, int(len(train_history) * 0.8))
            tail = train_history[tail_start:]
            losses = [_safe_float(r.get(loss_key)) for r in tail if r.get(loss_key)]
            losses = [v for v in losses if v is not None]
            if len(losses) >= 2:
                relative_improvement = abs(losses[0] - losses[-1]) / max(abs(losses[0]), 1e-8)
                if relative_improvement < 0.001:
                    return {"failure_type": "stagnation",
                            "severity": "warning",
                            "details": {"tail_start_loss": losses[0],
                                        "tail_end_loss": losses[-1],
                                        "relative_improvement": relative_improvement}}

        return {"failure_type": None, "severity": "ok", "details": {}}

    # ------------------------------------------------------------------

    def adjust_hyperparameters(self, failure_type: str,
                               prev_config: Dict[str, Any],
                               attempt_num: int) -> Dict[str, Any]:
        """
        Tool 6 — Ask the LLM to propose improved hyperparameters after a failure.

        Returns an updated HyperparamConfig dict.
        """
        rule_hints = {
            "latent_collapse":       "Reduce kld_weight by 50%. Consider annealing KL from 0.",
            "posterior_collapse":    "Reduce kld_weight by 50%. Increase latent_dims slightly.",
            "divergence":            "Reduce lr by 10x. Set gradient_clip_val=1.0.",
            "stagnation":            "Increase max_epochs by 50%. Increase latent_dims.",
            "numerical_instability": "Reduce lr by 10x. Set gradient_clip_val=0.5.",
        }
        hint = rule_hints.get(failure_type, "Review all hyperparameters for stability.")

        prompt = f"""You are an expert in training VAEs. A training run failed with:

## Failure type: {failure_type}
## Hint: {hint}

## Previous hyperparameters
{json.dumps(prev_config, indent=2)}

## Your task
Propose adjusted hyperparameters to fix the failure. Apply the hint strictly.
Attempt number: {attempt_num}

Respond in JSON only (same schema as previous config, with "rationale" added):
{{
  "latent_dims": <int>,
  "lr": <float>,
  "weight_decay": <float>,
  "kld_weight": <float or null>,
  "max_epochs": <int>,
  "batch_size": <int>,
  "gradient_clip_val": <float or null>,
  "encoder_configs": null,
  "decoder_configs": null,
  "rationale": "<one sentence>"
}}"""

        result_json, error = self._generate_json_from_text_parts([prompt])

        if error or not result_json:
            # Apply rule-based fallback adjustments
            result_json = dict(prev_config)
            if failure_type in ("latent_collapse", "posterior_collapse"):
                result_json["kld_weight"] = (prev_config.get("kld_weight") or 1.0) * 0.5
            elif failure_type in ("divergence", "numerical_instability"):
                result_json["lr"] = prev_config.get("lr", 1e-3) * 0.1
                result_json["gradient_clip_val"] = 0.5
            elif failure_type == "stagnation":
                result_json["max_epochs"] = int(prev_config.get("max_epochs", 50) * 1.5)
            result_json["rationale"] = f"Rule-based fix for {failure_type} (LLM unavailable)."

        self._log_action(
            action="adjust_hyperparameters",
            input_ctx={"failure_type": failure_type, "attempt_num": attempt_num,
                       "prev_lr": prev_config.get("lr"),
                       "prev_kld_weight": prev_config.get("kld_weight")},
            result={"status": "completed",
                    "new_lr": result_json.get("lr"),
                    "new_kld_weight": result_json.get("kld_weight")},
            rationale=result_json.get("rationale"),
        )
        return result_json

    # =========================================================================
    # Main entry point
    # =========================================================================

    def analyze(self,
                data: Union[str, np.ndarray],
                system_info: Optional[Dict[str, Any]] = None,
                task_description: str = "",
                max_retries: int = 3,
                **kwargs) -> Dict[str, Any]:
        """
        Main entry point.  Orchestrates model selection, training, failure
        detection, and latent space analysis with up to max_retries attempts.

        Args:
            data:             numpy array of shape (N, D) or (N, H, W) or
                              (N, 1, H, W); one sample per row.
            system_info:      Optional metadata dict (material, technique, etc.).
            task_description: Optional plain-text description of the scientific goal.
            max_retries:      Maximum training attempts before giving up.

        Returns (Principle 3 — Standardised I/O):
            {
                "status":                "success|error",
                "model_type":            str,
                "training_attempts":     int,
                "final_metrics":         {"loss": float, "val_loss": float, ...},
                "latent_space_analysis": {"active_dims": int, ...},
                "model_path":            str,
                "output_directory":      str,
                "action_history":        list,
            }
        """
        self._init_state(
            task_description=task_description,
            system_info=system_info or {},
        )

        # -- Input validation
        if isinstance(data, str):
            try:
                data = np.load(data)
            except Exception as e:
                return {"status": "error",
                        "error": f"Could not load data from path: {e}"}

        if not isinstance(data, np.ndarray) or data.ndim < 2:
            return {"status": "error",
                    "error": "data must be a numpy array with shape (N, ...)."}

        n_samples   = data.shape[0]
        sample_shape = data.shape[1:]

        self.logger.info(
            f"VAEAgent.analyze() — {n_samples} samples, shape {sample_shape}"
        )

        # Tool 1 — Model selection
        sel = self.select_model(sample_shape, task_description)
        model_type = sel["model_type"]
        self.state["model_type"] = model_type

        # Tool 2 — Initial hyperparameter configuration
        hparams = self.configure_hyperparameters(model_type, sample_shape, n_samples)

        training_result = None
        latent_stats    = None
        failure_info    = None

        for attempt in range(1, max_retries + 1):
            self.logger.info(f"[Attempt {attempt}/{max_retries}] Training {model_type}...")

            # Tool 3 — Train
            training_result = self.train_vae(data, model_type, hparams, attempt=attempt)

            if training_result["status"] == "error":
                self.state["failure_history"].append({
                    "attempt": attempt,
                    "failure_type": "training_error",
                    "error": training_result.get("error"),
                })
                if attempt < max_retries:
                    hparams = self.adjust_hyperparameters(
                        "numerical_instability", hparams, attempt
                    )
                continue

            model         = training_result["model"]
            train_history = training_result["train_history"]

            # Tool 4 — Latent space analysis
            latent_stats = self.analyze_latent_space(
                model, data, batch_size=hparams["batch_size"]
            )

            # Tool 5 — Failure detection
            failure_info = self.detect_training_failure(
                train_history, latent_stats, model_type
            )
            self.state["failure_history"].append({
                "attempt": attempt, **failure_info
            })

            if failure_info["severity"] == "ok":
                self.logger.info(f"Training succeeded on attempt {attempt}.")
                break

            self.logger.warning(
                f"Attempt {attempt}: {failure_info['failure_type']} "
                f"(severity={failure_info['severity']}). "
                f"{'Retraining...' if attempt < max_retries else 'Max retries reached.'}"
            )

            if attempt < max_retries:
                # Tool 6 — Adjust hyperparameters
                hparams = self.adjust_hyperparameters(
                    failure_info["failure_type"], hparams, attempt
                )

        # -- Compile final result
        if training_result is None or training_result["status"] == "error":
            return {
                "status":           "error",
                "error":            "All training attempts failed.",
                "failure_history":  self.state["failure_history"],
                "output_directory": str(self.output_dir),
                "action_history":   self.state.get("action_history", []),
            }

        # Record best metrics
        final_metrics = {
            k: _safe_float(v)
            for k, v in training_result.get("final_metrics", {}).items()
            if k != "epoch"
        }
        self.state["best_metrics"]         = final_metrics
        self.state["latent_space_summary"] = latent_stats
        self.state["status"]               = "completed"
        self._save_state()

        return {
            "status":                "success",
            "model_type":            model_type,
            "training_attempts":     len(self.state["failure_history"]),
            "final_metrics":         final_metrics,
            "latent_space_analysis": latent_stats,
            "model_path":            training_result.get("model_path", ""),
            "output_directory":      str(self.output_dir),
            "action_history":        self.state.get("action_history", []),
        }
