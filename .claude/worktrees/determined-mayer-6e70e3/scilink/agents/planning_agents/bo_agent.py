import pandas as pd
import numpy as np
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import PIL.Image as PIL_Image

from ...auth import get_internal_proxy_key
from .parser_utils import parse_json_from_response 
from ...tools.bo_tools import get_optimizer
from .instruct import (
    BO_CONFIG_SOO_PROMPT,
    BO_CONFIG_MOO_PROMPT,
    BO_VISUAL_INSPECTION_PROMPT,
    BO_VISUAL_INSPECTION_MOO_PROMPT,
    BO_CONSTRAINED_BATCH_PROMPT,
    BO_CONSTRAINED_BATCH_PROMPT_MOO
)

from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from ._deprecation import normalize_params

from .base_agent import BaseAgent


def _compute_budget_context(experimental_budget: Optional[int], history: List[Dict]) -> Dict[str, Any]:
    """
    Compute budget-aware context for strategy selection.
    
    Translates a raw experiment count into a structured context dict that the 
    LLM can reason about. The phase classification uses both absolute remaining 
    count and the fraction of total campaign budget to handle edge cases 
    (e.g., budget=5 means something different on step 2 vs step 50).
    
    Args:
        experimental_budget: Remaining optimization iterations (None = unlimited).
        history: List of past history entries (used to compute steps completed).
    
    Returns:
        Dict with keys:
            - budget_total: Remaining iterations (None if unlimited)
            - steps_completed: Number of BO iterations already done
            - budget_fraction_remaining: Float in [0, 1] — fraction of full 
              campaign still available. None if unlimited.
            - budget_phase: One of "final_shot", "critical", "low", "mid", 
              "high", "unlimited"
            - budget_guidance: Human-readable strategy guidance string for the LLM
    """
    steps_completed = len(history)
    
    if experimental_budget is None:
        return {
            "budget_total": None,
            "steps_completed": steps_completed,
            "budget_fraction_remaining": None,
            "budget_phase": "unlimited",
            "budget_guidance": (
                "No experimental budget constraint. "
                "Balance exploration and exploitation normally."
            ),
        }
    
    total_campaign = steps_completed + experimental_budget
    fraction_remaining = experimental_budget / total_campaign if total_campaign > 0 else 0.0
    
    # Classify into phases based on absolute remaining AND fraction
    if experimental_budget <= 1:
        phase = "final_shot"
        guidance = (
            "CRITICAL: This is the LAST experiment. You MUST exploit. "
            "Use 'log_ei' or 'ucb' with very low beta (< 0.3). "
            "Do NOT use 'max_variance' or 'thompson'. "
            "Every point must target the most promising region found so far."
        )
    elif experimental_budget <= 3:
        phase = "critical"
        guidance = (
            f"Only {experimental_budget} experiments remain. Strongly favor exploitation. "
            "Use 'log_ei' (preferred) or 'ucb' with low beta (0.3-1.0). "
            "Avoid 'max_variance'. 'thompson' acceptable only if batch_size > 10. "
            "Reserve at most 1 point for exploration if batch allows."
        )
    elif fraction_remaining < 0.25:
        phase = "low"
        guidance = (
            f"{experimental_budget} experiments remain ({fraction_remaining:.0%} of campaign budget). "
            "Late-stage optimization — lean toward exploitation. "
            "Use 'log_ei' or 'ucb' with moderate beta (1.0-2.0). "
            "'max_variance' only if model calibration is poor."
        )
    elif fraction_remaining < 0.6:
        phase = "mid"
        guidance = (
            f"{experimental_budget} experiments remain ({fraction_remaining:.0%} of campaign budget). "
            "Mid-campaign — balance exploration and exploitation. "
            "'log_ei' is a strong default. 'ucb' with beta ~2.0 also appropriate. "
            "'max_variance' acceptable if data is sparse or model uncertain."
        )
    else:
        phase = "high"
        guidance = (
            f"{experimental_budget} experiments remain ({fraction_remaining:.0%} of campaign budget). "
            "Early-stage — exploration is valuable. "
            "'max_variance' is appropriate if few data points exist. "
            "'log_ei' or 'thompson' for balanced exploration with some exploitation."
        )
    
    return {
        "budget_total": experimental_budget,
        "steps_completed": steps_completed,
        "budget_fraction_remaining": round(fraction_remaining, 3),
        "budget_phase": phase,
        "budget_guidance": guidance,
    }


class BOAgent(BaseAgent):
    """
    Autonomous Agent for Bayesian Optimization (BO) designed for "Stop-and-Go" experimental loops.

    This agent acts as an AI research partner that plans your next set of experiments.
    It combines valid statistical modeling (Gaussian Processes) with LLM-based reasoning 
    to adaptively configure the optimization strategy based on your data trends.

    **DATA FORMATTING REQUIREMENTS:**
    --------------------------------
    The agent expects a "Tidy Data" format (Excel .xlsx or CSV .csv) where:
    1.  **Rows** represent individual experiments.
    2.  **Columns** represent input parameters (e.g., 'Temperature', 'Pressure') and 
        measured objectives (e.g., 'Yield', 'Purity').
    3.  **No Merged Cells:** Ensure the header is a single row containing clean variable names.
    4.  **Missing Data:** The agent requires complete data rows for the optimization columns. 
        Rows with NaNs in inputs/targets should be removed or imputed before running.

    **PERSISTENCE & WORKFLOW:**
    ---------------------------
    This agent is stateless and persistent. It is safe to shut down between experiments.
    
    1.  **Run Agent:** Call `run_optimization_loop` pointing to your current data file.
    2.  **Get Recommendations:** The agent saves a new batch of experiments to 
        `./bo_artifacts/batch_step_N.csv`.
    3.  **Shut Down:** You can close the program while you perform the experiments in the lab 
        (whether it takes 1 hour or 1 week).
    4.  **Update Data:** Once results are in, append them as new rows to your original 
        data file (.xlsx/.csv).
    5.  **Restart:** Run the agent again. It automatically re-reads the updated data 
        and the history file (`bo_history.json`) to pick up exactly where it left off.

    Args:
        api_key: API key for the LLM provider.
        model_name: Model name. For public deployments, use LiteLLM format
            (e.g., "gemini/gemini-2.0-flash", "gpt-4o", "claude-sonnet-4-20250514").
        base_url: Base URL for internal proxy endpoint.
            When provided, uses OpenAI-compatible client.
            When None, uses LiteLLM for multi-provider support.
        output_dir: Output directory for artifacts.
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        output_dir: str = ".",
        # Deprecated
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        
        super().__init__(output_dir)
        self.agent_type = "bo"

        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="BOAgent"
        )
        
        if base_url:
            # INTERNAL PROXY
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
            
            logging.info(f"🏛️ BOAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # PUBLIC LITELLM
            logging.info(f"🌐 BOAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
        
        self.generation_config = None

        self.history_file = self.output_dir / "bo_history.json"

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Agent-specific state fields"""
        return {
            "objective": None,
            "data_path": None,
            "optimization_history": [],
            "current_config": None,
            "data_points_seen": 0,
            "experimental_budget": None,
        }
        
    def _load_history(self) -> List[Dict]:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f: return json.load(f)
        return []

    def _save_history(self, entry: Dict):
        history = self._load_history()
        history.append(entry)
        with open(self.history_file, 'w') as f: json.dump(history, f, indent=2)

    def _validate_config(self, config: Dict) -> Dict:
        clean = config.copy()
        m_conf = clean.get("model_config", {})
        if m_conf.get("kernel") not in ["matern_2.5", "matern_1.5", "matern_0.5", "rbf"]:
            logging.warning(f"Invalid kernel '{m_conf.get('kernel')}', defaulting to 'matern_2.5'")
            m_conf["kernel"] = "matern_2.5"
        if m_conf.get("noise") not in ["min_noise_low", "min_noise_med", "min_noise_high"]:
            logging.warning(f"Invalid noise '{m_conf.get('noise')}', defaulting to 'min_noise_low'")
            m_conf["noise"] = "min_noise_low"
        if m_conf.get("input_transform", "none") not in ["none", "warp"]:
            logging.warning(f"Invalid input_transform '{m_conf.get('input_transform')}', defaulting to 'none'")
            m_conf["input_transform"] = "none"
        clean["model_config"] = m_conf
        return clean

    # =====================================================================
    # Acquisition Landscape Summarization (for constrained batch planning)
    # =====================================================================

    def _summarize_acquisition_landscape(
        self,
        optimizer,
        input_cols: List[str],
        input_bounds: List[List[float]],
        is_moo: bool = False,
        n_regions: int = 15,
        grid_resolution: int = 40
    ) -> str:
        """
        Evaluate the acquisition function on a dense grid, cluster high-value 
        regions, and return a markdown summary table for LLM consumption.
        
        The summary gives the LLM a ranked "menu" of where the model expects the 
        most value, so it can map these regions onto physical constraints.
        
        Args:
            optimizer: Fitted optimizer object (from bo_tools.get_optimizer)
            input_cols: List of input parameter names
            input_bounds: List of [min, max] per parameter
            is_moo: Whether this is multi-objective optimization
            n_regions: Number of top regions to report
            grid_resolution: Points per dimension for grid evaluation
            
        Returns:
            Markdown-formatted string with ranked regions table
        """
        n_dims = len(input_cols)
        
        # 1. Build evaluation grid
        #    Full grid for low dimensions, Latin Hypercube for high dimensions
        if n_dims <= 3:
            axes = [
                np.linspace(bounds[0], bounds[1], grid_resolution) 
                for bounds in input_bounds
            ]
            mesh = np.meshgrid(*axes, indexing='ij')
            grid_points = np.column_stack([m.ravel() for m in mesh])
        else:
            # Latin Hypercube Sampling — cap total evaluations
            n_samples = min(grid_resolution ** 2, 5000)
            grid_points = np.random.rand(n_samples, n_dims)
            for d in range(n_dims):
                lo, hi = input_bounds[d]
                grid_points[:, d] = lo + grid_points[:, d] * (hi - lo)
        
        # 2. Evaluate acquisition function at all grid points
        #    Try direct acquisition evaluation first, then fall back to variance
        try:
            acq_values = optimizer.evaluate_acquisition(grid_points)
        except (AttributeError, NotImplementedError):
            try:
                _, acq_values = optimizer.predict(grid_points)
                print("    - ℹ️  Using posterior variance as acquisition proxy")
            except Exception as e:
                print(f"    - ⚠️  Cannot evaluate acquisition landscape: {e}")
                return self._fallback_landscape_summary(optimizer, input_cols, input_bounds)
        
        if acq_values is None or len(acq_values) == 0:
            return "Acquisition landscape evaluation returned no data."
        
        acq_values = np.array(acq_values).ravel()
        
        # 3. Cluster into regions
        regions = self._cluster_acquisition_regions(
            grid_points, acq_values, input_cols, input_bounds, n_regions
        )
        
        # 4. Format as markdown table
        header_cols = " | ".join(input_cols)
        header = f"| Rank | {header_cols} | Acq. Value | Spread | Notes |"
        separator = "|" + "|".join(["---"] * (len(input_cols) + 4)) + "|"
        
        rows = []
        for i, region in enumerate(regions):
            param_strs = " | ".join(
                f"{region['center'][j]:.4f}" for j in range(len(input_cols))
            )
            spread_str = ", ".join(
                f"{s:.3f}" for s in region.get('spread', [0.0] * len(input_cols))
            )
            notes = region.get('notes', '')
            rows.append(
                f"| {i+1} | {param_strs} | {region['acq_value']:.5f} | {spread_str} | {notes} |"
            )
        
        table = header + "\n" + separator + "\n" + "\n".join(rows)
        
        summary = f"""### Acquisition Landscape Summary
Total grid points evaluated: {len(grid_points)}
Number of dimensions: {n_dims}
Acquisition value range: [{acq_values.min():.5f}, {acq_values.max():.5f}]

#### Top {len(regions)} Regions (ranked by acquisition value)
{table}

**Interpretation:** Higher acquisition value = the model expects more information gain 
or improvement from sampling that region. "Spread" indicates how broad the high-value 
zone is around each center (per parameter). Wider spread = more forgiving placement.
"""
        return summary

    def _cluster_acquisition_regions(
        self,
        grid_points: np.ndarray,
        acq_values: np.ndarray,
        input_cols: List[str],
        input_bounds: List[List[float]],
        n_regions: int
    ) -> List[Dict[str, Any]]:
        """
        Identify distinct high-value regions via greedy peak selection 
        with exclusion zones. Prevents the LLM from seeing a table of 
        near-duplicate points that all cluster in one area.
        
        Args:
            grid_points: (N, D) array of evaluated points
            acq_values: (N,) array of acquisition values
            input_cols: Parameter names (for boundary detection notes)
            input_bounds: Parameter bounds
            n_regions: Max number of regions to return
            
        Returns:
            List of region dicts with center, acq_value, spread, notes
        """
        n_dims = len(input_cols)
        
        # Normalize coordinates to [0, 1] for distance computation
        bounds_array = np.array(input_bounds)
        ranges = bounds_array[:, 1] - bounds_array[:, 0]
        ranges[ranges == 0] = 1.0
        normalized = (grid_points - bounds_array[:, 0]) / ranges
        
        # Minimum separation between region centers (in normalized space)
        min_separation = 0.15
        
        sorted_idx = np.argsort(-acq_values)
        
        regions = []
        selected_centers = []
        
        for idx in sorted_idx:
            if len(regions) >= n_regions:
                break
            
            candidate = normalized[idx]
            
            # Check distance to already-selected centers
            too_close = False
            for center in selected_centers:
                dist = np.sqrt(np.sum((candidate - center) ** 2))
                if dist < min_separation:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            selected_centers.append(candidate)
            
            # Compute spread: std of nearby high-value points
            distances = np.sqrt(np.sum((normalized - candidate) ** 2, axis=1))
            high_value_mask = (distances < min_separation * 2) & (acq_values > acq_values[idx] * 0.5)
            
            if high_value_mask.sum() > 1:
                spread = np.std(grid_points[high_value_mask], axis=0).tolist()
            else:
                spread = [0.0] * n_dims
            
            # Boundary detection
            center_raw = grid_points[idx]
            notes_parts = []
            for d in range(n_dims):
                lo, hi = input_bounds[d]
                param_range = hi - lo
                if param_range > 0:
                    if (center_raw[d] - lo) / param_range < 0.05:
                        notes_parts.append(f"{input_cols[d]} at lower bound")
                    elif (hi - center_raw[d]) / param_range < 0.05:
                        notes_parts.append(f"{input_cols[d]} at upper bound")
            
            regions.append({
                'center': grid_points[idx].tolist(),
                'acq_value': float(acq_values[idx]),
                'spread': spread,
                'notes': "; ".join(notes_parts) if notes_parts else ""
            })
        
        return regions

    def _fallback_landscape_summary(self, optimizer, input_cols, input_bounds) -> str:
        """
        Minimal fallback when acquisition evaluation is unavailable.
        The LLM will rely on unconstrained suggestions and data summary instead.
        """
        return (
            "### Acquisition Landscape Summary\n"
            "⚠️ Direct acquisition function evaluation not available for this optimizer.\n"
            "Use the unconstrained BO suggestions and data summary below "
            "to design the constrained batch.\n"
        )

    # =====================================================================
    # Constrained Batch Planning
    # =====================================================================

    def _plan_constrained_batch(
        self,
        objective_text: str,
        input_cols: List[str],
        input_bounds: List[List[float]],
        batch_size: int,
        acq_summary: str,
        physical_constraints: str,
        unconstrained_recommendations: List[Dict[str, float]],
        data_summary_str: str,
        current_best: Dict[str, float],
        current_best_value: Dict[str, float],
        budget_ctx: Dict[str, Any],
        is_moo: bool = False,
        pareto_front: Optional[List[Dict]] = None
    ) -> Tuple[Optional[List[Dict[str, float]]], Optional[Dict[str, Any]], Optional[str]]:
        """
        Use LLM to design a physically constrained experiment batch informed by 
        the acquisition landscape.
        
        Args:
            objective_text: Scientific optimization objective
            input_cols: Parameter names
            input_bounds: Parameter bounds
            batch_size: Number of experiments to design
            acq_summary: Markdown summary from _summarize_acquisition_landscape
            physical_constraints: Natural language constraint description
            unconstrained_recommendations: Standard BO output (for reference/fallback)
            data_summary_str: df.describe() as markdown
            current_best: Best parameters found so far
            current_best_value: Best objective value(s) found so far
            budget_ctx: Budget context from _compute_budget_context
            is_moo: Multi-objective flag
            pareto_front: Pareto front points for MOO
            
        Returns:
            Tuple of (recommendations_list, metadata_dict, error_string_or_None)
        """
        prompt_template = BO_CONSTRAINED_BATCH_PROMPT_MOO if is_moo else BO_CONSTRAINED_BATCH_PROMPT
        
        prompt_parts = [
            prompt_template,
            f"## Optimization Objective\n{objective_text}",
            f"## Batch Size\n{batch_size} experiments to design",
            f"\n## REQUIRED Parameter Names (use these EXACT keys in params)\n"
            f"{json.dumps(input_cols)}\n"
            f"Every experiment in the batch must have ALL of these keys in its \"params\" dict. "
            f"Use these exact strings — do not rename, abbreviate, or expand them.",
            f"\n## Parameter Bounds\n" + "\n".join(
                f"- {col}: [{bounds[0]}, {bounds[1]}]" 
                for col, bounds in zip(input_cols, input_bounds)
            ),
            f"\n## Acquisition Landscape\n{acq_summary}",
            f"\n## Physical Constraints\n{physical_constraints}",
        ]
        
        # Budget context for constrained planner
        if budget_ctx["budget_phase"] != "unlimited":
            if budget_ctx["budget_phase"] in ("final_shot", "critical"):
                budget_block = (
                    f"\n## ⚠️ Experimental Budget — CRITICAL\n"
                    f"- Remaining iterations (including this batch): {budget_ctx['budget_total']}\n"
                    f"- Campaign phase: **{budget_ctx['budget_phase']}**\n"
                    f"- {budget_ctx['budget_guidance']}\n"
                    f"\n**THIS OVERRIDES DESIGN PRINCIPLE #1 (Maximize Coverage).**\n"
                    f"This is the LAST batch. Do NOT spread experiments uniformly across "
                    f"the parameter space. Instead:\n"
                    f"1. **Concentrate ≥60% of experiments** in the top 3-5 acquisition regions. "
                    f"These are the regions most likely to contain the optimum.\n"
                    f"2. **IMPORTANT — Include the predicted optimum.** The acquisition function "
                    f"assigns LOW values to already-observed locations (because uncertainty is low "
                    f"there). But the Current Best parameters (see below) and the GP-predicted peak "
                    f"are still the most promising locations. Allocate 3-5 experiments AT or very "
                    f"near the Current Best parameters (snapped to feasible values). This is "
                    f"essential because: (a) confirming reproducibility of the best result has "
                    f"high scientific value, and (b) the true optimum may coincide with an observed "
                    f"point that the acquisition function undervalues.\n"
                    f"3. **Do NOT allocate experiments** to low-acquisition regions just for "
                    f"coverage. Every experiment in a low-value region is wasted.\n"
                    f"4. **Non-uniform parameter allocation is expected.** If the acquisition "
                    f"landscape peaks at specific parameter combinations, most experiments should "
                    f"cluster there — not be evenly distributed across all feasible levels.\n"
                    f"5. Look at the Acquisition Landscape table above. The Acq. Value column "
                    f"tells you where to concentrate. Regions ranked 1-5 should get the bulk "
                    f"of the experiments.\n"
                    f"6. **Allocation guideline:** ~5 experiments replicating current best, "
                    f"~60% of remaining on top acquisition regions, ~40% of remaining on "
                    f"next-best regions."
                )
            else:
                budget_block = (
                    f"\n## Experimental Budget\n"
                    f"- Remaining experiments (including this batch): {budget_ctx['budget_total']}\n"
                    f"- Campaign phase: **{budget_ctx['budget_phase']}**\n"
                    f"- {budget_ctx['budget_guidance']}\n"
                    f"\n**Budget implication for batch design:** "
                    f"Balance coverage with exploitation based on remaining budget."
                )
            prompt_parts.append(budget_block)
        
        if is_moo and pareto_front:
            prompt_parts.append(
                f"\n## Current Pareto Front ({len(pareto_front)} points)\n"
                f"{json.dumps(pareto_front[:20], indent=2)}"
            )
        elif current_best:
            prompt_parts.append(
                f"\n## Current Best Parameters\n{json.dumps(current_best)}"
            )
            if current_best_value:
                prompt_parts.append(
                    f"\n## Current Best Result\n{json.dumps(current_best_value)}"
                )
        
        prompt_parts.append(
            f"\n## Unconstrained BO Suggestions (for reference)\n"
            f"{json.dumps(unconstrained_recommendations, indent=2)}"
        )
        prompt_parts.append(f"\n## Data Summary\n{data_summary_str}")
        
        print(f"  - 🏗️ BO Agent: Planning constrained batch ({batch_size} experiments)...")
        
        max_retries = 3
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"  - 🔄 Retry {attempt}/{max_retries}...")
                
                resp = self.model.generate_content(
                    prompt_parts, 
                    generation_config=self.generation_config
                )
                constrained_batch, parse_error = parse_json_from_response(resp)
                
                if parse_error:
                    last_error = f"JSON parse error: {parse_error}"
                    logging.warning(
                        f"Constrained batch attempt {attempt}: {last_error}"
                    )
                    continue
                
                if not constrained_batch or "batch" not in constrained_batch:
                    last_error = "LLM response missing 'batch' key"
                    logging.warning(
                        f"Constrained batch attempt {attempt}: {last_error}"
                    )
                    continue
                
                batch_items = constrained_batch["batch"]
                
                if not batch_items:
                    last_error = "LLM returned empty batch"
                    logging.warning(
                        f"Constrained batch attempt {attempt}: {last_error}"
                    )
                    continue
                
                # Validate and extract recommendations
                recommendations = []
                validation_errors = []
                
                for i, item in enumerate(batch_items):
                    params = item.get("params", {})
                    
                    if not params:
                        validation_errors.append(f"Experiment {i+1}: missing params")
                        continue
                    
                    # Check all input columns present
                    missing_cols = [c for c in input_cols if c not in params]
                    if missing_cols:
                        validation_errors.append(
                            f"Experiment {i+1}: missing columns {missing_cols}"
                        )
                        continue
                    
                    # Check values within bounds (with tolerance for constraint snapping)
                    tolerance = 0.01
                    for col, bounds in zip(input_cols, input_bounds):
                        val = float(params[col])
                        param_range = bounds[1] - bounds[0]
                        tol = param_range * tolerance if param_range > 0 else 0.01
                        if val < bounds[0] - tol or val > bounds[1] + tol:
                            validation_errors.append(
                                f"Experiment {i+1}: {col}={val} outside bounds "
                                f"[{bounds[0]}, {bounds[1]}]"
                            )
                    
                    rec = {col: float(params[col]) for col in input_cols}
                    recommendations.append(rec)
                
                if validation_errors:
                    for err in validation_errors:
                        print(f"    - ⚠️  {err}")
                
                if not recommendations:
                    last_error = f"No valid experiments in batch. Errors: {validation_errors}"
                    logging.warning(
                        f"Constrained batch attempt {attempt}: {last_error}"
                    )
                    continue
                
                # Warn if we got fewer than requested — do NOT pad with unconstrained
                # values since they won't respect discrete constraints
                if len(recommendations) < batch_size:
                    shortfall = batch_size - len(recommendations)
                    print(
                        f"    - ⚠️  Got {len(recommendations)}/{batch_size} valid experiments. "
                        f"    {shortfall} slots unfilled (unconstrained padding disabled)."
                    )
                
                metadata = {
                    "allocation_strategy": constrained_batch.get("allocation_strategy", ""),
                    "coverage_summary": constrained_batch.get("coverage_summary", ""),
                    "trade_offs": constrained_batch.get("trade_offs", ""),
                    "validation_points": constrained_batch.get("validation_points", ""),
                    "pareto_strategy": constrained_batch.get("pareto_strategy", ""),
                    "valid_count": len(recommendations),
                    "requested_count": batch_size,
                    "shortfall": max(0, batch_size - len(recommendations)),
                    "validation_errors": validation_errors if validation_errors else None,
                    "attempts": attempt,
                }
                
                print(f"  - ✅ Constrained batch planned: {len(recommendations)} experiments"
                      + (f" (attempt {attempt})" if attempt > 1 else ""))
                return recommendations[:batch_size], metadata, None
                
            except Exception as e:
                last_error = str(e)
                logging.warning(
                    f"Constrained batch attempt {attempt} exception: {e}"
                )
                if attempt == max_retries:
                    logging.error(
                        f"Constrained batch planning failed after {max_retries} attempts",
                        exc_info=True
                    )
        
        return None, None, f"Failed after {max_retries} attempts. Last error: {last_error}"

    # =====================================================================
    # Main Optimization Loop
    # =====================================================================

    def run_optimization_loop(self, data_path: str, objective_text: str,
                             input_cols: List[str], input_bounds: List[List[float]],
                             target_cols: List[str],
                             target_directions: Optional[Dict[str, str]] = None,
                             output_dir: str = "./bo_artifacts",
                             batch_size: int = 1,
                             experimental_budget: Optional[int] = None,
                             physical_constraints: Optional[str] = None,
                             strategy_hint: Optional[str] = None,
                             save_acq: bool = True,
                             plot_acq: bool = True,
                             fixed_noise_std: Optional[float] = None) -> Dict[str, Any]:
        """
        Run one iteration of the Bayesian Optimization loop.
        
        Args:
            data_path: Path to the data file (.xlsx or .csv).
            objective_text: Natural language description of the optimization goal.
            input_cols: List of input column names.
            input_bounds: List of [min, max] bounds for each input.
            target_cols: List of target/objective column names.
            output_dir: Directory for saving artifacts.
            batch_size: Number of candidates to recommend. When physical_constraints 
                is provided, the constrained planner uses this as the target number
                of experiments to design on the plate.
            experimental_budget: Optional number of remaining experiments (iterations) 
                in the campaign, INCLUDING this one. Controls the 
                exploration-vs-exploitation balance:
                - None (default): No budget constraint; standard behavior.
                - 1: Final experiment — forces pure exploitation.
                - 2-3: Critical budget — strongly favors exploitation.
                - Higher values: Scaled guidance based on fraction of total 
                  campaign completed.
                The budget is passed to the LLM as strategic context in the 
                strategy configuration and constrained batch planning prompts.
                Note: this counts optimization iterations (calls to this method), 
                not individual experiments. A batch_size=10 call with 
                experimental_budget=2 means 2 more calls (up to 20 experiments).
            physical_constraints: Optional natural language description of physical 
                experimental constraints. When provided, the agent evaluates the 
                acquisition landscape and uses LLM reasoning to design a batch that 
                maximizes information gain while respecting the constraints. Examples:
                - "96-well plate: rows share temperature (8 values), columns share pH (12 values)"
                - "Only 5 catalyst concentrations available: 0.1, 0.5, 1.0, 2.0, 5.0 mM"
                - "Reactor zones A,B share cooling; C,D share heating. Max 4 temps total."
                When None, standard unconstrained BO is used (original behavior).
            strategy_hint: Optional natural language hint from the user to guide
                strategy selection (kernel, acquisition function, noise prior).
                Examples: "use RBF kernel", "try UCB with high exploration",
                "switch to Thompson sampling". Respected unless it conflicts
                with budget constraints.
            save_acq: If True, saves acquisition function landscape data to .npz file.
                Supported for single-objective only; ignored for multi-objective.
            plot_acq: If True, generates and saves a plot of the acquisition function.
                Supported for single-objective only; ignored for multi-objective.
            
        Returns:
            Dict with status, recommendations, strategy, plot paths, budget context,
            and optionally acquisition function plot/data paths (single-objective only)
            and constrained planning metadata (when physical_constraints provided).
        """
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialize state
        self._init_state(objective=objective_text, data_path=data_path)
        
        # 1. Load Data
        minimize_mask = []
        try:
            df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
            for col in input_cols + target_cols:
                if col not in df.columns:
                    return {"error": f"Column '{col}' not found in data."}
            X = df[input_cols].values
            y = df[target_cols].values

            # Negate minimize targets so the optimizer always maximizes
            if target_directions:
                for i, col in enumerate(target_cols):
                    if target_directions.get(col, "maximize") == "minimize":
                        y[:, i] *= -1
                        minimize_mask.append(i)
                if minimize_mask:
                    minimized_names = [target_cols[i] for i in minimize_mask]
                    print(f"  - 🔄 Negated for minimization: {minimized_names}")

            # Track data points
            self.state["data_points_seen"] = len(df)

        except Exception as e:
            return {"error": f"Data load failed: {e}"}

        is_moo = len(target_cols) > 1
        history = self._load_history()

        is_retry = history and history[-1].get("data_points") == len(df)
        if is_retry:
            print(f"  - 🔄 Re-run detected (same {len(df)} data points). Replacing previous step.")
            history.pop()
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)

        # Compute budget context
        budget_ctx = _compute_budget_context(experimental_budget, history)
        self.state["experimental_budget"] = experimental_budget
        
        if budget_ctx["budget_phase"] != "unlimited":
            print(
                f"  - 💰 Budget: {budget_ctx['budget_total']} iterations remaining "
                f"(phase: {budget_ctx['budget_phase']}, "
                f"{budget_ctx['budget_fraction_remaining']:.0%} of campaign left)"
            )

        # 2. Configure Strategy (LLM)
        if history:
            trend_parts = []
            for h in history[-5:]:
                rationale = h.get("config", {}).get("rationale", "N/A")
                insp = h.get("inspection", {})
                insp_status = insp.get("status", "")
                insp_reason = insp.get("reason", "")
                insp_adj = insp.get("suggested_adjustments", {})
                entry = f"Strategy: {rationale}"
                if insp_status:
                    entry += f" | Diagnostics: {insp_status}"
                    if insp_reason:
                        entry += f" — {insp_reason}"
                    if insp_adj:
                        entry += f" | Suggested adjustments: {insp_adj}"
                trend_parts.append(entry)
            trend_context = "Last steps:\n" + "\n".join(
                f"  Step {i+1}: {p}" for i, p in enumerate(trend_parts)
            )
        else:
            trend_context = "No history."
        
        prompt_tmpl = BO_CONFIG_MOO_PROMPT if is_moo else BO_CONFIG_SOO_PROMPT
        prompt_parts = [
            prompt_tmpl,
            f"Objective: {objective_text}",
            *(
                [f"Optimization direction: {', '.join(f'{c} → {d}' for c, d in target_directions.items() if c in target_cols)}"]
                if target_directions and any(
                    target_directions.get(c) == "minimize" for c in target_cols
                ) else []
            ),
            f"Constraint: Fixed Batch Size = {batch_size}",
            f"Meta-Data Trend: {trend_context}",
            f"Data Summary:\n{df.describe().to_markdown()}"
        ]
        
        # Budget context for strategy LLM
        prompt_parts.append(
            f"\n**Experimental Budget:**\n{budget_ctx['budget_guidance']}\n"
            f"Steps completed so far: {budget_ctx['steps_completed']}. "
            f"Data points in dataset: {len(df)}."
        )
        
        # Inform strategy LLM about constraints (for better acq strategy selection)
        if physical_constraints:
            prompt_parts.append(
                f"\n**Physical Constraints (informational for strategy selection):**\n"
                f"{physical_constraints}\n"
                f"Note: A separate step will handle constraint-aware batch design. "
                f"Focus on selecting the best kernel, noise, and acquisition strategy."
            )
        
        if strategy_hint:
            prompt_parts.append(
                f"\n**User Strategy Hint:**\n{strategy_hint}\n"
                f"The user has requested specific strategy preferences. "
                f"Respect this hint when selecting kernel, noise, and acquisition, "
                f"unless it directly conflicts with budget constraints."
            )

        print(f"  - 🤖 BO Agent: Configuring strategy (Batch={batch_size})...")
        resp = self.model.generate_content(prompt_parts, generation_config=self.generation_config)
        raw_config, parse_error = parse_json_from_response(resp)
        if parse_error:
            return {"error": f"JSON Error: {parse_error}"}

        valid_config = self._validate_config(raw_config)
        valid_config["batch_size"] = int(batch_size)

        # Store current config in state
        self.state["current_config"] = valid_config

        # Log strategy selection details
        model_cfg = valid_config.get("model_config", {})
        acq_cfg = valid_config.get("acquisition_strategy", {})
        rationale = valid_config.get("rationale", "No rationale provided")
        print(f"    📋 Strategy config:")
        print(f"       Kernel: {model_cfg.get('kernel', 'N/A')}")
        print(f"       Noise: {model_cfg.get('noise', 'N/A')}")
        print(f"       Acquisition: {acq_cfg.get('type', 'N/A')}")
        acq_params = acq_cfg.get("params", {})
        if acq_params:
            print(f"       Acq params: {acq_params}")
        print(f"       Rationale: {rationale}")

        # 3. Fit Model
        optimizer = get_optimizer(is_moo=is_moo)
        optimizer.fit(
            X, y,
            bounds=input_bounds,
            model_config=valid_config["model_config"],
            feature_names=input_cols,
            fixed_noise_std=fixed_noise_std,
        )
        if fixed_noise_std is not None:
            # Echo the override back into the config so downstream history /
            # trend summaries reflect the noise that was actually used, not
            # the LLM's choice that got bypassed.
            valid_config["model_config"]["noise"] = f"fixed_std={fixed_noise_std}"

        # 4. Recommend (Unconstrained)
        acq_conf = valid_config.get("acquisition_strategy", {})
        strategy_name = acq_conf.get("type", "pareto" if is_moo else "log_ei")
        
        print(f"  - 🚀 Optimizing {strategy_name}...")
        next_x_batch = optimizer.recommend(
            n_candidates=batch_size,
            strategy=strategy_name,
            params=acq_conf.get("params", {})
        )

        # Build unconstrained recommendations (used as reference and fallback)
        unconstrained_recommendations = []
        for row in next_x_batch:
            unconstrained_recommendations.append({k: float(v) for k, v in zip(input_cols, row)})

        # 4b. Constrained Batch Planning
        constrained_metadata = None
        
        if physical_constraints:
            print(f"  - 📐 Physical constraints detected. Generating acquisition landscape...")
            
            acq_summary = self._summarize_acquisition_landscape(
                optimizer=optimizer,
                input_cols=input_cols,
                input_bounds=input_bounds,
                is_moo=is_moo
            )
            
            # Get current best for context (un-negate for display)
            def _unnegate_target_val(col_name, val):
                """Restore original sign for minimization targets."""
                idx = target_cols.index(col_name) if col_name in target_cols else -1
                return -val if idx in minimize_mask else val

            if is_moo:
                current_best = {}
                current_best_value = {}
                try:
                    pareto_indices = optimizer.get_pareto_indices() if hasattr(optimizer, 'get_pareto_indices') else []
                    pareto_front = [
                        {**{k: float(v) for k, v in zip(input_cols, X[i])},
                         **{tc: float(_unnegate_target_val(tc, y[i, j]))
                            for j, tc in enumerate(target_cols)}}
                        for i in pareto_indices
                    ] if len(pareto_indices) > 0 else []
                except Exception:
                    pareto_front = []
            else:
                best_idx = int(np.argmax(y[:, 0]))
                current_best = {k: float(v) for k, v in zip(input_cols, X[best_idx])}
                raw_best = float(y[best_idx, 0])
                current_best_value = {
                    target_cols[0]: float(_unnegate_target_val(target_cols[0], raw_best))
                }
                pareto_front = None

            # df still has original values (only y array was negated),
            # so df.describe() shows natural units for LLM display
            data_summary_str = df.describe().to_markdown()

            constrained_recs, constrained_metadata, constraint_error = self._plan_constrained_batch(
                objective_text=objective_text,
                input_cols=input_cols,
                input_bounds=input_bounds,
                batch_size=batch_size,
                acq_summary=acq_summary,
                physical_constraints=physical_constraints,
                unconstrained_recommendations=unconstrained_recommendations,
                data_summary_str=data_summary_str,
                current_best=current_best,
                current_best_value=current_best_value,
                budget_ctx=budget_ctx,
                is_moo=is_moo,
                pareto_front=pareto_front
            )
            
            if constraint_error:
                print(f"  - ⚠️  Constrained planning failed: {constraint_error}")
                print(f"  - ↩️  Falling back to unconstrained recommendations")
                recommendations = unconstrained_recommendations
            else:
                recommendations = constrained_recs
                next_x_batch = np.array([
                    [rec[col] for col in input_cols] 
                    for rec in recommendations
                ])
                print(f"  - ✅ Using constrained batch ({len(recommendations)} experiments)")
        else:
            recommendations = unconstrained_recommendations

        # 5. Diagnostics
        step_num = len(history) + 1
        n_initial = history[0]["data_points"] if history and "data_points" in history[0] else len(df)
        plot_path = f"{output_dir}/step_{step_num}.png"
        sensitivity_data = {}
        if is_moo:
            optimizer.generate_diagnostics(save_path=plot_path)
        else:
            sensitivity_data = optimizer.generate_diagnostics(next_x_batch, df[target_cols[0]].values.tolist(), save_path=plot_path, n_initial=n_initial) or {}

        # 5b. Acquisition Function Plot & Data (SOO only)
        acq_plot_path = None
        acq_data_path = None
        
        if not is_moo and plot_acq:
            print("  - 📊 BO Agent: Plotting acquisition function...")
            try:
                acq_plot_path = f"{output_dir}/acq_step_{step_num}.png"
                optimizer.plot_acquisition(
                    candidate_x=next_x_batch,
                    save_path=acq_plot_path
                )
                print(f"  - 💾 Acquisition plot saved: {acq_plot_path}")
            except RuntimeError as e:
                logging.warning(f"Could not plot acquisition function: {e}")
                acq_plot_path = None
        
        if not is_moo and save_acq:
            print("  - 💾 BO Agent: Saving acquisition function data...")
            try:
                acq_data_path = f"{output_dir}/acq_data_step_{step_num}.npz"
                acq_meta = optimizer.save_acquisition_data(
                    candidate_x=next_x_batch,
                    save_path=acq_data_path
                )
                print(f"  - 💾 Acquisition data saved: {acq_data_path} "
                      f"(keys: {len(acq_meta['keys'])})")
            except RuntimeError as e:
                logging.warning(f"Could not save acquisition data: {e}")
                acq_data_path = None

        # 6. Inspection
        print("  - 👀 BO Agent: Inspecting visuals...")
        visual_prompt = BO_VISUAL_INSPECTION_MOO_PROMPT if is_moo else BO_VISUAL_INSPECTION_PROMPT
        if sensitivity_data:
            sobol_text = "\n".join(f"  {k}: {v}" for k, v in sensitivity_data.items())
            visual_prompt += (
                f"\n\nACTUAL SOBOL INDEX VALUES (use these exact numbers, "
                f"do not estimate from the bar chart):\n{sobol_text}"
            )
        try:
            img = PIL_Image.open(plot_path)
            insp_resp = self.model.generate_content([visual_prompt, img], generation_config=self.generation_config)
            inspection, _ = parse_json_from_response(insp_resp)
        except Exception as e:
            inspection = {"status": "skipped", "reason": str(e)}

        # 7. Save History
        log_entry = {
            "step": step_num,
            "data_points": len(df),
            "config": valid_config,
            "recommendation_batch": recommendations,
            "inspection": inspection,
            "budget": budget_ctx,
        }
        # Include acquisition paths in history when available
        if acq_plot_path or acq_data_path:
            log_entry["acquisition"] = {
                "strategy": strategy_name,
                "plot_path": acq_plot_path,
                "data_path": acq_data_path,
            }
        # Include constrained planning metadata in history
        if constrained_metadata:
            log_entry["constrained_planning"] = constrained_metadata
        if physical_constraints:
            log_entry["physical_constraints"] = physical_constraints
            
        self._save_history(log_entry)

        # 8. Output
        if batch_size > 1:
            batch_csv = f"{output_dir}/batch_step_{step_num}.csv"
            pd.DataFrame(recommendations).to_csv(batch_csv, index=False)
            print(f"  - 💾 Batch saved: {batch_csv}")

        result = {
            "status": "success",
            "next_parameters": recommendations[0] if batch_size == 1 else recommendations,
            "strategy": valid_config,
            "plot_path": plot_path,
            "inspection": inspection,
            "budget": budget_ctx,
        }
        if acq_plot_path:
            result["acq_plot_path"] = acq_plot_path
        if acq_data_path:
            result["acq_data_path"] = acq_data_path
        # Include constrained planning info in result
        if constrained_metadata:
            result["constrained_planning"] = constrained_metadata
        if physical_constraints:
            result["constraint_aware"] = True
        
        # Log this action to state
        self._log_action(
            action="run_optimization_loop",
            input_ctx={
                "data_path": data_path,
                "input_cols": input_cols,
                "target_cols": target_cols,
                "batch_size": batch_size,
                "experimental_budget": experimental_budget,
                "budget_phase": budget_ctx["budget_phase"],
                "physical_constraints": physical_constraints is not None,
                "save_acq": save_acq,
                "plot_acq": plot_acq,
            },
            result=result,
            rationale=valid_config.get("rationale")
        )
        
        return result