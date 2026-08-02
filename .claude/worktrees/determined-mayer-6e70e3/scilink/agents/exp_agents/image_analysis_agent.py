"""
ImageAnalysisAgent: General-Purpose Image Analysis Agent

This module provides an LLM-driven image analysis agent that handles both
single image analysis and image series analysis using the same unified
architecture. The LLM observes the image, plans an analysis approach,
writes custom Python code, executes it in a sandbox, and verifies quality.

Quality control features:
- LLM-driven quality assessment with task-specific criteria
- Statistical outlier detection for series (may indicate interesting physics)
- Adaptive refit of flagged images with independent approach selection
- Consistency pass to align refitted approaches when a majority agrees
- Human feedback integration for unresolved quality issues

For series analysis:
1. Carefully plan the analysis approach on a representative image
2. Lock the analysis pipeline and strategy for remaining images
3. Detect and flag images where the locked approach fails
4. Adaptive refit: re-analyze flagged images independently with full QC,
   injecting experimental context and series context into the refit prompt
5. Consistency pass: if a majority of refitted images converge on the same
   approach, re-refit outliers using the consensus approach as peer evidence
6. Generate custom analysis code for feature trend visualization
7. Synthesize findings across the series, including refit analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .base_agent import BaseAnalysisAgent, AnalysisInput
from .human_feedback import SimpleFeedbackMixin
from ...executors import ScriptExecutor, require_sandbox_approval
from ..lit_agents.literature_agent import FittingModelLiteratureAgent
from .pipelines.image_analysis_pipelines import create_unified_image_analysis_pipeline
from .controllers.image_analysis_controllers import compute_image_statistics
from ...tools.image_analysis_tools import (
    load_image_data,
    image_to_thumbnail_bytes,
    create_image_montage,
)
from ._deprecation import normalize_params
from ...skills.loader import load_skill

from .instruct import (
    IMAGE_ANALYSIS_INTERPRETATION_INSTRUCTIONS,
    IMAGE_ANALYSIS_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS,
    IMAGE_ANALYSIS_PLANNING_INSTRUCTIONS,
    IMAGE_ANALYSIS_PIPELINE_DISCIPLINE_SUFFIX,
    IMAGE_ANALYSIS_TIER2_PLANNING_INSTRUCTIONS,
    IMAGE_ANALYSIS_TIER2_DECISION_INSTRUCTIONS,
)


logger = logging.getLogger(__name__)


class ImageAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Unified Image Analysis Agent for general-purpose scientific image analysis.

    The LLM plans the analysis approach, writes custom Python code using
    scikit-image, OpenCV, scipy, scikit-learn, etc., executes it in a
    sandboxed environment, and verifies quality through visual inspection
    and quantitative metrics.

    Two-tier analysis pipeline:
    - Tier 1: Foundational analysis (detection, basic measurements)
    - Tier 2: Deep analysis (sublattice separation, strain mapping, etc.)
      conditionally triggered based on Tier 1 findings
    - Controlled via ``analysis_depth``: "auto" (LLM decides), "basic"
      (Tier 1 only), or "deep" (always run both)

    Series support:
    - Single image = series of 1
    - Analysis approach is locked after planning for series consistency
    - Adaptive refit of flagged images with independent approach

    Security:
    - Executes LLM-generated Python code in a sandbox
    - Sandbox check at initialization (Docker/VM/Colab)
    - Use UNSAFE_EXECUTION_OK=true to bypass in CI/CD

    Args:
        api_key: LLM API key
        model_name: LLM model name
        base_url: LLM API base URL
        output_dir: Output directory
        futurehouse_api_key: FutureHouse API key for literature
        use_literature: Enable literature search (default: False)
        analysis_depth: "auto" (default), "basic", or "deep"
        enable_human_feedback: Enable feedback loop
        executor_timeout: Script timeout in seconds
        outlier_sigma: Sigma threshold for outlier detection (default: 2.0)
        max_verification_iterations: Max LLM verification iterations (default: 7)

    Example:
        agent = ImageAnalysisAgent(api_key="...")

        # Single image
        result = agent.analyze("image.tif")

        # With metadata and objective
        result = agent.analyze(
            "image.tif",
            system_info={"sample": "steel alloy", "instrument": "SEM"},
            objective="Measure grain size distribution"
        )

        # Two-tier: basic detection only
        agent = ImageAnalysisAgent(api_key="...", analysis_depth="basic")

        # Two-tier: always run deep analysis
        agent = ImageAnalysisAgent(api_key="...", analysis_depth="deep")

        # With domain skill
        result = agent.analyze("image.tif", skill="atomic_stem")

        # Series with metadata
        result = agent.analyze(
            image_paths,
            series_metadata={
                "variable": "temperature",
                "values": [300, 350, 400, 450, 500],
                "unit": "K"
            }
        )

    Raises:
        RuntimeError: If sandbox check fails and user declines to proceed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-opus-4-6",
        base_url: str | None = None,
        output_dir: str = "image_analysis_output",
        # Deprecated parameters
        google_api_key: str | None = None,
        local_model: str | None = None,
        # Agent configuration
        futurehouse_api_key: str | None = None,
        use_literature: bool = False,
        enable_human_feedback: bool = True,
        executor_timeout: int = 300,
        max_wait_time: int = 1000,
        # Analysis depth
        analysis_depth: str = "auto",
        # Quality control settings
        outlier_sigma: float = 2.0,
        max_verification_iterations: int = 7,
        # Planning settings
        num_plan_candidates: int = 1,
        **kwargs,
    ):
        # ====================================================================
        # SANDBOX CHECK - Must happen first, before any expensive operations
        # ====================================================================
        if not require_sandbox_approval(
            context="ImageAnalysisAgent (image analysis)"
        ):
            raise RuntimeError(
                "ImageAnalysisAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )

        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="ImageAnalysisAgent"
        )

        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
        )

        self.agent_type = "image_analysis"
        self.use_literature = use_literature
        self.output_dir = Path(self.output_dir).resolve()

        # Quality control settings
        self.outlier_sigma = outlier_sigma
        self.max_verification_iterations = max_verification_iterations
        self.num_plan_candidates = num_plan_candidates

        self.executor = ScriptExecutor(timeout=executor_timeout)

        self.analysis_depth = analysis_depth

        # Optional literature agent
        self.literature_agent = None
        if use_literature:
            lit_key = futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY")
            if lit_key:
                try:
                    self.literature_agent = FittingModelLiteratureAgent(
                        api_key=lit_key, max_wait_time=max_wait_time
                    )
                    logger.info("Literature agent initialized")
                except Exception as e:
                    logger.error(f"Literature agent failed: {e}")
            else:
                logger.warning("use_literature=True but no API key provided")

    def _get_initial_state_fields(self) -> dict:
        """Return initial state fields for the agent."""
        return {
            "current_image": None,
            "pipeline_type": "image_analysis_unified",
            "is_series": False,
        }

    def analyze(
        self,
        data: AnalysisInput,
        system_info: Dict[str, Any] | str | None = None,
        objective: str | None = None,
        hints: str | None = None,
        series_metadata: Optional[dict] = None,
        auxiliary_data: Optional[str] = None,
        auxiliary_label: Optional[str] = None,
        skill: Optional[str] = None,
        prior_knowledge: Optional[List[Dict[str, Any]]] = None,
        prior_analysis_paths: Optional[List[str]] = None,
        # Quality control overrides
        outlier_sigma: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified analysis method — handles single images and series identically.

        Single image analysis is internally converted to a series of 1.
        For series, the analysis approach is locked after planning.

        Args:
            data: Input data. Can be:
                - str: Single image path (.npy, .png, .tif, .jpg)
                - List[str]: Multiple image paths (series)
                - np.ndarray: 2D (single grayscale), 3D (single RGB or
                  grayscale stack), or 4D (RGB stack) array
            system_info: Sample/experiment metadata
            objective: High-level scientific objective
            hints: Tactical guidance for analysis
            series_metadata: Metadata about the series::

                    {
                        "variable": "temperature",
                        "values": [300, 350, 400],
                        "unit": "K"
                    }

            auxiliary_data: Path to auxiliary reference data
            auxiliary_label: Description of auxiliary data
            skill: Domain skill name or path to .md skill file
            prior_knowledge: Reference findings from previous analyses
            prior_analysis_paths: List of folder or file paths from previous
                analyses. Folders containing ``analysis_results.json`` surface
                a compact state summary (pipeline, quality score, extracted
                features, scientific claims, saved-arrays catalog) to the
                planner; all paths surface a file listing to the code
                generator so generated scripts can load prior outputs via
                absolute path.
            outlier_sigma: Override default outlier sigma

        Returns:
            Dict with status, detailed_analysis, scientific_claims,
            analysis_approach, extracted_features, output_directory.
            For series: individual_results, trend_analysis, flagged_images.
            When Tier 2 runs: tier1_results, tier2_results sub-dicts
            for traceability.
        """
        # Use provided overrides or fall back to instance defaults
        effective_outlier_sigma = (
            outlier_sigma if outlier_sigma is not None else self.outlier_sigma
        )

        # Parse input
        data_path, data_paths, data_array, error = self._parse_data_input(data)

        if error:
            return {
                "status": "error",
                "error": error,
                "output_directory": str(self.output_dir),
            }

        # Normalize to internal variables
        image_path = data_path
        image_paths = data_paths
        image_stack = data_array

        # Convert single image to series of 1
        if image_path is not None:
            image_paths = [image_path]
            self.logger.info("Single image mode: treating as series of 1")

        # Determine input type and count
        if image_stack is not None:
            # Handle numpy array input
            if image_stack.ndim == 2:
                # 2D: single grayscale image
                image_stack = image_stack[np.newaxis, :, :]
                self.logger.info(
                    f"2D array provided, converted to shape {image_stack.shape}"
                )
            elif image_stack.ndim == 3:
                # 3D: could be single multi-channel (H, W, C) or grayscale stack (N, H, W)
                if image_stack.shape[2] in (2, 3, 4):
                    # Single multi-channel image (2-ch, RGB, or RGBA)
                    image_stack = image_stack[np.newaxis, :, :, :]
                    self.logger.info(
                        f"3D {image_stack.shape[3]}-channel array provided, "
                        f"converted to shape {image_stack.shape}"
                    )
                # else: grayscale stack (N, H, W) — already correct
            elif image_stack.ndim == 4:
                # 4D: RGB stack (N, H, W, C) — already correct
                pass
            else:
                return {
                    "status": "error",
                    "error": {
                        "error": "Invalid shape",
                        "details": f"Array must be 2D, 3D, or 4D, got {image_stack.ndim}D",
                    },
                    "output_directory": str(self.output_dir),
                }

            num_images = image_stack.shape[0]
            input_type = "numpy_array"
        else:
            num_images = len(image_paths)
            input_type = "file_paths"

        is_single_image = num_images == 1

        self.logger.info("")
        self.logger.info(f"🖼️  IMAGE ANALYSIS - {num_images} image{'s' if num_images > 1 else ''}")
        self.logger.info(f"   Quality: depth={self.analysis_depth}")
        if not is_single_image:
            self.logger.info(f"   Outlier detection: {effective_outlier_sigma}σ")

        # Load first image for initial analysis
        if image_stack is not None:
            first_image = image_stack[0]
            first_image_name = "image_0000"
        else:
            try:
                first_image = load_image_data(image_paths[0])
                first_image_name = Path(image_paths[0]).stem
            except Exception as e:
                return {
                    "status": "error",
                    "error": {"error": "Failed to load image", "details": str(e)},
                    "output_directory": str(self.output_dir),
                }

        # Generate thumbnail for LLM
        original_image_bytes = image_to_thumbnail_bytes(first_image)

        # Compute statistics
        image_statistics = self._compute_image_statistics(first_image)

        # Load auxiliary data if provided
        aux_state = {
            "auxiliary_plot_bytes": None,
            "auxiliary_label": None,
            "auxiliary_summary": None,
            "auxiliary_mime_type": None,
        }
        if auxiliary_data:
            aux_state = self._load_auxiliary_data(auxiliary_data, auxiliary_label)
            if aux_state.get("auxiliary_plot_bytes"):
                self.logger.info(
                    f"   📎 Auxiliary data loaded: {aux_state['auxiliary_label']}"
                )

        # Load skill if provided
        skill_state = {"skill_name": None, "skill_sections": None}
        if skill:
            try:
                parsed = load_skill(skill, domain="image_analysis")
                skill_state = {"skill_name": parsed["name"], "skill_sections": parsed}
                self.logger.info(f"   📖 Skill loaded: {parsed['name']}")
            except FileNotFoundError:
                self.logger.warning(
                    f"   Skill '{skill}' not found — proceeding without domain skill"
                )

        # Extract series metadata from system_info if not provided explicitly
        handled_system_info = self._handle_system_info(system_info)
        handled_system_info, series_metadata = self._extract_series_metadata(
            handled_system_info, series_metadata
        )

        # Build initial state
        state = {
            # Input data
            "image_paths": image_paths,
            "image_stack": image_stack,
            "input_type": input_type,
            "num_images": num_images,
            "is_single_image": is_single_image,
            # System info
            "system_info": handled_system_info,
            "series_metadata": series_metadata or {},
            "analysis_hints": hints,
            "analysis_objective": objective,
            # Auxiliary reference data
            **aux_state,
            # Domain skill
            **skill_state,
            # Prior knowledge
            "prior_knowledge": prior_knowledge or [],
            "prior_analysis_paths": prior_analysis_paths or [],
            # First image (for planning)
            "image_path": (
                image_paths[0] if image_paths else first_image_name
            ),
            "image_data": first_image,
            "original_image_bytes": original_image_bytes,
            "image_statistics": image_statistics,
            # Sub-agent preprocessing results
            "fft_preprocessing": None,
            "sam_preprocessing": None,
            # Every planning call gets the same complexity discipline.
            # `analysis_depth` only gates whether in-agent Tier 2 escalation
            # runs after Tier 1; it does not change what the planner is told.
            "planning_instructions_override": (
                IMAGE_ANALYSIS_PLANNING_INSTRUCTIONS
                + IMAGE_ANALYSIS_PIPELINE_DISCIPLINE_SUFFIX
            ),
            # Pipeline state
            "analysis_images": [
                {"label": "Original Image", "data": original_image_bytes}
            ],
            "result_json": {},
            "error_dict": None,
        }

        # Create unified pipeline
        pipeline = create_unified_image_analysis_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            image_to_bytes_fn=image_to_thumbnail_bytes,
            montage_fn=create_image_montage,
            executor=self.executor,
            output_dir=str(self.output_dir),
            literature_agent=self.literature_agent,
            enable_human_feedback=self.enable_human_feedback,
            outlier_sigma=effective_outlier_sigma,
            max_verification_iterations=self.max_verification_iterations,
            num_plan_candidates=self.num_plan_candidates,
        )

        # Execute pipeline
        for i, controller in enumerate(pipeline, 1):
            step_name = controller.__class__.__name__
            self.logger.info(f"\n📍 STEP {i}: {step_name}\n")

            try:
                state = controller.execute(state)

                if state.get("error_dict"):
                    self.logger.error(
                        f"❌ Pipeline failed at {step_name}: {state['error_dict']}"
                    )
                    break

            except Exception as e:
                import traceback
                self.logger.error(
                    f"❌ Pipeline step {step_name} raised exception: {e}\n"
                    f"{traceback.format_exc()}"
                )
                state["error_dict"] = {
                    "error": f"Pipeline step failed: {step_name}",
                    "details": str(e),
                }
                break

        # Handle errors
        if state.get("error_dict"):
            return {
                "status": "error",
                "error": state["error_dict"],
                "output_directory": str(self.output_dir),
            }

        # Compile Tier 1 results
        tier1_results = self._compile_results(state)
        tier1_state = state

        # Save Tier 1 results
        self._save_analysis_scripts(state)

        # ================================================================
        # TIER 2: Evaluate and optionally run
        # ================================================================
        tier2_results = None

        # Common args for _run_tier2 (avoids repeating in both branches)
        _tier2_ctx = dict(
            tier1_state=state, tier1_results=tier1_results,
            first_image=first_image,
            original_image_bytes=original_image_bytes,
            image_statistics=image_statistics,
            handled_system_info=handled_system_info,
            series_metadata=series_metadata, hints=hints,
            objective=objective, skill_state=skill_state,
            aux_state=aux_state, image_paths=image_paths,
            image_stack=image_stack, input_type=input_type,
            num_images=num_images, is_single_image=is_single_image,
            first_image_name=first_image_name,
            effective_outlier_sigma=effective_outlier_sigma,
        )

        if self.analysis_depth == "auto" and tier1_results["status"] == "success":
            tier2_decision = self._evaluate_tier2_needed(
                tier1_results, objective
            )
            if tier2_decision and tier2_decision.get("tier2_needed"):
                run_tier2 = True
                if self.enable_human_feedback:
                    run_tier2, user_guidance = self._prompt_tier2_approval(
                        tier1_results, tier2_decision
                    )
                    if user_guidance:
                        state["tier2_user_guidance"] = user_guidance

                if run_tier2:
                    self.logger.info("\n🔬 TIER 2: Running (auto, approved)")
                    tier2_results = self._run_tier2(
                        **_tier2_ctx, tier2_decision=tier2_decision,
                    )
                else:
                    self.logger.info("\n📊 Tier 2: skipped by user")
            else:
                self.logger.info(
                    "\n📊 Tier 2: not warranted by Tier 1 findings"
                )

        elif self.analysis_depth == "deep" and tier1_results["status"] == "success":
            self.logger.info("\n🔬 TIER 2: Running (analysis_depth='deep')")
            tier2_results = self._run_tier2(**_tier2_ctx)

        # Merge results
        if tier2_results and tier2_results["status"] == "success":
            final_results = self._merge_tiered_results(
                tier1_results, tier2_results
            )
        else:
            final_results = tier1_results

        # Save final merged results
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, "w") as f:
            serializable = self._make_serializable(final_results)
            json.dump(serializable, f, indent=2, default=str)

        self.logger.info("")
        self.logger.info("✅ ANALYSIS COMPLETE")
        self.logger.info(f"   📄 Results: {results_path}")
        if tier2_results:
            self.logger.info("   🔬 Tier 2 deep analysis: included")
        if tier1_state.get("report_path"):
            self.logger.info(f"   📋 Report: {tier1_state['report_path']}")

        flagged = final_results.get("flagged_images", [])
        if flagged:
            self.logger.warning(f"   ⚠️ {len(flagged)} images flagged for review")

        # Log action
        self._log_action(
            action="image_analysis",
            input_ctx={
                "num_images": num_images,
                "input_type": input_type,
                "series_metadata": series_metadata,
                "analysis_depth": self.analysis_depth,
                "tier2_ran": tier2_results is not None,
            },
            result=(
                final_results.get("summary")
                if not is_single_image
                else final_results
            ),
            rationale=f"Approach: {final_results.get('analysis_approach', 'unknown')}",
        )

        return final_results

    @staticmethod
    def _compute_image_statistics(image: np.ndarray) -> dict:
        """Compute statistics for an image."""
        return compute_image_statistics(image)

    def _load_auxiliary_data(
        self, auxiliary_data: str, auxiliary_label: Optional[str]
    ) -> dict:
        """Load auxiliary data and return state fields for pipeline injection."""
        result = {
            "auxiliary_plot_bytes": None,
            "auxiliary_label": auxiliary_label or Path(auxiliary_data).stem,
            "auxiliary_summary": None,
            "auxiliary_mime_type": None,
        }

        if not os.path.exists(auxiliary_data):
            self.logger.warning(f"Auxiliary data file not found: {auxiliary_data}")
            return result

        ext = Path(auxiliary_data).suffix.lower()
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        curve_extensions = {".csv", ".txt", ".dat", ".tsv"}

        try:
            if ext in image_extensions or ext == ".npy":
                img = load_image_data(auxiliary_data)
                result["auxiliary_summary"] = (
                    f"Image with shape {img.shape} (dtype: {img.dtype})."
                )
                result["auxiliary_plot_bytes"] = image_to_thumbnail_bytes(img)
                result["auxiliary_mime_type"] = "image/jpeg"

            elif ext in curve_extensions:
                from ...tools.curve_fitting_tools import (
                    load_curve_data,
                    plot_curve_to_bytes,
                )

                curve = load_curve_data(auxiliary_data)
                if curve.ndim == 2 and curve.shape[0] == 2:
                    curve = curve.T

                if curve.ndim == 2 and curve.shape[1] == 2:
                    x, y = curve[:, 0], curve[:, 1]
                else:
                    x = np.arange(curve.shape[-1])
                    y = curve.flatten()

                result["auxiliary_summary"] = (
                    f"1D curve with {len(x)} points. "
                    f"X: [{float(np.nanmin(x)):.4g}, {float(np.nanmax(x)):.4g}]. "
                    f"Y: [{float(np.nanmin(y)):.4g}, {float(np.nanmax(y)):.4g}]."
                )

                plot_info = {"title": result["auxiliary_label"]}
                plot_data = np.column_stack([x, y])
                result["auxiliary_plot_bytes"] = plot_curve_to_bytes(
                    plot_data, plot_info
                )
                result["auxiliary_mime_type"] = "image/png"
            else:
                self.logger.warning(
                    f"Unrecognized auxiliary file extension: {ext}"
                )

        except Exception as e:
            self.logger.warning(f"Failed to load auxiliary data: {e}")

        return result

    def _save_analysis_scripts(self, state: dict) -> None:
        """Save LLM-generated analysis scripts to disk for reproducibility."""
        scripts_dir = self.output_dir / "scripts"
        saved = []

        if state.get("is_single_image", True):
            script = state.get("final_script")
            if script:
                scripts_dir.mkdir(parents=True, exist_ok=True)
                path = scripts_dir / "analysis_script.py"
                path.write_text(script, encoding="utf-8")
                saved.append(str(path))
        else:
            series_results = state.get("series_results", [])
            for r in series_results:
                script = r.get("script")
                if script and r.get("success"):
                    scripts_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = "".join(
                        c if c.isalnum() or c in ("_", "-") else "_"
                        for c in str(r.get("name", f"image_{r['index']}"))
                    )
                    path = scripts_dir / f"{safe_name}.py"
                    path.write_text(script, encoding="utf-8")
                    saved.append(str(path))

        if saved:
            self.logger.info(f"   📝 Scripts: {scripts_dir} ({len(saved)} file(s))")

    def _evaluate_tier2_needed(
        self, tier1_results: dict, objective: Optional[str]
    ) -> Optional[dict]:
        """Ask the LLM whether Tier 2 deep analysis is warranted."""
        features = tier1_results.get("extracted_features", {})
        claims = tier1_results.get("scientific_claims", [])

        features_str = json.dumps(features, indent=2, default=str)
        if len(features_str) > 3000:
            features_str = features_str[:3000] + "\n... (truncated)"

        claims_str = "\n".join(
            f"- {c.get('claim', '')}" for c in claims[:5]
        ) or "No claims generated."

        prompt = IMAGE_ANALYSIS_TIER2_DECISION_INSTRUCTIONS.format(
            tier1_summary=tier1_results.get("detailed_analysis", "")[:2000],
            tier1_features=features_str,
            tier1_claims=claims_str,
            objective=objective or "General image analysis",
        )

        try:
            response = self.model.generate_content(prompt)
            result, error = self._parse_llm_response(response)
            if error or not result:
                self.logger.warning(f"Tier 2 decision failed: {error}")
                return None
            return result
        except Exception as e:
            self.logger.warning(f"Tier 2 decision error: {e}")
            return None

    def _prompt_tier2_approval(
        self, tier1_results: dict, tier2_decision: dict
    ) -> tuple:
        """Prompt the user for Tier 2 approval.

        Returns:
            (run_tier2, user_guidance) — *run_tier2* is True to proceed,
            *user_guidance* is a string if the user provided guidance, else None.
        """
        focus = tier2_decision.get("suggested_focus", "deeper analysis")
        reasoning = tier2_decision.get("reasoning", "")

        # Clear visual break so the recommendation starts fresh
        print("\n\n")
        print("─" * 60)
        print()
        print("🔬 TIER 2 — DEEP ANALYSIS RECOMMENDATION")
        print()
        print("─" * 60)

        # Brief Tier 1 summary — first 3 lines only
        summary = tier1_results.get("summary", "")
        if not summary:
            summary = tier1_results.get("detailed_analysis", "")
        if summary:
            lines = summary.strip().split("\n")[:3]
            print(f"\n📋 Tier 1 summary:\n   "
                  + " ".join(l.strip() for l in lines)[:300])

        # Concise Tier 2 proposal
        print(f"\n🎯 Proposed analysis:\n   {focus}")
        if reasoning:
            # Keep reasoning to first sentence or 200 chars
            short_reason = reasoning.split(". ")[0].rstrip(".") + "."
            if len(short_reason) > 200:
                short_reason = short_reason[:200] + "..."
            print(f"\n💡 Why: {short_reason}")

        print()
        print("─" * 60)

        try:
            response = input(
                "Proceed with deeper analysis? "
                "(yes / skip / or provide guidance): "
            ).strip()
        except EOFError:
            return True, None

        if not response or response.lower() in ("yes", "y", "proceed"):
            return True, None
        if response.lower() in ("skip", "no", "n"):
            return False, None
        # Anything else is treated as guidance
        return True, response

    def _run_tier2(
        self,
        tier1_state, tier1_results,
        first_image, original_image_bytes, image_statistics,
        handled_system_info, series_metadata, hints, objective,
        skill_state, aux_state,
        image_paths, image_stack, input_type,
        num_images, is_single_image, first_image_name,
        effective_outlier_sigma,
        tier2_decision=None,
    ) -> Optional[dict]:
        """Run Tier 2 pipeline and return compiled results, or None on failure."""
        import shutil

        # Preserve Tier 1 outputs — move to tier1/ subdirectory so the
        # main output dir only contains Tier 2 results + the tier1/ archive.
        tier1_dir = self.output_dir / "tier1"
        tier1_dir.mkdir(exist_ok=True)
        for item in self.output_dir.iterdir():
            if item.name in ("tier1", "tier2") or item.name.startswith("."):
                continue
            dst = tier1_dir / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(item), str(dst))
            elif item.is_file():
                shutil.move(str(item), str(dst))
        self.logger.info(f"   Tier 1 outputs preserved in {tier1_dir}")

        # Copy Tier 1 data artifacts (.npy, .json, .csv) into the
        # corresponding Tier 2 working directories so LLM-generated
        # scripts can load them by filename without absolute paths.
        # We copy rather than symlink because np.save writes through
        # symlinks, which would corrupt the tier1/ archive.
        for tier1_subdir in tier1_dir.iterdir():
            if not tier1_subdir.is_dir():
                continue
            tier2_subdir = self.output_dir / tier1_subdir.name
            tier2_subdir.mkdir(parents=True, exist_ok=True)
            for src in tier1_subdir.iterdir():
                if src.suffix in (".npy", ".json", ".csv"):
                    dst = tier2_subdir / src.name
                    if not dst.exists():
                        try:
                            shutil.copy2(str(src), str(dst))
                        except OSError:
                            pass

        tier2_state = self._build_tier2_state(
            tier1_state, tier1_results,
            first_image, original_image_bytes, image_statistics,
            handled_system_info, series_metadata, hints, objective,
            skill_state, aux_state,
            image_paths, image_stack, input_type,
            num_images, is_single_image, first_image_name,
            tier2_decision=tier2_decision,
        )

        tier2_pipeline = create_unified_image_analysis_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            image_to_bytes_fn=image_to_thumbnail_bytes,
            montage_fn=create_image_montage,
            executor=self.executor,
            output_dir=str(self.output_dir),
            literature_agent=self.literature_agent,
            enable_human_feedback=self.enable_human_feedback,
            outlier_sigma=effective_outlier_sigma,
            max_verification_iterations=self.max_verification_iterations,
            num_plan_candidates=self.num_plan_candidates,
        )

        for i, controller in enumerate(tier2_pipeline, 1):
            step_name = controller.__class__.__name__
            self.logger.info(f"\n📍 TIER 2 STEP {i}: {step_name}\n")
            try:
                tier2_state = controller.execute(tier2_state)
                if tier2_state.get("error_dict"):
                    self.logger.error(
                        f"Tier 2 failed at {step_name}: "
                        f"{tier2_state['error_dict']}"
                    )
                    break
            except Exception as e:
                self.logger.error(
                    f"Tier 2 step {step_name} raised exception: {e}"
                )
                tier2_state["error_dict"] = {
                    "error": f"Tier 2 step failed: {step_name}",
                    "details": str(e),
                }
                break

        if not tier2_state.get("error_dict"):
            tier2_results = self._compile_results(tier2_state)
            self._save_analysis_scripts(tier2_state)
            return tier2_results

        return None

    def _build_tier2_state(
        self, tier1_state, tier1_results,
        first_image, original_image_bytes, image_statistics,
        handled_system_info, series_metadata, hints, objective,
        skill_state, aux_state,
        image_paths, image_stack, input_type,
        num_images, is_single_image, first_image_name,
        tier2_decision=None,
    ) -> dict:
        """Build state dict for Tier 2 pipeline run."""
        # Collect Tier 1 output files, excluding the tier1/ archive to
        # avoid duplicate entries (copies already live in the working dirs).
        tier1_archive = self.output_dir / "tier1"
        tier1_files = []
        for ext in ("*.npy", "*.json", "*.png"):
            for f in self.output_dir.rglob(ext):
                if not f.is_relative_to(tier1_archive):
                    tier1_files.append(str(f))

        features = tier1_results.get("extracted_features", {})
        claims = tier1_results.get("scientific_claims", [])

        features_str = json.dumps(features, indent=2, default=str)
        if len(features_str) > 3000:
            features_str = features_str[:3000] + "\n... (truncated)"

        claims_str = "\n".join(
            f"- {c.get('claim', '')}" for c in claims[:5]
        ) or "No claims."

        # Build file listing with array descriptions where available
        saved_arrays = tier1_state.get("analysis_result", {}).get(
            "saved_arrays", {}
        )
        file_lines = []
        for f in tier1_files[:20]:
            fname = Path(f).name
            meta = saved_arrays.get(fname)
            if meta:
                desc = meta.get("description", "")
                shape = meta.get("shape", "")
                dtype = meta.get("dtype", "")
                file_lines.append(
                    f"- `{f}` — {desc} (shape={shape}, dtype={dtype})"
                )
            else:
                file_lines.append(f"- {f}")
        files_str = "\n".join(file_lines)

        # Build Tier 2 planning instructions. Append the pipeline-discipline
        # suffix so Tier 2 stays bounded to the same complexity ceiling as
        # Tier 1 — Tier 2 builds on Tier 1, it is not a more elaborate pipeline.
        tier2_instructions = IMAGE_ANALYSIS_TIER2_PLANNING_INSTRUCTIONS.format(
            tier1_summary=tier1_results.get("detailed_analysis", "")[:2000],
            tier1_features=features_str,
            tier1_claims=claims_str,
            tier1_files=files_str,
        ) + IMAGE_ANALYSIS_PIPELINE_DISCIPLINE_SUFFIX

        if tier2_decision and tier2_decision.get("suggested_focus"):
            tier2_instructions += (
                f"\n\n**Suggested focus:** {tier2_decision['suggested_focus']}"
            )

        # Include user guidance from CO_PILOT mode if provided
        user_guidance = tier1_state.get("tier2_user_guidance")
        if user_guidance:
            tier2_instructions += (
                f"\n\n**User guidance:** {user_guidance}"
            )

        return {
            # Input data (same as Tier 1)
            "image_paths": image_paths,
            "image_stack": image_stack,
            "input_type": input_type,
            "num_images": num_images,
            "is_single_image": is_single_image,
            # System info
            "system_info": handled_system_info,
            "series_metadata": series_metadata or {},
            "analysis_hints": hints,
            "analysis_objective": objective,
            # Auxiliary
            **aux_state,
            # Skill
            **skill_state,
            # Prior knowledge
            "prior_knowledge": tier1_state.get("prior_knowledge", []),
            "prior_analysis_paths": tier1_state.get(
                "prior_analysis_paths", []
            ),
            # Sub-agent results
            "fft_preprocessing": None,
            "sam_preprocessing": None,
            # First image
            "image_path": (
                image_paths[0] if image_paths else first_image_name
            ),
            "image_data": first_image,
            "original_image_bytes": original_image_bytes,
            "image_statistics": image_statistics,
            # Tier 2 planning instructions override
            "planning_instructions_override": tier2_instructions,
            # Pipeline state
            "analysis_images": [
                {"label": "Original Image", "data": original_image_bytes}
            ],
            "result_json": {},
            "error_dict": None,
        }

    def _merge_tiered_results(
        self, tier1: dict, tier2: dict
    ) -> dict:
        """Merge Tier 1 and Tier 2 results into a unified output."""
        merged = tier1.copy()

        # Merge extracted features (Tier 2 overwrites on conflict)
        t1_features = tier1.get("extracted_features", {})
        t2_features = tier2.get("extracted_features", {})
        if t1_features or t2_features:
            merged_features = {}
            merged_features.update(t1_features)
            merged_features.update(t2_features)
            merged["extracted_features"] = merged_features

        # Combine claims (deduplicate by claim text)
        t1_claims = tier1.get("scientific_claims", [])
        t2_claims = tier2.get("scientific_claims", [])
        seen = {c.get("claim", "") for c in t1_claims}
        for c in t2_claims:
            if c.get("claim", "") not in seen:
                t1_claims.append(c)
                seen.add(c.get("claim", ""))
        merged["scientific_claims"] = t1_claims

        # Use Tier 2 detailed analysis if available (more comprehensive)
        if tier2.get("detailed_analysis"):
            merged["detailed_analysis"] = tier2["detailed_analysis"]

        # Keep both tier results for traceability
        merged["tier1_results"] = tier1
        merged["tier2_results"] = tier2

        return merged

    def _compile_results(self, state: dict) -> Dict[str, Any]:
        """Compile results into a consistent output structure."""
        is_single = state.get("is_single_image", True)
        num_images = state.get("num_images", 1)
        series_results = state.get("series_results", [])
        synthesis = state.get("synthesis_result", {})
        flagged_images = state.get("flagged_images", [])

        results = {
            "status": "success",
            "output_directory": str(self.output_dir),
        }

        if is_single:
            # Single image: compact structure
            analysis_result = state.get("analysis_result", {})

            results["detailed_analysis"] = synthesis.get("detailed_analysis")
            results["scientific_claims"] = self._validate_scientific_claims(
                synthesis.get("scientific_claims", [])
            )
            results["analysis_approach"] = state.get(
                "locked_analysis_config", {}
            ).get("analysis_approach")
            results["literature_files"] = state.get("literature_files")

            if series_results and series_results[0].get("quality_warning"):
                results["quality_warning"] = series_results[0]["quality_warning"]
            if series_results and series_results[0].get("quality_history"):
                results["quality_history"] = series_results[0]["quality_history"]

        else:
            # Series: full structure with trends and flagged images
            successful = sum(
                1 for r in series_results if r.get("success", False)
            )

            refit_summary = state.get("refit_summary", [])
            results["summary"] = {
                "total_images": num_images,
                "successful_analyses": successful,
                "flagged_count": len(flagged_images),
                "refitted_count": sum(
                    1 for r in refit_summary if r.get("improved")
                ),
                "input_type": state.get("input_type"),
                "locked_approach": state.get(
                    "locked_analysis_config", {}
                ).get("analysis_approach"),
                "is_single_image": False,
            }

            results["detailed_analysis"] = synthesis.get("detailed_analysis", "")
            results["scientific_claims"] = self._validate_scientific_claims(
                synthesis.get("scientific_claims", [])
            )

            results["individual_results"] = [
                {
                    "index": r["index"],
                    "name": r["name"],
                    "success": r["success"],
                    "analysis_type": r.get("analysis_type"),
                    "visualization_path": r.get("visualization_path"),
                    "error": r.get("error"),
                    "flagged": r.get("flagged", False),
                    "flag_reason": r.get("flag_reason"),
                    "adaptively_refitted": r.get("adaptively_refitted", False),
                    "verification_score": (
                        r.get("quality_history", {}).get("final_score")
                        if r.get("quality_history") else None
                    ),
                }
                for r in series_results
            ]

            results["flagged_images"] = flagged_images
            results["flagged_images_analysis"] = synthesis.get(
                "flagged_images_analysis", {}
            )
            results["refit_summary"] = refit_summary
            results["refit_analysis"] = synthesis.get("refit_analysis", {})
            results["trend_analysis"] = state.get("trend_analysis_results", {})
            results["feature_trends"] = synthesis.get("feature_trends", {})
            results["caveats"] = synthesis.get("caveats", "")
            results["literature_files"] = state.get("literature_files")
            results["locked_analysis_config"] = state.get(
                "locked_analysis_config"
            )

            # Aggregate verification summary
            vh_entries = [
                r.get("quality_history") for r in series_results
                if r.get("quality_history")
            ]
            if vh_entries:
                scores = [
                    v["final_score"] for v in vh_entries
                    if v.get("final_score") is not None
                ]
                results["verification_summary"] = {
                    "verified_count": len(vh_entries),
                    "approved_count": sum(
                        1 for v in vh_entries if v.get("approved")
                    ),
                    "score_range": (
                        [min(scores), max(scores)] if scores else None
                    ),
                    "mean_score": (
                        sum(scores) / len(scores) if scores else None
                    ),
                }

        return results

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, bytes):
            return None
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    # =========================================================================
    # BACKWARD COMPATIBLE METHODS
    # =========================================================================

    def analyze_image_series(
        self,
        image_paths: Optional[List[str]] = None,
        image_stack: Optional[np.ndarray] = None,
        system_info: Optional[Union[dict, str]] = None,
        series_metadata: Optional[dict] = None,
        objective: str | None = None,
        hints: str | None = None,
    ) -> Dict[str, Any]:
        """
        Analyze a series of images.

        BACKWARD COMPATIBLE: Delegates to unified analyze() method.
        """
        if image_paths is not None:
            return self.analyze(
                image_paths,
                system_info=system_info,
                series_metadata=series_metadata,
                hints=hints,
                objective=objective,
            )
        elif image_stack is not None:
            return self.analyze(
                image_stack,
                system_info=system_info,
                series_metadata=series_metadata,
                hints=hints,
                objective=objective,
            )
        else:
            return {
                "status": "error",
                "error": {
                    "error": "No input",
                    "details": "Must provide image_paths or image_stack",
                },
                "output_directory": str(self.output_dir),
            }

    def _get_claims_instruction_prompt(self) -> str:
        return IMAGE_ANALYSIS_INTERPRETATION_INSTRUCTIONS

    def _get_measurement_recommendations_prompt(self) -> str:
        return IMAGE_ANALYSIS_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
