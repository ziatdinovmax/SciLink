"""
Hyperspectral Analysis Agent
"""


import json
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any

from .base_agent import BaseAnalysisAgent, AnalysisInput
from .instruct import (
    SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
    SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)
from .human_feedback import SimpleFeedbackMixin
from .preprocess import HyperspectralPreprocessingAgent
from .pipelines.hyperspectral_pipelines import (
    create_hyperspectral_iteration_pipeline,
    create_hyperspectral_synthesis_pipeline
)
from ...skills._shared.image_processor import load_image, convert_numpy_to_jpeg_bytes
from ...skills._shared.curve_fitting_tools import load_curve_data, plot_curve_to_bytes
from ...executors import require_sandbox_approval, ScriptExecutor
from ...skills.loader import load_skill
from ...utils.text_io import read_text_utf8

from ._deprecation import normalize_params
from .controllers import hyperspectral_series as _series


from ._reference_scripts import load_reference_scripts as _load_reference_scripts

def _empty_auxiliary_state() -> dict:
    """Default auxiliary state — no companion datasets loaded. ``auxiliary_items``
    is the list of per-dataset dicts (label / array / axis / plot_bytes /
    summary / mime_type); labels become operand keys downstream. (#226)"""
    return {"auxiliary_items": []}


class HyperspectralAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Hyperspectral Analysis Agent.

    Single-pass pipeline: preprocess → optional NMF/PCA/ICA decomposition
    (gated by an LLM that may skip it for direct per-pixel objectives) →
    LLM interpretation → optional dynamic-analysis (custom-code) refinement
    → synthesis + HTML report.

    Features:
        - Automatic component number selection via elbow method (NMF/PCA)
        - Optional structure image correlation
        - Human-in-the-loop feedback
        - HTML report generation

    Example:
        agent = HyperspectralAnalysisAgent(api_key="...")

        # Single file
        result = agent.analyze("spectrum.npy")

        # With metadata
        result = agent.analyze(
            "spectrum.npy",
            system_info={"sample": "TiO2", "technique": "EELS"}
        )

        # Get measurement recommendations
        recommendations = agent.recommend_measurements(analysis_result=result)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-opus-4-6",
        base_url: str | None = None,
        output_dir: str = "hyperspectral_analysis_output",
        # Deprecated params
        google_api_key: str | None = None,
        local_model: str | None = None,
        # Agent specific params
        spectral_unmixing_settings: dict | None = None,
        run_preprocessing: bool = True,
        enable_human_feedback: bool = True,
        executor_timeout: int = 600,
        # Retry budget for the dynamic-analysis codegen loop (#271):
        # initial attempt + N retries. 0 => single attempt, accepted when
        # the task succeeds (fast/in-situ), salvage path otherwise. None
        # keeps the built-in default (4 retries, 5 total attempts).
        max_verification_iterations: int | None = None,
    ):
        
        if not require_sandbox_approval(
            context="HyperspectralAnalysisAgent (hyperspectral analysis)"
        ):
            raise RuntimeError(
                "HyperspectralAnalysisAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )
        
        # Normalize params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model,
            source="HyperspectralAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )

        self.agent_type = "hyperspectral"
        
        # Settings
        default_settings = {
            'method': 'nmf',
            'n_components': 4,
            'normalize': True,
            'enabled': True,
            'auto_components': True,
            'min_auto_components': 2,
            'max_auto_components': 8,
            'enable_human_feedback': enable_human_feedback
        }
        self.spectral_settings = spectral_unmixing_settings if spectral_unmixing_settings else default_settings
        self.spectral_settings['run_preprocessing'] = run_preprocessing
        self.spectral_settings['output_dir'] = str(self.output_dir)
        self.spectral_settings['feedback_depths'] = [0]
        
        # Sub-agent initialization. Pass executor_timeout so the
        # preprocessor's custom-script execution honors the same limit
        # the user set on the parent agent.
        self.executor_timeout = executor_timeout
        self.max_verification_iterations = max_verification_iterations
        # Series bookkeeping: a per-dataset child agent created by
        # _analyze_series carries its role ("anchor" / "replay" / "refit").
        # Replay children replay the anchor's scripts verbatim, so they must
        # not re-select skills, re-bank the anchor's script or stage T=2
        # solutions — those are once-per-series decisions the parent owns.
        self._series_role: str | None = None
        self._skill_autoselect: bool = True
        preprocess_dir = self.output_dir / "preprocessing"
        self.preprocessor = HyperspectralPreprocessingAgent(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=str(preprocess_dir),
            executor_timeout=executor_timeout,
        )

        # Pipeline initialization. Shared kwargs go into pipeline_args;
        # executor_timeout is iteration-only (the codegen sandbox lives
        # in the iteration pipeline; the synthesis pipeline runs LLM
        # calls + plotting only and has no code-execution step).
        pipeline_args = {
            "model": self.model,
            "logger": self.logger,
            "generation_config": self.generation_config,
            "safety_settings": self.safety_settings,
            "settings": self.spectral_settings,
            "parse_fn": self._parse_llm_response,
        }

        self.iteration_pipeline = create_hyperspectral_iteration_pipeline(
            **pipeline_args,
            preprocessor=self.preprocessor,
            executor_timeout=executor_timeout,
            max_verification_iterations=max_verification_iterations,
        )
        self.synthesis_pipeline = create_hyperspectral_synthesis_pipeline(
            **pipeline_args,
            store_fn=self._store_analysis_images
        )
        
        self.logger.info(f"HyperspectralAnalysisAgent initialized. Output: {self.output_dir}")

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        return {
            "data_path": None,
            "analysis_depth": 0,
            "components_found": []
        }

    # =========================================================================
    # PRIMARY ENTRY POINT
    # =========================================================================

    @staticmethod
    def _plain_total_dynamic_failure(records) -> bool:
        """True when EVERY dynamic-analysis target failed PLAINLY — no
        success, nothing salvaged, and no honest not-measurable resolution
        (which is a legitimate answered outcome, not a silent failure)."""
        records = records or []
        if not records or any(r.get("task_success") for r in records):
            return False
        plain = [r for r in records
                 if not r.get("salvaged") and not r.get("not_measurable")]
        return bool(plain)

    def analyze(
        self,
        data: AnalysisInput,
        system_info: Dict[str, Any] | str | None = None,
        # Hyperspectral-specific options
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        objective: str | None = None,
        hints: str | None = None,
        skill: str | None = None,
        skill_hint: str | list[str] | None = None,
        custom_skills: dict | None = None,
        prior_knowledge: list | None = None,
        auxiliary_data: str | list[str] | None = None,
        auxiliary_label: str | list[str] | None = None,
        literature_file: str | None = None,
        # Per-call thoroughness override (fast/in-situ vs thorough) of the
        # dynamic-analysis retry budget (#271). 0 => single codegen attempt,
        # accepted when the task succeeds; higher => more retries. None
        # falls back to the construction default.
        max_verification_iterations: int | None = None,
        # User-attached scripts to ADAPT (reference, like a script-bank hit).
        reference_scripts: list | None = None,
        # Locked-script replay (harmonized re-run across sibling datasets):
        # prior_analysis_paths + reuse_locked_script=True replays the prior
        # run's APPROVED dynamic-analysis script(s) verbatim on THIS cube —
        # no fresh analysis plan, no codegen, decomposition skipped; the
        # per-map QC still verifies every output on this dataset. Mirrors
        # the curve/image locked-reuse surface (#172).
        prior_analysis_paths: list | None = None,
        reuse_locked_script: bool = False,
        # Series analysis (list of cubes along a control variable). Mirrors
        # the image / curve agents: series_metadata may also ride inside
        # system_info["series"]; max_series_refits budgets the independent
        # re-analysis of datasets the locked recipe failed on (None =
        # unlimited, 0 = none); outlier_sigma flags feature outliers.
        series_metadata: dict | None = None,
        max_series_refits: int | None = None,
        outlier_sigma: float = 2.0,
        # Locked TARGETS, fresh code: the analysis targets and their required
        # output names are fixed (a series' schema source), decomposition and
        # planning are skipped, and the code is regenerated through the full
        # codegen ladder. Used by the series driver for regime anchors and
        # adaptive refits so every dataset reports the same quantities.
        locked_targets: list | None = None,
        # Series fan-out: locked replays run concurrently on this many
        # workers (None → SCILINK_HS_SERIES_WORKERS env var → 1, serial).
        series_workers: int | None = None,
        # Locked replay: the anchor's per-map stats ({name: {min,max,mean}})
        # for the deterministic replay gate.
        replay_reference: dict | None = None,
        # Operating profile (#346): plumbed for parity with the curve agent;
        # realtime toggles are wired for curve only in v1 (hyperspectral
        # per-frame cost is numerics-dominated). Thorough is unaffected.
        profile: Any = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Primary analysis entry point for hyperspectral data.

        Args:
            data: Input data. Can be:
                - str: Path to a .npy / .h5 hyperspectral data file
                - List[str]: A SERIES of datacubes (one per value of a
                  control variable). The first dataset is analysed in full;
                  its approved dynamic-analysis scripts are locked and
                  replayed verbatim on every later dataset, then failures
                  are re-analysed, outliers flagged, trends extracted and
                  the series synthesized. A one-element list is a single
                  analysis.
                - np.ndarray: 4D (N, H, W, E) stack → series (each cube
                  is written to <output_dir>/series_input/); 3D arrays
                  are not accepted directly (save to .npy first).
            system_info: Metadata dictionary or path to metadata file
            structure_image_path: Optional path to structural reference image
            structure_system_info: Optional metadata for structure image
            objective: Optional high-level scientific objective that frames
                the entire analysis (e.g., "Determine the oxidation state
                of Ti from the L-edge fine structure", "Map the spatial
                distribution of phase segregation"). Unlike hints which
                guide *how* to analyze, objective specifies *why* you are
                analyzing and *what question* to answer.
            hints: Optional tactical guidance to steer analysis (e.g.,
                "focus on the Ti L-edge around 460 eV"). The agent will
                prioritize these suggestions but still report other
                significant features.
            skill: Optional domain skill name (e.g., "eels") or path to a
                custom ``.md`` skill file. Injects domain-specific knowledge
                into LLM prompts for planning and interpretation stages.
            prior_knowledge: Optional list of knowledge entries synthesized
                from prior reference analyses. Automatically injected into
                LLM prompts to guide analysis approach and interpretation.
            auxiliary_data: Optional path to a complementary dataset (1D
                curve file or image) provided as context for the analysis.
                The agent will consider this data in its interpretation but
                will not attempt to unmix or quantitatively analyze it.
            auxiliary_label: Optional human-readable label for the auxiliary
                data (e.g., "TGA curve collected simultaneously").
            literature_file: Optional path to a pre-fetched literature report
                (typically written by the orchestrator's ``search_literature``
                tool). Its text is injected as advisory context into the
                target-planning, interpretation, and synthesis prompts.
                Previously this argument was silently swallowed by
                ``**kwargs`` — hyperspectral had no literature integration.
            max_verification_iterations: Per-call override of the
                dynamic-analysis retry budget (initial attempt + N
                retries). 0 runs a single codegen attempt and accepts it
                when the task succeeds (fast/in-situ); a failed attempt
                still goes through the salvage path. None uses the
                construction default.
            series_metadata: For a series — ``{"variable": "temperature",
                "values": [300, 350, ...] | {filename: value}, "unit": "K"}``.
                May also be supplied as ``system_info["series"]``; the
                explicit argument wins. Values keyed by filename are
                aligned to the list order.
            max_series_refits: Budget for the independent re-analysis of
                datasets the locked recipe failed on (None = unlimited,
                0 = none). Statistical outliers are never re-analysed.
            outlier_sigma: Per-feature deviation (in standard deviations
                from the series mean) beyond which a dataset is flagged.
            **kwargs: Additional options

        Returns:
            dict containing:
                - "status": "success" | "error"
                - "detailed_analysis": str
                - "scientific_claims": list[dict]
                - "output_directory": str
                - "error": dict (when status="error")

        Examples:
            # Single file
            result = agent.analyze("spectrum.npy")

            # With metadata and structure image
            result = agent.analyze(
                "spectrum.npy",
                system_info={"sample": "TiO2", "technique": "EELS"},
                structure_image_path="stem_image.png"
            )
        """
        # Operating profile (#346): accepted for surface parity; see the
        # parameter note — realtime is curve-only in v1.
        # Use the per-call override or fall back to the instance default
        # (None = the controller's built-in retry budget).
        effective_max_verification = (
            max_verification_iterations if max_verification_iterations is not None
            else self.max_verification_iterations)
        if effective_max_verification is not None and effective_max_verification < 0:
            raise ValueError("max_verification_iterations must be >= 0")

        # --- Locked-script replay (harmonized re-run) -----------------------
        reuse_records = None
        if reuse_locked_script:
            if not prior_analysis_paths:
                return {
                    "status": "error",
                    "error": {
                        "error": "reuse_locked_script requires prior_analysis_paths",
                        "details": ("Pass prior_analysis_paths=[<prior "
                                    "hyperspectral run dir>] whose "
                                    "dynamic_analysis_records.json holds the "
                                    "approved script(s) to replay."),
                    },
                    "output_directory": str(self.output_dir),
                }
            reuse_records = self._load_prior_dynamic_records(prior_analysis_paths)
            if not reuse_records:
                return {
                    "status": "error",
                    "error": {
                        "error": "No approved prior script to replay",
                        "details": (
                            f"None of {list(prior_analysis_paths)!r} carries a "
                            "dynamic_analysis_records.json with an approved "
                            "(task_success) script. Run the donor analysis "
                            "first, then point prior_analysis_paths at its "
                            "result directory."),
                    },
                    "output_directory": str(self.output_dir),
                }
            # Verbatim replay is a single-attempt contract: a failure must be
            # reported (or salvaged), never regenerated into a different
            # method — that would silently break cross-dataset comparability.
            effective_max_verification = 0
            self.logger.info(
                f"🔒 Locked-script replay: {len(reuse_records)} approved "
                f"prior script(s) will be executed verbatim (no fresh plan, "
                f"no codegen; retry budget forced to 0).")
        elif prior_analysis_paths:
            self.logger.warning(
                "prior_analysis_paths without reuse_locked_script is not "
                "used by the hyperspectral agent yet — ignoring. Pass "
                "reuse_locked_script=True for a locked replay.")

        from ._qc_profile import resolve_profile
        if resolve_profile(profile).name == "realtime":
            self.logger.warning(
                "profile='realtime' is not wired for hyperspectral analysis "
                "yet (per-frame cost is numerics-dominated); running under "
                "the thorough profile."
            )

        # Parse input
        data_path, data_paths, data_array, error = self._parse_data_input(data)
        
        if error:
            return {
                "status": "error",
                "error": error,
                "output_directory": str(self.output_dir)
            }
        
        # A 4D stack is a series of cubes; materialize it as files so every
        # dataset goes through the same file-based single-cube path.
        if data_array is not None:
            if data_array.ndim == 4:
                stack_dir = self.output_dir / "series_input"
                stack_dir.mkdir(parents=True, exist_ok=True)
                data_paths = []
                for i in range(data_array.shape[0]):
                    p = stack_dir / f"cube_{i:04d}.npy"
                    np.save(p, data_array[i])
                    data_paths.append(str(p))
                data_array = None
            else:
                return {
                    "status": "error",
                    "error": {
                        "error": "Direct array input not supported",
                        "details": ("Save the array to a .npy file and pass "
                                    "the path (a 4D (N, H, W, E) stack is "
                                    "accepted as a series)."),
                    },
                    "output_directory": str(self.output_dir)
                }

        # Series of datacubes → anchor + locked replay driver. A one-element
        # list is simply a single analysis.
        if data_paths is not None:
            if len(data_paths) == 1:
                data_path = data_paths[0]
            else:
                if reuse_locked_script:
                    self.logger.warning(
                        "reuse_locked_script applies to the anchor only in "
                        "series mode: the anchor replays the prior scripts, "
                        "and the series then locks on the anchor's records.")
                return self._analyze_series(
                    data_paths, system_info=system_info,
                    series_metadata=series_metadata,
                    structure_image_path=structure_image_path,
                    structure_system_info=structure_system_info,
                    objective=objective, hints=hints, skill=skill,
                    skill_hint=skill_hint, custom_skills=custom_skills,
                    prior_knowledge=prior_knowledge,
                    literature_file=literature_file,
                    max_verification_iterations=effective_max_verification,
                    reference_scripts=reference_scripts,
                    prior_analysis_paths=prior_analysis_paths,
                    reuse_locked_script=reuse_locked_script,
                    max_series_refits=max_series_refits,
                    outlier_sigma=outlier_sigma,
                    series_workers=series_workers,
                )

        # Initialize and Run Pipeline
        self._init_state(data_path=data_path, metadata=system_info)
        # State file from the start, for the Worker-agents panel (#566).
        self._log_action("analysis_started", {"data_path": data_path},
                         {"status": "running"})

        # Load skill(s) if provided. Accepts a single name/path or a list
        # — see PR 3 multi-skill support.
        skill_state = self._load_skills_to_state(skill, domain="hyperspectral")

        # Auto-select skill(s) when none were explicitly provided, mirroring
        # the image/curve agents. Conservative, technique-aware (issue #251);
        # may pick zero, one, or several skills from the metadata.
        if not skill_state.get("skills_loaded") and getattr(self, "_skill_autoselect", True):
            selected = self._auto_select_skills(
                system_info, hint=skill_hint, custom_skills=custom_skills
            )
            if selected:
                # Resolve any selected custom-skill name to its registered path.
                cs = custom_skills or {}
                resolved = [cs.get(n, n) for n in selected]
                skill_state = self._load_skills_to_state(
                    resolved, domain="hyperspectral"
                )

        # Load auxiliary data if provided (one or several companion datasets).
        auxiliary_state = _empty_auxiliary_state()
        if auxiliary_data:
            auxiliary_state = self._load_auxiliary_items(
                auxiliary_data, auxiliary_label
            )
            n = len(auxiliary_state.get("auxiliary_items", []))
            if n:
                names = ", ".join(it["label"] for it in auxiliary_state["auxiliary_items"])
                self.logger.info(f"   Auxiliary data loaded ({n}): {names}")

        # Pre-fetched literature (Channel A passthrough — mirrors the curve /
        # image agents' literature_file handling).
        literature_context = None
        literature_files = None
        if literature_file:
            lit_p = Path(literature_file)
            if lit_p.is_file():
                literature_context = read_text_utf8(lit_p)
                literature_files = {"provided_file": str(lit_p)}
                self.logger.info(f"📚 Loaded literature context from {lit_p.name}")
            else:
                self.logger.warning(f"literature_file not found: {literature_file}")

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 HYPERSPECTRAL ANALYSIS")
        self.logger.info(f"   Data: {data_path}")
        self.logger.info(f"{'='*80}\n")

        # Run the analysis pipeline
        result_json, error_dict = self._run_analysis_pipeline(
            data_path=data_path,
            system_info=system_info,
            instruction_prompt=SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info,
            hints=hints,
            reference_scripts=reference_scripts,
            objective=objective,
            skill_state=skill_state,
            prior_knowledge=prior_knowledge or [],
            auxiliary_state=auxiliary_state,
            literature_context=literature_context,
            max_verification_iterations=effective_max_verification,
            reuse_records=reuse_records,
            locked_targets=locked_targets,
            replay_reference=replay_reference,
        )
        
        # Handle Errors
        if error_dict:
            self._log_action("analyze", {"data": data_path}, {"error": error_dict})
            return {
                "status": "error",
                "error": error_dict,
                "output_directory": str(self.output_dir)
            }
        
        if result_json is None:
            return {
                "status": "error",
                "error": {
                    "error": "Analysis failed",
                    "details": "Pipeline returned no results"
                },
                "output_directory": str(self.output_dir)
            }
        
        # Process Successful Results
        valid_claims = self._validate_scientific_claims(
            result_json.get("scientific_claims", [])
        )
        
        # Build Response
        response = {
            "status": "success",
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis not provided."),
            "scientific_claims": valid_claims,
            "output_directory": str(self.output_dir)
        }

        # Surface the dynamic-analysis features at top level (additive) —
        # previously trapped in custom_analysis_metadata_list, which blocked
        # feature-conditioned literature (#323) and cross-mode consumers.
        if result_json.get("extracted_features"):
            response["extracted_features"] = result_json["extracted_features"]
        # Per-target verification records (HS-1) — the hyperspectral
        # counterpart of curve/image quality_history.
        if result_json.get("dynamic_analysis_records"):
            response["dynamic_analysis_records"] = result_json["dynamic_analysis_records"]
        if reuse_records:
            _new_recs = result_json.get("dynamic_analysis_records") or []
            _supplied = {r.get("script") for r in reuse_records}
            # #518: a replay that could not reproduce a mask-scoped donor's
            # scoping ran full-frame — a degraded harmonization the caller
            # (and fusion) must see, not a silent methodological drift.
            _scope_warnings = [
                (nr or {}).get("replay_scope_degraded") for nr in _new_recs
                if (nr or {}).get("replay_scope_degraded")]
            response["script_reuse"] = {
                "prior_analysis_paths": [str(p) for p in prior_analysis_paths],
                "n_replayed": len(reuse_records),
                # False when a mechanical execution repair had to modify a
                # script — the run then is NOT byte-comparable to the donor.
                "verbatim": bool(_new_recs) and all(
                    (nr or {}).get("script") in _supplied for nr in _new_recs),
                **({"scope_degraded": True,
                    "scope_warnings": _scope_warnings}
                   if _scope_warnings else {}),
            }
        if literature_files:
            response["literature_files"] = literature_files

        # A salvage / approximate / withheld outcome is NOT a clean success:
        # downgrade the status and surface the honest caveats so a programmatic
        # caller (or the meta agent) sees the uncertainty instead of a bare
        # 'success'. Notes come from the dynamic-analysis salvage judge.
        degradation = result_json.get("degradation_notes", [])
        if degradation:
            _rank = {"none": 0, "low": 1, "medium": 2}
            worst = min(degradation,
                        key=lambda d: _rank.get(d.get("confidence", "low"), 1))
            response["status"] = "partial"
            response["confidence"] = worst.get("confidence", "low")
            response["warnings"] = [d["caveat"] for d in degradation if d.get("caveat")]
            response["degraded_outputs"] = degradation

        # Total dynamic-analysis failure with nothing salvageable used to
        # slip through as a bare "success" with empty features: degradation
        # notes exist only when some maps passed QC (valid_count > 0), so a
        # zero-valid total failure recorded nothing. Downgrade honestly —
        # unless every failed target was resolved through the honest
        # not-measurable channel, which is a legitimate answered outcome.
        if (response.get("status") != "partial"
                and self._plain_total_dynamic_failure(
                    result_json.get("dynamic_analysis_records"))):
            records = result_json.get("dynamic_analysis_records") or []
            response["status"] = "partial"
            response["confidence"] = "none"
            response.setdefault("warnings", []).append(
                f"Dynamic analysis failed for all {len(records)} "
                f"target(s): no requested output passed verification "
                f"and nothing was salvageable. Decomposition-level "
                f"outputs and the descriptive analysis (if any) are "
                f"unaffected.")

        # #518: degraded-harmonization warnings land AFTER the status blocks
        # above (the partial branch ASSIGNS response["warnings"], which would
        # drop an earlier append).
        for _sw in (response.get("script_reuse") or {}).get(
                "scope_warnings", []):
            response.setdefault("warnings", []).append(
                f"DEGRADED HARMONIZATION: {_sw}")

        # Stage novel hot-annealing successes (method-family abandoned and a
        # later attempt approved) for review-gated skill distillation —
        # brings hyperspectral into the same T=2 flywheel as curve/image.
        # Failure-isolated; SCILINK_T2_AUTODISTILL=0 disables.
        # Bank every approved working script as episodic memory (script bank,
        # #346) — deterministic, no LLM, failure-isolated. Runs BEFORE T=2
        # staging so a hot win is nominated by promoting its bank record.
        banked = self._maybe_bank_scripts(
            response.get("dynamic_analysis_records") or [], skill_state,
            data_path, system_info,
        )
        if banked:
            response["banked_scripts"] = banked

        staged = self._maybe_stage_t2_solutions(
            response.get("dynamic_analysis_records") or [], skill_state
        )
        if staged:
            response["staged_solutions"] = staged

        # Persist the numeric results to <output_dir>/analysis_results.json
        # so the shared feature-table writer (feature_table.py, generic
        # extracted_features adapter) can emit features.csv — the file the
        # orchestrator's run_task collects into `feature_tables` and the
        # meta fusion's numerics bundle reads. Without this, hyperspectral
        # branches can never feed the computed reconciliation.
        self._write_results_file(response)

        self._log_action(
            action="analyze",
            input_ctx={"data": data_path},
            result=response,
            rationale="Hyperspectral analysis completed."
        )

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ ANALYSIS COMPLETE")
        self.logger.info(f"   Output: {self.output_dir}")
        self.logger.info(f"{'='*80}\n")
        
        return response

    _LIGHT_SYNTHESIS_SKIP = ("RunSelfReflectionController", "ApplyReflectionUpdatesController")

    def _synthesis_controllers(self) -> list:
        """The synthesis pipeline — without the critic/editor pair for a
        series replay child (``_light_synthesis``): the series-level
        synthesis interprets the series; a replay's own narrative only needs
        the draft interpretation and the report."""
        if getattr(self, "_light_synthesis", False):
            return [c for c in self.synthesis_pipeline
                    if c.__class__.__name__ not in self._LIGHT_SYNTHESIS_SKIP]
        return list(self.synthesis_pipeline)

    def _write_results_file(self, response: dict) -> str | None:
        """Persist a compact ``analysis_results.json`` with a FLAT
        ``extracted_features`` dict, feature-table ready.

        The dynamic-analysis features live in ``response`` as a LIST of
        per-map records ({name, units, description, stats:{min,max,mean}})
        — the shape the synthesis prompts consume — but the shared feature
        -table adapter (``feature_table._extracted_feature_rows``) needs a
        dict of scalars. Flatten each committed map's stats into
        ``<Map_Name>_<stat>[_<units>]`` columns. Judged honest-null
        determinations are recorded as ``<feature>_not_measurable = 1`` so
        an absence survives into the table as data, not as a missing row.
        Failure-isolated: never breaks the analysis.
        """
        try:
            # Shared with the series driver so a series row and a standalone
            # run of the same cube carry identical column names.
            feats = _series.flatten_feature_records(
                response.get("extracted_features"))
            if not feats:
                return None
            payload = {
                "agent_type": "hyperspectral",
                "status": response.get("status"),
                "extracted_features": feats,
            }
            path = self.output_dir / "analysis_results.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            self.logger.info(
                f"   📄 Numeric features persisted for downstream fusion: "
                f"{path.name} ({len(feats)} scalars)")
            return str(path)
        except Exception as e:  # noqa: BLE001 - table emit must never break analyze
            self.logger.warning(f"Could not write analysis_results.json: {e}")
            return None

    # =========================================================================
    # SERIES ANALYSIS (anchor + locked replay) — mirrors the image/curve agents
    # =========================================================================

    _UNIT_SETTINGS_EXCLUDE = frozenset(
        {"output_dir", "run_preprocessing", "feedback_depths", "enable_human_feedback"})

    def _make_unit_agent(self, unit_dir: Path, role: str,
                         human_feedback: bool) -> "HyperspectralAnalysisAgent":
        """A per-dataset child agent writing into ``unit_dir``.

        The single-cube pipeline is built around one output directory (the
        decomposition, dynamic-analysis records, report and preprocessing
        artifacts all land there), so each dataset of a series gets its own
        agent instance instead of re-pointing the parent's controllers. The
        child shares credentials, model, unmixing settings and budgets; the
        parent owns the once-per-series decisions (skill choice, banking).
        """
        child = HyperspectralAnalysisAgent(**self._unit_agent_kwargs(unit_dir, human_feedback))
        child._series_role = role
        child._skill_autoselect = False
        child._light_synthesis = (role == "replay")
        return child

    def _unit_agent_kwargs(self, unit_dir: Path, human_feedback: bool) -> dict:
        """Plain-data constructor kwargs for a per-dataset child agent (also
        shipped to replay worker processes, so nothing here may be live)."""
        settings = {k: v for k, v in self.spectral_settings.items()
                    if k not in self._UNIT_SETTINGS_EXCLUDE}
        settings["enable_human_feedback"] = human_feedback
        return dict(
            api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.base_url,
            output_dir=str(unit_dir),
            spectral_unmixing_settings=settings,
            run_preprocessing=self.spectral_settings.get("run_preprocessing", True),
            enable_human_feedback=human_feedback,
            executor_timeout=self.executor_timeout,
            max_verification_iterations=self.max_verification_iterations,
        )

    @staticmethod
    def _unit_system_info(base_si: dict, idx: int, n: int, data_path: str,
                          series_metadata: dict | None) -> dict:
        """Per-dataset metadata: the shared metadata plus this dataset's
        sidecar fields (when the orchestrator collected them) and its position
        on the series axis, so every prompt knows which point it is looking at."""
        si = {k: v for k, v in (base_si or {}).items() if k != "per_file_metadata"}
        pfm = (base_si or {}).get("per_file_metadata")
        if isinstance(pfm, dict):
            name = os.path.basename(str(data_path))
            own = pfm.get(name) or pfm.get(Path(name).stem) or pfm.get(str(data_path))
            if isinstance(own, dict):
                for k, v in own.items():
                    si.setdefault(k, v)
        ctx = {"index": idx, "n_datasets": n}
        if isinstance(series_metadata, dict):
            ctx["variable"] = series_metadata.get("variable")
            ctx["unit"] = series_metadata.get("unit")
            vals = series_metadata.get("values")
            if isinstance(vals, list) and idx < len(vals):
                ctx["value"] = vals[idx]
            elif isinstance(vals, dict):
                ctx["value"] = vals.get(os.path.basename(str(data_path)))
        si["series_context"] = ctx
        return si

    def _locked_targets(self, anchor_dir: Path, row: dict | None = None) -> list:
        """The anchor's approved targets, as ``locked_targets`` entries. With
        the anchor's series row, every map it reported beyond the required
        outputs is attached as ``extra_outputs`` so a fresh-code run for the
        same targets is asked to report them under the same names too."""
        recs = self._load_prior_dynamic_records([anchor_dir])
        targets = [{"target": r.get("target"),
                    "required_outputs": list(r.get("required_outputs") or [])}
                   for r in recs]
        if row and targets:
            required = {n for t in targets for n in t["required_outputs"]}
            extra = []
            for m in row.get("feature_records") or []:
                nm = (m or {}).get("name") if isinstance(m, dict) else None
                if nm and nm not in required and nm not in extra:
                    extra.append(nm)
            if extra:
                targets[0]["extra_outputs"] = extra
        return targets

    def _analyze_series(
        self,
        data_paths: list,
        system_info,
        series_metadata: dict | None,
        structure_image_path: str | None,
        structure_system_info: dict | None,
        objective: str | None,
        hints: str | None,
        skill,
        skill_hint,
        custom_skills: dict | None,
        prior_knowledge: list | None,
        literature_file: str | None,
        max_verification_iterations: int | None,
        reference_scripts: list | None,
        prior_analysis_paths: list | None,
        reuse_locked_script: bool,
        max_series_refits: int | None,
        outlier_sigma: float,
        series_workers: int | None = None,
    ) -> Dict[str, Any]:
        """Anchor + locked-replay driver over a list of datacubes.

        Stage order (see ``controllers/hyperspectral_series.py``): anchor
        (full analysis, recipe locked from its approved scripts) → verbatim
        replay on every later dataset → flagging → budgeted refit of failed
        datasets → series JSON → trend codegen → series synthesis → report.
        The first dataset that yields an approved recipe becomes the anchor;
        datasets before it were analysed independently.
        """
        n = len(data_paths)
        series_dir = self.output_dir
        series_dir.mkdir(parents=True, exist_ok=True)

        base_si = self._handle_system_info(system_info) or {}
        base_si, series_metadata = self._extract_series_metadata(base_si, series_metadata)
        series_metadata = self._normalize_series_values(series_metadata, data_paths)
        if not series_metadata:
            self.logger.warning(
                "Series without series_metadata: datasets are ordered as given "
                "and trends are reported against the index.")
            series_metadata = {"variable": "index", "values": list(range(n)), "unit": ""}
        outlier_sigma = float(outlier_sigma) if outlier_sigma else 2.0
        workers = _series.resolve_series_workers(series_workers)

        self._init_state(data_path=data_paths[0], num_datasets=n,
                         metadata=base_si, series_metadata=series_metadata)
        self._log_action("analysis_started",
                         {"data_paths": [str(p) for p in data_paths], "series": True},
                         {"status": "running"})

        # Skills: resolved ONCE for the series (explicit request, else the
        # metadata-driven auto-selector), then handed to every child.
        skill_state = (self._load_skills_to_state(skill, domain="hyperspectral")
                       if skill else {"skill_name": None, "skill_sections": None,
                                      "skills_loaded": []})
        unit_skill = skill
        if not skill_state.get("skills_loaded"):
            selected = self._auto_select_skills(base_si, hint=skill_hint,
                                                custom_skills=custom_skills)
            if selected:
                cs = custom_skills or {}
                unit_skill = [cs.get(s, s) for s in selected]
                skill_state = self._load_skills_to_state(unit_skill, domain="hyperspectral")
        if literature_file and not Path(literature_file).is_file():
            self.logger.warning(f"literature_file not found: {literature_file}")
            literature_file = None

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 HYPERSPECTRAL SERIES ANALYSIS ({n} datasets)")
        self.logger.info(f"   Variable: {series_metadata.get('variable')} "
                         f"[{series_metadata.get('unit')}]  sigma={outlier_sigma}  "
                         f"refit budget={max_series_refits}  replay workers={workers}")
        self.logger.info(f"{'='*80}\n")

        # ---- Scout + regime plan (before any dataset is analysed) ----------
        # Mean spectrum of every cube → full-series SVD change detection and
        # an overlay of representative datasets → one planning call that
        # decides whether one locked script can serve the whole series or
        # the series splits into regimes, each with its own locked script.
        self.logger.info("\n🔭 --- Scouting series ---")
        from ...skills.hyperspectral.eels.eels import create_axis as _create_axis

        def _axis_fn(n_channels: int):
            return _create_axis(n_channels, base_si, axis_index=2)

        scout = _series.scout_series(
            data_paths, self._load_hyperspectral_data, _axis_fn,
            series_metadata, self.logger)
        plan_state = {
            "num_images": n, "series_metadata": series_metadata,
            "system_info": base_si, "analysis_objective": objective,
            "analysis_hints": hints, "prior_knowledge": prior_knowledge or [],
            **skill_state,
        }
        series_plan = _series.plan_series_regimes(
            self.model, self.generation_config, self.safety_settings,
            self._parse_llm_response, plan_state, scout, self.logger)
        if series_plan:
            regimes = series_plan["regimes"]
        else:
            regimes = [{"name": "series", "dataset_indices": list(range(n)),
                        "description": None}]
        regime_of = {i: r for r in regimes for i in r["dataset_indices"]}
        self.logger.info(f"   Regimes: {len(regimes)} → "
                         + "; ".join(f"{r['name']}: {r['dataset_indices']}" for r in regimes))

        common = dict(
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info,
            objective=objective, hints=hints, skill=unit_skill,
            custom_skills=custom_skills, prior_knowledge=prior_knowledge,
            literature_file=literature_file,
            max_verification_iterations=max_verification_iterations,
            reference_scripts=reference_scripts,
        )

        def _join_hints(*parts):
            parts = [p for p in parts if p]
            return "\n".join(parts) if parts else None

        # ---- Per-regime anchor + verbatim replay -----------------------------
        # The first dataset to yield an approved recipe (any regime) is the
        # SCHEMA SOURCE: its targets and output names are locked for the whole
        # series. Every later fresh-code run — the anchor of another regime,
        # or an adaptive refit — runs in locked-targets mode (same targets and
        # names, new code), and its row is completed against the schema
        # source's columns, so feature names align across regimes and refits.
        rows: list = [None] * n
        locks: dict = {}            # regime name -> lock
        schema: dict | None = None  # {"anchor_index", "targets", "columns"}
        deferred: list = []         # replay specs handed to the pool (parallel mode)
        pool = None                 # created on the first queued replay
        for idx, path in enumerate(data_paths):
            regime = regime_of[idx]
            rname = regime["name"]
            unit_dir = series_dir / _series.UNIT_DIR_FMT.format(idx=idx)
            unit_si = self._unit_system_info(base_si, idx, n, path, series_metadata)
            regime_hint = (
                f"Series regime '{rname}' ({len(regimes)} regimes planned): "
                f"{regime.get('description') or 'no description'}"
                if series_plan else None)
            lock = locks.get(rname)
            if lock is None and schema is None:
                role = "anchor"
                child = self._make_unit_agent(unit_dir, role, self.enable_human_feedback)
                extra = ({"prior_analysis_paths": prior_analysis_paths,
                          "reuse_locked_script": True}
                         if reuse_locked_script and prior_analysis_paths else {})
                self.logger.info(
                    f"\n──── [{idx + 1}/{n}] ANCHOR ({rname}): {Path(path).name} "
                    f"(full analysis{' via locked replay of a prior run' if extra else ''}) ────")
            elif lock is None:
                role = "regime_anchor"
                child = self._make_unit_agent(unit_dir, role, self.enable_human_feedback)
                extra = {"locked_targets": schema["targets"]}
                self.logger.info(
                    f"\n──── [{idx + 1}/{n}] REGIME ANCHOR ({rname}): {Path(path).name} "
                    f"(locked targets from dataset {schema['anchor_index']}, fresh code) ────")
            else:
                role = "replay"
                extra = {"prior_analysis_paths": [lock["anchor_output_dir"]],
                         "reuse_locked_script": True,
                         "replay_reference": lock.get("reference_maps") or {}}
                if workers > 1:
                    # Replays are independent of each other and of the
                    # anchors still to run: hand this one to the pool now so
                    # it overlaps with the rest of the loop.
                    spec = {
                        "index": idx, "data_path": path, "unit_dir": str(unit_dir),
                        "regime": rname, "lock": lock, "log_to_file": True,
                        "sandbox_approved": True,
                        "agent_kwargs": self._unit_agent_kwargs(unit_dir, False),
                        "analyze_kwargs": {"system_info": unit_si, **common,
                                           "hints": _join_hints(hints, regime_hint), **extra},
                    }
                    if pool is None:
                        pool = _series.ReplayPool(workers, self.logger)
                    deferred.append(spec)
                    pool.submit(spec)
                    self.logger.info(
                        f"   [{idx + 1}/{n}] REPLAY ({rname}) queued: {Path(path).name}")
                    continue
                child = self._make_unit_agent(unit_dir, role, human_feedback=False)
                self.logger.info(
                    f"\n──── [{idx + 1}/{n}] REPLAY ({rname}): {Path(path).name} "
                    f"(locked recipe from dataset {lock['anchor_index']}) ────")
            try:
                res = child.analyze(path, system_info=unit_si,
                                    **{**common, "hints": _join_hints(hints, regime_hint)},
                                    **extra)
            except Exception as e:  # noqa: BLE001 - one dataset must not kill the series
                self.logger.exception(f"Dataset {idx} raised: {e}")
                res = {"status": "error", "error": {"error": type(e).__name__,
                                                    "details": str(e)}}
            row = _series.build_series_row(idx, path, res, role, unit_dir)
            row["regime"] = rname
            if lock is None:
                if row["success"]:
                    targets = self._locked_targets(unit_dir, row)
                    if targets:
                        lock = {"anchor_index": idx,
                                "anchor_output_dir": str(unit_dir),
                                "targets": targets, "n_scripts": len(targets),
                                "regime": rname,
                                # the anchor's per-map stats: the deterministic
                                # replay gate's plausible-range reference
                                "reference_maps": {
                                    m["name"]: m["stats"] for m in row.get("feature_records") or []
                                    if isinstance(m, dict) and isinstance(m.get("stats"), dict)},
                                "source": ("prior_run" if extra.get("reuse_locked_script")
                                           else "anchor")}
                        locks[rname] = lock
                        self.logger.info(
                            f"🔒 Recipe locked for regime '{rname}' on dataset {idx}: "
                            f"{len(targets)} approved script(s).")
                        if schema is None:
                            schema = {"anchor_index": idx, "targets": targets,
                                      "columns": list(row["extracted_features"].keys())}
                            self.logger.info(
                                f"🧩 Schema source: dataset {idx}, "
                                f"{len(schema['columns'])} feature column(s).")
                    else:
                        row["role"] = "independent"
                        self.logger.warning(
                            f"Dataset {idx} succeeded without an approved "
                            f"dynamic-analysis script — nothing to lock for "
                            f"regime '{rname}'; its next dataset is analysed in full.")
                else:
                    row["role"] = "independent"
                    self.logger.warning(
                        f"Anchor candidate {idx} for regime '{rname}' failed "
                        f"({row['error']}); the regime's next dataset becomes "
                        f"the anchor candidate.")
            # Every successful row but the schema source is completed against
            # the schema: fresh-code rows get their drifted names aliased, and
            # a salvaged replay gets its missing columns REPORTED as a gap.
            if (schema is not None and row["success"]
                    and idx != schema["anchor_index"]):
                row = _series.complete_locked_schema(row, schema["columns"], self.logger)
            rows[idx] = row
            self.logger.info(
                f"   {'✅' if row['success'] else '❌'} dataset {idx} [{rname}]: "
                f"{row['status']}, {len(row['extracted_features'])} feature(s)")

        # ---- Deferred replays on the worker pool ----------------------------
        if deferred:
            self.logger.info(f"⚡ Collecting {len(deferred)} pooled replay(s)…")
            results = pool.collect()
            for sp in deferred:
                idx = sp["index"]
                res = results.get(idx) or {"status": "error",
                                           "error": {"error": "no result", "details": "worker returned nothing"}}
                row = _series.build_series_row(idx, sp["data_path"], res, "replay", sp["unit_dir"])
                row["regime"] = sp["regime"]
                if schema is not None and row["success"]:
                    row = _series.complete_locked_schema(row, schema["columns"], self.logger)
                rows[idx] = row
                self.logger.info(
                    f"   {'✅' if row['success'] else '❌'} dataset {idx} [{sp['regime']}]: "
                    f"{row['status']}, {len(row['extracted_features'])} feature(s)")

        # Independent successes that ran BEFORE the schema source existed
        # (anchor candidates that committed features without an approved
        # script) are completed against the schema retroactively.
        if schema is not None:
            for r in rows:
                if (r.get("success") and r.get("role") == "independent"
                        and "locked_schema_gap" not in r):
                    rows[r["index"]] = _series.complete_locked_schema(
                        r, schema["columns"], self.logger)

        # ---- Flagging + budgeted refit of failed datasets --------------------
        _ctrl = series_metadata.get("values") if isinstance(series_metadata.get("values"), list) else None
        # Outliers are judged on the locked PRIMARY outputs only (diagnostics of
        # a refit's different method are not comparable to the locked script's).
        _primary = [nm for t in (schema or {}).get("targets", [])
                    for nm in (t.get("required_outputs") or [])]
        _groups = self._outlier_groups(series_plan, scout)
        flagged = _series.detect_outliers(rows, outlier_sigma, control_values=_ctrl,
                                          feature_prefixes=_primary, groups=_groups)
        refit_summary: list = []
        refit_skipped: list = []
        if locks:
            cands, refit_skipped = _series.select_refit_candidates(flagged, max_series_refits)
            for cand in cands:
                idx = cand["index"]
                # An anchor candidate that failed before its regime locked
                # already had a full independent run; repeating THAT is the
                # same experiment. Once a schema source exists, a refit is a
                # different attempt — locked targets proven on this series
                # with fresh code — so it gets one.
                if rows[idx].get("role") == "independent" and schema is None:
                    continue
                path = data_paths[idx]
                rname = rows[idx].get("regime")
                unit_dir = series_dir / f"{_series.UNIT_DIR_FMT.format(idx=idx)}_refit"
                child = self._make_unit_agent(unit_dir, "refit", human_feedback=False)
                lock = locks.get(rname) or next(iter(locks.values()))
                why = (f"failed on it ({rows[idx].get('error')})" if not rows[idx].get("success")
                       else "was not verified on it (only a salvaged attempt committed features)")
                refit_hints = (
                    f"Series context: this is dataset {idx + 1}/{n} of a series "
                    f"(regime '{rname}'); the series' locked analysis "
                    f"({[t.get('target') for t in lock['targets']]}) {why}. Write a "
                    f"NEW method for the same targets that suits this dataset — the "
                    f"earlier fit windows, thresholds or model did not.")
                self.logger.info(f"\n──── REFIT dataset {idx} [{rname}]: {Path(path).name} "
                                 f"(locked targets, fresh code) ────")
                unit_si = self._unit_system_info(base_si, idx, n, path, series_metadata)
                extra = {"locked_targets": schema["targets"]} if schema else {}
                try:
                    res = child.analyze(path, system_info=unit_si,
                                        **{**common, "hints": _join_hints(hints, refit_hints)},
                                        **extra)
                except Exception as e:  # noqa: BLE001
                    self.logger.exception(f"Refit of dataset {idx} raised: {e}")
                    res = {"status": "error", "error": {"error": type(e).__name__,
                                                        "details": str(e)}}
                new_row = _series.build_series_row(idx, path, res, "refit", unit_dir)
                new_row["regime"] = rname
                if new_row["success"] and schema:
                    new_row = _series.complete_locked_schema(new_row, schema["columns"], self.logger)
                entry = {"index": idx, "name": new_row["name"], "regime": rname,
                         "original_error": rows[idx].get("error") or "unverified (salvaged attempt)",
                         "new_status": new_row["status"],
                         "n_features": len(new_row["extracted_features"]),
                         "improved": bool(new_row["success"]),
                         "locked_schema_gap": new_row.get("locked_schema_gap", [])}
                refit_summary.append(entry)
                if new_row["success"]:
                    new_row["adaptively_refitted"] = True
                    new_row["original_output_directory"] = rows[idx]["output_directory"]
                    rows[idx] = new_row
            if cands:
                flagged = _series.detect_outliers(rows, outlier_sigma, control_values=_ctrl,
                                                  feature_prefixes=_primary, groups=_groups)
        for r in rows:
            r.pop("flagged", None); r.pop("flag_reason", None); r.pop("flag_details", None)
        for f in flagged:
            r = rows[f["index"]]
            r["flagged"] = True
            r["flag_reason"] = f["reason"]
            r["flag_details"] = f.get("details")
        locked = None
        if schema:
            locked = {"anchor_index": schema["anchor_index"],
                      "targets": schema["targets"],
                      "columns": schema["columns"],
                      "regimes": {k: {kk: vv for kk, vv in v.items()} for k, v in locks.items()}}

        state: Dict[str, Any] = {
            "series_results": rows,
            "series_metadata": series_metadata,
            "system_info": base_si,
            "flagged_images": flagged,        # key read by the shared trend codegen
            "series_analysis_plan": series_plan,
            "scout": {k: v for k, v in scout.items() if k != "overlay_png"},
            "refit_summary": refit_summary,
            "refit_skipped_by_budget": refit_skipped,
            "locked_config": locked,
            "outlier_sigma": outlier_sigma,
            "series_workers": workers,
            "num_images": n,
            "is_single_image": False,
            "analysis_objective": objective,
            "analysis_hints": hints,
            "prior_knowledge": prior_knowledge or [],
            "literature_context": (read_text_utf8(Path(literature_file))
                                   if literature_file else None),
            **skill_state,
        }
        state["series_results_path"] = _series.write_series_results(series_dir, state)
        state["flagged_path"] = _series.write_flagged_file(series_dir, state)

        # ---- Trend codegen over the series JSON ------------------------------
        n_ok = sum(1 for r in rows if r["success"])
        if n_ok >= 2:
            try:
                trend = _series.HyperspectralSeriesTrendController(
                    self.model, self.logger, self.generation_config,
                    self.safety_settings, self._parse_llm_response,
                    executor=ScriptExecutor(timeout=self.executor_timeout),
                    output_dir=str(series_dir), max_corrections=3)
                state = trend.execute(state)
            except Exception as e:  # noqa: BLE001
                self.logger.exception(f"Trend analysis failed: {e}")
                state["trend_analysis_results"] = {"success": False, "error": str(e)}
        else:
            state["trend_analysis_results"] = {
                "success": True, "skipped": True,
                "reason": f"only {n_ok} successful dataset(s) — no trend"}

        # ---- Series synthesis + report ---------------------------------------
        synth = {}
        if n_ok:
            synth = _series.synthesize_series(
                self.model, self.generation_config, self.safety_settings,
                self._parse_llm_response, state, self.logger)
        state["synthesis_result"] = synth
        state["report_path"] = _series.generate_series_report(series_dir, state)

        response = self._compile_series_results(state, literature_file)
        try:
            (series_dir / "analysis_results.json").write_text(
                json.dumps(_series._serializable(response), indent=2), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Could not write analysis_results.json: {e}")
        self._log_action("analyze_series",
                         {"data_paths": [str(p) for p in data_paths]},
                         {k: response.get(k) for k in ("status", "summary")},
                         rationale="Hyperspectral series analysis completed.")
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ SERIES ANALYSIS COMPLETE — {n_ok}/{n} datasets, "
                         f"{len(flagged)} flagged, {len(refit_summary)} refit(s)")
        self.logger.info(f"   Output: {series_dir}")
        self.logger.info(f"{'='*80}\n")
        return response

    @staticmethod
    def _outlier_groups(series_plan: dict | None, scout: dict | None) -> dict | None:
        """Regime groups for outlier scoring — only when the planned regimes
        interleave along the series axis (change detection found the axis
        not coherent); contiguous regimes are scored as one series."""
        regimes = (series_plan or {}).get("regimes") or []
        if len(regimes) < 2:
            return None
        coherent = (((scout or {}).get("reduction") or {}).get("axis_coherence") or {}
                    ).get("coherent", True)
        if coherent:
            return None
        return {r["name"]: list(r.get("dataset_indices") or []) for r in regimes}

    @staticmethod
    def _scout_summary(scout: dict | None) -> dict | None:
        """Compact, serializable view of the scout stage for the result."""
        if not isinstance(scout, dict):
            return None
        red = scout.get("reduction") or {}
        out = {"n_loaded": scout.get("n_loaded"),
               "scouted_indices": [s.get("index") for s in scout.get("scout_data") or []],
               "change_detection": None}
        if red:
            out["change_detection"] = {
                "change_point": red.get("change_point"),
                "change_sharpness": red.get("change_sharpness"),
                "control_variable": (red.get("control_variable") or {}).get("source"),
                "axis_coherent": (red.get("axis_coherence") or {}).get("coherent"),
                "variance_explained": red.get("variance_explained"),
                "flags": red.get("flags"),
            }
        return out

    def _compile_series_results(self, state: dict, literature_file) -> Dict[str, Any]:
        rows = state["series_results"]
        n = len(rows)
        n_ok = sum(1 for r in rows if r["success"])
        synth = state.get("synthesis_result") or {}
        flagged = state.get("flagged_images") or []
        locked = state.get("locked_config")
        trend = state.get("trend_analysis_results") or {}
        warnings: list = []
        if n_ok == 0:
            status = "error"
        elif (n_ok < n or synth.get("error")
              or any(r.get("status") == "partial" for r in rows)):
            status = "partial"
        else:
            status = "success"
        if locked is None:
            warnings.append("No dataset produced an approved dynamic-analysis "
                            "script, so no recipe was locked: datasets were "
                            "analysed independently and their features may "
                            "not be method-comparable.")
        for r in rows:
            if r.get("reuse_validity") and not r["reuse_validity"].get("verbatim"):
                warnings.append(f"dataset {r['index']}: locked script was repaired "
                                f"or scope-degraded — not byte-comparable to the anchor.")
            if r.get("locked_schema_gap"):
                warnings.append(f"dataset {r['index']}: {len(r['locked_schema_gap'])} locked "
                                f"feature column(s) not reported "
                                f"({', '.join(r['locked_schema_gap'][:4])}"
                                f"{'…' if len(r['locked_schema_gap']) > 4 else ''}).")
        if synth.get("error"):
            warnings.append(f"Series synthesis failed: {synth['error']}")
        if trend and not trend.get("success"):
            warnings.append(f"Trend analysis failed: {trend.get('error') or 'script failed'}")

        anchor = next((r for r in rows if r.get("role") == "anchor"), None)
        detailed = synth.get("detailed_analysis")
        if not detailed:
            detailed = ((anchor or {}).get("detailed_analysis") or "") + (
                "\n\n[Series synthesis unavailable; anchor interpretation shown.]"
                if anchor else "Series synthesis unavailable.")

        individual = []
        for r in rows:
            individual.append({
                k: r.get(k) for k in (
                    "index", "name", "data_path", "success", "status", "role",
                    "confidence", "output_directory", "error", "flagged",
                    "flag_reason", "flag_details", "adaptively_refitted",
                    "reuse_validity", "quality_metrics", "warnings", "regime",
                    "locked_schema_gap", "schema_aliases", "verified")
                if k in r})
            individual[-1]["n_features"] = len(r.get("extracted_features") or {})

        response: Dict[str, Any] = {
            "status": status,
            "output_directory": str(self.output_dir),
            "summary": {
                "total_datasets": n,
                "successful_analyses": n_ok,
                "flagged_count": len(flagged),
                "refitted_count": sum(1 for r in rows if r.get("adaptively_refitted")),
                "unverified_count": sum(1 for r in rows if r.get("success") and r.get("verified") is False),
                "series_workers": state.get("series_workers"),
                "anchor_index": (locked or {}).get("anchor_index"),
                "locked_targets": [t.get("target") for t in (locked or {}).get("targets", [])],
                "regimes": len((state.get("series_analysis_plan") or {}).get("regimes") or []) or 1,
                "regime_anchors": {k: v.get("anchor_index")
                                   for k, v in ((locked or {}).get("regimes") or {}).items()},
                "is_single_dataset": False,
            },
            "series_analysis_plan": state.get("series_analysis_plan"),
            "scout": self._scout_summary(state.get("scout")),
            "detailed_analysis": detailed,
            "scientific_claims": self._validate_scientific_claims(
                synth.get("scientific_claims") or []),
            "individual_results": individual,
            "flagged_datasets": flagged,
            "flagged_analysis": synth.get("flagged_analysis"),
            "refit_summary": state.get("refit_summary") or [],
            "refit_skipped_by_budget": state.get("refit_skipped_by_budget") or [],
            "trend_analysis": trend,
            "feature_trends": synth.get("feature_trends") or {},
            "series_features": _series.series_feature_matrix(rows),
            "series_metadata": state.get("series_metadata"),
            "locked_config": locked,
            "caveats": synth.get("caveats") or [],
            "series_results_path": state.get("series_results_path"),
            "report_path": state.get("report_path"),
        }
        if status == "error":
            response["error"] = {
                "error": f"All {n} dataset analyses failed",
                "details": "; ".join(f"{r['index']}: {r.get('error')}" for r in rows)[:1500],
            }
        if warnings:
            response["warnings"] = warnings
        if literature_file:
            response["literature_files"] = {"provided_file": str(literature_file)}
        return response

    def _maybe_stage_t2_solutions(self, records: list, skill_state: dict) -> list:
        """Stage novel hot-retry successes for later, review-gated distillation.

        Hyperspectral mirror of the curve/image T=2 hooks, reading the
        per-target ``dynamic_analysis_records`` (HS-1). Gate (all must hold):
        the task succeeded (``approved`` — fraction + required-outputs); its
        winning attempt came AFTER the retry ladder reached the hot stage
        ("abandon the method family"), i.e. the model discarded its first
        method family and found one that works — the novelty signal.

        Returns staged solution ids. Fully failure-isolated;
        ``SCILINK_T2_AUTODISTILL=0`` disables staging.
        """
        if getattr(self, "_series_role", None) == "replay":
            return []   # verbatim replays are the anchor's solutions, not novel
        from scilink.skills.loader import memory_enabled
        if not memory_enabled():
            return []
        flag = os.environ.get("SCILINK_T2_AUTODISTILL", "").strip().lower()
        if flag in ("0", "false", "off", "no"):
            return []

        staged: list = []
        try:
            from .instruct import T2_TECHNIQUE_LABEL_INSTRUCTIONS
            from scilink.skills._shared import _staging

            active_skills = [
                s.get("name") for s in (skill_state or {}).get("skills_loaded", [])
                if isinstance(s, dict)
            ]

            def _llm_call(prompt: str) -> str:
                response = self.model.generate_content(
                    contents=[prompt],
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                return response.text if hasattr(response, "text") else str(response)

            for rec in records:
                qh = rec.get("quality_history") or {}
                if not qh.get("approved") or not rec.get("script"):
                    continue
                levels = [
                    it.get("annealing_level", 0)
                    for it in qh.get("verification_iterations", [])
                ]
                reached_hot = (max(levels) if levels else 0) >= 2
                if not reached_hot:
                    continue

                target = rec.get("target") or "dynamic analysis"
                deviation = (
                    "Initial method family failed repeatedly; the retry ladder "
                    "escalated to 'abandon the method family' and a structurally "
                    "different estimator succeeded."
                )
                technique = _staging.assign_technique_label(
                    "hyperspectral", target, deviation, _llm_call,
                    T2_TECHNIQUE_LABEL_INSTRUCTIONS,
                )
                record = {
                    "analysis_target": target,
                    "required_outputs": rec.get("required_outputs"),
                    "deviation_from_plan": deviation,
                    "final_passed_fraction": qh.get("final_passed_fraction"),
                    "active_skills": active_skills,
                    "working_script": rec["script"],
                    "session": self.output_dir.name,
                }
                # Unified path (see curve agent): promote the fresh bank
                # record; legacy direct staging when the bank is disabled.
                from scilink.skills._shared import _script_bank
                sid = None
                bank_rec = (_script_bank.find_by_script("hyperspectral", rec["script"])
                            if _script_bank.bank_enabled() else None)
                if bank_rec is not None:
                    out = _script_bank.promote_to_staging(
                        "hyperspectral", bank_rec["id"], technique=technique,
                        provenance="t2_hot_win",
                        extra={k: v for k, v in record.items()
                               if k not in ("working_script", "session")},
                    )
                    if out.get("status") == "success":
                        sid = out["staged_id"]
                if sid is None:
                    sid = _staging.stage_solution("hyperspectral", technique, record)
                staged.append(sid)
                self.logger.info(
                    f"   🧠 Staged hot-retry hyperspectral solution "
                    f"[{technique}] id={sid}"
                )
        except Exception as e:  # noqa: BLE001 - staging never affects results
            self.logger.warning(f"T=2 staging skipped: {e}")
            return staged

        if staged:
            self.logger.info(
                f"   🧠 {len(staged)} solution(s) staged; review with "
                f"`scilink memory staged`."
            )
        return staged

    def _load_prior_dynamic_records(self, prior_analysis_paths: list) -> list:
        """Collect the APPROVED dynamic-analysis records of prior runs.

        Each path may be a result directory, a directory of result
        directories, or a direct path to ``dynamic_analysis_records.json``.
        Returns the records with ``task_success`` and a saved ``script`` —
        the replayable pipeline of the donor run.
        """
        records: list = []
        for p in (prior_analysis_paths or []):
            base = Path(p)
            candidates: list = []
            if base.is_file() and base.name == "dynamic_analysis_records.json":
                candidates = [base]
            elif base.is_dir():
                direct = base / "dynamic_analysis_records.json"
                candidates = ([direct] if direct.is_file() else
                              sorted(base.glob("*/dynamic_analysis_records.json"))
                              + sorted(base.glob("results/*/dynamic_analysis_records.json")))
            for cand in candidates:
                try:
                    recs = json.loads(read_text_utf8(cand))
                except Exception as e:  # noqa: BLE001 - skip unreadable, keep looking
                    self.logger.warning(f"Could not read {cand}: {e}")
                    continue
                for rec in (recs if isinstance(recs, list) else []):
                    if (isinstance(rec, dict) and rec.get("script")
                            and rec.get("task_success")):
                        records.append(rec)
        return records

    def _handle_system_info(self, system_info):
        """Base handling plus the Tier-1 deterministic alias normalizer.

        Hyperspectral analysis hard-requires a resolvable signal axis, and
        externally-authored metadata often spells the range as
        ``spectral_axis: {min, max}`` rather than the canonical
        ``energy_range: {start, end}``. Folding aliases here — the single
        entry point for metadata on this agent — means the axis
        precondition and every downstream reader see the canonical shape.
        A no-op for already-conformant dicts.
        """
        si = super()._handle_system_info(system_info)
        if si:
            from .metadata_converter import normalize_metadata_dict
            si, was_modified = normalize_metadata_dict(si)
            if was_modified:
                self.logger.info(
                    "Metadata aliases folded to the canonical schema "
                    "(Tier-1 deterministic normalizer)."
                )
        return si

    def _maybe_bank_scripts(self, records: list, skill_state: dict,
                            data_path: str, system_info) -> list:
        """Bank every approved working script in the script bank (#346).

        Hyperspectral mirror of the curve/image hooks, reading the per-target
        ``dynamic_analysis_records`` — episodic complement to T=2 staging,
        no hot gate, no LLM. The cube is reloaded lazily (only when there is
        something to bank) and fingerprinted once for all records. Fully
        failure-isolated; gated by ``SCILINK_SCRIPT_BANK`` /
        persistent-memory setting.
        """
        if getattr(self, "_series_role", None) == "replay":
            # The anchor banked this script once for the series; banking each
            # verbatim replay would inflate its proven-N per dataset.
            return []
        from scilink.skills._shared import _script_bank
        if not _script_bank.bank_enabled():
            return []

        banked: list = []
        try:
            bankable = [
                rec for rec in records
                if rec.get("script") and (rec.get("quality_history") or {}).get("approved")
            ]
            if not bankable:
                return []

            from .metadata_converter import resolve_axis_spec, signal_axis_values

            si = self._handle_system_info(system_info)
            active_skills = [
                s.get("name") for s in (skill_state or {}).get("skills_loaded", [])
                if isinstance(s, dict)
            ]

            fingerprint = None
            try:
                cube = self._load_hyperspectral_data(data_path)
                axis_2 = resolve_axis_spec(si)["axis_2"]
                e = cube.shape[-1]
                axis = signal_axis_values(axis_2, e, logger=self.logger)
                if axis is not None:
                    axis_units = axis_2.get("units", "arbitrary units")
                elif "start" in axis_2 and "end" in axis_2:
                    axis = np.linspace(axis_2["start"], axis_2["end"], e)
                    axis_units = axis_2.get("units", "arbitrary units")
                else:
                    axis, axis_units = np.arange(e), "channels"
                fingerprint = _script_bank.hyperspectral_fingerprint(
                    cube, axis, axis_units
                )
            except Exception as e:  # noqa: BLE001 - fingerprint is best-effort
                self.logger.warning(f"Bank fingerprint skipped: {e}")

            context = _script_bank.measurement_context(si)
            seen_hashes = set()
            for rec in bankable:
                h = _script_bank.script_hash(rec["script"])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                qh = rec.get("quality_history") or {}
                frac = qh.get("final_passed_fraction")
                res = _script_bank.add_record("hyperspectral", {
                    "technique_signals": {
                        "active_skills": active_skills,
                        "analysis_target": rec.get("target"),
                    },
                    "measurement_context": context,
                    "data_fingerprint": fingerprint,
                    "outcome": {
                        "analysis_target": rec.get("target"),
                        "required_outputs": rec.get("required_outputs"),
                        "metric": ({"name": "passed_fraction", "value": round(float(frac), 4)}
                                   if isinstance(frac, (int, float)) else None),
                    },
                    "provenance": {"session": self.output_dir.name,
                                   "data_file": os.path.basename(str(data_path))},
                    "working_script": rec["script"],
                })
                if res.get("id"):
                    banked.append(res["id"])
                    self.logger.info(
                        f"   🏦 Banked script [{res['action']}] id={res['id']}"
                    )
        except Exception as e:  # noqa: BLE001 - banking never affects results
            self.logger.warning(f"Script banking skipped: {e}")
        return banked

    def _auto_select_skills(self, system_info, hint=None, custom_skills=None) -> list:
        """Pick relevant hyperspectral skill(s) from the metadata.

        Uses the shared technique-aware selector. ``hint`` is the orchestrator's
        non-binding suggestion (the agent has final authority). ``custom_skills``
        ({name: path}) folds user-registered skills into the catalog. Returns a
        possibly-empty, ranked list of skill names; never raises.
        """
        if not getattr(self, "_skill_autoselect", True):
            return []   # series child: the parent resolved the skills once
        from ...skills._shared._skill_selector import select_relevant_skills

        context_parts = []
        if isinstance(system_info, dict) and system_info:
            context_parts.append(f"Metadata: {str(system_info)[:1500]}")
        elif isinstance(system_info, str) and system_info.strip():
            context_parts.append(f"Metadata: {system_info.strip()[:1500]}")
        if not context_parts:
            return []

        return select_relevant_skills(
            model=self.model,
            parse_fn=self._parse_llm_response,
            domain="hyperspectral",
            context_parts=context_parts,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            hint=hint,
            custom_skills=custom_skills,
            logger=self.logger,
        )

    # =========================================================================
    # BACKWARD COMPATIBLE METHODS
    # =========================================================================
    
    def analyze_for_claims(
        self,
        data_path: str,
        metadata_path: Dict[str, Any] | str | None = None,
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        objective: str | None = None,
        hints: str | None = None,
        skill: str | None = None,
        auxiliary_data: str | None = None,
        auxiliary_label: str | None = None
    ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data to generate scientific claims.

        BACKWARD COMPATIBLE: Delegates to analyze().
        """
        result = self.analyze(
            data_path,
            system_info=metadata_path,
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info,
            hints=hints,
            objective=objective,
            skill=skill,
            auxiliary_data=auxiliary_data,
            auxiliary_label=auxiliary_label
        )
        
        if result.get("status") == "success":
            return {
                "detailed_analysis": result.get("detailed_analysis", ""),
                "scientific_claims": result.get("scientific_claims", [])
            }
        else:
            return result.get("error", result)
    
    def analyze_hyperspectral_data(
        self,
        data_path: str,
        metadata_path: str,
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        objective: str | None = None,
        hints: str | None = None,
        skill: str | None = None,
        auxiliary_data: str | None = None,
        auxiliary_label: str | None = None
    ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data for materials characterization.

        BACKWARD COMPATIBLE: Delegates to analyze().
        """
        return self.analyze_for_claims(
            data_path=data_path,
            metadata_path=metadata_path,
            structure_image_path=structure_image_path,
            structure_system_info=structure_system_info,
            hints=hints,
            objective=objective,
            skill=skill,
            auxiliary_data=auxiliary_data,
            auxiliary_label=auxiliary_label
        )

    # =========================================================================
    # INSTRUCTION PROMPTS
    # =========================================================================
    
    def _get_claims_instruction_prompt(self) -> str:
        return SPECTROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _regenerate_report_with_feedback(
        self,
        final_result: Dict[str, Any],
        system_info: Any,
        data_path: str
    ) -> None:
        """Regenerate HTML report after feedback modifications."""
        stored_images = self._get_stored_analysis_images()
        
        report_state = {
            "result_json": final_result,
            "system_info": self._handle_system_info(system_info),
            "analysis_images": stored_images,
            "image_path": data_path
        }
        
        from .controllers.hyperspectral_controllers import GenerateHTMLReportController
        report_gen = GenerateHTMLReportController(self.logger, self.spectral_settings)
        report_gen.execute(report_state)
        
        self.logger.info("✅ Refined HTML report generated.")

    def _load_hyperspectral_data(self, data_path: str) -> np.ndarray:
        """Load hyperspectral data from .npy or .h5/.hdf5/.nxs (NeXus)."""
        try:
            lower = data_path.lower()
            if lower.endswith('.npy'):
                data = np.load(data_path)
            elif lower.endswith(('.h5', '.hdf5', '.nxs')):
                from ...utils.hdf5_utils import load_hdf5_signal
                data = load_hdf5_signal(data_path)
            else:
                raise ValueError(
                    f"Expected .npy or .h5/.hdf5/.nxs file, got: {data_path}"
                )
            self.logger.info(f"Loaded hyperspectral data: shape {data.shape}")
            
            if data.ndim == 2:
                self.logger.warning("2D data detected, reshaping to (1, 1, n_channels)")
                data = data.reshape(1, 1, -1)
            elif data.ndim != 3:
                raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    def _load_auxiliary_items(self, auxiliary_data, auxiliary_label) -> dict:
        """Load one or several auxiliary datasets into the multi-aux state.

        Accepts ``str | list[str]`` for both ``auxiliary_data`` and
        ``auxiliary_label`` (parallel lists). Each file is loaded via
        ``_load_auxiliary_data``; labels are made unique (auto-named ``aux_<i>``
        when missing). Labels become the operand keys downstream. (#226)
        """
        paths = list(auxiliary_data) if isinstance(auxiliary_data, (list, tuple)) else [auxiliary_data]
        labels = list(auxiliary_label) if isinstance(auxiliary_label, (list, tuple)) else [auxiliary_label]

        items = []
        used = set()
        for i, p in enumerate(paths):
            lbl = labels[i] if i < len(labels) else None
            one = self._load_auxiliary_data(p, lbl)
            name = one.get("auxiliary_label") or f"aux_{i}"
            base, k = name, 1
            while name in used:
                name = f"{base}_{k}"; k += 1
            used.add(name)
            items.append({
                "label": name,
                "array": one.get("auxiliary_array"),
                "axis": one.get("auxiliary_axis"),
                "plot_bytes": one.get("auxiliary_plot_bytes"),
                "summary": one.get("auxiliary_summary"),
                "mime_type": one.get("auxiliary_mime_type"),
            })

        return {"auxiliary_items": items}

    def _load_auxiliary_data(
        self, auxiliary_data: str, auxiliary_label: str | None
    ) -> dict:
        """
        Load auxiliary data and return state fields for pipeline injection.

        Supports 1D curve files (.csv, .txt, .dat, .tsv) and images
        (.png, .jpg, .tif, etc.). For .npy files, inspects array shape
        to distinguish curves from images.

        Returns dict with auxiliary_plot_bytes, auxiliary_label,
        auxiliary_summary, and auxiliary_mime_type (all None on failure).
        """
        result = {
            "auxiliary_plot_bytes": None,
            "auxiliary_label": auxiliary_label or Path(auxiliary_data).stem,
            "auxiliary_summary": None,
            "auxiliary_mime_type": None,
            # Raw numbers retained so the per-pixel code-gen may use the
            # auxiliary as an OPTIONAL numerical operand (e.g. reference
            # division), not only as a rendered picture for the LLM. None when
            # the array cannot be loaded. ``auxiliary_axis`` holds the x-axis of
            # a 1D curve (for alignment); None for images.
            "auxiliary_array": None,
            "auxiliary_axis": None,
        }

        if not os.path.exists(auxiliary_data):
            self.logger.warning(f"Auxiliary data file not found: {auxiliary_data}")
            return result

        ext = Path(auxiliary_data).suffix.lower()
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        curve_extensions = {'.csv', '.txt', '.dat', '.tsv'}

        try:
            is_curve = False
            is_image = False
            is_cube = False

            if ext == '.npy':
                arr = np.load(auxiliary_data, mmap_mode='r')
                if arr.ndim == 1:
                    is_curve = True
                elif arr.ndim == 2 and min(arr.shape) <= 2:
                    is_curve = True
                elif arr.ndim >= 3 and arr.shape[-1] > 4:
                    # A spectral datacube companion (e.g. an I0/flat-field
                    # baseline), NOT an RGB(A) image. Keep it as a raw numerical
                    # operand so the per-pixel code-gen can normalize against it
                    # (e.g. transmission = data / baseline). A trailing axis of
                    # 3 or 4 is treated as an image below.
                    is_cube = True
                else:
                    is_image = True
            elif ext in curve_extensions:
                is_curve = True
            elif ext in image_extensions:
                is_image = True
            else:
                self.logger.warning(
                    f"Unrecognized auxiliary file extension: {ext}"
                )
                return result

            if is_curve:
                if ext == '.npy':
                    curve = np.load(auxiliary_data)
                    if curve.ndim == 1:
                        curve = np.column_stack(
                            [np.arange(len(curve)), curve]
                        )
                    elif curve.shape[0] == 2:
                        curve = curve.T
                else:
                    curve = load_curve_data(auxiliary_data)
                    if curve.ndim == 2 and curve.shape[0] == 2:
                        curve = curve.T

                if curve.ndim == 2 and curve.shape[1] == 2:
                    x, y = curve[:, 0], curve[:, 1]
                elif curve.ndim == 2 and curve.shape[0] == 2:
                    x, y = curve[0], curve[1]
                else:
                    x = np.arange(curve.shape[-1])
                    y = curve.flatten()

                result["auxiliary_summary"] = (
                    f"1D curve with {len(x)} points. "
                    f"X range: [{float(np.nanmin(x)):.4g}, {float(np.nanmax(x)):.4g}]. "
                    f"Y range: [{float(np.nanmin(y)):.4g}, {float(np.nanmax(y)):.4g}]."
                )
                result["auxiliary_array"] = np.asarray(y, dtype=float)
                result["auxiliary_axis"] = np.asarray(x, dtype=float)

                plot_info = {"title": result["auxiliary_label"]}
                plot_data = np.column_stack([x, y])
                result["auxiliary_plot_bytes"] = plot_curve_to_bytes(
                    plot_data, plot_info
                )
                result["auxiliary_mime_type"] = "image/png"

            elif is_image:
                img = load_image(auxiliary_data)
                # The numerical operand carries the RAW values — same rule the
                # cube branch below states: the uint8 display rendering
                # quantizes to 256 levels and discards the physical scale
                # (an STM topography in meters became a constant-zero operand,
                # silently NaN-ing every correlation computed against it).
                try:
                    raw = np.asarray(load_image(auxiliary_data, raw=True))
                    if np.issubdtype(raw.dtype, np.number):
                        raw = raw.astype(float, copy=False)
                    result["auxiliary_array"] = raw
                    result["auxiliary_summary"] = (
                        f"Image with shape {raw.shape} (dtype: {raw.dtype}); "
                        f"raw value range [{float(np.nanmin(raw)):.4g}, "
                        f"{float(np.nanmax(raw)):.4g}]."
                    )
                except Exception:  # noqa: BLE001 - fall back to display array
                    result["auxiliary_array"] = img
                    result["auxiliary_summary"] = (
                        f"Image with shape {img.shape} "
                        f"(dtype: {img.dtype})."
                    )
                if img.ndim == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    result["auxiliary_plot_bytes"] = (
                        convert_numpy_to_jpeg_bytes(img_gray)
                    )
                else:
                    result["auxiliary_plot_bytes"] = (
                        convert_numpy_to_jpeg_bytes(img)
                    )
                result["auxiliary_mime_type"] = "image/jpeg"

            elif is_cube:
                # Same-grid companion datacube (e.g. an I0 / flat-field
                # baseline). Loaded as a raw float array so it can serve as a
                # per-pixel numerical operand for the code-gen (the operand
                # alignment gate keeps it only if its shape matches the primary
                # cube). We deliberately do NOT route it through ``load_image``,
                # which would cast the counts to uint8 and destroy their scale.
                cube = np.asarray(np.load(auxiliary_data), dtype=float)
                result["auxiliary_array"] = cube
                result["auxiliary_summary"] = (
                    f"Companion datacube with shape {cube.shape} "
                    f"(dtype: {cube.dtype}). Value range: "
                    f"[{float(np.nanmin(cube)):.4g}, {float(np.nanmax(cube)):.4g}]. "
                    "Intended as a same-grid per-pixel operand (e.g. an I0 / "
                    "flat-field baseline to divide the primary by)."
                )
                # Show the spatial-mean spectrum so the LLM can see the
                # baseline's spectral shape (a cube has no single 2D rendering).
                try:
                    mean_spec = np.asarray(cube).reshape(-1, cube.shape[-1]).mean(0)
                    plot_data = np.column_stack(
                        [np.arange(mean_spec.size), mean_spec]
                    )
                    result["auxiliary_plot_bytes"] = plot_curve_to_bytes(
                        plot_data,
                        {"title": f"{result['auxiliary_label']} "
                                  "(spatial-mean spectrum)"},
                    )
                    result["auxiliary_mime_type"] = "image/png"
                except Exception as _plot_err:
                    self.logger.warning(
                        f"Could not render mean spectrum for cube auxiliary: "
                        f"{_plot_err}"
                    )

        except Exception as e:
            self.logger.warning(f"Failed to load auxiliary data: {e}")

        return result

    def _run_analysis_pipeline(
        self,
        data_path: str,
        system_info: Dict[str, Any] | str | None,
        instruction_prompt: str,
        structure_image_path: str | None = None,
        structure_system_info: Dict[str, Any] | None = None,
        objective: str | None = None,
        hints: str | None = None,
        reference_scripts: list | None = None,
        skill_state: Dict[str, Any] | None = None,
        prior_knowledge: list | None = None,
        auxiliary_state: Dict[str, Any] | None = None,
        literature_context: str | None = None,
        max_verification_iterations: int | None = None,
        reuse_records: list | None = None,
        locked_targets: list | None = None,
        replay_reference: dict | None = None,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        """
        Main execution engine using Queue-Based Branching architecture.
        """
        if skill_state is None:
            skill_state = {"skill_name": None, "skill_sections": None, "skills_loaded": []}
        if auxiliary_state is None:
            auxiliary_state = _empty_auxiliary_state()

        try:
            self.logger.info(f"--- Starting analysis pipeline for {data_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            # Load data
            original_hspy_data = self._load_hyperspectral_data(data_path)

            # Fail-fast precondition: the physical axis-2 range is a hard
            # requirement (interpretation prep and every summary plot call
            # create_axis, which raises without it — deliberate, fcb77007).
            # Enforce it here, BEFORE preprocessing and the component test
            # loop burn minutes of compute on a run that cannot finish.
            from ...skills.hyperspectral.eels.eels import create_axis
            try:
                create_axis(original_hspy_data.shape[-1], system_info, axis_index=2)
            except ValueError as e:
                self.logger.error(f"Axis precondition failed: {e}")
                return None, {
                    "error": "Missing physical axis range for the spectral axis",
                    "details": (
                        f"{e} Supply the physical range of the third (signal/"
                        f"sweep) axis in the metadata before re-running — no "
                        f"analysis was performed."
                    ),
                }

            # Handle structure image
            structure_image_blob = None
            if structure_image_path and os.path.exists(structure_image_path):
                try:
                    img = load_image(structure_image_path)
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    structure_image_blob = {
                        "mime_type": "image/jpeg",
                        "data": convert_numpy_to_jpeg_bytes(img)
                    }
                except Exception as e:
                    self.logger.warning(f"Could not load structure image: {e}")
            
            # Single-pass iteration. Recursive refinement was removed (custom_code
            # refinement now executes in-place inside RunDynamicAnalysisController).
            self.logger.info("\n=== Global_Analysis ===\n")
            iteration_state = {
                "data_path": data_path,
                "hspy_data": original_hspy_data,
                "original_hspy_data": original_hspy_data,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "settings": self.spectral_settings.copy(),
                "iteration_title": "Global_Analysis",
                # IterativeFeedbackController gates on current_depth in
                # feedback_depths (default [0]) — must be set so the
                # human-feedback prompt fires when enable_human_feedback
                # is on. Phase C collapsed the recursion loop but kept
                # this depth-based gate intact.
                "current_depth": 0,
                "structure_image_path": structure_image_path,
                "structure_system_info": self._handle_system_info(structure_system_info),
                "structure_image_blob": structure_image_blob,
                "analysis_hints": hints,
                "analysis_objective": objective,
                "reference_scripts": _load_reference_scripts(reference_scripts),
                "prior_knowledge": prior_knowledge or [],
                "literature_context": literature_context,
                # Locked-script replay: SelectRefinementTarget builds the plan
                # deterministically from these records, and decomposition is
                # skipped — the replayed per-pixel scripts use the raw cube.
                **({"reuse_records": reuse_records,
                    "skip_decomposition": True,
                    "replay_reference": replay_reference or {}} if reuse_records else {}),
                # Locked targets, fresh code: plan fixed, decomposition skipped,
                # codegen ladder intact (see SelectRefinementTargetController).
                **({"locked_targets": locked_targets,
                    "skip_decomposition": True}
                   if locked_targets and not reuse_records else {}),
                "analysis_images": [],
                "error_dict": None,
                # Per-run retry-budget override (#271) — read by
                # RunDynamicAnalysisController.execute(); None = default.
                "max_verification_iterations": max_verification_iterations,
                **skill_state,
                **auxiliary_state,
            }

            for controller in self.iteration_pipeline:
                iteration_state = controller.execute(iteration_state)
                if iteration_state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at {controller.__class__.__name__}")
                    break

            all_completed_results = []
            if not iteration_state.get("error_dict"):
                all_completed_results.append({
                    "iteration_title": iteration_state.get("iteration_title"),
                    "iteration_analysis_text": iteration_state.get("result_json", {}).get("detailed_analysis", ""),
                    "analysis_images": iteration_state.get("analysis_images", []),
                    "refinement_decision": iteration_state.get("refinement_decision", {}),
                    "custom_analysis_metadata_list": iteration_state.get("custom_analysis_metadata_list"),
                })

            # Run synthesis
            self.logger.info(f"\n=== Synthesizing {len(all_completed_results)} analyses ===\n")
            
            synthesis_state = {
                "all_iteration_results": all_completed_results,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "analysis_hints": hints,
                "analysis_objective": objective,
                "reference_scripts": _load_reference_scripts(reference_scripts),
                "prior_knowledge": prior_knowledge or [],
                "literature_context": literature_context,
                "result_json": None,
                # Carry the iteration pipeline's failure forward so the caller
                # sees the ORIGINAL error (e.g. the decomposition exception),
                # not the derivative "No iteration results found for
                # synthesis" it used to be masked by (#381). The synthesis
                # controllers all no-op on a pre-set error_dict.
                "error_dict": iteration_state.get("error_dict"),
                **skill_state,
                **auxiliary_state
            }

            for controller in self._synthesis_controllers():
                synthesis_state = controller.execute(synthesis_state)
                if synthesis_state.get("error_dict"):
                    self.logger.error(f"Synthesis failed at {controller.__class__.__name__}")
                    break

            self.logger.info("--- Analysis pipeline finished ---")
            # Surface any degradation the dynamic-analysis stage recorded (salvage
            # / withheld / approximate) so the top-level status is not a clean
            # 'success'. The notes live on iteration_state (set by the salvage
            # judge); attach them to result_json for analyze() to read.
            _rj = synthesis_state.get("result_json")
            _notes = iteration_state.get("degradation_notes", [])
            if _rj is not None and _notes:
                _rj["degradation_notes"] = _notes
            if _rj is not None:
                # Surface dynamic-analysis features + per-target verification
                # records (HS-1 / #323 prereq) — both additive.
                _feat = iteration_state.get("custom_analysis_metadata_list")
                if _feat:
                    _rj["extracted_features"] = _feat
                _recs = iteration_state.get("dynamic_analysis_records")
                if _recs:
                    _rj["dynamic_analysis_records"] = _recs
            return _rj, synthesis_state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            return None, {"error": "File not found", "details": f"Path: {data_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"Unexpected error: {e}")
            return None, {"error": "Unexpected error", "details": str(e)}