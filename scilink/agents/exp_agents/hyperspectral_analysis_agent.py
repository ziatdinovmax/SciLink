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
from ...executors import require_sandbox_approval
from ...skills.loader import load_skill

from ._deprecation import normalize_params


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
                - str: Path to .npy hyperspectral data file
                - List[str]: Batch processing (not supported)
                - np.ndarray: Direct array (not supported)
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
        
        # Batch processing not supported
        if data_paths is not None:
            return {
                "status": "error",
                "error": {
                    "error": "Batch processing not supported",
                    "details": "HyperspectralAnalysisAgent processes one file at a time. Pass a single file path."
                },
                "output_directory": str(self.output_dir)
            }
        
        # Direct array not supported
        if data_array is not None:
            return {
                "status": "error",
                "error": {
                    "error": "Direct array input not supported",
                    "details": "Save array to .npy file and pass the file path."
                },
                "output_directory": str(self.output_dir)
            }
        
        # Initialize and Run Pipeline
        self._init_state(data_path=data_path, metadata=system_info)

        # Load skill(s) if provided. Accepts a single name/path or a list
        # — see PR 3 multi-skill support.
        skill_state = self._load_skills_to_state(skill, domain="hyperspectral")

        # Auto-select skill(s) when none were explicitly provided, mirroring
        # the image/curve agents. Conservative, technique-aware (issue #251);
        # may pick zero, one, or several skills from the metadata.
        if not skill_state.get("skills_loaded"):
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
                literature_context = lit_p.read_text()
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
            objective=objective,
            skill_state=skill_state,
            prior_knowledge=prior_knowledge or [],
            auxiliary_state=auxiliary_state,
            literature_context=literature_context,
            max_verification_iterations=effective_max_verification,
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
            feats: dict = {}
            for m in response.get("extracted_features") or []:
                if not isinstance(m, dict):
                    continue
                nm = m.get("not_measurable")
                if isinstance(nm, dict) and nm.get("feature"):
                    base = str(nm["feature"]).strip().replace(" ", "_")[:60]
                    feats[f"{base}_not_measurable"] = 1
                    continue
                stats = m.get("stats")
                if not isinstance(stats, dict):
                    continue
                base = str(m.get("name") or "feature").strip().replace(" ", "_")
                units = str(m.get("units") or "").strip()
                suffix = ("_" + units.replace(" ", "").replace("/", "per")
                          if units and units.lower() not in ("a.u.", "au", "")
                          else "")
                for k, v in stats.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        feats[f"{base}_{k}{suffix}"] = float(v)
            if not feats:
                return None
            payload = {
                "agent_type": "hyperspectral",
                "status": response.get("status"),
                "extracted_features": feats,
            }
            path = self.output_dir / "analysis_results.json"
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
            self.logger.info(
                f"   📄 Numeric features persisted for downstream fusion: "
                f"{path.name} ({len(feats)} scalars)")
            return str(path)
        except Exception as e:  # noqa: BLE001 - table emit must never break analyze
            self.logger.warning(f"Could not write analysis_results.json: {e}")
            return None

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

            from .metadata_converter import resolve_axis_spec

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
                if "start" in axis_2 and "end" in axis_2:
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
                result["auxiliary_summary"] = (
                    f"Image with shape {img.shape} "
                    f"(dtype: {img.dtype})."
                )
                result["auxiliary_array"] = img
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
        skill_state: Dict[str, Any] | None = None,
        prior_knowledge: list | None = None,
        auxiliary_state: Dict[str, Any] | None = None,
        literature_context: str | None = None,
        max_verification_iterations: int | None = None,
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
                "prior_knowledge": prior_knowledge or [],
                "literature_context": literature_context,
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
                "prior_knowledge": prior_knowledge or [],
                "literature_context": literature_context,
                "result_json": None,
                "error_dict": None,
                **skill_state,
                **auxiliary_state
            }

            for controller in self.synthesis_pipeline:
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