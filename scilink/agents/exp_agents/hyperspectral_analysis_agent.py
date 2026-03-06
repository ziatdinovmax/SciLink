"""
Hyperspectral Analysis Agent
"""


import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any
from collections import deque

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
from ...tools.image_processor import load_image, convert_numpy_to_jpeg_bytes
from ...tools.curve_fitting_tools import load_curve_data, plot_curve_to_bytes
from ...executors import require_sandbox_approval
from ...skills.loader import load_skill

from ._deprecation import normalize_params


class HyperspectralAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Hyperspectral Analysis Agent with recursive "survey-then-focus" loop.
    
    This agent analyzes hyperspectral/spectroscopic data using NMF decomposition
    and LLM-guided interpretation.
    
    Features:
        - Automatic component number selection via elbow method
        - Recursive refinement with spatial/spectral zooming
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
    
    MAX_REFINEMENT_ITERATIONS = 2

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.1-pro-preview",
        base_url: str | None = None,
        output_dir: str = "hyperspectral_analysis_output",
        # Deprecated params
        google_api_key: str | None = None,
        local_model: str | None = None,
        # Agent specific params
        spectral_unmixing_settings: dict | None = None,
        run_preprocessing: bool = True,
        enable_human_feedback: bool = True
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
        
        # Sub-agent initialization
        preprocess_dir = self.output_dir / "preprocessing"
        self.preprocessor = HyperspectralPreprocessingAgent(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=str(preprocess_dir)
        )

        # Pipeline initialization
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
            preprocessor=self.preprocessor
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
        prior_knowledge: list | None = None,
        auxiliary_data: str | None = None,
        auxiliary_label: str | None = None,
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

        # Load skill if provided
        skill_state = {"skill_name": None, "skill_sections": None}
        if skill:
            try:
                parsed = load_skill(skill, domain="hyperspectral")
                skill_state = {"skill_name": parsed["name"], "skill_sections": parsed}
                self.logger.info(f"   Skill loaded: {parsed['name']}")
            except FileNotFoundError:
                self.logger.warning(
                    f"   Skill '{skill}' not found — proceeding without domain skill"
                )

        # Load auxiliary data if provided
        auxiliary_state = {
            "auxiliary_plot_bytes": None,
            "auxiliary_label": None,
            "auxiliary_summary": None,
            "auxiliary_mime_type": None,
        }
        if auxiliary_data:
            auxiliary_state = self._load_auxiliary_data(
                auxiliary_data, auxiliary_label
            )
            if auxiliary_state.get("auxiliary_plot_bytes"):
                self.logger.info(
                    f"   Auxiliary data loaded: {auxiliary_state['auxiliary_label']}"
                )

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
            auxiliary_state=auxiliary_state
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
        """Load hyperspectral data from numpy array."""
        try:
            if not data_path.endswith('.npy'):
                raise ValueError(f"Expected .npy file, got: {data_path}")
            
            data = np.load(data_path)
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

            if ext == '.npy':
                arr = np.load(auxiliary_data)
                if arr.ndim == 1:
                    is_curve = True
                elif arr.ndim == 2 and min(arr.shape) <= 2:
                    is_curve = True
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
        auxiliary_state: Dict[str, Any] | None = None
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        """
        Main execution engine using Queue-Based Branching architecture.
        """
        if skill_state is None:
            skill_state = {"skill_name": None, "skill_sections": None}
        if auxiliary_state is None:
            auxiliary_state = {
                "auxiliary_plot_bytes": None,
                "auxiliary_label": None,
                "auxiliary_summary": None,
                "auxiliary_mime_type": None,
            }

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
            
            # Initialize task queue
            initial_task = {
                "data": original_hspy_data,
                "system_info": system_info,
                "title": "Global_Analysis",
                "parent_reasoning": None,
                "depth": 0
            }
            
            task_queue = deque([initial_task])
            all_completed_results = []
            task_counter = 0
            
            # Process tasks
            while task_queue:
                current_task = task_queue.popleft()
                task_counter += 1
                
                if current_task["depth"] > self.MAX_REFINEMENT_ITERATIONS:
                    self.logger.info(f"Skipping '{current_task['title']}': max depth reached")
                    continue

                self.logger.info(f"\n=== TASK {task_counter}: {current_task['title']} (Depth {current_task['depth']}) ===\n")

                iteration_state = {
                    "data_path": data_path,
                    "hspy_data": current_task["data"],
                    "original_hspy_data": original_hspy_data,
                    "system_info": current_task["system_info"],
                    "instruction_prompt": instruction_prompt,
                    "settings": self.spectral_settings.copy(),
                    "iteration_title": current_task["title"],
                    "parent_refinement_reasoning": current_task["parent_reasoning"],
                    "current_depth": current_task["depth"],
                    "structure_image_path": structure_image_path,
                    "structure_system_info": self._handle_system_info(structure_system_info),
                    "structure_image_blob": structure_image_blob,
                    "analysis_hints": hints,
                    "analysis_objective": objective,
                    "prior_knowledge": prior_knowledge or [],
                    "analysis_images": [],
                    "error_dict": None,
                    **skill_state,
                    **auxiliary_state
                }

                if current_task["depth"] > 0:
                    iteration_state["settings"]['run_preprocessing'] = False

                # Run iteration pipeline
                for controller in self.iteration_pipeline:
                    iteration_state = controller.execute(iteration_state)
                    if iteration_state.get("error_dict"):
                        self.logger.error(f"Pipeline failed at {controller.__class__.__name__}")
                        break
                
                if iteration_state.get("error_dict"):
                    continue

                # Store results
                result_summary = {
                    "iteration_title": iteration_state.get("iteration_title"),
                    "iteration_analysis_text": iteration_state.get("result_json", {}).get("detailed_analysis", ""),
                    "analysis_images": iteration_state.get("analysis_images", []),
                    "refinement_decision": iteration_state.get("refinement_decision", {}),
                    "depth": current_task["depth"],
                    "custom_analysis_metadata": iteration_state.get("custom_analysis_metadata")
                }
                all_completed_results.append(result_summary)

                # Process new tasks
                for t in iteration_state.get("new_tasks", []):
                    task_queue.append({
                        "data": t["data"],
                        "system_info": t["system_info"],
                        "title": t["title"],
                        "parent_reasoning": t["parent_reasoning"],
                        "depth": t["source_depth"]
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
            return synthesis_state.get("result_json"), synthesis_state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            return None, {"error": "File not found", "details": f"Path: {data_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"Unexpected error: {e}")
            return None, {"error": "Unexpected error", "details": str(e)}