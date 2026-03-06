"""
SAM Microscopy Analysis Agent
"""

import warnings

from scilink.executors import require_sandbox_approval
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Optional, Union

from .base_agent import BaseAnalysisAgent, AnalysisInput
from .human_feedback import SimpleFeedbackMixin
from .instruct import (
    SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .pipelines.sam_microscopy_pipelines import create_unified_sam_pipeline
from ._deprecation import normalize_params

from ...tools.image_processor import (
    load_image,
    preprocess_image,
    convert_numpy_to_jpeg_bytes
)
from ...tools.sam import (
    run_sam_analysis,
    calculate_sam_statistics,
)

from ...executors import require_sandbox_approval, ScriptExecutor


class SAMMicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Segment Anything Model Analysis Agent.
    
    ALL analysis follows the batch processing pattern:
    - Single image analysis = batch of 1
    - Multiple images = standard batch processing
    - Numpy array stack = batch processing
    
    Example:
        agent = SAMMicroscopyAnalysisAgent(api_key="...")
        
        # Single image
        result = agent.analyze("sample.tif")
        
        # Multiple images  
        result = agent.analyze(["img1.tif", "img2.tif"])
        
        # Numpy stack
        result = agent.analyze(my_stack)
        
        # Get measurement recommendations
        recommendations = agent.recommend_measurements(analysis_result=result)
    
    Settings (sam_settings dict):
        SAM Analysis Parameters:
            checkpoint_path (str): Path to SAM model checkpoint. Default: None (auto-download)
            model_type (str): SAM model variant. Default: 'vit_h'
            device (str): Compute device. Default: 'auto'
            use_clahe (bool): Enable CLAHE contrast enhancement. Default: False
            sam_parameters (str): Detection sensitivity preset. Default: 'default'
                Options: 'default', 'sensitive', 'ultra-permissive'
            min_area (int): Minimum particle area in pixels. Default: 500
            max_area (int): Maximum particle area in pixels. Default: 50000
            use_pruning (bool): Enable duplicate removal. Default: True
            pruning_iou_threshold (float): IoU threshold for duplicate removal. Default: 0.5
        
        Pipeline Control:
            SAM_ENABLED (bool): Enable SAM analysis. Default: True
            enable_human_feedback (bool): Enable interactive refinement. Default: False
            max_feedback_iterations (int): Max human feedback rounds. Default: 3
            max_auto_refinement_iterations (int): Max automated LLM refinement rounds. Default: 2
            max_script_corrections (int): Max LLM script fix attempts. Default: 3
            script_timeout (int): Script execution timeout in seconds. Default: 300
            save_visualizations (bool): Save overlay PNGs. Default: True
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3.1-pro-preview",
        base_url: str | None = None,
        google_api_key: str | None = None,
        local_model: str = None,
        sam_settings: dict | None = None,
        enable_human_feedback: bool = False,
        output_dir: str = "sam_output"
    ):
        
        if not require_sandbox_approval(
            context="SAMMicroscopyAnalysisAgent (SAM microscopy analysis)"
        ):
            raise RuntimeError(
                "SAMMicroscopyAnalysisAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )
    
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, 
            source="SAMMicroscopyAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "sam_microscopy"
        
        # Resolve output directory
        self.output_dir = self.output_dir.resolve()
        
        # Define sub-directories
        self.viz_dir = self.output_dir / "sam_visualizations"
        self.data_dir = self.output_dir / "sam_analysis"
        self.scripts_dir = self.output_dir / "scripts"
        
        # Create directories
        for d in [self.viz_dir, self.data_dir, self.scripts_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Prepare settings — consolidate ALL defaults here as single source of truth.
        # Controllers read from settings but should not need to define their own
        # fallback defaults for any key listed here.
        self.settings = sam_settings if sam_settings else {}
        
        # --- Feature flags ---
        self.settings.setdefault('SAM_ENABLED', True)
        self.settings.setdefault('enable_human_feedback', enable_human_feedback)
        self.settings.setdefault('save_visualizations', True)
        
        # --- SAM analysis parameters ---
        self.settings.setdefault('checkpoint_path', None)
        self.settings.setdefault('model_type', 'vit_h')
        self.settings.setdefault('device', 'auto')
        self.settings.setdefault('use_clahe', False)
        self.settings.setdefault('sam_parameters', 'default')
        self.settings.setdefault('min_area', 500)
        self.settings.setdefault('max_area', 50000)
        self.settings.setdefault('use_pruning', True)
        self.settings.setdefault('pruning_iou_threshold', 0.5)
        
        # --- Pipeline iteration limits ---
        self.settings.setdefault('max_feedback_iterations', 3)
        self.settings.setdefault('max_auto_refinement_iterations', 4)
        self.settings.setdefault('max_script_corrections', 3)
        self.settings.setdefault('script_timeout', 300)
        
        # --- Directories (set explicitly, not defaulted) ---
        self.settings['visualization_dir'] = str(self.viz_dir)
        self.settings['output_dir'] = str(self.data_dir)

        self.executor = ScriptExecutor(timeout=self.settings.get('script_timeout', 300))
        
        if self.settings.get('SAM_ENABLED', True):
            self.logger.info(f"SAMMicroscopyAnalysisAgent initialized. Outputs: {self.output_dir}")
        else:
            self.logger.warning("SAMMicroscopyAnalysisAgent initialized, but 'SAM_ENABLED' is False.")
    
    def _get_initial_state_fields(self) -> dict:
        """Return initial state fields for the agent."""
        return {
            "current_image": None,
            "pipeline_type": "sam_unified",
            "particles_detected": 0,
            "batch_mode": False
        }
    
    def _initialize_sam_params(self) -> dict:
        """
        Get initial SAM parameters from settings.
        
        All keys here MUST have corresponding setdefault() calls in __init__,
        so .get() calls without a fallback default are safe.
        """
        return {
            "checkpoint_path": self.settings.get('checkpoint_path'),
            "model_type": self.settings.get('model_type'),
            "device": self.settings.get('device'),
            "use_clahe": self.settings.get('use_clahe'),
            "sam_parameters": self.settings.get('sam_parameters'),
            "min_area": self.settings.get('min_area'),
            "max_area": self.settings.get('max_area'),
            "use_pruning": self.settings.get('use_pruning'),
            "pruning_iou_threshold": self.settings.get('pruning_iou_threshold')
        }
    
    def analyze(
        self,
        data: AnalysisInput,
        system_info: Optional[Union[Dict[str, Any], str]] = None,
        # SAM-specific options
        series_metadata: Optional[dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Unified analysis method - handles single images and batches identically.
        
        Single image analysis is internally converted to a batch of 1.
        
        Args:
            data: Input data. Can be:
                - str: Single image path
                - List[str]: Multiple image paths
                - np.ndarray: 2D (single) or 3D (stack) array
            system_info: System/sample information. May include a ``"series"``
                key with series metadata; it will be extracted automatically.
            series_metadata: Optional metadata describing the experimental
                variable that changes across images in a series. Can also
                be provided inside ``system_info["series"]``. Expected
                structure::

                    {
                        "variable": "temperature",  # independent variable name
                        "values": [300, 350, 400],   # one value per image, in file order
                        "unit": "K"                  # unit for values
                    }

        Returns:
            Dict with status, detailed_analysis, scientific_claims,
            summary, output_directory, and SAM-specific fields
        
        Examples:
            # Single image
            result = agent.analyze("sample.tif")
            
            # Multiple images
            result = agent.analyze(["img1.tif", "img2.tif"])
            
            # Numpy stack
            result = agent.analyze(my_stack)
        """
        # Parse input
        data_path, data_paths, data_array, error = self._parse_data_input(data)
        
        if error:
            return {
                "status": "error",
                "error": error,
                "output_directory": str(self.output_dir)
            }
        
        # Normalize to internal variables
        image_path = data_path
        image_paths = data_paths
        image_stack = data_array
        
        # Convert single image to batch of 1
        if image_path is not None:
            image_paths = [image_path]
            self.logger.info(f"Single image mode: treating as batch of 1")
        
        # Determine input type and count
        if image_stack is not None:
            if image_stack.ndim == 2:
                # Single 2D image provided as array - convert to 3D
                image_stack = image_stack[np.newaxis, :, :]
                self.logger.info("Single 2D array provided, converted to shape (1, h, w)")
            if image_stack.ndim != 3:
                return {
                    "status": "error",
                    "error": {"error": "Invalid shape", "details": f"Array must be 2D or 3D, got {image_stack.ndim}D"},
                    "output_directory": str(self.output_dir)
                }
            num_images = image_stack.shape[0]
            input_type = "numpy_array"
        else:
            num_images = len(image_paths)
            input_type = "file_paths"
        
        is_single_image = (num_images == 1)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"🔬 SAM ANALYSIS - {num_images} image{'s' if num_images > 1 else ''}")
        self.logger.info(f"{'='*80}\n")
        
        # Load and preprocess first image for initial analysis
        if image_stack is not None:
            first_image = image_stack[0]
            first_image_name = "frame_0000"
        else:
            try:
                first_image = load_image(image_paths[0])
                first_image_name = Path(image_paths[0]).stem
            except Exception as e:
                return {
                    "status": "error",
                    "error": {"error": "Failed to load image", "details": str(e)},
                    "output_directory": str(self.output_dir)
                }
        
        preprocessed_img, _ = preprocess_image(first_image)
        image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img)
        
        nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(
            self._handle_system_info(system_info), first_image.shape
        )
        
        # Build initial state dict
        _sys_info = self._handle_system_info(system_info)
        _sys_info, series_metadata = self._extract_series_metadata(_sys_info, series_metadata)
        state = {
            # Input data
            "image_paths": image_paths,
            "image_stack": image_stack,
            "input_type": input_type,
            "num_images": num_images,
            "is_single_image": is_single_image,

            # System info
            "system_info": _sys_info,
            "series_metadata": series_metadata or {},
            
            # First image (preprocessed)
            "image_path": image_paths[0] if image_paths else first_image_name,
            "first_image_name": first_image_name,
            "preprocessed_image_array": preprocessed_img,
            "image_blob": {"mime_type": "image/jpeg", "data": image_bytes},
            
            # Spatial calibration
            "nm_per_pixel": nm_per_pixel,
            "fov_in_nm": fov_in_nm,
            
            # Settings and params
            "settings": self.settings,
            "enable_human_feedback": self.settings.get('enable_human_feedback', False),
            "current_params": self._initialize_sam_params(),
            
            # Results placeholders
            "sam_result": None,
            "summary_stats": None,
            "analysis_images": [
                {"label": "Primary Microscopy Image", "data": image_bytes}
            ],
            "error_dict": None,
        }
        
        # Run Initial SAM Analysis on First Image
        self.logger.info("📍 Running initial SAM analysis on first image...\n")
        
        try:
            sam_result = run_sam_analysis(preprocessed_img, params=state["current_params"])
            state["sam_result"] = sam_result
            
            summary_stats = calculate_sam_statistics(
                sam_result=sam_result,
                image_path=state["image_path"],
                preprocessed_image_shape=preprocessed_img.shape,
                nm_per_pixel=nm_per_pixel
            )
            state["summary_stats"] = summary_stats
            
            self.logger.info(f"   Initial detection: {sam_result['total_count']} particles\n")
            
        except Exception as e:
            self.logger.error(f"Initial SAM analysis failed: {e}")
            return {
                "status": "error",
                "error": {"error": "Initial SAM analysis failed", "details": str(e)},
                "output_directory": str(self.output_dir)
            }
        
        # Create and Execute Unified Pipeline
        pipeline = create_unified_sam_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
            executor=self.executor
        )
        
        # Execute pipeline steps
        for i, controller in enumerate(pipeline, 1):
            step_name = controller.__class__.__name__
            self.logger.info(f"\n📍 STEP {i}: {step_name}\n")
            
            try:
                state = controller.execute(state)
                
                # Check for errors
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {step_name}: {state['error_dict']}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Pipeline step {step_name} raised exception: {e}")
                state["error_dict"] = {"error": f"Pipeline step failed: {step_name}", "details": str(e)}
                break
        
        if state.get("error_dict"):
            return {
                "status": "error",
                "error": state["error_dict"],
                "output_directory": str(self.output_dir)
            }
        
        final_results = self._compile_results(state)
        
        # Save final results JSON
        final_path = self.output_dir / "analysis_results.json"
        with open(final_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"✅ ANALYSIS COMPLETE")
        self.logger.info(f"   Results saved to: {final_path}")
        self.logger.info(f"{'='*80}\n")
        
        # Log action
        self._log_action(
            action="analyze",
            input_ctx={
                "num_images": num_images,
                "input_type": input_type,
                "series_metadata": series_metadata
            },
            result=final_results.get("summary"),
            rationale="SAM analysis completed."
        )
        
        return final_results
    
    def _compile_results(self, state: dict) -> Dict[str, Any]:
        """
        Compile results into a consistent output structure.
        Includes segmentation quality metadata from refinement/judge.
        """
        is_single = state.get("is_single_image", False)
        num_images = state.get("num_images", 1)
        batch_results = state.get("batch_results", [])
        
        # Build segmentation quality metadata
        segmentation_quality = {}
        
        llm_eval = state.get("llm_quality_evaluation", {})
        if llm_eval:
            segmentation_quality["evaluation"] = llm_eval
        
        refinement_iters = state.get("llm_refinement_iterations", 0)
        if refinement_iters:
            segmentation_quality["refinement_iterations"] = refinement_iters
        
        refinement_history = state.get("refinement_history", [])
        if refinement_history:
            segmentation_quality["refinement_history"] = refinement_history
        
        judge_invoked = state.get("judge_invoked", False)
        if judge_invoked:
            segmentation_quality["judge_invoked"] = True
            segmentation_quality["judge_selected_iteration"] = state.get("judge_selected_iteration")
        
        results = {
            "status": "success",
            "summary": {
                "total_images": num_images,
                "successful": sum(1 for r in batch_results if r.get("success", False)),
                "input_type": state.get("input_type"),
                "parameters_used": state.get("final_params_for_batch", state.get("current_params", {})),
                "is_single_image": is_single
            },
            "output_directory": str(self.output_dir)
        }
        
        # Include segmentation quality metadata if present
        if segmentation_quality:
            results["segmentation_quality"] = segmentation_quality
        
        if is_single:
            # Single image: provide simplified structure for backward compatibility
            synthesis = state.get("synthesis_result", {})
            
            results["detailed_analysis"] = synthesis.get(
                "detailed_analysis", 
                "Analysis complete."
            )
            results["scientific_claims"] = synthesis.get("scientific_claims", [])
            
            # Also include the raw data for those who want it
            if batch_results:
                results["statistics"] = batch_results[0].get("statistics", {})
                results["particle_count"] = batch_results[0].get("particle_count", 0)
                results["visualization_path"] = batch_results[0].get("visualization_path")
        else:
            # Batch: full structure
            results["individual_results"] = batch_results
            results["custom_analysis"] = state.get("custom_analysis_results", {})
            
            synthesis = state.get("synthesis_result", {})
            results["detailed_analysis"] = synthesis.get("detailed_analysis", "")
            results["scientific_claims"] = synthesis.get("scientific_claims", [])
            results["synthesis"] = synthesis
        
        return results
    
    # =========================================================================
    # BACKWARD COMPATIBLE METHODS
    # =========================================================================
    
    def analyze_image_series(
        self,
        image_paths: Optional[List[str]] = None,
        image_stack: Optional[np.ndarray] = None,
        system_info: Optional[Union[dict, str]] = None,
        series_metadata: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze a series of images.

        BACKWARD COMPATIBLE: Delegates to unified analyze() method.

        Args:
            image_paths: List of file paths to images
            image_stack: 3D numpy array (n_images x height x width)
            system_info: System/sample metadata
            series_metadata: Dict with ``"variable"`` (str), ``"values"``
                (list), and ``"unit"`` (str) describing the independent
                variable across the series
        """
        if image_paths is not None:
            return self.analyze(
                image_paths,
                system_info=system_info,
                series_metadata=series_metadata
            )
        elif image_stack is not None:
            return self.analyze(
                image_stack,
                system_info=system_info,
                series_metadata=series_metadata
            )
        else:
            return {
                "status": "error",
                "error": {"error": "No input", "details": "Must provide image_paths or image_stack"},
                "output_directory": str(self.output_dir)
            }
    
    def _get_claims_instruction_prompt(self) -> str:
        return SAM_MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return SAM_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS