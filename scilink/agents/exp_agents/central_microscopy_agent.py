from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent
from .human_feedback import SimpleFeedbackMixin
from .pipeline_selector import PipelineSelector
from .pipeline_registry import (
    get_available_pipelines,
    create_pipeline_for_agent,
    get_prompt_for_pipeline
)
from ._deprecation import normalize_params

from ...tools.image_processor import (
    load_image, 
    preprocess_image, 
    convert_numpy_to_jpeg_bytes
)


class CentralMicroscopyAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """      
    Unified microscopy agent that can use multiple analysis pipelines.
    
    The agent uses an LLM-based pipeline selector to automatically choose
    the most appropriate pipeline based on the input image and metadata:
    - 'general': FFT/NMF analysis for standard microstructures
    - 'sam': Particle segmentation for countable objects
    - 'atomistic': Atomic-resolution analysis for crystalline materials
    """

    def __init__(self,
                 api_key: str | None = None,
                 model_name: str = "gemini-3-flash-preview",
                 base_url: str | None = None,
                 # Deprecated params
                 google_api_key: str | None = None,
                 local_model: str = None,
                 # Agent specific params
                 agent_settings: dict | None = None,
                 enable_human_feedback: bool = True,
                 selector_model_name="gemini-3-flash-preview",
                 output_dir: str = "central_microscopy_output",
                 # Backward compatibility parameters
                 fft_nmf_settings: dict | None = None,
                 sam_settings: dict | None = None,
                 atomistic_analysis_settings: dict | None = None):
        
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="CentralMicroscopyAgent"
        )

        super().__init__(
            api_key=self.api_key,
            model_name=model_name,
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "central_microscopy"
        
        # 1. Resolve main output directory to absolute path
        self.output_dir = self.output_dir.resolve()
        
        # 2. Create standard nested folders for ANY pipeline that might run
        # We enforce these paths so tools don't scatter files elsewhere
        self.viz_dir = self.output_dir / "visualizations"
        self.data_dir = self.output_dir / "analysis_data"
        
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Handle backward compatibility for settings
        if agent_settings is None:
            agent_settings = self._build_legacy_settings(
                fft_nmf_settings, sam_settings, atomistic_analysis_settings
            )
        
        self.agent_settings = agent_settings
        self._recommendation_agent = None
        
        # Initialize Pipeline Selector
        self.auto_select = agent_settings.get('auto_select_pipeline', True)
        if self.auto_select:
             self.pipeline_selector = PipelineSelector(
                api_key=self.api_key,
                model_name=selector_model_name,
                base_url=self.base_url
            )
        else:
             self.pipeline_selector = None

        self.default_pipeline_id = agent_settings.get('default_pipeline', 'general')
        
        # These track the active pipeline for the current run
        self.current_pipeline = None
        self.current_pipeline_id = None
        
        self.logger.info(f"CentralMicroscopyAgent initialized. Output: {self.output_dir}")

    def _get_initial_state_fields(self) -> dict:
        return {
            "current_image": None,
            "pipeline_used": None,
            "analysis_results": []
        }

    def _build_legacy_settings(self, fft_nmf_settings, sam_settings, atomistic_settings) -> dict:
        """Convert old-style settings to new unified format."""
        agent_settings = {
            'auto_select_pipeline': True,
            'default_pipeline': 'general',
            'pipeline_settings': {}
        }
        
        if fft_nmf_settings:
            agent_settings['pipeline_settings']['general'] = fft_nmf_settings
            agent_settings['default_pipeline'] = 'general'
        
        if sam_settings:
            agent_settings['pipeline_settings']['sam'] = sam_settings
            if not fft_nmf_settings: 
                agent_settings['default_pipeline'] = 'sam'
        
        if atomistic_settings:
            agent_settings['pipeline_settings']['atomistic'] = atomistic_settings
            if not fft_nmf_settings and not sam_settings: 
                agent_settings['default_pipeline'] = 'atomistic'
        
        return agent_settings

    def _select_and_create_pipeline(self, image_blob: dict, system_info: dict) -> tuple[list, str, str]:
        """
        Select and create the appropriate pipeline, INJECTING directory settings.
        """
        available_pipelines = get_available_pipelines('microscopy')
        
        # 1. Select Pipeline
        if self.auto_select and self.pipeline_selector:
            pipeline_id, reasoning = self.pipeline_selector.select_pipeline(
                available_pipelines=available_pipelines,
                image_blob=image_blob,
                system_info=system_info
            )
            if pipeline_id is None:
                pipeline_id = self.default_pipeline_id
                reasoning = "Fallback to default (Selector failed)."
        else:
            pipeline_id = self.default_pipeline_id
            reasoning = "Auto-selection disabled."

        # 2. Get Settings & Inject Paths
        # Retrieve settings specific to this pipeline ID
        pipeline_settings = self.agent_settings.get('pipeline_settings', {}).get(pipeline_id, {}).copy()
        
        # --- FIX: Inject global output paths into specific pipeline settings ---
        # This ensures the tools (Global FFT, SAM Visualizer, etc.) use the correct folders
        pipeline_settings['visualization_dir'] = str(self.viz_dir)
        pipeline_settings['output_dir'] = str(self.data_dir)
        
        self.logger.info(f"Creating pipeline '{pipeline_id}' with output to {self.data_dir}")
        
        # 3. Instantiate Pipeline
        pipeline = create_pipeline_for_agent(
            pipeline_id=pipeline_id,
            agent_type='microscopy',
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=pipeline_settings,
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images,
        )
        
        return pipeline, pipeline_id, reasoning

    def _run_analysis_pipeline(
        self, 
        image_path: str, 
        system_info: dict, 
        prompt_type: str = 'claims',
        additional_context: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        The agent's main execution engine.
        It selects the appropriate pipeline, prepares the initial state, and runs it.
        """
        try:
            # --- 1. Common State Initialization ---
            self.logger.info(f"--- Starting analysis pipeline for {image_path} ---")
            self._clear_stored_images()
            system_info = self._handle_system_info(system_info)
            
            loaded_image = load_image(image_path)
            nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(system_info, loaded_image.shape)
            
            preprocessed_img_array, _ = preprocess_image(loaded_image)
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}
            
            # --- 2. Pipeline Selection ---
            pipeline, pipeline_id, selection_reasoning = self._select_and_create_pipeline(
                image_blob, system_info
            )
            
            # Store for later use
            self.current_pipeline = pipeline
            self.current_pipeline_id = pipeline_id
            
            # Display selection reasoning
            print("\n" + "="*80)
            print("🔀 PIPELINE SELECTION")
            print(f"  Selected Pipeline: '{pipeline_id}'")
            print(f"  Reasoning: {selection_reasoning}")
            print("="*80 + "\n")
            
            # Get the appropriate instruction prompt for this pipeline
            instruction_prompt = get_prompt_for_pipeline(pipeline_id, 'microscopy', prompt_type)
            
            # --- 3. Create Initial State ---
            # We must pass the INJECTED settings (with paths) to the state
            pipeline_settings = self.agent_settings.get('pipeline_settings', {}).get(pipeline_id, {}).copy()
            pipeline_settings['visualization_dir'] = str(self.viz_dir)
            pipeline_settings['output_dir'] = str(self.data_dir)

            state = {
                "image_path": image_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "additional_top_level_context": additional_context,
                "image_blob": image_blob,
                "preprocessed_image_array": preprocessed_img_array,
                "nm_per_pixel": nm_per_pixel,
                "fov_in_nm": fov_in_nm,
                "analysis_images": [
                    {"label": "Primary Microscopy Image", "data": image_bytes}
                ],
                "result_json": None,
                "error_dict": None,
                "settings": pipeline_settings
            }
            
            # Add atomistic-specific state logic if needed
            if pipeline_id == 'atomistic':
                from ...tools.atomistic_model_manager import get_or_download_atomistic_model
                model_dir_path = get_or_download_atomistic_model(state["settings"], self.logger)
                if not model_dir_path:
                    return None, {"error": "DCNN model directory not available"}
                state["model_dir_path"] = model_dir_path
                
                if fov_in_nm is not None:
                    from ...tools import atomistic_tools
                    rescaled_image, _, final_pixel_size_A = atomistic_tools.rescale_for_model(
                        preprocessed_img_array, fov_in_nm
                    )
                    state["preprocessed_image_array"] = rescaled_image
                    state["nm_per_pixel"] = final_pixel_size_A / 10.0
                    self.logger.info(f"Image rescaled. New pixel size: {state['nm_per_pixel']*10:.3f} Å/px.")

            # --- 4. Run the Pipeline ---
            for controller in pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}. Stopping execution.")
                    break

            # --- 5. Return Final Results ---
            self.logger.info(f"--- Analysis pipeline finished. ---")
            return state.get("result_json"), state.get("error_dict")

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"An unexpected error occurred during the analysis pipeline: {e}")
            return None, {"error": "An unexpected error occurred", "details": str(e)}

    # --- Public API Methods ---

    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        """
        Analyze microscopy image to generate DFT structure recommendations.
        """
        # 1. Init State
        self._init_state(
            action="dft_recommendations", 
            image=image_path, 
            has_cached_analysis=bool(cached_detailed_analysis)
        )

        final_result = None
        error_result = None

        # Text-Only Path
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")
            if not self._recommendation_agent:
                self._recommendation_agent = RecommendationAgent(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    base_url=self.base_url
                )
            
            final_result = self._recommendation_agent.generate_dft_recommendations_from_text(
                cached_detailed_analysis=cached_detailed_analysis,
                additional_prompt_context=additional_prompt_context,
                system_info=system_info
            )
        
        # Image-Based Path
        elif image_path:
            self.logger.info("Generating DFT recommendations via selected pipeline.")
            result_json, error_dict = self._run_analysis_pipeline(
                image_path, 
                system_info, 
                prompt_type='analysis', # Use analysis prompt base
                additional_context=additional_prompt_context
            )
            
            if error_dict:
                error_result = error_dict
            elif result_json is None:
                error_result = {"error": "Analysis failed unexpectedly."}
            else:
                recommendations = result_json.get("structure_recommendations", [])
                sorted_recs = self._validate_structure_recommendations(recommendations)
                
                if not sorted_recs:
                    self.logger.warning("Pipeline ran but LLM returned no valid recommendations.")

                final_result = {
                    "analysis_summary_or_reasoning": result_json.get("detailed_analysis", "Analysis complete, but no text was returned."), 
                    "recommendations": sorted_recs
                }
        else:
            error_result = {"error": "Either image_path or (cached_detailed_analysis...) must be provided."}

        # 2. Log Action
        if error_result:
            self._log_action("recommend_structures", {"image": image_path}, {"error": error_result})
            return error_result
        
        self._log_action(
            action="recommend_structures",
            input_ctx={"image": image_path},
            result=final_result,
            rationale=f"Generated recommendations using pipeline: {self.current_pipeline_id}"
        )
        return final_result

    def analyze_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """Analyze microscopy image to generate scientific claims."""
        
        # 1. Init State
        self._init_state(current_image=image_path, system_info=system_info)
        
        # 2. Run Pipeline (Selection happens inside _run_analysis_pipeline)
        result_json, error_dict = self._run_analysis_pipeline(
            image_path, 
            system_info, 
            prompt_type='claims'
        )

        if error_dict: 
            self._log_action("analyze_for_claims", {"image": image_path}, {"error": error_dict})
            return error_dict

        if result_json is None: 
            return {"error": "Analysis for claims failed unexpectedly."}

        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))
        
        if not valid_claims:
            self.logger.warning("Pipeline ran but LLM returned no valid claims.")
            
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis complete, but no text was returned."), 
            "scientific_claims": valid_claims,
            "pipeline_used": self.current_pipeline_id
        }
        
        # 3. Feedback Loop
        final_result = self._apply_feedback_if_enabled(
            initial_result, 
            image_path=image_path, 
            system_info=system_info
        )

        # 4. Log Success
        self._log_action(
            action="analyze_for_claims",
            input_ctx={"image": image_path},
            result=final_result,
            rationale=f"Selected pipeline: {self.current_pipeline_id}"
        )
        
        return final_result
    
    def _get_claims_instruction_prompt(self) -> str:
        """Return the appropriate claims prompt for the current pipeline."""
        if self.current_pipeline_id:
            return get_prompt_for_pipeline(self.current_pipeline_id, 'microscopy', 'claims')
        return get_prompt_for_pipeline('general', 'microscopy', 'claims')
    
    def _get_measurement_recommendations_prompt(self) -> str:
        """Return the appropriate recommendations prompt for the current pipeline."""
        if self.current_pipeline_id:
            return get_prompt_for_pipeline(self.current_pipeline_id, 'microscopy', 'recommendations')
        return get_prompt_for_pipeline('general', 'microscopy', 'recommendations')