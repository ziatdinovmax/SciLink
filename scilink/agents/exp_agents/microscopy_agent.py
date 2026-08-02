from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent
from .human_feedback import SimpleFeedbackMixin
from .instruct import (
    MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    MICROSCOPY_CLAIMS_INSTRUCTIONS,
    MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .pipelines.microscopy_pipelines import create_fftnmf_pipeline

from ...tools.image_processor import (
    load_image, 
    preprocess_image, 
    convert_numpy_to_jpeg_bytes
)
from ._deprecation import normalize_params


class MicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """    
    This agent executes a modular pipeline of "controllers".
    Its behavior is defined by the pipeline loaded in __init__.
    
    Configuration (`fft_nmf_settings`):
    ---------------------------------
    - FFT_NMF_ENABLED (bool): Master switch. If True, the Sliding
      FFT/NMF pipeline will run (in addition to Global FFT).
    - output_dir (str): [Tool Setting] Where to save NMF numpy arrays.
    - visualization_dir (str): [Tool Setting] Where to save NMF plots.
    - ... (other atomai tool settings)
    """

    def __init__(self,
                 api_key: str | None = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: str | None = None,
                 # Deprecated params
                 google_api_key: str | None = None,
                 local_model: str = None,
                 # Agent specific params
                 fft_nmf_settings: dict | None = None,
                 enable_human_feedback: bool = True,
                 output_dir: str = "microscopy_analysis_output"):
        
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="MicroscopyAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key, 
            model_name=model_name, 
            base_url=self.base_url, 
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback,
        )

        self.agent_type = "microscopy"

        self.output_dir = self.output_dir.resolve()
        
        # Define sub-directories nested inside the main output_dir
        # 1. Visualization Dir (for plots, FFTs)
        viz_dir = self.output_dir / "fft_nmf_visualizations"
        # 2. Analysis Dir (for numpy arrays, raw data)
        data_dir = self.output_dir / "analysis_output"
        
        # Agent-Specific Settings
        self.settings = fft_nmf_settings if fft_nmf_settings else {}

        # Tools typically expect strings
        self.settings['visualization_dir'] = str(viz_dir)
        self.settings['output_dir'] = str(data_dir)
        
        # Create them now to be sure
        viz_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        self._recommendation_agent = None
        
        # Pipeline Initialization
        # Pass self.generation_config (which is None) to pipelines
        self.pipeline = create_fftnmf_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings,
            # Pass the agent's methods as dependencies for the controllers
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images
        )
        self.logger.info(f"MicroscopyAnalysisAgent initialized with a pipeline of {len(self.pipeline)} controllers.")

    def _get_initial_state_fields(self) -> dict:
        return {
            "current_image": None,
            "pipeline_type": "general",
            "analysis_results": []
        }
    
    def _run_analysis_pipeline(
        self, 
        image_path: str, 
        system_info: dict, 
        instruction_prompt: str, 
        additional_context: str | None = None
    ) -> tuple[dict | None, dict | None]:
        """
        The agent's main execution engine.
        It prepares the initial state and runs the loaded pipeline.
        All specific logic is now in the controllers.
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

            # This is the "state" dictionary that is passed through the pipeline
            state = {
                "image_path": image_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "additional_top_level_context": additional_context,
                "image_blob": {"mime_type": "image/jpeg", "data": image_bytes},
                "preprocessed_image_array": preprocessed_img_array,
                "nm_per_pixel": nm_per_pixel,
                "fov_in_nm": fov_in_nm,
                "analysis_images": [
                    {"label": "Primary Microscopy Image", "data": image_bytes}
                ],
                "result_json": None, # Will be filled by the pipeline
                "error_dict": None   # Will be filled by the pipeline
            }

            # --- 2. Run the Pipeline ---
            for controller in self.pipeline:
                state = controller.execute(state)
                # Stop pipeline if a major error occurred
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}. Stopping execution.")
                    break

            # --- 3. Return Final Results (from the state) ---
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


    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        """
        Async-safe analyze microscopy image to generate DFT structure recommendations.
        """
        # Text-Only Path
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")
            if not self._recommendation_agent:
                # Use base_url and api_key here as well
                self._recommendation_agent = RecommendationAgent(
                    api_key=self.api_key, 
                    model_name=self.model_name, 
                    base_url=self.base_url
                )
            return self._recommendation_agent.generate_dft_recommendations_from_text(
                cached_detailed_analysis=cached_detailed_analysis,
                additional_prompt_context=additional_prompt_context,
                system_info=system_info
            )
        
        # Image-Based Path
        elif image_path:
            self.logger.info("Generating DFT recommendations via modular pipeline.")
            result_json, error_dict = self._run_analysis_pipeline(
                image_path, 
                system_info, 
                MICROSCOPY_ANALYSIS_INSTRUCTIONS, 
                additional_prompt_context
            )
            
            if error_dict: return error_dict
            if result_json is None: return {"error": "Analysis failed unexpectedly."}

            recommendations = result_json.get("structure_recommendations", [])
            sorted_recs = self._validate_structure_recommendations(recommendations)
            
            if not sorted_recs:
                self.logger.warning("Pipeline ran but LLM returned no valid recommendations.")

            return {
                "analysis_summary_or_reasoning": result_json.get("detailed_analysis", "Analysis complete, but no text was returned."), 
                "recommendations": sorted_recs
            }
        
        else:
            return {"error": "Either image_path or (cached_detailed_analysis...) must be provided."}


    def analyze_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims.
        """
        # 1. Initialize State for this run
        self._init_state(current_image=image_path, system_info=system_info)

        # 2. Run Pipeline
        result_json, error_dict = self._run_analysis_pipeline(
            image_path, 
            system_info, 
            MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict: 
            self._log_action("analyze_for_claims", {"image": image_path}, {"error": error_dict})
            return error_dict

        if result_json is None: 
            return {"error": "Analysis failed."}

        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))
        
        initial_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis complete."),
            "scientific_claims": valid_claims
        }
        
        # 3. Apply Feedback
        final_result = self._apply_feedback_if_enabled(
            initial_result, 
            image_path=image_path, 
            system_info=system_info
        )

        # 4. Log Success
        self._log_action(
            action="analyze_for_claims",
            input_ctx={"image": image_path, "system_info": system_info},
            result=final_result,
            rationale="Standard microscopy analysis pipeline completed."
        )

        return final_result
    
    def _get_claims_instruction_prompt(self) -> str:
        return MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return MICROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS