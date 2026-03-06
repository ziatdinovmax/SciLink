import os
import glob
from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent
from .human_feedback import SimpleFeedbackMixin

from .instruct import (
    ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from ...tools import atomistic_tools 
from ...tools.image_processor import (
    load_image, 
    preprocess_image, 
    convert_numpy_to_jpeg_bytes
)
from .pipelines.atomistic_microcopy_pipelines import create_atomistic_pipeline
from ._deprecation import normalize_params


class AtomisticMicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing atomistic microscopy images using a modular,
    controller-based pipeline.
    """

    def __init__(self, 
                 api_key: str | None = None, 
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: str | None = None,
                 # Deprecated params
                 google_api_key: str | None = None,
                 local_model: str = None,
                 # Agent params
                 atomistic_analysis_settings: dict | None = None, 
                 enable_human_feedback: bool = True,
                 output_dir: str = "atomistic_analysis_output"):
        
        # Normalize Params
        self.api_key, self.base_url = normalize_params(
            api_key, google_api_key, base_url, local_model, source="AtomisticMicroscopyAnalysisAgent"
        )
        
        super().__init__(
            api_key=self.api_key, 
            model_name=model_name, 
            base_url=self.base_url,
            output_dir=output_dir,
            enable_human_feedback=enable_human_feedback
        )
        
        self.agent_type = "atomistic_microscopy"

        # --- FIX: Enforce Nested Directory Structure ---
        self.output_dir = self.output_dir.resolve()
        
        # Sub-directories
        viz_dir = self.output_dir / "atomistic_visualizations"
        data_dir = self.output_dir / "atomistic_analysis"
        viz_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Settings
        self.settings = atomistic_analysis_settings if atomistic_analysis_settings else {}
        
        # Inject paths
        self.settings['visualization_dir'] = str(viz_dir)
        self.settings['output_dir'] = str(data_dir)
        
        # Model download configuration
        self.DCNN_MODEL_GDRIVE_ID = self.settings.get('dcnn_model_gdrive_id', '16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p')
        self.DEFAULT_MODEL_DIR = self.settings.get('default_model_dir', "dcnn_trained")
        
        self._recommendation_agent = None

        # --- Pipeline Initialization ---
        self.pipeline = create_atomistic_pipeline(
            model=self.model,
            logger=self.logger,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            settings=self.settings, # Contains the nested paths
            parse_fn=self._parse_llm_response,
            store_fn=self._store_analysis_images
        )
        self.logger.info(f"AtomisticMicroscopyAnalysisAgent initialized. Outputs: {self.output_dir}")

    def _get_initial_state_fields(self) -> dict:
        return {
            "current_image": None,
            "detected_atoms": 0,
            "pipeline_type": "atomistic"
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

            # --- 2. Atomistic-Specific Setup ---
            image_for_analysis = preprocessed_img_array
            if fov_in_nm is not None:
                rescaled_image, _, final_pixel_size_A = atomistic_tools.rescale_for_model(
                    image_for_analysis, fov_in_nm
                )
                image_for_analysis = rescaled_image
                nm_per_pixel = final_pixel_size_A / 10.0
                self.logger.info(f"Image rescaled. New pixel size: {nm_per_pixel*10:.3f} Å/px.")
            else:
                self.logger.warning("Field of view not provided. Skipping rescaling.")
                
            model_dir_path = self._get_or_download_model_path()
            if not model_dir_path:
                return None, {"error": "DCNN model directory not available"}

            # 3. Create State
            state = {
                "image_path": image_path,
                "system_info": system_info,
                "instruction_prompt": instruction_prompt,
                "additional_top_level_context": additional_context,
                "image_blob": {"mime_type": "image/jpeg", "data": image_bytes},
                "preprocessed_image_array": image_for_analysis, # Use the rescaled image
                "model_dir_path": model_dir_path,
                "nm_per_pixel": nm_per_pixel,
                "analysis_images": [
                    {"label": "Primary Microscopy Image", "data": image_bytes}
                ],
                "result_json": None,
                "error_dict": None
            }

            # 4. Run Pipeline
            for controller in self.pipeline:
                state = controller.execute(state)
                if state.get("error_dict"):
                    self.logger.error(f"Pipeline failed at step {controller.__class__.__name__}.")
                    break

            # 5. Return Results
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

    def _get_or_download_model_path(self) -> str | None:
         user_provided_path = self.settings.get('model_dir_path')

         if user_provided_path:
             if not os.path.isdir(user_provided_path):
                 self.logger.error(f"Provided 'model_dir_path' ('{user_provided_path}') does not exist.")
                 return None
             self.logger.info(f"Using user-provided model path: {user_provided_path}")
             return user_provided_path
         
         default_path = self.DEFAULT_MODEL_DIR

         if not os.path.isdir(default_path):
             self.logger.warning(f"Default model directory '{default_path}' not found. Downloading...")
             zip_filename = f"{self.DEFAULT_MODEL_DIR}.zip"
             
             downloaded_zip_path = atomistic_tools.download_file_with_gdown(
                 self.DCNN_MODEL_GDRIVE_ID, zip_filename, self.logger
             )
             
             if not downloaded_zip_path or not os.path.exists(downloaded_zip_path):
                 self.logger.error("Failed to download the model.")
                 return None

             unzip_success = atomistic_tools.unzip_file(downloaded_zip_path, default_path, self.logger)
             
             try:
                 os.remove(downloaded_zip_path)
                 self.logger.info(f"Cleaned up downloaded zip file: {downloaded_zip_path}")
             except OSError as e:
                 self.logger.warning(f"Could not remove zip file {downloaded_zip_path}: {e}")

             if not unzip_success:
                 self.logger.error(f"Failed to unzip model from '{downloaded_zip_path}'.")
                 return None
         
         try:
             if glob.glob(os.path.join(default_path, 'atomnet3*.tar')):
                 return default_path
             
             for item in os.listdir(default_path):
                 sub_path = os.path.join(default_path, item)
                 if os.path.isdir(sub_path) and glob.glob(os.path.join(sub_path, 'atomnet3*.tar')):
                     self.logger.info(f"Found models in nested directory: {sub_path}")
                     return sub_path
         except FileNotFoundError:
             self.logger.error(f"The model directory '{default_path}' does not exist.")
         
         self.logger.error(f"Could not find model files in '{default_path}' or subdirectories.")
         return None

    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        """
        Analyze atomistic microscopy image for DFT structure recommendations.
        """
        # 1. Init State (New)
        self._init_state(
            action="dft_recommendations", 
            image=image_path, 
            has_cached_analysis=bool(cached_detailed_analysis)
        )

        final_result = None
        error_result = None

        # Text-Only path (delegate to RecommendationAgent)
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")
            if not self._recommendation_agent:
                # Updated to use normalized API params
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
        
        # Image-Based path
        elif image_path:
            self.logger.info("Generating DFT recommendations from atomistic analysis pipeline.")
            # Note: We pass the additional context here to influence the pipeline
            result_json, error_dict = self._run_analysis_pipeline(
                image_path, 
                system_info, 
                ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS,  
                additional_context=additional_prompt_context
            )
            
            if error_dict:
                error_result = error_dict
            elif result_json is None:
                error_result = {"error": "Atomistic analysis failed unexpectedly."}
            else:
                recommendations = result_json.get("structure_recommendations", [])
                sorted_recommendations = self._validate_structure_recommendations(recommendations)

                final_result = {
                    "analysis_summary_or_reasoning": result_json.get("detailed_analysis", "Analysis not provided."),
                    "recommendations": sorted_recommendations
                }
        else:
            error_result = {"error": "Either image_path or (cached_detailed_analysis AND ...) must be provided."}

        # 2. Log Action (New)
        if error_result:
            self._log_action(
                action="recommend_structures", 
                input_ctx={"image": image_path}, 
                result={"status": "error", "error": error_result}
            )
            return error_result
        else:
            self._log_action(
                action="recommend_structures",
                input_ctx={"image": image_path},
                result=final_result,
                rationale="Generated DFT structure recommendations."
            )
            return final_result

    def analyze_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """Analyze microscopy image to generate scientific claims"""
        # 1. Init State
        self._init_state(current_image=image_path, system_info=system_info)

        # 2. Run Pipeline
        result_json, error_dict = self._run_analysis_pipeline(
            image_path, system_info, ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict: 
            self._log_action("analyze_for_claims", {"image": image_path}, {"error": error_dict})
            return error_dict

        if result_json is None: 
            return {"error": "Atomistic analysis failed."}

        valid_claims = self._validate_scientific_claims(result_json.get("scientific_claims", []))

        final_result = {
            "detailed_analysis": result_json.get("detailed_analysis", "Analysis not provided."),
            "scientific_claims": valid_claims
        }

        # 4. Log Success
        self._log_action(
            action="analyze_for_claims",
            input_ctx={"image": image_path},
            result=final_result,
            rationale="Atomistic pipeline completed."
        )
        return final_result

    def _get_claims_instruction_prompt(self) -> str:
        return ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS