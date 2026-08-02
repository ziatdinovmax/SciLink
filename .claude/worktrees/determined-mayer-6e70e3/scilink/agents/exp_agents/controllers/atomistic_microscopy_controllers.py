import logging
import logging
import atomai as aoi
import json
from typing import Callable

from ....tools import atomistic_tools 

# --- TOOL CONTROLLERS ---

class RunAtomDetectionController:
    """
    [🛠️ Tool Step]
    Runs the DCNN ensemble to detect atomic coordinates.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.refine = settings.get('refine_positions', False)
        self.max_shift = settings.get('max_refinement_shift', 1.5)
        self.thresh = settings.get('detection_threshold', 0.8)

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n🛠️ --- CALLING TOOL: ATOM DETECTION (DCNN) --- 🛠️\n")
        
        try:
            nn_output, coordinates = atomistic_tools.predict_with_ensemble(
                dir_path=state["model_dir_path"],
                image=state["preprocessed_image_array"],
                logger=self.logger,
                thresh=self.thresh,
                refine=self.refine,
                max_refinement_shift=self.max_shift
            )
            
            if coordinates is None or len(coordinates) == 0:
                self.logger.warning("No atoms detected by NN ensemble.")
                state["error_dict"] = {"error": "No atoms detected"}
                return state

            self.logger.info(f"✅ Tool Complete: Detected {len(coordinates)} atoms.")
            state["nn_output"] = nn_output
            state["coordinates"] = coordinates

        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Atom detection failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Atom detection failed", "details": str(e)}
            
        return state

class RunIntensityAnalysisController:
    """
    [🛠️ Tool Step]
    Extracts atomic intensities and generates the initial histogram.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.box_size = settings.get('intensity_box_size', 2)

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: INTENSITY EXTRACTION --- 🛠️\n")
        
        try:
            intensities = atomistic_tools.extract_atomic_intensities(
                state["preprocessed_image_array"],
                state["coordinates"],
                self.box_size
            )
            hist_bytes = atomistic_tools.create_intensity_histogram_plot(intensities)
            
            self.logger.info(f"✅ Tool Complete: Extracted {len(intensities)} intensities.")
            state["intensities"] = intensities
            state["intensity_histogram_bytes"] = hist_bytes
        
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Intensity analysis failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Intensity analysis failed", "details": str(e)}
            
        return state

class RunIntensityGMMController:
    """
    [🛠️ Tool Step]
    Performs 1D GMM on intensities and creates visualizations.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings 

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: 1D GMM CLUSTERING --- 🛠️\n")
        
        try:
            n_components = state.get("intensity_gmm_components", 3) # Use LLM value or default
            
            gmm_labels, spatial_maps = atomistic_tools.perform_1d_intensity_gmm(
                state["intensities"],
                state["coordinates"],
                state["preprocessed_image_array"].shape,
                n_components
            )
            
            viz_list = atomistic_tools.create_intensity_gmm_visualization(
                state["intensities"],
                gmm_labels,
                n_components,
                state["coordinates"],
                state["preprocessed_image_array"]
            )
            
            self.logger.info(f"✅ Tool Complete: Performed 1D GMM with {n_components} components.")
            state["intensity_gmm_labels"] = gmm_labels
            state["intensity_spatial_maps"] = spatial_maps
            state["intensity_visualizations"] = viz_list
        
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: 1D GMM clustering failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "1D GMM clustering failed", "details": str(e)}

        return state

class RunLocalEnvAnalysisController:
    """
    [🛠️ Tool Step]
    Runs atomai.stat.imlocal for local environment GMM clustering.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.window_size = settings.get('window_size', 32)

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: LOCAL ENVIRONMENT ANALYSIS --- 🛠️\n")

        try:
            n_components = state.get("local_env_components", 4) # Use LLM value or default
            image_array = state["preprocessed_image_array"]
            coordinates = state["coordinates"]
            
            self.logger.info(f"Starting local environment analysis with {n_components} components and window size {self.window_size}px")

            if image_array.ndim == 2:
                expdata_reshaped = image_array[None, ..., None]
            else:
                expdata_reshaped = image_array
            
            coordinates_for_imlocal = {0: coordinates}
            imstack = aoi.stat.imlocal(expdata_reshaped, coordinates_for_imlocal, window_size=self.window_size)
            
            centroids, _, local_env_coords_and_class = imstack.gmm(n_components)
            
            if local_env_coords_and_class is not None:
                local_env_coords_and_class[:, 2] = local_env_coords_and_class[:, 2] - 1  # Make 0-indexed
            
            self.logger.info("✅ Tool Complete: Local environment analysis finished.")
            state["local_env_centroids"] = centroids
            state["local_env_coords_class"] = local_env_coords_and_class

        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Local environment analysis: {e}", exc_info=True)
            state["local_env_centroids"] = None
            state["local_env_coords_class"] = None
            # Do not set state["error_dict"] as this step is non-critical
            
        return state

class RunNNAnalysisController:
    """
    [🛠️ Tool Step]
    Runs nearest-neighbor distance analysis.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings 

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🛠️ --- CALLING TOOL: NEAREST-NEIGHBOR ANALYSIS --- 🛠️\n")

        try:
            coordinates = state["coordinates"]
            if len(coordinates) > 1:
                final_coordinates_2d = coordinates[:, :2]
                nn_distances = atomistic_tools.analyze_nearest_neighbor_distances(
                    final_coordinates_2d, 
                    pixel_scale=1.0 # Scale is applied later using nm_per_pixel
                )
                self.logger.info("✅ Tool Complete: Nearest-neighbor analysis finished.")
                state["nn_distances"] = nn_distances
            else:
                self.logger.warning("Skipping NN analysis: insufficient atoms.")
                state["nn_distances"] = None
        
        except Exception as e:
            self.logger.error(f"❌ Tool Failed: Nearest-neighbor analysis: {e}", exc_info=True)
            state["nn_distances"] = None
            # Do not set state["error_dict"] as this step is non-critical
            
        return state

# --- LLM CONTROLLERS ---

class GetIntensityGMMParamsController:
    """
    [🧠 LLM Step]
    Asks LLM for n_components for intensity GMM.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        from ..instruct import INTENSITY_GMM_COMPONENT_SELECTION_INSTRUCTIONS
        self.instructions = INTENSITY_GMM_COMPONENT_SELECTION_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- LLM STEP: SELECT INTENSITY GMM PARAMS --- 🧠\n")
        
        prompt_parts = [self.instructions]
        prompt_parts.append("\nOriginal microscopy image:")
        prompt_parts.append(state["image_blob"])
        prompt_parts.append("\nIntensity histogram of detected atoms:")
        prompt_parts.append({"mime_type": "image/jpeg", "data": state["intensity_histogram_bytes"]})
        
        if state.get("system_info"):
            # This helper is in BaseAnalysisAgent, so we build it manually
            system_info_text = f"\n\nAdditional System Information (Metadata):\n{json.dumps(state['system_info'], indent=2)}"
            prompt_parts.append(system_info_text)
        
        prompt_parts.append("\nBased on the histogram and context, determine the optimal number of GMM components.")
        
        param_gen_config = None#GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"LLM intensity GMM param selection failed: {error_dict}")
                state["intensity_gmm_components"] = 3 # Default fallback
                return state

            n_components = result_json.get("n_components")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            
            # --- Pretty-print for user ---
            print("\n" + "="*80)
            print("🧠 LLM REASONING (Intensity GMM Params)")
            print(f"  Suggested n_components: {n_components}")
            print(f"  Explanation: {reasoning}")
            print("="*80 + "\n")
            # --- End pretty-print ---
            
            if isinstance(n_components, int) and 1 <= n_components <= 8:
                self.logger.info(f"✅ LLM Step Complete: Suggested {n_components} components.")
                state["intensity_gmm_components"] = n_components
            else:
                self.logger.warning(f"Invalid component number from LLM: {n_components}. Using default.")
                state["intensity_gmm_components"] = 3 # Default fallback

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Intensity param selection: {e}", exc_info=True)
            state["intensity_gmm_components"] = 3 # Default fallback
            
        return state

class GetLocalEnvParamsController:
    """
    [🧠 LLM Step]
    Asks LLM for n_components for local environment GMM.
    """
    def __init__(self, model, logger, generation_config, safety_settings, parse_fn: Callable):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        from ..instruct import LOCAL_ENV_COMPONENT_SELECTION_INSTRUCTIONS
        self.instructions = LOCAL_ENV_COMPONENT_SELECTION_INSTRUCTIONS

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n🧠 --- LLM STEP: SELECT LOCAL ENV GMM PARAMS --- 🧠\n")
        
        prompt_parts = [self.instructions]
        prompt_parts.append("\nOriginal microscopy image:")
        prompt_parts.append(state["image_blob"])
        
        prompt_parts.append("\nIntensity analysis results:")
        for viz in state.get("intensity_visualizations", []):
            prompt_parts.append(f"\n{viz['label']}:")
            prompt_parts.append({"mime_type": "image/jpeg", "data": viz['bytes']})
        
        if state.get("system_info"):
            system_info_text = f"\n\nAdditional System Information (Metadata):\n{json.dumps(state['system_info'], indent=2)}"
            prompt_parts.append(system_info_text)
        
        prompt_parts.append("\nBased on the intensity analysis and context, determine the optimal number of components for local environment GMM.")
        
        param_gen_config = None#GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"LLM local env param selection failed: {error_dict}")
                state["local_env_components"] = 4 # Default fallback
                return state

            n_components = result_json.get("n_components")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            
            # --- Pretty-print for user ---
            print("\n" + "="*80)
            print("🧠 LLM REASONING (Local Env GMM Params)")
            print(f"  Suggested n_components: {n_components}")
            print(f"  Explanation: {reasoning}")
            print("="*80 + "\n")
            # --- End pretty-print ---
            
            if isinstance(n_components, int) and 1 <= n_components <= 8:
                self.logger.info(f"✅ LLM Step Complete: Suggested {n_components} components.")
                state["local_env_components"] = n_components
            else:
                self.logger.warning(f"Invalid component number from LLM: {n_components}. Using default.")
                state["local_env_components"] = 4 # Default fallback

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: Local env param selection: {e}", exc_info=True)
            state["local_env_components"] = 4 # Default fallback
            
        return state

# --- PREP CONTROLLER ---

class BuildAtomisticPromptController:
    """
    [📝 Prep Step]
    Gathers all analysis results into the final prompt.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.save_viz = settings.get('save_visualizations', True)
        self.viz_output_dir = settings.get('visualization_dir', 'atomistic_analysis_visualizations')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"): return state
        self.logger.info("\n\n📝 --- PREP STEP: BUILDING FINAL PROMPT --- 📝\n")
        
        try:
            # 1. Create comprehensive visualizations
            all_visualizations = atomistic_tools.create_comprehensive_visualization(
                analysis_results=state, # Pass the whole state dict
                original_image=state["preprocessed_image_array"],
                nm_per_pixel=state.get("nm_per_pixel")
            )
            
            # 2. Save visualizations if enabled
            if self.save_viz:
                for viz in all_visualizations:
                    atomistic_tools.save_visualization_to_disk(
                        viz['bytes'], viz['label'], self.logger, self.viz_output_dir
                    )
            
            # 3. Store images for agent/feedback
            analysis_images = []
            for viz in all_visualizations:
                analysis_images.append({
                    "label": viz['label'],
                    "data": viz['bytes']
                })
            state["analysis_images"] = analysis_images # This will be stored by the final controller
            
            # 4. Build final prompt
            prompt_parts = [state["instruction_prompt"]]
            
            if state.get("additional_top_level_context"):
                prompt_parts.append(f"\n\n## Special Considerations:\n{state['additional_top_level_context']}\n")
            
            prompt_parts.append("\nPrimary Microscopy Image:\n")
            prompt_parts.append(state["image_blob"])
            
            prompt_parts.append("\n\nComprehensive Atomistic Analysis Results:")
            
            summary_text = f"""
Analysis Summary:
- Total atoms detected: {len(state.get('coordinates', []))}
- Intensity GMM components: {state.get('intensity_gmm_components', 'N/A')}
- Local environment GMM components: {state.get('local_env_components', 'N/A')}
- Nearest neighbor analysis: {'Completed' if state.get('nn_distances') is not None else 'Skipped'}
"""
            prompt_parts.append(summary_text)
            
            for viz in all_visualizations:
                prompt_parts.append(f"\n{viz['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": viz['bytes']})
                
            if state.get("system_info"):
                system_info_text = f"\n\nAdditional System Information (Metadata):\n{json.dumps(state['system_info'], indent=2)}"
                prompt_parts.append(system_info_text)
            
            prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
            
            state["final_prompt_parts"] = prompt_parts
            self.logger.info("✅ Prep Step Complete: Final prompt is ready.")

        except Exception as e:
            self.logger.error(f"❌ Prep Step Failed: Prompt building failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Failed to build final prompt", "details": str(e)}

        return state