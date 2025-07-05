import json
import os
import logging
import numpy as np
from io import BytesIO
import cv2 # For grayscale conversion in _run_gmm_analysis
import matplotlib.pyplot as plt # For visualizations
import matplotlib.cm as cm # For visualizations
import glob

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent

from .instruct import (
    ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    GMM_PARAMETER_ESTIMATION_INSTRUCTIONS,
)
from .utils import (
    load_image, preprocess_image, convert_numpy_to_jpeg_bytes, 
    predict_with_ensemble, analyze_nearest_neighbor_distances, 
    rescale_for_model, download_file_with_gdown, unzip_file
)

import atomai as aoi # For imlocal and gmm


class AtomisticMicroscopyAnalysisAgent(BaseAnalysisAgent):
    """
    Agent for analyzing microscopy images using a neural network ensemble for atom finding
    and Gaussian Mixture Model (GMM) clustering for local atomic structure classification.
    """

    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05", atomistic_analysis_settings: dict | None = None):
        super().__init__(google_api_key, model_name)
        
        self.atomistic_analysis_settings = atomistic_analysis_settings if atomistic_analysis_settings else {}
        
        # --- Model Download Configuration ---
        self.DCNN_MODEL_GDRIVE_ID = '16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p'
        self.DEFAULT_MODEL_DIR = "dcnn_trained"
        # ---
        
        self.RUN_GMM = self.atomistic_analysis_settings.get('GMM_ENABLED', True)
        self.GMM_AUTO_PARAMS = self.atomistic_analysis_settings.get('GMM_AUTO_PARAMS', True)
        self.save_visualizations = self.atomistic_analysis_settings.get('save_visualizations', True)
        self.refine_positions = self.atomistic_analysis_settings.get('refine_positions', False)
        self.max_refinement_shift = self.atomistic_analysis_settings.get('max_refinement_shift', 1.5)
        self.original_preprocessed_image = None # Store for potential visualization

        self._recommendation_agent = None

    def _get_gmm_params_from_llm(self, image_blob, system_info) -> tuple[float | None, int | None, str | None]:
        """
        Asks the LLM to suggest GMM parameters based on the image and system info.

        Returns:
            tuple: (window_size_nm, n_components, explanation)
                window_size_nm (float): Suggested window size in nanometers.
                n_components (int): Suggested number of GMM components.
                explanation (str): LLM's reasoning for the parameters.
        """
        self.logger.info("\n\n🤖 -------------------- AGENT STEP: DEFINING ANALYSIS PARAMETERS -------------------- 🤖\n")
        
        prompt_parts = [GMM_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)
        
        if system_info:
            system_info_text = "\n\nAdditional System Information:\n"
            if isinstance(system_info, str):
                try: 
                    system_info_text += json.dumps(json.loads(system_info), indent=2)
                except json.JSONDecodeError: 
                    system_info_text += system_info
            elif isinstance(system_info, dict): 
                system_info_text += json.dumps(system_info, indent=2)
            else: 
                system_info_text += str(system_info)[:1000]
            prompt_parts.append(system_info_text)
            
        prompt_parts.append("\n\nBased on the material science context, output ONLY the JSON object with 'window_size_nm', 'n_components', and 'explanation'.")
        
        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            
            self.logger.debug(f"LLM GMM parameter estimation raw response: {response}")
            
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                self.logger.error(f"LLM GMM parameter estimation blocked: {response.prompt_feedback.block_reason}")
                return None, None, None
                
            if response.candidates and response.candidates[0].finish_reason != 1:
                self.logger.warning(f"LLM GMM parameter estimation finished unexpectedly: {response.candidates[0].finish_reason}")

            raw_text_params = response.text
            result_json_params = json.loads(raw_text_params)
            
            window_size_nm = result_json_params.get("window_size_nm")
            n_components = result_json_params.get("n_components")
            explanation = result_json_params.get("explanation", "No explanation provided.")
            
            # Validate parameters
            params_valid = True
            if not isinstance(window_size_nm, (float, int)) or window_size_nm <= 0:
                self.logger.warning(f"LLM invalid window_size_nm: {window_size_nm}")
                params_valid = False
                window_size_nm = None
                
            # Constraints for n_components for GMM are typically smaller than NMF
            if not isinstance(n_components, int) or not (1 <= n_components <= 8): # Adjusted constraint
                self.logger.warning(f"LLM invalid n_components: {n_components}")
                params_valid = False
                n_components = None
                
            if not isinstance(explanation, str) or not explanation.strip():
                explanation = "Invalid/empty explanation from LLM."
                
            if params_valid:
                self.logger.info(f"LLM suggested params: window_size_nm={window_size_nm}, n_components={n_components}")
                return window_size_nm, n_components, explanation
                
            return None, None, explanation
            
        except Exception as e:
            self.logger.error(f"LLM call for GMM params failed: {e}", exc_info=True)
            return None, None, None

    def _run_gmm_analysis(self, image_array: np.ndarray, model_dir_path: str, window_size: int, n_components: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Performs atom finding using a neural network ensemble and then GMM clustering
        on local patches around detected atoms.
        
        Returns:
            tuple: (nn_output, coordinates, centroids, coords_class)
                nn_output (np.ndarray): Heatmap from NN ensemble.
                coordinates (np.ndarray): Detected atom coordinates (N, 3) with initial class.
                centroids (np.ndarray): Mean images of GMM classes (n_components, h, w, 1).
                coords_class (np.ndarray): Nx3 array of (y, x) coordinates and corresponding GMM classes.
        """
        try:
            self.logger.info("\n\n🤖 -------------------- AGENT STEP: Starting NN Ensemble + GMM Analysis -------------------- 🤖\n")
            
            # Image is already loaded, rescaled, and preprocessed.
            expdata = image_array

            # 1. Make an ensemble prediction with neural nets
            if not model_dir_path or not os.path.isdir(model_dir_path):
                self.logger.error(f"Model directory not provided or does not exist: {model_dir_path}")
                return None, None, None, None
            
            self.logger.info(f"Running NN ensemble prediction from models in: {model_dir_path}")
            nn_output, coordinates = predict_with_ensemble(
                model_dir_path,
                expdata,
                refine=self.refine_positions,
                max_refinement_shift=self.max_refinement_shift
            )
            if coordinates is None or len(coordinates) == 0:
                self.logger.warning("NN ensemble did not detect any coordinates. Aborting GMM analysis.")
                return None, None, None, None
                
            print()
            self.logger.info(f"Detected {len(coordinates)} atomic coordinates.")
            if self.refine_positions:
                self.logger.info("Atomic positions refined to sub-pixel accuracy using 2D Gaussian fitting.")

            # 2. Run sliding window + GMM analysis
            self.logger.info(f"Extracting local patches with window size: {window_size}")
            # atomai expects (N, H, W, 1) for imlocal input
            if expdata.ndim == 2:
                expdata_reshaped = expdata[None, ..., None] # Add batch and channel dims
            else: # Should not happen after grayscale conversion
                self.logger.error("Image data for GMM is not 2D after preprocessing.")
                return None, None, None, None
            
            # Format for imlocal: it expects a dictionary {class_id: coordinates_array}
            coordinates_for_imlocal = {0: coordinates}
            imstack = aoi.stat.imlocal(expdata_reshaped, coordinates_for_imlocal, window_size=window_size)
            
            self.logger.info(f"Running GMM with {n_components} components.")
            centroids, _, gmm_output_coords_and_class = imstack.gmm(n_components)

            # The GMM analysis may discard atoms near the image edge.
            # We use the returned coordinates from GMM for GMM-specific visualizations.
            # We use the original full list of refined coordinates for other analyses (e.g., NN distances).
            coords_class = gmm_output_coords_and_class
            if coords_class is not None:
                # Make GMM classes 0-indexed for consistency
                coords_class[:, 2] = coords_class[:, 2] - 1
                num_initial = len(coordinates)
                num_gmm = len(coords_class)
                if num_initial != num_gmm:
                    self.logger.info(
                        f"GMM analysis processed {num_gmm} atoms out of {num_initial} initial detections "
                        f"(difference is {num_initial - num_gmm}, likely due to atoms near image edges). "
                        "Visualizations will be adjusted accordingly."
                    )

            self.logger.info("GMM analysis complete.")
            # centroids are (n_components, h, w, 1)
            # coords_class is (N, 3) with y, x, class_id
            # nn_output is (H, W)
            return nn_output, coordinates, centroids, coords_class

        except Exception as gmm_e:
            self.logger.error(f"NN+GMM analysis failed: {gmm_e}", exc_info=True)
            return None, None, None, None

    def _save_gmm_visualization_to_disk(self, plot_bytes: bytes, label: str):
        """Save visualization images to disk."""
        try:
            from datetime import datetime
            
            output_dir = "atomistic_analysis_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Add microseconds for uniqueness
            # Sanitize label for filename
            filename = f"{label.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.jpeg"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(plot_bytes)
            
            self.logger.info(f"📸 Saved analysis visualization: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis visualization: {e}")

    def _create_gmm_visualization(self, original_image: np.ndarray, nn_output: np.ndarray, coords_class: np.ndarray, full_coords: np.ndarray, centroids: np.ndarray, nn_distances: np.ndarray | None, nn_dist_units: str) -> list[dict]:
        """
        Creates visualizations for GMM analysis:
        1. GMM Centroids (mean local structures)
        2. Classified Atom Map (atoms on original image, colored by class)
        3. Nearest-Neighbor Distance Map (atoms on original image, colored by NN distance)
        4. Nearest-Neighbor Distance Histogram
        
        Args:
            original_image (np.ndarray): The preprocessed original image (2D grayscale).
            coords_class (np.ndarray): Nx3 array of xy coordinates and corresponding gmm classes.
            full_coords (np.ndarray): The full list of refined coordinates before GMM processing.
            nn_output (np.ndarray): Heatmap from NN ensemble (raw DCNN prediction).
            centroids (np.ndarray): Mean images of GMM classes (n_components, h, w, 1).
            nn_distances (np.ndarray | None): 1D array of nearest-neighbor distances for each atom.
            nn_dist_units (str): The units for the nearest-neighbor distances (e.g., 'nm' or 'pixels').
            
        Returns:
            list[dict]: A list of dictionaries, each containing 'label' and 'bytes' for an image.
        """
        all_images_for_llm = []

        # 0. Plot of NN Ensemble Heatmap overlaid on original image
        fig_nn, ax_nn = plt.subplots(1, 1, figsize=(8, 8))
        ax_nn.imshow(original_image, cmap='gray')
        # Overlay the NN output heatmap with transparency
        ax_nn.imshow(nn_output, cmap='hot', alpha=0.5) 
        ax_nn.set_title("NN Ensemble Prediction Heatmap")
        ax_nn.axis('off')
        plt.tight_layout()
        
        buf_nn = BytesIO()
        plt.savefig(buf_nn, format='jpeg')
        buf_nn.seek(0)
        image_data_nn = buf_nn.getvalue()
        all_images_for_llm.append({'label': 'NN Ensemble Prediction Heatmap', 'bytes': image_data_nn})
        plt.close(fig_nn)

        n_components = centroids.shape[0]

        # 1. Plot of GMM centroids
        n_cols = min(4, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols
        fig_c, axes_c = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        if n_rows == 1 and n_cols == 1: 
            axes_c = np.array([axes_c]) # Handle single subplot case
        axes_c = axes_c.flatten()
        
        for i in range(n_components):
            axes_c[i].imshow(centroids[i, :, :, 0], cmap='viridis') # Centroids are (h, w, 1)
            axes_c[i].set_title(f'Class {i}')
            axes_c[i].axis('off')
            
        for i in range(n_components, len(axes_c)): # Hide unused subplots
            axes_c[i].axis('off')
            
        fig_c.suptitle("GMM Centroids (Mean Local Structures)")
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg') # Save to buffer for LLM
        buf.seek(0)
        image_data_c = buf.getvalue()
        all_images_for_llm.append({'label': 'GMM Centroids (Mean Local Structures)', 'bytes': image_data_c})
        plt.close(fig_c)

        # 2. Plot of classified coordinates on original image
        fig_coords, ax_coords = plt.subplots(1, 1, figsize=(8, 8))
        ax_coords.imshow(original_image, cmap='gray')
        
        colors = cm.get_cmap('viridis', n_components) # Get distinct colors for classes
        for i in range(n_components):
            class_coords = coords_class[coords_class[:, 2] == i]
            # atomai returns coordinates as (row, col) which is (y, x).
            # matplotlib.pyplot.scatter expects (x, y), so we swap the columns.
            ax_coords.scatter(class_coords[:, 1], class_coords[:, 0], color=colors(i), label=f'Class {i}', s=10, alpha=0.8)
        
        ax_coords.legend()
        ax_coords.set_title("Spatially-Resolved GMM Classes")
        ax_coords.axis('off')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='jpeg') # Save to buffer for LLM
        buf.seek(0)
        image_data_coords = buf.getvalue()
        all_images_for_llm.append({'label': 'Classified Atom Map', 'bytes': image_data_coords})
        plt.close(fig_coords)

        # 3. Plot of nearest-neighbor distances as a scatter plot
        # This plot uses the full coordinate list to match the nn_distances array
        if nn_distances is not None and full_coords is not None:
            if len(nn_distances) != len(full_coords):
                self.logger.warning(f"Mismatch in NN distance plot: {len(nn_distances)} distances vs {len(full_coords)} coordinates. Skipping plot.")
                return all_images_for_llm # Return what we have so far
                
            fig_nn_dist, ax_nn_dist = plt.subplots(1, 1, figsize=(8, 8))
            ax_nn_dist.imshow(original_image, cmap='gray')

            # atomai returns coordinates as (row, col) which is (y, x).
            # matplotlib.pyplot.scatter expects (x, y), so we swap the columns.
            x_coords = full_coords[:, 1]
            y_coords = full_coords[:, 0]
            
            # Create the scatter plot, coloring points by distance
            scatter = ax_nn_dist.scatter(x_coords, y_coords, c=nn_distances, cmap='inferno', s=10, alpha=0.9)
            
            # Add a colorbar
            cbar = fig_nn_dist.colorbar(scatter, ax=ax_nn_dist, fraction=0.046, pad=0.04)
            cbar.set_label(f'Nearest-Neighbor Distance ({nn_dist_units})')
            
            ax_nn_dist.set_title("Nearest-Neighbor Distance Map")
            ax_nn_dist.axis('off')
            plt.tight_layout()

            buf_nn_dist = BytesIO()
            plt.savefig(buf_nn_dist, format='jpeg')
            buf_nn_dist.seek(0)
            image_data_nn_dist = buf_nn_dist.getvalue()
            all_images_for_llm.append({'label': 'Nearest-Neighbor Distance Map', 'bytes': image_data_nn_dist})
            plt.close(fig_nn_dist)

        # 4. Plot of nearest-neighbor distances histogram
        if nn_distances is not None:
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            ax_hist.hist(nn_distances, bins='auto', color='skyblue', edgecolor='black')
            ax_hist.set_title("Nearest-Neighbor Distance Distribution")
            ax_hist.set_xlabel(f"Distance ({nn_dist_units})")
            ax_hist.set_ylabel("Frequency (Number of Atoms)")
            plt.tight_layout()

            buf_hist = BytesIO()
            plt.savefig(buf_hist, format='jpeg')
            buf_hist.seek(0)
            image_data_hist = buf_hist.getvalue()
            all_images_for_llm.append({'label': 'Nearest-Neighbor Distance Histogram', 'bytes': image_data_hist})
            plt.close(fig_hist)

        # Save to disk if enabled
        if self.save_visualizations:
            for img_info in all_images_for_llm:
                self._save_gmm_visualization_to_disk(img_info['bytes'], img_info['label'])

        return all_images_for_llm

    def _get_or_download_model_path(self) -> str | None:
        """
        Manages finding or downloading the DCNN models.
        It checks for a user-provided path, a default path, or downloads the models.
        It also robustly handles common nested directory structures within zip files.

        Returns:
            A validated path to the DCNN models, or None if they cannot be found or acquired.
        """
        user_provided_path = self.atomistic_analysis_settings.get('model_dir_path')

        if user_provided_path:
            if not os.path.isdir(user_provided_path):
                self.logger.error(f"The provided 'model_dir_path' ('{user_provided_path}') does not exist or is not a directory.")
                return None
            # If user provides a path, we trust they know what they are doing and don't search deeper.
            self.logger.info(f"Using user-provided model path: {user_provided_path}")
            return user_provided_path
        
        # No user path, so we manage the default path.
        default_path = self.DEFAULT_MODEL_DIR

        # If default path doesn't exist, download and unzip.
        if not os.path.isdir(default_path):
            self.logger.warning(f"Default model directory '{default_path}' not found. Attempting to download and unzip from Google Drive.")
            zip_filename = f"{self.DEFAULT_MODEL_DIR}.zip"
            
            downloaded_zip_path = download_file_with_gdown(self.DCNN_MODEL_GDRIVE_ID, zip_filename)
            
            if not downloaded_zip_path or not os.path.exists(downloaded_zip_path):
                self.logger.error("Failed to download the model.")
                return None

            unzip_success = unzip_file(downloaded_zip_path, default_path)
            
            # Clean up the zip file
            try:
                os.remove(downloaded_zip_path)
                self.logger.info(f"Cleaned up downloaded zip file: {downloaded_zip_path}")
            except OSError as e:
                self.logger.warning(f"Could not remove zip file {downloaded_zip_path}: {e}")

            if not unzip_success:
                self.logger.error(f"Failed to unzip model from '{downloaded_zip_path}'.")
                return None
        
        # At this point, the default_path directory should exist. Now, find the actual model files within it.
        try:
            # Check for models in the root of the unzipped directory.
            if glob.glob(os.path.join(default_path, 'atomnet3*.tar')):
                return default_path
            
            # If not in root, search one level deeper in subdirectories.
            for item in os.listdir(default_path):
                sub_path = os.path.join(default_path, item)
                if os.path.isdir(sub_path) and glob.glob(os.path.join(sub_path, 'atomnet3*.tar')):
                    self.logger.info(f"Found models in nested directory. Adjusting model path to '{sub_path}'.")
                    return sub_path
        except FileNotFoundError:
            self.logger.error(f"The model directory '{default_path}' does not exist, cannot search for models.")
        
        self.logger.error(f"Could not find model files ('atomnet3*.tar') in '{default_path}' or its immediate subdirectories.")
        return None

    def _analyze_image_base(self, image_path: str, system_info: dict | str | None,
                            instruction_prompt: str, 
                            additional_top_level_context: str | None = None) -> tuple[dict | None, dict | None]:
        """
        Internal helper for image-based analysis, including optional NN+GMM.
        Now uses base class methods for common functionality.
        """
        try:
            # Use base class methods for common operations
            system_info = self._handle_system_info(system_info)
            loaded_image = load_image(image_path)
            nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(system_info, loaded_image.shape)

            # --- Rescale image for optimal NN performance ---
            image_for_analysis = loaded_image
            if fov_in_nm is not None:
                rescaled_image, _, final_pixel_size_A = rescale_for_model(image_for_analysis, fov_in_nm)
                # Update image and scale for subsequent steps
                image_for_analysis = rescaled_image
                nm_per_pixel = final_pixel_size_A / 10.0 # Update nm_per_pixel to the new value
                self.logger.info(f"Image rescaled for optimal NN performance. New pixel size: {nm_per_pixel*10:.3f} Å/px.")
            else:
                self.logger.warning("Field of view not provided. Skipping image rescaling for optimal NN performance. Analysis will proceed on the original image scale.")

            # Preprocess the (potentially rescaled) image
            preprocessed_img_array, _ = preprocess_image(image_for_analysis)

            self.original_preprocessed_image = preprocessed_img_array # Store for GMM visualization
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            nn_output, coordinates, centroids, coords_class = None, None, None, None
            if self.RUN_GMM:
                ws_nm, nc, gmm_explanation = None, None, None
                model_dir_path = self._get_or_download_model_path()

                if not model_dir_path:
                    self.logger.error("GMM analysis will be skipped because a valid model directory is not available.")
                else:
                    self.logger.info(f"Using DCNN models from: {model_dir_path}")
                    if self.GMM_AUTO_PARAMS:
                        auto_params = self._get_gmm_params_from_llm(image_blob, system_info)
                        if auto_params: 
                            ws_nm, nc, gmm_explanation = auto_params
                            
                    if gmm_explanation: 
                        self.logger.info(f"Explanation for the selected parameters: {gmm_explanation}")
                    
                    # Determine window size in pixels. Prioritize LLM physical size, then fall back to config.
                    ws_pixels = self.atomistic_analysis_settings.get('window_size', 32) # Default pixel size
                    if ws_nm is not None:
                        if nm_per_pixel is not None and nm_per_pixel > 0:
                            calculated_ws_pixels = int(round(ws_nm / nm_per_pixel))
                            # atomai's imlocal works best with even window sizes
                            if calculated_ws_pixels % 2 != 0:
                                calculated_ws_pixels += 1
                            # Ensure a minimum size
                            ws_pixels = max(8, calculated_ws_pixels)
                            self.logger.info(f"Using LLM-suggested physical window size: {ws_nm:.2f} nm -> {ws_pixels} pixels.")
                        else:
                            self.logger.warning(f"LLM suggested a physical window size ({ws_nm} nm), but image scale (nm/pixel) is unknown. Falling back to default pixel window size of {ws_pixels} px.")
                    else:
                        self.logger.info(f"Using default/configured window size of {ws_pixels} pixels.")

                    if nc is None: 
                        nc = self.atomistic_analysis_settings.get('n_components', 4) # Default n_components
                    
                    nn_output, coordinates, centroids, coords_class = self._run_gmm_analysis(preprocessed_img_array, model_dir_path, ws_pixels, nc)
            
            prompt_parts = [instruction_prompt]
            if additional_top_level_context:
                prompt_parts.append("\n\n## Special Considerations for This Analysis (Based on Prior Review):\n")
                prompt_parts.append(additional_top_level_context)
                prompt_parts.append("\nPlease ensure your DFT structure recommendations and scientific justifications specifically address or investigate these special considerations alongside your general analysis of the image features. The priority should be given to structures that elucidate these highlighted aspects.\n")
            
            analysis_request_text = "\nPlease analyze the following microscopy image"
            if system_info: 
                analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nPrimary Microscopy Image:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)

            if centroids is not None and coords_class is not None:
                prompt_parts.append("\n\nSupplemental Analysis Data (NN Prediction + GMM Clustering):")
                
                # Nearest-neighbor distance analysis
                # This uses the FULL list of refined coordinates for a complete picture.
                nn_distances = None
                nn_dist_units = "pixels"
                if coordinates is not None and len(coordinates) > 1:
                    final_coordinates_2d = coordinates[:, :2] # (y, x)
                    scale = nm_per_pixel if nm_per_pixel else 1.0
                    if nm_per_pixel:
                        nn_dist_units = "nm"
                    nn_distances = analyze_nearest_neighbor_distances(final_coordinates_2d, pixel_scale=scale)

                # Create and add visualizations
                gmm_visualizations = self._create_gmm_visualization(
                    preprocessed_img_array, nn_output, coords_class, coordinates, centroids, nn_distances, nn_dist_units
                )
                for viz in gmm_visualizations:
                    prompt_parts.append(f"\n{viz['label']}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": viz['bytes']})
                    
                self.logger.info(f"Adding {len(gmm_visualizations)} analysis visualizations to prompt.")
                self.logger.info("\n\n🤖 -------------------- AGENT STEP: INTERPRETING RESULTS -------------------- 🤖\n")
            else:
                prompt_parts.append("\n\n(No supplemental NN/GMM analysis results are provided or it was disabled/failed)")

            # Use base class method for system info prompt section
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)
            
            prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
            
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            return self._parse_llm_response(response)  # Using base class method

        except FileNotFoundError:
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except ImportError as e: # Should not happen if dependencies are met
            self.logger.error(f"Missing dependency for image processing: {e}")
            return None, {"error": "Missing image processing dependency", "details": str(e)}
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during image analysis setup or GMM analysis: {e}")
            return None, {"error": "An unexpected error occurred during analysis setup", "details": str(e)}

    def analyze_microscopy_image_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims for literature comparison.
        This path always uses image-based analysis with NN+GMM.
        Now uses base class validation methods.
        """
        result_json, error_dict = self._analyze_image_base(
            image_path, system_info, ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict:
            return error_dict
        if result_json is None:
            return {"error": "Atomistic microscopy analysis for claims failed unexpectedly after LLM processing."}

        detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided by LLM.")
        scientific_claims = result_json.get("scientific_claims", [])
        
        # Use base class validation method
        valid_claims = self._validate_scientific_claims(scientific_claims)

        if not valid_claims and not detailed_analysis == "Analysis not provided by LLM.":
            self.logger.warning("Atomistic microscopy analysis for claims successful ('detailed_analysis' provided) but no valid claims found or parsed.")
        elif not valid_claims:
             self.logger.warning("LLM call did not yield valid claims or analysis text for atomistic microscopy claims workflow.")

        return {"detailed_analysis": detailed_analysis, "scientific_claims": valid_claims}
    
    def analyze_microscopy_image_for_structure_recommendations(
            self,
            image_path: str | None = None,
            system_info: dict | str | None = None,
            additional_prompt_context: str | None = None,
            cached_detailed_analysis: str | None = None
    ):
        """
        Analyze atomistic microscopy image to generate DFT structure recommendations.
        Supports both image-based and text-based analysis paths.
        Now uses base class validation methods.
        
        Args:
            image_path: Path to the microscopy image (required for image-based analysis)
            system_info: System metadata (dict, file path, or None)
            additional_prompt_context: Special considerations/novelty insights for DFT recommendations
            cached_detailed_analysis: Previously generated analysis text (for text-based path)
        
        Returns:
            Dictionary containing analysis summary/reasoning and DFT structure recommendations
        """
        result_json, error_dict = None, None
        # Determine the key for the main textual output from LLM based on the path taken
        output_analysis_key = "detailed_analysis"  # Default for image-based path

        # Text-Only path
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")

            # Lazy initialization of the recommendation agent
            if not self._recommendation_agent:
                self._recommendation_agent = RecommendationAgent(self.google_api_key, self.model_name)
            
            # Delegate the task to the specialized agent
            return self._recommendation_agent.generate_dft_recommendations_from_text(
                cached_detailed_analysis=cached_detailed_analysis,
                additional_prompt_context=additional_prompt_context,
                system_info=system_info
            )
        
        # Image-Based path
        elif image_path:
            self.logger.info("Generating DFT recommendations from atomistic image analysis.")
            instruction_prompt_text = ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS  # Use atomistic-specific instructions
            # additional_prompt_context (novelty string) is passed to _analyze_image_base to be appended
            result_json, error_dict = self._analyze_image_base(
                image_path, system_info, instruction_prompt_text, additional_top_level_context=additional_prompt_context
            )
            # output_analysis_key remains "detailed_analysis"
        else:
            # Neither path is viable
            return {"error": "Either image_path or (cached_detailed_analysis AND additional_prompt_context) must be provided for DFT recommendations."}

        if error_dict:
            return error_dict  # Return error if LLM call or parsing failed
        if result_json is None:  # Safeguard, should be covered by error_dict
            return {"error": "Atomistic analysis failed unexpectedly after LLM processing."}

        # Use the determined key to fetch the main textual output from LLM
        analysis_output_text = result_json.get(output_analysis_key, "Analysis/Reasoning not provided by LLM.")
        recommendations = result_json.get("structure_recommendations", [])
        
        # Use base class validation method
        sorted_recommendations = self._validate_structure_recommendations(recommendations)

        if not sorted_recommendations and not analysis_output_text == "Analysis/Reasoning not provided by LLM.":
            self.logger.warning(f"Atomistic LLM call successful ('{output_analysis_key}' provided) but no valid recommendations found or parsed.")
        elif not sorted_recommendations:
            self.logger.warning("Atomistic LLM call did not yield valid recommendations or analysis text.")

        # Return a consistent key for the main textual output for the calling script
        return {"analysis_summary_or_reasoning": analysis_output_text, "recommendations": sorted_recommendations}