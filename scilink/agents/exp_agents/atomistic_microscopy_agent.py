import os
import numpy as np
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.mixture import GaussianMixture

from google.generativeai.types import GenerationConfig

from .base_agent import BaseAnalysisAgent
from .recommendation_agent import RecommendationAgent

from .instruct import (
    ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS,
    ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS,
    INTENSITY_GMM_COMPONENT_SELECTION_INSTRUCTIONS,
    LOCAL_ENV_COMPONENT_SELECTION_INSTRUCTIONS,
    ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .utils import (
    load_image, preprocess_image, convert_numpy_to_jpeg_bytes, 
    predict_with_ensemble, analyze_nearest_neighbor_distances, 
    rescale_for_model, download_file_with_gdown, unzip_file,
    extract_atomic_intensities, create_intensity_histogram_plot
)

from .human_feedback import SimpleFeedbackMixin

import atomai as aoi


class AtomisticMicroscopyAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing atomistic microscopy images
    """

    def __init__(self, google_api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 atomistic_analysis_settings: dict | None = None, enable_human_feedback: bool = True):
        super().__init__(google_api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        
        self.atomistic_analysis_settings = atomistic_analysis_settings if atomistic_analysis_settings else {}
        
        # Model download configuration
        self.DCNN_MODEL_GDRIVE_ID = '16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p'
        self.DEFAULT_MODEL_DIR = "dcnn_trained"
        
        # Analysis settings
        self.save_visualizations = self.atomistic_analysis_settings.get('save_visualizations', True)
        self.refine_positions = self.atomistic_analysis_settings.get('refine_positions', False)
        self.max_refinement_shift = self.atomistic_analysis_settings.get('max_refinement_shift', 1.5)
        self.intensity_box_size = self.atomistic_analysis_settings.get('intensity_box_size', 2)  # 2x2 pixel box
        
        self.original_preprocessed_image = None
        self._recommendation_agent = None

    def _get_intensity_gmm_components_from_llm(self, image_blob, intensity_hist_blob, 
                                             system_info) -> tuple[int | None, str | None]:
        """
        Ask LLM to determine number of components for 1D intensity GMM.
        """
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: INTENSITY GMM COMPONENT SELECTION -------------------- 🤖\n")
        
        prompt_parts = [INTENSITY_GMM_COMPONENT_SELECTION_INSTRUCTIONS]
        prompt_parts.append("\nOriginal microscopy image:")
        prompt_parts.append(image_blob)
        prompt_parts.append("\nIntensity histogram of detected atoms:")
        prompt_parts.append(intensity_hist_blob)
        
        if system_info:
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)
        
        prompt_parts.append("\nBased on the intensity histogram and material context, determine the optimal number of GMM components.")
        
        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"LLM intensity GMM component selection failed: {error_dict}")
                return None, None
            
            n_components = result_json.get("n_components")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            
            if isinstance(n_components, int) and 1 <= n_components <= 8:
                self.logger.info(f"LLM suggested {n_components} components for intensity GMM")
                self.logger.info(f"Reasoning: {reasoning}")
                return n_components, reasoning
            else:
                self.logger.warning(f"Invalid component number from LLM: {n_components}")
                return None, reasoning
                
        except Exception as e:
            self.logger.error(f"LLM intensity component selection failed: {e}", exc_info=True)
            return None, None

    def _perform_1d_intensity_gmm(self, intensities: np.ndarray, coordinates: np.ndarray, 
                                image_shape: tuple, n_components: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform 1D GMM on intensities and create spatial maps.
        
        Returns:
            tuple: (gmm_labels, spatial_maps)
                gmm_labels: 1D array of cluster assignments for each atom
                spatial_maps: 3D array (h, w, n_components) showing spatial distribution
        """
        # Fit 1D GMM
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        intensities_reshaped = intensities.reshape(-1, 1)
        gmm_labels = gmm.fit_predict(intensities_reshaped)
        
        # Create spatial maps
        h, w = image_shape
        spatial_maps = np.zeros((h, w, n_components))
        
        for i, (y, x) in enumerate(coordinates[:, :2].astype(int)):
            if 0 <= y < h and 0 <= x < w:
                component = gmm_labels[i]
                spatial_maps[y, x, component] = 1.0
        
        # Apply Gaussian smoothing to make maps more visible
        for c in range(n_components):
            spatial_maps[:, :, c] = cv2.GaussianBlur(spatial_maps[:, :, c], (15, 15), 3)
        
        return gmm_labels, spatial_maps

    def _create_intensity_gmm_visualization(self, intensities: np.ndarray, gmm_labels: np.ndarray,
                                          spatial_maps: np.ndarray, coordinates: np.ndarray,
                                          original_image: np.ndarray) -> list[dict]:
        """Create visualizations for 1D intensity GMM results."""
        visualizations = []
        n_components = spatial_maps.shape[2]
        
        # Intensity histogram with GMM components colored
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use high-contrast colors that are visible on grayscale
        if n_components <= 2:
            colors = ['red', 'cyan']
        elif n_components <= 4:
            colors = ['red', 'cyan', 'lime', 'yellow']
        else:
            colors = ['red', 'cyan', 'lime', 'yellow', 'magenta', 'orange', 'white', 'lightblue']
        
        for c in range(n_components):
            component_intensities = intensities[gmm_labels == c]
            if len(component_intensities) > 0:
                ax.hist(component_intensities, bins=30, alpha=0.7, 
                       color=colors[c % len(colors)], edgecolor='black', linewidth=0.5,
                       label=f'Cluster {c+1} ({len(component_intensities)} atoms)')
        
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Number of Atoms')
        ax.set_title('Intensity Distribution by GMM Component')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations.append({'label': 'Intensity GMM Histogram', 'bytes': buf.getvalue()})
        plt.close()
                
        # Atoms on original image colored by intensity component with better colors
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(original_image, cmap='gray')
        
        for c in range(n_components):
            component_coords = coordinates[gmm_labels == c]
            if len(component_coords) > 0:
                ax.scatter(component_coords[:, 1], component_coords[:, 0], 
                          color=colors[c % len(colors)], s=20, alpha=0.9, 
                          edgecolors='black', linewidth=0.3,
                          label=f'Intensity Cluster {c+1} ({len(component_coords)} atoms)')
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.set_title('Intensity-Based Atomic Clustering')
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations.append({'label': 'Intensity-Based Atomic Clustering', 'bytes': buf.getvalue()})
        plt.close()
        
        return visualizations

    def _get_local_env_components_from_llm(self, image_blob, intensity_visualizations,
                                         system_info) -> tuple[int | None, str | None]:
        """
        Ask LLM to determine number of components for local environment GMM.
        """
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: LOCAL ENVIRONMENT COMPONENT SELECTION -------------------- 🤖\n")
        
        prompt_parts = [LOCAL_ENV_COMPONENT_SELECTION_INSTRUCTIONS]
        prompt_parts.append("\nOriginal microscopy image:")
        prompt_parts.append(image_blob)
        
        prompt_parts.append("\nIntensity analysis results:")
        for viz in intensity_visualizations:
            prompt_parts.append(f"\n{viz['label']}:")
            prompt_parts.append({"mime_type": "image/jpeg", "data": viz['bytes']})
        
        if system_info:
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)
        
        prompt_parts.append("\nBased on the intensity analysis and material context, determine the optimal number of components for local environment GMM.")
        
        param_gen_config = GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"LLM local environment component selection failed: {error_dict}")
                return None, None
            
            n_components = result_json.get("n_components")
            reasoning = result_json.get("reasoning", "No reasoning provided.")
            
            if isinstance(n_components, int) and 1 <= n_components <= 8:
                self.logger.info(f"LLM suggested {n_components} components for local environment GMM")
                self.logger.info(f"Reasoning: {reasoning}")
                return n_components, reasoning
            else:
                self.logger.warning(f"Invalid component number from LLM: {n_components}")
                return None, reasoning
                
        except Exception as e:
            self.logger.error(f"LLM local environment component selection failed: {e}", exc_info=True)
            return None, None

    def _run_atomistic_analysis(self, image_array: np.ndarray, model_dir_path: str, 
                                       image_blob, system_info) -> dict:
        """
        Run the complete atomistic analysis workflow.
        """
        results = {}
        
        # Step 1: Run NN ensemble to find atoms
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: NEURAL NETWORK ATOM DETECTION -------------------- 🤖\n")
        
        nn_output, coordinates = predict_with_ensemble(
            model_dir_path,
            image_array,
            refine=self.refine_positions,
            max_refinement_shift=self.max_refinement_shift
        )
        print()
        if coordinates is None or len(coordinates) == 0:
            self.logger.warning("No atoms detected by NN ensemble")
            return {"error": "No atoms detected"}
        
        self.logger.info(f"Detected {len(coordinates)} atomic coordinates")
        results['nn_output'] = nn_output
        results['coordinates'] = coordinates
        
        # Step 2: Analyze intensities
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: INTENSITY ANALYSIS -------------------- 🤖\n")
        
        intensities = extract_atomic_intensities(
            image_array, coordinates, self.intensity_box_size
        )
        intensity_hist_bytes = create_intensity_histogram_plot(intensities)
        intensity_hist_blob = {"mime_type": "image/jpeg", "data": intensity_hist_bytes}
        
        self.logger.info(f"Extracted intensities from {len(intensities)} atoms using {self.intensity_box_size}x{self.intensity_box_size} pixel boxes")
        self.logger.info(f"Intensity statistics: mean={np.mean(intensities):.3f}, std={np.std(intensities):.3f}, range=[{np.min(intensities):.3f}, {np.max(intensities):.3f}]")
        
        results['intensities'] = intensities
        results['intensity_histogram'] = intensity_hist_bytes
        
        # Step 3: LLM selects number of GMM components for intensity
        intensity_gmm_components, intensity_reasoning = self._get_intensity_gmm_components_from_llm(
            image_blob, intensity_hist_blob, system_info
        )
        
        if intensity_gmm_components is None:
            intensity_gmm_components = 3  # Default fallback
            self.logger.warning("Using default 3 components for intensity GMM")
        
        # Step 4: Do 1D GMM and plot spatial maps
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: 1D INTENSITY GMM ANALYSIS -------------------- 🤖\n")
        
        gmm_labels, spatial_maps = self._perform_1d_intensity_gmm(
            intensities, coordinates, image_array.shape, intensity_gmm_components
        )
        
        # Log component statistics
        for c in range(intensity_gmm_components):
            n_atoms_in_component = np.sum(gmm_labels == c)
            component_intensities = intensities[gmm_labels == c]
            if len(component_intensities) > 0:
                self.logger.info(f"Intensity component {c}: {n_atoms_in_component} atoms, mean intensity={np.mean(component_intensities):.3f}")
        
        intensity_visualizations = self._create_intensity_gmm_visualization(
            intensities, gmm_labels, spatial_maps, coordinates, image_array
        )
        
        results['intensity_gmm_labels'] = gmm_labels
        results['intensity_spatial_maps'] = spatial_maps
        results['intensity_visualizations'] = intensity_visualizations
        
        # Step 5: LLM selects components for local environment analysis
        local_env_components, local_env_reasoning = self._get_local_env_components_from_llm(
            image_blob, intensity_visualizations, system_info
        )
        
        if local_env_components is None:
            local_env_components = 4  # Default fallback
            self.logger.warning("Using default 4 components for local environment GMM")
        
        # Step 6: Local environment analysis (same as before)
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: LOCAL ENVIRONMENT GMM ANALYSIS -------------------- 🤖\n")
        
        # Calculate window size for local environment analysis
        window_size = self.atomistic_analysis_settings.get('window_size', 32)
        self.logger.info(f"Starting local environment analysis with {local_env_components} components and window size {window_size}px")
        
        try:
            # Use atomai's imlocal for local environment analysis
            if image_array.ndim == 2:
                expdata_reshaped = image_array[None, ..., None]
            else:
                expdata_reshaped = image_array
            
            coordinates_for_imlocal = {0: coordinates}
            self.logger.info(f"Extracting local patches around {len(coordinates)} atoms...")
            imstack = aoi.stat.imlocal(expdata_reshaped, coordinates_for_imlocal, window_size=window_size)
            
            self.logger.info(f"Running GMM clustering with {local_env_components} components...")
            centroids, _, local_env_coords_and_class = imstack.gmm(local_env_components)
            
            if local_env_coords_and_class is not None:
                local_env_coords_and_class[:, 2] = local_env_coords_and_class[:, 2] - 1  # Make 0-indexed
                
                # Log local environment statistics
                for c in range(local_env_components):
                    n_atoms_in_env = np.sum(local_env_coords_and_class[:, 2] == c)
                    self.logger.info(f"Local environment {c}: {n_atoms_in_env} atoms")
                
                self.logger.info(f"Local environment analysis completed successfully. Processed {len(local_env_coords_and_class)} atoms (out of {len(coordinates)} detected)")
            else:
                self.logger.warning("Local environment GMM returned no results")
            
            results['local_env_centroids'] = centroids
            results['local_env_coords_class'] = local_env_coords_and_class
            
        except Exception as e:
            self.logger.error(f"Local environment analysis failed: {e}")
            results['local_env_centroids'] = None
            results['local_env_coords_class'] = None
        
        # Step 7: Nearest neighbor distance analysis
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: NEAREST NEIGHBOR ANALYSIS -------------------- 🤖\n")
        
        if len(coordinates) > 1:
            final_coordinates_2d = coordinates[:, :2]
            nn_distances = analyze_nearest_neighbor_distances(final_coordinates_2d, pixel_scale=1.0)
            self.logger.info(f"Computed nearest neighbor distances for {len(coordinates)} atoms")
            self.logger.info(f"NN distance statistics: mean={np.mean(nn_distances):.3f}, std={np.std(nn_distances):.3f}, range=[{np.min(nn_distances):.3f}, {np.max(nn_distances):.3f}] pixels")
            results['nn_distances'] = nn_distances
        else:
            self.logger.warning("Insufficient atoms for nearest neighbor analysis (need >1 atom)")
            results['nn_distances'] = None
        
        return results

    def _create_comprehensive_visualization(self, analysis_results: dict, original_image: np.ndarray,
                                          nm_per_pixel: float = None) -> list[dict]:
        """Create comprehensive visualizations for the final LLM analysis."""
        all_visualizations = []
        
        # Add intensity analysis visualizations
        if 'intensity_visualizations' in analysis_results:
            all_visualizations.extend(analysis_results['intensity_visualizations'])
        
        # Add local environment visualizations (if available)
        if analysis_results.get('local_env_centroids') is not None:
            local_env_viz = self._create_local_env_visualization(
                original_image, 
                analysis_results['local_env_centroids'],
                analysis_results['local_env_coords_class']
            )
            all_visualizations.extend(local_env_viz)
        
        # Add nearest neighbor visualizations (if available)
        if analysis_results.get('nn_distances') is not None:
            nn_viz = self._create_nn_distance_visualization(
                original_image,
                analysis_results['coordinates'],
                analysis_results['nn_distances'],
                nm_per_pixel
            )
            all_visualizations.extend(nn_viz)
        
        return all_visualizations

    def _create_local_env_visualization(self, original_image: np.ndarray, centroids: np.ndarray,
                                      coords_class: np.ndarray) -> list[dict]:
        """Create visualizations for local environment GMM results."""
        visualizations = []
        n_components = centroids.shape[0]
        
        # 1. GMM Centroids - only show non-empty components
        actual_components = []
        component_counts = []
        
        # Check which components actually have atoms assigned
        for i in range(n_components):
            count = np.sum(coords_class[:, 2] == i)
            if count > 0:
                actual_components.append(i)
                component_counts.append(count)
        
        if len(actual_components) == 0:
            self.logger.warning("No atoms assigned to any local environment component")
            return visualizations
        
        # Create subplot grid for actual components only
        n_actual = len(actual_components)
        n_cols = min(4, n_actual)
        n_rows = (n_actual + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
        if n_actual == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Calculate global normalization for consistent scaling
        global_min = np.min(centroids[actual_components])
        global_max = np.max(centroids[actual_components])
        
        for idx, comp_idx in enumerate(actual_components):
            axes[idx].imshow(centroids[comp_idx, :, :, 0], cmap='viridis', 
                           vmin=global_min, vmax=global_max)
            axes[idx].set_title(f'Local Env Class {comp_idx}\n({component_counts[idx]} atoms)')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(n_actual, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle("Local Environment GMM Centroids")
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations.append({'label': 'Local Environment Centroids', 'bytes': buf.getvalue()})
        plt.close()
        
        # 2. Classified atom map - use high contrast colors
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(original_image, cmap='gray')
        
        # Use distinct, high-contrast colors
        colors = ['red', 'cyan', 'lime', 'yellow', 'magenta', 'orange', 'white', 'lightblue']
        
        for idx, comp_idx in enumerate(actual_components):
            class_coords = coords_class[coords_class[:, 2] == comp_idx]
            if len(class_coords) > 0:
                color = colors[idx % len(colors)]
                ax.scatter(class_coords[:, 1], class_coords[:, 0], 
                          color=color, s=20, alpha=0.9, edgecolors='black', linewidth=0.3,
                          label=f'Local Env {comp_idx} ({len(class_coords)} atoms)')
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.set_title("Local Environment Classification")
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations.append({'label': 'Local Environment Classification Map', 'bytes': buf.getvalue()})
        plt.close()
        
        return visualizations

    def _create_nn_distance_visualization(self, original_image: np.ndarray, coordinates: np.ndarray,
                                        nn_distances: np.ndarray, nm_per_pixel: float = None) -> list[dict]:
        """Create nearest neighbor distance visualizations."""
        visualizations = []
        
        # Distance units
        units = "nm" if nm_per_pixel else "pixels"
        scale = nm_per_pixel if nm_per_pixel else 1.0
        scaled_distances = nn_distances * scale
        
        # 1. Distance map
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(original_image, cmap='gray')
        
        scatter = ax.scatter(coordinates[:, 1], coordinates[:, 0], 
                           c=scaled_distances, cmap='inferno', s=10, alpha=0.9)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'Nearest-Neighbor Distance ({units})')
        
        ax.set_title("Nearest-Neighbor Distance Map")
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations.append({'label': 'NN Distance Map', 'bytes': buf.getvalue()})
        plt.close()
        
        # 2. Distance histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scaled_distances, bins='auto', color='skyblue', edgecolor='black')
        ax.set_xlabel(f"Distance ({units})")
        ax.set_ylabel("Frequency")
        ax.set_title("Nearest-Neighbor Distance Distribution")
        
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        visualizations.append({'label': 'NN Distance Histogram', 'bytes': buf.getvalue()})
        plt.close()
        
        return visualizations

    def _save_visualization_to_disk(self, image_bytes: bytes, label: str):
        """Save visualization to disk."""
        if self.save_visualizations:
            try:
                from datetime import datetime
                output_dir = "atomistic_analysis_visualizations"
                os.makedirs(output_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{label.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.jpeg"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                self.logger.info(f"📸 Saved visualization: {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save visualization: {e}")

    def _analyze_image_base(self, image_path: str, system_info: dict | str | None,
                          instruction_prompt: str, 
                          additional_top_level_context: str | None = None) -> tuple[dict | None, dict | None]:
        """
        Analysis workflow
        """
        try:
            # Clear any previous stored images
            self._clear_stored_images()
            # Use base class methods for common operations
            system_info = self._handle_system_info(system_info)
            loaded_image = load_image(image_path)
            nm_per_pixel, fov_in_nm = self._calculate_spatial_scale(system_info, loaded_image.shape)

            # Rescale image for optimal NN performance
            image_for_analysis = loaded_image
            if fov_in_nm is not None:
                rescaled_image, _, final_pixel_size_A = rescale_for_model(image_for_analysis, fov_in_nm)
                image_for_analysis = rescaled_image
                nm_per_pixel = final_pixel_size_A / 10.0
                self.logger.info(f"Image rescaled for optimal NN performance. New pixel size: {nm_per_pixel*10:.3f} Å/px.")
            else:
                self.logger.warning("Field of view not provided. Skipping image rescaling.")

            # Preprocess the image
            preprocessed_img_array, _ = preprocess_image(image_for_analysis)
            self.original_preprocessed_image = preprocessed_img_array

            # Create image blob for LLM
            image_bytes = convert_numpy_to_jpeg_bytes(preprocessed_img_array)
            image_blob = {"mime_type": "image/jpeg", "data": image_bytes}

            # Get model path
            model_dir_path = self._get_or_download_model_path()
            if not model_dir_path:
                return None, {"error": "DCNN model directory not available"}

            # Run the atomistic analysis workflow
            self.logger.info(f"Using DCNN models from: {model_dir_path}")
            analysis_results = self._run_atomistic_analysis(
                preprocessed_img_array, model_dir_path, image_blob, system_info
            )

            if "error" in analysis_results:
                return None, analysis_results

            # Create comprehensive visualizations for final LLM analysis
            self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: CREATING COMPREHENSIVE VISUALIZATIONS -------------------- 🤖\n")
            
            all_visualizations = self._create_comprehensive_visualization(
                analysis_results, preprocessed_img_array, nm_per_pixel
            )

            # Save visualizations to disk if enabled
            for viz in all_visualizations:
                self._save_visualization_to_disk(viz['bytes'], viz['label'])

            analysis_images = []
            for viz in all_visualizations:
                analysis_images.append({
                    "label": viz['label'],
                    "data": viz['bytes']
                })

            analysis_metadata = {
                "image_path": image_path,
                "system_info": system_info,
                "atoms_detected": len(analysis_results.get('coordinates', [])),
                "num_visualizations": len(all_visualizations),
                "num_stored_images": len(analysis_images)
            }
            self._store_analysis_images(analysis_images, analysis_metadata)

            # Build prompt for final LLM analysis
            prompt_parts = [instruction_prompt]
            
            if additional_top_level_context:
                prompt_parts.append("\n\n## Special Considerations for This Analysis (Based on Prior Review):\n")
                prompt_parts.append(additional_top_level_context)
                prompt_parts.append("\nPlease ensure your analysis specifically addresses these considerations.\n")
            
            # Add original image
            analysis_request_text = "\nPlease analyze the following atomic-resolution microscopy image"
            if system_info: 
                analysis_request_text += " using the additional context provided."
            analysis_request_text += "\n\nPrimary Microscopy Image:\n"
            prompt_parts.append(analysis_request_text)
            prompt_parts.append(image_blob)

            # Add comprehensive analysis results
            prompt_parts.append("\n\nComprehensive Atomistic Analysis Results:")
            
            # Add analysis summary
            summary_text = f"""
Analysis Summary:
- Total atoms detected: {len(analysis_results.get('coordinates', []))}
- Intensity analysis: {len(analysis_results.get('intensities', []))} intensity values extracted
- Intensity GMM components: {analysis_results['intensity_spatial_maps'].shape[2] if 'intensity_spatial_maps' in analysis_results else 'N/A'}
- Local environment GMM components: {analysis_results['local_env_centroids'].shape[0] if analysis_results.get('local_env_centroids') is not None else 'N/A'}
- Nearest neighbor analysis: {'Completed' if analysis_results.get('nn_distances') is not None else 'Skipped (insufficient atoms)'}
"""
            prompt_parts.append(summary_text)

            # Add all visualizations
            for viz in all_visualizations:
                prompt_parts.append(f"\n{viz['label']}:")
                prompt_parts.append({"mime_type": "image/jpeg", "data": viz['bytes']})

            # Add system info
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)
            
            prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
            
            # Final LLM analysis
            self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: FINAL COMPREHENSIVE ANALYSIS -------------------- 🤖\n")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            return self._parse_llm_response(response)

        except FileNotFoundError:
            self._clear_stored_images()
            self.logger.error(f"Image file not found: {image_path}")
            return None, {"error": "Image file not found", "details": f"Path: {image_path}"}
        except Exception as e:
            self._clear_stored_images()
            self.logger.exception(f"Atomistic analysis failed: {e}")
            return None, {"error": "Atomistic analysis failed", "details": str(e)}

    # Keep the existing methods for getting/downloading model path
    def _get_or_download_model_path(self) -> str | None:
        """
        Manages finding or downloading the DCNN models.
        """
        user_provided_path = self.atomistic_analysis_settings.get('model_dir_path')

        if user_provided_path:
            if not os.path.isdir(user_provided_path):
                self.logger.error(f"The provided 'model_dir_path' ('{user_provided_path}') does not exist.")
                return None
            self.logger.info(f"Using user-provided model path: {user_provided_path}")
            return user_provided_path
        
        default_path = self.DEFAULT_MODEL_DIR

        if not os.path.isdir(default_path):
            self.logger.warning(f"Default model directory '{default_path}' not found. Downloading...")
            zip_filename = f"{self.DEFAULT_MODEL_DIR}.zip"
            
            downloaded_zip_path = download_file_with_gdown(self.DCNN_MODEL_GDRIVE_ID, zip_filename)
            
            if not downloaded_zip_path or not os.path.exists(downloaded_zip_path):
                self.logger.error("Failed to download the model.")
                return None

            unzip_success = unzip_file(downloaded_zip_path, default_path)
            
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

    # Keep existing public interface methods but update them to use the new workflow
    def analyze_microscopy_image_for_claims(self, image_path: str, system_info: dict | str | None = None):
        """
        Analyze microscopy image to generate scientific claims
        """
        result_json, error_dict = self._analyze_image_base(
            image_path, system_info, ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS
        )

        if error_dict:
            return error_dict
        if result_json is None:
            return {"error": "Atomistic analysis for claims failed unexpectedly."}

        detailed_analysis = result_json.get("detailed_analysis", "Analysis not provided by LLM.")
        scientific_claims = result_json.get("scientific_claims", [])
        
        valid_claims = self._validate_scientific_claims(scientific_claims)

        if not valid_claims and not detailed_analysis == "Analysis not provided by LLM.":
            self.logger.warning("Analysis successful but no valid claims found.")
        elif not valid_claims:
            self.logger.warning("LLM did not yield valid claims or analysis text.")

        initial_result = {"detailed_analysis": detailed_analysis, "scientific_claims": valid_claims}
        return self._apply_feedback_if_enabled(
            initial_result,
            image_path=image_path,
            system_info=system_info
        )
    
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
        result_json, error_dict = None, None
        output_analysis_key = "detailed_analysis"

        # Text-Only path (delegate to RecommendationAgent)
        if cached_detailed_analysis and additional_prompt_context:
            self.logger.info("Delegating DFT recommendations to RecommendationAgent.")

            if not self._recommendation_agent:
                self._recommendation_agent = RecommendationAgent(self.google_api_key, self.model_name)
            
            return self._recommendation_agent.generate_dft_recommendations_from_text(
                cached_detailed_analysis=cached_detailed_analysis,
                additional_prompt_context=additional_prompt_context,
                system_info=system_info
            )
        
        # Image-Based path
        elif image_path:
            self.logger.info("Generating DFT recommendations from atomistic analysis.")
            result_json, error_dict = self._analyze_image_base(
                image_path, system_info, ATOMISTIC_MICROSCOPY_ANALYSIS_INSTRUCTIONS, 
                additional_top_level_context=additional_prompt_context
            )
        else:
            return {"error": "Either image_path or (cached_detailed_analysis AND additional_prompt_context) must be provided."}

        if error_dict:
            return error_dict
        if result_json is None:
            return {"error": "Atomistic analysis failed unexpectedly."}

        analysis_output_text = result_json.get(output_analysis_key, "Analysis not provided by LLM.")
        recommendations = result_json.get("structure_recommendations", [])
        
        sorted_recommendations = self._validate_structure_recommendations(recommendations)

        if not sorted_recommendations and not analysis_output_text == "Analysis not provided by LLM.":
            self.logger.warning("Analysis successful but no valid recommendations found.")
        elif not sorted_recommendations:
            self.logger.warning("LLM did not yield valid recommendations or analysis text.")

        return {"analysis_summary_or_reasoning": analysis_output_text, "recommendations": sorted_recommendations}
    
    def _get_claims_instruction_prompt(self) -> str:
        return ATOMISTIC_MICROSCOPY_CLAIMS_INSTRUCTIONS
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return ATOMISTIC_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS