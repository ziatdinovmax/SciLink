import os
import logging
import numpy as np
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.mixture import GaussianMixture
import atomai as aoi
from scipy.optimize import curve_fit
from scipy.spatial import KDTree


def create_intensity_gmm_visualization(
    intensities: np.ndarray, 
    gmm_labels: np.ndarray,
    n_components: int,
    coordinates: np.ndarray,
    original_image: np.ndarray
) -> list[dict]:
    """Create visualizations for 1D intensity GMM results."""
    visualizations = []
    
    # Intensity histogram with GMM components colored
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    visualizations.append({'label': 'Intensity GMM Histogram', 'bytes': buf.getvalue()})
    plt.close(fig)
          
    # Atoms on original image colored by intensity component
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
    fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    visualizations.append({'label': 'Intensity-Based Atomic Clustering', 'bytes': buf.getvalue()})
    plt.close(fig)
    
    return visualizations

def create_local_env_visualization(
    original_image: np.ndarray, 
    centroids: np.ndarray,
    coords_class: np.ndarray
) -> list[dict]:
    """Create visualizations for local environment GMM results."""
    visualizations = []
    if centroids is None or coords_class is None:
        return visualizations
        
    n_components = centroids.shape[0]
    
    # 1. GMM Centroids
    actual_components = []
    component_counts = []
    
    for i in range(n_components):
        count = np.sum(coords_class[:, 2] == i)
        if count > 0:
            actual_components.append(i)
            component_counts.append(count)
    
    if len(actual_components) == 0:
        return visualizations
    
    n_actual = len(actual_components)
    n_cols = min(4, n_actual)
    n_rows = (n_actual + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_actual == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    global_min = np.min(centroids[actual_components])
    global_max = np.max(centroids[actual_components])
    
    for idx, comp_idx in enumerate(actual_components):
        axes[idx].imshow(centroids[comp_idx, :, :, 0], cmap='viridis',  
                           vmin=global_min, vmax=global_max)
        axes[idx].set_title(f'Local Env Class {comp_idx}\n({component_counts[idx]} atoms)')
        axes[idx].axis('off')
    
    for idx in range(n_actual, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle("Local Environment GMM Centroids")
    fig.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    visualizations.append({'label': 'Local Environment Centroids', 'bytes': buf.getvalue()})
    plt.close(fig)
    
    # 2. Classified atom map
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_image, cmap='gray')
    
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
    fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    visualizations.append({'label': 'Local Environment Classification Map', 'bytes': buf.getvalue()})
    plt.close(fig)
    
    return visualizations

def create_nn_distance_visualization(
    original_image: np.ndarray, 
    coordinates: np.ndarray,
    nn_distances: np.ndarray, 
    nm_per_pixel: float = None
) -> list[dict]:
    """Create nearest neighbor distance visualizations."""
    visualizations = []
    
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
    fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    visualizations.append({'label': 'NN Distance Map', 'bytes': buf.getvalue()})
    plt.close(fig)
    
    # 2. Distance histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scaled_distances, bins='auto', color='skyblue', edgecolor='black')
    ax.set_xlabel(f"Distance ({units})")
    ax.set_ylabel("Frequency")
    ax.set_title("Nearest-Neighbor Distance Distribution")
    
    buf = BytesIO()
    fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    visualizations.append({'label': 'NN Distance Histogram', 'bytes': buf.getvalue()})
    plt.close(fig)
    
    return visualizations

def create_comprehensive_visualization(
    analysis_results: dict, 
    original_image: np.ndarray,
    nm_per_pixel: float = None
) -> list[dict]:
    """Create comprehensive visualizations for the final LLM analysis."""
    all_visualizations = []
    
    # Add intensity analysis visualizations
    if 'intensity_visualizations' in analysis_results:
        all_visualizations.extend(analysis_results['intensity_visualizations'])
    
    # Add local environment visualizations (if available)
    if analysis_results.get('local_env_centroids') is not None:
        local_env_viz = create_local_env_visualization(
            original_image,  
            analysis_results['local_env_centroids'],
            analysis_results['local_env_coords_class']
        )
        all_visualizations.extend(local_env_viz)
    
    # Add nearest neighbor visualizations (if available)
    if analysis_results.get('nn_distances') is not None:
        nn_viz = create_nn_distance_visualization(
            original_image,
            analysis_results['coordinates'],
            analysis_results['nn_distances'],
            nm_per_pixel
        )
        all_visualizations.extend(nn_viz)
    
    return all_visualizations

def save_visualization_to_disk(
    image_bytes: bytes, 
    label: str,
    logger: logging.Logger,
    output_dir: str = "atomistic_analysis_visualizations"
):
    """Save visualization to disk."""
    try:
        from datetime import datetime
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{label.replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.jpeg"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        logger.info(f"📸 Saved visualization: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")


def rescale_for_model(image, current_fov_nm, target_pixel_size_A=0.25):
    """
    Rescale to achieve target pixel size while keeping field of view constant
    """
    target_size_px = int((current_fov_nm * 10) / target_pixel_size_A)
    scale_factor = target_size_px / image.shape[0]
    
    if scale_factor > 1.5:
        interpolation = cv2.INTER_CUBIC  # Upscaling
    elif scale_factor < 0.7:
        interpolation = cv2.INTER_AREA   # Downscaling  
    else:
        interpolation = cv2.INTER_LINEAR # Moderate scaling
    
    rescaled_image = cv2.resize(image, (target_size_px, target_size_px), interpolation=interpolation)
    final_pixel_size = (current_fov_nm * 10) / target_size_px
    
    return rescaled_image, scale_factor, final_pixel_size

def _2d_gaussian(xy, amplitude, y0, x0, sigma_y, sigma_x, offset):
    """Helper 2D Gaussian function for fitting."""
    y, x = xy
    y0 = float(y0)
    x0 = float(x0)
    g = offset + amplitude * np.exp(
        -(((y - y0)**2 / (2 * sigma_y**2)) + ((x - x0)**2 / (2 * sigma_x**2)))
    )
    return g.ravel()

def refine_coordinates_gaussian_fit(image_data, coordinates, window_size=7, max_refinement_shift=1.5):
    """Refines atomic coordinates to sub-pixel precision using 2D Gaussian fitting."""
    if coordinates is None or len(coordinates) == 0:
        return coordinates

    refined_coords = []
    h, w = image_data.shape
    half_w = window_size // 2

    for y_int, x_int in coordinates.astype(int):
        y_min, y_max = y_int - half_w, y_int + half_w + 1
        x_min, x_max = x_int - half_w, x_int + half_w + 1

        if y_min < 0 or y_max > h or x_min < 0 or x_max > w:
            refined_coords.append([y_int, x_int])
            continue

        patch = image_data[y_min:y_max, x_min:x_max]
        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]

        try:
            initial_guess = (patch.max(), y_int, x_int, 1, 1, patch.min())
            lower_bounds = [0, y_int - max_refinement_shift, x_int - max_refinement_shift, 0.5, 0.5, 0]
            upper_bounds = [patch.max() * 1.5, y_int + max_refinement_shift, x_int + max_refinement_shift, half_w, half_w, patch.max()]
            bounds = (lower_bounds, upper_bounds)

            popt, _ = curve_fit(_2d_gaussian, (y_grid, x_grid), patch.ravel(), p0=initial_guess, bounds=bounds)
            y_refined, x_refined = popt[1], popt[2]
            
            if (y_min <= y_refined < y_max) and (x_min <= x_refined < x_max):
                refined_coords.append([y_refined, x_refined])
            else:
                refined_coords.append([y_int, x_int])
        except (RuntimeError, ValueError):
            refined_coords.append([y_int, x_int])

    return np.array(refined_coords)

def predict_with_ensemble(dir_path, image, logger, thresh=0.8, refine=True, max_refinement_shift=1.5):
    """Runs the AtomAI DCNN model ensemble."""
    all_predictions = []
    model_pattern = os.path.join(dir_path, 'atomnet3*.tar')
    logger.info(f"Searching for DCNN model files with pattern: {model_pattern}")
    model_files = glob.glob(model_pattern)

    if not model_files:
        logger.error(f"No model files found in '{dir_path}' matching the pattern 'atomnet3*.tar'.")
        raise FileNotFoundError(f"Could not find any DCNN model files ('atomnet3*.tar') in {dir_path}")

    logger.info(f"Found {len(model_files)} models for ensemble prediction.\n")
    for model_file in model_files:
        logger.debug(f"Loading model: {model_file}")
        model = aoi.load_model(model_file)
        prediction = model.predict(image)[0]
        all_predictions.append(prediction)
        
    prediction_mean = np.mean(np.stack(all_predictions), axis=0)
    locator_output = aoi.predictors.Locator(thresh=thresh).run(prediction_mean)

    if isinstance(locator_output, dict):
        coarse_coords_with_class = locator_output.get(0)
    else:
        coarse_coords_with_class = locator_output

    if coarse_coords_with_class is None or len(coarse_coords_with_class) == 0:
        return prediction_mean.squeeze(), None

    if refine:
        coarse_coords_2d = coarse_coords_with_class[:, :2]
        refined_coords_2d = refine_coordinates_gaussian_fit(image, coarse_coords_2d, max_refinement_shift=max_refinement_shift)
        final_coords = np.concatenate((refined_coords_2d, coarse_coords_with_class[:, 2][:, np.newaxis]), axis=1)
    else:
        final_coords = coarse_coords_with_class

    return prediction_mean.squeeze(), final_coords

def analyze_nearest_neighbor_distances(coordinates, pixel_scale=1.0):
    """Calculates the nearest-neighbor distance for each coordinate."""
    if coordinates is None or len(coordinates) < 2:
        return None
    tree = KDTree(coordinates)
    distances, _ = tree.query(coordinates, k=2)
    nearest_neighbor_distances = distances[:, 1] * pixel_scale
    return nearest_neighbor_distances

def download_file_with_gdown(file_id, output_path, logger):
    """Downloads a file from Google Drive using gdown."""
    import gdown
    try:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        gdown.download(id=file_id, output=output_path, quiet=False)
        logger.info(f"File downloaded to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        return None

def unzip_file(zip_filepath, extract_to_dir, logger):
    """Unzips a file to a specified directory."""
    import zipfile
    if not os.path.exists(zip_filepath):
        logger.error(f"Error: Zip file not found at {zip_filepath}")
        return False

    logger.info(f"Unzipping '{zip_filepath}' to '{extract_to_dir}'...")
    try:
        os.makedirs(extract_to_dir, exist_ok=True)
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        logger.info(f"Successfully unzipped to: {extract_to_dir}")
        return True
    except Exception as e:
        logger.error(f"An error occurred during unzipping: {e}")
        return False

def extract_atomic_intensities(image_array: np.ndarray, coordinates: np.ndarray, box_size: int = 2) -> np.ndarray:
    """Extract intensity values from small boxes around detected atomic positions."""
    if coordinates is None or len(coordinates) == 0:
        return np.array([])
    
    intensities = []
    h, w = image_array.shape
    half_box = box_size // 2
    
    for y, x in coordinates[:, :2].astype(int):
        y_min = max(0, y - half_box)
        y_max = min(h, y + half_box + 1)
        x_min = max(0, x - half_box)
        x_max = min(w, x + half_box + 1)
        
        box_intensity = np.mean(image_array[y_min:y_max, x_min:x_max])
        intensities.append(box_intensity)
    
    return np.array(intensities)

def create_intensity_histogram_plot(intensities: np.ndarray, n_bins: int = 50) -> bytes:
    """Create histogram plot of atomic intensities."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(intensities, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Number of Atoms')
    ax.set_title(f'Atomic Intensity Distribution ({len(intensities)} atoms)')
    ax.grid(True, alpha=0.3)
    
    ax.axvline(np.mean(intensities), color='red', linestyle='--', label=f'Mean: {np.mean(intensities):.2f}')
    ax.axvline(np.median(intensities), color='orange', linestyle='--', label=f'Median: {np.median(intensities):.2f}')
    ax.legend()
    
    fig.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close(fig)
    
    return image_bytes

def perform_1d_intensity_gmm(
    intensities: np.ndarray, 
    coordinates: np.ndarray, 
    image_shape: tuple, 
    n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform 1D GMM on intensities and create spatial maps.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    intensities_reshaped = intensities.reshape(-1, 1)
    gmm_labels = gmm.fit_predict(intensities_reshaped)
    
    h, w = image_shape
    spatial_maps = np.zeros((h, w, n_components))
    
    for i, (y, x) in enumerate(coordinates[:, :2].astype(int)):
        if 0 <= y < h and 0 <= x < w:
            component = gmm_labels[i]
            spatial_maps[y, x, component] = 1.0
    
    for c in range(n_components):
        spatial_maps[:, :, c] = cv2.GaussianBlur(spatial_maps[:, :, c], (15, 15), 3)
    
    return gmm_labels, spatial_maps