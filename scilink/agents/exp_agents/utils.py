import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import os
import glob

import atomai as aoi


MAX_IMG_DIM = 1024

def load_image(image_path):
    """Load an image from file."""
    try:
        _, ext = os.path.splitext(image_path)
        ext = ext.lower()

        if ext == '.npy':
            # Load .npy file, assuming it's always 2D grayscale based
            img_array = np.load(image_path)
            if img_array.dtype == np.uint8:
                # If it's already uint8, return directly
                return img_array
            else:
                float_array = img_array.astype(np.float64)
                min_val, max_val = np.min(float_array), np.max(float_array)
                normalized_array = (float_array - min_val) / (max_val - min_val)
                # Scale 0-255 and convert to uint8
                uint8_array = (normalized_array * 255).astype(np.uint8)
                return uint8_array

        else:
            # Standard image loading (can be grayscale or color)
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

def preprocess_image(image: np.ndarray, max_dim: int = MAX_IMG_DIM) -> tuple[np.ndarray, float]:
    """
    Preprocess microscopy image for better analysis and ensure it's within model context limits.
    Returns the preprocessed image and the scaling factor used for resizing.
    """
    scale_factor = 1.0
    # Resize if the image is too large
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        # Calculate new dimensions while preserving aspect ratio
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        scale_factor = h / new_h  # or w / new_w
        
        # Resize the image
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Image resized from {w}x{h} to {new_w}x{new_h} to fit model context window")
    
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply noise reduction
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Return the preprocessed image
    return denoised, scale_factor


def convert_numpy_to_jpeg_bytes(image_array: np.ndarray, quality: int = 85) -> bytes:
    """
    Converts a NumPy array image representation into compressed JPEG bytes.
    """
    try:
        pil_img = Image.fromarray(image_array)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=quality)
        return buffered.getvalue()
    except Exception as e:
        print(f"Error converting NumPy array to JPEG bytes: {e}")
        raise # Re-raise the exception to be handled by the caller


def normalize_and_convert_to_image_bytes(array: np.ndarray, mode='L', format='JPEG', quality=85, log_scale=False) -> bytes:
    """
    Normalizes a 2D numpy array and converts it to image bytes (e.g., grayscale JPEG).

    Args:
        array: The 2D numpy array to convert.
        mode: The PIL image mode (e.g., 'L' for grayscale).
        format: The output image format ('JPEG', 'PNG').
        quality: The quality setting for JPEG (1-95).
        log_scale: Apply log1p scaling before normalization (good for FFT magnitudes).

    Returns:
        Image data as bytes.
    """
    if array.ndim != 2:
        # If we get e.g. (1, H, W), try squeezing
        if array.ndim == 3 and array.shape[0] == 1:
            array = np.squeeze(array, axis=0)
        else:
             raise ValueError(f"Input array must be 2D, but got shape {array.shape}")

    try:
        # Make a copy to avoid modifying the original array
        processed_array = array.copy().astype(np.float32) # Ensure float for calculations

        # Optional log scaling
        if log_scale:
            processed_array = np.log1p(processed_array)

        # Handle potential NaN/Inf values introduced by log or present in input
        if not np.all(np.isfinite(processed_array)):
             max_finite = np.max(processed_array[np.isfinite(processed_array)]) if np.any(np.isfinite(processed_array)) else 1.0
             min_finite = np.min(processed_array[np.isfinite(processed_array)]) if np.any(np.isfinite(processed_array)) else 0.0
             processed_array = np.nan_to_num(processed_array, nan=min_finite, posinf=max_finite, neginf=min_finite)

        # Normalize to 0-1 range
        min_val, max_val = np.min(processed_array), np.max(processed_array)
        if max_val > min_val:
            normalized_array = (processed_array - min_val) / (max_val - min_val)
        else:
            # Handle flat array case (all values the same)
            normalized_array = np.zeros_like(processed_array)

        # Scale to 0-255 and convert to uint8
        uint8_array = (normalized_array * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_img = Image.fromarray(uint8_array, mode=mode) # 'L' for grayscale

        # Save to bytes buffer
        buffered = BytesIO()
        if format.upper() == 'JPEG':
             pil_img.save(buffered, format="JPEG", quality=quality)
        elif format.upper() == 'PNG':
             pil_img.save(buffered, format="PNG") # PNG is lossless
        else:
             raise ValueError(f"Unsupported image format: {format}")

        return buffered.getvalue()

    except Exception as e:
        raise # Re-raise to be handled by the calling method


def rescale_for_model(image, current_fov_nm, target_pixel_size_A=0.25): # 0.2-0.3 seems to be the optimal range
    """
    Rescale to achieve target pixel size while keeping field of view constant
    """

    import cv2

    # Calculate target image size to achieve desired pixel size
    target_size_px = int((current_fov_nm * 10) / target_pixel_size_A)
    
    # Calculate scale factor
    scale_factor = target_size_px / image.shape[0]
    
    print(f"Current: {image.shape[0]}px, Target: {target_size_px}px")
    print(f"Scale factor: {scale_factor:.3f} ({'upsize' if scale_factor > 1 else 'downsize'})")
    
    # Choose interpolation
    if scale_factor > 1.5:
        interpolation = cv2.INTER_CUBIC  # Upscaling
    elif scale_factor < 0.7:
        interpolation = cv2.INTER_AREA   # Downscaling  
    else:
        interpolation = cv2.INTER_LINEAR # Moderate scaling
    
    rescaled_image = cv2.resize(image, (target_size_px, target_size_px), 
                               interpolation=interpolation)
    
    # Verify final pixel size
    final_pixel_size = (current_fov_nm * 10) / target_size_px
    
    return rescaled_image, scale_factor, final_pixel_size


def _2d_gaussian(xy, amplitude, y0, x0, sigma_y, sigma_x, offset):
    """Helper 2D Gaussian function for fitting."""
    from scipy.optimize import curve_fit
    y, x = xy
    y0 = float(y0)
    x0 = float(x0)
    g = offset + amplitude * np.exp(
        -(((y - y0)**2 / (2 * sigma_y**2)) + ((x - x0)**2 / (2 * sigma_x**2)))
    )
    return g.ravel()

def refine_coordinates_gaussian_fit(image_data, coordinates, window_size=7, max_refinement_shift=1.5):
    """
    Refines atomic coordinates to sub-pixel precision using 2D Gaussian fitting.

    Args:
        image_data (np.ndarray): The image data to fit on (e.g., original image).
        coordinates (np.ndarray): Nx2 array of coarse (y, x) coordinates from Locator.
        window_size (int): The size of the fitting window around each coordinate.
        max_refinement_shift (float): The maximum distance (in pixels) the center can shift during fitting.

    Returns:
        np.ndarray: Refined coordinates.
    """
    from scipy.optimize import curve_fit
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
            
            # Define bounds to constrain the fit and prevent it from jumping to another atom.
            # params: amplitude, y0, x0, sigma_y, sigma_x, offset
            lower_bounds = [0, y_int - max_refinement_shift, x_int - max_refinement_shift, 0.5, 0.5, 0]
            upper_bounds = [patch.max() * 1.5, y_int + max_refinement_shift, x_int + max_refinement_shift, half_w, half_w, patch.max()]
            bounds = (lower_bounds, upper_bounds)

            popt, _ = curve_fit(_2d_gaussian, (y_grid, x_grid), patch.ravel(), p0=initial_guess, bounds=bounds)
            y_refined, x_refined = popt[1], popt[2]
            # The bounds in curve_fit should already handle this, but an extra check for patch boundaries is good practice.
            if (y_min <= y_refined < y_max) and (x_min <= x_refined < x_max): # Check if it's within the patch
                 refined_coords.append([y_refined, x_refined])
            else:
                 refined_coords.append([y_int, x_int]) # Revert to original if fit is outside patch
        except (RuntimeError, ValueError):
            refined_coords.append([y_int, x_int])

    return np.array(refined_coords)

def predict_with_ensemble(dir_path, image, thresh=0.8, refine=True, max_refinement_shift=1.5):
    import logging
    logger = logging.getLogger(__name__)

    all_predictions = []
    model_pattern = os.path.join(dir_path, 'atomnet3*.tar')
    logger.info(f"Searching for DCNN model files with pattern: {model_pattern}")
    model_files = glob.glob(model_pattern)

    if not model_files:
        logger.error(f"No model files found in '{dir_path}' matching the pattern 'atomnet3*.tar'.")
        raise FileNotFoundError(f"Could not find any DCNN model files ('atomnet3*.tar') in the specified directory: {dir_path}")

    logger.info(f"Found {len(model_files)} models for ensemble prediction.\n")
    for model_file in model_files:
        logger.debug(f"Loading model: {model_file}")
        model = aoi.load_model(model_file)
        prediction = model.predict(image)[0]
        all_predictions.append(prediction)
    prediction_mean = np.mean(np.stack(all_predictions), axis=0)
    locator_output = aoi.predictors.Locator(thresh=thresh).run(prediction_mean)

    # The locator can return a dict {class: coords} or an array. We handle both.
    if isinstance(locator_output, dict):
        # For single-channel prediction, we expect the key to be 0.
        coarse_coords_with_class = locator_output.get(0)
    else:
        coarse_coords_with_class = locator_output # It's an array or None

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
    """
    Calculates the nearest-neighbor distance for each coordinate.

    Args:
        coordinates (np.ndarray): Nx2 array of (y, x) coordinates of atoms.
        pixel_scale (float): Conversion factor from pixels to real units (e.g., nm/pixel).

    Returns:
        np.ndarray: A 1D array of nearest-neighbor distances for each coordinate,
                    or None if there are not enough coordinates.
    """
    from scipy.spatial import KDTree
    import numpy as np

    if coordinates is None or len(coordinates) < 2:
        print("Warning: Not enough coordinates to compute nearest-neighbor distances.")
        return None

    # Build a KD-tree for efficient nearest-neighbor search
    tree = KDTree(coordinates)

    # Query the KD-tree for the nearest neighbor of each atom (excluding itself)
    distances, _ = tree.query(coordinates, k=2)
    # The nearest-neighbor distance is the second element (first is distance to self, which is 0)
    nearest_neighbor_distances = distances[:, 1] * pixel_scale

    return nearest_neighbor_distances


def download_file_with_gdown(file_id, output_path):
    """
    Downloads a file from Google Drive using gdown.

    Args:
        file_id (str): The ID of the Google Drive file.
        output_path (str): The local path where the file will be saved.
    """

    import gdown

    try:
        # Create directory if it doesn't exist
        parent_dir = os.path.dirname(output_path)
        # If parent_dir is an empty string, it means the file is in the current directory.
        # We only need to create the directory if a path is specified.
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # gdown.download handles the direct download link construction and large file quirks
        gdown.download(id=file_id, output=output_path, quiet=False, fuzzy=True) # fuzzy=True for slightly more relaxed ID matching
        
        print(f"File downloaded to: {output_path}")
        return output_path
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None

def unzip_file(zip_filepath, extract_to_dir):
    """
    Unzips a file to a specified directory.

    Args:
        zip_filepath (str): The path to the zip file.
        extract_to_dir (str): The directory where contents will be extracted.
    """

    import zipfile

    if not os.path.exists(zip_filepath):
        print(f"Error: Zip file not found at {zip_filepath}")
        return False

    print(f"Unzipping '{zip_filepath}' to '{extract_to_dir}'...")
    try:
        os.makedirs(extract_to_dir, exist_ok=True)

        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_dir)
        print(f"Successfully unzipped to: {extract_to_dir}")
        return True
    except zipfile.BadZipFile:
        print(f"Error: '{zip_filepath}' is not a valid zip file or is corrupted.")
        return False
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        return False
