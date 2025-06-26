import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import os


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