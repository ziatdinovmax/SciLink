import cv2
from PIL import Image
import numpy as np
from io import BytesIO


MAX_IMG_DIM = 1024

def load_image(image_path):
    """Load an image from file."""
    try:
        # Try standard image loading
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

def preprocess_image(image, max_dim=MAX_IMG_DIM):
    """
    Preprocess microscopy image for better analysis and ensure it's within model context limits.
    """
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
    return denoised


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