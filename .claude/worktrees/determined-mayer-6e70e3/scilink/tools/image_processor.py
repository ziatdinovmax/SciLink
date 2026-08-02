import cv2
from PIL import Image
import numpy as np
from io import BytesIO
import os
import logging
import matplotlib.pyplot as plt

MAX_IMG_DIM = 1024
logger = logging.getLogger(__name__)


def load_image(image_path):
    """Load an image from file (PNG, JPG, TIF, .npy, or .h5/.hdf5)."""
    try:
        _, ext = os.path.splitext(image_path)
        ext = ext.lower()

        if ext == '.npy' or ext in ('.h5', '.hdf5'):
            if ext == '.npy':
                img_array = np.load(image_path)
            else:
                from .hdf5_utils import load_hdf5_signal
                img_array = load_hdf5_signal(image_path)
            if img_array.dtype == np.uint8:
                return img_array
            else:
                # Normalize float arrays to uint8
                float_array = img_array.astype(np.float64)
                min_val, max_val = np.min(float_array), np.max(float_array)
                if max_val - min_val > 1e-6:
                    normalized_array = (float_array - min_val) / (max_val - min_val)
                else:
                    normalized_array = np.zeros_like(float_array)
                uint8_array = (normalized_array * 255).astype(np.uint8)
                return uint8_array
        else:
            # Standard image loading
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
            # Convert OpenCV's BGR to standard RGB
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

def preprocess_image(image: np.ndarray, max_dim: int = MAX_IMG_DIM) -> tuple[np.ndarray, float]:
    """
    Preprocess microscopy image: resize, grayscale, CLAHE, denoise.
    Returns the preprocessed image and the scaling factor.
    """
    scale_factor = 1.0
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        scale_factor = h / new_h
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Image resized from {w}x{h} to {new_w}x{new_h}")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return denoised, scale_factor

def convert_numpy_to_jpeg_bytes(image_array: np.ndarray, quality: int = 85) -> bytes:
    """Converts a NumPy array into compressed JPEG bytes."""
    try:
        pil_img = Image.fromarray(image_array)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=quality)
        return buffered.getvalue()
    except Exception as e:
        print(f"Error converting NumPy array to JPEG bytes: {e}")
        raise

def normalize_and_convert_to_image_bytes(array: np.ndarray, mode='L', format='JPEG', quality=85, log_scale=False) -> bytes:
    """Normalizes a 2D numpy array and converts it to image bytes."""
    if array.ndim == 3 and array.shape[0] == 1:
        array = np.squeeze(array, axis=0)
    if array.ndim != 2:
         raise ValueError(f"Input array must be 2D, but got shape {array.shape}")

    try:
        processed_array = array.copy().astype(np.float32)
        if log_scale:
            processed_array = np.log1p(processed_array) # log1p for log(1+x)

        # Handle potential NaN/Inf values
        if not np.all(np.isfinite(processed_array)):
             max_finite = np.max(processed_array[np.isfinite(processed_array)]) if np.any(np.isfinite(processed_array)) else 1.0
             min_finite = np.min(processed_array[np.isfinite(processed_array)]) if np.any(np.isfinite(processed_array)) else 0.0
             processed_array = np.nan_to_num(processed_array, nan=min_finite, posinf=max_finite, neginf=min_finite)

        # Normalize to 0-1 range
        min_val, max_val = np.min(processed_array), np.max(processed_array)
        if max_val - min_val > 1e-6: # Avoid division by zero
            normalized_array = (processed_array - min_val) / (max_val - min_val)
        else:
            normalized_array = np.zeros_like(processed_array)
            
        uint8_array = (normalized_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(uint8_array, mode=mode)
        
        buffered = BytesIO()
        pil_img.save(buffered, format=format.upper(), quality=quality)
        return buffered.getvalue()
    except Exception as e:
        raise

def calculate_global_fft(image_array: np.ndarray, save_path: str | None = None) -> np.ndarray:
    """
    Calculates the 2D Fast Fourier Transform of an image and returns
    a visualizable (log-scaled, centered) magnitude array.
    
    Optionally saves a plot of the FFT to 'save_path'.
    """
    try:
        # 1. Take the FFT
        f_transform = np.fft.fft2(image_array)
        
        # 2. Shift the zero-frequency component to the center
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        # 3. Get the magnitude (log scale for visualization)
        magnitude_spectrum = np.log1p(np.abs(f_transform_shifted))
        
        # 4. Save the plot if a path is provided
        if save_path:
            try:
                # We normalize the log-scaled image for a good colormap
                viz_bytes = normalize_and_convert_to_image_bytes(magnitude_spectrum, log_scale=False)
                img_to_plot = np.array(Image.open(BytesIO(viz_bytes)))
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(img_to_plot, cmap='inferno')
                ax.set_title("Global FFT (Log Magnitude)")
                ax.axis('off')
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"   (Tool Info: ✅ Saved Global FFT plot to: {save_path})")
                
            except Exception as plot_e:
                # Don't fail the calculation if plotting fails
                logger.warning(f"   (Tool Info: Global FFT calculated, but failed to save plot: {plot_e})")

        return magnitude_spectrum
        
    except Exception as e:
        logger.error(f"   (Tool Info: ❌ Global FFT calculation failed: {e})")
        raise # Re-raise the exception for the controller to catch


def create_multi_abundance_overlays(structure_image: np.ndarray,
                                    abundance_maps: np.ndarray,
                                    threshold_percentile: float = 80.0,
                                    alpha: float = 0.6,
                                    use_simple_colors: bool = True) -> bytes:
    """
    Create overlays for all NMF components in a single image for LLM analysis.
    
    Args:
        structure_image: 2D grayscale structure image
        abundance_maps: 3D array (height, width, n_components)
        threshold_percentile: Show pixels above this percentile
        alpha: Transparency of overlays
        use_simple_colors: If True, use solid colors; if False, use intensity gradients
    
    Returns:
        Image bytes showing structural image + all component overlays
    """
    n_components = abundance_maps.shape[2]
    
    if use_simple_colors:
        # Simple solid colors - easier to distinguish, unlimited components
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        # Generate more colors if needed
        while len(colors) < n_components:
            colors.extend(['darkred', 'darkblue', 'darkgreen', 'indigo', 'brown', 'pink'])
    else:
        # Traditional colormaps with intensity gradients
        colormaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'plasma', 'viridis', 'inferno']
    
    # Calculate grid layout: +1 for original structure image
    total_plots = n_components + 1
    cols = min(4, total_plots)  # Max 4 columns for readability
    rows = (total_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Normalize structure image
    struct_norm = (structure_image - structure_image.min()) / (structure_image.max() - structure_image.min())
    
    # Plot 1: Original structural image
    axes[0].imshow(struct_norm, cmap='gray', aspect='equal')
    axes[0].set_title('Original Structure\n(Reference)', fontweight='bold', fontsize=14)
    axes[0].axis('off')
    
    # Plot 2+: Component overlays
    for i in range(n_components):
        ax_idx = i + 1  # Offset by 1 for the structure image
        abundance_map = abundance_maps[..., i]
        
        # Resize if needed
        if structure_image.shape != abundance_map.shape:
            abundance_map = cv2.resize(abundance_map, 
                                     (structure_image.shape[1], structure_image.shape[0]))
        
        # Threshold
        threshold = np.percentile(abundance_map, threshold_percentile)
        mask = abundance_map >= threshold
        
        # Create overlay
        axes[ax_idx].imshow(struct_norm, cmap='gray', aspect='equal')
        
        if np.any(mask):
            if use_simple_colors:
                # Simple solid color overlay - just show the mask
                color_array = np.zeros((*mask.shape, 4))  # RGBA
                if i < len(colors):
                    from matplotlib.colors import to_rgba
                    rgba = to_rgba(colors[i])
                    color_array[mask] = rgba
                    color_array[mask, 3] = alpha  # Set alpha
                    axes[ax_idx].imshow(color_array, aspect='equal')
            else:
                # Traditional intensity-based overlay
                if abundance_map.max() > abundance_map.min():
                    abund_norm = (abundance_map - abundance_map.min()) / (abundance_map.max() - abundance_map.min())
                else:
                    abund_norm = abundance_map
                
                overlay_data = np.where(mask, abund_norm, np.nan)
                axes[ax_idx].imshow(overlay_data, cmap=colormaps[i % len(colormaps)], 
                                 alpha=alpha, aspect='equal')
        
        # Calculate coverage
        coverage = np.sum(mask) / mask.size * 100
        color_name = colors[i] if use_simple_colors and i < len(colors) else f"comp{i+1}"
        axes[ax_idx].set_title(f'Component {i+1}\n({color_name}, {coverage:.1f}% coverage)', fontsize=14)
        axes[ax_idx].axis('off')
    
    # Hide unused subplots
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert to bytes
    buf = BytesIO()
    plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.getvalue()