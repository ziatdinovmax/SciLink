"""
SAM Analysis Tools with Model Caching Support

This module provides SAM-based particle analysis tools that integrate with
scilink's ParticleAnalyzer (supports both grayscale and RGB input) while adding:
1. Model caching - loads SAM model once and reuses for batch processing
2. Proper statistics calculation with actual area values
3. Shape statistics (circularity, solidity, aspect ratio)
4. Support for spatial calibration (nm/pixel)
"""

import json
import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

# Global model cache to avoid reloading the ~2.5GB model for each image
_SAM_MODEL_CACHE: Dict[str, Any] = {}

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL CACHING FUNCTIONS
# =============================================================================

def get_or_create_sam_model(params: dict) -> Any:
    """
    Get a cached SAM model or create a new one.
    
    Caches models by (checkpoint_path, model_type, device) to avoid
    reloading the ~2.5GB model for each image. This provides ~10x speedup
    for batch processing.
    
    Args:
        params: Dictionary containing:
            - checkpoint_path: Path to SAM weights (optional)
            - model_type: 'vit_h', 'vit_l', or 'vit_b'
            - device: 'auto', 'cpu', or 'cuda'
        
    Returns:
        Loaded SAM model (ParticleAnalyzer instance)
    """
    from scilink.tools.particle_analyzer import ParticleAnalyzer
    
    # Create cache key from model-defining parameters
    cache_key = (
        params.get('checkpoint_path', 'default'),
        params.get('model_type', 'vit_h'),
        params.get('device', 'auto')
    )
    
    if cache_key in _SAM_MODEL_CACHE:
        logger.info("Using cached SAM model")
        return _SAM_MODEL_CACHE[cache_key]
    
    logger.info(f"Loading new SAM model: type={cache_key[1]}, device={cache_key[2]}")

    # Resolve checkpoint path to absolute to avoid re-downloading
    # when scripts run from different working directories
    checkpoint = params.get('checkpoint_path')
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
    else:
        # Default to a stable location rather than CWD-relative ./checkpoints/
        model_type = params.get('model_type', 'vit_h')
        default_dir = Path.home() / ".cache" / "scilink" / "checkpoints"
        default_path = default_dir / f"sam_{model_type}.pth"
        # Also check legacy location
        legacy_path = Path("./checkpoints") / f"sam_{model_type}.pth"
        if default_path.exists():
            checkpoint = str(default_path)
        elif legacy_path.resolve().exists():
            checkpoint = str(legacy_path.resolve())
        else:
            # Will download to stable location
            default_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = str(default_path)

    analyzer = ParticleAnalyzer(
        checkpoint_path=checkpoint,
        model_type=params.get('model_type', 'vit_h'),
        device=params.get('device', 'auto')
    )
    
    _SAM_MODEL_CACHE[cache_key] = analyzer
    logger.info("SAM model loaded and cached successfully")
    return analyzer


def clear_sam_model_cache():
    """Clear all cached SAM models to free memory."""
    global _SAM_MODEL_CACHE
    _SAM_MODEL_CACHE.clear()
    logger.info("SAM model cache cleared")


def get_cached_model_info() -> Dict[str, Any]:
    """Get information about currently cached models."""
    return {
        "num_cached_models": len(_SAM_MODEL_CACHE),
        "cache_keys": list(_SAM_MODEL_CACHE.keys())
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_sam_analysis(
    image_array: np.ndarray,
    params: dict,
    analyzer: Optional[Any] = None
) -> dict:
    """
    Run SAM analysis on an image.
    
    This function wraps atomai's ParticleAnalyzer.analyze() method with
    additional preprocessing, filtering, and result formatting.
    
    Args:
        image_array: Preprocessed image as numpy array (2D grayscale)
        params: Analysis parameters dict containing:
            - checkpoint_path: Path to SAM weights (optional)
            - model_type: 'vit_h', 'vit_l', or 'vit_b'
            - device: 'auto', 'cpu', or 'cuda'
            - use_clahe: Whether to apply CLAHE preprocessing
            - sam_parameters: 'default', 'sensitive', or 'ultra-permissive'
            - min_area: Minimum particle area in pixels
            - max_area: Maximum particle area in pixels
            - use_pruning: Whether to remove overlapping masks
            - pruning_iou_threshold: IoU threshold for pruning
        analyzer: Optional pre-loaded ParticleAnalyzer. If None, will use cache.
                  Pass this parameter to avoid model reloading in batch processing.
        
    Returns:
        Dictionary with analysis results:
            - particles: List of particle info dicts (from ParticleAnalyzer)
            - masks: List of binary mask arrays
            - areas: Area in pixels for each particle
            - total_count: Number of detected particles
            - raw_mask_count: Number of masks before filtering
            - original_image: Preprocessed image used for analysis
            - rgb_image: RGB version of the image
            - parameters: Parameters used for this analysis
    """
    # Get or create analyzer (uses cache if analyzer not provided)
    if analyzer is None:
        analyzer = get_or_create_sam_model(params)
    
    # Run the analysis using atomai's ParticleAnalyzer
    # This handles preprocessing, SAM inference, filtering, and property extraction
    sam_result = analyzer.analyze(image_array, params=params)
    
    # Extract areas from particles for convenient access
    particles = sam_result.get('particles', [])
    areas = [p.get('area', 0) for p in particles]
    masks = [p.get('mask') for p in particles if p.get('mask') is not None]
    
    # Add convenience fields
    sam_result['areas'] = areas
    sam_result['masks'] = masks
    sam_result['parameters'] = params
    
    return sam_result


# =============================================================================
# STATISTICS CALCULATION
# =============================================================================

def calculate_sam_statistics(
    sam_result: dict,
    image_path: str,
    preprocessed_image_shape: tuple,
    nm_per_pixel: Optional[float] = None
) -> dict:
    """
    Calculate comprehensive morphological statistics from SAM results.
    
    Uses atomai's ParticleAnalyzer.particles_to_dataframe() for detailed
    statistics, with additional scaling for physical units.
    
    Args:
        sam_result: Output from run_sam_analysis
        image_path: Path to original image (for metadata)
        preprocessed_image_shape: Shape of the processed image
        nm_per_pixel: Optional spatial calibration (nanometers per pixel)
        
    Returns:
        Dictionary of statistics including:
            - total_particles: Number of detected particles
            - mean_area_pixels/nm2: Mean area
            - std_area_pixels/nm2: Standard deviation of area
            - mean_circularity: Mean circularity (1.0 = perfect circle)
            - mean_aspect_ratio: Mean aspect ratio
            - mean_solidity: Mean solidity
            - Plus calibrated measurements if nm_per_pixel provided
    """
    from scilink.tools.particle_analyzer import ParticleAnalyzer
    
    logger.info("   (Tool Info: Extracting morphological statistics...)")
    
    # Use atomai's built-in DataFrame conversion
    particles_df = ParticleAnalyzer.particles_to_dataframe(sam_result)
    
    current_params = sam_result.get("parameters", {})
    
    # Build statistics dictionary
    if not particles_df.empty:
        # Determine scaling factors
        if nm_per_pixel is not None and nm_per_pixel > 0:
            linear_scale = nm_per_pixel
            area_scale = nm_per_pixel ** 2
            unit_suffix = "nm"
            area_unit_suffix = "nm2"
        else:
            linear_scale = 1.0
            area_scale = 1.0
            unit_suffix = "pixels"
            area_unit_suffix = "pixels"
        
        summary_stats = {
            'total_particles': sam_result.get('total_count', len(particles_df)),
            
            # Area statistics
            f'mean_area_{area_unit_suffix}': float(particles_df['area'].mean()) * area_scale,
            f'std_area_{area_unit_suffix}': float(particles_df['area'].std()) * area_scale,
            f'min_area_{area_unit_suffix}': float(particles_df['area'].min()) * area_scale,
            f'max_area_{area_unit_suffix}': float(particles_df['area'].max()) * area_scale,
            f'median_area_{area_unit_suffix}': float(particles_df['area'].median()) * area_scale,
            
            # Also store pixel values for reference
            'mean_area_pixels': float(particles_df['area'].mean()),
            'std_area_pixels': float(particles_df['area'].std()),
            'min_area_pixels': float(particles_df['area'].min()),
            'max_area_pixels': float(particles_df['area'].max()),
            
            # Shape statistics (dimensionless)
            'mean_circularity': float(particles_df['circularity'].mean()) if 'circularity' in particles_df else None,
            'std_circularity': float(particles_df['circularity'].std()) if 'circularity' in particles_df else None,
            'mean_aspect_ratio': float(particles_df['aspect_ratio'].mean()) if 'aspect_ratio' in particles_df else None,
            'std_aspect_ratio': float(particles_df['aspect_ratio'].std()) if 'aspect_ratio' in particles_df else None,
            'mean_solidity': float(particles_df['solidity'].mean()) if 'solidity' in particles_df else None,
            'std_solidity': float(particles_df['solidity'].std()) if 'solidity' in particles_df else None,
            
            # Equivalent diameter
            f'mean_equiv_diameter_{unit_suffix}': float(particles_df['equiv_diameter'].mean()) * linear_scale if 'equiv_diameter' in particles_df else None,
            f'std_equiv_diameter_{unit_suffix}': float(particles_df['equiv_diameter'].std()) * linear_scale if 'equiv_diameter' in particles_df else None,
            
            # Perimeter
            f'mean_perimeter_{unit_suffix}': float(particles_df['perimeter'].mean()) * linear_scale if 'perimeter' in particles_df else None,
            f'std_perimeter_{unit_suffix}': float(particles_df['perimeter'].std()) * linear_scale if 'perimeter' in particles_df else None,
            
            # Metadata
            'image_path': str(image_path),
            'image_shape': preprocessed_image_shape,
            'nm_per_pixel': nm_per_pixel if nm_per_pixel is not None else "N/A",
            'parameters_used': current_params,
        }
    else:
        # No particles detected
        summary_stats = {
            'total_particles': 0,
            'mean_area_pixels': None,
            'std_area_pixels': None,
            'mean_circularity': None,
            'std_circularity': None,
            'image_path': str(image_path),
            'image_shape': preprocessed_image_shape,
            'nm_per_pixel': nm_per_pixel if nm_per_pixel is not None else "N/A",
            'parameters_used': current_params,
        }
    
    logger.info(f"   (Tool Info: Statistics calculation complete. Final count: {sam_result.get('total_count', 0)} particles.)")
    return summary_stats


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_sam_results(
    sam_result: dict,
    image_array: Optional[np.ndarray] = None,
    show_labels: bool = True,
    show_centroids: bool = True
) -> np.ndarray:
    """
    Create visualization overlay of SAM results.
    
    Uses atomai's ParticleAnalyzer.visualize_particles() method.
    
    Args:
        sam_result: Output from run_sam_analysis
        image_array: Optional background image (uses sam_result's image if None)
        show_labels: Whether to show particle ID labels
        show_centroids: Whether to show particle centroids
        
    Returns:
        RGB visualization array (uint8)
    """
    from scilink.tools.particle_analyzer import ParticleAnalyzer
    
    # Use atomai's visualization
    overlay_image = ParticleAnalyzer.visualize_particles(
        sam_result,
        show_plot=False,
        show_labels=show_labels,
        show_centroids=show_centroids
    )
    
    return overlay_image


def save_sam_visualization(
    overlay_image: np.ndarray,
    stage: str,
    cycle: int,
    particle_count: int,
    params: dict,
    logger_instance: logging.Logger,
    output_dir: str = "sam_analysis_visualizations"
) -> str:
    """
    Save visualization image with metadata in filename.
    
    Args:
        overlay_image: RGB image array to save
        stage: Analysis stage name (e.g., 'initial', 'refined', 'final')
        cycle: Iteration/cycle number
        particle_count: Number of detected particles
        params: Parameters used (saved to separate file)
        logger_instance: Logger for output messages
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved image file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stage}_cycle{cycle:02d}_{particle_count}particles_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save image
        Image.fromarray(overlay_image).save(filepath)
        
        # Save parameters to companion file
        params_filename = f"{stage}_cycle{cycle:02d}_params_{timestamp}.txt"
        params_filepath = os.path.join(output_dir, params_filename)
        with open(params_filepath, 'w') as f:
            f.write(f"Stage: {stage}\n")
            f.write(f"Cycle: {cycle}\n")
            f.write(f"Particle Count: {particle_count}\n")
            f.write(f"Parameters:\n{json.dumps(params, indent=2)}")
        
        logger_instance.info(f"   (Tool Info: 📸 Saved {stage} visualization: {filepath})")
        
        return filepath
        
    except Exception as e:
        logger_instance.error(f"   (Tool Info: Failed to save visualization: {e})")
        return ""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_particles_dataframe(sam_result: dict):
    """
    Convert SAM results to pandas DataFrame.
    
    Convenience wrapper around ParticleAnalyzer.particles_to_dataframe().
    
    Args:
        sam_result: Output from run_sam_analysis
        
    Returns:
        pandas DataFrame with particle properties
    """
    from scilink.tools.particle_analyzer import ParticleAnalyzer
    return ParticleAnalyzer.particles_to_dataframe(sam_result)


def filter_particles_by_property(
    sam_result: dict,
    property_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> dict:
    """
    Filter particles by a specific property.
    
    Args:
        sam_result: Output from run_sam_analysis
        property_name: Name of property to filter on (e.g., 'area', 'circularity')
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        
    Returns:
        New sam_result dict with only particles matching criteria
    """
    particles = sam_result.get('particles', [])
    
    filtered_particles = []
    for p in particles:
        value = p.get(property_name)
        if value is None:
            continue
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        filtered_particles.append(p)
    
    # Create new result dict
    filtered_result = sam_result.copy()
    filtered_result['particles'] = filtered_particles
    filtered_result['total_count'] = len(filtered_particles)
    filtered_result['areas'] = [p.get('area', 0) for p in filtered_particles]
    filtered_result['masks'] = [p.get('mask') for p in filtered_particles if p.get('mask') is not None]
    
    return filtered_result


def merge_sam_results(results: List[dict]) -> dict:
    """
    Merge multiple SAM results into one.
    
    Useful for combining results from multiple images or analysis runs.
    
    Args:
        results: List of sam_result dictionaries
        
    Returns:
        Merged sam_result dict
    """
    if not results:
        return {'particles': [], 'total_count': 0, 'areas': [], 'masks': []}
    
    all_particles = []
    for r in results:
        all_particles.extend(r.get('particles', []))
    
    # Reassign IDs
    for i, p in enumerate(all_particles):
        p['id'] = i + 1
    
    return {
        'particles': all_particles,
        'total_count': len(all_particles),
        'areas': [p.get('area', 0) for p in all_particles],
        'masks': [p.get('mask') for p in all_particles if p.get('mask') is not None],
        'merged_from': len(results)
    }


# =============================================================================
# TOOL SPEC
# =============================================================================

from ._spec import ToolSpec

TOOL_SPEC = ToolSpec(
    name="run_sam_analysis",
    description=(
        "Segment Anything Model (SAM) instance segmentation. Detects individual "
        "object instances even when they touch or overlap."
    ),
    import_line="from scilink.tools.sam import run_sam_analysis",
    signature="run_sam_analysis(image_array, params) -> dict",
    agents=["image_analysis"],
    when_to_use=(
        "2D images where discrete objects touch or overlap (particles, cells, pores, "
        "grains). Accepts a 2D grayscale array or an HxWx3 RGB uint8 array. For non-RGB "
        "multi-channel images, pass a single channel (e.g. image[:, :, 0]). "
        "For objects that do not touch, plain connected-component labeling is usually "
        "sufficient."
    ),
    parameters={
        "image_array": {
            "type": "ndarray",
            "description": "2D grayscale array or HxWx3 RGB uint8 array.",
        },
        "params": {
            "type": "dict",
            "description": (
                "Analysis parameters. Keys: "
                "sam_parameters ('default' | 'sensitive' | 'ultra-permissive', "
                "always start with 'default'), "
                "min_area and max_area (pixel area filters, set from expected object size), "
                "pruning_iou_threshold (float, default 0.5 — lower is stricter, higher keeps "
                "more overlapping masks), "
                "use_clahe (bool, default False — contrast enhancement)."
            ),
        },
    },
    required=["image_array", "params"],
    returns=(
        "dict with 'particles' (list with 'mask' and 'area' per particle), "
        "'total_count' (int), 'masks' (list of binary arrays), 'areas' (list of ints)."
    ),
    example=(
        "result = run_sam_analysis(image_array, params={\n"
        "    'sam_parameters': 'default',\n"
        "    'min_area': 50, 'max_area': 5000,\n"
        "    'pruning_iou_threshold': 0.5,\n"
        "})"
    ),
)
