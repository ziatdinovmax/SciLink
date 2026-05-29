import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from .spectral_unmixer import SpectralUnmixer
from scilink.skills._shared.image_processor import create_multi_abundance_overlays
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import cv2


# Keys that are agent-specific configuration and should NOT be passed to sklearn models
AGENT_METADATA_KEYS_TO_STRIP = [
    # Human feedback / agent workflow settings
    'enable_human_feedback',
    'feedback_depths',
    
    # Preprocessing settings
    'run_preprocessing',
    'output_dir',
    'visualization_dir',
    
    # Auto-component selection settings
    'enabled',
    'auto_components',
    'min_auto_components',
    'max_auto_components',
]


# =============================================================================
# IMAGE RESIZE UTILITY
# =============================================================================

def resize_image_bytes(image_bytes: bytes, max_dim: int = 1500) -> bytes:
    """
    Resize image if any dimension exceeds max_dim.
    
    This is required for LLM APIs (especially Claude via Bedrock) which have
    a 2000px limit for multi-image requests. We use 1500px as a safe default.
    
    Args:
        image_bytes: JPEG/PNG image as bytes
        max_dim: Maximum allowed dimension (width or height). Default 1500.
    
    Returns:
        Resized image bytes (or original if already small enough)
    """
    if not image_bytes:
        return image_bytes
        
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return image_bytes
        
        h, w = img.shape[:2]
        
        # Check if resize is needed
        if max(h, w) <= max_dim:
            return image_bytes
        
        # Calculate new size maintaining aspect ratio
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize using INTER_AREA for downscaling (best quality)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Re-encode as JPEG
        _, buf = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()
        
    except Exception:
        # If anything fails, return original
        return image_bytes


# =============================================================================
# SPECTRAL UNMIXING
# =============================================================================

def run_spectral_unmixing(
    hspy_data: np.ndarray,
    n_components: int,
    settings: dict,
    logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Runs the SpectralUnmixer tool.
    
    Returns:
        tuple: (components, abundance_maps, reconstruction_error)
    """
    try:
        logger.info(f"  (Tool Info: Running SpectralUnmixer with n_components={n_components})")
        
        # Create a copy to avoid mutating the original settings
        tool_kwargs = settings.copy()
        
        # Remove all agent-specific keys that sklearn doesn't understand
        for key in AGENT_METADATA_KEYS_TO_STRIP:
            tool_kwargs.pop(key, None)
        
        # Extract SpectralUnmixer-specific args (passed explicitly below)
        method = tool_kwargs.pop('method', 'nmf')
        normalize = tool_kwargs.pop('normalize', True)
        tool_kwargs.pop('n_components', None)  # Passed as explicit parameter
        
        # Anything remaining in tool_kwargs gets passed to the sklearn model
        unmixer = SpectralUnmixer(
            method=method,
            n_components=n_components,
            normalize=normalize,
            random_state=tool_kwargs.pop('random_state', 42),
            **tool_kwargs
        )
        
        components, abundance_maps = unmixer.fit(hspy_data)

        # Method-aware error metric
        if method == 'nmf':
            error = getattr(unmixer.model, 'reconstruction_err_', 0.0)
        elif method == 'pca':
            # Unexplained variance: 1 - cumulative explained variance ratio
            error = 1.0 - sum(unmixer.model.explained_variance_ratio_)
        elif method == 'ica':
            # FastICA has no built-in reconstruction error and no monotonic
            # elbow trend in n_components — downstream controllers skip the
            # elbow loop for ICA, so this is a placeholder.
            error = 0.0
        else:
            error = 0.0

        return components, abundance_maps, float(error)
        
    except Exception as e:
        logger.error(f"  (Tool Error: Spectral unmixing failed: {e})", exc_info=True)
        raise


# =============================================================================
# ENERGY AXIS UTILITIES
# =============================================================================

def create_axis(
    n_channels: int,
    system_info: dict = None,
    axis_index: int = 2,
) -> tuple[np.ndarray, str, bool]:
    """
    Create a physical axis for a hyperspectral dataset axis.

    Defaults to ``axis_index=2`` (the third / "signal" axis), which preserves
    the historical (x, y, energy) layout. Reads through
    ``metadata_converter.resolve_axis_spec`` so that callers see the same
    behavior whether the metadata uses legacy ``energy_range`` or a generic
    ``axis_spec``.

    Returns ``(axis_values, xlabel, has_axis_info)``. Raises ``ValueError``
    when the requested axis has no physical range (``start`` / ``end``)
    available — same contract as the legacy ``create_energy_axis``.
    """
    # Local import to avoid an agents -> skills -> agents cycle at module load.
    from ....agents.exp_agents.metadata_converter import resolve_axis_spec

    spec = resolve_axis_spec(system_info)
    key = f"axis_{axis_index}"
    if key not in spec:
        raise ValueError(f"axis_index {axis_index} is out of range for a 3D dataset.")

    axis_info = spec[key]
    name = axis_info.get("name", "axis")
    units = axis_info.get("units", "a.u.")

    if "start" not in axis_info or "end" not in axis_info:
        # Preserve the legacy error message for the common energy case so that
        # existing call sites and user-facing messaging are unchanged.
        if name == "energy" and not (system_info and "axis_spec" in system_info):
            raise ValueError(
                "Energy axis information is required for hyperspectral analysis. "
                "Metadata must include 'energy_range' with 'start' and 'end' values "
                "(and optionally 'units')."
            )
        raise ValueError(
            f"Axis {axis_index} ('{name}') is missing physical range. "
            f"axis_spec.{key} must include 'start' and 'end' "
            f"(or legacy energy_range for axis_2)."
        )

    start = axis_info["start"]
    end = axis_info["end"]
    axis_values = np.linspace(start, end, n_channels)
    # Title-case the axis name for the label so "energy" -> "Energy (eV)"
    # and "time" -> "Time (s)" without callers having to remember.
    label_name = name[:1].upper() + name[1:] if name else "Axis"
    xlabel = f"{label_name} ({units})"
    return axis_values, xlabel, True


def create_energy_axis(n_channels: int, system_info: dict = None) -> tuple[np.ndarray, str, bool]:
    """
    Backwards-compatible shim for the legacy energy-axis API.

    Delegates to ``create_axis(n_channels, system_info, axis_index=2)`` so
    existing call sites continue to work unchanged. New callers should use
    ``create_axis`` directly when they need a non-default axis_index.
    """
    return create_axis(n_channels, system_info, axis_index=2)


def convert_energy_to_indices(
    energy_axis: np.ndarray, 
    target_start: float, 
    target_end: float, 
    min_channels: int = 10
) -> tuple[int, int]:
    """
    Calculates array indices from physical energy values.
    """
    start_idx = (np.abs(energy_axis - target_start)).argmin()
    end_idx = (np.abs(energy_axis - target_end)).argmin()
    
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
        
    # Guardrail: Padding
    if end_idx - start_idx < min_channels:
        padding = min_channels // 2
        start_idx = max(0, start_idx - padding)
        end_idx = min(len(energy_axis), end_idx + padding)
        
    return start_idx, end_idx


# =============================================================================
# PLOTTING FUNCTIONS (All return resized images)
# =============================================================================

def create_nmf_summary_plot(
    components: np.ndarray,
    abundance_maps: np.ndarray,
    n_comp: int,
    system_info: dict,
    logger: logging.Logger,
    method_name: str = "NMF"
) -> bytes:
    """
    Create a single summary plot showing all components and abundance maps.
    """
    try:
        n_channels = components.shape[1]
        energy_axis, xlabel, has_energy_info = create_energy_axis(n_channels, system_info)

        fig, axes = plt.subplots(2, n_comp, figsize=(n_comp * 3, 6))

        if n_comp == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_comp):
            # Top row: Component spectra
            axes[0, i].plot(energy_axis, components[i, :], 'b-', linewidth=1.5)
            axes[0, i].set_title(f'{method_name} Component {i+1}', fontsize=10)
            axes[0, i].set_xlabel(xlabel)
            if i == 0:
                axes[0, i].set_ylabel('Intensity')
            axes[0, i].grid(True, alpha=0.3)

            # Bottom row: Abundance maps
            im = axes[1, i].imshow(abundance_maps[..., i], cmap='seismic', aspect='auto')
            axes[1, i].set_title(f'Abundance Map {i+1}', fontsize=10)
            axes[1, i].axis('off')
            fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        title = f'{method_name} Analysis: {n_comp} Components'
        if has_energy_info:
            title += " (Energy Calibrated)"
        fig.suptitle(title, fontsize=14, y=0.95)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return resize_image_bytes(image_bytes)

    except Exception as e:
        logger.error(f"  (Tool Error: Failed to create summary plot for {n_comp} components: {e})")
        plt.close('all')
        return None


def create_elbow_plot(component_range: list[int], errors: list[float], logger: logging.Logger, method_name: str = "NMF") -> bytes | None:
    """Create an elbow plot of reconstruction error vs. number of components."""
    if not component_range or not errors or len(component_range) != len(errors):
        logger.warning("  (Tool Info: Invalid input for creating elbow plot.)")
        return None

    if method_name.upper() == "PCA":
        ylabel = "Unexplained Variance (1 - Cumulative Explained)"
    else:
        ylabel = f"{method_name} Reconstruction Error (Frobenius Norm)"

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(component_range, errors, 'bo-', markersize=6)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{method_name} Reconstruction Error vs. Number of Components (Elbow Plot)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(component_range)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)
        logger.info(f"  (Tool Info: Successfully created {method_name} elbow plot.)")
        return resize_image_bytes(image_bytes)
    except Exception as e:
        logger.error(f"  (Tool Error: Failed to create elbow plot: {e})", exc_info=True)
        plt.close('all')
        return None


def create_component_abundance_pairs(
    components: np.ndarray, 
    abundance_maps: np.ndarray, 
    system_info: dict,
    logger: logging.Logger
) -> list[dict]:
    """
    Create individual component-abundance pair images with consistent y-scaling.
    Returns a list of dictionaries: [{'label': str, 'bytes': bytes}, ...]
    """
    pair_images_list = []
    n_components = components.shape[0]
    
    try:
        n_channels = components.shape[1]
        energy_axis, xlabel, has_energy_info = create_energy_axis(n_channels, system_info)
        
        global_min = np.min(components)
        global_max = np.max(components)
        y_margin = (global_max - global_min) * 0.05
        y_limits = (global_min - y_margin, global_max + y_margin)
        
        logger.info(f"  (Tool Info: Creating {n_components} component-abundance pairs with y-scale: {y_limits})")
        
        for i in range(n_components):
            fig, (ax_spectrum, ax_abundance) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Left plot: Component spectrum
            ax_spectrum.plot(energy_axis, components[i, :], 'b-', linewidth=2)
            ax_spectrum.set_ylim(y_limits)
            ax_spectrum.set_xlabel(xlabel)
            ax_spectrum.set_ylabel('Intensity')
            ax_spectrum.set_title(f'Component {i+1} Spectrum')
            ax_spectrum.grid(True, alpha=0.3)
            
            # Right plot: Abundance map
            im = ax_abundance.imshow(abundance_maps[..., i], cmap='viridis', aspect='equal')
            ax_abundance.set_title(f'Component {i+1} Abundance Map')
            ax_abundance.axis('off')
            fig.colorbar(im, ax=ax_abundance, fraction=0.046, pad=0.04, label='Abundance')

            fig.suptitle(f'Component {i+1} Analysis', fontsize=12, y=0.98)
            fig.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_bytes = resize_image_bytes(buf.getvalue())
            pair_images_list.append({
                "label": f"Component {i+1} Pair (Spectrum + Abundance Map)",
                "bytes": image_bytes
            })
            plt.close(fig)

        return pair_images_list

    except Exception as e:
        logger.error(f"  (Tool Error: Failed to create component-abundance pairs: {e})")
        plt.close('all')
        return []


def create_structure_overlays(
    structure_img_gray: np.ndarray,
    abundance_maps: np.ndarray,
    logger: logging.Logger
) -> bytes | None:
    """
    Wrapper for create_multi_abundance_overlays.
    """
    try:
        logger.info(f"  (Tool Info: Creating abundance overlays for {abundance_maps.shape[2]} components)")
        
        overlay_bytes = create_multi_abundance_overlays(
            structure_image=structure_img_gray,
            abundance_maps=abundance_maps,
            threshold_percentile=85.0, # Show top 15%
            alpha=0.5,
            use_simple_colors=True
        )
        return resize_image_bytes(overlay_bytes)
    except Exception as e:
        logger.warning(f"  (Tool Warning: Failed to create abundance overlays: {e})")
        return None


def compare_component_with_weighted_raw(
    hspy_data: np.ndarray, 
    component_spectrum: np.ndarray, 
    abundance_map: np.ndarray, 
    component_idx: int,
    logger
) -> bytes:
    """
    Calculates the Abundance-Weighted Average Spectrum of the raw data
    and plots it against the NMF component for validation.
    """
    try:
        h, w, e = hspy_data.shape
        
        # Flatten
        flat_data = hspy_data.reshape(-1, e) 
        flat_abundance = abundance_map.ravel() 
        
        # Weighted Average
        total_weight = np.sum(flat_abundance)
        if total_weight < 1e-10:
            return None

        weighted_raw_spectrum = np.dot(flat_abundance, flat_data) / total_weight
        
        # Scale NMF to match raw data max
        scale_factor = np.max(weighted_raw_spectrum) / (np.max(component_spectrum) + 1e-6)
        scaled_nmf = component_spectrum * scale_factor

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(weighted_raw_spectrum, color='black', linewidth=2, alpha=0.8, label='Mean Spectrum')
        ax.plot(scaled_nmf, color='red', linestyle='--', linewidth=1.5, label=f'NMF Comp {component_idx+1} (Model)')
        
        residual = weighted_raw_spectrum - scaled_nmf
        ax.fill_between(range(len(residual)), residual, 0, color='gray', alpha=0.2, label='Residual')

        ax.set_title(f"Validation: Component {component_idx+1} vs. Abundance-Weighted Mean Spectrum")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        buf = BytesIO()
        fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return resize_image_bytes(buf.getvalue())

    except Exception as e:
        logger.error(f"Failed to create weighted comparison plot: {e}")
        plt.close('all')
        return None
    

def create_validated_component_pair(
    hspy_data: np.ndarray, 
    component_spectrum: np.ndarray, 
    abundance_map: np.ndarray, 
    component_idx: int,
    system_info: dict,
    logger
) -> bytes:
    """
    Generates a Split-Panel visualization:
    - Left: Spatial Map
    - Right Top: Spectrum (Mean ± StdDev vs Model)
    - Right Bottom: Residual
    """
    try:
        h, w, e = hspy_data.shape
        energy_axis, xlabel, _ = create_energy_axis(e, system_info)
        
        # --- Data Calculation ---
        flat_data = hspy_data.reshape(-1, e)
        flat_abundance = abundance_map.ravel()
        total_weight = np.sum(flat_abundance)
        
        if total_weight < 1e-10: return None

        weighted_raw_spectrum = np.dot(flat_abundance, flat_data) / total_weight
        variance_sq = np.average((flat_data - weighted_raw_spectrum)**2, axis=0, weights=flat_abundance)
        std_dev = np.sqrt(variance_sq)
        
        scale_factor = np.max(weighted_raw_spectrum) / (np.max(component_spectrum) + 1e-6)
        scaled_nmf = component_spectrum * scale_factor
        residual = weighted_raw_spectrum - scaled_nmf

        # --- PLOTTING SETUP ---
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1.5])
        
        # 1. Spatial Map (Takes up the whole Left column)
        ax_map = fig.add_subplot(gs[:, 0])
        im = ax_map.imshow(abundance_map, cmap='viridis')
        ax_map.set_title(f"Component {component_idx+1} Distribution", fontsize=12, fontweight='bold')
        ax_map.axis('off')
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

        # 2. Main Spectrum (Top Right)
        ax_spec = fig.add_subplot(gs[0, 1])

        # Plot Variance Band (Background)
        ax_spec.fill_between(energy_axis,
                             weighted_raw_spectrum - std_dev,
                             weighted_raw_spectrum + std_dev,
                             color='blue', alpha=0.15, label='Raw Variance (±1σ)')

        # Plot Mean and Model
        ax_spec.plot(energy_axis, weighted_raw_spectrum, color='black', linewidth=2, label='Weighted Mean')
        ax_spec.plot(energy_axis, scaled_nmf, color='red', linestyle='--', linewidth=1.5, label='NMF Model')

        ax_spec.set_title("Validation: Spectrum & Model Fit", fontsize=12, fontweight='bold')
        ax_spec.legend(loc='best', fontsize=9)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.set_ylabel("Intensity")
        plt.setp(ax_spec.get_xticklabels(), visible=False)

        # 3. Residual Plot (Bottom Right)
        ax_res = fig.add_subplot(gs[1, 1], sharex=ax_spec)
        ax_res.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax_res.plot(energy_axis, residual, color='gray', linewidth=1)
        ax_res.fill_between(energy_axis, residual, 0, color='gray', alpha=0.3)
        ax_res.set_ylabel("Residual")
        ax_res.set_xlabel(xlabel)
        ax_res.grid(True, alpha=0.3)

        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return resize_image_bytes(buf.getvalue())

    except Exception as e:
        logger.error(f"Failed to create validated pair: {e}")
        plt.close('all')
        return None
    

def create_validated_component_pair_reconstruction(
    hspy_data: np.ndarray,
    components: np.ndarray,
    abundance_maps: np.ndarray,
    component_idx: int,
    system_info: dict,
    logger: logging.Logger,
    purity_percentile: float = 90.0,
    show_basis_component: bool = True,
    method_name: str = "NMF"
) -> tuple[bytes, dict] | None:
    """
    Validates a decomposition by comparing raw data to reconstruction in high-purity regions.

    This addresses the "all components look the same" problem by:
    1. Masking to high-purity regions (adaptive or top 10% abundance by default)
    2. Comparing raw data vs. full reconstruction (apples-to-apples)
    3. Showing the basis component as reference

    Args:
        hspy_data: Raw hyperspectral data (H, W, E)
        components: ALL decomposition components (n_components, E)
        abundance_maps: ALL abundance maps (H, W, n_components)
        component_idx: Which component to validate (0-indexed)
        system_info: Metadata for axis calibration
        logger: Logger instance
        purity_percentile: Threshold for high-purity masking (default 90 = top 10%)
        show_basis_component: Whether to show the orange reference line
        method_name: Name of the decomposition method for plot labels (default "NMF")

    Returns:
        Tuple of (JPEG bytes, metrics dict), or None if failed.
        Metrics dict contains: rmse, max_error, cosine_similarity,
        basis_cosine_similarity, purity_pixel_percent, residual_autocorrelation
    """
    try:
        h, w, e = hspy_data.shape
        n_components = components.shape[0]
        energy_axis, xlabel, _ = create_energy_axis(e, system_info)
        
        # Extract this component's data
        component_spectrum = components[component_idx]
        abundance_map = abundance_maps[..., component_idx]
        
        # =================================================================
        # STEP 1: COMPUTE FULL NMF RECONSTRUCTION
        # =================================================================
        nmf_reconstruction = np.zeros_like(hspy_data)
        for i in range(n_components):
            nmf_reconstruction += abundance_maps[..., i, np.newaxis] * components[i]
        
        # =================================================================
        # STEP 2: ADAPTIVE HIGH-PURITY THRESHOLD
        # =================================================================
        flat_abundance_all = abundance_map.ravel()
        positive_values = flat_abundance_all[flat_abundance_all > 1e-6]
        
        if len(positive_values) > 0:
            median_val = np.median(positive_values)
            mean_val = np.mean(positive_values)
            
            # Skewness heuristic: if mean >> median, distribution has a long tail
            # (localized feature) — use stricter threshold to isolate the tail
            if mean_val > 2 * median_val:
                threshold = mean_val + np.std(positive_values)
                threshold_method = "adaptive-localized"
                logger.info(
                    f"  Component {component_idx+1}: Adaptive threshold "
                    f"(localized feature): {threshold:.4f}"
                )
            else:
                threshold = np.percentile(abundance_map, purity_percentile)
                threshold_method = "standard-percentile"
                logger.info(
                    f"  Component {component_idx+1}: Standard threshold "
                    f"({purity_percentile}th percentile): {threshold:.4f}"
                )
        else:
            threshold = np.percentile(abundance_map, purity_percentile)
            threshold_method = "standard-percentile"
        
        high_purity_mask_2d = (abundance_map >= threshold)
        
        n_pixels_selected = np.sum(high_purity_mask_2d)
        if n_pixels_selected == 0:
            logger.warning(
                f"Component {component_idx+1}: No pixels exceed threshold. "
                f"Method: {threshold_method}"
            )
            return None
        
        purity_pixel_percent = 100.0 * n_pixels_selected / (h * w)
        logger.info(
            f"  Component {component_idx+1}: Using {n_pixels_selected} pixels "
            f"({purity_pixel_percent:.1f}% of data)"
        )
        
        # =================================================================
        # STEP 3: EXTRACT HIGH-PURITY PIXELS
        # =================================================================
        flat_raw_data = hspy_data.reshape(-1, e)
        flat_nmf_reconstruction = nmf_reconstruction.reshape(-1, e)
        flat_abundance = abundance_map.ravel()
        flat_mask = high_purity_mask_2d.ravel()
        
        high_purity_raw = flat_raw_data[flat_mask]
        high_purity_nmf = flat_nmf_reconstruction[flat_mask]
        high_purity_abundance = flat_abundance[flat_mask]
        
        # =================================================================
        # STEP 4: COMPUTE WEIGHTED MEANS
        # =================================================================
        total_weight = np.sum(high_purity_abundance)
        
        # Ground truth: Weighted mean of RAW data
        weighted_raw_spectrum = np.dot(high_purity_abundance, high_purity_raw) / total_weight
        
        # Model prediction: Weighted mean of NMF RECONSTRUCTION
        weighted_nmf_spectrum = np.dot(high_purity_abundance, high_purity_nmf) / total_weight
        
        # =================================================================
        # STEP 5: COMPUTE VARIANCE
        # =================================================================
        variance_sq = np.average(
            (high_purity_raw - weighted_raw_spectrum)**2, 
            axis=0, 
            weights=high_purity_abundance
        )
        std_dev = np.sqrt(variance_sq)
        
        # =================================================================
        # STEP 6: COMPUTE RESIDUALS
        # =================================================================
        residual = weighted_raw_spectrum - weighted_nmf_spectrum
        
        # =================================================================
        # STEP 7: SCALE BASIS COMPONENT (Area-based normalization)
        # =================================================================
        if show_basis_component:
            raw_area = np.trapezoid(weighted_raw_spectrum)
            basis_area = np.trapezoid(component_spectrum)
            scale_factor = raw_area / (basis_area + 1e-10)
            scaled_basis_component = component_spectrum * scale_factor
        
        # =================================================================
        # STEP 8: COMPUTE QUANTITATIVE METRICS
        # =================================================================
        rmse = float(np.sqrt(np.mean(residual**2)))
        max_error = float(np.max(np.abs(residual)))
        
        # Cosine similarity: measured vs reconstruction
        norm_raw = np.linalg.norm(weighted_raw_spectrum)
        norm_nmf = np.linalg.norm(weighted_nmf_spectrum)
        cosine_similarity = float(
            np.dot(weighted_raw_spectrum, weighted_nmf_spectrum) 
            / (norm_raw * norm_nmf + 1e-10)
        )
        
        # Cosine similarity: measured vs basis component
        if show_basis_component:
            norm_basis = np.linalg.norm(scaled_basis_component)
            basis_cosine_similarity = float(
                np.dot(weighted_raw_spectrum, scaled_basis_component) 
                / (norm_raw * norm_basis + 1e-10)
            )
        else:
            basis_cosine_similarity = None
        
        # Residual autocorrelation (detects structured residuals)
        if len(residual) > 2:
            residual_autocorrelation = float(
                np.corrcoef(residual[:-1], residual[1:])[0, 1]
            )
        else:
            residual_autocorrelation = 0.0
        
        metrics = {
            "rmse": rmse,
            "max_error": max_error,
            "cosine_similarity": cosine_similarity,
            "basis_cosine_similarity": basis_cosine_similarity,
            "purity_pixel_percent": purity_pixel_percent,
            "residual_autocorrelation": residual_autocorrelation,
            "threshold_method": threshold_method,
            "n_pixels_selected": int(n_pixels_selected),
        }
        
        # =================================================================
        # PLOTTING
        # =================================================================
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1.5])
        
        # -----------------------------------------------------------------
        # PANEL 1: SPATIAL MAP WITH HIGH-PURITY CONTOUR
        # -----------------------------------------------------------------
        ax_map = fig.add_subplot(gs[:, 0])
        
        im = ax_map.imshow(abundance_map, cmap='viridis')
        ax_map.set_title(
            f"Component {component_idx+1} Abundance Map\n"
            f"Red Contour = High-Purity Region ({purity_pixel_percent:.1f}% of pixels)",
            fontsize=11, fontweight='bold'
        )
        ax_map.axis('off')
        
        # Overlay high-purity contour
        ax_map.contour(
            high_purity_mask_2d,
            levels=[0.5],
            colors='red',
            linewidths=2.5,
            linestyles='--'
        )

        # Pixel count annotation
        ax_map.text(
            0.02, 0.02,
            f"{n_pixels_selected} pixels\n({purity_pixel_percent:.1f}% of data)",
            transform=ax_map.transAxes,
            fontsize=9,
            color='red',
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label='Abundance')

        # -----------------------------------------------------------------
        # PANEL 2: MAIN SPECTRUM PLOT
        # -----------------------------------------------------------------
        ax_spec = fig.add_subplot(gs[0, 1])

        # 1. Variance band (background)
        ax_spec.fill_between(
            energy_axis, 
            weighted_raw_spectrum - std_dev, 
            weighted_raw_spectrum + std_dev, 
            color='lightblue',
            alpha=0.3,
            label='Natural Variance (±1σ)',
            zorder=1
        )

        # 2. Orange basis component
        if show_basis_component:
            ax_spec.plot(
                energy_axis, 
                scaled_basis_component, 
                color='darkorange',
                linestyle=':', 
                linewidth=4,
                alpha=1.0,
                label=f'{method_name} Basis Component {component_idx+1}',
                zorder=2,
            )

        # 3. Red reconstruction
        ax_spec.plot(
            energy_axis, 
            weighted_nmf_spectrum, 
            color='red', 
            linestyle='--', 
            linewidth=4,
            dashes=(8, 4),
            alpha=1.0,
            label=f'{method_name} Reconstruction',
            zorder=4
        )

        # 4. Black measured (semi-transparent so red/orange show through)
        ax_spec.plot(
            energy_axis, 
            weighted_raw_spectrum, 
            color='black', 
            linewidth=2.5,
            alpha=0.6,
            label='Measured Spectrum',
            zorder=5
        )

        ax_spec.set_title(
            f"High-Purity Region: Measured vs. {method_name} Reconstruction", 
            fontsize=12, 
            fontweight='bold'
        )
        ax_spec.legend(loc='best', fontsize=9, framealpha=0.9)
        ax_spec.grid(True, alpha=0.3)
        ax_spec.set_ylabel("Intensity", fontsize=10)
        plt.setp(ax_spec.get_xticklabels(), visible=False)
        
        # -----------------------------------------------------------------
        # PANEL 3: RESIDUAL PLOT
        # -----------------------------------------------------------------
        ax_res = fig.add_subplot(gs[1, 1], sharex=ax_spec)
        
        ax_res.axhline(0, color='black', linewidth=1, alpha=0.5)
        
        ax_res.plot(energy_axis, residual, color='gray', linewidth=1.5)
        ax_res.fill_between(energy_axis, residual, 0, color='gray', alpha=0.3)
        
        # Statistics annotation
        ax_res.text(
            0.98, 0.95, 
            f"RMSE: {rmse:.2f}\nMax |Error|: {max_error:.2f}",
            transform=ax_res.transAxes,
            fontsize=9, 
            verticalalignment='top', 
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax_res.set_ylabel("Residual\n(Measured - Reconstructed)", fontsize=10)
        ax_res.set_xlabel(xlabel, fontsize=10)
        ax_res.grid(True, alpha=0.3)

        fig.tight_layout()

        # =================================================================
        # SAVE & RETURN
        # =================================================================
        buf = BytesIO()
        fig.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        logger.info(
            f"  Component {component_idx+1}: Validation plot created. "
            f"RMSE={rmse:.3f}, Max Error={max_error:.3f}, "
            f"CosSim={cosine_similarity:.4f}, "
            f"ResidualAutoCorr={residual_autocorrelation:.3f}"
        )
        
        return resize_image_bytes(buf.getvalue()), metrics
        
    except Exception as e:
        logger.error(
            f"Failed to create reconstruction validation plot for "
            f"component {component_idx+1}: {e}",
            exc_info=True
        )
        plt.close('all')
        return None


def create_annotated_heatmap(data_map: np.ndarray, title: str, units: str) -> bytes:
    """
    Creates a clean, publication-quality heatmap.
    Moved from RunDynamicAnalysisController.
    """
    # Slightly larger figure for clarity
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Robust scaling to handle outliers (2nd to 98th percentile)
    vmin = np.nanpercentile(data_map, 2)
    vmax = np.nanpercentile(data_map, 98)
    
    # Plot Data
    im = ax.imshow(data_map, cmap='plasma', vmin=vmin, vmax=vmax, origin='upper')
    
    # Clean Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{units}", rotation=270, labelpad=15, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Title only (Large and bold)
    clean_title = title.replace("_", " ")
    ax.set_title(f"{clean_title}", fontsize=14, fontweight='bold', pad=12)

    # Remove axes ticks for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove the border spine for a modern look
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = BytesIO()
    fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return resize_image_bytes(buf.getvalue())


def create_feature_dashboard(
    data_map: np.ndarray,
    feature_name: str,
    units: str,
    axis_spec: dict | None = None,
) -> bytes:
    """
    Creates a combined dashboard: 2D Heatmap (Left) + Statistical Histogram (Right).

    ``axis_spec`` (from ``resolve_axis_spec``) controls the heatmap-panel
    title and histogram count label: legacy spatial layouts keep the
    "Spatial Map" / "Pixel Count" wording, generic layouts switch to
    axis-name-driven wording. ``axis_spec=None`` preserves legacy text.
    """

    # Decide labels from axis_spec. Default to the legacy spatial wording.
    map_title_prefix = "Spatial Map"
    count_label = "Pixel Count"
    if axis_spec is not None:
        a0 = axis_spec.get("axis_0", {})
        a1 = axis_spec.get("axis_1", {})
        leading_axes_are_spatial = (
            a0.get("kind") == "spatial" and a1.get("kind") == "spatial"
        )
        if not leading_axes_are_spatial:
            n0 = a0.get("name", "axis 0")
            n1 = a1.get("name", "axis 1")
            map_title_prefix = f"{n0.capitalize()}-{n1.capitalize()} Map"
            count_label = "Sample Count"

    # 1. Clean Data for Histogram
    flat_data = data_map.ravel()
    valid_data = flat_data[~np.isnan(flat_data)]

    if len(valid_data) == 0:
        return None

    # 2. Setup Figure (2 Columns)
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1]) # Map is slightly wider

    # --- LEFT PANEL: 2D Heatmap ---
    ax_map = fig.add_subplot(gs[0])

    # Robust scaling (2nd-98th percentile) to ignore hot pixels
    vmin = np.nanpercentile(data_map, 2)
    vmax = np.nanpercentile(data_map, 98)

    im = ax_map.imshow(data_map, cmap='plasma', vmin=vmin, vmax=vmax, origin='upper')
    ax_map.set_title(f"{map_title_prefix}: {feature_name}", fontsize=12, fontweight='bold')
    ax_map.axis('off') # Clean look

    # Colorbar attached to map
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label(units, rotation=270, labelpad=15)

    # --- RIGHT PANEL: Histogram ---
    ax_hist = fig.add_subplot(gs[1])

    # Dynamic binning
    n_bins = min(50, max(15, int(len(valid_data)**0.4)))
    ax_hist.hist(valid_data, bins=n_bins, color='#2c3e50', alpha=0.75, edgecolor='white', linewidth=0.5)

    # Statistics Box
    mu = np.mean(valid_data)
    sigma = np.std(valid_data)
    stats_text = f"Mean: {mu:.2f}\nStd Dev: {sigma:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

    ax_hist.set_xlabel(f"{feature_name} ({units})")
    ax_hist.set_ylabel(count_label)
    ax_hist.set_title("Population Statistics")
    ax_hist.grid(True, linestyle=':', alpha=0.6)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    fig.tight_layout()

    # 3. Save
    buf = BytesIO()
    fig.savefig(buf, format='jpeg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return resize_image_bytes(buf.getvalue())


def create_image_grid(image_bytes_list: list, logger: logging.Logger = None) -> bytes:
    """
    Stitches a list of JPEG bytes into a single grid image using OpenCV.
    """
    if not image_bytes_list:
        return None
        
    try:
        # Decode all images
        images = []
        for b in image_bytes_list:
            nparr = np.frombuffer(b, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        
        if not images:
            return None

        n_imgs = len(images)
        
        # If only one, return it directly (re-encoded to ensure consistency)
        if n_imgs == 1:
            return resize_image_bytes(image_bytes_list[0])

        # Determine grid size (target ~2 columns)
        cols = 2
        rows = (n_imgs + cols - 1) // cols
        
        # Find max dimensions to standardize cells
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        
        # Create blank canvas (White background)
        grid_h = rows * max_h
        grid_w = cols * max_w
        grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8) + 255 
        
        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            
            # Resize current img to fit cell (centering logic)
            h, w = img.shape[:2]
            y_offset = r * max_h + (max_h - h) // 2
            x_offset = c * max_w + (max_w - w) // 2
            
            grid_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
            
        # Encode back to jpeg
        retval, buf = cv2.imencode('.jpg', grid_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return resize_image_bytes(buf.tobytes())

    except Exception as e:
        if logger:
            logger.warning(f"Failed to stitch validation grid: {e}")
        return None


# =============================================================================
# SPATIAL/SPECTRAL MASKING
# =============================================================================

def apply_spatial_mask(
    current_hspy_data: np.ndarray, 
    abundance_maps: np.ndarray, 
    component_index: int, 
    percentile: float = 85.0
) -> np.ndarray:
    """
    Masks hyperspectral data based on an abundance map.
    Uses the *current* iteration's data as the base.
    """
    if abundance_maps is None:
        raise ValueError("Abundance maps are None, cannot apply spatial mask.")
    
    mask_map = abundance_maps[..., component_index]
    
    if mask_map.ndim != 2:
        raise ValueError(f"Abundance map must be 2D, but got shape {mask_map.shape}")
        
    # Resize mask map to match data if needed
    if mask_map.shape != current_hspy_data.shape[:2]:
        mask_map = cv2.resize(mask_map, (current_hspy_data.shape[1], current_hspy_data.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Threshold non-zero pixels to find the mask
    positive_pixels = mask_map[mask_map > 1e-6]
    if positive_pixels.size == 0:
        # No positive pixels found, return original data
        return current_hspy_data 

    threshold_val = np.percentile(positive_pixels, percentile)
    mask_2d = mask_map >= threshold_val
    
    if np.sum(mask_2d) == 0:
        # Mask is empty, return original data
        return current_hspy_data 

    # Apply mask
    masked_data = current_hspy_data.copy()
    masked_data[~mask_2d] = 0 # Zero out pixels *not* in the mask
    return masked_data


def apply_spectral_slice(
    original_hspy_data: np.ndarray, 
    system_info: dict, 
    energy_range: list
) -> tuple[np.ndarray, dict]:
    """
    Slices hyperspectral data based on an energy range.
    Uses the *original* data as the base and returns an updated system_info.
    """
    energy_axis, _, _ = create_energy_axis(original_hspy_data.shape[2], system_info)
        
    if energy_range is None or len(energy_range) != 2:
        raise ValueError(f"Invalid energy_range: {energy_range}")
        
    start_e, end_e = min(energy_range), max(energy_range)
    
    slice_indices = np.where((energy_axis >= start_e) & (energy_axis <= end_e))[0]
    
    if len(slice_indices) == 0:
        raise ValueError(f"No data found in energy range {energy_range}.")
        
    sliced_data = original_hspy_data[..., slice_indices]
    
    # We must also update the system_info to reflect this slice. Update
    # both legacy energy_range (when present) and the generic axis_spec
    # (when present) so downstream callers using either form see the slice.
    new_start = float(energy_axis[slice_indices[0]])
    new_end = float(energy_axis[slice_indices[-1]])

    new_system_info = system_info.copy()
    legacy_units = system_info.get("energy_range", {}).get("units", "unknown")
    new_system_info["energy_range"] = {
        "start": new_start,
        "end": new_end,
        "units": legacy_units,
    }
    if "axis_spec" in system_info and system_info["axis_spec"]:
        # Deep-copy the axis_spec so we don't mutate the caller's dict.
        new_axis_spec = {k: dict(v) if isinstance(v, dict) else v
                         for k, v in system_info["axis_spec"].items()}
        axis_2 = new_axis_spec.get("axis_2") or {}
        if not isinstance(axis_2, dict):
            axis_2 = {}
        axis_2["start"] = new_start
        axis_2["end"] = new_end
        new_axis_spec["axis_2"] = axis_2
        new_system_info["axis_spec"] = new_axis_spec

    return sliced_data, new_system_info


# =============================================================================
# FILE I/O
# =============================================================================

def save_image_bytes(image_bytes: bytes, output_dir: str, filename: str, logger: logging.Logger = None) -> str:
    """
    Helper to save image bytes to disk. Returns the full filepath.
    """
    if not image_bytes:
        return None
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        if logger:
            logger.info(f"📸 Saved image to: {filepath}")
        return filepath
    except Exception as e:
        if logger:
            logger.error(f"Failed to save image {filename}: {e}")
        return None


# =============================================================================
# SNR ESTIMATION & DATA PREPROCESSING
# =============================================================================

def estimate_global_snr(hspy_data: np.ndarray) -> float:
    """
    Estimates SNR using median absolute deviation (MAD) of spectral
    derivatives.  MAD is robust to outlier channels caused by sharp
    spectral features (edges, narrow peaks) that would inflate a
    std-based estimate and make clean data look noisy.

    Returns a float (e.g., 50.0 is clean, 3.0 is noisy).
    """
    h, w, c = hspy_data.shape
    flat_data = hspy_data.reshape(-1, c)

    # 1. Signal strength — median of the global average spectrum.
    #    Median is robust to a small number of extreme-intensity
    #    channels (e.g. one huge peak) that would inflate the mean
    #    and make noisy data look cleaner than it is.
    signal_mean = np.median(flat_data)
    if signal_mean <= 0:
        return 0.0

    # 2. Noise estimate — MAD of channel-to-channel differences.
    #    MAD ≈ 0.6745 * σ for Gaussian noise, so we convert back.
    diffs = np.diff(flat_data, axis=1)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    noise_est = (mad / 0.6745) / np.sqrt(2)

    if noise_est <= 0:
        return 100.0  # Perfect signal

    return float(signal_mean / noise_est)


def get_optimal_analysis_data(hspy_data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Returns the raw data together with an estimated SNR so that
    downstream code generators can decide how to handle noise.

    Previous versions applied PCA denoising here, but that could
    silently remove real spectral features (sharp peaks, fine
    structure) that the custom code was specifically asked to model.
    """
    snr = estimate_global_snr(hspy_data)
    if snr >= 50.0:
        quality = "High"
    elif snr >= 15.0:
        quality = "Medium"
    elif snr >= 5.0:
        quality = "Low"
    else:
        quality = "Very Low"
    note = f"Raw Data ({quality} quality, SNR≈{snr:.1f})"
    return hspy_data, note


def reconstruct_cube(components: np.ndarray, abundance_maps: np.ndarray) -> np.ndarray:
    """Rank-k reconstruction of the data cube from a decomposition.

    Args:
        components: ``(n_components, n_channels)`` spectral signatures.
        abundance_maps: ``(H, W, n_components)`` spatial loadings.

    Returns:
        ``(H, W, n_channels)`` reconstructed cube = ``abundance_maps @ components``.

    This is the decomposition's denoised, rank-k approximation of the data.
    It lives in whatever intensity space the decomposition was fit in — which
    may be normalized/clipped relative to the raw cube — so it is reliable for
    *shape*-based features (peak position, width) but its absolute intensity
    scale should not be trusted for quantification. Offered to the per-pixel
    codegen as an *optional* fit target for noisy features (issue #219); the
    raw cube remains the base input.
    """
    return abundance_maps @ components