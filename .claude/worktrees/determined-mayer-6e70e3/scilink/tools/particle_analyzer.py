"""
ParticleAnalyzer — SAM-based particle segmentation and analysis.

Forked from atomai's ParticleAnalyzer with added support for RGB images.
Grayscale images are still fully supported; RGB images are passed directly
to SAM without the lossy gray-to-RGB triplication.
"""

import os
import urllib.request
import logging
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class ParticleAnalyzer:
    """
    End-to-end particle segmentation and analysis using the Segment Anything
    Model (SAM).

    Supports both 2D grayscale and 3-channel RGB input images.

    Example::

        analyzer = ParticleAnalyzer(model_type="vit_h")
        result = analyzer.analyze(image)      # image: 2D or HxWx3
        df = ParticleAnalyzer.particles_to_dataframe(result)
    """

    # SAM checkpoint URLs
    _MODEL_URLS = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    # SAM parameter presets
    _SAM_PRESETS: Dict[str, dict] = {
        "default": {},
        "sensitive": {
            "points_per_side": 96,
            "pred_iou_thresh": 0.80,
            "stability_score_thresh": 0.85,
        },
        "ultra-permissive": {
            "points_per_side": 96,
            "pred_iou_thresh": 0.60,
            "stability_score_thresh": 0.70,
        },
    }

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "auto",
    ):
        logger.info("Initializing ParticleAnalyzer...")
        self.device = self._resolve_device(device)
        final_path = self._ensure_checkpoint(checkpoint_path, model_type)
        self.sam_model = self._load_model(final_path, model_type)
        logger.info(f"SAM model loaded on device: {self.device}")

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @classmethod
    def _ensure_checkpoint(cls, checkpoint_path: Optional[str], model_type: str) -> str:
        if checkpoint_path is None:
            ckpt_dir = os.path.join(os.path.expanduser("~"), ".cache", "scilink", "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            checkpoint_path = os.path.join(ckpt_dir, f"sam_{model_type}.pth")

        if not os.path.exists(checkpoint_path):
            url = cls._MODEL_URLS.get(model_type)
            if url is None:
                raise ValueError(f"Unknown model type: '{model_type}'. Cannot download.")
            logger.info(f"Downloading SAM checkpoint for '{model_type}' ...")
            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info(f"Saved to '{checkpoint_path}'.")

        return checkpoint_path

    def _load_model(self, checkpoint_path: str, model_type: str):
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            raise ImportError(
                "The 'segment-anything' package is required.\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return sam

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, image_array: np.ndarray, params: Optional[dict] = None) -> dict:
        """
        Run the full analysis pipeline.

        Args:
            image_array: 2D grayscale **or** HxWx3 RGB uint8/float image.
            params: Analysis parameters (uses defaults when *None*).

        Returns:
            Dict with ``particles``, ``original_image``, ``rgb_image``,
            ``total_count``.
        """
        if params is None:
            params = {
                "use_clahe": False,
                "sam_parameters": "default",
                "min_area": 500,
                "max_area": 50000,
                "use_pruning": False,
                "pruning_iou_threshold": 0.5,
            }

        is_rgb = image_array.ndim == 3 and image_array.shape[2] == 3

        # 1. Pre-process
        processed, image_rgb = self._preprocess_image(
            image_array,
            use_clahe=params.get("use_clahe", False),
            is_rgb=is_rgb,
        )

        # 2. SAM inference
        all_masks = self._run_sam(image_rgb, params.get("sam_parameters", "default"))
        logger.info(f"Generated {len(all_masks)} raw masks.")

        # 3. Filter & prune
        final_masks = self._filter_and_prune(all_masks, params)
        logger.info(f"Kept {len(final_masks)} masks after filtering/pruning.")

        # 4. Extract per-particle properties
        # For intensity stats use a single-channel representation
        intensity_image = (
            cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY) if is_rgb else processed
        )

        particles: List[dict] = []
        for i, mask_info in enumerate(final_masks):
            props = self._extract_particle_properties(mask_info, intensity_image, i + 1)
            if is_rgb:
                props["mean_color_rgb"] = self._mean_color(mask_info, processed)
            particles.append(props)

        # Sort by area (largest first) and reassign IDs
        particles.sort(key=lambda p: p["area"], reverse=True)
        for i, p in enumerate(particles):
            p["id"] = i + 1

        return {
            "particles": particles,
            "original_image": processed,
            "rgb_image": image_rgb,
            "total_count": len(particles),
        }

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_to_uint8(channel: np.ndarray) -> np.ndarray:
        if channel.dtype == np.uint8:
            return channel
        if channel.max() <= 1.0 and channel.min() >= 0.0:
            return (channel * 255).astype(np.uint8)
        lo, hi = channel.min(), channel.max()
        if hi > lo:
            return ((channel - lo) / (hi - lo) * 255).astype(np.uint8)
        return np.zeros_like(channel, dtype=np.uint8)

    def _preprocess_image(
        self,
        image_array: np.ndarray,
        use_clahe: bool,
        is_rgb: bool,
    ) -> tuple:
        """
        Returns ``(processed, image_rgb)`` where *processed* keeps the
        original channel layout and *image_rgb* is the 3-channel array
        fed to SAM.
        """
        if is_rgb:
            # Normalize each channel independently
            processed = np.stack(
                [self._normalize_to_uint8(image_array[:, :, c]) for c in range(3)],
                axis=2,
            )
            if use_clahe:
                logger.info("Applying CLAHE (per-channel on RGB)...")
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = np.stack(
                    [clahe.apply(processed[:, :, c]) for c in range(3)], axis=2
                )
            image_rgb = processed  # already 3-channel
        else:
            processed = self._normalize_to_uint8(image_array)
            if use_clahe:
                logger.info("Applying CLAHE...")
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
            image_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        return processed, image_rgb

    # ------------------------------------------------------------------
    # SAM inference
    # ------------------------------------------------------------------

    def _run_sam(self, image_rgb: np.ndarray, preset_name: str) -> list:
        from segment_anything import SamAutomaticMaskGenerator

        sam_params = self._SAM_PRESETS.get(preset_name, {})
        logger.info(f"Running SAM with preset: '{preset_name}'")
        generator = SamAutomaticMaskGenerator(self.sam_model, **sam_params)
        return generator.generate(image_rgb)

    # ------------------------------------------------------------------
    # Filtering / pruning
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_and_prune(masks: list, params: dict) -> list:
        min_area = params.get("min_area", 0)
        max_area = params.get("max_area", float("inf"))
        filtered = [m for m in masks if min_area <= m["area"] <= max_area]

        if params.get("use_pruning", False):
            logger.info("Applying shape-based pruning...")
            iou_thresh = params.get("pruning_iou_threshold", 0.5)
            return ParticleAnalyzer._prune_by_shape_and_iou(filtered, iou_thresh)
        return filtered

    @staticmethod
    def _prune_by_shape_and_iou(masks: list, iou_threshold: float) -> list:
        if not masks:
            return []
        for m in masks:
            m["solidity"] = ParticleAnalyzer._calculate_solidity(m)
            m["score"] = m["area"] * (m["solidity"] ** 2)
        sorted_masks = sorted(masks, key=lambda x: x["score"], reverse=True)
        kept: list = []
        for mask in sorted_masks:
            if not any(
                ParticleAnalyzer._calculate_iou(mask, k) > iou_threshold for k in kept
            ):
                kept.append(mask)
        return kept

    @staticmethod
    def _calculate_iou(m1: dict, m2: dict) -> float:
        b1, b2 = m1["bbox"], m2["bbox"]
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[0] + b1[2], b2[0] + b2[2])
        y_bottom = min(b1[1] + b1[3], b2[1] + b2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        inter = (x_right - x_left) * (y_bottom - y_top)
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    # Property extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_perimeter(binary_mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return cv2.arcLength(contours[0], True) if contours else 0.0

    @staticmethod
    def _calculate_solidity(mask: dict) -> float:
        binary = mask["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        return area / hull_area if hull_area > 0 else 0.0

    @staticmethod
    def _extract_particle_properties(
        mask: dict, intensity_image: np.ndarray, particle_id: int
    ) -> dict:
        binary = mask["segmentation"]
        area = mask["area"]
        y, x = np.where(binary)
        centroid = (float(np.mean(x)), float(np.mean(y)))
        pixels = intensity_image[binary]
        perimeter = ParticleAnalyzer._calculate_perimeter(binary)
        bbox = mask["bbox"]

        return {
            "id": particle_id,
            "area": area,
            "centroid": centroid,
            "bbox": bbox,
            "mean_intensity": float(np.mean(pixels)),
            "std_intensity": float(np.std(pixels)),
            "min_intensity": float(np.min(pixels)),
            "max_intensity": float(np.max(pixels)),
            "perimeter": perimeter,
            "circularity": 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0,
            "equiv_diameter": 2 * np.sqrt(area / np.pi),
            "aspect_ratio": bbox[3] / bbox[2] if bbox[2] > 0 else 1.0,
            "solidity": mask.get("solidity", ParticleAnalyzer._calculate_solidity(mask)),
            "mask": binary,
        }

    @staticmethod
    def _mean_color(mask: dict, rgb_image: np.ndarray) -> tuple:
        """Mean RGB color inside the particle mask."""
        binary = mask["segmentation"]
        pixels = rgb_image[binary]  # shape (N, 3)
        return tuple(float(v) for v in np.mean(pixels, axis=0))

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def particles_to_dataframe(result: dict) -> pd.DataFrame:
        """Convert the ``particles`` list to a DataFrame (excludes mask arrays)."""
        particles = result.get("particles", [])
        if not particles:
            return pd.DataFrame()
        rows = []
        for p in particles:
            row = {k: v for k, v in p.items() if k != "mask"}
            row["centroid_x"], row["centroid_y"] = p["centroid"]
            row["bbox_x"], row["bbox_y"], row["bbox_width"], row["bbox_height"] = p["bbox"]
            del row["centroid"], row["bbox"]
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def visualize_particles(
        result: dict,
        original_image_for_plot: Optional[np.ndarray] = None,
        show_plot: bool = False,
        show_labels: bool = True,
        show_centroids: bool = True,
    ) -> np.ndarray:
        """Draw particle contours / centroids / labels on the RGB image."""
        import matplotlib.pyplot as plt

        overlay = result["rgb_image"].copy()
        for particle in result.get("particles", []):
            contours, _ = cv2.findContours(
                particle["mask"].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
            cx, cy = int(particle["centroid"][0]), int(particle["centroid"][1])
            if show_centroids:
                cv2.circle(overlay, (cx, cy), 5, (0, 255, 0), -1)
            if show_labels:
                cv2.putText(
                    overlay,
                    str(particle["id"]),
                    (cx + 5, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            display = (
                original_image_for_plot
                if original_image_for_plot is not None
                else result["original_image"]
            )
            cmap = "gray" if display.ndim == 2 else None
            axes[0].imshow(display, cmap=cmap)
            axes[0].set_title("Original Input")
            axes[1].imshow(overlay)
            axes[1].set_title(f"Detected Particles (n={result['total_count']})")
            for ax in axes:
                ax.set_axis_off()
            plt.tight_layout()
            plt.show()

        return overlay
