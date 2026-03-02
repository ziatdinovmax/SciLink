"""
SAMVAEWorkflow — Proof-of-concept pipeline: SAM particle segmentation → VAE latent analysis.

Architecture (Principle 5 — Decoupled):
    Orchestrator (this workflow) is the ONLY glue between agents.
    SAMMicroscopyAnalysisAgent and VAEAgent share no domain logic.

Data flow:
    image → SAMMicroscopyAnalysisAgent → particle masks & LLM analysis
                                       → _extract_particle_patches()
                                       → VAEAgent → latent space analysis
                                       → LLM synthesis
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from scilink.agents.exp_agents.sam_microscopy_agent import SAMMicroscopyAnalysisAgent
from scilink.agents.exp_agents.vae_agent import VAEAgent
from scilink.tools.sam import run_sam_analysis


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch extraction utilities
# ---------------------------------------------------------------------------

def _load_image_as_array(image_data: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load image into a 2-D float32 numpy array (grayscale, values in [0, 1]).
    Accepts a file path string or a numpy array.
    """
    if isinstance(image_data, str):
        img = Image.open(image_data).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    arr = np.asarray(image_data, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)  # collapse colour channels
    return arr / arr.max() if arr.max() > 1.0 else arr


def _extract_particle_patches(
    image_array: np.ndarray,
    masks: List[np.ndarray],
    patch_size: int,
    max_patches: int = 512,
) -> np.ndarray:
    """
    Extract square crops centred on each detected particle mask.

    Each mask is a binary array with the same spatial dimensions as
    ``image_array``.  The bounding box of each mask determines the crop
    centre; crops are padded / clipped to exactly ``patch_size x patch_size``.

    Args:
        image_array:  2-D grayscale image (H x W), values in [0, 1].
        masks:        Binary masks returned by ``run_sam_analysis``, one per particle.
        patch_size:   Edge length (pixels) for each square crop.
        max_patches:  Cap on the number of patches extracted.

    Returns:
        Float32 numpy array of shape (N, patch_size, patch_size).
    """
    H, W = image_array.shape
    half = patch_size // 2
    patches: List[np.ndarray] = []

    for mask in masks[:max_patches]:
        # Find bounding box of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            continue
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cy = int((rmin + rmax) / 2)
        cx = int((cmin + cmax) / 2)

        # Extract patch with zero-padding for border particles
        r0, r1 = cy - half, cy + half
        c0, c1 = cx - half, cx + half

        patch = np.zeros((patch_size, patch_size), dtype=np.float32)

        # Clamp to image bounds and compute corresponding patch indices
        ir0, ic0 = max(r0, 0), max(c0, 0)
        ir1, ic1 = min(r1, H), min(c1, W)
        pr0, pc0 = ir0 - r0, ic0 - c0
        pr1, pc1 = pr0 + (ir1 - ir0), pc0 + (ic1 - ic0)

        patch[pr0:pr1, pc0:pc1] = image_array[ir0:ir1, ic0:ic1]
        patches.append(patch)

    if not patches:
        logger.warning("No valid patches extracted from SAM masks.")
        return np.empty((0, patch_size, patch_size), dtype=np.float32)

    return np.stack(patches, axis=0)


def _choose_patch_size(masks: List[np.ndarray]) -> int:
    """
    Choose a square patch size as the next power-of-two above the median
    particle bounding-box diagonal.  Clamped to [16, 64].
    """
    if not masks:
        return 32

    diags = []
    for mask in masks:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            continue
        h = int(np.where(rows)[0][-1]) - int(np.where(rows)[0][0]) + 1
        w = int(np.where(cols)[0][-1]) - int(np.where(cols)[0][0]) + 1
        diags.append(max(h, w))

    if not diags:
        return 32

    target = int(np.median(diags)) + 4  # a little padding
    # Round up to nearest power of two, clamped to [16, 64]
    # shifts the bits of the integer p to the left by 1 position. 
    # This is effectively the same as multiplying p by 2E 
    p = 1
    while p < target:
        p <<= 1
    return int(np.clip(p, 16, 64))


# ---------------------------------------------------------------------------
# SAMVAEWorkflow
# ---------------------------------------------------------------------------

class SAMVAEWorkflow:
    """
    Proof-of-concept orchestration workflow connecting the SAM microscopy
    analysis agent to the VAE latent-space analysis agent.

    Usage::

        workflow = SAMVAEWorkflow(api_key="...", output_dir="sam_vae_output")
        result = workflow.run("path/to/image.tif",
                              system_info={"material": "nanoparticles",
                                           "technique": "TEM"})

    Result keys:
        status               "success" | "error"
        sam_result           Full output dict from SAMMicroscopyAnalysisAgent
        vae_result           Full output dict from VAEAgent
        n_particle_patches   Number of patches fed to the VAE
        patch_size           Pixel edge length of each patch
        unified_analysis     LLM synthesis combining SAM + VAE findings
        output_directory     Path to all saved outputs
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro-preview-06-05",
        base_url: Optional[str] = None,
        output_dir: str = "sam_vae_output",
        sam_settings: Optional[Dict[str, Any]] = None,
        enable_human_feedback: bool = False,
        vae_max_retries: int = 3,
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ) -> None:
        self.output_dir          = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vae_max_retries     = vae_max_retries
        self.enable_human_feedback = enable_human_feedback

        sam_out = str(self.output_dir / "sam_output")
        vae_out = str(self.output_dir / "vae_output")

        # Instantiate agents — they share no domain logic (Principle 5)
        self.sam_agent = SAMMicroscopyAnalysisAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            google_api_key=google_api_key,
            local_model=local_model,
            sam_settings=sam_settings or {},
            enable_human_feedback=enable_human_feedback,
            output_dir=sam_out,
        )

        self.vae_agent = VAEAgent(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            google_api_key=google_api_key,
            local_model=local_model,
            output_dir=vae_out,
            enable_human_feedback=enable_human_feedback,
        )

        # Shared LLM model for synthesis (reuse vae_agent's model)
        self._llm = self.vae_agent.model
        self._gen_cfg = self.vae_agent.generation_config
        self._safety  = self.vae_agent.safety_settings

    # =========================================================================
    # Main entry point
    # =========================================================================

    def run(
        self,
        image_data: Union[str, np.ndarray],
        system_info: Optional[Dict[str, Any]] = None,
        task_description: str = "",
        sam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full SAM → patch extraction → VAE pipeline.

        Args:
            image_data:       File path (str) or numpy array (HxW or HxWxC).
            system_info:      Metadata dict (material, technique, scale, etc.).
            task_description: Optional scientific goal description for the VAE agent.
            sam_params:       Optional SAM parameter overrides (min_area, etc.).

        Returns:
            Standardised result dict (see class docstring for keys).
        """
        system_info = system_info or {}
        logger.info("=" * 70)
        logger.info("SAMVAEWorkflow.run() — starting")
        logger.info("=" * 70)

        # -----------------------------------------------------------------
        # Stage 1 — SAM analysis (LLM interpretation)
        # -----------------------------------------------------------------
        logger.info("[Stage 1] Running SAM microscopy analysis...")
        try:
            sam_result = self.sam_agent.analyze(
                data=image_data,
                system_info=system_info,
            )
        except Exception as e:
            logger.error(f"SAM agent failed: {e}")
            return {"status": "error", "error": f"SAM analysis failed: {e}",
                    "output_directory": str(self.output_dir)}

        if sam_result.get("status") != "success":
            return {"status": "error",
                    "error": f"SAM agent returned error: {sam_result.get('error', 'unknown')}",
                    "sam_result": sam_result,
                    "output_directory": str(self.output_dir)}

        particle_count = sam_result.get("particle_count", 0)
        logger.info(f"  SAM detected {particle_count} particles.")

        if particle_count == 0:
            return {"status": "error",
                    "error": "SAM detected no particles; cannot run VAE.",
                    "sam_result": sam_result,
                    "output_directory": str(self.output_dir)}

        # -----------------------------------------------------------------
        # Stage 2 — Particle mask extraction via SAM tools
        # The workflow (as orchestrator) is permitted to call tools directly
        # to bridge data between agents (Principle 5).
        # -----------------------------------------------------------------
        logger.info("[Stage 2] Extracting particle masks from image...")
        image_array = _load_image_as_array(image_data)

        params = {
            "device":                self.sam_agent.settings.get("device", "auto"),
            "checkpoint_path":       self.sam_agent.settings.get("checkpoint_path"),
            "model_type":            self.sam_agent.settings.get("model_type", "vit_h"),
            "use_clahe":             self.sam_agent.settings.get("use_clahe", False),
            "min_area":              self.sam_agent.settings.get("min_area", 500),
            "max_area":              self.sam_agent.settings.get("max_area", 50000),
            "use_pruning":           self.sam_agent.settings.get("use_pruning", True),
            "pruning_iou_threshold": self.sam_agent.settings.get("pruning_iou_threshold", 0.5),
        }
        if sam_params:
            params.update(sam_params)

        try:
            raw_sam = run_sam_analysis(image_array, params)
            masks: List[np.ndarray] = raw_sam.get("masks", [])
        except Exception as e:
            logger.warning(f"Direct SAM mask extraction failed: {e}. "
                           f"Continuing without per-particle patches.")
            masks = []

        if not masks:
            logger.warning("No masks available. Falling back to uniform grid crops.")
            patches, patch_size = self._grid_crops(image_array)
        else:
            patch_size = _choose_patch_size(masks)
            patches    = _extract_particle_patches(image_array, masks, patch_size)

        n_patches = len(patches)
        logger.info(f"  Extracted {n_patches} patches of size {patch_size}x{patch_size}.")

        if n_patches == 0:
            return {"status": "error",
                    "error": "No particle patches could be extracted.",
                    "sam_result": sam_result,
                    "output_directory": str(self.output_dir)}

        # Conv models in nn_tools add the channel dim internally via ConvBlock.unsqueeze(1),
        # so patches are passed as (N, H, W) — no extra axis needed here.

        # -----------------------------------------------------------------
        # Stage 3 — VAE latent space analysis
        # -----------------------------------------------------------------
        logger.info("[Stage 3] Training VAE on particle patches...")
        vae_system_info = {
            **system_info,
            "source":          "SAM particle segmentation",
            "particle_count":  n_patches,
            "patch_size_px":   patch_size,
        }
        vae_task = (
            task_description
            or f"Learn a latent representation of {n_patches} particle patches "
               f"extracted from a {system_info.get('technique', 'microscopy')} image "
               f"of {system_info.get('material', 'unknown material')}."
        )

        try:
            vae_result = self.vae_agent.analyze(
                data=patches,
                system_info=vae_system_info,
                task_description=vae_task,
                max_retries=self.vae_max_retries,
            )
        except Exception as e:
            logger.error(f"VAE agent failed: {e}")
            return {"status": "error", "error": f"VAE analysis failed: {e}",
                    "sam_result": sam_result,
                    "output_directory": str(self.output_dir)}

        # -----------------------------------------------------------------
        # Stage 4 — LLM synthesis of SAM + VAE findings
        # -----------------------------------------------------------------
        logger.info("[Stage 4] Synthesising SAM and VAE findings...")
        unified_analysis = self._synthesise(sam_result, vae_result, system_info)

        # -----------------------------------------------------------------
        # Save combined results
        # -----------------------------------------------------------------
        combined = {
            "status":              "success",
            "sam_result":          sam_result,
            "vae_result":          vae_result,
            "n_particle_patches":  n_patches,
            "patch_size":          patch_size,
            "unified_analysis":    unified_analysis,
            "output_directory":    str(self.output_dir),
        }
        out_path = self.output_dir / "sam_vae_results.json"
        try:
            with open(out_path, "w") as f:
                json.dump(combined, f, indent=2, default=str)
            logger.info(f"Results saved to {out_path}")
        except Exception as e:
            logger.warning(f"Could not save combined results: {e}")

        logger.info("=" * 70)
        logger.info("SAMVAEWorkflow.run() — complete")
        logger.info("=" * 70)

        return combined

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _grid_crops(self, image_array: np.ndarray,
                    patch_size: int = 32) -> Tuple[np.ndarray, int]:
        """
        Fallback: extract a uniform grid of non-overlapping crops from the image.
        Used when SAM masks are unavailable.
        """
        H, W = image_array.shape
        patches = []
        for r in range(0, H - patch_size + 1, patch_size):
            for c in range(0, W - patch_size + 1, patch_size):
                patches.append(image_array[r:r + patch_size, c:c + patch_size])
        if not patches:
            return np.empty((0, patch_size, patch_size), dtype=np.float32), patch_size
        return np.stack(patches, axis=0).astype(np.float32), patch_size

    def _synthesise(self, sam_result: Dict[str, Any],
                    vae_result: Dict[str, Any],
                    system_info: Dict[str, Any]) -> str:
        """
        Ask the LLM to produce a unified scientific interpretation combining
        SAM particle statistics and VAE latent space findings.
        """
        sam_summary = {
            "particle_count":  sam_result.get("particle_count"),
            "detailed_analysis": (sam_result.get("detailed_analysis") or "")[:500],
        }
        vae_latent = vae_result.get("latent_space_analysis", {})
        vae_summary = {
            "model_type":     vae_result.get("model_type"),
            "active_dims":    vae_latent.get("active_dims"),
            "interpretation": vae_latent.get("interpretation"),
            "final_metrics":  vae_result.get("final_metrics"),
        }

        prompt = f"""You are an expert materials scientist. You have analysed a
microscopy image of {system_info.get('material', 'an unknown material')} using
{system_info.get('technique', 'an unknown technique')}.

Two analyses have been performed:

## 1. SAM Particle Segmentation
{json.dumps(sam_summary, indent=2)}

## 2. VAE Latent Space Analysis (trained on {vae_result.get('n_particle_patches', 'N')} particle patches)
{json.dumps(vae_summary, indent=2)}

Write a concise (3-5 sentence) scientific interpretation that:
1. Summarises what the particle population looks like (from SAM).
2. Describes what the VAE latent space reveals about particle diversity or structure.
3. Highlights any scientifically interesting findings or suggests next experiments.

Respond in JSON: {{"unified_analysis": "<3-5 sentence paragraph>"}}"""

        try:
            response = self._llm.generate_content(
                contents=[prompt],
                generation_config=self._gen_cfg,
                safety_settings=self._safety,
            )
            from scilink.agents.exp_agents.base_agent import LLMAgentMixin
            # Minimal parse — extract text from response object
            if hasattr(response, "text"):
                raw = response.text
            elif hasattr(response, "choices"):
                raw = response.choices[0].message.content
            else:
                raw = str(response)

            import json as _json
            # Try to parse JSON
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = _json.loads(raw)
            return parsed.get("unified_analysis", raw)
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}. Using template summary.")
            n_active = vae_latent.get("active_dims", "?")
            n_total  = len(vae_latent.get("mu_mean", []))
            interp   = vae_latent.get("interpretation", "")
            return (
                f"SAM segmented {sam_result.get('particle_count', 0)} particles. "
                f"The VAE ({vae_result.get('model_type')}) used "
                f"{n_active}/{n_total} active latent dimensions to represent "
                f"the particle population. {interp}"
            )
