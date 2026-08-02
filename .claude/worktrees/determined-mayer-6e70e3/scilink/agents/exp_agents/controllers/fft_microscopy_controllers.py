"""
FFT Microscopy Analysis Controllers

This module contains unified controllers that handle both single image (n=1)
and batch (n>1) analysis identically. The key principle is:

    Single image = Batch of 1

All controllers adapt their behavior based on state["is_single_image"]
and state["num_images"], but use the same code paths.
"""

import subprocess
import json
import logging
import io
import os
import base64
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Dict
import numpy as np

from PIL import Image

from ..instruct import (
    FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS,
    SINGLE_IMAGE_ANALYSIS_INSTRUCTIONS,
    SERIES_ANALYSIS_INSTRUCTIONS,
)

from ....tools.image_processor import (
    load_image,
    preprocess_image,
    convert_numpy_to_jpeg_bytes,
    normalize_and_convert_to_image_bytes,
    calculate_global_fft
)
from ....tools.fft_nmf import SlidingFFTNMF


# ============================================================================
# INITIAL FFT ANALYSIS CONTROLLER
# ============================================================================

class InitialFFTAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Analyzes the first frame with LLM-guided parameters.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if state.get("locked_params"):
            self.logger.info("📋 Preset parameters provided, skipping initial analysis.")
            return state
        
        is_single = state.get("is_single_image", False)
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({state.get('num_images', 1)} images)"
        self.logger.info(f"\n\n🔬 --- INITIAL FFT ANALYSIS ({mode_str}) --- 🔬\n")
        
        # Get LLM parameter suggestions
        self.logger.info("🧠 LLM Step: Reasoning about FFT/NMF parameters...")
        
        image_blob = state["image_blob"]
        system_info = state.get("system_info", {})
        
        prompt_parts = [FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)
        if system_info:
            prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(system_info, indent=2)}")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            llm_params = json.loads(response.text)
            state["llm_params"] = llm_params
            state["current_params"] = llm_params
            
            print("\n" + "="*60)
            print("🧠 LLM REASONING")
            print(f"   Explanation: {llm_params.get('explanation', 'N/A')}")
            print(f"   Params: window_size_nm={llm_params.get('window_size_nm')}, n_components={llm_params.get('n_components')}")
            print("="*60 + "\n")
            
        except Exception as e:
            self.logger.error(f"❌ LLM parameter estimation failed: {e}")
            llm_params = {"window_size_nm": 10.0, "n_components": 4, "explanation": "Using defaults"}
            state["llm_params"] = llm_params
            state["current_params"] = llm_params
        
        # Run Global FFT
        self.logger.info("🛠️ Running Global FFT...")
        try:
            image_array = state["preprocessed_image_array"]
            output_dir = self.settings.get("visualization_dir", ".")
            
            base_name = os.path.splitext(os.path.basename(str(state.get("image_path", "image"))))[0]
            safe_name = "".join(c if c.isalnum() else "_" for c in base_name)
            filepath = os.path.join(output_dir, f"{safe_name}_global_fft.png")
            
            global_fft_image = calculate_global_fft(image_array, save_path=filepath)
            state["global_fft_image"] = global_fft_image
            self.logger.info("✅ Global FFT complete.")
        except Exception as e:
            self.logger.error(f"❌ Global FFT failed: {e}")
            state["global_fft_image"] = None
        
        # Run FFT/NMF
        self.logger.info("🛠️ Running Sliding FFT + NMF...")
        
        ws_nm = llm_params.get("window_size_nm")
        nc = llm_params.get("n_components", 4)
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        if ws_nm and nm_per_pixel and nm_per_pixel > 0:
            ws_pixels = int(round(ws_nm / nm_per_pixel))
            good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
            ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        else:
            ws_pixels = 64
        
        step = max(1, ws_pixels // 4)
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels, window_size_y=ws_pixels,
                window_step_x=step, window_step_y=step, components=nc
            )
            
            image_array = state["preprocessed_image_array"]
            components, abundances = analyzer.analyze(image_array, output_dir=None)
            
            state["fft_components"] = components
            state["fft_abundances"] = abundances
            state["summary_stats"] = self._compute_stats(components, abundances)
            
            self.logger.info(f"✅ FFT/NMF complete. {nc} components extracted.")
            
        except Exception as e:
            self.logger.error(f"❌ FFT/NMF failed: {e}")
            state["fft_components"] = None
            state["fft_abundances"] = None
        
        state["first_frame_results"] = {
            "components": state.get("fft_components"),
            "abundances": state.get("fft_abundances"),
            "llm_params": llm_params
        }
        
        return state
    
    def _compute_stats(self, components: np.ndarray, abundances: np.ndarray) -> dict:
        if components is None or abundances is None:
            return {}
        
        stats = {"n_components": components.shape[0], "components": []}
        
        for i in range(components.shape[0]):
            abun = abundances[i]
            stats["components"].append({
                "index": i + 1,
                "abundance_mean": float(np.mean(abun)),
                "abundance_std": float(np.std(abun)),
                "spatial_coverage": float(np.sum(abun > np.mean(abun)) / abun.size)
            })
        
        return stats


# ============================================================================
# HUMAN FEEDBACK REFINEMENT CONTROLLER
# ============================================================================

class HumanFeedbackRefinementController:
    """
    [👤 Human Step]
    Facilitates human-in-the-loop parameter refinement for the first image.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.max_refinement_iterations = settings.get('max_feedback_iterations', 3)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if not state.get('enable_human_feedback', False):
            self.logger.info("Human feedback disabled. Using current parameters.")
            state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
            return state
        
        if state.get("preset_params"):
            self.logger.info("Preset parameters provided. Skipping feedback loop.")
            state["locked_params"] = state.get("preset_params")
            return state
        
        is_single = state.get("is_single_image", False)
        mode_str = "SINGLE IMAGE" if is_single else "BATCH"
        self.logger.info(f"\n\n👤 --- HUMAN FEEDBACK REFINEMENT ({mode_str}) --- 👤\n")
        
        iteration = 0
        while iteration < self.max_refinement_iterations:
            iteration += 1
            
            self._display_analysis_for_review(state, iteration)
            feedback = self._collect_human_feedback(state)
            
            if feedback["action"] == "accept":
                self.logger.info("✅ User accepted current results.")
                state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
                break

            elif feedback["action"] == "modify" and feedback.get("params"):
                state = self._rerun_analysis(state, feedback["params"])
        
        if iteration >= self.max_refinement_iterations:
            self.logger.warning(f"⚠️ Max iterations reached. Using current parameters.")
            state["locked_params"] = state.get("llm_params", state.get("current_params", {}))
        
        return state
    
    def _display_analysis_for_review(self, state: dict, iteration: int) -> None:
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        llm_params = state.get("llm_params", {})
        is_single = state.get("is_single_image", False)
        
        review_viz_path = self.output_dir / f"review_iteration_{iteration}.png"
        
        if components is not None and abundances is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = components.shape[0]
            fig, axes = plt.subplots(2, n_comps, figsize=(4*n_comps, 8))
            if n_comps == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_comps):
                axes[0, i].imshow(components[i], cmap='viridis')
                axes[0, i].set_title(f'Component {i+1}')
                axes[0, i].axis('off')
                axes[1, i].imshow(abundances[i], cmap='hot')
                axes[1, i].set_title(f'Abundance {i+1}')
                axes[1, i].axis('off')
            
            plt.suptitle(f'FFT/NMF Analysis - Iteration {iteration}', fontsize=14)
            plt.tight_layout()
            plt.savefig(review_viz_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print("\n" + "=" * 80)
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({state.get('num_images', 1)} images)"
        print(f"🔬 FFT/NMF ANALYSIS REVIEW - {mode_str} - Iteration {iteration}")
        print("=" * 80)
        print(f"\n🖼️  Visualization saved to: {review_viz_path}")
        if components is not None:
            print(f"\n📊 Components extracted: {components.shape[0]}")
        print(f"\n⚙️ Parameters: window_size_nm={llm_params.get('window_size_nm', 'auto')}, n_components={llm_params.get('n_components', 4)}")
        if not is_single:
            print(f"\n📦 Note: These parameters will be applied to all {state.get('num_images', 1)} images.")
        print("-" * 80)
    
    def _collect_human_feedback(self, state: dict) -> dict:
        """Collect human feedback on the FFT/NMF analysis.

        Uses a single ``input()`` call so it maps directly to the
        Streamlit UI's "Accept as-is" / "Submit feedback" buttons:
        - Empty response  → accept current parameters
        - Non-empty text  → LLM converts natural-language suggestion to params
        """
        llm_params = state.get("llm_params", {})
        print(f"\nCurrent parameters: window_size_nm={llm_params.get('window_size_nm', 'auto')}, "
              f"n_components={llm_params.get('n_components', 4)}")
        print("Press Enter to accept, or describe what to change "
              "(e.g. 'use 6 components', 'smaller window around 2nm').")

        try:
            user_feedback = input("\n🤔 Your feedback (or press Enter to accept): ").strip()
        except (KeyboardInterrupt, EOFError):
            return {"action": "accept"}

        if not user_feedback:
            return {"action": "accept"}

        params = self._convert_feedback_to_params(user_feedback, llm_params)
        if params:
            return {"action": "modify", "params": params}
        return {"action": "accept"}

    def _convert_feedback_to_params(self, user_feedback: str, current_params: dict) -> dict:
        """Use LLM to convert natural language feedback to FFT/NMF parameters."""
        self.logger.info("   🧠 Converting feedback to parameters...")

        prompt = f"""Convert user feedback into FFT/NMF parameter adjustments.

**Current Parameters:**
{json.dumps(current_params, indent=2)}

**Available Parameters:**
- window_size_nm (float): FFT window size in nanometers
- n_components (integer): Number of NMF components to extract

**User Feedback:**
"{user_feedback}"

Return JSON with ONLY the parameters to change:
{{"n_components": 6, "window_size_nm": 2.0}}
"""

        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)

            if error_dict or not result_json:
                return None

            print(f"\n   ✅ Interpreted as: {json.dumps(result_json, indent=2)}")
            return result_json

        except Exception as e:
            self.logger.error(f"Error converting feedback: {e}")
            return None
    
    def _rerun_analysis(self, state: dict, new_params: dict) -> dict:
        self.logger.info(f"🔄 Re-running FFT/NMF with updated parameters...")
        
        llm_params = state.get("llm_params", {}).copy()
        llm_params.update(new_params)
        state["llm_params"] = llm_params
        state["current_params"] = llm_params
        
        ws_nm = llm_params.get("window_size_nm")
        nc = llm_params.get("n_components", 4)
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        if ws_nm and nm_per_pixel and nm_per_pixel > 0:
            ws_pixels = int(round(ws_nm / nm_per_pixel))
            good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
            ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        else:
            ws_pixels = 64
        
        step = max(1, ws_pixels // 4)
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels, window_size_y=ws_pixels,
                window_step_x=step, window_step_y=step, components=nc
            )
            components, abundances = analyzer.analyze(state["preprocessed_image_array"], output_dir=None)
            state["fft_components"] = components
            state["fft_abundances"] = abundances
            state["first_frame_results"] = {"components": components, "abundances": abundances, "llm_params": llm_params}
            self.logger.info(f"✅ Re-analysis complete. {nc} components extracted.")
        except Exception as e:
            self.logger.error(f"❌ Re-analysis failed: {e}")
        
        return state


# ============================================================================
# UNIFIED BATCH PROCESSING CONTROLLER
# ============================================================================

class UnifiedBatchProcessingController:
    """
    [🛠️ Tool Step]
    Processes ALL images using the refined parameters.
    Single image = batch of 1.
    """
    
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings
        self.save_visualizations = settings.get('save_visualizations', True)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        num_images = state.get("num_images", 1)
        is_single = state.get("is_single_image", False)
        
        mode_str = "SINGLE IMAGE" if is_single else f"BATCH ({num_images} images)"
        self.logger.info(f"\n\n🔄 --- PROCESSING: {mode_str} --- 🔄\n")
        
        # For single image, use existing results
        if is_single and state.get("fft_components") is not None:
            self.logger.info("📋 Using results from initial analysis (single image).")
            stats = state.get("summary_stats", {})
            state["batch_results"] = [{
                "index": 0,
                "image_path": state.get("image_path"),
                "image_name": state.get("first_image_name", "image"),
                "n_components": state["fft_components"].shape[0],
                "statistics": stats,
                "success": True,
                "error": None
            }]
            if self.save_visualizations:
                self._save_visualization(state, 0)
            return state
        
        # Batch processing
        locked_params = state.get("locked_params", state.get("current_params", {}))
        
        ws_nm = locked_params.get("window_size_nm", 10.0)
        n_components = locked_params.get("n_components", 4)
        nm_per_pixel = state.get("nm_per_pixel", 1.0)
        
        if ws_nm and nm_per_pixel and nm_per_pixel > 0:
            ws_pixels = int(round(ws_nm / nm_per_pixel))
            good_sizes = [16, 32, 48, 64, 96, 128, 192, 256]
            ws_pixels = next((s for s in good_sizes if s >= ws_pixels), 64)
        else:
            ws_pixels = 64
        
        step = max(1, ws_pixels // 4)
        
        self.logger.info(f"📦 Processing with: window={ws_pixels}px, components={n_components}")
        
        try:
            analyzer = SlidingFFTNMF(
                window_size_x=ws_pixels, window_size_y=ws_pixels,
                window_step_x=step, window_step_y=step, components=n_components
            )
        except Exception as e:
            state["error_dict"] = {"error": "Analyzer creation failed", "details": str(e)}
            return state
        
        batch_results = []
        all_abundances = []
        
        for idx in range(num_images):
            try:
                raw_image, image_name = self._get_image(state, idx)
                self.logger.info(f"   [{idx + 1}/{num_images}] Processing: {image_name}")
                
                if raw_image.dtype in [np.float32, np.float64]:
                    img_min, img_max = raw_image.min(), raw_image.max()
                    if img_max > img_min:
                        raw_image = ((raw_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        raw_image = np.zeros_like(raw_image, dtype=np.uint8)
                
                preprocessed_img, _ = preprocess_image(raw_image)
                components, abundances = analyzer.analyze(preprocessed_img, output_dir=None)
                all_abundances.append(abundances)
                
                batch_results.append({
                    "index": idx, "image_name": image_name,
                    "n_components": components.shape[0],
                    "statistics": {"components": [{"index": i+1, "mean": float(np.mean(abundances[i]))} for i in range(components.shape[0])]},
                    "success": True, "error": None
                })
                self.logger.info(f"      ✅ Extracted {components.shape[0]} components")
                
            except Exception as e:
                self.logger.error(f"      ❌ Failed: {e}")
                batch_results.append({"index": idx, "image_name": f"frame_{idx:04d}", "success": False, "error": str(e)})
        
        if num_images > 1 and all_abundances:
            state["series_abundances"] = np.stack(all_abundances, axis=0)
            state["series_components"] = components
            np.save(self.output_dir / "series_components.npy", state["series_components"])
            np.save(self.output_dir / "series_abundances.npy", state["series_abundances"])
        
        state["batch_results"] = batch_results
        state["batch_params"] = {"window_size_pixels": ws_pixels, "window_size_nm": ws_nm, "n_components": n_components, "n_frames": num_images}
        
        return state
    
    def _get_image(self, state: dict, idx: int) -> tuple:
        image_stack = state.get("image_stack")
        image_paths = state.get("image_paths")
        
        if image_stack is not None:
            return image_stack[idx], f"frame_{idx:04d}"
        elif image_paths:
            return load_image(image_paths[idx]), Path(image_paths[idx]).stem
        raise ValueError("No image data found")
    
    def _save_visualization(self, state: dict, idx: int) -> None:
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        if components is None:
            return
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        n_comps = components.shape[0]
        fig, axes = plt.subplots(2, n_comps, figsize=(4*n_comps, 8))
        if n_comps == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_comps):
            axes[0, i].imshow(components[i], cmap='viridis')
            axes[0, i].set_title(f'Component {i+1}')
            axes[0, i].axis('off')
            axes[1, i].imshow(abundances[i], cmap='hot')
            axes[1, i].set_title(f'Abundance {i+1}')
            axes[1, i].axis('off')
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(viz_dir / f"fft_nmf_{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# CONDITIONAL CUSTOM ANALYSIS CONTROLLER
# ============================================================================

class ConditionalCustomAnalysisController:
    """
    [🧠 LLM Step + 🛠️ Tool Step]
    Generates trend analysis script for n>=2, skipped for n=1.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
        self.max_correction_attempts = settings.get('max_script_corrections', 3)
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        if state.get("is_single_image", False) or state.get("num_images", 1) < 2:
            self.logger.info("\n📊 Custom trend analysis skipped (single image mode).\n")
            state["custom_analysis_results"] = {"success": True, "skipped": True, "reason": "Single image"}
            return state
        
        self.logger.info("\n\n🧠 --- CUSTOM ANALYSIS SCRIPT GENERATION --- 🧠\n")
        
        # Generate and execute script (simplified)
        script = self._fallback_script()
        script_path = self.output_dir / "analyze_results.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            result = subprocess.run(['python', str(script_path)], capture_output=True, text=True, timeout=300, cwd=str(self.output_dir))
            success = result.returncode == 0
            if success:
                print(result.stdout)
        except Exception as e:
            success = False
        
        state["custom_analysis_results"] = {"success": success, "skipped": False, "script_path": str(script_path)}
        state["analysis_script_path"] = str(script_path)
        
        return state
    
    def _fallback_script(self) -> str:
        return f'''#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTPUT_DIR = Path("{self.output_dir}")

def main():
    components = np.load(OUTPUT_DIR / "series_components.npy")
    abundances = np.load(OUTPUT_DIR / "series_abundances.npy")
    print(f"Loaded: components {{components.shape}}, abundances {{abundances.shape}}")
    
    n_comps = components.shape[0]
    mean_ab = abundances.mean(axis=(2, 3))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_comps):
        ax.plot(mean_ab[:, i], 'o-', label=f'Component {{i+1}}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Abundance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "abundance_timeseries.png", dpi=150)
    plt.close()
    
    trends = {{}}
    for i in range(n_comps):
        data = mean_ab[:, i]
        slope = np.polyfit(range(len(data)), data, 1)[0] if len(data) > 1 else 0
        trends[f"component_{{i+1}}"] = {{"mean": float(np.mean(data)), "slope": float(slope)}}
    
    with open(OUTPUT_DIR / "trends.json", 'w') as f:
        json.dump(trends, f, indent=2)
    print("Saved: trends.json, abundance_timeseries.png")

if __name__ == "__main__":
    main()
'''


# ============================================================================
# UNIFIED SYNTHESIS CONTROLLER
# ============================================================================

class UnifiedSynthesisController:
    """
    [🧠 LLM Step]
    Synthesizes findings - adapts for single vs batch.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict, store_fn: Callable = None):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self._store_analysis_images = store_fn or (lambda *a, **k: None)
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        is_single = state.get("is_single_image", False)
        
        if is_single:
            return self._synthesize_single(state)
        else:
            return self._synthesize_batch(state)
    
    def _synthesize_single(self, state: dict) -> dict:
        """
        Synthesize findings for a single image analysis.
        
        Sends the original image, global FFT, and a composite of FFT/NMF components
        and abundances to the LLM for scientific interpretation.
        
        Uses SINGLE_IMAGE_ANALYSIS_INSTRUCTIONS which expects these visuals.
        """
        self.logger.info("\n\n🔬 --- SINGLE IMAGE SYNTHESIS --- 🔬\n")
        
        # Use the existing instruction prompt from instruct.py
        prompt_parts = [SINGLE_IMAGE_ANALYSIS_INSTRUCTIONS]
        
        # 1. Include original microscopy image
        if state.get("image_blob"):
            prompt_parts.append("\n\n## Primary Microscopy Image\n")
            prompt_parts.append(state["image_blob"])
            self.logger.info("   📷 Added primary microscopy image")
        
        # 2. Include system/sample information
        system_info = state.get("system_info", {})
        if system_info:
            prompt_parts.append(f"\n\n## System Information\n```json\n{json.dumps(system_info, indent=2)}\n```")
        
        # 3. Include global FFT (resized, as raw bytes)
        global_fft = state.get("global_fft_image")
        if global_fft is not None:
            global_fft_bytes = self._array_to_png_bytes_resized(global_fft, max_size=400)
            if global_fft_bytes:
                prompt_parts.append("\n\n## Global FFT (Frequency Space Overview)\n")
                prompt_parts.append("Bright spots indicate dominant periodic structures. Distance from center corresponds to spatial frequency.\n")
                prompt_parts.append({"mime_type": "image/png", "data": global_fft_bytes})
                self.logger.info("   Added global FFT image")
        
        # 4. Create composite visualization of NMF components + abundances
        #    This sends ONE image instead of 2*N images to save context window
        composite_bytes = self._create_composite_visualization(state, max_size=800)
        if composite_bytes:
            prompt_parts.append("\n\n## FFT/NMF Decomposition Results\n")
            prompt_parts.append("Top row: NMF components (dominant spatial frequency patterns extracted from sliding FFT windows).\n")
            prompt_parts.append("Bottom row: Abundance maps (spatial distribution showing where each frequency pattern is located in the original image).\n")
            prompt_parts.append({"mime_type": "image/png", "data": composite_bytes})
            self.logger.info("   Added composite visualization (components + abundances)")
        else:
            self.logger.warning("   ⚠️ No FFT/NMF results available for synthesis")
        
        # 5. Include numerical statistics
        stats = state.get("summary_stats", {})
        if stats:
            prompt_parts.append(f"\n\n## Quantitative Statistics\n```json\n{json.dumps(stats, indent=2)}\n```")
        
        # 6. Include analysis parameters used
        params = state.get("locked_params") or state.get("llm_params") or state.get("current_params", {})
        if params:
            prompt_parts.append(f"\n\n## Analysis Parameters\n```json\n{json.dumps(params, indent=2)}\n```")
        
        # Log what we're sending
        n_images = sum(1 for p in prompt_parts if isinstance(p, dict) and p.get("mime_type"))
        self.logger.info(f"   📤 Sending {n_images} images to LLM for synthesis")
        
        # Call LLM
        try:
            self.logger.info("   🧠 Calling LLM for synthesis...")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"   ❌ Synthesis parsing failed: {error_dict}")
                state["synthesis_result"] = self._fallback_single(state)
            else:
                state["synthesis_result"] = result_json
                state["result_json"] = result_json
                self.logger.info("   ✅ Single image synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"   ❌ Synthesis error: {e}")
            state["synthesis_result"] = self._fallback_single(state)
        
        return state


    def _array_to_png_bytes_resized(self, array: np.ndarray, max_size: int = 512) -> Optional[bytes]:
        """
        Convert numpy array to PNG bytes (NOT base64) with size limit.
        
        The OpenAIAsGenerativeModel wrapper expects raw bytes for image data,
        not base64-encoded strings. This is critical for proper image handling.
        
        Args:
            array: 2D numpy array to convert
            max_size: Maximum dimension (width or height) of output image
        
        Returns:
            Raw PNG bytes, or None if conversion fails
        """
        if array is None:
            return None
            
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(array, cmap='viridis')
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
            
            # Resize if needed to limit context window usage
            buf.seek(0)
            img = Image.open(buf)
            
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Return raw bytes, NOT base64 string
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', optimize=True)
            return out_buf.getvalue()
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ Array to PNG conversion failed: {e}")
            return None


    def _create_composite_visualization(self, state: dict, max_size: int = 800) -> Optional[bytes]:
        """
        Create a single composite image showing all NMF components and their abundance maps.
        
        This reduces context window usage by sending ONE image instead of 2*N_components images.
        
        Layout:
            Row 1: Component 1 FFT | Component 2 FFT | Component 3 FFT | ...
            Row 2: Abundance 1     | Abundance 2     | Abundance 3     | ...
        
        Args:
            state: Pipeline state containing fft_components and fft_abundances
            max_size: Maximum dimension of output image
        
        Returns:
            Raw PNG bytes, or None if creation fails
        """
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")
        
        if components is None or abundances is None:
            self.logger.warning("   ⚠️ No components/abundances available for composite visualization")
            return None
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = components.shape[0]
            
            # Create figure: components on top row, abundances on bottom row
            fig, axes = plt.subplots(2, n_comps, figsize=(3 * n_comps, 6))
            
            # Handle single component case
            if n_comps == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_comps):
                # Top row: NMF Component (frequency pattern)
                axes[0, i].imshow(components[i], cmap='viridis')
                axes[0, i].set_title(f'Component {i+1}\n(Frequency)', fontsize=10, fontweight='bold')
                axes[0, i].axis('off')
                
                # Bottom row: Abundance map (spatial distribution)
                axes[1, i].imshow(abundances[i], cmap='hot')
                axes[1, i].set_title(f'Component {i+1}\n(Spatial)', fontsize=10, fontweight='bold')
                axes[1, i].axis('off')
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Resize if too large to fit context window
            buf.seek(0)
            img = Image.open(buf)
            
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                self.logger.info(f"   Resized composite to {new_size}")
            
            # Return raw bytes, NOT base64 string
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', optimize=True)
            return out_buf.getvalue()
            
        except Exception as e:
            self.logger.error(f"   ❌ Failed to create composite visualization: {e}")
            return None


    def _array_to_png_bytes(self, array: np.ndarray) -> Optional[str]:
        """
        Convert numpy array to base64 PNG string.
        
        NOTE: This method returns base64 for backward compatibility with _synthesize_batch.
        For new code using OpenAI-compatible APIs, prefer _array_to_png_bytes_resized 
        which returns raw bytes.
        
        Args:
            array: 2D numpy array to convert
        
        Returns:
            Base64-encoded PNG string, or None if conversion fails
        """
        if array is None:
            return None
            
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(array, cmap='viridis')
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
            
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            self.logger.warning(f"Array to PNG conversion failed: {e}")
            return None
    
    def _synthesize_batch(self, state: dict) -> dict:
        """
        Synthesize findings across multiple images - includes trend analysis results.
        
        This version uses raw bytes for images (compatible with OpenAI wrapper).
        """
        self.logger.info("\n\n🔬 --- BATCH SYNTHESIS --- 🔬\n")
        
        batch_results = state.get("batch_results", [])
        custom_results = state.get("custom_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        batch_params = state.get("batch_params", {})
        
        # Build summary of individual results
        stats_summary = []
        for r in batch_results:
            if r.get("success"):
                stats_summary.append({
                    "index": r["index"],
                    "name": r["image_name"],
                    "n_components": r.get("n_components", 0),
                    "statistics": r.get("statistics", {})
                })
        
        # Load trends.json if available
        trends_data = {}
        trends_path = self.output_dir / "trends.json"
        if trends_path.exists():
            try:
                with open(trends_path, 'r') as f:
                    trends_data = json.load(f)
                self.logger.info(f"   📊 Loaded trend analysis from {trends_path}")
            except Exception as e:
                self.logger.warning(f"   Failed to load trends.json: {e}")
        
        # Build prompt
        prompt_parts = [SERIES_ANALYSIS_INSTRUCTIONS]
        
        # Add batch parameters
        prompt_parts.append(f"\n\n**ANALYSIS PARAMETERS:**")
        prompt_parts.append(f"- Frames analyzed: {batch_params.get('n_frames', len(batch_results))}")
        prompt_parts.append(f"- NMF components: {batch_params.get('n_components', 'N/A')}")
        prompt_parts.append(f"- Window size: {batch_params.get('window_size_nm', 'auto')} nm ({batch_params.get('window_size_pixels', 'auto')} px)")
        
        # Add individual frame statistics (condensed)
        prompt_parts.append(f"\n\n**INDIVIDUAL FRAME SUMMARY:**\n{json.dumps(stats_summary[:10], indent=2)}")
        if len(stats_summary) > 10:
            prompt_parts.append(f"\n... and {len(stats_summary) - 10} more frames")
        
        # Include trend analysis results
        if trends_data:
            prompt_parts.append(f"\n\n**TREND ANALYSIS RESULTS (from automated script):**\n```json\n{json.dumps(trends_data, indent=2)}\n```")
        
        # Include script stdout if available
        if custom_results.get("stdout"):
            stdout_text = custom_results["stdout"]
            if len(stdout_text) > 2000:
                stdout_text = stdout_text[:2000] + "\n... [truncated]"
            prompt_parts.append(f"\n\n**SCRIPT OUTPUT:**\n```\n{stdout_text}\n```")
        
        # Add series metadata
        if series_metadata:
            prompt_parts.append(f"\n\n**SERIES METADATA:**\n{json.dumps(series_metadata, indent=2)}")
        
        # Include generated visualization images (as raw bytes)
        visualization_files = []
        for pattern in ["*.png", "abundance_timeseries.png", "components.png", "correlation_matrix.png"]:
            visualization_files.extend(self.output_dir.glob(pattern))
        
        # Deduplicate and filter
        seen = set()
        unique_viz_files = []
        for f in visualization_files:
            if f.name not in seen and not f.name.startswith("review_iteration"):
                seen.add(f.name)
                unique_viz_files.append(f)
        
        if unique_viz_files:
            prompt_parts.append("\n\n**TREND ANALYSIS VISUALIZATIONS:**")
            for viz_path in unique_viz_files[:5]:  # Limit to 5 images
                if viz_path.exists():
                    try:
                        with open(viz_path, 'rb') as f:
                            img_bytes = f.read()  # Raw bytes, not base64!
                            prompt_parts.append(f"\n{viz_path.name}:")
                            prompt_parts.append({"mime_type": "image/png", "data": img_bytes})
                            self.logger.info(f"   📈 Added visualization: {viz_path.name}")
                    except Exception as e:
                        self.logger.warning(f"   Failed to load {viz_path}: {e}")
        
        # Add NMF components visualization (as raw bytes)
        components = state.get("series_components")
        if components is not None:
            prompt_parts.append("\n\n**NMF FREQUENCY COMPONENTS:**")
            for i in range(min(components.shape[0], 4)):
                comp_bytes = self._array_to_png_bytes_resized(components[i], max_size=300)
                if comp_bytes:
                    prompt_parts.append(f"\nComponent {i+1}:")
                    prompt_parts.append({"mime_type": "image/png", "data": comp_bytes})
        
        prompt_parts.append("\n\nBased on ALL the above information (individual statistics, trend analysis, and visualizations), provide a comprehensive scientific synthesis as a JSON object. Output ONLY the JSON.")
        
        # Log image count
        n_images = sum(1 for p in prompt_parts if isinstance(p, dict) and p.get("mime_type"))
        self.logger.info(f"   📤 Sending {n_images} images to LLM for batch synthesis")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = self._fallback_batch(state, trends_data)
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Batch synthesis complete.")
                
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = self._fallback_batch(state, trends_data)
        
        return state
    
    def _fallback_single(self, state: dict) -> dict:
        n_comps = state.get("fft_components").shape[0] if state.get("fft_components") is not None else 0
        return {
            "detailed_analysis": f"FFT/NMF extracted {n_comps} components.",
            "scientific_claims": [{"claim": f"Image shows {n_comps} structural components.", "scientific_impact": "Complex microstructure identified."}]
        }
    
    def _fallback_batch(self, state: dict, trends_data: dict = None) -> dict:
        """Fallback for batch synthesis - incorporates trends if available."""
        n_frames = state.get("num_images", 0)
        n_comps = 0
        if state.get("series_components") is not None:
            n_comps = state["series_components"].shape[0]
        
        # Build trend summary from trends_data if available
        trend_summary = ""
        if trends_data:
            trend_parts = []
            for comp_name, comp_data in trends_data.items():
                if isinstance(comp_data, dict):
                    trend = comp_data.get("trend", "stable")
                    slope = comp_data.get("slope", 0)
                    trend_parts.append(f"{comp_name}: {trend} (slope={slope:.4f})")
            if trend_parts:
                trend_summary = f" Trends: {'; '.join(trend_parts)}."
        
        return {
            "detailed_analysis": f"Time-series analysis of {n_frames} frames with {n_comps} NMF components.{trend_summary} See visualizations for detailed temporal evolution.",
            "temporal_interpretation": "Quantitative trends available in the generated analysis plots and trends.json file.",
            "scientific_claims": [{
                "claim": f"Time-series FFT/NMF analysis of {n_frames} microscopy frames reveals {n_comps} distinct evolving structural frequency components.",
                "scientific_impact": "Temporal decomposition enables tracking of structural dynamics and phase evolution.",
                "has_anyone_question": f"Has anyone used sliding-window FFT/NMF decomposition for in-situ microscopy time-series to track structural evolution?",
                "keywords": ["in-situ microscopy", "FFT", "NMF", "time-series", "structural dynamics"]
            }]
        }
    
    def _array_to_png_bytes(self, array: np.ndarray) -> Optional[str]:
        """Convert numpy array to base64 PNG string."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(array, cmap='viridis')
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
            
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            self.logger.warning(f"Array to PNG conversion failed: {e}")
            return None
        

class UnifiedReportGenerationController:
    """
    [📄 Report Step]
    Generates HTML report with embedded visualizations - adapts for single vs batch.
    """
    
    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings, 
                 parse_fn: Callable, settings: dict):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse_llm_response = parse_fn
        self.settings = settings
        self.output_dir = Path(settings.get('output_dir', 'analysis_output'))
    
    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("batch_cancelled"):
            return state
        
        is_single = state.get("is_single_image", False)
        self.logger.info("\n\n📄 --- GENERATING REPORT --- 📄\n")
        
        try:
            # For batch mode, generate trend visualizations first
            if not is_single:
                self._generate_trend_visualizations(state)
            
            if is_single:
                self._generate_single_image_report(state)
            else:
                self._generate_batch_report(state)
        except Exception as e:
            self.logger.error(f"   ❌ Report generation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return state
    
    def _generate_trend_visualizations(self, state: dict) -> None:
        """
        Generate trend analysis visualizations for batch/series data.
        
        Creates:
            - abundance_timeseries.png: Mean abundance per component over time
            - trends.json: Numerical trend data
        """
        series_abundances = state.get("series_abundances")
        series_components = state.get("series_components")
        
        if series_abundances is None:
            self.logger.warning("   No series abundances available for trend visualization")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import json
            
            # series_abundances shape: (n_frames, n_components, h, w)
            n_frames, n_comps = series_abundances.shape[:2]
            
            # Calculate mean abundance per component per frame
            mean_abundances = series_abundances.mean(axis=(2, 3))  # (n_frames, n_components)
            
            # === 1. Abundance Time Series Plot ===
            fig, ax = plt.subplots(figsize=(10, 5))
            
            colors = plt.cm.viridis(np.linspace(0, 1, n_comps))
            for i in range(n_comps):
                ax.plot(mean_abundances[:, i], 'o-', color=colors[i], 
                       label=f'Component {i+1}', linewidth=2, markersize=4)
            
            ax.set_xlabel('Frame', fontsize=12)
            ax.set_ylabel('Mean Abundance', fontsize=12)
            ax.set_title('Component Abundance Over Time', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "abundance_timeseries.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info("   📈 Generated abundance_timeseries.png")
            
            # === 2. Trend Statistics ===
            trends = {}
            for i in range(n_comps):
                data = mean_abundances[:, i]
                
                # Calculate slope (trend direction)
                if len(data) > 1:
                    slope = np.polyfit(range(len(data)), data, 1)[0]
                else:
                    slope = 0.0
                
                # Determine trend direction
                if abs(slope) < 0.001:
                    trend = "stable"
                elif slope > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
                
                trends[f"component_{i+1}"] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "slope": float(slope),
                    "trend": trend
                }
            
            # Save trends.json
            with open(self.output_dir / "trends.json", 'w') as f:
                json.dump(trends, f, indent=2)
            
            self.logger.info("   📊 Generated trends.json")
            
            # Store in state for synthesis
            state["trend_data"] = trends
            
        except Exception as e:
            self.logger.error(f"   ❌ Trend visualization failed: {e}")
    
    # =========================================================================
    # SINGLE IMAGE REPORT
    # =========================================================================
    
    def _generate_single_image_report(self, state: dict) -> None:
        """
        Generate comprehensive HTML report for single image analysis.
        
        Structure:
            1. System Information
            2. Scientific Analysis
            3. Visualizations
            4. Scientific Claims
        """
        
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        result = batch_results[0] if batch_results else {}
        
        detailed_analysis = synthesis.get("detailed_analysis", "No analysis available.")
        scientific_claims = synthesis.get("scientific_claims", [])
        
        # Get parameters used
        params = state.get("locked_params") or state.get("llm_params") or state.get("current_params", {})
        stats = state.get("summary_stats", {})
        system_info = state.get("system_info", {})
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        image_name = result.get('image_name', state.get('first_image_name', 'unknown'))
        n_components = result.get('n_components', stats.get('n_components', 0))
        
        # Generate visualizations as base64 for embedding
        original_img_b64 = self._get_original_image_b64(state)
        global_fft_b64 = self._array_to_b64_for_report(state.get("global_fft_image"))
        composite_b64 = self._create_composite_for_report(state)
        
        # Start HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FFT/NMF-based Microscopy Image Analysis Report - {image_name}</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            background: #f4f4f9; 
            line-height: 1.6;
        }}
        .container {{ 
            background: #fff; 
            padding: 40px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 15px; 
            margin-bottom: 30px;
        }}
        h2 {{ 
            color: #2980b9; 
            margin-top: 40px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 25px;
        }}
        .info-box {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .info-box p {{
            margin: 8px 0;
        }}
        .analysis-text {{ 
            white-space: pre-wrap; 
            background: #fafafa; 
            padding: 25px; 
            border-radius: 8px; 
            border: 1px solid #eee;
            font-size: 0.95em;
        }}
        .claim-card {{ 
            background: linear-gradient(135deg, #e8f6f3 0%, #d5f5e3 100%);
            border-left: 5px solid #1abc9c; 
            padding: 20px; 
            margin: 15px 0; 
            border-radius: 0 8px 8px 0;
        }}
        .claim-card strong {{
            color: #16a085;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .viz-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .viz-card img {{
            max-width: 100%;
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .viz-card .caption {{
            color: #666;
            font-size: 0.9em;
            font-style: italic;
        }}
        .full-width-viz {{
            width: 100%;
            margin: 20px 0;
            text-align: center;
        }}
        .full-width-viz img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .full-width-viz .caption {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .footer {{ 
            margin-top: 50px; 
            text-align: center; 
            color: #7f8c8d; 
            font-size: 0.8em;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 FFT/NMF-based Microscopy Image Analysis Report</h1>
"""
        
        # === 1. SYSTEM INFORMATION ===
        html += """
    <h2>📋 System Information</h2>
"""
        if system_info and isinstance(system_info, dict) and len(system_info) > 0:
            html += """
    <div class="info-box">
"""
            for key, value in system_info.items():
                if isinstance(value, np.ndarray):
                    value = f"Array shape: {value.shape}"
                html += f"        <p><strong>{key}:</strong> {value}</p>\n"
            html += """
    </div>
"""
        else:
            html += """
    <div class="info-box">
        <p>No system information provided.</p>
    </div>
"""
        
        # === 2. SCIENTIFIC ANALYSIS ===
        html += f"""
    <h2>🔍 Scientific Analysis</h2>
    <div class="analysis-text">{detailed_analysis}</div>
"""
        
        # === 3. VISUALIZATIONS ===
        html += """
    <h2>📊 Visualizations</h2>
"""
        
        # Original image and Global FFT side by side
        if original_img_b64 or global_fft_b64:
            html += """
    <div class="viz-grid">
"""
            if original_img_b64:
                html += f"""
        <div class="viz-card">
            <img src="data:image/png;base64,{original_img_b64}" alt="Original Microscopy Image">
            <div class="caption">Original Microscopy Image</div>
        </div>
"""
            if global_fft_b64:
                html += f"""
        <div class="viz-card">
            <img src="data:image/png;base64,{global_fft_b64}" alt="Global FFT">
            <div class="caption">Global FFT</div>
        </div>
"""
            html += """
    </div>
"""
        
        # Composite visualization (components + abundances)
        if composite_b64:
            html += f"""
    <h3>FFT/NMF Decomposition Results</h3>
    <div class="full-width-viz">
        <img src="data:image/png;base64,{composite_b64}" alt="FFT/NMF Components and Abundances">
        <p class="caption">
            <strong>Top row:</strong> NMF components (frequency patterns) &nbsp;|&nbsp; 
            <strong>Bottom row:</strong> Abundance maps (spatial distribution)
        </p>
    </div>
"""
        
        # === 4. SCIENTIFIC CLAIMS ===
        html += """
    <h2>💡 Scientific Claims</h2>
"""
        if scientific_claims:
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                html += f"""
    <div class="claim-card">
        <strong>Claim {i}:</strong> {claim.get('claim', 'N/A')}<br><br>
        <strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}<br><br>
        <strong>Literature Search Query:</strong> {claim.get('has_anyone_question', 'N/A')}<br><br>
        <strong>Keywords:</strong> {keywords_str}
    </div>
"""
        else:
            html += """
    <div class="info-box">
        <p>No scientific claims generated.</p>
    </div>
"""
        
        # Footer
        html += """
    <div class="footer">
        Generated by FFT/NMF-based Microscopy Analysis Agent<br>
    </div>
</div>
</body>
</html>"""
        
        # Write report
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated report: {report_path}")
    
    # =========================================================================
    # BATCH REPORT
    # =========================================================================
    
    def _generate_batch_report(self, state: dict) -> None:
        """
        Generate comprehensive HTML report for batch analysis.
        
        Structure:
            1. System Information
            2. Scientific Analysis
            3. Visualizations
            4. Scientific Claims
        """
        
        synthesis = state.get("synthesis_result", {})
        batch_results = state.get("batch_results", [])
        batch_params = state.get("batch_params", {})
        series_metadata = state.get("series_metadata", {})
        
        detailed_analysis = synthesis.get("detailed_analysis", "No synthesis available.")
        scientific_claims = synthesis.get("scientific_claims", [])
        temporal_interpretation = synthesis.get("temporal_interpretation", "")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_images = state.get("num_images", len(batch_results))
        successful = sum(1 for r in batch_results if r.get("success"))
        
        # Collect all PNG visualizations
        embedded_images = []
        for png_path in sorted(self.output_dir.glob("*.png")):
            if png_path.name.startswith("review_iteration"):
                continue
            try:
                with open(png_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode('utf-8')
                    embedded_images.append({
                        "name": png_path.stem,
                        "data": b64,
                        "filename": png_path.name
                    })
            except Exception:
                continue
        
        # Get series components for visualization
        series_components = state.get("series_components")
        
        # Get system info (provided by user) - this is the metadata for the experiment
        system_info = state.get("system_info", {})
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FFT/NMF-based Microscopy Image Analysis Report</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, sans-serif; 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px; 
            background: #f4f4f9;
            line-height: 1.6;
        }}
        .container {{ 
            background: #fff; 
            padding: 40px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 15px;
        }}
        h2 {{ 
            color: #2980b9; 
            margin-top: 40px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 25px;
        }}
        .info-box {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .info-box p {{
            margin: 8px 0;
        }}
        .analysis-text {{ 
            white-space: pre-wrap; 
            background: #fafafa; 
            padding: 25px; 
            border-radius: 8px; 
            border: 1px solid #eee;
        }}
        .claim-card {{ 
            background: linear-gradient(135deg, #e8f6f3 0%, #d5f5e3 100%);
            border-left: 5px solid #1abc9c; 
            padding: 20px; 
            margin: 15px 0; 
            border-radius: 0 8px 8px 0;
        }}
        .image-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); 
            gap: 25px; 
            margin: 25px 0; 
        }}
        .image-card {{ 
            background: white; 
            border: 1px solid #ddd; 
            padding: 20px; 
            border-radius: 8px; 
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .image-card img {{ 
            max-width: 100%; 
            border-radius: 4px;
            margin-bottom: 10px;
        }}
        .image-card .caption {{
            color: #666;
            font-size: 0.9em;
        }}
        .full-width-viz {{
            width: 100%;
            margin: 25px 0;
            text-align: center;
        }}
        .full-width-viz img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .full-width-viz .caption {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .footer {{ 
            margin-top: 50px; 
            text-align: center; 
            color: #7f8c8d; 
            font-size: 0.8em;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔬 FFT/NMF-based Microscopy Image Analysis Report</h1>
"""
        
        # === 1. SYSTEM INFORMATION ===
        html += """
    <h2>📋 System Information</h2>
"""
        if system_info and isinstance(system_info, dict) and len(system_info) > 0:
            html += """
    <div class="info-box">
"""
            for key, value in system_info.items():
                if isinstance(value, np.ndarray):
                    value = f"Array shape: {value.shape}"
                html += f"        <p><strong>{key}:</strong> {value}</p>\n"
            html += """
    </div>
"""
        else:
            html += """
    <div class="info-box">
        <p>No system information provided.</p>
    </div>
"""
        
        # === 2. SCIENTIFIC ANALYSIS ===
        html += f"""
    <h2>🔍 Scientific Analysis</h2>
    <div class="analysis-text">{detailed_analysis}</div>
"""
        
        # === 3. VISUALIZATIONS ===
        html += """
    <h2>📊 Visualizations</h2>
"""
        
        # For batch: show only components (shared across all frames)
        # Abundances vary per frame, so they're represented in trend plots instead
        components_b64 = self._create_components_only_viz(state)
        if components_b64:
            html += f"""
    <h3>NMF Frequency Components</h3>
    <div class="full-width-viz">
        <img src="data:image/png;base64,{components_b64}" alt="NMF Components">
        <p class="caption">Shared frequency patterns extracted across the entire time series</p>
    </div>
"""
        
        # Trend visualizations
        if embedded_images:
            html += """
    <h3>Trend Analysis</h3>
    <div class="image-grid">
"""
            for img in embedded_images[:8]:  # Limit to 8 images
                nice_name = img['name'].replace('_', ' ').title()
                html += f"""
        <div class="image-card">
            <img src="data:image/png;base64,{img['data']}" alt="{nice_name}">
            <div class="caption">{nice_name}</div>
        </div>
"""
            html += """
    </div>
"""
        
        # === 4. SCIENTIFIC CLAIMS ===
        html += """
    <h2>💡 Scientific Claims</h2>
"""
        if scientific_claims:
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                html += f"""
    <div class="claim-card">
        <strong>Claim {i}:</strong> {claim.get('claim', 'N/A')}<br><br>
        <strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}<br><br>
        <strong>Literature Search Query:</strong> {claim.get('has_anyone_question', 'N/A')}<br><br>
        <strong>Keywords:</strong> {keywords_str}
    </div>
"""
        else:
            html += """
    <div class="info-box">
        <p>No scientific claims generated.</p>
    </div>
"""
        
        # Footer
        html += """
    <div class="footer">
        Generated by FFT/NMF-based Microscopy Analysis Agent<br>
    </div>
</div>
</body>
</html>"""
        
        report_path = self.output_dir / "analysis_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"   ✅ Generated batch report: {report_path}")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_original_image_b64(self, state: dict) -> Optional[str]:
        """Get original image as base64 for report embedding."""
        try:
            img_array = state.get("preprocessed_image_array")
            if img_array is None:
                return None
            # Use isinstance check to avoid numpy truth value ambiguity
            if not isinstance(img_array, np.ndarray):
                return None
            return self._array_to_b64_for_report(img_array, cmap='gray')
        except Exception as e:
            self.logger.warning(f"Could not get original image: {e}")
            return None
    
    def _array_to_b64_for_report(
        self, 
        array: np.ndarray, 
        max_size: int = 500,
        cmap: str = 'viridis'
    ) -> Optional[str]:
        """
        Convert numpy array to base64 PNG for HTML embedding.
        
        Args:
            array: 2D numpy array
            max_size: Maximum dimension
            cmap: Matplotlib colormap
        
        Returns:
            Base64 encoded PNG string
        """
        # Check for None using 'is' to avoid numpy truth value issues
        if array is None:
            return None
        if not isinstance(array, np.ndarray):
            return None
        if array.size == 0:
            return None
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(array, cmap=cmap)
            ax.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close(fig)
            
            # Resize if needed
            buf.seek(0)
            img = Image.open(buf)
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', optimize=True)
            out_buf.seek(0)
            
            return base64.b64encode(out_buf.read()).decode('utf-8')
            
        except Exception as e:
            self.logger.warning(f"Array to b64 conversion failed: {e}")
            return None
    
    def _create_composite_for_report(self, state: dict, max_size: int = 900) -> Optional[str]:
        """
        Create composite visualization for report (components + abundances).
        
        Works for both single image and batch mode by checking multiple state keys.
        
        Returns base64 encoded PNG.
        """
        # Try different state keys for components/abundances (single vs batch)
        components = state.get("fft_components")
        if components is None:
            components = state.get("series_components")
        
        abundances = state.get("fft_abundances")
        if abundances is None:
            abundances = state.get("series_abundances")
        
        # Check if we have valid arrays
        if components is None or not isinstance(components, np.ndarray):
            return None
        if abundances is None or not isinstance(abundances, np.ndarray):
            return None
        
        # For batch mode, series_abundances has shape (n_frames, n_components, h, w)
        # We want to show the first frame's abundances or average across frames
        if abundances.ndim == 4:
            # Use first frame's abundances for visualization
            abundances = abundances[0]
        
        # Ensure shapes are compatible
        if components.shape[0] != abundances.shape[0]:
            self.logger.warning(f"Component/abundance shape mismatch: {components.shape} vs {abundances.shape}")
            n_comps = min(components.shape[0], abundances.shape[0])
            components = components[:n_comps]
            abundances = abundances[:n_comps]
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = components.shape[0]
            
            fig, axes = plt.subplots(2, n_comps, figsize=(4 * n_comps, 8))
            
            if n_comps == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_comps):
                # Components (top row)
                im1 = axes[0, i].imshow(components[i], cmap='viridis')
                axes[0, i].set_title(f'Component {i+1}\n(Frequency Pattern)', fontsize=11, fontweight='bold')
                axes[0, i].axis('off')
                plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
                
                # Abundances (bottom row)
                im2 = axes[1, i].imshow(abundances[i], cmap='hot')
                axes[1, i].set_title(f'Component {i+1}\n(Spatial Distribution)', fontsize=11, fontweight='bold')
                axes[1, i].axis('off')
                plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            # Resize if needed
            buf.seek(0)
            img = Image.open(buf)
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', optimize=True)
            out_buf.seek(0)
            
            return base64.b64encode(out_buf.read()).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Composite for report failed: {e}")
            return None
    
    def _create_series_components_viz(self, components: np.ndarray, max_size: int = 900) -> Optional[str]:
        """Create visualization of series components for batch report."""
        if components is None:
            return None
        if not isinstance(components, np.ndarray):
            return None
        if components.size == 0:
            return None
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = min(components.shape[0], 6)  # Limit to 6 components
            
            fig, axes = plt.subplots(1, n_comps, figsize=(4 * n_comps, 4))
            
            if n_comps == 1:
                axes = [axes]
            
            for i in range(n_comps):
                im = axes[i].imshow(components[i], cmap='viridis')
                axes[i].set_title(f'Component {i+1}', fontsize=12, fontweight='bold')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            buf.seek(0)
            img = Image.open(buf)
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', optimize=True)
            out_buf.seek(0)
            
            return base64.b64encode(out_buf.read()).decode('utf-8')
            
        except Exception as e:
            self.logger.warning(f"Series components viz failed: {e}")
            return None

    def _create_components_only_viz(self, state: dict, max_size: int = 900) -> Optional[str]:
        """
        Create visualization showing only NMF components (no abundances).
        
        For batch/series analysis, components are shared across all frames,
        while abundances vary per frame (shown in trend plots instead).
        """
        components = state.get("series_components")
        if components is None:
            components = state.get("fft_components")
        
        if components is None:
            return None
        if not isinstance(components, np.ndarray):
            return None
        if components.size == 0:
            return None
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            n_comps = min(components.shape[0], 8)  # Limit to 8 components
            
            # Determine grid layout
            if n_comps <= 4:
                n_cols = n_comps
                n_rows = 1
            else:
                n_cols = 4
                n_rows = (n_comps + 3) // 4
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            
            if n_comps == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n_comps):
                row, col = i // n_cols, i % n_cols
                im = axes[row, col].imshow(components[i], cmap='viridis')
                axes[row, col].set_title(f'Component {i+1}', fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(n_comps, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            buf.seek(0)
            img = Image.open(buf)
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            out_buf = io.BytesIO()
            img.save(out_buf, format='PNG', optimize=True)
            out_buf.seek(0)
            
            return base64.b64encode(out_buf.read()).decode('utf-8')
            
        except Exception as e:
            self.logger.warning(f"Components-only viz failed: {e}")
            return None