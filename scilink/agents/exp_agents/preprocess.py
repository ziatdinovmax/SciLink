import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
import logging
import json
from typing import Tuple, Dict, Any
import os
import re
import matplotlib.pyplot as plt 
from io import BytesIO

from .base_agent import BaseUtilityAgent

from .instruct import (
    PRE_PROCESSING_STRATEGY_INSTRUCTIONS,
    CUSTOM_PREPROCESSING_SCRIPT_INSTRUCTIONS,
    CUSTOM_SCRIPT_CORRECTION_INSTRUCTIONS,
    CUSTOM_PREPROCESSING_SCRIPT_1D_INSTRUCTIONS, 
    CUSTOM_SCRIPT_CORRECTION_1D_INSTRUCTIONS,
    CURVE_PREPROCESSING_STRATEGY_INSTRUCTIONS,
    PREPROCESSING_QUALITY_ASSESSMENT_INSTRUCTIONS
)
from ...executors import ScriptExecutor, require_sandbox_approval

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class HyperspectralPreprocessingAgent(BaseUtilityAgent):
    """
    An agent that uses an LLM to determine the optimal pre-processing strategy
    AND can run custom Python scripts for non-standard processing.
    """

    MAX_SCRIPT_ATTEMPTS = 3

    def __init__(self, *args,
                 output_dir: str = "preprocessing_output",
                 executor_timeout: int = 300,
                 **kwargs):
        """Initialize the pre-processing agent."""

        if not require_sandbox_approval(
            context="HyperspectralPreprocessingAgent (hyperspectral preprocessing analysis)"
        ):
            raise RuntimeError(
                "HyperspectralPreprocessingAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )
        
        # Pass output_dir to BaseAnalysisAgent for state management
        super().__init__(*args, output_dir=output_dir, **kwargs) 
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent_type = "hyperspectral_preprocessing"
        
        # BaseAgent created self.output_dir as a Path. We just ensure it's absolute.
        self.output_dir = self.output_dir.resolve()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.executor = ScriptExecutor(timeout=executor_timeout)
        self.logger.info(f"HyperspectralPreprocessingAgent initialized. Custom script output dir: {self.output_dir}")

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Initialize state fields specific to preprocessing."""
        return {
            "processed_files": [],
            "strategies_used": []
        }

    def _calculate_statistics(self, hspy_data: np.ndarray) -> Dict[str, Any]:
        """Calculates robust statistics for the LLM and for deterministic SNR."""
        self.logger.debug("Calculating robust statistics...")
        
        # Use a subset for faster percentile calculation if data is large
        sample_data = hspy_data
        if np.prod(hspy_data.shape) > 1e7: # > 10 million points
            self.logger.debug(f"Data is large ({np.prod(hspy_data.shape)} points), sampling for statistics.")
            indices = np.random.choice(hspy_data.size, 10**7, replace=False)
            sample_data = hspy_data.ravel()[indices]

        stats = {
            "shape": hspy_data.shape,
            "mean": np.mean(sample_data),
            "std": np.std(sample_data),
            "min": np.min(hspy_data),
            "max": np.max(hspy_data),
            "p1": np.percentile(sample_data, 1),
            "p5": np.percentile(sample_data, 5),
            "p50": np.percentile(sample_data, 50),
            "p99": np.percentile(sample_data, 99),
            "p99_9": np.percentile(sample_data, 99.9)
        }
        return stats

    def _calculate_snr(self, stats: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculates a robust, deterministic SNR from the pre-calculated stats.
        """
        self.logger.debug("Calculating deterministic SNR...")
        
        signal = stats["p99"]
        noise = stats["p50"]
        
        # Avoid division by zero if median is 0
        if noise > 1e-9:
            snr = (signal - noise) / noise
            reasoning = f"Calculated as (P99 - P50) / P50: ({signal:.2e} - {noise:.2e}) / {noise:.2e}"
        else:
            # Fallback if median is zero: use std dev (less ideal but won't crash)
            if stats["std"] > 1e-9:
                snr = signal / stats["std"]
                reasoning = f"Calculated as P99 / Std (P50 was zero): {signal:.2e} / {stats['std']:.2e}"
            else:
                snr = 0.0 # No signal or no variance
                reasoning = "SNR is 0.0 (no variance or signal)"

        # Cap SNR at a reasonable max to avoid infinity/huge numbers
        snr = min(max(snr, 0.0), 1000.0) 
        
        return float(snr), reasoning

    def run_preprocessing(self, hspy_data: np.ndarray, system_info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Runs the full LLM-guided pre-processing pipeline.
        """
        if hspy_data.ndim != 3:
            self.logger.error(f"Input data must be 3D (h, w, e), but got {hspy_data.ndim}D. Skipping processing.")
            default_mask = np.ones(hspy_data.shape[:2], dtype=bool)
            default_quality = {
                "snr_estimate": 0.0,
                "reasoning": "Processing skipped: Input data was not 3D."
            }
            return hspy_data, default_mask, default_quality

        # 1. Calculate robust statistics
        stats = self._calculate_statistics(hspy_data)
        
        # 2. Deterministically calculate SNR
        snr_value, snr_reasoning = self._calculate_snr(stats)
        data_quality = {
            "snr_estimate": snr_value,
            "reasoning": f"SNR of *original* data: {snr_reasoning}"
        }
        self.logger.info(f"Deterministic Data Quality: SNR = {snr_value:.2f} ({snr_reasoning})")

        # Check for a custom, overriding instruction
        custom_instruction = system_info.get("custom_processing_instruction")
        
        if custom_instruction:
            self.logger.info(f"Detected custom processing instruction. Diverting to script executor.")
            self.logger.info(f"Instruction: {custom_instruction}")
            
            try:
                processed_data, mask_2d, script_path = self._run_custom_script_processing(
                    hspy_data, stats, custom_instruction
                )
                self.logger.info("Custom script processing successful.")
                if script_path:
                    data_quality["custom_script_path"] = script_path
            except Exception as e:
                self.logger.error(f"Custom script processing failed: {e}. Returning original data.")
                processed_data = hspy_data
                mask_2d = np.ones(hspy_data.shape[:2], dtype=bool)
                data_quality["reasoning"] += f" | CUSTOM SCRIPT FAILED: {e}"

        else:
            self.logger.info("No custom instruction. Running standard LLM strategy selection.")
            
            # 4. Get cleaning strategy from LLM
            strategy = self._llm_select_preprocessing_strategy(stats, system_info)
            
            # 5. Apply the LLM's cleaning strategy
            processed_data, mask_2d = self._apply_preprocessing(hspy_data, strategy)

        # 6. Return all results
        return processed_data, mask_2d, data_quality

    def _llm_select_preprocessing_strategy(self, stats: Dict[str, Any], system_info: dict) -> dict:
        """
        Asks an LLM to choose the best pre-processing steps based on data stats.
        Returns ONLY the strategy dictionary.
        """
        #self.logger.info("\n\n🤖 --- DATA AGENT STEP: PRE-PROCESSING STRATEGY SELECTION --- 🤖")
        try:
            prompt_parts = [
                PRE_PROCESSING_STRATEGY_INSTRUCTIONS,
                "\n--- Data Statistics ---",
                f"Data Shape: {stats['shape']}",
                f"Data Mean: {stats['mean']:.4e}",
                f"Data Std: {stats['std']:.4e}",
                f"Data Min: {stats['min']:.4e}",
                f"Data Max: {stats['max']:.4e}",
                f"1st Percentile: {stats['p1']:.4e}",
                f"5th Percentile: {stats['p5']:.4e}",
                f"50th Percentile (Median): {stats['p50']:.4e}",
                f"99th Percentile: {stats['p99']:.4e}",
                f"99.9th Percentile: {stats['p99_9']:.4e}",
            ]
            
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)

            prompt_parts.append("\n\nProvide your strategy in the requested JSON format.")

            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM pre-processing strategy selection failed: {error_dict}. Using default strategy.")
                raise ValueError("LLM parsing failed")

            self.logger.info(f"LLM Pre-processing Strategy: {result_json.get('reasoning', 'No reasoning provided.')}")
            return result_json 

        except Exception as e:
            self.logger.error(f"LLM strategy selection failed: {e}. Falling back to default (clip and mask).")
            # Fallback strategy
            return {
              "apply_despike": False,
              "despike_kernel_size": 3,
              "apply_masking": True,
              "mask_threshold_percentile": 5.0,
              "reasoning": "Fallback: Default clipping and masking strategy."
            }

    def _apply_preprocessing(self, hspy_data: np.ndarray, strategy: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies a robust pre-processing pipeline.
        
        """
        #self.logger.info("\n\n🤖 --- DATA AGENT STEP: APPLYING PRE-PROCESSING --- 🤖")
        data_to_process = hspy_data.copy()
        mask_2d = None

        # 1. Apply Despiking
        if strategy.get('apply_despike', False):
            kernel_size = int(strategy.get('despike_kernel_size', 3))
            kernel_tuple = (kernel_size, kernel_size, 1)
            self.logger.info(f"Applying 3D Median Filter (despike) with kernel {kernel_tuple}...")
            data_to_process = median_filter(data_to_process, size=kernel_tuple)
        
        # 2. Clip all negative values to zero
        self.logger.info("Clipping all negative data points to 0.0...")
        np.clip(data_to_process, 0, None, out=data_to_process)

        # 3. Calculate Masking strategy
        if strategy.get('apply_masking', False):
            total_intensities = np.sum(data_to_process, axis=2) # (h, w)
            signal_pixels = total_intensities[total_intensities > 1e-9]
            
            if signal_pixels.size == 0:
                self.logger.warning("No signal found after despiking and clipping. Aborting mask.")
                mask_2d = np.ones(data_to_process.shape[:2], dtype=bool)
            else:
                threshold_percentile = float(strategy.get('mask_threshold_percentile', 5.0))
                self.logger.info(f"Using LLM-defined mask percentile: {threshold_percentile}")

                intensity_threshold = np.percentile(signal_pixels, threshold_percentile)
                
                self.logger.info(f"Calculated robust intensity mask threshold: {intensity_threshold:.4e}")
                mask_2d = total_intensities > intensity_threshold
                
                num_kept = np.sum(mask_2d)
                total_pixels = mask_2d.size
                self.logger.info(f"Mask will *keep* {num_kept} / {total_pixels} pixels ({(num_kept/total_pixels)*100:.1f}%)")

        # 4. Apply Mask
        if mask_2d is not None:
            self.logger.info("Applying final intensity mask to data...")
            data_to_process *= mask_2d[..., np.newaxis]
        else:
            self.logger.info("No mask was created, returning despiked and clipped data.")
            mask_2d = np.ones(data_to_process.shape[:2], dtype=bool)

        return data_to_process, mask_2d

    def _extract_script_from_response(self, response_text: str) -> str:
        """Extracts Python code from an LLM response."""
        script_content = response_text.strip()
        match = re.search(r"```(?:python)?\n(.*?)\n```", script_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        if script_content.lower().startswith("python"):
            potential_code = script_content[len("python"):].strip()
            if potential_code.startswith(("import ", "def ", "#")):
                return potential_code
        
        if script_content.startswith(("import ", "def ", "#")):
            return script_content

        self.logger.error(f"LLM response did not contain a recognizable Python code block: {script_content[:300]}")
        raise ValueError("LLM failed to generate Python script in a recognizable format.")

    def _generate_custom_script(self, stats: dict, instruction: str, input_filename: str) -> str:
        self.logger.info("Generating Python script for custom preprocessing...")
        
        prompt = CUSTOM_PREPROCESSING_SCRIPT_INSTRUCTIONS.format(
            instruction=instruction,
            stats_json=json.dumps(stats, indent=2),
            input_filename=input_filename
        )
        
        response = self.model.generate_content(prompt)
        fitting_script = self._extract_script_from_response(response.text)
        
        if not fitting_script:
            raise ValueError("LLM generated an empty or unextractable script.")

        return fitting_script

    def _generate_and_execute_custom_script_with_retry(
        self, stats: dict, instruction: str, input_filename: str 
    ) -> dict:
        last_error = "No script generated yet."
        custom_script = None

        for attempt in range(1, self.MAX_SCRIPT_ATTEMPTS + 1):
            try:
                if attempt == 1:
                    self.logger.info(f"Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Generating initial script...")
                    custom_script = self._generate_custom_script(stats, instruction, input_filename)
                else:
                    self.logger.warning(f"Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Script failed. Requesting correction...")
                    correction_prompt = CUSTOM_SCRIPT_CORRECTION_INSTRUCTIONS.format(
                        instruction=instruction,
                        failed_script=custom_script,
                        error_message=last_error,
                        input_filename=input_filename 
                    )
                    response = self.model.generate_content(correction_prompt)
                    custom_script = self._extract_script_from_response(response.text)
                
                self.logger.info(f"Executing script (attempt {attempt})...")
                exec_result = self.executor.execute_script(custom_script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    self.logger.info("✅ Script executed successfully.")
                    return {
                        "status": "success",
                        "exec_result": exec_result,
                        "final_script": custom_script,
                        "attempts": attempt
                    }
                else:
                    last_error = exec_result.get("message", "Unknown execution error.")
                    self.logger.warning(f"Script execution attempt {attempt} failed. Error: {last_error}")

            except Exception as e:
                last_error = f"An error occurred during script generation/execution: {str(e)}"
                self.logger.error(last_error, exc_info=True)

        self.logger.error(f"Script processing failed after {self.MAX_SCRIPT_ATTEMPTS} attempts.")
        return {
            "status": "error",
            "message": f"Failed to generate a working script after {self.MAX_SCRIPT_ATTEMPTS} attempts. Last error: {last_error}",
            "last_script": custom_script
        }

    def _run_custom_script_processing(
        self, hspy_data: np.ndarray, stats: dict, instruction: str
    ) -> tuple[np.ndarray, np.ndarray, str]:
        
        input_filename = f"input_data_{os.getpid()}.npy"
        input_data_path = os.path.join(self.output_dir, input_filename)
        try:
            np.save(input_data_path, hspy_data)
            self.logger.info(f"Saved input data for script to: {input_data_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save temporary input data: {e}")

        script_bundle = self._generate_and_execute_custom_script_with_retry(
            stats, instruction, input_filename
        )

        if script_bundle["status"] != "success":
            if os.path.exists(input_data_path):
                os.remove(input_data_path)
            raise RuntimeError(script_bundle["message"])
        
        final_script = script_bundle.get("final_script", "# No script was returned.")
        script_save_path = os.path.join(self.output_dir, "custom_preprocessing_script.py")
        try:
            with open(script_save_path, "w") as f:
                f.write(f"# --- SciLink Auto-Generated Preprocessing Script ---\n")
                f.write(f"# Original Instruction: {instruction}\n")
                f.write(f"# --------------------------------------------------\n\n")
                f.write(final_script)
            self.logger.info(f"✅ Saved final script for transparency to: {script_save_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save final script: {e}")
            script_save_path = None 
        
        processed_data_path = os.path.join(self.output_dir, "processed_data.npy")
        mask_path = os.path.join(self.output_dir, "mask_2d.npy")
        
        if not os.path.exists(processed_data_path):
            raise RuntimeError(f"Script finished but did not create 'processed_data.npy' in {self.output_dir}")
        if not os.path.exists(mask_path):
            raise RuntimeError(f"Script finished but did not create 'mask_2d.npy' in {self.output_dir}")

        self.logger.info("Loading processed data and mask from script output...")
        processed_data = np.load(processed_data_path)
        mask_2d = np.load(mask_path)
        
        os.remove(input_data_path)
        os.remove(processed_data_path)
        os.remove(mask_path)
        
        return processed_data, mask_2d, script_save_path
    

class CurvePreprocessingAgent(BaseUtilityAgent):
    """
    An agent that pre-processes 1D (X, Y) curve data.
    """
    
    MAX_SCRIPT_ATTEMPTS = 5
    MAX_MODEL_ATTEMPTS = 5

    def __init__(self, *args,
                 output_dir: str = "preprocessing_output",
                 executor_timeout: int = 300,
                 **kwargs):
        """Initialize the 1D pre-processing agent."""
        # Pass output_dir to BaseAnalysisAgent for state management

        if not require_sandbox_approval(
            context="CurvePreprocessingAgent (curve preprocessing analysis)"
        ):
            raise RuntimeError(
                "CurvePreprocessingAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )
        
        super().__init__(*args, output_dir=output_dir, **kwargs) 
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agent_type = "curve_preprocessing"
        
        self.output_dir = self.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.executor = ScriptExecutor(timeout=executor_timeout)
        self.logger.info(f"CurvePreprocessingAgent initialized. Custom script output dir: {self.output_dir}")

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Initialize state fields specific to curve preprocessing."""
        return {
            "processed_curves": [],
            "custom_scripts": []
        }

    def run_preprocessing(
        self, 
        curve_data: np.ndarray, 
        system_info: Dict[str, Any],
        locked_strategy: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs the full 1D pre-processing pipeline.
        
        Args:
            curve_data: Input curve data shape (N, 2) with columns [x, y]
            system_info: System/sample metadata dictionary
            locked_strategy: If provided, use this strategy instead of asking LLM.
                            Used to ensure consistent preprocessing across a series.
        
        Returns:
            Tuple of (processed_data, data_quality_dict)
            data_quality_dict includes 'strategy' key with the strategy used,
            which can be passed as locked_strategy for subsequent spectra.
        """
        if curve_data.ndim != 2 or curve_data.shape[1] != 2:
            self.logger.error(f"Input data must be 2D (N, 2), but got {curve_data.shape}. Skipping processing.")
            return curve_data, {"reasoning": "Processing skipped: Input data was not 2-column (N, 2)."}
        
        data_quality = {}
        
        # Check for a custom, overriding instruction
        custom_instruction = system_info.get("custom_processing_instruction")
        
        if custom_instruction:  # Custom script path
            self.logger.info(f"Detected custom 1D processing instruction. Diverting to script executor.")
            self.logger.info(f"Instruction: {custom_instruction}")
            
            try:
                processed_data, script_path = self._run_custom_script_processing_1d(
                    curve_data, system_info, custom_instruction
                )
                self.logger.info("Custom 1D script processing successful.")
                if script_path:
                    data_quality["custom_script_path"] = script_path
            except Exception as e:
                self.logger.error(f"Custom 1D script processing failed: {e}. Returning original data.")
                processed_data = curve_data
                data_quality["reasoning"] = f"CUSTOM 1D SCRIPT FAILED: {e}"

        else:  # Standard LLM-guided path
            self.logger.info("Running LLM-guided standard 1D processing.")
            
            # 1. Calculate stats (needed for the LLM if no locked strategy)
            stats = self._calculate_statistics_1d(curve_data)
            
            # 2. Use locked strategy if provided, otherwise ask LLM
            if locked_strategy is not None:
                self.logger.info("Using locked preprocessing strategy for series consistency.")
                strategy = locked_strategy
            else:
                self.logger.info("No locked strategy - asking LLM for preprocessing strategy.")
                strategy = self._llm_select_1d_strategy(stats, system_info)
            
            # 3. Apply the strategy
            processed_data = self._apply_1d_strategy(curve_data, strategy)
            
            # 4. Return strategy in data_quality so it can be locked for series
            data_quality["reasoning"] = strategy.get('reasoning', 'LLM-guided standard processing applied.')
            data_quality["strategy"] = strategy

        return processed_data, data_quality

    def _llm_select_1d_strategy(self, stats: Dict[str, Any], system_info: dict) -> dict:
        """Asks an LLM to choose the best standard 1D pre-processing steps."""
        self.logger.info("🤖 Asking LLM for standard 1D processing strategy...")
        
        # Fallback strategy (the "dumb" one we're replacing)
        fallback_strategy = {
            "apply_clip": True,
            "apply_smoothing": True,
            "smoothing_window": 5,
            "reasoning": "Fallback: Default clipping and smoothing."
        }
        
        try:
            prompt_parts = [
                CURVE_PREPROCESSING_STRATEGY_INSTRUCTIONS,
                "\n--- Data Statistics (Y-axis) ---",
                f"Data Shape: {stats['shape']}",
                f"Y Mean: {stats['y_mean']:.4e}",
                f"Y Std: {stats['y_std']:.4e}",
                f"Y Min: {stats['y_min']:.4e}",
                f"Y Max: {stats['y_max']:.4e}",
                f"Y Median (p50): {stats['y_p50']:.4e}",
                f"Y p99: {stats['y_p99']:.4e}",
            ]
            
            system_info_section = self._build_system_info_prompt_section(system_info)
            if system_info_section:
                prompt_parts.append(system_info_section)

            prompt_parts.append("\n\nProvide your strategy in the requested JSON format.")

            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM 1D strategy selection failed: {error_dict}. Using default strategy.")
                return fallback_strategy

            self.logger.info(f"LLM 1D Strategy: {result_json.get('reasoning', 'No reasoning provided.')}")
            return result_json

        except Exception as e:
            self.logger.error(f"LLM 1D strategy selection failed: {e}. Falling back to default.")
            return fallback_strategy

    def _apply_1d_strategy(self, curve_data: np.ndarray, strategy: dict) -> np.ndarray:
        """
        Applies the simple 1D processing strategy chosen by the LLM.
        """
        #self.logger.info("\n\n🤖 --- DATA AGENT STEP: Applying Standard 1D Strategy --- 🤖")
                
        processed_data = curve_data.copy()
        y_data = processed_data[:, 1] # Extract Y-data
        
        # 1. Apply Clipping (if LLM said to)
        if strategy.get('apply_clip', False):
            negative_count = np.sum(y_data < 0)
            if negative_count > 0:
                self.logger.info(f"Applying clipping: Setting {negative_count} negative Y-values to 0.")
                # Re-assign the y_data variable
                y_data = np.clip(y_data, 0, None)
            else:
                self.logger.info("Clipping was True, but no negative values were found.")
        else:
            self.logger.info("Skipping clipping (as per LLM strategy or default).")

        # 2. Apply Smoothing (if LLM said to)
        if strategy.get('apply_smoothing', False):
            try:
                window_length = int(strategy.get('smoothing_window', 5))
                if window_length % 2 == 0:
                    window_length += 1
                polyorder = 2
                
                if len(y_data) > window_length:
                    self.logger.info(f"Applying Savitzky-Golay smoothing (window={window_length}, order={polyorder}).")
                    # Re-assign the y_data variable again
                    y_data = savgol_filter(y_data, window_length, polyorder)
                else:
                    self.logger.warning(f"Skipping smoothing: Data length ({len(y_data)}) is too short for window ({window_length}).")
            except Exception as e:
                self.logger.error(f"Failed to apply smoothing: {e}")
        else:
            self.logger.info("Skipping smoothing (as per LLM strategy).")
        
        # Finally, put the processed Y-data back into the copied array
        processed_data[:, 1] = y_data
        
        return processed_data
    
    def _calculate_statistics_1d(self, curve_data: np.ndarray) -> Dict[str, Any]:
        """Calculates robust statistics for the 1D data."""
        x = curve_data[:, 0]
        y = curve_data[:, 1]
        stats = {
            "shape": curve_data.shape,
            "x_min": np.min(x),
            "x_max": np.max(x),
            "y_mean": np.mean(y),
            "y_std": np.std(y),
            "y_min": np.min(y),
            "y_max": np.max(y),
            "y_p50": np.percentile(y, 50),
            "y_p99": np.percentile(y, 99),
        }
        return stats

    def _extract_script_from_response(self, response_text: str) -> str:
        """Extracts Python code from an LLM response. (Same as 3D agent)"""
        # This method is identical to the one in the hyperspectral agent
        script_content = response_text.strip()
        match = re.search(r"```(?:python)?\n(.*?)\n```", script_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        if script_content.lower().startswith("python"):
            potential_code = script_content[len("python"):].strip()
            if potential_code.startswith(("import ", "def ", "#")):
                return potential_code
        
        if script_content.startswith(("import ", "def ", "#")):
            return script_content

        self.logger.error(f"LLM response did not contain a recognizable Python code block: {script_content[:300]}")
        raise ValueError("LLM failed to generate Python script in a recognizable format.")

    def _generate_custom_script_1d(self, stats: dict, instruction: str, input_filename: str) -> str:
        """Uses an LLM to generate a Python 1D processing script."""
        self.logger.info("Generating Python script for custom 1D preprocessing...")
        
        prompt = CUSTOM_PREPROCESSING_SCRIPT_1D_INSTRUCTIONS.format(
            instruction=instruction,
            stats_json=json.dumps(stats, indent=2),
            input_filename=input_filename
        )
        
        response = self.model.generate_content(prompt)
        fitting_script = self._extract_script_from_response(response.text)
        
        if not fitting_script:
            raise ValueError("LLM generated an empty or unextractable 1D script.")
        return fitting_script

    def _generate_and_execute_custom_script_1d(
        self, stats: dict, instruction: str, input_filename: str
    ) -> dict:
        """
        Generates and executes the custom 1D script, with a retry loop.
        """
        last_error = "No script generated yet."
        custom_script = None

        for attempt in range(1, self.MAX_SCRIPT_ATTEMPTS + 1):
            try:
                if attempt == 1:
                    custom_script = self._generate_custom_script_1d(stats, instruction, input_filename)
                else:
                    self.logger.warning(f"Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: 1D script failed. Requesting correction...")
                    correction_prompt = CUSTOM_SCRIPT_CORRECTION_1D_INSTRUCTIONS.format(
                        instruction=instruction,
                        failed_script=custom_script,
                        error_message=last_error,
                        input_filename=input_filename 
                    )
                    response = self.model.generate_content(correction_prompt)
                    custom_script = self._extract_script_from_response(response.text)
                
                exec_result = self.executor.execute_script(custom_script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    return { "status": "success", "final_script": custom_script }
                else:
                    last_error = exec_result.get("message", "Unknown execution error.")
                    self.logger.warning(f"1D script execution attempt {attempt} failed. Error: {last_error}")

            except Exception as e:
                last_error = f"An error occurred during 1D script generation/execution: {str(e)}"
                self.logger.error(last_error, exc_info=True)

        return { "status": "error", "message": f"Failed after {self.MAX_SCRIPT_ATTEMPTS} attempts. Last error: {last_error}" }

    def _run_custom_script_processing_1d(
        self, curve_data: np.ndarray, system_info: dict, instruction: str
    ) -> tuple[np.ndarray, str]:
        """
        Orchestrates the custom 1D script pipeline and validates the result
        using a logic-correction loop.
        """
        stats = self._calculate_statistics_1d(curve_data)
        
        input_filename = f"input_data_1d_{os.getpid()}.npy"
        input_data_path = os.path.join(self.output_dir, input_filename)
        processed_data_path = os.path.join(self.output_dir, "processed_data.npy")
        script_save_path = os.path.join(self.output_dir, "custom_preprocessing_script_1d.py")

        # This context string will be updated with critiques if validation fails
        processing_context = instruction

        try:
            np.save(input_data_path, curve_data)
            self.logger.info(f"Saved 1D input data for script to: {input_data_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save temporary 1D input data: {e}")

        
        # Loop for logic/model correction
        for model_attempt in range(1, self.MAX_MODEL_ATTEMPTS + 1):
            self.logger.info(f"\n🤖 PREPROCESSING MODEL ATTEMPT {model_attempt}/{self.MAX_MODEL_ATTEMPTS}")

            # 1. Run the inner script generation/execution loop
            #    This loop handles code-level bugs (SyntaxError, etc.)
            script_bundle = self._generate_and_execute_custom_script_1d(
                stats, processing_context, input_filename
            )

            if script_bundle["status"] != "success":
                # This means the *inner* loop failed 5 times. We must stop.
                raise RuntimeError(script_bundle["message"])
            
            # 2. Load the data created by the script
            if not os.path.exists(processed_data_path):
                raise RuntimeError(f"1D script finished but did not create 'processed_data.npy'")
            processed_data = np.load(processed_data_path)

            # 3. Validation step (LLM quality check)
            self.logger.info("🤖 Validating custom preprocessing script output...")
            raw_plot_bytes = self._plot_curve_to_bytes(curve_data, "1. Raw Data (Before)")
            processed_plot_bytes = self._plot_curve_to_bytes(processed_data, "2. Processed Data (After)")
            
            assessment = self._evaluate_preprocessing_quality(
                raw_plot_bytes,
                processed_plot_bytes,
                instruction # Use the *original* instruction for validation
            )
            self.logger.info(f"Preprocessing Assessment: {assessment['critique']}")
            
            # 4. Check assessment and decide to loop or break
            if assessment.get("is_good_preprocessing", False):
                self.logger.info("✅ Preprocessing quality is acceptable.")
                
                # Save the final script and return the good data
                final_script = script_bundle.get("final_script", "# No script was returned.")
                try:
                    with open(script_save_path, "w") as f:
                        f.write(f"# --- SciLink Auto-Generated 1D Preprocessing Script ---\n")
                        f.write(f"# Original Instruction: {instruction}\n")
                        f.write(f"# Final Validation: {assessment['critique']}\n")
                        f.write(f"# --------------------------------------------------\n\n")
                        f.write(final_script)
                    self.logger.info(f"✅ Saved final 1D script for transparency to: {script_save_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save final 1D script: {e}")
                    script_save_path = None
                
                # Clean up temp files and return
                os.remove(input_data_path)
                os.remove(processed_data_path)
                return processed_data, script_save_path
            
            elif model_attempt < self.MAX_MODEL_ATTEMPTS:
                self.logger.warning("⚠️ Preprocessing quality is unacceptable. Attempting to correct the logic.")
                
                # Update the context for the *next* loop iteration
                processing_context = (
                    f"The original instruction was: '{instruction}'\n\n"
                    f"--- CRITIQUE OF YOUR LAST ATTEMPT ---\n"
                    f"Critique: {assessment['critique']}\n"
                    f"Suggestion: {assessment.get('suggestion', 'No suggestion provided.')}\n\n"
                    f"Please provide a new script with a different approach to fix this."
                )
                # We don't return; the loop continues to the next model_attempt

            else:
                # Max model attempts reached, this is a final failure
                self.logger.error(f"Failed to produce a good preprocessing script after {self.MAX_MODEL_ATTEMPTS} attempts.")
                raise ValueError(f"Custom preprocessing failed LLM validation after {self.MAX_MODEL_ATTEMPTS} attempts. Last critique: {assessment['critique']}")

        # This part should not be reachable, but as a fallback:
        raise RuntimeError("Agent failed to preprocess data.")
    
    def _plot_curve_to_bytes(self, curve_data: np.ndarray, title: str) -> bytes:
        """Helper to plot a 1D curve into in-memory bytes."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curve_data[:, 0], curve_data[:, 1], 'b-')
        ax.set_title(title)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.grid(True, linestyle='--')
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='jpeg', dpi=100)
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)
        return image_bytes

    def _evaluate_preprocessing_quality(self, raw_data_plot_bytes: bytes, processed_data_plot_bytes: bytes, instruction: str) -> dict:
        """Uses an LLM to visually assess the quality of the preprocessing."""
        self.logger.info("🤖 Assessing the quality of the preprocessing script...")
        if not instruction:
            instruction = "No custom instruction was provided."
            
        prompt = [
            PREPROCESSING_QUALITY_ASSESSMENT_INSTRUCTIONS,
            "## User Instruction\n" + instruction,
            "## 1. Raw Data (Before)", {"mime_type": "image/jpeg", "data": raw_data_plot_bytes},
            "## 2. Processed Data (After)", {"mime_type": "image/jpeg", "data": processed_data_plot_bytes},
        ]
        
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result_json, error = self._parse_llm_response(response)
            
            if error or not result_json:
                self.logger.warning("Failed to get a valid preprocessing assessment from LLM. Assuming it's acceptable.")
                return {"is_good_preprocessing": True, "critique": "Assessment failed."}
            
            return result_json
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing assessment: {e}", exc_info=True)
            return {"is_good_preprocessing": True, "critique": f"Assessment call failed: {e}"}