import os
import json
import logging
from ..instruct import FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS
from ....tools.fft_nmf import run_fft_nmf_analysis
from ....tools.image_processor import normalize_and_convert_to_image_bytes, calculate_global_fft

class GetFFTParamsController:
    """
    [🧠 LLM Step] Asks an LLM to suggest FFT/NMF parameters.
    """
    def __init__(self, model, logger, generation_config, safety_settings):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings

    def execute(self, state: dict) -> dict:
        self.logger.info("🧠 LLM Step: Reasoning about sFFT/NMF parameters...")
        image_blob = state["image_blob"]
        system_info = state["system_info"]
        
        prompt_parts = [FFT_NMF_PARAMETER_ESTIMATION_INSTRUCTIONS]
        prompt_parts.append("\nImage to analyze for parameters:\n")
        prompt_parts.append(image_blob)
        if system_info:
             prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(system_info, indent=2)}")
        
        param_gen_config = None#GenerationConfig(response_mime_type="application/json")
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=param_gen_config,
                safety_settings=self.safety_settings,
            )
            result = json.loads(response.text)
            state["llm_params"] = result

            # Pretty-print the reasoning for the human user
            print("\n" + "="*80)
            print("🧠 LLM REASONING (GetFFTParamsController)")
            print(f"   Explanation: {result.get('explanation', 'No reasoning provided.')}")
            print(f"   Suggested Params: window_size_nm={result.get('window_size_nm')}, n_components={result.get('n_components')}")
            print("="*80 + "\n")
            self.logger.info("✅ LLM Step Complete.")

        except Exception as e:
            self.logger.error(f"❌ LLM Step Failed: {e}")
            state["llm_params"] = {} # Store empty dict on failure
            
        return state

class RunFFTNMFController:
    """
    [🛠️ Tool Step] Runs the FFT/NMF tool.
    This controller relies *exclusively* on the LLM's parameters.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings # Still needed for output_dir, etc.

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n🛠️ --- CALLING TOOL: Sliding FFT + NMF --- 🛠️\n")
        llm_params = state.get("llm_params")
        
        if not llm_params or "window_size_nm" not in llm_params or "n_components" not in llm_params:
            self.logger.error("❌ Tool Step Failed: LLM parameters (window_size_nm, n_components) not found in state.")
            self.logger.error("   This likely means the GetFFTParamsController failed. Skipping FFT/NMF.")
            state["fft_components"] = None
            state["fft_abundances"] = None
            return state

        ws_nm = llm_params.get("window_size_nm")
        nc = llm_params.get("n_components")
        nm_per_pixel = state.get("nm_per_pixel")
        ws_pixels = None

        if ws_nm is not None and nm_per_pixel is not None and nm_per_pixel > 0:
             calculated_ws_pixels = int(round(ws_nm / nm_per_pixel))
             good_fft_sizes = [16, 24, 32, 48, 64, 80, 96, 120, 128, 160, 180, 192, 240, 256, 360, 384, 480, 512]
             ws_pixels = next((s for s in good_fft_sizes if s >= calculated_ws_pixels), 512)
             self.logger.info(f"   (Tool Info: LLM-guided size: {ws_nm:.2f} nm -> {calculated_ws_pixels}px. Using FFT-optimal size: {ws_pixels}px)")
        else:
            self.logger.error(f"❌ Tool Step Failed: Cannot calculate window size in pixels. ws_nm={ws_nm}, nm_per_pixel={nm_per_pixel}")
            state["fft_components"] = None
            state["fft_abundances"] = None
            return state

        step = max(1, ws_pixels // 4)

        components, abundances = run_fft_nmf_analysis(
            state["image_path"], ws_pixels, nc, step, self.settings, self.logger
        )
        
        state["fft_components"] = components
        state["fft_abundances"] = abundances
        self.logger.info("✅ Tool Step Complete: FFT/NMF results received.")
        return state

class RunGlobalFFTController:
    """
    [🛠️ Tool Step]
    A modular controller that calculates the global FFT of the primary image
    and tells the tool to save a visualization to disk.
    """
    def __init__(self, logger: logging.Logger, settings: dict):
        self.logger = logger
        self.settings = settings

    def execute(self, state: dict) -> dict:
        self.logger.info("\n\n🛠️ --- CALLING TOOL: Global FFT --- 🛠️\n")
        
        try:
            image_array = state["preprocessed_image_array"]
            
            # 1. Prepare the save path
            output_dir = self.settings.get("visualization_dir", "fft_nmf_visualizations")
            base_name = os.path.splitext(os.path.basename(state["image_path"]))[0]
            safe_base_name = "".join(c if c.isalnum() else "_" for c in base_name)
            # Use a simpler timestamp or just the name for predictability
            filename = f"{safe_base_name}_global_fft.png"
            filepath = os.path.join(output_dir, filename)
            
            # 2. Call the tool and pass the save_path
            global_fft_image = calculate_global_fft(image_array, save_path=filepath)
            
            # 3. Store the result back in the state
            state["global_fft_image"] = global_fft_image

            self.logger.info("✅ Tool Step Complete: Global FFT calculated.")

        except Exception as e:
            # The tool will log its own error, but we log the step failure
            self.logger.error(f"❌ Tool Step Failed: Global FFT calculation failed: {e}")
            state["global_fft_image"] = None
            
        return state

class BuildFFTNMFPromptController:
    """
    [📝 Prep Step] Builds the final prompt from the state,
    adding Global FFT and Sliding FFT/NMF results.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, state: dict) -> dict:
        self.logger.info("📝 Prep Step: Building final prompt with analysis results...")
        
        prompt_parts = [state["instruction_prompt"]]
        
        if state.get("additional_top_level_context"):
            prompt_parts.append(f"\n\n## Special Considerations:\n{state['additional_top_level_context']}\n")
            
        prompt_parts.append("\n\nPrimary Microscopy Image:\n")
        prompt_parts.append(state["image_blob"])

        # Add the Global FFT image if it was calculated
        global_fft_image = state.get("global_fft_image")
        if global_fft_image is not None:
            try:
                fft_bytes = normalize_and_convert_to_image_bytes(global_fft_image, log_scale=False) # Already log-scaled
                prompt_parts.append("\n\nGlobal FFT (Frequency Spectrum of entire image):")
                prompt_parts.append({"mime_type": "image/jpeg", "data": fft_bytes})
                state["analysis_images"].append({"label": "Global FFT", "data": fft_bytes})
            except Exception as e:
                self.logger.error(f"Failed to convert Global FFT for prompt: {e}")

        # Add Sliding FFT/NMF results
        components = state.get("fft_components")
        abundances = state.get("fft_abundances")

        if components is not None and abundances is not None:
            prompt_parts.append("\n\nSupplemental Analysis Data (Sliding FFT + NMF):")
            for i in range(components.shape[0]):
                try:
                    comp_bytes = normalize_and_convert_to_image_bytes(components[i], log_scale=True)
                    abun_bytes = normalize_and_convert_to_image_bytes(abundances[i])
                    
                    prompt_parts.append(f"\nNMF Component {i+1} (Frequency Pattern):")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": comp_bytes})
                    prompt_parts.append(f"\nNMF Abundance Map {i+1} (Spatial Distribution):")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": abun_bytes})
                    
                    state["analysis_images"].append({"label": f"NMF Abundance Map {i+1}", "data": abun_bytes})
                except Exception as e:
                    self.logger.error(f"Failed to convert NMF result {i+1} for prompt: {e}")
        else:
            prompt_parts.append("\n\n(No supplemental *sliding* FFT/NMF analysis was run or it failed)")

        prompt_parts.append(f"\n\nAdditional System Information:\n{json.dumps(state['system_info'], indent=2)}")
        prompt_parts.append("\n\nProvide your analysis strictly in the requested JSON format.")
        
        state["final_prompt_parts"] = prompt_parts
        self.logger.info("✅ Prep Step Complete: Final prompt is ready.")
        return state