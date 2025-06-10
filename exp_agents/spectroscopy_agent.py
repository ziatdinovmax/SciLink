# exp_agents/spectroscopy_agent.py
import json
import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
from datetime import datetime

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .utils import convert_numpy_to_jpeg_bytes, normalize_and_convert_to_image_bytes
from .instruct import (
    SPECTROSCOPY_ANALYSIS_INSTRUCTIONS, 
    COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS,
    COMPONENT_VISUAL_COMPARISON_INSTRUCTIONS
)

# Import atomai's SpectralUnmixer
from atomai.stat import SpectralUnmixer


class GeminiSpectroscopyAnalysisAgent:
    """
    Agent for analyzing hyperspectral/spectroscopy data using Gemini models.
    Integrates with SciLinkLLM framework and includes LLM-guided spectral unmixing.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-05-06", 
                 spectral_unmixing_settings: dict | None = None,
                 output_dir: str = "spectroscopy_output"):
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and GOOGLE_API_KEY environment variable is not set.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        self.logger = logging.getLogger(__name__)
        
        # Spectral unmixing settings
        default_settings = {
            'method': 'nmf',
            'n_components': 4,
            'normalize': True,
            'enabled': True,
            'auto_components': True
        }
        self.spectral_settings = spectral_unmixing_settings if spectral_unmixing_settings else default_settings
        self.run_spectral_unmixing = self.spectral_settings.get('enabled', True)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _parse_llm_response(self, response) -> Tuple[Dict | None, Dict | None]:
        """Parse the LLM response, expecting JSON."""
        result_json = None
        error_dict = None
        raw_text = None
        json_string = None

        try:
            raw_text = response.text
            first_brace_index = raw_text.find('{')
            last_brace_index = raw_text.rfind('}')
            if first_brace_index != -1 and last_brace_index != -1 and last_brace_index > first_brace_index:
                json_string = raw_text[first_brace_index : last_brace_index + 1]
                result_json = json.loads(json_string)
            else:
                raise ValueError("Could not find valid JSON object delimiters '{' and '}' in the response text.")

        except (json.JSONDecodeError, AttributeError, IndexError, ValueError) as e:
            error_details = str(e)
            error_raw_response = raw_text if raw_text is not None else getattr(response, 'text', 'N/A')
            self.logger.error(f"Error parsing Gemini JSON response: {e}")
            
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                self.logger.error(f"Request blocked due to: {block_reason}")
                error_dict = {"error": f"Content blocked by safety filters", "details": f"Reason: {block_reason}"}
            elif response.candidates and response.candidates[0].finish_reason != 1:
                finish_reason = response.candidates[0].finish_reason
                self.logger.error(f"Generation finished unexpectedly: {finish_reason}")
                error_dict = {"error": f"Generation finished unexpectedly: {finish_reason}", "details": error_details}
            else:
                error_dict = {"error": "Failed to parse valid JSON from LLM response", "details": error_details}
        except Exception as e:
            self.logger.exception(f"Unexpected error processing response: {e}")
            error_dict = {"error": "Unexpected error processing LLM response", "details": str(e)}
        
        return result_json, error_dict

    def _load_hyperspectral_data(self, data_path: str) -> np.ndarray:
        """
        Load hyperspectral data from various formats.
        Supports .npy, .h5, and attempts to load other formats.
        """
        try:
            if data_path.endswith('.npy'):
                data = np.load(data_path)
            elif data_path.endswith('.h5') or data_path.endswith('.hdf5'):
                import h5py
                with h5py.File(data_path, 'r') as f:
                    # Try common keys for hyperspectral data
                    possible_keys = ['data', 'spectrum', 'hyperspectral', 'cube']
                    data_key = None
                    for key in possible_keys:
                        if key in f.keys():
                            data_key = key
                            break
                    
                    if data_key is None:
                        # Use the first dataset found
                        data_key = list(f.keys())[0]
                        self.logger.warning(f"No standard hyperspectral key found, using: {data_key}")
                    
                    data = f[data_key][:]
            else:
                # Try loading as numpy array
                data = np.load(data_path)
                
            self.logger.info(f"Loaded hyperspectral data with shape: {data.shape}")
            
            # Ensure 3D format (h, w, spectral_channels)
            if data.ndim == 2:
                self.logger.warning("2D data detected, assuming single spectrum. Reshaping to (1, 1, n_channels)")
                data = data.reshape(1, 1, -1)
            elif data.ndim != 3:
                raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
                
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load hyperspectral data from {data_path}: {e}")
            raise

    def _llm_estimate_components_from_system(self, hspy_data: np.ndarray, 
                                           system_info: Dict[str, Any] = None) -> int:
        """
        Step 1: LLM estimates optimal number of components based on system description.
        """
        try:
            self.logger.info("Step 1: Getting LLM initial component estimate from system description")
            
            # Build prompt for initial estimation
            prompt_parts = [COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS]
            
            # Add hyperspectral data information
            h, w, e = hspy_data.shape
            prompt_parts.append(f"\n\n--- Hyperspectral Data Information ---")
            prompt_parts.append(f"Data dimensions: {h}x{w} spatial pixels, {e} spectral channels")
            prompt_parts.append(f"Data statistics:")
            prompt_parts.append(f"- Mean intensity: {np.mean(hspy_data):.3f}")
            prompt_parts.append(f"- Intensity range: {np.min(hspy_data):.3f} to {np.max(hspy_data):.3f}")
            prompt_parts.append(f"- Signal-to-noise estimate: {np.mean(hspy_data) / np.std(hspy_data):.2f}")
            
            # Add system information
            if system_info:
                prompt_parts.append("\n\n--- System Information ---")
                if isinstance(system_info, dict):
                    prompt_parts.append(json.dumps(system_info, indent=2))
                else:
                    prompt_parts.append(str(system_info))
            else:
                prompt_parts.append("\n\n--- System Information ---")
                prompt_parts.append("No specific system information provided.")
            
            prompt_parts.append("\n\nBased on the system description and data characteristics, estimate the optimal number of spectral components.")
            
            # Query LLM
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM initial estimation failed: {error_dict}")
                return self.spectral_settings.get('n_components', 4)
            
            estimated_components = result_json.get('estimated_components', None)
            reasoning = result_json.get('reasoning', 'No reasoning provided')
            
            self.logger.info(f"LLM initial estimate: {estimated_components} components")
            self.logger.info(f"LLM reasoning: {reasoning}")
            
            # Validate estimate (must be reasonable)
            if isinstance(estimated_components, int) and 2 <= estimated_components <= 15:
                return estimated_components
            else:
                self.logger.warning(f"Invalid LLM estimate {estimated_components}, using default")
                return self.spectral_settings.get('n_components', 4)
                
        except Exception as e:
            self.logger.error(f"LLM initial estimation failed: {e}")
            return self.spectral_settings.get('n_components', 4)

    def _create_nmf_summary_plot(self, components: np.ndarray, abundance_maps: np.ndarray, n_comp: int) -> bytes:
        """
        Create a single summary plot showing all components and abundance maps.
        Similar to your example images.
        """
        import matplotlib.pyplot as plt
        
        try:
            # Create figure with 2 rows: spectra on top, abundance maps on bottom
            fig, axes = plt.subplots(2, n_comp, figsize=(n_comp * 3, 6))
            
            if n_comp == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_comp):
                # Top row: Component spectra
                axes[0, i].plot(components[i, :], 'b-', linewidth=1.5)
                axes[0, i].set_title(f'NMF Component {i+1}', fontsize=10)
                axes[0, i].set_xlabel('Energy Bin')
                if i == 0:
                    axes[0, i].set_ylabel('Intensity')
                axes[0, i].grid(True, alpha=0.3)
                
                # Bottom row: Abundance maps
                im = axes[1, i].imshow(abundance_maps[..., i], cmap='seismic', aspect='auto')
                axes[1, i].set_title(f'Abundance Map {i+1}', fontsize=10)
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            plt.suptitle(f'NMF Analysis: {n_comp} Components', fontsize=14, y=0.95)
            plt.tight_layout()
            
            # Save to bytes
            buf = BytesIO()
            plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_bytes = buf.getvalue()
            plt.close()
            
            return image_bytes
            
        except Exception as e:
            self.logger.error(f"Failed to create summary plot for {n_comp} components: {e}")
            return None

    def _llm_compare_visual_results(self, test_images: List[Dict], initial_estimate: int,
                                   system_info: Dict[str, Any] = None) -> int:
        """
        Step 3: LLM compares visual results to make final decision.
        """
        try:
            self.logger.info("Step 3: LLM comparing visual results for final decision")
            
            # Build prompt for visual comparison
            prompt_parts = [COMPONENT_VISUAL_COMPARISON_INSTRUCTIONS]
            
            # Add context
            prompt_parts.append(f"\n\n--- Context ---")
            prompt_parts.append(f"Initial LLM estimate: {initial_estimate} components")
            
            if system_info:
                prompt_parts.append("\nSystem Information:")
                if isinstance(system_info, dict):
                    prompt_parts.append(json.dumps(system_info, indent=2))
                else:
                    prompt_parts.append(str(system_info))
            
            # Add visual comparison
            prompt_parts.append(f"\n\n--- Visual Comparison ---")
            prompt_parts.append("Compare these NMF results:")
            
            for result in test_images:
                n_comp = result['n_components']
                label = "Under-sampling" if n_comp < initial_estimate else "Over-sampling"
                prompt_parts.append(f"\n\n**{n_comp} Components ({label}):**")
                prompt_parts.append({
                    "mime_type": "image/jpeg", 
                    "data": result['image']
                })
            
            prompt_parts.append(f"\n\nBased on visual analysis, decide between {test_images[0]['n_components']} or {test_images[1]['n_components']} components, or recommend your initial estimate of {initial_estimate}.")
            
            # Query LLM
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                self.logger.warning(f"LLM visual comparison failed: {error_dict}")
                return initial_estimate
            
            final_components = result_json.get('final_components', None)
            reasoning = result_json.get('reasoning', 'No reasoning provided')
            
            self.logger.info(f"LLM final decision: {final_components} components")
            self.logger.info(f"LLM reasoning: {reasoning}")
            
            # Validate final decision
            tested_values = [r['n_components'] for r in test_images] + [initial_estimate]
            if final_components in tested_values:
                return final_components
            else:
                self.logger.warning(f"LLM chose untested value {final_components}, using closest")
                return min(tested_values, key=lambda x: abs(x - final_components))
                
        except Exception as e:
            self.logger.error(f"LLM visual comparison failed: {e}")
            return initial_estimate

    def _llm_guided_component_workflow(self, hspy_data: np.ndarray, system_info: dict = None) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Four-step LLM-guided component optimization workflow with human-viewable outputs:
        1. LLM estimates optimal components based on system description
        2. Test under/over-sampling around that estimate
        3. LLM compares visual results to finalize decision
        4. Run final analysis with chosen number
        
        Returns:
            Tuple of (final_n_components, final_components, final_abundance_maps)
        """
        from datetime import datetime
        
        # Initialize reasoning log
        reasoning_log = {
            "workflow_start": datetime.now().isoformat(),
            "data_shape": hspy_data.shape,
            "system_info": system_info
        }
        
        # Console output for human monitoring
        print("\n🔍 Starting LLM-Guided Component Selection...")
        print("=" * 50)
        
        # Step 1: LLM initial estimate based on system description
        print("Step 1: Getting LLM initial component estimate...")
        initial_estimate = self._llm_estimate_components_from_system(hspy_data, system_info)
        
        # Log and save reasoning
        reasoning_log["initial_estimate"] = initial_estimate
        print(f"  💡 LLM suggests: {initial_estimate} components")
        
        # Save Step 1 reasoning
        step1_data = {
            "step": "initial_estimation",
            "estimated_components": initial_estimate,
            "system_info": system_info,
            "data_characteristics": {
                "shape": hspy_data.shape,
                "mean_intensity": float(np.mean(hspy_data)),
                "std_intensity": float(np.std(hspy_data))
            }
        }
        self._save_component_reasoning("step1_initial_estimate", step1_data)
        
        # Step 2: Test under-sampling and over-sampling around estimate
        test_components = [
            max(2, initial_estimate - 2),  # Under-sampling
            initial_estimate + 3           # Over-sampling  
        ]
        
        reasoning_log["test_range"] = test_components
        print(f"Step 2: Testing component numbers: {test_components}")
        
        # Run test analyses
        test_images = []
        for n_comp in test_components:
            print(f"  🧪 Testing {n_comp} components...")
            
            try:
                temp_unmixer = SpectralUnmixer(
                    method=self.spectral_settings.get('method', 'nmf'),
                    n_components=n_comp,
                    normalize=self.spectral_settings.get('normalize', True),
                    max_iter=self.spectral_settings.get('max_iter', 500),
                    random_state=42
                )
                
                components, abundance_maps = temp_unmixer.fit(hspy_data)
                summary_image = self._create_nmf_summary_plot(components, abundance_maps, n_comp)
                
                if summary_image:
                    test_images.append({
                        'n_components': n_comp,
                        'image': summary_image
                    })
                    print(f"    ✅ Generated test result for {n_comp} components")
                
            except Exception as e:
                print(f"    ❌ Failed test with {n_comp} components: {e}")
                self.logger.warning(f"Failed test analysis with {n_comp} components: {e}")
        
        # Save test images for human review
        self._save_component_comparison_plot(test_images, initial_estimate)
        
        # Step 3: LLM visual comparison to finalize decision
        print("Step 3: LLM comparing test results...")
        if len(test_images) >= 2:
            final_n_components = self._llm_compare_visual_results(
                test_images, initial_estimate, system_info
            )
            print(f"  🎯 Final decision: {final_n_components} components")
        else:
            print("  ⚠️  Insufficient test results, using initial estimate")
            final_n_components = initial_estimate
            self.logger.warning("Insufficient test results for comparison, using initial estimate")
        
        # Log final decision
        reasoning_log["final_decision"] = final_n_components
        
        # Save Step 3 reasoning
        step3_data = {
            "step": "final_comparison",
            "test_components": test_components,
            "initial_estimate": initial_estimate,
            "final_decision": final_n_components,
            "test_results_available": len(test_images)
        }
        self._save_component_reasoning("step3_final_decision", step3_data)
        
        # Step 4: Run final analysis with chosen number
        print("Step 4: Running final analysis...")
        final_unmixer = SpectralUnmixer(
            method=self.spectral_settings.get('method', 'nmf'),
            n_components=final_n_components,
            normalize=self.spectral_settings.get('normalize', True),
            **{k: v for k, v in self.spectral_settings.items() 
            if k not in ['method', 'n_components', 'normalize', 'enabled', 'auto_components']}
        )
        
        final_components, final_abundance_maps = final_unmixer.fit(hspy_data)
        
        # Complete reasoning log
        reasoning_log["workflow_end"] = datetime.now().isoformat()
        reasoning_log["final_reasoning"] = f"Selected {final_n_components} components based on LLM comparison of test results"
        
        # Create final summary files
        summary_path, reasoning_path = self._create_final_results_summary(
            final_n_components, final_components, final_abundance_maps, reasoning_log
        )
        
        # Final console summary
        print("\n📋 Component Selection Summary:")
        print(f"   Initial estimate: {initial_estimate}")
        print(f"   Test range: {test_components}")
        print(f"   Final decision: {final_n_components}")
        print(f"   Results saved to: {self.output_dir}/")
        print("=" * 50)
        
        self.logger.info(f"LLM-guided component workflow completed: {final_n_components} components selected")
        
        return final_n_components, final_components, final_abundance_maps

    def _perform_spectral_unmixing(self, hspy_data: np.ndarray, system_info: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform spectral unmixing using the complete LLM-guided workflow.
        """
        try:
            self.logger.info("--- Starting Spectral Unmixing Analysis ---")
            
            if self.spectral_settings.get('auto_components', True):
                # Use complete LLM-guided workflow
                final_n_components, components, abundance_maps = self._llm_guided_component_workflow(
                    hspy_data, system_info
                )
                return components, abundance_maps
            else:
                # Use user-specified number
                n_components = self.spectral_settings.get('n_components', 4)
                self.logger.info(f"Using user-specified {n_components} components")
                
                unmixer = SpectralUnmixer(
                    method=self.spectral_settings.get('method', 'nmf'),
                    n_components=n_components,
                    normalize=self.spectral_settings.get('normalize', True),
                    **{k: v for k, v in self.spectral_settings.items() 
                       if k not in ['method', 'n_components', 'normalize', 'enabled', 'auto_components']}
                )
                
                components, abundance_maps = unmixer.fit(hspy_data)
                return components, abundance_maps
            
        except Exception as e:
            self.logger.error(f"Spectral unmixing failed: {e}")
            raise

    def _create_summary_images(self, hspy_data: np.ndarray, components: np.ndarray, 
                             abundance_maps: np.ndarray) -> List[bytes]:
        """
        Create summary images for LLM analysis including:
        - Mean spectrum
        - Component spectra
        - Abundance maps
        """
        import matplotlib.pyplot as plt
        
        images = []
        
        try:
            # 1. Mean spectrum and component spectra
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Mean spectrum
            mean_spectrum = np.mean(hspy_data.reshape(-1, hspy_data.shape[-1]), axis=0)
            ax1.plot(mean_spectrum, 'k-', linewidth=2, label='Mean Spectrum')
            ax1.set_xlabel('Channel')
            ax1.set_ylabel('Intensity')
            ax1.set_title('Mean Spectrum')
            ax1.grid(True, alpha=0.3)
            
            # Component spectra
            for i, component in enumerate(components):
                ax2.plot(component, label=f'Component {i+1}')
            ax2.set_xlabel('Channel')
            ax2.set_ylabel('Intensity')
            ax2.set_title('Unmixed Component Spectra')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to bytes
            buf = BytesIO()
            plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
            buf.seek(0)
            images.append(buf.getvalue())
            plt.close()
            
            # 2. Abundance maps
            n_components = abundance_maps.shape[-1]
            n_cols = min(4, n_components)
            n_rows = (n_components + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n_components):
                row, col = i // n_cols, i % n_cols
                im = axes[row, col].imshow(abundance_maps[..., i], cmap='viridis')
                axes[row, col].set_title(f'Component {i+1} Abundance')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(n_components, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Save to bytes
            buf = BytesIO()
            plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
            buf.seek(0)
            images.append(buf.getvalue())
            plt.close()
            
            self.logger.info(f"Created {len(images)} summary images for LLM analysis")
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to create summary images: {e}")
            return []

    def analyze_hyperspectral_data(self, data_path: str, system_info: Dict[str, Any] | None = None,
                                 analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze hyperspectral data for materials characterization.
        
        Args:
            data_path: Path to hyperspectral data file
            system_info: Additional metadata about the sample/experiment
            analysis_type: Type of analysis ("general", "phase_mapping", "defect_analysis", etc.)
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load hyperspectral data
            self.logger.info(f"Loading hyperspectral data from: {data_path}")
            hspy_data = self._load_hyperspectral_data(data_path)
            
            components = None
            abundance_maps = None
            
            # Perform spectral unmixing if enabled
            if self.run_spectral_unmixing:
                components, abundance_maps = self._perform_spectral_unmixing(hspy_data, system_info)
            
            # Create summary images for LLM
            summary_images = []
            if components is not None and abundance_maps is not None:
                summary_images = self._create_summary_images(hspy_data, components, abundance_maps)
            
            # Build prompt for LLM analysis
            prompt_parts = [SPECTROSCOPY_ANALYSIS_INSTRUCTIONS]
            
            # Add analysis type context
            if analysis_type != "general":
                prompt_parts.append(f"\n\nSpecific Analysis Focus: {analysis_type}")
                prompt_parts.append("Please tailor your analysis to emphasize aspects relevant to this focus area.")
            
            prompt_parts.append(f"\n\nHyperspectral Data Information:")
            prompt_parts.append(f"- Data shape: {hspy_data.shape}")
            prompt_parts.append(f"- Number of spectral channels: {hspy_data.shape[-1]}")
            prompt_parts.append(f"- Spatial dimensions: {hspy_data.shape[:2]}")
            
            if components is not None:
                prompt_parts.append(f"- Spectral unmixing method: {self.spectral_settings.get('method', 'nmf').upper()}")
                prompt_parts.append(f"- Number of components found: {components.shape[0]}")
            
            # Add summary images
            if summary_images:
                for i, img_bytes in enumerate(summary_images):
                    prompt_parts.append(f"\n\nSummary Image {i+1}:")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": img_bytes})
            
            # Add system information if provided
            if system_info:
                prompt_parts.append("\n\nAdditional System Information:")
                if isinstance(system_info, dict):
                    prompt_parts.append(json.dumps(system_info, indent=2))
                else:
                    prompt_parts.append(str(system_info))
            
            prompt_parts.append("\n\nProvide your analysis in the requested JSON format.")
            
            # Send to LLM
            self.logger.info("Sending hyperspectral analysis request to LLM...")
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            
            result_json, error_dict = self._parse_llm_response(response)
            
            if error_dict:
                return error_dict
            
            # Enhance results with quantitative data
            if result_json and components is not None:
                result_json["quantitative_analysis"] = {
                    "spectral_unmixing_method": self.spectral_settings.get('method', 'nmf'),
                    "n_components": components.shape[0],
                    "component_spectra_shape": components.shape,
                    "abundance_maps_shape": abundance_maps.shape,
                    "data_statistics": {
                        "mean_intensity": float(np.mean(hspy_data)),
                        "std_intensity": float(np.std(hspy_data)),
                        "min_intensity": float(np.min(hspy_data)),
                        "max_intensity": float(np.max(hspy_data)),
                    }
                }
            
            self.logger.info("Hyperspectral analysis completed successfully")
            return result_json
            
        except Exception as e:
            self.logger.exception(f"Error during hyperspectral analysis: {e}")
            return {"error": "Hyperspectral analysis failed", "details": str(e)}
    

    def _save_component_reasoning(self, step_name: str, reasoning_data: dict):
        """Save LLM reasoning for human review."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(reasoning_data, f, indent=2)
        
        print(f"💾 Saved {step_name} reasoning: {filename}")
        return filepath

    def _save_component_comparison_plot(self, test_images: list, initial_estimate: int):
        """Save the component comparison plots for human review."""
        if not test_images:
            return
            
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual test result images
        for i, result in enumerate(test_images):
            filename = f"component_test_{result['n_components']}comp_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(result['image'])
            
            print(f"📊 Saved component test ({result['n_components']} components): {filename}")
    
    def _create_final_results_summary(self, final_n_components: int, components: np.ndarray, 
                                    abundance_maps: np.ndarray, reasoning_log: dict):
        """Create comprehensive summary for human review."""
        from datetime import datetime
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save detailed reasoning log
        reasoning_file = f"component_selection_log_{timestamp}.json"
        reasoning_path = os.path.join(self.output_dir, reasoning_file)
        
        with open(reasoning_path, 'w') as f:
            json.dump(reasoning_log, f, indent=2)
        
        # 2. Create clean final results plot
        fig = plt.figure(figsize=(16, 10))
        
        # Top section: Component spectra
        ax1 = plt.subplot(2, 1, 1)
        for i in range(final_n_components):
            plt.plot(components[i], label=f'Component {i+1}', linewidth=2)
        plt.title(f'Final Spectral Components (n={final_n_components})')
        plt.xlabel('Channel')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bottom section: Abundance maps
        n_cols = min(4, final_n_components)
        n_rows = (final_n_components + n_cols - 1) // n_cols
        
        for i in range(final_n_components):
            row = i // n_cols
            col = i % n_cols
            ax = plt.subplot(2, n_cols, n_cols + 1 + i)
            im = plt.imshow(abundance_maps[..., i], cmap='viridis')
            plt.title(f'Component {i+1}')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save final results plot
        results_file = f"final_results_{timestamp}.png"
        results_path = os.path.join(self.output_dir, results_file)
        plt.savefig(results_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved final results: {results_file}")
        print(f"📝 Saved reasoning log: {reasoning_file}")
        
        return results_path, reasoning_path


    def generate_analysis_claims(self, data_path: str, system_info: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Generate scientific claims from hyperspectral analysis for literature comparison.
        Similar to microscopy agent's analyze_microscopy_image_for_claims method.
        """
        # This would use modified instructions focused on claim generation
        # Implementation would be similar to analyze_hyperspectral_data but with different prompt
        pass