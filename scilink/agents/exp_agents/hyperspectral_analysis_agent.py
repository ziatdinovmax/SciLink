import json
import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
from datetime import datetime

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

from .base_agent import BaseAnalysisAgent
from .utils import convert_numpy_to_jpeg_bytes, normalize_and_convert_to_image_bytes
from .utils import create_multi_abundance_overlays
from .instruct import (
    SPECTROSCOPY_ANALYSIS_INSTRUCTIONS, 
    COMPONENT_INITIAL_ESTIMATION_INSTRUCTIONS,
    COMPONENT_VISUAL_COMPARISON_INSTRUCTIONS,
    SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
    SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS
)

from .human_feedback import SimpleFeedbackMixin

from atomai.stat import SpectralUnmixer


class HyperspectralAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing hyperspectral/spectroscopy data using generative AI models.
    Refactored to inherit from BaseAnalysisAgent and includes LLM-guided spectral unmixing.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-pro-preview-06-05", 
                 local_model: str = None,
                 spectral_unmixing_settings: dict | None = None,
                 output_dir: str = "spectroscopy_output",
                 enable_human_feedback: bool = False):
        super().__init__(api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        
        # Spectral unmixing settings
        default_settings = {
            'method': 'nmf',
            'n_components': 4,
            'normalize': True,
            'enabled': True,
            'auto_components': True,
            'max_iter': 500
        }
        self.spectral_settings = spectral_unmixing_settings if spectral_unmixing_settings else default_settings
        self.run_spectral_unmixing = self.spectral_settings.get('enabled', True)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _load_hyperspectral_data(self, data_path: str) -> np.ndarray:
        """
        Load hyperspectral data from numpy array.
        Assumes data_path points to a .npy file.
        """
        try:
            if not data_path.endswith('.npy'):
                raise ValueError(f"Expected .npy file, got: {data_path}")
            
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

    def _load_metadata_from_json(self, data_path: str) -> dict:
        """
        Load metadata from companion JSON file.
        Assumes JSON file has same name as .npy file but with .json extension.
        """
        import json
        import os
        
        # Get JSON file path (same name as .npy but with .json extension)
        base_path = os.path.splitext(data_path)[0]
        json_path = f"{base_path}.json"
        
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.info(f"Loaded metadata from: {json_path}")
                return metadata
            else:
                self.logger.warning(f"No metadata file found at: {json_path}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to load metadata from {json_path}: {e}")
            return {}

    def _llm_estimate_components_from_system(self, hspy_data: np.ndarray, 
                                           system_info: Dict[str, Any] = None) -> int:
        """
        Step 1: LLM estimates optimal number of components based on system description.
        """
        try:
            self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: COMPONENT PARAMETER ESTIMATION -------------------- 🤖\n")
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
            
            result_json, error_dict = self._parse_llm_response(response)  # Using base class method
            
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

    def _create_nmf_summary_plot(self, components: np.ndarray, abundance_maps: np.ndarray, 
                           n_comp: int, system_info: Dict[str, Any] = None) -> bytes:
        """
        Create a single summary plot showing all components and abundance maps.
        Now includes proper energy axis labeling.
        """
        import matplotlib.pyplot as plt
        
        try:
            # Create energy axis
            n_channels = components.shape[1]
            energy_axis, xlabel, has_energy_info = self._create_energy_axis(n_channels, system_info)
            
            # Create figure with 2 rows: spectra on top, abundance maps on bottom
            fig, axes = plt.subplots(2, n_comp, figsize=(n_comp * 3, 6))
            
            if n_comp == 1:
                axes = axes.reshape(2, 1)
            
            for i in range(n_comp):
                # Top row: Component spectra with proper energy axis
                axes[0, i].plot(energy_axis, components[i, :], 'b-', linewidth=1.5)
                axes[0, i].set_title(f'NMF Component {i+1}', fontsize=10)
                axes[0, i].set_xlabel(xlabel)
                if i == 0:
                    axes[0, i].set_ylabel('Intensity')
                axes[0, i].grid(True, alpha=0.3)
                
                # Bottom row: Abundance maps
                im = axes[1, i].imshow(abundance_maps[..., i], cmap='seismic', aspect='auto')
                axes[1, i].set_title(f'Abundance Map {i+1}', fontsize=10)
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # Add energy info to title if available
            if has_energy_info:
                plt.suptitle(f'NMF Analysis: {n_comp} Components (Energy Calibrated)', fontsize=14, y=0.95)
            else:
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
            self.logger.info("\n\n🤖 ------------------ ANALYSIS AGENT STEP: DECIDING ON THE FINAL NUMBER OF COMPONENTS ------------------ 🤖\n")
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
            
            result_json, error_dict = self._parse_llm_response(response)  # Using base class method
            
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
        
        # Step 1: LLM initial estimate based on system description
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
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: COMPONENT TESTING -------------------- 🤖\n")
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
                # Pass system_info here!
                summary_image = self._create_nmf_summary_plot(components, abundance_maps, n_comp, system_info)
                
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
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: FINAL SPECTRAL UNMIXING -------------------- 🤖\n")
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

    def _create_component_abundance_pairs(self, components: np.ndarray, abundance_maps: np.ndarray, 
                                    system_info: Dict[str, Any] = None, save_plots: bool = True) -> List[bytes]:
        """
        Create individual component-abundance pair images with consistent y-scaling for final analysis.
        Each pair shows one component spectrum alongside its abundance map.
        
        Args:
            components: Array of shape (n_components, n_channels)
            abundance_maps: Array of shape (height, width, n_components)
            system_info: System metadata for energy axis creation
            save_plots: Whether to save plots to disk for inspection
            
        Returns:
            List of image bytes, one for each component-abundance pair
        """
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        pair_images = []
        n_components = components.shape[0]
        saved_files = []
        
        try:
            # Create energy axis using system info
            n_channels = components.shape[1]
            energy_axis, xlabel, has_energy_info = self._create_energy_axis(n_channels, system_info)
            
            # Calculate global y-scale for consistent spectrum scaling
            global_min = np.min(components)
            global_max = np.max(components)
            y_margin = (global_max - global_min) * 0.05  # 5% margin
            y_limits = (global_min - y_margin, global_max + y_margin)
            
            self.logger.info(f"Creating {n_components} component-abundance pairs with consistent y-scale: {y_limits}")
            
            # Create timestamp for saved files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i in range(n_components):
                fig, (ax_spectrum, ax_abundance) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Left plot: Component spectrum with consistent y-scaling
                ax_spectrum.plot(energy_axis, components[i, :], 'b-', linewidth=2)
                ax_spectrum.set_ylim(y_limits)  # Apply consistent y-scaling
                ax_spectrum.set_xlabel(xlabel)
                ax_spectrum.set_ylabel('Intensity')
                ax_spectrum.set_title(f'Component {i+1} Spectrum')
                ax_spectrum.grid(True, alpha=0.3)
                
                # Add energy range info to title if available
                if has_energy_info:
                    energy_range = f" ({energy_axis[0]:.0f}-{energy_axis[-1]:.0f} {xlabel.split('(')[1].rstrip(')')})"
                    ax_spectrum.set_title(f'Component {i+1} Spectrum{energy_range}')
                
                # Right plot: Abundance map with proper aspect ratio
                im = ax_abundance.imshow(abundance_maps[..., i], cmap='viridis', aspect='equal')
                ax_abundance.set_title(f'Component {i+1} Abundance Map')
                ax_abundance.axis('off')
                
                # Add colorbar for abundance map
                plt.colorbar(im, ax=ax_abundance, fraction=0.046, pad=0.04, label='Abundance')
                
                # Add overall title with y-scale info
                fig.suptitle(f'Component {i+1} Analysis (Y-scale: {y_limits[0]:.2e} to {y_limits[1]:.2e})', 
                            fontsize=12, y=0.98)
                
                plt.tight_layout()
                
                # Save to disk if requested
                if save_plots:
                    filename = f"component_{i+1}_pair_{timestamp}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    saved_files.append(filepath)
                    self.logger.info(f" Saved component pair {i+1}: {filename}")
                
                # Save to bytes for LLM
                buf = BytesIO()
                plt.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
                buf.seek(0)
                pair_images.append(buf.getvalue())
                plt.close()
                
            self.logger.info(f"Successfully created {len(pair_images)} component-abundance pair images")
            
            if save_plots:
                print(f"\n📊 Component-abundance pairs saved to: {self.output_dir}/")
                print("    These are the exact plots being sent to the LLM for analysis")
            
            return pair_images
            
        except Exception as e:
            self.logger.error(f"Failed to create component-abundance pairs: {e}")
            return []

    def _validate_structure_inputs(self, structure_image_path: str = None, 
                                 structure_system_info: Dict[str, Any] = None) -> tuple[str, Dict[str, Any]]:
        """
        Validate and clean up structure inputs, ensuring consistency.
        
        Returns:
            tuple: (validated_structure_image_path, validated_structure_system_info)
        """
        # If structure_system_info provided without structure_image_path, warn and ignore
        if structure_system_info and not structure_image_path:
            self.logger.warning("structure_system_info provided but no structure_image_path - ignoring metadata")
            structure_system_info = None
        
        # Validate structure image path exists
        if structure_image_path and not os.path.exists(structure_image_path):
            self.logger.warning(f"Structure image not found: {structure_image_path}")
            structure_image_path = None
            structure_system_info = None  # Also clear metadata if image doesn't exist
        
        return structure_image_path, structure_system_info

    def analyze_hyperspectral_data_for_claims(self, data_path: str, metadata_path: Dict[str, Any] | None = None,
                                            structure_image_path: str = None, structure_system_info: Dict[str, Any] = None
                                            ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data to generate scientific claims for literature comparison.
        
        Args:
            data_path: Path to hyperspectral data file (.npy)
            metadata_path: Dictionary with experimental metadata OR path to JSON file
            structure_image_path: Optional path to 2D greyscale structure image for context
            structure_system_info: Optional metadata for the structure image
            
        Returns:
            Dictionary containing detailed analysis and scientific claims
        """
        try:
            # Validate structure inputs
            structure_image_path, structure_system_info = self._validate_structure_inputs(
                structure_image_path, structure_system_info
            )
            
            # Handle metadata_path properly - it can be a dict or a string path
            if isinstance(metadata_path, dict):
                # Already a dictionary, use it directly
                system_info = metadata_path
            elif isinstance(metadata_path, str):
                # It's a file path, load from JSON
                system_info = self._load_metadata_from_json(metadata_path)
            elif metadata_path is None:
                # Try to load companion JSON file based on data_path
                system_info = self._load_metadata_from_json(data_path)
            else:
                self.logger.warning(f"Invalid metadata_path type: {type(metadata_path)}")
                system_info = {}

            # Use the shared analysis workflow but with claims-specific instructions
            result = self._analyze_hyperspectral_data_base(
                data_path=data_path,
                system_info=system_info,
                instruction_prompt=SPECTROSCOPY_CLAIMS_INSTRUCTIONS,
                analysis_type="claims",
                structure_image_path=structure_image_path,
                structure_system_info=structure_system_info
            )
            
            if "error" in result:
                return result
            
            # Extract claims-specific fields
            detailed_analysis = result.get("detailed_analysis", "Analysis not provided by LLM.")
            scientific_claims = result.get("scientific_claims", [])
            
            # Use base class validation method
            valid_claims = self._validate_scientific_claims(scientific_claims)
            
            # Log results
            if not valid_claims and detailed_analysis != "Analysis not provided by LLM.":
                self.logger.warning("Spectroscopic claims analysis successful but no valid claims found.")
            elif not valid_claims:
                self.logger.warning("LLM call did not yield valid claims or analysis text for spectroscopic workflow.")
            else:
                self.logger.info(f"Successfully generated {len(valid_claims)} scientific claims from spectroscopic analysis.")
            
            initial_result = {"detailed_analysis": detailed_analysis, "scientific_claims": valid_claims}
            return self._apply_feedback_if_enabled(
                initial_result,
                data_path=data_path,
                system_info=system_info,
                structure_image_path=structure_image_path
            )
            
        except Exception as e:
            self.logger.exception(f"Error during hyperspectral claims analysis: {e}")
            return {"error": "Hyperspectral claims analysis failed", "details": str(e)}
        
    def _get_claims_instruction_prompt(self) -> str:
        return SPECTROSCOPY_CLAIMS_INSTRUCTIONS

    def _analyze_hyperspectral_data_base(self, data_path: str, system_info: Dict[str, Any] | None = None,
                                    instruction_prompt: str = None, analysis_type: str = "standard",
                                    structure_image_path: str = None, structure_system_info: Dict[str, Any] = None
                                    ) -> Dict[str, Any]:
        """
        Base method for hyperspectral data analysis
        """
        self._clear_stored_images()
        # Use base class method for system info handling
        if isinstance(system_info, str):
            system_info = self._handle_system_info(system_info)
        elif system_info is None:
            system_info = self._load_metadata_from_json(data_path)
        
        # Use base class method for structure_system_info handling
        if isinstance(structure_system_info, str) and os.path.exists(structure_system_info):
            structure_system_info = self._handle_system_info(structure_system_info)

        # Load hyperspectral data
        analysis_desc = "claims generation" if analysis_type == "claims" else "analysis"
        self.logger.info(f"Loading hyperspectral data for {analysis_desc}: {data_path}")
        hspy_data = self._load_hyperspectral_data(data_path)
        
        components = None
        abundance_maps = None
        
        # Perform spectral unmixing if enabled
        if self.run_spectral_unmixing:
            self.logger.info(f"Performing spectral unmixing for {analysis_desc}...")
            components, abundance_maps = self._perform_spectral_unmixing(hspy_data, system_info)
        
        # Create component-abundance pairs for final analysis
        component_pair_images = []
        if components is not None and abundance_maps is not None:
            self.logger.info("Creating component-abundance pairs for LLM analysis...")
            component_pair_images = self._create_component_abundance_pairs(components, abundance_maps, system_info)
        
        analysis_images = []
        for i, pair_img_bytes in enumerate(component_pair_images):
            analysis_images.append({
                "label": f"Component {i+1} Pair (Spectrum + Abundance Map)",
                "data": pair_img_bytes
            })


        # Build prompt for LLM analysis
        prompt_parts = [instruction_prompt or SPECTROSCOPY_ANALYSIS_INSTRUCTIONS]
        
        # Add data information (shared between both analysis types)
        energy_info_text = self._build_energy_info_for_prompt(hspy_data, system_info)
        prompt_parts.append(f"\n\nHyperspectral Data Information:\n{energy_info_text}")
        
        # Add spectral unmixing information if available
        if components is not None:
            prompt_parts.append(f"- Spectral unmixing method: {self.spectral_settings.get('method', 'nmf').upper()}")
            prompt_parts.append(f"- Number of spectral components identified: {components.shape[0]}")
            prompt_parts.append(f"- Component spectra shape: {components.shape}")
            prompt_parts.append(f"- Spatial abundance maps shape: {abundance_maps.shape}")
        else:
            prompt_parts.append("- No spectral unmixing performed")

        # Add structure image and overlays if provided
        overlay_bytes = None
        if structure_image_path and os.path.exists(structure_image_path):
            try:
                from .utils import convert_numpy_to_jpeg_bytes, load_image
                import cv2                
                
                self.logger.info(f"Loading structural context image: {structure_image_path}")
                structure_img = load_image(structure_image_path)
                
                # For the structural context image, we want minimal processing to preserve
                # the original contrast, unlike primary microscopy analysis. We will just
                # ensure it's a standard 8-bit grayscale image without aggressive enhancement.
                if len(structure_img.shape) == 3:
                    structure_img_gray = cv2.cvtColor(structure_img, cv2.COLOR_RGB2GRAY)
                else:
                    structure_img_gray = structure_img # Already grayscale

                # Try to create abundance overlays first
                if abundance_maps is not None:
                    overlay_bytes = self._create_structure_abundance_overlays(
                        structure_img_gray, abundance_maps, system_info
                    )
                
                if overlay_bytes:
                    # Use overlays (which include the original structure image as first panel)
                    prompt_parts.append("\n\n**Structure-Abundance Correlation Analysis:**")
                    prompt_parts.append("These overlays show where each NMF component is most abundant (top 15% of values) overlaid on the structural image. The first panel shows the original structural image for reference. Each subsequent panel shows the same structure with colored overlays indicating where each component is most concentrated. Look for spatial correlations between these (thresholded) abundance patterns and structural features.")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": overlay_bytes})
                    self.logger.info("Added abundance overlays to LLM prompt (includes structure image)")
                else:
                    # Fallback: use standalone structure image if overlays failed
                    structure_img_bytes = convert_numpy_to_jpeg_bytes(structure_img_gray)
                    prompt_parts.append("\n\n**Structural Context Image for Correlation:**")
                    prompt_parts.append("This is a structural image providing spatial context. Try to correlate the spectroscopic components and their abundance maps with the spatial features in this image.")
                    prompt_parts.append({"mime_type": "image/jpeg", "data": structure_img_bytes})
                    self.logger.info("Added standalone structure image to analysis prompt (overlays failed)")
                
                # Add structure system info only if structure image was successfully processed
                structure_info_section = self._build_system_info_prompt_section(structure_system_info)
                if structure_info_section:
                    prompt_parts.append("\n\nStructural Context Metadata:")
                    prompt_parts.append(structure_info_section.replace("\n\nAdditional System Information (Metadata):\n", ""))
                
                self.logger.info("Successfully processed structure image for analysis")
                
            except Exception as e:
                self.logger.warning(f"Failed to load structure image: {e}")
        elif structure_image_path:
            self.logger.warning(f"Structure image not found: {structure_image_path}")
        
        # Add component-abundance pairs for LLM interpretation
        if component_pair_images:
            prompt_parts.append("\n\nSpectral Component Analysis (Individual Component-Abundance Pairs):")
            prompt_parts.append("Each component is shown with its spectrum (left) and spatial abundance map (right).")
            prompt_parts.append("All component spectra use the same y-axis scale for direct comparison.")
            
            for i, pair_img_bytes in enumerate(component_pair_images):
                prompt_parts.append(f"\n\nComponent {i+1} Pair (Spectrum + Abundance Map):")
                prompt_parts.append({"mime_type": "image/jpeg", "data": pair_img_bytes})
        else:
            prompt_parts.append("\n\n(No spectroscopic component analysis images available)")
        
        # Use base class method for system info prompt section
        system_info_section = self._build_system_info_prompt_section(system_info)
        if system_info_section:
            prompt_parts.append(system_info_section)
        
        prompt_parts.append("\n\nProvide your analysis in the requested JSON format.")

        analysis_metadata = {
            "data_path": data_path,
            "system_info": system_info,
            "spectral_unmixing_enabled": self.run_spectral_unmixing,
            "n_components": components.shape[0] if components is not None else 0,
            "structure_image_included": structure_image_path is not None,
            "num_stored_images": len(analysis_images)
        }
        self._store_analysis_images(analysis_images, analysis_metadata)
        
        # Send to LLM for analysis
        self.logger.info("\n\n🤖 -------------------- ANALYSIS AGENT STEP: INTERPRETING RESULTS -------------------- 🤖\n")
        self.logger.info(f"Sending hyperspectral {analysis_desc} request to LLM with {len(component_pair_images)} component pairs...")
        response = self.model.generate_content(
            contents=prompt_parts,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
        )
        
        # Use base class method for parsing LLM response
        result_json, error_dict = self._parse_llm_response(response)
        
        if error_dict:
            return error_dict
        
        if result_json is None:
            return {"error": f"Hyperspectral {analysis_desc} failed unexpectedly after LLM processing."}
        
        # Add quantitative data if available
        if components is not None and analysis_type == "standard":
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
        
        return result_json

    def _save_claims_results(self, claims_result: Dict[str, Any], output_filename: str = None) -> str:
        """
        Save claims analysis results to file for further processing.
        
        Args:
            claims_result: Result dictionary from generate_analysis_claims
            output_filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if output_filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"spectroscopy_claims_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(claims_result, f, indent=2, default=str)
            
            self.logger.info(f"Claims analysis results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save claims results: {e}")
            raise

    def analyze_hyperspectral_data(self, data_path: str, metadata_path: str,
                                   structure_image_path: str = None,
                                   structure_system_info: Dict[str, Any] = None
                                   ) -> Dict[str, Any]:
        """
        Analyze hyperspectral data for materials characterization.
        
        Args:
            data_path: Path to hyperspectral data file
            metadata_path: Additional metadata about the sample/experiment
            structure_image_path: Optional path to 2D greyscale structure image for context
            structure_system_info: Optional metadata for the structure image
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Validate structure inputs
            structure_image_path, structure_system_info = self._validate_structure_inputs(
                structure_image_path, structure_system_info
            )
            
            # Load system info from metadata path
            system_info = self._load_metadata_from_json(metadata_path)
            
            # Use the shared base method with standard analysis instructions
            result = self._analyze_hyperspectral_data_base(
                data_path=data_path,
                system_info=system_info,
                instruction_prompt=SPECTROSCOPY_ANALYSIS_INSTRUCTIONS,
                analysis_type="standard",
                structure_image_path=structure_image_path,
                structure_system_info=structure_system_info
            )
            
            if "error" in result:
                return result
            
            self.logger.info("Hyperspectral analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.exception(f"Error during hyperspectral analysis: {e}")
            return {"error": "Hyperspectral analysis failed", "details": str(e)}

    def _build_energy_info_for_prompt(self, hspy_data: np.ndarray, system_info: dict = None) -> str:
        """Build energy information string for LLM prompt."""
        
        info_lines = [
            f"- Data shape: {hspy_data.shape}",
            f"- Spatial dimensions: {hspy_data.shape[:2]}"
        ]
        
        if system_info and "energy_range" in system_info:
            energy_info = system_info["energy_range"]
            start = energy_info.get("start")
            end = energy_info.get("end")
            units = energy_info.get("units", "eV")
            
            if start is not None and end is not None:
                dispersion = (end - start) / (hspy_data.shape[-1] - 1)
                info_lines.extend([
                    f"- Energy range: {start} to {end} {units}",
                    f"- Number of energy channels: {hspy_data.shape[-1]}",
                    f"- Energy dispersion: {dispersion:.3f} {units}/channel"
                ])
            else:
                info_lines.append(f"- Number of spectral channels: {hspy_data.shape[-1]} (energy not calibrated)")
        else:
            info_lines.append(f"- Number of spectral channels: {hspy_data.shape[-1]} (energy axis not provided)")
        
        return "\n".join(info_lines)

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

    def _create_structure_abundance_overlays(self, structure_img_gray: np.ndarray, 
                                           abundance_maps: np.ndarray, 
                                           save_plots: bool = True) -> bytes:
        """
        Create abundance overlays on structure image for LLM correlation analysis.
        
        Args:
            structure_img_gray: 2D grayscale structure image
            abundance_maps: NMF abundance maps (height, width, n_components)
            save_plots: Whether to save plots to disk for inspection
            
        Returns:
            Image bytes for LLM prompt, or None if failed
        """
        try:
            self.logger.info(f"Creating abundance overlays for {abundance_maps.shape[2]} components")
            
            # Create multi-component overlay using the utility function
            overlay_bytes = create_multi_abundance_overlays(
                structure_image=structure_img_gray,
                abundance_maps=abundance_maps,
                threshold_percentile=85.0,  # Show top 15% of abundance values
                alpha=0.5,
                use_simple_colors=True  # Use solid colors for clearer LLM analysis
            )
            
            # Save the overlay that goes to LLM (following existing pattern)
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"abundance_overlays_{timestamp}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(overlay_bytes)
                
                self.logger.info(f" Saved abundance overlays: {filename}")
            
            return overlay_bytes
            
        except Exception as e:
            self.logger.warning(f"Failed to create abundance overlays: {e}")
            return None
    
    def _create_final_results_summary(self, final_n_components: int, components: np.ndarray, 
                                    abundance_maps: np.ndarray, reasoning_log: dict):
        """Create comprehensive summary for human review."""
        from datetime import datetime
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create energy axis
        n_channels = components.shape[1]
        system_info = reasoning_log.get('system_info')  # Get from reasoning log
        energy_axis, xlabel, has_energy_info = self._create_energy_axis(n_channels, system_info)
        
        # 1. Save detailed reasoning log
        reasoning_file = f"component_selection_log_{timestamp}.json"
        reasoning_path = os.path.join(self.output_dir, reasoning_file)
        
        with open(reasoning_path, 'w') as f:
            json.dump(reasoning_log, f, indent=2)
        
        # 2. Create clean final results plot
        fig = plt.figure(figsize=(16, 10))
        
        # Top section: Component spectra
        ax_spectra = plt.subplot(2, 1, 1)
        for i in range(final_n_components):
            plt.plot(energy_axis, components[i], label=f'Component {i+1}', linewidth=2)
        plt.title(f'Final Spectral Components (n={final_n_components})')
        plt.xlabel(xlabel)
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bottom section: Abundance maps with proper grid
        n_cols = min(4, final_n_components)
        n_rows = (final_n_components + n_cols - 1) // n_cols
        
        # Create abundance map subplots
        for i in range(final_n_components):
            row = i // n_cols
            col = i % n_cols
            ax = plt.subplot(2 * n_rows, n_cols, n_cols + row * n_cols + col + 1)
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

    def _create_energy_axis(self, n_channels: int, system_info: dict = None) -> tuple[np.ndarray, str, bool]:
        """
        Create energy axis from system_info if available, otherwise use channel indices.
        
        Returns:
            tuple: (energy_axis, xlabel, has_energy_info)
        """
        if system_info and "energy_range" in system_info:
            energy_info = system_info["energy_range"]
            
            if "start" in energy_info and "end" in energy_info:
                start = energy_info["start"]
                end = energy_info["end"]
                units = energy_info.get("units", "eV")  # Default to eV if not specified
                
                # Simple linear conversion
                energy_axis = np.linspace(start, end, n_channels)
                xlabel = f"Energy ({units})"
                has_energy_info = True
                
                self.logger.info(f"Using energy axis: {start} to {end} {units}")
                return energy_axis, xlabel, has_energy_info
        
        # Fallback: channel indices
        energy_axis = np.arange(n_channels)
        xlabel = "Channel"
        has_energy_info = False
        
        self.logger.info("Using channel indices (no energy range provided)")
        return energy_axis, xlabel, has_energy_info
    
    def _get_measurement_recommendations_prompt(self) -> str:
        return SPECTROSCOPY_MEASUREMENT_RECOMMENDATIONS_INSTRUCTIONS