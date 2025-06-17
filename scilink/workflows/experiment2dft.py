import os
import sys
import logging
import json
from io import StringIO
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import other workflows
from .microscopy_novelty_workflow import MicroscopyNoveltyAssessmentWorkflow
from .spectroscopy_novelty_workflow import SpectroscopyNoveltyAssessmentWorkflow
from .dft_recommendation_workflow import DFTRecommendationsWorkflow
from .dft_workflow import DFTWorkflow

# Import auth for API key management
from ..auth import get_api_key, APIKeyNotFoundError


class Experimental2DFT:
    """
    Complete pipeline from experimental data (microscopy or spectroscopy) to atomic structures.
    
    This workflow orchestrates the entire process:
    1. Experimental data analysis with novelty assessment (microscopy or spectroscopy)
    2. DFT structure recommendations based on novelty
    3. Interactive or automatic structure selection
    4. Atomic structure generation with validation
    5. Complete results summary
    
    Supports both microscopy images and hyperspectral/spectroscopy data.
    """
    
    def __init__(self, 
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 mp_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 generator_model: str = "gemini-2.5-pro-preview-06-05",
                 validator_model: str = "gemini-2.5-pro-preview-06-05",
                 output_dir: str = "complete_experimental_pipeline_output",
                 max_wait_time: int = 400,
                 max_refinement_cycles: int = 2,
                 script_timeout: int = 300,
                 microscopy_analysis_enabled: bool = True,
                 spectroscopy_analysis_enabled: bool = True):
        
        # Auto-discover API keys
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        if futurehouse_api_key is None:
            futurehouse_api_key = get_api_key('futurehouse')
            if not futurehouse_api_key:
                import warnings
                warnings.warn(
                    "FutureHouse API key not found. Literature search will be disabled."
                )
        
        if mp_api_key is None:
            mp_api_key = get_api_key('materials_project')
            # Optional
        
        # Setup logging
        self.log_capture = StringIO()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.StreamHandler(self.log_capture)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Store configuration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize sub-workflows
        self.microscopy_workflow = MicroscopyNoveltyAssessmentWorkflow(
            google_api_key=google_api_key,
            futurehouse_api_key=futurehouse_api_key,
            analysis_model=analysis_model,
            output_dir=str(self.output_dir / "microscopy_novelty_assessment"),
            max_wait_time=max_wait_time,
            analysis_enabled=microscopy_analysis_enabled
        )
        
        self.spectroscopy_workflow = SpectroscopyNoveltyAssessmentWorkflow(
            google_api_key=google_api_key,
            futurehouse_api_key=futurehouse_api_key,
            analysis_model=analysis_model,
            output_dir=str(self.output_dir / "spectroscopy_novelty_assessment"),
            max_wait_time=max_wait_time,
            spectral_unmixing_enabled=spectroscopy_analysis_enabled
        )
        
        self.dft_rec_workflow = DFTRecommendationsWorkflow(
            google_api_key=google_api_key,
            analysis_model=analysis_model,
            output_dir=str(self.output_dir / "dft_recommendations")
        )
        
        self.dft_workflow = DFTWorkflow(
            google_api_key=google_api_key,
            futurehouse_api_key=futurehouse_api_key,
            mp_api_key=mp_api_key,
            generator_model=generator_model,
            validator_model=validator_model,
            output_dir=str(self.output_dir / "generated_structures"),
            max_refinement_cycles=max_refinement_cycles,
            script_timeout=script_timeout
        )
        
        self.logger.info("Experimental2DFT workflow initialized")
    
    def run_complete_pipeline(self, 
                             data_path: Union[str, Path],
                             data_type: str,
                             system_info: Union[str, Path, Dict[str, Any]],
                             interactive: bool = True,
                             auto_select_top_n: int = 2,
                             max_structures: int = 5) -> Dict[str, Any]:
        """
        Run the complete experimental pipeline.
        
        Args:
            data_path: Path to experimental data (image for microscopy, .npy for spectroscopy)
            data_type: Type of data - either 'microscopy' or 'spectroscopy'
            system_info: Path to system metadata JSON file or dict
            interactive: Whether to allow interactive structure selection
            auto_select_top_n: Number of top recommendations to auto-select in non-interactive mode
            max_structures: Maximum number of structures to generate
            
        Returns:
            Complete pipeline results dictionary
        """
        
        # Validate data type
        if data_type not in ['microscopy', 'spectroscopy']:
            raise ValueError("data_type must be either 'microscopy' or 'spectroscopy'")
        
        pipeline_result = {
            "input_data": str(data_path),
            "data_type": data_type,
            "input_system_info": str(system_info) if not isinstance(system_info, dict) else system_info,
            "steps_completed": [],
            "final_status": "started",
            "generated_structures": []
        }
        
        self.logger.info(f"🚀 Starting Complete {data_type.title()} → Structure Pipeline")
        print(f"🚀 Starting Complete {data_type.title()} → Structure Pipeline")
        print("=" * 60)
        
        # Step 1: Novelty Assessment (data type specific)
        novelty_result = self._run_novelty_assessment(data_path, data_type, system_info)
        pipeline_result["novelty_assessment"] = novelty_result
        
        if novelty_result.get('final_status') != 'success':
            pipeline_result["final_status"] = f"failed_novelty: {novelty_result.get('final_status')}"
            return pipeline_result
        
        pipeline_result["steps_completed"].append("novelty_assessment")
        
        # Step 2: DFT Recommendations
        dft_rec_result = self._run_dft_recommendations(novelty_result, system_info)
        pipeline_result["dft_recommendations"] = dft_rec_result
        
        if dft_rec_result.get('status') != 'success':
            pipeline_result["final_status"] = f"failed_dft_recommendations: {dft_rec_result.get('message')}"
            return pipeline_result
            
        pipeline_result["steps_completed"].append("dft_recommendations")
        
        # Step 3: Structure Selection
        selected_structures = self._select_structures(
            dft_rec_result.get('recommendations', []),
            interactive=interactive,
            auto_select_top_n=auto_select_top_n,
            max_structures=max_structures
        )
        
        if not selected_structures:
            pipeline_result["final_status"] = "no_structures_selected"
            return pipeline_result
            
        pipeline_result["selected_recommendations"] = selected_structures
        pipeline_result["steps_completed"].append("structure_selection")
        
        # Step 4: Structure Generation
        structure_results = self._generate_structures(selected_structures)
        pipeline_result["generated_structures"] = structure_results
        pipeline_result["steps_completed"].append("structure_generation")
        
        # Step 5: Final Summary
        self._generate_final_summary(pipeline_result)
        
        # Determine final status
        successful_structures = [s for s in structure_results if s.get('success', False)]
        if successful_structures:
            pipeline_result["final_status"] = "success"
        else:
            pipeline_result["final_status"] = "no_successful_structures"
        
        # Save complete results
        self._save_pipeline_results(pipeline_result)
        
        return pipeline_result
    
    def run_microscopy_pipeline(self, 
                               image_path: Union[str, Path],
                               system_info: Union[str, Path, Dict[str, Any]],
                               interactive: bool = True,
                               auto_select_top_n: int = 2,
                               max_structures: int = 5) -> Dict[str, Any]:
        """
        Convenience method for microscopy data analysis.
        """
        return self.run_complete_pipeline(
            data_path=image_path,
            data_type='microscopy',
            system_info=system_info,
            interactive=interactive,
            auto_select_top_n=auto_select_top_n,
            max_structures=max_structures
        )
    
    def run_spectroscopy_pipeline(self, 
                                 data_path: Union[str, Path],
                                 system_info: Union[str, Path, Dict[str, Any]],
                                 interactive: bool = True,
                                 auto_select_top_n: int = 2,
                                 max_structures: int = 5,
                                 structure_image_path: Union[str, Path] = None,
                                 structure_system_info: Dict[str, Any] = None
                                 ) -> Dict[str, Any]:
        """
        Convenience method for spectroscopy data analysis.
        """

        # Store structure image info for the workflow
        self._temp_structure_image_path = structure_image_path
        self._temp_structure_system_info = structure_system_info

        return self.run_complete_pipeline(
            data_path=data_path,
            data_type='spectroscopy',
            system_info=system_info,
            interactive=interactive,
            auto_select_top_n=auto_select_top_n,
            max_structures=max_structures
        )
    
    def _run_novelty_assessment(self, 
                               data_path: Union[str, Path], 
                               data_type: str,
                               system_info: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Run novelty assessment based on data type."""
        
        print(f"\n📊 STEP 1: {data_type.title()} Analysis & Novelty Assessment")
        print("-" * 50)
        
        try:
            if data_type == 'microscopy':
                novelty_result = self.microscopy_workflow.run_complete_workflow(
                    image_path=str(data_path),
                    system_info=system_info
                )
            elif data_type == 'spectroscopy':
                structure_image_path = getattr(self, '_temp_structure_image_path', None)
                structure_system_info = getattr(self, '_temp_structure_system_info', None)
                novelty_result = self.spectroscopy_workflow.run_complete_workflow(
                    data_path=str(data_path),
                    system_info=system_info,
                    structure_image_path=structure_image_path,
                    structure_system_info=structure_system_info    
                )
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            if novelty_result.get('final_status') == 'success':
                claims_count = len(novelty_result.get("claims_generation", {}).get("claims", []))
                novel_count = len(novelty_result.get("novelty_assessment", {}).get("potentially_novel", []))
                
                print(f"✅ {data_type.title()} novelty assessment completed")
                print(f"   📋 Generated {claims_count} claims")
                print(f"   🔍 Found {novel_count} potentially novel findings")
            else:
                print(f"❌ {data_type.title()} novelty assessment failed: {novelty_result.get('final_status')}")
            
            return novelty_result
            
        except Exception as e:
            self.logger.exception(f"Failed in {data_type} novelty assessment step")
            return {"final_status": f"error: {str(e)}"}
    
    def _run_dft_recommendations(self, novelty_result: Dict[str, Any], 
                                system_info: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate DFT structure recommendations based on novelty findings."""
        
        print("\n⚛️  STEP 2: DFT Structure Recommendations")
        print("-" * 50)
        
        try:
            # Extract data from novelty assessment
            analysis_text = novelty_result["claims_generation"]["full_analysis"]
            novel_claims = novelty_result.get("novelty_assessment", {}).get("potentially_novel", [])
            
            # Load system info if it's a file path
            if isinstance(system_info, (str, Path)):
                with open(system_info, 'r') as f:
                    system_info_dict = json.load(f)
            else:
                system_info_dict = system_info
            
            # Generate recommendations
            dft_result = self.dft_rec_workflow.run_from_data(
                analysis_text=analysis_text,
                novel_claims=novel_claims,
                system_info=system_info_dict
            )
            
            if dft_result.get('status') == 'success':
                rec_count = len(dft_result.get('recommendations', []))
                print(f"✅ Generated {rec_count} DFT structure recommendations")
                
                # Display top recommendations
                recommendations = dft_result.get('recommendations', [])
                if recommendations:
                    print("\n📋 Top Structure Recommendations:")
                    for i, rec in enumerate(recommendations[:5], 1):
                        priority = rec.get('priority', 'N/A')
                        desc = rec.get('description', 'N/A')
                        print(f"   [{i}] Priority {priority}: {desc}")
            else:
                print(f"❌ DFT recommendations failed: {dft_result.get('message')}")
            
            return dft_result
            
        except Exception as e:
            self.logger.exception("Failed in DFT recommendations step")
            return {"status": f"error: {str(e)}"}
    
    def _select_structures(self, recommendations: List[Dict[str, Any]], 
                          interactive: bool = True,
                          auto_select_top_n: int = 2,
                          max_structures: int = 5) -> List[Dict[str, Any]]:
        """Select which structures to generate."""
        
        print("\n🎯 STEP 3: Structure Selection")
        print("-" * 50)
        
        if not recommendations:
            print("⚠️  No structure recommendations available")
            return []
        
        # Limit to max_structures
        available_recommendations = recommendations[:max_structures]
        
        if interactive:
            return self._interactive_structure_selection(available_recommendations)
        else:
            # Auto-select top N
            selected = available_recommendations[:auto_select_top_n]
            print(f"🤖 Auto-selected top {len(selected)} recommendations (non-interactive mode)")
            for i, struct in enumerate(selected, 1):
                print(f"   [{i}] {struct.get('description', 'N/A')}")
            return selected
    
    def _interactive_structure_selection(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle interactive structure selection."""
        
        print("\nAvailable structures:")
        for i, rec in enumerate(recommendations, 1):
            priority = rec.get('priority', 'N/A')
            desc = rec.get('description', 'N/A')
            print(f"   [{i}] Priority {priority}: {desc}")
        
        print("\nSelection options:")
        print("  • Enter numbers: '1,2,3' (select specific structures)")
        print("  • Enter 'all' (select all available)")
        print(f"  • Enter 'top3' (select top 3)")
        print("  • Enter 'skip' (skip structure generation)")
        
        while True:
            try:
                choice = input(f"\nSelect from 1-{len(recommendations)} or option: ").strip().lower()
                
                if choice == 'skip':
                    print("Structure generation skipped by user")
                    return []
                elif choice == 'all':
                    print(f"Selected all {len(recommendations)} structures")
                    return recommendations
                elif choice.startswith('top'):
                    try:
                        n = int(choice.replace('top', ''))
                        selected = recommendations[:n]
                        print(f"Selected top {len(selected)} structures")
                        return selected
                    except ValueError:
                        print("Invalid 'top' format. Use 'top3', 'top5', etc.")
                        continue
                else:
                    # Parse comma-separated numbers
                    indices = [int(x.strip()) - 1 for x in choice.split(',')]
                    selected = [recommendations[i] for i in indices if 0 <= i < len(recommendations)]
                    if selected:
                        print(f"Selected {len(selected)} structures:")
                        for i, struct in enumerate(selected, 1):
                            print(f"   [{i}] {struct.get('description', 'N/A')}")
                        return selected
                    else:
                        print("Invalid selection. Please try again.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter numbers, 'all', 'topN', or 'skip'.")
            except (KeyboardInterrupt, EOFError):
                print("\nSelection cancelled by user")
                return []
    
    def _generate_structures(self, selected_structures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate atomic structures for selected recommendations."""
        
        print("\n🏗️  STEP 4: Atomic Structure Generation")
        print("-" * 50)
        
        generated_structures = []
        
        for i, structure_rec in enumerate(selected_structures, 1):
            print(f"\n   🔨 Generating structure {i}/{len(selected_structures)}")
            print(f"      Description: {structure_rec.get('description', 'N/A')}")
            
            try:
                # Create structure-specific subdirectory
                structure_name = f"structure_{i:02d}"
                structure_output_dir = self.output_dir / "generated_structures" / structure_name
                structure_output_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"      📁 Output directory: {structure_name}/")
                
                # Create a new DFT workflow instance for this structure
                structure_dft_workflow = DFTWorkflow(
                    google_api_key=self.dft_workflow.google_api_key,
                    futurehouse_api_key=self.dft_workflow.futurehouse_api_key,
                    mp_api_key=getattr(self.dft_workflow.structure_generator, 'mp_api_key', None),
                    generator_model=self.dft_workflow.structure_generator.llm_client.model_name,
                    validator_model=self.dft_workflow.structure_validator.model_name,
                    output_dir=str(structure_output_dir),  # Structure-specific directory
                    max_refinement_cycles=self.dft_workflow.max_refinement_cycles,
                    script_timeout=self.dft_workflow.structure_generator.ase_executor.timeout
                )
                
                # Create detailed request for structure generation
                user_request = structure_rec.get('description', '')
                if 'scientific_interest' in structure_rec:
                    user_request += f". Scientific justification: {structure_rec['scientific_interest']}"
                user_request += ". Save the structure in POSCAR format."
                
                # Generate the structure
                struct_result = structure_dft_workflow.run_complete_workflow(user_request)
                
                structure_info = {
                    "recommendation": structure_rec,
                    "generation_result": struct_result,
                    "structure_number": i,
                    "user_request": user_request,
                    "output_directory": str(structure_output_dir),
                    "structure_name": structure_name
                }
                
                if struct_result.get('final_status') == 'success':
                    # Extract structure file path
                    if 'structure_generation' in struct_result:
                        struct_gen = struct_result['structure_generation']
                        if struct_gen.get('status') == 'success':
                            structure_info["structure_file"] = struct_gen.get('final_structure_path')
                            structure_info["script_file"] = struct_gen.get('final_script_path')
                            structure_info["success"] = True
                            
                            # Create a summary file for this structure
                            self._create_structure_summary(structure_info, structure_output_dir)
                            
                            filename = os.path.basename(struct_gen.get('final_structure_path', 'unknown'))
                            print(f"      ✅ Structure generated: {filename}")
                            print(f"      📁 Files saved to: {structure_name}/")
                        else:
                            structure_info["success"] = False
                            error_msg = struct_gen.get('message', 'Unknown error')
                            print(f"      ❌ Structure generation failed: {error_msg}")
                    else:
                        structure_info["success"] = False
                        print(f"      ❌ No structure generation result")
                else:
                    structure_info["success"] = False
                    status = struct_result.get('final_status', 'Unknown status')
                    print(f"      ❌ Workflow failed: {status}")
                
                generated_structures.append(structure_info)
                
            except Exception as struct_e:
                self.logger.exception(f"Failed to generate structure {i}")
                generated_structures.append({
                    "recommendation": structure_rec,
                    "error": str(struct_e),
                    "success": False,
                    "structure_number": i,
                    "structure_name": f"structure_{i:02d}"
                })
                print(f"      ❌ Error generating structure {i}: {struct_e}")
        
        # Summary
        successful = [s for s in generated_structures if s.get('success', False)]
        print(f"\n✅ Structure generation completed: {len(successful)}/{len(generated_structures)} successful")
        
        return generated_structures
    
    def _create_structure_summary(self, structure_info: Dict[str, Any], output_dir: Path):
        """Create a summary file for each generated structure."""
        
        summary = {
            "structure_number": structure_info["structure_number"],
            "structure_name": structure_info["structure_name"],
            "recommendation": structure_info["recommendation"],
            "user_request": structure_info["user_request"],
            "generation_status": "success" if structure_info["success"] else "failed",
            "files_generated": [],
            "output_directory": str(output_dir)
        }
        
        # List all files in the output directory
        if output_dir.exists():
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    summary["files_generated"].append(file_path.name)
        
        # Save summary
        summary_file = output_dir / "structure_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_final_summary(self, pipeline_result: Dict[str, Any]):
        """Generate and display final pipeline summary."""
        
        data_type = pipeline_result.get("data_type", "experimental")
        
        print(f"\n📋 {data_type.upper()} PIPELINE SUMMARY")
        print("=" * 60)
        
        # Extract key metrics
        novelty_result = pipeline_result.get("novelty_assessment", {})
        claims_count = len(novelty_result.get("claims_generation", {}).get("claims", []))
        novel_count = len(novelty_result.get("novelty_assessment", {}).get("potentially_novel", []))
        
        dft_result = pipeline_result.get("dft_recommendations", {})
        recommendations_count = len(dft_result.get("recommendations", []))
        
        structures = pipeline_result.get("generated_structures", [])
        successful_structures = [s for s in structures if s.get('success', False)]
        
        if successful_structures:
            print(f"🎉 {data_type.title()} pipeline completed successfully!")
            print(f"   📊 Claims generated: {claims_count}")
            print(f"   🔍 Novel findings: {novel_count}")
            print(f"   ⚛️  DFT recommendations: {recommendations_count}")
            print(f"   🏗️  Structures generated: {len(successful_structures)}")
            
            print(f"\n📁 Generated Structure Directories:")
            for i, struct in enumerate(successful_structures, 1):
                structure_name = struct.get('structure_name', f'structure_{i:02d}')
                desc = struct['recommendation'].get('description', 'N/A')
                print(f"   [{i}] {structure_name}/")
                print(f"       {desc}")
                print(f"       📄 Contains: POSCAR, INCAR, KPOINTS, scripts, logs")
            
            print(f"\n📂 Main Output Directory Structure:")
            print(f"   {self.output_dir}/")
            print(f"   ├── {data_type}_novelty_assessment/")
            print(f"   ├── dft_recommendations/") 
            print(f"   ├── generated_structures/")
            for struct in successful_structures:
                structure_name = struct.get('structure_name', 'structure_XX')
                print(f"   │   ├── {structure_name}/")
                print(f"   │   │   ├── POSCAR")
                print(f"   │   │   ├── INCAR")
                print(f"   │   │   ├── KPOINTS")
                print(f"   │   │   ├── structure_summary.json")
                print(f"   │   │   └── workflow_log.txt")
            print(f"   └── complete_pipeline_result.json")
        else:
            print(f"⚠️  {data_type.title()} pipeline completed but no structures were successfully generated")
            print(f"   📊 Claims generated: {claims_count}")
            print(f"   🔍 Novel findings: {novel_count}")
            print(f"   ⚛️  DFT recommendations: {recommendations_count}")

    def _save_pipeline_results(self, pipeline_result: Dict[str, Any]):
        """Save complete pipeline results to file."""
        
        try:
            output_file = self.output_dir / "complete_pipeline_result.json"
            
            # Convert to JSON-serializable format
            json_result = json.loads(json.dumps(pipeline_result, default=str))
            
            with open(output_file, 'w') as f:
                json.dump(json_result, f, indent=2)
            
            self.logger.info(f"Complete pipeline result saved to: {output_file}")
            print(f"\n💾 Complete pipeline result saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline results: {e}")
    
    def get_summary(self, pipeline_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the pipeline results."""
        
        data_type = pipeline_result.get("data_type", "experimental")
        
        summary = f"Complete {data_type.title()} Pipeline Summary\n{'='*40}\n"
        summary += f"Status: {pipeline_result.get('final_status', 'Unknown')}\n"
        summary += f"Data type: {data_type}\n"
        summary += f"Steps completed: {', '.join(pipeline_result.get('steps_completed', []))}\n"
        summary += f"Input data: {os.path.basename(str(pipeline_result.get('input_data', 'Unknown')))}\n\n"
        
        # Novelty assessment summary
        novelty_result = pipeline_result.get("novelty_assessment", {})
        if novelty_result:
            claims_count = len(novelty_result.get("claims_generation", {}).get("claims", []))
            novel_count = len(novelty_result.get("novelty_assessment", {}).get("potentially_novel", []))
            summary += f"Novelty Assessment:\n"
            summary += f"  Claims generated: {claims_count}\n"
            summary += f"  Novel findings: {novel_count}\n\n"
        
        # DFT recommendations summary
        dft_result = pipeline_result.get("dft_recommendations", {})
        if dft_result:
            rec_count = len(dft_result.get("recommendations", []))
            summary += f"DFT Recommendations: {rec_count} structures recommended\n\n"
        
        # Structure generation summary
        structures = pipeline_result.get("generated_structures", [])
        if structures:
            successful = [s for s in structures if s.get('success', False)]
            summary += f"Structure Generation:\n"
            summary += f"  Attempted: {len(structures)}\n"
            summary += f"  Successful: {len(successful)}\n"
            
            if successful:
                summary += f"  Generated files:\n"
                for i, struct in enumerate(successful, 1):
                    if 'structure_file' in struct:
                        filename = os.path.basename(struct['structure_file'])
                        summary += f"    {i}. {filename}\n"
        
        return summary
