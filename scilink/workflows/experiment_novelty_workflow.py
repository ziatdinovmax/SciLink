import os
import sys
import logging
import json
from io import StringIO
from typing import Dict, Any, List, Union
import textwrap
from pathlib import Path

from .analyzers import MicroscopyAnalyzer, SpectroscopyAnalyzer, BaseExperimentAnalyzer, CurveAnalyzer

from ..agents.exp_agents.microscopy_agent import MicroscopyAnalysisAgent
from ..agents.lit_agents.literature_agent import OwlLiteratureAgent
from ..agents.lit_agents.novelty_scorer import NoveltyScorer, enhanced_novelty_assessment, display_enhanced_novelty_summary

import warnings
from ..auth import get_api_key, APIKeyNotFoundError


def select_claims_interactive(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Allows user to interactively select which claims to search for in the literature.
    """
    if not claims:
        print("❌ No claims available to select from.")
        return []
        
    print(f"\n📋 Claim Selection")
    print(f"{'─'*50}")
    print("Select claims for literature search:")
    print("• Enter comma-separated numbers (e.g., '1,3,5')")
    print("• Enter 'all' for all claims")
    
    # Display the "Has Anyone" questions with clean formatting
    for i, claim in enumerate(claims):
        print(f"\n[{i+1}] {claim.get('has_anyone_question', 'No question formulated')}")
    
    print(f"{'─'*50}")
    
    # Get user selection
    try:
        selection = input("\n🔍 Select claims to search (or 'all'): ").strip().lower()
        
        if selection == 'all':
            print(f"✅ Selected all {len(claims)} claims.")
            return claims
            
        # Parse the comma-separated list of numbers
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_claims = []
            
            for idx in indices:
                if 0 <= idx < len(claims):
                    selected_claims.append(claims[idx])
                else:
                    print(f"⚠️  Index {idx+1} is out of range and will be skipped.")
            
            if not selected_claims:
                print("❌ No valid claims were selected.")
                return []
                
            print(f"✅ Selected {len(selected_claims)} claims.")
            return selected_claims
            
        except ValueError:
            print("❌ Invalid selection format. Please use comma-separated numbers or 'all'.")
            return select_claims_interactive(claims)  # Try again
            
    except KeyboardInterrupt:
        print("\n❌ Selection canceled by user.")
        return []


class ExperimentNoveltyAssessment:
    """
    Unified workflow for analyzing experimental data and assessing novelty.
    
    Supports multiple experimental data types (microscopy, spectroscopy, etc.)
    through a pluggable analyzer system. Can be easily extended for new data types.
    """
    
    # Registry of available analyzers
    ANALYZER_REGISTRY = {
        'microscopy': MicroscopyAnalyzer,
        'spectroscopy': SpectroscopyAnalyzer,
        'curve': CurveAnalyzer
    }
    
    def __init__(self,
                 data_type: str,
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 local_model: str = None,
                 output_dir: str = "experiment_novelty_output",
                 max_wait_time: int = 600,
                 dft_recommendations: bool = False,
                 measurement_recommendations: bool = False,
                 enable_human_feedback: bool = True,
                 display_agent_logs: bool = True,
                 **analyzer_kwargs):
        """
        Initialize the unified experiment novelty assessment workflow.
        
        Args:
            data_type: Type of experimental data ('microscopy', 'spectroscopy', etc.)
            google_api_key: Google API key for analysis
            futurehouse_api_key: FutureHouse API key for literature search
            analysis_model: Model name for analysis
            output_dir: Directory for outputs
            max_wait_time: Maximum wait time for literature searches
            dft_recommendations: Whether to generate DFT recommendations (microscopy only)
            enable_human_feedback: Whether to enable human feedback in analyzers
            display_agent_logs: Whether to show detailed agent logs (default: True)
            **analyzer_kwargs: Additional arguments passed to the specific analyzer
        """
        # Validate data type
        if data_type not in self.ANALYZER_REGISTRY:
            available_types = list(self.ANALYZER_REGISTRY.keys())
            raise ValueError(f"Unsupported data_type '{data_type}'. Available types: {available_types}")
        
        self.data_type = data_type
        
        # Auto-discover API keys
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        if futurehouse_api_key is None:
            futurehouse_api_key = get_api_key('futurehouse')
            if not futurehouse_api_key:
                warnings.warn(
                    "FutureHouse API key not found. Literature search will be disabled."
                )
        
        # Setup logging with better formatting while preserving agent visibility
        self.log_capture = StringIO()
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(name)s: %(message)s', 
            force=True, 
            handlers=[
                logging.StreamHandler(sys.stdout),  # Show logs for agent visibility
                logging.StreamHandler(self.log_capture)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.display_agent_logs = display_agent_logs
        
        self.google_api_key = google_api_key
        self.analysis_model = analysis_model
        self.local_model = local_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dft_recommendations = dft_recommendations
        self.measurement_recommendations = measurement_recommendations 
        
        # Initialize data-specific analyzer
        analyzer_class = self.ANALYZER_REGISTRY[data_type]
        self.analyzer = analyzer_class(
            google_api_key=google_api_key,
            futurehouse_api_key=futurehouse_api_key, # for exp agents that utilize lit search
            analysis_model=analysis_model,
            local_model=local_model,
            output_dir=str(self.output_dir),
            enable_human_feedback=enable_human_feedback,
            **analyzer_kwargs
        )
        
        # Initialize literature novelty agent
        if futurehouse_api_key:
            self.lit_agent = OwlLiteratureAgent(
                api_key=futurehouse_api_key,
                max_wait_time=max_wait_time
            )
        else:
            self.lit_agent = None
            print("ℹ️  Literature search disabled (no FutureHouse API key)")
    
    @classmethod
    def register_analyzer(cls, data_type: str, analyzer_class):
        """Register a new analyzer for a data type."""
        if not issubclass(analyzer_class, BaseExperimentAnalyzer):
            raise TypeError("Analyzer must inherit from BaseExperimentAnalyzer")
        cls.ANALYZER_REGISTRY[data_type] = analyzer_class
    
    @classmethod
    def get_supported_data_types(cls) -> List[str]:
        """Get list of supported data types."""
        return list(cls.ANALYZER_REGISTRY.keys())
    
    def run_complete_workflow(self, data_path: Union[str, Path], 
                             system_info: Union[str, Path, Dict[str, Any]] = None, 
                             **analysis_kwargs) -> Dict[str, Any]:
        """
        Run the complete novelty assessment workflow.
        
        Args:
            data_path: Path to experimental data
            system_info: System information (dict or path to JSON file)
            **analysis_kwargs: Additional arguments for the analyzer
        """
        
        # Convert system_info if it's a file path
        if isinstance(system_info, (str, Path)):
            with open(system_info, 'r') as f:
                system_info = json.load(f)
        
        workflow_result = {
            "data_path": str(data_path),
            "data_type": self.data_type,
            "system_info": system_info,
            "steps_completed": [],
            "final_status": "started"
        }
        
        data_type_name = self.analyzer.get_data_type_name()
        
        print(f"\n🔬 Experiment Novelty Assessment Starting")
        print(f"{'='*60}")
        print(f"📊 Data Type: {data_type_name.title()}")
        print(f"📁 Data File: {os.path.basename(str(data_path))}")
        print(f"📂 Output:   {self.output_dir}/")
        print(f"{'='*60}")
        
        # === Step 1: Experimental Data Analysis ===
        print(f"\n🔍 WORKFLOW STEP 1: {data_type_name.title()} Analysis")
        print(f"{'─'*50}")
        
        analysis_result = self._analyze_experimental_data(data_path, system_info, **analysis_kwargs)
        workflow_result["claims_generation"] = analysis_result
        
        if analysis_result["status"] != "success":
            print(f"❌ {data_type_name.title()} analysis failed: {analysis_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_analysis"
            return workflow_result
        
        workflow_result["steps_completed"].append("claims_generation")
        claims = analysis_result["claims"]
        
        print(f"✅ Analysis complete: {len(claims)} scientific claims generated")
        print(f"📄 Claims saved: {os.path.basename(analysis_result['claims_file'])}")
        
        # === Step 2: Literature Search ===
        if self.lit_agent is None:
            print(f"⚠️  Literature search skipped (no FutureHouse API key)")
            workflow_result["final_status"] = "no_literature_search"
            self._print_final_summary(workflow_result, data_type_name)
            return workflow_result
        
        print(f"\n📚 WORKFLOW STEP 2: Literature Search")
        print(f"{'─'*50}")
        
        literature_result = self._conduct_literature_search(claims)
        workflow_result["literature_search"] = literature_result
        
        if literature_result["status"] != "success":
            print(f"❌ Literature search failed: {literature_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_literature"
            return workflow_result
        
        workflow_result["steps_completed"].append("literature_search")
        literature_results = literature_result["results"]
        
        print(f"✅ Literature search complete: {len(literature_results)} claims processed")
        print(f"📄 Results saved: {os.path.basename(literature_result['results_file'])}")
        
        # === Step 3: Novelty Assessment ===
        print(f"\n🎯 WORKFLOW STEP 3: Novelty Assessment")
        print(f"{'─'*50}")
        
        novelty_result = self._assess_novelty(literature_results, data_type_name)
        workflow_result["novelty_assessment"] = novelty_result["assessment"]
        
        if novelty_result["status"] != "success":
            print(f"❌ Novelty assessment failed: {novelty_result.get('message', 'Unknown error')}")
            workflow_result["final_status"] = "failed_novelty"
            return workflow_result
        
        workflow_result["steps_completed"].append("novelty_assessment")
        assessment = novelty_result["assessment"]
        
        print(f"✅ Novelty assessment complete")
        print(f"📊 Average novelty score: {assessment.get('average_novelty_score', 0):.2f}/5.0")
        print(f"📄 Assessment saved: {os.path.basename(novelty_result['novelty_file'])}")
        
        # === Step 4: DFT Recommendations (Optional) ===
        if self.dft_recommendations and self.data_type == 'microscopy':
            print(f"\n⚛️  WORKFLOW STEP 4: DFT Recommendations")
            print(f"{'─'*50}")
            
            dft_result = self._generate_dft_recommendations(
                workflow_result["claims_generation"]["detailed_analysis"],
                workflow_result["novelty_assessment"] 
            )
            workflow_result["dft_recommendations"] = dft_result
            
            if dft_result["status"] == "success":
                workflow_result["steps_completed"].append("dft_recommendations")
                print(f"✅ DFT recommendations generated: {dft_result['total_recommendations']} structures")
                print(f"📄 Recommendations saved: {os.path.basename(dft_result['dft_file'])}")
            else:
                print(f"⚠️  DFT recommendations failed: {dft_result.get('message', 'Unknown error')}")
        
        # === Step 5: Measurement Recommendations (Optional) ===
        if self.measurement_recommendations:
            step_number = "5" if (self.dft_recommendations and self.data_type == 'microscopy') else "4"
            print(f"\n📋 WORKFLOW STEP {step_number}: Measurement Recommendations")
            print(f"{'─'*50}")
            
            measurement_result = self._generate_measurement_recommendations(
                workflow_result["claims_generation"],
                workflow_result.get("novelty_assessment", {}),
                system_info
            )
            workflow_result["measurement_recommendations"] = measurement_result
            
            if measurement_result["status"] == "success":
                workflow_result["steps_completed"].append("measurement_recommendations")
                rec_count = measurement_result.get('total_recommendations', 0)
                print(f"✅ Measurement recommendations generated: {rec_count} suggestions")
                print(f"📄 Recommendations saved: {os.path.basename(measurement_result['recommendations_file'])}")
            else:
                print(f"⚠️  Measurement recommendations failed: {measurement_result.get('message', 'Unknown error')}")
        
        workflow_result["final_status"] = "success"
        
        # Save workflow log
        self._save_workflow_log(data_type_name)
        
        # Final summary
        self._print_final_summary(workflow_result, data_type_name)
        
        return workflow_result
    
    def _analyze_experimental_data(self, data_path: Union[str, Path],
                                     system_info: Dict[str, Any],
                                     **analysis_kwargs) -> Dict[str, Any]:
        """Analyze experimental data and generate claims."""

        print(f"🔨 Analyzing {self.data_type} data...")
        if self.display_agent_logs:
            print(f"    (Agent logs will appear below)")
            print(f"{'─'*30}")

        try:
            analysis_result = self.analyzer.analyze_for_claims(
                str(data_path),
                system_info=system_info,
                **analysis_kwargs
            )

            if self.display_agent_logs:
                print(f"{'─'*30}")

            if "error" in analysis_result:
                return {
                    "status": "error",
                    "message": analysis_result["error"]
                }

            claims = analysis_result.get("scientific_claims", [])
            detailed_analysis = analysis_result.get("detailed_analysis", "No analysis text provided.")

            if not claims:
                return {
                    "status": "error",
                    "message": "No scientific claims generated from analysis"
                }

            if analysis_result.get("human_feedback", {}).get("feedback_provided"):
                # If feedback was given, display the final, refined results.
                print("\n--- Analysis Summary (After Applying User Feedback) ---")
                self._display_analysis_summary(claims, detailed_analysis)
                
            # Save claims
            claims_file = self.output_dir / f"generated_{self.data_type}_claims.json"
            with open(claims_file, 'w') as f:
                json.dump(analysis_result, f, indent=2)

            return {
                "status": "success",
                "claims": claims,
                "claims_file": str(claims_file),
                "detailed_analysis": detailed_analysis
            }

        except Exception as e:
            if self.display_agent_logs:
                print(f"{'─'*30}")
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }
            
        except Exception as e:
            if self.display_agent_logs:
                print(f"{'─'*30}")
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }
    
    def _conduct_literature_search(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Conduct literature search for selected claims."""
        
        # Interactive claim selection with better formatting
        selected_claims = select_claims_interactive(claims)
        
        if not selected_claims:
            return {
                "status": "error",
                "message": "No claims selected for literature search"
            }
        
        print(f"🔍 Searching literature for {len(selected_claims)} claims...")
        if self.display_agent_logs:
            print(f"   (Literature agent logs will appear below)")
            print(f"{'─'*30}")
        
        try:
            literature_results = []
            
            for i, claim in enumerate(selected_claims):
                claim_num = i + 1
                total_claims = len(selected_claims)
                
                has_anyone_question = claim.get("has_anyone_question")
                if not has_anyone_question:
                    print(f"   ⚠️  Claim {claim_num}/{total_claims}: No search question available")
                    continue
                
                print(f"\n   🔍 Searching claim {claim_num}/{total_claims}...")
                if not self.display_agent_logs:
                    print(f"   Question: {has_anyone_question}")
                
                # Query literature (agent logs will appear here if display_agent_logs=True)
                owl_result = self.lit_agent.query_literature(has_anyone_question)
                
                if owl_result.get("status") == "success" and owl_result.get("formatted_answer"):
                    try:
                        # Create a dedicated directory for these readable responses
                        responses_dir = self.output_dir / "literature_responses"
                        responses_dir.mkdir(exist_ok=True)
                        
                        response_filename = f"literature_response_claim_{claim_num:02d}.md"
                        response_filepath = responses_dir / response_filename
                        
                        with open(response_filepath, 'w', encoding='utf-8') as f:
                            f.write(f"# Literature Response for Claim {claim_num}\n\n")
                            f.write(f"**Question:** {has_anyone_question}\n\n---\n\n")
                            f.write(owl_result["formatted_answer"])
                        
                        # Inform the user about the saved file
                        print(f"        📄 Saved formatted response: {response_filepath.relative_to(self.output_dir.parent)}")
                    except Exception as e:
                        print(f"        ⚠️  Could not save formatted response for claim {claim_num}: {e}")

                literature_results.append({
                    "original_claim": claim,
                    "owl_result": owl_result
                })
                
                # Show brief result
                if owl_result["status"] == "success":
                    print(f"   ✅ Search {claim_num}/{total_claims} complete")
                else:
                    print(f"   ❌ Search {claim_num}/{total_claims} failed: {owl_result.get('message', 'Unknown error')}")
            
            if self.display_agent_logs:
                print(f"{'─'*30}")
            
            # Save results
            results_file = self.output_dir / f"{self.data_type}_literature_search_results.json"
            with open(results_file, 'w') as f:
                json.dump(literature_results, f, indent=2)
            
            return {
                "status": "success",
                "results": literature_results,
                "results_file": str(results_file)
            }
            
        except Exception as e:
            if self.display_agent_logs:
                print(f"{'─'*30}")
            return {
                "status": "error",
                "message": f"Literature search failed: {str(e)}"
            }
    
    def _assess_novelty(self, literature_results: List[Dict[str, Any]], 
                       data_type_name: str) -> Dict[str, Any]:
        """Run enhanced novelty assessment with structured scoring."""
        
        print(f"🎯 Scoring novelty based on literature findings...")
        if self.display_agent_logs:
            print(f"   (Novelty scorer logs will appear below)")
            print(f"{'─'*30}")
        
        try:
            # Initialize novelty scorer
            novelty_scorer = NoveltyScorer(google_api_key=self.google_api_key)
            
            # Run enhanced assessment (scorer logs will appear here if display_agent_logs=True)
            novelty_assessment = enhanced_novelty_assessment(literature_results, novelty_scorer)
            
            if self.display_agent_logs:
                print(f"{'─'*30}")
            
            # Display results in DFT style
            self._display_novelty_summary(novelty_assessment)
            
            # Save assessment
            novelty_file = self.output_dir / f"{data_type_name}_enhanced_novelty_assessment.json"
            with open(novelty_file, 'w') as f:
                json.dump(novelty_assessment, f, indent=2)
            
            return {
                "status": "success",
                "assessment": novelty_assessment,
                "novelty_file": str(novelty_file)
            }
            
        except Exception as e:
            if self.display_agent_logs:
                print(f"{'─'*30}")
            return {
                "status": "error",
                "message": f"Novelty assessment failed: {str(e)}"
            }
    
    def _display_analysis_summary(self, claims: List[Dict[str, Any]], detailed_analysis: str):
        """Display generated claims and analysis in DFT-style format."""

        print("\n--- Analysis Summary (After User Feedback) ---")
        print("\n📋 DETAILED ANALYSIS:")
        print(detailed_analysis)

        print(f"\n🎯 GENERATED CLAIMS ({len(claims)}):")
        for i, claim in enumerate(claims, 1):
            claim_text = claim.get('claim', 'No claim text')
            print(f"    {i}. {claim_text}")

        if claims:
            keywords_count = sum(len(claim.get('keywords', [])) for claim in claims)
            print(f"    🏷️  Total keywords identified: {keywords_count}")
    
    def _display_novelty_summary(self, assessment: Dict[str, Any]):
        """Display novelty assessment in DFT-style format."""
        
        categories = assessment.get("novelty_categories", {})
        avg_score = assessment.get("average_novelty_score", 0)
        
        print(f"   📊 Novelty scoring complete:")
        print(f"      Average score: {avg_score:.2f}/5.0")
        
        high_novel = categories.get("highly_novel", [])
        moderate_novel = categories.get("moderately_novel", [])
        low_novel = categories.get("low_novelty", [])
        
        if high_novel:
            print(f"      🚀 Highly novel findings: {len(high_novel)}")
        if moderate_novel:
            print(f"      📈 Moderately novel findings: {len(moderate_novel)}")
        if low_novel:
            print(f"      📚 Known findings: {len(low_novel)}")
    
    def _print_final_summary(self, workflow_result: Dict[str, Any], data_type_name: str):
        """Print a clean final summary in DFT style."""
        
        print(f"\n🎉 Experiment Novelty Assessment Complete!")
        print(f"{'='*60}")
        
        # Basic info
        status = workflow_result.get('final_status')
        steps = workflow_result.get('steps_completed', [])
        
        print(f"📋 Status: {status}")
        print(f"✅ Steps: {' → '.join(steps)}")
        print(f"📁 Output: {self.output_dir}/")
        
        # Claims info
        if "claims_generation" in workflow_result:
            claims_result = workflow_result["claims_generation"]
            if claims_result["status"] == "success":
                claims_count = len(claims_result['claims'])
                print(f"🔍 Analysis: {claims_count} scientific claims generated")
        
        # Literature info
        if "literature_search" in workflow_result:
            lit_result = workflow_result["literature_search"]
            if lit_result["status"] == "success":
                search_count = len(lit_result['results'])
                print(f"📚 Literature: {search_count} claims searched")
        
        # Novelty info
        if "novelty_assessment" in workflow_result:
            assessment = workflow_result["novelty_assessment"]
            avg_score = assessment.get('average_novelty_score', 0)
            categories = assessment.get("novelty_categories", {})
            high_count = len(categories.get("highly_novel", []))
            
            print(f"🎯 Novelty: {avg_score:.2f}/5.0 average score")
            if high_count > 0:
                print(f"   🚀 {high_count} highly novel finding{'s' if high_count > 1 else ''}")
        
        # DFT info
        if "dft_recommendations" in workflow_result:
            dft_result = workflow_result["dft_recommendations"]
            if dft_result["status"] == "success":
                rec_count = dft_result.get('total_recommendations', 0)
                print(f"⚛️  DFT: {rec_count} structure recommendation{'s' if rec_count > 1 else ''}")
        
        # Measurement recommendations info (NEW)
        if "measurement_recommendations" in workflow_result:
            measurement_result = workflow_result["measurement_recommendations"]
            if measurement_result["status"] == "success":
                rec_count = measurement_result.get('total_recommendations', 0)
                print(f"📋 Measurements: {rec_count} recommendation{'s' if rec_count > 1 else ''}")
        
        print(f"{'='*60}")
    
    def _save_workflow_log(self, data_type_name: str):
        """Save the complete workflow log."""
        try:
            log_content = self.log_capture.getvalue()
            log_path = self.output_dir / f"{data_type_name}_novelty_workflow_log.txt"
            
            with open(log_path, 'w') as f:
                f.write(f"{data_type_name.title()} Novelty Assessment Workflow Log\n")
                f.write("=" * (len(data_type_name) + 35) + "\n\n")
                f.write(log_content)
                
        except Exception as e:
            print(f"⚠️  Could not save workflow log: {e}")
    
    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the workflow results."""
        
        data_type = workflow_result.get("data_type", "experimental")
        
        summary = f"{data_type.title()} Novelty Assessment Summary\n{'='*40}\n"
        summary += f"Data: {os.path.basename(str(workflow_result.get('data_path', 'Unknown')))}\n"
        summary += f"Status: {workflow_result.get('final_status', 'Unknown')}\n"
        summary += f"Steps completed: {', '.join(workflow_result.get('steps_completed', []))}\n\n"
        
        # Claims summary
        if "claims_generation" in workflow_result:
            claims_data = workflow_result["claims_generation"]
            claims_count = len(claims_data.get("claims", []))
            summary += f"Generated {claims_count} scientific claims\n\n"
        
        # Novelty assessment summary
        if "novelty_assessment" in workflow_result:
            assessment = workflow_result["novelty_assessment"]
            summary += f"Literature Assessment:\n"
            summary += f"  Total claims searched: {assessment.get('total_claims_searched', 0)}\n"
            summary += f"  Successfully scored: {assessment.get('successful_searches', 0)}\n"
            summary += f"  Average novelty score: {assessment.get('average_novelty_score', 0):.2f}/5.0\n\n"
            
            categories = assessment.get("novelty_categories", {})
            
            if categories.get("highly_novel"):
                summary += f"🚀 Potentially Highly Novel Findings ({len(categories['highly_novel'])}):\n"
                for i, claim in enumerate(categories["highly_novel"], 1):
                    summary += f"  {i}. {claim}\n"
                summary += "\n"
            
            if categories.get("moderately_novel"):
                summary += f"📊 Moderately Novel Findings ({len(categories['moderately_novel'])}):\n"
                for i, claim in enumerate(categories["moderately_novel"], 1):
                    summary += f"  {i}. {claim}\n"
                summary += "\n"
        
        # DFT recommendations summary
        if "dft_recommendations" in workflow_result and workflow_result["dft_recommendations"].get("status") == "success":
            dft_result = workflow_result["dft_recommendations"]
            summary += f"DFT Recommendations: {dft_result.get('total_recommendations', 0)} structures recommended\n"
        
        return summary
    
    def _display_dft_recommendations(self, reasoning: str, recommendations: List[Dict], novel_claims: List[str]):
        """Display the generated DFT recommendations."""
        
        print("\n" + "="*60)
        print("DFT STRUCTURE RECOMMENDATIONS")
        print("="*60)
        
        print(f"\nNovel claims processed: {len(novel_claims)}")
        print(f"Total recommendations: {len(recommendations)}")
        
        print(f"\n--- Reasoning ---")
        print(reasoning)
        print("-" * 40)
        
        if recommendations:
            print("\n--- Recommended Structures ---")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n[{i}] Priority: {rec.get('priority', 'N/A')}")
                print(f"    {rec.get('description', 'N/A')}")
                print(f"    Justification: {rec.get('scientific_interest', 'N/A')}")
        else:
            print("\n⚠️  No recommendations generated")
        
        print("\n" + "="*60)
    
    def _generate_dft_recommendations(self, initial_analysis_text: str, novelty_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DFT recommendations based on enhanced novelty analysis (microscopy only)"""
        
        print(f"⚛️  Generating DFT structure recommendations...")
        
        try:
            # Extract high and moderate novelty claims for prioritization
            high_novel = novelty_assessment.get("novelty_categories", {}).get("highly_novel", [])
            moderate_novel = novelty_assessment.get("novelty_categories", {}).get("moderately_novel", [])
            
            # Generate enhanced novelty context for DFT recommendations
            if high_novel or moderate_novel:
                novelty_section_text = "The following claims/observations have been assessed through literature review with their novelty scores:\n\n"
                
                if high_novel:
                    novelty_section_text += "HIGH NOVELTY FINDINGS (Scores 4-5):\n"
                    for i, claim in enumerate(high_novel, 1):
                        novelty_section_text += f"{i}. {claim}\n"
                    novelty_section_text += "\n"
                
                if moderate_novel:
                    novelty_section_text += "MODERATE NOVELTY FINDINGS (Score 3):\n"
                    for i, claim in enumerate(moderate_novel, 1):
                        novelty_section_text += f"{i}. {claim}\n"
                    novelty_section_text += "\n"
                
                avg_score = novelty_assessment.get("average_novelty_score", 0)
                novelty_section_text += f"Average novelty score: {avg_score:.2f}/5.0\n\n"
                
                novelty_section_text += ("Your primary goal for the DFT recommendations should be to propose structures and simulations "
                                    "that can rigorously investigate these novel aspects, with priority given to the highest-scoring claims. "
                                    "Explain the connection between each recommended structure and the specific novel findings it would help investigate.")
                
                novelty_context = novelty_section_text
            else:
                novelty_context = "No high-novelty claims were identified through literature review. Please make DFT recommendations based on the most scientifically interesting aspects of the provided analysis."
            
            # Instantiate a text-only analysis agent for this step
            dft_agent = MicroscopyAnalysisAgent(
                google_api_key=self.google_api_key,
                model_name=self.analysis_model
            )
            
            # Generate DFT recommendations using text-only path
            dft_recommendations_result = dft_agent.analyze_microscopy_image_for_structure_recommendations(
                image_path=None,  # Text-only path
                system_info=None,
                additional_prompt_context=novelty_context,
                cached_detailed_analysis=initial_analysis_text
            )
            
            if "error" in dft_recommendations_result:
                return {
                    "status": "error",
                    "message": f"DFT recommendation generation failed: {dft_recommendations_result.get('details', dft_recommendations_result.get('error'))}"
                }
            
            reasoning_text = dft_recommendations_result.get("analysis_summary_or_reasoning", "No reasoning provided")
            recommendations = dft_recommendations_result.get("recommendations", [])
            
            print(f"   ✅ Generated {len(recommendations)} DFT structure recommendations")
            
            self._display_dft_recommendations(reasoning_text, recommendations, high_novel + moderate_novel)

            # Save DFT recommendations
            dft_file = self.output_dir / "dft_recommendations.json"
            dft_output = {
                "reasoning_for_recommendations": reasoning_text,
                "recommendations": recommendations,
                "novelty_context": novelty_context,
                "novelty_scores_used": {
                    "high_novel_count": len(high_novel),
                    "moderate_novel_count": len(moderate_novel),
                    "average_score": novelty_assessment.get("average_novelty_score", 0)
                }
            }
            with open(dft_file, 'w') as f:
                json.dump(dft_output, f, indent=2)
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "reasoning": reasoning_text,
                "dft_file": str(dft_file),
                "total_recommendations": len(recommendations)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"DFT recommendations failed: {str(e)}"
            }
        
    def _generate_measurement_recommendations(self, 
                                            claims_result: Dict[str, Any],
                                            novelty_assessment: Dict[str, Any],
                                            system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate measurement recommendations using the stored analysis agent."""
        
        print(f"🤖 Generating measurement recommendations...")
        
        try:
            # Create novelty context
            novelty_context = self._create_novelty_context(novelty_assessment)
            
            # Use the stored agent that already did the analysis
            if not hasattr(self.analyzer, 'analysis_agent') or self.analyzer.analysis_agent is None:
                return {
                    "status": "error", 
                    "message": "No analysis agent available - analysis may have failed"
                }
            
            agent = self.analyzer.analysis_agent
            
            # The agent already has all stored analysis images and proper context
            recommendations_result = agent.generate_measurement_recommendations(
                analysis_result=claims_result,
                system_info=system_info,
                novelty_context=novelty_context
            )
            
            if "error" in recommendations_result:
                return {"status": "error", "message": recommendations_result["error"]}
            
            # Display recommendations summary
            self._display_measurement_recommendations_summary(recommendations_result)
            
            # Save recommendations
            rec_file = self.output_dir / f"{self.data_type}_measurement_recommendations.json"
            with open(rec_file, 'w') as f:
                json.dump(recommendations_result, f, indent=2)
            
            return {
                "status": "success",
                "total_recommendations": recommendations_result.get("total_recommendations", 0),
                "recommendations_file": str(rec_file)
            }
            
        except Exception as e:
            self.logger.exception("Failed to generate measurement recommendations")
            return {"status": "error", "message": str(e)}
    
    def _create_novelty_context(self, novelty_assessment: Dict[str, Any]) -> str:
        """Create novelty context for measurement recommendations."""
        
        categories = novelty_assessment.get("novelty_categories", {})
        avg_score = novelty_assessment.get("average_novelty_score", 0)
        
        high_novel = categories.get("highly_novel", [])
        moderate_novel = categories.get("moderately_novel", [])
        
        if not high_novel and not moderate_novel:
            return "No high-novelty findings identified. Focus on standard follow-up measurements."
        
        context = f"Literature review indicates significant novelty (average score: {avg_score:.2f}/5.0). "
        context += "Prioritize measurements that can investigate these novel aspects:\n\n"
        
        if high_novel:
            context += "HIGH NOVELTY FINDINGS:\n"
            for i, claim in enumerate(high_novel, 1):
                context += f"{i}. {claim}\n"
            context += "\n"
        
        if moderate_novel:
            context += "MODERATE NOVELTY FINDINGS:\n"
            for i, claim in enumerate(moderate_novel, 1):
                context += f"{i}. {claim}\n"
            context += "\n"
        
        context += "Focus measurements on validating, expanding, or providing complementary evidence."
        return context
    
    def _display_measurement_recommendations_summary(self, recommendations_result: Dict[str, Any]):
        """Display detailed top recommendations and summary for workflow with improved formatting."""
        
        recommendations = recommendations_result.get("measurement_recommendations", [])
        if not recommendations:
            print("   ⚠️  No recommendations generated")
            return
            
        print(f"   🎯 Generated {len(recommendations)} measurement recommendations.")
        
        num_to_show_detail = min(len(recommendations), 3)

        print(f"\n   📋 TOP {num_to_show_detail} RECOMMENDATION{'S' if num_to_show_detail > 1 else ''}")

        # --- Display each of the top recommendations in detail ---
        for i, rec in enumerate(recommendations[:num_to_show_detail], 1):
            # Header for each recommendation
            print(f"\n   {'─' * 70}")
            print(f"   [ {i} ] Recommendation (Priority: {rec.get('priority', 'N/A')})")
            print(f"   {'─' * 70}")

            detail_keys = {
                "category": "Category",
                "description": "Description",
                "target_regions": "Target Regions",
                "scientific_justification": "Justification",
                "expected_outcomes": "Expected Outcomes"
            }
            
            base_indent = "     "
            # Set width based on the longest label to align colons
            label_column_width = max(len(label) for label in detail_keys.values())
            
            keys_to_print = [key for key in detail_keys if rec.get(key)]

            for idx, key in enumerate(keys_to_print):
                label = detail_keys[key]
                value = rec[key]

                # Pad the label, then add the colon and a guaranteed space
                formatted_label = f"{label}:".ljust(label_column_width + 2)
                
                # The prefix for the first line of text
                prefix = base_indent + formatted_label
                
                # The indent for subsequent wrapped lines
                subsequent_indent = ' ' * len(prefix)
                
                wrapped_text = textwrap.fill(
                    str(value),
                    width=80,
                    initial_indent=prefix,
                    subsequent_indent=subsequent_indent
                )
                print(wrapped_text)
                
                # Add a blank line if this is not the last item being printed
                if idx < len(keys_to_print) - 1:
                    print()

        # --- Show a compact summary for any remaining recommendations ---
        if len(recommendations) > num_to_show_detail:
            remaining_count = len(recommendations) - num_to_show_detail
            print(f"\n   {'─' * 70}")
            print(f"   📝 Additional {remaining_count} recommendation{'s' if remaining_count > 1 else ''}:")
            
            for i, rec in enumerate(recommendations[num_to_show_detail:], num_to_show_detail + 1):
                category = rec.get("category", "Unknown")
                priority = rec.get("priority", "N/A")
                print(f"     [{i}] {category} (Priority {priority})")
            
        print(f"\n   📄 See saved JSON file for complete details of all recommendations.")