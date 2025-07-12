import os
import sys
import logging
import json
from io import StringIO
from typing import Dict, Any, List, Union
from pathlib import Path

from .analyzers import MicroscopyAnalyzer, SpectroscopyAnalyzer, BaseExperimentAnalyzer

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
        print("No claims available to select from.")
        return []
        
    print("\n--- Select Claims for Literature Search ---")
    print("Enter comma-separated numbers of claims to search, or 'all' for all claims.")
    print("Examples: '1,3,5' or 'all'")
    
    # Display the "Has Anyone" questions
    for i, claim in enumerate(claims):
        print(f"\n[{i+1}] {claim.get('has_anyone_question', 'No question formulated')}")
    
    print("-" * 70)
    
    # Get user selection
    try:
        selection = input("\nSelect claims to search (or 'all'): ").strip().lower()
        
        if selection == 'all':
            print(f"Selected all {len(claims)} claims.")
            return claims
            
        # Parse the comma-separated list of numbers
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_claims = []
            
            for idx in indices:
                if 0 <= idx < len(claims):
                    selected_claims.append(claims[idx])
                else:
                    print(f"Warning: Index {idx+1} is out of range and will be skipped.")
            
            if not selected_claims:
                print("No valid claims were selected. Exiting.")
                return []
                
            print(f"Selected {len(selected_claims)} claims.")
            return selected_claims
            
        except ValueError:
            print("Invalid selection format. Please use comma-separated numbers or 'all'.")
            return select_claims_interactive(claims)  # Try again
            
    except KeyboardInterrupt:
        print("\nSelection canceled by user. Exiting.")
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
    }
    
    def __init__(self,
                 data_type: str,
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 output_dir: str = "experiment_novelty_output",
                 max_wait_time: int = 500,
                 dft_recommendations: bool = False,
                 enable_human_feedback: bool = False,
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
        
        self.google_api_key = google_api_key
        self.analysis_model = analysis_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dft_recommendations = dft_recommendations
        
        # Initialize data-specific analyzer
        analyzer_class = self.ANALYZER_REGISTRY[data_type]
        self.analyzer = analyzer_class(
            google_api_key=google_api_key,
            analysis_model=analysis_model,
            output_dir=str(self.output_dir),
            enable_human_feedback=enable_human_feedback,
            **analyzer_kwargs
        )
        
        # Initialize literature agent
        if futurehouse_api_key:
            self.lit_agent = OwlLiteratureAgent(
                api_key=futurehouse_api_key,
                max_wait_time=max_wait_time
            )
        else:
            self.lit_agent = None
        
        self.logger.info(f"ExperimentNoveltyAssessment initialized for {data_type} data")
    
    @classmethod
    def register_analyzer(cls, data_type: str, analyzer_class):
        """Register a new analyzer for a data type."""
        if not issubclass(analyzer_class, BaseExperimentAnalyzer):
            raise TypeError("Analyzer must inherit from BaseExperimentAnalyzer")
        cls.ANALYZER_REGISTRY[data_type] = analyzer_class
        logging.info(f"Registered analyzer for data type: {data_type}")
    
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
        
        # === Step 1: Experimental Data Analysis ===
        try:
            logging.info(f"\n\n 🔄 -------------------- WORKFLOW STEP 1: {data_type_name.upper()} ANALYSIS -------------------- 🔄\n")
            
            analysis_result = self.analyzer.analyze_for_claims(
                str(data_path), 
                system_info=system_info, 
                **analysis_kwargs
            )
            
            if "error" in analysis_result:
                logging.error(f"{data_type_name.title()} analysis step failed.")
                workflow_result["final_status"] = "failed_analysis"
                workflow_result["error"] = analysis_result["error"]
                return workflow_result
            
            logging.info(f"--- {data_type_name.title()} Analysis Result Received ---")
            print(f"\n--- {data_type_name.title()} Analysis Summary ---")
            print(analysis_result.get("detailed_analysis", "No detailed analysis text found."))
            print("-" * 22)
            
            claims = analysis_result.get("scientific_claims", [])
            if not claims:
                logging.warning("Analysis completed, but no claims were found.")
                workflow_result["final_status"] = "no_claims"
                return workflow_result
            
            # Display claims
            self._display_claims(claims, data_type_name)
            
            # Save claims
            claims_file = self.output_dir / f"generated_{data_type_name}_claims.json"
            with open(claims_file, 'w') as f:
                json.dump(claims, f, indent=2)
            logging.info(f"Claims saved to: {claims_file}")
            
            workflow_result["claims_generation"] = {
                "status": "success",
                "claims": claims,
                "claims_file": str(claims_file),
                "detailed_analysis": analysis_result.get("detailed_analysis", "")
            }
            workflow_result["steps_completed"].append("claims_generation")
            
        except Exception as e:
            logging.exception(f"An unexpected error occurred during {data_type_name} analysis step:")
            workflow_result["final_status"] = "error_analysis"
            workflow_result["error"] = str(e)
            return workflow_result
        
        # === Step 2: Literature Search ===
        if self.lit_agent is None:
            logging.warning("Literature search disabled (no FutureHouse API key)")
            workflow_result["final_status"] = "no_literature_search"
            return workflow_result
        
        try:
            logging.info("\n\n\n 🔄 ------------------------- WORKFLOW STEP 2: LITERATURE SEARCH ------------------------- 🔄\n")
            
            # Interactive claim selection
            selected_claims = select_claims_interactive(claims)
            
            if not selected_claims:
                workflow_result["final_status"] = "no_claims_selected"
                return workflow_result
            
            # Process each selected claim
            literature_results = []
            
            for i, claim in enumerate(selected_claims):
                has_anyone_question = claim.get("has_anyone_question")
                if not has_anyone_question:
                    logging.warning(f"Claim {i+1} does not have a 'has_anyone_question'. Skipping.")
                    continue
                
                print(f"\n[{i+1}/{len(selected_claims)}] Searching literature for:")
                print(f"   {has_anyone_question}")
                
                # Query literature
                owl_result = self.lit_agent.query_literature(has_anyone_question)
                
                literature_results.append({
                    "original_claim": claim,
                    "owl_result": owl_result
                })
                
                # Display result
                if owl_result["status"] == "success":
                    print(f"\n   Literature Search Complete:")
                    print(f"   {owl_result['formatted_answer']}")
                else:
                    print(f"   Literature Search Failed: {owl_result.get('message', 'Unknown error')}")
                
                print("-" * 70)
            
            # Save results
            results_file = self.output_dir / f"{data_type_name}_literature_search_results.json"
            with open(results_file, 'w') as f:
                json.dump(literature_results, f, indent=2)
            
            workflow_result["literature_search"] = {
                "status": "success",
                "results": literature_results,
                "results_file": str(results_file)
            }
            workflow_result["steps_completed"].append("literature_search")
            
        except Exception as e:
            logging.exception("An unexpected error occurred during Literature Search step:")
            workflow_result["final_status"] = "error_literature"
            workflow_result["error"] = str(e)
            return workflow_result
        
        # === Step 3: Novelty Assessment ===
        try:
            novelty_result = self._run_enhanced_novelty_assessment(literature_results, data_type_name)
            
            if novelty_result["status"] != "success":
                workflow_result["final_status"] = "error_novelty"
                workflow_result["error"] = novelty_result.get("message")
                return workflow_result
            
            novelty_assessment = novelty_result["assessment"]
            workflow_result["novelty_assessment"] = novelty_assessment
            workflow_result["steps_completed"].append("novelty_assessment")
            
        except Exception as e:
            logging.exception("An unexpected error occurred during Novelty Assessment:")
            workflow_result["final_status"] = "error_novelty"
            workflow_result["error"] = str(e)
            return workflow_result
        
        workflow_result["final_status"] = "success"
        
        # === Step 4: DFT Recommendations (Optional, mainly for microscopy) ===
        if self.dft_recommendations and self.data_type == 'microscopy':
            try:
                logging.info("\n\n\n 🔄 ------------------------- WORKFLOW STEP 4: DFT RECOMMENDATIONS ------------------------- 🔄\n")
                
                dft_result = self._generate_dft_recommendations(
                    workflow_result["claims_generation"]["detailed_analysis"],
                    workflow_result["novelty_assessment"] 
                )
                workflow_result["dft_recommendations"] = dft_result
                workflow_result["steps_completed"].append("dft_recommendations")
                
            except Exception as e:
                logging.exception("An unexpected error occurred during DFT Recommendations:")
                workflow_result["dft_recommendations"] = {
                    "status": "error",
                    "message": f"DFT recommendations failed: {str(e)}"
                }
        
        # Save workflow log
        self._save_workflow_log(data_type_name)
        
        return workflow_result
    
    def _display_claims(self, claims: List[Dict[str, Any]], data_type_name: str):
        """Display generated claims in a formatted way."""
        print(f"\n--- Generated {data_type_name.title()} Claims ---")
        
        for i, claim in enumerate(claims):
            print(f"\n[{i+1}] Claim:")
            print(f"   {claim.get('claim', 'No claim text')}")
            
            # Data-type specific evidence field
            if data_type_name == 'spectroscopy':
                evidence_key = 'spectroscopic_evidence'
            else:
                evidence_key = 'evidence'
            
            if evidence_key in claim:
                print(f"   Evidence: {claim.get(evidence_key, 'No evidence specified')}")
            
            print(f"   Scientific Impact: {claim.get('scientific_impact', 'No impact specified')}")
            print(f"   Has Anyone Question: {claim.get('has_anyone_question', 'No question formulated')}")
            print(f"   Keywords: {', '.join(claim.get('keywords', []))}")
            
            if 'confidence' in claim:
                print(f"   Confidence: {claim.get('confidence')}")
            
            print("-" * 70)
    
    def _run_enhanced_novelty_assessment(self, literature_results: List[Dict[str, Any]], 
                                       data_type_name: str) -> Dict[str, Any]:
        """Run enhanced novelty assessment with structured scoring."""
        try:
            logging.info("\n\n\n 🔄 ------------------------- WORKFLOW STEP 3: NOVELTY ASSESSMENT ------------------------- 🔄\n")
            
            # Initialize novelty scorer
            novelty_scorer = NoveltyScorer(google_api_key=self.google_api_key)
            
            # Run enhanced assessment
            novelty_assessment = enhanced_novelty_assessment(literature_results, novelty_scorer)
            
            # Display results
            display_enhanced_novelty_summary(novelty_assessment)
            
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
            logging.exception("Enhanced novelty assessment failed:")
            return {
                "status": "error",
                "message": str(e)
            }
    
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
            print(f"Warning: Could not save workflow log: {e}")
    
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
            
            # Add detailed scoring information for high-impact findings
            # detailed_scores = assessment.get("detailed_scores", [])
            # high_scoring = [r for r in detailed_scores if r.get('novelty_assessment', {}).get('novelty_score', 0) >= 4]
            
            # if high_scoring:
            #     summary += "🔍 High-Impact Findings (Score ≥4):\n"
            #     for result in high_scoring:
            #         score_info = result.get('novelty_assessment', {})
            #         score = score_info.get('novelty_score', 0)
            #         explanation = score_info.get('explanation', 'N/A')
            #         claim = result.get('original_claim', {}).get('claim', 'N/A')
                    
            #         summary += f"  • Score {score}/5: {claim}\n"
            #         summary += f"    Reasoning: {explanation}\n\n"
        
        # DFT recommendations summary
        if "dft_recommendations" in workflow_result and workflow_result["dft_recommendations"].get("status") == "success":
            dft_result = workflow_result["dft_recommendations"]
            summary += f"DFT Recommendations: {dft_result.get('total_recommendations', 0)} structures recommended\n"
        
        return summary
    
    def _generate_dft_recommendations(self, initial_analysis_text: str, novelty_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DFT recommendations based on enhanced novelty analysis (microscopy only)"""
        
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
        
        print("\n--- DFT Structure Recommendations (Enhanced Novelty-Informed) ---")
        print(reasoning_text)
        print("-" * 65)
        
        if recommendations:
            print("\n--- Recommended DFT Structures ---")
            for i, rec in enumerate(recommendations):
                print(f"\n[{i+1}] (Priority: {rec.get('priority', 'N/A')})")
                print(f"Description: {rec.get('description', 'N/A')}")
                print(f"Scientific justification: {rec.get('scientific_interest', 'N/A')}")
                print("-" * 50)
        else:
            print("\nNo DFT structure recommendations were generated.")
        
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
        
        logging.info(f"Enhanced DFT recommendations saved to: {dft_file}")
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "reasoning": reasoning_text,
            "dft_file": str(dft_file),
            "total_recommendations": len(recommendations)
        }
