import os
import sys
import logging
import json
from io import StringIO
from typing import Dict, Any, List

from ..agents.exp_agents.spectroscopy_agent import GeminiSpectroscopyAnalysisAgent
from ..agents.lit_agents.literature_agent import OwlLiteratureAgent

import warnings
from ..auth import get_api_key, APIKeyNotFoundError


def select_claims_interactive(claims):
    """
    Allows user to interactively select which claims to search for in the literature.
    (Adapted from exp2lit.py)
    """
    if not claims:
        print("No claims available to select from.")
        return []
        
    print("\n--- Select Claims for Literature Search ---")
    print("Enter comma-separated numbers of claims to search, or 'all' for all claims.")
    print("Examples: '1,3,5' or 'all'")
    
    # Display only the "Has Anyone" questions for spectroscopic claims
    for i, claim in enumerate(claims):
        print(f"\n[{i+1}] {claim.get('has_anyone_question', 'No question formulated')}")
        print(f"    Evidence: {claim.get('spectroscopic_evidence', 'No evidence specified')[:100]}...")
    
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


class SpectroscopyNoveltyAssessmentWorkflow:
    """
    Workflow for analyzing spectroscopic experimental results and assessing their novelty.
    Based on MicroscopyNoveltyAssessmentWorkflow structure but adapted for hyperspectral data.
    Focuses purely on novelty assessment through literature validation.
    """
    
    def __init__(self, 
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 output_dir: str = "spectroscopy_novelty_output",
                 max_wait_time: int = 400,
                 spectral_unmixing_enabled: bool = True):
        
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
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Fixed spectral unmixing settings
        spectral_settings = {
            'method': 'nmf',
            'n_components': 4,  # Default value, will be auto-determined if auto_components=True
            'normalize': True,
            'enabled': spectral_unmixing_enabled,
            'auto_components': True,  # Always use auto component selection
            'max_iter': 500
        }
        
        # Initialize agents
        self.analysis_agent = GeminiSpectroscopyAnalysisAgent(
            api_key=google_api_key,
            model_name=analysis_model,
            spectral_unmixing_settings=spectral_settings,
            output_dir=output_dir
        )
        
        if futurehouse_api_key:
            self.lit_agent = OwlLiteratureAgent(
                api_key=futurehouse_api_key,
                max_wait_time=max_wait_time
            )
        else:
            self.lit_agent = None
    
    def run_complete_workflow(self, data_path: str, system_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete spectroscopy novelty assessment workflow.
        """

        workflow_result = {
            "data_path": data_path,
            "steps_completed": [],
            "final_status": "started"
        }
        
        # === Step 1: Spectroscopic Analysis for Scientific Claims ===
        try:
            logging.info("--- Starting Step 1: Spectroscopic Analysis for Scientific Claims ---")
            
            analysis_result = self.analysis_agent.analyze_hyperspectral_data_for_claims(
                data_path, metadata_path=system_info
            )

            if "error" in analysis_result:
                logging.error("Spectroscopic analysis step failed.")
                workflow_result["final_status"] = "failed_analysis"
                return workflow_result

            logging.info("--- Spectroscopic Analysis Result Received ---")
            print("\n--- Spectroscopic Analysis Summary ---")
            print(analysis_result.get("full_analysis", "No detailed analysis text found."))
            print("-" * 22)

            claims = analysis_result.get("claims", [])
            if not claims:
                logging.warning("Analysis completed, but no claims were found.")
                workflow_result["final_status"] = "no_claims"
                return workflow_result

            # Format and print the spectroscopic claims
            print("\n--- Generated Spectroscopic Claims ---")
            for i, claim in enumerate(claims):
                print(f"\n[{i+1}] Claim:")
                print(f"   {claim.get('claim', 'No claim text')}")
                print(f"   Spectroscopic Evidence: {claim.get('spectroscopic_evidence', 'No evidence specified')}")
                print(f"   Scientific Impact: {claim.get('scientific_impact', 'No impact specified')}")
                print(f"   Has Anyone Question: {claim.get('has_anyone_question', 'No question formulated')}")
                print(f"   Keywords: {', '.join(claim.get('keywords', []))}")
                if 'confidence' in claim:
                    print(f"   Confidence: {claim.get('confidence')}")
                print("-" * 70)

            # Save claims to JSON file for reference
            claims_file = os.path.join(self.output_dir, "generated_spectroscopy_claims.json")
            with open(claims_file, 'w') as f:
                json.dump(claims, f, indent=2)
            logging.info(f"Spectroscopic claims saved to: {claims_file}")
            
            workflow_result["claims_generation"] = {
                "status": "success",
                "claims": claims,
                "claims_file": claims_file,
                "full_analysis": analysis_result.get("full_analysis", "")
            }
            workflow_result["steps_completed"].append("claims_generation")
            
        except Exception as e:
            logging.exception("An unexpected error occurred during Spectroscopic Analysis step:")
            workflow_result["final_status"] = "error_analysis"
            return workflow_result

        # === Step 2: Literature Search with OWL ===
        try:
            logging.info("--- Starting Step 2: Literature Search with OWL ---")
            
            # Let user select which claims to search
            selected_claims = select_claims_interactive(claims)
            
            if not selected_claims:
                workflow_result["final_status"] = "no_claims_selected"
                return workflow_result
            
            # Process each selected claim with OWL
            literature_results = []
            
            for i, claim in enumerate(selected_claims):
                has_anyone_question = claim.get("has_anyone_question")
                if not has_anyone_question:
                    logging.warning(f"Claim {i+1} does not have a 'has_anyone_question'. Skipping.")
                    continue
                    
                print(f"\n[{i+1}/{len(selected_claims)}] Searching literature for:")
                print(f"   {has_anyone_question}")
                
                # Query OWL for this claim
                owl_result = self.lit_agent.query_literature(has_anyone_question)
                
                # Store the result with the original claim
                literature_results.append({
                    "original_claim": claim,
                    "owl_result": owl_result
                })
                
                # Display the result
                if owl_result["status"] == "success":
                    print(f"\n   OWL Search Complete:")
                    print(f"   {owl_result['formatted_answer']}")
                else:
                    print(f"   OWL Search Failed: {owl_result.get('message', 'Unknown error')}")
                
                print("-" * 70)
            
            # Save comprehensive results to a file
            results_file = os.path.join(self.output_dir, "spectroscopy_literature_search_results.json")
            with open(results_file, 'w') as f:
                json.dump(literature_results, f, indent=2)
            
            workflow_result["literature_search"] = {
                "status": "success",
                "results": literature_results,
                "results_file": results_file
            }
            workflow_result["steps_completed"].append("literature_search")
            
        except Exception as e:
            logging.exception("An unexpected error occurred during Literature Search step:")
            workflow_result["final_status"] = "error_literature"
            return workflow_result

        # === Step 3: Novelty Assessment ===
        novel_claims = []
        try:
            logging.info("--- Starting Step 3: Novelty Assessment ---")
            
            # Display summary of findings
            known_claims = []
            
            for result in literature_results:
                if result['owl_result']['status'] == "success":
                    answer = json.loads(result['owl_result']["json"])['answer'].lower()
                    if 'no' in answer[:3]:
                        novel_claims.append(result['original_claim']['claim'])
                    else:
                        known_claims.append(result['original_claim']['claim'])
            
            # Generate novelty assessment
            novelty_assessment = {
                "total_claims_searched": len(literature_results),
                "successful_searches": sum(1 for r in literature_results if r['owl_result']['status'] == 'success'),
                "potentially_novel": novel_claims,
                "known_findings": known_claims
            }
            
            # Save novelty assessment
            novelty_file = os.path.join(self.output_dir, "spectroscopy_novelty_assessment.json")
            with open(novelty_file, 'w') as f:
                json.dump(novelty_assessment, f, indent=2)
            
            workflow_result["novelty_assessment"] = novelty_assessment
            workflow_result["steps_completed"].append("novelty_assessment")
            
            # Display literature search summary
            print("\n--- Spectroscopic Literature Search Summary ---")
            print(f"Total claims analyzed: {len(selected_claims)}")
            print(f"Successfully searched: {sum(1 for r in literature_results if r['owl_result']['status'] == 'success')}")
            print(f"Failed searches: {sum(1 for r in literature_results if r['owl_result']['status'] != 'success')}")
            
            if novel_claims:
                print("\nPotentially Novel Spectroscopic Findings:")
                for i, claim in enumerate(novel_claims):
                    print(f"  {i+1}. {claim}")
                    
            if known_claims:
                print("\nPreviously Reported Spectroscopic Findings:")
                for i, claim in enumerate(known_claims):
                    print(f"  {i+1}. {claim}")
            
            print("-" * 50)
            
        except Exception as e:
            logging.exception("An unexpected error occurred during Novelty Assessment:")
            workflow_result["final_status"] = "error_novelty"
            return workflow_result

        workflow_result["final_status"] = "success"

        # Save workflow log
        self._save_workflow_log()
        
        return workflow_result
    
    def _save_workflow_log(self):
        """Save the complete workflow log."""
        try:
            log_content = self.log_capture.getvalue()
            log_path = os.path.join(self.output_dir, "spectroscopy_workflow_log.txt")
            
            with open(log_path, 'w') as f:
                f.write("Spectroscopy Novelty Assessment Workflow Log\n")
                f.write("=" * 45 + "\n\n")
                f.write(log_content)
                
        except Exception as e:
            print(f"Warning: Could not save workflow log: {e}")
    
    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the workflow results."""
        
        summary = f"Spectroscopy Novelty Assessment Summary\n{'='*35}\n"
        summary += f"Data: {os.path.basename(workflow_result.get('data_path', 'Unknown'))}\n"
        summary += f"Status: {workflow_result.get('final_status', 'Unknown')}\n"
        summary += f"Steps completed: {', '.join(workflow_result.get('steps_completed', []))}\n\n"
        
        if "novelty_assessment" in workflow_result:
            assessment = workflow_result["novelty_assessment"]
            summary += f"Literature Search Results:\n"
            summary += f"  Total claims searched: {assessment.get('total_claims_searched', 0)}\n"
            summary += f"  Successful searches: {assessment.get('successful_searches', 0)}\n\n"
            
            novel_claims = assessment.get("potentially_novel", [])
            known_claims = assessment.get("known_findings", [])
            
            if novel_claims:
                summary += "Potentially Novel Spectroscopic Findings:\n"
                for i, claim in enumerate(novel_claims, 1):
                    summary += f"  {i}. {claim}\n"
                summary += "\n"
                
            if known_claims:
                summary += "Previously Reported Spectroscopic Findings:\n"
                for i, claim in enumerate(known_claims, 1):
                    summary += f"  {i}. {claim}\n"
                summary += "\n"
        
        return summary