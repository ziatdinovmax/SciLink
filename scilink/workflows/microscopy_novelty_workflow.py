import os
import sys
import logging
import json
from io import StringIO
from typing import Dict, Any, List

from ..agents.exp_agents.orchestrator_agent import OrchestratorAgent, AGENT_MAP
from ..agents.exp_agents.microscopy_agent import MicroscopyAnalysisAgent
from ..agents.exp_agents.sam_microscopy_agent import SAMMicroscopyAnalysisAgent
from ..agents.exp_agents.atomistic_microscopy_agent import AtomisticMicroscopyAnalysisAgent
from ..agents.lit_agents.literature_agent import OwlLiteratureAgent

import warnings
from ..auth import get_api_key, APIKeyNotFoundError


def select_claims_interactive(claims):
    """
    Allows user to interactively select which claims to search for in the literature.
    (Simplified from exp2lit.py)
    """
    if not claims:
        print("No claims available to select from.")
        return []
        
    print("\n--- Select Claims for Literature Search ---")
    print("Enter comma-separated numbers of claims to search, or 'all' for all claims.")
    print("Examples: '1,3,5' or 'all'")
    
    # Display only the "Has Anyone" questions
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


class MicroscopyNoveltyAssessmentWorkflow:
    """
    Workflow for analyzing experimental results and assessing their novelty.
    Based on exp2lit.py structure.
    """
    def __init__(self,
                 agent_id: int | None = None,
                 google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 analysis_model: str = "gemini-2.5-pro-preview-06-05",
                 output_dir: str = "novelty_assessment_output",
                 max_wait_time: int = 500,
                 dft_recommendations: bool = False
                 ):
        """
        Initializes the workflow.

        Args:
            agent_id (int | None, optional): The ID of the agent to use. If None, the
                OrchestratorAgent will be used to select one. Defaults to None.
                The available agent IDs for microscopy are:
                - 0: `MicroscopyAnalysisAgent` (General microscopy with FFT/NMF)
                - 1: `SAMMicroscopyAnalysisAgent` (Particle analysis via SAM)
                - 2: `AtomisticMicroscopyAnalysisAgent` (Atomistic analysis with GMM)
            google_api_key (str, optional): Google API key. Defaults to auto-discovery.
            futurehouse_api_key (str, optional): FutureHouse API key. Defaults to auto-discovery.
            analysis_model (str, optional): The name of the generative AI model to use for analysis.
            output_dir (str, optional): Directory to save outputs.
            max_wait_time (int, optional): Max wait time for literature search.
            dft_recommendations (bool, optional): Whether to generate DFT recommendations.
        """
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
        
        self.agent_id = agent_id
        # Auto-discover API keys if not provided
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        if futurehouse_api_key is None:
            futurehouse_api_key = get_api_key('futurehouse')
            if not futurehouse_api_key:
                warnings.warn(
                    "FutureHouse API key not found. Literature search will be disabled. "
                    "Set FUTUREHOUSE_API_KEY environment variable or use "
                    "scilinkllm.configure('futurehouse', 'your-key') to enable literature search."
                )
        
        self.google_api_key = google_api_key
        self.analysis_model = analysis_model
        self.output_dir = output_dir
        self.dft_recommendations = dft_recommendations
        os.makedirs(output_dir, exist_ok=True)
        
        if futurehouse_api_key:
            self.lit_agent = OwlLiteratureAgent(
                api_key=futurehouse_api_key,
                max_wait_time=max_wait_time
            )
        else:
            self.lit_agent = None
    
    def run_complete_workflow(self, image_path: str, system_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete workflow. Based on exp2lit.py main workflow.
        """
        
        if isinstance(system_info, str):
            # It's a file path, load it
            import json
            with open(system_info, 'r') as f:
                system_info = json.load(f)
        elif system_info is None:
            # Use default from config
            system_info = getattr(config, 'SYSTEM_INFO', None)

        workflow_result = {
            "image_path": image_path,
            "steps_completed": [],
            "final_status": "started"
        }
        
        # === Step 1: Select and Run Analysis Agent for Scientific Claims ===
        try:
            logging.info("\n\n 🔄 -------------------- WORKFLOW STEP 1: AGENT SELECTION & AUTOMATED DATA ANALYSIS -------------------- 🔄\n")

            # --- 1a. Select Analysis Agent ---
            if self.agent_id is not None:
                if self.agent_id not in AGENT_MAP:
                    raise ValueError(f"Invalid agent_id: {self.agent_id}. Available agents are: {list(AGENT_MAP.keys())}")
                selected_agent_id = self.agent_id
                reasoning = "Agent selected manually by user."
                logging.info(f"Bypassing orchestrator. Using manually selected agent: {AGENT_MAP[selected_agent_id].__name__}")
            else:
                logging.info("Using orchestrator to select the best analysis agent...")
                orchestrator = OrchestratorAgent(google_api_key=self.google_api_key)
                selected_agent_id, reasoning = orchestrator.select_agent(
                    data_type="microscopy",
                    system_info=system_info,
                    image_path=image_path
                )

            if selected_agent_id == -1:
                logging.error(f"Agent selection failed. Reason: {reasoning}")
                workflow_result["final_status"] = "failed_orchestration"
                return workflow_result

            # --- 1b. Instantiate and Run Analysis Agent ---
            AnalysisAgentClass = AGENT_MAP[selected_agent_id]
            logging.info(f"✅ Running analysis with: {AnalysisAgentClass.__name__}")
            
            agent_kwargs = {
                'model_name': self.analysis_model
            }
            if selected_agent_id == 0: # MicroscopyAnalysisAgent
                agent_kwargs['fft_nmf_settings'] = {
                    'FFT_NMF_ENABLED': True,
                    'FFT_NMF_AUTO_PARAMS': True,
                    'components': 3,
                    'output_dir': self.output_dir
                }
            
            analysis_agent = AnalysisAgentClass(**agent_kwargs)
            analysis_result = analysis_agent.analyze_microscopy_image_for_claims(
                image_path, system_info=system_info
            )

            if "error" in analysis_result:
                logging.error("Analysis step failed.")
                workflow_result["final_status"] = "failed_analysis"
                return workflow_result

            logging.info("--- Analysis Result Received ---")
            print("\n--- Analysis Summary ---")
            print(analysis_result.get("detailed_analysis", "No detailed analysis text found."))
            print("-" * 22)

            claims = analysis_result.get("scientific_claims", [])
            if not claims:
                logging.warning("Analysis completed, but no claims were found.")
                workflow_result["final_status"] = "no_claims"
                return workflow_result

            # Format and print the claims
            print("\n--- Generated Scientific Claims ---")
            for i, claim in enumerate(claims):
                print(f"\n[{i+1}] Claim:")
                print(f"   {claim.get('claim', 'No claim text')}")
                print(f"   Scientific Impact: {claim.get('scientific_impact', 'No impact specified')}")
                print(f"   Has Anyone Question: {claim.get('has_anyone_question', 'No question formulated')}")
                print(f"   Keywords: {', '.join(claim.get('keywords', []))}")
                print("-" * 70)

            # Save claims to JSON file for reference
            claims_file = os.path.join(self.output_dir, "generated_claims.json")
            with open(claims_file, 'w') as f:
                json.dump(claims, f, indent=2)
            logging.info(f"Claims saved to: {claims_file}")
            
            workflow_result["claims_generation"] = {
                "status": "success",
                "claims": claims,
                "claims_file": claims_file,
                "detailed_analysis": analysis_result.get("detailed_analysis", "")
            }
            workflow_result["steps_completed"].append("claims_generation")
            
        except Exception as e:
            logging.exception("An unexpected error occurred during Analysis step:")
            workflow_result["final_status"] = "error_analysis"
            return workflow_result

        # === Step 2: Literature Search with OWL ===
        try:
            logging.info("\n\n\n 🔄 ------------------------- WORKFLOW STEP 2: LITERATURE SEARCH ------------------------- 🔄\n")
            
            # Let user select which claims to search
            selected_claims = select_claims_interactive(claims)
            
            if not selected_claims:
                workflow_result["final_status"] = "no_claims_selected"
                return workflow_result
            
            # Process each selected claim with OWL (same as exp2lit.py)
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
            results_file = os.path.join(self.output_dir, "literature_search_results.json")
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
            logging.info("\n\n\n 🔄 ------------------------- WORKFLOW STEP 3: NOVELTY ASSESSMENT ------------------------- 🔄\n")
            
            # Display summary of findings (same logic as exp2lit.py)
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
            novelty_file = os.path.join(self.output_dir, "novelty_assessment.json")
            with open(novelty_file, 'w') as f:
                json.dump(novelty_assessment, f, indent=2)
            
            workflow_result["novelty_assessment"] = novelty_assessment
            workflow_result["steps_completed"].append("novelty_assessment")
            
            # Display literature search summary
            print("\n--- Literature Search Summary ---")
            print(f"Total claims analyzed: {len(selected_claims)}")
            print(f"Successfully searched: {sum(1 for r in literature_results if r['owl_result']['status'] == 'success')}")
            print(f"Failed searches: {sum(1 for r in literature_results if r['owl_result']['status'] != 'success')}")
            
            if novel_claims:
                print("\nPotentially Novel Findings:")
                for i, claim in enumerate(novel_claims):
                    print(f"  {i+1}. {claim}")
                    
            if known_claims:
                print("\nPreviously Reported Findings:")
                for i, claim in enumerate(known_claims):
                    print(f"  {i+1}. {claim}")
            
            print("-" * 50)
            
        except Exception as e:
            logging.exception("An unexpected error occurred during Novelty Assessment:")
            workflow_result["final_status"] = "error_novelty"
            return workflow_result

        # === Step 4: DFT Recommendations (Optional) ===
        if self.dft_recommendations:
            try:
                logging.info("\n\n\n 🔄 ------------------------- WORKFLOW STEP 4: DFT RECOMMENDATIONS ------------------------- 🔄\n")
                
                dft_result = self._generate_dft_recommendations(
                    workflow_result["claims_generation"]["detailed_analysis"],
                    novel_claims
                )
                workflow_result["dft_recommendations"] = dft_result
                workflow_result["steps_completed"].append("dft_recommendations")
                
            except Exception as e:
                logging.exception("An unexpected error occurred during DFT Recommendations:")
                workflow_result["dft_recommendations"] = {
                    "status": "error",
                    "message": f"DFT recommendations failed: {str(e)}"
                }

        workflow_result["final_status"] = "success"

        # Save workflow log
        self._save_workflow_log()
        
        return workflow_result
    
    def _save_workflow_log(self):
        """Save the complete workflow log."""
        try:
            log_content = self.log_capture.getvalue()
            log_path = os.path.join(self.output_dir, "workflow_log.txt")
            
            with open(log_path, 'w') as f:
                f.write("Novelty Assessment Workflow Log\n")
                f.write("=" * 35 + "\n\n")
                f.write(log_content)
                
        except Exception as e:
            print(f"Warning: Could not save workflow log: {e}")
    
    def _generate_dft_recommendations(self, initial_analysis_text: str, novel_claims: List[str]) -> Dict[str, Any]:
        """Generate DFT recommendations based on novelty analysis"""
        
        # Generate novelty context for DFT recommendations
        if novel_claims:
            novelty_section_text = "The following claims/observations, derived from the initial image analysis, have been identified through literature review as potentially novel or under-explored:\n"
            for claim in novel_claims:
                novelty_section_text += f"- {claim}\n"
            novelty_section_text += ("\nYour primary goal for the DFT recommendations below should be to propose structures and simulations "
                                   "that can rigorously investigate these novel aspects. Explain this link clearly in your scientific "
                                   "justification for each recommended structure. If some image features are significant but not covered "
                                   "by these novel claims, you may also include recommendations for them, but prioritize novelty.")
            novelty_context = novelty_section_text
        else:
            novelty_context = "No specific novel claims were identified or prioritized from literature search. Please make DFT recommendations based on the most scientifically interesting aspects of the provided initial image analysis."
        
        # Instantiate a text-only analysis agent for this step.
        # This ensures we use the correct agent with the text-only prompt,
        # regardless of which agent was used for the initial image analysis.
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
        
        print("\n--- DFT Structure Recommendations (Based on Novelty Analysis) ---")
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
        dft_file = os.path.join(self.output_dir, "dft_recommendations.json")
        dft_output = {
            "reasoning_for_recommendations": reasoning_text,
            "recommendations": recommendations,
            "novelty_context": novelty_context
        }
        with open(dft_file, 'w') as f:
            json.dump(dft_output, f, indent=2)
        
        logging.info(f"DFT recommendations saved to: {dft_file}")
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "reasoning": reasoning_text,
            "dft_file": dft_file,
            "total_recommendations": len(recommendations)
        }
    
    def get_summary(self, workflow_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of the workflow results."""
        
        summary = f"Novelty Assessment Summary\n{'='*25}\n"
        summary += f"Image: {os.path.basename(workflow_result.get('image_path', 'Unknown'))}\n"
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
                summary += "Potentially Novel Findings:\n"
                for i, claim in enumerate(novel_claims, 1):
                    summary += f"  {i}. {claim}\n"
                summary += "\n"
                
            if known_claims:
                summary += "Previously Reported Findings:\n"
                for i, claim in enumerate(known_claims, 1):
                    summary += f"  {i}. {claim}\n"
                summary += "\n"
        
        if "dft_recommendations" in workflow_result and workflow_result["dft_recommendations"].get("status") == "success":
            dft_result = workflow_result["dft_recommendations"]
            summary += f"DFT Recommendations: {dft_result.get('total_recommendations', 0)} structures recommended\n"
        
        return summary