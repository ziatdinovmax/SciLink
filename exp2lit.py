#!/usr/bin/env python3

import sys
import logging
import pprint
import os
import json
import time

try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure it exists.")
    sys.exit(1)

from exp_agents.microscopy_agent import GeminiMicroscopyAnalysisAgent as AnalysisAgent
from lit_agents.literature_agent import OWLLiteratureAgent

# Add FUTUREHOUSE_API_KEY to config if not already there
if not hasattr(config, 'FUTUREHOUSE_API_KEY'):
    config.FUTUREHOUSE_API_KEY = os.getenv("FUTUREHOUSE_API_KEY")

# ===========================================
#                  MAIN WORKFLOW
# ===========================================
if __name__ == "__main__":

    # --- Use Config for Logging ---
    logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

    # === Step 1: Image Analysis for Scientific Claims ===
    try:
        logging.info("--- Starting Step 1: Image Analysis for Scientific Claims ---")
        logging.info(f"Initializing analysis agent with model: {config.ANALYSIS_AGENT_MODEL}")
        analysis_agent = AnalysisAgent(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.ANALYSIS_AGENT_MODEL
        )

        logging.info(f"Analyzing image for claims: {config.IMAGE_PATH}...")
        analysis_result = analysis_agent.analyze_microscopy_image_for_claims(
            config.IMAGE_PATH,
            system_info=config.SYSTEM_INFO
        )

        logging.info("--- Analysis Result Received ---")
        if "error" in analysis_result:
            logging.error("Analysis step failed.")
            print("\n--- Analysis Failed ---")
            pprint.pprint(analysis_result)
            print("-" * 21)
            sys.exit(1)

        logging.debug("Full Analysis JSON:\n%s", pprint.pformat(analysis_result))
        print("\n--- Analysis Summary ---")
        print(analysis_result.get("full_analysis", "No detailed analysis text found."))
        print("-" * 22)

        # Extract and validate claims
        claims = analysis_result.get("claims", [])
        if not claims:
            logging.warning("Analysis completed, but no claims were found.")
            print("\nNo claims were generated from the analysis. Cannot proceed to literature search.")
            sys.exit(1)

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
        claims_file = "generated_claims.json"
        with open(claims_file, 'w') as f:
            json.dump(claims, f, indent=2)
        logging.info(f"Claims saved to: {claims_file}")
        
    except ValueError as ve:
        logging.error(f"Configuration Error during Analysis step: {ve}")
        sys.exit(1)
    except Exception as e:
        logging.exception("An unexpected error occurred during Analysis step:")
        sys.exit(1)

    # === Step 2: Literature Search with OWL ===
    try:
        logging.info("--- Starting Step 2: Literature Search with OWL ---")
        
        # Initialize the literature agent
        logging.info("Initializing OWL literature agent...")
        lit_agent = OWLLiteratureAgent(
            api_key=config.FUTUREHOUSE_API_KEY,
            max_wait_time=config.OWL_MAX_WAIT_TIME,
        )
        
        # Process each claim with OWL
        literature_results = []
        print("\n--- Starting Literature Search for Each Claim ---")
        
        for i, claim in enumerate(claims):
            has_anyone_question = claim.get("has_anyone_question")
            if not has_anyone_question:
                logging.warning(f"Claim {i+1} does not have a 'has_anyone_question'. Skipping.")
                print(f"\n[{i+1}] Skipping claim due to missing question format.")
                continue
                
            print(f"\n[{i+1}] Searching literature for:")
            print(f"   {has_anyone_question}")
            
            # Query OWL for this claim
            owl_result = lit_agent.query_literature(has_anyone_question)
            
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
        results_file = "literature_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(literature_results, f, indent=2)
            
        logging.info(f"Complete literature search results saved to: {results_file}")
        print(f"\nComplete literature search results saved to: {results_file}")
        
        # Display summary of findings
        print("\n--- Literature Search Summary ---")
        print(f"Total claims analyzed: {len(claims)}")
        print(f"Successfully searched: {sum(1 for r in literature_results if r['owl_result']['status'] == 'success')}")
        print(f"Failed searches: {sum(1 for r in literature_results if r['owl_result']['status'] != 'success')}")
        
        # Display novel vs. known findings
        novel_claims = []
        known_claims = []
        
        for result in literature_results:
            if result['owl_result']['status'] == 'success':
                answer = result['owl_result']['formatted_answer'].lower()
                # Simple heuristic - can be improved with more sophisticated analysis
                if any(phrase in answer for phrase in ["no evidence", "no research", "not found", "no studies"]):
                    novel_claims.append(result['original_claim']['claim'])
                else:
                    known_claims.append(result['original_claim']['claim'])
        
        if novel_claims:
            print("\nPotentially Novel Findings:")
            for i, claim in enumerate(novel_claims):
                print(f"  {i+1}. {claim}")
                
        if known_claims:
            print("\nPreviously Reported Findings:")
            for i, claim in enumerate(known_claims):
                print(f"  {i+1}. {claim}")
                
    except ValueError as ve:
        logging.error(f"Configuration Error during Literature Search step: {ve}")
    except Exception as e:
        logging.exception("An unexpected error occurred during Literature Search step:")

    logging.info("--- End Complete Workflow ---")
    print("\n--- End Complete Workflow ---")