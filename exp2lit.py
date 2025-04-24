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
from lit_agents.literature_agent import OwlLiteratureAgent

# Add FUTUREHOUSE_API_KEY to config if not already there
if not hasattr(config, 'FUTUREHOUSE_API_KEY'):
    config.FUTUREHOUSE_API_KEY = os.getenv("FUTUREHOUSE_API_KEY")


# Prepare FFT+NMF settings dictionary
fft_nmf_settings = {
    'FFT_NMF_ENABLED': getattr(config, 'FFT_NMF_ENABLED', False),
    'FFT_NMF_AUTO_PARAMS': getattr(config, 'FFT_NMF_AUTO_PARAMS', False),
    'FFT_NMF_WINDOW_SIZE_X': getattr(config, 'FFT_NMF_WINDOW_SIZE_X', 64),
    'FFT_NMF_WINDOW_SIZE_Y': getattr(config, 'FFT_NMF_WINDOW_SIZE_Y', 64),
    'FFT_NMF_WINDOW_STEP_X': getattr(config, 'FFT_NMF_WINDOW_STEP_X', 16),
    'FFT_NMF_WINDOW_STEP_Y': getattr(config, 'FFT_NMF_WINDOW_STEP_Y', 16),
    'FFT_NMF_INTERPOLATION_FACTOR': getattr(config, 'FFT_NMF_INTERPOLATION_FACTOR', 2),
    'FFT_NMF_ZOOM_FACTOR': getattr(config, 'FFT_NMF_ZOOM_FACTOR', 2),
    'FFT_NMF_HAMMING_FILTER': getattr(config, 'FFT_NMF_HAMMING_FILTER', True),
    'FFT_NMF_COMPONENTS': getattr(config, 'FFT_NMF_COMPONENTS', 4),
    'FFT_NMF_OUTPUT_DIR': getattr(config, 'FFT_NMF_OUTPUT_DIR', 'fft_nmf_results')
}


def select_claims_interactive(claims):
    """
    Allows user to interactively select which claims to search for in the literature.
    
    Args:
        claims: List of claim objects
        
    Returns:
        List of selected claim objects
    """
    if not claims:
        print("No claims available to select from.")
        return []
        
    print("\n--- Select Claims for Literature Search ---")
    print("Enter comma-separated numbers of claims to search, or 'all' for all claims.")
    print("Examples: '1,3,5' or 'all'")
    
    # Display the available claims
    for i, claim in enumerate(claims):
        print(f"\n[{i+1}] Claim:")
        print(f"   {claim.get('claim', 'No claim text')}")
        print(f"   Has Anyone Question: {claim.get('has_anyone_question', 'No question formulated')}")
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
                sys.exit(0)
                
            print(f"Selected {len(selected_claims)} claims.")
            return selected_claims
            
        except ValueError:
            print("Invalid selection format. Please use comma-separated numbers or 'all'.")
            return select_claims_interactive(claims)  # Try again
            
    except KeyboardInterrupt:
        print("\nSelection canceled by user. Exiting.")
        sys.exit(0)

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
            model_name=config.ANALYSIS_AGENT_MODEL,
            fft_nmf_settings=fft_nmf_settings
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
        
        # Let user select which claims to search
        selected_claims = select_claims_interactive(claims)
        
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
        lit_agent = OwlLiteratureAgent(
            api_key=config.FUTUREHOUSE_API_KEY,
            max_wait_time=config.OWL_MAX_WAIT_TIME,
        )
        
        # Process each selected claim with OWL
        literature_results = []
        print(f"\n--- Starting Literature Search for {len(selected_claims)} Selected Claims ---")
        
        for i, claim in enumerate(selected_claims):
            has_anyone_question = claim.get("has_anyone_question")
            if not has_anyone_question:
                logging.warning(f"Claim {i+1} does not have a 'has_anyone_question'. Skipping.")
                print(f"\n[{i+1}] Skipping claim due to missing question format.")
                continue
                
            print(f"\n[{i+1}/{len(selected_claims)}] Searching literature for:")
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
        print(f"Total claims analyzed: {len(selected_claims)}")
        print(f"Successfully searched: {sum(1 for r in literature_results if r['owl_result']['status'] == 'success')}")
        print(f"Failed searches: {sum(1 for r in literature_results if r['owl_result']['status'] != 'success')}")
        
        # Display novel vs. known findings
        novel_claims = []
        known_claims = []
        
        for result in literature_results:
            if result['owl_result']['status'] == "success":
                answer = json.loads(result['owl_result']["json"])['answer'].lower()
                if 'no' in answer[:3]:
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