#!/usr/bin/env python3

import sys
import logging
import pprint
import json

import config

from exp_agents.microscopy_agent import GeminiMicroscopyAnalysisAgent as AnalysisAgent


# ===========================================
#                  MAIN WORKFLOW
# ===========================================
if __name__ == "__main__":

    # --- Use Config for Logging ---
    logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

    # === Generate Scientific Claims from Image Analysis ===
    try:
        logging.info("--- Starting Image Analysis for Scientific Claims ---")
        logging.info(f"Initializing analysis agent with model: {config.ANALYSIS_AGENT_MODEL}")
        analysis_agent = AnalysisAgent(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.ANALYSIS_AGENT_MODEL,
            fft_nmf_settings=config.FFT_NMF_SETTINGS
        )

        logging.info(f"Analyzing image for claims: {config.IMAGE_PATH}...")
        analysis_result = analysis_agent.analyze_microscopy_image_for_claims(
            config.IMAGE_PATH,
            system_info=config.SYSTEM_INFO
        )

        logging.info("--- Analysis Result Received ---")
        if "error" not in analysis_result:
             logging.debug("Full Analysis JSON:\n%s", pprint.pformat(analysis_result))
             print("\n--- Analysis Summary ---")
             print(analysis_result.get("full_analysis", "No detailed analysis text found."))
             print("-" * 22)

             # Format and print the claims
             if analysis_result.get("claims"):
                  print("\n--- Generated Scientific Claims ---")
                  for i, claim in enumerate(analysis_result["claims"]):
                      print(f"\n[{i+1}] Claim:")
                      print(f"   {claim.get('claim', 'No claim text')}")
                      print(f"   Scientific Impact: {claim.get('scientific_impact', 'No impact specified')}")
                      print(f"   Has Anyone Question: {claim.get('has_anyone_question', 'No question formulated')}")
                      print(f"   Keywords: {', '.join(claim.get('keywords', []))}")
                      print("-" * 70)
                  
                  # Save to JSON file for further use
                  output_file = "generated_claims.json"
                  with open(output_file, 'w') as f:
                      json.dump(analysis_result["claims"], f, indent=2)
                  logging.info(f"Claims saved to: {output_file}")
                  print(f"\nClaims saved to: {output_file}")
             else:
                  logging.warning("Analysis completed, but no claims were found.")
        else:
             logging.error("Analysis step failed.")
             print("\n--- Analysis Failed ---")
             pprint.pprint(analysis_result)
             print("-" * 21)

    except ValueError as ve:
        logging.error(f"Configuration Error during Analysis step: {ve}")
    except Exception as e:
        logging.exception("An unexpected error occurred during Analysis step:")

    logging.info("--- End Claims Generation Workflow ---")
    print("\n--- End Claims Generation Workflow ---")