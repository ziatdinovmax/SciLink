#!/usr/bin/env python3

import sys
import logging

# Use the installed package
import scilink

scilink.configure('google', '')
scilink.configure('futurehouse', '')


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    print("--- Starting Complete Experiment → Literature → DFT Pipeline ---")

    # Step 1: Microscopy analysis with novelty assessment
    print("\n=== Step 1: Microscopy Novelty Assessment ===")
    
    novelty_workflow = scilink.MicroscopyNoveltyAssessmentWorkflow(
        analysis_model="gemini-2.5-pro-preview-06-05",
        analysis_enabled=True,
        output_dir="exp2lit2dft_novelty_output",
        dft_recommendations=True
    )

    # Define image and metadata
    image_path = "data/GH_stm.tif"
    system_info_path = "data/GH_stm.json"

    # Run novelty assessment
    novelty_result = novelty_workflow.run_complete_workflow(
        image_path=image_path,
        system_info=system_info_path
    )

    if novelty_result.get('final_status') != 'success':
        print(f"Novelty assessment failed: {novelty_result.get('final_status')}")
        sys.exit(1)

    # Step 2: Generate DFT recommendations based on novelty
    print("\n=== Step 2: DFT Recommendations Based on Novelty ===")
    
    dft_rec_workflow = scilink.DFTRecommendationsWorkflow(
        analysis_model="gemini-2.5-pro-preview-06-05",
        output_dir="exp2lit2dft_dft_output"
    )

    # Extract data from novelty assessment
    analysis_text = novelty_result["claims_generation"]["full_analysis"]
    novel_claims = novelty_result.get("novelty_assessment", {}).get("potentially_novel", [])

    # Generate DFT recommendations
    dft_result = dft_rec_workflow.run_from_data(
        analysis_text=analysis_text,
        novel_claims=novel_claims
    )

    if dft_result.get('status') == 'success':
        print(f"\n✓ Generated {len(dft_result.get('recommendations', []))} DFT recommendations")
        print(f"  Based on {len(novel_claims)} novel findings")
        print(f"  Results saved to: {dft_result.get('output_file')}")
    else:
        print(f"DFT recommendations failed: {dft_result.get('message')}")

    print("\n--- End Complete Pipeline ---")
