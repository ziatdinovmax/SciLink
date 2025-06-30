#!/usr/bin/env python3

import os
import sys
import logging

import sys
sys.path.append("../scilink")

# Use the installed package
import scilink


scilink.configure('google', '')
scilink.configure('futurehouse', '')


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    # Define image and metadata
    image_path = "data/MoS2_stem.npy"
    system_info_path = "data/MoS2_stem.json"

    if not os.path.exists(image_path) or not os.path.exists(system_info_path):
        print(f"Error: Make sure data files exist at the specified paths.")
        print(f"  Image: {image_path}")
        print(f"  Metadata: {system_info_path}")
        sys.exit(1)

    # --- Example 1: Orchestrated Workflow ---
    # The workflow will automatically select the best analysis agent.
    print("\n" + "="*50)
    print("🚀 RUNNING WORKFLOW 1: ORCHESTRATED AGENT SELECTION")
    print("="*50)

    orchestrated_workflow = scilink.MicroscopyNoveltyAssessmentWorkflow(
        analysis_model="gemini-2.5-pro-preview-06-05",
        output_dir="orchestrated_novelty_output",
        dft_recommendations=True  # Run the full pipeline including DFT
    )

    result_orchestrated = orchestrated_workflow.run_complete_workflow(
        image_path=image_path,
        system_info=system_info_path
    )

    print("\n--- Orchestrated Workflow Summary ---")
    print(orchestrated_workflow.get_summary(result_orchestrated))

    # --- Example 2: Manual Agent Selection ---
    # Forcing the workflow to use a specific agent (e.g., Agent 0 for general analysis).
    print("\n" + "="*50)
    print("🚀 RUNNING WORKFLOW 2: MANUAL AGENT SELECTION (Agent 0)")
    print("="*50)

    manual_workflow = scilink.MicroscopyNoveltyAssessmentWorkflow(
        agent_id=0,  # Manually select the General Microscopy Agent
        analysis_model="gemini-2.5-pro-preview-06-05",
        output_dir="manual_novelty_output",
        dft_recommendations=True
    )

    result_manual = manual_workflow.run_complete_workflow(
        image_path=image_path,
        system_info=system_info_path
    )

    print("\n--- Manual Workflow Summary ---")
    print(manual_workflow.get_summary(result_manual))
