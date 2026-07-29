#!/usr/bin/env python3

import sys
sys.path.append("../")

import scilink

# Configure APIs
scilink.configure('google', '') #I suggest change 'google' to LLM key
scilink.configure('futurehouse', '')


# Define image and metadata
image_path = "data/MoS2_stem_Exp-001-5.npy" # GO_cafm.tif, nanoparticles.npy
system_info_path = "data/MoS2_stem_Exp-001-5.json" # GO_cafm.json, nanoparticles.json

if __name__ == "__main__":

    workflow = scilink.workflows.ExperimentNoveltyAssessment(
        data_type='microscopy',
        enable_human_feedback=True,
        dft_recommendations=True,
        measurement_recommendations=True,
    )

    result_unified = workflow.run_complete_workflow(
        data_path=image_path,
        system_info=system_info_path
    )

    print("\n--- Workflow Summary ---")
    print(workflow.get_summary(result_unified))
