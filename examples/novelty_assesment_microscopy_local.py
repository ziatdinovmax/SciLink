#!/usr/bin/env python3

import sys
sys.path.append("../scilink")

import scilink

# Configure APIs
scilink.configure('google', 'AIzaSyBc0lxq2Rk-f6e0xnG60UKV5zL3YweJkGg')
scilink.configure('futurehouse','2skcTZ36nXb9ddgrpJa7rg.platformv01.eyJqdGkiOiI5ZDhmMWI0MS01NDRiLTRkMzAtYWIwOC04ODBkMzk1OTk1MjMiLCJzdWIiOiJGSXRVRDBHanRYWUo2Y1lXOGQ3RkJhbFBHZXUxIiwiaWF0IjoxNzQ3NzAxMTAxfQ.W2AQ2Q0mfgJcCvTtXd/SP/6JqOIIGd9Z+dDWyw3T9AQ')


# Define image and metadata
image_path = "data/MoS2_stem_Exp-000-2.npy" # GO_cafm.tif, nanoparticles.npy, MoS2_stem_Exp-001-5.npy
system_info_path = "data/MoS2_stem_Exp-000-2.json" # GO_cafm.json, nanoparticles.json, MoS2_stem_Exp-001-5.json

if __name__ == "__main__":

    workflow = scilink.workflows.ExperimentNoveltyAssessment(
        data_type='microscopy',
        dft_recommendations=True,
        enable_human_feedback=True,
        local_model = "../../gemma3_27B_QAT_local/gemma-3-27b-it-q4_0.gguf",
    )

    result_unified = workflow.run_complete_workflow(
        data_path=image_path,
        system_info=system_info_path
    )

    print("\n--- Workflow Summary ---")
    print(workflow.get_summary(result_unified))