#!/usr/bin/env python3

import sys
sys.path.append("../../scilink")

import scilink

# Configure APIs
scilink.configure('google', '')
scilink.configure('futurehouse', '')


data_path="data/tepl.npy"
system_info="data/tepl.json"


if __name__ == "__main__":
    
    workflow = scilink.workflows.ExperimentNoveltyAssessment(
        data_type='spectroscopy',
        enable_human_feedback=True,
        measurement_recommendations=True
    )

    result_unified = workflow.run_complete_workflow(
        data_path=data_path,
        system_info=system_info,
    )

    print("\n--- Workflow Summary ---")
    print(workflow.get_summary(result_unified))
