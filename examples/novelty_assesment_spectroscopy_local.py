#!/usr/bin/env python3

import sys
sys.path.append("../scilink")

import os
import gdown
import zipfile

import scilink

# Configure APIs
scilink.configure('google', '')
scilink.configure('futurehouse','')


# Download and extract data files
print("Downloading data files...")
gdown.download("https://drive.google.com/uc?id=1E_V7elBTOPisShXHpw-skHi4fZ_GLHbU", "data.zip", quiet=False)

print("Extracting data files...")
with zipfile.ZipFile("data.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")

# Clean up zip file
os.remove("data.zip")


data_path="data/eels_plasmon2.npy"
system_info="data/eels_plasmon2.json"
structure_image_path="data/haadf_plasmon2.npy"
structure_image_system_info="data/haadf_plasmon2.json"

if __name__ == "__main__":
    
    workflow = scilink.workflows.ExperimentNoveltyAssessment(
        data_type='spectroscopy',
        enable_human_feedback=True,
        local_model = "../../gemma3_27B_QAT_local/gemma-3-27b-it-q4_0.gguf",
    )

    result_unified = workflow.run_complete_workflow(
        data_path=data_path,
        system_info=system_info,
        structure_image_path=structure_image_path,
        structure_system_info=structure_image_system_info
    )

    print("\n--- Workflow Summary ---")
    print(workflow.get_summary(result_unified))
