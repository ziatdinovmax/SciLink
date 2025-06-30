#!/usr/bin/env python3

import sys
import logging

sys.path.append("../scilink")

import scilink


# Configure APIs
scilink.configure('google', '')
scilink.configure('futurehouse', '')


def run_spectroscopy_to_dft_workflow(data_path: str, system_info: str,
                                     structure_image_path: str, structure_image_system_info: str):
    """
    Complete workflow: Spectroscopic analysis → Novelty assessment → DFT recommendations
    """
    
    # Step 1: Run spectroscopy novelty assessment
    print("🔬 Starting Spectroscopic Novelty Assessment...")
    spectroscopy_workflow = scilink.SpectroscopyNoveltyAssessmentWorkflow(
        analysis_model="gemini-2.5-pro-preview-06-05",
        spectral_unmixing_enabled=True,
        output_dir="spectroscopy_novelty_output",
        max_wait_time=400
    )
    
    spectroscopy_result = spectroscopy_workflow.run_complete_workflow(
        data_path=data_path,
        system_info=system_info,
        structure_image_path=structure_image_path,
        structure_system_info=structure_image_system_info
    )
    
    print("📋 Spectroscopy workflow summary:")
    print(spectroscopy_workflow.get_summary(spectroscopy_result))
    
    if spectroscopy_result["final_status"] != "success":
        print("❌ Spectroscopy workflow failed, cannot proceed to DFT recommendations")
        return spectroscopy_result
    
    # Step 2: Extract data for DFT recommendations
    analysis_text = spectroscopy_result["claims_generation"]["detailed_analysis"]
    novel_claims = spectroscopy_result["novelty_assessment"]["potentially_novel"]
    
    print(f"\n🧬 Proceeding to DFT recommendations...")
    print(f"   Analysis text length: {len(analysis_text)} characters")
    print(f"   Novel claims found: {len(novel_claims)}")
    
    # Step 3: Generate DFT recommendations based on spectroscopic findings
    dft_workflow = scilink.DFTRecommendationsWorkflow(
        analysis_model="gemini-2.5-pro-preview-06-05",
        output_dir="spectroscopy_dft_output"
    )
    
    dft_result = dft_workflow.run_from_data(
        analysis_text=analysis_text,
        novel_claims=novel_claims
    )
    
    # Step 4: Display integrated results
    print(f"\n✅ Complete Spectroscopy → DFT Workflow Summary:")
    print(f"   Spectroscopic claims generated: {len(spectroscopy_result['claims_generation']['claims'])}")
    print(f"   Literature searches performed: {spectroscopy_result['novelty_assessment']['total_claims_searched']}")
    print(f"   Novel findings identified: {len(novel_claims)}")
    print(f"   DFT structures recommended: {len(dft_result.get('recommendations', []))}")
    
    return {
        "spectroscopy_result": spectroscopy_result,
        "dft_result": dft_result,
        "integration_status": "success"
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    result = run_spectroscopy_to_dft_workflow(
        data_path="data/eels_plasmon2.npy",
        system_info="data/eels_plasmon2.json",
        structure_image_path="data/haadf_plasmon2_resized.npy",
        structure_image_system_info="data/haadf_plasmon2.json"
    )
    
    print("\n🎉 Spectroscopy → DFT workflow completed!")
