#!/usr/bin/env python3

import sys
import logging

# Use the installed package
import scilink


scilink.configure('google', 'AIzaSyB9lyiu0D9gZfHyAfKPIFkcU3a7sa9V-kE')
scilink.configure('futurehouse', '+MZqvbTtjHVywIJ1GWJ8Zw.platformv01.eyJqdGkiOiI1MDZiZjI2OS0wNThmLTRjNDUtYmM1OC1iMDE2NjYyYTBjMGUiLCJzdWIiOiJuaUt3MDBwVk1nUmV4MDhocUg3RTBTRFVXQ3UyIiwiaWF0IjoxNzQ0NzM4OTA5fQ.9xtT+1ZfVaKWHQurUAV69viXqaTh7YSH9nmDZ0DjnQU')


def run_spectroscopy_to_dft_workflow(data_path: str, system_info: str):
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
        system_info=system_info
    )
    
    print("📋 Spectroscopy workflow summary:")
    print(spectroscopy_workflow.get_summary(spectroscopy_result))
    
    if spectroscopy_result["final_status"] != "success":
        print("❌ Spectroscopy workflow failed, cannot proceed to DFT recommendations")
        return spectroscopy_result
    
    # Step 2: Extract data for DFT recommendations
    analysis_text = spectroscopy_result["claims_generation"]["full_analysis"]
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
        data_path="/Users/ziat263/code/SciLinkLLM/eels.npy",
        system_info="/Users/ziat263/code/SciLinkLLM/eels.json"
    )
    
    print("\n🎉 Spectroscopy → DFT workflow completed!")