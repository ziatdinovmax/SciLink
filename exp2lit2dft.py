#!/usr/bin/env python3
# exp2lit2dft.py

import sys
import logging
import pprint
import json
import os

import config
from exp_agents.microscopy_agent import GeminiMicroscopyAnalysisAgent
from lit_agents.literature_agent import OwlLiteratureAgent
from sim_agents.structure_agent import StructureGenerator
from sim_agents.val_agent import StructureValidatorAgent 
from sim_agents.utils import ask_user_proceed_or_refine


def select_claims_interactive(claims: list) -> list:
    """
    Allows user to interactively select which claims to search for in the literature.
    """
    if not claims:
        print("No claims available to select from.")
        return []

    print("\n--- Select Claims for Literature Search ---")
    print("Enter comma-separated numbers of claims to search (e.g., '1,3,5'), 'all', or 'none'.")

    for i, claim_obj in enumerate(claims):
        print(f"\n[{i+1}] Claim: {claim_obj.get('claim', 'N/A')}")
        print(f"    Impact: {claim_obj.get('scientific_impact', 'N/A')}")
        print(f"    Question: {claim_obj.get('has_anyone_question', 'N/A')}")
        print(f"    Keywords: {', '.join(claim_obj.get('keywords', []))}")
        print("-" * 70)

    while True:
        try:
            selection = input("\nSelect claims to search (or 'all'/'none'): ").strip().lower()
            if not selection: 
                print("No selection made. Please enter choices, 'all', or 'none'.")
                continue
            if selection == 'none':
                print("No claims selected for literature search.")
                return []
            if selection == 'all':
                print(f"Selected all {len(claims)} claims for literature search.")
                return claims

            indices_str = selection.split(',')
            if not all(s.strip().isdigit() for s in indices_str if s.strip()): 
                print("Invalid input. Please use comma-separated numbers only (e.g., '1,3,5').")
                continue

            indices = [int(idx.strip()) - 1 for idx in indices_str if idx.strip()]
            selected_claims_list = []
            
            for idx in indices:
                if 0 <= idx < len(claims):
                    if claims[idx] not in selected_claims_list: 
                        selected_claims_list.append(claims[idx])
                else:
                    print(f"Warning: Index {idx + 1} is out of range and will be skipped.")
            
            if not selected_claims_list:
                print("No valid claims were selected based on input. Please try again or type 'none'.")
                continue 

            print(f"\n--- You have selected {len(selected_claims_list)} Claims for Literature Search ---")
            for i, claim_obj in enumerate(selected_claims_list):
                 print(f"  [{i+1}] {claim_obj.get('claim')}")
            print("-" * 70)
            confirm = input("Confirm selection? (yes/no): ").strip().lower()
            if confirm == 'yes':
                return selected_claims_list
            else:
                print("Selection not confirmed. Please re-select.")
        except ValueError:
            print("Invalid input format. Please use comma-separated numbers, 'all', or 'none'.")
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            return []
        except EOFError:
            print("\nInput stream closed. No claims selected.")
            return []

def select_recommendation_interactive(recommendations: list) -> dict | None:
    """
    Displays recommendations and prompts the user to select one.
    """
    if not recommendations:
        print("No DFT recommendations available to select from.")
        return None

    print("\n--- Available DFT Structure Recommendations ---")
    for i, rec in enumerate(recommendations):
        priority = rec.get('priority', 'N/A')
        description = rec.get('description', 'N/A')
        justification = rec.get('scientific_interest', 'N/A')
        print(f"\n[{i+1}] (Priority: {priority})")
        print(f"  Description: {description}")
        print(f"  Scientific Justification: {justification}")
        print("-" * 70)

    while True:
        try:
            if not sys.stdin.isatty(): 
                print("\nWarning: Non-interactive environment. Selecting highest priority recommendation by default.")
                logging.warning("Non-interactive environment. Selecting highest priority DFT recommendation.")
                return recommendations[0] if recommendations else None

            choice = input(f"Enter the number (1-{len(recommendations)}) to select, or 'q' to quit: ").strip().lower()
            if choice == 'q':
                print("Selection cancelled.")
                return None
            if not choice.isdigit():
                print("Invalid input. Please enter a number.")
                continue

            choice_index = int(choice) - 1
            if 0 <= choice_index < len(recommendations):
                selected = recommendations[choice_index]
                print(f"\n--- Selection Confirmed: Recommendation #{choice_index + 1} ---")
                pprint.pprint(selected)
                print("-" * 70)
                return selected
            else:
                print(f"Invalid number. Please enter between 1 and {len(recommendations)}.")
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            return None
        except EOFError:
             print("\nInput stream closed. Selecting highest priority recommendation by default.")
             logging.warning("Input stream closed. Selecting highest priority DFT recommendation.")
             return recommendations[0] if recommendations else None
        except Exception as e:
            logging.error(f"Error during DFT recommendation selection: {e}")
            return None


def generate_additional_context_for_dft_prompt(initial_analysis_text: str, novel_claims_details: list[str]) -> str | None:
    if not novel_claims_details: 
        return None
    novelty_section_text = "The following claims/observations, derived from the initial image analysis, have been identified through literature review as potentially novel or under-explored:\n"
    for detail in novel_claims_details:
        novelty_section_text += f"- {detail}\n"
    novelty_section_text += ("\nYour primary goal for the DFT recommendations below should be to propose structures and simulations "
                             "that can rigorously investigate these novel aspects. Explain this link clearly in your scientific "
                             "justification for each recommended structure. If some image features are significant but not covered "
                             "by these novel claims, you may also include recommendations for them, but prioritize novelty.")
    context_to_append = f"""
**Recap of Initial Image Findings (for context):**
The image was initially analyzed as follows: "{initial_analysis_text}"

**Focus for Current DFT Recommendations (Incorporating Literature Review Insights):**
{novelty_section_text}
"""
    return context_to_append
# --- End Helper Functions ---

# ===========================================
#         MAIN exp2lit2dft WORKFLOW
# ===========================================
if __name__ == "__main__":
    logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)
    logger = logging.getLogger(__name__) # Use a named logger
    logger.info("--- Starting Experiment to Literature to DFT (exp2lit2dft) Workflow ---")

    # Initialize agents
    analysis_agent = None
    lit_agent = None
    structure_generator = None
    structure_validator = None 

    try:
        analysis_agent = GeminiMicroscopyAnalysisAgent(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.ANALYSIS_AGENT_MODEL,
            fft_nmf_settings=config.FFT_NMF_SETTINGS
        )
        if not config.FUTUREHOUSE_API_KEY:
            logger.warning("FUTUREHOUSE_API_KEY not found in config or environment. Literature search (Step 3) will be skipped.")
        else:
            lit_agent = OwlLiteratureAgent(
                api_key=config.FUTUREHOUSE_API_KEY,
                max_wait_time=config.OWL_MAX_WAIT_TIME
            )
        structure_generator = StructureGenerator(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.GENERATOR_AGENT_MODEL,
            executor_timeout=config.GENERATOR_SCRIPT_TIMEOUT,
            generated_script_dir=config.GENERATED_SCRIPT_DIR,
            mp_api_key=getattr(config, 'MP_API_KEY', None)
        )
        # Initialize StructureValidatorAgent here as it will be needed in Step 6
        structure_validator = StructureValidatorAgent(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.VALIDATOR_AGENT_MODEL 
        )
        
        logger.info("All agents initialized successfully.")
    except ValueError as e:
        logger.error(f"Error initializing agents: {e}. Please check your API keys and configuration.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred during agent initialization:")
        sys.exit(1)

    # --- STEP 1: Initial Comprehensive Image Analysis for Claims ---
    initial_comprehensive_analysis_text = "No detailed analysis provided in Step 1."
    scientific_claims = []
    step1_success = False
    try:
        logger.info("--- Step 1: Initial Image Analysis for Claims & Comprehensive Overview ---")
        initial_analysis_claims_result = analysis_agent.analyze_microscopy_image_for_claims(
            config.IMAGE_PATH,
            system_info=config.SYSTEM_INFO
        )
        if "error" in initial_analysis_claims_result:
            logger.error(f"Step 1 Failed: Initial image analysis. Error: {initial_analysis_claims_result.get('details', initial_analysis_claims_result.get('error'))}")
        else:
            initial_comprehensive_analysis_text = initial_analysis_claims_result.get("full_analysis", initial_comprehensive_analysis_text)
            scientific_claims = initial_analysis_claims_result.get("claims", [])
            claims_output_file = "exp2lit2dft_initial_claims.json"
            with open(claims_output_file, 'w') as f:
                json.dump(initial_analysis_claims_result, f, indent=2)
            logger.info(f"Initial analysis and claims saved to: {claims_output_file}")
            if not scientific_claims:
                logger.warning("No scientific claims generated in Step 1. Subsequent DFT recommendations will be general.")
            else:
                logger.info(f"Generated {len(scientific_claims)} claims from initial analysis.")
                print("\n--- Initial Analysis Summary (from Step 1) ---")
                print(initial_comprehensive_analysis_text)
            step1_success = True
    except Exception as e:
        logger.exception("An unexpected error occurred during Step 1 (Initial Image Analysis):")
    
    if not step1_success:
        logger.error("Exiting due to failure in Step 1.")
        sys.exit(1)

    # --- STEP 2: Claim Selection for Literature Search ---
    selected_claims_for_lit_search = []
    if scientific_claims and lit_agent:
        logger.info("\n--- Step 2: Claim Selection for Literature Search ---")
        selected_claims_for_lit_search = select_claims_interactive(scientific_claims)
    elif not lit_agent:
        logger.info("Skipping Step 2 & 3 (Literature Search) as FutureHouse API key is not configured.")
    else: 
        logger.info("Skipping Step 2 (Claim Selection) as no claims were generated in Step 1.")

    novel_claims_details_for_dft_context = [] 
    step3_success = True # Assume success unless an error occurs or no search is done

    if not selected_claims_for_lit_search:
        logger.info("No claims selected or available for literature search. DFT recommendations will rely solely on initial general analysis.")
        step3_success = False 
    elif lit_agent: 
        # --- STEP 3: Literature Search & Novelty Assessment ---
        try:
            logger.info(f"\n--- Step 3: Performing Literature Search for {len(selected_claims_for_lit_search)} Selected Claims ---")
            all_literature_search_outcomes = [] 
            for i, claim_obj_from_step1 in enumerate(selected_claims_for_lit_search):
                question_to_ask_owl = claim_obj_from_step1.get("has_anyone_question")
                current_claim_text = claim_obj_from_step1.get("claim", "Unknown claim")
                logger.info(f"Searching literature for claim {i+1}/{len(selected_claims_for_lit_search)}: '{question_to_ask_owl}'")
                if not question_to_ask_owl:
                    logger.warning(f"Skipping claim '{current_claim_text}' as it has no 'has_anyone_question'.")
                    all_literature_search_outcomes.append({"original_claim_object": claim_obj_from_step1, "owl_query": question_to_ask_owl, "status": "skipped", "reason": "No 'has_anyone_question'"})
                    continue
                owl_result_dict = lit_agent.query_literature(question_to_ask_owl)
                current_search_outcome = {"original_claim_object": claim_obj_from_step1, "owl_query": question_to_ask_owl, "owl_response_direct": owl_result_dict}
                all_literature_search_outcomes.append(current_search_outcome)
                print(f"\n--- Literature Search Result for: {current_claim_text} ---")
                if owl_result_dict.get("status") == "success":
                    formatted_answer_for_display = owl_result_dict.get('formatted_answer', 'N/A')
                    print(f"  OWL Formatted Answer: {formatted_answer_for_display}")
                    owl_json_str = owl_result_dict.get("json")
                    if owl_json_str:
                        try:
                            owl_data_from_json_str = json.loads(owl_json_str)
                            answer_field_from_json = owl_data_from_json_str.get('answer', '').lower()
                            if 'no' in answer_field_from_json[:3]: 
                                novel_detail = f"Claim: '{current_claim_text}'. Lit Search: Potentially novel. OWL Answer: '{answer_field_from_json[:100]}...'. Context: '{formatted_answer_for_display[:150]}...'"
                                novel_claims_details_for_dft_context.append(novel_detail)
                                logger.info(f"Claim '{current_claim_text}' marked as potentially novel.")
                                current_search_outcome["novelty_assessment"] = "Potentially Novel"
                            else:
                                logger.info(f"Claim '{current_claim_text}' likely known. OWL Answer: '{answer_field_from_json[:100]}...'")
                                current_search_outcome["novelty_assessment"] = "Likely Known"
                        except json.JSONDecodeError as e_json:
                            logger.error(f"Failed to parse OWL JSON for claim '{current_claim_text}': {e_json}.")
                            current_search_outcome["novelty_assessment"] = "Error parsing OWL JSON"
                    else:
                        logger.warning(f"OWL success for '{current_claim_text}' but 'json' key missing.")
                        current_search_outcome["novelty_assessment"] = "Missing 'json' in OWL success"
                else: 
                    failure_message = owl_result_dict.get('message', owl_result_dict.get('status', 'Unknown error'))
                    print(f"  OWL Search Failed: {failure_message}")
                    current_search_outcome["novelty_assessment"] = f"OWL Search Failed ({failure_message})"
                print("-" * 70)
            lit_results_file = "exp2lit2dft_literature_search_outcomes.json"
            with open(lit_results_file, 'w') as f_lit:
                try: json.dump(all_literature_search_outcomes, f_lit, indent=2)
                except TypeError: 
                    simplified_outcomes = [{"claim_text": o["original_claim_object"].get("claim", "N/A") if isinstance(o.get("original_claim_object"), dict) else "N/A", "owl_query": o["owl_query"], "novelty": o.get("novelty_assessment", "N/A"), "owl_status": o["owl_response_direct"].get("status", "N/A")} for o in all_literature_search_outcomes]
                    json.dump(simplified_outcomes, f_lit, indent=2)
                logging.info(f"Literature search outcomes saved to: {lit_results_file}")
        except Exception as e:
            logger.exception("An unexpected error occurred during Step 3 (Literature Search & Novelty Assessment):")
            logger.warning("Proceeding to DFT recommendations based on initial analysis only due to literature search error.")
            novel_claims_details_for_dft_context = [] 
            step3_success = False
    
    if not novel_claims_details_for_dft_context and step3_success and selected_claims_for_lit_search: 
        logger.info("Literature search completed, but no claims were marked as 'potentially novel' to specifically guide DFT recommendations.")
    elif not step3_success and selected_claims_for_lit_search : 
         logger.info("Literature search was not fully completed or skipped. DFT recommendations will rely on general analysis.")

    # --- STEP 4: Novelty-Informed DFT Recommendation ---
    dft_recommendations = []
    step4_success = False
    try:
        logger.info("\n--- Step 4: Generating DFT Recommendations ---")
        
        # Prepare the novelty context string (this part is mostly the same as your current helper)
        novelty_context_for_agent = None
        if novel_claims_details_for_dft_context:
            novelty_section_text = "The following claims/observations, derived from the initial image analysis, have been identified through literature review as potentially novel or under-explored:\n"
            for detail in novel_claims_details_for_dft_context:
                novelty_section_text += f"- {detail}\n"
            # The prompt for the agent will now be TEXT_ONLY_DFT_RECOMMENDATION_INSTRUCTIONS,
            # which already guides it on how to use this novelty information.
            # So, we just need to pass this section as the 'additional_prompt_context'.
            novelty_context_for_agent = novelty_section_text 
            logger.info("Calling analysis agent with cached analysis and novelty context for DFT recommendations.")
        else:
            logger.info("No specific novelty insights from literature. DFT recommendations will be based on the initial general analysis text.")
            # Provide a generic context if no novel claims. The TEXT_ONLY prompt should handle this.
            novelty_context_for_agent = "No specific novel claims were identified or prioritized from literature search. Please make DFT recommendations based on the most scientifically interesting aspects of the provided initial image analysis."

        dft_recommendations_result = analysis_agent.analyze_microscopy_image_for_structure_recommendations(
            image_path=None, # Crucial: Set to None to trigger text-only path
            system_info=config.SYSTEM_INFO,
            additional_prompt_context=novelty_context_for_agent, # This is the "Special Considerations" string
            cached_detailed_analysis=initial_comprehensive_analysis_text # Pass the cached analysis
        )

        if "error" in dft_recommendations_result:
            logger.error(f"Step 4 Failed: DFT recommendation generation. Error: {dft_recommendations_result.get('details', dft_recommendations_result.get('error'))}")
        else:
            # Use the consistent output key from the agent
            reasoning_or_analysis_text = dft_recommendations_result.get("analysis_summary_or_reasoning", "No analysis text from DFT recommendation step.")
            dft_recommendations = dft_recommendations_result.get("recommendations", [])
            
            print("\n--- Reasoning for DFT Recommendations (Based on Textual Analysis & Novelty) ---")
            print(reasoning_or_analysis_text)
            
            if not dft_recommendations:
                logger.warning("No DFT structure recommendations were generated in Step 4.")
            else:
                logger.info(f"Generated {len(dft_recommendations)} DFT recommendations.")
            
            dft_recs_output_file = "exp2lit2dft_dft_recommendations.json"
            # Save the result which includes the reasoning and recommendations
            full_dft_recs_output = {
                "reasoning_for_recommendations": reasoning_or_analysis_text,
                "recommendations": dft_recommendations
            }
            with open(dft_recs_output_file, 'w') as f_dft_recs:
                json.dump(full_dft_recs_output, f_dft_recs, indent=2)
            logger.info(f"DFT recommendations and reasoning saved to: {dft_recs_output_file}")
            step4_success = True
    except Exception as e:
        logger.exception("An unexpected error occurred during Step 4 (DFT Recommendation):")
    
    if not step4_success:
        logger.error("Exiting due to failure in Step 4 (DFT Recommendation Generation).")
        sys.exit(1)

    # --- STEP 5: DFT Recommendation Selection ---
    selected_dft_recommendation_obj = None 
    if dft_recommendations:
        logger.info("\n--- Step 5: Selection of DFT Structure Recommendation ---")
        selected_dft_recommendation_obj = select_recommendation_interactive(dft_recommendations)
    elif not dft_recommendations and scientific_claims : 
        logger.info("No DFT recommendations were generated in Step 4. Cannot proceed to structure generation.")
    else: 
        logger.info("No initial claims and no DFT recommendations. Workflow ends before structure generation.")

    # --- STEP 6: Structure Generation with Validation Loop ---
    if selected_dft_recommendation_obj:
        logger.info("\n--- Step 6: Structure Generation with Validation ---")
        
        base_dft_request_description = selected_dft_recommendation_obj.get("description")
        additional_instructions_from_config = getattr(config, 'GENERATOR_ADDITIONAL_INSTRUCTIONS', None)
        
        combined_request_for_generator = base_dft_request_description
        if additional_instructions_from_config:
            logger.info(f"Appending additional generator instructions: '{additional_instructions_from_config}'")
            combined_request_for_generator += f". Additional Instructions: {additional_instructions_from_config}"
        
        if not base_dft_request_description:
            logger.error("Selected DFT recommendation is missing 'description'. Cannot generate structure.")
        else:
            current_script_content_for_validation = None
            validator_feedback_for_refinement = None
            final_outcome_achieved = False
            final_generated_structure_file_path = None # Use a distinct name for the full path
            final_generating_script_path = None

            for overall_cycle_num in range(config.MAX_REFINEMENT_CYCLES + 1):
                logger.info(f"--- Generation/Validation Cycle {overall_cycle_num + 1} of {config.MAX_REFINEMENT_CYCLES + 1} ---")
                is_refinement_cycle = overall_cycle_num > 0

                generator_result = structure_generator.generate_script(
                    original_user_request=combined_request_for_generator,
                    attempt_number_overall=overall_cycle_num + 1,
                    is_refinement_from_validation=is_refinement_cycle,
                    previous_script_content=current_script_content_for_validation if is_refinement_cycle else None,
                    validator_feedback=validator_feedback_for_refinement if is_refinement_cycle else None
                )

                print("\n--- [Cycle {}] Structure Generation Script Result ---".format(overall_cycle_num + 1))
                result_to_print = generator_result.copy()
                result_to_print.pop("final_script_content", None)
                result_to_print.pop("last_attempted_script_content", None)
                pprint.pprint(result_to_print)
                print("-" * 70)

                if generator_result.get("status") == "success":
                    generated_structure_file_name_only = generator_result.get("output_file")
                    current_script_content_for_validation = generator_result.get("final_script_content")
                    final_generating_script_path = generator_result.get("final_script_path") 
                    
                    # Correctly determine the path of the generated structure file
                    # It's expected to be in the CWD unless the script specifies otherwise.
                    # The AseExecutor verifies its existence in the CWD.
                    if generated_structure_file_name_only:
                        final_generated_structure_file_path = os.path.abspath(generated_structure_file_name_only)
                        if not os.path.exists(final_generated_structure_file_path):
                             logger.warning(f"Structure file '{generated_structure_file_name_only}' reported by generator not found in CWD '{os.getcwd()}'. Trying in GENERATED_SCRIPT_DIR.")
                             # Fallback to checking in GENERATED_SCRIPT_DIR, though less likely for ASE output
                             potential_path_in_script_dir = os.path.join(config.GENERATED_SCRIPT_DIR, generated_structure_file_name_only)
                             if os.path.exists(potential_path_in_script_dir):
                                 final_generated_structure_file_path = potential_path_in_script_dir
                             else:
                                 logger.error(f"Structure file '{generated_structure_file_name_only}' also not found in '{config.GENERATED_SCRIPT_DIR}'. Cannot validate.")
                                 final_generated_structure_file_path = None # Mark as not found
                    else:
                        final_generated_structure_file_path = None


                    if not final_generated_structure_file_path or not current_script_content_for_validation or not final_generating_script_path:
                        logger.error("Generator reported success but crucial info (output file path, script content, or script path) is missing or file not found. Aborting.")
                        break 

                    logger.info(f"Script executed. Structure file expected at: {final_generated_structure_file_path}")
                    logger.info(f"Generated script path: {final_generating_script_path}")
                    
                    logger.info(f"--- [Cycle {overall_cycle_num + 1}] Validating Structure ---")
                    validator_feedback_for_refinement = structure_validator.validate_structure_and_script(
                        structure_file_path=final_generated_structure_file_path, 
                        generating_script_content=current_script_content_for_validation,
                        original_request=combined_request_for_generator,
                        tool_documentation=structure_generator._select_tool(combined_request_for_generator).docs_content  # ← Add this line 
                    )

                    print("\n--- [Cycle {}] Structure Validation Result ---".format(overall_cycle_num + 1))
                    pprint.pprint(validator_feedback_for_refinement)
                    print("-" * 70)

                    validation_status = validator_feedback_for_refinement.get("status")
                    if validation_status == "success":
                        logger.info(f"Validation successful for structure: {final_generated_structure_file_path}.")
                        print(f"\nSUCCESS: Validated structure generated at {final_generated_structure_file_path}")
                        print(f"Generating script: {final_generating_script_path}")
                        final_outcome_achieved = True
                        break 
                    elif validation_status == "needs_correction":
                        logger.warning(f"Validation found issues: {validator_feedback_for_refinement.get('all_identified_issues')}")
                        
                        # ASK USER FOR DECISION
                        user_decision = ask_user_proceed_or_refine(
                            validation_feedback=validator_feedback_for_refinement,
                            structure_file=final_generated_structure_file_path
                        )
                        
                        if user_decision == 'proceed':
                            logger.info("User chose to proceed with current structure despite validation issues.")
                            print(f"\n✓ PROCEEDING: Using structure at {final_generated_structure_file_path}")
                            print(f"Generating script: {final_generating_script_path}")
                            final_outcome_achieved = True
                            break
                            
                        elif overall_cycle_num < config.MAX_REFINEMENT_CYCLES:
                            logger.info("User chose refinement. Proceeding to next refinement cycle...")
                            # Continue to next cycle
                        else:
                            logger.error("Max refinement cycles reached but user requested refinement.")
                            print(f"\n⚠ WARNING: Max refinement cycles reached. Cannot refine further.")
                            print(f"Final structure: {final_generated_structure_file_path}")
                            print(f"Script: {final_generating_script_path}")
                            
                            # Ask for final decision
                            print("\nFinal options:")
                            print("  [p] PROCEED - Use current structure")
                            final_choice = input("Choice [p]: ").strip().lower()
                            if final_choice in ['p', 'proceed', '']:  # Default to proceed
                                print(f"\n✓ PROCEEDING: Using structure at {final_generated_structure_file_path}")
                                final_outcome_achieved = True
                            else:
                                print(f"\n⏹ WORKFLOW STOPPED")
                                final_outcome_achieved = True
                            break
                    else: 
                        logger.error(f"Validation error: {validator_feedback_for_refinement.get('overall_assessment')}. Aborting.")
                        print(f"\nERROR: Validation process failed. Last structure: {final_generated_structure_file_path}")
                        final_outcome_achieved = True 
                        break
                else: 
                    logger.error(f"Generation failed in Cycle {overall_cycle_num + 1}: {generator_result.get('message')}")
                    final_generating_script_path = generator_result.get("last_attempted_script_path")
                    errors_encountered = generator_result.get("internal_execution_errors_encountered", [])
                    if errors_encountered:
                        print("INFO: Internal execution error(s) during generation attempt:")
                        for i, err in enumerate(errors_encountered):
                            print(f"  Error {i+1}: {err[:500]}" + ("..." if len(err) > 500 else ""))
                    break
            
            print("\n" + "=" * 70 + "\nStep 6 Summary:")
            if final_outcome_achieved and final_generated_structure_file_path: # Check if a file path was determined
                val_status_final = validator_feedback_for_refinement.get("status") if validator_feedback_for_refinement else "N/A"
                summary_status_msg = "SUCCESS" if val_status_final == "success" else f"COMPLETED (Validation: {val_status_final})"
                print(f"Structure Generation {summary_status_msg}")
                print(f"  Final Structure File: '{final_generated_structure_file_path}'")
                print(f"  Final Generating Script: '{final_generating_script_path}'")
                if val_status_final != "success" and validator_feedback_for_refinement:
                    print(f"  Last Validation Assessment: {validator_feedback_for_refinement.get('overall_assessment')}")
            elif final_generating_script_path : 
                 print(f"Structure Generation FAILED.")
                 print(f"  Last Attempted Script: '{final_generating_script_path}'")
            else: 
                print("Structure Generation did not produce a final file.")

    else: 
        logger.info("Skipping Step 6 (Structure Generation & Validation) as no DFT recommendation was selected or available.")

    logger.info("\n--- End of exp2lit2dft Workflow ---")

