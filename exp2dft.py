#!/usr/bin/env python3

import sys
import logging
import pprint

import config 

from exp_agents.microscopy_agent import GeminiMicroscopyAnalysisAgent as AnalysisAgent
from sim_agents.ase_agent import StructureGenerator
from sim_agents.val_agent import StructureValidatorAgent


# --- Interactive Selection Function  ---
def select_recommendation_interactive(recommendations: list) -> dict | None:
    """
    Displays recommendations and prompts the user to select one.
    Relies on interactive prompt or default for non-interactive envs.
    """
    if not recommendations:
        print("No recommendations available to select from.")
        return None

    print("\n--- Available Structure Recommendations ---")
    for i, rec in enumerate(recommendations):
        print(f"\n[{i+1}] (Priority: {rec.get('priority', 'N/A')})")
        print(f"Description: {rec.get('description', 'N/A')}")
        print(f"Scientific justification: {rec.get('scientific_interest', 'N/A')}")
        print("-" * 50)
    
    while True:
        try:
            if not sys.stdin.isatty(): # Non-interactive environment
                logging.warning("Non-interactive environment detected. Selecting the highest priority recommendation by default.")
                print("\nWarning: Non-interactive environment. Selecting highest priority recommendation.")
                return recommendations[0] if recommendations else None
            
            choice = input(f"Enter the number (1-{len(recommendations)}) to select, or 'q' to quit: ").strip().lower()

            if choice == 'q':
                print("Selection cancelled by user.")
                return None
            if not choice.isdigit():
                print("Invalid input. Please enter a number.")
                continue

            choice_index = int(choice) - 1
            if 0 <= choice_index < len(recommendations):
                selected = recommendations[choice_index]
                print(f"\n--- Selection Confirmed: Recommendation #{choice_index + 1} ---")
                pprint.pprint(selected) # Print the selected recommendation for clarity
                return selected
            else:
                print(f"Invalid number. Please enter between 1 and {len(recommendations)}.")
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            return None
        except EOFError: # For non-interactive environments if input stream closes
            logging.warning("Input stream closed in non-interactive environment. Selecting highest priority recommendation.")
            print("\nInput stream closed. Selecting highest priority recommendation.")
            return recommendations[0] if recommendations else None
        except Exception as e:
            logging.error(f"An error occurred during selection: {e}")
            return None
# --- End Selection Function ---


# ===========================================
#                 MAIN WORKFLOW
# ===========================================
if __name__ == "__main__":
    # --- Use Config for Logging ---
    logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)
    logger = logging.getLogger(__name__) # Get a logger for this script

    selected_recommendation_obj = None # Changed variable name for clarity
    analysis_step_succeeded = False # Flag to track if analysis step was successful

    # === Workflow Step 1: Image Analysis ===
    try:
        logger.info("--- Starting Step 1: Image Analysis ---")
        logger.info(f"Initializing analysis agent with model: {config.ANALYSIS_AGENT_MODEL}")
        analysis_agent = AnalysisAgent(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.ANALYSIS_AGENT_MODEL,
            fft_nmf_settings=config.FFT_NMF_SETTINGS
        )

        logger.info(f"Analyzing image: {config.IMAGE_PATH} with metadata from {config.SYSTEM_INFO_PATH}...")
        analysis_result = analysis_agent.analyze_microscopy_image_for_structure_recommendations(
            config.IMAGE_PATH, system_info=config.SYSTEM_INFO
        )

        logger.info("--- Analysis Result Received ---")
        if "error" not in analysis_result:
            logger.debug("Full Analysis JSON:\n%s", pprint.pformat(analysis_result))
            print("\n--- Analysis Summary ---")
            print(analysis_result.get("full_analysis", "No detailed analysis text found."))
            print("-" * 22)

            if analysis_result.get("recommendations"):
                logger.info("Proceeding to recommendation selection.")
                selected_recommendation_obj = select_recommendation_interactive(
                    analysis_result["recommendations"]
                )
                if selected_recommendation_obj:
                    analysis_step_succeeded = True # Mark analysis as successful if a recommendation is made and selected
            else:
                logger.warning("Analysis completed, but no structure recommendations were found.")
        else:
            logger.error(f"Analysis step failed: {analysis_result.get('error')} - {analysis_result.get('details')}")
            print("\n--- Analysis Failed ---")
            pprint.pprint(analysis_result)
            print("-" * 21)

    except ValueError as ve: # Catch configuration-related errors during Analysis step
        logger.error(f"Configuration Error during Analysis step: {ve}")
    except Exception as e:
        logger.exception("An unexpected error occurred during Image Analysis step:")

    # === Workflow Step 2 & 3: Structure Generation & Validation Loop ===
    if analysis_step_succeeded and selected_recommendation_obj:
        original_user_request = selected_recommendation_obj.get("description")
        # Additional instructions for the generator can be combined here if needed
        
        logger.info(f"Proceeding with structure generation for: '{original_user_request}'")

        if not original_user_request:
            logger.error("Selected recommendation is missing 'description'. Cannot proceed with generation.")
        else:
            # Initialize Agents for Generation and Validation
            try:
                structure_generator = StructureGenerator(
                    api_key=config.GOOGLE_API_KEY,
                    model_name=config.GENERATOR_AGENT_MODEL,
                    executor_timeout=config.GENERATOR_SCRIPT_TIMEOUT,
                    generated_script_dir=config.GENERATED_SCRIPT_DIR
                )
                structure_validator = StructureValidatorAgent(
                    api_key=config.GOOGLE_API_KEY,
                    model_name=config.VALIDATOR_AGENT_MODEL
                )
                logger.info("StructureGenerator and StructureValidatorAgent initialized.")
            except ValueError as ve:
                logger.error(f"Failed to initialize generation/validation agents (likely API key issue): {ve}")
                # End workflow if agents can't be initialized
                selected_recommendation_obj = None # Prevent further steps
            except Exception as e:
                logger.exception("Unexpected error initializing generation/validation agents.")
                selected_recommendation_obj = None # Prevent further steps


            if selected_recommendation_obj: # Check again in case agent init failed
                current_script_content_for_validation = None
                validator_feedback_for_refinement = None
                final_outcome_achieved = False
                final_generated_structure_file = None
                final_generating_script_path = None

                # Loop for initial generation + refinement attempts
                for overall_cycle_num in range(config.MAX_REFINEMENT_CYCLES + 1):
                    logger.info(f"--- Starting Generation/Validation Cycle {overall_cycle_num + 1} of {config.MAX_REFINEMENT_CYCLES + 1} ---")
                    
                    is_refinement_cycle = overall_cycle_num > 0

                    generator_result = structure_generator.generate_script(
                        original_user_request=original_user_request,
                        attempt_number_overall=overall_cycle_num + 1,
                        is_refinement_from_validation=is_refinement_cycle,
                        previous_script_content=current_script_content_for_validation if is_refinement_cycle else None,
                        validator_feedback=validator_feedback_for_refinement if is_refinement_cycle else None
                    )

                    print("\n--- Structure Generation Script Attempt Result ---")
                    pprint.pprint(generator_result)
                    print("-" * 33)

                    if generator_result.get("status") == "success":
                        generated_structure_file = generator_result.get("output_file")
                        current_script_content_for_validation = generator_result.get("final_script_content")
                        # Storing the latest successful paths/files
                        final_generated_structure_file = generated_structure_file 
                        final_generating_script_path = generator_result.get("final_script_path")


                        if not generated_structure_file or not current_script_content_for_validation or not final_generating_script_path:
                            logger.error("Generator reported success but crucial information (output file, script content, or path) is missing. Aborting cycle.")
                            break 

                        logger.info(f"Script generated and executed. Structure file: {generated_structure_file}")
                        logger.info(f"Generated script path: {final_generating_script_path}")
                        
                        logger.info("--- Proceeding to Structure Validation ---")
                        validator_feedback_for_refinement = structure_validator.validate_structure_and_script(
                            structure_file_path=generated_structure_file,
                            generating_script_content=current_script_content_for_validation,
                            original_request=original_user_request
                        )

                        print("\n--- Structure Validation Result ---")
                        pprint.pprint(validator_feedback_for_refinement)
                        print("-" * 33)

                        validation_status = validator_feedback_for_refinement.get("status")
                        if validation_status == "success":
                            logger.info(f"Validation successful for structure: {generated_structure_file}. Workflow complete.")
                            print(f"\nSUCCESS: Validated structure generated at {generated_structure_file}")
                            print(f"Generating script: {final_generating_script_path}")
                            final_outcome_achieved = True
                            break 
                        elif validation_status == "needs_correction":
                            logger.warning(f"Validation found issues requiring script correction: {validator_feedback_for_refinement.get('all_identified_issues')}")
                            if overall_cycle_num < config.MAX_REFINEMENT_CYCLES:
                                logger.info("Preparing for next refinement cycle based on validation feedback...")
                            else:
                                logger.error("Maximum refinement cycles reached. Structure may still have issues.")
                                print(f"\nWARNING: Max refinement cycles reached. Structure at {generated_structure_file} (from script {final_generating_script_path}) may have unresolved validation issues.")
                                final_outcome_achieved = True 
                                break 
                        else: 
                            logger.error(f"Structure validation process encountered an error: {validator_feedback_for_refinement.get('overall_assessment')}. Aborting refinement.")
                            print(f"\nERROR: Validation process failed. Last structure at {generated_structure_file} (from script {final_generating_script_path}).")
                            final_outcome_achieved = True 
                            break
                    else: 
                        logger.error(f"Structure script generation failed in cycle {overall_cycle_num + 1}. Error: {generator_result.get('message')}")
                        print(f"\nERROR: Script generation failed. Last attempted script (if any): {generator_result.get('last_attempted_script_path')}")
                        final_generating_script_path = generator_result.get("last_attempted_script_path") # Store last attempted script
                        break 

                # After the loop for generation/validation cycles
                if final_outcome_achieved:
                    if validator_feedback_for_refinement and validator_feedback_for_refinement.get("status") == "success":
                        logger.info(f"Workflow successfully completed with a validated structure: {final_generated_structure_file}")
                    else: # Handled cases where loop broke due to max cycles or validation error
                        logger.warning(f"Workflow completed. Final structure (if any): {final_generated_structure_file}, Script (if any): {final_generating_script_path}")
                else: # Loop completed all cycles without a "success" or forced break with outcome_achieved=True
                    logger.error("--- Workflow ended without a successfully validated structure after all cycles. ---")
                    print("\nFAILURE: Could not produce a validated structure after all attempts.")
                    if final_generating_script_path:
                        print(f"Last attempted script was: {final_generating_script_path}")


    elif not analysis_step_succeeded:
        logger.info("--- Skipping Structure Generation & Validation (Analysis step failed or no recommendations) ---")
    elif not selected_recommendation_obj: # This case means analysis might have succeeded but user chose 'q' or no recommendations
        logger.info("--- Skipping Structure Generation & Validation (No recommendation was selected by the user or available) ---")


    logger.info("--- End Full DFT Workflow ---")
    print("\n--- End Full DFT Workflow ---")
