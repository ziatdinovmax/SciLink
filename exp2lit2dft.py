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
from sim_agents.ase_agent import StructureGenerator

# --- Helper Functions ---

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
            if not selection: # Handle empty input
                print("No selection made. Please enter choices, 'all', or 'none'.")
                continue
            if selection == 'none':
                print("No claims selected for literature search.")
                return []
            if selection == 'all':
                print(f"Selected all {len(claims)} claims for literature search.")
                return claims

            indices_str = selection.split(',')
            if not all(s.strip().isdigit() for s in indices_str if s.strip()): # Check for non-numeric inputs
                print("Invalid input. Please use comma-separated numbers only (e.g., '1,3,5').")
                continue

            indices = [int(idx.strip()) - 1 for idx in indices_str if idx.strip()]
            selected_claims_list = []
            
            for idx in indices:
                if 0 <= idx < len(claims):
                    if claims[idx] not in selected_claims_list: # Avoid duplicates
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
             return recommendations[0] if recommendations else None

def generate_additional_context_for_dft_prompt(initial_analysis_text: str, novel_claims_details: list[str]) -> str | None:
    """
    Generates the string to be appended to the base DFT recommendation prompt IF there are novel claims.
    Returns None if no novel claims are provided to detail.
    """
    if not novel_claims_details: # If the list is empty (meaning no true novel claims to pass on)
        return None

    # If we reach here, novel_claims_details contains actual summaries of novel findings
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

# ===========================================
#         MAIN exp2lit2dft WORKFLOW
# ===========================================
if __name__ == "__main__":
    logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)
    logging.info("--- Starting Experiment to Literature to DFT (exp2lit2dft) Workflow ---")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Workflow started on: {current_date}")


    # Initialize agents
    try:
        analysis_agent = GeminiMicroscopyAnalysisAgent(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.ANALYSIS_AGENT_MODEL,
            fft_nmf_settings=config.FFT_NMF_SETTINGS
        )
        # Ensure FutureHouse API key is available for OwlLiteratureAgent
        if not config.FUTUREHOUSE_API_KEY:
            logging.error("FUTUREHOUSE_API_KEY not found in config or environment. Literature search will be skipped.")
            lit_agent = None
        else:
            lit_agent = OwlLiteratureAgent(
                api_key=config.FUTUREHOUSE_API_KEY,
                max_wait_time=config.OWL_MAX_WAIT_TIME
            )
        structure_generator = StructureGenerator(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.GENERATOR_AGENT_MODEL,
            executor_timeout=config.GENERATOR_SCRIPT_TIMEOUT,
            generated_script_dir=config.GENERATED_SCRIPT_DIR
        )
    except ValueError as e:
        logging.error(f"Error initializing agents: {e}. Please check your API keys and configuration.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during agent initialization: {e}")
        sys.exit(1)


    # --- STEP 1: Initial Comprehensive Image Analysis for Claims ---
    initial_comprehensive_analysis_text = "No detailed analysis provided in Step 1."
    scientific_claims = []
    try:
        logging.info("--- Step 1: Initial Image Analysis for Claims & Comprehensive Overview ---")
        initial_analysis_claims_result = analysis_agent.analyze_microscopy_image_for_claims(
            config.IMAGE_PATH,
            system_info=config.SYSTEM_INFO
        )

        if "error" in initial_analysis_claims_result:
            logging.error(f"Step 1 Failed: Initial image analysis. Error: {initial_analysis_claims_result.get('details', initial_analysis_claims_result.get('error'))}")
            sys.exit(1)

        initial_comprehensive_analysis_text = initial_analysis_claims_result.get("full_analysis", initial_comprehensive_analysis_text)
        scientific_claims = initial_analysis_claims_result.get("claims", [])
        
        claims_output_file = "exp2lit2dft_initial_claims.json"
        with open(claims_output_file, 'w') as f:
            json.dump(initial_analysis_claims_result, f, indent=2)
        logging.info(f"Initial analysis and claims saved to: {claims_output_file}")

        if not scientific_claims:
            logging.warning("No scientific claims generated in Step 1. Subsequent DFT recommendations will be general.")
        else:
            logging.info(f"Generated {len(scientific_claims)} claims from initial analysis.")
            print("\n--- Initial Analysis Summary (from Step 1) ---")
            print(initial_comprehensive_analysis_text)
    except Exception as e:
        logging.exception("An unexpected error occurred during Step 1 (Initial Image Analysis):")
        sys.exit(1)


    # --- STEP 2: Claim Selection for Literature Search ---
    selected_claims_for_lit_search = []
    if scientific_claims and lit_agent: # Only proceed if claims were generated and lit_agent is available
        logging.info("\n--- Step 2: Claim Selection for Literature Search ---")
        selected_claims_for_lit_search = select_claims_interactive(scientific_claims)
    elif not lit_agent:
        logging.info("Skipping Step 2 & 3 (Literature Search) as FutureHouse API key is not configured.")
    else: # No scientific claims
        logging.info("Skipping Step 2 (Claim Selection) as no claims were generated in Step 1.")

    novel_claims_details_for_dft_context = [] 

    if not selected_claims_for_lit_search: # This handles cases where claims were empty, selection was 'none', or lit_agent missing
        logging.info("No claims selected or available for literature search. DFT recommendations will rely solely on initial general analysis.")
    else:
        # --- STEP 3: Literature Search & Novelty Assessment (Using User's Established Logic) ---
        try:
            logging.info(f"\n--- Step 3: Performing Literature Search for {len(selected_claims_for_lit_search)} Selected Claims ---")
            all_literature_search_outcomes = [] 

            for i, claim_obj_from_step1 in enumerate(selected_claims_for_lit_search):
                question_to_ask_owl = claim_obj_from_step1.get("has_anyone_question")
                current_claim_text = claim_obj_from_step1.get("claim", "Unknown claim")
                logging.info(f"Searching literature for claim {i+1}/{len(selected_claims_for_lit_search)}: '{question_to_ask_owl}'")

                if not question_to_ask_owl:
                    logging.warning(f"Skipping claim '{current_claim_text}' as it has no 'has_anyone_question'.")
                    all_literature_search_outcomes.append({
                        "original_claim_object": claim_obj_from_step1, # Store the original claim object
                        "owl_query": question_to_ask_owl,
                        "status": "skipped",
                        "reason": "No 'has_anyone_question' provided in claim object."
                    })
                    continue
                
                owl_result_dict = lit_agent.query_literature(question_to_ask_owl)
                current_search_outcome = {
                    "original_claim_object": claim_obj_from_step1,
                    "owl_query": question_to_ask_owl,
                    "owl_response_direct": owl_result_dict 
                }
                all_literature_search_outcomes.append(current_search_outcome)

                print(f"\n--- Literature Search Result for: {current_claim_text} ---")
                if owl_result_dict.get("status") == "success":
                    formatted_answer_for_display = owl_result_dict.get('formatted_answer', 'N/A')
                    print(f"  OWL Formatted Answer: {formatted_answer_for_display}")

                    owl_json_str = owl_result_dict.get("json") # As per OwlLiteratureAgent output
                    if owl_json_str:
                        try:
                            owl_data_from_json_str = json.loads(owl_json_str)
                            # Ensure 'answer' key exists, default to empty string if not
                            answer_field_from_json = owl_data_from_json_str.get('answer', '').lower()

                            if 'no' in answer_field_from_json[:3]: 
                                novel_detail_for_next_prompt = (
                                    f"Claim: '{current_claim_text}'. Literature Search Suggestion: Potentially novel. "
                                    f"OWL's direct answer started with 'no' (e.g., '{answer_field_from_json[:100]}...'). "
                                    f"Context from formatted answer: '{formatted_answer_for_display[:150]}...'"
                                )
                                novel_claims_details_for_dft_context.append(novel_detail_for_next_prompt)
                                logging.info(f"Claim '{current_claim_text}' marked as potentially novel based on OWL answer starting with 'no'.")
                                current_search_outcome["novelty_assessment"] = "Potentially Novel (Answer started with 'no')"
                            else:
                                logging.info(f"Claim '{current_claim_text}' appears to be known or OWL answer did not start with 'no'. Answer: '{answer_field_from_json[:100]}...'")
                                current_search_outcome["novelty_assessment"] = "Likely Known (Answer did not start with 'no' or was affirmative)"
                        except json.JSONDecodeError as e_json:
                            logging.error(f"Failed to parse 'json' string from OWL response for claim '{current_claim_text}': {e_json}. Raw JSON string: {owl_json_str[:500]}...")
                            current_search_outcome["novelty_assessment"] = "Error parsing OWL JSON response"
                        # Removed KeyError catch as .get() handles missing 'answer' key
                    else:
                        logging.warning(f"OWL response for claim '{current_claim_text}' was successful but the 'json' key was missing. Cannot assess novelty using the standard method.")
                        current_search_outcome["novelty_assessment"] = "Missing 'json' key in successful OWL response"
                else: 
                    failure_message = owl_result_dict.get('message', owl_result_dict.get('status', 'Unknown error'))
                    print(f"  OWL Search Failed or other status: {failure_message}")
                    current_search_outcome["novelty_assessment"] = f"OWL Search Failed ({failure_message})"
                print("-" * 70)

            lit_results_file = "exp2lit2dft_literature_search_outcomes.json"
            with open(lit_results_file, 'w') as f_lit:
                # Attempt to dump. If original_claim_object is not serializable, this will fail.
                # For now, assuming it's a dict from JSON and serializable.
                try:
                    json.dump(all_literature_search_outcomes, f_lit, indent=2)
                except TypeError as te:
                    logging.error(f"Could not serialize all_literature_search_outcomes to JSON: {te}")
                    # Fallback: try to serialize a simpler version
                    simplified_outcomes = []
                    for outcome in all_literature_search_outcomes:
                        simplified_outcome = {
                            "claim_text": outcome["original_claim_object"].get("claim", "N/A") if isinstance(outcome.get("original_claim_object"), dict) else "N/A",
                            "owl_query": outcome["owl_query"],
                            "novelty_assessment": outcome.get("novelty_assessment", "N/A"),
                            "owl_status": outcome["owl_response_direct"].get("status", "N/A")
                        }
                        simplified_outcomes.append(simplified_outcome)
                    json.dump(simplified_outcomes, f_lit, indent=2)
                    logging.info(f"Simplified literature search outcomes saved to: {lit_results_file}")
                else:
                    logging.info(f"Full literature search outcomes saved to: {lit_results_file}")


        except Exception as e:
            logging.exception("An unexpected error occurred during Step 3 (Literature Search & Novelty Assessment):")
            logging.warning("Proceeding to DFT recommendations based on initial analysis only due to literature search error.")
            novel_claims_details_for_dft_context = [] # Clear any partially gathered novel claims

    # If after all that, novel_claims_details_for_dft_context is still empty,
    # generate_additional_context_for_dft_prompt will return None.
    if not novel_claims_details_for_dft_context:
        logging.info("No specific novel claims identified to focus DFT recommendations. Will use general analysis.")


    # --- STEP 4: Novelty-Informed DFT Recommendation ---
    dft_recommendations = []
    try:
        logging.info("\n--- Step 4: Generating DFT Recommendations ---")
        additional_context_string = generate_additional_context_for_dft_prompt(
            initial_comprehensive_analysis_text,
            novel_claims_details_for_dft_context # This list is empty if no novel claims were found/processed
        )

        if additional_context_string:
            logging.info("Calling analysis agent with appended novelty context for DFT recommendations.")
        else:
            logging.info("Calling analysis agent with base instructions for general DFT recommendations (no specific novelty context).")
        
        # This call assumes `analyze_microscopy_image_for_structure_recommendations`
        # in `GeminiMicroscopyAnalysisAgent` has been modified to accept `additional_prompt_context`
        dft_recommendations_result = analysis_agent.analyze_microscopy_image_for_structure_recommendations(
            config.IMAGE_PATH,
            system_info=config.SYSTEM_INFO,
            additional_prompt_context=additional_context_string 
        )

        if "error" in dft_recommendations_result:
            logging.error(f"Step 4 Failed: DFT recommendation generation. Error: {dft_recommendations_result.get('details', dft_recommendations_result.get('error'))}")
            sys.exit(1)
        
        final_analysis_text_for_dft = dft_recommendations_result.get("full_analysis", "No analysis text from DFT recommendation step.")
        dft_recommendations = dft_recommendations_result.get("recommendations", [])

        print("\n--- Analysis Summary for DFT Recommendations ---")
        print(final_analysis_text_for_dft) # This text is now potentially novelty-informed

        if not dft_recommendations:
            logging.error("No DFT structure recommendations were generated in Step 4.")
            # Allow workflow to continue to see if user wants to manually specify something, or just end.
        else:
            logging.info(f"Generated {len(dft_recommendations)} DFT recommendations.")

        dft_recs_output_file = "exp2lit2dft_dft_recommendations.json"
        with open(dft_recs_output_file, 'w') as f_dft_recs:
            json.dump(dft_recommendations_result, f_dft_recs, indent=2)
        logging.info(f"DFT recommendations (potentially novelty-informed) saved to: {dft_recs_output_file}")

    except Exception as e:
        logging.exception("An unexpected error occurred during Step 4 (DFT Recommendation):")
        sys.exit(1)


    # --- STEP 5: DFT Recommendation Selection ---
    selected_dft_recommendation = None
    if dft_recommendations: # Only ask for selection if recommendations were made
        logging.info("\n--- Step 5: Selection of DFT Structure Recommendation ---")
        selected_dft_recommendation = select_recommendation_interactive(dft_recommendations)
    elif not dft_recommendations and scientific_claims : # If no DFT recs but there was an initial analysis
        logging.info("No DFT recommendations were generated in Step 4. Cannot proceed to structure generation.")
    else: # No claims, no recs
        logging.info("No initial claims and no DFT recommendations. Workflow ends before structure generation.")


    # --- STEP 6: Structure Generation ---
    if selected_dft_recommendation:
        try:
            logging.info("\n--- Step 6: Structure Generation ---")
            description_for_generator = selected_dft_recommendation.get("description")
            additional_instructions_from_config = getattr(config, 'GENERATOR_ADDITIONAL_INSTRUCTIONS', None)
            
            final_description_for_ase_agent = description_for_generator
            if additional_instructions_from_config:
                final_description_for_ase_agent = f"{description_for_generator}. Additional Instructions: {additional_instructions_from_config}"

            logging.info(f"Requesting structure generation for: '{final_description_for_ase_agent}'")
            generator_input_data = {"description": final_description_for_ase_agent}
            generator_result = structure_generator.generate(generator_input_data)

            print("\n--- Structure Generation Result ---")
            pprint.pprint(generator_result)
            print("-" * 70)

            if generator_result.get("status") == "success":
                logging.info(f"Structure generation successful. Output file: {generator_result.get('output_file')}")
                logging.info(f"Final ASE script saved to: {generator_result.get('final_script_path')}")
            else:
                logging.error(f"Structure generation failed. Last error: {generator_result.get('message')}")
        except Exception as e:
            logging.exception("An unexpected error occurred during Step 6 (Structure Generation):")
    else:
        logging.info("Skipping Step 6 (Structure Generation) as no DFT recommendation was selected or available.")

    logging.info("\n--- End of exp2lit2dft Workflow ---")