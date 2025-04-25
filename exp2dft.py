#!/usr/bin/env python3

import sys
import logging
import pprint

import config


from exp_agents.microscopy_agent import GeminiMicroscopyAnalysisAgent as AnalysisAgent
from sim_agents.ase_agent import StructureGenerator


# --- Interactive Selection Function ---
def select_recommendation_interactive(recommendations: list) -> dict | None:
    """
    Displays recommendations and prompts the user to select one.
    Relies on interactive prompt or default for non-interactive envs.
    """
    if not recommendations:
        print("No recommendations available to select from.")
        return None

    # --- Display Options With Scientific Justification ---
    print("\n--- Available Structure Recommendations ---")
    for i, rec in enumerate(recommendations):
        print(f"\n[{i+1}] (Priority: {rec.get('priority', 'N/A')})")
        print(f"Description: {rec.get('description', 'N/A')}")
        print(f"Scientific justification: {rec.get('scientific_interest', 'N/A')}")
        print("-" * 50)  # Separator between recommendations
    
    # --- Interactive Prompt Loop ---
    while True:
        try:
            if not sys.stdin.isatty():
                 print("\nWarning: Cannot prompt for input in a non-interactive environment.")
                 print("Selecting the highest priority recommendation by default.")
                 return recommendations[0] if recommendations else None

            choice = input(f"Enter the number (1-{len(recommendations)}) to select, or 'q' to quit: ").strip().lower()

            if choice == 'q': print("Selection cancelled."); return None
            if not choice.isdigit(): print("Invalid input. Please enter a number."); continue

            choice_index = int(choice) - 1
            if 0 <= choice_index < len(recommendations):
                selected = recommendations[choice_index]
                print(f"\n--- Selection Confirmed: Recommendation #{choice} ---")
                print(f"Priority: {selected.get('priority', 'N/A')}")
                print(f"Description: {selected.get('description', 'N/A')}")
                print(f"Scientific justification: {selected.get('scientific_interest', 'N/A')}")
                return selected
            else:
                print(f"Invalid number. Please enter between 1 and {len(recommendations)}.")
        except KeyboardInterrupt: print("\nSelection cancelled by user."); return None
        except EOFError: print("\nInput stream closed..."); return recommendations[0] if recommendations else None
        except Exception as e: logging.error(f"An error occurred during selection: {e}"); return None
# --- End Selection Function ---


# ===========================================
#                  MAIN WORKFLOW
# ===========================================
if __name__ == "__main__":

    # --- Use Config for Logging ---
    logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

    selected_recommendation = None

    # === Workflow Step 1: Image Analysis ===
    try:
        logging.info("--- Starting Step 1: Image Analysis ---")
        logging.info(f"Initializing analysis agent with model: {config.ANALYSIS_AGENT_MODEL}")
        analysis_agent = AnalysisAgent(
            api_key=config.GOOGLE_API_KEY,
            model_name=config.ANALYSIS_AGENT_MODEL,
            fft_nmf_settings=config.FFT_NMF_SETTINGS
        )

        logging.info(f"Analyzing image: {config.IMAGE_PATH}...")
        analysis_result = analysis_agent.analyze_microscopy_image_for_structure_recommendations(
            config.IMAGE_PATH, system_info=config.SYSTEM_INFO
        )

        logging.info("--- Analysis Result Received ---")
        if "error" not in analysis_result:
             logging.debug("Full Analysis JSON:\n%s", pprint.pformat(analysis_result))
             print("\n--- Analysis Summary ---")
             print(analysis_result.get("full_analysis", "No detailed analysis text found."))
             print("-" * 22)

             if analysis_result.get("recommendations"):
                  logging.info("Proceeding to recommendation selection.")
                  # --- Updated Function Call (no manual_choice) ---
                  selected_recommendation = select_recommendation_interactive(
                      analysis_result["recommendations"]
                  )
                  # -------------------------------------------------
             else:
                  logging.warning("Analysis completed, but no recommendations were found.")
        else:
             logging.error("Analysis step failed.")
             print("\n--- Analysis Failed ---")
             pprint.pprint(analysis_result)
             print("-" * 21)

    except ValueError as ve:
        logging.error(f"Configuration Error during Analysis step: {ve}")
    except Exception as e:
        logging.exception("An unexpected error occurred during Analysis step:")

    # === Workflow Step 2: Structure Generation ===
    if selected_recommendation:
        try:
            logging.info("--- Starting Step 2: Structure Generation ---")
            selected_description = selected_recommendation.get("description")
            additional_instructions = getattr(config, 'GENERATOR_ADDITIONAL_INSTRUCTIONS', None)

            if not selected_description:
                logging.error("Selected recommendation is missing 'description'. Cannot proceed.")
            else:
                combined_input_description = selected_description
                if additional_instructions:
                    logging.info(f"Adding additional instructions from config: '{additional_instructions}'")
                    # Combine them (simple concatenation with a separator)
                    combined_input_description += f". Additional Instructions: {additional_instructions}"
                else:
                    logging.info("No additional generator instructions found in config.")

                logging.info(f"Using combined input for generation: '{combined_input_description}'")
                # Pass the combined string as the description
                generator_input_data = {"description": combined_input_description}

                logging.info(f"Initializing structure generator: {config.GENERATOR_AGENT_MODEL}")
                generator = StructureGenerator(
                    api_key=config.GOOGLE_API_KEY,
                    model_name=config.GENERATOR_AGENT_MODEL,
                    executor_timeout=config.GENERATOR_SCRIPT_TIMEOUT,
                    generated_script_dir=config.GENERATED_SCRIPT_DIR
                )

                logging.info("Calling structure generator...")
                generator_result = generator.generate(generator_input_data)

                print("\n--- Structure Generation Result ---")
                pprint.pprint(generator_result)
                print("-" * 33)

                gen_status = generator_result.get("status", "unknown")
                if gen_status == "success":
                    logging.info(f"Generation successful. Output: {generator_result.get('output_file')}")
                else:
                    logging.error(f"Generation status: {gen_status}. Message: {generator_result.get('message')}")

        except ValueError as ve:
             logging.error(f"Configuration Error during Generation step: {ve}")
        except Exception as e:
             logging.exception("An unexpected error occurred during Generation step:")

    elif analysis_result and "error" not in analysis_result:
        logging.info("--- Skipping Step 2: Generation (No recommendation selected) ---")
    else:
        logging.info("--- Skipping Step 2: Generation (Analysis step failed) ---")

    logging.info("--- End Full Workflow ---")
    print("\n--- End Full Workflow ---")
