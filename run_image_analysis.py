# run_analysis.py

import os
import json
import sys
import pathlib

# --- Import the agent from the package ---
from exp_agents.microscopy_agent import GeminiMicroscopyAnalysisAgent
# -----------------------------------------

# --- Interactive Selection Function ---
# (Keep the select_recommendation_interactive function definition here,
#  as it's part of this script's interactive workflow)
def select_recommendation_interactive(recommendations: list, manual_choice: int | None = None) -> dict | None:
    if not recommendations:
        print("No recommendations available to select from.")
        return None
    # (Keep implementation including manual_choice check, isatty check, input loop)
    # --- Check for Manual Override ---
    if manual_choice is not None:
        try:
            choice_index = int(manual_choice) - 1
            if 0 <= choice_index < len(recommendations):
                selected = recommendations[choice_index]
                print(f"--- Manually Selected Recommendation #{manual_choice} ---")
                print(f"  Description: {selected.get('description')}")
                print("-------------------------------------------\n")
                return selected
            else:
                print(f"Warning: Manual choice ({manual_choice}) is out of range (1-{len(recommendations)}). Falling back.")
        except ValueError:
            print(f"Warning: Manual choice '{manual_choice}' is not a valid number. Falling back.")

    print("\n--- Available Structure Recommendations ---")
    for i, rec in enumerate(recommendations):
         print(f"[{i+1}] Priority: {rec.get('priority', 'N/A')}")
         print(f"    Description: {rec.get('description', 'N/A')}\n")

    while True:
        try:
            if not sys.stdin.isatty():
                 print("Warning: Cannot prompt for input in a non-interactive environment.")
                 print("Selecting the highest priority recommendation by default.")
                 # Ensure recommendations list isn't empty before indexing
                 return recommendations[0] if recommendations else None

            choice = input(f"Enter the number (1-{len(recommendations)}) of the recommendation to select, or 'q' to quit: ").strip().lower()

            if choice == 'q': print("Selection cancelled."); return None
            if not choice.isdigit(): print("Invalid input. Please enter a number."); continue

            choice_index = int(choice) - 1
            if 0 <= choice_index < len(recommendations):
                selected = recommendations[choice_index]
                print(f"\n--- You selected recommendation #{choice}: ---")
                print(f"  Description: {selected.get('description')}")
                print("-------------------------------------------\n")
                return selected
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(recommendations)}.")
        except KeyboardInterrupt: print("\nSelection cancelled by user."); return None
        except EOFError: print("\nInput stream closed..."); return recommendations[0] if recommendations else None
        except Exception as e: print(f"An error occurred: {e}"); return None
# --- End Selection Function ---


if __name__ == "__main__":
    # --- Configuration ---
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # <<< Set this manually to choose, or leave as None for default/interactive mode >>>
    MANUAL_SELECTION_NUMBER = None # e.g., set to 1

    IMAGE_PATH = "MoS2_1.png" # <<< SET THIS PATH
    SYSTEM_INFO = {
        "material_type": "MoS2 monolayer",
        "synthesis_details": "V-doped MoS2 films were synthesized on c-plane sapphire substrates using a custom-built MOCVD system. Mo(CO)6 (99.99% purity, Sigma-Aldrich) and V(C5H5)2 (99.99% purity, Sigma-Aldrich) powders are placed into stainless steel bubblers and serve as the Mo and V precursors, respectively. During synthesis, the pressure inside the bubblers are kept constant at 735 torr, and the temperature is set to 24°C for Mo(CO)6 and 50°C for V(C5H5)2 to maintain a constant precursor vapor pressure. Concentration of Mo and V precursor in the growth chamber can be tightly controlled by introducing precise flow of H2 carrier gas through the bubblers. A high-purity H2S gas lecture bottle (99.5%, Sigma-Aldrich) is used as the sulfur source. Growths are carried out at a growth temperature of 1000°C and a pressure of 50 torr, using a multistep growth process",
        "microscopy_type": "HAADF-STEM",
        "image_path": "MoS2_1.png",
        "experimental_details": {
            "voltage": "60 kV",
            "probe_current": "30 pA"
        }
    }
    MODEL = "gemini-2.5-pro-exp-03-25" # Or your preferred model
    # -----------------------

    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        exit(1)

    if not pathlib.Path(IMAGE_PATH).is_file():
         print(f"Error: Image file not found at '{IMAGE_PATH}'")
         exit(1)

    # --- Workflow Execution ---
    try:
        print(f"Initializing agent with model: {MODEL}")
        # Instantiate the imported agent
        agent = GeminiMicroscopyAnalysisAgent(api_key=GOOGLE_API_KEY, model_name=MODEL)

        print(f"Analyzing image: {IMAGE_PATH}...")
        analysis_result = agent.analyze_microscopy_image(IMAGE_PATH, system_info=SYSTEM_INFO)

        print("\n--- Analysis Result ---")
        if "error" not in analysis_result:
             print(json.dumps(analysis_result, indent=2))

             # --- SELECTION STEP ---
             if analysis_result.get("recommendations"):
                  selected_recommendation = select_recommendation_interactive(
                      analysis_result["recommendations"],
                      manual_choice=MANUAL_SELECTION_NUMBER # Pass the manual choice
                  )

                  if selected_recommendation:
                       print("--- Ready for Next Step ---")
                       print("Selected Recommendation Data:")
                       print(json.dumps(selected_recommendation, indent=2))
                       print("\nThis 'selected_recommendation' dictionary can now be passed to the next LLM or function.")
                       # (Add call to next step here)
                  else:
                       print("No recommendation was selected for the next step.")
             else:
                  print("Analysis completed, but no recommendations were found to select from.")

        else:
             print("Analysis failed. Result:")
             print(json.dumps(analysis_result, indent=2))

        print("--- End Script ---")

    except ValueError as ve: print(f"Configuration Error: {ve}")
    except Exception as e: print(f"An unexpected error occurred during script execution: {e}")