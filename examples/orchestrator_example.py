import os
import json
from scilink.agents.exp_agents import OrchestratorAgent, AGENT_MAP

def run_orchestrator_example():
    """
    A minimalistic example to demonstrate the OrchestratorAgent's functionality.
    """
    # Initialize the orchestrator
    # This will automatically use your configured GOOGLE_API_KEY
    try:
        orchestrator = OrchestratorAgent()
        print("OrchestratorAgent initialized successfully.")
    except Exception as e:
        print(f"Error initializing OrchestratorAgent: {e}")
        print("Please ensure your GOOGLE_API_KEY is configured correctly.")
        return

    # --- Example 1: High-Resolution Atomic Microscopy ---
    print("\n--- Example 1: Selecting an agent for atomic resolution analysis ---")
    data_type_1 = "microscopy"
    system_info_1 = {
        "material": "MoS2 monolayer",
        "technique": "STEM",
        "analysis_goal": "Find and classify atomic defects in the lattice."
    }
    print(f"Request -> data_type: '{data_type_1}', system_info: {json.dumps(system_info_1)}")
    
    selected_agent_id_1 = orchestrator.select_agent(data_type_1, system_info_1)
    
    if selected_agent_id_1 in AGENT_MAP:
        selected_agent_class_1 = AGENT_MAP[selected_agent_id_1]
        print(f"  -> Orchestrator selected agent ID: {selected_agent_id_1}")
        print(f"  -> Corresponding Agent Class: {selected_agent_class_1.__name__}")
    else:
        print("  -> Orchestrator failed to select an agent.")


    # --- Example 2: Particle Segmentation ---
    print("\n--- Example 2: Selecting an agent for particle analysis ---")
    data_type_2 = "microscopy"
    system_info_2 = {
        "material": "Au nanoparticles on Carbon",
        "technique": "TEM",
        "analysis_goal": "Analyze the size distribution and morphology of the nanoparticles."
    }
    print(f"Request -> data_type: '{data_type_2}', system_info: {json.dumps(system_info_2)}")

    selected_agent_id_2 = orchestrator.select_agent(data_type_2, system_info_2)

    if selected_agent_id_2 in AGENT_MAP:
        selected_agent_class_2 = AGENT_MAP[selected_agent_id_2]
        print(f"  -> Orchestrator selected agent ID: {selected_agent_id_2}")
        print(f"  -> Corresponding Agent Class: {selected_agent_class_2.__name__}")
    else:
        print("  -> Orchestrator failed to select an agent.")


    # --- Example 3: General Spectroscopy ---
    print("\n--- Example 3: Selecting an agent for spectroscopy ---")
    data_type_3 = "spectroscopy"
    system_info_3 = {
        "material": "Graphene Oxide",
        "technique": "EELS",
        "analysis_goal": "Identify the primary chemical phases."
    }
    print(f"Request -> data_type: '{data_type_3}', system_info: {json.dumps(system_info_3)}")

    selected_agent_id_3 = orchestrator.select_agent(data_type_3, system_info_3)

    if selected_agent_id_3 in AGENT_MAP:
        selected_agent_class_3 = AGENT_MAP[selected_agent_id_3]
        print(f"  -> Orchestrator selected agent ID: {selected_agent_id_3}")
        print(f"  -> Corresponding Agent Class: {selected_agent_class_3.__name__}")
    else:
        print("  -> Orchestrator failed to select an agent.")

    # Note: The OrchestratorAgent requires the 'system_info' parameter to make
    # an informed decision. Providing a clear 'analysis_goal' helps it select
    # the most appropriate specialized agent for the task.

if __name__ == '__main__':
    run_orchestrator_example()