from typing import Dict, Any, Optional
import sys

def display_plan_summary(result: Dict[str, Any], quiet_mode: bool = False) -> None:
    """
    Parses the agent's results and prints a structured, pretty-printed 
    summary to the console for human review.
    
    Args:
        result: The agent's result dictionary
        quiet_mode: If True, suppresses output (useful for MCP mode)
    """
    if quiet_mode:
        return
        
    # 1. Error Handling
    if result.get("error"):
        print(f"\n❌ Agent finished with an error: {result['error']}\n")
        return

    # 2. Structure Validation
    experiments = result.get("proposed_experiments")
    if not experiments or not isinstance(experiments, list):
        print("\n⚠️  The agent returned a result, but no experiments were found.")
        return

    # 3. Header
    print("\n" + "="*80)
    print("✅ PROPOSED EXPERIMENTAL PLAN")
    print("="*80)

    # 4. Loop through Experiments
    for i, exp in enumerate(experiments, 1):
        
        # --- Name & Hypothesis ---
        print(f"\n🔬 EXPERIMENT {i}: {exp.get('experiment_name', 'Unnamed Experiment')}")
        print("-" * 80)
        print(f"\n> 🎯 Hypothesis:\n> {exp.get('hypothesis', 'N/A')}")

        # --- Experimental Steps (Numbered) ---
        print("\n--- 🧪 Experimental Steps ---")
        steps = exp.get('experimental_steps', [])
        if steps:
            for j, step in enumerate(steps, 1):
                print(f" {j}. {step}")
        else:
            print("  (No steps provided)")
        
        # --- Equipment ---
        print("\n--- 🛠️  Required Equipment ---")
        equipment = exp.get('required_equipment', [])
        if equipment:
            if len(equipment) > 5:
                for item in equipment: print(f"  * {item}")
            else:
                print(f"  {', '.join(equipment)}")
        else:
            print("  (No equipment specified)")

        # --- Outcome & Justification ---
        print("\n--- 📈 Expected Outcome ---")
        print(f"  {exp.get('expected_outcome', 'N/A')}")

        print("\n--- 💡 Justification ---")
        print(f"  {exp.get('justification', 'N/A')}")
        
        # --- Source Documents ---
        print("\n--- 📄 Source Documents ---")
        sources = exp.get('source_documents', [])
        if sources:
            for src in sources:
                print(f"  - {src}")
        else:
            print("  (No sources listed)")

        # --- Code Indicator ---
        if "implementation_code" in exp:
            print("\n--- 💻 Implementation Code ---")
            print("  ✅ Python script generated (saved to file).")

    print("\n" + "="*80)


def get_user_feedback(enable_interactive: bool = True) -> Optional[str]:
    """
    Pauses execution to get user input via the CLI. 
    Returns None if the user just presses ENTER (indicating approval)
    or if running in non-interactive mode.
    
    Args:
        enable_interactive: If False, returns None immediately (auto-approve).
                          This is used for MCP mode where no TTY is available.
    
    Returns:
        None if approved or non-interactive, otherwise the feedback string.
    """
    # MCP/Non-Interactive Mode: Auto-approve
    if not enable_interactive:
        return None
    
    # Check if we actually have a TTY (terminal)
    if not sys.stdin.isatty():
        return None
    
    # Interactive Mode: Prompt user
    print("\n" + "-"*60)
    print("👤 HUMAN FEEDBACK STEP")
    print("-" * 60)
    print("Review the plan above.")
    print("• To APPROVE: Press [ENTER] directly.")
    print("• To REQUEST CHANGES: Type your feedback/instructions and press [ENTER].")
    
    try:
        feedback = input("\n> Instruction: ").strip()
    except (EOFError, OSError):
        return None
    
    if not feedback:
        return None
        
    return feedback