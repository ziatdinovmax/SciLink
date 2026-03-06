#!/usr/bin/env python3
"""
scilink plan - Interactive Planning Orchestrator
Experimental design, data analysis, and Bayesian optimization
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def main():
    """Main entry point for 'scilink plan' command"""
    
    parser = argparse.ArgumentParser(
        prog='scilink plan',
        description='SciLink Planning Orchestrator - Interactive AI Research Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings (co-pilot mode)
  scilink plan
  
  # Use supervised mode (AI leads, human reviews plans/code)
  scilink plan --autonomy supervised --data-dir ./experimental_results
  
  # Full autonomous mode (no human review)
  scilink plan --autonomy autonomous --data-dir ./data --knowledge-dir ./papers --code-dir ./code
  
  # Use a different model
  scilink plan --model gemini-2.0-flash
  
  # Use Claude
  scilink plan --model claude-sonnet-4-20250514
  
  # Use OpenAI
  scilink plan --model gpt-4o
  
  # Use internal proxy with custom endpoint
  scilink plan --base-url https://my-proxy.example.com/v1 --model my-model

Autonomy Levels:
  co-pilot (default)  Human leads, AI assists. Reviews every step.
  supervised          AI leads, human supervises. Human reviews plans/code only.
  autonomous          Full autonomy. No human review, AI chains all tools.

  Note: supervised and autonomous modes require --data-dir to be specified.

Environment Variables:
  SCILINK_API_KEY          API key for internal proxy
  GEMINI_API_KEY           Google Gemini API key
  GOOGLE_API_KEY           Google API key (alias for GEMINI_API_KEY)
  OPENAI_API_KEY           OpenAI API key
  ANTHROPIC_API_KEY        Anthropic API key
  CLAUDE_API_KEY           Anthropic API key (alias)
  FUTUREHOUSE_API_KEY      FutureHouse API key for literature search (optional)

Supported Models:
  Google:    gemini-3.1-pro-preview, gemini-2.0-flash, gemini-1.5-pro, etc.
  OpenAI:    gpt-4o, gpt-4-turbo, o1-preview, etc.
  Anthropic: claude-sonnet-4-20250514, claude-opus-4-20250514, etc.
        """
    )
    
    # Model and API arguments
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-3.1-pro-preview',
        help='Model name (default: gemini-3.1-pro-preview)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        dest='base_url',
        help='Base URL for OpenAI-compatible endpoint'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        dest='embedding_model',
        default='gemini-embedding-001',
        help='Embedding model name (default: gemini-embedding-001)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        dest='api_key',
        help='API key for LLM provider (overrides environment variables)'
    )
    
    # Autonomy arguments
    parser.add_argument(
        '--autonomy',
        type=str,
        choices=['co-pilot', 'supervised', 'autonomous'],
        default='co-pilot',
        help='Autonomy level (default: co-pilot). Higher levels require --data-dir.'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        dest='data_dir',
        help='Path to experimental data directory (required for supervised/autonomous)'
    )
    
    parser.add_argument(
        '--knowledge-dir',
        type=str,
        dest='knowledge_dir',
        help='Path to papers/literature directory (optional)'
    )
    
    parser.add_argument(
        '--code-dir',
        type=str,
        dest='code_dir',
        help='Path to code/API documentation directory (optional)'
    )
    
    # Deprecated arguments (hidden but functional)
    parser.add_argument(
        '--local-model',
        type=str,
        dest='local_model',
        help=argparse.SUPPRESS
    )
    
    parser.add_argument(
        '--google-api-key',
        type=str,
        dest='google_api_key',
        help=argparse.SUPPRESS
    )
    
    args = parser.parse_args()
    
    # Handle deprecated --local-model -> --base-url
    base_url = args.base_url
    if args.local_model:
        import warnings
        warnings.warn(
            "'--local-model' is deprecated. Use '--base-url' instead.",
            DeprecationWarning
        )
        print("⚠️  Warning: '--local-model' is deprecated. Use '--base-url' instead.")
        if not base_url:
            base_url = args.local_model
    
    # Handle deprecated --google-api-key -> --api-key
    api_key = args.api_key
    if args.google_api_key:
        import warnings
        warnings.warn(
            "'--google-api-key' is deprecated. Use '--api-key' instead.",
            DeprecationWarning
        )
        print("⚠️  Warning: '--google-api-key' is deprecated. Use '--api-key' instead.")
        if not api_key:
            api_key = args.google_api_key
    
    # Validate: higher autonomy requires data-dir
    if args.autonomy in ('supervised', 'autonomous') and not args.data_dir:
        parser.error(
            f"--data-dir is required for {args.autonomy} mode.\n"
            f"Example: scilink plan --autonomy {args.autonomy} --data-dir ./experimental_results"
        )
    
    # Validate data-dir exists if provided
    if args.data_dir and not Path(args.data_dir).exists():
        parser.error(f"--data-dir path does not exist: {args.data_dir}")
    
    # Validate optional dirs if provided
    if args.knowledge_dir and not Path(args.knowledge_dir).exists():
        parser.error(f"--knowledge-dir path does not exist: {args.knowledge_dir}")
    if args.code_dir and not Path(args.code_dir).exists():
        parser.error(f"--code-dir path does not exist: {args.code_dir}")
    
    # Build config dict
    config = {
        'model_name': args.model,
        'base_url': base_url,
        'embedding_model': args.embedding_model,
        'api_key': api_key,
        'autonomy_level': args.autonomy,
        'data_dir': args.data_dir,
        'knowledge_dir': args.knowledge_dir,
        'code_dir': args.code_dir,
    }
    
    # Run the interactive orchestrator
    try:
        playground = OrchestratorPlayground(config)
        playground.run()
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ==============================================================================
# Interactive Orchestrator
# ==============================================================================

class OrchestratorPlayground:
    """Interactive session manager for the Planning Orchestrator Agent."""
    
    def __init__(self, config: dict = None):
        self.agent = None
        self.session_dir = None
        self.config = config or {}
        
        # Will be set during setup
        self.data_dir = None
        self.knowledge_dir = None
        self.code_dir = None
        
    def _infer_provider(self, model_name: str) -> tuple:
        """
        Infer provider info from model name.
        
        Returns:
            (provider_name, env_var_hint, env_vars_to_check)
        """
        model_lower = model_name.lower()
        
        if 'claude' in model_lower:
            return (
                "Anthropic",
                "ANTHROPIC_API_KEY or CLAUDE_API_KEY",
                ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"]
            )
        elif model_lower.startswith(('gpt-', 'o1-', 'o3-', 'text-embedding')):
            return (
                "OpenAI",
                "OPENAI_API_KEY",
                ["OPENAI_API_KEY"]
            )
        else:
            return (
                "Google Gemini",
                "GEMINI_API_KEY or GOOGLE_API_KEY",
                ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
            )
    
    def _get_api_key_from_env(self, env_vars: list) -> str:
        """Get API key from list of environment variables."""
        for var in env_vars:
            key = os.getenv(var)
            if key:
                return key
        return None
        
    def setup(self):
        """Setup the agent with user configuration."""
        from scilink.agents.planning_agents.planning_orchestrator import (
            PlanningOrchestratorAgent as OrchestratorAgent,
            AutonomyLevel
        )
        
        # Read from config (passed from CLI)
        model_name = self.config.get('model_name', 'gemini-3.1-pro-preview')
        base_url = self.config.get('base_url')
        embedding_model = self.config.get('embedding_model', 'gemini-embedding-001')
        api_key = self.config.get('api_key')
        autonomy_level_str = self.config.get('autonomy_level', 'co-pilot')
        self.data_dir = self.config.get('data_dir')
        self.knowledge_dir = self.config.get('knowledge_dir')
        self.code_dir = self.config.get('code_dir')
        
        # Convert autonomy level string to enum
        autonomy_map = {
            'co-pilot': AutonomyLevel.CO_PILOT,
            'supervised': AutonomyLevel.SUPERVISED,
            'autonomous': AutonomyLevel.AUTONOMOUS,
        }
        autonomy_level = autonomy_map.get(autonomy_level_str, AutonomyLevel.CO_PILOT)
        
        # Store for use in _process_initial_inputs
        self._autonomy_level = autonomy_level
        
        # === SHOW DIRECTORY GUIDE (CO-PILOT MODE ONLY) ===
        if autonomy_level == AutonomyLevel.CO_PILOT:
            print("\n" + "="*60)
            print("📁 RECOMMENDED DIRECTORY STRUCTURE")
            print("="*60)
            print("""
    Run orchestrator from your project directory:

    📁 my_project/
    ├── 📚 papers/               ← PDFs, scientific literature
    ├── 📊 experimental_results/ ← CSV/XLSX data files  
    └── 💻 code/                 ← (Optional) Scripts, API docs

    Then use natural language:
    "Generate a plan using ./papers/"
    "Analyze ./experimental_results/batch_001.csv"
    "Run optimization"
""")
            print("="*60)
        else:
            # Show workspace config for higher autonomy modes
            print("\n" + "="*60)
            print(f"📂 WORKSPACE CONFIGURATION ({autonomy_level.value} mode)")
            print("="*60)
            print(f"  Data directory:      {self.data_dir}")
            print(f"  Knowledge directory: {self.knowledge_dir or 'not configured'}")
            print(f"  Code directory:      {self.code_dir or 'not configured'}")
            print("="*60)
        
        # === API KEY RESOLUTION ===
        if not api_key:
            if base_url:
                # Internal proxy
                api_key = os.getenv("SCILINK_API_KEY")
                if not api_key:
                    print(f"\n⚠️  No SCILINK_API_KEY found in environment.")
                    print(f"   When using --base-url, set SCILINK_API_KEY for authentication.")
                    api_key = input(f"Enter your proxy API key (SCILINK_API_KEY): ").strip()
                    if not api_key:
                        print("❌ Cannot proceed without API key for internal proxy.")
                        sys.exit(1)
            else:
                # Public deployment - check provider-specific keys
                provider_name, env_var_hint, env_vars = self._infer_provider(model_name)
                api_key = self._get_api_key_from_env(env_vars)
                
                if not api_key:
                    print(f"\n⚠️  No {env_var_hint} found in environment.")
                    print(f"   LiteLLM will attempt to auto-detect credentials.")
                    user_key = input(f"Enter your {provider_name} API key (or Enter to auto-detect): ").strip()
                    if user_key:
                        api_key = user_key

        # === FUTUREHOUSE API KEY (Optional) ===
        futurehouse_key = os.getenv("FUTUREHOUSE_API_KEY")
        if not futurehouse_key:
            print("\n📚 FutureHouse API Key (Optional - for literature search)")
            print("   Enables scientific literature queries if you have a key.")
            futurehouse_key = input("   FutureHouse API key (or Enter to skip): ").strip()
            if futurehouse_key:
                print("   ✅ Literature search enabled")
            else:
                futurehouse_key = None
                print("   ℹ️  Literature search disabled")
        
        # === RESEARCH OBJECTIVE ===
        print("\n📋 What's your research objective?")
        print("Examples:")
        print("  - Optimize reaction yield")
        print("  - Screen drug candidates")
        print("  - Find optimal fermentation conditions")
        
        objective = input("\nYour objective: ").strip()
        if not objective:
            objective = "Optimize experimental conditions"
            print(f"   Using default: {objective}")
        
        # Store objective for use in _process_initial_inputs
        self._objective = objective
        
        # === SESSION DIRECTORY ===
        default_dir = f"./campaign_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n📁 Where should I save session data?")
        session_dir = input(f"   Path (default: {default_dir}): ").strip()
        if not session_dir:
            session_dir = default_dir
        
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # === INITIALIZE AGENT ===
        print("\n🔧 Initializing agent...")
        try:
            self.agent = OrchestratorAgent(
                objective=objective,
                base_dir=str(self.session_dir),
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                embedding_model=embedding_model,
                futurehouse_api_key=futurehouse_key,
                autonomy_level=autonomy_level,
                data_dir=self.data_dir,
                knowledge_dir=self.knowledge_dir,
                code_dir=self.code_dir,
            )
            print("✅ Agent ready!")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 Troubleshooting:")
            print("   1. Check that planning_agents package is installed")
            print("   2. Verify all dependencies are installed")
            print("   3. Check your API key is valid")
            print("   4. For supervised/autonomous: ensure directories exist")
            sys.exit(1)
        
        # === SHOW SESSION INFO ===
        print("\n" + "="*60)
        print("SESSION INFO")
        print("="*60)
        print(f"Objective: {objective}")
        print(f"Session Directory: {self.session_dir}")
        print(f"Autonomy Level: {autonomy_level.value}")
        print(f"Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
        
        # Model info
        provider_name, _, _ = self._infer_provider(model_name)
        if base_url:
            print(f"Model: {model_name}")
            print(f"Endpoint: {base_url}")
        else:
            print(f"Model: {model_name} ({provider_name})")
        
        print(f"Embedding Model: {embedding_model}")
        print(f"Literature Search: {'Enabled' if futurehouse_key else 'Disabled'}")
        
        # Directory info for higher autonomy
        if autonomy_level in (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS):
            print(f"\nWorkspace Directories:")
            print(f"  Data: {self.data_dir}")
            print(f"  Knowledge: {self.knowledge_dir or 'not configured'}")
            print(f"  Code: {self.code_dir or 'not configured'}")
        
        # Tools info
        print(f"\nAvailable Tools: {len(self.agent.tools.functions_map)}")
        tool_names = list(self.agent.tools.functions_map.keys())
        if len(tool_names) > 5:
            print(f"  {', '.join(tool_names[:5])}...")
        else:
            print(f"  {', '.join(tool_names)}")
        
    def print_help(self):
        """Print available commands."""
        print("\n" + "="*60)
        print("AVAILABLE COMMANDS")
        print("="*60)
        print("  /help              Show this help message")
        print("  /tools             List available tools")
        print("  /files             List files in workspace")
        print("  /state             Show agent state")
        print("  /autonomy [level]  Show or change autonomy level")
        print("  /checkpoint        Save checkpoint")
        print("  /clear             Clear screen")
        print("  /quit or /exit     Exit playground")
        print("\nOr just chat naturally with the agent!")
        print("="*60)
    
    def handle_command(self, user_input: str) -> bool:
        """
        Handle special commands.
        
        Returns:
            True if command was handled
            'QUIT' to exit
            False if not a command
        """
        cmd = user_input.lower().strip()
        
        if cmd == "/help":
            self.print_help()
            return True
        
        elif cmd == "/tools":
            print("\n📦 Available Tools:")
            for i, tool_name in enumerate(self.agent.tools.functions_map.keys(), 1):
                print(f"  {i}. {tool_name}")
            return True
        
        elif cmd == "/files":
            print("\n📁 Workspace Files:")
            files = list(self.session_dir.iterdir())
            if files:
                for f in sorted(files):
                    size = f.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    print(f"  - {f.name} ({size_str})")
            else:
                print("  (empty)")
            return True
        
        elif cmd == "/state":
            print("\n🔍 Agent State:")
            print(f"  Objective: {self.agent.objective}")
            print(f"  Autonomy Level: {self.agent.autonomy_level.value}")
            print(f"  Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Active Script: {Path(self.agent.active_scalarizer_script).name if self.agent.active_scalarizer_script else 'None'}")
            print(f"  Input Columns: {self.agent.expected_input_columns}")
            print(f"  Target Columns: {self.agent.expected_target_columns}")
            
            # Workspace directories
            if self.data_dir:
                print(f"  Data Directory: {self.data_dir}")
            if self.knowledge_dir:
                print(f"  Knowledge Directory: {self.knowledge_dir}")
            if self.code_dir:
                print(f"  Code Directory: {self.code_dir}")
            
            # Check data points
            if self.agent.bo_data_path.exists():
                import pandas as pd
                try:
                    df = pd.read_csv(self.agent.bo_data_path)
                    print(f"  Data Points: {len(df)}")
                except Exception:
                    print(f"  Data Points: Error reading file")
            else:
                print(f"  Data Points: 0")
            return True
        
        elif cmd.startswith("/autonomy"):
            from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
            
            parts = cmd.split()
            if len(parts) == 1:
                # Show current level
                print(f"\n🎛️  Current Autonomy Level: {self.agent.autonomy_level.value}")
                print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                print("\n   To change: /autonomy <co-pilot|supervised|autonomous>")
                print("   Note: Higher autonomy works best when started with --data-dir")
            else:
                # Change level
                level_map = {
                    'co-pilot': AutonomyLevel.CO_PILOT,
                    'copilot': AutonomyLevel.CO_PILOT,
                    'supervised': AutonomyLevel.SUPERVISED,
                    'autonomous': AutonomyLevel.AUTONOMOUS,
                }
                new_level = level_map.get(parts[1].lower())
                
                if new_level:
                    # Warn if switching to higher autonomy without directories
                    if new_level in (AutonomyLevel.SUPERVISED, AutonomyLevel.AUTONOMOUS):
                        if not self.data_dir:
                            print(f"\n   ⚠️  Warning: No data directory configured.")
                            print(f"   For best results, restart with: scilink plan --autonomy {new_level.value} --data-dir ./your_data")
                            print(f"   Proceeding anyway - agent may need to ask for file locations.")
                    
                    self.agent.set_autonomy_level(new_level)
                    print(f"\n   ✅ Autonomy level changed to: {new_level.value}")
                    print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                else:
                    print(f"\n   ❌ Unknown level: {parts[1]}")
                    print("   Valid options: co-pilot, supervised, autonomous")
            return True
        
        elif cmd == "/checkpoint":
            print("\n💾 Saving checkpoint...")
            response = self.agent.chat("Save checkpoint")
            print(response)
            return True
        
        elif cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🤖 Orchestrator Agent - Session Resumed\n")
            return True
        
        elif cmd in ["/quit", "/exit", "/q", "quit", "exit"]:
            return "QUIT"
        
        return False
    
    def _process_initial_inputs(self):
        """
        Process initial workspace directories based on autonomy level.
        
        Behavior by mode:
        - co-pilot: Do nothing (human leads, full backward compatibility)
        - supervised: Survey workspace, analyze available data, suggest next steps
        - autonomous: Execute full pipeline (survey → TEA → plan → checkpoint)
        """
        from scilink.agents.planning_agents.planning_orchestrator import AutonomyLevel
        
        autonomy = self.agent.autonomy_level
        
        # === CO-PILOT MODE: Human leads, don't auto-start anything ===
        # This maintains full backward compatibility
        if autonomy == AutonomyLevel.CO_PILOT:
            return
        
        has_data = self.data_dir is not None
        has_knowledge = self.knowledge_dir is not None
        has_code = self.code_dir is not None
        
        # Nothing to process if no directories configured
        # (This shouldn't happen for supervised/autonomous due to CLI validation,
        # but we handle it gracefully anyway)
        if not has_data and not has_knowledge:
            return
        
        print("\n" + "-"*60)
        print(f"🚀 AUTO-PROCESSING WORKSPACE ({autonomy.value} mode)")
        print("-"*60)
        
        # === AUTONOMOUS MODE: Full pipeline execution ===
        if autonomy == AutonomyLevel.AUTONOMOUS:
            print("🤖 AUTONOMOUS MODE: Executing complete research workflow...")
            print(f"   Objective: {self.agent.objective}")
            print(f"   Data: {self.data_dir}")
            print(f"   Knowledge: {self.knowledge_dir or 'not provided'}")
            print(f"   Code: {self.code_dir or 'not provided'}")
            print("-"*60 + "\n")
            
            # Build comprehensive instruction for full autonomous execution
            instruction_parts = [
                f"Execute the complete research workflow for objective: '{self.agent.objective}'."
            ]
            
            # Step 1: Survey
            instruction_parts.append(
                "Step 1: Survey the workspace using list_workspace_files to understand available data."
            )
            
            # Step 2: Economic analysis (if knowledge available)
            if has_knowledge:
                instruction_parts.append(
                    f"Step 2: Run economic analysis using knowledge from {self.knowledge_dir} "
                    f"and experimental data from {self.data_dir} to assess viability."
                )
            else:
                instruction_parts.append(
                    f"Step 2: Skip economic analysis (no knowledge directory provided)."
                )
            
            # Step 3: Generate plan
            instruction_parts.append(
                f"Step 3: Generate an initial experimental plan based on the objective, "
                f"available data in {self.data_dir}"
                + (f", and literature in {self.knowledge_dir}" if has_knowledge else "")
                + "."
            )
            
            # Step 4: Generate code (if code KB available)
            if has_code:
                instruction_parts.append(
                    f"Step 4: Generate implementation code using the code knowledge base in {self.code_dir}."
                )
            else:
                instruction_parts.append(
                    "Step 4: Skip code generation (no code directory provided)."
                )
            
            # Step 5: Checkpoint
            instruction_parts.append(
                "Step 5: Save a checkpoint to preserve the campaign state."
            )
            
            instruction_parts.append(
                "Execute ALL steps without stopping for confirmation. "
                "Chain tool calls as needed to complete the entire workflow."
            )
            
            # Execute the full pipeline
            self.agent.chat(" ".join(instruction_parts))
            
            print("\n" + "-"*60)
            print("✅ Autonomous workflow complete.")
            print("   Entering interactive mode for follow-up questions.")
            print("-"*60)
            return
        
        # === SUPERVISED MODE: Survey and recommend ===
        if autonomy == AutonomyLevel.SUPERVISED:
            print("🔄 SUPERVISED MODE: Surveying workspace and preparing recommendations...")
            print(f"   Objective: {self.agent.objective}")
            print(f"   Data: {self.data_dir}")
            print(f"   Knowledge: {self.knowledge_dir or 'not provided'}")
            print("-"*60 + "\n")
            
            # Step 1: Survey workspace
            print("📂 Step 1: Surveying workspace...")
            self.agent.chat(
                "Survey the workspace using list_workspace_files. "
                "Report what data files, papers, and other resources are available."
            )
            
            # Step 2: Analyze and recommend next steps
            print("\n🧠 Step 2: Analyzing and suggesting next steps...")
            
            recommendation_prompt = (
                f"Based on the workspace contents, recommend the best next steps for "
                f"achieving the objective: '{self.agent.objective}'. "
                f"Consider the following options and recommend which to do first:\n"
            )
            
            if has_knowledge:
                recommendation_prompt += (
                    f"- Run economic/TEA analysis using papers in {self.knowledge_dir}\n"
                )
            
            recommendation_prompt += (
                f"- Generate an experimental plan based on available data\n"
                f"- Analyze existing experimental results\n"
                f"\nProvide a clear recommendation with reasoning, then proceed with "
                f"the recommended action."
            )
            
            self.agent.chat(recommendation_prompt)
            
            print("\n" + "-"*60)
            print("✅ Workspace analysis complete.")
            print("   Review the recommendations above and provide direction,")
            print("   or let the agent continue with its suggested approach.")
            print("-"*60)
            return
    
    def run(self):
        """Main interactive loop."""
        self.setup()
        self.print_help()
        
        print("\n" + "="*60)
        print("💬 CHAT SESSION STARTED")
        print("="*60)
        print("Type /help for commands, or just chat naturally!\n")
        
        # Process initial workspace based on autonomy level ===
        # Note: This does nothing for co-pilot mode (backward compatible)
        self._process_initial_inputs()
        
        # === Main chat loop ===
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                result = self.handle_command(user_input)
                if result == "QUIT":
                    print("\n👋 Goodbye! Your session is saved at:")
                    print(f"   {self.session_dir}")
                    break
                elif result:
                    continue
                
                # Send to agent
                response = self.agent.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type /quit to exit properly.")
            except EOFError:
                print("\n\n👋 Session ended.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                print("   Type /help for commands or /quit to exit")


if __name__ == '__main__':
    sys.exit(main())