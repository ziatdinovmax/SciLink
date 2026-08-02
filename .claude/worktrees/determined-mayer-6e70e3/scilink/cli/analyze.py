#!/usr/bin/env python3
"""
scilink analyze - Interactive Analysis Orchestrator
Experimental data analysis with automatic agent selection
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def main():
    """Main entry point for 'scilink analyze' command"""
    
    parser = argparse.ArgumentParser(
        prog='scilink analyze',
        description='SciLink Analysis Orchestrator - Interactive AI Analysis Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings (co-pilot mode)
  scilink analyze
  
  # Start with data file
  scilink analyze --data ./sample.tif --metadata ./metadata.json
  
  # Use supervised mode (AI leads, human approves)
  scilink analyze --mode supervised --data ./data/
  
  # Full autonomous mode (runs entire pipeline automatically)
  scilink analyze --mode autonomous --data ./sample.npy --metadata ./description.txt
  
  # Use a different model
  scilink analyze --model gemini-2.0-flash
  
  # Use Claude
  scilink analyze --model claude-sonnet-4-20250514
  
  # Use OpenAI
  scilink analyze --model gpt-4o

Analysis Modes (matching 'scilink plan' for consistent UX):
  co-pilot (default)   Human leads, AI assists. Reviews all agent selections.
  supervised           AI leads, human approves. AI proceeds with reasonable defaults.
  autonomous           Full autonomy. AI selects agents and runs without confirmation.

Environment Variables:
  SCILINK_API_KEY          API key for internal proxy
  GEMINI_API_KEY           Google Gemini API key
  GOOGLE_API_KEY           Google API key (alias for GEMINI_API_KEY)
  OPENAI_API_KEY           OpenAI API key
  ANTHROPIC_API_KEY        Anthropic API key
  CLAUDE_API_KEY           Anthropic API key (alias)
  FUTUREHOUSE_API_KEY      FutureHouse API key for literature search (optional)

Supported Data Types:
  Microscopy:    .tif, .tiff, .png, .jpg, .jpeg, .bmp
  Spectroscopy:  .npy (3D hyperspectral)
  Curves:        .npy (1D/2D), .csv, .txt

Metadata Options:
  --metadata file.json     Load structured JSON metadata
  --metadata file.txt      Convert natural language to metadata
  (or provide metadata interactively during session)
        """
    )
    
    # Model and API arguments
    parser.add_argument(
        '--model',
        type=str,
        default='claude-opus-4-6',
        help='Model name (default: claude-opus-4-6)'
    )
    
    parser.add_argument(
        '--base-url',
        type=str,
        dest='base_url',
        help='Base URL for OpenAI-compatible endpoint'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        dest='api_key',
        help='API key for LLM provider (overrides environment variables)'
    )

    parser.add_argument(
        '--mp-api-key',
        type=str,
        dest='mp_api_key',
        help='Materials Project API key (or set MP_API_KEY env var). '
             'Enables the MP tool-resolver in structure generation.'
    )
    
    # Mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['co-pilot', 'supervised', 'autonomous'],
        default='co-pilot',
        help='Analysis mode (default: co-pilot)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        dest='data_path',
        help='Path to data file or directory'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        dest='metadata_path',
        help='Path to metadata file (.json or .txt)'
    )
    
    # Custom agent arguments
    parser.add_argument(
        '--agents',
        type=str,
        nargs='+',
        dest='agent_files',
        metavar='AGENT_FILE',
        help=(
            'Path(s) to Python files containing custom BaseAnalysisAgent subclasses. '
            'All subclasses found in each file are registered automatically. '
            'Example: scilink analyze --agents ./my_xrd_agent.py'
        )
    )

    # Custom skill arguments
    parser.add_argument(
        '--skills',
        type=str,
        nargs='+',
        dest='skill_files',
        metavar='SKILL_FILE',
        help=(
            'Path(s) to custom skill .md files providing domain-specific '
            'guidance for analysis agents (e.g. fitting strategy, interpretation '
            'rules). Skills are made available by name to the run_analysis tool. '
            'Example: scilink analyze --skills ./raman_skill.md ./ftir_skill.md'
        )
    )

    # Custom tool arguments
    parser.add_argument(
        '--tools',
        type=str,
        nargs='+',
        dest='tool_files',
        metavar='TOOL_FILE',
        help=(
            'Path(s) to Python files containing domain-specific tool functions to '
            'expose directly to the orchestrator\'s LLM loop. '
            'Each file must define: (1) a list of OpenAI-format tool schemas named '
            '\'tool_schemas\', \'openai_schemas\', or any top-level list of '
            'OpenAI function dicts; (2) a factory function named '
            '\'create_tool_functions\' (or any function ending in \'_tool_functions\') '
            'that accepts data and returns a dict mapping tool names to callables. '
            'Example: scilink analyze --tools ./image_analysis_tools.py'
        )
    )

    # MCP server arguments
    parser.add_argument(
        '--mcp',
        type=str,
        nargs='+',
        dest='mcp_servers',
        metavar='MCP_CONFIG',
        help=(
            'MCP server configurations. Each entry can be:\n'
            '  - A JSON config file ({"name":"...", "command":["..."], "env":{}})\n'
            '  - stdio shorthand:  stdio:name:command,arg1,arg2\n'
            '  - SSE shorthand:    sse:name:http://host:port/sse\n'
            'Example: scilink analyze --mcp stdio:fs:npx,-y,@modelcontextprotocol/server-filesystem,/tmp'
        )
    )

    # Session arguments
    parser.add_argument(
        '--session-dir',
        type=str,
        dest='session_dir',
        help='Session directory for outputs (default: auto-generated)'
    )
    
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore from previous checkpoint in session directory'
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
    
    # Handle deprecated arguments
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

    # Register the MP key (if provided via flag) into the in-memory keystore so
    # auto-discovery picks it up when run_dft_workflow constructs DFTOrchestrator.
    if args.mp_api_key:
        import scilink
        scilink.set_api_key('materials_project', args.mp_api_key)
    
    # Validate data path if provided
    if args.data_path and not Path(args.data_path).exists():
        parser.error(f"--data path does not exist: {args.data_path}")
    
    # Validate metadata path if provided
    if args.metadata_path and not Path(args.metadata_path).exists():
        parser.error(f"--metadata path does not exist: {args.metadata_path}")
    
    # Validate custom agent files if provided
    if args.agent_files:
        for af in args.agent_files:
            if not Path(af).exists():
                parser.error(f"--agents path does not exist: {af}")
            if not af.endswith('.py'):
                parser.error(f"--agents file must be a .py file: {af}")

    # Validate custom skill files if provided
    if args.skill_files:
        for sf in args.skill_files:
            if not Path(sf).exists():
                parser.error(f"--skills path does not exist: {sf}")
            if not sf.endswith('.md'):
                parser.error(f"--skills file must be a .md file: {sf}")

    # Validate custom tool files if provided
    if args.tool_files:
        for tf in args.tool_files:
            if not Path(tf).exists():
                parser.error(f"--tools path does not exist: {tf}")
            if not tf.endswith('.py'):
                parser.error(f"--tools file must be a .py file: {tf}")

    # Build config dict
    config = {
        'model_name': args.model,
        'base_url': base_url,
        'api_key': api_key,
        'analysis_mode': args.mode,
        'data_path': args.data_path,
        'metadata_path': args.metadata_path,
        'session_dir': args.session_dir,
        'restore': args.restore,
        'agent_files': args.agent_files or [],
        'tool_files': args.tool_files or [],
        'skill_files': args.skill_files or [],
        'mcp_servers': args.mcp_servers or [],
    }
    
    # Run the interactive orchestrator
    try:
        playground = AnalysisPlayground(config)
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

class AnalysisPlayground:
    """Interactive session manager for the Analysis Orchestrator Agent."""
    
    def __init__(self, config: dict = None):
        self.agent = None
        self.session_dir = None
        self.config = config or {}
        
        # Store initial paths explicitly
        self._initial_data_path = None
        self._initial_metadata_path = None
        
    def _infer_provider(self, model_name: str) -> tuple:
        """Infer provider info from model name."""
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
        from scilink.agents.exp_agents.analysis_orchestrator import (
            AnalysisOrchestratorAgent,
            AnalysisMode
        )
        
        # Read from config
        model_name = self.config.get('model_name', 'claude-opus-4-6')
        base_url = self.config.get('base_url')
        api_key = self.config.get('api_key')
        analysis_mode_str = self.config.get('analysis_mode', 'co-pilot')
        data_path = self.config.get('data_path')
        metadata_path = self.config.get('metadata_path')
        session_dir = self.config.get('session_dir')
        restore = self.config.get('restore', False)
        
        # Store initial paths and agent/tool files for later use in run()
        self._initial_data_path = data_path
        self._initial_metadata_path = metadata_path
        self._agent_files = self.config.get('agent_files', [])
        self._tool_files = self.config.get('tool_files', [])
        self._skill_files = self.config.get('skill_files', [])
        self._mcp_servers = self.config.get('mcp_servers', [])
        
        # Convert mode string to enum
        mode_map = {
            'co-pilot': AnalysisMode.CO_PILOT,
            'supervised': AnalysisMode.SUPERVISED,
            'autonomous': AnalysisMode.AUTONOMOUS,
        }
        analysis_mode = mode_map.get(analysis_mode_str, AnalysisMode.CO_PILOT)
        
        # Show welcome message
        print("\n" + "="*60)
        print("🔬 SCILINK ANALYSIS ORCHESTRATOR")
        print("="*60)
        print("""
This agent helps you analyze experimental data by:
1. Examining your data to determine the appropriate analysis
2. Managing metadata (loading or converting from text)
3. Selecting the best analysis agent for your data
4. Running analysis and generating scientific insights
5. Assessing novelty of findings against literature (New!)

Supported data types:
  • Microscopy images (.tif, .png, .jpg)
  • Spectroscopy data (.npy - hyperspectral)
  • 1D curves/spectra (.npy, .csv, .txt)
""")
        print("="*60)
        
        # === API KEY RESOLUTION ===
        if not api_key:
            if base_url:
                api_key = os.getenv("SCILINK_API_KEY")
                if not api_key:
                    print(f"\n⚠️  No SCILINK_API_KEY found in environment.")
                    api_key = input(f"Enter your proxy API key (SCILINK_API_KEY): ").strip()
                    if not api_key:
                        print("❌ Cannot proceed without API key for internal proxy.")
                        sys.exit(1)
            else:
                provider_name, env_var_hint, env_vars = self._infer_provider(model_name)
                api_key = self._get_api_key_from_env(env_vars)
                
                if not api_key:
                    print(f"\n⚠️  No {env_var_hint} found in environment.")
                    user_key = input(f"Enter your {provider_name} API key (or Enter to auto-detect): ").strip()
                    if user_key:
                        api_key = user_key

        # === FUTUREHOUSE API KEY (Optional) ===
        futurehouse_key = os.getenv("FUTUREHOUSE_API_KEY")
        if not futurehouse_key:
            print("\n📚 FutureHouse API Key (Optional - for novelty assessment)")
            print("   Enables literature search to check if findings are novel.")
            futurehouse_key = input("   FutureHouse API key (or Enter to skip): ").strip()
            if futurehouse_key:
                print("   ✅ Novelty assessment enabled")
            else:
                futurehouse_key = None
                print("   ℹ️  Novelty assessment disabled")
        
        # === SESSION DIRECTORY ===
        if session_dir:
            self.session_dir = Path(session_dir)
        elif restore:
            # Look for existing session
            default_pattern = "./analysis_session_*"
            import glob
            sessions = sorted(glob.glob(default_pattern), reverse=True)
            if sessions:
                self.session_dir = Path(sessions[0])
                print(f"\n📂 Found session to restore: {self.session_dir}")
            else:
                print("\n⚠️  No existing sessions found. Creating new session.")
                self.session_dir = Path(f"./analysis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            default_dir = f"./analysis_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n📁 Where should I save session data?")
            session_input = input(f"   Path (default: {default_dir}): ").strip()
            self.session_dir = Path(session_input) if session_input else Path(default_dir)
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # === INITIALIZE AGENT ===
        print("\n🔧 Initializing agent...")
        try:
            self.agent = AnalysisOrchestratorAgent(
                base_dir=str(self.session_dir),
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                analysis_mode=analysis_mode,
                restore_checkpoint=restore,
                futurehouse_api_key=futurehouse_key
            )
            print("✅ Agent ready!")

        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # === LOAD CUSTOM AGENT FILES ===
        if self._agent_files:
            self._load_custom_agents(self._agent_files)

        # === LOAD CUSTOM TOOL FILES ===
        if self._tool_files:
            self._load_custom_tools(self._tool_files)

        # === REGISTER CUSTOM SKILLS ===
        if self._skill_files:
            self._register_custom_skills(self._skill_files)

        # === CONNECT MCP SERVERS ===
        if self._mcp_servers:
            self._connect_mcp_servers(self._mcp_servers)

        # === SHOW SESSION INFO ===
        print("\n" + "="*60)
        print("SESSION INFO")
        print("="*60)
        print(f"Session Directory: {self.session_dir}")
        print(f"Analysis Mode: {analysis_mode.value}")
        print(f"Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
        print(f"Novelty Assessment: {'Enabled' if futurehouse_key else 'Disabled'}")
        
        provider_name, _, _ = self._infer_provider(model_name)
        if base_url:
            print(f"Model: {model_name}")
            print(f"Endpoint: {base_url}")
        else:
            print(f"Model: {model_name} ({provider_name})")
        
        print(f"\nAvailable Tools: {len(self.agent.tools.functions_map)}")
        tool_names = list(self.agent.tools.functions_map.keys())
        print(f"  {', '.join(tool_names[:5])}...")
        
        # Show initial inputs (if any)
        if self._initial_data_path:
            print(f"\n📊 Initial data: {self._initial_data_path}")
        
        if self._initial_metadata_path:
            print(f"📋 Initial metadata: {self._initial_metadata_path}")
        
    def _load_custom_agents(self, agent_files: list) -> None:
        """Load BaseAnalysisAgent subclasses from user-supplied .py files and register them."""
        import importlib.util
        import inspect
        import sys
        from scilink.agents.exp_agents.base_agent import BaseAnalysisAgent

        for file_path in agent_files:
            path = Path(file_path).resolve()
            print(f"\n🔌 Loading custom agents from: {path}")
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                module = importlib.util.module_from_spec(spec)
                _prev = sys.dont_write_bytecode
                sys.dont_write_bytecode = True
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.dont_write_bytecode = _prev
            except Exception as e:
                print(f"   ❌ Failed to load {path.name}: {e}")
                continue

            found = 0
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(cls, BaseAnalysisAgent)
                    and cls is not BaseAnalysisAgent
                    and cls.__module__ == module.__name__
                ):
                    next_id = max(self.agent._agent_registry.keys()) + 1
                    self.agent.register_agent(next_id, cls)
                    print(f"   ✅ Registered '{cls.__name__}' as agent ID {next_id}")
                    found += 1

            if found == 0:
                print(f"   ⚠️  No BaseAnalysisAgent subclasses found in {path.name}")

    def _load_custom_tools(self, tool_files: list) -> None:
        """Load external tool functions from user-supplied .py files and register them.

        Each file must expose:
          1. A list of OpenAI-format tool schemas — discovered by looking for a
             module-level variable named ``tool_schemas`` or ``openai_schemas``
             first, then falling back to any top-level list whose first element is
             a dict with ``type == "function"``.
          2. A factory function that accepts data and returns a dict mapping tool
             names to callables.  The factory is discovered by looking for
             ``create_tool_functions`` first, then any module-level function whose
             name ends with ``_tool_functions``.

        The orchestrator's ``register_tools()`` method decides at call-time whether
        to pass the current data path or a loaded NumPy/DataFrame object to the
        factory, based on the name of the factory's first parameter.
        """
        import importlib.util
        import inspect
        import sys

        for file_path in tool_files:
            path = Path(file_path).resolve()
            print(f"\n🔧 Loading custom tools from: {path}")
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                module = importlib.util.module_from_spec(spec)
                _prev = sys.dont_write_bytecode
                sys.dont_write_bytecode = True
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.dont_write_bytecode = _prev
            except Exception as e:
                print(f"   ❌ Failed to load {path.name}: {e}")
                continue

            # ── Discover schemas ──────────────────────────────────────────────
            schemas = (
                getattr(module, 'tool_schemas', None)
                or getattr(module, 'openai_schemas', None)
            )
            if schemas is None:
                # Auto-detect: any top-level list of OpenAI function dicts
                for attr_name in dir(module):
                    obj = getattr(module, attr_name, None)
                    if (
                        isinstance(obj, list)
                        and obj
                        and isinstance(obj[0], dict)
                        and obj[0].get('type') == 'function'
                    ):
                        schemas = obj
                        print(f"   📋 Auto-detected schemas in '{attr_name}'")
                        break

            if not schemas:
                print(
                    f"   ⚠️  No tool schemas found in {path.name}.\n"
                    "       Define 'tool_schemas' as a list of OpenAI-format tool dicts."
                )
                continue

            # ── Discover factory ──────────────────────────────────────────────
            factory = getattr(module, 'create_tool_functions', None)
            if factory is None:
                for name, fn in inspect.getmembers(module, inspect.isfunction):
                    if (
                        name.endswith('_tool_functions')
                        and fn.__module__ == module.__name__
                    ):
                        factory = fn
                        print(f"   🏭 Auto-detected factory '{name}'")
                        break

            if factory is None:
                print(
                    f"   ⚠️  No factory function found in {path.name}.\n"
                    "       Define 'create_tool_functions(data)' returning "
                    "a dict mapping tool names to callables."
                )
                continue

            self.agent.register_tools(schemas, factory)
            count = sum(1 for s in schemas if s.get('type') == 'function')
            print(f"   ✅ Registered {count} tool(s) from {path.name}")

    def _register_custom_skills(self, skill_files: list) -> None:
        """Register custom skill .md files with the orchestrator."""
        for file_path in skill_files:
            path = Path(file_path).resolve()
            print(f"\n📖 Registering custom skill: {path}")
            try:
                name = self.agent.register_skill(str(path))
                print(f"   ✅ Registered skill '{name}'")
            except Exception as e:
                print(f"   ❌ Failed to register {path.name}: {e}")

    def _connect_mcp_servers(self, mcp_configs: list) -> None:
        """Parse MCP server configs and connect to each."""
        import json as _json

        for entry in mcp_configs:
            try:
                if entry.startswith("stdio:"):
                    # stdio:name:cmd,arg1,arg2
                    parts = entry[len("stdio:"):].split(":", 1)
                    name = parts[0]
                    command = parts[1].split(",") if len(parts) > 1 else []
                    print(f"\n🔌 Connecting to MCP server '{name}' (stdio)...")
                    count = self.agent.connect_mcp_server(
                        name, command=command
                    )
                    print(f"   ✅ Registered {count} tool(s) from '{name}'")

                elif entry.startswith("sse:"):
                    # sse:name:http://host:port/path
                    parts = entry[len("sse:"):].split(":", 1)
                    name = parts[0]
                    url = parts[1] if len(parts) > 1 else ""
                    print(f"\n🔌 Connecting to MCP server '{name}' (SSE)...")
                    count = self.agent.connect_mcp_server(name, url=url)
                    print(f"   ✅ Registered {count} tool(s) from '{name}'")

                else:
                    # JSON config file
                    path = Path(entry).resolve()
                    with open(path) as f:
                        cfg = _json.load(f)
                    name = cfg.get("name", path.stem)
                    print(f"\n🔌 Connecting to MCP server '{name}'...")
                    count = self.agent.connect_mcp_server(
                        name,
                        command=cfg.get("command"),
                        url=cfg.get("url"),
                        env=cfg.get("env"),
                    )
                    print(f"   ✅ Registered {count} tool(s) from '{name}'")

            except Exception as e:
                print(f"   ❌ Failed to connect MCP server '{entry}': {e}")

    def print_help(self):
        """Print available commands."""
        print("\n" + "="*60)
        print("AVAILABLE COMMANDS")
        print("="*60)
        print("  /help              Show this help message")
        print("  /tools             List available tools")
        print("  /agents            List available analysis agents")
        print("  /status            Show current session state")
        print("  /mode [level]      Show or change analysis mode")
        print("  /checkpoint        Save checkpoint")
        print("  /schema            Show metadata schema")
        print("  /clear             Clear screen")
        print("  /quit or /exit     Exit")
        print("\nOr just chat naturally with the agent!")
        print("="*60)
    
    def handle_command(self, user_input: str) -> bool:
        """Handle special commands. Returns True if handled, 'QUIT' to exit, False otherwise."""
        cmd = user_input.lower().strip()
        
        if cmd == "/help":
            self.print_help()
            return True
        
        elif cmd == "/tools":
            print("\n📦 Available Tools:")
            for i, tool_name in enumerate(self.agent.tools.functions_map.keys(), 1):
                print(f"  {i}. {tool_name}")
            return True
        
        elif cmd == "/agents":
            print("\n🤖 Available Analysis Agents:")
            for agent_id, name in self.agent.tools.AGENT_NAMES.items():
                desc = self.agent.tools.AGENT_DESCRIPTIONS[agent_id]
                selected = " ← selected" if agent_id == self.agent.selected_agent_id else ""
                print(f"  {agent_id}: {name}")
                print(f"     {desc}{selected}")
            return True
        
        elif cmd == "/status":
            print("\n🔍 Session State:")
            print(f"  Session Directory: {self.session_dir}")
            print(f"  Analysis Mode: {self.agent.analysis_mode.value}")
            print(f"  Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Current Data: {self.agent.current_data_path or 'None'}")
            print(f"  Current Data Type: {self.agent.current_data_type or 'None'}")
            print(f"  Selected Agent: {self.agent.selected_agent_id}")
            print(f"  Has Metadata: {'Yes' if self.agent.current_metadata else 'No'}")
            print(f"  Analyses Completed: {len(self.agent.analysis_results)}")
            return True
        
        elif cmd.startswith("/mode"):
            from scilink.agents.exp_agents.analysis_orchestrator import AnalysisMode
            
            parts = cmd.split()
            if len(parts) == 1:
                print(f"\n🎛️  Current Analysis Mode: {self.agent.analysis_mode.value}")
                print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                print("\n   To change: /mode <co-pilot|supervised|autonomous>")
            else:
                mode_map = {
                    'co-pilot': AnalysisMode.CO_PILOT,
                    'supervised': AnalysisMode.SUPERVISED,
                    'autonomous': AnalysisMode.AUTONOMOUS,
                }
                new_mode = mode_map.get(parts[1].lower())
                
                if new_mode:
                    self.agent.set_analysis_mode(new_mode)
                    print(f"\n   ✅ Analysis mode changed to: {new_mode.value}")
                    print(f"   Human Feedback: {'Enabled' if self.agent._enable_human_feedback else 'Disabled'}")
                else:
                    print(f"\n   ❌ Unknown mode: {parts[1]}")
                    print("   Valid options: co-pilot, supervised, autonomous")
            return True
        
        elif cmd == "/checkpoint":
            print("\n💾 Saving checkpoint...")
            response = self.agent.chat("Save checkpoint")
            print(response)
            return True
        
        elif cmd == "/schema":
            from scilink.agents.exp_agents.metadata_converter import METADATA_SCHEMA_DICT
            import json
            print("\n📋 Metadata Schema:")
            print(json.dumps(METADATA_SCHEMA_DICT, indent=2))
            return True
        
        elif cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("🔬 Analysis Orchestrator - Session Resumed\n")
            return True
        
        elif cmd in ["/quit", "/exit", "/q", "quit", "exit"]:
            return "QUIT"
        
        return False
    
    def _process_initial_inputs(self):
        """
        Process initial --data and --metadata inputs.
        Handles different modes appropriately:
        - co-pilot: Examine data, load metadata, then wait for user
        - supervised: Examine, load metadata, suggest agent, wait for approval
        - autonomous: Run the entire pipeline automatically
        """
        has_data = self._initial_data_path is not None
        has_metadata = self._initial_metadata_path is not None
        
        if not has_data and not has_metadata:
            return  # Nothing to auto-process
        
        mode = self.agent.analysis_mode.value
        
        print("\n" + "-"*60)
        print("🚀 AUTO-PROCESSING INITIAL INPUTS")
        print("-"*60)
        
        # === AUTONOMOUS MODE: Full pipeline ===
        if mode == 'autonomous':
            if has_data and has_metadata:
                print(f"🤖 AUTONOMOUS MODE: Running complete analysis pipeline...")
                print(f"   Data: {self._initial_data_path}")
                print(f"   Metadata: {self._initial_metadata_path}")
                print("-"*60 + "\n")
                
                # Determine metadata type
                meta_ext = Path(self._initial_metadata_path).suffix.lower()
                if meta_ext == '.json':
                    meta_instruction = f"load the metadata from {self._initial_metadata_path}"
                else:
                    meta_instruction = f"convert the metadata from {self._initial_metadata_path}"
                
                # Single comprehensive instruction for autonomous execution
                self.agent.chat(
                    f"Analyze the data at {self._initial_data_path}. "
                    f"First examine the data, then {meta_instruction}, "
                    f"then select the appropriate agent based on the data type and metadata, "
                    f"and finally run the analysis. Execute the complete workflow."
                )
                
                print("\n" + "-"*60)
                print("✅ Autonomous analysis complete.")
                print("   Entering interactive mode for follow-up questions.")
                print("-"*60)
                return
            
            elif has_data and not has_metadata:
                print("⚠️  AUTONOMOUS MODE requires both --data and --metadata")
                print("   Reason: Cannot auto-select agent without metadata context.")
                print("   Falling back to examining data only...\n")
                # Fall through to examine data
            
            elif has_metadata and not has_data:
                print("⚠️  AUTONOMOUS MODE requires both --data and --metadata")
                print("   Reason: No data to analyze.")
                print("   Falling back to loading metadata only...\n")
                # Fall through to load metadata
        
        # === CO-PILOT / SUPERVISED / Fallback: Step-by-step ===
        
        # Step 1: Examine data (if provided)
        if has_data:
            data_path = str(Path(self._initial_data_path).absolute())
            print(f"📊 Examining data: {self._initial_data_path}")
            self.agent.chat(f"Examine the data at {data_path}")
            print()
        
        # Step 2: Load/convert metadata (if provided)
        if has_metadata:
            meta_path = str(Path(self._initial_metadata_path).absolute())
            meta_ext = Path(self._initial_metadata_path).suffix.lower()
            
            if meta_ext == '.json':
                print(f"📋 Loading metadata: {self._initial_metadata_path}")
                self.agent.chat(f"Load the metadata from {meta_path}")
            else:
                print(f"📋 Converting metadata: {self._initial_metadata_path}")
                self.agent.chat(f"Convert the text description to metadata from {meta_path}")
            print()
        
        # Step 3: In supervised mode with both inputs, suggest next step
        if mode == 'supervised' and has_data and has_metadata:
            print("🔄 SUPERVISED MODE: Suggesting agent selection...")
            self.agent.chat(
                "Based on the data type and metadata, recommend the appropriate "
                "analysis agent and explain your reasoning."
            )
            print()
        
        print("-"*60)
        print("✅ Initial processing complete. Ready for your input.")
        print("-"*60)
    
    def run(self):
        """Main interactive loop."""
        self.setup()
        self.print_help()
        
        print("\n" + "="*60)
        print("💬 CHAT SESSION STARTED")
        print("="*60)
        print("Type /help for commands, or just chat naturally!\n")
        
        # === NEW: Process initial --data and --metadata inputs ===
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