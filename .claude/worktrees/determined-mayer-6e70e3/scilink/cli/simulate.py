#!/usr/bin/env python3
"""
scilink simulate - Interactive Simulation Orchestrator

VASP DFT input preparation: build atomic structures, generate INCAR/KPOINTS,
analyze post-run outputs. Local prep only — does not submit jobs to HPC
clusters or run VASP itself (HPC submission tools are planned follow-up
work; see CLAUDE.md).
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


# ==============================================================================
# CLI entry point
# ==============================================================================

def main():
    """Main entry point for 'scilink simulate' command."""

    parser = argparse.ArgumentParser(
        prog='scilink simulate',
        description='SciLink Simulation Orchestrator - Interactive VASP DFT input prep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start an empty co-pilot session and chat naturally
  scilink simulate

  # Seed the session with an initial structure request
  scilink simulate --request "Build a rutile TiO2 supercell with one O vacancy"

  # Supervised mode (AI proceeds with reasonable defaults; surfaces big decisions)
  scilink simulate --mode supervised

  # Use a different model
  scilink simulate --model gemini-2.0-flash

  # Internal proxy
  scilink simulate --base-url https://my-proxy.example.com/v1 --model my-model

Modes (matching `scilink analyze` / `scilink plan` for consistent UX):
  co-pilot (default)   Human leads, AI assists. Confirms before each tool call.
  supervised           AI leads with defaults; surfaces significant decisions.
  autonomous           Full autonomy. AI executes without confirmation.

Environment Variables:
  SCILINK_API_KEY          API key for internal proxy
  GEMINI_API_KEY           Google Gemini API key
  GOOGLE_API_KEY           Google API key (alias for GEMINI_API_KEY)
  OPENAI_API_KEY           OpenAI API key
  ANTHROPIC_API_KEY        Anthropic API key
  CLAUDE_API_KEY           Anthropic API key (alias)
  MP_API_KEY               Materials Project API key (enables MP tool-resolver)
  FUTUREHOUSE_API_KEY      FutureHouse API key (enables INCAR literature validation)

Scope (for now):
  • VASP DFT input prep only (LAMMPS support is planned follow-up).
  • Local prep only — you run VASP elsewhere and bring back outputs for analysis.
        """
    )

    # Model / API
    parser.add_argument('--model', type=str, default='claude-opus-4-6',
                        help='Model name (default: claude-opus-4-6)')
    parser.add_argument('--base-url', type=str, dest='base_url',
                        help='Base URL for OpenAI-compatible endpoint')
    parser.add_argument('--api-key', type=str, dest='api_key',
                        help='API key for LLM provider (overrides environment variables)')
    parser.add_argument('--mp-api-key', type=str, dest='mp_api_key',
                        help='Materials Project API key (or set MP_API_KEY env var). '
                             'Enables the MP tool-resolver in structure generation.')
    parser.add_argument('--futurehouse-api-key', type=str, dest='futurehouse_api_key',
                        help='FutureHouse API key (or set FUTUREHOUSE_API_KEY env var). '
                             'Enables INCAR literature validation.')

    # Mode
    parser.add_argument('--mode', type=str, dest='mode',
                        choices=['co-pilot', 'supervised', 'autonomous'],
                        default='co-pilot',
                        help='Autonomy mode (default: co-pilot)')

    # Initial request — seeds the first chat turn
    parser.add_argument('--request', type=str, dest='initial_request',
                        help='Optional initial request to seed the session '
                             '(e.g., "Build a rutile TiO2 supercell"). '
                             'If omitted, starts at an empty chat prompt.')

    # Custom tools and skills
    parser.add_argument('--tools', type=str, nargs='+', dest='tool_files', metavar='TOOL_FILE',
                        help='Path(s) to Python files containing domain-specific tool '
                             'functions to expose to the orchestrator. Each file must define '
                             'a tool_schemas list and a create_tool_functions factory.')
    parser.add_argument('--skills', type=str, nargs='+', dest='skill_files', metavar='SKILL_FILE',
                        help='Path(s) to custom skill .md files.')

    # Session
    parser.add_argument('--session-dir', type=str, dest='session_dir',
                        help='Session directory for outputs (default: auto-generated)')
    parser.add_argument('--restore', action='store_true',
                        help='Restore from previous checkpoint in session directory')

    args = parser.parse_args()

    # Auto-discover keys from env where possible
    api_key = args.api_key
    if args.mp_api_key:
        import scilink
        scilink.set_api_key('materials_project', args.mp_api_key)

    config = {
        'model_name': args.model,
        'base_url': args.base_url,
        'api_key': api_key,
        'simulation_mode': args.mode,
        'mp_api_key': args.mp_api_key,
        'futurehouse_api_key': args.futurehouse_api_key,
        'initial_request': args.initial_request,
        'session_dir': args.session_dir,
        'restore': args.restore,
        'tool_files': args.tool_files or [],
        'skill_files': args.skill_files or [],
    }

    try:
        playground = SimulatePlayground(config)
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

class SimulatePlayground:
    """Interactive session manager for the Simulation Orchestrator Agent."""

    def __init__(self, config: dict = None):
        self.agent = None
        self.session_dir = None
        self.config = config or {}

    def _infer_provider(self, model_name: str) -> tuple:
        """Infer provider info from model name."""
        m = model_name.lower()
        if 'claude' in m:
            return ("Anthropic", "ANTHROPIC_API_KEY or CLAUDE_API_KEY",
                    ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"])
        if m.startswith(('gpt-', 'o1-', 'o3-', 'text-embedding')):
            return ("OpenAI", "OPENAI_API_KEY", ["OPENAI_API_KEY"])
        return ("Google Gemini", "GEMINI_API_KEY or GOOGLE_API_KEY",
                ["GEMINI_API_KEY", "GOOGLE_API_KEY"])

    def _api_key_from_env(self, env_vars: list) -> str:
        for var in env_vars:
            v = os.getenv(var)
            if v:
                return v
        return None

    def setup(self):
        """Resolve config, build the agent, register custom tools/skills."""
        from scilink.agents.sim_agents.simulation_orchestrator import (
            SimulationOrchestratorAgent, SimulationMode,
        )

        model_name = self.config.get('model_name', 'claude-opus-4-6')
        base_url = self.config.get('base_url')
        api_key = self.config.get('api_key')
        mode_str = self.config.get('simulation_mode', 'co-pilot')
        mp_api_key = self.config.get('mp_api_key')
        futurehouse_api_key = self.config.get('futurehouse_api_key')
        session_dir = self.config.get('session_dir')
        restore = self.config.get('restore', False)

        self._tool_files = self.config.get('tool_files', [])
        self._skill_files = self.config.get('skill_files', [])
        self._initial_request = self.config.get('initial_request')

        mode_map = {
            'co-pilot': SimulationMode.CO_PILOT,
            'supervised': SimulationMode.SUPERVISED,
            'autonomous': SimulationMode.AUTONOMOUS,
        }
        simulation_mode = mode_map.get(mode_str, SimulationMode.CO_PILOT)

        print("\n" + "=" * 60)
        print("⚛️  SCILINK SIMULATION ORCHESTRATOR (VASP DFT prep)")
        print("=" * 60)
        print("""
This agent helps you prepare VASP DFT calculations by:
  1. Building atomic structures from natural-language descriptions
  2. Validating structures and refining them based on issues found
  3. Generating VASP input files (INCAR, KPOINTS) for your objective
  4. Analyzing VASP output files (you bring them back from your cluster)

Local prep only — you run VASP elsewhere. HPC submission tools are
planned follow-up work; see CLAUDE.md.
""")
        print("=" * 60)

        # ── API key resolution ───────────────────────────────────────
        if not api_key:
            if base_url:
                api_key = os.getenv("SCILINK_API_KEY")
                if not api_key:
                    print("\n⚠️  No SCILINK_API_KEY found in environment.")
                    api_key = input("Enter your proxy API key (SCILINK_API_KEY): ").strip()
                    if not api_key:
                        print("❌ Cannot proceed without API key for internal proxy.")
                        sys.exit(1)
            else:
                provider_name, env_var_hint, env_vars = self._infer_provider(model_name)
                api_key = self._api_key_from_env(env_vars)
                if not api_key:
                    print(f"\n⚠️  No {env_var_hint} found in environment.")
                    user_key = input(f"Enter your {provider_name} API key (or Enter to skip): ").strip()
                    if user_key:
                        api_key = user_key

        # MP key (auto-discovered by the orchestrator from MP_API_KEY env if not passed)
        if not mp_api_key:
            mp_api_key = os.getenv("MP_API_KEY") or os.getenv("MATERIALS_PROJECT_API_KEY")
        # FutureHouse key (auto-discovered similarly)
        if not futurehouse_api_key:
            futurehouse_api_key = os.getenv("FUTUREHOUSE_API_KEY")

        # ── Session directory ────────────────────────────────────────
        if session_dir:
            self.session_dir = Path(session_dir)
        elif restore:
            import glob
            sessions = sorted(glob.glob("./simulate_session_*"), reverse=True)
            if sessions:
                self.session_dir = Path(sessions[0])
                print(f"\n📂 Found session to restore: {self.session_dir}")
            else:
                print("\n⚠️  No existing simulate sessions found. Creating new session.")
                self.session_dir = Path(f"./simulate_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            default_dir = f"./simulate_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"\n📁 Where should I save session data?")
            session_input = input(f"   Path (default: {default_dir}): ").strip()
            self.session_dir = Path(session_input) if session_input else Path(default_dir)

        self.session_dir.mkdir(parents=True, exist_ok=True)

        # ── Initialize agent ────────────────────────────────────────
        print("\n🔧 Initializing agent...")
        try:
            self.agent = SimulationOrchestratorAgent(
                base_dir=str(self.session_dir),
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                simulation_mode=simulation_mode,
                restore_checkpoint=restore,
                mp_api_key=mp_api_key,
                futurehouse_api_key=futurehouse_api_key,
            )
            print("✅ Agent ready!")
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # ── Custom tools / skills ───────────────────────────────────
        if self._tool_files:
            self._load_custom_tools(self._tool_files)
        if self._skill_files:
            self._register_custom_skills(self._skill_files)

        # ── Session info ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SESSION INFO")
        print("=" * 60)
        print(f"Session Directory: {self.session_dir}")
        print(f"Simulation Mode: {simulation_mode.value}")
        print(f"Human Feedback: {'Enabled' if self.agent.get_human_feedback_setting() else 'Disabled'}")
        print(f"Materials Project: {'Connected' if mp_api_key else 'Not configured'}")
        print(f"INCAR literature validation: {'Enabled' if futurehouse_api_key else 'Disabled'}")

        provider_name, _, _ = self._infer_provider(model_name)
        if base_url:
            print(f"Model: {model_name}")
            print(f"Endpoint: {base_url}")
        else:
            print(f"Model: {model_name} ({provider_name})")

        print(f"\nAvailable Tools: {len(self.agent.tools.functions_map)}")
        tool_names = list(self.agent.tools.functions_map.keys())
        # Print up to 12 tools across 2 lines
        chunks = [tool_names[i:i+6] for i in range(0, len(tool_names), 6)]
        for ch in chunks:
            print(f"  {', '.join(ch)}")

    def _load_custom_tools(self, tool_files: list) -> None:
        """Load external tool functions from .py files (mirrors cli/analyze.py)."""
        import importlib.util
        import inspect
        import sys as _sys

        for file_path in tool_files:
            path = Path(file_path).resolve()
            print(f"\n🔧 Loading custom tools from: {path}")
            try:
                spec = importlib.util.spec_from_file_location(path.stem, path)
                module = importlib.util.module_from_spec(spec)
                _prev = _sys.dont_write_bytecode
                _sys.dont_write_bytecode = True
                try:
                    spec.loader.exec_module(module)
                finally:
                    _sys.dont_write_bytecode = _prev
            except Exception as e:
                print(f"   ❌ Failed to load {path.name}: {e}")
                continue

            schemas = (
                getattr(module, 'tool_schemas', None)
                or getattr(module, 'openai_schemas', None)
            )
            if schemas is None:
                for attr_name in dir(module):
                    obj = getattr(module, attr_name, None)
                    if (isinstance(obj, list) and obj
                            and isinstance(obj[0], dict)
                            and obj[0].get('type') == 'function'):
                        schemas = obj
                        break
            if not schemas:
                print(f"   ⚠️  No tool schemas found in {path.name}")
                continue

            factory = getattr(module, 'create_tool_functions', None)
            if factory is None:
                for name, fn in inspect.getmembers(module, inspect.isfunction):
                    if name.endswith('_tool_functions') and fn.__module__ == module.__name__:
                        factory = fn
                        break
            if factory is None:
                print(f"   ⚠️  No factory function found in {path.name}")
                continue

            self.agent.register_tools(schemas, factory)
            count = sum(1 for s in schemas if s.get('type') == 'function')
            print(f"   ✅ Registered {count} tool(s) from {path.name}")

    def _register_custom_skills(self, skill_files: list) -> None:
        for file_path in skill_files:
            path = Path(file_path).resolve()
            print(f"\n📖 Registering custom skill: {path}")
            try:
                name = self.agent.register_skill(str(path))
                print(f"   ✅ Registered skill '{name}'")
            except Exception as e:
                print(f"   ❌ Failed to register {path.name}: {e}")

    def print_help(self):
        print("\n" + "=" * 60)
        print("AVAILABLE COMMANDS")
        print("=" * 60)
        print("  /help              Show this help message")
        print("  /tools             List available tools")
        print("  /structures        List structures generated this session")
        print("  /status            Show current session state")
        print("  /mode [level]      Show or change simulation mode")
        print("  /clear             Clear screen")
        print("  /quit or /exit     Exit")
        print("\nOr just chat naturally with the agent!")
        print("=" * 60)

    def handle_command(self, user_input: str):
        """Returns True if handled, 'QUIT' to exit, False otherwise."""
        cmd = user_input.lower().strip()

        if cmd == "/help":
            self.print_help()
            return True

        if cmd == "/tools":
            print("\n📦 Available Tools:")
            for i, tool_name in enumerate(self.agent.tools.functions_map.keys(), 1):
                print(f"  {i}. {tool_name}")
            return True

        if cmd == "/structures":
            structures = self.agent.generated_structures or []
            print(f"\n🏗️  Structures generated this session: {len(structures)}")
            for s in structures:
                print(f"  • {s.get('slug')}: {s.get('description')}")
                print(f"      POSCAR: {s.get('poscar_path')}")
                if s.get('incar_path'):
                    print(f"      INCAR/KPOINTS: ✓")
            return True

        if cmd == "/status":
            print("\n🔍 Session State:")
            print(f"  Session Directory: {self.session_dir}")
            print(f"  Simulation Mode: {self.agent.simulation_mode.value}")
            print(f"  Human Feedback: {'Enabled' if self.agent.get_human_feedback_setting() else 'Disabled'}")
            print(f"  Message Count: {self.agent.message_count}")
            print(f"  Structures: {len(self.agent.generated_structures or [])}")
            print(f"  Default calc params: {self.agent.default_calc_params or 'none'}")
            return True

        if cmd.startswith("/mode"):
            from scilink.agents.sim_agents.simulation_orchestrator import SimulationMode
            parts = cmd.split()
            if len(parts) == 1:
                print(f"\n🎛️  Current Simulation Mode: {self.agent.simulation_mode.value}")
                print("\n   To change: /mode <co-pilot|supervised|autonomous>")
            else:
                mode_map = {
                    'co-pilot': SimulationMode.CO_PILOT,
                    'supervised': SimulationMode.SUPERVISED,
                    'autonomous': SimulationMode.AUTONOMOUS,
                }
                new_mode = mode_map.get(parts[1].lower())
                if new_mode:
                    self.agent.set_simulation_mode(new_mode)
                    print(f"\n   ✅ Simulation mode changed to: {new_mode.value}")
                else:
                    print(f"\n   ❌ Unknown mode: {parts[1]}")
                    print("   Valid options: co-pilot, supervised, autonomous")
            return True

        if cmd == "/clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("⚛️  Simulation Orchestrator - Session Resumed\n")
            return True

        if cmd in ("/quit", "/exit", "/q", "quit", "exit"):
            return "QUIT"

        return False

    def run(self):
        """Main interactive loop."""
        self.setup()
        self.print_help()

        print("\n" + "=" * 60)
        print("💬 CHAT SESSION STARTED")
        print("=" * 60)
        print("Type /help for commands, or just chat naturally!\n")

        # Process initial request if provided
        if self._initial_request:
            print(f"📝 Initial request: {self._initial_request}\n")
            try:
                response = self.agent.chat(self._initial_request)
                print(f"\n🤖 Agent: {response}\n")
            except Exception as e:
                print(f"❌ Error processing initial request: {e}")

        # Main chat loop
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

            if not user_input:
                continue

            handled = self.handle_command(user_input)
            if handled == "QUIT":
                print("\n👋 Goodbye!")
                break
            if handled is True:
                continue

            try:
                response = self.agent.chat(user_input)
                print(f"\n🤖 Agent: {response}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    sys.exit(main())
