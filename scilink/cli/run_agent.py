"""
CLI entry point for running a single SciLink experimental agent.

This script allows for direct execution of an analysis agent on experimental
data to generate a detailed analysis and a set of scientific claims

Usage:
    scilink-agent [PATH] --agent [AGENT_NAME] [OPTIONS]

Example:
    scilink-agent my_data/ --agent general-microscopy --fft-nmf
    scilink-agent image.tif --system-info metadata.json --agent atomistic
    scilink-agent curve_data.csv --agent curve
"""

import argparse
import sys
import os
import logging
import json
from pathlib import Path
import textwrap
import inspect
from typing import Dict, Any, Tuple, Optional, Type

# --- SciLink Component Imports ---
try:
    # Import agent classes directly
    from ..agents.exp_agents import (
        MicroscopyAnalysisAgent,
        SAMMicroscopyAnalysisAgent,
        AtomisticMicroscopyAnalysisAgent,
        HyperspectralAnalysisAgent,
        CurveFittingAgent,
        HolisticMicroscopyAgent
    )
    from .agents import add_agent_args
except ImportError:
    # Fallback for direct script execution during development
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from agents.exp_agents import (
        MicroscopyAnalysisAgent,
        SAMMicroscopyAnalysisAgent,
        AtomisticMicroscopyAnalysisAgent,
        HyperspectralAnalysisAgent,
        CurveFittingAgent,
        HolisticMicroscopyAgent
    )
    from cli.agents import add_agent_args


# --- CLI Utility Functions ---

# Map user-friendly names to the actual agent classes
AGENT_NAME_TO_CLASS: Dict[str, Type] = {
    'general-microscopy': MicroscopyAnalysisAgent,
    'sam': SAMMicroscopyAnalysisAgent,
    'atomistic': AtomisticMicroscopyAnalysisAgent,
    'hyperspectral': HyperspectralAnalysisAgent,
    'curve': CurveFittingAgent,
    'holistic-microscopy': HolisticMicroscopyAgent
}

# Map agent names to their expected data file extensions
AGENT_DATA_REQUIREMENTS: Dict[str, Tuple[str, ...]] = {
    'general-microscopy': ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.npy', '.npz'),
    'sam': ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.npy', '.npz'),
    'atomistic': ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.npy', '.npz'),
    'hyperspectral': ('.npy', '.npz'),
    'curve': ('.txt', '.csv', '.dat', '.npy', '.npz'),
    'holistic-microscopy': ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.npy', '.npz')
}


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    NC = '\033[0m'  # No Color

def print_info(msg: str): print(f"{Colors.BLUE}ℹ️  {msg}{Colors.NC}")
def print_success(msg: str): print(f"{Colors.GREEN}✅ {msg}{Colors.NC}")
def print_warning(msg: str): print(f"{Colors.YELLOW}⚠️  {msg}{Colors.NC}")
def print_error(msg: str): print(f"{Colors.RED}❌ {msg}{Colors.NC}", file=sys.stderr)
def print_header(msg: str): print(f"{Colors.PURPLE}{msg}{Colors.NC}")

def discover_files(directory: Path) -> Tuple[Optional[str], Optional[str]]:
    """Discovers the main data and metadata files in a directory."""
    data_file, metadata_file = None, None
    data_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.npy', '.npz', '.txt', '.csv', '.dat']
    
    for ext in data_exts:
        if (candidate := directory / f"data{ext}").exists():
            data_file = str(candidate)
            break
    
    if not data_file:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in data_exts:
                data_file = str(p)
                break

    if data_file:
        if (candidate := Path(data_file).with_suffix('.json')).exists():
            metadata_file = str(candidate)
        elif (candidate := directory / "metadata.json").exists():
            metadata_file = str(candidate)
            
    return data_file, metadata_file

def validate_json_file(file_path: str) -> bool:
    """Validates that a file exists and contains valid JSON."""
    if not os.path.exists(file_path): return False
    try:
        with open(file_path, 'r') as f: json.load(f)
        return True
    except (json.JSONDecodeError, IOError):
        return False

def display_analysis_results(result: Dict[str, Any]):
    """Prints the analysis results in a readable format."""
    print_header("\n--- Agent Analysis Results ---")
    
    if "error" in result or result.get("status") == "error":
        error_msg = result.get("error") or result.get("message", "Unknown error")
        details = result.get("details")
        full_error = f"{error_msg}"
        if details:
            indented_details = textwrap.indent(str(details), '  ')
            full_error += f"\nDetails:\n{indented_details}"
        print_error(f"Analysis failed: {full_error}")
        return

    detailed_analysis = result.get("detailed_analysis", "No detailed analysis was provided.")
    claims = result.get("scientific_claims", [])
    
    print("\n📋 DETAILED ANALYSIS:")
    print(textwrap.indent(detailed_analysis, '  '))
    
    if 'fitting_parameters' in result:
        print("\n⚙️ FITTED PARAMETERS:")
        params_str = json.dumps(result['fitting_parameters'], indent=2)
        print(textwrap.indent(params_str, '  '))
    
    print(f"\n🎯 GENERATED CLAIMS ({len(claims)}):")
    if not claims:
        print("  No scientific claims were generated.")
    else:
        for i, claim_data in enumerate(claims, 1):
            claim = claim_data.get('claim', 'N/A')
            question = claim_data.get('has_anyone_question', 'N/A')
            print(f"\n  [{i}] Claim: {claim}")
            print(f"      Research Question: {question}")
    
    print_success("\nAnalysis complete.")


def create_agent_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the agent runner CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a SciLink Experimental Agent directly on data.\n"
            "The --agent flag is REQUIRED to specify which agent to run."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('path', help='Path to the experimental data file or directory.')
    parser.add_argument('--system-info', help='Path to the system information JSON file.')
    parser.add_argument('--output-dir', default='agent_output', help='Directory to save agent outputs (default: %(default)s).')
    parser.add_argument('--model', default='gemini-2.5-pro-preview-06-05', help='Generative model to use (default: %(default)s).')

    parser.add_argument(
        '--local-model',
        type=str,
        default=None,
        help='URL for a local OpenAI-compatible API endpoint (e.g., "http://host.docker.internal:8000/v1").'
    )
        
    parser.add_argument(
        '--fft-nmf',
        action='store_true',
        help="Enable Sliding FFT-NMF analysis for the 'general' microscopy agent."
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging to see detailed, real-time agent output."
    )

    add_agent_args(parser, required=True)
    
    return parser


def main():
    """Main entry point for the scilink-agent CLI."""
    parser = create_agent_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(name)s: %(message)s',
            force=True  # Override any existing default logger configuration
        )
    
    print_header("🔬 SciLink Experimental Agent Runner")
    print_header("=================================")

    input_path = Path(args.path)
    if not input_path.exists():
        print_error(f"Input path does not exist: {args.path}")
        sys.exit(1)

    # Discover and validate files
    if input_path.is_dir():
        data_file, metadata_file = discover_files(input_path)
        if not data_file:
            print_error(f"Could not find a valid data file in directory: {input_path}")
            sys.exit(1)
    else:
        data_file = str(input_path)
        metadata_file = args.system_info

    if args.system_info:
        metadata_file = args.system_info

    if not os.path.exists(data_file):
        print_error(f"Data file not found: {data_file}")
        sys.exit(1)
    if metadata_file and not validate_json_file(metadata_file):
        print_error(f"Metadata file is invalid or not found: {metadata_file}")
        sys.exit(1)

    # Directly use the selected agent from the required --agent flag
    agent_name_to_run = args.agent
    agent_class = AGENT_NAME_TO_CLASS.get(agent_name_to_run)

    if not agent_class:
        print_error(f"Invalid agent name specified: '{agent_name_to_run}'.")
        sys.exit(1)

    # Validate file type against the selected agent
    file_extension = Path(data_file).suffix.lower()
    allowed_extensions = AGENT_DATA_REQUIREMENTS.get(agent_name_to_run, ())
    if file_extension not in allowed_extensions:
        print_error(f"File type '{file_extension}' is not compatible with the '{agent_name_to_run}' agent.")
        print_info(f"Allowed file types for this agent are: {', '.join(allowed_extensions)}")
        sys.exit(1)
        
    print_info(f"Selected Agent: {agent_class.__name__}")

    # Instantiate and run the agent
    try:
        # Build arguments dynamically based on the agent's constructor signature
        agent_kwargs = {
            'model_name': args.model,
            'local_model': args.local_model,
            'enable_human_feedback': False
        }
        
        # Conditionally add fft_nmf_settings for the correct agent
        if agent_class is MicroscopyAnalysisAgent and args.fft_nmf:
            print_info("Sliding FFT-NMF analysis is enabled.")
            agent_kwargs['fft_nmf_settings'] = {
                'FFT_NMF_ENABLED': True,
                'FFT_NMF_AUTO_PARAMS': True,
                'output_dir': args.output_dir
            }
        
        sig = inspect.signature(agent_class.__init__)
        if 'output_dir' in sig.parameters:
            agent_kwargs['output_dir'] = args.output_dir

        agent_instance = agent_class(**agent_kwargs)
        
        print_info(f"Running {agent_class.__name__}...")
        
        result = agent_instance.analyze_for_claims(data_path=data_file, system_info=metadata_file)

        display_analysis_results(result)

    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
