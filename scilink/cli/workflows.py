#!/usr/bin/env python3
"""
SciLink CLI - Extensible Command Pattern

Usage: scilink [WORKFLOW] [PATH/DESCRIPTION] [OPTIONS]

Examples:
    scilink novelty my_data/
    scilink novelty my_data/ --dft-recommendations
    scilink novelty image.tif --system-info metadata.json
    scilink dft "Generate graphene supercell with a single vacancy"
    scilink experiment2dft my_experiment/
    scilink experiment2dft my_experiment/ --non-interactive --auto-select 3
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Relative imports for scilink package
try:
    from ..workflows.experiment_novelty_workflow import ExperimentNoveltyAssessment
    from ..workflows.experiment2dft import Experimental2DFT
except ImportError:
    # Fallback for direct execution during development
    sys.path.append(str(Path(__file__).parent.parent))
    from workflows.experiment_novelty_workflow import ExperimentNoveltyAssessment
    from workflows.experiment2dft import Experimental2DFT


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.NC}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.NC}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.NC}")


def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.NC}", file=sys.stderr)


def print_header(msg: str):
    print(f"{Colors.PURPLE}{msg}{Colors.NC}")


def detect_data_type(file_path: str) -> str:
    """Auto-detect data type from file extension and numpy array dimensions"""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
        return 'microscopy'
    elif ext in ['.npy', '.npz']:
        # Check numpy array dimensions
        try:
            arr = np.load(file_path)
            if arr.ndim == 2:
                # 2D array -> likely microscopy image
                return 'microscopy'
            elif arr.ndim == 3:
                # 3D array -> likely hyperspectral data
                return 'spectroscopy'
            else:
                return 'unknown'
        except Exception:
            return 'unknown'
    elif ext in ['.txt', '.csv', '.dat']:
        return 'spectroscopy'
    else:
        return 'unknown'


def get_array_info(file_path: str) -> str:
    """Get shape and dimension info for numpy arrays"""
    try:
        arr = np.load(file_path)
        return f"Shape: {arr.shape}, Dimensions: {arr.ndim}D"
    except Exception:
        return "Could not read array info"


def discover_files(directory: Path) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Auto-discover files in directory"""
    data_file = None
    metadata_file = None
    structure_file = None
    structure_info_file = None
    
    # Look for main data file
    data_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.npy', '.npz', '.txt', '.csv', '.dat']
    
    # First try data.* files
    for ext in data_extensions:
        candidate = directory / f"data{ext}"
        if candidate.exists():
            data_file = str(candidate)
            break
    
    # If no data.* file, look for any data file
    if not data_file:
        for file_path in directory.iterdir():
            if file_path.is_file():
                basename = file_path.name.lower()
                ext = file_path.suffix.lower()
                if (ext in data_extensions and 
                    not basename.startswith('structure') and 
                    not basename.startswith('metadata')):
                    data_file = str(file_path)
                    break
    
    # Look for metadata - first try same basename as data file, then generic names
    if data_file:
        # Try same basename as data file
        data_path = Path(data_file)
        same_basename_json = data_path.with_suffix('.json')
        if same_basename_json.exists():
            metadata_file = str(same_basename_json)
    
    # If no same-basename metadata found, try generic names
    if not metadata_file:
        metadata_names = ['metadata.json', 'system_info.json', 'meta.json', 'info.json']
        for name in metadata_names:
            candidate = directory / name
            if candidate.exists():
                metadata_file = str(candidate)
                break
    
    # Look for structure files
    structure_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    for ext in structure_extensions:
        candidate = directory / f"structure{ext}"
        if candidate.exists():
            structure_file = str(candidate)
            break
    
    # Look for structure info
    structure_info_candidates = ['structure_info.json', 'structure.json']
    for name in structure_info_candidates:
        candidate = directory / name
        if candidate.exists():
            structure_info_file = str(candidate)
            break
    
    return data_file, metadata_file, structure_file, structure_info_file


def validate_json_file(file_path: str) -> bool:
    """Validate that file exists and contains valid JSON"""
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description="SciLink - Intelligent Materials Research CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Experimental novelty assessment
  scilink novelty my_data/
  scilink novelty image.tif --system-info metadata.json
  scilink novelty my_data/ --dft-recommendations
  
  # Complete experimental to DFT pipeline (INTERACTIVE BY DEFAULT)
  scilink experiment2dft my_data/                              # Interactive mode (default)
  scilink experiment2dft my_data/ --non-interactive            # Automated mode
  scilink experiment2dft image.tif --auto-select 3            # Still interactive, but sets auto-select for non-interactive
  scilink experiment2dft spectral_data.npy --non-interactive --auto-select 1
  
  # DFT structure generation
  scilink dft "Generate a graphene supercell with Stone-Wales defects"
  scilink dft "Create MoS2 monolayer with sulfur vacancies" --output-dir dft_results/

Commands:
  novelty                Experimental novelty assessment with literature search
  experiment2dft         Complete pipeline: experiment → novelty → DFT structures
  dft                    Generate DFT structures from text description

Directory Structure (for novelty/experiment2dft commands):
  experiment_data/
  ├── data.{jpg,tif,npy,txt,csv}     # Main experimental data
  ├── metadata.json                  # System information
  ├── structure.{jpg,tif,png}        # Optional: structure image
  └── structure_info.json            # Optional: structure metadata
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'command',
        choices=['novelty', 'experiment2dft', 'dft', 'full_pipeline'],
        help='Workflow to run'
    )
    parser.add_argument(
        'path',
        help='Path to experimental data file/directory (for novelty/experiment2dft) or text description (for dft)'
    )
    
    # Data type options
    parser.add_argument(
        '--microscopy',
        action='store_true',
        help='Force microscopy analysis'
    )
    parser.add_argument(
        '--spectroscopy', 
        action='store_true',
        help='Force spectroscopy analysis'
    )
    
    # Analysis configuration
    parser.add_argument(
        '--agent-id',
        type=int,
        choices=[0, 1, 2],
        help='Microscopy agent (0=General, 1=SAM, 2=Atomistic)'
    )
    parser.add_argument(
        '--model',
        default='gemini-2.5-pro-preview-06-05',
        help='Analysis model (default: %(default)s)'
    )

    parser.add_argument(
        '--local-model',
        type=str,
        default=None,
        help='URL for a local OpenAI-compatible API endpoint (e.g., "http://host.docker.internal:8000/v1").'
    )
    
    parser.add_argument(
        '--no-spectral-unmixing',
        action='store_true',
        help='Disable spectral unmixing for spectroscopy'
    )
    
    # File specifications
    parser.add_argument(
        '--system-info',
        help='System information JSON file'
    )
    parser.add_argument(
        '--structure-image',
        help='Structure/morphology image'
    )
    parser.add_argument(
        '--structure-info',
        help='Structure metadata JSON'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        default='novelty_output',
        help='Output directory (default: input_dir/results when using directory input, otherwise %(default)s)'
    )
    parser.add_argument(
        '--dft-recommendations',
        action='store_true',
        help='Generate DFT structure recommendations (for novelty command)'
    )

    parser.add_argument(
        '--measurement-recommendations',
        action='store_true',
        help='Generate next experiment recommendations (for novelty command)'
    )
    
    # Structure generation options (for experiment2dft)
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive structure selection (DEFAULT for experiment2dft)'
    )
    parser.add_argument(
        '--auto-select',
        type=int,
        default=2,
        help='Number of top recommendations to auto-select in non-interactive mode (default: %(default)s)'
    )
    parser.add_argument(
        '--max-structures',
        type=int,
        default=5,
        help='Maximum number of structures to consider (default: %(default)s)'
    )
    parser.add_argument(
        '--max-refinement-cycles',
        type=int,
        default=2,
        help='Maximum structure refinement cycles (default: %(default)s)'
    )
    parser.add_argument(
        '--vasp-generator',
        choices=['llm', 'atomate2'],
        default='llm',
        help="Method for VASP input generation: 'llm' or 'atomate2' (default: %(default)s)"
    )
    parser.add_argument(
        '--script-timeout',
        type=int,
        default=300,
        help='Script execution timeout in seconds (default: %(default)s)'
    )
    
    # General options
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Skip interactive modes (claim selection, structure selection)'
    )
    parser.add_argument(
        '--wait-time',
        type=int,
        default=400,
        help='Max literature search wait time in seconds (default: %(default)s)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Detailed logging'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    return parser


def run_novelty_workflow(args, data_file: str, metadata_file: str, structure_file: str, structure_info_file: str, data_type: str):
    """Run experimental novelty assessment workflow"""
    
    if not args.quiet:
        print_header("🔬 Experimental Novelty Assessment")
        print("-" * 40)
    
    # Load system info
    system_info = None
    if metadata_file:
        try:
            with open(metadata_file, 'r') as f:
                system_info = json.load(f)
        except Exception as e:
            if not args.quiet:
                print_warning(f"Could not load system info: {e}")
    
    # Load structure info
    structure_system_info = None
    if structure_info_file:
        try:
            with open(structure_info_file, 'r') as f:
                structure_system_info = json.load(f)
        except Exception as e:
            if not args.quiet:
                print_warning(f"Could not load structure info: {e}")
    
    # Create workflow kwargs
    workflow_kwargs = {
        'data_type': data_type,
        'analysis_model': args.model,
        'output_dir': args.output_dir,
        'max_wait_time': args.wait_time,
        'dft_recommendations': args.dft_recommendations,
        'measurement_recommendations': args.measurement_recommendations 
    }
    
    # Add data-type specific parameters
    if data_type == 'microscopy' and args.agent_id is not None:
        workflow_kwargs['agent_id'] = args.agent_id
    elif data_type == 'spectroscopy':
        workflow_kwargs['spectral_unmixing_enabled'] = not args.no_spectral_unmixing
    
    # Create workflow
    workflow = ExperimentNoveltyAssessment(**workflow_kwargs)
    
    # Prepare analysis kwargs
    analysis_kwargs = {}
    if data_type == 'spectroscopy':
        if structure_file:
            analysis_kwargs['structure_image_path'] = structure_file
        if structure_system_info:
            analysis_kwargs['structure_system_info'] = structure_system_info
    
    # Run workflow (it prints its own comprehensive output)
    result = workflow.run_complete_workflow(
        data_path=data_file,
        system_info=system_info,
        **analysis_kwargs
    )
    
    return result


def run_experiment2dft_workflow(args, data_file: str, metadata_file: str, structure_file: str, structure_info_file: str, data_type: str):
    """Run complete experimental to DFT pipeline"""
    
    if not args.quiet:
        print_header("🧬 Complete Experimental → DFT Pipeline")
        print("-" * 45)
    
    # Load system info
    system_info = None
    if metadata_file:
        try:
            with open(metadata_file, 'r') as f:
                system_info = json.load(f)
        except Exception as e:
            if not args.quiet:
                print_warning(f"Could not load system info: {e}")
    
    # Load structure info
    structure_system_info = None
    if structure_info_file:
        try:
            with open(structure_info_file, 'r') as f:
                structure_system_info = json.load(f)
        except Exception as e:
            if not args.quiet:
                print_warning(f"Could not load structure info: {e}")
    
    # Create workflow kwargs
    workflow_kwargs = {
        'analysis_model': args.model,
        'generator_model': args.model,
        'validator_model': args.model,
        'local_model': args.local_model,
        'output_dir': args.output_dir,
        'max_wait_time': args.wait_time,
        'max_refinement_cycles': args.max_refinement_cycles,
        'script_timeout': args.script_timeout,
        'spectroscopy_analysis_enabled': not args.no_spectral_unmixing,
        'vasp_generator_method': args.vasp_generator
    }
    
    # Add data-type specific parameters
    if data_type == 'microscopy' and args.agent_id is not None:
        workflow_kwargs['agent_id'] = args.agent_id
    
    # Create workflow
    workflow = Experimental2DFT(**workflow_kwargs)
    
    # Determine interaction mode (interactive is default unless --non-interactive is specified)
    interactive = not args.non_interactive
    
    # Run the appropriate pipeline method based on data type
    if data_type == 'microscopy':
        result = workflow.run_microscopy_pipeline(
            image_path=data_file,
            system_info=system_info,
            interactive=interactive,
            auto_select_top_n=args.auto_select,
            max_structures=args.max_structures
        )
    elif data_type == 'spectroscopy':
        result = workflow.run_spectroscopy_pipeline(
            data_path=data_file,
            system_info=system_info,
            interactive=interactive,
            auto_select_top_n=args.auto_select,
            max_structures=args.max_structures,
            structure_image_path=structure_file,
            structure_system_info=structure_system_info
        )
    else:
        # Fallback to generic pipeline
        result = workflow.run_complete_pipeline(
            data_path=data_file,
            data_type=data_type,
            system_info=system_info,
            interactive=interactive,
            auto_select_top_n=args.auto_select,
            max_structures=args.max_structures
        )
    
    return result


def run_dft_workflow(args, user_request: str):
    """Run DFT structure generation workflow"""
    
    if not args.quiet:
        print_header("⚛️  DFT Structure Generation")
        print("-" * 30)
        print_info(f"Request: {user_request}")
        print()
    
    try:
        from ..workflows.dft_workflow import DFTWorkflow
    except ImportError:
        try:
            from workflows.dft_workflow import DFTWorkflow
        except ImportError:
            print_error("DFT workflow not available. Make sure dft_workflow.py exists.")
            return {"final_status": "error", "message": "DFT workflow not found"}
    
    # Create DFT workflow
    workflow = DFTWorkflow(
        generator_model=args.model,
        validator_model=args.model,
        local_model=args.local_model,
        output_dir=args.output_dir,
        max_refinement_cycles=getattr(args, 'max_refinement_cycles', 2),
        script_timeout=getattr(args, 'script_timeout', 300)
    )
    
    # Run complete workflow
    result = workflow.run_complete_workflow(user_request)
    
    return result


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print_header("🧬 SciLink - Intelligent Materials Research")
        print_header("=========================================")
    
    # Handle data type conflicts (only relevant for data-based commands)
    if args.command in ['novelty', 'experiment2dft'] and args.microscopy and args.spectroscopy:
        print_error("Cannot specify both --microscopy and --spectroscopy")
        sys.exit(1)
    
    # Determine data type from flags (only for data-based commands)
    forced_data_type = None
    if args.command in ['novelty', 'experiment2dft']:
        if args.microscopy:
            forced_data_type = 'microscopy'
        elif args.spectroscopy:
            forced_data_type = 'spectroscopy'
    
    # Handle DFT workflow (doesn't need file discovery)
    if args.command == 'dft':
        try:
            result = run_dft_workflow(args, args.path)
            final_status = result.get('final_status', 'unknown')
            if final_status == 'success':
                if not args.quiet:
                    print("\n🎉 DFT workflow completed successfully!")
                sys.exit(0)
            else:
                if not args.quiet:
                    print(f"\n⚠️  DFT workflow completed with status: {final_status}")
                sys.exit(1)
        except Exception as e:
            print_error(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # Validate path exists (for data-based workflows)
    input_path = Path(args.path)
    if not input_path.exists():
        print_error(f"Path does not exist: {input_path}")
        sys.exit(1)
    
    # Handle directory vs file input
    if input_path.is_dir():
        if not args.quiet:
            print_info(f"Auto-discovering files in directory: {input_path}")
        
        data_file, metadata_file, structure_file, structure_info_file = discover_files(input_path)
        
        # Override with explicitly provided files
        if args.system_info:
            metadata_file = args.system_info
        if args.structure_image:
            structure_file = args.structure_image
        if args.structure_info:
            structure_info_file = args.structure_info
        
        # Validate discovery
        if not data_file:
            print_error("No experimental data file found")
            print_info("Expected files: data.{jpg,tif,npy,txt,csv} or similar")
            sys.exit(1)
        
        if not args.quiet:
            print_success(f"Found data file: {Path(data_file).name}")
            if metadata_file:
                print_success(f"Found metadata: {Path(metadata_file).name}")
            if structure_file:
                print_success(f"Found structure image: {Path(structure_file).name}")
            if structure_info_file:
                print_success(f"Found structure info: {Path(structure_info_file).name}")
    
    elif input_path.is_file():
        data_file = str(input_path)
        metadata_file = args.system_info
        structure_file = args.structure_image
        structure_info_file = args.structure_info
        
        if not args.quiet:
            print_info(f"Using data file: {input_path.name}")
        
        if not metadata_file and not args.quiet:
            print_warning("No metadata file specified. Use --system-info for better analysis.")
    
    else:
        print_error(f"Input path is neither a file nor directory: {input_path}")
        sys.exit(1)
    
    # Validate files exist
    if not os.path.exists(data_file):
        print_error(f"Data file not found: {data_file}")
        sys.exit(1)
    
    if metadata_file and not validate_json_file(metadata_file):
        print_error(f"Invalid or missing metadata file: {metadata_file}")
        sys.exit(1)
    
    if structure_info_file and not validate_json_file(structure_info_file):
        print_error(f"Invalid structure info file: {structure_info_file}")
        sys.exit(1)
    
    # Auto-detect data type if not forced
    if forced_data_type:
        data_type = forced_data_type
        if not args.quiet:
            print_info(f"Using forced data type: {data_type}")
    else:
        if not args.quiet:
            print_info("Auto-detecting data type...")
        data_type = detect_data_type(data_file)
        
        if data_type == 'unknown':
            print_error("Cannot determine data type from file. Use --microscopy or --spectroscopy")
            sys.exit(1)
        
        # Show detection reasoning for numpy arrays
        if not args.quiet:
            if data_file.endswith(('.npy', '.npz')):
                array_info = get_array_info(data_file)
                print_info(f"Detected {data_type} from numpy array ({array_info})")
            else:
                print_info(f"Auto-detected data type: {data_type}")
    
    # Determine output directory
    if input_path.is_dir() and args.output_dir == 'novelty_output':
        # Use the input directory for output when default output dir is used
        output_dir = str(input_path / 'results')
        if not args.quiet:
            print_info(f"Using input directory for output: {output_dir}")
    else:
        # Use specified output dir or environment variable
        output_dir = os.environ.get('SCILINK_OUTPUT_DIR', args.output_dir)
    
    args.output_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration
    if not args.quiet:
        print_info("Configuration:")
        print(f"  Command: {args.command}")
        print(f"  Data Type: {data_type}")
        print(f"  Data Path: {data_file}")
        if metadata_file:
            print(f"  Metadata: {metadata_file}")
        if structure_file:
            print(f"  Structure Image: {structure_file}")
        if structure_info_file:
            print(f"  Structure Info: {structure_info_file}")
        print(f"  Output Dir: {output_dir}")
        if args.agent_id is not None:
            print(f"  Agent ID: {args.agent_id}")
        if args.command == 'novelty' and args.dft_recommendations:
            print(f"  DFT Recommendations: Enabled")
        if args.command == 'experiment2dft':
            print(f"  Interactive Mode: {not args.non_interactive} (default)")
            print(f"  Auto-select: {args.auto_select}")
            print(f"  Max Structures: {args.max_structures}")
        print("-" * 50)
        print()  # Extra space before workflow starts
    
    try:
        # Route to appropriate workflow
        if args.command == 'novelty':
            result = run_novelty_workflow(args, data_file, metadata_file, structure_file, structure_info_file, data_type)
        elif args.command in ['experiment2dft', 'full-pipeline']:
            result = run_experiment2dft_workflow(args, data_file, metadata_file, structure_file, structure_info_file, data_type)
        else:
            print_error(f"Unknown command: {args.command}")
            sys.exit(1)
        
        # Handle results
        final_status = result.get('final_status', 'unknown')
        if final_status == 'success':
            if not args.quiet:
                print("\n🎉 Workflow completed successfully!")
                
                # Show additional success info for experiment2dft
                if args.command == 'experiment2dft':
                    structures = result.get('generated_structures', [])
                    successful = [s for s in structures if s.get('success', False)]
                    if successful:
                        print_success(f"Generated {len(successful)} atomic structure(s)")
                        print_info("Ready-to-use VASP input files created:")
                        for i, struct in enumerate(successful, 1):
                            structure_name = struct.get('structure_name', f'structure_{i:02d}')
                            print(f"  • {structure_name}/: POSCAR, INCAR, KPOINTS")
            sys.exit(0)
        else:
            if not args.quiet:
                print(f"\n⚠️  Workflow completed with status: {final_status}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()