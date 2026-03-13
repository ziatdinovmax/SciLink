#!/usr/bin/env python3
"""
SciLink CLI - Main entry point
Routes to different agent types: plan, simulate, analyze
"""

import sys
import os


def get_terminal_color_support():
    """
    Detect terminal color support level.
    Returns: 'truecolor', '256', 'basic', or 'none'
    """
    # No color if not a TTY or NO_COLOR is set
    if not sys.stdout.isatty() or os.environ.get('NO_COLOR'):
        return 'none'
    
    # Check for true color support
    colorterm = os.environ.get('COLORTERM', '').lower()
    if colorterm in ('truecolor', '24bit'):
        return 'truecolor'
    
    term = os.environ.get('TERM', '').lower()
    term_program = os.environ.get('TERM_PROGRAM', '').lower()
    
    # Known true color terminals
    if term_program in ('iterm.app', 'vscode', 'hyper', 'alacritty', 'kitty', 'warp'):
        return 'truecolor'
    
    # 256 color support
    if '256color' in term:
        return '256'
    
    # Basic ANSI support
    if term in ('xterm', 'screen', 'linux', 'ansi'):
        return 'basic'
    
    # Default to basic if we have any TERM set
    if term:
        return 'basic'
    
    return 'none'


def rgb_to_256(r, g, b):
    """Convert RGB to closest 256-color palette index."""
    # Use the 6x6x6 color cube (indices 16-231)
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


def print_gradient_logo():
    """Prints the SciLink ASCII logo with appropriate color support."""
    logo_text = [
        "  ____       _ _     _       _    ",
        " / ___|  ___(_) |   (_)_ __ | | __",
        " \\___ \\ / __| | |   | | '_ \\| |/ /",
        "  ___) | (__| | |___| | | | |   < ",
        " |____/ \\___|_|_____|_|_| |_|_|\\_\\"
    ]

    color_support = get_terminal_color_support()
    
    # SciLink Colors: Blue (#4285F4) to Green (#34A853)
    start_rgb = (66, 133, 244)
    end_rgb = (52, 168, 83)

    term_width = 60
    logo_width = max(len(line) for line in logo_text)
    padding = " " * ((term_width - logo_width) // 2)

    for line in logo_text:
        if color_support == 'none':
            # No color - just print plain text
            print(padding + line)
        
        elif color_support == 'basic':
            # Basic ANSI - use cyan as a simple blue-green compromise
            print(f"{padding}\033[36m{line}\033[0m")
        
        elif color_support == '256':
            # 256 color - gradient with palette approximation
            colored_line = padding
            length = len(line)
            
            for i, char in enumerate(line):
                ratio = i / max(length - 1, 1)
                r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
                g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
                b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
                color_idx = rgb_to_256(r, g, b)
                colored_line += f"\033[38;5;{color_idx}m{char}"
            
            print(colored_line + "\033[0m")
        
        else:
            # True color - full RGB gradient
            colored_line = padding
            length = len(line)
            
            for i, char in enumerate(line):
                ratio = i / max(length - 1, 1)
                r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
                g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
                b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
                colored_line += f"\033[38;2;{r};{g};{b}m{char}"
            
            print(colored_line + "\033[0m")
    
    print()

def main():
    """Main CLI entry point with subcommands"""
    
    # Skip logo for MCP server mode (stdout is the transport)
    if len(sys.argv) >= 2 and sys.argv[1] == 'serve':
        from scilink.cli.serve import main as serve_main
        sys.argv = [sys.argv[0] + ' serve'] + sys.argv[2:]
        return serve_main()

    # Always show logo first
    print_gradient_logo()

    if len(sys.argv) < 2:
        print()  # Spacing after logo
        print_usage()
        return 1
    
    command = sys.argv[1]
    
    # Route to appropriate handler
    if command == 'plan':
        from scilink.cli.plan import main as plan_main
        sys.argv = [sys.argv[0] + ' plan'] + sys.argv[2:]
        return plan_main()
    
    elif command == 'simulate':
        from scilink.cli.simulate import main as simulate_main
        sys.argv = [sys.argv[0] + ' simulate'] + sys.argv[2:]
        return simulate_main()
    
    elif command == 'analyze':
        from scilink.cli.analyze import main as analyze_main
        sys.argv = [sys.argv[0] + ' analyze'] + sys.argv[2:]
        return analyze_main()

    elif command == 'ui':
        from scilink.cli.ui import main as ui_main
        sys.argv = [sys.argv[0] + ' ui'] + sys.argv[2:]
        return ui_main()

    elif command == 'serve':
        from scilink.cli.serve import main as serve_main
        sys.argv = [sys.argv[0] + ' serve'] + sys.argv[2:]
        return serve_main()

    elif command in ['-h', '--help', 'help']:
        print()  # Spacing after logo
        print_usage()
        return 0
    
    else:
        print()  # Spacing after logo
        print(f"❌ Unknown command: {command}")
        print_usage()
        return 1


def print_usage():
    """Print main CLI usage"""
    usage = """
╔══════════════════════════════════════════════════════════════════════════╗
║                              SciLink CLI                                 ║
║              AI-Powered Scientific Research Automation                   ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage: scilink <command> [options]

Available Commands:
  plan          Interactive planning orchestrator for experimental design
                and Bayesian optimization workflows
                
  simulate      Simulation agents for MD, DFT, LAMMPS, VASP workflows
                (Coming soon)
                
  analyze       Analysis agents for microscopy, spectroscopy, and
                experimental data processing
                (Coming soon)

  ui            Launch the Streamlit web interface for interactive
                analysis (requires: pip install scilink[ui])

  serve         Start SciLink as an MCP tool server so external clients
                (Claude Desktop, Cursor) can use SciLink's tools

Examples:
  scilink plan                              # Start planning orchestrator
  scilink plan --model gemini-2.0-flash-exp # Use different model
  scilink simulate --help                   # See simulation options
  scilink analyze --help                    # See analysis options

Get Help:
  scilink <command> --help                  # Command-specific help
  scilink help                              # This message

For more information, visit: https://github.com/your-org/scilink
    """
    print(usage)


if __name__ == '__main__':
    sys.exit(main())