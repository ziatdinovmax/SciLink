#!/usr/bin/env python3
"""
scilink analyze — the analysis orchestrator in the terminal: examine
experimental data (microscopy, spectroscopy, hyperspectral cubes, curves),
manage metadata, pick and run the right analysis agent, assess novelty.

The interactive loop is the shared terminal shell (``scilink.cli.shell``);
this module keeps the historical entry point.
"""

import sys


def main(argv=None):
    """Entry point for ``scilink analyze``."""
    from scilink.cli.shell import main as shell_main
    return shell_main(["analyze", *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    sys.exit(main())
