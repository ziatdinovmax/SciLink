#!/usr/bin/env python3
"""
scilink simulate — the simulation orchestrator in the terminal: build
atomic structures, generate VASP inputs, analyze the outputs you bring back.

The interactive loop is the shared terminal shell (``scilink.cli.shell``);
this module keeps the historical entry point.
"""

import sys


def main(argv=None):
    """Entry point for ``scilink simulate``."""
    from scilink.cli.shell import main as shell_main
    return shell_main(["simulate", *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    sys.exit(main())
