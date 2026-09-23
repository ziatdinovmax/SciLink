#!/usr/bin/env python3
"""
scilink plan — the planning orchestrator in the terminal: design
experimental campaigns and optimization loops grounded in your papers,
code and data.

The interactive loop is the shared terminal shell (``scilink.cli.shell``);
this module keeps the historical entry point.
"""

import sys


def main(argv=None):
    """Entry point for ``scilink plan``."""
    from scilink.cli.shell import main as shell_main
    return shell_main(["plan", *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    sys.exit(main())
