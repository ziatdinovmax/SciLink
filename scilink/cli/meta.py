#!/usr/bin/env python3
"""
scilink explore — Mission Control, SciLink's default entry point (bare
``scilink``): one chat surface that routes each request to the right
specialist (analysis, planning, simulation) and bridges findings between
them. Each delegation runs as a nested child session under the meta session.

The interactive loop is the shared terminal shell (``scilink.cli.shell``);
this module keeps the historical entry point.
"""

import sys


def main(argv=None):
    """Entry point for ``scilink explore`` (and bare ``scilink``)."""
    from scilink.cli.shell import main as shell_main
    return shell_main(["meta", *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    sys.exit(main())
