"""Entry point: ``main(["meta" | "analyze" | "plan" | "simulate", *argv])``.

The mode's own flags (kept from the four old CLIs) plus the shell's global
ones: ``-p/--print`` for headless mode, ``--output-format``, ``--yes`` for
code-execution consent, ``--resume [ID]``, ``--verbose``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .headless import OUTPUT_FORMATS
from .modes import ADAPTERS


def build_parser(adapter) -> argparse.ArgumentParser:
    # The dispatcher (scilink.cli.main) rewrites argv[0] to "scilink <cmd>";
    # a direct module run falls back to the launcher name.
    argv0 = None
    if " " in sys.argv[0]:
        first, _, rest = sys.argv[0].partition(" ")
        argv0 = f"{os.path.basename(first)} {rest}"
    launcher = os.environ.get("SCILINK_ARGV0", "scilink")
    prog = argv0 or (launcher if adapter.key == "meta" else f"{launcher} {adapter.key}")
    p = argparse.ArgumentParser(
        prog=prog, description=adapter.description,
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=adapter.epilog)
    g = p.add_argument_group("shell")
    g.add_argument("-p", "--print", dest="print_task", metavar="TASK",
                   help="Run one task without the interactive prompt and print the result "
                        "(autonomous; every question is auto-accepted)")
    g.add_argument("--output-format", dest="output_format", choices=OUTPUT_FORMATS,
                   default="text", help="Headless output format (default: text)")
    g.add_argument("--yes", "-y", action="store_true",
                   help="Pre-approve generated-code execution without asking")
    g.add_argument("--resume", dest="resume", nargs="?", const=True, default=None,
                   metavar="ID", help="Resume a past session: an id, or none for a picker")
    g.add_argument("--verbose", action="store_true",
                   help="Show the agents' full narration from the start (Ctrl+O toggles)")
    adapter.add_arguments(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv[0] if argv and argv[0] in ADAPTERS else "meta"
    if argv and argv[0] in ADAPTERS:
        argv = argv[1:]
    adapter = ADAPTERS[mode]()
    parser = build_parser(adapter)
    args = parser.parse_args(argv)
    adapter.validate_args(parser, args)

    if args.print_task is not None:
        from . import headless
        return headless.run(adapter, args, args.print_task)

    from .shell import Shell
    try:
        return Shell(adapter, args).run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
