"""
Defines command-line arguments for experimental agent selection.
"""
import argparse

def add_agent_args(parser: argparse.ArgumentParser, required: bool = False):
    """
    Adds arguments related to experimental agent selection to the parser.

    This function creates a dedicated argument group for agent-related options,
    allowing users to manually specify which analysis agent to use by name.

    Args:
        parser (argparse.ArgumentParser): The main argument parser instance.
        required (bool): If True, the --agent flag will be mandatory.
                         Defaults to False.
    """
    agent_group = parser.add_argument_group(
        'Agent Selection',
        'Options for choosing the experimental analysis agent.'
    )

    agent_group.add_argument(
        '--agent',
        type=str,
        choices=['general-microscopy', 'sam', 'atomistic', 'hyperspectral', 'curve', 'holistic-microscopy'],
        default=None,
        required=required,
        help=(
            "Specify the analysis agent by its short name. "
            "Choices: 'general-microscopy', 'sam', 'atomistic', 'hyperspectral', 'curve', 'holistic-microscopy'. "
            f"{'(REQUIRED for this command)' if required else '(Optional: overrides auto-selection)'}"
        )
    )
