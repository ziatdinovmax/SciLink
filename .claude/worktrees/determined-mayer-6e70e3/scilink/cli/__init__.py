"""SciLink CLI interface."""
# Main CLI entry point
from .main import main

# Individual command modules
from . import plan
from . import simulate
from . import analyze
from . import prepare_ff

__all__ = ['main', 'plan', 'simulate', 'analyze', 'prepare-ff']
