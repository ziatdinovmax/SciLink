"""Prompt registries for the critic-validator experiment.

Same `Prompt` shape across engines. Per-engine `PROMPTS` tuple is the
authoritative set; `generate_source1.py` iterates over it.
"""
from .vasp import PROMPTS as VASP_PROMPTS
from .lammps import PROMPTS as LAMMPS_PROMPTS

__all__ = ["VASP_PROMPTS", "LAMMPS_PROMPTS"]
