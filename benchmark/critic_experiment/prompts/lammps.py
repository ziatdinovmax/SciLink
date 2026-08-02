"""LAMMPS prompts for Source 1 fixture generation.

Three systems chosen so script choices carry physics information (vanilla
LJ argon would have one obvious answer; these have several plausible
setups and known foot-guns):

  - water_box        — rigid-water constraints (SHAKE/RATTLE), Ewald,
                       timestep stability
  - nacl_aq          — Ewald long-range for ions, neutralization
  - lipf6_in_ec      — organic + ionic, pair_style combinations,
                       cross-interactions

Each parallels a VASP prompt in difficulty (easy / hard / medium).
"""
from __future__ import annotations

from typing import Tuple

from .vasp import Prompt  # same shape


PROMPTS: Tuple[Prompt, ...] = (
    Prompt(
        id="water_tip3p_nvt",
        system_name="water_box",
        request=(
            "Run an MD simulation of liquid water at 300 K and 1 atm "
            "for ~100 ps to compute the equilibrium density.  Use "
            "rigid water with appropriate bond/angle constraints."
        ),
        difficulty="easy",
        physics_focus=(
            "SHAKE/RATTLE for rigid water + timestep (>2 fs unstable "
            "without constraints) + units real/metal mismatch"
        ),
    ),
    Prompt(
        id="lipf6_ec_electrolyte",
        system_name="lipf6_in_ec",
        request=(
            "Run an MD simulation of LiPF₆ in ethylene carbonate at "
            "300 K to compute the Li⁺ self-diffusion coefficient.  "
            "Use an appropriate force field for the organic + ionic "
            "system."
        ),
        difficulty="hard",
        physics_focus=(
            "pair_style hybrid for organic + ions, cross-interactions, "
            "Ewald for Li⁺/PF₆⁻, MSD computation setup"
        ),
    ),
    Prompt(
        id="nacl_aq_ions",
        system_name="nacl_aq",
        request=(
            "Run an MD simulation of 1 M NaCl in water at 300 K to "
            "compute the Na⁺-Cl⁻ radial distribution function.  "
            "Treat long-range electrostatics appropriately."
        ),
        difficulty="medium",
        physics_focus=(
            "kspace_style pppm/ewald for ions, charge-neutrality check, "
            "compute rdf + ave/time"
        ),
    ),
)
