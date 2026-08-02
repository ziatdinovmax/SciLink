"""VASP prompts for Source 1 fixture generation.

Same three prompts as ``benchmark/test_incar_variability.py``. Each picks a
system where INCAR choices carry physics information (vanilla Si bulk has
one obvious answer; Fe / UO₂ / Pt111+CO have several plausible setups and
known foot-guns).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Prompt:
    id: str
    system_name: str       # key into benchmark.systems
    request: str           # natural-language goal
    difficulty: str        # easy | medium | hard
    physics_focus: str     # what makes this prompt foot-gunny


PROMPTS: Tuple[Prompt, ...] = (
    Prompt(
        id="fe_bcc_magnetic",
        system_name="fe_bcc",
        request=(
            "Set up a cell + ionic relaxation of BCC iron to obtain the "
            "equilibrium lattice constant.  Apply appropriate settings "
            "for a ferromagnetic metal."
        ),
        difficulty="easy",
        physics_focus="ISPIN + MAGMOM (Fe-specific; ISPN-for-ISPIN typo class)",
    ),
    Prompt(
        id="uo2_dftU",
        system_name="uo2_fluorite",
        request=(
            "Set up a relaxation of UO₂ in the fluorite structure.  "
            "UO₂ is a strongly-correlated antiferromagnetic Mott "
            "insulator; treat the U 5f electrons appropriately."
        ),
        difficulty="hard",
        physics_focus="LDAUL / LDAUU / LDAUJ + ISPIN + MAGMOM-AFM",
    ),
    Prompt(
        id="pt111_co_dipole",
        system_name="pt111_co_top",
        request=(
            "Set up a relaxation of CO adsorbed at the Pt(111) top site "
            "to compute the adsorption energy.  Account for the "
            "asymmetric slab geometry."
        ),
        difficulty="medium",
        physics_focus="IDIPOL / LDIPOL + ISMEAR for metal + selective dynamics",
    ),
)
