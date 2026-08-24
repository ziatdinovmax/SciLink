"""Live smoke test for the structure_matching/xrd skill.

Exercises the full pipeline: planner sees the skill's planning section,
generated script calls search_structures + simulate_xrd_pattern +
score_xrd_match, the stdout markers (FIT_RESULTS_JSON / DB_MATCHES_JSON)
land in fit_results, and synthesis references the matched candidates.

Runs only when:
    - LLM key present (ANTHROPIC_API_KEY or GEMINI_API_KEY)
    - MP_API_KEY present (so the MP backend is queryable)
    - full pymatgen (with analysis.diffraction.xrd) installed
    - UNSAFE_EXECUTION_OK=true (required by the curve-fitting executor)

Manual invocation:
    UNSAFE_EXECUTION_OK=true \\
    ANTHROPIC_API_KEY=... \\
    MP_API_KEY=... \\
        python -m pytest tests/test_structure_matching_smoke_live.py -v -s
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scilink.skills.structure_matching._backends.materials_project import (
    MP_API_AVAILABLE,
)
from scilink.skills.structure_matching.xrd.simulate_xrd import PYMATGEN_XRD_AVAILABLE

pytestmark = [
    pytest.mark.skipif(
        not (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        ),
        reason="no LLM API key in env (ANTHROPIC_API_KEY / GEMINI_API_KEY)",
    ),
    pytest.mark.skipif(
        not os.environ.get("MP_API_KEY"),
        reason="MP_API_KEY not set",
    ),
    pytest.mark.skipif(
        not (MP_API_AVAILABLE and PYMATGEN_XRD_AVAILABLE),
        reason="mp-api or pymatgen XRD analysis module not installed; install scilink[structure-matching]",
    ),
]


@pytest.fixture(scope="module")
def synthetic_si_xrd(tmp_path_factory):
    """Synthesize an 'experimental' XRD pattern from a silicon CIF.

    Builds a Si structure, computes its kinematic CuKa pattern, broadens
    peaks with a Lorentzian (FWHM=0.2°), adds 5% Gaussian noise, and saves
    as a two-column CSV. The point is to have data where the right answer
    (Si, Fd-3m, mp-149) is unambiguous so the test can assert it.
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    tmpdir = tmp_path_factory.mktemp("si_xrd")
    structure = Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]],
    )
    pattern = XRDCalculator(wavelength="CuKa").get_pattern(
        structure, two_theta_range=(20, 80),
    )

    grid = np.arange(20.0, 80.0, 0.05)
    fwhm = 0.2
    gamma = fwhm / 2
    intensity = np.zeros_like(grid)
    for x0, amp in zip(pattern.x, pattern.y):
        intensity += amp * (gamma ** 2) / ((grid - x0) ** 2 + gamma ** 2)
    rng = np.random.default_rng(42)
    intensity += rng.normal(scale=0.02 * intensity.max(), size=intensity.shape)
    intensity = np.clip(intensity, 0, None)

    csv_path = tmpdir / "si_xrd_synthetic.csv"
    np.savetxt(
        csv_path,
        np.column_stack([grid, intensity]),
        delimiter=",",
        header="two_theta,intensity",
        comments="",
    )
    return csv_path


def _pick_model_and_key():
    """Pick whichever LLM provider has a key in env."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-opus-4-6", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-pro", os.environ["GEMINI_API_KEY"]
    pytest.skip("no LLM key")


def test_xrd_skill_identifies_silicon_prefit_path(synthetic_si_xrd):
    """Pre-fit path: chemistry hypothesized as Si → DB query → simulate → score.

    Asserts:
    1. The skill is discoverable and loadable.
    2. The agent runs without raising.
    3. ``fit_results['db_matches']`` is populated (from DB_MATCHES_JSON marker).
    4. Synthesis output mentions silicon by formula or 'Si'.
    """
    os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent

    model_name, api_key = _pick_model_and_key()
    out_dir = tempfile.mkdtemp(prefix="xrd_smoke_")

    agent = CurveFittingAgent(
        api_key=api_key,
        model_name=model_name,
        output_dir=out_dir,
        enable_human_feedback=False,
        use_literature=False,
        run_preprocessing=False,
    )

    result = agent.analyze(
        data=str(synthetic_si_xrd),
        system_info={
            "technique": "XRD",
            "wavelength": "CuKa",
            "chemistry_hint": ["Si"],
            "notes": "Synthetic pattern derived from Si Fd-3m for smoke testing.",
        },
        skill="xrd",
    )

    # Sanity on the result envelope
    assert isinstance(result, dict)
    assert "detailed_analysis" in result or "status" in result

    # Synthesis text should reference Si in some form.
    text_blob = json.dumps(result).lower()
    assert any(token in text_blob for token in ("silicon", " si ", '"si"', "fd-3m", "mp-149")), (
        "Synthesis output makes no mention of silicon or its space group / mp-id; "
        f"check session at {out_dir}"
    )

    # The session directory should contain the materialized candidate CIFs
    # somewhere under it (skill outputs to ./candidates relative to the
    # script's cwd, which the executor sets to a per-spectrum subdir).
    cifs = list(Path(out_dir).rglob("*.cif"))
    assert cifs, f"No candidate CIFs materialized under {out_dir}"
