"""Planner scoping of required_outputs to per-pixel (H,W) maps (issue #385).

The dynamic-analysis contract can only validate spatial maps, and
required_outputs is locked per task — a 1-D artifact (Mean_Spectrum) planned
there makes the target unsatisfiable and burns the whole retry budget. The
fix is planner-side prompt scoping; this test pins the load-bearing wording
so it cannot silently disappear from the refinement instructions.
"""

from scilink.agents.exp_agents.instruct import (
    SPECTROSCOPY_REFINEMENT_INSTRUCTIONS,
)


def test_refinement_prompt_scopes_required_outputs_to_maps():
    text = SPECTROSCOPY_REFINEMENT_INSTRUCTIONS
    block = text[text.index("Required outputs"):]
    # the per-pixel (H, W) constraint
    assert "PER-PIXEL SCALAR" in block
    assert "(H, W) spatial map" in block
    # the named unsatisfiable class
    assert "mean/representative spectrum" in block
    # the redundancy note: the mean spectrum is auto-produced context
    assert "produced automatically" in block
    assert "never in `required_outputs`" in block
