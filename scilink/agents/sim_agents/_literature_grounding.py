"""Literature grounding for the physics-validation critics.

Thin wrappers over SciLink's Edison literature agent that turn a critic's
information need into a literature query and return the answer as plain text.
These are the reasoning-first, sourced-anchor seams: the critic reasons about
the physics, and these functions fetch the published behaviour that anchors
that reasoning to real data rather than model recall.

Engine- and chemistry-neutral: the caller supplies the quantity, the swept
parameter, and a free-text system description; nothing here hardcodes a
species, property, or backend.
"""

import logging

logger = logging.getLogger("sim.grounding")


def search_reference_behavior(quantity, parameter, system_description, *,
                              api_key=None, max_wait_time=300):
    """Retrieve the published behaviour of ``quantity`` as ``parameter`` varies.

    Grounds a trend check: returns the literature answer describing the
    experimentally established trend direction (and endpoint values where
    available) so :class:`TrendCritic` reasons against real data instead of
    recall. The returned text is passed verbatim as the critic's
    ``reference_context``.

    Args:
        quantity: The plotted property (e.g. ``"mass density"``).
        parameter: The swept variable (e.g. ``"sulfone mole fraction"``).
        system_description: Free-text description of the system.
        api_key: FutureHouse/Edison key; falls back to ``FUTUREHOUSE_API_KEY``.
        max_wait_time: Poll timeout in seconds.

    Returns:
        The literature answer text, or ``None`` if the search fails or returns
        nothing. ``None`` lets the critic fall back to reasoning-only rather
        than blocking — the grounding is an anchor, not a gate.
    """
    from ..lit_agents.literature_agent import LiteratureSearchAgent

    query = (
        f"For the system: {system_description}. "
        f"How does the {quantity} change as the {parameter} increases? "
        f"State the experimentally established trend direction — does it "
        f"increase, decrease, or vary non-monotonically — and report measured "
        f"values at the composition endpoints if available. Cite sources."
    )
    try:
        agent = LiteratureSearchAgent(api_key=api_key, max_wait_time=max_wait_time)
        result = agent._execute_crow_task(query, task_type="reference-trend")
    except Exception as exc:
        logger.warning("Reference-behaviour search failed: %s", exc)
        return None

    if result.get("status") == "success" and result.get("content"):
        return result["content"]
    logger.warning("Reference-behaviour search returned no answer (%s).",
                   result.get("status"))
    return None


def search_force_field_parameters(recommendation, tried=(), *,
                                  api_key=None, max_wait_time=300):
    """Search the literature for a validated parameter set for a flagged fix.

    Given an advisor's recommendation for what is miscalibrated, retrieve the
    published, validated parameters for that chemistry so the fix is sourced
    rather than invented. Returns the literature answer describing the
    recommended parameters and their provenance.

    This returns the *recommendation text*, not an applied force-field file:
    converting a cited parameter set into an engine-specific override is a
    separate, backend-dependent step the caller owns. Kept a distinct seam so
    the sourcing (general, engine-neutral) and the applying (engine-specific)
    do not entangle.

    Args:
        recommendation: The advisor's description of the miscalibration/fix.
        tried: Descriptions of already-attempted parameter sets to exclude.
        api_key: FutureHouse/Edison key; falls back to ``FUTUREHOUSE_API_KEY``.
        max_wait_time: Poll timeout in seconds.

    Returns:
        The literature answer text describing the recommended parameters, or
        ``None`` if the search fails or returns nothing.
    """
    from ..lit_agents.literature_agent import LiteratureSearchAgent

    exclude = ""
    if tried:
        exclude = (" Do not recommend these already-tried parameter sets: "
                   + "; ".join(str(t) for t in tried) + ".")
    query = (
        f"{recommendation} What validated, published force-field parameters "
        f"address this? Report the specific parameter set (partial charges, "
        f"Lennard-Jones terms, or the named force field) and cite the source "
        f"paper where they were validated against experiment.{exclude}"
    )
    try:
        agent = LiteratureSearchAgent(api_key=api_key, max_wait_time=max_wait_time)
        result = agent._execute_crow_task(query, task_type="ff-parameters")
    except Exception as exc:
        logger.warning("Force-field-parameter search failed: %s", exc)
        return None

    if result.get("status") == "success" and result.get("content"):
        return result["content"]
    logger.warning("Force-field-parameter search returned no answer (%s).",
                   result.get("status"))
    return None
