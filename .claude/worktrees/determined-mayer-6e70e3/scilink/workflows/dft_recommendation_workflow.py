# scilink/workflows/dft_recommendation_workflow.py
#
# Backward-compatibility shim. The class moved to scilink.agents.exp_agents
# (alongside the RecommendationAgent it wraps) and was renamed DFTRecommender.

import warnings

from ..agents.exp_agents.dft_recommender import DFTRecommender as DFTRecommendationsWorkflow

warnings.warn(
    "scilink.workflows.dft_recommendation_workflow.DFTRecommendationsWorkflow is "
    "deprecated; import DFTRecommender from scilink.agents.exp_agents instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DFTRecommendationsWorkflow"]
