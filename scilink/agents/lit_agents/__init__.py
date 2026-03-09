"""
Literature & External Knowledge Agents Module

Provides agents for:
- Literature search via FutureHouse/Edison API (OWL, CROW)
- Molecule design and synthesis planning via Edison MOLECULES tool
- Novelty scoring of scientific claims
- Query optimization for scientific databases

Updated to use LiteLLM for multi-provider LLM support.
"""

from .literature_agent import (
    OwlLiteratureAgent,
    IncarLiteratureAgent,
    FittingModelLiteratureAgent,
    LiteratureSearchAgent
)
from .molecules_agent import MoleculesAgent
from .novelty_scorer import (
    NoveltyScorer,
    enhanced_novelty_assessment,
    display_enhanced_novelty_summary,
    save_enhanced_novelty_results,
    get_novelty_priorities_for_dft,
    get_novel_claims_legacy,
    get_known_claims_legacy
)
from .optimize_query import optimize_search_query, is_molecule_design_objective

__all__ = [
    # Literature agents
    'OwlLiteratureAgent',
    'IncarLiteratureAgent',
    'FittingModelLiteratureAgent',
    'LiteratureSearchAgent',
    # Molecules agent
    'MoleculesAgent',
    # Novelty scoring
    'NoveltyScorer',
    'enhanced_novelty_assessment',
    'display_enhanced_novelty_summary',
    'save_enhanced_novelty_results',
    'get_novelty_priorities_for_dft',
    'get_novel_claims_legacy',
    'get_known_claims_legacy',
    # Query optimization
    'optimize_search_query',
    'is_molecule_design_objective',
]