"""
Enhanced novelty scoring system for scientific claims assessment.

"""

import json
import logging
import os
from typing import Dict, Any, List, Tuple

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from ...auth import get_api_key, APIKeyNotFoundError
from .instruct import NOVELTY_SCORING_INSTRUCTIONS
class NoveltyScorer:
    """Enhanced novelty assessment using structured scoring instead of binary Yes/No."""
    
    def __init__(self, google_api_key: str = None, model_name: str = "gemini-2.5-flash-preview-05-20"):
        """
        Initialize the novelty scorer.
        
        Args:
            google_api_key: Google API key for Gemini access
            model_name: Gemini model to use for scoring (Flash recommended for speed)
        """
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')
        
        genai.configure(api_key=google_api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"NoveltyScorer initialized with model: {model_name}")
    
    def score_novelty(self, question: str, owl_response: str) -> Dict[str, Any]:
        """
        Score the novelty of a scientific claim based on literature analysis.
        
        Args:
            question: The original "Has anyone..." research question
            owl_response: The literature analysis response from OWL
            
        Returns:
            Dictionary with novelty score, explanation, and confidence
        """
        try:
            prompt = NOVELTY_SCORING_INSTRUCTIONS.format(
                question=question,
                response=owl_response
            )
            
            self.logger.debug(f"Scoring novelty for question: {question[:100]}...")
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            result = json.loads(response.text)
            
            # Validate the response
            if not self._validate_scoring_response(result):
                self.logger.warning("Invalid scoring response, using fallback")
                return self._create_fallback_score(question, owl_response)
            
            self.logger.info(f"Novelty scored: {result['novelty_score']}/5")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to score novelty: {e}")
            return self._create_fallback_score(question, owl_response)
    
    def _validate_scoring_response(self, result: Dict[str, Any]) -> bool:
        """Validate the scoring response format and content."""
        required_keys = ["novelty_score", "explanation"]
        
        if not all(key in result for key in required_keys):
            self.logger.warning(f"Missing required keys in response: {result}")
            return False
        
        score = result.get("novelty_score")
        if not isinstance(score, int) or not (1 <= score <= 5):
            self.logger.warning(f"Invalid novelty score: {score}")
            return False
        
        explanation = result.get("explanation", "")
        if not isinstance(explanation, str) or len(explanation.strip()) == 0:
            self.logger.warning("Empty or invalid explanation")
            return False
        
        return True
    
    def _create_fallback_score(self, question: str, owl_response: str) -> Dict[str, Any]:
        """Create a fallback score when AI scoring fails."""
        # Simple heuristic based on the original Yes/No logic
        response_lower = owl_response.lower()
        
        if 'no' in response_lower[:10]:  # Likely novel
            score = 4
            explanation = "Literature suggests this may be a novel finding (fallback assessment)"
        elif 'yes' in response_lower[:10]:  # Likely known
            score = 2
            explanation = "Literature suggests this is a known finding (fallback assessment)"
        else:
            score = 3
            explanation = "Uncertain novelty based on literature analysis (fallback assessment)"
        
        self.logger.info(f"Using fallback score: {score}/5")
        
        return {
            "novelty_score": score,
            "explanation": explanation
        }
    
    @staticmethod
    def get_novelty_description(score: int) -> str:
        """Get human-readable description of novelty score."""
        descriptions = {
            1: "Not novel at all",
            2: "Minor novelty/Incremental improvement", 
            3: "Moderate novelty/Significant confirmation or refinement",
            4: "High novelty/New insight or significant breakthrough",
            5: "Groundbreaking/Revolutionary"
        }
        return descriptions.get(score, "Unknown novelty level")
    
    @staticmethod
    def categorize_claims_by_novelty(scored_claims: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize claims by their novelty scores."""
        categories = {
            "highly_novel": [],      # Scores 4-5
            "moderately_novel": [],  # Score 3
            "minimally_novel": [],   # Score 2
            "not_novel": []         # Score 1
        }
        
        for claim_result in scored_claims:
            score = claim_result.get("novelty_assessment", {}).get("novelty_score", 0)
            
            if score >= 4:
                categories["highly_novel"].append(claim_result)
            elif score == 3:
                categories["moderately_novel"].append(claim_result)
            elif score == 2:
                categories["minimally_novel"].append(claim_result)
            else:
                categories["not_novel"].append(claim_result)
        
        return categories


def enhanced_novelty_assessment(literature_results: List[Dict[str, Any]], 
                               novelty_scorer: NoveltyScorer = None) -> Dict[str, Any]:
    """
    Enhanced novelty assessment with structured scoring.
    
    Args:
        literature_results: List of literature search results from OWL
        novelty_scorer: Optional NoveltyScorer instance
        
    Returns:
        Enhanced novelty assessment with scores and categorization
    """
    if novelty_scorer is None:
        novelty_scorer = NoveltyScorer()
    
    scored_results = []
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running enhanced novelty assessment on {len(literature_results)} literature results")
    
    for i, result in enumerate(literature_results, 1):
        if result['owl_result']['status'] != "success":
            logger.warning(f"Skipping result {i} due to failed OWL search")
            continue
        
        original_claim = result['original_claim']
        question = original_claim.get('has_anyone_question', '')
        owl_response = result['owl_result'].get('formatted_answer', '')
        
        if not question or not owl_response:
            logger.warning(f"Skipping result {i} due to missing question or response")
            continue
        
        logger.info(f"Scoring claim {i}/{len(literature_results)}")
        
        # Score the novelty
        novelty_assessment = novelty_scorer.score_novelty(question, owl_response)
        print()
        print(f"🔍 Claim: {original_claim['claim']}")
        print(f"❓ Question: {question}")
        print(f"✅ Novelty Score: {novelty_assessment['novelty_score']}/5")
        print(f"📝 Explanation: {novelty_assessment['explanation']}")
        print()
            
        
        # Add to the result
        enhanced_result = result.copy()
        enhanced_result['novelty_assessment'] = novelty_assessment
        scored_results.append(enhanced_result)
    
    if not scored_results:
        logger.warning("No claims could be scored for novelty")
        return {
            "total_claims_searched": len(literature_results),
            "successful_searches": 0,
            "average_novelty_score": 0.0,
            "novelty_categories": {
                "highly_novel": [],
                "moderately_novel": [],
                "minimally_novel": [],
                "not_novel": []
            },
            "detailed_scores": [],
            "potentially_novel": [],
            "known_findings": []
        }
    
    # Categorize claims by novelty
    categories = NoveltyScorer.categorize_claims_by_novelty(scored_results)
    
    # Create summary statistics
    total_scored = len(scored_results)
    avg_score = sum(r['novelty_assessment']['novelty_score'] for r in scored_results) / total_scored
    
    # Create flat lists for categories
    novelty_categories = {
        "highly_novel": [r['original_claim']['claim'] for r in categories["highly_novel"]],
        "moderately_novel": [r['original_claim']['claim'] for r in categories["moderately_novel"]], 
        "minimally_novel": [r['original_claim']['claim'] for r in categories["minimally_novel"]],
        "not_novel": [r['original_claim']['claim'] for r in categories["not_novel"]]
    }
    
    # For backward compatibility: combine moderate and high novelty as "potentially novel"
    potentially_novel = novelty_categories["highly_novel"] + novelty_categories["moderately_novel"]
    known_findings = novelty_categories["minimally_novel"] + novelty_categories["not_novel"]
    
    assessment = {
        "total_claims_searched": len(literature_results),
        "successful_searches": total_scored,
        "average_novelty_score": round(avg_score, 1),  # Round to 1 decimal for cleaner display
        "novelty_categories": novelty_categories,
        "detailed_scores": scored_results,
        # For backward compatibility with existing workflows
        "potentially_novel": potentially_novel,
        "known_findings": known_findings
    }
    
    logger.info(f"Enhanced novelty assessment complete: avg score {avg_score:.1f}/5")
    
    return assessment


def display_enhanced_novelty_summary(assessment: Dict[str, Any]):
    """Display enhanced novelty assessment results with simple formatting."""
    print("\n--- Literature Search Summary ---")
    print(f"Total claims analyzed: {assessment['total_claims_searched']}")
    print(f"Successfully searched: {assessment['successful_searches']}")
    print(f"Average novelty score: {assessment['average_novelty_score']:.1f}/5.0")
    
    categories = assessment['novelty_categories']
    
    # Combine high and moderate novelty as "potentially novel" (scores 3-5)
    potentially_novel = categories['highly_novel'] + categories['moderately_novel']
    known_findings = categories['minimally_novel'] + categories['not_novel']
    
    if potentially_novel:
        print("\nPotentially Novel Findings:")
        detailed_scores = assessment.get('detailed_scores', [])
        
        # Create lookup for scores
        score_lookup = {}
        for result in detailed_scores:
            claim = result['original_claim']['claim']
            score = result['novelty_assessment']['novelty_score']
            score_lookup[claim] = score
        
        for i, claim in enumerate(potentially_novel, 1):
            score = score_lookup.get(claim, 0)
            print(f"  {i}. {claim} (Score: {score}/5)")
    
    if known_findings:
        print("\nPreviously Reported Findings:")
        detailed_scores = assessment.get('detailed_scores', [])
        
        # Create lookup for scores
        score_lookup = {}
        for result in detailed_scores:
            claim = result['original_claim']['claim']
            score = result['novelty_assessment']['novelty_score']
            score_lookup[claim] = score
        
        for i, claim in enumerate(known_findings, 1):
            score = score_lookup.get(claim, 0)
            print(f"  {i}. {claim} (Score: {score}/5)")
    
    print("-" * 50)


def save_enhanced_novelty_results(assessment: Dict[str, Any], output_dir: str, 
                                filename: str = "enhanced_novelty_assessment.json") -> str:
    """Save enhanced novelty assessment results to file."""
    
    logger = logging.getLogger(__name__)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Add metadata to the assessment
        enhanced_assessment = assessment.copy()
        enhanced_assessment['metadata'] = {
            'version': '2.0',
            'scoring_method': 'enhanced_llm_scoring',
            'novelty_scale': {
                1: NoveltyScorer.get_novelty_description(1),
                2: NoveltyScorer.get_novelty_description(2),
                3: NoveltyScorer.get_novelty_description(3),
                4: NoveltyScorer.get_novelty_description(4),
                5: NoveltyScorer.get_novelty_description(5)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(enhanced_assessment, f, indent=2, default=str)
        
        logger.info(f"Enhanced novelty assessment saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save enhanced novelty assessment: {e}")
        raise


def get_novelty_priorities_for_dft(assessment: Dict[str, Any]) -> List[str]:
    """
    Extract prioritized claims for DFT structure recommendations.
    
    Returns claims ordered by novelty score (highest first).
    """
    detailed_scores = assessment.get('detailed_scores', [])
    
    # Sort by novelty score (descending)
    sorted_claims = sorted(
        detailed_scores, 
        key=lambda x: x.get('novelty_assessment', {}).get('novelty_score', 0),
        reverse=True
    )
    
    # Extract just the claim text in priority order
    prioritized_claims = [
        result['original_claim']['claim'] 
        for result in sorted_claims
    ]
    
    return prioritized_claims


# Backward compatibility functions
def get_novel_claims_legacy(assessment: Dict[str, Any]) -> List[str]:
    """Get potentially novel claims in legacy format (scores >= 3)."""
    return assessment.get('potentially_novel', [])


def get_known_claims_legacy(assessment: Dict[str, Any]) -> List[str]:
    """Get known claims in legacy format (scores <= 2)."""
    return assessment.get('known_findings', [])