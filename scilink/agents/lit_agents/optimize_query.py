"""
Query optimization utilities for literature search.

Works with both OpenAI-compatible and LiteLLM model wrappers.
"""

from typing import Any


def optimize_search_query(objective: str, 
                          model: Any, 
                          ) -> str:
    """
    Translates a raw, messy user objective into a clean, targeted search query.
    
    Works with any model wrapper that implements generate_content() method.
    
    Args:
        objective: Raw user objective that may contain local file references
        model: LLM model instance (OpenAIAsGenerativeModel or LiteLLMGenerativeModel)
        
    Returns:
        Cleaned search query suitable for scientific databases
    """
    prompt = f"""
    You are a Research Librarian.

    **USER INPUT:** "{objective}"

    **TASK:** Convert this input (which may refer to specific local files, results, or objectives) into a **clear, natural-language scientific query** suitable for an AI-powered literature search system.

    **RULES:**
    1. REMOVE LOCAL CONTEXT: Strip out references to specific files, provided datasets, or user actions (e.g., "analyze this spreadsheet", "my results", "provided data", "the attached file").
    2. PRESERVE SCIENTIFIC NOUNS: Preserve specific chemical names, material sources (e.g., "Produced Water", "Lithium-Ion Batteries"), and key analytes.
    3. STANDALONE: The output must make sense without knowing anything about the user's computer.
    4. NATURAL LANGUAGE: Write a proper sentence or question, NOT a list of keywords. For example: "What experimental methods exist for reversible CO2 capture using secondary amines as thermochemical cooling fluids?" — not a keyword dump.
    5. Return ONLY the query string.
    """
    
    try:
        response = model.generate_content([prompt])
        
        # Robust extraction - handle both wrapper types
        query = _extract_response_text(response)
        
        # Clean artifacts
        query = query.replace('"', '').replace("Search Query:", "").strip()
        
        # Remove any markdown formatting that might have been added
        if query.startswith("```"):
            query = query.strip("`").strip()
        
        print(f"  - 🧠 Query Optimized: '{query}'")
        return query
        
    except Exception as e:
        print(f"  - ⚠️ Query optimization failed: {e}. Using raw input.")
        return objective


def is_molecule_design_objective(objective: str, model: Any) -> bool:
    """
    Uses the LLM to classify whether an objective requires molecular
    design, synthesis planning, or de novo molecule generation.

    Returns True only for objectives that need cheminformatics capabilities,
    NOT for objectives that merely mention molecules in passing.

    Args:
        objective: Raw user objective.
        model: LLM model instance.

    Returns:
        True if the objective involves molecule design/synthesis.
    """
    prompt = f"""
    You are a scientific research classifier.

    **USER OBJECTIVE:** "{objective}"

    **TASK:** Determine if this objective requires **molecular design, synthesis planning, or de novo molecule generation**.

    Answer YES if the objective involves:
    - Designing or discovering new molecules or chemical compounds
    - Planning synthesis routes for target molecules
    - Optimizing molecular properties or structures
    - Drug design or lead compound optimization

    Answer NO if the objective merely:
    - Mentions molecules in passing (e.g., "molecular dynamics simulation")
    - Involves characterizing or analyzing existing molecular samples
    - Uses "molecular" as an adjective for a technique (e.g., "molecular beam epitaxy")
    - Is about data analysis, spectroscopy, or materials characterization

    Respond with ONLY "YES" or "NO".
    """

    try:
        response = model.generate_content([prompt])
        answer = _extract_response_text(response).strip().upper()
        result = answer.startswith("YES")
        if result:
            print(f"  - 🧪 Objective classified as molecule design task.")
        return result
    except Exception as e:
        print(f"  - ⚠️ Molecule classification failed: {e}. Skipping MOLECULES agent.")
        return False


def _extract_response_text(response) -> str:
    """
    Extract text from various response formats.
    
    Handles:
    - OpenAIAsGenerativeModel responses (SimpleNamespace with .text)
    - LiteLLMGenerativeModel responses (SimpleNamespace with .text)
    - Legacy Google GenAI responses (with .parts)
    - String responses
    """
    # Direct text attribute (both wrappers)
    if hasattr(response, 'text') and response.text:
        return response.text.strip()
    
    # Parts attribute (legacy Google format)
    if hasattr(response, 'parts') and response.parts:
        text_parts = []
        for part in response.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
        if text_parts:
            return " ".join(text_parts).strip()
    
    # Candidates with content (some response formats)
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content'):
            content = candidate.content
            if isinstance(content, str):
                return content.strip()
            if hasattr(content, 'parts') and content.parts:
                text_parts = []
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return " ".join(text_parts).strip()
    
    # Fallback: convert to string
    return str(response).strip()