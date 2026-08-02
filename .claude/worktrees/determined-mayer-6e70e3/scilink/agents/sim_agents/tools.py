import os
import logging

# Tool configurations - keywords and documentation paths
TOOL_CONFIGS = {
    "GrainBoundary": {
        "docs_path": "docs/aimsgb.txt",
        "keywords": ["grain boundary", "grain-boundary", "gb ", "sigma", "csl", 
                    "twist", "tilt", "bicrystal", "rotation axis", "aimsgb"],
    },
    "ASE": {
        "docs_path": None,
        "keywords": [],  # Default fallback
    }
}


def load_tool_documentation(docs_path: str, base_dir: str = None) -> str:
    """Load documentation from file if it exists."""
    if not docs_path:
        return None
    
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
        
    possible_paths = [
        docs_path,
        os.path.join(base_dir, docs_path),
        os.path.join(base_dir, "../..", docs_path),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                max_length = 60000
                if len(content) > max_length:
                    content = content[:max_length] + "\n\n[... Documentation truncated ...]"
                logging.info(f"Loaded docs from: {path}")
                return content
            except Exception as e:
                logging.error(f"Failed to read docs from {path}: {e}")
    return None