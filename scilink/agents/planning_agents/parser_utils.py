import os
import re
from typing import List, Dict, Any
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

import json
import pandas as pd
import PIL.Image as PIL_Image

from scilink.parsers import parse_adaptive_excel, SUPPORTED_EXTENSIONS

# get_files_from_directory / SUPPORTED_EXTENSIONS / table_to_markdown now live
# in scilink.parsers; parse_json_from_response lives in scilink.knowledge.


def generate_repo_map(root_dir: str) -> str:
    """
    Generates a visual tree structure of the repository.
    Useful for giving the LLM context on where files live for imports.
    """
    root = Path(root_dir)
    if not root.exists(): return ""

    tree_lines = [f"{root.name}/"]
    
    def _nat_key(p):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(p))]
    for path in sorted(root.rglob('*'), key=_nat_key):
        # Skip hidden files/dirs
        if any(part.startswith('.') or part in ('__pycache__', 'venv', 'env') for part in path.parts):
            continue
        
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel_path = path.relative_to(root)
            depth = len(rel_path.parts)
            indent = '    ' * (depth - 1)
            tree_lines.append(f"{indent}├── {path.name}")
            
    return "\n".join(tree_lines)

def append_experiment_result(file_path: str, parameters: Dict[str, float], results: Dict[str, float]):
    """
    Appends a completed experiment (Params + Results) to the cumulative dataset.
    This 'closes the loop' for the BO Agent.
    """
    path = Path(file_path)
    
    # Merge input parameters and lab results into one row
    new_row = {**parameters, **results}
    
    if not path.exists():
        # Create new if doesn't exist
        df = pd.DataFrame([new_row])
    else:
        if path.suffix == '.xlsx':
            df = pd.read_excel(path)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        # Append
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save back
    if path.suffix == '.xlsx':
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"✅ Appended result to {path.name}. New size: {len(df)}")


def write_experiments_to_disk(result_json: Dict[str, Any], target_dir: str) -> List[str]:
    """
    Parses the result JSON and writes 'implementation_code' to .py files in the target directory.
    Returns a list of filenames that were successfully saved.
    """
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    experiments = result_json.get("proposed_experiments", [])
    saved_files = []
    
    if not experiments:
        logging.warning(f"No experiments found to save in {target_dir}")
        return []
    
    for i, exp in enumerate(experiments):
        code_content = exp.get("implementation_code")
        exp_name = exp.get("experiment_name", f"Experiment_{i+1}")
        
        # 1. Clean filename
        # Replace spaces with underscores and remove non-alphanumeric chars (except _ and .)
        safe_name = "".join(c for c in exp_name if c.isalnum() or c in (' ', '_', '.')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        
        # Fallback if name becomes empty after cleaning
        if not safe_name: 
            safe_name = f"experiment_code_{i+1}"
            
        filename = f"{safe_name}.py"
        file_path = path / filename

        # 2. Extract and Write
        if code_content and "No relevant code found" not in code_content:
            try:
                # Strip markdown code blocks (```python ... ```)
                code_lines = code_content.splitlines()
                
                # Logic to find the content between the backticks
                start_index = next((j for j, line in enumerate(code_lines) if line.strip().startswith('```')), -1)
                end_index = next((j for j, line in enumerate(code_lines[start_index+1:]) if line.strip().endswith('```')), -1)
                
                if start_index != -1 and end_index != -1:
                    # Adjust end_index because we sliced the list
                    actual_end = start_index + 1 + end_index
                    extracted_code = "\n".join(code_lines[start_index + 1 : actual_end]).strip()
                else:
                    # Fallback: assume the whole string is code if no backticks found
                    extracted_code = code_content.strip()

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_code)
                
                saved_files.append(filename)
                
            except Exception as e:
                logging.error(f"Failed to write {filename}: {e}")
        else:
            logging.info(f"Experiment {i+1} ('{exp_name}') has no executable code.")

    return saved_files


def resolve_primary_data_path(data_input: Union[str, Dict[str, str], None]) -> Optional[Dict[str, str]]:
    """
    Normalizes input into a standard Dict format for primary data.
    
    Capabilities:
    1. Normalizes String path ("data.xlsx") -> Dict
    2. Auto-Discovers metadata: Checks for 'data.json' or 'data.user_desc.json'
    3. Interactive Fallback: Prompts user if no metadata is found (and saves the result).
    """
    if not data_input:
        return None

    # 1. Normalize String input to Dict
    if isinstance(data_input, str):
        data_input = {"file_path": data_input}

    path = Path(data_input["file_path"])
    if not path.exists():
        print(f"❌ Primary data file not found: {path}")
        return None

    # 2. Check if metadata path is already explicitly provided
    if "metadata_path" in data_input and data_input["metadata_path"]:
        return data_input 

    # 3. Auto-Discovery Logic
    # Priority A: Check for existing matching JSON (e.g., data.json)
    candidate_json = path.with_suffix('.json')
    if candidate_json.exists():
        print(f"  - 🔍 Auto-discovered metadata: {candidate_json.name}")
        return {"file_path": str(path), "metadata_path": str(candidate_json)}

    # Priority B: Check for previously saved user description (e.g., data.user_desc.json)
    saved_desc_file = path.with_suffix('.user_desc.json')
    if saved_desc_file.exists():
        print(f"  - 🔍 Found saved user description: {saved_desc_file.name}")
        return {"file_path": str(path), "metadata_path": str(saved_desc_file)}

    # Priority D: Session-internal artifacts (produced by the agent itself
    # earlier in this session — screening CSVs, scalarizer outputs, etc.).
    # The agent already knows what's in them; the metadata prompt is asking
    # the human about something the agent just wrote. Skip the prompt and
    # let the parser infer from headers.
    for parent in path.parents:
        if parent.name.startswith("meta_session_") or parent.name == "campaign_outputs":
            print(f"  - ℹ️  Session-internal file ({path.name}); "
                  "skipping metadata prompt — parser will use headers.")
            return {"file_path": str(path), "metadata_path": None}

    # 4. Interactive Fallback
    from .user_interface import get_dataset_description
    
    user_desc = get_dataset_description(path.name)
    
    if user_desc:
        try:
            # We create a minimal valid JSON structure for the excel_parser
            # We map 'description' to 'objective' so it gets picked up by excel_parser logic
            meta_content = {
                "title": path.stem,
                "objective": user_desc, 
                "generated_by": "user_interactive_prompt"
            }
            
            with open(saved_desc_file, 'w', encoding='utf-8') as f:
                json.dump(meta_content, f, indent=2)
            
            print(f"  - 💾 Saved description to: {saved_desc_file.name}")
            return {"file_path": str(path), "metadata_path": str(saved_desc_file)}
        except Exception as e:
            print(f"  - ⚠️ Could not save description file: {e}")
            return {"file_path": str(path), "metadata_path": None}
    
    # User chose to skip
    return {"file_path": str(path), "metadata_path": None}


def parse_data_file(file_path: str, 
                   metadata_path: Optional[str] = None) -> str:
    """
    Unified data file parsing for both initial planning and iteration.
    Auto-discovers metadata JSON if not provided.
    
    Args:
        file_path: Path to data file (.csv, .xlsx, .xls)
        metadata_path: Optional explicit metadata path (overrides auto-discovery)
    
    Returns:
        String containing formatted data summary
    """
    # Auto-discover metadata using existing logic
    data_dict = resolve_primary_data_path(file_path)
    
    if data_dict is None:
        return f"[Error: File not found - {file_path}]"
    
    # Override metadata if explicitly provided (for backward compatibility)
    if metadata_path is not None:
        data_dict['metadata_path'] = metadata_path
    
    try:
        chunks = parse_adaptive_excel(
            data_dict['file_path'],
            data_dict.get('metadata_path')
        )
        
        if chunks:
            # Return the summary chunk (prioritize dataset_summary or dataset_package)
            summary = next(
                (c for c in chunks 
                 if c['metadata'].get('content_type') in 
                    ('dataset_summary', 'dataset_package')), 
                chunks[0]
            )
            return summary['text']
        
        return f"[No data extracted from {file_path}]"
        
    except Exception as e:
        return f"[Error parsing {file_path}: {e}]"


def load_image_file(image_path: str) -> Optional[Any]:
    """
    Unified image loading with error handling.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object or None if loading fails
    """
    if PIL_Image is None:
        logging.warning("PIL not installed. Cannot load images.")
        return None
    
    try:
        with PIL_Image.open(image_path) as img:
            img.load()
            return img.copy()
    except Exception as e:
        logging.warning(f"Failed to load image {image_path}: {e}")
        return None


def parse_multimodal_results(results: Any) -> Tuple[str, List]:
    """
    Extracts text and images from various result formats.
    
    Handles multiple input formats:
    - String: "Yield was 85%"
    - File path: "./data.csv" or "./plot.png"
    - Dict: {"path": "./file.csv", "description": "..."}
    - List: Mix of above types
    
    Args:
        results: Experimental results in any supported format
        
    Returns:
        Tuple of (consolidated_text, loaded_images)
        
    Example:
        >>> text, images = parse_multimodal_results([
        ...     "./experiment.csv",
        ...     {"path": "./plot.png", "description": "Results"},
        ...     "Precipitation observed"
        ... ])
    """
    parsed_text_results = []
    loaded_images = []
    
    def process_item(item: Any, description: str = "") -> str:
        text_output = ""
        
        # If it's a file path (skip strings that are too long or contain newlines)
        def _is_file_path(s: str) -> bool:
            try:
                return len(s) <= 260 and "\n" not in s and Path(s).exists()
            except OSError:
                return False

        if isinstance(item, str) and _is_file_path(item):
            path = Path(item)
            suffix = path.suffix.lower()
            
            # A. Data Files
            if suffix in ['.xlsx', '.xls', '.csv']:
                print(f"  - 📄 Parsing data file: {path.name}")
                text_output = parse_data_file(str(path))
                text_output = f"DATA FILE ({path.name}):\n{text_output}"

            # B. Images
            elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                print(f"  - 🖼️  Loading result image: {path.name}")
                img = load_image_file(str(path))
                if img:
                    loaded_images.append(img)
                    text_output = f"[Attached Image: {path.name}]"
                else:
                    text_output = f"[Error loading image: {path.name}]"
            
            # C. Logs/Text
            elif suffix in ['.txt', '.log', '.md', '.json']:
                try:
                    content = path.read_text(encoding='utf-8')
                    text_output = f"LOG FILE ({path.name}):\n{content}"
                except Exception as e:
                    text_output = f"[Error reading log {path.name}: {e}]"
            
            else:
                text_output = f"FILE ({path.name})"

        # If not a file, treat as raw text/data
        else:
            if isinstance(item, (dict, list)):
                text_output = json.dumps(item, indent=2)
            else:
                text_output = str(item)
        
        # Append description if provided
        if description:
            text_output += f"\n(Context: {description})"
        
        return text_output

    # Process results
    items_to_process = results if isinstance(results, list) else [results]
    
    for entry in items_to_process:
        if isinstance(entry, dict):
            # Structured file entry
            path_val = entry.get('path') or entry.get('file') or entry.get('image')
            desc_val = (entry.get('description') or entry.get('desc') or 
                       entry.get('caption') or entry.get('notes'))
            
            if path_val and isinstance(path_val, str):
                parsed_text_results.append(process_item(path_val, desc_val or ""))
            else:
                parsed_text_results.append(json.dumps(entry, indent=2))
        else:
            parsed_text_results.append(process_item(entry))

    consolidated_feedback = "\n\n".join(parsed_text_results)
    return consolidated_feedback, loaded_images

# ── Plan payload accessors ───────────────────────────────────────────
# A plan-mode artifact is one of two things: an EXPERIMENT (a testable
# hypothesis with measurements, in `proposed_experiments`) or a PORTFOLIO of
# research directions (ideation). The portfolio rode the experiment schema
# for a long time — one `proposed_experiments` entry wrapping the whole
# portfolio, with the directions in `concepts` if the author complied and in
# `experimental_steps` as pseudo-sections if it did not.
#
# Everything that reads directions goes through here, so consumers never
# branch on payload shape and old sessions stay readable. Resolution order:
#   1. top-level `directions`      — the portfolio contract
#   2. `proposed_experiments[*].concepts` — the PR #394 shape
#   3. a single direction synthesised from the experiment fields — plans
#      that predate `concepts` entirely
# Tiers 2 and 3 are what let a checkpoint written before this change restore.

def plan_is_portfolio(plan: Optional[Dict[str, Any]]) -> bool:
    """Is this plan-mode artifact a research portfolio rather than an
    experiment? Reads the plan's own stamp; falls back to the presence of
    directions for a plan dict read in isolation."""
    if not isinstance(plan, dict):
        return False
    if plan.get("type") == "ideation":
        return True
    return bool(plan.get("directions"))


def plan_thesis(plan: Optional[Dict[str, Any]]) -> str:
    """The portfolio's organizing idea, or the experiment's hypothesis."""
    if not isinstance(plan, dict):
        return ""
    if plan.get("thesis"):
        return str(plan["thesis"])
    exps = plan.get("proposed_experiments") or []
    return str((exps[0] or {}).get("hypothesis") or "") if exps else ""


def plan_directions(plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every research direction in a portfolio, in the author's order.

    Returns [] for an experiment plan — callers use that to tell the shapes
    apart without reading the schema themselves.
    """
    if not isinstance(plan, dict):
        return []
    top = plan.get("directions")
    if isinstance(top, list) and top:
        return [d for d in top if isinstance(d, dict)]

    nested: List[Dict[str, Any]] = []
    for exp in plan.get("proposed_experiments") or []:
        if isinstance(exp, dict):
            nested.extend(c for c in (exp.get("concepts") or [])
                          if isinstance(c, dict))
    if nested:
        return nested

    # Tier 3: an ideation plan authored before `concepts` existed carries one
    # direction spread across the experiment fields. Synthesised rather than
    # returned empty so the portfolio consumers render SOMETHING true for it.
    if plan.get("type") == "ideation":
        for exp in plan.get("proposed_experiments") or []:
            if isinstance(exp, dict) and (exp.get("experiment_name")
                                          or exp.get("hypothesis")):
                return [{"id": "D1",
                         "title": exp.get("experiment_name") or "Direction",
                         "hypothesis": exp.get("hypothesis"),
                         "rationale": exp.get("justification"),
                         "_synthesised": True}]
    return []


def portfolio_to_experiment_shim(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Give a portfolio the minimal experiment shape its legacy readers need.

    Transitional. Fifty-odd sites read `proposed_experiments` — validity
    checks, counters, log rationales, the best-of-N judge, the HTML report.
    Rewriting them all at once would leave every intermediate state a
    half-migration, so a portfolio carries BOTH shapes: `directions` is the
    real payload, and this one-entry shim keeps the legacy readers correct
    rather than empty-handed. Removed once nothing depends on it.
    """
    dirs = plan_directions(plan)
    if not dirs or plan.get("proposed_experiments"):
        return plan
    titles = [str(d.get("title") or d.get("id") or "").strip()
              for d in dirs]
    plan["proposed_experiments"] = [{
        "experiment_name": (plan.get("portfolio_title")
                            or "Research portfolio"),
        "hypothesis": plan_thesis(plan),
        "concepts": dirs,
        "experimental_steps": plan.get("shared_protocol") or [],
        "required_equipment": plan.get("required_equipment") or [],
        "expected_outcome": plan.get("expected_outcome") or "",
        "justification": plan.get("rationale") or "",
        "source_documents": plan.get("source_documents") or [],
        "_portfolio_shim": True,
        "_direction_titles": [t for t in titles if t],
    }]
    return plan


def resync_portfolio(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Keep a portfolio's two copies of its directions consistent.

    Through the transition a portfolio carries `directions` (the payload) and
    a shimmed copy under `proposed_experiments[0].concepts` (for the legacy
    readers). Every pass that RE-EMITS a plan — conformance, critic,
    refinement — is shown the whole dict and told to return the same
    structure, and those prompts name `proposed_experiments` explicitly. So
    the nested copy is the one that reliably gets edited, while a stale
    top-level `directions` would silently outrank it in plan_directions.

    The nested copy therefore wins whenever both exist and differ. On the
    authoring path they are identical and this is a no-op.
    """
    if not isinstance(plan, dict) or not plan.get("directions"):
        return plan
    nested = []
    for exp in plan.get("proposed_experiments") or []:
        if isinstance(exp, dict):
            nested.extend(c for c in (exp.get("concepts") or [])
                          if isinstance(c, dict))
    if nested and nested != plan["directions"]:
        plan["directions"] = nested
    return plan
