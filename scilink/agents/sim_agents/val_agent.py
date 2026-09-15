import os
import re
import logging
import json
from typing import Dict, List, Optional

from ase.io import read as ase_read


# Phrases that indicate the validator LLM is *confirming* a feature rather than
# flagging an issue. Entries containing any of these phrases are filtered out
# of identified_issues_detail before the success/needs-correction decision.
_CONFIRMATION_PATTERNS = (
    r"\bthis is (?:actually )?correct\b",
    r"\bnot an? issue\b",
    r"\b(?:correctly|successfully) (?:placed|created|removed|implemented|matches)",
    r"\bconfirm(?:s|ed|ing)\b",
    r"\bmatches the (?:user'?s? )?request\b",
    r"\bis expected\b",
    r"\bas expected\b",
    r"\bexpected and (?:matches|consistent)",
    r"\bphysically (?:sound|reasonable|valid|correct)",
    r"\bvalidates? the\b",
    r"\bthis is fine\b",
    r"\bis (?:also |actually )?(?:fine|acceptable|appropriate)\b",
    r"\bno (?:real )?issue\b",
    r"\bcosmetic(?:ally)? (?:only|unusual|issue)?\b",
    r"\bnot physically wrong\b",
    r"\bthis should not cause issues\b",
    r"\bcorrect behavior\b",
    r"\bcorrect (?:and|behavior|implementation|coordination|stoichiometry)\b",
)
_CONFIRMATION_RE = re.compile("|".join(_CONFIRMATION_PATTERNS), re.IGNORECASE)


def _filter_confirmations(issues: List[str]) -> List[str]:
    """Drop entries that look like confirmations / positive observations
    rather than actionable problems. Used to stop the structure-refinement
    loop from spinning on cosmetic remarks the validator put in its issue list.
    """
    out = []
    for item in issues or []:
        text = item if isinstance(item, str) else str(item)
        if not text.strip():
            continue
        if _CONFIRMATION_RE.search(text):
            continue
        out.append(item)
    return out

# Replaced google.generativeai imports with wrappers
from ...auth import (
    APIKeyNotFoundError, get_api_key, get_internal_proxy_key, infer_provider,
    require_vendor_credentials,
)
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from .instruct import VALIDATOR_PROMPT_TEMPLATE
from .utils import generate_structure_views
from ._deprecation import normalize_params


class StructureValidatorAgent:
    def __init__(self, api_key: str = None, 
                 model_name: str = "gemini-3.1-pro-preview", 
                 base_url: Optional[str] = None,
                 # Legacy params
                 local_model: str = None,
                 google_api_key: str = None):
        
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="StructureValidatorAgent"
        )
        
        if base_url:
            # Internal Proxy
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            self.logger.info(f"StructureValidatorAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public / LiteLLM — delegate model→provider→env-var resolution
            # to LiteLLM (works for any model LiteLLM supports; raises a
            # message naming the missing vendor env var if not).
            if api_key is None:
                require_vendor_credentials(model_name)
            self.logger.info(f"StructureValidatorAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
            
        # We rely on the prompt to enforce JSON, 
        # rather than the generation config.
        self.generation_config = None

    def _compute_structure_stats(self, structure_file_path: str) -> str:
        """Compute and format structured stats for the validator's prompt.

        Aim: give the validator concrete numbers (composition, cell,
        coordination, vacuum thickness, min interatomic distance) that it
        can verify against expectations directly — not infer from PNG
        renders. The PNG renderer is low-res and bond-free; for similar-
        looking lattices (honeycomb vs close-packed vs distorted) the
        validator can't reliably tell from images alone.

        Returns a markdown block to inject into the validator prompt, or
        empty string on parse failure (validator falls through to the
        existing image+file-content path).
        """
        try:
            from collections import Counter
            import numpy as np
            from ase.io import read as ase_read
            from ase.neighborlist import NeighborList
        except Exception as e:
            self.logger.warning(f"Could not import deps for stats: {e}")
            return ""

        try:
            atoms = ase_read(structure_file_path)
        except Exception as e:
            self.logger.warning(f"Could not parse structure for stats: {e}")
            return ""

        lines = ["", "## STRUCTURE STATS (computed from the structure file — verify these against expectations):"]

        syms = atoms.get_chemical_symbols()
        comp = Counter(syms)
        comp_str = ", ".join(f"{el}: {n}" for el, n in sorted(comp.items()))
        lines.append(f"- Composition: {comp_str} (total {len(atoms)} atoms)")

        try:
            a, b, c = atoms.cell.lengths()
            alpha, beta, gamma = atoms.cell.angles()
            lines.append(
                f"- Cell: a={a:.3f}, b={b:.3f}, c={c:.3f} Å; "
                f"α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°"
            )
        except Exception:
            pass

        # Per-species z-extents — useful for slab/interface stacking checks
        try:
            pos = atoms.get_positions()
            if len(comp) > 1:
                lines.append("- z-extent per species (min, max in Å):")
                for el in sorted(comp.keys()):
                    mask = np.array([s == el for s in syms])
                    z = pos[mask, 2]
                    lines.append(f"    {el}: [{z.min():.2f}, {z.max():.2f}]")
        except Exception:
            pass

        # Vacuum (cell length minus atomic extent) per axis
        try:
            extents = pos.max(axis=0) - pos.min(axis=0)
            cell_lengths = atoms.cell.lengths()
            vac = cell_lengths - extents
            lines.append(
                f"- Vacuum (cell − atomic extent): a={vac[0]:.1f}, "
                f"b={vac[1]:.1f}, c={vac[2]:.1f} Å"
            )
        except Exception:
            pass

        # Min pairwise distance — catches atom overlap / unphysical bonds.
        # Use a KD-tree (O(N log N) memory) rather than the dense N×N
        # get_all_distances matrix, which OOMs for large systems (a 20k-atom
        # solvated protein needs ~3.4 GB and gets SIGKILLed).
        try:
            from scipy.spatial import cKDTree
            pos = atoms.get_positions()
            boxsize = None
            if bool(atoms.pbc.all()) and atoms.cell.orthorhombic:
                boxsize = atoms.cell.lengths()
                pos = pos % boxsize          # cKDTree(boxsize=...) needs coords in [0, L)
            tree = cKDTree(pos, boxsize=boxsize)
            nn = tree.query(pos, k=2)[0][:, 1]   # k=1 is the atom itself
            min_d = float(nn.min())
            lines.append(f"- Min pairwise distance{' (PBC)' if boxsize is not None else ''}: {min_d:.3f} Å")
        except Exception:
            pass

        # Nearest-neighbor coordination distribution — catches lattice topology
        # errors (honeycomb vs close-packed, broken bonds, etc.)
        try:
            cutoffs = [0.85] * len(atoms)  # ~1.7 Å pair cutoff, generous
            nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
            nl.update(atoms)
            coord_counts = []
            for i in range(len(atoms)):
                indices, _ = nl.get_neighbors(i)
                coord_counts.append(len(indices))
            arr = np.array(coord_counts)
            mean_c = float(arr.mean())
            dist = Counter(arr.tolist())
            dist_str = ", ".join(f"{n}NN: {c} atoms" for n, c in sorted(dist.items()))
            lines.append(
                f"- Coordination (within 1.7 Å): mean={mean_c:.2f}; {dist_str}"
            )
        except Exception:
            pass

        # All atoms within cell?
        try:
            fracs = atoms.get_scaled_positions(wrap=False)
            in_cell = bool(((fracs >= -1e-6) & (fracs < 1 + 1e-6)).all())
            lines.append(f"- All atoms within cell (no wrap needed): {in_cell}")
        except Exception:
            pass

        return "\n".join(lines) + "\n"

    def _read_structure_file_content(self, structure_file_path: str) -> str:
        """
        Read the raw content of the structure file for LLM analysis.
        """
        try:
            with open(structure_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Limit content size for LLM context window
            max_chars = 10000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[... File truncated for context limits ...]"
                self.logger.warning(f"Structure file content truncated from {len(content)} to {max_chars} characters")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to read structure file content: {e}")
            return f"Error reading file: {str(e)}"

    def _check_file_parsable_and_load(self, structure_file_path: str) -> tuple[object | None, list, str]:
        """
        Checks if the structure file is parsable by ASE and reads raw content for LLM.
        """
        try:
            atoms = ase_read(structure_file_path)
            if not atoms or len(atoms) == 0:
                msg = f"Structure file '{structure_file_path}' is empty or could not be parsed into a valid ASE Atoms object."
                self.logger.warning(msg)
                return None, [msg], ""
            
            self.logger.info(f"Successfully parsed structure file: {structure_file_path}, {len(atoms)} atoms.")
            
            # Read raw file content for LLM analysis
            file_content = self._read_structure_file_content(structure_file_path)
            
            return atoms, [], file_content
            
        except Exception as e:
            self.logger.error(f"Failed to parse structure file '{structure_file_path}' with ASE: {e}")
            return None, [f"ASE could not parse structure file '{structure_file_path}'. Error: {e}"], ""


    def _get_llm_validation_and_hints(self, original_request: str, generating_script_content: str,
                                      structure_file_path: str, structure_file_content: str = "",
                                      image_paths: Dict[str, str] = None,
                                      tool_documentation: str = None,
                                      validation_rubric: str = None) -> dict:
        """
        Uses an LLM to perform full validation including analysis of the actual structure file content and images.

        ``validation_rubric`` (optional) carries the structure-class-specific
        acceptance criteria (the ``## Validation`` section of a
        structure_generation skill, e.g. crystal / molecular / condensed /
        biomolecular). The base prompt is class-neutral; this rubric supplies the
        authoritative, class-specific standard so e.g. a molecule isn't judged by
        periodic-crystal supercell/vacuum expectations.
        """

        # Format tool documentation section
        doc_section = ""
        if tool_documentation:
            doc_section = f"""
    ## SPECIALIZED LIBRARY DOCUMENTATION:
    {tool_documentation}

    Please consider this documentation when validating the structure and providing script modification hints.
    Use the proper syntax, classes, and methods shown in the documentation above when suggesting improvements.

    """

        # Format structure-class validation rubric (authoritative per-class criteria)
        rubric_section = ""
        if validation_rubric:
            rubric_section = f"""

## STRUCTURE-CLASS VALIDATION CRITERIA (authoritative for this structure):
{validation_rubric}
"""
        
        # Format structure file content section
        structure_section = ""
        if structure_file_content:
            structure_section = f"""

    ## ACTUAL STRUCTURE FILE CONTENT:
    ```
    {structure_file_content}
    ```

    """

        # Compute structured stats (composition, cell, coordination, vacuum
        # thickness, etc.) so the validator can verify them numerically
        # rather than inferring from low-resolution PNG renders. Catches
        # cases where the renderer's eyeball look is misleading — e.g., a
        # broken honeycomb that looks vaguely hexagonal in PNG but has
        # mean coordination ≠ 3.
        stats_section = self._compute_structure_stats(structure_file_path)

        base_prompt = VALIDATOR_PROMPT_TEMPLATE.format(
            tool_documentation=doc_section,
            original_request=original_request,
            generating_script_content=generating_script_content,
            structure_file_path=structure_file_path,
        ) + rubric_section + stats_section + structure_section

        # --- Build multi-modal prompt ---
        # The wrapper handles list inputs [text, image, image...]
        prompt_parts = [base_prompt]
        
        if image_paths:
            self.logger.info("Adding structure view images to validation prompt.")
            prompt_parts.append("\n\n## STRUCTURE VISUALIZATION:\n")
            # Read images into bytes and append. Labels are 'x'/'y'/'z' for
            # plain axis views or semantic ('surface', 'edge1', 'layers',
            # 'along_short', etc.) for adaptive views from
            # _get_optimal_rotations — render the label appropriately.
            for label, img_path in sorted(image_paths.items()):
                pretty_label = (
                    f"{label.upper()}-axis"
                    if label in ("x", "y", "z")
                    else label.replace("_", " ")
                )
                try:
                    with open(img_path, 'rb') as f:
                        img_bytes = f.read()
                    prompt_parts.append(f"View ({pretty_label}):")
                    # Wrapper format: {"mime_type": "...", "data": bytes}
                    prompt_parts.append({"mime_type": "image/png", "data": img_bytes})
                except Exception as e:
                    self.logger.error(f"Could not read image file {img_path} for prompt: {e}")
                    prompt_parts.append(f"(Error loading image for {pretty_label} view)")

        
        self.logger.info("Sending request to Validator LLM for full validation and script hints...")
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
            )
            
            raw_text = response.text
            first_brace = raw_text.find('{')
            last_brace = raw_text.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_string = raw_text[first_brace : last_brace + 1]
                try:
                    llm_feedback = json.loads(json_string)
                    self.logger.info("LLM full validation feedback and script hints received successfully.")
                    if not all(k in llm_feedback for k in ["overall_assessment", "identified_issues_detail", "script_modification_hints"]):
                        self.logger.warning("LLM feedback JSON is missing one or more expected keys.")
                        return {
                            "overall_assessment": llm_feedback.get("overall_assessment", "LLM assessment incomplete (missing keys)."),
                            "identified_issues_detail": llm_feedback.get("identified_issues_detail", ["LLM feedback structure error: missing 'identified_issues_detail'."]),
                            "script_modification_hints": llm_feedback.get("script_modification_hints", [])
                        }
                    return llm_feedback
                except json.JSONDecodeError as e_json:
                    self.logger.error(f"Failed to decode JSON from LLM response substring. Error: {e_json}. Substring: '{json_string[:200]}...'")
                    error_msg = f"LLM response could not be parsed as JSON: {e_json}"
            else:
                self.logger.error(f"Could not find valid JSON object delimiters '{{' and '}}' in LLM response. Raw text: {raw_text[:500]}...")
                error_msg = "LLM response did not contain a recognizable JSON object."

            return {
                "overall_assessment": "Error: Failed to get valid structured feedback from LLM.",
                "identified_issues_detail": [error_msg],
                "script_modification_hints": []
            }

        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during LLM call for validation/hints: {e}")
            return {
                "overall_assessment": "Critical Error: LLM call for validation failed.",
                "identified_issues_detail": [f"Unexpected error during LLM communication: {e}"],
                "script_modification_hints": []
            }

    def validate_structure_and_script(self, structure_file_path: str, generating_script_content: str,
                                      original_request: str, tool_documentation: str = None,
                                      validation_rubric: str = None) -> dict:
        """
        Main validation method. Generates images and relies on LLM for all checks.
        Returns a dictionary with validation status, issues, and script modification hints.

        ``validation_rubric`` (optional) supplies structure-class-specific
        acceptance criteria injected as authoritative guidance (see
        ``_get_llm_validation_and_hints``).
        """
        self.logger.info(f"Starting LLM-based validation with structure file analysis for '{structure_file_path}'")
    
        final_feedback = {
            "status": "error", 
            "original_structure_file": structure_file_path,
            "overall_assessment": "Validation did not complete.",
            "all_identified_issues": [], 
            "script_modification_hints": []
        }

        # 1. Load structure and read file content for LLM
        atoms_obj, parsing_issues, structure_file_content = self._check_file_parsable_and_load(structure_file_path)
        if atoms_obj is None:
            final_feedback["overall_assessment"] = "Structure file is unparsable or invalid."
            final_feedback["all_identified_issues"] = parsing_issues
            self.logger.error(f"Validation aborted: Structure file unparsable. Issues: {parsing_issues}")
            return final_feedback

        # 2. Generate images for visual validation
        image_paths = generate_structure_views(structure_file_path)
        if not image_paths:
            self.logger.warning("Could not generate images for visual validation. Proceeding with text-only validation.")

        # 3. Get LLM-based validation with actual file content and images
        llm_feedback = self._get_llm_validation_and_hints(
            original_request=original_request,
            generating_script_content=generating_script_content,
            structure_file_path=structure_file_path,
            structure_file_content=structure_file_content, # Pass raw file content
            image_paths=image_paths, # Pass generated image paths
            tool_documentation=tool_documentation,
            validation_rubric=validation_rubric,
        )

        final_feedback["overall_assessment"] = llm_feedback.get("overall_assessment", "LLM assessment missing or failed.")
        # Issues identified by LLM are now the sole source of issues, but the
        # LLM frequently puts confirmations / positive observations in this list
        # ("this is correct", "matches the request", "confirms the vacancy",
        # etc). Filter those out — they should not trigger a refinement cycle.
        raw_issues = llm_feedback.get("identified_issues_detail", [])
        final_feedback["all_identified_issues"] = _filter_confirmations(raw_issues)
        final_feedback["script_modification_hints"] = llm_feedback.get("script_modification_hints", [])

        dropped = len(raw_issues) - len(final_feedback["all_identified_issues"])
        if dropped > 0:
            self.logger.info(
                f"Filtered {dropped} confirmation/positive observation(s) "
                f"from validator's issue list (kept {len(final_feedback['all_identified_issues'])} "
                f"actionable issue(s))."
            )

        if not final_feedback["all_identified_issues"]:
            final_feedback["status"] = "success"
            self.logger.info(f"LLM validation successful for '{structure_file_path}'. No issues reported by LLM.")
        else:
            final_feedback["status"] = "needs_correction"
            self.logger.warning(f"LLM validation for '{structure_file_path}' found issues requiring script correction: {final_feedback['all_identified_issues']}")
            if not final_feedback["script_modification_hints"] and final_feedback["all_identified_issues"]: # Issues exist but no hints
                self.logger.warning("LLM identified issues but provided no script modification hints. Adding a generic hint.")
                final_feedback["script_modification_hints"].append(
                    "LLM identified issues but gave no specific script hints. Review the script against the identified issues."
                )
        
        self.logger.debug(f"Final LLM-only validation feedback for '{structure_file_path}': {final_feedback}")
        return final_feedback


# IncarValidatorAgent moved to ``vasp_input_validator.py`` on 2026-05-17
# as part of separating engine-neutral validators (structure) from
# engine-specific ones (INCAR is VASP).  Re-exported here so existing
# importers (``from .val_agent import IncarValidatorAgent``) keep
# working; new code should import from ``vasp_input_validator`` directly.
# See memory: project_engine_validator_generalize.md for the follow-up
# PR that finishes the split (rename file, drop this shim).
from .vasp_input_validator import IncarValidatorAgent  # noqa: F401
