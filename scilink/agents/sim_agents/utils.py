import os
import logging
from typing import Optional, List

from datetime import datetime


try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False
    # Define a dummy MPRester to avoid runtime errors
    class MPRester: # type: ignore
        def __init__(self, api_key: Optional[str] = None):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


class MaterialsProjectHelper:
    """Minimal MP helper for automatic material resolution by searching for mp-ids."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MP_API_KEY")
        self.enabled = MP_API_AVAILABLE and bool(self.api_key)
        self.logger = logging.getLogger(__name__)

        if self.enabled:
            self.logger.info("Materials Project helper enabled and API key found for mp-id search.")
        else:
            if not MP_API_AVAILABLE:
                self.logger.warning("Materials Project helper disabled: mp-api library not installed.")
            elif not self.api_key:
                self.logger.warning("Materials Project helper disabled: MP_API_KEY not set or provided.")
            else:
                self.logger.warning("Materials Project helper disabled for an unknown reason.")

    def get_common_materials_info(self) -> str:
        """Return info about common materials and the mp-id search capability for LLM context."""
        if not self.enabled: # Added this check for consistency
            return "Materials Project integration is disabled (no API key or mp-api not installed).\n"

        common_materials = {
            'YBCO': 'mp-20674', # YBa2Cu3O7
            'YBa2Cu3O7': 'mp-20674',
            'silicon': 'mp-149', # Elemental Si
            'Si': 'mp-149',
            'copper': 'mp-30', # Elemental Cu
            'Cu': 'mp-30',
            'graphite': 'mp-48', # Carbon
            'C (graphite)': 'mp-48',
            'diamond': 'mp-66', # Carbon
            'C (diamond)': 'mp-66',
            'iron': 'mp-13', # Elemental Fe
            'Fe': 'mp-13',
            'NaCl': 'mp-22862', # Sodium chloride, Rocksalt
            'salt': 'mp-22862',
            'LiFePO4': 'mp-19017', # Lithium iron phosphate, LFP
            'LFP': 'mp-19017',
            'SiO2 (alpha quartz)': 'mp-6930', # Silicon dioxide
            'quartz': 'mp-6930',
            'TiO2 (rutile)': 'mp-2657', # Titanium dioxide
            'rutile': 'mp-2657',
            'MoS2': 'mp-2815', # Molybdenum disulfide (2H phase, common)
            'graphene': 'mp-1040425'
        }

        info = "\n## MATERIALS PROJECT INTEGRATION (for mp-id lookup):\n"
        info += "This system can leverage the Materials Project (MP) database to find Material IDs (mp-ids).\n"
        info += "1.  **Common Materials by Name/Formula:**\n"
        info += "    If a user requests one of the following, you can often directly use its mp-id for structure generation (e.g., in ASE `bulk(mpid='mp-XYZ')`).\n"
        info += "    Known mp-ids:\n"
        for material, mp_id in common_materials.items():
            info += f"    - {material}: {mp_id}\n"
        info += "\n"
        info += "2.  **Searching for Other Material IDs (Conceptual Function):**\n"
        info += "    This helper provides a function to find mp-ids if a material is not in the common list or if a general chemical system is given.\n"
        info += "    - **`search_material_id(chemical_query: str, search_type: str = 'formula') -> Optional[str]`**\n"
        info += "        - Use this to find an mp-id. `chemical_query` can be a formula (e.g., 'LiCoO2', 'GaAs') or a chemical system (e.g., 'Fe-S-O', 'Si-C').\n"
        info += "        - `search_type` can be 'formula' (finds most stable polymorph for a specific stoichiometry) or 'chemsys' (finds the most stable material within the given elements).\n"
        info += "        - Example: If asked for 'gallium arsenide', you should note a search for `search_material_id('GaAs', search_type='formula')` is needed by the system.\n"
        info += "\n"
        info += "3.  **Using mp-ids in Scripts:**\n"
        info += "    Once an mp-id is known (e.g., 'mp-149' for Si), it can be used in ASE scripts like: `atoms = bulk(mpid='mp-149')`.\n"
        info += "\n"
        info += "**Guidance for LLM:** When a material is mentioned:\n"
        info += "   a. Check the common list above. If found, note the mp-id for use.\n"
        info += "   b. If not in the common list, state that a search using `search_material_id` is required, specifying the `chemical_query` and appropriate `search_type` for the system to perform.\n"

        return info

    def search_material_id(self, chemical_query: str, search_type: str = "formula") -> Optional[str]:
        """
        Search for a Materials Project ID given a chemical formula or system.
        Prioritizes the most stable material (lowest energy_above_hull).

        Args:
            chemical_query (str): The chemical formula (e.g., "SiO2") or
                                  chemical system (e.g., "Si-O") to search for.
            search_type (str): Type of search, either "formula" (default) or "chemsys".

        Returns:
            Optional[str]: The mp-id of the most likely material, or None if not found or error.
        """
        if not self.enabled:
            self.logger.warning("MP search_material_id attempted but helper is disabled.")
            return None
        if not chemical_query:
            self.logger.warning("MP search_material_id attempted with an empty query.")
            return None

        self.logger.info(f"Searching MP for '{chemical_query}' using search_type '{search_type}'")
        try:
            with MPRester(self.api_key) as mpr:
                # We only need material_id and energy_above_hull for sorting to find the best match.
                # formula_pretty is useful for logging/verification.
                fields_to_retrieve = ["material_id", "energy_above_hull", "formula_pretty"]
                results = []

                if search_type == "formula":
                    results = mpr.materials.summary.search(
                        formula=chemical_query,
                        fields=fields_to_retrieve
                    )
                elif search_type == "chemsys":
                    results = mpr.materials.summary.search(
                        chemsys=chemical_query,
                        fields=fields_to_retrieve
                    )
                else:
                    self.logger.error(f"Invalid search_type: {search_type}. Must be 'formula' or 'chemsys'.")
                    return None

                if results:
                    # Sort by energy_above_hull (lowest first), treating None as high energy.
                    # Then by material_id as a tie-breaker for reproducibility.
                    sorted_results = sorted(
                        results,
                        key=lambda x: (float('inf') if x.energy_above_hull is None else x.energy_above_hull, x.material_id)
                    )
                    best_match = sorted_results[0]
                    self.logger.info(
                        f"Found mp-id: {best_match.material_id} for {search_type} '{chemical_query}'. "
                        f"Actual Formula: {best_match.formula_pretty}, "
                        f"E_above_hull: {best_match.energy_above_hull:.4f} eV/atom" if best_match.energy_above_hull is not None else "E_above_hull: N/A"
                    )
                    return str(best_match.material_id)
                else:
                    self.logger.warning(f"No material found for {search_type}: {chemical_query}")
                    return None

        except Exception as e:
            self.logger.error(f"Error during Materials Project search for '{chemical_query}': {e}", exc_info=True)
            return None
    

def save_generated_script(script_content: str, description: str, attempt: int, output_dir: str) -> str | None:
    """Saves the script content to a file and returns the path."""
    try:
        # Ensure directory exists (might be better done once on agent init)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize description for use in filename
        safe_desc = "".join(c if c.isalnum() else "_" for c in description[:30]).rstrip("_")
        filename = f"script_{safe_desc}_attempt{attempt}_{timestamp}.py"
        saved_script_path = os.path.join(output_dir, filename)

        with open(saved_script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        logging.info(f"Saved script for attempt {attempt} to: {saved_script_path}")
        return saved_script_path
    except IOError as e:
        logging.error(f"Failed to save script for attempt {attempt}: {e}")
        return None
    except Exception as e: # Catch broader exceptions during save
        logging.error(f"Unexpected error saving script for attempt {attempt}: {e}")
        return None
    

def ask_user_proceed_or_refine(validation_feedback, structure_file):
    """Ask user whether to proceed with current structure or attempt refinement."""
    import sys
    
    print(f"\n--- Validation Issues Found ---")
    issues = validation_feedback.get('all_identified_issues', [])
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\nOptions:")
    print(f"  [p] PROCEED - Use current structure: {structure_file}")
    print(f"  [r] REFINE  - Attempt to fix issues")
    
    while True:
        try:
            choice = input("Choice [p/r]: ").strip().lower()
            if choice in ['p', 'proceed']: return 'proceed'
            elif choice in ['r', 'refine']: return 'refine'
            else: print("Please enter 'p' or 'r'")
        except (KeyboardInterrupt, EOFError):
            return 'refine'