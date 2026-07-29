import os
import re
import shutil
import logging
import importlib.util
from typing import Optional, List, Dict

import numpy as np

from datetime import datetime

from ...auth import get_api_key 

try:
    from ase.io import read as ase_read
    from ase.io import write as ase_write
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    # Define dummy functions to avoid runtime errors if ASE is not installed
    def ase_read(*args, **kwargs):
        raise ImportError("ASE is not installed. Please install it to use this functionality.")
    def ase_write(*args, **kwargs):
        raise ImportError("ASE is not installed. Please install it to use this functionality.")


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


# OpenAI-compatible JSONSchema describing the MP search tool exposed to the
# StructureGenerator's pre-script-gen resolution step. Kept here next to the
# helper so the tool definition and its implementation stay in one place.
MP_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_material_id",
        "description": (
            "Look up a Materials Project ID (mp-id) for a chemical formula or "
            "chemical system. Use this whenever a structure-building request "
            "names a specific material (e.g., 'rutile TiO2', 'monoclinic HfO2', "
            "'graphene', 'NaCl', 'GaAs'). Returns the mp-id of the most stable "
            "polymorph matching the query, plus its canonical formula, space "
            "group, and energy above hull. Do not call for generic requests "
            "like 'a Lennard-Jones solid' or 'a 4-atom cubic cell'.\n\n"
            "When the request names a polymorph by name (e.g., 'rutile TiO2', "
            "'anatase TiO2', 'wurtzite GaN', 'monoclinic HfO2', 'cubic SiC'), "
            "you MUST set `spacegroup_symbol` or `crystal_system` to "
            "disambiguate — otherwise the search returns the lowest-energy "
            "polymorph, which may not be the one the user asked for."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chemical_query": {
                    "type": "string",
                    "description": (
                        "A chemical formula like 'TiO2', 'GaAs', 'LiCoO2' "
                        "(use search_type='formula'), or a chemical system "
                        "like 'Fe-S-O' (use search_type='chemsys')."
                    ),
                },
                "search_type": {
                    "type": "string",
                    "enum": ["formula", "chemsys"],
                    "description": (
                        "'formula' (default) finds the most stable polymorph "
                        "for a specific stoichiometry. 'chemsys' finds the "
                        "most stable material across all stoichiometries "
                        "within the given elements."
                    ),
                },
                "spacegroup_symbol": {
                    "type": "string",
                    "description": (
                        "Optional Hermann–Mauguin space-group symbol to "
                        "disambiguate polymorphs. Set this when the request "
                        "names a specific polymorph: 'rutile TiO2' → "
                        "'P4_2/mnm', 'anatase TiO2' → 'I4_1/amd', 'brookite "
                        "TiO2' → 'Pbca', 'wurtzite GaN' → 'P6_3mc', "
                        "'zincblende GaAs' → 'F-43m', 'rocksalt NaCl' → "
                        "'Fm-3m'. Use underscores for subscripts (P4_2, "
                        "P6_3, etc.). Omit when only a stoichiometry is "
                        "given without a polymorph qualifier."
                    ),
                },
                "crystal_system": {
                    "type": "string",
                    "enum": [
                        "triclinic", "monoclinic", "orthorhombic",
                        "tetragonal", "trigonal", "hexagonal", "cubic",
                    ],
                    "description": (
                        "Optional crystal-system filter — less specific than "
                        "spacegroup_symbol. Use when only the crystal family "
                        "is named (e.g., 'cubic SiC', 'hexagonal BN') and "
                        "you don't have a definite space group. Prefer "
                        "spacegroup_symbol when you know it."
                    ),
                },
            },
            "required": ["chemical_query"],
        },
    },
}


class MaterialsProjectHelper:
    """Minimal MP helper for automatic material resolution by searching for mp-ids."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key('materials_project') or os.getenv("MP_API_KEY")
        self.enabled = MP_API_AVAILABLE and bool(self.api_key)
        self.logger = logging.getLogger(__name__)

        # Per-instance cache of MP record lookups. Keyed by (chemical_query,
        # search_type, spacegroup_symbol, crystal_system). Survives the
        # lifetime of the helper — when the orchestrator reuses the same
        # StructureGenerator across calls, the same MP query doesn't hit
        # the network twice in a session.
        self._record_cache: Dict[tuple, Optional[Dict]] = {}

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
        info += "    Once an mp-id is known, fetch the structure via pymatgen and convert to ASE Atoms.\n"
        info += "    The MP_API_KEY environment variable is set automatically inside the sandbox:\n"
        info += "    ```python\n"
        info += "    import os\n"
        info += "    from mp_api.client import MPRester\n"
        info += "    from pymatgen.io.ase import AseAtomsAdaptor\n"
        info += "    with MPRester(os.getenv(\"MP_API_KEY\")) as mpr:\n"
        info += "        structure = mpr.get_structure_by_material_id(\"mp-149\")\n"
        info += "    atoms = AseAtomsAdaptor.get_atoms(structure)\n"
        info += "    ```\n"
        info += "\n"
        info += "**Guidance for LLM:** When a material is mentioned:\n"
        info += "    a. Check the common list above. If found, note the mp-id for use.\n"
        info += "    b. If not in the common list, state that a search using `search_material_id` is required, specifying the `chemical_query` and appropriate `search_type` for the system to perform.\n"

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

    def search_material_record(self, chemical_query: str, search_type: str = "formula",
                               spacegroup_symbol: Optional[str] = None,
                               crystal_system: Optional[str] = None) -> Optional[Dict]:
        """
        Like ``search_material_id`` but returns the full best-match record:
        ``{"material_id", "formula_pretty", "energy_above_hull",
        "spacegroup_symbol"}``, or None.

        Optional ``spacegroup_symbol`` / ``crystal_system`` narrow the search
        to a specific polymorph — required when the request specifies one
        (e.g., 'rutile TiO2'), since plain formula search returns whichever
        polymorph has the lowest e_above_hull regardless of which the user
        asked for.

        Used by the structure agent's pre-script tool-resolution step so the
        LLM can verify it got the polymorph it asked for without a second
        round-trip to MP.

        Results are cached per (query, search_type, spacegroup, crystal_system)
        for the lifetime of this helper instance — so when the orchestrator
        reuses the same StructureGenerator across `generate_structure` calls
        for variants of the same material, the second and subsequent lookups
        hit the cache rather than the MP API.
        """
        if not self.enabled or not chemical_query:
            return None

        cache_key = (chemical_query, search_type, spacegroup_symbol, crystal_system)
        if cache_key in self._record_cache:
            self.logger.info(f"MP cache hit for {cache_key}")
            return self._record_cache[cache_key]

        self.logger.info(
            f"Searching MP record for '{chemical_query}' ({search_type}"
            + (f", spacegroup={spacegroup_symbol}" if spacegroup_symbol else "")
            + (f", crystal_system={crystal_system}" if crystal_system else "")
            + ")"
        )
        try:
            with MPRester(self.api_key) as mpr:
                fields = ["material_id", "energy_above_hull", "formula_pretty", "symmetry"]
                kwargs = {"fields": fields}
                if spacegroup_symbol:
                    kwargs["spacegroup_symbol"] = spacegroup_symbol
                if crystal_system:
                    # MP API expects the CrystalSystem enum value (capitalized),
                    # but accepts the lowercase string too via pymatgen's coercion.
                    kwargs["crystal_system"] = crystal_system.capitalize()

                if search_type == "formula":
                    results = mpr.materials.summary.search(
                        formula=chemical_query, **kwargs,
                    )
                elif search_type == "chemsys":
                    results = mpr.materials.summary.search(
                        chemsys=chemical_query, **kwargs,
                    )
                else:
                    self.logger.error(f"Invalid search_type: {search_type}")
                    # Don't cache invalid-input failures (caller can fix and retry).
                    return None

                if not results:
                    self._record_cache[cache_key] = None
                    return None

                best = sorted(
                    results,
                    key=lambda x: (
                        float('inf') if x.energy_above_hull is None else x.energy_above_hull,
                        x.material_id,
                    ),
                )[0]

                # `symmetry` is a Symmetry pydantic model with .symbol, .number,
                # .crystal_system. Extract the HM symbol if present.
                sg_symbol = None
                sym = getattr(best, "symmetry", None)
                if sym is not None:
                    sg_symbol = getattr(sym, "symbol", None)

                record = {
                    "material_id": str(best.material_id),
                    "formula_pretty": best.formula_pretty,
                    "energy_above_hull": best.energy_above_hull,
                    "spacegroup_symbol": sg_symbol,
                }
                self._record_cache[cache_key] = record
                return record
        except Exception as e:
            self.logger.error(
                f"MP search_material_record error for '{chemical_query}': {e}",
                exc_info=True,
            )
            # Don't cache transient API/network errors — let the next call retry.
            return None


# Optional structure-building libraries the codegen LLM may reach for. Probed
# at generation time so the prompt can tell the model what is actually
# importable here instead of guessing — without an availability signal the model
# defaults to dependency-free hand-rolled code even when a better purpose-built
# tool is installed. This is a factual capability list; which tool to prefer for
# a given task is the skill's domain guidance, stated there, not here.
_OPTIONAL_BUILD_TOOLS = ("pymatgen", "rdkit", "openmm", "mbuild")


def detect_structure_build_tools() -> Dict[str, bool]:
    """Return {tool_name: importable} for optional structure-building libraries.

    Fail-closed: a tool whose probe raises is reported unavailable. ``packmol``
    requires both its pymatgen wrapper and the ``packmol`` executable on PATH —
    the wrapper alone cannot pack a box.
    """
    available: Dict[str, bool] = {}
    for name in _OPTIONAL_BUILD_TOOLS:
        try:
            available[name] = importlib.util.find_spec(name) is not None
        except Exception:
            available[name] = False
    try:
        wrapper = importlib.util.find_spec("pymatgen.io.packmol") is not None
    except Exception:
        wrapper = False
    available["packmol"] = bool(wrapper and shutil.which("packmol"))
    return available


def format_available_tools_block(available: Dict[str, bool]) -> str:
    """Render an availability dict into a prompt block, or '' if nothing is
    available (in which case the prompt is left unchanged)."""
    present = sorted(name for name, ok in available.items() if ok)
    if not present:
        return ""
    return (
        "\n\n## AVAILABLE LIBRARIES (importable in this environment):\n"
        f"{', '.join(present)}\n"
        "When one of these purpose-built libraries fits the task, prefer it "
        "over a hand-rolled implementation — it handles edge cases that ad-hoc "
        "code commonly gets wrong. Fall back to manual construction only when "
        "no listed library covers the operation.\n"
    )


def save_generated_script(script_content: str, description: str, attempt: int, output_dir: str) -> str | None:
    """Saves the script content to a file and returns the path."""
    try:
        # Ensure directory exists (might be better done once on agent init)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize description for use in filename: collapse each run of
        # non-alphanumeric chars into a single underscore (so "a, b" / "a. b"
        # don't become "a__b"), strip edges, then cap the length.
        safe_desc = re.sub(r"[^0-9A-Za-z]+", "_", description).strip("_")[:40]
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
    print(f"  [r] REFINE   - Attempt to fix issues")
    
    while True:
        try:
            choice = input("Choice [p/r]: ").strip().lower()
            if choice in ['p', 'proceed']: return 'proceed'
            elif choice in ['r', 'refine']: return 'refine'
            else: print("Please enter 'p' or 'r'")
        except (KeyboardInterrupt, EOFError):
            return 'refine'


def generate_structure_views(structure_path: str, output_dir: str = None) -> Dict[str, str]:
    """Render PNG views of a structure for the validator and the user.

    Picks rotations adaptively via ``_get_optimal_rotations``: slab
    structures get a top-down + two edge views (so stacking is visible),
    layered structures get layer-perpendicular views, anisotropic cells
    get views aligned with the principal directions, and everything else
    falls back to plain X/Y/Z orthogonal views.

    The dict keys are semantic labels ('surface', 'edge1', 'layers', 'x',
    etc.) — they're surfaced to the validator as "View ({label}):" so the
    multimodal prompt knows which view it's looking at.
    """
    if not ASE_AVAILABLE:
        logging.warning("ASE not found, skipping image generation for validation.")
        return {}

    logger = logging.getLogger(__name__)
    image_paths = {}

    try:
        atoms = ase_read(structure_path)
        atoms = _center_structure_for_visualization(atoms)
    except Exception as e:
        logger.error(f"Failed to read structure file {structure_path} with ASE: {e}")
        return image_paths

    if output_dir is None:
        output_dir = os.path.dirname(structure_path) or '.'
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(structure_path))[0]

    rotations = _get_optimal_rotations(atoms)
    logger.info(
        f"Generating {len(rotations)} structure view image(s) for "
        f"{structure_path} (view labels: {list(rotations)})..."
    )
    # ASE PNG render tuning (Option 1 from the visualization-practices
    # discussion): smaller atomic spheres so underlying lattice topology
    # is visible (oversized spheres were a big factor in why broken
    # honeycombs looked superficially OK), draw cell box for orientation
    # cues, and bump the pixels-per-Å scale so fine features are legible.
    render_kwargs = dict(
        radii=0.5,        # half-default; prevents sphere overlap from hiding pattern
        show_unit_cell=2, # draw cell edges (0=none, 1=plain, 2=fancy)
        scale=20,         # pixels per Å — default is ~10
    )
    for label, rotation in rotations.items():
        try:
            output_path = os.path.join(output_dir, f"{base_name}_view_{label}.png")
            ase_write(output_path, atoms, format='png',
                      rotation=rotation, **render_kwargs)
            image_paths[label] = output_path
            logger.info(f"Saved structure view: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write image for {label} view: {e}")

    return image_paths

def _center_structure_for_visualization(atoms):
    """Center structure in cell for better visualization"""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    # Center of mass
    center_of_mass = positions.mean(axis=0)
    
    # Cell center  
    cell_center = cell.sum(axis=0) / 2
    
    # Shift to center
    shift = cell_center - center_of_mass
    atoms.translate(shift)
    atoms.wrap()  # Keep atoms in cell
    
    return atoms

def _get_optimal_rotations(atoms):
    """Automatically determine optimal viewing angles with simple heuristics"""
    
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    
    # Quick checks using existing ASE functionality
    
    # 1. Check for slab structure (large vacuum gap)
    if _is_slab_structure(cell, positions):
        return _get_slab_rotations(cell)
    
    # 2. Check for highly anisotropic cell
    elif _is_anisotropic(cell):
        return _get_anisotropic_rotations(cell)
    
    # 3. Check for layered structure
    elif _is_layered_structure(atoms):
        return _get_layered_rotations(cell)
    
    # 4. Default: use orthogonal views
    else:
        return {
            'x': '0y,90x,0z',
            'y': '-90x,0y,0z', 
            'z': '0x,0y,0z'
        }

def _is_slab_structure(cell, positions):
    """Detect slab by checking for a >5 Å vacuum gap along any axis.

    Compares each cell length against the atomic extent along the SAME
    axis (not just z), so slabs with vacuum along a or b are detected
    correctly and anisotropic crystals without vacuum aren't false-
    positive."""
    cell_lengths = np.linalg.norm(cell, axis=1)
    extents = positions.max(axis=0) - positions.min(axis=0)
    return any(length > extent + 5.0 for length, extent in zip(cell_lengths, extents))

def _get_slab_rotations(cell):
    """Views optimized for slab structures"""
    # Find the longest cell vector (likely the vacuum direction)
    lengths = np.linalg.norm(cell, axis=1)
    vacuum_dir = np.argmax(lengths)
    
    if vacuum_dir == 2:  # vacuum along z
        return {
            'surface': '0x,0y,0z',      # Top view of surface
            'edge1': '90x,0y,0z',       # Edge view 1  
            'edge2': '0x,90y,0z'        # Edge view 2
        }
    elif vacuum_dir == 1:  # vacuum along y
        return {
            'surface': '90x,0y,0z',
            'edge1': '0x,0y,0z', 
            'edge2': '0x,0y,90z'
        }
    else:  # vacuum_dir == 0, vacuum along x
        return {
            'surface': '0x,90y,0z',
            'edge1': '0x,0y,0z',
            'edge2': '90x,0y,0z'
        }

def _is_anisotropic(cell):
    """Check if cell is highly anisotropic"""
    lengths = np.linalg.norm(cell, axis=1)
    ratio = lengths.max() / lengths.min()
    return ratio > 2.0  # If one direction is 2x larger than another

def _get_anisotropic_rotations(cell):
    """Views aligned with cell vectors for anisotropic structures"""
    # Find principal directions
    lengths = np.linalg.norm(cell, axis=1)
    
    # Sort by length to identify short/long directions
    sorted_indices = np.argsort(lengths)
    
    return {
        'along_short': f'0x,0y,{90 if sorted_indices[0] != 2 else 0}z',
        'along_medium': f'{90 if sorted_indices[1] == 0 else 0}x,{90 if sorted_indices[1] == 1 else 0}y,0z',
        'along_long': f'{90 if sorted_indices[2] == 0 else 0}x,{90 if sorted_indices[2] == 1 else 0}y,{90 if sorted_indices[2] == 2 else 0}z'
    }

def _is_layered_structure(atoms):
    """Simple check for layered materials based on z-coordinates"""
    positions = atoms.get_positions()
    z_coords = positions[:, 2]
    
    # Check if atoms form distinct layers (gaps > 2 Å between groups)
    sorted_z = np.sort(z_coords)
    gaps = np.diff(sorted_z)
    
    return np.any(gaps > 2.0)  # Significant gaps indicate layers

def _get_layered_rotations(cell):
    """Views for layered structures"""
    return {
        'layers': '90x,0y,0z',      # Side view to see layers
        'in_plane': '0x,0y,0z',     # Top view of layer plane
        'oblique': '45x,0y,45z'     # Oblique view for 3D perspective
    }



