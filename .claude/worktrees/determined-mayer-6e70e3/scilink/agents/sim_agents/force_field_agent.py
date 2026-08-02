import os
import re
import logging
import json
import shutil
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
# MDAnalysis is imported lazily inside _analyze_system_composition so that
# loading scilink.agents.sim_agents (e.g., via DFTOrchestrator) doesn't
# require the LAMMPS-side optional dep.
from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params

# ── SKILL INTEGRATION ── skill loader
from ...skills.loader import load_skill, list_skills

# ── SKILL INTEGRATION ── amber tools (conditional import)
try:
    from ...tools import amber_tools
    _AMBER_TOOLS_AVAILABLE = True
except ImportError:
    _AMBER_TOOLS_AVAILABLE = False


class ForceFieldAgent:
    """
    AI-driven agent for optimal force field selection and parameter acquisition.

    This agent leverages LLMs to analyze molecular systems and research goals, then:
    1. Selects the most appropriate force field based on system composition and research objectives
    2. Determines the best method to obtain parameters (database, manual, QM, etc.)
    3. Executes the chosen parameterization strategy
    4. Validates parameters for scientific rigor

    Now skill-aware: loads domain-specific knowledge from
    scilink/skills/force_field/*.md and uses it as LLM context (RAG)
    to make better decisions. When AmberTools are available and the AMBER
    skill is active, delegates to the full AMBER pipeline
    (antechamber → tleap → ParmEd → LAMMPS data file).

    The agent works in conjunction with other simulation agents in the pipeline.
    """

    def __init__(self,
                 working_dir: str,
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: Optional[str] = None,
                 skill: Optional[str] = None,           # ── SKILL INTEGRATION ──
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initialize the ForceFieldAgent.

        Args:
            working_dir: Directory for output files and intermediate calculations
            api_key: API key for the LLM provider
            model_name: Model name to use
            base_url: Optional base URL for internal proxy
            skill: Optional skill name (e.g., "amber") or path to a .md file.
                   If None, the agent auto-selects based on force field choice.
            local_model: Deprecated, use base_url instead
            google_api_key: Deprecated, use api_key instead
        """
        self.working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        # Normalize deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="ForceFieldAgent"
        )

        # Initialize model using wrapper structure
        if base_url:
            # Internal Proxy
            if api_key is None:
                api_key = get_internal_proxy_key()

            if not api_key:
                raise ValueError("API key required for internal proxy.")
            self.logger.info(f"ForceFieldAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public / LiteLLM
            self.logger.info(f"ForceFieldAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )

        self.generation_config = None

        # Force field databases and methods
        self.ff_databases = {
            "water": ["SPC/E", "TIP3P", "TIP4P", "OPC", "TIP5P"],
            "proteins": ["AMBER ff14SB", "AMBER ff19SB", "CHARMM36m", "OPLS-AA/M"],
            "lipids": ["CHARMM36", "Slipids", "Lipid17", "Martini"],
            "carbohydrates": ["GLYCAM06", "CHARMM36-carb", "GROMOS 56A(CARBO)"],
            "small_molecules": ["GAFF", "GAFF2", "CGenFF", "OpenFF"],
            "ions": ["Joung-Cheatham", "CHARMM", "Åqvist", "Li-Merz"],
            "polymers": ["PCFF", "COMPASS", "OPLS-AA"],
            "metals": ["EAM", "COMB", "ReaxFF"],
            "interfaces": ["INTERFACE-AMBER", "CHARMM-METAL"]
        }

        # Parameter acquisition methods
        self.param_methods = {
            "database": {
                "description": "Direct parameter extraction from established databases",
                "tools": ["ParmEd", "AmberTools", "CGenFF", "MATCH"],
                "strengths": ["Fast", "Reliable for standard molecules"]
            },
            "analogy": {
                "description": "Parameters by chemical analogy to known molecules",
                "tools": ["ffTK", "LigParGen", "CGenFF"],
                "strengths": ["Good balance of speed and accuracy"]
            },
            "quantum": {
                "description": "Ab initio parameterization from QM calculations",
                "tools": ["ffTK", "ForceBalance", "poltype2"],
                "strengths": ["Highest accuracy", "Novel molecules"]
            }
        }
        self._mass_to_element = self._build_mass_lookup()

        # ── SKILL INTEGRATION ── Skill state
        self.skill_name: Optional[str] = None
        self.skill_sections: Optional[Dict[str, str]] = None
        try:
            self._available_ff_skills = list_skills(domain="force_field")
        except Exception:
            self._available_ff_skills = []

        if skill:
            self._load_skill(skill)

    # ================================================================
    # SKILL METHODS
    # ================================================================

    def _load_skill(self, skill: str) -> bool:
        """
        Load a force-field skill by name or path.

        Args:
            skill: Skill name (e.g., "amber") or path to a .md file.

        Returns:
            True if the skill was loaded successfully.
        """
        try:
            parsed = load_skill(skill, domain="force_field")
            self.skill_name = parsed["name"]
            self.skill_sections = parsed
            self.logger.info(f"📖 Loaded force field skill: {self.skill_name}")
            return True
        except FileNotFoundError:
            self.logger.warning(
                f"Skill '{skill}' not found. "
                f"Available: {self._available_ff_skills}"
            )
            return False

    def _auto_select_skill(self, force_field: str) -> bool:
        """
        Automatically select and load the best skill for a force field.

        Args:
            force_field: Force field name from LLM selection.

        Returns:
            True if a skill was loaded.
        """
        if self.skill_sections is not None:
            self.logger.info(f"Skill already loaded: {self.skill_name}")
            return True

        ff_lower = (force_field or "").lower()

        # Mapping of keywords → skill names
        skill_map = {
            "amber": ["amber", "ff14sb", "ff19sb", "ff99sb", "gaff", "lipid17",
                       "lipid21", "glycam", "ol15", "ol3", "bsc1"],
            # Future skills:
            # "charmm": ["charmm", "charmm36", "cgenff"],
            # "opls": ["opls", "opls-aa"],
        }

        for skill_name, keywords in skill_map.items():
            if any(kw in ff_lower for kw in keywords):
                if skill_name in self._available_ff_skills:
                    return self._load_skill(skill_name)
                else:
                    self.logger.info(
                        f"Skill '{skill_name}' would match but is not installed"
                    )

        self.logger.info(f"No matching skill for '{force_field}'")
        return False

    def _get_skill_context(
        self,
        section: Optional[str] = None,
        include_all: bool = False,
    ) -> str:
        """
        Get skill knowledge formatted for LLM prompt injection (RAG).

        Args:
            section: Specific section to retrieve (e.g., "planning", "validation").
                     If None, returns the overview.
            include_all: Include all sections (for complex decisions).

        Returns:
            Formatted context string, or "" if no skill loaded.
        """
        if not self.skill_sections:
            return ""

        if include_all:
            parts = [f"=== Domain Knowledge: {self.skill_name} ==="]
            for key in ("overview", "planning", "analysis", "interpretation",
                        "validation", "implementation"):
                content = self.skill_sections.get(key, "")
                if content:
                    parts += [f"\n--- {key.upper()} ---", content]
            return "\n".join(parts)

        if section:
            content = self.skill_sections.get(section, "")
            if content:
                return (
                    f"=== Domain Knowledge ({self.skill_name} — {section}) ===\n"
                    f"{content}"
                )
            return ""

        # Default: overview
        overview = self.skill_sections.get("overview", "")
        return (
            f"=== Domain Knowledge: {self.skill_name} ===\n{overview}"
            if overview else ""
        )

    def _is_amber_force_field(self, force_field: str) -> bool:
        """Return True if the selected force field is AMBER-family."""
        if not force_field:
            return False
        ff_lower = force_field.lower()
        amber_keywords = [
            "amber", "ff14sb", "ff19sb", "ff99sb", "fb15",
            "gaff", "gaff2", "lipid17", "lipid21",
            "glycam", "ol15", "ol3", "bsc1",
        ]
        return any(kw in ff_lower for kw in amber_keywords)

    # ================================================================
    # AMBER PIPELINE METHODS
    # ================================================================

    def _check_amber_tools_available(self) -> Dict[str, Any]:
        """Check if AmberTools + ParmEd are available."""
        if _AMBER_TOOLS_AVAILABLE:
            return amber_tools.check_amber_tools()

        # Fallback: check manually
        result = {"available": False, "missing": [], "tools": {}}
        for tool in ["antechamber", "tleap", "parmchk2"]:
            path = shutil.which(tool)
            result["tools"][tool] = {"found": path is not None, "path": path}
            if path is None:
                result["missing"].append(tool)
        try:
            import parmed
            result["tools"]["parmed"] = {"found": True}
        except ImportError:
            result["missing"].append("parmed")
        result["available"] = len(result["missing"]) == 0
        return result

    def _run_amber_pipeline(
        self,
        selection_info: Dict[str, Any],
        pdb_file: str,
        small_molecule_info: Optional[List[Dict[str, Any]]] = None,
        solvate: bool = False,
        box_buffer: float = 10.0,
        neutralize: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full AMBER parameterization pipeline using AmberTools.

        Uses skill knowledge to configure the pipeline correctly
        (FF variant, water model, charge method, etc.).

        Args:
            selection_info: Force field selection info from select_force_field().
            pdb_file: Path to PDB structure file.
            small_molecule_info: List of dicts for non-standard residues:
                [{"pdb": <path>, "name": "LIG", "charge": 0}, ...]
            solvate: Add explicit solvent box via tleap.
            box_buffer: Solvent buffer distance (A).
            neutralize: Add counter-ions for charge neutrality.

        Returns:
            Dictionary with paths and metadata.
        """
        self.logger.info("=" * 60)
        self.logger.info("AMBER Parameterization Pipeline")
        self.logger.info("=" * 60)

        ff_info = selection_info.get("force_field", {})
        system_info = selection_info.get("system_info", {})
        composition = system_info.get("composition", {})

        # Extract settings from LLM selection (informed by skill context)
        protein_ff = self._resolve_protein_ff(ff_info.get("force_field", "ff19SB"))
        water_model = ff_info.get("compatible_water_model", "tip3p")
        gaff_version = "gaff2"
        charge_method = self._decide_charge_method(selection_info)

        result: Dict[str, Any] = {
            "pipeline": "amber",
            "source": "amber_pipeline",
            "force_field": protein_ff,
            "force_field_info": ff_info,
            "water_model": water_model,
            "system_info": system_info,
            "skill_used": self.skill_name,
            "atom_types": {},
            "parameter_files": {},
        }

        # Step 1: Clean PDB
        cleaned_pdb = pdb_file
        if shutil.which("pdb4amber"):
            try:
                if _AMBER_TOOLS_AVAILABLE:
                    cleaned_pdb = amber_tools.run_pdb4amber(pdb_file, self.working_dir)
                else:
                    cleaned_pdb = self._run_pdb4amber_inline(pdb_file)
            except Exception as e:
                self.logger.warning(f"pdb4amber failed, using original PDB: {e}")
        result["cleaned_pdb"] = cleaned_pdb

        # Step 2: Parameterize small molecules
        mol2_files: List[Dict[str, Any]] = []
        frcmod_files: List[str] = []
        sm_results: List[Dict[str, Any]] = []

        if small_molecule_info:
            for sm in small_molecule_info:
                sm_file = sm.get("pdb") or sm.get("file")
                sm_name = sm.get("name", "LIG")
                sm_charge = sm.get("charge", 0)

                self.logger.info(f"Parameterizing small molecule: {sm_name}")
                try:
                    if _AMBER_TOOLS_AVAILABLE:
                        ac = amber_tools.run_antechamber(
                            input_file=sm_file, working_dir=self.working_dir,
                            net_charge=sm_charge, charge_method=charge_method,
                            atom_type=gaff_version, output_prefix=sm_name.lower(),
                        )
                        frcmod = amber_tools.run_parmchk2(
                            mol2_file=ac["mol2"], working_dir=self.working_dir,
                            atom_type=gaff_version, output_prefix=sm_name.lower(),
                        )
                    else:
                        ac = self._run_antechamber_inline(
                            sm_file, sm_name.lower(), sm_charge, charge_method, gaff_version
                        )
                        frcmod = self._run_parmchk2_inline(ac["mol2"], sm_name.lower(), gaff_version)

                    mol2_files.append({"mol2": ac["mol2"], "name": sm_name})
                    frcmod_files.append(frcmod)
                    sm_results.append({"name": sm_name, "mol2": ac["mol2"],
                                       "frcmod": frcmod, "status": "success"})
                except Exception as e:
                    self.logger.error(f"Failed to parameterize {sm_name}: {e}")
                    sm_results.append({"name": sm_name, "status": "failed", "error": str(e)})

        result["small_molecule_params"] = sm_results

        # Step 3: Generate and run tleap
        if _AMBER_TOOLS_AVAILABLE:
            script = amber_tools.generate_tleap_script(
                pdb_file=cleaned_pdb, working_dir=self.working_dir,
                composition=composition,
                mol2_files=mol2_files or None, frcmod_files=frcmod_files or None,
                protein_ff=protein_ff, water_model=water_model,
                gaff_version=gaff_version, solvate=solvate,
                box_buffer=box_buffer, neutralize=neutralize,
            )
            prmtop, inpcrd = amber_tools.run_tleap(script, self.working_dir)
        else:
            script = self._generate_tleap_inline(
                cleaned_pdb, composition, mol2_files, frcmod_files,
                protein_ff, water_model, gaff_version, solvate, box_buffer, neutralize,
            )
            prmtop, inpcrd = self._run_tleap_inline(script)

        result["tleap_script"] = script
        result["prmtop"] = prmtop
        result["inpcrd"] = inpcrd

        # Step 4: Convert to LAMMPS data file
        if _AMBER_TOOLS_AVAILABLE:
            data_file = amber_tools.convert_amber_to_lammps(
                prmtop=prmtop, inpcrd=inpcrd,
                output_data=os.path.join(self.working_dir, "system.data"),
            )
        else:
            data_file = self._convert_amber_to_lammps_inline(prmtop, inpcrd)

        result["data_file"] = data_file

        # Step 5: Validate
        validation = self._validate_amber_lammps_data(data_file)
        result["validation"] = validation

        if not validation["valid"]:
            self.logger.error(f"Validation failed: {validation['errors']}")
        else:
            self.logger.info("AMBER → LAMMPS data file validated successfully")

        # Step 6: Generate LAMMPS input header
        result["input_header"] = self._generate_amber_input_header(data_file, ff_info)

        # Step 7: Interpret results using skill
        if validation.get("warnings") or validation.get("errors"):
            result["interpretation"] = self._interpret_amber_results(validation)

        # Save pipeline summary
        summary_file = os.path.join(self.working_dir, "amber_pipeline_summary.json")
        try:
            serializable = {
                k: v for k, v in result.items()
                if isinstance(v, (str, int, float, bool, list, dict, type(None)))
            }
            with open(summary_file, "w") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save pipeline summary: {e}")

        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline complete → {data_file}")
        self.logger.info(f"  Atoms: {validation.get('n_atoms', '?')}")
        self.logger.info(f"  Net charge: {validation.get('total_charge', '?')}")
        self.logger.info("=" * 60)

        return result

    # ── AMBER skill-informed helpers ─────────────────────────────

    def _decide_charge_method(self, selection_info: Dict[str, Any]) -> str:
        """Use skill knowledge + LLM to decide the charge method."""
        skill_context = self._get_skill_context(section="planning")
        if not skill_context:
            return "bcc"

        research_goal = selection_info.get("research_goal", "")
        prompt = f"""
Based on this research goal and domain knowledge, select the best
charge method for antechamber.

RESEARCH GOAL: {research_goal}

{skill_context}

Return ONLY one of: bcc, resp, gas
"""
        try:
            response = self._generate_text(prompt).strip().lower()
            if response in ("bcc", "resp", "gas", "mul"):
                self.logger.info(f"Charge method selected: {response}")
                return response
        except Exception:
            pass
        return "bcc"

    def _resolve_protein_ff(self, ff_name: str) -> str:
        """Normalize protein force field name."""
        ff_lower = ff_name.lower().replace("amber", "").replace(" ", "")
        mapping = {
            "ff19sb": "ff19SB", "ff14sb": "ff14SB",
            "ff99sb": "ff99SB", "fb15": "fb15",
        }
        for key, canonical in mapping.items():
            if key in ff_lower:
                return canonical
        return ff_name

    def _interpret_amber_results(self, validation: Dict[str, Any]) -> str:
        """Use skill's interpretation section to explain validation results."""
        skill_context = self._get_skill_context(section="interpretation")
        if not skill_context:
            return ""

        prompt = f"""
Interpret these AMBER pipeline validation results for the user.
Be concise and actionable.

VALIDATION RESULTS:
{json.dumps(validation, indent=2)}

INTERPRETATION GUIDE:
{skill_context}

Provide a brief summary of what the results mean and any actions needed.
"""
        try:
            return self._generate_text(prompt)
        except Exception:
            return ""

    def _detect_tip4p_types(self, data_file: str) -> Optional[Dict[str, int]]:
        """
        Detect TIP4P atom types (O, H, M) from a LAMMPS data file.
    
        The virtual site (M/EP) is identified by its near-zero mass.
        O and H are identified by mass (~16 and ~1) among types that
        appear in 3- or 4-atom water molecules.
    
        Returns:
            {"O": type_id, "H": type_id, "M": type_id, "M_dist": float}
            or None if TIP4P types can't be identified.
        """
        masses = {}       # type_id -> mass
        type_names = {}   # type_id -> comment name
    
        try:
            with open(data_file) as f:
                lines = f.readlines()
    
            # Parse Masses section
            in_masses = False
            for line in lines:
                stripped = line.strip()
                if stripped == "Masses" or stripped.startswith("Masses"):
                    in_masses = True
                    continue
                if in_masses:
                    if not stripped:
                        continue
                    if stripped[0].isalpha() and not stripped[0].isdigit():
                        break  # Hit next section
                    if stripped.startswith("#"):
                        continue
                    parts = stripped.split()
                    if len(parts) >= 2:
                        try:
                            type_id = int(parts[0])
                            mass = float(parts[1])
                            masses[type_id] = mass
                            # Extract name from comment
                            if "#" in stripped:
                                name = stripped.split("#")[1].strip().split()[0]
                                type_names[type_id] = name
                        except (ValueError, IndexError):
                            continue
    
            if not masses:
                return None
    
            # Identify types by mass
            o_type = None   # mass ~16
            h_type = None   # mass ~1
            m_type = None   # mass ~0 (virtual site)
    
            for type_id, mass in masses.items():
                name = type_names.get(type_id, "").upper()
    
                # Virtual site: mass is 0 or near-zero
                if mass < 0.1:
                    m_type = type_id
                    continue
    
                # Oxygen: mass ~15.999
                if 15.0 < mass < 17.0:
                    # Prefer water-oxygen names over other oxygens
                    if o_type is None:
                        o_type = type_id
                    elif name in ("OW", "O_W", "OH2", "OT"):
                        o_type = type_id  # Override with water-specific type
    
                # Hydrogen: mass ~1.008
                if 0.5 < mass < 2.0:
                    if h_type is None:
                        h_type = type_id
                    elif name in ("HW", "H_W", "HT"):
                        h_type = type_id  # Override with water-specific type
    
            if m_type is None:
                self.logger.info("No virtual site (mass ~0) found — not a TIP4P system")
                return None
    
            if o_type is None or h_type is None:
                self.logger.warning(
                    f"Found virtual site (type {m_type}) but could not identify "
                    f"O and H types for TIP4P"
                )
                return None
    
            # Determine M-site distance from water model
            # These are the standard distances for common TIP4P variants
            m_name = type_names.get(m_type, "").upper()
            tip4p_distances = {
                "TIP4P":     0.15,
                "TIP4PEW":   0.125,
                "TIP4P2005": 0.1546,
                "TIP4PFB":   0.10527,
                "TIP4PICE":  0.1577,
            }
    
            # Try to guess from type name
            m_dist = 0.125  # Default to TIP4P-Ew (most common)
            for model, dist in tip4p_distances.items():
                if model in m_name or any(
                    model in type_names.get(t, "").upper()
                    for t in [o_type, h_type, m_type]
                ):
                    m_dist = dist
                    break
    
            self.logger.info(
                f"Detected TIP4P types: O={o_type} ({type_names.get(o_type, '?')}), "
                f"H={h_type} ({type_names.get(h_type, '?')}), "
                f"M={m_type} ({type_names.get(m_type, '?')}), dist={m_dist}"
            )
    
            return {
                "O": o_type,
                "H": h_type,
                "M": m_type,
                "M_dist": m_dist,
            }
    
        except Exception as e:
            self.logger.warning(f"Could not detect TIP4P types: {e}")
            return None

    def _generate_amber_input_header(
        self, data_file: str, ff_info: Dict[str, Any]
    ) -> str:
        """Generate LAMMPS force field style commands for AMBER data file."""
        ff_name = ff_info.get("force_field", "AMBER")
        water = ff_info.get("compatible_water_model", "TIP3P")
        water_lower = water.lower().replace("-", "").replace(" ", "")
    
        # Detect which interaction types exist in the data file
        has = {"bonds": False, "angles": False, "dihedrals": False, "impropers": False}
        try:
            with open(data_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[2] == "types":
                        key = parts[1] + "s"
                        if key in has:
                            has[key] = int(parts[0]) > 0
        except Exception:
            has = {k: True for k in has}
    
        tip4p_variants = {"tip4p", "tip4pew", "tip4p2005", "tip4pfb", "tip4pice"}
        is_tip4p = water_lower in tip4p_variants
    
        lines = [
            f"# LAMMPS force field styles for {ff_name}",
            f"# Water model: {water}",
            f"# Generated by ForceFieldAgent (skill: {self.skill_name})",
            f"# Include BEFORE read_data in run script",
            "",
            "# ── Universal AMBER settings ──",
            "units real",
            "atom_style full",
            "special_bonds amber",
            "pair_modify mix arithmetic",
            "",
            "# ── Interaction styles ──",
        ]
    
        if is_tip4p:
            tip4p = self._detect_tip4p_types(data_file)
            if tip4p:
                lines += [
                    f"# TIP4P water: O=type {tip4p['O']}, H=type {tip4p['H']}, "
                    f"M=type {tip4p['M']}, dist={tip4p['M_dist']}",
                    f"pair_style lj/cut/tip4p/long "
                    f"{tip4p['O']} {tip4p['H']} {tip4p['M']} {tip4p['M_dist']} 10.0 12.0",
                ]
            else:
                lines += [
                    f"# WARNING: TIP4P water selected but could not auto-detect types",
                    f"# Replace <O> <H> <M> <dist> with actual type IDs from data file",
                    f"pair_style lj/cut/tip4p/long <O> <H> <M> <dist> 10.0 12.0",
                ]
        else:
            lines += [
                f"# 3-site water ({water})",
                f"pair_style lj/charmm/coul/long 10.0 12.0",
            ]
    
        lines += [
            "kspace_style pppm 1.0e-5",
            "",
        ]
    
        if has["bonds"]:
            lines.append("bond_style harmonic")
        if has["angles"]:
            lines.append("angle_style harmonic")
        if has["dihedrals"]:
            lines.append("dihedral_style fourier")
        if has["impropers"]:
            lines.append("improper_style cvff")
        lines.append("")
    
        return "\n".join(lines)

    def _validate_amber_lammps_data(self, data_file: str) -> Dict[str, Any]:
        """Validate a LAMMPS data file produced by the AMBER pipeline."""
        if _AMBER_TOOLS_AVAILABLE:
            return amber_tools.validate_amber_data_file(data_file)

        # Inline validation
        validation: Dict[str, Any] = {
            "valid": True, "errors": [], "warnings": [],
            "sections_found": [], "n_atoms": 0, "total_charge": 0.0,
        }
        required = {"Masses", "Atoms", "Pair Coeffs"}
        expected = [
            "Masses", "Atoms", "Bonds", "Angles", "Dihedrals", "Impropers",
            "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
            "Dihedral Coeffs", "Improper Coeffs",
        ]
        try:
            with open(data_file) as f:
                content = f.read()
            for s in expected:
                if s in content:
                    validation["sections_found"].append(s)
            for req in required:
                if req not in validation["sections_found"]:
                    validation["errors"].append(f"Missing required section: {req}")
                    validation["valid"] = False

            in_atoms = False
            total_q = 0.0
            n_atoms = 0
            n_nonzero = 0
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("Atoms"):
                    in_atoms = True
                    continue
                if in_atoms:
                    if not stripped:
                        if n_atoms > 0:
                            break
                        continue
                    if stripped[0].isalpha():
                        if n_atoms > 0:
                            break
                        continue
                    parts = stripped.split()
                    if len(parts) >= 7:
                        try:
                            q = float(parts[3])
                            total_q += q
                            n_atoms += 1
                            if abs(q) > 1e-6:
                                n_nonzero += 1
                        except (ValueError, IndexError):
                            pass

            validation["n_atoms"] = n_atoms
            validation["total_charge"] = round(total_q, 4)
            if abs(total_q) > 0.01:
                validation["warnings"].append(
                    f"Net charge {total_q:+.4f} e (expected ~0 for neutralized system)"
                )
            if n_atoms > 10 and n_nonzero == 0:
                validation["errors"].append(
                    "All charges are 0.0 — conversion likely dropped charges"
                )
                validation["valid"] = False
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Could not read {data_file}: {e}")

        return validation

    # ── Inline AmberTools wrappers (used when amber_tools module is absent) ──

    def _run_pdb4amber_inline(self, pdb_file: str) -> str:
        output_file = os.path.join(self.working_dir, "cleaned.pdb")
        cmd = ["pdb4amber", "-i", pdb_file, "-o", output_file]
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           cwd=self.working_dir, timeout=120)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return output_file
        except Exception as e:
            self.logger.warning(f"pdb4amber inline failed: {e}")
        return pdb_file

    def _run_antechamber_inline(self, input_file, prefix, charge, method, gaff):
        ext = os.path.splitext(input_file)[1].lower()
        fmt_map = {".pdb": "pdb", ".mol2": "mol2", ".sdf": "sdf"}
        output_mol2 = os.path.join(self.working_dir, f"{prefix}.mol2")
        cmd = [
            "antechamber", "-i", input_file, "-fi", fmt_map.get(ext, "pdb"),
            "-o", output_mol2, "-fo", "mol2", "-c", method,
            "-s", "2", "-at", gaff, "-nc", str(charge), "-m", "1", "-pf", "y",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=self.working_dir, timeout=600)
        if proc.returncode != 0 or not os.path.exists(output_mol2):
            raise RuntimeError(f"antechamber failed: {proc.stderr[-500:]}")
        return {"mol2": output_mol2, "atom_type": gaff,
                "charge_method": method, "net_charge": charge}

    def _run_parmchk2_inline(self, mol2_file, prefix, gaff):
        frcmod = os.path.join(self.working_dir, f"{prefix}.frcmod")
        s = {"gaff": "1", "gaff2": "2"}.get(gaff, "2")
        cmd = ["parmchk2", "-i", mol2_file, "-f", "mol2", "-o", frcmod, "-s", s]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=self.working_dir, timeout=60)
        if proc.returncode != 0 or not os.path.exists(frcmod):
            raise RuntimeError(f"parmchk2 failed: {proc.stderr[-300:]}")
        return frcmod

    def _generate_tleap_inline(self, pdb_file, composition, mol2_files,
                                frcmod_files, protein_ff, water_model,
                                gaff_version, solvate, box_buffer, neutralize):
        """Generate and return path to a tleap input script."""
        PROTEIN_LEAPRC = {
            "ff14sb": "leaprc.protein.ff14SB", "ff19sb": "leaprc.protein.ff19SB",
            "ff99sb": "leaprc.protein.ff99SB", "fb15": "leaprc.protein.fb15",
        }
        WATER_LEAPRC = {
            "tip3p": "leaprc.water.tip3p", "spce": "leaprc.water.spce",
            "spc/e": "leaprc.water.spce", "opc": "leaprc.water.opc",
            "opc3": "leaprc.water.opc3", "tip4pew": "leaprc.water.tip4pew",
        }
        WATER_BOX = {
            "tip3p": "TIP3PBOX", "spce": "SPCBOX", "spc/e": "SPCBOX",
            "opc": "OPCBOX", "opc3": "OPC3BOX", "tip4pew": "TIP4PEWBOX",
        }
        lines = ["# tleap script — generated by ForceFieldAgent (AMBER skill)", ""]
        if composition.get("small_molecules") or mol2_files:
            lines.append(f"source leaprc.{gaff_version}")
        if composition.get("proteins"):
            key = protein_ff.lower().replace("amber", "").replace(" ", "")
            lines.append(f"source {PROTEIN_LEAPRC.get(key, f'leaprc.protein.{protein_ff}')}")
        if composition.get("nucleic_acids"):
            lines += ["source leaprc.DNA.OL15", "source leaprc.RNA.OL3"]
        if composition.get("lipids"):
            lines.append("source leaprc.lipid21")
        if composition.get("carbohydrates"):
            lines.append("source leaprc.GLYCAM_06j-1")
        wm = water_model.lower().replace(" ", "")
        lines.append(f"source {WATER_LEAPRC.get(wm, f'leaprc.water.{wm}')}")
        lines.append("")
        if mol2_files:
            for info in mol2_files:
                name = info.get("name", "MOL")
                lines.append(f"{name} = loadmol2 {info['mol2']}")
        if frcmod_files:
            for frc in frcmod_files:
                lines.append(f"loadamberparams {frc}")
        if mol2_files or frcmod_files:
            lines.append("")
        lines += [f"SYS = loadpdb {pdb_file}", ""]
        if solvate:
            box = WATER_BOX.get(wm, "TIP3PBOX")
            lines += [f"solvatebox SYS {box} {box_buffer:.1f}", ""]
        if neutralize:
            lines += ["addIonsRand SYS Na+ 0 Cl- 0", ""]
        prmtop = os.path.join(self.working_dir, "system.prmtop")
        inpcrd = os.path.join(self.working_dir, "system.inpcrd")
        lines += ["check SYS", f"saveamberparm SYS {prmtop} {inpcrd}", "quit"]
        script_path = os.path.join(self.working_dir, "system_tleap.in")
        with open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return script_path

    def _run_tleap_inline(self, script_file):
        cmd = ["tleap", "-f", script_file]
        subprocess.run(cmd, capture_output=True, text=True,
                       cwd=self.working_dir, timeout=300)
        log_path = os.path.join(self.working_dir, "leap.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_text = f.read()
            fatals = [l.strip() for l in log_text.splitlines() if "FATAL" in l]
            if fatals:
                raise RuntimeError(f"tleap FATAL: {fatals[0]}")
        prmtop, inpcrd = None, None
        with open(script_file) as f:
            for line in f:
                if line.strip().lower().startswith("saveamberparm"):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        prmtop, inpcrd = parts[2], parts[3]
        if prmtop and inpcrd and os.path.exists(prmtop) and os.path.exists(inpcrd):
            return prmtop, inpcrd
        raise FileNotFoundError(f"tleap output missing. Check {log_path}")

    def _convert_amber_to_lammps_inline(self, prmtop, inpcrd):
        output_data = os.path.join(self.working_dir, "system.data")
        from ...tools.amber_tools import convert_amber_to_lammps
        return convert_amber_to_lammps(prmtop, inpcrd, output_data)

    # ================================================================
    # CORE UTILITY METHODS (unchanged)
    # ================================================================

    def _build_mass_lookup(self) -> Dict[str, float]:
        """Build a dictionary mapping element symbols to their atomic masses."""
        from ase.data import atomic_masses, chemical_symbols
        import math

        lookup = {}
        for i, symbol in enumerate(chemical_symbols):
            if i == 0:
                continue
            mass = atomic_masses[i]
            if mass is not None and mass > 0:
                try:
                    if not math.isnan(mass):
                        lookup[symbol] = mass
                except (TypeError, ValueError):
                    pass
        return lookup

    def _guess_element_from_mass(self, mass: float, tolerance: float = 0.5) -> str:
        """Guess element from atomic mass using ASE's periodic table data."""
        best_match = "X"
        best_diff = float('inf')
        for symbol, ref_mass in self._mass_to_element.items():
            diff = abs(mass - ref_mass)
            if diff < tolerance:
                return symbol
            if diff < best_diff:
                best_diff = diff
                best_match = symbol
        if best_diff > tolerance:
            self.logger.warning(
                f"Could not identify element for mass {mass:.4f}. "
                f"Closest: {best_match} (diff: {best_diff:.4f})"
            )
        return best_match

    def _generate_json(self, prompt: str) -> dict:
        """Generate JSON response from LLM."""
        self.logger.info("Sending request to LLM...")
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            if not response or not response.text:
                raise ValueError("Empty response from LLM")
            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                raw_text = "\n".join(json_lines)
            if not raw_text.startswith("{"):
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                if start != -1 and end > start:
                    raw_text = raw_text[start:end]
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Error during LLM content generation: {e}")
            raise

    def _generate_text(self, prompt: str) -> str:
        """Generate text response from LLM."""
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            if not response or not response.text:
                raise ValueError("Empty response from LLM")
            return response.text.strip()
        except Exception as e:
            self.logger.exception(f"Error during LLM content generation: {e}")
            raise

    # ================================================================
    # PUBLIC API (modified to be skill-aware)
    # ================================================================

    def select_force_field(self,
                         pdb_file: str,
                         research_goal: str,
                         system_description: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze system composition and research goal to select optimal force field.
        """
        self.logger.info(f"Analyzing system in {pdb_file} for force field selection")

        system_info = self._analyze_system_composition(pdb_file)

        if not system_description:
            system_description = self._generate_system_description(system_info)

        self.logger.info(f"System description: {system_description}")

        force_field_selection = self._select_optimal_force_field(
            system_info=system_info,
            research_goal=research_goal,
            system_description=system_description
        )

        # ── SKILL INTEGRATION ── auto-select skill after FF selection
        ff_name = force_field_selection.get("force_field", "")
        self._auto_select_skill(ff_name)

        param_method = self._determine_parameter_method(
            system_info=system_info,
            force_field=force_field_selection["force_field"],
            research_goal=research_goal
        )

        self.logger.info(f"Selected force field: {force_field_selection['force_field']}")
        self.logger.info(f"Parameter acquisition method: {param_method['method']}")
        if self.skill_name:
            self.logger.info(f"Active skill: {self.skill_name}")

        result = {
            "system_info": system_info,
            "system_description": system_description,
            "force_field": force_field_selection,
            "parameter_method": param_method,
            "working_dir": self.working_dir,
            "research_goal": research_goal,         # ── SKILL INTEGRATION ── save for pipeline
            "skill_used": self.skill_name,          # ── SKILL INTEGRATION ──
        }

        selection_file = os.path.join(self.working_dir, "force_field_selection.json")
        with open(selection_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def acquire_parameters(self,
                           selection_info: Dict[str, Any],
                           data_file: Optional[str] = None,
                           pdb_file: Optional[str] = None,
                           small_molecule_info: Optional[List[Dict[str, Any]]] = None,
                           solvate: bool = False,
                           box_buffer: float = 10.0,
                           neutralize: bool = True) -> Dict[str, Any]:
        """
        Acquire force field parameters using the selected method.

        Routes to AMBER pipeline when the AMBER skill is active and
        AmberTools are available. Otherwise falls back to LLM-based
        parameterization (with skill context if loaded).

        Args:
            selection_info: Force field selection info from select_force_field()
            data_file: Optional existing LAMMPS data file to enhance
            pdb_file: Optional PDB file (required for AMBER pipeline)
            small_molecule_info: Non-standard residues for antechamber
            solvate: Add solvent box (AMBER pipeline only)
            box_buffer: Solvent buffer in Angstroms (AMBER pipeline only)
            neutralize: Add counter-ions (AMBER pipeline only)

        Returns:
            Dictionary with parameter files and information
        """
        force_field_info = selection_info.get("force_field", {})
        force_field = force_field_info.get("force_field", "OPLS-AA")
        system_info = selection_info.get("system_info", {})

        # ── SKILL INTEGRATION ── Auto-select skill if not already loaded
        self._auto_select_skill(force_field)

        # ── SKILL INTEGRATION ── Route to AMBER pipeline if skill + tools available
        if self.skill_name == "amber" and self._is_amber_force_field(force_field):
            tools_status = self._check_amber_tools_available()

            if tools_status["available"] and pdb_file:
                self.logger.info(
                    "🧪 Routing to AMBER pipeline (skill + tools available)"
                )
                return self._run_amber_pipeline(
                    selection_info=selection_info,
                    pdb_file=pdb_file,
                    small_molecule_info=small_molecule_info,
                    solvate=solvate,
                    box_buffer=box_buffer,
                    neutralize=neutralize,
                )
            elif not tools_status["available"]:
                self.logger.warning(
                    f"AMBER skill loaded but tools missing: {tools_status.get('missing', [])}. "
                    f"Falling back to LLM-based parameterization with AMBER knowledge."
                )
            elif not pdb_file:
                self.logger.warning(
                    "AMBER pipeline requires pdb_file. "
                    "Falling back to LLM-based parameterization."
                )

        # ── Existing LLM-based path (now with skill context) ──
        method = selection_info.get("parameter_method", {}).get("method", "database")

        self.logger.info(f"Acquiring parameters via {method} method for {force_field}")

        if method in ["quantum", "analogy"]:
            self.logger.warning(
                f"Method '{method}' requires external tools. "
                f"Falling back to 'database' method for automated pipeline."
            )
            method = "database"

        params = {
            "source": method,
            "force_field": force_field,
            "force_field_info": force_field_info,
            "system_info": system_info,
            "atom_types": {},
            "parameter_files": {},
            "skill_used": self.skill_name,          # ── SKILL INTEGRATION ──
            "lammps_settings": {
                "pair_style": force_field_info.get("lammps_pair_style", "lj/cut/coul/long 10.0"),
                "bond_style": force_field_info.get("lammps_bond_style", "harmonic"),
                "angle_style": force_field_info.get("lammps_angle_style", "harmonic"),
                "dihedral_style": force_field_info.get("lammps_dihedral_style", "opls"),
                "kspace_style": "pppm 1.0e-4",
                "special_bonds": "lj/coul 0.0 0.0 0.5",
            }
        }

        if data_file and os.path.exists(data_file):
            self.logger.info(f"Extracting atom types from data file: {data_file}")
            try:
                atom_types = self._extract_atom_types_from_data(data_file)
                params["atom_types"] = atom_types
                self.logger.info(f"Found {len(atom_types)} atom types in data file")
            except Exception as e:
                self.logger.warning(f"Could not extract atom types from data file: {e}")

        self.logger.info("Generating force field parameters...")
        try:
            generated_params = self._generate_parameters_with_llm(
                force_field=force_field,
                system_info=system_info,
                data_file=data_file
            )
            params.update(generated_params)
        except Exception as e:
            self.logger.error(f"Parameter generation failed: {e}")
            params["generation_error"] = str(e)

        self.logger.info("Validating parameters...")
        validation = self._validate_parameters(params, system_info)
        params["validation"] = validation

        if validation.get("errors"):
            self.logger.warning(f"Parameter validation errors: {validation['errors']}")
        if validation.get("warnings"):
            self.logger.info(f"Parameter validation warnings: {validation['warnings']}")

        params["summary"] = self._generate_parameter_summary(params, selection_info)

        param_file = os.path.join(self.working_dir, "parameter_info.json")
        try:
            with open(param_file, 'w') as f:
                serializable_params = {
                    k: v for k, v in params.items()
                    if k not in ["raw_data", "quantum_results"]
                }
                json.dump(serializable_params, f, indent=2)
            self.logger.info(f"Saved parameter info to {param_file}")
        except Exception as e:
            self.logger.warning(f"Could not save parameter info: {e}")

        return params

    def generate_lammps_parameters(self,
                               parameter_info: Dict[str, Any],
                               data_file: str) -> Dict[str, str]:
        """
        Generate LAMMPS parameter files based on the acquired parameters.
        """
        self.logger.info(f"Generating LAMMPS parameter files for {data_file}")

        # ── SKILL INTEGRATION ── If AMBER pipeline produced a self-contained data file,
        # no separate parameter file is needed — return the input header instead.
        if parameter_info.get("pipeline") == "amber":
            self.logger.info(
                "AMBER pipeline produced a self-contained data file. "
                "Writing LAMMPS input header instead of separate parameter file."
            )
            header = parameter_info.get("input_header", "")
            if header:
                header_file = os.path.join(self.working_dir, "ff_params.lammps")
                with open(header_file, 'w') as f:
                    f.write(header)
                return {"main": header_file}
            # Fall through to LLM-based generation if header is missing

        param_content = self._generate_lammps_parameters_from_data(
            data_file=data_file,
            force_field_info={"force_field": parameter_info.get("force_field", {})}
        )

        files = {}
        param_file = os.path.join(self.working_dir, "ff_params.lammps")
        with open(param_file, 'w') as f:
            f.write(param_content["main"])
        files["main"] = param_file

        if "additional" in param_content:
            for name, content in param_content["additional"].items():
                if content:
                    add_file = os.path.join(self.working_dir, f"{name}.lammps")
                    with open(add_file, 'w') as f:
                        f.write(content)
                    files[name] = add_file

        self.logger.info(f"Generated {len(files)} parameter files")
        return files

    # ================================
    # PRIVATE METHODS
    # ================================

    def _analyze_system_composition(self, pdb_file: str) -> Dict[str, Any]:
        """Analyze the molecular system's composition from a PDB file."""
        self.logger.info(f"Analyzing composition of {pdb_file}")

        try:
            from MDAnalysis import Universe
            u = Universe(pdb_file)

            elements = {}
            for atom in u.atoms:
                if len(atom.name) > 1 and atom.name[1].islower():
                    element = atom.name[:2]
                else:
                    element = atom.name[0]
                elements[element] = elements.get(element, 0) + 1

            has_water = self._detect_water(u, elements)
            has_proteins = self._detect_proteins(u)
            has_lipids = self._detect_lipids(u)
            has_nucleic_acids = self._detect_nucleic_acids(u)
            has_ions = self._detect_ions(elements)
            has_small_molecules = self._detect_small_molecules(u, elements)
            has_metals = self._detect_metals(elements)
            has_carbohydrates = self._detect_carbohydrates(u)
            is_interface = self._detect_interface(u)
            is_gas_phase = self._detect_gas_phase(u)

            box_dimensions = u.dimensions[:3] if hasattr(u, 'dimensions') and u.dimensions is not None else [0, 0, 0]

            system_info = {
                "filename": os.path.basename(pdb_file),
                "n_atoms": len(u.atoms),
                "n_residues": len(u.residues),
                "n_molecules": len(u.segments),
                "elements": elements,
                "composition": {
                    "water": has_water,
                    "proteins": has_proteins,
                    "lipids": has_lipids,
                    "nucleic_acids": has_nucleic_acids,
                    "ions": has_ions,
                    "small_molecules": has_small_molecules,
                    "metals": has_metals,
                    "carbohydrates": has_carbohydrates
                },
                "system_type": {
                    "interface": is_interface,
                    "gas_phase": is_gas_phase
                },
                "box_dimensions": box_dimensions.tolist() if isinstance(box_dimensions, np.ndarray) else box_dimensions,
            }

            if has_proteins or has_nucleic_acids or has_small_molecules:
                residue_names = [res.resname for res in u.residues]
                residue_counts = {}
                for name in residue_names:
                    residue_counts[name] = residue_counts.get(name, 0) + 1
                system_info["residue_counts"] = residue_counts

            return system_info

        except Exception as e:
            self.logger.error(f"Error analyzing PDB file: {e}")
            return {
                "filename": os.path.basename(pdb_file),
                "n_atoms": 0,
                "elements": {},
                "composition": {
                    "water": False, "proteins": False, "lipids": False,
                    "nucleic_acids": False, "ions": False,
                    "small_molecules": False, "metals": False,
                    "carbohydrates": False
                }
            }

    def _detect_water(self, universe, elements):
        water_residues = ['WAT', 'HOH', 'H2O', 'SOL', 'TIP', 'SPC']
        for res in universe.residues:
            if any(water in res.resname for water in water_residues):
                return True
        if 'H' in elements and 'O' in elements:
            h_atoms = elements.get('H', 0)
            o_atoms = elements.get('O', 0)
            if h_atoms > 0 and o_atoms > 0 and (h_atoms / o_atoms) > 1.5:
                return True
        return False

    def _detect_proteins(self, universe):
        amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                      'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                      'TYR', 'VAL']
        for res in universe.residues:
            if res.resname in amino_acids:
                return True
        return False

    def _detect_lipids(self, universe):
        lipid_residues = ['POPC', 'POPE', 'DPPC', 'DOPC', 'DMPC', 'CHOL', 'CHL',
                         'DLPE', 'DLPC', 'DSPC', 'DAPC', 'DOPE', 'POPG', 'DPPG']
        for res in universe.residues:
            if res.resname in lipid_residues:
                return True
        return False

    def _detect_nucleic_acids(self, universe):
        nucleotides = ['ADE', 'THY', 'GUA', 'CYT', 'URA', 'A', 'T', 'G', 'C', 'U',
                      'DA', 'DT', 'DG', 'DC', 'DU', 'AMP', 'GMP', 'CMP', 'TMP', 'UMP']
        for res in universe.residues:
            if res.resname in nucleotides:
                return True
        return False

    def _detect_ions(self, elements):
        ion_elements = ['Na', 'K', 'Cl', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu', 'Li']
        return any(ion in elements for ion in ion_elements)

    def _detect_small_molecules(self, universe, elements):
        if 'C' in elements and elements['C'] > 0:
            if not self._detect_proteins(universe) and not self._detect_lipids(universe) and not self._detect_nucleic_acids(universe):
                return True
        return False

    def _detect_metals(self, elements):
        metal_elements = ['Fe', 'Zn', 'Cu', 'Ni', 'Co', 'Mn', 'Mg', 'Ca', 'Na', 'K',
                         'Al', 'Ti', 'V', 'Cr', 'Pd', 'Pt', 'Au', 'Ag', 'Hg']
        return any(metal in elements for metal in metal_elements)

    def _detect_carbohydrates(self, universe):
        carb_residues = ['GLC', 'GAL', 'MAN', 'FUC', 'XYL', 'NAG', 'SIA', 'RIB',
                        'AGLC', 'BGLC', 'GLCA', 'GLCN']
        for res in universe.residues:
            if res.resname in carb_residues:
                return True
        return False

    def _detect_interface(self, universe):
        try:
            z_coords = universe.atoms.positions[:, 2]
            n_bins = 20
            hist, edges = np.histogram(z_coords, bins=n_bins)
            middle_bins = hist[int(n_bins*0.15):int(n_bins*0.85)]
            if min(middle_bins) < max(middle_bins) * 0.1:
                return True
            return False
        except:
            return False

    def _detect_gas_phase(self, universe):
        has_water = self._detect_water(universe, {})
        if not has_water and hasattr(universe, 'dimensions') and universe.dimensions is not None:
            try:
                volume = universe.dimensions[0] * universe.dimensions[1] * universe.dimensions[2] / 1000
                n_atoms = len(universe.atoms)
                density = (n_atoms * 12) / (volume * 0.6022)
                if density < 0.1:
                    return True
            except:
                pass
        return False

    def _generate_system_description(self, system_info: Dict[str, Any]) -> str:
        description_parts = []
        comp = system_info["composition"]
        if comp["water"]:
            description_parts.append("water")
        if comp["ions"]:
            ion_elements = [e for e in system_info.get("elements", {}) if e in ['Na', 'K', 'Cl', 'Ca', 'Mg', 'Zn', 'Fe']]
            if ion_elements:
                description_parts.append("+".join(ion_elements) + " ions")
            else:
                description_parts.append("ions")
        if comp["proteins"]:
            n_residues = system_info.get("n_residues", 0)
            if n_residues > 300:
                description_parts.append("large protein")
            else:
                description_parts.append("protein")
        if comp["lipids"]:
            description_parts.append("lipids")
        if comp["nucleic_acids"]:
            description_parts.append("nucleic acids")
        if comp["carbohydrates"]:
            description_parts.append("carbohydrates")
        if comp["small_molecules"]:
            description_parts.append("organic molecules")
        if comp["metals"]:
            description_parts.append("metals")
        system_type = system_info.get("system_type", {})
        if system_type.get("interface"):
            description_parts.append("interface")
        if system_type.get("gas_phase"):
            description_parts.append("gas phase")
        if description_parts:
            description = " with ".join(description_parts)
        else:
            description = "molecular system"
        return f"{description} ({system_info['n_atoms']} atoms)"

    def _select_optimal_force_field(self,
                                    system_info: Dict[str, Any],
                                    research_goal: str,
                                    system_description: str) -> Dict[str, Any]:
        """Select optimal force field using LLM, enhanced with skill knowledge."""

        # ── SKILL INTEGRATION ── inject skill context
        skill_context = self._get_skill_context(section="planning")
        if not skill_context:
            skill_context = self._get_skill_context(section="overview")

        skill_block = ""
        if skill_context:
            skill_block = f"""
DOMAIN-SPECIFIC GUIDANCE (use this to inform your selection):
{skill_context}
"""

        prompt = f"""
You are an expert in molecular dynamics force field selection for LAMMPS simulations.

SYSTEM INFORMATION:
- Total atoms: {system_info.get('n_atoms', 'Unknown')}
- Elements present: {system_info.get('elements', {})}
- Composition: {system_info.get('composition', {})}
- System description: {system_description}

RESEARCH GOAL:
{research_goal}

TARGET MD SOFTWARE: LAMMPS
{skill_block}
Select the optimal force field for this LAMMPS simulation. Consider:
1. Scientific accuracy for the research goals
2. **LAMMPS compatibility** - the force field must be implementable with standard LAMMPS pair_styles
3. Parameter availability for all species
4. Computational efficiency

IMPORTANT LAMMPS CONSTRAINTS:
- Standard pair_styles: lj/cut/coul/long, lj/charmm/coul/long, buck/coul/long
- For metal ions, consider: simple point charge models (Joung-Cheatham, Aqvist)
  which work with standard LJ, rather than 12-6-4 models requiring special implementation
- Water models: TIP3P, TIP4P, SPC/E are well-supported
- For organic molecules: OPLS-AA, GAFF, CHARMM are well-supported

Return a JSON object with:
{{
    "force_field": "<primary force field name>",
    "compatible_water_model": "<water model>",
    "ion_parameters": "<ion parameter set>",
    "justification": "<scientific reasoning>",
    "lammps_pair_style": "<recommended LAMMPS pair_style>",
    "alternatives": ["<alternative 1>", "<alternative 2>"],
    "cautions": "<important considerations>",
    "parameter_availability": "high|medium|low"
}}
"""
        return self._generate_json(prompt)

    def _determine_parameter_method(self,
                                system_info: Dict[str, Any],
                                force_field: str,
                                research_goal: str) -> Dict[str, Any]:
        """Determine the best method to obtain force field parameters."""
        self.logger.info(f"Determining parameter acquisition method for {force_field}")

        composition = system_info["composition"]
        comp_str = "\n".join([f"- {k.replace('_', ' ')}: {'Yes' if v else 'No'}" for k, v in composition.items()])

        method_info = ""
        for method, details in self.param_methods.items():
            tools = ", ".join(details["tools"])
            strengths = ", ".join(details["strengths"])
            method_info += f"- {method.upper()}: {details['description']}\n  Tools: {tools}\n  Strengths: {strengths}\n\n"

        # ── SKILL INTEGRATION ── add skill context
        skill_context = self._get_skill_context(section="planning")
        skill_block = ""
        if skill_context:
            skill_block = f"""
DOMAIN-SPECIFIC GUIDANCE:
{skill_context}
"""

        prompt = f"""
As an expert in molecular dynamics parameterization, determine the best method to obtain force field parameters.

SELECTED FORCE FIELD: {force_field}

SYSTEM COMPOSITION:
{comp_str}

RESEARCH GOAL: "{research_goal}"

PARAMETER ACQUISITION METHODS:
{method_info}
{skill_block}
Based on this information, what is the best method to obtain parameters for this system?
Consider these factors:
1. Parameter availability for the selected force field
2. Presence of non-standard molecules requiring custom parameterization
3. Accuracy requirements implied by the research goal
4. Computational resources and time constraints
5. Availability of QM-level parameterization tools if needed

Provide your response as JSON with this structure:
{{
    "method": "database|analogy|quantum",
    "justification": "Detailed scientific explanation for this choice",
    "recommended_tools": ["Tool1", "Tool2"],
    "estimated_effort": "low|medium|high",
    "specific_approaches": [
        "Detailed step-by-step approaches to obtain parameters"
    ]
}}
Include only the JSON response with no additional text.
"""

        try:
            param_method = self._generate_json(prompt)
            param_method.setdefault("method", "database")
            param_method.setdefault("recommended_tools", [])
            param_method.setdefault("estimated_effort", "medium")
            param_method.setdefault("specific_approaches", [])
            return param_method
        except Exception as e:
            self.logger.error(f"Error determining parameter method: {e}")
            return {
                "method": "database",
                "justification": "Default selection due to LLM analysis failure.",
                "recommended_tools": ["ParmEd", "AmberTools"],
                "estimated_effort": "medium",
                "specific_approaches": ["Use standard force field databases"]
            }

    def _acquire_parameters_from_database(self,
                                       force_field: str,
                                       system_info: Dict[str, Any],
                                       data_file: Optional[str] = None) -> Dict[str, Any]:
        self.logger.info(f"Acquiring parameters for {force_field} from databases")
        ff_files = self._determine_force_field_files(force_field, system_info)
        parameters = {
            "source": "database", "force_field": force_field,
            "parameter_files": ff_files, "atom_types": {},
            "bonds": {}, "angles": {}, "dihedrals": {},
            "impropers": {}, "nonbonded": {},
        }
        if data_file:
            atom_types = self._extract_atom_types_from_data(data_file)
            parameters["atom_types"] = atom_types
        parameters.update(self._generate_parameters_with_llm(force_field, system_info, data_file))
        self.logger.info(f"Acquired parameters for {len(parameters['atom_types'])} atom types")
        return parameters

    def _acquire_parameters_by_analogy(self, force_field, system_info, data_file=None):
        self.logger.info(f"Acquiring parameters for {force_field} by chemical analogy")
        parameters = self._acquire_parameters_from_database(force_field, system_info, data_file)
        parameters["source"] = "analogy"
        unique_molecules = self._extract_unique_molecules(system_info)
        analogy_params = {}
        for molecule in unique_molecules:
            if molecule not in parameters["atom_types"]:
                analogy = self._find_molecular_analogy(molecule, force_field)
                if analogy:
                    analogy_params[molecule] = analogy
        parameters["analogies"] = analogy_params
        parameters.update(self._enhance_parameters_with_llm(parameters, "analogy"))
        return parameters

    def _acquire_parameters_from_quantum(self, force_field, system_info, data_file=None):
        self.logger.info(f"Acquiring parameters for {force_field} via quantum calculations")
        parameters = self._acquire_parameters_from_database(force_field, system_info, data_file)
        parameters["source"] = "quantum"
        unique_molecules = self._extract_unique_molecules(system_info)
        standard_molecules = self._identify_standard_molecules(unique_molecules, force_field)
        qm_needed = [m for m in unique_molecules if m not in standard_molecules]
        if not qm_needed:
            self.logger.info("No molecules need QM parameterization, using database parameters")
            return parameters
        parameters["qm_molecules"] = qm_needed
        parameters.update(self._enhance_parameters_with_llm(parameters, "quantum"))
        return parameters

    def _determine_force_field_files(self, force_field, system_info):
        ff_files = {}
        comp = system_info["composition"]
        if "AMBER" in force_field:
            ff_base = "amber"
            if comp["proteins"]: ff_files["proteins"] = f"{ff_base}/ff14SB.dat"
            if comp["water"]: ff_files["water"] = f"{ff_base}/tip3p.dat"
            if comp["ions"]: ff_files["ions"] = f"{ff_base}/ions.dat"
            if comp["nucleic_acids"]: ff_files["nucleic_acids"] = f"{ff_base}/DNA.OL15.dat"
            if comp["small_molecules"]: ff_files["small_molecules"] = f"{ff_base}/gaff.dat"
        elif "CHARMM" in force_field:
            ff_base = "charmm"
            if comp["proteins"]: ff_files["proteins"] = f"{ff_base}/prot.prm"
            if comp["water"]: ff_files["water"] = f"{ff_base}/water.prm"
            if comp["lipids"]: ff_files["lipids"] = f"{ff_base}/lipid.prm"
        elif "OPLS" in force_field:
            ff_files["main"] = "opls/oplsaa.prm"
        else:
            ff_files["main"] = f"generic/{force_field.lower().replace(' ', '_')}.dat"
        return ff_files

    def _extract_atom_types_from_data(self, data_file):
        atom_types = {}
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
            in_masses = False
            for line in lines:
                line = line.strip()
                if "Masses" in line:
                    in_masses = True; continue
                elif in_masses and line.startswith("#"):
                    continue
                elif in_masses and not line:
                    in_masses = False
                elif in_masses:
                    parts = line.split()
                    if len(parts) >= 2:
                        atom_type = int(parts[0])
                        mass = float(parts[1])
                        element = self._guess_element_from_mass(mass)
                        atom_types[atom_type] = {"mass": mass, "element": element}
            in_pair_coeffs = False
            for line in lines:
                line = line.strip()
                if "Pair Coeffs" in line:
                    in_pair_coeffs = True; continue
                elif in_pair_coeffs and line.startswith("#"):
                    continue
                elif in_pair_coeffs and not line:
                    in_pair_coeffs = False
                elif in_pair_coeffs:
                    parts = line.split()
                    if len(parts) >= 3:
                        atom_type = int(parts[0])
                        epsilon = float(parts[1])
                        sigma = float(parts[2])
                        if atom_type in atom_types:
                            atom_types[atom_type].update({"epsilon": epsilon, "sigma": sigma})
        except Exception as e:
            self.logger.error(f"Error extracting atom types from data file: {e}")
        return atom_types

    def _extract_unique_molecules(self, system_info):
        unique_molecules = []
        if "residue_counts" in system_info:
            unique_molecules = list(system_info["residue_counts"].keys())
        else:
            comp = system_info["composition"]
            if comp["water"]: unique_molecules.append("HOH")
            if comp["ions"]:
                elements = system_info.get("elements", {})
                for ion in ["Na", "K", "Cl", "Ca", "Mg"]:
                    if ion in elements: unique_molecules.append(ion)
        return unique_molecules

    def _identify_standard_molecules(self, molecules, force_field):
        standard_molecules = ["HOH", "WAT", "TIP3", "SOL", "Na", "K", "Cl", "Ca", "Mg"]
        amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
                      "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
                      "TYR", "VAL"]
        nucleotides = ["ADE", "THY", "GUA", "CYT", "URA", "A", "T", "G", "C", "U",
                      "DA", "DT", "DG", "DC", "DU"]
        if "AMBER" in force_field or "CHARMM" in force_field:
            standard_molecules.extend(amino_acids)
            standard_molecules.extend(nucleotides)
        return [m for m in molecules if m in standard_molecules]

    def _find_molecular_analogy(self, molecule, force_field):
        return {
            "similar_to": "similar molecule name",
            "similarity": 0.85,
            "modifications_needed": ["replace methyl with ethyl"],
            "parameter_adjustments": ["increase C-C bond length by 0.02 Å"]
        }

    def _generate_parameters_with_llm(self, force_field, system_info, data_file=None):
        """Generate force field parameters using LLM, enhanced with skill knowledge."""
        self.logger.info(f"Generating parameters for {force_field} using LLM")

        elements_str = ", ".join([f"{e}: {c}" for e, c in system_info.get("elements", {}).items()])
        comp = system_info["composition"]
        comp_str = "\n".join([f"- {k.replace('_', ' ')}: {'Yes' if v else 'No'}" for k, v in comp.items()])

        # ── SKILL INTEGRATION ── add skill context
        skill_context = self._get_skill_context(section="implementation")
        skill_block = ""
        if skill_context:
            skill_block = f"""
DOMAIN-SPECIFIC PARAMETER GUIDANCE:
{skill_context}
"""

        prompt = f"""
As an expert in molecular dynamics force fields, provide appropriate LAMMPS parameters for this system.

FORCE FIELD: {force_field}

SYSTEM ELEMENTS: {elements_str}

SYSTEM COMPOSITION:
{comp_str}
{skill_block}
You need to generate scientifically accurate force field parameters for LAMMPS based on the {force_field} force field.
For each parameter type, provide values compatible with LAMMPS syntax and the selected force field.

Please provide parameters in the following JSON format:
{{
    "atom_types": {{
        "1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}},
        "2": {{"name": "H", "mass": 1.008, "description": "Water hydrogen", "charge": 0.4238, "epsilon": 0.0, "sigma": 0.0}}
    }},
    "bonds": {{
        "1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}
    }},
    "angles": {{
        "1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}
    }},
    "dihedrals": {{
        "1": {{"type": "periodic", "atoms": ["X", "X", "X", "X"], "k": 0.0, "d": 1, "n": 1, "description": "Example dihedral"}}
    }},
    "nonbonded_terms": {{
        "mixing_rule": "geometric for epsilon, arithmetic for sigma",
        "cutoff": 10.0
    }}
}}

Include only parameters relevant to the system composition. The parameters should be scientifically accurate for the {force_field} force field.
Include only the JSON response with no additional text.
"""

        try:
            parameters = self._generate_json(prompt)
            for category in ["atom_types", "bonds", "angles", "dihedrals", "nonbonded_terms"]:
                if category not in parameters:
                    parameters[category] = {}
            return parameters
        except Exception as e:
            self.logger.error(f"Error generating parameters with LLM: {e}")
            return {
                "atom_types": {}, "bonds": {}, "angles": {}, "dihedrals": {},
                "nonbonded_terms": {"mixing_rule": "geometric for epsilon, arithmetic for sigma", "cutoff": 10.0}
            }

    def _enhance_parameters_with_llm(self, parameters, method):
        force_field = parameters.get("force_field", "Unknown")
        if method == "analogy":
            analogies_str = ""
            for molecule, analogy in parameters.get("analogies", {}).items():
                similar_to = analogy.get("similar_to", "unknown")
                similarity = analogy.get("similarity", 0)
                modifications = ", ".join(analogy.get("modifications_needed", []))
                analogies_str += f"- {molecule}: similar to {similar_to} (similarity: {similarity})\n  Modifications: {modifications}\n"
            prompt = f"""
As an expert in molecular force field parameterization by analogy, enhance these parameters.

FORCE FIELD: {force_field}

CURRENT PARAMETERS:
{json.dumps(parameters.get('atom_types', {}), indent=2)}

MOLECULAR ANALOGIES:
{analogies_str}

Based on the molecular analogies provided, enhance the parameters to account for the chemical differences.

Please provide enhanced parameters in this JSON format:
{{
    "atom_types": {{"1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}}}},
    "bonds": {{"1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}}},
    "angles": {{"1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}}}
}}
Include only the JSON response with no additional text.
"""
        elif method == "quantum":
            qm_molecules = ", ".join(parameters.get("qm_molecules", []))
            prompt = f"""
As an expert in quantum-derived force field parameterization, enhance these parameters.

FORCE FIELD: {force_field}

CURRENT PARAMETERS:
{json.dumps(parameters.get('atom_types', {}), indent=2)}

MOLECULES NEEDING QM PARAMETERIZATION: {qm_molecules}

Please provide quantum-derived parameters in this JSON format:
{{
    "atom_types": {{"1": {{"name": "O", "mass": 15.9994, "description": "Water oxygen", "charge": -0.8476, "epsilon": 0.1553, "sigma": 3.166}}}},
    "bonds": {{"1": {{"type": "harmonic", "atoms": ["O", "H"], "k": 450.0, "r0": 1.0, "description": "O-H bond in water"}}}},
    "angles": {{"1": {{"type": "harmonic", "atoms": ["H", "O", "H"], "k": 55.0, "theta0": 109.47, "description": "H-O-H angle in water"}}}}
}}
Include only the JSON response with no additional text.
"""
        else:
            return parameters

        try:
            enhanced_params = self._generate_json(prompt)
            for category in ["atom_types", "bonds", "angles", "dihedrals"]:
                if category in enhanced_params:
                    for key, value in enhanced_params[category].items():
                        if key not in parameters.get(category, {}):
                            if category not in parameters:
                                parameters[category] = {}
                            parameters[category][key] = value
            return parameters
        except Exception as e:
            self.logger.error(f"Error enhancing parameters with LLM: {e}")
            return parameters

    def _validate_parameters(self, parameters, system_info):
        """Validate parameters for scientific rigor, enhanced with skill knowledge."""
        validation = {
            "passed": True, "warnings": [], "errors": [], "quality_metrics": {}
        }

        # ── SKILL INTEGRATION ── get validation criteria from skill
        skill_validation = self._get_skill_context(section="validation")

        missing_atom_types = []
        elements = system_info.get("elements", {})
        for element in elements:
            found = False
            for atom_type in parameters.get("atom_types", {}).values():
                if atom_type.get("name", "") == element or atom_type.get("element", "") == element:
                    found = True
                    break
            if not found:
                missing_atom_types.append(element)
                validation["warnings"].append(f"Missing parameters for element {element}")
        if missing_atom_types:
            validation["passed"] = False

        for atom_type, props in parameters.get("atom_types", {}).items():
            mass = props.get("mass", 0)
            if mass <= 0:
                validation["errors"].append(f"Invalid mass for atom type {atom_type}: {mass}")
                validation["passed"] = False
            charge = props.get("charge", 0)
            if abs(charge) > 2.0:
                validation["warnings"].append(f"Unusual charge for atom type {atom_type}: {charge}")
            epsilon = props.get("epsilon", 0)
            sigma = props.get("sigma", 0)
            if epsilon < 0:
                validation["errors"].append(f"Invalid epsilon for atom type {atom_type}: {epsilon}")
                validation["passed"] = False
            if sigma <= 0:
                validation["errors"].append(f"Invalid sigma for atom type {atom_type}: {sigma}")
                validation["passed"] = False

            # ── SKILL INTEGRATION ── skill-informed range checks
            if skill_validation and self.skill_name == "amber":
                if epsilon > 0.5 and props.get("name", "") not in ["X", "?"]:
                    validation["warnings"].append(
                        f"Epsilon {epsilon} for type {atom_type} exceeds typical AMBER range (0-0.5 kcal/mol)"
                    )
                if 0 < sigma < 1.0 and props.get("name", "") != "H":
                    validation["warnings"].append(
                        f"Sigma {sigma} for type {atom_type} below typical AMBER range (>1.0 A)"
                    )

        param_source = parameters.get("source", "unknown")
        source_quality = {
            "database": 70, "analogy": 80, "quantum": 95,
            "amber_pipeline": 95,                       # ── SKILL INTEGRATION ──
        }.get(param_source, 50)
        coverage = 100 - (len(missing_atom_types) / max(1, len(elements)) * 100)
        quality_score = (source_quality * 0.7) + (coverage * 0.3)
        validation["quality_metrics"] = {
            "overall_score": int(quality_score),
            "parameter_source": param_source,
            "coverage": int(coverage),
            "missing_elements": missing_atom_types
        }
        return validation

    def _parse_data_file(self, data_file):
        info = {
            "atom_types": 0, "bond_types": 0, "angle_types": 0,
            "dihedral_types": 0, "improper_types": 0,
            "atoms": 0, "bonds": 0, "angles": 0, "dihedrals": 0, "impropers": 0,
            "masses": {}, "box": []
        }
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if "atoms" in line and "types" not in line:
                    info["atoms"] = int(line.split()[0])
                elif "bonds" in line and "types" not in line:
                    info["bonds"] = int(line.split()[0])
                elif "angles" in line and "types" not in line:
                    info["angles"] = int(line.split()[0])
                elif "dihedrals" in line and "types" not in line:
                    info["dihedrals"] = int(line.split()[0])
                elif "impropers" in line and "types" not in line:
                    info["impropers"] = int(line.split()[0])
                elif "atom types" in line:
                    info["atom_types"] = int(line.split()[0])
                elif "bond types" in line:
                    info["bond_types"] = int(line.split()[0])
                elif "angle types" in line:
                    info["angle_types"] = int(line.split()[0])
                elif "dihedral types" in line:
                    info["dihedral_types"] = int(line.split()[0])
                elif "improper types" in line:
                    info["improper_types"] = int(line.split()[0])
                elif "xlo xhi" in line:
                    parts = line.split()
                    info["box"].append((float(parts[0]), float(parts[1])))
            in_masses = False
            for line in lines:
                line = line.strip()
                if "Masses" in line:
                    in_masses = True; continue
                elif in_masses and not line:
                    in_masses = False
                elif in_masses and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        info["masses"][int(parts[0])] = float(parts[1])
        except Exception as e:
            self.logger.error(f"Error parsing data file: {e}")
        return info

    def _generate_lammps_parameters_from_data(self, data_file, force_field_info):
        self.logger.info(f"Generating parameters matched to data file: {data_file}")
        data_info = self._parse_data_file_for_charges(data_file)
        n_atom_types = len(data_info["masses"])
        n_bond_types = n_angle_types = n_dihedral_types = n_improper_types = 0
        with open(data_file, 'r') as f:
            for line in f:
                line_lower = line.lower()
                if "bond types" in line_lower: n_bond_types = int(line.split()[0])
                elif "angle types" in line_lower: n_angle_types = int(line.split()[0])
                elif "dihedral types" in line_lower: n_dihedral_types = int(line.split()[0])
                elif "improper types" in line_lower: n_improper_types = int(line.split()[0])

        type_info = []
        for type_id in sorted(data_info["masses"].keys()):
            mass = data_info["masses"][type_id]
            element = data_info["type_elements"].get(type_id, "X")
            name = data_info["type_names"].get(type_id, element)
            count = data_info["type_counts"].get(type_id, 0)
            type_info.append(f"  Type {type_id}: element={element}, mass={mass:.4f}, count={count}")
        type_info_str = "\n".join(type_info)

        force_field = force_field_info.get("force_field", {})
        if isinstance(force_field, dict):
            ff_name = force_field.get("force_field", "OPLS-AA")
            water_model = force_field.get("compatible_water_model", "TIP3P")
        else:
            ff_name = str(force_field) if force_field else "OPLS-AA"
            water_model = "TIP3P"

        self.logger.info(f"Data file has {n_atom_types} atom types, {n_bond_types} bond types, "
                         f"{n_angle_types} angle types, {n_dihedral_types} dihedral types")

        # ── SKILL INTEGRATION ── add skill context to parameter generation
        skill_context = self._get_skill_context(section="implementation")
        skill_block = ""
        if skill_context:
            skill_block = f"""
DOMAIN-SPECIFIC PARAMETER GUIDANCE:
{skill_context}
"""

        prompt = f"""
Generate a LAMMPS force field parameter file for this molecular system.

CRITICAL CONSTRAINTS - READ CAREFULLY:
1. The data file has EXACTLY {n_atom_types} atom types (numbered 1 to {n_atom_types})
2. DO NOT define parameters for atom types outside the range 1-{n_atom_types}
3. DO NOT include 'mass' commands - masses are already defined in the data file
4. DO NOT include 'units' or 'atom_style' commands - these are set elsewhere

ATOM TYPES IN THE DATA FILE:
{type_info_str}

TOPOLOGY FROM DATA FILE:
- Atom types: {n_atom_types}
- Bond types: {n_bond_types}
- Angle types: {n_angle_types}
- Dihedral types: {n_dihedral_types}
- Improper types: {n_improper_types}

FORCE FIELD: {ff_name}
WATER MODEL (if applicable): {water_model}
{skill_block}
GENERATE PARAMETERS IN THIS EXACT ORDER:

1. Pair style and settings:
   pair_style lj/cut/coul/long 10.0
   pair_modify mix geometric
   kspace_style pppm 1.0e-4
   special_bonds lj/coul 0.0 0.0 0.5

2. Bond/angle/dihedral styles (ONLY if corresponding types > 0):
   bond_style harmonic          # only if {n_bond_types} > 0
   angle_style harmonic         # only if {n_angle_types} > 0
   dihedral_style opls          # only if {n_dihedral_types} > 0
   improper_style harmonic      # only if {n_improper_types} > 0

3. Pair coefficients for EACH atom type (self-interactions):
   Format: pair_coeff I I epsilon sigma

4. Bond coefficients (if {n_bond_types} > 0):
   Format: bond_coeff N K r0

5. Angle coefficients (if {n_angle_types} > 0):
   Format: angle_coeff N K theta0

6. Dihedral coefficients (if {n_dihedral_types} > 0):
   Format for OPLS: dihedral_coeff N K1 K2 K3 K4

IMPORTANT REMINDERS:
- Generate pair_coeff for types 1 through {n_atom_types} ONLY
- Generate bond_coeff for types 1 through {n_bond_types} ONLY
- Generate angle_coeff for types 1 through {n_angle_types} ONLY
- Generate dihedral_coeff for types 1 through {n_dihedral_types} ONLY
- Do NOT generate any *_coeff commands if that type count is 0

Output ONLY valid LAMMPS commands. No markdown formatting, no explanations, no code blocks.
"""
        response = self._generate_text(prompt)
        param_content = response.strip()
        if param_content.startswith("```"):
            lines = param_content.split("\n")
            param_content = "\n".join(line for line in lines if not line.strip().startswith("```"))

        param_content = self._validate_and_fix_param_types(
            param_content, n_atom_types, n_bond_types,
            n_angle_types, n_dihedral_types, n_improper_types
        )

        header = f"""# Force field parameters
# Generated by ForceFieldAgent{' (skill: ' + self.skill_name + ')' if self.skill_name else ''}
# Force field: {ff_name}
# Water model: {water_model}
#
# Data file type counts:
#   Atom types: {n_atom_types}
#   Bond types: {n_bond_types}
#   Angle types: {n_angle_types}
#   Dihedral types: {n_dihedral_types}
#   Improper types: {n_improper_types}

"""
        return {"main": header + param_content, "additional": {}}

    def _validate_and_fix_param_types(self, param_content, n_atom_types,
                                       n_bond_types, n_angle_types,
                                       n_dihedral_types, n_improper_types=0):
        lines = param_content.split('\n')
        valid_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                valid_lines.append(line); continue
            if stripped.startswith('mass ') or stripped.startswith('mass\t'):
                self.logger.warning(f"Removing mass command: {stripped}"); continue
            if stripped.startswith('units '):
                self.logger.warning(f"Removing units command: {stripped}"); continue
            if stripped.startswith('atom_style '):
                self.logger.warning(f"Removing atom_style command: {stripped}"); continue
            if stripped.startswith('pair_coeff'):
                match = re.match(r'pair_coeff\s+(\d+)\s+(\d+)', stripped)
                if match:
                    t1, t2 = int(match.group(1)), int(match.group(2))
                    if t1 > n_atom_types or t2 > n_atom_types:
                        self.logger.warning(f"Removing invalid pair_coeff: {stripped}"); continue
                elif re.match(r'pair_coeff\s+\*\s+\*', stripped):
                    valid_lines.append(line); continue
                valid_lines.append(line); continue
            if stripped.startswith('bond_coeff'):
                if n_bond_types == 0: continue
                match = re.match(r'bond_coeff\s+(\d+)', stripped)
                if match and int(match.group(1)) > n_bond_types: continue
                valid_lines.append(line); continue
            if stripped.startswith('angle_coeff'):
                if n_angle_types == 0: continue
                match = re.match(r'angle_coeff\s+(\d+)', stripped)
                if match and int(match.group(1)) > n_angle_types: continue
                valid_lines.append(line); continue
            if stripped.startswith('dihedral_coeff'):
                if n_dihedral_types == 0: continue
                match = re.match(r'dihedral_coeff\s+(\d+)', stripped)
                if match and int(match.group(1)) > n_dihedral_types: continue
                valid_lines.append(line); continue
            if stripped.startswith('improper_coeff'):
                if n_improper_types == 0: continue
                match = re.match(r'improper_coeff\s+(\d+)', stripped)
                if match and int(match.group(1)) > n_improper_types: continue
                valid_lines.append(line); continue
            if stripped.startswith('bond_style') and n_bond_types == 0: continue
            if stripped.startswith('angle_style') and n_angle_types == 0: continue
            if stripped.startswith('dihedral_style') and n_dihedral_types == 0: continue
            if stripped.startswith('improper_style') and n_improper_types == 0: continue
            valid_lines.append(line)
        return '\n'.join(valid_lines)

    def _generate_lammps_parameters(self, data_file_info, parameter_info):
        self.logger.info("Generating LAMMPS parameter file content")
        force_field = parameter_info.get("force_field", "Unknown")
        param_str = json.dumps(parameter_info, indent=2)
        data_file_str = json.dumps(data_file_info, indent=2)

        prompt = f"""
As an expert in molecular dynamics force fields, generate LAMMPS parameter files for this system.

FORCE FIELD: {force_field}

DATA FILE INFORMATION:
{data_file_str}

PARAMETER INFORMATION:
{param_str}

IMPORTANT: Do NOT include 'units' commands in the parameter file.

Format the output as a JSON object:
{{
    "main": "# LAMMPS parameters for {force_field}\\n\\npair_style lj/cut/coul/long 10.0\\n...",
    "additional": {{}}
}}
Include only the JSON response with no additional text.
"""
        try:
            param_files = self._generate_json(prompt)
            if "main" not in param_files:
                param_files["main"] = self._generate_fallback_parameters(data_file_info, parameter_info)
            if "additional" not in param_files:
                param_files["additional"] = {}
            return param_files
        except Exception as e:
            self.logger.error(f"Error generating LAMMPS parameters: {e}")
            return {
                "main": self._generate_fallback_parameters(data_file_info, parameter_info),
                "additional": {}
            }

    def _generate_fallback_parameters(self, data_file_info, parameter_info):
        force_field = parameter_info.get("force_field", "Unknown")
        lines = [
            f"# LAMMPS parameters for {force_field}",
            "# Generated by ForceFieldAgent (fallback generator)", "",
            "pair_style lj/cut/coul/long 10.0",
            "bond_style harmonic", "angle_style harmonic", ""
        ]
        if data_file_info.get("dihedral_types", 0) > 0:
            lines.append("dihedral_style harmonic")
        if data_file_info.get("improper_types", 0) > 0:
            lines.append("improper_style harmonic")
        lines.append("")
        atom_types = data_file_info.get("atom_types", 0)
        if atom_types > 0:
            lines.append("# Pair coefficients")
            for i in range(1, atom_types + 1):
                mass = data_file_info.get("masses", {}).get(i, 12.0)
                element = self._guess_element_from_mass(mass)
                defaults = {"O": (0.1553, 3.166), "H": (0.0, 0.0), "C": (0.1094, 3.4),
                           "N": (0.17, 3.25), "Na": (0.1, 2.8), "K": (0.1, 2.8),
                           "Cl": (0.1, 4.4), "Br": (0.1, 4.4)}
                eps, sig = defaults.get(element, (0.1, 3.0))
                lines.append(f"pair_coeff {i} {i} {eps} {sig}  # {element}")
            lines.append("")
        bond_types = data_file_info.get("bond_types", 0)
        if bond_types > 0:
            lines.append("# Bond coefficients")
            for i in range(1, bond_types + 1):
                lines.append(f"bond_coeff {i} 450.0 1.0")
            lines.append("")
        angle_types = data_file_info.get("angle_types", 0)
        if angle_types > 0:
            lines.append("# Angle coefficients")
            for i in range(1, angle_types + 1):
                lines.append(f"angle_coeff {i} 55.0 109.47")
            lines.append("")
        lines += ["special_bonds lj/coul 0.0 0.0 0.5", "", "kspace_style pppm 1.0e-5"]
        return "\n".join(lines)

    def _generate_parameter_summary(self, params, selection_info):
        force_field = selection_info.get("force_field", {}).get("force_field", "Unknown")
        water_model = selection_info.get("force_field", {}).get("compatible_water_model", "Unknown")
        justification = selection_info.get("force_field", {}).get("justification", "")
        method = selection_info.get("parameter_method", {}).get("method", "unknown")
        method_desc = {"database": "Direct extraction from established databases",
                      "analogy": "Parameters by chemical analogy to known molecules",
                      "quantum": "Ab initio parameterization from quantum calculations"}.get(method, "Unknown method")
        atom_types_count = len(params.get("atom_types", {}))
        validation = params.get("validation", {})
        quality_score = validation.get("quality_metrics", {}).get("overall_score", 0)
        passed = validation.get("passed", False)
        warnings = validation.get("warnings", [])

        lines = [
            f"# Force Field Parameters Summary", "",
            f"## Selection Details",
            f"- **Force Field**: {force_field}",
            f"- **Water Model**: {water_model}",
            f"- **Parameter Source**: {method} ({method_desc})",
            f"- **Quality Score**: {quality_score}/100",
        ]
        # ── SKILL INTEGRATION ──
        if self.skill_name:
            lines.append(f"- **Skill Used**: {self.skill_name}")
        if params.get("pipeline") == "amber":
            lines.append(f"- **Pipeline**: AMBER (AmberTools → ParmEd → LAMMPS)")

        lines += [
            "", f"## Justification", f"{justification}", "",
            f"## Parameter Statistics",
            f"- **Atom Types**: {atom_types_count}",
            f"- **Bond Types**: {len(params.get('bonds', {}))}",
            f"- **Angle Types**: {len(params.get('angles', {}))}",
            f"- **Dihedral Types**: {len(params.get('dihedrals', {}))}",
            "", f"## Validation",
            f"- **Passed**: {'Yes' if passed else 'No'}", ""
        ]
        if warnings:
            lines.append("### Warnings")
            for warning in warnings:
                lines.append(f"- {warning}")
            lines.append("")
        return "\n".join(lines)

    def _fix_lammps_syntax(self, lammps_text):
        lines = lammps_text.split('\n')
        fixed_lines = []
        for line in lines:
            line = re.sub(r',(?=\S)', ', ', line)
            line = re.sub(r'(//.+)$', r'#\1', line)
            if line.strip().startswith("pair_coeff") and re.search(r'pair_coeff\s+\*\s+\*\s+[\d\.]+\s*$', line):
                line = line + " # Missing sigma parameter"
            fixed_lines.append(line)
        return '\n'.join(fixed_lines)

    # ================================================================
    # CHARGE ASSIGNMENT (unchanged except where noted)
    # ================================================================

    def assign_charges_to_data_file(self, data_file, parameter_info=None,
                                     pdb_file=None, research_goal=None,
                                     output_file=None):
        self.logger.info(f"Assigning charges to {data_file}")
        if output_file is None:
            base, ext = os.path.splitext(data_file)
            output_file = f"{base}_charged{ext}"
        data_info = self._parse_data_file_for_charges(data_file)
        if parameter_info and parameter_info.get("atom_types"):
            charge_assignments = self._generate_charge_assignments(
                data_info=data_info, parameter_info=parameter_info,
                pdb_file=pdb_file, research_goal=research_goal
            )
        else:
            charge_assignments = self._generate_charges_with_llm(
                data_info=data_info, pdb_file=pdb_file, research_goal=research_goal
            )
        validation = self._validate_charge_assignments(charge_assignments, data_info)
        if not validation["valid"]:
            self.logger.warning(f"Charge validation warnings: {validation['warnings']}")
        self._write_data_file_with_charges(
            input_file=data_file, output_file=output_file,
            charge_assignments=charge_assignments, data_info=data_info
        )
        charge_info_file = os.path.join(self.working_dir, "charge_assignments.json")
        with open(charge_info_file, 'w') as f:
            json.dump({
                "charge_assignments": charge_assignments,
                "validation": validation,
                "input_file": data_file, "output_file": output_file
            }, f, indent=2)
        self.logger.info(f"Charges assigned successfully: {output_file}")
        return output_file

    def _parse_data_file_for_charges(self, data_file):
        info = {
            "n_atoms": 0, "n_atom_types": 0, "masses": {},
            "type_names": {}, "type_elements": {}, "type_counts": {},
            "molecules": {}, "atom_style": "full", "box": None,
            "header_lines": [], "atoms_section_start": 0, "atoms_section_end": 0,
        }
        with open(data_file, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            parts = stripped.split()
            if len(parts) >= 2:
                if parts[1] == "atoms":
                    info["n_atoms"] = int(parts[0])
                elif parts[1] == "atom" and len(parts) >= 3 and parts[2] == "types":
                    info["n_atom_types"] = int(parts[0])
                elif "atom types" in stripped:
                    info["n_atom_types"] = int(parts[0])
            if "xlo xhi" in stripped:
                info["box"] = {"xlo": float(parts[0]), "xhi": float(parts[1])}
        self.logger.info(f"Data file header: {info['n_atoms']} atoms, {info['n_atom_types']} atom types")
        section_lines = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped in ["Masses", "Atoms", "Velocities", "Bonds", "Angles",
                           "Dihedrals", "Impropers", "Pair Coeffs", "Bond Coeffs",
                           "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"]:
                section_lines[stripped] = i
            elif stripped.startswith("Atoms"):
                section_lines["Atoms"] = i
                if "#" in stripped:
                    info["atom_style"] = stripped.split("#")[1].strip()
        if "Masses" in section_lines:
            start = section_lines["Masses"] + 1
            for i in range(start, len(lines)):
                line = lines[i].strip()
                if not line: continue
                if line in section_lines or line.startswith("Atoms") or line.startswith("Pair") or line.startswith("Bond"):
                    break
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        type_id = int(parts[0])
                        mass = float(parts[1])
                        if type_id > info["n_atom_types"]: continue
                        info["masses"][type_id] = mass
                        if "#" in line:
                            comment = line.split("#")[1].strip()
                            type_name = comment.split()[0] if comment.split() else None
                            if type_name:
                                info["type_names"][type_id] = type_name
                                info["type_elements"][type_id] = type_name
                        else:
                            element = self._guess_element_from_mass(mass)
                            info["type_elements"][type_id] = element
                            info["type_names"][type_id] = element
                    except (ValueError, IndexError):
                        continue
        if "Atoms" in section_lines:
            start = section_lines["Atoms"] + 1
            while start < len(lines) and (not lines[start].strip() or lines[start].strip().startswith("#")):
                start += 1
            info["atoms_section_start"] = start
            for type_id in info["masses"].keys():
                info["type_counts"][type_id] = 0
            for i in range(start, len(lines)):
                line = lines[i].strip()
                if not line:
                    blank_count = 0
                    for j in range(i, min(i + 3, len(lines))):
                        if not lines[j].strip(): blank_count += 1
                        elif lines[j].strip() in section_lines or lines[j].strip().startswith(("Velocities", "Bonds", "Angles")):
                            info["atoms_section_end"] = i; break
                    if blank_count >= 2:
                        info["atoms_section_end"] = i; break
                    continue
                if line.startswith("#"): continue
                if line in section_lines or line.startswith(("Velocities", "Bonds", "Angles", "Dihedrals", "Impropers")):
                    info["atoms_section_end"] = i; break
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        atom_id = int(parts[0])
                        mol_id = int(parts[1])
                        atom_type = int(parts[2])
                        if atom_type < 1 or atom_type > info["n_atom_types"]: continue
                        info["type_counts"][atom_type] = info["type_counts"].get(atom_type, 0) + 1
                        if mol_id not in info["molecules"]:
                            info["molecules"][mol_id] = []
                        info["molecules"][mol_id].append((atom_id, atom_type))
                    except (ValueError, IndexError):
                        continue
            else:
                info["atoms_section_end"] = len(lines)
        total_counted = sum(info["type_counts"].values())
        if total_counted != info["n_atoms"]:
            self.logger.warning(f"Atom count mismatch: header says {info['n_atoms']}, counted {total_counted}")
        return info

    def _get_default_charge_for_element(self, element):
        from ase.data import atomic_numbers
        if element not in atomic_numbers: return 0.0
        alkali = ["Li", "Na", "K", "Rb", "Cs"]
        if element in alkali: return 1.0
        alkaline_earth = ["Be", "Mg", "Ca", "Sr", "Ba"]
        if element in alkaline_earth: return 2.0
        transition_metals = {"Zn": 2.0, "Cu": 2.0, "Fe": 2.0, "Co": 2.0, "Ni": 2.0,
                            "Mn": 2.0, "Cr": 3.0, "Ti": 4.0, "Ag": 1.0, "Au": 1.0,
                            "Pt": 2.0, "Pd": 2.0, "Cd": 2.0, "Hg": 2.0}
        if element in transition_metals: return transition_metals[element]
        halogens = ["F", "Cl", "Br", "I"]
        if element in halogens: return -0.2
        defaults = {"O": -0.4, "H": 0.3, "N": -0.3, "S": 0.0}
        return defaults.get(element, 0.0)

    def _get_default_charges(self, data_info):
        charges = {}
        for type_id, element in data_info["type_elements"].items():
            charges[type_id] = self._get_default_charge_for_element(element)
        self.logger.warning(f"Using fallback default charges: {charges}")
        return charges

    def _analyze_molecular_compositions(self, data_info):
        mol_signatures = {}
        for mol_id, atoms in data_info["molecules"].items():
            type_counts = {}
            for atom_id, atom_type in atoms:
                type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
            signature = tuple(sorted(type_counts.items()))
            if signature not in mol_signatures:
                mol_signatures[signature] = {"count": 0, "type_counts": type_counts, "example_mol_id": mol_id}
            mol_signatures[signature]["count"] += 1
        mol_types = {}
        for signature, data in mol_signatures.items():
            mol_name = self._identify_molecule_type(data["type_counts"], data_info)
            composition_str = ", ".join([
                f"{data_info['type_elements'].get(t, '?')}{c}"
                for t, c in sorted(data["type_counts"].items())
            ])
            mol_types[mol_name] = {"count": data["count"], "composition": composition_str, "type_counts": data["type_counts"]}
        return mol_types

    def _identify_molecule_type(self, type_counts, data_info):
        element_counts = {}
        for type_id, count in type_counts.items():
            element = data_info["type_elements"].get(type_id, "X")
            element_counts[element] = element_counts.get(element, 0) + count
        n_atoms = sum(element_counts.values())
        if n_atoms == 1:
            return f"{list(element_counts.keys())[0].lower()}_ion"
        if n_atoms == 3 and element_counts.get("H", 0) == 2 and element_counts.get("O", 0) == 1:
            return "water"
        if n_atoms == 2 and element_counts.get("H", 0) == 1 and element_counts.get("O", 0) == 1:
            return "hydroxide"
        if n_atoms == 4 and element_counts.get("H", 0) == 3 and element_counts.get("O", 0) == 1:
            return "hydronium"
        if n_atoms == 4 and element_counts.get("N", 0) == 1 and element_counts.get("H", 0) == 3:
            return "ammonia"
        if n_atoms == 5 and element_counts.get("N", 0) == 1 and element_counts.get("H", 0) == 4:
            return "ammonium"
        if n_atoms == 5 and element_counts.get("C", 0) == 1 and element_counts.get("H", 0) == 4:
            return "methane"
        if n_atoms == 6 and element_counts.get("C", 0) == 1 and element_counts.get("H", 0) == 4 and element_counts.get("O", 0) == 1:
            return "methanol"
        if n_atoms == 9 and element_counts.get("C", 0) == 2 and element_counts.get("H", 0) == 6 and element_counts.get("O", 0) == 1:
            return "ethanol"
        if n_atoms == 3 and element_counts.get("C", 0) == 1 and element_counts.get("O", 0) == 2:
            return "carbon_dioxide"
        if n_atoms == 4 and element_counts.get("C", 0) == 1 and element_counts.get("O", 0) == 3:
            return "carbonate"
        if n_atoms == 5 and element_counts.get("H", 0) == 1 and element_counts.get("C", 0) == 1 and element_counts.get("O", 0) == 3:
            return "bicarbonate"
        if n_atoms == 4 and element_counts.get("N", 0) == 1 and element_counts.get("O", 0) == 3:
            return "nitrate"
        if n_atoms == 5 and element_counts.get("S", 0) == 1 and element_counts.get("O", 0) == 4 and element_counts.get("C", 0) == 0:
            return "sulfate"
        if n_atoms == 6 and element_counts.get("H", 0) == 1 and element_counts.get("S", 0) == 1 and element_counts.get("O", 0) == 4:
            return "bisulfate"
        if n_atoms == 5 and element_counts.get("P", 0) == 1 and element_counts.get("O", 0) == 4:
            return "phosphate"
        if n_atoms == 5 and element_counts.get("Cl", 0) == 1 and element_counts.get("O", 0) == 4:
            return "perchlorate"
        if element_counts.get("S", 0) >= 1 and element_counts.get("O", 0) >= 3 and element_counts.get("F", 0) >= 1:
            return "fluorosulfonate"
        if element_counts.get("S", 0) >= 1 and element_counts.get("O", 0) >= 3 and element_counts.get("F", 0) == 0 and element_counts.get("C", 0) >= 1:
            return "sulfonate"
        if element_counts.get("C", 0) >= 1 and element_counts.get("O", 0) >= 2 and element_counts.get("S", 0) == 0 and element_counts.get("N", 0) == 0 and element_counts.get("P", 0) == 0:
            c_count = element_counts.get("C", 0)
            o_count = element_counts.get("O", 0)
            if o_count == 2 or (o_count >= 2 and o_count <= c_count + 1):
                return "carboxylate"
        if element_counts.get("N", 0) >= 1 and element_counts.get("H", 0) >= 1 and element_counts.get("O", 0) == 0:
            return "amine"
        if element_counts.get("N", 0) >= 2 and element_counts.get("C", 0) >= 3 and element_counts.get("H", 0) >= 4:
            return "imidazolium"
        formula_parts = []
        if "C" in element_counts:
            cnt = element_counts["C"]
            formula_parts.append(f"C{cnt}" if cnt > 1 else "C")
        if "H" in element_counts:
            cnt = element_counts["H"]
            formula_parts.append(f"H{cnt}" if cnt > 1 else "H")
        for el in sorted(element_counts.keys()):
            if el not in ["C", "H"]:
                cnt = element_counts[el]
                formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
        return f"molecule_{''.join(formula_parts)}"

    def _map_data_types_to_ff_types(self, data_info, parameter_info, research_goal=None):
        self.logger.info("Mapping data file types to force field atom types via LLM")
        type_to_element = data_info["type_elements"]
        type_to_name = data_info.get("type_names", {})
        mol_compositions = self._analyze_molecular_compositions(data_info)
        type_to_context = self._build_type_to_context_map(mol_compositions)
        data_type_lines = []
        for type_id in sorted(data_info["masses"].keys()):
            element = type_to_element.get(type_id, "?")
            mass = data_info["masses"][type_id]
            count = data_info["type_counts"].get(type_id, 0)
            contexts = type_to_context.get(type_id, ["unknown"])
            name = type_to_name.get(type_id, element)
            data_type_lines.append(
                f"  DATA_TYPE_{type_id}: element={element}, mass={mass:.3f}, "
                f"count={count}, appears_in={contexts}, label='{name}'"
            )
        data_types_str = "\n".join(data_type_lines)
        ff_atom_types = parameter_info.get("atom_types", {})
        ff_type_lines = []
        ff_type_names = list(ff_atom_types.keys())
        for ff_name in ff_type_names:
            params = ff_atom_types[ff_name]
            mass = params.get("mass", 0)
            charge = params.get("charge", 0)
            desc = params.get("description", "")
            ff_type_lines.append(f"  FF_TYPE '{ff_name}': mass={mass:.3f}, charge={charge:+.4f}, description='{desc}'")
        ff_types_str = "\n".join(ff_type_lines)
        mol_desc_lines = []
        for mol_name, mol_data in mol_compositions.items():
            ec = {}
            for type_id, count in mol_data["type_counts"].items():
                el = type_to_element.get(type_id, "X")
                ec[el] = ec.get(el, 0) + count
            formula = self._build_molecular_formula(ec)
            type_list = [f"DATA_TYPE_{t}" for t in sorted(mol_data["type_counts"].keys())]
            mol_desc_lines.append(f"  {mol_name}: {mol_data['count']} molecules, formula={formula}, uses: [{', '.join(type_list)}]")
        mol_desc_str = "\n".join(mol_desc_lines)
        type_ids = sorted(data_info["masses"].keys())
        ff_names_list = ", ".join([f"'{name}'" for name in ff_type_names])

        prompt = f"""
You are mapping atom types from a LAMMPS data file to force field parameter names.

DATA FILE ATOM TYPES (these need to be mapped):
{data_types_str}

AVAILABLE FORCE FIELD TYPES (use these EXACT names in your response):
{ff_types_str}

MOLECULAR CONTEXT:
{mol_desc_str}

YOUR TASK:
For each DATA_TYPE, select the best matching FF_TYPE based on:
1. Element must match
2. Molecular context (water O != sulfonate O)
3. Description should match the chemical environment

AVAILABLE FF_TYPE NAMES: {ff_names_list}

Return a JSON object mapping data file type ID to FF type NAME:
{{
    {', '.join(f'"{t}": "<ff_type_name>"' for t in type_ids)}
}}

Data file type IDs to map: {', '.join(str(t) for t in type_ids)}
"""
        try:
            response = self._generate_json(prompt)
            type_mapping = {}
            valid_ff_types = set(ff_atom_types.keys())
            for key, ff_type in response.items():
                try:
                    type_id = int(key)
                    if type_id not in data_info["masses"]: continue
                    if ff_type.isdigit() or (isinstance(ff_type, str) and ff_type.lstrip('-').isdigit()):
                        try:
                            idx = int(ff_type) - 1
                            if 0 <= idx < len(ff_type_names):
                                ff_type = ff_type_names[idx]
                            else:
                                ff_type = self._find_closest_ff_type(type_to_element.get(type_id, "?"), ff_atom_types)
                        except (ValueError, IndexError):
                            ff_type = self._find_closest_ff_type(type_to_element.get(type_id, "?"), ff_atom_types)
                    if ff_type not in valid_ff_types:
                        ff_type = self._find_closest_ff_type(type_to_element.get(type_id, "?"), ff_atom_types)
                    type_mapping[type_id] = ff_type
                except (ValueError, TypeError):
                    continue
            missing = set(data_info["masses"].keys()) - set(type_mapping.keys())
            for type_id in missing:
                element = type_to_element.get(type_id, "?")
                type_mapping[type_id] = self._find_closest_ff_type(element, ff_atom_types)
            return type_mapping
        except Exception as e:
            self.logger.error(f"Error in LLM type mapping: {e}")
            return self._fallback_type_mapping(data_info, parameter_info)

    def _find_closest_ff_type(self, element, ff_atom_types, context=None):
        candidates = []
        for ff_name, params in ff_atom_types.items():
            desc = params.get("description", "").lower()
            mass = params.get("mass", 0)
            element_lower = element.lower()
            ff_name_lower = ff_name.lower()
            element_match = (element_lower in ff_name_lower or element_lower in desc or ff_name_lower.startswith(element_lower))
            try:
                from ase.data import atomic_masses, atomic_numbers
                target_mass = atomic_masses[atomic_numbers[element]]
                mass_match = abs(mass - target_mass) < 1.0
            except:
                mass_match = False
            if element_match or mass_match:
                score = 0
                if context:
                    context_lower = context.lower()
                    if "water" in context_lower:
                        if "water" in desc or ff_name_lower in ["ow", "hw", "o_w", "h_w", "oh2"]:
                            score += 10
                    elif "sulfon" in context_lower or "triflate" in context_lower:
                        if "sulfon" in desc or "sulfate" in desc:
                            score += 10
                        if "water" in desc:
                            score -= 5
                candidates.append((ff_name, score))
        if candidates:
            candidates.sort(key=lambda x: (-x[1], x[0]))
            return candidates[0][0]
        if ff_atom_types:
            return list(ff_atom_types.keys())[0]
        return element

    def _fallback_type_mapping(self, data_info, parameter_info):
        self.logger.warning("Using fallback type mapping (element-based)")
        type_to_element = data_info["type_elements"]
        ff_atom_types = parameter_info.get("atom_types", {})
        from ase.data import atomic_masses, chemical_symbols
        ff_by_element = {}
        for ff_name, params in ff_atom_types.items():
            mass = params.get("mass", 0)
            element = None
            for i, sym in enumerate(chemical_symbols[1:], 1):
                if abs(atomic_masses[i] - mass) < 0.5:
                    element = sym; break
            if element:
                ff_by_element.setdefault(element, []).append(ff_name)
        type_mapping = {}
        for type_id, element in type_to_element.items():
            if element in ff_by_element and ff_by_element[element]:
                type_mapping[type_id] = ff_by_element[element][0]
            else:
                type_mapping[type_id] = element
        return type_mapping

    def _generate_charges_with_llm(self, data_info, parameter_info=None,
                                      pdb_file=None, research_goal=None):
        self.logger.info("Generating charge assignments")
        ff_atom_types = parameter_info.get("atom_types", {}) if parameter_info else {}
        if ff_atom_types:
            self.logger.info("Using force field parameter mapping approach")
            type_mapping = self._map_data_types_to_ff_types(
                data_info=data_info, parameter_info=parameter_info, research_goal=research_goal
            )
            charge_assignments = {}
            type_to_element = data_info["type_elements"]
            for type_id, ff_type in type_mapping.items():
                if ff_type in ff_atom_types:
                    charge_assignments[type_id] = ff_atom_types[ff_type].get("charge", 0.0)
                else:
                    charge_assignments[type_id] = 0.0
            mol_compositions = self._analyze_molecular_compositions(data_info)
            self._log_molecular_charge_validation(charge_assignments, mol_compositions, type_to_element)
            validation_issues = self._check_molecule_charge_totals(
                charge_assignments, mol_compositions, type_to_element)
            if validation_issues:
                self.logger.warning(f"Molecule charge validation issues: {validation_issues}")
            return charge_assignments
        else:
            self.logger.info("No FF parameters available, using direct LLM charge generation")
            return self._generate_charges_with_llm(data_info=data_info, pdb_file=pdb_file, research_goal=research_goal)

    def _build_type_to_context_map(self, mol_compositions):
        type_to_context = {}
        for mol_name, mol_data in mol_compositions.items():
            for type_id in mol_data["type_counts"].keys():
                type_to_context.setdefault(type_id, [])
                if mol_name not in type_to_context[type_id]:
                    type_to_context[type_id].append(mol_name)
        return type_to_context

    def _build_molecular_formula(self, element_counts):
        formula_parts = []
        for el in ["C", "H"]:
            if el in element_counts:
                cnt = element_counts[el]
                formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
        for el in sorted(element_counts.keys()):
            if el not in ["C", "H"]:
                cnt = element_counts[el]
                formula_parts.append(f"{el}{cnt}" if cnt > 1 else el)
        return "".join(formula_parts)

    def _validate_and_fix_molecule_charges(self, charge_assignments, mol_compositions,
                                            expected_mol_charges, type_to_element, type_to_context):
        self.logger.info("Validating and fixing molecule charges...")
        for mol_name, mol_data in mol_compositions.items():
            expected = expected_mol_charges.get(mol_name, 0.0)
            current = sum(charge_assignments.get(t, 0.0) * c for t, c in mol_data["type_counts"].items())
            error = current - expected
            if abs(error) < 0.01: continue
            self.logger.warning(f"  {mol_name}: {current:.4f} (expected {expected}), error={error:.4f}")
            exclusive_types = [t for t in mol_data["type_counts"]
                             if len(type_to_context.get(t, [])) == 1 and type_to_context.get(t, [None])[0] == mol_name]
            if not exclusive_types:
                exclusive_types = [max(mol_data["type_counts"].items(), key=lambda x: x[1])[0]]
            total_atoms = sum(mol_data["type_counts"].get(t, 0) for t in exclusive_types)
            if total_atoms > 0:
                correction = -error / total_atoms
                for type_id in exclusive_types:
                    old = charge_assignments.get(type_id, 0.0)
                    charge_assignments[type_id] = old + correction
        return charge_assignments

    def _generate_charge_assignments(self, data_info, parameter_info=None,
                                      pdb_file=None, research_goal=None):
        self.logger.info("Generating charge assignments")
        ff_atom_types = {}
        if parameter_info:
            ff_atom_types = parameter_info.get("atom_types", {})
        if ff_atom_types:
            self.logger.info("Using force field parameter mapping approach")
            first_key = next(iter(ff_atom_types.keys()), "")
            if first_key.isdigit():
                ff_by_name = {}
                for key, params in ff_atom_types.items():
                    name = params.get("name", key)
                    ff_by_name[name] = params
            else:
                ff_by_name = ff_atom_types
            type_mapping = self._map_data_types_to_ff_types(
                data_info=data_info, parameter_info={"atom_types": ff_by_name}, research_goal=research_goal)
            charge_assignments = {}
            type_to_element = data_info["type_elements"]
            for type_id, ff_type in type_mapping.items():
                if ff_type in ff_by_name:
                    charge_assignments[type_id] = ff_by_name[ff_type].get("charge", 0.0)
                else:
                    charge_assignments[type_id] = 0.0
            mol_compositions = self._analyze_molecular_compositions(data_info)
            validation_issues = self._check_molecule_charge_totals(
                charge_assignments, mol_compositions, type_to_element)
            if validation_issues:
                expected_mol_charges = self._determine_expected_molecule_charges(mol_compositions, data_info)
                type_to_context = self._build_type_to_context_map(mol_compositions)
                charge_assignments = self._validate_and_fix_molecule_charges(
                    charge_assignments, mol_compositions, expected_mol_charges,
                    type_to_element, type_to_context)
            self._log_molecular_charge_validation(charge_assignments, mol_compositions, type_to_element)
            return charge_assignments
        else:
            self.logger.info("No FF parameters, using direct LLM generation")
            return self._generate_charges_direct_llm(data_info=data_info, pdb_file=pdb_file, research_goal=research_goal)

    def _check_molecule_charge_totals(self, charge_assignments, mol_compositions, type_to_element):
        issues = []
        expected_charges = self._determine_expected_molecule_charges(
            mol_compositions, {"type_elements": type_to_element})
        for mol_name, mol_data in mol_compositions.items():
            total_charge = sum(charge_assignments.get(type_id, 0.0) * count
                             for type_id, count in mol_data["type_counts"].items())
            expected = expected_charges.get(mol_name, 0.0)
            if abs(total_charge - expected) > 0.05:
                issues.append(f"{mol_name}: got {total_charge:.3f}, expected {expected:.1f}")
        return issues

    def _determine_expected_molecule_charges(self, mol_compositions, data_info):
        expected_charges = {}
        type_to_element = data_info.get("type_elements", {})
        for mol_name, mol_data in mol_compositions.items():
            element_counts = {}
            for type_id, count in mol_data["type_counts"].items():
                element = type_to_element.get(type_id, "X")
                element_counts[element] = element_counts.get(element, 0) + count
            expected_charges[mol_name] = self._infer_formal_charge_from_composition(mol_name, element_counts)
        return expected_charges

    def _infer_formal_charge_from_composition(self, mol_name, element_counts):
        if element_counts == {"H": 2, "O": 1}: return 0.0
        if len(element_counts) == 1 and list(element_counts.values())[0] == 1:
            element = list(element_counts.keys())[0]
            monatomic = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
                        "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2,
                        "Fe": 2, "Cu": 2, "Ni": 2, "Co": 2, "Al": 3,
                        "F": -1, "Cl": -1, "Br": -1, "I": -1}
            return float(monatomic.get(element, 0))
        if "S" in element_counts and element_counts.get("O", 0) == 3: return -1.0
        if element_counts.get("S", 0) == 1 and element_counts.get("O", 0) == 4 and "C" not in element_counts: return -2.0
        if element_counts == {"N": 1, "O": 3}: return -1.0
        if element_counts == {"P": 1, "O": 4}: return -3.0
        if element_counts == {"O": 1, "H": 1}: return -1.0
        if element_counts == {"N": 1, "H": 4}: return 1.0
        return 0.0

    def _generate_missing_charges(self, missing_types, data_info, type_to_context, existing_charges):
        self.logger.info(f"Generating charges for missing types: {missing_types}")
        type_to_element = data_info["type_elements"]
        missing_info = []
        for type_id in sorted(missing_types):
            element = type_to_element.get(type_id, "X")
            mass = data_info["masses"].get(type_id, 0)
            contexts = type_to_context.get(type_id, ["unknown"])
            missing_info.append(f"  Type {type_id}: element={element}, mass={mass:.3f}, appears_in=[{', '.join(contexts)}]")
        existing_str = ", ".join(f"{t}:{c:.4f}" for t, c in sorted(existing_charges.items()))
        prompt = f"""
You are assigning partial charges to atom types in a molecular simulation.

The following types still need charges:
{chr(10).join(missing_info)}

Already assigned charges (for context): {existing_str}

Return ONLY a JSON object:
{{
    {', '.join(f'"{t}": <charge>' for t in sorted(missing_types))}
}}
"""
        try:
            response = self._generate_json(prompt)
            charges = {}
            for key, value in response.items():
                try:
                    type_id = int(key)
                    if type_id in missing_types:
                        charges[type_id] = float(value) if not isinstance(value, dict) else float(value.get("charge", 0.0))
                except (ValueError, TypeError):
                    continue
            for type_id in missing_types - set(charges.keys()):
                charges[type_id] = self._get_default_charge_for_element(type_to_element.get(type_id, "X"))
            return charges
        except Exception:
            charges = {}
            for type_id in missing_types:
                charges[type_id] = self._get_default_charge_for_element(type_to_element.get(type_id, "X"))
            return charges

    def _generate_charges_fallback(self, data_info, type_to_context):
        self.logger.info("Using fallback charge generation")
        type_to_element = data_info["type_elements"]
        type_lines = []
        for type_id in sorted(data_info["masses"].keys()):
            element = type_to_element.get(type_id, "X")
            contexts = type_to_context.get(type_id, ["unknown"])
            type_lines.append(f"Type {type_id}: {element} in {contexts}")
        type_ids = sorted(data_info["masses"].keys())
        prompt = f"""
Assign partial charges for these atom types:

{chr(10).join(type_lines)}

Return JSON mapping type ID to charge:
{{
    {', '.join(f'"{t}": <charge>' for t in type_ids)}
}}
"""
        try:
            response = self._generate_json(prompt)
            charges = {}
            for key, value in response.items():
                try:
                    type_id = int(key)
                    if type_id in data_info["masses"]:
                        charges[type_id] = float(value) if not isinstance(value, dict) else float(value.get("charge", 0.0))
                except (ValueError, TypeError):
                    continue
            for type_id in data_info["masses"].keys():
                if type_id not in charges:
                    charges[type_id] = self._get_default_charge_for_element(type_to_element.get(type_id, "X"))
            return charges
        except Exception:
            return self._get_default_charges(data_info)

    def _log_molecular_charge_validation(self, charges, mol_compositions, type_to_element):
        self.logger.info("Validating molecular charges:")
        for mol_name, mol_data in mol_compositions.items():
            mol_charge = 0.0
            breakdown = []
            for type_id, count in mol_data["type_counts"].items():
                charge = charges.get(type_id, 0.0)
                contribution = charge * count
                mol_charge += contribution
                element = type_to_element.get(type_id, "?")
                breakdown.append(f"{element}(type{type_id}):{charge:.3f}x{count}={contribution:.3f}")
            self.logger.info(f"  {mol_name}: total_charge={mol_charge:.4f} [{', '.join(breakdown)}]")

    def _extract_charges_from_parameters(self, parameter_info, data_info):
        self.logger.info("Attempting to extract charges from parameter_info")
        n_param_types = len(parameter_info.get("atom_types", {}))
        n_data_types = len(data_info["masses"])
        if n_data_types > n_param_types:
            return self._generate_charge_assignments(data_info)
        if n_param_types == 0:
            return self._generate_charge_assignments(data_info)
        element_charges = {}
        atom_types = parameter_info.get("atom_types", {})
        for type_id_str, type_info in atom_types.items():
            if not isinstance(type_info, dict): continue
            element = type_info.get("name") or type_info.get("element")
            charge = type_info.get("charge")
            if element is not None and charge is not None:
                try:
                    charge = float(charge)
                    element_normalized = self._normalize_element_name(element)
                    if element_normalized not in element_charges:
                        element_charges[element_normalized] = charge
                except (ValueError, TypeError):
                    pass
        if not element_charges:
            return self._generate_charge_assignments(data_info)
        charges = {}
        data_type_to_element = data_info["type_elements"]
        for type_id, element in data_type_to_element.items():
            element_normalized = self._normalize_element_name(element)
            if element_normalized in element_charges:
                charges[type_id] = element_charges[element_normalized]
        missing_types = set(data_info["masses"].keys()) - set(charges.keys())
        if missing_types:
            llm_charges = self._generate_charge_assignments(data_info)
            for type_id in missing_types:
                charges[type_id] = llm_charges.get(type_id, self._get_default_charge_for_element(data_type_to_element.get(type_id, "X")))
        return charges

    def _normalize_element_name(self, element):
        if not element: return "X"
        element = str(element).strip()
        if len(element) <= 2 and element.isalpha():
            return element.capitalize()
        element_lower = element.lower()
        element_names = {
            "hydrogen": "H", "carbon": "C", "nitrogen": "N", "oxygen": "O",
            "fluorine": "F", "sulfur": "S", "chlorine": "Cl", "sodium": "Na",
            "potassium": "K", "calcium": "Ca", "magnesium": "Mg", "zinc": "Zn",
            "iron": "Fe", "copper": "Cu", "phosphorus": "P", "bromine": "Br",
        }
        for name, symbol in element_names.items():
            if name in element_lower: return symbol
        variations = {
            "ow": "O", "oh": "O", "o_w": "O", "ot": "O", "os": "O", "o": "O",
            "hw": "H", "ho": "H", "h_w": "H", "ht": "H", "hs": "H", "h": "H",
            "ct": "C", "ca": "C", "c3": "C", "c2": "C", "cx": "C", "c": "C",
            "s": "S", "sh": "S", "ss": "S", "f": "F", "f1": "F",
            "zn": "Zn", "zn2+": "Zn", "zn+2": "Zn",
            "na": "Na", "na+": "Na", "cl": "Cl", "cl-": "Cl",
            "fe": "Fe", "ca_ion": "Ca", "mg_ion": "Mg", "zn_ion": "Zn",
        }
        if element_lower in variations: return variations[element_lower]
        first_word = element_lower.split()[0] if element_lower.split() else element_lower
        if first_word in variations: return variations[first_word]
        if len(element) >= 2 and element[1].islower():
            return element[:2].capitalize()
        return element[0].upper()

    def _validate_charge_assignments(self, charge_assignments, data_info):
        validation = {"valid": True, "warnings": [], "errors": [], "total_charge": 0.0, "molecule_charges": {}}
        for type_id in data_info["masses"].keys():
            if type_id not in charge_assignments:
                validation["errors"].append(f"Missing charge for atom type {type_id}")
                validation["valid"] = False
        total_charge = 0.0
        for type_id, count in data_info["type_counts"].items():
            total_charge += charge_assignments.get(type_id, 0.0) * count
        validation["total_charge"] = total_charge
        rounded_total = round(total_charge)
        if abs(total_charge - rounded_total) > 0.1:
            validation["warnings"].append(f"Total system charge ({total_charge:.4f}) is not close to integer {rounded_total}")
        mol_compositions = self._analyze_molecular_compositions(data_info)
        for mol_name, mol_data in mol_compositions.items():
            mol_charge = sum(charge_assignments.get(type_id, 0.0) * count for type_id, count in mol_data["type_counts"].items())
            validation["molecule_charges"][mol_name] = mol_charge
            rounded_mol_charge = round(mol_charge)
            if abs(mol_charge - rounded_mol_charge) > 0.15:
                validation["warnings"].append(f"{mol_name} charge ({mol_charge:.4f}) deviates from integer {rounded_mol_charge}")
        for type_id, charge in charge_assignments.items():
            element = data_info["type_elements"].get(type_id, "?")
            if abs(charge) > 4.0:
                validation["warnings"].append(f"Unusually large charge for type {type_id} ({element}): {charge:.4f}")
            if element == "H" and charge < -0.5:
                validation["warnings"].append(f"Unusually negative hydrogen (type {type_id}): {charge:.4f}")
            if element == "O" and charge > 1.0:
                validation["warnings"].append(f"Unusually positive oxygen (type {type_id}): {charge:.4f}")
            if element == "F" and charge > 0.3:
                validation["warnings"].append(f"Positive charge on fluorine (type {type_id}): {charge:.4f}")
            if element in ["Cl", "Br", "I"] and charge > 0.5:
                validation["warnings"].append(f"Unusually positive halogen {element} (type {type_id}): {charge:.4f}")
            if element in ["Li", "Na", "K", "Rb", "Cs"] and charge < 0.5:
                validation["warnings"].append(f"Alkali metal {element} (type {type_id}) has low charge: {charge:.4f}")
            if element in ["Mg", "Ca", "Zn", "Fe", "Cu", "Ni", "Co", "Mn"] and charge < 0:
                validation["warnings"].append(f"Negative charge on metal {element} (type {type_id}): {charge:.4f}")
        return validation

    def _write_data_file_with_charges(self, input_file, output_file, charge_assignments, data_info):
        with open(input_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        in_atoms_section = False
        charges_written = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                in_atoms_section = True
                new_lines.append(line); continue
            if in_atoms_section:
                if stripped in ["Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
                              "Pair Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"]:
                    in_atoms_section = False
                    new_lines.append(line); continue
                if not stripped or stripped.startswith("#"):
                    new_lines.append(line); continue
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        atom_id = parts[0]
                        mol_id = parts[1]
                        atom_type = int(parts[2])
                        x, y, z = parts[4], parts[5], parts[6]
                        new_charge = charge_assignments.get(atom_type, 0.0)
                        comment = ""
                        if "#" in line:
                            comment = " #" + line.split("#", 1)[1].rstrip("\n")
                        new_line = f"{atom_id:>8} {mol_id:>8} {atom_type:>4} {new_charge:>12.6f} {x:>14} {y:>14} {z:>14}{comment}\n"
                        new_lines.append(new_line)
                        charges_written += 1; continue
                    except (ValueError, IndexError):
                        pass
            new_lines.append(line)
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
        self.logger.info(f"Wrote data file with charges: {output_file} ({charges_written} atoms)")
        if charges_written == 0:
            self.logger.error("WARNING: No charges were written!")

    # ================================================================
    # COMPLETE PARAMETERIZATION WORKFLOW (modified for skill awareness)
    # ================================================================

    def complete_parameterization(self,
                                   pdb_file: str,
                                   data_file: str,
                                   research_goal: str,
                                   system_description: Optional[str] = None,
                                   small_molecule_info: Optional[List[Dict[str, Any]]] = None,
                                   solvate: bool = False,
                                   box_buffer: float = 10.0,
                                   neutralize: bool = True) -> Dict[str, Any]:
        """
        Complete parameterization workflow.

        If the AMBER skill is active and AmberTools are available, uses the
        AMBER pipeline (which produces a self-contained data file with all
        coefficients and charges). Otherwise uses the existing LLM-based workflow.
        """
        self.logger.info("=" * 60)
        self.logger.info("COMPLETE PARAMETERIZATION WORKFLOW")
        self.logger.info("=" * 60)

        results = {
            "status": "success",
            "input_files": {"pdb_file": pdb_file, "data_file": data_file},
            "output_files": {},
            "force_field": None,
            "errors": [],
            "warnings": []
        }

        try:
            # Step 1: Select force field (may auto-load skill)
            self.logger.info("\n[Step 1] Selecting force field...")
            selection_info = self.select_force_field(
                pdb_file=pdb_file,
                research_goal=research_goal,
                system_description=system_description
            )
            results["force_field"] = selection_info["force_field"]
            results["skill_used"] = self.skill_name

            # Step 2: Acquire parameters (may route to AMBER pipeline)
            self.logger.info("\n[Step 2] Acquiring force field parameters...")
            parameter_info = self.acquire_parameters(
                selection_info=selection_info,
                data_file=data_file,
                pdb_file=pdb_file,
                small_molecule_info=small_molecule_info,
                solvate=solvate,
                box_buffer=box_buffer,
                neutralize=neutralize,
            )
            results["output_files"]["parameter_info"] = os.path.join(
                self.working_dir, "parameter_info.json"
            )

            # ── SKILL INTEGRATION ── Check if AMBER pipeline was used
            if parameter_info.get("pipeline") == "amber":
                # AMBER pipeline produces a complete data file
                self.logger.info("AMBER pipeline produced a self-contained data file")
                results["output_files"]["charged_data_file"] = parameter_info["data_file"]
                results["output_files"]["prmtop"] = parameter_info.get("prmtop")
                results["output_files"]["inpcrd"] = parameter_info.get("inpcrd")

                # Generate LAMMPS parameter file (just the input header)
                param_files = self.generate_lammps_parameters(
                    parameter_info=parameter_info,
                    data_file=parameter_info["data_file"]
                )
                results["output_files"]["parameter_files"] = param_files

            else:
                # Standard LLM-based workflow
                # Step 2.5: Split atom types if needed
                self.logger.info("\n[Step 2.5] Checking for atom types needing splitting...")
                data_file = self.split_atom_types_by_molecule_context(data_file)

                # Step 3: Assign charges
                self.logger.info("\n[Step 3] Assigning charges to data file...")
                charged_data_file = self.assign_charges_to_data_file(
                    data_file=data_file,
                    parameter_info=parameter_info,
                    pdb_file=pdb_file,
                    research_goal=research_goal
                )
                results["output_files"]["charged_data_file"] = charged_data_file

                # Step 4: Generate LAMMPS parameter files
                self.logger.info("\n[Step 4] Generating LAMMPS parameter files...")
                param_files = self.generate_lammps_parameters(
                    parameter_info=parameter_info,
                    data_file=charged_data_file
                )
                results["output_files"]["parameter_files"] = param_files

            # Collect warnings
            charge_info_file = os.path.join(self.working_dir, "charge_assignments.json")
            if os.path.exists(charge_info_file):
                with open(charge_info_file, 'r') as f:
                    charge_info = json.load(f)
                    results["warnings"].extend(
                        charge_info.get("validation", {}).get("warnings", [])
                    )

            self.logger.info("\n" + "=" * 60)
            self.logger.info("PARAMETERIZATION COMPLETE")
            self.logger.info("=" * 60)
            self.logger.info(f"Force field: {results['force_field'].get('force_field', 'Unknown')}")
            if self.skill_name:
                self.logger.info(f"Skill used: {self.skill_name}")
            self.logger.info(f"Output files: {list(results['output_files'].keys())}")

        except Exception as e:
            self.logger.error(f"Parameterization failed: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
            import traceback
            results["traceback"] = traceback.format_exc()

        return results

    def split_atom_types_by_molecule_context(self, data_file, output_file=None):
        self.logger.info(f"Analyzing atom types for context-based splitting: {data_file}")
        if output_file is None:
            output_file = data_file
        data_info = self._parse_data_file_for_charges(data_file)
        mol_compositions = self._analyze_molecular_compositions(data_info)
        element_to_molecules = {}
        for mol_name, mol_data in mol_compositions.items():
            for type_id in mol_data["type_counts"].keys():
                element = data_info["type_elements"].get(type_id, "X")
                element_to_molecules.setdefault(element, set()).add(mol_name)
        elements_to_split = {el: mols for el, mols in element_to_molecules.items() if len(mols) > 1}
        if not elements_to_split:
            self.logger.info("No atom types need splitting")
            return data_file
        self.logger.info(f"Elements in multiple contexts: {elements_to_split}")
        with open(data_file, 'r') as f:
            lines = f.readlines()
        old_n_types = data_info["n_atom_types"]
        new_type_counter = old_n_types
        type_split_map = {}
        new_type_info = {}
        for element, mol_names in elements_to_split.items():
            original_type = None
            for type_id, el in data_info["type_elements"].items():
                if el == element and type_id <= old_n_types:
                    original_type = type_id; break
            if original_type is None: continue
            mol_list = sorted(mol_names, key=lambda m: -mol_compositions[m]["count"])
            type_split_map[original_type] = {}
            for i, mol_name in enumerate(mol_list):
                if i == 0:
                    type_split_map[original_type][mol_name] = original_type
                else:
                    new_type_counter += 1
                    type_split_map[original_type][mol_name] = new_type_counter
                    new_type_info[new_type_counter] = {
                        "mass": data_info["masses"][original_type],
                        "element": element, "context": mol_name
                    }
        if not new_type_info:
            self.logger.info("No new types needed after analysis")
            return data_file
        mol_id_to_name = {}
        for mol_name, mol_data in mol_compositions.items():
            for mol_id, atoms in data_info["molecules"].items():
                type_counts = {}
                for atom_id, atom_type in atoms:
                    type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
                if type_counts == mol_data["type_counts"]:
                    mol_id_to_name[mol_id] = mol_name
        new_lines = []
        in_masses = False
        in_atoms = False
        masses_collected = []
        header_updated = False
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not header_updated and "atom types" in stripped.lower():
                parts = stripped.split()
                try:
                    new_lines.append(f" {new_type_counter} atom types\n")
                    header_updated = True
                    i += 1; continue
                except (ValueError, IndexError):
                    pass
            if stripped == "Masses":
                in_masses = True
                new_lines.append(line); i += 1; continue
            if in_masses:
                if stripped.startswith("Atoms") or stripped.startswith("Pair"):
                    masses_collected.sort(key=lambda x: x[0])
                    new_lines.append("\n")
                    for type_id, mass_line in masses_collected:
                        new_lines.append(mass_line)
                    for new_type in sorted(new_type_info.keys()):
                        info = new_type_info[new_type]
                        new_lines.append(f" {new_type} {info['mass']:.6f} # {info['element']} ({info['context']})\n")
                    new_lines.append("\n")
                    in_masses = False
                    new_lines.append(line); i += 1; continue
                if not stripped: i += 1; continue
                if stripped.startswith("#"): i += 1; continue
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        masses_collected.append((int(parts[0]), line))
                    except ValueError:
                        new_lines.append(line)
                i += 1; continue
            if stripped.startswith("Atoms"):
                in_atoms = True
                new_lines.append(line); i += 1; continue
            if in_atoms and stripped in ["Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"]:
                in_atoms = False
                new_lines.append(line); i += 1; continue
            if in_atoms and stripped and not stripped.startswith("#"):
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        atom_id = int(parts[0])
                        mol_id = int(parts[1])
                        old_type = int(parts[2])
                        charge = parts[3]
                        x, y, z = parts[4], parts[5], parts[6]
                        comment = ""
                        if "#" in line:
                            comment = " #" + line.split("#", 1)[1].rstrip("\n")
                        new_type = old_type
                        if old_type in type_split_map:
                            mol_name = mol_id_to_name.get(mol_id)
                            if mol_name and mol_name in type_split_map[old_type]:
                                new_type = type_split_map[old_type][mol_name]
                        new_line = f"{atom_id:>8} {mol_id:>8} {new_type:>4} {charge:>12} {x:>14} {y:>14} {z:>14}{comment}\n"
                        new_lines.append(new_line); i += 1; continue
                    except (ValueError, IndexError):
                        pass
            new_lines.append(line); i += 1
        with open(output_file, 'w') as f:
            f.writelines(new_lines)
        self.logger.info(f"Split {len(new_type_info)} atom types, total now: {new_type_counter}")
        return output_file

    # ================================================================
    # DIAGNOSIS AND FIXING (unchanged)
    # ================================================================

    def diagnose_and_fix_force_field(self, quality_result, research_goal, data_file, ff_params_path, stage):
        self.logger.info(f"Diagnosing quality issues for stage: {stage}")
        result = {"ff_modified": False, "charges_modified": False, "diagnosis": "", "details": {}}
        issues = quality_result.get("issues", [])
        recommendations = quality_result.get("recommendations", [])
        issue_text = " ".join([i.get("description", "").lower() for i in issues])
        rec_text = " ".join([(r.get("description", "") if isinstance(r, dict) else str(r)).lower() for r in recommendations])
        all_text = issue_text + " " + rec_text
        needs_ff = any(kw in all_text for kw in [
            "density", "volume", "coordination", "rdf", "radial", "solvation",
            "energy", "lj", "lennard", "sigma", "epsilon", "mixing rule", "pair_modify"])
        needs_charges = any(kw in all_text for kw in [
            "charge", "electrostatic", "coulomb", "neutral", "dipole", "density", "structure"])
        if needs_ff and os.path.exists(ff_params_path):
            ff_fixed, ff_info = self._diagnose_ff_params(quality_result, research_goal, data_file, ff_params_path, stage)
            result["ff_modified"] = ff_fixed
            result["details"]["force_field"] = ff_info
            if ff_fixed: result["ff_backup"] = ff_info.get("backup")
        if needs_charges:
            charge_fixed, charge_info = self._diagnose_charges(quality_result, research_goal, data_file, stage)
            result["charges_modified"] = charge_fixed
            result["details"]["charges"] = charge_info
            if charge_fixed: result["charge_backup"] = charge_info.get("backup")
        diagnosis_parts = []
        if result["ff_modified"]:
            diagnosis_parts.append(f"FF parameters adjusted: {result['details']['force_field'].get('summary', '')}")
        if result["charges_modified"]:
            diagnosis_parts.append(f"Charges adjusted: {result['details']['charges'].get('summary', '')}")
        if not diagnosis_parts:
            diagnosis_parts.append("No parameter changes needed")
        result["diagnosis"] = "; ".join(diagnosis_parts)
        return result

    def _diagnose_ff_params(self, quality_result, research_goal, data_file, ff_params_path, stage):
        with open(ff_params_path, 'r') as f:
            current_ff = f.read()
        data_info = self._parse_data_file_for_charges(data_file)
        type_info = []
        for type_id in sorted(data_info["masses"].keys()):
            element = data_info["type_elements"].get(type_id, "?")
            mass = data_info["masses"][type_id]
            name = data_info.get("type_names", {}).get(type_id, element)
            count = data_info["type_counts"].get(type_id, 0)
            type_info.append(f"  Type {type_id}: {name} ({element}), mass={mass:.3f}, count={count}")
        type_info_str = "\n".join(type_info)
        log_file = os.path.join(self.working_dir, "log.lammps")
        thermo_excerpt = ""
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
            thermo_lines = [l for l in log_lines if l.strip() and l.strip()[0].isdigit() and len(l.split()) >= 5]
            if thermo_lines: thermo_excerpt = "".join(thermo_lines[-20:])
        issues_str = "\n".join([f"- [{i.get('severity', '?')}] {i.get('description', '')}" for i in quality_result.get("issues", [])])
        metrics_str = json.dumps(quality_result.get("quality_metrics", {}), indent=2, default=str)

        prompt = f"""
You are an expert molecular dynamics force field specialist. A LAMMPS simulation
completed but quality analysis found physics problems.

RESEARCH GOAL: {research_goal}
STAGE: {stage}

QUALITY ISSUES:
{issues_str}

QUALITY METRICS:
{metrics_str}

ATOM TYPES IN DATA FILE:
{type_info_str}

CURRENT FORCE FIELD (ff_params.lammps):
{current_ff}

RECENT THERMO OUTPUT:
{thermo_excerpt[-1500:]}

Return JSON:
{{
    "diagnosis": "What is wrong and why",
    "changes_needed": true/false,
    "changes": [{{"what": "Description", "why": "Physical justification", "original_line": "...", "corrected_line": "..."}}],
    "corrected_ff_params": "Complete corrected ff_params.lammps content or null"
}}
"""
        try:
            result = self._generate_json(prompt)
            diagnosis = result.get("diagnosis", "No diagnosis")
            if not result.get("changes_needed", False):
                return False, {"diagnosis": diagnosis, "changes_needed": False}
            corrected_ff = result.get("corrected_ff_params")
            if corrected_ff and corrected_ff.strip() != current_ff.strip():
                backup_path = ff_params_path + f".before_quality_{stage}"
                shutil.copy2(ff_params_path, backup_path)
                with open(ff_params_path, 'w') as f:
                    f.write(corrected_ff)
                changes = result.get("changes", [])
                return True, {"summary": f"{len(changes)} parameter changes: {diagnosis[:80]}", "diagnosis": diagnosis, "changes": changes, "backup": backup_path}
            return False, {"diagnosis": diagnosis, "no_actual_changes": True}
        except Exception as e:
            self.logger.error(f"FF diagnosis failed: {e}")
            return False, {"error": str(e)}

    def _diagnose_charges(self, quality_result, research_goal, data_file, stage):
        data_info = self._parse_data_file_for_charges(data_file)
        mol_compositions = self._analyze_molecular_compositions(data_info)
        charge_file = os.path.join(self.working_dir, "charge_assignments.json")
        current_charges = {}
        if os.path.exists(charge_file):
            with open(charge_file, 'r') as f:
                charge_data = json.load(f)
                current_charges = charge_data.get("charge_assignments", {})
        if not current_charges:
            return False, {"error": "No charge assignments found"}
        type_info = []
        for type_id in sorted(data_info["masses"].keys()):
            element = data_info["type_elements"].get(type_id, "?")
            name = data_info.get("type_names", {}).get(type_id, element)
            charge = current_charges.get(str(type_id), "?")
            count = data_info["type_counts"].get(type_id, 0)
            type_info.append(f"  Type {type_id}: {name} ({element}), charge={charge}, count={count}")
        type_info_str = "\n".join(type_info)
        mol_info = []
        for mol_name, mol_data in mol_compositions.items():
            mol_charge = sum(float(current_charges.get(str(tid), 0)) * cnt for tid, cnt in mol_data["type_counts"].items())
            mol_info.append(f"  {mol_name}: {mol_data['count']} molecules, charge={mol_charge:.4f}")
        mol_info_str = "\n".join(mol_info)
        issues_str = "\n".join([f"- [{i.get('severity', '?')}] {i.get('description', '')}" for i in quality_result.get("issues", [])])

        prompt = f"""
You are an expert in molecular dynamics charge parameterization.

RESEARCH GOAL: {research_goal}

QUALITY ISSUES:
{issues_str}

ATOM TYPES AND CURRENT CHARGES:
{type_info_str}

MOLECULAR CHARGES:
{mol_info_str}

Return JSON:
{{
    "diagnosis": "Analysis of charge issues",
    "changes_needed": true/false,
    "corrected_charges": {{"type_id_string": new_charge_float}}
}}
"""
        try:
            result = self._generate_json(prompt)
            diagnosis = result.get("diagnosis", "No diagnosis")
            if not result.get("changes_needed", False):
                return False, {"diagnosis": diagnosis}
            corrected_charges = result.get("corrected_charges")
            if not corrected_charges:
                return False, {"diagnosis": diagnosis, "no_corrections": True}
            int_charges = {}
            for k, v in corrected_charges.items():
                try:
                    int_charges[int(k)] = float(v)
                except (ValueError, TypeError):
                    continue
            backup_path = data_file + f".before_charge_fix_{stage}"
            shutil.copy2(data_file, backup_path)
            self._write_data_file_with_charges(data_file, data_file, int_charges, data_info)
            new_validation = self._validate_charge_assignments(int_charges, data_info)
            with open(charge_file, 'w') as f:
                json.dump({
                    "charge_assignments": {str(k): v for k, v in int_charges.items()},
                    "validation": new_validation,
                    "quality_fix_stage": stage,
                    "previous_charges": current_charges
                }, f, indent=2)
            return True, {"summary": f"Updated {len(int_charges)} types", "diagnosis": diagnosis, "backup": backup_path}
        except Exception as e:
            self.logger.error(f"Charge diagnosis failed: {e}")
            return False, {"error": str(e)}
