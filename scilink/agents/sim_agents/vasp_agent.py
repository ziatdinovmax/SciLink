"""
VASP-flavored backward-compat subclass of PeriodicDFTAgent.

The general agent has been moved to PeriodicDFTAgent (scale-aware,
software-agnostic). New code should use::

    from scilink.agents.sim_agents import PeriodicDFTAgent
    agent = PeriodicDFTAgent(...)
    result = agent.generate_inputs(structure_file, request, software="vasp")
    agent.save_inputs(result, output_dir)

This module preserves the historical VASP-specific API for existing
callers (simulation_orchestrator_tools, vasp_quality, vasp_updater,
user scripts, the wizard, etc.):

    from scilink.agents.sim_agents.vasp_agent import VaspInputAgent
    agent = VaspInputAgent(config=VASPProjectConfig(...))
    result = agent.generate_vasp_inputs(poscar_path, request)
    # result has top-level "incar" / "kpoints" keys alongside input_files
    agent.save_inputs(result)         # writes INCAR + KPOINTS
"""

import os
import logging
from typing import Optional

from .periodic_dft_agent import PeriodicDFTAgent

__all__ = ["VaspInputAgent"]


class VaspInputAgent(PeriodicDFTAgent):
    """
    VASP-specific subclass of PeriodicDFTAgent.

    Adds back:
      - ``config`` (VASPProjectConfig) handling for deterministic
        post-LLM INCAR corrections
      - ``generate_vasp_inputs(poscar_path, original_request, ...)``
        legacy method signature with VASP-flat result shape
      - ``apply_improvements(original_incar, ..., poscar_path, ...)``
        legacy signature

    Internally delegates to PeriodicDFTAgent's generic methods.
    """

    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 config: Optional["VASPProjectConfig"] = None,
                 local_model: str = None,
                 google_api_key: str = None):
        super().__init__(
            api_key=api_key, model_name=model_name, base_url=base_url,
            local_model=local_model, google_api_key=google_api_key,
        )
        self.config = config

    @staticmethod
    def _to_legacy_shape(result: dict) -> dict:
        """Copy result['input_files'][<NAME>] up to result['<name>'] keys."""
        files = result.get("input_files") or {}
        for filename, content in files.items():
            result[filename.lower()] = content
        return result

    def _apply_config(self, result: dict,
                      config: Optional["VASPProjectConfig"] = None) -> dict:
        """Apply VASPProjectConfig to the generated INCAR (if present)."""
        active = config or self.config
        if active is None or result.get("status") != "success":
            return result
        if "incar" not in result:
            return result
        config_result = active.apply_to_incar(result["incar"])
        result["incar"] = config_result["incar"]
        # Keep input_files in sync with the corrected INCAR
        if "input_files" in result and "INCAR" in result["input_files"]:
            result["input_files"]["INCAR"] = config_result["incar"]
        result["config_changes"] = config_result["changes"]
        if config_result["changes"]:
            logging.getLogger(__name__).info(
                f"Applied {len(config_result['changes'])} config corrections"
            )
        return result

    def generate_vasp_inputs(self, poscar_path: str,
                             original_request: str,
                             skill: Optional[str] = "vasp",
                             config: Optional["VASPProjectConfig"] = None) -> dict:
        """
        Backward-compat wrapper. Generates VASP INCAR + KPOINTS.

        Returns the historical flat shape with top-level ``"incar"`` /
        ``"kpoints"`` keys, alongside the new generic
        ``"input_files"`` dict.
        """
        result = self.generate_inputs(
            structure_file=poscar_path,
            request=original_request,
            software="vasp",
            skill=skill,
        )
        result = self._to_legacy_shape(result)
        result = self._apply_config(result, config=config)
        return result

    def save_inputs(self, result: dict, output_dir: str = ".") -> dict:
        """
        Save VASP input files. Uses the generic input_files map when
        present; falls back to writing flat ``incar`` / ``kpoints`` keys
        (for results produced before this refactor was deployed).
        """
        if result.get("status") != "success":
            return {"error": "Generation was not successful"}

        if "input_files" in result and isinstance(result["input_files"], dict):
            saved = super().save_inputs(result, output_dir)
        else:
            # Legacy fallback: result has top-level incar/kpoints only.
            os.makedirs(output_dir, exist_ok=True)
            saved = {}
            try:
                if "incar" in result:
                    p = os.path.join(output_dir, "INCAR")
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(result["incar"])
                    saved["INCAR"] = p
                if "kpoints" in result:
                    p = os.path.join(output_dir, "KPOINTS")
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(result["kpoints"])
                    saved["KPOINTS"] = p
            except Exception as e:
                return {"error": f"Save failed: {e}"}

        if isinstance(saved, dict) and "error" not in saved:
            saved.setdefault("incar", saved.get("INCAR"))
            saved.setdefault("kpoints", saved.get("KPOINTS"))

        if result.get("config_changes"):
            log_path = os.path.join(output_dir, "config_changes.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("# Changes applied by VASPProjectConfig\n")
                for change in result["config_changes"]:
                    f.write(f"  {change}\n")
            saved["config_log"] = log_path

        return saved

    def apply_improvements(self, original_incar: str, validation_result: dict,
                           poscar_path: str, original_request: str,
                           output_dir: str = ".",
                           skill: Optional[str] = "vasp",
                           config: Optional["VASPProjectConfig"] = None) -> dict:
        """
        Backward-compat wrapper around the generic apply_improvements.

        Accepts the historical signature (``original_incar`` as a raw
        string, ``poscar_path`` as the structure path, no ``software``
        kwarg) and returns the legacy flat shape.
        """
        original_inputs = {"INCAR": original_incar}
        result = super().apply_improvements(
            original_inputs=original_inputs,
            validation_result=validation_result,
            structure_file=poscar_path,
            request=original_request,
            output_dir=output_dir,
            software="vasp",
            skill=skill,
        )
        if result.get("status") == "success":
            result = self._to_legacy_shape(result)
            result = self._apply_config(result, config=config)
            if "INCAR" in result.get("improved_paths", {}):
                result["improved_incar_path"] = result["improved_paths"]["INCAR"]
        return result
