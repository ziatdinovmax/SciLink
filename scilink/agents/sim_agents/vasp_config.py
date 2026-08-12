# scilink/skills/vasp_config.py
"""
User-configurable VASP project settings.
System-agnostic — users specify what they need, nothing is assumed.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import yaml
from pathlib import Path


@dataclass
class VASPConfig:
    """
    Project-level VASP configuration.
    
    These are settings that should be consistent across all calculations
    in a project. Users define them; the system enforces them.
    
    Parameters
    ----------
    gga : str, optional
        Functional tag. "PE" for PBE, "91" for PW91, None for LDA.
        Controls which pseudopotential directory ASE uses.
    enforced_params : dict
        INCAR parameters that must always be set to these values.
        Applied after LLM generation, overriding any conflicts.
    default_params : dict
        INCAR parameters applied only if the LLM didn't set them.
        Will NOT override LLM choices.
    potcar_map : dict, optional
        Maps element symbols to POTCAR variants.
        e.g., {"Ni": "Ni_pv", "Ti": "Ti_pv"}
    """
    gga: Optional[str] = None
    enforced_params: Dict[str, str] = field(default_factory=dict)
    default_params: Dict[str, str] = field(default_factory=dict)
    potcar_map: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "VASPProjectConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            gga=data.get("gga"),
            enforced_params=data.get("enforced_params", {}),
            default_params=data.get("default_params", {}),
            potcar_map=data.get("potcar_map", {}),
        )
    
    def to_yaml(self, path: str):
        """Save configuration to a YAML file."""
        data = {
            "gga": self.gga,
            "enforced_params": self.enforced_params,
            "default_params": self.default_params,
            "potcar_map": self.potcar_map,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def apply_to_incar(self, incar_text: str) -> dict:
        """
        Apply this config to an INCAR string.
        
        Returns dict with corrected INCAR text and list of changes made.
        """
        params = _parse_incar(incar_text)
        changes = []
        
        # GGA tag
        if self.gga is not None:
            current = params.get("GGA")
            if current != self.gga:
                params["GGA"] = self.gga
                changes.append(
                    f"{'Set' if current is None else 'Changed'} GGA = {self.gga}"
                    f"{f' (was {current})' if current else ''}"
                )
        
        # Enforced params always override
        for key, val in self.enforced_params.items():
            key_upper = key.upper()
            current = params.get(key_upper)
            if current != val:
                params[key_upper] = val
                if current is None:
                    changes.append(f"Added {key_upper} = {val} (enforced)")
                else:
                    changes.append(f"Changed {key_upper} = {current} → {val} (enforced)")
        
        # Default params only fill gaps
        for key, val in self.default_params.items():
            key_upper = key.upper()
            if key_upper not in params:
                params[key_upper] = val
                changes.append(f"Added {key_upper} = {val} (default)")
        
        return {
            "incar": _rebuild_incar(params),
            "params": params,
            "changes": changes,
        }


def _parse_incar(incar_text: str) -> Dict[str, str]:
    """Parse INCAR text into a dictionary."""
    params = {}
    for line in incar_text.split("\n"):
        line = line.strip()
        if not line or line.startswith(("#", "!")):
            continue
        if "=" in line:
            key, val = line.split("=", 1)
            val = val.split("#")[0].split("!")[0].strip()
            params[key.strip().upper()] = val
    return params


def _rebuild_incar(params: Dict[str, str]) -> str:
    """Rebuild INCAR text from parameter dictionary."""
    lines = []
    for key, val in params.items():
        lines.append(f"  {key} = {val}")
    return "\n".join(lines)
