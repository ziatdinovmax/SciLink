import os
import logging
from datetime import datetime


import os
import logging
from typing import Optional, Dict

try:
    from mp_api.client import MPRester
    MP_API_AVAILABLE = True
except ImportError:
    MP_API_AVAILABLE = False


class MaterialsProjectHelper:
    """Minimal MP helper for automatic material resolution."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MP_API_KEY")
        self.enabled = MP_API_AVAILABLE and bool(self.api_key)
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self.logger.info("Materials Project helper enabled")
        else:
            self.logger.warning("Materials Project helper disabled (no API key or mp-api not installed)")
    
    def get_common_materials_info(self) -> str:
        """Return info about common materials for LLM context."""
        if not self.enabled:
            return ""
        
        # Common materials that users often request
        common_materials = {
            'YBCO': 'mp-20674',
            'YBa2Cu3O7': 'mp-20674', 
            'silicon': 'mp-149',
            'Si': 'mp-149',
            'copper': 'mp-30',
            'Cu': 'mp-30',
            'graphite': 'mp-48',
            'iron': 'mp-13'
        }
        
        info = "\n## COMMON MATERIALS PROJECT IDs:\n"
        for material, mp_id in common_materials.items():
            info += f"- {material}: {mp_id}\n"
        
        return info
    

def save_generated_script(script_content: str, description: str, attempt: int, output_dir: str) -> str | None:
    """Saves the script content to a file and returns the path."""
    try:
        # Ensure directory exists (might be better done once on agent init)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize description for use in filename
        safe_desc = "".join(c if c.isalnum() else "_" for c in description[:30]).rstrip("_")
        filename = f"ase_script_{safe_desc}_attempt{attempt}_{timestamp}.py"
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