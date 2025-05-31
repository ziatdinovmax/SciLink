import os
import json
import logging
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .instruct import VASP_INPUT_GENERATION_INSTRUCTIONS


class VaspInputAgent:
    """Agent for generating VASP INCAR and KPOINTS files."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.logger = logging.getLogger(__name__)

    def generate_vasp_inputs(self, poscar_path: str, original_request: str) -> dict:
        """Generate VASP INCAR and KPOINTS files."""
        
        # Read POSCAR
        try:
            with open(poscar_path, 'r') as f:
                poscar_content = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read POSCAR: {e}"}

        # Build prompt
        prompt = VASP_INPUT_GENERATION_INSTRUCTIONS.format(
            poscar_content=poscar_content,
            original_request=original_request
        )

        # Get LLM response
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result = json.loads(response.text)
            result["status"] = "success"
            return result
        except Exception as e:
            return {"status": "error", "message": f"Generation failed: {e}"}

    def save_inputs(self, result: dict, output_dir: str = ".") -> dict:
        """Save INCAR and KPOINTS files."""
        if result.get("status") != "success":
            return {"error": "Generation was not successful"}
        
        os.makedirs(output_dir, exist_ok=True)
        saved = {}
        
        try:
            # Save INCAR
            with open(os.path.join(output_dir, "INCAR"), 'w') as f:
                f.write(result["incar"])
            saved["incar"] = os.path.join(output_dir, "INCAR")
            
            # Save KPOINTS  
            with open(os.path.join(output_dir, "KPOINTS"), 'w') as f:
                f.write(result["kpoints"])
            saved["kpoints"] = os.path.join(output_dir, "KPOINTS")
            
            return saved
        except Exception as e:
            return {"error": f"Save failed: {e}"}