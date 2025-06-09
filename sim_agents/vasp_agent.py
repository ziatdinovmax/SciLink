import os
import json
import logging
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .instruct import VASP_INPUT_GENERATION_INSTRUCTIONS


class VaspInputAgent:
    """Agent for generating VASP INCAR and KPOINTS files."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro-preview-05-06"):
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
             
    def apply_improvements(self, original_incar: str, validation_result: dict, 
                          poscar_path: str, original_request: str, output_dir: str = ".") -> dict:
        """Regenerate INCAR using LLM with improvement instructions."""
        
        if validation_result.get("validation_status") != "needs_adjustment":
            return {
                "status": "no_changes", 
                "message": "No improvements needed - INCAR is already good"
            }
        
        adjustments = validation_result.get("suggested_adjustments", [])
        if not adjustments:
            return {"status": "error", "message": "No adjustments available"}
        
        # Read POSCAR content
        try:
            with open(poscar_path, 'r') as f:
                poscar_content = f.read()
        except Exception as e:
            return {"status": "error", "message": f"Failed to read POSCAR: {e}"}
        
        # Build improvement instructions
        improvement_instructions = "IMPROVEMENT INSTRUCTIONS:\n"
        improvement_instructions += "Please modify the provided INCAR file based on these literature-validated suggestions:\n\n"
        
        for adj in adjustments:
            improvement_instructions += f"• {adj.get('parameter')}: {adj.get('current_value')} → {adj.get('suggested_value')}\n"
            improvement_instructions += f"  Reason: {adj.get('reason')}\n\n"
        
        improvement_instructions += f"Literature assessment: {validation_result.get('overall_assessment', '')}\n\n"
        improvement_instructions += "Generate an improved INCAR file incorporating these changes."
        
        # Build the prompt with original INCAR and improvement instructions
        prompt = f"""{VASP_INPUT_GENERATION_INSTRUCTIONS}

## ORIGINAL INCAR TO IMPROVE:
{original_incar}

## {improvement_instructions}

## POSCAR STRUCTURE:
{poscar_content}

## ORIGINAL SYSTEM DESCRIPTION:
{original_request}

Please generate an improved INCAR file based on the improvement instructions above."""

        # Get improved INCAR from LLM
        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result = json.loads(response.text)
            
            if result.get("incar"):
                # Save improved INCAR
                os.makedirs(output_dir, exist_ok=True)
                improved_path = os.path.join(output_dir, "INCAR_improved")
                
                with open(improved_path, 'w') as f:
                    f.write(result["incar"])
                
                result.update({
                    "status": "success",
                    "improvements_applied": True,
                    "adjustments_count": len(adjustments),
                    "improved_incar_path": improved_path
                })
                
                self.logger.info(f"Generated improved INCAR with {len(adjustments)} literature-based improvements")
                return result
            else:
                return {"status": "error", "message": "No INCAR generated in LLM response"}
            
        except Exception as e:
            self.logger.error(f"Failed to generate improved INCAR: {e}")
            return {"status": "error", "message": f"Failed to generate improved INCAR: {e}"}