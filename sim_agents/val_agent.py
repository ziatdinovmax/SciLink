import os
import logging
import json

from ase.io import read as ase_read

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from .instruct import VALIDATOR_PROMPT_TEMPLATE, INCAR_VALIDATION_INSTRUCTIONS
from lit_agents.literature_agent import IncarLiteratureAgent


class StructureValidatorAgent:
    def __init__(self, api_key: str, model_name: str):
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided for StructureValidatorAgent and GOOGLE_API_KEY not set.")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"StructureValidatorAgent initialized with model: {self.model_name}.")

    def _read_structure_file_content(self, structure_file_path: str) -> str:
        """
        Read the raw content of the structure file for LLM analysis.
        """
        try:
            with open(structure_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Limit content size for LLM context window
            max_chars = 8000  # Adjust based on your needs
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
                                     tool_documentation: str = None) -> dict:
        """
        Uses an LLM to perform full validation including analysis of the actual structure file content.
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
        
        # Format structure file content section
        structure_section = ""
        if structure_file_content:
            structure_section = f"""

    ## ACTUAL STRUCTURE FILE CONTENT:
    ```
    {structure_file_content}
    ```

    """

        prompt = VALIDATOR_PROMPT_TEMPLATE.format(
            tool_documentation=doc_section,
            original_request=original_request,
            generating_script_content=generating_script_content,
            structure_file_path=structure_file_path,
        ) + structure_section
        
        self.logger.info("Sending request to Validator LLM for full validation and script hints...")
        self.logger.debug(f"Validator LLM Prompt (first 500 chars):\n{prompt[:500]}...")

        try:
            response = self.model.generate_content(
                contents=[prompt],
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
                                 original_request: str, tool_documentation: str = None) -> dict:
        """
        Main validation method. Relies on LLM for all checks.
        Returns a dictionary with validation status, issues, and script modification hints.
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

        # 2. Get LLM-based validation with actual file content
        llm_feedback = self._get_llm_validation_and_hints(
            original_request=original_request,
            generating_script_content=generating_script_content,
            structure_file_path=structure_file_path,
            structure_file_content=structure_file_content, # Pass raw file content to LLM
            tool_documentation=tool_documentation
        )

        final_feedback["overall_assessment"] = llm_feedback.get("overall_assessment", "LLM assessment missing or failed.")
        # Issues identified by LLM are now the sole source of issues
        final_feedback["all_identified_issues"] = llm_feedback.get("identified_issues_detail", [])
        final_feedback["script_modification_hints"] = llm_feedback.get("script_modification_hints", [])

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




class IncarValidatorAgent:
    """Agent that validates and suggests improvements to VASP INCAR files using literature."""

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-pro-preview-05-06", 
                 futurehouse_api_key: str = None, max_wait_time: int = 500):
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = GenerationConfig(response_mime_type="application/json")
        self.logger = logging.getLogger(__name__)
        
        # Initialize literature agent
        self.literature_agent = IncarLiteratureAgent(
            api_key=futurehouse_api_key, 
            max_wait_time=max_wait_time
        )

    def validate_and_improve_incar(self, incar_content: str, system_description: str) -> dict:
        """Validate INCAR parameters and suggest improvements based on literature."""
        
        # Step 1: Get literature review
        self.logger.info("Getting literature review of INCAR parameters...")
        lit_result = self.literature_agent.validate_incar(incar_content, system_description)
        
        if lit_result["status"] != "success":
            return {
                "status": "error",
                "message": f"Literature review failed: {lit_result.get('message')}",
                "validation_status": "unknown"
            }
        
        # Step 2: Analyze literature review and suggest improvements
        self.logger.info("Analyzing literature review for potential improvements...")
        
        prompt = f"""{INCAR_VALIDATION_INSTRUCTIONS}

## ORIGINAL INCAR:
{incar_content}

## SYSTEM DESCRIPTION:
{system_description}

## LITERATURE REVIEW:
{lit_result['response']}

Analyze the literature review and suggest specific parameter adjustments if needed."""

        try:
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            result = json.loads(response.text)
            
            # Add literature review to result
            result.update({
                "status": "success",
                "literature_review": lit_result['response'],
                "literature_task_id": lit_result.get('task_id')
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing literature review: {e}")
            return {
                "status": "error", 
                "message": f"Analysis failed: {str(e)}",
                "literature_review": lit_result['response']
            }

    def save_validation_report(self, validation_result: dict, output_dir: str = ".") -> dict:
        """Save validation report and revised INCAR if needed."""
        if validation_result.get("status") != "success":
            return {"error": "Validation was not successful"}
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        try:
            # Save validation report
            report_path = os.path.join(output_dir, "incar_validation_report.json")
            with open(report_path, 'w') as f:
                json.dump(validation_result, f, indent=2, default=str)
            saved_files["validation_report"] = report_path
            
            # Save revised INCAR if adjustments were made
            if (validation_result.get("validation_status") == "needs_adjustment" and 
                validation_result.get("revised_incar")):
                
                revised_path = os.path.join(output_dir, "INCAR_revised")
                with open(revised_path, 'w') as f:
                    f.write(validation_result["revised_incar"])
                saved_files["revised_incar"] = revised_path
                
                # Also save adjustment summary
                summary_path = os.path.join(output_dir, "incar_adjustments.txt")
                with open(summary_path, 'w') as f:
                    f.write("INCAR Parameter Adjustments\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(f"Overall Assessment: {validation_result.get('overall_assessment', 'N/A')}\n\n")
                    
                    adjustments = validation_result.get("suggested_adjustments", [])
                    if adjustments:
                        f.write("Suggested Changes:\n")
                        for adj in adjustments:
                            f.write(f"\n• {adj.get('parameter')}:\n")
                            f.write(f"  Current: {adj.get('current_value')}\n")
                            f.write(f"  Suggested: {adj.get('suggested_value')}\n")
                            f.write(f"  Reason: {adj.get('reason')}\n")
                    else:
                        f.write("No specific adjustments suggested.\n")
                        
                saved_files["adjustment_summary"] = summary_path
            
            self.logger.info(f"Validation report saved: {saved_files}")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving validation files: {e}")
            return {"error": f"Save failed: {str(e)}"}