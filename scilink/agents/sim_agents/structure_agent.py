import logging
from typing import Optional
import os
import json

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from .instruct import (
    INITIAL_PROMPT_TEMPLATE, 
    CORRECTION_PROMPT_TEMPLATE, 
    SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE,
    DOCS_ENHANCED_INITIAL_PROMPT_TEMPLATE,
    DOCS_ENHANCED_CORRECTION_PROMPT_TEMPLATE
)
from .utils import save_generated_script, MaterialsProjectHelper
from ...executors import ScriptExecutor, DEFAULT_TIMEOUT

from ._deprecation import normalize_params


MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS = 5

# Tool configurations for keyword matching and documentation
TOOL_CONFIGS = {
    "GrainBoundary": {
        "docs_path": "docs/aimsgb.txt",
        "keywords": ["grain boundary", "grain-boundary", "gb ", "sigma", "csl", 
                     "twist", "tilt", "bicrystal", "rotation axis", "aimsgb"],
    },
    "ASE": {
        "docs_path": None,
        "keywords": [],
    }
}

JSON_OUTPUT_INSTRUCTION = """

IMPORTANT: You must respond with a JSON object in this exact format:
{
    "script_content": "your complete Python script here as a single string"
}

The script_content must be a valid, complete Python script that can be executed directly.
Escape any special characters properly for JSON (newlines as \\n, quotes as \\", etc.).
"""


class StructureGenerator:
    def __init__(self, api_key: str = None, 
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 executor_timeout: int = DEFAULT_TIMEOUT,
                 generated_script_dir: str = "generated_scripts",
                 mp_api_key: str = None,
                 # Legacy parameters
                 local_model: str = None,
                 google_api_key: str = None):
        """
        Initialize StructureGenerator with Multi-Backend Support.
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.generated_script_dir = generated_script_dir

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="StructureGenerator"
        )
        
        if base_url:
            # Internal Proxy
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError("API key required for internal proxy.")

            self.logger.info(f"StructureGenerator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public / LiteLLM
            self.logger.info(f"StructureGenerator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )        
        
        # We rely on the prompt to enforce JSON, 
        # rather than the generation config.
        self.generation_config = None

        # Executor Setup
        self.ase_executor = ScriptExecutor(timeout=executor_timeout, mp_api_key=mp_api_key)
        
        # Load tool configurations
        self.tool_configs = self._load_tool_configs()
        self.mp_helper = MaterialsProjectHelper(api_key=mp_api_key)
        
        # Initialization message
        tools_with_docs = [name for name, cfg in self.tool_configs.items() if cfg.get("docs_content")]
        print(f"🔧 Structure Generator Ready ({model_name})")
        print(f"   📚 Available tools: ASE (default)" + (f", {', '.join(tools_with_docs)}" if tools_with_docs else ""))
        if self.mp_helper.enabled:
            print(f"   🗃️  Materials Project: Connected")
        else:
            print(f"   🗃️  Materials Project: Not configured")

    def _load_tool_configs(self) -> dict:
        """Load tool configurations and their documentation."""
        configs = {}
        for name, config in TOOL_CONFIGS.items():
            configs[name] = {
                "keywords": config.get("keywords", []),
                "docs_content": self._load_docs(config.get("docs_path"))
            }
        return configs

    def _load_docs(self, docs_path: Optional[str]) -> Optional[str]:
        """Load documentation from file if it exists."""
        if not docs_path:
            return None
            
        possible_paths = [
            docs_path,
            os.path.join(os.path.dirname(__file__), docs_path),
            os.path.join(os.path.dirname(__file__), "../..", docs_path),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    max_length = 60000
                    if len(content) > max_length:
                        content = content[:max_length] + "\n\n[... Documentation truncated ...]"
                    self.logger.info(f"Loaded docs from: {path}")
                    return content
                except Exception as e:
                    self.logger.error(f"Failed to read docs from {path}: {e}")
        return None

    def _select_tool(self, request_text: str) -> str:
        """Select the appropriate tool based on request content."""
        request_lower = request_text.lower()
        
        for name, config in self.tool_configs.items():
            keywords = config.get("keywords", [])
            if keywords and any(kw in request_lower for kw in keywords):
                self.logger.info(f"Selected {name} tool based on keywords")
                return name
        
        self.logger.info("Selected default ASE tool")
        return "ASE"

    def _get_tool_docs(self, tool_name: str) -> Optional[str]:
        """Get documentation for a tool."""
        return self.tool_configs.get(tool_name, {}).get("docs_content")

    def _build_initial_prompt(self, description: str, tool_name: str) -> str:
        """Build initial prompt with tool-specific documentation and MP integration."""
        docs_content = self._get_tool_docs(tool_name)
        
        if docs_content and self.mp_helper.enabled:
            docs_content += self.mp_helper.get_common_materials_info()
        
        if docs_content:
            base_prompt = DOCS_ENHANCED_INITIAL_PROMPT_TEMPLATE.format(
                description=description, 
                tool_name=tool_name, 
                documentation=docs_content
            )
        else:
            base_prompt = INITIAL_PROMPT_TEMPLATE.format(
                description=description, 
                tool_name=tool_name
            )
        
        return base_prompt + JSON_OUTPUT_INSTRUCTION

    def _build_correction_prompt(self, original_request: str, failed_script: str, 
                                 error_message: str, tool_name: str) -> str:
        """Build correction prompt with tool-specific documentation if available."""
        max_error_len = 2000
        if len(error_message) > max_error_len:
            error_message = error_message[:max_error_len] + "\n[... Error message truncated ...]"
        
        docs_content = self._get_tool_docs(tool_name)
        
        if docs_content:
            base_prompt = DOCS_ENHANCED_CORRECTION_PROMPT_TEMPLATE.format(
                original_request=original_request,
                failed_script=failed_script,
                error_message=error_message,
                tool_name=tool_name,
                documentation=docs_content
            )
        else:
            base_prompt = CORRECTION_PROMPT_TEMPLATE.format(
                original_request=original_request,
                failed_script=failed_script,
                error_message=error_message,
                tool_name=tool_name
            )
        
        return base_prompt + JSON_OUTPUT_INSTRUCTION

    def _build_validation_correction_prompt(self, original_request: str, 
                                            attempted_script_content: str,
                                            validator_feedback: dict,
                                            tool_name: str) -> str:
        """Build validation correction prompt."""
        validator_issues = validator_feedback.get("all_identified_issues", [])
        validator_hints = validator_feedback.get("script_modification_hints", [])
        
        issues_str = "\n".join([f"- {issue}" for issue in validator_issues]) if validator_issues else "No specific issues listed."
        hints_str = "\n".join([f"- {hint}" for hint in validator_hints]) if validator_hints else "No specific hints provided."

        base_prompt = SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE.format(
            original_request=original_request,
            attempted_script_content=attempted_script_content,
            validator_overall_assessment=validator_feedback.get("overall_assessment", "N/A"),
            validator_specific_issues=issues_str,
            validator_script_hints=hints_str,
            tool_name=tool_name
        )
        
        return base_prompt + JSON_OUTPUT_INSTRUCTION

    def _generate_json(self, prompt: str) -> dict:
        """Generate JSON response from LLM."""
        self.logger.info("Sending request to LLM...")
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        try:
            # Using the wrapper's generate_content
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            
            if not response or not response.text:
                raise ValueError("Empty response from LLM")
            
            raw_text = response.text.strip()
            
            # Handle markdown code blocks if present
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
            
            # Try to find JSON object if there's extra text
            if not raw_text.startswith("{"):
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                if start != -1 and end > start:
                    raw_text = raw_text[start:end]
            
            return json.loads(raw_text)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw response: {response.text[:500] if response and response.text else 'None'}")
            raise
        except Exception as e:
            self.logger.exception(f"Error during LLM content generation: {e}")
            raise

    def generate_script(self, original_user_request: str, attempt_number_overall: int, 
                        is_refinement_from_validation: bool = False,
                        previous_script_content: Optional[str] = None,
                        validator_feedback: Optional[dict] = None) -> dict:
        """Generate or refine a script using appropriate tool and documentation."""
        
        # Select tool based on request
        tool_name = self._select_tool(original_user_request)
        
        if is_refinement_from_validation:
            print(f"   🔄 Refining script using {tool_name} (cycle {attempt_number_overall})")
            if not previous_script_content or not validator_feedback:
                return {"status": "error", "message": "Internal error: Refinement requires previous script and validation feedback."}
            current_prompt = self._build_validation_correction_prompt(
                original_request=original_user_request,
                attempted_script_content=previous_script_content,
                validator_feedback=validator_feedback,
                tool_name=tool_name
            )
        else:
            print(f"   🤖 Generating script using {tool_name} (cycle {attempt_number_overall})")
            current_prompt = self._build_initial_prompt(original_user_request, tool_name)

        last_error_message = "No attempts made yet."
        current_script = previous_script_content if is_refinement_from_validation else None
        final_script_path = None
        
        # Internal correction loop for script execution errors
        for attempt in range(1, MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS + 1):
            try:
                if attempt > 1:
                    print(f"      🔧 Fixing script issues (attempt {attempt}/{MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS})")
                
                # Generate script via JSON response
                response_data = self._generate_json(current_prompt)
                script_content = response_data.get("script_content")
                
                if not script_content:
                    last_error_message = f"LLM response missing 'script_content' (attempt {attempt})"
                    self.logger.error(last_error_message)
                    if attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                        break
                    continue

                current_script = script_content

                # Save script
                script_desc = f"{original_user_request[:30]}_cycle{attempt_number_overall}_attempt{attempt}"
                final_script_path = save_generated_script(
                    current_script, script_desc, 1, output_dir=self.generated_script_dir
                )
                
                if not final_script_path:
                    last_error_message = f"Failed to save script (attempt {attempt})"
                    self.logger.error(last_error_message)
                    break

                # Execute script
                if attempt == 1:
                    print(f"      ⚙️  Executing script...")
                else:
                    print(f"      ⚙️  Re-executing corrected script...")
                    
                exec_result = self.ase_executor.execute_script(
                    current_script, working_dir=self.generated_script_dir
                )

                if exec_result["status"] == "success":
                    output_file = None
                    for line in exec_result.get("stdout", "").splitlines():
                        if line.startswith("STRUCTURE_SAVED:"):
                            output_file = line.split(":", 1)[1].strip()
                            break

                    if output_file and os.path.exists(os.path.join(self.generated_script_dir, output_file)):
                        print(f"      ✅ Script executed successfully")
                        full_output_path = os.path.abspath(os.path.join(self.generated_script_dir, output_file))
                        return {
                            "status": "success",
                            "message": f"Script generated and executed successfully on attempt {attempt}",
                            "output_file": full_output_path,
                            "final_script_path": final_script_path,
                            "final_script_content": current_script,
                            "tool_used": tool_name,
                            "execution_attempts": attempt
                        }
                    else:
                        last_error_message = f"Script ran but output file not found (attempt {attempt})"
                else:
                    last_error_message = exec_result.get("message", f"Unknown execution error (attempt {attempt})")
                
                if attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                    print(f"      ❌ Script execution failed after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts")
                    break
                
                # Prepare correction prompt for next attempt
                error_summary = self._summarize_error(last_error_message)
                print(f"      ⚠️  Execution failed: {error_summary}")
                print(f"         Attempting to fix...")
                
                current_prompt = self._build_correction_prompt(
                    original_request=original_user_request,
                    failed_script=current_script,
                    error_message=last_error_message,
                    tool_name=tool_name
                )

            except Exception as e:
                self.logger.exception(f"Unexpected error (attempt {attempt}): {e}")
                last_error_message = f"Unexpected error: {e}"
                if attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                    break
                continue
        
        # All attempts failed
        print(f"      ❌ Failed to generate working script after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts")
        return {
            "status": "error",
            "message": f"Failed to generate executable script after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts",
            "last_error": last_error_message,
            "last_attempted_script_path": final_script_path,
            "last_attempted_script_content": current_script,
            "tool_attempted": tool_name
        }

    def _summarize_error(self, error_message: str) -> str:
        """Create a brief, user-friendly error summary."""
        error_lower = error_message.lower()
        
        error_patterns = {
            "modulenotfounderror": "Missing Python module/import",
            "nameerror": "Undefined variable or function",
            "syntaxerror": "Python syntax error",
            "indexerror": "Array/list index out of range",
            "keyerror": "Dictionary key not found",
            "typeerror": "Incorrect data type usage",
            "valueerror": "Invalid value or parameter",
            "filenotfounderror": "File or path not found",
            "timeout": "Script execution timeout",
            "structure_saved": "Missing output confirmation",
        }
        
        for pattern, message in error_patterns.items():
            if pattern in error_lower:
                return message
        
        first_line = error_message.split('\n')[0]
        return first_line[:80] + "..." if len(first_line) > 80 else first_line