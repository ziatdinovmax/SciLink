import logging
from typing import Optional, Tuple
import os

from .llm_client import LLMClient
from .tools import get_available_tools
from .instruct import (
    INITIAL_PROMPT_TEMPLATE, 
    CORRECTION_PROMPT_TEMPLATE, 
    SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE,
    DOCS_ENHANCED_INITIAL_PROMPT_TEMPLATE,
    DOCS_ENHANCED_CORRECTION_PROMPT_TEMPLATE
)
from .utils import save_generated_script, MaterialsProjectHelper
from ...executors import ScriptExecutor, DEFAULT_TIMEOUT


MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS = 5 

class StructureGenerator:
    def __init__(self, api_key: str, model_name: str,
                executor_timeout: int = DEFAULT_TIMEOUT,
                generated_script_dir: str = "generated_scripts",
                mp_api_key: str = None):
        """Initialize StructureGenerator with improved logging."""
        self.llm_client = LLMClient(api_key=api_key, model_name=model_name)
        self.ase_executor = ScriptExecutor(timeout=executor_timeout, mp_api_key=mp_api_key)
        self.generated_script_dir = generated_script_dir
        self.logger = logging.getLogger(__name__)
        
        self.tools = get_available_tools()
        self.mp_helper = MaterialsProjectHelper(api_key=mp_api_key)
        
        # Improved initialization message
        print(f"🔧 Structure Generator Ready")
        print(f"   📚 Available tools: {len(self.tools)} ({', '.join(t.name for t in self.tools)})")
        if self.mp_helper.enabled:
            print(f"   🗃️  Materials Project: Connected")
        else:
            print(f"   🗃️  Materials Project: Not configured")

    def _select_tool(self, request_text: str):
        """Select the appropriate tool based on request content."""
        # Check specialized tools first
        for tool in self.tools:
            if tool.matches_request(request_text):
                self.logger.info(f"Selected {tool.name} tool based on keywords")
                return tool
        
        # Default to ASE tool (first one without keywords)
        ase_tool = next((t for t in self.tools if not t.keywords), self.tools[0])
        self.logger.info(f"Selected default {ase_tool.name} tool")
        return ase_tool

    def _build_initial_prompt(self, description: str, use_fallback: bool = False) -> str:
        """Build initial prompt with tool-specific documentation and MP integration."""
        selected_tool = self._select_tool(description)
        tool_name = next(iter(selected_tool.tool.function_declarations)).name
        
        enhanced_docs = selected_tool.docs_content
        if enhanced_docs and self.mp_helper.enabled:
            enhanced_docs += self.mp_helper.get_common_materials_info()
        
        if use_fallback or not enhanced_docs:
            return INITIAL_PROMPT_TEMPLATE.format(description=description, tool_name=tool_name)
        else:
            return DOCS_ENHANCED_INITIAL_PROMPT_TEMPLATE.format(
                description=description, tool_name=tool_name, documentation=enhanced_docs
            )

    def _build_script_execution_error_correction_prompt(self, original_request: str, failed_script: str, error_message: str) -> str:
        """Build correction prompt with tool-specific documentation if available."""
        selected_tool = self._select_tool(original_request)
        tool_name = next(iter(selected_tool.tool.function_declarations)).name
        
        max_error_len = 2000
        if len(error_message) > max_error_len:
            error_message = error_message[:max_error_len] + "\n[... Error message truncated ...]"
        
        if selected_tool.docs_content:
            return DOCS_ENHANCED_CORRECTION_PROMPT_TEMPLATE.format(
                original_request=original_request,
                failed_script=failed_script,
                error_message=error_message,
                tool_name=tool_name,
                documentation=selected_tool.docs_content
            )
        else:
            return CORRECTION_PROMPT_TEMPLATE.format(
                original_request=original_request,
                failed_script=failed_script,
                error_message=error_message,
                tool_name=tool_name
            )

    def _build_script_correction_from_validation_prompt(
            self, original_request: str, attempted_script_content: str,
            validator_assessment: str, validator_issues: list, validator_hints: list) -> str:
        """Build validation correction prompt with appropriate tool."""
        selected_tool = self._select_tool(original_request)
        tool_name = next(iter(selected_tool.tool.function_declarations)).name
        
        issues_str = "\n".join([f"- {issue}" for issue in validator_issues]) if validator_issues else "No specific issues listed."
        hints_str = "\n".join([f"- {hint}" for hint in validator_hints]) if validator_hints else "No specific hints provided."

        return SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE.format(
            original_request=original_request,
            attempted_script_content=attempted_script_content,
            validator_overall_assessment=validator_assessment,
            validator_specific_issues=issues_str,
            validator_script_hints=hints_str,
            tool_name=tool_name
        )

    def _parse_llm_response(self, response) -> Tuple[Optional[str], Optional[dict]]:
        """Parse LLM response to extract text content or function call."""
        try:
            if not response or not response.candidates:
                feedback = getattr(response, 'prompt_feedback', None)
                block_reason = getattr(feedback, 'block_reason', 'Unknown reason')
                self.logger.warning(f"LLM response empty/no candidates. Block Reason: {block_reason}")
                return f"[Warning: LLM response was empty or blocked. Reason: {block_reason}]", None

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', 'N/A')
            if finish_reason not in [1, "STOP", "TOOL_CALL"]:
                self.logger.warning(f"LLM response candidate finished unexpectedly: {finish_reason}")

            if not candidate.content or not candidate.content.parts:
                self.logger.warning(f"LLM response candidate has no content parts. Finish reason: {finish_reason}")
                return "[Warning: LLM response content was empty.]", None

            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and getattr(part.function_call, 'name', None):
                    args_dict = {key: value for key, value in getattr(part.function_call, 'args', {}).items()}
                    return None, {"name": part.function_call.name, "args": args_dict}
            
            text_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
            if text_content:
                return text_content, None
            
            self.logger.warning(f"LLM response contained neither function call nor text. Finish reason: {finish_reason}")
            return "[Warning: LLM response was valid but contained neither function call nor text.]", None

        except (AttributeError, IndexError, ValueError, TypeError) as e:
            self.logger.exception(f"Error parsing LLM response structure: {e}")
            return f"[Error: Could not parse LLM response: {e}]", None

    def generate_script(self, original_user_request: str, attempt_number_overall: int, 
                       is_refinement_from_validation: bool = False,
                       previous_script_content: Optional[str] = None,
                       validator_feedback: Optional[dict] = None) -> dict:
        """Generate or refine a script using appropriate tool and documentation with improved output."""
        
        # Select tool and get its info
        selected_tool = self._select_tool(original_user_request)
        tool_name = next(iter(selected_tool.tool.function_declarations)).name
        
        if is_refinement_from_validation:
            print(f"   🔄 Refining script using {selected_tool.name} (cycle {attempt_number_overall})")
            if not previous_script_content or not validator_feedback:
                return {"status": "error", "message": "Internal error: Refinement requires previous script and validation feedback."}
            current_prompt = self._build_script_correction_from_validation_prompt(
                original_request=original_user_request,
                attempted_script_content=previous_script_content,
                validator_assessment=validator_feedback.get("overall_assessment", "N/A"),
                validator_issues=validator_feedback.get("all_identified_issues", []),
                validator_hints=validator_feedback.get("script_modification_hints", [])
            )
        else:
            print(f"   🤖 Generating script using {selected_tool.name} (cycle {attempt_number_overall})")
            current_prompt = self._build_initial_prompt(original_user_request)

        tools_list = [selected_tool.tool]
        last_error_message_internal = "No internal attempts made yet."
        current_script_being_processed = previous_script_content if is_refinement_from_validation else None
        
        # Internal correction loop for script execution errors
        for internal_exec_attempt in range(1, MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS + 1):
            
            generated_script_this_llm_call = None
            final_script_path_this_attempt = None

            try:
                # Show progress for multiple attempts
                if internal_exec_attempt > 1:
                    print(f"      🔧 Fixing script issues (attempt {internal_exec_attempt}/{MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS})")
                
                llm_response = self.llm_client.generate_with_tools(current_prompt, tools_list)
                text_content, function_call = self._parse_llm_response(llm_response)

                if function_call and function_call["name"] == tool_name:
                    generated_script_this_llm_call = function_call["args"].get("script_content")
                    if not generated_script_this_llm_call:
                        last_error_message_internal = f"LLM tool call missing 'script_content' (attempt {internal_exec_attempt})"
                        self.logger.error(last_error_message_internal)
                        if internal_exec_attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS: 
                            break
                        continue

                    current_script_being_processed = generated_script_this_llm_call

                    # Save script
                    script_desc_for_filename = f"{original_user_request[:30]}_cycle{attempt_number_overall}_exec_attempt{internal_exec_attempt}"
                    final_script_path_this_attempt = save_generated_script(
                        current_script_being_processed,
                        script_desc_for_filename,
                        1,
                        output_dir=self.generated_script_dir
                    )
                    
                    if not final_script_path_this_attempt:
                        last_error_message_internal = f"Failed to save script (attempt {internal_exec_attempt})"
                        self.logger.error(last_error_message_internal)
                        break

                    # Execute script with progress indication
                    if internal_exec_attempt == 1:
                        print(f"      ⚙️  Executing script...")
                    else:
                        print(f"      ⚙️  Re-executing corrected script...")
                        
                    exec_result = self.ase_executor.execute_script(current_script_being_processed, working_dir=self.generated_script_dir)

                    if exec_result["status"] == "success":
                        output_file = None
                        for line in exec_result.get("stdout", "").splitlines():
                            if line.startswith("STRUCTURE_SAVED:"):
                                output_file = line.split(":", 1)[1].strip()
                                break

                        if output_file and os.path.exists(os.path.join(self.generated_script_dir, output_file)):
                            print(f"     ✅ Script executed successfully")
                            full_output_path = os.path.abspath(os.path.join(self.generated_script_dir, output_file))
                            return {
                                "status": "success",
                                "message": f"Script generated and executed successfully on attempt {internal_exec_attempt}",
                                "output_file": full_output_path,
                                "final_script_path": final_script_path_this_attempt,
                                "final_script_content": current_script_being_processed,
                                "tool_used": selected_tool.name,
                                "execution_attempts": internal_exec_attempt
                            }
                    else:
                        last_error_message_internal = exec_result.get("message", f"Unknown script execution error (attempt {internal_exec_attempt})")
                        
                        if internal_exec_attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                            print(f"      ❌ Script execution failed after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts")
                            self.logger.error("Max internal script execution correction attempts reached.")
                            break
                        
                        # Show what went wrong for user awareness
                        error_summary = self._summarize_error(last_error_message_internal)
                        print(f"      ⚠️  Execution failed: {error_summary}")
                        print(f"         Attempting to fix...")
                        
                        current_prompt = self._build_script_execution_error_correction_prompt(
                            original_request=original_user_request,
                            failed_script=current_script_being_processed,
                            error_message=last_error_message_internal
                        )
                        continue

                elif text_content:
                    last_error_message_internal = f"LLM responded with text instead of tool call (attempt {internal_exec_attempt})"
                    if internal_exec_attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS: 
                        break
                    continue

                else:
                    last_error_message_internal = text_content if text_content else f"LLM gave unusable response (attempt {internal_exec_attempt})"
                    self.logger.error(last_error_message_internal)
                    break

            except Exception as e: 
                self.logger.exception(f"Unexpected error during script generation/execution (attempt {internal_exec_attempt}): {e}")
                last_error_message_internal = f"Unexpected workflow error: {e}"
                break
        
        # Internal loop finished without success
        print(f"      ❌ Failed to generate working script after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts")
        return {
            "status": "error",
            "message": f"Failed to generate executable script after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts",
            "last_error": last_error_message_internal,
            "last_attempted_script_path": final_script_path_this_attempt if 'final_script_path_this_attempt' in locals() else None,
            "last_attempted_script_content": current_script_being_processed,
            "tool_attempted": selected_tool.name
        }

    def _summarize_error(self, error_message: str) -> str:
        """Create a brief, user-friendly error summary."""
        error_lower = error_message.lower()
        
        # Common error patterns
        if "modulenotfounderror" in error_lower or "import" in error_lower:
            return "Missing Python module/import"
        elif "nameerror" in error_lower:
            return "Undefined variable or function"
        elif "syntaxerror" in error_lower:
            return "Python syntax error"
        elif "indexerror" in error_lower:
            return "Array/list index out of range"
        elif "keyerror" in error_lower:
            return "Dictionary key not found"
        elif "typeerror" in error_lower:
            return "Incorrect data type usage"
        elif "valueerror" in error_lower:
            return "Invalid value or parameter"
        elif "filenotfounderror" in error_lower:
            return "File or path not found"
        elif "timeout" in error_lower:
            return "Script execution timeout"
        elif "structure_saved" in error_lower:
            return "Missing output confirmation"
        else:
            # Extract first line of error for brevity
            first_line = error_message.split('\n')[0]
            return first_line[:80] + "..." if len(first_line) > 80 else first_line