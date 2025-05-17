import os
import logging

from .llm_client import LLMClient
from .executors import AseExecutor, DEFAULT_TIMEOUT
from .tools import define_ase_tool, ASE_EXECUTE_TOOL_NAME
from .instruct import (
    INITIAL_PROMPT_TEMPLATE, 
    CORRECTION_PROMPT_TEMPLATE, 
    SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE
)
from .utils import save_generated_script

# Max attempts for *internal* script execution error correction
MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS = 2 

class StructureGenerator:
    TOOL_NAME = ASE_EXECUTE_TOOL_NAME

    def __init__(self, api_key: str, model_name: str,
                 executor_timeout: int = DEFAULT_TIMEOUT,
                 generated_script_dir: str = "generated_scripts"):  
        self.llm_client = LLMClient(api_key=api_key, model_name=model_name)
        self.ase_executor = AseExecutor(timeout=executor_timeout)
        self.ase_tool = define_ase_tool()
        self.generated_script_dir = generated_script_dir
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"StructureGenerator initialized with model: {model_name}.")
        self.logger.info(f"Script output directory: {self.generated_script_dir}")
        self.logger.info(f"Max internal script execution correction attempts: {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS}")

    def _build_initial_prompt(self, description: str) -> str:
        """Builds the initial prompt for generating a script from a description."""
        return INITIAL_PROMPT_TEMPLATE.format(
            description=description,
            tool_name=self.TOOL_NAME
        )

    def _build_script_execution_error_correction_prompt(self, original_request: str, failed_script: str, error_message: str) -> str:
        """Builds the prompt for correcting a script that failed during execution."""
        max_error_len = 2000  # To avoid overly long prompts
        if len(error_message) > max_error_len:
            error_message = error_message[:max_error_len] + "\n[... Error message truncated ...]"
        return CORRECTION_PROMPT_TEMPLATE.format(
            original_request=original_request,
            failed_script=failed_script,
            error_message=error_message,
            tool_name=self.TOOL_NAME
        )

    # New method to build prompt for correction based on validation feedback
    def _build_script_correction_from_validation_prompt(
            self, original_request: str, attempted_script_content: str,
            validator_assessment: str, validator_issues: list, validator_hints: list) -> str:
        """Builds the prompt for correcting a script based on feedback from the StructureValidatorAgent."""
        
        issues_str = "\n".join([f"- {issue}" for issue in validator_issues]) if validator_issues else "No specific issues listed by validator."
        hints_str = "\n".join([f"- {hint}" for hint in validator_hints]) if validator_hints else "No specific hints provided by validator."

        return SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE.format(
            original_request=original_request,
            attempted_script_content=attempted_script_content,
            validator_overall_assessment=validator_assessment,
            validator_specific_issues=issues_str,
            validator_script_hints=hints_str,
            tool_name=self.TOOL_NAME
        )

    def _parse_llm_response(self, response) -> tuple[str | None, dict | None]:
        """Parses the LLM response to extract text content or a function call."""
        try:
            if not response or not response.candidates:
                feedback = getattr(response, 'prompt_feedback', None)
                block_reason = getattr(feedback, 'block_reason', 'Unknown reason')
                self.logger.warning(f"LLM response empty/no candidates. Block Reason: {block_reason}.")
                return f"[Warning: LLM response was empty or blocked. Reason: {block_reason}]", None

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', 'N/A')
            if finish_reason not in [1, "STOP", "TOOL_CALL"]: # 1 for Vertex AI, "STOP" for some, "TOOL_CALL" for some
                self.logger.warning(f"LLM response candidate finished unexpectedly: {finish_reason}. Content: {candidate.content.parts if candidate.content else 'N/A'}")

            if not candidate.content or not candidate.content.parts:
                self.logger.warning(f"LLM response candidate has no content parts. Finish reason: {finish_reason}")
                return "[Warning: LLM response content was empty.]", None

            for part in candidate.content.parts:
                if hasattr(part, 'function_call') and getattr(part.function_call, 'name', None):
                    self.logger.debug(f"LLM response contains function call: {part.function_call.name}")
                    args_dict = {key: value for key, value in getattr(part.function_call, 'args', {}).items()}
                    return None, {"name": part.function_call.name, "args": args_dict}
            
            text_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
            if text_content:
                self.logger.debug("LLM response contains text content instead of a tool call.")
                return text_content, None
            
            self.logger.warning(f"LLM response contained neither function call nor text. Finish reason: {finish_reason}")
            return "[Warning: LLM response was valid but contained neither function call nor text.]", None

        except (AttributeError, IndexError, ValueError, TypeError) as e:
            self.logger.exception(f"Error parsing LLM response structure: {e}\nResponse: {response}")
            return f"[Error: Could not parse LLM response: {e}]", None


    def generate_script(self,
                        original_user_request: str,
                        attempt_number_overall: int, # For file naming and logging
                        is_refinement_from_validation: bool = False,
                        previous_script_content: str | None = None,
                        validator_feedback: dict | None = None
                       ) -> dict:
        """
        Generates or refines an ASE script using an LLM.
        Handles internal retries for script *execution* errors.
        
        Args:
            original_user_request (str): The initial request for the structure.
            attempt_number_overall (int): The current cycle number from the main workflow (for logging/naming).
            is_refinement_from_validation (bool): True if this call is to refine a script based on validator feedback.
            previous_script_content (str | None): The content of the script that needs refinement.
            validator_feedback (dict | None): The feedback from the StructureValidatorAgent.
        
        Returns:
            dict: Result of the generation attempt.
                  Includes 'status', 'message', and if successful:
                  'output_file', 'final_script_path', 'final_script_content'.
                  If error: 'last_error', 'last_attempted_script_path', 'last_attempted_script_content'.
        """
        
        if is_refinement_from_validation:
            self.logger.info(f"Starting SCRIPT REFINEMENT (Overall Cycle {attempt_number_overall}) based on validation feedback for: '{original_user_request[:100]}...'")
            if not previous_script_content or not validator_feedback:
                self.logger.error("Refinement from validation called without previous script or validator feedback.")
                return {"status": "error", "message": "Internal error: Refinement requires previous script and validation feedback."}
            current_prompt = self._build_script_correction_from_validation_prompt(
                original_request=original_user_request,
                attempted_script_content=previous_script_content,
                validator_assessment=validator_feedback.get("overall_assessment", "N/A"),
                validator_issues=validator_feedback.get("all_identified_issues", []),
                validator_hints=validator_feedback.get("script_modification_hints", [])
            )
        else:
            self.logger.info(f"Starting INITIAL SCRIPT GENERATION (Overall Cycle {attempt_number_overall}) for: '{original_user_request[:100]}...'")
            current_prompt = self._build_initial_prompt(original_user_request)

        tools_list = [self.ase_tool]
        last_error_message_internal = "No internal attempts made yet."
        # `current_script_being_processed` holds the script content for the current internal attempt
        current_script_being_processed = previous_script_content if is_refinement_from_validation else None
        
        # This loop is for internal correction of SCRIPT EXECUTION errors
        for internal_exec_attempt in range(1, MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS + 1):
            self.logger.info(f"--- Script Generation/LLM Call (Overall Cycle {attempt_number_overall}, Internal Exec Attempt {internal_exec_attempt}) ---")
            
            generated_script_this_llm_call = None # Script from the current LLM call
            final_script_path_this_attempt = None # Path if script is saved in this attempt

            try:
                llm_response = self.llm_client.generate_with_tools(current_prompt, tools_list)
                text_content, function_call = self._parse_llm_response(llm_response)

                if function_call and function_call["name"] == self.TOOL_NAME:
                    generated_script_this_llm_call = function_call["args"].get("script_content")
                    if not generated_script_this_llm_call:
                        last_error_message_internal = f"LLM tool call missing 'script_content' (Overall Cycle {attempt_number_overall}, Internal Exec Attempt {internal_exec_attempt})."
                        self.logger.error(last_error_message_internal)
                        if internal_exec_attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS: break # Max internal attempts reached
                        # Rebuild prompt: if it was a refinement, try again with same validation prompt. If initial, try initial again.
                        self.logger.warning("Retrying LLM call with the same prompt due to missing script content.")
                        continue # Retry LLM call

                    current_script_being_processed = generated_script_this_llm_call # Update with the latest script from LLM

                    # Save script for this attempt
                    # Use a descriptive name that includes the overall cycle and internal attempt
                    script_desc_for_filename = f"{original_user_request[:30]}_cycle{attempt_number_overall}_exec_attempt{internal_exec_attempt}"
                    
                    final_script_path_this_attempt = save_generated_script(
                        current_script_being_processed,
                        script_desc_for_filename, # Using the modified description
                        1, # The 'attempt' for save_generated_script is just for its internal timestamping uniqueness here
                        output_dir=self.generated_script_dir
                    )
                    if not final_script_path_this_attempt:
                        last_error_message_internal = f"Failed to save script (Overall Cycle {attempt_number_overall}, Internal Exec Attempt {internal_exec_attempt})."
                        self.logger.error(last_error_message_internal)
                        break # Critical error, cannot proceed with execution

                    self.logger.info(f"Executing script: {final_script_path_this_attempt}")
                    exec_result = self.ase_executor.execute_script(current_script_being_processed)

                    if exec_result["status"] == "success":
                        self.logger.info(f"Script executed successfully (Overall Cycle {attempt_number_overall}, Internal Exec Attempt {internal_exec_attempt}).")
                        return {
                            "status": "success",
                            "message": "Script generated and executed successfully.",
                            "output_file": exec_result.get("output_file"),
                            "final_script_path": final_script_path_this_attempt,
                            "final_script_content": current_script_being_processed,
                        }
                    else: # SCRIPT EXECUTION FAILED
                        last_error_message_internal = exec_result.get("message", f"Unknown script execution error (Internal Exec Attempt {internal_exec_attempt})")
                        self.logger.warning(f"Script execution failed: {last_error_message_internal}")
                        if internal_exec_attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                            self.logger.error("Max internal script execution correction attempts reached. Aborting this generation cycle.")
                            break # Break from internal loop, will return error for this generate_script call
                        
                        self.logger.info("Building prompt for SCRIPT EXECUTION ERROR correction.")
                        current_prompt = self._build_script_execution_error_correction_prompt(
                            original_request=original_user_request,
                            failed_script=current_script_being_processed, # The script that just failed execution
                            error_message=last_error_message_internal
                        )
                        continue # Next internal execution attempt with a corrected script from LLM

                elif text_content: 
                    last_error_message_internal = f"LLM responded with text instead of a tool call (Overall Cycle {attempt_number_overall}, Internal Exec Attempt {internal_exec_attempt}). Text: {text_content[:200]}..."
                    self.logger.warning(last_error_message_internal)
                    if internal_exec_attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS: break
                    # If LLM gives text, it means it didn't understand it should use the tool.
                    # We retry with the *same* prompt that led to this text response.
                    self.logger.warning("Retrying LLM call with the same prompt as it returned text instead of tool call.")
                    continue

                else: # Unusable LLM response (e.g., empty, blocked, no function call, no text)
                    last_error_message_internal = text_content if text_content else f"LLM gave unusable response (Overall Cycle {attempt_number_overall}, Internal Exec Attempt {internal_exec_attempt})."
                    self.logger.error(last_error_message_internal)
                    break # Break from internal loop

            except Exception as e: 
                self.logger.exception(f"Unexpected error during script generation/execution (Internal Exec Attempt {internal_exec_attempt}): {e}")
                last_error_message_internal = f"Unexpected workflow error: {e}"
                break # Break from internal loop
        
        # --- Internal loop finished (either success returned or max internal attempts reached for execution errors) ---
        # If we are here, it means the internal loop failed to produce an executable script.
        self.logger.error(f"Failed to generate an executable script after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} internal execution attempts for original request: '{original_user_request[:100]}...' (Overall Cycle {attempt_number_overall}).")
        return {
            "status": "error",
            "message": f"Failed to generate an executable script. Last internal error: {last_error_message_internal}",
            "last_error": last_error_message_internal,
            "last_attempted_script_path": final_script_path_this_attempt if 'final_script_path_this_attempt' in locals() else None,
            "last_attempted_script_content": current_script_being_processed,
        }
