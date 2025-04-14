import os
import logging

# Import refactored components using relative imports
from .llm_client import LLMClient
from .executors import AseExecutor, DEFAULT_TIMEOUT # Import constants if needed
from .tools import define_ase_tool, ASE_EXECUTE_TOOL_NAME # Import tool func and name
from .instruct import INITIAL_PROMPT_TEMPLATE, CORRECTION_PROMPT_TEMPLATE
from .utils import save_generated_script

# Define constants or get from config if not imported
# DEFAULT_TIMEOUT = 120 # Already imported from executor
MAX_CORRECTION_ATTEMPTS = 5

class StructureGenerator:
    """
    Orchestrates the workflow for generating atomic structures using
    an LLM and an ASE script executor, with retries for correction.
    """
    TOOL_NAME = ASE_EXECUTE_TOOL_NAME # Use imported constant

    def __init__(self, api_key: str, model_name: str,
                 executor_timeout: int = DEFAULT_TIMEOUT,
                 generated_script_dir: str = "generated_scripts"): # Added param
        self.llm_client = LLMClient(api_key=api_key, model_name=model_name)
        self.ase_executor = AseExecutor(timeout=executor_timeout)
        self.ase_tool = define_ase_tool()
        self.generated_script_dir = generated_script_dir # Store path
        logging.info("StructureGenerator initialized.")
        logging.info(f"Script output directory: {self.generated_script_dir}")
        logging.info(f"Max correction attempts set to: {MAX_CORRECTION_ATTEMPTS}")


    def _build_initial_prompt(self, description: str) -> str:
        """Builds the initial prompt using the template."""
        return INITIAL_PROMPT_TEMPLATE.format(
            description=description,
            tool_name=self.TOOL_NAME
        )

    def _build_correction_prompt(self, original_request: str, failed_script: str, error_message: str) -> str:
        """Builds the correction prompt using the template."""
        max_error_len = 2000 # Example limit
        if len(error_message) > max_error_len:
            error_message = error_message[:max_error_len] + "\n[... Error message truncated ...]"
        return CORRECTION_PROMPT_TEMPLATE.format(
            original_request=original_request,
            failed_script=failed_script,
            error_message=error_message,
            tool_name=self.TOOL_NAME
        )

    def _parse_llm_response(self, response) -> tuple[str | None, dict | None]: # Kept hints
         try:
            if not response or not response.candidates:
                # Handle cases like safety blocks or no response
                feedback = getattr(response, 'prompt_feedback', None)
                block_reason = getattr(feedback, 'block_reason', None)
                safety_ratings = getattr(feedback, 'safety_ratings', None)
                logging.warning(f"LLM response empty/no candidates. Block Reason: {block_reason}. Safety: {safety_ratings}")
                return f"[Warning: LLM response was empty or blocked. Reason: {block_reason}]", None

            candidate = response.candidates[0]
            # Check finish reason for potential issues (e.g., MAX_TOKENS, SAFETY)
            finish_reason = getattr(candidate, 'finish_reason', 'N/A')
            if finish_reason not in ["STOP", "TOOL_CALL"]: # Adjust based on actual API enum/values
                 logging.warning(f"LLM response candidate finished unexpectedly: {finish_reason}")
                 # Potentially return error or specific text based on finish_reason

            if not candidate.content or not candidate.content.parts:
                logging.warning(f"LLM response candidate has no content parts. Finish reason: {finish_reason}")
                return "[Warning: LLM response content was empty.]", None

            for part in candidate.content.parts:
                 if hasattr(part, 'function_call') and getattr(part.function_call, 'name', None):
                     logging.debug(f"LLM response contains function call: {part.function_call.name}")
                     args_dict = {key: value for key, value in getattr(part.function_call, 'args', {}).items()}
                     return None, {"name": part.function_call.name, "args": args_dict}

            # Check for text content only if no function call was found
            # Aggregate text from all parts if needed
            text_content = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
            if text_content:
                 logging.debug(f"LLM response contains text content.")
                 return text_content, None

            # If no function call and no text
            logging.warning(f"LLM response contained neither function call nor text. Finish reason: {finish_reason}")
            return "[Warning: LLM response was valid but contained neither function call nor text.]", None

         except (AttributeError, IndexError, ValueError, TypeError) as e:
            logging.exception(f"Error parsing LLM response structure: {e}\nResponse: {response}")
            return f"[Error: Could not parse LLM response: {e}]", None



    def generate(self, request_data: dict) -> dict:
        """Runs the structure generation workflow with correction attempts."""
        description = request_data.get("description")
        if not description:
            logging.error("Request data missing 'description'.")
            return {"status": "error", "message": "Input must contain 'description'."}

        logging.info(f"Starting structure generation for: '{description}'")
        current_prompt = self._build_initial_prompt(description)
        tools_list = [self.ase_tool] # Renamed variable for clarity
        last_error_message = "No attempts made."
        last_script_content = None
        final_script_path = None

        for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):
            logging.info(f"--- Attempt {attempt}/{MAX_CORRECTION_ATTEMPTS} ---")
            final_script_path = None

            try:
                llm_response = self.llm_client.generate_with_tools(current_prompt, tools_list)
                text_content, function_call = self._parse_llm_response(llm_response)

                if function_call and function_call["name"] == self.TOOL_NAME:
                    script_content = function_call["args"].get("script_content")
                    if not script_content:
                        last_error_message = f"LLM call missing 'script_content' on attempt {attempt}."
                        logging.error(last_error_message); break

                    last_script_content = script_content

                    # Use imported utility function to save script
                    saved_script_path = save_generated_script(
                        script_content, description, attempt,
                        output_dir=self.generated_script_dir)
                    if not saved_script_path:
                        last_error_message = f"Failed to save script on attempt {attempt}."
                        logging.error(last_error_message); break

                    final_script_path = saved_script_path
                    logging.info(f"Executing script from attempt {attempt}: {final_script_path}")
                    exec_result = self.ase_executor.execute_script(script_content)

                    if exec_result["status"] == "success":
                        logging.info(f"Script execution successful on attempt {attempt}.")
                        # (Return success dictionary as before)
                        return {
                            "status": "success",
                            "message": f"Structure generated successfully on attempt {attempt}.",
                            "output_file": exec_result.get("output_file"),
                            "final_script_path": final_script_path,
                            "attempts_made": attempt,
                            "input_request": request_data
                        }
                    else: # EXECUTION FAILED
                        last_error_message = exec_result.get("message", f"Unknown execution error on attempt {attempt}")
                        logging.warning(f"Attempt {attempt} execution failed: {last_error_message}")
                        if attempt == MAX_CORRECTION_ATTEMPTS:
                            logging.error("Maximum attempts reached. Failing."); break
                        logging.info("Building correction prompt for next attempt.")
                        current_prompt = self._build_correction_prompt(
                            original_request=description,
                            failed_script=script_content,
                            error_message=last_error_message
                        )
                        # Loop continues

                elif text_content: # Handle text response from LLM
                    last_error_message = f"LLM responded with text instead of script call on attempt {attempt}."
                    logging.warning(f"{last_error_message} Text: {text_content[:200]}...")
                    if attempt == MAX_CORRECTION_ATTEMPTS: break
                    # Try building correction prompt based on previous failure if available
                    if last_script_content and "failed" in last_error_message.lower():
                        current_prompt = self._build_correction_prompt(description, last_script_content, last_error_message)
                    else: # Fallback to initial prompt
                        current_prompt = self._build_initial_prompt(description)

                else: # Handle unusable LLM response
                    last_error_message = text_content if text_content else f"Failed to get valid function call or text from LLM on attempt {attempt}."
                    logging.error(last_error_message); break

            except Exception as e: # Handle unexpected errors in the loop
                logging.exception(f"Unexpected error during workflow attempt {attempt}: {e}")
                last_error_message = f"Workflow error on attempt {attempt}: {e}"; break

        # --- Loop finished (failure) ---
        logging.error(f"Structure generation failed after {attempt} attempt(s).")
        # (Return error dictionary as before)
        return {
            "status": "error",
            "message": f"Failed to generate structure after {attempt} attempt(s). Last error: {last_error_message}",
            "last_error": last_error_message,
            "attempts_made": attempt,
            "last_attempted_script_path": final_script_path,
            "input_request": request_data
        }
