# sim_agents/executor.py
import subprocess
import tempfile
import os
import logging

DEFAULT_TIMEOUT = 120

class StructureExecutor:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, mp_api_key: str = None):
        self.timeout = timeout
        self.mp_api_key = mp_api_key or os.getenv("MP_API_KEY")
        logging.info(f"AseExecutor initialized with timeout: {self.timeout}s")
        if self.mp_api_key:
            logging.info("MP API key configured for script execution")
        else:
            logging.warning("No MP API key - scripts using Materials Project may fail")

    def execute_script(self, script_content: str, working_dir: str = None) -> dict:
        logging.info("Attempting to execute generated ASE script...")
        # ** SECURITY RISK **: Executes arbitrary code. Ensure sandboxing!
        temp_script_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tf:
                tf.write(script_content)
                temp_script_file = tf.name
            logging.debug(f"Temporary script saved to: {temp_script_file}")

            env = os.environ.copy()
            if self.mp_api_key:
                env['MP_API_KEY'] = self.mp_api_key
                logging.debug("Added MP_API_KEY to script execution environment")

            original_cwd = os.getcwd()
            if working_dir:
                os.makedirs(working_dir, exist_ok=True)
                os.chdir(working_dir)
            
            result = subprocess.run(
                ['python', temp_script_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,  # Pass environment with API key
                check=False
            )

            logging.debug("--- Script Execution Finished ---")
            logging.debug(f"STDOUT:\n{result.stdout}")
            logging.debug(f"STDERR:\n{result.stderr}")

            if result.returncode == 0:
                output_file = None
                for line in result.stdout.splitlines():
                    if line.startswith("STRUCTURE_SAVED:"):
                        output_file = line.split(":", 1)[1].strip()
                        break
                if output_file and os.path.exists(output_file):
                     logging.info(f"Script executed successfully. Output file found: {output_file}")
                     full_output_path = os.path.abspath(output_file)
                     return {"status": "success", "output_file": full_output_path}
                elif output_file:
                     error_msg = f"Script reported saving to {output_file}, but file not found."
                     logging.error(error_msg)
                     # Include stdout/stderr for context even if file isn't found but script exited 0
                     error_msg += f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                     return {"status": "error", "message": error_msg}
                else:
                    error_msg = "Script executed successfully (exit code 0), but did not report an output filename using 'STRUCTURE_SAVED:<filename>'."
                    logging.error(error_msg)
                    error_msg += f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                    return {"status": "error", "message": error_msg}
            else:
                # FAILURE CASE - return stderr
                error_msg = f"Script execution failed with return code {result.returncode}.\nSTDERR:\n{result.stderr}"
                logging.error(error_msg)
                # Return the raw error message (stderr is crucial for correction)
                return {"status": "error", "message": error_msg} # Ensure message contains stderr

        except FileNotFoundError:
             error_msg = "'python' command not found. Is Python installed and in PATH with ASE?"
             logging.exception(error_msg)
             return {"status": "error", "message": error_msg}
        except subprocess.TimeoutExpired:
             error_msg = f"Script execution timed out after {self.timeout} seconds."
             logging.error(error_msg)
             # Include stdout/stderr available before timeout if possible
             stderr_so_far = result.stderr if 'result' in locals() and hasattr(result, 'stderr') else 'N/A'
             stdout_so_far = result.stdout if 'result' in locals() and hasattr(result, 'stdout') else 'N/A'
             error_msg += f"\nSTDOUT (before timeout):\n{stdout_so_far}\nSTDERR (before timeout):\n{stderr_so_far}"
             return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred during script execution: {e}"
            logging.exception(error_msg)
            return {"status": "error", "message": error_msg}
        finally:
            os.chdir(original_cwd)
            if temp_script_file and os.path.exists(temp_script_file):
                try:
                    os.remove(temp_script_file)
                    logging.debug(f"Temporary script file removed: {temp_script_file}")
                except OSError as e:
                    logging.warning(f"Could not remove temporary file {temp_script_file}: {e}")
