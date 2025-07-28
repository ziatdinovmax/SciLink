import os
import sys
import subprocess
import re
import uuid
import tempfile
import logging

from ...auth import get_api_key

DEFAULT_TIMEOUT = 120

def is_in_colab():
    """Check for Google Colab environment."""
    if 'COLAB_GPU' in os.environ or 'GCE_METADATA_TIMEOUT' in os.environ:
        return True
    if 'google.colab' in sys.modules:
        return True
    return False

def check_security_sandbox_indicators(verbose=False):
    """Check for OS-level sandboxing indicators."""
    score = 0
    positive_indicators = []

    # Tier 1: High-Confidence Environments (Score: 10)
    if is_in_colab():
        score += 10
        positive_indicators.append("google_colab")
        if verbose: 
            logging.info("High-Confidence Indicator: Google Colab environment detected.")
        return score, positive_indicators

    # Tier 2: Strong Indicators (Score: 5)
    # Docker/Container check
    if os.path.exists('/.dockerenv') or ('docker' in (open('/proc/1/cgroup').read() if os.path.exists('/proc/1/cgroup') else '')):
        score += 5
        positive_indicators.append("docker_container")
        if verbose: 
            logging.info("Strong Indicator: Docker or container environment detected.")

    # Virtual Machine check
    try:
        if sys.platform.startswith("linux"):
            result = subprocess.run(['systemd-detect-virt'], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip() != 'none':
                score += 5
                positive_indicators.append(f"virtual_machine:{result.stdout.strip()}")
                if verbose: 
                    logging.info(f"Strong Indicator: Virtual Machine detected ('{result.stdout.strip()}').")
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Tier 3: Corroborating Evidence (Score: 2)
    try:
        mac = ':'.join(re.findall('..', f'{uuid.getnode():012x}'))
        vm_mac_prefixes = ["08:00:27", "00:05:69", "00:0c:29", "00:1c:14", "00:50:56"]
        if any(mac.lower().startswith(prefix) for prefix in vm_mac_prefixes):
            score += 2
            positive_indicators.append("vm_mac_address")
            if verbose: 
                logging.info("Corroborating Indicator: VM-associated MAC address found.")
    except Exception:
        pass

    return score, list(set(positive_indicators))

def enforce_security_sandbox(required_score=4, allow_override=False):
    """Enforce security sandbox requirement before code execution."""
    # Override check
    if allow_override and os.environ.get("UNSAFE_EXECUTION_OK", "false").lower() == "true":
        logging.warning("⚠️  WARNING: Safety check explicitly bypassed by user override.")
        logging.warning("          Executing on the host machine at your own risk.")
        return

    logging.info("Running Security Sandbox Check...")
    score, indicators = check_security_sandbox_indicators(verbose=True)

    if score >= required_score:
        friendly_name = indicators[0] if indicators else "Unknown Sandbox"
        logging.info(f"✅ Security Check Passed (Score: {score}, Indicator: {friendly_name})")
        logging.info("   OS-level isolated environment detected. Proceeding with code execution.")
        return
    else:
        # Critical security failure
        error_msg = f"""
❌ DANGER: SECURITY CHECK FAILED (Score: {score}, Required: {required_score}) ❌
{'='*70}
This system executes AI-generated code and requires a secure, isolated
environment to protect your computer from unintended or harmful actions.

Running directly on your main operating system is **EXTREMELY UNSAFE**.
{'='*70}

▶️ HOW TO RUN SCILINK SAFELY:
You MUST restart SciLink inside a container or virtual machine.

1. Docker (Strongly Recommended):
   Use the provided Dockerfile or run in a Docker container.

2. Virtual Machine:
   Install and run SciLink within a VM (VMware, VirtualBox, Cloud VM, etc.).

3. Google Colab:
   Use Colab's isolated notebook environment.

4. Advanced Override (NOT RECOMMENDED):
   Set environment variable: UNSAFE_EXECUTION_OK=true
{'='*70}
"""
        logging.error(error_msg)
        raise RuntimeError("Security sandbox requirement not met. Halting execution for safety.")


class StructureExecutor:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, mp_api_key: str = None, 
                 enforce_sandbox: bool = True, allow_unsafe_override: bool = False):
        self.timeout = timeout
        self.mp_api_key = mp_api_key or get_api_key('materials_project') or os.getenv("MP_API_KEY")
        self.enforce_sandbox = enforce_sandbox
        self.allow_unsafe_override = allow_unsafe_override
        
        # Perform security check on initialization
        if self.enforce_sandbox:
            enforce_security_sandbox(allow_override=self.allow_unsafe_override)
        
        logging.info(f"StructureExecutor initialized with timeout: {self.timeout}s")
        if self.mp_api_key:
            logging.info("MP API key configured for script execution")
        else:
            logging.warning("No MP API key - scripts using Materials Project may fail")

    def execute_script(self, script_content: str, working_dir: str = None) -> dict:
        """Execute AI-generated script with security enforcement."""
        
        # Additional security check before each execution (optional but recommended)
        if self.enforce_sandbox:
            try:
                enforce_security_sandbox(allow_override=self.allow_unsafe_override)
            except RuntimeError as e:
                return {"status": "error", "message": f"Security check failed: {str(e)}"}
        
        logging.info("Attempting to execute generated ASE script...")
        logging.warning("⚠️  EXECUTING AI-GENERATED CODE - Security sandbox verified")
        
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
                env=env,
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
                     error_msg += f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                     return {"status": "error", "message": error_msg}
                else:
                    error_msg = "Script executed successfully (exit code 0), but did not report an output filename using 'STRUCTURE_SAVED:<filename>'."
                    logging.error(error_msg)
                    error_msg += f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                    return {"status": "error", "message": error_msg}
            else:
                error_msg = f"Script execution failed with return code {result.returncode}.\nSTDERR:\n{result.stderr}"
                logging.error(error_msg)
                return {"status": "error", "message": error_msg}

        except FileNotFoundError:
             error_msg = "'python' command not found. Is Python installed and in PATH with ASE?"
             logging.exception(error_msg)
             return {"status": "error", "message": error_msg}
        except subprocess.TimeoutExpired:
             error_msg = f"Script execution timed out after {self.timeout} seconds."
             logging.error(error_msg)
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
