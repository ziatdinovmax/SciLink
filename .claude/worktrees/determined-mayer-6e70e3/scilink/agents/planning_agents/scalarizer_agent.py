import subprocess
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import PIL.Image as PIL_Image

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...executors import require_sandbox_approval
from .parser_utils import parse_json_from_response
from .instruct import SCALARIZER_PROMPT, SCALARIZER_REFLECTION_PROMPT

from ._deprecation import normalize_params

from .base_agent import BaseAgent


class ScalarizerAgent(BaseAgent):
    """
    Agent for converting raw experimental data into scalar descriptors
    suitable for Bayesian Optimization.

    Example:
        >>> agent = ScalarizerAgent()
        >>> context = {
        ...     "hypothesis": "Product peak expected at 5.5 min",
        ...     "expected_outcome": "High yield > 80%"
        ... }
        >>> result = agent.scalarize(
        ...     data_path="data/hplc_run_01.csv",
        ...     objective_query="Integrate peak at 5.5 min. Calculate Purity %.",
        ...     experiment_context=context
        ... )
        >>> print(result["metrics"])
        {'purity': 98.5, 'peak_area': 12504.2}

    Args:
        api_key: API key for the LLM provider.
        model_name: Model name. For public deployments, use LiteLLM format
            (e.g., "gemini/gemini-2.0-flash", "gpt-4o", "claude-sonnet-4-20250514").
        base_url: Base URL for internal proxy endpoint.
            When provided, uses OpenAI-compatible client.
            When None, uses LiteLLM for multi-provider support.
        output_dir: Output directory for artifacts.
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        output_dir: str = ".",
        # Deprecated
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        if not require_sandbox_approval(
            context="ScalarizerAgent (scalarization of experimental data)"
        ):
            raise RuntimeError(
                "ScalarizerAgent requires code execution but user declined. "
                "Run in Docker, VM, or Colab for safe execution."
            )
        super().__init__(output_dir)
        self.agent_type = "scalarizer"

        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="ScalarizerAgent"
        )
        
        if base_url:
            # INTERNAL PROXY
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
            
            logging.info(f"🏛️ ScalarizerAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # PUBLIC LITELLM
            logging.info(f"🌐 ScalarizerAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )

        self.generation_config = None

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Agent-specific state fields"""
        return {
            "current_data_path": None,
            "current_objective": None,
            "active_script": None
        }

    def _read_file_head(self, file_path: str, n_lines=25) -> str:
        """Reads raw file header to help LLM handle delimiters/metadata."""
        path = Path(file_path)
        if not path.exists(): return "Error: File not found."
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                head = [next(f) for _ in range(n_lines)]
            return "".join(head)
        except Exception as e:
            return f"Error reading file head: {str(e)}"
        
    def _read_metadata(self, metadata_path: str) -> str:
        """Safely reads a sidecar JSON file."""
        if not metadata_path: 
            return "None"
        
        path = Path(metadata_path)
        if not path.exists():
            return f"Error: Metadata file not found at {path}"
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error reading metadata: {str(e)}"

    def _execute_script(self, script_path: Path, args: List[str] = None) -> Dict[str, Any]:
        """Runs the generated python script in a subprocess."""
       
        # Construct command with arguments (if any)
        cmd = ["python", str(script_path)]
        if args:
            cmd.extend(args)
        try:
            process = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=45
            )
            # Parse STDOUT for JSON
            json_match = re.search(r'\{.*\}', process.stdout.strip(), re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                # Check if the script itself reported an error
                if data.get("metrics") is None and data.get("error"):
                    return {
                        "status": "failure",
                        "error": data.get("error", "Script error"),
                        "stderr": data.get("traceback", process.stderr),
                        "stdout": process.stdout
                    }
                return {
                    "status": "success",
                    "metrics": data.get("metrics", {}),
                    "plot_path": data.get("plot_path", ""),
                    "stdout": process.stdout
                }
            else:
                return {
                    "status": "failure",
                    "error": "No JSON output found in stdout",
                    "stdout": process.stdout,
                    "stderr": process.stderr
                }
        except subprocess.TimeoutExpired:
            return {"status": "failure", "error": "Script execution timed out"}
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    def _verify_analysis(self, 
                        objective: str, 
                        context_str: str,
                        script_content: str, 
                        metrics: Dict, 
                        plot_path: str,
                        schema_requirements: Optional[Dict] = None) -> Dict[str, Any]:
        """Multimodal Self-Reflection: Checks plot vs. objective."""
        try:
            image = PIL_Image.open(plot_path)
        except Exception as e:
            return {"status": "fail", "feedback": f"Could not load visual proof: {e}"}

        # Build schema verification section
        schema_check = ""
        if schema_requirements:
            schema_check = f"""
    **REQUIRED SCHEMA TO VERIFY:**
    - Input columns that MUST be present: {schema_requirements.get('input_columns', [])}
    - Target columns that MUST be present: {schema_requirements.get('target_columns', [])}
    Verify ALL these columns appear in the metrics.
    """

        prompt = f"""
    **AUDIT REQUEST:**
    **1. EXTRACTION OBJECTIVE:** "{objective}"
    {schema_check}
    **2. EXTRACTED METRICS:** {json.dumps(metrics, indent=2)}
    **3. CODE SNIPPET:** 
    ```python
    {script_content[:3000]}
    ```
    **4. VISUAL PROOF:** (See Attached Image)
    **CONTEXT (reference only):** {context_str[:500]}...

    Verify the extraction is technically correct.
    """
        
        try:
            response = self.model.generate_content(
                [SCALARIZER_REFLECTION_PROMPT, prompt, image],
                generation_config=self.generation_config
            )
            return parse_json_from_response(response)[0]
        except Exception as e:
            logging.warning(f"Reflection failed: {e}")
            return {"status": "pass", "reasoning": "Auto-reflection unavailable."}

    def scalarize(self,
                 data_path: str,
                 objective_query: str = "",
                 reuse_script_path: str = None,
                 experiment_context: Optional[Dict[str, Any]] = None,
                 metadata_path: Optional[str] = None,
                 enable_human_review: bool = True,
                 column_role_hints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point. Converts raw data -> Scalar Metrics.

        Example:
            >>> agent = ScalarizerAgent()
            >>> context = {
            ...     "hypothesis": "Product peak expected at 5.5 min",
            ...     "expected_outcome": "High yield > 80%"
            ... }
            >>> result = agent.scalarize(
            ...     data_path="data/hplc_run_01.csv",
            ...     objective_query="Integrate peak at 5.5 min. Calculate Purity %.",
            ...     experiment_context=context
            ... )
            >>> print(result["metrics"])
            {'purity': 98.5, 'peak_area': 12504.2}

        Args:
            data_path: Path to raw data (csv, xlsx, txt).
            objective_query: Natural language instruction (e.g. "Calculate yield").
            experiment_context: Dict of high-level plan info (Hypothesis, etc).
            metadata_path: Path to sidecar JSON describing the data file (Units, Columns).
            enable_human_review: Pause for human check of the plot/logic.

        Returns:
            Dict containing:
            - 'status': 'success' or 'failure'
            - 'metrics': Dict of extracted scalars
            - 'source_script': Path to the generated Python script
        """
        path_obj = Path(data_path)
        
        # Initialize state
        self._init_state(current_data_path=data_path, current_objective=objective_query)

        # Path 1: Re-use existing script
        if reuse_script_path and Path(reuse_script_path).exists():
            print(f"  🔄 Reusing scalarizer script: {Path(reuse_script_path).name}")
            exec_res = self._execute_script(Path(reuse_script_path), args=[str(data_path)])
            
            result = {
                "status": exec_res["status"],
                "metrics": exec_res.get("metrics", {}),
                "source_script": str(reuse_script_path),
                "column_roles": self.state.get("column_roles", {}),
                "error": exec_res.get("error")
            }
            
            # Log the reuse action
            self._log_action(
                action="reuse_script",
                input_ctx={"data_path": data_path, "script": reuse_script_path},
                result=result,
                rationale="Reusing previously validated analysis script for consistency"
            )
            
            return result
        
        # Path 2: Generate new script
        file_context = self._read_file_head(data_path)
        
        # Metadata Auto-Discovery
        if not metadata_path:
            potential_json = path_obj.with_suffix('.json')
            if potential_json.exists():
                metadata_path = str(potential_json)
                print(f"  - ℹ️  Auto-discovered metadata file: {potential_json.name}")
        
        metadata_str = self._read_metadata(metadata_path)
        exp_context_str = json.dumps(experiment_context) if experiment_context else "None"
        plot_output_dir = str(self.output_dir.resolve())
        
        schema_section = ""
        if experiment_context and "_schema_requirements" in experiment_context:
            schema = experiment_context["_schema_requirements"]
            schema_section = f"""
    **REQUIRED OUTPUT SCHEMA (MANDATORY):**
    - INPUT COLUMNS: {schema.get('input_columns', [])}
    - TARGET COLUMNS: {schema.get('target_columns', [])}
    - OPTIMIZATION TYPE: {schema.get('optimization_type', 'single-objective')}

    Your output metrics MUST include ALL of these columns for each data point.
    """
        
        base_prompt = f"""
        **INPUT DATA:** {data_path}
        **HEAD SNIPPET:** \n{file_context}\n

        **METADATA SIDECAR (Column Defs / Units):**
        {metadata_str}
        {schema_section}
        
        **EXPERIMENTAL CONTEXT (Hypothesis / Steps):**
        {exp_context_str}
        
        **GOAL:** "{objective_query}"
        
        **REQ:** Parse, Calculate, Plot (save to {plot_output_dir}/debug_{path_obj.stem}.png), Print JSON.

        **CRITICAL:** In your code, replace OUTPUT_DIR_PLACEHOLDER with exactly: {plot_output_dir}
        This is an absolute path - use it directly without modification.
        """

        # Inject column role hints if user specified inputs/targets
        if column_role_hints:
            base_prompt += f"""
        **USER-SPECIFIED COLUMN ROLES (use these exactly):**
        - Inputs: {column_role_hints.get('inputs', [])}
        - Targets: {column_role_hints.get('targets', [])}
        Your column_roles output MUST match these exactly.
        """

        current_prompt = base_prompt
        max_retries = 5
        human_feedback_collected = None

        for attempt in range(max_retries):
            print(f"  - 📉 Scalarizer (Attempt {attempt+1}): Generating script...")
            
            # Generate Script
            try:
                response = self.model.generate_content(
                    [SCALARIZER_PROMPT, current_prompt], 
                    generation_config=self.generation_config
                )
                result, error = parse_json_from_response(response)
            except Exception as e:
                return {"status": "failure", "error": f"LLM Generation Error: {e}"}

            if error or not result or "implementation_code" not in result:
                err_msg = error if error else "Missing 'implementation_code' key"
                print(f"    ⚠️ Generation Failed (Invalid JSON): {err_msg}")
                current_prompt = base_prompt + f"\n\n**PREVIOUS ERROR:** JSON parsing failed ({err_msg}). Return ONLY valid JSON."
                continue

            # Extract and store column roles classification
            column_roles = result.get("column_roles", {})
            if column_roles:
                self.state["column_roles"] = column_roles

            # Save Script — replace OUTPUT_DIR_PLACEHOLDER if the LLM left it in
            code = result["implementation_code"]
            if "OUTPUT_DIR_PLACEHOLDER" in code:
                code = code.replace("OUTPUT_DIR_PLACEHOLDER", plot_output_dir)
            sanitized_name = Path(data_path).stem.replace(" ", "_")
            script_path = self.output_dir / f"proc_{sanitized_name}.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            # Track active script in state
            self.state["active_script"] = str(script_path)
            
            # Execute Script
            exec_res = self._execute_script(script_path, args=[str(data_path)])
            
            if exec_res["status"] == "failure":
                err_msg = exec_res.get('stderr', 'Unknown Error').strip()
                display_err = (err_msg[:300] + '...') if len(err_msg) > 300 else err_msg
                print(f"    ❌ Runtime Error:\n    {display_err}")
                current_prompt = base_prompt + f"\n\n**RUNTIME ERROR:**\n{err_msg}\nFix the code."
                continue
                
            schema_for_verification = None
            if experiment_context and "_schema_requirements" in experiment_context:
                schema_for_verification = experiment_context["_schema_requirements"]

            # Auto-Reflection
            plot_path = exec_res.get("plot_path")
            if plot_path and Path(plot_path).exists():
                print(f"    🤔 Auto-Reflecting on visual proof...")
                verification = self._verify_analysis(
                    objective=objective_query,
                    context_str=exp_context_str,
                    script_content=result["implementation_code"],
                    metrics=exec_res["metrics"],
                    plot_path=plot_path,
                    schema_requirements=schema_for_verification
                )

                if verification.get("status") == "fail":
                    feedback = verification.get("feedback", "Unknown logic error")
                    print(f"    ❌ Self-Correction Triggered: {feedback}")
                    current_prompt = base_prompt + f"\n\n**AUTO-CRITIQUE:** {feedback}\nAdjust the code and visuals."
                    continue
            else:
                # Plot missing — script ran but didn't save the plot.
                # Trigger retry so the LLM can fix the plotting code.
                print(f"    ⚠️  No plot file found — triggering retry")
                current_prompt = base_prompt + (
                    "\n\n**AUTO-CRITIQUE:** The script ran but did not produce a plot file. "
                    "Ensure plt.savefig() writes to the correct path and is called before plt.close()."
                )
                continue
            
            print(f"    ✅ Auto-Reflection Passed.")

            # Human Review
            if enable_human_review:
                metrics = exec_res['metrics']
                rows = metrics if isinstance(metrics, list) else [metrics]
                columns = list(rows[0].keys()) if rows else []
                print("\n" + "="*60)
                print(f"👀 SCALARIZER REVIEW: {path_obj.name}")
                print(f"  Extracted {len(columns)} columns from {len(rows)} data point(s):")
                for col in columns:
                    vals = [r.get(col) for r in rows[:3]]
                    preview = ", ".join(str(v) for v in vals)
                    if len(rows) > 3:
                        preview += ", ..."
                    print(f"    • {col}: [{preview}]")
                print(f"  Plot: {exec_res['plot_path']}")
                print("-" * 60)
                user_fb = input("> Press [ENTER] to confirm or type feedback: ").strip()
                
                if user_fb:
                    human_feedback_collected = user_fb
                    current_prompt = base_prompt + f"\n\n**HUMAN FEEDBACK:**\n{user_fb}"
                    continue

            # Success - log and return
            final_result = {
                "status": "success",
                "metrics": exec_res["metrics"],
                "source_script": str(script_path),
                "column_roles": self.state.get("column_roles", {})
            }
            
            self._log_action(
                action="generate_and_execute_script",
                input_ctx={
                    "data_path": data_path,
                    "objective": objective_query,
                    "metadata_path": metadata_path,
                    "attempt": attempt + 1
                },
                result=final_result,
                rationale=result.get("thought_process"),
                feedback=human_feedback_collected
            )
            
            self.state["status"] = "success"
            return final_result

        # Max retries exceeded
        failure_result = {"status": "failure", "error": "Max retries exceeded"}
        
        self._log_action(
            action="generate_and_execute_script",
            input_ctx={
                "data_path": data_path,
                "objective": objective_query,
                "attempt": max_retries
            },
            result=failure_result,
            rationale="All retry attempts exhausted"
        )
        
        self.state["status"] = "failed"
        return failure_result
