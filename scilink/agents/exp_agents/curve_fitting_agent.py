import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json
import os
import re

from .base_agent import BaseAnalysisAgent
from .human_feedback import SimpleFeedbackMixin
from ...executors import ScriptExecutor
from ..lit_agents.literature_agent import FittingModelLiteratureAgent
from .instruct import (
    LITERATURE_QUERY_GENERATION_INSTRUCTIONS,
    FITTING_SCRIPT_GENERATION_INSTRUCTIONS,
    FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS,
    FITTING_SCRIPT_CORRECTION_INSTRUCTIONS
)

class CurveFittingAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing 1D curves via automated, literature-informed fitting.
    """

    MAX_SCRIPT_ATTEMPTS = 3

    def __init__(self, google_api_key: str = None, futurehouse_api_key: str = None, 
                 model_name: str = "gemini-2.5-pro-preview-06-05", local_model: str = None, 
                 enable_human_feedback: bool = True, executor_timeout: int = 60, 
                 output_dir: str = "curve_analysis_output", max_wait_time: int = 600, **kwargs):
        super().__init__(google_api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        self.executor = ScriptExecutor(timeout=executor_timeout, enforce_sandbox=False)
        self.literature_agent = FittingModelLiteratureAgent(api_key=futurehouse_api_key, max_wait_time=max_wait_time)
        self.output_dir = output_dir
        if kwargs:
            self.logger.warning(f"Unused arguments passed to CurveFittingAgent: {kwargs}")

    def _load_curve_data(self, data_path: str) -> np.ndarray:
        if data_path.endswith(('.csv', '.txt')):
            data = np.loadtxt(data_path, delimiter=',')
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Data must be a 2-column array (X, Y).")
        return data

    def _plot_curve(self, curve_data: np.ndarray, system_info: dict, title_suffix="") -> bytes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(curve_data[:, 0], curve_data[:, 1], 'b.', markersize=4)
        ax.set_title(system_info.get("title", "1D Data") + title_suffix)
        ax.set_xlabel(system_info.get("xlabel", "X-axis"))
        ax.set_ylabel(system_info.get("ylabel", "Y-axis"))
        ax.grid(True, linestyle='--')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150)
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)
        return image_bytes

    def _generate_lit_search_query(self, plot_bytes: bytes, system_info: dict) -> str:
        """Uses an LLM to formulate a query for the literature agent."""
        self.logger.info("Generating literature search query for fitting models...")
        prompt = [
            LITERATURE_QUERY_GENERATION_INSTRUCTIONS,
            "## Data Plot", {"mime_type": "image/jpeg", "data": plot_bytes},
            self._build_system_info_prompt_section(system_info)
        ]
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result_json, error = self._parse_llm_response(response)
        if error or "search_query" not in result_json:
            raise ValueError("Failed to generate a valid literature search query.")
        return result_json["search_query"]

    def _generate_fitting_script(self, curve_data: np.ndarray, literature_context: str, data_path: str) -> str:
        """Uses an LLM to generate a Python fitting script."""
        self.logger.info("Generating Python script for data fitting...")
        data_preview = np.array2string(curve_data[:10], precision=4, separator=', ')
        prompt = (
            f"{FITTING_SCRIPT_GENERATION_INSTRUCTIONS}\n"
            f"## Literature Context\n{literature_context}\n"
            f"## Curve Data Preview\n{data_preview}\n"
            f"## Data File Path\nThe script should load data from this absolute path: '{os.path.abspath(data_path)}'"
        )
        response = self.model.generate_content(prompt)
        script_content = response.text
        match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
        if match:
            script_content = match.group(1).strip()
        else:
            if script_content.strip().startswith("import"):
                 script_content = script_content.strip()
            else:
                self.logger.error(f"LLM response did not contain a valid python code block. Response: {script_content[:500]}")
                raise ValueError("LLM failed to generate a valid Python script in a markdown block.")
        if not script_content:
            raise ValueError("LLM generated an empty fitting script.")
        return script_content

    def _save_literature_step_results(self, query: str, report: str) -> dict:
        """Saves the literature search query and the resulting report to files."""
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature_step")
            os.makedirs(lit_dir, exist_ok=True)

            # Save the query
            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, 'w') as f:
                f.write(query)
            saved_files["query_file"] = query_path
            self.logger.info(f"Saved literature query to: {query_path}")

            # Save the report
            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, 'w') as f:
                f.write(report)
            saved_files["report_file"] = report_path
            self.logger.info(f"Saved literature report to: {report_path}")

        except Exception as e:
            self.logger.error(f"Failed to save literature step results: {e}")
        
        return saved_files
    
    def _generate_and_execute_fitting_script_with_retry(
        self, curve_data: np.ndarray, literature_context: str, data_path: str
    ) -> dict:
        """
        Generates and executes the fitting script, with a retry loop for self-correction.
        """
        last_error = "No script generated yet."
        fitting_script = None

        for attempt in range(1, self.MAX_SCRIPT_ATTEMPTS + 1):
            try:
                if attempt == 1:
                    # First attempt: generate the initial script
                    print(f"⚙️  Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Generating initial fitting script...")
                    fitting_script = self._generate_fitting_script(curve_data, literature_context, data_path)
                else:
                    # Subsequent attempts: generate a corrected script
                    print(f"⚠️  Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Script failed. Requesting correction from LLM...")
                    correction_prompt = FITTING_SCRIPT_CORRECTION_INSTRUCTIONS.format(
                        literature_context=literature_context,
                        failed_script=fitting_script,
                        error_message=last_error
                    )
                    response = self.model.generate_content(correction_prompt)
                    script_content = response.text
                    match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
                    if match:
                        fitting_script = match.group(1).strip()
                    else: # Fallback if markdown block is missing
                        fitting_script = script_content.strip()

                # Execute the current version of the script
                print(f"   Executing script...")
                exec_result = self.executor.execute_script(fitting_script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    print("   ✅ Script executed successfully.")
                    return {
                        "status": "success",
                        "exec_result": exec_result,
                        "final_script": fitting_script,
                        "attempts": attempt
                    }
                else:
                    last_error = exec_result.get("message", "Unknown execution error.")
                    self.logger.warning(f"Script execution attempt {attempt} failed. Error: {last_error}")

            except Exception as e:
                last_error = f"An error occurred during script generation: {str(e)}"
                self.logger.error(last_error, exc_info=True)

        # If loop finishes without success
        print(f"❌ Script generation failed after {self.MAX_SCRIPT_ATTEMPTS} attempts.")
        return {
            "status": "error",
            "message": f"Failed to generate a working script after {self.MAX_SCRIPT_ATTEMPTS} attempts. Last error: {last_error}",
            "last_script": fitting_script
        }

    def analyze_for_claims(self, data_path: str, system_info: dict = None, **kwargs) -> dict:
        self.logger.info(f"Starting advanced curve analysis with fitting for: {data_path}")
        try:
            # Step 0: Load Data and Visualize
            curve_data = self._load_curve_data(data_path)
            system_info = self._handle_system_info(system_info)
            original_plot_bytes = self._plot_curve(curve_data, system_info, " (Original Data)")

            # Step 1: Search Literature for Fitting Models
            print(f"\n🤖 -------------------- ANALYSIS AGENT STEP: LITERATURE SEARCH FOR FITTING MODELS -------------------- 🤖")
            lit_query = self._generate_lit_search_query(original_plot_bytes, system_info)
            
            # Initialize variables
            literature_context = None
            saved_lit_files = {}

            # Attempt literature search
            lit_result = self.literature_agent.query_for_models(lit_query)
            
            if lit_result["status"] == "success":
                self.logger.info("Literature search successful.")
                print("✅ Literature search successful.")
                literature_context = lit_result["formatted_answer"]
                saved_lit_files = self._save_literature_step_results(lit_query, literature_context)
            else:
                # This is the new fallback logic
                warning_message = f"Literature search failed ({lit_result['message']}). Falling back to the LLM's internal knowledge."
                self.logger.warning(warning_message)
                print(f"⚠️  {warning_message}")
                
                literature_context = (
                    "The external literature search failed. Fall back to your internal knowledge. "
                    "Analyze the plot's shape and the system metadata to propose and implement a suitable physical fitting model."
                )
            
            # Save literature query and report
            saved_lit_files = self._save_literature_step_results(lit_query, literature_context)

            # Step 2 & 3: Generate and Execute Fitting Script with Retry Logic
            print(f"\n🤖 -------------------- ANALYSIS AGENT STEP: SCRIPT GENERATION & EXECUTION -------------------- 🤖")
            script_execution_bundle = self._generate_and_execute_fitting_script_with_retry(
                curve_data, literature_context, data_path
            )

            if script_execution_bundle["status"] != "success":
                # Propagate the failure from the retry loop
                raise RuntimeError(script_execution_bundle["message"])
            
            exec_result = script_execution_bundle["exec_result"]

            # Step 4: Parse Results
            fit_params = {}
            # The key for stdout is 'stdout', not 'output'
            for line in exec_result.get("stdout", "").splitlines():
                if line.startswith("FIT_RESULTS_JSON:"):
                    fit_params = json.loads(line.replace("FIT_RESULTS_JSON:", ""))
                    break
            if not fit_params:
                raise ValueError("Could not parse fitting parameters from script output.")
            
            fit_plot_path = os.path.join(self.output_dir, "fit_visualization.png")
            with open(fit_plot_path, "rb") as f:
                fit_plot_bytes = f.read()

            # Step 5: Final Interpretation and Claim Generation
            print(f"\n🤖 -------------------- ANALYSIS AGENT STEP: INTERPRETING FIT RESULTS & GENERATING CLAIMS -------------------- 🤖")
            final_prompt = [
                FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS,
                "\n## Original Data Plot", {"mime_type": "image/jpeg", "data": original_plot_bytes},
                "\n## Fit Visualization", {"mime_type": "image/png", "data": fit_plot_bytes},
                "\n## Fitted Parameters\n" + json.dumps(fit_params, indent=2),
                "\n## Literature Context\n" + literature_context,
                self._build_system_info_prompt_section(system_info)
            ]
            response = self.model.generate_content(final_prompt, generation_config=self.generation_config)
            
            result_json, error_dict = self._parse_llm_response(response)
            if error_dict:
                raise RuntimeError(f"Failed to interpret fitting results: {error_dict}")

            # Store the generated plots for the feedback loop
            analysis_images = [
                {'label': 'Original Data Plot', 'data': original_plot_bytes},
                {'label': 'Fit Visualization', 'data': fit_plot_bytes}
            ]
            self._store_analysis_images(analysis_images)

            # Prepare the initial result for the feedback function
            initial_result = {
                "detailed_analysis": result_json.get("detailed_analysis"),
                "scientific_claims": self._validate_scientific_claims(result_json.get("scientific_claims", []))
            }

            # Call the feedback function, which will display the results and ask for input
            final_result = self._apply_feedback_if_enabled(
                initial_result,
                system_info=system_info
            )

            # 4. Add the unique outputs from this agent back into the final result
            final_result["status"] = "success"
            final_result["fitting_parameters"] = fit_params
            final_result["literature_files"] = saved_lit_files

            return final_result

        except Exception as e:
            self.logger.exception(f"Curve analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def _get_claims_instruction_prompt(self) -> str:
        return FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS