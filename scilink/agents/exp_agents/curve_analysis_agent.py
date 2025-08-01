# scilink/agents/exp_agents/curve_analysis_agent.py

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json
import os

from .base_agent import BaseAnalysisAgent
from .human_feedback import SimpleFeedbackMixin
from .executors import StructureExecutor
from ..lit_agents.literature_agent import FittingModelLiteratureAgent
from .instruct import (
    LITERATURE_QUERY_GENERATION_INSTRUCTIONS,
    FITTING_SCRIPT_GENERATION_INSTRUCTIONS,
    FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS
)

class CurveAnalysisAgent(SimpleFeedbackMixin, BaseAnalysisAgent):
    """
    Agent for analyzing 1D curves via automated, literature-informed fitting.
    """
    def __init__(self, google_api_key: str = None, futurehouse_api_key: str = None, 
                 model_name: str = "gemini-2.5-pro-preview-06-05", local_model: str = None, 
                 enable_human_feedback: bool = True, executor_timeout: int = 60):
        super().__init__(google_api_key, model_name, local_model, enable_human_feedback=enable_human_feedback)
        self.executor = StructureExecutor(timeout=executor_timeout, enforce_sandbox=True)
        self.literature_agent = FittingModelLiteratureAgent(api_key=futurehouse_api_key)

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
        data_preview = np.array2string(curve_data[:10], precision=4, separator=', ')
        prompt = (
            f"{FITTING_SCRIPT_GENERATION_INSTRUCTIONS}\n"
            f"## Literature Context\n{literature_context}\n"
            f"## Curve Data Preview\n{data_preview}\n"
            f"## Instructions\nWrite a script that loads data from the file '{os.path.abspath(data_path)}' and performs the fit."
        )
        response = self.model.generate_content(prompt)
        script_content = response.text.strip()
        if script_content.startswith("```python"):
            script_content = script_content[9:]
        if script_content.endswith("```"):
            script_content = script_content[:-3]
        if not script_content:
            raise ValueError("LLM failed to generate a fitting script.")
        return script_content

    def analyze_for_claims(self, data_path: str, system_info: dict = None, output_dir: str = ".") -> dict:
        try:
            # Step 1: Load Data and Visualize
            curve_data = self._load_curve_data(data_path)
            system_info = self._handle_system_info(system_info)
            original_plot_bytes = self._plot_curve(curve_data, system_info, " (Original Data)")

            # Step 2: Search Literature for Fitting Models
            lit_query = self._generate_lit_search_query(original_plot_bytes, system_info)
            lit_result = self.literature_agent.query_for_models(lit_query)
            if lit_result["status"] != "success":
                raise RuntimeError(f"Literature search for models failed: {lit_result['message']}")
            literature_context = lit_result["formatted_answer"]

            # Step 3: Generate Fitting Script
            fitting_script = self._generate_fitting_script(curve_data, literature_context, data_path)
            
            # Step 4: Execute Fitting Script
            exec_result = self.executor.execute_script(fitting_script, working_dir=output_dir)
            if exec_result.get("status") != "success":
                raise RuntimeError(f"Fitting script execution failed: {exec_result.get('message')}")

            # Step 5: Parse Results
            fit_params = {}
            for line in exec_result.get("stdout", "").splitlines():
                if line.startswith("FIT_RESULTS_JSON:"):
                    fit_params = json.loads(line.replace("FIT_RESULTS_JSON:", ""))
                    break
            if not fit_params:
                raise ValueError("Could not parse fitting parameters from script output.")
            
            fit_plot_path = os.path.join(output_dir, "fit_visualization.png")
            with open(fit_plot_path, "rb") as f:
                fit_plot_bytes = f.read()

            # Step 6: Final Interpretation and Claim Generation
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

            return {
                "status": "success",
                "detailed_analysis": result_json.get("detailed_analysis"),
                "scientific_claims": self._validate_scientific_claims(result_json.get("scientific_claims", [])),
                "fitting_parameters": fit_params
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_claims_instruction_prompt(self) -> str:
        return FITTING_RESULTS_INTERPRETATION_INSTRUCTIONS