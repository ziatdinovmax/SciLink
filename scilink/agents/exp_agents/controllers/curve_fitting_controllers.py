# controllers/curve_fitting_controllers.py

"""
Curve Fitting Controllers - Complete Module

This module contains:
1. Original controllers for single-spectrum analysis steps
2. Unified controllers that handle both single spectrum (n=1) and series (n>1) analysis

Key principle for series analysis: Single spectrum = Series of 1

Quality control features:
- Automatic model retry when R² is inadequate
- Statistical outlier detection for series
- Human feedback integration for unresolved quality issues
"""

# Set non-interactive backend BEFORE importing pyplot anywhere
import matplotlib
matplotlib.use('Agg')

import subprocess
import json
import logging
import os
import base64
import re
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Any, Dict, List
import numpy as np


def build_verification_prompt_with_history(
    current_fit: dict,
    previous_iterations: List[dict],
) -> str:
    """Build history context string for verification prompt."""
    if not previous_iterations:
        return ""
    
    lines = [
        "\n\n## PREVIOUS VERIFICATION ATTEMPTS",
        "Review what was tried before. Don't suggest fixes that already failed.\n"
    ]
    
    for i, prev in enumerate(previous_iterations, 1):
        lines.append(f"\n### Attempt {i}")
        r2 = prev.get('r_squared')
        lines.append(f"- R² = {r2:.4f}" if r2 is not None else "- R² = N/A")
        lines.append(f"- Config: {prev.get('config_used', {}).get('physical_model', 'N/A')}")
        lines.append(f"- Assessment: {prev.get('overall_assessment', 'N/A')}")
        
        issues = prev.get('issues_found', [])
        if issues:
            lines.append(f"- Issues ({len(issues)}):")
            for issue in issues:
                lines.append(f"  • {issue.get('location', '?')}: {issue.get('problem', '?')}")
        
        if prev.get('recommended_action'):
            lines.append(f"- Action taken: {prev['recommended_action']}")
    
    lines.extend([
        "\n\n## IMPORTANT",
        "1. Check if previous issues were RESOLVED or still PERSIST",
        "2. If a fix didn't work, suggest something DIFFERENT",
    ])
    
    return "\n".join(lines)


def _append_auxiliary_context(prompt: list, state: dict) -> None:
    """Append auxiliary reference data to an LLM prompt if available."""
    if not state.get("auxiliary_plot_bytes"):
        return
    label = state.get("auxiliary_label", "Auxiliary data")
    summary = state.get("auxiliary_summary", "")
    prompt.append(f"\n## Auxiliary Reference Data: {label}")
    prompt.append(
        f"The user provided this auxiliary reference data: {label}. "
        "Take it into account in your analysis and interpretation, but do NOT "
        "fit or quantitatively analyze this auxiliary data."
    )
    if summary:
        prompt.append(f"\nData summary: {summary}")
    prompt.append({
        "mime_type": state.get("auxiliary_mime_type", "image/png"),
        "data": state["auxiliary_plot_bytes"]
    })


def _append_skill_context(prompt: list, state: dict, stage: str) -> None:
    """Append domain skill knowledge to an LLM prompt for the given stage.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``skill_sections`` and ``skill_name``.
        stage: One of ``"planning"``, ``"analysis"``, ``"interpretation"``, ``"validation"``.
    """
    sections = state.get("skill_sections")
    if not sections:
        return

    skill_name = state.get("skill_name", "domain skill")
    content = sections.get(stage, "")
    if not content:
        return

    prompt.append(f"\n## MANDATORY Domain Skill Rules: {skill_name} ({stage})")
    prompt.append(
        "The following rules are MANDATORY. Your analysis plan and implementation "
        "MUST conform to these domain-specific requirements. These rules encode "
        "validated domain expertise and take precedence over general-purpose defaults. "
        "Do NOT substitute your own preferences where these rules specify a method, "
        "treatment, or constraint."
    )
    prompt.append(content)

    # Include validation rules during planning and interpretation
    # so the LLM knows quality criteria upfront
    if stage in ("planning", "interpretation"):
        validation = sections.get("validation", "")
        if validation:
            prompt.append(f"\n## MANDATORY Domain Validation Rules ({skill_name})")
            prompt.append(validation)


def _append_prior_knowledge_context(prompt: list, state: dict) -> None:
    """Append prior knowledge from reference analyses to an LLM prompt.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``prior_knowledge`` list.
    """
    knowledge = state.get("prior_knowledge", [])
    if not knowledge:
        return
    prompt.append("\n## Prior Knowledge from Reference Analyses")
    prompt.append(
        "The following knowledge was derived from prior reference analyses. "
        "Use it to inform your analysis approach, model selection, and interpretation."
    )
    for entry in knowledge:
        prompt.append(f"\n### {entry.get('focus', 'Reference findings')}")
        prompt.append(entry.get("summary", ""))
        findings = entry.get("key_findings", [])
        if findings:
            prompt.append("\nKey findings:")
            for f in findings:
                prompt.append(f"- {f}")


def _append_objective_context(prompt: list, state: dict) -> None:
    """Append high-level scientific objective to an LLM prompt.

    The objective is injected as a top-level framing directive that tells the
    LLM *why* the analysis is being performed and *what question* to answer.
    It is distinct from ``analysis_hints`` which provide tactical guidance on
    *how* to analyze.

    Args:
        prompt: Mutable list of prompt parts to extend.
        state: Pipeline state dict containing ``analysis_objective``.
    """
    objective = state.get("analysis_objective")
    if not objective:
        return
    prompt.append(
        f"\n## Analysis Objective\n"
        f"The overarching scientific objective of this analysis is: {objective}\n"
        f"Frame your analysis, model selection, and interpretation around "
        f"answering this objective. All findings should be evaluated in terms "
        f"of how they contribute to resolving this question."
    )


class AnalyzeDataController:
    """Compute data statistics and create initial visualization."""

    def __init__(self, logger: logging.Logger, plot_fn: Callable):
        self.logger = logger
        self.plot_fn = plot_fn

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        self.logger.info("\n🔍 --- Analyzing Data ---\n")

        try:
            data = state["curve_data"]

            if data.ndim == 1:
                x = np.arange(len(data))
                y = data
            elif data.shape[0] == 2:
                x, y = data[0], data[1]
            elif data.shape[1] == 2:
                x, y = data[:, 0], data[:, 1]
            else:
                raise ValueError(f"Unexpected data shape: {data.shape}")

            state["data_statistics"] = {
                "n_points": len(x),
                "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
                "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
                "y_mean": float(np.nanmean(y)),
                "y_std": float(np.nanstd(y)),
                "has_nans": bool(np.any(np.isnan(data))),
            }

            plot_bytes = self.plot_fn(state["curve_data"], state.get("system_info", {}))
            state["original_plot_bytes"] = plot_bytes
            state["analysis_images"] = [{"label": "Raw Data", "data": plot_bytes}]

            self.logger.info(f"  Points: {state['data_statistics']['n_points']}")
            self.logger.info(f"  X: {state['data_statistics']['x_range']}")
            self.logger.info(f"  Y: {state['data_statistics']['y_range']}")

        except Exception as e:
            self.logger.error(f"❌ Data analysis failed: {e}", exc_info=True)
            state["error_dict"] = {"error": "Data analysis failed", "details": str(e)}

        return state


class SeriesScoutController:
    """Scout representative spectra across a series before planning.

    For series with n > 1, loads and plots representative spectra
    (evenly spaced, capped at 7) so the LLM can see how data evolves
    across the series and plan fitting regimes proactively.

    For n == 1: no-op (state passes through unchanged).
    """

    def __init__(
        self,
        logger: logging.Logger,
        plot_fn: Callable,
    ):
        self.logger = logger
        self.plot_fn = plot_fn

    @staticmethod
    def _select_scout_indices(num_spectra: int) -> list:
        """Select evenly spaced representative indices, capped at 7."""
        if num_spectra <= 3:
            return list(range(num_spectra))
        if num_spectra <= 6:
            mid = num_spectra // 2
            return sorted({0, mid, num_spectra - 1})
        if num_spectra <= 15:
            indices = {0, num_spectra // 4, num_spectra // 2,
                       3 * num_spectra // 4, num_spectra - 1}
            return sorted(indices)
        # Large series: 7 evenly spaced
        step = (num_spectra - 1) / 6
        indices = {round(i * step) for i in range(7)}
        return sorted(indices)

    @staticmethod
    def _compute_statistics(curve_data: np.ndarray) -> dict:
        if curve_data.ndim == 1:
            x = np.arange(len(curve_data))
            y = curve_data
        elif curve_data.shape[0] == 2:
            x, y = curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            x, y = curve_data[:, 0], curve_data[:, 1]
        else:
            raise ValueError(f"Unexpected data shape: {curve_data.shape}")
        return {
            "n_points": len(x),
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
            "has_nans": bool(np.any(np.isnan(curve_data))),
        }

    @staticmethod
    def _extract_xy(curve_data: np.ndarray):
        """Extract x, y arrays from various data shapes."""
        if curve_data.ndim == 1:
            return np.arange(len(curve_data)), curve_data
        elif curve_data.shape[0] == 2:
            return curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            return curve_data[:, 0], curve_data[:, 1]
        raise ValueError(f"Unexpected data shape: {curve_data.shape}")

    @staticmethod
    def _create_overlay_plot(
        scout_curves: list,
        system_info: dict,
    ) -> bytes:
        """Create a single overlay figure with all scout spectra.

        Args:
            scout_curves: list of {"label": str, "curve_data": np.ndarray}
            system_info: metadata dict for axis labels
        Returns:
            PNG image bytes.
        """
        import matplotlib.pyplot as plt
        import io

        fig, ax = plt.subplots(figsize=(10, 6))

        cmap = plt.cm.viridis
        n = len(scout_curves)
        for i, entry in enumerate(scout_curves):
            x, y = SeriesScoutController._extract_xy(entry["curve_data"])
            color = cmap(i / max(n - 1, 1))
            ax.plot(x, y, color=color, linewidth=1.2, label=entry["label"])

        ax.set_xlabel(system_info.get("xlabel", "X"))
        ax.set_ylabel(system_info.get("ylabel", "Y"))
        ax.set_title(
            system_info.get("title", "Data")
            + " — Scout Overlay"
        )
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _load_spectrum(self, idx: int, state: dict) -> np.ndarray:
        spectrum_stack = state.get("spectrum_stack")
        if spectrum_stack is not None:
            return spectrum_stack[idx]
        data_path = state.get("spectrum_paths", [])[idx]
        try:
            from ....tools.curve_fitting_tools import load_curve_data
            return load_curve_data(data_path)
        except ImportError:
            if data_path.endswith('.npy'):
                return np.load(data_path)
            return np.loadtxt(data_path, delimiter=',')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict") or state.get("is_single_spectrum", True):
            return state

        num_spectra = state.get("num_spectra", 1)
        if num_spectra <= 1:
            return state

        self.logger.info("\n🔭 --- Scouting Series ---\n")

        scout_indices = self._select_scout_indices(num_spectra)
        series_metadata = state.get("series_metadata", {})
        values = series_metadata.get("values", [])
        variable = series_metadata.get("variable", "index")
        unit = series_metadata.get("unit", "")

        scout_data = []
        scout_curves = []  # for overlay plot
        for idx in scout_indices:
            try:
                curve_data = self._load_spectrum(idx, state)

                stats = self._compute_statistics(curve_data)

                if idx < len(values):
                    label = f"{variable}={values[idx]} {unit}".strip()
                else:
                    label = f"index {idx}"

                plot_bytes = self.plot_fn(
                    curve_data, state.get("system_info", {}),
                    title_suffix=f" [{label}]",
                )

                scout_data.append({
                    "index": idx,
                    "label": label,
                    "statistics": stats,
                    "plot_bytes": plot_bytes,
                })
                scout_curves.append({
                    "label": label,
                    "curve_data": curve_data,
                })
                self.logger.info(f"  Scouted spectrum {idx}: {label}")
            except Exception as e:
                self.logger.warning(f"  Failed to scout spectrum {idx}: {e}")

        # Generate overlay comparison plot
        if len(scout_curves) >= 2:
            try:
                overlay_bytes = self._create_overlay_plot(
                    scout_curves, state.get("system_info", {})
                )
                state["scout_overlay_plot"] = overlay_bytes
                self.logger.info("  Generated overlay comparison plot")
            except Exception as e:
                self.logger.warning(f"  Failed to create overlay plot: {e}")
                state["scout_overlay_plot"] = None
        else:
            state["scout_overlay_plot"] = None

        state["scout_data"] = scout_data
        self.logger.info(f"  Scouted {len(scout_data)} of {num_spectra} spectra")

        return state


class LiteratureSearchController:
    """Search literature if enabled and query provided."""

    def __init__(
        self,
        logger: logging.Logger,
        literature_agent: Any | None,
        output_dir: str,
    ):
        self.logger = logger
        self.literature_agent = literature_agent
        self.output_dir = output_dir

    def _save_results(self, query: str, report: str) -> dict:
        saved_files = {}
        try:
            lit_dir = os.path.join(self.output_dir, "literature")
            os.makedirs(lit_dir, exist_ok=True)

            query_path = os.path.join(lit_dir, "search_query.txt")
            with open(query_path, "w") as f:
                f.write(query)
            saved_files["query_file"] = query_path

            report_path = os.path.join(lit_dir, "literature_report.md")
            with open(report_path, "w") as f:
                f.write(report)
            saved_files["report_file"] = report_path
        except Exception as e:
            self.logger.warning(f"Failed to save literature: {e}")
        return saved_files

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if self.literature_agent is None:
            self.logger.info("\n📚 --- Skipping Literature (disabled) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        query = state.get("literature_query")
        if not query:
            self.logger.info("\n📚 --- Skipping Literature (no query needed) ---\n")
            state["literature_context"] = None
            state["literature_files"] = None
            return state

        self.logger.info("\n📚 --- Searching Literature ---\n")
        self.logger.info(f"  Query: {query}")

        try:
            result = self.literature_agent.query_for_models(query)
            if result.get("status") == "success":
                state["literature_context"] = result["formatted_answer"]
                self.logger.info("  ✅ Success")
            else:
                state["literature_context"] = None
                self.logger.warning("  ⚠️ No results")

            state["literature_files"] = self._save_results(
                query, state["literature_context"] or f"No results: {result.get('message')}"
            )
        except Exception as e:
            self.logger.error(f"  ❌ Failed: {e}")
            state["literature_context"] = None
            state["literature_files"] = self._save_results(query, f"Error: {e}")

        return state


class GenerateCurveFittingReportController:
    """Generates a human-readable HTML report for curve fitting analysis."""
    
    DEFAULT_R2_THRESHOLD = 0.95

    def __init__(self, logger: logging.Logger, output_dir: str, r2_threshold: float = None):
        self.logger = logger
        self.output_dir = output_dir
        self.r2_threshold = r2_threshold if r2_threshold is not None else self.DEFAULT_R2_THRESHOLD

    def _image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        if not state.get("is_single_spectrum", True):
            return state
            
        self.logger.info("\n📄 --- Generating HTML Report ---\n")
        
        result_json = state.get("result_json", {})
        fit_results = state.get("fit_results", {})
        synthesis_result = state.get("synthesis_result", {})
        
        detailed_analysis = synthesis_result.get("detailed_analysis") or result_json.get("detailed_analysis", "No analysis provided.")
        scientific_claims = synthesis_result.get("scientific_claims") or result_json.get("scientific_claims", [])
        system_info = state.get("system_info", {})
        model_type = fit_results.get("model_type", result_json.get("model_type", "N/A"))
        parameters = fit_results.get("parameters", result_json.get("fitting_parameters", {}))
        fit_quality = fit_results.get("fit_quality", result_json.get("fit_quality", {}))
        caveats = synthesis_result.get("caveats") or result_json.get("caveats", "")
        
        quality_warning = None
        series_results = state.get("series_results", [])
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]
        
        original_plot = state.get("original_plot_bytes")
        fit_plot = state.get("final_plot_bytes")
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CurveFitting_Report_{file_timestamp}.html"
        filepath = output_dir / filename

        params_html = self._format_parameters(parameters)
        quality_html = self._format_fit_quality(fit_quality, quality_warning)

        html_content = self._build_html_report(
            timestamp, state, system_info, model_type, quality_html, 
            detailed_analysis, original_plot, fit_plot, params_html,
            scientific_claims, caveats
        )

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"  ✅ Report saved: {filepath}")
            state["report_path"] = str(filepath)
        except Exception as e:
            self.logger.error(f"  ❌ Failed to write report: {e}")

        return state

    def _build_html_report(self, timestamp, state, system_info, model_type, 
                          quality_html, detailed_analysis, original_plot, 
                          fit_plot, params_html, scientific_claims, caveats):
        """Build the complete HTML report."""
        system_info_str = self._format_system_info(system_info)
        
        images_html = ""
        if original_plot:
            b64_original = self._image_to_base64(original_plot)
            images_html += f'<div class="image-card"><img src="data:image/png;base64,{b64_original}" alt="Original Data"><div class="image-label">Original Data</div></div>'
        if fit_plot:
            b64_fit = self._image_to_base64(fit_plot)
            images_html += f'<div class="image-card"><img src="data:image/png;base64,{b64_fit}" alt="Fit Visualization"><div class="image-label">Fit Result with Residuals</div></div>'

        claims_html = ""
        if not scientific_claims:
            claims_html = "<p>No specific claims generated.</p>"
        else:
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                claims_html += f"""
        <div class="claim-card">
            <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
            <p><strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
            <p><strong>Literature Search Query:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
            <p><strong>Keywords:</strong> {keywords_str}</p>
        </div>"""

        caveats_html = ""
        if caveats:
            caveats_html = f"""
        <h2>5. Caveats & Limitations</h2>
        <div class="caveats">{caveats}</div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curve Fitting Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
        .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        h3 {{ color: #16a085; margin-top: 20px; }}
        .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
        .model-box {{ background-color: #e8f4fc; padding: 15px; border-radius: 5px; border-left: 5px solid #2980b9; margin-bottom: 15px; }}
        .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }}
        .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; margin-top: 20px; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; }}
        .image-label {{ margin-top: 12px; font-weight: bold; color: #444; font-size: 1em; border-top: 1px solid #eee; padding-top: 10px; }}
        .params-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .params-table th, .params-table td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        .params-table th {{ background-color: #f8f9fa; font-weight: 600; color: #2c3e50; }}
        .params-table tr:hover {{ background-color: #f5f5f5; }}
        .quality-badge {{ display: inline-block; padding: 5px 12px; border-radius: 20px; font-weight: bold; margin-right: 10px; }}
        .quality-good {{ background-color: #d4edda; color: #155724; }}
        .quality-ok {{ background-color: #fff3cd; color: #856404; }}
        .quality-poor {{ background-color: #f8d7da; color: #721c24; }}
        .quality-warning-box {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px 15px; margin-top: 10px; border-radius: 0 5px 5px 0; font-size: 0.9em; }}
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Curve Fitting Analysis Report</h1>
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Data Source:</strong> {state.get('data_path', 'N/A')}</p>
            <p><strong>Sample Info:</strong> {system_info_str}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <h3>Fitting Model</h3>
        <div class="model-box">{model_type}</div>
        <h3>Fit Quality</h3>
        {quality_html}
        <h3>Interpretation</h3>
        <div class="analysis-text">{detailed_analysis}</div>
        <h2>2. Visualizations</h2>
        <div class="image-grid">{images_html}</div>
        <h2>3. Fitted Parameters</h2>
        {params_html}
        <h2>4. Scientific Claims</h2>
        {claims_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Curve Fitting Analysis Agent</div>
    </div>
</body>
</html>"""

    def _format_system_info(self, system_info: dict) -> str:
        if not system_info:
            return "N/A"
        parts = [f"{k}: {v}" for k, v in system_info.items() if v]
        return ", ".join(parts) if parts else "N/A"

    def _format_parameters(self, parameters: dict) -> str:
        if not parameters:
            return "<p>No parameters extracted.</p>"

        rows = ""
        for component, params in parameters.items():
            if isinstance(params, dict):
                first_row = True
                for param_name, value in params.items():
                    if param_name.endswith("_err"):
                        continue
                    err_key = f"{param_name}_err"
                    err_value = params.get(err_key, "—")
                    if isinstance(err_value, (int, float)):
                        err_value = f"± {err_value:.4g}"
                    if isinstance(value, (int, float)):
                        value_str = f"{value:.4g}"
                    else:
                        value_str = str(value)
                    component_display = component if first_row else ""
                    rows += f"<tr><td><strong>{component_display}</strong></td><td>{param_name}</td><td>{value_str}</td><td>{err_value}</td></tr>"
                    first_row = False
            else:
                rows += f"<tr><td><strong>{component}</strong></td><td>—</td><td>{parameters[component]}</td><td>—</td></tr>"

        return f"""<table class="params-table">
            <thead><tr><th>Component</th><th>Parameter</th><th>Value</th><th>Uncertainty</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>"""

    def _format_fit_quality(self, fit_quality: dict, quality_warning: str = None) -> str:
        if not fit_quality:
            return "<p>No quality metrics available.</p>"

        r_squared = fit_quality.get("r_squared", fit_quality.get("r2"))
        rmse = fit_quality.get("rmse")
        chi_squared = fit_quality.get("chi_squared_reduced", fit_quality.get("reduced_chi_squared"))

        html = "<div>"
        if r_squared is not None:
            if r_squared >= self.r2_threshold + 0.04:
                badge_class, label = "quality-good", "Excellent"
            elif r_squared >= self.r2_threshold:
                badge_class, label = "quality-ok", "Good"
            else:
                badge_class, label = "quality-poor", "Poor"
            html += f'<span class="quality-badge {badge_class}">{label}</span><strong>R² = {r_squared:.4f}</strong>'

        if rmse is not None:
            html += f" &nbsp;|&nbsp; <strong>RMSE = {rmse:.4g}</strong>"
        if chi_squared is not None:
            html += f" &nbsp;|&nbsp; <strong>χ²/DOF = {chi_squared:.3f}</strong>"
        html += "</div>"
        
        if quality_warning:
            html += f'<div class="quality-warning-box">⚠️ <strong>Note:</strong> {quality_warning}. Alternative models were attempted but could not improve fit quality significantly.</div>'
        
        return html


# ============================================================================
# UNIFIED CONTROLLERS (for series analysis support)
# ============================================================================

class HumanFeedbackRefinementController:
    """
    Facilitates human-in-the-loop parameter refinement for the first spectrum.
    
    Works identically for single spectra and series:
    - Single spectrum: Refine fitting, then process that one spectrum
    - Series: Refine fitting on first spectrum, then apply to all
    """
    
    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        instructions: str,
        output_dir: str,
        enable_human_feedback: bool = False,
        max_iterations: int = 5
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.instructions = instructions
        self.output_dir = Path(output_dir)
        self.enable_human_feedback = enable_human_feedback
        self.max_iterations = max_iterations

    def _display_plan(self, state: dict) -> None:
        is_single = state.get("is_single_spectrum", True)
        num_spectra = state.get("num_spectra", 1)
        
        print("\n" + "=" * 60)
        mode_str = "SINGLE SPECTRUM" if is_single else f"SERIES ({num_spectra} spectra)"
        print(f"📋 PROPOSED FITTING PLAN - {mode_str}")
        print("=" * 60)
        
        if state.get("observations"):
            print(f"\n🔍 Observations:\n   {state['observations']}")
        
        print(f"\n📊 Approach:\n   {state.get('analysis_approach', 'N/A')}")
        print(f"\n📐 Physical Model:\n   {state.get('physical_model', 'N/A')}")
        print(f"\n🎯 Parameters to Extract:\n   {', '.join(state.get('parameters_to_extract', [])) or 'N/A'}")
        import re as _re
        _strategy = state.get("fitting_strategy", "N/A")
        # Put each numbered step on its own line with consistent indentation.
        # Only split on step numbers that follow a sentence-ending ". " to
        # avoid mangling numbers in text (e.g. "cm-1.", "8.7").
        _strategy = _re.sub(r"\. (\d+)\. ", r".\n   \1. ", _strategy)
        print(f"\n⚙️  Fitting Strategy:\n   {_strategy}")
        
        if state.get("literature_query"):
            print(f"\n📚 Literature Query:\n   {state['literature_query']}")
        
        # Display regime plan if present
        series_plan = state.get("series_analysis_plan")
        if series_plan and series_plan.get("regimes") and not is_single:
            regimes = series_plan["regimes"]
            print(f"\n{'=' * 60}")
            print(f"📦 SERIES FITTING REGIMES ({len(regimes)} regimes)")
            print(f"{'=' * 60}")
            if series_plan.get("rationale"):
                print(f"\nRationale: {series_plan['rationale']}")

            series_metadata = state.get("series_metadata", {})
            values = series_metadata.get("values", [])
            unit = series_metadata.get("unit", "")

            for i, regime in enumerate(regimes, 1):
                indices = regime.get("spectrum_indices", [])
                if values and indices:
                    valid_vals = [values[idx] for idx in indices if idx < len(values)]
                    range_str = f" ({min(valid_vals)}-{max(valid_vals)} {unit})" if valid_vals else ""
                else:
                    range_str = ""
                print(f"\n  Regime {i}: {regime.get('name', 'Unnamed')}")
                print(f"    Spectra: indices {indices}{range_str}")
                print(f"    Model: {regime.get('physical_model', 'N/A')}")
                print(f"    Strategy: {regime.get('fitting_strategy', 'N/A')}")
                print(f"    Parameters: {', '.join(regime.get('parameters_to_extract', []))}")

            transitions = series_plan.get("transition_points", [])
            if transitions:
                print(f"\n  Transition Points:")
                for t in transitions:
                    print(f"    Between indices {t.get('between_indices', '?')}: "
                          f"{t.get('description', 'N/A')}")
        elif not is_single:
            print(f"\n📦 **Note:** This fitting model will be LOCKED and applied to all {num_spectra} spectra.")

        print("\n" + "=" * 60)

    def _get_human_feedback(self, state: dict) -> dict:
        self._display_plan(state)
        feedback = input("\n🤔 Your feedback (or Enter to accept): ").strip()
        
        if feedback == "":
            print("✅ Plan accepted.")
            return state
        else:
            state["_refine_requested"] = True
            state["_refine_feedback"] = feedback
            return state

    def _plan_analysis(self, state: dict) -> dict:
        prompt = [
            self.instructions,
            "\n## Data Plot",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ]

        _append_objective_context(prompt, state)

        if state.get("analysis_hints"):
            prompt.append(f"\n## User Guidance\n{state['analysis_hints']}")

        _append_auxiliary_context(prompt, state)
        _append_skill_context(prompt, state, "planning")
        _append_prior_knowledge_context(prompt, state)

        # Series context: use scout data if available, otherwise basic notice
        num_spectra = state.get("num_spectra", 1)
        scout_data = state.get("scout_data", [])
        if scout_data and not state.get("is_single_spectrum", True):
            self._append_scout_context(prompt, state, scout_data)
        elif not state.get("is_single_spectrum", True):
            prompt.append(
                f"\n## Series Context\nThis is the first spectrum in a series of {num_spectra}. "
                "The fitting model you choose will be applied to ALL spectra in the series."
            )

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            raise ValueError(f"Failed to parse: {error}")

        state["observations"] = result.get("observations", "")
        state["analysis_approach"] = result.get("analysis_approach", "Curve fitting")
        state["physical_model"] = result.get("physical_model", "Appropriate model")
        state["parameters_to_extract"] = result.get("parameters_to_extract", [])
        state["fitting_strategy"] = result.get("fitting_strategy", "Standard fitting")
        state["literature_query"] = result.get("literature_query")

        # Extract series analysis plan if present
        self._extract_series_plan(state, result)

        return state

    def _append_scout_context(self, prompt: list, state: dict, scout_data: list) -> None:
        """Append scout spectrum plots and series regime planning instructions."""
        from ..instruct import SERIES_REGIME_PLANNING_SUPPLEMENT

        num_spectra = state.get("num_spectra", 1)
        series_metadata = state.get("series_metadata", {})

        prompt.append(f"\n## Series Overview ({num_spectra} spectra)")
        prompt.append(
            "Below are representative spectra from across the series. "
            "Examine how the data changes. If the spectral character changes "
            "significantly (e.g., peak splitting, new features, major shape "
            "changes), plan multiple fitting regimes. Otherwise, a single "
            "model is fine."
        )

        if series_metadata.get("variable"):
            values = series_metadata.get("values", [])
            unit = series_metadata.get("unit", "")
            prompt.append(
                f"\nSeries variable: {series_metadata['variable']} ({unit})"
            )
            if values:
                prompt.append(f"Range: {values[0]} to {values[-1]} {unit}")

        # Overlay comparison plot (all scouts on one figure)
        overlay = state.get("scout_overlay_plot")
        if overlay:
            prompt.append(
                "\n### Overlay Comparison\n"
                "All scout spectra plotted together for direct visual comparison. "
                "Look for shifts, shape changes, peak splitting, or new features "
                "emerging across the series."
            )
            prompt.append({
                "mime_type": "image/png",
                "data": overlay,
            })

        prompt.append("\n### Individual Scout Spectra")
        for scout in scout_data:
            prompt.append(
                f"\n### Spectrum at {scout['label']} (index {scout['index']})"
            )
            prompt.append(f"Statistics: {json.dumps(scout['statistics'], indent=2)}")
            prompt.append({
                "mime_type": "image/png",
                "data": scout["plot_bytes"],
            })

        prompt.append(SERIES_REGIME_PLANNING_SUPPLEMENT.format(
            num_spectra=num_spectra,
            num_spectra_minus_1=num_spectra - 1,
        ))

    def _extract_series_plan(self, state: dict, result: dict) -> None:
        """Extract and validate series_analysis_plan from LLM response."""
        series_plan = result.get("series_analysis_plan")
        if not series_plan or state.get("is_single_spectrum", True):
            state["series_analysis_plan"] = None
            return

        num_spectra = state.get("num_spectra", 1)
        regimes = series_plan.get("regimes", [])

        if not regimes:
            state["series_analysis_plan"] = None
            return

        # Validate index coverage
        all_indices = set()
        for regime in regimes:
            indices = regime.get("spectrum_indices", [])
            # Filter to valid range
            regime["spectrum_indices"] = [i for i in indices if 0 <= i < num_spectra]
            all_indices.update(regime["spectrum_indices"])

        missing = set(range(num_spectra)) - all_indices
        if missing:
            self.logger.warning(
                f"  Series plan missing indices {sorted(missing)}, "
                f"assigning to first regime"
            )
            regimes[0]["spectrum_indices"] = sorted(
                set(regimes[0]["spectrum_indices"]) | missing
            )

        state["series_analysis_plan"] = series_plan
        self.logger.info(
            f"  Series analysis plan: {len(regimes)} regime(s)"
        )
        for regime in regimes:
            self.logger.info(
                f"    {regime.get('name', 'unnamed')}: "
                f"indices {regime.get('spectrum_indices', [])}, "
                f"model: {regime.get('physical_model', 'N/A')}"
            )

    def _refine_plan(self, state: dict, feedback: str) -> dict:
        current_plan = (
            f"Observations: {state.get('observations', 'N/A')}\n"
            f"Approach: {state.get('analysis_approach', 'N/A')}\n"
            f"Physical Model: {state.get('physical_model', 'N/A')}\n"
            f"Parameters: {', '.join(state.get('parameters_to_extract', []))}\n"
            f"Strategy: {state.get('fitting_strategy', 'N/A')}"
        )

        prompt = [
            self.instructions,
            "\n## Data Plot",
            {"mime_type": "image/png", "data": state["original_plot_bytes"]},
            "\n## Data Statistics\n" + json.dumps(state["data_statistics"], indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
            f"\n## Current Plan\n{current_plan}",
            f"\n## User Feedback\nAdjust the plan based on this feedback: \"{feedback}\"",
        ]

        _append_objective_context(prompt, state)

        if state.get("analysis_hints"):
            prompt.append(f"\n## Original Guidance\n{state['analysis_hints']}")

        _append_auxiliary_context(prompt, state)
        _append_skill_context(prompt, state, "planning")
        _append_prior_knowledge_context(prompt, state)

        # Include current series plan and scout data in refinement context
        if state.get("series_analysis_plan"):
            prompt.append(
                f"\n## Current Series Analysis Plan\n"
                f"{json.dumps(state['series_analysis_plan'], indent=2)}"
            )
            prompt.append(
                "\nThe user may want to adjust regime boundaries, merge regimes, "
                "change models for specific regimes, or switch to a single model. "
                "Adjust the series_analysis_plan accordingly, or remove it entirely "
                "if the user wants a single model."
            )
        scout_data = state.get("scout_data", [])
        if scout_data and not state.get("is_single_spectrum", True):
            self._append_scout_context(prompt, state, scout_data)

        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        result, error = self._parse(response)

        if error or not result:
            self.logger.warning(f"Refinement failed: {error}. Keeping current plan.")
            return state

        state["observations"] = result.get("observations", state.get("observations", ""))
        state["analysis_approach"] = result.get("analysis_approach", state.get("analysis_approach"))
        state["physical_model"] = result.get("physical_model", state.get("physical_model"))
        state["parameters_to_extract"] = result.get("parameters_to_extract", state.get("parameters_to_extract", []))
        state["fitting_strategy"] = result.get("fitting_strategy", state.get("fitting_strategy"))
        state["literature_query"] = result.get("literature_query", state.get("literature_query"))

        # Re-extract series plan (may have been updated or removed)
        self._extract_series_plan(state, result)

        return state

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        is_single = state.get("is_single_spectrum", True)
        mode_str = "SINGLE SPECTRUM" if is_single else "SERIES"
        self.logger.info(f"\n🧠 --- Planning Analysis ({mode_str}) ---\n")

        try:
            state = self._plan_analysis(state)
            self.logger.info(f"  Approach: {state['analysis_approach']}")
            self.logger.info(f"  Model: {state['physical_model']}")

            if self.enable_human_feedback:
                iteration = 0
                while iteration < self.max_iterations:
                    state = self._get_human_feedback(state)
                    if state.pop("_refine_requested", False):
                        feedback = state.pop("_refine_feedback", "")
                        self.logger.info(f"  Refining with feedback: {feedback}")
                        print("\n🔄 Refining plan...\n")
                        state = self._refine_plan(state, feedback)
                        iteration += 1
                    else:
                        break

                if iteration >= self.max_iterations:
                    self.logger.warning("  Max iterations reached.")
                    print("⚠️  Max refinements reached. Proceeding with current plan.")

            state["locked_fitting_config"] = {
                "analysis_approach": state.get("analysis_approach"),
                "physical_model": state.get("physical_model"),
                "parameters_to_extract": state.get("parameters_to_extract", []),
                "fitting_strategy": state.get("fitting_strategy"),
            }

            # Build per-regime configs if series plan has multiple regimes
            series_plan = state.get("series_analysis_plan")
            if series_plan and series_plan.get("regimes"):
                regime_configs = {}
                for regime in series_plan["regimes"]:
                    regime_config = {
                        "analysis_approach": state.get("analysis_approach"),
                        "physical_model": regime.get(
                            "physical_model", state.get("physical_model")
                        ),
                        "parameters_to_extract": regime.get(
                            "parameters_to_extract",
                            state.get("parameters_to_extract", []),
                        ),
                        "fitting_strategy": regime.get(
                            "fitting_strategy", state.get("fitting_strategy")
                        ),
                    }
                    for idx in regime.get("spectrum_indices", []):
                        regime_configs[idx] = regime_config
                state["regime_configs"] = regime_configs
                self.logger.info(
                    f"  ✅ Locked {len(series_plan['regimes'])} regime "
                    f"configuration(s) for series processing."
                )
            else:
                state["regime_configs"] = None
                self.logger.info(
                    "  ✅ Fitting configuration locked for series processing."
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Planning failed: {e}, using fallback")
            state["observations"] = ""
            state["analysis_approach"] = "Fit the data with an appropriate model"
            state["physical_model"] = "To be determined"
            state["parameters_to_extract"] = []
            state["fitting_strategy"] = "Standard curve fitting"
            state["literature_query"] = None
            state["locked_fitting_config"] = None
            state["series_analysis_plan"] = None
            state["regime_configs"] = None

        return state


class UnifiedSeriesProcessingController:
    """
    Processes ALL spectra using the locked fitting model.
    
    Quality control features:
    - If R² < threshold, automatically tries alternative models (max_model_retries)
    - If still inadequate and human feedback enabled, asks for guidance
    - Otherwise proceeds with best available fit
    - For series: detects statistical outliers that may indicate interesting physics
    """
    
    MAX_ATTEMPTS = 5
    DEFAULT_R2_THRESHOLD = 0.95
    DEFAULT_MAX_MODEL_RETRIES = 3
    DEFAULT_OUTLIER_SIGMA = 2.0
    DEFAULT_MAX_VERIFICATION_ITERATIONS = 3

    JUDGE_PROMPT = '''You are a scientific data fitting expert acting as a judge.

Multiple fitting attempts were made but none passed automated verification. 
Review all attempts and select the most physically reasonable fit, or declare all unacceptable.

**SELECTION CRITERIA:**
1. Physical plausibility - are the model parameters reasonable for this type of data?
2. Residual structure - random noise is good, systematic patterns are bad
3. Component necessity - each component should fit a real feature in the data, not noise or baseline artifacts
4. Parsimony - prefer simpler models if fit quality is similar

**ATTEMPTS:**
{attempts_summary}

**VISUALIZATIONS:**
(See images below for each attempt)

Examine each fit carefully. Look at:
- Whether the model captures the key features in the data
- Whether component parameters are physically reasonable
- Whether residuals show random scatter or systematic patterns
- Whether any components appear to be fitting noise rather than real features

**Return JSON:**
{{
    "selected_index": <0, 1, 2, etc., or null if ALL are unacceptable>,
    "acceptable": true/false,
    "reasoning": "detailed explanation of your choice or why all are unacceptable",
    "issues_with_selected": "any remaining concerns with the chosen fit, or null if none"
}}

IMPORTANT: If one fit is clearly better than others (better residuals, more physical parameters), 
select it even if it's not perfect. Only return acceptable=false if ALL fits are fundamentally flawed.
'''

    ALTERNATIVE_MODELS_PROMPT = '''The current fitting model achieved R² = {r_squared:.4f}, which is below the quality threshold of {threshold}.

**Current Model:** {current_model}
**Current Parameters:** {current_params}

**Data Statistics:**
{data_stats}

**Original Analysis Approach:** {analysis_approach}

**IMPORTANT:** Examine the fit visualization provided below carefully. Look for:
1. Systematic deviations between the fit and data (not just noise)
2. Missing features (additional peaks, shoulders, asymmetry)
3. Incorrect baseline behavior
4. Wrong peak shape (too sharp, too broad, wrong tail behavior)

Based on your visual inspection and the numerical results, suggest an alternative fitting model.

**Physics-first rule:** Before adding more components, first try improving the existing model:
1. Different peak shapes (e.g., Voigt instead of Gaussian, asymmetric profiles)
2. Better baseline treatment (higher-order polynomial, different correction method)
3. Physically motivated constraints (known peak ratios, position bounds from literature)

Only add new components if you can identify a specific physical feature in the data that the
current model is missing (a visible shoulder, a second peak, etc.). Do NOT add components
just to absorb residual structure — that is overfitting, not physics.

Return JSON with:
{{
    "diagnosis": "specific description of what you observe is wrong with the current fit based on the visualization",
    "alternative_model": "description of the new model to try",
    "fitting_strategy": "specific fitting approach for the new model",
    "parameters_to_extract": ["list", "of", "parameters"]
}}
'''

    HUMAN_FEEDBACK_PROMPT = '''## Fit Quality Issue

The automated fitting could not achieve adequate fit quality.

**Best Result:** R² = {best_r2:.4f} (threshold: {threshold})
**Models Tried:**
{models_tried}

**Options:**
1. Suggest a different model or approach
2. Adjust the R² threshold for this analysis (e.g., "threshold {example_threshold:.2f}")
3. Accept the best available fit (type "accept")

Your guidance: '''

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        script_instructions: str,
        correction_instructions: str,
        quality_instructions: str,
        output_dir: str,
        plot_fn: Callable,
        r2_threshold: float = None,
        max_model_retries: int = None,
        enable_human_feedback: bool = False,
        outlier_sigma: float = None,
        max_verification_iterations: int = None,
        preprocessor: Any = None,
        conformance_instructions: str = "",
    ):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.script_instructions = script_instructions
        self.correction_instructions = correction_instructions
        self.quality_instructions = quality_instructions
        self.output_dir = Path(output_dir)
        self.plot_fn = plot_fn
        self.r2_threshold = r2_threshold if r2_threshold is not None else self.DEFAULT_R2_THRESHOLD
        self.max_model_retries = max_model_retries if max_model_retries is not None else self.DEFAULT_MAX_MODEL_RETRIES
        self.enable_human_feedback = enable_human_feedback
        self.outlier_sigma = outlier_sigma if outlier_sigma is not None else self.DEFAULT_OUTLIER_SIGMA
        self.max_verification_iterations = max_verification_iterations if max_verification_iterations is not None else self.DEFAULT_MAX_VERIFICATION_ITERATIONS
        self.preprocessor = preprocessor
        self.conformance_instructions = conformance_instructions

    def _generate_fitting_script(self, state: dict, data_path: str, stats: dict) -> str:
        config = state.get("locked_fitting_config", {})
        context_parts = []
        if state.get("literature_context"):
            context_parts.append(state["literature_context"])
        skill_sections = state.get("skill_sections")
        if skill_sections and skill_sections.get("analysis"):
            context_parts.append(
                f"## MANDATORY Domain Skill Rules ({state.get('skill_name', 'skill')})\n"
                "The following domain rules are MANDATORY and must be implemented exactly "
                "as specified. They take precedence over general-purpose fitting defaults.\n\n"
                + skill_sections["analysis"]
            )

        prompt = self.script_instructions.format(
            analysis_approach=config.get("analysis_approach", "Fit the data"),
            physical_model=config.get("physical_model", "Appropriate model"),
            parameters_to_extract=", ".join(config.get("parameters_to_extract", [])) or "relevant parameters",
            fitting_strategy=config.get("fitting_strategy", "Standard fitting"),
            context="\n".join(context_parts) or "Use your expertise.",
            data_path=data_path,
            n_points=stats["n_points"],
            x_min=stats["x_range"][0],
            x_max=stats["x_range"][1],
            y_min=stats["y_range"][0],
            y_max=stats["y_range"][1],
        )

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Script generation failed: {error or 'no script'}")

        return result["script"]

    def _correct_script(self, state: dict, script: str, error_msg: str) -> str:
        config = state.get("locked_fitting_config", {})
        prompt = self.correction_instructions.format(
            analysis_approach=config.get("analysis_approach", ""),
            physical_model=config.get("physical_model", ""),
            failed_script=script,
            error_message=error_msg,
        )
        skill_sections = state.get("skill_sections")
        if skill_sections and skill_sections.get("analysis"):
            prompt += (
                f"\n\n## MANDATORY Domain Skill Rules ({state.get('skill_name', 'skill')})\n"
                "While fixing the error, you MUST continue to follow these domain rules. "
                "Do not change the fitting model or workflow to deviate from these rules.\n\n"
                + skill_sections["analysis"]
            )

        response = self.model.generate_content(prompt)
        result, error = self._parse(response)

        if error or not result or "script" not in result:
            raise ValueError(f"Correction failed: {error or 'no script'}")

        if "diagnosis" in result:
            self.logger.info(f"    Diagnosis: {result['diagnosis']}")

        return result["script"]

    def _check_plan_conformance(self, state: dict, script: str) -> dict | None:
        """Use the LLM to verify a generated script implements the locked plan.

        Returns a dict with ``conformant``, ``justified_deviations``,
        ``unjustified_deviations``, and ``summary`` keys, or ``None`` if the
        check cannot be performed (missing config, LLM error, etc.).
        """
        config = state.get("locked_fitting_config", {})
        if not config or not config.get("physical_model"):
            return None
        if not self.conformance_instructions:
            return None

        # Build skill rules text for conformance checking
        skill_rules_text = ""
        skill_sections = state.get("skill_sections")
        if skill_sections:
            skill_name = state.get("skill_name", "domain skill")
            rules_parts = []
            for stage in ("planning", "analysis", "validation"):
                content = skill_sections.get(stage, "")
                if content:
                    rules_parts.append(f"### {stage.title()} rules\n{content}")
            if rules_parts:
                skill_rules_text = (
                    f"\n**MANDATORY Domain Skill Rules ({skill_name}):**\n"
                    + "\n".join(rules_parts)
                    + "\n"
                )

        prompt = self.conformance_instructions.format(
            analysis_approach=config.get("analysis_approach", ""),
            physical_model=config.get("physical_model", ""),
            parameters_to_extract=", ".join(
                config.get("parameters_to_extract", [])
            ),
            fitting_strategy=config.get("fitting_strategy", ""),
            skill_rules=skill_rules_text,
            script=script,
        )

        try:
            response = self.model.generate_content(contents=[prompt])
            result, error = self._parse(response)
            if error or not result:
                self.logger.debug(
                    "Plan conformance check parse failed: %s", error
                )
                return None
            return result
        except Exception as exc:
            self.logger.debug("Plan conformance check failed: %s", exc)
            return None

    def _adapt_script_for_spectrum(self, base_script: str, data_path: str, output_prefix: str) -> str:
        adapted = base_script
        adapted = adapted.replace('fit_visualization.png', f'{output_prefix}_fit.png')
        adapted = re.sub(r'spectrum_\d{4}_fit\.png', f'{output_prefix}_fit.png', adapted)
        adapted = re.sub(r'spectrum_\d{4}_T\d+K_fit\.png', f'{output_prefix}_fit.png', adapted)
        adapted = re.sub(
            r'np\.load\s*\(\s*["\'"].*?temp_spectrum_\d+\.npy["\'"]\s*\)',
            f'np.load("{data_path}")',
            adapted
        )
        adapted = re.sub(r'(["\'"]).*?temp_spectrum_\d+\.npy\1', f'"{data_path}"', adapted)
        adapted = re.sub(r'DATA_PATH\s*=\s*["\'"].*?["\'"]', f'DATA_PATH = "{data_path}"', adapted)
        adapted = re.sub(r'data_path\s*=\s*["\'"].*?["\'"]', f'data_path = "{data_path}"', adapted)
        return adapted

    def _compute_statistics(self, curve_data: np.ndarray) -> dict:
        if curve_data.ndim == 1:
            x = np.arange(len(curve_data))
            y = curve_data
        elif curve_data.shape[0] == 2:
            x, y = curve_data[0], curve_data[1]
        elif curve_data.shape[1] == 2:
            x, y = curve_data[:, 0], curve_data[:, 1]
        else:
            raise ValueError(f"Unexpected data shape: {curve_data.shape}")

        return {
            "n_points": len(x),
            "x_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "y_range": [float(np.nanmin(y)), float(np.nanmax(y))],
            "y_mean": float(np.nanmean(y)),
            "y_std": float(np.nanstd(y)),
            "has_nans": bool(np.any(np.isnan(curve_data))),
        }

    def _load_curve_data(self, data_path: str) -> np.ndarray:
        """Load curve data from file, handling various formats."""
        # Try using the project's load_curve_data function first
        try:
            from ....tools.curve_fitting_tools import load_curve_data
            return load_curve_data(data_path)
        except ImportError:
            pass
        
        
        # Fallback: handle common formats manually
        if data_path.endswith('.npy'):
            return np.load(data_path)
        elif data_path.endswith('.csv'):
            # Try to load CSV, handling potential headers
            try:
                # First try without header
                return np.loadtxt(data_path, delimiter=',')
            except ValueError:
                # If that fails, try skipping first row (header)
                try:
                    return np.loadtxt(data_path, delimiter=',', skiprows=1)
                except ValueError:
                    # Try pandas as last resort for complex CSVs
                    try:
                        import pandas as pd
                        df = pd.read_csv(data_path)
                        return df.values.T  # Transpose to get (2, n_points) shape
                    except ImportError:
                        # Re-raise the original error if pandas not available
                        raise ValueError(f"Could not parse CSV file: {data_path}. "
                                        "File may have headers or non-numeric data.")
        elif data_path.endswith('.txt'):
            # Try common text formats
            try:
                return np.loadtxt(data_path)
            except ValueError:
                try:
                    return np.loadtxt(data_path, skiprows=1)
                except ValueError:
                    # Try tab-delimited
                    try:
                        return np.loadtxt(data_path, delimiter='\t', skiprows=1)
                    except:
                        raise ValueError(f"Could not parse text file: {data_path}")
        else:
            # Generic attempt
            try:
                return np.loadtxt(data_path)
            except ValueError:
                return np.loadtxt(data_path, skiprows=1)

    def _fit_single_spectrum(
        self, 
        state: dict, 
        curve_data: np.ndarray, 
        data_path: str,
        spectrum_name: str,
        spectrum_idx: int,
        base_script: Optional[str] = None
    ) -> dict:
        stats = self._compute_statistics(curve_data)
        temp_data_path = self.output_dir / f"temp_spectrum_{spectrum_idx}.npy"
        np.save(temp_data_path, curve_data)
        output_prefix = f"spectrum_{spectrum_idx:04d}"
        
        # Clean up any existing visualization files for this spectrum to ensure fresh output
        for old_viz in [
            self.output_dir / f"{output_prefix}_fit.png",
            self.output_dir / "fit_visualization.png",
        ]:
            if old_viz.exists():
                try:
                    os.remove(old_viz)
                except:
                    pass
        
        script = None
        last_error = ""
        exec_result = None
        
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                if base_script is not None and attempt == 1:
                    script = self._adapt_script_for_spectrum(base_script, str(temp_data_path), output_prefix)
                elif attempt == 1:
                    script = self._generate_fitting_script(state, str(temp_data_path), stats)
                    # Check conformance with locked plan on fresh generation
                    conformance = self._check_plan_conformance(state, script)
                    if conformance and not conformance.get("conformant", True):
                        issues = "; ".join(
                            conformance.get("unjustified_deviations", [])
                        )
                        self.logger.warning(
                            "    \u26a0\ufe0f Plan conformance issue: %s", issues
                        )
                        last_error = (
                            "PLAN CONFORMANCE: Script deviates from the "
                            "locked plan without justification. Issues: "
                            f"{issues}. Plan model: "
                            f"{state.get('locked_fitting_config', {}).get('physical_model', '')}. "
                            "Either fix the script to match the plan, or if "
                            "the plan cannot work, implement the closest "
                            "viable alternative and explain why in the summary."
                        )
                        continue
                    if conformance and conformance.get("justified_deviations"):
                        self.logger.info(
                            "    \u2139\ufe0f Justified plan deviations: %s",
                            "; ".join(conformance["justified_deviations"]),
                        )
                else:
                    script = self._correct_script(state, script, last_error)

                exec_result = self.executor.execute_script(script, working_dir=str(self.output_dir))

                if exec_result.get("status") == "success":
                    # Check if the script actually produced the expected outputs
                    stdout = exec_result.get("stdout", "")
                    has_fit_results = "FIT_RESULTS_JSON:" in stdout
                    
                    # Check for visualization file
                    viz_path = self.output_dir / f"{output_prefix}_fit.png"
                    if not viz_path.exists():
                        viz_path = self.output_dir / "fit_visualization.png"
                    has_visualization = viz_path.exists()
                    
                    if has_fit_results and has_visualization:
                        # Script truly succeeded - produced expected outputs
                        break
                    else:
                        # Script ran but didn't produce expected outputs
                        missing = []
                        if not has_fit_results:
                            missing.append("FIT_RESULTS_JSON output")
                        if not has_visualization:
                            missing.append("visualization file")
                        last_error = f"Script executed but did not produce expected outputs. Missing: {', '.join(missing)}. The script must print 'FIT_RESULTS_JSON:{{...}}' with fit results and save a visualization to '{output_prefix}_fit.png'."
                        self.logger.warning(f"    ⚠️ Attempt {attempt}: Script ran but missing outputs: {', '.join(missing)}")
                        if attempt >= self.MAX_ATTEMPTS:
                            # Last attempt also failed to produce outputs
                            exec_result["status"] = "failed"
                            exec_result["message"] = last_error
                else:
                    last_error = exec_result.get("message", "Unknown error")
                    self.logger.warning(f"    ⚠️ Attempt {attempt} failed: {last_error[:100]}")
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"    ❌ Attempt {attempt} error: {e}")
        
        if temp_data_path.exists():
            try:
                os.remove(temp_data_path)
            except:
                pass
        
        if exec_result is None or exec_result.get("status") != "success":
            return {
                "index": spectrum_idx,
                "name": spectrum_name,
                "success": False,
                "error": last_error,
                "parameters": {},
                "fit_quality": {},
                "script": script
            }
        
        fit_results = {}
        for line in (exec_result.get("stdout") or "").splitlines():
            if line.startswith("FIT_RESULTS_JSON:"):
                try:
                    fit_results = json.loads(line.replace("FIT_RESULTS_JSON:", "").strip())
                except json.JSONDecodeError:
                    pass
                break
        
        viz_path = self.output_dir / f"{output_prefix}_fit.png"
        if not viz_path.exists():
            viz_path = self.output_dir / "fit_visualization.png"
        
        viz_bytes = None
        if viz_path.exists():
            with open(viz_path, "rb") as f:
                viz_bytes = f.read()
            final_viz_path = self.output_dir / f"{output_prefix}_fit.png"
            if viz_path != final_viz_path:
                viz_path.rename(final_viz_path)
            viz_path = final_viz_path
        
        return {
            "index": spectrum_idx,
            "name": spectrum_name,
            "data_path": data_path,
            "success": True,
            "error": None,
            "model_type": fit_results.get("model_type"),
            "parameters": fit_results.get("parameters", {}),
            "fit_quality": fit_results.get("fit_quality", {}),
            "summary": fit_results.get("summary"),
            "visualization_path": str(viz_path) if viz_path.exists() else None,
            "visualization_bytes": viz_bytes,
            "statistics": stats,
            "script": script
        }

    def _suggest_alternative_model(self, state: dict, current_result: dict) -> Optional[dict]:
        """Use LLM to suggest an alternative fitting model, showing it the actual poor fit."""
        config = state.get("locked_fitting_config", {})
        
        prompt_text = self.ALTERNATIVE_MODELS_PROMPT.format(
            r_squared=current_result.get("fit_quality", {}).get("r_squared", 0),
            threshold=self.r2_threshold,
            current_model=config.get("physical_model", "Unknown"),
            current_params=json.dumps(current_result.get("parameters", {}), indent=2),
            data_stats=json.dumps(current_result.get("statistics", {}), indent=2),
            analysis_approach=config.get("analysis_approach", ""),
        )
        
        # Build prompt with visualization if available
        prompt_parts = [prompt_text]
        
        # Include the actual fit visualization so LLM can see what went wrong
        if current_result.get("visualization_bytes"):
            prompt_parts.append("\n\n**CURRENT FIT VISUALIZATION (showing the poor fit):**")
            prompt_parts.append({
                "mime_type": "image/png", 
                "data": current_result["visualization_bytes"]
            })
            prompt_parts.append("\nExamine this fit carefully. Look at where the model deviates from the data and suggest a better model.")
        
        # Also include original data plot if available for comparison
        if state.get("original_plot_bytes"):
            prompt_parts.append("\n\n**ORIGINAL DATA:**")
            prompt_parts.append({
                "mime_type": "image/png",
                "data": state["original_plot_bytes"]
            })
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            if error or not result:
                return None
            return result
        except Exception as e:
            self.logger.error(f"Failed to get alternative model suggestion: {e}")
            return None

    FIT_VERIFICATION_PROMPT = '''You are a scientific data analysis expert reviewing a curve/spectral fit.

**TASK:** Examine this fit visualization and determine if the fit is acceptable for scientific use.

**FIT STATISTICS:**
- R² = {r_squared:.4f}
- Model: {model_type}
- Number of components: {n_components}

**FITTED PARAMETERS:**
{parameters}

## STEP 1: CHECK FOR BROKEN FITS (reject immediately if ANY are true)

- **Wrong x-range?** Does the plot show a completely different x-range than where the model components are defined? (e.g., plot shows 135-200 but components are at 300, 520, 860) → REJECT
- **Featureless fit?** Is R² ≈ 1.0 but the plot shows only a simple line/curve with no actual data structure being fitted? → REJECT  
- **RMSE ≈ 0 with trivial fit?** Near-zero error but no meaningful features captured suggests fitting wrong data subset → REJECT
- **Model components outside plot?** Legend shows components at positions not visible in the plotted x-range? → REJECT

If ANY box above is checked: set fit_acceptable: FALSE, explain the data range or data loading problem.

---

## STEP 2: IF STEP 1 PASSED, evaluate fit quality

**Accept if:**
- R² ≥ {accept_threshold:.2f} AND residuals are mostly random noise AND main data features are captured

**Reject if:**
- R² < {reject_threshold:.2f}
- Major systematic residual pattern across ENTIRE spectrum  
- A prominent data feature is completely missed by the model

**Do NOT reject for:**
- Ambiguous or subtle features
- Minor position offsets (<5%)
- Large parameter uncertainties (that's just uncertainty, not failure)
- "Could try different model" suggestions

---

## RESPONSE FORMAT

Return JSON:
{{
    "fit_acceptable": true/false,
    "issues_found": [
        {{
            "location": "where in the data",
            "problem": "what is wrong",
            "evidence": "what you see in the plot/residuals",
            "suggested_fix": "how to fix it"
        }}
    ],
    "spurious_components": ["list of components fitting noise, not real features"],
    "missing_features": ["list of obvious data features not captured by model"],
    "overall_assessment": "one sentence summary",
    "recommended_action": "specific fix OR 'none'"
}}


Remember: Rejecting a good fit (R² > 0.98) to chase marginal improvements often makes things WORSE through overfitting or convergence failures.
'''

    def _verify_fit_with_llm(self, state: dict, fit_result: dict, history: List[dict] = None) -> Optional[dict]:
        """
        Use LLM to verify fit quality by examining the visualization.
        Returns verification result with any issues found, or None if verification fails.
        """
        if not fit_result.get("visualization_bytes"):
            self.logger.warning("      No visualization available for LLM verification")
            return None
        
        # Gather fit info
        r_squared = fit_result.get("fit_quality", {}).get("r_squared", 0)
        model_type = fit_result.get("model_type", "Unknown")
        parameters = fit_result.get("parameters", {})
        
        # Count components
        n_components = len(parameters) if isinstance(parameters, dict) else 0
        
        # Format parameters for prompt
        params_str = json.dumps(parameters, indent=2) if parameters else "No parameters extracted"
        
        prompt_text = self.FIT_VERIFICATION_PROMPT.format(
            r_squared=r_squared,
            model_type=model_type,
            n_components=n_components,
            parameters=params_str,
            accept_threshold=self.r2_threshold,
            reject_threshold=self.r2_threshold - 0.05,
        )
        
        # Add history context
        history_context = build_verification_prompt_with_history(
            current_fit={
                "r_squared": r_squared,
                "model_type": model_type,
                "parameters": parameters,
            },
            previous_iterations=history or [],
        )

        prompt_parts = [
            prompt_text + history_context,
            "\n\n**FIT VISUALIZATION (examine carefully, especially the residual plot):**",
        ]
        
        # Add the actual fit visualization
        prompt_parts.append({
            "mime_type": "image/png", 
            "data": fit_result["visualization_bytes"]
        })
        
        # Also include original data if available for comparison
        if state.get("original_plot_bytes"):
            prompt_parts.append("\n\n**ORIGINAL DATA (for reference):**")
            prompt_parts.append({"mime_type": "image/png", "data": state["original_plot_bytes"]})

        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"      LLM verification parse failed: {error}")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"      LLM verification failed: {e}")
            return None

    def _apply_llm_verification_feedback(self, state: dict, verification: dict) -> dict:
        """
        Apply LLM verification feedback to refine the fitting configuration.
        Returns updated config.
        """
        config = state.get("locked_fitting_config", {}).copy()
        
        recommended_action = verification.get("recommended_action", "")
        if not recommended_action or recommended_action.lower() == "none":
            return config
        
        # Build a refinement prompt based on verification results
        issues_summary = []
        for issue in verification.get("issues_found", []):
            issues_summary.append(f"- {issue.get('location', 'Unknown')}: {issue.get('problem', '')} -> {issue.get('suggested_fix', '')}")
        
        spurious = verification.get("spurious_components", [])
        missing = verification.get("missing_features", [])
        
        refinement_prompt = f"""Refine the fitting approach based on automated verification feedback.

**CURRENT APPROACH:**
- Model: {config.get('physical_model', 'Unknown')}
- Strategy: {config.get('fitting_strategy', 'Unknown')}

**VERIFICATION FINDINGS:**
{chr(10).join(issues_summary) if issues_summary else 'No specific issues listed'}

**SPURIOUS COMPONENTS TO REMOVE:** {', '.join(spurious) if spurious else 'None identified'}

**MISSING FEATURES TO ADD:** {', '.join(missing) if missing else 'None identified'}

**RECOMMENDED ACTION:** {recommended_action}

Return JSON with the refined fitting approach:
{{
    "physical_model": "updated model description incorporating the fixes",
    "fitting_strategy": "updated fitting strategy",
    "parameters_to_extract": ["list", "of", "parameters"],
    "analysis_approach": "updated approach"
}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[refinement_prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"      Could not parse refinement: {error}")
                return config
            
            # Update config with refinements
            config.update(result)
            return config
            
        except Exception as e:
            self.logger.error(f"      Refinement failed: {e}")
            return config

    def _get_human_feedback_for_poor_fit(self, state: dict, best_result: dict, all_attempts: List[dict]) -> Optional[dict]:
        models_tried = "\n".join([f"  - {a['model']}: R² = {a['r2']:.4f}" for a in all_attempts])
        
        print("")
        print("=" * 60)
        print("⚠️  FIT QUALITY BELOW THRESHOLD")
        print("=" * 60)
        
        if best_result.get("visualization_bytes"):
            viz_path = self.output_dir / "quality_review_fit.png"
            with open(viz_path, 'wb') as f:
                f.write(best_result["visualization_bytes"])
            print(f"[Best fit visualization saved to: {viz_path}]")
        
        prompt = self.HUMAN_FEEDBACK_PROMPT.format(
            best_r2=best_result.get("fit_quality", {}).get("r_squared", 0),
            threshold=self.r2_threshold,
            models_tried=models_tried,
            example_threshold=self.r2_threshold - 0.05,
        )
        print(prompt)
        
        feedback = input("\nYour input: ").strip()
        
        if not feedback:
            print("No feedback provided. Proceeding with best available fit.")
            return None
        
        if "accept" in feedback.lower() or "proceed" in feedback.lower():
            print("✓ Accepting best available fit.")
            return None
        
        if "threshold" in feedback.lower():
            try:
                match = re.search(r'(\d+\.?\d*)', feedback)
                if match:
                    new_threshold = float(match.group(1))
                    if new_threshold <= 1.0:
                        print(f"✓ Adjusting threshold to {new_threshold}")
                        return {"action": "adjust_threshold", "new_threshold": new_threshold}
            except:
                pass
        
        print("🔄 Will retry with your suggested approach...")
        return {"action": "retry", "feedback": feedback}

    def _get_user_feedback_on_fit(self, state: dict, fit_result: dict, r2: float) -> Optional[str]:
        """
        Show user the first spectrum fit and ask for optional feedback.
        Returns feedback string if user wants changes, None if satisfied.
        """
        print("\n" + "=" * 70)
        print("📊 FIRST SPECTRUM FIT RESULT - Review Before Processing Series")
        print("=" * 70)
        
        # Save and display fit visualization path
        review_viz_path = None
        if fit_result.get("visualization_bytes"):
            review_viz_path = self.output_dir / "first_spectrum_fit_review.png"
            with open(review_viz_path, 'wb') as f:
                f.write(fit_result["visualization_bytes"])
            print(f"\n[Fit visualization saved to: {review_viz_path}]")
        
        # Show fit summary
        print(f"\n📈 Model: {fit_result.get('model_type', 'N/A')}")
        print(f"📊 R² = {r2:.4f} (threshold: {self.r2_threshold})")
        
        params = fit_result.get("parameters", {})
        if params:
            print("\n📋 Fitted Parameters:")
            for comp, values in params.items():
                if isinstance(values, dict):
                    print(f"   {comp}:")
                    for k, v in values.items():
                        if not k.endswith('_err'):
                            if isinstance(v, float):
                                print(f"      {k}: {v:.4g}")
                            else:
                                print(f"      {k}: {v}")
        
        num_spectra = state.get("num_spectra", 1)
        print(f"\n⚠️  This fitting model will be applied to all {num_spectra} spectra in the series.")
        print("\n" + "-" * 60)
        print("Options:")
        print("  • Press Enter to accept this fit and proceed with series")
        print("  • Type feedback to modify the fitting approach (e.g., 'add baseline', ")
        print("    'use Voigt instead of Gaussian', 'fit two peaks instead of one')")
        print("-" * 60)
        
        feedback = input("\n🤔 Your feedback (or Enter to accept): ").strip()
        
        # Clean up the review file - it's only for user viewing during this step
        if review_viz_path and review_viz_path.exists():
            try:
                os.remove(review_viz_path)
            except:
                pass
        
        if not feedback:
            print("✅ Fit accepted. Proceeding with series...")
            return None
        
        return feedback

    def _ask_keep_user_guided_fit(self, user_r2: float, original_r2: float) -> bool:
        """Ask user whether to keep the user-guided fit even if R² is worse."""
        print("\n" + "-" * 60)
        print(f"⚠️  User-guided fit has lower R² ({user_r2:.4f}) than original ({original_r2:.4f})")
        print("-" * 60)
        print("Options:")
        print(f"  • Type 'keep' to use the user-guided fit anyway (R² = {user_r2:.4f})")
        print(f"  • Press Enter to revert to original fit (R² = {original_r2:.4f})")
        
        response = input("\nYour choice: ").strip().lower()
        
        if response == 'keep':
            print("✅ Keeping user-guided fit.")
            return True
        else:
            print("✅ Reverting to original fit.")
            return False

    def _refine_model_from_feedback(self, state: dict, feedback: str) -> dict:
        config = state.get("locked_fitting_config", {})
        prompt = f"""Refine the fitting approach based on user feedback.

**Current Approach:**
- Model: {config.get('physical_model', 'Unknown')}
- Strategy: {config.get('fitting_strategy', 'Unknown')}

**User Feedback:** {feedback}

Return JSON with:
{{
    "physical_model": "updated model description",
    "fitting_strategy": "updated fitting strategy",
    "parameters_to_extract": ["list", "of", "parameters"],
    "analysis_approach": "updated approach"
}}
"""
        
        try:
            response = self.model.generate_content(
                contents=[prompt],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            if error or not result:
                return config
            updated = config.copy()
            updated.update(result)
            return updated
        except Exception as e:
            self.logger.error(f"Failed to refine model from feedback: {e}")
            return config

    def _fit_with_quality_control(self, state: dict, curve_data: np.ndarray, data_path: str, spectrum_name: str, spectrum_idx: int, is_regime_anchor: bool = False) -> dict:
        """
        Fit a single spectrum with quality control, verification, and optional judge selection.

        Flow:
        1. Initial fit attempt
        2. For anchor spectrum (first in series or first in regime): LLM verification loop
        - Each iteration: verify current fit -> if issues, refit
        - After loop: verify final refit
        - If still not approved: call judge to select best
        3. If still below R² threshold: try alternative models
        4. If human feedback enabled: allow user to guide refinement
        """
        all_attempts = []
        best_result = None
        best_r2 = -1.0

        # Anchor = first spectrum overall OR first in a regime; gets full QC
        _is_anchor = spectrum_idx == 0 or is_regime_anchor

        # --- Initial fit ---
        initial_model = state.get('locked_fitting_config', {}).get('physical_model', 'Initial model')
        self.logger.info(f"   Attempt 1: {initial_model[:80]}...")

        result = self._fit_single_spectrum(
            state=state, curve_data=curve_data, data_path=data_path,
            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
        )

        if result["success"]:
            r2 = result.get("fit_quality", {}).get("r_squared", 0)
            all_attempts.append({"model": initial_model, "r2": r2, "result": result})

            if r2 > best_r2:
                best_r2 = r2
                best_result = result

            # Track if user explicitly accepted
            user_accepted_fit = False

            # --- Verification loop (for anchor spectra: first overall or first in regime) ---
            if _is_anchor:
                if not best_result or not best_result.get("success") or best_r2 < 0.1:
                    self.logger.warning(f"   Initial fit failed or R² too low ({best_r2:.4f}), skipping verification")
                else:
                    # Track all verification attempts for potential judge review
                    verification_attempts = []
                    verification_history = [] # track history for context
                    fit_was_approved = False

                    for verification_iter in range(self.max_verification_iterations):
                        self.logger.info(f"   Verification {verification_iter + 1}/{self.max_verification_iterations}...")
                        
                        verification = self._verify_fit_with_llm(state, best_result, history=verification_history)  # Pass history
                        
                        if verification is None:
                            self.logger.warning(f"   Verification failed, skipping")
                            break
                        
                        # Store this attempt for potential judge review
                        verification_attempts.append({
                            "result": best_result.copy() if best_result else {},
                            "verification": verification,
                            "config": state.get("locked_fitting_config", {}).copy(),
                            "r2": best_r2
                        })
                        
                        # Store in history for next iteration's context
                        verification_history.append({
                            "r_squared": best_r2,
                            "config_used": state.get("locked_fitting_config", {}),
                            "issues_found": verification.get("issues_found", []),
                            "overall_assessment": verification.get("overall_assessment", ""),
                            "recommended_action": verification.get("recommended_action", ""),
                        })

                        if verification.get("fit_acceptable", True):
                            self.logger.info(f"   ✅ Fit approved (R² = {best_r2:.4f})")
                            fit_was_approved = True
                            break
                        
                        # Log issues
                        self._log_verification_issues(verification)
                        
                        # Apply LLM's recommended fixes
                        refined_config = self._apply_llm_verification_feedback(state, verification)
                        
                        if refined_config == state.get("locked_fitting_config", {}):
                            self.logger.info(f"   No config changes suggested, stopping verification")
                            break
                        
                        # Clean up old visualization
                        old_viz_path = best_result.get("visualization_path")
                        if old_viz_path and Path(old_viz_path).exists():
                            try:
                                os.remove(old_viz_path)
                            except:
                                pass
                        
                        state["locked_fitting_config"] = refined_config
                        
                        # Refit with refined config
                        self.logger.info(f"   Refitting with verification feedback...")
                        verified_result = self._fit_single_spectrum(
                            state=state, curve_data=curve_data, data_path=data_path,
                            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                        )
                        
                        if verified_result["success"]:
                            verified_r2 = verified_result.get("fit_quality", {}).get("r_squared", 0)
                            self.logger.info(f"   Refit R² = {verified_r2:.4f} (was {best_r2:.4f})")
                            
                            # Update best result for next iteration
                            best_r2 = verified_r2
                            best_result = verified_result
                            all_attempts.append({
                                "model": f"Verification-{verification_iter + 1}", 
                                "r2": verified_r2, 
                                "result": verified_result
                            })
                        else:
                            self.logger.warning(f"   Refit failed, stopping verification")
                            break
                    
                    else:
                        # Loop exhausted without approval - verify the final refit before calling judge
                        self.logger.info(f"   Verifying final refit...")
                        final_verification = self._verify_fit_with_llm(state, best_result)
                        
                        if final_verification:
                            # Store the final attempt
                            verification_attempts.append({
                                "result": best_result.copy() if best_result else {},
                                "verification": final_verification,
                                "config": state.get("locked_fitting_config", {}).copy(),
                                "r2": best_r2
                            })
                            
                            if final_verification.get("fit_acceptable", True):
                                self.logger.info(f"   ✅ Final fit approved (R² = {best_r2:.4f})")
                                fit_was_approved = True
                            else:
                                # Log issues for the final attempt too
                                self._log_verification_issues(final_verification)
                        
                        # Only call judge if still not approved
                        if not fit_was_approved and len(verification_attempts) > 1:
                            judge_result = self._judge_select_best_fit(verification_attempts)
                            
                            selected_index = judge_result.get("selected_index")
                            is_acceptable = judge_result.get("acceptable", False)
                            
                            if selected_index is not None:
                                # Judge selected a best attempt - use it regardless of acceptable flag
                                idx = selected_index
                                selected_attempt = verification_attempts[idx]
                                best_result = selected_attempt["result"]
                                best_r2 = selected_attempt["r2"]
                                state["locked_fitting_config"] = selected_attempt["config"]
                                
                                if is_acceptable:
                                    # Judge approved the fit
                                    if judge_result.get("issues_with_selected"):
                                        best_result["judge_note"] = judge_result["issues_with_selected"]
                                    self.logger.info(f"   ✅ Using judge-selected fit (Attempt {idx + 1}, R² = {best_r2:.4f})")
                                else:
                                    # Judge selected best available but flagged it as below standards
                                    best_result["judge_warning"] = (
                                        f"Judge selected this as best available (R² = {best_r2:.4f}) "
                                        f"but noted it does not meet acceptance criteria. "
                                        f"Reason: {judge_result.get('reasoning', 'No reason provided')[:200]}"
                                    )
                                    self.logger.warning(
                                        f"   ⚠️ Using judge-selected fit (Attempt {idx + 1}, R² = {best_r2:.4f}) "
                                        f"despite not meeting acceptance criteria"
                                    )
                            else:
                                # Judge couldn't select any attempt (selected_index is None)
                                best_result["judge_warning"] = (
                                    f"Judge could not select any acceptable fit. "
                                    f"Reason: {judge_result.get('reasoning', 'No reason provided')[:200]}"
                                )
                                self.logger.warning(f"   ⚠️ Judge could not select any fit - keeping current best (R² = {best_r2:.4f})")
                    
                    # Human feedback opportunity (if enabled and we have a fit to show)
                    user_accepted_fit = False  # Track if user explicitly accepted
                    if self.enable_human_feedback and best_result and best_result.get("visualization_bytes"):
                        user_feedback = self._get_user_feedback_on_fit(state, best_result, best_r2)
                        
                        if user_feedback:
                            best_result, best_r2 = self._apply_user_feedback(
                                state, user_feedback, best_result, best_r2,
                                curve_data, data_path, spectrum_name, spectrum_idx, all_attempts
                            )
                        else:
                            # User pressed Enter without feedback = explicit acceptance
                            user_accepted_fit = True
            
            # --- Check if we meet threshold ---
            if best_r2 >= self.r2_threshold:
                self.logger.info(f"✅ R² = {best_r2:.4f} (meets threshold {self.r2_threshold})")
                return best_result
            else:
                self.logger.warning(f"⚠️ R² = {best_r2:.4f} (below threshold {self.r2_threshold})")
                
                # If user explicitly accepted, skip alternative model retries
                if user_accepted_fit:
                    self.logger.info(f"   User accepted fit - skipping alternative model attempts")
                    best_result["user_accepted"] = True
                    best_result["quality_warning"] = f"R² = {best_r2:.4f} below threshold {self.r2_threshold} (user accepted)"
                    return best_result
        else:
            self.logger.error(f"   Initial fit failed: {result.get('error', 'Unknown')[:50]}")
            all_attempts.append({"model": initial_model, "r2": 0, "result": result})
            user_accepted_fit = False
        
        # --- Alternative model retries ---
        current_config = state.get("locked_fitting_config", {}).copy()
        
        for retry in range(self.max_model_retries):
            self.logger.info(f"   Alternative model {retry + 1}/{self.max_model_retries}...")
            
            alternative = self._suggest_alternative_model(state, best_result or result)
            
            if not alternative:
                self.logger.warning("   Could not generate alternative model suggestion")
                break
            
            self.logger.info(f"   Diagnosis: {alternative.get('diagnosis', 'N/A')[:80]}")
            self.logger.info(f"   Trying: {alternative.get('alternative_model', 'N/A')[:60]}")
            
            temp_config = current_config.copy()
            temp_config["physical_model"] = alternative.get("alternative_model", temp_config.get("physical_model"))
            temp_config["fitting_strategy"] = alternative.get("fitting_strategy", temp_config.get("fitting_strategy"))
            temp_config["parameters_to_extract"] = alternative.get("parameters_to_extract", temp_config.get("parameters_to_extract", []))
            
            original_config = state.get("locked_fitting_config")
            state["locked_fitting_config"] = temp_config
            
            alt_result = self._fit_single_spectrum(
                state=state, curve_data=curve_data, data_path=data_path,
                spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
            )
            
            state["locked_fitting_config"] = original_config
            
            if alt_result["success"]:
                alt_r2 = alt_result.get("fit_quality", {}).get("r_squared", 0)
                all_attempts.append({
                    "model": alternative.get("alternative_model", f"Alternative {retry + 1}"),
                    "r2": alt_r2, "result": alt_result, "config": temp_config
                })
                
                if alt_r2 > best_r2:
                    best_r2 = alt_r2
                    best_result = alt_result
                    best_result["_winning_config"] = temp_config
                
                if alt_r2 >= self.r2_threshold:
                    self.logger.info(f"✅ R² = {alt_r2:.4f} (meets threshold with alternative model)")
                    if _is_anchor:
                        state["locked_fitting_config"] = temp_config

                        # Offer human review for alternative model before locking
                        if self.enable_human_feedback and alt_result.get("visualization_bytes"):
                            user_feedback = self._get_user_feedback_on_fit(state, alt_result, alt_r2)
                            if user_feedback:
                                alt_result, alt_r2 = self._apply_user_feedback(
                                    state, user_feedback, alt_result, alt_r2,
                                    curve_data, data_path, spectrum_name, spectrum_idx, all_attempts
                                )
                    
                    return alt_result
                else:
                    self.logger.warning(f"   R² = {alt_r2:.4f} (still below threshold)")
            else:
                self.logger.warning(f"   Alternative model failed: {alt_result.get('error', 'Unknown')[:50]}")
                all_attempts.append({
                    "model": alternative.get("alternative_model", f"Alternative {retry + 1}"),
                    "r2": 0, "result": alt_result
                })
        
        # --- Human feedback for poor fit (if enabled) ---
        if self.enable_human_feedback and _is_anchor:
            feedback_result = self._get_human_feedback_for_poor_fit(state, best_result, all_attempts)

            if feedback_result:
                if feedback_result.get("action") == "adjust_threshold":
                    self.r2_threshold = feedback_result["new_threshold"]
                    if best_r2 >= self.r2_threshold:
                        self.logger.info(f"✅ Best fit now meets adjusted threshold")
                        return best_result

                elif feedback_result.get("action") == "retry":
                    refined_config = self._refine_model_from_feedback(state, feedback_result["feedback"])
                    original_config = state.get("locked_fitting_config")
                    state["locked_fitting_config"] = refined_config

                    human_guided_result = self._fit_single_spectrum(
                        state=state, curve_data=curve_data, data_path=data_path,
                        spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
                    )

                    if human_guided_result["success"]:
                        human_r2 = human_guided_result.get("fit_quality", {}).get("r_squared", 0)
                        self.logger.info(f"   Human-guided fit: R² = {human_r2:.4f}")

                        if human_r2 > best_r2:
                            best_r2 = human_r2
                            best_result = human_guided_result
                            if _is_anchor:
                                state["locked_fitting_config"] = refined_config
                        else:
                            state["locked_fitting_config"] = original_config
                    else:
                        state["locked_fitting_config"] = original_config

        # --- Return best available result ---
        if best_result:
            best_result["quality_warning"] = f"R² = {best_r2:.4f} below threshold {self.r2_threshold}"
            best_result["attempted_models"] = [a["model"] for a in all_attempts]
            self.logger.warning(f"⚠️ Proceeding with best available fit (R² = {best_r2:.4f})")

            if _is_anchor and best_result.get("_winning_config"):
                state["locked_fitting_config"] = best_result["_winning_config"]
            
            return best_result
        else:
            return {
                "index": spectrum_idx, "name": spectrum_name, "success": False,
                "error": "All fitting attempts failed", "attempts": len(all_attempts),
                "parameters": {}, "fit_quality": {},
            }

    def _log_verification_issues(self, verification: dict) -> None:
        """Log verification issues in a readable format."""
        issues_count = len(verification.get("issues_found", []))
        overall_assessment = verification.get('overall_assessment', 'No assessment provided')

        self.logger.info(f"   ⚠️ Found {issues_count} issue(s)")
        self.logger.info("")
        self.logger.info(f"   Assessment:")
        for line in self._wrap_text(overall_assessment, width=70):
            self.logger.info(f"      {line}")

        if verification.get("issues_found"):
            self.logger.info("")
            self.logger.info(f"   Issues:")
            
            for i, issue in enumerate(verification.get("issues_found", []), 1):
                location = issue.get('location', 'Unknown')
                problem = issue.get('problem', 'No description')
                suggested_fix = issue.get('suggested_fix', '')
                
                self.logger.info("")
                self.logger.info(f"   [{i}] {location}")

                # Wrap problem text
                problem_lines = self._wrap_text(problem, width=65)
                self.logger.info(f"       Problem: {problem_lines[0]}")
                for line in problem_lines[1:]:
                    self.logger.info(f"                {line}")

                # Wrap fix text
                if suggested_fix:
                    fix_lines = self._wrap_text(suggested_fix, width=65)
                    self.logger.info(f"       Fix: {fix_lines[0]}")
                    for line in fix_lines[1:]:
                        self.logger.info(f"            {line}")

        recommended = verification.get("recommended_action", "")
        if recommended and recommended.lower() != "none":
            self.logger.info("")
            self.logger.info(f"   Recommended action:")
            for line in self._wrap_text(recommended, width=65):
                self.logger.info(f"      {line}")
        
        self.logger.info("")

    def _apply_user_feedback(
        self, 
        state: dict, 
        user_feedback: str, 
        best_result: dict, 
        best_r2: float,
        curve_data: np.ndarray, 
        data_path: str, 
        spectrum_name: str, 
        spectrum_idx: int,
        all_attempts: list
    ) -> tuple:
        """
        Apply user feedback to refine the fit.
        
        Returns:
            Tuple of (best_result, best_r2) after applying feedback
        """
        refined_config = self._refine_model_from_feedback(state, user_feedback)
        original_config = state.get("locked_fitting_config")
        state["locked_fitting_config"] = refined_config
        
        # Clean up old visualization
        old_viz_path = best_result.get("visualization_path")
        if old_viz_path and Path(old_viz_path).exists():
            try:
                os.remove(old_viz_path)
            except:
                pass
        
        self.logger.info(f"   Refitting with user feedback...")
        user_guided_result = self._fit_single_spectrum(
            state=state, curve_data=curve_data, data_path=data_path,
            spectrum_name=spectrum_name, spectrum_idx=spectrum_idx, base_script=None
        )
        
        if user_guided_result["success"]:
            user_r2 = user_guided_result.get("fit_quality", {}).get("r_squared", 0)
            self.logger.info(f"   User-guided fit: R² = {user_r2:.4f}")
            all_attempts.append({"model": "User-guided", "r2": user_r2, "result": user_guided_result})
            
            if user_r2 > best_r2:
                return user_guided_result, user_r2
            else:
                # Save the new fit as a review image so the UI can display it
                if user_guided_result.get("visualization_bytes"):
                    review_viz_path = self.output_dir / "first_spectrum_fit_review.png"
                    with open(review_viz_path, 'wb') as f:
                        f.write(user_guided_result["visualization_bytes"])
                # User-guided was worse - ask what to do
                keep_user = self._ask_keep_user_guided_fit(user_r2, best_r2)
                if keep_user:
                    return user_guided_result, user_r2
                else:
                    state["locked_fitting_config"] = original_config
                    return best_result, best_r2
        else:
            self.logger.warning(f"   User-guided fit failed, keeping previous")
            state["locked_fitting_config"] = original_config
            return best_result, best_r2

    def _detect_outliers(self, series_results: List[dict]) -> List[dict]:
        r2_values = []
        for r in series_results:
            if r["success"]:
                r2 = r.get("fit_quality", {}).get("r_squared")
                if r2 is not None:
                    r2_values.append(r2)
        
        if len(r2_values) < 3:
            return []
        
        r2_array = np.array(r2_values)
        mean_r2 = np.mean(r2_array)
        std_r2 = np.std(r2_array)
        
        flagged = []
        
        for r in series_results:
            if not r["success"]:
                flagged.append({
                    "index": r["index"], "name": r["name"], "reason": "fit_failed",
                    "r_squared": None, "series_mean": float(mean_r2), "series_std": float(std_r2),
                    "deviation_sigma": None,
                    "recommendation": "Check data quality and consider manual inspection. The fitting script failed to execute successfully."
                })
                continue
            
            r2 = r.get("fit_quality", {}).get("r_squared")
            if r2 is None:
                continue
            
            below_threshold = r2 < self.r2_threshold
            
            if std_r2 > 0.001:
                deviation_sigma = (mean_r2 - r2) / std_r2
                is_outlier = deviation_sigma > self.outlier_sigma
            else:
                deviation_sigma = 0
                is_outlier = False
            
            if below_threshold or is_outlier:
                if is_outlier and not below_threshold:
                    reason = "statistical_outlier"
                    recommendation = "Fit quality significantly worse than series average. Possible causes: phase transition, sample change, or instrument artifact. Consider detailed inspection - may indicate interesting physics."
                elif below_threshold and not is_outlier:
                    reason = "below_threshold"
                    recommendation = "Fit quality below threshold but consistent with series. The chosen model may not be optimal for this data type."
                else:
                    reason = "outlier_and_below_threshold"
                    recommendation = "Significant fit quality issue. This spectrum behaves differently from others in the series. Strongly recommend manual review - could indicate interesting physics, phase transition, or data quality issue."
                
                flagged.append({
                    "index": r["index"], "name": r["name"], "reason": reason,
                    "r_squared": float(r2), "series_mean": float(mean_r2), "series_std": float(std_r2),
                    "deviation_sigma": float(deviation_sigma) if deviation_sigma else None,
                    "recommendation": recommendation
                })
        
        return flagged

    def _generate_outlier_report(self, flagged: List[dict], series_results: List[dict]) -> str:
        if not flagged:
            return ""
        
        lines = ["", "=" * 60, "⚠️  FLAGGED SPECTRA - REQUIRE ATTENTION", "=" * 60, ""]
        
        total = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        r2_values = [r.get("fit_quality", {}).get("r_squared", 0) for r in series_results if r["success"]]
        
        if r2_values:
            lines.append(f"Series statistics: {successful}/{total} successful fits")
            lines.append(f"R² range: {min(r2_values):.4f} - {max(r2_values):.4f}")
            lines.append(f"R² mean ± std: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
            lines.append(f"Quality threshold: {self.r2_threshold}")
            lines.append(f"Outlier detection: {self.outlier_sigma}σ below mean")
            lines.append("")
        
        by_reason = {}
        for f in flagged:
            reason = f["reason"]
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(f)
        
        reason_labels = {
            "fit_failed": "❌ Failed Fits",
            "statistical_outlier": "📊 Statistical Outliers (possible interesting physics)",
            "below_threshold": "⚠️ Below Threshold",
            "outlier_and_below_threshold": "🔴 Critical: Outlier + Below Threshold"
        }
        
        for reason, items in by_reason.items():
            lines.append(f"\n{reason_labels.get(reason, reason)} ({len(items)} spectra):")
            lines.append("-" * 50)
            
            for f in items:
                lines.append(f"  • {f['name']} (index {f['index']})")
                if f["r_squared"] is not None:
                    lines.append(f"    R² = {f['r_squared']:.4f} (series mean: {f['series_mean']:.4f})")
                    if f["deviation_sigma"] is not None:
                        lines.append(f"    Deviation: {f['deviation_sigma']:.1f}σ below mean")
                lines.append(f"    → {f['recommendation']}")
                lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)

    def _get_config_for_spectrum(self, state: dict, idx: int) -> dict:
        """Return the fitting config for a given spectrum index.

        If regime_configs is present, return the regime-specific config.
        Otherwise, return the single locked_fitting_config (backward compatible).
        """
        regime_configs = state.get("regime_configs")
        if regime_configs and idx in regime_configs:
            return regime_configs[idx]
        return state.get("locked_fitting_config", {})

    def _get_regime_for_spectrum(self, state: dict, idx: int) -> Optional[str]:
        """Return the regime name for a given spectrum index, or None."""
        series_plan = state.get("series_analysis_plan")
        if not series_plan:
            return None
        for regime in series_plan.get("regimes", []):
            if idx in regime.get("spectrum_indices", []):
                return regime.get("name", "unnamed")
        return None

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        mode_str = "SINGLE SPECTRUM" if is_single else f"SERIES ({num_spectra} spectra)"
        self.logger.info("")
        self.logger.info(f"⚙️ FITTING: {mode_str}")
        self.logger.info(f"   R² threshold: {self.r2_threshold}")
        self.logger.info(f"   Max model retries: {self.max_model_retries}")
        if not is_single:
            self.logger.info(f"   Outlier detection: {self.outlier_sigma}σ")
        
        spectrum_paths = state.get("spectrum_paths", [])
        spectrum_stack = state.get("spectrum_stack")

        # Determine regime structure for per-regime execution
        series_plan = state.get("series_analysis_plan")
        regime_configs = state.get("regime_configs")

        if series_plan and regime_configs:
            first_in_regime: set = set()
            for regime in series_plan.get("regimes", []):
                indices = sorted(regime.get("spectrum_indices", []))
                if indices:
                    first_in_regime.add(indices[0])
            self.logger.info(f"   Regimes: {len(series_plan.get('regimes', []))}")
            self.logger.info(
                f"   First-in-regime spectra (full QC): {sorted(first_in_regime)}"
            )
        else:
            first_in_regime = {0}  # Only spectrum 0 needs full QC

        series_results = []
        base_scripts: Dict[str, str] = {}  # keyed by regime name
        locked_preprocessing_strategy = None
        original_locked_config = state.get("locked_fitting_config", {})
        if original_locked_config:
            original_locked_config = original_locked_config.copy()

        for idx in range(num_spectra):
            if spectrum_stack is not None:
                curve_data = spectrum_stack[idx]
                spectrum_name = f"spectrum_{idx:04d}"
                data_path = f"stack_index_{idx}"
            else:
                data_path = spectrum_paths[idx]
                spectrum_name = Path(data_path).stem
                curve_data = self._load_curve_data(data_path)

            # Apply preprocessing with locking support for series consistency
            if self.preprocessor is not None:
                try:
                    if idx == 0 and state.get("first_spectrum_preprocessed"):
                        # Reuse preprocessing from analyze() — avoid redundant LLM call
                        curve_data = state.get("curve_data", curve_data)
                        preprocess_quality = state.get(
                            "first_spectrum_preprocess_quality", {}
                        ) or {}
                        locked_preprocessing_strategy = preprocess_quality.get("strategy")
                        if locked_preprocessing_strategy:
                            state["locked_preprocessing_strategy"] = locked_preprocessing_strategy
                            self.logger.info(
                                "📝 Preprocessing strategy locked (from planning stage): "
                                f"{locked_preprocessing_strategy.get('reasoning', 'N/A')[:60]}"
                            )
                        else:
                            self.logger.info("Preprocessed (reused from planning stage)")
                    elif idx == 0:
                        curve_data, preprocess_quality = self.preprocessor.run_preprocessing(
                            curve_data, state.get("system_info", {})
                        )
                        locked_preprocessing_strategy = preprocess_quality.get("strategy")
                        if locked_preprocessing_strategy:
                            state["locked_preprocessing_strategy"] = locked_preprocessing_strategy
                            self.logger.info(f"📝 Preprocessing strategy locked: {locked_preprocessing_strategy.get('reasoning', 'N/A')[:60]}")
                        else:
                            self.logger.info(f"Preprocessed (no lockable strategy returned)")
                    else:
                        curve_data, _ = self.preprocessor.run_preprocessing(
                            curve_data,
                            state.get("system_info", {}),
                            locked_strategy=locked_preprocessing_strategy
                        )
                    self.logger.info(f"Preprocessed: {spectrum_name}")
                except Exception as e:
                    self.logger.warning(f"Preprocessing failed for {spectrum_name}: {e}, using raw data")

            # Determine regime and set appropriate config
            regime_name = self._get_regime_for_spectrum(state, idx) or "default"
            spectrum_config = self._get_config_for_spectrum(state, idx)

            # Temporarily set the config for this spectrum
            state["locked_fitting_config"] = spectrum_config

            if is_single:
                self.logger.info(f"Fitting: {spectrum_name}")
            elif regime_configs:
                self.logger.info(
                    f"[{idx + 1}/{num_spectra}] Fitting: {spectrum_name} "
                    f"(regime: {regime_name})"
                )
            else:
                self.logger.info(f"[{idx + 1}/{num_spectra}] Fitting: {spectrum_name}")

            if idx in first_in_regime:
                if regime_configs and idx != 0:
                    self.logger.info(
                        f"  First in regime '{regime_name}' — full quality control"
                    )

                # For regime anchors that aren't spectrum 0, temporarily swap
                # original_plot_bytes and data_statistics so the verification
                # and script generation steps reference the correct spectrum.
                _saved_original_plot = None
                _saved_data_statistics = None
                if idx != 0 and idx in first_in_regime:
                    _saved_original_plot = state.get("original_plot_bytes")
                    _saved_data_statistics = state.get("data_statistics")
                    try:
                        anchor_plot = self.plot_fn(
                            curve_data, state.get("system_info", {})
                        )
                        state["original_plot_bytes"] = anchor_plot
                    except Exception:
                        pass
                    state["data_statistics"] = self._compute_statistics(curve_data)

                result = self._fit_with_quality_control(
                    state=state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=spectrum_name, spectrum_idx=idx,
                    is_regime_anchor=(idx != 0 and idx in first_in_regime),
                )

                # Restore original state
                if _saved_original_plot is not None:
                    state["original_plot_bytes"] = _saved_original_plot
                if _saved_data_statistics is not None:
                    state["data_statistics"] = _saved_data_statistics

                if result["success"] and result.get("script"):
                    base_scripts[regime_name] = result["script"]
                    if idx == 0:
                        state["base_fitting_script"] = result["script"]
                    self.logger.info(
                        f"📝 Base fitting script locked for regime '{regime_name}'."
                    )

                    # If QC changed the config, propagate to all spectra in this regime
                    updated_config = state.get("locked_fitting_config", spectrum_config)
                    if regime_configs and updated_config != spectrum_config:
                        for r_regime in (series_plan or {}).get("regimes", []):
                            if idx in r_regime.get("spectrum_indices", []):
                                for other_idx in r_regime["spectrum_indices"]:
                                    regime_configs[other_idx] = updated_config
                                break
            else:
                base_script = base_scripts.get(regime_name)
                result = self._fit_single_spectrum(
                    state=state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=spectrum_name, spectrum_idx=idx,
                    base_script=base_script,
                )

            # Tag result with regime info
            if regime_configs:
                result["regime"] = regime_name

            series_results.append(result)

            if result["success"]:
                r2 = result.get("fit_quality", {}).get("r_squared")
                r2_str = f"R²: {r2:.4f}" if r2 else "R²: N/A"
                self.logger.info(f"✅ {result.get('model_type', 'Fit')} - {r2_str}")
            else:
                self.logger.error(f"❌ Failed: {result.get('error', 'Unknown')[:50]}")

        # Restore original locked config
        if original_locked_config:
            state["locked_fitting_config"] = original_locked_config
        
        flagged_spectra = []
        if num_spectra > 1:
            flagged_spectra = self._detect_outliers(series_results)
            
            if flagged_spectra:
                report = self._generate_outlier_report(flagged_spectra, series_results)
                self.logger.warning(report)
                
                flagged_indices = {f["index"] for f in flagged_spectra}
                for r in series_results:
                    if r["index"] in flagged_indices:
                        flag_info = next(f for f in flagged_spectra if f["index"] == r["index"])
                        r["flagged"] = True
                        r["flag_reason"] = flag_info["reason"]
                        r["flag_recommendation"] = flag_info["recommendation"]
                        r["deviation_sigma"] = flag_info.get("deviation_sigma")
                
                flagged_report_path = self.output_dir / "flagged_spectra.json"
                with open(flagged_report_path, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "r2_threshold": self.r2_threshold,
                        "outlier_sigma": self.outlier_sigma,
                        "total_spectra": num_spectra,
                        "flagged_count": len(flagged_spectra),
                        "flagged_spectra": flagged_spectra
                    }, f, indent=2)
                
                state["flagged_spectra_path"] = str(flagged_report_path)
        
        state["series_results"] = series_results
        state["flagged_spectra"] = flagged_spectra
        
        if is_single and series_results and series_results[0]["success"]:
            first_result = series_results[0]
            state["fit_results"] = {
                "model_type": first_result.get("model_type"),
                "parameters": first_result.get("parameters", {}),
                "fit_quality": first_result.get("fit_quality", {}),
                "summary": first_result.get("summary"),
            }
            state["final_script"] = first_result.get("script")
            state["final_plot_bytes"] = first_result.get("visualization_bytes")
            
            if first_result.get("visualization_bytes"):
                state["analysis_images"].append({
                    "label": first_result.get("model_type", "Fit"),
                    "data": first_result["visualization_bytes"],
                })
        
        successful = sum(1 for r in series_results if r["success"])
        flagged_count = len(flagged_spectra)
        
        self.logger.info("")
        self.logger.info(f"✅ Fitting complete: {successful}/{num_spectra} successful")
        if flagged_count > 0:
            self.logger.warning(f"⚠️ {flagged_count} spectra flagged for review")
        
        results_path = self.output_dir / "series_fit_results.json"
        with open(results_path, 'w') as f:
            serializable_results = []
            for r in series_results:
                r_copy = {k: v for k, v in r.items() if k not in ("visualization_bytes", "_winning_config")}
                serializable_results.append(r_copy)
            
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_spectra": num_spectra,
                "successful": successful,
                "flagged_count": flagged_count,
                "is_single_spectrum": is_single,
                "series_metadata": state.get("series_metadata", {}),
                "quality_settings": {
                    "r2_threshold": self.r2_threshold,
                    "max_model_retries": self.max_model_retries,
                    "outlier_sigma": self.outlier_sigma,
                },
                "locked_config": state.get("locked_fitting_config"),
                "series_analysis_plan": state.get("series_analysis_plan"),
                "locked_preprocessing_strategy": state.get("locked_preprocessing_strategy"),
                "results": serializable_results
            }, f, indent=2, default=str)
        
        state["series_results_path"] = str(results_path)
        
        return state
    
    def _wrap_text(self, text: str, width: int = 70) -> list:
        """Wrap text to specified width, preserving words."""
        if not text:
            return [""]
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [""]
    
    def _judge_select_best_fit(self, attempts: List[dict]) -> dict:
        """
        Present all verification attempts to a judge LLM to select the best one.
        
        Called when the verification loop exhausts without any fit being approved.
        
        Args:
            attempts: List of dicts with keys:
                - result: the fit result dict (includes visualization_bytes)
                - verification: the LLM verification dict
                - config: the fitting config used
                - r2: the R² value
        
        Returns:
            Dict with:
                - selected_index: int or None
                - acceptable: bool
                - reasoning: str
                - issues_with_selected: str or None
        """
        self.logger.info("")
        self.logger.info("⚖️ No fit approved after verification loop - calling judge...")
        
        # Build attempts summary
        attempts_summary = []
        for i, attempt in enumerate(attempts):
            r2 = attempt.get("r2", 0)
            model = attempt["config"].get("physical_model", "Unknown")
            verification = attempt.get("verification", {})
            assessment = verification.get("overall_assessment", "No assessment available")
            issues = verification.get("issues_found", [])
            
            issues_brief = []
            for issue in issues[:3]:  # Limit to first 3 issues
                issues_brief.append(f"  - {issue.get('location', '?')}: {issue.get('problem', '?')}")
            issues_str = "\n".join(issues_brief) if issues_brief else "  (no specific issues listed)"
            
            summary = f"""
    **Attempt {i + 1}:**
    - Model: {model}
    - R² = {r2:.4f}
    - Assessment: {assessment}
    - Issues ({len(issues)} found):
    {issues_str}
    """
            attempts_summary.append(summary)
        
        prompt_parts = [
            self.JUDGE_PROMPT.format(attempts_summary="\n".join(attempts_summary))
        ]
        
        # Add all visualizations
        for i, attempt in enumerate(attempts):
            viz_bytes = attempt["result"].get("visualization_bytes")
            if viz_bytes:
                prompt_parts.append(f"\n\n**Attempt {i + 1} Visualization:**")
                prompt_parts.append({
                    "mime_type": "image/png",
                    "data": viz_bytes
                })
        
        try:
            response = self.model.generate_content(
                contents=prompt_parts,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )
            result, error = self._parse(response)
            
            if error or not result:
                self.logger.warning(f"   Judge failed to parse response: {error}")
                return {
                    "selected_index": None, 
                    "acceptable": False, 
                    "reasoning": f"Judge parse failed: {error}"
                }
            
            # Log judge decision
            selected = result.get("selected_index")
            acceptable = result.get("acceptable", False)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            if acceptable and selected is not None:
                self.logger.info(f"   ✅ Judge selected attempt {selected + 1}")
            else:
                self.logger.warning(f"   ⚠️ Judge found no acceptable fit")
            
            # Wrap reasoning for readability
            self.logger.info(f"   Reasoning:")
            for line in self._wrap_text(reasoning, width=70):
                self.logger.info(f"      {line}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"   Judge call failed: {e}")
            return {
                "selected_index": None, 
                "acceptable": False, 
                "reasoning": f"Judge call failed: {str(e)}"
            }



class AdaptiveRefitController:
    """
    Post-processing recovery step that re-analyzes flagged spectra independently.

    After the locked-config series processing completes, this controller:
    1. Identifies spectra flagged for quality reasons (below_threshold, fit_failed)
    2. Re-runs each one with full LLM planning + model selection + verification
    3. Updates series_results with improved fits where possible
    4. Re-runs outlier detection on updated results

    Statistical outliers (reason="statistical_outlier") are NOT re-fitted,
    because their low R² may reflect genuine physical phenomena rather
    than model inadequacy.
    """

    REFIT_REASONS = frozenset({"below_threshold", "fit_failed", "outlier_and_below_threshold"})

    def __init__(
        self,
        model,
        logger: logging.Logger,
        generation_config,
        safety_settings,
        parse_fn: Callable,
        executor: Any,
        script_instructions: str,
        correction_instructions: str,
        quality_instructions: str,
        output_dir: str,
        plot_fn: Callable,
        r2_threshold: float = 0.95,
        max_model_retries: int = 3,
        max_verification_iterations: int = 3,
        preprocessor: Any = None,
        enable_human_feedback: bool = False,
        conformance_instructions: str = "",
    ):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.plot_fn = plot_fn
        self.r2_threshold = r2_threshold
        self.enable_human_feedback = enable_human_feedback

        # Compose a fitting helper to reuse _fit_with_quality_control
        self._fitting_helper = UnifiedSeriesProcessingController(
            model=model,
            logger=logger,
            generation_config=generation_config,
            safety_settings=safety_settings,
            parse_fn=parse_fn,
            executor=executor,
            script_instructions=script_instructions,
            correction_instructions=correction_instructions,
            quality_instructions=quality_instructions,
            output_dir=output_dir,
            plot_fn=plot_fn,
            r2_threshold=r2_threshold,
            max_model_retries=max_model_retries,
            enable_human_feedback=False,
            max_verification_iterations=max_verification_iterations,
            preprocessor=preprocessor,
            conformance_instructions=conformance_instructions,
        )

    def _load_spectrum(self, idx, spectrum_paths, spectrum_stack):
        """Load spectrum data for re-analysis."""
        if spectrum_stack is not None:
            return spectrum_stack[idx]
        if spectrum_paths and idx < len(spectrum_paths):
            try:
                return self._fitting_helper._load_curve_data(spectrum_paths[idx])
            except Exception as e:
                self.logger.error(f"Failed to load {spectrum_paths[idx]}: {e}")
                return None
        return None

    def _preprocess_spectrum(self, curve_data, state):
        """Apply locked preprocessing strategy if available."""
        if self._fitting_helper.preprocessor is None:
            return curve_data
        locked_strategy = state.get("locked_preprocessing_strategy")
        try:
            curve_data, _ = self._fitting_helper.preprocessor.run_preprocessing(
                curve_data, state.get("system_info", {}),
                locked_strategy=locked_strategy,
            )
        except Exception as e:
            self.logger.warning(f"Preprocessing failed during refit: {e}")
        return curve_data

    def _build_refit_state(self, state, curve_data, idx, name):
        """Build a temporary state dict for independent re-analysis."""
        locked_config = state.get("locked_fitting_config", {})
        original_result = state["series_results"][idx]
        original_r2 = original_result.get("fit_quality", {}).get("r_squared", 0)

        # Build experimental context so the LLM knows what it's fitting
        system_info = state.get("system_info", {})
        series_metadata = state.get("series_metadata", {})
        num_spectra = state.get("num_spectra", 0)

        # Serialize full system_info so the LLM gets all metadata
        # regardless of key structure (flat or nested)
        exp_context_parts = []
        if system_info:
            exp_context_parts.append(json.dumps(system_info, indent=2, default=str))
        if series_metadata.get("variable") and series_metadata.get("values"):
            values = series_metadata["values"]
            units = series_metadata.get("units", "")
            if idx < len(values):
                exp_context_parts.append(
                    f"Series position: spectrum {idx + 1}/{num_spectra}, "
                    f"{series_metadata['variable']} = {values[idx]} {units}"
                )
        exp_context = "\n".join(exp_context_parts)

        # Summarize series context: what worked, what failed, neighbors
        series_results = state.get("series_results", [])
        series_context_parts = []
        successful = [r for r in series_results if r.get("success") and not r.get("flagged")]
        if successful:
            r2_vals = [r.get("fit_quality", {}).get("r_squared", 0) for r in successful]
            series_context_parts.append(
                f"Successful fits (locked model): {len(successful)}/{len(series_results)} spectra, "
                f"R² range {min(r2_vals):.4f}–{max(r2_vals):.4f}, "
                f"model: {successful[0].get('model_type', 'N/A')}"
            )
        flagged = [r for r in series_results if r.get("flagged") or not r.get("success")]
        if flagged:
            flagged_indices = [str(r["index"]) for r in flagged]
            series_context_parts.append(f"Failed spectra indices: [{', '.join(flagged_indices)}]")
        # Nearest successful neighbor summary
        for offset in (-1, 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < len(series_results):
                nr = series_results[neighbor_idx]
                if nr.get("success") and not nr.get("flagged"):
                    nr2 = nr.get("fit_quality", {}).get("r_squared", 0)
                    series_context_parts.append(
                        f"Neighbor spectrum [{neighbor_idx}] fitted successfully: "
                        f"model={nr.get('model_type', 'N/A')}, R²={nr2:.4f}"
                    )
        series_context = "\n".join(series_context_parts)

        refit_context = (
            f"This spectrum was previously fitted using the locked series model but achieved "
            f"inadequate fit quality (R² = {original_r2:.4f}, threshold = {self.r2_threshold}).\n\n"
            f"The locked model was: {locked_config.get('physical_model', 'Unknown')}\n"
            f"The locked strategy was: {locked_config.get('fitting_strategy', 'Unknown')}\n\n"
        )

        # Add regime context if available
        series_plan = state.get("series_analysis_plan")
        if series_plan:
            for regime in series_plan.get("regimes", []):
                if idx in regime.get("spectrum_indices", []):
                    refit_context += (
                        f"**Regime context:** This spectrum was assigned to regime "
                        f"'{regime.get('name', 'unnamed')}' with expected model: "
                        f"{regime.get('physical_model', 'Unknown')}.\n\n"
                    )
                    break

        if exp_context:
            refit_context += f"**Experimental context:**\n{exp_context}\n\n"
        if series_context:
            refit_context += f"**Series context:**\n{series_context}\n\n"
        refit_context += (
            f"IMPORTANT: The locked model failed for this specific spectrum. You MUST try a DIFFERENT "
            f"fitting approach. Consider:\n"
            f"1. Different functional forms (the locked model's shape may not match this spectrum)\n"
            f"2. Additional components (this spectrum may have features others don't)\n"
            f"3. Different physical models (this spectrum may represent a different physical regime)\n\n"
            f"Do NOT simply retry the same model with different initial parameters.\n\n"
            f"PARSIMONY: Use the SIMPLEST model that achieves R² ≥ {self.r2_threshold}. "
            f"Do not add extra components beyond what the data clearly requires. "
            f"If two peaks are visible, use a two-component model — not three or more."
        )

        fresh_config = {
            "analysis_approach": refit_context,
            "physical_model": f"Alternative to: {locked_config.get('physical_model', 'Unknown')}",
            "fitting_strategy": "Independent analysis - try different approach than locked model",
            "parameters_to_extract": locked_config.get("parameters_to_extract", []),
        }

        spectrum_paths = state.get("spectrum_paths", [])
        data_path = spectrum_paths[idx] if spectrum_paths and idx < len(spectrum_paths) else name

        stats = self._fitting_helper._compute_statistics(curve_data)
        plot_bytes = self.plot_fn(curve_data, state.get("system_info", {}))

        return {
            "data_path": data_path,
            "curve_data": curve_data,
            "original_plot_bytes": plot_bytes,
            "data_statistics": stats,
            "locked_fitting_config": fresh_config,
            "system_info": state.get("system_info", {}),
            "literature_context": state.get("literature_context"),
            "analysis_hints": state.get("analysis_hints"),
            "analysis_objective": state.get("analysis_objective"),
            "skill_name": state.get("skill_name"),
            "skill_sections": state.get("skill_sections"),
            "auxiliary_plot_bytes": state.get("auxiliary_plot_bytes"),
            "auxiliary_label": state.get("auxiliary_label"),
            "auxiliary_summary": state.get("auxiliary_summary"),
            "auxiliary_mime_type": state.get("auxiliary_mime_type"),
            "prior_knowledge": state.get("prior_knowledge", []),
            "analysis_images": [],
        }

    def _ask_user_for_consensus(self, improved, model_counts):
        """Ask user which model to use when refits found no consensus."""
        print("\n" + "=" * 60)
        print("🔄 ADAPTIVE REFIT: No model consensus among re-fitted spectra")
        print("=" * 60)
        print("\nThe re-fitted spectra used different models:")
        for i, (model, count) in enumerate(
            sorted(model_counts.items(), key=lambda x: -x[1]), 1
        ):
            indices = [str(r["index"]) for r in improved if r["new_model"] == model]
            r2s = [r["new_r2"] for r in improved if r["new_model"] == model]
            r2_str = ", ".join(f"{v:.4f}" for v in r2s)
            print(f"  {i}. '{model}' — spectra [{', '.join(indices)}], R²: {r2_str}")

        print("\nOptions:")
        print("  • Enter a number (1, 2, ...) to use that model for all re-fitted spectra")
        print("  • Type a model name to suggest a different model")
        print("  • Press Enter to keep the independent results as-is")
        print("-" * 60)

        response = input("\n🤔 Your choice: ").strip()
        if not response:
            print("✅ Keeping independent refit results.")
            return None

        # Check if user entered a number
        try:
            choice = int(response)
            models = sorted(model_counts.keys(), key=lambda m: -model_counts[m])
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                print(f"✅ Will re-fit with '{selected}'")
                return selected
        except ValueError:
            pass

        # User typed a model name directly
        print(f"✅ Will re-fit with '{response}'")
        return response

    def _run_consistency_refit(
        self, minority, target_model, improved, state, series_results,
        spectrum_paths, spectrum_stack,
    ):
        """Re-fit minority spectra using the target model."""
        peer_r2 = [r["new_r2"] for r in improved if r["new_model"] == target_model]
        peer_count = len(peer_r2)

        for entry in minority:
            idx = entry["index"]
            name = entry["name"]
            self.logger.info(f"  Re-fitting [{idx}] {name} with '{target_model}'")

            curve_data = self._load_spectrum(idx, spectrum_paths, spectrum_stack)
            if curve_data is None:
                continue
            curve_data = self._preprocess_spectrum(curve_data, state)

            refit_state = self._build_refit_state(state, curve_data, idx, name)
            if peer_r2:
                refit_state["locked_fitting_config"]["analysis_approach"] += (
                    f"\n\n**Peer evidence:** {peer_count} other spectra in this series "
                    f"were successfully refitted with '{target_model}' "
                    f"(R² {min(peer_r2):.4f}–{max(peer_r2):.4f}). "
                    f"Strongly prefer this model unless the data clearly "
                    f"requires something different."
                )
            refit_state["locked_fitting_config"]["physical_model"] = target_model

            data_path = (spectrum_paths[idx]
                         if spectrum_paths and idx < len(spectrum_paths) else name)
            try:
                result = self._fitting_helper._fit_with_quality_control(
                    state=refit_state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=name, spectrum_idx=idx,
                )
            except Exception as e:
                self.logger.error(f"  Consistency refit failed for {name}: {e}")
                continue

            new_r2 = result.get("fit_quality", {}).get("r_squared", 0)
            prev_r2 = entry["new_r2"] or 0

            if result["success"] and new_r2 >= prev_r2 * 0.99:
                self.logger.info(f"  ✅ Consistent: R² {new_r2:.4f} with '{target_model}'")
                result["adaptively_refitted"] = True
                result["original_r2"] = entry["original_r2"]
                result["refit_model_type"] = result.get("model_type")
                result["locked_model_type"] = state.get(
                    "locked_fitting_config", {}
                ).get("physical_model")
                series_results[idx] = result
                entry["new_r2"] = new_r2
                entry["new_model"] = result.get("model_type")
            elif self.enable_human_feedback:
                keep = self._ask_keep_consistency_result(
                    name, idx, target_model, new_r2,
                    entry["new_model"], prev_r2,
                )
                if keep:
                    result["adaptively_refitted"] = True
                    result["original_r2"] = entry["original_r2"]
                    result["refit_model_type"] = result.get("model_type")
                    result["locked_model_type"] = state.get(
                        "locked_fitting_config", {}
                    ).get("physical_model")
                    series_results[idx] = result
                    entry["new_r2"] = new_r2
                    entry["new_model"] = result.get("model_type")
                else:
                    self.logger.info(f"  Keeping original refit for [{idx}] {name}")
            else:
                self.logger.info(
                    f"  Keeping original refit: consensus R²={new_r2:.4f} "
                    f"vs previous R²={prev_r2:.4f}"
                )

    def _ask_keep_consistency_result(
        self, name, idx, consensus_model, consensus_r2,
        original_model, original_r2,
    ):
        """Ask user whether to keep consensus model when R² dropped."""
        print("\n" + "-" * 60)
        print(f"⚠️  Spectrum [{idx}] {name}: consensus model has lower R²")
        print("-" * 60)
        print(f"  Consensus: '{consensus_model}' → R² = {consensus_r2:.4f}")
        print(f"  Independent: '{original_model}' → R² = {original_r2:.4f}")
        print("\nOptions:")
        print(f"  • Type 'consensus' to use '{consensus_model}' for consistency")
        print(f"  • Press Enter to keep '{original_model}'")

        response = input("\nYour choice: ").strip().lower()
        if response == "consensus":
            print(f"✅ Using consensus model for [{idx}] {name}")
            return True
        print(f"✅ Keeping independent model for [{idx}] {name}")
        return False

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state

        if state.get("is_single_spectrum", True):
            return state

        flagged_spectra = state.get("flagged_spectra", [])
        if not flagged_spectra:
            self.logger.info("\n🔄 Adaptive refit: No flagged spectra, skipping.")
            return state

        refit_candidates = [f for f in flagged_spectra if f["reason"] in self.REFIT_REASONS]
        if not refit_candidates:
            self.logger.info("\n🔄 Adaptive refit: Flagged spectra are statistical outliers only, skipping.")
            return state

        self.logger.info(f"\n🔄 ADAPTIVE REFIT: {len(refit_candidates)} spectra to re-analyze independently")

        series_results = state.get("series_results", [])
        spectrum_paths = state.get("spectrum_paths", [])
        spectrum_stack = state.get("spectrum_stack")
        refit_summary = []

        for flagged in refit_candidates:
            idx = flagged["index"]
            name = flagged["name"]
            original_r2 = flagged.get("r_squared")

            self.logger.info(f"\n  Re-analyzing [{idx}] {name} (original R²={original_r2})")

            curve_data = self._load_spectrum(idx, spectrum_paths, spectrum_stack)
            if curve_data is None:
                self.logger.warning(f"  Could not load spectrum data for {name}, skipping")
                continue

            curve_data = self._preprocess_spectrum(curve_data, state)

            refit_state = self._build_refit_state(state, curve_data, idx, name)
            spectrum_paths_list = state.get("spectrum_paths", [])
            data_path = spectrum_paths_list[idx] if spectrum_paths_list and idx < len(spectrum_paths_list) else name

            try:
                refit_result = self._fitting_helper._fit_with_quality_control(
                    state=refit_state, curve_data=curve_data, data_path=data_path,
                    spectrum_name=name, spectrum_idx=idx,
                )
            except Exception as e:
                self.logger.error(f"  Refit failed for {name}: {e}")
                refit_summary.append({
                    "index": idx, "name": name,
                    "original_r2": original_r2, "new_r2": None,
                    "improved": False,
                })
                continue

            new_r2 = refit_result.get("fit_quality", {}).get("r_squared", 0)
            locked_model = state.get("locked_fitting_config", {}).get("physical_model")

            if refit_result["success"] and (original_r2 is None or new_r2 > original_r2):
                self.logger.info(f"  ✅ Improved: R² {original_r2} → {new_r2:.4f}")
                refit_result["adaptively_refitted"] = True
                refit_result["original_r2"] = original_r2
                refit_result["refit_model_type"] = refit_result.get("model_type")
                refit_result["locked_model_type"] = locked_model
                series_results[idx] = refit_result

                refit_summary.append({
                    "index": idx, "name": name,
                    "original_r2": original_r2, "new_r2": new_r2,
                    "original_model": locked_model,
                    "new_model": refit_result.get("model_type"),
                    "improved": True,
                })
            else:
                self.logger.info(f"  No improvement: R² {original_r2} → {new_r2:.4f}, keeping original")
                refit_summary.append({
                    "index": idx, "name": name,
                    "original_r2": original_r2, "new_r2": new_r2,
                    "improved": False,
                })

        # --- Consistency pass ---
        # If a majority of improved refits converged on the same model type,
        # re-refit outlier models using the consensus as guidance.
        # If no consensus, optionally ask the user for guidance.
        improved = [r for r in refit_summary if r["improved"] and r.get("new_model")]
        if len(improved) >= 2:
            model_counts = {}
            for r in improved:
                model_counts[r["new_model"]] = model_counts.get(r["new_model"], 0) + 1
            top_model, top_count = max(model_counts.items(), key=lambda x: x[1])
            has_majority = top_count > len(improved) / 2
            minority = [r for r in improved if r["new_model"] != top_model]

            if has_majority and minority:
                self.logger.info(
                    f"\n🔄 Consistency pass: majority model is '{top_model}' "
                    f"({top_count}/{len(improved)}), re-fitting {len(minority)} outlier(s)"
                )
                self._run_consistency_refit(
                    minority, top_model, improved, state, series_results,
                    spectrum_paths, spectrum_stack,
                )
            elif not has_majority and len(model_counts) > 1:
                if self.enable_human_feedback:
                    # No consensus — ask user for guidance
                    user_model = self._ask_user_for_consensus(improved, model_counts)
                    if user_model:
                        user_minority = [r for r in improved if r["new_model"] != user_model]
                        if user_minority:
                            self.logger.info(
                                f"\n🔄 User-guided consistency: re-fitting "
                                f"{len(user_minority)} spectra with '{user_model}'"
                            )
                            self._run_consistency_refit(
                                user_minority, user_model, improved, state,
                                series_results, spectrum_paths, spectrum_stack,
                            )
                else:
                    # No human feedback and no consensus — keep independent results.
                    # The parsimony prompt should minimize this case; if models still
                    # disagree, the inconsistency is noted in the synthesis.
                    self.logger.info(
                        f"\n⚠️ No model consensus among refitted spectra "
                        f"({dict(model_counts)}). Keeping independent results."
                    )

        state["series_results"] = series_results
        state["refit_summary"] = refit_summary

        # Re-run outlier detection with updated results
        updated_flagged = self._fitting_helper._detect_outliers(series_results)
        state["flagged_spectra"] = updated_flagged

        improved_count = sum(1 for r in refit_summary if r["improved"])
        self.logger.info(f"\n🔄 Adaptive refit complete: {improved_count}/{len(refit_candidates)} spectra improved")

        return state


class ConditionalTrendAnalysisController:
    """Generates and executes custom Python script for trend analysis. Only for n>=2."""
    
    TREND_ANALYSIS_INSTRUCTIONS = '''You are analyzing a series of fitted spectra/curves to identify trends.
{objective}
**SERIES SUMMARY:**
{series_summary}

**SERIES METADATA:**
{series_metadata}

**FLAGGED SPECTRA:**
{flagged_info}

**CRITICAL REQUIREMENTS:**
1. DO NOT use plt.show() anywhere in the script - only save figures with plt.savefig()
2. DO NOT include individual spectrum fit visualizations - only create parameter trend dashboard
3. Use plt.close('all') after saving each figure to free memory

**VISUALIZATION SCOPE - TRENDS:**
Create a SINGLE dashboard figure showing how fitted PARAMETERS evolve across the series.
DO NOT recreate individual spectrum fits - those already exist separately.
The dashboard should show:
- Parameter values (y-axis) vs series variable like temperature/time/index (x-axis)
- Error bars if uncertainties are available
- Fit quality (R²) evolution
- Mark flagged spectra with distinct markers

**FIGURE REQUIREMENTS:**
- Create ONE summary dashboard figure (parameter_trends.png)
- 2x2 or 2x3 subplot layout with 4-6 most important parameters
- Clean, publication-quality appearance
- Mark flagged spectra with red X markers
- Include linear regression trend lines where appropriate
- NO plt.show() calls
- Use plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
- Call plt.close('all') at the end

**DATA EXTRACTION PATTERN:**
```python
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - REQUIRED
import matplotlib.pyplot as plt

# Load data
with open('series_fit_results.json', 'r') as f:
    data = json.load(f)

results = data['results']
series_metadata = data.get('series_metadata', {{}})
# series_metadata has: variable, values (one per spectrum), unit

# Extract series variable and parameters...
# Create figure with subplots...
# Plot parameter trends (NOT individual fits)...

plt.savefig('parameter_trends.png', dpi=150, bbox_inches='tight')
plt.close('all')  # REQUIRED - prevent memory leaks and display
```

Return JSON with:
{{
    "analysis_approach": "brief description",
    "key_metrics": ["list", "of", "parameters", "tracked"],
    "flagged_handling": "how flagged spectra are marked",
    "expected_outputs": ["parameter_trends.png"],
    "script": "full python script - NO plt.show()"
}}
'''

    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings,
                 parse_fn: Callable, executor: Any, output_dir: str, max_corrections: int = 3):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.executor = executor
        self.output_dir = Path(output_dir)
        self.max_corrections = max_corrections

    def _generate_trend_script(self, state: dict) -> Optional[Dict]:
        series_results = state.get("series_results", [])
        series_metadata = state.get("series_metadata", {})
        flagged_spectra = state.get("flagged_spectra", [])

        param_summary = []
        for r in series_results:
            if r["success"]:
                summary = {"index": r["index"], "name": r["name"], "model_type": r.get("model_type"),
                          "parameters": r.get("parameters", {}), "fit_quality": r.get("fit_quality", {})}
                if r.get("flagged"):
                    summary["flagged"] = True
                    summary["flag_reason"] = r.get("flag_reason")
                param_summary.append(summary)

        flagged_info = json.dumps(flagged_spectra, indent=2) if flagged_spectra else "No spectra were flagged."

        objective = state.get("analysis_objective")
        objective_block = (
            f"\n**ANALYSIS OBJECTIVE:**\n{objective}\n"
            "Frame the trend analysis around answering this objective. "
            "If the objective involves calibration or quantitative modeling, "
            "the script must compute and output regression models.\n"
        ) if objective else ""

        prompt = self.TREND_ANALYSIS_INSTRUCTIONS.format(
            series_summary=json.dumps(param_summary, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            flagged_info=flagged_info,
            objective=objective_block,
        )
        
        try:
            response = self.model.generate_content(contents=[prompt], generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, error_dict = self._parse(response)
            if error_dict and not (result_json and 'script' in result_json):
                return None
            return result_json
        except Exception as e:
            self.logger.error(f"Error generating trend script: {e}")
            return None

    def _execute_script(self, script: str) -> tuple:
        # Remove any plt.show() calls that might have slipped through
        script = re.sub(r'plt\.show\s*\(\s*\)', '# plt.show() removed', script)
        
        # Ensure matplotlib backend is set at the top
        if 'matplotlib.use' not in script:
            script = "import matplotlib\nmatplotlib.use('Agg')\n" + script
        
        script_path = self.output_dir / "trend_analysis.py"
        with open(script_path, 'w') as f:
            f.write(script)
        result = self.executor.execute_script(script, working_dir=str(self.output_dir))
        return result.get("status") == "success", result.get("stdout", ""), result.get("message", "")

    def _correct_script(self, original_script: str, error_message: str, attempt: int) -> Optional[str]:
        self.logger.info(f"   🔧 Attempting script correction (attempt {attempt})...")
        if len(error_message) > 1000:
            error_message = error_message[:500] + "\n...[truncated]...\n" + error_message[-500:]
        
        prompt = f"""Fix this Python script that failed:

**SCRIPT:**
```python
{original_script}
```

**ERROR:**
```
{error_message}
```

Return JSON with: {{"diagnosis": "...", "script": "corrected script"}}
"""
        
        try:
            response = self.model.generate_content(contents=[prompt], generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, _ = self._parse(response)
            if result_json:
                self.logger.info(f"   📋 Diagnosis: {result_json.get('diagnosis', 'N/A')}")
                return result_json.get("script")
            return None
        except Exception as e:
            self.logger.error(f"Script correction failed: {e}")
            return None

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        num_spectra = state.get("num_spectra", 1)
        is_single = state.get("is_single_spectrum", True)
        
        if is_single or num_spectra < 2:
            self.logger.info("\n📊 Trend analysis skipped (single spectrum mode).\n")
            state["trend_analysis_results"] = {"success": True, "skipped": True, "reason": "Single spectrum - no trend analysis applicable"}
            return state
        
        self.logger.info("")
        self.logger.info("📈 TREND ANALYSIS")
        
        flagged_count = len(state.get("flagged_spectra", []))
        if flagged_count > 0:
            self.logger.info(f"   Note: {flagged_count} flagged spectra will be highlighted in visualizations")
        
        script_result = self._generate_trend_script(state)
        
        if not script_result or "script" not in script_result:
            self.logger.error("Failed to generate trend analysis script.")
            state["trend_analysis_results"] = {"success": False, "error": "Script generation failed"}
            return state
        
        self.logger.info(f"   📊 Approach: {script_result.get('analysis_approach', 'unknown')}")
        self.logger.info(f"   📈 Metrics: {script_result.get('key_metrics', [])}")
        
        script = script_result["script"]
        success, stdout, stderr = False, "", ""
        
        for attempt in range(self.max_corrections + 1):
            if attempt > 0:
                self.logger.info(f"   🔄 Execution attempt {attempt + 1}")
            
            success, stdout, stderr = self._execute_script(script)
            
            if success:
                self.logger.info("   ✅ Trend analysis completed!")
                break
            
            self.logger.warning(f"   ⚠️ Script failed: {stderr[:200]}...")
            
            if attempt < self.max_corrections:
                corrected = self._correct_script(script, stderr, attempt + 1)
                if corrected:
                    script = corrected
                else:
                    break
        
        generated_files = []
        # Only include trend analysis outputs, NOT individual fit images or review files
        for f in self.output_dir.glob('*.png'):
            # Exclude individual fit visualizations (they have _fit.png suffix or spectrum_ prefix with _fit)
            fname = f.name
            if '_fit.png' in fname:
                continue
            if fname.startswith('spectrum_') and fname.endswith('.png'):
                continue
            # Exclude other known non-trend files
            if fname in ['quality_review_fit.png', 'first_spectrum_fit_review.png']:
                continue
            generated_files.append(str(f))
        
        # Include any CSV/JSON outputs from trend analysis
        for f in self.output_dir.glob('*.csv'):
            if f.name not in ['series_fit_results.json', 'flagged_spectra.json']:
                generated_files.append(str(f))
        
        state["trend_analysis_results"] = {
            "success": success, "skipped": False,
            "approach": script_result.get("analysis_approach"),
            "metrics_tracked": script_result.get("key_metrics"),
            "flagged_handling": script_result.get("flagged_handling"),
            "stdout": stdout, "stderr": stderr if not success else None,
            "generated_files": generated_files,
            "script_path": str(self.output_dir / "trend_analysis.py")
        }
        
        return state


class UnifiedCurveSynthesisController:
    """Synthesizes findings into scientific claims. Adapts to single vs series."""
    
    SERIES_SYNTHESIS_INSTRUCTIONS = '''You are synthesizing findings from a curve fitting analysis of a spectral series.

**SERIES OVERVIEW:**
- Total spectra: {num_spectra}
- Successful fits: {successful_fits}
- Fitting model: {model_type}
- Flagged spectra: {flagged_count}

**INDIVIDUAL FIT SUMMARIES:**
{fit_summaries}

**FLAGGED SPECTRA (require attention):**
{flagged_summary}

**ADAPTIVE REFIT RESULTS:**
{refit_summary}

**TREND ANALYSIS RESULTS:**
{trend_results}

**SERIES METADATA:**
{series_metadata}

**SYSTEM INFORMATION:**
{system_info}

Provide comprehensive scientific synthesis including:
1. Overall quality assessment
2. Key trends in fitted parameters
3. Physical interpretation of parameter evolution
4. **Analysis of flagged spectra** - what might explain why these fit poorly?
5. Scientific claims supported by the data
6. Caveats and limitations
7. **Analysis of adaptively re-fitted spectra** - if any spectra were re-analyzed with different models, interpret what this means scientifically (e.g., phase transition, different regime, instrumental change)

Return JSON with:
{{
    "detailed_analysis": "comprehensive scientific interpretation",
    "scientific_claims": [
        {{
            "claim": "specific claim statement",
            "scientific_impact": "why this matters",
            "has_anyone_question": "research question formulation",
            "keywords": ["keyword1", "keyword2"]
        }}
    ],
    "parameter_trends": {{
        "parameter_name": {{"trend": "increasing/decreasing/stable", "interpretation": "physical meaning"}}
    }},
    "flagged_spectra_analysis": {{
        "summary": "interpretation of why spectra were flagged",
        "possible_causes": ["list of explanations"],
        "recommended_followup": ["suggested investigations"],
        "scientific_significance": "whether outliers represent interesting physics"
    }},
    "refit_analysis": {{
        "summary": "interpretation of why different models were needed",
        "model_changes": [{{"index": 0, "from_model": "...", "to_model": "...", "interpretation": "..."}}],
        "scientific_implications": "what the model changes tell us about the system"
    }},
    "caveats": "limitations and considerations"
}}
'''

    def __init__(self, model, logger: logging.Logger, generation_config, safety_settings,
                 parse_fn: Callable, single_spectrum_instructions: str, output_dir: str):
        self.model = model
        self.logger = logger
        self.generation_config = generation_config
        self.safety_settings = safety_settings
        self._parse = parse_fn
        self.single_spectrum_instructions = single_spectrum_instructions
        self.output_dir = Path(output_dir)

    def _synthesize_single_spectrum(self, state: dict) -> dict:
        self.logger.info("")
        self.logger.info("🔬 SINGLE SPECTRUM INTERPRETATION")
        
        fit_results = state.get("fit_results", {})
        series_results = state.get("series_results", [])
        
        quality_warning = None
        if series_results and series_results[0].get("quality_warning"):
            quality_warning = series_results[0]["quality_warning"]
        
        formatted = self.single_spectrum_instructions.format(
            model_type=fit_results.get("model_type", "Curve fit"),
            summary=fit_results.get("summary", "Fitting complete"),
        )
        
        prompt_parts = [formatted, "\n## Original Data", {"mime_type": "image/png", "data": state["original_plot_bytes"]}]
        
        if state.get("final_plot_bytes"):
            prompt_parts.extend(["\n## Fit Result", {"mime_type": "image/png", "data": state["final_plot_bytes"]}])
        
        prompt_parts.extend([
            "\n## Parameters\n" + json.dumps(fit_results.get("parameters", {}), indent=2),
            "\n## Fit Quality\n" + json.dumps(fit_results.get("fit_quality", {}), indent=2),
            "\n## Metadata\n" + json.dumps(state.get("system_info", {}), indent=2),
        ])
        
        if quality_warning:
            prompt_parts.append(f"\n## Quality Warning\n{quality_warning}\nNote: Alternative models were attempted but this was the best fit achieved.")
        
        if state.get("literature_context"):
            prompt_parts.extend(["\n## Literature", state["literature_context"]])

        _append_objective_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)

        try:
            response = self.model.generate_content(contents=prompt_parts, generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, error_dict = self._parse(response)

            if error_dict:
                self.logger.error(f"Synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Single spectrum synthesis complete.")
        except Exception as e:
            self.logger.error(f"Synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state

    def _synthesize_series(self, state: dict) -> dict:
        self.logger.info("")
        self.logger.info("🔬 SERIES SYNTHESIS")
        
        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        series_metadata = state.get("series_metadata", {})
        flagged_spectra = state.get("flagged_spectra", [])
        
        successful_fits = [r for r in series_results if r["success"]]
        refit_summary_data = state.get("refit_summary", [])

        fit_summaries = []
        for r in successful_fits[:15]:
            summary = {"index": r["index"], "name": r["name"], "model": r.get("model_type"),
                      "key_params": r.get("parameters", {}), "r_squared": r.get("fit_quality", {}).get("r_squared")}
            if r.get("flagged"):
                summary["flagged"] = True
                summary["flag_reason"] = r.get("flag_reason")
            if r.get("adaptively_refitted"):
                summary["adaptively_refitted"] = True
                summary["original_r2"] = r.get("original_r2")
                summary["refit_model"] = r.get("refit_model_type")
                summary["locked_model"] = r.get("locked_model_type")
            fit_summaries.append(summary)

        flagged_summary = json.dumps(flagged_spectra, indent=2) if flagged_spectra else "No spectra were flagged."
        refit_summary_str = json.dumps(refit_summary_data, indent=2) if refit_summary_data else "No spectra were adaptively re-fitted."

        # Handle mixed model types when refitting occurred
        model_types_used = set()
        for r in successful_fits:
            mt = r.get("model_type")
            if mt:
                model_types_used.add(mt)

        if len(model_types_used) <= 1:
            model_type = successful_fits[0].get("model_type") if successful_fits else "Unknown"
        else:
            locked_model = state.get("locked_fitting_config", {}).get("physical_model", "Unknown")
            refitted_models = [r.get("refit_model_type") for r in successful_fits
                             if r.get("adaptively_refitted") and r.get("refit_model_type")]
            unique_refit = sorted(set(refitted_models))
            model_type = f"Primary: {locked_model}; Re-fitted: {', '.join(unique_refit)}"

        prompt = self.SERIES_SYNTHESIS_INSTRUCTIONS.format(
            num_spectra=state.get("num_spectra", 1),
            successful_fits=len(successful_fits),
            model_type=model_type,
            flagged_count=len(flagged_spectra),
            fit_summaries=json.dumps(fit_summaries, indent=2),
            flagged_summary=flagged_summary,
            refit_summary=refit_summary_str,
            trend_results=json.dumps(trend_results, indent=2),
            series_metadata=json.dumps(series_metadata, indent=2),
            system_info=json.dumps(state.get("system_info", {}), indent=2)
        )
        
        prompt_parts = [prompt]
        
        if flagged_spectra:
            prompt_parts.append("\n\n**FLAGGED SPECTRA VISUALIZATIONS:**")
            flagged_indices = {f["index"] for f in flagged_spectra}
            included_count = 0
            for r in series_results:
                if r["index"] in flagged_indices and r.get("visualization_bytes") and included_count < 5:
                    prompt_parts.append(f"\n{r['name']} (flagged: {r.get('flag_reason', 'unknown')}):")
                    prompt_parts.append({"mime_type": "image/png", "data": r["visualization_bytes"]})
                    included_count += 1
        
        if trend_results.get("success") and trend_results.get("generated_files"):
            prompt_parts.append("\n\n**TREND VISUALIZATIONS:**")
            for file_path in trend_results["generated_files"][:5]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        prompt_parts.append(f"\n{Path(file_path).name}:")
                        prompt_parts.append({"mime_type": "image/png", "data": f.read()})

        _append_objective_context(prompt_parts, state)
        _append_auxiliary_context(prompt_parts, state)
        _append_skill_context(prompt_parts, state, "interpretation")
        _append_prior_knowledge_context(prompt_parts, state)

        try:
            response = self.model.generate_content(contents=prompt_parts, generation_config=self.generation_config, safety_settings=self.safety_settings)
            result_json, error_dict = self._parse(response)

            if error_dict:
                self.logger.error(f"Series synthesis failed: {error_dict}")
                state["synthesis_result"] = {"error": str(error_dict)}
            else:
                state["synthesis_result"] = result_json
                self.logger.info("✅ Series synthesis complete.")
        except Exception as e:
            self.logger.error(f"Series synthesis error: {e}")
            state["synthesis_result"] = {"error": str(e)}
        
        return state

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            return self._synthesize_single_spectrum(state)
        else:
            return self._synthesize_series(state)


class UnifiedCurveReportController:
    """Generates final HTML report for series analysis. Only for n>=2."""
    
    def __init__(self, logger: logging.Logger, output_dir: str):
        self.logger = logger
        self.output_dir = Path(output_dir)

    def _image_to_base64(self, image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')

    def _generate_flagged_spectra_section(self, flagged_spectra: List[dict], series_results: List[dict], synthesis: dict) -> str:
        if not flagged_spectra:
            return ""
        
        flagged_analysis = synthesis.get("flagged_spectra_analysis", {})
        
        html = f"""
        <h2>⚠️ Flagged Spectra</h2>
        <div class="flagged-summary">
            <p><strong>{len(flagged_spectra)} spectra flagged for review</strong></p>
            <p>{flagged_analysis.get("summary", "Some spectra showed anomalous fitting behavior.")}</p>
        </div>
"""
        
        causes = flagged_analysis.get("possible_causes", [])
        if causes:
            html += "<h3>Possible Causes</h3><ul>"
            for cause in causes:
                html += f"<li>{cause}</li>"
            html += "</ul>"
        
        followup = flagged_analysis.get("recommended_followup", [])
        if followup:
            html += "<h3>Recommended Follow-up</h3><ul>"
            for item in followup:
                html += f"<li>{item}</li>"
            html += "</ul>"
        
        significance = flagged_analysis.get("scientific_significance", "")
        if significance:
            html += f"<h3>Scientific Significance</h3><p>{significance}</p>"
        
        html += '<h3>Flagged Spectra Details</h3><div class="flagged-grid">'
        
        badge_colors = {
            "fit_failed": ("#dc3545", "Failed"),
            "statistical_outlier": ("#fd7e14", "Outlier"),
            "below_threshold": ("#ffc107", "Low R²"),
            "outlier_and_below_threshold": ("#dc3545", "Critical"),
        }
        
        for f in flagged_spectra:
            result = next((r for r in series_results if r["index"] == f["index"]), None)
            color, label = badge_colors.get(f["reason"], ("#6c757d", "Flagged"))
            
            html += f'<div class="flagged-card" style="border-color: {color};">'
            html += f'<div class="flagged-card-header"><strong>{f["name"]}</strong>'
            html += f'<span class="flagged-badge" style="background-color: {color};">{label}</span></div>'
            
            if f.get("r_squared") is not None:
                html += f'<p><strong>R²:</strong> {f["r_squared"]:.4f} (series mean: {f["series_mean"]:.4f})</p>'
                if f.get("deviation_sigma") is not None:
                    html += f'<p><strong>Deviation:</strong> {f["deviation_sigma"]:.1f}σ below mean</p>'
            
            html += f'<p class="flagged-recommendation">{f["recommendation"]}</p>'
            
            if result and result.get("visualization_path") and Path(result["visualization_path"]).exists():
                with open(result["visualization_path"], 'rb') as img_f:
                    b64 = self._image_to_base64(img_f.read())
                html += f'<img src="data:image/png;base64,{b64}" alt="{f["name"]}">'
            
            html += '</div>'
        
        html += '</div>'
        return html

    def _generate_refit_section(self, refit_summary: List[dict], series_results: List[dict]) -> str:
        """Generate HTML section for adaptive refit results."""
        if not refit_summary:
            return ""

        improved = [r for r in refit_summary if r["improved"]]
        not_improved = [r for r in refit_summary if not r["improved"]]

        html = f"""
        <h2>🔄 Adaptive Re-Fitting Results</h2>
        <div class="refit-summary">
            <p><strong>{len(improved)}/{len(refit_summary)}</strong> spectra improved through independent re-analysis</p>
        </div>
"""

        if improved:
            html += '<h3>Improved Fits</h3><table class="params-table"><thead><tr>'
            html += '<th>Spectrum</th><th>Original R²</th><th>New R²</th><th>Original Model</th><th>New Model</th>'
            html += '</tr></thead><tbody>'
            for r in improved:
                orig_r2 = f"{r['original_r2']:.4f}" if r.get("original_r2") is not None else "Failed"
                new_r2 = f"{r['new_r2']:.4f}" if r.get("new_r2") is not None else "N/A"
                html += f'<tr><td>{r["name"]}</td><td>{orig_r2}</td><td>{new_r2}</td>'
                html += f'<td>{r.get("original_model", "N/A")}</td><td>{r.get("new_model", "N/A")}</td></tr>'
            html += '</tbody></table>'

        if not_improved:
            html += '<h3>Unchanged Fits</h3><p>The following spectra could not be improved with alternative models:</p><ul>'
            for r in not_improved:
                html += f'<li>{r["name"]} (R² remained {r.get("original_r2", "N/A")})</li>'
            html += '</ul>'

        # Include visualizations for improved spectra
        for r in improved:
            result = next((sr for sr in series_results if sr.get("index") == r["index"]), None)
            if result and result.get("visualization_path") and Path(result["visualization_path"]).exists():
                with open(result["visualization_path"], 'rb') as f:
                    b64 = self._image_to_base64(f.read())
                html += f'<div class="image-card" style="border-left: 4px solid #17a2b8;"><img src="data:image/png;base64,{b64}" alt="{r["name"]}"><div class="image-label">{r["name"]} (Re-fitted, R²: {r.get("new_r2", 0):.4f})</div></div>'

        return html

    def _generate_individual_fits_section(self, series_results: List[dict], num_spectra: int) -> str:
        results_with_viz = [(i, r) for i, r in enumerate(series_results) 
                           if r.get("visualization_path") and Path(r["visualization_path"]).exists()]
        
        if not results_with_viz:
            return ""
        
        failed_indices = {i for i, r in enumerate(series_results) if not r["success"]}
        flagged_indices = {i for i, r in enumerate(series_results) if r.get("flagged")}
        priority_indices = failed_indices | flagged_indices
        
        if num_spectra <= 10:
            indices_to_show = set(range(num_spectra))
            section_note = ""
        elif num_spectra <= 30:
            indices_to_show = set(range(min(3, num_spectra))) | set(range(max(0, num_spectra - 3), num_spectra))
            if num_spectra > 6:
                step = (num_spectra - 6) // 5
                for i in range(3, num_spectra - 3, max(1, step)):
                    if len(indices_to_show) < 10:
                        indices_to_show.add(i)
            indices_to_show.update(priority_indices)
            not_shown = num_spectra - len(indices_to_show)
            section_note = f"<p><em>Showing {len(indices_to_show)} of {num_spectra} fits. {not_shown} fits not displayed.</em></p>"
        else:
            indices_to_show = {0, 1, num_spectra - 2, num_spectra - 1}
            indices_to_show.update(list(priority_indices)[:10])
            section_note = f"<p><em>Large series ({num_spectra} spectra): Showing boundary fits and flagged/failed spectra.</em></p>"
        
        indices_to_show = sorted(indices_to_show)
        
        html = f"\n        <h2>Individual Fit Results</h2>\n{section_note}"
        html += '        <div class="image-grid" style="grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));">\n'
        
        for idx in indices_to_show:
            if idx >= len(series_results):
                continue
            r = series_results[idx]
            viz_path = r.get("visualization_path")
            
            if viz_path and Path(viz_path).exists():
                with open(viz_path, 'rb') as f:
                    b64 = self._image_to_base64(f.read())
                
                if not r["success"]:
                    status, status_color = "✗ FAILED", "#e74c3c"
                elif r.get("adaptively_refitted"):
                    status, status_color = "🔄 Re-fitted", "#17a2b8"
                elif r.get("flagged"):
                    status, status_color = f"⚠ {r.get('flag_reason', 'Flagged')}", "#fd7e14"
                else:
                    status, status_color = "✓", "#27ae60"

                r_squared = r.get("fit_quality", {}).get("r_squared", 0)
                r2_str = f"R² = {r_squared:.4f}" if isinstance(r_squared, float) else ""
                refit_note = ""
                if r.get("adaptively_refitted") and r.get("original_r2") is not None:
                    refit_note = f"<br><small>Original R²: {r['original_r2']:.4f}</small>"

                html += f'''
            <div class="image-card" style="border-left: 4px solid {status_color};">
                <img src="data:image/png;base64,{b64}" alt="{r['name']}">
                <div style="margin-top: 8px;">
                    <strong>{r['name']}</strong><br>
                    <span style="color: {status_color};">{status}</span> {r2_str}{refit_note}
                </div>
            </div>
'''
        
        html += "        </div>\n"
        return html

    def execute(self, state: dict) -> dict:
        if state.get("error_dict"):
            return state
        
        is_single = state.get("is_single_spectrum", True)
        
        if is_single:
            self.logger.info("")
            self.logger.info("📄 Single spectrum report handled by standard controller")
            return state
        
        self._generate_series_report(state)
        return state

    def _generate_series_report(self, state: dict) -> None:
        self.logger.info("")
        self.logger.info("📄 GENERATING SERIES REPORT")
        
        series_results = state.get("series_results", [])
        trend_results = state.get("trend_analysis_results", {})
        synthesis = state.get("synthesis_result", {})
        series_metadata = state.get("series_metadata", {})
        locked_config = state.get("locked_fitting_config", {})
        flagged_spectra = state.get("flagged_spectra", [])
        refit_summary = state.get("refit_summary", [])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        num_spectra = len(series_results)
        successful = sum(1 for r in series_results if r["success"])
        flagged_count = len(flagged_spectra)
        refitted_count = sum(1 for r in refit_summary if r.get("improved"))
        
        # Quality status indicator
        if flagged_count == 0:
            quality_indicator = '<span class="quality-indicator quality-good">✓ All fits acceptable</span>'
        elif flagged_count <= num_spectra * 0.1:
            quality_indicator = f'<span class="quality-indicator quality-warning">⚠ {flagged_count} spectra flagged</span>'
        else:
            quality_indicator = f'<span class="quality-indicator quality-critical">⚠ {flagged_count} spectra flagged ({100*flagged_count/num_spectra:.0f}%)</span>'
        
        # Build trend visualizations HTML
        trend_viz_html = ""
        if trend_results.get("success") and trend_results.get("generated_files"):
            trend_viz_html = '<h2>3. Trend Visualizations</h2><div class="image-grid">'
            for file_path in trend_results["generated_files"]:
                if file_path.endswith('.png') and Path(file_path).exists():
                    with open(file_path, 'rb') as f:
                        b64 = self._image_to_base64(f.read())
                    name = Path(file_path).stem.replace('_', ' ').title()
                    trend_viz_html += f'<div class="image-card"><img src="data:image/png;base64,{b64}" alt="{name}"><div class="image-label">{name}</div></div>'
            trend_viz_html += '</div>'
        
        # Parameter trends HTML
        param_trends_html = ""
        param_trends = synthesis.get('parameter_trends', {})
        if param_trends:
            param_trends_html = "<h2>2. Parameter Trends</h2>"
            for param_name, trend_info in param_trends.items():
                if isinstance(trend_info, dict):
                    param_trends_html += f'<div class="trend-card"><strong>{param_name}</strong><br>Trend: {trend_info.get("trend", "N/A")}<br><em>{trend_info.get("interpretation", "")}</em></div>'
        
        # Scientific claims HTML
        claims_html = ""
        scientific_claims = synthesis.get('scientific_claims', [])
        if scientific_claims:
            claims_html = "<h2>5. Scientific Claims</h2>"
            for i, claim in enumerate(scientific_claims, 1):
                keywords = claim.get('keywords', [])
                keywords_str = ', '.join(keywords) if keywords else 'N/A'
                claims_html += f'''<div class="claim-card">
            <div class="claim-title">Claim {i}: {claim.get('claim', 'N/A')}</div>
            <p><strong>Scientific Impact:</strong> {claim.get('scientific_impact', 'N/A')}</p>
            <p><strong>Literature Search Query:</strong> <em>{claim.get('has_anyone_question', 'N/A')}</em></p>
            <p><strong>Keywords:</strong> {keywords_str}</p>
        </div>'''
        
        # Caveats HTML
        caveats_html = ""
        caveats = synthesis.get('caveats', '')
        if caveats:
            caveats_html = f'<h2>6. Caveats & Limitations</h2><div class="caveats">{caveats}</div>'
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curve Fitting Series Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1400px; margin: 0 auto; padding: 20px; background-color: #f4f4f9; }}
        .container {{ background-color: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        h3 {{ color: #16a085; margin-top: 20px; }}
        .metadata-box {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 5px solid #3498db; margin-bottom: 20px; }}
        .analysis-text {{ white-space: pre-wrap; background-color: #fafafa; padding: 20px; border-radius: 5px; border: 1px solid #eee; margin-top: 15px; }}
        .claim-card {{ background-color: #e8f6f3; border-left: 5px solid #1abc9c; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .claim-title {{ font-weight: bold; font-size: 1.1em; color: #0e6655; }}
        .trend-card {{ background-color: #fef9e7; border-left: 5px solid #f39c12; padding: 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 25px; margin-top: 20px; }}
        .image-card {{ background: white; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 3px; }}
        .image-label {{ margin-top: 12px; font-weight: bold; color: #444; }}
        .caveats {{ background-color: #fff8e6; border-left: 5px solid #f0ad4e; padding: 15px; margin-top: 20px; border-radius: 0 5px 5px 0; }}
        .footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.8em; }}
        .quality-indicator {{ display: inline-block; padding: 5px 12px; border-radius: 15px; font-weight: bold; font-size: 0.9em; }}
        .quality-good {{ background-color: #d4edda; color: #155724; }}
        .quality-warning {{ background-color: #fff3cd; color: #856404; }}
        .quality-critical {{ background-color: #f8d7da; color: #721c24; }}
        .flagged-summary {{ background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 15px; margin-bottom: 20px; border-radius: 0 5px 5px 0; }}
        .flagged-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-top: 15px; }}
        .flagged-card {{ background: white; border: 2px solid #ffc107; border-radius: 8px; padding: 15px; }}
        .flagged-card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .flagged-badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.85em; color: white; }}
        .flagged-card img {{ max-width: 100%; margin-top: 10px; border-radius: 4px; }}
        .flagged-recommendation {{ margin: 10px 0; font-size: 0.9em; color: #666; }}
        .refit-summary {{ background-color: #d1ecf1; border-left: 5px solid #17a2b8; padding: 15px; margin-bottom: 20px; border-radius: 0 5px 5px 0; }}
        .params-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .params-table th, .params-table td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
        .params-table th {{ background-color: #e9ecef; font-weight: bold; }}
        .params-table tr:nth-child(even) {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Spectral Series Analysis Report</h1>
        <div class="metadata-box">
            <p><strong>Date:</strong> {timestamp}</p>
            <p><strong>Spectra Processed:</strong> {successful}/{num_spectra}</p>
            <p><strong>Series Variable:</strong> {series_metadata.get('variable', 'N/A')}</p>
            <p><strong>Fitting Model:</strong> {locked_config.get('physical_model', 'N/A')}{f' ({refitted_count} spectra re-fitted with alternative models)' if refitted_count > 0 else ''}</p>
            <p><strong>Quality Status:</strong> {quality_indicator}</p>
        </div>
        <h2>1. Scientific Analysis</h2>
        <div class="analysis-text">{synthesis.get('detailed_analysis', 'No analysis available.')}</div>
        {param_trends_html}
        {trend_viz_html}
        {self._generate_individual_fits_section(series_results, num_spectra)}
        {self._generate_refit_section(refit_summary, series_results) if refit_summary else ''}
        {self._generate_flagged_spectra_section(flagged_spectra, series_results, synthesis) if flagged_spectra else ''}
        {claims_html}
        {caveats_html}
        <div class="footer">Generated by SciLink Curve Fitting Series Analysis Agent</div>
    </div>
</body>
</html>"""

        report_path = self.output_dir / "series_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        state["report_path"] = str(report_path)
        self.logger.info(f"   ✅ Report saved: {report_path}")