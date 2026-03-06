# scilink/agents/sim_agents/lammps_analysis_agent.py

import os
import re
import ast
import sys
import time
import json
import logging
import shutil
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...executors import check_security_sandbox_indicators, ScriptExecutor
from ._deprecation import normalize_params


class LAMMPSAnalysisAgent:
    """
    Flexible agent for analyzing LAMMPS simulations.
    
    Generates and executes custom Python code based on available 
    simulation files and research goals.
    
    Supports multiple package management modes:
    - 'strict': Only use pre-installed packages
    - 'permissive': Install missing packages at runtime
    - 'dockerfile': Generate custom Dockerfile
    """
    
    PACKAGE_MODES = {
        'strict': 'Only use pre-installed packages (recommended for containers)',
        'permissive': 'Install missing packages at runtime (local development)',
        'dockerfile': 'Generate custom Dockerfile for this analysis'
    }
    
    STANDARD_PACKAGES = [
        'numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn',
        'json', 'csv', 'os', 'sys', 're', 'math', 'pathlib',
        'collections', 'itertools', 'functools', 'warnings'
    ]
    
    OPTIONAL_PACKAGES = [
        'MDAnalysis', 'plotly', 'statsmodels', 'sklearn'
    ]
    
    def __init__(self, 
                 sim_dir: str,
                 output_dir: Optional[str] = None,
                 model_name: str = "gemini-3-pro-preview",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 executor_timeout: int = 120,
                 enforce_sandbox: bool = True,
                 package_mode: str = 'strict',
                 max_refinement_attempts: int = 2,
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initialize the LAMMPS Analysis agent.
        
        Args:
            sim_dir: Directory containing simulation files
            output_dir: Directory for analysis results
            model_name: Model name to use
            api_key: API key for LLM provider
            base_url: Optional base URL for internal proxy
            executor_timeout: Timeout for script execution
            enforce_sandbox: Whether to enforce sandbox restrictions
            package_mode: 'strict', 'permissive', or 'dockerfile'
            max_refinement_attempts: Max attempts to refine failed analyses
            local_model: Deprecated, use base_url
            google_api_key: Deprecated, use api_key
        """
        # Validate paths
        self.sim_dir = Path(sim_dir).resolve()
        if not self.sim_dir.exists():
            raise ValueError(f"Simulation directory does not exist: {sim_dir}")
        
        self.output_dir = Path(output_dir).resolve() if output_dir else self.sim_dir / "analysis"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Validate package mode
        if package_mode not in self.PACKAGE_MODES:
            raise ValueError(f"package_mode must be one of {list(self.PACKAGE_MODES.keys())}")
        
        self.package_mode = package_mode
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Normalize deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="LAMMPSAnalysisAgent"
        )
        
        # Initialize model
        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError("API key required for internal proxy")
            
            self.logger.info(f"Using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.logger.info(f"Using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )
        
        self.generation_config = None
        
        # Initialize script executor
        self.executor = ScriptExecutor(
            timeout=executor_timeout,
            enforce_sandbox=enforce_sandbox,
            allow_unsafe_override=False
        )
        
        # Detect container environment
        score, indicators = check_security_sandbox_indicators()
        self.in_container = score >= 4
        
        # Storage
        self.input_files = {}
        self.output_files = {}
        self.analysis_code = {}
        self.required_packages = set()
        self.max_refinement_attempts = max_refinement_attempts
        self._sim_details_cache = None
        
        # Log initialization
        self.logger.info(f"LAMMPSAnalysisAgent initialized")
        self.logger.info(f"  Mode: {package_mode}")
        self.logger.info(f"  Container: {self.in_container}")
        self.logger.info(f"  Sim dir: {self.sim_dir}")
        self.logger.info(f"  Output dir: {self.output_dir}")

    def run_quality_check(self,
                          research_goal: str,
                          stage: str = "equilibration") -> Dict[str, Any]:
        """
        Run quality checks on the current simulation output.
    
        Analyzes thermo output, trajectory, and other available data
        to determine if the simulation is producing physically reasonable results.
    
        Args:
            research_goal: Research objective
            stage: Current simulation stage name
    
        Returns:
            Dictionary with:
                - status: "healthy", "warning", or "critical"
                - can_continue: bool
                - issues: list of issue dicts
                - recommendations: list of recommendation strings
                - quality_metrics: dict of computed metrics
                - assessment_summary: str
        """
        self.logger.info(f"Running quality check for stage: {stage}")
    
        try:
            # Step 1: Inventory available files
            self._inventory_files()
    
            # Step 2: Get simulation details
            sim_details = self._get_sim_details()
    
            # Step 3: Identify available output data
            output_data = self._identify_output_data()
    
            if not output_data:
                self.logger.warning("No output data found for quality check")
                return {
                    "status": "unknown",
                    "can_continue": True,
                    "issues": [{"severity": "warning", "description": "No output data found to check"}],
                    "recommendations": ["Verify simulation produced output files"],
                    "quality_metrics": {},
                    "assessment_summary": "No data available for quality assessment"
                }
    
            # Step 4: Generate quality check plan
            quality_plan = self._generate_quality_check_plan(
                research_goal=research_goal,
                sim_details=sim_details,
                output_data=output_data,
                stage=stage
            )
    
            checks = quality_plan.get("checks", [])
            if not checks:
                self.logger.warning("No quality checks could be planned")
                return {
                    "status": "unknown",
                    "can_continue": True,
                    "issues": [],
                    "recommendations": ["Could not plan quality checks with available data"],
                    "quality_metrics": {},
                    "assessment_summary": "No quality checks applicable"
                }
    
            self.logger.info(f"Running {len(checks)} quality checks")
    
            # Step 5: Generate and execute quality check code
            check_results = {}
    
            for check in checks:
                check_name = check.get("name", "unnamed_check")
                self.logger.info(f"  Running check: {check_name}")
    
                try:
                    # Generate code for this check
                    code = self._generate_quality_check_code(
                        check=check,
                        sim_details=sim_details,
                        output_data=output_data,
                        stage=stage
                    )
    
                    # Save code
                    code_path = self.output_dir / f"quality_check_{check_name}.py"
                    with open(code_path, 'w') as f:
                        f.write(code)
    
                    # Execute
                    result = self._execute_analysis_code(
                        code=code,
                        analysis_name=f"quality_{check_name}"
                    )
    
                    check_results[check_name] = result
    
                    status = result.get("status", "error")
                    if status == "success":
                        self.logger.info(f"    ✓ {check_name}: passed")
                    elif status == "error":
                        self.logger.warning(f"    ✗ {check_name}: execution failed")
    
                        # Try refinement
                        if self.max_refinement_attempts > 0:
                            refined_result = self._refine_failed_analysis(
                                code=code,
                                error_info=result,
                                analysis_name=f"quality_{check_name}",
                                sim_details=sim_details,
                                output_data=output_data
                            )
                            if refined_result.get("status") == "success":
                                check_results[check_name] = refined_result
                                self.logger.info(f"    ✓ {check_name}: passed after refinement")
    
                except Exception as e:
                    self.logger.error(f"    ✗ {check_name} failed: {e}")
                    check_results[check_name] = {
                        "status": "error",
                        "message": str(e)
                    }
    
            # Step 6: Synthesize results into assessment
            assessment = self._synthesize_quality_assessment(
                check_results=check_results,
                research_goal=research_goal,
                stage=stage,
                sim_details=sim_details
            )
    
            # Save assessment
            assessment_path = self.output_dir / f"quality_assessment_{stage}.json"
            with open(assessment_path, 'w') as f:
                json.dump(assessment, f, indent=2, default=str)
    
            self.logger.info(f"Quality assessment: {assessment.get('status', 'unknown')}")
    
            return assessment
    
        except Exception as e:
            self.logger.error(f"Quality check failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "unknown",
                "can_continue": True,
                "issues": [{"severity": "error", "description": f"Quality check failed: {e}"}],
                "recommendations": ["Manual review needed"],
                "quality_metrics": {},
                "assessment_summary": f"Quality check error: {e}"
            }
    
    
    def run_analysis(self, research_goal: str) -> Dict[str, Any]:
        """
        Run comprehensive analysis of simulation results.
    
        This is the main entry point for final analysis after simulation completes.
    
        Args:
            research_goal: Research objective
    
        Returns:
            Dictionary with:
                - status: "success", "partial", or "error"
                - results: dict of analysis name -> result dict
                - report_path: path to HTML report
                - figures: list of generated figure paths
        """
        self.logger.info("Running comprehensive simulation analysis")
    
        # Handle dockerfile mode
        if self.package_mode == 'dockerfile':
            return self._dockerfile_workflow(research_goal)
    
        try:
            # Step 1: Inventory files
            print(f"\n📊 Step 1: Inventorying simulation files...")
            self._inventory_files()
    
            # Step 2: Analyze simulation details
            print(f"📊 Step 2: Analyzing simulation setup...")
            sim_details = self._get_sim_details()
    
            if sim_details.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Could not analyze simulation: {sim_details.get('message', 'Unknown')}",
                    "results": {}
                }
    
            # Step 3: Identify output data
            print(f"📊 Step 3: Identifying output data...")
            output_data = self._identify_output_data()
    
            if not output_data:
                return {
                    "status": "error",
                    "message": "No simulation output data found",
                    "results": {}
                }
    
            print(f"   Found {len(output_data)} data sources")
            for name, info in output_data.items():
                print(f"     - {name}: {info.get('description', 'Unknown')}")
    
            # Step 4: Generate analysis plan
            print(f"📊 Step 4: Generating analysis plan...")
            analysis_plan = self._generate_analysis_plan(research_goal, sim_details, output_data)
    
            analyses = analysis_plan.get("analyses", [])
            if not analyses:
                return {
                    "status": "error",
                    "message": "Could not generate analysis plan",
                    "results": {}
                }
    
            print(f"   Planned {len(analyses)} analyses")
            for analysis in analyses:
                print(f"     - {analysis['name']}: {analysis.get('description', '')[:60]}")
    
            # Step 5: Execute analyses
            print(f"\n📊 Step 5: Executing analyses...")
            results = {}
    
            for analysis in analyses:
                name = analysis["name"]
                print(f"\n   🔬 {name}...")
    
                try:
                    # Generate code
                    code = self._generate_analysis_code(analysis, sim_details, output_data)
    
                    # Save code
                    code_path = self.output_dir / f"analysis_{name}.py"
                    with open(code_path, 'w') as f:
                        f.write(code)
    
                    # Execute
                    result = self._execute_analysis_code(code, name)
    
                    if result.get("status") == "success":
                        print(f"      ✓ Success")
                        results[name] = result
                    else:
                        concise = result.get("concise_error", result.get("message", "Unknown")[:120])
                        print(f"      ✗ Failed: {concise}")

                        # Try refinement
                        if self.max_refinement_attempts > 0:
                            print(f"      🔧 Attempting refinement...")
                            refined = self._refine_failed_analysis(
                                code=code,
                                error_info=result,
                                analysis_name=name,
                                sim_details=sim_details,
                                output_data=output_data
                            )
    
                            if refined.get("status") == "success":
                                print(f"      ✓ Succeeded after refinement")
                                refined["was_refined"] = True
                                results[name] = refined
                            else:
                                print(f"      ✗ Refinement also failed")
                                results[name] = result
                        else:
                            results[name] = result
    
                except Exception as e:
                    self.logger.error(f"Analysis {name} failed: {e}")
                    results[name] = {"status": "error", "message": str(e)}
    
            # Step 6: Generate report
            print(f"\n📊 Step 6: Generating report...")
    
            successful = sum(1 for r in results.values() if r.get("status") == "success")
            total = len(results)
            print(f"   {successful}/{total} analyses succeeded")
    
            report_path = None
            try:
                report_path = self._generate_final_report(research_goal, results, analysis_plan)
                print(f"   📄 Report: {report_path}")
            except Exception as e:
                self.logger.error(f"Report generation failed: {e}")
    
            # Collect figures
            figures = []
            for name, data in results.items():
                if data.get("status") == "success":
                    for key, value in data.items():
                        if isinstance(value, str) and value.endswith(('.png', '.jpg', '.jpeg')):
                            if os.path.exists(value):
                                figures.append(value)
    
            status = "success" if successful == total else "partial" if successful > 0 else "error"
    
            return {
                "status": status,
                "results": results,
                "report_path": report_path,
                "figures": figures,
                "analyses_planned": total,
                "analyses_succeeded": successful
            }
    
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "results": {}
            }

    # ============================================================================
    # HELPER METHODS FOR LLM CALLS
    # ============================================================================
    
    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON response from LLM."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            raise ValueError(f"Could not parse JSON: {e}")
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text response from LLM."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise
    
   
    def _ask_llm_to_interpret_outputs(self, output_commands: Dict[str, Dict[str, Any]], 
                                      script_excerpt: str) -> Dict[str, Dict[str, Any]]:
        """Use LLM to interpret what each output command produces."""
        
        commands_summary = []
        for cmd_id, cmd_info in output_commands.items():
            summary = {
                "id": cmd_id,
                "type": cmd_info["command_type"],
                "filename": cmd_info.get("filename", ""),
                "command": cmd_info.get("context", "")
            }
            
            if "referenced_computes" in cmd_info:
                summary["computes"] = {
                    cid: cinfo.get("type", "") for cid, cinfo in cmd_info["referenced_computes"].items()
                }
            
            commands_summary.append(summary)
        
        prompt = f"""
Analyze LAMMPS output commands and describe what data each produces.

LAMMPS OUTPUT COMMANDS:
{json.dumps(commands_summary, indent=2)}

SCRIPT CONTEXT:
{script_excerpt[:2000]}

For each output command, determine:
1. Physical/computational quantity being output
2. Type of analysis this enables
3. Clear description

Return JSON mapping command ID to data info:
{{
  "command_id": {{
    "data_type": "trajectory|time_series|correlation_function|...",
    "physical_quantity": "positions|energy|temperature|density|...",
    "description": "Clear description",
    "analysis_potential": "What analysis this enables",
    "is_time_series": true/false,
    "dimensionality": "scalar|vector|..."
  }}
}}

Return ONLY JSON.
"""
        
        try:
            return self._generate_json(prompt)  # ✅ Use helper
        except Exception as e:
            self.logger.error(f"LLM interpretation failed: {e}")
            return {}
    
    def _generate_analysis_plan(self, research_goal: str, sim_details: Dict[str, Any],
                             output_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive analysis plan."""
        params = sim_details.get('parameters', {})
        
        example_json = """{
      "analyses": [
        {
          "name": "thermodynamic_analysis",
          "description": "Analyze thermodynamic properties",
          "required_data": ["thermodynamics"],
          "outputs": ["temperature_plot.png", "energy_plot.png"],
          "properties_to_calculate": ["average temperature", "average energy"],
          "importance": "high",
          "connection_to_research_goal": "Validates equilibration"
        }
      ],
      "summary": "Brief summary"
    }"""
        available_data_list = [
              {"name": name, "description": info["description"], "analysis_potential": info.get("analysis_potential", "")}
              for name, info in output_data.items()
        ]
        available_data_str = json.dumps(available_data_list, indent=2)

        prompt = f"""
I need to analyze a LAMMPS simulation. Generate a comprehensive analysis plan.

RESEARCH GOAL:
{research_goal}

SIMULATION DETAILS:
- Type: {sim_details.get('summary', 'Unknown')}
- Ensemble: {params.get('ensemble', 'Unknown')}
- Temperature: {params.get('temperature', 'Unknown')}
- Simulation time: {params.get('simulation_time', 'Unknown')}

AVAILABLE OUTPUT DATA:
{available_data_str}

Create analysis plan with this structure:
{example_json}

IMPORTANT:
- Include only analyses with available data
- Be specific about calculations
- Connect to research goal

Return ONLY JSON.
"""
        
        try:
            analysis_plan = self._generate_json(prompt)  # ✅ Use helper
            
            # Filter analyses with available data
            filtered_analyses = []
            for analysis in analysis_plan.get("analyses", []):
                required_data = set(analysis.get("required_data", []))
                available_data = set(output_data.keys())
                
                if required_data.issubset(available_data):
                    filtered_analyses.append(analysis)
                else:
                    missing = required_data - available_data
                    self.logger.warning(f"Skipping '{analysis['name']}' - missing: {missing}")
            
            analysis_plan["analyses"] = filtered_analyses
            return analysis_plan
            
        except Exception as e:
            self.logger.error(f"Error generating analysis plan: {e}")
            return {
                "analyses": [],
                "summary": "Fallback plan due to error"
            }
    
    def _generate_analysis_code(self, analysis: Dict[str, Any],
                             sim_details: Dict[str, Any],
                             output_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate Python code for a specific analysis.
        """
        # Get required data files with full metadata
        data_files = {}
        data_metadata = {}
        
        for data_type in analysis.get("required_data", []):
            if data_type in output_data:
                data_info = output_data[data_type]
                
                # Use all_files if it's a series, otherwise use single file
                if data_info.get("all_files"):
                    data_files[data_type] = data_info["all_files"]  # Pass list of files
                else:
                    data_files[data_type] = data_info["file"]  # Pass single file
                
                # Provide rich metadata to help code generation
                data_metadata[data_type] = {
                    "file": data_info["file"],
                    "all_files": data_info.get("all_files"),
                    "description": data_info.get("description", ""),
                    "format": data_info.get("format", "dat"),
                    "physical_quantity": data_info.get("physical_quantity", "unknown"),
                    "dimensionality": data_info.get("dimensionality", "unknown"),
                    "is_series": data_info.get("is_series", False),
                    "num_files": data_info.get("num_files", 1)
                }
        
        # Extract parameters
        params = sim_details.get('parameters', {})
        
        # Get available packages and create constraint
        available_packages = self._get_available_packages()
        
        if self.package_mode == 'strict':
            package_constraint = f"""
    CRITICAL PACKAGE CONSTRAINT:
    You MUST use ONLY these pre-installed packages:
    {', '.join(available_packages)}
    DO NOT import or use any packages not in this list.
    If you need functionality not available, implement it using available packages or numpy/scipy primitives.
    """
        else:
            package_constraint = """
    PACKAGE PREFERENCES:
    Prefer standard packages: numpy, scipy, matplotlib, pandas, seaborn
    You may use additional packages if necessary for specialized analysis.
    """
        
        # Create example code template
        example_template = """import numpy as np
    import matplotlib.pyplot as plt
    import json
    import os
    
    def main(data_files, output_dir):
        \"\"\"
        CRITICAL: ALL file saves must use output_dir parameter!
        NEVER use relative paths like './file.png' or 'file.csv'
        ALWAYS use: os.path.join(output_dir, 'file.png')
        \"\"\"
        results = {}
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Load data
            data = np.loadtxt(data_files['example'])
            
            # Calculate properties
            mean_value = np.mean(data)
            results['mean'] = float(mean_value)
            
            # Create plots - CORRECT PATH USAGE
            plt.figure()
            plt.plot(data)
            plt.xlabel('Time')
            plt.ylabel('Value')
            
            # CORRECT: Use os.path.join with output_dir
            plot_path = os.path.join(output_dir, 'plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store ABSOLUTE path in results
            results['plot'] = plot_path
            
            # Save CSV - CORRECT PATH USAGE
            csv_path = os.path.join(output_dir, 'data.csv')
            np.savetxt(csv_path, data, delimiter=',')
            results['csv_file'] = csv_path
            
            results['status'] = 'success'
            return results
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    if __name__ == "__main__":
        data_files = DATA_FILES_PLACEHOLDER
        output_dir = "OUTPUT_DIR_PLACEHOLDER"
        
        results = main(data_files, output_dir)
        print(json.dumps(results))
    """
        
        # Build the full prompt
        prompt = f"""
    Generate complete Python code for analyzing molecular dynamics simulation data.
    
    {package_constraint}
    
    ANALYSIS GOAL: {analysis["description"]}
    
    AVAILABLE DATA:
    {json.dumps(data_metadata, indent=2)}
    
    SIMULATION DETAILS:
    - Type: {sim_details.get('summary', 'Unknown')}
    - Ensemble: {params.get('ensemble', 'Unknown')}
    - Temperature: {params.get('temperature', 'Unknown')}
    - Timestep: {params.get('timestep', 'Unknown')} fs
    - Run steps: {params.get('run_steps', 'Unknown')}
    
    OUTPUTS TO GENERATE:
    {", ".join(analysis.get("outputs", []))}
    
    PROPERTIES TO CALCULATE:
    {", ".join(analysis.get("properties_to_calculate", []))}
    
    ═══════════════════════════════════════════════════════════════
    🚨 CRITICAL FILE PATH REQUIREMENTS 🚨
    ═══════════════════════════════════════════════════════════════
    
    YOU WILL BE PENALIZED FOR USING RELATIVE PATHS!
    
    ✅ CORRECT - Use output_dir parameter:
       plot_path = os.path.join(output_dir, "my_plot.png")
       plt.savefig(plot_path)
       results['plot'] = plot_path  # Already absolute!
    
    ❌ WRONG - DO NOT use relative paths:
       plt.savefig("./my_plot.png")  # ❌ WRONG
       plt.savefig("my_plot.png")    # ❌ WRONG
       results['plot'] = "./my_plot.png"  # ❌ WRONG
    
    ✅ CORRECT pattern for ALL file operations:
       output_path = os.path.join(output_dir, "filename.ext")
       save_function(output_path)
       results['key'] = output_path
    
    ═══════════════════════════════════════════════════════════════
    
    DATA FILE HANDLING:
    - Files may have comment lines starting with #
    - For temperature series: data_files['type'] will be a LIST
    - For single files: data_files['type'] will be a STRING
    - Always check: if isinstance(data_files['type'], list): ...
    - Validate file paths: os.path.exists(filepath)
    
    CODE STRUCTURE REQUIREMENTS:
    {example_template}
    
    IMPORTANT:
    - Print results as JSON at the end using json.dumps()
    - Include 'status' key in results
    - Handle files that may not be perfectly formatted
    - ALL output file paths MUST use os.path.join(output_dir, ...)
    - Return ABSOLUTE paths in results dictionary
    
    CRITICAL RULES FOR OUTPUT:
    1. The VERY FIRST LINE of your output must be an import statement (e.g., "import numpy as np")
    2. Do NOT start with "python", "#!/usr/bin/env python", or any shebang line
    3. Do NOT include any shell commands
    4. Do NOT wrap code in markdown code blocks
    5. Do NOT include any text before or after the Python code
    6. Always include matplotlib.use('Agg') before importing pyplot
    7. Always print results as JSON on the last line using print(json.dumps(results))
    8. The code will be executed directly by a Python interpreter - output ONLY valid Python
    """
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text
            
            # Clean code - remove any markdown formatting
            code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
            code = code.strip()
           
            code = self._clean_generated_code(code)

            # Replace placeholders
            code = code.replace('DATA_FILES_PLACEHOLDER', json.dumps(data_files))
            code = code.replace('OUTPUT_DIR_PLACEHOLDER', str(self.output_dir))
            code = code.replace('"OUTPUT_DIR_PLACEHOLDER"', f'"{self.output_dir}"')
            
            return code
            
        except Exception as e:
            self.logger.error(f"Error generating code for {analysis['name']}: {e}")
            # Minimal error-reporting script
            error_script = f"""import json
    def main(data_files, output_dir):
        return {{
            "status": "error",
            "message": "Failed to generate analysis code: {str(e)}",
            "analysis": "{analysis['name']}"
        }}
    if __name__ == "__main__":
        data_files = {json.dumps(data_files)}
        output_dir = "{self.output_dir}"
        results = main(data_files, output_dir)
        print(json.dumps(results))
    """
            return error_script

    def _generate_quality_check_plan(self,
                                      research_goal: str,
                                      sim_details: Dict[str, Any],
                                      output_data: Dict[str, Dict[str, Any]],
                                      stage: str) -> Dict[str, Any]:
        """Generate a plan for quality checks based on simulation stage and available data."""
        
        params = sim_details.get('parameters', {})
        
        # Build data descriptions OUTSIDE the f-string
        available_data_list = [
            {"name": name, "description": info.get("description", ""), "data_type": info.get("data_type", "")}
            for name, info in output_data.items()
        ]
        available_data_str = json.dumps(available_data_list, indent=2)
        
        prompt = f"""
    Generate a quality check plan for a LAMMPS molecular dynamics simulation.
    
    RESEARCH GOAL: {research_goal}
    SIMULATION STAGE: {stage}
    
    SIMULATION DETAILS:
    - Ensemble: {params.get('ensemble', 'Unknown')}
    - Temperature target: {params.get('temperature', 'Unknown')} K
    - Pressure target: {params.get('pressure', 'Unknown')} atm
    - Timestep: {params.get('timestep', 'Unknown')} fs
    
    AVAILABLE DATA:
    {available_data_str}
    
    Generate quality checks appropriate for the "{stage}" stage.
    
    For equilibration stages, check:
    - Temperature stability (within +/-5% of target)
    - Pressure stability (reasonable fluctuations)
    - Energy convergence (no drift)
    - Density (if NPT, should be physically reasonable)
    
    For production stages, also check:
    - Statistical stationarity
    - Correlation times
    - Sampling adequacy
    
    Return JSON:
    {{
        "checks": [
            {{
                "name": "check_name",
                "description": "What this checks",
                "required_data": ["data_type_name"],
                "metrics_to_compute": ["metric1", "metric2"],
                "thresholds": {{
                    "metric1": {{"min": 290, "max": 310, "description": "Temperature range"}}
                }},
                "importance": "critical|important|informational"
            }}
        ],
        "critical_thresholds": {{
            "description": "What would cause a critical failure"
        }},
        "stage_notes": "Notes specific to this stage"
    }}
    """
        
        try:
            plan = self._generate_json(prompt)
            
            # Filter checks with available data
            available_checks = []
            for check in plan.get("checks", []):
                required = set(check.get("required_data", []))
                available = set(output_data.keys())
                
                if required.issubset(available):
                    available_checks.append(check)
                else:
                    missing = required - available
                    self.logger.warning(f"Skipping check '{check.get('name')}' - missing: {missing}")
            
            plan["checks"] = available_checks
            return plan
            
        except Exception as e:
            self.logger.error(f"Error generating quality plan: {e}")
            return {"checks": [], "critical_thresholds": {}, "stage_notes": ""}   
        
    def _synthesize_quality_assessment(self,
                                        check_results: Dict[str, Any],
                                        research_goal: str,
                                        stage: str,
                                        sim_details: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize individual check results into an overall quality assessment."""
        
        # Collect metrics from all checks
        metrics = {}
        failed_checks = []
        
        for check_name, result in check_results.items():
            if result.get("status") == "success":
                check_metrics = {}
                for key, value in result.items():
                    if key not in ["status", "message", "execution_time", "raw_output",
                                 "was_refined", "refinement_attempts", "stderr", "stdout"]:
                        if isinstance(value, (int, float, bool)):
                            check_metrics[key] = value
                
                if check_metrics:
                    metrics[check_name] = check_metrics
            else:
                failed_checks.append({
                    "check": check_name,
                    "error": result.get("message", "Unknown error")
                })
        
        # Build JSON strings OUTSIDE f-string
        metrics_str = json.dumps(metrics, indent=2)
        failed_str = json.dumps(failed_checks, indent=2)
        params_str = json.dumps(sim_details.get('parameters', {}), indent=2)
        
        prompt = f"""
    Synthesize these quality check results into an overall assessment.
    
    RESEARCH GOAL: {research_goal}
    SIMULATION STAGE: {stage}
    
    CHECK RESULTS:
    {metrics_str}
    
    FAILED CHECKS:
    {failed_str}
    
    SIMULATION PARAMETERS:
    {params_str}
    
    Assess the overall simulation quality. Determine:
    1. Overall status: "healthy" (all good), "warning" (minor issues), or "critical" (physics wrong)
    2. Whether the simulation can continue to the next stage
    3. Specific issues found
    4. Recommendations for fixes
    
    Return JSON:
    {{
        "status": "healthy|warning|critical",
        "can_continue": true/false,
        "issues": [
            {{
                "severity": "critical|warning|info",
                "description": "Clear description of the issue",
                "metric": "which metric flagged this",
                "value": 0.0,
                "expected_range": "what the value should be"
            }}
        ],
        "recommendations": [
            "Specific actionable recommendation"
        ],
        "assessment_summary": "One paragraph summary",
        "next_action": "continue|adjust_parameters|restart|investigate"
    }}
    """
        
        try:
            assessment = self._generate_json(prompt)
            
            assessment['quality_metrics'] = metrics
            assessment['stage'] = stage
            assessment['failed_checks'] = failed_checks
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error synthesizing assessment: {e}")
            return {
                "status": "unknown",
                "can_continue": True,
                "issues": [],
                "recommendations": ["Manual review recommended - assessment synthesis failed"],
                "quality_metrics": metrics,
                "assessment_summary": f"Assessment synthesis failed: {e}",
                "next_action": "investigate",
                "failed_checks": failed_checks
            }

    def _generate_final_report(self, research_goal: str, 
                             results: Dict[str, Dict[str, Any]],
                             analysis_plan: Dict[str, Any]) -> str:
        """
        Generate a final HTML report summarizing all analysis results.
        """
        report_path = self.output_dir / "md_analysis_report.html"
        
        # Gather successful analyses and their outputs
        successful_analyses = {name: data for name, data in results.items() 
                              if data.get('status') == 'success'}
        
        # Find all figures generated
        figures = []
        for name, data in successful_analyses.items():
            for key, value in data.items():
                if isinstance(value, str) and value.endswith(('.png', '.jpg', '.jpeg')):
                    if os.path.exists(value):
                        figures.append({
                            "path": value,
                            "title": f"{key.replace('_', ' ').title()}",
                            "analysis": name.replace('_', ' ').title()
                        })
        
        # Find all CSV files generated
        csv_files = []
        for name, data in successful_analyses.items():
            for key, value in data.items():
                if isinstance(value, str) and value.endswith('.csv'):
                    if os.path.exists(value):
                        csv_files.append({
                            "path": value,
                            "title": f"{key.replace('_', ' ').title()}",
                            "analysis": name.replace('_', ' ').title()
                        })
        
        try:
            # Create analyses summary
            analyses_summary = []
            for name, data in results.items():
                analysis_info = {
                    "name": name.replace('_', ' ').title(),
                    "status": "✓ Success" if data.get('status') == 'success' else "✗ Failed"
                }
                
                if data.get('status') == 'success':
                    # Extract key findings
                    findings = []
                    for key, value in data.items():
                        if key not in ['status', 'analysis_name', 'execution_time', 'message', 'raw_output', 'was_refined', 'refinement_attempts']:
                            if isinstance(value, (int, float)):
                                findings.append(f"{key.replace('_', ' ').title()}: {value:.4g}")
                    
                    analysis_info["findings"] = findings
                    analysis_info["execution_time"] = data.get('execution_time', 'N/A')
                    
                    if data.get('was_refined'):
                        analysis_info["note"] = f"Succeeded after {data.get('refinement_attempts', 0)} refinement(s)"
                else:
                    analysis_info["error"] = data.get('message', 'Unknown error')
                    if data.get('refinement_attempts', 0) > 0:
                        analysis_info["note"] = f"Failed after {data.get('refinement_attempts')} refinement attempt(s)"
                
                analyses_summary.append(analysis_info)
            
            # Convert figure paths to relative paths for HTML
            figure_info = []
            for fig in figures:
                rel_path = os.path.relpath(fig["path"], self.output_dir)
                figure_info.append({
                    "path": rel_path,
                    "title": fig["title"],
                    "analysis": fig["analysis"]
                })
            
            # Generate HTML report using LLM
            prompt = f"""
    Create a professional HTML report for molecular dynamics simulation analysis.
    
    RESEARCH GOAL:
    {research_goal}
    
    ANALYSES PERFORMED:
    {json.dumps(analyses_summary, indent=2)}
    
    FIGURES AVAILABLE:
    {json.dumps(figure_info, indent=2)}
    
    DATA FILES:
    {len(csv_files)} CSV files generated
    
    Create an HTML report with:
    1. Professional header with title "Molecular Dynamics Analysis Report"
    2. Executive summary connecting results to research goal
    3. Detailed sections for each analysis with:
       - Analysis description
       - Key findings (numerical results)
       - Embedded figures with captions
    4. Conclusions and recommendations
    5. Data files section listing available CSVs
    
    Use modern CSS for styling (or Bootstrap CDN). Make it visually appealing and easy to read.
    Include proper figure sizing and responsive layout.
    
    Return ONLY complete HTML code with no markdown or explanations.
    """
            
            response = self.model.generate_content(prompt)
            html_content = response.text
            
            # Clean up response - remove markdown if present
            html_content = re.sub(r'```html\s*', '', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'```\s*', '', html_content, flags=re.MULTILINE)
            html_content = html_content.strip()
            
            # Save report
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {report_path}")
            
            # Try to generate PDF if wkhtmltopdf available
            pdf_path = self.output_dir / "md_analysis_report.pdf"
            if shutil.which("wkhtmltopdf"):
                try:
                    subprocess.run(
                        ["wkhtmltopdf", str(report_path), str(pdf_path)],
                        check=True, 
                        capture_output=True
                    )
                    self.logger.info(f"PDF report generated: {pdf_path}")
                    return str(pdf_path)
                except Exception as e:
                    self.logger.warning(f"Could not generate PDF: {e}")
            
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}", exc_info=True)
            
            # Fallback HTML report
            fallback_html = self._generate_fallback_report(research_goal, results, figures)
            with open(report_path, 'w') as f:
                f.write(fallback_html)
            
            return str(report_path)

    def _generate_fallback_report(self, research_goal: str, 
                                 results: Dict[str, Dict[str, Any]],
                                 figures: List[Dict[str, str]]) -> str:
        """Generate a simple fallback HTML report."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '    <title>MD Analysis Report</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }',
            '        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }',
            '        h2 { color: #3498db; margin-top: 30px; }',
            '        .success { color: #27ae60; font-weight: bold; }',
            '        .error { color: #e74c3c; font-weight: bold; }',
            '        .analysis-section { background: #ecf0f1; padding: 15px; margin: 15px 0; border-radius: 5px; }',
            '        img { max-width: 100%; height: auto; margin: 15px 0; border: 1px solid #bdc3c7; }',
            '        .figure-caption { font-style: italic; color: #7f8c8d; margin-top: -10px; margin-bottom: 20px; }',
            '    </style>',
            '</head>',
            '<body>',
            '    <h1>Molecular Dynamics Analysis Report</h1>',
            '    ',
            '    <div class="analysis-section">',
            '        <h2>Research Goal</h2>',
            f'        <p>{research_goal}</p>',
            '    </div>',
            '    ',
            '    <h2>Analysis Results</h2>'
        ]
        
        for name, data in results.items():
            status_class = 'success' if data.get('status') == 'success' else 'error'
            html_parts.extend([
                '    <div class="analysis-section">',
                f'        <h3>{name.replace("_", " ").title()}</h3>',
                f'        <p>Status: <span class="{status_class}">{data.get("status", "unknown")}</span></p>'
            ])
            
            if data.get('status') == 'success':
                html_parts.append(f'        <p>Execution time: {data.get("execution_time", "N/A")}</p>')
            else:
                html_parts.append(f'        <p>Error: {data.get("message", "Unknown error")}</p>')
            
            html_parts.append('    </div>')
        
        # Add figures
        if figures:
            html_parts.append('\n    <h2>Generated Figures</h2>')
            for fig in figures:
                rel_path = os.path.relpath(fig["path"], self.output_dir)
                html_parts.extend([
                    '    <div>',
                    f'        <img src="{rel_path}" alt="{fig["title"]}">',
                    f'        <p class="figure-caption">{fig["title"]} ({fig["analysis"]})</p>',
                    '    </div>'
                ])
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    # ==================================================================================
    # DOCKERFILE MODE
    # ==================================================================================
    
    def _dockerfile_workflow(self, research_goal: str) -> Dict[str, Any]:
        """
        Generate a custom Dockerfile and provide instructions to user.
        """
        print(f"\n📦 Dockerfile Generation Mode")
        print(f"{'='*60}")
        print("Analyzing requirements to generate custom container...")
        
        # Do the analysis planning steps to understand requirements
        self._inventory_files()
        sim_details = self._analyze_lammps_input()
        if sim_details["status"] != "success":
            return {"status": "error", "message": "Cannot generate Dockerfile without valid simulation"}
        
        output_data = self._identify_output_data()
        analysis_plan = self._generate_analysis_plan(research_goal, sim_details, output_data)
        
        # Generate code for all analyses to understand package requirements
        for analysis in analysis_plan['analyses']:
            code = self._generate_analysis_code(analysis, sim_details, output_data)
            self.analysis_code[analysis['name']] = code
            self._extract_required_packages(code)
        
        # Generate the Dockerfile
        dockerfile_path = self._generate_custom_dockerfile(self.required_packages, research_goal)
        build_script_path = self._generate_build_script(dockerfile_path)
        
        # Provide instructions
        instructions = f"""
    {'='*60}
    📦 Custom Dockerfile Generated
    {'='*60}
    A custom Dockerfile has been generated at:
      {dockerfile_path}
    
    This Dockerfile includes all required packages:
      {', '.join(sorted(self.required_packages))}
    
    TO USE THIS ENVIRONMENT:
    1. Review the Dockerfile (optional)
    2. Build the container:
       sbatch {build_script_path.name}
       
    3. Wait for the build to complete (~20-30 minutes)
    4. Re-run your analysis with package_mode='strict':
       
       agent = MDAnalysisAgent(
           sim_dir="{self.sim_dir}",
           package_mode='strict'
       )
       agent.run_analysis(research_goal)
    {'='*60}
        """
        
        print(instructions)
        
        return {
            "status": "dockerfile_generated",
            "dockerfile_path": str(dockerfile_path),
            "build_script_path": str(build_script_path),
            "required_packages": sorted(list(self.required_packages)),
            "instructions": instructions
        }
    
    def _generate_custom_dockerfile(self, required_packages: Set[str], research_goal: str) -> Path:
        """Generate a custom Dockerfile based on analysis requirements."""
        
        # Build package list for pip install
        pkg_list = ' '.join(sorted(required_packages))
        
        # Create Dockerfile content
        dockerfile_lines = [
            '# Custom Dockerfile for MD Analysis',
            f'# Generated for: {research_goal}',
            f'# Required packages: {", ".join(sorted(required_packages))}',
            '',
            'FROM python:3.12-slim',
            '',
            '# Install system dependencies',
            'RUN apt-get update && apt-get install -y --no-install-recommends \\',
            '    build-essential \\',
            '    gcc \\',
            '    gfortran \\',
            '    libopenblas-dev \\',
            '    liblapack-dev \\',
            '    libgomp1 \\',
            '    git \\',
            '    libgl1 \\',
            '    libglib2.0-0 \\',
            '    && rm -rf /var/lib/apt/lists/*',
            '',
            'WORKDIR /app',
            '',
            '# Upgrade pip',
            'RUN pip install --upgrade pip',
            '',
            '# Install core scientific stack',
            'RUN pip install --no-cache-dir \\',
            '    numpy \\',
            '    scipy \\',
            '    matplotlib \\',
            '    pandas \\',
            '    seaborn',
            '',
            '# Install analysis-specific packages',
            f'RUN pip install --no-cache-dir {pkg_list}' if pkg_list else '# No additional packages needed',
            '',
            '# Install Google Generative AI',
            'RUN pip install --no-cache-dir google-generativeai',
            '',
            '# Copy SciLink',
            'COPY . /app/scilink',
            'WORKDIR /app/scilink',
            'RUN pip install --no-cache-dir -e .',
            '',
            '# Create non-root user',
            'RUN useradd -m -u 1000 scilink && \\',
            '    chown -R scilink:scilink /app',
            '',
            'USER scilink',
            '',
            'ENV RUNNING_IN_CONTAINER=true \\',
            '    PYTHONUNBUFFERED=1 \\',
            '    PYTHONDONTWRITEBYTECODE=1 \\',
            '    MPLBACKEND=Agg',
            '',
            'CMD ["/bin/bash"]'
        ]
        
        dockerfile_content = '\n'.join(dockerfile_lines)
        
        # Save Dockerfile
        dockerfile_path = self.output_dir / "Dockerfile.analysis"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        self.logger.info(f"Custom Dockerfile generated: {dockerfile_path}")
        return dockerfile_path
    
    def _generate_build_script(self, dockerfile_path: Path) -> Path:
        """Generate an sbatch script to build the custom container on HPC."""
        
        script_lines = [
            '#!/bin/bash',
            '#SBATCH -A CHANGEME',
            '#SBATCH -t "00:30:00"',
            '#SBATCH -N 1',
            '#SBATCH -p short',
            '#SBATCH -J scilink_analysis_build',
            '#SBATCH -o scilink_analysis_build_%j.out',
            '#SBATCH -e scilink_analysis_build_%j.err',
            '',
            '# Load modules',
            'source /etc/profile.d/modules.sh',
            'module purge',
            'module load apptainer/1.2.4',
            '',
            '# Set up scratch space',
            'export APPTAINER_TMPDIR=/scratch/$USER/APPTAINER',
            'export APPTAINER_CACHEDIR=/scratch/$USER/APPTAINER',
            '',
            'rm -rf $APPTAINER_TMPDIR',
            'rm -rf $APPTAINER_CACHEDIR',
            'mkdir -p $APPTAINER_TMPDIR',
            'mkdir -p $APPTAINER_CACHEDIR',
            '',
            'echo "Converting Dockerfile to Apptainer definition..."',
            f'spython recipe --force --parser docker --writer singularity {dockerfile_path.name} scilink_analysis.def',
            '',
            'echo "Building Apptainer container..."',
            'apptainer build --force --fakeroot scilink_analysis.sif scilink_analysis.def',
            '',
            'echo "✓ Container built successfully: scilink_analysis.sif"',
            'echo "  You can now run your analysis with package_mode=\'strict\'"'
        ]
        
        build_script_content = '\n'.join(script_lines)
        
        script_path = self.output_dir / "build_scilink_analysis.sbatch"
        with open(script_path, 'w') as f:
            f.write(build_script_content)
        
        os.chmod(script_path, 0o755)
        
        self.logger.info(f"Build script generated: {script_path}")
        return script_path

    def _get_sim_details(self) -> Dict[str, Any]:
        """Get simulation details, with caching."""
        if self._sim_details_cache is None:
            self._sim_details_cache = self._analyze_lammps_input()
        return self._sim_details_cache
    
    
    def _generate_quality_check_code(self,
                                      check: Dict[str, Any],
                                      sim_details: Dict[str, Any],
                                      output_data: Dict[str, Dict[str, Any]],
                                      stage: str) -> str:
        """
        Generate Python code for a specific quality check.
        Uses the same pattern as _generate_analysis_code but focused on quality metrics.
        """
        # Get data files
        data_files = {}
        for data_type in check.get("required_data", []):
            if data_type in output_data:
                data_info = output_data[data_type]
                if data_info.get("all_files"):
                    data_files[data_type] = data_info["all_files"]
                else:
                    data_files[data_type] = data_info["file"]
        
        params = sim_details.get('parameters', {})
        available_packages = self._get_available_packages()
        
        if self.package_mode == 'strict':
            package_constraint = f"""
    CRITICAL: Use ONLY these packages: {', '.join(available_packages)}
    """
        else:
            package_constraint = "Use standard scientific Python packages."
        
        thresholds = check.get("thresholds", {})
        thresholds_str = json.dumps(thresholds, indent=2) if thresholds else "Use standard physical chemistry thresholds"

        data_file_info = {
            k: {
                "file": v if isinstance(v, str) else v[0],
                "description": output_data.get(k, {}).get("description", "")
            }
            for k, v in data_files.items()
        }
        data_file_info_str = json.dumps(data_file_info, indent=2)

        prompt = f"""
    Generate Python code for a simulation quality check.
    
    {package_constraint}
    
    CHECK: {check.get('name', 'unnamed')}
    DESCRIPTION: {check.get('description', '')}
    STAGE: {stage}
    
    WHAT TO CHECK:
    {data_file_info_str}

    PASS/FAIL THRESHOLDS:
    {thresholds_str}
    
    AVAILABLE DATA:
    data_file_info_str

    SIMULATION PARAMETERS:
    - Ensemble: {params.get('ensemble', 'Unknown')}
    - Temperature: {params.get('temperature', 'Unknown')} K
    - Pressure: {params.get('pressure', 'Unknown')} atm
    - Timestep: {params.get('timestep', 'Unknown')} fs
    
    CODE REQUIREMENTS:
    1. Define main(data_files, output_dir) function
    2. Return dict with:
       - "status": "success" or "error"
       - Computed metrics as key-value pairs (floats)
       - "pass": true/false for each threshold check
    3. Save any diagnostic plots to output_dir using os.path.join(output_dir, filename)
    4. Handle missing/malformed data gracefully
    5. Print results as JSON
    
    For thermo data (log.lammps format):
    - Lines starting with numbers are data rows
    - Skip comment lines starting with #
    - Common columns: Step, Temp, Press, Density, PotEng, KinEng, TotEng
    
    CRITICAL RULES FOR OUTPUT:
    1. The VERY FIRST LINE must be an import statement (e.g., "import numpy as np")
    2. Do NOT start with "python", "#!/usr/bin/env python", or any shebang line
    3. Do NOT include any shell commands or markdown
    4. Always include matplotlib.use('Agg') before importing pyplot
    5. Always print results as JSON on the last line
    6. Return numeric metrics as floats in the results dict
    7. The code will be executed directly by a Python interpreter - output ONLY valid Python
    """
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text
            
            # Clean markdown
            code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
            code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
            code = code.strip()

            code = self._clean_generated_code(code)

            # Replace placeholders
            code = code.replace('DATA_FILES_PLACEHOLDER', json.dumps(data_files))
            code = code.replace('OUTPUT_DIR_PLACEHOLDER', str(self.output_dir))
            code = code.replace('"OUTPUT_DIR_PLACEHOLDER"', f'"{self.output_dir}"')
            
            return code
            
        except Exception as e:
            self.logger.error(f"Failed to generate quality check code: {e}")
            return f"""import json
    def main(data_files, output_dir):
        return {"status": "error", "message": f"Code generation failed: {str(e)}"} 
    if __name__ == "__main__":
        data_files = {json.dumps(data_files)}
        output_dir = "{self.output_dir}"
        results = main(data_files, output_dir)
        print(json.dumps(results))
    """
    
    def _execute_analysis_code(self,
                                code: str,
                                analysis_name: str) -> Dict[str, Any]:
        """
        Execute analysis code using the ScriptExecutor.
        """
        self.logger.info(f"Executing analysis: {analysis_name}")
        
        try:
            # Save code to file
            code_path = self.output_dir / f"{analysis_name}.py"
            with open(code_path, 'w') as f:
                f.write(code)
            
            # Execute using ScriptExecutor
            exec_result = self.executor.execute_script(
                script_content=code,
                working_dir=str(self.output_dir)
            )
            
            exec_status = exec_result.get("status")
            self.logger.info(f"Executor status for {analysis_name}: {exec_status}")
            
            if exec_status == "success":
                stdout = exec_result.get("stdout", "")
                
                result = self._extract_json_from_output(stdout)
                
                if result:
                    result["execution_time"] = exec_result.get("execution_time", 0)
                    return result
                else:
                    return {
                        "status": "success",
                        "message": "Code executed but no JSON output found",
                        "raw_output": stdout[:2000]
                    }
            else:
                # The executor puts everything in "message" - extract the useful parts
                raw_message = exec_result.get("message", "")
                stderr = exec_result.get("stderr", "")
                stdout = exec_result.get("stdout", "")
                
                # Parse stderr out of the message if it's embedded
                error_message = raw_message
                
                # Extract the actual Python traceback
                traceback_text = ""
                
                # Check in message first (executor often embeds stderr here)
                if "Traceback" in raw_message:
                    tb_start = raw_message.find("Traceback")
                    traceback_text = raw_message[tb_start:]
                elif "Traceback" in stderr:
                    tb_start = stderr.find("Traceback")
                    traceback_text = stderr[tb_start:]
                elif "Traceback" in stdout:
                    tb_start = stdout.find("Traceback")
                    traceback_text = stdout[tb_start:]
                
                # Extract just the final error line for a concise message
                concise_error = ""
                if traceback_text:
                    tb_lines = traceback_text.strip().split('\n')
                    # Last line is usually the actual error
                    for line in reversed(tb_lines):
                        line = line.strip()
                        if line and not line.startswith("File") and not line.startswith("Traceback"):
                            concise_error = line
                            break
                
                # Build comprehensive error for the updater
                if concise_error:
                    full_error = f"{concise_error}\n\nFull traceback:\n{traceback_text[-2000:]}"
                elif traceback_text:
                    full_error = traceback_text[-2000:]
                elif raw_message:
                    full_error = raw_message[-2000:]
                else:
                    full_error = "Script failed with no error output captured"
                
                self.logger.error(f"Analysis {analysis_name} failed: {concise_error or full_error[:200]}")
                
                return {
                    "status": "error",
                    "message": full_error,
                    "stderr": stderr[:2000] if stderr else traceback_text[:2000],
                    "stdout": stdout[:2000],
                    "concise_error": concise_error
                }
            
        except Exception as e:
            self.logger.error(f"Execution failed for {analysis_name}: {e}")
            import traceback
            tb = traceback.format_exc()
            return {
                "status": "error",
                "message": f"{str(e)}\n{tb}",
                "stderr": tb,
                "concise_error": str(e)
            }    
    
    def _extract_json_from_output(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Extract the last JSON object from stdout."""
        # Try to find JSON in stdout (code prints results as JSON)
        lines = stdout.strip().split('\n')
        
        # Try last line first
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON anywhere
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', stdout, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _refine_failed_analysis(self,
                                 code: str,
                                 error_info: Dict[str, Any],
                                 analysis_name: str,
                                 sim_details: Dict[str, Any],
                                 output_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Attempt to refine failed analysis code using LAMMPSAnalysisUpdater.
        """
        from .lammps_analysis_updater import LAMMPSAnalysisUpdater
    
        # Build data_files dict from output_data
        data_files = {}
        for name, info in output_data.items():
            if info.get("all_files"):
                data_files[name] = info["all_files"]
            else:
                data_files[name] = info.get("file", "")
    
        available_packages = self._get_available_packages()
    
        for attempt in range(1, self.max_refinement_attempts + 1):
            self.logger.info(f"Refinement attempt {attempt}/{self.max_refinement_attempts} for {analysis_name}")
    
            try:
                updater = LAMMPSAnalysisUpdater(
                    api_key=self.model.api_key if hasattr(self.model, 'api_key') else None,
                    model_name=self.model.model if hasattr(self.model, 'model') else "gemini-3-pro-preview",
                    base_url=self.model.base_url if hasattr(self.model, 'base_url') else None
                )
    
                # Step 1: Analyze the failure
                failure_analysis = updater.analyze_failure(
                    analysis_name=analysis_name,
                    generated_code=code,
                    error_result=error_info,
                    data_files=data_files,
                    available_packages=available_packages
                )
    
                # Step 2: Generate corrected code based on failure analysis
                corrected_code = updater.generate_corrected_code(
                    analysis_name=analysis_name,
                    original_code=code,
                    failure_analysis=failure_analysis,
                    data_files=data_files,
                    available_packages=available_packages,
                    analysis_description=sim_details.get("summary", "")
                )
    
                if corrected_code and corrected_code.strip() != code.strip():
                    # Save refined code
                    refined_path = self.output_dir / f"{analysis_name}_refined_{attempt}.py"
                    with open(refined_path, 'w') as f:
                        f.write(corrected_code)
    
                    # Execute refined code
                    result = self._execute_analysis_code(corrected_code, f"{analysis_name}_refined_{attempt}")
    
                    if result.get("status") == "success":
                        result["refinement_attempts"] = attempt
                        return result
    
                    # Update for next attempt
                    code = corrected_code
                    error_info = result
                else:
                    self.logger.warning(f"Updater returned unchanged code on attempt {attempt}")
                    break
    
            except Exception as e:
                self.logger.error(f"Refinement attempt {attempt} failed: {e}")
    
        error_info["refinement_attempts"] = self.max_refinement_attempts
        return error_info    

    def _detect_container_environment(self) -> bool:
        """Detect if running inside a container (Docker, Apptainer, etc.)."""
        indicators = [
            os.getenv("RUNNING_IN_CONTAINER", "").lower() == "true",
            os.path.exists("/.dockerenv"),
            os.path.exists("/run/.containerenv"),
            os.path.exists("/.singularity.d"),
            os.getenv("APPTAINER_CONTAINER") is not None,
            os.getenv("SINGULARITY_CONTAINER") is not None,
        ]
        return any(indicators)

    def _inventory_files(self):
        """
        Scan the simulation directory and catalog all relevant files.
        
        Populates self.input_files and self.output_files dictionaries.
        """
        self.logger.info(f"Inventorying files in {self.sim_dir}")
        
        self.input_files = {}
        self.output_files = {}
        
        # File type categories
        input_extensions = {
            '.lammps': 'lammps_script',
            '.in': 'lammps_script',
            '.data': 'data_file',
            '.lmp': 'data_file',
            '.pdb': 'pdb_file',
        }
        
        output_extensions = {
            '.lammpstrj': 'trajectory',
            '.dcd': 'trajectory',
            '.xyz': 'trajectory',
            '.csv': 'csv_data',
            '.dat': 'data_output',
            '.png': 'figure',
            '.jpg': 'figure',
            '.pdf': 'figure',
        }
        
        output_prefixes = {
            'log.lammps': 'lammps_log',
            'thermo': 'thermo_data',
        }
        
        for filepath in sorted(self.sim_dir.iterdir()):
            if not filepath.is_file():
                continue
            
            name = filepath.name
            suffix = filepath.suffix.lower()
            size = filepath.stat().st_size
            
            # Skip empty files
            if size == 0:
                continue
            
            file_info = {
                "path": str(filepath),
                "name": name,
                "size": size,
                "suffix": suffix,
            }
            
            # Categorize by extension
            if suffix in input_extensions:
                category = input_extensions[suffix]
                self.input_files[name] = {**file_info, "category": category}
            elif suffix in output_extensions:
                category = output_extensions[suffix]
                self.output_files[name] = {**file_info, "category": category}
            
            # Categorize by name patterns
            if name.startswith("log.lammps"):
                self.output_files[name] = {**file_info, "category": "lammps_log"}
            elif name.startswith("restart."):
                self.output_files[name] = {**file_info, "category": "restart"}
            elif suffix == '.lammps' and name.startswith("ff_"):
                self.input_files[name] = {**file_info, "category": "force_field"}
            
            # Check for fix output files (fix ave/time, fix ave/chunk, etc.)
            # These often have no extension or .dat extension
            if suffix in ['.dat', '.txt', '.out', '.profile', ''] and size > 100:
                # Check if it looks like columnar data
                try:
                    with open(filepath, 'r') as f:
                        first_lines = [f.readline() for _ in range(5)]
                    
                    # If it has comment headers and numeric data, it's output
                    has_comment = any(line.startswith('#') for line in first_lines if line.strip())
                    has_numeric = any(
                        line.strip() and line.strip()[0].isdigit() 
                        for line in first_lines if line.strip()
                    )
                    
                    if has_comment or has_numeric:
                        if name not in self.output_files and name not in self.input_files:
                            self.output_files[name] = {**file_info, "category": "fix_output"}
                except (UnicodeDecodeError, IOError):
                    pass
        
        self.logger.info(f"Found {len(self.input_files)} input files, {len(self.output_files)} output files")
        
        for name, info in self.input_files.items():
            self.logger.debug(f"  Input: {name} ({info['category']})")
        for name, info in self.output_files.items():
            self.logger.debug(f"  Output: {name} ({info['category']})")
    
    
    def _analyze_lammps_input(self) -> Dict[str, Any]:
        """
        Analyze LAMMPS input script(s) to understand simulation setup.
        
        Returns:
            Dictionary with simulation details including parameters,
            ensemble, output commands, etc.
        """
        self.logger.info("Analyzing LAMMPS input script(s)")
        
        # Find LAMMPS input scripts
        scripts = {}
        for name, info in self.input_files.items():
            if info["category"] == "lammps_script":
                scripts[name] = info["path"]
        
        if not scripts:
            # Also check output files for scripts that ran
            for name, info in self.output_files.items():
                if info.get("category") == "lammps_script":
                    scripts[name] = info["path"]
        
        if not scripts:
            self.logger.warning("No LAMMPS input scripts found")
            return {
                "status": "error",
                "message": "No LAMMPS input scripts found",
                "parameters": {},
                "summary": "Unknown simulation"
            }
        
        # Read all scripts and concatenate for analysis
        all_script_content = ""
        for name, path in scripts.items():
            try:
                with open(path, 'r') as f:
                    content = f.read()
                all_script_content += f"\n# === {name} ===\n{content}\n"
            except Exception as e:
                self.logger.warning(f"Could not read {name}: {e}")
        
        if not all_script_content.strip():
            return {
                "status": "error",
                "message": "Could not read any LAMMPS scripts",
                "parameters": {},
                "summary": "Unknown simulation"
            }
        
        # Also read log file for additional context
        log_excerpt = ""
        log_files = [f for f, info in self.output_files.items() if info["category"] == "lammps_log"]
        if log_files:
            log_path = self.output_files[log_files[0]]["path"]
            try:
                with open(log_path, 'r') as f:
                    log_content = f.read()
                # Get header (setup info) and last thermo block
                log_lines = log_content.split('\n')
                # First 100 lines have setup info
                header = '\n'.join(log_lines[:100])
                # Last 30 lines have final thermo
                tail = '\n'.join(log_lines[-30:])
                log_excerpt = f"LOG HEADER:\n{header}\n\nLOG TAIL:\n{tail}"
            except Exception as e:
                self.logger.warning(f"Could not read log file: {e}")
        
        prompt = f"""
    Analyze this LAMMPS simulation setup and extract key parameters.
    
    LAMMPS INPUT SCRIPT(S):
    {all_script_content[:8000]}
    
    {log_excerpt[:3000] if log_excerpt else ""}
    
    Extract the following information and return as JSON:
    {{
        "status": "success",
        "summary": "Brief description of the simulation",
        "parameters": {{
            "units": "real/metal/lj/etc",
            "atom_style": "full/atomic/charge/etc",
            "ensemble": "NVT/NPT/NVE/etc",
            "temperature": 300.0,
            "pressure": 1.0,
            "timestep": 1.0,
            "run_steps": 1000000,
            "simulation_time": "1 ns",
            "thermostat": "nose-hoover/berendsen/etc",
            "barostat": "nose-hoover/berendsen/etc",
            "pair_style": "lj/cut/coul/long etc",
            "kspace_style": "pppm/ewald/etc",
            "special_bonds": "description",
            "thermo_frequency": 1000,
            "dump_frequency": 5000,
            "restart_frequency": 50000
        }},
        "output_commands": [
            {{
                "type": "thermo_style",
                "details": "custom step temp press density pe ke etotal",
                "frequency": 1000
            }},
            {{
                "type": "dump",
                "filename": "trajectory.lammpstrj",
                "frequency": 5000,
                "what": "atom positions"
            }},
            {{
                "type": "fix_ave",
                "filename": "output.dat",
                "frequency": 100,
                "what": "averaged quantity"
            }}
        ],
        "data_file": "name of data file read",
        "force_field_file": "name of included ff file or null",
        "stages": ["minimization", "equilibration", "production"],
        "atom_count": 3940,
        "element_counts": {{"H": 2400, "O": 1200, "etc": 0}}
    }}
    
    Return ONLY JSON.
    """
        
        try:
            result = self._generate_json(prompt)
            
            if "status" not in result:
                result["status"] = "success"
            
            self.logger.info(f"Simulation: {result.get('summary', 'Unknown')}")
            self.logger.info(f"Ensemble: {result.get('parameters', {}).get('ensemble', 'Unknown')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing input: {e}")
            return {
                "status": "error",
                "message": str(e),
                "parameters": {},
                "summary": "Analysis failed"
            }
    
    
    def _identify_output_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Identify and categorize all output data available for analysis.
        
        Maps output data types to their files and metadata.
        
        Returns:
            Dictionary mapping data type name to info dict with:
                - file: path to the data file
                - all_files: list of paths (for series)
                - description: what the data contains
                - format: file format
                - data_type: category
                - physical_quantity: what physical quantity
        """
        self.logger.info("Identifying output data")
        
        output_data = {}
        
        # Get simulation details for context
        sim_details = self._get_sim_details()
        output_commands = sim_details.get("output_commands", [])
        
        # 1. Log file (thermodynamic data)
        log_files = sorted([
            f for f, info in self.output_files.items() 
            if info["category"] == "lammps_log"
        ])
        
        if log_files:
            main_log = log_files[-1]  # Most recent
            log_path = self.output_files[main_log]["path"]
            
            # Parse thermo columns from log
            thermo_columns = self._parse_thermo_columns(log_path)
            
            output_data["thermodynamics"] = {
                "file": log_path,
                "all_files": [self.output_files[f]["path"] for f in log_files],
                "description": f"Thermodynamic output: {', '.join(thermo_columns) if thermo_columns else 'standard thermo'}",
                "format": "lammps_log",
                "data_type": "time_series",
                "physical_quantity": "thermodynamics",
                "columns": thermo_columns,
                "is_series": len(log_files) > 1,
                "num_files": len(log_files)
            }
        
        # 2. Trajectory files
        traj_files = sorted([
            f for f, info in self.output_files.items()
            if info["category"] == "trajectory"
        ])
        
        if traj_files:
            output_data["trajectory"] = {
                "file": self.output_files[traj_files[-1]]["path"],
                "all_files": [self.output_files[f]["path"] for f in traj_files],
                "description": "Atomic trajectory (positions over time)",
                "format": traj_files[-1].split('.')[-1],
                "data_type": "trajectory",
                "physical_quantity": "positions",
                "is_series": len(traj_files) > 1,
                "num_files": len(traj_files)
            }
        
        # 3. Fix output files (averaged data, profiles, etc.)
        fix_files = sorted([
            f for f, info in self.output_files.items()
            if info["category"] == "fix_output"
        ])
        
        for fix_file in fix_files:
            path = self.output_files[fix_file]["path"]
            name = Path(fix_file).stem
            
            # Try to determine what this file contains
            description = self._describe_data_file(path)
            
            output_data[name] = {
                "file": path,
                "description": description,
                "format": "columnar_data",
                "data_type": "time_series",
                "physical_quantity": name,
                "is_series": False,
                "num_files": 1
            }
        
        # 4. CSV files
        csv_files = sorted([
            f for f, info in self.output_files.items()
            if info["category"] == "csv_data"
        ])
        
        for csv_file in csv_files:
            path = self.output_files[csv_file]["path"]
            name = Path(csv_file).stem
            
            output_data[name] = {
                "file": path,
                "description": f"CSV data: {name}",
                "format": "csv",
                "data_type": "tabular",
                "physical_quantity": name,
                "is_series": False,
                "num_files": 1
            }
        
        # 5. Use LLM to interpret output commands if we have them
        if output_commands and output_data:
            # Build command info for LLM interpretation
            output_cmd_info = {}
            for i, cmd in enumerate(output_commands):
                cmd_id = cmd.get("type", f"cmd_{i}")
                output_cmd_info[cmd_id] = {
                    "command_type": cmd.get("type", "unknown"),
                    "filename": cmd.get("filename", ""),
                    "context": json.dumps(cmd)
                }
            
            if output_cmd_info:
                try:
                    # Get script excerpt for context
                    script_excerpt = ""
                    for name, info in self.input_files.items():
                        if info["category"] == "lammps_script":
                            with open(info["path"], 'r') as f:
                                script_excerpt = f.read()[:3000]
                            break
                    
                    interpretations = self._ask_llm_to_interpret_outputs(
                        output_cmd_info, script_excerpt
                    )
                    
                    # Merge interpretations into output_data
                    for cmd_id, interp in interpretations.items():
                        # Try to match to existing output data
                        for data_name, data_info in output_data.items():
                            if data_info.get("physical_quantity", "").lower() in cmd_id.lower():
                                data_info["description"] = interp.get("description", data_info["description"])
                                data_info["physical_quantity"] = interp.get("physical_quantity", data_info["physical_quantity"])
                                data_info["data_type"] = interp.get("data_type", data_info["data_type"])
                                
                except Exception as e:
                    self.logger.warning(f"LLM interpretation of outputs failed: {e}")
        
        self.logger.info(f"Identified {len(output_data)} data sources:")
        for name, info in output_data.items():
            self.logger.info(f"  {name}: {info.get('description', 'Unknown')[:60]}")
        
        return output_data
    
    
    def _parse_thermo_columns(self, log_path: str) -> List[str]:
        """
        Parse thermo column headers from a LAMMPS log file.
        
        Returns list of column names.
        """
        columns = []
        
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            # Look for thermo_style line in the script portion of the log
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Direct thermo_style definition
                if stripped.startswith("thermo_style"):
                    parts = stripped.split()
                    if len(parts) > 2 and parts[1] == "custom":
                        columns = parts[2:]
                        break
                
                # Look for the header line right before numeric data starts
                # This is typically a line with text labels followed by a line with numbers
                if stripped and not stripped.startswith('#'):
                    parts = stripped.split()
                    if len(parts) >= 3 and all(not p[0].isdigit() and not p.startswith('-') for p in parts):
                        # Check if next non-empty line is numeric
                        for j in range(i + 1, min(i + 3, len(lines))):
                            next_line = lines[j].strip()
                            if next_line:
                                next_parts = next_line.split()
                                try:
                                    float(next_parts[0])
                                    # This looks like the header line
                                    columns = parts
                                    break
                                except (ValueError, IndexError):
                                    pass
                        
                        if columns:
                            break
            
        except Exception as e:
            self.logger.warning(f"Could not parse thermo columns from {log_path}: {e}")
        
        if not columns:
            # Default thermo columns
            columns = ["Step", "Temp", "Press", "PotEng", "KinEng", "TotEng"]
        
        return columns
    
    
    def _describe_data_file(self, filepath: str) -> str:
        """
        Read the first few lines of a data file and generate a description.
        """
        try:
            with open(filepath, 'r') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    lines.append(line)
            
            # Look for comment headers
            comments = [l.strip() for l in lines if l.strip().startswith('#')]
            
            # Count data columns
            data_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            n_columns = 0
            if data_lines:
                n_columns = len(data_lines[0].split())
            
            name = Path(filepath).name
            
            if comments:
                # Use comment as description
                header = comments[0].lstrip('#').strip()
                return f"{name}: {header} ({n_columns} columns)"
            else:
                return f"{name}: {n_columns}-column numeric data"
            
        except Exception as e:
            return f"{Path(filepath).name}: data file"
    
    
    def _get_available_packages(self) -> List[str]:
        """
        Get list of Python packages available in the current environment.
        
        In strict mode, only returns packages that are actually importable.
        """
        available = list(self.STANDARD_PACKAGES)  # Always available
        
        # Check optional packages
        for pkg in self.OPTIONAL_PACKAGES:
            try:
                # Normalize package name for import
                import_name = pkg.replace('-', '_')
                spec = importlib.util.find_spec(import_name)
                if spec is not None:
                    available.append(pkg)
            except (ModuleNotFoundError, ValueError):
                pass
        
        # Also check for commonly needed packages
        extra_packages = [
            'scipy.signal', 'scipy.optimize', 'scipy.interpolate',
            'scipy.stats', 'scipy.fft',
        ]
        
        for pkg in extra_packages:
            try:
                top_level = pkg.split('.')[0]
                spec = importlib.util.find_spec(top_level)
                if spec is not None and top_level not in available:
                    available.append(top_level)
            except (ModuleNotFoundError, ValueError):
                pass
        
        self.logger.debug(f"Available packages: {available}")
        return available
    
    
    def _extract_required_packages(self, code: str):
        """
        Extract required packages from generated Python code.
        
        Parses import statements and adds them to self.required_packages.
        
        Args:
            code: Python source code to analyze
        """
        # Match import patterns
        import_patterns = [
            r'^import\s+(\w+)',           # import numpy
            r'^from\s+(\w+)',             # from numpy import ...
            r'^import\s+(\w+)\s+as',     # import numpy as np
        ]
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            for pattern in import_patterns:
                match = re.match(pattern, stripped)
                if match:
                    pkg = match.group(1)
                    
                    # Skip standard library modules
                    stdlib_modules = {
                        'os', 'sys', 're', 'json', 'csv', 'math', 'pathlib',
                        'collections', 'itertools', 'functools', 'warnings',
                        'time', 'datetime', 'copy', 'io', 'tempfile', 'shutil',
                        'glob', 'operator', 'string', 'textwrap', 'struct',
                        'hashlib', 'pickle', 'gzip', 'zipfile', 'abc',
                        'typing', 'enum', 'dataclasses', 'contextlib',
                        'traceback', 'logging', 'argparse', 'unittest',
                        'statistics', 'random', 'fractions', 'decimal',
                        'subprocess', 'multiprocessing', 'threading',
                        'configparser', 'pprint',
                    }
                    
                    if pkg not in stdlib_modules:
                        self.required_packages.add(pkg)
                        self.logger.debug(f"Required package: {pkg}")
                    
                    break  # Only match first pattern per line

    def _clean_generated_code(self, code: str) -> str:
        """
        Clean LLM-generated Python code to ensure it's executable.
        """
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip shebang lines
            if stripped.startswith('#!'):
                continue
            
            # Skip bare 'python' invocations
            if stripped in ('python', 'python3', 'python3.12'):
                continue
            if re.match(r'^python[3]?\s+', stripped):
                continue
            
            # Skip shell commands
            if stripped.startswith('$ ') or stripped.startswith('% '):
                continue
            
            # Skip markdown remnants
            if stripped.startswith('```'):
                continue
            
            cleaned_lines.append(line)
        
        # Remove leading/trailing blank lines
        while cleaned_lines and not cleaned_lines[0].strip():
            cleaned_lines.pop(0)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        result = '\n'.join(cleaned_lines)
        
        # Fix JavaScript-style booleans/nulls in Python code
        # Only fix outside of strings (simple heuristic: not inside quotes)
        result = self._fix_json_literals_in_python(result)
        
        return result
    
    
    def _fix_json_literals_in_python(self, code: str) -> str:
        """
        Fix JavaScript/JSON-style literals in Python code.
        
        Converts:
            true  -> True
            false -> False
            null  -> None
            
        Only fixes bare keywords, not those inside strings.
        """
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip comment lines and string-only lines
            stripped = line.strip()
            if stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            
            # Skip lines that are inside triple-quoted strings
            # (Simple heuristic - not perfect but catches most cases)
            if stripped.startswith('"""') or stripped.startswith("'''"):
                fixed_lines.append(line)
                continue
            
            # Split line into code and inline comment
            # Handle # inside strings roughly
            code_part = line
            comment_part = ""
            
            in_single = False
            in_double = False
            for i, char in enumerate(line):
                if char == "'" and not in_double:
                    in_single = not in_single
                elif char == '"' and not in_single:
                    in_double = not in_double
                elif char == '#' and not in_single and not in_double:
                    code_part = line[:i]
                    comment_part = line[i:]
                    break
            
            # Fix JSON literals in code part only (not in strings)
            # Use word boundary regex to avoid replacing inside variable names
            code_part = re.sub(r'\btrue\b', 'True', code_part)
            code_part = re.sub(r'\bfalse\b', 'False', code_part)
            code_part = re.sub(r'\bnull\b', 'None', code_part)
            
            fixed_lines.append(code_part + comment_part)
        
        return '\n'.join(fixed_lines)
