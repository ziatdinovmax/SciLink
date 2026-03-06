# scilink/agents/sim_agents/lammps_updater.py

import os
import re
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ._deprecation import normalize_params


class LAMMPSUpdater:
    """
    Self-evolving updater that analyzes LAMMPS errors and generates solutions.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: str = "gemini-3-pro-preview",
                 base_url: Optional[str] = None,
                 # Legacy parameters
                 local_model: Optional[str] = None,
                 google_api_key: Optional[str] = None):
        """
        Initialize the LAMMPS updater.
        
        Args:
            api_key: API key for the LLM provider
            model_name: Model name to use
            base_url: Optional base URL for internal proxy
            local_model: Deprecated, use base_url instead
            google_api_key: Deprecated, use api_key instead
        """
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Normalize deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="LAMMPSUpdater"
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
    
    # ============================================================================
    # HELPER METHODS FOR LLM CALLS
    # ============================================================================
    
    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON from LLM, handling providers that don't support response_mime_type."""
        try:
            # Try with response_mime_type first
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            if "type conflicts" in str(e) or "BadRequestError" in str(e):
                # Bedrock doesn't support response_mime_type - fall back to text
                self.logger.warning("Provider doesn't support response_mime_type, using text fallback")
                response = self.model.generate_content(prompt)
                text = response.text.strip()
                # Extract JSON from response
                if text.startswith("```"):
                    text = re.sub(r'```(?:json)?', '', text).strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                raise ValueError(f"Could not extract JSON from response: {text[:200]}")
            raise

    def _generate_text(self, prompt: str) -> str:
        """Generate text response from LLM."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise
    
   
    def _analyze_issues(self, errors: List[str], input_script: str, data_content: str = "", 
                       ff_content: str = "", log_content: str = "") -> Dict[str, Any]:
        """Have the LLM analyze the issues and suggest a correction strategy."""
        self.logger.info("Analyzing LAMMPS errors and suggesting correction strategy")
        
        analysis_prompt = f"""
You are a LAMMPS simulation expert. Analyze these LAMMPS errors/warnings, the input script, and log file to:
1. Identify the root cause of each issue
2. Suggest specific corrections
3. Determine what parts of the script need modification
4. Analyze how far the simulation progressed before the error

ERRORS/WARNINGS:
{chr(10).join(errors)}

INPUT SCRIPT:
{input_script}

{'DATA FILE (excerpt):' + data_content if data_content else ''}
{'FORCE FIELD FILE:' + ff_content if ff_content else ''}
{'LOG FILE EXCERPT:' + log_content[:2000] if log_content else ''}

Respond with a JSON object:
{{
    "issues": [
        {{
            "error_text": "The exact error message",
            "root_cause": "Detailed explanation",
            "fix_strategy": "Specific strategy to fix",
            "script_sections_to_modify": ["section1", "section2"]
        }}
    ],
    "overall_assessment": "Overall assessment of the problems",
    "is_data_file_problem": true/false,
    "is_force_field_problem": true/false,
    "simulation_progress": {{
        "stage": "minimization/equilibration/production",
        "completed_steps": 0,
        "total_steps": 0,
        "percent_complete": 0
    }},
    "correction_approach": "Detailed approach to correct the script",
    "critical_commands_to_add": ["command1", "command2"],
    "critical_commands_to_remove": ["command1", "command2"]
}}
"""
        
        try:
            analysis = self._generate_json(analysis_prompt)  # ✅ Use helper
            self.logger.info(f"Analysis completed: {len(analysis.get('issues', []))} issues identified")
            return analysis
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return {
                "issues": [{"error_text": err, "root_cause": "Unknown", 
                           "fix_strategy": "Review LAMMPS documentation", 
                           "script_sections_to_modify": ["Unknown"]} for err in errors],
                "overall_assessment": "Failed to analyze errors properly",
                "is_data_file_problem": False,
                "is_force_field_problem": False,
                "simulation_progress": {
                    "stage": "unknown",
                    "completed_steps": 0,
                    "total_steps": 0,
                    "percent_complete": 0
                },
                "correction_approach": "Manual review needed",
                "critical_commands_to_add": [],
                "critical_commands_to_remove": []
            }
    
    def refine_inputs(self,
                    input_path: str,
                    research_goal: str,
                    ff_path: Optional[str] = None,
                    data_path: Optional[str] = None,
                    lammps_log: str = 'log.lammps') -> Tuple[str, Dict[str, Any]]:
        """Refine LAMMPS input files based on error output."""
        self.logger.info(f"Refining LAMMPS input: {input_path}")
                
        input_txt = Path(input_path).read_text()
        
        ff_txt = ""
        if ff_path and os.path.exists(ff_path):
            ff_txt = Path(ff_path).read_text()
        
        data_txt = ""
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data_txt = "".join(f.readlines()[:100])
        
        log_txt = ""
        if os.path.exists(lammps_log):
            with open(lammps_log, 'r') as f:
                log_txt = f.read()
        
        error_list = self._extract_errors(log_txt)
        if not error_list:
            self.logger.warning("No errors found in log file")
            return input_txt, {"issues": [], "overall_assessment": "No errors found"}
        
        # Check for restart files
        working_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else "."
        restart_files = list(Path(working_dir).glob('restart.*'))
        
        has_restart = False
        latest_restart = None
        if restart_files:
            restart_files.sort(key=os.path.getmtime)
            latest_restart = restart_files[-1]
            has_restart = True
            self.logger.info(f"Found latest restart file: {latest_restart}")
        
        # Analyze issues
        analysis = self._analyze_issues(error_list, input_txt, data_txt, ff_txt, log_txt)
        
        if has_restart:
            analysis["restart_file"] = str(latest_restart)
            analysis["should_restart"] = True
        else:
            analysis["should_restart"] = False
        
        # Generate correction prompt
        correction_prompt = self._generate_correction_prompt(
            analysis, input_txt, research_goal, data_txt, ff_txt
        )
        
        # Add restart instructions if needed
        if has_restart:
            restart_basename = os.path.basename(str(latest_restart))
            correction_prompt += f"""

IMPORTANT: A restart file was found at {latest_restart}. 
Please modify the script to:
1. Use this restart file instead of the data file
2. Comment out 'read_data' and use 'read_restart {restart_basename}'
3. Skip minimization if already completed
4. Resume from where it left off

CRITICAL RESTART RULES:
1. Command order: units → atom_style → read_restart → groups → set → force_field → fixes
2. NEVER place 'set' commands before 'read_restart'
3. NEVER place 'fix shake' after box-changing fixes like 'fix npt'
4. Always redefine force field parameters even with restart
"""
        
        # Generate corrected script
        self.logger.info("Generating corrected LAMMPS script")
        corrected_script = self._generate_text(correction_prompt)  # ✅ Use helper
        corrected_script = self._clean_script(corrected_script)
        
        # Ensure restart handling
        if has_restart:
            corrected_script = self._ensure_restart_handling(
                corrected_script, 
                os.path.basename(str(latest_restart))
            )
        
        self.logger.info(f"Script correction completed - {len(corrected_script)} characters")
        return corrected_script, analysis
    
    def refine_for_quality_issues(self,
                                  input_path: str,
                                  research_goal: str,
                                  quality_assessment: Dict[str, Any],
                                  system_info: Dict[str, Any],
                                  stage: str,
                                  ff_was_modified: bool = False,
                                  charges_were_modified: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Refine LAMMPS input based on quality assessment."""
        self.logger.info(f"Refining for quality issues at stage: {stage}")
        
        input_txt = Path(input_path).read_text()
        
        issues = quality_assessment.get("issues", [])
        recommendations = quality_assessment.get("recommendations", [])
        metrics = quality_assessment.get("quality_metrics", {})
        
        if not issues:
            self.logger.warning("No quality issues provided")
            return input_txt, {"status": "no_issues", "message": "No quality issues"}
        
        # Build data strings OUTSIDE the f-string to avoid {{}} issues
        issues_list = [
            {"severity": i.get("severity"), "description": i.get("description"),
             "metric": i.get("metric"), "value": i.get("value")}
            for i in issues
        ]
        issues_str = json.dumps(issues_list, indent=2, default=str)
        
        recs_str = "\n".join(f"- {rec}" for rec in recommendations)
        metrics_str = json.dumps(metrics, indent=2, default=str)
        
        prior_fixes = []
        if ff_was_modified:
            prior_fixes.append("Force field parameters (ff_params.lammps) were updated")
        if charges_were_modified:
            prior_fixes.append("Partial charges in the data file were updated")
        prior_text = "\n".join(prior_fixes) if prior_fixes else "None"
        
        atom_count = system_info.get('atom_count', 'Unknown')
        element_counts = system_info.get('element_counts', {})
        element_str = json.dumps(element_counts, default=str)
        
        prompt = f"""
    You are a LAMMPS expert. Adjust this simulation script to address quality issues.
    
    RESEARCH GOAL: {research_goal}
    STAGE: {stage}
    
    QUALITY ISSUES DETECTED:
    {issues_str}
    
    RECOMMENDATIONS:
    {recs_str}
    
    QUALITY METRICS:
    {metrics_str}
    
    PRIOR FIXES ALREADY APPLIED:
    {prior_text}
    
    SYSTEM INFO:
    - Atoms: {atom_count}
    - Composition: {element_str}
    
    CURRENT SCRIPT:
    {input_txt}
    
    IMPORTANT:
    - If FF or charges were modified, use read_data instead of read_restart
    - If FF or charges were modified, add minimization before dynamics
    - For density problems, check pair_modify mix rule
    - For temperature issues, adjust Tdamp (~100*timestep)
    - For pressure issues, adjust Pdamp (~100*timestep)
    - Keep restart file functionality
    - Only modify parameters relevant to the issues
    
    Return ONLY the corrected LAMMPS script without markdown.
    """
        
        try:
            corrected_script = self._generate_text(prompt)
            corrected_script = self._clean_script(corrected_script)
            
            correction_analysis = {
                "status": "corrected",
                "issues_addressed": [i.get("description") for i in issues],
                "corrections_made": "Quality-based parameter adjustments",
                "stage": stage,
                "ff_was_modified": ff_was_modified,
                "charges_was_modified": charges_were_modified
            }
            
            self.logger.info("Quality-based correction completed")
            return corrected_script, correction_analysis
            
        except Exception as e:
            self.logger.error(f"Error during quality correction: {e}")
            return input_txt, {
                "status": "error",
                "message": f"Failed to generate quality corrections: {e}"
            }

    def _extract_errors(self, log_content: str) -> List[str]:
        """Extract all errors from LAMMPS output."""
        lines = log_content.splitlines()
        issues = []
        
        for i, line in enumerate(lines):
            if re.search(r"ERROR", line, flags=re.IGNORECASE):
                error_msg = line.strip()
                if i + 1 < len(lines):
                    error_msg += "\nLast command: " + lines[i + 1].strip()
                issues.append(error_msg)
        
        return issues
    
    def _generate_correction_prompt(self, analysis: Dict[str, Any], 
                                  input_script: str, 
                                  research_goal: str,
                                  data_content: str = "", 
                                  ff_content: str = "") -> str:
        """Generate targeted correction prompt."""
        self.logger.info("Generating correction prompt based on analysis")       

        issues_summary = "\n".join([f"- {issue['error_text']}: {issue['root_cause']}" 
                                  for issue in analysis.get("issues", [])])
        fix_strategies = "\n".join([f"- {issue['fix_strategy']}" 
                                  for issue in analysis.get("issues", [])])
        
        sim_progress = analysis.get("simulation_progress", {})
        progress_info = ""
        if sim_progress:
            stage = sim_progress.get("stage", "unknown")
            completed = sim_progress.get("completed_steps", 0)
            total = sim_progress.get("total_steps", 0)
            percent = sim_progress.get("percent_complete", 0)
            progress_info = f"\nSimulation progress: {stage} phase, {completed}/{total} steps ({percent}% complete)"
        
        correction_prompt = f"""
As a LAMMPS expert, correct this input script for: "{research_goal}"

Issues:
{issues_summary}

Fix strategies:
{fix_strategies}

Overall: {analysis.get("overall_assessment", "Unknown")}{progress_info}

Critical commands to add: {', '.join(analysis.get("critical_commands_to_add", []))}
Critical commands to remove: {', '.join(analysis.get("critical_commands_to_remove", []))}

ORIGINAL SCRIPT:
{input_script}

{'DATA FILE:' + data_content if data_content else ''}
{'FORCE FIELD:' + ff_content if ff_content else ''}

Provide complete, corrected LAMMPS script ready to run.

IMPORTANT: 
- Include restart file writing (every 10k-50k steps)
- Use naming like "restart.*.equil" and "restart.*.prod"
- Only fix issues listed above for systematic refinement

Return ONLY raw LAMMPS script without markdown.
"""
        
        return correction_prompt
    
    def _clean_script(self, script_text: str) -> str:
        """Remove markdown formatting."""
        script_text = re.sub(r'```(?:lammps|bash)?', '', script_text)
        script_text = script_text.replace('```', '')
        script_text = script_text.strip()
        return script_text
    
    def _ensure_restart_handling(self, script_text: str, restart_file: str) -> str:
        """Ensure script properly uses restart file."""
        lines = script_text.split('\n')
        has_read_restart = False
        
        for i, line in enumerate(lines):
            if "read_data" in line and not line.strip().startswith('#'):
                lines[i] = f"# {line}  # Commented out for restart"
                
            if "read_restart" in line and not line.strip().startswith('#'):
                has_read_restart = True
                if restart_file not in line:
                    lines[i] = f"read_restart {restart_file}  # Updated restart file"
        
        if not has_read_restart:
            insert_pos = 0
            for i, line in enumerate(lines):
                if any(cmd in line.lower() for cmd in ["units", "atom_style"]):
                    insert_pos = max(insert_pos, i + 1)
                elif "read_data" in line and line.strip().startswith('#'):
                    insert_pos = i
                    break
            
            lines.insert(insert_pos, f"read_restart {restart_file}  # Added for restart")
        
        return '\n'.join(lines)
