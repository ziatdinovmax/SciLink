# scilink/agents/sim_agents/simulation_orchestrator.py

import os
import re
import subprocess
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from .lammps_agent import LAMMPSSimulationAgent
from .lammps_updater import LAMMPSUpdater
from .lammps_analysis import LAMMPSAnalysisAgent
from .lammps_utils import VMDLAMMPSConverter

class LAMMPSOrchestrator:
    """
    Orchestrates LAMMPS simulations with adaptive quality monitoring.
    
    Does NOT directly use LLMs - delegates to sub-agents.
    """
    
    def __init__(self,
                 working_dir: str,
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-3-pro-preview",
                 base_url: Optional[str] = None,
                 lammps_command: str = "lmp",
                 max_stage_attempts: int = 3,
                 stage_timeout: int = 3600):
        """
        Initialize the simulation orchestrator.
        
        Args:
            working_dir: Working directory for simulation
            api_key: API key for LLM provider
            model_name: Model name to use
            base_url: Optional base URL for internal proxy
            lammps_command: Command to run LAMMPS
            max_stage_attempts: Max correction attempts per stage
            stage_timeout: Timeout per stage in seconds
        """
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(exist_ok=True, parents=True)
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key required for orchestrator")
        
        self.model_name = model_name
        self.base_url = base_url
        self.lammps_command = lammps_command
        self.max_stage_attempts = max_stage_attempts
        self.stage_timeout = stage_timeout
        
        # Initialize sub-agents (lazy loading) - pass through API config
        self._sim_agent = None
        self._analysis_agent = None
        self._updater = None
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Tracking
        self.quality_history = []
        self.correction_history = []
        self.stage_results = {}
    
    @property
    def sim_agent(self):
        """Lazy-load simulation agent."""
        if self._sim_agent is None:
            self._sim_agent = LAMMPSSimulationAgent(
                working_dir=str(self.working_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url
            )
        return self._sim_agent
    
    @property
    def analysis_agent(self):
        """Lazy-load analysis agent."""
        if self._analysis_agent is None:
            self._analysis_agent = LAMMPSAnalysisAgent(
                sim_dir=str(self.working_dir),
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                package_mode='strict',
                executor_timeout=120
            )
        return self._analysis_agent
    
    @property
    def updater(self):
        """Lazy-load updater agent."""
        if self._updater is None:
            self._updater = LAMMPSUpdater(
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url
            )
        return self._updater


    def run_supervised_simulation(self,
                                  data_file: str,
                                  research_goal: str,
                                  system_description: Optional[str] = None,
                                  force_field_files: Optional[Dict[str, str]] = None,
                                  run_final_analysis: bool = True,
                                  **kwargs) -> Dict[str, Any]:
        """
        Run a fully supervised simulation with quality checks and adaptive corrections.
        
        Args:
            data_file: Path to LAMMPS data file
            research_goal: Research objective
            system_description: System description (optional)
            force_field_files: Force field parameter files (optional)
            run_final_analysis: Whether to run comprehensive analysis at end
            **kwargs: Additional parameters for LAMMPSSimulationAgent
            
        Returns:
            Dictionary with complete results:
                - status: "success", "failed", or "partial"
                - stage_results: Results for each stage
                - quality_history: Quality checks performed
                - correction_history: Corrections made
                - final_analysis: Comprehensive analysis (if run_final_analysis=True)
        """
        print(f"\n{'='*80}")
        print(f"🎯 SUPERVISED LAMMPS SIMULATION")
        print(f"{'='*80}")
        print(f"Research Goal: {research_goal}")
        print(f"Working Directory: {self.working_dir}")
        print(f"LAMMPS Command: {self.lammps_command}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # ========================================================================
        # STAGE 1: Generate Staged Simulation
        # ========================================================================
        print(f"📝 STAGE 1: Generating staged simulation")
        print(f"{'─'*80}")
        
        try:
            sim_info = self.sim_agent.generate_staged_simulation(
                data_file=data_file,
                research_goal=research_goal,
                system_description=system_description,
                force_field_files=force_field_files,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Failed to generate simulation: {e}")
            return self._failed_result(f"Simulation generation failed: {e}")
        
        stages = sim_info.get("stages", [])
        stage_scripts = sim_info.get("staged_scripts", {})
        
        if not stages or not stage_scripts:
            # Fallback to single script
            self.logger.warning("No staged scripts generated, using single script")
            stages = ["full_simulation"]
            stage_scripts = {"full_simulation": sim_info["script_path"]}
        
        print(f"✓ Generated {len(stages)} stages: {', '.join(stages)}")
        print(f"{'─'*80}\n")
        
        # ========================================================================
        # STAGE 2: Execute Each Stage with Quality Monitoring
        # ========================================================================
        print(f"🔬 STAGE 2: Executing simulation stages with quality monitoring")
        print(f"{'='*80}\n")
        
        completed_stages = []
        
        for stage_idx, stage_name in enumerate(stages):
            print(f"\n{'─'*80}")
            print(f"🔬 STAGE: {stage_name.upper()} ({stage_idx + 1}/{len(stages)})")
            print(f"{'─'*80}")
            
            stage_script_path = stage_scripts.get(stage_name)
            if not stage_script_path or not os.path.exists(stage_script_path):
                self.logger.error(f"Script not found for stage: {stage_name}")
                return self._partial_result(
                    completed_stages, 
                    f"Script not found for {stage_name}"
                )
            
            # Execute stage with correction loop
            stage_success = False
            
            for attempt in range(1, self.max_stage_attempts + 1):
                print(f"\n  🔄 Attempt {attempt}/{self.max_stage_attempts}")
                
                # ============================================================
                # Step A: Execute LAMMPS
                # ============================================================
                print(f"  ▶  Running LAMMPS for {stage_name}...")
                
                exec_result = self._execute_lammps(stage_script_path)
                
                if exec_result["status"] == "lammps_error":
                    error_msg = exec_result.get('error', 'Unknown')
                    print(f"  ❌ LAMMPS error detected")
                    print(f"     Return code: {exec_result.get('returncode', '?')}")
                    print(f"     Error details:")
                    for line in error_msg.split('\n'):
                        if line.strip():
                            print(f"       {line.strip()}")
                    
                    # Save error details for this attempt
                    error_log_path = self.working_dir / f"error_details_{stage_name}_attempt{attempt}.txt"
                    with open(error_log_path, 'w') as f:
                        f.write(f"Stage: {stage_name}\n")
                        f.write(f"Attempt: {attempt}\n")
                        f.write(f"Script: {stage_script_path}\n")
                        f.write(f"Return code: {exec_result.get('returncode', '?')}\n")
                        f.write(f"\n{'='*60}\nERROR MESSAGE:\n{'='*60}\n")
                        f.write(error_msg)
                        f.write(f"\n\n{'='*60}\nSTDERR:\n{'='*60}\n")
                        f.write(getattr(self, '_last_stderr', 'N/A'))
                        f.write(f"\n\n{'='*60}\nLOG CONTENT:\n{'='*60}\n")
                        f.write(getattr(self, '_last_log_content', 'N/A'))
                    
                    # Try to fix LAMMPS error
                    print(f"\n  🔧 Attempting LAMMPS error correction...")
                    corrected, new_script_path, correction_info = self._fix_lammps_error(
                        stage_script_path,
                        research_goal,
                        sim_info
                    )
                    
                    if corrected:
                        print(f"  ✓  LAMMPS error corrected → {Path(new_script_path).name}")
                        stage_script_path = new_script_path
                        self.correction_history.append({
                            "stage": stage_name,
                            "attempt": attempt,
                            "type": "lammps_error",
                            "correction": correction_info
                        })
                        continue  # Retry
                    else:
                        print(f"  ✗  Could not fix LAMMPS error")
                        print(f"     Reason: {correction_info.get('error', 'Unknown')}")
                        return self._partial_result(
                            completed_stages,
                            f"Unrecoverable LAMMPS error in {stage_name}: {error_msg[:200]}"
                        )                
                # LAMMPS completed successfully
                print(f"  ✓  LAMMPS completed")
                
                # ============================================================
                # Step B: Quality Check
                # ============================================================
                print(f"  🔍 Running quality check...")
                
                quality_result = self.analysis_agent.run_quality_check(
                    research_goal=research_goal,
                    stage=stage_name
                )
                
                self.quality_history.append({
                    "stage": stage_name,
                    "attempt": attempt,
                    "result": quality_result
                })
                
                status = quality_result.get("status", "unknown")
                can_continue = quality_result.get("can_continue", True)
                
                # Print quality summary
                self._print_quality_summary(quality_result)
                
                # ============================================================
                # Step C: Decide Action Based on Quality
                # ============================================================
                
                if status == "healthy":
                    print(f"  ✅ Stage {stage_name} passed - quality is healthy")
                    stage_success = True
                    completed_stages.append(stage_name)
                    self.stage_results[stage_name] = {
                        "status": "success",
                        "attempts": attempt,
                        "quality": quality_result
                    }
                    break  # Move to next stage
                
                elif status == "warning" and can_continue:
                    print(f"  ⚠️  Warnings detected but continuing")
                    print(f"     Issues: {len(quality_result.get('issues', []))}")
                    stage_success = True
                    completed_stages.append(stage_name)
                    self.stage_results[stage_name] = {
                        "status": "warning",
                        "attempts": attempt,
                        "quality": quality_result
                    }
                    break  # Move to next stage
                
                elif status == "critical" or not can_continue:
                    print(f"  ❌ Critical quality issues detected")
                    
                    # Try to fix quality issues
                    if attempt < self.max_stage_attempts:
                        print(f"  🔧 Attempting quality-based correction...")
                        corrected, new_script_path, correction_info = self._fix_quality_issues(
                            stage_script_path,
                            quality_result,
                            research_goal,
                            sim_info,
                            stage_name
                        )
                        
                        if corrected:
                            print(f"  ✓  Script adjusted for quality issues")
                            stage_script_path = new_script_path
                            self.correction_history.append({
                                "stage": stage_name,
                                "attempt": attempt,
                                "type": "quality_issue",
                                "correction": correction_info
                            })
                            continue  # Retry with corrected script
                        else:
                            print(f"  ✗  Could not correct quality issues")
                    
                    # If we've exhausted attempts
                    print(f"  ✗  Failed {stage_name} after {attempt} attempts")
                    return self._partial_result(
                        completed_stages,
                        f"Critical quality issues in {stage_name}"
                    )
                
                else:  # Unknown status
                    print(f"  ❓ Unknown quality status: {status}")
                    if attempt < self.max_stage_attempts:
                        print(f"     Retrying...")
                        continue
                    else:
                        stage_success = True  # Continue with warnings
                        completed_stages.append(stage_name)
                        break
            
            # Check if stage succeeded
            if not stage_success:
                return self._partial_result(
                    completed_stages,
                    f"Failed {stage_name} after {self.max_stage_attempts} attempts"
                )
        
        # ========================================================================
        # STAGE 3: All simulation stages completed successfully
        # ========================================================================
        print(f"\n{'='*80}")
        print(f"✅ ALL SIMULATION STAGES COMPLETED")
        print(f"{'='*80}")
        print(f"Completed stages: {', '.join(completed_stages)}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Quality checks: {len(self.quality_history)}")
        print(f"Corrections made: {len(self.correction_history)}")
        print(f"{'='*80}\n")
        
        # ========================================================================
        # STAGE 4: Final Comprehensive Analysis (optional)
        # ========================================================================
        final_analysis = None
        
        if run_final_analysis:
            print(f"📊 STAGE 3: Running final comprehensive analysis")
            print(f"{'─'*80}")
            
            try:
                final_analysis = self.analysis_agent.run_analysis(research_goal)
                print(f"✓ Final analysis complete")
            except Exception as e:
                self.logger.error(f"Final analysis failed: {e}")
                final_analysis = {
                    "status": "error",
                    "message": f"Final analysis failed: {e}"
                }
        
        # ========================================================================
        # Return Complete Results
        # ========================================================================
        return {
            "status": "success",
            "working_directory": str(self.working_dir),
            "simulation_info": sim_info,
            "completed_stages": completed_stages,
            "stage_results": self.stage_results,
            "quality_history": self.quality_history,
            "correction_history": self.correction_history,
            "total_quality_checks": len(self.quality_history),
            "total_corrections": len(self.correction_history),
            "final_analysis": final_analysis,
            "execution_time": time.time() - start_time
        }
    
    # ============================================================================
    # LAMMPS EXECUTION
    # ============================================================================
    def _execute_lammps(self, script_path: str) -> Dict[str, Any]:
        """Execute a LAMMPS script."""
        log_file = self.working_dir / "log.lammps"
        
        # Backup previous log if it exists
        if log_file.exists():
            backup_log = self.working_dir / f"log.lammps.bak{int(time.time())}"
            log_file.rename(backup_log)
        
        self._last_stderr = ""
        self._last_stdout = ""
        self._last_log_content = ""  # Store log content before it gets overwritten
        
        try:
            self.logger.info(f"Executing: {self.lammps_command} -in {script_path}")
            result = subprocess.run(
                f"{self.lammps_command} -in {script_path}",
                shell=True,
                check=False,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.stage_timeout
            )
            
            self._last_stderr = result.stderr or ""
            self._last_stdout = result.stdout or ""
            
            # Read log file immediately and store it
            if log_file.exists():
                with open(log_file, 'r') as f:
                    self._last_log_content = f.read()
            
            if result.returncode != 0:
                self.logger.warning(f"LAMMPS exited with code {result.returncode}")
                
                error_parts = []
                
                # Check log file for LAMMPS ERROR lines
                if self._last_log_content:
                    error_lines = [l for l in self._last_log_content.split('\n') if 'ERROR' in l]
                    if error_lines:
                        error_parts.append("LAMMPS ERROR: " + '\n'.join(error_lines[:5]))
                        
                    # Also get the "Last command" line
                    last_cmd_lines = [l for l in self._last_log_content.split('\n') if 'Last command' in l]
                    if last_cmd_lines:
                        error_parts.append("Last command: " + last_cmd_lines[-1].strip())
                
                # Check stderr
                if result.stderr:
                    error_parts.append("STDERR: " + result.stderr[:500])
                
                # Check stdout for errors too
                if result.stdout:
                    stdout_errors = [l for l in result.stdout.split('\n') if 'ERROR' in l]
                    if stdout_errors:
                        error_parts.append("STDOUT ERRORS: " + '\n'.join(stdout_errors[:3]))
                
                error_msg = '\n'.join(error_parts) if error_parts else f"Unknown error (exit code {result.returncode})"
                
                return {
                    "status": "lammps_error",
                    "error": error_msg,
                    "returncode": result.returncode
                }
            
            # Check log for ERROR even with return code 0
            if self._last_log_content and "ERROR" in self._last_log_content:
                error_lines = [l for l in self._last_log_content.split('\n') if 'ERROR' in l]
                return {
                    "status": "lammps_error",
                    "error": '\n'.join(error_lines[:5]),
                    "returncode": result.returncode
                }
            
            self.logger.info("LAMMPS execution completed successfully")
            return {
                "status": "success",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            self._last_stderr = f"Timed out after {self.stage_timeout}s"
            return {"status": "lammps_error", "error": self._last_stderr}
        except FileNotFoundError:
            self._last_stderr = f"LAMMPS executable not found: {self.lammps_command}"
            return {"status": "lammps_error", "error": self._last_stderr}
        except Exception as e:
            self._last_stderr = str(e)
            return {"status": "lammps_error", "error": str(e)}

    # ============================================================================
    # ERROR CORRECTION
    # ============================================================================
    
    def _fix_lammps_error(self,
                         script_path: str,
                         research_goal: str,
                         sim_info: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Fix LAMMPS errors using LAMMPSUpdater.
        Shows the updater's LLM reasoning for debugging.
        """
        try:
            log_file = self.working_dir / "log.lammps"
            
            # Use stored log content from the execution that just failed
            stored_log = getattr(self, '_last_log_content', '')
            stderr_content = getattr(self, '_last_stderr', '')
            stdout_content = getattr(self, '_last_stdout', '')
            
            # Determine the best source of error information
            if stored_log and "ERROR" in stored_log:
                log_content = stored_log
                self.logger.info("Using stored log content (has ERROR)")
            elif log_file.exists():
                with open(log_file, 'r') as f:
                    log_content = f.read()
                self.logger.info("Using log.lammps file")
            else:
                log_content = ""
            
            # If current log has no error, search backup logs
            if "ERROR" not in log_content:
                backup_logs = sorted(
                    self.working_dir.glob("log.lammps.bak*"),
                    key=lambda f: f.stat().st_mtime,
                    reverse=True
                )
                for backup in backup_logs[:3]:
                    with open(backup, 'r') as f:
                        backup_content = f.read()
                    if "ERROR" in backup_content:
                        self.logger.info(f"Found error in backup log: {backup.name}")
                        log_content = backup_content
                        break
            
            # Create comprehensive error log for the updater
            combined_log = self.working_dir / "log_for_correction.lammps"
            with open(combined_log, 'w') as f:
                f.write(log_content)
                if stderr_content:
                    f.write("\n\n# === STDERR FROM LAMMPS EXECUTION ===\n")
                    f.write(stderr_content)
                if stdout_content and "ERROR" in stdout_content:
                    f.write("\n\n# === STDOUT ERRORS ===\n")
                    f.write(stdout_content)
            
            # Check if we have error information
            has_error_info = (
                "ERROR" in log_content or 
                "ERROR" in stderr_content or 
                "MPI_ABORT" in stderr_content
            )
            
            if not has_error_info:
                self.logger.error("No error information found in any source")
                print(f"     ⚠️  No error information found in:")
                print(f"        - log.lammps ({len(log_content)} chars)")
                print(f"        - stderr ({len(stderr_content)} chars)")
                print(f"        - {len(list(self.working_dir.glob('log.lammps.bak*')))} backup logs")
                return False, script_path, {"error": "No error information available"}
            
            # Print what we're sending to the updater
            print(f"     📋 Error log: {len(log_content)} chars, stderr: {len(stderr_content)} chars")
            
            # Parse data file for type information
            data_path = sim_info.get("data_path")
            enhanced_goal = research_goal
            if data_path and os.path.exists(data_path):
                data_file_context = self._get_data_file_context(data_path)
                if data_file_context:
                    enhanced_goal = (
                        f"{research_goal}\n\n"
                        f"CRITICAL - DATA FILE TYPE INFORMATION:\n"
                        f"{data_file_context}\n"
                        f"You MUST use these exact type numbers."
                    )
            
            # Call the updater
            print(f"     🤖 Calling LLM updater for analysis...")
            corrected_script, analysis = self.updater.refine_inputs(
                input_path=script_path,
                research_goal=enhanced_goal,
                data_path=data_path,
                lammps_log=str(combined_log)
            )
            
            # ============================================================
            # PRINT THE UPDATER'S REASONING
            # ============================================================
            print(f"\n     {'─'*60}")
            print(f"     📊 UPDATER ANALYSIS:")
            print(f"     {'─'*60}")
            
            # Print issues found
            issues = analysis.get("issues", [])
            if issues:
                print(f"     Issues identified: {len(issues)}")
                for i, issue in enumerate(issues, 1):
                    error_text = issue.get("error_text", "No error text")
                    root_cause = issue.get("root_cause", "Unknown")
                    fix_strategy = issue.get("fix_strategy", "Unknown")
                    print(f"\n     Issue {i}:")
                    print(f"       Error: {error_text[:120]}")
                    print(f"       Cause: {root_cause[:120]}")
                    print(f"       Fix:   {fix_strategy[:120]}")
            else:
                print(f"     ⚠️  No issues identified by updater!")
            
            # Print overall assessment
            overall = analysis.get("overall_assessment", "")
            if overall:
                print(f"\n     Overall: {overall[:200]}")
            
            # Print correction approach
            approach = analysis.get("correction_approach", "")
            if approach:
                print(f"     Approach: {approach[:200]}")
            
            # Print if data file or force field problem
            if analysis.get("is_data_file_problem"):
                print(f"     ⚠️  Data file problem detected")
            if analysis.get("is_force_field_problem"):
                print(f"     ⚠️  Force field problem detected")
            
            # Print simulation progress
            progress = analysis.get("simulation_progress", {})
            if progress:
                pct = progress.get("percent_complete", 0)
                stage = progress.get("stage", "unknown")
                print(f"     Progress: {stage} - {pct}% complete")
            
            # Print should_restart
            if analysis.get("should_restart"):
                restart_file = analysis.get("restart_file", "unknown")
                print(f"     🔄 Recommends restart from: {restart_file}")
            
            # Print modified commands
            modified = analysis.get("modified_commands", [])
            if modified:
                print(f"\n     Modified commands ({len(modified)}):")
                for mod in modified[:5]:
                    orig = mod.get("original", "?")
                    corr = mod.get("corrected", "?")
                    print(f"       - {orig[:60]}")
                    print(f"       + {corr[:60]}")
                if len(modified) > 5:
                    print(f"       ... and {len(modified) - 5} more")
            
            # Print commands to add/remove
            to_add = analysis.get("critical_commands_to_add", [])
            if to_add:
                print(f"\n     Commands to add:")
                for cmd in to_add[:5]:
                    print(f"       + {cmd[:80]}")
            
            to_remove = analysis.get("critical_commands_to_remove", [])
            if to_remove:
                print(f"\n     Commands to remove:")
                for cmd in to_remove[:5]:
                    print(f"       - {cmd[:80]}")
            
            print(f"     {'─'*60}")
            
            # ============================================================
            # SAVE ANALYSIS TO FILE
            # ============================================================
            analysis_path = self.working_dir / f"error_analysis_{Path(script_path).stem}.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"     💾 Analysis saved: {analysis_path.name}")
            
            # ============================================================
            # CHECK IF UPDATER ACTUALLY CHANGED ANYTHING
            # ============================================================
            with open(script_path, 'r') as f:
                original_content = f.read()
            
            if corrected_script.strip() == original_content.strip():
                print(f"     ⚠️  Updater returned IDENTICAL script - no fix applied!")
                
                if not issues:
                    print(f"     ❌ Updater found no issues and made no changes")
                    print(f"     Possible causes:")
                    print(f"       - Error info not reaching the updater")
                    print(f"       - LLM doesn't understand this LAMMPS error")
                    print(f"       - Error is in data file, not script")
                    return False, script_path, {
                        "error": "Updater could not identify or fix the error",
                        "analysis": analysis,
                        "log_excerpt": log_content[-300:] if log_content else "empty"
                    }
                else:
                    print(f"     ⚠️  Updater identified {len(issues)} issues but didn't change the script!")
                    print(f"     This may indicate the LLM suggested changes it couldn't implement")
                    return False, script_path, {
                        "error": "Updater identified issues but failed to modify script",
                        "analysis": analysis
                    }
            
            # Show diff summary
            orig_lines = set(original_content.strip().split('\n'))
            new_lines = set(corrected_script.strip().split('\n'))
            added = new_lines - orig_lines
            removed = orig_lines - new_lines
            
            if added or removed:
                print(f"\n     📝 Script changes: +{len(added)} lines, -{len(removed)} lines")
                if removed:
                    print(f"     Removed:")
                    for line in list(removed)[:3]:
                        stripped = line.strip()
                        if stripped and not stripped.startswith('#'):
                            print(f"       - {stripped[:80]}")
                if added:
                    print(f"     Added:")
                    for line in list(added)[:3]:
                        stripped = line.strip()
                        if stripped and not stripped.startswith('#'):
                            print(f"       + {stripped[:80]}")
            
            # ============================================================
            # SAVE CORRECTED SCRIPT
            # ============================================================
            base_name = Path(script_path).stem.split("_corrected_")[0]
            attempt_num = len(list(self.working_dir.glob(f"{base_name}_corrected_*.lammps"))) + 1
            new_script_path = self.working_dir / f"{base_name}_corrected_{attempt_num}.lammps"
            
            with open(new_script_path, 'w') as f:
                f.write(corrected_script)
            
            correction_info = {
                "original_script": script_path,
                "corrected_script": str(new_script_path),
                "analysis": analysis,
                "correction_type": "lammps_error",
                "lines_added": len(added),
                "lines_removed": len(removed)
            }
            
            print(f"     ✓ Corrected script: {new_script_path.name}")
            
            return True, str(new_script_path), correction_info
            
        except Exception as e:
            self.logger.error(f"Error correction failed: {e}")
            import traceback
            traceback.print_exc()
            return False, script_path, {"error": str(e)}

    def _get_data_file_context(self, data_path: str) -> str:
        """Extract type information from data file for error correction context."""
        context_lines = []
        
        try:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            
            # Extract comment headers with type info
            section = None
            for line in lines:
                stripped = line.strip()
                
                if stripped == "# Pair Coeffs":
                    section = "atom"
                    context_lines.append("ATOM TYPES:")
                    continue
                elif stripped == "# Bond Coeffs":
                    section = "bond"
                    context_lines.append("BOND TYPES:")
                    continue
                elif stripped == "# Angle Coeffs":
                    section = "angle"
                    context_lines.append("ANGLE TYPES:")
                    continue
                elif stripped == "# Dihedral Coeffs":
                    section = "dihedral"
                    context_lines.append("DIHEDRAL TYPES:")
                    continue
                elif stripped == "#":
                    continue
                elif not stripped.startswith("#"):
                    section = None
                    continue
                
                if section and stripped.startswith("#"):
                    parts = stripped[1:].strip().split()
                    if len(parts) >= 2:
                        try:
                            type_id = int(parts[0])
                            name = parts[1]
                            context_lines.append(f"  Type {type_id}: {name}")
                        except ValueError:
                            pass
            
            # Also get Masses section
            in_masses = False
            context_lines.append("MASSES:")
            for line in lines:
                stripped = line.strip()
                if stripped == "Masses":
                    in_masses = True
                    continue
                if in_masses:
                    if not stripped or stripped.startswith("Atoms"):
                        break
                    if stripped.startswith("#"):
                        continue
                    parts = stripped.split()
                    if len(parts) >= 2 and "#" in stripped:
                        try:
                            type_id = int(parts[0])
                            mass = float(parts[1])
                            comment = stripped.split("#")[1].strip()
                            context_lines.append(f"  Type {type_id}: mass={mass:.3f} ({comment})")
                        except (ValueError, IndexError):
                            pass
            
            # Count header info
            for line in lines[:20]:
                stripped = line.strip()
                for keyword in ["atoms", "bonds", "angles", "dihedrals", "atom types", 
                               "bond types", "angle types", "dihedral types"]:
                    if keyword in stripped.lower():
                        context_lines.append(f"  {stripped}")
                        break
            
        except Exception as e:
            context_lines.append(f"  (Could not parse data file: {e})")
        
        return "\n".join(context_lines)

    def _fix_quality_issues(self,
                           script_path: str,
                           quality_result: Dict[str, Any],
                           research_goal: str,
                           sim_info: Dict[str, Any],
                           stage: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Coordinate quality fixes across sub-agents.
        
        Delegates to:
        - ForceFieldAgent for FF params and charges
        - LAMMPSUpdater for script adjustments
        """
        from .force_field_agent import ForceFieldAgent
        
        issues = quality_result.get("issues", [])
        recommendations = quality_result.get("recommendations", [])
        
        print(f"\n     {'─'*60}")
        print(f"     🔬 QUALITY ISSUE DIAGNOSIS:")
        print(f"     {'─'*60}")
        
        if issues:
            print(f"     Issues ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                severity = issue.get("severity", "?")
                desc = issue.get("description", "")
                print(f"       {i}. [{severity.upper()}] {desc}")
        
        if recommendations:
            print(f"\n     Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                desc = rec.get("description", rec) if isinstance(rec, dict) else str(rec)
                print(f"       {i}. {desc}")
        
        print(f"     {'─'*60}")
        
        corrections_made = []
        
        try:
            # ============================================================
            # Step 1: Ask ForceFieldAgent to diagnose and fix parameters
            # ============================================================
            ff_params_path = self.working_dir / "ff_params.lammps"
            data_path = sim_info.get("data_path")
            
            if ff_params_path.exists() and data_path:
                print(f"\n     🧪 Step 1: ForceFieldAgent diagnosing parameters...")
                
                ff_agent = ForceFieldAgent(
                    working_dir=str(self.working_dir),
                    api_key=self.api_key,
                    model_name=self.model_name,
                    base_url=self.base_url
                )
                
                ff_result = ff_agent.diagnose_and_fix_force_field(
                    quality_result=quality_result,
                    research_goal=research_goal,
                    data_file=data_path,
                    ff_params_path=str(ff_params_path),
                    stage=stage
                )
                
                print(f"     Diagnosis: {ff_result.get('diagnosis', 'N/A')[:150]}")
                
                if ff_result.get("ff_modified"):
                    print(f"     ✓ Force field parameters adjusted")
                    corrections_made.append(("force_field", ff_result["details"].get("force_field", {})))
                
                if ff_result.get("charges_modified"):
                    print(f"     ✓ Charges adjusted")
                    corrections_made.append(("charges", ff_result["details"].get("charges", {})))
                
                if not ff_result.get("ff_modified") and not ff_result.get("charges_modified"):
                    print(f"     → No parameter changes needed")
            
            # ============================================================
            # Step 2: Ask LAMMPSUpdater to fix the script
            # ============================================================
            print(f"\n     ⚙️  Step 2: LAMMPSUpdater adjusting script...")
            
            ff_was_modified = any(c[0] == "force_field" for c in corrections_made)
            charges_were_modified = any(c[0] == "charges" for c in corrections_made)
            
            corrected_script, script_analysis = self.updater.refine_for_quality_issues(
                input_path=script_path,
                research_goal=research_goal,
                quality_assessment=quality_result,
                system_info=sim_info.get("system_info", {}),
                stage=stage,
                ff_was_modified=ff_was_modified,
                charges_were_modified=charges_were_modified
            )
            
            # Check if script changed
            with open(script_path, 'r') as f:
                original_script = f.read()
            
            if corrected_script.strip() != original_script.strip():
                # Save corrected script
                base_name = Path(script_path).stem.split("_quality_")[0].split("_corrected_")[0]
                attempt_num = len(list(self.working_dir.glob(f"{base_name}_quality_*.lammps"))) + 1
                new_script_path = self.working_dir / f"{base_name}_quality_{attempt_num}.lammps"
                
                with open(new_script_path, 'w') as f:
                    f.write(corrected_script)
                
                corrections_made.append(("script", {"new_script": str(new_script_path)}))
                print(f"     ✓ Script adjusted → {new_script_path.name}")
            else:
                new_script_path = script_path
                print(f"     → Script unchanged")
            
            # ============================================================
            # Summary
            # ============================================================
            if not corrections_made:
                print(f"\n     ❌ No corrections could be made")
                return False, script_path, {"error": "No corrections applied"}
            
            print(f"\n     📝 Corrections applied: {len(corrections_made)}")
            for corr_type, corr_info in corrections_made:
                summary = corr_info.get("summary", "modified")
                print(f"       - {corr_type}: {summary}")
            
            correction_info = {
                "original_script": script_path,
                "corrected_script": str(new_script_path),
                "quality_issues": [i.get("description", "") for i in issues],
                "correction_type": "quality_issue",
                "corrections_made": [{"type": t, "info": i} for t, i in corrections_made]
            }
            
            return True, str(new_script_path), correction_info
            
        except Exception as e:
            self.logger.error(f"Quality correction failed: {e}")
            import traceback
            traceback.print_exc()
            return False, script_path, {"error": str(e)}

    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _print_quality_summary(self, quality_result: Dict[str, Any]):
        """Print a formatted summary of quality check results."""
        status = quality_result.get("status", "unknown")
        can_continue = quality_result.get("can_continue", True)
        issues = quality_result.get("issues", [])
        
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "unknown": "❓"
        }
        
        print(f"  {status_emoji.get(status, '❓')} Quality: {status.upper()}")
        print(f"     Can continue: {'Yes' if can_continue else 'No'}")
        
        if issues:
            print(f"     Issues found: {len(issues)}")
            for issue in issues[:3]:  # Show first 3
                severity = issue.get("severity", "unknown")
                desc = issue.get("description", "No description")
                print(f"       [{severity.upper()}] {desc[:70]}")
            if len(issues) > 3:
                print(f"       ... and {len(issues) - 3} more")
        
        # Print key metrics if available
        metrics = quality_result.get("quality_metrics", {})
        if metrics:
            print(f"     Key metrics:")
            for check_name, check_metrics in list(metrics.items())[:2]:
                for key, value in list(check_metrics.items())[:3]:
                    if isinstance(value, (int, float)):
                        print(f"       {key}: {value:.4g}")
    
    def _failed_result(self, reason: str) -> Dict[str, Any]:
        """Generate a failed result dictionary."""
        return {
            "status": "failed",
            "reason": reason,
            "working_directory": str(self.working_dir),
            "quality_history": self.quality_history,
            "correction_history": self.correction_history,
            "stage_results": self.stage_results
        }
    
    def _partial_result(self, completed_stages: List[str], reason: str) -> Dict[str, Any]:
        """Generate a partial success result dictionary."""
        return {
            "status": "partial",
            "reason": reason,
            "working_directory": str(self.working_dir),
            "completed_stages": completed_stages,
            "stage_results": self.stage_results,
            "quality_history": self.quality_history,
            "correction_history": self.correction_history
        }
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the orchestrated simulation.
        
        Returns:
            Path to generated report
        """
        report_path = self.working_dir / "simulation_orchestration_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Supervised Simulation Report\n\n")
            f.write(f"**Working Directory:** `{self.working_dir}`\n\n")
            
            # Stage results
            f.write("## Stage Results\n\n")
            for stage_name, result in self.stage_results.items():
                status = result.get("status", "unknown")
                attempts = result.get("attempts", 0)
                
                status_emoji = {"success": "✅", "warning": "⚠️", "failed": "❌"}
                f.write(f"### {status_emoji.get(status, '❓')} {stage_name}\n")
                f.write(f"- Status: {status}\n")
                f.write(f"- Attempts: {attempts}\n\n")
            
            # Quality checks
            f.write("## Quality Checks\n\n")
            f.write(f"Total checks performed: {len(self.quality_history)}\n\n")
            
            for i, check in enumerate(self.quality_history, 1):
                stage = check.get("stage", "unknown")
                attempt = check.get("attempt", 0)
                result = check.get("result", {})
                status = result.get("status", "unknown")
                
                f.write(f"{i}. **{stage}** (attempt {attempt}): {status}\n")
                
                issues = result.get("issues", [])
                if issues:
                    for issue in issues[:3]:
                        f.write(f"   - [{issue.get('severity', '?')}] {issue.get('description', 'No description')}\n")
            
            f.write("\n")
            
            # Corrections
            f.write("## Corrections Made\n\n")
            
            if not self.correction_history:
                f.write("No corrections needed - simulation ran cleanly!\n\n")
            else:
                for i, correction in enumerate(self.correction_history, 1):
                    stage = correction.get("stage", "unknown")
                    attempt = correction.get("attempt", 0)
                    corr_type = correction.get("type", "unknown")
                    
                    f.write(f"{i}. **{stage}** (attempt {attempt})\n")
                    f.write(f"   - Type: {corr_type}\n")
                    
                    if corr_type == "lammps_error":
                        analysis = correction.get("correction", {}).get("analysis", {})
                        issues = analysis.get("issues", [])
                        if issues:
                            f.write(f"   - Issues fixed: {len(issues)}\n")
                            for issue in issues[:2]:
                                f.write(f"     - {issue.get('error_text', 'Unknown')}\n")
                    
                    elif corr_type == "quality_issue":
                        issues = correction.get("correction", {}).get("quality_issues", [])
                        if issues:
                            f.write(f"   - Quality issues addressed: {len(issues)}\n")
                            for issue in issues[:2]:
                                f.write(f"     - {issue.get('description', 'Unknown')}\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Summary report generated: {report_path}")
        return str(report_path)
      
    @staticmethod
    def _prepare_data_file(
                          input_file: str, 
                          working_dir: str,
                          box_dimensions: float = 40.0,
                          vmd_path: str = None,
                          force_reconvert: bool = False,
                          assign_charges: bool = True,
                          research_goal: Optional[str] = None,
                          api_key: Optional[str] = None,
                          model_name: str = "gemini-3-pro-preview",
                          base_url: Optional[str] = None,
                          small_molecule_info: Optional[List[Dict]] = None,
                          solvate: bool = False,
                          box_buffer: float = 10.0,
                          neutralize: bool = True,
                          prefer_amber: bool = True) -> Tuple[str, Optional[Dict]]:
        """
        Prepare LAMMPS data file from PDB or verify existing data file.
        
        When AmberTools are available and prefer_amber=True, uses the AMBER
        pipeline (antechamber → tleap → ParmEd) which produces a self-contained
        data file with all coefficients and charges. Falls back to VMD + LLM
        parameterization if AmberTools are not available.
        
        Args:
            input_file: Path to input structure file (PDB, LAMMPS data, etc.)
            working_dir: Working directory
            box_dimensions: Box size in Angstroms (only used for VMD conversion)
            vmd_path: Path to VMD executable (only needed for VMD fallback)
            force_reconvert: Force reconversion even if data file exists
            assign_charges: Whether to assign charges using ForceFieldAgent
            research_goal: Research goal
            api_key: API key for ForceFieldAgent
            model_name: Model name for ForceFieldAgent
            base_url: Base URL for ForceFieldAgent
            small_molecule_info: Non-standard residues for antechamber
                [{"pdb": <path>, "name": "LIG", "charge": 0}, ...]
            solvate: Add solvent box (AMBER pipeline only)
            box_buffer: Solvent buffer distance in Angstroms (AMBER pipeline)
            neutralize: Add counter-ions (AMBER pipeline)
            prefer_amber: Try AMBER pipeline before VMD (default True)
            
        Returns:
            Tuple of (path to LAMMPS data file, force field info dict or None)
        """
        from .force_field_agent import ForceFieldAgent
    
        input_path = Path(input_file)
    
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
    
        force_field_info = None
    
        # ────────────────────────────────────────────────────────────
        # Case 1: Already a LAMMPS data file
        # ────────────────────────────────────────────────────────────
        if input_path.suffix.lower() in ['.lammps', '.data', '.lmp']:
            logging.info(f"✓ Input is already a LAMMPS data file: {input_file}")
            data_file = str(input_path)
    
            # Still run charge assignment if requested and no coefficients present
            if assign_charges:
                try:
                    ff_agent = ForceFieldAgent(
                        working_dir=working_dir,
                        api_key=api_key,
                        model_name=model_name,
                        base_url=base_url
                    )
                    force_field_info = ff_agent.complete_parameterization(
                        pdb_file=None,
                        data_file=data_file,
                        research_goal=research_goal or "Molecular dynamics simulation"
                    )
                    if force_field_info["status"] == "success":
                        data_file = force_field_info["output_files"]["charged_data_file"]
                except Exception as e:
                    logging.warning(f"Charge assignment failed for existing data file: {e}")
    
            return data_file, force_field_info
    
        # ────────────────────────────────────────────────────────────
        # Case 2: PDB file — try AMBER pipeline first, then VMD fallback
        # ────────────────────────────────────────────────────────────
        if input_path.suffix.lower() != '.pdb':
            raise ValueError(f"Unsupported input format: {input_path.suffix}")
    
        logging.info(f"📄 Input is a PDB file: {input_file}")
    
        # Check if output already exists
        output_file = Path(working_dir) / f"{input_path.stem}.data"
        if output_file.exists() and not force_reconvert:
            logging.info(f"✓ Converted data file already exists: {output_file}")
            return str(output_file), None
    
        # ── Try AMBER pipeline ───────────────────────────────────
        if prefer_amber:
            try:
                from ...tools.amber_tools import check_amber_tools
                tools_status = check_amber_tools()
            except ImportError:
                tools_status = {"available": False, "missing": ["amber_tools module"]}
    
            if tools_status["available"]:
                logging.info("🧪 AmberTools available — using AMBER pipeline")
                try:
                    ff_agent = ForceFieldAgent(
                        working_dir=working_dir,
                        api_key=api_key,
                        model_name=model_name,
                        base_url=base_url,
                        skill="amber"
                    )
    
                    # Step 1: Select force field
                    selection = ff_agent.select_force_field(
                        pdb_file=str(input_path),
                        research_goal=research_goal or "Molecular dynamics simulation"
                    )
    
                    # Step 2: Acquire parameters via AMBER pipeline
                    force_field_info = ff_agent.acquire_parameters(
                        selection_info=selection,
                        pdb_file=str(input_path),
                        small_molecule_info=small_molecule_info,
                        solvate=solvate,
                        box_buffer=box_buffer,
                        neutralize=neutralize,
                    )
    
                    if force_field_info.get("pipeline") == "amber":
                        data_file = force_field_info["data_file"]
                        logging.info(f"✅ AMBER pipeline produced: {data_file}")
    
                        # Generate the LAMMPS input header (style commands)
                        param_files = ff_agent.generate_lammps_parameters(
                            parameter_info=force_field_info,
                            data_file=data_file,
                        )
                        force_field_info["output_files"] = {
                            "charged_data_file": data_file,
                            "parameter_files": param_files,
                        }
                        force_field_info["status"] = "success"
    
                        return data_file, force_field_info
                    else:
                        logging.info(
                            "AMBER skill active but pipeline not used "
                            "(FF may not be AMBER-family). Falling through to VMD."
                        )
    
                except Exception as e:
                    logging.warning(f"⚠️ AMBER pipeline failed: {e}")
                    logging.info("Falling back to VMD + LLM parameterization")
                    import traceback
                    traceback.print_exc()
            else:
                logging.info(
                    f"AmberTools not available (missing: {tools_status.get('missing', [])}). "
                    f"Using VMD + LLM parameterization."
                )
    
        # ── VMD fallback ─────────────────────────────────────────
        logging.info(f"🔄 Converting PDB to LAMMPS data file via VMD...")
    
        if vmd_path is None:
            vmd_path = os.getenv('VMD_PATH')
            if vmd_path is None:
                converter_temp = VMDLAMMPSConverter()
                vmd_path = converter_temp.vmd_path
    
        if not vmd_path:
            raise RuntimeError(
                "Neither AmberTools nor VMD available for PDB conversion. "
                "Install one:\n"
                "  conda install -c conda-forge ambertools parmed\n"
                "  OR set VMD_PATH environment variable"
            )
    
        converter = VMDLAMMPSConverter(
            vmd_path=vmd_path,
            working_dir=working_dir
        )
    
        data_file = converter.convert(
            pdb_file=str(input_path),
            output_file=str(output_file),
            box_dimensions=box_dimensions,
            options={
                'autobonds': True, 'retypebonds': True,
                'guessangles': True, 'guess_dihedrals': True,
                'guess_impropers': True, 'style': 'full',
                'atom_style': 'full', 'center_system': True
            }
        )
        logging.info(f"✓ VMD conversion complete: {data_file}")
    
        # Assign charges via LLM (VMD doesn't set charges)
        if assign_charges:
            logging.info(f"⚡ Assigning charges using ForceFieldAgent (LLM)...")
            try:
                ff_agent = ForceFieldAgent(
                    working_dir=working_dir,
                    api_key=api_key,
                    model_name=model_name,
                    base_url=base_url
                )
                force_field_info = ff_agent.complete_parameterization(
                    pdb_file=str(input_path),
                    data_file=data_file,
                    research_goal=research_goal or "Molecular dynamics simulation"
                )
                if force_field_info["status"] == "success":
                    data_file = force_field_info["output_files"]["charged_data_file"]
                    logging.info(f"✓ Charges assigned: {data_file}")
                else:
                    logging.warning(f"⚠️ Charge assignment issues: {force_field_info.get('errors', [])}")
            except Exception as e:
                logging.error(f"❌ Charge assignment failed: {e}")
                logging.warning("Continuing with uncharged data file")
    
        return data_file, force_field_info

    @classmethod
    def quick_run(cls,
                  input_file: str,
                  research_goal: str,
                  working_dir: str,
                  lammps_command: str = "lmp",
                  vmd_path: str = None,
                  max_stage_attempts: int = 3,
                  run_final_analysis: bool = True,
                  temperature: float = 300.0,
                  pressure: float = 1.0,
                  box_dimensions: float = 40.0,
                  force_reconvert: bool = False,
                  assign_charges: bool = True,
                  stage_timeout: int = 14400,
                  small_molecule_info: Optional[List[Dict]] = None,
                  solvate: bool = False,
                  box_buffer: float = 10.0,
                  neutralize: bool = True,
                  prefer_amber: bool = True,
                  **kwargs) -> Dict:
        """
        Quick run with automatic file format detection, conversion, and parameterization.
        
        When AmberTools are available and the input is a PDB, automatically uses the
        AMBER pipeline (antechamber → tleap → ParmEd) for production-quality parameters.
        Falls back to VMD + LLM parameterization otherwise.
        """
        os.makedirs(working_dir, exist_ok=True)
    
        logging.info("=" * 80)
        logging.info("LAMMPS ORCHESTRATOR - QUICK RUN")
        logging.info("=" * 80)
    
        api_key = kwargs.get('api_key') or os.getenv("GOOGLE_API_KEY")
        model_name = kwargs.get('model_name', "gemini-3-pro-preview")
        base_url = kwargs.get('base_url')
    
        # Step 1: Prepare data file
        logging.info("\n📋 Step 1: Preparing data file...")
    
        conversion_info = {
            'input_file': input_file,
            'input_type': Path(input_file).suffix,
            'converted': False,
            'charges_assigned': False,
            'pipeline': None,
            'conversion_time': None
        }
    
        try:
            import time as _time
            start_time = _time.time()
    
            prepared_data_file, force_field_info = cls._prepare_data_file(
                input_file=input_file,
                working_dir=working_dir,
                box_dimensions=box_dimensions,
                vmd_path=vmd_path,
                force_reconvert=force_reconvert,
                assign_charges=assign_charges,
                research_goal=research_goal,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                small_molecule_info=small_molecule_info,
                solvate=solvate,
                box_buffer=box_buffer,
                neutralize=neutralize,
                prefer_amber=prefer_amber,
            )
    
            conversion_time = _time.time() - start_time
    
            # Detect which pipeline was used
            pipeline_used = "unknown"
            if force_field_info:
                if force_field_info.get("pipeline") == "amber":
                    pipeline_used = "amber"
                elif force_field_info.get("status") == "success":
                    pipeline_used = "vmd+llm"
    
            conversion_info.update({
                'prepared_file': prepared_data_file,
                'converted': (Path(prepared_data_file).name != Path(input_file).name),
                'charges_assigned': force_field_info is not None,
                'pipeline': pipeline_used,
                'conversion_time': conversion_time,
                'force_field_info': force_field_info
            })
    
            logging.info(f"✓ Data preparation completed in {conversion_time:.2f}s")
            logging.info(f"  Pipeline: {pipeline_used}")
    
        except Exception as e:
            logging.error(f"❌ Data file preparation failed: {e}")
            return {
                'status': 'failed',
                'reason': f'Data file preparation failed: {str(e)}',
                'stage': 'preparation',
                'working_directory': working_dir,
                'conversion_info': conversion_info,
                'error': str(e)
            }
    
        # Step 2: Initialize orchestrator
        logging.info("\n🔧 Step 2: Initializing orchestrator...")
    
        try:
            orchestrator = cls(
                working_dir=working_dir,
                lammps_command=lammps_command,
                max_stage_attempts=max_stage_attempts,
                stage_timeout=stage_timeout,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url
            )
        except Exception as e:
            logging.error(f"❌ Orchestrator initialization failed: {e}")
            return {
                'status': 'failed',
                'reason': f'Orchestrator initialization failed: {str(e)}',
                'stage': 'initialization',
                'working_directory': working_dir,
                'conversion_info': conversion_info,
                'error': str(e)
            }
    
        # Step 3: Run simulation
        logging.info("\n🚀 Step 3: Running simulation stages...")
    
        force_field_files = None
        if force_field_info and force_field_info.get("status") == "success":
            param_files = force_field_info.get("output_files", {}).get("parameter_files", {})
            if param_files:
                force_field_files = param_files
    
        results = orchestrator.run_supervised_simulation(
            data_file=prepared_data_file,
            research_goal=research_goal,
            force_field_files=force_field_files,
            run_final_analysis=run_final_analysis
        )
    
        results['conversion_info'] = conversion_info
    
        return results
