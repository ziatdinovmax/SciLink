"""
Planning Tools for MCP - FIXED VERSION (Lazy Imports)

Wrapper layer that adapts SciLink planning agents for MCP context.
Handles non-interactive execution and output formatting.

IMPORTANT: All heavy imports are done inside functions to prevent
timeout during MCP server initialization.
"""

import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

# NO HEAVY IMPORTS HERE - They're moved inside functions below
# This prevents loading FAISS, Google AI, etc. during module import

logger = logging.getLogger(__name__)


class MCPPlanningAgent:
    """
    Wrapper around PlanningAgent that enforces MCP-compatible behavior:
    - No interactive prompts
    - Structured JSON responses
    - Comprehensive error handling
    """
    
    def __init__(self):
        """Initialize with MCP-optimized settings."""
        try:
            # LAZY IMPORT - Only loads when actually creating an agent
            from scilink.agents.planning_agents.planning_agent import PlanningAgent
            
            self.agent = PlanningAgent(quiet_mode=True)
            logger.info("Planning agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize planning agent: {e}")
            raise
    
    def _suppress_output(self, func, *args, **kwargs):
        """
        Execute function while suppressing print statements.
        Returns (result, captured_output)
        """
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        
        try:
            result = func(*args, **kwargs)
            output = captured.getvalue()
            return result, output
        finally:
            sys.stdout = old_stdout
    
    def _ensure_mcp_mode(self, kwargs: Dict) -> Dict:
        """
        Ensure arguments are set for non-interactive MCP execution.
        """
        mcp_kwargs = kwargs.copy()
        
        # Force non-interactive mode
        mcp_kwargs['enable_human_feedback'] = False
        
        # Set default output paths if not provided
        if 'output_code_dir' not in mcp_kwargs:
            mcp_kwargs['output_code_dir'] = './mcp_outputs/scripts'
        
        if 'output_json_path' in mcp_kwargs and mcp_kwargs['output_json_path']:
            # Ensure output directory exists
            Path(mcp_kwargs['output_json_path']).parent.mkdir(parents=True, exist_ok=True)
        
        return mcp_kwargs
    
    def propose_experiments(self, **kwargs) -> Dict[str, Any]:
        """
        Generate experimental plan in MCP-compatible way.
        
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        try:
            mcp_kwargs = self._ensure_mcp_mode(kwargs)
            
            logger.info(f"Proposing experiments: {kwargs.get('objective', 'N/A')[:50]}...")
            
            # Execute with output suppression
            state, output = self._suppress_output(
                self.agent.propose_experiments,
                **mcp_kwargs
            )
            
            if state.get('status') == 'failed':
                return {
                    'success': False,
                    'error': state.get('last_error', 'Unknown error'),
                    'log': output
                }
            
            # Format successful response
            return {
                'success': True,
                'data': {
                    'session_id': state.get('session_id'),
                    'iteration': state.get('iteration_index'),
                    'status': state.get('status'),
                    'objective': state.get('objective'),
                    'current_plan': state.get('current_plan'),
                    'outputs': {
                        'json_path': mcp_kwargs.get('output_json_path'),
                        'code_dir': mcp_kwargs.get('output_code_dir'),
                        'state_path': f"{mcp_kwargs.get('output_json_path')}.state.json" if mcp_kwargs.get('output_json_path') else None
                    }
                },
                'log': output
            }
            
        except Exception as e:
            logger.error(f"Error in propose_experiments: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_plan_with_results(self, **kwargs) -> Dict[str, Any]:
        """
        Update experimental plan based on results.
        
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        try:
            mcp_kwargs = self._ensure_mcp_mode(kwargs)
            
            logger.info("Updating plan with results...")
            
            # Execute with output suppression
            state, output = self._suppress_output(
                self.agent.update_plan_with_results,
                **mcp_kwargs
            )
            
            if state.get('status') == 'failed':
                return {
                    'success': False,
                    'error': state.get('last_error', 'Unknown error'),
                    'log': output
                }
            
            return {
                'success': True,
                'data': {
                    'session_id': state.get('session_id'),
                    'iteration': state.get('iteration_index'),
                    'status': state.get('status'),
                    'current_plan': state.get('current_plan'),
                    'results_processed': len(state.get('experimental_results', [])),
                    'outputs': {
                        'json_path': mcp_kwargs.get('output_json_path'),
                        'code_dir': mcp_kwargs.get('output_code_dir'),
                        'state_path': f"{mcp_kwargs.get('output_json_path')}.state.json" if mcp_kwargs.get('output_json_path') else None
                    }
                },
                'log': output
            }
            
        except Exception as e:
            logger.error(f"Error in update_plan_with_results: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def perform_tea(self, **kwargs) -> Dict[str, Any]:
        """
        Perform technoeconomic analysis.
        
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        try:
            logger.info(f"Performing TEA: {kwargs.get('objective', 'N/A')[:50]}...")
            
            # Execute with output suppression
            result, output = self._suppress_output(
                self.agent.perform_technoeconomic_analysis,
                **kwargs
            )
            
            if result.get('error'):
                return {
                    'success': False,
                    'error': result['error'],
                    'log': output
                }
            
            return {
                'success': True,
                'data': {
                    'assessment': result.get('technoeconomic_assessment'),
                    'literature_search': result.get('literature_search'),
                    'output_path': kwargs.get('output_json_path')
                },
                'log': output
            }
            
        except Exception as e:
            logger.error(f"Error in perform_tea: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def restore_state(self, state_file_path: str) -> Dict[str, Any]:
        """
        Restore agent state from file.
        
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        try:
            logger.info(f"Restoring state from: {state_file_path}")
            
            self.agent.restore_state(state_file_path)
            
            return {
                'success': True,
                'data': {
                    'message': f"State restored from {state_file_path}",
                    'session_id': self.agent.state.get('session_id'),
                    'iteration': self.agent.state.get('iteration_index'),
                    'objective': self.agent.state.get('objective')
                }
            }
            
        except Exception as e:
            logger.error(f"Error restoring state: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def refine_plan(self, feedback: str, state_file_path: str, 
                    output_json_path: str = None) -> Dict[str, Any]:
        """
        Refine experimental plan based on user feedback.
        This implements the FIRST critical feedback stage (after hypothesis generation).
        
        Args:
            feedback: User's feedback or requested changes
            state_file_path: Path to current state file
            output_json_path: Optional path to save refined plan
        
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        try:
            logger.info(f"Refining plan with feedback: {feedback[:50]}...")
            
            # Restore state
            self.agent.restore_state(state_file_path)
            
            # Get current plan
            current_plan = self.agent.state.get('current_plan')
            if not current_plan:
                return {
                    'success': False,
                    'error': 'No current plan found in state'
                }
            
            objective = self.agent.state.get('objective')
            
            # LAZY IMPORT - Only load when actually needed
            from scilink.agents.planning_agents.rag_engine import refine_plan_with_feedback
            
            # Refine the plan with output suppression
            refined_plan, output = self._suppress_output(
                refine_plan_with_feedback,
                original_result=current_plan,
                feedback=feedback,
                objective=objective,
                model=self.agent.model,
                generation_config=self.agent.generation_config
            )
            
            # Update state
            self.agent.state['current_plan'] = refined_plan
            self.agent.state['plan_history'].append({
                **refined_plan,
                'iteration': self.agent.state['iteration_index'],
                'stage': 'User Refined (Plan)'
            })
            self.agent.state['human_feedback_history'].append({
                'phase': 'plan_refinement',
                'feedback': feedback
            })
            
            # Save if path provided
            if output_json_path:
                self.agent._save_results_to_json(refined_plan, output_json_path)
                self.agent._save_state_to_json(output_json_path + ".state.json")
                self.agent._generate_html_report(output_json_path)
            
            return {
                'success': True,
                'data': {
                    'session_id': self.agent.state.get('session_id'),
                    'iteration': self.agent.state.get('iteration_index'),
                    'refined_plan': refined_plan,
                    'outputs': {
                        'json_path': output_json_path,
                        'state_path': f"{output_json_path}.state.json" if output_json_path else None
                    }
                },
                'log': output
            }
            
        except Exception as e:
            logger.error(f"Error refining plan: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def refine_code(self, feedback: str, state_file_path: str,
                    output_json_path: str = None,
                    output_code_dir: str = "./mcp_outputs/scripts") -> Dict[str, Any]:
        """
        Refine implementation code based on user feedback.
        This implements the SECOND critical feedback stage (after code generation).
        
        Args:
            feedback: User's feedback or requested changes to code
            state_file_path: Path to current state file
            output_json_path: Optional path to save plan with refined code
            output_code_dir: Directory to save refined scripts
        
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        try:
            logger.info(f"Refining code with feedback: {feedback[:50]}...")
            
            # Restore state
            self.agent.restore_state(state_file_path)
            
            # Get current plan
            current_plan = self.agent.state.get('current_plan')
            if not current_plan:
                return {
                    'success': False,
                    'error': 'No current plan found in state'
                }
            
            # LAZY IMPORTS - Only load when actually needed
            from scilink.agents.planning_agents.rag_engine import refine_code_with_feedback
            from scilink.agents.planning_agents.parser_utils import write_experiments_to_disk
            
            # Refine the code with output suppression
            refined_plan, output = self._suppress_output(
                refine_code_with_feedback,
                result=current_plan,
                feedback=feedback,
                model=self.agent.model,
                generation_config=self.agent.generation_config
            )
            
            # Update state
            self.agent.state['current_plan'] = refined_plan
            self.agent.state['plan_history'].append({
                **refined_plan,
                'iteration': self.agent.state['iteration_index'],
                'stage': 'User Refined (Code)'
            })
            self.agent.state['human_feedback_history'].append({
                'phase': 'code_refinement',
                'feedback': feedback
            })
            
            # Write refined code to disk
            write_experiments_to_disk(refined_plan, output_code_dir)
            
            # Save if path provided
            if output_json_path:
                self.agent._save_results_to_json(refined_plan, output_json_path)
                self.agent._save_state_to_json(output_json_path + ".state.json")
                self.agent._generate_html_report(output_json_path)
            
            return {
                'success': True,
                'data': {
                    'session_id': self.agent.state.get('session_id'),
                    'iteration': self.agent.state.get('iteration_index'),
                    'refined_plan': refined_plan,
                    'outputs': {
                        'json_path': output_json_path,
                        'state_path': f"{output_json_path}.state.json" if output_json_path else None,
                        'code_dir': output_code_dir
                    }
                },
                'log': output
            }
            
        except Exception as e:
            logger.error(f"Error refining code: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


class MCPBOAgent:
    """
    Wrapper around BOAgent for MCP context.
    """
    
    def __init__(self):
        """Initialize BO agent."""
        try:
            # LAZY IMPORT - Only loads when actually creating an agent
            from scilink.agents.planning_agents.bo_agent import BOAgent
            
            self.agent = BOAgent()
            logger.info("BO agent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize BO agent: {e}")
            raise
    
    def run_optimization(self, **kwargs) -> Dict[str, Any]:
        """
        Run Bayesian optimization loop.
        
        Returns:
            Dict with 'success', 'data', and optional 'error' keys
        """
        try:
            logger.info(f"Running BO: {kwargs.get('objective_text', 'N/A')[:50]}...")
            
            result = self.agent.run_optimization_loop(**kwargs)
            
            if result.get('error'):
                return {
                    'success': False,
                    'error': result['error']
                }
            
            return {
                'success': True,
                'data': {
                    'status': result.get('status'),
                    'next_parameters': result.get('next_parameters'),
                    'strategy': result.get('strategy'),
                    'plot_path': result.get('plot_path')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in run_optimization: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instances - NOT created here, only when get_* functions are called
_planning_agent: Optional[MCPPlanningAgent] = None
_bo_agent: Optional[MCPBOAgent] = None


def get_planning_agent() -> MCPPlanningAgent:
    """Get or create planning agent singleton."""
    global _planning_agent
    if _planning_agent is None:
        logger.info("Creating planning agent singleton...")
        _planning_agent = MCPPlanningAgent()
    return _planning_agent


def get_bo_agent() -> MCPBOAgent:
    """Get or create BO agent singleton."""
    global _bo_agent
    if _bo_agent is None:
        logger.info("Creating BO agent singleton...")
        _bo_agent = MCPBOAgent()
    return _bo_agent