import json
import logging
import uuid
from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class BaseAgent(ABC):
    """
    Base class for all agents in the planning_agents module.
    
    Provides standardized:
    - State management (initialization, persistence, restoration)
    - Action logging (input, result, rationale, feedback tracking)
    - Session management (unique IDs, timestamps)
    
    All planning agents should inherit from this class to ensure 
    consistent state handling and traceability.
    
    Subclasses must:
    1. Call super().__init__(output_dir) in their __init__
    2. Set self.agent_type to a unique identifier (e.g., "planning", "bo", "scalarizer")
    3. Override _get_initial_state_fields() to add agent-specific state fields
    
    Example:
        class MyAgent(BaseAgent):
            def __init__(self, output_dir: str = "."):
                super().__init__(output_dir)
                self.agent_type = "my_agent"
            
            def _get_initial_state_fields(self) -> Dict[str, Any]:
                return {
                    "my_custom_field": [],
                    "another_field": None
                }
    """
    
    def __init__(self, output_dir: str = "."):
        """
        Initialize the base agent.
        
        Args:
            output_dir: Directory for state persistence and outputs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent identifier - subclasses should override
        self.agent_type: str = "base"
        
        # Core state dictionary
        self.state: Dict[str, Any] = {}
    
    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """
        Override in subclasses to add agent-specific state fields.
        
        Returns:
            Dict of additional fields to include in state initialization.
        """
        return {}
    
    def _init_state(self, **context) -> None:
        """
        Initialize state for a new session.
        
        Creates a fresh state dictionary with:
        - Unique session ID
        - Timestamp
        - Action history for traceability
        - Any context passed as kwargs
        - Agent-specific fields from _get_initial_state_fields()
        
        Args:
            **context: Key-value pairs to store in state (e.g., objective, data_path)
        """
        if self.state.get("session_id") is None:
            self.state = {
                "session_id": str(uuid.uuid4()),
                "start_time": datetime.now().isoformat(),
                "agent_type": self.agent_type,
                "action_history": [],
                "status": "initialized"
            }
            
            # Add agent-specific fields
            self.state.update(self._get_initial_state_fields())
        
        # Update with provided context
        for key, value in context.items():
            self.state[key] = value
        
        self.state["status"] = "active"
    
    def _log_action(self, 
                    action: str, 
                    input_ctx: Dict[str, Any], 
                    result: Dict[str, Any], 
                    rationale: Optional[str] = None, 
                    feedback: Optional[str] = None) -> None:
        """
        Record an atomic action to state history.
        
        Captures the full context chain:
        - Input: What was asked?
        - Result: What was the output?
        - Rationale: Why did the agent choose this path?
        - Feedback: Did a human intervene or correct?
        
        Auto-saves state after logging.
        
        Args:
            action: Name of the action (e.g., "generate_plan", "run_optimization")
            input_ctx: Dictionary describing the input/request
            result: Dictionary describing the outcome
            rationale: Optional explanation of why this action was taken
            feedback: Optional human feedback that influenced this action
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "input": input_ctx,
            "rationale": rationale,
            "result": self._normalize_result(result),
            "feedback": feedback
        }
        
        if "action_history" not in self.state:
            self.state["action_history"] = []
        
        self.state["action_history"].append(entry)
        self._save_state()
    
    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """
        Normalize result to a consistent dictionary format.
        
        Args:
            result: Raw result (dict, string, or other)
            
        Returns:
            Normalized dictionary with status, error, and other fields
        """
        if isinstance(result, dict):
            return {
                "status": result.get("status", "completed"),
                "error": result.get("error"),
                "iteration": result.get("iteration"),
                "stage": result.get("stage")
            }
        return {
            "status": "completed",
            "error": None,
            "iteration": None,
            "stage": None
        }
    
    def _get_state_filename(self) -> str:
        """
        Get the filename for state persistence.
        
        Returns:
            Filename based on agent_type (e.g., "planning_state.json")
        """
        return f"{self.agent_type}_state.json"
    
    def _save_state(self) -> None:
        """
        Persist state to disk.
        
        Saves to {output_dir}/{agent_type}_state.json
        Called automatically after each _log_action().
        """
        state_file = self.output_dir / self._get_state_filename()
        try:
            with open(state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save {self.agent_type} state: {e}")
    
    def load_state(self, state_path: str) -> bool:
        """
        Restore state from disk.
        
        Args:
            state_path: Path to the state JSON file
            
        Returns:
            True if successful, False otherwise
        """
        path = Path(state_path)
        if not path.exists():
            logging.warning(f"State file not found: {state_path}")
            return False
        
        try:
            with open(path, 'r') as f:
                self.state = json.load(f)
            
            if "action_history" not in self.state:
                self.state["action_history"] = []
            
            logging.info(f"Restored {self.agent_type} state: session {self.state.get('session_id')}")
            return True
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in state file: {e}")
            return False
        except Exception as e:
            logging.warning(f"Failed to load {self.agent_type} state: {e}")
            return False
    
    def get_action_count(self) -> int:
        """Get the number of logged actions in this session."""
        return len(self.state.get("action_history", []))
    
    def get_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.state.get("session_id")
    
    def is_initialized(self) -> bool:
        """Check if the agent has an active session."""
        return bool(self.state.get("session_id"))
