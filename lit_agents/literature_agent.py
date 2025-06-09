import os
import logging
import json
from time import sleep

try:
    from futurehouse_client import FutureHouseClient, JobNames
except ImportError:
    logging.error("Error: futurehouse_client is not installed. Please install it with pip.")
    raise

class OwlLiteratureAgent:
    """
    Agent for querying scientific literature using the OWL system
    through the FutureHouse API client.
    """

    def __init__(self, api_key: str | None = None, max_wait_time: int = 300):
        """
        Initialize the OWL literature agent.
        
        Args:
            api_key: FutureHouse API key
            max_wait_time: Maximum time to wait for response in seconds
        """
        if api_key is None:
            api_key = os.environ.get("FUTUREHOUSE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and FUTUREHOUSE_API_KEY environment variable is not set.")
        
        self.client = FutureHouseClient(api_key=api_key)
        self.max_wait_time = max_wait_time
        logging.info("OWLLiteratureAgent initialized with max wait time of %d seconds.", max_wait_time)

    def query_literature(self, has_anyone_question: str) -> dict:
        """
        Query the scientific literature using the OWL system.
        
        Args:
            has_anyone_question: The question in "Has anyone..." format
            
        Returns:
            Dictionary containing the search results and metadata
        """
        if not has_anyone_question or not isinstance(has_anyone_question, str):
            error_msg = "Invalid question format. Must provide a non-empty string."
            logging.error(error_msg)
            return {"status": "error", "message": error_msg}
            
        try:
            logging.info(f"Submitting literature query: {has_anyone_question}")
            
            # Create the task in OWL
            task_data = {
                "name": JobNames.OWL,
                "query": has_anyone_question
            }
            
            task_id = self.client.create_task(task_data)
            logging.info(f"OWL task created with ID: {task_id}")
            
            # Get the initial response
            task_status = self.client.get_task(task_id)
            
            # Check if the response is already complete
            if task_status.status == "success":
                logging.info("OWL query completed immediately.")
                return {
                    "status": "success",
                    "task_id": task_id,
                    "formatted_answer": task_status.formatted_answer,
                    "has_successful_answer": getattr(task_status, 'has_successful_answer', True),
                    "search_results": getattr(task_status, 'search_results', []),
                    "query": has_anyone_question
                }
            
            # If not complete, wait for the response with a single timeout
            logging.info(f"OWL query in progress. Waiting up to {self.max_wait_time} seconds for completion.")
            
            # Calculate end time based on max_wait_time
            import time
            start_time = time.time()
            end_time = start_time + self.max_wait_time
            
            while time.time() < end_time:
                # Wait a bit before checking again
                sleep_time = min(10, max(1, (end_time - time.time()) / 10))
                sleep(sleep_time)
                
                # Check status again
                task_status = self.client.get_task(task_id)
                
                # If complete, return the results
                if task_status.status == "success":
                    elapsed = time.time() - start_time
                    logging.info(f"OWL query completed after {elapsed:.1f} seconds.")
                    return {
                        "status": "success",
                        "task_id": task_id,
                        "formatted_answer": task_status.formatted_answer,
                        "json": task_status.model_dump_json(),
                        "has_successful_answer": getattr(task_status, 'has_successful_answer', True),
                        "search_results": getattr(task_status, 'search_results', []),
                        "query": has_anyone_question
                    }
                
                if task_status.status in ["FAILED", "ERROR", "error"]:
                    error_msg = f"OWL query failed with status: {task_status.status}"
                    logging.error(error_msg)
                    return {"status": "error", "message": error_msg, "task_id": task_id}
                
                #logging.info(f"OWL query still in progress. Status: {task_status.status}")
            
            # If we get here, we've exceeded the maximum wait time
            error_msg = f"OWL query timed out after {self.max_wait_time} seconds."
            logging.error(error_msg)
            return {"status": "timeout", "message": error_msg, "task_id": task_id}
            
        except Exception as e:
            error_msg = f"An unexpected error occurred during OWL query: {str(e)}"
            logging.exception(error_msg)
            return {"status": "error", "message": error_msg}
        


class IncarLiteratureAgent:
    """Agent for validating VASP INCAR parameters against literature using CROW."""

    def __init__(self, api_key: str = None, max_wait_time: int = 300):
        if not api_key:
            api_key = os.environ.get("FUTUREHOUSE_API_KEY")
        if not api_key:
            raise ValueError("API key required")
        
        self.client = FutureHouseClient(api_key=api_key)
        self.max_wait_time = max_wait_time
        self.logger = logging.getLogger(__name__)

    def validate_incar(self, incar_content: str, system_description: str) -> dict:
        """Validate INCAR parameters against literature."""
        
        # Clean system description - remove additional instructions
        clean_description = self._clean_system_description(system_description)
        
        query = f"""Are these VASP INCAR parameters appropriate for {clean_description}?

{incar_content}"""

        try:
            # Submit to CROW
            task_data = {"name": JobNames.CROW, "query": query}
            task_id = self.client.create_task(task_data)
            
            # Wait for completion
            import time
            start_time = time.time()
            
            while time.time() - start_time < self.max_wait_time:
                task_status = self.client.get_task(task_id)
                
                if task_status.status == "success":
                    # Clean response to remove repeated question
                    clean_response = self._clean_response(task_status.formatted_answer, query)
                    return {
                        "status": "success", 
                        "response": clean_response,
                        "task_id": task_id
                    }
                elif task_status.status in ["FAILED", "ERROR", "error"]:
                    return {"status": "error", "message": f"CROW failed: {task_status.status}"}
                
                sleep(10)
            
            return {"status": "timeout", "message": f"Timed out after {self.max_wait_time}s"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _clean_system_description(self, description: str) -> str:
        """Remove additional instructions from system description."""
        # Remove common additional instruction patterns
        patterns_to_remove = [
            r"\.?\s*Additional Instructions?:.*",
            r"\.?\s*Save.*format\.?",
            r"\.?\s*Output.*format\.?",
        ]
        
        import re
        cleaned = description
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    def _clean_response(self, response: str, original_query: str) -> str:
        """Remove repeated question from CROW response."""
        if not response:
            return response
            
        # If response starts with "Question:" remove everything up to the actual answer
        if response.startswith("Question:"):
            lines = response.split('\n')
            # Find where the actual answer starts (after the question block)
            answer_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith("Question:") and not line.startswith("INCAR content:"):
                    answer_start = i
                    break
            response = '\n'.join(lines[answer_start:])
        
        return response.strip()