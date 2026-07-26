import os
import logging
import json
import time
from time import sleep
from typing import Dict, Any, Optional


try:
    from edison_client import EdisonClient, JobNames
    _EDISON_IMPORT_ERROR = None
except Exception as _e:  # noqa: BLE001 - ImportError OR any transitive
    # breakage inside the FutureHouse chain (edison-client -> ldp -> fhlmi;
    # e.g. a mismatched fhlmi raising "No module named 'lmi.config'").
    EdisonClient, JobNames = None, None
    _EDISON_IMPORT_ERROR = _e
    logging.warning(
        "Literature stack unavailable (%s). Core analysis is unaffected; "
        "literature tools will report this when invoked.", _e)


def _require_edison():
    """Fail at CONSTRUCTION, not import, when the optional literature stack
    is broken or absent — a module-scope raise here used to brick the whole
    agent initialization (UI startup) over an ancillary feature. Same
    pattern as the meta agent's guarded `ase` import (see CLAUDE.md,
    'guarded import inside the function')."""
    if EdisonClient is None:
        raise RuntimeError(
            f"Literature features are unavailable: {_EDISON_IMPORT_ERROR}. "
            "The literature stack is optional for core analysis. To enable "
            "it, install a mutually consistent set: "
            "pip install -U edison-client ldp fhlmi fhaviary"
        )

class OwlLiteratureAgent:
    """
    Agent for querying scientific literature using the OWL system
    through the FutureHouse (now knon as Edison) API client.
    """

    def __init__(self, api_key: str | None = None, max_wait_time: int = 300):
        """
        Initialize the OWL literature agent.
        
        Args:
            api_key: FutureHouse (Edison) API key
            max_wait_time: Maximum time to wait for response in seconds
        """
        if api_key is None:
            api_key = os.environ.get("FUTUREHOUSE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and FUTUREHOUSE_API_KEY environment variable is not set.")
        
        _require_edison()
        self.client = EdisonClient(api_key=api_key)
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
                "name": JobNames.PRECEDENT,
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
    """Agent for validating VASP INCAR parameters against literature using the FutureHouse CROW system."""

    def __init__(self, api_key: str = None, max_wait_time: int = 300):
        if not api_key:
            api_key = os.environ.get("FUTUREHOUSE_API_KEY")
        if not api_key:
            raise ValueError("API key required")
        
        _require_edison()
        self.client = EdisonClient(api_key=api_key)
        self.max_wait_time = max_wait_time
        self.logger = logging.getLogger(__name__)

    def validate_inputs(self, input_files_text: str, system_description: str,
                        engine_label: str = "DFT") -> dict:
        """Validate engine input parameters against literature.

        Engine-neutral entry point: ``engine_label`` names the engine in
        the query (e.g. "VASP INCAR", "Quantum ESPRESSO", "LAMMPS") so the
        same literature mechanism grounds parameter review for any engine.

        Args:
            input_files_text: The input file contents to review.
            system_description: What system the inputs are for.
            engine_label: Human-readable engine name for the query.

        Returns:
            ``{status, response, task_id}`` on success, or an error /
            timeout status dict.
        """
        clean_description = self._clean_system_description(system_description)
        query = (
            f"Are these {engine_label} input parameters appropriate for "
            f"{clean_description}?\n\n{input_files_text}"
        )
        return self._run_literature_query(query)

    def validate_incar(self, incar_content: str, system_description: str) -> dict:
        """Validate VASP INCAR parameters against literature.

        Thin VASP-specific wrapper over :meth:`validate_inputs`.
        """
        return self.validate_inputs(
            input_files_text=incar_content,
            system_description=system_description,
            engine_label="VASP INCAR",
        )

    def _run_literature_query(self, query: str) -> dict:
        """Submit a literature query to CROW and poll until it resolves.

        Args:
            query: The fully-built natural-language query.

        Returns:
            ``{status, response, task_id}`` on success; an error or
            timeout status dict otherwise.
        """
        try:
            task_data = {"name": JobNames.LITERATURE, "query": query}
            task_id = self.client.create_task(task_data)

            import time
            start_time = time.time()

            while time.time() - start_time < self.max_wait_time:
                task_status = self.client.get_task(task_id)

                if task_status.status == "success":
                    clean_response = self._clean_response(task_status.formatted_answer, query)
                    return {
                        "status": "success",
                        "response": clean_response,
                        "task_id": task_id,
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
    

class FittingModelLiteratureAgent:
    """
    Agent for querying scientific literature for physical models and
    analysis methods using the FutureHouse CROW system.
    """

    def __init__(self, api_key: str | None = None, max_wait_time: int = 300):
        if api_key is None:
            api_key = os.environ.get("FUTUREHOUSE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and FUTUREHOUSE_API_KEY env variable is not set.")
        
        _require_edison()
        self.client = EdisonClient(api_key=api_key)
        self.max_wait_time = max_wait_time
        self.logger = logging.getLogger(__name__)
        logging.info("FittingModelLiteratureAgent initialized to use CROW.")

    def query_for_models(self, search_query: str) -> dict:
        """
        Query the literature for analysis models and methods using CROW.
        
        Args:
            search_query: A specific question about finding models or methods.
            
        Returns:
            Dictionary with the search results.
        """
        self.logger.info(f"Submitting model search query to CROW: {search_query}")
        
        try:
            # Submit to CROW
            task_data = {"name": JobNames.LITERATURE, "query": search_query}
            task_id = self.client.create_task(task_data)
            
            # Wait for completion using the correct polling pattern
            import time
            start_time = time.time()
            
            while time.time() - start_time < self.max_wait_time:
                task_status = self.client.get_task(task_id)
                
                if task_status.status == "success":
                    self.logger.info("CROW model search query completed.")
                    return {
                        "status": "success",
                        "formatted_answer": task_status.formatted_answer,
                        "task_id": task_id
                    }
                elif task_status.status in ["FAILED", "ERROR", "error"]:
                    error_msg = f"CROW model search failed with status: {task_status.status}"
                    self.logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
                
                sleep(10) # Wait before polling again
            
            return {"status": "timeout", "message": f"Timed out after {self.max_wait_time}s"}
            
        except Exception as e:
            error_msg = f"An unexpected error occurred during CROW model search: {str(e)}"
            self.logger.exception(error_msg)
            return {"status": "error", "message": error_msg}
        

class LiteratureSearchAgent:
    """
    A dedicated agent for querying external scientific literature systems.
    It abstracts the polling logic and provides specialized query formats for 
    different research phases (Hypothesis, Modeling, TEA).
    """

    def __init__(self, api_key: Optional[str] = None, max_wait_time: int = 300):
        self.api_key = api_key or os.environ.get("FUTUREHOUSE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key required for Literature Agent.")
        
        _require_edison()
        self.client = EdisonClient(api_key=self.api_key)
        self.max_wait_time = max_wait_time
        self.logger = logging.getLogger("LitAgent")

    def _execute_crow_task(self, query: str, task_type: str = "general") -> Dict[str, Any]:
        """
        Internal helper: Handles the async submission and polling logic.
        """
        try:
            self.logger.info(f"🚀 Submitting ({task_type}) query: {query}")
            
            # 1. Create Task
            task_data = {"name": JobNames.LITERATURE, "query": query}
            task_id = self.client.create_task(task_data)
            
            # 2. Poll for Completion
            start_time = time.time()
            while (time.time() - start_time) < self.max_wait_time:
                task_status = self.client.get_task(task_id)
                status = task_status.status.lower()
                
                if status == "success":
                    self.logger.info(f"✅ {task_type} search completed.")
                    return {
                        "status": "success",
                        "content": task_status.formatted_answer,
                        "sources": [s.url for s in getattr(task_status, 'sources', [])] 
                    }
                elif status in ["failed", "error"]:
                    return {"status": "error", "message": f"Remote status: {status}"}
                
                time.sleep(5) # Wait before next poll
            
            return {"status": "timeout", "message": "Request timed out."}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    # --- DEDICATED QUERY STRATEGIES ---

    def search_for_hypothesis_context(self, objective: str) -> Dict[str, Any]:
        """
        Formats a query for grounding a research plan. Focuses on methods,
        mechanisms, and gaps — framed neutrally across experimental and
        computational work, so a modeling/simulation objective is not
        answered through a lab-technique lens (and vice versa).
        """
        formatted_query = (
            f"Provide a comprehensive review of the relevant methods — "
            f"experimental and/or computational, as fits the topic — "
            f"underlying physical mechanisms, and recent advancements "
            f"related to: '{objective}'. "
            f"Highlight common pitfalls of these methods."
        )
        return self._execute_crow_task(formatted_query, task_type="Hypothesis")

    def search_for_cross_domain(self, objective: str) -> Dict[str, Any]:
        """
        Formats a query for INSPIRATION retrieval: mechanisms and design
        principles from adjacent or unrelated fields that could TRANSFER to
        this problem.

        Deliberately not a topical review. ``search_for_hypothesis_context``
        retrieves the problem's own subfield — which grounds a plan but also
        anchors it to established approaches. This one asks for analogies
        whose mechanism might carry over, which is where a genuinely new
        mechanistic idea usually comes from.

        Intended for ideation. Benchmarking showed cross-domain context
        raises idea novelty and non-obviousness, but degrades adherence to
        hard equipment/process constraints — so pair it with grounding
        retrieval, and prefer grounding alone when a plan must satisfy
        stated constraints.
        """
        formatted_query = (
            "Survey mechanisms and design principles from ADJACENT and "
            "UNRELATED domains that could TRANSFER to the problem below. "
            "State the FUNCTION to be transferred toward in one sentence "
            "first — drawing on the problem's own field only for that, never "
            "as the answer — then leave that field behind: the body must NOT "
            "be a topical review of its methods. For each analogous system "
            "(a different chemistry, biology, or engineering field achieving "
            "a similar function), name the field and describe the underlying "
            "mechanism, how it is measured or realized, why it might "
            "transfer, and what would break in transfer. Emphasize "
            "unconventional and emerging approaches.\n\n"
            f"PROBLEM: {objective}"
        )
        return self._execute_crow_task(formatted_query, task_type="CrossDomain")

    def search_for_fitting_models(self, objective: str) -> Dict[str, Any]:
        """
        Formats a query specifically for finding mathematical equations.
        (Retains the original functionality).
        """
        formatted_query = (
            f"Identify specific physical or mathematical models used to analyze "
            f"data related to: '{objective}'. Provide specific equations or "
            f"fitting parameters where possible."
        )
        return self._execute_crow_task(formatted_query, task_type="Modeling")

    def search_for_economic_data(self, objective: str) -> Dict[str, Any]:
        """
        Formats a query specifically for Technoeconomic Analysis (TEA).
        """
        formatted_query = (
            f"Find current market prices, cost drivers, and economic feasibility studies "
            f"associated with: '{objective}'. Focus on CAPEX, OPEX, and raw material costs."
        )
        return self._execute_crow_task(formatted_query, task_type="TEA")