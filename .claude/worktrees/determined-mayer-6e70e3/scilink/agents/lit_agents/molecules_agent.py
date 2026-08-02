"""
Molecules Agent Module

Provides a cheminformatics agent for synthesis planning and molecule design
using the FutureHouse/Edison MOLECULES tool.
"""

import os
import logging
import time
from time import sleep
from typing import Dict, Any, Optional


try:
    from edison_client import EdisonClient, JobNames
except ImportError:
    logging.error("Error: edison_client is not installed. Please install it with 'pip install edison-client'")
    raise


class MoleculesAgent:
    """
    Agent for synthesis planning and molecule design using the Edison MOLECULES tool.

    Uses cheminformatics tools to plan synthesis routes and design new molecules.
    Only triggered when the research objective clearly involves molecular design or discovery.
    """

    def __init__(self, api_key: str | None = None, max_wait_time: int = 600):
        """
        Initialize the Molecules agent.

        Args:
            api_key: FutureHouse (Edison) API key.
            max_wait_time: Maximum time to wait for response in seconds.
        """
        if api_key is None:
            api_key = os.environ.get("FUTUREHOUSE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and FUTUREHOUSE_API_KEY environment variable is not set.")

        self.client = EdisonClient(api_key=api_key)
        self.max_wait_time = max_wait_time
        self.logger = logging.getLogger(__name__)
        self.logger.info("MoleculesAgent initialized with max wait time of %d seconds.", max_wait_time)

    def query(self, prompt: str) -> Dict[str, Any]:
        """
        Submit a cheminformatics query for synthesis planning or molecule design.

        Args:
            prompt: A research objective involving molecule design, synthesis
                planning, or molecular property optimization.

        Returns:
            Dictionary with status, content, and query fields.
        """
        if not prompt or not isinstance(prompt, str):
            error_msg = "Invalid prompt. Must provide a non-empty string."
            self.logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        try:
            self.logger.info(f"Submitting MOLECULES query: {prompt}")

            task_data = {
                "name": JobNames.MOLECULES,
                "query": prompt,
            }

            task_id = self.client.create_task(task_data)
            self.logger.info(f"MOLECULES task created with ID: {task_id}")

            start_time = time.time()

            while time.time() - start_time < self.max_wait_time:
                task_status = self.client.get_task(task_id)
                status = task_status.status.lower() if isinstance(task_status.status, str) else task_status.status

                if status == "success":
                    elapsed = time.time() - start_time
                    self.logger.info(f"MOLECULES query completed after {elapsed:.1f} seconds.")
                    answer = getattr(task_status, 'formatted_answer', None) or getattr(task_status, 'answer', '')
                    return {
                        "status": "success",
                        "content": answer,
                        "query": prompt,
                    }

                if status in ["failed", "error"]:
                    error_msg = f"MOLECULES query failed with status: {status}"
                    self.logger.error(error_msg)
                    return {"status": "error", "message": error_msg, "task_id": task_id}

                sleep(10)

            error_msg = f"MOLECULES query timed out after {self.max_wait_time} seconds."
            self.logger.error(error_msg)
            return {"status": "timeout", "message": error_msg, "task_id": task_id}

        except Exception as e:
            error_msg = f"An unexpected error occurred during MOLECULES query: {str(e)}"
            self.logger.exception(error_msg)
            return {"status": "error", "message": error_msg}
