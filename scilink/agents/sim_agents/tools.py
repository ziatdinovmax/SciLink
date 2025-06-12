import google.generativeai as genai

import os
import logging


# Define constants for tool names
ASE_EXECUTE_TOOL_NAME = "execute_ase_script"
GB_EXECUTE_TOOL_NAME = "execute_gb_script"


def define_ase_tool():
    """Defines the Tool structure for the ASE script execution function."""
    return genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name=ASE_EXECUTE_TOOL_NAME,
                description=(
                    "Executes a Python script using the Atomic Simulation Environment (ASE) "
                    "library to generate or modify an atomic structure based on the user request. "
                    "The script must save the final structure (e.g., as POSCAR, CIF, XYZ) "
                    "and then print a confirmation line: 'STRUCTURE_SAVED:<filename.ext>'."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        'script_content': genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description=(
                                "The complete Python script content as a string. Ensure it "
                                "includes necessary imports (like ase, ase.io, ase.build), "
                                "defines the structure, performs any requested modifications, "
                                "saves the structure using ase.io.write(), and prints "
                                "'STRUCTURE_SAVED:<filename>' on success."
                            )
                        )
                    },
                    required=['script_content']
                )
            )
        ]
    )


def define_gb_tool():
    """Defines the grain boundary tool (requires aimsgb documentation)."""
    return genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name=GB_EXECUTE_TOOL_NAME,
                description=(
                    "Executes a Python script using the aimsgb library to generate grain boundary "
                    "structures based on Coincident Site Lattice (CSL) theory. Use this specifically "
                    "for creating bicrystal interfaces, twist/tilt grain boundaries, and related "
                    "crystallographic interfaces. The script must save the structure and print "
                    "confirmation: 'STRUCTURE_SAVED:<filename.ext>'."
                ),
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        'script_content': genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description=(
                                "Complete Python script using aimsgb library. Must include "
                                "imports (from aimsgb import Grain, GrainBoundary), create GB "
                                "structure, save with .to() method, and print 'STRUCTURE_SAVED:<filename>'."
                            )
                        )
                    },
                    required=['script_content']
                )
            )
        ]
    )


class ToolWithDocs:
    """Container for a tool and its optional documentation."""
    
    def __init__(self, name: str, tool_func, docs_path: str = None, keywords: list = None):
        self.name = name
        self.tool = tool_func()
        self.docs_path = docs_path
        self.keywords = keywords or []
        self.docs_content = self._load_docs() if docs_path else None
        
        logger = logging.getLogger(__name__)
        if self.docs_content:
            logger.info(f"Loaded documentation for {name} tool ({len(self.docs_content)} chars)")
        else:
            logger.info(f"Initialized {name} tool (no additional documentation)")
    
    def _load_docs(self) -> str:
        """Load documentation from file if it exists."""
        if not self.docs_path or not os.path.exists(self.docs_path):
            logging.warning(f"Documentation file not found: {self.docs_path}")
            return None
        
        try:
            with open(self.docs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Optionally truncate if too long
            max_length = 15000
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[... Documentation truncated ...]"
                logging.info(f"Truncated {self.name} docs from {len(content)} to {max_length} chars")
            
            return content
        except Exception as e:
            logging.error(f"Failed to load docs for {self.name} from {self.docs_path}: {e}")
            return None
    
    def matches_request(self, request_text: str) -> bool:
        """Check if this tool matches the request based on keywords."""
        if not self.keywords:
            return False
        
        request_lower = request_text.lower()
        return any(keyword in request_lower for keyword in self.keywords)