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

# Tool configurations - moved from config.py
TOOL_CONFIGS = {
    "GrainBoundary": {
        "docs_path": "docs/aimsgb.txt",  # Relative to sim_agents directory
        "keywords": ["grain boundary", "grain-boundary", "gb ", "sigma", "csl", 
                    "twist", "tilt", "bicrystal", "rotation axis", "aimsgb"],
        "tool_func": "define_gb_tool"
    }
    # Future tools can be added here
}

# Tool function mapping
TOOL_FUNCTION_MAP = {
    "define_ase_tool": define_ase_tool,
    "define_gb_tool": define_gb_tool,
    # New tool functions will be added here
}

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
        if not self.docs_path:
            return None
            
        # Try different path resolutions
        possible_paths = [
            self.docs_path,  # As provided
            os.path.join(os.path.dirname(__file__), self.docs_path),  # Relative to this file
            os.path.join(os.path.dirname(__file__), "../..", self.docs_path),  # From package root
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Optionally truncate if too long
                    max_length = 60000
                    if len(content) > max_length:
                        content = content[:max_length] + "\n\n[... Documentation truncated ...]"
                        logging.info(f"Truncated {self.name} docs from {len(content)} to {max_length} chars")
                    
                    logging.info(f"Loaded docs for {self.name} from: {path}")
                    return content
                except Exception as e:
                    logging.error(f"Failed to read docs for {self.name} from {path}: {e}")
                    continue
        
        logging.warning(f"Documentation file not found for {self.name}: {self.docs_path}")
        return None
    
    def matches_request(self, request_text: str) -> bool:
        """Check if this tool matches the request based on keywords."""
        if not self.keywords:
            return False
        
        request_lower = request_text.lower()
        return any(keyword in request_lower for keyword in self.keywords)

def get_available_tools():
    """Get all available tools with their configurations."""
    tools = []
    
    # Always include ASE as the default fallback tool
    tools.append(ToolWithDocs(
        name="ASE",
        tool_func=define_ase_tool,
        docs_path=None,
        keywords=[]  # Empty keywords means it's the fallback
    ))
    
    # Add configured tools
    for tool_name, config_dict in TOOL_CONFIGS.items():
        if tool_name == "ASE":
            continue  # Skip ASE, already added
        
        func_name = config_dict.get("tool_func")
        tool_func = TOOL_FUNCTION_MAP.get(func_name)
        
        if tool_func is None:
            logging.warning(f"Tool function '{func_name}' not found for tool '{tool_name}'. Skipping.")
            continue
        
        tools.append(ToolWithDocs(
            name=tool_name,
            tool_func=tool_func,
            docs_path=config_dict.get("docs_path"),
            keywords=config_dict.get("keywords", [])
        ))
        
        logging.info(f"Registered tool: {tool_name}")
    
    return tools