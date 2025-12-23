"""
SciLink MCP Server - FIXED VERSION (No Timeout)

Model Context Protocol server exposing SciLink Planning Agents.
Agents are initialized LAZILY to prevent timeout on list_tools().
"""

import json
import logging
import sys
from typing import Any
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
)

# Configure logging - MCP uses stderr for logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize MCP server
app = Server("scilink-planning")

# Lazy-loaded agents (initialized on first use)
_planning_agent = None
_bo_agent = None


def get_planning_agent():
    """Get or create planning agent (lazy initialization)."""
    global _planning_agent
    if _planning_agent is None:
        logger.info("Initializing planning agent...")
        from .planning_tools import get_planning_agent as _get_agent
        _planning_agent = _get_agent()
        logger.info("Planning agent ready")
    return _planning_agent


def get_bo_agent():
    """Get or create BO agent (lazy initialization)."""
    global _bo_agent
    if _bo_agent is None:
        logger.info("Initializing BO agent...")
        from .planning_tools import get_bo_agent as _get_agent
        _bo_agent = _get_agent()
        logger.info("BO agent ready")
    return _bo_agent


# ============================================================================
# TOOL DEFINITIONS - Fast, no initialization
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available planning tools.
    
    IMPORTANT: This must be FAST (<1 second) to avoid timeout.
    Agents are initialized lazily when tools are actually called.
    """
    return [
        Tool(
            name="propose_experiments",
            description=(
                "Generate an experimental plan using RAG over scientific literature "
                "and implementation knowledge. The agent will:\n"
                "• Build/load knowledge bases from documents and code\n"
                "• Query external literature databases (if configured)\n"
                "• Generate testable hypotheses\n"
                "• Map experimental steps to executable code\n"
                "• Save complete plan with state for iteration"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Research goal (specific and measurable)"
                    },
                    "science_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to scientific documents (PDFs, .txt, .md, directories)"
                    },
                    "code_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to code repos, API docs, or Git URLs"
                    },
                    "structured_data_sets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {"type": "string"},
                                "metadata_path": {"type": "string"}
                            }
                        },
                        "description": "Excel/CSV datasets with optional metadata files"
                    },
                    "additional_context": {
                        "type": "object",
                        "description": "Additional context as key-value pairs (constraints, equipment, etc.)"
                    },
                    "primary_data_set": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "metadata_path": {"type": "string"}
                        },
                        "description": "Main dataset to analyze"
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to images for multimodal analysis"
                    },
                    "image_descriptions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Text descriptions for images (same order)"
                    },
                    "output_json_path": {
                        "type": "string",
                        "description": "Path to save plan (also creates .state.json and .html)"
                    },
                    "output_code_dir": {
                        "type": "string",
                        "description": "Directory for generated scripts (default: ./mcp_outputs/scripts)"
                    },
                    "reset_state": {
                        "type": "boolean",
                        "description": "Clear existing state and start fresh",
                        "default": False
                    }
                },
                "required": ["objective"]
            }
        ),
        
        Tool(
            name="update_plan_with_results",
            description=(
                "Iterate on experimental plan based on results. Supports:\n"
                "• Multimodal inputs (text, Excel, CSV, images, logs)\n"
                "• Result-aware literature search\n"
                "• Nuanced reasoning (confirmed/optimization/failure analysis)\n"
                "• Updated code generation with incremental improvements"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": ["string", "object", "array"],
                        "description": (
                            "Experimental results. Supports:\n"
                            "- Text: Natural language description\n"
                            "- Dict: Structured data\n"
                            "- Array: File paths or mixed types\n"
                            "- File: .xlsx, .csv, .txt, .png, .jpg paths"
                        )
                    },
                    "output_json_path": {
                        "type": "string",
                        "description": "Path to save updated plan"
                    },
                    "output_code_dir": {
                        "type": "string",
                        "description": "Directory for scripts (default: ./mcp_outputs/scripts)"
                    },
                    "state_file_path": {
                        "type": "string",
                        "description": "Path to .state.json to restore (optional)"
                    }
                },
                "required": ["results"]
            }
        ),
        
        Tool(
            name="perform_technoeconomic_analysis",
            description=(
                "Perform technoeconomic assessment (TEA) of a technology or process.\n"
                "Analyzes:\n"
                "• Cost drivers and benefits\n"
                "• Economic risks\n"
                "• Comparison to alternatives\n"
                "• Data gaps for quantitative analysis"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "Technology/process to evaluate economically"
                    },
                    "science_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to market data, reports, TEA studies"
                    },
                    "code_paths": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "structured_data_sets": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "primary_data_set": {
                        "type": "object",
                        "description": "Main economic dataset"
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Criticality matrices, cost breakdowns, etc."
                    },
                    "image_descriptions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "output_json_path": {
                        "type": "string",
                        "description": "Path to save TEA results"
                    }
                },
                "required": ["objective"]
            }
        ),
        
        Tool(
            name="restore_state",
            description="Restore agent state from a .state.json file to continue a previous session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "state_file_path": {
                        "type": "string",
                        "description": "Path to .state.json file"
                    }
                },
                "required": ["state_file_path"]
            }
        ),
        
        Tool(
            name="run_bayesian_optimization",
            description=(
                "Run Bayesian Optimization for experimental design.\n"
                "Uses Gaussian Processes with LLM-configured strategies:\n"
                "• Fits statistical models to data\n"
                "• Recommends next experiments\n"
                "• Generates diagnostic plots\n"
                "• Supports single and multi-objective optimization"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data_path": {
                        "type": "string",
                        "description": "Path to experimental data (.xlsx or .csv)"
                    },
                    "objective_text": {
                        "type": "string",
                        "description": "Natural language optimization goal"
                    },
                    "input_cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Input parameter column names"
                    },
                    "input_bounds": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "description": "Bounds for each input: [[min1, max1], ...]"
                    },
                    "target_cols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Objective column names"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory (default: ./bo_artifacts)",
                        "default": "./bo_artifacts"
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Number of experiments to recommend (1-20)",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["data_path", "objective_text", "input_cols", 
                           "input_bounds", "target_cols"]
            }
        ),
        
        Tool(
            name="refine_experimental_plan",
            description=(
                "Refine the scientific experimental plan based on feedback.\n"
                "This is the FIRST critical review stage - after hypothesis generation "
                "but before code implementation.\n\n"
                "Use this when the user wants to:\n"
                "• Modify experimental steps or hypothesis\n"
                "• Adjust equipment or materials\n"
                "• Change experimental parameters\n"
                "• Add safety constraints\n\n"
                "The agent will update the plan while preserving the overall objective."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "feedback": {
                        "type": "string",
                        "description": "User's feedback or requested changes to the experimental plan"
                    },
                    "state_file_path": {
                        "type": "string",
                        "description": "Path to .state.json file containing current plan"
                    },
                    "output_json_path": {
                        "type": "string",
                        "description": "Path to save refined plan"
                    }
                },
                "required": ["feedback", "state_file_path"]
            }
        ),
        
        Tool(
            name="refine_implementation_code",
            description=(
                "Refine the implementation code based on feedback.\n"
                "This is the SECOND critical review stage - after code generation.\n\n"
                "Use this when the user wants to:\n"
                "• Fix bugs or syntax errors\n"
                "• Change API usage patterns\n"
                "• Add error handling or logging\n"
                "• Modify hardware-specific parameters\n"
                "• Adjust code style or comments\n\n"
                "The agent will update ONLY the implementation code, not the experimental strategy."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "feedback": {
                        "type": "string",
                        "description": "User's feedback or requested changes to the implementation code"
                    },
                    "state_file_path": {
                        "type": "string",
                        "description": "Path to .state.json file containing current plan with code"
                    },
                    "output_json_path": {
                        "type": "string",
                        "description": "Path to save plan with refined code"
                    },
                    "output_code_dir": {
                        "type": "string",
                        "description": "Directory to save refined code scripts",
                        "default": "./mcp_outputs/scripts"
                    }
                },
                "required": ["feedback", "state_file_path"]
            }
        ),
    ]


# ============================================================================
# TOOL IMPLEMENTATIONS - Agents initialized lazily here
# ============================================================================

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool execution with comprehensive error handling."""
    
    logger.info(f"Tool called: {name}")
    
    try:
        if name == "propose_experiments":
            agent = get_planning_agent()  # Lazy init here
            result = agent.propose_experiments(**arguments)
            
            logger.info(f"propose_experiments completed: success={result['success']}")
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "update_plan_with_results":
            agent = get_planning_agent()
            result = agent.update_plan_with_results(**arguments)
            
            logger.info(f"update_plan_with_results completed: success={result['success']}")
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "perform_technoeconomic_analysis":
            agent = get_planning_agent()
            result = agent.perform_tea(**arguments)
            
            logger.info(f"perform_technoeconomic_analysis completed: success={result['success']}")
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "restore_state":
            agent = get_planning_agent()
            result = agent.restore_state(arguments["state_file_path"])
            
            logger.info(f"restore_state completed: success={result['success']}")
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "run_bayesian_optimization":
            agent = get_bo_agent()  # Lazy init here
            result = agent.run_optimization(**arguments)
            
            logger.info(f"run_bayesian_optimization completed: success={result['success']}")
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "refine_experimental_plan":
            agent = get_planning_agent()
            result = agent.refine_plan(**arguments)
            
            logger.info(f"refine_experimental_plan completed: success={result['success']}")
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "refine_implementation_code":
            agent = get_planning_agent()
            result = agent.refine_code(**arguments)
            
            logger.info(f"refine_implementation_code completed: success={result['success']}")
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        else:
            error_result = {
                "success": False,
                "error": f"Unknown tool: {name}"
            }
            logger.warning(f"Unknown tool: {name}")
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        error_result = {
            "success": False,
            "error": str(e),
            "tool": name
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


# ============================================================================
# RESOURCE EXPOSURE
# ============================================================================

@app.list_resources()
async def list_resources() -> list[Any]:
    """List available state files and outputs as resources."""
    resources = []
    
    # REMOVED "./" to prevent scanning the entire project root recursively
    search_dirs = ["./outputs", "./mcp_outputs", "./results", "./bo_artifacts"]
    
    for search_dir in search_dirs:
        state_dir = Path(search_dir)
        if not state_dir.exists():
            continue
        
        try:
            # 1. State files
            for state_file in state_dir.glob("*.state.json"): # Removed ** to stay shallow if possible, or keep ** if nested structure is strict
                resources.append({
                    "uri": f"file://{state_file.absolute()}",
                    "name": state_file.name,
                    "mimeType": "application/json",
                    "description": f"Session state: {state_file.stem}"
                })
            
            # 2. Result JSON files (Excluding state files)
            for result_file in state_dir.glob("*.json"):
                if not result_file.name.endswith(".state.json"):
                    resources.append({
                        "uri": f"file://{result_file.absolute()}",
                        "name": result_file.name,
                        "mimeType": "application/json",
                        "description": f"Results: {result_file.stem}"
                    })
            
            # 3. HTML reports
            for html_file in state_dir.glob("*.html"):
                resources.append({
                    "uri": f"file://{html_file.absolute()}",
                    "name": html_file.name,
                    "mimeType": "text/html",
                    "description": f"Report: {html_file.stem}"
                })
        except Exception as e:
            logger.error(f"Error scanning directory {search_dir}: {e}")
            continue
    
    logger.debug(f"Listed {len(resources)} resources")
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read resource by URI."""
    try:
        if uri.startswith("file://"):
            path = Path(uri[7:])
            if path.exists():
                content = path.read_text(encoding='utf-8')
                logger.debug(f"Resource read: {path.name}")
                return content
            else:
                raise FileNotFoundError(f"Resource not found: {path}")
        else:
            raise ValueError(f"Unsupported URI scheme: {uri}")
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Run the MCP server using stdio transport."""
    logger.info("Starting SciLink Planning MCP server")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server running")
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


def run():
    """Entry point for command-line execution."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()