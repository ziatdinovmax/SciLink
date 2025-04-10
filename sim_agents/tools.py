import google.generativeai as genai

# Define constants for tool names if used elsewhere
ASE_EXECUTE_TOOL_NAME = "execute_ase_script"

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