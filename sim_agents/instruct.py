# sim_agents/prompt_templates.py

# Using f-string notation means these need .format() called on them later

INITIAL_PROMPT_TEMPLATE = """
User request: "{description}"

Your task is to:
1. Parse the user request to understand the desired atomic structure (material, phase, defects, surface, etc.).
2. Generate a *complete* and *executable* Python script using the Atomic Simulation Environment (ASE) library to create this structure.
3. The script MUST include necessary imports (e.g., `from ase import Atoms`, `from ase.build import ...`, `from ase.io import write`).
4. The script MUST define or load the base structure correctly (e.g., using `bulk`, `surface`, `molecule`, or reading a file if appropriate).
5. The script MUST perform any requested modifications (e.g., creating vacancies, substituting atoms, adding adsorbates, applying strain). Use standard ASE functionalities.
6. The script MUST save the final `Atoms` object to a file (e.g., 'structure.xyz', 'POSCAR', 'structure.cif'). Choose a suitable, simple filename.
7. CRITICALLY: Immediately after successfully saving the file, the script MUST print *exactly* this confirmation line to standard output: `STRUCTURE_SAVED:<filename.ext>` (replace `<filename.ext>` with the actual filename used). No other output should precede or follow this specific line unless it's part of error handling.
8. Ensure the script handles potential issues gracefully if possible (e.g., checks for valid indices if modifying atoms).
9. Call the '{tool_name}' function/tool with the *entire generated Python script content* as the 'script_content' argument. Do not add any explanatory text before or after the function call itself in your response.
"""

CORRECTION_PROMPT_TEMPLATE = """
The user's original request was: "{original_request}"

You previously generated the following Python ASE script:
```python
{failed_script}
However, executing this script failed with the following error (traceback included):
{error_message}
Your task is to:

Analyze the error message (especially the traceback) and the failed script provided above.
Identify the specific bug or issue in the script that caused the error. Common issues include incorrect imports, wrong function arguments, index errors, undefined variables, or logical errors in structure manipulation.
Generate a corrected, complete, and executable Python ASE script that fulfills the original request ("{original_request}") and specifically avoids the previous error.
The corrected script MUST still include all necessary imports, structure definition/modification, saving the file with ase.io.write(), and printing the exact confirmation line 'STRUCTURE_SAVED:<filename.ext>' upon successful saving. Use a simple filename.
Call the '{tool_name}' function/tool again, providing the entire corrected Python script content as the 'script_content' argument. Do not add explanatory text around the function call itself. 
"""