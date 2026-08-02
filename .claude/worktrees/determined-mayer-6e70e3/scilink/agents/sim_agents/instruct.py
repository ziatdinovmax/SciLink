INITIAL_PROMPT_TEMPLATE = """
You are an expert in computational modeling of materials. The user requested to build: "{description}"

First, think step-by-step about how you would create this structure:
1. What material(s) and crystal structure(s) are needed? (For multi-material systems: identify each component)
2. What supercell sizes, layer thicknesses, or geometric parameters are required?
3. What defects, substitutions, or other modifications need to be applied?
4. What atomic constraints are physically appropriate for this model? (e.g., fixing the bottom layers of a slab to mimic bulk).
5. How should the materials be combined? (For interfaces/heterostructures, do NOT invent supercell sizes by hand — that frequently produces unphysical strain. Use `pymatgen.analysis.interfaces.zsl.ZSLGenerator` (or `CoherentInterfaceBuilder`) to enumerate lattice-matched supercell combinations and pick one with low strain (typically <5%). If the user pinned specific supercell sizes that don't lattice-match, prefer the user's explicit request only if the resulting strain is reasonable; otherwise raise an error explaining the mismatch rather than producing a 50%+-strain structure.)

For **2D materials** (monolayers / bilayers / few-layer structures), prefer manual construction over fetching the 3D parent from Materials Project and extracting a layer. Build the basis explicitly in a hexagonal (or appropriate) cell with known lattice parameters and add vacuum along the perpendicular axis. The bulk → monolayer extraction path is error-prone: γ-convention mismatches between the parent's stored cell and the extracted layer, and confusion between inter-layer-stacking basis vs in-layer basis, produce structures with unphysical short interatomic distances (a tell-tale sign is a min pairwise distance well below the expected nearest-neighbor bond length, often a constant fraction of the lattice parameter).
6. If the user requests an **amorphous** material as a substrate or component, do NOT fake it by adding small random displacements to a crystalline polymorph — that produces a disordered crystal, not a true amorphous network with proper coordination chemistry. A real amorphous structure requires melt-quench molecular dynamics (out of scope for a one-shot script). Either (a) raise a `NotImplementedError` explaining that a pre-equilibrated amorphous structure file should be supplied by the user, or (b) build with a crystalline polymorph and **explicitly disclose in a printed message** that the substrate is crystalline, not amorphous as requested. Do not silently produce a "pseudo-amorphous" structure.
7. **Stacking convention** for substrate-on-overlayer geometries: the named *substrate* (the material the user says the overlayer is "on") must occupy the lower z-range of the cell, and the *overlayer* must occupy the higher z-range above it. Vacuum goes above the overlayer. Surface passivation (e.g., H termination) goes on the substrate's bottom surface (its z-min side, away from the overlayer). After assembling the structure, verify this with explicit min/max z prints per species before saving — getting this inverted is a common failure.
8. In what order should these steps be performed?

Then, generate a *complete* and *executable* Python script implementing your approach. Use ASE as the primary structure-manipulation and I/O library; bring in other Python libraries as needed (e.g., `pymatgen` for Materials Project structure fetching and interface lattice matching, `aimsgb` for grain boundaries):
1. The script MUST include all necessary imports (e.g., `from ase import Atoms`, `from ase.build import ...`, `from ase.io import write`, plus any non-ASE libraries the approach requires).
2. The script MUST define or load the base structure correctly (e.g., via ASE's `bulk` / `surface` / `molecule`, by reading a file, or by fetching from Materials Project with `MPRester`).
3. The script MUST perform any requested modifications (e.g., creating vacancies, substituting atoms, adding adsorbates, applying strain). Use standard ASE functionality when available; fall back to pymatgen or other libraries when ASE doesn't cover the operation cleanly.
4. The script MUST save the final `Atoms` object to a file (e.g., 'structure.xyz', 'POSCAR', 'structure.cif'). Choose a suitable, simple filename.
5. CRITICALLY: Immediately after successfully saving the file, the script MUST print *exactly* this confirmation line to standard output: `STRUCTURE_SAVED:<filename.ext>` (replace `<filename.ext>` with the actual filename used). No other output should precede or follow this specific line unless it's part of error handling.
6. Ensure the script handles potential issues gracefully if possible (e.g., checks for valid indices if modifying atoms).
7. Call the '{tool_name}' function/tool with the *entire generated Python script content* as the 'script_content' argument. Do not add any explanatory text before or after the function call itself in your response.
"""

MODIFICATION_PROMPT_TEMPLATE = """
The user previously built an atomic structure with this script:

```python
{prior_script}
```

The user is now asking to modify that structure. Their description of the
desired change:

"{description}"

Your task:
1. Read the prior script carefully and understand what it built.
2. Apply the requested change as a **minimal delta** to the prior script —
   preserve lattice parameters, supercell size, vacuum, naming conventions,
   and helper logic that don't need to change. The goal is the smallest
   correct edit, not a rewrite. If you find yourself rewriting more than
   half the script, stop and reconsider whether the prior script's setup is
   actually being kept.
3. The modified script MUST still save the final `Atoms` (or pymatgen
   `Structure`) object to a file and print *exactly* `STRUCTURE_SAVED:<filename.ext>`
   on success — same contract as the original.
4. Use ASE as the primary structure library; pull in pymatgen, aimsgb, etc.
   only as already used in the prior script (don't introduce new
   dependencies for a simple delta).
5. Call the '{tool_name}' function/tool with the *entire modified Python script
   content* as the 'script_content' argument. No explanatory text around the
   function call.
"""


CORRECTION_PROMPT_TEMPLATE = """
The user's original request was: "{original_request}"

You previously generated the following Python script:
```python
{failed_script}
```
However, executing this script failed with the following error (traceback included):
{error_message}
Your task is to:

Analyze the error message (especially the traceback) and the failed script provided above.
Identify the specific bug or issue in the script that caused the error. Common issues include incorrect imports, wrong function arguments, index errors, undefined variables, or logical errors in structure manipulation.
Generate a corrected, complete, and executable Python script that fulfills the original request ("{original_request}") and specifically avoids the previous error. Use ASE as the primary structure library; bring in pymatgen or other Python libraries when the approach requires them.
The corrected script MUST still include all necessary imports, structure definition/modification, saving the file with `ase.io.write()` (or equivalent), and printing the exact confirmation line 'STRUCTURE_SAVED:<filename.ext>' upon successful saving. Use a simple filename.
Call the '{tool_name}' function/tool again, providing the entire corrected Python script content as the 'script_content' argument. Do not add explanatory text around the function call itself.
"""


VALIDATOR_PROMPT_TEMPLATE = """You are an expert materials scientist and computational modeling specialist.
Your task is to critically review an unrelaxed atomic structure generated by a Python script. This structure is intended as an initial input for DFT relaxation. Therefore, the purpose is to create a reasonable starting geometry, not a perfect relaxed structure.

**Trust the STRUCTURE STATS section over the images.** The numerical stats are computed directly from the POSCAR and are decisive; the PNG renders are bond-free and easy to misread. When a stat contradicts what the image looks like, trust the stat. Use images as a confirmatory aid, not the primary source of truth.

**What's NORMAL and should NOT be flagged as issues in an unrelaxed structure:**
- Atomic clashes and close contacts (<1.0 Å) at grain boundaries, interfaces, or surface terminations — these resolve during DFT relaxation.
- Absence of explicit defect bond reconstruction. DFT relaxation introduces such relaxations from a sensible unrelaxed starting geometry; the script doesn't need to pre-apply them.
- Atoms placed on ideal lattice sites near defects rather than displaced toward final relaxed positions.
- Vacuum thicknesses anywhere from 12 Å upward (15 Å is a common but not strict requirement).
- Minor coordinate-wrap artifacts that don't change the periodic image of the structure.

Flag ONLY substantive issues: wrong composition, wrong supercell size relative to the request, missing requested defects, fundamentally wrong bonding indicative of a script bug, severely insufficient vacuum (<10 Å), gross stoichiometry errors. When the structure is a reasonable starting point for DFT, prefer reporting fewer issues over more — refinement cycles are expensive.

**Crucially: do NOT report issues you walk back.** If your reasoning about a candidate issue concludes it's acceptable, normal, or definitionally ambiguous, omit the entry — the issues list is for confident, decisive complaints. Uncertainty about whether something counts as a problem means it doesn't.

{tool_documentation}

**Input Provided to You:**
1.  **Original User Request for Structure:** A textual description of the desired atomic structure.
    Example: "{original_request}"
2.  **Generating Script Content:** The Python script used to create the structure.
    Example:
    ```python
    {generating_script_content}
    ```
3.  **Structure File Content:** The raw content of the structure file generated with this exact script.
4.  **Structure Images (Visual Aid):** Images of the generated structure viewed along the X, Y, and Z axes. These are provided as a supplementary visual reference.

**Your Task & Output Format:**

Based on a holistic analysis of ALL provided information (request, script, file content, and images), you MUST output a valid JSON object with the following keys:

1.  `"overall_assessment"`: (String) A brief (2-3 sentences) overall assessment of the structure's suitability for DFT, its adherence to the original request, and its physical/chemical soundness. Your analysis should be centered on the **script logic and the structure file content**, using the images as a helpful visual reference.
2.  `"identified_issues_detail"`: (List of Strings) A list of ALL specific issues you identified. Analyze the script and structure file for:
    * Discrepancies from the "Original User Request" (e.g., wrong composition, incorrect lattice, missing defects, wrong surface termination).
    * Gross physical or chemical unreasonableness (e.g., severe atomic clashes that relaxation might not fix, fundamentally wrong bonding indicative of incorrect script logic).
    * Stoichiometry errors.
    * For slabs/surfaces: insufficient vacuum, incorrect layer stacking.
    * Any other obvious issues visible in the file content or images that would cause DFT problems.
    If no critical issues are found, this should be an empty list.
3.  `"script_modification_hints"`: (List of Strings) Actionable suggestions on how the *provided script* could be modified to address the identified issues. Base these suggestions on your analysis of both the script and the resulting structure. If specialized library documentation is provided, use that library's specific syntax. If the structure is a good starting point, provide an empty list.

Ensure your output is ONLY the valid JSON object described above. Do not include any other text, explanations, or markdown formatting outside the JSON structure.
"""


# NOTE: DOCS_ENHANCED_INITIAL_PROMPT_TEMPLATE and
# DOCS_ENHANCED_CORRECTION_PROMPT_TEMPLATE used to live here. They were the
# specialized-library counterparts to the regular templates above, picked
# via keyword-routed TOOL_CONFIGS in StructureGenerator. That routing has
# been replaced by an explicit `skill` parameter on the simulate-orchestrator's
# tools (see scilink/skills/structure_generation/aimsgb.md and
# StructureGenerator._format_skill_block). The skill content is now appended
# as a section to the regular templates rather than swapping the whole prompt.


SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE = """
The user's original request for an atomic structure was: "{original_request}"

A previous attempt to generate an ASE script for this request was made.
- **Previously Attempted Script:**
  ```python
  {attempted_script_content}
  ```
- **Validation Feedback on the Structure Produced by the Above Script:**
  - **Overall Assessment by Validator:** {validator_overall_assessment}
  - **Specific Issues Identified by Validator:**
    {validator_specific_issues}
  - **Validator's Hints for Modifying the Script:**
    {validator_script_hints}
{prior_attempts_summary}
Your task is to generate a **new, corrected, complete, and executable Python script** that:
1.  Precisely fulfills the original user request: "{original_request}".
2.  Directly addresses **substantive** "Specific Issues Identified by Validator" — but recognize when an "issue" is actually cosmetic or a non-problem (see rule 8 below).
3.  Intelligently incorporates the "Validator's Hints for Modifying the Script". If hints conflict or are unclear, prioritize fulfilling the original request and fixing real issues.
4.  The script MUST include all necessary imports. Use ASE as the primary structure library (e.g., `from ase import Atoms`, `from ase.build import ...`, `from ase.io import write`); bring in pymatgen or other Python libraries when the approach requires them (e.g., `MPRester` for MP fetching, `pymatgen.analysis.interfaces` for lattice matching, `aimsgb` for grain boundaries).
5.  The script MUST save the final structure to a file (e.g., 'structure.xyz', 'POSCAR'). Choose a suitable, simple filename.
6.  CRITICALLY: Immediately after successfully saving the file, the script MUST print *exactly* this confirmation line to standard output: `STRUCTURE_SAVED:<filename.ext>` (replace `<filename.ext>` with the actual filename used).
7.  Call the '{tool_name}' function/tool with the *entire new corrected Python script content* as the 'script_content' argument. Do not add any explanatory text before or after the function call itself in your response.
8.  **STOP CHASING COSMETIC ISSUES.** If the prior-attempts history shows the same kind of complaint repeating (e.g., "negative Cartesian coordinates in hexagonal cell", "atom near edge could be wrapped", "structure could be cleaner"), AND no prior fix has eliminated it, treat it as cosmetic and **return the previous script unchanged** (or with only trivial whitespace changes). Cosmetic remarks do not warrant another refinement cycle. Save the user's compute budget by accepting the structure as-is when validator complaints are recurring or non-substantive.
"""


VASP_INPUT_GENERATION_INSTRUCTIONS = """You are an expert computational materials scientist specializing in VASP (Vienna Ab-initio Simulation Package) calculations.

Your task is to generate appropriate INCAR and KPOINTS files based on:
1. The provided POSCAR structure file content
2. The original user request describing the scientific objective

**GUIDELINES FOR VASP INPUT GENERATION:**

## INCAR File Guidelines:
- **ENCUT**: Set based on POTCAR requirements (typically 400-600 eV, higher for accurate forces/stress)
- **PREC**: Use "Accurate" for production calculations, "Normal" for testing
- **ALGO**: "Normal" for most cases, "VeryFast" for large systems, "All" for difficult convergence
- **ISMEAR**: 
  - -5 (tetrahedron) for static calculations and DOS
  - 0 (Gaussian) for insulators/semiconductors with appropriate SIGMA
  - 1 (Methfessel-Paxton) for metals
- **Relaxation parameters**: IBRION, ISIF, NSW, EDIFFG based on what needs to be relaxed
- **Electronic convergence**: EDIFF (typically 1E-6 for forces, 1E-8 for accurate energies)
- **Special considerations**:
  - Surface/slab: LDIPOL, DIPOL for dipole corrections
  - Magnetic systems: ISPIN=2, MAGMOM
  - Hybrid functionals: HSE06 parameters if needed
  - van der Waals: include DFT‑D3 corrections (`IVDW = 11`) **only if** the POSCAR geometry indicates a slab, surface, or molecular cluster; otherwise omit.

## KPOINTS File Guidelines:
- **Grid density**: Balance accuracy vs computational cost
- **Monkhorst-Pack**: Most common, specify grid and shift
- **Gamma-centered**: For even grids, often more efficient
- **Special cases**:
  - 2D/surface systems: 1 k-point along vacuum direction
  - 1D systems: 1 k-point in confined directions
  - Large supercells: Gamma-point only might be sufficient
- **Convergence**: Ensure k-point density is converged for the property of interest

## Calculation Type Recognition:
Based on the user request, determine the appropriate calculation type:
- **Structure relaxation**: Full optimization of atomic positions and/or cell
- **Static calculation**: Single-point energy at fixed geometry
- **Electronic structure**: Band structure, DOS calculations
- **Optical properties**: Dielectric function, absorption
- **Defect calculations**: Special considerations for charged defects
- **Surface calculations**: Slab models with vacuum and dipole corrections

## Parameter Selection Logic:
1. **Identify the main scientific objective** from the user request
2. **Analyze the structure type** (bulk, surface, 2D, defective, etc.) from the POSCAR
3. **Choose appropriate calculation workflow** (relax → static → analysis)
4. **Set parameters based on required accuracy** vs computational efficiency
5. **Include relevant physical effects** (magnetism, van der Waals, etc.)

## Structure Analysis:
From the POSCAR content provided below, analyze:
- System size and composition
- Dimensionality (bulk, surface/slab, 2D, etc.)
- Presence of vacuum gaps
- Chemical elements (check for magnetic species)
- Cell parameters and symmetry

## INPUT DATA:

**POSCAR Structure File:**
{poscar_content}

**Scientific Objective:**
{original_request}

## Output Requirements:
You MUST provide a JSON response with exactly these keys:
{{
  "incar": "complete INCAR file content",
  "kpoints": "complete KPOINTS file content", 
  "summary": "brief calculation description"
}}

Analyze the provided POSCAR structure and user request, then generate appropriate VASP input files following the guidelines above."""



INCAR_VALIDATION_INSTRUCTIONS = """You are an expert VASP computational materials scientist. 

Your task is to analyze a literature review of VASP INCAR parameters and suggest specific adjustments if needed.

You will receive:
1. The original INCAR file content
2. A literature review assessment of these parameters
3. The system description

Based on the literature review, determine if any parameters should be adjusted and provide specific recommendations.

You MUST respond with a JSON object containing:
{{
  "validation_status": "good" or "needs_adjustment",
  "overall_assessment": "brief summary of the literature findings",
  "suggested_adjustments": [
    {{
      "parameter": "PARAMETER_NAME",
      "current_value": "current value",
      "suggested_value": "suggested value", 
      "reason": "explanation based on literature"
    }}
  ],
  "revised_incar": "complete revised INCAR content if adjustments needed, or null if no changes"
}}

Focus on actionable parameter changes. Only suggest adjustments if the literature clearly indicates issues.
"""

"""
LLM instruction templates for the PackmolGeneratorAgent.
Contains all major prompts used for molecule extraction, SMILES generation, and PACKMOL script creation.
"""

MOLECULE_EXTRACTION_TEMPLATE = """
Analyze this molecular system description and extract ALL molecules mentioned:

"{description}"

For each molecule, provide:
1. The identifier used in the description
2. The most likely chemical formula (if determinable)
3. Alternative names that might work in databases
4. SMILES string (if you know it)
5. Estimated count/concentration from context

Respond as JSON:
{{
  "molecules": [
    {{
      "identifier": "what was mentioned in description",
      "formula": "chemical formula like H2O, C6H6, etc.",
      "alternative_names": ["list", "of", "possible", "database", "names"],
      "smiles": "SMILES string if known",
      "estimated_count": "number or description like '1.0 M' or 'solvent'"
    }}
  ],
  "box_info": {{
    "dimensions": "box dimensions from description",
    "volume_cubic_angstrom": "calculated volume"
  }}
}}

Focus on identifying the specific molecules mentioned in the description without using any external examples.

Response:"""

SMILES_GENERATION_TEMPLATE = """
Provide the SMILES string for this molecule: {molecule_identifier}

IMPORTANT GUIDELINES:
- For ionic compounds, provide the SIMPLEST possible SMILES that RDKit can handle
- Avoid complex charged species if possible
- For metal complexes, try to provide the organic ligand SMILES instead
- Focus on the main organic component that can be built with standard force fields

If you know a working SMILES, respond with just the SMILES string.
If uncertain or if it's a complex ionic compound, respond with: UNKNOWN

Examples of simple molecules:
- "water" -> O
- "benzene" -> c1ccccc1
- "methanol" -> CO
- "acetone" -> CC(=O)C

For complex metal salts, prefer the organic component only.

Molecule: {molecule_identifier}
SMILES:"""

PACKMOL_SCRIPT_GENERATION_TEMPLATE = """
You are an expert in PACKMOL for molecular dynamics simulation setup.

ORIGINAL REQUEST: "{description}"

AVAILABLE MOLECULES (successfully built):
{molecule_list}

YOUR TASK: Generate a complete PACKMOL input script

PACKMOL REFERENCE:
1. Start with global parameters:
   tolerance 2.0
   filetype pdb
   output {{output_filename}}

2. For each molecule type:
   structure components/molecule_name.pdb
     number COUNT
     inside cube 0. 0. 0. 40.
   end structure

CONCENTRATION CALCULATIONS:
- Box volume: 40×40×40 = 64,000 Å³
- 1 M concentration ≈ 38-40 molecules in this volume
- 0.1 M concentration ≈ 4-5 molecules in this volume
- Water solvent: typically 500-2000 molecules
- Cosolvents: 50-500 molecules depending on ratio
- Solutes: 10-100 molecules for reasonable concentrations

BEST PRACTICES:
- Larger molecules need more space (fewer count)
- Ensure realistic density (don't overpack)
- Use whole numbers for molecule counts
- Consider molecular sizes when setting counts
- For electrolytes, use reasonable ion concentrations

OUTPUT FORMAT (JSON only):
{{
  "thought": "Explain your packing strategy and molecule count reasoning",
  "components": {{
    "molecule_name": integer_count
  }},
  "output_filename": "descriptive_filename.pdb",
  "packmol_script": "complete script with proper syntax"
}}

IMPORTANT: Use ONLY the molecules listed above. Do not reference any molecules that failed to build.

Generate the PACKMOL script:"""

LAMMPS_INPUT_GENERATION_TEMPLATE = """
Generate a complete LAMMPS script for molecular dynamics simulation to achieve the following research goal:

RESEARCH GOAL: "{research_goal}"

SYSTEM DESCRIPTION: {system_description}

SYSTEM COMPOSITION:
  - {element_info_str}
  - Total atoms: {atom_count}
  - Box dimensions: {box_dimensions}
  - Bond types: {bond_types}
  - Angle types: {angle_types}

EXACT TYPE INFORMATION FROM DATA FILE:
{data_type_info}

CRITICAL: You MUST use the EXACT type numbers listed above for ALL commands including:
  - group definitions (e.g., if Zn is type 6, use "group zinc type 6")
  - fix shake (use the correct bond type number for water O-H bonds and angle type for H-O-H)
  - compute rdf (use correct atom type pairs from the list above)
  - compute coord/atom (use correct atom type numbers)
  - DO NOT guess or assume type numbers. Use ONLY what is listed above.
  - DO NOT include pair_coeff, bond_coeff, angle_coeff, or dihedral_coeff commands.
    Instead use: include ff_params.lammps
  - DO NOT include pair_style, bond_style, angle_style, dihedral_style, kspace_style,
    or special_bonds commands. These are defined in ff_params.lammps.

DETECTED COMPONENTS:
  - Water: {has_water}
  - Ions: {has_ions}
  - Organic molecules: {has_organic}

SIMULATION PARAMETERS:
  - Properties to calculate: {properties_to_calculate_str}
  - Temperature: {temperature} K
  - Pressure: {pressure} atm
  - Ensemble: {ensemble}
  - Timestep: {timestep} fs
  - Total simulation time: {simulation_time} ns
  - Equilibration steps: {equil_steps}
  - Production steps: {prod_steps}
  - Required outputs: {required_outputs_str}

REQUIRED SECTIONS:
1. Initialization (units, atom_style, etc.)
2. System setup (read data file "{data_filename}")
3. Force field settings (complete with all coefficients)
4. Energy minimization
5. Equilibration phase(s) 
6. Production phase with appropriate outputs
7. Analysis commands for the specified properties

SPECIAL OUTPUT REQUIREMENTS:
{output_commands}

IMPORTANT: Include regular restart file writing capabilities in the script using these guidelines:
1. Write restart files periodically (every 10,000-50,000 steps) during both equilibration and production
2. Use timestep-based naming like "restart.*.equil" for equilibration and "restart.*.prod" for production
3. Include a commented-out 'read_restart' command that can be uncommented if needed to restart the simulation
4. Ensure all necessary variables and settings are properly initialized even when reading from a restart file

Include thorough comments explaining each section and its purpose. The script should be directly executable in LAMMPS.

IMPORTANT: Return ONLY the raw LAMMPS script content without any markdown formatting, code block markers, or backticks.
"""
