import logging
from typing import Optional
import os
import json

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from .instruct import (
    INITIAL_PROMPT_TEMPLATE,
    CORRECTION_PROMPT_TEMPLATE,
    SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE,
    MODIFICATION_PROMPT_TEMPLATE,
)
from .utils import save_generated_script, MaterialsProjectHelper, MP_SEARCH_TOOL_SCHEMA
from ...executors import ScriptExecutor, DEFAULT_TIMEOUT

from ._deprecation import normalize_params


MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS = 5
MAX_MP_RESOLVER_ITERATIONS = 3


JSON_OUTPUT_INSTRUCTION = """

IMPORTANT: You must respond with a JSON object in this exact format:
{
    "script_content": "your complete Python script here as a single string"
}

The script_content must be a valid, complete Python script that can be executed directly.
Escape any special characters properly for JSON (newlines as \\n, quotes as \\", etc.).
"""


class StructureGenerator:
    def __init__(self, api_key: str = None, 
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 executor_timeout: int = DEFAULT_TIMEOUT,
                 generated_script_dir: str = "generated_scripts",
                 mp_api_key: str = None,
                 # Legacy parameters
                 local_model: str = None,
                 google_api_key: str = None):
        """
        Initialize StructureGenerator with Multi-Backend Support.
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.generated_script_dir = generated_script_dir

        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="StructureGenerator"
        )
        
        if base_url:
            # Internal Proxy
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError("API key required for internal proxy.")

            self.logger.info(f"StructureGenerator using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
        else:
            # Public / LiteLLM
            self.logger.info(f"StructureGenerator using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key
            )        
        
        # We rely on the prompt to enforce JSON, 
        # rather than the generation config.
        self.generation_config = None

        # Executor Setup
        self.ase_executor = ScriptExecutor(timeout=executor_timeout, mp_api_key=mp_api_key)

        self.mp_helper = MaterialsProjectHelper(api_key=mp_api_key)

        # Initialization message
        print(f"🔧 Structure Generator Ready ({model_name})")
        if self.mp_helper.enabled:
            print(f"   🗃️  Materials Project: Connected")
        else:
            print(f"   🗃️  Materials Project: Not configured")

    @staticmethod
    def _format_skill_block(skill_content: Optional[str]) -> str:
        """Render a skill content block into a labeled prompt section, or
        empty string if no skill is provided."""
        if not skill_content:
            return ""
        return (
            "\n\n## SPECIALIZED LIBRARY GUIDANCE (skill loaded for this request):\n"
            f"{skill_content}\n"
            "Use the patterns and constraints above when relevant.\n"
        )

    def _build_initial_prompt(self, description: str,
                              skill_content: Optional[str] = None) -> str:
        """Build the initial script-generation prompt.

        skill_content is an optional curated guidance block (e.g. an aimsgb
        skill loaded by the simulate orchestrator) that is appended verbatim
        before the script-gen instructions. The MP-resolver block (if MP is
        configured) goes first as ground truth.
        """
        base_prompt = INITIAL_PROMPT_TEMPLATE.format(
            description=description,
            tool_name="ASE",
        )
        base_prompt = base_prompt + self._format_skill_block(skill_content)

        # Pre-resolve any named materials in the request to mp-ids via tool
        # calls, then inject the resolved facts as ground truth before the
        # script-generation instructions. Empty string when MP is disabled,
        # nothing needed resolving, or the resolver call itself fails.
        resolved_block = self._resolve_materials_via_tools(description)
        if resolved_block:
            base_prompt = resolved_block + base_prompt

        return base_prompt + JSON_OUTPUT_INSTRUCTION

    def _build_modification_prompt(self, prior_script: str, description: str,
                                   skill_content: Optional[str] = None) -> str:
        """Build a prompt that asks the LLM to modify a prior script as a
        minimal delta rather than write a new script from scratch. Used when
        the user request is a variant of an already-generated structure
        (e.g., 'now add a single vacancy to the structure I just built')."""
        base_prompt = MODIFICATION_PROMPT_TEMPLATE.format(
            prior_script=prior_script,
            description=description,
            tool_name="ASE",
        )
        base_prompt = base_prompt + self._format_skill_block(skill_content)
        return base_prompt + JSON_OUTPUT_INSTRUCTION

    def _build_correction_prompt(self, original_request: str, failed_script: str,
                                 error_message: str,
                                 skill_content: Optional[str] = None) -> str:
        """Build correction prompt for the inner script-execution retry loop."""
        max_error_len = 2000
        if len(error_message) > max_error_len:
            error_message = error_message[:max_error_len] + "\n[... Error message truncated ...]"

        base_prompt = CORRECTION_PROMPT_TEMPLATE.format(
            original_request=original_request,
            failed_script=failed_script,
            error_message=error_message,
            tool_name="ASE",
        )
        base_prompt = base_prompt + self._format_skill_block(skill_content)
        return base_prompt + JSON_OUTPUT_INSTRUCTION

    def _build_validation_correction_prompt(self, original_request: str,
                                            attempted_script_content: str,
                                            validator_feedback: dict,
                                            attempt_history: Optional[list] = None,
                                            skill_content: Optional[str] = None) -> str:
        """Build validation correction prompt, optionally with prior-cycle history."""
        validator_issues = validator_feedback.get("all_identified_issues", [])
        validator_hints = validator_feedback.get("script_modification_hints", [])

        issues_str = "\n".join([f"- {issue}" for issue in validator_issues]) if validator_issues else "No specific issues listed."
        hints_str = "\n".join([f"- {hint}" for hint in validator_hints]) if validator_hints else "No specific hints provided."

        # Render prior-attempts block, if any. Only show cycles BEFORE the
        # immediately-previous one (which is already included as the
        # "Previously Attempted Script" + its validator feedback).
        prior_block = ""
        if attempt_history and len(attempt_history) >= 2:
            earlier = attempt_history[:-1]
            lines = [
                "",
                "- **Prior-cycle history (for context — to detect recurring/cosmetic complaints):**",
            ]
            for i, entry in enumerate(earlier, start=1):
                ents = entry.get("issues") or []
                ents_str = "; ".join(str(e)[:140] for e in ents) if ents else "(no issues recorded)"
                lines.append(f"  - Cycle {i} → validator complaints: {ents_str}")
            prior_block = "\n".join(lines) + "\n"

        base_prompt = SCRIPT_CORRECTION_FROM_VALIDATION_TEMPLATE.format(
            original_request=original_request,
            attempted_script_content=attempted_script_content,
            validator_overall_assessment=validator_feedback.get("overall_assessment", "N/A"),
            validator_specific_issues=issues_str,
            validator_script_hints=hints_str,
            prior_attempts_summary=prior_block,
            tool_name="ASE",
        )
        base_prompt = base_prompt + self._format_skill_block(skill_content)

        return base_prompt + JSON_OUTPUT_INSTRUCTION

    def _resolve_materials_via_tools(self, description: str) -> str:
        """
        Pre-script-generation step: ask the LLM to identify any named materials
        in the request and resolve them to Materials Project IDs via tool
        calls. Returns a formatted block to prepend to the script-generation
        prompt, or "" when MP is unavailable, no materials need resolution,
        or the resolution call itself fails.

        Fails closed: any exception is logged as a warning and returns "" so
        the caller falls through to the existing prompt unchanged.
        """
        if not self.mp_helper.enabled:
            return ""

        system_prompt = (
            "You are a materials-science assistant whose only job is to "
            "resolve named materials in a structure-building request to "
            "Materials Project IDs (mp-ids). Use the `search_material_id` "
            "tool for each specific material that appears named in the "
            "request (e.g., 'rutile TiO2', 'NaCl', 'graphene', 'monoclinic "
            "HfO2', 'GaAs'). For multiple materials, call the tool multiple "
            "times — one call per material is fine. For generic requests "
            "like 'a 4-atom cubic cell' or 'a Lennard-Jones solid', do not "
            "call any tool and respond with the single token "
            "NO_LOOKUP_NEEDED.\n\n"
            "After all lookups are done, briefly summarize the resolved "
            "ids in plain text. Do not write any Python code."
        )
        user_prompt = (
            f"Structure-building request:\n{description}\n\n"
            "Resolve any named materials to mp-ids."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tools = [MP_SEARCH_TOOL_SCHEMA]
        resolved: list[dict] = []

        try:
            base_url = getattr(self.model, "base_url", None)
            api_key = getattr(self.model, "api_key", None)
            model_name = getattr(self.model, "model", None) or self.model_name

            if base_url:
                self._run_resolver_loop_openai(
                    messages, tools, resolved,
                    api_key=api_key, base_url=base_url, model_name=model_name,
                )
            else:
                self._run_resolver_loop_litellm(
                    messages, tools, resolved,
                    api_key=api_key, model_name=model_name,
                )
        except Exception as e:
            # Resolution is best-effort; don't fail the workflow over it.
            self.logger.warning(f"MP resolution step failed (non-fatal): {e}")
            return ""

        if not resolved:
            return ""

        return self._format_resolved_block(resolved)

    def _run_resolver_loop_openai(self, messages, tools, resolved,
                                  *, api_key, base_url, model_name):
        """Tool-call loop using the OpenAI Python SDK (internal-proxy path)."""
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)

        for _ in range(MAX_MP_RESOLVER_ITERATIONS):
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = self._dispatch_resolver_tool(tc.function.name, args, resolved)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    def _run_resolver_loop_litellm(self, messages, tools, resolved,
                                   *, api_key, model_name):
        """Tool-call loop using LiteLLM (public-API path)."""
        import litellm

        for _ in range(MAX_MP_RESOLVER_ITERATIONS):
            resp = litellm.completion(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                api_key=api_key,
                timeout=60,
            )
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None)
            if not tcs:
                break
            messages.append({
                "role": "assistant",
                "content": getattr(msg, "content", None),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in tcs
                ],
            })
            for tc in tcs:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = self._dispatch_resolver_tool(tc.function.name, args, resolved)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    def _dispatch_resolver_tool(self, name: str, args: dict, resolved: list) -> str:
        """Execute a resolver tool call. Records the outcome in `resolved` and
        returns a JSON string the LLM can read back."""
        if name != "search_material_id":
            return json.dumps({"error": f"Unknown tool: {name}"})

        query = (args.get("chemical_query") or "").strip()
        search_type = args.get("search_type") or "formula"
        spacegroup_symbol = (args.get("spacegroup_symbol") or "").strip() or None
        crystal_system = (args.get("crystal_system") or "").strip() or None
        if not query:
            return json.dumps({"error": "chemical_query is required"})

        record = self.mp_helper.search_material_record(
            query,
            search_type=search_type,
            spacegroup_symbol=spacegroup_symbol,
            crystal_system=crystal_system,
        )

        # Build a label that captures the polymorph constraint so the resolved
        # block reads as "rutile TiO2 → mp-XXXX" rather than just "TiO2".
        label_parts = []
        if spacegroup_symbol:
            label_parts.append(spacegroup_symbol)
        elif crystal_system:
            label_parts.append(crystal_system)
        label_parts.append(query)
        query_label = " ".join(label_parts)

        if record is None:
            resolved.append({
                "query": query_label,
                "search_type": search_type,
                "mp_id": None,
                "formula": None,
                "e_above_hull": None,
                "spacegroup_symbol": None,
            })
            return json.dumps({
                "found": False,
                "query": query_label,
                "message": (
                    "No Materials Project entry found for this query."
                    + (" Try without the polymorph filter." if (spacegroup_symbol or crystal_system) else "")
                ),
            })

        resolved.append({
            "query": query_label,
            "search_type": search_type,
            "mp_id": record["material_id"],
            "formula": record.get("formula_pretty"),
            "e_above_hull": record.get("energy_above_hull"),
            "spacegroup_symbol": record.get("spacegroup_symbol"),
        })
        return json.dumps({
            "found": True,
            "query": query_label,
            "mp_id": record["material_id"],
            "formula": record.get("formula_pretty"),
            "spacegroup_symbol": record.get("spacegroup_symbol"),
            "e_above_hull": record.get("energy_above_hull"),
        })

    @staticmethod
    def _format_resolved_block(resolved: list) -> str:
        """Format the resolver's findings as a markdown block to prepend to
        the script-generation prompt."""
        lines = ["## RESOLVED MATERIALS (Materials Project lookup results):"]
        for entry in resolved:
            if entry["mp_id"] is None:
                lines.append(
                    f'- Query "{entry["query"]}" ({entry["search_type"]}): '
                    f'no MP entry found — build the structure from scratch '
                    f'using ASE primitives.'
                )
            else:
                ehull = entry.get("e_above_hull")
                ehull_str = (
                    f"{ehull:.4f} eV/atom" if isinstance(ehull, (int, float)) else "N/A"
                )
                sg = entry.get("spacegroup_symbol")
                sg_str = f", space group: {sg}" if sg else ""
                lines.append(
                    f'- Query "{entry["query"]}" → **{entry["mp_id"]}** '
                    f'(formula: {entry["formula"]}{sg_str}, '
                    f'e_above_hull: {ehull_str})'
                )
        lines.append("")
        lines.append(
            "To use a resolved mp-id in the script, fetch the structure via "
            "pymatgen and convert to ASE Atoms. The MP_API_KEY environment "
            "variable is already set in the sandbox:"
        )
        lines.append("```python")
        lines.append("import os")
        lines.append("from mp_api.client import MPRester")
        lines.append("from pymatgen.io.ase import AseAtomsAdaptor")
        lines.append('with MPRester(os.getenv("MP_API_KEY")) as mpr:')
        lines.append('    structure = mpr.get_structure_by_material_id("mp-XXXX")')
        lines.append("atoms = AseAtomsAdaptor.get_atoms(structure)")
        lines.append("```")
        lines.append("")
        return "\n".join(lines) + "\n"

    def _generate_json(self, prompt: str) -> dict:
        """Generate JSON response from LLM."""
        self.logger.info("Sending request to LLM...")
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        
        try:
            # Using the wrapper's generate_content
            response = self.model.generate_content(prompt, generation_config=self.generation_config)
            
            if not response or not response.text:
                raise ValueError("Empty response from LLM")
            
            raw_text = response.text.strip()
            
            # Handle markdown code blocks if present
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                raw_text = "\n".join(json_lines)
            
            # Try to find JSON object if there's extra text
            if not raw_text.startswith("{"):
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                if start != -1 and end > start:
                    raw_text = raw_text[start:end]
            
            return json.loads(raw_text)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw response: {response.text[:500] if response and response.text else 'None'}")
            raise
        except Exception as e:
            self.logger.exception(f"Error during LLM content generation: {e}")
            raise

    def generate_script(self, original_user_request: str, attempt_number_overall: int,
                        is_refinement_from_validation: bool = False,
                        previous_script_content: Optional[str] = None,
                        validator_feedback: Optional[dict] = None,
                        attempt_history: Optional[list] = None,
                        skill_content: Optional[str] = None,
                        prior_script_to_modify: Optional[str] = None) -> dict:
        """Generate, refine, or modify a structure-building script.

        Three mutually-exclusive modes:
          - INITIAL (default): write a new script from scratch.
          - REFINEMENT (`is_refinement_from_validation=True`): re-write to
            address validator feedback. Requires `previous_script_content`
            and `validator_feedback`.
          - MODIFICATION (`prior_script_to_modify` set, refinement flag off):
            apply a minimal delta to a prior script instead of rewriting.
            Used when the user request is a variant of an already-built
            structure (e.g., "now add a vacancy to the structure I just
            built").

        Args:
            attempt_history: Optional list of {script, issues, hints} dicts from
                all prior cycles in this refinement loop. Lets the generator
                detect recurring/cosmetic complaints and stop chasing them.
            skill_content: Optional curated library-guidance block (loaded
                from a built-in skill, e.g. "aimsgb" for grain boundaries) to
                inject into the script-generation prompt as a "specialized
                library guidance" section. When None, the prompt is generic.
            prior_script_to_modify: Optional prior-cycle script content. When
                set (and refinement mode off), the LLM is asked to apply
                `original_user_request` as a minimal delta to this script
                rather than write from scratch.
        """
        if is_refinement_from_validation:
            print(f"   🔄 Refining script (cycle {attempt_number_overall})")
            if not previous_script_content or not validator_feedback:
                return {"status": "error", "message": "Internal error: Refinement requires previous script and validation feedback."}
            current_prompt = self._build_validation_correction_prompt(
                original_request=original_user_request,
                attempted_script_content=previous_script_content,
                validator_feedback=validator_feedback,
                attempt_history=attempt_history,
                skill_content=skill_content,
            )
        elif prior_script_to_modify:
            print(f"   📝 Modifying prior script (cycle {attempt_number_overall})")
            current_prompt = self._build_modification_prompt(
                prior_script=prior_script_to_modify,
                description=original_user_request,
                skill_content=skill_content,
            )
        else:
            print(f"   🤖 Generating script (cycle {attempt_number_overall})")
            current_prompt = self._build_initial_prompt(
                original_user_request, skill_content=skill_content,
            )

        last_error_message = "No attempts made yet."
        current_script = previous_script_content if is_refinement_from_validation else None
        final_script_path = None
        
        # Internal correction loop for script execution errors
        for attempt in range(1, MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS + 1):
            try:
                if attempt > 1:
                    print(f"      🔧 Fixing script issues (attempt {attempt}/{MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS})")
                
                # Generate script via JSON response
                response_data = self._generate_json(current_prompt)
                script_content = response_data.get("script_content")
                
                if not script_content:
                    last_error_message = f"LLM response missing 'script_content' (attempt {attempt})"
                    self.logger.error(last_error_message)
                    if attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                        break
                    continue

                current_script = script_content

                # Save script
                script_desc = f"{original_user_request[:30]}_cycle{attempt_number_overall}_attempt{attempt}"
                final_script_path = save_generated_script(
                    current_script, script_desc, 1, output_dir=self.generated_script_dir
                )
                
                if not final_script_path:
                    last_error_message = f"Failed to save script (attempt {attempt})"
                    self.logger.error(last_error_message)
                    break

                # Execute script
                if attempt == 1:
                    print(f"      ⚙️  Executing script...")
                else:
                    print(f"      ⚙️  Re-executing corrected script...")
                    
                exec_result = self.ase_executor.execute_script(
                    current_script, working_dir=self.generated_script_dir
                )

                if exec_result["status"] == "success":
                    output_file = None
                    for line in exec_result.get("stdout", "").splitlines():
                        if line.startswith("STRUCTURE_SAVED:"):
                            output_file = line.split(":", 1)[1].strip()
                            break

                    if output_file and os.path.exists(os.path.join(self.generated_script_dir, output_file)):
                        print(f"      ✅ Script executed successfully")
                        full_output_path = os.path.abspath(os.path.join(self.generated_script_dir, output_file))
                        return {
                            "status": "success",
                            "message": f"Script generated and executed successfully on attempt {attempt}",
                            "output_file": full_output_path,
                            "final_script_path": final_script_path,
                            "final_script_content": current_script,
                            "execution_attempts": attempt
                        }
                    else:
                        last_error_message = f"Script ran but output file not found (attempt {attempt})"
                else:
                    last_error_message = exec_result.get("message", f"Unknown execution error (attempt {attempt})")
                
                if attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                    print(f"      ❌ Script execution failed after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts")
                    break
                
                # Prepare correction prompt for next attempt
                error_summary = self._summarize_error(last_error_message)
                print(f"      ⚠️  Execution failed: {error_summary}")
                print(f"         Attempting to fix...")
                
                current_prompt = self._build_correction_prompt(
                    original_request=original_user_request,
                    failed_script=current_script,
                    error_message=last_error_message,
                    skill_content=skill_content,
                )

            except Exception as e:
                self.logger.exception(f"Unexpected error (attempt {attempt}): {e}")
                last_error_message = f"Unexpected error: {e}"
                if attempt == MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS:
                    break
                continue
        
        # All attempts failed
        print(f"      ❌ Failed to generate working script after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts")
        return {
            "status": "error",
            "message": f"Failed to generate executable script after {MAX_INTERNAL_SCRIPT_EXEC_CORRECTION_ATTEMPTS} attempts",
            "last_error": last_error_message,
            "last_attempted_script_path": final_script_path,
            "last_attempted_script_content": current_script,
        }

    def _summarize_error(self, error_message: str) -> str:
        """Create a brief, user-friendly error summary."""
        error_lower = error_message.lower()
        
        error_patterns = {
            "modulenotfounderror": "Missing Python module/import",
            "nameerror": "Undefined variable or function",
            "syntaxerror": "Python syntax error",
            "indexerror": "Array/list index out of range",
            "keyerror": "Dictionary key not found",
            "typeerror": "Incorrect data type usage",
            "valueerror": "Invalid value or parameter",
            "filenotfounderror": "File or path not found",
            "timeout": "Script execution timeout",
            "structure_saved": "Missing output confirmation",
        }
        
        for pattern, message in error_patterns.items():
            if pattern in error_lower:
                return message
        
        first_line = error_message.split('\n')[0]
        return first_line[:80] + "..." if len(first_line) > 80 else first_line