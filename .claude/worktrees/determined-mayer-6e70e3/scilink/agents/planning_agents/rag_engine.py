import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import PIL.Image as PIL_Image

from .excel_parser import parse_adaptive_excel
from .parser_utils import parse_json_from_response
from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS,
    HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK,
    TEA_INSTRUCTIONS_FALLBACK
)


def verify_plan_relevance(objective: str, 
                          result: Dict[str, Any], 
                          model: Any, 
                          generation_config: Any) -> Tuple[bool, str]: 
    """
    Self-reflection step. Returns (True, "") if relevant, or (False, "Reason") if not.
    
    Logic:
    1. Checks if the plan was generated via Fallback (General Knowledge).
    2. If Fallback: Verifies only scientific soundness (Relaxed).
    3. If Strict: Verifies document grounding and specific constraint adherence (Strict).
    """
    experiments = result.get("proposed_experiments", [])
    if not experiments: 
        return False, "No experiments generated."

    # 1. Detect Fallback Mode
    # We check if ANY experiment contains the mandatory fallback warning defined in instruct.py
    is_fallback = False
    for exp in experiments:
        justification = exp.get('justification', '').lower()
        if "general scientific knowledge" in justification or "documents lacked specific context" in justification:
            is_fallback = True
            break

    # 2. Build Plan Summary for the Verifier
    plan_summary_lines = []
    for i, exp in enumerate(experiments):
        name = exp.get('experiment_name', 'N/A')
        hyp = exp.get('hypothesis', 'N/A')
        justification = exp.get('justification', 'No justification provided.')
        
        plan_summary_lines.append(f"Experiment {i+1}: {name}")
        plan_summary_lines.append(f"  Hypothesis: {hyp}")
        plan_summary_lines.append(f"  Justification: {justification}") 
        plan_summary_lines.append("---")
        
    plan_summary = "\n".join(plan_summary_lines)

    # 3. Construct Context-Aware Prompt
    if is_fallback:
        print("    - ℹ️  Verifying Fallback Plan (Relaxed Constraints)...")
        eval_prompt = f"""
        You are a scientific research evaluator.
        
        **CONTEXT:** The system failed to find specific documents for the User Objective in the Knowledge Base.
        Therefore, it generated a plan based on **General Scientific Knowledge**.
        
        1. User Objective: "{objective}"
        2. Proposed Plan (General Knowledge): 
        {plan_summary}

        **TASK:**
        Determine if the Proposed Plan makes scientific sense for the Objective, acknowledging that it CANNOT cite specific documents.
        
        **CRITERIA FOR PASS:**
        - The plan addresses the objective using standard, correct scientific principles.
        - The logic is sound and actionable.
        - **DO NOT FAIL** the plan simply because it uses general knowledge or lacks specific context (this is expected in fallback mode).
        
        **Output:**
        Respond with a single JSON object: {{ "is_relevant": boolean, "reason": "string explanation" }}
        """
    else:
        print("    - ℹ️  Verifying Strict Plan (Document Constraints)...")
        eval_prompt = f"""
        You are a scientific research evaluator.
        
        1. User Objective: "{objective}"
        2. Proposed Plan: 
        {plan_summary}

        **TASK:**
        Review the "Hypothesis" and "Justification" for each experiment.
        Determine if the Proposed Plan is directly relevant to the User Objective AND supported by the cited context.
        
        **CRITERIA FOR FAIL:**
        - The plan ignores specific constraints in the objective (e.g., "Use X method" but the plan uses "Y").
        - The justification contradicts the hypothesis.
        - The plan is logically incoherent.
        
        **Output:**
        Respond with a single JSON object: {{ "is_relevant": boolean, "reason": "string explanation" }}
        """

    # 4. Execute Verification
    try:
        response = model.generate_content([eval_prompt], generation_config=generation_config)
        eval_result, _ = parse_json_from_response(response)
        
        if eval_result and not eval_result.get("is_relevant"):
            reason = eval_result.get('reason', 'Unknown irrelevance.')
            print(f"    - ⚠️  Plan Verification Failed: {reason}")
            return False, reason
            
        print(f"    - ✅ Plan Verification Passed.")
        return True, ""
        
    except Exception as e:
        logging.error(f"Verification step failed: {e}")
        # Fail open: If the verifier crashes, we assume the plan is okay to avoid blocking the user.
        return True, ""


def perform_science_rag(objective: str,
                        instructions: str,
                        task_name: str,
                        kb_docs: Any,  # Pass the KB object here
                        model: Any,    # Pass the LLM object here
                        generation_config: Any,
                        primary_data_set: Optional[Dict[str, str]] = None,
                        image_paths: Optional[List[str]] = None,
                        image_descriptions: Optional[List[str]] = None,
                        additional_context: Optional[str] = None,
                        external_context: Optional[str] = None,
                        skill_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes the Scientific/TEA RAG loop using the Docs KnowledgeBase.
    Includes logic for handling Primary Data (Excel) and Fallback generation.
    """
    
    # --- 1. Process Primary Data (e.g., Excel) ---
    primary_data_str = None
    if primary_data_set:
        try:
            chunks = parse_adaptive_excel(primary_data_set['file_path'], primary_data_set['metadata_path'])
            if chunks: 
                summary = next((c for c in chunks if c['metadata'].get('content_type') in ('dataset_summary', 'dataset_package')), chunks[0])
                primary_data_str = summary['text']
        except Exception as e:
            print(f"  - ⚠️ Warning: Failed to parse primary data set: {e}")

    # --- 2. Retrieve Scientific Context (Docs KB Only) ---
    print(f"\n--- Retrieving Scientific Context for {task_name} ---")
    
    doc_chunks = []
    if kb_docs.index and kb_docs.index.ntotal > 0:
        doc_chunks = kb_docs.retrieve(objective, top_k=10)
    
    unique_chunks = {c['text']: c for c in doc_chunks}.values()
    
    if not unique_chunks and not primary_data_str and not external_context:
        retrieved_context_str = "No specific documents found in Knowledge Base."
    else:
        rag_str = "\n\n---\n\n".join(
            f"Source: {Path(c['metadata'].get('source', 'N/A')).name}\nType: {c['metadata'].get('content_type')}\n\n{c['text']}" 
            for c in unique_chunks
        )
        retrieved_context_str = ""

        # Primary Data
        # if primary_data_str: 
        #     retrieved_context_str += f"## 📊 Primary Lab Data Summary\n{primary_data_str}\n\n"
        
        
        # B. External Literature
        if external_context:
            retrieved_context_str += f"## 🌍 External Scientific Literature\n{external_context}\n\n"

        # C. Local Documents
        if rag_str: 
            retrieved_context_str += f"## 📂 Retrieved Local Documents\n{rag_str}"

    # --- 3. Construct Multimodal Prompt ---
    loaded_images = []
    img_desc_str = ""
    
    if image_paths and PIL_Image:
        for p in image_paths:
            try: 
                loaded_images.append(PIL_Image.open(p))
            except Exception as e:
                print(f"  - ⚠️ Could not load image {p}: {e}")

    if image_descriptions:
        img_desc_str = json.dumps(image_descriptions, indent=2)

    prompt_parts = [instructions, f"## User Objective:\n{objective}"]

    if primary_data_str:
        prompt_parts.append(f"\n## 📊 Primary Experimental Data:\n{primary_data_str}")
    
    if loaded_images:
        prompt_parts.append("\n## Provided Images: (See attached)")
        prompt_parts.extend(loaded_images)
        if img_desc_str: prompt_parts.append(f"\n## Image Descriptions:\n{img_desc_str}")
    
    if additional_context:
        prompt_parts.append(f"\n## Additional Context:\n{additional_context}")
        
    prompt_parts.append(f"\n## Retrieved Context:\n{retrieved_context_str}")

    if skill_context:
        prompt_parts.append(skill_context)

    # --- 4. Generation & Fallback Logic ---
    print(f"--- Generating {task_name} ---")
    try:
        # Attempt 1: Strict RAG Generation
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        result, error_msg = parse_json_from_response(response)
        
        if error_msg: 
            return {"error": f"JSON Parsing Error: {error_msg}"}

        # Check for Insufficient Context
        needs_fallback = False
        if result.get("error") and "Insufficient" in str(result.get("error")):
            needs_fallback = True
            print(f"    - ⚠️ Strict generation failed: {result.get('error')}")
        
        # --- 5. Execution of Fallback ---
        if needs_fallback:
            print("    - 🔄 Entering Fallback Mode (General Knowledge)...")
            
            fallback_inst = None
            if instructions == HYPOTHESIS_GENERATION_INSTRUCTIONS:
                fallback_inst = HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
            elif instructions == TEA_INSTRUCTIONS:
                fallback_inst = TEA_INSTRUCTIONS_FALLBACK
            
            if not fallback_inst:
                return result # No fallback available for this instruction set

            prompt_parts[0] = fallback_inst
            
            fallback_response = model.generate_content(prompt_parts, generation_config=generation_config)
            result, error_msg_fb = parse_json_from_response(fallback_response)
            
            if error_msg_fb:
                return {"error": f"Fallback JSON Parsing Error: {error_msg_fb}"}
            
            print("    - ✅ Fallback generation successful.")

        return result

    except Exception as e:
        logging.error(f"Error in perform_science_rag: {e}")
        return {"error": str(e)}


def normalize_code(code: str) -> str:
    """Normalizes code by collapsing all whitespace to single spaces."""
    if not code: return ""
    return " ".join(code.split())


def perform_code_rag(
    result: Dict[str, Any],
    kb_code: Any,
    model: Any,
    generation_config: Any,
    previous_implementations: Optional[List[Dict[str, Any]]] = None,
    skill_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves API syntax from the Code KB and generates implementation scripts.
    If previous code implementations are provided, lets the LLM decide whether to:
    - Preserve existing code (no changes needed)
    - Update existing code (incremental edits)
    - Rewrite from scratch (major procedural changes)
    """
    
    experiments = result.get("proposed_experiments", [])
    if not experiments:
        return result
    
    # 1. Retrieve API documentation from Code KB
    all_steps_text = " ".join([
        " ".join(e.get('experimental_steps', [])) 
        for e in experiments
    ])
    
    print(f"  - 🔍 Retrieving API syntax for implementation...")
    hits = kb_code.retrieve(f"python implementation for {all_steps_text}", top_k=5)
    
    repo_map_context = kb_code.get_relevant_maps(hits) if hits else ""
    code_ctx = "\n\n".join([
        f"FILE: {c['metadata']['source']}\n{c['text']}" 
        for c in hits
    ]) if hits else "No API examples found in Code KB."
    
    code_files = list(set([Path(c['metadata']['source']).name for c in hits])) if hits else []
    
    # 2. Build mapping of previous implementations by experiment name
    previous_code_map = {}
    if previous_implementations:
        for impl in previous_implementations:
            exp_name = impl.get('experiment_name', '')
            if exp_name:
                previous_code_map[exp_name] = impl
    
    # 3. Generate/Update code for each experiment
    for exp in experiments:
        steps = exp.get("experimental_steps", [])
        exp_name = exp.get("experiment_name", "Experiment")
        hypothesis = exp.get("hypothesis", "N/A")
        
        # Find matching previous implementation
        prev_impl = previous_code_map.get(exp_name)
        
        # Build the master prompt
        prompt = f"""
You are an expert Research Software Engineer working on an iterative scientific project.

**EXPERIMENT OVERVIEW:**
Name: {exp_name}
Hypothesis: {hypothesis}

**NEW EXPERIMENTAL STEPS:**
{json.dumps(steps, indent=2)}

"""

        # Add previous implementation context if it exists
        if prev_impl:
            prev_code = prev_impl.get('code', '')
            prev_iteration = prev_impl.get('iteration', 'unknown')
            
            prompt += f"""
**PREVIOUS IMPLEMENTATION (Iteration {prev_iteration}):**
```python
{prev_code}
```

**YOUR DECISION:**
You must choose one of three strategies:

1. **PRESERVE** - If the new steps are identical or the change is only a parameter/value:
   - Return the exact same code unchanged
   - Example: "Increase temperature from 50°C to 60°C" → just parameter change

2. **UPDATE** - If the procedure changed but the overall structure is similar:
   - Keep the working framework (imports, error handling, setup)
   - Modify only the changed sections
   - Add comments marking what changed
   - Example: "Add a centrifugation step after mixing" → insert new function call

3. **REWRITE** - If this is a fundamentally different approach:
   - Start fresh using the API Reference below
   - Example: "Switch from batch processing to real-time streaming"

"""
        else:
            prompt += f"""
**PREVIOUS IMPLEMENTATION:**
None - this is the first implementation for this experiment.

**YOUR TASK:**
Write a complete Python script from scratch using the API Reference below.

"""

        # Add skill context if available
        if skill_context:
            prompt += f"\n{skill_context}\n"

        # Add API context
        prompt += f"""
**REPOSITORY STRUCTURES (for correct import paths):**
{repo_map_context}

**API SYNTAX REFERENCE (Official Documentation/Examples):**
{code_ctx}

**INSTRUCTIONS:**
- Use the "API Syntax Reference" to find the correct functions.
- Map the scientific intent of the Steps to the code.
- You must prioritize using classes and functions from the API Reference over generic external libraries.
- If updating existing code, preserve working patterns
- Return ONLY valid JSON.

**OUTPUT FORMAT:**
Respond with a JSON object:
{{"implementation_code": "COMPLETE_PYTHON_CODE_HERE"}}
"""
        
        try:
            print(f"    - 🤖 Analyzing '{exp_name}'...")
            resp = model.generate_content([prompt], generation_config=generation_config)
            code_res, parse_error = parse_json_from_response(resp)
            
            if parse_error:
                print(f"    - ⚠️ JSON parsing error for '{exp_name}': {parse_error}")
                continue
            
            if code_res and "implementation_code" in code_res:
                new_code = code_res["implementation_code"]
                exp["implementation_code"] = new_code
                exp["code_source_files"] = code_files
                
                if prev_impl:
                    old_code = prev_impl.get('code', '')
                    
                    # Compare normalized versions to ignore harmless whitespace/indentation differences
                    if normalize_code(new_code) == normalize_code(old_code):
                        print(f"    - ⏹️  Preserved (No logic changes): {exp_name}")
                    else:
                        print(f"    - 🔄 Updated: {exp_name}")

                else:
                    print(f"    - ✨ Generated: {exp_name}")
                            
            else:
                print(f"    - ⚠️ LLM did not return code for '{exp_name}'")
                
        except Exception as e:
            print(f"    - ❌ Failed to process '{exp_name}': {e}")
    
    return result


def refine_plan_with_feedback(original_result: Dict[str, Any],
                              feedback: str,
                              objective: str,
                              model: Any,
                              generation_config: Any,
                              new_context: Optional[str] = None,
                              result_images: Optional[List[Any]] = None,
                              skill_context: Optional[str] = None
                              ) -> Dict[str, Any]:
    """
    Refines the experimental plan based on user input or experimental results.
    Now supports injecting fresh RAG context relevant to the feedback/results.
    """
    
    # Construct the context block if available
    context_block = ""
    if new_context:
        context_block = (
            f"\n**📚 RELEVANT LITERATURE FOR OBSERVED RESULTS:**\n"
            f"{new_context}\n"
            f"(Use this literature to interpret the results and adjust the plan accordingly.)\n"
        )

    # Strip source_documents from plan so the LLM only cites references
    # it actually uses during refinement (from KB RAG or external context)
    plan_for_prompt = {k: v for k, v in original_result.items() if k != "source_documents"}

    refinement_prompt = f"""
    You are an expert Research Strategist acting as an editor.

    **Original Objective:** {objective}

    **Current Plan (JSON):**
    {json.dumps(plan_for_prompt, indent=2)}

    **Experimental Results / Feedback:** "{feedback}"
    {context_block}

    **Task:**
    Update the "Current Plan" to strictly address the Feedback and Results.
    - If the results indicate failure, use the Literature Context to propose a fix.
    - If the results indicate success, move to the next logical step.

    **Constraints:**
    - You MUST return the exact same JSON structure (keys: "proposed_experiments", etc.).
    - Update "experimental_steps", "hypothesis", or "required_equipment" as requested.
    - Do NOT add explanations outside the JSON.
    - Do NOT carry forward quantitative claims from the original plan that contradict the experimental results.
    - For "source_documents", list ONLY references you actually used from the provided Literature Context. Do NOT invent or carry forward references not present in the context.

    **Output:**
    A single valid JSON object containing the updated plan.
    """

    prompt_parts = [refinement_prompt]

    if skill_context:
        prompt_parts.append(skill_context)

    if result_images:
        print(f"    + 📎 Attaching {len(result_images)} images to refinement prompt.")
        prompt_parts.extend(result_images)

    try:
        # Generate Content (Sending List of Text + Images)
        response = model.generate_content(prompt_parts, generation_config=generation_config)
        refined_result, error_msg = parse_json_from_response(response)
        
        if error_msg:
            print(f"    - ⚠️ JSON Parsing Failed: {error_msg}. Retrying...")

            raw_text = ""
            if hasattr(response, 'text'):
                raw_text = response.text
            elif hasattr(response, 'parts') and response.parts:
                raw_text = response.parts[0].text

            repair_prompt = (
                "The following text was intended to be valid JSON but has a formatting error.\n\n"
                f"**Error:** {error_msg}\n\n"
                f"**Raw text:**\n{raw_text}\n\n"
                "Fix ONLY the JSON formatting issues (missing commas, unescaped characters, "
                "trailing commas, etc.). Do NOT change any content or values. "
                "Return ONLY the corrected JSON object with no explanation."
            )

            try:
                retry_response = model.generate_content(
                    [repair_prompt], generation_config=generation_config
                )
                refined_result, retry_error = parse_json_from_response(retry_response)

                if retry_error:
                    print(f"    - ⚠️ JSON Retry Also Failed: {retry_error}")
                    return {
                        "error": "JSON_PARSE_ERROR",
                        "message": f"LLM output invalid after retry: {retry_error}",
                        "raw_output": str(raw_text)[:500]
                    }
                else:
                    print(f"    - ✅ JSON repair succeeded on retry.")
            except Exception as retry_exc:
                print(f"    - ⚠️ JSON retry call failed: {retry_exc}")
                return {
                    "error": "JSON_PARSE_ERROR",
                    "message": f"LLM output invalid: {error_msg}",
                    "raw_output": str(raw_text)[:500]
                }
        
        # Structure Validation
        if "proposed_experiments" not in refined_result:
            return {
                "error": "INVALID_STRUCTURE",
                "message": "JSON parsed but missing 'proposed_experiments' key.",
                "raw_output": str(refined_result)[:200]
            }
            
        return refined_result
        
    except Exception as e:
        print(f"    - ⚠️ Error during refinement: {e}")
        return original_result
    

def refine_code_with_feedback(result: Dict[str, Any], 
                              feedback: str, 
                              model: Any, 
                              generation_config: Any) -> Dict[str, Any]:
    """
    Refines the implementation code based on user feedback.
    """
    experiments = result.get("proposed_experiments", [])
    if not experiments:
        return result

    # Context construction: We dump the current code so the LLM knows what to fix
    current_code_state = ""
    for i, exp in enumerate(experiments):
        name = exp.get('experiment_name', f'Experiment {i+1}')
        code = exp.get("implementation_code", "# No code generated")
        current_code_state += f"--- CODE FOR: {name} ---\n{code}\n\n"

    prompt = f"""
    You are a Senior Research Software Engineer.
    
    **TASK:** Refine the Python implementation code based on User Feedback.
    
    **CURRENT CODE STATE:**
    {current_code_state}
    
    **USER FEEDBACK / ERROR REPORT:**
    "{feedback}"
    
    **INSTRUCTIONS:**
    1. Apply the user's fixes to the relevant code blocks.
    2. If the user refers to a specific experiment, only update that one.
    3. You must return a JSON object with a list of "updated_codes". 
       Each item in the list must match the order of the experiments above.
    4. Provide the FULL updated code for each script, not just the diffs.
    
    **OUTPUT FORMAT:**
    {{
        "updated_codes": [
            "FULL_PYTHON_SCRIPT_1...",
            "FULL_PYTHON_SCRIPT_2..."
        ]
    }}
    """
    
    print(f"    - ↻ Refine Code RAG: Generating updates based on feedback...")
    try:
        response = model.generate_content([prompt], generation_config=generation_config)
        updates, error = parse_json_from_response(response)
        
        if updates and "updated_codes" in updates:
            new_codes = updates["updated_codes"]
            # Map back to the result structure
            if len(new_codes) == len(experiments):
                for i, code in enumerate(new_codes):
                    experiments[i]["implementation_code"] = code
                print("    - ✅ Code successfully refined.")
            else:
                print("    - ⚠️ Warning: LLM returned wrong number of code blocks. Skipping update.")
        elif error:
            print(f"    - ⚠️ JSON Error during refinement: {error}")
        
        return result
        
    except Exception as e:
        print(f"    - ❌ Error during code refinement: {e}")
        return result