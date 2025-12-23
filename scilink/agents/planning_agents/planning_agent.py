import os
from pathlib import Path
import google.generativeai as genai
import json
import logging
import shutil
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import PIL.Image as PIL_Image

from .knowledge_base import KnowledgeBase
from .pdf_parser import extract_pdf_two_pass, chunk_text
from .excel_parser import parse_adaptive_excel
from .parser_utils import (
    get_files_from_directory, 
    generate_repo_map, 
    write_experiments_to_disk
)
from .repo_loader import clone_git_repository

from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS
)

from ...auth import get_api_key, APIKeyNotFoundError
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ..lit_agents.literature_agent import LiteratureSearchAgent
from ..lit_agents.optimize_query import optimize_search_query

from .rag_engine import (
    perform_science_rag, 
    perform_code_rag, 
    refine_plan_with_feedback,
    refine_code_with_feedback,
    verify_plan_relevance
)
from .user_interface import display_plan_summary, get_user_feedback

from .html_generator import HTMLReportGenerator


logger = logging.getLogger(__name__)

class PlanningAgent:
    """
    Stateful AI Agent for Autonomous Experimental Planning and Iteration.
    
    The PlanningAgent orchestrates end-to-end research workflows by combining:
    - Dual Knowledge Base system (scientific literature + implementation code)
    - RAG-based hypothesis generation and technoeconomic analysis
    - LLM-driven code generation from experimental procedures
    - Human-in-the-loop feedback at strategic decision points
    - Iterative refinement based on experimental results
    
    Maintains a persistent 'state' dictionary to track:
    - The Research Objective
    - The Evolving Experimental Plan (Science -> Code)
    - Results from executed experiments
    - Feedback history (both Scientific Plan and Code Implementation)

    Args:
        google_api_key (str, optional): API key for Gemini models.
            If not provided, attempts to load from environment.
        futurehouse_api_key (str, optional): FutureHouse API key for literature search.
            If not provided, literature search will be skipped.
        model_name (str, optional): Name of the LLM to use. 
            Defaults to "gemini-3-pro-preview".
        local_model (str, optional): Base URL for OpenAI-compatible local models.
            If provided, uses OpenAI wrapper instead of Gemini.
        embedding_model (str, optional): Embedding model for knowledge bases.
            Defaults to "gemini-embedding-001".
        kb_base_path (str, optional): Base path for knowledge base storage.
            Creates separate `_docs` and `_code` knowledge bases.
            Defaults to "./kb_storage/default_kb".
        code_chunk_size (int, optional): Chunk size for code files in tokens.
            Defaults to 20000 (larger than docs for context preservation).
    """
    def __init__(self, google_api_key: str = None,
                 futurehouse_api_key: str = None,
                 model_name: str = "gemini-3-pro-preview",
                 local_model: str = None,
                 embedding_model: str = "gemini-embedding-001",
                 kb_base_path: str = "./kb_storage/default_kb",
                 code_chunk_size: int = 20000,
                 quiet_mode: bool = False): 
        
        self.quiet_mode = quiet_mode
        
        if google_api_key is None:
            google_api_key = get_api_key('google')
            if not google_api_key:
                raise APIKeyNotFoundError('google')

        # --- LLM Backend Configuration ---
        if local_model and ('ai-incubator' in local_model or 'openai' in local_model):
            if not quiet_mode:
                logging.info(f"🏛️  Using OpenAI-compatible model for generation: {model_name}")
            self.model = OpenAIAsGenerativeModel(model_name, api_key=google_api_key, base_url=local_model)
            self.generation_config = None
        else:
            if not quiet_mode:
                logging.info(f"☁️  Using Google Gemini model for generation: {model_name}")
            genai.configure(api_key=google_api_key)
            self.model = genai.GenerativeModel(model_name)
            self.generation_config = genai.types.GenerationConfig(response_mime_type="application/json")

        self.lit_agent = None
        if futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY"):
            try:
                self.lit_agent = LiteratureSearchAgent(futurehouse_api_key, max_wait_time=1000)
                if not quiet_mode:
                    logging.info("✅ Literature Search Agent initialized.")
            except Exception as e:
                if not quiet_mode:
                    logging.warning(f"⚠️ Failed to initialize Literature Agent: {e}")
        else:
            if not quiet_mode:
                logging.info("ℹ️ No FutureHouse API key provided. Literature search will be skipped.")
                    
        self.code_chunk_size = code_chunk_size

        # --- Dual KnowledgeBase Initialization ---
        base_path = Path(kb_base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Scientific/Docs KB
        self.kb_docs = KnowledgeBase(google_api_key=google_api_key, embedding_model=embedding_model, local_model=local_model)
        self.kb_docs_prefix = base_path.parent / f"{base_path.name}_docs"
        self.kb_docs_index = str(self.kb_docs_prefix.with_suffix(".faiss"))
        self.kb_docs_chunks = str(self.kb_docs_prefix.with_suffix(".json"))

        # 2. Implementation/Code KB
        self.kb_code = KnowledgeBase(google_api_key=google_api_key, embedding_model=embedding_model, local_model=local_model)
        self.kb_code_prefix = base_path.parent / f"{base_path.name}_code"
        self.kb_code_index = str(self.kb_code_prefix.with_suffix(".faiss"))
        self.kb_code_chunks = str(self.kb_code_prefix.with_suffix(".json"))
        self.kb_code_map_path = str(self.kb_code_prefix.with_suffix(".maps.json"))

        if not quiet_mode:
            print("--- Initializing Agent (Dual-KB System) ---")
        self._load_knowledge_bases()

        # --- STATE MANAGEMENT ---
        self.state: Dict[str, Any] = {}

    def restore_state(self, state_file_path: str) -> None:
        """
        Restore agent state from a saved .state.json file.
        
        Args:
            state_file_path: Path to the .state.json file
            
        Example:
            agent = PlanningAgent()
            agent.restore_state("./outputs/session.state.json")
        """        
        path = Path(state_file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {state_file_path}")
        
        if path.suffix != '.json':
            raise ValueError(f"State file must be a .json file, got: {path.suffix}")
        
        print(f"  - 📂 Loading state from: {path.name}")
        
        try:
            with open(path, 'r') as f:
                saved_state = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in state file: {e}")
        
        # Validate structure
        required = ["objective", "current_plan", "iteration_index", "session_id"]
        missing = [f for f in required if f not in saved_state]
        
        if missing:
            raise ValueError(
                f"Invalid state file structure. Missing required fields: {missing}\n"
                f"Expected a complete .state.json file with keys: {required}"
            )
        
        # Restore
        self.state = saved_state
        
        if not self.quiet_mode:
            print(f"  - ✅ Restored session: {saved_state['session_id']}")
            print(f"     • Objective: {saved_state['objective'][:80]}...")
            print(f"     • Current iteration: {saved_state['iteration_index']}")
            print(f"     • History entries: {len(saved_state.get('plan_history', []))}")
            print(f"     • Previous results: {len(saved_state.get('experimental_results', []))}")
        
    def _load_knowledge_bases(self):
        """Attempts to load both KBs from disk."""
        if not self.quiet_mode:
            print(f"  - Docs KB: Loading from {self.kb_docs_prefix}...")
        docs_loaded = self.kb_docs.load(self.kb_docs_index, self.kb_docs_chunks)
        
        if not self.quiet_mode:
            print(f"  - Code KB: Loading from {self.kb_code_prefix}...")
        code_loaded = self.kb_code.load(self.kb_code_index, self.kb_code_chunks, self.kb_code_map_path)

        self._kb_is_built = docs_loaded or code_loaded
        
        if not self.quiet_mode:
            if docs_loaded: print("    - ✅ Docs KB loaded.")
            if code_loaded: print("    - ✅ Code KB loaded.")
            if not self._kb_is_built: print("    - ⚠️  No pre-built KBs found.")

    def _initialize_state(self, objective: str, **kwargs) -> Dict[str, Any]:
        """Creates the foundational state dictionary for a new research task."""
        return {
            "session_id": str(uuid.uuid4()),
            "start_time": datetime.now().isoformat(),
            "objective": objective,
            "iteration_index": 0,
            
            # Inputs
            "inputs": {
                "science_paths": kwargs.get("science_paths", []),
                "code_paths": kwargs.get("code_paths", []),
                "additional_context": kwargs.get("additional_context"),
                "primary_data_set": kwargs.get("primary_data_set"),
                "image_paths": kwargs.get("image_paths", []),
                "image_descriptions": kwargs.get("image_descriptions", [])
            },

            # Plan Evolution
            "current_plan": None,   # The active plan dict
            "plan_history": [],     # Snapshots of previous plans
            
            # Feedback Loop
            "experimental_results": [],  # List of result dicts from the lab
            "human_feedback_history": [],
            
            # Status
            "last_error": None,
            "status": "initialized"
        }

    def _save_results_to_json(self, results: Dict[str, Any], file_path: str):
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f: json.dump(results, f, indent=2)
            if not self.quiet_mode:
                print(f"    - ✅ Results successfully saved to: {file_path}")
        except Exception as e: logging.error(f"    - ❌ Failed to save results: {e}")

    def _save_state_to_json(self, file_path: str):
        """Saves the full state dictionary (history, results) to a sidecar file."""
        try:
            p = Path(file_path)
            with p.open('w', encoding='utf-8') as f: json.dump(self.state, f, indent=2)
        except Exception as e: logging.error(f"    - ❌ Failed to save state: {e}")

    def _generate_html_report(self, json_path: str):
        """Helper to generate HTML report alongside JSON."""
        if not json_path: return
        html_path = str(Path(json_path).with_suffix('.html'))
        try:
            generator = HTMLReportGenerator(self.state)
            generator.generate(html_path)
        except Exception as e:
            if not self.quiet_mode:
                print(f"⚠️ Failed to generate HTML report: {e}")

    def _process_file_list(self, file_paths: List[str], is_code_mode: bool, repo_name: str = None) -> List[Dict[str, Any]]:
        """Generic helper to process a list of files OR directories."""
        chunks = []
        expanded_paths = []
        if file_paths:
            for f_path in file_paths:
                path_obj = Path(f_path)
                if path_obj.is_dir():
                    expanded_paths.extend(get_files_from_directory(f_path))
                else:
                    expanded_paths.append(f_path)

        for f_path in expanded_paths:
            path = Path(f_path)
            if not path.exists():
                print(f"  - ⚠️ File not found: {f_path}")
                continue
            
            file_ext = path.suffix.lower()
            if file_ext == '.pdf':
                pdf_chunks = extract_pdf_two_pass(f_path)
                if is_code_mode:
                    for c in pdf_chunks: c['metadata']['content_type'] = 'code'
                chunks.extend(pdf_chunks)
            elif file_ext in ['.txt', '.md', '.py', '.java', '.r', '.cpp', '.h', '.js', '.json', '.csv']:
                try:
                    with path.open('r', encoding='utf-8') as f: content = f.read()
                    if is_code_mode:
                        formatted_text = f"CODE FILE: {path.name}\n\n```\n{content}\n```"
                        chunk_sz = self.code_chunk_size
                        ctype = 'code'
                    else:
                        formatted_text = f"DOCUMENT: {path.name}\n\n{content}"
                        chunk_sz = 1000
                        ctype = 'text'
                    new_chunks = chunk_text(formatted_text, page_num=1, chunk_size=chunk_sz, overlap=50)
                    for c in new_chunks: 
                        c['metadata']['content_type'] = ctype
                        c['metadata']['source'] = f_path
                    chunks.extend(new_chunks)
                    print(f"  - Extracted {len(new_chunks)} chunks from {path.name} ({'Code' if is_code_mode else 'Docs'} Mode)")
                except Exception as e:
                    print(f"  - ❌ Error reading {f_path}: {e}")
            else:
                print(f"  - ⚠️ Unsupported file type: {f_path}")
        return chunks

    def _build_and_save_kb(self, science_paths: Optional[List[str]] = None, code_paths: Optional[List[str]] = None, structured_data_sets: Optional[List[Dict[str, str]]] = None) -> bool:
        """Builds TWO separate knowledge bases based on explicit input lists."""
        print("\n--- Rebuilding Knowledge Bases ---")
        
        # 1. Build Docs KB (Science)
        doc_chunks = []
        if science_paths:
            print(f"Processing {len(science_paths)} Scientific Documents...")
            doc_chunks.extend(self._process_file_list(science_paths, is_code_mode=False))
        if structured_data_sets:
            print(f"Processing {len(structured_data_sets)} Structured Data Sets...")
            for data_set in structured_data_sets:
                try:
                    if Path(data_set['file_path']).suffix.lower() in ['.xlsx', '.xls', '.csv']:
                        excel_chunks = parse_adaptive_excel(data_set['file_path'], data_set['metadata_path'])
                        if excel_chunks: doc_chunks.extend(excel_chunks)
                except Exception as e: print(f"  - ❌ Error processing Excel: {e}")

        if doc_chunks:
            print(f"  - Building Scientific KB with {len(doc_chunks)} chunks...")
            self.kb_docs.build(doc_chunks)
            self.kb_docs.save(self.kb_docs_index, self.kb_docs_chunks)
        else:
            print("  - ℹ️  No Scientific docs provided. Docs KB unchanged (or empty).")

        # 2. Build Code KB (Implementation)
        code_chunks = []
        if code_paths:
            print(f"Processing {len(code_paths)} Implementation/Code Documents...")
            for p in code_paths:
                path_obj = Path(p)
                if path_obj.is_dir():
                    repo_name = path_obj.name
                    print(f"  - 📦 Processing Repo: {repo_name}")
                    self.kb_code.repo_maps[repo_name] = generate_repo_map(str(path_obj))
                    repo_chunks = self._process_file_list([p], is_code_mode=True, repo_name=repo_name)
                    code_chunks.extend(repo_chunks)
                else:
                    file_chunks = self._process_file_list([p], is_code_mode=True)
                    code_chunks.extend(file_chunks)
            
        if code_chunks:
            print(f"  - Building Code KB with {len(code_chunks)} chunks...")
            self.kb_code.build(code_chunks)
            self.kb_code.save(self.kb_code_index, self.kb_code_chunks, self.kb_code_map_path)
        else:
            print("  - ℹ️  No Code docs provided. Code KB unchanged (or empty).")

        self._kb_is_built = True
        print("✅ Dual-KB Build Complete.")
        return True

    def _ensure_kb_is_ready(self, science_paths, code_paths, structured_data_sets) -> bool:
        new_inputs = (science_paths or []) or (code_paths or []) or (structured_data_sets or [])
        if new_inputs:
            return self._build_and_save_kb(science_paths, code_paths, structured_data_sets)
        elif not self._kb_is_built:
            logging.error("Knowledge base is not built.")
            return False
        return True

    def generate_experimental_plan(self,
                                   objective: str,
                                   science_paths: Optional[List[str]] = None,
                                   code_paths: Optional[List[str]] = None,
                                   structured_data_sets: Optional[List[Dict[str, str]]] = None,
                                   additional_context: Optional[Dict[str, str]] = None,
                                   primary_data_set: Optional[Dict[str, str]] = None,
                                   image_paths: Optional[List[str]] = None,
                                   image_descriptions: Optional[List[str]] = None,
                                   output_json_path: Optional[str] = None,
                                   reset_state: bool = False) -> Dict[str, Any]:
        """
        Generate experimental plan from scientific literature.
        
        This is the first discrete step in the workflow. It:
        1. Builds/loads knowledge bases
        2. Performs RAG retrieval
        3. Generates hypotheses and experimental steps
        4. Saves plan and state
        
        Does NOT generate code or prompt for feedback.
        
        Args:
            objective: Research goal
            science_paths: Paths to scientific documents
            code_paths: Paths to code (stored for later use)
            structured_data_sets: Excel/CSV datasets
            additional_context: Extra context as key-value pairs
            primary_data_set: Main dataset to analyze
            image_paths: Paths to images
            image_descriptions: Descriptions for images
            output_json_path: Where to save plan
            reset_state: Clear existing state
        
        Returns:
            {
                'session_id': str,
                'plan': Dict,  # The generated plan
                'state_path': str,  # For next steps
                'status': str
            }
        """
        if not self.quiet_mode:
            print(f"\n=== STEP 1: Generating Experimental Plan ===")
        
        # Initialize or reset state
        if reset_state or not self.state:
            self.state = self._initialize_state(
                objective=objective,
                science_paths=science_paths,
                code_paths=code_paths,
                additional_context=additional_context,
                primary_data_set=primary_data_set,
                image_paths=image_paths,
                image_descriptions=image_descriptions
            )
        else:
            if objective:
                self.state["objective"] = objective

        self.state["iteration_index"] = self.state.get("iteration_index", 0) + 1
        current_iter = self.state["iteration_index"]

        # Ensure KB is ready
        if not self._ensure_kb_is_ready(science_paths, code_paths, structured_data_sets):
            self.state["status"] = "failed"
            self.state["last_error"] = "KB initialization failed"
            return {
                'success': False,
                'error': 'KB initialization failed',
                'session_id': self.state.get('session_id')
            }

        # Build context string
        ctx_string = ""
        if additional_context:
            for header, content in additional_context.items():
                ctx_string += f"## {header}\n{content}\n\n"
            ctx_string = ctx_string.strip()

        # Literature search (optional)
        lit_context = ""
        if self.lit_agent:
            if not self.quiet_mode:
                print(f"  - 🌍 Querying literature...")
            try:
                lit_res = self.lit_agent.search_for_hypothesis_context(
                    optimize_search_query(objective=objective, model=self.model)
                )
                if lit_res['status'] == 'success':
                    lit_context = lit_res['content']
            except Exception as e:
                logger.warning(f"Literature search failed: {e}")

        # Perform RAG to generate plan
        if not self.quiet_mode:
            print(f"  - 🔬 Generating experimental hypotheses...")
        
        res = perform_science_rag(
            objective=objective,
            instructions=HYPOTHESIS_GENERATION_INSTRUCTIONS,
            task_name="Experimental Plan",
            kb_docs=self.kb_docs,
            model=self.model,
            generation_config=self.generation_config,
            primary_data_set=primary_data_set,
            image_paths=image_paths,
            image_descriptions=image_descriptions,
            additional_context=ctx_string,
            external_context=lit_context
        )

        if lit_context:
            res["literature_search"] = lit_context

        # Save to history
        res["iteration"] = current_iter
        res["stage"] = "Plan Generated"
        self.state["plan_history"].append(res.copy())
        self.state["current_plan"] = res
        self.state["status"] = "plan_generated"

        # Self-verification (optional quality check)
        if not res.get("error"):
            is_relevant, critique = verify_plan_relevance(objective, res, self.model, self.generation_config)
            
            if not is_relevant:
                if not self.quiet_mode:
                    print(f"\n🔄 Self-correction: {critique}")
                
                res = refine_plan_with_feedback(
                    original_result=res,
                    feedback=f"CORRECTION: {critique}",
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                
                res["iteration"] = current_iter
                res["stage"] = "Plan Auto-Corrected"
                self.state["plan_history"].append(res.copy())
                self.state["current_plan"] = res

        # Save outputs
        state_path = None
        if output_json_path:
            self._save_results_to_json(res, output_json_path)
            state_path = f"{output_json_path}.state.json"
            self._save_state_to_json(state_path)
            self._generate_html_report(output_json_path)
        
        if not self.quiet_mode:
            print(f"  - ✅ Plan generation complete")
        
        return {
            'success': True,
            'session_id': self.state['session_id'],
            'plan': res,
            'state_path': state_path,
            'status': self.state['status']
        }
    
    def revise_experimental_plan(self,
                                 feedback: str,
                                 state_path: str,
                                 output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Revise experimental plan based on user feedback.
        
        Can be called multiple times to iteratively refine the plan.
        Only modifies the experimental strategy, NOT the code.
        
        Args:
            feedback: User's requested changes
            state_path: Path to .state.json from previous step
            output_json_path: Where to save revised plan
        
        Returns:
            {
                'session_id': str,
                'revised_plan': Dict,
                'state_path': str,
                'status': str
            }
        """
        if not self.quiet_mode:
            print(f"\n=== STEP 2: Revising Experimental Plan ===")
            print(f"  - Feedback: {feedback[:100]}...")
        
        # Restore state
        self.restore_state(state_path)
        
        current_plan = self.state.get('current_plan')
        if not current_plan:
            return {
                'success': False,
                'error': 'No current plan found in state'
            }
        
        objective = self.state.get('objective')
        
        # Apply refinement
        if not self.quiet_mode:
            print(f"  - 🔄 Applying feedback to plan...")
        
        refined_plan = refine_plan_with_feedback(
            original_result=current_plan,
            feedback=feedback,
            objective=objective,
            model=self.model,
            generation_config=self.generation_config
        )
        
        # Update state
        refined_plan["iteration"] = self.state['iteration_index']
        refined_plan["stage"] = "Plan Revised"
        self.state["plan_history"].append(refined_plan.copy())
        self.state["current_plan"] = refined_plan
        self.state["human_feedback_history"].append({
            'phase': 'plan_revision',
            'feedback': feedback
        })
        
        # Save outputs
        state_path_out = None
        if output_json_path:
            self._save_results_to_json(refined_plan, output_json_path)
            state_path_out = f"{output_json_path}.state.json"
            self._save_state_to_json(state_path_out)
            self._generate_html_report(output_json_path)
        
        if not self.quiet_mode:
            print(f"  - ✅ Plan revision complete")
        
        return {
            'success': True,
            'session_id': self.state['session_id'],
            'revised_plan': refined_plan,
            'state_path': state_path_out,
            'status': 'plan_revised'
        }
    
    def generate_implementation_code(self,
                                     state_path: str,
                                     output_code_dir: str = "./output_scripts",
                                     output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate implementation code from experimental plan.
        
        Reads the plan from state and generates Python scripts using
        the Code KB (API documentation, examples).
        
        Args:
            state_path: Path to .state.json containing plan
            output_code_dir: Where to save generated scripts
            output_json_path: Where to save plan with code
        
        Returns:
            {
                'session_id': str,
                'code_files': List[str],
                'code_dir': str,
                'state_path': str,
                'status': str
            }
        """
        if not self.quiet_mode:
            print(f"\n=== STEP 3: Generating Implementation Code ===")
        
        # Restore state
        self.restore_state(state_path)
        
        current_plan = self.state.get('current_plan')
        if not current_plan:
            return {
                'success': False,
                'error': 'No current plan found in state'
            }
        
        # Check if Code KB is available
        if not self.kb_code.index or self.kb_code.index.ntotal == 0:
            if not self.quiet_mode:
                print("  - ⚠️  Code KB is empty. Skipping code generation.")
            return {
                'success': True,
                'session_id': self.state['session_id'],
                'code_files': [],
                'code_dir': output_code_dir,
                'message': 'No code KB available'
            }
        
        # Perform code RAG
        if not self.quiet_mode:
            print(f"  - 💻 Generating implementation scripts...")
        
        result = perform_code_rag(
            result=current_plan,
            kb_code=self.kb_code,
            model=self.model,
            generation_config=self.generation_config
        )
        
        # Update state
        result["iteration"] = self.state['iteration_index']
        result["stage"] = "Code Generated"
        self.state["plan_history"].append(result.copy())
        self.state["current_plan"] = result
        
        # Write code to disk
        Path(output_code_dir).mkdir(parents=True, exist_ok=True)
        code_files = write_experiments_to_disk(result, output_code_dir)
        
        if not self.quiet_mode:
            print(f"  - 💾 Saved {len(code_files)} scripts to {output_code_dir}")
        
        # Save outputs
        state_path_out = None
        if output_json_path:
            self._save_results_to_json(result, output_json_path)
            state_path_out = f"{output_json_path}.state.json"
            self._save_state_to_json(state_path_out)
            self._generate_html_report(output_json_path)
        
        if not self.quiet_mode:
            print(f"  - ✅ Code generation complete")
        
        return {
            'success': True,
            'session_id': self.state['session_id'],
            'code_files': code_files,
            'code_dir': output_code_dir,
            'state_path': state_path_out,
            'status': 'code_generated'
        }
    
    def revise_implementation_code(self,
                                   feedback: str,
                                   state_path: str,
                                   output_code_dir: str = "./output_scripts",
                                   output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Revise implementation code based on user feedback.
        
        Can be called multiple times to iteratively fix bugs, adjust
        API usage, add error handling, etc.
        
        Only modifies the code, NOT the experimental strategy.
        
        Args:
            feedback: User's requested code changes
            state_path: Path to .state.json containing current code
            output_code_dir: Where to save revised scripts
            output_json_path: Where to save plan with revised code
        
        Returns:
            {
                'session_id': str,
                'revised_code_files': List[str],
                'code_dir': str,
                'state_path': str,
                'status': str
            }
        """
        if not self.quiet_mode:
            print(f"\n=== STEP 4: Revising Implementation Code ===")
            print(f"  - Feedback: {feedback[:100]}...")
        
        # Restore state
        self.restore_state(state_path)
        
        current_plan = self.state.get('current_plan')
        if not current_plan:
            return {
                'success': False,
                'error': 'No current plan found in state'
            }
        
        # Apply code refinement
        if not self.quiet_mode:
            print(f"  - 🔄 Applying feedback to code...")
        
        refined_result = refine_code_with_feedback(
            result=current_plan,
            feedback=feedback,
            model=self.model,
            generation_config=self.generation_config
        )
        
        # Update state
        refined_result["iteration"] = self.state['iteration_index']
        refined_result["stage"] = "Code Revised"
        self.state["plan_history"].append(refined_result.copy())
        self.state["current_plan"] = refined_result
        self.state["human_feedback_history"].append({
            'phase': 'code_revision',
            'feedback': feedback
        })
        
        # Write revised code to disk
        Path(output_code_dir).mkdir(parents=True, exist_ok=True)
        code_files = write_experiments_to_disk(refined_result, output_code_dir)
        
        if not self.quiet_mode:
            print(f"  - 💾 Updated {len(code_files)} scripts in {output_code_dir}")
        
        # Save outputs
        state_path_out = None
        if output_json_path:
            self._save_results_to_json(refined_result, output_json_path)
            state_path_out = f"{output_json_path}.state.json"
            self._save_state_to_json(state_path_out)
            self._generate_html_report(output_json_path)
        
        if not self.quiet_mode:
            print(f"  - ✅ Code revision complete")
        
        return {
            'success': True,
            'session_id': self.state['session_id'],
            'revised_code_files': code_files,
            'code_dir': output_code_dir,
            'state_path': state_path_out,
            'status': 'code_revised'
        }
    
    def propose_experiments(self,
                           objective: str,
                           science_paths: Optional[List[str]] = None,
                           code_paths: Optional[List[str]] = None,
                           structured_data_sets: Optional[List[Dict[str, str]]] = None,
                           additional_context: Optional[Dict[str, str]] = None,
                           primary_data_set: Optional[Dict[str, str]] = None,
                           image_paths: Optional[List[str]] = None,
                           image_descriptions: Optional[List[str]] = None,
                           output_json_path: Optional[str] = None,
                           output_code_dir: str = "./output_scripts",
                           enable_human_feedback: bool = True,
                           reset_state: bool = False) -> Dict[str, Any]:
        """
        Generate an experimental plan based on scientific literature and implementation knowledge.

        This is the primary entry point for starting a new research workflow. The agent:
        1. Builds/loads dual knowledge bases (scientific docs + implementation code)
        2. Optionally queries external literature databases
        3. Generates experimental hypotheses via RAG
        4. Maps experimental steps to executable code
        5. Provides human-in-the-loop review at both science and code stages

        Args:
            objective (str): High-level research goal. This guides all hypothesis generation
                and plan refinement. Should be specific and measurable.
                Examples:
                    - "Optimize the yield of the Suzuki coupling reaction"
                    - "Screen 96 conditions to selectively precipitate magnesium"
                    - "Develop a high-throughput assay for enzyme activity"
            
            science_paths (Optional[List[str]]): Paths to scientific documents/data.
                Supported formats: PDFs, .txt, .md, directories (recursively searched)
                These populate the Docs Knowledge Base for hypothesis generation.
                Examples: ["./papers/", "./lab_notebooks/protocol.pdf"]
            
            code_paths (Optional[List[str]]): Paths to code repositories or API documentation.
                Supported formats: Local directories, Git URLs, Python files
                These populate the Code Knowledge Base for implementation.
                Examples:
                    - ["./opentrons_api/"]  # Local repo
                    - ["https://github.com/org/automation-lib.git"]  # Git URL
            
            structured_data_sets (Optional[List[Dict[str, str]]]): Large Excel/CSV datasets
                with metadata for adaptive parsing. Each dict should contain:
                    - 'file_path': Path to .xlsx or .csv file
                    - 'metadata_path': Path to .json metadata file (optional)
                Example: [{"file_path": "./data.xlsx", "metadata_path": "./data.json"}]
            
            additional_context (Optional[Dict[str, str]]): Additional text context
                to inject into the prompt. Keys become section headers.
                Example: {
                    "Safety Constraints": "Maximum temperature is 80°C",
                    "Equipment Available": "Opentrons OT-2, plate reader"
                }
            
            primary_data_set (Optional[Dict[str, str]]): Main dataset to analyze.
                Similar format to structured_data_sets, but gets priority placement
                in the prompt. Use for the dataset that drives the research objective.
                Example: {"file_path": "./screening_results.xlsx"}
            
            image_paths (Optional[List[str]]): Paths to images (plots, diagrams, photos).
                Supported formats: .png, .jpg, .jpeg, .tiff, .bmp
                These are passed to the vision model for multimodal analysis.
                Examples: ["./criticality_matrix.png", "./reaction_scheme.jpg"]
            
            image_descriptions (Optional[List[str]]): Text descriptions for each image.
                Should be in same order as image_paths. Helps LLM interpret images.
                Examples: ["Criticality matrix showing material supply risks"]
            
            output_json_path (Optional[str]): Path to save the generated plan.
                Also saves full state to {output_json_path}.state.json
                and generates HTML report at {output_json_path}.html
                Example: "./outputs/experiment_plan.json"
            
            enable_human_feedback (bool): If True, pauses for user input at:
                - Strategy review (after hypothesis generation)
                - Code review (after script generation)
                Set to False for fully autonomous operation.
                Defaults to True.
            
            reset_state (bool): If True, clears any existing state and starts fresh.
                If False, appends to existing research session (cumulative workflow).
                Defaults to False.
        
        Returns:
            Dict[str, Any]: Complete agent state containing:
                - session_id: Unique identifier for this session
                - objective: The research objective
                - iteration_index: Current iteration number (1 for initial plan)
                - current_plan: The active experimental plan, structure
        """
        
        if not self.quiet_mode:
            print("\n" + "="*60)
            print("LEGACY MODE: propose_experiments()")
            print("="*60)
        
        # Step 1: Generate Plan
        result = self.generate_experimental_plan(
            objective=objective,
            science_paths=science_paths,
            code_paths=code_paths,
            structured_data_sets=structured_data_sets,
            additional_context=additional_context,
            primary_data_set=primary_data_set,
            image_paths=image_paths,
            image_descriptions=image_descriptions,
            output_json_path=output_json_path,
            reset_state=reset_state
        )
        
        if not result['success']:
            self.state["status"] = "failed"
            return self.state
        
        # Step 2: Human Feedback on Plan (if enabled)
        if enable_human_feedback and result['plan'].get("proposed_experiments"):
            display_plan_summary(result['plan'], quiet_mode=self.quiet_mode)
            user_feedback = get_user_feedback(enable_interactive=enable_human_feedback)
            
            if user_feedback:
                if not self.quiet_mode:
                    print(f"\n📝 Refining plan based on feedback...")
                
                result = self.revise_experimental_plan(
                    feedback=user_feedback,
                    state_path=result['state_path'],
                    output_json_path=output_json_path
                )
                
                display_plan_summary(result['revised_plan'], quiet_mode=self.quiet_mode)
        
        # Step 3: Generate Code
        if result.get('state_path'):
            code_result = self.generate_implementation_code(
                state_path=result['state_path'],
                output_code_dir=output_code_dir,
                output_json_path=output_json_path
            )
            
            # Step 4: Human Feedback on Code (if enabled)
            if enable_human_feedback and code_result.get('code_files'):
                while True:
                    if not self.quiet_mode:
                        print("\n" + "="*60)
                        print("👀 CODE REVIEW")
                        print("="*60)
                        print(f"Scripts saved to: {code_result['code_dir']}")
                        print("Review the code and provide feedback, or press Enter to continue.")
                    
                    code_feedback = get_user_feedback(enable_interactive=enable_human_feedback)
                    
                    if not code_feedback:
                        if not self.quiet_mode:
                            print("✅ Code accepted.")
                        break
                    
                    if not self.quiet_mode:
                        print(f"\n🔄 Refining code based on feedback...")
                    
                    code_result = self.revise_implementation_code(
                        feedback=code_feedback,
                        state_path=code_result['state_path'],
                        output_code_dir=output_code_dir,
                        output_json_path=output_json_path
                    )
        
        self.state["status"] = "completed"
        return self.state

    def update_plan_with_results(self,
                                 results: Any,
                                 output_json_path: Optional[str] = None,
                                 enable_human_feedback: bool = True,
                                 state_file_path: Optional[str] = None 
                                 ) -> Dict[str, Any]:
        """
        Iterates on the current experimental plan based on new experimental results, 
        observations, or data files.

        This method acts as the "feedback loop" of the agent, transforming the system from 
        a linear planner into an iterative scientific partner. It performs Smart Result Parsing, 
        Result-Aware RAG, and Human-in-the-Loop refinement.

        **Capabilities & Workflow:**

        1.  **Smart Result Parsing (Multimodal):**
            -   Detects and parses input types automatically.
            -   **Text/Dicts/Lists:** Converted to JSON strings for the LLM prompt.
            -   **Data Files (.xlsx, .csv):** Automatically summarized using `excel_parser` and injected as text context.
            -   **Images (.png, .jpg):** Loaded and passed to the vision model for visual analysis (e.g., plot trends, failures).
            -   **Logs (.txt, .log):** Read and injected as context.

        2.  **Result-Aware RAG (Retrieval Augmented Generation):**
            -   Uses the content of the results to perform a *new* targeted search in the Docs Knowledge Base (`kb_docs`).
            -   Example: If results mention "precipitation," it retrieves papers discussing solubility limits, even if those papers weren't relevant to the initial plan.

        3.  **Nuanced Scientific Reasoning:**
            -   Prompts the LLM to categorize the outcome into one of five strategic buckets:
                * **CONFIRMED:** Validated hypothesis -> Propose next step.
                * **OPTIMIZATION NEEDED:** Valid sub-optimal result -> Tune parameters (Do not change hypothesis).
                * **INCONCLUSIVE:** Noisy data -> Refine measurement technique.
                * **OPERATIONAL FAILURE:** Code/Equipment error -> Fix implementation (Do not change science).
                * **SCIENTIFIC FAILURE:** Disproven hypothesis -> Pivot to new approach.

        4.  **Human-in-the-Loop (Dual-Phase):**
            -   **Phase A (Strategy):** Pauses after generating the new scientific plan to allow user critique (e.g., "Don't increase temp, safety limit is 50C").
            -   **Phase B (Code):** Pauses after generating the Python scripts. Writes them to a temp folder (`./temp_code_review_iter`) for inspection before finalization.

        Args:
            results (Any): The outcome of the previous experiment. 
                Supported formats:
                -   **String:** Natural language description (e.g., "Yield was 5%").
                -   **Dict/List:** Structured data (e.g., `{"yield": 0.05, "error": None}`).
                -   **File Path (str):** Path to a local file (.xlsx, .csv, .txt, .png, .jpg).
                -   **Structured List:** A list containing a mix of the above, or dictionaries with metadata 
                    (e.g., `[{"path": "./plot.png", "description": "Graph showing thermal runaway"}]`).
            output_json_path (Optional[str]): If provided, saves the updated plan JSON to this path.
                The full state is also saved to `{output_json_path}.state.json`.
            enable_human_feedback (bool): If True, pauses execution for console input at the 
                Strategy and Code review stages. Defaults to True.
            state_file_path: Optional path to .state.json file.
                If provided, restores agent state before processing results.
                Equivalent to calling restore_state() first.

        Returns:
            Dict[str, Any]: Updated state dictionary containing:
                - current_plan: Latest experimental plan
                - plan_history: All historical plans
                - experimental_results: All results received
                - iteration_index: Current iteration number
        """

        # --- 0. STATE RESTORATION ---

        if state_file_path is not None:
            print(f"\n--- 🔄 Restoring State from File ---")
            self.restore_state(state_file_path)

        if not self.state or not self.state.get("current_plan"):
            raise ValueError(
                "No active state found.\n"
                "You must initialize the agent first using one of:\n"
                "  1. agent.propose_experiments(...) - Start new session\n"
                "  2. agent.restore_state('path.state.json') - Restore saved session\n"
                "  3. Pass state_file_path='path.state.json' to this method"
            )
        
        print(f"\n--- 🔄 Iterating Plan based on New Results ---")
        executed_plan_idx = self.state["iteration_index"]
        
        # Extract from state
        objective = self.state["objective"]
        current_plan = self.state["current_plan"]
        
        # --- 1. SMART RESULT PARSING ---
        parsed_text_results = []
        loaded_images = []
        
        # Helper to process a single item (path or text)
        def process_item(item: Any, description: str = "") -> str:
            text_output = ""
            
            # If it's a file path
            if isinstance(item, str) and (Path(item).exists()):
                path = Path(item)
                suffix = path.suffix.lower()
                
                # A. Data Files
                if suffix in ['.xlsx', '.xls', '.csv']:
                    print(f"  - 📄 Parsing data file: {path.name}")
                    try:
                        chunks = parse_adaptive_excel(str(path), context_path="")
                        if chunks:
                            summary = chunks[0]['text']
                            text_output = f"DATA FILE ({path.name}):\n{summary}"
                    except Exception as e:
                        text_output = f"[Error parsing {path.name}: {e}]"

                # B. Images
                elif suffix in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                    print(f"  - 🖼️  Loading result image: {path.name}")
                    try:
                        with PIL_Image.open(path) as img:
                            img.load()  
                            loaded_images.append(img.copy())
                        text_output = f"[Attached Image: {path.name}]"
                    except Exception as e:
                        text_output = f"[Error loading image {path.name}: {e}]"
                
                # C. Logs/Text
                elif suffix in ['.txt', '.log', '.md', '.json']:
                    try:
                        content = path.read_text(encoding='utf-8')
                        text_output = f"LOG FILE ({path.name}):\n{content}"
                    except Exception as e:
                        text_output = f"[Error reading log {path.name}: {e}]"
                
                else:
                    text_output = f"FILE ({path.name})"

            # If not a file, treat as raw text/data
            else:
                if isinstance(item, (dict, list)):
                    text_output = json.dumps(item, indent=2)
                else:
                    text_output = str(item)
            
            # Append description if provided
            if description:
                text_output += f"\n(Context: {description})"
            
            return text_output

        # Recursive Parser to handle Lists and Dictionaries
        items_to_process = results if isinstance(results, list) else [results]
        
        for entry in items_to_process:
            if isinstance(entry, dict):
                # Check for common keys indicating a file + desc structure
                path_val = entry.get('path') or entry.get('file') or entry.get('image')
                desc_val = entry.get('description') or entry.get('desc') or entry.get('caption') or entry.get('notes')
                
                if path_val and isinstance(path_val, str):
                    # It's a structured file entry
                    parsed_text_results.append(process_item(path_val, desc_val if desc_val else ""))
                else:
                    # It's just a data dictionary
                    parsed_text_results.append(json.dumps(entry, indent=2))
            else:
                # It's a direct item (string, number, or path string)
                parsed_text_results.append(process_item(entry))

        # Join all text findings
        consolidated_feedback = "\n\n".join(parsed_text_results)

        # Update State History
        self.state["experimental_results"].append({
            "iteration": executed_plan_idx,
            "timestamp": datetime.now().isoformat(),
            "data_summary": str(results) # Keep reference to raw input
        })
        self.state["iteration_index"] += 1 
        next_plan_idx = self.state["iteration_index"]
        
        # --- 2. Construct Feedback Prompt ---
        feedback_prompt = (
            f"We executed the previous plan. Here are the experimental results:\n"
            f"{consolidated_feedback}\n\n"
            f"**TASK:** Analyze these results (including any attached plots) to Refine or Update the plan.\n"
            f"Select the most appropriate strategy:\n"
            f"1. **CONFIRMED:** If hypothesis is validated, propose next step.\n"
            f"2. **OPTIMIZATION NEEDED:** If result is valid but sub-optimal, tune parameters.\n"
            f"3. **INCONCLUSIVE:** If data is noisy, propose refined experiment.\n"
            f"4. **OPERATIONAL FAILURE:** If failure was code/equipment, propose fix.\n"
            f"5. **SCIENTIFIC FAILURE:** If hypothesis is disproven, propose new approach.\n"
        )
        
        # --- 3. RESULT-AWARE RAG ---
        new_literature_context = None
        if self.kb_docs.index and self.kb_docs.index.ntotal > 0:
            search_query = f"Implications and causes of: {consolidated_feedback[:400]}"
            print(f"  - 🔍 Searching literature for context on results...")
            hits = self.kb_docs.retrieve(search_query, top_k=3)
            if hits:
                new_literature_context = "\n---\n".join([c['text'] for c in hits])
                print(f"    -> Found {len(hits)} relevant document chunks.")
        
        # --- 4. Generate Refined Plan ---
        print(f"  - Reasoning over results with literature context...")
        objective = self.state["objective"]
        current_plan = self.state["current_plan"]
        
        new_plan = refine_plan_with_feedback(
            original_result=current_plan,
            feedback=feedback_prompt,
            objective=objective,
            model=self.model,
            generation_config=self.generation_config,
            new_context=new_literature_context,
            result_images=loaded_images
        )
        
        # SNAPSHOT: REASONING DRAFT
        new_plan["iteration"] = next_plan_idx
        new_plan["stage"] = "Reasoning Draft"
        self.state["plan_history"].append(new_plan.copy())
        self.state["current_plan"] = new_plan

        # =====================================================
        # 5. HUMAN STRATEGY FEEDBACK
        # =====================================================
        if enable_human_feedback and not new_plan.get("error"):
            print("\n" + "="*60)
            print("🧠 AGENT'S PROPOSED REVISION BASED ON RESULTS")
            print("="*60)
            display_plan_summary(new_plan)
            
            user_feedback = get_user_feedback()
            
            if user_feedback:
                print(f"\n📝 Feedback received. Adjusting strategy...")
                self.state["human_feedback_history"].append({"phase": "science_iteration", "feedback": user_feedback})
                new_plan = refine_plan_with_feedback(
                    original_result=new_plan,
                    feedback=user_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                # SNAPSHOT: HUMAN REFINED
                new_plan["iteration"] = next_plan_idx
                new_plan["stage"] = "Human Refined (Science)"
                self.state["plan_history"].append(new_plan.copy())
                self.state["current_plan"] = new_plan
                print("✅ Strategic revision updated.")

        # =====================================================
        # 6. Generate Code
        # =====================================================
        if self.kb_code.index and self.kb_code.index.ntotal > 0 and not new_plan.get("error"):
             
            # Extract previous implementations
            previous_implementations = []
            if current_plan and "proposed_experiments" in current_plan:                
                for exp in current_plan["proposed_experiments"]:
                    if "implementation_code" in exp:
                        previous_implementations.append({
                            'experiment_name': exp.get('experiment_name', 'Unnamed'),
                            'code': exp['implementation_code'],
                            'iteration': executed_plan_idx,
                            'source_files': exp.get('code_source_files', []),
                            'previous_steps': exp.get('experimental_steps', [])
                        })
            
            print(f"\n--- Code Implementation Analysis ---")
            if previous_implementations:
                print(f"  - Context: {len(previous_implementations)} existing implementation(s)")
            else:
                print(f"  - Context: Writing from scratch (no previous code)")
            
            new_plan = perform_code_rag(
                 result=new_plan,
                 kb_code=self.kb_code,
                 model=self.model,
                 generation_config=self.generation_config,
                 previous_implementations=previous_implementations
             )
            
             # SNAPSHOT: CODE GENERATED
            new_plan["iteration"] = next_plan_idx
            new_plan["stage"] = "Code Generated"
            self.state["plan_history"].append(new_plan.copy())
            self.state["current_plan"] = new_plan

        # =====================================================
        # 7. HUMAN CODE REVIEW
        # =====================================================
        if enable_human_feedback and not new_plan.get("error"):
            temp_dir = Path("./temp_code_review_iter")
            print(f"\n--- Human Code Review (Iteration {next_plan_idx}) ---")
            
            if temp_dir.exists(): shutil.rmtree(temp_dir)
            files = write_experiments_to_disk(new_plan, str(temp_dir))
            
            if files:
                while True:
                    print("\n" + "="*60)
                    print(f"👀 ACTION REQUIRED: Code Review")
                    print("="*60)
                    print(f"1. Open folder: {temp_dir.resolve()}")
                    print(f"2. Inspect the {len(files)} new Python file(s).")
                    print("3. Return here to Approve or Request Changes.")
                    
                    code_feedback = get_user_feedback()
                    
                    if not code_feedback:
                        print("✅ Code accepted.")
                        break
                    
                    self.state["human_feedback_history"].append({"phase": "code_iteration", "feedback": code_feedback})
                    print(f"\n🛠️  Refining code based on: '{code_feedback}'...")
                    
                    new_plan = refine_code_with_feedback(
                        result=new_plan,
                        feedback=code_feedback,
                        model=self.model,
                        generation_config=self.generation_config
                    )
                    
                    # SNAPSHOT: CODE REFINED
                    new_plan["iteration"] = next_plan_idx
                    new_plan["stage"] = "Code Refined"
                    self.state["plan_history"].append(new_plan.copy())
                    self.state["current_plan"] = new_plan
                    
                    print(f"  - 💾 Overwriting files in {temp_dir} with refined code...")
                    files = write_experiments_to_disk(new_plan, str(temp_dir))

        # 8. Commit to State & Save
        self.state["current_plan"] = new_plan
        # (Already appended snapshots above, so no final append needed unless we want a 'Final' tag)
        self.state["status"] = "iterated"
        
        final_out = "./output_scripts"
        print(f"\n--- Saving Final Scripts to: {final_out} ---")
        write_experiments_to_disk(new_plan, final_out)
        
        if output_json_path:
            self._save_results_to_json(new_plan, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            
            # TRIGGER HTML REPORT GENERATION
            self._generate_html_report(output_json_path)
            
        return self.state

    def perform_technoeconomic_analysis(self, objective: str,
                                        science_paths: Optional[List[str]] = None,
                                        code_paths: Optional[List[str]] = None, 
                                        structured_data_sets: Optional[List[Dict[str, str]]] = None,
                                        primary_data_set: Optional[Dict[str, str]] = None,
                                        image_paths: Optional[List[str]] = None,
                                        image_descriptions: Optional[List[str]] = None,
                                        output_json_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs TEA using Dual-KB retrieval. 

        **Workflow:**
        
        1. Knowledge Base Construction (if needed)
        2. External Literature Search (optional, via FutureHouse)
        3. RAG-based Economic Analysis
        4. State Initialization (if starting fresh with TEA)
        5. Report Generation (JSON + HTML)

        **Integration with Planning:**
    
        TEA results are stored in the agent's state and can inform subsequent
        experimental planning:
            >>> # Perform TEA first
            >>> tea_results = agent.perform_technoeconomic_analysis(
            ...     objective="Recover lithium from brine",
            ...     science_paths=["./market_data/", "./reports/"],
            ... )
            >>> 
            >>> # Use TEA insights in experimental planning
            >>> plan = agent.propose_experiments(
            ...             objective="Develop lithium extraction process",
            ...             science_paths=["./extraction_methods/"],
            ...             additional_context=tea_results,
            ...             primary_data_set={
            ...                "file_path": "./brine_composition.xlsx",
            ...                "metadata_path": ./metadata.json}
            ... )
        Args:
        objective (str): Research objective to evaluate economically.
            Should describe the material, process, or technology to assess.
            Examples:
                - "Recover rare earth elements from coal ash"
                - "Evaluate magnesium extraction from produced water"
                - "Assess economic viability of direct air capture"
        
        science_paths (Optional[List[str]]): Paths to documents for TEA context.
            Should include market data, pricing reports, criticality assessments,
            existing TEA studies, and process descriptions.
            Examples: ["./market_reports/", "./critical_materials_report.pdf"]
        
        code_paths (Optional[List[str]]): Paths to code (typically unused for TEA).
            Included for consistency with propose_experiments API.
            TEA rarely requires code generation.
        
        structured_data_sets (Optional[List[Dict[str, str]]]): Excel/CSV datasets
            containing economic data (prices, concentrations, yields, etc.).
            Example: [{"file_path": "./commodity_prices.xlsx"}]
        
        primary_data_set (Optional[Dict[str, str]]): Main dataset for analysis.
            Typically contains composition, concentration, or yield data.
            Example: {"file_path": "./feedstock_composition.xlsx"}
        
        image_paths (Optional[List[str]]): Images to support TEA analysis.
            Examples: criticality matrices, supply chain diagrams, cost breakdowns.
        
        image_descriptions (Optional[List[str]]): Descriptions for each image.
            Example: ["Criticality matrix showing supply risk vs. importance"]
        
        output_json_path (Optional[str]): Path to save TEA results.
            Saves to {output_json_path} (results only)
            Saves to {output_json_path}.state.json (full state)
            Generates {output_json_path}.html (formatted report)
    
    Returns:
        Dict[str, Any]: Technoeconomic analysis results  

    Example - Basic Usage:
        >>> agent = PlanningAgent()
        >>> state = agent.propose_experiments(
        ...     objective="Optimize enzyme kinetics",
        ...     science_paths=["./enzyme_papers/"],
        ...     code_paths=["./plate_reader_api/"],
        ...     output_json_path="./plan.json"
        ... )
        >>> # User reviews in console, provides feedback or approves
        >>> # Final scripts saved to ./output_scripts/

    Example - Advanced with Data:
        >>> state = agent.propose_experiments(
        ...     objective="Identify optimal precipitation conditions",
        ...     science_paths=["./papers/", "./protocols.pdf"],
        ...     code_paths=["https://github.com/opentrons/opentrons"],
        ...     primary_data_set={
        ...         "file_path": "./icpms_results.xlsx",
        ...         "metadata_path": "./icpms_metadata.json"
        ...     },
        ...     image_paths=["./criticality_matrix.jpg"],
        ...     image_descriptions=["Material criticality assessment"],
        ...     additional_context={
        ...         "Constraints": "Use only commodity chemicals",
        ...         "Equipment": "Opentrons OT-2, 96-well plates, ICP-MS"
        ...     },
        ...     output_json_path="./precipitation_plan.json",
        ...     enable_human_feedback=True
        ... )
    """
        
        # 1. State Initialization (if starting fresh with TEA)
        if not self.state:
            self.state = self._initialize_state(
                objective=objective,
                science_paths=science_paths,
                code_paths=code_paths,
                primary_data_set=primary_data_set,
                image_paths=image_paths,
                image_descriptions=image_descriptions
            )

        #  TEA is always step 0 (pre-planning)
        self.state["iteration_index"] = 0

        # 2. Build KB if needed
        if not self._ensure_kb_is_ready(science_paths, code_paths, structured_data_sets):
            return {"error": "KB Init Failed"}
        
        # 3. Literature Search
        lit_context = ""
        if self.lit_agent:
            print(f"  - 🌍 Querying literature for TEA context...")
            lit_res = self.lit_agent.search_for_economic_data(
                optimize_search_query(objective=objective, model=self.model)
            )
            if lit_res['status'] == 'success':
                lit_context = lit_res['content']

        # 4. Perform RAG
        res = perform_science_rag(
            objective=objective, 
            instructions=TEA_INSTRUCTIONS, 
            task_name="Technoeconomic Analysis",
            kb_docs=self.kb_docs,
            model=self.model,
            generation_config=self.generation_config,
            primary_data_set=primary_data_set, 
            image_paths=image_paths, 
            image_descriptions=image_descriptions,
            external_context=lit_context
        )

        if lit_context:
            res["literature_search"] = lit_context

        # 5. Commit to State
        if not res.get("error"):
            # Tags for the HTML Generator
            res["type"] = "technoeconomic_analysis"
            res["stage"] = "TEA Initial"
            res["iteration"] = 0 # TEA is step 0 (pre-planning)
            
            # Append copy to history (Full Traceability)
            self.state["plan_history"].append(res.copy())
            
            # Update Active Pointer
            self.state["current_plan"] = res

        # 6. Save & Generate Report
        if output_json_path:
            self._save_results_to_json(res, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            
            # Trigger HTML Generation (will show TEA card)
            self._generate_html_report(output_json_path)

        return res