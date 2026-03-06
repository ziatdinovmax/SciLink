import os
from pathlib import Path
import json
import logging
import shutil
import uuid
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from .knowledge_base import KnowledgeBase
from .parser_utils import (
    generate_repo_map, 
    write_experiments_to_disk,
    resolve_primary_data_path,
    parse_multimodal_results
)
from .repo_loader import clone_git_repository

from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    TEA_INSTRUCTIONS
)

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from ._deprecation import normalize_params
from .base_agent import BaseAgent
from .knowledge_base import KnowledgeBase

from ..lit_agents.literature_agent import LiteratureSearchAgent
from ..lit_agents.optimize_query import optimize_search_query

from .rag_engine import (
    perform_science_rag, 
    perform_code_rag, 
    refine_plan_with_feedback,
    refine_code_with_feedback,
    verify_plan_relevance
)

from .ingestor import ingest_files, extract_images

from .user_interface import display_plan_summary, get_user_feedback

from .html_generator import HTMLReportGenerator

from .base_agent import BaseAgent



class PlanningAgent(BaseAgent):
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
        api_key: API key for the LLM provider.
        model_name: Model name. For public deployments, use LiteLLM format
            (e.g., "gemini/gemini-2.0-flash", "gpt-4o", "claude-sonnet-4-20250514").
        base_url: Base URL for internal proxy endpoint.
            When provided, uses OpenAI-compatible client.
            When None, uses LiteLLM for multi-provider support.
        embedding_model: Embedding model name.
        embedding_api_key: API key for the embedding LLM provider.
        futurehouse_api_key: Optional API key for literature search.
        kb_base_path: Path for knowledge base storage.
        code_chunk_size: Chunk size for code files.
        output_dir: Output directory for artifacts.
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(self, api_key: str = None,
                 model_name: str = "gemini-3.1-pro-preview",
                 base_url: Optional[str] = None,
                 embedding_model: str = "gemini-embedding-001",
                 embedding_api_key: Optional[str] = None,
                 futurehouse_api_key: str = None,
                 kb_base_path: str = "./kb_storage/default_kb",
                 code_chunk_size: int = 20000,
                 output_dir: str = ".",
                 google_api_key: Optional[str] = None,
                 local_model: str = None,): 
        
        super().__init__(output_dir)
        self.agent_type = "planning"

        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="PlanningAgent"
        )
        
        # Store config
        self._base_url = base_url
        self.code_chunk_size = code_chunk_size
        
        # Initialize LLM client based on deployment mode
        use_litellm = False
        
        if base_url:
            # INTERNAL PROXY
            if api_key is None:
                api_key = get_internal_proxy_key()
            
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy.\n"
                    "Set SCILINK_API_KEY environment variable or pass api_key parameter."
                )
            
            if embedding_api_key is not None:
                logging.warning(
                    "⚠️ embedding_api_key is ignored for internal proxy. "
                    "Using api_key for all requests."
                )
            
            logging.info(f"🏛️ PlanningAgent using internal proxy: {base_url}")
            self.model = OpenAIAsGenerativeModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
            use_litellm = False
            embedding_api_key = api_key
            
        else:
            # PUBLIC LITELLM - can use different keys per provider
            logging.info(f"🌐 PlanningAgent using LiteLLM: {model_name}")
            self.model = LiteLLMGenerativeModel(
                model=model_name,
                api_key=api_key  # Can be None - LiteLLM reads env vars
            )
            use_litellm = True
            # embedding_api_key stays as passed (can be None for auto-detect)
        
        self._api_key = api_key
        self.generation_config = None

        self.lit_agent = None
        if futurehouse_api_key or os.getenv("FUTUREHOUSE_API_KEY"):
            try:
                self.lit_agent = LiteratureSearchAgent(futurehouse_api_key, max_wait_time=1000)
                logging.info("✅ Literature Search Agent initialized.")
            except Exception as e:
                logging.warning(f"⚠️ Failed to initialize Literature Agent: {e}")
        else:
            logging.info("ℹ️ No FutureHouse API key provided. Literature search will be skipped.")
                    
        # --- Dual KnowledgeBase Initialization ---
        base_path = Path(kb_base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Scientific/Docs KB
        self.kb_docs = KnowledgeBase(
            api_key=embedding_api_key,
            embedding_model=embedding_model,
            base_url=base_url,
            use_litellm=use_litellm
        )
        self.kb_docs_prefix = base_path.parent / f"{base_path.name}_docs"
        self.kb_docs_index = str(self.kb_docs_prefix.with_suffix(".faiss"))
        self.kb_docs_chunks = str(self.kb_docs_prefix.with_suffix(".json"))
        self.kb_docs_sources_path = str(self.kb_docs_prefix.with_suffix(".sources.json"))

        # 2. Implementation/Code KB
        self.kb_code = KnowledgeBase(
            api_key=embedding_api_key,
            embedding_model=embedding_model,
            base_url=base_url,
            use_litellm=use_litellm
        )
        self.kb_code_prefix = base_path.parent / f"{base_path.name}_code"
        self.kb_code_index = str(self.kb_code_prefix.with_suffix(".faiss"))
        self.kb_code_chunks = str(self.kb_code_prefix.with_suffix(".json"))
        self.kb_code_map_path = str(self.kb_code_prefix.with_suffix(".maps.json"))
        self.kb_code_sources_path = str(self.kb_code_prefix.with_suffix(".sources.json"))

        print("--- Initializing Agent (Dual-KB System) ---")
        self._load_knowledge_bases()

    def _get_initial_state_fields(self) -> Dict[str, Any]:
        """Agent-specific state fields"""
        return {
            "objective": None,
            "iteration_index": 0,
            "inputs": {
                "knowledge_paths": [],
                "code_paths": [],
                "additional_context": None,
                "primary_data_set": None,
                "image_paths": [],
                "image_descriptions": []
            },
            "current_plan": None,
            "plan_history": [],
            "experimental_results": [],
            "human_feedback_history": [],
            "last_error": None
        }

    def restore_state(self, state_file_path: str) -> None:
        """
        Restore agent state from a saved .state.json file.
        Raises FileNotFoundError if file doesn't exist.
        """        
        path = Path(state_file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {state_file_path}")
        
        if path.suffix != '.json':
            raise ValueError(f"State file must be a .json file, got: {path.suffix}")
        
        print(f"  - 📂 Loading state from: {path.name}")
        
        if not self.load_state(state_file_path):  # Uses inherited method
            raise ValueError(f"Failed to parse state file: {state_file_path}")
        
        # User feedback
        print(f"  - ✅ Restored session: {self.state['session_id']}")
        print(f"     • Objective: {self.state['objective'][:80]}...")
        print(f"     • Current iteration: {self.state['iteration_index']}")
        print(f"     • History entries: {len(self.state.get('plan_history', []))}")
        print(f"     • Previous results: {len(self.state.get('experimental_results', []))}")
        print(f"     • Actions logged: {len(self.state.get('action_history', []))}")

        
    def _load_knowledge_bases(self):
        """Attempts to load both KBs from disk."""
        print(f"  - Docs KB: Loading from {self.kb_docs_prefix}...")
        docs_loaded = self.kb_docs.load(
            self.kb_docs_index, self.kb_docs_chunks,
            sources_path=self.kb_docs_sources_path
        )
        
        print(f"  - Code KB: Loading from {self.kb_code_prefix}...")
        code_loaded = self.kb_code.load(
            self.kb_code_index, self.kb_code_chunks, self.kb_code_map_path,
            sources_path=self.kb_code_sources_path
        )

        self._kb_is_built = docs_loaded or code_loaded
        
        if docs_loaded: print("    - ✅ Docs KB loaded.")
        if code_loaded: print("    - ✅ Code KB loaded.")
        if not self._kb_is_built: print("    - ⚠️  No pre-built KBs found.")

    def _initialize_state(self, objective: str, **kwargs) -> Dict[str, Any]:
        """Creates the foundational state dictionary for a new research task."""
        self._init_state(
            objective=objective,
            inputs={
                "knowledge_paths": kwargs.get("knowledge_paths", []),
                "code_paths": kwargs.get("code_paths", []),
                "additional_context": kwargs.get("additional_context"),
                "primary_data_set": kwargs.get("primary_data_set"),
                "image_paths": kwargs.get("image_paths", []),
                "image_descriptions": kwargs.get("image_descriptions", [])
            }
        )
        return self.state
    
    def _save_results_to_json(self, results: Dict[str, Any], file_path: str):
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f: json.dump(results, f, indent=2)
            print(f"    - ✅ Results successfully saved to: {file_path}")
        except Exception as e: logging.error(f"    - ❌ Failed to save results: {e}")

    def _save_state_to_json(self, file_path: str):
        """Saves state to a specific path (legacy interface)."""
        try:
            p = Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w', encoding='utf-8') as f: 
                json.dump(self.state, f, indent=2)
        except Exception as e: 
            logging.error(f"Failed to save state: {e}")

    def _build_and_save_kb(self, knowledge_paths: Optional[List[str]] = None, code_paths: Optional[List[str]] = None) -> bool:
        print("\n--- Rebuilding Knowledge Bases ---")
        
        # 1. Science KB
        doc_chunks = []
        if knowledge_paths:
            print(f"Processing {len(knowledge_paths)} Scientific Paths...")
            doc_chunks.extend(ingest_files(knowledge_paths, is_code_mode=False))

        if doc_chunks:
            print(f"  - Building Scientific KB with {len(doc_chunks)} chunks...")
            self.kb_docs.build(doc_chunks)
            self.kb_docs.save(self.kb_docs_index, self.kb_docs_chunks, sources_path=self.kb_docs_sources_path)
        else:
            print("  - ℹ️  No Scientific docs provided. Docs KB unchanged.")

        # 2. Code KB
        code_chunks = []
        if code_paths:
            print(f"Processing {len(code_paths)} Code Paths...")
            for p in code_paths:
                path_obj = Path(p)
                if path_obj.is_dir():
                    repo_name = path_obj.name
                    print(f"  - 📦 Processing Repo: {repo_name}")
                    self.kb_code.repo_maps[repo_name] = generate_repo_map(str(path_obj))
                    code_chunks.extend(ingest_files([p], is_code_mode=True, code_chunk_size=self.code_chunk_size, repo_name=repo_name))
                else:
                    code_chunks.extend(ingest_files([p], is_code_mode=True, code_chunk_size=self.code_chunk_size))
            
        if code_chunks:
            print(f"  - Building Code KB with {len(code_chunks)} chunks...")
            self.kb_code.build(code_chunks)
            self.kb_code.save(self.kb_code_index, self.kb_code_chunks, self.kb_code_map_path, self.kb_code_sources_path)
        else:
            print("  - ℹ️  No Code docs provided. Code KB unchanged.")

        self._kb_is_built = True
        return True

    def _ensure_kb_is_ready(self, knowledge_paths: Optional[List[str]] = None, code_paths: Optional[List[str]] = None) -> bool:
        new_science = self.kb_docs.source_difference(knowledge_paths)
        new_code = self.kb_code.source_difference(code_paths)
        
        if new_science or new_code:
            return self._build_and_save_kb(new_science, new_code)
        elif not self._kb_is_built:
            logging.error("Knowledge base is not built.")
            return False
        return True
    
    def generate_plan(self,
                    objective: str,
                    knowledge_paths: Optional[List[str]] = None,
                    primary_data_set: Optional[Union[str, Dict[str, str]]] = None,
                    additional_context: Optional[Dict[str, str]] = None,
                    image_paths: Optional[List[str]] = None,
                    image_descriptions: Optional[List[str]] = None,
                    enable_human_feedback: bool = True,
                    reset_state: bool = False) -> Dict[str, Any]:
        """
        Generate experimental plan (science only, no implementation code/protocol).
        
        This method performs:
        1. Knowledge base initialization (docs only)
        2. Literature search (optional)
        3. RAG-based hypothesis generation
        4. Self-correction loop
        5. Human feedback on strategy
        
        Does NOT generate implementation code. Use generate_implementation_code() for that.
        
        Returns:
            Dict with proposed_experiments
        """
        
        # Resolve data and images
        primary_data_set = resolve_primary_data_path(primary_data_set)
        manual_images = image_paths or []
        auto_images = [img for img in extract_images(knowledge_paths) if img not in manual_images]
        all_image_paths = manual_images + auto_images
        
        # Initialize or update state
        if reset_state or not self.state:
            self.state = self._initialize_state(
                objective=objective,
                knowledge_paths=knowledge_paths,
                code_paths=None,  # ← Not used in plan generation
                additional_context=additional_context,
                primary_data_set=primary_data_set,
                image_paths=all_image_paths,
                image_descriptions=image_descriptions
            )
        else:
            print(f"  - 🔄 Appending to existing research session...")
            if objective:
                self.state["objective"] = objective
        
        # Increment iteration
        existing_iter = self.state.get("iteration_index", 0)
        self.state["iteration_index"] = existing_iter + 1
        current_iter = self.state["iteration_index"]
        
        # Build KB (docs only)
        if not self._ensure_kb_is_ready(knowledge_paths, code_paths=None):
            self.state["status"] = "failed"
            self.state["last_error"] = "KB Init Failed"
            self._log_action(
                action="generate_plan",
                input_ctx={"objective": objective},
                result={"status": "failed", "error": "KB Init Failed"},
                rationale=None
            )
            return self.state
        
        # Build context string
        ctx_string = ""
        if additional_context:
            for header, content in additional_context.items():
                ctx_string += f"## {header}\n{content}\n\n"
            ctx_string = ctx_string.strip() if ctx_string else None
        
        # Literature search
        lit_context = ""
        if self.lit_agent:
            print(f"  - 🌍 Querying literature...")
            lit_res = self.lit_agent.search_for_hypothesis_context(
                optimize_search_query(objective=objective, model=self.model)
            )
            if lit_res['status'] == 'success':
                lit_context = lit_res['content']
        
        # RAG for science plan
        print(f"\n--- Generating Experimental Strategy ---")
        res = perform_science_rag(
            objective=objective,
            instructions=HYPOTHESIS_GENERATION_INSTRUCTIONS,
            task_name="Experimental Plan",
            kb_docs=self.kb_docs,
            model=self.model,
            generation_config=self.generation_config,
            primary_data_set=primary_data_set,
            image_paths=all_image_paths,
            image_descriptions=image_descriptions,
            additional_context=ctx_string,
            external_context=lit_context
        )
        
        if lit_context:
            res["literature_search"] = lit_context

        self._log_action(
            action="perform_science_rag",
            input_ctx={
                "objective": objective,
                "knowledge_paths": knowledge_paths,
                "has_primary_data": primary_data_set is not None,
                "has_literature": bool(lit_context)
            },
            result=res,
            rationale=res.get("proposed_experiments", [{}])[0].get("justification") if res.get("proposed_experiments") else None
        )
        
        # Snapshot 1: Science Draft
        res["iteration"] = current_iter
        res["stage"] = "Science Draft"
        self.state["plan_history"].append(res.copy())
        self.state["current_plan"] = res
        
        # Self-correction
        if not res.get("error"):
            is_relevant, critique = verify_plan_relevance(objective, res, self.model, self.generation_config)
            
            if not is_relevant:
                print(f"\n🔄 Self-correction triggered: {critique}")
                res = refine_plan_with_feedback(
                    original_result=res,
                    feedback=f"CRITICAL: {critique}",
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                
                res["iteration"] = current_iter
                res["stage"] = "Auto-Corrected"
                self.state["plan_history"].append(res.copy())
                self.state["current_plan"] = res

                self._log_action(
                    action="self_correction",
                    input_ctx={"critique": critique},
                    result=res,
                    rationale=f"Auto-corrected due to: {critique}"
                )
        
        # Human feedback on strategy
        human_feedback = None
        if enable_human_feedback and res.get("proposed_experiments") and not res.get("error"):
            display_plan_summary(res)
            human_feedback = get_user_feedback()
            
            if human_feedback:
                print(f"\n📝 Refining plan...")
                self.state["human_feedback_history"].append({"phase": "science", "feedback": human_feedback})
                refined = refine_plan_with_feedback(
                    original_result=res,
                    feedback=human_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )

                if refined.get("error"):
                    print(f"⚠️  Refinement failed: {refined.get('message', 'unknown error')}")
                    print("    Keeping original plan.")
                else:
                    res = refined
                    res["iteration"] = current_iter
                    res["stage"] = "Human Refined (Science)"
                    self.state["plan_history"].append(res.copy())
                    self.state["current_plan"] = res
                    display_plan_summary(res)
                    print("✅ Plan updated.")
            else:
                print("✅ Plan accepted.")
        
        self._log_action(
                action="generate_plan",
                input_ctx={
                    "objective": objective,
                    "iteration": current_iter
                },
                result=res,
                rationale=res.get("proposed_experiments", [{}])[0].get("justification") if res.get("proposed_experiments") else None,
                feedback=human_feedback
        )

        self.state["status"] = "planned"
        
        return res
    
    def generate_implementation_code(self,
                                    plan: Dict[str, Any],
                                    code_paths: List[str],
                                    enable_human_feedback: bool = True) -> Dict[str, Any]:
        """
        Add implementation code to an existing experimental plan.
        
        This method:
        1. Builds code knowledge base
        2. Performs code RAG to map experiments to APIs
        3. Provides human code review
        
        Args:
            plan: Existing plan dict (must have proposed_experiments)
            code_paths: Paths to code/API repositories
            enable_human_feedback: If True, pauses for code review
        
        Returns:
            Updated plan dict with implementation_code added to experiments
        """
        
        # Resolve code paths (handle Git URLs)
        print("\n--- Resolving Code Paths ---")
        effective_code_paths = []
        for path in code_paths:
            if path.strip().startswith(('http://', 'https://', 'git@')):
                print(f"  - 🔗 Cloning: {path}")
                local_path = clone_git_repository(path)
                if local_path:
                    effective_code_paths.append(local_path)
            else:
                effective_code_paths.append(path)
        
        # Build code KB
        if not self._ensure_kb_is_ready(knowledge_paths=None, code_paths=effective_code_paths):
            error_result = {"error": "Code KB build failed"}
            self._log_action(
                action="generate_implementation_code",
                input_ctx={"code_paths": code_paths},
                result=error_result,
                rationale=None
            )

            return error_result
        
        # Check if code KB has content
        if not (self.kb_code.index and self.kb_code.index.ntotal > 0):
            print("  - ⚠️  Code KB is empty, skipping code generation")
            self._log_action(
                action="generate_implementation_code",
                input_ctx={"code_paths": code_paths},
                result={"status": "skipped", "error": "Empty Code KB"},
                rationale="No code documents found in knowledge base"
            )
            return plan
        
        # Generate code
        print(f"\n--- Generating Implementation Code ---")
        current_iter = plan.get("iteration", self.state.get("iteration_index", 1))
        
        res = perform_code_rag(
            result=plan,
            kb_code=self.kb_code,
            model=self.model,
            generation_config=self.generation_config
        )
        
        # Snapshot: Code Generated
        res["iteration"] = current_iter
        res["stage"] = "Code Generated"
        self.state["plan_history"].append(res.copy())
        self.state["current_plan"] = res

        self._log_action(
            action="perform_code_rag",
            input_ctx={
                "code_paths": effective_code_paths,
                "num_experiments": len(res.get("proposed_experiments", []))
            },
            result=res,
            rationale="Mapped experimental steps to API code"
        )
        
        human_feedback = None

        # Human code review
        if enable_human_feedback:
            temp_dir = self.output_dir / "temp_code_review"
            print(f"\n--- Code Review ---")
            print(f"  - 💾 Saving to: {temp_dir}")
            
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            files = write_experiments_to_disk(res, str(temp_dir))
            
            if not files:
                print("  - ⚠️  No code generated")
            else:
                while True:
                    print("\n" + "="*60)
                    print(f"👀 CODE REVIEW REQUIRED")
                    print("="*60)
                    print(f"1. Review files in: {temp_dir.resolve()}")
                    print(f"2. Press ENTER to approve, or type feedback to refine")
                    print("-"*60)
                    
                    code_feedback = get_user_feedback()
                    
                    if not code_feedback:
                        print("✅ Code accepted")
                        break
                    
                    human_feedback = code_feedback
                    print(f"\n🛠️  Refining code...")
                    self.state["human_feedback_history"].append({"phase": "code", "feedback": code_feedback})
                    
                    res = refine_code_with_feedback(
                        result=res,
                        feedback=code_feedback,
                        model=self.model,
                        generation_config=self.generation_config
                    )
                    
                    res["iteration"] = current_iter
                    res["stage"] = "Code Refined"
                    self.state["plan_history"].append(res.copy())
                    self.state["current_plan"] = res

                    self._log_action(
                        action="refine_code",
                        input_ctx={"feedback": code_feedback},
                        result=res,
                        rationale=f"Human requested: {code_feedback}",
                        feedback=code_feedback
                    )
                    
                    print(f"  - 💾 Updating files...")
                    files = write_experiments_to_disk(res, str(temp_dir))
        
        self._log_action(
            action="generate_implementation_code",
            input_ctx={
                "code_paths": effective_code_paths,
                "iteration": current_iter
            },
            result=res,
            rationale="Code generation complete",
            feedback=human_feedback
        )
        return res

    def propose_experiments(self, objective: str, 
                            knowledge_paths: Optional[List[str]] = None, 
                            code_paths: Optional[List[str]] = None,
                            additional_context: Optional[Dict[str, str]] = None,
                            primary_data_set: Optional[Union[str, Dict[str, str]]] = None,
                            image_paths: Optional[List[str]] = None,
                            image_descriptions: Optional[List[str]] = None,
                            output_json_path: Optional[str] = None,
                            enable_human_feedback: bool = True,
                            reset_state: bool = False) -> Dict[str, Any]: # Default False to enable cumulative workflows
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
            
            knowledge_paths (Optional[List[str]]): Paths to scientific documents/data.
                Supported formats: PDFs, .txt, .md, .xlsx, .csv, directories.
                You can pass Excel/CSV files directly here. If a .json file 
                with the same name exists next to the data file, it is automatically 
                loaded as metadata.
                These populate the Docs Knowledge Base for hypothesis generation.
                Example: ["./papers/", "./lab_notebooks/protocol.pdf", "./public_data.xlsx", "./public_data.json" ]
            
            code_paths (Optional[List[str]]): Paths to code repositories or API documentation.
                Supported formats: Local directories, Git URLs, Python files
                These populate the Code Knowledge Base for implementation.
                Examples:
                    - ["./opentrons_api/"]  # Local repo
                    - ["https://github.com/org/automation-lib.git"]  # Git URL
            
            additional_context (Optional[Dict[str, str]]): Additional text context
                to inject into the prompt. Keys become section headers.
                Example: {
                    "Safety Constraints": "Maximum temperature is 80°C",
                    "Equipment Available": "Opentrons OT-2, plate reader"
                }
            
            primary_data_set (Optional[Dict[str, str]]): Main dataset to analyze.
                Use for the dataset that drives the research objective.
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
        # Phase 1: Generate experimental plan (science only)
        plan = self.generate_plan(
            objective=objective,
            knowledge_paths=knowledge_paths,
            primary_data_set=primary_data_set,
            additional_context=additional_context,
            image_paths=image_paths,
            image_descriptions=image_descriptions,
            enable_human_feedback=enable_human_feedback,
            reset_state=reset_state
        )
        
        if plan.get("error"):
            if output_json_path:
                self._save_results_to_json(plan, output_json_path)
            return self.state
        
        # Phase 2: Add implementation code (if code_paths provided)
        if code_paths:
            plan = self.generate_implementation_code(
                plan=plan,
                code_paths=code_paths,
                enable_human_feedback=enable_human_feedback
            )
        
        # Save final results
        if output_json_path:
            self._save_results_to_json(plan, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            self._generate_html_report(output_json_path)
        
        # Save scripts
        final_out = "./output_scripts"
        print(f"\n--- Saving Scripts to: {final_out} ---")
        write_experiments_to_disk(plan, final_out)
        
        return self.state
    
    def refine_plan(self,
                    results: Any,
                    enable_human_feedback: bool = True,
                    state_file_path: Optional[str] = None,
                    use_literature_rag: bool = False) -> Dict[str, Any]:
        """
        Refines the experimental plan (science strategy only) based on new results.
        
        Args:
            results: Experimental outcomes (text, dict, file path, or list of files/images)
            enable_human_feedback: If True, pauses for strategy review
            state_file_path: Optional path to restore state from checkpoint
            use_literature_rag: If True, searches knowledge base for context relevant 
                           to the results. Defaults to False for faster iteration.
            
        Returns:
            Dict with refined plan (proposed_experiments)
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
        
        print(f"\n--- 🔄 Refining Plan based on New Results ---")
        executed_plan_idx = self.state["iteration_index"]
        
        # Extract from state
        objective = self.state["objective"]
        current_plan = self.state["current_plan"]
        
        # --- 1. PARSE RESULTS (Use utility function) ---
        consolidated_feedback, loaded_images = parse_multimodal_results(results)
        
        # Update State History
        self.state["experimental_results"].append({
            "iteration": executed_plan_idx,
            "timestamp": datetime.now().isoformat(),
            "data_summary": str(results)
        })
        self.state["iteration_index"] += 1 
        next_plan_idx = self.state["iteration_index"]
        
        # --- 2. BUILD FEEDBACK PROMPT ---
        feedback_prompt = f"""We executed the previous plan. Here are the experimental results:
{consolidated_feedback}

**TASK:** Analyze these results (including any attached plots) to Refine or Update the plan.
Select the most appropriate strategy:
1. **CONFIRMED:** If hypothesis is validated, propose next step.
2. **OPTIMIZATION NEEDED:** If result is valid but sub-optimal, tune parameters.
3. **INCONCLUSIVE:** If data is noisy, propose refined experiment.
4. **OPERATIONAL FAILURE:** If failure was code/equipment, propose fix.
5. **SCIENTIFIC FAILURE:** If hypothesis is disproven, propose new approach.
"""
        
        # --- 3. RESULT-AWARE RAG ---
        new_literature_context = None
        
        if use_literature_rag:
            if self.kb_docs.index and self.kb_docs.index.ntotal > 0:
                search_query = f"Implications and causes of: {consolidated_feedback[:400]}"
                print(f"  - 🔍 Searching literature for context on results...")
                hits = self.kb_docs.retrieve(search_query, top_k=3)
                if hits:
                    new_literature_context = "\n---\n".join([c['text'] for c in hits])
                    print(f"    -> Found {len(hits)} relevant document chunks.")
                else:
                    print(f"    -> No relevant documents found.")
            else:
                print(f"  - ℹ️  Literature RAG requested but no docs KB available.")
        else:
            print(f"  - ℹ️  Skipping literature RAG (use_literature_rag=False)")
        
        # --- 4. GENERATE REFINED PLAN ---
        if new_literature_context:
            print(f"  - Reasoning over results with literature context...")
        else:
            print(f"  - Reasoning over results...")

        
        new_plan = refine_plan_with_feedback(
            original_result=current_plan,
            feedback=feedback_prompt,
            objective=objective,
            model=self.model,
            generation_config=self.generation_config,
            new_context=new_literature_context,
            result_images=loaded_images
        )

        if new_plan.get("error"):
            print(f"\n❌ Refinement Failed: {new_plan.get('message')}") 
            self._log_action(
                action="refine_plan",
                input_ctx={
                    "results_summary": consolidated_feedback[:200],
                    "use_literature_rag": use_literature_rag
                },
                result=new_plan,
                rationale=None
            )           
            return new_plan
        
        # Snapshot: Reasoning Draft
        new_plan["iteration"] = next_plan_idx
        new_plan["stage"] = "Reasoning Draft"
        self.state["plan_history"].append(new_plan.copy())
        self.state["current_plan"] = new_plan

        self._log_action(
            action="refine_plan_reasoning",
            input_ctx={
                "results_summary": consolidated_feedback[:200],
                "has_literature_context": new_literature_context is not None,
                "num_images": len(loaded_images)
            },
            result=new_plan,
            rationale=new_plan.get("proposed_experiments", [{}])[0].get("justification") if new_plan.get("proposed_experiments") else None
        )

        # --- 5. HUMAN STRATEGY FEEDBACK ---
        human_feedback = None
        if enable_human_feedback and not new_plan.get("error"):
            print("\n" + "="*60)
            print("🧠 AGENT'S PROPOSED REVISION BASED ON RESULTS")
            print("="*60)
            display_plan_summary(new_plan)
            
            human_feedback = get_user_feedback()
            
            if human_feedback: 
                print(f"\n📝 Feedback received. Adjusting strategy...")
                self.state["human_feedback_history"].append({
                    "phase": "science_iteration", 
                    "feedback": human_feedback
                })
                new_plan = refine_plan_with_feedback(
                    original_result=new_plan,
                    feedback=human_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config
                )
                # Snapshot: Human Refined
                new_plan["iteration"] = next_plan_idx
                new_plan["stage"] = "Human Refined (Science)"
                self.state["plan_history"].append(new_plan.copy())
                self.state["current_plan"] = new_plan
                print("✅ Strategic revision updated.")

        self._log_action(
            action="refine_plan",
            input_ctx={
                "iteration": next_plan_idx,
                "results_provided": True
            },
            result=new_plan,
            rationale=new_plan.get("proposed_experiments", [{}])[0].get("justification") if new_plan.get("proposed_experiments") else None,
            feedback=human_feedback
        )
        
        self.state["status"] = "refined"
        return new_plan
    
    def refine_implementation_code(self,
                                   plan: Dict[str, Any],
                                   enable_human_feedback: bool = True) -> Dict[str, Any]:
        """
        Updates implementation code for a refined plan.
        
        This is Step 2 of the iteration process - maps the refined experimental
        strategy to executable code using the Code KB.
        
        Args:
            plan: Refined plan from refine_plan() (must have proposed_experiments)
            enable_human_feedback: If True, pauses for code review
            
        Returns:
            Updated plan dict with implementation_code added/updated
        """
        
        if not self.kb_code.index or self.kb_code.index.ntotal == 0:
            print("  - ℹ️  No Code KB available, skipping implementation update")
            self._log_action(
                action="refine_implementation_code",
                input_ctx={},
                result={"status": "skipped", "error": "No Code KB"},
                rationale="Code knowledge base is empty"
            )
            return plan
        
        if plan.get("error"):
            return plan
        
        next_plan_idx = plan.get("iteration", self.state.get("iteration_index", 1))
        
        # Extract previous implementations from current state
        current_plan = self.state.get("current_plan", {})
        previous_implementations = []
        
        if current_plan and "proposed_experiments" in current_plan:                
            for exp in current_plan["proposed_experiments"]:
                if "implementation_code" in exp:
                    previous_implementations.append({
                        'experiment_name': exp.get('experiment_name', 'Unnamed'),
                        'code': exp['implementation_code'],
                        'iteration': self.state.get("iteration_index", 0) - 1,
                        'source_files': exp.get('code_source_files', []),
                        'previous_steps': exp.get('experimental_steps', [])
                    })
        
        print(f"\n--- Code Implementation Analysis ---")
        if previous_implementations:
            print(f"  - Context: {len(previous_implementations)} existing implementation(s)")
        else:
            print(f"  - Context: Writing from scratch (no previous code)")
        
        # Generate/Update code
        new_plan = perform_code_rag(
            result=plan,
            kb_code=self.kb_code,
            model=self.model,
            generation_config=self.generation_config,
            previous_implementations=previous_implementations
        )
        
        # Snapshot: Code Generated
        new_plan["iteration"] = next_plan_idx
        new_plan["stage"] = "Code Generated"
        self.state["plan_history"].append(new_plan.copy())
        self.state["current_plan"] = new_plan

        self._log_action(
            action="refine_code_rag",
            input_ctx={
                "num_previous_implementations": len(previous_implementations),
                "iteration": next_plan_idx
            },
            result=new_plan,
            rationale="Updated code based on refined experimental steps"
        )

        # --- HUMAN CODE REVIEW ---
        human_feedback = None
        if enable_human_feedback and not new_plan.get("error"):
            temp_dir = self.output_dir / "temp_code_review_iter"
            print(f"\n--- Human Code Review (Iteration {next_plan_idx}) ---")
            
            if temp_dir.exists(): 
                shutil.rmtree(temp_dir)
            files = write_experiments_to_disk(new_plan, str(temp_dir))
            
            if files:
                while True:
                    print("\n" + "="*60)
                    print(f"👀 CODE REVIEW REQUIRED")
                    print("="*60)
                    print(f"1. Review files in: {temp_dir.resolve()}")
                    print(f"2. Inspect the {len(files)} new Python file(s).")
                    print("3. Press ENTER to approve, or type feedback to refine")
                    
                    code_feedback = get_user_feedback()
                    
                    if not code_feedback:
                        print("✅ Code accepted.")
                        break
                    
                    human_feedback = code_feedback
                    self.state["human_feedback_history"].append({
                        "phase": "code_iteration", 
                        "feedback": code_feedback
                    })
                    print(f"\n🛠️  Refining code based on: '{code_feedback}'...")
                    
                    new_plan = refine_code_with_feedback(
                        result=new_plan,
                        feedback=code_feedback,
                        model=self.model,
                        generation_config=self.generation_config
                    )
                    
                    # Snapshot: Code Refined
                    new_plan["iteration"] = next_plan_idx
                    new_plan["stage"] = "Code Refined"
                    self.state["plan_history"].append(new_plan.copy())
                    self.state["current_plan"] = new_plan
                    
                    print(f"  - 💾 Overwriting files in {temp_dir} with refined code...")
                    files = write_experiments_to_disk(new_plan, str(temp_dir))
        
        self._log_action(
            action="refine_implementation_code",
            input_ctx={"iteration": next_plan_idx},
            result=new_plan,
            rationale="Code refinement complete",
            feedback=human_feedback
        )
        return new_plan

    def update_plan_with_results(self,
                                 results: Any,
                                 output_json_path: Optional[str] = None,
                                 enable_human_feedback: bool = True,
                                 state_file_path: Optional[str] = None,
                                 use_literature_rag: bool = False) -> Dict[str, Any]:
        """
        Iterates on the current experimental plan based on new results.
        
        This is the main entry point for the iteration loop. It orchestrates:
        1. Scientific plan refinement (refine_plan)
        2. Implementation code updates (refine_implementation_code)
        3. File saving and report generation
        
        For more granular control, call refine_plan() and refine_implementation_code()
        separately.
        
        **Supported Result Formats:**
        
        The `results` parameter is highly flexible and accepts:
        
        **1. Text String (Qualitative Observations)**
            >>> agent.update_plan_with_results(
            ...     results="Yield was 12%, unexpected precipitation"
            ... )
        
        **2. Single File Path**
            >>> agent.update_plan_with_results(
            ...     results="./experiments/run_005.csv"
            ... )
            >>> # Auto-discovers ./experiments/run_005.json metadata
        
        **3. Image Path (Visual Analysis)**
            >>> agent.update_plan_with_results(
            ...     results="./plots/failure_analysis.png"
            ... )
        
        **4. Data Dictionary**
            >>> agent.update_plan_with_results(
            ...     results={
            ...         "yield": 45.2,
            ...         "purity": 87.3,
            ...         "observations": "Product color changed to yellow"
            ...     }
            ... )
        
        **5. File + Description (Recommended for Images)**
            >>> agent.update_plan_with_results(
            ...     results={
            ...         "path": "./microscopy/crystals.tiff",
            ...         "description": "Crystal morphology shows needle-like structure"
            ...     }
            ... )
        
        **6. List of Mixed Formats (Most Flexible)**
            >>> agent.update_plan_with_results(
            ...     results=[
            ...         "Experiment date: 2024-01-15",
            ...         "./data/icpms_run12.csv",              # Quantitative data
            ...         "./data/icpms_run12.json",             # Optional metadata
            ...         {
            ...             "path": "./photos/product.jpg",
            ...             "description": "White crystalline solid"
            ...         },
            ...         {
            ...             "temp_max": 78.5,
            ...             "pressure_stable": True
            ...         },
            ...         "./logs/errors.txt",                   # Equipment logs
            ...         "Stirrer stopped at t=15min, restarted manually"
            ...     ]
            ... )
        
        **Data File Handling:**
        - **CSV/Excel files** (.csv, .xlsx, .xls):
          * Automatically parsed and summarized
          * Metadata JSON auto-discovered (e.g., data.csv → data.json)
          * Column definitions and units included if metadata present
        
        - **Image files** (.png, .jpg, .jpeg, .tiff, .bmp):
          * Loaded and passed to vision model for analysis
          * Supports plots, microscopy, photos, diagrams
        
        - **Log files** (.txt, .log, .md, .json):
          * Read as text and included in context
          * Useful for equipment errors, timestamps, notes
        
        **Workflow Overview:**
        
        Phase 1 - Scientific Refinement:
            1. Parse results (multimodal)
            2. Search knowledge base for relevant context
            3. LLM analyzes and proposes strategy revision
            4. Human review (if enabled)
            5. Incorporate feedback and regenerate
        
        Phase 2 - Implementation Update:
            1. Extract previous code implementations
            2. LLM decides: preserve, update, or rewrite
            3. Generate updated scripts
            4. Human code review (if enabled)
            5. Save to ./output_scripts/
        
        Phase 3 - Persistence:
            1. Save plan JSON
            2. Save state JSON (for resumption)
            3. Generate HTML report
        
        Args:
            results: Experimental outcomes. Accepts:
                - String: Text description
                - String: File path (data, image, or log)
                - Dict: Structured data or {path: ..., description: ...}
                - List: Mix of any above formats
                See format examples above for details.
            
            output_json_path: Path to save the updated plan. If provided:
                - Saves plan to: {output_json_path}
                - Saves state to: {output_json_path}.state.json
                - Saves report to: {output_json_path}.html
                Example: "./outputs/iteration_2.json"
            
            enable_human_feedback: If True, pauses twice for user review:
                1. After scientific plan generation
                2. After code generation
                Set to False for fully autonomous operation.
                Defaults to True.
            
            state_file_path: Optional path to restore state from a checkpoint.
                Useful for resuming after shutdown. Equivalent to calling
                agent.restore_state() before this method.
                Example: "./outputs/session.state.json"
            
            use_literature_rag: If True, searches knowledge base for context 
                           relevant to the experimental results. 
                           Defaults to False for faster iteration.
        
        Returns:
            Dict containing the complete agent state:
            {
                "session_id": "...",
                "objective": "...",
                "iteration_index": 2,
                "current_plan": {...},
                "plan_history": [...],
                "experimental_results": [...],
                "status": "iterated"
            }
        
        Raises:
            ValueError: If no active state found and no state_file_path provided
        
        Example 1 - Simple Text Results:
            >>> agent.update_plan_with_results(
            ...     results="Yield dropped to 15%, likely due to low temperature"
            ... )
        
        Example 2 - Data File Results:
            >>> agent.update_plan_with_results(
            ...     results="./lab_data/hplc_run_005.csv",
            ...     output_json_path="./outputs/iteration_2.json"
            ... )
        
        Example 3 - Complete Multi-Modal Results:
            >>> agent.update_plan_with_results(
            ...     results=[
            ...         "Run completed successfully on 2024-01-15 at 14:30",
            ...         "./data/gc_ms_results.csv",
            ...         {
            ...             "path": "./plots/conversion_vs_time.png",
            ...             "description": "Conversion plateaus at 60min"
            ...         },
            ...         {
            ...             "yield": 78.5,
            ...             "selectivity": 92.3,
            ...             "notes": "Product purity excellent"
            ...         },
            ...         "./logs/temperature_profile.txt"
            ...     ],
            ...     output_json_path="./outputs/iteration_3.json",
            ...     enable_human_feedback=True
            ... )
        
        Example 4 - Resume from Checkpoint:
            >>> # After restarting Python
            >>> agent = PlanningAgent()
            >>> agent.update_plan_with_results(
            ...     results="./new_data.csv",
            ...     state_file_path="./outputs/session.state.json"
            ... )
        
        Example 5 - Step-by-Step Control:
            >>> # For maximum control, use individual methods:
            >>> plan = agent.refine_plan(results="...")
            >>> # Review plan, make modifications...
            >>> plan = agent.refine_implementation_code(plan)
            >>> # Review code, make modifications...
            >>> agent._save_results_to_json(plan, "./plan.json")
        
        Notes:
            - The method is stateful - maintains session history across calls
            - Safe to shut down between calls (use state_file_path to resume)
            - Automatically includes previous code when generating updates
            - All outputs saved to ./output_scripts/ directory
        """
        
        # Phase 1: Refine scientific strategy
        plan = self.refine_plan(
            results=results,
            enable_human_feedback=enable_human_feedback,
            state_file_path=state_file_path,
            use_literature_rag=use_literature_rag
        )
        
        if plan.get("error"):
            if output_json_path:
                self._save_results_to_json(plan, output_json_path)
            return self.state
        
        # Phase 2: Update implementation code
        plan = self.refine_implementation_code( 
            plan=plan,
            enable_human_feedback=enable_human_feedback
        )
        
        # Final state update
        self.state["current_plan"] = plan
        self.state["status"] = "iterated"
        
        # Save outputs
        final_out = "./output_scripts"
        print(f"\n--- Saving Final Scripts to: {final_out} ---")
        write_experiments_to_disk(plan, final_out)
        
        if output_json_path:
            self._save_results_to_json(plan, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            self._generate_html_report(output_json_path)
            
        return self.state
    
    def _generate_html_report(self, json_path: str):
        """Helper to generate HTML report alongside JSON."""
        if not json_path: return
        html_path = str(Path(json_path).with_suffix('.html'))
        try:
            generator = HTMLReportGenerator(self.state)
            generator.generate(html_path)
        except Exception as e:
            print(f"⚠️ Failed to generate HTML report: {e}")

    def perform_technoeconomic_analysis(self, objective: str,
                                        knowledge_paths: Optional[List[str]] = None,
                                        primary_data_set: Optional[Union[str, Dict[str, str]]] = None,
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
            ...     knowledge_paths=["./market_data/", "./reports/"],
            ... )
            >>> 
            >>> # Use TEA insights in experimental planning
            >>> plan = agent.propose_experiments(
            ...             objective="Develop lithium extraction process",
            ...             knowledge_paths=["./extraction_methods/"],
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
        
        knowledge_paths (Optional[List[str]]): Paths to documents for TEA context.
            Should include market data, pricing reports, criticality assessments,
            existing TEA studies, and process descriptions. Supports both PDF/TXT and Excel/CSV.
            Examples: ["./market_reports/", "./critical_materials_report.pdf", "./public_data.xlsx", "./public_data.json"]
        
        primary_data_set (Optional[Dict[str, str]]): Main dataset for analysis.
            Can contain composition, concentration, or yield data.
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
        ...     knowledge_paths=["./enzyme_papers/"],
        ...     code_paths=["./plate_reader_api/"],
        ...     output_json_path="./plan.json"
        ... )
        >>> # User reviews in console, provides feedback or approves
        >>> # Final scripts saved to ./output_scripts/

    Example - Advanced with Data:
        >>> state = agent.propose_experiments(
        ...     objective="Identify optimal precipitation conditions",
        ...     knowledge_paths=["./papers/", "./protocols.pdf"],
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
        
        # 0a. Resolve Primary Data
        primary_data_set = resolve_primary_data_path(primary_data_set)
        # 0b. Resolve image paths
        # Images explicitly specified by user undr image_paths (will be deprecated in the future)
        manual_images = image_paths or []
        # Find new images under the provided knowledge paths but exclude any that are already in manual_images
        auto_images = [img for img in extract_images(knowledge_paths) if img not in manual_images]
        # Append auto-images to the end so manual descriptions stay aligned with manual images
        all_image_paths = manual_images + auto_images

        # 1. State Initialization (if starting fresh with TEA)
        if not self.state:
            self.state = self._initialize_state(
                objective=objective,
                knowledge_paths=knowledge_paths,
                code_paths=None,
                primary_data_set=primary_data_set,
                image_paths=all_image_paths,
                image_descriptions=image_descriptions
            )

        #  TEA is always step 0 (pre-planning)
        self.state["iteration_index"] = 0

        # 2. Build KB if needed
        if not self._ensure_kb_is_ready(knowledge_paths, code_paths=None):
            error_result = {"error": "KB Init Failed"}
            self._log_action(
                action="perform_technoeconomic_analysis",
                input_ctx={"objective": objective},
                result=error_result,
                rationale=None
            )
            return error_result
        
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
            image_paths=all_image_paths, 
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
            # Append copy to history
            self.state["plan_history"].append(res.copy())
     
        self._log_action(
            action="perform_technoeconomic_analysis",
            input_ctx={
                "objective": objective,
                "knowledge_paths": knowledge_paths,
                "has_primary_data": primary_data_set is not None,
                "has_literature": bool(lit_context)
            },
            result=res,
            rationale=res.get("technoeconomic_assessment", {}).get("summary") if not res.get("error") else None
        )
        
        # 6. Save & Generate Report
        if output_json_path:
            self._save_results_to_json(res, output_json_path)
            self._save_state_to_json(output_json_path + ".state.json")
            
            # Trigger HTML Generation (will show TEA card)
            self._generate_html_report(output_json_path)

        return res