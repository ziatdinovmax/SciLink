from pathlib import Path
import copy
import hashlib
import json
import logging
import re
import shutil
import uuid
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from scilink.knowledge import KnowledgeBase
from .parser_utils import (
    plan_directions,
    portfolio_to_experiment_shim,
    resync_portfolio,
    generate_repo_map,
    write_experiments_to_disk,
    resolve_primary_data_path,
    parse_multimodal_results
)
from .repo_loader import clone_git_repository

from .instruct import (
    HYPOTHESIS_GENERATION_INSTRUCTIONS,
    HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK,
    IDEATION_OUTPUT_RULES,
    WHITE_PAPER_INSTRUCTIONS,
    TEA_INSTRUCTIONS
)

from ...auth import get_internal_proxy_key
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel

from ._deprecation import normalize_params
from .base_agent import BaseAgent

from .planning_rag import (
    author_portfolio,
    portfolio_contract,
    perform_science_rag,
    perform_code_rag,
    refine_plan_with_feedback,
    refine_code_with_feedback,
    verify_plan_relevance,
    critique_plan,
    generate_plan_candidates,
    judge_plan_candidates
)

from ...skills.loader import load_skill

from scilink.parsers import ingest_files, extract_images

from .user_interface import (
    display_plan_summary,
    get_user_feedback,
    display_plan_candidates,
    get_candidate_selection,
    format_caveats
)

from .html_generator import HTMLReportGenerator

from .base_agent import BaseAgent


# Generic research verbiage that says nothing about WHICH campaign an
# objective belongs to. Kept small and domain-agnostic: the point is to
# strip words shared by virtually every objective ("investigate", "novel",
# "materials"), so the overlap test below compares actual topic vocabulary.
_CAMPAIGN_STOPWORDS = frozenset("""
    the and for with from into using via under over between during
    across within without about this that these those than not non
    study studies investigate investigation explore exploration
    research ideate ideation brainstorm brainstorming directions
    direction propose proposal proposals develop development design
    optimize optimization improve improvement improving maximize
    minimize enhance enhancing understand understanding characterize
    characterization analysis analyze effect effects impact impacts
    influence driven based related property properties performance
    condition conditions experimental experiment experiments campaign
    campaigns plan plans planning strategy strategies approach
    approaches method methods material materials system systems
    new novel high low advanced controlled
    """.split())


def objectives_share_campaign(previous: str, new: str,
                              min_overlap: float = 0.35) -> bool:
    """Lexical continuity test between two campaign objectives (issue #396).

    Returns True when ``new`` plausibly continues the campaign ``previous``
    belongs to. Content words (stopwords and generic research verbiage
    removed, naive plural fold) are compared by overlap over the smaller
    set: a rewording or narrowing of the same topic keeps most of its
    domain vocabulary, while an unrelated topic shares at most a stray
    generic term. Deterministic on purpose — this is the fallback when the
    caller does not pass ``new_campaign`` explicitly, and it must not cost
    an LLM call. Ties toward continuity only when either side has no
    content words at all (an objective with nothing specific in it is not
    a statement of a new topic).

    The threshold leans DELIBERATELY toward declaring a new campaign,
    because the two mistakes are not symmetric (review, PR #394): calling
    two topics one campaign carries the previous corpus forward — the #396
    leak — while wrongly splitting only loses carry-forward, which the
    caller can override with ``new_campaign=False``. Measured over real
    objectives from live sessions, unrelated pairs score <= 0.20 (a
    perovskite-solar vs solar-wind pair lands exactly there) and genuine
    continuations >= 0.43, so 0.35 sits mid-gap instead of 0.05 above the
    dangerous side.
    """
    def _depluralize(w: str) -> str:
        """Drop ONE trailing plural 's'.

        `rstrip("s")` strips every trailing s, which mangles ordinary
        domain words — stress→stre, mass→ma, gas→ga, analysis→analysi,
        process→proce — and then fails at the job it was doing: gas→'ga'
        but gases→'gase', so the two no longer match. Words ending in
        ss/us/is are not plurals; very short words are left alone.
        (Still a naive fold, not a stemmer: gas/gases remains unmatched.)
        """
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            return w[:-1]
        return w

    def _tokens(text: str) -> set:
        words = re.findall(r"[a-z][a-z0-9]{2,}", (text or "").lower())
        return {_depluralize(w) for w in words
                if w not in _CAMPAIGN_STOPWORDS}
    prev_t, new_t = _tokens(previous), _tokens(new)
    if not prev_t or not new_t:
        return True
    overlap = len(prev_t & new_t) / min(len(prev_t), len(new_t))
    return overlap >= min_overlap


_LIT_REF = "__scilink_lit_ref__:"
_LIT_MIN_CHARS = 4000          # below this, a copy is cheaper than a lookup


def compact_planner_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Serialization view of planner state with literature stored ONCE.

    A campaign's corpus is carried on every plan snapshot, so a state file
    holds the same text N times — live, 1.73 MB of a 1.98 MB state file was
    five copies of two corpora. Each copy is replaced by a reference into
    `_literature_store`; `expand_planner_state` restores them verbatim on
    load, so no consumer ever sees a reference and no context is lost.

    Returns a shallow-copied view: the live state is never mutated.
    """
    if not isinstance(state, dict):
        return state
    store: Dict[str, str] = {}

    def _ref(text: str) -> str:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        store.setdefault(key, text)
        return _LIT_REF + key

    def _plan(p: Any) -> Any:
        if not isinstance(p, dict):
            return p
        lit = p.get("literature_search")
        if isinstance(lit, str) and len(lit) >= _LIT_MIN_CHARS:
            p = dict(p)
            p["literature_search"] = _ref(lit)
        return p

    out = dict(state)
    if state.get("current_plan") is not None:
        out["current_plan"] = _plan(state.get("current_plan"))
    if state.get("plan_history"):
        out["plan_history"] = [_plan(p) for p in state["plan_history"]]
    pc = state.get("plan_candidates")
    if isinstance(pc, dict) and pc.get("candidates"):
        out["plan_candidates"] = dict(pc)
        out["plan_candidates"]["candidates"] = [
            _plan(c) for c in pc["candidates"]]
    if store:
        out["_literature_store"] = store
    return out


def expand_planner_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`compact_planner_state` — restores literature text.

    Tolerates a state that was never compacted (no store, no refs) and an
    unresolvable reference (left as-is with a warning rather than dropped,
    so a truncated file degrades visibly instead of silently losing text).
    """
    if not isinstance(state, dict):
        return state
    store = state.get("_literature_store") or {}
    if not store and not any(
        isinstance((p or {}).get("literature_search"), str)
        and str((p or {}).get("literature_search")).startswith(_LIT_REF)
        for p in ([state.get("current_plan")]
                  + list(state.get("plan_history") or []))
    ):
        return state

    def _plan(p: Any) -> Any:
        if not isinstance(p, dict):
            return p
        lit = p.get("literature_search")
        if isinstance(lit, str) and lit.startswith(_LIT_REF):
            key = lit[len(_LIT_REF):]
            if key in store:
                p = dict(p)
                p["literature_search"] = store[key]
            else:
                logging.warning(
                    "Literature reference %s missing from the state store; "
                    "leaving the reference in place.", key)
        return p

    out = dict(state)
    if state.get("current_plan") is not None:
        out["current_plan"] = _plan(state.get("current_plan"))
    if state.get("plan_history"):
        out["plan_history"] = [_plan(p) for p in state["plan_history"]]
    pc = state.get("plan_candidates")
    if isinstance(pc, dict) and pc.get("candidates"):
        out["plan_candidates"] = dict(pc)
        out["plan_candidates"]["candidates"] = [
            _plan(c) for c in pc["candidates"]]
    out.pop("_literature_store", None)
    return out


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
            (e.g., "claude-opus-4-6", "gemini-3.1-pro-preview").
        base_url: Base URL for internal proxy endpoint.
            When provided, uses OpenAI-compatible client.
            When None, uses LiteLLM for multi-provider support.
        embedding_model: Embedding model name.
        embedding_api_key: API key for the embedding LLM provider.
        futurehouse_api_key: UNUSED. Retained for call-site compatibility;
            the internal literature fallback was removed. Literature flows
            in via ``external_context`` (orchestrator's search_literature).
        kb_base_path: Path for knowledge base storage.
        code_chunk_size: Chunk size for code files.
        output_dir: Output directory for artifacts.
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(self, api_key: str = None,
                 model_name: str = "claude-opus-4-6",
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

        # No literature agent here: the internal fallback was removed —
        # literature reaches plan generation only as external_context, via
        # the orchestrator's search_literature tool (which owns its own
        # LiteratureSearchAgent). futurehouse_api_key stays accepted for
        # call-site compatibility but is unused at this level.


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
            "campaign_id": 1,
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

        
    def starts_new_campaign(self, objective: Optional[str],
                            new_campaign: Optional[bool] = None) -> bool:
        """Would generating a plan for ``objective`` start a NEW campaign?

        A session can hold several unrelated brainstorms/campaigns, and
        every continuity heuristic (literature carry-forward, white-paper
        corpus selection, refine auto-load) is scoped to ONE campaign
        (issue #396). An explicit ``new_campaign`` signal wins in both
        directions; otherwise lexical disjointness between the new
        objective and the current campaign's objective decides.
        Non-mutating — ``generate_plan`` applies the transition.
        """
        if not self.state or not self.state.get("objective") or not objective:
            return False
        if new_campaign is not None:
            return bool(new_campaign)
        return not objectives_share_campaign(self.state["objective"], objective)

    def _apply_campaign_transition(self, objective: str) -> None:
        """Open a new campaign: bump the id and drop the working state the
        previous campaign would otherwise leak through (current_plan and
        the best-of-N candidate set). History entries stay archived under
        their own campaign_id stamps."""
        prev_cid = int(self.state.get("campaign_id") or 1)
        self.state["campaign_id"] = prev_cid + 1
        self.state["current_plan"] = None
        self.state.pop("plan_candidates", None)
        # The new campaign decides its own kind — an ideation brainstorm
        # must not make the next topic's lab plan render as ideation.
        self.state.pop("plan_kind", None)
        print(f"  - 🧭 Objective changed materially — starting campaign "
              f"#{prev_cid + 1}. The previous campaign's plans and "
              f"literature stay archived and are NOT carried forward.")

    def _save_state(self) -> None:
        """Persist state with literature stored once (see
        :func:`compact_planner_state`). Falls back to the plain dump if
        compaction fails — a state file must always get written."""
        state_file = self.output_dir / self._get_state_filename()
        try:
            payload = compact_planner_state(self.state)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Literature compaction skipped: {e}")
            payload = self.state
        try:
            with open(state_file, 'w') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save {self.agent_type} state: {e}")

    def load_state(self, state_path: str) -> bool:
        """Restore state, re-inflating any stored literature references."""
        if not super().load_state(state_path):
            return False
        self.state = expand_planner_state(self.state)
        return True

    def _review_preview_path(self) -> Optional[str]:
        """Render the plan under review to HTML and return its path.

        The saved `plan.html` is written by the orchestrator only AFTER this
        call returns, so a reviewer prompted mid-run has nothing to open —
        yet these summaries run to thousands of words in a terminal. Render
        a preview from the live state instead, refreshed at every review so
        it always shows the plan being asked about. Never fatal: a failed
        preview must not block the review prompt.
        """
        try:
            path = Path(self.output_dir) / "plan_preview.html"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Swallow the generator's own "HTML Report updated" line: the
            # review block announces the path itself, one line above.
            import io as _io
            import contextlib as _ctx
            with _ctx.redirect_stdout(_io.StringIO()):
                HTMLReportGenerator(self.state).generate(str(path))
            return str(path)
        except Exception as e:  # noqa: BLE001 - cosmetic aid only
            logging.debug(f"Review preview unavailable: {e}")
            return None

    def _is_ideation_campaign(self) -> bool:
        """Was the current plan authored under the ideation profile?

        Reads the plan's own ``type`` stamp first — a plan dict restored or
        read in isolation then still knows what it is — and falls back to
        the best-of-N selection state for plans authored before the stamp
        existed. Single-plan runs ignore the profile by design, so they
        report as lab.
        """
        cur = (self.state or {}).get("current_plan") or {}
        if cur.get("type") == "ideation":
            return True
        # An explicit lab stamp is the caller saying "this plan is a bench
        # plan" and outranks the campaign — the case is a campaign that
        # ideated, then was asked for the runnable protocol of the direction
        # that won. Other types (TEA) still fall through, as before.
        if cur.get("type") == "lab":
            return False
        if (self.state or {}).get("plan_kind") == "ideation":
            return True
        pc = (self.state or {}).get("plan_candidates") or {}
        return pc.get("profile") == "ideation"

    def _stamp_campaign(self, plan_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Stamp campaign id and plan kind onto a snapshot.

        The kind (`type`) is re-applied here because conformance, critic and
        refinement passes re-emit the plan JSON and drop fields they were not
        told to keep — the same way `literature_search` had to be
        final-stamped. An existing `type` is never overwritten (TEA sets its
        own).
        """
        if isinstance(plan_dict, dict):
            plan_dict["campaign_id"] = int(self.state.get("campaign_id") or 1)
            kind = self.state.get("plan_kind")
            if kind and not plan_dict.get("type"):
                plan_dict["type"] = kind
            # Every snapshot that becomes current_plan passes through here,
            # which makes it the one place a refined portfolio can be kept
            # self-consistent.
            resync_portfolio(plan_dict)
        return plan_dict

    def _finalize_literature(self, plan_dict: Dict[str, Any],
                             literature: Optional[str]) -> None:
        """Final-stamp SYSTEM-OWNED literature provenance onto a plan.

        ``literature_search`` must never be model prose — but conformance,
        critic and feedback-refinement passes RE-EMIT the plan JSON, so a
        stamp applied before them can be replaced by a model placeholder
        ('See prior iteration context (unchanged).', seen live). Call this
        LAST, after every rewrite pass: it stamps the returned plan and the
        state's current_plan / latest history snapshot (which may be earlier
        copies).
        """
        if not literature:
            return
        plan_dict["literature_search"] = literature
        cur = self.state.get("current_plan")
        if isinstance(cur, dict):
            cur["literature_search"] = literature
        hist = self.state.get("plan_history") or []
        if hist and isinstance(hist[-1], dict):
            hist[-1]["literature_search"] = literature

    def generate_white_paper(self, audience_context: Optional[str] = None) -> str:
        """Write a sponsor-facing white paper from the current campaign plan.

        Distills the selected plan (plus, after a best-of-N run, the
        alternative candidate strategies and the judge's comparative
        reasoning, and the critic's caveats) into a technical pre-proposal
        aimed at sponsors with technical backgrounds — significance and
        payoff forward, mechanisms rigorous, no bench-level protocol detail.

        Args:
            audience_context: Optional targeting notes (e.g. "emphasize
                fundamental-science significance" or "lead with cost and
                scalability impact").

        Returns:
            The white paper as markdown.

        Raises:
            ValueError: when no plan exists yet in this campaign state.
        """
        # Campaign scoping (issue #396): a session can hold several
        # unrelated campaigns; the white paper must be built ONLY from the
        # current one. Entries without a stamp predate campaign tracking
        # and are treated as campaign 1.
        cid = int(self.state.get("campaign_id") or 1)

        def _same_campaign(p: Dict[str, Any]) -> bool:
            return int(p.get("campaign_id") or 1) == cid

        plan = self.state.get("current_plan") or next(
            (p for p in reversed(self.state.get("plan_history") or [])
             if _same_campaign(p)), None
        )
        if not plan or not plan.get("proposed_experiments"):
            raise ValueError(
                "No campaign plan exists yet — generate a plan first; the "
                "white paper is distilled from it."
            )

        parts = [WHITE_PAPER_INSTRUCTIONS,
                 f"## Research Objective:\n{self.state.get('objective', '')}"]
        if audience_context:
            parts.append(f"\n## Sponsor / Audience Targeting Notes:\n"
                         f"{audience_context}")

        selected = copy.deepcopy(plan)
        literature = selected.pop("literature_search", None)
        findings = selected.pop("critic_findings", None)
        # Literature is CAMPAIGN context, not per-iteration: a restructured
        # or refined plan may carry none of its own — or worse, a short
        # model-authored note ('no new search was executed...') that shadows
        # the real corpus. Take the MOST SUBSTANTIAL literature across the
        # current plan and the SAME-CAMPAIGN history; length is the tiebreak
        # because the campaign's actual corpus dwarfs any stub. History from
        # other campaigns is excluded outright (issue #396) — length must
        # never arbitrate between two topics' corpora.
        candidates_lit = [literature] + [
            prev.get("literature_search")
            for prev in (self.state.get("plan_history", []) or [])
            if _same_campaign(prev)
        ]
        candidates_lit = [c for c in candidates_lit if c]
        literature = (max(candidates_lit, key=lambda c: len(str(c)))
                      if candidates_lit else None)
        # A portfolio is the deliverable of an ideation run, and it sits deep
        # in the plan JSON where the 20k truncation can cut it — surface it
        # first, in full, so the paper is written from all N directions.
        _portfolio = plan_directions(selected)
        if _portfolio:
            parts.append(
                f"\n## RESEARCH DIRECTIONS IN THE SELECTED PLAN "
                f"({len(_portfolio)}) — this is the program; give each one "
                f"its due weight, keep the author's ranking, and cite each "
                f"by its own id so the paper and the dossier can be read "
                f"side by side:\n"
                + json.dumps(_portfolio, indent=2)[:20000])
        parts.append("\n## SELECTED Campaign Plan:\n"
                     + json.dumps(selected, indent=2)[:20000])

        cand_state = self.state.get("plan_candidates") or {}
        candidates = cand_state.get("candidates") or []
        if len(candidates) > 1:
            sel_idx = cand_state.get("selected_index", 1)
            alt_blocks = []
            judge = cand_state.get("judge") or {}
            if cand_state.get("human_override"):
                parts.append(
                    f"\n## SELECTION PROVENANCE: The PI personally selected "
                    f"Candidate {sel_idx} as the flagship, OVERRIDING the "
                    f"judge (which preferred Candidate "
                    f"{judge.get('selected_candidate', '?')}). Treat the "
                    f"PI's choice as the primary thrust; the judge's "
                    f"comparative reasoning below is context, not the "
                    f"selection rationale."
                )
            scores = {s.get("candidate"): s for s in judge.get("scores", [])}
            for ci, cand in enumerate(candidates, 1):
                if ci == sel_idx:
                    continue
                exp = dict((cand.get("proposed_experiments") or [{}])[0])
                exp.pop("optimization_params", None)  # protocol detail
                sc = scores.get(ci, {})
                alt_blocks.append(
                    f"### Candidate {ci} "
                    f"[judge comment: {sc.get('comment', 'n/a')}]\n"
                    + json.dumps(exp, indent=2)[:6000]
                )
            if alt_blocks:
                parts.append(
                    "\n## ALTERNATIVE Candidate Strategies (judge-scored "
                    "runners-up; use mechanistically distinct ones as "
                    "secondary thrusts, faithfully to their content):\n"
                    + "\n".join(alt_blocks)
                    + f"\nJudge's comparative reasoning: "
                      f"{judge.get('reasoning', 'n/a')}"
                )
        if findings:
            parts.append("\n## Reviewer Caveats (fold into Risks and "
                         "Mitigation):\n" + json.dumps(findings, indent=2))
        if literature:
            lit = str(literature)
            # Guard: a stored plan may carry file PATH(S) instead of content
            # (historical sessions where a comma-joined path list slipped
            # through as raw text) — resolve them to the actual documents so
            # citation never sees bare filenames.
            if len(lit) < 4096 and "\n" not in lit.strip():
                def _is_file(p: Path) -> bool:
                    try:
                        return p.is_file()
                    except OSError:  # e.g. name-too-long for raw prose
                        return False
                cand_paths = [Path(t.strip()) for t in lit.split(",")
                              if t.strip()]
                if cand_paths and all(_is_file(p) for p in cand_paths):
                    lit = "\n\n".join(p.read_text() for p in cand_paths)
            parts.append("\n## Literature Context:\n" + lit[:15000])
            # Long syntheses put their bibliographies well past any sane
            # truncation — extract every DOI-bearing line from the FULL
            # context so the paper can cite with real DOIs, never invented
            # ones.
            ref_lines = [ln.strip() for ln in lit.splitlines()
                         if re.search(r"10\.\d{4,}/", ln)]
            refs = "\n".join(dict.fromkeys(ref_lines))[:8000]
            if refs:
                parts.append("\n## Bibliography extracted from the full "
                             "literature context (cite from these lines; "
                             "they carry the DOIs):\n" + refs)

        print("\n--- Generating White Paper ---")
        response = self.model.generate_content(
            ["\n".join(parts)], generation_config=self.generation_config
        )
        text = response.text if hasattr(response, "text") else str(response)
        # Strip an accidental fence around the whole document.
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n", "", text)
            text = re.sub(r"\n```$", "", text).strip()
        return text

    def rebind_kb(self, kb_base_path: str) -> bool:
        """Re-point both knowledge bases at a new storage path and reload.

        Mirrors the constructor's path derivation so an already-constructed
        agent can switch to another persisted KB (e.g. the meta attaching a
        stable knowledge dir after the planning child exists). In-memory KB
        state is cleared first — a target with no persisted KB yields empty
        KBs, never a stale carry-over.

        Returns:
            True if either KB loaded from the new location.
        """
        base_path = Path(kb_base_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        self.kb_docs_prefix = base_path.parent / f"{base_path.name}_docs"
        self.kb_docs_index = str(self.kb_docs_prefix.with_suffix(".faiss"))
        self.kb_docs_chunks = str(self.kb_docs_prefix.with_suffix(".json"))
        self.kb_docs_sources_path = str(self.kb_docs_prefix.with_suffix(".sources.json"))

        self.kb_code_prefix = base_path.parent / f"{base_path.name}_code"
        self.kb_code_index = str(self.kb_code_prefix.with_suffix(".faiss"))
        self.kb_code_chunks = str(self.kb_code_prefix.with_suffix(".json"))
        self.kb_code_map_path = str(self.kb_code_prefix.with_suffix(".maps.json"))
        self.kb_code_sources_path = str(self.kb_code_prefix.with_suffix(".sources.json"))

        for kb in (self.kb_docs, self.kb_code):
            kb.index = None
            kb.chunks = []
            kb.sources = []
            kb.repo_maps = {}

        print(f"--- Rebinding Knowledge Bases to {base_path.parent} ---")
        self._load_knowledge_bases()
        return self._kb_is_built

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

    def _build_skill_context(self, stage: str) -> Optional[str]:
        """Build a skill context string for injection into LLM prompts.

        Args:
            stage: Primary skill section to use. One of ``"planning"``,
                ``"implementation"``, ``"interpretation"``, ``"overview"``.

        Returns:
            Formatted skill context string, or ``None`` if no skill is loaded
            or the requested section is empty.
        """
        skill_sections = self.state.get("skill_sections") if self.state else None
        if not skill_sections:
            return None

        skill_name = self.state.get("skill_name", "domain skill")

        parts = []

        # Always include overview for domain context
        overview = skill_sections.get("overview", "")
        if overview and stage != "overview":
            parts.append(f"### Overview\n{overview}")

        # Primary section
        content = skill_sections.get(stage, "")
        if content:
            parts.append(f"### {stage.title()}\n{content}")

        # Include validation rules alongside planning, interpretation, and implementation
        if stage in ("planning", "interpretation", "implementation"):
            validation = skill_sections.get("validation", "")
            if validation:
                parts.append(f"### Validation Criteria\n{validation}")

        if not parts:
            return None

        return (
            f"\n## MANDATORY Domain Skill Rules: {skill_name}\n"
            "The following rules are MANDATORY constraints on your experimental plan. "
            "These encode validated domain expertise and override general-purpose defaults.\n\n"
            + "\n\n".join(parts)
        )

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
            doc_chunks.extend(ingest_files(knowledge_paths, is_code_mode=False,
                                           ocr_model=self.model))

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
                    reset_state: bool = False,
                    skill: Optional[str] = None,
                    external_context: Optional[str] = None,
                    n_candidates: int = 1,
                    candidate_report_dir: Optional[str] = None,
                    selection_profile: str = "lab",
                    new_campaign: Optional[bool] = None,
                    kind: str = "experiment") -> Dict[str, Any]:
        """
        Generate experimental plan (science only, no implementation code/protocol).

        This method performs:
        1. Knowledge base initialization (docs only)
        2. Literature search (optional)
        3. RAG-based hypothesis generation
        4. Self-correction loop
        5. Human feedback on strategy

        Does NOT generate implementation code. Use generate_implementation_code() for that.

        Args:
            objective: High-level research goal guiding hypothesis generation.
            knowledge_paths: Paths to scientific documents/data for the Docs KB.
                Supports PDFs, .txt, .md, .xlsx, .csv, and directories.
            primary_data_set: Main dataset to analyze. Can be a file path string
                or a dict with "file_path" (and optional "metadata_path") keys.
            additional_context: Extra text context injected into the prompt.
                Keys become section headers, values become content.
            image_paths: Paths to images (.png, .jpg, .tiff, .bmp) for
                multimodal analysis. Images in knowledge_paths are auto-discovered.
            image_descriptions: Text descriptions for each image, in the same
                order as image_paths.
            enable_human_feedback: If True, pauses for user review after
                hypothesis generation. Defaults to True.
            reset_state: If True, clears existing state and starts fresh.
                If False, appends to the current research session.
            external_context: Pre-fetched external context (e.g. from
                orchestrator's search_literature/query_molecules tools).
                The ONLY way literature enters plan generation — there is
                no internal search.
            n_candidates: Best-of-N width (clamped 1-4). At 1 (default) the
                single-plan path runs unchanged. Above 1, candidates are
                generated sequentially — each conditioned to test a DIFFERENT
                mechanistic approach — an LLM judge picks one, and (in
                interactive modes) the human may override before the usual
                plan review. A cap, not a quota: generation stops early when
                the evidence supports no further distinct approach. Note that
                even a narrowly-specified objective usually admits multiple
                strategies — the meaningful reason to stay at 1 is a strategy
                already being committed (follow-up iterations), not objective
                specificity. (The orchestrator tool layer applies exactly that
                policy: campaign-first plans default to 3.)
            candidate_report_dir: Directory for per-candidate HTML reports
                (persisted; referenced from the selection cards). Only used
                when n_candidates > 1.
            selection_profile: How the best-of-N judge WEIGHTS its pick
                (criteria and scores are identical either way). "lab"
                (default): feasibility/actionability first — the plan must
                run on the available platform. "ideation": information
                gain/novelty first, feasibility as tiebreaker, plus
                author-side grounding latitude — for research ideation
                rather than executable campaign planning. Applies only when
                n_candidates >= 2: the single-plan path ignores the profile
                entirely (documented behavior — ideation is a property of
                the multi-candidate process).
            new_campaign: Campaign-boundary signal (issue #396). True forces
                a new campaign (prior campaign's plans/literature archived,
                not carried forward); False forces continuation of the
                current campaign despite a reworded objective; None (default)
                auto-detects from lexical objective similarity. Ignored on a
                fresh state.

        Returns:
            Dict containing the experimental plan with keys:
                - proposed_experiments: List of experiment dicts with hypotheses,
                  steps, justifications, and expected outcomes
                - literature_search: Literature context (when external
                  context was supplied)
                - iteration: Current iteration number
                - stage: Pipeline stage that produced this plan
                - error/message: Present only if generation failed
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
                if self.starts_new_campaign(objective, new_campaign):
                    self._apply_campaign_transition(objective)
                self.state["objective"] = objective
        
        # Load skill (once, at entry point)
        if skill:
            try:
                parsed = load_skill(skill, domain="planning")
                self.state["skill_name"] = parsed["name"]
                self.state["skill_sections"] = parsed
                print(f"  - 📖 Skill loaded: {parsed['name']}")
            except FileNotFoundError:
                logging.warning(f"Skill '{skill}' not found — proceeding without domain skill")

        # Increment iteration
        existing_iter = self.state.get("iteration_index", 0)
        self.state["iteration_index"] = existing_iter + 1
        current_iter = self.state["iteration_index"]

        # Build KB (docs only). A missing / unbuilt knowledge base is NOT fatal:
        # perform_science_rag below falls back to the LLM's general scientific
        # knowledge when no documents are retrievable (its "Insufficient context"
        # path swaps in HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK). Aborting
        # here instead would break every planning task delegated without
        # literature documents (e.g. from the meta-agent).
        if not self._ensure_kb_is_ready(knowledge_paths, code_paths=None):
            logging.warning(
                "Knowledge base unavailable — generating the plan from general "
                "scientific knowledge without document retrieval."
            )
        
        # Build context string
        ctx_string = ""
        if additional_context:
            for header, content in additional_context.items():
                ctx_string += f"## {header}\n{content}\n\n"
            ctx_string = ctx_string.strip() if ctx_string else None
        
        # Literature enters ONLY as caller-provided external_context (the
        # orchestrator's search_literature tool is the sanctioned path).
        # The old internal fallback searched silently whenever a FutureHouse
        # key was present and no context was passed — an uninstructable
        # Edison call that neither the user nor the tool-calling LLM could
        # veto (seen firing against an explicit "no literature" request
        # during the #396 live probes). Removed; no replacement.

        # Build skill context for plan generation
        skill_planning_context = self._build_skill_context("planning")

        # RAG for science plan
        # A PORTFOLIO of research directions, or an EXPERIMENT. Ideation asked
        # for the first and was handed the second for a long time; the plan
        # tool designs bench experiments, and a portfolio squeezed into that
        # schema came back with its directions as pseudo-steps.
        _portfolio_run = (kind == "portfolio")
        print(f"\n--- Generating {'Research Portfolio' if _portfolio_run else 'Experimental Strategy'} ---")
        n_candidates = max(1, min(int(n_candidates or 1), 4))
        bestofn_candidates = None
        bestofn_judge = None
        bestofn_selected = None
        bestofn_reports = []
        if n_candidates > 1:
            # One engine, two contracts: a portfolio candidate is a set of
            # research directions, an experiment candidate a protocol. The
            # tier pinning, distinctness conditioning and early stop are
            # shape-agnostic and stay shared.
            candidates, author_context, tier = generate_plan_candidates(
                contract=(portfolio_contract() if _portfolio_run else None),
                objective=objective,
                kb_docs=self.kb_docs,
                model=self.model,
                generation_config=self.generation_config,
                n_candidates=n_candidates,
                primary_data_set=primary_data_set,
                image_paths=all_image_paths,
                image_descriptions=image_descriptions,
                additional_context=ctx_string,
                external_context=external_context,
                skill_context=skill_planning_context,
                selection_profile=selection_profile,
            )
            # Shim BEFORE judging: the judge, the candidate cards and the
            # dossier all read `proposed_experiments`, and re-teaching three
            # consumers the portfolio shape to judge a portfolio is the
            # migration this transition device exists to avoid.
            if _portfolio_run:
                candidates = [portfolio_to_experiment_shim(c)
                              for c in candidates]
            if len(candidates) > 1:
                print(f"\n--- Judging {len(candidates)} Candidate "
                      f"{'Portfolios' if _portfolio_run else 'Plans'} ---")
                bestofn_judge = judge_plan_candidates(
                    objective=objective,
                    candidates=candidates,
                    model=self.model,
                    generation_config=self.generation_config,
                    retrieved_context=author_context.get("retrieved_context"),
                    primary_data=author_context.get("primary_data"),
                    images=all_image_paths or None,
                    image_descriptions=image_descriptions,
                    additional_context=ctx_string,
                    skill_context=skill_planning_context,
                    fallback_tier=(tier == "fallback"),
                    selection_profile=selection_profile,
                )
                bestofn_selected = bestofn_judge["selected_candidate"]
                print(f"  - 🧑‍⚖️ Judge pick: Candidate {bestofn_selected}")
                # Persisted per-candidate reports (referenced from the
                # selection cards; kept for runner-up fallback and audit).
                if candidate_report_dir:
                    Path(candidate_report_dir).mkdir(parents=True, exist_ok=True)
                    for ci, cand in enumerate(candidates, 1):
                        rp = Path(candidate_report_dir) / f"candidate_{ci}.html"
                        try:
                            HTMLReportGenerator(self.state).generate_single_plan(
                                cand, str(rp), title=f"Plan Candidate {ci}")
                            bestofn_reports.append(str(rp))
                        except Exception as e:
                            logging.warning(f"Candidate report {ci} failed: {e}")
            else:
                print("  - ℹ️  Only one distinct candidate produced — "
                      "proceeding as a single-plan run.")
            bestofn_candidates = candidates
            res = copy.deepcopy(candidates[(bestofn_selected or 1) - 1])
            self.state["plan_candidates"] = {
                "candidates": bestofn_candidates,
                "judge": bestofn_judge,
                "selected_index": bestofn_selected or 1,
                "human_override": False,
                "tier": tier,
                "profile": selection_profile,
                "reports": bestofn_reports,
            }
        else:
            # The portfolio OUTPUT contract rides the campaign, not the
            # best-of-N knob. It used to be injected only when
            # `selection_profile == "ideation"`, which the tool documents as
            # best-of-N only — so every single-plan follow-up in an ideation
            # campaign authored without it. Live: a consolidation delegation
            # then encoded its portfolio as 56 `experimental_steps` document
            # sections, while the best-of-N delegations either side of it
            # emitted clean `concepts` lists. Passed on both tiers, since a
            # fallback run reverts to cramming otherwise.
            _ideation_out = (selection_profile == "ideation"
                             or self._is_ideation_campaign())
            if _portfolio_run:
                res, author_context = author_portfolio(
                    objective=objective,
                    kb_docs=self.kb_docs,
                    model=self.model,
                    generation_config=self.generation_config,
                    primary_data_set=primary_data_set,
                    image_paths=all_image_paths,
                    image_descriptions=image_descriptions,
                    additional_context=ctx_string,
                    external_context=external_context,
                    skill_context=skill_planning_context,
                )
            else:
                res, author_context = perform_science_rag(
                    objective=objective,
                    instructions=(HYPOTHESIS_GENERATION_INSTRUCTIONS
                                  + (IDEATION_OUTPUT_RULES if _ideation_out else "")),
                    fallback_instructions=(
                        HYPOTHESIS_GENERATION_INSTRUCTIONS_FALLBACK
                        + IDEATION_OUTPUT_RULES) if _ideation_out else None,
                    task_name="Experimental Plan",
                    kb_docs=self.kb_docs,
                    model=self.model,
                    generation_config=self.generation_config,
                    primary_data_set=primary_data_set,
                    image_paths=all_image_paths,
                    image_descriptions=image_descriptions,
                    additional_context=ctx_string,
                    external_context=external_context,
                    skill_context=skill_planning_context,
                    return_context=True
                )

        if external_context:
            res["literature_search"] = external_context

        # Plan-kind stamp, mirroring TEA's `type="technoeconomic_analysis"`:
        # a plan dict read on its own (restored checkpoint, a delegation's
        # plan.json) can then tell what it is without the session state that
        # produced it. Only a real ideation run is stamped — the profile is
        # documented as a no-op on the single-plan path.
        _ideation_run = (selection_profile == "ideation" and n_candidates > 1)
        if _ideation_run or _portfolio_run:
            res["type"] = "ideation"
            self.state["plan_kind"] = "ideation"

        # A portfolio carries BOTH shapes through the transition: `directions`
        # is the payload, and a one-entry shim keeps the fifty-odd legacy
        # readers of `proposed_experiments` correct rather than empty-handed —
        # the validity gates especially, where a missing key reads as a FAILED
        # plan and aborts the run.
        if _portfolio_run and isinstance(res, dict) and not res.get("error"):
            res = portfolio_to_experiment_shim(res)

        self._log_action(
            action=("generate_plan_candidates" if n_candidates > 1
                    else "perform_science_rag"),
            input_ctx={
                "objective": objective,
                "knowledge_paths": knowledge_paths,
                "has_primary_data": primary_data_set is not None,
                "has_external_context": bool(external_context),
                **({"n_candidates": n_candidates,
                    "n_produced": len(bestofn_candidates),
                    "judge_pick": bestofn_selected}
                   if n_candidates > 1 else {})
            },
            result=res,
            rationale=res.get("proposed_experiments", [{}])[0].get("justification") if res.get("proposed_experiments") else None
        )
        
        # Snapshot 1: Science Draft
        res["iteration"] = current_iter
        res["stage"] = "Science Draft"
        self.state["plan_history"].append(self._stamp_campaign(res).copy())
        self.state["current_plan"] = res
        
        def _conform_and_critique(res):
            # 1) Objective-conformance check (enforcing). A non-conforming plan is
            # automatically adjusted — the proven self-correction loop, unchanged.
            if not res.get("error"):
                is_relevant, critique = verify_plan_relevance(objective, res, self.model, self.generation_config)

                if not is_relevant:
                    # The full critique was just printed by the verifier
                    # ('Plan Verification Failed: ...') — announce the action
                    # only, or the same paragraph shows twice back to back.
                    print("\n🔄 Self-correction triggered — regenerating the "
                          "plan against the verification failure above.")
                    res = refine_plan_with_feedback(
                        original_result=res,
                        feedback=f"CRITICAL: {critique}",
                        objective=objective,
                        model=self.model,
                        generation_config=self.generation_config,
                        skill_context=skill_planning_context
                    )

                    res["iteration"] = current_iter
                    res["stage"] = "Auto-Corrected"
                    self.state["plan_history"].append(self._stamp_campaign(res).copy())
                    self.state["current_plan"] = res

                    self._log_action(
                        action="self_correction",
                        input_ctx={"critique": critique},
                        result=res,
                        rationale=f"Auto-corrected due to: {critique}"
                    )

            # 2) Advisory critic (separate call, AFTER conformance + any adjustment).
            # Checks PHYSICAL REALISM and INTERNAL CONSISTENCY of the final plan and
            # records caveats — it NEVER rewrites the plan. Findings surface as
            # "Caveats & Potential Limitations" for the human (CO_PILOT / AUTOPILOT)
            # at the feedback prompt, or as run_task `warnings` (AUTONOMOUS / meta).
            # Auto-applying critic findings was shown to rescope plans unreliably, so
            # acting on them is left to an explicit human/consumer decision.
            if not res.get("error"):
                verdict = critique_plan(
                    objective, res, self.model, self.generation_config,
                    retrieved_context=author_context.get("retrieved_context"),
                    primary_data=author_context.get("primary_data"),
                    images=all_image_paths or None,
                    image_descriptions=image_descriptions,
                    additional_context=ctx_string,
                    skill_context=skill_planning_context,
                )
                findings = verdict.get("findings", [])
                if findings:
                    # Order critical-first so the caveats list and warnings lead with
                    # the most material concerns. Record on the plan (no rewrite).
                    _order = {"critical": 0, "minor": 1}
                    findings = sorted(findings, key=lambda f: _order.get(f.get("severity"), 1))
                    res["critic_findings"] = findings
                    self.state["current_plan"] = res
                    # Stamp onto the latest history snapshot too so the HTML report
                    # (which renders plan_history, not current_plan) shows the caveats.
                    if self.state.get("plan_history"):
                        self.state["plan_history"][-1]["critic_findings"] = findings
                    n_crit = sum(1 for f in findings if f.get("severity") == "critical")
                    print(f"\n⚠️  Critic noted {len(findings)} caveat(s)"
                          f"{f' ({n_crit} significant)' if n_crit else ''} "
                          "— recorded under Caveats & Potential Limitations (plan unchanged).")
                    self._log_action(
                        action="critic_review",
                        input_ctx={"findings": findings},
                        result=res,
                        rationale="Advisory caveats recorded; plan not modified."
                    )
            return res

        res = _conform_and_critique(res)

        # Stage-1 best-of-N selection: candidate cards + judge pick + the
        # pick's caveats, then accept-or-override. Selection only — free-text
        # refinement stays with the stage-2 prompt below, on whichever
        # candidate wins here. The critic stays advisory throughout: caveats
        # are DISPLAYED at this prompt, never acted on automatically, and a
        # critical finding never auto-switches the selection.
        if (bestofn_candidates and len(bestofn_candidates) > 1
                and enable_human_feedback and not res.get("error")):
            display_plan_candidates(
                bestofn_candidates, bestofn_judge or {}, bestofn_selected,
                report_paths=bestofn_reports,
                pick_caveats=format_caveats(res.get("critic_findings")),
            )
            choice = get_candidate_selection(len(bestofn_candidates), bestofn_selected)
            if choice != bestofn_selected:
                print(f"  - 👤 Human override: Candidate {choice} "
                      f"(judge picked {bestofn_selected}).")
                self.state["plan_candidates"]["selected_index"] = choice
                self.state["plan_candidates"]["human_override"] = True
                res = copy.deepcopy(bestofn_candidates[choice - 1])
                if external_context:
                    res["literature_search"] = external_context
                res["iteration"] = current_iter
                res["stage"] = "Science Draft (human-selected candidate)"
                self.state["plan_history"].append(self._stamp_campaign(res).copy())
                self.state["current_plan"] = res
                self._log_action(
                    action="bestofn_human_override",
                    input_ctx={"judge_pick": bestofn_selected, "human_pick": choice},
                    result=res,
                    rationale="Human overrode the judge's candidate selection."
                )
                bestofn_selected = choice
                # Lazy-critique invariant: every plan that becomes
                # current_plan has been conformance-checked and critiqued.
                res = _conform_and_critique(res)

        # Human feedback on strategy
        human_feedback = None
        if enable_human_feedback and res.get("proposed_experiments") and not res.get("error"):
            display_plan_summary(res, ideation=self._is_ideation_campaign(),
                                 report_path=self._review_preview_path())
            human_feedback = get_user_feedback()
            
            if human_feedback:
                # Snapshot the pre-refinement plan + its caveats. The re-critique
                # below gets the full before/criticism/request/after picture and
                # reasons about the revised plan itself — the human may address the
                # caveats, change something unrelated, or override the critic.
                prior_plan = res.copy()
                prior_findings = res.get("critic_findings")
                print(f"\n📝 Refining plan...")
                self.state["human_feedback_history"].append({"phase": "science", "feedback": human_feedback})
                refined = refine_plan_with_feedback(
                    original_result=res,
                    feedback=human_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config,
                    skill_context=skill_planning_context
                )

                if refined.get("error"):
                    print(f"⚠️  Refinement failed: {refined.get('message', 'unknown error')}")
                    print("    Keeping original plan.")
                else:
                    res = refined
                    res["iteration"] = current_iter
                    res["stage"] = "Human Refined (Science)"

                    # Re-critique the refined plan so the caveats describe the
                    # CURRENT plan, not the pre-refinement one. Pass prior_findings
                    # so it acts as a resolution check (confirm fixes, drop resolved
                    # ones, flag anything new) rather than re-deriving blindly.
                    res.pop("critic_findings", None)  # discard any echoed-stale caveats
                    verdict = critique_plan(
                        objective, res, self.model, self.generation_config,
                        retrieved_context=author_context.get("retrieved_context"),
                        primary_data=author_context.get("primary_data"),
                        images=all_image_paths or None,
                        image_descriptions=image_descriptions,
                        additional_context=ctx_string,
                        skill_context=skill_planning_context,
                        prior_plan=prior_plan,
                        prior_findings=prior_findings,
                        human_feedback=human_feedback,
                    )
                    fresh = verdict.get("findings", [])
                    if fresh:
                        _order = {"critical": 0, "minor": 1}
                        fresh = sorted(fresh, key=lambda f: _order.get(f.get("severity"), 1))
                        res["critic_findings"] = fresh

                    self.state["plan_history"].append(self._stamp_campaign(res).copy())
                    self.state["current_plan"] = res
                    display_plan_summary(res, ideation=self._is_ideation_campaign(),
                                 report_path=self._review_preview_path())
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

        # Final provenance stamp — conformance/critic/feedback passes above
        # may have re-emitted the plan JSON over the earlier stamp.
        self._finalize_literature(res, external_context)

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
        if not self.kb_code.chunks:   # keyword-only KBs count as content
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

        # Build skill context for implementation
        skill_impl_context = self._build_skill_context("implementation")

        res = perform_code_rag(
            result=plan,
            kb_code=self.kb_code,
            model=self.model,
            generation_config=self.generation_config,
            skill_context=skill_impl_context
        )
        
        # Snapshot: Code Generated
        res["iteration"] = current_iter
        res["stage"] = "Code Generated"
        self.state["plan_history"].append(self._stamp_campaign(res).copy())
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
                    self.state["plan_history"].append(self._stamp_campaign(res).copy())
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
                            reset_state: bool = False,
                            skill: Optional[str] = None) -> Dict[str, Any]:
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
                - current_plan: The active experimental plan dict with
                  proposed_experiments (and implementation_code if code_paths given)
                - plan_history: List of all plan snapshots across stages
                - experimental_results: List of result entries from iterations
                - human_feedback_history: List of feedback entries by phase
                - status: Current status ("planned", "failed", etc.)
                - action_history: Audit log of all agent actions
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
            reset_state=reset_state,
            skill=skill
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
    
    def _recritique_revision(self,
                             new_plan: Dict[str, Any],
                             prior_plan: Dict[str, Any],
                             revision_request: str,
                             skill_context: Optional[str] = None,
                             images: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Resolution-check re-critique after a plan revision, so the recorded
        caveats describe the CURRENT plan — the refinement LLM echoes the old
        ``critic_findings`` into the revised JSON, which otherwise persists
        stale (possibly already-fixed) caveats into reports and warnings.

        Pops the echoed caveats, then re-runs the advisory critic with the
        before/request/after picture (drop resolved, keep unresolved, flag
        new). Annotation only — never modifies the plan and never triggers
        further action, so the advisory-only contract is untouched. Fails
        open like the critic itself. Evidence here is the revision context
        (prior plan + findings + request); the original retrieval context is
        not persisted across calls.
        """
        if new_plan.get("error") or not new_plan.get("proposed_experiments"):
            return new_plan

        prior_findings = prior_plan.get("critic_findings")
        new_plan.pop("critic_findings", None)  # discard echoed-stale caveats

        try:
            verdict = critique_plan(
                self.state["objective"], new_plan, self.model,
                self.generation_config,
                images=images,
                skill_context=skill_context,
                prior_plan=prior_plan,
                prior_findings=prior_findings,
                human_feedback=revision_request,
            )
        except Exception as e:
            # critique_plan fails open internally; this guard keeps the
            # revision alive even if the call itself dies.
            logging.error(f"Revision re-critique failed open: {e}")
            verdict = {"findings": []}
        findings = verdict.get("findings", [])
        if findings:
            _order = {"critical": 0, "minor": 1}
            findings = sorted(findings,
                              key=lambda f: _order.get(f.get("severity"), 1))
            new_plan["critic_findings"] = findings
            if self.state.get("plan_history"):
                self.state["plan_history"][-1]["critic_findings"] = findings
        elif self.state.get("plan_history"):
            # the snapshot appended before this call carries the echoed copy
            self.state["plan_history"][-1].pop("critic_findings", None)

        self.state["current_plan"] = new_plan
        return new_plan

    def refine_plan(self,
                    results: Any,
                    enable_human_feedback: bool = True,
                    state_file_path: Optional[str] = None,
                    use_literature_rag: bool = False,
                    external_context: Optional[str] = None,
                    literature_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Refines the experimental plan (science strategy only) based on new results.

        Args:
            results: Experimental outcomes (text, dict, file path, or list of files/images)
            enable_human_feedback: If True, pauses for strategy review
            state_file_path: Optional path to restore state from checkpoint
            use_literature_rag: If True, searches knowledge base for context relevant
                           to the results. Defaults to False for faster iteration.
            external_context: Pre-fetched external context (e.g. from
                orchestrator's search_literature/query_molecules tools).
                Merged with any local KB hits from use_literature_rag.
            literature_text: The LITERATURE-ONLY portion of the external
                context, when the caller can separate it. Used to stamp the
                refined plan's ``literature_search`` provenance; without it,
                the prior plan's literature carries forward.

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
        
        # Update State History — store the parsed content so knowledge
        # synthesis can access full experimental outcomes without relying
        # on chat history (which may be compressed in long campaigns).
        self.state["experimental_results"].append({
            "iteration": executed_plan_idx,
            "timestamp": datetime.now().isoformat(),
            "data_summary": consolidated_feedback,
            "raw_input": str(results)
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
        
        # --- 3. RESULT-AWARE CONTEXT ---
        context_parts = []

        # External context from orchestrator tools (literature/molecules)
        if external_context:
            context_parts.append(external_context)
            print(f"  - 📚 Using external context ({len(external_context)} chars)")

        # Local KB RAG (optional, additive)
        if use_literature_rag:
            if self.kb_docs.chunks:   # keyword-only KBs count as content
                search_query = f"Implications and causes of: {consolidated_feedback[:400]}"
                print(f"  - 🔍 Searching local KB for context on results...")
                try:
                    hits = self.kb_docs.retrieve(search_query, top_k=3)
                except Exception as e:  # noqa: BLE001 - degrade, never kill refinement
                    try:
                        hits = self.kb_docs.retrieve_sparse(search_query, top_k=3)
                        logging.warning(
                            f"Dense KB retrieval failed ({e}); refining with "
                            "keyword (BM25) fallback context."
                        )
                    except Exception:  # noqa: BLE001
                        logging.warning(
                            f"KB retrieval failed ({e}); refining without "
                            "local KB context."
                        )
                        hits = []
                if hits:
                    context_parts.append("\n---\n".join([c['text'] for c in hits]))
                    print(f"    -> Found {len(hits)} relevant document chunks.")
                else:
                    print(f"    -> No relevant documents found.")
            else:
                print(f"  - ℹ️  Literature RAG requested but no docs KB available.")
        elif not external_context:
            print(f"  - ℹ️  No external context or literature RAG for refinement")

        new_literature_context = "\n\n".join(context_parts) if context_parts else None
        
        # --- 4. GENERATE REFINED PLAN ---
        if new_literature_context:
            print(f"  - Reasoning over results with literature context...")
        else:
            print(f"  - Reasoning over results...")

        # Build skill context for refinement (interpretation + validation)
        skill_refine_context = self._build_skill_context("interpretation")

        new_plan = refine_plan_with_feedback(
            original_result=current_plan,
            feedback=feedback_prompt,
            objective=objective,
            model=self.model,
            generation_config=self.generation_config,
            new_context=new_literature_context,
            result_images=loaded_images,
            skill_context=skill_refine_context
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
        
        # literature_search is SYSTEM-OWNED provenance: the refining LLM
        # sometimes authors a prose note into that field ('no new search was
        # executed...'), which then shadows the campaign's real corpus for
        # every downstream consumer. Stamp it here — the one place the
        # refined plan enters both current_plan and history — with the
        # literature actually supplied this round, else carry the prior
        # plan's literature forward; the model's value survives only if it
        # is genuinely the most substantial (i.e. it faithfully copied the
        # corpus).
        _lit_candidates = [str(new_plan.get("literature_search") or ""),
                           str(literature_text or ""),
                           str(current_plan.get("literature_search") or "")]
        _best_lit = max(_lit_candidates, key=len)
        if _best_lit:
            new_plan["literature_search"] = _best_lit

        # Snapshot: Reasoning Draft
        new_plan["iteration"] = next_plan_idx
        new_plan["stage"] = "Reasoning Draft"
        self.state["plan_history"].append(self._stamp_campaign(new_plan).copy())
        self.state["current_plan"] = new_plan

        # Resolution-check re-critique so caveats describe the revised plan.
        new_plan = self._recritique_revision(
            new_plan, prior_plan=current_plan,
            revision_request=("Plan revised in response to experimental "
                             f"results:\n{consolidated_feedback[:2000]}"),
            skill_context=skill_refine_context,
            images=loaded_images or None,
        )

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
            display_plan_summary(new_plan, ideation=self._is_ideation_campaign(),
                                 report_path=self._review_preview_path())
            
            human_feedback = get_user_feedback()
            
            if human_feedback: 
                print(f"\n📝 Feedback received. Adjusting strategy...")
                self.state["human_feedback_history"].append({
                    "phase": "science_iteration", 
                    "feedback": human_feedback
                })
                prior_plan = new_plan
                new_plan = refine_plan_with_feedback(
                    original_result=new_plan,
                    feedback=human_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config,
                    skill_context=skill_refine_context
                )
                # Snapshot: Human Refined
                new_plan["iteration"] = next_plan_idx
                new_plan["stage"] = "Human Refined (Science)"
                self.state["plan_history"].append(self._stamp_campaign(new_plan).copy())
                self.state["current_plan"] = new_plan
                new_plan = self._recritique_revision(
                    new_plan, prior_plan=prior_plan,
                    revision_request=human_feedback,
                    skill_context=skill_refine_context,
                )
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
        self._finalize_literature(new_plan, _best_lit)
        return new_plan
    
    def adjust_plan_for_constraints(self,
                                    constraint_description: str,
                                    enable_human_feedback: bool = True) -> Dict[str, Any]:
        """
        Adjusts the experimental plan to accommodate implementation or
        instrument constraints discovered during protocol/code generation.

        Unlike refine_plan(), this does NOT increment the iteration counter
        or log as experimental results — the experiment hasn't run yet.

        Args:
            constraint_description: Description of the constraint or
                incompatibility that requires plan adjustment.
            enable_human_feedback: If True, pauses for user review.

        Returns:
            Updated plan dict with the same JSON structure.
        """
        if not self.state or not self.state.get("current_plan"):
            raise ValueError(
                "No active plan to adjust. Generate a plan first."
            )

        print(f"\n--- 🔧 Adjusting Plan for Implementation Constraints ---")

        objective = self.state["objective"]
        current_plan = self.state["current_plan"]

        adjustment_prompt = (
            f"An implementation constraint was discovered BEFORE running the experiment. "
            f"The experiment has NOT been executed yet.\n\n"
            f"**Constraint / Incompatibility:**\n{constraint_description}\n\n"
            f"**TASK:** Make the MINIMUM changes necessary to accommodate this constraint.\n"
            f"- ONLY modify the specific parts of the plan affected by the constraint.\n"
            f"- Do NOT change the scientific hypothesis, objective, or rationale.\n"
            f"- Do NOT modify experimental parameters, conditions, or steps that are "
            f"unrelated to the constraint.\n"
            f"- Keep all unaffected experiments, steps, and details EXACTLY as they are.\n"
            f"- If the constraint forces a scope reduction, explain the trade-off in the "
            f"justification but do not redesign unaffected parts of the plan.\n"
            f"- Do NOT treat this as experimental results — no experiment was run."
        )

        # Build skill context for constraint adjustment
        skill_constraint_context = self._build_skill_context("planning")

        print(f"  - Reasoning over constraint...")
        new_plan = refine_plan_with_feedback(
            original_result=current_plan,
            feedback=adjustment_prompt,
            objective=objective,
            model=self.model,
            generation_config=self.generation_config,
            skill_context=skill_constraint_context
        )

        if new_plan.get("error"):
            print(f"\n❌ Adjustment Failed: {new_plan.get('message')}")
            self._log_action(
                action="adjust_plan_for_constraints",
                input_ctx={"constraint": constraint_description[:200]},
                result=new_plan,
                rationale=None
            )
            return new_plan

        # Keep same iteration — this is an in-place adjustment, not a new cycle
        new_plan["iteration"] = current_plan.get("iteration", 0)
        new_plan["stage"] = "Constraint Adjusted"
        self.state["plan_history"].append(self._stamp_campaign(new_plan).copy())
        self.state["current_plan"] = new_plan

        # Resolution-check re-critique: a constraint adjustment often resolves
        # a recorded caveat — the caveat channel must reflect the CURRENT plan.
        new_plan = self._recritique_revision(
            new_plan, prior_plan=current_plan,
            revision_request=constraint_description,
            skill_context=skill_constraint_context,
        )

        self._log_action(
            action="adjust_plan_for_constraints",
            input_ctx={"constraint": constraint_description[:200]},
            result=new_plan,
            rationale=constraint_description
        )

        # Human feedback
        human_feedback = None
        if enable_human_feedback and not new_plan.get("error"):
            print("\n" + "=" * 60)
            print("🔧 AGENT'S PROPOSED PLAN ADJUSTMENT (Constraint)")
            print("=" * 60)
            display_plan_summary(new_plan, ideation=self._is_ideation_campaign(),
                                 report_path=self._review_preview_path())

            human_feedback = get_user_feedback()

            if human_feedback:
                print(f"\n📝 Feedback received. Adjusting...")
                self.state["human_feedback_history"].append({
                    "phase": "constraint_adjustment",
                    "feedback": human_feedback
                })
                prior_plan = new_plan
                new_plan = refine_plan_with_feedback(
                    original_result=new_plan,
                    feedback=human_feedback,
                    objective=objective,
                    model=self.model,
                    generation_config=self.generation_config,
                    skill_context=skill_constraint_context
                )
                new_plan["iteration"] = current_plan.get("iteration", 0)
                new_plan["stage"] = "Human Refined (Constraint)"
                self.state["plan_history"].append(self._stamp_campaign(new_plan).copy())
                self.state["current_plan"] = new_plan
                new_plan = self._recritique_revision(
                    new_plan, prior_plan=prior_plan,
                    revision_request=human_feedback,
                    skill_context=skill_constraint_context,
                )
                print("✅ Constraint adjustment updated.")

        self.state["status"] = "constraint_adjusted"
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
        
        if not self.kb_code.chunks:   # keyword-only KBs count as content
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
        self.state["plan_history"].append(self._stamp_campaign(new_plan).copy())
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
                    self.state["plan_history"].append(self._stamp_campaign(new_plan).copy())
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
                                        output_json_path: Optional[str] = None,
                                        external_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Performs technoeconomic analysis (TEA) using Dual-KB retrieval.

        **Workflow:**

        1. Knowledge Base Construction (if needed)
        2. External Literature Search (optional, via FutureHouse)
        3. RAG-based Economic Analysis
        4. State Initialization (if starting fresh with TEA)
        5. Report Generation (JSON + HTML)

        **Integration with Planning:**

        TEA results are stored in the agent's state and can inform subsequent
        experimental planning:
            >>> tea_results = agent.perform_technoeconomic_analysis(
            ...     objective="Recover lithium from brine",
            ...     knowledge_paths=["./market_data/", "./reports/"],
            ... )
            >>> plan = agent.propose_experiments(
            ...     objective="Develop lithium extraction process",
            ...     knowledge_paths=["./extraction_methods/"],
            ...     additional_context=tea_results,
            ...     primary_data_set={"file_path": "./brine_composition.xlsx"},
            ... )

        Args:
            objective: Research objective to evaluate economically.
                Should describe the material, process, or technology to assess.
                Examples:
                    - "Recover rare earth elements from coal ash"
                    - "Evaluate magnesium extraction from produced water"
                    - "Assess economic viability of direct air capture"

            knowledge_paths: Paths to documents for TEA context.
                Should include market data, pricing reports, criticality assessments,
                existing TEA studies, and process descriptions. Supports PDF/TXT
                and Excel/CSV.
                Example: ["./market_reports/", "./critical_materials_report.pdf"]

            primary_data_set: Main dataset for analysis.
                Can contain composition, concentration, or yield data.
                Example: {"file_path": "./feedstock_composition.xlsx"}

            image_paths: Images to support TEA analysis.
                Examples: criticality matrices, supply chain diagrams, cost breakdowns.

            image_descriptions: Descriptions for each image.
                Example: ["Criticality matrix showing supply risk vs. importance"]

            output_json_path: Path to save TEA results.
                Saves results to {output_json_path}, state to
                {output_json_path}.state.json, and HTML report to
                {output_json_path}.html.

        Returns:
            Dict[str, Any]: Technoeconomic analysis results containing
                cost breakdowns, market analysis, and economic feasibility
                assessment. Structure mirrors the plan dict with
                proposed_experiments replaced by economic analysis sections.

        Example - Basic TEA:
            >>> agent = PlanningAgent()
            >>> tea = agent.perform_technoeconomic_analysis(
            ...     objective="Recover rare earth elements from coal ash",
            ...     knowledge_paths=["./market_reports/", "./process_data/"],
            ...     output_json_path="./tea_results.json"
            ... )

        Example - TEA with Data and Images:
            >>> tea = agent.perform_technoeconomic_analysis(
            ...     objective="Evaluate magnesium extraction from produced water",
            ...     knowledge_paths=["./reports/", "./pricing.xlsx"],
            ...     primary_data_set={"file_path": "./brine_composition.xlsx"},
            ...     image_paths=["./criticality_matrix.png"],
            ...     image_descriptions=["Material criticality assessment"],
            ...     output_json_path="./mg_tea.json"
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
        
        # 3. External context is caller-provided only (orchestrator's
        # search_literature economic_data type is the sanctioned path; the
        # silent internal fallback was removed with the one in generate_plan).
        lit_context = external_context or ""

        # 4. Perform RAG
        # Build skill context for TEA (overview section if relevant)
        skill_tea_context = self._build_skill_context("overview")

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
            external_context=lit_context,
            skill_context=skill_tea_context
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
            self.state["plan_history"].append(self._stamp_campaign(res).copy())
     
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