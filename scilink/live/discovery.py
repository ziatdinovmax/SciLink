"""From "the data changed" to "is this new?": the slow half of a discovery.

The loop's ``novelty`` event is statistical: this frame is unlike the stream so
far, here, with no model. Whether it is scientifically new is the question of
the first SciLink pipeline (arXiv:2508.06569): observation -> falsifiable claims
-> novelty scored against the literature -> what to measure or compute next.
That chain is slow-clock work. A pause is what buys the time for it, while the
sample is still in the state that looked new; it also runs without one, on the
frame the event names.

``assess_change`` is that chain for one frame:

1. an analysis of the frame, TOLD what changed and where (the loop's own
   description, from the data alone), that returns scientific claims;
2. for each claim, a literature search and a novelty score from 1 (well known)
   to 5 (not found), when a literature key is available. Without one the claims
   are returned as they are and the result says the literature was not asked.

Nothing here is on a frame's path, and nothing here imports the server, chat or
an orchestrator: the analysis agent comes from the modality, the literature
agents are the package's own.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def assess_change(data_path: str, *, modality: Any, system_info: Any, what_changed: Optional[str],
                  agent_kwargs: Dict[str, Any], out_dir: str, profile: str = "quick",
                  futurehouse_api_key: Optional[str] = None, max_claims: int = 3,
                  agent_factory: Optional[Callable[[str], Any]] = None,
                  literature: Any = None, scorer: Any = None,
                  logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Claims about a changed frame, and how new each is. Never raises.

    ``literature`` / ``scorer`` are injectable (tests, another search backend);
    by default they are ``OwlLiteratureAgent`` and ``NoveltyScorer``."""
    log = logger or logging.getLogger("scilink.live.discovery")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result: Dict[str, Any] = {"status": "error", "data": str(data_path), "claims": [],
                              "literature": "not asked"}
    try:
        from types import SimpleNamespace
        from .measurement_loop import with_sidecar
        from .modality import resolve_modality
        modality = resolve_modality(modality)
        shim = SimpleNamespace(api_key=agent_kwargs.get("api_key"), model_name=agent_kwargs.get("model_name"),
                               base_url=agent_kwargs.get("base_url"), _human_feedback=False)
        agent = (agent_factory(str(out / "analysis")) if agent_factory is not None
                 else modality.make_agent(shim, str(out / "analysis")))
        kwargs: Dict[str, Any] = {"system_info": with_sidecar(system_info, data_path), "profile": profile}
        if what_changed:
            kwargs["hints"] = what_changed
        res = agent.analyze(str(data_path), **kwargs) or {}
        result["analysis_status"] = res.get("status")
        result["analysis_dir"] = res.get("output_directory")
        result["summary"] = str(res.get("detailed_analysis") or "")[:1500]
        result["llm_calls"] = (res.get("stage_timings") or {}).get("llm_calls")
        claims = [c for c in (res.get("scientific_claims") or []) if isinstance(c, dict) and c.get("claim")]
        result["claims"] = [{"claim": c.get("claim"), "question": c.get("has_anyone_question"),
                             "impact": c.get("scientific_impact")} for c in claims[:max_claims]]
        result["status"] = "success" if claims else "no_claims"
    except Exception as e:  # noqa: BLE001 - reported, never raised
        result["error"] = f"{type(e).__name__}: {e}"
        log.warning(f"the analysis of the changed frame failed: {e}")
    if result["claims"] and (futurehouse_api_key or literature is not None):
        try:
            if literature is None:
                from ..agents.lit_agents import OwlLiteratureAgent
                literature = OwlLiteratureAgent(api_key=futurehouse_api_key, max_wait_time=600)
            if scorer is None:
                from ..agents.lit_agents import NoveltyScorer
                scorer = NoveltyScorer(api_key=agent_kwargs.get("api_key"),
                                       model_name=agent_kwargs.get("model_name"),
                                       base_url=agent_kwargs.get("base_url"))
            asked = 0
            for c in result["claims"]:
                if not c.get("question"):
                    continue
                found = literature.query_literature(c["question"]) or {}
                if found.get("status") != "success":
                    c["literature"] = "the search failed"
                    continue
                score = scorer.score_novelty(c["question"], found.get("formatted_answer", "")) or {}
                c["novelty_score"] = score.get("novelty_score")
                c["novelty_explanation"] = str(score.get("explanation") or "")[:600]
                asked += 1
            result["literature"] = f"asked for {asked} claim(s)"
            scores = [c["novelty_score"] for c in result["claims"] if isinstance(c.get("novelty_score"), (int, float))]
            if scores:
                result["highest_novelty"] = max(scores)
        except Exception as e:  # noqa: BLE001
            result["literature"] = f"failed: {type(e).__name__}: {e}"
    elif result["claims"]:
        result["literature"] = "not asked: no literature key (FutureHouse) was given"
    result["seconds"] = round(time.perf_counter() - t0, 1)
    try:
        (out / "discovery.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass
    return result


def describe(result: Dict[str, Any]) -> List[str]:
    """One line per claim, for a log or a card."""
    lines = []
    for c in result.get("claims") or []:
        score = c.get("novelty_score")
        lines.append((f"[novelty {score}/5] " if isinstance(score, (int, float)) else "") + str(c.get("claim")))
    return lines
