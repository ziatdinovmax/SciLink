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
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def assess_change(data_path: str, *, modality: Any, system_info: Any, what_changed: Optional[str],
                  agent_kwargs: Dict[str, Any], out_dir: str, profile: str = "quick",
                  futurehouse_api_key: Optional[str] = None, max_claims: int = 3,
                  agent_factory: Optional[Callable[[str], Any]] = None,
                  literature: Any = None, scorer: Any = None,
                  analysis_kwargs: Optional[Dict[str, Any]] = None,
                  logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Claims about a changed frame, and how new each is. Never raises.

    ``literature`` / ``scorer`` are injectable (tests, another search backend);
    by default they are ``OwlLiteratureAgent`` and ``NoveltyScorer``.
    ``analysis_kwargs`` go to the analysis as they are (the loop uses it to ask
    for the tracked quantities by name, so the result's ``features`` can be put
    beside the recipe's)."""
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
        kwargs: Dict[str, Any] = {**(analysis_kwargs or {}),
                                  "system_info": with_sidecar(system_info, data_path)}
        if what_changed:
            kwargs["hints"] = what_changed
        usable = tuple(getattr(modality, "usable_status", ("success",)))
        res: Dict[str, Any] = {}
        # An analysis that fails leaves claims made from LOOKING at the frame and
        # no numbers (seen live: a quick image script that raised, one attempt).
        # Like an audit that could not be formed, it is tried once more, deeper.
        for attempt, depth in enumerate(dict.fromkeys([profile, _deeper(profile)])):
            if depth is None:
                continue
            where = str(out / ("analysis" if attempt == 0 else "analysis_retry"))
            agent = agent_factory(where) if agent_factory is not None else modality.make_agent(shim, where)
            res = agent.analyze(str(data_path), **{**kwargs, "profile": depth}) or {}
            result["profile"] = depth
            if res.get("status") in usable:
                break
            result["retried"] = True
        measured = res.get("status") in usable
        result["analysis_status"] = res.get("status")
        result["analysis_dir"] = res.get("output_directory")
        result["summary"] = str(res.get("detailed_analysis") or "")[:1500]
        result["llm_calls"] = (res.get("stage_timings") or {}).get("llm_calls")
        try:
            result["features"] = ({k: float(v) for k, v in (modality.features(res) or {}).items()}
                                  if measured else {})
        except Exception:  # noqa: BLE001 - the claims stand without the numbers
            result["features"] = {}
        claims = [c for c in (res.get("scientific_claims") or []) if isinstance(c, dict) and c.get("claim")]
        result["claims"] = [{"claim": c.get("claim"), "question": c.get("has_anyone_question"),
                             "impact": c.get("scientific_impact")} for c in claims[:max_claims]]
        # "unmeasured": there are claims, and no analysis behind them finished.
        result["status"] = ("no_claims" if not claims else "success" if measured else "unmeasured")
        if not measured:
            err = res.get("error")
            result["analysis_error"] = str((err or {}).get("error") if isinstance(err, dict) else err or "")[:300]
    except Exception as e:  # noqa: BLE001 - reported, never raised
        result["error"] = f"{type(e).__name__}: {e}"
        log.warning(f"the analysis of the changed frame failed: {e}")
    # The literature agents read FUTUREHOUSE_API_KEY themselves; a key given here wins.
    futurehouse_api_key = futurehouse_api_key or os.environ.get("FUTUREHOUSE_API_KEY") or None
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
        result["literature"] = "not asked: no literature key (FutureHouse, or FUTUREHOUSE_API_KEY) was given"
    result["seconds"] = round(time.perf_counter() - t0, 1)
    try:
        (out / "discovery.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass
    return result


def _deeper(profile: Any) -> Optional[str]:
    from .measurement_loop import MeasurementLoop
    return MeasurementLoop._deeper_profile(profile)


def describe(result: Dict[str, Any]) -> List[str]:
    """One line per claim, for a log or a card."""
    lines = []
    for c in result.get("claims") or []:
        score = c.get("novelty_score")
        lines.append((f"[novelty {score}/5] " if isinstance(score, (int, float)) else "") + str(c.get("claim")))
    return lines
