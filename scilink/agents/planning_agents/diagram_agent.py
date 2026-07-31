"""Workflow-diagram generation with the same discipline as code generation.

Mermaid source is code, so it gets the codegen treatment: the LLM writes
it grounded in the STRUCTURED plan (not free-form prose, so it cannot
invent steps), the renderer is the compile check (render errors feed a
retry), and the rendered image passes a visual quality gate (a vision
call judging legibility and faithfulness) before the diagram is accepted.

Diagrams default to a compact stage-level overview — white papers and
roadmaps stay simple unless the user explicitly asks for an elaborate
diagram (``detail="elaborate"``).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image as PIL_Image

from scilink.knowledge import parse_json_from_response

from ...utils.mermaid_render import render_mermaid, mermaid_available  # noqa: F401
from .base_agent import BaseAgent

_GEN_PROMPT = """You are drawing a workflow diagram for a scientific \
campaign document. Produce ONE Mermaid flowchart (```mermaid fenced block, \
`flowchart TD`) of the experimental campaign below.

Rules:
- {detail_rule}
- Node labels: short plain phrases in double quotes, no code tokens, no
  parentheses or special characters inside labels.
- Decision points are diamonds (`{{"..."}}`); loops (e.g. an
  optimize-measure-refine cycle) are drawn as cycles, not unrolled.
- Stay faithful to the campaign below — do not invent steps that are not
  in it.
- Output ONLY the fenced mermaid block, no commentary.

Campaign (structured):
{plan_json}
{extra}{feedback}"""

_DETAIL_RULES = {
    "simple": ("Compact overview: at most ~10 nodes, stage-level (setup, "
               "the main loop, decision gates, outcome) — NOT every "
               "individual step."),
    "elaborate": ("Detailed view: include the individual experimental "
                  "steps and branch conditions; still keep labels short."),
}

_QC_PROMPT = """You are reviewing a workflow diagram rendered for a \
scientific document. Judge it and answer in JSON only:
{{"approved": true/false, "issues": ["..."]}}

Reject when: text overlaps or is clipped; arrows cross confusingly; the
diagram contradicts the campaign description below; or it is overcrowded
for a document figure ({detail} level was requested — {detail_rule})

Campaign (structured):
{plan_json}"""


class DiagramAgent(BaseAgent):
    """Generate → render → visually verify Mermaid workflow diagrams."""

    def __init__(self, model=None, output_dir: str = "."):
        """``model`` is an already-constructed generative-model wrapper
        (e.g. the planning orchestrator passes its BO agent's) — reusing
        it means no new credential plumbing. For standalone use, build
        one via ``BOAgent(...).model`` or the wrappers directly."""
        super().__init__(output_dir)
        self.agent_type = "diagram"
        self.model = model
        self.generation_config = None

    # ── prompt assembly ─────────────────────────────────────────────

    @staticmethod
    def _plan_json(plan: Dict[str, Any]) -> str:
        """The diagram-relevant slice of the plan, compact."""
        keep: Dict[str, Any] = {}
        for exp in (plan.get("proposed_experiments") or [])[:3]:
            keep.setdefault("experiments", []).append({
                k: exp.get(k) for k in
                ("experiment_name", "hypothesis", "experimental_steps",
                 "expected_outcomes") if exp.get(k)})
        for k in ("objective", "scientific_context", "directions"):
            if plan.get(k):
                keep[k] = plan[k]
        txt = json.dumps(keep, indent=0, default=str)
        return txt[:6000]

    @staticmethod
    def _extract_mermaid(text: str) -> Optional[str]:
        import re
        m = re.search(r"```mermaid\s*(.*?)```", text or "", re.S)
        if m:
            return m.group(1).strip()
        stripped = (text or "").strip()
        if stripped.startswith(("flowchart", "graph ")):
            return stripped
        return None

    # ── main entry ──────────────────────────────────────────────────

    def generate_workflow_diagram(
            self, plan: Dict[str, Any], out_dir=None,
            stem: str = "campaign_workflow", detail: str = "simple",
            extra_instructions: Optional[str] = None,
            max_render_attempts: int = 3,
            max_qc_rounds: int = 2) -> Dict[str, Any]:
        """Returns ``{status, png_path, mmd_path, code, attempts,
        qc_rounds, inspection, error}``. ``status != "success"`` never
        raises — callers ship the document without a diagram."""
        if self.model is None:
            return {"status": "error", "error": "no model client"}
        detail = detail if detail in _DETAIL_RULES else "simple"
        out_dir = Path(out_dir) if out_dir else self.output_dir
        png_path = out_dir / f"{stem}.png"
        mmd_path = out_dir / f"{stem}.mmd"
        plan_json = self._plan_json(plan or {})
        extra = (f"\nAdditional instructions: {extra_instructions}\n"
                 if extra_instructions else "")

        feedback = ""
        code = None
        attempts = 0
        qc_rounds = 0
        inspection: Dict[str, Any] = {}
        while attempts < max_render_attempts:
            attempts += 1
            prompt = _GEN_PROMPT.format(
                detail_rule=_DETAIL_RULES[detail], plan_json=plan_json,
                extra=extra, feedback=feedback)
            try:
                resp = self.model.generate_content(
                    [prompt], generation_config=self.generation_config)
                code = self._extract_mermaid(resp.text)
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "error": f"LLM call failed: {exc}",
                        "attempts": attempts}
            if not code:
                feedback = ("\nYour previous reply contained no mermaid "
                            "block. Output only the fenced block.")
                continue

            ok, err = render_mermaid(code, png_path)
            if not ok:
                print(f"    🔁 Diagram render error (attempt {attempts}): "
                      f"{err.splitlines()[-1] if err else 'unknown'}")
                feedback = (f"\nYour previous diagram failed to render. "
                            f"Renderer error:\n{err}\nFix the syntax and "
                            f"output the corrected fenced block.")
                continue

            # Visual quality gate — same shape as the BO agent's plot
            # inspection. A QC failure regenerates with the issues named;
            # an unavailable QC call accepts the render (gate, not wall).
            inspection = self._inspect(png_path, plan_json, detail)
            if inspection.get("approved", True):
                break
            qc_rounds += 1
            if qc_rounds > max_qc_rounds:
                inspection["note"] = "accepted after exhausting QC rounds"
                break
            issues = "; ".join(inspection.get("issues") or [])[:600]
            print(f"    👀 Diagram QC requests changes: {issues}")
            feedback = (f"\nYour previous diagram rendered but failed "
                        f"visual review: {issues}\nRedraw it fixing these "
                        f"issues. Output only the fenced block.")
        else:
            return {"status": "error",
                    "error": "no valid render within attempt budget",
                    "attempts": attempts}

        mmd_path.write_text(code)
        return {"status": "success", "png_path": str(png_path),
                "mmd_path": str(mmd_path), "code": code,
                "attempts": attempts, "qc_rounds": qc_rounds,
                "inspection": inspection}

    def _inspect(self, png_path: Path, plan_json: str,
                 detail: str) -> Dict[str, Any]:
        try:
            img = PIL_Image.open(png_path)
            prompt = _QC_PROMPT.format(
                plan_json=plan_json, detail=detail,
                detail_rule=_DETAIL_RULES[detail])
            resp = self.model.generate_content(
                [prompt, img], generation_config=self.generation_config)
            verdict, _ = parse_json_from_response(resp)
            if isinstance(verdict, dict) and "approved" in verdict:
                return verdict
            return {"approved": True, "note": "unparseable QC reply"}
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"Diagram visual QC skipped: {exc}")
            return {"approved": True, "note": f"qc skipped: {exc}"}
