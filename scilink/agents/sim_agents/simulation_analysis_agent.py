"""The simulation-analysis agent: compute properties from run output.

One engine-neutral agent for all simulation analysis. The property × technique
differentiation lives entirely in skills (domain ``simulation_analysis``), each
declaring — in frontmatter — the property it ``computes`` and the input ``data``
it ``requires``. Selection is **availability-gated**: a skill is eligible only
when the data it requires is present on disk, so a trajectory analysis fires only
with a trajectory, a DFT analysis only with DFT output, and an overlapping
property (elastic constants from MD vs DFT) resolves by what was actually run.
The verified codegen loop that turns data into a number lives in the base.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_analysis_agent import BaseAnalysisAgent


class SimulationAnalysisAgent(BaseAnalysisAgent):
    """Compute the properties a research goal calls for from a run's output.

    Pipeline: classify the output files into data kinds → find the technique
    skills whose required data is present → let the LLM pick which of their
    computable properties the goal actually wants → run each through the base's
    verified codegen loop, guided by the skill's implementation recipe.
    """

    DOMAIN = "simulation_analysis"

    def _output_format_map(self) -> Dict[str, set]:
        """Build ``{data_kind: {patterns}}`` from engine skills' ``outputs:``.

        Engine specifics stay out of this agent: each engine skill declares, in
        frontmatter, which output-file patterns realize which data kind (the
        ``vasp`` skill maps ``vasprun.xml`` -> ``dft_output``, ``lammps`` maps
        ``log.lammps`` -> ``thermo_log``, and so on). This aggregates those
        declarations across every skill, so adding an engine is a skill-only
        change — no filename ever appears here. Cached per instance.
        """
        if getattr(self, "_fmt_map_cache", None) is not None:
            return self._fmt_map_cache
        from ...skills.loader import list_all_skills, load_skill

        fmt: Dict[str, set] = defaultdict(set)
        for domain, names in list_all_skills().items():
            for name in names:
                try:
                    meta = load_skill(name, domain=domain).get("meta") or {}
                except Exception:
                    continue
                outputs = meta.get("outputs")
                if not isinstance(outputs, dict):
                    continue
                for kind, pats in outputs.items():
                    if isinstance(pats, str):
                        pats = [pats]
                    fmt[kind].update(str(p).lower() for p in pats)
        self._fmt_map_cache = dict(fmt)
        return self._fmt_map_cache

    def classify_outputs(self, run_dir: str) -> Dict[str, List[str]]:
        """Map the present data kinds to the files that realize them.

        Returns ``{data_kind: [paths]}`` for every recognized output in
        ``run_dir`` (recursively), using the engine-declared output patterns
        (:meth:`_output_format_map`). Unrecognized files are ignored.
        """
        fmt = self._output_format_map()
        present: Dict[str, List[str]] = defaultdict(list)
        root = Path(run_dir)
        if not root.exists():
            return {}
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name.lower()
            ext = p.suffix.lower().lstrip(".")
            for kind, pats in fmt.items():
                if any(name == pat or name.endswith(pat) or ext == pat
                       for pat in pats):
                    present[kind].append(str(p))
                    break
        return dict(present)

    def _skill_catalog(self) -> List[Dict[str, Any]]:
        """Return the loaded analysis skills (``{name, meta, sections}``).

        Separated so tests can substitute a catalog without on-disk skills.
        """
        from ...skills.loader import list_skills, load_skill

        catalog: List[Dict[str, Any]] = []
        for name in list_skills(domain=self.DOMAIN):
            try:
                catalog.append(load_skill(name, domain=self.DOMAIN))
            except Exception as exc:  # a broken skill must not sink selection
                self.logger.warning("Skill %r failed to load: %s", name, exc)
        return catalog

    def eligible_skills(self, present_kinds, catalog=None) -> List[Dict[str, Any]]:
        """Skills whose required data is all present — the availability gate.

        A skill with ``requires: [trajectory]`` is eligible only when a
        trajectory is present; a skill declaring no ``requires`` is always
        eligible. ``present_kinds`` is the set of available data kinds.
        """
        present = set(present_kinds)
        out = []
        for skill in (catalog if catalog is not None else self._skill_catalog()):
            required = (skill.get("meta") or {}).get("requires") or []
            if isinstance(required, str):
                required = [required]
            if set(required).issubset(present):
                out.append(skill)
        return out

    def _select_properties(self, research_goal: str,
                           eligible: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pick which eligible (property, skill) pairs the goal actually wants.

        Presents the eligible skills' declared ``computes`` to the LLM and asks
        which the goal requires, so a broad run doesn't compute every possible
        property. Returns the chosen skills (subset of ``eligible``).
        """
        if not eligible:
            return []
        options = []
        for s in eligible:
            meta = s.get("meta") or {}
            computes = meta.get("computes") or []
            if isinstance(computes, str):
                computes = [computes]
            options.append({"skill": s["name"], "computes": computes,
                            "technique": meta.get("technique"),
                            "description": meta.get("description", "")})
        import json
        prompt = (
            "Given a research goal and a list of available analysis techniques "
            "(each computes one or more properties), return the names of the "
            "techniques whose properties the goal actually requires — omit the "
            "rest. Respond with JSON: {\"skills\": [<skill name>, ...]}.\n\n"
            f"RESEARCH GOAL: {research_goal}\n\n"
            f"AVAILABLE TECHNIQUES:\n{json.dumps(options, indent=2)}"
        )
        chosen = self._extract_json(self._llm(prompt)) or {}
        names = set(chosen.get("skills") or [])
        selected = [s for s in eligible if s["name"] in names]
        # If the LLM named nothing usable, fall back to all eligible (compute
        # everything available rather than nothing).
        return selected or eligible

    def run_analysis(self, research_goal: str, run_dir: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """Compute the goal's properties from the run output in ``run_dir``.

        Returns ``{"status", "results", "output_directory", "data_kinds",
        "skills_used"}`` where ``results`` maps property → the base engine's
        result dict (value, units, verification, …). ``status`` is ``"error"``
        when no output data is recognized, else ``"success"`` even if individual
        analyses fail (their per-property error is recorded).
        """
        run_dir = run_dir or str(self.output_dir)
        by_kind = self.classify_outputs(run_dir)
        if not by_kind:
            return {"status": "error", "message": f"no recognized output in {run_dir}",
                    "results": {}, "output_directory": str(self.output_dir)}

        eligible = self.eligible_skills(by_kind.keys())
        selected = self._select_properties(research_goal, eligible)

        # All classified files are offered to each analysis; the generated code
        # reads whichever it needs.
        data_files = {Path(p).name: p for paths in by_kind.values() for p in paths}
        results: Dict[str, Any] = {}
        for skill in selected:
            meta = skill.get("meta") or {}
            computes = meta.get("computes") or [skill["name"]]
            if isinstance(computes, str):
                computes = [computes]
            recipe = skill.get("implementation") or skill.get("analysis") or ""
            for prop in computes:
                results[prop] = self.compute_property(
                    task=f"{prop} for the research goal: {research_goal}",
                    data_files=data_files, recipe=recipe)

        return {"status": "success", "results": results,
                "output_directory": str(self.output_dir),
                "data_kinds": sorted(by_kind), "skills_used":
                [s["name"] for s in selected]}
