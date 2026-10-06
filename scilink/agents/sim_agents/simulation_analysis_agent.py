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
    # Sibling domains served by the same agent — e.g. forward models that return
    # a curve/image/datacube instead of a scalar. Discovered, availability-gated,
    # and selected identically to the scalar skills; the only difference is the
    # skill's ``output:`` frontmatter, which routes ``compute_property``.
    EXTRA_DOMAINS = ("forward_models",)

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
            # Match a full filename (e.g. `log.lammps`) or the exact extension
            # (e.g. `lammpstrj`) — NOT a bare `endswith`, which mis-classifies
            # `mesh.inc` as a trajectory because it ends with `nc`.
            for kind, pats in fmt.items():
                if any(name == pat or ext == pat for pat in pats):
                    present[kind].append(str(p))
                    break
        return dict(present)

    @staticmethod
    def _peek(path: Path, n_lines: int = 10, max_chars: int = 2000):
        """First ``n_lines`` of a text file (or ``None`` if binary/unreadable)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = []
                for _ in range(n_lines):
                    ln = f.readline()
                    if not ln:
                        break
                    lines.append(ln.rstrip("\n"))
        except (OSError, UnicodeDecodeError):
            return None
        return "\n".join(lines)[:max_chars]

    def resolve_outputs(self, run_dir: str) -> Dict[str, List[str]]:
        """Map output files to data kinds by reading the run, deck-grounded.

        :meth:`classify_outputs` only matches fixed filename patterns, so an
        output the engine wrote to a self-named file -- e.g. a
        ``fix ave/time ... v_pxy v_pxz v_pyz ... file stress.dat`` pressure-
        tensor series -- is missed and never reaches the analysis. This
        generalizes it without putting any filename in code: the LLM reads a
        header peek of every file (the input deck included) and maps the OUTPUT
        files to the skills' data-kind vocabulary, using the deck as ground
        truth for what each file holds. The static classification is the floor;
        the LLM can only ADD files (existing, non-empty) to it, and the method
        falls back to the static map when there is nothing unrecognized to
        resolve or the LLM step errors -- so it is a strict superset of
        :meth:`classify_outputs` and never loses a match.
        """
        import json

        static = self.classify_outputs(run_dir)
        root = Path(run_dir)
        vocab = sorted(self._output_format_map().keys())
        if not root.exists() or not vocab:
            return static

        classified = {p for paths in static.values() for p in paths}
        peeks: List[tuple] = []
        has_unclassified = False
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            head = self._peek(p)
            if head is None:            # binary / unreadable -> not an analysis input
                continue
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                rel = p.name
            peeks.append((rel, head))
            if str(p) not in classified:
                has_unclassified = True
        # Nothing the static map missed -> no reason to spend an LLM call.
        if not peeks or not has_unclassified:
            return static

        listing = "\n\n".join(f"### {rel}\n{head}" for rel, head in peeks[:60])
        prompt = (
            "A simulation run produced the files below (path, then first lines). "
            "Some are INPUT decks (a LAMMPS run script with fix/run/dump "
            "commands, a VASP INCAR, an NWChem .nw); use them as ground truth "
            "for what each OUTPUT file contains -- e.g. a line "
            "`fix ... ave/time ... v_pxy v_pxz v_pyz ... file stress.dat` means "
            "stress.dat is the pressure-tensor time series. Map only the OUTPUT "
            "files to data kinds, using ONLY these kinds: "
            f"{json.dumps(vocab)}. A file may map to more than one kind; omit "
            "input decks and files that fit no kind. Respond with JSON: "
            "{\"<data_kind>\": [\"<relative path>\", ...]}.\n\n"
            f"FILES:\n\n{listing}"
        )
        try:
            mapping = self._extract_json(self._llm(prompt)) or {}
        except Exception as exc:        # LLM/parse failure -> static is the floor
            self.logger.warning("resolve_outputs LLM step failed: %s", exc)
            return static

        out = {k: list(v) for k, v in static.items()}
        for kind, rels in mapping.items():
            if kind not in vocab or not isinstance(rels, list):
                continue
            for rel in rels:
                try:
                    p = (root / rel)
                    if p.is_file() and p.stat().st_size > 0:
                        bucket = out.setdefault(kind, [])
                        if str(p) not in bucket:
                            bucket.append(str(p))
                except OSError:
                    continue
        return out

    def _skill_catalog(self) -> List[Dict[str, Any]]:
        """Return the loaded analysis skills (``{name, meta, sections}``).

        Separated so tests can substitute a catalog without on-disk skills.
        """
        from ...skills.loader import list_skills, load_skill

        catalog: List[Dict[str, Any]] = []
        for domain in (self.DOMAIN, *self.EXTRA_DOMAINS):
            for name in list_skills(domain=domain):
                try:
                    catalog.append(load_skill(name, domain=domain))
                except Exception as exc:  # a broken skill must not sink selection
                    self.logger.warning("Skill %r (%s) failed to load: %s",
                                        name, domain, exc)
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

    def _input_deck_patterns(self) -> set:
        """fnmatch patterns for engine input decks, from skills' ``inputs:``.

        Mirrors :meth:`_output_format_map` for inputs: the analysis needs the run
        deck (the ground truth for what each output column is) and learns the
        deck's filename convention from the engine skill, not from code. Cached.
        """
        if getattr(self, "_input_pat_cache", None) is not None:
            return self._input_pat_cache
        from ...skills.loader import list_all_skills, load_skill

        pats: set = set()
        for domain, names in list_all_skills().items():
            for name in names:
                try:
                    meta = load_skill(name, domain=domain).get("meta") or {}
                except Exception:
                    continue
                inp = meta.get("inputs")
                groups = (inp.values() if isinstance(inp, dict)
                          else [inp] if inp else [])
                for g in groups:
                    if isinstance(g, str):
                        g = [g]
                    pats.update(str(p) for p in g)
        self._input_pat_cache = pats
        return pats

    def _gather_input_decks(self, run_dir: str, *, max_decks: int = 10,
                            max_chars: int = 20000) -> Dict[str, str]:
        """Read the run's input deck files (text), keyed by path relative to run_dir.

        Fed to :meth:`compute_property` as ``INPUT_DECKS`` so the analysis maps
        output columns to physical quantities by reading the deck -- e.g. follow
        a ``fix ave/time ... v_pxy ... file stress.dat`` line back to
        ``variable pxy equal pxy`` -- instead of guessing from a column header.
        Dry-run decks are skipped.
        """
        import fnmatch

        pats = self._input_deck_patterns()
        root = Path(run_dir)
        if not pats or not root.exists():
            return {}
        decks: Dict[str, str] = {}
        for p in sorted(root.rglob("*")):
            if len(decks) >= max_decks:
                break
            if not p.is_file() or "dryrun" in str(p).lower():
                continue
            if not any(fnmatch.fnmatch(p.name, pat) for pat in pats):
                continue
            text = self._peek(p, n_lines=500, max_chars=max_chars)
            if not text:
                continue
            try:
                key = str(p.relative_to(root))
            except ValueError:
                key = p.name
            decks[key] = text
        return decks

    def run_analysis(self, research_goal: str, run_dir: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """Compute the goal's properties from the run output in ``run_dir``.

        Returns ``{"status", "results", "output_directory", "data_kinds",
        "skills_used"}`` where ``results`` maps property → the base engine's
        result dict (value, units, verification, …). ``status`` is ``"error"``
        when no output data is recognized, else ``"success"`` even if individual
        analyses fail (their per-property error is recorded).
        """
        # Resolve to an ABSOLUTE path up front: DATA_FILES paths must be openable
        # from the sandbox's own working directory, not the caller's CWD, so a
        # relative run_dir (what a tool call often passes) would otherwise yield
        # relative, unopenable paths in the generated code.
        run_dir = str(Path(run_dir or self.output_dir).resolve())
        # Deck-grounded resolution (superset of the static filename match) so an
        # output written to a self-named file still reaches the analysis.
        by_kind = self.resolve_outputs(run_dir)
        if not by_kind:
            return {"status": "error", "message": f"no recognized output in {run_dir}",
                    "results": {}, "output_directory": str(self.output_dir)}

        eligible = self.eligible_skills(by_kind.keys())
        selected = self._select_properties(research_goal, eligible)

        # Offer every classified file to each analysis, keyed by its path RELATIVE
        # to run_dir — so fan-out members (member_0/log.lammps, member_1/…) keep
        # distinct keys instead of colliding on basename and silently dropping all
        # but one. The generated code reads whichever it needs from DATA_FILES.
        root = Path(run_dir)
        data_files: Dict[str, str] = {}
        for paths in by_kind.values():
            for p in paths:
                try:
                    key = str(Path(p).relative_to(root))
                except ValueError:
                    key = Path(p).name
                data_files[key] = p
        # The run deck(s): ground truth for what each output column is, so the
        # analysis maps columns by reading the deck rather than guessing headers.
        input_decks = self._gather_input_decks(run_dir)
        results: Dict[str, Any] = {}
        for skill in selected:
            meta = skill.get("meta") or {}
            computes = meta.get("computes") or [skill["name"]]
            if isinstance(computes, str):
                computes = [computes]
            recipe = skill.get("implementation") or skill.get("analysis") or ""
            output_type = meta.get("output", "scalar")
            for prop in computes:
                results[prop] = self.compute_property(
                    task=f"{prop} for the research goal: {research_goal}",
                    data_files=data_files, recipe=recipe,
                    output_type=output_type, input_decks=input_decks)

        return {"status": "success", "results": results,
                "output_directory": str(self.output_dir),
                "data_kinds": sorted(by_kind), "skills_used":
                [s["name"] for s in selected]}
