"""Sim-side base for simulation-analysis agents.

Simulation analysis is **data → number**: compute a property (viscosity,
diffusion, an RDF, a band gap) from a finished run's output. Unlike the
experimental analysis agents — which split by data shape and carry
shape-specific pipeline stages — simulation analysis is codegen-dominated: the
reusable engine is a ``generate code → run it sandboxed → verify the number →
refine on failure`` loop, and the data shape is absorbed inside the generated
code and its library (MDAnalysis / pymatgen / numpy). This base owns that
engine; a subclass adds the analysis pipeline (which properties, from which
inputs, via which technique skill).

Kept in ``sim_agents`` with its own base — it does not subclass the experimental
``BaseAnalysisAgent`` — so the simulation and experimental modality families stay
decoupled. Only genuinely shared infrastructure is reused (the LLM wrappers, the
skill loader, and ``ScriptExecutor``).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ...auth import get_internal_proxy_key, require_vendor_credentials
from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
from ...executors import ScriptExecutor, check_security_sandbox_indicators
from ._deprecation import normalize_params

import logging

# Packages a generated analysis script may assume are importable.
_STANDARD_PACKAGES = (
    "numpy", "scipy", "pandas", "matplotlib", "json", "math", "os", "sys",
    "re", "pathlib", "collections", "itertools", "MDAnalysis", "ase",
)


class BaseAnalysisAgent(ABC):
    """Base for simulation-analysis agents: a verified codegen engine.

    Provides the LLM client, skill loading, and the reusable
    ``compute_property`` loop (generate → execute sandboxed → verify → refine).
    A subclass implements :meth:`run_analysis` with the modality's pipeline
    (identify outputs, select the technique skill by available data, plan the
    properties to compute, and call :meth:`compute_property` for each).

    Attributes:
        model: The resolved generative-model client.
        executor: The sandboxed script executor.
        output_dir: Directory analysis artifacts are written to.
        state: Per-run scratch state (skills loaded, intermediate results).
    """

    def __init__(
        self,
        output_dir: str = ".",
        model_name: str = "claude-opus-4-6",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        executor_timeout: int = 120,
        max_refinement_attempts: int = 2,
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
        """Construct the agent and its LLM client + sandbox executor.

        Args:
            output_dir: Where analysis artifacts (scripts, figures) are written.
            model_name: Model identifier in the resolved provider's form.
            api_key: LLM key. With ``base_url`` set this is the internal-proxy
                key (or read from the environment); without it, forwarded to
                LiteLLM, which falls back to the vendor's env var.
            base_url: OpenAI-compatible internal-proxy URL; when set, requests
                route through the proxy client, otherwise through LiteLLM.
            executor_timeout: Per-script sandbox timeout, in seconds.
            max_refinement_attempts: How many times a failed script is
                regenerated from its error before giving up.
            google_api_key: Deprecated. Use ``api_key``.
            local_model: Deprecated. Use ``base_url``.

        Raises:
            ValueError: If ``base_url`` is set and no key can be resolved.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_refinement_attempts = max_refinement_attempts
        self.state: Dict[str, Any] = {}

        api_key, base_url = normalize_params(
            api_key=api_key, google_api_key=google_api_key,
            base_url=base_url, local_model=local_model,
            source=self.__class__.__name__,
        )
        if base_url:
            if api_key is None:
                api_key = get_internal_proxy_key()
            if not api_key:
                raise ValueError(
                    "API key required for internal proxy. Set SCILINK_API_KEY "
                    "or pass api_key."
                )
            self.model = OpenAIAsGenerativeModel(
                model=model_name, api_key=api_key, base_url=base_url)
        else:
            if api_key is None:
                require_vendor_credentials(model_name)
            self.model = LiteLLMGenerativeModel(model=model_name, api_key=api_key)

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.generation_config = None

        self.executor = ScriptExecutor(timeout=executor_timeout)
        score, _ = check_security_sandbox_indicators()
        self.in_container = score >= 4

    # ── skills ────────────────────────────────────────────────────────

    def _load_skills_to_state(
        self, skill: Union[str, List[str], None], domain: str,
    ) -> Dict[str, Any]:
        """Load one or more skills and return a state-dict fragment.

        Accepts a single name/path or a list. Returns ``skill_name`` /
        ``skill_sections`` (the first loaded skill) plus ``skills_loaded`` (all
        of them). Skills that fail to load are logged and dropped; an empty list
        is acceptable.
        """
        from ...skills.loader import load_skill

        if not skill:
            inputs: List[str] = []
        elif isinstance(skill, str):
            inputs = [skill]
        else:
            inputs = list(skill)

        loaded: List[Dict[str, Any]] = []
        seen: set = set()
        for s in inputs:
            try:
                parsed = load_skill(s, domain=domain)
            except FileNotFoundError:
                self.logger.warning("Skill %r not found in %r — skipping", s, domain)
                continue
            if parsed["name"] in seen:
                continue
            seen.add(parsed["name"])
            loaded.append(parsed)
        first = loaded[0] if loaded else None
        return {
            "skill_name": first["name"] if first else None,
            "skill_sections": first,
            "skills_loaded": loaded,
        }

    # ── the codegen engine ────────────────────────────────────────────

    def compute_property(
        self,
        task: str,
        data_files: Dict[str, str],
        *,
        recipe: str = "",
        packages: Optional[List[str]] = None,
        verify: bool = True,
        output_type: str = "scalar",
    ) -> Dict[str, Any]:
        """Compute one property from ``data_files`` via verified codegen.

        Generates a Python script for ``task`` (guided by ``recipe`` — a skill's
        implementation section), runs it sandboxed, and — on success — optionally
        checks the returned value is physically plausible. A failed run is
        regenerated from its error up to ``max_refinement_attempts`` times.

        Args:
            task: What to compute (e.g. ``"shear viscosity via Green-Kubo"``).
            data_files: Mapping of logical name to path the script reads.
            recipe: Optional technique recipe (a skill's implementation section)
                the generated code should follow.
            packages: Packages the script may import (defaults to the standard set).
            verify: When True, run the LLM plausibility gate on the result.

        Returns:
            ``{"status", "value"?, "units"?, "verification"?, "code_path",
            "attempts"}``. ``status`` is ``"success"`` when the script ran and
            returned a value, else ``"error"`` with a ``message``.
        """
        pkgs = packages or list(_STANDARD_PACKAGES)
        # Inject the inputs the generated body references as globals, so it needs
        # no hardcoded paths and stays engine-neutral.
        preamble = (
            f"DATA_FILES = {json.dumps(data_files)}\n"
            f"OUTPUT_DIR = {json.dumps(str(self.output_dir))}\n\n"
        )
        code = self._generate_code(task, data_files, recipe, pkgs, output_type)
        error_info: Dict[str, Any] = {}
        for attempt in range(self.max_refinement_attempts + 1):
            if attempt > 0:
                code = self._refine_code(code, error_info, task, recipe, pkgs)
            result = self._execute_script(preamble + code, task)
            if result.get("ok"):
                # The script ran and produced its answer (success OR an honest
                # error). Return it as-is — do not retry, and never overwrite an
                # honest error with success.
                out = {k: v for k, v in result.items() if k != "ok"}
                out["attempts"] = attempt + 1
                # Non-scalar outputs (curve/image/datacube) hand back a file; the
                # script writes it into OUTPUT_DIR and references it. Resolve and
                # require it. A success with no artifact on disk is a fixable
                # codegen slip — feed the refine loop and retry (like a
                # missing-JSON result), not a terminal error.
                if output_type != "scalar" and out.get("status") == "success":
                    artifact = self._resolve_artifact(out.get("artifact"))
                    if artifact is None:
                        error_info = {
                            "message": ("reported success but wrote no readable "
                                        f"artifact for output_type {output_type!r}"),
                            "concise_error": "success without artifact on disk"}
                        continue
                    out["artifact"] = artifact
                    out["output_type"] = output_type
                if (verify and out.get("status") == "success"
                        and (out.get("value") is not None or out.get("artifact"))):
                    out["verification"] = self._verify_result(
                        task, out, output_type)
                return out
            error_info = result
        return {"status": "error", "message": error_info.get("message", "unknown"),
                "attempts": self.max_refinement_attempts + 1}

    def _generate_code(self, task: str, data_files: Dict[str, str],
                       recipe: str, packages: List[str],
                       output_type: str = "scalar") -> str:
        """Generate a self-contained analysis script for ``task``.

        ``output_type`` selects the output contract the script must satisfy: a
        ``scalar`` prints ``value``/``units``; a ``curve``/``image``/``datacube``
        writes the artifact into ``OUTPUT_DIR`` and returns a reference plus
        summary statistics the verification gate can judge.
        """
        files_desc = "\n".join(f"  - {name}: {path}"
                               for name, path in data_files.items())
        if output_type == "scalar":
            output_contract = (
                "- Compute the property and print EXACTLY ONE JSON object as the "
                'last stdout line: {"status": "success", "value": <number>, '
                '"units": <str>, ...} — or {"status": "error", "message": <str>} '
                "if it cannot be computed.\n"
            )
        else:
            output_contract = (
                f"- This is a {output_type} observable. WRITE the computed "
                f"{output_type} into OUTPUT_DIR (a .npy or .csv for a curve or "
                "datacube; a .png for an image), then print EXACTLY ONE JSON "
                'object as the last stdout line: {"status": "success", '
                f'"output_type": "{output_type}", "artifact": {{"path": '
                '"<file you wrote under OUTPUT_DIR>", "format": "<npy|csv|png>", '
                '"shape": [...]}, "summary": {<a few SCALAR summary statistics so '
                "the result can be judged — e.g. n_points, x/y ranges, peak "
                'value, NaN count>}} — or {"status": "error", "message": <str>} '
                "if it cannot be computed.\n"
            )
        prompt = (
            "You are a scientific data-analysis engineer. Write a complete, "
            "self-contained Python script that computes the requested property "
            "from the provided data files and prints the result as a single "
            "JSON object on the LAST line of stdout.\n\n"
            f"TASK: {task}\n\n"
            f"DATA FILES (globals `DATA_FILES` maps name -> path):\n{files_desc}\n\n"
            + (f"TECHNIQUE RECIPE (follow this):\n{recipe}\n\n" if recipe else "")
            + "REQUIREMENTS:\n"
            f"- Import only from: {', '.join(packages)}.\n"
            "- `DATA_FILES` (dict name->path) and `OUTPUT_DIR` (str) are ALREADY "
            "defined as globals at runtime — reference them directly; do NOT "
            "import, redefine, guard, or reassign them. Read inputs from "
            "DATA_FILES; write any files into OUTPUT_DIR.\n"
            + output_contract
            + "- Handle missing/short data gracefully; never hang or prompt.\n\n"
            "Return ONLY the Python code, no markdown."
        )
        return self._clean_code(self._llm(prompt))

    def _refine_code(self, code: str, error_info: Dict[str, Any], task: str,
                     recipe: str, packages: List[str]) -> str:
        """Regenerate a failed script from its error."""
        err = error_info.get("concise_error") or error_info.get("message", "")
        prompt = (
            "The following analysis script failed. Fix it and return the full "
            "corrected script (same contract: use globals DATA_FILES / OUTPUT_DIR, "
            "print one JSON object on the last stdout line).\n\n"
            f"TASK: {task}\n\n"
            + (f"RECIPE:\n{recipe}\n\n" if recipe else "")
            + f"ERROR:\n{err}\n\nSCRIPT:\n{code}\n\n"
            f"Import only from: {', '.join(packages)}. Return ONLY the code."
        )
        return self._clean_code(self._llm(prompt))

    def _execute_script(self, code: str, name: str) -> Dict[str, Any]:
        """Run a script sandboxed with DATA_FILES/OUTPUT_DIR injected; parse JSON.

        The script's globals are prepended, so the generated body reads its
        inputs and writes figures without hardcoded paths. Syntactically-invalid
        code is caught by a compile check before any sandbox run, so the refine
        loop gets a precise error cheaply. On success the last JSON object on
        stdout is returned (merged into the result); on failure a concise error
        plus the traceback tail is returned for refinement.
        """
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            msg = f"SyntaxError: {e.msg} (line {e.lineno})"
            return {"ok": False, "status": "error", "message": msg,
                    "concise_error": msg}
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_") or "analysis"
        exec_result = self.executor.execute_script(
            script_content=code, working_dir=str(self.output_dir))
        if exec_result.get("status") == "success":
            (self.output_dir / f"{slug}.py").write_text(code, encoding="utf-8")
            parsed = self._extract_json(exec_result.get("stdout", ""))
            if parsed is not None:
                # The script RAN and returned its JSON answer — whether it reports
                # success or an honest "cannot compute" error. ok=True means don't
                # retry: retrying an honest error only invites a fabricated value,
                # the one failure mode a validation-feeding agent must not have.
                # The script's own status is preserved (not forced to success).
                parsed.setdefault("status", "success")
                return {"ok": True, "code_path":
                        str(self.output_dir / f"{slug}.py"), **parsed}
            # Ran but violated the contract (no JSON on stdout) — retryable.
            return {"ok": False, "status": "error",
                    "message": "ran but produced no JSON on the last stdout line",
                    "concise_error": "no JSON output",
                    "raw_output": exec_result.get("stdout", "")[:2000]}
        # The script crashed — retryable.
        raw = exec_result.get("message", "") or exec_result.get("stderr", "")
        tb = raw[raw.find("Traceback"):] if "Traceback" in raw else raw
        concise = ""
        for line in reversed(tb.strip().split("\n")):
            line = line.strip()
            if line and not line.startswith(("File", "Traceback")):
                concise = line
                break
        return {"ok": False, "status": "error",
                "message": tb[-2000:] or "no error output", "concise_error": concise}

    @staticmethod
    def _extract_json(stdout: str) -> Optional[Dict[str, Any]]:
        """Return the last JSON object printed on stdout, or None."""
        for line in reversed(stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        # Fallback: the LAST balanced object anywhere (matches the docstring).
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", stdout, re.DOTALL)
        for candidate in reversed(matches):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _resolve_artifact(self, artifact: Any) -> Optional[Dict[str, Any]]:
        """Validate a non-scalar artifact reference and absolutize its path.

        Returns the artifact dict with an absolute, existing ``path`` (resolved
        against ``OUTPUT_DIR`` when relative), or ``None`` when the script
        claimed an artifact it did not actually write.
        """
        if not isinstance(artifact, dict) or not artifact.get("path"):
            return None
        p = Path(artifact["path"])
        if not p.is_absolute():
            p = self.output_dir / p
        if not p.exists():
            return None
        resolved = dict(artifact)
        resolved["path"] = str(p)
        return resolved

    def _verify_result(self, task: str, result: Dict[str, Any],
                       output_type: str = "scalar") -> Dict[str, Any]:
        """LLM plausibility gate on a computed result (the exp-agent-style check).

        For a scalar it judges the value (+ convergence flags); for a
        curve/image/datacube it judges the artifact's summary statistics and
        shape. Returns ``{"plausible": bool, "reasoning": str}``, erring toward
        accepting when unsure — it flags clearly-wrong results, it is not a
        second physics reviewer. A failure to parse is non-blocking.
        """
        if output_type == "scalar":
            details = {
                k: v for k, v in result.items()
                if k not in ("code_path", "ok", "status", "attempts",
                             "verification", "raw_output", "artifact")
                and isinstance(v, (int, float, str, bool))
            }
            prompt = (
                "A simulation-analysis script computed a property. Judge whether "
                "the result is physically plausible AND adequately converged — "
                "take the reported diagnostic/convergence fields into account: a "
                "`plateau_reached` / `linear_regime` / `converged`-style flag that "
                "is false means the value is NOT trustworthy and must NOT be "
                "judged plausible on magnitude alone. Respond with JSON: "
                '{"plausible": true|false, "reasoning": "<one sentence>"}.\n\n'
                f"TASK: {task}\n"
                f"RESULT FIELDS: {json.dumps(details, default=str)}\n"
            )
        else:
            artifact = result.get("artifact") or {}
            details = {
                "output_type": output_type,
                "summary": result.get("summary") or {},
                "artifact_shape": artifact.get("shape"),
                "artifact_format": artifact.get("format"),
            }
            prompt = (
                f"A forward-model script produced a {output_type} observable "
                "(saved to a file) and reported summary statistics about it. "
                "Judge whether those statistics and the shape are physically "
                "plausible for the task — e.g. a spectrum/curve spans a sensible "
                "range with no NaNs and a reasonable number of points; an image "
                "has non-degenerate dimensions. This is a sanity gate, not a "
                "second physics review. Respond with JSON: "
                '{"plausible": true|false, "reasoning": "<one sentence>"}.\n\n'
                f"TASK: {task}\n"
                f"RESULT: {json.dumps(details, default=str)}\n"
            )
        parsed = self._extract_json(self._llm(prompt))
        if not parsed or "plausible" not in parsed:
            return {"plausible": True, "reasoning": "verification unavailable"}
        return {"plausible": bool(parsed["plausible"]),
                "reasoning": parsed.get("reasoning", "")}

    def _llm(self, prompt: str) -> str:
        """Send a prompt to the model and return the UNCLEANED text.

        Uses ``raw_text`` in preference to ``.text``: the wrapper's ``.text``
        runs an embedded-JSON extraction that greedily slices a ``{...}`` out of
        the response, which mangles generated code (a script's result dict looks
        like JSON). JSON callers here re-extract via :meth:`_extract_json`, so raw
        text is correct for both code and JSON responses.
        """
        response = self.model.generate_content(
            prompt, generation_config=self.generation_config)
        return (getattr(response, "raw_text", None)
                or getattr(response, "text", None) or str(response))

    @staticmethod
    def _clean_code(text: str) -> str:
        """Strip markdown fences from an LLM code response."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n", "", text)
            text = re.sub(r"\n```$", "", text)
        return text.strip()

    # ── contract ──────────────────────────────────────────────────────

    @abstractmethod
    def run_analysis(self, research_goal: str, **kwargs) -> Dict[str, Any]:
        """Compute the properties a research goal calls for from run output.

        Implemented by the concrete agent: identify the available outputs, select
        the technique skill by what data is present, plan the properties, and
        call :meth:`compute_property` for each. Returns
        ``{"status", "results", "output_directory", ...}``.
        """
        raise NotImplementedError
