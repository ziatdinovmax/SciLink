"""Tool registry for MetaOrchestratorAgent.

Mirrors the AnalysisOrchestratorTools shape — a ``ToolsClass(orchestrator)``
that builds ``functions_map`` + ``openai_schemas`` and exposes
``execute_tool``. The meta-agent's tools delegate to child orchestrators via
their ``run_task`` contract and introspect the delegation ledger. See
CLAUDE.md "The meta agent".

The duplication with AnalysisOrchestratorTools is intentional and acceptable
at this development stage — see CLAUDE.md "Why no BaseChatOrchestrator
refactor".
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict


def _handoff(text: str) -> str:
    """Bold a meta -> specialist handoff banner on a real terminal so the
    delegation transition stands out; plain on non-tty (captured/redirected
    output, e.g. the UI, which styles the banner itself in `_log_to_html`)."""
    return f"\033[1;33m{text}\033[0m" if sys.stdout.isatty() else text


def _task_summary(task: str, n: int = 80) -> str:
    """One-line preview of a delegated task for the handoff banner.

    The banner styles its whole line (bold on the terminal, a band in the UI).
    A task is often multi-line — a one-line instruction followed by structured
    detail such as a 'Primary data file: <path>' line — and the raw ``task[:80]``
    used to carry those newlines into the banner, so the styling bled across the
    blank line onto the next line in the terminal. Collapse to the first
    non-blank line, truncated, with an ellipsis when anything was dropped."""
    first = next((ln.strip() for ln in (task or "").splitlines() if ln.strip()), "")
    dropped = len(first) > n or len((task or "").strip().splitlines()) > 1
    return first[:n] + ("…" if dropped else "")


# ── Upload inspection ────────────────────────────────────────────────
# Lightweight, content-based file probes so the meta-agent routes each
# uploaded file from evidence (array shape, table columns, document text)
# rather than guessing from the filename. Every heavy / optional import is
# done lazily inside the relevant branch, so this module stays importable
# without those packages and a missing reader degrades one file's probe
# instead of breaking the whole tool.

_PROBE_MAX_FILES = 60
_PROBE_TEXT_HEAD = 400


def _has_comment_header(path: Path) -> bool:
    """True if the file's first non-blank line is a ``#`` comment — a quick signal
    that a CSV/text carries a leading metadata header block worth splitting out."""
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    return s.startswith("#")
    except OSError:
        return False
    return False


def _probe_delimited_text(path: Path, max_rows: int = 200):
    """Probe a .dat/.txt as a delimited NUMERIC table, skipping a #/comment header.

    Returns ``{n_columns, sampled_rows, dtypes}`` (positional dtypes, since such
    files are usually headerless) when the body parses as numeric data, else None
    (prose). A column counts as numeric when >80% of its sampled values parse as
    numbers — tolerant of a single column-name header row above the data.
    """
    import pandas as pd
    for sep in (r"\s+", ","):
        try:
            df = pd.read_csv(path, sep=sep, comment="#", header=None, nrows=max_rows,
                             skip_blank_lines=True, engine="python")
        except Exception:  # noqa: BLE001 - try the next separator
            continue
        if df.shape[0] < 3 or df.shape[1] < 1:
            continue
        frac_numeric = df.apply(pd.to_numeric, errors="coerce").notna().mean(axis=0)
        ncols = int((frac_numeric > 0.8).sum())
        if ncols >= 1:
            return {"n_columns": ncols, "sampled_rows": int(df.shape[0]),
                    "dtypes": {str(i): "float64" for i in range(ncols)}}
    return None


def _probe_file(path: Path) -> Dict[str, Any]:
    """Content-probe a single file for routing. Never raises — any failure
    is reported in the returned dict's ``note`` field."""
    ext = path.suffix.lower()
    info: Dict[str, Any] = {"file": str(path), "ext": ext}
    try:
        info["size_kb"] = round(path.stat().st_size / 1024, 1)
    except OSError:
        pass
    try:
        if ext == ".npy":
            import numpy as np
            try:
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                info.update(kind="array", shape=list(arr.shape),
                            dtype=str(arr.dtype))
            except ValueError:
                # Pickled object array — e.g. an instrument header dict saved
                # via np.save. It's a trusted local upload (same trust as every
                # other format this probe parses), so unpickle to describe its
                # structure instead of reporting "unreadable" — that blindness
                # is what starved the prepare_inputs codegen of evidence
                # (issue #380). Size-capped: object arrays can't be mmap'd, so
                # a huge pickle would be loaded whole.
                if path.stat().st_size > 50 * 1024 * 1024:
                    raise
                from ...utils.file_prep import probe_pickled_npy
                info.update(probe_pickled_npy(path))
        elif ext == ".npz":
            import numpy as np
            with np.load(path, allow_pickle=False) as z:
                files = list(z.files)
                info.update(kind="npz", keys=sorted(files))
                try:
                    info["shapes"] = {k: list(z[k].shape) for k in files[:40]}
                except Exception:  # noqa: BLE001 - shapes are a best-effort extra
                    pass
        elif ext in (".h5", ".hdf5", ".nxs", ".nx"):
            import h5py
            names: list = []
            with h5py.File(path, "r") as f:
                f.visititems(lambda n, o: names.append(n))
                root_attrs = sorted(f.attrs.keys())
            info.update(kind="hdf5", datasets=sorted(names)[:80],
                        root_attrs=root_attrs[:40])
        elif ext == ".mat":
            # whosmat reads the directory without loading arrays (cheap); v7.3
            # .mat are HDF5 and raise NotImplementedError → probe via h5py.
            try:
                from scipy.io import whosmat
                info.update(kind="mat",
                            keys=sorted(n for n, _, _ in whosmat(path)),
                            mat_version="<=v7")
            except NotImplementedError:
                import h5py
                names = []
                with h5py.File(path, "r") as f:
                    f.visititems(lambda n, o: names.append(n))
                info.update(kind="hdf5", datasets=sorted(names)[:80],
                            mat_version="v7.3")
        elif ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            from PIL import Image
            with Image.open(path) as im:
                info.update(kind="image", height=im.height, width=im.width,
                            mode=im.mode, n_frames=getattr(im, "n_frames", 1))
        elif ext in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            # comment="#" skips a leading metadata/comment header block — the
            # canonical "combined" CSV (instrument/ImageJ exports) — so the probed
            # columns reflect the real table, not the first comment line. Without
            # it, combined CSVs get garbage column signatures and fail to cluster.
            df = pd.read_csv(path, sep=sep, nrows=200, comment="#",
                             skip_blank_lines=True)
            info.update(kind="table", n_columns=int(df.shape[1]),
                        sampled_rows=int(df.shape[0]),
                        columns=[str(c) for c in df.columns[:40]],
                        dtypes={str(c): str(t)
                                for c, t in list(df.dtypes.items())[:40]},
                        has_comment_header=_has_comment_header(path))
        elif ext == ".xlsx":
            import pandas as pd
            df = pd.read_excel(path, nrows=200)
            info.update(kind="table", n_columns=int(df.shape[1]),
                        sampled_rows=int(df.shape[0]),
                        columns=[str(c) for c in df.columns[:40]])
        elif ext == ".json":
            with open(path, "r", errors="replace") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                info.update(kind="json", json_type="object",
                            top_level_keys=[str(k) for k in list(obj)[:40]])
            elif isinstance(obj, list):
                info.update(kind="json", json_type="array", length=len(obj))
            else:
                info.update(kind="json", json_type=type(obj).__name__)
        elif ext == ".pdf":
            info.update(kind="document", doc_type="pdf")
            try:
                from scilink.parsers import extract_text
                # max_pages=1 keeps the probe cheap — only a text head is needed.
                doc_info = extract_text(path, max_pages=1)
                info["n_pages"] = doc_info.get("n_pages")
                info["text_head"] = doc_info["text"][:_PROBE_TEXT_HEAD].strip()
            except Exception as e:  # noqa: BLE001 - optional reader / bad PDF
                info["note"] = f"page/text probe unavailable: {e}"
        elif ext == ".docx":
            info.update(kind="document", doc_type="docx")
            try:
                from scilink.parsers import extract_text
                doc_info = extract_text(path)
                info["n_paragraphs"] = doc_info.get("n_paragraphs")
                info["text_head"] = doc_info["text"][:_PROBE_TEXT_HEAD].strip()
            except Exception as e:  # noqa: BLE001 - optional reader / bad docx
                info["note"] = f"text probe unavailable: {e}"
        elif ext in (".dat", ".txt"):
            # .dat/.txt is often a delimited numeric table (frequently with a
            # leading comment/metadata header) but may be prose. Probe it as a
            # table when it parses as numeric data — so combined .dat/.txt files
            # cluster like CSV in batch prep — else fall back to text.
            parsed = _probe_delimited_text(path)
            if parsed:
                info.update(kind="table",
                            has_comment_header=_has_comment_header(path), **parsed)
            else:
                text = path.read_text(errors="replace")
                info.update(kind="text", n_chars=len(text),
                            text_head=text[:_PROBE_TEXT_HEAD].strip())
        elif ext == ".md":
            text = path.read_text(errors="replace")
            info.update(kind="text", n_chars=len(text),
                        text_head=text[:_PROBE_TEXT_HEAD].strip())
        elif ext == ".py":
            text = path.read_text(errors="replace")
            info.update(kind="code", n_lines=text.count("\n") + 1,
                        text_head=text[:_PROBE_TEXT_HEAD].strip())
        elif ext in (".yaml", ".yml"):
            text = path.read_text(errors="replace")
            info.update(kind="config", text_head=text[:_PROBE_TEXT_HEAD].strip())
        else:
            info["kind"] = "unknown"
    except Exception as e:  # noqa: BLE001 - probe must never break the tool
        info.setdefault("kind", "unreadable")
        info["note"] = f"probe failed: {e}"
    return info


class MetaOrchestratorTools:
    """Tool definitions, schemas, and execution for MetaOrchestratorAgent."""

    def __init__(self, orchestrator_instance):
        """
        Args:
            orchestrator_instance: the parent MetaOrchestratorAgent.
        """
        self.orch = orchestrator_instance
        self.logger = logging.getLogger(self.__class__.__name__)

        self.functions_map: Dict[str, Callable] = {}
        self.openai_schemas: list = []

        self._register_all_tools()
        # Bounded access to this session's own action history (#462) —
        # registered after the mode's own tools so schemas append cleanly.
        # The meta also reaches through to its child sessions' logs
        # (persistent children, per-delegation dirs, fan-out branches), so
        # a long investigation can recover specialist-level actions, not
        # just the delegation ledger's summaries.
        from ...session_events import register_history_tools

        def _child_logs():
            base = Path(self.orch.base_dir)
            logs = []
            for sub in ("analysis", "planning", "simulation", "fanout"):
                root = base / sub
                if root.is_dir():
                    for p in sorted(root.rglob("events.jsonl")):
                        logs.append((str(p.parent.relative_to(base)), p))
            return logs

        register_history_tools(
            self._register_tool,
            lambda: Path(self.orch.base_dir) / "events.jsonl",
            child_logs_fn=_child_logs,
        )
        # Name the scope surface in the schema (the shared registrar keeps
        # the parameter undeclared for single-session orchestrators).
        for schema in self.openai_schemas:
            fn = schema.get("function", {})
            if fn.get("name") == "search_session_history":
                fn["parameters"]["properties"]["scope"] = {
                    "type": "string",
                    "enum": ["own", "children", "all"],
                    "description": (
                        "own (default): this meta session's log. children: "
                        "the specialist child sessions' logs (delegations "
                        "and fan-out branches), hits labeled with their "
                        "session. all: both."
                    ),
                }
            if fn.get("name") == "get_history_events":
                fn["parameters"]["properties"]["session"] = {
                    "type": "string",
                    "description": (
                        "Optional `session` label from a children/all "
                        "search hit, to drill into that child's log "
                        "(default: this session's own log)."
                    ),
                }

    def _register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: list = None,
    ):
        """Register a tool in OpenAI function-calling format."""
        self.functions_map[name] = func
        self.openai_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        })

    def _vision_model(self):
        """A CLEAN model instance for one-off vision calls (view_image).

        The meta's chat model carries the orchestrator system prompt and the
        full tool registry — a describe-this-image call through it gets
        answered in the meta's persona ('I'll view the image...') instead of
        with an actual description (seen live on the Bedrock path). Same
        model/credentials, no system instruction, no tools. Lazily built and
        cached.
        """
        if getattr(self, "_vision_model_cache", None) is None:
            if self.orch.base_url:
                from ...wrappers.openai_wrapper import OpenAIAsGenerativeModel
                self._vision_model_cache = OpenAIAsGenerativeModel(
                    model=self.orch.model_name,
                    api_key=self.orch.api_key,
                    base_url=self.orch.base_url,
                )
            else:
                from ...wrappers.litellm_wrapper import LiteLLMGenerativeModel
                self._vision_model_cache = LiteLLMGenerativeModel(
                    model=self.orch.model_name,
                    api_key=self.orch.api_key,
                )
        return self._vision_model_cache

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name; always returns a JSON string."""
        result = self._dispatch_tool(tool_name, **kwargs)
        # One bounded line per tool call into the session's events.jsonl
        # (no-op when the thread has no bound log). See session_events.
        from ...session_events import append_event
        append_event(tool_name, kwargs, result)
        return result

    def _dispatch_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.functions_map:
            return json.dumps({
                "status": "error",
                "message": f"Tool '{tool_name}' not found",
            })

        # A tool call whose arguments hit the output-token cap mid-generation
        # arrives as VALID but incomplete JSON — later keys simply absent —
        # and dispatching it raises a bare TypeError about a missing
        # positional argument. Seen repeatedly on delegate_to_planning, whose
        # `task` brief runs to thousands of words: the model spends a whole
        # round trip re-emitting the same oversized call. The planning
        # orchestrator has guarded this for a while; the meta had not.
        missing = [p for p in self._required_params(tool_name)
                   if p not in kwargs]
        if missing:
            logging.warning(f"Tool {tool_name}: missing {missing} "
                            "(likely a truncated tool call)")
            # Advice has to fit the tool. The delegate_* tools carry a long
            # free-text brief and can shed weight into `context`; view_document
            # (paths) and run_fanout (branches) cannot, and telling them to
            # would be nonsense.
            import inspect as _inspect
            try:
                accepted = set(_inspect.signature(
                    self.functions_map[tool_name]).parameters)
            except (TypeError, ValueError):
                accepted = set()
            if {"task", "context"} <= accepted:
                how = ("Re-send it SHORTER: keep the essential instruction in "
                       "`task` and move supporting detail into `context`, "
                       "rather than re-sending the same text.")
            else:
                how = ("Re-send the call with every required argument. If the "
                       "payload is large, split the work across several "
                       "smaller calls rather than repeating this one.")
            return json.dumps({
                "status": "error",
                "tool": tool_name,
                "message": (
                    f"Missing required argument(s): {', '.join(missing)}. "
                    f"The call was most likely truncated by the response "
                    f"length limit. {how}"
                ),
            })

        try:
            return self.functions_map[tool_name](**kwargs)
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                import inspect as _inspect
                try:
                    accepted = list(_inspect.signature(
                        self.functions_map[tool_name]).parameters)
                except (TypeError, ValueError):
                    accepted = []
                logging.warning(f"Tool {tool_name}: {e}")
                return json.dumps({
                    "status": "error",
                    "tool": tool_name,
                    "message": (f"{e}. Accepted arguments: "
                                f"{', '.join(accepted) or 'unknown'}. "
                                "Re-send the call using only those."),
                })
            logging.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name,
            })
        except Exception as e:
            logging.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": str(e),
                "tool": tool_name,
            })

    def _required_params(self, tool_name: str) -> list:
        """Schema-declared required parameter names for a tool."""
        for schema in getattr(self, "openai_schemas", []) or []:
            fn = schema.get("function", {})
            if fn.get("name") == tool_name:
                return fn.get("parameters", {}).get("required", []) or []
        return []

    def _register_all_tools(self):
        """Register the meta-agent's delegation and introspection tools."""

        # -- delegate_to_analysis -------------------------------------------
        def delegate_to_analysis(task: str, context: dict = None,
                                 context_from: list = None,
                                 label: str = None,
                                 data_path: str = None,
                                 metadata: str = None) -> str:
            print("  " + _handoff(f"🧪 Delegating to analysis specialist: {_task_summary(task)}"))
            return self.orch._delegate("analysis", task, context, context_from,
                                       label, data_path=data_path,
                                       metadata=metadata)

        self._register_tool(
            func=delegate_to_analysis,
            name="delegate_to_analysis",
            description=(
                "Delegate an experimental-data-analysis task to the analysis "
                "specialist (microscopy, spectroscopy, curve fitting, "
                "hyperspectral datacubes, quality assessment, feature "
                "extraction, novelty checks). The specialist runs autonomously "
                "with no interactive user and returns a structured JSON result "
                "(status, summary, key_findings, files_produced, "
                "suggested_followups, warnings, delegation_index). `task` must "
                "be a complete, self-contained instruction including absolute "
                "paths to any data files. If the user supplies COMPANION / "
                "REFERENCE datasets alongside the primary — e.g. an empty-sample "
                "or baseline spectrum, an incident-beam / I0 reference, or a "
                "co-registered channel — name "
                "their absolute paths and their role (subtract / divide-by / "
                "mask-with) in `task` (or `context`); the specialist passes them "
                "through `run_analysis`'s `auxiliary_data`/`auxiliary_label` so "
                "the generated code can use them as numerical operands. "
                "For an ENSEMBLE / best-of-N request — several INDEPENDENT "
                "analysis trajectories over the SAME single dataset, compared by "
                "a judge (NOT several different datasets — that is "
                "`delegate_to_analyses`) — keep it ONE delegation and state the "
                "candidate count in `task` (e.g. 'run as best-of-3: 3 "
                "independent candidate analyses'); the specialist maps that to "
                "`run_analysis`'s `n_candidates`. Curve fitting, image, and "
                "hyperspectral all support this."
            ),
            parameters={
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained analysis instruction.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Optional upstream findings / file paths (e.g. from an "
                        "earlier delegation) to inform the task. May include "
                        "companion/reference dataset paths and their role "
                        "(baseline to subtract, reference to divide by, channel to "
                        "mask with) for the specialist to pass as auxiliary operands."
                    ),
                },
                "context_from": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "delegation_index numbers of earlier delegations whose "
                        "findings you threaded into `context` — records provenance."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "REQUIRED short label for the UI delegation tree — a "
                        "2-5 word noun phrase naming the data type being "
                        "analyzed (e.g. '1-D Raman spectra', 'STEM image', "
                        "'hyperspectral datacube'). NOT a sentence or a "
                        "restatement of the task."
                    ),
                },
                "data_path": {
                    "type": "string",
                    "description": (
                        "Absolute path of THIS delegation's primary dataset "
                        "(also stated in `task`). ALWAYS provide it when the "
                        "task analyzes a dataset: it lets a later "
                        "`fuse_delegations` that mixes this delegation with "
                        "others re-run the complementarity gate — without it "
                        "an incremental fusion demotes to ungated (no "
                        "computed reconciliation)."
                    ),
                },
                "metadata": {
                    "type": "string",
                    "description": (
                        "Optional: metadata JSON path or short inline "
                        "description of the dataset (feeds a later fusion's "
                        "complementarity re-gate)."
                    ),
                },
            },
            required=["task", "label"],
        )

        # -- delegate_to_planning -------------------------------------------
        def delegate_to_planning(task: str, context: dict = None,
                                 context_from: list = None,
                                 label: str = None) -> str:
            print("  " + _handoff(f"📋 Delegating to planning specialist: {_task_summary(task)}"))
            return self.orch._delegate("planning", task, context, context_from, label)

        self._register_tool(
            func=delegate_to_planning,
            name="delegate_to_planning",
            description=(
                "Delegate an experimental-campaign-planning task to the "
                "planning specialist (experiment design, multi-objective "
                "Bayesian optimization, hypothesis generation, deciding what "
                "to measure or run next). The specialist runs autonomously "
                "with no interactive user and returns a structured JSON result "
                "(status, summary, key_findings, files_produced, "
                "suggested_followups, warnings, delegation_index). `task` must "
                "be a complete, self-contained instruction. The specialist "
                "supports best-of-N plan generation (an LLM judge picks among "
                "distinct candidate strategies) and DEFAULTS to best-of-3 for "
                "a campaign's first plan. Only add an explicit "
                "'use n_candidates=N on generate_initial_plan' instruction "
                "when the user wants a specific width — including N=1 for a "
                "single plan. Keep the scientific goal itself SINGULAR "
                "(never phrase the objective as 'propose N strategies'; the "
                "tool parameter provides the multiplicity). When the user is "
                "brainstorming/ideating rather than planning an executable "
                "run, add 'use selection_profile=ideation' to the task so "
                "the candidate judge weights novelty over feasibility."
            ),
            parameters={
                "task": {
                    "type": "string",
                    "description": "Complete, self-contained planning instruction.",
                },
                "context": {
                    "type": "object",
                    "description": (
                        "Optional upstream findings / file paths (e.g. analysis "
                        "key_findings) to inform the task."
                    ),
                },
                "context_from": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "delegation_index numbers of earlier delegations whose "
                        "findings you threaded into `context` — records provenance."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "REQUIRED short label for the UI delegation tree — a "
                        "2-5 word noun phrase naming the focus of the planning "
                        "task (e.g. 'follow-up BO campaign', 'experiment "
                        "design'). NOT a sentence."
                    ),
                },
            },
            required=["task", "label"],
        )

        # -- assess_complementarity -----------------------------------------
        def assess_complementarity(datasets: list) -> str:
            print(f"  🔎 Assessing complementarity of {len(datasets or [])} dataset(s)...")
            return self.orch._assess_complementarity(datasets)

        self._register_tool(
            func=assess_complementarity,
            name="assess_complementarity",
            description=(
                "Judge whether two or more datasets are GENUINELY COMPLEMENTARY "
                "— same physical system, different (non-redundant) information, "
                "and a shared join axis to reconcile them on — and partition "
                "them: a `fanout_set` worth analyzing together, `redundant_"
                "clusters` (duplicates), and `unrelated` outliers. Call this "
                "BEFORE proposing `delegate_to_analyses` so you can show the "
                "user the verdict and discuss it. Read-only; does not run any "
                "analysis. (The same gate also runs INSIDE delegate_to_analyses, "
                "so a non-complementary set is refused even if you skip this.)"
            ),
            parameters={
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string",
                                     "description": "Absolute path to the dataset."},
                            "role": {"type": "string",
                                     "description": "Optional: what this dataset is / "
                                                    "its stated relationship to the others."},
                            "metadata": {"type": "string",
                                         "description": "Optional: path to a metadata "
                                                        "JSON, or an inline description."},
                        },
                        "required": ["path"],
                    },
                    "description": "The datasets to assess (>= 2).",
                },
            },
            required=["datasets"],
        )

        # -- delegate_to_analyses (parallel fan-out, full-mesh aux) ----------
        def delegate_to_analyses(branches: list,
                                 figure_style: str = None,
                                 harmonize: bool = False) -> str:
            print(f"  🔀 Parallel analysis over {len(branches or [])} dataset(s)"
                  + (" (harmonized pipeline replay)" if harmonize else "")
                  + "...")
            return self.orch._run_fanout(branches, figure_style=figure_style,
                                         harmonize=harmonize)

        self._register_tool(
            func=delegate_to_analyses,
            name="delegate_to_analyses",
            description=(
                "Run SEVERAL analysis branches CONCURRENTLY over complementary "
                "datasets, then fuse with `fuse_delegations`. Branches run "
                "INDEPENDENTLY by default (fusion reconciles their reduced "
                "results — independence is what makes cross-dataset agreement "
                "meaningful); a CO-REGISTERED set (gate join_type) additionally "
                "wires the companions in as auxiliary operands, and a branch "
                "may opt into a steering hint via `steer`. Use this — NOT "
                "repeated `delegate_to_analysis` — when the user has 2+ datasets "
                "that are complementary measurements of ONE system and you want "
                "each analysis informed by the others plus a final cross-dataset "
                "synthesis. A complementarity GATE runs first and PRUNES to the "
                "genuinely-complementary subset: a redundant or unrelated set is "
                "declined (analyze those independently instead). This is for "
                "multiple DIFFERENT complementary datasets — NOT for several "
                "independent trajectories over ONE dataset; for that ensemble / "
                "best-of-N, use a single `delegate_to_analysis` and state the "
                "candidate count in its `task`. In AUTOPILOT the "
                "user is asked to confirm before launching; branches always run "
                "AUTONOMOUSLY (no per-branch approval pauses — concurrent prompts "
                "can't interleave). Each branch's `task` must be a complete, "
                "self-contained instruction with the absolute data path; the "
                "companions are wired in automatically. Returns the per-branch "
                "delegation_index list and the indices to pass to "
                "`fuse_delegations`."
            ),
            parameters={
                "branches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "data_path": {"type": "string",
                                          "description": "Absolute path to THIS branch's "
                                                         "primary dataset. Branches MAY "
                                                         "share one path (a directory "
                                                         "holding several measurement "
                                                         "series): each series is its own "
                                                         "branch, distinguished by its "
                                                         "task — never merge distinct "
                                                         "datasets into one branch."},
                            "task": {"type": "string",
                                     "description": "Complete, self-contained analysis "
                                                    "instruction for this dataset."},
                            "pattern": {"type": "string",
                                        "description": "Filename glob selecting THIS "
                                                       "branch's files when data_path is "
                                                       "a directory with several datasets "
                                                       "(e.g. 'sample_*C.txt'). ALWAYS "
                                                       "provide it when branches share a "
                                                       "directory."},
                            "label": {"type": "string",
                                      "description": "Short 2-5 word UI label (the data "
                                                     "type, e.g. 'STEM image')."},
                            "metadata": {"type": "string",
                                         "description": "Optional: path to a metadata JSON "
                                                        "or inline description (also feeds "
                                                        "the complementarity gate)."},
                            "steer": {"type": "boolean",
                                      "description": "Explicit opt-in (default false): give "
                                                     "THIS branch a change-point hint from "
                                                     "each companion SERIES (cheap "
                                                     "unsupervised reduction) as an "
                                                     "additive-only hypothesis — useful when "
                                                     "the branch risks fitting the wrong "
                                                     "model order. TRADE: steering spends "
                                                     "the branch's independence; fusion is "
                                                     "told (informed_by) and discounts its "
                                                     "agreement with the steering companion "
                                                     "as partly by construction. Opt in only "
                                                     "when guidance is worth that cost."},
                        },
                        "required": ["data_path", "task", "label"],
                    },
                    "description": "The analysis branches to run in parallel (>= 2).",
                },
                "figure_style": {
                    "type": "string",
                    "description": (
                        "Optional figure-presentation preference applied to "
                        "EVERY branch's figures AND the fusion figure (e.g. "
                        "'place legends outside the axes — never on top of "
                        "data', 'use colorblind-safe colors'). Set it when "
                        "the user expresses any preference about how plots "
                        "should look; leave unset otherwise."
                    ),
                },
                "harmonize": {
                    "type": "boolean",
                    "description": (
                        "Opt-in (default false): harmonized pipeline replay "
                        "for SAME-TECHNIQUE sibling datasets (e.g. one "
                        "hyperspectral cube per experimental condition). The "
                        "FIRST branch runs as the pipeline DONOR; every other "
                        "branch then REPLAYS the donor's approved analysis "
                        "script verbatim, so extracted magnitudes are "
                        "measured by ONE frozen pipeline and are directly "
                        "comparable across datasets (fusion is told). Use it "
                        "when the goal is a cross-condition comparison of "
                        "the same observable — and put the most "
                        "representative dataset FIRST as the donor. In this "
                        "mode the complementarity gate only checks that the "
                        "datasets measure the SAME system (unrelated ones "
                        "are pruned); its redundancy criterion is waived — "
                        "a same-technique series is the intended input, so "
                        "do NOT avoid this tool because the cubes 'look "
                        "redundant'. Leave "
                        "false for cross-modality sets (different techniques "
                        "cannot share a script). Falls back loudly to "
                        "independent branches if the donor yields no "
                        "approved script."
                    ),
                },
            },
            required=["branches"],
        )

        # -- fuse_delegations -----------------------------------------------
        def fuse_delegations(delegation_indices: list, focus: str = None) -> str:
            print("  " + _handoff(f"🧬 Fusing delegations {delegation_indices}..."))
            return self.orch._fuse_delegations(delegation_indices, focus)

        self._register_tool(
            func=fuse_delegations,
            name="fuse_delegations",
            description=(
                "Reconcile the findings of two or more COMPLETED analysis "
                "delegations (typically the branches from `delegate_to_analyses`, "
                "but any successful delegations work) into ONE cross-dataset "
                "scientific narrative + synthesized claims. Reconciles spatial / "
                "shared-axis / local-vs-bulk / complementary-observable evidence, "
                "and ATTACHES one representative figure per dataset to the "
                "(multimodal) synthesis so spatial correlations are verified from "
                "the actual plots, not just the text. Writes BOTH a JSON and a "
                "human-readable HTML fusion report (figures inline) — surface the "
                "`report_html_path` to the user. Crucially, 'no significant "
                "correlation found' is a VALID result — it will not manufacture a "
                "correlation the findings don't support. Pass the "
                "`delegation_index` numbers to fuse; optional `focus` weights the "
                "synthesis toward a specific question."
            ),
            parameters={
                "delegation_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "delegation_index numbers of the completed "
                                   "delegations to fuse (>= 2).",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional question/aspect to weight the synthesis toward.",
                },
            },
            required=["delegation_indices"],
        )

        # -- delegate_to_simulation -----------------------------------------
        def delegate_to_simulation(task: str, context: dict = None,
                                   context_from: list = None,
                                   label: str = None) -> str:
            # Guarded import INSIDE the function: scilink.agents.sim_agents
            # hard-imports the optional `ase` dependency, and the meta module
            # must stay importable without it. Return a clean error if the
            # [sim] extra is absent rather than crashing the meta.
            try:
                from ..sim_agents.simulation_orchestrator import (  # noqa: F401
                    SimulationOrchestratorAgent,
                )
            except ImportError as e:
                return json.dumps({
                    "status": "error",
                    "message": ("Simulation support requires the optional [sim] "
                                "extra (pip install scilink[sim])."),
                    "detail": str(e),
                })
            print(f"  ⚛️  Delegating to simulation specialist: {task[:80]}...")
            return self.orch._delegate("simulation", task, context,
                                       context_from, label)

        self._register_tool(
            func=delegate_to_simulation,
            name="delegate_to_simulation",
            description=(
                "Delegate a computational-simulation task to the simulation "
                "specialist (periodic DFT, classical molecular dynamics, and "
                "ML-potential-driven MD). From a natural-language goal it "
                "autonomously builds and validates an atomic structure, "
                "generates and runs engine inputs (VASP, LAMMPS), and refines "
                "them — no input data file required. The specialist runs "
                "autonomously and returns a structured JSON result (status, "
                "summary, key_findings, files_produced, structures, "
                "suggested_followups, warnings, delegation_index). `task` must "
                "be a complete, self-contained instruction (system + goal); the "
                "specialist derives the chemistry, force field, and parameters "
                "itself. To start from an existing structure or an upstream "
                "result, name its absolute path in `task`/`context`."
            ),
            parameters={
                "task": {
                    "type": "string",
                    "description": ("Complete, self-contained simulation "
                                    "instruction (system + goal)."),
                },
                "context": {
                    "type": "object",
                    "description": ("Optional upstream findings / file paths "
                                    "(e.g. a structure path, or analysis "
                                    "key_findings) to inform the task."),
                },
                "context_from": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": ("delegation_index numbers of earlier "
                                    "delegations whose findings you threaded "
                                    "into `context` — records provenance."),
                },
                "label": {
                    "type": "string",
                    "description": ("REQUIRED short label for the UI delegation "
                                    "tree — a 2-5 word noun phrase naming the "
                                    "system/method (e.g. 'Zn(OTf)2 MD', 'Si DFT "
                                    "relaxation', 'MACE-MD water'). NOT a "
                                    "sentence or a restatement of the task."),
                },
            },
            required=["task", "label"],
        )

        # -- summarize_session_state ----------------------------------------
        def summarize_session_state() -> str:
            return self.orch._session_state_summary()

        self._register_tool(
            func=summarize_session_state,
            name="summarize_session_state",
            description=(
                "Report the cross-specialist session state: which specialists "
                "have been instantiated, how many delegations have run, and "
                "per-specialist counters (analyses run, optimization targets, "
                "collected data points). Read-only."
            ),
            parameters={},
            required=[],
        )

        # -- get_delegation_history -----------------------------------------
        def get_delegation_history(limit: int = None) -> str:
            return self.orch._delegation_history(limit)

        self._register_tool(
            func=get_delegation_history,
            name="get_delegation_history",
            description=(
                "Retrieve the delegation ledger — the results of prior "
                "delegations (status, summary, key_findings, files_produced, "
                "suggested_followups). Use it to pull an earlier specialist's "
                "result and thread the relevant pieces as the `context` "
                "argument of the next delegate_to_* call. Optional `limit` "
                "returns only the most recent N entries."
            ),
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Return only the most recent N delegations.",
                },
            },
            required=[],
        )

        # -- attach_knowledge_base -------------------------------------------
        def attach_knowledge_base(path: str = None) -> str:
            print("  📚 Tool: Attaching knowledge base...")
            try:
                attached = self.orch.attach_knowledge_dir(path)
                from ...knowledge.kb_store import (
                    read_manifest, embedding_compat_warning,
                )
                out = {"status": "success", "knowledge_dir": attached}
                warn = embedding_compat_warning(
                    read_manifest(Path(attached)), self.orch.embedding_model
                )
                if warn:
                    out["warning"] = (
                        f"{warn} Surface this to the user: retrieval from "
                        "this KB may be degraded or unavailable in this "
                        "session."
                    )
                return json.dumps(out)
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        self._register_tool(
            func=attach_knowledge_base,
            name="attach_knowledge_base",
            description=(
                "Ground planning delegations in a stable knowledge base — "
                "either a NAMED KB from the user's persistent store (listed "
                "in the system prompt; pass its name as `path`) or a folder "
                "holding a persisted KB index. With no `path`, attaches the "
                "launch directory's detached shared KB, if any. Call this "
                "ONLY after the user has agreed (or explicitly asked) to use "
                "their knowledge base — except when running autonomously, "
                "where you decide from the listed sources' relevance to the "
                "task. Takes effect for all subsequent planning delegations "
                "and persists across session resume."
            ),
            parameters={
                "path": {
                    "type": "string",
                    "description": (
                        "A named KB from the store (e.g. 'produced-water'), "
                        "or a knowledge directory path. Omit to attach the "
                        "launch directory's detached shared KB."
                    ),
                },
            },
            required=[],
        )

        # -- add_to_knowledge_base -------------------------------------------
        def add_to_knowledge_base(paths, name: str = None) -> str:
            if isinstance(paths, str):
                paths = [paths]
            print(f"  📚 Tool: Adding {len(paths)} document(s) to knowledge base...")
            try:
                manifest = self.orch.add_documents_to_kb(paths, name=name)
                return json.dumps({
                    "status": "success",
                    "knowledge_base": manifest.get("name"),
                    "n_vectors": manifest.get("n_vectors"),
                    "sources": manifest.get("sources", [])[-10:],
                })
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        self._register_tool(
            func=add_to_knowledge_base,
            name="add_to_knowledge_base",
            description=(
                "PERMANENTLY add document files to a named knowledge base in "
                "the user's persistent store — the KB is a shared artifact "
                "reused across sessions, so call this ONLY when the user "
                "explicitly asks to add/save documents to their knowledge "
                "base (in any autonomy mode; never on your own initiative). "
                "Embeds only the new documents, using the KB's own embedding "
                "model. If that KB is attached to this session, the session "
                "picks up the additions immediately. For using a document in "
                "just this session's planning, pass it in the delegation "
                "task instead."
            ),
            parameters={
                "paths": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Absolute path(s) of the document file(s) "
                                   "or folder(s) to add.",
                },
                "name": {
                    "type": "string",
                    "description": ("Target KB name. Omit to use the "
                                    "currently attached named KB."),
                },
            },
            required=["paths"],
        )

        # -- save_checkpoint -------------------------------------------------
        def save_checkpoint() -> str:
            print("  💾 Tool: Saving meta checkpoint...")
            try:
                path = self.orch.save_checkpoint()
                return json.dumps({
                    "status": "success",
                    "checkpoint_path": path,
                    "delegations_saved": len(self.orch._delegation_ledger),
                })
            except Exception as e:
                return json.dumps({"status": "error", "message": str(e)})

        self._register_tool(
            func=save_checkpoint,
            name="save_checkpoint",
            description=(
                "Save the meta session state (mode, delegation ledger, chat "
                "history) to checkpoint.json so the session can be resumed "
                "later. The state is ALREADY auto-saved every few turns, so "
                "call this ONLY when the user asks to save/checkpoint the "
                "session or says they are stopping — never as a routine "
                "end-of-turn step."
            ),
            parameters={},
            required=[],
        )

        # -- review_distilled_skills ----------------------------------------
        def review_distilled_skills(action: str = "list", skill: str = None,
                                    to_domain: str = None, staged: str = None,
                                    into: str = None, technique: str = None) -> str:
            from ...skills._shared import _memory, _staging
            from ...skills.loader import list_all_skills

            def _llm_call(prompt: str) -> str:
                r = self.orch.model.generate_content(contents=[prompt])
                return r.text if hasattr(r, "text") else str(r)

            act = (action or "list").lower()

            # Curated (built-in) skills are ALREADY available and auto-selected by
            # the domain agents — surface them so the orchestrator has the full
            # picture and never claims "no skill exists" or proposes consolidating
            # a NEW skill that just duplicates a curated one.
            try:
                curated = list_all_skills()
            except Exception:
                curated = {}

            # --- skills (graduated_skills) ---
            if act == "list":
                rows = _memory.list_memory(provisional=True)
                print(f"  🧠 {len(rows)} provisional skill(s); "
                      f"{sum(len(v) for v in curated.values())} curated skill(s) "
                      f"already available.")
                return json.dumps({"status": "success", "action": "list",
                                   "curated_skills": curated,
                                   "provisional_skills": rows}, default=str)

            if act == "list_staged":
                rows = _staging.list_staged()
                # Per-technique readiness so the LLM accumulates before suggesting a
                # NEW skill: consolidation needs >= consolidate_min_n() examples of a
                # technique; a single staged solution is too idiosyncratic to graduate
                # (upgrading an EXISTING skill via `upgrade` is exempt).
                need = _staging.consolidate_min_n()
                by_tech: dict = {}
                for r in rows:
                    key = f"{r.get('domain')}/{r.get('technique') or 'unlabeled'}"
                    by_tech[key] = by_tech.get(key, 0) + 1
                techniques = [
                    {"technique": k, "n_staged": n, "ready_to_consolidate": n >= need}
                    for k, n in sorted(by_tech.items())
                ]
                print(f"  🧠 {len(rows)} staged solution(s) awaiting distillation "
                      f"({sum(1 for t in techniques if t['ready_to_consolidate'])} "
                      f"technique(s) ready to consolidate; threshold {need}); "
                      f"{sum(len(v) for v in curated.values())} curated skill(s) "
                      f"already available.")
                return json.dumps({"status": "success", "action": "list_staged",
                                   "consolidate_threshold": need,
                                   "techniques": techniques,
                                   "curated_skills": curated,
                                   "staged_solutions": rows}, default=str)

            try:
                if act in ("show", "promote", "discard", "prune"):
                    if not skill or "/" not in skill:
                        return json.dumps({"status": "error",
                            "message": "Pass skill as '<domain>/<name>'."})
                    domain, name = skill.split("/", 1)
                    if act == "show":
                        return json.dumps({"status": "success", "action": "show", "skill": skill,
                            "markdown": _memory.show_memory(domain, name)}, default=str)
                    if act == "promote":
                        res = _memory.promote_memory(domain, name, to_domain=to_domain)
                        print(f"  ✅ Promoted '{skill}' (now auto-routable).")
                        return json.dumps({"status": "success", "action": "promote", **res}, default=str)
                    res = _memory.prune_memory(domain, name)
                    print(f"  🗑️  Discarded '{skill}'.")
                    return json.dumps({"status": "success", "action": "discard", **res}, default=str)

                if act == "upgrade":
                    # Merge a staged solution INTO an existing skill.
                    from ..exp_agents.instruct import (
                        KNOWLEDGE_TO_SKILL_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS,
                    )
                    if not staged or "/" not in staged or not into or "/" not in into:
                        return json.dumps({"status": "error",
                            "message": "Pass staged='<domain>/<id>' and into='<domain>/<name>'."})
                    sdomain, sid = staged.split("/", 1)
                    tdomain, tname = into.split("/", 1)
                    res = _staging.upgrade_skill_from_staged(
                        sdomain, [sid], target_domain=tdomain, target_name=tname,
                        llm_call=_llm_call,
                        fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                        update_template=SKILL_UPDATE_INSTRUCTIONS)
                    if res.get("status") == "success":
                        print(f"  ✅ Upgraded '{into}' from staged {sid}.")
                    return json.dumps({"action": "upgrade", **res}, default=str)

                if act == "consolidate":
                    from ..exp_agents.instruct import (
                        T2_CONSOLIDATION_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS,
                    )
                    if not technique or "/" not in technique:
                        return json.dumps({"status": "error",
                            "message": "Pass technique='<domain>/<technique>'."})
                    cdomain, ctech = technique.split("/", 1)
                    res = _staging.consolidate_technique(
                        cdomain, ctech, llm_call=_llm_call,
                        consolidation_template=T2_CONSOLIDATION_INSTRUCTIONS,
                        update_template=SKILL_UPDATE_INSTRUCTIONS)
                    if res.get("status") == "success":
                        print(f"  ✅ Consolidated {res.get('n_examples')} staged → '{cdomain}/auto_{ctech}'.")
                    return json.dumps({"action": "consolidate", **res}, default=str)
            except FileNotFoundError as e:
                return json.dumps({"status": "error", "message": str(e)})
            return json.dumps({"status": "error", "message": f"Unknown action: {action}"})

        self._register_tool(
            func=review_distilled_skills,
            name="review_distilled_skills",
            description=(
                "Review and act on learned knowledge from hard runs the agent solved "
                "from scratch (constraint-annealing hot stage). "
                "runs. Such runs STAGE a raw solution (delegate_to_analysis reports "
                "`staged_solutions`); skills are produced from staged solutions two "
                "ways, both review-gated: `upgrade` merges ONE staged solution into an "
                "EXISTING skill, or `consolidate` distills all staged solutions of a "
                "technique into a NEW (provisional) skill. "
                "IMPORTANT — full picture first: `list`/`list_staged` also return "
                "`curated_skills` (the built-in skills ALREADY available and "
                "auto-applied by the domain agents). Before suggesting anything, "
                "consult it: a run that used a curated skill was NOT 'solved from "
                "scratch with no skill', and you must NOT `consolidate` a NEW skill "
                "that duplicates a curated one — if a curated skill already covers the "
                "technique, prefer `upgrade`-ing it (or just improving it) over "
                "creating a duplicate, or leave the solution staged. "
                "Prefer `upgrade` when a matching skill exists. Do NOT `consolidate` a "
                "technique into a new skill until it has accumulated enough examples — "
                "`list_staged` reports `ready_to_consolidate` per technique against "
                "`consolidate_threshold`; if not ready, leave it staged to accumulate. "
                "Also manages already-distilled provisional skills. Actions: `list` "
                "(provisional + curated skills), `list_staged` (staged solutions by "
                "technique + curated skills), `show` a skill, "
                "`promote`/`discard` a skill, `upgrade` (needs `staged` + `into`), "
                "`consolidate` (needs `technique`). In autopilot, ASK the user before "
                "upgrading/consolidating/promoting/discarding; in autonomous mode, "
                "leave things staged and just note them."
            ),
            parameters={
                "action": {
                    "type": "string",
                    "enum": ["list", "list_staged", "show", "promote", "discard",
                             "upgrade", "consolidate"],
                    "description": "What to do (default: list).",
                },
                "skill": {"type": "string",
                          "description": "Skill ref '<domain>/<name>' (show/promote/discard)."},
                "to_domain": {"type": "string",
                              "description": "Optional: on promote, move the skill into a curated domain."},
                "staged": {"type": "string",
                           "description": "Staged solution ref '<domain>/<id>' (for upgrade)."},
                "into": {"type": "string",
                         "description": "Target skill '<domain>/<name>' to upgrade into (for upgrade)."},
                "technique": {"type": "string",
                              "description": "Technique ref '<domain>/<technique>' (for consolidate)."},
            },
            required=[],
        )

        # -- inspect_uploads ------------------------------------------------
        def inspect_uploads(path: str = None) -> str:
            base = Path(path) if path else (self.orch.base_dir / "uploads")
            print(f"  🔍 Inspecting uploads at {base} ...")
            if not base.exists():
                return json.dumps({
                    "status": "error",
                    "message": f"Path not found: {base}",
                })
            if base.is_file():
                files = [base]
                directory = str(base.parent)
            else:
                files = sorted(
                    f for f in base.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                )
                directory = str(base)
            probes = [_probe_file(f) for f in files[:_PROBE_MAX_FILES]]
            return json.dumps({
                "status": "success",
                "directory": directory,
                "n_files": len(files),
                "truncated": len(files) > _PROBE_MAX_FILES,
                "files": probes,
            }, default=str)

        self._register_tool(
            func=inspect_uploads,
            name="inspect_uploads",
            description=(
                "Inspect uploaded files to decide how to route them. Returns a "
                "lightweight CONTENT probe of each file — array shape/dtype, "
                "table column names, document text snippets, JSON keys — so you "
                "classify from evidence, not from filenames. Call this FIRST "
                "whenever the user refers to uploaded files or points you at a "
                "folder. With no argument it inspects the meta session's "
                "uploads/ directory; pass `path` for a specific file or folder. "
                "Read-only — use the result only to choose a specialist, never "
                "to interpret the data yourself."
            ),
            parameters={
                "path": {
                    "type": "string",
                    "description": (
                        "Optional file or directory to inspect. Defaults to "
                        "the meta session's uploads/ directory."
                    ),
                },
            },
            required=[],
        )

        # ----- materialize_sidecars (manifest -> per-file sidecar JSONs) -------
        # Conditions supplied as a free-text manifest reach the specialist only as
        # prose unless turned into per-file sidecars; this writes <stem>.json next
        # to each data file so the analysis feature table gets a column per
        # condition. Deterministic, additive, and NEVER overwrites an existing
        # sidecar (so genuine per-file metadata can't be clobbered).
        def materialize_sidecars(conditions, data_dir: str = None) -> str:
            if isinstance(conditions, str):
                try:
                    conditions = json.loads(conditions)
                except Exception:
                    return json.dumps({"status": "error", "message":
                                       "conditions must be a JSON object {filename: {key: value}}"})
            if not isinstance(conditions, dict) or not conditions:
                return json.dumps({"status": "error", "message":
                                   "conditions must be a non-empty {filename: {key: value}} mapping"})
            base = Path(data_dir) if data_dir else None
            written, skipped, unresolved = [], [], []
            for fname, cond in conditions.items():
                if not isinstance(cond, dict):
                    unresolved.append(f"{fname} (conditions not an object)")
                    continue
                p = Path(fname)
                cands = [p] if p.is_absolute() else (
                    ([base / fname] if base is not None else []) + [Path.cwd() / fname])
                target = next((c for c in cands if c.is_file()), None)
                if target is None:
                    unresolved.append(str(fname))
                    continue
                sidecar = target.with_suffix(".json")
                if sidecar.exists():                       # never clobber real metadata
                    skipped.append(str(sidecar))
                    continue
                scalars = {k: v for k, v in cond.items()
                           if isinstance(v, (int, float, str, bool))}
                sidecar.write_text(json.dumps(scalars, indent=2), encoding="utf-8")
                written.append(str(sidecar))
            return json.dumps({
                "status": "success" if written or skipped else "error",
                "sidecars_written": len(written), "paths": written,
                "skipped_existing": skipped, "unresolved": unresolved,
            })

        self._register_tool(
            func=materialize_sidecars,
            name="materialize_sidecars",
            description=(
                "Write per-file sidecar metadata JSONs from a conditions MANIFEST the "
                "user supplied as free text (a .txt/.csv/README mapping each data file "
                "to its experimental conditions, e.g. 'filename, temperature, pH'). Pass "
                "`conditions` as a JSON object {data_filename: {condition_key: value, "
                "...}} that you read from the manifest, plus `data_dir` (the folder "
                "holding the data files). It writes <stem>.json next to each data file so "
                "the analysis specialist's feature table gains one COLUMN per condition — "
                "conditions quoted only in the delegation task text do NOT become feature "
                "columns (which downstream optimization needs as inputs). Use this BEFORE "
                "delegating whenever per-file conditions arrive as a manifest and the data "
                "files have no matching sidecar JSONs. It never overwrites an existing "
                "sidecar, and only records conditions — it never transforms data."
            ),
            parameters={
                "conditions": {
                    "type": "object",
                    "description": (
                        "{data_filename: {condition_key: value}} read from the manifest; "
                        "values are scalars (numbers or strings). Filenames may be bare "
                        "(resolved against data_dir) or absolute."
                    ),
                },
                "data_dir": {
                    "type": "string",
                    "description": "Absolute path to the folder holding the data files "
                                   "(used to resolve bare filenames).",
                },
            },
            required=["conditions"],
        )

        # ----- prepare_inputs (lossless data/metadata split) ------------------
        # The meta's ONLY code-generation surface, restricted to LOSSLESS file
        # repackaging before delegation: split a single combined data+metadata
        # file into a data file + a metadata JSON. Round-trip verified; NEVER
        # used for analysis/computation — that is always delegated.
        def prepare_inputs(path) -> str:
            import hashlib as _hashlib
            from ...utils.file_prep import (prepare_inputs as _split_file,
                                            prepare_inputs_batch as _split_batch,
                                            stage_pairs_flat as _stage_flat)
            from ...executors import ScriptExecutor, require_sandbox_approval
            paths = [path] if isinstance(path, str) else list(path or [])
            if not paths:
                return json.dumps({"status": "error",
                                   "message": "No file path provided."})
            pths = [Path(p) for p in paths]
            missing = [str(p) for p in pths if not (p.exists() and p.is_file())]
            if missing:
                return json.dumps({"status": "error",
                                   "message": "File(s) not found: " + ", ".join(missing)})
            if not require_sandbox_approval(
                context="Meta agent file preparation (lossless data/metadata split)"
            ):
                return json.dumps({
                    "status": "error",
                    "message": "Code execution declined; cannot prepare the file(s). "
                               "Delegate them to the specialist as-is.",
                })
            probes = []
            for p in pths:
                try:
                    probes.append(_probe_file(p))
                except Exception:  # noqa: BLE001
                    probes.append(None)
            out_dir = self.orch.base_dir / "prepared"
            executor = ScriptExecutor(timeout=120)
            # max_retries=2 → 3 attempts: binary containers often need a correction
            # pass; the round-trip net keeps a bad accept unlikely, so retries cheap.
            if len(pths) == 1:
                result = _split_file(pths[0], model=self.orch.model, executor=executor,
                                     output_dir=out_dir, probe=probes[0],
                                     logger=self.logger, max_retries=2)
            else:
                result = _split_batch(pths, model=self.orch.model, executor=executor,
                                      output_dir=out_dir, probes=probes,
                                      logger=self.logger, max_retries=2)
                # Stage the verified pairs into ONE flat, stem-matched directory so
                # the specialist can consume the whole batch as a series directory
                # (run_analysis pairs each data file with its <stem>.json sidecar).
                # Without this the model is left with scattered per-file subfolders
                # and resorts to passing a path LIST, which the series loader rejects.
                if result.get("status") in ("success", "partial"):
                    try:
                        key = _hashlib.sha1(
                            "|".join(sorted(str(p) for p in pths)).encode()
                        ).hexdigest()[:8]
                        staged = _stage_flat(result.get("results", []),
                                             out_dir / f"series_{key}")
                        if staged["n"] > 1:
                            result["prepared_dir"] = staged["staged_dir"]
                    except Exception as e:  # noqa: BLE001
                        self.logger.warning(f"Flat staging skipped: {e}")
            return json.dumps(result, default=str)

        self._register_tool(
            func=prepare_inputs,
            name="prepare_inputs",
            description=(
                "Split combined file(s) that hold BOTH data and metadata into a "
                "separate data file + metadata JSON each, so the specialist receives "
                "clean (data, metadata) pairs. Single file → returns data_path + "
                "metadata_path. Pass a LIST of paths to split several SAME-TYPE files "
                "in ONE call → returns a per-file 'results' list + a 'summary'; thread "
                "the pairs into ONE batched delegation. When >=2 files split, the result "
                "also has a 'prepared_dir' — one flat folder of the cleaned data files "
                "each beside its stem-matched sidecar JSON; pass that 'prepared_dir' "
                "DIRECTORY as the delegation's data (do NOT pass the per-file data paths "
                "as a list — the series loader takes a directory), and never loop one "
                "delegation per file. In batch mode a single split is generated and reused across "
                "structurally identical files (each still round-trip verified), so it "
                "is cheaper and yields a uniform schema; a file that doesn't match "
                "falls back to its own split. Use after inspect_uploads when a probe "
                "shows data and metadata mixed in one file (HDF5/NeXus with attributes, "
                ".npz/.mat with data+meta keys, a CSV/text with a header/comment "
                "metadata block, a TIFF with tags/ImageDescription). Also use it on a "
                "METADATA-ONLY file — e.g. a pickled .npy header/object-array the probe "
                "reports as kind 'object_array' — where it deterministically serializes "
                "the container to a metadata JSON and returns data_path=null with "
                "metadata_only=true; pair that metadata with its sibling data file in "
                "the delegation. This is the META "
                "AGENT'S ONLY code-generation tool and it is STRICTLY LIMITED to "
                "lossless file repackaging: the generated code may only separate "
                "existing data from metadata and is round-trip verified (the "
                "reconstruction must match the original) — it NEVER transforms, "
                "computes, fits, or analyzes. All analysis is delegated. On error (no "
                "verified lossless split), do NOT silently delegate — tell the user "
                "and ask how to proceed (analyze as-is, or supply data and metadata "
                "separately); fall back to as-is only if no user. With a 'partial' "
                "batch result, surface which files failed and ask how to proceed."
            ),
            parameters={
                "path": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": (
                        "Absolute path to the combined data+metadata file to split, "
                        "OR a list of absolute paths to split several same-type files "
                        "in one batched call."
                    ),
                },
            },
            required=["path"],
        )

        # ----- view_image -----------------------------------------------------
        # Generic "view & describe an arbitrary image" — the meta itself has no
        # multimodal input path, so this is how a notebook photo / diagram /
        # screenshot / figure becomes content the meta can reason about. NOT
        # for scientific images (route those to analysis for quantification).

        _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff",
                       ".bmp", ".gif", ".webp"}
        _VIEW_IMAGE_DEFAULT_PROMPT = (
            "Describe the contents of this image in detail. If it contains "
            "any readable text — printed or handwritten — transcribe it "
            "faithfully, rendering tables as GitHub-flavored Markdown. "
            "Return your description and any transcription as plain text, "
            "with no extra commentary."
        )

        def view_image(paths, question: str = None) -> str:
            """Open one or more images and have the vision model describe
            (and transcribe text/tables in) them."""
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                return json.dumps({"status": "error",
                                   "message": "No image path provided."})
            print(f"  🖼️  Tool: Viewing {len(paths)} image(s)...")

            import io
            from PIL import Image as _PILImage
            from scilink.parsers.ocr import describe_image

            prompt = question.strip() if question else _VIEW_IMAGE_DEFAULT_PROMPT
            results, errors = [], []
            for p in paths:
                pp = Path(p)
                if not pp.is_file():
                    errors.append(f"Not a file: {p}")
                    continue
                if pp.suffix.lower() not in _IMAGE_EXTS:
                    errors.append(f"Not an image file: {p}")
                    continue
                try:
                    img = _PILImage.open(pp)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    # Cap the longest side at 2048 px — keeps fine print legible
                    # while keeping payload size reasonable.
                    img.thumbnail((2048, 2048))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    description = describe_image(
                        buf.getvalue(), self._vision_model(), prompt
                    )
                    results.append({"name": pp.name,
                                    "description": description})
                except Exception as e:  # noqa: BLE001 - one bad image must not break the tool
                    logging.error(f"view_image failed for {p}: {e}")
                    errors.append(f"Could not view {pp.name}: {e}")
            if not results:
                return json.dumps({
                    "status": "error",
                    "message": "No images could be viewed.",
                    "errors": errors,
                })
            return json.dumps({
                "status": "success",
                "n_images": len(results),
                "images": results,
                "errors": errors or None,
            })

        self._register_tool(
            func=view_image,
            name="view_image",
            description=(
                "Open one or more image files and have the vision model "
                "describe them — including faithfully transcribing any "
                "readable text or tables (printed or handwritten). Use this "
                "for a photo of a notebook page, a diagram, a screenshot, a "
                "figure, or any image that needs to be interpreted as "
                "content. NOT for scientific images that need feature "
                "extraction or quantification — route those to analysis. "
                "Accepts .png, .jpg/.jpeg, .tif/.tiff, .bmp, .gif, .webp."
            ),
            parameters={
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Absolute path(s) to the image file(s) to view."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "Optional question or instruction (e.g. "
                        "'transcribe the table' or 'what does the diagram "
                        "show?'). Default is to describe + transcribe."
                    ),
                },
            },
            required=["paths"],
        )

        # ----- view_document -------------------------------------------------
        # Open & inspect a document's text content directly in the meta's
        # chat — the symmetric counterpart of view_image, for routing /
        # summarization decisions without having to delegate to a specialist
        # just to extract text. Scanned PDFs are OCR'd automatically via the
        # parsers vision-OCR fallback (the orchestrator's model is the OCR
        # model). NOT for registering a document into a planning KB — that's
        # what delegate_to_planning's `knowledge_paths` is for.

        _DOCUMENT_EXTS = {".pdf", ".docx", ".md", ".txt",
                          ".json", ".yaml", ".yml",
                          ".csv", ".xlsx", ".xls"}
        _VIEW_DOC_MAX_CHARS = 200_000  # ~50k tokens; long docs are truncated
        # Figures ride along automatically up to this many across the whole
        # call. Chosen to cover the documents figures actually help with — a
        # note or white paper carrying a few diagrams — while a figure-heavy
        # atlas reports its count instead of spending the context on it.
        _VIEW_DOC_MAX_FIGURES = 8
        # The most images ever put in one reply, even when explicitly asked
        # for. An unbounded payload under the agent's control is how a
        # context window gets spent in a single call, so 'on' raises the
        # ceiling rather than removing it — and says when it had to stop.
        _VIEW_DOC_HARD_FIGURE_CAP = 24

        def view_document(paths, figures: str = "auto") -> str:
            """Read one or more documents and return their text content."""
            if isinstance(paths, str):
                paths = [paths]
            if not paths:
                return json.dumps({"status": "error",
                                   "message": "No document path provided."})
            print(f"  📄 Tool: Reading {len(paths)} document(s)...")

            from scilink.parsers import extract_document, extract_text
            from scilink.parsers.docx_document import count_docx_figures

            # Whether to carry embedded figures back as images. Counted
            # exactly rather than estimated from document length — for DOCX
            # the images are enumerable without decoding them — so the
            # decision is made on the real cost, before paying it. A short
            # document with a handful of figures is the case worth spending
            # on; an atlas of forty is not, and gets a count instead so the
            # agent still knows they exist and can ask for a subset.
            want = str(figures or "auto").strip().lower()
            n_fig_total = sum(count_docx_figures(p) for p in paths
                              if str(p).lower().endswith(".docx"))
            if want == "on":
                # Asked for explicitly: attach as many as the hard cap allows
                # rather than silently stopping at the auto threshold, which
                # made 'on' quietly mean 'on, up to 8' (caught live).
                attach, fig_ceiling = True, _VIEW_DOC_HARD_FIGURE_CAP
            elif want == "off":
                attach, fig_ceiling = False, 0
            else:
                attach = 0 < n_fig_total <= _VIEW_DOC_MAX_FIGURES
                fig_ceiling = _VIEW_DOC_MAX_FIGURES

            docs, errors, images_b64 = [], [], []
            undeliverable = []
            for p in paths:
                pp = Path(p)
                if not pp.is_file():
                    errors.append(f"Not a file: {p}")
                    continue
                if pp.suffix.lower() not in _DOCUMENT_EXTS:
                    errors.append(
                        f"Not a supported document: {p} "
                        f"(handles {', '.join(sorted(_DOCUMENT_EXTS))})"
                    )
                    continue
                try:
                    info = extract_text(pp, ocr_model=self.orch.model)
                    if attach and pp.suffix.lower() == ".docx":
                        parsed = extract_document(pp)
                        for f in parsed.figures:
                            if len(images_b64) >= fig_ceiling:
                                break
                            if not f.deliverable:
                                # Keeps its number; say which one and why,
                                # so the marker is not an unexplained gap.
                                undeliverable.append(
                                    f"[Figure {f.index + 1}] {f.note}")
                                continue
                            images_b64.append(f.to_base64())
                    text = info.get("text", "")
                    truncated = len(text) > _VIEW_DOC_MAX_CHARS
                    if truncated:
                        text = text[:_VIEW_DOC_MAX_CHARS]
                    doc_info = {
                        "name": pp.name,
                        "text": text,
                        "n_chars": len(text),
                        "truncated": truncated,
                    }
                    # Format-specific metadata flows through transparently
                    # (n_pages for PDFs, n_paragraphs for DOCX, plus the
                    # OCR page count when the vision fallback fired).
                    for k in ("n_pages", "n_paragraphs", "n_ocr_pages",
                              "n_tables", "n_figures"):
                        if k in info:
                            doc_info[k] = info[k]
                    docs.append(doc_info)
                except ValueError as e:
                    # extract_text raises ValueError for genuinely unsupported
                    # extensions — surface it but don't crash the tool.
                    errors.append(str(e))
                except Exception as e:  # noqa: BLE001 - one bad doc must not break the tool
                    logging.error(f"view_document failed for {p}: {e}")
                    errors.append(f"Could not read {pp.name}: {e}")
            if not docs:
                return json.dumps({"status": "error",
                                   "message": "No documents could be read.",
                                   "errors": errors})
            n_ocr = sum(d.get("n_ocr_pages", 0) for d in docs)
            payload_extra = {}
            if images_b64:
                # tool_media lifts this into real image content on providers
                # that render images inside a tool result, and the [Figure N]
                # markers in the text say which is which.
                payload_extra["images_base64"] = images_b64
                if len(images_b64) < n_fig_total:
                    # Partial delivery must be stated, not inferred. Left
                    # unsaid, the reply looks complete while [Figure N] for
                    # the rest points at nothing.
                    payload_extra["figures_attached"] = (
                        f"{len(images_b64)} of {n_fig_total} figures are "
                        f"attached, in document order — [Figure 1] through "
                        f"[Figure {len(images_b64)}]. The rest were not "
                        f"delivered; their markers remain in the text.")
            elif n_fig_total:
                payload_extra["figures_not_attached"] = (
                    f"{n_fig_total} embedded figure(s) were found but not "
                    f"attached"
                    + (" (figures='off')" if want == "off" else
                       f" (over the {_VIEW_DOC_MAX_FIGURES}-figure auto "
                       f"limit)")
                    + ". The text marks their positions as [Figure N]; "
                    "re-read with figures='on' to see them.")
            if undeliverable:
                payload_extra["figures_undisplayable"] = undeliverable
            return json.dumps({
                "status": "success",
                "n_documents": len(docs),
                "n_ocr_pages": n_ocr,
                **payload_extra,
                "ocr_note": (
                    f"{n_ocr} scanned page(s) had no text layer and were "
                    "transcribed by vision-OCR — verify any figures/numerics."
                ) if n_ocr else None,
                "documents": docs,
                "errors": errors or None,
            })

        self._register_tool(
            func=view_document,
            name="view_document",
            description=(
                "Open one or more documents and return their text content "
                "in this conversation. Supports text-like files (.pdf, "
                ".docx, .md, .txt, .json, .yaml/.yml) and tabular files "
                "(.csv, .xlsx/.xls). Scanned / image-only PDFs are "
                "automatically OCR'd via the vision model (the result "
                "reports n_ocr_pages + a note to verify any "
                "figures/numerics). Tabular files are previewed via the "
                "adaptive parser — small files return the full table as "
                "Markdown, large files a statistical summary; a sibling "
                "JSON metadata file (e.g. data.json next to data.xlsx) "
                "auto-enriches the preview. A Word document's tables are "
                "read in place, and its embedded figures come back as "
                "images you can see, marked [Figure N] where they sit in "
                "the text. Use this to inspect / "
                "summarize a file's contents right here — for a routing "
                "decision, to extract context to thread into a "
                "delegate_to_* call, or to answer a quick question about "
                "a single file. NOT for ingesting into a planning "
                "KnowledgeBase — for that, pass the path as "
                "`knowledge_paths` in delegate_to_planning."
            ),
            parameters={
                "figures": {
                    "type": "string",
                    "description": (
                        "Whether to attach a Word document's embedded "
                        "figures as viewable images: 'auto' (default) "
                        "attaches them when there are few enough to be "
                        "worth the context, 'on' raises the limit for a "
                        "document whose figures you specifically need to "
                        "see, 'off' keeps the reply text-only. Even 'on' "
                        "stops at a ceiling; when not everything fits, the "
                        "result says how many of how many arrived. Either "
                        "way the text marks where each figure sits."
                    ),
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Absolute path(s) to the document file(s) to read."
                    ),
                },
            },
            required=["paths"],
        )
