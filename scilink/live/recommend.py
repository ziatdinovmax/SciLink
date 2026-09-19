"""What to measure next — a contract, not an algorithm.

Bayesian optimization is one way to produce the next measurement. A rule table
is another; so is a language model that reads the run so far and writes the
next acquisition parameters as JSON, or a revised protocol. The loop does not
care which — it needs three things from any of them, and those are what this
module defines.

1. **An instrument schema** (:class:`InstrumentSchema`). The acquisition
   parameters the controller accepts: name, kind, bounds or choices, units.
   It is what makes a recommendation *mappable* onto a controller interface —
   and it is supplied by the caller, who owns the instrument and its safe
   limits.
2. **Validation by the loop, not by the recommender.** Every recommendation is
   checked against the schema before it is returned: an unknown name or an
   out-of-bounds value is *refused*, never silently clamped (a clamped value is
   a different experiment from the one recommended). SciLink recommends; it
   never actuates, and a protocol / code recommendation is written out as an
   artifact, never executed.
3. **A clock.** ``clock = "fast"`` recommenders (rule table, surrogate
   optimizer) are deterministic, model-free and run inside ``step()``.
   ``clock = "slow"`` recommenders (an LLM) take seconds to tens of seconds and
   run off the fast path: ``step()`` returns the most recent finished
   recommendation, stamped with the step it was based on, and never waits.

A recommendation is a plain dict::

    {"kind": "params" | "protocol", "params": {...} | None,
     "protocol": str | None, "rationale": str, "source": str,
     "based_on_step": int, "valid": bool, "problems": [...],
     "requires_approval": bool}
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

KINDS = ("float", "int", "choice", "bool")


# ──────────────────────────────────────────────────────────────
# The instrument schema
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParameterSpec:
    """One acquisition parameter of the instrument controller."""
    name: str
    kind: str = "float"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Sequence[Any]] = None
    units: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind in ("float", "int"):
            if self.low is None or self.high is None or not self.low < self.high:
                raise ValueError(f"{self.name}: a numeric parameter needs low < high "
                                 "(the caller owns the safe limits; there is no default)")
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"{self.name}: a choice parameter needs `choices`")

    def check(self, value: Any) -> Optional[str]:
        """``None`` when ``value`` is acceptable, else what is wrong with it."""
        if self.kind == "bool":
            return None if isinstance(value, bool) else f"{self.name}: expected true/false, got {value!r}"
        if self.kind == "choice":
            return None if value in self.choices else (
                f"{self.name}: {value!r} is not one of {list(self.choices)}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{self.name}: expected a number, got {value!r}"
        if not math.isfinite(value):
            return f"{self.name}: not a finite number"
        if self.kind == "int" and float(value) != int(value):
            return f"{self.name}: expected an integer, got {value!r}"
        if value < self.low or value > self.high:
            return (f"{self.name}: {value} is outside [{self.low}, {self.high}]"
                    + (f" {self.units}" if self.units else ""))
        return None


@dataclass(frozen=True)
class InstrumentSchema:
    """The acquisition parameters a controller accepts."""
    parameters: Sequence[ParameterSpec]

    def __post_init__(self) -> None:
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter names in the instrument schema")
        if not names:
            raise ValueError("an instrument schema needs at least one parameter")

    @classmethod
    def from_dict(cls, spec: Dict[str, Dict[str, Any]]) -> "InstrumentSchema":
        """``{"dwell_ms": {"low": 1, "high": 500, "units": "ms"}, ...}``"""
        return cls([ParameterSpec(name=k, **v) for k, v in spec.items()])

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for p in self.parameters:
            d = {"kind": p.kind}
            for k in ("low", "high", "units", "description"):
                if getattr(p, k) is not None:
                    d[k] = getattr(p, k)
            if p.choices is not None:
                d["choices"] = list(p.choices)
            out[p.name] = d
        return out

    def get(self, name: str) -> Optional[ParameterSpec]:
        return next((p for p in self.parameters if p.name == name), None)

    @property
    def numeric(self) -> List[ParameterSpec]:
        return [p for p in self.parameters if p.kind in ("float", "int")]

    def validate(self, params: Any) -> List[str]:
        """Problems with a proposed parameter set; empty when it may be returned.

        A PARTIAL set is fine (change only what needs changing); an unknown
        name is not — it cannot be mapped onto the controller.
        """
        if not isinstance(params, dict) or not params:
            return ["no parameters were proposed"]
        problems = []
        for name, value in params.items():
            spec = self.get(str(name))
            if spec is None:
                problems.append(f"{name}: not a parameter of this instrument "
                                f"(known: {[p.name for p in self.parameters]})")
                continue
            problem = spec.check(value)
            if problem:
                problems.append(problem)
        return problems

    def describe(self) -> str:
        lines = []
        for p in self.parameters:
            rng = (f"one of {list(p.choices)}" if p.kind == "choice" else
                   "true/false" if p.kind == "bool" else
                   f"{p.kind} in [{p.low}, {p.high}]" + (f" {p.units}" if p.units else ""))
            lines.append(f"- {p.name}: {rng}" + (f" — {p.description}" if p.description else ""))
        return "\n".join(lines)


def finalize(raw: Any, schema: Optional[InstrumentSchema], *, source: str,
             based_on_step: int, closed_loop: bool = False) -> Dict[str, Any]:
    """Turn whatever a recommender returned into the record the loop hands out.

    This is the loop's side of the contract: the recommender is not trusted to
    have respected the schema. A recommendation with problems is returned
    ``valid: False`` with its ``params`` withheld, so a caller that maps
    ``params`` onto a controller cannot act on it by accident.
    """
    raw = raw if isinstance(raw, dict) else {"params": None}
    kind = "protocol" if raw.get("protocol") and not raw.get("params") else "params"
    problems: List[str] = list(raw.get("problems") or [])
    params = raw.get("params")
    if kind == "params":
        problems += (schema.validate(params) if schema is not None
                     else ([] if isinstance(params, dict) and params
                           else ["no parameters were proposed"]))
    valid = not problems
    return {
        "kind": kind,
        "params": params if (valid and kind == "params") else None,
        "protocol": raw.get("protocol") if kind == "protocol" else None,
        "rationale": str(raw.get("rationale") or "")[:600],
        "source": source, "based_on_step": based_on_step,
        "valid": valid, "problems": problems,
        # A protocol is never auto-applied; parameters only in a closed loop.
        "requires_approval": (kind == "protocol") or not closed_loop or not valid,
        **({"rejected_params": params} if (not valid and params) else {}),
    }


# ──────────────────────────────────────────────────────────────
# Recommenders
# ──────────────────────────────────────────────────────────────

class Recommender:
    """Base contract. ``observe`` is called for every frame that produced
    features and came with parameters — flagged frames included, with their
    flags: a frame below the fit gate or reading as drift is often the very
    thing a recommendation is for. ``suggest`` returns ``{"params": ...}`` or
    ``{"protocol": ...}`` with a ``rationale``. ``clock`` decides where the
    loop runs ``suggest``."""

    clock = "fast"
    name = "recommender"

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def observe(self, params: Dict[str, Any], features: Dict[str, float],
                step: Optional[int] = None, flags: Optional[List[str]] = None) -> None:
        self.history.append({"step": step, "params": dict(params or {}),
                             "features": dict(features or {}),
                             "flags": list(flags or [])})

    def suggest(self) -> Dict[str, Any]:        # pragma: no cover - interface
        raise NotImplementedError


_OPS: Dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
}


class RuleTableRecommender(Recommender):
    """Deterministic symptom → parameter corrections. Zero LLM, zero model.

    ``rules``: evaluated in order against the latest clean frame; the first
    that fires wins. Each is ``{"when": {"feature": ..., "op": "<", "value": ...},
    "then": {"param": ..., "scale": 1.5 | "add": 10 | "set": 40}, "why": ...}``.
    The result is clipped to the schema's bounds — for a rule table a bound is
    a stop, not an error ("dwell is already at its maximum").
    """

    name = "rule_table"

    def __init__(self, schema: InstrumentSchema, rules: List[Dict[str, Any]]) -> None:
        super().__init__()
        self.schema, self.rules = schema, list(rules)
        for r in self.rules:
            if (r.get("when") or {}).get("op") not in _OPS:
                raise ValueError(f"rule {r}: op must be one of {sorted(_OPS)}")
            if schema.get((r.get("then") or {}).get("param")) is None:
                raise ValueError(f"rule {r}: `then.param` is not in the instrument schema")

    def suggest(self) -> Dict[str, Any]:
        if not self.history:
            return {"params": None, "problems": ["no clean frame observed yet"]}
        last = self.history[-1]
        for r in self.rules:
            w, then = r["when"], r["then"]
            value = last["features"].get(w["feature"])
            if value is None or not _OPS[w["op"]](value, w["value"]):
                continue
            spec = self.schema.get(then["param"])
            current = last["params"].get(spec.name)
            if "set" in then:
                new = then["set"]
            elif current is None:
                return {"params": None,
                        "problems": [f"{spec.name}: current value unknown — pass it in step(params=...)"]}
            else:
                new = current * then["scale"] if "scale" in then else current + then["add"]
            if spec.kind in ("float", "int"):
                new = min(max(new, spec.low), spec.high)
                new = int(round(new)) if spec.kind == "int" else float(new)
            if new == current:
                return {"params": None, "problems": [
                    f"{spec.name} is already at its limit ({current}) for: {r.get('why', w)}"]}
            return {"params": {spec.name: new},
                    "rationale": r.get("why") or f"{w['feature']} {w['op']} {w['value']}"}
        return {"params": dict(last["params"]) or None,
                "rationale": "no rule fired — keep the current parameters",
                **({} if last["params"] else {"problems": ["no rule fired and no current parameters known"]})}


class GPRecommender(Recommender):
    """A surrogate optimizer with its strategy LOCKED at construction — the
    way a series locks a recipe. Reuses ``bo_tools.SingleObjectiveOptimizer``
    (BoTorch, no LLM); the first ``n_init`` suggestions are a deterministic
    space-filling design. Numeric parameters only.
    """

    name = "gp"

    def __init__(self, schema: InstrumentSchema, objective_key: str, *,
                 direction: str = "maximize", n_init: int = 4,
                 acquisition: str = "log_ei",
                 model_config: Optional[Dict[str, str]] = None, seed: int = 0) -> None:
        super().__init__()
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if not schema.numeric or len(schema.numeric) != len(schema.parameters):
            raise ValueError("GPRecommender optimizes numeric parameters only")
        self.schema, self.objective_key = schema, objective_key
        self.direction, self.n_init, self.acquisition = direction, max(2, n_init), acquisition
        self.model_config = dict(model_config or {"surrogate": "single_task",
                                                  "kernel": "matern_2.5",
                                                  "noise": "min_noise_low"})
        self._design = self._space_filling(self.n_init, seed)

    def _space_filling(self, n: int, seed: int) -> List[Dict[str, float]]:
        import numpy as np
        rng = np.random.default_rng(seed)
        specs = self.schema.numeric
        # Latin hypercube: one stratum per point in every dimension.
        cols = [(rng.permutation(n) + rng.uniform(0.25, 0.75, n)) / n for _ in specs]
        return [{s.name: self._cast(s, s.low + c[i] * (s.high - s.low))
                 for s, c in zip(specs, cols)} for i in range(n)]

    @staticmethod
    def _cast(spec: ParameterSpec, value: float):
        value = min(max(float(value), spec.low), spec.high)
        return int(round(value)) if spec.kind == "int" else value

    def _xy(self):
        import numpy as np
        names = [s.name for s in self.schema.numeric]
        rows = [(h["params"], h["features"].get(self.objective_key)) for h in self.history]
        rows = [(p, y) for p, y in rows if y is not None and all(n in p for n in names)]
        if not rows:
            return names, np.empty((0, len(names))), np.empty((0,))
        X = np.array([[float(p[n]) for n in names] for p, _ in rows], dtype=float)
        y = np.array([float(v) for _, v in rows], dtype=float)
        return names, X, (y if self.direction == "maximize" else -y)

    def suggest(self) -> Dict[str, Any]:
        names, X, y = self._xy()
        if len(y) < self.n_init:
            return {"params": self._design[len(y)],
                    "rationale": f"space-filling design point {len(y) + 1}/{self.n_init}"}
        from ..agents.planning_agents.bo_tools import SingleObjectiveOptimizer
        opt = SingleObjectiveOptimizer()
        opt.fit(X, y, [(s.low, s.high) for s in self.schema.numeric],
                self.model_config, feature_names=names)
        x = opt.recommend(1, strategy=self.acquisition)[0]
        best = float(y.max() if self.direction == "maximize" else -y.max())
        return {"params": {s.name: self._cast(s, v) for s, v in zip(self.schema.numeric, x)},
                "rationale": (f"{self.acquisition} on a {self.model_config['surrogate']} GP over "
                              f"{len(y)} observations; best {self.objective_key} so far {best:.4g}")}


LLM_RECOMMENDER_PROMPT = """You are choosing the next measurement for a running experiment.

## Goal
{objective}

## The experiment
{context}

## Acquisition parameters you may set (the instrument controller's interface)
{schema}

## The run so far (most recent last): parameters used -> quantities measured
{history}
(A flag after a frame is the analysis loop's own quality signal for it, e.g. the
fit was below its acceptance gate or the data looked unlike the reference.)

Reason from the numbers above. Respond with ONE JSON object and nothing else:
{output_contract}"""

_PARAMS_CONTRACT = ('{"params": {"<parameter name>": <value>, ...}, "rationale": "<one or two sentences>"}\n'
                    "Use only the parameter names listed, with values inside their ranges. "
                    "Set only what should change.")
_PROTOCOL_CONTRACT = ('{"protocol": "<the revised acquisition protocol, as text or code>", '
                      '"rationale": "<one or two sentences>"}\n'
                      "The protocol is handed to a person for review; it is not executed automatically.")


class LLMRecommender(Recommender):
    """A language model reads the run so far and writes the next acquisition
    parameters as JSON against the instrument schema — or, with
    ``output="protocol"``, a revised protocol / code for a person to review.

    Slow clock: a call takes seconds to tens of seconds, so the loop runs it off
    the fast path every ``every`` clean frames and ``step()`` never waits.
    ``model`` is any object with ``generate_content(prompt) -> .text`` (the
    SciLink LLM wrappers).
    """

    clock = "slow"
    name = "llm"

    def __init__(self, model: Any, schema: InstrumentSchema, objective: str, *,
                 output: str = "params", every: int = 5, max_history: int = 25,
                 feature_keys: Optional[List[str]] = None,
                 context: Any = None,
                 generation_config: Any = None) -> None:
        super().__init__()
        #: What is being measured and what is being done to the sample — a dict
        #: (the loop's ``system_info``) or text. Left None, the loop fills it
        #: with its own ``system_info``: without it a model cannot tell a sample
        #: that is evolving on purpose from an artefact of its own settings.
        self.context = context
        if output not in ("params", "protocol"):
            raise ValueError("output must be 'params' or 'protocol'")
        self.model, self.schema, self.objective = model, schema, objective
        self.output, self.every = output, max(1, int(every))
        self.max_history, self.feature_keys = max_history, feature_keys
        self.generation_config = generation_config

    def _history_table(self) -> str:
        rows = self.history[-self.max_history:]
        keys = self.feature_keys or sorted({
            k for h in rows for k in h["features"]
            if not k.endswith(("_err", "_stderr", "_std"))})[:12]
        lines = []
        for h in rows:
            feats = {k: round(h["features"][k], 5) for k in keys if k in h["features"]}
            flagged = f"  [flags: {', '.join(h['flags'])}]" if h.get("flags") else ""
            lines.append(f"step {h['step']}: {json.dumps(h['params'])} -> "
                         f"{json.dumps(feats)}{flagged}")
        return "\n".join(lines) or "(no clean frames yet)"

    def _context_text(self) -> str:
        c = self.context
        if not c:
            return "(not described)"
        if isinstance(c, dict):
            return "\n".join(f"- {k}: {v}" for k, v in c.items())
        return str(c)

    def build_prompt(self) -> str:
        return LLM_RECOMMENDER_PROMPT.format(
            objective=self.objective, context=self._context_text(),
            schema=self.schema.describe(),
            history=self._history_table(),
            output_contract=_PARAMS_CONTRACT if self.output == "params" else _PROTOCOL_CONTRACT)

    def suggest(self) -> Dict[str, Any]:
        from ..skills._shared._graduation import parse_json_response
        kwargs = ({"generation_config": self.generation_config}
                  if self.generation_config is not None else {})
        raw = self.model.generate_content(self.build_prompt(), **kwargs)
        text = raw.text if hasattr(raw, "text") else str(raw)
        try:
            parsed = parse_json_response(text)
        except ValueError:
            return {"params": None, "problems": ["the model did not return JSON"],
                    "rationale": text[:200]}
        if not isinstance(parsed, dict):
            return {"params": None, "problems": ["the model did not return a JSON object"]}
        return parsed


# ──────────────────────────────────────────────────────────────
# Running a slow recommender off the fast path
# ──────────────────────────────────────────────────────────────

class SlowSlot:
    """At most one ``suggest()`` of a slow recommender in flight, on a daemon
    thread (a pure network call — unlike a re-anchor there is no plotting or
    sandbox, so a thread is enough). The loop polls; nothing here blocks."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self.based_on_step: Optional[int] = None
        self.logger = logger or logging.getLogger("MeasurementLoop")

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, recommender: Recommender, step: int) -> bool:
        if self.busy:
            return False
        self.based_on_step = step

        def run() -> None:
            try:
                # Off the fast path in the accounting too: this call must not
                # be charged to whichever frame happens to be running.
                from .. import tracing
                with tracing.off_path():
                    out = recommender.suggest()
            except Exception as e:  # noqa: BLE001 - reported as a problem
                out = {"params": None, "problems": [f"{type(e).__name__}: {e}"]}
            with self._lock:
                self._result = {"raw": out, "based_on_step": step}

        self._thread = threading.Thread(target=run, daemon=True, name="scilink-recommender")
        self._thread.start()
        return True

    def take(self) -> Optional[Dict[str, Any]]:
        """The finished result, once; ``None`` while running or when empty."""
        with self._lock:
            out, self._result = self._result, None
        return out

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
