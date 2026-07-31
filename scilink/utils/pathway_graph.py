"""Deterministic pathway-graph emitter: transition table -> Mermaid.

A controllability map is a Markov-style object, so its figure should be
computed, not drawn freehand: given a fitted transition table (states,
branch probabilities with uncertainties, endpoints), this module

  * validates that outgoing branch probabilities normalize per state,
  * DERIVES each state's reachable-endpoint distribution by absorption
    (rather than trusting numbers written into node labels), and
  * emits the Mermaid source, plus a separate stimulus table.

Probability mass flowing through a state whose branches are unresolvable
(an unidentifiable mixture) is absorbed into an explicit ``unresolved``
outcome instead of being silently redistributed — the honest rendering
of "this fraction cannot be resolved with the current observation
model."

The LLM's job is the scientific structure (which classes, which
branches, which stimuli); arithmetic and layout are the code's.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

UNRESOLVED = "unresolved"


class PathwaySpecError(ValueError):
    """The transition table is not a usable graph."""


def _states_by_kind(spec: Dict[str, Any]) -> Tuple[List[dict], List[dict]]:
    states = spec.get("states") or []
    if not states:
        raise PathwaySpecError("spec has no states")
    endpoints = [s for s in states if s.get("kind") == "endpoint"]
    transient = [s for s in states if s.get("kind") != "endpoint"]
    if not endpoints:
        raise PathwaySpecError("spec has no endpoint states")
    return transient, endpoints


def validate_spec(spec: Dict[str, Any], tol: float = 1e-6) -> List[str]:
    """Structural + normalization check. Returns warnings; raises on
    errors that make the graph unusable."""
    transient, endpoints = _states_by_kind(spec)
    ids = {s["id"] for s in (spec.get("states") or [])}
    warnings: List[str] = []

    for t in spec.get("transitions") or []:
        for side in ("from", "to"):
            if t.get(side) not in ids:
                raise PathwaySpecError(
                    f"transition references unknown state: {t.get(side)}")

    for s in transient:
        out = [t for t in (spec.get("transitions") or [])
               if t["from"] == s["id"]]
        if not out:
            warnings.append(f"state {s['id']} has no outgoing transitions")
            continue
        if any(t.get("unresolved") for t in out):
            continue  # normalization is not claimable for a mixture
        total = sum(float(t.get("p", 0.0)) for t in out)
        if abs(total - 1.0) > max(tol, 1e-3):
            raise PathwaySpecError(
                f"outgoing branch probabilities from {s['id']} sum to "
                f"{total:.3f}, not 1.0")
    return warnings


def absorption_distributions(spec: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Reachable-endpoint distribution from every transient state.

    Endpoints (and an implicit ``unresolved`` outcome, fed by mixture
    states whose branches cannot be resolved) are the absorbing set;
    solves ``B = (I - Q)^-1 R``.
    """
    validate_spec(spec)
    transient, endpoints = _states_by_kind(spec)
    t_ids = [s["id"] for s in transient]
    e_ids = [s["id"] for s in endpoints]
    t_idx = {i: k for k, i in enumerate(t_ids)}
    e_idx = {i: k for k, i in enumerate(e_ids)}
    n, m = len(t_ids), len(e_ids)

    Q = np.zeros((n, n))
    R = np.zeros((n, m + 1))          # last column = unresolved
    for s in transient:
        i = t_idx[s["id"]]
        out = [t for t in (spec.get("transitions") or [])
               if t["from"] == s["id"]]
        if not out or any(t.get("unresolved") for t in out):
            R[i, m] = 1.0             # mass stops here, unresolvable
            continue
        for t in out:
            p = float(t.get("p", 0.0))
            j = t["to"]
            if j in e_idx:
                R[i, e_idx[j]] += p
            else:
                Q[i, t_idx[j]] += p

    try:
        B = np.linalg.solve(np.eye(n) - Q, R)
    except np.linalg.LinAlgError as exc:      # absorbing cycle
        raise PathwaySpecError(f"graph is not absorbing: {exc}") from exc

    labels = e_ids + [UNRESOLVED]
    return {sid: {labels[k]: float(B[t_idx[sid], k])
                  for k in range(m + 1) if B[t_idx[sid], k] > 5e-3}
            for sid in t_ids}


def _fmt_dist(dist: Dict[str, float], names: Dict[str, str]) -> str:
    parts = [f"{names.get(k, k)} {v:.2f}".replace("0.", ".")
             for k, v in sorted(dist.items(), key=lambda kv: -kv[1])]
    return " / ".join(parts)


def emit_mermaid(spec: Dict[str, Any], style: str = "physics",
                 direction: str = "LR") -> str:
    """Mermaid source for the pathway graph.

    ``style="physics"``: edges carry only the branch probability and its
    uncertainty (with a bracketed key into the stimulus table), keeping
    the figure about the probability structure.
    ``style="full"``: stimulus text is inlined on the edges.
    """
    dists = absorption_distributions(spec)
    states = spec.get("states") or []
    short = {s["id"]: (s.get("short") or s.get("label") or s["id"])
             for s in states}
    short[UNRESOLVED] = "unresolved"
    zero_auth = set(spec.get("zero_authority") or [])

    lines = [f"flowchart {direction}"]
    for s in states:
        sid, label = s["id"], (s.get("label") or s["id"])
        if s.get("kind") == "endpoint":
            lines.append(f'  {sid}["{label}"]:::endpoint')
            continue
        rows = [label]
        if s.get("kind") == "mixture":
            ent = s.get("entropy_bits")
            rows.append(f"H {ent} bits — branches unresolvable"
                        if ent is not None else "branches unresolvable")
        else:
            rows.append(_fmt_dist(dists.get(sid, {}), short))
        if sid in zero_auth:
            rows.append("no steering: stimuli leave distribution unchanged")
        body = "<br/>".join(rows)
        cls = (":::mixture" if s.get("kind") == "mixture"
               else ":::zeroauth" if sid in zero_auth else "")
        lines.append(f'  {sid}["{body}"]{cls}')

    for k, t in enumerate(spec.get("transitions") or [], start=1):
        arrow = "==>" if t.get("gate") else "-->"
        if t.get("unresolved"):
            lines.append(f'  {t["from"]} -.->|"unresolvable"| {t["to"]}')
            continue
        p = float(t.get("p", 0.0))
        sig = t.get("sigma")
        lab = f"p {p:.2f}".replace("0.", ".")
        if sig is not None:
            lab += f" ± {float(sig):.2f}".replace("0.", ".")
        if style == "full" and t.get("stimulus"):
            lab += f" — {t['stimulus']}"
        elif t.get("stimulus"):
            lab += f" [s{k}]"
        lines.append(f'  {t["from"]} {arrow}|"{lab}"| {t["to"]}')

    lines += [
        "  classDef endpoint stroke-width:3px,stroke:#1b5e20,fill:#e8f5e9;",
        "  classDef mixture stroke-dasharray: 5 5,stroke:#b26a00;",
        "  classDef zeroauth stroke:#546e7a,fill:#eceff1;",
    ]
    return "\n".join(lines)


def stimulus_table(spec: Dict[str, Any]) -> str:
    """Markdown table of the interventions keyed to the edge labels —
    what the physics-style figure deliberately keeps off the edges."""
    rows = ["| Key | Transition | Intervention | Timing | Authority |",
            "|---|---|---|---|---|"]
    for k, t in enumerate(spec.get("transitions") or [], start=1):
        if t.get("unresolved") or not t.get("stimulus"):
            continue
        rows.append(
            f"| s{k} | {t['from']} → {t['to']} | {t.get('stimulus','')} | "
            f"{t.get('timing','—')} | {t.get('authority','—')} |")
    return "\n".join(rows) if len(rows) > 2 else ""
