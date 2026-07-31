"""Offline tests for the deterministic pathway-graph emitter.

Pure arithmetic + string emission: no LLM, no renderer.

  conda run -n scilink python tests/test_pathway_graph.py
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

from scilink.utils.pathway_graph import (
    PathwaySpecError, absorption_distributions, emit_mermaid,
    stimulus_table, validate_spec, UNRESOLVED)

results = {}


def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


SPEC = {
    "states": [
        {"id": "A", "label": "Amorphous gel", "short": "A"},
        {"id": "B1", "label": "Nuclei-rich transient", "short": "B1"},
        {"id": "B2", "label": "Layered precursor", "short": "B2"},
        {"id": "Ea", "label": "Target alpha", "short": "alpha",
         "kind": "endpoint"},
        {"id": "Eb", "label": "Metastable beta", "short": "beta",
         "kind": "endpoint"},
        {"id": "Et", "label": "Kinetic trap", "short": "trap",
         "kind": "endpoint"},
    ],
    "transitions": [
        {"from": "A", "to": "B1", "p": 0.6, "sigma": 0.05,
         "stimulus": "fast ramp", "timing": "t=8-12 min",
         "authority": "high"},
        {"from": "A", "to": "B2", "p": 0.4, "sigma": 0.05,
         "stimulus": "slow ramp", "timing": "t=8-12 min",
         "authority": "high"},
        {"from": "B1", "to": "Ea", "p": 0.9, "sigma": 0.03, "gate": True},
        {"from": "B1", "to": "Et", "p": 0.1, "sigma": 0.03},
        {"from": "B2", "to": "Eb", "p": 0.8, "sigma": 0.04, "gate": True},
        {"from": "B2", "to": "Et", "p": 0.2, "sigma": 0.04},
    ],
}

print("1) normalization")
check("valid spec passes", validate_spec(SPEC) == [])
bad = {**SPEC, "transitions": [dict(SPEC["transitions"][0], p=0.55),
                               dict(SPEC["transitions"][1], p=0.30)]
       + SPEC["transitions"][2:]}
try:
    validate_spec(bad)
    check("non-normalized rejected", False)
except PathwaySpecError as e:
    check("non-normalized rejected", "sum to 0.850" in str(e), f"({e})")

print("2) absorption distributions are derived, not asserted")
d = absorption_distributions(SPEC)
# A: alpha .6*.9=.54, beta .4*.8=.32, trap .6*.1+.4*.2=.14
check("A alpha 0.54", abs(d["A"]["Ea"] - 0.54) < 1e-9, f"{d['A']['Ea']:.3f}")
check("A beta 0.32", abs(d["A"]["Eb"] - 0.32) < 1e-9)
check("A trap 0.14", abs(d["A"]["Et"] - 0.14) < 1e-9)
check("A sums to 1", abs(sum(d["A"].values()) - 1.0) < 1e-9)
check("B1 narrows to alpha .90", abs(d["B1"]["Ea"] - 0.9) < 1e-9)

print("3) unresolvable mixture absorbs mass honestly")
mix = {
    "states": SPEC["states"] + [{"id": "M", "label": "Mixture",
                                 "kind": "mixture", "entropy_bits": 1.35}],
    "transitions": [
        {"from": "A", "to": "B1", "p": 0.5},
        {"from": "A", "to": "M", "p": 0.5},
        {"from": "M", "to": "B1", "unresolved": True},
        {"from": "B1", "to": "Ea", "p": 0.9},
        {"from": "B1", "to": "Et", "p": 0.1},
        {"from": "B2", "to": "Eb", "p": 1.0},
    ],
}
dm = absorption_distributions(mix)
check("half the mass is unresolved",
      abs(dm["A"][UNRESOLVED] - 0.5) < 1e-9, f"{dm['A'][UNRESOLVED]:.2f}")
check("resolved part still exact", abs(dm["A"]["Ea"] - 0.45) < 1e-9)
check("mixture normalization not required", validate_spec(mix) == [])

print("4) emission")
phys = emit_mermaid(SPEC, style="physics")
full = emit_mermaid(SPEC, style="full")
check("physics keeps stimulus off edges",
      "fast ramp" not in phys and "[s1]" in phys)
check("full inlines stimulus", "fast ramp" in full)
check("computed distribution in node label", "alpha .54" in phys)
check("gate styled distinctly", "==>" in phys)
check("deterministic", emit_mermaid(SPEC) == phys)
check("endpoints classed", ":::endpoint" in phys)

mm = emit_mermaid(mix)
check("mixture node marked",
      "unresolvable" in mm and ":::mixture" in mm)

print("5) stimulus table")
tab = stimulus_table(SPEC)
check("table keys match edge keys", "| s1 |" in tab and "fast ramp" in tab)
check("timing/authority carried", "t=8-12 min" in tab and "high" in tab)

print("6) errors")
try:
    absorption_distributions({"states": [{"id": "A"}], "transitions": []})
    check("missing endpoints rejected", False)
except PathwaySpecError:
    check("missing endpoints rejected", True)
try:
    validate_spec({**SPEC, "transitions": SPEC["transitions"] +
                   [{"from": "A", "to": "ZZ", "p": 1.0}]})
    check("unknown state rejected", False)
except PathwaySpecError:
    check("unknown state rejected", True)

print("=" * 50)
n = sum(results.values())
print(f"PATHWAY GRAPH: {n}/{len(results)} checks passed")
if n != len(results):
    raise SystemExit(1)
