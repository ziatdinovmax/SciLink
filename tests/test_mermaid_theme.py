"""Offline tests: the palette must MEAN something.

Colors are reconciled with what the graph structure says each node is,
so a figure cannot ship with decorative-but-wrong coloring.

  conda run -n scilink python tests/test_mermaid_theme.py
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-dummy")

from scilink.utils.mermaid_theme import (
    PALETTE, apply_theme, enforce_semantics, theme_block)

results = {}


def check(name, cond, detail=""):
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def cls_of(code, nid):
    import re
    # Quoted-label aware, like the module: a label may legitimately
    # contain (parens) or [brackets].
    m = re.search(
        rf'\b{nid}(?:\[|\{{|\()(?:"[^"]*"|[^"\]\}}\)]*)(?:\]|\}}|\)):::(\w+)',
        code)
    return m.group(1) if m else None


print("1) shape and role agree")
code = ('flowchart LR\n'
        '  A["Start"]:::outcome --> B{"Gate"}:::stage\n'
        '  B --> C["Done"]:::stage\n')
out = enforce_semantics(code)
check("diamond forced to decision", cls_of(out, "B") == "decision")
check("non-terminal outcome demoted to stage", cls_of(out, "A") == "stage")
check("terminal promoted to outcome", cls_of(out, "C") == "outcome")

print("2) a non-diamond cannot claim to be a gate")
out2 = enforce_semantics('flowchart LR\n  X["Check"]:::decision --> Y["End"]\n')
check("box demoted from decision", cls_of(out2, "X") == "stage")

print("3) a dead end stays a caution, not a happy outcome")
out3 = enforce_semantics(
    'flowchart LR\n  A["Run"] --> T["Kinetic trap"]:::caution\n'
    '  A --> I["No control here"]:::inactive\n')
check("caution terminal preserved", cls_of(out3, "T") == "caution")
check("inactive terminal preserved", cls_of(out3, "I") == "inactive")

print("4) accent means THE most important node")
out4 = enforce_semantics(
    'flowchart LR\n  A["a"]:::accent --> B["b"]:::accent\n'
    '  B --> C["c"]:::accent\n')
check("only the first accent survives",
      [cls_of(out4, n) for n in "AB"] == ["accent", "stage"])

print("5) untagged nodes are still classified")
out5 = enforce_semantics('flowchart LR\n  A["a"] --> B["b"]\n')
check("untagged mid-graph -> stage", cls_of(out5, "A") == "stage")
check("untagged terminal -> outcome", cls_of(out5, "B") == "outcome")

print("6) loops do not make every node terminal")
out6 = enforce_semantics(
    'flowchart LR\n  A["a"] --> B{"g"}\n  B -->|"no"| A\n'
    '  B -->|"yes"| C["done"]\n')
check("looped node is not terminal", cls_of(out6, "A") == "stage")
check("gate in loop is decision", cls_of(out6, "B") == "decision")
check("true terminal found", cls_of(out6, "C") == "outcome")

print("7) theme application")
themed = apply_theme(code)
check("palette appended", "classDef stage" in themed
      and PALETTE["stage"][0] in themed)
check("semantics enforced through apply_theme",
      cls_of(themed, "B") == "decision")
check("default class present", "classDef default" in theme_block())


print("8) science inside labels must not be mistaken for node shapes")
sci = ('flowchart LR\n'
       '  Mix["Mix KMnO4 (aq) with 30% H2O2"] --> Gate{"Mn(III)/Mn(IV) = 0.35?"}\n'
       '  Gate --> Done["Optimised [H2O2] recipe"]\n')
out8 = enforce_semantics(sci)
check("parenthesised label kept whole",
      '"Mix KMnO4 (aq) with 30% H2O2"' in out8)
check("bracketed label kept whole", '"Optimised [H2O2] recipe"' in out8)
check("no tag spliced inside a label", ":::stage with" not in out8
      and ":::decision/" not in out8)
check("classes still correct on science labels",
      [cls_of(out8, n) for n in ("Mix", "Gate", "Done")]
      == ["stage", "decision", "outcome"])
uni = enforce_semantics('flowchart LR\n  A["α→β in TiO₂ at 450 °C"] --> B["Δ ≥ 0.9"]\n')
check("unicode notation preserved", "α→β" in uni and "TiO₂" in uni)

print("9) unquoted labels do not hide the edge source")
unq = enforce_semantics('flowchart LR\n  A[Unquoted label] --> B["ok"]\n')
check("unquoted source is not treated as terminal",
      cls_of(unq, "A") == "stage", f"({cls_of(unq, 'A')})")

print("=" * 50)
n = sum(results.values())
print(f"MERMAID THEME: {n}/{len(results)} checks passed")
if n != len(results):
    raise SystemExit(1)
