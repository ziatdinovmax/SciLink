"""Live test of the real read_file body, lifted verbatim from the source."""
import ast, json, re, sys, tempfile, textwrap
from pathlib import Path
import pandas as pd

SRC = Path("/Users/maxim.ziatdinov/Code/SciLink/scilink/agents/planning_agents/orchestrator_tools.py")
tree = ast.parse(SRC.read_text())

# pull the nested read_file def and the module-level constants it closes over
fn = consts = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "read_file":
        fn = node
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_FULL_READ_STEMS":
        consts = node
assert fn and consts, "could not locate read_file / constants in source"

lines = SRC.read_text().split("\n")
def grab(n, end):  return textwrap.dedent("\n".join(lines[n.lineno-1:end]))
ns = {"json": json, "re": re, "Path": Path, "pd": pd}
exec(grab(consts, consts.lineno + 1), ns)          # _FULL_READ_STEMS, _FULL_READ_MAX_CHARS
exec(grab(consts, consts.end_lineno + 1), ns)
exec(grab(fn, fn.end_lineno), ns)
read_file = ns["read_file"]
ns["self"] = type("S", (), {"_resolve_data_path": staticmethod(lambda p: (p, None))})()

LIT = ("/Users/maxim.ziatdinov/Code/SciLink/meta_session_20260729_141649/planning/"
       "delegations/37_answer_a_quantitative_estimate_question_/"
       "literature_search_hypothesis_context.md")

def call(**kw):
    r = json.loads(read_file(**kw)); return r, r.get("content", "")
def qcount(c): return len(re.findall(r'^# Question \d', c, re.M))

fails = []
def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond: fails.append(msg)

print("=== 1. THE REGRESSION: default read of the file that failed (792 lines, 4 Qs) ===")
r, c = call(file_path=LIT)
print(f"     shown={r['shown_lines']} truncated={r['truncated']} chars={len(c):,} Qsections={qcount(c)}/4")
check(qcount(c) == 4, "default read returns ALL FOUR question sections (was 1)")
check(r["truncated"] is False, "not reported as truncated")
for q, ln in [("Question 2", 198), ("Question 3", 389), ("Question 4", 579)]:
    check(q in c, f"{q} present in a single default read")

print("\n=== 2. offset reaches the middle (the capability that did not exist) ===")
r2, c2 = call(file_path=LIT, offset=198, max_lines=60)
print(f"     shown={r2['shown_lines']}  first line: {c2.lstrip().splitlines()[0][:66]}")
check(c2.lstrip().startswith("# Question 2"), "offset=198 lands exactly on Q2")
r2b, c2b = call(file_path=LIT, offset=579, max_lines=40)
check(c2b.lstrip().startswith("# Question 4"), "offset=579 lands exactly on Q4")

print("\n=== 3. index appears when the cap DOES bite (ordinary long file) ===")
tmp = Path(tempfile.mkdtemp()) / "big_report.md"
tmp.write_text("".join(f"# Section {i//200+1}: topic\n" if i % 200 == 0 else f"line {i}\n"
                       for i in range(800)))
r3, c3 = call(file_path=str(tmp))
check(r3["truncated"], "long non-lit file still truncates at max_lines")
check("TRUNCATED READ" in c3, "notice says it is a TRUNCATED READ, not a short file")
check("Sections:" in c3, "outline of section headings + line numbers is emitted")
check("offset=" in c3, "notice names offset as the escape hatch")
print("     " + [l for l in c3.splitlines() if "Sections:" in l][0][:104])

print("\n=== 4. oversized lit report still capped (no runaway context) ===")
big = Path(tempfile.mkdtemp()) / "literature_search_huge.md"
big.write_text("".join(f"# Question {i//2500+1}: topic\n" if i % 2500 == 0 else "padding line of text\n" for i in range(15000)))
r4, c4 = call(file_path=str(big))
check(r4["truncated"], f"lit report over the cap falls back to truncation ({big.stat().st_size:,} chars)")
check("Sections:" in c4, "over-cap lit report still emits the section outline")

print("\n=== 5. regressions: tail / search / small files unchanged ===")
r5, _ = call(file_path=LIT, tail=True, max_lines=30)
check(r5["mode"] == "tail" and r5["shown_lines"] == "763-792", "tail still reads the END")
r6, _ = call(file_path=LIT, search=r"^# Question")
check(r6["matches"] == 4 and r6["match_lines"] == [3, 198, 389, 579], "search still works")
r7, _ = call(file_path=str(tmp), max_lines=5000)
check(not r7["truncated"], "small/whole read still untruncated")
r8 = json.loads(read_file(file_path="/nope/missing.md"))
check(r8["status"] == "error", "missing file still errors cleanly")

print("\n" + ("ALL PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
