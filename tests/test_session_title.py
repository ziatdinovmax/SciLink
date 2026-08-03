"""generate_session_title must work through whatever transport the agent has."""
import sys, types
from types import SimpleNamespace
sys.path.insert(0, "/Users/maxim.ziatdinov/Code/SciLink")
from scilink.ui.session_meta import generate_session_title

fails = []
def check(c, m):
    print(("  PASS  " if c else "  FAIL  ") + m)
    if not c: fails.append(m)

class FakeModel:
    """Stands in for OpenAIAsGenerativeModel / LiteLLMGenerativeModel — both
    expose generate_content(contents, generation_config=...) -> .text"""
    def __init__(self, text='"Operando synthesis pathway mapping."'):
        self.text, self.seen = text, {}
    def generate_content(self, contents, generation_config=None, safety_settings=None):
        self.seen = {"contents": contents, "cfg": generation_config}
        return SimpleNamespace(text=self.text)

print("=== titles through the agent's model object (both wrappers share this API) ===")
m = FakeModel()
t = generate_session_title(m, "Analyze the operando XRD data from run 12", "Sure, I'll start by")
print(f"     title -> {t!r}")
check(t == "Operando synthesis pathway mapping", "title generated and cleaned")
check(isinstance(m.seen["contents"], list), "prompt passed as the flat list both wrappers expect")
check(getattr(m.seen["cfg"], "max_output_tokens", None) == 24,
      "max_output_tokens=24 passed via generation_config (maps to max_tokens on both)")
check("Analyze the operando XRD" in m.seen["contents"][0], "first user message reaches the prompt")

print("\n=== THE REGRESSION: proxy transport must not be bypassed ===")
import scilink.ui.session_meta as sm
import inspect
src = inspect.getsource(sm.generate_session_title)
# strip the docstring: it explains the bug and legitimately names litellm/base_url
import ast as _ast
_fn = _ast.parse(src.lstrip()).body[0]
code = "\n".join(src.splitlines()[_fn.body[0].end_lineno:])
check("litellm" not in code, "no direct litellm call (was: bypassed the proxy wrapper)")
check("api_key" not in code and "base_url" not in code, "no credential re-derivation")
check("generate_content" in code, "goes through the model object's transport")

print("\n=== fails safe, never raises into the turn ===")
class Boom:
    def generate_content(self, *a, **k): raise RuntimeError("proxy down")
check(generate_session_title(Boom(), "x") is None, "model raising -> None, not an exception")
check(generate_session_title(None, "x") is None, "no model -> None")
check(generate_session_title(FakeModel(), "") is None, "no first message -> None")
check(generate_session_title(FakeModel(text=""), "x") is None, "empty completion -> None")
check(generate_session_title(FakeModel(text="   ...  "), "x") is None, "punctuation-only -> None")

print("\n=== agent_config no longer carries credentials ===")
sb = open("/Users/maxim.ziatdinov/Code/SciLink/scilink/ui/components/sidebar.py").read()
import re
blocks = re.findall(r'st\.session_state\.agent_config = \{[^}]*\}', sb, re.S)
# the third assignment predates this PR (carries fh_api_key); scope to the two it touched
blocks = [b for b in blocks if "fh_api_key" not in b]
check(len(blocks) >= 2, f"found {len(blocks)} agent_config assignments")
check(all("api_key" not in b and "base_url" not in b for b in blocks),
      "no agent_config assignment carries api_key/base_url")
app = open("/Users/maxim.ziatdinov/Code/SciLink/scilink/ui/app.py").read()
check("_cfg.get(\"api_key\")" not in app, "call site no longer reads credentials from agent_config")
check("getattr(st.session_state.agent, \"model\", None)" in app, "call site passes the agent's model")

print("\n" + ("ALL PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
