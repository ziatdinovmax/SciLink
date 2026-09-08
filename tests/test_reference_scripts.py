"""Attach-a-script: a user's script rides into the analysis agents as
reference material to ADAPT (script-bank semantics for a bare file)."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scilink.agents.exp_agents._reference_scripts import (
    load_reference_scripts, reference_script_block)


def test_load_skips_bad_inputs_and_truncates(tmp_path):
    good = tmp_path / "fit.py"; good.write_text("import numpy as np\nprint('hi')\n")
    binary = tmp_path / "blob.py"; binary.write_bytes(b"\x00\x01\x02" * 100)
    image = tmp_path / "img.png"; image.write_bytes(b"\x89PNG")
    big = tmp_path / "big.py"; big.write_text("x = 1\n" * 20000)
    out = load_reference_scripts([str(good), str(binary), str(image),
                                  str(tmp_path / "missing.py"), str(big)], max_bytes=1000)
    assert [s["label"] for s in out] == ["fit.py", "big.py"]
    assert out[0]["text"].startswith("import numpy") and out[0]["truncated"] is False
    assert out[1]["truncated"] is True and len(out[1]["text"]) <= 1000
    assert load_reference_scripts(None) == [] and load_reference_scripts([]) == []


def test_block_carries_the_adapt_semantics_and_the_code(tmp_path):
    p = tmp_path / "user_edge_fit.py"; p.write_text("def power_law(E, A, r):\n    return A * E**-r\n")
    state = {"reference_scripts": load_reference_scripts([str(p)])}
    block = reference_script_block(state)
    assert block.startswith("\n## User-Provided Reference Script\n")
    for phrase in ("ADAPT", "script bank", "never" if False else "Do not run it verbatim",
                   "what was kept and what was changed", "### user_edge_fit.py", "def power_law"):
        assert phrase in block, phrase
    assert reference_script_block({}) == "" and reference_script_block({"reference_scripts": []}) == ""
    assert reference_script_block(state, heading="---").startswith("\n--- User-Provided")


def test_all_three_agents_accept_reference_scripts():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    from scilink.agents.exp_agents.image_analysis_agent import ImageAnalysisAgent
    from scilink.agents.exp_agents.hyperspectral_analysis_agent import HyperspectralAnalysisAgent
    for cls in (CurveFittingAgent, ImageAnalysisAgent, HyperspectralAnalysisAgent):
        assert "reference_scripts" in inspect.signature(cls.analyze).parameters, cls.__name__


def test_every_guidance_site_also_shows_the_script():
    """Wherever a controller shows the user's `analysis_hints`, the reference
    script block must follow — planning, codegen, refinement, correction."""
    import re
    base = Path("scilink/agents/exp_agents/controllers")
    for f in ("curve_fitting_controllers.py", "image_analysis_controllers.py",
              "hyperspectral_controllers.py"):
        src = (base / f).read_text()
        sites = [m.start() for m in re.finditer(r'if state\.get\("analysis_hints"\)', src)]
        assert sites, f
        for pos in sites:
            window = src[pos:pos + 900]
            assert "_reference_script_block(state" in window, f"{f} @ {src[:pos].count(chr(10)) + 1}"


def test_run_analysis_schema_exposes_reference_scripts():
    src = Path("scilink/agents/exp_agents/analysis_orchestrator_tools.py").read_text()
    i = src.index('"reference_scripts": {')
    desc = src[i:i + 900]
    assert "ADAPT" in desc and "script-bank" in desc and "never" in desc


def test_scripts_upload_category(tmp_path):
    from scilink.server import files as files_mod
    r = files_mod.save_uploads(str(tmp_path), "scripts", [("user_edge_fit.py", b"print(1)")])
    assert r["paths"] == [str(tmp_path / "scripts" / "user_edge_fit.py")]
    with pytest.raises(files_mod.UploadError):
        files_mod.save_uploads(str(tmp_path), "scripts", [("data.npy", b"")])
    with pytest.raises(files_mod.UploadError):          # still rejected by the data category
        files_mod.save_uploads(str(tmp_path), "data", [("user_edge_fit.py", b"")])


def test_multiple_scripts_guidance_and_total_budget(tmp_path):
    a = tmp_path / "preprocess.py"; a.write_text("def smooth(y): return y\n")
    b = tmp_path / "fit.py"; b.write_text("def fit(y): return y\n")
    c = tmp_path / "third.py"; c.write_text("x = 1\n" * 50)
    state = {"reference_scripts": load_reference_scripts([str(a), str(b)])}
    block = reference_script_block(state)
    assert block.startswith("\n## User-Provided Reference Scripts\n")
    assert "compose them in pipeline order" in block and "do not blend competing methods" in block
    assert "### preprocess.py" in block and "### fit.py" in block
    # a single script gets no multi-script paragraph
    assert "compose them" not in reference_script_block({"reference_scripts": load_reference_scripts([str(a)])})
    # total budget: later scripts are listed by name only, never dropped silently
    out = load_reference_scripts([str(a), str(b), str(c)], max_total_bytes=40)
    assert [s["label"] for s in out] == ["preprocess.py", "fit.py", "third.py"]
    assert out[0]["truncated"] is False and out[1]["truncated"] is True
    assert out[2].get("omitted") is True and out[2]["text"] == ""
    block = reference_script_block({"reference_scripts": out})
    assert "third.py — attached, but omitted here" in block and "read_file" in block
