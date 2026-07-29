#!/usr/bin/env python3
"""Live test for #170 — series trend analysis with 2 control variables.

Drives the curve-fitting trend stage in isolation:
  1. takes a real fitted 10-spectrum series (a temperature x pH grid),
  2. injects the pH secondary control variable into series_metadata
     (the way _meta_for builds it — values keyed by filename),
  3. builds the trend prompt exactly as
     ConditionalTrendAnalysisController._generate_trend_script does, using
     the #170 branch's TREND_ANALYSIS_INSTRUCTIONS (read via `git show`),
  4. has the LLM author the trend script, runs it,
  5. checks the script consumed secondary_variables and produced a 2-D plot.

Ad-hoc live test — NOT committed. Needs ANTHROPIC_API_KEY.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scilink import auth
from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel

SESSION = "/Users/maxim.ziatdinov/Code/meta_session_20260517_172726"
SFR = (SESSION + "/analysis/results/"
       "analysis_uploads_CurveFit_20260517_172831_001/series_fit_results.json")
CONDITIONS = SESSION + "/uploads/conditions.json"
CONTROLLER = "scilink/agents/exp_agents/controllers/curve_fitting_controllers.py"
BRANCH = "series-trend-multivar"


def _trend_template() -> str:
    """TREND_ANALYSIS_INSTRUCTIONS from the #170 branch (not the checkout)."""
    src = subprocess.run(
        ["git", "show", f"{BRANCH}:{CONTROLLER}"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.search(r"TREND_ANALYSIS_INSTRUCTIONS = '''(.*?)'''", src, re.S)
    return m.group(1)


def _extract_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        s, e = t.find("{"), t.rfind("}")
        return json.loads(t[s:e + 1])


def main() -> int:
    model_name = "claude-opus-4-6"
    api_key = auth.get_api_key_for_model(model_name)
    if not api_key:
        print("ERROR: no API key in environment.")
        return 2
    model = LiteLLMGenerativeModel(model=model_name, api_key=api_key)

    sfr = json.loads(Path(SFR).read_text())
    conditions = json.loads(Path(CONDITIONS).read_text())

    # Inject pH as a secondary control variable — values keyed by filename,
    # exactly as _meta_for builds it.
    ph_values = {fname: c["pH"] for fname, c in conditions.items()}
    sfr["series_metadata"]["secondary_variables"] = [
        {"variable": "pH", "values": ph_values, "unit": ""}
    ]
    print(f"primary   : {sfr['series_metadata']['variable']} "
          f"{sfr['series_metadata']['values']}")
    print(f"secondary : pH {sorted(set(ph_values.values()))}")
    print()

    work = Path(tempfile.mkdtemp(prefix="trend_mv_"))
    (work / "series_fit_results.json").write_text(json.dumps(sfr))

    # Build the prompt exactly as _generate_trend_script does.
    param_summary = []
    for r in sfr["results"]:
        if r.get("success"):
            s = {"index": r["index"], "name": r["name"],
                 "model_type": r.get("model_type"),
                 "parameters": r.get("parameters", {}),
                 "fit_quality": r.get("fit_quality", {})}
            if r.get("flagged"):
                s["flagged"] = True
                s["flag_reason"] = r.get("flag_reason")
            param_summary.append(s)

    prompt = _trend_template().format(
        series_summary=json.dumps(param_summary, indent=2),
        series_metadata=json.dumps(sfr["series_metadata"], indent=2),
        flagged_info="No spectra were flagged.",
        objective="",
    )

    print("Generating trend script via LLM ...")
    resp = model.generate_content(contents=[prompt])
    raw = resp.text if hasattr(resp, "text") else str(resp)
    result = _extract_json(raw)
    script = result.get("script", "")

    print(f"analysis_approach : {result.get('analysis_approach')}")
    print(f"key_metrics       : {result.get('key_metrics')}")
    print()

    (work / "trend_analysis.py").write_text(script)
    print("=== generated trend_analysis.py (first 55 lines) ===")
    for ln in script.splitlines()[:55]:
        print("  " + ln)
    print("  ...\n")

    proc = subprocess.run(
        [sys.executable, "trend_analysis.py"], cwd=work,
        capture_output=True, text=True, timeout=120,
    )
    png = work / "parameter_trends.png"
    print("=== execution ===")
    print(f"exit code    : {proc.returncode}")
    if proc.stderr.strip():
        print(f"stderr       : {proc.stderr.strip()[-700:]}")
    print(f"PNG produced : {png.is_file()}"
          + (f" ({png.stat().st_size} bytes)" if png.is_file() else ""))
    if png.is_file():
        dest = Path.cwd() / "trend_multivar_test_output.png"
        shutil.copy(png, dest)
        print(f"PNG copied to: {dest}")

    sl = script.lower()
    uses_secondary = "secondary_variables" in script
    twod = any(k in sl for k in
               ("heatmap", "contour", "pcolormesh", "imshow", "scatter"))
    print()
    print(f"script references secondary_variables : {uses_secondary}")
    print(f"script uses a 2-D representation       : {twod}")
    ok = proc.returncode == 0 and png.is_file() and uses_secondary and twod
    print()
    print("RESULT:", "PASS" if ok else "CHECK — inspect the script/PNG above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
