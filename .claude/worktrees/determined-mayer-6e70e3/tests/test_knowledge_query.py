"""
Knowledge query feature test suite.

Tests the query_knowledge_data tool, read_file Excel/CSV support,
and the data inspection workflow end-to-end through the orchestrator.

Requires GEMINI_API_KEY env var. Run with:

    GEMINI_API_KEY=<key> python tests/test_knowledge_query.py

Or run a subset by number:

    GEMINI_API_KEY=<key> python tests/test_knowledge_query.py 1 3 5
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("UNSAFE_EXECUTION_OK", "true")

from scilink.agents.planning_agents.planning_orchestrator import (
    PlanningOrchestratorAgent,
    AutonomyLevel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp():
    return Path(tempfile.mkdtemp(prefix="kq_test_"))


_MODEL = os.environ.get("SCILINK_TEST_MODEL", "gemini-3.1-pro-preview")
_API_KEY = os.environ.get("SCILINK_TEST_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
_EMBEDDING_API_KEY = os.environ.get("GEMINI_API_KEY", "")

PWS_PATH = Path("planning_session_20260318_163925/knowledge/PWSdatabase.xlsx")


def _orch(base_dir, knowledge_dir=None, data_dir=None):
    if data_dir is None:
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        base_dir=str(base_dir / "session"),
        api_key=_API_KEY,
        model_name=_MODEL,
        embedding_api_key=_EMBEDDING_API_KEY,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        data_dir=str(data_dir),
    )
    if knowledge_dir:
        kwargs["knowledge_dir"] = str(knowledge_dir)
    return kwargs


def _synthetic_reference_db(path, n_rows=300):
    """Create a synthetic produced water reference database."""
    np.random.seed(42)
    basins = ["Permian", "Appalachian", "Williston", "Gulf Coast", "Denver"]
    states = ["Texas", "West Virginia", "North Dakota", "Louisiana", "Colorado"]
    records = []
    for i in range(n_rows):
        b_idx = np.random.randint(0, len(basins))
        records.append({
            "WELLID": f"W{i:04d}",
            "BASIN": basins[b_idx],
            "STATE": states[b_idx],
            "Li": round(np.random.lognormal(1.5, 1.2), 2),
            "Na": round(np.random.lognormal(9, 1), 1),
            "Mg": round(np.random.lognormal(5, 1.5), 2),
            "Sr": round(np.random.lognormal(3, 1.5), 2),
            "Ca": round(np.random.lognormal(7, 1), 1),
            "TDS": round(np.random.lognormal(10, 1), 0),
        })
    df = pd.DataFrame(records)
    if path.suffix == '.csv':
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)


def _icpms_data(path):
    """Create a simple ICP-MS primary data file."""
    df = pd.DataFrame({
        "Na_ppb": [257.92],
        "Mg_ppb": [5.026],
        "K_ppb": [0.696],
        "Ca_ppb": [8.327],
        "Cu_ppb": [0.002],
        "Sr_ppb": [0.273],
        "Ba_ppb": [0.001],
    })
    meta = {
        "title": "ICP-MS on produced water from Permian Basin",
        "objective": "Critical material recovery",
    }
    df.to_excel(path, index=False)
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS = []


def _test(fn):
    TESTS.append(fn)
    return fn


# ===== GROUP 1: query_knowledge_data tool (direct) =====

@_test
def query_csv_simple():
    """Query a CSV knowledge file for a simple aggregation."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.csv")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="What is the average Li concentration?",
        file_name="water_ref.csv",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success" and result["answer"] is not None
    return ok, f"status={result['status']}, answer={result.get('answer')}, summary={result.get('summary', '')}"


@_test
def query_excel_with_filter():
    """Query an Excel knowledge file with a filter condition."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="What is the average Li concentration in the Permian basin?",
        file_name="water_ref.xlsx",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success" and result["answer"] is not None
    return ok, f"status={result['status']}, answer={result.get('answer')}"


@_test
def query_groupby():
    """Query with a group-by aggregation."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.csv")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="Show the average Li concentration per basin.",
        file_name="water_ref.csv",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success" and result["answer"] is not None
    return ok, f"status={result['status']}, answer={result.get('answer')}"


@_test
def query_top_n():
    """Query for top-N records."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="Show the top 5 wells with the highest Sr concentration, including basin and state.",
        file_name="water_ref.xlsx",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success" and result["answer"] is not None
    return ok, f"status={result['status']}, answer={result.get('answer')}"


@_test
def query_auto_select_single_file():
    """With one queryable file, auto-selects without file_name."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.csv")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="How many rows are there?",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success"
    return ok, f"status={result['status']}, answer={result.get('answer')}"


@_test
def query_multiple_files_no_name():
    """With multiple files and no file_name, returns file list."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.csv")
    _synthetic_reference_db(kdir / "water_ref2.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="How many rows?",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "file_selection_needed" and len(result.get("available_files", [])) == 2
    return ok, f"status={result['status']}, files={result.get('available_files')}"


@_test
def query_fuzzy_file_name():
    """Fuzzy file name matching suggests corrections."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.csv")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="count rows",
        file_name="watr_ref.csv",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    # Should either fuzzy-resolve or suggest corrections
    ok = result["status"] in ("success", "error")
    return ok, f"status={result['status']}, message={result.get('message', '')}"


@_test
def query_empty_knowledge_dir():
    """Empty knowledge dir returns clear error."""
    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="How many rows?",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "error" and "No queryable" in result.get("message", "")
    return ok, f"status={result['status']}, message={result.get('message', '')}"


# ===== GROUP 2: Real PWSdatabase.xlsx =====

@_test
def pws_unique_basins():
    """Query real PWS database for unique basin count."""
    if not (Path.cwd() / PWS_PATH).exists():
        return False, f"SKIP: {PWS_PATH} not found"

    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    shutil.copy(Path.cwd() / PWS_PATH, kdir / "PWSdatabase.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="How many unique basins are in the dataset?",
        file_name="PWSdatabase.xlsx",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success" and result["answer"] is not None
    return ok, f"answer={result.get('answer')}, summary={result.get('summary', '')}"


@_test
def pws_permian_li():
    """Query real PWS database for average Li in Permian Basin."""
    if not (Path.cwd() / PWS_PATH).exists():
        return False, f"SKIP: {PWS_PATH} not found"

    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    shutil.copy(Path.cwd() / PWS_PATH, kdir / "PWSdatabase.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="What is the average lithium (Li) concentration in the Permian Basin? "
              "Note: the first row may be column descriptions, not data.",
        file_name="PWSdatabase.xlsx",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success" and result["answer"] is not None
    return ok, f"answer={result.get('answer')}, summary={result.get('summary', '')}"


@_test
def pws_top_sr():
    """Query real PWS database for top 5 wells by Sr."""
    if not (Path.cwd() / PWS_PATH).exists():
        return False, f"SKIP: {PWS_PATH} not found"

    d = _tmp()
    kdir = d / "knowledge"
    kdir.mkdir()
    shutil.copy(Path.cwd() / PWS_PATH, kdir / "PWSdatabase.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir))
    result_json = o.tools.execute_tool(
        "query_knowledge_data",
        query="Show the top 5 wells with the highest strontium (Sr) concentration, "
              "including their basin, state, and well name.",
        file_name="PWSdatabase.xlsx",
    )
    shutil.rmtree(d, True)

    result = json.loads(result_json)
    ok = result["status"] == "success" and result["answer"] is not None
    return ok, f"answer={result.get('answer')}"


# ===== GROUP 3: End-to-end orchestration tests =====


def _extract_tool_calls(orch):
    """Extract ordered list of tool names from orchestrator message history."""
    messages = orch.messages if hasattr(orch, 'messages') else []
    tool_names = []
    for m in messages:
        if hasattr(m, 'tool_calls') and m.tool_calls:
            for tc in m.tool_calls:
                tool_names.append(tc.function.name if hasattr(tc, 'function') else str(tc))
        elif isinstance(m, dict) and m.get('role') == 'assistant':
            for tc in m.get('tool_calls', []):
                fn = tc.get('function', {})
                tool_names.append(fn.get('name', ''))
    return tool_names


@_test
def e2e_read_file_then_tea():
    """Orchestrator should read_file first, then route to TEA without scalarizer."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _icpms_data(data_dir / "icpms.xlsx")

    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir, data_dir=data_dir))
    r = o.chat(
        f"Using the reference water database in knowledge and the ICP-MS results at "
        f"{data_dir / 'icpms.xlsx'}, assess which elements might be worth recovering. "
        f"Run a technoeconomic analysis."
    )
    shutil.rmtree(d, True)

    tool_names = _extract_tool_calls(o)
    used_read = "read_file" in tool_names
    used_tea = "run_economic_analysis" in tool_names
    used_scalarizer = "analyze_file" in tool_names

    ok = used_tea and not used_scalarizer
    return ok, f"tools={tool_names}, read_file={used_read}, tea={used_tea}, scalarizer={used_scalarizer}"


@_test
def e2e_read_file_before_data_tools():
    """read_file should be called BEFORE any data processing tool."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _icpms_data(data_dir / "icpms.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    r = o.chat(
        f"I have ICP-MS data at {data_dir / 'icpms.xlsx'}. "
        f"Assess which elements are present and run a technoeconomic analysis."
    )
    shutil.rmtree(d, True)

    tool_names = _extract_tool_calls(o)
    data_tools = {"analyze_file", "run_economic_analysis", "generate_initial_plan", "query_knowledge_data"}

    # Find first read_file and first data tool
    first_read = next((i for i, t in enumerate(tool_names) if t == "read_file"), None)
    first_data = next((i for i, t in enumerate(tool_names) if t in data_tools), None)

    if first_read is not None and first_data is not None:
        ok = first_read < first_data
    else:
        ok = False

    return ok, f"tools={tool_names}, first_read_at={first_read}, first_data_at={first_data}"


@_test
def e2e_clean_data_routes_to_tea():
    """Clean simple data should route directly to TEA, not through scalarizer."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    # Clean, simple 1-row data — should go straight to TEA
    _icpms_data(data_dir / "icpms.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    r = o.chat(
        f"Run a technoeconomic analysis on the ICP-MS data at {data_dir / 'icpms.xlsx'}. "
        f"Determine which elements are worth recovering."
    )
    shutil.rmtree(d, True)

    tool_names = _extract_tool_calls(o)
    used_tea = "run_economic_analysis" in tool_names
    used_scalarizer = "analyze_file" in tool_names

    ok = used_tea and not used_scalarizer
    return ok, f"tools={tool_names}, tea={used_tea}, scalarizer={used_scalarizer}"


@_test
def e2e_messy_data_routes_to_query():
    """Messy data with description rows should route through query_knowledge_data."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()

    # Create messy data with a description row (like PWSdatabase)
    desc_row = {"Element": "Element name", "Conc": "Concentration in ppm", "Unit": "Measurement unit"}
    data_rows = [
        {"Element": "Li", "Conc": "1.5", "Unit": "ppm"},
        {"Element": "Na", "Conc": "25000", "Unit": "ppm"},
        {"Element": "Mg", "Conc": "1200", "Unit": "ppm"},
    ]
    df = pd.concat([pd.DataFrame([desc_row]), pd.DataFrame(data_rows)], ignore_index=True)
    messy_path = data_dir / "messy_data.xlsx"
    df.to_excel(messy_path, index=False)

    kdir = d / "knowledge"
    kdir.mkdir()
    # Put the messy file in knowledge so query_knowledge_data can find it
    shutil.copy(messy_path, kdir / "messy_data.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir, data_dir=data_dir))
    r = o.chat(
        f"I have compositional data at {messy_path}. The first row contains column "
        f"descriptions, not actual data. Extract the element concentrations and tell me "
        f"which elements are present and at what concentrations."
    )
    shutil.rmtree(d, True)

    tool_names = _extract_tool_calls(o)
    used_read = "read_file" in tool_names
    used_query = "query_knowledge_data" in tool_names

    # Should inspect first, then use query to handle the messy format
    ok = used_read and used_query
    return ok, f"tools={tool_names}, read_file={used_read}, query_kd={used_query}"


@_test
def e2e_txt_data_as_additional_context():
    """Non-tabular .txt data should be passed as additional_context, not to scalarizer."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()

    # Simple text data file
    txt_path = data_dir / "sample_notes.txt"
    txt_path.write_text(
        "Sample: Permian Basin produced water PB-2024-03\n"
        "Collection date: 2024-03-15\n"
        "pH: 6.8\n"
        "TDS: 180,000 mg/L\n"
        "Na: 58,000 ppm\n"
        "Ca: 12,500 ppm\n"
        "Mg: 1,800 ppm\n"
        "Li: 45 ppm\n"
        "Sr: 620 ppm\n"
        "Notes: High salinity sample, slight H2S odor\n"
    )

    o = PlanningOrchestratorAgent(**_orch(d, data_dir=data_dir))
    r = o.chat(
        f"I have sample analysis notes at {txt_path}. "
        f"Based on these measurements, run a technoeconomic analysis to assess "
        f"which elements are worth recovering."
    )
    shutil.rmtree(d, True)

    tool_names = _extract_tool_calls(o)
    used_read = "read_file" in tool_names
    used_scalarizer = "analyze_file" in tool_names
    used_tea = "run_economic_analysis" in tool_names

    # Should read the txt, then pass content to TEA (not scalarizer)
    ok = used_read and not used_scalarizer
    return ok, f"tools={tool_names}, read_file={used_read}, tea={used_tea}, scalarizer={used_scalarizer}"


@_test
def e2e_query_knowledge_in_planning():
    """Orchestrator uses query_knowledge_data and passes context to planning."""
    d = _tmp()
    data_dir = d / "data"
    data_dir.mkdir()
    _icpms_data(data_dir / "icpms.xlsx")

    kdir = d / "knowledge"
    kdir.mkdir()
    _synthetic_reference_db(kdir / "water_ref.xlsx")

    o = PlanningOrchestratorAgent(**_orch(d, knowledge_dir=kdir, data_dir=data_dir))
    r = o.chat(
        f"I have ICP-MS data at {data_dir / 'icpms.xlsx'} and a reference water database "
        f"in knowledge. First query the reference database to find out which basins have "
        f"the highest lithium concentrations and compare with my sample. Then generate "
        f"an experimental plan for lithium recovery using an Opentrons OT-2."
    )
    shutil.rmtree(d, True)

    tool_names = _extract_tool_calls(o)
    used_query = "query_knowledge_data" in tool_names
    used_plan = "generate_initial_plan" in tool_names
    has_plan = o.planner.state is not None and o.planner.state.get("current_plan") is not None

    ok = used_query and has_plan
    return ok, f"tools={tool_names}, query_kd={used_query}, plan={used_plan}, has_plan={has_plan}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _API_KEY:
        print("Set GEMINI_API_KEY (or SCILINK_TEST_API_KEY) env var first.")
        sys.exit(1)
    if not _EMBEDDING_API_KEY:
        print("Set GEMINI_API_KEY env var (needed for embeddings).")
        sys.exit(1)
    print(f"Model: {_MODEL}")
    print(f"API key: {_API_KEY[:10]}...")
    print(f"Embedding key: {_EMBEDDING_API_KEY[:10]}...")

    args = sys.argv[1:]
    if not args:
        to_run = TESTS
        print(f"Running all {len(TESTS)} tests")
    else:
        indices = [int(a) - 1 for a in args]
        to_run = [TESTS[i] for i in indices]

    results = {}
    for fn in to_run:
        name = fn.__name__
        desc = (fn.__doc__ or "").strip().split("\n")[0]
        print(f"\n{'=' * 60}")
        print(f"[{TESTS.index(fn) + 1}/{len(TESTS)}] {name}: {desc}")
        print("=" * 60)
        try:
            ok, detail = fn()
            status = "PASS" if ok else "FAIL"
            results[name] = status
            print(f"  -> {status}: {detail[:300]}")
        except KeyboardInterrupt:
            results[name] = "SKIP"
            break
        except Exception as e:
            results[name] = "ERROR"
            print(f"  -> ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        marker = {"PASS": "V", "FAIL": "X", "ERROR": "!", "SKIP": "-"}.get(status, "?")
        print(f"  [{marker}] {name}: {status}")
    total = len(results)
    passed = sum(1 for v in results.values() if v == "PASS")
    print(f"\n{passed}/{total} passed")
