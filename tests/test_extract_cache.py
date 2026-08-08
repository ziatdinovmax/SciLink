"""Extraction cache: windowed readers stop re-paying parse + vision-OCR.

Live: a planning delegation made three windowed read_file calls on one
proposal PDF and each call re-ran the full extraction INCLUDING a
200-DPI vision-OCR transcription of the same scanned page — one vision
LLM call per window. The cache keys on (path, mtime, size) plus every
output-changing parameter, so the first read pays once and later
windows are string slices; a rewritten file misses and re-extracts.
"""

import json
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("markdown_it")

from scilink.parsers import extract as ex


@pytest.fixture(autouse=True)
def fresh_cache():
    ex._EXTRACT_CACHE.clear()
    yield
    ex._EXTRACT_CACHE.clear()


class CountingOCR:
    def __init__(self):
        self.calls = 0

    def generate_content(self, *a, **k):
        self.calls += 1
        return SimpleNamespace(text="OCR-TRANSCRIBED PAGE CONTENT")


def make_pdf_with_scanned_page(tmp_path):
    """One text page + one image-only (sparse) page -> OCR candidate."""
    from scilink.utils.md_to_pdf import markdown_to_pdf
    md = tmp_path / "doc.md"
    md.write_text("# Proposal\n\n" + "Text-layer prose. " * 200)
    pdf = tmp_path / "doc.pdf"
    markdown_to_pdf(md, pdf)
    doc = fitz.open(str(pdf))
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 400, 400)
    page.draw_rect(rect, fill=(0.5, 0.5, 0.5))     # pixels, no text layer
    doc.saveIncr()
    doc.close()
    return pdf


def test_ocr_paid_once_across_repeated_reads(tmp_path):
    pdf = make_pdf_with_scanned_page(tmp_path)
    ocr = CountingOCR()
    r1 = ex.extract_text(pdf, ocr_model=ocr)
    assert r1["n_ocr_pages"] == 1 and ocr.calls == 1
    assert "OCR-TRANSCRIBED" in r1["text"]

    r2 = ex.extract_text(pdf, ocr_model=ocr)
    r3 = ex.extract_text(pdf, ocr_model=ocr)
    assert ocr.calls == 1                       # cache hits, no re-OCR
    assert r2["text"] == r1["text"] == r3["text"]


def test_returned_dict_is_a_copy(tmp_path):
    pdf = make_pdf_with_scanned_page(tmp_path)
    ocr = CountingOCR()
    r1 = ex.extract_text(pdf, ocr_model=ocr)
    r1["text"] = "MUTATED"
    r2 = ex.extract_text(pdf, ocr_model=ocr)
    assert r2["text"] != "MUTATED"


def test_rewrite_invalidates(tmp_path):
    pdf = make_pdf_with_scanned_page(tmp_path)
    ocr = CountingOCR()
    ex.extract_text(pdf, ocr_model=ocr)
    assert ocr.calls == 1
    # rewrite the file (content + mtime/size change)
    make_pdf_with_scanned_page(tmp_path)
    ex.extract_text(pdf, ocr_model=ocr)
    assert ocr.calls == 2                       # miss -> fresh extraction


def test_output_changing_params_key_separately(tmp_path):
    pdf = make_pdf_with_scanned_page(tmp_path)
    ocr = CountingOCR()
    with_ocr = ex.extract_text(pdf, ocr_model=ocr)
    without = ex.extract_text(pdf)              # no ocr_model: distinct key
    assert "OCR-TRANSCRIBED" in with_ocr["text"]
    assert "OCR-TRANSCRIBED" not in without["text"]
    # and the no-OCR result did not overwrite the OCR'd cache entry
    again = ex.extract_text(pdf, ocr_model=ocr)
    assert "OCR-TRANSCRIBED" in again["text"] and ocr.calls == 1


def test_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("SCILINK_EXTRACT_CACHE", "0")
    pdf = make_pdf_with_scanned_page(tmp_path)
    ocr = CountingOCR()
    ex.extract_text(pdf, ocr_model=ocr)
    ex.extract_text(pdf, ocr_model=ocr)
    assert ocr.calls == 2                       # cache disabled


def test_lru_bound(tmp_path):
    for i in range(ex._EXTRACT_CACHE_MAX + 3):
        sub = tmp_path / f"d{i}"
        sub.mkdir()
        ex.extract_text(make_pdf_with_scanned_page(sub))
    assert len(ex._EXTRACT_CACHE) <= ex._EXTRACT_CACHE_MAX


def test_windowed_read_file_extracts_once(tmp_path):
    """Tool-level integration: three windowed reads of one PDF (the live
    trace's shape) run ONE extraction."""
    from scilink.agents.planning_agents import orchestrator_tools as ot
    from scilink.agents.planning_agents.orchestrator_tools import (
        OrchestratorTools)

    pdf = make_pdf_with_scanned_page(tmp_path)
    t = OrchestratorTools.__new__(OrchestratorTools)
    t.functions_map, t.openai_schemas = {}, []
    t.orch = SimpleNamespace(base_dir=tmp_path, _active_output_subdir=None,
                             planner=SimpleNamespace())
    t._resolve_data_path = lambda p: (str(p), None)
    t._register_tool = (lambda func, name, **kw:
                        t.functions_map.setdefault(name, func))
    OrchestratorTools._register_all_tools(t)
    read = t.functions_map["read_file"]

    heavy = {"n": 0}
    real_blocks = ex._extract_pdf_blocks

    def counting_blocks(*a, **k):
        heavy["n"] += 1
        return real_blocks(*a, **k)

    ex._extract_pdf_blocks = counting_blocks
    try:
        out1 = json.loads(read(file_path=str(pdf)))
        out2 = json.loads(read(file_path=str(pdf), tail=True, max_lines=5))
        out3 = json.loads(read(file_path=str(pdf), search="Proposal"))
    finally:
        ex._extract_pdf_blocks = real_blocks

    assert out1["status"] == out2["status"] == out3["status"] == "success"
    assert heavy["n"] == 1          # one parse; windows 2 and 3 were hits
