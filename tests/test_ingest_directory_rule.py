"""A folder of ten or more documents is embedded; a folder of
ten or more records is a directory database; a keyword-only knowledge base
is read by plan generation.

The folder rule in ``ingest_files`` was written for folders of thousands of
JSON records (queried, not embedded) but decided from the count alone, so
ten PDFs in one folder — every web upload of ten papers lands in one
``knowledge/`` folder — were all skipped, with one misleading console
line. And ``retrieve_context`` returned nothing unless a dense index
existed, so a knowledge base built with no embedding model (the terminal's
default) was never read.
"""

import json
from pathlib import Path

import pytest

from scilink.parsers import ingestor
from scilink.parsers.ingestor import (
    DIR_DB_MIN_FILES, DOCUMENT_EXTENSIONS, IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS, directory_databases, ingest_files,
    is_directory_database_extension,
)
from scilink.knowledge.rag_engine import kb_has_content, retrieve_context
from scilink.knowledge.sparse_retrieval import bm25_top_k


@pytest.fixture
def fake_pdf(monkeypatch):
    """One chunk per PDF, no parser: the rule under test is the folder rule."""
    def _extract(path, ocr_model=None):
        return [{"text": f"paper {Path(path).stem}",
                 "metadata": {"source": str(path), "content_type": "text"}}]
    monkeypatch.setattr(ingestor, "extract_pdf_two_pass", _extract)


def _pdfs(folder: Path, n: int):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"paper_{i:02d}.pdf").write_bytes(b"%PDF-1.4\n")


def _records(folder: Path, n: int, ext=".json"):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"rec_{i:04d}{ext}").write_text('{"id": %d, "value": 1.0}' % i)


def _sources(chunks):
    return sorted(Path(c["metadata"]["source"]).name for c in chunks)


class TestDocumentsAreAlwaysEmbedded:
    def test_ten_pdfs_in_one_folder_are_ingested(self, tmp_path, fake_pdf, capsys):
        _pdfs(tmp_path / "papers", 10)
        chunks = ingest_files([str(tmp_path / "papers")], is_code_mode=False)
        assert len(chunks) == 10
        assert "directory database" not in capsys.readouterr().out

    @pytest.mark.parametrize("n", [9, DIR_DB_MIN_FILES, 25])
    def test_count_does_not_matter_for_documents(self, tmp_path, fake_pdf, n):
        _pdfs(tmp_path / "papers", n)
        assert len(ingest_files([str(tmp_path / "papers")], is_code_mode=False)) == n

    def test_ten_pdfs_passed_as_file_paths(self, tmp_path, fake_pdf):
        _pdfs(tmp_path / "papers", 10)
        paths = [str(p) for p in sorted((tmp_path / "papers").glob("*.pdf"))]
        assert len(ingest_files(paths, is_code_mode=False)) == 10

    def test_ten_text_documents_are_ingested(self, tmp_path):
        folder = tmp_path / "notes"; folder.mkdir()
        for i in range(12):
            (folder / f"note_{i}.md").write_text(f"# note {i}\n\nlithium sorbent {i}")
        chunks = ingest_files([str(folder)], is_code_mode=False)
        assert len(chunks) == 12

    def test_code_mode_repo_folder_is_ingested(self, tmp_path):
        """The same count-only rule dropped a repo folder of ten or more
        ``.py`` files from the code KB."""
        folder = tmp_path / "pkg"; folder.mkdir()
        for i in range(12):
            (folder / f"mod_{i}.py").write_text(f"def f{i}():\n    return {i}\n")
        chunks = ingest_files([str(folder)], is_code_mode=True)
        assert len(chunks) == 12
        assert {c["metadata"]["content_type"] for c in chunks} == {"code"}

    def test_documents_kept_beside_a_records_folder(self, tmp_path, fake_pdf, capsys):
        """A folder of records with a few papers in it: the records are the
        database, the papers are embedded."""
        folder = tmp_path / "mixed"
        _records(folder, 12)
        _pdfs(folder, 3)
        chunks = ingest_files([str(folder)], is_code_mode=False)
        assert _sources(chunks) == ["paper_00.pdf", "paper_01.pdf", "paper_02.pdf"]
        out = capsys.readouterr().out
        assert "12 .json files in mixed" in out and "query_knowledge_data" in out


class TestRecordsAreStillADirectoryDatabase:
    def test_ten_json_records_are_skipped_and_said(self, tmp_path, capsys):
        _records(tmp_path / "db", 10)
        assert ingest_files([str(tmp_path / "db")], is_code_mode=False) == []
        out = capsys.readouterr().out
        assert "10 .json files in db" in out
        assert "not embedded" in out and "query_knowledge_data" in out

    def test_nine_json_files_are_documents(self, tmp_path):
        _records(tmp_path / "few", 9)
        assert len(ingest_files([str(tmp_path / "few")], is_code_mode=False)) == 9

    def test_csv_records_too(self, tmp_path):
        _records(tmp_path / "db", 10, ext=".csv")
        assert ingest_files([str(tmp_path / "db")], is_code_mode=False) == []

    def test_unreadable_record_formats_are_one_line_not_one_per_file(self, tmp_path, capsys):
        _records(tmp_path / "cifs", 30, ext=".cif")
        paths = [str(p) for p in (tmp_path / "cifs").glob("*.cif")]
        assert ingest_files(paths, is_code_mode=False) == []
        out = capsys.readouterr().out
        assert out.count("Unsupported file type") == 0
        assert "30 .cif files in cifs" in out


class TestOneRuleForIngestorAndScanner:
    @pytest.mark.parametrize("ext,expected", [
        (".pdf", False), (".txt", False), (".md", False), (".py", False),
        (".png", False), (".json", True), (".csv", True), (".xlsx", True),
        (".cif", True), ("", True), (".JSON", True), (".PDF", False),
    ])
    def test_predicate(self, ext, expected):
        assert is_directory_database_extension(ext) is expected

    def test_directory_databases_keys(self, tmp_path):
        _records(tmp_path / "db", 10)
        _pdfs(tmp_path / "db", 10)
        files = [str(p) for p in (tmp_path / "db").iterdir()]
        assert directory_databases(files) == {(str(tmp_path / "db"), ".json"): 10}

    def test_partition_of_supported_extensions(self):
        """Every supported extension is a document, an image or a record.
        A newly supported document type added to the ingestor's routes but
        not to DOCUMENT_EXTENSIONS would fall under the folder rule again —
        the dropped-papers bug for that type."""
        assert SUPPORTED_EXTENSIONS - DOCUMENT_EXTENSIONS - IMAGE_EXTENSIONS == {
            '.csv', '.json', '.xls', '.xlsx'}
        assert DOCUMENT_EXTENSIONS <= SUPPORTED_EXTENSIONS
        assert not (DOCUMENT_EXTENSIONS & IMAGE_EXTENSIONS)

    def test_constants_are_the_ingestors(self):
        src = Path("scilink/agents/planning_agents/orchestrator_tools.py").read_text()
        assert "DIR_DB_MIN_FILES = 10" not in src       # one definition, the ingestor's


def _tools(tmp_path, knowledge_dir):
    """The planning tools over a mock orchestrator, enough for the
    knowledge-data tool's discovery and directory query."""
    from unittest.mock import MagicMock
    from scilink.agents.planning_agents.orchestrator_tools import OrchestratorTools
    orch = MagicMock()
    orch.base_dir = str(tmp_path / "session")
    orch.knowledge_dir = str(knowledge_dir)
    orch.planner.model.generate_content.side_effect = RuntimeError("reached code generation")
    return OrchestratorTools(orch), orch


@pytest.fixture
def knowledge_folder(tmp_path):
    k = tmp_path / "knowledge"
    _pdfs(k / "papers", 12)                      # documents: never a database
    _records(k / "records_json", 12)             # a database
    _records(k / "records_csv", 12, ext=".csv")  # a database
    folder = k / "txt_records"; folder.mkdir()  # text-file records
    for i in range(12):
        (folder / f"run_{i:03d}.txt").write_text(
            "\n".join(f"{j}\t{0.5 + 0.001 * j}" for j in range(50)))
    return k


class TestKnowledgeDataScanner:
    def test_discovery_offers_record_folders_not_documents(self, tmp_path, knowledge_folder):
        tools, _ = _tools(tmp_path, knowledge_folder)
        out = json.loads(tools.functions_map["query_knowledge_data"]("how many records"))
        assert out["status"] == "file_selection_needed"
        offered = out["available_files"]
        assert any(n.startswith("records_json (") for n in offered)
        assert any(n.startswith("records_csv (") for n in offered)
        assert not any(n.startswith("papers") for n in offered)
        assert not any(n.startswith("txt_records") for n in offered)

    def test_explicit_path_to_text_records_reaches_code_generation(
            self, tmp_path, knowledge_folder, monkeypatch):
        """The review's case: 200 two-column .txt files queried by their
        explicit path. A document test in the SAMPLING step returned
        'Directory is empty or unreadable'; discovery alone carries the
        rule, so the caller's folder is inspected and the query goes on."""
        from scilink import executors
        monkeypatch.setenv("UNSAFE_EXECUTION_OK", "true")
        monkeypatch.setattr(executors, "_GLOBAL_SANDBOX_APPROVED", False)  # restored after
        tools, orch = _tools(tmp_path, knowledge_folder)
        out = json.loads(tools.functions_map["query_knowledge_data"](
            "mean of the second column", file_name=str(knowledge_folder / "txt_records")))
        assert out["status"] == "error"
        assert "reached code generation" in out["message"]
        assert "empty or unreadable" not in out["message"]
        prompt = orch.planner.model.generate_content.call_args[0][0][0]
        assert "run_006.txt" in prompt                 # a text record was sampled


CORPUS = [
    {"text": "Lithium recovery from produced water using LDH sorbents.",
     "metadata": {"source": "lithium_note.md", "content_type": "doc"}},
    {"text": "Iodine membranes recovered iodide from Permian brine.",
     "metadata": {"source": "iodine_note.md", "content_type": "doc"}},
]


class _KeywordOnlyKB:
    """What KnowledgeBase.build leaves when no embedding model is set: chunks,
    no dense index; retrieve() serves BM25 itself."""
    index = None
    chunks = CORPUS

    def retrieve(self, query, top_k=10):
        hits, _ = bm25_top_k(self.chunks, query, top_k=top_k)
        return hits

    def retrieve_sparse(self, query, top_k=10):
        return self.retrieve(query, top_k)


class _EmptyKB(_KeywordOnlyKB):
    chunks = []


class TestKeywordOnlyKBIsRead:
    def test_retrieve_context_serves_a_keyword_only_kb(self):
        out = retrieve_context(_KeywordOnlyKB(), "lithium sorbent recovery")
        assert "lithium_note.md" in out and "LDH" in out

    def test_empty_kb_still_yields_nothing(self):
        assert retrieve_context(_EmptyKB(), "lithium") == ""
        assert retrieve_context(None, "lithium") == ""

    def test_kb_has_content(self):
        assert kb_has_content(_KeywordOnlyKB())
        assert not kb_has_content(_EmptyKB())
        assert not kb_has_content(None)

    def test_real_keyword_only_kb_end_to_end(self):
        """The reproduction from the issue on a real KnowledgeBase whose
        build failed over to keyword-only."""
        from tests.test_kb_keyword_only import make_kb, CHUNKS
        kb = make_kb()
        kb.build(list(CHUNKS))
        assert kb.index is None
        out = retrieve_context(kb, "throughput Opentrons handoff", top_k=2)
        assert "Opentrons" in out
