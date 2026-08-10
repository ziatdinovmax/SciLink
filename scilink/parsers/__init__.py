"""Document parsing layer — files → text / tables / chunks.

A bottom-level utility layer with light dependencies (``pymupdf``,
``pdfplumber``, ``pandas``, ``openpyxl``, ``python-docx``). It imports nothing
else in ``scilink``; agents depend on it, never the reverse.

Two entry points:

- ``extract_text(path)`` — flat text + markdown tables, no embeddings. For
  reading a handful of documents straight into context.
- ``extract_document(path)`` — the same text, plus a .docx's embedded figures
  as image bytes, for callers that can deliver images.
- ``ingest_files(...)`` — chunked output for ``scilink.knowledge.KnowledgeBase``
  ingestion (the heavyweight retrieval path).
"""

from .extract import extract_document, extract_text
from .pdf_parser import (
    extract_pdf_text,
    extract_pdf_two_pass,
    chunk_text,
    table_to_markdown,
)
from .excel_parser import parse_adaptive_excel
from .ingestor import (
    ingest_files,
    extract_images,
    get_files_from_directory,
    SUPPORTED_EXTENSIONS,
)

__all__ = [
    "extract_text",
    "extract_document",
    "extract_pdf_text",
    "extract_pdf_two_pass",
    "chunk_text",
    "table_to_markdown",
    "parse_adaptive_excel",
    "ingest_files",
    "extract_images",
    "get_files_from_directory",
    "SUPPORTED_EXTENSIONS",
]
