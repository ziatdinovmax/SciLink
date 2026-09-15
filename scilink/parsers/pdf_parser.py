import threading
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from .ocr import (
    SPARSE_PAGE_CHAR_THRESHOLD,
    DEFAULT_OCR_DPI,
    MAX_OCR_PAGES,
    OCR_MARKER,
    ocr_pdf_pages,
)

# `fitz` (PyMuPDF) and `pdfplumber` are imported lazily inside the functions
# that need them, so importing this module stays cheap for callers that only
# touch the lighter helpers.


class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=15, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message
        self.timer = None

    def _timeout_handler(self):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self._timeout_handler)
        self.timer.daemon = True
        self.timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer:
            self.timer.cancel()
        return False

@dataclass
class ContentBlock:
    text: str; page: int; content_type: str


def table_to_markdown(table: List[List[str]]) -> str:
    """Converts a 2D list representation of a table into Markdown format."""
    if not table or not table[0]: return ""
    # Ensure all cells are strings before joining
    cleaned_table = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
    header, *rows = cleaned_table
    md = f"| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n"
    for row in rows:
        # Pad rows that are shorter than the header
        while len(row) < len(header): row.append("")
        # Truncate rows that are longer than the header
        md += f"| {' | '.join(row[:len(header)])} |\n"
    return md


def chunk_text(text: str, page_num: int, chunk_size: int, overlap: int) -> List[Dict[str, any]]:
    """Chunks a single block of text with overlap."""
    chunks = []
    start = 0
    text_length = len(text)
    chunk_idx = 0
    while start < text_length:
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'page': page_num,
                    'content_type': 'text',
                    'chunk_id': f"p{page_num}-t-{chunk_idx}"
                }
            })
            chunk_idx += 1
        start = end - overlap if end < text_length else end
    return chunks


def _extract_pdf_blocks(pdf_path: str, table_timeout: int = 15,
                        max_pages: Optional[int] = None,
                        ocr_model: Any = None,
                        ocr_dpi: int = DEFAULT_OCR_DPI,
                        max_ocr_pages: int = MAX_OCR_PAGES
                        ) -> Tuple[List[Tuple[int, str]], List[Dict[str, any]], int, set]:
    """Shared two-pass PDF extraction, with an optional vision-OCR fallback.

    Pass 1 (PyMuPDF): fast per-page text + detection of pages with tables.
    Pass 2 (pdfplumber): high-accuracy table extraction on flagged pages only.
    OCR fallback: pages whose embedded text layer is empty/sparse are rendered
    and transcribed with ``ocr_model`` — when one is supplied.

    Returns ``(page_texts, table_chunks, n_pages, ocr_pages)`` where
    ``page_texts`` is an ordered list of ``(page_number, text)``,
    ``table_chunks`` are ``{'text', 'metadata'}`` dicts, ``n_pages`` is the
    true page count, and ``ocr_pages`` is the set of 1-indexed pages whose
    text came from vision-OCR. When ``max_pages`` is set, Pass 1 stops after
    that many pages and both Pass 2 and OCR are skipped — a lightweight mode
    for previews/probes.
    """
    import fitz  # PyMuPDF

    page_texts: List[Tuple[int, str]] = []
    table_chunks: List[Dict[str, any]] = []
    table_page_nums = set()
    ocr_pages: set = set()
    sparse_pages: List[int] = []

    # === PASS 1: Fast text extraction + table-page location (PyMuPDF) ===
    doc = fitz.open(pdf_path)
    try:
        n_pages = len(doc)
        last_page = n_pages if max_pages is None else min(n_pages, max_pages)
        for page_num_zero_indexed in range(last_page):
            page = doc[page_num_zero_indexed]
            page_num_one_indexed = page_num_zero_indexed + 1

            text_blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
            full_page_text = "\n\n".join(
                [block[4].strip() for block in text_blocks if block[4].strip()]
            )
            if full_page_text:
                page_texts.append((page_num_one_indexed, full_page_text))

            # A near-empty text layer marks a scanned/sparse page — a candidate
            # for the OCR fallback, and not worth handing to pdfplumber.
            if len(full_page_text) < SPARSE_PAGE_CHAR_THRESHOLD:
                sparse_pages.append(page_num_one_indexed)
            elif page.find_tables():
                table_page_nums.add(page_num_zero_indexed)
    finally:
        doc.close()

    # === PASS 2: Targeted high-accuracy table extraction (pdfplumber) ===
    if table_page_nums and max_pages is None:
        import pdfplumber
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num_zero_indexed in sorted(list(table_page_nums)):
                    page_num_one_indexed = page_num_zero_indexed + 1
                    try:
                        with timeout(seconds=table_timeout):
                            page = pdf.pages[page_num_zero_indexed]
                            tables = page.extract_tables()
                            for table in (tables or []):
                                if table and len(table) > 1:
                                    table_chunks.append({
                                        'text': table_to_markdown(table),
                                        'metadata': {'page': page_num_one_indexed,
                                                     'content_type': 'table'}
                                    })
                    except TimeoutError:
                        print(f"    - ⚠️  Table extraction on page {page_num_one_indexed} timed out. Skipping.")
                    except Exception as e:
                        print(f"    - ⚠️  Error extracting tables from page {page_num_one_indexed}: {e}")
        except Exception as e:
            print(f"❌ Error during table extraction (pdfplumber): {e}")

    # === OCR FALLBACK: transcribe scanned/sparse pages with a vision model ===
    if ocr_model is not None and sparse_pages and max_pages is None:
        transcriptions, _ = ocr_pdf_pages(
            pdf_path, sparse_pages, ocr_model,
            dpi=ocr_dpi, max_ocr_pages=max_ocr_pages,
        )
        if transcriptions:
            ocr_pages = set(transcriptions)
            # Replace any sparse text already captured for these pages with
            # the OCR transcription, then keep page_texts in page order.
            page_texts = [(p, t) for (p, t) in page_texts if p not in ocr_pages]
            page_texts.extend(transcriptions.items())
            page_texts.sort(key=lambda pt: pt[0])

    return page_texts, table_chunks, n_pages, ocr_pages


def _assemble_flat_text(page_texts: List[Tuple[int, str]],
                        table_chunks: List[Dict[str, any]],
                        ocr_pages: frozenset = frozenset()) -> str:
    """Join per-page text and per-page table markdown into one flat string,
    in page order — tables placed after the text of their page. Pages whose
    text came from vision-OCR are prefixed with a provenance marker."""
    tables_by_page: Dict[int, List[str]] = {}
    for tc in table_chunks:
        tables_by_page.setdefault(tc['metadata']['page'], []).append(tc['text'])

    parts: List[str] = []
    seen_pages = set()
    for page_num, text in page_texts:
        parts.append(f"{OCR_MARKER}\n{text}" if page_num in ocr_pages else text)
        seen_pages.add(page_num)
        parts.extend(tables_by_page.get(page_num, []))
    # Tables on pages with no extracted text
    for page_num in sorted(tables_by_page):
        if page_num not in seen_pages:
            parts.extend(tables_by_page[page_num])
    return "\n\n".join(parts)


def extract_pdf_text(pdf_path: str, table_timeout: int = 15,
                     max_pages: Optional[int] = None,
                     ocr_model: Any = None, ocr_dpi: int = DEFAULT_OCR_DPI,
                     max_ocr_pages: int = MAX_OCR_PAGES) -> str:
    """Flat two-pass text extraction (PyMuPDF text + pdfplumber tables as
    markdown), no chunking. For lightweight document reads. When ``ocr_model``
    is supplied, scanned/sparse pages are transcribed via vision-OCR."""
    page_texts, table_chunks, _, ocr_pages = _extract_pdf_blocks(
        pdf_path, table_timeout, max_pages, ocr_model, ocr_dpi, max_ocr_pages)
    return _assemble_flat_text(page_texts, table_chunks, ocr_pages)


def extract_pdf_two_pass(pdf_path: str, chunk_size: int = 500, overlap: int = 50,
                         table_timeout: int = 15, ocr_model: Any = None,
                         ocr_dpi: int = DEFAULT_OCR_DPI,
                         max_ocr_pages: int = MAX_OCR_PAGES) -> List[Dict[str, any]]:
    """
    A robust two-pass hybrid extraction pipeline for RAG.
    Pass 1 (PyMuPDF): Fast extraction of all text and identification of pages containing tables.
    Pass 2 (pdfplumber): High-accuracy extraction of tables from only the identified pages.
    When ``ocr_model`` is supplied, scanned/sparse pages are transcribed via
    vision-OCR; chunks from those pages are tagged ``metadata['ocr'] = True``.
    """
    print(f"Starting robust two-pass processing for: {pdf_path}")

    try:
        page_texts, table_chunks, _, ocr_pages = _extract_pdf_blocks(
            pdf_path, table_timeout, ocr_model=ocr_model,
            ocr_dpi=ocr_dpi, max_ocr_pages=max_ocr_pages)
    except Exception as e:
        print(f"❌ Error during PDF extraction: {e}")
        return []

    text_chunks: List[Dict[str, any]] = []
    for page_num, full_page_text in page_texts:
        page_chunks = chunk_text(full_page_text, page_num, chunk_size, overlap)
        if page_num in ocr_pages:
            for c in page_chunks:
                c['metadata']['ocr'] = True
        text_chunks.extend(page_chunks)

    print(f"  - Extracted {len(text_chunks)} text chunks; "
          f"found {len(table_chunks)} tables"
          f"{f'; {len(ocr_pages)} page(s) via OCR' if ocr_pages else ''}.")

    # === Final Merge and Post-processing ===
    all_content = text_chunks + table_chunks
    all_content.sort(key=lambda x: (x['metadata']['page'], 0 if x['metadata']['content_type'] == 'text' else 1))

    for i, chunk in enumerate(all_content):
        chunk['metadata']['source'] = pdf_path
        chunk['metadata']['chunk_id'] = f"{Path(pdf_path).stem}-{i}"

    print(f"✓ Created {len(all_content)} total chunks ({len(table_chunks)} tables)")
    return all_content
