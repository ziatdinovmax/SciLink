import fitz  # PyMuPDF
import pdfplumber
import threading
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

from .parser_utils import table_to_markdown


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

def extract_pdf_two_pass(pdf_path: str, chunk_size: int = 500, overlap: int = 50, table_timeout: int = 15) -> List[Dict[str, any]]:
    """
    A robust two-pass hybrid extraction pipeline for RAG. This is the stable version.
    Pass 1 (PyMuPDF): Fast extraction of all text and identification of pages containing tables.
    Pass 2 (pdfplumber): High-accuracy extraction of tables from only the identified pages.
    """
    print(f"Starting robust two-pass processing for: {pdf_path}")
    
    text_chunks = []
    table_chunks = []
    table_page_nums = set()

    # === PASS 1: Fast Text Extraction and Table Location with PyMuPDF ===
    print("  - Pass 1: Extracting text and locating potential tables...")
    try:
        doc = fitz.open(pdf_path)
        for page_num_zero_indexed in range(len(doc)):
            page = doc[page_num_zero_indexed]
            page_num_one_indexed = page_num_zero_indexed + 1

            # 1.1 Extract and chunk text for the current page
            text_blocks = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
            full_page_text = "\n\n".join([block[4].strip() for block in text_blocks if block[4].strip()])
            
            if full_page_text:
                text_chunks.extend(chunk_text(full_page_text, page_num_one_indexed, chunk_size, overlap))

            # 1.2 Identify pages that might contain tables for the next pass
            if page.find_tables():
                table_page_nums.add(page_num_zero_indexed)
        doc.close()
        print(f"  - Pass 1 Complete: Extracted {len(text_chunks)} text chunks.")
        print(f"  - Found {len(table_page_nums)} pages that may contain tables.")

    except Exception as e:
        print(f"❌ Error during Pass 1 (PyMuPDF processing): {e}")
        return []

    # === PASS 2: Targeted, High-Accuracy Table Extraction with pdfplumber ===
    if table_page_nums:
        print("  - Pass 2: Performing high-accuracy table extraction on specific pages...")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num_zero_indexed in sorted(list(table_page_nums)):
                    page_num_one_indexed = page_num_zero_indexed + 1
                    try:
                        with timeout(seconds=table_timeout):
                            page = pdf.pages[page_num_zero_indexed]
                            tables = page.extract_tables()
                            if tables:
                                #print(f"    - Extracted {len(tables)} table(s) from page {page_num_one_indexed}.")
                                for table in tables:
                                    if table and len(table) > 1:
                                        markdown_table = table_to_markdown(table)
                                        table_chunks.append({
                                            'text': markdown_table,
                                            'metadata': {'page': page_num_one_indexed, 'content_type': 'table'}
                                        })
                    except TimeoutError:
                        print(f"    - ⚠️  Table extraction on page {page_num_one_indexed} timed out. Skipping.")
                    except Exception as e:
                        print(f"    - ⚠️  Error extracting tables from page {page_num_one_indexed}: {e}")
            print("  - Pass 2 Complete.")
        except Exception as e:
            print(f"❌ Error during Pass 2 (pdfplumber processing): {e}")

    # === Final Merge and Post-processing ===
    print("  - Merging and finalizing chunks...")
    all_content = text_chunks + table_chunks
    all_content.sort(key=lambda x: (x['metadata']['page'], 0 if x['metadata']['content_type'] == 'text' else 1))

    for i, chunk in enumerate(all_content):
        chunk['metadata']['source'] = pdf_path
        chunk['metadata']['chunk_id'] = f"{Path(pdf_path).stem}-{i}"

    print(f"✓ Created {len(all_content)} total chunks ({len(table_chunks)} tables)")
    return all_content