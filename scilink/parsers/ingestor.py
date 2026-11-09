import os
from pathlib import Path
from typing import List, Dict, Any

from .pdf_parser import extract_pdf_two_pass, chunk_text
from .excel_parser import parse_adaptive_excel


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

# Extensions the ingestor reads as DOCUMENTS: every file is parsed and
# embedded, however many sit in one folder. A paper is a paper whether it
# arrives alone or as the tenth of a folder.
DOCUMENT_EXTENSIONS = {'.pdf', '.txt', '.md', '.py', '.java', '.r', '.cpp', '.h', '.js'}

# A folder holding this many files of one non-document extension is a
# DIRECTORY DATABASE (thousands of JSON / CSV records): its files are left
# out of the embedded knowledge base and served by plan mode's
# `query_knowledge_data`, which runs generated code over the records.
DIR_DB_MIN_FILES = 10


def is_directory_database_extension(ext: str) -> bool:
    """Whether files of this extension can form a directory database.

    Documents (PDF, text, markdown, code) never do: they are embedded one
    by one. Images never do: they go to the vision path. Everything else —
    JSON and CSV records, spreadsheets, formats the ingestor does not read
    (.cif, .xyz, ...) — does, once a folder holds ``DIR_DB_MIN_FILES`` of
    them. One predicate for the ingestor and the knowledge-data scanner,
    so a folder is never a database to one and documents to the other.
    """
    ext = (ext or "").lower()
    return ext not in DOCUMENT_EXTENSIONS and ext not in IMAGE_EXTENSIONS


def directory_databases(file_paths: List[str]) -> Dict[tuple, int]:
    """Map ``(folder, extension) -> count`` for every folder that holds
    ``DIR_DB_MIN_FILES`` or more files of one directory-database extension
    among ``file_paths`` (already expanded, one file per entry)."""
    from collections import Counter
    counts: Dict[tuple, int] = Counter()
    for f_path in file_paths:
        p = Path(f_path)
        counts[(str(p.parent), p.suffix.lower())] += 1
    return {key: n for key, n in counts.items()
            if n >= DIR_DB_MIN_FILES and is_directory_database_extension(key[1])}

# File extensions the ingestor / directory scanner recognises.
SUPPORTED_EXTENSIONS = {
    '.py', '.java', '.r', '.cpp', '.h', '.js', '.json',
    '.csv', '.txt', '.md', '.pdf',
    '.xlsx', '.xls',
    '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'
}


def get_files_from_directory(directory_path: str) -> List[str]:
    """
    Recursively finds all supported files in a directory, ignoring hidden files.
    """
    found_files = []
    path = Path(directory_path)

    if not path.exists():
        print(f"  - ⚠️ Directory not found: {directory_path}")
        return []

    print(f"  - 📂 Scanning directory: {path.name}...")

    for root, dirs, files in os.walk(path):
        # In-place modification to skip hidden dirs and common junk
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'venv', 'env', 'node_modules', '.git')]

        for file in files:
            if file.startswith('.'): continue

            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found_files.append(str(file_path))

    print(f"    -> Found {len(found_files)} files in directory.")
    return found_files

def extract_images(file_paths: List[str]) -> List[str]:
    """
    Scans directories or file lists specifically for images.
    Used to pass visual context to the Agent without embedding it in the Vector DB.
    """
    found_images = []
    if not file_paths:
        return found_images

    # Reuse the same expansion logic as ingest_files for consistency
    for f_path in file_paths:
        path_obj = Path(f_path)
        
        if path_obj.is_file():
            if path_obj.suffix.lower() in IMAGE_EXTENSIONS:
                found_images.append(str(path_obj))
                
        elif path_obj.is_dir():
            for root, _, files in os.walk(path_obj):
                for file in files:
                    if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                        found_images.append(str(Path(root) / file))
                        
    if found_images:
        print(f"  - 🖼️  Found {len(found_images)} images in input paths.")
        
    return found_images


def ingest_files(file_paths: List[str], is_code_mode: bool, code_chunk_size: int = 20000,
                 repo_name: str = None, ocr_model: Any = None) -> List[Dict[str, Any]]:
    """
    Recursively finds files and routes them to the
    correct parser (PDF, Excel, or Text) based on extension.

    When ``ocr_model`` is supplied, scanned/sparse PDF pages are transcribed
    via the vision-OCR fallback.
    """
    chunks = []
    expanded_paths = []
    
    # 1. Expand directories
    if file_paths:
        for f_path in file_paths:
            path_obj = Path(f_path)
            if path_obj.is_dir():
                expanded_paths.extend(get_files_from_directory(f_path))
            else:
                expanded_paths.append(f_path)

    # 1b. Detect directory databases — skip files that belong to large
    #     uniform collections of RECORDS (e.g., 4764 JSON records). These
    #     are queried structurally via query_knowledge_data, not embedded
    #     as text chunks. Documents never form one: a folder of ten papers
    #     is ten papers (a count-only rule dropped them all and nothing
    #     told the user).
    _skip_dir_ext = directory_databases(expanded_paths)
    for (dir_path, ext), count in _skip_dir_ext.items():
        print(f"  - 📁 {count} {ext or 'extensionless'} files in "
              f"{Path(dir_path).name}: a directory database of records, "
              "not embedded in the knowledge base; query them with "
              "query_knowledge_data")

    # 2. Process each file
    for f_path in expanded_paths:
        path = Path(f_path)
        if not path.exists():
            print(f"  - ⚠️ File not found: {f_path}")
            continue

        # Skip files belonging to detected directory databases
        if (str(path.parent), path.suffix.lower()) in _skip_dir_ext:
            continue

        file_ext = path.suffix.lower()
        
        # --- ROUTE A: PDF Documents ---
        if file_ext == '.pdf':
            pdf_chunks = extract_pdf_two_pass(f_path, ocr_model=ocr_model)
            if is_code_mode:
                for c in pdf_chunks: c['metadata']['content_type'] = 'code'
            chunks.extend(pdf_chunks)

        # --- ROUTE B: Structured Data (Excel/CSV) ---
        elif file_ext in ['.xlsx', '.xls', '.csv'] and not is_code_mode:
            print(f"  - 📊 Auto-detected Data File: {path.name}")
            potential_meta = path.with_suffix('.json')
            meta_context = str(potential_meta) if potential_meta.exists() else None
            
            try:
                data_chunks = parse_adaptive_excel(str(path), context_path=meta_context)
                chunks.extend(data_chunks)
            except Exception as e:
                print(f"    - ❌ Error parsing data file: {e}")

        elif file_ext in IMAGE_EXTENSIONS:
            continue # Skip, handled by extract_images
        
        # --- ROUTE C: Text & Code Files ---
        elif file_ext in ['.txt', '.md', '.py', '.java', '.r', '.cpp', '.h', '.js', '.json', '.csv']:
            try:
                with path.open('r', encoding='utf-8') as f: content = f.read()
                
                if is_code_mode:
                    formatted_text = f"CODE FILE: {path.name}\n\n```\n{content}\n```"
                    chunk_sz = code_chunk_size  # Passed as argument now
                    ctype = 'code'
                else:
                    formatted_text = f"DOCUMENT: {path.name}\n\n{content}"
                    chunk_sz = 1000
                    ctype = 'text'
                
                new_chunks = chunk_text(formatted_text, page_num=1, chunk_size=chunk_sz, overlap=50)
                for c in new_chunks: 
                    c['metadata']['content_type'] = ctype
                    c['metadata']['source'] = f_path
                    if repo_name: c['metadata']['repo_name'] = repo_name
                chunks.extend(new_chunks)
            except Exception as e:
                print(f"  - ❌ Error reading text file {f_path}: {e}")
        else:
            if not path.name.startswith('.'):
                print(f"  - ⚠️ Unsupported file type: {f_path}")

    return chunks