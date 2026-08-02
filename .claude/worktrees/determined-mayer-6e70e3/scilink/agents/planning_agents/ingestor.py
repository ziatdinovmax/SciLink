import os
from pathlib import Path
from typing import List, Dict, Any

from .pdf_parser import extract_pdf_two_pass, chunk_text
from .excel_parser import parse_adaptive_excel
from .parser_utils import get_files_from_directory


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

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


def ingest_files(file_paths: List[str], is_code_mode: bool, code_chunk_size: int = 20000, repo_name: str = None) -> List[Dict[str, Any]]:
    """
    Recursively finds files and routes them to the 
    correct parser (PDF, Excel, or Text) based on extension.
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
    #     uniform collections (e.g., 4764 JSON records). These are queried
    #     structurally via query_knowledge_data, not embedded as text chunks.
    from collections import Counter
    _dir_ext_counts: Dict[str, Counter] = {}
    for f_path in expanded_paths:
        p = Path(f_path)
        parent_key = str(p.parent)
        if parent_key not in _dir_ext_counts:
            _dir_ext_counts[parent_key] = Counter()
        _dir_ext_counts[parent_key][p.suffix.lower()] += 1
    _skip_dir_ext = set()
    for dir_path, ext_counts in _dir_ext_counts.items():
        for ext, count in ext_counts.items():
            if count >= 10:
                _skip_dir_ext.add((dir_path, ext))
                print(f"  - 📁 Skipping {count} {ext} files in {Path(dir_path).name} (directory database, queryable via query_knowledge_data)")

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
            pdf_chunks = extract_pdf_two_pass(f_path)
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