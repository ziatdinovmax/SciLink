import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# `pandas` is imported lazily inside parse_adaptive_excel so importing this
# module stays cheap for callers that don't touch tabular data.

# If a file has this many rows or fewer, we embed it all in one chunk.
SMALL_FILE_THRESHOLD = 150

def parse_adaptive_excel(file_path: str, context_path: Optional[str] = None, row_chunk_size: int = 200) -> List[Dict[str, Any]]:
    """
    Reads a Data file (Excel or CSV) and an optional JSON context file.

    Adaptive Strategy:
    - If rows <= SMALL_FILE_THRESHOLD:
      Creates ONE chunk containing the summary, definitions, AND the full data table.
    - If rows > SMALL_FILE_THRESHOLD:
      Creates TWO types of chunks:
      1. A single "summary chunk" with statistical info.
      2. Multiple "data chunks" by batching the rows.
    """
    import pandas as pd

    path_obj = Path(file_path)
    print(f"  - Processing Data File '{path_obj.name}' with adaptive strategy...")
    all_chunks = []

    try:
        # --- 1. Robust Context Loading ---
        context = {}
        if context_path and Path(context_path).exists():
            try:
                with open(context_path, 'r', encoding='utf-8') as f:
                    context = json.load(f)
            except Exception as e:
                print(f"    - ⚠️ Warning: Could not load context file {context_path}: {e}")

        # --- 2. Load the Data File (CSV vs Excel detection) ---
        try:
            suffix = path_obj.suffix.lower()
            if suffix == '.csv':
                df = pd.read_csv(file_path)
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                print(f"    - ❌ Error: Unsupported file extension '{suffix}'")
                return []
        except ImportError:
            print("    - ❌ Error: 'pandas' or 'openpyxl' not installed. Please run: pip install pandas openpyxl")
            return []
        except Exception as e:
            print(f"    - ❌ Error reading file: {e}")
            return []
        
        total_rows = len(df)
        print(f"    - Loaded {total_rows} rows.")

        # --- 3. Base Content Construction ---
        
        description_parts = []
        
        # Get title: Use 'title' from context if present, else fallback to filename
        title = context.get('title', path_obj.stem)
        description_parts.append(f"### Experiment Data: {title}")
        
        # Get objective: Only add if present
        if context.get("objective"):
            description_parts.append(f"#### Objective\n{context['objective']}")

        # Get or create column definitions
        column_defs_dict = context.get('column_definitions')
        if not column_defs_dict:
            # Create dummy definitions from DataFrame column headers
            column_defs_dict = {str(header): "No definition provided." for header in df.columns}

        col_defs = "\n".join([f"- `{col}`: {desc}" for col, desc in column_defs_dict.items()])
        description_parts.append(f"#### Data Column Definitions\n{col_defs}")
        
        statistical_summary = df.describe().to_markdown() if not df.empty else "No statistical summary available."

        # --- 4. Adaptive Chunking Logic ---
        
        if total_rows <= SMALL_FILE_THRESHOLD:
            # --- STRATEGY A: Small File (One Rich Chunk) ---
            # print(f"    - File is small ({total_rows} rows). Creating one single, comprehensive chunk.")
            
            full_data_table = df.to_markdown(index=False)
            
            base_description = "\n\n".join(description_parts)
            
            combined_text = f"""
{base_description}

#### Statistical Summary
{statistical_summary}

#### Full Experimental Data ({total_rows} rows)
{full_data_table}
            """.strip()

            single_chunk = {
                'text': combined_text,
                'metadata': {
                    'source': file_path,
                    'context_source': context_path if context_path else "N/A",
                    'content_type': 'dataset_package', 
                    'page': 1 
                }
            }
            all_chunks.append(single_chunk)
            # print(f"    - ✅ Created 1 'dataset_package' chunk.")

        else:
            # --- STRATEGY B: Large File (Summary + Data Chunks) ---
            print(f"    - File is large ({total_rows} rows). Creating summary + batched data chunks.")
            
            # 4.1 Create the "Summary Chunk"
            base_description = "\n\n".join(description_parts)
            
            summary_text = f"""
{base_description}

#### Statistical Summary of {total_rows} Rows
{statistical_summary}
            """.strip()

            summary_chunk = {
                'text': summary_text,
                'metadata': {
                    'source': file_path,
                    'context_source': context_path if context_path else "N/A",
                    'content_type': 'dataset_summary',
                    'page': 1 
                }
            }
            all_chunks.append(summary_chunk)
            
            # Large tabular data is best accessed via code execution (scalarizer),
            # not RAG retrieval. Only the summary chunk is embedded.
            print(f"    - ✅ Created 1 summary chunk (row data accessed via code execution).")
        
        return all_chunks

    except Exception as e:
        print(f"    - ❌ Error processing data pair for '{file_path}': {e}")
        return []