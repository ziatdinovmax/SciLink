"""File preview component for the File Explorer page."""

from pathlib import Path

import streamlit as st


def render_file_preview(file_path: Path) -> None:
    """Display an appropriate preview for the given file and a download button."""
    suffix = file_path.suffix.lower()

    st.subheader(file_path.name)

    # Download button
    with open(file_path, "rb") as fh:
        st.download_button(
            "Download",
            data=fh.read(),
            file_name=file_path.name,
            key=f"dl_{file_path}",
        )

    # Image
    if suffix in (".png", ".jpg", ".jpeg"):
        st.image(str(file_path))
        return

    if suffix in (".tif", ".tiff"):
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(file_path)
            arr = np.array(img, dtype=np.float64)
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            st.image(arr, clamp=True)
        except ImportError:
            st.info("Install Pillow to preview TIFF files.")
        except Exception as exc:
            st.error(f"Could not preview TIFF file: {exc}")
        return

    # JSON
    if suffix == ".json":
        import json
        try:
            data = json.loads(file_path.read_text())
            st.json(data)
        except Exception:
            st.code(file_path.read_text(), language="json")
        return

    # Tabular
    if suffix in (".csv", ".tsv"):
        try:
            import pandas as pd
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(file_path, sep=sep)
            st.dataframe(df)
        except Exception:
            st.code(file_path.read_text()[:100000], language="text")
        return

    # Excel
    if suffix in (".xlsx", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(file_path)
            if len(df) > 100:
                st.caption(f"Showing first 100 of {len(df):,} rows")
                st.dataframe(df.head(100))
            else:
                st.dataframe(df)
        except Exception:
            st.info("Install `openpyxl` to preview Excel files.")
        return

    # NumPy
    if suffix == ".npy":
        try:
            import numpy as np
            arr = np.load(file_path)
            st.markdown(f"**shape:** `{arr.shape}` &nbsp; **dtype:** `{arr.dtype}`")
            if arr.ndim <= 2 and max(arr.shape) <= 2048:
                st.image(arr, clamp=True)
            else:
                st.text(f"min={arr.min():.4g}  max={arr.max():.4g}  mean={arr.mean():.4g}")
        except Exception as exc:
            st.error(f"Could not load .npy file: {exc}")
        return

    # HTML reports — render as interactive HTML
    if suffix == ".html":
        st.iframe(
            file_path.read_text(encoding="utf-8"),
            height=600,
        )
        return

    # Source code
    _code_langs = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".r": "r",
        ".jl": "julia",
    }
    if suffix in _code_langs:
        st.code(file_path.read_text()[:100000], language=_code_langs[suffix])
        return

    # Plain text fallback
    if suffix in (".txt", ".md", ".log"):
        st.code(file_path.read_text()[:100000], language="text")
        return

    # Extension-less plain-text scientific files (VASP DFT inputs, LAMMPS data,
    # etc). Detected by exact filename match. Render as plain text.
    _PLAIN_TEXT_NAMES = {
        "POSCAR", "CONTCAR", "INCAR", "KPOINTS", "POTCAR",
        "OUTCAR", "OSZICAR", "DOSCAR", "EIGENVAL", "PROCAR",
        "vasprun.xml", "XDATCAR", "CHGCAR", "WAVECAR",
    }
    if file_path.name in _PLAIN_TEXT_NAMES:
        # Show small files in full; clip large binary-ish ones to a head preview.
        try:
            content = file_path.read_text(errors="replace")
            lang = "xml" if file_path.name == "vasprun.xml" else "text"
            st.code(content[:100000], language=lang)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
        return

    st.info(f"No preview available for `{suffix}` files.")
