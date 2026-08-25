"""Pre-chat upload zones dispatched by app_mode."""

from pathlib import Path

import streamlit as st

from ..config import (
    SUPPORTED_CODE_EXTENSIONS,
    SUPPORTED_DATA_EXTENSIONS,
    SUPPORTED_KNOWLEDGE_EXTENSIONS,
    SUPPORTED_META_EXTENSIONS,
    SUPPORTED_METADATA_EXTENSIONS,
    SUPPORTED_PLANNING_DATA_EXTENSIONS,
    extra_data_extensions_for,
)
from .sidebar import save_metadata_to_series, save_upload, save_upload_batch


def render_pre_chat_uploads(start_task_fn) -> None:
    """Show the appropriate upload zone based on the current app mode.

    *start_task_fn* is called with the initial prompt string when the user
    clicks the action button (Analyze / Start Planning).
    """
    mode = st.session_state.app_mode
    if mode == "analyze":
        _render_analyze_uploads(start_task_fn)
    elif mode == "plan":
        _render_planning_uploads(start_task_fn)
    elif mode == "meta":
        _render_meta_uploads(start_task_fn)


# ── Analyze mode uploads ─────────────────────────────────────────

def _render_analyze_uploads(start_task_fn) -> None:
    st.markdown(
        '<div class="upload-hero-box">'
        '<p class="upload-hero-title">'
        "Upload your data to get started</p>"
        '<p class="upload-hero-subtitle">'
        "Images, CSV, NumPy arrays, and more</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    extra_exts = extra_data_extensions_for(st.session_state.get("agent"))
    data_exts = SUPPORTED_DATA_EXTENSIONS + extra_exts

    up_data, up_meta = st.columns(2)
    with up_data:
        main_data = st.file_uploader(
            "Data file(s)",
            type=[e.lstrip(".") for e in data_exts],
            key="main_uploader_data",
            accept_multiple_files=True,
        )
        if extra_exts:
            st.caption("Vendor formats enabled via SciFiReaders MCP")
        if main_data:
            if len(main_data) == 1:
                save_upload(main_data[0], "data", auto_dispatch=False)
            else:
                save_upload_batch(main_data, "data", auto_dispatch=False)
    with up_meta:
        main_meta = st.file_uploader(
            "Metadata (optional)",
            type=[e.lstrip(".") for e in SUPPORTED_METADATA_EXTENSIONS],
            key="main_uploader_meta",
            accept_multiple_files=True,
        )
        if main_meta:
            if len(main_meta) == 1:
                save_upload(main_meta[0], "metadata", auto_dispatch=False)
            else:
                save_metadata_to_series(main_meta, auto_dispatch=False)

    # Show "Analyze" button once data is uploaded
    if st.session_state.uploaded_data_path:
        if st.button("Analyze", type="primary", width="stretch"):
            data_path = st.session_state.uploaded_data_path
            meta_path = st.session_state.uploaded_metadata_path
            has_sidecars = st.session_state.get("uploaded_sidecar_metadata", False)
            if data_path and meta_path:
                prompt = (
                    f"I uploaded a data file at `{data_path}` "
                    f"and a metadata file at `{meta_path}`. "
                    f"Please examine the data and load the metadata."
                )
            elif data_path and has_sidecars:
                prompt = (
                    f"I uploaded data files at `{data_path}` along with "
                    f"per-file JSON sidecar metadata in the same directory. "
                    f"Please examine the data and load the metadata "
                    f"(pass the directory path `{data_path}` to load_metadata)."
                )
            else:
                prompt = (
                    f"I uploaded a data file at `{data_path}`. "
                    f"Please examine it."
                )
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            start_task_fn(prompt)


# ── Plan mode uploads ────────────────────────────────────────────

def _render_planning_uploads(start_task_fn) -> None:
    st.text_area(
        "Research objective",
        key="planning_objective",
        placeholder="e.g., Optimize reaction yield for polymer synthesis",
        height=80,
    )

    st.markdown(
        '<div class="upload-hero-box">'
        '<p class="upload-hero-title">'
        "Upload resources for the planning agent</p>"
        '<p class="upload-hero-subtitle">'
        "Papers, images, code, and experimental data</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Knowledge (papers, images)", expanded=True):
        knowledge_files = st.file_uploader(
            "Upload knowledge files",
            type=[e.lstrip(".") for e in SUPPORTED_KNOWLEDGE_EXTENSIONS],
            key="main_uploader_knowledge",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if knowledge_files:
            save_planning_uploads(knowledge_files, "knowledge")
        knowledge_folder = st.text_input(
            "or paste folder path",
            key="knowledge_folder_path",
            placeholder="/path/to/papers/ or /path/to/database/",
            label_visibility="collapsed",
        )

    with st.expander("Code (scripts, API docs)"):
        code_files = st.file_uploader(
            "Upload code files",
            type=[e.lstrip(".") for e in SUPPORTED_CODE_EXTENSIONS],
            key="main_uploader_code",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if code_files:
            save_planning_uploads(code_files, "code")
        code_folder = st.text_input(
            "or paste folder path",
            key="code_folder_path",
            placeholder="/path/to/code/",
            label_visibility="collapsed",
        )

    with st.expander("Data (experimental results)"):
        data_files = st.file_uploader(
            "Upload data files",
            type=[e.lstrip(".") for e in SUPPORTED_PLANNING_DATA_EXTENSIONS],
            key="main_uploader_planning_data",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if data_files:
            save_planning_uploads(data_files, "data")
        data_folder = st.text_input(
            "or paste folder path",
            key="data_folder_path",
            placeholder="/path/to/data/",
            label_visibility="collapsed",
        )

    # Collect valid folder paths
    _folders = {}
    for label, val in [("knowledge", knowledge_folder), ("code", code_folder), ("data", data_folder)]:
        if val and Path(val.strip()).is_dir():
            _folders[label] = val.strip()

    # Show "Start Planning" button
    objective = st.session_state.get("planning_objective", "").strip()
    has_uploads = (
        st.session_state.uploaded_knowledge_paths
        or st.session_state.uploaded_code_paths
        or st.session_state.uploaded_planning_data_paths
    )
    can_start = bool(objective) or has_uploads or bool(_folders)
    if st.button(
        "Start Planning",
        type="primary",
        width="stretch",
        disabled=not can_start,
    ):
        parts = []
        if objective:
            parts.append(f"Research objective: {objective}")
        if st.session_state.uploaded_knowledge_paths:
            paths = ", ".join(f"`{p}`" for p in st.session_state.uploaded_knowledge_paths)
            parts.append(f"Knowledge files: {paths}")
        if _folders.get("knowledge"):
            parts.append(f"Knowledge folder: `{_folders['knowledge']}`")
        if st.session_state.uploaded_code_paths:
            paths = ", ".join(f"`{p}`" for p in st.session_state.uploaded_code_paths)
            parts.append(f"Code files: {paths}")
        if _folders.get("code"):
            parts.append(f"Code folder: `{_folders['code']}`")
        if st.session_state.uploaded_planning_data_paths:
            data_paths = [p for p in st.session_state.uploaded_planning_data_paths if not p.endswith(".json")]
            json_paths = [p for p in st.session_state.uploaded_planning_data_paths if p.endswith(".json")]
            if data_paths:
                paths_str = ", ".join(f"`{p}`" for p in data_paths)
                if len(data_paths) > 1 and json_paths:
                    # Multiple data files + JSON metadata → suggest analyze_batch
                    json_str = ", ".join(f"`{p}`" for p in json_paths)
                    parts.append(f"Data files: {paths_str}")
                    parts.append(f"Conditions/metadata JSON: {json_str}")
                    parts.append(
                        "Use `analyze_batch` to process these files together, "
                        "using the JSON as the conditions source."
                    )
                elif len(data_paths) > 1:
                    parts.append(f"Data files: {paths_str}")
                    parts.append(
                        "Use `analyze_batch` to process these files together. "
                        "If these are measurement-only files (e.g., spectra), "
                        "you will need experimental conditions for each file."
                    )
                else:
                    parts.append(f"Data files: {paths_str}")
            if json_paths and not data_paths:
                paths_str = ", ".join(f"`{p}`" for p in json_paths)
                parts.append(f"Data/metadata files: {paths_str}")
        if _folders.get("data"):
            data_dir = Path(_folders["data"])
            data_exts = {".csv", ".xlsx", ".tsv", ".txt"}
            dir_data_files = sorted(
                [f for f in data_dir.iterdir() if f.suffix.lower() in data_exts],
                key=lambda f: [int(c) if c.isdigit() else c.lower()
                               for c in __import__("re").split(r"(\d+)", f.name)]
            )
            dir_json_files = sorted(
                [f for f in data_dir.iterdir() if f.suffix.lower() == ".json"],
                key=lambda f: f.name
            )
            if len(dir_data_files) > 1:
                paths_str = ", ".join(f"`{f}`" for f in dir_data_files)
                parts.append(f"Data files: {paths_str}")
                if dir_json_files:
                    json_str = ", ".join(f"`{f}`" for f in dir_json_files)
                    parts.append(f"Conditions/metadata JSON: {json_str}")
                    parts.append(
                        "Use `analyze_batch` to process these files together, "
                        "using the JSON as the conditions source."
                    )
                else:
                    parts.append(
                        "Use `analyze_batch` to process these files together. "
                        "If these are measurement-only files (e.g., spectra), "
                        "you will need experimental conditions for each file."
                    )
            elif len(dir_data_files) == 1:
                parts.append(f"Data file: `{dir_data_files[0]}`")
            else:
                parts.append(f"Data folder: `{_folders['data']}`")
        prompt = "\n\n".join(parts) if parts else "Please help me plan my experiment."

        # Point the agent's dirs to the original folder paths so that
        # source_difference() in the KB sees stable paths across sessions.
        # This allows the FAISS indexes in kb_storage/ to be reused instead
        # of rebuilding every time a new session dir is created.
        agent = st.session_state.agent
        if _folders.get("knowledge"):
            agent.knowledge_dir = Path(_folders["knowledge"])
        if _folders.get("code"):
            agent.code_dir = Path(_folders["code"])
        if _folders.get("data"):
            agent.data_dir = Path(_folders["data"])

        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        start_task_fn(prompt)

    if not can_start:
        st.caption("Enter a research objective or upload files to begin.")


# ── Meta mode (free-text goal + one combined uploader) ───────────

def _render_meta_uploads(start_task_fn) -> None:
    st.markdown(
        '<div class="upload-hero-box">'
        '<p class="upload-hero-title">What would you like to do? '
        '<span style="font-size:0.45em;vertical-align:middle;'
        'background:#8893a5;color:#fff;padding:2px 8px;border-radius:9px;'
        'letter-spacing:1px;font-weight:600;">BETA</span></p>'
        '<p class="upload-hero-subtitle">'
        "Describe your research goal — mission control agent will route "
        "it to the specialist agents</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    goal = st.text_area(
        "Research goal",
        key="meta_objective",
        placeholder=(
            "e.g., Analyze the STEM image I uploaded, then plan a "
            "follow-up experiment campaign based on what you find"
        ),
        height=110,
        label_visibility="collapsed",
    )

    with st.expander("Add files (optional) — papers, code, data, metadata",
                     expanded=True):
        extra_exts = extra_data_extensions_for(st.session_state.get("agent"))
        meta_exts = tuple(SUPPORTED_META_EXTENSIONS) + tuple(extra_exts)
        files = st.file_uploader(
            "Upload files",
            type=[e.lstrip(".") for e in meta_exts],
            key="main_uploader_auto",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if files:
            save_meta_uploads(files)
        folder = st.text_input(
            "or paste folder path(s)",
            key="meta_folder_path",
            placeholder="/path/to/data/    (separate multiple with ',')",
            label_visibility="collapsed",
        )
        st.caption(
            "One drop zone for everything — the meta-agent sorts each file "
            "to the analysis or planning specialist."
        )

    # Multi-folder input via comma separation — paths with a literal ',' in
    # their name are rare; users with those can place them under a common
    # parent and paste that instead.
    candidates = [p.strip() for p in (folder or "").split(",") if p.strip()]
    valid_folders = [p for p in candidates if Path(p).is_dir()]
    missing = [p for p in candidates if not Path(p).is_dir()]
    for m in missing:
        st.warning(f"Folder not found: {m}")
    folder_ok = bool(valid_folders)

    uploads = st.session_state.uploaded_meta_paths
    goal = (goal or "").strip()
    can_start = bool(goal) or bool(uploads) or folder_ok
    if st.button(
        "Start",
        type="primary",
        width="stretch",
        disabled=not can_start,
    ):
        parts = []
        if goal:
            parts.append(goal)
        if uploads:
            listed = "\n".join(f"  - `{p}`" for p in uploads)
            parts.append(
                f"I uploaded {len(uploads)} file(s):\n{listed}\n\n"
                "Inspect them to determine what each file is, then route them "
                "to the right specialist."
            )
        if valid_folders:
            if len(valid_folders) == 1:
                parts.append(
                    f"Additional resources are in the folder "
                    f"`{valid_folders[0]}` — inspect it as well."
                )
            else:
                listed = "\n".join(f"  - `{p}`" for p in valid_folders)
                parts.append(
                    f"Additional resources are in {len(valid_folders)} "
                    f"folders — inspect them as well:\n{listed}"
                )
        prompt = "\n\n".join(parts) if parts else "Please help with my research."
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        start_task_fn(prompt)

    if not can_start:
        st.caption("Describe a research goal or add files to begin.")


def save_planning_uploads(files, category: str) -> None:
    """Save uploaded files to the appropriate planning subdirectory."""
    session_dir = Path(st.session_state.session_dir)
    target_dir = session_dir / category
    target_dir.mkdir(parents=True, exist_ok=True)

    state_key = {
        "knowledge": "uploaded_knowledge_paths",
        "code": "uploaded_code_paths",
        "data": "uploaded_planning_data_paths",
    }[category]

    for f in files:
        dest = target_dir / f.name
        upload_key = (category, str(dest))
        if upload_key in st.session_state._processed_uploads:
            continue
        dest.write_bytes(f.getvalue())
        st.session_state._processed_uploads.add(upload_key)
        st.session_state[state_key].append(str(dest))
        st.sidebar.success(f"Uploaded {category}: {f.name}")


def save_meta_uploads(files) -> None:
    """Save Explore-mode uploads into the meta session's ``uploads/`` directory.

    Everything lands in one place; the meta-agent decides which child
    specialist each file belongs to when it builds its delegations.
    """
    session_dir = Path(st.session_state.session_dir)
    target_dir = session_dir / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = target_dir / f.name
        upload_key = ("meta", str(dest))
        if upload_key in st.session_state._processed_uploads:
            continue
        dest.write_bytes(f.getvalue())
        st.session_state._processed_uploads.add(upload_key)
        st.session_state.uploaded_meta_paths.append(str(dest))
        st.sidebar.success(f"Uploaded: {f.name}")
