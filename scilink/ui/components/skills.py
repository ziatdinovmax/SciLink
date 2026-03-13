"""Skills tab — upload custom skills and view available built-in skills."""

from pathlib import Path

import streamlit as st

from scilink.skills.loader import list_all_skills


def render_skills_tab() -> None:
    """Render the Skills tab content."""
    agent = st.session_state.get("agent")

    if agent is None:
        st.info("Start a session to view skills.")
        return

    left_col, _, right_col = st.columns([10, 1, 10])

    with left_col:
        _render_upload_section(agent)

    with right_col:
        _render_available_skills(agent)


def _render_upload_section(agent) -> None:
    """Upload custom skill files."""
    st.subheader("Upload Skills")
    uploaded = st.file_uploader(
        "Upload a custom skill file (.md)",
        type=["md"],
        key="skill_file_uploader",
        accept_multiple_files=True,
        help=(
            "Markdown file with structured sections (## Overview, ## Planning, "
            "## Analysis, ## Interpretation, ## Validation) providing "
            "domain-specific guidance for analysis agents."
        ),
    )

    if uploaded:
        for f in uploaded:
            upload_key = ("custom_skill", f.name)
            if upload_key in st.session_state._processed_uploads:
                continue
            _load_skill_file(agent, f)
            st.session_state._processed_uploads.add(upload_key)


def _load_skill_file(agent, uploaded_file) -> None:
    """Save an uploaded skill .md file and register it with the agent."""
    session_dir = st.session_state.get("session_dir")
    if session_dir is None:
        st.error("No active session.")
        return

    skills_dir = Path(session_dir) / "custom_skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    dest = skills_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())

    try:
        name = agent.register_skill(str(dest))
        st.success(f"Registered skill '{name}' from {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to register {uploaded_file.name}: {e}")


def _render_available_skills(agent) -> None:
    """Show built-in and custom skills."""
    st.subheader("Available Skills")

    # Built-in subsection
    st.markdown("**Built-in**")
    builtin = list_all_skills()
    if builtin:
        for domain, names in builtin.items():
            label = domain.replace("_", " ").title()
            with st.expander(f"{label} ({len(names)})", expanded=False):
                for name in names:
                    st.markdown(f"- `{name}`")
    else:
        st.caption("No built-in skills found.")

    # Custom subsection
    st.markdown("**Custom**")
    custom = getattr(agent, "_custom_skills", {})
    if custom:
        for name in sorted(custom.keys()):
            st.markdown(f"- `{name}`")
    else:
        st.caption("No custom skills registered yet.")
