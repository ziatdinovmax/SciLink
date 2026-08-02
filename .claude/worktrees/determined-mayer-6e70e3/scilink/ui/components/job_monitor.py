"""
Shared job monitoring components for the simulation UI.

Provides reusable widgets for job selection, live monitoring, and
output file browsing. Used by both the general Monitor tab
(``simulations.py``) and the LAMMPS workflow (``sim_workflow.py``).
"""

import os
import re
import stat as stat_mod
from pathlib import PurePosixPath
from typing import Callable, Dict, Optional, Set, Tuple

import streamlit as st

from scilink.hpc.scheduler import HPCJob, JobStatus, Scheduler


# ══════════════════════════════════════════════════════════════
# Stage parsing (generic + LAMMPS-specific)
# ══════════════════════════════════════════════════════════════

# LAMMPS orchestrator stages (in order)
_LAMMPS_STAGES = [
    ("complete", re.compile(r"Simulation complete|All stages finished", re.I)),
    ("analysis", re.compile(r"Stage:\s*Analysis|Running analysis", re.I)),
    ("production", re.compile(r"Stage:\s*Production|Starting production", re.I)),
    ("equilibration", re.compile(r"Stage:\s*Equilibration|Starting equilibration", re.I)),
    ("minimization", re.compile(r"Stage:\s*Minimization|Starting minimization", re.I)),
]

_LAMMPS_PROGRESS = {
    "pending": 0.0,
    "starting": 0.02,
    "minimization": 0.15,
    "equilibration": 0.40,
    "production": 0.70,
    "analysis": 0.90,
    "complete": 1.0,
}


def parse_lammps_stage(log_text: str) -> Tuple[str, float]:
    """Parse LAMMPS orchestrator log for current stage and progress."""
    if not log_text:
        return "Pending", 0.0
    for stage_name, pattern in _LAMMPS_STAGES:
        if pattern.search(log_text):
            return stage_name.title(), _LAMMPS_PROGRESS.get(stage_name, 0.5)
    return "Starting", 0.02


def parse_generic_stage(job: HPCJob) -> Tuple[str, float]:
    """Derive stage/progress from job status alone (no log parsing)."""
    mapping = {
        JobStatus.PENDING: ("Queued", 0.0),
        JobStatus.RUNNING: ("Running", 0.5),
        JobStatus.COMPLETED: ("Completed", 1.0),
        JobStatus.FAILED: ("Failed", 1.0),
        JobStatus.CANCELLED: ("Cancelled", 1.0),
        JobStatus.TIMEOUT: ("Timed out", 1.0),
    }
    return mapping.get(job.status, ("Unknown", 0.0))


def count_errors_warnings(log_text: str) -> Tuple[int, int]:
    """Count error and warning lines in log output."""
    errors = len(re.findall(r"^.*(?:ERROR|FAILED|FATAL).*$", log_text, re.M | re.I))
    warnings = len(re.findall(r"^.*(?:WARNING|WARN).*$", log_text, re.M | re.I))
    return errors, warnings


# ══════════════════════════════════════════════════════════════
# Job selector widget
# ══════════════════════════════════════════════════════════════

def render_job_selector(
    tracked_jobs: Dict[str, HPCJob],
    key_prefix: str = "hpc",
) -> Optional[str]:
    """
    Dropdown to pick a tracked job. Returns the selected job_id or None.

    Updates ``st.session_state.hpc_monitored_job_id`` on selection change.
    """
    if not tracked_jobs:
        return None

    jids = list(tracked_jobs.keys())
    labels = [
        f"{jid} — {tracked_jobs[jid].name} "
        f"({tracked_jobs[jid].status.emoji} {tracked_jobs[jid].status.value})"
        for jid in jids
    ]
    cur_mon = st.session_state.get("hpc_monitored_job_id")
    sel_idx = jids.index(cur_mon) if cur_mon in jids else 0

    chosen = st.selectbox(
        "Select job",
        range(len(labels)),
        index=sel_idx,
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_mon_sel",
    )
    job_id = jids[chosen]
    st.session_state.hpc_monitored_job_id = job_id
    return job_id


# ══════════════════════════════════════════════════════════════
# Live monitoring fragment
# ══════════════════════════════════════════════════════════════

def render_job_monitor(
    job_id: str,
    poll_interval: str = "5s",
    show_output_files: bool = False,
    show_details_tab: bool = True,
    stage_parser: Optional[Callable[[str], Tuple[str, float]]] = None,
    key_prefix: str = "jmon",
) -> None:
    """
    Render a live-updating job monitor fragment.

    This is the core monitoring widget shared by both the workflow
    monitor and the general Monitor tab. It creates a ``st.fragment``
    that polls the scheduler at ``poll_interval``.

    Args:
        job_id: Job ID to monitor.
        poll_interval: Streamlit fragment polling interval (e.g. "3s", "5s").
            Set to None for terminal jobs (no polling).
        show_output_files: Whether to render the output files tab.
        show_details_tab: Whether to show the job details tab.
        stage_parser: Optional callable(log_text) → (stage_name, progress_float).
            If None, stage/progress derived from job status only.
        key_prefix: Unique prefix for widget keys (avoids collisions when
            multiple monitors are rendered on the same page).
    """
    tracked = st.session_state.get("hpc_tracked_jobs", {})
    if job_id not in tracked:
        st.warning("Job not found in tracked jobs.")
        return

    job = tracked[job_id]
    is_terminal = job.status.is_terminal if hasattr(job.status, "is_terminal") else False
    effective_interval = None if is_terminal else poll_interval

    @st.fragment(run_every=effective_interval)
    def _monitor_fragment():
        _sched: Optional[Scheduler] = st.session_state.get("hpc_scheduler")
        _tracked: dict = st.session_state.get("hpc_tracked_jobs", {})
        _jid = job_id  # captured from closure

        if not _sched or _jid not in _tracked:
            return

        _job = _tracked[_jid]

        # ── Refresh status ────────────────────────────────
        try:
            fresh = _sched.status(_jid)
            old_status = _job.status
            # Preserve metadata that the scheduler query might not return
            fresh.stdout_file = fresh.stdout_file or _job.stdout_file
            fresh.stderr_file = fresh.stderr_file or _job.stderr_file
            fresh.work_dir = fresh.work_dir or _job.work_dir
            fresh.script_path = fresh.script_path or getattr(_job, "script_path", None)
            _tracked[_jid] = fresh
            st.session_state.hpc_tracked_jobs = _tracked
            _job = fresh
        except Exception:
            old_status = _job.status

        # ── Read stdout / stderr ──────────────────────────
        stdout_text = ""
        stderr_text = ""
        try:
            stdout_text = _sched.tail_output(_job, "stdout", lines=500)
        except Exception:
            pass
        try:
            stderr_text = _sched.tail_output(_job, "stderr", lines=100)
        except Exception:
            pass

        # ── Stage / progress ──────────────────────────────
        if stage_parser is not None:
            stage, progress = stage_parser(stdout_text)
        else:
            stage, progress = parse_generic_stage(_job)

        n_errors, n_warnings = count_errors_warnings(stdout_text)

        # ── Metrics row ───────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Status", f"{_job.status.emoji} {_job.status.value}")
        with m2:
            st.metric("Stage", stage)
        with m3:
            st.metric("Runtime", _job.time_used or "—")
        with m4:
            if n_errors:
                st.metric("Issues", f"🔴 {n_errors} errors")
            elif n_warnings:
                st.metric("Issues", f"🟡 {n_warnings} warnings")
            else:
                st.metric("Issues", "✅ None")

        # ── Progress bar ──────────────────────────────────
        if stage_parser is not None:
            st.progress(progress, text=f"{stage} — {progress:.0%}")

        # ── Cancel button ─────────────────────────────────
        if not _job.status.is_terminal:
            if st.button("🛑 Cancel job", key=f"{key_prefix}_cancel"):
                try:
                    _sched.cancel(_jid)
                    st.toast("Cancel signal sent.")
                except Exception as e:
                    st.error(str(e))

        # ── Tabs: stdout / stderr / (output files) / (details) ──
        tab_names = ["stdout", "stderr"]
        if show_output_files:
            tab_names.append("Output Files")
        if show_details_tab:
            tab_names.append("Details")

        tabs = st.tabs(tab_names)
        tab_idx = 0

        # stdout tab
        with tabs[tab_idx]:
            if stdout_text.strip():
                _render_scrolling_log(stdout_text[-8000:], key=f"{key_prefix}_stdout")
            else:
                st.caption("(no output yet — job may be queued)")
        tab_idx += 1

        # stderr tab
        with tabs[tab_idx]:
            if stderr_text.strip():
                st.code(stderr_text[-3000:], language="text")
            else:
                st.caption("(no stderr)")
        tab_idx += 1

        # Output files tab (optional)
        if show_output_files:
            with tabs[tab_idx]:
                conn = st.session_state.get("hpc_connection")
                if conn and conn.is_connected:
                    render_remote_output_files(conn, _job)
                else:
                    st.caption("HPC connection required to browse output files.")
            tab_idx += 1

        # Details tab (optional)
        if show_details_tab:
            with tabs[tab_idx]:
                details = {
                    "Job ID": _job.job_id,
                    "Name": _job.name,
                    "Partition": _job.partition,
                    "Nodes": _job.node_list or str(_job.nodes),
                    "Tasks": str(_job.ntasks),
                    "Submit time": _job.submit_time,
                    "Start time": _job.start_time,
                    "End time": _job.end_time,
                    "Time limit": _job.time_limit,
                    "Work dir": _job.work_dir,
                    "stdout file": _job.stdout_file,
                    "stderr file": _job.stderr_file,
                    "Exit code": (
                        str(_job.exit_code)
                        if _job.exit_code is not None
                        else "—"
                    ),
                }
                for k, v in details.items():
                    if v:
                        st.text(f"{k:>14}: {v}")

        # ── Trigger full rerun when job transitions to terminal ──
        if _job.status.is_terminal and not old_status.is_terminal:
            st.rerun(scope="app")

    _monitor_fragment()


# ══════════════════════════════════════════════════════════════
# Output file browser
# ══════════════════════════════════════════════════════════════

_REMOTE_ICONS = {
    ".py": "🐍", ".sh": "⚙️", ".data": "🔬", ".lmp": "🔬",
    ".pdb": "🧬", ".cif": "🧬", ".gro": "🧬",
    ".csv": "📊", ".json": "📋", ".xml": "📋",
    ".txt": "📝", ".log": "📝", ".out": "📝",
    ".prm": "⚛️", ".top": "⚛️", ".itp": "⚛️",
    ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼",
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def render_remote_output_files(conn, job: HPCJob) -> None:
    """List and display output files from the job's remote working directory."""
    if not job.work_dir:
        st.caption("No working directory known.")
        return

    downloaded_images: Dict[str, bytes] = st.session_state.get(
        "hpc_mon_downloaded_images", {},
    )

    try:
        entries = conn.listdir(job.work_dir)
    except Exception as exc:
        st.caption(f"Cannot list remote dir: {exc}")
        return

    files = [
        e for e in entries
        if not (stat_mod.S_ISDIR(e.st_mode) if e.st_mode else False)
        and not e.filename.startswith(".")
    ]
    files.sort(key=lambda e: e.filename)

    if not files:
        st.caption("(no output files yet)")
        return

    # Discover new image files and download them
    for entry in files:
        ext = PurePosixPath(entry.filename).suffix.lower()
        full = f"{job.work_dir}/{entry.filename}"
        if ext in _IMAGE_EXTS and full not in downloaded_images:
            try:
                data = conn.read_bytes(full)
                downloaded_images[full] = data
                st.session_state.hpc_mon_downloaded_images = downloaded_images
            except Exception:
                pass

    # Show images
    new_images = {
        k: v for k, v in downloaded_images.items()
        if PurePosixPath(k).suffix.lower() in _IMAGE_EXTS
    }
    if new_images:
        st.markdown("**Generated plots:**")
        for path, data in new_images.items():
            st.image(data, caption=PurePosixPath(path).name)

    # File listing with download
    st.markdown("**All files:**")
    for entry in files:
        name = entry.filename
        size = fmt_size(entry.st_size) if entry.st_size else ""
        icon = _REMOTE_ICONS.get(PurePosixPath(name).suffix.lower(), "📄")
        full = f"{job.work_dir}/{name}"

        c_n, c_s, c_dl = st.columns([4, 1.5, 1])
        with c_n:
            st.caption(f"{icon} {name}")
        with c_s:
            st.caption(size)
        with c_dl:
            if full in downloaded_images:
                st.download_button(
                    "⬇", data=downloaded_images[full],
                    file_name=name,
                    key=f"jmon_dl_{name}",
                )
            else:
                if st.button("⬇", key=f"jmon_fetch_{name}"):
                    try:
                        data = conn.read_bytes(full)
                        downloaded_images[full] = data
                        st.session_state.hpc_mon_downloaded_images = downloaded_images
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))


# ══════════════════════════════════════════════════════════════
# Agent job bridging
# ══════════════════════════════════════════════════════════════

def merge_agent_tracked_jobs() -> Dict[str, HPCJob]:
    """
    Merge manually tracked jobs with jobs submitted by the
    SimulationOrchestratorAgent, returning a unified dict.

    The agent stores job IDs on its ``generated_structures`` records.
    This function creates HPCJob stubs for any agent-tracked jobs not
    already present in ``hpc_tracked_jobs``, then refreshes their status
    from the scheduler if available.
    """
    tracked: Dict[str, HPCJob] = dict(
        st.session_state.get("hpc_tracked_jobs", {}),
    )
    sched: Optional[Scheduler] = st.session_state.get("hpc_scheduler")

    # Check for agent-tracked jobs
    agent = st.session_state.get("agent")
    structures = getattr(agent, "generated_structures", None)
    if structures is None:
        # Also check hpc_sim_agent (the simulation agent may be stored separately)
        sim_agent = st.session_state.get("hpc_sim_agent")
        structures = getattr(sim_agent, "generated_structures", None)

    if structures:
        for slug, record in structures.items():
            job_id = record.get("hpc_job_id")
            if not job_id or job_id in tracked:
                continue

            # Create a stub HPCJob from the agent's record
            job = HPCJob(
                job_id=job_id,
                name=f"vasp_{slug}",
                status=JobStatus.UNKNOWN,
                work_dir=record.get("hpc_remote_dir", ""),
            )

            # Try to hydrate from scheduler
            if sched:
                try:
                    job = sched.status(job_id)
                except Exception:
                    pass

            tracked[job_id] = job

    st.session_state.hpc_tracked_jobs = tracked
    return tracked


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def fmt_size(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _render_scrolling_log(text: str, key: str = "so") -> None:
    """Render a log block with auto-scroll-to-bottom inside an iframe."""
    import html as _html

    escaped = _html.escape(text)
    st.iframe(
        f'<pre id="{key}" style="height:320px;overflow-y:auto;'
        f"margin:0;background:#0e1117;padding:10px;"
        f"border-radius:6px;border:1px solid #333;"
        f"font-family:monospace;font-size:13px;"
        f'white-space:pre-wrap;color:#e0e0e0">'
        f"{escaped}</pre>"
        f'<script>var e=document.getElementById("{key}");'
        f"e.scrollTop=e.scrollHeight;</script>",
        height=340,
    )
