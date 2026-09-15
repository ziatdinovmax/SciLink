"""Streamlit component — Simulations tab (HPC integration)."""
from __future__ import annotations

import os
import stat as stat_mod
import tempfile
from datetime import datetime
from pathlib import PurePosixPath
from typing import Optional

import streamlit as st

# Graceful fallback when paramiko is not installed
try:
    import paramiko as _paramiko
except ImportError:
    _paramiko = None  # type: ignore[assignment]

from scilink.hpc.connection import HPCConnection, HPCProfile
from scilink.hpc.scheduler import (
    HPCJob,
    JobStatus,
    Scheduler,
    detect_scheduler,
)

from scilink.ui.components.engines import get_engine, get_engines

import logging

# ── SLURM script templates ────────────────────────────────────

_TEMPLATES: dict[str, str] = {
    "Serial": """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={time_limit}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
# module load python/3.11

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

# ── your commands ──
# python run_simulation.py

echo "Job finished at $(date)"
""",
    "MPI parallel": """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --time={time_limit}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
# module load openmpi

echo "Job $SLURM_JOB_ID — $SLURM_NTASKS tasks across $SLURM_NNODES nodes"
srun ./my_simulation
""",
    "GPU": """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time_limit}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
# module load cuda/12

nvidia-smi
echo "Job $SLURM_JOB_ID started at $(date)"

# python train.py --gpus {gpus}
""",
    "Array job": """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --time={time_limit}
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

echo "Array task $SLURM_ARRAY_TASK_ID of job $SLURM_ARRAY_JOB_ID"
# python run_case.py --index $SLURM_ARRAY_TASK_ID
""",
    "Custom (blank)": """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}

""",
}


# ══════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════

def render_simulations_tab() -> None:
    if _paramiko is None:
        st.error(
            "**Paramiko** is required for HPC connectivity.  "
            "Install it with `pip install paramiko`."
        )
        return

    conn: Optional[HPCConnection] = st.session_state.get("hpc_connection")
    has_hpc = conn is not None and conn.is_connected

    if has_hpc:
        _render_connection_bar(conn)
        gen, dash, submit, monitor, terminal, files = st.tabs(
            ["🧪 Generate", "📊 Dashboard", "🚀 Submit Job",
             "📡 Monitor", "💻 Terminal", "📁 Remote Files"],
        )
    else:
        gen, dash = st.tabs(["🧪 Generate", "📊 Dashboard"])
        submit = monitor = terminal = files = None

    with gen:
        _render_engine_selector_and_dispatch()

    with dash:
        if has_hpc:
            _render_dashboard()
        else:
            st.info(
                "**No HPC connection.** Connect to a cluster via the "
                "sidebar to access the job dashboard, terminal, and "
                "remote file browser.\n\n"
                "You can still **generate simulation inputs** in the "
                "Generate tab and download them locally."
            )

    if submit is not None:
        with submit:
            _render_submit()
    if monitor is not None:
        with monitor:
            _render_monitor()
    if terminal is not None:
        with terminal:
            _render_terminal()
    if files is not None:
        with files:
            _render_remote_files()

# ── Engine selector ───────────────────────────────────────────

# Phase / artifact keys cleared when the user switches engines, to avoid
# bleeding LAMMPS state into a VASP run (and vice versa).
_PHASE_KEYS_TO_CLEAR_ON_ENGINE_CHANGE = (
    "hpc_workflow_phase",
    "hpc_gen_run_script",
    "hpc_gen_slurm_script",
    "hpc_gen_remote_work_dir",
    "hpc_gen_job_name",
    "hpc_gen_vasp_files",
)


def _render_engine_selector_and_dispatch() -> None:
    engines = get_engines()
    if not engines:
        st.error("No simulation engines registered.")
        return

    keys = [e.key for e in engines]
    current = st.session_state.get("hpc_engine", keys[0])
    if current not in keys:
        current = keys[0]

    sel_col, _ = st.columns([1, 4])
    with sel_col:
        chosen = st.selectbox(
            "Engine",
            keys,
            index=keys.index(current),
            format_func=lambda k: f"{get_engine(k).icon} {get_engine(k).label}",
            key="hpc_engine_select",
        )

    if chosen != current:
        for k in _PHASE_KEYS_TO_CLEAR_ON_ENGINE_CHANGE:
            st.session_state.pop(k, None)
        st.session_state.hpc_engine = chosen
        st.rerun()
    else:
        st.session_state.hpc_engine = chosen

    get_engine(chosen).render_workflow()


# ── Connection bar ────────────────────────────────────────────

def _render_connection_bar(conn: HPCConnection) -> None:
    sched: Optional[Scheduler] = st.session_state.get("hpc_scheduler")
    tracked: dict[str, HPCJob] = st.session_state.get("hpc_tracked_jobs", {})
    n_run = sum(1 for j in tracked.values() if j.status == JobStatus.RUNNING)
    n_pen = sum(1 for j in tracked.values() if j.status == JobStatus.PENDING)

    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
    with c1:
        st.markdown(f"🟢 **{conn.profile.username}@{conn.profile.hostname}**")
    with c2:
        st.caption(f"Scheduler: **{sched.name if sched else 'none detected'}**")
    with c3:
        st.caption(f"Tracked jobs: {n_run} running · {n_pen} pending")
    with c4:
        if st.button("Disconnect", key="hpc_disconnect"):
            conn.disconnect()
            st.session_state.hpc_connection = None
            st.session_state.hpc_scheduler = None
            st.rerun()


# ── Dashboard ─────────────────────────────────────────────────

def _render_dashboard() -> None:
    sched: Optional[Scheduler] = st.session_state.get("hpc_scheduler")
    if sched is None:
        st.warning(
            "No job scheduler detected on this system. "
            "You can still use the Terminal and Remote Files tabs."
        )
        return

    # ── Job queue ──
    st.subheader("Your Job Queue")
    c_ref, c_ts = st.columns([1, 4])
    with c_ref:
        do_refresh = st.button("🔄 Refresh", key="hpc_dash_refresh")
    with c_ts:
        ts = st.session_state.get("hpc_queue_time")
        if ts:
            st.caption(f"Last refreshed {ts:%H:%M:%S}")

    if do_refresh or "hpc_queue_cache" not in st.session_state:
        try:
            jobs = sched.queue()
            st.session_state.hpc_queue_cache = jobs
            st.session_state.hpc_queue_time = datetime.now()
            # Merge into tracked jobs
            tracked: dict[str, HPCJob] = st.session_state.get(
                "hpc_tracked_jobs", {},
            )
            for j in jobs:
                tracked[j.job_id] = j
            st.session_state.hpc_tracked_jobs = tracked
        except Exception as exc:
            st.error(f"Failed to query queue: {exc}")
            return

    jobs: list[HPCJob] = st.session_state.get("hpc_queue_cache", [])
    if not jobs:
        st.info("No jobs in queue.")
    else:
        import pandas as pd

        rows = [
            {
                "Status": f"{j.status.emoji} {j.status.value}",
                "ID": j.job_id,
                "Name": j.name,
                "Partition": j.partition,
                "Nodes": j.nodes,
                "Used": j.time_used,
                "Limit": j.time_limit,
            }
            for j in jobs
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── Partitions ──
    with st.expander("Cluster partitions"):
        try:
            parts = sched.partitions()
            if parts:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(parts), width="stretch", hide_index=True,
                )
            else:
                st.caption("No partition info available.")
        except Exception as exc:
            st.caption(f"Could not fetch partition info: {exc}")


# ── Submit ────────────────────────────────────────────────────

def _render_submit() -> None:
    sched: Optional[Scheduler] = st.session_state.get("hpc_scheduler")
    if sched is None:
        st.warning(
            "No scheduler detected — upload scripts and submit via Terminal instead."
        )
        return

    st.subheader("Submit a Job")

    c1, c2, c3 = st.columns(3)
    with c1:
        job_name = st.text_input("Job name", "scilink_sim", key="hpc_jname")
        partition = st.text_input("Partition", key="hpc_part")
    with c2:
        nodes = st.number_input("Nodes", min_value=1, value=1, key="hpc_nodes")
        tpn = st.number_input("Tasks / node", min_value=1, value=1, key="hpc_tpn")
    with c3:
        time_limit = st.text_input("Time limit", "01:00:00", key="hpc_tlim")
        gpus = st.number_input("GPUs / node", min_value=0, value=0, key="hpc_gpus")

    work_dir = st.text_input(
        "Remote working directory",
        value=st.session_state.get("hpc_remote_cwd", "~"),
        key="hpc_wdir",
    )

    # Template + editor
    tpl_name = st.selectbox("Template", list(_TEMPLATES.keys()), key="hpc_tpl")
    default_body = _TEMPLATES[tpl_name].format(
        job_name=job_name,
        partition=partition or "default",
        nodes=nodes,
        tasks_per_node=tpn,
        time_limit=time_limit,
        gpus=gpus,
    )
    # Keep user edits across reruns unless template changed
    if st.session_state.get("_hpc_last_tpl") != tpl_name:
        st.session_state["_hpc_script_body"] = default_body
        st.session_state["_hpc_last_tpl"] = tpl_name

    script = st.text_area(
        "Job script",
        value=st.session_state.get("_hpc_script_body", default_body),
        height=340,
        key="hpc_script_editor",
    )
    st.session_state["_hpc_script_body"] = script

    uploaded = st.file_uploader(
        "Extra input files to upload (optional)",
        accept_multiple_files=True,
        key="hpc_submit_uploads",
    )

    err_slot = st.empty()

    if st.button("🚀 Submit", type="primary", key="hpc_submit_btn"):
        conn: HPCConnection = st.session_state.hpc_connection
        try:
            with st.spinner("Uploading & submitting …"):
                conn.mkdir_p(work_dir)

                # Upload auxiliary files
                for uf in uploaded or []:
                    _upload_streamlit_file(conn, uf, f"{work_dir}/{uf.name}")

                # Upload the script itself
                script_name = f"{job_name}.sh"
                remote_script = f"{work_dir}/{script_name}"
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".sh", delete=False,
                ) as tmp:
                    tmp.write(script)
                    tmp_path = tmp.name
                try:
                    conn.upload(tmp_path, remote_script)
                finally:
                    os.unlink(tmp_path)

                job_id = sched.submit(remote_script, work_dir=work_dir)

            # Track
            tracked = st.session_state.get("hpc_tracked_jobs", {})
            tracked[job_id] = HPCJob(
                job_id=job_id,
                name=job_name,
                status=JobStatus.PENDING,
                work_dir=work_dir,
                script_path=remote_script,
                partition=partition,
            )
            st.session_state.hpc_tracked_jobs = tracked
            st.session_state.hpc_monitored_job_id = job_id
            st.success(f"✅ Submitted — job **{job_id}**")
            st.balloons()
        except Exception as exc:
            err_slot.error(f"Submission failed: {exc}")


# ── Monitor ───────────────────────────────────────────────────

def _render_monitor() -> None:
    sched: Optional[Scheduler] = st.session_state.get("hpc_scheduler")
    if sched is None:
        st.warning("No scheduler detected.")
        return

    tracked: dict[str, HPCJob] = st.session_state.get("hpc_tracked_jobs", {})

    # Allow manually adding a job ID
    if not tracked:
        st.info("No tracked jobs. Submit a job or enter an ID to start monitoring.")

    with st.expander("Track a job by ID", expanded=not tracked):
        with st.form("hpc_track_form", clear_on_submit=True):
            manual_id = st.text_input("Job ID", key="hpc_manual_jid")
            if st.form_submit_button("Track"):
                if manual_id.strip():
                    try:
                        job = sched.status(manual_id.strip())
                        tracked[job.job_id] = job
                        st.session_state.hpc_tracked_jobs = tracked
                        st.session_state.hpc_monitored_job_id = job.job_id
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    if not tracked:
        return

    # Job selector
    jids = list(tracked.keys())
    labels = [
        f"{jid} — {tracked[jid].name} "
        f"({tracked[jid].status.emoji} {tracked[jid].status.value})"
        for jid in jids
    ]
    cur_mon = st.session_state.get("hpc_monitored_job_id")
    sel_idx = jids.index(cur_mon) if cur_mon in jids else 0

    chosen = st.selectbox(
        "Select job",
        range(len(labels)),
        index=sel_idx,
        format_func=lambda i: labels[i],
        key="hpc_mon_sel",
    )
    job_id = jids[chosen]
    st.session_state.hpc_monitored_job_id = job_id
    job = tracked[job_id]

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Status", f"{job.status.emoji} {job.status.value}")
    with m2:
        st.metric("Runtime", job.time_used or "—")
    with m3:
        st.metric("Nodes", job.node_list or str(job.nodes))
    with m4:
        if not job.status.is_terminal:
            if st.button("🛑 Cancel job", key="hpc_cancel_job"):
                try:
                    sched.cancel(job_id)
                    st.toast("Cancel signal sent.")
                except Exception as exc:
                    st.error(str(exc))

    # ── live output fragment (polls every 3 s while job is active) ──
    _poll_interval = "3s" if not job.status.is_terminal else None

    @st.fragment(run_every=_poll_interval)
    def _live_output() -> None:
        _sched: Optional[Scheduler] = st.session_state.get("hpc_scheduler")
        _tracked: dict[str, HPCJob] = st.session_state.get("hpc_tracked_jobs", {})
        _jid = st.session_state.get("hpc_monitored_job_id")
        if not _sched or not _jid or _jid not in _tracked:
            return

        # Refresh status
        try:
            fresh = _sched.status(_jid)
            old_status = _tracked[_jid].status
            _tracked[_jid] = fresh
            st.session_state.hpc_tracked_jobs = _tracked
        except Exception:
            fresh = _tracked[_jid]
            old_status = fresh.status

        out_tab, err_tab, detail_tab = st.tabs(["stdout", "stderr", "Details"])
        with out_tab:
            try:
                stdout = _sched.tail_output(fresh, "stdout", lines=250)
                if stdout.strip():
                    # Auto-scrolling code block
                    import html as _html

                    escaped = _html.escape(stdout)
                    st.iframe(
                        f'<pre id="so" style="height:350px;overflow-y:auto;'
                        f"margin:0;background:#0e1117;padding:10px;"
                        f"border-radius:6px;border:1px solid #333;"
                        f"font-family:monospace;font-size:13px;"
                        f'white-space:pre-wrap;color:#e0e0e0">'
                        f"{escaped}</pre>"
                        f"<script>var e=document.getElementById('so');"
                        f"e.scrollTop=e.scrollHeight;</script>",
                        height=370,
                    )
                else:
                    st.caption("(no output yet)")
            except Exception as exc:
                st.caption(f"Cannot read stdout: {exc}")

        with err_tab:
            try:
                stderr = _sched.tail_output(fresh, "stderr", lines=100)
                if stderr.strip():
                    st.code(stderr, language="text")
                else:
                    st.caption("(no stderr)")
            except Exception as exc:
                st.caption(f"Cannot read stderr: {exc}")

        with detail_tab:
            details = {
                "Job ID": fresh.job_id,
                "Name": fresh.name,
                "Partition": fresh.partition,
                "Nodes": fresh.node_list or str(fresh.nodes),
                "Tasks": str(fresh.ntasks),
                "Submit time": fresh.submit_time,
                "Start time": fresh.start_time,
                "End time": fresh.end_time,
                "Time limit": fresh.time_limit,
                "Work dir": fresh.work_dir,
                "stdout file": fresh.stdout_file,
                "stderr file": fresh.stderr_file,
                "Exit code": str(fresh.exit_code) if fresh.exit_code is not None else "—",
            }
            for k, v in details.items():
                if v:
                    st.text(f"{k:>14}: {v}")

        # Rerun the parent on any status change so the connection-bar
        # job counts (rendered outside this fragment) stay fresh.
        if fresh.status != old_status:
            st.rerun(scope="app")

    _live_output()


# ── Terminal ──────────────────────────────────────────────────

def _render_terminal() -> None:
    conn: Optional[HPCConnection] = st.session_state.get("hpc_connection")
    if conn is None or not conn.is_connected:
        st.warning("Not connected.")
        return

    cwd = st.session_state.get("hpc_remote_cwd", "~")
    st.caption(f"`{conn.profile.username}@{conn.profile.hostname}:{cwd}$`")

    history: list[dict] = st.session_state.get("hpc_terminal_history", [])

    # Scrollable history
    if history:
        lines: list[str] = []
        for entry in history:
            lines.append(f"$ {entry['cmd']}")
            if entry["stdout"]:
                lines.append(entry["stdout"].rstrip())
            if entry["stderr"]:
                lines.append(f"[stderr] {entry['stderr'].rstrip()}")
            if entry.get("rc", 0) != 0:
                lines.append(f"[exit code {entry['rc']}]")
            lines.append("")

        import html as _html

        escaped = _html.escape("\n".join(lines))
        st.iframe(
            f'<pre id="term" style="height:340px;overflow-y:auto;'
            f"margin:0;background:#0e1117;padding:12px;"
            f"border-radius:6px;border:1px solid #333;"
            f"font-family:'Fira Code',monospace;font-size:13px;"
            f'white-space:pre-wrap;color:#a8db8f">'
            f"{escaped}</pre>"
            f"<script>var e=document.getElementById('term');"
            f"e.scrollTop=e.scrollHeight;</script>",
            height=360,
        )

    # Command input
    with st.form("hpc_term_form", clear_on_submit=True):
        fc1, fc2 = st.columns([6, 1])
        with fc1:
            cmd = st.text_input(
                "cmd",
                placeholder="ls -la",
                label_visibility="collapsed",
                key="hpc_term_cmd",
            )
        with fc2:
            submitted = st.form_submit_button("Run", type="primary")

    if submitted and cmd:
        full_cmd = f"cd {_q(cwd)} && {cmd}"
        try:
            stdout, stderr, rc = conn.run(full_cmd, timeout=120)
        except Exception as exc:
            stdout, stderr, rc = "", str(exc), -1

        # Track directory changes
        stripped = cmd.strip()
        if stripped == "cd" or stripped.startswith("cd "):
            pwd_out, _, _ = conn.run(
                f"cd {_q(cwd)} && {cmd} && pwd", timeout=10,
            )
            new_cwd = pwd_out.strip()
            if new_cwd:
                st.session_state.hpc_remote_cwd = new_cwd

        history.append({"cmd": cmd, "stdout": stdout, "stderr": stderr, "rc": rc})
        st.session_state.hpc_terminal_history = history[-200:]
        st.rerun()

    if history and st.button("Clear history", key="hpc_clear_hist"):
        st.session_state.hpc_terminal_history = []
        st.rerun()


# ── Remote Files ──────────────────────────────────────────────

def _render_remote_files() -> None:
    conn: Optional[HPCConnection] = st.session_state.get("hpc_connection")
    if conn is None or not conn.is_connected:
        st.warning("Not connected.")
        return

    cwd = st.session_state.get("hpc_remote_cwd", "~")
    if cwd == "~":
        cwd = conn.home_dir()
        st.session_state.hpc_remote_cwd = cwd

    # Navigation
    c_path, c_up, c_home, c_ref = st.columns([5, 1, 1, 1])
    with c_path:
        new_path = st.text_input(
            "Path", value=cwd, key="hpc_path_nav", label_visibility="collapsed",
        )
        if new_path != cwd:
            st.session_state.hpc_remote_cwd = new_path
            st.rerun()
    with c_up:
        if st.button("⬆ Up", key="hpc_nav_up"):
            st.session_state.hpc_remote_cwd = str(PurePosixPath(cwd).parent)
            st.rerun()
    with c_home:
        if st.button("🏠", key="hpc_nav_home"):
            st.session_state.hpc_remote_cwd = conn.home_dir()
            st.rerun()
    with c_ref:
        if st.button("🔄", key="hpc_nav_ref"):
            st.rerun()

    show_hidden = st.checkbox("Show hidden files", key="hpc_show_hidden")

    # Upload
    with st.expander("📤 Upload files here"):
        up_files = st.file_uploader(
            "Select files",
            accept_multiple_files=True,
            key="hpc_fbrowser_upload",
        )
        if up_files and st.button("Upload", key="hpc_do_fb_upload"):
            for uf in up_files:
                try:
                    _upload_streamlit_file(conn, uf, f"{cwd}/{uf.name}")
                    st.toast(f"✅ {uf.name}")
                except Exception as exc:
                    st.error(f"Failed: {uf.name} — {exc}")
            st.rerun()

    # Listing
    try:
        entries = conn.listdir(cwd)
    except Exception as exc:
        st.error(f"Cannot list `{cwd}`: {exc}")
        return

    entries.sort(key=lambda e: (not _is_dir(e), e.filename.lower()))

    # Table header
    h1, h2, h3, h4 = st.columns([4, 1.2, 2, 1])
    with h1:
        st.caption("**Name**")
    with h2:
        st.caption("**Size**")
    with h3:
        st.caption("**Modified**")
    with h4:
        st.caption("**Action**")

    for entry in entries:
        name = entry.filename
        if not show_hidden and name.startswith("."):
            continue
        is_dir = _is_dir(entry)
        size = _fmt_size(entry.st_size) if not is_dir else "—"
        mtime = (
            datetime.fromtimestamp(entry.st_mtime).strftime("%Y-%m-%d %H:%M")
            if entry.st_mtime
            else ""
        )
        icon = "📁" if is_dir else _file_icon(name)

        r1, r2, r3, r4 = st.columns([4, 1.2, 2, 1])
        with r1:
            if is_dir:
                if st.button(
                    f"{icon} {name}/",
                    key=f"hpc_fd_{name}",
                    width="stretch",
                ):
                    st.session_state.hpc_remote_cwd = f"{cwd}/{name}"
                    st.rerun()
            else:
                st.text(f"{icon} {name}")
        with r2:
            st.caption(size)
        with r3:
            st.caption(mtime)
        with r4:
            if not is_dir:
                if st.button("⬇", key=f"hpc_fdl_{name}", help=f"Download {name}"):
                    st.session_state["_hpc_download_pending"] = f"{cwd}/{name}"
                    st.rerun()

    # Handle pending download (must be outside the loop for st.download_button)
    pending_dl = st.session_state.pop("_hpc_download_pending", None)
    if pending_dl:
        fname = PurePosixPath(pending_dl).name
        try:
            data = conn.read_bytes(pending_dl)
            st.download_button(
                f"💾 Save **{fname}**",
                data=data,
                file_name=fname,
                key="hpc_dl_ready",
            )
        except Exception as exc:
            st.error(f"Download failed: {exc}")

    # Preview selected file
    preview_path = st.session_state.get("_hpc_preview_path")
    if preview_path:
        with st.expander(f"Preview: {PurePosixPath(preview_path).name}", expanded=True):
            try:
                content = conn.read_text(preview_path, tail=500)
                st.code(content, language=_guess_lang(preview_path))
            except Exception as exc:
                st.caption(f"Cannot preview: {exc}")


# ── Helpers ───────────────────────────────────────────────────

def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _is_dir(entry) -> bool:
    return stat_mod.S_ISDIR(entry.st_mode) if entry.st_mode else False


def _fmt_size(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} P"


_ICON_MAP = {
    ".py": "🐍", ".sh": "⚙️", ".bash": "⚙️",
    ".csv": "📊", ".tsv": "📊", ".dat": "📊",
    ".json": "📋", ".yaml": "📋", ".yml": "📋", ".toml": "📋",
    ".txt": "📝", ".log": "📝", ".out": "📝", ".err": "📝",
    ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼",
    ".h5": "🔢", ".hdf5": "🔢", ".npy": "🔢", ".npz": "🔢",
    ".xml": "📄", ".html": "📄", ".inp": "📄", ".com": "📄",
    ".f90": "💻", ".f": "💻", ".c": "💻", ".cpp": "💻",
}


def _file_icon(name: str) -> str:
    return _ICON_MAP.get(PurePosixPath(name).suffix.lower(), "📄")


def _guess_lang(path: str) -> str:
    ext = PurePosixPath(path).suffix.lower()
    return {
        ".py": "python", ".sh": "bash", ".bash": "bash",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".xml": "xml", ".html": "html",
        ".c": "c", ".cpp": "cpp", ".f90": "fortran",
    }.get(ext, "text")


def _upload_streamlit_file(conn: HPCConnection, uf, remote_path: str) -> None:
    """Upload a Streamlit UploadedFile to the remote host."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uf.getvalue())
        tmp_path = tmp.name
    try:
        conn.upload(tmp_path, remote_path)
    finally:
        os.unlink(tmp_path)
