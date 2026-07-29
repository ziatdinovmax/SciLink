"""VASP DFT wizard — Configure -> Review -> Submit -> Monitor.

Symmetric to the LAMMPS wizard in `sim_workflow.py`, but drives
`SimulationOrchestratorAgent.run_task()` to generate VASP inputs
(POSCAR, INCAR, KPOINTS) instead of building a self-contained run
script around `LAMMPSOrchestrator`. Submission, monitoring, and results
phases reuse the engine-agnostic implementations from `sim_workflow.py`.
"""
from __future__ import annotations

import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st

from scilink.ui.components.sim_workflow import (
    _render_monitoring,
    _render_results,
    _upload_text,
    _q,
)
from scilink.ui.components.wizard_state import (
    apply_vasp_defaults,
    save_vasp,
)


# ══════════════════════════════════════════════════════════════
# SLURM template
# ══════════════════════════════════════════════════════════════

def _build_slurm_script(
    *,
    job_name: str,
    account: str,
    partition: str,
    time_limit: str,
    nodes: int,
    ntasks_per_node: int,
    extra_sbatch: str,
    work_dir: str,
    module_commands: str,
    pseudo_dir: str,
    elements: List[str],
    vasp_command: str,
) -> str:
    """Construct the SLURM script: SBATCH directives, module loads,
    POTCAR assembly from `pseudo_dir`, then `vasp_command`."""
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --nodes={int(nodes)}",
        f"#SBATCH --ntasks-per-node={int(ntasks_per_node)}",
        f"#SBATCH --output={job_name}_%j.out",
        f"#SBATCH --error={job_name}_%j.err",
    ]
    if account.strip():
        sbatch_lines.append(f"#SBATCH --account={account.strip()}")
    if partition.strip():
        sbatch_lines.append(f"#SBATCH --partition={partition.strip()}")
    for line in (extra_sbatch or "").splitlines():
        line = line.strip()
        if line:
            sbatch_lines.append(f"#SBATCH {line}")

    elements_arr = " ".join(_q(e) for e in elements) if elements else ""

    body = textwrap.dedent(f"""\

        set -e
        cd {_q(work_dir)}

        # ---- Module environment ----
        {module_commands.strip() or "# (no modules specified)"}

        # ---- Assemble POTCAR from pseudo dir ----
        PSEUDO_DIR={_q(pseudo_dir)}
        ELEMENTS=({elements_arr})
        if [ ${{#ELEMENTS[@]}} -eq 0 ]; then
            echo "No elements parsed from POSCAR; aborting." >&2
            exit 1
        fi
        : > POTCAR
        for e in "${{ELEMENTS[@]}}"; do
            if [ ! -f "$PSEUDO_DIR/$e/POTCAR" ]; then
                echo "Missing POTCAR for element $e at $PSEUDO_DIR/$e/POTCAR" >&2
                exit 1
            fi
            cat "$PSEUDO_DIR/$e/POTCAR" >> POTCAR
        done

        # ---- Run VASP ----
        {vasp_command}
    """)

    return "\n".join(sbatch_lines) + body


def _elements_from_poscar(poscar_text: str) -> List[str]:
    """Parse the element symbols from a VASP-5+ POSCAR (line 6)."""
    lines = poscar_text.splitlines()
    if len(lines) < 6:
        return []
    return lines[5].split()


# ══════════════════════════════════════════════════════════════
# Workflow state machine
# ══════════════════════════════════════════════════════════════

def render_agent_workflow() -> None:
    """Main entry point — dispatches to the current workflow phase."""
    phase = st.session_state.get("hpc_workflow_phase", "configure")

    if phase == "configure":
        _render_configure()
    elif phase == "review":
        _render_review_inputs()
    elif phase == "monitoring":
        _render_monitoring()
    elif phase == "results":
        _render_results()
    else:
        st.session_state.hpc_workflow_phase = "configure"
        st.rerun()


# ══════════════════════════════════════════════════════════════
# Phase 1: Configure
# ══════════════════════════════════════════════════════════════

def _render_configure() -> None:
    # Pre-fill infrastructure fields from the last successful Generate
    # before any widget is created.
    apply_vasp_defaults()

    st.subheader("⚛️ VASP DFT Simulation")

    cfg = st.session_state.get("agent_config", {})
    _model = cfg.get("model", "")
    _has_key = bool(
        cfg.get("api_key")
        or os.environ.get("SCILINK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    conn = st.session_state.get("hpc_connection")
    has_hpc = conn is not None and conn.is_connected

    if not _has_key:
        st.warning(
            "No LLM API key detected. Launch a simulate session from the "
            "sidebar with a key set, or export SCILINK_API_KEY before starting."
        )

    if has_hpc:
        if _model:
            st.caption(
                f"Model: **{_model}** · HPC: 🟢 **{conn.profile.hostname}**"
            )

        env = st.session_state.get("hpc_env_probe")
        if env is None:
            from scilink.hpc.probe import probe_remote
            with st.spinner("Probing remote environment…"):
                env = probe_remote(conn)
            st.session_state.hpc_env_probe = env

        with st.expander("🔍 Detected environment"):
            det_c1, det_c2 = st.columns(2)
            with det_c1:
                st.caption(f"**Home:** `{env.home}`")
                if env.scratch:
                    st.caption(f"**Scratch:** `{env.scratch}`")
                if env.vasp_binaries:
                    st.caption(
                        f"**VASP binaries:** "
                        f"{', '.join(f'`{b}`' for b in env.vasp_binaries)}"
                    )
                else:
                    st.caption("**VASP binaries:** none found in PATH")
            with det_c2:
                if env.vasp_modules:
                    st.caption(
                        f"**VASP modules:** "
                        f"{', '.join(f'`{m}`' for m in env.vasp_modules)}"
                    )
            if st.button("🔄 Re-probe", key="vasp_reprobe"):
                st.session_state.hpc_env_probe = None
                st.rerun()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Scientific objective
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("**Scientific objective**")
    structure_description = st.text_area(
        "Structure description",
        height=80,
        key="vasp_structure_description",
        placeholder="e.g. rutile TiO2, 2x2x2 supercell with one oxygen vacancy",
    )
    research_goal = st.text_area(
        "Research goal",
        height=100,
        key="vasp_research_goal",
        placeholder="e.g. Compute the band structure with HSE06 and the GGA+U formation energy of the oxygen vacancy.",
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INCAR generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("**INCAR generation**")
    incar_method = st.selectbox(
        "Method",
        ["llm", "atomate2"],
        key="vasp_incar_method",
        help="`llm` uses the orchestrator's LLM-driven INCAR builder; `atomate2` uses atomate2 templates when applicable.",
    )
    with st.expander("Advanced (optional)"):
        futurehouse_key = st.text_input(
            "FutureHouse API key (for INCAR validation)",
            type="password",
            key="vasp_fh_key",
            help="When set, the agent will validate the generated INCAR against literature and apply suggested fixes.",
        )
        incar_overrides = st.text_area(
            "Manual INCAR additions / overrides",
            height=80,
            key="vasp_incar_overrides",
            placeholder="ENCUT = 520\nISMEAR = 0\nSIGMA = 0.05",
            help="Free-text INCAR snippets to fold into the agent's request. Leave blank to let the agent decide.",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HPC environment
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("**HPC environment**")
    if not has_hpc:
        st.info(
            "No HPC connection. You can still generate inputs; submission "
            "will be unavailable until you connect via the sidebar."
        )
    vasp_command = st.text_input(
        "VASP command",
        value="mpirun vasp_std",
        key="vasp_command",
        help="The exact command that runs VASP after POTCAR is assembled.",
    )
    module_commands = st.text_area(
        "Module setup",
        height=80,
        key="vasp_module_commands",
        placeholder="module purge\nmodule load vasp/6.4.2 intel-mpi/2021",
    )
    pseudo_dir = st.text_input(
        "PSEUDO_DIR (remote)",
        key="vasp_pseudo_dir",
        placeholder="/path/to/potpaw_PBE.54",
        help="Remote directory containing element subdirs (e.g. Si/POTCAR, O/POTCAR). The submit script `cat`s these to build POTCAR.",
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Remote layout
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    remote_work_dir = st.text_input(
        "Remote working directory",
        key="vasp_remote_work_dir",
        placeholder="/scratch/$USER/scilink/vasp_runs",
        help="Base directory; a per-job subdirectory will be created at submit time.",
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SLURM
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("**SLURM settings**")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        account = st.text_input("Account (-A)", key="vasp_slurm_account")
        time_limit = st.text_input(
            "Time limit (-t)", value="04:00:00", key="vasp_slurm_time"
        )
    with sc2:
        nodes = st.number_input(
            "Nodes (-N)", value=1, min_value=1, key="vasp_slurm_nodes"
        )
        ntasks = st.number_input(
            "Tasks/node", value=16, min_value=1, key="vasp_slurm_ntasks"
        )
    with sc3:
        job_name = st.text_input(
            "Job name (-J)", value="scilink_vasp", key="vasp_slurm_jname"
        )
        partition = st.text_input(
            "Partition (-p, optional)", key="vasp_slurm_partition"
        )
    extra_sbatch = st.text_area(
        "Extra SBATCH directives (one per line, without `#SBATCH`)",
        key="vasp_extra_sbatch",
        height=60,
        placeholder="--exclusive\n--mem=0",
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Generate
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    can_proceed = bool(
        structure_description.strip()
        and research_goal.strip()
        and remote_work_dir.strip()
        and pseudo_dir.strip()
        and _has_key
    )
    if not can_proceed:
        missing: list[str] = []
        if not structure_description.strip():
            missing.append("structure description")
        if not research_goal.strip():
            missing.append("research goal")
        if not pseudo_dir.strip():
            missing.append("PSEUDO_DIR")
        if not remote_work_dir.strip():
            missing.append("remote working directory")
        if not _has_key:
            missing.append("LLM API key")
        st.info(f"Still needed: {', '.join(missing)}")

    err_slot = st.empty()

    if st.button(
        "⚛️ Generate VASP inputs",
        type="primary",
        disabled=not can_proceed,
        key="vasp_generate_btn",
    ):
        save_vasp()
        try:
            with st.spinner(
                "Asking SimulationOrchestratorAgent to build POSCAR / INCAR / KPOINTS…"
            ):
                structure = _run_generation(
                    structure_description=structure_description.strip(),
                    research_goal=research_goal.strip(),
                    incar_method=incar_method,
                    incar_overrides=incar_overrides.strip(),
                    futurehouse_key=futurehouse_key or None,
                )
        except Exception as exc:
            err_slot.error(f"Generation failed: {exc}")
            return

        if structure is None:
            err_slot.error(
                "Agent returned no structure. Check the orchestrator logs for "
                "the agent's reply (it may have asked a clarifying question)."
            )
            return

        # Stash files + draft SLURM script in session state. The backend
        # record stores a generic input_files map; this VASP wizard reads
        # its engine's files (INCAR/KPOINTS) from it by name. (Engine-agnostic
        # UI rework is a separate PR — see task list.)
        try:
            files = structure.get("input_files") or {}
            poscar_text = Path(structure["structure_path"]).read_text()
            incar_text = Path(files["INCAR"]).read_text() if files.get("INCAR") else ""
            kpoints_text = Path(files["KPOINTS"]).read_text() if files.get("KPOINTS") else ""
        except Exception as exc:
            err_slot.error(f"Could not read generated files: {exc}")
            return

        elements = _elements_from_poscar(poscar_text)
        slurm_script = _build_slurm_script(
            job_name=job_name,
            account=account,
            partition=partition,
            time_limit=time_limit,
            nodes=int(nodes),
            ntasks_per_node=int(ntasks),
            extra_sbatch=extra_sbatch,
            work_dir=remote_work_dir.rstrip("/") + f"/{structure['slug']}",
            module_commands=module_commands,
            pseudo_dir=pseudo_dir.strip(),
            elements=elements,
            vasp_command=vasp_command.strip(),
        )

        st.session_state.hpc_gen_vasp_files = {
            "POSCAR": poscar_text,
            "INCAR": incar_text,
            "KPOINTS": kpoints_text,
        }
        st.session_state.hpc_gen_slurm_script = slurm_script
        st.session_state.hpc_gen_remote_work_dir = (
            remote_work_dir.rstrip("/") + f"/{structure['slug']}"
        )
        st.session_state.hpc_gen_job_name = job_name
        st.session_state.vasp_last_slug = structure["slug"]
        st.session_state.hpc_workflow_phase = "review"
        st.rerun()


def _get_or_create_agent():
    """Lazily build SimulationOrchestratorAgent and cache it in session state.

    The simulate-mode session is otherwise "lightweight" (no agent) — see
    `sidebar.py:_start_simulate_session`. We only spin up the agent when
    the user actually clicks Generate, so LAMMPS-only users never pay the cost.
    """
    if "hpc_vasp_agent" in st.session_state and st.session_state.hpc_vasp_agent is not None:
        agent = st.session_state.hpc_vasp_agent
        # Refresh HPC plumbing in case the user (re)connected after agent build.
        agent.hpc_connection = st.session_state.get("hpc_connection")
        agent.hpc_scheduler = st.session_state.get("hpc_scheduler")
        return agent

    from scilink.agents.sim_agents.simulation_orchestrator import (
        SimulationOrchestratorAgent,
        SimulationMode,
    )

    cfg = st.session_state.get("agent_config", {})
    api_key = cfg.get("api_key") or None
    base_url = cfg.get("base_url") or None

    agent = SimulationOrchestratorAgent(
        base_dir=st.session_state.get("session_dir", "./simulate_session"),
        api_key=api_key,
        model_name=cfg.get("model") or "claude-opus-4-6",
        base_url=base_url,
        simulation_mode=SimulationMode.CO_PILOT,
        futurehouse_api_key=st.session_state.get("vasp_fh_key") or None,
        hpc_connection=st.session_state.get("hpc_connection"),
        hpc_scheduler=st.session_state.get("hpc_scheduler"),
    )
    st.session_state.hpc_vasp_agent = agent
    return agent


def _run_generation(
    *,
    structure_description: str,
    research_goal: str,
    incar_method: str,
    incar_overrides: str,
    futurehouse_key: Optional[str],
) -> Optional[dict]:
    """Call agent.run_task() with a focused prompt; return the new structure
    record (or None if the agent produced no structure)."""
    agent = _get_or_create_agent()

    overrides_block = ""
    if incar_overrides:
        overrides_block = (
            "\nUser-supplied INCAR additions/overrides to fold into the request:\n"
            f"```\n{incar_overrides}\n```\n"
        )
    fh_step = ""
    if futurehouse_key:
        fh_step = (
            "3. Validate the generated INCAR with `validate_incar` and apply "
            "improvements if validation flags substantive issues.\n"
        )

    task = textwrap.dedent(f"""\
        Build VASP inputs for the following research objective.

        Structure: {structure_description}
        Research goal: {research_goal}
        INCAR generation method: {incar_method}
        {overrides_block}
        Steps:
        1. Generate the structure (use `generate_structure`).
        2. Generate VASP inputs (use `generate_vasp_inputs` with method={incar_method!r}).
        {fh_step}
        Return when POSCAR, INCAR, and KPOINTS are written. Do NOT submit
        the job — the wizard handles HPC submission separately in a later
        step where the user reviews and edits the inputs.
    """)

    n_before = len(agent.generated_structures)
    result = agent.run_task(task)

    new = agent.generated_structures[n_before:]
    if not new:
        return None
    # Use the most recent record (the call may have produced multiple if the
    # agent decided to refine; the wizard always uses the latest).
    candidate = new[-1]
    files = candidate.get("input_files") or {}
    if not (
        candidate.get("structure_path")
        and files.get("INCAR")
        and files.get("KPOINTS")
    ):
        return None
    return candidate


# ══════════════════════════════════════════════════════════════
# Phase 2: Review & Submit
# ══════════════════════════════════════════════════════════════

def _render_review_inputs() -> None:
    st.subheader("📝 Review & Submit VASP Inputs")
    st.caption("Edit the generated files if needed, then submit to the cluster.")

    remote_dir = st.session_state.get("hpc_gen_remote_work_dir", "")
    st.info(f"**Remote working directory:** `{remote_dir}`")

    files = st.session_state.get("hpc_gen_vasp_files", {})
    poscar = files.get("POSCAR", "")
    incar = files.get("INCAR", "")
    kpoints = files.get("KPOINTS", "")
    slurm = st.session_state.get("hpc_gen_slurm_script", "")

    tab_poscar, tab_incar, tab_kpoints, tab_submit = st.tabs(
        ["POSCAR", "INCAR", "KPOINTS", "submit.sh"]
    )
    with tab_poscar:
        poscar_edit = st.text_area(
            "POSCAR", value=poscar, height=350, key="vasp_edit_poscar",
        )
    with tab_incar:
        incar_edit = st.text_area(
            "INCAR", value=incar, height=350, key="vasp_edit_incar",
        )
    with tab_kpoints:
        kpoints_edit = st.text_area(
            "KPOINTS", value=kpoints, height=200, key="vasp_edit_kpoints",
        )
    with tab_submit:
        slurm_edit = st.text_area(
            "submit.sh", value=slurm, height=350, key="vasp_edit_slurm",
        )

    st.markdown("---")
    err_slot = st.empty()

    c_back, c_submit = st.columns(2)
    with c_back:
        if st.button(
            "← Back to configure",
            key="vasp_review_back",
            width="stretch",
        ):
            st.session_state.hpc_workflow_phase = "configure"
            st.rerun()

    with c_submit:
        if st.button(
            "🚀 Upload & Submit",
            type="primary",
            key="vasp_review_submit",
            width="stretch",
        ):
            conn = st.session_state.get("hpc_connection")
            sched = st.session_state.get("hpc_scheduler")
            if not conn or not conn.is_connected:
                err_slot.error("HPC connection lost. Reconnect via sidebar.")
                return
            if sched is None:
                err_slot.error("No scheduler detected on this cluster.")
                return

            try:
                with st.spinner("Uploading inputs and submitting job…"):
                    conn.run(f"mkdir -p {_q(remote_dir)}", timeout=30)
                    _upload_text(conn, poscar_edit, f"{remote_dir}/POSCAR")
                    _upload_text(conn, incar_edit, f"{remote_dir}/INCAR")
                    _upload_text(conn, kpoints_edit, f"{remote_dir}/KPOINTS")
                    _upload_text(conn, slurm_edit, f"{remote_dir}/submit.sh")
                    conn.run(f"chmod +x {_q(remote_dir + '/submit.sh')}", timeout=30)

                    job_id = sched.submit(
                        f"{remote_dir}/submit.sh", work_dir=remote_dir,
                    )

                from scilink.hpc.scheduler import HPCJob, JobStatus

                job_name = st.session_state.get("hpc_gen_job_name", "scilink_vasp")
                job = HPCJob(
                    job_id=job_id,
                    name=job_name,
                    status=JobStatus.PENDING,
                    work_dir=remote_dir,
                    script_path=f"{remote_dir}/submit.sh",
                    stdout_file=f"{remote_dir}/{job_name}_{job_id}.out",
                    stderr_file=f"{remote_dir}/{job_name}_{job_id}.err",
                )

                tracked = st.session_state.get("hpc_tracked_jobs", {})
                tracked[job_id] = job
                st.session_state.hpc_tracked_jobs = tracked
                st.session_state.hpc_monitored_job_id = job_id
                st.session_state.hpc_mon_known_files = set()
                st.session_state.hpc_mon_downloaded_images = {}
                st.session_state.hpc_workflow_phase = "monitoring"

                st.success(f"✅ Submitted job **{job_id}**")
                st.balloons()
                st.rerun()

            except Exception as exc:
                err_slot.error(f"Submission failed: {exc}")
