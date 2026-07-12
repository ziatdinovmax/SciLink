"""Skills tab — upload custom skills and view available built-in skills."""

from pathlib import Path

import os
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

    st.divider()
    _render_memory_section()


def _render_upload_section(agent) -> None:
    """Upload custom skill files."""
    st.subheader("Upload Skills")
    st.caption("Available for this session only — not saved to persistent memory.")
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


def _render_skill_markdown(domain: str, name: str) -> None:
    """Render a discoverable skill's markdown (built-in / user / graduated)."""
    from scilink.skills.loader import _resolve_skill_path
    try:
        st.markdown(_resolve_skill_path(name, domain).read_text())
    except Exception as e:
        st.warning(f"Could not load skill: {e}")


def _render_available_skills(agent) -> None:
    """Show built-in and custom skills."""
    st.subheader("Available Skills")
    st.caption("Active in this session: shipped built-in skills plus any you uploaded.")

    # Built-in subsection
    st.markdown("**Built-in**")
    builtin = list_all_skills()
    if builtin:
        for domain, names in builtin.items():
            label = domain.replace("_", " ").title()
            with st.expander(f"{label} ({len(names)})", expanded=False):
                for name in names:
                    nc, vc = st.columns([3, 1])
                    nc.markdown(f"`{name}`")
                    with vc.popover("view", width="stretch"):
                        _render_skill_markdown(domain, name)
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


def _render_memory_section() -> None:
    """Persistent memory — graduated and auto-distilled skills under ~/.scilink.

    Provisional skills (auto-distilled from hard fits the agent solved only after
    escalating — T=2 auto-distillation, today wired in the curve-fitting agent;
    the provisional mechanism itself is domain-agnostic) are shown with a badge
    and can be promoted (made auto-routable) or pruned. Promoted skills survive
    sessions and pip upgrades and are auto-discovered by the loader.
    """
    from scilink.skills._shared import _memory

    from scilink.skills import loader

    st.subheader("Persistent Memory")
    st.caption(
        "Graduated and auto-distilled skills stored under `~/.scilink` — they "
        "survive sessions and upgrades. Provisional skills (auto-distilled from "
        "hard problems the agent had to solve from scratch) "
        "are held out of auto-routing until you promote them."
    )

    # Master on/off switch (opt-in; off by default). When off, persistent memory
    # is inert: nothing is staged and graduated skills are not loaded into runs.
    cur = loader.memory_enabled()
    new = st.toggle(
        "Enable persistent memory", value=cur, key="mem_enabled_toggle",
        help=("Off (default): inert — nothing is staged and graduated skills are "
              "not loaded into runs. On: auto-staging + reuse of promoted skills."),
    )
    if new != cur:
        loader.set_memory_enabled(new)
        st.rerun()
    if os.environ.get("SCILINK_MEMORY", "").strip():
        st.caption("⚠️ `SCILINK_MEMORY` env var is set and overrides this toggle.")
    if not new:
        st.info("Persistent memory is **OFF** (inert). Turn it on to capture "
                "hard-won solutions, your feedback, and error fixes — and to load "
                "promoted skills into runs. Existing items below are kept and can "
                "still be reviewed.")

    try:
        rows = _memory.list_memory()
    except Exception as e:
        st.error(f"Could not read persistent memory: {e}")
        return

    provisional = [r for r in rows if r["provisional"]]
    promoted = [r for r in rows if not r["provisional"]]

    if provisional:
        st.markdown(f"**Provisional — awaiting review ({len(provisional)})**")
        for r in _paged(provisional, "provpage"):
            _render_memory_row(_memory, r, provisional=True)
    if promoted:
        st.markdown(f"**Promoted ({len(promoted)})**")
        for r in _paged(promoted, "prompage"):
            _render_memory_row(_memory, r, provisional=False)
    if not rows:
        st.caption("No persisted skills yet.")

    _render_staged_section()
    _render_bank_section()


def _render_staged_section() -> None:
    """Staged raw T=2 solutions — distill into skills (upgrade an existing skill,
    or consolidate N of a technique into a new one)."""
    from scilink.skills._shared import _staging, _memory
    from scilink.skills import loader
    mem_on = loader.memory_enabled()

    st.markdown("---")
    st.markdown("**Staged knowledge** — solved-from-scratch solutions, your feedback & error fixes")
    st.caption(
        "Hard problems the agent solved only after rebuilding its approach from "
        "scratch — plus your feedback and recurring error fixes. Upgrade an "
        "existing skill from one, or consolidate several of the same technique "
        "into a new skill. Both use the active session's model."
    )

    groups = {}
    for rec in _staging.list_staged():
        groups.setdefault((rec["domain"], rec.get("technique") or "unlabeled"), []).append(rec)
    if not groups:
        st.caption("No staged solutions.")
        return

    agent = st.session_state.get("agent")
    model = getattr(agent, "model", None)

    def _llm_call(prompt: str) -> str:
        r = model.generate_content(contents=[prompt])
        return r.text if hasattr(r, "text") else str(r)

    for (domain, technique), recs in sorted(groups.items()):
        with st.expander(f"`{domain}/{technique}` — {len(recs)} staged", expanded=False):
            for r in _paged(recs, f"stagedpage::{domain}/{technique}"):
                metric = r.get("r_squared") or r.get("quality_score")
                prov = _staging.PROVENANCE_LABELS.get(r.get("provenance", "t2_solution"), r.get("provenance", ""))
                meta_col, view_col = st.columns([3, 1])
                meta_col.caption(
                    f"id={r['id']} · {prov} · session={r.get('session','?')}"
                    + (f" · {_staging.metric_label(r)}" if _staging.metric_label(r) else "")
                )
                with view_col.popover("View", width="stretch"):
                    _lazy_content(f"stagedlazy::{r['id']}",
                                  lambda r=r: _render_staged_record(r))
                    if st.button("Delete staged record",
                                 key=f"destage::{domain}/{r['id']}",
                                 help="Remove from the staging buffer without "
                                      "distilling it (de-stage)."):
                        _staging.remove_staged(domain, [r["id"]])
                        st.warning(f"De-staged {r['id']}.")
                        st.rerun()
            if model is None:
                st.info("Start a session to enable upgrade/consolidate (needs a model).")
                continue
            # candidate existing skills to upgrade into (same domain).
            # Keep the action buttons in narrow columns + a trailing spacer so
            # they land at a modest size (like the sidebar's Reset/Quit) rather
            # than stretching across the full-width memory panel.
            persistent = {s["name"] for s in _memory.list_memory(domain=domain)}
            targets = [f"{domain}/{n}" for n in sorted(persistent)]
            # Built-ins are valid targets via copy-on-write: selecting one
            # forks it into the persistent store first (shadowing the shipped
            # copy), then the normal review-gated upgrade applies to the fork.
            from scilink.skills import loader as _loader
            builtin_only = [n for n in _loader.list_skills(domain)
                            if n not in persistent]
            targets += [f"{domain}/{n} (built-in — forks on upgrade)"
                        for n in sorted(builtin_only)]
            c1, c2, _ = st.columns([2, 2, 4])
            prop_key = f"upgprop::{domain}/{technique}"
            with c1:
                if targets:
                    tgt = st.selectbox("Upgrade into", targets, key=f"tgt::{domain}/{technique}")
                    sid = recs[0]["id"]
                    # Upgrade mutates an existing skill in place, so preview first:
                    # build the merged skill, show a diff, and apply only on confirm.
                    if st.button("Preview upgrade", key=f"up::{domain}/{technique}",
                                 disabled=not mem_on,
                                 help=(None if mem_on else
                                       "Persistent memory is off — turn it on to distill "
                                       "staged solutions into skills.")):
                        from scilink.agents.exp_agents.instruct import (
                            KNOWLEDGE_TO_SKILL_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS)
                        td, tn = tgt.split("/", 1)
                        if tn.endswith(" (built-in — forks on upgrade)"):
                            tn = tn.split(" (built-in", 1)[0]
                            fout = _memory.fork_builtin(td, tn)
                            if fout.get("status") != "success":
                                st.error(fout.get("message", "Fork failed."))
                                st.rerun()
                            st.info(f"Forked built-in {td}/{tn} — the fork now "
                                    f"shadows the shipped skill and receives "
                                    f"this upgrade.")
                        prop = _staging.propose_skill_upgrade(
                            domain, [sid], target_domain=td, target_name=tn,
                            llm_call=_llm_call,
                            fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                            update_template=SKILL_UPDATE_INSTRUCTIONS)
                        if prop.get("status") == "success":
                            st.session_state[prop_key] = prop
                        else:
                            st.error(prop.get("message", "Could not build upgrade."))
                        st.rerun()
                else:
                    st.caption("No existing skills in this domain to upgrade into.")
            with c2:
                # New-skill consolidation accumulates first: only suggest it once
                # enough examples of this technique are staged. Below the threshold
                # the agent is still gathering evidence (one fit is too idiosyncratic
                # to generalize into a standalone skill). Upgrading an existing skill
                # is exempt — that's the upgrade@1 path on the left.
                need = _staging.consolidate_min_n()
                ready = len(recs) >= need
                if not mem_on:
                    con_help = ("Persistent memory is off — turn it on to distill "
                                "staged solutions into skills.")
                elif not ready:
                    con_help = (f"Accumulating {len(recs)}/{need} — consolidation into a "
                                f"new skill unlocks once {need} solutions of this technique "
                                f"are staged (set SCILINK_CONSOLIDATE_N to change; "
                                f"`scilink memory consolidate` can force it).")
                else:
                    con_help = None
                if st.button("Consolidate → new skill", key=f"con::{domain}/{technique}",
                             disabled=(not ready) or (not mem_on), help=con_help):
                    from scilink.agents.exp_agents.instruct import (
                        T2_CONSOLIDATION_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS)
                    res = _staging.consolidate_technique(
                        domain, technique, llm_call=_llm_call,
                        consolidation_template=T2_CONSOLIDATION_INSTRUCTIONS,
                        update_template=SKILL_UPDATE_INSTRUCTIONS)
                    st.success(f"Consolidated {res.get('n_examples')} → auto_{technique} (provisional).")
                    st.rerun()

            # Pending upgrade preview — review the merged content before applying.
            prop = st.session_state.get(prop_key)
            if prop:
                import difflib
                st.markdown(
                    f"**Review upgrade → `{prop['target_domain']}/{prop['target_name']}`** "
                    "— applies in place; the current version is backed up to `.md.bak`."
                )
                for w in prop.get("regression_warnings") or []:
                    st.warning(f"Additivity check: {w}")
                diff = "\n".join(difflib.unified_diff(
                    prop["existing_content"].splitlines(),
                    prop["proposed_content"].splitlines(),
                    fromfile="current", tofile="after upgrade", lineterm=""))
                st.code(diff or "(no textual change)", language="diff")
                a1, a2, _ = st.columns([2, 2, 4])
                if a1.button("Apply upgrade", key=f"upapply::{domain}/{technique}",
                             type="primary"):
                    res = _staging.apply_skill_upgrade(
                        domain, prop["staged_ids"],
                        target_domain=prop["target_domain"],
                        target_name=prop["target_name"],
                        proposed_content=prop["proposed_content"])
                    st.session_state.pop(prop_key, None)
                    if res.get("status") == "success":
                        st.success(f"Upgraded {prop['target_domain']}/{prop['target_name']} "
                                   f"(backup saved).")
                    else:
                        st.error(res.get("message", "Apply failed."))
                    st.rerun()
                if a2.button("Cancel", key=f"upcancel::{domain}/{technique}"):
                    st.session_state.pop(prop_key, None)
                    st.rerun()
                if not ready:
                    st.caption(f"Accumulating {len(recs)}/{need} examples before a new skill.")


# Streamlit renders popover/expander CONTENT eagerly on every rerun, open or
# not — with many memory records that meant re-reading every file and
# re-highlighting every script per chat interaction (the observed panel
# slowdown). Heavy content therefore loads only behind an explicit tick, and
# long lists paginate.
_PANEL_PAGE = 15
_SCRIPT_PREVIEW_CHARS = 8000


def _lazy_content(key: str, render_fn) -> None:
    """Render heavy content only when the user asks for it."""
    if st.checkbox("Load full content", key=key):
        render_fn()
    else:
        st.caption("Tick to load — kept lazy so a large store doesn't slow the panel.")


def _paged(items: list, key: str) -> list:
    """First page of a long list, with a 'Show all' toggle."""
    if len(items) <= _PANEL_PAGE:
        return items
    if st.session_state.get(key):
        return items
    if st.button(f"Show all {len(items)} (first {_PANEL_PAGE} shown)", key=f"btn::{key}"):
        st.session_state[key] = True
        st.rerun()
    return items[:_PANEL_PAGE]


def _script_block(script: str, full_hint: str) -> None:
    script = (script or "").strip()
    if not script:
        return
    st.markdown("**working script:**")
    if len(script) > _SCRIPT_PREVIEW_CHARS:
        st.code(script[:_SCRIPT_PREVIEW_CHARS]
                + f"\n# … truncated — full script via {full_hint}",
                language="python")
    else:
        st.code(script, language="python")


def _render_bank_section() -> None:
    """Script bank — episodic memory of successful analysis scripts.

    Lists banked scripts with their cross-session usage stats; a record
    proven across sessions (★) can be promoted into distill staging, where
    the existing review-gated upgrade/consolidate ceremony turns it into a
    skill. Inspection and pruning are always available.
    """
    from scilink.skills._shared import _script_bank

    st.markdown("---")
    st.markdown("**Script bank** — proven analysis scripts, retrieved & adapted on new data")
    st.caption(
        "Every approved analysis banks its working script with a fingerprint "
        "of the data it solved; later runs retrieve the closest match as a "
        "starting point, and real-time mode can cold-start a campaign from "
        "one. Records that keep succeeding across sessions (★) are "
        "evidence-backed skill candidates — promote one to send it into the "
        "staged-knowledge review path above."
    )

    try:
        rows = _script_bank.bank_summary()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the script bank: {e}")
        return
    if not rows:
        st.caption("No banked scripts yet — they accumulate as analyses succeed.")
        return

    threshold = _script_bank.proven_n()
    by_domain: dict = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r)

    for domain, recs in sorted(by_domain.items()):
        n_proven = sum(1 for r in recs if r["proven"])
        star = f", ★ {n_proven} proven" if n_proven else ""
        with st.expander(f"`{domain}` — {len(recs)} banked{star}", expanded=False):
            for r in _paged(recs, f"bankpage::{domain}"):
                metric = r["metric"]
                mtxt = (f" · {metric['name']}={metric['value']}"
                        if isinstance(metric, dict) and metric.get("value") is not None
                        else "")
                badge = "★ proven" if r["proven"] else \
                    f"{r['n_successes']}/{threshold} sessions"
                if r["promoted_to_staging"]:
                    badge += " · ✓ promoted"
                meta_col, view_col, act_col = st.columns([4, 1, 1])
                meta_col.caption(
                    f"id={r['id']} · {badge} · retrieved {r['n_retrievals']}×{mtxt}\n\n"
                    f"{r['label'][:110]}"
                )
                with view_col.popover("View", width="stretch"):
                    def _load(domain=domain, rid=r["id"]):
                        rec = _script_bank.get_record(domain, rid)
                        if rec:
                            _render_bank_record(rec)
                        else:
                            st.error("Record unreadable.")
                    _lazy_content(f"banklazy::{domain}/{r['id']}", _load)
                with act_col.popover("···", width="stretch"):
                    if st.button(
                        "Promote → staged knowledge",
                        key=f"bankprom::{domain}/{r['id']}",
                        disabled=bool(r["promoted_to_staging"]),
                        help=("Already promoted." if r["promoted_to_staging"] else
                              "Copies this script into the staged-knowledge "
                              "buffer above for the review-gated skill path. "
                              "The bank record is kept."),
                    ):
                        out = _script_bank.promote_to_staging(domain, r["id"])
                        if out.get("status") == "success":
                            st.success(f"Staged as {out['staged_id']} "
                                       f"[{out['technique']}] — review above.")
                        else:
                            st.error(out.get("message", "Promotion failed."))
                        st.rerun()
                    if st.button("Delete record",
                                 key=f"bankdel::{domain}/{r['id']}"):
                        _script_bank.remove_records(domain, [r["id"]])
                        st.rerun()


# Bank bookkeeping keys not worth echoing in the per-record viewer.
_BANK_HIDDEN_KEYS = {"id", "domain", "script_hash", "working_script"}


def _render_bank_record(rec: dict) -> None:
    """Show one bank record's matching tiers, stats, and script."""
    for k, v in rec.items():
        if k in _BANK_HIDDEN_KEYS or v in (None, "", [], {}):
            continue
        st.markdown(f"**{k.replace('_', ' ')}:** {v}")
    _script_block(rec.get("working_script"),
                  f"`scilink memory bank-show {rec.get('domain')}/{rec.get('id')}`")


# Bookkeeping keys not worth showing in the per-record viewer.
_STAGED_HIDDEN_KEYS = {"id", "domain", "technique", "session", "working_script", "script"}


def _render_staged_record(r: dict) -> None:
    """Show one staged T=2 solution's actual content (planned vs final model,
    deviation, metric, and the working script) so it can be inspected before
    upgrading/consolidating."""
    for k, v in r.items():
        if k in _STAGED_HIDDEN_KEYS or v in (None, "", [], {}):
            continue
        st.markdown(f"**{k.replace('_', ' ')}:** {v}")
    _script_block(r.get("working_script") or r.get("script"),
                  "the staging record file")


def _render_memory_row(_memory, r, *, provisional: bool) -> None:
    from scilink.skills._shared import _staging
    ref = f"{r['domain']}/{r['name']}"
    badge = "🟡 provisional" if provisional else "✅ promoted"
    _m = _staging.metric_label(r)
    r2 = f" · {_m}" if _m else ""
    with st.expander(f"{badge} · `{ref}`{r2}", expanded=False):
        if r.get("description"):
            st.markdown(f"_{r['description']}_")
        if r.get("provenance"):
            st.caption(f"provenance: {r['provenance']}"
                       + (f" · session: {r['session']}" if r.get("session") else ""))

        def _load_md(domain=r["domain"], name=r["name"]):
            try:
                st.markdown(_memory.show_memory(domain, name))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not render skill: {e}")
        _lazy_content(f"skilllazy::{ref}", _load_md)

        from scilink.skills import loader
        mem_on = loader.memory_enabled()
        c1, c2, _ = st.columns([2, 2, 4])
        if provisional:
            if c1.button("Promote", key=f"promote::{ref}", type="primary",
                         disabled=not mem_on,
                         help=(None if mem_on else
                               "Persistent memory is off — promoted skills won't load "
                               "until you turn it on.")):
                _memory.promote_memory(r["domain"], r["name"])
                st.success(f"Promoted {ref} — now auto-routable.")
                st.rerun()
        else:
            if c1.button("Demote", key=f"demote::{ref}",
                         help=("Set back to provisional — taken out of the "
                               "auto-routing menu (still explicitly loadable) "
                               "until you re-promote it.")):
                _memory.demote_memory(r["domain"], r["name"])
                st.warning(f"Demoted {ref} to provisional.")
                st.rerun()
        if c2.button("Prune", key=f"prune::{ref}"):
            _memory.prune_memory(r["domain"], r["name"])
            st.warning(f"Pruned {ref}.")
            st.rerun()
