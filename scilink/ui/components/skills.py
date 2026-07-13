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

    # The memory pipeline reads top-to-bottom in the order knowledge flows:
    # every success lands in the bank automatically; nominations and lessons
    # wait in the review inbox; reviewed knowledge becomes skills that guide
    # planning. One summary strip shows the whole state.
    _render_pipeline_summary(rows)
    _render_bank_section()
    _render_staged_section()
    _render_skills_section(_memory, rows)


def _render_pipeline_summary(skill_rows) -> None:
    """One-line state of the memory pipeline: bank → inbox → skills."""
    from scilink.skills._shared import _script_bank, _staging

    try:
        bank = _script_bank.bank_summary()
        n_proven = sum(1 for r in bank if r["proven"])
        staged = _staging.list_staged()
        need = _staging.consolidate_min_n()
        by_tech = {}
        for s in staged:
            by_tech.setdefault((s.get("domain"), s.get("technique")), []).append(s)
        n_ready = sum(1 for recs in by_tech.values() if len(recs) >= need)
        n_prov = sum(1 for r in skill_rows if r["provisional"])
        star = f", ★{n_proven} proven" if n_proven else ""
        ready = f", {n_ready} ready to distill" if n_ready else ""
        prov = f", {n_prov} provisional" if n_prov else ""
        st.markdown(
            f"Script bank ({len(bank)}{star}) → "
            f"Review inbox ({len(staged)}{ready}) → "
            f"Skills ({len(skill_rows)}{prov})"
        )
    except Exception:  # noqa: BLE001 - the strip must never break the panel
        pass


def _render_skills_section(_memory, rows) -> None:
    """Stage 3 — curated skills (the end of the pipeline)."""
    st.markdown("---")
    st.markdown("**3 · Skills** — curated knowledge that guides planning")
    st.caption(
        "The end of the pipeline: reviewed knowledge with planning-layer "
        "authority. Approved skills auto-route; provisional ones wait for "
        "your call; a fork of a built-in shadows the shipped copy."
    )
    provisional = [r for r in rows if r["provisional"]]
    promoted = [r for r in rows if not r["provisional"]]

    if provisional:
        st.markdown(f"**Provisional — awaiting review ({len(provisional)})**")
        for r in _paged(provisional, "provpage"):
            _render_memory_row(_memory, r, provisional=True)
    if promoted:
        st.markdown(f"**Approved for routing ({len(promoted)})**")
        for r in _paged(promoted, "prompage"):
            _render_memory_row(_memory, r, provisional=False)
    if not rows:
        st.caption("No skills yet — they are distilled from the review inbox above.")


def _render_staged_section() -> None:
    """Stage 2 — the review inbox as a selection workspace.

    Records (script nominations, error lessons, feedback) are grouped by
    technique for orientation, but the ACTIONS operate on the user's
    checkbox SELECTION, which may span groups — assemble the knowledge set
    (e.g. a group's scripts plus the error lessons their sessions produced
    under other labels), then choose its destination: consolidate into a
    NEW skill, or upgrade an EXISTING one.
    """
    from scilink.skills._shared import _staging, _memory
    from scilink.skills import loader
    mem_on = loader.memory_enabled()

    st.markdown("---")
    st.markdown("**2 · Review inbox** — select records, then distill")
    st.caption(
        "Where the three knowledge streams converge before becoming a skill: "
        "📜 script nominations, 🐛 error lessons (what went wrong + the fix), "
        "and 💬 your feedback (treated as ground truth). Tick the records to "
        "distill — selection can span groups — then consolidate them into a "
        "new skill or upgrade an existing one below. Distilling merges them "
        "into one skill: method + pitfalls + your constraints."
    )

    all_staged = _staging.list_staged()
    if not all_staged:
        st.caption("No staged records.")
        return
    groups: dict = {}
    for rec in all_staged:
        groups.setdefault((rec["domain"], rec.get("technique") or "unlabeled"),
                          []).append(rec)

    agent = st.session_state.get("agent")
    model = getattr(agent, "model", None)

    def _llm_call(prompt: str) -> str:
        r = model.generate_content(contents=[prompt])
        return r.text if hasattr(r, "text") else str(r)

    _icon = {"error_fix": "🐛", "user_correction": "💬"}

    def _sel_key(domain, rid):
        return f"sel::{domain}/{rid}"

    # ── record rows, grouped, each with a selection checkbox ──
    for (domain, technique), recs in sorted(groups.items()):
        n_sel = sum(1 for r in recs
                    if st.session_state.get(_sel_key(domain, r["id"])))
        sel_note = f" · {n_sel} selected" if n_sel else ""
        with st.expander(f"`{domain}/{technique}` — {len(recs)} staged{sel_note}",
                         expanded=False):
            st.caption("distills into: method 📜 + pitfalls 🐛 + your constraints 💬")
            b1, b2, _sp = st.columns([1, 1, 4])
            if b1.button("Select group", key=f"gsel::{domain}/{technique}"):
                for r in recs:
                    st.session_state[_sel_key(domain, r["id"])] = True
                st.rerun()
            if b2.button("Deselect", key=f"gdesel::{domain}/{technique}"):
                for r in recs:
                    st.session_state[_sel_key(domain, r["id"])] = False
                st.rerun()
            for r in _paged(recs, f"stagedpage::{domain}/{technique}"):
                prov_key = r.get("provenance", "t2_solution")
                icon = _icon.get(prov_key, "📜")
                prov = _staging.PROVENANCE_LABELS.get(prov_key, prov_key)
                bank_link = ""
                if r.get("bank_id"):
                    from scilink.skills._shared import _script_bank
                    brec = _script_bank.get_record(domain, r["bank_id"])
                    n_succ = ((brec or {}).get("stats") or {}).get("n_successes")
                    bank_link = (f" · from bank `{r['bank_id']}`"
                                 + (f" (succeeded in {n_succ} sessions)"
                                    if n_succ and n_succ > 1 else ""))
                metric = _staging.metric_label(r)
                sel_col, view_col = st.columns([4, 1])
                sel_col.checkbox(
                    f"{icon} `{r['id']}` · {prov} · session={r.get('session', '?')}"
                    + (f" · {metric}" if metric else "") + bank_link,
                    key=_sel_key(domain, r["id"]))
                with view_col.popover("View", width="stretch"):
                    _lazy_content(f"stagedlazy::{r['id']}",
                                  lambda r=r: _render_staged_record(r))
                    if st.button("Discard",
                                 key=f"destage::{domain}/{r['id']}",
                                 help="Remove from the review inbox without "
                                      "distilling it. A nominated bank record "
                                      "becomes nominatable again."):
                        _staging.remove_staged(domain, [r["id"]])
                        st.warning(f"De-staged {r['id']}.")
                        st.rerun()
            # Discovery aid: lessons born in the SAME sessions as this
            # group's records but filed under other labels — the natural
            # candidates to co-select for a braided skill.
            sessions = {r.get("session") for r in recs if r.get("session")}
            related = [o for o in all_staged
                       if o["domain"] == domain
                       and (o.get("technique") or "unlabeled") != technique
                       and o.get("session") in sessions]
            if related:
                st.caption(
                    "🔗 From the same sessions, under other groups: "
                    + ", ".join(
                        f"{_icon.get(o.get('provenance'), '📜')} `{o['id']}` "
                        f"({o.get('technique')})" for o in related[:6])
                    + (" …" if len(related) > 6 else "")
                    + " — select them there to include."
                )

    # ── per-domain action bar operating on the SELECTION ──
    for domain in sorted({d for (d, _) in groups}):
        domain_recs = [r for r in all_staged if r["domain"] == domain]
        selected = [r for r in domain_recs
                    if st.session_state.get(_sel_key(domain, r["id"]))]
        st.markdown(f"**Distill selection** — `{domain}`")
        if not selected:
            st.caption("Nothing selected — tick records above.")
            continue
        counts = {"📜": 0, "🐛": 0, "💬": 0}
        for r in selected:
            counts[_icon.get(r.get("provenance"), "📜")] += 1
        st.caption(f"{len(selected)} selected — "
                   + " ".join(f"{k}{v}" for k, v in counts.items() if v))
        if model is None:
            st.info("Start a session to enable distillation (needs a model).")
            continue

        sel_ids = [r["id"] for r in selected]
        need = _staging.consolidate_min_n()
        c1, c2 = st.columns(2)

        with c1:
            st.caption("**Consolidate** — the selection becomes a **new** skill")
            default_label = max(
                ((t, sum(1 for r in selected
                         if (r.get("technique") or "") == t))
                 for (_d, t) in groups if _d == domain),
                key=lambda x: x[1])[0] if selected else ""
            label = st.text_input("New skill technique label",
                                  value=default_label,
                                  key=f"conlabel::{domain}")
            import re as _re
            norm_label = _re.sub(r"[^a-z0-9]+", "_",
                                 (label or "").lower()).strip("_")[:48]
            swept = [r["id"] for r in domain_recs
                     if (r.get("technique") or "") == norm_label
                     and r["id"] not in sel_ids]
            if swept:
                st.warning(f"Label `{norm_label}` is already used by "
                           f"{len(swept)} UNSELECTED record(s) — they would "
                           f"be consolidated too. Select them or pick "
                           f"another label.")
            con_ok = (len(selected) >= need and norm_label and not swept
                      and mem_on)
            con_help = None
            if len(selected) < need:
                con_help = (f"Select at least {need} records "
                            f"(SCILINK_CONSOLIDATE_N) — one example is too "
                            f"idiosyncratic to generalize.")
            if st.button(f"Consolidate {len(selected)} → new skill "
                         f"(`auto_{norm_label or '…'}`)",
                         key=f"con::{domain}", disabled=not con_ok,
                         help=con_help or "One large model call — typically "
                                          "1–3 minutes."):
                from scilink.agents.exp_agents.instruct import (
                    T2_CONSOLIDATION_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS)
                for rid in sel_ids:
                    _staging.relabel_staged(domain, rid, norm_label)
                with st.spinner(
                        f"Distilling {len(selected)} records into "
                        f"`auto_{norm_label}` — one large model call, "
                        f"typically 1–3 min. Leave this tab open..."):
                    res = _staging.consolidate_technique(
                        domain, norm_label, llm_call=_llm_call,
                        consolidation_template=T2_CONSOLIDATION_INSTRUCTIONS,
                        update_template=SKILL_UPDATE_INSTRUCTIONS)
                st.success(f"Consolidated {res.get('n_examples')} → "
                           f"auto_{norm_label} (provisional — approve it in "
                           f"Skills below).")
                st.rerun()

        prop_key = f"upgprop::{domain}"
        with c2:
            st.caption("**Upgrade** — the selection folds into an "
                       "**existing** skill")
            persistent = {s["name"] for s in _memory.list_memory(domain=domain)}
            targets = [f"{domain}/{n}" for n in sorted(persistent)]
            from scilink.skills import loader as _loader
            builtin_only = [n for n in _loader.list_skills(domain)
                            if n not in persistent]
            targets += [f"{domain}/{n} (built-in — forks on upgrade)"
                        for n in sorted(builtin_only)]
            if not targets:
                st.caption("No skills in this domain to upgrade.")
            else:
                def _tname(t):
                    return t.split("/", 1)[1].split(" (built-in", 1)[0]

                def _group_match(t):
                    votes = [_target_matches_record(r, domain, _tname(t))
                             for r in selected]
                    if any(v is True for v in votes):
                        return True
                    if all(v is False for v in votes):
                        return False
                    return None
                matches = {t: _group_match(t) for t in targets}
                ordered = sorted(targets, key=lambda t: {True: 0, None: 1,
                                                         False: 2}[matches[t]])
                tgt = st.selectbox("Into skill", ordered, key=f"tgt::{domain}")
                if matches.get(tgt) is False:
                    st.warning(
                        f"⚠️ Technique mismatch: the selection's context "
                        f"doesn't match `{_tname(tgt)}`'s technique routing — "
                        f"upgrading would pollute an unrelated skill.")
                if st.button(f"Preview upgrade ({len(selected)} → {_tname(tgt)})",
                             key=f"up::{domain}", disabled=not mem_on,
                             help=("Builds the merged skill for review — one "
                                   "large model call, typically 1–3 min. "
                                   "Writes nothing." if mem_on else
                                   "Persistent memory is off.")):
                    from scilink.agents.exp_agents.instruct import (
                        KNOWLEDGE_TO_SKILL_INSTRUCTIONS, SKILL_UPDATE_INSTRUCTIONS)
                    td, tn = tgt.split("/", 1)
                    skills_root = None
                    builtin_target = None
                    if tn.endswith(" (built-in — forks on upgrade)"):
                        import shutil
                        import tempfile
                        from pathlib import Path as _P
                        from scilink.skills.loader import _SKILLS_DIR
                        tn = tn.split(" (built-in", 1)[0]
                        builtin_target = (td, tn)
                        tmp_root = _P(tempfile.mkdtemp(prefix="scilink_preview_"))
                        (tmp_root / td / tn).mkdir(parents=True)
                        shutil.copy(_SKILLS_DIR / td / tn / f"{tn}.md",
                                    tmp_root / td / tn / f"{tn}.md")
                        skills_root = tmp_root
                    with st.spinner(
                            f"Building the merged `{td}/{tn}` for review — "
                            f"one large model call, typically 1–3 min. "
                            f"Nothing is written until you Apply..."):
                        prop = _staging.propose_skill_upgrade(
                            domain, sel_ids, target_domain=td, target_name=tn,
                            llm_call=_llm_call,
                            fresh_template=KNOWLEDGE_TO_SKILL_INSTRUCTIONS,
                            update_template=SKILL_UPDATE_INSTRUCTIONS,
                            skills_root=skills_root)
                    if prop.get("status") == "success":
                        prop["builtin_target"] = builtin_target
                        if matches.get(tgt) is False:
                            prop.setdefault("regression_warnings", []).append(
                                "technique mismatch: the selection's context "
                                "does not match this skill's routing — "
                                "confirm this upgrade belongs here")
                        st.session_state[prop_key] = prop
                    else:
                        st.error(prop.get("message", "Could not build upgrade."))
                    st.rerun()

        # ── review block (upgrade proposal) ──
        prop = st.session_state.get(prop_key)
        if prop:
            import difflib
            st.markdown(
                f"**Review upgrade → `{prop['target_domain']}/{prop['target_name']}`** "
                "— applies in place; the current version is backed up to `.md.bak`."
            )
            final_content = prop["proposed_content"]
            if st.toggle("Edit proposal before applying",
                         key=f"upediton::{domain}"):
                final_content = st.text_area(
                    "Proposed skill (editable)", value=prop["proposed_content"],
                    height=400, key=f"upedit::{domain}")
            from scilink.skills._shared._staging import _regression_warnings
            for w in _regression_warnings(prop["existing_content"],
                                          final_content):
                st.warning(f"Additivity check: {w}")
            for w in prop.get("regression_warnings") or []:
                if "technique mismatch" in w:
                    st.warning(f"Additivity check: {w}")
            diff = "\n".join(difflib.unified_diff(
                prop["existing_content"].splitlines(),
                final_content.splitlines(),
                fromfile="current", tofile="after upgrade", lineterm=""))
            st.code(diff or "(no textual change)", language="diff")
            a1, a2, _ = st.columns([2, 2, 4])
            if a1.button("Apply upgrade", key=f"upapply::{domain}"):
                if prop.get("builtin_target"):
                    td, tn = prop["builtin_target"]
                    fout = _memory.fork_builtin(td, tn)
                    if fout.get("status") != "success" and \
                            "already forked" not in str(fout.get("message")):
                        st.error(fout.get("message", "Fork failed."))
                        st.rerun()
                res = _staging.apply_skill_upgrade(
                    domain, prop["staged_ids"],
                    target_domain=prop["target_domain"],
                    target_name=prop["target_name"],
                    proposed_content=final_content)
                st.session_state.pop(prop_key, None)
                if res.get("status") == "success":
                    st.success(f"Upgraded {prop['target_domain']}/"
                               f"{prop['target_name']} (backup saved).")
                else:
                    st.error(res.get("message", "Apply failed."))
                st.rerun()
            if a2.button("Cancel", key=f"upcancel::{domain}"):
                st.session_state.pop(prop_key, None)
                st.rerun()


# Streamlit renders popover/expander CONTENT eagerly on every rerun, open or
# not — with many memory records that meant re-reading every file and
# re-highlighting every script per chat interaction (the observed panel
# slowdown). Heavy content therefore loads only behind an explicit tick, and
# long lists paginate.
_PANEL_PAGE = 15
# Scripts render ONLY when a viewer is explicitly opened (lazy tick), so the
# full text is shown — the old 8k preview cap predates lazy loading and cut
# real ~10-15 KB scripts mid-file. Only a pathological-size guard remains.
_SCRIPT_HARD_CAP = 120_000


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
    if len(script) > _SCRIPT_HARD_CAP:
        st.code(script[:_SCRIPT_HARD_CAP]
                + f"\n# … truncated at {_SCRIPT_HARD_CAP // 1000} KB — "
                f"full script via {full_hint}",
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
    st.markdown("**1 · Script bank** — every success, recorded automatically")
    st.caption(
        "The start of the pipeline: each approved analysis banks its working "
        "script with a fingerprint of the data it solved. Later runs retrieve "
        "the closest match as a starting point, and real-time mode can "
        "cold-start a campaign from one. Records that keep succeeding across "
        "sessions (★) are evidence-backed skill candidates — nominate one to "
        "send it into the review inbox below (hard-won hot-annealing wins are "
        "nominated automatically)."
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
            _render_variant_group_suggestions(domain)
            for r in _paged(recs, f"bankpage::{domain}"):
                metric = r["metric"]
                mtxt = (f" · {metric['name']}={metric['value']}"
                        if isinstance(metric, dict) and metric.get("value") is not None
                        else "")
                badge = "★ proven" if r["proven"] else \
                    f"{r['n_successes']}/{threshold} sessions"
                if r["promoted_to_staging"]:
                    badge += f" · ✓ in review inbox (`{r['promoted_to_staging']}`)"
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
                        "Nominate for review",
                        key=f"bankprom::{domain}/{r['id']}",
                        disabled=bool(r["promoted_to_staging"]),
                        help=("Already in the review inbox." if r["promoted_to_staging"] else
                              "Sends this script to the review inbox below, "
                              "where it can be distilled into a skill. The "
                              "bank record is kept."),
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


def _render_variant_group_suggestions(domain: str) -> None:
    """Surface same-system variant clusters as one-click group promotions.

    N different successful treatments of one system are exactly what skill
    consolidation needs to generalize from — but only if they reach the
    staging buffer under ONE technique label. This suggestion does the
    grouping (by fingerprint similarity) and the shared label for the user.
    """
    from scilink.skills._shared import _script_bank

    try:
        groups = [g for g in _script_bank.find_variant_groups(domain)
                  if g["n_unpromoted"] > 0]
    except Exception:  # noqa: BLE001 - suggestions must never break the panel
        return
    for g in groups:
        ids = g["ids"]
        with st.container(border=True):
            st.markdown(
                f"💡 **{len(ids)} records look like the same system** "
                f"(min pairwise similarity {g['min_similarity']}): "
                + ", ".join(f"`{i}`" for i in ids)
            )
            st.caption(
                "Nominate them together under one technique label so the "
                "review-gated consolidation can distill a general skill "
                "from all variants at once."
            )
            gkey = "::".join([domain] + ids)
            label = st.text_input(
                "Technique label", value=g["suggested_technique"],
                key=f"bankgrouplabel::{gkey}")
            if st.button(f"Nominate {len(ids)} as one technique",
                         key=f"bankgroup::{gkey}"):
                out = _script_bank.promote_group_to_staging(
                    domain, ids, technique=label or None)
                if out.get("status") == "success":
                    msg = (f"Staged {len(out['staged_ids'])} under "
                           f"[{out['technique']}].")
                    if out.get("ready_to_consolidate"):
                        msg += " Ready to consolidate in Staged knowledge above."
                    else:
                        msg += (f" {out['n_staged_total']} staged so far — "
                                f"consolidation is suggested at "
                                f"{_consolidate_min_n_ui()}.")
                    st.success(msg)
                else:
                    st.error(out.get("message", "Group promotion failed."))
                st.rerun()


def _consolidate_min_n_ui() -> int:
    from scilink.skills._shared import _staging
    return _staging.consolidate_min_n()


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


_SKILL_TOKEN_CACHE: dict = {}

# Family words shared by many techniques — useless for telling a Raman
# record from an IR skill, and they drown the distinctive tokens.
_GENERIC_TECH_TOKENS = {"spectroscopy", "spectrometry", "spectrum", "spectra",
                        "microscopy", "micro", "imaging", "absorption",
                        "emission", "analysis", "scattering"}


def _skill_tokens(domain: str, name: str):
    """Technique/name/description tokens of a skill (cached per session)."""
    import re
    key = (domain, name)
    if key in _SKILL_TOKEN_CACHE:
        return _SKILL_TOKEN_CACHE[key]
    tech_tokens, all_tokens = set(), set()
    try:
        from scilink.skills import loader
        parsed = loader.load_skill(name, domain=domain)
        meta = parsed.get("meta") or {}
        tech = meta.get("technique") or []
        if isinstance(tech, str):
            tech = [tech]
        for t in tech:
            tech_tokens |= {w for w in re.findall(r"[a-z0-9]+", str(t).lower())
                            if len(w) >= 3 and w not in _GENERIC_TECH_TOKENS}
        for s in [name, meta.get("description") or ""]:
            all_tokens |= {w for w in re.findall(r"[a-z0-9]+", str(s).lower())
                           if len(w) >= 3}
    except Exception:  # noqa: BLE001
        pass
    _SKILL_TOKEN_CACHE[key] = (tech_tokens, all_tokens | tech_tokens)
    return _SKILL_TOKEN_CACHE[key]


def _record_tokens(rec: dict):
    import re
    toks = set()
    ctx = rec.get("measurement_context") or {}
    fields = list(ctx.values()) + [
        rec.get("model"), rec.get("technique"), rec.get("planned_model"),
        rec.get("final_model_type"), rec.get("analysis_target")]
    for v in fields:
        toks |= {w for w in re.findall(r"[a-z0-9]+", str(v or "").lower())
                 if len(w) >= 3}
    return toks


def _target_matches_record(rec: dict, domain: str, skill_name: str):
    """True / False / None(unknown): does this record's technique plausibly
    belong to this skill? Judged on the skill's `technique:` routing tokens
    when present (e.g. a Raman-context record vs the epr skill → False), so
    a mismatched upgrade is flagged BEFORE the expensive preview call."""
    tech_tokens, all_tokens = _skill_tokens(domain, skill_name)
    rec_toks = _record_tokens(rec)
    if not rec_toks:
        return None
    if tech_tokens:
        return bool(rec_toks & tech_tokens)
    if all_tokens:
        return bool(rec_toks & all_tokens) or None
    return None


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
    badge = "🟡 provisional" if provisional else "✅ approved"
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

        # Manual curation: persistent skills are the user's own store, so a
        # direct edit (validated, with a .bak backup) is fair game. Built-ins
        # never appear here — improving one goes through fork + upgrade.
        if st.toggle("Edit skill", key=f"editon::{ref}"):
            from scilink.skills.loader import graduated_skills_dir
            md_path = (graduated_skills_dir() / r["domain"] / r["name"]
                       / f"{r['name']}.md")
            try:
                current = md_path.read_text()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read skill file: {e}")
                current = None
            if current is not None:
                edited = st.text_area("Skill markdown", value=current,
                                      height=400, key=f"edit::{ref}")
                if st.button("Save changes", key=f"editsave::{ref}",
                             disabled=(edited == current),
                             help="Validates frontmatter/sections; the "
                                  "previous version is backed up to .md.bak."):
                    out = _memory.edit_memory(r["domain"], r["name"], edited)
                    if out.get("status") == "success":
                        st.success(f"Saved {ref} (backup: .md.bak).")
                        st.rerun()
                    else:
                        st.error(out.get("message", "Save failed."))

        from scilink.skills import loader
        mem_on = loader.memory_enabled()
        c1, c2, _ = st.columns([2, 2, 4])
        if provisional:
            if c1.button("Approve for routing", key=f"promote::{ref}",
                         disabled=not mem_on,
                         help=(None if mem_on else
                               "Persistent memory is off — approved skills won't load "
                               "until you turn it on.")):
                _memory.promote_memory(r["domain"], r["name"])
                st.success(f"Approved {ref} — now auto-routable.")
                st.rerun()
        else:
            if c1.button("Suspend (provisional)", key=f"demote::{ref}",
                         help=("Set back to provisional — taken out of the "
                               "auto-routing menu (still explicitly loadable) "
                               "until you approve it again.")):
                _memory.demote_memory(r["domain"], r["name"])
                st.warning(f"Suspended {ref} — provisional again.")
                st.rerun()
        if c2.button("Delete", key=f"prune::{ref}"):
            _memory.prune_memory(r["domain"], r["name"])
            st.warning(f"Pruned {ref}.")
            st.rerun()
