import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type MemoryBankRecord,
  type MemoryInboxGroup,
  type MemoryInboxRecord,
  type MemoryInboxRow,
  type MemoryOverview,
  type MemoryProposal,
  type MemorySkill,
  type MemoryTarget,
} from "../api";
import { MarkdownBody } from "./MarkdownBody";

/** Persistent memory — the port of the Streamlit memory panel over the
 * `/api/v1/memory` endpoints. One store per server host (`~/.scilink`),
 * shared by every session. The pipeline reads top to bottom in the order
 * knowledge flows:
 *
 *   1 · Script bank   every approved analysis banks its working script;
 *                     proven records (★) can be nominated for review
 *   2 · Review inbox  nominations, error lessons and your feedback wait
 *                     here; distill a selection into a NEW skill or into
 *                     an EXISTING one (one large LLM call, run as a
 *                     background job and polled)
 *   3 · Skills        the curated end: provisional skills wait for your
 *                     approval; approved ones auto-route; a fork of a
 *                     built-in shadows the shipped copy
 *
 * Every destructive action is a two-click confirm (no browser dialogs). */

const PROV_ICON: Record<string, string> = {
  error_fix: "🐛",
  user_correction: "💬",
};

function normalizeLabel(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 48);
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** A button whose first click arms it ("confirm?") and second click acts;
 * it disarms after three seconds. */
function ConfirmButton({
  label,
  confirmLabel = "confirm?",
  onConfirm,
  className = "link-btn danger",
  disabled,
  title,
}: {
  label: string;
  confirmLabel?: string;
  onConfirm: () => void;
  className?: string;
  disabled?: boolean;
  title?: string;
}) {
  const [armed, setArmed] = useState(false);
  useEffect(() => {
    if (!armed) return;
    const t = setTimeout(() => setArmed(false), 3000);
    return () => clearTimeout(t);
  }, [armed]);
  return (
    <button
      type="button"
      className={className + (armed ? " armed" : "")}
      disabled={disabled}
      title={title}
      onClick={() => {
        if (armed) {
          setArmed(false);
          onConfirm();
        } else setArmed(true);
      }}
    >
      {armed ? confirmLabel : label}
    </button>
  );
}

function DiffView({ diff }: { diff: string }) {
  if (!diff) return <pre className="mem-diff">(no textual change)</pre>;
  return (
    <pre className="mem-diff">
      {diff.split("\n").map((l, i) => (
        <span
          key={i}
          className={
            l.startsWith("+") && !l.startsWith("+++") ? "add"
              : l.startsWith("-") && !l.startsWith("---") ? "del"
              : l.startsWith("@@") ? "hunk" : ""
          }
        >
          {l}
          {"\n"}
        </span>
      ))}
    </pre>
  );
}

function Fields({ fields }: { fields: Record<string, unknown> }) {
  return (
    <dl className="mem-fields">
      {Object.entries(fields).map(([k, v]) => (
        <div key={k}>
          <dt>{k.replace(/_/g, " ")}</dt>
          <dd>{typeof v === "string" ? v : JSON.stringify(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

export function MemoryPanel({ sessionId, active }: { sessionId: string; active: boolean }) {
  const [ov, setOv] = useState<MemoryOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ kind: "ok" | "warn"; text: string } | null>(null);

  const refresh = useCallback(async () => {
    try {
      setOv(await api.memory());
      setError(null);
    } catch (e) {
      setError(errText(e));
    }
  }, []);

  useEffect(() => {
    if (active) void refresh();
  }, [active, refresh]);

  const notify = useCallback((kind: "ok" | "warn", text: string) => {
    setNotice({ kind, text });
  }, []);

  if (error) {
    return (
      <section className="tools-section">
        <h3>Persistent memory</h3>
        <p className="caption warn">Could not read persistent memory: {error}</p>
      </section>
    );
  }
  if (!ov) return null;
  const p = ov.pipeline;
  const memOn = ov.enabled;

  return (
    <section className="tools-section mem-panel">
      <h3>Persistent memory</h3>
      <p className="caption">
        Graduated and auto-distilled skills stored under <code>{ov.home}</code> — they
        survive sessions and upgrades. Provisional skills (auto-distilled from hard problems
        the agent had to solve from scratch) are held out of auto-routing until you approve them.
      </p>

      <label className="mem-switch">
        <input
          type="checkbox"
          checked={memOn}
          disabled={Boolean(ov.env_override)}
          onChange={async (e) => {
            try {
              await api.setMemoryEnabled(e.target.checked);
              await refresh();
            } catch (err) {
              notify("warn", errText(err));
            }
          }}
        />
        <span>Enable persistent memory</span>
        <span className="caption">
          {memOn
            ? "on: auto-staging + reuse of approved skills"
            : "off (default): inert — nothing is staged and approved skills are not loaded"}
        </span>
      </label>
      {ov.env_override && (
        <p className="caption warn">
          ⚠️ <code>SCILINK_MEMORY={ov.env_override}</code> is set in the server's environment and
          overrides this switch.
        </p>
      )}
      {!memOn && (
        <p className="caption mem-info">
          Persistent memory is <strong>OFF</strong>. Turn it on to capture hard-won solutions, your
          feedback and error fixes, and to load approved skills into runs. Existing items below are
          kept and can still be reviewed.
        </p>
      )}

      <p className="mem-pipeline">
        Script bank ({p.bank_total}{p.bank_proven ? `, ★${p.bank_proven} proven` : ""}) →
        Review inbox ({p.inbox_total}{p.inbox_ready ? `, ${p.inbox_ready} ready to distill` : ""}) →
        Skills ({p.skills_total}{p.skills_provisional ? `, ${p.skills_provisional} provisional` : ""})
      </p>

      {notice && (
        <p className={`mem-notice ${notice.kind}`}>
          {notice.text}
          <button type="button" className="icon-btn" title="Dismiss" onClick={() => setNotice(null)}>
            ✕
          </button>
        </p>
      )}

      <BankSection ov={ov} onChange={refresh} notify={notify} />
      <InboxSection ov={ov} sessionId={sessionId} onChange={refresh} notify={notify} />
      <SkillsSection ov={ov} onChange={refresh} notify={notify} />
    </section>
  );
}

/* ── 1 · script bank ─────────────────────────────────────────────── */

function BankSection({
  ov,
  onChange,
  notify,
}: {
  ov: MemoryOverview;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
}) {
  const [viewing, setViewing] = useState<Record<string, MemoryBankRecord | "loading">>({});
  const [groupLabels, setGroupLabels] = useState<Record<string, string>>({});

  const toggleView = async (domain: string, id: string) => {
    const key = `${domain}/${id}`;
    if (viewing[key]) {
      setViewing((v) => { const n = { ...v }; delete n[key]; return n; });
      return;
    }
    setViewing((v) => ({ ...v, [key]: "loading" }));
    try {
      const rec = await api.memoryBankRecord(domain, id);
      setViewing((v) => ({ ...v, [key]: rec }));
    } catch (e) {
      notify("warn", errText(e));
      setViewing((v) => { const n = { ...v }; delete n[key]; return n; });
    }
  };

  return (
    <div className="mem-stage">
      <h4>1 · Script bank <span className="caption">— every success, recorded automatically</span></h4>
      <p className="caption">
        Each approved analysis banks its working script with a fingerprint of the data it solved;
        later runs retrieve the closest match as a starting point. Records that keep succeeding
        across sessions (★, {ov.proven_n} or more) are evidence-backed skill candidates — nominate one
        to send it into the review inbox below.
      </p>
      {ov.bank.length === 0 && (
        <p className="caption">No banked scripts yet — they accumulate as analyses succeed.</p>
      )}
      {ov.bank.map((d) => (
        <details key={d.domain} className="mem-group">
          <summary>
            <code>{d.domain}</code> — {d.records.length} banked
            {d.n_proven ? `, ★ ${d.n_proven} proven` : ""}
          </summary>
          {d.variant_groups.map((g) => {
            const gkey = `${d.domain}::${g.ids.join(",")}`;
            const label = groupLabels[gkey] ?? g.suggested_technique ?? "";
            return (
              <div key={gkey} className="mem-card">
                <div>
                  💡 <strong>{g.ids.length} records look like the same system</strong>
                  <span className="caption"> (min pairwise similarity {g.min_similarity}): </span>
                  {g.ids.map((i) => <code key={i}>{i} </code>)}
                </div>
                <p className="caption">
                  Nominate them together under one technique label so consolidation can distill a
                  general skill from all variants at once.
                </p>
                <div className="mem-row-actions">
                  <input
                    type="text"
                    value={label}
                    placeholder="technique label"
                    onChange={(e) => setGroupLabels((m) => ({ ...m, [gkey]: e.target.value }))}
                  />
                  <button
                    type="button"
                    className="link-btn"
                    onClick={async () => {
                      try {
                        const out = await api.memoryBankNominateGroup(d.domain, g.ids, label || null);
                        notify("ok", `Staged ${out.staged_ids.length} under [${out.technique}]` +
                          (out.ready_to_consolidate ? " — ready to consolidate in the review inbox." : "."));
                        await onChange();
                      } catch (e) {
                        notify("warn", errText(e));
                      }
                    }}
                  >
                    Nominate {g.ids.length} as one technique
                  </button>
                </div>
              </div>
            );
          })}
          {d.records.map((r) => {
            const key = `${d.domain}/${r.id}`;
            const v = viewing[key];
            const metric = r.metric && typeof r.metric === "object" && (r.metric as { value?: unknown }).value != null
              ? ` · ${(r.metric as { name?: string }).name ?? "metric"}=${String((r.metric as { value?: unknown }).value)}`
              : typeof r.metric === "number" ? ` · metric=${r.metric}` : "";
            return (
              <div key={r.id} className="mem-row">
                <div className="mem-row-main">
                  <span className="caption">
                    id={r.id} · {r.proven ? "★ proven" : `${r.n_successes}/${ov.proven_n} sessions`}
                    {r.promoted_to_staging ? ` · ✓ in review inbox (${r.promoted_to_staging})` : ""}
                    {" · "}retrieved {r.n_retrievals}×{metric}
                  </span>
                  <div className="mem-label">{r.label}</div>
                </div>
                <div className="mem-row-actions">
                  <button type="button" className="link-btn" onClick={() => void toggleView(d.domain, r.id)}>
                    {v ? "hide" : "view"}
                  </button>
                  <button
                    type="button"
                    className="link-btn"
                    disabled={Boolean(r.promoted_to_staging)}
                    title={r.promoted_to_staging ? "Already in the review inbox." :
                      "Sends this script to the review inbox, where it can be distilled into a skill. The bank record is kept."}
                    onClick={async () => {
                      try {
                        const out = await api.memoryBankNominate(d.domain, r.id);
                        notify("ok", `Staged as ${out.staged_id} [${out.technique}] — review it in the inbox below.`);
                        await onChange();
                      } catch (e) {
                        notify("warn", errText(e));
                      }
                    }}
                  >
                    nominate for review
                  </button>
                  <ConfirmButton
                    label="delete"
                    onConfirm={async () => {
                      try {
                        await api.memoryBankDelete(d.domain, r.id);
                        await onChange();
                      } catch (e) {
                        notify("warn", errText(e));
                      }
                    }}
                  />
                </div>
                {v === "loading" && <p className="caption">loading…</p>}
                {v && v !== "loading" && (
                  <div className="mem-detail">
                    <Fields fields={v.fields} />
                    {v.script && (
                      <>
                        <div className="caption">working script:</div>
                        <pre className="tel-json mem-script">{v.script}</pre>
                      </>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </details>
      ))}
    </div>
  );
}

/* ── 2 · review inbox ────────────────────────────────────────────── */

function recordLabel(r: MemoryInboxRow, groupTechnique: string): string {
  let s = `${PROV_ICON[r.provenance] ?? "📜"} ${r.id} · ${r.provenance_label}`;
  if (r.metric) s += ` · ${r.metric}`;
  if (r.technique !== groupTechnique) s += ` (from ${r.technique})`;
  if (r.session) s += ` · ${r.session}`;
  return s;
}

function InboxSection({
  ov,
  sessionId,
  onChange,
  notify,
}: {
  ov: MemoryOverview;
  sessionId: string;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
}) {
  return (
    <div className="mem-stage">
      <h4>2 · Review inbox <span className="caption">— lessons awaiting your call</span></h4>
      <p className="caption">
        Three knowledge streams converge here: 📜 script nominations, 🐛 error lessons, 💬 your
        feedback. Select records and distill them into a <strong>new</strong> skill or into an{" "}
        <strong>existing</strong> one — either way they merge in one pass: method + pitfalls + your
        constraints.
      </p>
      {ov.inbox.length === 0 && <p className="caption">No staged records.</p>}
      {ov.inbox.map((g) => (
        <InboxGroup
          key={`${g.domain}/${g.technique}`}
          group={g}
          ov={ov}
          sessionId={sessionId}
          onChange={onChange}
          notify={notify}
        />
      ))}
    </div>
  );
}

function InboxGroup({
  group: g,
  ov,
  sessionId,
  onChange,
  notify,
}: {
  group: MemoryInboxGroup;
  ov: MemoryOverview;
  sessionId: string;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
}) {
  const pool = useMemo(() => [...g.records, ...g.related], [g]);
  const [selected, setSelected] = useState<Set<string>>(() => new Set(g.records.map((r) => r.id)));
  const [inspect, setInspect] = useState<Record<string, MemoryInboxRecord | "loading">>({});
  const [dest, setDest] = useState<"new" | "existing">("new");
  const [label, setLabel] = useState(g.technique);
  const [targets, setTargets] = useState<MemoryTarget[] | null>(null);
  const [target, setTarget] = useState<string>("");
  const [job, setJob] = useState<{ id: string; label: string; started: number; kind: "consolidate" | "upgrade" } | null>(null);
  const [proposal, setProposal] = useState<MemoryProposal | null>(null);
  const [editing, setEditing] = useState(false);
  const [edited, setEdited] = useState("");
  const [check, setCheck] = useState<{ warnings: string[]; diff: string } | null>(null);
  const checkTimer = useRef<number | null>(null);
  const [, setTick] = useState(0);

  // Records may change under us (a nomination, a discard): keep the
  // selection to what still exists, default-select the group's own rows.
  useEffect(() => {
    setSelected((s) => {
      const ids = new Set(pool.map((r) => r.id));
      const next = new Set([...s].filter((i) => ids.has(i)));
      for (const r of g.records) if (!s.size || !next.size) next.add(r.id);
      return next;
    });
  }, [pool, g.records]);

  const selectedIds = useMemo(() => pool.filter((r) => selected.has(r.id)).map((r) => r.id), [pool, selected]);
  const need = ov.consolidate_min_n;
  const memOn = ov.enabled;

  // targets for "an existing skill": fetched when that destination is
  // chosen and re-fetched when the selection changes (the match verdict
  // depends on which records are selected)
  useEffect(() => {
    if (dest !== "existing") return;
    let cancelled = false;
    api.memoryTargets(g.domain, selectedIds).then((t) => {
      if (cancelled) return;
      setTargets(t);
      setTarget((cur) => (t.some((x) => x.name === cur) ? cur : (t[0]?.name ?? "")));
    }).catch((e) => notify("warn", errText(e)));
    return () => { cancelled = true; };
  }, [dest, g.domain, selectedIds, notify]);

  // poll a running job
  useEffect(() => {
    if (!job) return;
    const t = setInterval(async () => {
      setTick((n) => n + 1);
      try {
        const j = await api.memoryJob(job.id);
        if (j.status === "running") return;
        setJob(null);
        if (j.status === "error") {
          notify("warn", job.kind === "consolidate"
            ? `Consolidation failed — no skill was written; the records are unchanged. Try again. (${j.error})`
            : `Could not build the upgrade — nothing was written. Try again. (${j.error})`);
          return;
        }
        if (job.kind === "consolidate") {
          const r = j.result as { n_examples?: number; skill_name?: string };
          notify("ok", `Consolidated ${r.n_examples ?? ""} → ${r.skill_name} (provisional — approve it in Skills below).`);
          await onChange();
        } else {
          const prop = j.result as MemoryProposal;
          setProposal(prop);
          setEdited(prop.proposed_content);
          setCheck({ warnings: prop.warnings, diff: prop.diff });
          setEditing(false);
        }
      } catch (e) {
        setJob(null);
        notify("warn", errText(e));
      }
    }, 2000);
    return () => clearInterval(t);
  }, [job, notify, onChange]);

  // re-check an edited proposal (debounced)
  useEffect(() => {
    if (!proposal || !editing) return;
    if (checkTimer.current) window.clearTimeout(checkTimer.current);
    checkTimer.current = window.setTimeout(async () => {
      try {
        setCheck(await api.memoryCheckUpgrade(proposal.existing_content, edited));
      } catch { /* keep the last check */ }
    }, 600);
    return () => { if (checkTimer.current) window.clearTimeout(checkTimer.current); };
  }, [edited, editing, proposal]);

  const toggleInspect = async (id: string) => {
    if (inspect[id]) {
      setInspect((m) => { const n = { ...m }; delete n[id]; return n; });
      return;
    }
    setInspect((m) => ({ ...m, [id]: "loading" }));
    try {
      const rec = await api.memoryInboxRecord(g.domain, id);
      setInspect((m) => ({ ...m, [id]: rec }));
    } catch (e) {
      notify("warn", errText(e));
      setInspect((m) => { const n = { ...m }; delete n[id]; return n; });
    }
  };

  const norm = normalizeLabel(label);
  const swept = ov.inbox
    .filter((x) => x.domain === g.domain && x.technique === norm)
    .flatMap((x) => x.records.map((r) => r.id))
    .filter((id) => !selected.has(id));
  const canConsolidate = memOn && selectedIds.length >= need && Boolean(norm) && swept.length === 0 && !job;
  const tgt = targets?.find((t) => t.name === target) ?? null;
  const elapsed = job ? Math.round((Date.now() - job.started) / 1000) : 0;

  return (
    <details className="mem-group">
      <summary>
        <code>{g.domain}/{g.technique}</code> — {g.records.length} staged
        {g.ready && <span className="tel-status status-success">ready to distill</span>}
      </summary>

      {pool.map((r) => {
        const ins = inspect[r.id];
        return (
          <div key={r.id} className={`mem-row${r.technique !== g.technique ? " related" : ""}`}>
            <div className="mem-row-main">
              <label className="mem-check">
                <input
                  type="checkbox"
                  checked={selected.has(r.id)}
                  onChange={(e) => setSelected((s) => {
                    const n = new Set(s);
                    if (e.target.checked) n.add(r.id); else n.delete(r.id);
                    return n;
                  })}
                />
                <span>{recordLabel(r, g.technique)}</span>
              </label>
            </div>
            <div className="mem-row-actions">
              <button type="button" className="link-btn" onClick={() => void toggleInspect(r.id)}>
                {ins ? "hide" : "inspect"}
              </button>
              <ConfirmButton
                label="discard"
                title="Remove from the review inbox without distilling it. A nominated bank record becomes nominatable again."
                onConfirm={async () => {
                  try {
                    await api.memoryInboxDiscard(g.domain, r.id);
                    notify("warn", `De-staged ${r.id}.`);
                    await onChange();
                  } catch (e) {
                    notify("warn", errText(e));
                  }
                }}
              />
            </div>
            {ins === "loading" && <p className="caption">loading…</p>}
            {ins && ins !== "loading" && (
              <div className="mem-detail">
                {ins.bank && (
                  <p className="caption">
                    from bank <code>{ins.bank.bank_id}</code>
                    {ins.bank.n_successes && ins.bank.n_successes > 1
                      ? ` (succeeded in ${ins.bank.n_successes} sessions)` : ""}
                  </p>
                )}
                <Fields fields={ins.fields} />
                {ins.script && (
                  <>
                    <div className="caption">working script:</div>
                    <pre className="tel-json mem-script">{ins.script}</pre>
                  </>
                )}
              </div>
            )}
          </div>
        );
      })}

      <div className="mem-distill">
        <strong>Distill</strong>
        <span className="caption"> {selectedIds.length} selected</span>
        <div className="mem-dest">
          <label><input type="radio" checked={dest === "new"} onChange={() => setDest("new")} /> a new skill</label>
          <label><input type="radio" checked={dest === "existing"} onChange={() => setDest("existing")} /> an existing skill</label>
        </div>
        {!memOn && <p className="caption warn">Persistent memory is off — turn it on to distill.</p>}

        {dest === "new" ? (
          <div className="mem-row-actions">
            <input type="text" value={label} placeholder="new skill name" onChange={(e) => setLabel(e.target.value)} />
            <button
              type="button"
              className="primary small"
              disabled={!canConsolidate}
              title={canConsolidate ? "One large model call — typically 1–3 minutes."
                : selectedIds.length < need ? `Needs at least ${need} selected records (one example is too idiosyncratic to generalize).`
                : undefined}
              onClick={async () => {
                try {
                  const j = await api.memoryConsolidate(g.domain, selectedIds, label, sessionId);
                  setJob({ id: j.job_id, label: j.label, started: Date.now(), kind: "consolidate" });
                } catch (e) {
                  notify("warn", errText(e));
                }
              }}
            >
              Consolidate {selectedIds.length} → auto_{norm || "…"}
            </button>
          </div>
        ) : (
          <div className="mem-row-actions">
            {targets === null ? (
              <span className="caption">loading targets…</span>
            ) : targets.length === 0 ? (
              <span className="caption">No skills in this domain to upgrade.</span>
            ) : (
              <select value={target} onChange={(e) => setTarget(e.target.value)}>
                {targets.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.match === true ? "✓ " : t.match === false ? "⚠ " : "· "}
                    {t.domain}/{t.name}{t.builtin ? " (built-in — forks on upgrade)" : ""}
                  </option>
                ))}
              </select>
            )}
            <button
              type="button"
              className="primary small"
              disabled={!memOn || !selectedIds.length || !tgt || Boolean(job)}
              title={memOn ? "Builds the merged skill for review — one large model call, typically 1–3 min. Writes nothing."
                : "Persistent memory is off."}
              onClick={async () => {
                if (!tgt) return;
                try {
                  const j = await api.memoryProposeUpgrade(g.domain, selectedIds, tgt.domain, tgt.name, sessionId);
                  setJob({ id: j.job_id, label: j.label, started: Date.now(), kind: "upgrade" });
                } catch (e) {
                  notify("warn", errText(e));
                }
              }}
            >
              Preview upgrade ({selectedIds.length} → {tgt?.name ?? "…"})
            </button>
          </div>
        )}
        {dest === "new" && swept.length > 0 && (
          <p className="caption warn">
            <code>{norm}</code> is also the label of {swept.length} unselected record(s) — they would be
            included. Select them or rename.
          </p>
        )}
        {dest === "existing" && tgt?.match === false && (
          <p className="caption warn">
            ⚠️ Technique mismatch: the selection doesn't match <code>{tgt.name}</code>'s technique
            routing — upgrading would pollute an unrelated skill.
          </p>
        )}
        {job && (
          <p className="mem-job">
            <span className="spinner" /> {job.kind === "consolidate" ? "Distilling" : "Building the merged skill for review"}{" "}
            {job.label} — typically 1–3 min, {elapsed}s so far. You can keep working; this box updates on its own.
          </p>
        )}
      </div>

      {proposal && (
        <div className="mem-review">
          <strong>
            Review upgrade → <code>{proposal.target_domain}/{proposal.target_name}</code>
          </strong>
          <span className="caption">
            {" "}— applies in place; the current version is backed up to <code>.md.bak</code>
            {proposal.builtin_target ? "; the built-in is forked into the store first" : ""}.
          </span>
          <label className="mem-check">
            <input type="checkbox" checked={editing} onChange={(e) => setEditing(e.target.checked)} />
            <span>Edit proposal before applying</span>
          </label>
          {editing && (
            <textarea className="mem-editor" value={edited} onChange={(e) => setEdited(e.target.value)} rows={18} />
          )}
          {(check?.warnings ?? []).map((w, i) => (
            <p key={i} className="caption warn">Additivity check: {w}</p>
          ))}
          <DiffView diff={check?.diff ?? proposal.diff} />
          <div className="mem-row-actions">
            <button
              type="button"
              className="primary small"
              onClick={async () => {
                try {
                  await api.memoryApplyUpgrade(g.domain, proposal.staged_ids, proposal.target_domain,
                    proposal.target_name, editing ? edited : proposal.proposed_content, proposal.builtin_target);
                  notify("ok", `Upgraded ${proposal.target_domain}/${proposal.target_name} (backup saved).`);
                  setProposal(null);
                  await onChange();
                } catch (e) {
                  notify("warn", errText(e));
                }
              }}
            >
              Apply upgrade
            </button>
            <button type="button" className="link-btn" onClick={() => setProposal(null)}>Cancel</button>
          </div>
        </div>
      )}
    </details>
  );
}

/* ── 3 · skills ──────────────────────────────────────────────────── */

function SkillsSection({
  ov,
  onChange,
  notify,
}: {
  ov: MemoryOverview;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
}) {
  const provisional = ov.skills.filter((s) => s.provisional);
  const approved = ov.skills.filter((s) => !s.provisional);
  return (
    <div className="mem-stage">
      <h4>3 · Skills <span className="caption">— curated knowledge that guides planning</span></h4>
      <p className="caption">
        The end of the pipeline: reviewed knowledge with planning-layer authority. Approved skills
        auto-route; provisional ones wait for your call; a fork of a built-in shadows the shipped copy.
      </p>
      {provisional.length > 0 && <div className="mem-subhead">Provisional — awaiting review ({provisional.length})</div>}
      {provisional.map((s) => (
        <SkillRow key={`${s.domain}/${s.name}`} skill={s} memOn={ov.enabled} onChange={onChange} notify={notify} />
      ))}
      {approved.length > 0 && <div className="mem-subhead">Approved for routing ({approved.length})</div>}
      {approved.map((s) => (
        <SkillRow key={`${s.domain}/${s.name}`} skill={s} memOn={ov.enabled} onChange={onChange} notify={notify} />
      ))}
      {ov.skills.length === 0 && (
        <p className="caption">No skills yet — they are distilled from the review inbox above.</p>
      )}
    </div>
  );
}

function SkillRow({
  skill: s,
  memOn,
  onChange,
  notify,
}: {
  skill: MemorySkill;
  memOn: boolean;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
}) {
  const ref = `${s.domain}/${s.name}`;
  const [text, setText] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [diff, setDiff] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const load = async () => api.memorySkillText(s.domain, s.name);
  const front = text ? /^---\n([\s\S]*?)\n---\n?/.exec(text) : null;

  return (
    <details className="mem-group mem-skill">
      <summary>
        <span className={`tel-status ${s.provisional ? "status-pending" : "status-success"}`}>
          {s.provisional ? "provisional" : "approved"}
        </span>
        <code>{ref}</code>
        {s.metric && <span className="caption">{s.metric}</span>}
        {s.shadows_builtin && <span className="deleg-tag" title="A fork: shadows the shipped built-in of the same name">fork of built-in</span>}
      </summary>
      {s.description && <p className="mem-desc"><em>{s.description}</em></p>}
      {s.provenance && (
        <p className="caption">provenance: {s.provenance}{s.session ? ` · session: ${s.session}` : ""}</p>
      )}
      <div className="mem-row-actions">
        <button
          type="button"
          className="link-btn"
          onClick={async () => {
            if (text) { setText(null); return; }
            try { setText(await load()); } catch (e) { notify("warn", errText(e)); }
          }}
        >
          {text ? "hide" : "view"}
        </button>
        <button
          type="button"
          className="link-btn"
          onClick={async () => {
            if (editing !== null) { setEditing(null); setSaveError(null); return; }
            try { setEditing(await load()); } catch (e) { notify("warn", errText(e)); }
          }}
        >
          {editing !== null ? "close editor" : "edit"}
        </button>
        {s.provisional ? (
          <button
            type="button"
            className="primary small"
            disabled={!memOn}
            title={memOn ? "Clears the provisional flag — the skill enters the auto-routing menu."
              : "Persistent memory is off — approved skills won't load until you turn it on."}
            onClick={async () => {
              try {
                await api.memorySkillAction(s.domain, s.name, "promote");
                notify("ok", `Approved ${ref} — now auto-routable.`);
                await onChange();
              } catch (e) { notify("warn", errText(e)); }
            }}
          >
            Approve for routing
          </button>
        ) : (
          <button
            type="button"
            className="link-btn"
            title="Set back to provisional — taken out of the auto-routing menu (still explicitly loadable) until you approve it again."
            onClick={async () => {
              try {
                await api.memorySkillAction(s.domain, s.name, "demote");
                notify("warn", `Suspended ${ref} — provisional again.`);
                await onChange();
              } catch (e) { notify("warn", errText(e)); }
            }}
          >
            Suspend (provisional)
          </button>
        )}
        {s.shadows_builtin && (
          <button
            type="button"
            className="link-btn"
            onClick={async () => {
              if (diff !== null) { setDiff(null); return; }
              try {
                const d = await api.memorySkillAction(s.domain, s.name, "diff") as { diff: string; identical: boolean };
                setDiff(d.identical ? "" : d.diff);
              } catch (e) { notify("warn", errText(e)); }
            }}
          >
            {diff !== null ? "hide diff" : "diff vs built-in"}
          </button>
        )}
        <ConfirmButton
          label="delete"
          onConfirm={async () => {
            try {
              await api.memorySkillAction(s.domain, s.name, "prune");
              notify("warn", `Pruned ${ref}.`);
              await onChange();
            } catch (e) { notify("warn", errText(e)); }
          }}
        />
      </div>
      {diff !== null && (diff ? <DiffView diff={diff} /> : <p className="caption">Identical to the shipped built-in.</p>)}
      {text && (
        <div className="mem-detail md-body">
          {front && <pre className="skill-front caption">{front[1].trim()}</pre>}
          <MarkdownBody text={front ? text.slice(front[0].length) : text} />
        </div>
      )}
      {editing !== null && (
        <div className="mem-detail">
          <textarea className="mem-editor" value={editing} onChange={(e) => setEditing(e.target.value)} rows={18} />
          {saveError && <p className="caption warn">{saveError}</p>}
          <div className="mem-row-actions">
            <button
              type="button"
              className="primary small"
              title="Validates frontmatter/sections; the previous version is backed up to .md.bak."
              onClick={async () => {
                try {
                  await api.memorySkillEdit(s.domain, s.name, editing);
                  notify("ok", `Saved ${ref} (backup: .md.bak).`);
                  setEditing(null);
                  setSaveError(null);
                  setText(null);
                  await onChange();
                } catch (e) { setSaveError(errText(e)); }
              }}
            >
              Save changes
            </button>
          </div>
        </div>
      )}
    </details>
  );
}
