import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type MemoryBankRecord,
  type MemoryBankRow,
  type MemoryEvidence,
  type MemoryInboxGroup,
  type MemoryInboxRecord,
  type MemoryInboxRow,
  type MemoryJobRow,
  type MemoryOverview,
  type MemoryProposal,
  type MemorySkill,
  type MemorySweepRow,
  type MemoryTarget,
} from "../api";
import { MarkdownBody } from "./MarkdownBody";

/** Persistent memory — its own tab over the `/api/v1/memory` endpoints.
 * One store per server host (`~/.scilink`), shared by every session. The
 * pipeline reads top to bottom in the order knowledge flows:
 *
 *   jobs           every distillation this server ran, with a link to what
 *                  it produced (state is per server process: a restart
 *                  forgets it, and the strip says so)
 *   1 · Script bank   every approved analysis banks its working script;
 *                     the evidence behind a record is shown, not counted
 *   2 · Review inbox  nominations, error lessons and your feedback wait
 *                     here; distill a selection into a NEW skill or into an
 *                     EXISTING one (a background job)
 *   3 · Skills        provisional skills wait for approval; approved ones
 *                     auto-route on their technique list; a fork shadows
 *                     the shipped copy; every edit keeps a backup
 *
 * Destructive actions are two-click confirms (no browser dialogs) and are
 * hidden on a shared server, where the API answers 403 for them. */

const PROV_ICON: Record<string, string> = {
  error_fix: "🐛",
  user_correction: "💬",
};

const JOBS_KEY = "scilink.memory.jobs.v1";

function normalizeLabel(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 48);
}

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function shortDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** A button whose first click arms it and second click acts. Armed, it
 * turns into a solid red "confirm" button and stays armed for eight
 * seconds — long enough to read it, short enough not to be forgotten. */
function ConfirmButton({
  label,
  confirmLabel = "confirm delete",
  onConfirm,
  disabled,
  title,
}: {
  label: string;
  confirmLabel?: string;
  onConfirm: () => void;
  disabled?: boolean;
  title?: string;
}) {
  const [armed, setArmed] = useState(false);
  const [left, setLeft] = useState(8);
  useEffect(() => {
    if (!armed) return;
    setLeft(8);
    const tick = setInterval(() => setLeft((n) => n - 1), 1000);
    const t = setTimeout(() => setArmed(false), 8000);
    return () => { clearTimeout(t); clearInterval(tick); };
  }, [armed]);
  return (
    <button
      type="button"
      className={armed ? "mem-confirm armed" : "link-btn danger"}
      disabled={disabled}
      title={armed ? "Click again to confirm; this cannot be undone." : title}
      onClick={() => {
        if (armed) {
          setArmed(false);
          onConfirm();
        } else setArmed(true);
      }}
    >
      {armed ? `${confirmLabel} (${left}s)` : label}
    </button>
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

function dataSummary(d: Record<string, unknown> | null | undefined): string {
  if (!d) return "";
  const parts: string[] = [];
  if (d.kind) parts.push(String(d.kind));
  if (d.n_points != null) parts.push(`${d.n_points} pts`);
  if (Array.isArray(d.shape)) parts.push(`${(d.shape as unknown[]).join("×")}`);
  if (d.peaks != null) parts.push(`${d.peaks} peak${d.peaks === 1 ? "" : "s"}`);
  if (Array.isArray(d.x_range) && (d.x_range as unknown[]).length === 2) {
    const [a, b] = d.x_range as [number, number];
    parts.push(`x ${Number(a).toPrecision(3)}–${Number(b).toPrecision(3)}${d.x_units ? ` ${d.x_units}` : ""}`);
  }
  return parts.join(" · ");
}

/* ── diff rendering (word-level, unchanged blocks collapsed) ─────── */

type LineOp = { t: "eq" | "del" | "add"; text: string };

/** Longest-common-subsequence line diff; small documents (a skill is a
 * few hundred lines) so the quadratic table is fine. */
function lineDiff(a: string[], b: string[]): LineOp[] {
  const n = a.length, m = b.length;
  const dp: Uint16Array[] = [];
  for (let i = 0; i <= n; i++) dp.push(new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops: LineOp[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push({ t: "eq", text: a[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push({ t: "del", text: a[i] }); i++; }
    else { ops.push({ t: "add", text: b[j] }); j++; }
  }
  while (i < n) ops.push({ t: "del", text: a[i++] });
  while (j < m) ops.push({ t: "add", text: b[j++] });
  return ops;
}

type WordOp = { t: "eq" | "del" | "add"; text: string };

function wordDiff(a: string, b: string): WordOp[] {
  const ta = a.split(/(\s+)/), tb = b.split(/(\s+)/);
  const n = ta.length, m = tb.length;
  if (n * m > 250_000) return [{ t: "del", text: a }, { t: "add", text: b }];
  const dp: Uint16Array[] = [];
  for (let i = 0; i <= n; i++) dp.push(new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = ta[i] === tb[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const ops: WordOp[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (ta[i] === tb[j]) { ops.push({ t: "eq", text: ta[i] }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push({ t: "del", text: ta[i] }); i++; }
    else { ops.push({ t: "add", text: tb[j] }); j++; }
  }
  while (i < n) ops.push({ t: "del", text: ta[i++] });
  while (j < m) ops.push({ t: "add", text: tb[j++] });
  return ops;
}

type Block =
  | { kind: "same"; lines: string[] }
  | { kind: "change"; dels: string[]; adds: string[] };

function blocks(ops: LineOp[]): Block[] {
  const out: Block[] = [];
  for (const op of ops) {
    const last = out[out.length - 1];
    if (op.t === "eq") {
      if (last && last.kind === "same") last.lines.push(op.text);
      else out.push({ kind: "same", lines: [op.text] });
    } else {
      const blk = last && last.kind === "change" ? last : null;
      if (blk) (op.t === "del" ? blk.dels : blk.adds).push(op.text);
      else out.push({ kind: "change", dels: op.t === "del" ? [op.text] : [], adds: op.t === "add" ? [op.text] : [] });
    }
  }
  return out;
}

/** The review diff: side by side (current | proposed), changed words
 * marked inside changed lines, unchanged runs folded to one line. */
function ReviewDiff({ before, after }: { before: string; after: string }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const bl = useMemo(() => blocks(lineDiff(before.split("\n"), after.split("\n"))), [before, after]);
  const changed = bl.filter((b) => b.kind === "change").length;
  if (!changed) return <p className="caption">No textual change.</p>;
  const CONTEXT = 2;
  return (
    <div className="mem-sbs">
      <div className="mem-sbs-head"><span>current</span><span>after upgrade</span></div>
      {bl.map((b, idx) => {
        if (b.kind === "same") {
          const isEdge = idx === 0 || idx === bl.length - 1;
          const open = expanded.has(idx) || b.lines.length <= CONTEXT * 2 + 1;
          const head = isEdge && idx === 0 ? [] : b.lines.slice(0, CONTEXT);
          const tail = isEdge && idx === bl.length - 1 ? [] : b.lines.slice(-CONTEXT);
          const shown = open ? b.lines : [...head, null, ...tail];
          return shown.map((l, k) => l === null ? (
            <button key={`${idx}-fold`} type="button" className="mem-fold" onClick={() => setExpanded((s) => new Set(s).add(idx))}>
              … {b.lines.length - head.length - tail.length} unchanged lines
            </button>
          ) : (
            <div key={`${idx}-${k}`} className="mem-sbs-row same"><span>{l || " "}</span><span>{l || " "}</span></div>
          ));
        }
        const rows = Math.max(b.dels.length, b.adds.length);
        return Array.from({ length: rows }, (_, k) => {
          const d = b.dels[k], a = b.adds[k];
          if (d != null && a != null) {
            const w = wordDiff(d, a);
            return (
              <div key={`${idx}-${k}`} className="mem-sbs-row changed">
                <span>{w.filter((x) => x.t !== "add").map((x, i) => x.t === "del" ? <mark key={i} className="del">{x.text}</mark> : <span key={i}>{x.text}</span>)}</span>
                <span>{w.filter((x) => x.t !== "del").map((x, i) => x.t === "add" ? <mark key={i} className="add">{x.text}</mark> : <span key={i}>{x.text}</span>)}</span>
              </div>
            );
          }
          return (
            <div key={`${idx}-${k}`} className="mem-sbs-row changed">
              <span className={d != null ? "whole del" : "empty"}>{d ?? ""}</span>
              <span className={a != null ? "whole add" : "empty"}>{a ?? ""}</span>
            </div>
          );
        });
      })}
    </div>
  );
}

/* ── jobs ───────────────────────────────────────────────────────── */

interface LocalJob { id: string; kind: "consolidate" | "upgrade"; label: string; started: number }

function loadLocalJobs(): LocalJob[] {
  try {
    const raw = localStorage.getItem(JOBS_KEY);
    const v = raw ? (JSON.parse(raw) as LocalJob[]) : [];
    return Array.isArray(v) ? v.slice(-20) : [];
  } catch {
    return [];
  }
}

function saveLocalJobs(jobs: LocalJob[]) {
  try { localStorage.setItem(JOBS_KEY, JSON.stringify(jobs.slice(-20))); } catch { /* per-viewer convenience only */ }
}

function JobStrip({
  local,
  server,
  onJump,
}: {
  local: LocalJob[];
  server: MemoryJobRow[] | null;
  onJump: (skill: string) => void;
}) {
  const [, setTick] = useState(0);
  const running = (server ?? []).some((j) => j.status === "running");
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [running]);
  const byId = new Map((server ?? []).map((j) => [j.id, j]));
  const rows = [...local].reverse().map((l) => ({ l, s: byId.get(l.id) ?? null }));
  const orphan = (server ?? []).filter((j) => !local.some((l) => l.id === j.id));
  if (!rows.length && !orphan.length) return null;
  return (
    <div className="mem-jobs">
      <div className="mem-jobs-head">Distillation jobs <span className="caption">(this server process; a restart forgets them)</span></div>
      {rows.map(({ l, s }) => {
        const elapsed = Math.max(0, Math.round((Date.now() - l.started) / 1000));
        if (!s) {
          return (
            <div key={l.id} className="mem-job-row unknown">
              <span className="mem-job-status">?</span>
              <span>{l.label}</span>
              <span className="caption">started {shortDate(new Date(l.started).toISOString())} — unknown to the server (restarted?). If it finished, its result is in the Skills list below.</span>
            </div>
          );
        }
        return (
          <div key={l.id} className={`mem-job-row ${s.status}`}>
            <span className="mem-job-status">{s.status === "running" ? <span className="spinner" /> : s.status === "done" ? "✓" : "✗"}</span>
            <span>{s.kind === "consolidate" ? "Consolidate" : "Upgrade"} {s.label}</span>
            {s.status === "running" && <span className="caption">typically 1–3 min, {elapsed}s so far; you can keep working</span>}
            {s.status === "done" && s.skill_name && (
              <button type="button" className="link-btn" onClick={() => onJump(`${s.domain}/${s.skill_name}`)}>
                → {s.skill_name} (provisional, approve it below)
              </button>
            )}
            {s.status === "done" && !s.skill_name && s.kind === "upgrade" && (
              <span className="caption">proposal built{s.target ? ` for ${s.target}` : ""} — the inbox card shows it until it is applied or cancelled</span>
            )}
            {s.status === "done" && !s.skill_name && s.kind !== "upgrade" && <span className="caption">done</span>}
            {s.status === "error" && <span className="caption warn">{s.error ?? "failed"} — the records are unchanged</span>}
          </div>
        );
      })}
      {orphan.map((j) => (
        <div key={j.id} className={`mem-job-row ${j.status}`}>
          <span className="mem-job-status">{j.status === "running" ? <span className="spinner" /> : j.status === "done" ? "✓" : "✗"}</span>
          <span>{j.kind === "consolidate" ? "Consolidate" : "Upgrade"} {j.label} <span className="caption">(started elsewhere)</span></span>
          {j.status === "done" && j.skill_name && (
            <button type="button" className="link-btn" onClick={() => onJump(`${j.domain}/${j.skill_name}`)}>→ {j.skill_name}</button>
          )}
          {j.status === "error" && <span className="caption warn">{j.error ?? "failed"}</span>}
        </div>
      ))}
    </div>
  );
}

/* ── the panel ──────────────────────────────────────────────────── */

export function MemoryPanel({ sessionId, active }: { sessionId: string; active: boolean }) {
  const [ov, setOv] = useState<MemoryOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ kind: "ok" | "warn"; text: string } | null>(null);
  const [localJobs, setLocalJobs] = useState<LocalJob[]>(() => loadLocalJobs());
  const [serverJobs, setServerJobs] = useState<MemoryJobRow[] | null>(null);
  const [lastNominated, setLastNominated] = useState<string | null>(null);
  const [focusSkill, setFocusSkill] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [o, j] = await Promise.all([api.memory(), api.memoryJobs().catch(() => ({ jobs: [] }))]);
      setOv(o);
      setServerJobs(j.jobs);
      setError(null);
    } catch (e) {
      setError(errText(e));
    }
  }, []);

  useEffect(() => {
    if (active) void refresh();
  }, [active, refresh]);

  // While a job runs, keep the strip and the counts current.
  const anyRunning = (serverJobs ?? []).some((j) => j.status === "running");
  useEffect(() => {
    if (!active || !anyRunning) return;
    const t = setInterval(() => void refresh(), 2000);
    return () => clearInterval(t);
  }, [active, anyRunning, refresh]);

  const notify = useCallback((kind: "ok" | "warn", text: string) => {
    setNotice({ kind, text });
  }, []);
  useEffect(() => {
    if (!notice || notice.kind !== "ok") return;
    const t = setTimeout(() => setNotice(null), 8000);
    return () => clearTimeout(t);
  }, [notice]);

  const trackJob = useCallback((job: LocalJob) => {
    setLocalJobs((jobs) => {
      const next = [...jobs.filter((j) => j.id !== job.id), job];
      saveLocalJobs(next);
      return next;
    });
    void refresh();
  }, [refresh]);

  const jump = useCallback((ref: string) => {
    setFocusSkill(ref);
    window.setTimeout(() => {
      document.getElementById(`mem-skill-${ref}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 50);
  }, []);

  if (error) {
    return (
      <div className="skills-panel">
        <section className="tools-section">
          <h3>Persistent memory</h3>
          <p className="caption warn">Could not read persistent memory: {error}</p>
        </section>
      </div>
    );
  }
  if (!ov) return <div className="skills-panel"><p className="caption">loading…</p></div>;
  const p = ov.pipeline;
  const memOn = ov.enabled;

  return (
    <div className="skills-panel mem-panel">
      <section className="tools-section">
        <h3>Persistent memory</h3>
        <p className="caption">
          One store per server (<code>{ov.home}</code>), shared by every session: what the agents
          learned, kept across sessions and upgrades.
          {ov.shared_server && " This is a shared server — deleting is left to the operator."}
        </p>

        <label className="mem-switch">
          <span className="toggle-switch">
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
            <span className="track" />
          </span>
          <span>Persistent memory is <strong>{memOn ? "on" : "off"}</strong></span>
          <span className="caption">
            {memOn
              ? "runs bank their scripts, stage lessons, and load approved skills"
              : "inert: nothing is banked or staged and approved skills are not loaded (existing items stay reviewable)"}
          </span>
        </label>
        {ov.env_override && (
          <p className="caption warn">
            ⚠️ <code>SCILINK_MEMORY={ov.env_override}</code> is set in the server's environment and
            overrides this switch.
          </p>
        )}

        <p className="mem-pipeline">
          <span>1 · Script bank <b>{p.bank_total}</b>{p.bank_proven ? <span className="caption"> · ★ {p.bank_proven} proven</span> : null}{p.bank_archived ? <span className="caption"> · {p.bank_archived} archived</span> : null}</span>
          <span className="mem-arrow">→</span>
          <span>2 · Review inbox <b>{p.inbox_total}</b>{p.inbox_ready ? <span className="caption"> · {p.inbox_ready} ready to distill</span> : null}</span>
          <span className="mem-arrow">→</span>
          <span>3 · Skills <b>{p.skills_total}</b>{p.skills_provisional ? <span className="caption"> · {p.skills_provisional} awaiting approval</span> : null}</span>
        </p>

        {notice && (
          <p className={`mem-notice ${notice.kind}`}>
            {notice.text}
            <button type="button" className="icon-btn" title="Dismiss" onClick={() => setNotice(null)}>✕</button>
          </p>
        )}

        <JobStrip local={localJobs} server={serverJobs} onJump={jump} />
      </section>

      <section className="tools-section mem-stage">
        <h4>1 · Script bank <span className="caption">— every success, recorded automatically</span></h4>
        <BankSection ov={ov} onChange={refresh} notify={notify} onNominated={setLastNominated} />
      </section>

      <section className="tools-section mem-stage">
        <h4>2 · Review inbox <span className="caption">— lessons awaiting your call</span></h4>
        <InboxSection ov={ov} sessionId={sessionId} onChange={refresh} notify={notify}
          lastNominated={lastNominated} trackJob={trackJob} />
      </section>

      <section className="tools-section mem-stage">
        <h4>3 · Skills <span className="caption">— curated knowledge with planning-layer authority</span></h4>
        <SkillsSection ov={ov} onChange={refresh} notify={notify} focus={focusSkill} />
      </section>
    </div>
  );
}

/* ── 1 · script bank ─────────────────────────────────────────────── */

function EvidenceList({ row, provenN }: { row: MemoryBankRow; provenN: number }) {
  const ev = row.evidence ?? [];
  const own = row.provenance ?? {};
  const ownData = dataSummary(row.data);
  return (
    <div className="mem-evidence">
      <div className="caption">
        Evidence — {row.n_independent ?? row.n_successes} of {provenN} independent datasets
        {row.proven ? " (★ proven)" : ""}; a record is credited when its script, or a small adaptation of it, passes review on new data.
      </div>
      <table>
        <thead><tr><th>session</th><th>when</th><th>data</th><th>how</th><th>model it fit</th></tr></thead>
        <tbody>
          <tr>
            <td><code>{String(own.session ?? row.sessions[0] ?? "?")}</code></td>
            <td>{shortDate(row.created_at)}</td>
            <td>{ownData || (own.data_file ? String(own.data_file) : "")}</td>
            <td>banked (original)</td>
            <td className="mem-model">{row.model_type ?? row.label}</td>
          </tr>
          {ev.map((e: MemoryEvidence, i) => (
            <tr key={i} className={e.cross_kind ? "cross" : ""}>
              <td><code>{e.session ?? "?"}</code></td>
              <td>{shortDate(e.at)}</td>
              <td>{dataSummary(e.data)}</td>
              <td>{e.adapted ? "adapted" : "verbatim"}</td>
              <td className="mem-model">
                {e.model_type ?? "—"}
                {e.cross_kind && (
                  <span className="mem-badge warn" title="This success came from a different kind of signal than the record models — the credit is suspect.">
                    ⚠ different kind of signal
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {ev.length === 0 && row.sessions.length > 1 && (
        <p className="caption">Older credits ({row.sessions.slice(1).join(", ")}) predate the evidence ledger — only the session names are known.</p>
      )}
    </div>
  );
}

function BankSection({
  ov,
  onChange,
  notify,
  onNominated,
}: {
  ov: MemoryOverview;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
  onNominated: (stagedId: string) => void;
}) {
  const [viewing, setViewing] = useState<Record<string, MemoryBankRecord | "loading">>({});
  const [evidenceOpen, setEvidenceOpen] = useState<Set<string>>(new Set());
  const [groupLabels, setGroupLabels] = useState<Record<string, string>>({});
  const [sweep, setSweep] = useState<{ days: number; rows: MemorySweepRow[] } | null>(null);
  const [sweepDays, setSweepDays] = useState<string>("");
  const [archivedOpen, setArchivedOpen] = useState(false);

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
    <div>
      <p className="caption">
        Each approved analysis banks its working script with a fingerprint of the data it solved;
        later runs start from the closest match. A record that keeps passing on new data (★,{" "}
        {ov.proven_n} or more independent datasets) is an evidence-backed skill candidate — nominate
        it into the review inbox.
      </p>
      {ov.bank.length === 0 && (
        <p className="caption">No banked scripts yet — they accumulate as analyses succeed.</p>
      )}
      {ov.bank.map((d, di) => (
        <details key={d.domain} className="mem-group" open={di === 0}>
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
                        if (out.staged_ids[0]) onNominated(out.staged_ids[0]);
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
            const evOpen = evidenceOpen.has(key);
            const metric = r.metric && typeof r.metric === "object" && (r.metric as { value?: unknown }).value != null
              ? `${(r.metric as { name?: string }).name ?? "metric"}=${String((r.metric as { value?: unknown }).value)}`
              : typeof r.metric === "number" ? `metric=${r.metric}` : "";
            const crossKind = (r.evidence ?? []).some((e) => e.cross_kind);
            return (
              <div key={r.id} className="mem-row">
                <div className="mem-row-main">
                  <div className="mem-badges">
                    <code>{r.id}</code>
                    {r.proven
                      ? <span className="mem-badge ok" title="Passed on independent datasets">★ proven</span>
                      : <span className="mem-badge" title="Independent datasets it passed on / needed to be proven">{r.n_independent ?? r.n_successes}/{ov.proven_n} datasets</span>}
                    {r.data_kind && <span className="mem-badge dim">{r.data_kind}</span>}
                    {r.n_failures ? <span className="mem-badge warn">{r.n_failures} miss{r.n_failures === 1 ? "" : "es"}</span> : null}
                    {crossKind && <span className="mem-badge warn" title="Some credit came from a different kind of signal — open the evidence">⚠ suspect credit</span>}
                    {r.promoted_to_staging && <span className="mem-badge pending" title={`Staged as ${r.promoted_to_staging}`}>in review inbox</span>}
                    <span className="caption">retrieved {r.n_retrievals}×{metric ? ` · ${metric}` : ""}</span>
                  </div>
                  <div className="mem-label">{r.label}</div>
                </div>
                <div className="mem-row-actions">
                  <button type="button" className="link-btn" onClick={() =>
                    setEvidenceOpen((s) => { const n = new Set(s); if (n.has(key)) n.delete(key); else n.add(key); return n; })}>
                    {evOpen ? "hide evidence" : "evidence"}
                  </button>
                  <button type="button" className="link-btn" onClick={() => void toggleView(d.domain, r.id)}>
                    {v ? "hide script" : "script"}
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
                        notify("ok", `Staged as ${out.staged_id} [${out.technique}] — it is selected in the inbox below.`);
                        onNominated(out.staged_id);
                        await onChange();
                      } catch (e) {
                        notify("warn", errText(e));
                      }
                    }}
                  >
                    nominate for review
                  </button>
                  {ov.can_delete && (
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
                  )}
                </div>
                {evOpen && <EvidenceList row={r} provenN={ov.proven_n} />}
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

      <div className="mem-card mem-aging">
        <div>
          <strong>Aging</strong>
          <span className="caption"> — records never retrieved, never succeeding again, or superseded by a proven variant are archived, not deleted; an archived record can be restored.</span>
        </div>
        <div className="mem-row-actions">
          <label className="caption">idle for at least
            <input type="number" min={0} className="mem-days" value={sweepDays} placeholder="default"
              onChange={(e) => setSweepDays(e.target.value)} /> days
          </label>
          <button
            type="button"
            className="link-btn"
            onClick={async () => {
              try {
                const days = sweepDays.trim() === "" ? null : Number(sweepDays);
                const out = await api.memoryBankSweep({ days, dry_run: true });
                setSweep({ days: days ?? -1, rows: out.records });
              } catch (e) { notify("warn", errText(e)); }
            }}
          >
            preview sweep
          </button>
          <button type="button" className="link-btn" onClick={() => setArchivedOpen((o) => !o)}>
            {archivedOpen ? "hide archived" : `archived (${ov.archived.length})`}
          </button>
        </div>
        {sweep && (
          <div className="mem-detail">
            {sweep.rows.length === 0 ? (
              <p className="caption">Nothing would be archived{sweep.days >= 0 ? ` at ${sweep.days} idle days` : ""}.</p>
            ) : (
              <>
                <p className="caption">{sweep.rows.length} record{sweep.rows.length === 1 ? "" : "s"} would be archived:</p>
                <ul className="mem-list">
                  {sweep.rows.map((r) => (
                    <li key={`${r.domain}/${r.id}`}><code>{r.domain}/{r.id}</code> — {r.reason.replace(/_/g, " ")}{r.idle_days != null ? ` (idle ${r.idle_days} d)` : ""}: <span className="caption">{r.label}</span></li>
                  ))}
                </ul>
                {ov.can_delete ? (
                  <ConfirmButton
                    label={`archive these ${sweep.rows.length}`}
                    confirmLabel="confirm archive"
                    onConfirm={async () => {
                      try {
                        const out = await api.memoryBankSweep({ days: sweep.days >= 0 ? sweep.days : null, dry_run: false });
                        notify("ok", `Archived ${out.records.length} record(s) — restorable below.`);
                        setSweep(null);
                        await onChange();
                      } catch (e) { notify("warn", errText(e)); }
                    }}
                  />
                ) : <p className="caption">Archiving on a shared server is the operator's: <code>scilink memory bank-sweep</code>.</p>}
              </>
            )}
          </div>
        )}
        {archivedOpen && (
          <div className="mem-detail">
            {ov.archived.length === 0 ? <p className="caption">No archived records.</p> : (
              <ul className="mem-list">
                {ov.archived.map((a) => (
                  <li key={`${a.domain}/${a.id}`}>
                    <code>{a.domain}/{a.id}</code> — {(a.reason ?? "archived").replace(/_/g, " ")}{a.archived_at ? ` on ${shortDate(a.archived_at)}` : ""}
                    : <span className="caption">{a.label}</span>{" "}
                    <button type="button" className="link-btn" onClick={async () => {
                      try {
                        await api.memoryBankRestore(a.domain, a.id);
                        notify("ok", `Restored ${a.id} to the bank.`);
                        await onChange();
                      } catch (e) { notify("warn", errText(e)); }
                    }}>restore</button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
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
  lastNominated,
  trackJob,
}: {
  ov: MemoryOverview;
  sessionId: string;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
  lastNominated: string | null;
  trackJob: (job: LocalJob) => void;
}) {
  return (
    <div>
      <p className="caption">
        Three streams converge here: 📜 script nominations, 🐛 error lessons, 💬 your feedback.
        Select records and distill them into a <strong>new</strong> skill or into an{" "}
        <strong>existing</strong> one — either way they merge in one pass: method, pitfalls, your
        constraints. Consolidation needs {ov.consolidate_min_n} records; one example is too
        idiosyncratic to generalise.
      </p>
      {ov.inbox.length === 0 && <p className="caption">Nothing staged. Nominate a bank record above, or let runs stage their lessons.</p>}
      {ov.inbox.map((g) => (
        <InboxGroup
          key={`${g.domain}/${g.technique}`}
          group={g}
          ov={ov}
          sessionId={sessionId}
          onChange={onChange}
          notify={notify}
          lastNominated={lastNominated}
          trackJob={trackJob}
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
  lastNominated,
  trackJob,
}: {
  group: MemoryInboxGroup;
  ov: MemoryOverview;
  sessionId: string;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
  lastNominated: string | null;
  trackJob: (job: LocalJob) => void;
}) {
  const pool = useMemo(() => [...g.records, ...g.related], [g]);
  const [open, setOpen] = useState(true);
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
  const [showRaw, setShowRaw] = useState(false);
  const checkTimer = useRef<number | null>(null);

  // Records may change under us (a nomination, a discard): keep the
  // selection to what still exists; a record just nominated from the
  // bank is selected on arrival, so the next click is "distill".
  useEffect(() => {
    setSelected((s) => {
      const ids = new Set(pool.map((r) => r.id));
      const next = new Set([...s].filter((i) => ids.has(i)));
      for (const r of g.records) if (!s.size || !next.size) next.add(r.id);
      if (lastNominated && ids.has(lastNominated)) next.add(lastNominated);
      return next;
    });
    if (lastNominated && pool.some((r) => r.id === lastNominated)) setOpen(true);
  }, [pool, g.records, lastNominated]);

  const selectedIds = useMemo(() => pool.filter((r) => selected.has(r.id)).map((r) => r.id), [pool, selected]);
  const need = ov.consolidate_min_n;
  const memOn = ov.enabled;

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

  // poll a running job of this card (the strip above shows every job)
  useEffect(() => {
    if (!job) return;
    const t = setInterval(async () => {
      try {
        const j = await api.memoryJob(job.id);
        if (j.status === "running") return;
        setJob(null);
        if (j.status === "error") {
          notify("warn", job.kind === "consolidate"
            ? `Consolidation failed — no skill was written and the records are as they were. (${j.error})`
            : `Could not build the upgrade — nothing was written. (${j.error})`);
          await onChange();
          return;
        }
        if (job.kind === "consolidate") {
          const r = j.result as { n_examples?: number; skill_name?: string };
          notify("ok", `Consolidated ${r.n_examples ?? ""} → ${r.skill_name} — provisional; approve it in Skills below.`);
          await onChange();
        } else {
          const prop = j.result as MemoryProposal;
          setProposal(prop);
          setEdited(prop.proposed_content);
          setCheck({ warnings: prop.warnings, diff: prop.diff });
          setEditing(false);
          await onChange();
        }
      } catch (e) {
        setJob(null);
        notify("warn", errText(e));
      }
    }, 2000);
    return () => clearInterval(t);
  }, [job, notify, onChange]);

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
  const whyNot = !memOn ? "Turn persistent memory on to distill."
    : selectedIds.length < need ? `Select ${need - selectedIds.length} more record${need - selectedIds.length === 1 ? "" : "s"} (needs ${need}).`
    : !norm ? "Give the new skill a name."
    : swept.length ? `"${norm}" is also the label of ${swept.length} unselected record(s); select them or rename.`
    : job ? "A job is already running for this group." : "";

  return (
    <div className="mem-group mem-inbox-card">
      <div className="mem-group-head">
        <button type="button" className="link-btn mem-caret" onClick={() => setOpen((o) => !o)}>{open ? "▾" : "▸"}</button>
        <code>{g.domain}/{g.technique}</code>
        <span className="caption">{g.records.length} staged{g.related.length ? ` · ${g.related.length} related from the same sessions` : ""}</span>
        {g.ready && <span className="mem-badge ok">ready to distill</span>}
      </div>
      {open && (
        <>
          {pool.map((r) => {
            const ins = inspect[r.id];
            return (
              <div key={r.id} className={`mem-row${r.technique !== g.technique ? " related" : ""}${r.id === lastNominated ? " just-nominated" : ""}`}>
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
                    {r.id === lastNominated && <span className="mem-badge ok">just nominated</span>}
                  </label>
                </div>
                <div className="mem-row-actions">
                  <button type="button" className="link-btn" onClick={() => void toggleInspect(r.id)}>
                    {ins ? "hide" : "inspect"}
                  </button>
                  {ov.can_delete && (
                    <ConfirmButton
                      label="discard"
                      confirmLabel="confirm discard"
                      title="Remove from the review inbox without distilling it. A nominated bank record becomes nominatable again."
                      onConfirm={async () => {
                        try {
                          await api.memoryInboxDiscard(g.domain, r.id);
                          notify("warn", `Discarded ${r.id}.`);
                          await onChange();
                        } catch (e) {
                          notify("warn", errText(e));
                        }
                      }}
                    />
                  )}
                </div>
                {ins === "loading" && <p className="caption">loading…</p>}
                {ins && ins !== "loading" && (
                  <div className="mem-detail">
                    {ins.bank && (
                      <p className="caption">
                        from bank <code>{ins.bank.bank_id}</code>
                        {ins.bank.n_successes && ins.bank.n_successes > 1
                          ? ` (${ins.bank.n_successes} successes)` : ""}
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
            <div className="mem-distill-head">
              <strong>Distill</strong>
              <span className="caption">{selectedIds.length} of {pool.length} selected</span>
              <span className="mem-dest">
                <label><input type="radio" checked={dest === "new"} onChange={() => setDest("new")} /> into a new skill</label>
                <label><input type="radio" checked={dest === "existing"} onChange={() => setDest("existing")} /> into an existing skill</label>
              </span>
            </div>

            {dest === "new" ? (
              <>
                <div className="mem-row-actions">
                  <input type="text" value={label} placeholder="new skill name" onChange={(e) => setLabel(e.target.value)} />
                  <span className="caption">will be saved as <code>auto_{norm || "…"}</code>, provisional until approved</span>
                  <button
                    type="button"
                    className="primary small"
                    disabled={!canConsolidate}
                    title={canConsolidate ? "One large model call — typically 1–3 minutes." : whyNot}
                    onClick={async () => {
                      try {
                        const j = await api.memoryConsolidate(g.domain, selectedIds, label, sessionId);
                        const lj: LocalJob = { id: j.job_id, label: j.label, started: Date.now(), kind: "consolidate" };
                        setJob(lj);
                        trackJob(lj);
                      } catch (e) {
                        notify("warn", errText(e));
                      }
                    }}
                  >
                    Consolidate {selectedIds.length}
                  </button>
                </div>
                {!canConsolidate && !job && <p className="caption mem-why">{whyNot}</p>}
              </>
            ) : (
              <>
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
                        const lj: LocalJob = { id: j.job_id, label: j.label, started: Date.now(), kind: "upgrade" };
                        setJob(lj);
                        trackJob(lj);
                      } catch (e) {
                        notify("warn", errText(e));
                      }
                    }}
                  >
                    Preview upgrade ({selectedIds.length} → {tgt?.name ?? "…"})
                  </button>
                </div>
                {tgt?.match === false && (
                  <p className="caption warn">
                    ⚠️ Technique mismatch: the selection does not match <code>{tgt.name}</code>'s routing —
                    upgrading would pollute an unrelated skill.
                  </p>
                )}
                {!memOn && <p className="caption mem-why">Turn persistent memory on to distill.</p>}
              </>
            )}
            {job && (
              <p className="mem-job">
                <span className="spinner" /> {job.kind === "consolidate" ? "Distilling" : "Building the merged skill for review"}{" "}
                {job.label} — see the job strip at the top; this card updates when it finishes.
              </p>
            )}
          </div>

          {proposal && (
            <div className="mem-review">
              <div className="mem-review-head">
                <strong>Review upgrade → <code>{proposal.target_domain}/{proposal.target_name}</code></strong>
                <span className="caption">
                  applies in place; the current version is kept as a backup you can restore
                  {proposal.builtin_target ? "; the built-in is forked into the store first" : ""}.
                </span>
              </div>
              <div className="mem-row-actions">
                <label className="mem-check">
                  <input type="checkbox" checked={editing} onChange={(e) => setEditing(e.target.checked)} />
                  <span>Edit the proposal before applying</span>
                </label>
                <label className="mem-check">
                  <input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} />
                  <span>Show the raw unified diff</span>
                </label>
              </div>
              {editing && (
                <textarea className="mem-editor" value={edited} onChange={(e) => setEdited(e.target.value)} rows={18} />
              )}
              {showRaw
                ? <pre className="mem-diff">{check?.diff ?? proposal.diff}</pre>
                : <ReviewDiff before={proposal.existing_content} after={editing ? edited : proposal.proposed_content} />}
              <div className="mem-review-foot">
                <div className="mem-review-warnings">
                  {(check?.warnings ?? []).length === 0
                    ? <span className="caption">Additivity check: nothing the current skill says is lost.</span>
                    : (check?.warnings ?? []).map((w, i) => <p key={i} className="caption warn">⚠ {w}</p>)}
                </div>
                <div className="mem-row-actions">
                  <button
                    type="button"
                    className="primary small"
                    onClick={async () => {
                      try {
                        await api.memoryApplyUpgrade(g.domain, proposal.staged_ids, proposal.target_domain,
                          proposal.target_name, editing ? edited : proposal.proposed_content, proposal.builtin_target);
                        notify("ok", `Upgraded ${proposal.target_domain}/${proposal.target_name} (previous version kept as backup).`);
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
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ── 3 · skills ──────────────────────────────────────────────────── */

function SkillsSection({
  ov,
  onChange,
  notify,
  focus,
}: {
  ov: MemoryOverview;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
  focus: string | null;
}) {
  const provisional = ov.skills.filter((s) => s.provisional);
  const approved = ov.skills.filter((s) => !s.provisional);
  return (
    <div>
      <p className="caption">
        Approved skills auto-route when the data's technique matches their technique list;
        provisional ones wait for your call; a fork of a built-in shadows the shipped copy. Every
        edit and upgrade keeps the previous version as a backup.
      </p>
      {provisional.length > 0 && <div className="mem-subhead">Awaiting approval ({provisional.length})</div>}
      {provisional.map((s) => (
        <SkillRow key={`${s.domain}/${s.name}`} skill={s} ov={ov} onChange={onChange} notify={notify}
          focused={focus === `${s.domain}/${s.name}`} />
      ))}
      {approved.length > 0 && <div className="mem-subhead">Approved for routing ({approved.length})</div>}
      {approved.map((s) => (
        <SkillRow key={`${s.domain}/${s.name}`} skill={s} ov={ov} onChange={onChange} notify={notify}
          focused={focus === `${s.domain}/${s.name}`} />
      ))}
      {ov.skills.length === 0 && (
        <p className="caption">No skills yet — they are distilled from the review inbox above, or forked from the catalog on the Skills tab.</p>
      )}
    </div>
  );
}

function provenanceLabel(s: MemorySkill): string {
  if (s.shadows_builtin) return "fork of built-in";
  if (s.provenance === "t2_consolidated") return `consolidated${s.n_examples ? ` from ${s.n_examples} examples` : ""}`;
  if (s.provenance === "t2_autodistill" || s.provenance === "t2_hot_win") return "auto-distilled from a hot win";
  if (s.provenance) return s.provenance.replace(/_/g, " ");
  if (s.name.startsWith("auto_")) return "auto-distilled";
  return "graduated in a session";
}

function SkillRow({
  skill: s,
  ov,
  onChange,
  notify,
  focused,
}: {
  skill: MemorySkill;
  ov: MemoryOverview;
  onChange: () => Promise<void>;
  notify: (k: "ok" | "warn", t: string) => void;
  focused: boolean;
}) {
  const ref = `${s.domain}/${s.name}`;
  const memOn = ov.enabled;
  const [open, setOpen] = useState(false);
  const [text, setText] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [diff, setDiff] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [techEdit, setTechEdit] = useState<string | null>(null);

  useEffect(() => { if (focused) setOpen(true); }, [focused]);

  const load = async () => api.memorySkillText(s.domain, s.name);
  const front = text ? /^---\n([\s\S]*?)\n---\n?/.exec(text) : null;

  return (
    <div id={`mem-skill-${ref}`} className={`mem-group mem-skill${focused ? " focused" : ""}`}>
      <div className="mem-group-head" onClick={() => setOpen((o) => !o)} role="button" tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") setOpen((o) => !o); }}>
        <span className="mem-caret">{open ? "▾" : "▸"}</span>
        <span className={`mem-badge ${s.provisional ? "pending" : "ok"}`}>{s.provisional ? "provisional" : "approved"}</span>
        <code>{ref}</code>
        <span className="mem-badge dim" title="Where this skill came from">{provenanceLabel(s)}</span>
        {s.technique.length
          ? <span className="mem-techs">{s.technique.map((t) => <span key={t} className="mem-chip">{t}</span>)}</span>
          : <span className="mem-badge warn" title="Without a technique list the selectors have nothing to match this skill on">no technique — set one</span>}
        {s.metric && <span className="caption">{s.metric}</span>}
      </div>
      {open && (
        <>
          {s.description && <p className="mem-desc"><em>{s.description}</em></p>}
          {s.session && <p className="caption">session: {s.session}</p>}
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
            <button
              type="button"
              className="link-btn"
              title="The technique names the selectors match the data's measurement technique against."
              onClick={() => setTechEdit((t) => (t === null ? s.technique.join(", ") : null))}
            >
              {techEdit === null ? "technique" : "close technique"}
            </button>
            {s.provisional ? (
              <button
                type="button"
                className="primary small"
                disabled={!memOn}
                title={memOn ? "Clears the provisional flag — the skill enters the auto-routing menu."
                  : "Persistent memory is off — approved skills will not load until you turn it on."}
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
                suspend
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
            {s.has_backup && (
              <ConfirmButton
                label="restore previous version"
                confirmLabel="confirm restore"
                title="Swap the backup written by the last edit or upgrade back into place. The current version becomes the backup, so this can be undone the same way."
                onConfirm={async () => {
                  try {
                    await api.memorySkillAction(s.domain, s.name, "restore-backup");
                    notify("ok", `Restored the previous version of ${ref}; the replaced version is now the backup.`);
                    setText(null);
                    await onChange();
                  } catch (e) { notify("warn", errText(e)); }
                }}
              />
            )}
            {ov.can_delete && (
              <ConfirmButton
                label="delete"
                onConfirm={async () => {
                  try {
                    await api.memorySkillAction(s.domain, s.name, "prune");
                    notify("warn", `Deleted ${ref}.`);
                    await onChange();
                  } catch (e) { notify("warn", errText(e)); }
                }}
              />
            )}
          </div>
          {techEdit !== null && (
            <div className="mem-detail mem-tech-edit">
              <label className="caption">technique names, comma-separated (the data's technique must match one of them to route here)</label>
              <div className="mem-row-actions">
                <input type="text" value={techEdit} placeholder="e.g. Raman spectroscopy, micro-Raman" onChange={(e) => setTechEdit(e.target.value)} />
                <button
                  type="button"
                  className="primary small"
                  onClick={async () => {
                    try {
                      const list = techEdit.split(",").map((t) => t.trim()).filter(Boolean);
                      const out = await api.memorySkillTechnique(s.domain, s.name, list);
                      notify("ok", out.technique.length ? `Technique of ${ref}: ${out.technique.join(", ")}.` : `Cleared the technique list of ${ref}.`);
                      setTechEdit(null);
                      await onChange();
                    } catch (e) { notify("warn", errText(e)); }
                  }}
                >
                  Save technique
                </button>
              </div>
            </div>
          )}
          {diff !== null && (diff
            ? <pre className="mem-diff">{diff}</pre>
            : <p className="caption">Identical to the shipped built-in.</p>)}
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
                  title="Validates frontmatter and sections; the previous version is kept as a backup."
                  onClick={async () => {
                    try {
                      await api.memorySkillEdit(s.domain, s.name, editing);
                      notify("ok", `Saved ${ref}; the previous version is the backup.`);
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
        </>
      )}
    </div>
  );
}
