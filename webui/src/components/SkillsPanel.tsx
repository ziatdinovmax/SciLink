import { useCallback, useEffect, useState } from "react";
import { api, type SkillCatalog } from "../api";
import { Dropzone } from "./Dropzone";
import { MarkdownBody } from "./MarkdownBody";

/** Skills tab — upload custom skills for this session and browse the
 * catalog: shipped bundles by domain plus what persistent memory adds
 * (learned skills, forks — labelled, never folded into "built-in"), with
 * a markdown viewer. The memory pipeline itself lives on the Memory tab. */

const ORIGIN_LABEL: Record<string, string> = {
  learned: "learned",
  fork: "fork of built-in",
};

export function SkillsPanel({
  sessionId,
  active,
  onOpenMemory,
}: {
  sessionId: string;
  active: boolean;
  onOpenMemory?: () => void;
}) {
  const [notice, setNotice] = useState<string | null>(null);
  const [cat, setCat] = useState<SkillCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [filter, setFilter] = useState("");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [viewing, setViewing] = useState<{ title: string; front: string; text: string } | null>(null);

  const refresh = useCallback(() => {
    api.skills(sessionId).then(setCat).catch((e) => setError(String(e)));
  }, [sessionId]);

  useEffect(() => {
    if (active) refresh();
  }, [active, refresh]);

  const view = async (domain: string, name: string) => {
    try {
      const raw = await api.skillMarkdown(sessionId, domain, name);
      // The YAML frontmatter (description, technique tags) is loader
      // metadata, not prose: rendered as markdown its `---` fence turns the
      // description into a title. Show it as a caption instead.
      const m = /^---\n([\s\S]*?)\n---\n?/.exec(raw);
      const front = m ? m[1].trim() : "";
      setViewing({
        title: `${domain === "custom" ? "custom" : domain} / ${name}`,
        front,
        text: m ? raw.slice(m[0].length) : raw,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const q = filter.trim().toLowerCase();
  const domains = (cat?.builtin ?? [])
    .map((d) => ({
      ...d,
      skills: d.skills.filter(
        (s) => !q || s.name.toLowerCase().includes(q) || s.description.toLowerCase().includes(q)
          || d.domain.toLowerCase().includes(q),
      ),
    }))
    .filter((d) => d.skills.length > 0);
  const all = (cat?.builtin ?? []).flatMap((d) => d.skills);
  const nBuiltin = all.filter((s) => s.origin === "builtin").length;
  const nLearned = all.filter((s) => s.origin === "learned").length;
  const nForks = all.filter((s) => s.origin === "fork").length;

  return (
    <div className="skills-panel">
      <section className="tools-section">
        <h3>Custom skills</h3>
        <p className="caption">
          A skill is one markdown file with the sections the agents read:
          Overview, Planning, Implementation, Interpretation, Validation.
          Uploaded skills are selectable by the agents for this session only —
          they are not saved to persistent memory.
        </p>
        {cat?.skills_supported === false ? (
          <p className="caption warn">This session's agent does not take custom skills.</p>
        ) : (
          <Dropzone
            label="Drop skill .md files here or click to browse"
            accept=".md"
            onFiles={async (files) => {
              setError(null);
              const r = await api.uploadSkills(sessionId, files);
              setCat(r.catalog);
              const msgs = r.errors.map((e) => `${e.file}: ${e.error}`);
              setNotes(msgs);
              if (r.registered.length === 0 && msgs.length) throw new Error(msgs.join("; "));
              return r.registered.map((n) => `${n} (registered)`);
            }}
          />
        )}
        {notes.map((n) => (
          <p key={n} className="caption warn">{n}</p>
        ))}
        {cat && cat.custom.length === 0 && <p className="caption">No custom skills in this session.</p>}
        {cat?.custom.map((s) => (
          <div key={s.name} className="skill-row">
            <code>{s.name}</code>
            <span className="caption skill-path" title={s.path}>{s.path.split("/").pop()}</span>
            <button type="button" className="link-btn" onClick={() => void view("custom", s.name)}>
              view
            </button>
          </div>
        ))}
      </section>

      <section className="tools-section">
        <h3>
          Skill catalog{" "}
          <span className="caption">
            ({nBuiltin} built-in{nLearned ? ` · ${nLearned} learned` : ""}{nForks ? ` · ${nForks} fork${nForks === 1 ? "" : "s"}` : ""})
          </span>
        </h3>
        <p className="caption">
          Shipped bundles, plus what persistent memory adds when it is on — learned skills and forks
          are labelled. Curate them on the{" "}
          {onOpenMemory ? (
            <button type="button" className="link-btn inline" onClick={onOpenMemory}>Memory tab</button>
          ) : "Memory tab"}.
        </p>
        {notice && <p className="caption">{notice}</p>}
        <input
          type="text"
          placeholder="Filter by name, description, or domain…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        {error && <p className="caption warn">{error}</p>}
        {domains.map((d) => {
          const isOpen = open.has(d.domain) || Boolean(q);
          return (
            <div key={d.domain} className="skill-domain">
              <button
                type="button"
                className="link-btn skill-domain-head"
                onClick={() =>
                  setOpen((prev) => {
                    const next = new Set(prev);
                    if (next.has(d.domain)) next.delete(d.domain);
                    else next.add(d.domain);
                    return next;
                  })
                }
              >
                {isOpen ? "▾" : "▸"} {d.label} <span className="caption">({d.skills.length})</span>
              </button>
              {isOpen &&
                d.skills.map((s) => (
                  <div key={s.name} className="skill-row">
                    <code>{s.name}</code>
                    {s.origin !== "builtin" && (
                      <span className={`mem-badge ${s.origin}`} title={s.origin === "fork"
                        ? "A copy in persistent memory that shadows the shipped skill of the same name"
                        : "Distilled or graduated into persistent memory"}>
                        {ORIGIN_LABEL[s.origin]}
                      </span>
                    )}
                    {s.provisional && (
                      <span className="mem-badge pending" title="Not in auto-routing until approved on the Memory tab">provisional</span>
                    )}
                    <span className="caption skill-desc">{s.description}</span>
                    <button type="button" className="link-btn" onClick={() => void view(d.domain, s.name)}>
                      view
                    </button>
                    {s.origin === "builtin" && (
                      <button
                        type="button"
                        className="link-btn"
                        title="Copy this shipped skill into persistent memory so it can be edited and upgraded; the copy shadows the built-in."
                        onClick={async () => {
                          try {
                            await api.memorySkillAction(d.domain, s.name, "fork");
                            setNotice(`Forked ${d.domain}/${s.name} into persistent memory — edit or upgrade it on the Memory tab.`);
                            refresh();
                          } catch (e) {
                            setError(e instanceof Error ? e.message : String(e));
                          }
                        }}
                      >
                        fork into memory
                      </button>
                    )}
                  </div>
                ))}
            </div>
          );
        })}
      </section>

      {viewing && (
        <div className="skill-viewer" role="dialog" aria-label={viewing.title}>
          <div className="skill-viewer-head">
            <strong>{viewing.title}</strong>
            <button type="button" className="icon-btn" title="Close" onClick={() => setViewing(null)}>
              ✕
            </button>
          </div>
          <div className="skill-viewer-body md-body">
            {viewing.front && <pre className="skill-front caption">{viewing.front}</pre>}
            <MarkdownBody text={viewing.text} />
          </div>
        </div>
      )}
    </div>
  );
}
