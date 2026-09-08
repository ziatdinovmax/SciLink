import { useCallback, useEffect, useState } from "react";
import { api, type SkillCatalog } from "../api";
import { Dropzone } from "./Dropzone";
import { MarkdownBody } from "./MarkdownBody";

/** Skills tab — upload custom skills for this session and browse the
 * catalog (built-in bundles by domain, plus what you uploaded), with a
 * markdown viewer. Persistent memory (graduated / auto-distilled skills
 * under ~/.scilink) is a separate surface, not this panel. */

export function SkillsPanel({
  sessionId,
  active,
}: {
  sessionId: string;
  active: boolean;
}) {
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
  const total = (cat?.builtin ?? []).reduce((n, d) => n + d.skills.length, 0);

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
          Built-in skills <span className="caption">({total})</span>
        </h3>
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
                    <span className="caption skill-desc">{s.description}</span>
                    <button type="button" className="link-btn" onClick={() => void view(d.domain, s.name)}>
                      view
                    </button>
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
