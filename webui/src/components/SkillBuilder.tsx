import { useEffect, useMemo, useRef, useState } from "react";
import { api, type SkillCatalog } from "../api";
import { MarkdownBody } from "./MarkdownBody";

/** The skill builder — write a skill without touching markdown by hand.
 *
 * A skill is one markdown file: a description the agents route on, an
 * optional technique list the selectors match the data's technique
 * against, and the five sections the agents read at fixed points of a run
 * (Overview → Planning → Implementation → Interpretation → Validation).
 * The form is those parts; the server renders the file exactly as the
 * loader expects it, and the preview is that rendering. "Use in this
 * session" saves and registers it like an upload; "Save to persistent
 * memory" writes an approved bundle into the store; "Start from" loads an
 * existing skill's parts into the form to derive a new one. */

export const SECTIONS: { key: string; title: string; hint: string }[] = [
  { key: "overview", title: "Overview", hint: "What the technique or method is, what data it fits, and when to reach for it (and when not to)." },
  { key: "planning", title: "Planning", hint: "How to plan a use of it: the model form or pipeline, parameter heuristics, what varies between datasets and how to choose." },
  { key: "implementation", title: "Implementation", hint: "How to write the code: the recipe the generated script should follow — libraries, steps, the exact algorithm where it matters. A concrete snippet is stronger than prose." },
  { key: "interpretation", title: "Interpretation", hint: "How to read the output and judge plausibility: what the numbers mean, typical ranges, what a bad result looks like." },
  { key: "validation", title: "Validation", hint: "How to verify it: quality criteria, sanity checks, failure indicators, tolerances." },
];

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "").replace(/^[^a-z]+/, "").slice(0, 64);
}

/** Split a skill file into the builder's parts: description and technique
 * from the frontmatter, sections by `## heading`. Off-vocabulary headings
 * are appended to the overview so nothing is lost. */
export function parseSkill(raw: string): { description: string; technique: string[]; sections: Record<string, string> } {
  let description = "";
  const technique: string[] = [];
  let text = raw;
  const fm = /^---\n([\s\S]*?)\n---\n?/.exec(raw);
  if (fm) {
    text = raw.slice(fm[0].length);
    const lines = fm[1].split("\n");
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      const dm = /^description:\s*(.*)$/.exec(l);
      if (dm) { description = dm[1].trim().replace(/^["']|["']$/g, ""); continue; }
      const tm = /^technique:\s*(.*)$/.exec(l);
      if (tm) {
        const inline = tm[1].trim();
        if (inline.startsWith("[")) {
          inline.replace(/^\[|\]$/g, "").split(",").map((t) => t.trim().replace(/^["']|["']$/g, "")).filter(Boolean).forEach((t) => technique.push(t));
        } else if (!inline) {
          for (let j = i + 1; j < lines.length && /^\s*-\s+/.test(lines[j]); j++) {
            technique.push(lines[j].replace(/^\s*-\s+/, "").trim().replace(/^["']|["']$/g, ""));
          }
        } else technique.push(inline.replace(/^["']|["']$/g, ""));
      }
    }
  }
  const sections: Record<string, string> = {};
  const parts = text.split(/^##\s+/m);
  const preface = parts.shift()?.trim() ?? "";
  for (const p of parts) {
    const nl = p.indexOf("\n");
    const head = (nl < 0 ? p : p.slice(0, nl)).trim().toLowerCase();
    const body = (nl < 0 ? "" : p.slice(nl + 1)).trim();
    const key = head === "analysis" ? "implementation" : head;
    if (SECTIONS.some((s) => s.key === key)) sections[key] = sections[key] ? `${sections[key]}\n\n${body}` : body;
    else sections.overview = `${sections.overview ? sections.overview + "\n\n" : ""}### ${head}\n\n${body}`;
  }
  if (preface && !sections.overview) sections.overview = preface;
  return { description, technique, sections };
}

export function SkillBuilder({
  sessionId,
  catalog,
  memoryOn,
  onSaved,
  onOpenMemory,
}: {
  sessionId: string;
  catalog: SkillCatalog | null;
  memoryOn: boolean | null;
  onSaved: (cat: SkillCatalog | undefined, note: string) => void;
  onOpenMemory?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [domain, setDomain] = useState("curve_fitting");
  const [description, setDescription] = useState("");
  const [technique, setTechnique] = useState("");
  const [sections, setSections] = useState<Record<string, string>>({});
  const [startFrom, setStartFrom] = useState("");
  const [preview, setPreview] = useState<string>("");
  const [showRaw, setShowRaw] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const previewTimer = useRef<number | null>(null);

  const domains = useMemo(() => {
    const ds = (catalog?.builtin ?? []).map((d) => d.domain);
    return ds.includes("curve_fitting") ? ds : ["curve_fitting", ...ds];
  }, [catalog]);
  const techniqueList = technique.split(",").map((t) => t.trim()).filter(Boolean);
  const filled = SECTIONS.filter((s) => (sections[s.key] ?? "").trim()).length;
  const ready = Boolean(slug(name)) && description.trim().length > 0 && filled > 0;

  // The preview is the server's rendering (the same call that saves), so
  // what you see is the file the loader will read. Debounced while typing.
  useEffect(() => {
    if (!open) return;
    if (previewTimer.current) window.clearTimeout(previewTimer.current);
    if (!ready) { setPreview(""); return; }
    previewTimer.current = window.setTimeout(async () => {
      try {
        const r = await api.composeSkill(sessionId, { name: slug(name), domain, description, technique: techniqueList, sections, save: "preview" });
        setPreview(r.markdown);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }, 500);
    return () => { if (previewTimer.current) window.clearTimeout(previewTimer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, name, domain, description, technique, sections, ready, sessionId]);

  const loadFrom = async (ref: string) => {
    setStartFrom(ref);
    if (!ref) return;
    const [d, n] = ref.split("/");
    try {
      const raw = await api.skillMarkdown(sessionId, d, n);
      const p = parseSkill(raw);
      setDescription(p.description);
      setTechnique(p.technique.join(", "));
      setSections(p.sections);
      if (d !== "custom") setDomain(d);
      if (!name) setName(`${n}_v2`);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const save = async (mode: "session" | "memory") => {
    setBusy(mode);
    setError(null);
    try {
      const r = await api.composeSkill(sessionId, { name: slug(name), domain, description, technique: techniqueList, sections, save: mode });
      onSaved(r.catalog, mode === "session"
        ? `${r.name} is registered for this session — the agents can select it now.`
        : `${r.domain}/${r.name} saved to persistent memory (approved, authored by you) — curate it on the Memory tab.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const download = () => {
    if (!preview) return;
    const blob = new Blob([preview], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${slug(name) || "skill"}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const front = preview ? /^---\n([\s\S]*?)\n---\n?/.exec(preview) : null;

  return (
    <section className="tools-section skill-builder">
      <h3>
        <button type="button" className="link-btn skill-domain-head" onClick={() => setOpen((o) => !o)}>
          {open ? "▾" : "▸"} Build a skill
        </button>
        <span className="caption"> — write the five sections; the file is rendered for you</span>
      </h3>
      {open && (
        <div className="sb-body">
          <div className="sb-grid">
            <label>
              <span>Name</span>
              <input type="text" value={name} placeholder="e.g. powder_xrd_phase_fractions" onChange={(e) => setName(e.target.value)} />
              {name && slug(name) !== name && <span className="caption">saved as <code>{slug(name) || "…"}</code></span>}
            </label>
            <label>
              <span>Domain</span>
              <select value={domain} onChange={(e) => setDomain(e.target.value)}>
                {domains.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
              <span className="caption">the agent family that reads it</span>
            </label>
            <label className="sb-wide">
              <span>Description</span>
              <input type="text" value={description} placeholder="One sentence naming the technique and what the skill does — the agents route on this" onChange={(e) => setDescription(e.target.value)} />
            </label>
            <label className="sb-wide">
              <span>Technique</span>
              <input type="text" value={technique} placeholder="comma-separated names and aliases, e.g. XRD, powder X-ray diffraction" onChange={(e) => setTechnique(e.target.value)} />
              <span className="caption">the data's measurement technique must match one of these for the selectors to pick the skill; leave empty for a skill that is not technique-specific</span>
            </label>
            <label className="sb-wide">
              <span>Start from</span>
              <select value={startFrom} onChange={(e) => void loadFrom(e.target.value)}>
                <option value="">— blank —</option>
                {(catalog?.custom ?? []).map((s) => <option key={`custom/${s.name}`} value={`custom/${s.name}`}>custom / {s.name}</option>)}
                {(catalog?.builtin ?? []).flatMap((d) => d.skills.map((s) => (
                  <option key={`${d.domain}/${s.name}`} value={`${d.domain}/${s.name}`}>{d.domain} / {s.name}{s.origin !== "builtin" ? ` (${s.origin})` : ""}</option>
                )))}
              </select>
              <span className="caption">loads an existing skill's parts into the form to derive a new one</span>
            </label>
          </div>

          {SECTIONS.map((s) => (
            <label key={s.key} className="sb-section">
              <span>{s.title}{(sections[s.key] ?? "").trim() ? "" : <span className="caption"> (empty)</span>}</span>
              <span className="caption">{s.hint}</span>
              <textarea rows={s.key === "implementation" ? 8 : 4} value={sections[s.key] ?? ""}
                onChange={(e) => setSections((m) => ({ ...m, [s.key]: e.target.value }))} />
            </label>
          ))}

          {error && <p className="caption warn">{error}</p>}
          <div className="mem-row-actions sb-actions">
            <button type="button" className="primary small" disabled={!ready || busy !== null}
              title={ready ? "Saves it under this session's custom_skills/ and registers it with the agent; not kept after the session." : "Needs a name, a description and at least one section."}
              onClick={() => void save("session")}>
              {busy === "session" ? "saving…" : "Use in this session"}
            </button>
            <button type="button" className="primary small" disabled={!ready || busy !== null || memoryOn === false}
              title={memoryOn === false ? "Persistent memory is off — turn it on (Memory tab) to save into the store."
                : ready ? "Writes an approved, authored skill bundle into persistent memory: it loads in every future run." : "Needs a name, a description and at least one section."}
              onClick={() => void save("memory")}>
              {busy === "memory" ? "saving…" : "Save to persistent memory"}
            </button>
            {memoryOn === false && onOpenMemory && (
              <button type="button" className="link-btn inline" onClick={onOpenMemory}>memory is off — turn it on</button>
            )}
            <button type="button" className="link-btn" disabled={!preview} onClick={download}>download .md</button>
            <label className="mem-check"><input type="checkbox" checked={showRaw} onChange={(e) => setShowRaw(e.target.checked)} /><span>raw markdown</span></label>
            <span className="caption">{filled} of {SECTIONS.length} sections written</span>
          </div>

          {preview && (
            <div className="sb-preview md-body">
              <div className="caption">preview — the file exactly as the agents will read it</div>
              {showRaw ? <pre className="mem-diff">{preview}</pre> : (
                <>
                  {front && <pre className="skill-front caption">{front[1].trim()}</pre>}
                  <MarkdownBody text={front ? preview.slice(front[0].length) : preview} />
                </>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
