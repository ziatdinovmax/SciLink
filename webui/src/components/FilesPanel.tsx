import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type ProvenanceEvent, type TreeEntry } from "../api";
import { FilePreview } from "./FilePreview";

const FILE_ICONS: Record<string, string> = {
  png: "🖼", jpg: "🖼", jpeg: "🖼", tif: "🖼", tiff: "🖼",
  csv: "📊", tsv: "📊", xlsx: "📊",
  json: "📋", npy: "🔢",
  html: "📄", pdf: "📕", txt: "📝", md: "📝", log: "📝",
  py: "🐍",
};

const IMAGE_EXTS = new Set(["png", "jpg", "jpeg", "tif", "tiff", "npy"]);

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i + 1).toLowerCase() : "";
}

function flatten(entries: TreeEntry[]): TreeEntry[] {
  const out: TreeEntry[] = [];
  for (const e of entries) {
    if (e.is_dir) out.push(...flatten(e.children ?? []));
    else out.push(e);
  }
  return out;
}

function findEntry(entries: TreeEntry[], path: string): TreeEntry | null {
  for (const e of entries) {
    if (e.path === path) return e;
    if (e.is_dir && path.startsWith(e.path + "/")) {
      const hit = findEntry(e.children ?? [], path);
      if (hit) return hit;
    }
  }
  return null;
}

type ViewMode = "tree" | "gallery" | "timeline";

export function FilesPanel({
  sessionId,
  filesVersion,
  selectedPath,
  onSelect,
}: {
  sessionId: string;
  filesVersion: number; // bumped on files_changed / turn completion
  selectedPath: string | null;
  onSelect: (path: string | null) => void;
}) {
  const [entries, setEntries] = useState<TreeEntry[]>([]);
  const [truncated, setTruncated] = useState(false);
  const [provenance, setProvenance] = useState<ProvenanceEvent[]>([]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState("");
  const [recentFirst, setRecentFirst] = useState(false);
  const [view, setView] = useState<ViewMode>("tree");

  useEffect(() => {
    api
      .tree(sessionId)
      .then((d) => {
        setEntries(d.entries);
        setTruncated(d.truncated);
      })
      .catch(() => {});
    api
      .provenance(sessionId)
      .then((d) => setProvenance(d.events))
      .catch(() => setProvenance([]));
  }, [sessionId, filesVersion]);

  // Auto-expand ancestors of an externally selected file ("Open in Files").
  useEffect(() => {
    if (!selectedPath) return;
    const parts = selectedPath.split("/").slice(0, -1);
    if (!parts.length) return;
    setExpanded((prev) => {
      const next = new Set(prev);
      let acc = "";
      for (const p of parts) {
        acc = acc ? `${acc}/${p}` : p;
        next.add(acc);
      }
      return next;
    });
  }, [selectedPath]);

  const selected = useMemo(
    () => (selectedPath ? findEntry(entries, selectedPath) : null),
    [entries, selectedPath],
  );

  const allFiles = useMemo(() => flatten(entries), [entries]);
  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    let files = q
      ? allFiles.filter((f) => f.path.toLowerCase().includes(q))
      : allFiles;
    if (recentFirst) files = [...files].sort((a, b) => b.mtime - a.mtime);
    return files;
  }, [allFiles, filter, recentFirst]);

  const toggle = useCallback((path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const renderTree = (list: TreeEntry[], depth: number) =>
    list.map((e) => (
      <div key={e.path}>
        <div
          className={`tree-row${e.path === selectedPath ? " selected" : ""}`}
          style={{ paddingLeft: 8 + depth * 16 }}
          onClick={() => (e.is_dir ? toggle(e.path) : onSelect(e.path))}
        >
          {e.is_dir ? (
            <span>{expanded.has(e.path) ? "📂" : "📁"} {e.name}
              <span className="caption"> ({(e.children ?? []).length})</span>
            </span>
          ) : (
            <span>
              {FILE_ICONS[extOf(e.name)] ?? "📄"} {e.name}
              {e.new && <span className="new-badge">new</span>}
            </span>
          )}
          {e.is_dir && (
            <a
              href={api.zipUrl(sessionId, e.path)}
              download
              className="zip-link"
              title="Download folder as zip"
              onClick={(ev) => ev.stopPropagation()}
            >
              ⬇
            </a>
          )}
        </div>
        {e.is_dir && expanded.has(e.path) && renderTree(e.children ?? [], depth + 1)}
      </div>
    ));

  const useFlatList = filter.trim() !== "" || recentFirst;

  return (
    <div className="files-panel">
      <div className="files-tree">
        <div className="files-toolbar">
          <input
            type="text"
            placeholder="Filter files…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          <label className="caption" style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={recentFirst}
              onChange={(e) => setRecentFirst(e.target.checked)}
              style={{ width: "auto" }}
            />
            recent first
          </label>
        </div>
        <div className="files-viewmodes">
          {(["tree", "gallery", "timeline"] as ViewMode[]).map((m) => (
            <button
              key={m}
              className={view === m ? "primary" : ""}
              onClick={() => setView(m)}
            >
              {m}
            </button>
          ))}
          <a href={api.zipUrl(sessionId, "")} download title="Download whole session as zip">
            <button>zip all</button>
          </a>
        </div>
        <div className="files-scroll">
          {view === "timeline" ? (
            <Timeline provenance={provenance} onSelect={onSelect} />
          ) : view === "gallery" ? (
            <Gallery
              sessionId={sessionId}
              files={filtered.filter((f) => IMAGE_EXTS.has(extOf(f.name)))}
              selectedPath={selectedPath}
              onSelect={onSelect}
            />
          ) : useFlatList ? (
            filtered.map((f) => (
              <div
                key={f.path}
                className={`tree-row${f.path === selectedPath ? " selected" : ""}`}
                style={{ paddingLeft: 8 }}
                onClick={() => onSelect(f.path)}
                title={f.path}
              >
                <span>
                  {FILE_ICONS[extOf(f.name)] ?? "📄"} {f.path}
                  {f.new && <span className="new-badge">new</span>}
                </span>
              </div>
            ))
          ) : (
            renderTree(entries, 0)
          )}
          {truncated && (
            <p className="caption">Listing truncated (very large session).</p>
          )}
        </div>
      </div>
      <div className="files-preview">
        {selected && !selected.is_dir ? (
          <FilePreview
            key={selected.path}
            sessionId={sessionId}
            entry={selected}
            provenance={provenance}
          />
        ) : (
          <p className="caption" style={{ padding: 20 }}>
            Select a file to preview it. Files produced by the running turn
            are badged <span className="new-badge">new</span> and the tree
            updates live while the agent works.
          </p>
        )}
      </div>
    </div>
  );
}

function Gallery({
  sessionId,
  files,
  selectedPath,
  onSelect,
}: {
  sessionId: string;
  files: TreeEntry[];
  selectedPath: string | null;
  onSelect: (path: string) => void;
}) {
  if (!files.length) return <p className="caption">No images in this session yet.</p>;
  return (
    <div className="gallery-grid">
      {files.map((f) => (
        <figure
          key={f.path}
          className={f.path === selectedPath ? "selected" : ""}
          onClick={() => onSelect(f.path)}
          title={f.path}
        >
          <img
            src={api.thumbUrl(sessionId, f.path, 160)}
            alt={f.name}
            loading="lazy"
            onError={(e) => {
              // Unrenderable array: show the name-only card, not a broken icon.
              (e.currentTarget as HTMLImageElement).style.display = "none";
            }}
          />
          <figcaption>
            {f.name}
            {f.new && <span className="new-badge">new</span>}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

function Timeline({
  provenance,
  onSelect,
}: {
  provenance: ProvenanceEvent[];
  onSelect: (path: string) => void;
}) {
  if (!provenance.length)
    return (
      <p className="caption">
        No tool-call history yet — the timeline fills in as the agent runs
        (from the session event log).
      </p>
    );
  return (
    <div className="timeline">
      {provenance.map((ev, i) => (
        <div className="timeline-item" key={`${ev.log}-${ev.n}-${i}`}>
          <div>
            <code>{ev.tool}</code>
            <span className={`caption status-${ev.status}`}> {ev.status}</span>
            <span className="caption"> · {ev.ts?.replace("T", " ")}</span>
            {ev.log && <span className="caption"> · {ev.log}</span>}
          </div>
          {ev.summary && <div className="caption">{ev.summary}</div>}
          {ev.files.length > 0 && (
            <div className="timeline-files">
              {ev.files.map((f) => (
                <button key={f} className="file-chip-btn" onClick={() => onSelect(f)} title={f}>
                  {f.split("/").pop()}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
