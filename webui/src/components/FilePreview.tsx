import { useEffect, useMemo, useState } from "react";
import {
  api,
  type ProvenanceEvent,
  type TableData,
  type TreeEntry,
} from "../api";
import { useUIActions } from "../UIContext";
import { demoteHeadings, MarkdownBody } from "./MarkdownBody";

const IMAGE_EXTS = ["png", "jpg", "jpeg"];
const ARRAY_EXTS = ["npy", "tif", "tiff"];
const TABLE_EXTS = ["csv", "tsv", "xlsx"];
const CODE_EXTS = ["py", "sh", "yaml", "yml", "toml", "log", "txt", "jsonl", "cif", "xyz", "in"];
const CMAPS = ["viridis", "gray", "magma", "plasma", "inferno"];

function ext(path: string): string {
  const name = path.split("/").pop() ?? "";
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i + 1).toLowerCase() : "";
}

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function FilePreview({
  sessionId,
  entry,
  provenance,
}: {
  sessionId: string;
  entry: TreeEntry;
  provenance: ProvenanceEvent[];
}) {
  const { attachToChat } = useUIActions();
  const path = entry.path;
  const e = ext(path);
  const url = api.fileUrl(sessionId, path);

  const producedBy = useMemo(
    () => provenance.find((ev) => ev.files.includes(path)),
    [provenance, path],
  );

  return (
    <div className="preview-pane">
      <div className="preview-header">
        <div className="preview-title">
          <strong>{entry.name}</strong>
          <span className="caption">
            {" "}
            · {fmtSize(entry.size)} ·{" "}
            {new Date(entry.mtime * 1000).toLocaleString()}
            {entry.new && <span className="new-badge">new</span>}
          </span>
          {producedBy && (
            <div className="caption provenance-line">
              produced by <code>{producedBy.tool}</code>
              {producedBy.ts && ` · ${producedBy.ts.replace("T", " ")}`}
              {producedBy.summary && ` — ${producedBy.summary}`}
            </div>
          )}
        </div>
        <div className="preview-actions">
          <button onClick={() => attachToChat(path)} title="Reference this file in the chat input">
            Attach to chat
          </button>
          <a href={url} download={entry.name}>
            <button>Download</button>
          </a>
        </div>
      </div>
      <div className="preview-body">
        <PreviewBody sessionId={sessionId} path={path} e={e} url={url} />
      </div>
    </div>
  );
}

function PreviewBody({
  sessionId,
  path,
  e,
  url,
}: {
  sessionId: string;
  path: string;
  e: string;
  url: string;
}) {
  if (IMAGE_EXTS.includes(e)) return <ImagePreview url={url} />;
  if (ARRAY_EXTS.includes(e)) return <ArrayPreview sessionId={sessionId} path={path} />;
  if (TABLE_EXTS.includes(e)) return <TablePreview sessionId={sessionId} path={path} />;
  if (e === "md") return <MdPreview sessionId={sessionId} path={path} />;
  if (e === "html" || e === "htm")
    return <iframe className="preview-frame" src={url} title={path} sandbox="allow-scripts" />;
  if (e === "pdf")
    return <iframe className="preview-frame tall" src={url} title={path} />;
  if (e === "json") return <TextPreview url={url} pretty />;
  // code, extension-less files (POSCAR/INCAR/...), and anything text-ish
  if (CODE_EXTS.includes(e) || e === "") return <TextPreview url={url} />;
  return (
    <p className="caption">
      No inline preview for .{e} — use Download.
    </p>
  );
}

function ImagePreview({ url }: { url: string }) {
  const [zoom, setZoom] = useState(1);
  return (
    <div>
      <div className="preview-toolbar">
        <button onClick={() => setZoom((z) => Math.max(0.25, z / 1.5))}>−</button>
        <span className="caption">{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom((z) => Math.min(8, z * 1.5))}>+</button>
        <button onClick={() => setZoom(1)}>Fit</button>
      </div>
      <div className="image-scroll">
        <img
          src={url}
          alt=""
          style={
            zoom === 1
              ? { maxWidth: "100%" }
              : { width: `${zoom * 100}%`, maxWidth: "none" }
          }
        />
      </div>
    </div>
  );
}

function ArrayPreview({ sessionId, path }: { sessionId: string; path: string }) {
  const [cmap, setCmap] = useState("viridis");
  const [url, setUrl] = useState<string | null>(null);
  // "heatmap" shows the colormap selector; "line"/"image" renders don't use
  // one; null = unrenderable (message in `error`).
  const [kind, setKind] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let revoked: string | null = null;
    let cancelled = false;
    fetch(api.thumbUrl(sessionId, path, 1024, cmap))
      .then(async (r) => {
        if (!r.ok) {
          let detail = r.statusText;
          try {
            detail = (await r.json()).detail ?? detail;
          } catch {
            /* not json */
          }
          throw new Error(detail);
        }
        const blob = await r.blob();
        if (cancelled) return;
        revoked = URL.createObjectURL(blob);
        setUrl(revoked);
        setKind(r.headers.get("X-Preview-Kind") ?? "heatmap");
        setError(null);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setUrl(null);
          setKind(null);
        }
      });
    return () => {
      cancelled = true;
      if (revoked) URL.revokeObjectURL(revoked);
    };
  }, [sessionId, path, cmap]);

  if (error)
    return (
      <p className="caption" style={{ padding: "8px 0" }}>
        {error}
      </p>
    );
  return (
    <div>
      {kind === "heatmap" && (
        <div className="preview-toolbar">
          <span className="caption">colormap</span>
          <select value={cmap} onChange={(ev) => setCmap(ev.target.value)} style={{ width: "auto" }}>
            {CMAPS.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </div>
      )}
      <div className="image-scroll">
        {url ? (
          <img src={url} alt="" style={{ maxWidth: "100%" }} />
        ) : (
          <p className="caption">Loading…</p>
        )}
      </div>
    </div>
  );
}

function TablePreview({ sessionId, path }: { sessionId: string; path: string }) {
  const [data, setData] = useState<TableData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [sortAsc, setSortAsc] = useState(true);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    setData(null);
    setSortCol(null);
    setFilter("");
    api.table(sessionId, path).then(setData).catch((err) => setError(String(err)));
  }, [sessionId, path]);

  const rows = useMemo(() => {
    if (!data) return [];
    let r = data.rows;
    if (filter.trim()) {
      const q = filter.toLowerCase();
      r = r.filter((row) => row.some((c) => String(c ?? "").toLowerCase().includes(q)));
    }
    if (sortCol !== null) {
      r = [...r].sort((a, b) => {
        const av = a[sortCol], bv = b[sortCol];
        const cmp =
          typeof av === "number" && typeof bv === "number"
            ? av - bv
            : String(av ?? "").localeCompare(String(bv ?? ""));
        return sortAsc ? cmp : -cmp;
      });
    }
    return r;
  }, [data, filter, sortCol, sortAsc]);

  if (error) return <p className="caption">{error}</p>;
  if (!data) return <p className="caption">Loading…</p>;
  return (
    <div>
      <div className="preview-toolbar">
        <input
          type="text"
          placeholder="Filter rows…"
          value={filter}
          onChange={(ev) => setFilter(ev.target.value)}
          style={{ maxWidth: 220 }}
        />
        <span className="caption">
          {rows.length} of {data.total_rows} rows{data.truncated ? " (head)" : ""}
        </span>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {data.columns.map((c, i) => (
                <th
                  key={c}
                  onClick={() => {
                    if (sortCol === i) setSortAsc(!sortAsc);
                    else {
                      setSortCol(i);
                      setSortAsc(true);
                    }
                  }}
                >
                  {c} {sortCol === i ? (sortAsc ? "▲" : "▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 500).map((row, ri) => (
              <tr key={ri}>
                {row.map((c, ci) => (
                  <td key={ci}>{c === null ? "" : String(c)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MdPreview({ sessionId, path }: { sessionId: string; path: string }) {
  const [text, setText] = useState<string | null>(null);
  const [mode, setMode] = useState<"rendered" | "source">("rendered");
  useEffect(() => {
    setText(null);
    api.fetchFileText(sessionId, path)
      .then((t) => setText(t.slice(0, 100_000)))
      .catch((err) => setText(`Could not load: ${err}`));
  }, [sessionId, path]);
  const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/") + 1) : "";
  if (text === null) return <p className="caption">Loading…</p>;
  return (
    <div>
      <div className="preview-toolbar">
        <button className={mode === "rendered" ? "primary" : ""} onClick={() => setMode("rendered")}>
          Rendered
        </button>
        <button className={mode === "source" ? "primary" : ""} onClick={() => setMode("source")}>
          Source
        </button>
      </div>
      {mode === "rendered" ? (
        <MarkdownBody
          text={demoteHeadings(text)}
          transformImageUri={(src) =>
            /^(https?:|data:)/.test(src) ? src : api.fileUrl(sessionId, dir + src)
          }
        />
      ) : (
        <pre className="text-preview">{text}</pre>
      )}
    </div>
  );
}

function TextPreview({ url, pretty = false }: { url: string; pretty?: boolean }) {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    setText(null);
    fetch(url)
      .then((r) => r.text())
      .then((t) => {
        if (pretty) {
          try {
            t = JSON.stringify(JSON.parse(t), null, 2);
          } catch {
            /* leave as-is */
          }
        }
        setText(t.slice(0, 200_000));
      })
      .catch((err) => setText(`Could not load: ${err}`));
  }, [url, pretty]);
  if (text === null) return <p className="caption">Loading…</p>;
  return <pre className="text-preview">{text}</pre>;
}
