import { useState } from "react";
import { api, type FolderUploadResult } from "../api";
import { Dropzone } from "./Dropzone";
import { describeFolder, folderPromptLines } from "../folderfiles";

/** Pre-chat start forms — ports of chat_uploads.py's three hero zones,
 * composing the same dispatch prompts from the server-side saved paths.
 * Design: plain centered headline (dashed borders belong to drop targets
 * only), minimal copy. */

const DATA_ACCEPT = ".tif,.tiff,.png,.jpg,.npy,.csv,.txt,.tsv,.xlsx,.h5,.hdf5,.nxs";
const METADATA_ACCEPT = ".json,.txt";
const KNOWLEDGE_ACCEPT =
  ".pdf,.txt,.md,.docx,.png,.jpg,.jpeg,.tif,.tiff,.csv,.xlsx,.tsv,.json";
const CODE_ACCEPT = ".py,.txt,.md,.json,.yaml,.yml";
const PLANNING_DATA_ACCEPT = ".csv,.xlsx,.tsv,.txt,.npy,.json";




export function PreChatHero({
  mode,
  sessionId,
  localFiles = true,
  onStart,
}: {
  mode: string;
  sessionId: string;
  /** Server shares the browser's machine: pasted server-side folder paths
   * are offered. False on a remote deployment (the endpoint refuses too). */
  localFiles?: boolean;
  onStart: (prompt: string) => void;
}) {
  if (mode === "analyze")
    return <AnalyzeHero sessionId={sessionId} onStart={onStart} />;
  if (mode === "plan")
    return <PlanHero sessionId={sessionId} localFiles={localFiles} onStart={onStart} />;
  return <MetaHero sessionId={sessionId} localFiles={localFiles} onStart={onStart} />;
}

function AnalyzeHero({
  sessionId,
  onStart,
}: {
  sessionId: string;
  onStart: (prompt: string) => void;
}) {
  const [dataPath, setDataPath] = useState<string | null>(null);
  const [isSeries, setIsSeries] = useState(false);
  const [metaPath, setMetaPath] = useState<string | null>(null);
  const [hasSidecars, setHasSidecars] = useState(false);
  // A NESTED data folder: no single series dir, so the prompt spells out
  // the subfolders (the analyze agent lists a directory one level deep).
  const [dataFolder, setDataFolder] = useState<FolderUploadResult | null>(null);

  const start = () => {
    // Port of chat_uploads.py:83-100.
    let prompt: string;
    if (dataFolder) {
      prompt =
        `I uploaded a data folder with subfolders.\n${folderPromptLines(dataFolder)}\n\n` +
        "Examine the data: treat each subfolder as its own dataset (a " +
        "series when it holds several files) unless the contents show they " +
        "belong together" +
        (metaPath ? `. Metadata is at \`${metaPath}\`.` : ".");
    } else if (dataPath && metaPath) {
      prompt =
        `I uploaded a data file at \`${dataPath}\` and a metadata file at ` +
        `\`${metaPath}\`. Please examine the data and load the metadata.`;
    } else if (dataPath && hasSidecars) {
      prompt =
        `I uploaded data files at \`${dataPath}\` along with per-file JSON ` +
        `sidecar metadata in the same directory. Please examine the data ` +
        `and load the metadata (pass the directory path \`${dataPath}\` to ` +
        `load_metadata).`;
    } else {
      prompt = `I uploaded a data file at \`${dataPath}\`. Please examine it.`;
    }
    onStart(prompt);
  };

  return (
    <div className="hero-wrap">
      <h2 className="hero-title">Upload your data to get started</h2>
      <p className="hero-sub">Images, CSV, NumPy arrays, and more</p>
      <div className="uploader-row">
        <Dropzone
          label="Data file(s) — drop here or click to browse"
          accept={DATA_ACCEPT}
          onFiles={async (files) => {
            const r = await api.upload(sessionId, "data", files);
            setDataPath(r.series_dir ?? r.paths[0]);
            setIsSeries(Boolean(r.series_dir));
            return files.map((f) => f.name);
          }}
          onFolder={async (_name, entries) => {
            const r = await api.uploadFolder(sessionId, "data", entries);
            if (r.series_dir) {
              // Flat folder = a series, exactly like a multi-file drop.
              setDataPath(r.series_dir);
              setIsSeries(true);
              setDataFolder(null);
              if (r.paths.some((p) => p.endsWith(".json"))) setHasSidecars(true);
            } else if (r.dirs.length > 1) {
              setDataPath(r.root);
              setIsSeries(false);
              setDataFolder(r);
            } else {
              setDataPath(r.paths[0]);
              setIsSeries(false);
              setDataFolder(null);
            }
            return [describeFolder(r.root, r.paths.length, r.dirs.length)];
          }}
        />
        <Dropzone
          label="Metadata (optional)"
          accept={METADATA_ACCEPT}
          onFiles={async (files) => {
            const r = await api.upload(sessionId, "metadata", files);
            if (files.length === 1) setMetaPath(r.paths[0]);
            else {
              setHasSidecars(true);
              if (r.global_metadata) setMetaPath(r.global_metadata);
            }
            return files.map((f) => f.name);
          }}
        />
      </div>
      {isSeries && (
        <p className="caption">
          Multiple files were saved as a series and will be analyzed together.
        </p>
      )}
      {dataFolder && (
        <p className="caption">
          Nested folder: {dataFolder.dirs.length - 1} subfolder(s) will be
          described to the agent so each can be examined.
        </p>
      )}
      <button
        className="primary"
        style={{ width: "100%" }}
        disabled={!dataPath}
        onClick={start}
      >
        Analyze
      </button>
    </div>
  );
}

function PlanHero({
  sessionId,
  localFiles,
  onStart,
}: {
  sessionId: string;
  localFiles: boolean;
  onStart: (prompt: string) => void;
}) {
  const [objective, setObjective] = useState("");
  const [knowledge, setKnowledge] = useState<string[]>([]);
  const [code, setCode] = useState<string[]>([]);
  const [data, setData] = useState<string[]>([]);
  const [kFolder, setKFolder] = useState("");
  const [cFolder, setCFolder] = useState("");
  const [dFolder, setDFolder] = useState("");
  // Uploaded folders (saved under the session) — handled like pasted folder
  // paths from here on: validated + enumerated server-side, plan dirs
  // repointed, subfolders described in the prompt.
  const [uploadedFolders, setUploadedFolders] = useState<[string, string][]>([]);
  // The pasted-path inputs stay hidden until "Use a folder on this machine…"
  // is picked from a dropzone's menu (or a value is already set): the
  // in-place route is the local power-user path, not a peer of upload.
  const [showPath, setShowPath] = useState({ k: false, c: false, d: false });
  const [warnings, setWarnings] = useState<string[]>([]);

  const canStart =
    objective.trim().length > 0 ||
    knowledge.length > 0 ||
    code.length > 0 ||
    data.length > 0 ||
    uploadedFolders.length > 0 ||
    Boolean(kFolder.trim() || cFolder.trim() || dFolder.trim());

  const start = async () => {
    // Port of chat_uploads.py:196-282, folder branches included.
    const parts: string[] = [];
    const quote = (ps: string[]) => ps.map((p) => `\`${p}\``).join(", ");
    if (objective.trim()) parts.push(`Research objective: ${objective.trim()}`);
    if (knowledge.length) parts.push(`Knowledge files: ${quote(knowledge)}`);
    if (code.length) parts.push(`Code files: ${quote(code)}`);

    // Validate pasted folders server-side; warn on the missing ones and
    // proceed with the valid ones (Streamlit behavior).
    // Pasted paths first (they win the plan-dir repoint: stable external
    // folders let KB indexes be reused), then uploaded folders.
    const wanted = (
      [
        ["knowledge", kFolder.trim()],
        ["code", cFolder.trim()],
        ["data", dFolder.trim()],
      ] as [string, string][]
    )
      .filter(([, p]) => p)
      .concat(uploadedFolders);
    // First valid folder per label (for the plan-dir repoint) + all of them
    // (for the prompt).
    const valid: Record<string, string> = {};
    const validAll: Record<string, string[]> = {};
    const subdirNote: Record<string, string> = {};
    let dataInfo: { data_files: string[]; json_files: string[] } | null = null;
    if (wanted.length) {
      try {
        const res = await api.checkFolders(sessionId, wanted.map(([, p]) => p));
        const warn: string[] = [];
        for (let i = 0; i < wanted.length; i++) {
          const [label, p] = wanted[i];
          const r = res.results[i];
          if (r?.is_dir) {
            if (!valid[label]) valid[label] = p;
            validAll[label] = [...(validAll[label] ?? []), p];
            if (label === "data" && !dataInfo) dataInfo = r;
            if (r.subdirs?.length) {
              subdirNote[p] =
                ` (has ${r.subdirs.length} subfolder(s): ` +
                r.subdirs
                  .slice(0, 20)
                  .map((d) => `\`${d.path}\` [${d.n_files}]`)
                  .join(", ") +
                (r.subdirs.length > 20 ? ", …" : "") +
                ")";
            }
          } else warn.push(`Folder not found: ${p}`);
        }
        setWarnings(warn);
      } catch (e) {
        setWarnings([e instanceof Error ? e.message : String(e)]);
      }
      // Repoint the agent's resource dirs at the stable source folders so
      // KB indexes are reused across sessions instead of rebuilt.
      if (Object.keys(valid).length) {
        try {
          await api.setPlanDirs(sessionId, valid);
        } catch {
          /* prompt still carries the paths */
        }
      }
    }
    const withSubs = (p: string) => `\`${p}\`${subdirNote[p] ?? ""}`;
    if (validAll.knowledge)
      parts.push(
        `Knowledge folder${validAll.knowledge.length > 1 ? "s" : ""}: ` +
          validAll.knowledge.map(withSubs).join(", "),
      );
    if (validAll.code)
      parts.push(
        `Code folder${validAll.code.length > 1 ? "s" : ""}: ` +
          validAll.code.map(withSubs).join(", "),
      );
    if (data.length) {
      const dataPaths = data.filter((p) => !p.endsWith(".json"));
      const jsonPaths = data.filter((p) => p.endsWith(".json"));
      if (dataPaths.length) {
        parts.push(`Data files: ${quote(dataPaths)}`);
        if (dataPaths.length > 1 && jsonPaths.length) {
          parts.push(`Conditions/metadata JSON: ${quote(jsonPaths)}`);
          parts.push(
            "Use `analyze_batch` to process these files together, using the " +
              "JSON as the conditions source.",
          );
        } else if (dataPaths.length > 1) {
          parts.push(
            "Use `analyze_batch` to process these files together. If these " +
              "are measurement-only files (e.g., spectra), you will need " +
              "experimental conditions for each file.",
          );
        }
      }
      if (jsonPaths.length && !dataPaths.length)
        parts.push(`Data/metadata files: ${quote(jsonPaths)}`);
    }
    // Pasted data folder: enumerate its tabular files the way the Streamlit
    // hero did (chat_uploads.py:236-266) so multi-file sets get the
    // analyze_batch guidance.
    if (valid.data && dataInfo) {
      const { data_files, json_files } = dataInfo;
      if (data_files.length > 1) {
        parts.push(`Data files: ${quote(data_files)}`);
        if (json_files.length) {
          parts.push(`Conditions/metadata JSON: ${quote(json_files)}`);
          parts.push(
            "Use `analyze_batch` to process these files together, using the " +
              "JSON as the conditions source.",
          );
        } else {
          parts.push(
            "Use `analyze_batch` to process these files together. If these " +
              "are measurement-only files (e.g., spectra), you will need " +
              "experimental conditions for each file.",
          );
        }
      } else if (data_files.length === 1) {
        parts.push(`Data file: \`${data_files[0]}\``);
      } else {
        parts.push(`Data folder: ${withSubs(valid.data)}`);
      }
      if (subdirNote[valid.data] && data_files.length)
        parts.push(`The data folder is nested${subdirNote[valid.data]}`);
    }
    for (const extra of (validAll.data ?? []).slice(1))
      parts.push(`Additional data folder: ${withSubs(extra)}`);
    onStart(parts.length ? parts.join("\n\n") : "Please help me plan my experiment.");
  };

  const uploader =
    (category: string, setter: (fn: (prev: string[]) => string[]) => void) =>
    async (files: File[]) => {
      const r = await api.upload(sessionId, category, files);
      setter((prev) => [...prev, ...r.paths]);
      return files.map((f) => f.name);
    };

  // Folder uploads land under the session and then flow through the same
  // pasted-folder path (validate → enumerate → repoint → prompt).
  const folderUploader =
    (category: string, label: "knowledge" | "code" | "data") =>
    async (_name: string, entries: Parameters<typeof api.uploadFolder>[2]) => {
      const r = await api.uploadFolder(sessionId, category, entries);
      setUploadedFolders((prev) => [...prev, [label, r.root]]);
      return [describeFolder(r.root, r.paths.length, r.dirs.length)];
    };

  return (
    <div className="hero-wrap">
      <h2 className="hero-title">Plan your next experiment</h2>
      <p className="hero-sub">
        Describe a research objective — add papers, code, and data to ground
        the plan
      </p>
      <textarea
        aria-label="Research objective"
        value={objective}
        placeholder="e.g., Optimize reaction yield for polymer synthesis"
        onChange={(e) => setObjective(e.target.value)}
      />
      <details className="card hero-accordion" open>
        <summary>Knowledge (papers, images)</summary>
        <div className="card-body">
          <Dropzone label="Drop files here or click to browse" accept={KNOWLEDGE_ACCEPT}
            onFiles={uploader("knowledge", setKnowledge)}
            onFolder={folderUploader("knowledge", "knowledge")}
            onLocalPath={localFiles ? () => setShowPath((p) => ({ ...p, k: true })) : undefined} />
          {(showPath.k || kFolder) && (
            <input
              type="text"
              className="folder-input"
              autoFocus={showPath.k}
              placeholder="Folder on this machine (used in place) — /path/to/papers/"
              value={kFolder}
              onChange={(e) => setKFolder(e.target.value)}
            />
          )}
        </div>
      </details>
      <details className="card hero-accordion">
        <summary>Code (scripts, API docs)</summary>
        <div className="card-body">
          <Dropzone label="Drop files here or click to browse" accept={CODE_ACCEPT}
            onFiles={uploader("code", setCode)}
            onFolder={folderUploader("code", "code")}
            onLocalPath={localFiles ? () => setShowPath((p) => ({ ...p, c: true })) : undefined} />
          {(showPath.c || cFolder) && (
            <input
              type="text"
              className="folder-input"
              autoFocus={showPath.c}
              placeholder="Folder on this machine (used in place) — /path/to/code/"
              value={cFolder}
              onChange={(e) => setCFolder(e.target.value)}
            />
          )}
        </div>
      </details>
      <details className="card hero-accordion">
        <summary>Data (experimental results)</summary>
        <div className="card-body">
          <Dropzone label="Drop files here or click to browse" accept={PLANNING_DATA_ACCEPT}
            onFiles={uploader("planning_data", setData)}
            onFolder={folderUploader("planning_data", "data")}
            onLocalPath={localFiles ? () => setShowPath((p) => ({ ...p, d: true })) : undefined} />
          {(showPath.d || dFolder) && (
            <input
              type="text"
              className="folder-input"
              autoFocus={showPath.d}
              placeholder="Folder on this machine (used in place) — /path/to/data/"
              value={dFolder}
              onChange={(e) => setDFolder(e.target.value)}
            />
          )}
        </div>
      </details>
      {warnings.map((w) => (
        <p key={w} className="caption warn">{w}</p>
      ))}
      <button
        className="primary"
        style={{ width: "100%", marginTop: 8 }}
        disabled={!canStart}
        onClick={() => void start()}
      >
        Start Planning
      </button>
      {!canStart && (
        <p className="caption" style={{ textAlign: "center" }}>
          Enter a research objective or upload files to begin.
        </p>
      )}
    </div>
  );
}

function MetaHero({
  sessionId,
  localFiles,
  onStart,
}: {
  sessionId: string;
  localFiles: boolean;
  onStart: (prompt: string) => void;
}) {
  const [goal, setGoal] = useState("");
  const [uploads, setUploads] = useState<string[]>([]);
  const [uploadedFolders, setUploadedFolders] = useState<FolderUploadResult[]>([]);
  const [folders, setFolders] = useState("");
  const [showPath, setShowPath] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);
  const canStart =
    goal.trim().length > 0 ||
    uploads.length > 0 ||
    uploadedFolders.length > 0 ||
    folders.trim().length > 0;

  const start = async () => {
    // Port of chat_uploads.py:339-380 (comma-separated folder paths).
    const parts: string[] = [];
    if (goal.trim()) parts.push(goal.trim());
    if (uploads.length) {
      const listed = uploads.map((p) => `  - \`${p}\``).join("\n");
      parts.push(
        `I uploaded ${uploads.length} file(s):\n${listed}\n\n` +
          "Inspect them to determine what each file is, then route them to " +
          "the right specialist.",
      );
    }
    if (uploadedFolders.length) {
      const nested = uploadedFolders.some((r) => r.dirs.length > 1);
      const listed = uploadedFolders.map((r) => folderPromptLines(r)).join("\n\n");
      parts.push(
        `I uploaded ${uploadedFolders.length} folder(s):\n\n${listed}\n\n` +
          "Inspect the folder" +
          (uploadedFolders.length > 1 ? "s" : "") +
          (nested ? " (recursively — it has subfolders)" : "") +
          " to determine what the files are, then route them to the right " +
          "specialist" +
          (nested
            ? "; treat each subfolder as its own dataset unless the contents show they belong together."
            : "."),
      );
    }
    const candidates = folders
      .split(",")
      .map((p) => p.trim())
      .filter(Boolean);
    if (candidates.length) {
      try {
        const res = await api.checkFolders(sessionId, candidates);
        const validRes = res.results.filter((r) => r.is_dir);
        const valid = validRes.map((r) => r.path);
        setWarnings(
          res.results.filter((r) => !r.is_dir).map((r) => `Folder not found: ${r.path}`),
        );
        // Say when a pasted folder is nested: the agent's listing is one
        // level deep, so it must know to inspect recursively.
        const subs = (r: (typeof validRes)[number]) =>
          r.subdirs?.length
            ? ` (nested: ${r.subdirs.length} subfolder(s) — inspect recursively)`
            : "";
        if (valid.length === 1) {
          parts.push(
            `Additional resources are in the folder \`${valid[0]}\`${subs(validRes[0])} — ` +
              "inspect it as well.",
          );
        } else if (valid.length > 1) {
          const listed = validRes.map((r) => `  - \`${r.path}\`${subs(r)}`).join("\n");
          parts.push(
            `Additional resources are in ${valid.length} folders — ` +
              `inspect them as well:\n${listed}`,
          );
        }
      } catch (e) {
        setWarnings([e instanceof Error ? e.message : String(e)]);
      }
    }
    onStart(parts.length ? parts.join("\n\n") : "Please help with my research.");
  };

  return (
    <div className="hero-wrap">
      <h2 className="hero-title">
        What would you like to do?
      </h2>
      <p className="hero-sub">
        Mission control routes your goal — and any files — to the analysis
        and planning specialists
      </p>
      <textarea
        aria-label="Research goal"
        value={goal}
        style={{ minHeight: 110 }}
        placeholder="e.g., Analyze the STEM image I uploaded, then plan a follow-up experiment campaign based on what you find"
        onChange={(e) => setGoal(e.target.value)}
      />
      <details className="card hero-accordion" open>
        <summary>Add files (optional) — papers, code, data, metadata</summary>
        <div className="card-body">
          <Dropzone
            label="Drop files here or click to browse"
            onFiles={async (files) => {
              const r = await api.upload(sessionId, "meta", files);
              setUploads((prev) => [...prev, ...r.paths]);
              return files.map((f) => f.name);
            }}
            onFolder={async (_name, entries) => {
              const r = await api.uploadFolder(sessionId, "meta", entries);
              setUploadedFolders((prev) => [...prev, r]);
              return [describeFolder(r.root, r.paths.length, r.dirs.length)];
            }}
            onLocalPath={localFiles ? () => setShowPath(true) : undefined}
          />
          {(showPath || folders) && (
            <input
              type="text"
              className="folder-input"
              autoFocus={showPath}
              placeholder="Folder(s) on this machine (used in place) — separate multiple with ','"
              value={folders}
              onChange={(e) => setFolders(e.target.value)}
            />
          )}
        </div>
      </details>
      {warnings.map((w) => (
        <p key={w} className="caption warn">{w}</p>
      ))}
      <button
        className="primary"
        style={{ width: "100%", marginTop: 8 }}
        disabled={!canStart}
        onClick={() => void start()}
      >
        Start
      </button>
      {!canStart && (
        <p className="caption" style={{ textAlign: "center" }}>
          Describe a research goal or add files to begin.
        </p>
      )}
    </div>
  );
}
