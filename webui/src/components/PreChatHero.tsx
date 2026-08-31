import { useState } from "react";
import { api } from "../api";
import { Dropzone } from "./Dropzone";

/** Pre-chat start forms — ports of chat_uploads.py's three hero zones,
 * composing the same dispatch prompts from the server-side saved paths. */

const DATA_ACCEPT = ".tif,.tiff,.png,.jpg,.npy,.csv,.txt,.tsv,.xlsx,.h5,.hdf5,.nxs";
const METADATA_ACCEPT = ".json,.txt";
const KNOWLEDGE_ACCEPT =
  ".pdf,.txt,.md,.docx,.png,.jpg,.jpeg,.tif,.tiff,.csv,.xlsx,.tsv,.json";
const CODE_ACCEPT = ".py,.txt,.md,.json,.yaml,.yml";
const PLANNING_DATA_ACCEPT = ".csv,.xlsx,.tsv,.txt,.npy,.json";

export function PreChatHero({
  mode,
  sessionId,
  onStart,
}: {
  mode: string;
  sessionId: string;
  onStart: (prompt: string) => void;
}) {
  if (mode === "analyze")
    return <AnalyzeHero sessionId={sessionId} onStart={onStart} />;
  if (mode === "plan")
    return <PlanHero sessionId={sessionId} onStart={onStart} />;
  return <MetaHero sessionId={sessionId} onStart={onStart} />;
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

  const start = () => {
    // Port of chat_uploads.py:83-100.
    let prompt: string;
    if (dataPath && metaPath) {
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
      <div className="upload-hero-box">
        <p className="upload-hero-title">Upload your data to get started</p>
        <p className="upload-hero-subtitle">Images, CSV, NumPy arrays, and more</p>
      </div>
      <div className="uploader-row">
        <Dropzone
          label="Data file(s) — click or drop"
          accept={DATA_ACCEPT}
          onFiles={async (files) => {
            const r = await api.upload(sessionId, "data", files);
            setDataPath(r.series_dir ?? r.paths[0]);
            setIsSeries(Boolean(r.series_dir));
            return files.map((f) => f.name);
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
  onStart,
}: {
  sessionId: string;
  onStart: (prompt: string) => void;
}) {
  const [objective, setObjective] = useState("");
  const [knowledge, setKnowledge] = useState<string[]>([]);
  const [code, setCode] = useState<string[]>([]);
  const [data, setData] = useState<string[]>([]);

  const canStart =
    objective.trim().length > 0 ||
    knowledge.length > 0 ||
    code.length > 0 ||
    data.length > 0;

  const start = () => {
    // Port of the uploaded-files branches of chat_uploads.py:196-267
    // (pasted-folder paths are a follow-up).
    const parts: string[] = [];
    const quote = (ps: string[]) => ps.map((p) => `\`${p}\``).join(", ");
    if (objective.trim()) parts.push(`Research objective: ${objective.trim()}`);
    if (knowledge.length) parts.push(`Knowledge files: ${quote(knowledge)}`);
    if (code.length) parts.push(`Code files: ${quote(code)}`);
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
    onStart(parts.length ? parts.join("\n\n") : "Please help me plan my experiment.");
  };

  const uploader =
    (category: string, setter: (fn: (prev: string[]) => string[]) => void) =>
    async (files: File[]) => {
      const r = await api.upload(sessionId, category, files);
      setter((prev) => [...prev, ...r.paths]);
      return files.map((f) => f.name);
    };

  return (
    <div className="hero-wrap">
      <label className="field">
        <span>Research objective</span>
        <textarea
          value={objective}
          placeholder="e.g., Optimize reaction yield for polymer synthesis"
          onChange={(e) => setObjective(e.target.value)}
        />
      </label>
      <div className="upload-hero-box">
        <p className="upload-hero-title">Upload resources for the planning agent</p>
        <p className="upload-hero-subtitle">Papers, images, code, and experimental data</p>
      </div>
      <details className="card hero-accordion" open>
        <summary>Knowledge (papers, images)</summary>
        <div className="card-body">
          <Dropzone label="Upload knowledge files" accept={KNOWLEDGE_ACCEPT}
            onFiles={uploader("knowledge", setKnowledge)} />
        </div>
      </details>
      <details className="card hero-accordion">
        <summary>Code (scripts, API docs)</summary>
        <div className="card-body">
          <Dropzone label="Upload code files" accept={CODE_ACCEPT}
            onFiles={uploader("code", setCode)} />
        </div>
      </details>
      <details className="card hero-accordion">
        <summary>Data (experimental results)</summary>
        <div className="card-body">
          <Dropzone label="Upload data files" accept={PLANNING_DATA_ACCEPT}
            onFiles={uploader("planning_data", setData)} />
        </div>
      </details>
      <button
        className="primary"
        style={{ width: "100%", marginTop: 8 }}
        disabled={!canStart}
        onClick={start}
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
  onStart,
}: {
  sessionId: string;
  onStart: (prompt: string) => void;
}) {
  const [goal, setGoal] = useState("");
  const [uploads, setUploads] = useState<string[]>([]);
  const canStart = goal.trim().length > 0 || uploads.length > 0;

  const start = () => {
    // Port of chat_uploads.py:358-380 (folder paths are a follow-up).
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
    onStart(parts.length ? parts.join("\n\n") : "Please help with my research.");
  };

  return (
    <div className="hero-wrap">
      <div className="upload-hero-box">
        <p className="upload-hero-title">
          What would you like to do? <span className="beta-pill">BETA</span>
        </p>
        <p className="upload-hero-subtitle">
          Describe your research goal — mission control agent will route it to
          the specialist agents
        </p>
      </div>
      <label className="field">
        <span>Research goal</span>
        <textarea
          value={goal}
          style={{ minHeight: 110 }}
          placeholder="e.g., Analyze the STEM image I uploaded, then plan a follow-up experiment campaign based on what you find"
          onChange={(e) => setGoal(e.target.value)}
        />
      </label>
      <details className="card hero-accordion" open>
        <summary>Add files (optional) — papers, code, data, metadata</summary>
        <div className="card-body">
          <Dropzone
            label="One drop zone for everything — the meta-agent routes each file"
            onFiles={async (files) => {
              const r = await api.upload(sessionId, "meta", files);
              setUploads((prev) => [...prev, ...r.paths]);
              return files.map((f) => f.name);
            }}
          />
        </div>
      </details>
      <button
        className="primary"
        style={{ width: "100%", marginTop: 8 }}
        disabled={!canStart}
        onClick={start}
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
