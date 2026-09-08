/** REST client + shared types for the SciLink web backend (/api/v1). */

export interface ModeInfo {
  key: string;
  label: string;
  beta?: boolean;
  description: string;
}

export interface ProviderField {
  name: string;
  label: string;
  kind: "select" | "text";
  options: string[];
  default: string;
  help: string;
}

export interface AuthInfo {
  auth_required: boolean;
  user: string | null; // null = not signed in
  multi_user: boolean;
  local_files: boolean; // server shares the browser's machine (loopback bind)
}

export interface AppConfig {
  auth: { required: boolean; user: string | null; multi_user: boolean };
  local_files: boolean;
  modes: ModeInfo[];
  models: string[];
  embedding_models: string[];
  autonomy_options: Record<string, string[]>;
  consent_text: string;
  provider: {
    name: string;
    key_label: string;
    fields: ProviderField[];
    cred_error: string;
  };
  credentials: Record<string, { env_var: string | null; is_set: boolean }>;
}

export interface ReportRef {
  path: string;
  name: string;
  title?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  html_reports?: ReportRef[];
  md_reports?: ReportRef[];
  verbose?: string;
}

export interface PresentedQuestion {
  request_id: string;
  kind: string;
  widget:
    | "generic"
    | "dataset_description"
    | "code_review"
    | "keep_revert"
    | "bestofn"
    | "plan_candidates"
    | "fanout_confirm";
  labels: Record<string, string>;
  prompt: string;
  context_display: string;
  preview_images: string[];
  candidate_captions: Record<string, string>;
  code_files: { name: string; content: string }[];
  candidates?: { idx: number; label: string }[];
  judge_pick?: number;
  fanout?: {
    verdict: string | null;
    join_axis: string | null;
    rationale: string | null;
    branches: string[];
  };
  default: string;
}

export interface DelegationRow {
  index: number;
  mode: string; // analysis | planning | simulation | fusion | ...
  label: string;
  task: string;
  status: string; // running | success | error | interrupted | cancelled
  context_from: number[];
  informed_by: string[];
  fanout: boolean;
  fanout_group: string | null;
  labels: string[]; // fusion: the fused branch labels
  timestamp: string | null;
  completed_at: string | null;
  summary: string;
  key_findings: string[];
  files_produced: string[]; // session-relative when inside the session
  n_feature_tables: number;
  warnings: string[];
  error: string | null;
  timed_out: boolean;
  resumed: boolean;
}

export interface DelegationView {
  delegations: DelegationRow[];
  sub_agents: Record<string, string[]>; // specialist -> worker agents used
}

export interface SessionSnapshot {
  id: string;
  mode: string;
  model: string;
  autonomy: string;
  status: "idle" | "running" | "awaiting_input";
  name: string | null;
  session_dir: string;
  chat_messages: ChatMessage[];
  pending_question: PresentedQuestion | null;
  live_log: string;
  live_images: {
    path: string;
    label: string;
    branch: string | null;
    v?: number;
  }[];
  delegations: DelegationView | null; // meta sessions only
  event_cursor: number;
}

export interface LiveSession {
  id: string;
  mode: string;
  model: string;
  status: string;
  name: string | null;
  n_messages: number;
}

export interface ResumableSession {
  id: string;
  label: string;
  has_checkpoint: boolean;
  has_chat_history: boolean;
  summary: {
    analysis_count?: number;
    data_file?: string;
    message_count?: number;
  };
}

export interface TreeEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  mtime: number;
  new: boolean;
  children?: TreeEntry[];
}

export interface ProvenanceEvent {
  n: number | null;
  ts: string | null;
  tool: string | null;
  status: string | null;
  summary: string;
  files: string[];
  log: string;
}

export interface TableData {
  columns: string[];
  rows: (string | number | null)[][];
  total_rows: number;
  truncated: boolean;
}

export interface UploadResult {
  paths: string[];
  series_dir: string | null;
  global_metadata: string | null;
  category: string;
}

/** One file of a folder upload with its path relative to the picked
 * folder's parent (`<folder>/<sub>/<name>`, as `webkitRelativePath`). */
export interface FolderEntry {
  file: File;
  relPath: string;
}

export interface FolderUploadResult extends UploadResult {
  root: string; // absolute path of the saved top-level folder
  dirs: { path: string; n_files: number }[]; // root + every subfolder that got a file
  skipped: { path: string; reason: string }[];
}

export interface FolderCheck {
  path: string;
  is_dir: boolean;
  data_files: string[];
  json_files: string[];
  subdirs: { path: string; n_files: number }[];
}

export interface SkillCatalog {
  builtin: { domain: string; label: string; skills: { name: string; description: string }[] }[];
  custom: { name: string; path: string }[];
  skills_supported: boolean;
}

export interface ToolInventory {
  external: { name: string; description: string }[];
  mcp_servers: { name: string; transport: string; tools: string[] }[];
  mcp_supported: boolean;
}

export interface CreateSessionBody {
  mode: string;
  model: string;
  autonomy: string;
  consent: boolean;
  api_key?: string;
  base_url?: string;
  provider_fields?: Record<string, string>;
  fh_api_key?: string;
  mp_api_key?: string;
  embedding_model?: string | null;
  embedding_api_key?: string | null;
  objective?: string;
  resume_dir?: string | null;
}

const BASE = "/api/v1";

/** Thrown on a 401 so the app can drop to the sign-in screen (a cookie
 * session ends when the server restarts). */
export class UnauthorizedError extends Error {}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    if (r.status === 401) throw new UnauthorizedError(detail);
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  authMe: () => req<AuthInfo>(`/auth/me`),
  login: (token: string) => req<{ user: string }>(`/auth/login`, json({ token })),
  logout: () => req<{ ok: boolean }>(`/auth/logout`, { method: "POST" }),

  config: (model?: string, baseUrl?: string) =>
    req<AppConfig>(
      `/config?model=${encodeURIComponent(model ?? "")}&base_url=${encodeURIComponent(baseUrl ?? "")}`,
    ),

  listSessions: (mode: string) =>
    req<{ live: LiveSession[]; resumable: ResumableSession[] }>(
      `/sessions?mode=${encodeURIComponent(mode)}`,
    ),

  createSession: (body: CreateSessionBody) =>
    req<SessionSnapshot>(`/sessions`, json(body)),

  getSession: (id: string) => req<SessionSnapshot>(`/sessions/${id}`),

  renameSession: (id: string, name: string) =>
    req<{ ok: boolean }>(`/sessions/${id}`, {
      ...json({ name }),
      method: "PATCH",
    }),

  sendMessage: (id: string, content: string) =>
    req<{ status: string }>(`/sessions/${id}/messages`, json({ content })),

  stop: (id: string) => req<{ stopped: boolean }>(`/sessions/${id}/stop`, { method: "POST" }),

  resetSession: (id: string) =>
    req<{ ok: boolean }>(`/sessions/${id}`, { method: "DELETE" }),

  quit: () => req<{ ok: boolean }>(`/quit`, { method: "POST" }),

  sendFeedback: (id: string, requestId: string, response: string) =>
    req<{ ok: boolean }>(
      `/sessions/${id}/feedback`,
      json({ request_id: requestId, response }),
    ),

  upload: async (id: string, category: string, files: File[]) => {
    const form = new FormData();
    form.append("category", category);
    for (const f of files) form.append("files", f);
    const r = await fetch(`${BASE}/sessions/${id}/uploads`, {
      method: "POST",
      body: form,
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
    return r.json() as Promise<UploadResult>;
  },

  // Folder upload: the relative paths ride in their own JSON field (browsers
  // may strip directory parts from multipart filenames); the server keeps
  // the layout under the category root and skips files the category does
  // not accept instead of rejecting the whole folder.
  uploadFolder: async (id: string, category: string, entries: FolderEntry[]) => {
    const form = new FormData();
    form.append("category", category);
    for (const e of entries) form.append("files", e.file, e.file.name);
    form.append("paths", JSON.stringify(entries.map((e) => e.relPath)));
    const r = await fetch(`${BASE}/sessions/${id}/uploads`, {
      method: "POST",
      body: form,
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
    return r.json() as Promise<FolderUploadResult>;
  },

  // `v` (a file mtime) is a cache-busting version token: a figure rewritten
  // in place gets a distinct URL, so the browser re-fetches instead of
  // serving stale bytes for an unchanged path. The backend ignores it.
  fileUrl: (id: string, relPath: string, v?: number) =>
    `${BASE}/sessions/${id}/files?path=${encodeURIComponent(relPath)}` +
    (v ? `&v=${v}` : ""),

  tree: (id: string) =>
    req<{ entries: TreeEntry[]; truncated: boolean }>(`/sessions/${id}/tree`),

  delegations: (id: string) => req<DelegationView>(`/sessions/${id}/delegations`),

  skills: (id: string) => req<SkillCatalog>(`/sessions/${id}/skills`),
  skillMarkdown: async (id: string, domain: string, name: string) => {
    const r = await fetch(
      `${BASE}/sessions/${id}/skills/${encodeURIComponent(domain)}/${encodeURIComponent(name)}`,
    );
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? r.statusText);
    return r.text();
  },
  uploadSkills: async (id: string, files: File[]) => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const r = await fetch(`${BASE}/sessions/${id}/skills`, { method: "POST", body: form });
    if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
    return r.json() as Promise<{
      registered: string[];
      errors: { file: string; error: string }[];
      catalog: SkillCatalog;
    }>;
  },

  tools: (id: string) => req<ToolInventory>(`/sessions/${id}/tools`),
  connectMcp: (
    id: string,
    body: { name: string; transport: string; command?: string; url?: string; headers?: Record<string, string> },
  ) => req<{ registered: number; inventory: ToolInventory }>(`/sessions/${id}/mcp`, json(body)),
  disconnectMcp: (id: string, server: string) =>
    req<{ ok: boolean; inventory: ToolInventory }>(
      `/sessions/${id}/mcp/${encodeURIComponent(server)}`,
      { method: "DELETE" },
    ),

  provenance: (id: string) =>
    req<{ events: ProvenanceEvent[] }>(`/sessions/${id}/provenance`),

  table: (id: string, relPath: string, limit = 500) =>
    req<TableData>(
      `/sessions/${id}/table?path=${encodeURIComponent(relPath)}&limit=${limit}`,
    ),

  thumbUrl: (id: string, relPath: string, size = 256, cmap = "viridis") =>
    `${BASE}/sessions/${id}/thumb?path=${encodeURIComponent(relPath)}&size=${size}&cmap=${cmap}`,

  zipUrl: (id: string, relPath = "") =>
    `${BASE}/sessions/${id}/zip?path=${encodeURIComponent(relPath)}`,

  checkFolders: (id: string, paths: string[]) =>
    req<{ results: FolderCheck[] }>(`/sessions/${id}/folders`, json({ paths })),

  setPlanDirs: (
    id: string,
    dirs: { knowledge?: string; code?: string; data?: string },
  ) => req<{ applied: Record<string, string> }>(`/sessions/${id}/plan_dirs`, json(dirs)),

  fetchFileText: async (id: string, relPath: string) => {
    const r = await fetch(api.fileUrl(id, relPath));
    if (!r.ok) throw new Error(`Could not load ${relPath}`);
    return r.text();
  },
};
