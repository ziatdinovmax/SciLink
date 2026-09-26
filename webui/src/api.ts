/** REST client + shared types for the SciLink web backend (/api/v1). */

export interface ModeInfo {
  key: string;
  label: string;
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
  embedding_credential?: {
    env_var: string | null;
    is_set: boolean;
    proxied?: boolean;
    endpoint?: "vendor" | "base_url" | "embedding_base_url";
  };
}

export interface OpsStatus {
  ok: boolean;
  version: string;
  uptime_s: number;
  workspace?: string;
  state: "idle" | "busy" | "draining";
  draining: boolean;
  busy: string[];
  idle_for_s: number;
  sessions_live: number;
}

export interface UsageRow {
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
}

export interface UsageSummary {
  period_start: number;
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  llm_seconds: number;
  by_model: Record<string, UsageRow>;
  by_session: Record<string, UsageRow>;
  budget_tokens: number | null;
  remaining_tokens: number | null;
  over_budget: boolean;
  file: string;
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
  /** What the decision is about: the auto-correction a revert would undo,
   * or why an approved plan is being reopened. */
  notice?: { title: string; lines: string[] };
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

/** GET /sessions/{id}/telemetry — the meta session's read-only snapshot
 * (scilink.agents.meta_agent.telemetry.collect_session_telemetry). */
export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  result: unknown;
  status: string; // success | ok | error | pending | …
}

export interface WorkerAction {
  timestamp: string | null;
  action: string;
  status: string;
  rationale: string | null;
  input: unknown;
  result: unknown;
  feedback: unknown;
}

export interface WorkerAgent {
  specialist: string; // analysis | planning | other
  name: string;
  status: string | null;
  action_count: number;
  actions_by_type: Record<string, number>;
  outcomes: { success: number; error: number; other: number };
  first_timestamp: string | null;
  last_timestamp: string | null;
  actions: WorkerAction[];
  source_file: string;
}

export interface AnalysisReport {
  analysis_id: string;
  status: string | null;
  detailed_analysis: string;
  claims: { claim: string | null; impact: string | null }[];
  output_dir: string;
  report_file: string;
}

export interface TelemetrySnapshot {
  meta: {
    meta_mode: string;
    session_dir: string;
    delegations_total: number;
    delegations: unknown[];
  };
  specialists: Record<string, Record<string, unknown>>;
  agents: WorkerAgent[];
  sub_agents: Record<string, string[]>;
  analysis_reports: AnalysisReport[];
  tool_sequence: Record<string, { calls: ToolCall[]; source: string }>;
}


/** Persistent memory (GET /memory and friends — scilink/server/memory_api.py). */
export interface MemoryBankRow {
  id: string; label: string; n_successes: number; n_retrievals: number;
  n_independent?: number | null; n_failures?: number | null;
  sessions: string[]; metric: unknown; created_at: string | null;
  proven: boolean; promoted_to_staging: string | null;
}
export interface MemoryVariantGroup {
  ids: string[]; min_similarity: number | null; suggested_technique: string | null; n_unpromoted: number;
}
export interface MemoryInboxRow {
  id: string; domain: string; technique: string; provenance: string; provenance_label: string;
  metric: string; session: string | null; bank_id: string | null; model: string | null; has_script: boolean;
}
export interface MemoryInboxGroup {
  domain: string; technique: string; ready: boolean;
  records: MemoryInboxRow[]; related: MemoryInboxRow[];
}
export interface MemorySkill {
  name: string; domain: string; path: string; provisional: boolean; provenance: string | null;
  session: string | null; description: string; metric: string; shadows_builtin: boolean;
}
export interface MemoryOverview {
  enabled: boolean; env_override: string | null; home: string;
  consolidate_min_n: number; proven_n: number;
  pipeline: { bank_total: number; bank_proven: number; bank_archived?: number;
              inbox_total: number; inbox_ready: number;
              skills_total: number; skills_provisional: number };
  bank: { domain: string; n_proven: number; records: MemoryBankRow[]; variant_groups: MemoryVariantGroup[] }[];
  inbox: MemoryInboxGroup[];
  skills: MemorySkill[];
}
export interface MemoryBankRecord { id: string; domain: string; fields: Record<string, unknown>; script: string }
export interface MemoryInboxRecord extends MemoryInboxRow {
  fields: Record<string, unknown>; script: string; bank: { bank_id: string; n_successes: number | null } | null;
}
export interface MemoryTarget { domain: string; name: string; builtin: boolean; match: boolean | null }
export interface MemoryJob { id: string; kind: string; label: string; status: "running" | "done" | "error"; result: unknown; error: string | null }
export interface MemoryProposal {
  status: string; domain: string; staged_ids: string[]; target_domain: string; target_name: string;
  builtin_target: boolean; existing_content: string; proposed_content: string; warnings: string[]; diff: string;
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
  embedding_base_url?: string | null;
  objective?: string;
  resume_dir?: string | null;
}

const BASE = "/api/v1";

// ── Live tab ────────────────────────────────────────────────────
export interface LiveParamSpec {
  kind: string;
  low?: number;
  high?: number;
  units?: string;
  description?: string;
  choices?: (string | number)[];
}
export interface LiveInstrumentInfo {
  /** Which instrument this is, stable across runs and sessions. */
  id?: string;
  name: string;
  /** True when a pause really holds the experiment, not only the acquisition. */
  can_pause?: boolean;
  /** What a frame is: a 1D curve, or a datacube (spectrum image). */
  modality?: "curve" | "hyperspectral" | "image";
  technique?: string;
  sample?: string;
  x_axis?: string;
  y_axis?: string;
  about: string;
  schema: Record<string, LiveParamSpec>;
  defaults: Record<string, number | string>;
  outputs: Record<string, string>;
  targets: string[];
  events: { frame: number; what: string }[];
  simulated?: boolean;
  /** Parameters the instrument takes but the loop will not steer, with the reason. */
  held?: string[];
}
/** An instrument this machine remembers from live runs (kept outside any session). */
export interface RememberedInstrument {
  id: string;
  key: string;
  name?: string | null;
  technique?: string | null;
  modality?: string | null;
  first_seen?: string | null;
  last_seen?: string | null;
  recipes: number;
  runs: number;
}

export interface RememberedRecipe {
  recipe_id: string;
  modality?: string | null;
  technique?: string | null;
  sample?: string | null;
  outputs: Record<string, string>;
  reports: string[];
  source?: string | null;
  created?: string | null;
  last_used?: string | null;
  uses: number;
  size_mb: number;
  /** Adopted because two audits rejected the recipe before it, never verified. */
  contested?: boolean;
}

export interface RememberedRun {
  when: string;
  frames: number;
  clean_frames: number;
  reanchors?: number | null;
  audits?: number | null;
  novelties: { step?: number; onset?: string | null; region?: string | null }[];
}

export interface InstrumentMemory {
  instrument: RememberedInstrument;
  recipes: RememberedRecipe[];
  runs: RememberedRun[];
}

/** A curve analysis already in the session that a live loop can adopt as its reference. */
export interface LiveReferenceAnalysis {
  path: string;
  name: string;
  model: string;
  modified: number;
  from_live_run: boolean;
  /** What kind of data the analysis was of. Offered for instruments of the same kind. */
  modality?: "curve" | "image" | "hyperspectral";
  has_data: boolean;
}
export interface LiveFrame {
  step: number;
  features: Record<string, number | null>;
  truth?: Record<string, number>;
  flags: string[];
  latency_s: number | null;
  recipe_id: string | null;
  params: Record<string, number | string>;
  gate: Record<string, number | string | boolean | null>;
  needs_escalation?: boolean;
  escalation?: string | null;
}
export interface LiveRecommendation {
  kind: string;
  params: Record<string, number | string> | null;
  protocol?: string | null;
  rationale: string;
  source?: string;
  based_on_step?: number;
  valid: boolean;
  problems?: string[];
  requires_approval?: boolean;
  rejected_params?: Record<string, number | string>;
  acquisition_skill?: string;
}
export interface LiveEvent {
  event: string;
  step?: number;
  [key: string]: unknown;
}
export interface LiveNovelty {
  step: number;
  since_step: number;
  fraction: number | null;
  from_reference?: number | null;
  where: { kind: "new" | "missing" | "shifted" | "broad" | "window"; x_from: number; x_to: number;
           x_peak: number; share: number; region?: string;
           /** Images: the same stretch as a length scale (one period). */
           length_from?: number; length_to?: number; length_peak?: number; length_units?: string }[];
  /** A datacube is watched by region: where in the field the change is. */
  region?: string | null;
  /** "gradual": the stream moved from its reference without any frame looking new. */
  onset?: "gradual" | null;
  recipe_fits: boolean;
  window_share?: number | null;
  frame_path: string;
  frame_abs_path: string;
}
export interface LiveSnapshot {
  state: "idle" | "arming" | "running" | "paused" | "finishing" | "stopped" | "done" | "error";
  /** The frames have ended and a background analysis is being waited for. */
  finishing?: { mode?: string; reason?: string; profile?: string; started_step?: number; seconds?: number } | null;
  /** The run is waiting for a decision: why, and what the data showed. */
  paused?: (Partial<LiveNovelty> & { why: "novelty" | "breach"; experiment_held: boolean;
                                     timeout_s?: number | null; flags?: string[] }) | null;
  pause_on?: ("novelty" | "breach")[];
  /** Read back from disk after a server restart. Finished, read-only. */
  restored?: boolean;
  /** A closer look at the changed frame is running (claims, then the literature). */
  assessing?: boolean;
  /** What the changed frames turned out to be: claims, and how new each is (1 to 5). */
  discoveries?: { about_step: number; status: string; literature?: string; highest_novelty?: number;
    compared?: Record<string, { recipe: number; analysis: number }>;
    retried?: boolean; analysis_error?: string;
                  claims: { claim: string; question?: string; novelty_score?: number;
                            novelty_explanation?: string }[] }[];
  error?: string | null;
  note?: string | null;
  simulators?: LiveInstrumentInfo[];
  analyses?: LiveReferenceAnalysis[];
  /** MCP servers connected in this session; any tool may be an instrument's acquire. */
  mcp_servers?: { name: string; tools: string[] }[];
  /** What to trace: the named outputs, else the recipe's own quantities. */
  output_keys?: string[];
  run_dir?: string;
  elapsed_s?: number;
  config?: Record<string, unknown>;
  instrument?: LiveInstrumentInfo;
  status?: {
    frames?: number;
    clean_frames?: number;
    flag_counts?: Record<string, number>;
    llm_calls_in_frames?: number;
    latency_s?: { median: number; max: number };
    escalating?: boolean;
    /** What the background is doing: a recipe rebuild or an independent audit. */
    background?: "reanchor" | "audit" | null;
    drift_fraction_bar?: number;
    audits?: number;
    reanchors?: number;
    recipe?: { id?: string; source?: string } | null;
  };
  current_params?: Record<string, number | string>;
  /** null for an open-ended run (until stopped). */
  n_frames_total?: number | null;
  frames?: LiveFrame[];
  events?: LiveEvent[];
  recommendation?: LiveRecommendation | null;
  /** Lasting changes in the data: how much of a frame is new, and where on the axis. */
  novelties?: LiveNovelty[];
  /** The most recent independent audit of the locked recipe's named outputs. */
  last_audit?: {
    step: number;
    audited_step?: number;
    reason: string;
    agrees: boolean;
    /** Two audits split: one sides with the recipe, one does not. Kept, not verified. */
    split?: boolean;
    outputs: Record<string, { locked: number | null; audit: number | null; agrees: boolean;
                              relative_difference?: number }>;
  } | null;
  latest?: { step: number; x: number[]; y: number[]; fit?: (number | null)[] } | null;
  /** A datacube frame's maps, as session files (the tracked outputs' first). */
  maps?: { name: string; path: string; step: number; tracked: boolean; raw?: boolean }[];
}
export interface LiveConfig {
  instrument: string;
  /** null runs until stopped. */
  n_frames: number | null;
  reference_source?: "first_frame" | "analysis";
  /** First-frame reference: how many frames to plan the recipe from (1 to 25). */
  reference_frames?: number;
  /** Seconds a frame may take before it is flagged slow. null = no deadline. */
  frame_deadline_s?: number | null;
  /** Independent audit of the locked recipe every N frames. Omit for none. */
  audit_every?: number;
  /** After a lasting change the recipe still fits: report (default), audit or rebuild. */
  on_change?: "report" | "audit" | "rebuild";
  /** Stop acquiring and wait for a decision when the data changes or the recipe fails. */
  pause_on?: ("novelty" | "breach")[];
  /** Seconds a pause may last before the run goes on unchanged. Omit to wait. */
  pause_timeout_s?: number;
  /** While paused on a change: analyse the frame and ask the literature. Default on. */
  assess_on_pause?: boolean;
  /** Anything the analysis should know. Context, not a constraint. */
  notes?: string;
  /** Keep recipes per instrument across runs, and try the known ones first. */
  remember?: boolean;
  /** Replay or MCP: what a frame is. Omit to read it off the first file or the server. */
  frames_are?: "curve" | "image" | "hyperspectral";
  /** A cube's spectral range or an image's field of view, when the files carry none. */
  frame_metadata?: Record<string, string>;
  reference_analysis?: string;
  interval_s: number;
  apply: "never" | "approved" | "valid";
  recommender: "none" | "gp" | "llm";
  objective_key?: string;
  direction?: "maximize" | "minimize";
  /** More objectives for the GP recommender: a trade-off front is explored. */
  more_objectives?: { key: string; direction: "maximize" | "minimize" }[];
  objective?: string;
  every?: number;
  auto_escalate: boolean;
  reference_profile?: string;
  seed?: number;
  /** instrument === "mcp": a server connected in the MCP tab and its acquire tool. */
  mcp_server?: string;
  mcp_tool?: string;
  /** instrument === "replay": a folder of recorded measurements and what they are. */
  replay_dir?: string;
  system_info?: Record<string, string>;
  outputs?: Record<string, string>;
  targets?: string[];
}

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
  opsStatus: () => req<OpsStatus>(`/ops/status`),
  usage: () => req<UsageSummary>(`/usage`),
  login: (token: string) => req<{ user: string }>(`/auth/login`, json({ token })),
  logout: () => req<{ ok: boolean }>(`/auth/logout`, { method: "POST" }),

  config: (model?: string, baseUrl?: string, embeddingModel?: string, embeddingBaseUrl?: string) =>
    req<AppConfig>(
      `/config?model=${encodeURIComponent(model ?? "")}&base_url=${encodeURIComponent(baseUrl ?? "")}` +
        `&embedding_model=${encodeURIComponent(embeddingModel ?? "")}` +
        `&embedding_base_url=${encodeURIComponent(embeddingBaseUrl ?? "")}`,
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
  telemetry: (id: string) => req<TelemetrySnapshot>(`/sessions/${id}/telemetry`),

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

  // persistent memory (one store per server host, not per session)
  memory: () => req<MemoryOverview>("/memory"),
  setMemoryEnabled: (enabled: boolean) =>
    req<{ enabled: boolean }>("/memory/enabled", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled }) }),
  memorySkillText: async (domain: string, name: string) => {
    const r = await fetch(`${BASE}/memory/skills/${encodeURIComponent(domain)}/${encodeURIComponent(name)}`);
    if (!r.ok) throw new Error((await r.json()).detail ?? r.statusText);
    return r.text();
  },
  memorySkillEdit: (domain: string, name: string, content: string) =>
    req<{ status: string; backup_path: string }>(
      `/memory/skills/${encodeURIComponent(domain)}/${encodeURIComponent(name)}`,
      { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content }) }),
  memorySkillAction: (domain: string, name: string, action: "promote" | "demote" | "prune" | "diff" | "fork") =>
    req<Record<string, unknown>>(
      `/memory/skills/${encodeURIComponent(domain)}/${encodeURIComponent(name)}/${action}`, { method: "POST" }),
  memoryBankRecord: (domain: string, id: string) =>
    req<MemoryBankRecord>(`/memory/bank/${encodeURIComponent(domain)}/${encodeURIComponent(id)}`),
  memoryBankDelete: (domain: string, id: string) =>
    req<{ removed: number }>(`/memory/bank/${encodeURIComponent(domain)}/${encodeURIComponent(id)}`, { method: "DELETE" }),
  memoryBankNominate: (domain: string, id: string) =>
    req<{ status: string; staged_id: string; technique: string }>(
      `/memory/bank/${encodeURIComponent(domain)}/${encodeURIComponent(id)}/nominate`, { method: "POST" }),
  memoryBankNominateGroup: (domain: string, ids: string[], technique: string | null) =>
    req<{ status: string; staged_ids: string[]; technique: string; ready_to_consolidate?: boolean; n_staged_total?: number }>(
      `/memory/bank/${encodeURIComponent(domain)}/nominate-group`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids, technique }) }),
  memoryInboxRecord: (domain: string, id: string) =>
    req<MemoryInboxRecord>(`/memory/inbox/${encodeURIComponent(domain)}/${encodeURIComponent(id)}`),
  memoryInboxDiscard: (domain: string, id: string) =>
    req<{ removed: number }>(`/memory/inbox/${encodeURIComponent(domain)}/${encodeURIComponent(id)}`, { method: "DELETE" }),
  memoryTargets: async (domain: string, ids: string[]) =>
    (await req<{ targets: MemoryTarget[] }>(`/memory/inbox/${encodeURIComponent(domain)}/targets`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids }) })).targets,
  memoryConsolidate: (domain: string, ids: string[], label: string, session_id: string) =>
    req<{ job_id: string; label: string }>(`/memory/inbox/${encodeURIComponent(domain)}/consolidate`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids, label, session_id }) }),
  memoryProposeUpgrade: (domain: string, ids: string[], target_domain: string, target_name: string, session_id: string) =>
    req<{ job_id: string; label: string }>(`/memory/inbox/${encodeURIComponent(domain)}/propose-upgrade`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids, target_domain, target_name, session_id }) }),
  memoryApplyUpgrade: (domain: string, ids: string[], target_domain: string, target_name: string, content: string, fork_builtin: boolean) =>
    req<{ status: string; backup_path: string; n_consumed: number }>(`/memory/inbox/${encodeURIComponent(domain)}/apply-upgrade`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids, target_domain, target_name, content, fork_builtin }) }),
  memoryCheckUpgrade: (existing: string, proposed: string) =>
    req<{ warnings: string[]; diff: string }>("/memory/check-upgrade",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ existing, proposed }) }),
  memoryJob: (id: string) => req<MemoryJob>(`/memory/jobs/${id}`),

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

  liveInstruments: () =>
    req<{ instruments: RememberedInstrument[]; can_forget: boolean }>("/live/instruments"),
  liveInstrument: (instrument: string) =>
    req<InstrumentMemory>(`/live/instruments/${encodeURIComponent(instrument)}`),
  liveForgetRecipe: (instrument: string, recipeId: string) =>
    req<InstrumentMemory>(
      `/live/instruments/${encodeURIComponent(instrument)}/recipes/${encodeURIComponent(recipeId)}`,
      { method: "DELETE" },
    ),
  live: (id: string) => req<LiveSnapshot>(`/sessions/${id}/live`),
  liveStart: (id: string, config: LiveConfig) =>
    req<LiveSnapshot>(`/sessions/${id}/live/start`, json(config)),
  liveStop: (id: string) =>
    req<{ state: string }>(`/sessions/${id}/live/stop`, { method: "POST" }),
  liveParams: (id: string, params: Record<string, number | string>) =>
    req<{ queued: Record<string, number | string> }>(`/sessions/${id}/live/params`, json({ params })),
  liveResume: (id: string, action: "resume" | "stop", params?: Record<string, number | string>) =>
    req<{ decision: string }>(`/sessions/${id}/live/resume`, json({ action, params })),
  liveClear: (id: string) =>
    req<LiveSnapshot>(`/sessions/${id}/live/clear`, { method: "POST" }),

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
