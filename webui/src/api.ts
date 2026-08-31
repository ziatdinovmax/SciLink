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

export interface AppConfig {
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

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not json */
    }
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
    return r.json() as Promise<{
      paths: string[];
      series_dir: string | null;
      global_metadata: string | null;
      category: string;
    }>;
  },

  fileUrl: (id: string, relPath: string) =>
    `${BASE}/sessions/${id}/files?path=${encodeURIComponent(relPath)}`,

  fetchFileText: async (id: string, relPath: string) => {
    const r = await fetch(api.fileUrl(id, relPath));
    if (!r.ok) throw new Error(`Could not load ${relPath}`);
    return r.text();
  },
};
