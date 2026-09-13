import { useEffect, useState } from "react";
import {
  api,
  type AppConfig,
  type LiveSession,
  type ResumableSession,
  type SessionSnapshot,
  type DelegationView,
} from "../api";
import { DelegationTree } from "./DelegationTree";
import logoDark from "../assets/scilink_logo_dark_animated.svg";
import logoLight from "../assets/scilink_logo_light_animated.svg";

export interface SidebarConfig {
  model: string;
  autonomy: string;
  consent: boolean;
  apiKey: string;
  baseUrl: string;
  providerFields: Record<string, string>;
  fhApiKey: string;
  mpApiKey: string;
  embeddingModel: string;
  embeddingApiKey: string;
  embeddingBaseUrl: string;
  objective: string;
}

export function Sidebar({
  config,
  mode,
  session,
  sessionName,
  status,
  locked,
  onStart,
  onResume,
  onRename,
  onReset,
  onQuit,
  liveSessions,
  delegations = null,
  onSelectDelegation,
  onAttachSession,
  onCloseSession,
  onDetach,
  onCollapse,
  theme,
  onToggleTheme,
  authUser = null,
  onLogout,
}: {
  config: AppConfig | null;
  mode: string;
  session: SessionSnapshot | null;
  sessionName: string | null;
  status: string;
  locked: boolean;
  onStart: (cfg: SidebarConfig) => void;
  onResume: (cfg: SidebarConfig, dir: string) => void;
  onRename: (name: string) => Promise<void>;
  onReset: () => void;
  onQuit: () => void;
  liveSessions: LiveSession[];
  delegations?: DelegationView | null; // meta: live ledger for the tree
  onSelectDelegation?: (index: number) => void;
  onAttachSession: (id: string) => void;
  onCloseSession: (id: string) => void;
  onDetach: () => void;
  onCollapse: () => void;
  theme: "dark" | "light";
  onToggleTheme: () => void;
  authUser?: string | null; // set when the server requires sign-in
  onLogout?: () => void;
}) {
  const models = config?.models ?? [];
  const [model, setModel] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [providerFields, setProviderFields] = useState<Record<string, string>>({});
  const [fhApiKey, setFhApiKey] = useState("");
  const [mpApiKey, setMpApiKey] = useState("");
  // Embedding picker: a preset, "(default)", or "Custom" + a typed name —
  // the same shape as the Streamlit sidebar. `embeddingModel` is the
  // effective name sent to the server.
  const [embeddingPreset, setEmbeddingPreset] = useState("");
  const [customEmbeddingModel, setCustomEmbeddingModel] = useState("");
  // The default is keyword-only, EXCEPT on Amazon Bedrock, where the same AWS
  // credential also covers a first-party embedder — so a Bedrock chat model
  // auto-selects it (the user can still change it). Titan Text Embeddings V2
  // is first-party, needs no separate model-access opt-in, and works through
  // the same Bedrock credential.
  const BEDROCK_EMBED = "bedrock/amazon.titan-embed-text-v2:0";
  const isBedrock = (m: string) => m.startsWith("bedrock/");
  // true once the user picks the embedder themselves — then auto-select stops.
  const [embeddingUserSet, setEmbeddingUserSet] = useState(false);
  const embeddingModel =
    embeddingPreset === "__custom__" ? customEmbeddingModel.trim() : embeddingPreset;
  const [embeddingApiKey, setEmbeddingApiKey] = useState("");
  // Optional endpoint for the embeddings alone (chat keeps its own route).
  const [embeddingBaseUrl, setEmbeddingBaseUrl] = useState("");
  const [embeddingCred, setEmbeddingCred] = useState<{
    env_var: string | null; is_set: boolean; proxied?: boolean;
    endpoint?: "vendor" | "base_url" | "embedding_base_url";
  } | null>(null);
  const [autonomy, setAutonomy] = useState("");
  const [consent, setConsent] = useState(false);
  const [resumable, setResumable] = useState<ResumableSession[]>([]);
  const [resumeChoice, setResumeChoice] = useState("");
  const [nameDraft, setNameDraft] = useState("");
  const [providerInfo, setProviderInfo] = useState(config?.provider ?? null);
  const [credInfo, setCredInfo] = useState(config?.credentials ?? null);

  // While a session is live the fields are locked and must show THAT
  // session's model and autonomy (a reattached or resumed session did not
  // come from this form), not the form's defaults. Credentials are never
  // echoed by the server, so those stay blank.
  const shownModel = session
    ? models.includes(session.model) ? session.model : "__custom__"
    : model;
  const shownCustomModel = session ? session.model : customModel;
  const shownAutonomy = session ? session.autonomy : autonomy;
  const effectiveModel = shownModel === "__custom__" ? shownCustomModel : shownModel;
  const autonomyOptions = config?.autonomy_options[mode] ?? [];

  useEffect(() => {
    if (config && !model) setModel(config.models[0]);
  }, [config, model]);

  useEffect(() => {
    if (!autonomyOptions.includes(autonomy))
      setAutonomy(autonomyOptions[0] ?? "");
  }, [mode, autonomyOptions, autonomy]);

  // Provider-specific fields + credential availability follow the model.
  useEffect(() => {
    if (!effectiveModel) return;
    api
      .config(effectiveModel, baseUrl)
      .then((c) => {
        setProviderInfo(c.provider);
        setCredInfo(c.credentials);
        setProviderFields((prev) => {
          const next: Record<string, string> = {};
          for (const f of c.provider.fields)
            next[f.name] = prev[f.name] ?? f.default;
          return next;
        });
      })
      .catch(() => {});
  }, [effectiveModel, baseUrl]);

  // Auto-select the embedder from the chat provider until the user overrides:
  // Bedrock ⇒ Titan (same credential, dense retrieval for free), otherwise
  // keyword-only. A resumed/locked session keeps its persisted choice.
  useEffect(() => {
    if (locked || embeddingUserSet || !effectiveModel) return;
    setEmbeddingPreset(isBedrock(effectiveModel) ? BEDROCK_EMBED : "");
  }, [effectiveModel, locked, embeddingUserSet]);

  // The embedding key's availability follows the embedding model's vendor.
  useEffect(() => {
    if (!embeddingModel) {
      setEmbeddingCred(null);
      return;
    }
    api
      .config(effectiveModel, baseUrl, embeddingModel, embeddingBaseUrl.trim())
      .then((c) => setEmbeddingCred(c.embedding_credential ?? null))
      .catch(() => {});
  }, [embeddingModel, effectiveModel, baseUrl, embeddingBaseUrl]);

  useEffect(() => {
    if (!locked)
      api
        .listSessions(mode)
        .then((r) => setResumable(r.resumable))
        .catch(() => setResumable([]));
  }, [mode, locked]);

  useEffect(() => setNameDraft(sessionName ?? ""), [sessionName]);

  const showEmbedding = mode === "plan" || mode === "meta";

  const gather = (): SidebarConfig => ({
    model: effectiveModel,
    autonomy,
    consent,
    apiKey,
    baseUrl,
    providerFields,
    fhApiKey,
    mpApiKey,
    embeddingModel,
    embeddingApiKey,
    embeddingBaseUrl: embeddingBaseUrl.trim(),
    objective: "",
  });

  const cred = (field: string) => credInfo?.[field];
  const envCaption = (field: string) => {
    const c = cred(field);
    return c?.is_set && c.env_var ? (
      <span className="caption cred-hint">✓ available from <code>{c.env_var}</code></span>
    ) : null;
  };

  return (
    <div className="sidebar">
      <div className="sidebar-top">
        <button
          className="icon-btn"
          title="Hide sidebar"
          onClick={onCollapse}
        >
          ⟨
        </button>
        <button
          className="icon-btn"
          title="Toggle theme"
          onClick={onToggleTheme}
        >
          {theme === "dark" ? "☀️" : "🌙"}
        </button>
      </div>
      {authUser && (
        <p className="caption signed-in">
          Signed in as <strong>{authUser}</strong>
          {onLogout && (
            <>
              {" · "}
              <button type="button" className="link-btn" onClick={onLogout}>
                Sign out
              </button>
            </>
          )}
        </p>
      )}
      {session ? (
        <img
          className="logo"
          src={theme === "dark" ? logoDark : logoLight}
          alt="SciLink"
        />
      ) : (
        <h1>SciLink</h1>
      )}


      <label className="field">
        <span>Model</span>
        <select
          value={shownModel}
          disabled={locked}
          onChange={(e) => setModel(e.target.value)}
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
          <option value="__custom__">Custom</option>
        </select>
      </label>
      {shownModel === "__custom__" && (
        <label className="field">
          <span>Custom model name</span>
          <input
            type="text"
            value={shownCustomModel}
            disabled={locked}
            onChange={(e) => setCustomModel(e.target.value)}
          />
        </label>
      )}

      <label className="field">
        <span>{providerInfo?.key_label ?? "API key"}</span>
        <input
          type="password"
          value={apiKey}
          disabled={locked}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder={cred("api_key")?.is_set ? "(using environment key)" : ""}
        />
        {envCaption("api_key")}
      </label>

      {providerInfo?.fields.map((f) => (
        <label className="field" key={f.name} title={f.help}>
          <span>{f.label}</span>
          {f.kind === "select" ? (
            <select
              value={providerFields[f.name] ?? f.default}
              disabled={locked}
              onChange={(e) =>
                setProviderFields({ ...providerFields, [f.name]: e.target.value })
              }
            >
              {f.options.map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={providerFields[f.name] ?? f.default}
              disabled={locked}
              onChange={(e) =>
                setProviderFields({ ...providerFields, [f.name]: e.target.value })
              }
            />
          )}
        </label>
      ))}

      <label className="field">
        <span>Base URL (optional)</span>
        <input
          type="text"
          value={baseUrl}
          disabled={locked}
          onChange={(e) => setBaseUrl(e.target.value)}
        />
        {envCaption("base_url")}
      </label>

      <label className="field">
        <span>FutureHouse API key (optional)</span>
        <input
          type="password"
          value={fhApiKey}
          disabled={locked}
          onChange={(e) => setFhApiKey(e.target.value)}
        />
        {envCaption("fh")}
      </label>

      <label className="field">
        <span>Materials Project API key (optional)</span>
        <input
          type="password"
          value={mpApiKey}
          disabled={locked}
          onChange={(e) => setMpApiKey(e.target.value)}
        />
        {envCaption("mp")}
      </label>

      {showEmbedding && (
        <>
          <label className="field">
            <span>Embedding model (optional)</span>
            <select
              value={embeddingPreset}
              disabled={locked}
              onChange={(e) => {
                setEmbeddingUserSet(true);
                setEmbeddingPreset(e.target.value);
              }}
            >
              <option value="">(none — keyword-only)</option>
              {[
                ...(isBedrock(effectiveModel) ? [BEDROCK_EMBED] : []),
                ...(config?.embedding_models ?? []),
              ].map((m) => (
                <option key={m}>{m}</option>
              ))}
              <option value="__custom__">Custom</option>
            </select>
            {!embeddingModel && embeddingPreset !== "__custom__" && (
              <span className="caption">
                No embedding model: knowledge bases are searched by keyword (BM25)
              </span>
            )}
          </label>
          {embeddingPreset === "__custom__" && (
            <label className="field">
              <span>Embedding model name</span>
              <input
                type="text"
                value={customEmbeddingModel}
                disabled={locked}
                placeholder="e.g. voyage-3, nomic-embed-text, text-embedding-3-large"
                onChange={(e) => setCustomEmbeddingModel(e.target.value)}
              />
            </label>
          )}
          {embeddingModel && isBedrock(embeddingModel) && (
            <span className="caption">
              Embeddings use your Bedrock (AWS) credential — no separate key
              or endpoint needed.
            </span>
          )}
          {embeddingModel && !isBedrock(embeddingModel) && (
            <label className="field">
              <span>Embedding base URL (optional)</span>
              <input
                type="text"
                value={embeddingBaseUrl}
                disabled={locked}
                placeholder="OpenAI-compatible endpoint for embeddings only"
                onChange={(e) => setEmbeddingBaseUrl(e.target.value)}
              />
            </label>
          )}
          {embeddingModel && embeddingCred?.endpoint === "base_url" && (
            <span className="caption">
              Embeddings go through the base URL with the main API key; the
              model name is sent to the proxy as typed.
            </span>
          )}
          {embeddingModel && !isBedrock(embeddingModel) && embeddingCred?.endpoint !== "base_url" && (
            <label className="field">
              <span>Embedding API key (optional)</span>
              <input
                type="password"
                value={embeddingApiKey}
                disabled={locked}
                placeholder={embeddingCred?.is_set ? "(using environment key)" : ""}
                onChange={(e) => setEmbeddingApiKey(e.target.value)}
              />
              {embeddingCred?.endpoint === "embedding_base_url" ? (
                <span className="caption">
                  Embeddings go to the embedding base URL with this key (the
                  main API key if blank); the model name is sent as typed.
                </span>
              ) : embeddingCred?.is_set && embeddingCred.env_var ? (
                <span className="caption cred-hint">✓ available from <code>{embeddingCred.env_var}</code></span>
              ) : (
                <span className="caption">Leave blank to use the main API key</span>
              )}
            </label>
          )}
        </>
      )}

      <label className="field">
        <span>Autonomy mode</span>
        <select
          value={shownAutonomy}
          disabled={locked}
          onChange={(e) => setAutonomy(e.target.value)}
        >
          {autonomyOptions.map((a) => (
            <option key={a}>{a}</option>
          ))}
        </select>
      </label>

      <label className="field" style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <input
          type="checkbox"
          checked={consent}
          disabled={locked}
          onChange={(e) => setConsent(e.target.checked)}
          style={{ width: "auto", marginTop: 3 }}
        />
        <span style={{ marginBottom: 0 }}>{config?.consent_text ?? ""}</span>
      </label>

      {!locked && (
        <>
          <button
            className="primary"
            disabled={!consent || !effectiveModel}
            onClick={() => onStart(gather())}
          >
            Start Session
          </button>
          {resumable.length > 0 && (
            <div className="sidebar-section">
              <h3>Resume past session</h3>
              <select
                value={resumeChoice}
                onChange={(e) => setResumeChoice(e.target.value)}
              >
                <option value="">— select —</option>
                {resumable.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                    {!s.has_checkpoint ? " (no checkpoint)" : ""}
                  </option>
                ))}
              </select>
              <button
                style={{ marginTop: 8, width: "100%" }}
                disabled={!resumeChoice || !consent}
                onClick={() => onResume(gather(), resumeChoice)}
              >
                Resume Session
              </button>
            </div>
          )}
        </>
      )}

      {session && (
        <div className="sidebar-section">
          <h3>Session</h3>
          <p style={{ margin: "0 0 6px" }}>
            <span className={`status-badge ${status}`}>
              {status === "awaiting_input" ? "awaiting your input" : status}
            </span>
          </p>
          <label className="field">
            <span>Session name</span>
            <input
              type="text"
              value={nameDraft}
              onChange={(e) => setNameDraft(e.target.value)}
              onBlur={() => {
                if (nameDraft.trim() && nameDraft !== sessionName)
                  void onRename(nameDraft.trim());
              }}
            />
          </label>
          <p className="caption" style={{ wordBreak: "break-all" }}>
            {session.id} · {session.model} · {session.autonomy}
          </p>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <button
              style={{ flex: 1 }}
              title="Leave this session running and return to the start screen (reattach any time)"
              onClick={onDetach}
            >
              Detach
            </button>
            <button
              className="danger-hover"
              style={{ flex: 1 }}
              title="Stop the run, close this session, and return to the start screen (the session stays resumable)"
              onClick={onReset}
            >
              Reset Session
            </button>
          </div>
        </div>
      )}

      {session && session.mode === "meta" && (
        <div className="sidebar-section">
          <DelegationTree
            view={delegations}
            running={status === "running"}
            onSelect={(i) => onSelectDelegation?.(i)}
          />
        </div>
      )}

      {session && liveSessions.some((s) => s.id !== session.id) && (
        <div className="sidebar-section">
          <h3>Other live sessions</h3>
          <div className="session-list">
            {liveSessions
              .filter((s) => s.id !== session?.id)
              .map((s) => (
                <div
                  className="session-row"
                  key={s.id}
                  role="button"
                  tabIndex={0}
                  title={`${s.id} — click to attach`}
                  onClick={() => onAttachSession(s.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") onAttachSession(s.id);
                  }}
                >
                  <div className="session-info">
                    {s.name ?? s.id}
                    <span className="caption">
                      {s.status === "awaiting_input"
                        ? "🟠 awaiting input"
                        : s.status === "running"
                          ? "🟢 running"
                          : "⚪ idle"}{" "}
                      · {s.mode} · {s.n_messages} messages
                    </span>
                  </div>
                  <button
                    className="session-close"
                    title="Close this session (stops any run; stays resumable from disk)"
                    onClick={(e) => {
                      e.stopPropagation();
                      onCloseSession(s.id);
                    }}
                  >
                    ✕
                  </button>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="sidebar-section" style={{ marginTop: "auto" }}>
        <button
          className="danger-hover"
          style={{ width: "100%" }}
          title="Shut down the scilink-web server"
          onClick={onQuit}
        >
          Quit App
        </button>
      </div>
    </div>
  );
}
