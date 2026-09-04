import { useEffect, useState } from "react";
import {
  api,
  type AppConfig,
  type LiveSession,
  type ResumableSession,
  type SessionSnapshot,
} from "../api";
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
  onAttachSession,
  onDetach,
  theme,
  onToggleTheme,
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
  onAttachSession: (id: string) => void;
  onDetach: () => void;
  theme: "dark" | "light";
  onToggleTheme: () => void;
}) {
  const models = config?.models ?? [];
  const [model, setModel] = useState("");
  const [customModel, setCustomModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [providerFields, setProviderFields] = useState<Record<string, string>>({});
  const [fhApiKey, setFhApiKey] = useState("");
  const [mpApiKey, setMpApiKey] = useState("");
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [embeddingApiKey, setEmbeddingApiKey] = useState("");
  const [autonomy, setAutonomy] = useState("");
  const [consent, setConsent] = useState(false);
  const [resumable, setResumable] = useState<ResumableSession[]>([]);
  const [resumeChoice, setResumeChoice] = useState("");
  const [nameDraft, setNameDraft] = useState("");
  const [providerInfo, setProviderInfo] = useState(config?.provider ?? null);
  const [credInfo, setCredInfo] = useState(config?.credentials ?? null);

  const effectiveModel = model === "__custom__" ? customModel : model;
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
    objective: "",
  });

  const cred = (field: string) => credInfo?.[field];
  const envCaption = (field: string) => {
    const c = cred(field);
    return c?.is_set && c.env_var ? (
      <span className="caption">✓ available from {c.env_var}</span>
    ) : null;
  };

  return (
    <div className="sidebar">
      <button
        className="theme-toggle"
        title="Toggle theme"
        onClick={onToggleTheme}
      >
        {theme === "dark" ? "☀️" : "🌙"}
      </button>
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
          value={model}
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
      {model === "__custom__" && (
        <label className="field">
          <span>Custom model name</span>
          <input
            type="text"
            value={customModel}
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
              value={embeddingModel}
              disabled={locked}
              onChange={(e) => setEmbeddingModel(e.target.value)}
            >
              <option value="">(default)</option>
              {(config?.embedding_models ?? []).map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          {embeddingModel && (
            <label className="field">
              <span>Embedding API key</span>
              <input
                type="password"
                value={embeddingApiKey}
                disabled={locked}
                onChange={(e) => setEmbeddingApiKey(e.target.value)}
              />
            </label>
          )}
        </>
      )}

      <label className="field">
        <span>Autonomy mode</span>
        <select
          value={autonomy}
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

      {liveSessions.some((s) => s.id !== session?.id) && (
        <div className="sidebar-section">
          <h3>{session ? "Other live sessions" : "Live sessions"}</h3>
          <div className="session-list">
            {liveSessions
              .filter((s) => s.id !== session?.id)
              .map((s) => (
                <button
                  key={s.id}
                  className="session-item"
                  title={`${s.id} — click to attach`}
                  onClick={() => onAttachSession(s.id)}
                >
                  {s.name ?? s.id}
                  <span className="caption">
                    {s.status === "awaiting_input"
                      ? "🟠 awaiting input"
                      : s.status === "running"
                        ? "🟢 running"
                        : "⚪ idle"}{" "}
                    · {s.mode} · {s.n_messages} messages
                  </span>
                </button>
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
