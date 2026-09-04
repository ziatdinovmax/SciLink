import type { AppConfig, LiveSession } from "../api";
import logoDark from "../assets/scilink_logo_dark_animated.svg";
import logoLight from "../assets/scilink_logo_light_animated.svg";

const STATUS_LABEL: Record<string, string> = {
  running: "🟢 running",
  awaiting_input: "🟠 awaiting your input",
  idle: "⚪ idle (in memory)",
};

export function WelcomeScreen({
  config,
  mode,
  onSelectMode,
  busy,
  error,
  theme,
  liveSessions,
  onAttachSession,
}: {
  config: AppConfig | null;
  mode: string;
  onSelectMode: (mode: string) => void;
  busy: string | null;
  error: string | null;
  theme: "dark" | "light";
  liveSessions: LiveSession[];
  onAttachSession: (id: string) => void;
}) {
  const modes = config?.modes ?? [];
  const current = modes.find((m) => m.key === mode);

  return (
    <div className="welcome">
      {busy ? (
        <>
          <div className="init-status">
            <span className="agent-spinner-dot">•</span>
            <span className="agent-spinner-dot">•</span>
            <span className="agent-spinner-dot">•</span>
            <span className="init-label">{busy}</span>
          </div>
          <p className="init-sub">
            {busy.startsWith("Restoring")
              ? "Loading checkpoint and chat history"
              : "Setting up models and tools"}
          </p>
        </>
      ) : (
        <>
          {liveSessions.length > 0 && (
            <div className="reattach-banner">
              <p className="caption" style={{ margin: "0 0 8px" }}>
                Sessions still live on this server — reattach to continue:
              </p>
              {liveSessions.map((s) => (
                <div className="reattach-row" key={s.id}>
                  <span className="reattach-info">
                    <strong>{s.name ?? s.id}</strong>{" "}
                    <span className="caption">
                      {STATUS_LABEL[s.status] ?? s.status} · {s.mode} ·{" "}
                      {s.n_messages} messages
                    </span>
                  </span>
                  <button className="primary" onClick={() => onAttachSession(s.id)}>
                    Reattach
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="mode-row">
            {modes.map((m) => (
              <button
                key={m.key}
                className={m.key === mode ? "primary" : ""}
                onClick={() => onSelectMode(m.key)}
              >
                {m.label}
                {m.beta && (
                  <>
                    {" "}
                    <span className="beta-pill">BETA</span>
                  </>
                )}
              </button>
            ))}
          </div>
          {current && <p className="mode-desc">{current.description}</p>}
          <img
            className="logo-big"
            src={theme === "light" ? logoLight : logoDark}
            alt="SciLink"
          />
          <p className="tagline">
            LLM-powered agents for scientific research automation
          </p>
          <p className="hint">
            Configure the model in the sidebar, check the consent box, and
            press Start Session.
          </p>
          {error && <div className="error-banner">{error}</div>}
        </>
      )}
    </div>
  );
}
