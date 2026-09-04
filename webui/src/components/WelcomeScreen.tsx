import type { AppConfig } from "../api";
import logoDark from "../assets/scilink_logo_dark_animated.svg";
import logoLight from "../assets/scilink_logo_light_animated.svg";

export function WelcomeScreen({
  config,
  mode,
  onSelectMode,
  busy,
  error,
  theme,
}: {
  config: AppConfig | null;
  mode: string;
  onSelectMode: (mode: string) => void;
  busy: string | null;
  error: string | null;
  theme: "dark" | "light";
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
