import { useEffect, useRef, useState } from "react";
import type { AppConfig, LiveSession } from "../api";
import logoDark from "../assets/scilink_logo_dark_animated.svg";
import logoLight from "../assets/scilink_logo_light_animated.svg";

// Richer client-side identity per mode; unknown keys (a future mode the
// server starts listing, e.g. simulate) fall back to the server's own
// label/description so they appear without a frontend change.
const MODE_META: Record<
  string,
  { emoji: string; name: string; blurb: string }
> = {
  meta: {
    emoji: "🎛️",
    name: "Mission Control",
    blurb:
      "Describe a goal — the meta-agent routes work to the analysis and planning specialists and fuses the results",
  },
  analyze: {
    emoji: "🔬",
    name: "Analyze",
    blurb:
      "Interpret experimental data — images, spectra, hyperspectral cubes — with agentic fitting and reports",
  },
  plan: {
    emoji: "📋",
    name: "Plan",
    blurb:
      "Design experiments and optimization campaigns, grounded in your papers, code, and data",
  },
  simulate: {
    emoji: "⚛️",
    name: "Simulate",
    blurb: "Build structures and run DFT/MD simulations end to end",
  },
};

function ModeSelect({
  modes,
  mode,
  onSelect,
}: {
  modes: AppConfig["modes"];
  mode: string;
  onSelect: (key: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const info = (key: string, fallbackLabel: string, fallbackDesc: string) =>
    MODE_META[key] ?? { emoji: "🧩", name: fallbackLabel, blurb: fallbackDesc };
  const current = modes.find((m) => m.key === mode);
  const cur = info(mode, current?.label ?? mode, current?.description ?? "");

  return (
    <div className="mode-select" ref={rootRef}>
      <button
        className="mode-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="mode-emoji">{cur.emoji}</span>
        <span className="mode-name">
          {cur.name}
          {current?.beta && <span className="beta-pill">BETA</span>}
        </span>
        <span className="mode-caret">❯</span>
      </button>
      {open && (
        <div className="mode-menu" role="listbox">
          {modes.map((m) => {
            const mi = info(m.key, m.label, m.description);
            return (
              <button
                key={m.key}
                role="option"
                aria-selected={m.key === mode}
                className={`mode-option${m.key === mode ? " selected" : ""}`}
                onClick={() => {
                  onSelect(m.key);
                  setOpen(false);
                }}
              >
                <span className="mode-emoji">{mi.emoji}</span>
                <span className="mode-name">
                  {mi.name}
                  {m.beta && <span className="beta-pill">BETA</span>}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

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
  onCloseSession,
}: {
  config: AppConfig | null;
  mode: string;
  onSelectMode: (mode: string) => void;
  busy: string | null;
  error: string | null;
  theme: "dark" | "light";
  liveSessions: LiveSession[];
  onAttachSession: (id: string) => void;
  onCloseSession: (id: string) => void;
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
                  <span className="reattach-actions">
                    <button className="primary" onClick={() => onAttachSession(s.id)}>
                      Reattach
                    </button>
                    <button
                      className="session-close"
                      title="Close this session (stops any run; stays resumable from disk)"
                      onClick={() => onCloseSession(s.id)}
                    >
                      ✕
                    </button>
                  </span>
                </div>
              ))}
            </div>
          )}
          <ModeSelect modes={modes} mode={mode} onSelect={onSelectMode} />
          <p className="mode-desc">
            {(MODE_META[mode] ?? { blurb: current?.description ?? "" }).blurb}
          </p>
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
