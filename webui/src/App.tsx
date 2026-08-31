import { useCallback, useEffect, useReducer, useState } from "react";
import {
  api,
  type AppConfig,
  type ChatMessage,
  type PresentedQuestion,
  type SessionSnapshot,
} from "./api";
import { useSessionEvents, type SessionEvent } from "./useSessionEvents";
import { Sidebar, type SidebarConfig } from "./components/Sidebar";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ChatPanel } from "./components/ChatPanel";
import { PreChatHero } from "./components/PreChatHero";

interface SessionState {
  snapshot: SessionSnapshot | null;
  status: "idle" | "running" | "awaiting_input";
  messages: ChatMessage[];
  liveLog: string;
  pendingQuestion: PresentedQuestion | null;
  name: string | null;
  lastError: string | null;
}

const emptyState: SessionState = {
  snapshot: null,
  status: "idle",
  messages: [],
  liveLog: "",
  pendingQuestion: null,
  name: null,
  lastError: null,
};

type Action =
  | { type: "session_loaded"; snapshot: SessionSnapshot }
  | { type: "session_closed" }
  | { type: "user_message"; content: string }
  | { type: "clear_error" }
  | { type: "event"; event: SessionEvent };

function reducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case "session_loaded":
      return {
        ...emptyState,
        snapshot: action.snapshot,
        status: action.snapshot.status,
        messages: action.snapshot.chat_messages,
        pendingQuestion: action.snapshot.pending_question,
        liveLog: action.snapshot.live_log ?? "",
        name: action.snapshot.name,
      };
    case "session_closed":
      return emptyState;
    case "user_message":
      return {
        ...state,
        status: "running",
        liveLog: "",
        lastError: null,
        messages: [...state.messages, { role: "user", content: action.content }],
      };
    case "clear_error":
      return { ...state, lastError: null };
    case "event": {
      const ev = action.event;
      switch (ev.type) {
        case "log":
          return { ...state, liveLog: state.liveLog + ev.chunk };
        case "status":
          return {
            ...state,
            status: ev.status,
            ...(ev.status === "idle" ? { liveLog: "" } : {}),
          };
        case "question":
          return { ...state, pendingQuestion: ev.question };
        case "question_cleared":
          return { ...state, pendingQuestion: null };
        case "assistant_message": {
          const last = state.messages[state.messages.length - 1];
          // Reconnect replay can re-deliver the last message; de-dup on content.
          if (last?.role === "assistant" && last.content === ev.message.content)
            return state;
          return { ...state, messages: [...state.messages, ev.message] };
        }
        case "session_named":
          return { ...state, name: ev.name };
        case "error":
          return { ...state, lastError: ev.message };
        default:
          return state;
      }
    }
    default:
      return state;
  }
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [mode, setMode] = useState("meta");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [busy, setBusy] = useState<string | null>(null); // init overlay text
  const [startError, setStartError] = useState<string | null>(null);
  const [serverStopped, setServerStopped] = useState(false);
  const [state, dispatch] = useReducer(reducer, emptyState);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    api.config().then(setConfig).catch((e) => setStartError(String(e)));
    // Reattach on page load: if the server holds a live session (browser
    // refresh, second tab), rejoin it instead of showing the welcome screen.
    api
      .listSessions("meta")
      .then(async (r) => {
        const live = r.live[0];
        if (!live) return;
        const snapshot = await api.getSession(live.id);
        setMode(snapshot.mode);
        dispatch({ type: "session_loaded", snapshot });
      })
      .catch(() => {});
  }, []);

  const onEvent = useCallback(
    (ev: SessionEvent) => dispatch({ type: "event", event: ev }),
    [],
  );
  useSessionEvents(
    state.snapshot?.id ?? null,
    state.snapshot?.event_cursor ?? 0,
    onEvent,
  );

  const startSession = async (cfg: SidebarConfig, resumeDir?: string) => {
    setBusy(resumeDir ? "Restoring session…" : "Initializing agent…");
    setStartError(null);
    try {
      const snapshot = await api.createSession({
        mode,
        model: cfg.model,
        autonomy: cfg.autonomy,
        consent: cfg.consent,
        api_key: cfg.apiKey,
        base_url: cfg.baseUrl,
        provider_fields: cfg.providerFields,
        fh_api_key: cfg.fhApiKey,
        mp_api_key: cfg.mpApiKey,
        embedding_model: cfg.embeddingModel || null,
        embedding_api_key: cfg.embeddingApiKey || null,
        objective: cfg.objective,
        resume_dir: resumeDir ?? null,
      });
      dispatch({ type: "session_loaded", snapshot });
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const sendMessage = async (content: string) => {
    const id = state.snapshot?.id;
    if (!id) return;
    dispatch({ type: "user_message", content });
    try {
      await api.sendMessage(id, content);
    } catch (e) {
      dispatch({
        type: "event",
        event: { type: "error", message: e instanceof Error ? e.message : String(e) },
      });
      dispatch({ type: "event", event: { type: "status", status: "idle" } });
    }
  };

  const resetSession = async () => {
    const id = state.snapshot?.id;
    if (!id) return;
    try {
      await api.resetSession(id);
    } catch {
      /* already gone server-side */
    }
    dispatch({ type: "session_closed" });
  };

  const quitApp = async () => {
    try {
      await api.quit();
    } catch {
      /* connection may drop as the server exits */
    }
    setServerStopped(true);
  };

  const session = state.snapshot;

  if (serverStopped) {
    return (
      <div className="welcome">
        <h2>Server stopped.</h2>
        <p className="tagline">You can close this window.</p>
      </div>
    );
  }

  return (
    <div className="layout">
      <Sidebar
        config={config}
        mode={mode}
        session={session}
        sessionName={state.name}
        status={state.status}
        locked={session !== null}
        onStart={(cfg) => startSession(cfg)}
        onResume={(cfg, dir) => startSession(cfg, dir)}
        onRename={async (name) => {
          if (session) await api.renameSession(session.id, name);
        }}
        onReset={resetSession}
        onQuit={quitApp}
        theme={theme}
        onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
      />
      <div className="main">
        {!session ? (
          <WelcomeScreen
            config={config}
            mode={mode}
            onSelectMode={setMode}
            busy={busy}
            error={startError}
            theme={theme}
          />
        ) : state.messages.length === 0 && state.status === "idle" ? (
          <PreChatHero
            mode={mode}
            sessionId={session.id}
            onStart={sendMessage}
          />
        ) : (
          <ChatPanel
            session={session}
            mode={mode}
            status={state.status}
            messages={state.messages}
            liveLog={state.liveLog}
            pendingQuestion={state.pendingQuestion}
            lastError={state.lastError}
            onSend={sendMessage}
            onStop={() => api.stop(session.id)}
            onFeedback={(requestId, response) =>
              api.sendFeedback(session.id, requestId, response)
            }
            onDismissError={() => dispatch({ type: "clear_error" })}
          />
        )}
      </div>
    </div>
  );
}
