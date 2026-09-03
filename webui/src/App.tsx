import { useCallback, useEffect, useReducer, useState } from "react";
import {
  api,
  type AppConfig,
  type ChatMessage,
  type PresentedQuestion,
  type SessionSnapshot,
} from "./api";
import { useSessionEvents, type SessionEvent } from "./useSessionEvents";
import { UIContext } from "./UIContext";
import { Sidebar, type SidebarConfig } from "./components/Sidebar";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { ChatPanel } from "./components/ChatPanel";
import { FilesPanel } from "./components/FilesPanel";
import { PreChatHero } from "./components/PreChatHero";
import { AnalysisInset } from "./components/AnalysisInset";

interface SessionState {
  snapshot: SessionSnapshot | null;
  status: "idle" | "running" | "awaiting_input";
  messages: ChatMessage[];
  liveLog: string;
  pendingQuestion: PresentedQuestion | null;
  name: string | null;
  lastError: string | null;
  filesVersion: number; // bumped on files_changed → the Files tab refetches
  liveImages: LiveImage[]; // figures streamed during the turn (newest last)
}

export interface LiveImage {
  path: string;
  label: string;
  branch: string | null;
}

const LIVE_IMAGE_CAP = 30; // keep a bounded filmstrip of recent figures

const emptyState: SessionState = {
  snapshot: null,
  status: "idle",
  messages: [],
  liveLog: "",
  pendingQuestion: null,
  name: null,
  lastError: null,
  filesVersion: 0,
  liveImages: [],
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
        liveImages: [], // fresh turn → fresh filmstrip
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
            // Turn over: the completion message carries the figures/report,
            // so the live log and the figure inset both stand down.
            ...(ev.status === "idle" ? { liveLog: "", liveImages: [] } : {}),
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
        case "files_changed":
          return { ...state, filesVersion: state.filesVersion + 1 };
        case "analysis_image": {
          // De-dup a repeat of the current latest (reconnect replay / rewrite).
          const prev = state.liveImages[state.liveImages.length - 1];
          if (prev && prev.path === ev.path) return state;
          const next = [
            ...state.liveImages,
            { path: ev.path, label: ev.label, branch: ev.branch },
          ];
          return {
            ...state,
            liveImages: next.slice(-LIVE_IMAGE_CAP),
          };
        }
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
  const [tab, setTab] = useState<"chat" | "files">("chat");
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [attachRequest, setAttachRequest] = useState<string | null>(null);
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
        // Deep link: restore #files[:path] from before the refresh.
        const m = /^#files(?::(.+))?$/.exec(window.location.hash);
        if (m) {
          setTab("files");
          if (m[1]) setSelectedFile(decodeURIComponent(m[1]));
        }
      })
      .catch(() => {});
  }, []);

  // Keep the URL hash in sync so a refresh (or a shared link) lands on the
  // same tab and file.
  useEffect(() => {
    if (!state.snapshot) return;
    const hash =
      tab === "files"
        ? `#files${selectedFile ? ":" + encodeURIComponent(selectedFile) : ""}`
        : "#chat";
    window.history.replaceState(null, "", hash);
  }, [tab, selectedFile, state.snapshot]);

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
    setTab("chat");
    setSelectedFile(null);
    window.history.replaceState(null, "", "#");
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

  const uiActions = {
    openInFiles: (path: string) => {
      setSelectedFile(path);
      setTab("files");
    },
    // The agents take absolute paths in prompts (matching the upload
    // dispatch convention), so attach the full path.
    attachToChat: (path: string) => {
      const base = state.snapshot?.session_dir;
      setAttachRequest(base ? `${base}/${path}` : path);
      setTab("chat");
    },
  };

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
        ) : (
          <UIContext.Provider value={uiActions}>
            <div className="tab-bar">
              <button
                className={tab === "chat" ? "active" : ""}
                onClick={() => setTab("chat")}
              >
                Chat
              </button>
              <button
                className={tab === "files" ? "active" : ""}
                onClick={() => setTab("files")}
              >
                Files
              </button>
            </div>
            {/* Both tab bodies stay MOUNTED and toggle visibility: switching
                tabs must not destroy the hero's upload/objective state, a
                chat draft, or the explorer's tree state. */}
            <div className="tab-body" hidden={tab !== "files"}>
              <FilesPanel
                sessionId={session.id}
                filesVersion={state.filesVersion}
                active={tab === "files"}
                selectedPath={selectedFile}
                onSelect={setSelectedFile}
              />
            </div>
            <div className="tab-body" hidden={tab !== "chat"}>
              {state.messages.length === 0 && state.status === "idle" ? (
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
                  attachRequest={attachRequest}
                  onAttachConsumed={() => setAttachRequest(null)}
                  onSend={sendMessage}
                  onStop={() => api.stop(session.id)}
                  onFeedback={(requestId, response) =>
                    api.sendFeedback(session.id, requestId, response)
                  }
                  onDismissError={() => dispatch({ type: "clear_error" })}
                />
              )}
            </div>
            <AnalysisInset
              sessionId={session.id}
              images={state.liveImages}
              running={state.status === "running"}
            />
          </UIContext.Provider>
        )}
      </div>
    </div>
  );
}
