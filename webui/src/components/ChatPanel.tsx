import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type ChatMessage as Msg,
  type PresentedQuestion,
  type SessionSnapshot,
} from "../api";
import { ChatMessage } from "./ChatMessage";
import { FeedbackPanel } from "./FeedbackPanel";
import { currentActivity, LogView } from "./LogView";

export function ChatPanel({
  session,
  mode,
  status,
  messages,
  liveLog,
  pendingQuestion,
  lastError,
  attachRequest,
  onAttachConsumed,
  onSend,
  onStop,
  onFeedback,
  onDismissError,
}: {
  session: SessionSnapshot;
  mode: string;
  status: "idle" | "running" | "awaiting_input";
  messages: Msg[];
  liveLog: string;
  pendingQuestion: PresentedQuestion | null;
  lastError: string | null;
  attachRequest: string | null;
  onAttachConsumed: () => void;
  onSend: (content: string) => void;
  onStop: () => void;
  onFeedback: (requestId: string, response: string) => Promise<unknown>;
  onDismissError: () => void;
}) {
  const [draft, setDraft] = useState("");
  const [showVerbose, setShowVerbose] = useState(false);
  const [uploadNote, setUploadNote] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  // Mid-chat uploads: route each file to the mode's save convention (the
  // same categories the pre-chat heroes use), then drop the saved paths
  // into the draft so the user can add context before sending.
  const categoryFor = (name: string): string => {
    const e = name.slice(name.lastIndexOf(".") + 1).toLowerCase();
    if (mode === "meta") return "meta";
    if (mode === "analyze") return e === "json" ? "metadata" : "data";
    // plan mode
    if (["py", "yaml", "yml"].includes(e)) return "code";
    if (["csv", "xlsx", "tsv", "npy", "json", "txt"].includes(e))
      return "planning_data";
    return "knowledge";
  };

  const handleUploads = async (list: FileList | null) => {
    if (!list || list.length === 0) return;
    setUploadNote("Uploading…");
    const groups = new Map<string, File[]>();
    for (const f of Array.from(list)) {
      const cat = categoryFor(f.name);
      groups.set(cat, [...(groups.get(cat) ?? []), f]);
    }
    const paths: string[] = [];
    try {
      for (const [cat, files] of groups) {
        const r = await api.upload(session.id, cat, files);
        paths.push(...r.paths);
      }
    } catch (e) {
      setUploadNote(e instanceof Error ? e.message : String(e));
      return;
    }
    setUploadNote(null);
    const quoted = paths.map((p) => `\`${p}\``).join(", ");
    setDraft((d) => {
      const mention =
        paths.length === 1 && mode === "analyze" && !d
          ? `I uploaded a data file at ${quoted}. Please examine it.`
          : `I uploaded ${paths.length} file(s): ${quoted}.`;
      return d ? `${d.trimEnd()}\n\n${mention} ` : `${mention} `;
    });
    inputRef.current?.focus();
    setTimeout(autosize, 0);
  };

  // Auto-grow the input with the draft (44px single-line up to max-height),
  // so the send button stays vertically centered beside it.
  const autosize = () => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "44px";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  };

  const running = status !== "idle";
  // Short "what is it doing" line for the spinner pill, derived from the
  // streaming narration — between the bare spinner and full verbose output.
  const activity = useMemo(() => currentActivity(liveLog), [liveLog]);

  // "Attach to chat" from the Files tab: drop a backtick-quoted path into
  // the draft (the agents take paths in prompts).
  useEffect(() => {
    if (!attachRequest) return;
    setDraft((d) => (d ? `${d.trimEnd()} \`${attachRequest}\` ` : `\`${attachRequest}\` `));
    onAttachConsumed();
    inputRef.current?.focus();
    setTimeout(autosize, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attachRequest]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, pendingQuestion, status]);

  const placeholder = running
    ? "Agent is working…"
    : mode === "meta"
      ? "Message mission control..."
      : mode === "plan"
        ? "Message the planning agent..."
        : "Message the analysis agent...";

  const send = () => {
    const content = draft.trim();
    if (!content || running) return;
    setDraft("");
    if (inputRef.current) inputRef.current.style.height = "44px";
    onSend(content);
  };

  return (
    <div className="chat-panel">
      <div className="chat-scroll" ref={scrollRef}>
        {messages.map((m, i) => (
          <ChatMessage key={i} sessionId={session.id} message={m} />
        ))}

        {lastError && (
          <div className="error-banner" onClick={onDismissError} role="alert">
            {lastError}
          </div>
        )}

        {status === "awaiting_input" && pendingQuestion && (
          <FeedbackPanel
            key={pendingQuestion.request_id}
            sessionId={session.id}
            question={pendingQuestion}
            onRespond={(response) =>
              void onFeedback(pendingQuestion.request_id, response)
            }
          />
        )}

        {status === "running" && (
          <>
            <div className="live-row">
              <div className="agent-spinner-container">
                <span className="agent-spinner-dot">•</span>
                <span className="agent-spinner-dot">•</span>
                <span className="agent-spinner-dot">•</span>
                <span
                  className="agent-spinner-label"
                  title={activity ?? undefined}
                >
                  {activity ?? "Agent is working..."}
                </span>
              </div>
              <button className="stop-btn danger-hover" title="Stop agent" onClick={onStop}>
                ■
              </button>
            </div>
            {liveLog && (
              <label className="toggle-row">
                <span className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={showVerbose}
                    onChange={(e) => setShowVerbose(e.target.checked)}
                  />
                  <span className="track" />
                </span>
                Show verbose output
              </label>
            )}
            {showVerbose && liveLog && <LogView text={liveLog} />}
          </>
        )}
      </div>

      {uploadNote && (
        <p className="caption" style={{ padding: "0 8%" }}>
          {uploadNote}
        </p>
      )}
      <div className="chat-input-row">
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            void handleUploads(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          className="attach"
          title="Upload files into this session"
          disabled={running}
          onClick={() => fileRef.current?.click()}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round"
            aria-hidden="true">
            <path d="M21.4 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.2-9.19a4 4 0 015.65 5.66l-9.2 9.19a2 2 0 01-2.82-2.83l8.49-8.48" />
          </svg>
        </button>
        <textarea
          ref={inputRef}
          rows={1}
          value={draft}
          placeholder={placeholder}
          disabled={running}
          onChange={(e) => {
            setDraft(e.target.value);
            autosize();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button
          className="send"
          title="Send"
          disabled={running || !draft.trim()}
          onClick={send}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M3.4 20.4 22 12 3.4 3.6 3.4 10l13 2-13 2z" />
          </svg>
        </button>
      </div>
    </div>
  );
}
