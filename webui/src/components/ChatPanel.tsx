import { useEffect, useRef, useState } from "react";
import type {
  ChatMessage as Msg,
  PresentedQuestion,
  SessionSnapshot,
} from "../api";
import { ChatMessage } from "./ChatMessage";
import { FeedbackPanel } from "./FeedbackPanel";
import { LogView } from "./LogView";

export function ChatPanel({
  session,
  mode,
  status,
  messages,
  liveLog,
  pendingQuestion,
  lastError,
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
  onSend: (content: string) => void;
  onStop: () => void;
  onFeedback: (requestId: string, response: string) => Promise<unknown>;
  onDismissError: () => void;
}) {
  const [draft, setDraft] = useState("");
  const [showVerbose, setShowVerbose] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const running = status !== "idle";

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
                <span className="agent-spinner-label">Agent is working...</span>
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

      <div className="chat-input-row">
        <textarea
          value={draft}
          placeholder={placeholder}
          disabled={running}
          onChange={(e) => setDraft(e.target.value)}
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
