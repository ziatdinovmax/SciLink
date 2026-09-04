/** EventSource hook: subscribes to /sessions/{id}/events and dispatches
 * typed events into the session reducer. EventSource reconnects on its own
 * and replays via Last-Event-ID (the server keeps a bounded ring). */

import { useEffect } from "react";
import type { ChatMessage, PresentedQuestion } from "./api";

export type SessionEvent =
  | { type: "log"; chunk: string }
  | { type: "status"; status: "idle" | "running" | "awaiting_input" }
  | { type: "question"; question: PresentedQuestion }
  | { type: "question_cleared"; request_id: string }
  | { type: "assistant_message"; message: ChatMessage }
  | { type: "session_named"; name: string }
  | { type: "files_changed" }
  | {
      type: "analysis_image";
      path: string;
      label: string;
      branch: string | null;
      v?: number;
    }
  | { type: "error"; message: string };

export function useSessionEvents(
  sessionId: string | null,
  afterCursor: number,
  onEvent: (ev: SessionEvent) => void,
) {
  useEffect(() => {
    if (!sessionId) return;
    const src = new EventSource(
      `/api/v1/sessions/${sessionId}/events?after=${afterCursor}`,
    );
    const parse = (e: MessageEvent) => JSON.parse(e.data as string);

    src.addEventListener("log", (e) =>
      onEvent({ type: "log", chunk: parse(e).chunk ?? "" }),
    );
    src.addEventListener("status", (e) =>
      onEvent({ type: "status", status: parse(e).status }),
    );
    src.addEventListener("question", (e) =>
      onEvent({ type: "question", question: parse(e) }),
    );
    src.addEventListener("question_cleared", (e) =>
      onEvent({ type: "question_cleared", request_id: parse(e).request_id }),
    );
    src.addEventListener("assistant_message", (e) =>
      onEvent({ type: "assistant_message", message: parse(e) }),
    );
    src.addEventListener("session_named", (e) =>
      onEvent({ type: "session_named", name: parse(e).name }),
    );
    src.addEventListener("files_changed", () =>
      onEvent({ type: "files_changed" }),
    );
    src.addEventListener("analysis_image", (e) => {
      const d = parse(e);
      onEvent({
        type: "analysis_image",
        path: d.path,
        label: d.label ?? "",
        branch: d.branch ?? null,
        v: d.v,
      });
    });
    src.addEventListener("error", (e) => {
      // Only our payload-carrying errors; EventSource transport errors have
      // no data and are handled by its auto-reconnect.
      const data = (e as MessageEvent).data;
      if (data) onEvent({ type: "error", message: JSON.parse(data).message });
    });

    return () => src.close();
    // onEvent is kept stable by the caller (useCallback / dispatch);
    // afterCursor is fixed for the life of a loaded session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, onEvent]);
}
