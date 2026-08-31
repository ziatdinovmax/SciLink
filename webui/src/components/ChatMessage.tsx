import { api, type ChatMessage as Msg } from "../api";
import avatarAgent from "../assets/avatar_agent.svg";
import avatarUser from "../assets/avatar_user.svg";
import { LogView } from "./LogView";
import { MarkdownBody } from "./MarkdownBody";
import { HtmlReportCard, MdReportCard } from "./ReportCards";

export function ChatMessage({
  sessionId,
  message,
}: {
  sessionId: string;
  message: Msg;
}) {
  const isUser = message.role === "user";
  return (
    <div className="chat-message">
      <img
        className="avatar"
        src={isUser ? avatarUser : avatarAgent}
        alt={message.role}
      />
      <div className="bubble">
        <MarkdownBody text={message.content} escapeTilde={!isUser} />
        {(message.images ?? []).map((img) => (
          <img
            key={img}
            className="attachment"
            src={api.fileUrl(sessionId, img)}
            alt={img}
            title={
              img.split("/").pop()?.startsWith("debug_")
                ? `Sample fit: ${img.split("/").pop()}`
                : img
            }
          />
        ))}
        {(message.html_reports ?? []).map((r) => (
          <HtmlReportCard key={r.path} sessionId={sessionId} report={r} />
        ))}
        {(message.md_reports ?? []).map((r) => (
          <MdReportCard key={r.path} sessionId={sessionId} report={r} />
        ))}
        {message.verbose && (
          <details className="card">
            <summary>Verbose output</summary>
            <div className="card-body">
              <LogView text={message.verbose} maxLines={2000} />
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
