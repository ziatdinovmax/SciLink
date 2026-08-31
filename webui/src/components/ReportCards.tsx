import { useEffect, useState } from "react";
import { api, type ReportRef } from "../api";
import { demoteHeadings, MarkdownBody } from "./MarkdownBody";

export function HtmlReportCard({
  sessionId,
  report,
}: {
  sessionId: string;
  report: ReportRef;
}) {
  const url = api.fileUrl(sessionId, report.path);
  return (
    <details className="card">
      <summary>Report: {report.name}</summary>
      <div className="card-body">
        <iframe src={url} title={report.name} sandbox="allow-scripts" />
        <div className="card-actions">
          <a href={url} download={report.name}>
            <button>Download</button>
          </a>
        </div>
      </div>
    </details>
  );
}

export function MdReportCard({
  sessionId,
  report,
}: {
  sessionId: string;
  report: ReportRef;
}) {
  const [text, setText] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (open && text === null)
      api
        .fetchFileText(sessionId, report.path)
        .then((t) => setText(t.slice(0, 100_000)))
        .catch((e) => setText(`Could not load document: ${e}`));
  }, [open, text, sessionId, report.path]);

  // Resolve relative image refs against the document's directory via the
  // files endpoint (replaces md_images.inline_local_images).
  const docDir = report.path.includes("/")
    ? report.path.slice(0, report.path.lastIndexOf("/") + 1)
    : "";
  const transformImageUri = (src: string) =>
    /^(https?:|data:)/.test(src) ? src : api.fileUrl(sessionId, docDir + src);

  return (
    <details className="card" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        {report.title ?? report.name}: {report.name}
      </summary>
      <div className="card-body">
        <div className="md-report-scroll">
          {text !== null ? (
            <MarkdownBody
              text={demoteHeadings(text)}
              transformImageUri={transformImageUri}
            />
          ) : (
            <p className="caption">Loading…</p>
          )}
        </div>
        <div className="card-actions">
          <a href={api.fileUrl(sessionId, report.path)} download={report.name}>
            <button>Download</button>
          </a>
        </div>
      </div>
    </details>
  );
}
