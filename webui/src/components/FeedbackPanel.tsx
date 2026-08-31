import { useState } from "react";
import { api, type PresentedQuestion } from "../api";
import { MarkdownBody } from "./MarkdownBody";

/** Renders the parked HITL question — the React twin of the Streamlit
 * feedback branch (app.py:1053-1327). The response contracts are identical:
 * bare digit / "" for candidate selectors, "y"/"no" for fan-out, "keep"/""
 * for keep-revert, free text or "" elsewhere. */
export function FeedbackPanel({
  sessionId,
  question,
  onRespond,
}: {
  sessionId: string;
  question: PresentedQuestion;
  onRespond: (response: string) => void;
}) {
  const [text, setText] = useState("");
  const [choice, setChoice] = useState<number | null>(
    question.judge_pick ?? null,
  );
  const [sent, setSent] = useState(false);

  const respond = (response: string) => {
    if (sent) return;
    setSent(true);
    onRespond(response);
  };

  const previews = question.preview_images.map((p) => {
    const base = p.split("/").pop() ?? p;
    return (
      <figure key={p} style={{ margin: "0 0 8px" }}>
        <img className="preview" src={api.fileUrl(sessionId, p)} alt={base} />
        {question.candidate_captions[base] && (
          <figcaption className="caption">
            {question.candidate_captions[base]}
          </figcaption>
        )}
      </figure>
    );
  });

  const codeFiles = question.code_files.map((f, i) => (
    <details className="card" key={f.name} open={question.code_files.length === 1 && i === 0}>
      <summary>📄 {f.name}</summary>
      <div className="card-body">
        <pre style={{ margin: 0, overflowX: "auto" }}>
          <code>{f.content}</code>
        </pre>
      </div>
    </details>
  ));

  const contextBox = question.context_display ? (
    <div className="context-box">{question.context_display}</div>
  ) : null;

  if (question.widget === "keep_revert") {
    return (
      <div className="feedback-panel">
        {previews}
        {contextBox}
        <div className="feedback-actions">
          <button className="primary" onClick={() => respond("keep")} disabled={sent}>
            {question.labels.keep}
          </button>
          <button className="primary" onClick={() => respond("")} disabled={sent}>
            {question.labels.revert}
          </button>
        </div>
      </div>
    );
  }

  if (question.widget === "fanout_confirm") {
    const f = question.fanout;
    return (
      <div className="feedback-panel">
        <h4>🔀 Launch parallel multi-dataset analysis?</h4>
        {f?.verdict && (
          <MarkdownBody text={`**Complementarity:** ${f.verdict}`} />
        )}
        {f?.join_axis && <MarkdownBody text={`**Join axis:** ${f.join_axis}`} />}
        {f && f.branches.length > 0 && (
          <MarkdownBody
            text={
              "**Branches** — run concurrently, each seeing the others as auxiliary:\n" +
              f.branches.map((b) => `- ${b}`).join("\n")
            }
          />
        )}
        {f?.rationale && <MarkdownBody text={`**Why:** ${f.rationale}`} />}
        <p className="caption">
          Branches run autonomously — no per-branch approval pauses.
        </p>
        <div className="feedback-actions">
          <button onClick={() => respond("no")} disabled={sent}>
            {question.labels.cancel}
          </button>
          <button className="primary" onClick={() => respond("y")} disabled={sent}>
            {question.labels.confirm}
          </button>
        </div>
      </div>
    );
  }

  if (question.widget === "bestofn" || question.widget === "plan_candidates") {
    const cands = question.candidates ?? [];
    const pick = question.judge_pick;
    return (
      <div className="feedback-panel">
        {previews}
        {contextBox}
        <p style={{ marginTop: 0 }}>{question.labels.select}</p>
        <div className="radio-list">
          {cands.map((c) => (
            <label key={c.idx}>
              <input
                type="radio"
                name="candidate"
                checked={choice === c.idx}
                onChange={() => setChoice(c.idx)}
              />
              {c.label}
            </label>
          ))}
        </div>
        <div className="feedback-actions">
          <button
            className="primary"
            disabled={sent || choice === null}
            onClick={() => respond(String(choice))}
          >
            {question.labels.use}
          </button>
          <button
            className="success"
            disabled={sent}
            onClick={() => respond("")}
            title={`Accept the judge's pick (Candidate ${pick})`}
          >
            {question.labels.accept}
          </button>
        </div>
      </div>
    );
  }

  // generic / dataset_description / code_review
  return (
    <div className="feedback-panel">
      {previews}
      {codeFiles}
      {contextBox}
      <label className="field">
        <span>{question.labels.input}</span>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
        />
      </label>
      <div className="feedback-actions">
        <button
          className="primary"
          disabled={sent || !text.trim()}
          onClick={() => respond(text.trim())}
        >
          {question.labels.submit}
        </button>
        <button className="primary" disabled={sent} onClick={() => respond("")}>
          {question.labels.accept}
        </button>
      </div>
    </div>
  );
}
